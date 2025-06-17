from dataclasses import dataclass, asdict
from .common import (
    ALIFPartialConfig,
    LIFPartialConfig,
    ParameterInitializerV1,
    DenseNormalizationConfig,
    ParameterBoundingV1,
    RateEncoderConfigV1,
)
from .utils import TensorList
from firebrand.learn import FirstSpikeClassifier
from inferno._internal import argtest
import inferno
import inferno.functional as inff
import inferno.neural as snn
import inferno.learn as learn
import math
import torch
import torch.nn as nn
from typing import Any, Callable


@dataclass
class Progress:
    epoch: int
    train: int
    valid: int
    test: int
    sincelog: int
    sincechkpt: int
    runid: Any
    chkpt: Any


def default_progress() -> Progress:
    return Progress(
        epoch=0,
        train=0,
        valid=0,
        test=0,
        sincelog=0,
        sincechkpt=0,
        runid=None,
        chkpt=None,
    )


class TrainerGroup(inferno.Module):
    def __init__(
        self,
        weight_trainer: learn.IndependentCellTrainer,
        delay_trainer: learn.IndependentCellTrainer,
    ):
        # call superclass constructor
        inferno.Module.__init__(self)

        self.weight = weight_trainer
        self.delay = delay_trainer

    def clear(self, **kwargs) -> None:
        self.weight.clear(**kwargs)
        self.delay.clear(**kwargs)


class Log(nn.Module):
    def __init__(self):
        # call superclass constructor
        nn.Module.__init__(self)

        self.excspike = nn.ModuleDict(
            [
                ("avg", TensorList()),
                ("var", TensorList()),
            ]
        )

        self.inhspike = nn.ModuleDict(
            [
                ("avg", TensorList()),
                ("var", TensorList()),
            ]
        )

        self.stats = nn.ModuleDict(
            [
                (
                    "weight",
                    nn.ModuleDict([("avg", TensorList()), ("var", TensorList())]),
                ),
                (
                    "delay",
                    nn.ModuleDict([("avg", TensorList()), ("var", TensorList())]),
                ),
            ]
        )

        self.history = nn.ModuleDict(
            [
                ("label", TensorList()),
                ("rate", TensorList()),
                ("ttfs", TensorList()),
            ]
        )

    def forward(
        self,
        device: torch.device | str | None,
        *,
        excavg: torch.Tensor | None = None,
        excvar: torch.Tensor | None = None,
        inhavg: torch.Tensor | None = None,
        inhvar: torch.Tensor | None = None,
        swavg: torch.Tensor | None = None,
        swvar: torch.Tensor | None = None,
        sdavg: torch.Tensor | None = None,
        sdvar: torch.Tensor | None = None,
        label: torch.Tensor | None = None,
        rate: torch.Tensor | None = None,
        ttfs: torch.Tensor | None = None,
    ) -> None:
        if torch.device is None:
            if excavg is not None:
                self.excspike.avg.append(excavg)
            if excvar is not None:
                self.excspike.var.append(excvar)
            if inhavg is not None:
                self.inhspike.avg.append(inhavg)
            if inhvar is not None:
                self.inhspike.var.append(inhvar)
            if swavg is not None:
                self.stats.weight.avg.append(swavg)
            if swvar is not None:
                self.stats.weight.var.append(swvar)
            if sdavg is not None:
                self.stats.delay.avg.append(sdavg)
            if sdvar is not None:
                self.stats.delay.var.append(sdvar)
            if label is not None:
                self.history.label.append(label)
            if rate is not None:
                self.history.rate.append(rate)
            if ttfs is not None:
                self.history.ttfs.append(ttfs)
        else:
            if excavg is not None:
                self.excspike.avg.append(excavg.to(device=device))
            if excvar is not None:
                self.excspike.var.append(excvar.to(device=device))
            if inhavg is not None:
                self.inhspike.avg.append(inhavg.to(device=device))
            if inhvar is not None:
                self.inhspike.var.append(inhvar.to(device=device))
            if swavg is not None:
                self.stats.weight.avg.append(swavg.to(device=device))
            if swvar is not None:
                self.stats.weight.var.append(swvar.to(device=device))
            if sdavg is not None:
                self.stats.delay.avg.append(sdavg.to(device=device))
            if sdvar is not None:
                self.stats.delay.var.append(sdvar.to(device=device))
            if label is not None:
                self.history.label.append(label.to(device=device))
            if rate is not None:
                self.history.rate.append(rate.to(device=device))
            if ttfs is not None:
                self.history.ttfs.append(ttfs.to(device=device))


class Layer(snn.Layer):
    def __init__(
        self,
        feedfwd,
        inh2exc,
        exc2inh,
        excdyn,
        inhdyn,
    ):
        # call superclass constructor
        snn.Layer.__init__(self)

        # register layers and cells
        self.add_connection("enc2exc", feedfwd)
        self.add_connection("exc2inh", exc2inh)
        self.add_connection("inh2exc", inh2exc)
        self.add_neuron("exc", excdyn)
        self.add_neuron("inh", inhdyn)

    def wiring(
        self, inputs: dict[str, torch.Tensor], **kwargs
    ) -> dict[str, torch.Tensor]:
        return {
            "exc": inputs["enc2exc"] + inputs["inh2exc"],
            "inh": inputs["exc2inh"],
        }


class DCSNN(inferno.Module):
    """SNN model based off of Diehl and Cook 2015."""

    def __init__(
        self,
        step_time: float,
        batch_size: float,
        output_shape: int | tuple[int],
        input_shape: int | tuple[int],
        delay_max: float,
        feedfwd_w_init: Callable[[torch.Tensor], torch.Tensor],
        feedfwd_d_init: Callable[[torch.Tensor], torch.Tensor],
        exc2inh_w_init: Callable[[torch.Tensor], torch.Tensor],
        inh2exc_w_init: Callable[[torch.Tensor], torch.Tensor],
        exc_config: ALIFPartialConfig,
        inh_config: LIFPartialConfig,
        inplace: bool,
    ):
        ################
        # ## Module ## #
        ################

        # superclass constructor
        inferno.Module.__init__(self)

        # submodule collections
        self.neurons = nn.ModuleDict()
        self.connections = nn.ModuleDict()

        #################
        # ## Neurons ## #
        #################

        # excitatory neurons
        self.neurons.exc = snn.ALIF(
            output_shape,
            step_time,
            **asdict(exc_config),
            batch_size=batch_size,
        )

        # inhibitory neurons
        self.neurons.inh = snn.LIF(
            output_shape,
            step_time,
            **asdict(inh_config),
            batch_size=batch_size,
        )

        #####################
        # ## Connections ## #
        #####################

        # synapse partial constructor
        synapses_exc_ff = snn.DeltaCurrent.partialconstructor(
            spike_charge=exc_config.tc_membrane,
            interp_mode="previous",
            interp_tol=0.0,
            current_overbound=None,
            spike_overbound=None,
            inplace=inplace,
        )
        synapses_exc_lat = snn.DeltaCurrent.partialconstructor(
            spike_charge=exc_config.tc_membrane,
            interp_mode="previous",
            interp_tol=0.0,
            current_overbound=None,
            spike_overbound=None,
        )
        synapses_inh = snn.DeltaCurrent.partialconstructor(
            spike_charge=inh_config.time_constant,
            interp_mode="previous",
            interp_tol=0.0,
            current_overbound=None,
            spike_overbound=None,
        )

        # feedforward connection
        self.connections.input = snn.LinearDense(
            input_shape,
            output_shape,
            step_time,
            synapse=synapses_exc_ff,
            bias=False,
            delay=delay_max,
            batch_size=batch_size,
            weight_init=feedfwd_w_init,
            delay_init=feedfwd_d_init,
        )

        # lateral inhibition, exc -> inh
        self.connections.exc2inh = snn.LinearDirect(
            output_shape,
            step_time,
            synapse=synapses_inh,
            bias=False,
            delay=None,
            batch_size=batch_size,
            weight_init=exc2inh_w_init,
        )

        # lateral inhibition, inh -> exc
        self.connections.inh2exc = snn.LinearLateral(
            output_shape,
            step_time,
            synapse=synapses_exc_lat,
            bias=False,
            delay=None,
            batch_size=batch_size,
            weight_init=inh2exc_w_init,
        )

        ################
        # ## Layers ## #
        ################

        # create layers
        self.dclayer = Layer(
            self.connections.input,
            self.connections.inh2exc,
            self.connections.exc2inh,
            self.neurons.exc,
            self.neurons.inh,
        )

    @property
    def dt(self) -> float:
        return self.connections.input.dt

    @dt.setter
    def dt(self, value: float) -> None:
        self.connections.input.dt = value
        self.connections.exc2inh.dt = value
        self.connections.inh2exc.dt = value
        self.neurons.exc.dt = value
        self.neurons.inh.dt = value

    @property
    def inshape(self) -> int:
        return self.connections.input.inshape

    @property
    def outshape(self) -> int:
        return self.connections.input.outshape

    @property
    def batchsz(self) -> int:
        return self.connections.input.batchsz

    @batchsz.setter
    def batchsz(self, value: int) -> None:
        self.connections.input.batchsz = value
        self.connections.exc2inh.batchsz = value
        self.connections.inh2exc.batchsz = value
        self.neurons.exc.batchsz = value
        self.neurons.inh.batchsz = value

    def clear(self, keep_adaptations=True):
        self.neurons.exc.clear(keep_adaptations=keep_adaptations)
        self.neurons.inh.clear()
        self.connections.input.clear()
        self.connections.exc2inh.clear()
        self.connections.inh2exc.clear()

    def forward(
        self,
        inputs: torch.Tensor,
        trainers: list[learn.IndependentCellTrainer] | None = None,
        log: Log | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        r"""Runs through encoded sample, add detailed info to namespace if provided."""
        # allocate tensors for parameter records
        if log and trainers:
            swavg = torch.zeros(
                1,
                device=self.connections.input.weight.device,
                dtype=self.connections.input.weight.dtype,
            )
            swvar = torch.zeros(
                1,
                device=self.connections.input.weight.device,
                dtype=self.connections.input.weight.dtype,
            )
            sdavg = torch.zeros(
                1,
                device=self.connections.input.delay.device,
                dtype=self.connections.input.delay.dtype,
            )
            sdvar = torch.zeros(
                1,
                device=self.connections.input.delay.device,
                dtype=self.connections.input.delay.dtype,
            )

        # storage for spike tensors
        excspikes = []
        inhspikes = []

        # T x B x M..., interate over T
        for step in inputs:
            # run sample through combined layer
            res = self.dclayer(
                {
                    "enc2exc": (step,),
                    "exc2inh": (self.neurons.exc.spike,),
                    "inh2exc": (self.neurons.inh.spike,),
                }
            )

            # split results into expected tensors
            excspikes.append(res["exc"])
            inhspikes.append(res["inh"])

            # update if an updater is given
            if trainers:
                for trainer in trainers:
                    trainer()
                    for cell, _ in trainer.cells:
                        cell.connection.update()
                # add stats to batch total
                if log:
                    swvar_, swavg_ = torch.var_mean(self.connections.input.weight)
                    swavg += swavg_
                    swvar += swvar_
                    sdvar_, sdavg_ = torch.var_mean(self.connections.input.delay)
                    sdavg += sdavg_
                    sdvar += sdvar_

        # stack spikes into (T x B x N0 x ...) tensor
        excspikes = torch.stack(excspikes, dim=0)
        inhspikes = torch.stack(inhspikes, dim=0)

        # update log
        if log and trainers:
            log(
                None,
                swavg=swavg / inputs.shape[0],
                swvar=swvar / inputs.shape[0],
                sdavg=sdavg / inputs.shape[0],
                sdvar=sdvar / inputs.shape[0],
            )

        # T x B x N ..., stack
        return excspikes, inhspikes


class UDCSNN(inferno.Module):
    """SNN model based off of Diehl and Cook 2015."""

    def __init__(
        self,
        step_time: float,
        batch_size: float,
        output_shape: int | tuple[int],
        input_shape: int | tuple[int],
        feedfwd_w_init: Callable[[torch.Tensor], torch.Tensor],
        exc2inh_w_init: Callable[[torch.Tensor], torch.Tensor],
        inh2exc_w_init: Callable[[torch.Tensor], torch.Tensor],
        exc_config: ALIFPartialConfig,
        inh_config: LIFPartialConfig,
        inplace: bool,
    ):
        ################
        # ## Module ## #
        ################

        # superclass constructor
        inferno.Module.__init__(self)

        # submodule collections
        self.neurons = nn.ModuleDict()
        self.connections = nn.ModuleDict()

        #################
        # ## Neurons ## #
        #################

        # excitatory neurons
        self.neurons.exc = snn.ALIF(
            output_shape,
            step_time,
            **asdict(exc_config),
            batch_size=batch_size,
        )

        # inhibitory neurons
        self.neurons.inh = snn.LIF(
            output_shape,
            step_time,
            **asdict(inh_config),
            batch_size=batch_size,
        )

        #####################
        # ## Connections ## #
        #####################

        # synapse partial constructor
        synapses_exc_ff = snn.DeltaCurrent.partialconstructor(
            spike_charge=exc_config.tc_membrane,
            interp_mode="previous",
            interp_tol=0.0,
            current_overbound=None,
            spike_overbound=None,
            inplace=inplace,
        )
        synapses_exc_lat = snn.DeltaCurrent.partialconstructor(
            spike_charge=exc_config.tc_membrane,
            interp_mode="previous",
            interp_tol=0.0,
            current_overbound=None,
            spike_overbound=None,
        )
        synapses_inh = snn.DeltaCurrent.partialconstructor(
            spike_charge=inh_config.time_constant,
            interp_mode="previous",
            interp_tol=0.0,
            current_overbound=None,
            spike_overbound=None,
        )

        # feedforward connection
        self.connections.input = snn.LinearDense(
            input_shape,
            output_shape,
            step_time,
            synapse=synapses_exc_ff,
            bias=False,
            delay=None,
            batch_size=batch_size,
            weight_init=feedfwd_w_init,
        )

        # lateral inhibition, exc -> inh
        self.connections.exc2inh = snn.LinearDirect(
            output_shape,
            step_time,
            synapse=synapses_inh,
            bias=False,
            delay=None,
            batch_size=batch_size,
            weight_init=exc2inh_w_init,
        )

        # lateral inhibition, inh -> exc
        self.connections.inh2exc = snn.LinearLateral(
            output_shape,
            step_time,
            synapse=synapses_exc_lat,
            bias=False,
            delay=None,
            batch_size=batch_size,
            weight_init=inh2exc_w_init,
        )

        ################
        # ## Layers ## #
        ################

        # create layers
        self.dclayer = Layer(
            self.connections.input,
            self.connections.inh2exc,
            self.connections.exc2inh,
            self.neurons.exc,
            self.neurons.inh,
        )

    @property
    def dt(self) -> float:
        return self.connections.input.dt

    @dt.setter
    def dt(self, value: float) -> None:
        self.connections.input.dt = value
        self.connections.exc2inh.dt = value
        self.connections.inh2exc.dt = value
        self.neurons.exc.dt = value
        self.neurons.inh.dt = value

    @property
    def inshape(self) -> int:
        return self.connections.input.inshape

    @property
    def outshape(self) -> int:
        return self.connections.input.outshape

    @property
    def batchsz(self) -> int:
        return self.connections.input.batchsz

    @batchsz.setter
    def batchsz(self, value: int) -> None:
        self.connections.input.batchsz = value
        self.connections.exc2inh.batchsz = value
        self.connections.inh2exc.batchsz = value
        self.neurons.exc.batchsz = value
        self.neurons.inh.batchsz = value

    def clear(self, keep_adaptations=True):
        self.neurons.exc.clear(keep_adaptations=keep_adaptations)
        self.neurons.inh.clear()
        self.connections.input.clear()
        self.connections.exc2inh.clear()
        self.connections.inh2exc.clear()

    def forward(
        self,
        inputs: torch.Tensor,
        trainers: list[learn.IndependentCellTrainer] | None = None,
        log: Log | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        r"""Runs through encoded sample, add detailed info to namespace if provided."""
        # allocate tensors for parameter records
        if log and trainers:
            swavg = torch.zeros(
                1,
                device=self.connections.input.weight.device,
                dtype=self.connections.input.weight.dtype,
            )
            swvar = torch.zeros(
                1,
                device=self.connections.input.weight.device,
                dtype=self.connections.input.weight.dtype,
            )
            sdavg = torch.zeros(
                1,
                device=self.connections.input.weight.device,
                dtype=self.connections.input.weight.dtype,
            )
            sdvar = torch.zeros(
                1,
                device=self.connections.input.weight.device,
                dtype=self.connections.input.weight.dtype,
            )

        # storage for spike tensors
        excspikes = []
        inhspikes = []

        # T x B x M..., interate over T
        for step in inputs:
            # run sample through combined layer
            res = self.dclayer(
                {
                    "enc2exc": (step,),
                    "exc2inh": (self.neurons.exc.spike,),
                    "inh2exc": (self.neurons.inh.spike,),
                }
            )

            # split results into expected tensors
            excspikes.append(res["exc"])
            inhspikes.append(res["inh"])

            # update if an updater is given
            if trainers:
                for trainer in trainers:
                    trainer()
                    for cell, _ in trainer.cells:
                        cell.connection.update()
                # add stats to batch total
                if log:
                    swvar_, swavg_ = torch.var_mean(self.connections.input.weight)
                    swavg += swavg_
                    swvar += swvar_

        # stack spikes into (T x B x N0 x ...) tensor
        excspikes = torch.stack(excspikes, dim=0)
        inhspikes = torch.stack(inhspikes, dim=0)

        # update log
        if log and trainers:
            log(
                None,
                swavg=swavg / inputs.shape[0],
                swvar=swvar / inputs.shape[0],
                sdavg=sdavg / inputs.shape[0],
                sdvar=sdvar / inputs.shape[0],
            )

        # T x B x N ..., stack
        return excspikes, inhspikes


def build_encoder(
    config: RateEncoderConfigV1,
) -> snn.PoissonIntervalEncoder | snn.HomogeneousPoissonEncoder:
    match config.encoder.lower():
        case "poisson-int":
            return snn.PoissonIntervalEncoder(
                steps=config.n_steps,
                step_time=config.step_time,
                frequency=config.max_freq,
                generator=torch.Generator(),
            )
        case "exp-int":
            return snn.HomogeneousPoissonEncoder(
                steps=config.n_steps,
                step_time=config.step_time,
                frequency=config.max_freq,
                refrac=0.0,
                compensate=False,
                generator=torch.Generator(),
            )
        case "exprefrac-int":
            return snn.HomogeneousPoissonEncoder(
                steps=config.n_steps,
                step_time=config.step_time,
                frequency=config.max_freq,
                refrac=config.step_time,
                compensate=True,
                generator=torch.Generator(),
            )
        case _:
            raise RuntimeError(
                f"internal error, incorrect encoder setting '{config.encoder}' passed"
            )


def encoder_requires_mps_fallback(config: RateEncoderConfigV1) -> bool:
    match config.encoder.lower():
        case "poisson-int":
            return True
        case "exp-int":
            return False
        case "exprefrac-int":
            return False
        case _:
            raise RuntimeError(
                f"internal error, incorrect encoder setting '{config.encoder}' passed"
            )


def build_mnist_network(
    step_time: float,
    sqrt_output_sz: int,
    weight_init: ParameterInitializerV1,
    delay_init: ParameterInitializerV1,
    delay_max: float | None,
    exc2inh_weight: float,
    inh2exc_weight: float,
    exc_config: ALIFPartialConfig,
    inh_config: LIFPartialConfig,
    inplace: bool,
) -> DCSNN:
    # check arguments
    weight_init.pmin = argtest.lt(
        "weight_init.pmin",
        weight_init.pmin,
        weight_init.pmax,
        float,
        "weight_init.pmax",
    )
    weight_init.pmax = float(weight_init.pmax)
    exc2inh_weight = argtest.gte("exc2inh_weight", exc2inh_weight, 0, float)
    inh2exc_weight = argtest.lte("inh2exc_weight", inh2exc_weight, 0, float)

    delay_init.pmin = argtest.gte("delay_init.pmin", delay_init.pmin, 0.0, float)
    delay_init.pmax = argtest.gte(
        "delay_init.pmax", delay_init.pmax, delay_init.pmin, float, "delay_init.pmin"
    )
    if delay_max is None:
        delay_max = delay_init.pmax
    else:
        delay_max = argtest.lte(
            "delay_max", delay_max, delay_init.pmax, float, "delay_init.pmax"
        )

    # create and seed generator
    wgen = torch.Generator().manual_seed(weight_init.seed)
    dgen = torch.Generator().manual_seed(delay_init.seed)

    # create weight and delay initializers
    match weight_init.dist.lower():
        case "uniform":
            feedfwd_w_init = lambda x: inferno.rescale(  # noqa:E731
                inferno.uniform(x, generator=wgen), weight_init.pmin, weight_init.pmax
            )
        case "normal":
            feedfwd_w_init = lambda x: inferno.rescale(  # noqa:E731;
                inferno.normal(x, generator=wgen), weight_init.pmin, weight_init.pmax
            )
        case _:
            feedfwd_w_init = None
            raise RuntimeError(
                "'weight_init.dist' must be one of: 'uniform', 'normal'; "
                f"received '{weight_init.dist}'"
            )
    match delay_init.dist.lower():
        case "uniform":
            feedfwd_d_init = lambda x: inferno.rescale(  # noqa:E731
                inferno.uniform(x, generator=dgen), delay_init.pmin, delay_init.pmax
            )
        case "normal":
            feedfwd_d_init = lambda x: inferno.rescale(  # noqa:E731;
                inferno.normal(x, generator=dgen), delay_init.pmin, delay_init.pmax
            )
        case _:
            feedfwd_d_init = None
            raise RuntimeError(
                "'delay_init.dist' must be one of: 'uniform', 'normal'; "
                f"received '{delay_init.dist}'"
            )
    exc2inh_w_init = lambda x: inferno.full(x, exc2inh_weight)  # noqa:E731;
    inh2exc_w_init = lambda x: inferno.full(x, inh2exc_weight)  # noqa:E731;

    network = DCSNN(
        step_time=step_time,
        batch_size=1,
        output_shape=(sqrt_output_sz, sqrt_output_sz),
        input_shape=(28, 28),
        delay_max=delay_max,
        feedfwd_w_init=feedfwd_w_init,
        feedfwd_d_init=feedfwd_d_init,
        exc2inh_w_init=exc2inh_w_init,
        inh2exc_w_init=inh2exc_w_init,
        exc_config=exc_config,
        inh_config=inh_config,
        inplace=inplace,
    )

    # add updaters
    if not network.connections.input.updatable:
        network.connections.input.updater = network.connections.input.defaultupdater()

    return network


def build_undelayed_mnist_network(
    step_time: float,
    sqrt_output_sz: int,
    weight_init: ParameterInitializerV1,
    exc2inh_weight: float,
    inh2exc_weight: float,
    exc_config: ALIFPartialConfig,
    inh_config: LIFPartialConfig,
    inplace: bool,
) -> UDCSNN:
    # check arguments
    weight_init.pmin = argtest.lt(
        "weight_init.pmin",
        weight_init.pmin,
        weight_init.pmax,
        float,
        "weight_init.pmax",
    )
    weight_init.pmax = float(weight_init.pmax)
    exc2inh_weight = argtest.gte("exc2inh_weight", exc2inh_weight, 0, float)
    inh2exc_weight = argtest.lte("inh2exc_weight", inh2exc_weight, 0, float)

    # create and seed generator
    wgen = torch.Generator().manual_seed(weight_init.seed)

    # create weight and delay initializers
    match weight_init.dist.lower():
        case "uniform":
            feedfwd_w_init = lambda x: inferno.rescale(  # noqa:E731
                inferno.uniform(x, generator=wgen), weight_init.pmin, weight_init.pmax
            )
        case "normal":
            feedfwd_w_init = lambda x: inferno.rescale(  # noqa:E731;
                inferno.normal(x, generator=wgen), weight_init.pmin, weight_init.pmax
            )
        case _:
            feedfwd_w_init = None
            raise RuntimeError(
                "'weight_init.dist' must be one of: 'uniform', 'normal'; "
                f"received '{weight_init.dist}'"
            )

    exc2inh_w_init = lambda x: inferno.full(x, exc2inh_weight)  # noqa:E731;
    inh2exc_w_init = lambda x: inferno.full(x, inh2exc_weight)  # noqa:E731;

    network = UDCSNN(
        step_time=step_time,
        batch_size=1,
        output_shape=(sqrt_output_sz, sqrt_output_sz),
        input_shape=(28, 28),
        feedfwd_w_init=feedfwd_w_init,
        exc2inh_w_init=exc2inh_w_init,
        inh2exc_w_init=inh2exc_w_init,
        exc_config=exc_config,
        inh_config=inh_config,
        inplace=inplace,
    )

    # add updaters
    if not network.connections.input.updatable:
        network.connections.input.updater = network.connections.input.defaultupdater()

    return network


def add_weight_normalization(
    network: DCSNN,
    weight_norm: DenseNormalizationConfig,
) -> snn.Normalization:

    if weight_norm is not None:
        # common weight normalization parameters
        w_norm_params = {
            "order": weight_norm.order,
            "scale": weight_norm.scale,
            "train_update": True,
            "eval_update": False,
            "as_prehook": True,
        }
        match weight_norm.vector.lower():
            case "input":
                w_norm_params["dim"] = -1
                if weight_norm.autoscale:
                    w_norm_params["scale"] = w_norm_params["scale"] * math.prod(
                        network.inshape
                    )
            case "output":
                w_norm_params["dim"] = 0
                w_norm_params["scale"] = w_norm_params["scale"] * math.prod(
                    network.outshape
                )
            case _:
                raise RuntimeError(
                    "'weight_norm.vector' must be one of: 'input', 'output'; "
                    f"received '{weight_norm.vector}'"
                )

        # schedule weight normalization
        match weight_norm.schedule.lower():
            case "step":
                normhook = snn.Normalization(
                    network.dclayer, "connections.enc2exc.weight", **w_norm_params
                )
                normhook.deregister()

            case "batch":
                normhook = snn.Normalization(
                    network, "dclayer.connections.enc2exc.weight", **w_norm_params
                )
                normhook.deregister()

            case _:
                raise RuntimeError(
                    "'weight_norm.schedule' must be one of: 'step', 'batch'; "
                    f"received '{weight_norm.schedule}'"
                )

    return normhook


def add_delay_normalization(
    network: DCSNN,
    delay_norm: DenseNormalizationConfig | None,
) -> snn.Normalization:

    # common delay normalization parameters
    d_norm_params = {
        "order": delay_norm.order,
        "scale": delay_norm.scale,
        "train_update": True,
        "eval_update": False,
        "as_prehook": True,
    }
    match delay_norm.vector.lower():
        case "input":
            d_norm_params["dim"] = -1
            if delay_norm.autoscale:
                d_norm_params["scale"] = d_norm_params["scale"] * math.prod(
                    network.inshape
                )
        case "output":
            d_norm_params["dim"] = 0
            if delay_norm.autoscale:
                d_norm_params["scale"] = d_norm_params["scale"] * math.prod(
                    network.outshape
                )
        case _:
            raise RuntimeError(
                "'delay_norm.vector' must be one of: 'input', 'output'; "
                f"received '{delay_norm.vector}'"
            )

    # schedule delay normalization
    match delay_norm.schedule.lower():
        case "step":
            normhook = snn.Normalization(
                network.dclayer, "connections.enc2exc.delay", **d_norm_params
            )
            normhook.deregister()

        case "batch":
            normhook = snn.Normalization(
                network, "dclayer.connections.enc2exc.delay", **d_norm_params
            )
            normhook.deregister()

        case _:
            raise RuntimeError(
                "'delay_norm.schedule' must be one of: 'step', 'batch'; "
                f"received '{delay_norm.schedule}'"
            )

    return normhook


def add_weight_bounding(
    network: DCSNN,
    config: ParameterBoundingV1,
) -> snn.Clamping | None:
    # check arguments
    if config.lb_lim is not None and config.ub_lim is not None:
        config.lb_lim = argtest.lt(
            "config.lb_lim",
            config.lb_lim,
            config.ub_lim,
            float,
            "config.ub_lim",
        )
        config.ub_lim = float(config.ub_lim)

    _ = argtest.oneof(
        "config.lb_mode",
        config.lb_mode,
        "soft",
        "soft-scaled",
        "hard",
        "clamp",
        "none",
        op=(lambda x: x.lower()),
    )
    _ = argtest.oneof(
        "config.ub_mode",
        config.ub_mode,
        "soft",
        "soft-scaled",
        "hard",
        "clamp",
        "none",
        op=(lambda x: x.lower()),
    )

    config.lb_mode, config.ub_mode = (
        config.lb_mode.lower(),
        config.ub_mode.lower(),
    )

    if config.lb_mode != "none" and config.lb_lim is None:
        raise RuntimeError(
            f"'config.lb_lim' cannot be None with 'config.lb_mode' of '{config.lb_mode}'"
        )
    if config.ub_mode != "none" and config.ub_lim is None:
        raise RuntimeError(
            f"'config.ub_lim' cannot be None with 'config.ub_mode' of '{config.ub_mode}'"
        )

    if (config.lb_mode == "soft-scaled" or config.ub_mode == "soft-scaled") and (
        config.lb_lim is None or config.ub_lim is None
    ):
        raise RuntimeError(
            "both 'config.lb_lim' and 'config.ub_limit' cannot be None if either "
            "'config.lb_mode' or 'config.ub_mode' is 'soft-scaled'"
        )

    # add weight dependence bounding
    match config.lb_mode.lower:
        case "soft":
            if config.lb_pow is None or config.lb_pow == 1:
                network.connections.input.updater.weight.lowerbound(
                    inff.bound_lower_multiplicative, config.lb_lim
                )
            else:
                network.connections.input.updater.weight.lowerbound(
                    inff.bound_lower_power, config.lb_lim, power=config.lb_pow
                )
        case "soft-scaled":
            if config.lb_pow is None or config.lb_pow == 1:
                network.connections.input.updater.weight.lowerbound(
                    inff.bound_lower_scaled_multiplicative,
                    config.lb_lim,
                    range=(config.ub_lim - config.lb_lim),
                )
            else:
                network.connections.input.updater.weight.lowerbound(
                    inff.bound_lower_scaled_power,
                    config.lb_lim,
                    power=config.lb_pow,
                    range=(config.ub_lim - config.lb_lim),
                )
        case "hard":
            network.connections.input.updater.weight.lowerbound(
                inff.bound_lower_sharp, config.lb_lim
            )

    match config.ub_mode:
        case "soft":
            if config.ub_pow is None or config.ub_pow == 1:
                network.connections.input.updater.weight.upperbound(
                    inff.bound_upper_multiplicative, config.ub_lim
                )
            else:
                network.connections.input.updater.weight.upperbound(
                    inff.bound_upper_power, config.ub_lim, power=config.ub_pow
                )
        case "soft-scaled":
            if config.ub_pow is None or config.ub_pow == 1:
                network.connections.input.updater.weight.upperbound(
                    inff.bound_upper_scaled_multiplicative,
                    config.ub_lim,
                    range=(config.ub_lim - config.lb_lim),
                )
            else:
                network.connections.input.updater.weight.upperbound(
                    inff.bound_upper_scaled_power,
                    config.ub_lim,
                    power=config.ub_pow,
                    range=(config.ub_lim - config.lb_lim),
                )
        case "hard":
            network.connections.input.updater.weight.upperbound(
                inff.bound_upper_sharp, config.ub_lim
            )

    # add update-triggered clamping
    if config.lb_mode == "clamp" or config.ub_mode == "clamp":
        clamphook = snn.Clamping(
            network.connections.input.updater,
            "parent.weight",
            min=(config.lb_lim if config.lb_mode == "clamp" else None),
            max=(config.ub_lim if config.ub_mode == "clamp" else None),
        )
    else:
        clamphook = None

    return clamphook


def add_delay_bounding(
    network: DCSNN,
    config: ParameterBoundingV1,
) -> snn.Clamping | None:
    # check arguments
    if config.lb_lim is not None and config.ub_lim is not None:
        config.lb_lim = argtest.lt(
            "config.lb_lim",
            config.lb_lim,
            config.ub_lim,
            float,
            "config.ub_lim",
        )
        config.ub_lim = float(config.ub_lim)

    _ = argtest.oneof(
        "config.lb_mode",
        config.lb_mode,
        "soft",
        "soft-scaled",
        "hard",
        "clamp",
        "none",
        op=(lambda x: x.lower()),
    )
    _ = argtest.oneof(
        "config.ub_mode",
        config.ub_mode,
        "soft",
        "soft-scaled",
        "hard",
        "clamp",
        "none",
        op=(lambda x: x.lower()),
    )

    config.lb_mode, config.ub_mode = (
        config.lb_mode.lower(),
        config.ub_mode.lower(),
    )

    if config.lb_mode != "none" and config.lb_lim is None:
        raise RuntimeError(
            f"'config.lb_lim' cannot be None with 'config.lb_mode' of '{config.lb_mode}'"
        )
    if config.ub_mode != "none" and config.ub_lim is None:
        raise RuntimeError(
            f"'config.ub_lim' cannot be None with 'config.ub_mode' of '{config.ub_mode}'"
        )

    if (config.lb_mode == "soft-scaled" or config.ub_mode == "soft-scaled") and (
        config.lb_lim is None or config.ub_lim is None
    ):
        raise RuntimeError(
            "both 'config.lb_lim' and 'config.ub_limit' cannot be None if either "
            "'config.lb_mode' or 'config.ub_mode' is 'soft-scaled'"
        )

    # add delay dependence bounding
    match config.lb_mode.lower:
        case "soft":
            if config.lb_pow is None or config.lb_pow == 1:
                network.connections.input.updater.delay.lowerbound(
                    inff.bound_lower_multiplicative, config.lb_lim
                )
            else:
                network.connections.input.updater.delay.lowerbound(
                    inff.bound_lower_power, config.lb_lim, power=config.lb_pow
                )
        case "soft-scaled":
            if config.lb_pow is None or config.lb_pow == 1:
                network.connections.input.updater.delay.lowerbound(
                    inff.bound_lower_scaled_multiplicative,
                    config.lb_lim,
                    range=(config.ub_lim - config.lb_lim),
                )
            else:
                network.connections.input.updater.delay.lowerbound(
                    inff.bound_lower_scaled_power,
                    config.lb_lim,
                    power=config.lb_pow,
                    range=(config.ub_lim - config.lb_lim),
                )
        case "hard":
            network.connections.input.updater.delay.lowerbound(
                inff.bound_lower_sharp, config.lb_lim
            )

    match config.ub_mode:
        case "soft":
            if config.ub_pow is None or config.ub_pow == 1:
                network.connections.input.updater.delay.upperbound(
                    inff.bound_upper_multiplicative, config.ub_lim
                )
            else:
                network.connections.input.updater.delay.upperbound(
                    inff.bound_upper_power, config.ub_lim, power=config.ub_pow
                )
        case "soft-scaled":
            if config.ub_pow is None or config.ub_pow == 1:
                network.connections.input.updater.delay.upperbound(
                    inff.bound_upper_scaled_multiplicative,
                    config.ub_lim,
                    range=(config.ub_lim - config.lb_lim),
                )
            else:
                network.connections.input.updater.delay.upperbound(
                    inff.bound_upper_scaled_power,
                    config.ub_lim,
                    power=config.ub_pow,
                    range=(config.ub_lim - config.lb_lim),
                )
        case "hard":
            network.connections.input.updater.delay.upperbound(
                inff.bound_upper_sharp, config.ub_lim
            )

    # add update-triggered clamping
    if config.lb_mode == "clamp" or config.ub_mode == "clamp":
        clamphook = snn.Clamping(
            network.connections.input.updater,
            "parent.delay",
            min=(config.lb_lim if config.lb_mode == "clamp" else None),
            max=(config.ub_lim if config.ub_mode == "clamp" else None),
        )
    else:
        clamphook = None

    return clamphook


def build_mnist_classifiers(
    sqrt_sz: int,
    duration: float,
    step_time: float,
) -> tuple[learn.MaxRateClassifier, FirstSpikeClassifier]:
    return (
        learn.MaxRateClassifier(
            shape=(sqrt_sz, sqrt_sz),
            num_classes=10,
            decay=0.0,
        ),
        FirstSpikeClassifier(
            shape=(sqrt_sz, sqrt_sz),
            num_classes=10,
            duration=duration,
            step_time=step_time,
            decay=0.0,
        ),
    )


class Model(inferno.Module):
    def __init__(
        self,
        network: DCSNN | UDCSNN,
        encoder: inferno.Module,
        mr_classifier: learn.MaxRateClassifier,
        fs_classifier: FirstSpikeClassifier,
        weight_norm: snn.Normalization | None,
        delay_norm: snn.Normalization | None,
        weight_clamp: snn.Clamping | None,
        delay_clamp: snn.Clamping | None,
    ):
        inferno.Module.__init__(self)
        self.network = network
        self.encoder = encoder
        self.mr_classifier = mr_classifier
        self.fs_classifier = fs_classifier
        self.weight_norm = weight_norm
        self.delay_norm = delay_norm
        self.weight_clamp = weight_clamp
        self.delay_clamp = delay_clamp

    def clamp(self, **kwargs):
        if self.weight_clamp:
            self.weight_clamp(**kwargs)
        if self.delay_clamp:
            self.delay_clamp(**kwargs)

    def enable_clamp(self):
        if self.weight_clamp and not self.weight_clamp.registered:
            self.weight_clamp.register()
        if self.delay_clamp and not self.delay_clamp.registered:
            self.delay_clamp.register()

    def disable_clamp(self):
        if self.weight_clamp and self.weight_clamp.registered:
            self.weight_clamp.deregister()
        if self.delay_clamp and self.delay_clamp.registered:
            self.delay_clamp.deregister()

    def normalize(self, **kwargs):
        if self.weight_norm:
            self.weight_norm(**kwargs)
        if self.delay_norm:
            self.delay_norm(**kwargs)

    def enable_norm(self):
        if self.weight_norm and not self.weight_norm.registered:
            self.weight_norm.register()
        if self.delay_norm and not self.delay_norm.registered:
            self.delay_norm.register()

    def disable_norm(self):
        if self.weight_norm and self.weight_norm.registered:
            self.weight_norm.deregister()
        if self.delay_norm and self.delay_norm.registered:
            self.delay_norm.deregister()
