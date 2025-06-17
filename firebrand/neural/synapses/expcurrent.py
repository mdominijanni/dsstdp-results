from inferno.neural.synapses.mixins import SpikeCurrentMixin
from inferno.neural import InfernoSynapse
from inferno._internal import argtest
from inferno.functional import interp_nearest, interp_previous, interp_expdecay
from collections.abc import Sequence
import math
import torch
from typing import Literal


class SingleExponentialPlusCurrent(SpikeCurrentMixin, InfernoSynapse):
    r"""Instantly applied exponentially decaying current-based synapse, with injected current.

    .. math::
        I(t + \Delta t) = I(t) \exp\left(-\frac{\Delta t}{\tau}\right)
        + \frac{Q}{\tau} [t = t_f] + I_x(t)

    Attributes:
        spike_: :py:class:`~inferno.RecordTensor` interface for spikes.
        current_: :py:class:`~inferno.RecordTensor` interface for currents.

    Args:
        shape (Sequence[int] | int): shape of the group of synapses being simulated.
        step_time (float): length of a simulation time step, :math:`\Delta t`,
            in :math:`\text{ms}`.
        spike_charge (float): charge carried by each presynaptic spike, :math:`Q`,
            in :math:`\text{pC}`.
        time_constant (float): exponential time constant for current decay, :math:`\tau`,
            in :math:`\text{ms}`.
        delay (float, optional): maximum supported delay, in :math:`\text{ms}`.
            Defaults to ``0.0``.
        spike_interp_mode (Literal["nearest", "previous"], optional): interpolation mode
            for spike selectors between observations. Defaults to ``"nearest"``.
        interp_tol (float, optional): maximum difference in time from an observation
            to treat as co-occurring, in :math:`\text{ms}`. Defaults to ``0.0``.
        current_overbound (float | None, optional): value to replace currents out of
            bounds, uses values at observation limits if ``None``. Defaults to ``0.0``.
        spike_overbound (bool | None, optional): value to replace spikes out of bounds,
            uses values at observation limits if ``None``. Defaults to ``False``.
        batch_size (int, optional): size of input batches for simulation. Defaults to ``1``.
        inplace (bool): if write operations on :py:class:`~inferno.RecordTensor` attributes
            should be performed with in-place operations. Defaults to ``False``.

    See Also:
        For more details and references, visit
        :ref:`zoo/synapses-current:Single Exponential` in the zoo.
    """

    def __init__(
        self,
        shape: Sequence[int] | int,
        step_time: float,
        *,
        spike_charge: float,
        time_constant: float,
        delay: float = 0.0,
        spike_interp_mode: Literal["nearest", "previous"] = "previous",
        interp_tol: float = 0.0,
        current_overbound: float | None = 0.0,
        spike_overbound: bool | None = False,
        batch_size: int = 1,
        inplace: bool = False,
    ):
        # call superclass constructor
        InfernoSynapse.__init__(self, shape, step_time, delay, batch_size, inplace)

        # synapse attributes
        self.spike_charge = argtest.neq("spike_charge", spike_charge, 0, float)
        self.time_constant = argtest.gt("time_constant", time_constant, 0, float)

        match spike_interp_mode.lower():
            case "nearest":
                spike_interp_mode = interp_nearest
            case "previous":
                spike_interp_mode = interp_previous
            case _:
                raise RuntimeError(
                    f"invalid ispike_interp_modenterp_mode '{spike_interp_mode}' received, "
                    "must be one of 'nearest' or 'previous'."
                )

        # call mixin constructor
        SpikeCurrentMixin.__init__(
            self,
            torch.zeros(*self.batchedshape),
            torch.zeros(*self.batchedshape, dtype=torch.bool),
            current_interp=interp_expdecay,
            current_interp_kwargs={"time_constant": self.time_constant},
            spike_interp=spike_interp_mode,
            spike_interp_kwargs={},
            current_overbound=current_overbound,
            spike_overbound=spike_overbound,
            tolerance=interp_tol,
        )

    @classmethod
    def partialconstructor(
        cls,
        spike_charge: float,
        time_constant: float,
        spike_interp_mode: Literal["nearest", "previous"] = "previous",
        interp_tol: float = 0.0,
        current_overbound: float | None = 0.0,
        spike_overbound: bool | None = False,
        inplace: bool = False,
    ):
        r"""Returns a function with a common signature for synapse construction.

        Args:
            spike_charge (float): charge carried by each presynaptic spike, in :math:`\text{pC}`.
            time_constant (float): exponential time constant for current decay, in :math:`\text{ms}`.
            spike_interp_mode (Literal["nearest", "previous"], optional): interpolation mode
                for spike selectors between observations. Defaults to ``"nearest"``.
            interp_tol (float, optional): maximum difference in time from an observation
                to treat as co-occurring, in :math:`\text{ms}`. Defaults to ``0.0``.
            current_overbound (float | None, optional): value to replace currents out of
                bounds, uses values at observation limits if ``None``. Defaults to ``0.0``.
            spike_overbound (bool | None, optional): value to replace spikes out of bounds,
                uses values at observation limits if ``None``. Defaults to ``False``.
            inplace (bool): if write operations on :py:class:`~inferno.RecordTensor` attributes
                should be performed with in-place operations. Defaults to ``False``.

        Returns:
            SynapseConstructor: partial constructor for synapse.
        """

        def constructor(
            shape: tuple[int, ...] | int,
            step_time: float,
            delay: float,
            batch_size: int,
        ):
            return cls(
                shape=shape,
                step_time=step_time,
                spike_charge=spike_charge,
                time_constant=time_constant,
                delay=delay,
                spike_interp_mode=spike_interp_mode,
                interp_tol=interp_tol,
                current_overbound=current_overbound,
                spike_overbound=spike_overbound,
                batch_size=batch_size,
                inplace=inplace,
            )

        return constructor

    def clear(self, **kwargs) -> None:
        r"""Resets synapses to their resting state."""
        self.spike_.reset(False)
        self.current_.reset(0.0)

    def forward(self, *inputs: torch.Tensor, **kwargs) -> torch.Tensor:
        r"""Runs a simulation step of the synaptic dynamics.

        Args:
            *inputs (torch.Tensor): input spikes to the synapse.

        Returns:
            torch.Tensor: synaptic currents after simulation step.

        Important:
            The first tensor of ``*inputs`` will represent the input spikes. Any
            subsequent tensors will be treated as injected current. These must be
            broadcastable with
            :py:attr:`~inferno.neural.synapses.mixins.CurrentMixin.current`.
        """
        self.spike = inputs[0].bool()
        self.current = (
            self.current * math.exp(-self.dt / self.time_constant)
            + (self.spike_charge / self.time_constant) * inputs[0]
            + sum(inputs[1:])
        )
        return self.current
