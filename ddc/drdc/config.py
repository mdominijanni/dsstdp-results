import tomllib
import torch
from types import SimpleNamespace
from typing import Any
import warnings
from .. import common

_config_template = """
[meta]
name = "drdc"                          # name of the configuration, directory name, and used by WandB
version = "0.0.1"                      # internal versioning
use_wandb = true                       # if WandB should be used

[meta.wandb]                           # options passed to WandB init
pass_name = true                       # if meta.name should be passed to init
project = "DDC"                        # WandB project to which results will be sent
entity = ""                            # WandB username or team name
group = ""                             # WandB group for this run
notes = ""                             # WandB description for the run
tags = []                              # WandB tags for the run

[simulation]
nsqrt_neurons = 10                     # square root of the number of neurons (each side of a square matrix)
step_time = 1.0                        # length of each simulation step (in Hz)
num_steps = 250                        # number of simulation steps per batch
encoder = "exp-int"                    # encoder used for spikes, either "poisson-int", "exp-int", or "exprefrac-int"
spike_freq = 127.5                     # maximum spike frequency (in Hz) for input spikes (from feature-scaled input)
num_epochs = 1                         # number of simulation epochs
train_batch_size = 50                  # batch size for training
infer_batch_size = 100                 # batch size for validation and testing

[dataset]
ntrain = 50000                         # number of training samples, from MNIST training with 60k
nvalid = 10000                         # number of validation samples, from MNIST training with 60k
ntest = 10000                          # number of testing samples, from MNIST testing with 10k

[runtime]
device = "cpu"                         # PyTorch device string specifying the device used for computations
float_dtype = "float32"                # PyTorch floating point data type, can be one of: "float32", "float64", "float16", "bfloat16"
log_interval = 200                     # samples per wandb log
checkpoint_interval = 2000             # checkpoints which will be overwritten (for recovery)
inplace = true                         # use in-place operations for storing delayed synapse values and training

[reproducibility]
weight_init_seed = 31415926            # seed for determining initial connection weights
delay_init_seed = 62831853             # seed for determining initial connection delays
train_seed = 16180339                  # seed for sampling and ordering the training set
valid_seed = 14142135                  # seed for sampling and ordering the validation set
test_seed = 24142135                   # seed for sampling and ordering the testing set

[connections.feedforward.weight]       # weights for dense connection for inference
init_dist = "uniform"                  # distribution of initial weights, can be one of: "uniform", "normal"
init_min = 0.0                         # minimum value of initial weights
init_max = 0.3                         # maximum value of initial weights
norm_schedule = "batch"                # how often weights are normalized, can be one of: "step", "batch", "never"
norm_vector = "input"                  # if the vector for each input or output should be normalized, can be one of: "input", "output"
norm_order = 1.0                       # order of the target p-norm
norm_scale = 78.4                      # target norm for each vector of weights
norm_autoscale = false                 # if norm scale given is the per-element average
bind_lower_mode = "soft"               # bounding mode for minimums, can be one of: "none", "soft", "scaled-soft", "hard", "clamp"
bind_lower_lim = 0.0                   # bounding minimum, ignored when lower_bound_mode = "none"
bind_lower_pow = 1.0                   # power of lower bound, only used when lower_bound_mode = "soft" or "scaled-soft"
bind_upper_mode = "soft"               # bounding mode for maximums, can be one of: "none", "soft", "scaled-soft", "hard", "clamp"
bind_upper_lim = 1.0                   # bounding maximum, ignored when upper_bound_mode = "none"
bind_upper_pow = 1.0                   # power of upper bound, only used when upper_bound_mode = "soft" or "scaled-soft"

[connections.feedforward.delay]        # delays for dense connection for inference
delay_max = 10.0                       # maximum allowed delay, same as d_init_max when not given
init_dist = "uniform"                  # distribution of initial delays, can be one of: "uniform", "normal"
init_min = 0.0                         # minimum value of initial delays
init_max = 10.0                        # maximum value of initial delays
norm_schedule = "never"                # how often delays are normalized, can be one of: "step", "batch", "never"
norm_vector = "input"                  # if the vector for each input or output should be normalized, can be one of: "input", "output"
norm_order = 1.0                       # order of the target p-norm
norm_scale = 2352.0                    # target norm for each vector of delays
norm_autoscale = false                 # if norm scale given is the per-element average
bind_lower_mode = "clamp"              # bounding mode for minimums, can be one of: "none", "soft", "scaled-soft", "hard", "clamp"
bind_lower_lim = 0.0                   # bounding minimum, ignored when lower_bound_mode = "none"
bind_lower_pow = 1.0                   # power of lower bound, only used when lower_bound_mode = "soft" or "scaled-soft"
bind_upper_mode = "clamp"              # bounding mode for maximums, can be one of: "none", "soft", "scaled-soft", "hard", "clamp"
bind_upper_lim = 10.0                  # bounding maximum, ignored when upper_bound_mode = "none"
bind_upper_pow = 1.0                   # power of upper bound, only used when upper_bound_mode = "soft" or "scaled-soft"

[connections.lateral]                  # lateral and direct connections for lateral inhibition
exc2inh_weight = 22.5                  # weights from the excitatory neurons to the inhibitory neurons
inh2exc_weight = -180.0                # weights from the inhibitory neurons to the excitatory neurons

[neurons.exc]                          # excitatory neurons (ALIF)
rest_v = -65.0                         # membrane rest potential (in mV)
reset_v = -60.0                        # membrane reset potential (in mV)
thresh_eq_v = -52.0                    # equilibrium of membrane threshold potential (in mV)
refrac_t = 5.0                         # absolute refractory period (in ms)
tc_membrane = 100.0                    # time constant of exponential decay for membrane voltage (in ms)
tc_adaptation = 1e7                    # time constant of exponential decay for adaptive thresholds (in ms), can be a list
spike_adapt_incr = 0.05                # threshold increase on spiking (in mV), can be a list
resistance = 1.0                       # membrane resistance (in MOhm)

[neurons.inh]                          # inhibitory neurons (LIF)
rest_v = -60.0                         # membrane rest potential (in mV)
reset_v = -45.0                        # membrane reset potential (in mV)
thresh_v = -40.0                       # equilibrium of membrane threshold potential (in mV)
refrac_t = 2.0                         # absolute refractory period (in ms)
tc_membrane = 75.0                     # time constant of exponential decay for membrane voltage (in ms)
resistance = 1.0                       # membrane resistance (in MOhm)

[training.weight]
lr_pos = 2e-4                          # depressive learning rate for causal spike pairs
lr_neg = -2e-6                         # potentiative learning rate for anti-causal spike pairs
tc_pos = 20.0                          # time constant of exponential decay for adjusted presynaptic traces (in ms)
tc_neg = 20.0                          # time constant of exponential decay for adjusted postsynaptic traces (in ms)
batch_reduction = "mean"               # batch axis reduction, can be one of: "mean", "sum", "max", "median", ["quantile", q], ["quantile", q, interp], "geomean"

[training.delay]
lr_pos = 1e-3                          # depressive learning rate for causal spike pairs
lr_neg = -1e-3                         # potentiative learning rate for anti-causal spike pairs
tc_pos = 10.0                          # time constant of exponential decay for adjusted presynaptic traces (in ms)
tc_neg = 10.0                          # time constant of exponential decay for adjusted postsynaptic traces (in ms)
batch_reduction = "mean"               # batch axis reduction, can be one of: "mean", "sum", "max", "median", ["quantile", q], ["quantile", q, interp], "geomean"

[bounding.weight]                      # bounding of feedforward connection weights
lower_bound_mode = "soft"              # bounding mode for minimums, can be one of: "none", "soft", "scaled-soft", "hard", "clamp"
lower_bound_lim = 0.0                  # weight bounding minimum, ignored when lower_bound_mode = "none"
lower_bound_pow = 1.0                  # power of lower bound, only used when lower_bound_mode = "soft" or "scaled-soft"
upper_bound_mode = "soft"              # bounding mode for maximums, can be one of: "none", "soft", "scaled-soft", "hard", "clamp"
upper_bound_lim = 1.0                  # weight bounding maximum, ignored when upper_bound_mode = "none"
upper_bound_pow = 1.0                  # power of upper bound, only used when upper_bound_mode = "soft" or "scaled-soft"

[bounding.delay]                       # bounding of feedforward connection delays
lower_bound_mode = "clamp"             # bounding mode for minimums, can be one of: "none", "soft", "scaled-soft", "hard", "clamp"
lower_bound_lim = 0.0                  # delay bounding minimum, ignored when lower_bound_mode = "none"
lower_bound_pow = 1.0                  # power of lower bound, only used when lower_bound_mode = "soft" or "scaled-soft"
upper_bound_mode = "clamp"             # bounding mode for maximums, can be one of: "none", "soft", "scaled-soft", "hard", "clamp"
upper_bound_lim = 10.0                 # delay bounding maximum, ignored when upper_bound_mode = "none"
upper_bound_pow = 1.0                  # power of upper bound, only used when upper_bound_mode = "soft" or "scaled-soft"
"""


def default_config() -> list[str]:
    return _config_template.splitlines()[1:]


def ingest_config(config: list[str], /, **kwargs) -> SimpleNamespace:
    config = "\n".join(config)
    confd = tomllib.loads(config)
    if kwargs:
        confd = _inject_runtime_settings(confd, **kwargs)

    version = confd.get("meta", {}).get("version", None)

    if version is None:
        raise RuntimeError("configuration does not include 'meta.version' field")
    else:
        match version:
            case "0.0.1":
                confn = _ingest_current_version(confd)
                confn.wandb.config = confd | {"rawfile": config.split("\n")}
                return confn
            case _:
                raise RuntimeError(
                    f"configurations with version {version} cannot be loaded"
                )


def _inject_runtime_settings(confd: dict[str, Any], **kwargs) -> dict[str, Any]:
    version = confd.get("meta", {}).get("version", None)

    if version is None:
        raise RuntimeError("configuration does not include 'meta.version' field")
    else:
        match version:
            case "0.0.1":
                if "device" in kwargs:
                    confd["runtime"]["device"] = kwargs["device"]
                if "use_wandb" in kwargs:
                    confd["meta"]["use_wandb"] = kwargs["use_wandb"]
                return confd
            case _:
                warnings.warn("failed to override configuration with on-run options")


def _ingest_current_version(confd: dict[str, Any]) -> SimpleNamespace:
    config = SimpleNamespace()

    config.env = SimpleNamespace(
        name=confd["meta"]["name"],
    )

    config.wandb = SimpleNamespace(
        enable=confd["meta"]["use_wandb"],
        options=(
            {
                "name": (
                    confd["meta"]["name"]
                    if confd["meta"]["wandb"]["pass_name"]
                    else None
                )
            }
            | dict(
                map(
                    lambda e: e if e[1] else (e[0], None),
                    filter(
                        lambda e: e[0] != "pass_name", confd["meta"]["wandb"].items()
                    ),
                )
            )
        ),
    )

    config.verse = SimpleNamespace(
        dt=float(confd["simulation"]["step_time"]),
        device=confd["runtime"]["device"],
    )
    match confd["runtime"]["float_dtype"]:
        case "float32":
            config.verse.dtype = torch.float32
        case "float64":
            config.verse.dtype = torch.float64
        case "float16":
            config.verse.dtype = torch.float16
        case "bfloat16":
            config.verse.dtype = torch.bfloat16
        case _:
            config.verse.dtype = torch.float32

    config.iters = SimpleNamespace(
        epochs=int(confd["simulation"]["num_epochs"]),
        log_interval=int(confd["runtime"]["log_interval"]),
        checkpt_interval=int(confd["runtime"]["checkpoint_interval"]),
    )

    config.batchsz = SimpleNamespace(
        train=int(confd["simulation"]["train_batch_size"]),
        infer=int(confd["simulation"]["infer_batch_size"]),
    )

    config.seed = SimpleNamespace(
        weights=int(confd["reproducibility"]["weight_init_seed"]),
        delays=int(confd["reproducibility"]["delay_init_seed"]),
        train=int(confd["reproducibility"]["train_seed"]),
        valid=int(confd["reproducibility"]["valid_seed"]),
        test=int(confd["reproducibility"]["test_seed"]),
    )

    config.data = SimpleNamespace(
        ntrain=int(confd["dataset"]["ntrain"]),
        nvalid=int(confd["dataset"]["nvalid"]),
        ntest=int(confd["dataset"]["ntest"]),
    )

    config.split = common.DataSplit(
        n_train=config.data.ntrain,
        n_valid=config.data.nvalid,
        n_test=config.data.ntest,
        train_seed=config.seed.train,
        valid_seed=config.seed.valid,
        test_seed=config.seed.test,
    )

    config.sim = SimpleNamespace()
    config.sim.nsqrt_neurons = int(confd["simulation"]["nsqrt_neurons"])

    config.model = SimpleNamespace()
    config.model.encoder = common.RateEncoderConfigV1(
        encoder=confd["simulation"]["encoder"].lower(),
        step_time=config.verse.dt,
        n_steps=int(confd["simulation"]["num_steps"]),
        max_freq=int(confd["simulation"]["spike_freq"]),
    )
    config.model.netargs = {
        "step_time": config.verse.dt,
        "sqrt_output_sz": int(confd["simulation"]["nsqrt_neurons"]),
        "weight_init": common.ParameterInitializerV1(
            dist=confd["connections"]["feedforward"]["weight"]["init_dist"],
            seed=config.seed.weights,
            pmin=confd["connections"]["feedforward"]["weight"]["init_min"],
            pmax=confd["connections"]["feedforward"]["weight"]["init_max"],
        ),
        "delay_init": common.ParameterInitializerV1(
            dist=confd["connections"]["feedforward"]["delay"]["init_dist"],
            seed=config.seed.delays,
            pmin=confd["connections"]["feedforward"]["delay"]["init_min"],
            pmax=confd["connections"]["feedforward"]["delay"]["init_max"],
        ),
        "delay_max": confd["connections"]["feedforward"]["delay"].get(
            "delay_max", None
        ),
        "exc2inh_weight": float(confd["connections"]["lateral"]["exc2inh_weight"]),
        "inh2exc_weight": float(confd["connections"]["lateral"]["inh2exc_weight"]),
        "exc_config": common.ALIFPartialConfig(
            rest_v=confd["neurons"]["exc"]["rest_v"],
            reset_v=confd["neurons"]["exc"]["reset_v"],
            thresh_eq_v=confd["neurons"]["exc"]["thresh_eq_v"],
            refrac_t=confd["neurons"]["exc"]["refrac_t"],
            tc_membrane=confd["neurons"]["exc"]["tc_membrane"],
            tc_adaptation=confd["neurons"]["exc"]["tc_adaptation"],
            spike_increment=confd["neurons"]["exc"]["spike_adapt_incr"],
            resistance=confd["neurons"]["exc"]["resistance"],
        ),
        "inh_config": common.LIFPartialConfig(
            rest_v=confd["neurons"]["inh"]["rest_v"],
            reset_v=confd["neurons"]["inh"]["reset_v"],
            thresh_v=confd["neurons"]["inh"]["thresh_v"],
            refrac_t=confd["neurons"]["inh"]["refrac_t"],
            time_constant=confd["neurons"]["inh"]["tc_membrane"],
            resistance=confd["neurons"]["inh"]["resistance"],
        ),
        "inplace": confd["runtime"]["inplace"],
    }
    config.model.norms = SimpleNamespace()
    if (
        confd["connections"]["feedforward"]["weight"]["norm_schedule"].lower()
        == "never"
    ):
        config.model.norms.weight = None
    else:
        config.model.norms.weight = common.DenseNormalizationConfig(
            schedule=confd["connections"]["feedforward"]["weight"][
                "norm_schedule"
            ].lower(),
            vector=confd["connections"]["feedforward"]["weight"]["norm_vector"].lower(),
            scale=confd["connections"]["feedforward"]["weight"]["norm_scale"],
            order=confd["connections"]["feedforward"]["weight"]["norm_order"],
            autoscale=bool(
                confd["connections"]["feedforward"]["weight"]["norm_autoscale"]
            ),
        )
    if confd["connections"]["feedforward"]["delay"]["norm_schedule"].lower() == "never":
        config.model.norms.delay = None
    else:
        config.model.norms.delay = common.DenseNormalizationConfig(
            schedule=confd["connections"]["feedforward"]["delay"][
                "norm_schedule"
            ].lower(),
            vector=confd["connections"]["feedforward"]["delay"]["norm_vector"].lower(),
            scale=confd["connections"]["feedforward"]["delay"]["norm_scale"],
            order=confd["connections"]["feedforward"]["delay"]["norm_order"],
            autoscale=bool(
                confd["connections"]["feedforward"]["delay"]["norm_autoscale"]
            ),
        )
    config.model.bounds = SimpleNamespace()
    if (
        confd["connections"]["feedforward"]["weight"]["bind_lower_mode"].lower()
        == "none"
        and confd["connections"]["feedforward"]["weight"]["bind_upper_mode"].lower()
        == "none"
    ):
        config.model.bounds.weight = None
    else:
        config.model.bounds.weight = common.ParameterBoundingV1(
            lb_mode=confd["connections"]["feedforward"]["weight"][
                "bind_lower_mode"
            ].lower(),
            lb_lim=confd["connections"]["feedforward"]["weight"].get(
                "bind_lower_lim", None
            ),
            lb_pow=confd["connections"]["feedforward"]["weight"].get(
                "bind_lower_pow", None
            ),
            ub_mode=confd["connections"]["feedforward"]["weight"][
                "bind_upper_mode"
            ].lower(),
            ub_lim=confd["connections"]["feedforward"]["weight"].get(
                "bind_upper_lim", None
            ),
            ub_pow=confd["connections"]["feedforward"]["weight"].get(
                "bind_upper_pow", None
            ),
        )
    if (
        confd["connections"]["feedforward"]["delay"]["bind_lower_mode"].lower()
        == "none"
        and confd["connections"]["feedforward"]["delay"]["bind_upper_mode"].lower()
        == "none"
    ):
        config.model.bounds.delay = None
    else:
        config.model.bounds.delay = common.ParameterBoundingV1(
            lb_mode=confd["connections"]["feedforward"]["delay"][
                "bind_lower_mode"
            ].lower(),
            lb_lim=confd["connections"]["feedforward"]["delay"].get(
                "bind_lower_lim", None
            ),
            lb_pow=confd["connections"]["feedforward"]["delay"].get(
                "bind_lower_pow", None
            ),
            ub_mode=confd["connections"]["feedforward"]["delay"][
                "bind_upper_mode"
            ].lower(),
            ub_lim=confd["connections"]["feedforward"]["delay"].get(
                "bind_upper_lim", None
            ),
            ub_pow=confd["connections"]["feedforward"]["delay"].get(
                "bind_upper_pow", None
            ),
        )
    config.model.weighttrainerargs = {
        "lr_pos": confd["training"]["weight"]["lr_pos"],
        "lr_neg": confd["training"]["weight"]["lr_neg"],
        "tc_pos": confd["training"]["weight"]["tc_pos"],
        "tc_neg": confd["training"]["weight"]["tc_neg"],
        "batch_reduction": confd["training"]["weight"]["batch_reduction"],
        "inplace": confd["runtime"]["inplace"],
    }
    config.model.delaytrainerargs = {
        "lr_pos": confd["training"]["delay"]["lr_pos"],
        "lr_neg": confd["training"]["delay"]["lr_neg"],
        "tc_pos": confd["training"]["delay"]["tc_pos"],
        "tc_neg": confd["training"]["delay"]["tc_neg"],
        "batch_reduction": confd["training"]["delay"]["batch_reduction"],
        "inplace": confd["runtime"]["inplace"],
    }
    config.model.classargs = {
        "sqrt_sz": int(confd["simulation"]["nsqrt_neurons"]),
        "duration": int(confd["simulation"]["num_steps"]) * config.verse.dt,
        "step_time": config.verse.dt,
    }

    return config
