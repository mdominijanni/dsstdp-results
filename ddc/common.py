from dataclasses import dataclass
import torch
from torch.utils.data import Subset
from torchvision import datasets
from torchvision.transforms import v2
from .utils import splitdata
from typing import Literal


@dataclass
class RandomGroup:
    train: torch.Generator
    valid: torch.Generator
    test: torch.Generator


@dataclass
class RNG:
    sample: RandomGroup
    encode: RandomGroup


@dataclass
class RateEncoderConfigV1:
    encoder: Literal["poisson-int", "exp-int", "exprefrac-int"]
    step_time: float
    n_steps: int
    max_freq: float


@dataclass
class LIFPartialConfig:
    rest_v: float
    reset_v: float
    thresh_v: float
    refrac_t: float
    time_constant: float
    resistance: float


@dataclass
class ALIFPartialConfig:
    rest_v: float
    reset_v: float
    thresh_eq_v: float
    refrac_t: float
    tc_membrane: float
    tc_adaptation: float | tuple[float, ...]
    spike_increment: float | tuple[float, ...]
    resistance: float


@dataclass
class ParameterInitializerV1:
    dist: Literal["uniform", "normal"]
    seed: int
    pmin: float
    pmax: float


@dataclass
class ParameterBoundingV1:
    lb_mode: Literal["soft", "soft-scaled", "hard", "clamp", "none"]
    lb_lim: float | None
    lb_pow: float | None
    ub_mode: Literal["soft", "soft-scaled", "hard", "clamp", "none"]
    ub_lim: float | None
    ub_pow: float | None


@dataclass
class DenseNormalizationConfig:
    schedule: Literal["step", "batch"]
    vector: Literal["input", "output"]
    scale: float
    order: float
    autoscale: bool


@dataclass
class DataSplit:
    n_train: int
    n_valid: int
    n_test: int
    train_seed: int | None
    valid_seed: int | None
    test_seed: int | None


def mnist(
    split: DataSplit, dtype: torch.dtype, path: str
) -> tuple[Subset, Subset, Subset]:
    # get datasets via torchvision
    ftrain = datasets.MNIST(
        root=path,
        download=True,
        train=True,
        transform=v2.Compose([v2.ToImage(), v2.ToDtype(dtype, scale=True)]),
    )
    ftest = datasets.MNIST(
        root=path,
        download=True,
        train=False,
        transform=v2.Compose([v2.ToImage(), v2.ToDtype(dtype, scale=True)]),
    )

    # compute splits
    train_set, rem = splitdata(ftrain, split.n_train, split.train_seed, permissive=True)
    valid_set, _ = splitdata(rem, split.n_valid, split.valid_seed, permissive=True)
    test_set, _ = splitdata(ftest, split.n_test, split.test_seed, permissive=True)

    return train_set, valid_set, test_set


def fashion_mnist(
    split: DataSplit, dtype: torch.dtype, path: str
) -> tuple[Subset, Subset, Subset]:
    # get datasets via torchvision
    ftrain = datasets.FashionMNIST(
        root=path,
        download=True,
        train=True,
        transform=v2.Compose([v2.ToImage(), v2.ToDtype(dtype, scale=True)]),
    )
    ftest = datasets.FashionMNIST(
        root=path,
        download=True,
        train=False,
        transform=v2.Compose([v2.ToImage(), v2.ToDtype(dtype, scale=True)]),
    )

    # compute splits
    train_set, rem = splitdata(ftrain, split.n_train, split.train_seed, permissive=True)
    valid_set, _ = splitdata(rem, split.n_valid, split.valid_seed, permissive=True)
    test_set, _ = splitdata(ftest, split.n_test, split.test_seed, permissive=True)

    return train_set, valid_set, test_set
