from .. import dcshared
from functools import partial
import inferno.functional as inff
from inferno.learn import DelayAdjustedSTDP, DelayAdjustedSTDPD
import torch


def build_weight_trainer(
    network: dcshared.DCSNN,
    lr_pos: float,
    lr_neg: float,
    tc_pos: float,
    tc_neg: float,
    batch_reduction: str,
    inplace: bool,
) -> DelayAdjustedSTDP:
    # add updater to the connection
    if not network.connections.input.updatable:
        network.connections.input.updater = network.connections.input.defaultupdater()

    # replace batch reduction string with function
    if not isinstance(batch_reduction, str):
        batch_reduction, br_args = batch_reduction[0], batch_reduction[1:]

    match batch_reduction:
        case "mean":
            batch_reduction = torch.mean
        case "sum":
            batch_reduction = torch.sum
        case "max":
            batch_reduction = torch.amax
        case "median":
            batch_reduction = inff.median
        case "geomean":
            batch_reduction = inff.geomean
        case "quantile":
            if len(br_args) == 0:
                batch_reduction = partial(inff.quantile, q=0.5, interpolation="linear")
            elif len(br_args) == 1:
                batch_reduction = partial(
                    inff.quantile, q=br_args[0], interpolation="linear"
                )
            else:
                batch_reduction = partial(
                    inff.quantile, q=br_args[0], interpolation=br_args[2]
                )
        case _:
            raise RuntimeError(
                f"an invalid 'batch_reduction' of '{batch_reduction}' was specified"
            )

    # build the trainer
    trainer = DelayAdjustedSTDP(
        lr_pos=lr_pos,
        lr_neg=lr_neg,
        tc_pos=tc_pos,
        tc_neg=tc_neg,
        interp_tolerance=0.0,
        batch_reduction=batch_reduction,
        inplace=inplace,
    )

    # add input cell to the trainer
    trainer.register_cell("ff", network.dclayer.add_cell("enc2exc", "exc"))

    return trainer


def build_delay_trainer(
    network: dcshared.DCSNN,
    lr_pos: float,
    lr_neg: float,
    tc_pos: float,
    tc_neg: float,
    batch_reduction: str,
    inplace: bool,
) -> DelayAdjustedSTDPD:
    # add updater to the connection
    if not network.connections.input.updatable:
        network.connections.input.updater = network.connections.input.defaultupdater()

    # replace batch reduction string with function
    if not isinstance(batch_reduction, str):
        batch_reduction, br_args = batch_reduction[0], batch_reduction[1:]

    match batch_reduction:
        case "mean":
            batch_reduction = torch.mean
        case "sum":
            batch_reduction = torch.sum
        case "max":
            batch_reduction = torch.amax
        case "median":
            batch_reduction = inff.median
        case "geomean":
            batch_reduction = inff.geomean
        case "quantile":
            if len(br_args) == 0:
                batch_reduction = partial(inff.quantile, q=0.5, interpolation="linear")
            elif len(br_args) == 1:
                batch_reduction = partial(
                    inff.quantile, q=br_args[0], interpolation="linear"
                )
            else:
                batch_reduction = partial(
                    inff.quantile, q=br_args[0], interpolation=br_args[2]
                )
        case _:
            raise RuntimeError(
                f"an invalid 'batch_reduction' of '{batch_reduction}' was specified"
            )

    # build the trainer
    trainer = DelayAdjustedSTDPD(
        lr_neg=lr_neg,
        lr_pos=lr_pos,
        tc_neg=tc_neg,
        tc_pos=tc_pos,
        interp_tolerance=0.0,
        batch_reduction=batch_reduction,
        inplace=inplace,
    )

    # add input cell to the trainer
    trainer.register_cell("ff", network.dclayer.add_cell("enc2exc", "exc"))

    return trainer
