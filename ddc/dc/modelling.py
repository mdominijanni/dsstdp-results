from .. import dcshared
from inferno.learn import STDP
from functools import partial
import inferno.functional as inff
import torch


def build_trainer(
    network: dcshared.UDCSNN,
    lr_post: float,
    lr_pre: float,
    tc_post: float,
    tc_pre: float,
    batch_reduction: str,
) -> STDP:
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
    trainer = STDP(
        lr_post=lr_post,
        lr_pre=lr_pre,
        tc_post=tc_post,
        tc_pre=tc_pre,
        interp_tolerance=0.0,
        batch_reduction=batch_reduction,
    )

    # add input cell to the trainer
    trainer.register_cell("ff", network.dclayer.add_cell("enc2exc", "exc"))

    return trainer
