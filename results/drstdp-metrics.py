import pandas as pd
import pickle

# ONLY UP TO N=400
if __name__ == "__main__":
    # field constants
    EPOCH_FIELD = "epoch"
    SAMPLE_FIELD = "sample"

    # files to read/write
    ACC_METRICS_FN = "drstdp-acc.csv"
    DYN_METRICS_FN = "drstdp-dyn.csv"
    PICKLE_FN = "drstdp.pickle"

    # map of neuron counts to run name
    trials = {n: f"drdc-r2-n{n}" for n in (100, 225, 400, 625, 900)}

    # map of field names
    acc_fields = {
        "valid.acc.rate_prop": "validacc.rate",
        "test.acc.rate_prop": "acc.rate",
        **{f"test.clsacc.c{k}.rate_prop": f"clsacc.c{k}.rate" for k in range(10)},
    } | {
        "valid.acc.ttfs_prop": "validacc.ttfs",
        "test.acc.ttfs_prop": "acc.ttfs",
        **{f"test.clsacc.c{k}.ttfs_prop": f"clsacc.c{k}.ttfs" for k in range(10)},
    }

    dyn_fields = {
        "online.weight.mean": "weight.mean",
        "online.weight.var": "weight.var",
        "online.delay.mean": "delay.mean",
        "online.delay.var": "delay.var",
    }

    # load dataframes
    acc_df = pd.read_csv(ACC_METRICS_FN)
    dyn_df = pd.read_csv(DYN_METRICS_FN)

    # create metrics dictionaries
    acc_metrics = {
        n: acc_df.loc[:, [EPOCH_FIELD, *(f"{p} - {f}" for f in acc_fields)]]
        .rename(columns={f"{p} - {f}": t for f, t in acc_fields.items()})
        .dropna(axis=0, how="all", subset=acc_fields.values())
        for n, p in trials.items()
    }

    dyn_metrics = {
        n: dyn_df.loc[:, [SAMPLE_FIELD, *(f"{p} - {f}" for f in dyn_fields)]]
        .rename(columns={f"{p} - {f}": t for f, t in dyn_fields.items()})
        .dropna(axis=0, how="all", subset=dyn_fields.values())
        for n, p in trials.items()
    }

    # serialize data
    with open(PICKLE_FN, "wb") as handle:
        pickle.dump(
            {"acc": acc_metrics, "dyn": dyn_metrics},
            handle,
            protocol=pickle.HIGHEST_PROTOCOL,
        )
