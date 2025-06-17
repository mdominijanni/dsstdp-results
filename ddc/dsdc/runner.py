import os
from dataclasses import asdict
from matplotlib import pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
from tqdm import tqdm
import wandb
from .. import common, dcshared, utils
from .modelling import build_trainer
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, RandomSampler
from types import SimpleNamespace


def build_model(cfg: SimpleNamespace) -> dcshared.Model:
    with torch.no_grad():
        network = dcshared.build_mnist_network(**cfg.model.netargs)
        encoder = dcshared.build_encoder(cfg.model.encoder)
        mr_classifier, fs_classifier = dcshared.build_mnist_classifiers(**cfg.model.classargs)
        if cfg.model.norms.weight:
            weight_norm = dcshared.add_weight_normalization(network, cfg.model.norms.weight)
        else:
            weight_norm = None
        if cfg.model.norms.delay:
            delay_norm = dcshared.add_delay_normalization(network, cfg.model.norms.delay)
        else:
            delay_norm = None

        if cfg.model.bounds.weight:
            weight_clamp = dcshared.add_weight_bounding(network, cfg.model.bounds.weight)
        else:
            weight_clamp = None
        if cfg.model.bounds.delay:
            delay_clamp = dcshared.add_delay_bounding(network, cfg.model.bounds.delay)
        else:
            delay_clamp = None

    return dcshared.Model(
        network=network,
        encoder=encoder,
        mr_classifier=mr_classifier,
        fs_classifier=fs_classifier,
        weight_norm=weight_norm,
        delay_norm=delay_norm,
        weight_clamp=weight_clamp,
        delay_clamp=delay_clamp,
    )


def build_environ(cfg: SimpleNamespace) -> SimpleNamespace:
    # override for torch.poisson not being enabled on mps
    if cfg.verse.device.partition(":")[
        0
    ].lower() == "mps" and dcshared.encoder_requires_mps_fallback(cfg.model.encoder):
        cfg.verse.fallbackdevice = "cpu"
    else:
        cfg.verse.fallbackdevice = cfg.verse.device

    m = build_model(cfg).to(device=cfg.verse.device, dtype=cfg.verse.dtype)
    return SimpleNamespace(
        cfg=cfg,
        m=m,
        trainer=build_trainer(m.network, **cfg.model.trainerargs).to(
            device=cfg.verse.device, dtype=cfg.verse.dtype
        ),
        prog=dcshared.default_progress(),
        batchlog=dcshared.Log(),
        epochlog=dcshared.Log(),
        rng=common.RNG(
            sample=common.RandomGroup(
                train=torch.Generator(device="cpu").manual_seed(cfg.seed.train),
                valid=torch.Generator(device="cpu").manual_seed(cfg.seed.valid),
                test=torch.Generator(device="cpu").manual_seed(cfg.seed.test),
            ),
            encode=common.RandomGroup(
                train=torch.Generator(device=cfg.verse.fallbackdevice).manual_seed(
                    cfg.seed.train
                ),
                valid=torch.Generator(device=cfg.verse.fallbackdevice).manual_seed(
                    cfg.seed.valid
                ),
                test=torch.Generator(device=cfg.verse.fallbackdevice).manual_seed(
                    cfg.seed.test
                ),
            ),
        ),
    )


def dataexport(env: SimpleNamespace, fname: str) -> None:
    batchsz = env.m.network.batchsz
    env.m.network.batchsz = 1
    utils.save_model(
        fname,
        {
            "network": env.m.network,
            "mr_classifier": env.m.mr_classifier,
            "fs_classifier": env.m.fs_classifier,
            "batchlog": env.batchlog,
            "epochlog": env.epochlog,
        },
        {
            "sample_train": env.rng.sample.train,
            "sample_valid": env.rng.sample.valid,
            "sample_test": env.rng.sample.test,
            "encode_train": env.rng.encode.train,
            "encode_valid": env.rng.encode.valid,
            "encode_test": env.rng.encode.test,
        },
        {"progress": asdict(env.prog)},
    )
    env.m.network.batchsz = batchsz


def dataimport(env: SimpleNamespace, fname: str) -> None:
    batchsz = env.m.network.batchsz
    env.m.network.batchsz = 1
    _, _, aux = utils.load_model(
        fname,
        {
            "network": env.m.network,
            "mr_classifier": env.m.mr_classifier,
            "fs_classifier": env.m.fs_classifier,
            "batchlog": env.batchlog,
            "epochlog": env.epochlog,
        },
        {
            "sample_train": env.rng.sample.train,
            "sample_valid": env.rng.sample.valid,
            "sample_test": env.rng.sample.test,
            "encode_train": env.rng.encode.train,
            "encode_valid": env.rng.encode.valid,
            "encode_test": env.rng.encode.test,
        },
    )
    env.m.network.batchsz = batchsz
    env.prog = dcshared.Progress(**aux["progress"])


def clear(env: SimpleNamespace) -> None:
    env.m.network.clear(keep_adaptations=True)
    env.trainer.clear(keepshape=False)


def epochreset(env: SimpleNamespace, ds: str) -> None:
    match ds:
        case "train":
            if env.prog.train == 0:
                mr_classifier, fs_classifier = dcshared.build_mnist_classifiers(
                    **env.cfg.model.classargs
                )
                env.m.mr_classifier = mr_classifier.to(
                    device=env.cfg.verse.device, dtype=env.cfg.verse.dtype
                )
                env.m.fs_classifier = fs_classifier.to(
                    device=env.cfg.verse.device, dtype=env.cfg.verse.dtype
                )
            env.m.network.batchsz = env.cfg.batchsz.train
            env.m.encoder.generator = env.rng.encode.train
            env.m.train()
            env.trainer.train()
            env.m.normalize(force=True, ignore_mode=True)
            env.m.enable_norm()
            env.m.enable_clamp()
        case "valid":
            env.m.normalize(force=True, ignore_mode=True)
            env.m.disable_norm()
            env.m.disable_clamp()
            env.m.network.batchsz = env.cfg.batchsz.infer
            env.m.encoder.generator = env.rng.encode.valid
            env.m.eval()
            env.trainer.eval()
        case "test":
            env.m.normalize(force=True, ignore_mode=True)
            env.m.disable_norm()
            env.m.disable_clamp()
            env.m.network.batchsz = env.cfg.batchsz.infer
            env.m.encoder.generator = env.rng.encode.test
            env.m.eval()
            env.trainer.eval()
        case _:
            raise RuntimeError(f"invalid dataset name '{ds}' received")


def step(env: SimpleNamespace, ds: str, epoch: int) -> int:
    def samples_per_epoch(setsz, batchsz):
        return setsz - (setsz % batchsz)

    match ds:
        case "train":
            return (
                epoch * samples_per_epoch(env.cfg.data.ntrain, env.cfg.batchsz.train)
                + env.prog.train
            )
        case "valid":
            return (
                epoch * samples_per_epoch(env.cfg.data.nvalid, env.cfg.batchsz.infer)
                + env.prog.valid
            )
        case "test":
            return (
                epoch * samples_per_epoch(env.cfg.data.ntest, env.cfg.batchsz.infer)
                + env.prog.test
            )
        case _:
            raise RuntimeError(f"invalid dataset name '{ds}' received")


def online_logdict(env: SimpleNamespace) -> dict:
    label = env.batchlog.history.label.unsafe_cat()
    rate = env.batchlog.history.rate.unsafe_cat()
    ttfs = env.batchlog.history.ttfs.unsafe_cat()
    return {
        "excrate": {
            "mean": env.batchlog.excspike.avg.unsafe_cat().mean().item(),
            "var": env.batchlog.excspike.var.unsafe_cat().mean().item(),
        },
        "inhrate": {
            "mean": env.batchlog.inhspike.avg.unsafe_cat().mean().item(),
            "var": env.batchlog.inhspike.var.unsafe_cat().mean().item(),
        },
        "weight": {
            "mean": env.batchlog.stats.weight.avg.unsafe_cat().mean().item(),
            "var": env.batchlog.stats.weight.var.unsafe_cat().mean().item(),
        },
        "delay": {
            "mean": env.batchlog.stats.delay.avg.unsafe_cat().mean().item(),
            "var": env.batchlog.stats.delay.var.unsafe_cat().mean().item(),
        },
        "acc": {
            "rate_nonp": (env.m.mr_classifier.classify(rate, False) == label)
            .to(dtype=env.cfg.verse.dtype)
            .mean()
            .item(),
            "rate_prop": (env.m.mr_classifier.classify(rate, True) == label)
            .to(dtype=env.cfg.verse.dtype)
            .mean()
            .item(),
            "ttfs_nonp": (env.m.fs_classifier.classify(ttfs, False) == label)
            .to(dtype=env.cfg.verse.dtype)
            .mean()
            .item(),
            "ttfs_prop": (env.m.fs_classifier.classify(ttfs, True) == label)
            .to(dtype=env.cfg.verse.dtype)
            .mean()
            .item(),
        },
    }


def offline_logdict(env: SimpleNamespace, ds: str) -> dict:
    def mapcolors(dsname: str) -> str:
        match dsname:
            case "train":
                return "Blues"
            case "valid":
                return "Purples"
            case "test":
                return "Reds"
            case _:
                raise RuntimeError(f"invalid `dsname` {dsname} received.")

    def computecm(ytrue: torch.Tensor, ypred: torch.Tensor):
        cm = torch.mm(F.one_hot(ytrue, 10).t(), F.one_hot(ypred, 10))
        return np.round((cm / cm.sum(1, keepdim=True)).nan_to_num(0).numpy() * 100, 1)

    def makecm(ytrue: torch.Tensor, ypred: torch.Tensor):
        cmfig, cmax = plt.subplots(figsize=(10, 10), dpi=300)
        cdisp = ConfusionMatrixDisplay(confusion_matrix=computecm(ytrue, ypred))
        cdisp = cdisp.plot(
            cmap=mapcolors(ds), values_format=".1f", ax=cmax, colorbar=False
        )
        cdisp.ax_.set(
            ylabel="True Label",
            xlabel="Predicted Label",
        )
        cdisp.figure_.colorbar(cdisp.im_, ax=cdisp.ax_, fraction=0.045725, pad=0.04)
        plt.tight_layout()
        plt.close(cmfig)
        return cmfig

    label = env.epochlog.history.label.unsafe_cat().to(device=env.cfg.verse.device)
    rate = env.epochlog.history.rate.unsafe_cat().to(device=env.cfg.verse.device)
    ttfs = env.epochlog.history.ttfs.unsafe_cat().to(device=env.cfg.verse.device)

    pred_rate_nonp = env.m.mr_classifier.classify(rate, False)
    pred_rate_prop = env.m.mr_classifier.classify(rate, True)
    pred_ttfs_nonp = env.m.fs_classifier.classify(ttfs, False)
    pred_ttfs_prop = env.m.fs_classifier.classify(ttfs, True)

    rate_nonp = (pred_rate_nonp == label).to(dtype=env.cfg.verse.dtype)
    rate_prop = (pred_rate_prop == label).to(dtype=env.cfg.verse.dtype)
    ttfs_nonp = (pred_ttfs_nonp == label).to(dtype=env.cfg.verse.dtype)
    ttfs_prop = (pred_ttfs_prop == label).to(dtype=env.cfg.verse.dtype)

    d = {
        "acc": {
            "rate_nonp": rate_nonp.mean().item(),
            "rate_prop": rate_prop.mean().item(),
            "ttfs_nonp": ttfs_nonp.mean().item(),
            "ttfs_prop": ttfs_prop.mean().item(),
        },
        "clsacc": {},
    }

    for k in range(10):
        d["clsacc"][f"c{k}"] = {
            "rate_nonp": rate_nonp[label == k].mean().item(),
            "rate_prop": rate_prop[label == k].mean().item(),
            "ttfs_nonp": ttfs_nonp[label == k].mean().item(),
            "ttfs_prop": ttfs_prop[label == k].mean().item(),
        }

    label = label.cpu()
    pred_rate_nonp = pred_rate_nonp.cpu()
    pred_rate_prop = pred_rate_prop.cpu()
    pred_ttfs_nonp = pred_ttfs_nonp.cpu()
    pred_ttfs_prop = pred_ttfs_prop.cpu()

    d["cm"] = {
        "rate_nonp": makecm(label, pred_rate_nonp),
        "rate_prop": makecm(label, pred_rate_prop),
        "ttfs_nonp": makecm(label, pred_ttfs_nonp),
        "ttfs_prop": makecm(label, pred_ttfs_prop),
    }

    return d


def save_nonpersistent_chkpt(env: SimpleNamespace, savepath: str) -> None:
    env.prog.since_n_chkpt = 0

    if os.path.exists(os.path.join(savepath, "latest.pt")):
        os.rename(
            os.path.join(savepath, "latest.pt"),
            os.path.join(savepath, "latest.archive.pt"),
        )
        dataexport(env, os.path.join(savepath, "latest.pt"))
        os.remove(os.path.join(savepath, "latest.archive.pt"))
    else:
        dataexport(env, os.path.join(savepath, "latest.pt"))


def run(
    config: SimpleNamespace,
    resume_point: str | None,
    data_path: str,
    chkpt_path: str,
    wandb_path: str,
) -> None:
    # create environment and restore from checkpoint
    env = build_environ(config)
    if resume_point:
        dataimport(env, os.path.join(chkpt_path, resume_point))

    # helper properties (formerly functions)
    iscpu = env.cfg.verse.device.partition(":")[0].lower() == "cpu"
    iscuda = env.cfg.verse.device.partition(":")[0].lower() == "cuda"
    # ismps = env.cfg.verse.device.partition(":")[0].lower() == "mps"

    # create datasets
    train_set, valid_set, test_set = common.mnist(
        env.cfg.split, env.cfg.verse.dtype, data_path
    )

    # enable wandb if specified
    if env.cfg.wandb.enable:
        wr = wandb.init(
            dir=wandb_path,
            config=env.cfg.wandb.config,
            resume=bool(resume_point),
            id=env.prog.runid,
            **env.cfg.wandb.options,
        )
        env.prog.runid = wr.id
        env.prog.chkpt = wr.name

        # step metrics
        wandb.define_metric("sample")
        wandb.define_metric("epoch")

        # online stats
        wandb.define_metric("online.excrate.mean", step_metric="sample")
        wandb.define_metric("online.excrate.var", step_metric="sample")
        wandb.define_metric("online.inhrate.mean", step_metric="sample")
        wandb.define_metric("online.inhrate.var", step_metric="sample")
        wandb.define_metric("online.weight.mean", step_metric="sample")
        wandb.define_metric("online.weight.var", step_metric="sample")
        wandb.define_metric("online.delay.mean", step_metric="sample")
        wandb.define_metric("online.delay.var", step_metric="sample")
        wandb.define_metric("online.acc.rate_nonp", step_metric="sample")
        wandb.define_metric("online.acc.rate_prop", step_metric="sample")
        wandb.define_metric("online.acc.ttfs_nonp", step_metric="sample")
        wandb.define_metric("online.acc.ttfs_prop", step_metric="sample")

        # groups
        dst_group = ["train", "valid", "test"]
        cls_group = ["acc"] + [f"clsacc.c{k}" for k in range(10)]
        acc_group = ["rate_nonp", "rate_prop", "ttfs_nonp", "ttfs_prop"]

        # accuracies
        for d_ in dst_group:
            for c_ in cls_group:
                for a_ in acc_group:
                    wandb.define_metric(f"{d_}.{c_}.{a_}", step_metric="epoch")

        # confusion matrices
        for d_ in dst_group:
            for a_ in acc_group:
                wandb.define_metric(f"{d_}.cm.{a_}", step_metric="epoch")

        # model checkpoints
        artifact = wandb.Artifact("checkpoints", type="model")
    else:
        env.prog.runid = None
        env.prog.chkpt = env.cfg.env.name

    # create checkpoint subdirectory if required
    savepath = os.path.join(chkpt_path, env.prog.chkpt)
    if not os.path.exists(savepath):
        os.makedirs(savepath)

    # iterate through epochs
    with torch.no_grad():
        for epoch in tqdm(
            range(env.prog.epoch, env.cfg.iters.epochs),
            desc="Epochs",
            initial=env.prog.epoch,
            total=env.cfg.iters.epochs,
            position=0,
            ncols=90,
        ):

            # check if training needs to be entered
            if env.prog.train < env.cfg.data.ntrain:
                # reset state for new epoch
                epochreset(env, "train")

                # create training sampler and iterator
                sampler = RandomSampler(
                    train_set,
                    replacement=False,
                    generator=env.rng.sample.train,
                )
                sample_iter = iter(sampler)

                # advance through sampler to get to resume point
                for _ in range(env.prog.train):
                    next(sample_iter)

                # construct corresponding dataloader
                loader = DataLoader(
                    train_set,
                    env.cfg.batchsz.train,
                    sampler=sample_iter,
                    drop_last=True,
                    pin_memory=iscuda,
                    pin_memory_device="" if not iscuda else env.cfg.verse.device,
                )

                # do training
                for X, y in tqdm(
                    loader,
                    desc="Train",
                    initial=(env.prog.train // env.cfg.batchsz.train),
                    total=(env.cfg.data.ntrain // env.cfg.batchsz.train),
                    leave=False,
                    position=1,
                    ncols=90,
                ):
                    # move to device
                    if not iscpu:
                        X = X.to(device=env.cfg.verse.device)
                        y = y.to(device=env.cfg.verse.device)

                    # increment progress
                    env.prog.train += y.numel()
                    env.prog.sincelog += y.numel()
                    env.prog.sincechkpt += y.numel()

                    # compute timings
                    isfinal = (
                        env.cfg.data.ntrain - env.prog.train < env.cfg.batchsz.train
                    )
                    islog = env.prog.sincelog >= env.cfg.iters.log_interval
                    ischkpt = env.prog.sincechkpt >= env.cfg.iters.checkpt_interval

                    # encode data via interval poisson
                    encoded = env.m.encoder(X)

                    # perform inference and training on samples
                    exc, inh = env.m.network(encoded, [env.trainer], env.batchlog)
                    clear(env)

                    # get stats and update classifier
                    rate = exc.to(dtype=env.cfg.verse.dtype).mean(dim=0)
                    ttfs = utils.ttfs(
                        exc.to(dtype=env.cfg.verse.dtype),
                        env.cfg.verse.dt,
                        env.cfg.verse.dtype,
                    )

                    env.m.mr_classifier.update(rate, y)
                    env.m.fs_classifier.update(ttfs, y, from_ttfs=True)

                    # locally log results
                    excvar, excavg = torch.var_mean(rate)
                    inhvar, inhavg = torch.var_mean(
                        inh.to(dtype=env.cfg.verse.dtype).mean(dim=0)
                    )
                    env.batchlog(
                        None,
                        excavg=excavg.unsqueeze(0),
                        excvar=excvar.unsqueeze(0),
                        inhavg=inhavg.unsqueeze(0),
                        inhvar=inhvar.unsqueeze(0),
                        label=y,
                        rate=rate,
                        ttfs=ttfs,
                    )
                    env.epochlog(
                        "cpu",
                        label=y,
                        rate=rate,
                        ttfs=ttfs,
                    )

                    # perform remote logging
                    if isfinal or islog:
                        # reset log timing
                        env.prog.sincelog = 0

                        # collate and log
                        if env.cfg.wandb.enable:
                            wr.log({"sample": step(env, "train", epoch)}, commit=True)
                            wr.log(data={"online": online_logdict(env)}, commit=True)

                        # clear buffers
                        env.batchlog = dcshared.Log()

                    # perform checkpointing
                    if ischkpt and not isfinal:
                        save_nonpersistent_chkpt(env, savepath)

                # epoch level log
                if env.cfg.wandb.enable:
                    wr.log({"epoch": epoch + 1}, commit=True)
                    wr.log(data={"train": offline_logdict(env, "train")}, commit=True)

                # clear buffers
                env.epochlog = dcshared.Log()

                # set other properties for switch
                env.prog.train = env.cfg.data.ntrain
                env.prog.sincelog = 0
                env.prog.sincechkpt = 0

                # perform checkpointing
                dataexport(env, os.path.join(savepath, f"training.e{epoch + 1}.pt"))
                save_nonpersistent_chkpt(env, savepath)

            # check if validation needs to be entered
            if env.prog.valid < env.cfg.data.nvalid:
                # reset state for new epoch
                epochreset(env, "valid")

                # create training sampler and iterator
                sampler = RandomSampler(
                    valid_set,
                    replacement=False,
                    generator=env.rng.sample.valid,
                )
                sample_iter = iter(sampler)

                # advance through sampler to get to resume point
                for _ in range(env.prog.valid):
                    next(sample_iter)

                # construct corresponding dataloader
                loader = DataLoader(
                    valid_set,
                    env.cfg.batchsz.infer,
                    sampler=sample_iter,
                    drop_last=True,
                    pin_memory=iscuda,
                    pin_memory_device="" if not iscuda else env.cfg.verse.device,
                )

                # do training
                for X, y in tqdm(
                    loader,
                    desc="Validate",
                    initial=(env.prog.valid // env.cfg.batchsz.infer),
                    total=(env.cfg.data.nvalid // env.cfg.batchsz.infer),
                    leave=False,
                    position=1,
                    ncols=90,
                ):
                    # move to device
                    if not iscpu:
                        X = X.to(device=env.cfg.verse.device)
                        y = y.to(device=env.cfg.verse.device)

                    # increment progress
                    env.prog.valid += y.numel()
                    env.prog.sincelog += y.numel()
                    env.prog.sincechkpt += y.numel()

                    # compute timings
                    isfinal = (
                        env.cfg.data.nvalid - env.prog.valid < env.cfg.batchsz.infer
                    )
                    ischkpt = env.prog.sincechkpt >= env.cfg.iters.checkpt_interval

                    # encode data via interval poisson
                    encoded = env.m.encoder(X)

                    # perform inference on samples
                    exc, _ = env.m.network(encoded, None, env.batchlog)
                    clear(env)

                    # get stats and update classifier
                    rate = exc.to(dtype=env.cfg.verse.dtype).mean(dim=0)
                    ttfs = utils.ttfs(
                        exc.to(dtype=env.cfg.verse.dtype),
                        env.cfg.verse.dt,
                        env.cfg.verse.dtype,
                    )

                    # locally log results
                    env.epochlog(
                        "cpu",
                        label=y,
                        rate=rate,
                        ttfs=ttfs,
                    )

                    # perform checkpointing
                    if ischkpt and not isfinal:
                        save_nonpersistent_chkpt(env, savepath)

                # epoch level log
                if env.cfg.wandb.enable:
                    wr.log(data={"valid": offline_logdict(env, "valid")}, commit=True)

                # clear buffers
                env.epochlog = dcshared.Log()

                # set other properties for switch
                env.prog.valid = env.cfg.data.nvalid
                env.prog.sincelog = 0
                env.prog.sincechkpt = 0

                # perform checkpointing
                dataexport(env, os.path.join(savepath, f"validation.e{epoch + 1}.pt"))
                save_nonpersistent_chkpt(env, savepath)

            # check if testing needs to be entered
            if env.prog.test < env.cfg.data.ntest:
                # reset state for new epoch
                epochreset(env, "test")

                # create training sampler and iterator
                sampler = RandomSampler(
                    test_set,
                    replacement=False,
                    generator=env.rng.sample.test,
                )
                sample_iter = iter(sampler)

                # advance through sampler to get to resume point
                for _ in range(env.prog.test):
                    next(sample_iter)

                # construct corresponding dataloader
                loader = DataLoader(
                    test_set,
                    env.cfg.batchsz.infer,
                    sampler=sample_iter,
                    drop_last=True,
                    pin_memory=iscuda,
                    pin_memory_device="" if not iscuda else env.cfg.verse.device,
                )

                # do training
                for X, y in tqdm(
                    loader,
                    desc="Test",
                    initial=(env.prog.test // env.cfg.batchsz.infer),
                    total=(env.cfg.data.ntest // env.cfg.batchsz.infer),
                    leave=False,
                    position=1,
                    ncols=90,
                ):
                    # move to device
                    if not iscpu:
                        X = X.to(device=env.cfg.verse.device)
                        y = y.to(device=env.cfg.verse.device)

                    # increment progress
                    env.prog.test += y.numel()
                    env.prog.sincelog += y.numel()
                    env.prog.sincechkpt += y.numel()

                    # compute timings
                    isfinal = env.cfg.data.ntest - env.prog.test < env.cfg.batchsz.infer
                    ischkpt = env.prog.sincechkpt >= env.cfg.iters.checkpt_interval

                    # encode data via interval poisson
                    encoded = env.m.encoder(X)

                    # perform inference on samples
                    exc, _ = env.m.network(encoded, None, env.batchlog)
                    clear(env)

                    # get stats and update classifier
                    rate = exc.to(dtype=env.cfg.verse.dtype).mean(dim=0)
                    ttfs = utils.ttfs(
                        exc.to(dtype=env.cfg.verse.dtype),
                        env.cfg.verse.dt,
                        env.cfg.verse.dtype,
                    )

                    # locally log results
                    env.epochlog(
                        "cpu",
                        label=y,
                        rate=rate,
                        ttfs=ttfs,
                    )

                    # perform checkpointing
                    if ischkpt and not isfinal:
                        save_nonpersistent_chkpt(env, savepath)

                # epoch level log
                if env.cfg.wandb.enable:
                    wr.log(data={"test": offline_logdict(env, "test")}, commit=True)

                # clear buffers
                env.epochlog = dcshared.Log()

                # set other properties for switch
                env.prog.test = env.cfg.data.ntest
                env.prog.sincelog = 0
                env.prog.sincechkpt = 0

                # perform checkpointing
                save_nonpersistent_chkpt(env, savepath)

            # epoch teardown
            env.prog.epoch = epoch + 1
            env.prog.train = 0
            env.prog.valid = 0
            env.prog.test = 0
            env.prog.sincelog = 0
            env.prog.sincechkpt = 0

            dataexport(env, os.path.join(savepath, f"epochfinal.{epoch + 1}.pt"))

            # log checkpoint as an artifact
            if env.cfg.wandb.enable:
                artifact = wandb.Artifact(f"checkpoint-e{epoch + 1}", type="model")
                artifact.add_file(os.path.join(savepath, f"epochfinal.{epoch + 1}.pt"))
                wandb.log_artifact(artifact)

    # [optional] finish the wandb run, necessary in notebooks
    if env.cfg.wandb.enable:
        wandb.finish()
