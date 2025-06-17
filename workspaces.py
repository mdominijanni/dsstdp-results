import wandb_workspaces.workspaces as ws
import wandb_workspaces.reports.v2 as wr


def create_aggregate_view(entity: str, project: str, name: str = "Aggregate") -> str:
    r"""Creates a WandB workspace with a basic panel layout for this project and report aggregate fields.

    Args:
        entity (str): wandb entity (team or user) name.
        project (str): wandb project name.
        name (str, optional): name of the workspace view. Defaults to "Default".

    Returns:
        str: url for the new workspace view.
    """
    workspace = ws.Workspace(
        entity=entity,
        project=project,
        name=name,
        sections=[
            ws.Section(
                name="Aggregates",
                panels=[
                    wr.LinePlot(
                        layout=wr.Layout(x=0, y=0, w=8, h=6),
                        title="Accuracy Metrics",
                        x=wr.Metric(name="epoch"),
                        y=[
                            wr.Metric(name=f"{d}.clsacc.c{k}.{m}_prop")
                            for k in range(10)
                            for m in ("rate", "ttfs")
                            for d in ("train", "valid", "test")
                        ]
                        + [
                            wr.Metric(name=f"{d}.acc.{m}_prop")
                            for m in ("rate", "ttfs")
                            for d in ("train", "valid", "test")
                        ],
                    ),
                    wr.LinePlot(
                        layout=wr.Layout(x=0, y=0, w=8, h=6),
                        title="Training Metrics",
                        x=wr.Metric(name="sample"),
                        y=[
                            wr.Metric(name="online.weight.mean"),
                            wr.Metric(name="online.weight.var"),
                            wr.Metric(name="online.delay.mean"),
                            wr.Metric(name="online.delay.var"),
                        ],
                    ),
                ],
                layout_settings=ws.SectionLayoutSettings(
                    layout="standard", columns=2, rows=2
                ),
                is_open=True,
            ),
            ws.Section(
                name="Rate Confusion Matrices",
                panels=[
                    wr.MediaBrowser(
                        layout=wr.Layout(x=0, y=0, w=8, h=6),
                        media_keys=[f"{dset}.cm.rate_{mode}"],
                    )
                    for mode in ("prop", "nonp")
                    for dset in ("train", "valid", "test")
                ],
                layout_settings=ws.SectionLayoutSettings(
                    layout="standard", columns=3, rows=2
                ),
                is_open=True,
            ),
            ws.Section(
                name="TTFS Confusion Matrices",
                panels=[
                    wr.MediaBrowser(
                        layout=wr.Layout(x=0, y=0, w=8, h=6),
                        media_keys=[f"{dset}.cm.ttfs_{mode}"],
                    )
                    for mode in ("prop", "nonp")
                    for dset in ("train", "valid", "test")
                ],
                layout_settings=ws.SectionLayoutSettings(
                    layout="standard", columns=3, rows=2
                ),
                is_open=True,
            ),
            ws.Section(
                name="Accuracy",
                panels=[
                    wr.LinePlot(
                        layout=wr.Layout(x=0, y=0, w=8, h=6),
                        title=f"{dsetk} {modek} Accuracy",
                        x=wr.Metric(name="epoch"),
                        y=[
                            wr.Metric(name=f"{dsetv}.acc.{modev}_nonp"),
                            wr.Metric(name=f"{dsetv}.acc.{modev}_prop"),
                        ],
                    )
                    for modek, modev in {
                        "Rate": "rate",
                        "TTFS": "ttfs",
                    }.items()
                    for dsetk, dsetv in {
                        "Training": "train",
                        "Validation": "valid",
                        "Testing": "test",
                    }.items()
                ],
                layout_settings=ws.SectionLayoutSettings(
                    layout="standard", columns=3, rows=2
                ),
                is_open=True,
            ),
            ws.Section(
                name="Rate Class Accuracy",
                panels=[
                    wr.LinePlot(
                        layout=wr.Layout(x=0, y=0, w=8, h=6),
                        title=f"{dsetk} Rate {modek} Class Accuracy",
                        x=wr.Metric(name="epoch"),
                        y=[
                            wr.Metric(name=f"{dsetv}.clsacc.c{k}.rate_{modev}")
                            for k in range(10)
                        ],
                    )
                    for modek, modev in {
                        "Proportional": "prop",
                        "Non-Proportional": "nonp",
                    }.items()
                    for dsetk, dsetv in {
                        "Training": "train",
                        "Validation": "valid",
                        "Testing": "test",
                    }.items()
                ],
                layout_settings=ws.SectionLayoutSettings(
                    layout="standard", columns=3, rows=2
                ),
                is_open=True,
            ),
            ws.Section(
                name="TTFS Class Accuracy",
                panels=[
                    wr.LinePlot(
                        layout=wr.Layout(x=0, y=0, w=8, h=6),
                        title=f"{dsetk} TTFS {modek} Class Accuracy",
                        x=wr.Metric(name="epoch"),
                        y=[
                            wr.Metric(name=f"{dsetv}.clsacc.c{k}.ttfs_{modev}")
                            for k in range(10)
                        ],
                    )
                    for modek, modev in {
                        "Proportional": "prop",
                        "Non-Proportional": "nonp",
                    }.items()
                    for dsetk, dsetv in {
                        "Training": "train",
                        "Validation": "valid",
                        "Testing": "test",
                    }.items()
                ],
                layout_settings=ws.SectionLayoutSettings(
                    layout="standard", columns=3, rows=2
                ),
                is_open=True,
            ),
            ws.Section(
                name="Online Stats",
                panels=[
                    wr.LinePlot(
                        layout=wr.Layout(x=0, y=0, w=8, h=6),
                        title="Online Rate Accuracy",
                        x=wr.Metric(name="sample"),
                        y=[
                            wr.Metric(name="online.acc.rate_prop"),
                            wr.Metric(name="online.acc.rate_nonp"),
                        ],
                        smoothing_factor=20.0,
                        smoothing_type="gaussian",
                    ),
                    wr.LinePlot(
                        layout=wr.Layout(x=0, y=0, w=8, h=6),
                        title="Weights (Mean)",
                        x=wr.Metric(name="sample"),
                        y=[wr.Metric(name="online.weight.mean")],
                        legend_fields=["run:displayName"],
                    ),
                    wr.LinePlot(
                        layout=wr.Layout(x=0, y=0, w=8, h=6),
                        title="Weights (Variance)",
                        x=wr.Metric(name="sample"),
                        y=[wr.Metric(name="online.weight.var")],
                        legend_fields=["run:displayName"],
                    ),
                    wr.LinePlot(
                        layout=wr.Layout(x=0, y=0, w=8, h=6),
                        title="Online TTFS Accuracy",
                        x=wr.Metric(name="sample"),
                        y=[
                            wr.Metric(name="online.acc.ttfs_prop"),
                            wr.Metric(name="online.acc.ttfs_nonp"),
                        ],
                        smoothing_factor=20.0,
                        smoothing_type="gaussian",
                    ),
                    wr.LinePlot(
                        layout=wr.Layout(x=0, y=0, w=8, h=6),
                        title="Delays (Mean)",
                        x=wr.Metric(name="sample"),
                        y=[wr.Metric(name="online.delay.mean")],
                        legend_fields=["run:displayName"],
                    ),
                    wr.LinePlot(
                        layout=wr.Layout(x=0, y=0, w=8, h=6),
                        title="Delays (Variance)",
                        x=wr.Metric(name="sample"),
                        y=[wr.Metric(name="online.delay.var")],
                        legend_fields=["run:displayName"],
                    ),
                    wr.LinePlot(
                        layout=wr.Layout(x=0, y=0, w=8, h=6),
                        title="Excitatory Rate (Mean)",
                        x=wr.Metric(name="sample"),
                        y=[wr.Metric(name="online.excrate.mean")],
                        legend_fields=["run:displayName"],
                    ),
                    wr.LinePlot(
                        layout=wr.Layout(x=0, y=0, w=8, h=6),
                        title="Excitatory Rate (Variance)",
                        x=wr.Metric(name="sample"),
                        y=[wr.Metric(name="online.excrate.var")],
                        legend_fields=["run:displayName"],
                    ),
                    wr.LinePlot(
                        layout=wr.Layout(x=0, y=0, w=8, h=6),
                        title="Inhibitory Rate (Mean)",
                        x=wr.Metric(name="sample"),
                        y=[wr.Metric(name="online.inhrate.mean")],
                        legend_fields=["run:displayName"],
                    ),
                    wr.LinePlot(
                        layout=wr.Layout(x=0, y=0, w=8, h=6),
                        title="Inhibitory Rate (Variance)",
                        x=wr.Metric(name="sample"),
                        y=[wr.Metric(name="online.inhrate.var")],
                        legend_fields=["run:displayName"],
                    ),
                    wr.LinePlot(
                        layout=wr.Layout(x=0, y=0, w=8, h=6),
                        title="Execution Time",
                        x=wr.Metric(name="sample"),
                        y=[wr.Metric(name="RelativeTime(Process)")],
                        legend_fields=["run:displayName"],
                    ),
                ],
                layout_settings=ws.SectionLayoutSettings(
                    layout="standard", columns=3, rows=2
                ),
                is_open=True,
            ),
        ],
    )
    workspace.save()
    return workspace.url


def create_default_view(entity: str, project: str, name: str = "Default") -> str:
    r"""Creates a WandB workspace with a basic panel layout for this project.

    Args:
        entity (str): wandb entity (team or user) name.
        project (str): wandb project name.
        name (str, optional): name of the workspace view. Defaults to "Default".

    Returns:
        str: url for the new workspace view.
    """
    workspace = ws.Workspace(
        entity=entity,
        project=project,
        name=name,
        sections=[
            ws.Section(
                name="Rate Confusion Matrices",
                panels=[
                    wr.MediaBrowser(
                        layout=wr.Layout(x=0, y=0, w=8, h=6),
                        media_keys=[f"{dset}.cm.rate_{mode}"],
                    )
                    for mode in ("prop", "nonp")
                    for dset in ("train", "valid", "test")
                ],
                layout_settings=ws.SectionLayoutSettings(
                    layout="standard", columns=3, rows=2
                ),
                is_open=True,
            ),
            ws.Section(
                name="TTFS Confusion Matrices",
                panels=[
                    wr.MediaBrowser(
                        layout=wr.Layout(x=0, y=0, w=8, h=6),
                        media_keys=[f"{dset}.cm.ttfs_{mode}"],
                    )
                    for mode in ("prop", "nonp")
                    for dset in ("train", "valid", "test")
                ],
                layout_settings=ws.SectionLayoutSettings(
                    layout="standard", columns=3, rows=2
                ),
                is_open=True,
            ),
            ws.Section(
                name="Accuracy",
                panels=[
                    wr.LinePlot(
                        layout=wr.Layout(x=0, y=0, w=8, h=6),
                        title=f"{dsetk} {modek} Accuracy",
                        x=wr.Metric(name="epoch"),
                        y=[
                            wr.Metric(name=f"{dsetv}.acc.{modev}_nonp"),
                            wr.Metric(name=f"{dsetv}.acc.{modev}_prop"),
                        ],
                    )
                    for modek, modev in {
                        "Rate": "rate",
                        "TTFS": "ttfs",
                    }.items()
                    for dsetk, dsetv in {
                        "Training": "train",
                        "Validation": "valid",
                        "Testing": "test",
                    }.items()
                ],
                layout_settings=ws.SectionLayoutSettings(
                    layout="standard", columns=3, rows=2
                ),
                is_open=True,
            ),
            ws.Section(
                name="Rate Class Accuracy",
                panels=[
                    wr.LinePlot(
                        layout=wr.Layout(x=0, y=0, w=8, h=6),
                        title=f"{dsetk} Rate {modek} Class Accuracy",
                        x=wr.Metric(name="epoch"),
                        y=[
                            wr.Metric(name=f"{dsetv}.clsacc.c{k}.rate_{modev}")
                            for k in range(10)
                        ],
                    )
                    for modek, modev in {
                        "Proportional": "prop",
                        "Non-Proportional": "nonp",
                    }.items()
                    for dsetk, dsetv in {
                        "Training": "train",
                        "Validation": "valid",
                        "Testing": "test",
                    }.items()
                ],
                layout_settings=ws.SectionLayoutSettings(
                    layout="standard", columns=3, rows=2
                ),
                is_open=True,
            ),
            ws.Section(
                name="TTFS Class Accuracy",
                panels=[
                    wr.LinePlot(
                        layout=wr.Layout(x=0, y=0, w=8, h=6),
                        title=f"{dsetk} TTFS {modek} Class Accuracy",
                        x=wr.Metric(name="epoch"),
                        y=[
                            wr.Metric(name=f"{dsetv}.clsacc.c{k}.ttfs_{modev}")
                            for k in range(10)
                        ],
                    )
                    for modek, modev in {
                        "Proportional": "prop",
                        "Non-Proportional": "nonp",
                    }.items()
                    for dsetk, dsetv in {
                        "Training": "train",
                        "Validation": "valid",
                        "Testing": "test",
                    }.items()
                ],
                layout_settings=ws.SectionLayoutSettings(
                    layout="standard", columns=3, rows=2
                ),
                is_open=True,
            ),
            ws.Section(
                name="Online Stats",
                panels=[
                    wr.LinePlot(
                        layout=wr.Layout(x=0, y=0, w=8, h=6),
                        title="Online Rate Accuracy",
                        x=wr.Metric(name="sample"),
                        y=[
                            wr.Metric(name="online.acc.rate_prop"),
                            wr.Metric(name="online.acc.rate_nonp"),
                        ],
                        smoothing_factor=20.0,
                        smoothing_type="gaussian",
                    ),
                    wr.LinePlot(
                        layout=wr.Layout(x=0, y=0, w=8, h=6),
                        title="Weights (Mean)",
                        x=wr.Metric(name="sample"),
                        y=[wr.Metric(name="online.weight.mean")],
                        legend_fields=["run:displayName"],
                    ),
                    wr.LinePlot(
                        layout=wr.Layout(x=0, y=0, w=8, h=6),
                        title="Weights (Variance)",
                        x=wr.Metric(name="sample"),
                        y=[wr.Metric(name="online.weight.var")],
                        legend_fields=["run:displayName"],
                    ),
                    wr.LinePlot(
                        layout=wr.Layout(x=0, y=0, w=8, h=6),
                        title="Online TTFS Accuracy",
                        x=wr.Metric(name="sample"),
                        y=[
                            wr.Metric(name="online.acc.ttfs_prop"),
                            wr.Metric(name="online.acc.ttfs_nonp"),
                        ],
                        smoothing_factor=20.0,
                        smoothing_type="gaussian",
                    ),
                    wr.LinePlot(
                        layout=wr.Layout(x=0, y=0, w=8, h=6),
                        title="Delays (Mean)",
                        x=wr.Metric(name="sample"),
                        y=[wr.Metric(name="online.delay.mean")],
                        legend_fields=["run:displayName"],
                    ),
                    wr.LinePlot(
                        layout=wr.Layout(x=0, y=0, w=8, h=6),
                        title="Delays (Variance)",
                        x=wr.Metric(name="sample"),
                        y=[wr.Metric(name="online.delay.var")],
                        legend_fields=["run:displayName"],
                    ),
                    wr.LinePlot(
                        layout=wr.Layout(x=0, y=0, w=8, h=6),
                        title="Excitatory Rate (Mean)",
                        x=wr.Metric(name="sample"),
                        y=[wr.Metric(name="online.excrate.mean")],
                        legend_fields=["run:displayName"],
                    ),
                    wr.LinePlot(
                        layout=wr.Layout(x=0, y=0, w=8, h=6),
                        title="Excitatory Rate (Variance)",
                        x=wr.Metric(name="sample"),
                        y=[wr.Metric(name="online.excrate.var")],
                        legend_fields=["run:displayName"],
                    ),
                    wr.LinePlot(
                        layout=wr.Layout(x=0, y=0, w=8, h=6),
                        title="Inhibitory Rate (Mean)",
                        x=wr.Metric(name="sample"),
                        y=[wr.Metric(name="online.inhrate.mean")],
                        legend_fields=["run:displayName"],
                    ),
                    wr.LinePlot(
                        layout=wr.Layout(x=0, y=0, w=8, h=6),
                        title="Inhibitory Rate (Variance)",
                        x=wr.Metric(name="sample"),
                        y=[wr.Metric(name="online.inhrate.var")],
                        legend_fields=["run:displayName"],
                    ),
                    wr.LinePlot(
                        layout=wr.Layout(x=0, y=0, w=8, h=6),
                        title="Execution Time",
                        x=wr.Metric(name="sample"),
                        y=[wr.Metric(name="RelativeTime(Process)")],
                        legend_fields=["run:displayName"],
                    ),
                ],
                layout_settings=ws.SectionLayoutSettings(
                    layout="standard", columns=3, rows=2
                ),
                is_open=True,
            ),
        ],
    )
    workspace.save()
    return workspace.url


def create_propacc_view(entity: str, project: str, name: str = "Default") -> str:
    r"""Creates a WandB workspace with a basic panel layout,
    and a dedicated proportional accuracy section, for this project.

    Args:
        entity (str): wandb entity (team or user) name.
        project (str): wandb project name.
        name (str, optional): name of the workspace view. Defaults to "Default".

    Returns:
        str: url for the new workspace view.
    """
    workspace = ws.Workspace(
        entity=entity,
        project=project,
        name=name,
        sections=[
            ws.Section(
                name="Rate Confusion Matrices",
                panels=[
                    wr.MediaBrowser(
                        layout=wr.Layout(x=0, y=0, w=8, h=6),
                        media_keys=[f"{dset}.cm.rate_{mode}"],
                    )
                    for mode in ("prop", "nonp")
                    for dset in ("train", "valid", "test")
                ],
                layout_settings=ws.SectionLayoutSettings(
                    layout="standard", columns=3, rows=2
                ),
                is_open=True,
            ),
            ws.Section(
                name="TTFS Confusion Matrices",
                panels=[
                    wr.MediaBrowser(
                        layout=wr.Layout(x=0, y=0, w=8, h=6),
                        media_keys=[f"{dset}.cm.ttfs_{mode}"],
                    )
                    for mode in ("prop", "nonp")
                    for dset in ("train", "valid", "test")
                ],
                layout_settings=ws.SectionLayoutSettings(
                    layout="standard", columns=3, rows=2
                ),
                is_open=True,
            ),
            ws.Section(
                name="Accuracy",
                panels=[
                    wr.LinePlot(
                        layout=wr.Layout(x=0, y=0, w=8, h=6),
                        title=f"{dsetk} {modek} Accuracy",
                        x=wr.Metric(name="epoch"),
                        y=[
                            wr.Metric(name=f"{dsetv}.acc.{modev}_nonp"),
                            wr.Metric(name=f"{dsetv}.acc.{modev}_prop"),
                        ],
                    )
                    for modek, modev in {
                        "Rate": "rate",
                        "TTFS": "ttfs",
                    }.items()
                    for dsetk, dsetv in {
                        "Training": "train",
                        "Validation": "valid",
                        "Testing": "test",
                    }.items()
                ],
                layout_settings=ws.SectionLayoutSettings(
                    layout="standard", columns=3, rows=2
                ),
                is_open=True,
            ),
            ws.Section(
                name="Accuracy (Proportional Method)",
                panels=[
                    wr.LinePlot(
                        layout=wr.Layout(x=0, y=0, w=8, h=6),
                        title=f"{dsetk} {modek} Accuracy",
                        x=wr.Metric(name="epoch"),
                        y=[
                            wr.Metric(name=f"{dsetv}.acc.{modev}_prop"),
                        ],
                    )
                    for modek, modev in {
                        "Rate": "rate",
                        "TTFS": "ttfs",
                    }.items()
                    for dsetk, dsetv in {
                        "Training": "train",
                        "Validation": "valid",
                        "Testing": "test",
                    }.items()
                ],
                layout_settings=ws.SectionLayoutSettings(
                    layout="standard", columns=3, rows=2
                ),
                is_open=True,
            ),
            ws.Section(
                name="Rate Class Accuracy",
                panels=[
                    wr.LinePlot(
                        layout=wr.Layout(x=0, y=0, w=8, h=6),
                        title=f"{dsetk} Rate {modek} Class Accuracy",
                        x=wr.Metric(name="epoch"),
                        y=[
                            wr.Metric(name=f"{dsetv}.clsacc.c{k}.rate_{modev}")
                            for k in range(10)
                        ],
                    )
                    for modek, modev in {
                        "Proportional": "prop",
                        "Non-Proportional": "nonp",
                    }.items()
                    for dsetk, dsetv in {
                        "Training": "train",
                        "Validation": "valid",
                        "Testing": "test",
                    }.items()
                ],
                layout_settings=ws.SectionLayoutSettings(
                    layout="standard", columns=3, rows=2
                ),
                is_open=True,
            ),
            ws.Section(
                name="TTFS Class Accuracy",
                panels=[
                    wr.LinePlot(
                        layout=wr.Layout(x=0, y=0, w=8, h=6),
                        title=f"{dsetk} TTFS {modek} Class Accuracy",
                        x=wr.Metric(name="epoch"),
                        y=[
                            wr.Metric(name=f"{dsetv}.clsacc.c{k}.ttfs_{modev}")
                            for k in range(10)
                        ],
                    )
                    for modek, modev in {
                        "Proportional": "prop",
                        "Non-Proportional": "nonp",
                    }.items()
                    for dsetk, dsetv in {
                        "Training": "train",
                        "Validation": "valid",
                        "Testing": "test",
                    }.items()
                ],
                layout_settings=ws.SectionLayoutSettings(
                    layout="standard", columns=3, rows=2
                ),
                is_open=True,
            ),
            ws.Section(
                name="Online Stats",
                panels=[
                    wr.LinePlot(
                        layout=wr.Layout(x=0, y=0, w=8, h=6),
                        title="Online Rate Accuracy",
                        x=wr.Metric(name="sample"),
                        y=[
                            wr.Metric(name="online.acc.rate_prop"),
                            wr.Metric(name="online.acc.rate_nonp"),
                        ],
                        smoothing_factor=20.0,
                        smoothing_type="gaussian",
                    ),
                    wr.LinePlot(
                        layout=wr.Layout(x=0, y=0, w=8, h=6),
                        title="Weights (Mean)",
                        x=wr.Metric(name="sample"),
                        y=[wr.Metric(name="online.weight.mean")],
                        legend_fields=["run:displayName"],
                    ),
                    wr.LinePlot(
                        layout=wr.Layout(x=0, y=0, w=8, h=6),
                        title="Weights (Variance)",
                        x=wr.Metric(name="sample"),
                        y=[wr.Metric(name="online.weight.var")],
                        legend_fields=["run:displayName"],
                    ),
                    wr.LinePlot(
                        layout=wr.Layout(x=0, y=0, w=8, h=6),
                        title="Online TTFS Accuracy",
                        x=wr.Metric(name="sample"),
                        y=[
                            wr.Metric(name="online.acc.ttfs_prop"),
                            wr.Metric(name="online.acc.ttfs_nonp"),
                        ],
                        smoothing_factor=20.0,
                        smoothing_type="gaussian",
                    ),
                    wr.LinePlot(
                        layout=wr.Layout(x=0, y=0, w=8, h=6),
                        title="Delays (Mean)",
                        x=wr.Metric(name="sample"),
                        y=[wr.Metric(name="online.delay.mean")],
                        legend_fields=["run:displayName"],
                    ),
                    wr.LinePlot(
                        layout=wr.Layout(x=0, y=0, w=8, h=6),
                        title="Delays (Variance)",
                        x=wr.Metric(name="sample"),
                        y=[wr.Metric(name="online.delay.var")],
                        legend_fields=["run:displayName"],
                    ),
                    wr.LinePlot(
                        layout=wr.Layout(x=0, y=0, w=8, h=6),
                        title="Excitatory Rate (Mean)",
                        x=wr.Metric(name="sample"),
                        y=[wr.Metric(name="online.excrate.mean")],
                        legend_fields=["run:displayName"],
                    ),
                    wr.LinePlot(
                        layout=wr.Layout(x=0, y=0, w=8, h=6),
                        title="Excitatory Rate (Variance)",
                        x=wr.Metric(name="sample"),
                        y=[wr.Metric(name="online.excrate.var")],
                        legend_fields=["run:displayName"],
                    ),
                    wr.LinePlot(
                        layout=wr.Layout(x=0, y=0, w=8, h=6),
                        title="Inhibitory Rate (Mean)",
                        x=wr.Metric(name="sample"),
                        y=[wr.Metric(name="online.inhrate.mean")],
                        legend_fields=["run:displayName"],
                    ),
                    wr.LinePlot(
                        layout=wr.Layout(x=0, y=0, w=8, h=6),
                        title="Inhibitory Rate (Variance)",
                        x=wr.Metric(name="sample"),
                        y=[wr.Metric(name="online.inhrate.var")],
                        legend_fields=["run:displayName"],
                    ),
                    wr.LinePlot(
                        layout=wr.Layout(x=0, y=0, w=8, h=6),
                        title="Execution Time",
                        x=wr.Metric(name="sample"),
                        y=[wr.Metric(name="RelativeTime(Process)")],
                        legend_fields=["run:displayName"],
                    ),
                ],
                layout_settings=ws.SectionLayoutSettings(
                    layout="standard", columns=3, rows=2
                ),
                is_open=True,
            ),
        ],
    )
    workspace.save()
    return workspace.url
