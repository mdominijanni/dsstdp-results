from . import dc, drdc, dsdc
import argparse
from datetime import datetime
import os
import shutil
from typing import Any


def toml_export(filename: str, config: list[str]) -> None:
    export, ext = os.path.splitext(filename)
    if not ext.lower() == ".toml":
        export = export + ".toml"
    else:
        export = filename

    with open(export, "w") as file:
        file.write("\n".join(config))


def runtime_overrides(**kwargs) -> dict[str, Any]:
    return dict(filter(lambda e: e[1] is not None, kwargs.items()))


def main():
    parser = argparse.ArgumentParser(description="Runs Inferno Examples")
    parser.add_argument(
        "example",
        type=str,
        help="example to run, valid options are: 'dc', 'drdc', 'dsdc'",
    )
    parser.add_argument(
        "config",
        type=str,
        nargs="?",
        help="path to the configuration file, required unless '--export-default'",
    )
    parser.add_argument(
        "-r",
        "--resume",
        dest="resume",
        type=str,
        required=False,
        help="name of a given checkpoint",
    )
    parser.add_argument(
        "--export-default",
        dest="export",
        type=str,
        required=False,
        help="exports the default configuration instead of running the example",
    )
    parser.add_argument(
        "-d",
        "--device",
        dest="device",
        type=str,
        required=False,
        help="overrides the compute device specified in the configuration",
    )
    parser.add_argument(
        "--enable-wandb",
        dest="use_wandb",
        action="store_true",
        required=False,
        help="overrides the configuration to enable WandB",
    )
    parser.add_argument(
        "--disable-wandb",
        dest="use_wandb",
        action="store_false",
        required=False,
        help="overrides the configuration to disable WandB",
    )
    parser.set_defaults(use_wandb=None)

    parser.add_argument(
        "--autodelete",
        dest="autodelete",
        action="store_true",
        required=False,
        help="if existing files in the directory should be automatically deleted",
    )

    args = parser.parse_args()

    # creates local file structure if needed
    basepath = os.path.join(".", "ddc_data")
    if not os.path.exists(basepath):
        os.makedirs(basepath)

    sharedpath = os.path.join(basepath, "_shared")
    if not os.path.exists(sharedpath):
        os.makedirs(sharedpath)
    if not os.path.exists(os.path.join(sharedpath, "data")):
        os.makedirs(os.path.join(sharedpath, "data"))

    # selects given example project
    match args.example.lower():

        # Diehl & Cook
        case "dc":

            # exports default configuration file
            if args.export:
                toml_export(args.export, dc.default_config())

            # config must normally be specified
            elif not args.config:
                raise RuntimeError(
                    "when not using '--export-default', 'config' must be specified"
                )

            else:
                # creates the config file
                ingest_kwargs = runtime_overrides(
                    device=args.device, use_wandb=args.use_wandb
                )
                with open(args.config) as file:
                    conflns = file.read().splitlines()
                config = dc.ingest_config(conflns, **ingest_kwargs)

                # creates local file structure if needed
                projpath = os.path.join(basepath, "dc")
                if not os.path.exists(projpath):
                    os.makedirs(projpath)

                runpath = os.path.join(projpath, config.env.name)
                if not os.path.exists(runpath):
                    os.makedirs(runpath)
                elif not args.resume:
                    if args.autodelete:
                        shutil.rmtree(runpath)
                        os.makedirs(runpath)
                    else:
                        print(
                            f"Directory '{runpath}' is not empty, delete contents? [y/N]"
                        )
                        match input().lower():
                            case "y" | "yes" | "ye":
                                shutil.rmtree(runpath)
                                os.makedirs(runpath)
                            case "n" | "no" | "":
                                print("Exiting without alteration...")
                                exit()
                            case _:
                                print("Exiting without alteration...")
                                exit()

                confpath = os.path.join(runpath, "conf")
                if not os.path.exists(confpath):
                    os.makedirs(confpath)
                chkptpath = os.path.join(runpath, "chkpt")
                if not os.path.exists(chkptpath):
                    os.makedirs(chkptpath)
                wandbpath = os.path.join(runpath, "wandb")
                if not os.path.exists(wandbpath):
                    os.makedirs(wandbpath)

                # archives the configuration file
                with open(
                    os.path.join(
                        confpath,
                        datetime.now().isoformat().replace(":", "-").partition(".")[0]
                        + ".toml",
                    ),
                    "w",
                ) as file:
                    file.write("\n".join(conflns))

                # runs the example
                dc.run(
                    config,
                    args.resume if args.resume else None,
                    os.path.join(sharedpath, "data"),
                    chkptpath,
                    wandbpath,
                )

        # Diehl & Cook with DR-STDP
        case "drdc":

            # exports default configuration file
            if args.export:
                toml_export(args.export, drdc.default_config())

            # config must normally be specified
            elif not args.config:
                raise RuntimeError(
                    "when not using '--export-default', 'config' must be specified"
                )

            else:
                # creates the config file
                ingest_kwargs = runtime_overrides(
                    device=args.device, use_wandb=args.use_wandb
                )
                with open(args.config) as file:
                    conflns = file.read().splitlines()
                config = drdc.ingest_config(conflns, **ingest_kwargs)

                # creates local file structure if needed
                projpath = os.path.join(basepath, "drdc")
                if not os.path.exists(projpath):
                    os.makedirs(projpath)

                runpath = os.path.join(projpath, config.env.name)
                if not os.path.exists(runpath):
                    os.makedirs(runpath)
                elif not args.resume:
                    if args.autodelete:
                        shutil.rmtree(runpath)
                        os.makedirs(runpath)
                    else:
                        print(
                            f"Directory '{runpath}' is not empty, delete contents? [y/N]"
                        )
                        match input().lower():
                            case "y" | "yes" | "ye":
                                shutil.rmtree(runpath)
                                os.makedirs(runpath)
                            case "n" | "no" | "":
                                print("Exiting without alteration...")
                                exit()
                            case _:
                                print("Exiting without alteration...")
                                exit()

                confpath = os.path.join(runpath, "conf")
                if not os.path.exists(confpath):
                    os.makedirs(confpath)
                chkptpath = os.path.join(runpath, "chkpt")
                if not os.path.exists(chkptpath):
                    os.makedirs(chkptpath)
                wandbpath = os.path.join(runpath, "wandb")
                if not os.path.exists(wandbpath):
                    os.makedirs(wandbpath)

                # archives the configuration file
                with open(
                    os.path.join(
                        confpath,
                        datetime.now().isoformat().replace(":", "-").partition(".")[0]
                        + ".toml",
                    ),
                    "w",
                ) as file:
                    file.write("\n".join(conflns))

                # runs the example
                drdc.run(
                    config,
                    args.resume if args.resume else None,
                    os.path.join(sharedpath, "data"),
                    chkptpath,
                    wandbpath,
                )

        # Diehl & Cook with DS-STDP
        case "dsdc":

            # exports default configuration file
            if args.export:
                toml_export(args.export, dsdc.default_config())

            # config must normally be specified
            elif not args.config:
                raise RuntimeError(
                    "when not using '--export-default', 'config' must be specified"
                )

            else:
                # creates the config file
                ingest_kwargs = runtime_overrides(
                    device=args.device, use_wandb=args.use_wandb
                )
                with open(args.config) as file:
                    conflns = file.read().splitlines()
                config = dsdc.ingest_config(conflns, **ingest_kwargs)

                # creates local file structure if needed
                projpath = os.path.join(basepath, "dsdc")
                if not os.path.exists(projpath):
                    os.makedirs(projpath)

                runpath = os.path.join(projpath, config.env.name)
                if not os.path.exists(runpath):
                    os.makedirs(runpath)
                elif not args.resume:
                    if args.autodelete:
                        shutil.rmtree(runpath)
                        os.makedirs(runpath)
                    else:
                        print(
                            f"Directory '{runpath}' is not empty, delete contents? [y/N]"
                        )
                        match input().lower():
                            case "y" | "yes" | "ye":
                                shutil.rmtree(runpath)
                                os.makedirs(runpath)
                            case "n" | "no" | "":
                                print("Exiting without alteration...")
                                exit()
                            case _:
                                print("Exiting without alteration...")
                                exit()

                confpath = os.path.join(runpath, "conf")
                if not os.path.exists(confpath):
                    os.makedirs(confpath)
                chkptpath = os.path.join(runpath, "chkpt")
                if not os.path.exists(chkptpath):
                    os.makedirs(chkptpath)
                wandbpath = os.path.join(runpath, "wandb")
                if not os.path.exists(wandbpath):
                    os.makedirs(wandbpath)

                # archives the configuration file
                with open(
                    os.path.join(
                        confpath,
                        datetime.now().isoformat().replace(":", "-").partition(".")[0]
                        + ".toml",
                    ),
                    "w",
                ) as file:
                    file.write("\n".join(conflns))

                # runs the example
                dsdc.run(
                    config,
                    args.resume if args.resume else None,
                    os.path.join(sharedpath, "data"),
                    chkptpath,
                    wandbpath,
                )

        # invalid example
        case _:
            raise RuntimeError(f"the example project '{args.example}' does not exist")


if __name__ == "__main__":
    main()
