"""Command line for the training pipeline: python -m training.run <command>.

    python -m training.run verify --package data/packages/B_qc-v5
    python -m training.run train --backbone swin_v2_t --fold 0 [--set key=value ...]

Exit codes: 0 done, 1 problems found, 2 setup or usage problem.
"""
import argparse
import sys
from datetime import datetime, timezone

import loader
from training.config import DEFAULT_TRAIN_CONFIG


def verify(package):
    """Rehash a package against its manifest and print any problems; return the exit code."""
    try:
        problems = loader.verify_package(package)
    except (OSError, ValueError, KeyError) as error:
        print(f"cannot verify {package}: {error}", file=sys.stderr)
        return 2
    for problem in problems:
        print(problem)
    print("ok" if not problems else f"{len(problems)} problem(s)")
    return 1 if problems else 0


def train(args):
    """Train one validation fold and print the run folder; return the exit code."""
    # pylint: disable=import-outside-toplevel
    from training import config, experiment
    try:
        settings, record = config.load_config(args.config, args.backbone, args.overrides)
        name = args.name or (f"{args.backbone}_fold{args.fold}_"
                             f"{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}")
        run_dir = config.repo_path(settings["output_dir"]) / name
        print(experiment.run_fold(settings, record, args.fold, run_dir, args.device))
    except (RuntimeError, ValueError, NotImplementedError, OSError) as error:
        print(f"cannot train: {error}", file=sys.stderr)
        return 2
    return 0


def main(argv=None):
    """Run the chosen command and return its exit code."""
    parser = argparse.ArgumentParser(prog="python -m training.run",
                                     description=__doc__.split("\n", maxsplit=1)[0])
    commands = parser.add_subparsers(dest="command", required=True)
    verify_parser = commands.add_parser("verify", help="rehash a package against its manifest")
    verify_parser.add_argument("--package", required=True, help="package or combined folder")
    train_parser = commands.add_parser("train", help="train the real-only condition on one fold")
    train_parser.add_argument("--backbone", required=True, choices=["swin_v2_t", "convnext_v2_t"])
    train_parser.add_argument("--fold", required=True, type=int, choices=range(5))
    train_parser.add_argument("--config", default=DEFAULT_TRAIN_CONFIG)
    train_parser.add_argument("--set", dest="overrides", action="append", default=[],
                              metavar="KEY=VALUE", help="override a config key (YAML value)")
    train_parser.add_argument("--device", default="cpu")
    train_parser.add_argument("--name", help="run folder name (default: backbone, fold, UTC time)")
    args = parser.parse_args(argv)
    if args.command == "train":
        return train(args)
    return verify(args.package)


if __name__ == "__main__":
    sys.exit(main())
