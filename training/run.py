"""Command line for the training pipeline: python -m training.run <command>.

    python -m training.run verify --package data/packages/B_qc-v5
    python -m training.run references
    python -m training.run train --backbone swin_v2_t --fold 0 [--set key=value ...]
    python -m training.run train --backbone swin_v2_t --fold final [--set key=value ...]

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


def fold_argument(text):
    """A fold from the command line: 0 to 4, or "final"."""
    if text == "final":
        return text
    if text in ("0", "1", "2", "3", "4"):
        return int(text)
    raise argparse.ArgumentTypeError(f"fold must be 0 to 4 or final: {text}")


def train(args):
    """Train one validation fold or the final fit and print the run folder; return the exit code."""
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


def references(args):
    """Print a provisional Macenko reference tile per fold as config lines; return the exit code."""
    # pylint: disable=import-outside-toplevel
    from training import config, experiment, references as chooser
    try:
        settings, _ = config.load_config(args.config, "swin_v2_t")
        package = config.repo_path(settings["package_dir"])
        with loader.TileReader(package) as reader:
            chosen = chooser.choose_references(experiment.read_tiles(settings), reader,
                                               settings["seed"])
    except (RuntimeError, ValueError, OSError) as error:
        print(f"cannot choose references: {error}", file=sys.stderr)
        return 2
    print("macenko_reference_patch_ids:")
    for fold, patch_id in chosen.items():
        print(f'  "{fold}": {patch_id}')
    return 0


def main(argv=None):
    """Run the chosen command and return its exit code."""
    parser = argparse.ArgumentParser(prog="python -m training.run",
                                     description=__doc__.split("\n", maxsplit=1)[0])
    commands = parser.add_subparsers(dest="command", required=True)
    verify_parser = commands.add_parser("verify", help="rehash a package against its manifest")
    verify_parser.add_argument("--package", required=True, help="package or combined folder")
    train_parser = commands.add_parser("train", help="train the real-only condition on a fold "
                                                     "or the final fit")
    train_parser.add_argument("--backbone", required=True, choices=["swin_v2_t", "convnext_v2_t"])
    train_parser.add_argument("--fold", required=True, type=fold_argument,
                              help="validation fold 0 to 4, or final (all TCGA, scored on CPTAC)")
    train_parser.add_argument("--config", default=DEFAULT_TRAIN_CONFIG)
    train_parser.add_argument("--set", dest="overrides", action="append", default=[],
                              metavar="KEY=VALUE", help="override a config key (YAML value)")
    train_parser.add_argument("--device", default="cpu")
    train_parser.add_argument("--name", help="run folder name (default: backbone, fold, UTC time)")
    references_parser = commands.add_parser("references",
                                            help="draw provisional Macenko references")
    references_parser.add_argument("--config", default=DEFAULT_TRAIN_CONFIG)
    args = parser.parse_args(argv)
    if args.command == "references":
        return references(args)
    if args.command == "train":
        return train(args)
    return verify(args.package)


if __name__ == "__main__":
    sys.exit(main())
