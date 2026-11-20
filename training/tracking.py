"""Weights & Biases logging beside the CSV files (proposal 3.7).

Off unless `wandb_project` is set and WANDB_API_KEY is in the environment; the CSV files are
always written. Paths and the worker count stay out of what W&B receives.
"""
import os
import sys

PRIVATE_KEYS = ("package_dir", "folds_csv", "labels_csv", "output_dir", "num_workers")


def start(settings, name, fold):
    """A W&B run for this training run, or None when W&B is off."""
    project = settings["wandb_project"]
    if project is None:
        return None
    if not os.environ.get("WANDB_API_KEY"):
        print("wandb_project is set but WANDB_API_KEY is not: CSV only", file=sys.stderr)
        return None
    import wandb  # pylint: disable=import-outside-toplevel
    config = {key: value for key, value in settings.items() if key not in PRIVATE_KEYS}
    return wandb.init(project=project, name=name, config={**config, "fold": fold})


def log(run, row):
    """Send one epoch's non-empty values to W&B; nothing when it is off."""
    if run is not None:
        run.log({key: value for key, value in row.items() if value != ""})


def finish(run, failed=False):
    """Close the W&B run, marked failed when training raised."""
    if run is not None:
        run.finish(exit_code=1 if failed else 0)
