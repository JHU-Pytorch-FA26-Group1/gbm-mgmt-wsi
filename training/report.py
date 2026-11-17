"""Launch runs from a notebook and read their files back: curves and a one-line summary per run."""
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
from sklearn.metrics import roc_auc_score

from common import REPO_ROOT


def launch(backbone, fold, overrides=(), device="cpu", name=None):
    """Run `python -m training.run train` and return the run folder it printed."""
    command = [sys.executable, "-m", "training.run", "train", "--backbone", backbone,
               "--fold", str(fold), "--device", device]
    if name:
        command += ["--name", name]
    for override in overrides:
        command += ["--set", override]
    done = subprocess.run(command, cwd=REPO_ROOT, capture_output=True, text=True, check=False)
    if done.returncode != 0:
        raise RuntimeError(f"train exited {done.returncode}: {done.stderr.strip()}")
    return Path(done.stdout.strip().splitlines()[-1])


def load_run(run_dir):
    """{"metrics": DataFrame, "predictions": DataFrame, "record": dict} for a finished run."""
    run_dir = Path(run_dir)
    return {"metrics": pd.read_csv(run_dir / "metrics.csv"),
            "predictions": pd.read_csv(run_dir / "predictions.csv"),
            "record": json.loads((run_dir / "run_record.json").read_text(encoding="utf-8"))}


def auroc(predictions):
    """Area under the ROC curve of `score` for `label`, or NaN when only one class is present."""
    if predictions["label"].nunique() < 2:
        return float("nan")
    return roc_auc_score(predictions["label"], predictions["score"])


def summarize(runs):
    """One row per run ({name: run folder}): best validation loss, its epoch, AUROC, validity."""
    rows = []
    for name, run_dir in runs.items():
        run = load_run(run_dir)
        metrics = run["metrics"]
        best = metrics.loc[metrics["validation_loss"].idxmin()]
        rows.append({"run": name, "valid_for_results": run["record"]["valid_for_results"],
                     "best_validation_loss": best["validation_loss"], "best_epoch": best["epoch"],
                     "epochs_run": len(metrics), "validation_patients": len(run["predictions"]),
                     "auroc": auroc(run["predictions"])})
    return pd.DataFrame(rows)


def plot_curves(runs):
    """Training and validation loss per epoch for each run; returns the matplotlib figure."""
    from matplotlib import pyplot  # pylint: disable=import-outside-toplevel
    figure, axes = pyplot.subplots(1, 2, figsize=(10, 3.5), sharex=True)
    for name, run_dir in runs.items():
        metrics = load_run(run_dir)["metrics"]
        axes[0].plot(metrics["train_loss"].to_numpy(), label=name)
        axes[1].plot(metrics["validation_loss"].to_numpy(), label=name)
    axes[0].set_title("training loss")
    axes[1].set_title("validation loss")
    for axis in axes:
        axis.set_xlabel("epoch (stages in order)")
    axes[1].legend()
    figure.tight_layout()
    return figure
