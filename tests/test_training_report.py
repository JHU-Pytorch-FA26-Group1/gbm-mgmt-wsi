"""Regression tests for training/report.py."""
import json
import subprocess
from types import SimpleNamespace

import pytest

from training import report


def make_run(folder, scores, labels, valid=True):
    """A finished-run folder holding two epochs of metrics and the given predictions."""
    folder.mkdir()
    (folder / "metrics.csv").write_text(
        "stage,epoch,train_loss,validation_loss\n1,1,0.9,0.8\n1,2,0.7,0.6\n1,3,0.5,0.7\n",
        encoding="utf-8")
    rows = "".join(f"P{i},0,{label},{score}\n"
                   for i, (score, label) in enumerate(zip(scores, labels)))
    (folder / "predictions.csv").write_text("patient_id,fold,label,score\n" + rows,
                                            encoding="utf-8")
    (folder / "run_record.json").write_text(json.dumps({"valid_for_results": valid}),
                                            encoding="utf-8")
    return folder


def test_summary_reports_the_best_epoch_and_the_auroc_of_the_predictions(tmp_path):
    """The best epoch is the lowest validation loss, not the last; AUROC follows the scores."""
    perfect = make_run(tmp_path / "a", [0.9, 0.8, 0.2, 0.1], [1, 1, 0, 0])
    inverted = make_run(tmp_path / "b", [0.1, 0.2, 0.8, 0.9], [1, 1, 0, 0], valid=False)
    table = report.summarize({"perfect": perfect, "inverted": inverted}).set_index("run")
    assert table.loc["perfect", "best_epoch"] == 2
    assert table.loc["perfect", "best_validation_loss"] == pytest.approx(0.6)
    assert table.loc["perfect", "auroc"] == 1.0 and table.loc["inverted", "auroc"] == 0.0
    assert bool(table.loc["inverted", "valid_for_results"]) is False


def test_a_fold_with_one_class_has_no_auroc_rather_than_an_error(tmp_path):
    """A small fold can end up with a single class; the table must still be produced."""
    run = make_run(tmp_path / "a", [0.9, 0.8], [1, 1])
    assert report.summarize({"run": run})["auroc"].isna().all()


def test_launch_returns_the_printed_folder_and_surfaces_a_failure(monkeypatch):
    """The run folder is the last line printed; a refused run raises with the reason."""
    outcome = SimpleNamespace(returncode=0, stdout="warning\ndata/runs/x\n", stderr="")
    monkeypatch.setattr(subprocess, "run", lambda *args, **kwargs: outcome)
    assert str(report.launch("swin_v2_t", 0)) == "data/runs/x"
    outcome.returncode, outcome.stderr = 2, "cannot train: mixed_precision is not implemented"
    with pytest.raises(RuntimeError, match="mixed_precision"):
        report.launch("swin_v2_t", 0)
