"""Regression tests for training/report.py."""
import json
import subprocess
from types import SimpleNamespace

import pandas as pd
import pytest

from training import report


def make_run(folder, scores, labels, valid=True):
    """A finished-run folder holding two epochs of metrics and the given predictions."""
    folder.mkdir()
    (folder / "metrics.csv").write_text(
        "stage,epoch,train_loss,validation_loss,train_seconds,validation_seconds,peak_memory_mb\n"
        "1,1,0.9,0.8,10.0,2.0,\n1,2,0.7,0.6,10.0,4.0,\n1,3,0.5,0.7,10.0,6.0,\n",
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


def test_summary_reports_time_per_epoch_and_the_largest_peak_memory(tmp_path):
    """Epoch time covers training and validation; memory is the largest peak, blank on non-CUDA."""
    run = make_run(tmp_path / "a", [0.9, 0.1], [1, 0])
    table = report.summarize({"run": run})
    assert table.loc[0, "seconds_per_epoch"] == pytest.approx(((12 + 14 + 16) / 3))
    assert table["peak_memory_mb"].isna().all()
    cuda = make_run(tmp_path / "b", [0.9, 0.1], [1, 0])
    text = (cuda / "metrics.csv").read_text(encoding="utf-8")
    text = text.replace("2.0,\n", "2.0,1500.5\n", 1).replace("6.0,\n", "6.0,900.0\n", 1)
    (cuda / "metrics.csv").write_text(text, encoding="utf-8")
    assert report.summarize({"run": cuda}).loc[0, "peak_memory_mb"] == 1500.5


def test_a_fold_with_one_class_has_no_auroc_rather_than_an_error(tmp_path):
    """A small fold can end up with a single class; the table must still be produced."""
    run = make_run(tmp_path / "a", [0.9, 0.8], [1, 1])
    assert report.summarize({"run": run})["auroc"].isna().all()


def make_fold_runs(root, valid=True):
    """Five fold runs of two patients each (one methylated, one not), and their modeling table.

    Each fold ranks its own patients perfectly, but fold k's scores sit 0.2 above fold k-1's.
    """
    run_dirs, rows = {}, []
    for fold in range(5):
        folder = root / f"fold{fold}"
        folder.mkdir()
        negative = 0.1 + 0.2 * fold
        scores = f"N{fold},{fold},0,{negative}\nP{fold},{fold},1,{negative + 0.05}\n"
        (folder / "predictions.csv").write_text("patient_id,fold,label,score\n" + scores,
                                                encoding="utf-8")
        (folder / "run_record.json").write_text(json.dumps({"valid_for_results": valid}),
                                                encoding="utf-8")
        run_dirs[fold] = folder
        rows += [{"source": "TCGA", "patient_id": f"{kind}{fold}", "fold": str(fold)}
                 for kind in "NP" for _ in range(3)]
    rows.append({"source": "CPTAC", "patient_id": "C1", "fold": "external"})
    return run_dirs, pd.DataFrame(rows)


def test_out_of_fold_pools_all_folds_before_scoring(tmp_path):
    """Each fold ranks perfectly, but ranking across folds does not, so the pool scores 15/25."""
    run_dirs, tiles = make_fold_runs(tmp_path)
    pooled = report.out_of_fold(run_dirs, tiles)
    summary = report.out_of_fold_summary(pooled)
    assert len(pooled) == 10 and summary["patients"] == 10
    assert summary["fold_auroc"] == {fold: 1.0 for fold in range(5)}
    assert summary["auroc"] == pytest.approx(15 / 25)


def test_out_of_fold_summary_adds_the_thresholded_metrics_only_when_asked(tmp_path):
    """At 0.52, three of five methylated and two of five unmethylated patients score above it."""
    run_dirs, tiles = make_fold_runs(tmp_path)
    pooled = report.out_of_fold(run_dirs, tiles)
    assert "f1" not in report.out_of_fold_summary(pooled)
    with_threshold = report.out_of_fold_summary(pooled, threshold=0.52)
    assert with_threshold["recall_methylated"] == pytest.approx(3 / 5)
    assert with_threshold["recall_unmethylated"] == pytest.approx(3 / 5)


def test_out_of_fold_needs_all_five_folds(tmp_path):
    """Four folds would give an AUROC on 8 of the 10 patients without saying so."""
    run_dirs, tiles = make_fold_runs(tmp_path)
    del run_dirs[3]
    with pytest.raises(ValueError, match="folds 0 to 4"):
        report.out_of_fold(run_dirs, tiles)


def test_out_of_fold_refuses_a_run_that_is_not_valid_for_results(tmp_path):
    """A smoke run trains on its own validation patients, so its scores must not be pooled."""
    run_dirs, tiles = make_fold_runs(tmp_path, valid=False)
    with pytest.raises(ValueError, match="not valid"):
        report.out_of_fold(run_dirs, tiles)


def test_out_of_fold_refuses_patients_that_differ_from_the_modeling_table(tmp_path):
    """A fold holding another fold's patient is not that fold's held-out set."""
    run_dirs, tiles = make_fold_runs(tmp_path)
    tiles.loc[tiles["patient_id"] == "N2", "fold"] = "3"
    with pytest.raises(ValueError, match="exactly the fold's patients"):
        report.out_of_fold(run_dirs, tiles)


def test_out_of_fold_refuses_a_patient_scored_twice(tmp_path):
    """A repeated patient row would count that patient twice in the pooled AUROC."""
    run_dirs, tiles = make_fold_runs(tmp_path)
    path = run_dirs[2] / "predictions.csv"
    path.write_text(path.read_text(encoding="utf-8") + "N2,2,0,0.5\n", encoding="utf-8")
    with pytest.raises(ValueError, match="exactly the fold's patients"):
        report.out_of_fold(run_dirs, tiles)


def test_out_of_fold_refuses_a_run_filed_under_the_wrong_fold(tmp_path):
    """Predictions labelled fold 4 inside the fold 1 slot mean the folders were mixed up."""
    run_dirs, tiles = make_fold_runs(tmp_path)
    run_dirs[1], run_dirs[4] = run_dirs[4], run_dirs[1]
    with pytest.raises(ValueError, match="holds predictions for folds"):
        report.out_of_fold(run_dirs, tiles)


def test_launch_returns_the_printed_folder_and_surfaces_a_failure(monkeypatch):
    """The run folder is the last line printed; a refused run raises with the reason."""
    outcome = SimpleNamespace(returncode=0, stdout="warning\ndata/runs/x\n", stderr="")
    monkeypatch.setattr(subprocess, "run", lambda *args, **kwargs: outcome)
    assert str(report.launch("swin_v2_t", 0)) == "data/runs/x"
    outcome.returncode, outcome.stderr = 2, "cannot train: mixed_precision is not implemented"
    with pytest.raises(RuntimeError, match="mixed_precision"):
        report.launch("swin_v2_t", 0)
