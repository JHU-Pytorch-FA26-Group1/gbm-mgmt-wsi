"""Regression tests for training/experiment.py: a whole run on the small test package."""
import csv
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from tests.test_training_data import SIDE, make_package
from training import config, experiment, models

LABELS = {"T1": "Methylated", "T2": "Unmethylated", "T3": "Unmethylated", "C1": "Methylated"}
FOLDS = {"T1": ("TCGA", "0"), "T2": ("TCGA", "1"), "T3": ("TCGA", "2"), "C1": ("CPTAC", "external")}


@pytest.fixture(name="setup")
def fixture_setup(tmp_path, monkeypatch):
    """(settings, record, tiles) for a two-epoch run on the small package, with a tiny backbone."""
    package = tmp_path / "pkg"
    package.mkdir()
    tiles = make_package(package)
    (tmp_path / "labels.csv").write_text("patient_id,mgmt_status\n" + "".join(
        f"{patient},{status}\n" for patient, status in LABELS.items()))
    (tmp_path / "folds.csv").write_text("source,patient_id,fold\n" + "".join(
        f"{source},{patient},{fold}\n" for patient, (source, fold) in FOLDS.items()))
    overrides = [f"package_dir={package}", f"labels_csv={tmp_path / 'labels.csv'}",
                 f"folds_csv={tmp_path / 'folds.csv'}", "tiles_per_bag=4",
                 "smoke_train_on_validation=false", "macenko_reference_patch_ids={}",
                 "stages_real=[{learning_rate: 1.0e-3, epochs: 2, class_weighted: true}]"]
    settings, record = config.load_config(backbone="swin_v2_t", overrides=overrides)

    def tiny(cfg, backbone, pretrained=True):
        del backbone, pretrained
        torch.manual_seed(0)
        return models.BagModel(nn.Sequential(nn.Flatten(), nn.Linear(3 * SIDE * SIDE, 4)), 4, cfg)

    monkeypatch.setattr(models, "build_model", tiny)
    monkeypatch.setattr(experiment, "read_tiles", lambda _: tiles)
    monkeypatch.setattr(experiment, "build_normalizer",
                        lambda *args: (None, {"failure_log": str(tmp_path / "f.csv")}))
    monkeypatch.setattr(experiment, "sha256_file", lambda _: "x")
    return settings, record, tiles


def read(path):
    """Rows of a CSV as dicts."""
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def run_with(setup, tmp_path, **changes):
    """Run fold 1 with `changes` applied to the settings; return the run folder."""
    settings, record, _ = setup
    return experiment.run_fold({**settings, **changes}, record, 1, tmp_path / "run")


def test_stage_settings_reach_the_optimizer_and_the_scheduler(setup, tmp_path, monkeypatch):
    """The stage's learning rate starts Adam and the step scheduler decays it once per epoch."""
    made = []

    class Recording(torch.optim.Adam):
        """Adam that remembers each instance built."""

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            made.append(self)

    monkeypatch.setattr(experiment.torch.optim, "Adam", Recording)
    stage = [{"learning_rate": 2.0e-3, "epochs": 3, "class_weighted": True}]
    run_with(setup, tmp_path, stages_real=stage, scheduler="step", scheduler_step_size=1,
             scheduler_gamma=0.5)
    assert made[0].defaults["lr"] == 2.0e-3
    assert made[0].param_groups[0]["lr"] == pytest.approx(2.0e-3 * 0.5 ** 3)


def test_training_stops_early_when_validation_stops_improving(setup, tmp_path, monkeypatch):
    """Patience 1 ends the stage: a validation loss that never improves runs 2 epochs."""
    monkeypatch.setattr(experiment.train, "validation_loss", lambda *args, **kwargs: 1.0)
    stage = [{"learning_rate": 1.0e-3, "epochs": 10, "class_weighted": True}]
    run = run_with(setup, tmp_path, stages_real=stage, early_stopping_patience=1)
    assert len(read(run / "metrics.csv")) == 2


def test_the_worker_count_reaches_validation_and_final_scoring_and_the_record(setup, tmp_path,
                                                                             monkeypatch):
    """Both scoring paths must receive num_workers, and the run record must say what was used."""
    seen = []
    original = experiment.evaluate.predict

    def spy(model, tiles, patients, settings, pipeline, **kwargs):
        seen.append(kwargs["num_workers"])
        return original(model, tiles, patients, settings, pipeline, device=kwargs["device"])

    monkeypatch.setattr(experiment.evaluate, "predict", spy)
    real_loader = experiment.DataLoader
    monkeypatch.setattr(experiment, "DataLoader", lambda dataset, **kwargs: real_loader(
        dataset, **{**kwargs, "num_workers": 0, "persistent_workers": False,
                    "multiprocessing_context": None}))
    run = run_with(setup, tmp_path, num_workers=2)
    assert seen == [2, 2, 2]
    body = json.loads((run / "run_record.json").read_text(encoding="utf-8"))
    assert body["num_workers"] == 2


def test_each_epoch_draws_its_own_bags(setup, tmp_path, monkeypatch):
    """The sampler must be told the epoch, or every epoch would reuse the same bags."""
    epochs = []
    original = experiment.data.EpochSampler.set_epoch

    def spy(self, epoch):
        epochs.append(epoch)
        original(self, epoch)

    monkeypatch.setattr(experiment.data.EpochSampler, "set_epoch", spy)
    run_with(setup, tmp_path)
    assert epochs == [1, 2]


def test_patients_per_update_reaches_the_epoch(setup, tmp_path, monkeypatch):
    """The setting that groups patients into an update must be the one the epoch receives."""
    seen = []
    original = experiment.train.train_epoch

    def spy(model, batches, optimizer, loss_fn, patients_per_update):
        seen.append(patients_per_update)
        return original(model, batches, optimizer, loss_fn, patients_per_update)

    monkeypatch.setattr(experiment.train, "train_epoch", spy)
    run_with(setup, tmp_path, patients_per_update=2)
    assert seen == [2, 2]


def test_validation_and_final_scoring_use_their_own_pool_caps(setup, tmp_path, monkeypatch):
    """Per-epoch validation uses validation_tiles_per_patient; predictions.csv uses the eval cap."""
    caps = []
    original = experiment.evaluate.predict

    def spy(model, tiles, patients, settings, *args, **kwargs):
        caps.append(settings["max_eval_tiles_per_patient"])
        return original(model, tiles, patients, settings, *args, **kwargs)

    monkeypatch.setattr(experiment.evaluate, "predict", spy)
    run_with(setup, tmp_path, validation_tiles_per_patient=3, max_eval_tiles_per_patient=7)
    assert caps == [3, 3, 7]


def test_metrics_rows_hold_stage_epoch_train_loss_and_validation_loss(setup, tmp_path,
                                                                    monkeypatch):
    """Each column must carry its own value, in the order the header names them."""
    monkeypatch.setattr(experiment.train, "train_epoch", lambda *args, **kwargs: 0.25)
    monkeypatch.setattr(experiment.train, "validation_loss", lambda *args, **kwargs: 0.75)
    run = run_with(setup, tmp_path)
    assert read(run / "metrics.csv") == [
        {"stage": "1", "epoch": str(epoch), "train_loss": "0.25", "validation_loss": "0.75"}
        for epoch in (1, 2)]


def test_every_tile_goes_through_the_normalizer(setup, tmp_path, monkeypatch):
    """The fitted normalizer must be wired into the pipeline that reads the tiles."""
    calls = []
    normalizer = SimpleNamespace(transform=lambda rgb: calls.append(1) or rgb)
    record = {"failure_log": str(tmp_path / "failures.csv")}
    monkeypatch.setattr(experiment, "build_normalizer", lambda *args: (normalizer, record))
    run_with(setup, tmp_path)
    assert calls


def test_labels_that_do_not_match_the_fold_map_stop_the_run(setup, tmp_path):
    """A patient in the fold map without a label must be refused before any run folder exists."""
    settings, record, _ = setup
    path = Path(settings["labels_csv"])
    path.write_text("patient_id,mgmt_status\nT1,Methylated\nT2,Unmethylated\nT3,Unmethylated\n",
                    encoding="utf-8")
    with pytest.raises(RuntimeError, match="disagree"):
        experiment.run_fold(settings, record, 1, tmp_path / "run")
    assert not (tmp_path / "run").exists()


def test_the_reference_is_looked_up_by_fold_name(setup, tmp_path, monkeypatch):
    """The config keys folds as strings, so an integer fold must be converted to find its tile."""
    settings, _, tiles = setup
    monkeypatch.undo()
    chosen = []

    def fit(folder, fold, reference, reader):
        del folder, fold, reader
        chosen.append(reference)
        return None, {}

    monkeypatch.setattr(experiment.loader, "fit_fold_normalizer", fit)
    configured = {**settings, "macenko_reference_patch_ids": {"1": "T1_2"}}
    experiment.build_normalizer(configured, 1, tiles, None, tmp_path)
    assert chosen == ["T1_2"]


def test_a_fold_run_trains_on_the_other_folds_and_predicts_only_the_validation_patients(
        setup, tmp_path):
    """Fold 1 validates on T2 alone; neither the training patients nor CPTAC are scored."""
    settings, record, _ = setup
    run = experiment.run_fold(settings, record, 1, tmp_path / "run")
    predictions = read(run / "predictions.csv")
    assert [(row["patient_id"], row["fold"], row["label"]) for row in predictions] == [
        ("T2", "1", "0")]
    assert 0 < float(predictions[0]["score"]) < 1
    assert [row["epoch"] for row in read(run / "metrics.csv")] == ["1", "2"]
    body = json.loads((run / "run_record.json").read_text(encoding="utf-8"))
    assert body["valid_for_results"] is True
    assert body["fold"] == 1 and body["seed"] == settings["seed"] and body["config"] == record
    assert len(body["best_epochs"]) == 1 and body["best_epochs"][0] in (1, 2)
    assert body["patients"] == {"train": 2, "validation": 1, "external": 1}


def test_a_smoke_run_is_marked_invalid_for_results(setup, tmp_path):
    """Training on the validation patients must never produce a record that looks like a result."""
    settings, record, _ = setup
    unweighted = [{**settings["stages_real"][0], "class_weighted": False}]
    smoke = {**settings, "smoke_train_on_validation": True, "stages_real": unweighted}
    run = experiment.run_fold(smoke, record, 1, tmp_path / "run")
    body = json.loads((run / "run_record.json").read_text(encoding="utf-8"))
    assert body["valid_for_results"] is False


def test_mixed_precision_is_refused_until_it_is_implemented(setup, tmp_path):
    """A setting the loop ignores must stop the run rather than be silently dropped."""
    settings, record, _ = setup
    with pytest.raises(RuntimeError, match="mixed_precision"):
        experiment.run_fold({**settings, "mixed_precision": True}, record, 1, tmp_path / "run")


def test_a_run_folder_is_never_reused(setup, tmp_path):
    """A second run into the same folder would append to the first run's metrics."""
    settings, record, _ = setup
    experiment.run_fold(settings, record, 1, tmp_path / "run")
    with pytest.raises(FileExistsError):
        experiment.run_fold(settings, record, 1, tmp_path / "run")


def test_a_fold_without_a_reference_tile_needs_one_unless_smoke(setup, tmp_path, monkeypatch):
    """Macenko must be fitted on a recorded reference; only a smoke run may pick one itself."""
    settings, _, tiles = setup
    monkeypatch.undo()
    with pytest.raises(RuntimeError, match="macenko_reference_patch_ids"):
        experiment.build_normalizer(settings, 1, tiles, None, tmp_path)


def test_a_smoke_run_picks_its_reference_from_training_patients_of_the_real_split(
        setup, tmp_path, monkeypatch):
    """A smoke run trains on validation patients; its reference is still a real training tile."""
    settings, _, tiles = setup
    monkeypatch.undo()
    chosen = []

    def fit(folder, fold, reference, reader):
        del folder, fold, reader
        chosen.append(reference)
        return None, {}

    monkeypatch.setattr(experiment.loader, "fit_fold_normalizer", fit)
    smoke = {**settings, "smoke_train_on_validation": True}
    experiment.build_normalizer(smoke, 0, tiles, None, tmp_path)
    assert chosen == ["T2_0"]


def test_a_run_that_fails_its_checks_leaves_no_run_folder(setup, tmp_path):
    """A refused run must not leave an empty folder that later looks like a run."""
    settings, record, _ = setup
    bad = {**settings, "labels_csv": str(tmp_path / "missing.csv")}
    with pytest.raises(OSError):
        experiment.run_fold(bad, record, 1, tmp_path / "run")
    assert not (tmp_path / "run").exists()
