"""Regression tests for training/experiment.py: a whole run on the small test package."""
import csv
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from tests.test_training_data import SIDE, make_package
from tests.test_training_tracking import fake_wandb
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
    real_hash = experiment.sha256_file
    monkeypatch.setattr(experiment, "sha256_file",
                        lambda path: "x" if Path(path).name == "modeling_patches.csv"
                        else real_hash(path))
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


def test_training_workers_stay_alive_and_start_from_the_warm_server(setup, tmp_path, monkeypatch):
    """The training loader keeps its workers between epochs and starts them as scoring does."""
    seen = []
    real_loader = experiment.DataLoader

    def spy(dataset, **kwargs):
        seen.append((kwargs["num_workers"], kwargs["persistent_workers"],
                     kwargs["multiprocessing_context"]))
        return real_loader(dataset, **{**kwargs, "num_workers": 0, "persistent_workers": False,
                                       "multiprocessing_context": None})

    monkeypatch.setattr(experiment, "DataLoader", spy)
    run_with(setup, tmp_path, num_workers=3)
    assert seen == [(3, True, "forkserver")]


def scripted_epochs(monkeypatch, validation_losses):
    """Make epoch N set every weight to N and report the given validation losses in turn."""
    epochs = iter(range(1, 100))
    losses = iter(validation_losses)

    def train_epoch(model, *_):
        value = float(next(epochs))
        for parameter in model.parameters():
            parameter.data.fill_(value)
        return 0.25

    monkeypatch.setattr(experiment.train, "train_epoch", train_epoch)
    monkeypatch.setattr(experiment.train, "validation_loss", lambda *args: next(losses))


def test_the_checkpoint_holds_the_best_epochs_weights_not_the_last(setup, tmp_path, monkeypatch):
    """Epoch 1 has the lowest validation loss, so the saved weights are epoch 1's, not epoch 2's."""
    scripted_epochs(monkeypatch, [0.5, 0.9])
    run = run_with(setup, tmp_path)
    weights = torch.load(run / "model.pt")
    assert weights and all(bool((value == 1.0).all()) for value in weights.values())


def test_the_checkpoint_loads_into_a_fresh_model_of_the_same_shape(setup, tmp_path):
    """The saved file must restore a model built the same way, with no missing or extra weights."""
    settings, _, _ = setup
    run = run_with(setup, tmp_path)
    fresh = models.build_model(settings, "swin_v2_t")
    fresh.load_state_dict(torch.load(run / "model.pt"))


def test_the_record_holds_the_checkpoints_name_and_the_hash_of_its_bytes(setup, tmp_path):
    """A copy of the file can be verified against the record after it is pulled."""
    run = run_with(setup, tmp_path)
    body = json.loads((run / "run_record.json").read_text(encoding="utf-8"))
    expected = hashlib.sha256((run / "model.pt").read_bytes()).hexdigest()
    assert body["checkpoint"] == {"file": "model.pt", "sha256": expected}


def test_a_run_that_dies_mid_training_leaves_the_best_weights_so_far(setup, tmp_path,
                                                                  monkeypatch):
    """The best epoch is saved as it happens, so a crash in epoch 3 keeps epoch 2's weights."""
    scripted_epochs(monkeypatch, [0.9, 0.5, 0.7])
    epochs = {"count": 0}
    fill = experiment.train.train_epoch

    def crash_in_epoch_three(model, *args):
        epochs["count"] += 1
        if epochs["count"] == 3:
            raise RuntimeError("the box died")
        return fill(model, *args)

    monkeypatch.setattr(experiment.train, "train_epoch", crash_in_epoch_three)
    stage = [{"learning_rate": 1.0e-3, "epochs": 3, "class_weighted": True}]
    with pytest.raises(RuntimeError, match="the box died"):
        run_with(setup, tmp_path, stages_real=stage)
    weights = torch.load(tmp_path / "run" / "model.pt")
    assert all(bool((value == 2.0).all()) for value in weights.values())


def test_a_crash_while_saving_leaves_the_previous_checkpoint_untouched(tmp_path, monkeypatch):
    """The file is replaced only once the new one is complete."""
    model = torch.nn.Linear(2, 2)
    experiment.save_checkpoint(model, tmp_path)
    before = (tmp_path / "model.pt").read_bytes()

    def half_written(_, path):
        Path(path).write_bytes(b"partial")
        raise OSError("disk full")

    monkeypatch.setattr(experiment.torch, "save", half_written)
    with pytest.raises(OSError, match="disk full"):
        experiment.save_checkpoint(torch.nn.Linear(2, 2), tmp_path)
    assert (tmp_path / "model.pt").read_bytes() == before


def test_the_weights_are_saved_before_the_validation_patients_are_scored(setup, tmp_path,
                                                                       monkeypatch):
    """A failure while scoring must not cost the trained model."""
    def refuse(*args):
        raise RuntimeError("scoring failed")

    monkeypatch.setattr(experiment, "write_predictions", refuse)
    with pytest.raises(RuntimeError, match="scoring failed"):
        run_with(setup, tmp_path)
    assert (tmp_path / "run" / "model.pt").is_file()


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


def final_run(setup, tmp_path, **changes):
    """Run the final fit with `changes` applied to the settings; return the run folder."""
    settings, record, _ = setup
    return experiment.run_fold({**settings, **changes}, record, "final", tmp_path / "final")


def test_the_final_fit_trains_on_every_tcga_patient_and_scores_cptac_only(setup, tmp_path):
    """The training pool is all three TCGA patients and the only predictions are CPTAC's."""
    run = final_run(setup, tmp_path)
    rows = read(run / "predictions.csv")
    assert [(r["patient_id"], r["fold"], r["label"]) for r in rows] == [("C1", "final", "1")]
    record = json.loads((run / "run_record.json").read_text(encoding="utf-8"))
    assert record["patients"] == {"train": 3, "validation": 0, "external": 1}


def test_the_final_fit_runs_every_configured_epoch_with_no_validation(setup, tmp_path,
                                                                      monkeypatch):
    """Patience 1 would stop a validated run early, and validation is never computed here."""
    def refuse(*args):
        raise AssertionError("the final fit has nothing to validate on")

    monkeypatch.setattr(experiment.train, "validation_loss", refuse)
    run = final_run(setup, tmp_path, early_stopping_patience=1,
                    stages_real=[{"learning_rate": 1.0e-3, "epochs": 3, "class_weighted": True}])
    rows = read(run / "metrics.csv")
    assert [row["epoch"] for row in rows] == ["1", "2", "3"]
    assert all(row["validation_loss"] == "" and row["validation_seconds"] == "" for row in rows)
    record = json.loads((run / "run_record.json").read_text(encoding="utf-8"))
    assert record["epochs"] == [3] and "best_epochs" not in record and record["valid_for_results"]


def test_the_final_checkpoint_holds_the_last_epochs_weights(setup, tmp_path, monkeypatch):
    """With no validation loss to pick a best epoch, the weights kept are those after the last."""
    scripted_epochs(monkeypatch, [])
    run = final_run(setup, tmp_path)
    weights = torch.load(run / "model.pt")
    assert weights and all(bool((value == 2.0).all()) for value in weights.values())


def test_the_final_fit_scores_cptac_with_the_evaluation_cap(setup, tmp_path, monkeypatch):
    """Scoring uses max_eval_tiles_per_patient, not the tuning-time validation cap."""
    seen = []
    original = experiment.evaluate.predict

    def spy(model, tiles, patients, settings, *args, **kwargs):
        seen.append((list(patients), settings["max_eval_tiles_per_patient"]))
        return original(model, tiles, patients, settings, *args, **kwargs)

    monkeypatch.setattr(experiment.evaluate, "predict", spy)
    final_run(setup, tmp_path, validation_tiles_per_patient=3, max_eval_tiles_per_patient=7)
    assert seen == [(["C1"], 7)]


def test_the_final_fit_refuses_the_smoke_option(setup, tmp_path):
    """There are no validation patients to train on for a smoke run, so it would train on none."""
    with pytest.raises(RuntimeError, match="no validation patients"):
        final_run(setup, tmp_path, smoke_train_on_validation=True)
    assert not (tmp_path / "final").exists()


def test_the_final_fit_needs_a_configured_stain_reference(tmp_path):
    """No entry for "final" must stop the run, not fall back to a reference nobody chose."""
    settings, _ = config.load_config(backbone="swin_v2_t",
                                     overrides=["macenko_reference_patch_ids={'0': 'p'}"])
    with pytest.raises(RuntimeError, match="needs an entry for fold final"):
        experiment.build_normalizer(settings, "final", None, None, tmp_path)


def test_the_final_fit_builds_its_normalizer_without_a_validation_fold(tmp_path, monkeypatch):
    """The per-fold fit rejects "final" as a fold, so the final fit must use its own."""
    settings, _ = config.load_config(backbone="swin_v2_t",
                                     overrides=["macenko_reference_patch_ids={'final': 'ref'}"])

    def refuse(*args):
        raise AssertionError("the final fit has no validation fold")

    monkeypatch.setattr(experiment.loader, "fit_fold_normalizer", refuse)
    monkeypatch.setattr(experiment, "fit_final_normalizer",
                        lambda folder, reference, reader: ("normalizer", {"reference": reference}))
    normalizer, record = experiment.build_normalizer(settings, "final", None, None, tmp_path)
    assert normalizer == "normalizer" and record["reference"] == "ref"
    assert record["failure_log"] == str(tmp_path / "normalization_failures.csv")


def test_the_final_stain_reference_is_a_tcga_tile_fitted_from_its_own_pixels(setup, monkeypatch):
    """The reference tile's pixels reach the normalizer, and a CPTAC or unknown tile is refused."""
    settings, _, tiles = setup
    package = Path(settings["package_dir"])
    reader = experiment.data.TilePipeline(package, settings).reader
    fitted = []

    class Fake:  # pylint: disable=too-few-public-methods
        """Stands in for the Macenko normalizer and keeps the pixels it was fitted on."""
        stain_matrix_target = torch.zeros(2).numpy()
        maxC_target = torch.ones(2).numpy()

        def fit(self, rgb):
            """Remember the reference tile."""
            fitted.append(rgb.copy())

    monkeypatch.setattr("tiatoolbox.tools.stainnorm.MacenkoNormalizer", Fake)
    monkeypatch.setattr(experiment.loader, "load_modeling", lambda *args, **kwargs: tiles)
    reference = tiles[tiles["patient_id"] == "T1"].iloc[0]
    normalizer, record = experiment.fit_final_normalizer(package, reference["patch_id"], reader)
    assert isinstance(normalizer, Fake) and (fitted[0] == reader.read(reference)).all()
    assert record["fold"] == "final" and record["reference_patient"] == "T1"
    cptac = tiles[tiles["patient_id"] == "C1"].iloc[0]["patch_id"]
    for bad in (cptac, "no_such_tile"):
        with pytest.raises(ValueError, match="not a TCGA tile"):
            experiment.fit_final_normalizer(package, bad, reader)


@pytest.fixture(name="wandb_runs")
def fixture_wandb_runs(monkeypatch):
    """The W&B runs started during a test, with a key configured."""
    monkeypatch.setenv("WANDB_API_KEY", "key")
    return fake_wandb(monkeypatch)


def test_every_epoch_row_reaches_wandb_and_the_csv_is_unchanged(setup, tmp_path, wandb_runs):
    """W&B gets what metrics.csv gets, and the run is closed as finished."""
    run = run_with(setup, tmp_path, wandb_project="gbm")
    rows = read(run / "metrics.csv")
    logged = wandb_runs[0].logged
    assert len(logged) == len(rows) == 2
    assert [row["epoch"] for row in logged] == [1, 2]
    assert [row["validation_loss"] for row in logged] == [float(r["validation_loss"]) for r in rows]
    assert wandb_runs[0].exit_code == 0 and wandb_runs[0].kwargs["name"] == "run"


def test_a_crash_closes_the_wandb_run_as_failed_and_still_raises(setup, tmp_path, monkeypatch,
                                                               wandb_runs):
    """Training that dies must not leave a W&B run open as if it were still going."""
    def crash(*args, **kwargs):
        raise RuntimeError("out of memory")

    monkeypatch.setattr(experiment.train, "train_epoch", crash)
    with pytest.raises(RuntimeError, match="out of memory"):
        run_with(setup, tmp_path, wandb_project="gbm")
    assert wandb_runs[0].exit_code == 1


def test_the_final_fit_logs_no_validation_values_to_wandb(setup, tmp_path, wandb_runs):
    """The blank validation columns of a final-fit row are dropped, not sent as empty metrics."""
    final_run(setup, tmp_path, wandb_project="gbm")
    assert all("validation_loss" not in row and "train_loss" in row
               for row in wandb_runs[0].logged)


def test_without_a_project_nothing_starts(setup, tmp_path, wandb_runs):
    """The default config writes the CSV files only."""
    run_with(setup, tmp_path)
    assert not wandb_runs and (tmp_path / "run" / "metrics.csv").exists()


def ticking_clock(monkeypatch, step=2.0, origin=0.0):
    """Make experiment's clock start at `origin` and advance `step` seconds on every reading."""
    ticks = iter(range(10 ** 6))
    monkeypatch.setattr(experiment, "time",
                        SimpleNamespace(perf_counter=lambda: origin + next(ticks) * step))


def test_metrics_rows_hold_every_column_in_header_order(setup, tmp_path, monkeypatch):
    """Each column must carry its own value: losses, the two phase times and the peak memory."""
    monkeypatch.setattr(experiment.train, "train_epoch", lambda *args, **kwargs: 0.25)
    monkeypatch.setattr(experiment.train, "validation_loss", lambda *args, **kwargs: 0.75)
    monkeypatch.setattr(experiment, "peak_memory_mb", lambda device: 7.5)
    ticking_clock(monkeypatch)
    run = run_with(setup, tmp_path)
    assert read(run / "metrics.csv") == [
        {"stage": "1", "epoch": str(epoch), "train_loss": "0.25", "validation_loss": "0.75",
         "train_seconds": "2.0", "validation_seconds": "2.0", "peak_memory_mb": "7.5"}
        for epoch in (1, 2)]


def test_the_memory_window_restarts_every_epoch(setup, tmp_path, monkeypatch):
    """A peak carried over from epoch to epoch would hide which epoch used the memory."""
    resets = []
    monkeypatch.setattr(experiment, "reset_peak_memory", resets.append)
    run_with(setup, tmp_path)
    assert resets == ["cpu", "cpu"]


def test_peak_memory_is_read_only_on_cuda_and_in_megabytes(monkeypatch):
    """Only CUDA has a peak counter; other devices leave the column empty."""
    resets = []
    monkeypatch.setattr(experiment.torch.cuda, "max_memory_allocated", lambda device: 5 * 2 ** 20)
    monkeypatch.setattr(experiment.torch.cuda, "reset_peak_memory_stats", resets.append)
    assert experiment.peak_memory_mb("cuda:0") == 5.0
    assert experiment.peak_memory_mb("cpu") == "" and experiment.peak_memory_mb("mps") == ""
    experiment.reset_peak_memory("cuda:0")
    experiment.reset_peak_memory("mps")
    assert resets == ["cuda:0"]


def test_the_record_holds_the_time_the_run_took_not_the_clock_reading(setup, tmp_path,
                                                                    monkeypatch):
    """With a clock that starts at 1000 s, the time is end minus start and covers both epochs."""
    ticking_clock(monkeypatch, step=3.0, origin=1000.0)
    run = run_with(setup, tmp_path)
    body = json.loads((run / "run_record.json").read_text(encoding="utf-8"))
    assert 2 * (3.0 + 3.0) <= body["seconds"] < 1000.0


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
