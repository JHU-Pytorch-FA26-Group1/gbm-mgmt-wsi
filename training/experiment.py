"""One training run on one validation fold: stages, early stopping, and the files it leaves behind.

A run folder holds metrics.csv (one row per epoch), predictions.csv (validation patients, whole
pool) and run_record.json.
"""
import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import torch
import torchvision
from torch.utils.data import DataLoader

import loader
from common import read_folds, sha256_file
from training import config, data, evaluate, labels as label_tools, models, train

METRIC_FIELDS = ("stage", "epoch", "train_loss", "validation_loss")
PREDICTION_FIELDS = ("patient_id", "fold", "label", "score")


def read_tiles(settings):
    """The package's modeling list, checked; stops if a pool limit is above any patient's tiles."""
    tiles = loader.load_modeling(config.repo_path(settings["package_dir"]))
    config.check_pool_limits(settings, data.tiles_per_patient(tiles))
    return tiles


def build_normalizer(settings, fold, tiles, reader, run_dir):
    """(Macenko normalizer, its record) for `fold`, fitted on the configured reference tile.

    A smoke run with no configured reference takes the first TCGA training tile and records it.
    """
    reference = settings["macenko_reference_patch_ids"].get(str(fold))
    if reference is None:
        if not settings["smoke_train_on_validation"]:
            raise RuntimeError(f"macenko_reference_patch_ids needs an entry for fold {fold}")
        train_rows = loader.with_roles(tiles, fold)
        train_rows = train_rows[(train_rows["role"] == "train") & (train_rows["source"] == "TCGA")]
        reference = sorted(train_rows["patch_id"])[0]
    folder = config.repo_path(settings["package_dir"])
    normalizer, record = loader.fit_fold_normalizer(folder, fold, reference, reader)
    record["failure_log"] = str(Path(run_dir) / "normalization_failures.csv")
    return normalizer, record


def append_row(path, fields, row):
    """Append a row to a CSV, writing the header first when the file is new."""
    path = Path(path)
    new = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        if new:
            writer.writerow(fields)
        writer.writerow(row)


def make_scheduler(settings, optimizer):
    """The learning rate scheduler named in the config, or None."""
    if settings["scheduler"] == "none":
        return None
    return torch.optim.lr_scheduler.StepLR(optimizer, settings["scheduler_step_size"],
                                           settings["scheduler_gamma"])


def run_stage(model, stage, index, context):
    """Train one stage with early stopping; return its best epoch."""
    settings, groups, labels = context["settings"], context["groups"], context["labels"]
    optimizer = torch.optim.Adam(model.parameters(), lr=stage["learning_rate"])
    scheduler = make_scheduler(settings, optimizer)
    weight = (label_tools.methylated_weight(labels, groups["train"])
              if stage["class_weighted"] else None)
    loss_fn = train.make_loss(weight)
    pending = {}   # run_epoch hands its epoch and loss to validate, which writes the metrics row

    def run_epoch(epoch):
        context["sampler"].set_epoch(epoch)
        loss = train.train_epoch(model, context["loader"], optimizer, loss_fn,
                                 settings["patients_per_update"])
        if scheduler is not None:
            scheduler.step()
        pending.update(epoch=epoch, train_loss=loss)
        return loss

    def validate():
        rows = evaluate.predict(model, context["tiles"], groups["validation"], context["capped"],
                                context["pipeline"], device=context["device"],
                                num_workers=settings["num_workers"])
        loss = train.validation_loss(rows, labels, loss_fn)
        append_row(context["run_dir"] / "metrics.csv", METRIC_FIELDS,
                   (index + 1, pending["epoch"], pending["train_loss"], loss))
        return loss

    _, best = train.fit_stage(model, stage["epochs"], settings["early_stopping_patience"],
                              run_epoch, validate)
    return best


def write_predictions(model, fold, context):
    """Score each validation patient (pool capped by max_eval_tiles_per_patient); write the CSV."""
    settings, labels = context["settings"], context["labels"]
    rows = evaluate.predict(model, context["tiles"], context["groups"]["validation"], settings,
                            context["pipeline"], device=context["device"],
                            num_workers=settings["num_workers"])
    for patient, _, probability in rows:
        append_row(context["run_dir"] / "predictions.csv", PREDICTION_FIELDS,
                   (patient, fold, labels[patient], probability))


def run_fold(settings, record, fold, run_dir, device="cpu"):  # pylint: disable=too-many-locals
    """Train the real-only condition on validation fold `fold` and write the run folder."""
    if settings["mixed_precision"]:
        raise RuntimeError("mixed_precision is not implemented; set it to false")
    run_dir = Path(run_dir)
    torch.manual_seed(settings["seed"])
    tiles, groups = data.split_patients(read_tiles(settings), fold,
                                        settings["smoke_train_on_validation"])
    labels = label_tools.read_labels(settings)
    label_tools.check_against_folds(labels, read_folds(config.repo_path(settings["folds_csv"])))
    run_dir.mkdir(parents=True, exist_ok=False)
    package = config.repo_path(settings["package_dir"])
    plain = data.TilePipeline(package, settings)
    normalizer, normalizer_record = build_normalizer(settings, fold, tiles, plain.reader, run_dir)
    pipeline = data.TilePipeline(package, settings, normalizer, normalizer_record["failure_log"])
    dataset = data.BagDataset(tiles, groups["train"], labels, settings, pipeline)
    sampler = data.EpochSampler(len(dataset), settings["seed"])
    workers = settings["num_workers"]
    # spawn, not fork: forked workers crashed on macOS with HDF5 files already open
    batches = DataLoader(dataset, batch_size=1, sampler=sampler, num_workers=workers,
                         persistent_workers=workers > 0,
                         multiprocessing_context="spawn" if workers else None)
    model = models.build_model(settings, settings["backbone"]).to(device)
    context = {"settings": settings, "tiles": tiles, "groups": groups, "labels": labels,
               "pipeline": pipeline, "sampler": sampler, "loader": batches,
               "run_dir": run_dir, "device": device,
               "capped": {**settings,
                          "max_eval_tiles_per_patient": settings["validation_tiles_per_patient"]}}
    best = [run_stage(model, stage, index, context)
            for index, stage in enumerate(settings["stages_real"])]
    write_predictions(model, fold, context)
    write_record(run_dir, settings, {"fold": fold, "config": record, "best_epochs": best,
                                     "patients": {role: len(p) for role, p in groups.items()},
                                     "normalizer": normalizer_record})
    return run_dir


def write_record(run_dir, settings, facts):
    """run_record.json: `facts` plus what is needed to reproduce the run and judge it valid."""
    package = config.repo_path(settings["package_dir"])
    body = {
        **facts, "condition": "real",
        "valid_for_results": not settings["smoke_train_on_validation"],
        "modeling_sha256": sha256_file(package / "modeling_patches.csv"),
        "seed": settings["seed"], "num_workers": settings["num_workers"],
        "library_versions": {"torch": torch.__version__, "torchvision": torchvision.__version__},
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    (Path(run_dir) / "run_record.json").write_text(json.dumps(body, indent=2), encoding="utf-8")
