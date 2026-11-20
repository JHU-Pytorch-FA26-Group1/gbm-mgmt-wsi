"""One training run on one validation fold, or the final fit: the files it leaves behind.

A run folder holds metrics.csv (one row per epoch), predictions.csv (the validation patients, or
CPTAC for the final fit) and run_record.json. The final fit (fold "final") trains on every TCGA
patient for the configured number of epochs, with no validation and no early stopping.
"""
import csv
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
import torchvision
from torch.utils.data import DataLoader

import loader
from common import read_folds, sha256_file
from training import config, data, evaluate, labels as label_tools, models, tracking, train

METRIC_FIELDS = ("stage", "epoch", "train_loss", "validation_loss", "train_seconds",
                 "validation_seconds", "peak_memory_mb")
PREDICTION_FIELDS = ("patient_id", "fold", "label", "score")
CHECKPOINT_FILE = "model.pt"


def read_tiles(settings):
    """The package's modeling list, checked; stops if a pool limit is above any patient's tiles."""
    tiles = loader.load_modeling(config.repo_path(settings["package_dir"]))
    config.check_pool_limits(settings, data.tiles_per_patient(tiles))
    return tiles


def fit_final_normalizer(folder, reference_patch_id, reader):
    """(Macenko normalizer, record) for the final fit, where any TCGA tile can be the reference."""
    from tiatoolbox.tools.stainnorm import MacenkoNormalizer  # pylint: disable=import-outside-toplevel
    folder = Path(folder)
    match = loader.load_modeling(folder, allow_partial=True)
    match = match[match["patch_id"] == reference_patch_id]
    if len(match) != 1 or match.iloc[0]["source"] != "TCGA":
        raise ValueError(f"{reference_patch_id} is not a TCGA tile")
    reference = match.iloc[0]
    normalizer = MacenkoNormalizer()
    normalizer.fit(reader.read(reference).copy())   # Macenko may modify its input array
    return normalizer, {"fold": "final", "reference_patch_id": reference_patch_id,
                        "reference_patient": reference["patient_id"], "folder": str(folder),
                        "modeling_sha256": sha256_file(folder / "modeling_patches.csv"),
                        "stain_matrix_target": normalizer.stain_matrix_target.tolist(),
                        "maxC_target": normalizer.maxC_target.tolist(),
                        "fitted_at_utc": datetime.now(timezone.utc).isoformat()}


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
    if fold == "final":
        normalizer, record = fit_final_normalizer(folder, reference, reader)
    else:
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


def record_epoch(context, row):
    """Append one epoch's row to metrics.csv and send it to W&B when it is on."""
    append_row(context["run_dir"] / "metrics.csv", METRIC_FIELDS, row)
    tracking.log(context["wandb"], dict(zip(METRIC_FIELDS, row)))


def reset_peak_memory(device):
    """Start a new peak-memory window; only CUDA keeps a peak counter."""
    if str(device).startswith("cuda"):
        torch.cuda.reset_peak_memory_stats(device)


def peak_memory_mb(device):
    """Most CUDA memory allocated since `reset_peak_memory`, in MB; empty on other devices."""
    if str(device).startswith("cuda"):
        return round(torch.cuda.max_memory_allocated(device) / 2 ** 20, 1)
    return ""


def make_scheduler(settings, optimizer):
    """The learning rate scheduler named in the config, or None."""
    if settings["scheduler"] == "none":
        return None
    return torch.optim.lr_scheduler.StepLR(optimizer, settings["scheduler_step_size"],
                                           settings["scheduler_gamma"])


def run_stage(model, stage, index, context):
    """Train one stage (the final fit runs a fixed epoch count); return its best epoch."""
    settings, groups, labels = context["settings"], context["groups"], context["labels"]
    optimizer = torch.optim.Adam(model.parameters(), lr=stage["learning_rate"])
    scheduler = make_scheduler(settings, optimizer)
    weight = (label_tools.methylated_weight(labels, groups["train"])
              if stage["class_weighted"] else None)
    loss_fn = train.make_loss(weight)
    pending = {}   # run_epoch hands its epoch and loss to validate, which writes the metrics row

    def run_epoch(epoch):
        reset_peak_memory(context["device"])
        context["sampler"].set_epoch(epoch)
        started = time.perf_counter()
        loss = train.train_epoch(model, context["loader"], optimizer, loss_fn,
                                 settings["patients_per_update"])
        if scheduler is not None:
            scheduler.step()
        pending.update(epoch=epoch, train_loss=loss, train_seconds=time.perf_counter() - started)
        return loss

    def validate():
        started = time.perf_counter()
        rows = evaluate.predict(model, context["tiles"], groups["validation"], context["capped"],
                                context["pipeline"], device=context["device"],
                                num_workers=settings["num_workers"])
        loss = train.validation_loss(rows, labels, loss_fn)
        record_epoch(context, (index + 1, pending["epoch"], pending["train_loss"], loss,
                               round(pending["train_seconds"], 1),
                               round(time.perf_counter() - started, 1),
                               peak_memory_mb(context["device"])))
        return loss

    if context["final"]:
        train.fit_fixed(stage["epochs"], run_epoch, lambda epoch, loss: record_epoch(
            context, (index + 1, epoch, loss, "", round(pending["train_seconds"], 1), "",
                      peak_memory_mb(context["device"]))))
        save_checkpoint(model, context["run_dir"])
        return stage["epochs"]

    _, best = train.fit_stage(model, stage["epochs"], settings["early_stopping_patience"],
                              run_epoch, validate,
                              on_best=lambda: save_checkpoint(model, context["run_dir"]))
    return best


def save_checkpoint(model, run_dir):
    """Write the model's weights (CHECKPOINT_FILE), on the CPU so they load anywhere.

    The file is replaced in one step, so a crash while saving leaves the previous best intact.
    """
    path = Path(run_dir) / CHECKPOINT_FILE
    partial = path.with_name(path.name + ".tmp")
    torch.save({name: value.cpu() for name, value in model.state_dict().items()}, partial)
    os.replace(partial, path)


def write_predictions(model, fold, context):
    """Score the validation patients (CPTAC for the final fit), pool capped by the eval cap."""
    settings, labels = context["settings"], context["labels"]
    patients = context["groups"]["external" if fold == "final" else "validation"]
    rows = evaluate.predict(model, context["tiles"], patients, settings,
                            context["pipeline"], device=context["device"],
                            num_workers=settings["num_workers"])
    for patient, _, probability in rows:
        append_row(context["run_dir"] / "predictions.csv", PREDICTION_FIELDS,
                   (patient, fold, labels[patient], probability))


def run_fold(settings, record, fold, run_dir, device="cpu"):  # pylint: disable=too-many-locals
    """Train the real-only condition on validation fold `fold` and write the run folder."""
    if settings["mixed_precision"]:
        raise RuntimeError("mixed_precision is not implemented; set it to false")
    if fold == "final" and settings["smoke_train_on_validation"]:
        raise RuntimeError("the final fit has no validation patients to smoke on")
    run_dir = Path(run_dir)
    started = time.perf_counter()
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
    batches = DataLoader(dataset, batch_size=1, sampler=sampler, num_workers=workers,
                         persistent_workers=workers > 0,
                         multiprocessing_context=data.worker_context(workers))
    model = models.build_model(settings, settings["backbone"]).to(device)
    context = {"settings": settings, "tiles": tiles, "groups": groups, "labels": labels,
               "pipeline": pipeline, "sampler": sampler, "loader": batches,
               "run_dir": run_dir, "device": device, "final": fold == "final",
               "capped": {**settings,
                          "max_eval_tiles_per_patient": settings["validation_tiles_per_patient"]}}
    context["wandb"] = tracking.start(settings, run_dir.name, fold)
    try:
        best = [run_stage(model, stage, index, context)
                for index, stage in enumerate(settings["stages_real"])]
        checkpoint = {"file": CHECKPOINT_FILE, "sha256": sha256_file(run_dir / CHECKPOINT_FILE)}
        write_predictions(model, fold, context)
        write_record(run_dir, settings, {"fold": fold, "config": record,
                                         "epochs" if fold == "final" else "best_epochs": best,
                                         "checkpoint": checkpoint,
                                         "patients": {role: len(p) for role, p in groups.items()},
                                         "normalizer": normalizer_record,
                                         "seconds": round(time.perf_counter() - started, 1)})
    finally:
        tracking.finish(context["wandb"], failed=sys.exc_info()[0] is not None)
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
