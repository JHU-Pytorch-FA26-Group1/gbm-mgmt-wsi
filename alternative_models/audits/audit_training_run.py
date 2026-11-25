#!/usr/bin/env python3
"""Read-only audit of a real classifier run; no model execution or score relabelling."""
import argparse
from collections import Counter
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys


def rows(path):
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def sha256(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def unique(table, key):
    found = {}
    for row in table:
        if row[key] in found:
            raise ValueError(f"Duplicate {key}: {row[key]}")
        found[row[key]] = row
    return found


def auc(labels, scores):
    """Pairwise patient AUROC; ties count half. Labels must be 0/1."""
    if len(labels) != len(scores) or any(y not in (0, 1) for y in labels):
        raise ValueError("AUROC requires equal lengths and binary labels")
    if any(not math.isfinite(s) for s in scores):
        raise ValueError("Nonfinite prediction score")
    positives = [s for y, s in zip(labels, scores) if y == 1]
    negatives = [s for y, s in zip(labels, scores) if y == 0]
    if not positives or not negatives:
        return None
    return sum((p > n) + .5 * (p == n) for p in positives for n in negatives) / (len(positives)*len(negatives))


def audit(repo, run, package=None):
    repo, run = Path(repo).resolve(), Path(run).resolve()
    record = json.loads((run / "run_record.json").read_text())
    predictions = rows(run / "predictions.csv")
    metrics = rows(run / "metrics.csv")
    canonical = unique(rows(repo / "manifests/patient_labels.csv"), "patient_id")
    folds = unique(rows(repo / "manifests/patient_folds.csv"), "patient_id")
    predicted = unique(predictions, "patient_id")
    errors, warnings = [], []
    fold = str(record["fold"])
    if fold not in {"0", "1", "2", "3", "4", "final"}:
        raise ValueError("Unknown run fold")
    if record.get("valid_for_results") is not True:
        errors.append("Run is not marked valid_for_results=True (possibly a smoke run)")
    labels, scores = [], []
    for patient, row in predicted.items():
        if patient not in canonical or patient not in folds:
            errors.append(f"Unknown patient: {patient}")
            continue
        truth = canonical[patient]
        if truth["mgmt_status"] not in {"Methylated", "Unmethylated"}:
            raise ValueError(f"Unknown canonical MGMT status: {patient}")
        label = int(row["label"])
        if label != int(truth["mgmt_status"] == "Methylated"):
            errors.append(f"Prediction label disagrees with repository label: {patient}")
        f = folds[patient]
        if (truth["source"] != f["source"] or
                (f["source"], f["fold"]) != (("CPTAC", "external") if fold == "final" else ("TCGA", fold))):
            errors.append(f"Patient is outside the expected validation/test partition: {patient}")
        if str(row["fold"]) != fold:
            errors.append(f"Prediction fold differs from run record: {patient}")
        score = float(row["score"])
        if not math.isfinite(score) or not 0 <= score <= 1:
            raise ValueError(f"Expected finite sigmoid probability in [0,1]: {patient}")
        labels.append(label)
        scores.append(score)
    if not predictions:
        errors.append("No patient predictions")
    result_auc = auc(labels, scores)
    if result_auc is None:
        errors.append("Both classes are required for patient AUROC")
    elif result_auc < .5:
        warnings.append("AUROC below 0.5; investigate. This audit does not invert scores or change labels")
    if scores and len(set(scores)) == 1:
        warnings.append("All prediction scores are identical")
    checkpoint = record.get("checkpoint", {})
    filename = checkpoint.get("file")
    if filename:
        path = (run / filename).resolve()
        if not path.is_relative_to(run):
            raise ValueError("Checkpoint path escapes run folder")
        if not path.is_file() or sha256(path) != checkpoint.get("sha256"):
            errors.append("Checkpoint missing or differs from run record")
    else:
        warnings.append("Checkpoint identity/hash absent from run record")

    # Historical records may store hashes and overrides rather than a complete settings snapshot.
    settings, settings_verified = None, False
    try:
        sys.path.insert(0, str(repo))
        sys.path.insert(0, str(repo / "data_prep"))
        from training import config
        old = record["config"]
        # JSON is a valid YAML value and avoids scalar-document marker artefacts.
        overrides = [f"{key}={json.dumps(value)}" for key, value in old.get("overrides", {}).items()]
        # The active backbone is not recorded by historical run records; infer only from the
        # two committed file hashes, and refuse ambiguous identical override files.
        candidates = []
        if "settings" in record:
            saved = record["settings"]
            config._validate(saved)
            if hashlib.sha256(json.dumps(saved, sort_keys=True).encode()).hexdigest() == old["settings_sha256"]:
                candidates.append(saved)
            else:
                errors.append("Saved effective settings disagree with settings_sha256")
        else:
            names = [name for name in config.BACKBONES if
                     sha256(repo / "configs/backbones" / f"{name}.yaml") == old["backbone_config_sha256"]]
            for name in names:
                candidate, hashes = config.load_config(backbone=name, overrides=overrides)
                if all(hashes.get(key) == old.get(key) for key in
                       ("train_config_sha256", "backbone_config_sha256", "settings_sha256")):
                    candidates.append(candidate)
        if len(candidates) == 1:
            settings, settings_verified = candidates[0], True
        else:
            warnings.append("Cannot reconstruct exact historical settings from current config hashes")
    except (ImportError, OSError, KeyError, RuntimeError, ValueError) as error:
        warnings.append(f"Historical settings unavailable: {type(error).__name__}")
    if settings_verified and settings["validation_tiles_per_patient"] != settings["max_eval_tiles_per_patient"] and fold != "final":
        warnings.append("Checkpoint-selection validation and reported predictions use different evaluation-pool caps")
    if package is None and settings:
        inferred = config.repo_path(settings["package_dir"])
        if (inferred / "modeling_patches.csv").is_file():
            package = inferred
    populations = None
    if package is not None:
        package = Path(package).resolve()
        modeling = package / "modeling_patches.csv"
        if sha256(modeling) != record.get("modeling_sha256"):
            errors.append("Supplied package modeling table differs from run's modeling_sha256")
        patients = {}
        for row in rows(modeling):
            key = row["patient_id"]
            identity = (row["source"], row["fold"])
            if key in patients and patients[key] != identity:
                raise ValueError(f"Patient has multiple source/fold identities: {key}")
            patients[key] = identity
            if key not in folds or identity != (folds[key]["source"], folds[key]["fold"]):
                errors.append(f"Package source/fold disagrees with repository: {key}")
                break
        train = {p for p, (source, f) in patients.items() if source == "TCGA" and (fold == "final" or f != fold)}
        expected = {p for p, identity in patients.items() if identity == (("CPTAC", "external") if fold == "final" else ("TCGA", fold))}
        if set(predicted) != expected:
            errors.append(f"Validation population mismatch: missing={sorted(expected-set(predicted))}; extra={sorted(set(predicted)-expected)}")
        if set(predicted) & train:
            errors.append("Training patients occur in validation/test predictions")
        counts = Counter(canonical[p]["mgmt_status"] for p in train if p in canonical)
        populations = dict(training_patients=len(train), expected_scored_patients=len(expected),
                           training_classes=dict(counts), expected_positive_weight=
                           counts["Unmethylated"]/counts["Methylated"] if counts["Methylated"] else None)
        recorded = record.get("patients", {})
        expected_role = "external" if fold == "final" else "validation"
        if expected_role in recorded and recorded[expected_role] != len(expected):
            errors.append("Run-record scored-patient count differs from supplied package")
        if "train" in recorded and recorded["train"] != len(train):
            errors.append("Run-record training-patient count differs from supplied package")
    else:
        warnings.append("Package not supplied/located: cannot verify complete expected validation population or image lineage")

    losses = [(int(r["epoch"]), float(r["validation_loss"])) for r in metrics if r.get("validation_loss", "") != ""]
    if any(not math.isfinite(loss) for _, loss in losses):
        errors.append("Nonfinite validation loss in epoch log")
    best = min(losses, key=lambda r: r[1]) if losses else None
    if best and record.get("best_epochs") and len(record["best_epochs"]) == 1 and best[0] != record["best_epochs"][0]:
        errors.append("Recorded best epoch disagrees with lowest logged validation loss")
    return dict(run=str(run), audit_script_sha256=sha256(__file__),
                record_sha256=sha256(run / "run_record.json"), predictions_sha256=sha256(run / "predictions.csv"),
                patients=len(predictions), classes=dict(Counter(labels)), auroc=result_auc,
                score_mean_by_label={str(label): statistics.mean(s for y, s in zip(labels, scores) if y == label)
                                     for label in sorted(set(labels))},
                unique_scores=len(set(scores)), score_std=statistics.pstdev(scores) if scores else None,
                lowest_validation_loss=dict(epoch=best[0], loss=best[1]) if best else None,
                exact_settings_reconstructed=settings_verified,
                evaluation_caps={k: settings[k] for k in ("validation_tiles_per_patient", "max_eval_tiles_per_patient")} if settings else None,
                populations=populations, errors=errors, warnings=warnings,
                scope="Read-only metadata/metric audit; no image review, clinical-label rederivation, training or GPU execution")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--package", type=Path)
    args = parser.parse_args()
    try:
        result = audit(args.repo, args.run, args.package)
        print(json.dumps(result, indent=2, allow_nan=False))
        return 1 if result["errors"] else 0
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(f"Audit stopped: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
