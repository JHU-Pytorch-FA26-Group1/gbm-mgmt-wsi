"""Write one row per canonical TCGA patient and paired development metrics."""
import csv
import json
import math
import sys
from pathlib import Path

from sklearn.metrics import roc_auc_score
from alternative_models.prepare import read_csv, write_json
from alternative_models.patient_metrics import summarize


def collect(root):
    repo = Path(__file__).resolve().parents[1]
    rows = {r["patient_id"]: dict(patient_id=r["patient_id"], fold=int(r["fold"]))
            for r in read_csv(repo/"manifests/patient_folds.csv") if r["source"] == "TCGA"}
    labels = {r["patient_id"]: int(r["mgmt_status"] == "Methylated")
              for r in read_csv(repo/"manifests/patient_labels.csv")}
    for p, row in rows.items():
        row.update(label=labels[p], qc_source=None, tumor_roi=None)
    arms = ("qc_source", "tumor_roi")
    for fold in range(5):
        for arm in arms:
            dest = root/f"fold-{fold}"/arm
            predictions = read_csv(dest/"train/predictions.csv")
            seen = set()
            for r in predictions:
                p = r["patient_id"]
                if (p in seen or rows[p]["fold"] != fold or int(r["fold"]) != fold
                        or int(r["label"]) != labels[p] or not 0 <= float(r["score"]) <= 1):
                    raise ValueError(f"Invalid {arm} fold {fold} prediction: {p}")
                seen.add(p)
                rows[p][arm] = float(r["score"])
            if len(seen) != json.loads((dest/"train/patient_metrics.json").read_text())["patients"]:
                raise ValueError("Incomplete fold output")
    with (root/"patient_predictions.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(next(iter(rows.values()))))
        w.writeheader()
        w.writerows(rows.values())
    metrics = {}
    for arm in arms:
        available = [r for r in rows.values() if r[arm] is not None]
        paired = [r for r in available if all(r[a] is not None for a in arms)]
        for subset, data in (("available", available), ("paired", paired)):
            pred = [dict(patient_id=r["patient_id"], label=r["label"], score=r[arm]) for r in data]
            stats = summarize(pred)
            stats.update(auroc=roc_auc_score([r["label"] for r in data],[r[arm] for r in data]),
                         log_loss=-sum(math.log(max(1e-15,r[arm] if r["label"] else 1-r[arm])) for r in data)/len(data))
            metrics[f"{arm}_{subset}"] = stats
    write_json(root/"development_metrics.json", metrics)


if __name__ == "__main__":
    collect(Path(sys.argv[1]))
