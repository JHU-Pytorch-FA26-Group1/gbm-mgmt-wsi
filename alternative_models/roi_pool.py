"""Select TCGA classifier tiles by their recorded same-footprint U-Net fraction."""
import csv
from collections import Counter
from pathlib import Path

from alternative_models.prepare import sha


def select_roi(table, folder):
    selected, evidence, coverage = set(), {}, Counter()
    tcga = table.loc[table["source"] == "TCGA"]
    if tcga["patch_id"].duplicated().any():
        raise ValueError("Duplicate TCGA source tile")
    for (half, slide), group in tcga.groupby(["part", "slide_id"], sort=False):
        path = Path(folder) / f"{half}-{slide}.csv"
        expected = {r.patch_id: (r.patient_id, int(r.x_target), int(r.y_target))
                    for r in group.itertuples()}
        evidence[str(path.resolve())] = sha(path)
        with path.open(newline="") as stream:
            for row in csv.DictReader(stream):
                key = row["patch_id"]
                patient = row["patient_id"]
                if (expected.pop(key, None) != (patient, int(row["x_target"]), int(row["y_target"]))
                        or row["half"] != half or row["slide_id"] != slide
                        or row["source"] != "TCGA" or row["reason"]):
                    raise ValueError(f"Invalid ROI match in {path}: {key}")
                value = float(row["tumor_fraction"])
                if not 0 <= value <= 1:
                    raise ValueError(f"Invalid tumor fraction for {key}")
                coverage[patient] += int(value >= 0.5)
                if value >= 0.5:
                    selected.add(key)
        if expected:
            raise ValueError(f"Missing {len(expected)} ROI scores in {path}")
    return selected, evidence, coverage
