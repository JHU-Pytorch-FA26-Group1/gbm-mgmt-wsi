"""Deterministic patient-level split of the frozen cohort into two download subsets (A and B).

This is a download-work allocation, not a train/validation/test split, and it never uses
molecular labels. It reproduces the team notebook's split (gbm-team-halves-v1) exactly, so
the script and the notebook always agree on who downloads which patients.
"""
import csv
import hashlib
import io
import json

SPLIT_VERSION = "gbm-team-halves-v1"
SPLIT_TARGETS = {"TCGA": {"A": 76, "B": 75}, "CPTAC": {"A": 41, "B": 42}}
SUBSETS = ("A", "B")


def build_patient_split(all_files):
    """Assign every patient to A or B; all slides of a patient stay together."""
    assignments = []
    for source, targets in SPLIT_TARGETS.items():
        patients = {}
        for row in all_files:
            if row["source"] == source:
                patients.setdefault(row["patient_id"], []).append(row)
        if len(patients) != sum(targets.values()):
            raise RuntimeError(f"{source}: {len(patients)} patients, expected {sum(targets.values())}")

        # Largest patient bundles first, each to the subset with fewer bytes, within the
        # patient-count targets. The hash breaks size ties independent of row order.
        ordered = sorted(patients, key=lambda patient: (
            -sum(row["size_bytes"] for row in patients[patient]),
            hashlib.sha256(f"{SPLIT_VERSION}|{source}|{patient}".encode()).hexdigest()))
        counts = {"A": 0, "B": 0}
        sizes = {"A": 0, "B": 0}
        for patient in ordered:
            available = [half for half in SUBSETS if counts[half] < targets[half]]
            half = min(available, key=lambda c: (sizes[c], counts[c], c))
            size = sum(row["size_bytes"] for row in patients[patient])
            assignments.append({"source": source, "patient_id": patient, "half": half,
                                "slides": len(patients[patient]), "size_bytes": size})
            counts[half] += 1
            sizes[half] += size
        if counts != targets:
            raise RuntimeError(f"{source}: split counts {counts} != {targets}")
    return sorted(assignments, key=lambda row: (row["source"], row["patient_id"]))


def half_by_patient(assignments):
    """Map (source, patient_id) to its subset, "A" or "B"."""
    return {(row["source"], row["patient_id"]): row["half"] for row in assignments}


def select_files(all_files, subset, assignments):
    """All slides for subset 'full', otherwise only the slides of that subset's patients."""
    if subset == "full":
        return list(all_files)
    owner = half_by_patient(assignments)
    return [row for row in all_files if owner[(row["source"], row["patient_id"])] == subset]


def split_csv(rows):
    """Render rows as CSV text exactly as the team notebook does (same columns and line endings)."""
    buffer = io.StringIO(newline="")
    fields = list(dict.fromkeys(key for row in rows for key in row))
    writer = csv.DictWriter(buffer, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue()


def assignment_files(manifest, assignments):
    """The split records the notebook saves under manifests/, as {file name: text}."""
    all_files = manifest["files"]
    owner = half_by_patient(assignments)
    full_manifest_sha = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    outputs = {
        "patient_half_assignments.json": json.dumps({
            "split_version": SPLIT_VERSION, "full_manifest_sha256": full_manifest_sha,
            "patient_count_targets": SPLIT_TARGETS, "patients": assignments}, indent=2) + "\n",
        "patient_half_assignments.csv": split_csv(assignments),
    }
    for half in SUBSETS:
        patients = [row for row in assignments if row["half"] == half]
        slides = [{"half": half, **row} for row in all_files
                  if owner[(row["source"], row["patient_id"])] == half]
        outputs[f"patients_half_{half}.csv"] = split_csv(patients)
        outputs[f"slides_half_{half}.csv"] = split_csv(slides)
    return outputs


def subset_totals(assignments):
    """Patients, slides and GB per subset, for comparing tables with a teammate."""
    totals = []
    for half in SUBSETS:
        rows = [row for row in assignments if row["half"] == half]
        totals.append({"subset": half, "patients": len(rows),
                       "slides": sum(row["slides"] for row in rows),
                       "GB": round(sum(row["size_bytes"] for row in rows) / 1e9, 3)})
    return totals
