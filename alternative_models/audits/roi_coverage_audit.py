"""Audit geometric ROI coverage of the completed A+B pilot's tile packages.

Run with Python 3.10+ and point --repo at the repaired alternative worktree.
Reads CSV/JSON metadata only. Emits slide rows and explicit zero-tile patient rows.
Coverage is the potential union of all intact crops from the packaged tile grid.
It measures reconstruction feasibility. No tumor scores or eligibility rules are used.
"""

import argparse
from collections import Counter, defaultdict
import csv
from hashlib import sha256
from itertools import groupby, product
import json
from pathlib import Path
import sys


def axis_cells(offset, tile, grid, margin):
    """Return the intervals in a tile that lie inside this window's crop."""
    start = offset * tile
    edges = (0, margin, tile - margin, tile)
    return [
        i for i, (left, right) in enumerate(zip(edges, edges[1:]))
        if start + left >= margin and start + right <= grid * tile - margin
    ]


def crop_masks(tile, grid, margin):
    return [
        sum(1 << (3 * cell_y + cell_x)
            for cell_x, cell_y in product(axis_cells(x, tile, grid, margin),
                                          axis_cells(y, tile, grid, margin)))
        for y in range(grid) for x in range(grid)
    ]


def coverage(positions, windows, masks):
    covered = defaultdict(int)
    for _, _, _, keys in windows:
        for position, mask in zip(keys, masks):
            covered[position] |= mask
    full = sum(mask == 0b111111111 for mask in covered.values())
    return full, len(covered) - full, len(positions) - len(covered)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    receipt = json.loads(args.receipt.read_text())
    repo = args.repo.resolve()
    geometry_path = repo / "alternative_models/roi_preview.py"
    geometry_hash = receipt["source_sha256"]["roi_preview.py"]
    if sha256(geometry_path.read_bytes()).hexdigest() != geometry_hash:
        raise ValueError("roi_preview.py SHA-256 differs from the completed preview receipt")

    sys.path.insert(0, str(repo))
    from alternative_models import roi_preview
    from alternative_models.roi_checkpoint import sha
    from alternative_models.roi_preview import (
        GRID, MARGIN, TILE, neighborhoods, package_info, read_csv, spaced_choices,
    )
    if Path(roi_preview.__file__).resolve() != geometry_path.resolve():
        raise ValueError("Imported roi_preview.py is outside the verified worktree path")

    expected = receipt["source_metadata"]
    fold_path = repo / "manifests/patient_folds.csv"
    if sha(fold_path) != expected["fold_sha256"]:
        raise ValueError("Fold manifest differs from the completed preview")
    folds = {row["patient_id"]: row for row in read_csv(fold_path)}
    excluded = str(receipt["selection"]["exclude_fold"])
    seed = receipt["selection"]["seed"]
    masks = crop_masks(TILE, GRID, MARGIN)
    rows = []
    owners = set()
    totals = defaultdict(Counter)
    expected_windows = expected["region_availability"]["patients"]
    checked_patients = set()

    def cohort_role(canonical):
        if canonical["source"] == "CPTAC":
            return "cptac_external"
        return "tcga_held_out" if canonical["fold"] == excluded else "tcga_training"

    for half in ("A", "B"):
        package = args.base / "packages" / f"{half}_qc-v5"
        info = package_info(package, half, fold_path)
        expected_package = expected["packages"][half]
        if (sha(package / "package_manifest.json") != expected_package["manifest_sha256"]
                or info["files"]["modeling_patches.csv"]["sha256"] != expected_package["modeling_sha256"]):
            raise ValueError(f"Half {half} differs from the completed preview")
        index = read_csv(package / "patient_index.csv")
        tile_counts = {r["patient_id"]: int(r["model_tiles"]) for r in index}
        if len(tile_counts) != len(index) or owners.intersection(tile_counts) or not tile_counts.keys() <= folds.keys():
            raise ValueError(f"Unknown or duplicated patient in half {half}")
        owners.update(tile_counts)
        print(f"Scanning half {half} coordinates", flush=True)
        slide_ids = set()
        observed = Counter()
        patient_windows = Counter()
        separated_by_slide = Counter()

        def row_prefix(patient, slide):
            canonical = folds[patient]
            return (half, canonical["source"], canonical["fold"],
                    cohort_role(canonical), patient, slide)

        with (package / "modeling_patches.csv").open(newline="", encoding="utf-8-sig") as handle:
            for slide, group in groupby(csv.DictReader(handle), key=lambda row: row["slide_id"]):
                if slide in slide_ids:
                    raise ValueError(f"Slide {slide} appears in multiple blocks")
                slide_ids.add(slide)
                positions = set()
                for index_in_slide, patch in enumerate(group):
                    if index_in_slide == 0:
                        patient = patch["patient_id"]
                    canonical = folds[patient]
                    x, y = int(patch["x_target"]), int(patch["y_target"])
                    if (patient not in tile_counts or patch["patient_id"] != patient
                            or patch["source"] != canonical["source"] or patch["fold"] != canonical["fold"]
                            or patch["stored"] != "True" or min(x, y) < 0 or x % TILE or y % TILE
                            or (x, y) in positions):
                        raise ValueError(f"Inconsistent coordinate metadata for {slide}")
                    positions.add((x, y))
                windows = neighborhoods({slide: positions})
                separated = len(spaced_choices(windows, 2, seed, patient))
                full, partial, none = coverage(positions, windows, masks)
                rows.append((*row_prefix(patient, slide), len(positions), len(windows),
                             separated, full, partial, none))
                observed[patient] += len(positions)
                patient_windows[patient] += len(windows)
                separated_by_slide[patient] += separated

        for patient, tiles in tile_counts.items():
            if observed[patient] != tiles:
                raise ValueError(f"Tile count differs from the package index for {patient}")
            role = cohort_role(folds[patient])
            # The preview counts windows on different slides as separated.
            separated = min(2, separated_by_slide[patient])
            if role == "tcga_training" and tiles:
                pilot = expected_windows[patient]
                if (pilot["half"] != half
                        or pilot["complete_5x5_neighborhoods"] != patient_windows[patient]
                        or pilot["separated_regions_found"] != separated):
                    raise ValueError(f"Window availability differs from the completed preview for {patient}")
                checked_patients.add(patient)
            if not tiles:
                rows.append((*row_prefix(patient, ""), 0, 0, 0, 0, 0, 0))
            summary = totals[half, role]
            summary["canonical_patients"] += 1
            summary["patients_with_model_tiles"] += bool(tiles)
            summary["patients_without_model_tiles"] += not tiles
            summary["patients_with_tiles_and_zero_intact_windows"] += bool(tiles) and separated == 0
            summary["patients_with_at_least_one_intact_window"] += separated >= 1
            summary["patients_with_two_separated_regions"] += separated == 2

    if owners != set(folds):
        raise ValueError("A+B package index does not cover the canonical patient manifest")
    if checked_patients != set(expected_windows):
        raise ValueError("TCGA training patients with model tiles differ from the completed preview")
    with args.out.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("half", "cohort", "fold", "cohort_role", "patient_id", "slide_id",
                         "model_tiles", "intact_windows", "separated_regions_on_slide_capped_at_two",
                         "potential_fully_covered_tiles", "potential_partially_covered_tiles",
                         "potential_uncovered_tiles"))
        writer.writerows(sorted(rows))
    print("Potential tile coverage assumes every intact crop is processed.")
    print("Patient geometry by half and cohort role:")
    fields = ("canonical_patients", "patients_with_model_tiles", "patients_without_model_tiles",
              "patients_with_tiles_and_zero_intact_windows", "patients_with_at_least_one_intact_window",
              "patients_with_two_separated_regions")
    writer = csv.writer(sys.stdout, lineterminator="\n")
    writer.writerow(("half", "cohort_role", *fields))
    for (half, role), summary in sorted(totals.items()):
        writer.writerow((half, role, *(summary[field] for field in fields)))
    print(args.out)


if __name__ == "__main__":
    main()
