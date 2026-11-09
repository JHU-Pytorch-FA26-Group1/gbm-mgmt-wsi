"""Build the model input: modeling_patches.csv and a patient index, from QC decisions and reviews.

Examples:
    python data_prep/export.py --root ~/Bioinformatics/WSI/data --subset B

A tile is in modeling_patches.csv only if all of these hold:
  - its slide was tiled with the current config and CODE_VERSION, and the latest tile.py --audit
    covering the slide passed after it was tiled;
  - the QC policy is frozen (qc_policy.yaml: frozen, reviewer, notes, calibration patients, blur
    cutoffs), its calibration patients are TCGA patients in the fold map, and qc.py applied it;
  - after reviews (review.py), its decision is accept; review cases and slide exclusions stay out.
    Edge tiles are never accepted by QC, so an edge tile is in only if a reviewer accepted it;
  - its patient is in the shared fold map, and every one of the patient's slides in the subset is
    accounted for: exported, or explicitly excluded at sign-off. With require_signoff: true in the
    policy, a slide without a sign-off (approve or exclude) blocks its patient.
Accounting is per patient: a patient with a missing, failed or unaudited slide is listed as blocked,
without blocking anyone else. A patient with no accepted tiles is listed as an empty bag.

One modeling list serves every fold: each row carries the patient's fold, and loader.py marks rows
train / validation / external for the fold being trained. With stored_tiles_only: true in the policy
(the default), the model uses only tissue tiles whose pixels are stored (the seeded per-slide sample of
max_tiles_per_slide), so training needs only the .h5 files; accepted tiles beyond the cap are counted
but left out. Edge tiles a reviewer accepted are always included; package.py stores their pixels.
With stored_tiles_only: false, accepted tiles beyond the cap are included and reread from the slide.

The export is complete when no patient is blocked; export_settings.json records it, and loader.py
refuses an incomplete export unless told a partial one is intended (for previews).

Writes patches/<tiling version>/export/<policy version>/<subset>_<UTC time>/: modeling_patches.csv,
patient_index.csv, slide_status.csv and export_settings.json (hashes of every input, with paths
relative to --root). loader.py refuses an export whose recorded inputs have changed since (reviews,
audits, QC decisions, fold map): rerun export.py after any of them changes.

Exit codes: 0 = exported, no patient blocked, 1 = exported but some patients are blocked,
2 = setup problem (e.g. policy not frozen, QC not run).
"""
import argparse
import csv
import json
import logging
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import yaml

import qc
import review
import split
from common import (CODE_VERSION, DEFAULT_FOLDS, DEFAULT_POLICY, DEFAULT_TILING_CONFIG,
                    load_tiling_config, output_paths, read_folds, read_rows, sha256_file, tiled,
                    write_rows)
from download import DEFAULT_LOG_DIR, DEFAULT_MANIFEST, load_manifest, setup_logging, write_json

MODEL_FIELDS = ["source", "patient_id", "slide_id", "fold", "patch_id", "role", "grid_index", "x_target",
                "y_target", "x0", "y0", "x1", "y1", "stored", "h5", "h5_index", "slide_relpath",
                "pixel_sha256", "decided_by"]

log = logging.getLogger("export")


def parse_args(argv=None):
    """Parse command-line options; pass `argv` as a list to test without a shell."""
    parser = argparse.ArgumentParser(description="Build the modeling list from QC decisions and reviews.")
    parser.add_argument("--root", type=Path, required=True, help="data folder used by tile.py")
    parser.add_argument("--subset", choices=["full", "A", "B"], required=True,
                        help="full cohort, or team subset A or B")
    parser.add_argument("--tiling-config", type=Path, default=DEFAULT_TILING_CONFIG,
                        help="tiling config whose outputs to export (default: %(default)s)")
    parser.add_argument("--policy", type=Path, default=DEFAULT_POLICY,
                        help="QC policy; must be frozen (default: %(default)s)")
    parser.add_argument("--folds", type=Path, default=DEFAULT_FOLDS,
                        help="shared patient fold map (default: %(default)s)")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST,
                        help="frozen slide manifest (default: %(default)s)")
    parser.add_argument("--log-dir", type=Path, default=DEFAULT_LOG_DIR,
                        help="folder for run .log files (default: %(default)s)")
    return parser.parse_args(argv)


def audit_files(root, version, subset):
    """Every saved audit for a tiling version and subset, oldest first."""
    return sorted((Path(root) / "state" / "tiling" / version).glob(f"audit_{subset}_*.json"))


def qc_record(root, version, policy_version, subset):
    """The record qc.py writes when it applies a policy to a subset."""
    return Path(root) / "patches" / version / "qc" / policy_version / f"policy_{subset}.json"


def input_hashes(root, version, policy_version, subset, folds_path):
    """SHA-256 of every input an export depends on, keyed by path relative to `root`.

    loader.py recomputes this to detect a stale export; the fold map is keyed "folds".
    """
    root = Path(root)
    paths = [*review.review_files(root, version), *audit_files(root, version, subset),
             qc_record(root, version, policy_version, subset)]
    hashes = {str(p.relative_to(root)): sha256_file(p) for p in paths if p.exists()}
    hashes["folds"] = sha256_file(folds_path)
    return hashes


def latest_audits(root, version, subset):
    """{slide_id: (audit result, audit time, audit file)} from the newest audit covering each slide."""
    found = {}
    for path in audit_files(root, version, subset):
        report = json.loads(path.read_text())
        for result in report["slides"]:
            found[result["slide_id"]] = (result, report["audited_at_utc"], path)
    return found


def main(argv=None):
    """Check the gates, apply reviews, and write the modeling list. Returns the exit code."""
    args = parse_args(argv)
    args.root = args.root.expanduser().resolve()
    log_path = setup_logging(log, args.log_dir.expanduser().resolve(), f"export_{args.subset}")
    log.info("Command: %s", " ".join(sys.argv))
    try:
        cfg, config_sha, _ = load_tiling_config(args.tiling_config.expanduser().resolve())
        version = cfg["version"]
        policy_path = args.policy.expanduser().resolve()
        policy, policy_sha, _ = qc.load_policy(policy_path)
        if not policy["frozen"]:
            raise RuntimeError(f"QC policy {policy['version']} is not frozen; set frozen, reviewer "
                               f"and notes in {policy_path} once the team agrees the cutoffs")
        folds_path = args.folds.expanduser().resolve()
        folds = read_folds(folds_path)
        not_training = [p for p in policy["calibration_patients"] if ("TCGA", p) not in folds]
        if not_training:
            raise RuntimeError(f"calibration patients must be TCGA patients in the fold map: {not_training}")
        _, manifest = load_manifest(args.manifest.expanduser().resolve())
        all_files = manifest["files"]
        expected = split.select_files(all_files, args.subset, split.build_patient_split(all_files))
        reviews = review.load_reviews(args.root, version)
        audits = latest_audits(args.root, version, args.subset)
        record = qc_record(args.root, version, policy["version"], args.subset)
        if not record.exists() or json.loads(record.read_text())["policy_sha256"] != policy_sha:
            raise RuntimeError(f"run qc.py with policy {policy['version']} first (or again: the "
                               f"policy file changed since)")
    except (OSError, RuntimeError, ValueError, KeyError, yaml.YAMLError) as error:
        log.error("%s", error)
        return 2
    log.info("Tiling %s | policy %s (frozen by %s) | %d slides expected in subset %s | log %s",
             version, policy["version"], policy["reviewer"], len(expected), args.subset, log_path)

    # 1. Slide status: is each expected slide usable, excluded, or a blocker for its patient?
    status = []
    for row in expected:
        key = (row["source"], row["slide_id"])
        signoff = reviews.signoffs.get(key, {})
        audit = audits.get(row["slide_id"])
        receipt = output_paths(args.root, row, version)[4]
        if signoff.get("decision") == "exclude":
            state, why = "excluded", signoff.get("notes", "")
        elif not tiled(args.root, row, version, config_sha):
            state, why = "blocked", "not tiled with the current config"
        elif audit is None or audit[1] < json.loads(receipt.read_text())["tiled_at_utc"]:
            state, why = "blocked", "not audited since it was tiled"
        elif audit[0]["n_errors"]:
            state, why = "blocked", "failed the audit: " + "; ".join(audit[0]["errors"][:2])
        elif not qc.decisions_path(args.root, version, policy["version"], row).exists():
            state, why = "blocked", "no QC decisions; run qc.py"
        elif policy["require_signoff"] and signoff.get("decision") != "approve":
            state, why = "blocked", "no sign-off (the policy requires one per slide)"
        else:
            state, why = "ok", ""
        status.append(dict(source=row["source"], patient_id=row["patient_id"], slide_id=row["slide_id"],
                           file_name=row["file_name"], status=state, reason=why,
                           signed_off=signoff.get("decision", "")))
    by_patient = {}
    for s in status:
        by_patient.setdefault((s["source"], s["patient_id"]), []).append(s)
    blocked = {k: [f"{s['file_name']}: {s['reason']}" for s in v if s["status"] == "blocked"]
               for k, v in by_patient.items()}
    for key in by_patient:
        if key not in folds:
            blocked[key].append("not in the fold map")
    included = {k for k, reasons in blocked.items() if not reasons}

    # 2. Tiles: QC decision, then reviews; accepted tiles of included patients go to the model.
    stamp = datetime.now(timezone.utc)
    out = (args.root / "patches" / version / "export" / policy["version"]
           / f"{args.subset}_{stamp:%Y%m%d_%H%M%S}")
    out.mkdir(parents=True)
    patients = {k: Counter() for k in by_patient}
    seen_ids, by_whom = set(), Counter()
    model_path = out / "modeling_patches.csv"
    with model_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=MODEL_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for s in status:
            if s["status"] != "ok":
                continue
            key = (s["source"], s["patient_id"])
            row = next(r for r in expected if r["slide_id"] == s["slide_id"] and r["source"] == s["source"])
            h5_path, _, _, csv_path, _ = output_paths(args.root, row, version)
            tiles = {t["patch_id"]: t for t in read_rows(csv_path) if t["role"] in qc.ROLES}
            for d in read_rows(qc.decisions_path(args.root, version, policy["version"], row)):
                tile = tiles[d["patch_id"]]
                seen_ids.add(d["patch_id"])
                decision, _, decided_by = review.resolve(tile, d["decision"], d["reasons"], reviews)
                if tile["role"] == "tissue":
                    patients[key][decision] += 1
                    patients[key]["tissue_tiles"] += 1
                if decision == "accept" and key in included:
                    if (policy["stored_tiles_only"] and tile["role"] == "tissue"
                            and tile["stored"] != "True"):
                        patients[key]["not_stored"] += 1
                        continue
                    patients[key]["model_edge"] += tile["role"] == "edge"
                    writer.writerow({**tile, "fold": folds[key], "h5": str(h5_path.relative_to(args.root)),
                                     "decided_by": decided_by})
                    patients[key]["model_tiles"] += 1
                    patients[key]["model_stored"] += tile["stored"] == "True"
                    by_whom[decided_by] += 1
    unmatched = sorted(set(reviews.patches) - seen_ids)

    index = []
    for key, slides_ in sorted(by_patient.items()):
        c = patients[key]
        index.append(dict(source=key[0], patient_id=key[1], fold=folds.get(key, ""),
                          expected_slides=len(slides_),
                          exported_slides=sum(s["status"] == "ok" for s in slides_),
                          excluded_slides=sum(s["status"] == "excluded" for s in slides_),
                          included=key in included, tissue_tiles=c["tissue_tiles"], accept=c["accept"],
                          review=c["review"], reject=c["reject"], model_tiles=c["model_tiles"],
                          model_tiles_stored=c["model_stored"], model_edge_tiles=c["model_edge"],
                          accepted_not_stored=c["not_stored"],
                          empty_bag=key in included and c["model_tiles"] == 0,
                          blocked_reasons="; ".join(blocked[key])))
    write_rows(out / "patient_index.csv", index)
    write_rows(out / "slide_status.csv", status)
    counts = {"patients_expected": len(index), "patients_included": len(included),
              "patients_blocked": len(index) - len(included),
              "empty_bags": sum(r["empty_bag"] for r in index),
              "model_tiles": sum(r["model_tiles"] for r in index),
              "model_tiles_stored": sum(r["model_tiles_stored"] for r in index),
              "model_edge_tiles": sum(r["model_edge_tiles"] for r in index),
              "accepted_not_stored": sum(r["accepted_not_stored"] for r in index),
              "decided_by": dict(by_whom)}
    complete = counts["patients_blocked"] == 0
    write_json(out / "export_settings.json", {
        "subset": args.subset, "complete": complete, "created_at_utc": stamp.isoformat(),
        "tiling_version": version, "stored_tiles_only": policy["stored_tiles_only"],
        "config_sha256": config_sha, "code_version": CODE_VERSION, "policy": policy,
        "policy_sha256": policy_sha, "folds_file": folds_path.name, "folds_sha256": sha256_file(folds_path),
        "inputs": input_hashes(args.root, version, policy["version"], args.subset, folds_path),
        "audits_used": sorted({str(a[2].relative_to(args.root)) for a in audits.values()}),
        "modeling_sha256": sha256_file(model_path), "counts": counts,
        "unmatched_patch_reviews": unmatched})

    for r in index:
        if r["blocked_reasons"]:
            log.warning("Blocked %s %s: %s", r["source"], r["patient_id"], r["blocked_reasons"])
    if unmatched:
        log.warning("%d patch reviews match no tile of an exported slide (first: %s)", len(unmatched),
                    unmatched[0])
    log.info("%d of %d patients included (%d empty bags); %d model tiles, %d stored (%d accepted tiles "
             "beyond the cap left out); %s; written to %s", counts["patients_included"],
             counts["patients_expected"], counts["empty_bags"], counts["model_tiles"],
             counts["model_tiles_stored"], counts["accepted_not_stored"],
             "complete" if complete else "INCOMPLETE (some patients blocked)", out)
    return 1 if counts["patients_blocked"] else 0


if __name__ == "__main__":
    sys.exit(main())
