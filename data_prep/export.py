"""Build the model input: modeling_patches.csv and a patient index, from QC decisions and reviews.

Examples:
    python data_prep/export.py --root ~/Bioinformatics/WSI/data --subset B

A tile is in modeling_patches.csv only if all of these hold:
  - its slide was tiled with the current config and CODE_VERSION, and the latest tile.py --audit
    covering the slide passed after it was tiled;
  - the QC policy for its cohort is frozen (qc_policy.yaml: frozen, reviewer, notes, calibration
    patients, blur cutoff), its calibration patients are patients of that policy's cohorts in the fold
    map, and qc.py applied it with the same fold map and tiling index. One policy can cover both
    cohorts (the default), or --policy can name one policy per cohort (e.g. TCGA and CPTAC apart);
    every cohort in the subset must be covered by exactly one;
  - after reviews (review.py), its decision is accept; review cases and slide exclusions stay out.
    An edge tile is accepted by QC only when it meets the policy's edge_accept_* cutoffs, or by a reviewer;
  - its patient is in the shared fold map, and every one of the patient's slides in the subset is
    accounted for: exported, or explicitly excluded at sign-off. With require_signoff: true in the
    policy, a slide without a sign-off (approve or exclude) blocks its patient.
Accounting is per patient: a patient with a missing, failed or unaudited slide is listed as blocked,
without blocking anyone else. A patient with no accepted tiles is listed as an empty bag. A CPTAC patient
listed as a calibration patient of a policy is withheld (qc_calibration): accounted for, but not in the
modeling list, so the external test never contains patients whose tiles were used to set QC cutoffs.

One modeling list serves every fold: each row carries the patient's fold, and loader.py marks rows
train / validation / external for the fold being run. Each slide contributes at most max_tiles_per_slide
(the tiling config's cap) accepted tiles, filled in this order (the `fill` column): accepted stored tissue
tiles ("stored"), then accepted tissue tiles that were not stored ("tissue"), then accepted edge tiles
("edge"); within the last two, tiles are taken in a fixed order (SHA-256 of slide_id|patch_id), so the
selection is the same on every run. Accepted tiles left over are counted (accepted_over_cap). Tiles not
stored at tiling are reread from the slide; package.py stores their pixels.

The export is complete when no patient is blocked; export_settings.json records it, and loader.py
refuses an incomplete export unless told a partial one is intended (for previews).

Writes patches/<tiling version>/export/<policy version(s), joined by "+">/<subset>_<UTC time>/: modeling_patches.csv,
patient_index.csv, slide_status.csv and export_settings.json (hashes of every input, with paths
relative to --root). loader.py refuses an export whose recorded inputs have changed since (reviews,
audits, QC decisions, fold map): rerun export.py after any of them changes.

Exit codes: 0 = exported, no patient blocked, 1 = exported but some patients are blocked,
2 = setup problem (e.g. policy not frozen, QC not run).
"""
import argparse
import csv
import hashlib
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
                "pixel_sha256", "decided_by", "fill"]
FILL_ORDER = ("stored", "tissue", "edge")

log = logging.getLogger("export")


def parse_args(argv=None):
    """Parse command-line options; pass `argv` as a list to test without a shell."""
    parser = argparse.ArgumentParser(description="Build the modeling list from QC decisions and reviews.")
    parser.add_argument("--root", type=Path, required=True, help="data folder used by tile.py")
    parser.add_argument("--subset", choices=["full", "A", "B"], required=True,
                        help="full cohort, or team subset A or B")
    parser.add_argument("--tiling-config", type=Path, default=DEFAULT_TILING_CONFIG,
                        help="tiling config whose outputs to export (default: %(default)s)")
    parser.add_argument("--policy", type=Path, nargs="+", default=[DEFAULT_POLICY],
                        help="frozen QC policy, or one per cohort (e.g. a TCGA and a CPTAC policy); "
                             "each must have been applied by qc.py (default: %(default)s)")
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


def input_hashes(root, version, policy_versions, subset, folds_path):
    """SHA-256 of every input an export depends on, keyed by path relative to `root`.

    `policy_versions` lists the version of each policy used. loader.py recomputes this to detect a
    stale export; the fold map is keyed "folds".
    """
    root = Path(root)
    records = [qc_record(root, version, v, subset) for v in policy_versions]
    grandqc_files = [root / f for r in records if r.exists()
                     for f in json.loads(r.read_text()).get("grandqc_files", [])]
    paths = [*review.review_files(root, version), *audit_files(root, version, subset), *records,
             *grandqc_files]
    hashes = {str(p.relative_to(root)): sha256_file(p) for p in paths if p.exists()}
    hashes["folds"] = sha256_file(folds_path)
    return hashes


def load_policies(paths):
    """Load frozen policies with distinct versions and non-overlapping cohorts, in cohort order.

    Returns a list of (policy, SHA-256, path).
    """
    loaded = []
    for path in paths:
        policy, sha, _ = qc.load_policy(path)
        if not policy["frozen"]:
            raise RuntimeError(f"QC policy {policy['version']} is not frozen; set frozen, reviewer "
                               f"and notes in {path} once the team agrees the cutoffs")
        loaded.append((policy, sha, path))
    versions = [p["version"] for p, _, _ in loaded]
    sources = [s for p, _, _ in loaded for s in p["sources"]]
    if len(set(versions)) != len(versions) or len(set(sources)) != len(sources):
        raise RuntimeError(f"policies must have distinct versions and cohorts: {versions}, {sources}")
    return sorted(loaded, key=lambda item: min(qc.SOURCES.index(s) for s in item[0]["sources"]))


def policy_identity(policies):
    """One SHA-256 for the set of policies an export used (what package.py and combine.py compare)."""
    return hashlib.sha256(json.dumps([[p["version"], sha] for p, sha, _ in policies]).encode()).hexdigest()


def fill_tier(tile):
    """Where an accepted tile falls in the per-slide fill order: stored tissue, other tissue, edge."""
    if tile["role"] == "edge":
        return "edge"
    return "stored" if tile["stored"] == "True" else "tissue"


def fill_slide(accepted, cap):
    """Choose at most `cap` (None = no cap) of one slide's accepted tiles, in fill order.

    Returns (chosen tiles in their original order, with "fill" set; leftover tiles).
    """
    def rank(tile):
        order = hashlib.sha256(f"{tile['slide_id']}|{tile['patch_id']}".encode()).hexdigest()
        return FILL_ORDER.index(fill_tier(tile)), order

    ranked = sorted(accepted, key=rank)
    keep = ranked if cap is None else ranked[:cap]
    chosen = {id(t) for t in keep}
    return ([{**t, "fill": fill_tier(t)} for t in accepted if id(t) in chosen],
            [t for t in accepted if id(t) not in chosen])


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
        policies = load_policies([p.expanduser().resolve() for p in args.policy])
        folds_path = args.folds.expanduser().resolve()
        folds_sha = sha256_file(folds_path)
        folds = read_folds(folds_path)
        if sha256_file(folds_path) != folds_sha:
            raise RuntimeError("fold map changed while reading")
        for policy, _, _ in policies:
            qc.validate_calibration(policy, folds)
        withheld = set().union(*(qc.withheld_patients(p, folds) for p, _, _ in policies))
        _, manifest = load_manifest(args.manifest.expanduser().resolve())
        all_files = manifest["files"]
        for policy, _, _ in policies:
            qc.validate_pen_slides(policy, all_files)
        expected = split.select_files(all_files, args.subset, split.build_patient_split(all_files))
        reviews = review.load_reviews(args.root, version)
        audits = latest_audits(args.root, version, args.subset)
        by_source = {s: policy for policy, _, _ in policies for s in policy["sources"]}
        uncovered = sorted({row["source"] for row in expected} - set(by_source))
        if uncovered:
            raise RuntimeError(f"no QC policy covers {uncovered}; pass one policy per cohort")
        index_path = args.root / "patches" / version / f"index_{args.subset}.csv"
        missing_grandqc = {}
        for policy, policy_sha, _ in policies:
            record = qc_record(args.root, version, policy["version"], args.subset)
            applied = json.loads(record.read_text()) if record.exists() else {}
            missing_grandqc.update(applied.get("grandqc_missing", {}))
            if (applied.get("policy_sha256") != policy_sha
                    or applied.get("folds_sha256") != folds_sha
                    or applied.get("index_sha256") != sha256_file(index_path)):
                raise RuntimeError(f"run qc.py with policy {policy['version']}, the current tiling index "
                                   "and the same fold map before exporting")
    except (OSError, RuntimeError, ValueError, KeyError, yaml.YAMLError) as error:
        log.error("%s", error)
        return 2
    cap = cfg["max_tiles_per_slide"]
    policy_version = "+".join(p["version"] for p, _, _ in policies)
    log.info("Tiling %s | %s | %d slides expected in subset %s | cap %s tiles per slide | log %s", version,
             "; ".join(f"policy {p['version']} for {'/'.join(p['sources'])} (frozen by {p['reviewer']})"
                       for p, _, _ in policies), len(expected), args.subset, cap, log_path)

    # 1. Slide status: is each expected slide usable, excluded, or a blocker for its patient?
    status = []
    for row in expected:
        key = (row["source"], row["slide_id"])
        signoff = reviews.signoffs.get(key, {})
        audit = audits.get(row["slide_id"])
        receipt = output_paths(args.root, row, version)[4]
        policy = by_source[row["source"]]
        if signoff.get("decision") == "exclude":
            state, why = "excluded", signoff.get("notes", "")
        elif not tiled(args.root, row, version, config_sha):
            state, why = "blocked", "not tiled with the current config"
        elif audit is None or audit[1] < json.loads(receipt.read_text())["tiled_at_utc"]:
            state, why = "blocked", "not audited since it was tiled"
        elif audit[0]["n_errors"]:
            state, why = "blocked", "failed the audit: " + "; ".join(audit[0]["errors"][:2])
        elif row["slide_id"] in missing_grandqc:
            state, why = "blocked", f"{missing_grandqc[row['slide_id']]}; run grandqc.py, then qc.py"
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
    for key in withheld & set(blocked):
        blocked[key] = []   # withheld from the test anyway, so its slides cannot block it
    included = {k for k, reasons in blocked.items() if not reasons and k not in withheld}

    # 2. Tiles: QC decision, then reviews; accepted tiles of included patients go to the model.
    stamp = datetime.now(timezone.utc)
    out = (args.root / "patches" / version / "export" / policy_version
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
            accepted = []
            for d in read_rows(qc.decisions_path(args.root, version, by_source[row["source"]]["version"], row)):
                tile = tiles[d["patch_id"]]
                seen_ids.add(d["patch_id"])
                if qc.read_scores(tile)[1]:
                    decision, decided_by = "reject", "qc"
                else:
                    decision, _, decided_by = review.resolve(tile, d["decision"], d["reasons"], reviews)
                if tile["role"] == "tissue":
                    patients[key][decision] += 1
                    patients[key]["tissue_tiles"] += 1
                else:
                    patients[key][f"edge_{decision}"] += 1
                if decision == "accept" and key in included:
                    accepted.append({**tile, "fold": folds[key], "h5": str(h5_path.relative_to(args.root)),
                                     "decided_by": decided_by})
            chosen, leftover = fill_slide(accepted, cap)
            patients[key]["over_cap"] += len(leftover)
            for tile in chosen:
                writer.writerow(tile)
                patients[key]["model_tiles"] += 1
                patients[key]["model_stored"] += tile["stored"] == "True"
                patients[key][f"fill_{tile['fill']}"] += 1
                by_whom[tile["decided_by"]] += 1
    unmatched = sorted(set(reviews.patches) - seen_ids)

    index = []
    for key, slides_ in sorted(by_patient.items()):
        c = patients[key]
        index.append(dict(source=key[0], patient_id=key[1], fold=folds.get(key, ""),
                          expected_slides=len(slides_),
                          exported_slides=sum(s["status"] == "ok" for s in slides_),
                          excluded_slides=sum(s["status"] == "excluded" for s in slides_),
                          included=key in included, qc_calibration=key in withheld,
                          tissue_tiles=c["tissue_tiles"], accept=c["accept"],
                          review=c["review"], reject=c["reject"],
                          edge_accept=c["edge_accept"], edge_review=c["edge_review"],
                          edge_reject=c["edge_reject"], model_tiles=c["model_tiles"],
                          model_tiles_stored=c["model_stored"], model_tissue_fill_tiles=c["fill_tissue"],
                          model_edge_tiles=c["fill_edge"], accepted_over_cap=c["over_cap"],
                          empty_bag=key in included and c["model_tiles"] == 0,
                          blocked_reasons="; ".join(blocked[key])))
    write_rows(out / "patient_index.csv", index)
    write_rows(out / "slide_status.csv", status)
    counts = {"patients_expected": len(index), "patients_included": len(included),
              "patients_blocked": sum(bool(r["blocked_reasons"]) for r in index),
              "patients_withheld_qc_calibration": sum(r["qc_calibration"] for r in index),
              "empty_bags": sum(r["empty_bag"] for r in index),
              "model_tiles": sum(r["model_tiles"] for r in index),
              "model_tiles_stored": sum(r["model_tiles_stored"] for r in index),
              "model_tissue_fill_tiles": sum(r["model_tissue_fill_tiles"] for r in index),
              "model_edge_tiles": sum(r["model_edge_tiles"] for r in index),
              "accepted_over_cap": sum(r["accepted_over_cap"] for r in index),
              "decided_by": dict(by_whom)}
    complete = counts["patients_blocked"] == 0
    write_json(out / "export_settings.json", {
        "subset": args.subset, "complete": complete, "created_at_utc": stamp.isoformat(),
        "tiling_version": version, "max_tiles_per_slide": cap,
        "config_sha256": config_sha, "code_version": CODE_VERSION, "policy_version": policy_version,
        "policy_sha256": policy_identity(policies),
        "policies": [{"version": p["version"], "sources": p["sources"], "sha256": sha, "policy": p}
                     for p, sha, _ in policies],
        "folds_file": folds_path.name, "folds_sha256": folds_sha,
        "inputs": input_hashes(args.root, version, [p["version"] for p, _, _ in policies], args.subset,
                               folds_path),
        "audits_used": sorted({str(a[2].relative_to(args.root)) for a in audits.values()}),
        "modeling_sha256": sha256_file(model_path), "counts": counts,
        "unmatched_patch_reviews": unmatched})

    for r in index:
        if r["blocked_reasons"]:
            log.warning("Blocked %s %s: %s", r["source"], r["patient_id"], r["blocked_reasons"])
    if unmatched:
        log.warning("%d patch reviews match no tile of an exported slide (first: %s)", len(unmatched),
                    unmatched[0])
    if counts["patients_withheld_qc_calibration"]:
        log.info("%d CPTAC QC calibration patients withheld from the external test: %s",
                 counts["patients_withheld_qc_calibration"],
                 ", ".join(r["patient_id"] for r in index if r["qc_calibration"]))
    log.info("%d of %d patients included (%d empty bags); %d model tiles: %d stored, %d unstored tissue, "
             "%d edge (%d accepted tiles beyond the cap left out); %s; written to %s",
             counts["patients_included"], counts["patients_expected"], counts["empty_bags"],
             counts["model_tiles"], counts["model_tiles_stored"], counts["model_tissue_fill_tiles"],
             counts["model_edge_tiles"], counts["accepted_over_cap"],
             "complete" if complete else "INCOMPLETE (some patients blocked)", out)
    return 1 if counts["patients_blocked"] else 0


if __name__ == "__main__":
    sys.exit(main())
