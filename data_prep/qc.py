"""Apply the QC policy to tiled slides: an accept / review / reject decision for every scored tile.

Examples:
    python data_prep/qc.py --root ~/Bioinformatics/WSI/data --subset B
    python data_prep/qc.py --root ~/Bioinformatics/WSI/data --subset B --policy configs/qc_policy.yaml

tile.py stores every tissue tile's scores and makes no quality decisions. This script reads
patches/<tiling version>/index_<subset>.csv and each slide's _patches.csv, applies
configs/qc_policy.yaml, and writes under patches/<tiling version>/qc/<policy version>/:
  {source}/{patient_id}/{slide_id}_decisions.csv   one row per tissue or edge tile
  slides_<subset>.csv                              decision counts per slide (tissue, edge)
  review_queue_<subset>.csv                        every tissue tile sent to review or rejected, with its
                                                   coordinates and key scores, for the reviewer
  policy_<subset>.json                             the policy, its SHA-256 and the inputs used
Tissue tiles are accepted unless a rule rejects them or sends them to review. Edge tiles (touching the
mask, not more than half tissue) pass only when no rule rejects or reviews them and they meet the three
edge_accept_* cutoffs (little glass, in focus, few dark pixels; reason "edge_accepted"); the rest go to
review with reason "edge", so a reviewer can reinstate interpretable tissue (review.py). Tiles that were
not stored (edge tiles, tissue tiles beyond the cap) are reread from the slide. Only accepted tiles reach
the model (export.py). Blur: below blur_reject_below is rejected, below blur_review_below goes to
review; with both unset, a blur score of exactly 0 (a flat tile) still goes to review.

A policy applies to the cohorts in its `sources` (TCGA, CPTAC or both); by default one policy serves
every fold and both cohorts, and qc.py decides only the slides of those cohorts. To give CPTAC its own
policy, run qc.py once with a TCGA policy and once with a CPTAC policy, and pass both to export.py.
Calibration patients must be patients of the policy's cohorts in the fold map. TCGA ones may sit in any
fold (a small label-free leak accepted in exchange for a single policy, so freeze the policy before any
training); CPTAC ones are withheld from the external test by export.py. slides_<subset>.csv flags
slides for a sign-off decision (slide_flags): "pale" (median tissue-tile saturation below
pale_slide_saturation_below), "no_tissue" (no tissue tiles) and "mask_floor" (tile.py raised the tissue
mask threshold to its floor).

GrandQC (grandqc: true): each slide's per-tile artifact fractions from grandqc.py add review reasons
gq_oof (out of focus above oof_review_above), gq_pen (pen above pen_tile_review_above, only on slides
whose pen fraction is above pen_slide_above), gq_fold and gq_darkspot (dark spot not on pale slides);
an edge tile is accepted only if its fold, dark spot and out-of-focus fractions are each at or below
edge_accept_artifact_below. A tile with less than grandqc_min_assessed of its area assessed skips these
rules (an edge tile then goes to review). A slide without current GrandQC data gets no decisions, so
export.py blocks its patient; the QC receipt lists those slides and the hash inputs of the rest.

Exit codes: 0 = done, 1 = some slides had no GrandQC data, 2 = setup or policy problem.
"""
import argparse
import csv
import hashlib
import json
import logging
import math
import statistics
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import yaml

from common import (DEFAULT_FOLDS, DEFAULT_POLICY, DEFAULT_TILING_CONFIG, is_level, is_name, is_num, load_settings,
                    read_folds, sha256_file, tiling_version, write_rows)
from download import DEFAULT_LOG_DIR, setup_logging, write_json
import grandqc

GRANDQC_KEYS = ("grandqc", "grandqc_min_assessed", "oof_review_above", "pen_slide_above",
                "pen_tile_review_above", "fold_review_above", "darkspot_review_above",
                "edge_accept_artifact_below")
EDGE_ACCEPT_KEYS = ("edge_accept_glass_below", "edge_accept_blur_above", "edge_accept_dark_below")
SOURCES = ("TCGA", "CPTAC")
POLICY_KEYS = {"version", "sources", "frozen", "reviewer", "notes", "calibration_patients", "require_signoff",
               "max_glass_fraction", "blank_fraction", "blur_reject_below", "blur_review_below",
               "dark_mean_rgb", "glass_review_above", "pale_review_below", "pale_value_above",
               "dark_fraction_review_above", "green_fraction_review_above", *EDGE_ACCEPT_KEYS,
               "pale_slide_saturation_below", *GRANDQC_KEYS}
DECISION_FIELDS = ["patch_id", "role", "stored", "h5_index", "grid_index", "x_target", "y_target",
                   "decision", "reasons"]
ROLES = ("tissue", "edge")
FRACTION_SCORES = ("glass_fraction", "near_white_fraction", "near_black_fraction",
                   "dark_fraction", "green_fraction")
CHANNEL_SCORES = ("mean_r", "mean_g", "mean_b", "mean_saturation", "mean_value", "gray_median")
SCORE_MAXIMA = {**dict.fromkeys(FRACTION_SCORES, 1), **dict.fromkeys(CHANNEL_SCORES, 255),
                "laplacian_variance": math.inf}
INVALID_SCORE = "invalid_score"
QUEUE_FIELDS = ["source", "patient_id", "slide_id", "patch_id", "decision", "reasons", "stored", "h5_index",
                "x_target", "y_target", "x0", "y0", "x1", "y1", "laplacian_variance", "mean_saturation",
                "mean_value", "glass_fraction", "dark_fraction", "green_fraction"]

log = logging.getLogger("qc")


def parse_args(argv=None):
    """Parse command-line options; pass `argv` as a list to test without a shell."""
    parser = argparse.ArgumentParser(description="Apply the QC policy to tiled slides.")
    parser.add_argument("--root", type=Path, required=True, help="data folder used by tile.py")
    parser.add_argument("--subset", choices=["full", "A", "B"], required=True,
                        help="the subset whose tiling index to read")
    parser.add_argument("--policy", type=Path, default=DEFAULT_POLICY,
                        help="QC policy (default: %(default)s)")
    parser.add_argument("--folds", type=Path, default=DEFAULT_FOLDS, help="shared patient fold map")
    parser.add_argument("--tiling-config", type=Path, default=DEFAULT_TILING_CONFIG,
                        help="tiling config whose outputs to read (default: %(default)s)")
    parser.add_argument("--log-dir", type=Path, default=DEFAULT_LOG_DIR,
                        help="folder for run .log files (default: %(default)s)")
    return parser.parse_args(argv)


def load_policy(path=DEFAULT_POLICY):
    """Read and validate the QC policy; return (policy dict, file SHA-256, file text).

    A frozen policy must name its reviewer, say in `notes` how its cutoffs were chosen, list its
    TCGA calibration patients and set blur_reject_below. The edge_accept_* cutoffs are set together or
    all null (edge tiles then always go to review). With grandqc: true, grandqc_min_assessed is required
    and the pen cutoffs are set together; with grandqc: false the GrandQC cutoffs are ignored.
    """
    def optional(check):
        return lambda v, s: v is None or check(v)

    def fraction(v):
        return is_num(v) and 0 <= v <= 1

    def blur(v, s):
        if v is None:
            return not s["frozen"]
        return is_num(v) and v >= 0

    def edge(check):
        return lambda v, s: (all(s[k] is None for k in EDGE_ACCEPT_KEYS)
                             or (all(s[k] is not None for k in EDGE_ACCEPT_KEYS) and check(v)))

    def patients(v, s):
        return (isinstance(v, list) and all(isinstance(p, str) and p.strip() for p in v)
                and len(set(v)) == len(v) and (len(v) > 0 or not s["frozen"]))

    checks = {
        "version": lambda v, s: is_name(v),
        "sources": lambda v, s: (isinstance(v, list) and len(v) > 0 and len(set(v)) == len(v)
                                 and all(x in SOURCES for x in v)),
        "frozen": lambda v, s: isinstance(v, bool),
        "reviewer": lambda v, s: isinstance(v, str) and (v.strip() != "" or not s["frozen"]),
        "notes": lambda v, s: isinstance(v, str) and (v.strip() != "" or not s["frozen"]),
        "calibration_patients": patients,
        "require_signoff": lambda v, s: isinstance(v, bool),
        "max_glass_fraction": lambda v, s: fraction(v),
        "blank_fraction": lambda v, s: is_num(v) and 0 < v <= 1,
        "blur_reject_below": blur,
        "blur_review_below": lambda v, s: v is None or (
            is_num(v) and v >= 0 and (s["blur_reject_below"] is None
                                      or (is_num(s["blur_reject_below"]) and v >= s["blur_reject_below"]))),
        "dark_mean_rgb": optional(lambda v: is_num(v) and 0 <= v <= 255),
        "glass_review_above": optional(fraction),
        "pale_review_below": optional(lambda v: is_num(v) and 0 <= v <= 255),
        "pale_value_above": lambda v, s: is_level(v) or (is_num(v) and 0 <= v <= 255),
        "dark_fraction_review_above": optional(fraction),
        "green_fraction_review_above": optional(fraction),
        "edge_accept_glass_below": edge(fraction),
        "edge_accept_blur_above": edge(lambda v: is_num(v) and v >= 0),
        "edge_accept_dark_below": edge(fraction),
        "pale_slide_saturation_below": optional(lambda v: is_num(v) and 0 <= v <= 255),
        "grandqc": lambda v, s: isinstance(v, bool),
        "grandqc_min_assessed": lambda v, s: (v is None and not s["grandqc"]) or fraction(v),
        "oof_review_above": optional(fraction),
        "pen_slide_above": lambda v, s: ((v is None) == (s["pen_tile_review_above"] is None)
                                         and (v is None or fraction(v))),
        "pen_tile_review_above": optional(fraction),
        "fold_review_above": optional(fraction),
        "darkspot_review_above": optional(fraction),
        "edge_accept_artifact_below": optional(fraction),
    }
    return load_settings(path, POLICY_KEYS, checks, "policy")


def calibration_patients(folds, sources=("TCGA",)):
    """Patient IDs in the fold map from `sources` that may calibrate a policy for those cohorts."""
    return {patient for (source, patient) in folds if source in sources}


def validate_calibration(policy, folds):
    """Refuse calibration patients that are not in the fold map or not from the policy's cohorts."""
    invalid = sorted(set(policy["calibration_patients"]) - calibration_patients(folds, policy["sources"]))
    if invalid:
        raise ValueError(f"calibration patients not in the fold map for {policy['sources']}: {invalid}")


def withheld_patients(policy, folds):
    """External (CPTAC) calibration patients: export.py keeps them out of the external test."""
    return {(source, patient) for (source, patient), fold in folds.items()
            if fold == "external" and patient in policy["calibration_patients"]}


def read_scores(row):
    """Parse scores and report missing, malformed, nonfinite or out-of-range values."""
    scores, invalid = {}, []
    for name, maximum in SCORE_MAXIMA.items():
        try:
            value = float(row[name])
        except (KeyError, TypeError, ValueError, OverflowError):
            invalid.append(name)
            continue
        if not math.isfinite(value) or not 0 <= value <= maximum:
            invalid.append(name)
        else:
            scores[name] = value
    return scores, invalid


def decide(row, policy, gq=None):
    """Return (decision, reasons) for one tile's scores. Any reject reason outranks review.

    An edge tile (row["role"] == "edge") that no rule rejects or reviews is accepted with reason
    "edge_accepted" when it meets every edge_accept_* cutoff; otherwise it goes to review with reason "edge".

    With grandqc: true in the policy, `gq` holds the tile's GrandQC fractions (grandqc.py) plus the
    slide's "pen_slide" and "pale_slide" flags (see grandqc_context). A tile with less than
    grandqc_min_assessed of its area assessed (or no `gq`) skips the GrandQC rules; such an edge tile is
    not accepted automatically (reason "grandqc_unassessed").
    """
    reject, review = [], []
    score, invalid = read_scores(row)
    if invalid:
        return "reject", INVALID_SCORE + ":" + ",".join(invalid)
    if score["glass_fraction"] > policy["max_glass_fraction"]:
        reject.append("glass")
    if score["near_white_fraction"] >= policy["blank_fraction"]:
        reject.append("near_white")
    if score["near_black_fraction"] >= policy["blank_fraction"]:
        reject.append("near_black")
    focus = score["laplacian_variance"]
    if policy["blur_reject_below"] is not None and focus < policy["blur_reject_below"]:
        reject.append("blur")
    elif policy["blur_review_below"] is not None and focus < policy["blur_review_below"]:
        review.append("low_blur")
    elif focus == 0:
        review.append("flat_tile")
    dark = policy["dark_mean_rgb"]
    if dark is not None and all(score[k] < dark for k in ("mean_r", "mean_g", "mean_b")):
        review.append("very_dark")
    if policy["glass_review_above"] is not None and score["glass_fraction"] > policy["glass_review_above"]:
        review.append("glass_review")
    if (policy["pale_review_below"] is not None and score["mean_saturation"] < policy["pale_review_below"]
            and score["mean_value"] > policy["pale_value_above"]):
        review.append("pale")
    if (policy["dark_fraction_review_above"] is not None
            and score["dark_fraction"] > policy["dark_fraction_review_above"]):
        review.append("dark_area")
    if (policy["green_fraction_review_above"] is not None
            and score["green_fraction"] > policy["green_fraction_review_above"]):
        review.append("green_ink")
    assessed = True
    if policy["grandqc"]:
        assessed = gq is not None and gq["assessed"] >= policy["grandqc_min_assessed"]
        if assessed:
            review.extend(grandqc_reasons(gq, policy))
    if row.get("role") == "edge":
        if (not reject and not review and edge_passes(score, policy)
                and (not policy["grandqc"] or (assessed and edge_artifacts_pass(gq, policy)))):
            return "accept", "edge_accepted"
        review.insert(0, "edge")
        if not assessed:
            review.append("grandqc_unassessed")
    decision = "reject" if reject else "review" if review else "accept"
    return decision, ";".join(reject + review) or "pass"


def grandqc_reasons(gq, policy):
    """GrandQC review reasons for an assessed tile: gq_oof, gq_pen (pen slides only), gq_fold, gq_darkspot."""
    reasons = []
    if policy["oof_review_above"] is not None and gq["oof"] > policy["oof_review_above"]:
        reasons.append("gq_oof")
    if (policy["pen_tile_review_above"] is not None and gq["pen_slide"]
            and gq["pen"] > policy["pen_tile_review_above"]):
        reasons.append("gq_pen")
    if policy["fold_review_above"] is not None and gq["fold"] > policy["fold_review_above"]:
        reasons.append("gq_fold")
    if (policy["darkspot_review_above"] is not None and not gq["pale_slide"]
            and gq["darkspot"] > policy["darkspot_review_above"]):
        reasons.append("gq_darkspot")
    return reasons


def edge_artifacts_pass(gq, policy):
    """Whether an edge tile's GrandQC fold, dark spot and out-of-focus fractions are each at or below
    edge_accept_artifact_below (True when that cutoff is null)."""
    cutoff = policy["edge_accept_artifact_below"]
    return cutoff is None or all(gq[k] <= cutoff for k in ("fold", "darkspot", "oof"))


def grandqc_context(root, version, slide, policy, median_saturation):
    """Return ({patch_id: fractions + slide flags}, receipt, "") for a slide, or (None, None, reason).

    The slide is a pen slide when its pen fraction is above pen_slide_above, and a pale slide when its
    median tissue-tile saturation is below pale_slide_saturation_below.
    """
    receipt, why = grandqc.current(root, version, slide)
    if receipt is None:
        return None, None, why
    pen_cutoff, pale_cutoff = policy["pen_slide_above"], policy["pale_slide_saturation_below"]
    flags = {"pen_slide": pen_cutoff is not None and receipt["pen_fraction"] > pen_cutoff,
             "pale_slide": (pale_cutoff is not None and median_saturation is not None
                            and median_saturation < pale_cutoff)}
    return ({k: {**v, **flags} for k, v in grandqc.read_fractions(root, version, slide).items()},
            receipt, "")


def tissue_saturation(path):
    """Median mean_saturation of a tiling CSV's tissue tiles (None when there are none)."""
    values = []
    with Path(path).open(newline="") as handle:
        for row in csv.DictReader(handle):
            if row["role"] == "tissue":
                try:
                    values.append(float(row["mean_saturation"]))
                except (KeyError, TypeError, ValueError):
                    pass
    return statistics.median(values) if values else None


def edge_passes(score, policy):
    """Whether an edge tile meets the edge_accept_* cutoffs (always False when they are null)."""
    if policy["edge_accept_glass_below"] is None:
        return False
    return (score["glass_fraction"] < policy["edge_accept_glass_below"]
            and score["laplacian_variance"] >= policy["edge_accept_blur_above"]
            and score["dark_fraction"] < policy["edge_accept_dark_below"])


def slide_flags(policy, median_saturation, tissue_tiles, mask_floor_applied):
    """Reasons a slide needs a sign-off decision: pale, no_tissue, mask_floor (empty when none apply)."""
    flags = []
    cutoff = policy["pale_slide_saturation_below"]
    if cutoff is not None and median_saturation is not None and median_saturation < cutoff:
        flags.append("pale")
    if tissue_tiles == 0:
        flags.append("no_tissue")
    if mask_floor_applied:
        flags.append("mask_floor")
    return flags


def summary_row(root, slide, path, counts, reasons, median_saturation, flags, pen_fraction, grandqc_status,
                unassessed):
    """One row of slides_<subset>.csv (path None: the slide got no decisions)."""
    return {"source": slide["source"], "patient_id": slide["patient_id"],
            "slide_id": slide["slide_id"], "h5": slide["h5"],
            "decisions": "" if path is None else str(path.relative_to(root)),
            "tissue_tiles": slide["n_tissue_tiles"], "stored_tiles": slide["n_stored_tiles"],
            "accept": counts["tissue"]["accept"], "review": counts["tissue"]["review"],
            "reject": counts["tissue"]["reject"],
            "reasons": json.dumps(dict(sorted(reasons.items()))),
            "edge_tiles": sum(counts["edge"].values()), "edge_accept": counts["edge"]["accept"],
            "edge_review": counts["edge"]["review"], "edge_reject": counts["edge"]["reject"],
            "median_mean_saturation": "" if median_saturation is None else round(median_saturation, 2),
            "mask_floor_applied": slide.get("mask_floor_applied", ""),
            "slide_flags": ";".join(flags), "grandqc": grandqc_status, "pen_fraction": pen_fraction,
            "grandqc_unassessed_tissue": unassessed}


def decisions_path(root, version, policy_version, slide):
    """Where qc.py writes one slide's decisions."""
    return (Path(root) / "patches" / version / "qc" / policy_version / slide["source"]
            / slide["patient_id"] / f"{slide['slide_id']}_decisions.csv")


def main(argv=None):
    """Apply the policy to every slide in the tiling index. Returns the process exit code."""
    args = parse_args(argv)
    args.root = args.root.expanduser().resolve()
    log_path = setup_logging(log, args.log_dir.expanduser().resolve(), f"qc_{args.subset}")
    log.info("Command: %s", " ".join(sys.argv))
    try:
        policy, policy_sha, policy_text = load_policy(args.policy.expanduser().resolve())
        folds_path = args.folds.expanduser().resolve()
        fold_sha = sha256_file(folds_path)
        validate_calibration(policy, read_folds(folds_path))
        if sha256_file(folds_path) != fold_sha:
            raise RuntimeError("fold map changed while reading")
        version = tiling_version(args.tiling_config.expanduser().resolve())
        tiled_dir = args.root / "patches" / version
        index_path = tiled_dir / f"index_{args.subset}.csv"
        with index_path.open(newline="") as handle:
            slides = [r for r in csv.DictReader(handle) if r["source"] in policy["sources"]]
        out = tiled_dir / "qc" / policy["version"]
        for receipt in out.glob("policy_*.json"):
            prior = json.loads(receipt.read_text())
            if (policy["frozen"] or prior["policy"].get("frozen")) and (
                    prior["policy_sha256"] != policy_sha or prior.get("folds_sha256") != fold_sha):
                raise RuntimeError(f"{out} already contains another policy or fold map; bump policy version")
        if not slides:
            raise RuntimeError(f"no tiled {'/'.join(policy['sources'])} slides in {index_path}")
    except (OSError, RuntimeError, ValueError, KeyError, yaml.YAMLError) as error:
        log.error("%s", error)
        return 2
    log.info("Policy %s (sha256 %s): %s | tiling %s, %d %s slides | log %s", args.policy,
             policy_sha[:12], json.dumps(policy, sort_keys=True), version, len(slides),
             "/".join(policy["sources"]), log_path)
    if policy["blur_reject_below"] is None and policy["blur_review_below"] is None:
        log.warning("blur cutoffs are null: only flat tiles (blur score 0) are sent to review for blur")
    if not policy["frozen"]:
        log.warning("policy is not frozen: export.py will not build a modeling list from it")

    totals, summary, queue = Counter(), [], []
    grandqc_files, grandqc_missing = [], {}
    for slide in slides:
        patches_csv = args.root / slide["csv"]
        if patches_csv.stat().st_size != int(slide["csv_bytes"]):
            log.error("%s changed since tiling; run tile.py --audit", patches_csv)
            return 2
        counts, reasons, saturation = {role: Counter() for role in ROLES}, Counter(), []
        path = decisions_path(args.root, version, policy["version"], slide)
        path.parent.mkdir(parents=True, exist_ok=True)
        gq_tiles, pen_fraction, unassessed = None, "", 0
        if policy["grandqc"]:
            gq_tiles, gq_receipt, why = grandqc_context(args.root, version, slide, policy,
                                                        tissue_saturation(patches_csv))
            if gq_tiles is None:
                grandqc_missing[slide["slide_id"]] = why
                path.unlink(missing_ok=True)   # no decisions, so export blocks the patient
                summary.append(summary_row(args.root, slide, None, counts, reasons, None, [], "", why, 0))
                continue
            pen_fraction = round(gq_receipt["pen_fraction"], 5)
            grandqc_files += [str(p.relative_to(args.root))
                              for p in grandqc.output_paths(args.root, version, slide)]
        temporary = path.with_name(path.name + ".tmp")
        with patches_csv.open(newline="") as source, temporary.open("w", newline="") as target:
            writer = csv.DictWriter(target, fieldnames=DECISION_FIELDS, extrasaction="ignore")
            writer.writeheader()
            for row in csv.DictReader(source):
                if row["role"] not in ROLES:
                    continue
                gq = gq_tiles.get(row["patch_id"]) if gq_tiles is not None else None
                row["decision"], row["reasons"] = decide(row, policy, gq)
                if (policy["grandqc"] and row["role"] == "tissue"
                        and (gq is None or gq["assessed"] < policy["grandqc_min_assessed"])):
                    unassessed += 1
                writer.writerow(row)
                counts[row["role"]][row["decision"]] += 1
                if row["role"] == "tissue":
                    try:
                        saturation.append(float(row["mean_saturation"]))
                    except (KeyError, TypeError, ValueError):
                        pass
                    reasons.update(r for r in row["reasons"].split(";") if r != "pass")
                    if row["decision"] != "accept":
                        queue.append({k: row.get(k, "") for k in QUEUE_FIELDS})
        temporary.replace(path)
        for role in ROLES:
            expected = slide.get(f"n_{role}_tiles", "")
            if expected != "" and sum(counts[role].values()) != int(expected):
                log.error("%s: %d %s rows, index says %s", slide["slide_id"], sum(counts[role].values()),
                          role, expected)
                return 2
        totals.update(counts["tissue"])
        totals.update({f"edge_{k}": v for k, v in counts["edge"].items()})
        median_saturation = statistics.median(saturation) if saturation else None
        flags = slide_flags(policy, median_saturation, sum(counts["tissue"].values()),
                            slide.get("mask_floor_applied") == "True")
        summary.append(summary_row(args.root, slide, path, counts, reasons, median_saturation, flags,
                                   pen_fraction, "ok" if policy["grandqc"] else "", unassessed))

    with (out / f"slides_{args.subset}.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    queue.sort(key=lambda r: (r["decision"] != "review", r["source"], r["patient_id"], r["slide_id"],
                              int(r["y_target"]), int(r["x_target"])))
    write_rows(out / f"review_queue_{args.subset}.csv", queue, QUEUE_FIELDS)
    write_json(out / f"policy_{args.subset}.json", {
        "policy": policy, "policy_sha256": policy_sha, "policy_yaml": policy_text,
        "folds_sha256": fold_sha,
        "tiling_version": version, "index_sha256": hashlib.sha256(index_path.read_bytes()).hexdigest(),
        "grandqc_files": grandqc_files, "grandqc_missing": grandqc_missing,
        "counts": dict(totals), "applied_at_utc": datetime.now(timezone.utc).isoformat()})
    if grandqc_missing:
        log.warning("%d slides have no usable GrandQC data and got no decisions (export blocks their "
                    "patients); run grandqc.py, then qc.py again: %s", len(grandqc_missing),
                    ", ".join(f"{k} ({v})" for k, v in grandqc_missing.items()))
    flagged = [f"{r['slide_id']} ({r['slide_flags']})" for r in summary if r["slide_flags"]]
    if flagged:
        log.warning("%d slides flagged for a sign-off decision: %s", len(flagged), ", ".join(flagged))
    log.info("%d slides: tissue tiles %d accept, %d review, %d reject; edge tiles %d accept, %d review, "
             "%d reject; written to %s (review queue: %d tissue tiles)", len(summary), totals["accept"],
             totals["review"], totals["reject"], totals["edge_accept"], totals["edge_review"],
             totals["edge_reject"], out, len(queue))
    return 1 if grandqc_missing else 0


if __name__ == "__main__":
    sys.exit(main())
