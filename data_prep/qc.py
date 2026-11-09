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
mask, not more than half tissue) never pass automatically: the same rules can reject them, and the
rest go to review with reason "edge", so a reviewer can reinstate interpretable tissue (review.py).
Tiles that were not stored (edge tiles, tissue tiles beyond the cap) are reread from the slide.
Only accepted tiles reach the model (export.py). Blur: below blur_reject_below is rejected, below
blur_review_below goes to review; with both unset, a blur score of exactly 0 (a flat tile) still
goes to review.

Exit codes: 0 = done, 2 = setup or policy problem.
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

from common import (DEFAULT_POLICY, DEFAULT_TILING_CONFIG, is_level, is_name, is_num, load_settings,
                    tiling_version, write_rows)
from download import DEFAULT_LOG_DIR, setup_logging, write_json

POLICY_KEYS = {"version", "frozen", "reviewer", "notes", "calibration_patients", "require_signoff",
               "stored_tiles_only",
               "max_glass_fraction", "blank_fraction", "blur_reject_below", "blur_review_below",
               "dark_mean_rgb", "glass_review_above", "pale_review_below", "pale_value_above",
               "dark_fraction_review_above", "green_fraction_review_above"}
DECISION_FIELDS = ["patch_id", "role", "stored", "h5_index", "grid_index", "x_target", "y_target",
                   "decision", "reasons"]
ROLES = ("tissue", "edge")
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
    parser.add_argument("--tiling-config", type=Path, default=DEFAULT_TILING_CONFIG,
                        help="tiling config whose outputs to read (default: %(default)s)")
    parser.add_argument("--log-dir", type=Path, default=DEFAULT_LOG_DIR,
                        help="folder for run .log files (default: %(default)s)")
    return parser.parse_args(argv)


def load_policy(path=DEFAULT_POLICY):
    """Read and validate the QC policy; return (policy dict, file SHA-256, file text).

    A frozen policy must name its reviewer, say in `notes` how its cutoffs were chosen, list its
    calibration patients, and set both blur cutoffs. export.py checks the calibration patients
    against the fold map (TCGA only).
    """
    def optional(check):
        return lambda v, s: v is None or check(v)

    def fraction(v):
        return is_num(v) and 0 <= v <= 1

    def blur(v, s):
        if v is None:
            return not s["frozen"]
        return is_num(v) and v >= 0

    def patients(v, s):
        return (isinstance(v, list) and all(isinstance(p, str) and p.strip() for p in v)
                and len(set(v)) == len(v) and (len(v) > 0 or not s["frozen"]))

    checks = {
        "version": lambda v, s: is_name(v),
        "frozen": lambda v, s: isinstance(v, bool),
        "reviewer": lambda v, s: isinstance(v, str) and (v.strip() != "" or not s["frozen"]),
        "notes": lambda v, s: isinstance(v, str) and (v.strip() != "" or not s["frozen"]),
        "calibration_patients": patients,
        "require_signoff": lambda v, s: isinstance(v, bool),
        "stored_tiles_only": lambda v, s: isinstance(v, bool),
        "max_glass_fraction": lambda v, s: fraction(v),
        "blank_fraction": lambda v, s: is_num(v) and 0 < v <= 1,
        "blur_reject_below": blur,
        "blur_review_below": lambda v, s: blur(v, s) and (
            v is None or s["blur_reject_below"] is None or v >= s["blur_reject_below"]),
        "dark_mean_rgb": optional(lambda v: is_num(v) and 0 <= v <= 255),
        "glass_review_above": optional(fraction),
        "pale_review_below": optional(lambda v: is_num(v) and 0 <= v <= 255),
        "pale_value_above": lambda v, s: is_level(v) or (is_num(v) and 0 <= v <= 255),
        "dark_fraction_review_above": optional(fraction),
        "green_fraction_review_above": optional(fraction),
    }
    return load_settings(path, POLICY_KEYS, checks, "policy")


def decide(row, policy):
    """Return (decision, reasons) for one tile's scores. Any reject reason outranks review.

    An edge tile (row["role"] == "edge") that no rule rejects goes to review with reason "edge".
    """
    reject, review = [], []
    score = {k: float(row[k]) for k in ("glass_fraction", "near_white_fraction", "near_black_fraction",
                                        "laplacian_variance", "mean_r", "mean_g", "mean_b",
                                        "mean_saturation", "mean_value", "dark_fraction",
                                        "green_fraction")}
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
    if row.get("role") == "edge":
        review.insert(0, "edge")
    decision = "reject" if reject else "review" if review else "accept"
    return decision, ";".join(reject + review) or "pass"


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
        version = tiling_version(args.tiling_config.expanduser().resolve())
        tiled_dir = args.root / "patches" / version
        index_path = tiled_dir / f"index_{args.subset}.csv"
        with index_path.open(newline="") as handle:
            slides = list(csv.DictReader(handle))
        if not slides:
            raise RuntimeError(f"no tiled slides in {index_path}")
    except (OSError, RuntimeError, KeyError, yaml.YAMLError) as error:
        log.error("%s", error)
        return 2
    log.info("Policy %s (sha256 %s): %s | tiling %s, %d slides | log %s", args.policy,
             policy_sha[:12], json.dumps(policy, sort_keys=True), version, len(slides), log_path)
    if policy["blur_reject_below"] is None and policy["blur_review_below"] is None:
        log.warning("blur cutoffs are null: only flat tiles (blur score 0) are sent to review for blur")
    if not policy["frozen"]:
        log.warning("policy is not frozen: export.py will not build a modeling list from it")

    out = tiled_dir / "qc" / policy["version"]
    totals, summary, queue = Counter(), [], []
    for slide in slides:
        patches_csv = args.root / slide["csv"]
        if patches_csv.stat().st_size != int(slide["csv_bytes"]):
            log.error("%s changed since tiling; run tile.py --audit", patches_csv)
            return 2
        counts, reasons = {role: Counter() for role in ROLES}, Counter()
        path = decisions_path(args.root, version, policy["version"], slide)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + ".tmp")
        with patches_csv.open(newline="") as source, temporary.open("w", newline="") as target:
            writer = csv.DictWriter(target, fieldnames=DECISION_FIELDS, extrasaction="ignore")
            writer.writeheader()
            for row in csv.DictReader(source):
                if row["role"] not in ROLES:
                    continue
                row["decision"], row["reasons"] = decide(row, policy)
                writer.writerow(row)
                counts[row["role"]][row["decision"]] += 1
                if row["role"] == "tissue":
                    reasons.update(r for r in row["reasons"].split(";") if r != "pass")
                    if row["decision"] != "accept":
                        queue.append({k: row[k] for k in QUEUE_FIELDS})
        temporary.replace(path)
        for role in ROLES:
            expected = slide.get(f"n_{role}_tiles", "")
            if expected != "" and sum(counts[role].values()) != int(expected):
                log.error("%s: %d %s rows, index says %s", slide["slide_id"], sum(counts[role].values()),
                          role, expected)
                return 2
        totals.update(counts["tissue"])
        totals.update({f"edge_{k}": v for k, v in counts["edge"].items()})
        summary.append({"source": slide["source"], "patient_id": slide["patient_id"],
                        "slide_id": slide["slide_id"], "h5": slide["h5"],
                        "decisions": str(path.relative_to(args.root)),
                        "tissue_tiles": slide["n_tissue_tiles"], "stored_tiles": slide["n_stored_tiles"],
                        "accept": counts["tissue"]["accept"], "review": counts["tissue"]["review"],
                        "reject": counts["tissue"]["reject"],
                        "reasons": json.dumps(dict(sorted(reasons.items()))),
                        "edge_tiles": sum(counts["edge"].values()), "edge_review": counts["edge"]["review"],
                        "edge_reject": counts["edge"]["reject"]})

    with (out / f"slides_{args.subset}.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    queue.sort(key=lambda r: (r["decision"] != "review", r["source"], r["patient_id"], r["slide_id"],
                              int(r["y_target"]), int(r["x_target"])))
    write_rows(out / f"review_queue_{args.subset}.csv", queue, QUEUE_FIELDS)
    write_json(out / f"policy_{args.subset}.json", {
        "policy": policy, "policy_sha256": policy_sha, "policy_yaml": policy_text,
        "tiling_version": version, "index_sha256": hashlib.sha256(index_path.read_bytes()).hexdigest(),
        "counts": dict(totals), "applied_at_utc": datetime.now(timezone.utc).isoformat()})
    log.info("%d slides: tissue tiles %d accept, %d review, %d reject; edge tiles %d review, "
             "%d reject; written to %s (review queue: %d tissue tiles)", len(summary), totals["accept"],
             totals["review"], totals["reject"], totals["edge_review"], totals["edge_reject"], out,
             len(queue))
    return 0


if __name__ == "__main__":
    sys.exit(main())
