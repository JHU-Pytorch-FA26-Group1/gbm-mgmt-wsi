"""Apply the QC policy to tiled slides: an accept / review / reject decision for every tissue tile.

Examples:
    python data_prep/qc.py --root ~/Bioinformatics/WSI/data --subset B
    python data_prep/qc.py --root ~/Bioinformatics/WSI/data --subset B --policy configs/qc_policy.yaml

tile.py stores every tissue tile's scores and makes no quality decisions. This script reads
patches/<tiling version>/index_<subset>.csv and each slide's _patches.csv, applies
configs/qc_policy.yaml, and writes under patches/<tiling version>/qc/<policy version>/:
  {source}/{patient_id}/{slide_id}_decisions.csv   one row per tissue tile; stored ones carry h5_index
  slides_<subset>.csv                              accept / review / reject counts per slide
  policy_<subset>.json                             the policy, its SHA-256 and the inputs used
Edge tiles (touching the mask, not tissue) are never candidates. With a tiling cap, tissue tiles that
were not stored still get a decision and can be reread from the slide. Review cases need a person
(review.py); only accepted tiles reach the model (export.py). A blur score of exactly 0 (a flat tile)
is always sent to review, whatever the cutoffs.

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
                    tiling_version)
from download import DEFAULT_LOG_DIR, setup_logging, write_json

POLICY_KEYS = {"version", "frozen", "reviewer", "notes", "max_glass_fraction", "blank_fraction",
               "dark_mean_rgb", "min_blur", "glass_review_above", "pale_review_below",
               "pale_value_above", "dark_fraction_review_above", "green_fraction_review_above"}
DECISION_FIELDS = ["patch_id", "stored", "h5_index", "grid_index", "x_target", "y_target", "decision",
                   "reasons"]

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

    A frozen policy must name its reviewer and say in `notes` how its cutoffs were chosen.
    """
    def optional(check):
        return lambda v, s: v is None or check(v)

    def fraction(v):
        return is_num(v) and 0 <= v <= 1

    checks = {
        "version": lambda v, s: is_name(v),
        "frozen": lambda v, s: isinstance(v, bool),
        "reviewer": lambda v, s: isinstance(v, str) and (v.strip() != "" or not s["frozen"]),
        "notes": lambda v, s: isinstance(v, str) and (v.strip() != "" or not s["frozen"]),
        "max_glass_fraction": lambda v, s: fraction(v),
        "blank_fraction": lambda v, s: is_num(v) and 0 < v <= 1,
        "dark_mean_rgb": optional(lambda v: is_num(v) and 0 <= v <= 255),
        "min_blur": optional(lambda v: is_num(v) and v >= 0),
        "glass_review_above": optional(fraction),
        "pale_review_below": optional(lambda v: is_num(v) and 0 <= v <= 255),
        "pale_value_above": lambda v, s: is_level(v) or (is_num(v) and 0 <= v <= 255),
        "dark_fraction_review_above": optional(fraction),
        "green_fraction_review_above": optional(fraction),
    }
    return load_settings(path, POLICY_KEYS, checks, "policy")


def decide(row, policy):
    """Return (decision, reasons) for one tile's scores. Any reject reason outranks review."""
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
    if score["laplacian_variance"] == 0:
        review.append("flat_tile")
    elif policy["min_blur"] is not None and score["laplacian_variance"] < policy["min_blur"]:
        review.append("low_blur")
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
    if policy["min_blur"] is None:
        log.warning("min_blur is null: only flat tiles (blur score 0) are sent to review for blur")
    if not policy["frozen"]:
        log.warning("policy is not frozen: export.py will not build a modeling list from it")

    out = tiled_dir / "qc" / policy["version"]
    totals, summary = Counter(), []
    for slide in slides:
        patches_csv = args.root / slide["csv"]
        if patches_csv.stat().st_size != int(slide["csv_bytes"]):
            log.error("%s changed since tiling; run tile.py --audit", patches_csv)
            return 2
        counts, reasons = Counter(), Counter()
        path = decisions_path(args.root, version, policy["version"], slide)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + ".tmp")
        with patches_csv.open(newline="") as source, temporary.open("w", newline="") as target:
            writer = csv.DictWriter(target, fieldnames=DECISION_FIELDS, extrasaction="ignore")
            writer.writeheader()
            for row in csv.DictReader(source):
                if row["role"] != "tissue":
                    continue
                row["decision"], row["reasons"] = decide(row, policy)
                writer.writerow(row)
                counts[row["decision"]] += 1
                reasons.update(r for r in row["reasons"].split(";") if r != "pass")
        temporary.replace(path)
        if sum(counts.values()) != int(slide["n_tissue_tiles"]):
            log.error("%s: %d tissue rows, index says %s", slide["slide_id"], sum(counts.values()),
                      slide["n_tissue_tiles"])
            return 2
        totals.update(counts)
        summary.append({"source": slide["source"], "patient_id": slide["patient_id"],
                        "slide_id": slide["slide_id"], "h5": slide["h5"],
                        "decisions": str(path.relative_to(args.root)),
                        "tissue_tiles": slide["n_tissue_tiles"], "stored_tiles": slide["n_stored_tiles"],
                        "accept": counts["accept"], "review": counts["review"],
                        "reject": counts["reject"], "reasons": json.dumps(dict(sorted(reasons.items())))})

    with (out / f"slides_{args.subset}.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    write_json(out / f"policy_{args.subset}.json", {
        "policy": policy, "policy_sha256": policy_sha, "policy_yaml": policy_text,
        "tiling_version": version, "index_sha256": hashlib.sha256(index_path.read_bytes()).hexdigest(),
        "counts": dict(totals), "applied_at_utc": datetime.now(timezone.utc).isoformat()})
    log.info("%d slides: %d accept, %d review, %d reject; written to %s", len(summary),
             totals["accept"], totals["review"], totals["reject"], out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
