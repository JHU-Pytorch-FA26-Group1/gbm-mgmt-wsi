"""Apply the QC policy to tiled slides: an accept / review / reject decision for every tissue tile.

Examples:
    python data_prep/qc.py --root ~/Bioinformatics/WSI/data --subset B
    python data_prep/qc.py --root ~/Bioinformatics/WSI/data --subset B --policy configs/qc_policy.yaml

tile.py stores every tissue tile with its scores and makes no quality decisions. This script reads
patches/<tiling version>/index_<subset>.csv and each slide's _patches.csv, applies
configs/qc_policy.yaml, and writes under patches/<tiling version>/qc/<policy version>/:
  {source}/{patient_id}/{slide_id}_decisions.csv   one row per tissue tile; stored ones carry h5_index
  slides_<subset>.csv                              accept / review / reject counts per slide
  policy_<subset>.json                             the policy, its SHA-256 and the inputs used
Edge tiles (touching the mask, not tissue) are never candidates. With a tiling cap, tissue tiles that
were not stored still get a decision and can be reread from the slide. Review cases need a person.

Exit codes: 0 = done, 2 = setup or policy problem.
"""
import argparse
import csv
import hashlib
import json
import logging
import math
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import yaml

from download import DEFAULT_LOG_DIR, REPO_ROOT, setup_logging, write_json

DEFAULT_POLICY = REPO_ROOT / "configs" / "qc_policy.yaml"
DEFAULT_TILING_CONFIG = REPO_ROOT / "configs" / "tiling.yaml"
POLICY_KEYS = {"version", "max_glass_fraction", "blank_fraction", "dark_mean_rgb", "min_blur"}
DECISION_FIELDS = ["patch_id", "stored", "h5_index", "grid_index", "x_target", "y_target", "decision", "reasons"]

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


def load_policy(path):
    """Read and validate the QC policy; return (policy dict, file SHA-256, file text)."""
    text = path.read_text()
    policy = yaml.safe_load(text)
    if not isinstance(policy, dict):
        raise RuntimeError(f"policy is not a YAML mapping: {path}")
    if set(policy) != POLICY_KEYS:
        raise RuntimeError(f"policy keys {sorted(set(policy) ^ POLICY_KEYS)} missing or unexpected")

    def is_num(v):
        return (isinstance(v, int) and not isinstance(v, bool)) or (isinstance(v, float) and math.isfinite(v))

    v = policy
    checks = {
        "version": isinstance(v["version"], str) and v["version"].strip() != ""
                   and Path(v["version"]).name == v["version"] and v["version"] not in (".", ".."),
        "max_glass_fraction": is_num(v["max_glass_fraction"]) and 0 <= v["max_glass_fraction"] <= 1,
        "blank_fraction": is_num(v["blank_fraction"]) and 0 < v["blank_fraction"] <= 1,
        "dark_mean_rgb": v["dark_mean_rgb"] is None or (is_num(v["dark_mean_rgb"])
                                                        and 0 <= v["dark_mean_rgb"] <= 255),
        "min_blur": v["min_blur"] is None or (is_num(v["min_blur"]) and v["min_blur"] >= 0),
    }
    bad = [key for key, ok in checks.items() if not ok]
    if bad:
        raise RuntimeError("bad policy values: " + ", ".join(f"{k}={policy[k]!r}" for k in bad))
    return policy, hashlib.sha256(text.encode()).hexdigest(), text


def decide(row, policy):
    """Return (decision, reasons) for one tile's scores. Any reject reason outranks review."""
    reject, review = [], []
    if float(row["glass_fraction"]) > policy["max_glass_fraction"]:
        reject.append("glass")
    if float(row["near_white_fraction"]) >= policy["blank_fraction"]:
        reject.append("near_white")
    if float(row["near_black_fraction"]) >= policy["blank_fraction"]:
        reject.append("near_black")
    dark = policy["dark_mean_rgb"]
    if dark is not None and all(float(row[k]) < dark for k in ("mean_r", "mean_g", "mean_b")):
        review.append("very_dark")
    if policy["min_blur"] is not None and float(row["laplacian_variance"]) < policy["min_blur"]:
        review.append("low_blur")
    decision = "reject" if reject else "review" if review else "accept"
    return decision, ";".join(reject + review) or "pass"


def main(argv=None):
    """Apply the policy to every slide in the tiling index. Returns the process exit code."""
    args = parse_args(argv)
    args.root = args.root.expanduser().resolve()
    log_path = setup_logging(log, args.log_dir.expanduser().resolve(), f"qc_{args.subset}")
    log.info("Command: %s", " ".join(sys.argv))
    try:
        policy, policy_sha, policy_text = load_policy(args.policy.expanduser().resolve())
        tiling_version = yaml.safe_load(args.tiling_config.expanduser().resolve().read_text())["version"]
        tiled = args.root / "patches" / tiling_version
        index_path = tiled / f"index_{args.subset}.csv"
        with index_path.open(newline="") as handle:
            slides = list(csv.DictReader(handle))
        if not slides:
            raise RuntimeError(f"no tiled slides in {index_path}")
    except (OSError, RuntimeError, KeyError, yaml.YAMLError) as error:
        log.error("%s", error)
        return 2
    log.info("Policy %s (sha256 %s): %s | tiling %s, %d slides | log %s", args.policy,
             policy_sha[:12], json.dumps(policy, sort_keys=True), tiling_version, len(slides), log_path)
    if policy["min_blur"] is None:
        log.warning("min_blur is null: blur is measured but not used for any decision")

    out = tiled / "qc" / policy["version"]
    totals, summary = Counter(), []
    for slide in slides:
        patches_csv = args.root / slide["csv"]
        if patches_csv.stat().st_size != int(slide["csv_bytes"]):
            log.error("%s changed since tiling; run tile.py --audit", patches_csv)
            return 2
        counts, reasons = Counter(), Counter()
        path = out / slide["source"] / slide["patient_id"] / f"{slide['slide_id']}_decisions.csv"
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
                        "accept": counts["accept"],
                        "review": counts["review"], "reject": counts["reject"],
                        "reasons": json.dumps(dict(sorted(reasons.items())))})

    with (out / f"slides_{args.subset}.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    write_json(out / f"policy_{args.subset}.json", {
        "policy": policy, "policy_sha256": policy_sha, "policy_yaml": policy_text,
        "tiling_version": tiling_version, "index_sha256": hashlib.sha256(index_path.read_bytes()).hexdigest(),
        "counts": dict(totals), "applied_at_utc": datetime.now(timezone.utc).isoformat()})
    log.info("%d slides: %d accept, %d review, %d reject; written to %s", len(summary),
             totals["accept"], totals["review"], totals["reject"], out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
