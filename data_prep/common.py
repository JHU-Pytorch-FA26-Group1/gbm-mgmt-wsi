"""Shared helpers for the tiling pipeline: config checks, file writes and hashes, IDs, paths, fold map.

Kept free of TIAToolbox and HDF5 imports so qc.py, review.py and export.py stay light.
"""
import csv
import hashlib
import json
import math
import os
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_TILING_CONFIG = REPO_ROOT / "configs" / "tiling.yaml"
DEFAULT_POLICY = REPO_ROOT / "configs" / "qc_policy.yaml"
DEFAULT_FOLDS = REPO_ROOT / "manifests" / "patient_folds.csv"

# Bump whenever a change to tile.py or slides.py alters the tiling outputs. A config version folder
# holding outputs from another CODE_VERSION is refused, so bump the config `version` too.
CODE_VERSION = 7
TILING_KEYS = {"version", "tile_px", "target_mpp", "max_upsample_pct", "mask_mpp", "mask_method",
               "mask_median_px", "mask_close_px", "mask_min_saturation", "min_tissue_fraction",
               "max_tiles_per_slide", "seed", "glass_max_saturation", "glass_min_value",
               "near_white_value", "near_black_value", "dark_value", "green_hue_min", "green_hue_max",
               "green_min_saturation", "png_compress_level"}
MASK_METHODS = ("saturation_multiotsu", "saturation_otsu", "otsu_gray")


def is_int(v):
    return isinstance(v, int) and not isinstance(v, bool)


def is_num(v):
    return is_int(v) or (isinstance(v, float) and math.isfinite(v))


def is_level(v):
    """An 8-bit image value, 0-255."""
    return is_int(v) and 0 <= v <= 255


def is_name(v):
    """A non-empty name that is safe as a single folder name."""
    return isinstance(v, str) and v.strip() != "" and Path(v).name == v and v not in (".", "..")


def load_settings(path, keys, checks, what):
    """Read a YAML mapping, require exactly `keys`, apply `checks`; return (dict, SHA-256, text)."""
    text = Path(path).read_text()
    settings = yaml.safe_load(text)
    if not isinstance(settings, dict):
        raise RuntimeError(f"{what} is not a YAML mapping: {path}")
    if set(settings) != keys:
        raise RuntimeError(f"{what} keys {sorted(set(settings) ^ keys)} missing or unexpected")
    bad = [key for key, check in checks.items() if not check(settings[key], settings)]
    if bad:
        raise RuntimeError(f"bad {what} values: " + ", ".join(f"{k}={settings[k]!r}" for k in bad))
    return settings, hashlib.sha256(text.encode()).hexdigest(), text


def load_tiling_config(path=DEFAULT_TILING_CONFIG):
    """Read and validate the tiling config; return (settings, file SHA-256, file text)."""
    checks = {
        "version": lambda v, s: is_name(v),
        "tile_px": lambda v, s: is_int(v) and v > 0,
        "target_mpp": lambda v, s: is_num(v) and v > 0,
        "max_upsample_pct": lambda v, s: is_num(v) and v >= 0,
        "mask_mpp": lambda v, s: is_num(v) and v > 0,
        "mask_method": lambda v, s: v in MASK_METHODS,
        "mask_median_px": lambda v, s: is_int(v) and v >= 3 and v % 2 == 1,
        "mask_close_px": lambda v, s: is_int(v) and v >= 0,
        "mask_min_saturation": lambda v, s: is_level(v),
        "min_tissue_fraction": lambda v, s: is_num(v) and 0 <= v <= 1,
        "max_tiles_per_slide": lambda v, s: v is None or (is_int(v) and v >= 1),
        "seed": lambda v, s: is_int(v) and v >= 0,
        "glass_max_saturation": lambda v, s: is_level(v),
        "glass_min_value": lambda v, s: is_level(v),
        "near_white_value": lambda v, s: is_level(v),
        "near_black_value": lambda v, s: is_level(v) and v < s["near_white_value"],
        "dark_value": lambda v, s: is_level(v),
        "green_hue_min": lambda v, s: is_int(v) and 0 <= v <= 179,
        "green_hue_max": lambda v, s: is_int(v) and s["green_hue_min"] <= v <= 179,
        "green_min_saturation": lambda v, s: is_level(v),
        "png_compress_level": lambda v, s: is_int(v) and 0 <= v <= 9,
    }
    return load_settings(path, TILING_KEYS, checks, "config")


def tiling_version(path=DEFAULT_TILING_CONFIG):
    """The `version` of a tiling config (its output folder name), without the full validation."""
    return yaml.safe_load(Path(path).read_text())["version"]


def write_rows(path, rows, fields=None):
    """Write dict rows to a CSV atomically (temporary file, then rename). Returns the row count."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    fields = fields or (list(rows[0]) if rows else [])
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)
    return len(rows)


def read_rows(path):
    """All rows of a CSV as dicts."""
    with Path(path).open(newline="") as handle:
        return list(csv.DictReader(handle))


def sha256_file(path):
    """SHA-256 of a file, streamed in chunks rather than read into memory at once."""
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def slide_key(row):
    """Slide part of the patch ID, the same as gbm_patching.py: source + 16 hex of SHA-256(slide_id)."""
    return row["source"] + "_" + hashlib.sha256(row["slide_id"].encode()).hexdigest()[:16]


def output_paths(root, row, version):
    """Return (tiles .h5, QC .png, mask .png, patches .csv, receipt .json) for a slide."""
    folder = Path(root) / "patches" / version / row["source"] / row["patient_id"]
    stem = row["slide_id"]
    return (folder / f"{stem}.h5", folder / f"{stem}_qc.png", folder / f"{stem}_mask.png",
            folder / f"{stem}_patches.csv",
            Path(root) / "state" / "tiling" / version / row["source"] / f"{stem}.json")


def tiled(root, row, version, config_sha):
    """True if the slide was tiled with this config and CODE_VERSION and its outputs are intact.

    Intact means the .h5 and .csv files have the sizes recorded in the receipt and the QC image
    exists. The SHA-256 hashes in the receipt are only rechecked by tile.py --audit.
    """
    h5_path, qc_path, _, csv_path, receipt = output_paths(root, row, version)
    try:
        record = json.loads(receipt.read_text())
        sizes = (h5_path.stat().st_size, csv_path.stat().st_size)
    except (OSError, ValueError):
        return False
    return (record.get("config_sha256") == config_sha
            and record.get("code_version") == CODE_VERSION
            and (record.get("h5_bytes"), record.get("csv_bytes")) == sizes and qc_path.exists())


def check_output_folder(root, version, config_sha):
    """Refuse a config version folder that already holds slides from another config or code."""
    for path in sorted((Path(root) / "state" / "tiling" / version).glob("*/*.json")):
        record = json.loads(path.read_text())
        if record.get("config_sha256") != config_sha or record.get("code_version") != CODE_VERSION:
            raise RuntimeError(f"{path} is from another config or CODE_VERSION; "
                               f"bump `version` in the config")


def read_folds(path=DEFAULT_FOLDS):
    """The shared patient fold map: {(source, patient_id): "0".."4" for TCGA, "external" for CPTAC}."""
    rows = read_rows(path)
    folds = {(r["source"], r["patient_id"]): r["fold"] for r in rows}
    if len(folds) != len(rows) or any(
            (s == "TCGA" and f not in {"0", "1", "2", "3", "4"}) or (s == "CPTAC" and f != "external")
            or s not in ("TCGA", "CPTAC") for (s, _), f in folds.items()):
        raise RuntimeError(f"invalid or duplicate fold assignments: {path}")
    return folds
