"""Extract every tissue tile from downloaded slides with TIAToolbox (224 px at 0.5 um/px by default).

Examples:
    python data_prep/tile.py --root ~/Bioinformatics/WSI/data --subset B --dry-run
    python data_prep/tile.py --root ~/Bioinformatics/WSI/data --subset B --slides C3N-02786-23
    python data_prep/tile.py --root ~/Bioinformatics/WSI/data --subset B --masks-only --slides C3N-02786-23
    python data_prep/tile.py --root ~/Bioinformatics/WSI/data --subset B --workers 8
    python data_prep/tile.py --root ~/Bioinformatics/WSI/data --subset B --audit

The tile grid and tile reads are TIAToolbox's SlidingWindowPatchExtractor; this script adds the
tissue mask, per-tile scores, lossless storage, receipts, an integrity audit and parallelism.

Per slide, writes under <root>/patches/<config version>/{source}/{patient_id}/: {slide_id}.h5 (PNG
tiles plus coordinates and pixel hashes), {slide_id}_patches.csv (one row per grid position that
touches the mask, with its scores), a QC image and the mask. With `max_tiles_per_slide` set, only a
seeded sample of each slide's tissue tiles is stored; every tissue tile is still scored and recorded. A receipt goes under
<root>/state/tiling/<config version>/. On rerun, a slide is skipped only if its receipt matches the
current config and CODE_VERSION and its outputs have the recorded sizes. Tiles are not filtered for
quality here: configs/qc_policy.yaml and data_prep/qc.py decide which tiles are used.

Exit codes: 0 = every selected slide done (or audited clean), 1 = some slides failed, are not
downloaded yet, or failed the audit, 2 = setup or config problem, 130 = interrupted with Ctrl-C.
"""
import argparse
import csv
import hashlib
import io
import json
import logging
import math
import multiprocessing
import os
import platform
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version as package_version
from pathlib import Path

import cv2
import h5py
import numpy as np
import openslide
import tiatoolbox
import yaml
from PIL import Image
from skimage.filters import threshold_multiotsu, threshold_otsu
from tiatoolbox.tools.patchextraction import SlidingWindowPatchExtractor
from tiatoolbox.tools.tissuemask import OtsuTissueMasker
from tiatoolbox.wsicore.wsireader import WSIReader
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

import split
from download import (DEFAULT_LOG_DIR, DEFAULT_MANIFEST, REPO_ROOT, SOURCES, load_manifest,
                      paths_for, setup_logging, slide_progress, write_json)

DEFAULT_CONFIG = REPO_ROOT / "configs" / "tiling.yaml"
# Bump whenever a change to this file alters the outputs. A config folder holding outputs from
# another CODE_VERSION is refused, so bump the config `version` too.
CODE_VERSION = 7
CONFIG_KEYS = {"version", "tile_px", "target_mpp", "max_upsample_pct", "mask_mpp", "mask_method",
               "mask_median_px", "mask_close_px", "mask_min_saturation", "min_tissue_fraction",
               "max_tiles_per_slide", "seed",
               "glass_max_saturation", "glass_min_value", "near_white_value", "near_black_value",
               "png_compress_level"}
MASK_METHODS = ("saturation_multiotsu", "saturation_otsu", "otsu_gray")
# --masks-only compares these: (label, method, floor). A floor of None means the config's floor.
MASK_VARIANTS = (("two-class", "saturation_otsu", 0),
                 ("three-class", "saturation_multiotsu", 0),
                 ("three-class + floor", "saturation_multiotsu", None))
# Columns shared with the team's gbm_patching.py keep its names and meanings.
PATCH_FIELDS = ["source", "patient_id", "slide_id", "slide_relpath", "grid_index", "candidate_index",
                "patch_id", "role", "stored", "h5_index", "x_target", "y_target", "patch_px", "target_mpp",
                "x0", "y0", "x1", "y1", "native_mpp_x", "native_mpp_y", "mask_fraction",
                "mean_r", "mean_g", "mean_b", "gray_median", "laplacian_variance",
                "near_white_fraction", "near_black_fraction", "glass_fraction", "pixel_sha256"]
ROLE_CODES = {"background": 0, "edge": 1, "tissue": 2}
QC_MAX_BOXES = 300   # tiles outlined on the QC image; drawing every tile would hide the slide
THREAD_VARIABLES = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                    "NUMEXPR_NUM_THREADS")

log = logging.getLogger("tile")

# TIAToolbox warns on every tile it enlarges, which is expected for slides coarser than
# target_mpp (e.g. 0.5015 um/px). Drop only that message; other TIAToolbox warnings still show.
for _handler in logging.getLogger().handlers:
    _handler.addFilter(lambda record: "Scale > 1" not in record.getMessage())


def allocated_cpus():
    """CPUs this process may use: the container's CPU quota if one is set, else its usable CPUs.

    os.cpu_count() reports the host's CPUs, which on Run:ai can be far more than the allocation.
    """
    usable = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count() or 1
    quota = None
    try:   # cgroup v2: "<quota> <period>" or "max <period>"
        limit, period = Path("/sys/fs/cgroup/cpu.max").read_text().split()[:2]
        if limit != "max":
            quota = int(limit) / int(period)
    except (OSError, ValueError):
        try:   # cgroup v1
            limit = int(Path("/sys/fs/cgroup/cpu/cpu.cfs_quota_us").read_text())
            period = int(Path("/sys/fs/cgroup/cpu/cpu.cfs_period_us").read_text())
            if limit > 0:
                quota = limit / period
        except (OSError, ValueError):
            pass
    return max(1, min(usable, math.floor(quota)) if quota else usable)


def parse_args(argv=None):
    """Parse command-line options; pass `argv` as a list to test without a shell."""
    parser = argparse.ArgumentParser(description="Extract tissue tiles from downloaded slides.")
    parser.add_argument("--root", type=Path, required=True,
                        help="data folder used by download.py")
    parser.add_argument("--subset", choices=["full", "A", "B"], required=True,
                        help="full cohort, or team subset A or B")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG,
                        help="tiling settings (default: %(default)s)")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST,
                        help="frozen slide manifest (default: %(default)s)")
    parser.add_argument("--sources", nargs="+", choices=SOURCES, default=list(SOURCES),
                        help="cohorts to tile (default: TCGA CPTAC)")
    parser.add_argument("--slides", nargs="+", default=None,
                        help="only these slides: slide IDs or file-name prefixes (for a pilot)")
    parser.add_argument("--workers", type=int, default=max(1, allocated_cpus() - 1),
                        help="slides processed in parallel (default: allocated CPUs - 1 = %(default)s)")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true",
                      help="report done and pending slides, then exit")
    mode.add_argument("--masks-only", action="store_true",
                      help="save two-class / three-class / floor mask comparisons; tile nothing")
    mode.add_argument("--audit", action="store_true",
                      help="recheck tiled slides: file hashes and every tile's pixel hash")
    parser.add_argument("--log-dir", type=Path, default=DEFAULT_LOG_DIR,
                        help="folder for run .log files (default: %(default)s)")
    return parser.parse_args(argv)


def load_config(path):
    """Read and validate the tiling config; return (settings dict, file SHA-256, file text)."""
    text = path.read_text()
    cfg = yaml.safe_load(text)
    if not isinstance(cfg, dict):
        raise RuntimeError(f"config is not a YAML mapping: {path}")
    if set(cfg) != CONFIG_KEYS:
        raise RuntimeError(f"config keys {sorted(set(cfg) ^ CONFIG_KEYS)} missing or unexpected")

    def is_int(v):
        return isinstance(v, int) and not isinstance(v, bool)

    def is_num(v):
        return is_int(v) or (isinstance(v, float) and math.isfinite(v))

    def is_level(v):
        return is_int(v) and 0 <= v <= 255

    v = cfg
    checks = {
        "version": isinstance(v["version"], str) and v["version"].strip() != ""
                   and Path(v["version"]).name == v["version"] and v["version"] not in (".", ".."),
        "tile_px": is_int(v["tile_px"]) and v["tile_px"] > 0,
        "target_mpp": is_num(v["target_mpp"]) and v["target_mpp"] > 0,
        "max_upsample_pct": is_num(v["max_upsample_pct"]) and v["max_upsample_pct"] >= 0,
        "mask_mpp": is_num(v["mask_mpp"]) and v["mask_mpp"] > 0,
        "mask_method": v["mask_method"] in MASK_METHODS,
        "mask_median_px": is_int(v["mask_median_px"]) and v["mask_median_px"] >= 3
                          and v["mask_median_px"] % 2 == 1,
        "mask_close_px": is_int(v["mask_close_px"]) and v["mask_close_px"] >= 0,
        "mask_min_saturation": is_level(v["mask_min_saturation"]),
        "min_tissue_fraction": is_num(v["min_tissue_fraction"]) and 0 <= v["min_tissue_fraction"] <= 1,
        "max_tiles_per_slide": v["max_tiles_per_slide"] is None
                               or (is_int(v["max_tiles_per_slide"]) and v["max_tiles_per_slide"] >= 1),
        "seed": is_int(v["seed"]) and v["seed"] >= 0,
        "glass_max_saturation": is_level(v["glass_max_saturation"]),
        "glass_min_value": is_level(v["glass_min_value"]),
        "near_white_value": is_level(v["near_white_value"]),
        "near_black_value": is_level(v["near_black_value"]),
        "png_compress_level": is_int(v["png_compress_level"]) and 0 <= v["png_compress_level"] <= 9,
    }
    bad = [key for key, ok in checks.items() if not ok]
    if bad:
        raise RuntimeError("bad config values: " + ", ".join(f"{k}={cfg[k]!r}" for k in bad))
    return cfg, hashlib.sha256(text.encode()).hexdigest(), text


def library_versions():
    """Versions of the libraries that shape the outputs, including the native OpenSlide library."""
    versions = {"python": platform.python_version(), "tiatoolbox": tiatoolbox.__version__,
                "openslide": openslide.__library_version__, "openslide_python": openslide.__version__,
                "numpy": np.__version__, "opencv": cv2.__version__, "h5py": h5py.__version__}
    for name in ("pillow", "scikit-image"):
        try:
            versions[name] = package_version(name)
        except PackageNotFoundError:
            versions[name] = "unknown"
    return versions


def output_paths(root, row, version):
    """Return (tiles .h5, QC .png, mask .png, patches .csv, receipt .json) for a slide."""
    folder = root / "patches" / version / row["source"] / row["patient_id"]
    stem = row["slide_id"]
    return (folder / f"{stem}.h5", folder / f"{stem}_qc.png", folder / f"{stem}_mask.png",
            folder / f"{stem}_patches.csv",
            root / "state" / "tiling" / version / row["source"] / f"{stem}.json")


def tiled(root, row, version, config_sha):
    """True if the slide was tiled with this config and CODE_VERSION and its outputs are intact.

    Intact means the .h5 and .csv files have the sizes recorded in the receipt and the QC image
    exists. The SHA-256 hashes in the receipt are only rechecked by --audit.
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
    for path in sorted((root / "state" / "tiling" / version).glob("*/*.json")):
        record = json.loads(path.read_text())
        if record.get("config_sha256") != config_sha or record.get("code_version") != CODE_VERSION:
            raise RuntimeError(f"{path} is from another config or CODE_VERSION; "
                               f"bump `version` in the config")


def open_slide(path, cfg):
    """Open a slide and check its native resolution; return (reader, mpp array, upsampling %).

    Slides coarser than target_mpp are enlarged when read. More than `max_upsample_pct` fails the
    slide instead of silently inventing detail.
    """
    wsi = WSIReader.open(path)
    try:
        mpp = wsi.info.mpp
        if mpp is None or len(mpp) != 2 or not all(np.isfinite(mpp)) or min(mpp) <= 0:
            raise ValueError(f"missing or invalid mpp: {mpp}")
        mpp = np.asarray(mpp, dtype=float)
        upsample = np.maximum(0.0, 100 * (mpp / cfg["target_mpp"] - 1))
        if upsample.max() > cfg["max_upsample_pct"] + 1e-9:
            raise ValueError(f"needs {upsample.max():.2f}% upsampling (mpp {mpp.tolist()})")
    except Exception:
        close_slide(wsi)
        raise
    return wsi, mpp, upsample


def close_slide(wsi):
    """Close the file handles TIAToolbox keeps open (OpenSlide or tifffile)."""
    for name in ("openslide_wsi", "tiff"):
        handle = getattr(wsi, name, None)
        if handle is not None:
            handle.close()


def tissue_mask(thumb, cfg, method=None, floor=None):
    """Return (0/1 uint8 tissue mask, threshold used, Otsu threshold) for an RGB thumbnail.

    The saturation methods threshold the median-blurred HSV saturation channel: tissue is colored,
    background is white or gray. A pixel is tissue if its saturation >= the threshold used, which
    is the Otsu cut raised to at least `floor` (default: `mask_min_saturation`). Small gaps are then
    closed (`mask_close_px`, 0 = off).
    "saturation_multiotsu": the lower cut of a three-class Otsu (background / pale / strong), so pale
    tissue stays tissue on heavily stained slides. On very faint slides that cut can fall inside the
    background noise, which is what the floor is for.
    "saturation_otsu": a two-class Otsu, for comparison. skimage counts pixels ABOVE its threshold as
    foreground, so the cut recorded here is that threshold rounded down, plus 1.
    A thumbnail with too few saturation levels for the method uses the floor alone; its Otsu
    threshold is None.
    "otsu_gray": TIAToolbox's OtsuTissueMasker (grayscale Otsu, no cleanup); both thresholds are None.
    """
    method = method or cfg["mask_method"]
    floor = cfg["mask_min_saturation"] if floor is None else floor
    if method == "otsu_gray":
        return OtsuTissueMasker().fit_transform([thumb])[0].astype(np.uint8), None, None
    saturation = cv2.cvtColor(np.ascontiguousarray(thumb[..., :3]), cv2.COLOR_RGB2HSV)[..., 1]
    saturation = cv2.medianBlur(saturation, cfg["mask_median_px"])
    levels = len(np.unique(saturation))
    otsu = None
    if method == "saturation_multiotsu" and levels >= 3:
        otsu = float(threshold_multiotsu(saturation, classes=3)[0])
    elif method == "saturation_otsu" and levels >= 2:
        otsu = float(math.floor(threshold_otsu(saturation)) + 1)
    threshold = float(max(otsu, floor) if otsu is not None else floor)
    mask = (saturation >= threshold).astype(np.uint8)
    if cfg["mask_close_px"]:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (cfg["mask_close_px"],) * 2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask, threshold, otsu


def tile_grid(wsi, cfg):
    """TIAToolbox's full tile grid: a SlidingWindowPatchExtractor with no mask.

    `extractor.coordinate_list` holds every complete `tile_px` window at `target_mpp` as
    [x0, y0, x1, y1] (stride = tile_px, within_bound: no padded tiles at the right/bottom
    margins); its row number is the tile's grid index, and `extractor[i]` reads tile i.
    """
    return SlidingWindowPatchExtractor(
        wsi, cfg["tile_px"], resolution=cfg["target_mpp"], units="mpp", stride=cfg["tile_px"],
        within_bound=True)


def mask_roles(mask, coords, wsi_shape, min_fraction):
    """Classify grid tiles against the thumbnail mask; return (roles, masked fraction).

    Uses the same coordinate scaling and rule as TIAToolbox's PatchExtractor.filter_coordinates,
    so "tissue" here is exactly what an extractor given this mask and min_mask_ratio would keep:
    more than `min_fraction` of the tile is masked, or all of it. "edge" tiles touch the mask
    without being tissue; "background" tiles do not touch it.
    """
    scale = np.array(mask.shape[1::-1]) / np.array(wsi_shape)
    box = coords.copy().astype(np.float32)
    box[:, [0, 2]] *= scale[0]
    box[:, [0, 2]] = np.clip(box[:, [0, 2]], 0, mask.shape[1])
    box[:, [1, 3]] *= scale[1]
    box[:, [1, 3]] = np.clip(box[:, [1, 3]], 0, mask.shape[0])
    x0, y0, x1, y1 = box.astype(np.int32).T
    # Masked pixels per box from a summed-area table: one lookup per tile instead of a slice.
    summed = np.zeros((mask.shape[0] + 1, mask.shape[1] + 1), dtype=np.int64)
    summed[1:, 1:] = (mask > 0).cumsum(axis=0).cumsum(axis=1)
    area = np.maximum(x1 - x0, 0).astype(np.int64) * np.maximum(y1 - y0, 0)
    positive = np.where(area > 0, summed[y1, x1] - summed[y0, x1] - summed[y1, x0] + summed[y0, x0], 0)
    tissue = ((positive == area) | (positive > area * min_fraction)) & (positive > 0) & (area > 0)
    roles = np.where(tissue, ROLE_CODES["tissue"],
                     np.where(positive > 0, ROLE_CODES["edge"], ROLE_CODES["background"]))
    fraction = np.divide(positive, area, out=np.zeros(len(area)), where=area > 0)
    return roles.astype(np.int8), fraction


def stored_tiles(tissue, cfg, slide_id):
    """Which tissue tiles get their pixels stored: all of them, or a seeded random sample.

    Above `max_tiles_per_slide`, a sample of that size is drawn without replacement with a per-slide
    seed (hash of `seed` and the slide ID), so reruns and other machines pick the same tiles. Every
    tissue tile is still read, scored and recorded; tiles not stored can be reread from the slide.
    Returns grid indices in grid order.
    """
    cap = cfg["max_tiles_per_slide"]
    if cap is None or len(tissue) <= cap:
        return tissue
    seed = int.from_bytes(hashlib.sha256(f"{cfg['seed']}|{slide_id}".encode()).digest()[:8], "big")
    return np.sort(tissue[np.random.default_rng(seed).choice(len(tissue), size=cap, replace=False)])


def tile_scores(img, cfg):
    """Quality scores for an RGB tile. Names and definitions match gbm_patching.py, plus glass.

    laplacian_variance: variance of the Laplacian of the grayscale tile; low for blurred, empty or
    featureless tiles. glass_fraction: share of pixels with HSV saturation below
    `glass_max_saturation` and brightness above `glass_min_value` (bright, uncolored background).
    near_white / near_black: share of pixels with every channel >= `near_white_value` / <=
    `near_black_value`.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    mean = img.mean(axis=(0, 1))
    return {"mean_r": float(mean[0]), "mean_g": float(mean[1]), "mean_b": float(mean[2]),
            "gray_median": float(np.median(gray)),
            "laplacian_variance": float(cv2.Laplacian(gray, cv2.CV_64F).var()),
            "near_white_fraction": float(np.all(img >= cfg["near_white_value"], axis=2).mean()),
            "near_black_fraction": float(np.all(img <= cfg["near_black_value"], axis=2).mean()),
            "glass_fraction": float(((hsv[..., 1] < cfg["glass_max_saturation"])
                                     & (hsv[..., 2] > cfg["glass_min_value"])).mean())}


def slide_key(row):
    """Slide part of the patch ID, the same as gbm_patching.py: source + 16 hex of SHA-256(slide_id)."""
    return row["source"] + "_" + hashlib.sha256(row["slide_id"].encode()).hexdigest()[:16]


def write_qc(path, thumb, mask, coords, tile_px, scale_x, scale_y):
    """Save the thumbnail with the tissue-mask outline (green) and up to QC_MAX_BOXES tiles (blue).

    `coords` and `tile_px` are at target_mpp; `scale_x`, `scale_y` map them to thumbnail pixels.
    The outlined tiles are spread evenly over the stored tiles in grid order.
    """
    image = np.ascontiguousarray(thumb[..., :3].copy())
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(image, contours, -1, (0, 200, 0), 1)
    shown = np.unique(np.linspace(0, len(coords) - 1, min(len(coords), QC_MAX_BOXES)).astype(int))
    for x, y in coords[shown]:
        cv2.rectangle(image, (int(x * scale_x), int(y * scale_y)),
                      (int((x + tile_px) * scale_x), int((y + tile_px) * scale_y)), (0, 0, 255), 1)
    save_png(path, image)


def save_png(path, image):
    """Write an image atomically (temporary file, then rename)."""
    temporary = path.with_name(path.name + ".tmp.png")
    Image.fromarray(image).save(temporary)
    os.replace(temporary, path)


def sha256_file(path):
    """SHA-256 of a file, streamed in chunks rather than read into memory at once."""
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def tile_slide(root, row, cfg, config_sha, config_text):
    """Tile one slide (runs in a worker process). Returns a stats dict for its receipt.

    Every grid tile that touches the mask is read and scored. Tissue tiles are stored as PNG in the
    .h5 file (all of them, or a seeded sample of `max_tiles_per_slide`), written one at a time so
    memory does not grow with the slide. Edge tiles and tissue tiles beyond the cap are only scored;
    they can be reread from their coordinates. Nothing is dropped for quality here. A slide with no
    tissue tiles is still written, with zero tiles, so it is recorded in the index and not retried on
    every run.
    """
    started = time.monotonic()
    slide_path = paths_for(root, row)[0]
    h5_path, qc_path, mask_path, csv_path, _ = output_paths(root, row, cfg["version"])
    h5_path.parent.mkdir(parents=True, exist_ok=True)
    tile_px, target_mpp = cfg["tile_px"], cfg["target_mpp"]

    wsi, mpp, upsample = open_slide(slide_path, cfg)
    try:
        thumb = wsi.slide_thumbnail(cfg["mask_mpp"], "mpp")
        mask, threshold, otsu = tissue_mask(thumb, cfg)
        extractor = tile_grid(wsi, cfg)
        grid = extractor.coordinate_list.astype(np.int64)            # [x0, y0, x1, y1] at target_mpp
        width, height = wsi.slide_dimensions(target_mpp, "mpp")
        roles, fractions = mask_roles(mask, grid, (width, height), cfg["min_tissue_fraction"])
        tissue = np.flatnonzero(roles == ROLE_CODES["tissue"])       # grid indices, in grid order
        candidate = np.full(len(grid), -1, dtype=np.int64)
        candidate[tissue] = np.arange(len(tissue))
        stored = stored_tiles(tissue, cfg, row["slide_id"])         # grid indices, in grid order
        h5_index = np.full(len(grid), -1, dtype=np.int64)
        h5_index[stored] = np.arange(len(stored))
        key = slide_key(row)
        stored_coords = grid[stored, :2]
        level0, hashes = [], []

        h5_tmp = h5_path.with_name(h5_path.name + ".tmp")
        csv_tmp = csv_path.with_name(csv_path.name + ".tmp")
        with h5py.File(h5_tmp, "w") as h5, csv_tmp.open("w", newline="") as handle:
            tiles = h5.create_dataset("tiles", shape=(len(stored),), dtype=h5py.vlen_dtype(np.uint8))
            writer = csv.DictWriter(handle, fieldnames=PATCH_FIELDS)
            writer.writeheader()
            for grid_index in np.flatnonzero(roles != ROLE_CODES["background"]):
                grid_index = int(grid_index)
                x, y = (int(v) for v in grid[grid_index, :2])
                img = np.ascontiguousarray(extractor[grid_index][..., :3])
                if img.shape != (tile_px, tile_px, 3) or img.dtype != np.uint8:
                    raise ValueError(f"tile {grid_index}: unexpected image {img.shape} {img.dtype}")
                bounds = [int(v) for v in wsi.bounds_at_resolution_to_baseline(
                    [x, y, x + tile_px, y + tile_px], target_mpp, "mpp")]
                pixel_sha = hashlib.sha256(img.tobytes()).hexdigest()
                index = int(h5_index[grid_index])
                if index >= 0:
                    buffer = io.BytesIO()
                    Image.fromarray(img).save(buffer, format="PNG",
                                              compress_level=cfg["png_compress_level"])
                    tiles[index] = np.frombuffer(buffer.getvalue(), dtype=np.uint8)
                    level0.append(bounds[:2])
                    hashes.append(pixel_sha)
                rank = int(candidate[grid_index])
                writer.writerow({
                    "source": row["source"], "patient_id": row["patient_id"],
                    "slide_id": row["slide_id"], "slide_relpath": str(slide_path.relative_to(root)),
                    "grid_index": grid_index, "candidate_index": rank if rank >= 0 else "",
                    "patch_id": f"{key}_{x}_{y}", "role": "tissue" if rank >= 0 else "edge",
                    "stored": index >= 0, "h5_index": index if index >= 0 else "",
                    "x_target": x, "y_target": y,
                    "patch_px": tile_px, "target_mpp": target_mpp,
                    "x0": bounds[0], "y0": bounds[1], "x1": bounds[2], "y1": bounds[3],
                    "native_mpp_x": float(mpp[0]), "native_mpp_y": float(mpp[1]),
                    "mask_fraction": float(fractions[grid_index]), **tile_scores(img, cfg),
                    "pixel_sha256": pixel_sha})

            h5.create_dataset("coords", data=stored_coords.reshape(-1, 2))
            h5.create_dataset("coords_level0", data=np.array(level0, dtype=np.int64).reshape(-1, 2))
            h5.create_dataset("grid_index", data=stored.astype(np.int64))
            h5.create_dataset("pixel_sha256", data=np.array(hashes, dtype="S64"))
            h5.create_dataset("grid_coords", data=grid[:, :2])
            h5.create_dataset("grid_role", data=roles)
            h5.create_dataset("grid_h5_index", data=h5_index)
            h5.create_dataset("grid_mask_fraction", data=fractions.astype(np.float32))
            attrs = {"slide_id": row["slide_id"], "patient_id": row["patient_id"],
                     "source": row["source"], "file_name": row["file_name"],
                     "source_mpp_x": float(mpp[0]), "source_mpp_y": float(mpp[1]),
                     "upsample_pct": float(upsample.max()),
                     "target_mpp": target_mpp, "tile_px": tile_px, "tile_um": tile_px * target_mpp,
                     "tile_format": f"PNG (lossless), compress_level {cfg['png_compress_level']}",
                     "pixel_sha256_of": "decoded RGB uint8 array bytes (height, width, 3)",
                     "coords_space": "top-left at target_mpp (TIAToolbox coord_space='resolution')",
                     "coords_level0_space": "top-left of the level-0 region read for each tile",
                     "grid": "tiatoolbox SlidingWindowPatchExtractor, stride = tile_px, within_bound",
                     "grid_role_codes": json.dumps(ROLE_CODES),
                     "interpolation": "optimize (TIAToolbox default: area down, cubic up)",
                     "mask_method": cfg["mask_method"],
                     "mask_threshold": -1.0 if threshold is None else threshold,
                     "mask_otsu_threshold": -1.0 if otsu is None else otsu,
                     "mask_floor_applied": threshold is not None and (otsu is None or otsu < threshold),
                     "mask_fraction": float(mask.mean()),
                     "level_dimensions": json.dumps([list(map(int, d)) for d in wsi.info.level_dimensions]),
                     "level_downsamples": json.dumps([float(d) for d in wsi.info.level_downsamples]),
                     "n_grid": len(grid), "n_tissue_tiles": len(tissue), "n_stored_tiles": len(stored),
                     "n_edge_tiles": int((roles == ROLE_CODES["edge"]).sum()),
                     "max_tiles_per_slide": -1 if cfg["max_tiles_per_slide"] is None
                                            else cfg["max_tiles_per_slide"],
                     "tiling_version": cfg["version"], "code_version": CODE_VERSION,
                     "config_sha256": config_sha, "config_yaml": config_text,
                     "library_versions": json.dumps(library_versions())}
            h5.attrs.update(attrs)
        os.replace(h5_tmp, h5_path)
        os.replace(csv_tmp, csv_path)
        save_png(mask_path, mask * 255)
        write_qc(qc_path, thumb, mask, stored_coords, tile_px, mask.shape[1] / width,
                 mask.shape[0] / height)
    finally:
        close_slide(wsi)

    return {"slide_id": row["slide_id"], "source": row["source"], "patient_id": row["patient_id"],
            "file_name": row["file_name"], "h5": str(h5_path.relative_to(root)),
            "h5_sha256": sha256_file(h5_path), "h5_bytes": h5_path.stat().st_size,
            "csv": str(csv_path.relative_to(root)), "csv_sha256": sha256_file(csv_path),
            "csv_bytes": csv_path.stat().st_size,
            "n_grid": len(grid), "n_tissue_tiles": len(tissue), "n_stored_tiles": len(stored),
            "n_edge_tiles": attrs["n_edge_tiles"],
            "source_mpp": float(mpp[0]), "upsample_pct": attrs["upsample_pct"],
            "mask_threshold": threshold, "mask_otsu_threshold": otsu,
            "mask_floor_applied": attrs["mask_floor_applied"], "mask_fraction": attrs["mask_fraction"],
            "config_sha256": config_sha, "code_version": CODE_VERSION, "tiling_version": cfg["version"],
            "seconds": round(time.monotonic() - started, 1),
            "tiled_at_utc": datetime.now(timezone.utc).isoformat()}


def audit_slide(root, row, version):
    """Recheck one tiled slide against its receipt (runs in a worker process).

    Recomputes the .h5 and .csv SHA-256, decodes every stored PNG and compares the hash of its
    pixels with the hash recorded when the tile was read. Returns a result dict; never raises for
    a mismatch, only reports it.
    """
    h5_path, _, _, csv_path, receipt = output_paths(root, row, version)
    record = json.loads(receipt.read_text())
    errors = []
    if sha256_file(h5_path) != record["h5_sha256"]:
        errors.append("h5 sha256 differs from receipt")
    if sha256_file(csv_path) != record["csv_sha256"]:
        errors.append("csv sha256 differs from receipt")
    checked = 0
    with h5py.File(h5_path, "r") as h5:
        tiles, hashes = h5["tiles"], h5["pixel_sha256"]
        tile_px = int(h5.attrs["tile_px"])
        if len(tiles) != record["n_stored_tiles"] or len(hashes) != len(tiles):
            errors.append(f"{len(tiles)} tiles, {len(hashes)} hashes, receipt {record['n_stored_tiles']}")
        for i in range(min(len(tiles), len(hashes))):
            try:
                img = np.asarray(Image.open(io.BytesIO(tiles[i].tobytes())).convert("RGB"))
                if img.shape != (tile_px, tile_px, 3):
                    raise ValueError(f"shape {img.shape}")
                if hashlib.sha256(img.tobytes()).hexdigest() != hashes[i].decode():
                    raise ValueError("pixel hash differs")
            except Exception as error:
                errors.append(f"tile {i}: {error}")
            checked += 1
    return {"source": row["source"], "patient_id": row["patient_id"], "slide_id": row["slide_id"],
            "tiles_checked": checked, "n_errors": len(errors), "errors": errors[:10]}


def review_masks(root, row, cfg):
    """Save a 2 x 2 panel comparing MASK_VARIANTS on one slide's thumbnail (runs in a worker).

    All variants share the thumbnail, median blur and closing; only the Otsu classes and the floor
    differ. Returns one result row per variant (cut, masked fraction, tissue tiles).
    """
    wsi, _, _ = open_slide(paths_for(root, row)[0], cfg)
    try:
        thumb = np.ascontiguousarray(wsi.slide_thumbnail(cfg["mask_mpp"], "mpp")[..., :3])
        grid = tile_grid(wsi, cfg).coordinate_list.astype(np.int64)
        shape = wsi.slide_dimensions(cfg["target_mpp"], "mpp")
    finally:
        close_slide(wsi)
    def labeled(image, title, detail=""):
        """Enlarge a narrow thumbnail so its label fits, and put a two-line label strip above it."""
        scale = max(1, math.ceil(600 / image.shape[1]))
        image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
        strip = np.full((40, image.shape[1], 3), 255, dtype=np.uint8)
        for line, text in enumerate((title, detail)):
            cv2.putText(strip, text, (4, 16 + 18 * line), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0),
                        1, cv2.LINE_AA)
        return np.vstack([strip, image])

    panels, results = [labeled(thumb, row["slide_id"][:40], "thumbnail")], []
    for label, method, floor in MASK_VARIANTS:
        mask, threshold, otsu = tissue_mask(thumb, cfg, method, floor)
        roles, _ = mask_roles(mask, grid, shape, cfg["min_tissue_fraction"])
        n_tissue = int((roles == ROLE_CODES["tissue"]).sum())
        image = thumb.copy()
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(image, contours, -1, (0, 200, 0), 1)
        panels.append(labeled(image, label,
                              f"cut {threshold:g}, {mask.mean():.1%} masked, {n_tissue} tissue tiles"))
        results.append({"source": row["source"], "patient_id": row["patient_id"],
                        "slide_id": row["slide_id"], "variant": label, "method": method,
                        "floor": cfg["mask_min_saturation"] if floor is None else floor,
                        "otsu_threshold": otsu, "threshold_used": threshold,
                        "mask_fraction": float(mask.mean()), "n_tissue_tiles": n_tissue})
    path = root / "patches" / cfg["version"] / "mask_review" / row["source"] / f"{row['slide_id']}.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    save_png(path, np.vstack([np.hstack(panels[:2]), np.hstack(panels[2:])]))
    return results


def _worker_init():
    """Limit each worker to one OpenCV thread so parallel slides do not compete for CPUs."""
    cv2.setNumThreads(1)


def run_jobs(workers, jobs, handle, desc):
    """Run (row, function, args) jobs in spawned worker processes; call handle(row, result, error).

    Each worker opens its own slide and HDF5 files. Returns True if interrupted with Ctrl-C.
    """
    context = multiprocessing.get_context("spawn")
    with logging_redirect_tqdm(loggers=[log]), \
            ProcessPoolExecutor(max_workers=workers, mp_context=context, initializer=_worker_init) as pool:
        futures = {pool.submit(function, *args): row for row, function, args in jobs}
        bar = tqdm(total=len(futures), desc=desc, unit="slide", dynamic_ncols=True,
                   disable=not sys.stderr.isatty())
        try:
            for future in as_completed(futures):
                row = futures[future]
                try:
                    result = future.result()
                except Exception as error:  # A bad slide must not stop the run.
                    handle(row, None, error)
                else:
                    handle(row, result, None)
                bar.update(1)
        except KeyboardInterrupt:
            pool.shutdown(wait=False, cancel_futures=True)
            return True
        finally:
            bar.close()
    return False


def write_rows(path, rows):
    """Write dict rows to a CSV atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def summarize(root, files, version, config_sha, title):
    """Log a per-cohort table of downloaded, tiled and pending slides; return the receipts."""
    receipts = {}
    log.info("%s:", title)
    log.info("  %-6s %6s %10s %6s %7s %9s %8s", "cohort", "slides", "downloaded", "tiled",
             "pending", "stored", "GB")
    for source in SOURCES:
        rows = [r for r in files if r["source"] == source]
        if not rows:
            continue
        done = [r for r in rows if tiled(root, r, version, config_sha)]
        for r in done:
            receipts[r["slide_id"]] = json.loads(output_paths(root, r, version)[4].read_text())
        downloaded = sum(slide_progress(root, r)[0] for r in rows)
        log.info("  %-6s %6d %10d %6d %7d %9d %8.2f", source, len(rows), downloaded, len(done),
                 len(rows) - len(done), sum(receipts[r["slide_id"]]["n_stored_tiles"] for r in done),
                 sum(receipts[r["slide_id"]]["h5_bytes"] for r in done) / 1e9)
    return receipts


def run_audit(args, files, version, config_sha):
    """Audit every selected tiled slide and save the result next to the receipts; return exit code."""
    done = [r for r in files if tiled(args.root, r, version, config_sha)]
    log.info("Auditing %d tiled slides (%d selected slides are not tiled with this config)",
             len(done), len(files) - len(done))
    results = []

    def handle(row, result, error):
        if error is not None:
            result = {"source": row["source"], "patient_id": row["patient_id"],
                      "slide_id": row["slide_id"], "tiles_checked": 0, "n_errors": 1,
                      "errors": [f"audit failed: {error}"]}
        results.append(result)
        if result["n_errors"]:
            log.error("%s %s: %s", row["patient_id"], row["slide_id"], "; ".join(result["errors"]))

    interrupted = run_jobs(args.workers, [(r, audit_slide, (args.root, r, version)) for r in done],
                           handle, f"audit {args.subset}")
    stamp = datetime.now(timezone.utc)
    report = {"tiling_version": version, "config_sha256": config_sha, "code_version": CODE_VERSION,
              "subset": args.subset, "audited_at_utc": stamp.isoformat(), "complete": not interrupted,
              "slides_selected": len(files), "slides_tiled": len(done), "slides_audited": len(results),
              "tiles_checked": sum(r["tiles_checked"] for r in results),
              "slides_with_errors": sum(bool(r["n_errors"]) for r in results),
              "slides": sorted(results, key=lambda r: (r["source"], r["patient_id"], r["slide_id"]))}
    path = args.root / "state" / "tiling" / version / f"audit_{args.subset}_{stamp:%Y%m%d_%H%M%S}.json"
    write_json(path, report)
    log.info("Audit: %d slides, %d tiles checked, %d slides with errors; saved %s",
             len(results), report["tiles_checked"], report["slides_with_errors"], path)
    if interrupted:
        return 130
    return 1 if report["slides_with_errors"] or len(done) < len(files) else 0


def run_masks(args, files, cfg):
    """Save mask comparison panels and a CSV for the selected downloaded slides; return exit code."""
    ready = [r for r in files if slide_progress(args.root, r)[0]]
    log.info("Mask comparison for %d downloaded slides: %s", len(ready),
             ", ".join(label for label, _, _ in MASK_VARIANTS))
    rows, failed = [], []

    def handle(row, result, error):
        if error is not None:
            failed.append(row)
            log.error("%s %s failed: %s", row["patient_id"], row["file_name"], error)
        else:
            rows.extend(result)

    interrupted = run_jobs(args.workers, [(r, review_masks, (args.root, r, cfg)) for r in ready],
                           handle, f"masks {args.subset}")
    if rows:
        path = args.root / "patches" / cfg["version"] / "mask_review" / f"mask_review_{args.subset}.csv"
        write_rows(path, sorted(rows, key=lambda r: (r["source"], r["patient_id"], r["slide_id"])))
        log.info("Mask panels and %s saved", path)
    if interrupted:
        return 130
    return 1 if failed or len(ready) < len(files) else 0


def main(argv=None):
    """Select slides, then tile the pending ones in parallel (or audit, or compare masks).

    Returns the process exit code (see the module docstring).
    """
    args = parse_args(argv)
    args.root = args.root.expanduser().resolve()
    args.log_dir = args.log_dir.expanduser().resolve()
    mode = "audit" if args.audit else "masks" if args.masks_only else "tile"
    log_path = setup_logging(log, args.log_dir, f"{mode}_{args.subset}")
    log.info("Command: %s", " ".join(sys.argv))
    if args.workers < 1:
        log.error("--workers must be >= 1: %d", args.workers)
        return 2
    # Spawned workers inherit these: one numerical thread each, so N workers use about N CPUs.
    for name in THREAD_VARIABLES:
        os.environ.setdefault(name, "1")

    try:
        cfg, config_sha, config_text = load_config(args.config.expanduser().resolve())
        version = cfg["version"]
        log.info("Config %s (sha256 %s): %s", args.config, config_sha[:12],
                 json.dumps(cfg, sort_keys=True))
        log.info("%s | tile.py code version %d | workers %d (allocated CPUs %d) | log %s",
                 json.dumps(library_versions()), CODE_VERSION, args.workers, allocated_cpus(), log_path)
        _, manifest = load_manifest(args.manifest.expanduser().resolve())
        all_files = manifest["files"]
        files = split.select_files(all_files, args.subset, split.build_patient_split(all_files))
        subset_files = files
        files = [r for r in files if r["source"] in args.sources]
        if args.slides:
            files = [r for r in files if any(r["slide_id"] == s or r["file_name"].startswith(s)
                                             for s in args.slides)]
            if not files:
                raise RuntimeError(f"no selected slides match {args.slides}")
        if mode == "masks":
            return run_masks(args, files, cfg)
        check_output_folder(args.root, version, config_sha)
    except (OSError, RuntimeError, ValueError, yaml.YAMLError) as error:
        log.error("%s", error)
        return 2
    if mode == "audit":
        return run_audit(args, files, version, config_sha)

    summarize(args.root, files, version, config_sha, f"Start of run (subset {args.subset})")
    not_downloaded = [r for r in files if not slide_progress(args.root, r)[0]]
    pending = [r for r in files
               if r not in not_downloaded and not tiled(args.root, r, version, config_sha)]
    if not_downloaded:
        log.warning("%d selected slides are not downloaded and verified; skipping them",
                    len(not_downloaded))
    log.info("Tiling %d pending slides into %s", len(pending), args.root / "patches" / version)
    if args.dry_run:
        log.info("Dry run: nothing tiled.")
        return 0

    failed = []

    def handle(row, stats, error):
        if error is not None:
            failed.append(row)
            log.error("%s %s failed: %s", row["patient_id"], row["file_name"], error)
            return
        write_json(output_paths(args.root, row, version)[4], stats)
        log.log(logging.INFO if stats["n_tissue_tiles"] else logging.WARNING,
                "%s %s: %d of %d tissue tiles stored, %d grid positions (%d edge), %.0f MB, %.0f s%s",
                row["patient_id"], row["file_name"], stats["n_stored_tiles"], stats["n_tissue_tiles"],
                stats["n_grid"],
                stats["n_edge_tiles"], stats["h5_bytes"] / 1e6, stats["seconds"],
                ", mask floor applied" if stats["mask_floor_applied"] else "")

    started = time.monotonic()
    interrupted = run_jobs(
        args.workers, [(r, tile_slide, (args.root, r, cfg, config_sha, config_text)) for r in pending],
        handle, f"subset {args.subset}")

    log.info("Active time %.1f min", (time.monotonic() - started) / 60)
    receipts = summarize(args.root, files, version, config_sha, f"End of run (subset {args.subset})")
    # The index lists every tiled slide in the subset, not only this run's --sources/--slides.
    indexed = [json.loads(output_paths(args.root, r, version)[4].read_text())
               for r in subset_files if tiled(args.root, r, version, config_sha)]
    index = args.root / "patches" / version / f"index_{args.subset}.csv"
    if indexed:
        write_rows(index, sorted(indexed, key=lambda r: (r["source"], r["patient_id"], r["slide_id"])))
        log.info("Index of %d tiled slides: %s", len(indexed), index)
    for row in failed:
        log.error("Failed: %s %s %s", row["source"], row["patient_id"], row["file_name"])

    if interrupted:
        log.info("Stopped: interrupted. Finished slides are kept; rerun to continue.")
        return 130
    if failed or not_downloaded or len(receipts) < len(files):
        log.warning("%d selected slides are not tiled; rerun after fixing them.",
                    len(files) - len(receipts))
        return 1
    log.info("All selected slides are tiled.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
