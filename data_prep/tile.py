"""Extract tissue tiles from downloaded slides with TIAToolbox (256 px at 0.5 um/px by default).

Examples:
    python data_prep/tile.py --root ~/Bioinformatics/WSI/data --subset B --dry-run
    python data_prep/tile.py --root ~/Bioinformatics/WSI/data --subset B --slides C3N-02786-23
    python data_prep/tile.py --root ~/Bioinformatics/WSI/data --subset B --workers 8

The tile grid, tissue filtering and tile reads are TIAToolbox's SlidingWindowPatchExtractor; this
script adds the tissue mask, the capped random sample, storage, receipts and parallelism.

Per slide, writes <root>/patches/{source}/{patient_id}/{slide_id}.h5 (JPEG tiles plus their
coordinates), a QC image next to it, and a receipt under <root>/state/tiling/. On rerun, a slide is
skipped only if its receipt matches the current config and CODE_VERSION and its outputs are intact.

Exit codes: 0 = every selected slide tiled, 1 = some slides failed or are not downloaded yet,
2 = setup or config problem, 130 = interrupted with Ctrl-C.
"""
import argparse
import csv
import hashlib
import io
import json
import logging
import math
import os
import platform
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import cv2
import h5py
import numpy as np
import openslide
import tiatoolbox
import yaml
from PIL import Image
from skimage.filters import threshold_multiotsu
from tiatoolbox.tools.patchextraction import SlidingWindowPatchExtractor
from tiatoolbox.tools.tissuemask import OtsuTissueMasker
from tiatoolbox.wsicore.wsireader import VirtualWSIReader, WSIReader
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

import split
from download import (DEFAULT_LOG_DIR, DEFAULT_MANIFEST, REPO_ROOT, SOURCES, load_manifest,
                      paths_for, setup_logging, slide_progress, write_json)

DEFAULT_CONFIG = REPO_ROOT / "configs" / "tiling.yaml"
# Bump whenever a change to this file alters the outputs; slides tiled by an older version are redone.
CODE_VERSION = 5
CONFIG_KEYS = {"version", "tile_px", "target_mpp", "mask_mpp", "mask_method", "mask_median_px",
               "mask_close_px", "min_tissue_fraction", "max_tiles_per_slide", "max_glass_fraction",
               "min_blur", "seed", "jpeg_quality"}
MASK_METHODS = ("saturation_multiotsu", "otsu_gray")

log = logging.getLogger("tile")

# TIAToolbox warns on every tile it enlarges, which is expected for slides coarser than
# target_mpp (e.g. 0.5015 um/px). Drop only that message; other TIAToolbox warnings still show.
for _handler in logging.getLogger().handlers:
    _handler.addFilter(lambda record: "Scale > 1" not in record.getMessage())


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
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1),
                        help="slides processed in parallel (default: %(default)s)")
    parser.add_argument("--dry-run", action="store_true",
                        help="report done and pending slides, then exit")
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

    v = cfg
    checks = {
        "version": isinstance(v["version"], str) and v["version"].strip() != "",
        "tile_px": is_int(v["tile_px"]) and v["tile_px"] > 0,
        "target_mpp": is_num(v["target_mpp"]) and v["target_mpp"] > 0,
        "mask_mpp": is_num(v["mask_mpp"]) and v["mask_mpp"] > 0,
        "mask_method": v["mask_method"] in MASK_METHODS,
        "mask_median_px": is_int(v["mask_median_px"]) and v["mask_median_px"] >= 3
                          and v["mask_median_px"] % 2 == 1,
        "mask_close_px": is_int(v["mask_close_px"]) and v["mask_close_px"] >= 1,
        "min_tissue_fraction": is_num(v["min_tissue_fraction"]) and 0 <= v["min_tissue_fraction"] <= 1,
        "max_tiles_per_slide": is_int(v["max_tiles_per_slide"]) and v["max_tiles_per_slide"] >= 1,
        "max_glass_fraction": is_num(v["max_glass_fraction"]) and 0 <= v["max_glass_fraction"] <= 1,
        "min_blur": is_num(v["min_blur"]) and v["min_blur"] >= 0,
        "seed": is_int(v["seed"]) and v["seed"] >= 0,
        "jpeg_quality": is_int(v["jpeg_quality"]) and 1 <= v["jpeg_quality"] <= 100,
    }
    bad = [key for key, ok in checks.items() if not ok]
    if bad:
        raise RuntimeError("bad config values: " + ", ".join(f"{k}={cfg[k]!r}" for k in bad))
    return cfg, hashlib.sha256(text.encode()).hexdigest(), text


def output_paths(root, row):
    """Return (tiles .h5, QC .png, receipt .json) for a slide."""
    folder = root / "patches" / row["source"] / row["patient_id"]
    return (folder / f"{row['slide_id']}.h5", folder / f"{row['slide_id']}_qc.png",
            root / "state" / "tiling" / row["source"] / f"{row['slide_id']}.json")


def tiled(root, row, config_sha):
    """True if the slide was tiled with this config and CODE_VERSION and its outputs are intact.

    Intact means the .h5 file has the size recorded in the receipt and the QC image exists. The
    full SHA-256 in the receipt is not rechecked on every run.
    """
    h5_path, qc_path, receipt = output_paths(root, row)
    try:
        record = json.loads(receipt.read_text())
        size = h5_path.stat().st_size
    except (OSError, ValueError):
        return False
    return (record.get("config_sha256") == config_sha
            and record.get("code_version") == CODE_VERSION
            and record.get("h5_bytes") == size and qc_path.exists())


def tile_quality(img):
    """Return (glass fraction, blur score) for an RGB tile.

    glass: fraction of near-white pixels with almost no color (HSV saturation < 8, value > 200);
    background sits at saturation 0-3, while even very pale tissue is around 13 or more.
    blur: variance of the Laplacian of the grayscale tile; near 0 for empty or featureless tiles.
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    glass = float(((hsv[..., 1] < 8) & (hsv[..., 2] > 200)).mean())
    blur = float(cv2.Laplacian(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY), cv2.CV_64F).var())
    return glass, blur


def tissue_mask(thumb, cfg):
    """Return (0/1 uint8 tissue mask, threshold) for an RGB thumbnail.

    "saturation_multiotsu": tissue is colored and background is white or gray, so threshold the
    HSV saturation channel (median-blurred). A two-class Otsu cut lands between pale and strongly
    stained tissue on heavily stained slides, so use the lower cut of a three-class Otsu
    (background / pale / strong), then close small gaps.
    "otsu_gray": TIAToolbox's OtsuTissueMasker (grayscale Otsu, no cleanup); threshold is None.
    """
    if cfg["mask_method"] == "otsu_gray":
        return OtsuTissueMasker().fit_transform([thumb])[0].astype(np.uint8), None
    saturation = cv2.cvtColor(np.ascontiguousarray(thumb[..., :3]), cv2.COLOR_RGB2HSV)[..., 1]
    saturation = cv2.medianBlur(saturation, cfg["mask_median_px"])
    levels = len(np.unique(saturation))
    if levels < 3:   # a three-class Otsu needs at least three distinct values (blank slide?)
        raise ValueError(f"mask: only {levels} saturation levels in the thumbnail")
    thresholds = threshold_multiotsu(saturation, classes=3)
    mask = (np.digitize(saturation, thresholds) >= 1).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (cfg["mask_close_px"],) * 2)
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel), float(thresholds[0])


def tissue_grid(wsi, cfg):
    """Mask a thumbnail and build TIAToolbox's tile grid over the tissue.

    Returns (thumbnail, mask, mask threshold, extractor). The extractor is a
    SlidingWindowPatchExtractor: a non-overlapping grid of `tile_px` tiles at `target_mpp`,
    keeping tiles where more than `min_tissue_fraction` of the tile is tissue (or all of it).
    `extractor.coordinate_list` holds the kept tiles as [x0, y0, x1, y1] at `target_mpp`;
    `extractor[i]` reads tile i.
    """
    thumb = wsi.slide_thumbnail(cfg["mask_mpp"], "mpp")
    mask, threshold = tissue_mask(thumb, cfg)
    extractor = SlidingWindowPatchExtractor(
        wsi, cfg["tile_px"], input_mask=VirtualWSIReader(mask, info=wsi.info, mode="bool"),
        resolution=cfg["target_mpp"], units="mpp", stride=cfg["tile_px"],
        min_mask_ratio=cfg["min_tissue_fraction"], within_bound=True)
    return thumb, mask, threshold, extractor


def write_qc(path, thumb, mask, coords, tile_px, scale_x, scale_y):
    """Save the thumbnail with the tissue-mask outline (green) and sampled tiles (blue).

    `coords` and `tile_px` are at target_mpp; `scale_x`, `scale_y` map them to thumbnail pixels.
    """
    image = np.ascontiguousarray(thumb.copy())
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(image, contours, -1, (0, 200, 0), 1)
    for x, y in coords:
        cv2.rectangle(image, (int(x * scale_x), int(y * scale_y)),
                      (int((x + tile_px) * scale_x), int((y + tile_px) * scale_y)), (0, 0, 255), 1)
    temporary = path.with_name(path.name + ".tmp.png")
    Image.fromarray(image).save(temporary)
    os.replace(temporary, path)


def tile_slide(root, row, cfg, config_sha, config_text):
    """Tile one slide (runs in a worker process). Returns a stats dict for its receipt."""
    started = time.monotonic()
    slide_path = paths_for(root, row)[0]
    h5_path, qc_path, _ = output_paths(root, row)
    h5_path.parent.mkdir(parents=True, exist_ok=True)

    wsi = WSIReader.open(slide_path)
    info = wsi.info
    if info.mpp is None or not all(np.isfinite(info.mpp)) or min(info.mpp) <= 0:
        raise ValueError(f"missing or invalid mpp: {info.mpp}")
    thumb, mask, threshold, extractor = tissue_grid(wsi, cfg)
    tissue = extractor.coordinate_list[:, :2].astype(np.int64)   # top-left at target_mpp
    if not len(tissue):
        raise ValueError("no tissue tiles")

    # Visit tissue tiles in a seeded random order and keep those that pass the quality filter,
    # until max_tiles_per_slide are kept. Dropped tiles are replaced, so the cap is still reached.
    # Tiles after the cap is reached are never examined, so n_dropped counts only the drops among
    # the n_examined tiles, not every low-quality tile on the slide.
    seed = int.from_bytes(hashlib.sha256(f"{cfg['seed']}|{row['slide_id']}".encode()).digest()[:8], "big")
    encoded, quality, dropped = {}, {}, []
    for index in np.random.default_rng(seed).permutation(len(tissue)):
        if len(encoded) == cfg["max_tiles_per_slide"]:
            break
        img = np.ascontiguousarray(extractor[int(index)][..., :3])
        glass, blur = tile_quality(img)
        if glass > cfg["max_glass_fraction"] or blur < cfg["min_blur"]:
            dropped.append(int(index))
            continue
        buffer = io.BytesIO()
        Image.fromarray(img).save(buffer, format="JPEG", quality=cfg["jpeg_quality"])
        encoded[int(index)] = np.frombuffer(buffer.getvalue(), dtype=np.uint8)
        quality[int(index)] = (glass, blur)
    if not encoded:
        raise ValueError(f"no tiles left after the quality filter ({len(dropped)} dropped)")
    chosen = np.sort(list(encoded))       # grid order, so neighbouring tiles stay together
    count = len(chosen)
    examined = count + len(dropped)
    coords = tissue[chosen]
    dropped_coords = tissue[np.sort(dropped)] if dropped else np.zeros((0, 2), dtype=np.int64)
    tile_px = cfg["tile_px"]
    # The level-0 region TIAToolbox reads for each tile (its own conversion, which rounds up).
    coords_level0 = np.array([wsi.bounds_at_resolution_to_baseline(
        [x, y, x + tile_px, y + tile_px], cfg["target_mpp"], "mpp")[:2] for x, y in coords],
        dtype=np.int64)

    temporary = h5_path.with_name(h5_path.name + ".tmp")
    with h5py.File(temporary, "w") as h5:
        tiles = h5.create_dataset("tiles", shape=(count,), dtype=h5py.vlen_dtype(np.uint8))
        for i, index in enumerate(chosen):
            tiles[i] = encoded[int(index)]
        h5.create_dataset("coords", data=coords)
        h5.create_dataset("coords_level0", data=coords_level0)
        h5.create_dataset("tissue_coords", data=tissue)
        h5.create_dataset("dropped_coords", data=dropped_coords)
        # Per-tile quality, aligned with tiles/coords (computed on the tile before JPEG encoding).
        h5.create_dataset("glass", data=np.array([quality[int(i)][0] for i in chosen], dtype=np.float32))
        h5.create_dataset("blur", data=np.array([quality[int(i)][1] for i in chosen], dtype=np.float32))
        attrs = {"slide_id": row["slide_id"], "patient_id": row["patient_id"],
                 "source": row["source"], "file_name": row["file_name"],
                 "source_mpp_x": float(info.mpp[0]), "source_mpp_y": float(info.mpp[1]),
                 "target_mpp": cfg["target_mpp"], "tile_px": tile_px,
                 "tile_um": tile_px * cfg["target_mpp"],
                 "coords_space": "top-left at target_mpp (TIAToolbox coord_space='resolution')",
                 "coords_level0_space": "top-left of the level-0 region read for each tile",
                 "grid": "tiatoolbox SlidingWindowPatchExtractor, stride = tile_px, within_bound",
                 "interpolation": "optimize (TIAToolbox default: area down, cubic up)",
                 "mask_method": cfg["mask_method"],
                 "mask_threshold": -1.0 if threshold is None else threshold,
                 "mask_fraction": float(mask.mean()),
                 "level_dimensions": json.dumps([list(map(int, d)) for d in info.level_dimensions]),
                 "level_downsamples": json.dumps([float(d) for d in info.level_downsamples]),
                 "n_tissue_tiles": len(tissue), "n_sampled": count, "n_examined": examined,
                 "n_dropped": len(dropped),
                 "n_dropped_scope": "drops among the n_examined tiles visited before the cap was reached",
                 "quality_filter": f"drop if glass > {cfg['max_glass_fraction']} or blur < {cfg['min_blur']}",
                 "tiling_version": cfg["version"], "code_version": CODE_VERSION,
                 "config_sha256": config_sha,
                 "config_yaml": config_text, "tiatoolbox": tiatoolbox.__version__,
                 "openslide_python": openslide.__version__,
                 "openslide": openslide.__library_version__, "python": platform.python_version()}
        h5.attrs.update(attrs)
    os.replace(temporary, h5_path)
    width, height = wsi.slide_dimensions(cfg["target_mpp"], "mpp")
    write_qc(qc_path, thumb, mask, coords, tile_px, mask.shape[1] / width, mask.shape[0] / height)

    with h5_path.open("rb") as handle:   # streamed in chunks, not read into memory at once
        digest = hashlib.file_digest(handle, "sha256").hexdigest()
    return {"slide_id": row["slide_id"], "source": row["source"], "patient_id": row["patient_id"],
            "file_name": row["file_name"], "h5": str(h5_path.relative_to(root)),
            "h5_sha256": digest, "h5_bytes": h5_path.stat().st_size,
            "n_tissue_tiles": len(tissue), "n_sampled": count, "n_examined": examined,
            "n_dropped": len(dropped), "source_mpp": float(info.mpp[0]), "mask_threshold": threshold,
            "config_sha256": config_sha, "code_version": CODE_VERSION, "tiling_version": cfg["version"],
            "seconds": round(time.monotonic() - started, 1),
            "tiled_at_utc": datetime.now(timezone.utc).isoformat()}


def summarize(root, files, config_sha, title):
    """Log a per-cohort table of downloaded, tiled and pending slides; return the receipts."""
    receipts = {}
    log.info("%s:", title)
    log.info("  %-6s %6s %10s %6s %7s %9s %8s", "cohort", "slides", "downloaded", "tiled",
             "pending", "tiles", "GB")
    for source in SOURCES:
        rows = [r for r in files if r["source"] == source]
        if not rows:
            continue
        done = [r for r in rows if tiled(root, r, config_sha)]
        for r in done:
            receipts[r["slide_id"]] = json.loads(output_paths(root, r)[2].read_text())
        downloaded = sum(slide_progress(root, r)[0] for r in rows)
        log.info("  %-6s %6d %10d %6d %7d %9d %8.2f", source, len(rows), downloaded, len(done),
                 len(rows) - len(done), sum(receipts[r["slide_id"]]["n_sampled"] for r in done),
                 sum(receipts[r["slide_id"]]["h5_bytes"] for r in done) / 1e9)
    return receipts


def main(argv=None):
    """Select slides, tile the pending ones in parallel, and write receipts and an index CSV.

    Returns the process exit code (see the module docstring).
    """
    args = parse_args(argv)
    args.root = args.root.expanduser().resolve()
    args.log_dir = args.log_dir.expanduser().resolve()
    log_path = setup_logging(log, args.log_dir, f"tile_{args.subset}")
    log.info("Command: %s", " ".join(sys.argv))
    if args.workers < 1:
        log.error("--workers must be >= 1: %d", args.workers)
        return 2

    try:
        cfg, config_sha, config_text = load_config(args.config.expanduser().resolve())
        log.info("Config %s (sha256 %s): %s", args.config, config_sha[:12],
                 json.dumps(cfg, sort_keys=True))
        log.info("TIAToolbox %s | OpenSlide %s | tile.py code version %d | workers %d | log %s",
                 tiatoolbox.__version__, openslide.__library_version__, CODE_VERSION,
                 args.workers, log_path)
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
    except (OSError, RuntimeError, ValueError, yaml.YAMLError) as error:
        log.error("%s", error)
        return 2

    summarize(args.root, files, config_sha, f"Start of run (subset {args.subset})")
    not_downloaded = [r for r in files if not slide_progress(args.root, r)[0]]
    pending = [r for r in files if r not in not_downloaded and not tiled(args.root, r, config_sha)]
    if not_downloaded:
        log.warning("%d selected slides are not downloaded and verified; skipping them",
                    len(not_downloaded))
    log.info("Tiling %d pending slides", len(pending))
    if args.dry_run:
        log.info("Dry run: nothing tiled.")
        return 0

    failed = []
    interrupted = False
    started = time.monotonic()
    with logging_redirect_tqdm(loggers=[log]), \
            ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(tile_slide, args.root, row, cfg, config_sha, config_text): row
                   for row in pending}
        bar = tqdm(total=len(pending), desc=f"subset {args.subset}", unit="slide",
                   dynamic_ncols=True, disable=not sys.stderr.isatty())
        try:
            for future in as_completed(futures):
                row = futures[future]
                try:
                    stats = future.result()
                except Exception as error:  # A bad slide must not stop the run.
                    failed.append(row)
                    log.error("%s %s failed: %s", row["patient_id"], row["file_name"], error)
                else:
                    write_json(output_paths(args.root, row)[2], stats)
                    log.info("%s %s: %d of %d tissue tiles (%d dropped), %.0f MB, %.0f s",
                             row["patient_id"], row["file_name"], stats["n_sampled"],
                             stats["n_tissue_tiles"], stats["n_dropped"],
                             stats["h5_bytes"] / 1e6, stats["seconds"])
                bar.update(1)
        except KeyboardInterrupt:
            interrupted = True
            pool.shutdown(wait=False, cancel_futures=True)
        finally:
            bar.close()

    log.info("Active time %.1f min", (time.monotonic() - started) / 60)
    receipts = summarize(args.root, files, config_sha, f"End of run (subset {args.subset})")
    # The index lists every tiled slide in the subset, not only this run's --sources/--slides.
    indexed = [json.loads(output_paths(args.root, r)[2].read_text())
               for r in subset_files if tiled(args.root, r, config_sha)]
    index = args.root / "patches" / f"index_{args.subset}.csv"
    if indexed:
        index.parent.mkdir(parents=True, exist_ok=True)
        rows = sorted(indexed, key=lambda r: (r["source"], r["patient_id"], r["slide_id"]))
        with index.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        log.info("Index of %d tiled slides: %s", len(rows), index)
    elif index.exists():
        # No slide is tiled with the current config and code, so the old index is stale.
        index.unlink()
        log.warning("No slides tiled with the current config; removed stale %s", index)
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
