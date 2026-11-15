"""Per-tile GrandQC artifact fractions for tiled slides, read from GrandQC artifact masks.

    python data_prep/grandqc.py --root ~/Coding/WSI/data --subset B

GrandQC (https://github.com/cpath-ukk/grandqc) labels every pixel of a slide at 1.0 um/px:
1 tissue, 2 fold, 3 dark spot, 4 pen, 5 edge/air bubble, 6 out of focus, 7 background, and 0 where it
made no call. Masks are read from <masks>/<source>/<file_name>_mask.png, the naming of the released
TCGA masks (mask_qc/<file name>.svs_mask.png); put masks made for CPTAC in <masks>/CPTAC/ the same way.

For every tissue and edge tile of each indexed slide, it writes the fraction of the tile's level-0
footprint in each class, and `assessed` (the fraction not labelled 0; footprint outside the mask counts
as unassessed), to patches/<tiling version>/grandqc/<source>/<patient_id>/<slide_id>_grandqc.csv. A
receipt beside it (<slide_id>_grandqc.json) records the mask's SHA-256 and size, the slide geometry,
the tiling CSV it was computed from, and the slide's pen fraction: pen pixels over pixels labelled 1-6
(tissue and artifacts; background and unassessed pixels are left out, so glass does not dilute it).

A mask whose size does not match the slide at --mask-mpp (within 1%) is refused. A slide is redone when
its mask or tiling CSV changed; otherwise it is skipped. When a slide's mask is missing or fails, its old
receipt is removed, so qc.py never uses data from a mask that is no longer there. grandqc_<subset>.csv lists every slide's status.
qc.py uses these files when the policy sets `grandqc: true`.

Exit code: 0 = every slide done, 1 = some slides missing a mask or failed, 2 = setup problem.
"""
import argparse
import csv
import json
import logging
import sys
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

from common import DEFAULT_TILING_CONFIG, read_rows, sha256_file, tiling_version, write_rows
from download import DEFAULT_LOG_DIR, setup_logging, write_json

log = logging.getLogger("grandqc")

CLASSES = {0: "unassessed", 1: "tissue", 2: "fold", 3: "darkspot", 4: "pen", 5: "bubble", 6: "oof",
           7: "background"}
FIELDS = ["patch_id", "role", "assessed", *[CLASSES[k] for k in range(1, 8)]]
ROLES = ("tissue", "edge")
MASK_TOLERANCE = 0.01   # mask size may differ from the expected size by this fraction
SOURCES = ("TCGA", "CPTAC")


def parse_args(argv=None):
    """Parse command-line options; pass `argv` as a list to test without a shell."""
    parser = argparse.ArgumentParser(description="Per-tile GrandQC artifact fractions from GrandQC masks.")
    parser.add_argument("--root", type=Path, required=True, help="data folder used by tile.py")
    parser.add_argument("--subset", choices=["full", "A", "B"], required=True,
                        help="the subset whose tiling index to read")
    parser.add_argument("--masks", type=Path, default=None,
                        help="folder with one subfolder per cohort (default: <root>/external/grandqc)")
    parser.add_argument("--mask-mpp", type=float, default=1.0,
                        help="resolution of the masks in um/px (default: %(default)s)")
    parser.add_argument("--sources", nargs="+", choices=SOURCES, default=list(SOURCES),
                        help="cohorts to process (default: TCGA CPTAC)")
    parser.add_argument("--workers", type=int, default=1,
                        help="slides in parallel; each holds one decoded mask (~1 GB) (default: 1)")
    parser.add_argument("--tiling-config", type=Path, default=DEFAULT_TILING_CONFIG,
                        help="tiling config whose outputs to read (default: %(default)s)")
    parser.add_argument("--log-dir", type=Path, default=DEFAULT_LOG_DIR,
                        help="folder for run .log files (default: %(default)s)")
    return parser.parse_args(argv)


def mask_path(masks, row):
    """Where the GrandQC mask of a slide is expected."""
    return Path(masks) / row["source"] / f"{row['file_name']}_mask.png"


def output_paths(root, version, row):
    """(per-tile CSV, receipt JSON) of a slide's GrandQC fractions."""
    folder = Path(root) / "patches" / version / "grandqc" / row["source"] / row["patient_id"]
    return folder / f"{row['slide_id']}_grandqc.csv", folder / f"{row['slide_id']}_grandqc.json"


def current(root, version, row):
    """Return (receipt, "") if a slide's GrandQC files match its tiling CSV, else (None, reason)."""
    csv_path, receipt_path = output_paths(root, version, row)
    if not receipt_path.exists() or not csv_path.exists():
        return None, "no GrandQC data"
    receipt = json.loads(receipt_path.read_text())
    if receipt.get("patches_csv_sha256") != row["csv_sha256"]:
        return None, "GrandQC data is from an older tiling CSV"
    if sha256_file(csv_path) != receipt.get("csv_sha256"):
        return None, "GrandQC CSV changed since it was written"
    return receipt, ""


def read_fractions(root, version, row):
    """{patch_id: {class: fraction, "assessed": fraction}} from a slide's GrandQC CSV."""
    csv_path, _ = output_paths(root, version, row)
    return {r["patch_id"]: {k: float(r[k]) for k in FIELDS[2:]} for r in read_rows(csv_path)}


def slide_geometry(h5_path):
    """(level-0 width, height, mpp x, mpp y) recorded by tile.py in the slide's .h5 attributes."""
    import h5py
    with h5py.File(h5_path, "r") as h5:
        width, height = json.loads(h5.attrs["level_dimensions"])[0]
        return int(width), int(height), float(h5.attrs["source_mpp_x"]), float(h5.attrs["source_mpp_y"])


def class_counts(mask):
    """Pixels per class 0-7, counted in row blocks so no full-size int64 copy is made."""
    import numpy as np
    counts = np.zeros(256, dtype=np.int64)
    for start in range(0, mask.shape[0], 1024):
        counts += np.bincount(mask[start:start + 1024].ravel(), minlength=256)
    if counts[8:].any():
        raise ValueError(f"unexpected mask labels: {np.nonzero(counts[8:])[0] + 8}")
    return counts[:8]


def process_slide(root, version, row, masks, mask_mpp):
    """Compute and write one slide's GrandQC fractions (runs in a worker process); return a status row."""
    import numpy as np
    from PIL import Image

    status = {"source": row["source"], "patient_id": row["patient_id"], "slide_id": row["slide_id"],
              "file_name": row["file_name"], "status": "", "reason": "", "pen_fraction": ""}
    path = mask_path(masks, row)
    csv_path, receipt_path = output_paths(root, version, row)
    if not path.exists():
        receipt_path.unlink(missing_ok=True)   # data from a mask that is gone must not be used
        return {**status, "status": "missing", "reason": f"no mask at {path}"}
    mask_sha = sha256_file(path)
    receipt, _ = current(root, version, row)
    if receipt and receipt.get("mask_sha256") == mask_sha and receipt.get("mask_mpp") == mask_mpp:
        return {**status, "status": "ok", "reason": "unchanged", "pen_fraction": receipt["pen_fraction"]}
    try:
        width, height, mpp_x, mpp_y = slide_geometry(Path(root) / row["h5"])
        Image.MAX_IMAGE_PIXELS = None   # masks of whole slides are far above PIL's default limit
        with Image.open(path) as image:
            if image.mode not in ("L", "P"):
                raise ValueError(f"mask mode {image.mode}, expected 8-bit labels")
            mask = np.asarray(image)
        rows_px, cols_px = mask.shape
        expected = (width * mpp_x / mask_mpp, height * mpp_y / mask_mpp)
        if (abs(cols_px / expected[0] - 1) > MASK_TOLERANCE
                or abs(rows_px / expected[1] - 1) > MASK_TOLERANCE):
            raise ValueError(f"mask size {cols_px}x{rows_px}, expected {expected[0]:.0f}x{expected[1]:.0f}")
        scale_x, scale_y = cols_px / width, rows_px / height
        counts = class_counts(mask)
        labelled = int(counts[1:7].sum())
        pen_fraction = float(counts[4] / labelled) if labelled else 0.0
        out_rows = []
        for tile in read_rows(Path(root) / row["csv"]):
            if tile["role"] not in ROLES:
                continue
            c0, c1 = round(int(tile["x0"]) * scale_x), round(int(tile["x1"]) * scale_x)
            r0, r1 = round(int(tile["y0"]) * scale_y), round(int(tile["y1"]) * scale_y)
            area = max(1, (c1 - c0) * (r1 - r0))
            window = mask[max(r0, 0):max(min(r1, rows_px), 0), max(c0, 0):max(min(c1, cols_px), 0)]
            tile_counts = np.bincount(window.ravel(), minlength=8)[:8]
            out = {"patch_id": tile["patch_id"], "role": tile["role"],
                   "assessed": round(float(tile_counts[1:].sum() / area), 4)}
            out.update({CLASSES[k]: round(float(tile_counts[k] / area), 4) for k in range(1, 8)})
            out_rows.append(out)
        del mask
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = csv_path.with_name(csv_path.name + ".tmp")
        write_rows(temporary, out_rows, FIELDS)
        temporary.replace(csv_path)
        write_json(receipt_path, {
            "slide_id": row["slide_id"], "file_name": row["file_name"], "mask": str(path),
            "mask_sha256": mask_sha, "mask_mpp": mask_mpp, "mask_size": [cols_px, rows_px],
            "level0_size": [width, height], "mpp": [mpp_x, mpp_y], "scale": [scale_x, scale_y],
            "patches_csv_sha256": row["csv_sha256"], "csv": str(csv_path.relative_to(root)),
            "csv_sha256": sha256_file(csv_path), "tiles": len(out_rows),
            "class_pixels": {CLASSES[k]: int(counts[k]) for k in range(8)}, "pen_fraction": pen_fraction,
            "computed_at_utc": datetime.now(timezone.utc).isoformat()})
        return {**status, "status": "ok", "pen_fraction": round(pen_fraction, 5)}
    except Exception as error:   # one bad mask must not stop the other slides
        receipt_path.unlink(missing_ok=True)   # nor leave data from an older mask in use
        return {**status, "status": "failed", "reason": f"{type(error).__name__}: {error}"}


def main(argv=None):
    """Compute GrandQC fractions for every indexed slide of the subset. Returns the process exit code."""
    args = parse_args(argv)
    args.root = args.root.expanduser().resolve()
    masks = (args.masks.expanduser().resolve() if args.masks else args.root / "external" / "grandqc")
    log_path = setup_logging(log, args.log_dir.expanduser().resolve(), f"grandqc_{args.subset}")
    log.info("Command: %s", " ".join(sys.argv))
    try:
        if args.workers < 1 or not args.mask_mpp > 0:
            raise ValueError("--workers must be at least 1 and --mask-mpp positive")
        version = tiling_version(args.tiling_config.expanduser().resolve())
        index_path = args.root / "patches" / version / f"index_{args.subset}.csv"
        with index_path.open(newline="") as handle:
            slides = [r for r in csv.DictReader(handle) if r["source"] in args.sources]
        if not slides:
            raise RuntimeError(f"no tiled {'/'.join(args.sources)} slides in {index_path}")
    except (OSError, RuntimeError, ValueError, KeyError) as error:
        log.error("%s", error)
        return 2
    log.info("Tiling %s | %d slides | masks %s at %s um/px | %d workers | log %s", version, len(slides),
             masks, args.mask_mpp, args.workers, log_path)
    jobs = [(args.root, version, row, masks, args.mask_mpp) for row in slides]
    if args.workers == 1:
        results = [process_slide(*job) for job in jobs]
    else:
        with ProcessPoolExecutor(args.workers) as pool:
            results = list(pool.map(process_slide, *zip(*jobs)))
    for r in results:
        if r["status"] != "ok":
            log.warning("%s %s: %s", r["slide_id"], r["status"], r["reason"])
    out = args.root / "patches" / version / "grandqc" / f"grandqc_{args.subset}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    write_rows(out, results, list(results[0]))
    done = sum(r["status"] == "ok" for r in results)
    log.info("%d of %d slides done (%d missing a mask, %d failed); status in %s", done, len(results),
             sum(r["status"] == "missing" for r in results), sum(r["status"] == "failed" for r in results), out)
    return 0 if done == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
