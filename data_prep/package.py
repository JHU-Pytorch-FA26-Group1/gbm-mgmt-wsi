"""Copy one export and exactly its model tiles into a portable folder for training on any machine.

Examples:
    python data_prep/package.py --export ~/Bioinformatics/WSI/data/patches/tiling-v7/export/qc-v1/B_<time> \
        --out ~/Bioinformatics/WSI/packages/B_qc-v1
    python data_prep/package.py --export <export folder> --out <new folder> --allow-partial   # a preview

Run it on the processing machine (it needs the data root of the export, and the slides for any model
tile that was not stored at tiling, such as edge tiles a reviewer accepted). The package holds:
  tiles/{source}/{patient_id}/{slide_id}.h5   the model tiles of each slide, PNG bytes copied unchanged
                                               (tiles reread from the slide are encoded losslessly)
  modeling_patches.csv                        the export's rows, pointing at these files (all stored)
  patient_index.csv, slide_status.csv, export_settings.json   copied from the export
  package_manifest.json                       SHA-256 and size of every file, settings and versions
Every tile is checked against its recorded pixel hash as it is written. Training needs only this
folder, h5py, Pillow and NumPy (no slides, no OpenSlide). loader.load_modeling(package) checks the
modeling list and file sizes on every load; loader.verify_package(package) rehashes every file (run it
once after copying the package to a new machine). combine.py joins packages, e.g. halves A and B.

The export must be current (nothing it recorded has changed since) and complete (no blocked patients)
unless --allow-partial is given. The output folder must not exist yet.

Exit codes: 0 = packaged, 2 = setup problem (stale or incomplete export, existing output, bad tile).
"""
import argparse
import csv
import hashlib
import io
import itertools
import json
import logging
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np
import yaml
from PIL import Image

import export
import loader
from common import DEFAULT_TILING_CONFIG, load_tiling_config, sha256_file
from download import DEFAULT_LOG_DIR, setup_logging, write_json

PACKAGE_FIELDS = [*export.MODEL_FIELDS, "stored_at_tiling"]
COPIED = ("patient_index.csv", "slide_status.csv", "export_settings.json")

log = logging.getLogger("package")


def parse_args(argv=None):
    """Parse command-line options; pass `argv` as a list to test without a shell."""
    parser = argparse.ArgumentParser(description="Package an export and its model tiles.")
    parser.add_argument("--export", type=Path, required=True, help="export folder written by export.py")
    parser.add_argument("--out", type=Path, required=True, help="new folder for the package")
    parser.add_argument("--allow-partial", action="store_true",
                        help="package an export with blocked patients (a preview, not a final dataset)")
    parser.add_argument("--tiling-config", type=Path, default=DEFAULT_TILING_CONFIG,
                        help="tiling config of the export (default: %(default)s)")
    parser.add_argument("--log-dir", type=Path, default=DEFAULT_LOG_DIR,
                        help="folder for run .log files (default: %(default)s)")
    return parser.parse_args(argv)


def write_slide(rows, root, dest, cfg, reader):
    """Write one slide's model tiles to `dest`; return the rows pointing at it and the reread count."""
    first = rows[0]
    payloads, hashes, reread = [], [], 0
    source = h5py.File(root / first["h5"], "r") if any(r["stored"] == "True" for r in rows) else None
    try:
        for row in rows:
            if row["stored"] == "True":
                data = source["tiles"][int(row["h5_index"])].tobytes()
                rgb = np.asarray(Image.open(io.BytesIO(data)).convert("RGB"))
            else:
                rgb = reader.read(row)   # from the slide; read() checks the hash
                buffer = io.BytesIO()
                Image.fromarray(rgb).save(buffer, format="PNG", compress_level=cfg["png_compress_level"])
                data = buffer.getvalue()
                reread += 1
            if hashlib.sha256(rgb.tobytes()).hexdigest() != row["pixel_sha256"]:
                raise ValueError(f"{row['patch_id']}: pixels differ from the recorded hash")
            payloads.append(np.frombuffer(data, dtype=np.uint8))
            hashes.append(row["pixel_sha256"])
    finally:
        if source is not None:
            source.close()
    dest.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(dest, "w") as h5:
        tiles = h5.create_dataset("tiles", shape=(len(payloads),), dtype=h5py.vlen_dtype(np.uint8))
        for i, payload in enumerate(payloads):
            tiles[i] = payload
        h5.create_dataset("pixel_sha256", data=np.array(hashes, dtype="S64"))
        h5.create_dataset("patch_id", data=np.array([r["patch_id"] for r in rows], dtype="S"))
        h5.attrs.update({"slide_id": first["slide_id"], "patient_id": first["patient_id"],
                         "source": first["source"], "tiling_version": cfg["version"],
                         "tile_px": cfg["tile_px"], "target_mpp": cfg["target_mpp"],
                         "tile_format": "PNG (lossless)",
                         "pixel_sha256_of": "decoded RGB uint8 array bytes (height, width, 3)"})
    relative = str(dest.relative_to(dest.parents[3]))
    out_rows = [{**row, "stored_at_tiling": row["stored"], "stored": True, "h5": relative, "h5_index": i}
                for i, row in enumerate(rows)]
    return out_rows, reread


def main(argv=None):
    """Package an export. Returns the process exit code."""
    args = parse_args(argv)
    export_dir = args.export.expanduser().resolve()
    out = args.out.expanduser().resolve()
    log_path = setup_logging(log, args.log_dir.expanduser().resolve(), "package")
    log.info("Command: %s", " ".join(sys.argv))
    try:
        if loader.folder_kind(export_dir) != "export":
            raise RuntimeError(f"not an export folder: {export_dir}")
        settings = json.loads((export_dir / "export_settings.json").read_text())
        loader.modeling_fold(export_dir)
        stale = loader.stale_inputs(export_dir)
        if stale:
            raise RuntimeError(f"export is stale ({'; '.join(stale[:3])}); rerun export.py")
        if not settings["complete"] and not args.allow_partial:
            raise RuntimeError("export has blocked patients; fix them, or use --allow-partial for a preview")
        cfg, config_sha, _ = load_tiling_config(args.tiling_config.expanduser().resolve())
        if config_sha != settings["config_sha256"]:
            raise RuntimeError("the tiling config differs from the one this export was made with")
        if out.exists():
            raise RuntimeError(f"output folder already exists: {out}")
    except (OSError, RuntimeError, ValueError, KeyError, yaml.YAMLError) as error:
        log.error("%s", error)
        return 2
    root = loader.export_root(export_dir)
    log.info("Packaging %s (%s, %d model tiles) into %s | log %s", export_dir.name,
             "complete" if settings["complete"] else "PARTIAL", settings["counts"]["model_tiles"], out,
             log_path)

    temporary = out.with_name(out.name + ".part")
    shutil.rmtree(temporary, ignore_errors=True)
    temporary.mkdir(parents=True)
    n_tiles = n_reread = n_slides = 0
    seen = set()
    try:
        with (export_dir / "modeling_patches.csv").open(newline="") as handle, \
                (temporary / "modeling_patches.csv").open("w", newline="") as target, \
                loader.TileReader(root, args.tiling_config.expanduser().resolve()) as reader:
            writer = csv.DictWriter(target, fieldnames=PACKAGE_FIELDS)
            writer.writeheader()
            for slide_id, group in itertools.groupby(csv.DictReader(handle), key=lambda r: r["slide_id"]):
                if slide_id in seen:
                    raise ValueError(f"rows of slide {slide_id} are not together in modeling_patches.csv")
                seen.add(slide_id)
                rows = list(group)
                dest = temporary / "tiles" / rows[0]["source"] / rows[0]["patient_id"] / f"{slide_id}.h5"
                out_rows, reread = write_slide(rows, root, dest, cfg, reader)
                writer.writerows(out_rows)
                n_tiles += len(out_rows)
                n_reread += reread
                n_slides += 1
        for name in COPIED:
            shutil.copy2(export_dir / name, temporary / name)
        files = {str(p.relative_to(temporary)): {"sha256": sha256_file(p), "bytes": p.stat().st_size}
                 for p in sorted(temporary.rglob("*")) if p.is_file()}
        import slides
        write_json(temporary / loader.PACKAGE_MANIFEST, {
            "kind": "package", "source_export": export_dir.name, "subset": settings["subset"],
            "complete": settings["complete"], "tiling_version": settings["tiling_version"],
            "config_sha256": settings["config_sha256"], "code_version": settings["code_version"],
            "policy_version": settings["policy"]["version"], "policy_sha256": settings["policy_sha256"],
            "folds_sha256": settings["folds_sha256"], "validation_fold": settings["validation_fold"],
            "export_settings_sha256": sha256_file(export_dir / "export_settings.json"),
            "counts": {"slides": n_slides, "tiles": n_tiles, "tiles_reread_from_slides": n_reread},
            "library_versions": slides.library_versions(),
            "created_at_utc": datetime.now(timezone.utc).isoformat(), "files": files})
        os.replace(temporary, out)
    except (OSError, ValueError, KeyError) as error:
        log.error("%s; nothing written to %s", error, out)
        shutil.rmtree(temporary, ignore_errors=True)
        return 2
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    size = sum(f["bytes"] for f in files.values())
    log.info("Packaged %d tiles from %d slides (%d reread from slides), %.2f GB: %s", n_tiles, n_slides,
             n_reread, size / 1e9, out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
