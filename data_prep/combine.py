"""Join packages (e.g. halves A and B, processed on different machines) into one model input.

Example:
    python data_prep/combine.py --parts A=/data/packages/A_qc-v1 B=/data/packages/B_qc-v1 \
        --out /data/packages/combined_qc-v1

The parts must be intact packages (package.py) made with the same tiling config, code version, QC
policy and fold map (the repository's manifests/patient_folds.csv), with no patient in two parts.
The combination is complete when every part is complete and together they account for every
patient in the fold map; otherwise it is refused unless --allow-partial is given (a preview).

Writes into --out (a new folder): modeling_patches.csv (every part's rows, with `part` and `part_dir`,
the part's folder relative to --out), patient_index.csv (with `part`) and combined_settings.json.
No tiles are copied: keep the part folders where they are relative to --out (for example all inside
one folder), and loader.TileReader(--out) reads each row from its part.

Exit codes: 0 = combined, 2 = setup problem (damaged or mismatched parts, overlap, incomplete).
"""
import argparse
import csv
import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import loader
from common import DEFAULT_FOLDS, read_folds, read_rows, sha256_file, write_rows
from download import DEFAULT_LOG_DIR, setup_logging, write_json

SHARED = ("tiling_version", "config_sha256", "code_version", "policy_sha256", "folds_sha256", "validation_fold")

log = logging.getLogger("combine")


def parse_args(argv=None):
    """Parse command-line options; pass `argv` as a list to test without a shell."""
    parser = argparse.ArgumentParser(description="Join packages into one model input.")
    parser.add_argument("--parts", nargs="+", required=True, metavar="NAME=FOLDER",
                        help="packages to join, each as name=folder (e.g. A=... B=...)")
    parser.add_argument("--out", type=Path, required=True, help="new folder for the combination")
    parser.add_argument("--allow-partial", action="store_true",
                        help="join incomplete parts or leave fold-map patients missing (a preview)")
    parser.add_argument("--folds", type=Path, default=DEFAULT_FOLDS,
                        help="shared patient fold map (default: %(default)s)")
    parser.add_argument("--log-dir", type=Path, default=DEFAULT_LOG_DIR,
                        help="folder for run .log files (default: %(default)s)")
    return parser.parse_args(argv)


def main(argv=None):
    """Check and join the parts. Returns the process exit code."""
    args = parse_args(argv)
    out = args.out.expanduser().resolve()
    log_path = setup_logging(log, args.log_dir.expanduser().resolve(), "combine")
    log.info("Command: %s", " ".join(sys.argv))
    try:
        parts = {}
        for spec in args.parts:
            name, _, folder = spec.partition("=")
            if not name or not folder or name in parts:
                raise RuntimeError(f"use distinct name=folder pairs: {spec}")
            folder = Path(folder).expanduser().resolve()
            if loader.folder_kind(folder) != "package":
                raise RuntimeError(f"{name}: not a package folder: {folder}")
            loader.modeling_fold(folder)
            problems = loader.package_problems(folder)
            if problems:
                raise RuntimeError(f"{name}: {'; '.join(problems[:3])}")
            parts[name] = (folder, json.loads((folder / loader.PACKAGE_MANIFEST).read_text()))
        first = next(iter(parts.values()))[1]
        for name, (_, manifest) in parts.items():
            differ = [k for k in SHARED if manifest[k] != first[k]]
            if differ:
                raise RuntimeError(f"{name} differs from the other parts in {differ}")
        folds_path = args.folds.expanduser().resolve()
        if sha256_file(folds_path) != first["folds_sha256"]:
            raise RuntimeError(f"the parts were made with a different fold map than {folds_path}")
        folds = read_folds(folds_path)
        owner, indexes = {}, {}
        for name, (folder, _) in parts.items():
            indexes[name] = read_rows(folder / "patient_index.csv")
            for row in indexes[name]:
                if row["patient_id"] in owner:
                    raise RuntimeError(f"patient {row['patient_id']} is in {owner[row['patient_id']]} and {name}")
                owner[row["patient_id"]] = name
        missing = sorted(p for (_, p) in folds if p not in owner)
        incomplete = [name for name, (_, m) in parts.items() if not m["complete"]]
        complete = not missing and not incomplete
        if not complete and not args.allow_partial:
            raise RuntimeError(f"incomplete: parts with blocked patients {incomplete}, "
                               f"{len(missing)} fold-map patients in no part; --allow-partial for a preview")
        if out.exists():
            raise RuntimeError(f"output folder already exists: {out}")
    except (OSError, RuntimeError, ValueError, KeyError) as error:
        log.error("%s", error)
        return 2

    out.mkdir(parents=True)
    fields, n_tiles = None, 0
    model_path = out / "modeling_patches.csv"
    with model_path.open("w", newline="") as target:
        for name, (folder, _) in parts.items():
            relative = os.path.relpath(folder, out)
            with (folder / "modeling_patches.csv").open(newline="") as handle:
                reader = csv.DictReader(handle)
                if fields is None:
                    fields = [*reader.fieldnames, "part", "part_dir"]
                    writer = csv.DictWriter(target, fieldnames=fields)
                    writer.writeheader()
                for row in reader:
                    writer.writerow({**row, "part": name, "part_dir": relative})
                    n_tiles += 1
    write_rows(out / "patient_index.csv",
               [{**row, "part": name} for name, rows in indexes.items() for row in rows])
    write_json(out / loader.COMBINED_SETTINGS, {
        "kind": "combined", "complete": complete, "missing_patients": missing,
        "incomplete_parts": incomplete, **{k: first[k] for k in SHARED},
        "parts": {name: {"path": os.path.relpath(folder, out), "subset": m["subset"],
                         "manifest_sha256": sha256_file(folder / loader.PACKAGE_MANIFEST),
                         "modeling_sha256": m["files"]["modeling_patches.csv"]["sha256"]}
                  for name, (folder, m) in parts.items()},
        "counts": {"tiles": n_tiles, "patients": len(owner)},
        "modeling_sha256": sha256_file(model_path), "created_at_utc": datetime.now(timezone.utc).isoformat()})
    log.info("Combined %d parts: %d tiles, %d patients (%s) into %s | log %s", len(parts), n_tiles,
             len(owner), "complete" if complete else f"PARTIAL, {len(missing)} patients missing", out,
             log_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
