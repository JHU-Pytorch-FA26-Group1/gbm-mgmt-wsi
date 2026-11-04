"""Download the frozen GBM WSI cohort (full or one team subset) with resume and verification.

Examples:
    python data_prep/download.py --root ~/Bioinformatics/WSI/data --subset B --dry-run
    python data_prep/download.py --root ~/Bioinformatics/WSI/data --subset B
    python data_prep/download.py --root /content/drive/MyDrive/GBM_MGMT --subset A --minutes 600

Exit codes: 0 = every selected slide verified, 1 = some slides still pending or failed,
2 = setup or source-metadata problem, 130 = interrupted with Ctrl-C.
"""
import argparse
import csv
import hashlib
import io
import json
import logging
import os
import re
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import requests
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

import split

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MANIFEST = REPO_ROOT / "manifests" / "collection_manifest.json"
DEFAULT_LOG_DIR = REPO_ROOT / "logs"
SOURCES = ("TCGA", "CPTAC")

# The frozen slide list. Both subsets are only complementary if everyone uses this exact file.
MANIFEST_SHA256 = "0ba3b0f4d801938d1934c1ce92cfcc69e2fb172cd1ed584bddcaab9c9980e330"
EXPECTED_COUNTS = {"TCGA": {"patients": 151, "slides": 320},
                   "CPTAC": {"patients": 83, "slides": 204}}

CHUNK_BYTES = 8 * 1024**2
CHECKPOINT_BYTES = 64 * 1024**2
FREE_SPACE_MARGIN = 10e9
ATTEMPTS = 3
TIFF_HEADERS = (b"II*\x00", b"MM\x00*", b"II+\x00", b"MM\x00+")
BAR_OPTIONS = dict(unit="B", unit_scale=True, unit_divisor=1000, dynamic_ncols=True,
                   disable=not sys.stderr.isatty())

log = logging.getLogger("download")


class CollectionPaused(Exception):
    """Raised to stop cleanly at the time limit or --max-mb; checkpoints are kept."""


class SourceChanged(Exception):
    """The server's file no longer matches the frozen manifest; stop and inspect."""


def parse_args(argv=None):
    """Parse command-line options; pass `argv` as a list to test without a shell."""
    parser = argparse.ArgumentParser(
        description="Download GBM whole-slide images with resume and checksum verification.")
    parser.add_argument("--root", type=Path, required=True,
                        help="data folder; raw/, state/ and manifests/ are created here")
    parser.add_argument("--subset", choices=["full", "A", "B"], required=True,
                        help="full cohort, or team subset A or B")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST,
                        help="frozen slide manifest (default: %(default)s)")
    parser.add_argument("--sources", nargs="+", choices=SOURCES, default=list(SOURCES),
                        help="cohorts to download, in this order (default: TCGA CPTAC)")
    parser.add_argument("--minutes", type=float, default=None,
                        help="stop after this many minutes (default: run until done)")
    parser.add_argument("--max-mb", type=float, default=None,
                        help="stop after downloading this many MB in this run (for a pilot)")
    parser.add_argument("--dry-run", action="store_true",
                        help="print the selection summary and exit without downloading")
    parser.add_argument("--log-dir", type=Path, default=DEFAULT_LOG_DIR,
                        help="folder for run .log files (default: %(default)s)")
    return parser.parse_args(argv)


def load_manifest(path):
    """Read the frozen manifest and check its hash and contents.

    Returns (raw bytes, parsed manifest). Raises RuntimeError if the file is not the
    pinned version, or if slide counts, sizes or file names are wrong."""
    data = path.read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    if digest != MANIFEST_SHA256:
        raise RuntimeError(f"manifest sha256 {digest[:12]} != {MANIFEST_SHA256[:12]}: {path}")
    manifest = json.loads(data)
    files = manifest["files"]
    if len({(row["source"], row["slide_id"]) for row in files}) != len(files):
        raise RuntimeError("duplicate slide IDs in manifest")
    for source, expected in EXPECTED_COUNTS.items():
        rows = [row for row in files if row["source"] == source]
        found = {"patients": len({row["patient_id"] for row in rows}), "slides": len(rows)}
        if found != expected:
            raise RuntimeError(f"{source}: {found} != {expected}")
    for row in files:
        if row["size_bytes"] <= 0 or not row["file_name"].lower().endswith(".svs"):
            raise RuntimeError(f"bad manifest row: {row['slide_id']}")
        for field in ("patient_id", "slide_id", "file_name"):
            if Path(row[field]).name != row[field] or row[field] in (".", ".."):
                raise RuntimeError(f"unsafe {field}: {row[field]!r}")
    return data, manifest


# --------------------------------------------------------------------------- per-slide state

def flush_file(handle):
    """Flush `handle` to disk, tolerating filesystems without fsync (e.g. a Drive mount)."""
    handle.flush()
    try:
        os.fsync(handle.fileno())
    except OSError as error:
        if error.errno not in (22, 95):  # Some mounted filesystems do not implement fsync.
            raise


def write_json(path, value):
    """Write `value` as JSON atomically: to a temporary file first, then renamed over `path`."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w") as handle:
        json.dump(value, handle, indent=2)
        flush_file(handle)
    os.replace(temporary, path)


def fingerprint(row):
    """Hash of the manifest fields that identify a slide; state files from other slides never match."""
    fields = {key: row.get(key, "") for key in
              ("source", "patient_id", "slide_id", "file_name", "url", "size_bytes", "md5", "etag")}
    return hashlib.sha256(json.dumps(fields, sort_keys=True).encode()).hexdigest()


def paths_for(root, row):
    """Return (final .svs, .part file, checkpoint JSON, verification receipt JSON) for a slide."""
    folder = root / "raw" / row["source"] / row["patient_id"]
    state = root / "state" / row["source"]
    return (folder / row["file_name"], folder / (row["file_name"] + ".part"),
            state / (row["slide_id"] + ".partial.json"), state / (row["slide_id"] + ".verified.json"))


def slide_progress(root, row):
    """(complete, saved_bytes): complete means verified; otherwise bytes in a checked partial."""
    final, partial, checkpoint, receipt = paths_for(root, row)
    try:
        record = json.loads(receipt.read_text()) if receipt.exists() else {}
        if (final.exists() and final.stat().st_size == row["size_bytes"]
                and record.get("fingerprint") == fingerprint(row)
                and record.get("size_bytes") == row["size_bytes"]):
            return True, row["size_bytes"]
        record = json.loads(checkpoint.read_text()) if checkpoint.exists() else {}
        if partial.exists() and record.get("fingerprint") == fingerprint(row):
            count = int(record.get("saved_bytes", 0))
            if 0 <= count <= partial.stat().st_size and count <= row["size_bytes"]:
                return False, count
    except (ValueError, OSError):
        pass
    return False, 0


def digest_prefix(path, count, stop_check=lambda: False):
    """Return SHA-256 and MD5 hashers fed with the first `count` bytes of `path`."""
    sha, md5 = hashlib.sha256(), hashlib.md5()
    with path.open("rb") as handle:
        remaining = count
        while remaining:
            if stop_check():
                raise CollectionPaused()
            chunk = handle.read(min(CHUNK_BYTES, remaining))
            if not chunk:
                raise ValueError("partial file shorter than its checkpoint")
            sha.update(chunk)
            md5.update(chunk)
            remaining -= len(chunk)
    return sha, md5


def verify_file(row, path, sha=None, md5=None, stop_check=lambda: False):
    """Check size, TIFF/SVS header and publisher MD5 (when available); return a receipt.

    Pass `sha`/`md5` hashers that already cover the whole file to skip re-reading it.
    Raises ValueError on any mismatch."""
    if path.stat().st_size != row["size_bytes"]:
        raise ValueError(f"size {path.stat().st_size} != {row['size_bytes']}")
    with path.open("rb") as handle:
        if handle.read(4) not in TIFF_HEADERS:
            raise ValueError("not a TIFF/SVS file")
    if sha is None:
        sha, md5 = digest_prefix(path, row["size_bytes"], stop_check)
    if row.get("md5") and md5.hexdigest().lower() != row["md5"].lower():
        raise ValueError(f"md5 {md5.hexdigest()} != {row['md5']}")
    return {"fingerprint": fingerprint(row), "size_bytes": row["size_bytes"],
            "sha256": sha.hexdigest(), "md5_computed": md5.hexdigest(),
            "publisher_md5_checked": bool(row.get("md5")),
            "verified_at_utc": datetime.now(timezone.utc).isoformat()}


def checkpoint_partial(root, row, handle, count, sha, etag, modified):
    """Truncate the .part file to `count` checked bytes and record that position for resuming."""
    # Only the bytes included in the recorded digest are kept.
    handle.truncate(count)
    flush_file(handle)
    write_json(paths_for(root, row)[2], {
        "fingerprint": fingerprint(row), "saved_bytes": count, "prefix_sha256": sha.hexdigest(),
        "etag": etag, "last_modified": modified,
        "updated_at_utc": datetime.now(timezone.utc).isoformat()})


# --------------------------------------------------------------------------- downloading

def download_one(root, row, bars, stop_check=lambda: False, byte_budget=None):
    """Download, resume and verify one slide. `bars` = (file_bar, overall_bar)."""
    def show(count):
        """Move both progress bars so the file bar shows `count` bytes."""
        step = count - bars[0].n
        for bar in bars:
            bar.update(step)

    final, partial, checkpoint, receipt = paths_for(root, row)
    final.parent.mkdir(parents=True, exist_ok=True)
    if slide_progress(root, row)[0]:
        return "skipped"
    if final.exists():
        # Recover a completed rename if the run died before its receipt was saved.
        write_json(receipt, verify_file(row, final, stop_check=stop_check))
        show(row["size_bytes"])
        return "verified existing file"

    try:
        state = json.loads(checkpoint.read_text()) if checkpoint.exists() else {}
    except (ValueError, OSError):
        state = {}
    count = slide_progress(root, row)[1]
    if partial.exists() and partial.stat().st_size:
        sha, md5 = digest_prefix(partial, count, stop_check)
        if state.get("prefix_sha256") != sha.hexdigest():
            log.warning("%s: saved partial failed its checksum; restarting from zero", row["slide_id"])
            count = 0
            sha, md5 = hashlib.sha256(), hashlib.md5()
    else:
        count = 0
        sha, md5 = hashlib.sha256(), hashlib.md5()
    show(count)

    start_count = count
    etag = state.get("etag", row.get("etag", ""))
    modified = state.get("last_modified", row.get("last_modified", ""))
    with partial.open("r+b" if partial.exists() else "w+b") as handle:
        handle.seek(count)
        handle.truncate(count)
        try:
            if count < row["size_bytes"]:
                if stop_check():
                    raise CollectionPaused()
                headers = {"Accept-Encoding": "identity"}
                if count:
                    headers["Range"] = f"bytes={count}-"
                    if etag and not etag.startswith("W/"):
                        headers["If-Range"] = etag
                    elif modified:
                        headers["If-Range"] = modified
                with requests.get(row["url"], headers=headers, stream=True,
                                  allow_redirects=True, timeout=(15, 45)) as response:
                    response.raise_for_status()
                    if response.headers.get("Content-Encoding", "identity") != "identity":
                        raise SourceChanged("unexpected compression")
                    current_etag = response.headers.get("ETag", "")
                    expected_etag = row.get("etag", "")
                    if expected_etag and current_etag and expected_etag != current_etag:
                        raise SourceChanged(f"ETag {current_etag} != {expected_etag}")
                    expected_modified = row.get("last_modified", "")
                    current_modified = response.headers.get("Last-Modified", "")
                    if (not expected_etag and expected_modified and current_modified
                            and expected_modified != current_modified):
                        raise SourceChanged(f"Last-Modified {current_modified} != {expected_modified}")
                    if response.status_code == 206:
                        match = re.fullmatch(r"(?:bytes )?(\d+)-(\d+)/(\d+)",
                                             response.headers.get("Content-Range", ""))
                        if not match or int(match[1]) != count or int(match[3]) != row["size_bytes"]:
                            raise SourceChanged("invalid byte-range response")
                        if int(match[2]) + 1 != row["size_bytes"]:
                            raise SourceChanged("incomplete byte-range response")
                    elif response.status_code == 200:
                        if count:
                            log.warning("%s: server ignored resume; restarting this file", row["slide_id"])
                            count = start_count = 0
                            sha, md5 = hashlib.sha256(), hashlib.md5()
                            handle.seek(0)
                            handle.truncate(0)
                            show(0)
                        length = response.headers.get("Content-Length")
                        if length and int(length) != row["size_bytes"]:
                            raise SourceChanged(f"Content-Length {length} != {row['size_bytes']}")
                    else:
                        raise SourceChanged(f"HTTP {response.status_code}")

                    etag = current_etag or etag
                    modified = current_modified or modified
                    checkpoint_partial(root, row, handle, count, sha, etag, modified)
                    last_checkpoint = count
                    for chunk in response.iter_content(chunk_size=CHUNK_BYTES):
                        if stop_check() or (byte_budget is not None and count - start_count >= byte_budget):
                            raise CollectionPaused()
                        if not chunk:
                            continue
                        if count + len(chunk) > row["size_bytes"]:
                            raise SourceChanged("server sent more bytes than expected")
                        handle.write(chunk)
                        sha.update(chunk)
                        md5.update(chunk)
                        count += len(chunk)
                        show(count)
                        if count - last_checkpoint >= CHECKPOINT_BYTES:
                            checkpoint_partial(root, row, handle, count, sha, etag, modified)
                            last_checkpoint = count
        finally:
            checkpoint_partial(root, row, handle, count, sha, etag, modified)

    try:
        record = verify_file(row, partial, sha, md5, stop_check)
    except ValueError:
        # Keep a failed download for inspection; the next attempt starts fresh.
        os.replace(partial, partial.with_name(partial.name + f".bad-{time.time_ns()}"))
        if checkpoint.exists():
            checkpoint.unlink()
        raise
    os.replace(partial, final)
    write_json(receipt, record)
    if checkpoint.exists():
        checkpoint.unlink()
    return "completed"


def summarize(root, files, progress_csv, title):
    """Log a per-cohort table, write the per-slide progress CSV, and return the rows."""
    table = []
    for row in files:
        done, count = slide_progress(root, row)
        table.append({"source": row["source"], "patient_id": row["patient_id"],
                      "slide_id": row["slide_id"], "file_name": row["file_name"],
                      "size_bytes": row["size_bytes"], "saved_bytes": count,
                      "status": "complete" if done else ("partial" if count else "pending"),
                      "local_path": str(paths_for(root, row)[0]),
                      "publisher_md5_available": bool(row.get("md5"))})
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=list(table[0]))
    writer.writeheader()
    writer.writerows(table)
    temporary = progress_csv.with_suffix(".csv.tmp")
    temporary.write_text(buffer.getvalue())
    os.replace(temporary, progress_csv)

    log.info("%s:", title)
    log.info("  %-6s %8s %6s %8s %7s %7s %9s %9s", "cohort", "patients", "slides",
             "complete", "partial", "pending", "total_GB", "saved_GB")
    for source in SOURCES:
        rows = [r for r in table if r["source"] == source]
        if rows:
            log.info("  %-6s %8d %6d %8d %7d %7d %9.2f %9.2f", source,
                     len({r["patient_id"] for r in rows}), len(rows),
                     sum(r["status"] == "complete" for r in rows),
                     sum(r["status"] == "partial" for r in rows),
                     sum(r["status"] == "pending" for r in rows),
                     sum(r["size_bytes"] for r in rows) / 1e9,
                     sum(r["saved_bytes"] for r in rows) / 1e9)
    return table


# --------------------------------------------------------------------------- collection

def collect(args, files, transfer_log, progress_csv):
    """Download pending slides of the selected sources. Returns an exit code."""
    root = args.root
    selected = [r for r in files if r["source"] in args.sources]
    saved = {r["slide_id"]: slide_progress(root, r) for r in selected}
    pending = sorted((r for r in selected if not saved[r["slide_id"]][0]),
                     key=lambda r: (args.sources.index(r["source"]), r["size_bytes"], r["slide_id"]))
    log.info("Downloading %d pending slides", len(pending))

    started = time.monotonic()
    deadline = started + args.minutes * 60 if args.minutes is not None else float("inf")
    stop_check = lambda: time.monotonic() >= deadline
    budget_total = None if args.max_mb is None else args.max_mb * 1e6
    used_bytes = 0
    failed = []
    stopped = ""

    overall = tqdm(total=sum(r["size_bytes"] for r in selected), desc=f"subset {args.subset}",
                   initial=sum(count for _, count in saved.values()), position=0, **BAR_OPTIONS)
    try:
        with logging_redirect_tqdm(loggers=[log]):
            for row in pending:
                if stop_check():
                    stopped = "time limit reached"
                    break
                if budget_total is not None and used_bytes >= budget_total:
                    stopped = "--max-mb reached"
                    break
                file_bar = tqdm(total=row["size_bytes"], initial=saved[row["slide_id"]][1],
                                desc=row["patient_id"], position=1, leave=False, **BAR_OPTIONS)
                try:
                    for attempt in range(1, ATTEMPTS + 1):
                        before = slide_progress(root, row)[1]
                        attempt_start = time.monotonic()
                        outcome, message = "error", ""
                        try:
                            budget = None if budget_total is None else budget_total - used_bytes
                            outcome = download_one(root, row, (file_bar, overall), stop_check, budget)
                        except CollectionPaused:
                            outcome = "paused"
                        except KeyboardInterrupt:
                            outcome = "paused"
                            raise
                        except (requests.RequestException, ValueError) as error:
                            message = str(error)
                            log.warning("%s %s attempt %d/%d failed: %s", row["patient_id"],
                                        row["file_name"], attempt, ATTEMPTS, message)
                            if attempt < ATTEMPTS:
                                time.sleep(min(2 ** (attempt - 1), 4))
                        finally:
                            new_bytes = max(0, slide_progress(root, row)[1] - before)
                            used_bytes += new_bytes
                            with transfer_log.open("a") as handle:
                                handle.write(json.dumps({
                                    "time_utc": datetime.now(timezone.utc).isoformat(),
                                    "source": row["source"], "slide_id": row["slide_id"],
                                    "new_saved_bytes": new_bytes,
                                    "elapsed_seconds": time.monotonic() - attempt_start,
                                    "outcome": outcome, "error": message}) + "\n")
                                flush_file(handle)
                        if outcome == "paused":
                            stopped = "time limit reached" if stop_check() else "--max-mb reached"
                            break
                        if outcome != "error":
                            log.info("%s %s: %s", row["patient_id"], row["file_name"], outcome)
                            break
                    else:
                        failed.append(row)
                        log.error("%s %s failed after %d attempts", row["patient_id"],
                                  row["file_name"], ATTEMPTS)
                finally:
                    file_bar.close()
                if stopped:
                    break
    except KeyboardInterrupt:
        stopped = "interrupted"
    finally:
        overall.close()

    elapsed = time.monotonic() - started
    if stopped:
        log.info("Stopped: %s. Checkpoints are saved; rerun the same command to resume.", stopped)
    log.info("Active time %.1f min; downloaded %.2f GB (%.1f MB/s)", elapsed / 60,
             used_bytes / 1e9, used_bytes / 1e6 / elapsed if elapsed else 0.0)
    table = summarize(root, files, progress_csv, f"End of run (subset {args.subset})")
    for row in failed:
        log.error("Failed: %s %s %s", row["source"], row["patient_id"], row["file_name"])

    if stopped == "interrupted":
        return 130
    unfinished = [r for r in table if r["source"] in args.sources and r["status"] != "complete"]
    if unfinished:
        log.warning("%d selected slides are not complete; rerun to retry.", len(unfinished))
        return 1
    log.info("All selected slides are verified.")
    return 0


def main(argv=None):
    """Set up logging and the data folder, check the manifest, then dry-run or collect.

    Returns the process exit code (see the module docstring)."""
    args = parse_args(argv)
    args.root = args.root.expanduser().resolve()
    args.manifest = args.manifest.expanduser().resolve()
    args.log_dir = args.log_dir.expanduser().resolve()
    args.sources = tuple(dict.fromkeys(args.sources))
    # Same state file names as the team notebook, so existing progress carries over.
    suffix = "" if args.subset == "full" else f"_half_{args.subset}"
    progress_csv = args.root / "state" / f"progress{suffix}.csv"
    transfer_log = args.root / "state" / f"transfer_log{suffix}.jsonl"

    args.log_dir.mkdir(parents=True, exist_ok=True)
    log_path = args.log_dir / f"download_{args.subset}_{datetime.now():%Y%m%d_%H%M%S}.log"
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    console = logging.StreamHandler(sys.stderr)
    console.setFormatter(logging.Formatter("%(levelname)s %(message)s"))
    log.setLevel(logging.INFO)
    log.handlers = [file_handler, console]
    log.propagate = False
    log.info("Command: %s", " ".join(sys.argv))
    log.info("Root %s | subset %s | sources %s | log %s", args.root, args.subset,
             " ".join(args.sources), log_path)

    try:
        # Refuse folders that sync to the cloud; warn where iCloud sync is optional.
        home = Path.home()
        cloud = [home / "Library" / "CloudStorage", home / "Library" / "Mobile Documents",
                 home / "Google Drive"]
        if any(args.root.is_relative_to(p.resolve()) for p in cloud if p.exists()):
            raise RuntimeError(f"cloud-synced root: {args.root}")
        if any(args.root.is_relative_to(p.resolve()) for p in (home / "Desktop", home / "Documents")
               if p.exists()):
            log.warning("Root is under Desktop/Documents; confirm iCloud sync is off.")
        for folder in ("raw/TCGA", "raw/CPTAC", "manifests", "state/TCGA", "state/CPTAC"):
            (args.root / folder).mkdir(parents=True, exist_ok=True)

        manifest_bytes, manifest = load_manifest(args.manifest)
        all_files = manifest["files"]
        log.info("Manifest sha256 OK: %d slides", len(all_files))
        assignments = split.build_patient_split(all_files)

        # Same manifests/ records as the team notebook; existing copies must match byte for byte.
        buffer = io.StringIO()
        writer = csv.DictWriter(buffer, fieldnames=sorted({k for row in all_files for k in row}))
        writer.writeheader()
        writer.writerows(all_files)
        records = {"collection_manifest.json": manifest_bytes,
                   "collection_manifest.csv": buffer.getvalue().encode("utf-8")}
        records.update({name: text.encode("utf-8") for name, text
                        in split.assignment_files(manifest, assignments).items()})
        for name, content in records.items():
            path = args.root / "manifests" / name
            if path.exists():
                if path.read_bytes() != content:
                    raise RuntimeError(f"differs from frozen copy: {path}")
            else:
                path.with_name(name + ".tmp").write_bytes(content)
                os.replace(path.with_name(name + ".tmp"), path)

        for t in split.subset_totals(assignments):
            log.info("Split %s: subset %s = %d patients, %d slides, %.3f GB",
                     split.SPLIT_VERSION, t["subset"], t["patients"], t["slides"], t["GB"])
        files = split.select_files(all_files, args.subset, assignments)
        table = summarize(args.root, files, progress_csv, f"Start of run (subset {args.subset})")

        remaining = sum(r["size_bytes"] - r["saved_bytes"] for r in table if r["source"] in args.sources)
        need = remaining if args.max_mb is None else min(remaining, args.max_mb * 1e6)
        free = shutil.disk_usage(args.root).free
        log.info("Remaining for %s: %.2f GB; free disk: %.1f GB", "/".join(args.sources),
                 remaining / 1e9, free / 1e9)
        if need < remaining:
            log.info("This run: up to %.2f GB (--max-mb)", need / 1e9)
        short = need and free < need + FREE_SPACE_MARGIN
        if args.dry_run:
            if short:
                log.warning("Not enough disk space for a real run (%.0f GB + %.0f GB margin needed)",
                            need / 1e9, FREE_SPACE_MARGIN / 1e9)
            log.info("Dry run: nothing downloaded.")
            return 0
        if short:
            raise RuntimeError(f"need {need / 1e9:.0f} GB + {FREE_SPACE_MARGIN / 1e9:.0f} GB; "
                               f"free {free / 1e9:.0f} GB")
        return collect(args, files, transfer_log, progress_csv)
    except (OSError, RuntimeError, SourceChanged) as error:
        log.error("%s", error)
        return 2


if __name__ == "__main__":
    sys.exit(main())
