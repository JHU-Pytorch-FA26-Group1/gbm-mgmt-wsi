"""Human review records: patch decisions, artifact regions and slide sign-off, kept apart from tiling.

Use from the notebook (or Python):
    import review
    review.append_patch_review(ROOT, "tiling-v7", patch_id, "reject", "Couger", "fold")
    review.append_region(ROOT, "tiling-v7", "TCGA", slide_id, (x0, y0, x1, y1), "reject", "Couger", "pen")
    review.sign_off_slide(ROOT, "tiling-v7", "TCGA", slide_id, "approve", "Couger", mask=True, focus=True)

Records are appended to CSVs under <root>/review/<tiling version>/; nothing is edited in place, and
the latest record for a patch or slide wins. export.py applies them on top of the QC decisions:
  QC decision  ->  artifact region (any reject region rejects, else review)  ->  patch review
  ->  slide exclusion (removes the whole slide).
Defect types are technical damage only. Necrosis, hemorrhage, microvascular proliferation and low
cellularity are tissue, not defects. Keep MGMT labels out of review.
"""
import csv
import os
import re
from collections import namedtuple
from datetime import datetime, timezone
from pathlib import Path

DEFECTS = {"none", "background", "very_dark", "blur", "exposure", "fold", "tear", "bubble", "pen",
           "scanner_stripe", "stitching_seam", "missing_tile", "other"}
CHECKS = ("mask", "focus", "pale", "background", "artifacts")
PATCH_FIELDS = ["patch_id", "decision", "defect_type", "reviewer", "notes", "time_utc"]
REGION_FIELDS = ["source", "slide_id", "x0", "y0", "x1", "y1", "decision", "defect_type", "reviewer",
                 "notes", "time_utc"]
SIGNOFF_FIELDS = ["source", "slide_id", "decision", "reviewer", *(f"{c}_checked" for c in CHECKS),
                  "notes", "time_utc"]
PATCH_ID = re.compile(r"^(TCGA|CPTAC)_[0-9a-f]{16}_\d+_\d+$")

Reviews = namedtuple("Reviews", "patches regions signoffs")


def review_dir(root, version):
    """Where the review CSVs for one tiling version live."""
    return Path(root) / "review" / version


def _append(path, fields, record):
    """Append one record to a CSV, writing the header for a new file, and flush it to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        if handle.tell() == 0:
            writer.writeheader()
        writer.writerow(record)
        handle.flush()
        os.fsync(handle.fileno())


def _now():
    return datetime.now(timezone.utc).isoformat()


def append_patch_review(root, version, patch_id, decision, reviewer, defect_type="none", notes=""):
    """Record a reviewer's decision (accept / reject / review) for one tile."""
    if not PATCH_ID.match(patch_id):
        raise ValueError(f"not a patch ID: {patch_id!r}")
    if decision not in ("accept", "reject", "review") or defect_type not in DEFECTS or not reviewer.strip():
        raise ValueError(f"bad review: {decision!r}, {defect_type!r}, reviewer {reviewer!r}")
    _append(review_dir(root, version) / "patch_reviews.csv", PATCH_FIELDS,
            dict(zip(PATCH_FIELDS, [patch_id, decision, defect_type, reviewer, notes, _now()])))


def append_region(root, version, source, slide_id, bounds, decision, reviewer, defect_type="other",
                  notes=""):
    """Record a technical artifact as a level-0 box (x0, y0, x1, y1); tiles it touches get `decision`."""
    x0, y0, x1, y1 = (int(v) for v in bounds)
    if source not in ("TCGA", "CPTAC") or x0 < 0 or y0 < 0 or x1 <= x0 or y1 <= y0:
        raise ValueError(f"bad region: {source}, {bounds}")
    if decision not in ("reject", "review") or defect_type not in DEFECTS or not reviewer.strip():
        raise ValueError(f"bad region review: {decision!r}, {defect_type!r}, reviewer {reviewer!r}")
    _append(review_dir(root, version) / "artifact_regions.csv", REGION_FIELDS,
            dict(zip(REGION_FIELDS, [source, slide_id, x0, y0, x1, y1, decision, defect_type, reviewer,
                                     notes, _now()])))


def sign_off_slide(root, version, source, slide_id, decision, reviewer, notes="", **checks):
    """Record that a slide was reviewed: approve, or exclude it (with a reason in `notes`).

    Optional checks record what was looked at: mask, focus, pale, background, artifacts (True/False).
    Sign-off is not required for every slide; it records which slides a person reviewed.
    """
    if source not in ("TCGA", "CPTAC") or decision not in ("approve", "exclude") or not reviewer.strip():
        raise ValueError(f"bad sign-off: {source}, {decision!r}, reviewer {reviewer!r}")
    if set(checks) - set(CHECKS) or any(not isinstance(v, bool) for v in checks.values()):
        raise ValueError(f"checks must be True/False for {CHECKS}: {checks}")
    if decision == "exclude" and not notes.strip():
        raise ValueError("record why the slide is excluded in notes")
    _append(review_dir(root, version) / "slide_signoff.csv", SIGNOFF_FIELDS,
            dict(source=source, slide_id=slide_id, decision=decision, reviewer=reviewer, notes=notes,
                 time_utc=_now(), **{f"{c}_checked": checks.get(c, False) for c in CHECKS}))


def _rows(path):
    if not path.exists():
        return []
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def load_reviews(root, version):
    """All review records for a tiling version; the latest record per patch and per slide wins."""
    folder = review_dir(root, version)
    patches, regions, signoffs = {}, {}, {}
    for row in _rows(folder / "patch_reviews.csv"):
        if row["decision"] not in ("accept", "reject", "review") or row["defect_type"] not in DEFECTS:
            raise ValueError(f"invalid patch review row: {row}")
        patches[row["patch_id"]] = row
    for row in _rows(folder / "artifact_regions.csv"):
        if row["decision"] not in ("reject", "review") or row["defect_type"] not in DEFECTS:
            raise ValueError(f"invalid region row: {row}")
        regions.setdefault((row["source"], row["slide_id"]), []).append(row)
    for row in _rows(folder / "slide_signoff.csv"):
        if row["decision"] not in ("approve", "exclude"):
            raise ValueError(f"invalid sign-off row: {row}")
        signoffs[(row["source"], row["slide_id"])] = row
    return Reviews(patches, regions, signoffs)


def review_files(root, version):
    """The review CSVs that exist, for recording their hashes with an export."""
    folder = review_dir(root, version)
    return [p for p in (folder / "patch_reviews.csv", folder / "artifact_regions.csv",
                        folder / "slide_signoff.csv") if p.exists()]


def resolve(tile, decision, reasons, reviews):
    """Apply reviews on top of a tile's QC decision; return (decision, reasons, decided_by).

    `tile` needs source, slide_id, patch_id and its level-0 box x0, y0, x1, y1. A slide exclusion
    returns decision "excluded".
    """
    key = (tile["source"], tile["slide_id"])
    decided_by = "qc"
    hits = [r for r in reviews.regions.get(key, [])
            if max(int(tile["x0"]), int(r["x0"])) < min(int(tile["x1"]), int(r["x1"]))
            and max(int(tile["y0"]), int(r["y0"])) < min(int(tile["y1"]), int(r["y1"]))]
    if hits:
        decision = "reject" if any(r["decision"] == "reject" for r in hits) else "review"
        reasons = "region:" + ",".join(sorted({r["defect_type"] for r in hits}))
        decided_by = "region"
    patch = reviews.patches.get(tile["patch_id"])
    if patch is not None:
        decision, reasons, decided_by = patch["decision"], "patch_review:" + patch["defect_type"], "patch"
    signoff = reviews.signoffs.get(key)
    if signoff is not None and signoff["decision"] == "exclude":
        decision, reasons, decided_by = "excluded", "slide_excluded", "slide"
    return decision, reasons, decided_by
