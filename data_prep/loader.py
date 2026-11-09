"""Read model tiles for training and evaluation, sample patient bags, and fit Macenko per fold.

Not used by tiling. Works on three kinds of folder, all read with the same calls:
  export     patches/<tiling>/export/<policy>/<subset_time>/ on the processing machine (export.py)
  package    a portable copy of one export with its tiles (package.py): what goes to each platform
  combined   several packages, e.g. halves A and B, joined by combine.py

    import loader
    tiles = loader.with_roles(loader.load_modeling(FOLDER), fold=0)   # train / validation / external
    reader = loader.TileReader(FOLDER)        # one per data-loader worker process
    bag = loader.training_sample(tiles, patient_id, epoch)          # 64 tiles, the same for every backbone
    images = reader.read_bag(bag)

Every read checks the tile's pixel hash against the one recorded at tiling, so a tile is exactly the
pixels that were scored and QC'd. In a package every model tile is stored, so only the package's .h5
files are needed. On the processing machine, a tile with stored = False is reread from its slide.

load_modeling refuses a folder whose recorded inputs have changed (an export: review records,
audits, QC decisions, the fold map, the modeling list; a package or combination: its modeling list
and file sizes, with verify_package for a full hash check), and an incomplete export (blocked
patients) unless allow_partial=True.

Macenko is a training-time transform: fit it once per fold on a reference tile from that fold's
training patients, then normalize each tile before augmentation. A failed transform is logged and
raised; a raw tile is never passed on as if it had been normalized.
"""
import csv
import hashlib
import io
import json
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from PIL import Image

from common import DEFAULT_FOLDS, DEFAULT_TILING_CONFIG, load_tiling_config, sha256_file

PACKAGE_MANIFEST = "package_manifest.json"
COMBINED_SETTINGS = "combined_settings.json"
BLOCK_TILES = 8   # spatial sampling blocks are BLOCK_TILES x BLOCK_TILES tiles


def folder_kind(folder):
    """'package', 'combined' or 'export', from the files the folder holds."""
    folder = Path(folder)
    if (folder / PACKAGE_MANIFEST).exists():
        return "package"
    if (folder / COMBINED_SETTINGS).exists():
        return "combined"
    if (folder / "export_settings.json").exists():
        return "export"
    raise ValueError(f"not an export, package or combined folder: {folder}")


def export_root(export_dir):
    """The data root of an export folder: <root>/patches/<tiling>/export/<policy>/<subset_time>."""
    return Path(export_dir).resolve().parents[4]


def stale_inputs(export_dir, folds_path=DEFAULT_FOLDS):
    """What changed since an export was written: a list of messages, empty if it is current.

    Compares the modeling list and every input export.py recorded (review CSVs, audits, the QC
    record, the fold map) with the files now under the export's data root, including review or
    audit files added since. Only meaningful on the processing machine.
    """
    import export   # same definition of the inputs as export.py
    export_dir = Path(export_dir)
    settings = json.loads((export_dir / "export_settings.json").read_text())
    changes = []
    if sha256_file(export_dir / "modeling_patches.csv") != settings["modeling_sha256"]:
        changes.append("modeling_patches.csv changed")
    now = export.input_hashes(export_root(export_dir), settings["tiling_version"],
                              settings["policy"]["version"], settings["subset"], folds_path)
    then = settings["inputs"]
    for key in sorted(set(then) | set(now)):
        if key not in now:
            changes.append(f"{key} removed")
        elif key not in then:
            changes.append(f"{key} added")
        elif now[key] != then[key]:
            changes.append(f"{key} changed")
    return changes


def package_problems(package_dir, full=False):
    """What is wrong with a package: a list of messages, empty if it is intact.

    Checks the modeling list's hash and that every listed file exists with its recorded size; with
    full=True, rehashes every file (reads the whole package; run it once after copying).
    """
    package_dir = Path(package_dir)
    manifest = json.loads((package_dir / PACKAGE_MANIFEST).read_text())
    problems = []
    for name, record in manifest["files"].items():
        path = package_dir / name
        if not path.exists():
            problems.append(f"{name} missing")
        elif path.stat().st_size != record["bytes"]:
            problems.append(f"{name} size changed")
        elif (full or name == "modeling_patches.csv") and sha256_file(path) != record["sha256"]:
            problems.append(f"{name} changed")
    return problems


def verify_package(package_dir):
    """Rehash every file of a package (or of every part of a combined folder); return the problems."""
    package_dir = Path(package_dir)
    if folder_kind(package_dir) == "combined":
        settings = json.loads((package_dir / COMBINED_SETTINGS).read_text())
        return [f"{name}: {p}" for name, part in settings["parts"].items()
                for p in package_problems(package_dir / part["path"], full=True)]
    return package_problems(package_dir, full=True)


def folder_problems(folder, allow_partial=False):
    """Freshness and completeness problems of an export, package or combined folder."""
    folder = Path(folder)
    kind = folder_kind(folder)
    if kind == "export":
        problems = stale_inputs(folder)
        complete = json.loads((folder / "export_settings.json").read_text()).get("complete", False)
    elif kind == "package":
        problems = package_problems(folder)
        complete = json.loads((folder / PACKAGE_MANIFEST).read_text())["complete"]
    else:
        settings = json.loads((folder / COMBINED_SETTINGS).read_text())
        problems = [f"{name}: {p}" for name, part in settings["parts"].items()
                    for p in package_problems(folder / part["path"])]
        problems += [f"{name}: package_manifest.json changed" for name, part in settings["parts"].items()
                     if sha256_file(folder / part["path"] / PACKAGE_MANIFEST) != part["manifest_sha256"]]
        if sha256_file(folder / "modeling_patches.csv") != settings["modeling_sha256"]:
            problems.append("modeling_patches.csv changed")
        complete = settings["complete"]
    if not complete and not allow_partial:
        problems.append("incomplete: some patients are blocked (allow_partial=True for a preview)")
    return problems


def load_modeling(folder, check=True, allow_partial=False):
    """The model tiles of an export, package or combined folder, as a DataFrame (one row per tile).

    With check=True (the default), raises ValueError if the folder is stale or damaged (see
    folder_problems), or incomplete unless allow_partial=True.
    """
    if check:
        problems = folder_problems(folder, allow_partial)
        if problems:
            raise ValueError(f"{Path(folder).name}: {'; '.join(problems[:3])}"
                             f"{' ...' if len(problems) > 3 else ''}")
    return pd.read_csv(Path(folder) / "modeling_patches.csv",
                       dtype={"fold": str, "pixel_sha256": str, "part_dir": str},
                       keep_default_na=False)


def with_roles(tiles, fold):
    """Add `role` for a validation fold: external (CPTAC), validation (this fold) or train."""
    if fold not in range(5):
        raise ValueError(f"fold must be 0-4: {fold}")
    role = np.where(tiles["source"] == "CPTAC", "external",
                    np.where(tiles["fold"] == str(fold), "validation", "train"))
    return tiles.assign(role=role)


def training_sample(tiles, patient_id, epoch, seed=0, count=64, block_px=BLOCK_TILES * 224):
    """The tiles of one patient's training bag for one epoch, as rows of `tiles`.

    Spreads the bag over the patient's slides (round robin) and, within each slide, over spatial
    blocks of `block_px` (8 x 8 tiles by default; sampling strata, not labels). The draw depends
    only on seed, epoch and patient, so every backbone gets the same tile IDs. A patient with fewer
    than `count` tiles gets all of them, then the rest drawn again with replacement.
    """
    if not isinstance(epoch, int) or epoch < 0 or not isinstance(count, int) or count < 1:
        raise ValueError(f"epoch must be >= 0 and count >= 1: {epoch}, {count}")
    rows = tiles[tiles["patient_id"] == patient_id]
    if rows.empty:
        raise ValueError(f"no tiles for patient {patient_id}")
    if "role" in rows and set(rows["role"]) != {"train"}:
        raise ValueError(f"{patient_id} is not a training patient for this fold")
    key = hashlib.sha256(f"{seed}|{epoch}|{patient_id}".encode()).digest()
    rng = np.random.default_rng(int.from_bytes(key[:8], "big"))
    queues = []
    for _, slide in rows.groupby("slide_id", sort=True):
        blocks = [list(rng.permutation(group.index.to_numpy()))
                  for _, group in slide.groupby([slide["x_target"] // block_px,
                                                 slide["y_target"] // block_px], sort=True)]
        queue = []
        while any(blocks):
            for b in rng.permutation([i for i, block in enumerate(blocks) if block]):
                queue.append(blocks[b].pop())
        queues.append(queue)
    chosen, wanted = [], min(count, len(rows))
    while len(chosen) < wanted:
        for q in rng.permutation([i for i, queue in enumerate(queues) if queue]):
            chosen.append(queues[q].pop(0))
            if len(chosen) == wanted:
                break
    if count > len(chosen):
        chosen += list(rng.choice(rows.index.to_numpy(), count - len(chosen), replace=True))
    return tiles.loc[chosen]


def evaluation_batches(tiles, patient_id, batch_size=64):
    """Every tile of one patient exactly once, in a fixed order, as DataFrame chunks."""
    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1: {batch_size}")
    rows = tiles[tiles["patient_id"] == patient_id].sort_values(["slide_id", "y_target", "x_target"])
    for start in range(0, len(rows), batch_size):
        yield rows.iloc[start:start + batch_size]


class TileReader:
    """Reads tiles by their modeling-list row. Keeps up to `max_open` files open (least recently used).

    `base` is the folder the rows came from: a package or combined folder, or the data root on the
    processing machine. Create one per process (for example in each data-loader worker); HDF5 and
    slide handles should not be shared across processes.
    """

    def __init__(self, base, tiling_config=DEFAULT_TILING_CONFIG, verify=True, max_open=16):
        self.base = Path(base)
        if (self.base / "export_settings.json").exists() and not (self.base / PACKAGE_MANIFEST).exists():
            self.base = export_root(self.base)   # an export folder: rows are relative to its data root
        self.cfg = load_tiling_config(tiling_config)[0]
        self.verify = verify
        self.max_open = max_open
        self._open = OrderedDict()

    def _handle(self, path, opener, closer):
        handle = self._open.pop(path, None)
        if handle is None:
            handle = (opener(path), closer)
            while len(self._open) >= self.max_open:
                _, (old, close) = self._open.popitem(last=False)
                close(old)
        self._open[path] = handle
        return handle[0]

    def read(self, row):
        """RGB uint8 array of one tile; raises ValueError if its pixels differ from the recorded hash."""
        base = self.base / str(row.get("part_dir", "") or "")
        if str(row["stored"]) == "True":
            h5 = self._handle(base / row["h5"], lambda p: h5py.File(p, "r"), lambda h: h.close())
            data = h5["tiles"][int(row["h5_index"])].tobytes()
            rgb = np.asarray(Image.open(io.BytesIO(data)).convert("RGB"))
        else:
            import slides   # TIAToolbox is only needed for tiles that were not stored
            wsi = self._handle(base / row["slide_relpath"], lambda p: slides.open_slide(p, self.cfg)[0],
                               slides.close_slide)
            rgb = slides.read_tile(wsi, row["x_target"], row["y_target"], self.cfg)
        if self.verify and hashlib.sha256(rgb.tobytes()).hexdigest() != row["pixel_sha256"]:
            raise ValueError(f"{row['patch_id']}: pixels differ from the recorded hash")
        return rgb

    def read_bag(self, rows):
        """Read several tiles (e.g. a patient's sampled bag) as a list of arrays."""
        return [self.read(row) for _, row in rows.iterrows()]

    def close(self):
        for handle, close in self._open.values():
            close(handle)
        self._open = OrderedDict()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def fit_fold_normalizer(folder, fold, reference_patch_id, reader):
    """Fit Macenko on one reference tile for a validation fold; return (normalizer, record).

    The reference must be a model tile in this folder's modeling list, from a TCGA patient who is a
    training patient for `fold` (not that fold's validation patients, not CPTAC); load_modeling
    checks the folder first. Save `record` with the model run.
    """
    from tiatoolbox.tools.stainnorm import MacenkoNormalizer
    folder = Path(folder)
    tiles = with_roles(load_modeling(folder, allow_partial=True), fold)
    match = tiles[tiles["patch_id"] == reference_patch_id]
    if len(match) != 1 or match.iloc[0]["role"] != "train" or match.iloc[0]["source"] != "TCGA":
        raise ValueError(f"{reference_patch_id} is not a TCGA training tile for fold {fold}")
    reference = match.iloc[0]
    normalizer = MacenkoNormalizer()
    normalizer.fit(reader.read(reference).copy())   # Macenko may modify its input array
    record = {"fold": fold, "reference_patch_id": reference_patch_id,
              "reference_patient": reference["patient_id"], "folder": str(folder),
              "modeling_sha256": sha256_file(folder / "modeling_patches.csv"),
              "stain_matrix_target": np.asarray(normalizer.stain_matrix_target).tolist(),
              "maxC_target": np.asarray(normalizer.maxC_target).tolist(),
              "fitted_at_utc": datetime.now(timezone.utc).isoformat()}
    return normalizer, record


def normalize_tile(normalizer, rgb, patch_id, failure_log):
    """Macenko-normalize one tile. On failure, append the patch ID and error to `failure_log` and raise."""
    try:
        return normalizer.transform(rgb.copy())
    except Exception as error:
        path = Path(failure_log)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", newline="") as handle:
            writer = csv.writer(handle)
            if handle.tell() == 0:
                writer.writerow(["patch_id", "error", "time_utc"])
            writer.writerow([patch_id, f"{type(error).__name__}: {error}",
                             datetime.now(timezone.utc).isoformat()])
        raise
