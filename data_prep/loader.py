"""Read exported tiles for training and evaluation, and fit Macenko per fold. Not used by tiling.

    import loader
    tiles = loader.with_roles(loader.load_modeling(EXPORT_DIR), fold=0)   # train / validation / external
    reader = loader.TileReader(ROOT)          # one per data-loader worker process
    rgb = reader.read(tiles.iloc[0])          # stored tiles from the .h5, others from the slide

Every read checks the tile's pixel hash against the one recorded at tiling, so a tile is exactly
the pixels that were scored and QC'd. Tiles beyond the tiling cap (stored = False) need the slides
under <root>/raw/; stored tiles need only the .h5 files.

Macenko is a training-time transform: fit it once per fold on a reference tile from that fold's
training patients, then normalize each tile before augmentation. A failed transform is logged and
raised; a raw tile is never passed on as if it had been normalized.
"""
import csv
import hashlib
import io
import json
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from PIL import Image

from common import DEFAULT_TILING_CONFIG, load_tiling_config, sha256_file


def load_modeling(export_dir):
    """The modeling list of one export, as a DataFrame (one row per accepted tile)."""
    return pd.read_csv(Path(export_dir) / "modeling_patches.csv", dtype={"fold": str, "pixel_sha256": str})


def with_roles(tiles, fold):
    """Add `role` for a validation fold: external (CPTAC), validation (this fold) or train."""
    if fold not in range(5):
        raise ValueError(f"fold must be 0-4: {fold}")
    role = np.where(tiles["source"] == "CPTAC", "external",
                    np.where(tiles["fold"] == str(fold), "validation", "train"))
    return tiles.assign(role=role)


class TileReader:
    """Reads tiles by their modeling-list row. Opens files lazily and keeps them open.

    Create one per process (for example in each data-loader worker); HDF5 and slide handles should
    not be shared across processes.
    """

    def __init__(self, root, tiling_config=DEFAULT_TILING_CONFIG, verify=True):
        self.root = Path(root)
        self.cfg = load_tiling_config(tiling_config)[0]
        self.verify = verify
        self._h5, self._slides = {}, {}

    def read(self, row):
        """RGB uint8 array of one tile; raises ValueError if its pixels differ from the recorded hash."""
        if str(row["stored"]) == "True":
            h5 = self._h5.get(row["h5"])
            if h5 is None:
                h5 = self._h5[row["h5"]] = h5py.File(self.root / row["h5"], "r")
            data = h5["tiles"][int(row["h5_index"])].tobytes()
            rgb = np.asarray(Image.open(io.BytesIO(data)).convert("RGB"))
        else:
            import slides   # TIAToolbox is only needed for tiles that were not stored
            wsi = self._slides.get(row["slide_relpath"])
            if wsi is None:
                wsi = slides.open_slide(self.root / row["slide_relpath"], self.cfg)[0]
                self._slides[row["slide_relpath"]] = wsi
            rgb = slides.read_tile(wsi, row["x_target"], row["y_target"], self.cfg)
        if self.verify and hashlib.sha256(rgb.tobytes()).hexdigest() != row["pixel_sha256"]:
            raise ValueError(f"{row['patch_id']}: pixels differ from the recorded hash")
        return rgb

    def read_bag(self, rows):
        """Read several tiles (e.g. a patient's sampled bag) as a list of arrays."""
        return [self.read(row) for _, row in rows.iterrows()]

    def close(self):
        for h5 in self._h5.values():
            h5.close()
        if self._slides:
            import slides
            for wsi in self._slides.values():
                slides.close_slide(wsi)
        self._h5, self._slides = {}, {}

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def fit_fold_normalizer(export_dir, fold, reference_patch_id, reader):
    """Fit Macenko on one reference tile for a validation fold; return (normalizer, record).

    The reference must be an accepted tile in this export's modeling list, from a TCGA patient who
    is a training patient for `fold` (not that fold's validation patients, not CPTAC), and the
    modeling list must be the one the export recorded. Save `record` with the model run.
    """
    from tiatoolbox.tools.stainnorm import MacenkoNormalizer
    export_dir = Path(export_dir)
    settings = json.loads((export_dir / "export_settings.json").read_text())
    if sha256_file(export_dir / "modeling_patches.csv") != settings["modeling_sha256"]:
        raise ValueError("modeling_patches.csv changed since the export")
    tiles = with_roles(load_modeling(export_dir), fold)
    match = tiles[tiles["patch_id"] == reference_patch_id]
    if len(match) != 1 or match.iloc[0]["role"] != "train" or match.iloc[0]["source"] != "TCGA":
        raise ValueError(f"{reference_patch_id} is not an accepted TCGA training tile for fold {fold}")
    reference = match.iloc[0]
    normalizer = MacenkoNormalizer()
    normalizer.fit(reader.read(reference).copy())   # Macenko may modify its input array
    record = {"fold": fold, "reference_patch_id": reference_patch_id,
              "reference_patient": reference["patient_id"], "export": str(export_dir),
              "modeling_sha256": settings["modeling_sha256"],
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
