"""Patient bags for training and evaluation: roles, pool limits, tile processing and the dataset.

A pool is a patient's packaged tiles after any per-patient limit; a bag is the tiles drawn from the
pool for one patient in one epoch. A tile goes through: read and hash-check, optional JPEG round
trip, Macenko, augmentation (training only), then tensor normalization.
"""
import hashlib
import io
import os

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, Sampler

import loader

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _rng(*parts):
    """A generator seeded from the parts, so a draw depends only on them."""
    key = hashlib.sha256("|".join(str(part) for part in parts).encode()).digest()
    return np.random.default_rng(int.from_bytes(key[:8], "big"))


def split_patients(tiles, fold, smoke=False):
    """Return (tiles with `role` set, {"train", "validation", "external"} patient lists).

    `fold` is 0 to 4, or "final" (every TCGA patient trains). With `smoke`, the validation patients
    also train, so the pipeline can be shown to run; their results are not valid.
    """
    if fold == "final":
        tiles = tiles.assign(role=np.where(tiles["source"] == "CPTAC", "external", "train"))
    else:
        tiles = loader.with_roles(tiles, fold)
    groups = {role: sorted(tiles.loc[tiles["role"] == role, "patient_id"].unique())
              for role in ("train", "validation", "external")}
    if smoke:
        groups["train"] = groups["validation"]
        tiles = tiles.assign(role=np.where(tiles["role"] == "validation", "train", tiles["role"]))
    return tiles, groups


def limit_pool(tiles, limit, seed):
    """Keep at most `limit` tiles per patient; a smaller limit gives a subset of a larger one."""
    if limit is None:
        return tiles
    keep = []
    for patient, rows in tiles.groupby("patient_id", sort=True):
        order = _rng(seed, "pool", patient).permutation(len(rows))[:limit]
        keep.extend(rows.index[order])
    return tiles.loc[sorted(keep)]


def tiles_per_patient(tiles):
    """{patient_id: number of tiles}."""
    return tiles.groupby("patient_id").size().to_dict()


def jpeg_round_trip(rgb, quality):
    """The tile as it would look stored as JPEG at `quality`; unchanged when quality is None."""
    if quality is None:
        return rgb
    buffer = io.BytesIO()
    Image.fromarray(rgb).save(buffer, format="JPEG", quality=quality)
    return np.asarray(Image.open(io.BytesIO(buffer.getvalue())).convert("RGB"))


def augment(rgb, rng):
    """A random flip and quarter turn: tissue has no orientation, and no pixel value changes."""
    if rng.random() < 0.5:
        rgb = rgb[:, ::-1]
    if rng.random() < 0.5:
        rgb = rgb[::-1]
    return np.ascontiguousarray(np.rot90(rgb, int(rng.integers(4))))


def to_tensor(rgb):
    """Channels-first float tensor, scaled to 0-1 and normalized with the ImageNet statistics."""
    scaled = (rgb.astype(np.float32) / 255 - IMAGENET_MEAN) / IMAGENET_STD
    return torch.from_numpy(np.ascontiguousarray(scaled.transpose(2, 0, 1)))


class TilePipeline:
    """Reads and processes tiles for one process; the reader is rebuilt after a fork."""

    def __init__(self, package, settings, normalizer=None, failure_log=None):
        if normalizer is not None and failure_log is None:
            raise ValueError("a normalizer needs a failure_log, so failed tiles are recorded")
        self.package = package
        self.jpeg_quality = settings["jpeg_quality"]
        self.seed = settings["seed"]
        self.normalizer = normalizer
        self.failure_log = failure_log
        self._reader = None
        self._pid = None

    def __getstate__(self):
        """Pickle without the reader: HDF5 handles cannot be pickled, and a worker opens its own."""
        return {**self.__dict__, "_reader": None, "_pid": None}

    @property
    def reader(self):
        """A TileReader owned by the current process (HDF5 handles are not shared across forks)."""
        if self._reader is None or self._pid != os.getpid():
            self._reader = loader.TileReader(self.package)
            self._pid = os.getpid()
        return self._reader

    def tile(self, row, rng=None):
        """One processed tile tensor; `rng` turns augmentation on."""
        rgb = jpeg_round_trip(self.reader.read(row), self.jpeg_quality)
        if self.normalizer is not None:
            rgb = loader.normalize_tile(self.normalizer, rgb, row["patch_id"], self.failure_log)
        if rng is not None:
            rgb = augment(rgb, rng)
        return to_tensor(rgb)


class EpochSampler(Sampler):
    """Yields (epoch, patient index) in a seeded order that changes every epoch.

    The epoch travels with the index, so workers that outlive an epoch still draw its bags.
    """

    def __init__(self, count, seed):
        super().__init__()
        self.count = count
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch):
        """Choose the epoch the next pass belongs to."""
        self.epoch = epoch

    def __len__(self):
        return self.count

    def __iter__(self):
        order = _rng(self.seed, "order", self.epoch).permutation(self.count)
        return iter([(self.epoch, int(index)) for index in order])


class BagDataset(Dataset):
    """One patient's bag per item, asked for as (epoch, index): (tensor [tiles, 3, H, W], label)."""

    def __init__(self, tiles, patients, labels, settings, pipeline):
        pool = tiles[tiles["patient_id"].isin(patients)]
        self.tiles = limit_pool(pool, settings["max_tiles_per_patient"], settings["seed"])
        self.patients = list(patients)
        self.labels = labels
        self.seed = settings["seed"]
        self.count = settings["tiles_per_bag"]
        self.pipeline = pipeline

    def __len__(self):
        return len(self.patients)

    def bag_rows(self, index, epoch):
        """The pool rows drawn for patient `index` in `epoch`."""
        return loader.training_sample(self.tiles, self.patients[index], epoch,
                                      seed=self.seed, count=self.count)

    def __getitem__(self, key):
        epoch, index = key
        rows = self.bag_rows(index, epoch)
        tensors = [self.pipeline.tile(row, _rng(self.seed, "augment", epoch, row["patch_id"]))
                   for _, row in rows.iterrows()]
        return torch.stack(tensors), torch.tensor(float(self.labels[self.patients[index]]))


def evaluation_chunks(tiles, patient, settings, pipeline, batch_size=64):
    """Yield the patient's whole pool, once and in a fixed order, as tensors [n, 3, H, W]."""
    pool = limit_pool(tiles[tiles["patient_id"] == patient], settings["max_eval_tiles_per_patient"],
                      settings["seed"])
    for chunk in loader.evaluation_batches(pool, patient, batch_size):
        yield torch.stack([pipeline.tile(row) for _, row in chunk.iterrows()])
