"""Regression tests for training/data.py: roles, pool limits, tile processing and bags."""
import hashlib
import io
import os
import pickle
from types import SimpleNamespace

import h5py
import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image

from training import config, data

SIDE = 8
PATIENTS = {"T1": ("TCGA", "0", 10), "T2": ("TCGA", "1", 2), "T3": ("TCGA", "2", 6),
            "C1": ("CPTAC", "external", 5)}


def make_package(folder, patients=None):
    """Write one HDF5 of PNG tiles per patient; return the modeling table that points at them."""
    rng = np.random.default_rng(0)
    rows = []
    for patient, (source, fold, count) in (patients or PATIENTS).items():
        path = folder / "tiles" / source / patient / "slide.h5"
        path.parent.mkdir(parents=True)
        with h5py.File(path, "w") as handle:
            stored = handle.create_dataset("tiles", shape=(count,), dtype=h5py.vlen_dtype(np.uint8))
            for index in range(count):
                rgb = rng.integers(0, 256, (SIDE, SIDE, 3), dtype=np.uint8)
                buffer = io.BytesIO()
                Image.fromarray(rgb).save(buffer, format="PNG")
                stored[index] = np.frombuffer(buffer.getvalue(), dtype=np.uint8)
                rows.append({
                    "source": source, "patient_id": patient, "slide_id": "slide", "fold": fold,
                    "patch_id": f"{patient}_{index}", "role": "tissue", "x_target": 224 * index,
                    "y_target": 0, "stored": "True", "h5": str(path.relative_to(folder)),
                    "h5_index": index, "pixel_sha256": hashlib.sha256(rgb.tobytes()).hexdigest(),
                    "slide_relpath": "", "part_dir": ""})
    return pd.DataFrame(rows)


def settings_with(*overrides):
    """Config settings for the tests, with a small bag."""
    return config.load_config(backbone="swin_v2_t", overrides=["tiles_per_bag=4", *overrides])[0]


@pytest.fixture(name="package")
def fixture_package(tmp_path):
    """(package folder, modeling table) for the small package."""
    return tmp_path, make_package(tmp_path)


def test_tensor_values_follow_the_imagenet_statistics():
    """One pixel with known channels pins the mean, the std and the order they are applied in."""
    tensor = data.to_tensor(np.array([[[255, 0, 128]]], dtype=np.uint8))
    expected = [(1.0 - 0.485) / 0.229, (0.0 - 0.456) / 0.224, (128 / 255 - 0.406) / 0.225]
    assert tensor.flatten().tolist() == pytest.approx(expected, rel=1e-5)


def test_jpeg_quality_reaches_the_encoder():
    """A lower quality must change the pixels more than a higher one."""
    rgb = np.random.default_rng(3).integers(0, 256, (16, 16, 3), dtype=np.uint8)
    error = {quality: np.abs(data.jpeg_round_trip(rgb, quality).astype(int) - rgb).mean()
             for quality in (10, 95)}
    assert error[10] > error[95] > 0


def test_a_training_tile_is_augmented_and_an_evaluation_tile_is_not(package):
    """Augmentation is for training only: evaluation must see the stored tile as it is."""
    folder, tiles = package
    settings = settings_with()
    pipeline = data.TilePipeline(folder, settings)
    row = tiles[tiles["patient_id"] == "T1"].iloc[0]
    plain = pipeline.tile(row)
    assert any(not torch.equal(pipeline.tile(row, np.random.default_rng(seed)), plain)
               for seed in range(8))
    _, evaluated = data.EvalChunks(tiles, ["T1"], settings, pipeline, batch_size=1)[0]
    first = tiles[tiles["patient_id"] == "T1"].sort_values(["slide_id", "y_target", "x_target"])
    assert torch.equal(evaluated[0], pipeline.tile(first.iloc[0]))


def test_roles_for_a_fold_and_for_the_final_fit(package):
    """Validation is the fold's patients; the final fit trains on every TCGA patient."""
    _, tiles = package
    _, groups = data.split_patients(tiles, 1)
    assert groups == {"train": ["T1", "T3"], "validation": ["T2"], "external": ["C1"]}
    _, groups = data.split_patients(tiles, "final")
    assert groups == {"train": ["T1", "T2", "T3"], "validation": [], "external": ["C1"]}


def test_smoke_run_trains_on_the_validation_patients_and_still_validates_on_them(package):
    """Smoke mode must not empty validation or let training sample a non-train patient."""
    _, tiles = package
    marked, groups = data.split_patients(tiles, 1, smoke=True)
    assert groups["train"] == groups["validation"] == ["T2"]
    assert set(marked.loc[marked["patient_id"] == "T2", "role"]) == {"train"}


def test_pool_limit_keeps_nested_subsets_and_small_patients_whole(package):
    """A smaller pool is inside a larger one, so experiments on pool size compare cleanly."""
    _, tiles = package
    assert data.tiles_per_patient(data.limit_pool(tiles, 3, seed=0)) == {
        "T1": 3, "T2": 2, "T3": 3, "C1": 3}
    for seed in range(8):
        pools = [set(data.limit_pool(tiles, limit, seed).index) for limit in (2, 4, 6)]
        assert pools[0] <= pools[1] <= pools[2]
    other_seed = data.limit_pool(tiles, 3, seed=1)
    assert set(other_seed.index) != set(data.limit_pool(tiles, 3, seed=0).index)
    assert data.limit_pool(tiles, None, seed=0) is tiles


def test_jpeg_changes_pixels_only_when_a_quality_is_set():
    """No quality means the packaged tile is used as it is."""
    rgb = np.random.default_rng(1).integers(0, 256, (16, 16, 3), dtype=np.uint8)
    assert data.jpeg_round_trip(rgb, None) is rgb
    lossy = data.jpeg_round_trip(rgb, 50)
    assert lossy.shape == rgb.shape and lossy.dtype == np.uint8
    assert not np.array_equal(lossy, rgb)


def test_augmentation_only_rearranges_pixels():
    """Every output is one of the eight flip and rotation variants, so no pixel value changes."""
    rgb = np.random.default_rng(2).integers(0, 256, (6, 6, 3), dtype=np.uint8)
    variants = [np.rot90(view, k) for view in (rgb, rgb[:, ::-1]) for k in range(4)]
    seen = set()
    for seed in range(60):
        out = data.augment(rgb, np.random.default_rng(seed))
        matches = [i for i, variant in enumerate(variants) if np.array_equal(out, variant)]
        assert matches
        seen.add(matches[0])
    assert len(seen) > 1


def test_tensor_is_channels_first_and_keeps_rgb_order():
    """Red-only pixels must land in channel 0, or ImageNet statistics hit the wrong colors."""
    rgb = np.zeros((4, 5, 3), dtype=np.uint8)
    rgb[..., 0] = 255
    tensor = data.to_tensor(rgb)
    assert tensor.shape == (3, 4, 5)
    assert tensor[0].min() > 1 > tensor[1].max()


def test_a_bag_has_the_configured_size_and_label_and_repeats_within_an_epoch(package):
    """The same epoch gives the same bag and a new epoch a new one."""
    folder, tiles = package
    settings = settings_with()
    marked, groups = data.split_patients(tiles, 1)
    pipeline = data.TilePipeline(folder, settings)
    dataset = data.BagDataset(marked, groups["train"], {"T1": 1, "T3": 0}, settings, pipeline)
    bag, label = dataset[0, 0]
    assert bag.shape == (4, 3, SIDE, SIDE) and label.item() == 1
    assert dataset[0, 1][0].shape[0] == 4
    assert torch.equal(bag, dataset[0, 0][0])
    assert list(dataset.bag_rows(0, 1)["patch_id"]) != list(dataset.bag_rows(0, 0)["patch_id"])


def test_the_sampler_covers_every_patient_once_per_epoch_in_a_new_order_each_epoch():
    """Each pass visits each patient exactly once, tagged with its epoch."""
    sampler = data.EpochSampler(6, seed=0)
    assert len(sampler) == 6
    passes = []
    for epoch in range(4):
        sampler.set_epoch(epoch)
        keys = list(sampler)
        assert sorted(index for _, index in keys) == list(range(6))
        assert {tagged for tagged, _ in keys} == {epoch}
        passes.append([index for _, index in keys])
    assert len({tuple(order) for order in passes}) > 1
    sampler.set_epoch(2)
    assert [index for _, index in sampler] == passes[2]


def test_workers_that_outlive_an_epoch_draw_that_epochs_bags(package):
    """A persistent worker holds a copy of the dataset, so the epoch must come with the index."""
    folder, tiles = package
    settings = settings_with()
    marked, groups = data.split_patients(tiles, 1)
    dataset = data.BagDataset(marked, groups["train"], {"T1": 1, "T3": 0}, settings,
                              data.TilePipeline(folder, settings))
    sampler = data.EpochSampler(len(dataset), settings["seed"])
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, sampler=sampler, num_workers=1,
                                         persistent_workers=True,
                                         multiprocessing_context="spawn")
    seen = []
    for epoch in range(2):
        sampler.set_epoch(epoch)
        seen.append({float(bag.sum()) for bag, _ in loader})
    assert seen[0] != seen[1]


def test_a_bag_draws_only_from_the_limited_pool(package):
    """With a pool of two tiles, every tile in every bag comes from those two."""
    folder, tiles = package
    settings = settings_with("max_tiles_per_patient=2")
    marked, _ = data.split_patients(tiles, 1)
    pipeline = data.TilePipeline(folder, settings)
    dataset = data.BagDataset(marked, ["T1"], {"T1": 0}, settings, pipeline)
    pool = set(data.limit_pool(tiles[tiles["patient_id"] == "T1"], 2, 0)["patch_id"])
    for epoch in range(5):
        assert set(dataset.bag_rows(0, epoch)["patch_id"]) <= pool


def test_evaluation_visits_every_pool_tile_once_in_a_fixed_order(package):
    """Evaluation covers the whole evaluation pool in a fixed order, unlike a training bag."""
    folder, tiles = package
    settings = settings_with("max_eval_tiles_per_patient=7")
    pipeline = data.TilePipeline(folder, settings)
    first = data.EvalChunks(tiles, ["T1"], settings, pipeline, batch_size=3)
    second = data.EvalChunks(tiles, ["T1"], settings, pipeline, batch_size=3)
    assert [len(first[i][1]) for i in range(len(first))] == [3, 3, 1]
    assert all(torch.equal(first[i][1], second[i][1]) for i in range(len(first)))


def test_a_forked_process_gets_its_own_reader(package, monkeypatch):
    """HDF5 handles must not be shared across processes, so a new pid means a new reader."""
    folder, _ = package
    pipeline = data.TilePipeline(folder, settings_with())
    parent = pipeline.reader
    assert pipeline.reader is parent
    monkeypatch.setattr(os, "getpid", lambda: -1)
    assert pipeline.reader is not parent


def test_a_failed_normalization_stops_the_bag_and_is_logged(package, tmp_path):
    """A raw tile must never be passed on as if it had been normalized."""
    folder, tiles = package

    def fail(_):
        raise ValueError("no stain found")

    failing = SimpleNamespace(transform=fail)
    log = tmp_path / "failures.csv"
    pipeline = data.TilePipeline(folder, settings_with(), normalizer=failing, failure_log=log)
    with pytest.raises(ValueError, match="no stain found"):
        pipeline.tile(tiles.iloc[0])
    assert "T1_0" in log.read_text()
    with pytest.raises(ValueError, match="failure_log"):
        data.TilePipeline(folder, settings_with(), normalizer=failing)


def test_a_pipeline_that_has_read_tiles_can_be_sent_to_spawned_workers(package):
    """Workers started with spawn receive a pickle, which fails if open HDF5 handles come along."""
    folder, tiles = package
    pipeline = data.TilePipeline(folder, settings_with())
    row = tiles.iloc[0]
    expected = pipeline.tile(row)
    clone = pickle.loads(pickle.dumps(pipeline))
    assert clone.reader is not pipeline.reader
    assert torch.equal(clone.tile(row), expected)
