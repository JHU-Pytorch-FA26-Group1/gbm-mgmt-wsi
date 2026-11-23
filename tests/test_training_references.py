"""Regression tests for training/references.py."""
import json
from contextlib import nullcontext
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from training import experiment, references, run


READER = SimpleNamespace(read=lambda row: row["patch_id"])  # the patch id stands in for pixels


def make_tiles():
    """Four TCGA patients (one per fold 0 to 3, three tiles each) and one CPTAC patient."""
    rows = [{"source": "TCGA", "patient_id": f"T{fold}", "fold": str(fold),
             "patch_id": f"T{fold}_{index}"} for fold in range(4) for index in range(3)]
    rows += [{"source": "CPTAC", "patient_id": "C0", "fold": "external", "patch_id": f"C0_{index}"}
             for index in range(3)]
    return pd.DataFrame(rows)


def accept_all(_):
    """A fit that always works."""


def test_each_reference_comes_from_that_folds_tcga_training_patients():
    """Never the fold's own validation patients, never CPTAC."""
    chosen = references.choose_references(make_tiles(), READER, 0, folds=range(4), fit=accept_all)
    for fold, patch_id in chosen.items():
        assert patch_id.startswith("T") and patch_id[1] != fold


def test_the_draw_repeats_for_a_seed_and_can_change_with_it():
    """A reference must be reproducible from the config seed."""
    first = references.choose_references(make_tiles(), READER, 0, folds=range(4), fit=accept_all)
    again = references.choose_references(make_tiles(), READER, 0, folds=range(4), fit=accept_all)
    others = {tuple(references.choose_references(make_tiles(), READER, seed, folds=range(4),
                                                 fit=accept_all).values()) for seed in range(8)}
    assert first == again and len(others) > 1


def test_a_tile_that_cannot_be_fitted_is_skipped_and_none_fittable_is_an_error():
    """Macenko fails on tiles with too little stain; the next drawn tile is tried."""
    chosen = references.choose_references(make_tiles(), READER, 0, folds=[3],
                                          fit=lambda patch_id: patch_id.endswith("_1") or 1 / 0)
    assert chosen["3"].endswith("_1")

    def refuse(_):
        raise ValueError("no stain")

    with pytest.raises(RuntimeError, match="fold 3"):
        references.choose_references(make_tiles(), READER, 0, folds=[3], fit=refuse)


def test_the_references_command_prints_one_config_line_per_fold(monkeypatch, capsys):
    """The output is pasted into train.yaml: the key line, then one line per fold."""
    monkeypatch.setattr(experiment, "read_tiles", lambda settings: None)
    monkeypatch.setattr(run.loader, "TileReader", lambda package: nullcontext())
    monkeypatch.setattr(references, "choose_references",
                        lambda *args: {"0": "TCGA_a", "1": "TCGA_b"})
    assert run.main(["references"]) == 0
    assert capsys.readouterr().out.splitlines() == [
        "macenko_reference_patch_ids:", '  "0": TCGA_a', '  "1": TCGA_b']


H_TRUE = np.array([0.55, 0.75, 0.37]) / np.linalg.norm([0.55, 0.75, 0.37])
E_TRUE = np.array([0.12, 0.83, 0.55]) / np.linalg.norm([0.12, 0.83, 0.55])


def stained_tile(seed, hematoxylin=H_TRUE, eosin=E_TRUE, size=48):
    """A tile of pure H and E stain at random concentrations, with a white margin."""
    rng = np.random.default_rng(seed)
    conc = rng.uniform(0.2, 1.6, (size, size, 2))
    od = conc[..., :1] * hematoxylin + conc[..., 1:] * eosin
    rgb = (255 * np.exp(-od)).clip(0, 255).astype(np.uint8)
    rgb[:6] = 255
    return rgb


def stained_reader():
    """Reads a tile from its patch id (the number after the last underscore seeds it)."""
    return SimpleNamespace(read=lambda row: stained_tile(int(row["patch_id"].rsplit("_", 1)[1])))


def stained_tiles(counts, source="TCGA"):
    """One row per tile for {patient: count}; the id's number is globally unique."""
    rows = [{"source": source, "patient_id": patient, "fold": "0",
             "patch_id": f"{patient}_{number}"}
            for patient, count in counts.items() for number in range(count)]
    frame = pd.DataFrame(rows)
    frame["patch_id"] = [f"{p}_{i}" for i, p in enumerate(frame["patient_id"])]
    return frame


def test_the_cohort_reference_recovers_the_stains_the_tiles_were_made_from():
    """Tiles made from known H and E vectors must give them back, one stain per column."""
    he, max_c, _ = references.build_cohort_reference(
        stained_tiles({"P1": 6, "P2": 6}), stained_reader(), seed=0, per_patient=6)
    assert he.shape == (3, 2) and max_c.shape == (2,)
    assert abs(he[:, 0] @ H_TRUE) > 0.98 and abs(he[:, 1] @ E_TRUE) > 0.98
    assert max_c[0] == pytest.approx(1.6, rel=0.2) and max_c[1] == pytest.approx(1.6, rel=0.2)


def test_every_patient_weighs_the_same_whatever_their_tile_count():
    """A patient with 300 tiles and one with 5 each give 5 tiles when per_patient is 5."""
    _, _, facts = references.build_cohort_reference(
        stained_tiles({"BIG": 300, "SMALL": 5}), stained_reader(), seed=0, per_patient=5)
    assert facts["tiles_used"] == 10 and facts["patients"] == 2


def test_only_the_asked_cohorts_tiles_are_read():
    """A TCGA reference must not look at CPTAC pixels, which are the held-out test set."""
    tiles = pd.concat([stained_tiles({"T1": 4}), stained_tiles({"C1": 4}, source="CPTAC")],
                      ignore_index=True)
    tiles["patch_id"] = [f"{p}_{i}" for i, p in enumerate(tiles["patient_id"])]
    seen = []

    def read(row):
        seen.append(row["patient_id"])
        return stained_tile(int(row["patch_id"].rsplit("_", 1)[1]))

    references.build_cohort_reference(tiles, SimpleNamespace(read=read), seed=0, per_patient=4)
    assert set(seen) == {"T1"}
    seen.clear()
    references.build_cohort_reference(tiles, SimpleNamespace(read=read), seed=0, per_patient=4,
                                      sources=("TCGA", "CPTAC"))
    assert set(seen) == {"T1", "C1"}


def test_the_same_seed_gives_the_same_reference_and_another_seed_draws_other_tiles():
    """The reference is a file the team shares, so it must be reproducible from its seed."""
    tiles = stained_tiles({"P1": 20, "P2": 20})

    def build(seed):
        read = []

        def record(row):
            read.append(row["patch_id"])
            return stained_tile(int(row["patch_id"].rsplit("_", 1)[1]))

        he, max_c, _ = references.build_cohort_reference(
            tiles, SimpleNamespace(read=record), seed=seed, per_patient=6)
        return sorted(read), he, max_c

    first, again, other = build(3), build(3), build(4)
    assert first[0] == again[0] and np.array_equal(first[1], again[1])
    assert np.array_equal(first[2], again[2])
    assert first[0] != other[0]


def test_unfittable_tiles_are_counted_and_skipped_and_none_fittable_is_an_error():
    """A blank tile has no stain to estimate; the others still define the reference."""
    reader = SimpleNamespace(read=lambda row: np.full((48, 48, 3), 255, np.uint8)
                             if row["patch_id"].endswith("_0") else stained_tile(7))
    tiles = stained_tiles({"P1": 6})
    tiles["patch_id"] = [f"P1_{i}" for i in range(6)]
    _, _, facts = references.build_cohort_reference(tiles, reader, seed=0, per_patient=6)
    assert facts["tiles_failed"] == 1 and facts["tiles_used"] == 5
    blank = SimpleNamespace(read=lambda row: np.full((48, 48, 3), 255, np.uint8))
    with pytest.raises(RuntimeError, match="could be fitted"):
        references.build_cohort_reference(tiles, blank, seed=0, per_patient=6)


def test_the_saved_files_hold_the_arrays_and_the_hash_of_the_npz(tmp_path):
    """The JSON beside the npz names the exact file it describes."""
    he, max_c = np.eye(3, 2, dtype=np.float32), np.array([2.0, 1.0], np.float32)
    references.save_cohort_reference(tmp_path / "ref.npz", he, max_c, {"patients": 3})
    arrays = np.load(tmp_path / "ref.npz", allow_pickle=False)
    assert np.array_equal(arrays["ref_HE"], he) and np.array_equal(arrays["ref_max_C"], max_c)
    described = json.loads((tmp_path / "ref.json").read_text(encoding="utf-8"))
    assert described["patients"] == 3 and described["ref_max_C"] == [2.0, 1.0]
    assert described["npz_sha256"] == references.sha256_file(tmp_path / "ref.npz")


COMMAND = ["cohort-reference", "--package", "p", "--out", "r.npz", "--sources", "TCGA", "CPTAC",
           "--per-patient", "7", "--seed", "2"]


def fake_command_environment(monkeypatch, seen):
    """Stand in for the package, the reader and the file hash; record what the builder is given."""
    monkeypatch.setattr(run.loader, "load_modeling", lambda package: "tiles")
    monkeypatch.setattr(run.loader, "TileReader", lambda package: nullcontext())
    monkeypatch.setattr(run.loader, "sha256_file", lambda path: "h")

    def build(tiles, _reader, seed, per_patient, sources):
        seen.update(tiles=tiles, seed=seed, per_patient=per_patient, sources=sources)
        return "he", "max_c", {"tiles_used": 1, "patients": 1}

    monkeypatch.setattr(references, "build_cohort_reference", build)
    monkeypatch.setattr(references, "save_cohort_reference", lambda *args: seen.update(saved=args))


def test_the_cohort_reference_command_passes_its_options_to_the_builder(monkeypatch):
    """Cohorts, tiles per patient and seed reach the builder, and the package hash is recorded."""
    seen = {}
    fake_command_environment(monkeypatch, seen)
    assert run.main(COMMAND) == 0
    assert seen["per_patient"] == 7 and seen["seed"] == 2 and seen["sources"] == ("TCGA", "CPTAC")
    assert seen["saved"][3]["modeling_sha256"] == "h"


def test_a_cohort_reference_that_cannot_be_built_is_a_setup_problem(monkeypatch):
    """No fittable tile must exit 2, not print a traceback."""
    fake_command_environment(monkeypatch, {})

    def refuse(*unused):
        raise RuntimeError("no tile")

    monkeypatch.setattr(references, "build_cohort_reference", refuse)
    assert run.main(COMMAND) == 2
