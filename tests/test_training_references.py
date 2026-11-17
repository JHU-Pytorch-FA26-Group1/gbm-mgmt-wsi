"""Regression tests for training/references.py."""
from contextlib import nullcontext
from types import SimpleNamespace

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
