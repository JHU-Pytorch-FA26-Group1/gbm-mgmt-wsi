"""Regression tests for training/evaluate.py."""
import pytest
import torch
from torch import nn

from tests.test_training_data import make_package, settings_with, SIDE
from training import data, evaluate, models


def tiny_model(settings):
    """A BagModel on 8x8 tiles that is cheap and deterministic."""
    torch.manual_seed(0)
    backbone = nn.Sequential(nn.Flatten(), nn.Linear(3 * SIDE * SIDE, 4))
    return models.BagModel(backbone, 4, settings)


@pytest.fixture(name="scored")
def fixture_scored(tmp_path):
    """(tiles, settings, pipeline, model) over the small package."""
    tiles = make_package(tmp_path)
    settings = settings_with()
    return tiles, settings, data.TilePipeline(tmp_path, settings), tiny_model(settings)


def test_a_patient_is_scored_from_every_tile_whatever_the_chunk_size(scored):
    """Chunking only batches the encoding: the pooled score must match one pass over the pool."""
    tiles, settings, pipeline, model = scored
    pool = list(data.evaluation_chunks(tiles, "T1", settings, pipeline, batch_size=64))
    assert len(pool) == 1 and len(pool[0]) == 10
    model.eval()
    with torch.no_grad():
        expected, _ = model.pool(model.encode(pool[0])[None])
    for batch_size in (1, 3, 64):
        rows = evaluate.predict(model, tiles, ["T1"], settings, pipeline, batch_size=batch_size)
        _, logit, probability = rows[0]
        assert logit == pytest.approx(expected.item(), abs=1e-5)
        assert probability == pytest.approx(torch.sigmoid(expected).item(), abs=1e-5)


def test_scoring_runs_in_eval_mode_without_gradients_and_leaves_eval_mode(scored):
    """Scoring tracks no gradients and uses no training-mode layers; the model stays in eval."""
    tiles, settings, pipeline, model = scored
    seen = []
    forward = model.backbone.forward

    def spy(batch):
        seen.append((torch.is_grad_enabled(), model.training))
        return forward(batch)

    model.backbone.forward = spy
    model.train()
    evaluate.predict(model, tiles, ["T1"], settings, pipeline)
    assert seen and all(state == (False, False) for state in seen)
    assert not model.training


def test_scores_follow_the_requested_patients_and_ignore_the_others(scored):
    """Each patient is scored on their own tiles, in the order asked."""
    tiles, settings, pipeline, model = scored
    both = evaluate.predict(model, tiles, ["T3", "T1"], settings, pipeline)
    alone = evaluate.predict(model, tiles, ["T1"], settings, pipeline)
    assert [row[0] for row in both] == ["T3", "T1"]
    assert both[1][1] == pytest.approx(alone[0][1], abs=1e-5)


def test_a_patient_without_tiles_is_an_error_not_a_score(scored):
    """An empty pool must stop the run rather than score an empty bag."""
    tiles, settings, pipeline, model = scored
    with pytest.raises(RuntimeError):
        evaluate.predict(model, tiles, ["nobody"], settings, pipeline)
