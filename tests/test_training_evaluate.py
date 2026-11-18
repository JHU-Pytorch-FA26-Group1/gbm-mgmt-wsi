"""Regression tests for training/evaluate.py."""
import pytest
import torch
from torch import nn

from tests.test_training_data import make_package, settings_with, SIDE
from training import data, evaluate, models


class FailingNormalizer:  # pylint: disable=too-few-public-methods
    """Always fails; defined here so spawned workers can import it."""

    def transform(self, rgb):
        """Refuse every tile, as a failed Macenko fit would."""
        raise ValueError(f"no stain found in a {rgb.shape} tile")


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
    pool = [data.EvalChunks(tiles, ["T1"], settings, pipeline, batch_size=64)[0][1]]
    assert len(pool[0]) == 10
    model.eval()
    with torch.no_grad():
        expected, _ = model.pool(model.encode(pool[0])[None])
    for batch_size in (1, 3, 64):
        rows = evaluate.predict(model, tiles, ["T1"], settings, pipeline, batch_size=batch_size)
        _, logit, probability = rows[0]
        assert logit == pytest.approx(expected.item(), abs=1e-5)
        assert probability == pytest.approx(torch.sigmoid(expected).item(), abs=1e-5)


def test_workers_give_the_same_scores_for_every_patient_in_order(scored):
    """Reading and normalizing chunks in worker processes must not change a score or its patient."""
    tiles, settings, pipeline, model = scored
    patients = ["T3", "T1", "T2"]
    alone = evaluate.predict(model, tiles, patients, settings, pipeline, batch_size=3)
    shared = evaluate.predict(model, tiles, patients, settings, pipeline, batch_size=3,
                              num_workers=2)
    assert [row[0] for row in shared] == patients
    assert [row[1:] for row in shared] == [row[1:] for row in alone]


def test_workers_beyond_the_chunk_count_still_score_every_patient(scored):
    """More workers than chunks must leave no patient out and no hang."""
    tiles, settings, pipeline, model = scored
    alone = evaluate.predict(model, tiles, ["T2"], settings, pipeline)
    crowded = evaluate.predict(model, tiles, ["T2"], settings, pipeline, num_workers=3)
    assert crowded == alone


def test_every_chunk_is_scored_under_its_own_patient_with_workers(scored):
    """Patients with different tiles must not swap their chunks when the loader reorders work."""
    tiles, settings, pipeline, _ = scored
    chunks = data.EvalChunks(tiles, ["T3", "T1"], settings, pipeline, batch_size=3)
    assert [chunks[i][0] for i in range(len(chunks))] == ["T3", "T3", "T1", "T1", "T1", "T1"]
    for patient in ("T3", "T1"):
        rows = tiles[tiles["patient_id"] == patient].sort_values(
            ["slide_id", "y_target", "x_target"])
        expected = torch.stack([pipeline.tile(row) for _, row in rows.iterrows()])
        mine = torch.cat([chunk for name, chunk in (chunks[i] for i in range(len(chunks)))
                          if name == patient])
        assert torch.equal(mine, expected)


def test_the_worker_count_and_start_method_reach_the_loader(scored, monkeypatch):
    """Workers must use the warm-server start method, not the slow default."""
    tiles, settings, pipeline, model = scored
    seen = []
    real = evaluate.DataLoader

    def spy(dataset, **kwargs):
        seen.append((kwargs["num_workers"], kwargs["multiprocessing_context"]))
        return real(dataset, batch_size=None)

    monkeypatch.setattr(evaluate, "DataLoader", spy)
    evaluate.predict(model, tiles, ["T1"], settings, pipeline, num_workers=3)
    evaluate.predict(model, tiles, ["T1"], settings, pipeline)
    assert seen == [(3, "forkserver"), (0, None)]


def test_a_failed_normalization_stops_scoring_in_a_worker_too(scored, tmp_path):
    """A tile that cannot be normalized must raise, not be skipped or scored raw, workers or not."""
    tiles, settings, pipeline, model = scored
    failing = data.TilePipeline(pipeline.package, settings, FailingNormalizer(),
                                tmp_path / "failures.csv")
    for workers in (0, 1):
        with pytest.raises((ValueError, RuntimeError), match="no stain found"):
            evaluate.predict(model, tiles, ["T1"], settings, failing, num_workers=workers)


def test_a_patient_listed_twice_is_refused(scored):
    """Chunks are grouped by patient, so a repeat would pool two copies into one score."""
    tiles, settings, pipeline, model = scored
    with pytest.raises(ValueError, match="listed twice"):
        evaluate.predict(model, tiles, ["T1", "T1"], settings, pipeline)


def test_no_patients_gives_no_scores(scored):
    """An empty validation list is an empty result, not an error."""
    tiles, settings, pipeline, model = scored
    assert not evaluate.predict(model, tiles, [], settings, pipeline)


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
