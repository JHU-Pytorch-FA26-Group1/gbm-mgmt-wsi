"""Regression tests for the attention-pooling model."""
import sys
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from training import models

SETTINGS = {"attention_hidden": 8, "attention_dropout": 0.0}


def test_attention_sums_to_one_over_each_patients_own_tiles():
    """Normalizing over the wrong axis would mix patients or tiles."""
    pool = models.AttentionPool(4, 8, 0.0)
    _, attention = pool(torch.randn(3, 5, 4))
    assert attention.shape == (3, 5)
    assert torch.allclose(attention.sum(dim=1), torch.ones(3))


def test_the_logit_follows_the_tile_with_the_highest_attention_score():
    """Attention weights the sum: a dominant tile sets the logit, where a plain mean would not."""
    pool = models.AttentionPool(2, 2, 0.0).eval()
    with torch.no_grad():
        pool.score[0].weight.copy_(torch.tensor([[1.0, 0.0], [0.0, 0.0]]))
        pool.score[0].bias.zero_()
        pool.score[3].weight.copy_(torch.tensor([[20.0, 0.0]]))
        pool.score[3].bias.zero_()
        pool.classifier.weight.copy_(torch.tensor([[0.0, 1.0]]))
        pool.classifier.bias.zero_()
    tiles = torch.tensor([[[10.0, 5.0], [0.0, 1.0]]])
    logit, attention = pool(tiles)
    assert attention[0, 0] > 0.99
    assert logit.item() == pytest.approx(5.0, abs=0.01)


def test_the_attention_dropout_setting_reaches_the_pool():
    """The configured dropout must be the one the attention layer uses."""
    model = models.BagModel(nn.Identity(), 4, {"attention_hidden": 8, "attention_dropout": 0.5})
    rates = [layer.p for layer in model.pool.score if isinstance(layer, nn.Dropout)]
    assert rates == [0.5]


def test_the_backbone_is_trained_through_the_pooled_logit():
    """Fine-tuning needs gradients to reach the backbone, not just the pooling head."""
    backbone = nn.Linear(3, 4)
    model = models.BagModel(backbone, 4, SETTINGS)
    logits, _ = model(torch.randn(2, 5, 3))
    logits.sum().backward()
    assert backbone.weight.grad is not None and backbone.weight.grad.abs().sum() > 0


def test_build_model_calls_the_named_backbone_with_the_pretrained_flag(monkeypatch):
    """The model must come from the file for the chosen name, with `pretrained` passed through."""
    calls = []

    def build(_, pretrained):
        calls.append(pretrained)
        return nn.Linear(3, 4), 4

    monkeypatch.setitem(sys.modules, "fake_backbone", SimpleNamespace(build=build))
    monkeypatch.setitem(models.BACKBONE_MODULES, "swin_v2_t", "fake_backbone")
    model = models.build_model(SETTINGS, "swin_v2_t", pretrained=False)
    assert calls == [False] and isinstance(model.backbone, nn.Linear)


def test_a_patients_logit_does_not_depend_on_other_patients_or_tile_order():
    """One logit per patient, from the set of tiles: batching and ordering must not change it."""
    pool = models.AttentionPool(4, 8, 0.0).eval()
    bag = torch.randn(1, 6, 4)
    alone, _ = pool(bag)
    shuffled, _ = pool(torch.cat([bag[:, torch.randperm(6)], torch.randn(1, 6, 4)]))
    assert torch.allclose(alone, shuffled[:1], atol=1e-6)


def test_bag_model_flattens_tiles_for_the_backbone_and_restores_patients():
    """Tiles of different patients must not be regrouped when the bag is flattened."""
    backbone = nn.Sequential(nn.Flatten(), nn.Linear(3 * 2 * 2, 4))
    model = models.BagModel(backbone, 4, SETTINGS).eval()
    bags = torch.randn(2, 5, 3, 2, 2)
    logits, attention = model(bags)
    assert logits.shape == (2,) and attention.shape == (2, 5)
    single, _ = model(bags[1:])
    assert torch.allclose(logits[1:], single, atol=1e-6)


def test_unknown_backbone_is_refused():
    """A typo must stop the run, not fall back to some default model."""
    with pytest.raises(ValueError):
        models.build_model(SETTINGS, "resnet50")


class Stub(nn.Module):
    """A four-stage backbone: one linear layer and one dropout per stage, as `stages` lists them."""

    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList(nn.Linear(4, 4) for _ in range(4))
        self.dropouts = nn.ModuleList(nn.Dropout(0.5) for _ in range(4))

    def forward(self, tiles):
        """Each stage's layer then its dropout."""
        for layer, dropout in zip(self.layers, self.dropouts):
            tiles = dropout(layer(tiles))
        return tiles


def stub_stages(model):
    """The stub's stages, earliest first."""
    return [[layer, dropout] for layer, dropout in zip(model.layers, model.dropouts)]


@pytest.fixture(name="stub_model")
def fixture_stub_model(monkeypatch):
    """A BagModel on the stub backbone, registered under the swin name."""
    monkeypatch.setitem(sys.modules, "fake_stages", SimpleNamespace(stages=stub_stages))
    monkeypatch.setitem(models.BACKBONE_MODULES, "swin_v2_t", "fake_stages")
    return models.BagModel(Stub(), 4, SETTINGS)


@pytest.mark.parametrize("trainable, frozen",
                         [(4, []), (3, [0]), (1, [0, 1, 2]), (0, [0, 1, 2, 3])])
def test_only_the_stages_before_the_last_n_are_frozen(stub_model, trainable, frozen):
    """n counts from the last stage; the wrong end would train the stem and freeze the top."""
    models.freeze_backbone(stub_model, "swin_v2_t", trainable)
    layers = stub_model.backbone.layers
    flags = [all(p.requires_grad for p in layer.parameters()) for layer in layers]
    assert flags == [index not in frozen for index in range(4)]
    assert all(p.requires_grad for p in stub_model.pool.parameters())


def test_a_frozen_stage_does_not_move_and_the_stages_after_it_do(stub_model):
    """Frozen weights stay identical through an optimizer step; the trainable ones change."""
    models.freeze_backbone(stub_model, "swin_v2_t", 2)
    before = [layer.weight.clone() for layer in stub_model.backbone.layers]
    optimizer = torch.optim.Adam([p for p in stub_model.parameters() if p.requires_grad], lr=0.1)
    logits, _ = stub_model(torch.randn(2, 5, 4))
    logits.sum().backward()
    optimizer.step()
    layers = stub_model.backbone.layers
    moved = [not torch.equal(b, layer.weight) for b, layer in zip(before, layers)]
    assert moved == [False, False, True, True]


def test_frozen_stages_stay_in_eval_mode_when_the_model_trains(stub_model):
    """A frozen stage that kept dropping paths would embed a tile differently on every pass."""
    models.freeze_backbone(stub_model, "swin_v2_t", 2)
    for mode in (stub_model.train, stub_model.eval, stub_model.train):
        mode()
    modes = [dropout.training for dropout in stub_model.backbone.dropouts]
    assert modes == [False, False, True, True]
