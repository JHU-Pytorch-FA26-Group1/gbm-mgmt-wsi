"""Contract tests for the backbones: (B, 3, 224, 224) tiles in, (B, D) embeddings out."""
import torch

from training.backbones import convnext, swin


def test_swin_embeds_224_tiles_to_the_length_it_reports():
    """The pooling head is sized from the reported length, so it must match the real output."""
    model, length = swin.build({}, pretrained=False)
    model.eval()
    with torch.no_grad():
        out = model(torch.zeros(2, 3, 224, 224))
    assert out.shape == (2, length)


def test_each_backbones_stages_cover_every_parameter_once():
    """A parameter missing from the stages could never be frozen; one listed twice freezes two."""
    for module in (swin, convnext):
        model, _ = module.build({}, pretrained=False)
        stages = module.stages(model)
        listed = [id(p) for group in stages for part in group for p in part.parameters()]
        assert len(stages) == 4
        assert sorted(listed) == sorted(id(p) for p in model.parameters())
