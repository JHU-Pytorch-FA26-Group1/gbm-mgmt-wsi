"""Contract tests for the backbones: (B, 3, 224, 224) tiles in, (B, D) embeddings out."""
import torch

from training.backbones import swin


def test_swin_embeds_224_tiles_to_the_length_it_reports():
    """The pooling head is sized from the reported length, so it must match the real output."""
    model, length = swin.build({}, pretrained=False)
    model.eval()
    with torch.no_grad():
        out = model(torch.zeros(2, 3, 224, 224))
    assert out.shape == (2, length)
