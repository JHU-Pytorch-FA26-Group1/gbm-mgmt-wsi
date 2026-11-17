"""ConvNeXt V2-T: maps (B, 3, 224, 224) tiles to (B, D) embeddings."""


def build(settings, pretrained=True):
    """ConvNeXt V2-T with its classifier removed; returns (module, embedding length)."""
    # TODO(Couger): torchvision has only ConvNeXt V1 (convnext_tiny); pick the V2 source
    raise NotImplementedError("convnext_v2_t backbone is not written yet")
