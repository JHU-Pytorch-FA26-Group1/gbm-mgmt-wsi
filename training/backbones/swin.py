"""Swin V2-T from torchvision: maps (B, 3, 224, 224) tiles to (B, 768) embeddings."""
from torch import nn
from torchvision.models import Swin_V2_T_Weights, swin_v2_t


def build(settings, pretrained=True):
    """Swin V2-T with ImageNet weights (unless `pretrained` is False) and the classifier removed."""
    del settings
    model = swin_v2_t(weights=Swin_V2_T_Weights.IMAGENET1K_V1 if pretrained else None)
    length = model.head.in_features
    model.head = nn.Identity()
    return model, length


def stages(model):
    """The four stages, earliest first, as lists of modules, covering every parameter once.

    The patch embedding goes with stage 1, each patch merging with the stage it feeds, the final
    norm with stage 4.
    """
    features = model.features
    return [[features[0], features[1]], [features[2], features[3]], [features[4], features[5]],
            [features[6], features[7], model.norm]]
