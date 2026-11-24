"""ConvNeXt V2-T from timm: maps (B, 3, 224, 224) tiles to (B, 768) embeddings."""
import timm
import torch.nn as nn

def build(settings, pretrained=True):
    """ConvNeXt V2-T with FCMAE weights (unless `pretrained` is False) and the classifier removed."""
    del settings
    model = timm.create_model(
        "convnextv2_tiny.fcmae_ft_in22k_in1k",
        pretrained=pretrained,
        num_classes=0  # Removes classifier head, returns raw feature embeddings
    )
    length = model.num_features
    return model, length


def stages(model):
    """The four stages, earliest first, as lists of modules, covering every parameter once.

    The stem goes with stage 1 (timm keeps each downsampling inside it), the head norm with 4.
    """
    return [[model.stem, model.stages[0]], [model.stages[1]], [model.stages[2]],
            [model.stages[3], model.head]]
