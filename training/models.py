"""Attention-pooling model: a backbone embeds each tile, attention pools them to one logit."""
import importlib

import torch
from torch import nn

STAGES = 4   # every backbone exposes four stages (backbones/<name>.py `stages`)
BACKBONE_MODULES = {"swin_v2_t": "training.backbones.swin",
                    "convnext_v2_t": "training.backbones.convnext"}


class AttentionPool(nn.Module):
    """Softmax attention over a patient's tiles, then a linear classifier on the weighted sum."""

    def __init__(self, length, hidden, dropout):
        super().__init__()
        self.score = nn.Sequential(nn.Linear(length, hidden), nn.Tanh(), nn.Dropout(dropout),
                                   nn.Linear(hidden, 1))
        self.classifier = nn.Linear(length, 1)

    def forward(self, embeddings):
        """(patients, tiles, length) -> logits (patients,), attention (patients, tiles)."""
        attention = torch.softmax(self.score(embeddings).squeeze(-1), dim=1)
        pooled = (attention.unsqueeze(-1) * embeddings).sum(dim=1)
        return self.classifier(pooled).squeeze(-1), attention


class BagModel(nn.Module):
    """Backbone plus attention pooling; the patient label is predicted only after pooling."""

    def __init__(self, backbone, length, settings):
        super().__init__()
        self.backbone = backbone
        self.pool = AttentionPool(length, settings["attention_hidden"],
                                  settings["attention_dropout"])
        self.frozen = []   # kept in eval mode so a frozen layer does not drop paths while training

    def train(self, mode=True):
        """Switch modes, keeping frozen modules in eval mode."""
        super().train(mode)
        if mode:
            for module in self.frozen:
                module.eval()
        return self

    def encode(self, tiles):
        """(n, 3, H, W) -> (n, length)."""
        return self.backbone(tiles)

    def forward(self, bags):
        """(patients, tiles, 3, H, W) -> logits (patients,), attention (patients, tiles)."""
        patients, tiles = bags.shape[:2]
        embeddings = self.encode(bags.flatten(0, 1)).view(patients, tiles, -1)
        return self.pool(embeddings)


def build_model(settings, backbone, pretrained=True):
    """The BagModel for the named backbone; `pretrained` downloads ImageNet weights on first use."""
    if backbone not in BACKBONE_MODULES:
        raise ValueError(f"unknown backbone {backbone!r}, "
                         f"expected one of {sorted(BACKBONE_MODULES)}")
    module, length = importlib.import_module(BACKBONE_MODULES[backbone]).build(settings, pretrained)
    return BagModel(module, length, settings)


def freeze_backbone(model, backbone, trainable_stages):
    """Freeze all but the last `trainable_stages` of the four stages (the pool always trains)."""
    if trainable_stages == STAGES:
        return
    groups = importlib.import_module(BACKBONE_MODULES[backbone]).stages(model.backbone)
    for group in groups[:STAGES - trainable_stages]:
        for module in group:
            module.requires_grad_(False)
            model.frozen.append(module)
