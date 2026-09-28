"""Loader for a public pretrained CosFace face-embedding model."""

import torch
import torch.nn.functional as F
from torch import nn
import timm


PRETRAINED_COSFACE_MODEL = "hf_hub:gaunernst/vit_tiny_patch8_112.cosface_ms1mv3"


class CosFaceEmbeddingModel(nn.Module):
    """ViT-Tiny face encoder pretrained with the CosFace objective."""

    def __init__(self):
        super().__init__()
        self.backbone = timm.create_model(
            PRETRAINED_COSFACE_MODEL,
            pretrained=True,
        )

    def forward(self, images):
        embeddings = self.backbone(images)
        return F.normalize(embeddings, p=2, dim=1)
