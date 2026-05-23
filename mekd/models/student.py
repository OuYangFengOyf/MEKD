from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from .layers import FPNFuse
from .stdc import STDCEncoder


class STDCFPNStudent(nn.Module):
    def __init__(self, num_classes: int, base_channels=32, fpn_channels=128, dropout=0.1):
        super().__init__()
        self.encoder = STDCEncoder(base_channels=base_channels)
        self.decoder = FPNFuse(self.encoder.out_channels, fpn_channels, num_classes, dropout=dropout)

    @property
    def out_channels(self):
        return self.encoder.out_channels

    def forward(self, x: torch.Tensor):
        feats = self.encoder(x)
        logits, fpn_feats = self.decoder(feats)
        logits = F.interpolate(logits, size=x.shape[-2:], mode="bilinear", align_corners=False)
        return {"logits": logits, "features": feats, "fpn_features": fpn_feats}
