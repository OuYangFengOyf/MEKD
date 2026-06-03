from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class ConvBNAct(nn.Sequential):
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, groups=1, act=True):
        padding = kernel_size // 2
        layers = [
            nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_ch),
        ]
        if act:
            layers.append(nn.SiLU(inplace=True))
        super().__init__(*layers)


class DSConv(nn.Sequential):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__(
            ConvBNAct(in_ch, in_ch, 3, stride=stride, groups=in_ch),
            ConvBNAct(in_ch, out_ch, 1),
        )


class FPNFuse(nn.Module):
    def __init__(self, in_channels, fpn_channels, num_classes, dropout=0.1):
        super().__init__()
        self.proj = nn.ModuleList([ConvBNAct(ch, fpn_channels, 1) for ch in in_channels])
        self.smooth = nn.ModuleList([DSConv(fpn_channels, fpn_channels) for _ in in_channels])
        self.head = nn.Sequential(
            DSConv(fpn_channels, fpn_channels),
            nn.Dropout2d(dropout),
            nn.Conv2d(fpn_channels, num_classes, 1),
        )

    def forward(self, feats):
        feats = list(feats)
        x = self.proj[-1](feats[-1])
        outs = [None] * len(feats)
        outs[-1] = self.smooth[-1](x)
        for i in range(len(feats) - 2, -1, -1):
            x = F.interpolate(x, size=feats[i].shape[-2:], mode="bilinear", align_corners=False)
            x = x + self.proj[i](feats[i])
            outs[i] = self.smooth[i](x)
        logits = self.head(outs[0])
        return logits, outs


def sinusoidal_2d_pos_embed(h: int, w: int, dim: int, device) -> torch.Tensor:
    y, x = torch.meshgrid(torch.arange(h, device=device), torch.arange(w, device=device), indexing="ij")
    omega = torch.arange(dim // 4, device=device).float()
    omega = 1.0 / (10000 ** (omega / max(1, dim // 4)))
    y = y.flatten().float()[:, None] * omega[None, :]
    x = x.flatten().float()[:, None] * omega[None, :]
    pos = torch.cat([x.sin(), x.cos(), y.sin(), y.cos()], dim=1)
    if pos.shape[1] < dim:
        pos = F.pad(pos, (0, dim - pos.shape[1]))
    return pos[:, :dim]


def resize_like(x: torch.Tensor, ref: torch.Tensor, mode="bilinear") -> torch.Tensor:
    if x.shape[-2:] == ref.shape[-2:]:
        return x
    return F.interpolate(x, size=ref.shape[-2:], mode=mode, align_corners=False if mode == "bilinear" else None)
