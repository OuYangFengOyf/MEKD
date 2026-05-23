from __future__ import annotations

from typing import Tuple

import torch
from torch import nn
import torch.nn.functional as F

from .layers import ConvBNAct, DSConv, sinusoidal_2d_pos_embed


class TransformerSemanticExpert(nn.Module):
    def __init__(self, in16: int, in32: int, channels: int, num_classes: int, layers=2, heads=8):
        super().__init__()
        self.proj = ConvBNAct(in16 + in32, channels, 1)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=channels,
            nhead=heads,
            dim_feedforward=channels * 4,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=layers)
        self.head = nn.Conv2d(channels, num_classes, 1)

    def forward(self, f16: torch.Tensor, f32: torch.Tensor):
        f32 = F.interpolate(f32, size=f16.shape[-2:], mode="bilinear", align_corners=False)
        x = self.proj(torch.cat([f16, f32], dim=1))
        b, c, h, w = x.shape
        tokens = x.flatten(2).transpose(1, 2)
        pos = sinusoidal_2d_pos_embed(h, w, c, x.device).unsqueeze(0).to(tokens.dtype)
        tokens = self.encoder(tokens + pos)
        feat = tokens.transpose(1, 2).reshape(b, c, h, w)
        return feat, self.head(feat)


class SelectiveScan2D(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.in_proj = nn.Conv2d(channels, channels * 3, 1)
        self.out_proj = ConvBNAct(channels, channels, 1)

    def _scan_lr(self, x, a, b):
        state = torch.zeros_like(x[:, :, :, 0])
        outs = []
        for i in range(x.shape[-1]):
            state = a[:, :, :, i] * state + b[:, :, :, i] * x[:, :, :, i]
            outs.append(state)
        return torch.stack(outs, dim=-1)

    def forward(self, x: torch.Tensor, direction: str) -> torch.Tensor:
        u, a, b = self.in_proj(x).chunk(3, dim=1)
        a = torch.sigmoid(a)
        b = torch.sigmoid(b)
        if direction == "rl":
            y = torch.flip(self._scan_lr(torch.flip(u, [-1]), torch.flip(a, [-1]), torch.flip(b, [-1])), [-1])
        elif direction == "tb":
            y = self._scan_lr(u.transpose(-1, -2), a.transpose(-1, -2), b.transpose(-1, -2)).transpose(-1, -2)
        elif direction == "bt":
            uf = torch.flip(u.transpose(-1, -2), [-1])
            af = torch.flip(a.transpose(-1, -2), [-1])
            bf = torch.flip(b.transpose(-1, -2), [-1])
            y = torch.flip(self._scan_lr(uf, af, bf), [-1]).transpose(-1, -2)
        else:
            y = self._scan_lr(u, a, b)
        return self.out_proj(y)


class DirectionalMambaBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.norm = nn.BatchNorm2d(channels)
        self.scans = nn.ModuleDict({d: SelectiveScan2D(channels) for d in ["lr", "rl", "tb", "bt"]})
        self.dir_weight = nn.Conv2d(channels, 4, 1)
        self.ffn = nn.Sequential(ConvBNAct(channels, channels * 2, 1), ConvBNAct(channels * 2, channels, 1, act=False))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z = self.norm(x)
        weights = torch.softmax(self.dir_weight(z), dim=1)
        dirs = []
        for idx, key in enumerate(["lr", "rl", "tb", "bt"]):
            dirs.append(self.scans[key](z) * weights[:, idx : idx + 1])
        y = sum(dirs)
        x = x + y
        x = x + self.ffn(x)
        return x, weights


class MambaSpatialExpert(nn.Module):
    def __init__(self, in16: int, in32: int, channels: int, num_classes: int, blocks=2):
        super().__init__()
        self.proj = ConvBNAct(in16 + in32, channels, 1)
        self.local = DSConv(channels, channels)
        self.blocks = nn.ModuleList([DirectionalMambaBlock(channels) for _ in range(blocks)])
        self.integrate = ConvBNAct(channels * 2 + 1, channels, 1)
        self.head = nn.Conv2d(channels, num_classes, 1)
        self.gamma_l = nn.Parameter(torch.tensor(1.0))
        self.gamma_g = nn.Parameter(torch.tensor(1.0))

    def forward(self, f16: torch.Tensor, f32: torch.Tensor, density: torch.Tensor):
        f32 = F.interpolate(f32, size=f16.shape[-2:], mode="bilinear", align_corners=False)
        x_in = self.proj(torch.cat([f16, f32], dim=1))
        local = self.local(x_in)
        global_feat = x_in
        for block in self.blocks:
            global_feat, _ = block(global_feat)
        density = F.interpolate(density, size=x_in.shape[-2:], mode="bilinear", align_corners=False)
        fused = (
            x_in
            + self.gamma_l * density * local
            + self.gamma_g * (1.0 - density) * global_feat
            + self.integrate(torch.cat([local, global_feat, density], dim=1))
        )
        return fused, self.head(fused)
