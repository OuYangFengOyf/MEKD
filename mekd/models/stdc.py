from __future__ import annotations

import torch
from torch import nn

from .layers import ConvBNAct


class CatBottleneck(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1, blocks: int = 4):
        super().__init__()
        assert out_ch % blocks == 0
        branch_ch = out_ch // blocks
        self.stride = stride
        self.blocks = blocks
        self.down = nn.AvgPool2d(3, stride=2, padding=1) if stride == 2 else nn.Identity()
        self.convs = nn.ModuleList()
        self.convs.append(ConvBNAct(in_ch, branch_ch, 1, stride=1))
        for _ in range(1, blocks):
            self.convs.append(ConvBNAct(branch_ch, branch_ch, 3, stride=1))
        self.short = ConvBNAct(in_ch, branch_ch, 1, stride=1) if stride == 2 else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.stride == 2:
            x_down = self.down(x)
            y = self.convs[0](x_down)
            outs = [self.short(x_down)]
        else:
            y = self.convs[0](x)
            outs = [y]
        for conv in self.convs[1:]:
            y = conv(y)
            outs.append(y)
        return torch.cat(outs, dim=1)


class STDCEncoder(nn.Module):
    """STDC2-style lightweight encoder returning 1/4, 1/8, 1/16 and 1/32 features."""

    def __init__(self, base_channels: int = 32):
        super().__init__()
        c = base_channels
        self.stem = nn.Sequential(
            ConvBNAct(3, c, 3, stride=2),
            ConvBNAct(c, c * 2, 3, stride=2),
        )
        self.stage8 = nn.Sequential(
            CatBottleneck(c * 2, c * 4, stride=2),
            CatBottleneck(c * 4, c * 4),
            CatBottleneck(c * 4, c * 4),
            CatBottleneck(c * 4, c * 4),
        )
        self.stage16 = nn.Sequential(
            CatBottleneck(c * 4, c * 8, stride=2),
            CatBottleneck(c * 8, c * 8),
            CatBottleneck(c * 8, c * 8),
            CatBottleneck(c * 8, c * 8),
            CatBottleneck(c * 8, c * 8),
        )
        self.stage32 = nn.Sequential(
            CatBottleneck(c * 8, c * 16, stride=2),
            CatBottleneck(c * 16, c * 16),
            CatBottleneck(c * 16, c * 16),
        )
        self.out_channels = [c * 2, c * 4, c * 8, c * 16]

    def forward(self, x: torch.Tensor):
        f4 = self.stem(x)
        f8 = self.stage8(f4)
        f16 = self.stage16(f8)
        f32 = self.stage32(f16)
        return f4, f8, f16, f32
