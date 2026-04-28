from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1),
            nn.GroupNorm(4, out_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.GroupNorm(4, out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ResidualFeature(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GroupNorm(4, channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GroupNorm(4, channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.silu(x + self.conv(x))


class BusReconstructor(nn.Module):
    """Compact AE bus: three down blocks, four RFEs, three up blocks."""

    def __init__(self, base_channels: int = 16) -> None:
        super().__init__()
        c1, c2, c3 = base_channels, base_channels * 2, base_channels * 4
        self.down1 = ConvBlock(1, c1, stride=1)
        self.down2 = ConvBlock(c1, c2, stride=2)
        self.down3 = ConvBlock(c2, c3, stride=2)
        self.rfe1 = ResidualFeature(c3)
        self.rfe2 = ResidualFeature(c3)
        self.rfe3 = ResidualFeature(c3)
        self.rfe4 = ResidualFeature(c3)
        self.up2 = ConvBlock(c3 + c2, c2)
        self.up1 = ConvBlock(c2 + c1, c1)
        self.out = nn.Sequential(nn.Conv2d(c1, 1, 3, padding=1), nn.Sigmoid())

    def encode(self, x: torch.Tensor) -> list[torch.Tensor]:
        f1 = self.down1(x)
        f2 = self.down2(f1)
        f3 = self.down3(f2)
        r1 = self.rfe1(f3)
        r2 = self.rfe2(r1)
        r3 = self.rfe3(r2)
        r4 = self.rfe4(r3)
        return [f1, f2, r1, r4]

    def decode(self, features: list[torch.Tensor]) -> torch.Tensor:
        f1, f2, _r1, r4 = features
        x = F.interpolate(r4, size=f2.shape[-2:], mode="bilinear", align_corners=False)
        x = self.up2(torch.cat([x, f2], dim=1))
        x = F.interpolate(x, size=f1.shape[-2:], mode="bilinear", align_corners=False)
        x = self.up1(torch.cat([x, f1], dim=1))
        return self.out(x)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        features = self.encode(x)
        return self.decode(features), features
