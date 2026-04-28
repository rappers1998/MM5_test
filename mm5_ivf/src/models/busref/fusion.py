from __future__ import annotations

import torch
import torch.nn as nn


class GAFFusion(nn.Module):
    """Lightweight gradient-aware fusion head."""

    def __init__(self, channels: int = 24) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(4, channels, 3, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels, 1, 3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, visible: torch.Tensor, thermal: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        if mask is None:
            mask = torch.ones_like(visible)
        diff = torch.abs(visible - thermal)
        blend = self.encoder(torch.cat([visible, thermal, diff, mask], dim=1))
        fused = visible * (1.0 - blend * mask) + thermal * (blend * mask)
        return fused.clamp(0.0, 1.0)
