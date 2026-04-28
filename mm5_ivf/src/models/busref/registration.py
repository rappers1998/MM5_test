from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _identity_grid(batch: int, height: int, width: int, device: torch.device) -> torch.Tensor:
    yy, xx = torch.meshgrid(
        torch.linspace(-1.0, 1.0, height, device=device),
        torch.linspace(-1.0, 1.0, width, device=device),
        indexing="ij",
    )
    return torch.stack([xx, yy], dim=-1).unsqueeze(0).repeat(batch, 1, 1, 1)


def warp_with_flow(image: torch.Tensor, flow: torch.Tensor, padding_mode: str = "border") -> torch.Tensor:
    batch, _channels, height, width = image.shape
    grid = _identity_grid(batch, height, width, image.device)
    norm = torch.empty_like(flow)
    norm[:, 0] = flow[:, 0] * (2.0 / max(width - 1, 1))
    norm[:, 1] = flow[:, 1] * (2.0 / max(height - 1, 1))
    grid = grid + norm.permute(0, 2, 3, 1)
    return F.grid_sample(image, grid, mode="bilinear", padding_mode=padding_mode, align_corners=True)


def warp_with_affine(image: torch.Tensor, theta_delta: torch.Tensor) -> torch.Tensor:
    batch = image.shape[0]
    identity = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], device=image.device).unsqueeze(0).repeat(batch, 1, 1)
    theta = identity + theta_delta
    grid = F.affine_grid(theta, image.shape, align_corners=False)
    return F.grid_sample(image, grid, mode="bilinear", padding_mode="border", align_corners=False)


class AffineNet(nn.Module):
    def __init__(self, in_channels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 32, 7, padding=3),
            nn.SiLU(inplace=True),
            nn.Conv2d(32, 32, 3, stride=2, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(32, 48, 3, stride=2, padding=1),
            nn.SiLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(48, 32),
            nn.SiLU(inplace=True),
            nn.Linear(32, 6),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, fixed_feature: torch.Tensor, moving_feature: torch.Tensor) -> torch.Tensor:
        delta = self.net(torch.cat([fixed_feature, moving_feature], dim=1)).view(-1, 2, 3)
        return delta * 0.05


class DeformableNet(nn.Module):
    def __init__(self, in_channels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(32, 2, 3, padding=1),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, fixed: torch.Tensor, moving: torch.Tensor) -> torch.Tensor:
        flow = self.net(torch.cat([fixed, moving, torch.abs(fixed - moving)], dim=1))
        return torch.tanh(flow) * 6.0


class BusReFRegistration(nn.Module):
    def __init__(self, feature_channels: int = 64) -> None:
        super().__init__()
        self.affine = AffineNet(feature_channels * 2)
        self.deform = DeformableNet(3)

    def forward(
        self,
        fixed_visible: torch.Tensor,
        moving_thermal: torch.Tensor,
        fixed_deep_feature: torch.Tensor,
        moving_deep_feature: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        theta_delta = self.affine(fixed_deep_feature, moving_deep_feature)
        affine_moving = warp_with_affine(moving_thermal, theta_delta)
        flow = self.deform(fixed_visible, affine_moving)
        registered = warp_with_flow(affine_moving, flow)
        return {
            "theta_delta": theta_delta,
            "affine_moving": affine_moving,
            "flow": flow,
            "registered": registered,
        }
