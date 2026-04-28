from __future__ import annotations

import random
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class PatchOp:
    row: int
    col: int
    rot_k: int
    flip_h: bool
    flip_v: bool


def _apply_op(patch: torch.Tensor, op: PatchOp) -> torch.Tensor:
    out = torch.rot90(patch, op.rot_k, dims=(-2, -1))
    if op.flip_h:
        out = torch.flip(out, dims=(-1,))
    if op.flip_v:
        out = torch.flip(out, dims=(-2,))
    return out


def _invert_op(patch: torch.Tensor, op: PatchOp) -> torch.Tensor:
    out = patch
    if op.flip_v:
        out = torch.flip(out, dims=(-2,))
    if op.flip_h:
        out = torch.flip(out, dims=(-1,))
    return torch.rot90(out, -op.rot_k, dims=(-2, -1))


def pdg_tensor(image: torch.Tensor, n: int = 2, seed: int | None = None) -> tuple[torch.Tensor, list[PatchOp]]:
    if seed is not None:
        random.seed(seed)
    batch, channels, height, width = image.shape
    ph = height // n
    pw = width // n
    output = image.clone()
    ops: list[PatchOp] = []
    for row in range(n):
        for col in range(n):
            y0 = row * ph
            x0 = col * pw
            y1 = height if row == n - 1 else y0 + ph
            x1 = width if col == n - 1 else x0 + pw
            op = PatchOp(row=row, col=col, rot_k=random.randint(0, 3), flip_h=bool(random.getrandbits(1)), flip_v=bool(random.getrandbits(1)))
            patch = image[:, :, y0:y1, x0:x1]
            output[:, :, y0:y1, x0:x1] = F.interpolate(_apply_op(patch, op), size=(y1 - y0, x1 - x0), mode="bilinear", align_corners=False)
            ops.append(op)
    return output, ops


def ipdg_tensor(image: torch.Tensor, ops: list[PatchOp], n: int = 2) -> torch.Tensor:
    _batch, _channels, height, width = image.shape
    ph = height // n
    pw = width // n
    output = image.clone()
    for op in ops:
        y0 = op.row * ph
        x0 = op.col * pw
        y1 = height if op.row == n - 1 else y0 + ph
        x1 = width if op.col == n - 1 else x0 + pw
        patch = image[:, :, y0:y1, x0:x1]
        output[:, :, y0:y1, x0:x1] = F.interpolate(_invert_op(patch, op), size=(y1 - y0, x1 - x0), mode="bilinear", align_corners=False)
    return output


def warp_tensor(image: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
    batch, _channels, height, width = image.shape
    yy, xx = torch.meshgrid(
        torch.linspace(-1.0, 1.0, height, device=image.device),
        torch.linspace(-1.0, 1.0, width, device=image.device),
        indexing="ij",
    )
    grid = torch.stack([xx, yy], dim=-1).unsqueeze(0).repeat(batch, 1, 1, 1)
    norm = torch.empty_like(flow)
    norm[:, 0] = flow[:, 0] * (2.0 / max(width - 1, 1))
    norm[:, 1] = flow[:, 1] * (2.0 / max(height - 1, 1))
    return F.grid_sample(image, grid + norm.permute(0, 2, 3, 1), mode="bilinear", padding_mode="border", align_corners=True)


class _Conv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.GroupNorm(4, out_ch),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.GroupNorm(4, out_ch),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class BSRFlowNet(nn.Module):
    def __init__(self, base_channels: int = 16) -> None:
        super().__init__()
        self.enc1 = _Conv(2, base_channels)
        self.enc2 = _Conv(base_channels, base_channels * 2)
        self.mid = _Conv(base_channels * 2, base_channels * 2)
        self.dec1 = _Conv(base_channels * 3, base_channels)
        self.out = nn.Conv2d(base_channels, 4, 3, padding=1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, thermal: torch.Tensor, visible: torch.Tensor) -> dict[str, torch.Tensor]:
        x1 = self.enc1(torch.cat([thermal, visible], dim=1))
        x2 = self.enc2(F.avg_pool2d(x1, 2))
        mid = self.mid(x2)
        up = F.interpolate(mid, size=x1.shape[-2:], mode="bilinear", align_corners=False)
        flow = torch.tanh(self.out(self.dec1(torch.cat([up, x1], dim=1)))) * 6.0
        phi_p = flow[:, :2]
        phi_n = flow[:, 2:]
        return {
            "phi_p": phi_p,
            "phi_n": phi_n,
            "thermal_to_visible": warp_tensor(thermal, phi_p),
            "visible_to_thermal": warp_tensor(visible, phi_n),
        }
