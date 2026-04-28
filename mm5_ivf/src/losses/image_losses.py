from __future__ import annotations

import torch
import torch.nn.functional as F


def masked_mean(value: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    if mask is None:
        return value.mean()
    mask = mask.to(dtype=value.dtype)
    return (value * mask).sum() / mask.sum().clamp_min(1.0)


def masked_mse(a: torch.Tensor, b: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    return masked_mean((a - b) ** 2, mask)


def masked_ncc(a: torch.Tensor, b: torch.Tensor, mask: torch.Tensor | None = None, eps: float = 1e-6) -> torch.Tensor:
    if mask is None:
        mask = torch.ones_like(a)
    mask = mask.to(dtype=a.dtype)
    denom = mask.sum(dim=(-2, -1), keepdim=True).clamp_min(1.0)
    a_mean = (a * mask).sum(dim=(-2, -1), keepdim=True) / denom
    b_mean = (b * mask).sum(dim=(-2, -1), keepdim=True) / denom
    a0 = (a - a_mean) * mask
    b0 = (b - b_mean) * mask
    num = (a0 * b0).sum(dim=(-2, -1))
    den = torch.sqrt((a0.square().sum(dim=(-2, -1)) * b0.square().sum(dim=(-2, -1))).clamp_min(eps))
    return (num / den).mean()


def gradient_xy(image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    dx = image[..., :, 1:] - image[..., :, :-1]
    dy = image[..., 1:, :] - image[..., :-1, :]
    dx = F.pad(dx, (0, 1, 0, 0))
    dy = F.pad(dy, (0, 0, 0, 1))
    return dx, dy


def masked_gradient_loss(a: torch.Tensor, b: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    ax, ay = gradient_xy(a)
    bx, by = gradient_xy(b)
    return masked_mean((ax - bx).abs() + (ay - by).abs(), mask)


def flow_smoothness(flow: torch.Tensor) -> torch.Tensor:
    dx = flow[..., :, 1:] - flow[..., :, :-1]
    dy = flow[..., 1:, :] - flow[..., :-1, :]
    return dx.abs().mean() + dy.abs().mean()


def ssim_loss(a: torch.Tensor, b: torch.Tensor, window: int = 7, eps: float = 1e-6) -> torch.Tensor:
    pad = window // 2
    mu_a = F.avg_pool2d(a, window, stride=1, padding=pad)
    mu_b = F.avg_pool2d(b, window, stride=1, padding=pad)
    sigma_a = F.avg_pool2d(a * a, window, stride=1, padding=pad) - mu_a * mu_a
    sigma_b = F.avg_pool2d(b * b, window, stride=1, padding=pad) - mu_b * mu_b
    sigma_ab = F.avg_pool2d(a * b, window, stride=1, padding=pad) - mu_a * mu_b
    c1 = 0.01**2
    c2 = 0.03**2
    ssim = ((2 * mu_a * mu_b + c1) * (2 * sigma_ab + c2)) / ((mu_a.square() + mu_b.square() + c1) * (sigma_a + sigma_b + c2) + eps)
    return (1.0 - ssim.clamp(0.0, 1.0)).mean()


def fusion_gradient_target(visible: torch.Tensor, thermal: torch.Tensor) -> torch.Tensor:
    vx, vy = gradient_xy(visible)
    tx, ty = gradient_xy(thermal)
    vmag = vx.abs() + vy.abs()
    tmag = tx.abs() + ty.abs()
    return torch.maximum(vmag, tmag)


def image_stats(image: torch.Tensor) -> dict[str, float]:
    return {
        "min": float(image.min().detach().cpu()),
        "max": float(image.max().detach().cpu()),
        "mean": float(image.mean().detach().cpu()),
    }
