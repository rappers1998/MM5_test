from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .io_utils import draw_label, ensure_bgr, fit_image, write_image


def flow_to_rgb(flow: np.ndarray, clip_radius: float = 10.0) -> np.ndarray:
    flow = np.asarray(flow, dtype=np.float32)
    if flow.ndim != 3 or flow.shape[2] != 2:
        raise ValueError(f"Expected HxWx2 flow, got {flow.shape}")
    fx = np.clip(flow[..., 0], -clip_radius, clip_radius)
    fy = np.clip(flow[..., 1], -clip_radius, clip_radius)
    magnitude, angle = cv2.cartToPolar(fx, fy, angleInDegrees=True)
    hsv = np.zeros((*flow.shape[:2], 3), dtype=np.uint8)
    hsv[..., 0] = ((angle / 2.0) % 180).astype(np.uint8)
    hsv[..., 1] = 255
    hsv[..., 2] = np.clip((magnitude / max(clip_radius, 1e-6)) * 255.0, 0.0, 255.0).astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def mask_overlay(base: np.ndarray, mask: np.ndarray, color: tuple[int, int, int]) -> np.ndarray:
    output = ensure_bgr(base)
    mask_bool = mask.astype(bool)
    overlay = output.copy()
    overlay[mask_bool] = np.asarray(color, dtype=np.uint8)
    return cv2.addWeighted(output, 0.7, overlay, 0.3, 0.0)


def make_panel(items: list[tuple[str, np.ndarray]], tile_size: tuple[int, int] = (320, 256), columns: int = 2) -> np.ndarray:
    if not items:
        raise ValueError("No items provided")
    tile_w, tile_h = tile_size
    rows = (len(items) + columns - 1) // columns
    panel = np.full((rows * tile_h, columns * tile_w, 3), 16, dtype=np.uint8)
    for index, (label, image) in enumerate(items):
        row = index // columns
        col = index % columns
        tile = fit_image(ensure_bgr(image), tile_w, tile_h)
        tile = draw_label(tile, label)
        y0 = row * tile_h
        x0 = col * tile_w
        panel[y0 : y0 + tile_h, x0 : x0 + tile_w] = tile
    return panel


def save_panel(path: str | Path, items: list[tuple[str, np.ndarray]], tile_size: tuple[int, int] = (320, 256), columns: int = 2) -> None:
    write_image(path, make_panel(items, tile_size=tile_size, columns=columns))
