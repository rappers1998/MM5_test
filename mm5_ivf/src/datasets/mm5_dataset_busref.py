from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from ..utils.io_utils import read_image, read_json, to_gray_u8


def _to_tensor(image: np.ndarray, size: tuple[int, int]) -> torch.Tensor:
    gray = to_gray_u8(image)
    if (gray.shape[1], gray.shape[0]) != size:
        gray = cv2.resize(gray, size, interpolation=cv2.INTER_AREA)
    tensor = torch.from_numpy(gray.astype(np.float32) / 255.0)
    return tensor.unsqueeze(0)


def _mask_tensor(mask: np.ndarray, size: tuple[int, int]) -> torch.Tensor:
    gray = to_gray_u8(mask)
    if (gray.shape[1], gray.shape[0]) != size:
        gray = cv2.resize(gray, size, interpolation=cv2.INTER_NEAREST)
    return torch.from_numpy((gray > 0).astype(np.float32)).unsqueeze(0)


def _load_rows(split_path: str | Path, split: str, limit: int = 0) -> list[dict[str, Any]]:
    rows = read_json(split_path)
    rows = [row for row in rows if str(row.get("split", "")) == split]
    if limit > 0:
        rows = rows[:limit]
    return rows


class BusReFReconDataset(Dataset):
    def __init__(self, split_path: str | Path, split: str, image_size: tuple[int, int], limit: int = 0) -> None:
        self.rows = _load_rows(split_path, split, limit)
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        rgb = read_image(row["rgb_canonical_path"], cv2.IMREAD_COLOR)
        lwir = read_image(row["lwir_canonical_path"], cv2.IMREAD_UNCHANGED)
        if rgb is None or lwir is None:
            raise FileNotFoundError(row["rgb_canonical_path"])
        return {
            "visible": _to_tensor(rgb, self.image_size),
            "thermal": _to_tensor(lwir, self.image_size),
            "row": row,
        }


class BusReFRegDataset(Dataset):
    def __init__(self, split_path: str | Path, split: str, image_size: tuple[int, int], limit: int = 0) -> None:
        self.rows = _load_rows(split_path, split, limit)
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        fixed_rgb = read_image(row["fixed_rgb_path"], cv2.IMREAD_COLOR)
        fixed_lwir = read_image(row["fixed_lwir_path"], cv2.IMREAD_UNCHANGED)
        moving = read_image(row["moving_lwir_path"], cv2.IMREAD_UNCHANGED)
        mask = read_image(row["reconstructible_mask_path"], cv2.IMREAD_GRAYSCALE)
        if fixed_rgb is None or fixed_lwir is None or moving is None or mask is None:
            raise FileNotFoundError(row["moving_lwir_path"])
        return {
            "fixed_visible": _to_tensor(fixed_rgb, self.image_size),
            "fixed_thermal": _to_tensor(fixed_lwir, self.image_size),
            "moving_thermal": _to_tensor(moving, self.image_size),
            "mask": _mask_tensor(mask, self.image_size),
            "row": row,
        }


class BusReFFuseDataset(Dataset):
    def __init__(self, split_path: str | Path, split: str, image_size: tuple[int, int], limit: int = 0) -> None:
        self.rows = _load_rows(split_path, split, limit)
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        rgb = read_image(row["rgb_canonical_path"], cv2.IMREAD_COLOR)
        lwir = read_image(row["lwir_canonical_path"], cv2.IMREAD_UNCHANGED)
        mask = read_image(row["overlap_mask_path"], cv2.IMREAD_GRAYSCALE)
        if rgb is None or lwir is None or mask is None:
            raise FileNotFoundError(row["rgb_canonical_path"])
        return {
            "visible": _to_tensor(rgb, self.image_size),
            "thermal": _to_tensor(lwir, self.image_size),
            "mask": _mask_tensor(mask, self.image_size),
            "row": row,
        }
