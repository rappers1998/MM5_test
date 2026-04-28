from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from mm5_calib_benchmark.common import compute_plane_homography

from .io_utils import to_gray_u8


def scale_intrinsics(k_native: np.ndarray, native_size: tuple[int, int], actual_size: tuple[int, int]) -> np.ndarray:
    native_w, native_h = native_size
    actual_w, actual_h = actual_size
    scale_x = float(actual_w) / float(max(native_w, 1))
    scale_y = float(actual_h) / float(max(native_h, 1))
    scaled = np.asarray(k_native, dtype=np.float64).copy()
    scaled[0, 0] *= scale_x
    scaled[0, 2] *= scale_x
    scaled[1, 1] *= scale_y
    scaled[1, 2] *= scale_y
    return scaled


def thermal_to_rgb_homography(
    row: dict[str, Any],
    runtime_cfg: dict[str, Any],
    rgb_shape: tuple[int, int],
    lwir_shape: tuple[int, int],
    plane_depth_mm: float | None = None,
) -> np.ndarray:
    rgb_h, rgb_w = rgb_shape
    lwir_h, lwir_w = lwir_shape
    rgb_native = tuple(int(v) for v in runtime_cfg.get("calibration_rgb_native_size", [rgb_w, rgb_h]))
    lwir_native = tuple(int(v) for v in runtime_cfg.get("calibration_lwir_native_size", [lwir_w, lwir_h]))
    scaled_rgb = scale_intrinsics(np.asarray(row["calib_intrinsic_rgb"], dtype=np.float64), rgb_native, (rgb_w, rgb_h))
    scaled_lwir = scale_intrinsics(np.asarray(row["calib_intrinsic_lwir"], dtype=np.float64), lwir_native, (lwir_w, lwir_h))
    extrinsic = row["calib_extrinsic_rgb_to_lwir"]
    stereo = {
        "CM1": scaled_rgb,
        "CM2": scaled_lwir,
        "R": np.asarray(extrinsic["R"], dtype=np.float64),
        "T": np.asarray(extrinsic["T"], dtype=np.float64),
    }
    depth_mm = float(plane_depth_mm if plane_depth_mm is not None else runtime_cfg.get("plane_depth_mm", 700.0))
    rgb_to_lwir = compute_plane_homography(stereo, (rgb_w, rgb_h), depth_mm)
    return np.linalg.inv(rgb_to_lwir)


def warp_thermal_to_rgb(
    raw_lwir: np.ndarray,
    thermal_to_rgb: np.ndarray,
    rgb_shape: tuple[int, int],
    border_mode: int = cv2.BORDER_CONSTANT,
) -> tuple[np.ndarray, np.ndarray]:
    rgb_h, rgb_w = rgb_shape
    thermal_u8 = to_gray_u8(raw_lwir)
    projected = cv2.warpPerspective(
        thermal_u8,
        thermal_to_rgb,
        (rgb_w, rgb_h),
        flags=cv2.INTER_LINEAR,
        borderMode=border_mode,
        borderValue=0,
    )
    valid = cv2.warpPerspective(
        np.full(thermal_u8.shape[:2], 255, dtype=np.uint8),
        thermal_to_rgb,
        (rgb_w, rgb_h),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    return projected, (valid > 0).astype(np.uint8)


def normalize_projected_thermal(projected: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    thermal_f = projected.astype(np.float32)
    valid = valid_mask > 0
    if valid.any():
        values = thermal_f[valid]
        lo = float(np.percentile(values, 2.0))
        hi = float(np.percentile(values, 98.0))
        if hi <= lo:
            hi = lo + 1.0
        thermal_f = np.clip((thermal_f - lo) / (hi - lo), 0.0, 1.0) * 255.0
    return cv2.GaussianBlur(thermal_f.astype(np.uint8), (0, 0), 1.8)


def feather_mask(valid_mask: np.ndarray, width_px: float = 32.0) -> np.ndarray:
    distance = cv2.distanceTransform(valid_mask.astype(np.uint8), cv2.DIST_L2, 5)
    return np.clip(distance / max(width_px, 1.0), 0.0, 1.0)


def estimate_depth_mm(depth_image: np.ndarray | None, fallback_mm: float) -> float:
    if depth_image is None:
        return float(fallback_mm)
    depth = np.asarray(depth_image, dtype=np.float32)
    valid = np.isfinite(depth) & (depth > 50.0)
    if int(valid.sum()) < 64:
        return float(fallback_mm)
    value = float(np.median(depth[valid]))
    if value <= 50.0:
        return float(fallback_mm)
    return value


def resize_mask(mask: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    return (cv2.resize(mask.astype(np.uint8), size, interpolation=cv2.INTER_NEAREST) > 0).astype(np.uint8)
