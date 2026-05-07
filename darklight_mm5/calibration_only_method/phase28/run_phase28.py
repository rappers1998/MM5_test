from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import cv2
import numpy as np

PHASE28_DIR = Path(__file__).resolve().parent
METHOD_DIR = PHASE28_DIR.parent
DARKLIGHT_DIR = METHOD_DIR.parent
for path in (DARKLIGHT_DIR, METHOD_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from diagnose_aligned_canvas import evaluate_candidate  # noqa: E402
from run_calibration_only import CandidateOutput, crop_mask_with_border, crop_with_border, sample_id  # noqa: E402
from run_darklight import (  # noqa: E402
    auto_edges,
    collect_fieldnames,
    direct_alignment_metrics,
    direct_metrics_for_csv,
    enhance_lowlight_bgr,
    gray_u8,
    imread_unicode,
    imwrite_unicode,
    letterbox,
    normalize_u8,
    write_csv,
)
from run_phase22_stereo_recalib import add_metadata_to_metrics, summarize  # noqa: E402
from run_phase25_edge_optimization import make_shift_candidate, prepare_rows  # noqa: E402


PROFILE_IDS = {
    "core": "106,104,103",
    "review": "2,23,103,106,273,291,302",
    "broad": "200,209,248,291,161,187,137,103,23,50,100,272,266,262,296,123,110,120",
}


def json_default(value):
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"not JSON serializable: {type(value)!r}")


def component_filter(
    mask_u8: np.ndarray,
    valid: np.ndarray,
    min_area: int,
    max_area_frac: float,
    max_span_frac: float = 0.45,
) -> np.ndarray:
    height, width = mask_u8.shape
    component_count, labels, stats, _centroids = cv2.connectedComponentsWithStats(mask_u8.astype(np.uint8), 8)
    kept = np.zeros_like(mask_u8, dtype=np.uint8)
    valid_area = max(1, int(np.count_nonzero(valid)))
    max_area = max(min_area * 3, int(valid_area * float(max_area_frac)))
    for idx in range(1, component_count):
        x = int(stats[idx, cv2.CC_STAT_LEFT])
        y = int(stats[idx, cv2.CC_STAT_TOP])
        w = int(stats[idx, cv2.CC_STAT_WIDTH])
        h = int(stats[idx, cv2.CC_STAT_HEIGHT])
        area = int(stats[idx, cv2.CC_STAT_AREA])
        touches_border = x <= 2 or y <= 2 or x + w >= width - 2 or y + h >= height - 2
        spans_too_much = w > int(width * float(max_span_frac)) or h > int(height * float(max_span_frac))
        if min_area <= area <= max_area and not touches_border and not spans_too_much:
            kept[labels == idx] = 1
    return kept.astype(bool)


def rgb_object_mask(rgb_bgr: np.ndarray, valid: np.ndarray) -> np.ndarray:
    enhanced = enhance_lowlight_bgr(rgb_bgr)
    hsv = cv2.cvtColor(enhanced, cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]
    gray = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)
    valid = valid.astype(bool)
    vals = val[valid]
    sats = sat[valid]
    if vals.size == 0:
        return np.zeros(rgb_bgr.shape[:2], dtype=bool)

    bright = val >= max(35.0, float(np.percentile(vals, 70.0)))
    colorful = (sat >= max(35.0, float(np.percentile(sats, 60.0)))) & (val >= max(25.0, float(np.percentile(vals, 35.0))))
    edges = cv2.dilate((auto_edges(gray) > 0).astype(np.uint8), np.ones((5, 5), np.uint8), iterations=1) > 0
    mask = (bright | colorful | edges) & valid
    mask_u8 = mask.astype(np.uint8)
    mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)
    mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8), iterations=1)
    return component_filter(mask_u8, valid, min_area=60, max_area_frac=0.08)


def depth_support_mask(
    depth: np.ndarray | None,
    rgb_offset: tuple[int, int],
    target_size: tuple[int, int],
    valid: np.ndarray,
) -> np.ndarray:
    if depth is None:
        return np.zeros(valid.shape[:2], dtype=bool)
    depth_crop = crop_with_border(depth.astype(np.float32), rgb_offset, target_size)
    depth_valid = crop_mask_with_border(np.isfinite(depth) & (depth > 0), rgb_offset, target_size)
    support_valid = depth_valid & valid.astype(bool)
    vals = depth_crop[support_valid]
    vals = vals[np.isfinite(vals) & (vals > 0)]
    if vals.size < 300:
        return np.zeros(valid.shape[:2], dtype=bool)
    near_threshold = float(np.percentile(vals, 35.0))
    mask = support_valid & (depth_crop > 0) & (depth_crop <= near_threshold)
    mask_u8 = mask.astype(np.uint8)
    mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8), iterations=1)
    mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, np.ones((11, 11), np.uint8), iterations=1)
    return component_filter(mask_u8, support_valid, min_area=120, max_area_frac=0.25)


def annotation_support_mask(
    annotation_path: str,
    rgb_offset: tuple[int, int],
    target_size: tuple[int, int],
    valid: np.ndarray,
) -> np.ndarray:
    if not annotation_path:
        return np.zeros(valid.shape[:2], dtype=bool)
    path = Path(annotation_path)
    if not path.exists():
        return np.zeros(valid.shape[:2], dtype=bool)
    mask = imread_unicode(path, cv2.IMREAD_GRAYSCALE) > 0
    mask_crop = crop_mask_with_border(mask, rgb_offset, target_size) & valid.astype(bool)
    mask_u8 = cv2.morphologyEx(mask_crop.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)
    mask_u8 = cv2.dilate(mask_u8, np.ones((9, 9), np.uint8), iterations=1)
    return component_filter(mask_u8, valid, min_area=50, max_area_frac=0.12, max_span_frac=0.35)


def review_support_mask(
    rgb_bgr: np.ndarray,
    valid: np.ndarray,
    depth: np.ndarray | None = None,
    rgb_offset: tuple[int, int] = (0, 0),
    annotation_path: str = "",
) -> np.ndarray:
    target_size = (rgb_bgr.shape[1], rgb_bgr.shape[0])
    annotation_mask = annotation_support_mask(annotation_path, rgb_offset, target_size, valid)
    if int(np.count_nonzero(annotation_mask)) >= 20:
        return annotation_mask & valid.astype(bool)
    rgb_mask = rgb_object_mask(rgb_bgr, valid)
    if int(np.count_nonzero(rgb_mask)) >= 80:
        return rgb_mask & valid.astype(bool)
    depth_mask = depth_support_mask(depth, rgb_offset, target_size, valid)
    if int(np.count_nonzero(depth_mask)) >= 80:
        return depth_mask & valid.astype(bool)
    return np.zeros(valid.shape[:2], dtype=bool)


def thermal_foreground_mask(lwir_u8: np.ndarray, valid: np.ndarray, support: np.ndarray | None = None) -> np.ndarray:
    valid = valid.astype(bool)
    if not np.any(valid):
        return np.zeros(lwir_u8.shape[:2], dtype=bool)
    if support is None:
        support = valid
    else:
        support = support.astype(bool) & valid

    lwir_norm = normalize_u8(lwir_u8).astype(np.float32)
    vals = lwir_norm[support]
    if vals.size < 100:
        vals = lwir_norm[valid]
    median = float(np.median(vals))
    deviations = np.abs(vals.astype(np.float32) - median)
    contrast_threshold = max(8.0, float(np.percentile(deviations, 70.0)))
    mask = (np.abs(lwir_norm - median) >= contrast_threshold) & support
    mask_u8 = mask.astype(np.uint8)
    mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)
    mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8), iterations=1)
    kept = component_filter(mask_u8, support, min_area=40, max_area_frac=0.18)
    return kept if np.any(kept) else (mask_u8 > 0)


def fragmented_mask(mask: np.ndarray) -> bool:
    component_count, _labels, stats, _centroids = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    areas = [int(stats[idx, cv2.CC_STAT_AREA]) for idx in range(1, component_count)]
    total = int(sum(areas))
    largest = max(areas, default=0)
    if total == 0:
        return True
    return len(areas) >= 8 and largest < 900 and largest / max(total, 1) < 0.45


def review_thermal_target_mask(lwir_u8: np.ndarray, valid: np.ndarray, support: np.ndarray) -> np.ndarray:
    mask = thermal_foreground_mask(lwir_u8, valid, support)
    if fragmented_mask(mask):
        fallback = thermal_foreground_mask(lwir_u8, valid, valid)
        if int(np.count_nonzero(fallback)) >= max(500, int(np.count_nonzero(mask)) * 2):
            return fallback
    return mask


def thermal_blend(rgb_bgr: np.ndarray, lwir_u8: np.ndarray, valid: np.ndarray, support: np.ndarray, alpha: float = 0.22) -> np.ndarray:
    heat = cv2.applyColorMap(normalize_u8(lwir_u8), cv2.COLORMAP_TURBO)
    mask = review_thermal_target_mask(lwir_u8, valid, support)
    if not np.any(mask):
        return rgb_bgr.copy()
    feather = cv2.GaussianBlur(mask.astype(np.float32), (0, 0), 2.0)
    feather = np.clip(feather, 0.0, 1.0)[:, :, None]
    alpha_map = feather * float(alpha)
    out = rgb_bgr.astype(np.float32) * (1.0 - alpha_map) + heat.astype(np.float32) * alpha_map
    return np.clip(out, 0, 255).astype(np.uint8)


def simple_review_fusion(rgb_bgr: np.ndarray, lwir_u8: np.ndarray, valid: np.ndarray, support: np.ndarray, alpha: float = 0.18) -> np.ndarray:
    base = enhance_lowlight_bgr(rgb_bgr).astype(np.float32)
    heat = cv2.applyColorMap(normalize_u8(lwir_u8), cv2.COLORMAP_TURBO).astype(np.float32)
    target_mask = review_thermal_target_mask(lwir_u8, valid, support)
    if np.any(target_mask):
        alpha_map = cv2.GaussianBlur(target_mask.astype(np.float32), (0, 0), 1.5)
        alpha_map = np.clip(alpha_map, 0.0, 1.0)[:, :, None] * float(alpha)
        base = base * (1.0 - alpha_map) + heat * alpha_map

    if np.any(target_mask):
        edge_support = cv2.dilate(target_mask.astype(np.uint8), np.ones((17, 17), np.uint8), iterations=1) > 0
    else:
        edge_support = np.zeros(valid.shape[:2], dtype=bool)
    lwir_edges = (auto_edges(lwir_u8) > 0) & valid.astype(bool) & edge_support
    lwir_edges = cv2.dilate(lwir_edges.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=1) > 0

    out = np.clip(base, 0, 255).astype(np.uint8)
    edge_color = np.array([255, 210, 0], dtype=np.uint8)
    out[lwir_edges] = ((out[lwir_edges].astype(np.float32) * 0.35) + (edge_color.astype(np.float32) * 0.65)).astype(np.uint8)
    return out


def stabilize_lwir(lwir_u8: np.ndarray, kernel_px: int) -> np.ndarray:
    kernel_px = max(3, int(kernel_px))
    if kernel_px % 2 == 0:
        kernel_px += 1
    kernel = np.ones((kernel_px, kernel_px), np.uint8)
    return cv2.morphologyEx(lwir_u8, cv2.MORPH_CLOSE, kernel)


def foreground_crop(mask: np.ndarray, margin: int = 32) -> tuple[int, int, int, int] | None:
    ys, xs = np.where(mask.astype(bool))
    if xs.size == 0 or ys.size == 0:
        return None
    height, width = mask.shape[:2]
    return (
        max(0, int(xs.min()) - margin),
        max(0, int(ys.min()) - margin),
        min(width, int(xs.max()) + 1 + margin),
        min(height, int(ys.max()) + 1 + margin),
    )


def metadata_offset(metadata: dict, key: str, default: tuple[int, int] = (0, 0)) -> tuple[int, int]:
    value = metadata.get(key, default)
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        return int(round(float(value[0]))), int(round(float(value[1])))
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            if isinstance(parsed, (list, tuple)) and len(parsed) >= 2:
                return int(round(float(parsed[0]))), int(round(float(parsed[1])))
        except json.JSONDecodeError:
            pass
    return default


def prefixed_direct_metrics(prefix: str, metrics: dict) -> dict:
    return {f"{prefix}_{key}": value for key, value in direct_metrics_for_csv(metrics).items()}


def target_eval_mask(support: np.ndarray, valid: np.ndarray) -> tuple[np.ndarray, bool]:
    valid = valid.astype(bool)
    if not np.any(support):
        return valid, True
    mask = cv2.dilate(support.astype(np.uint8), np.ones((21, 21), np.uint8), iterations=1).astype(bool) & valid
    if int(np.count_nonzero(mask)) < 1000:
        return valid, True
    return mask, False


def add_target_metrics(item: dict, candidate: CandidateOutput, metrics: dict, support: np.ndarray) -> None:
    valid = candidate.rgb_valid.astype(bool) & candidate.lwir_valid.astype(bool)
    roi, fallback_valid = target_eval_mask(support, valid)
    rgb_metrics = direct_alignment_metrics(gray_u8(item["aligned_rgb"]), gray_u8(candidate.rgb), candidate.rgb_valid, roi)
    lwir_metrics = direct_alignment_metrics(item["aligned_lwir_u8"], candidate.lwir, candidate.lwir_valid, roi)
    cross_metrics = direct_alignment_metrics(gray_u8(candidate.rgb), candidate.lwir, valid, roi)
    metrics.update(prefixed_direct_metrics("eval_target_rgb_to_mm5_aligned_rgb", rgb_metrics))
    metrics.update(prefixed_direct_metrics("eval_target_lwir_to_mm5_aligned_t16", lwir_metrics))
    metrics.update(prefixed_direct_metrics("eval_target_cross_rgb_lwir", cross_metrics))
    metrics["target_support_pixels"] = int(np.count_nonzero(support))
    metrics["target_eval_pixels"] = int(np.count_nonzero(roi))
    metrics["target_eval_fallback_valid"] = bool(fallback_valid)


def make_phase28_candidate(item: dict, args) -> CandidateOutput:
    base = make_shift_candidate(
        item["phase24"],
        item["depth_projection"],
        args.dx_px,
        args.dy_px,
        fill_depth_border=not args.no_depth_border_fill,
    )
    lwir = stabilize_lwir(base.lwir, args.stabilization_kernel_px)
    metadata = dict(base.metadata)
    metadata.update(
        {
            "method": "phase28_calibrated_depth_acceptance",
            "candidate": "phase28_calibrated_depth_acceptance",
            "allowed_for_generation": True,
            "phase28_class": "calibration_depth_acceptance",
            "phase28_profile": args.current_profile,
            "lwir_source": "phase24_affine_lwir_subpixel_shift_depth_fill_then_morph_close",
            "depth_registration_dx_px": float(args.dx_px),
            "depth_registration_dy_px": float(args.dy_px),
            "stabilization_kernel_px": int(args.stabilization_kernel_px),
            "fill_depth_border": not bool(args.no_depth_border_fill),
            "uses_aligned_for_generation": False,
            "phase28_visual_mode": str(args.visual_mode),
            "anti_ghost_feather_px": float(args.anti_ghost_feather_px),
            "anti_ghost_erode_px": int(args.anti_ghost_erode_px),
            "max_support_area_frac": float(args.max_support_area_frac),
            "calibration_file": str(args.calibration),
            "thermal_camera_calibration_file": str(args.thermal_camera_calibration),
            "target_size": str(args.target_size),
            "rule_source": (
                "Phase28 generation uses calibration files, raw RGB/LWIR/depth inputs, "
                "Phase24 board-affine geometry, Phase25 depth projection support, a fixed "
                "subpixel residual shift, and LWIR morphology stabilization. MM5 aligned "
                "RGB/T16 are read only after generation for metrics and acceptance panels."
            ),
        }
    )
    return CandidateOutput(
        name="phase28_calibrated_depth_acceptance",
        rgb=base.rgb,
        lwir=lwir,
        rgb_valid=base.rgb_valid,
        lwir_valid=base.lwir_valid,
        metadata=metadata,
    )


def add_phase28_metadata(metrics: dict, candidate: CandidateOutput) -> dict:
    out = add_metadata_to_metrics(metrics, candidate)
    for key in (
        "allowed_for_generation",
        "phase28_class",
        "phase28_profile",
        "depth_registration_dx_px",
        "depth_registration_dy_px",
        "stabilization_kernel_px",
        "fill_depth_border",
        "uses_aligned_for_generation",
        "phase28_visual_mode",
        "anti_ghost_feather_px",
        "anti_ghost_erode_px",
        "max_support_area_frac",
        "depth_fill_pixels",
        "depth_fill_ratio",
        "depth_project_valid_ratio",
        "depth_valid_policy",
        "calibration_file",
        "thermal_camera_calibration_file",
        "target_size",
    ):
        if key in candidate.metadata:
            out[key] = candidate.metadata[key]
    return out


def make_six_panel(panels: list[tuple[np.ndarray, str]], out_path: Path, tile_size=(330, 245)) -> None:
    make_panel_grid(panels, out_path, tile_size=tile_size, columns=3)


def make_panel_grid(panels: list[tuple[np.ndarray, str]], out_path: Path, tile_size=(330, 245), columns: int = 3) -> None:
    tiles = [letterbox(img, tile_size, label) for img, label in panels]
    if not tiles:
        return
    columns = max(1, int(columns))
    blank = np.zeros_like(tiles[0])
    rows = []
    for start in range(0, len(tiles), columns):
        row_tiles = tiles[start : start + columns]
        while len(row_tiles) < columns:
            row_tiles.append(blank.copy())
        rows.append(np.concatenate(row_tiles, axis=1))
    imwrite_unicode(out_path, np.concatenate(rows, axis=0))


def crop_depth_to_canvas(item: dict, rgb_offset: tuple[int, int], target_size: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    depth = item.get("depth")
    if depth is None:
        return np.zeros((target_size[1], target_size[0]), dtype=np.float32), np.zeros((target_size[1], target_size[0]), dtype=bool)
    depth_f = depth.astype(np.float32)
    crop = crop_with_border(depth_f, rgb_offset, target_size)
    valid = crop_mask_with_border(np.isfinite(depth_f) & (depth_f > 0), rgb_offset, target_size)
    return crop, valid


def depth_support_view(depth_crop: np.ndarray, depth_valid: np.ndarray, support: np.ndarray, valid: np.ndarray) -> np.ndarray:
    out = np.zeros((*depth_crop.shape[:2], 3), dtype=np.uint8)
    vals = depth_crop[depth_valid.astype(bool)]
    vals = vals[np.isfinite(vals) & (vals > 0)]
    if vals.size:
        lo = float(np.percentile(vals, 2.0))
        hi = float(np.percentile(vals, 98.0))
        if hi <= lo:
            hi = lo + 1.0
        norm = np.clip((depth_crop.astype(np.float32) - lo) / (hi - lo), 0.0, 1.0)
        out = cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_VIRIDIS)
        out[~depth_valid.astype(bool)] = 0
    support_edge = cv2.morphologyEx(support.astype(np.uint8), cv2.MORPH_GRADIENT, np.ones((5, 5), np.uint8)) > 0
    valid_edge = cv2.morphologyEx(valid.astype(np.uint8), cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)) > 0
    out[valid_edge] = (180, 180, 180)
    out[support.astype(bool)] = (out[support.astype(bool)].astype(np.float32) * 0.45 + np.array([60, 255, 60]) * 0.55).astype(np.uint8)
    out[support_edge] = (255, 255, 255)
    return out


def filter_visual_components(
    mask: np.ndarray,
    valid: np.ndarray,
    min_area: int = 60,
    max_area_frac: float = 0.10,
    max_span_frac: float = 0.42,
) -> tuple[np.ndarray, np.ndarray]:
    height, width = mask.shape[:2]
    valid = valid.astype(bool)
    raw = mask.astype(bool) & valid
    kept = np.zeros(raw.shape, dtype=np.uint8)
    rejected = raw.copy()
    component_count, labels, stats, _centroids = cv2.connectedComponentsWithStats(raw.astype(np.uint8), 8)
    valid_area = max(1, int(np.count_nonzero(valid)))
    max_area = max(min_area * 3, int(valid_area * float(max_area_frac)))
    for idx in range(1, component_count):
        x = int(stats[idx, cv2.CC_STAT_LEFT])
        y = int(stats[idx, cv2.CC_STAT_TOP])
        w = int(stats[idx, cv2.CC_STAT_WIDTH])
        h = int(stats[idx, cv2.CC_STAT_HEIGHT])
        area = int(stats[idx, cv2.CC_STAT_AREA])
        touches_border = x <= 2 or y <= 2 or x + w >= width - 2 or y + h >= height - 2
        spans_too_much = w > int(width * float(max_span_frac)) or h > int(height * float(max_span_frac))
        if min_area <= area <= max_area and not touches_border and not spans_too_much:
            kept[labels == idx] = 1
    kept_bool = kept.astype(bool)
    rejected &= ~kept_bool
    return kept_bool, rejected


def prune_remote_visual_components(mask: np.ndarray, valid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mask = mask.astype(bool) & valid.astype(bool)
    component_count, labels, stats, centroids = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    if component_count <= 2:
        return mask, np.zeros_like(mask, dtype=bool)

    areas = [int(stats[idx, cv2.CC_STAT_AREA]) for idx in range(1, component_count)]
    if not areas:
        return mask, np.zeros_like(mask, dtype=bool)
    largest_area = max(areas)
    largest_idx = 1 + areas.index(largest_area)
    height, width = mask.shape[:2]
    diag = float(np.hypot(width, height))
    largest_cx, largest_cy = centroids[largest_idx]
    kept = np.zeros_like(mask, dtype=np.uint8)
    for idx in range(1, component_count):
        area = int(stats[idx, cv2.CC_STAT_AREA])
        cx, cy = centroids[idx]
        distance = float(np.hypot(float(cx - largest_cx), float(cy - largest_cy)))
        area_ratio = area / max(1, largest_area)
        near_main_cluster = distance <= diag * 0.23 and area >= 40
        substantial_target = area_ratio >= 0.18
        if idx == largest_idx or substantial_target or near_main_cluster:
            kept[labels == idx] = 1
    kept_bool = kept.astype(bool)
    rejected = mask & ~kept_bool
    return kept_bool, rejected


def depth_consensus_mask(
    depth_crop: np.ndarray,
    depth_valid: np.ndarray,
    valid: np.ndarray,
    support: np.ndarray,
) -> np.ndarray:
    valid_depth = depth_valid.astype(bool) & valid.astype(bool) & np.isfinite(depth_crop) & (depth_crop > 0)
    if int(np.count_nonzero(valid_depth)) < 300:
        return np.zeros(valid.shape[:2], dtype=bool)

    seed = support.astype(bool) & valid_depth
    all_vals = depth_crop[valid_depth].astype(np.float32)
    if int(np.count_nonzero(seed)) >= 60:
        seed_vals = depth_crop[seed].astype(np.float32)
        lo = float(np.percentile(seed_vals, 5.0))
        hi = float(np.percentile(seed_vals, 95.0))
        global_span = max(1e-6, float(np.percentile(all_vals, 98.0) - np.percentile(all_vals, 2.0)))
        pad = max((hi - lo) * 0.45, global_span * 0.03, 1e-6)
        mask = valid_depth & (depth_crop >= lo - pad) & (depth_crop <= hi + pad)
    else:
        near_threshold = float(np.percentile(all_vals, 35.0))
        mask = valid_depth & (depth_crop <= near_threshold)

    mask_u8 = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, np.ones((5, 5), np.uint8), iterations=1)
    mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, np.ones((11, 11), np.uint8), iterations=1)
    return mask_u8.astype(bool) & valid.astype(bool)


def make_anti_ghost_support(
    rgb_bgr: np.ndarray,
    lwir_u8: np.ndarray,
    valid: np.ndarray,
    support: np.ndarray,
    depth_crop: np.ndarray,
    depth_valid: np.ndarray,
    args,
) -> dict:
    valid = valid.astype(bool)
    support = support.astype(bool) & valid
    thermal = review_thermal_target_mask(lwir_u8, valid, support if np.any(support) else valid) & valid
    depth_mask = depth_consensus_mask(depth_crop, depth_valid, valid, support | thermal)

    support_band = cv2.dilate(support.astype(np.uint8), np.ones((9, 9), np.uint8), iterations=1).astype(bool)
    thermal_band = cv2.dilate(thermal.astype(np.uint8), np.ones((7, 7), np.uint8), iterations=1).astype(bool)
    consensus = (support & thermal) | (support_band & thermal) | (support & thermal_band)
    if int(np.count_nonzero(depth_mask)) >= 80:
        depth_band = cv2.dilate(depth_mask.astype(np.uint8), np.ones((9, 9), np.uint8), iterations=1).astype(bool)
        candidate = (consensus | (support & depth_band) | (thermal & depth_band)) & depth_band
    else:
        candidate = consensus | (support & thermal_band) | (thermal & support_band)
    if int(np.count_nonzero(candidate)) < 80:
        candidate = support | (thermal & support_band)

    candidate &= valid
    candidate_u8 = cv2.morphologyEx(candidate.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)
    candidate_u8 = cv2.morphologyEx(candidate_u8, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8), iterations=1)
    cleaned, rejected = filter_visual_components(
        candidate_u8.astype(bool),
        valid,
        min_area=60,
        max_area_frac=float(args.max_support_area_frac),
        max_span_frac=0.42,
    )

    if int(np.count_nonzero(cleaned)) < 80 and int(np.count_nonzero(support)) >= 80:
        cleaned, rejected = filter_visual_components(
            support,
            valid,
            min_area=60,
            max_area_frac=float(args.max_support_area_frac),
            max_span_frac=0.42,
        )

    erode_px = max(0, int(args.anti_ghost_erode_px))
    if erode_px > 0 and np.any(cleaned):
        kernel = np.ones((erode_px * 2 + 1, erode_px * 2 + 1), np.uint8)
        eroded = cv2.erode(cleaned.astype(np.uint8), kernel, iterations=1).astype(bool)
        if int(np.count_nonzero(eroded)) >= max(40, int(np.count_nonzero(cleaned) * 0.35)):
            rejected |= cleaned & ~eroded
            cleaned = eroded

    pruned, remote_rejected = prune_remote_visual_components(cleaned, valid)
    if int(np.count_nonzero(pruned)) >= max(40, int(np.count_nonzero(cleaned) * 0.50)):
        rejected |= remote_rejected
        cleaned = pruned

    raw = candidate_u8.astype(bool) & valid
    rejected |= raw & ~cleaned
    return {
        "mask": cleaned.astype(bool),
        "raw_mask": raw,
        "thermal_mask": thermal.astype(bool),
        "depth_mask": depth_mask.astype(bool),
        "rejected_mask": rejected.astype(bool),
    }


def feather_alpha(mask: np.ndarray, valid: np.ndarray, feather_px: float) -> np.ndarray:
    mask = mask.astype(bool) & valid.astype(bool)
    if not np.any(mask):
        return np.zeros(mask.shape, dtype=np.float32)
    sigma = max(0.5, float(feather_px))
    alpha = cv2.GaussianBlur(mask.astype(np.float32), (0, 0), sigmaX=sigma, sigmaY=sigma)
    max_value = float(alpha.max())
    if max_value > 1e-6:
        alpha /= max_value
    alpha *= valid.astype(np.float32)
    return np.clip(alpha, 0.0, 1.0).astype(np.float32)


def alpha_mask_view(alpha: np.ndarray, mask: np.ndarray, rejected: np.ndarray) -> np.ndarray:
    alpha_u8 = np.clip(alpha * 255.0, 0, 255).astype(np.uint8)
    out = cv2.applyColorMap(alpha_u8, cv2.COLORMAP_VIRIDIS)
    out[alpha_u8 == 0] = (0, 0, 0)
    rejected_band = cv2.dilate(rejected.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=1).astype(bool)
    boundary = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)).astype(bool)
    out[rejected_band] = (0, 0, 180)
    out[boundary] = (255, 255, 255)
    return out


def clean_registered_lwir_view(lwir_u8: np.ndarray, valid: np.ndarray, focus_mask: np.ndarray) -> np.ndarray:
    lwir_norm = normalize_u8(lwir_u8)
    valid = valid.astype(bool)
    focus = focus_mask.astype(bool) & valid
    out = (lwir_norm.astype(np.float32) * 0.18).astype(np.uint8)
    out[~valid] = 0
    if np.any(focus):
        focus_band = cv2.dilate(focus.astype(np.uint8), np.ones((7, 7), np.uint8), iterations=1).astype(bool) & valid
        vals = lwir_norm[focus_band]
        lo = float(np.percentile(vals, 1.0))
        hi = float(np.percentile(vals, 99.0))
        if hi <= lo:
            hi = lo + 1.0
        local = np.clip((lwir_norm.astype(np.float32) - lo) / (hi - lo), 0.0, 1.0)
        out[focus_band] = (local[focus_band] * 255.0).astype(np.uint8)
        boundary = cv2.morphologyEx(focus.astype(np.uint8), cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)).astype(bool)
    else:
        boundary = np.zeros_like(valid, dtype=bool)
    out_bgr = cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)
    out_bgr[boundary] = (255, 255, 255)
    return out_bgr


def depth_discontinuity_edges(depth_crop: np.ndarray, depth_valid: np.ndarray, valid: np.ndarray) -> np.ndarray:
    valid_depth = depth_valid.astype(bool) & valid.astype(bool) & np.isfinite(depth_crop) & (depth_crop > 0)
    if int(np.count_nonzero(valid_depth)) < 300:
        return np.zeros(valid.shape[:2], dtype=bool)
    vals = depth_crop[valid_depth].astype(np.float32)
    lo = float(np.percentile(vals, 2.0))
    hi = float(np.percentile(vals, 98.0))
    if hi <= lo:
        hi = lo + 1.0
    norm = np.zeros(depth_crop.shape[:2], dtype=np.float32)
    norm[valid_depth] = np.clip((depth_crop[valid_depth].astype(np.float32) - lo) / (hi - lo), 0.0, 1.0)
    grad = cv2.morphologyEx((norm * 255.0).astype(np.uint8), cv2.MORPH_GRADIENT, np.ones((5, 5), np.uint8))
    grad_vals = grad[valid_depth]
    if grad_vals.size == 0:
        return np.zeros(valid.shape[:2], dtype=bool)
    threshold = max(8.0, float(np.percentile(grad_vals, 88.0)))
    edges = (grad >= threshold) & valid_depth
    return cv2.morphologyEx(edges.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1).astype(bool)


def diagnostic_edge_masks(
    rgb_bgr: np.ndarray,
    lwir_u8: np.ndarray,
    valid: np.ndarray,
    focus_mask: np.ndarray,
    depth_crop: np.ndarray,
    depth_valid: np.ndarray,
) -> dict:
    valid = valid.astype(bool)
    if np.any(focus_mask):
        roi = cv2.dilate(focus_mask.astype(np.uint8), np.ones((31, 31), np.uint8), iterations=1).astype(bool) & valid
    else:
        roi = valid
    enhanced = enhance_lowlight_bgr(rgb_bgr)
    rgb_edges = (auto_edges(cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)) > 0) & roi
    lwir_edges = (auto_edges(lwir_u8) > 0) & roi
    depth_edges = depth_discontinuity_edges(depth_crop, depth_valid, valid) & roi
    return {
        "roi": roi,
        "rgb_edges": rgb_edges,
        "lwir_edges": lwir_edges,
        "depth_edges": depth_edges,
    }


def contour_alignment_overlay(
    rgb_bgr: np.ndarray,
    lwir_u8: np.ndarray,
    valid: np.ndarray,
    focus_mask: np.ndarray,
    depth_crop: np.ndarray,
    depth_valid: np.ndarray,
) -> np.ndarray:
    masks = diagnostic_edge_masks(rgb_bgr, lwir_u8, valid, focus_mask, depth_crop, depth_valid)
    out = (enhance_lowlight_bgr(rgb_bgr).astype(np.float32) * 0.78).astype(np.uint8)
    rgb_edges = cv2.dilate(masks["rgb_edges"].astype(np.uint8), np.ones((2, 2), np.uint8), iterations=1).astype(bool)
    lwir_edges = cv2.dilate(masks["lwir_edges"].astype(np.uint8), np.ones((2, 2), np.uint8), iterations=1).astype(bool)
    depth_edges = cv2.dilate(masks["depth_edges"].astype(np.uint8), np.ones((2, 2), np.uint8), iterations=1).astype(bool)
    out[depth_edges] = (60, 255, 60)
    out[rgb_edges] = (255, 255, 0)
    out[lwir_edges] = (0, 40, 255)
    out[rgb_edges & lwir_edges] = (255, 255, 255)
    return out


def anti_ghost_fusion(
    rgb_bgr: np.ndarray,
    lwir_u8: np.ndarray,
    valid: np.ndarray,
    focus_mask: np.ndarray,
    alpha: np.ndarray,
) -> np.ndarray:
    base = enhance_lowlight_bgr(rgb_bgr).astype(np.float32)
    heat = cv2.applyColorMap(normalize_u8(lwir_u8), cv2.COLORMAP_TURBO).astype(np.float32)
    alpha_map = np.clip(alpha, 0.0, 1.0)[:, :, None] * 0.20
    out = base * (1.0 - alpha_map) + heat * alpha_map
    edge_support = cv2.dilate(focus_mask.astype(np.uint8), np.ones((11, 11), np.uint8), iterations=1).astype(bool)
    lwir_edges = (auto_edges(lwir_u8) > 0) & valid.astype(bool) & edge_support
    lwir_edges = cv2.dilate(lwir_edges.astype(np.uint8), np.ones((2, 2), np.uint8), iterations=1).astype(bool)
    out_u8 = np.clip(out, 0, 255).astype(np.uint8)
    edge_color = np.array([0, 210, 255], dtype=np.uint8)
    out_u8[lwir_edges] = ((out_u8[lwir_edges].astype(np.float32) * 0.45) + (edge_color.astype(np.float32) * 0.55)).astype(np.uint8)
    return out_u8


def tear_ghost_diagnostics(
    rgb_bgr: np.ndarray,
    lwir_u8: np.ndarray,
    valid: np.ndarray,
    focus_mask: np.ndarray,
    alpha: np.ndarray,
    depth_crop: np.ndarray,
    depth_valid: np.ndarray,
) -> tuple[np.ndarray, dict]:
    masks = diagnostic_edge_masks(rgb_bgr, lwir_u8, valid, focus_mask, depth_crop, depth_valid)
    rgb_edges = masks["rgb_edges"]
    lwir_edges = masks["lwir_edges"]
    depth_edges = masks["depth_edges"]
    if np.any(rgb_edges):
        dt_rgb = cv2.distanceTransform((~rgb_edges).astype(np.uint8) * 255, cv2.DIST_L2, 3)
    else:
        dt_rgb = np.full(rgb_edges.shape, 99.0, dtype=np.float32)
    if np.any(lwir_edges):
        dt_lwir = cv2.distanceTransform((~lwir_edges).astype(np.uint8) * 255, cv2.DIST_L2, 3)
    else:
        dt_lwir = np.full(lwir_edges.shape, 99.0, dtype=np.float32)

    lwir_mismatch = lwir_edges & (dt_rgb > 2.0)
    rgb_mismatch = rgb_edges & (dt_lwir > 2.0)
    ghost_edges = lwir_mismatch | rgb_mismatch
    alpha_boundary = cv2.morphologyEx((alpha > 0.08).astype(np.uint8), cv2.MORPH_GRADIENT, np.ones((5, 5), np.uint8)).astype(bool)
    mismatch_band = cv2.dilate(ghost_edges.astype(np.uint8), np.ones((5, 5), np.uint8), iterations=1).astype(bool)
    depth_band = cv2.dilate(depth_edges.astype(np.uint8), np.ones((5, 5), np.uint8), iterations=1).astype(bool)
    tear_risk = alpha_boundary & (mismatch_band | depth_band)

    base_gray = cv2.cvtColor(enhance_lowlight_bgr(rgb_bgr), cv2.COLOR_BGR2GRAY)
    out = cv2.cvtColor((base_gray.astype(np.float32) * 0.42).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    out[depth_edges] = (60, 255, 60)
    out[ghost_edges] = (255, 0, 255)
    out[alpha_boundary] = (255, 255, 255)
    out[tear_risk] = (0, 0, 255)

    edge_count = max(1, int(np.count_nonzero(rgb_edges | lwir_edges)))
    boundary_count = max(1, int(np.count_nonzero(alpha_boundary)))
    metrics = {
        "tear_risk_pixels": int(np.count_nonzero(tear_risk)),
        "ghost_risk_pixels": int(np.count_nonzero(ghost_edges)),
        "ghost_edge_ratio": float(np.count_nonzero(ghost_edges) / edge_count),
        "alpha_edge_overlap": float(np.count_nonzero(alpha_boundary & (rgb_edges | lwir_edges)) / boundary_count),
        "depth_discontinuity_pixels": int(np.count_nonzero(depth_edges)),
    }
    return out, metrics


def edge_error_heatmap(reference_lwir_u8: np.ndarray, candidate_lwir_u8: np.ndarray, valid: np.ndarray, max_error_px: float = 8.0) -> np.ndarray:
    fixed_edges = auto_edges(reference_lwir_u8) > 0
    moving_edges = (auto_edges(candidate_lwir_u8) > 0) & valid.astype(bool)
    dt = cv2.distanceTransform((~fixed_edges).astype(np.uint8) * 255, cv2.DIST_L2, 3)
    heat_values = np.clip(dt / float(max_error_px), 0.0, 1.0)
    heat = cv2.applyColorMap((heat_values * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    base = cv2.cvtColor(normalize_u8(reference_lwir_u8), cv2.COLOR_GRAY2BGR)
    edge_band = cv2.dilate(moving_edges.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=1) > 0
    out = base.copy()
    out[edge_band] = (base[edge_band].astype(np.float32) * 0.30 + heat[edge_band].astype(np.float32) * 0.70).astype(np.uint8)
    out[fixed_edges] = (255, 220, 0)
    out[moving_edges] = (0, 40, 255)
    out[fixed_edges & moving_edges] = (255, 255, 255)
    return out


def target_edge_overlay(rgb_bgr: np.ndarray, lwir_u8: np.ndarray, valid: np.ndarray, support: np.ndarray) -> np.ndarray:
    out = enhance_lowlight_bgr(rgb_bgr)
    gray_rgb = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)
    rgb_edges = auto_edges(gray_rgb) > 0
    lwir_edges = (auto_edges(lwir_u8) > 0) & valid.astype(bool)
    if int(np.count_nonzero(support)) >= 80:
        edge_roi = cv2.dilate(support.astype(np.uint8), np.ones((25, 25), np.uint8), iterations=1) > 0
        rgb_edges &= edge_roi
        lwir_edges &= edge_roi
    out[rgb_edges] = (255, 210, 0)
    out[lwir_edges] = (0, 40, 255)
    out[rgb_edges & lwir_edges] = (255, 255, 255)
    return out


def save_phase28_outputs(output_dir: Path, item: dict, candidate: CandidateOutput, metrics: dict, args) -> None:
    row = item["row"]
    sample_tag = f"s{int(row.aligned_id):03d}"
    valid = candidate.rgb_valid.astype(bool) & candidate.lwir_valid.astype(bool)
    rgb_offset = metadata_offset(candidate.metadata, "rgb_crop_offset_xy")
    target_size = (candidate.rgb.shape[1], candidate.rgb.shape[0])
    support = review_support_mask(candidate.rgb, valid, item.get("depth"), rgb_offset, row.raw_rgb_anno_class_path)
    add_target_metrics(item, candidate, metrics, support)

    depth_crop, depth_valid = crop_depth_to_canvas(item, rgb_offset, target_size)
    depth_view = depth_support_view(depth_crop, depth_valid, support, valid)
    fusion_mask = review_thermal_target_mask(candidate.lwir, valid, support)
    fusion_preview = thermal_blend(candidate.rgb, candidate.lwir, valid, support)
    review_blend = simple_review_fusion(candidate.rgb, candidate.lwir, valid, support)
    anti_result = make_anti_ghost_support(candidate.rgb, candidate.lwir, valid, support, depth_crop, depth_valid, args)
    if args.visual_mode == "legacy":
        anti_mask = (support | fusion_mask) & valid
        rejected_mask = np.zeros_like(anti_mask, dtype=bool)
    else:
        anti_mask = anti_result["mask"]
        rejected_mask = anti_result["rejected_mask"]
    focus_mask = anti_mask if np.any(anti_mask) else ((support | fusion_mask) & valid)
    alpha = feather_alpha(focus_mask, valid, args.anti_ghost_feather_px)
    anti_blend = anti_ghost_fusion(candidate.rgb, candidate.lwir, valid, focus_mask, alpha) if np.any(focus_mask) else review_blend
    clean_depth_view = depth_support_view(depth_crop, depth_valid, focus_mask, valid)
    clean_lwir_view = clean_registered_lwir_view(candidate.lwir, candidate.lwir_valid, focus_mask)
    alpha_view = alpha_mask_view(alpha, focus_mask, rejected_mask)
    contour_overlay = contour_alignment_overlay(candidate.rgb, candidate.lwir, valid, focus_mask, depth_crop, depth_valid)
    risk_view, risk_metrics = tear_ghost_diagnostics(candidate.rgb, candidate.lwir, valid, focus_mask, alpha, depth_crop, depth_valid)
    error_heat = edge_error_heatmap(item["aligned_lwir_u8"], candidate.lwir, candidate.lwir_valid)

    metrics["sample_id"] = sample_id(row)
    metrics["phase28_profile"] = args.current_profile
    metrics["phase28_visual_mode"] = str(args.visual_mode)
    metrics["depth_canvas_valid_ratio"] = float(depth_valid.mean())
    metrics["depth_support_pixels"] = int(np.count_nonzero(support))
    metrics["depth_support_ratio"] = float(np.count_nonzero(support) / max(1, support.size))
    metrics["fusion_target_pixels"] = int(np.count_nonzero(fusion_mask))
    metrics["anti_ghost_support_pixels"] = int(np.count_nonzero(focus_mask))
    metrics["anti_ghost_support_ratio"] = float(np.count_nonzero(focus_mask) / max(1, focus_mask.size))
    metrics["anti_ghost_raw_pixels"] = int(np.count_nonzero(anti_result["raw_mask"]))
    metrics["anti_ghost_depth_pixels"] = int(np.count_nonzero(anti_result["depth_mask"]))
    metrics["anti_ghost_thermal_pixels"] = int(np.count_nonzero(anti_result["thermal_mask"]))
    metrics["rejected_background_pixels"] = int(np.count_nonzero(rejected_mask))
    metrics["anti_ghost_feather_px"] = float(args.anti_ghost_feather_px)
    metrics["anti_ghost_erode_px"] = int(args.anti_ghost_erode_px)
    metrics["max_support_area_frac"] = float(args.max_support_area_frac)
    metrics.update(risk_metrics)
    metrics["phase28_sample_pass"] = bool(
        float(metrics["eval_lwir_to_mm5_aligned_t16_edge_distance"]) < float(args.edge_target_px)
        and str(metrics.get("uses_aligned_for_generation", "True")).lower() == "false"
    )

    imwrite_unicode(output_dir / "registered_lwir" / f"p28_{sample_tag}_lwir.png", candidate.lwir)
    imwrite_unicode(output_dir / "clean_registered_lwir" / f"p28_{sample_tag}_clean_lwir.png", clean_lwir_view)
    imwrite_unicode(output_dir / "fusion_review" / f"p28_{sample_tag}_rgb_lwir_review.png", anti_blend)
    imwrite_unicode(output_dir / "depth_support" / f"p28_{sample_tag}_depth_support.png", depth_view)
    imwrite_unicode(output_dir / "anti_ghost_support" / f"p28_{sample_tag}_anti_ghost_support.png", clean_depth_view)
    imwrite_unicode(output_dir / "alpha_masks" / f"p28_{sample_tag}_anti_ghost_alpha.png", alpha_view)
    imwrite_unicode(output_dir / "contour_overlays" / f"p28_{sample_tag}_contours.png", contour_overlay)
    imwrite_unicode(output_dir / "tear_ghost_maps" / f"p28_{sample_tag}_tear_ghost_risk.png", risk_view)
    imwrite_unicode(output_dir / "edge_error_heatmaps" / f"p28_{sample_tag}_edge_error.png", error_heat)
    imwrite_unicode(output_dir / "fusion_preview" / f"p28_{sample_tag}_rgb_lwir_fuse.png", fusion_preview)

    make_six_panel(
        [
            (candidate.rgb, "Generated RGB"),
            (clean_depth_view, "Clean depth support"),
            (clean_lwir_view, "Clean LWIR target"),
            (anti_blend, "Anti-ghost fusion"),
            (contour_overlay, f"Contours {metrics['eval_lwir_to_mm5_aligned_t16_edge_distance']:.3f}px"),
            (risk_view, "Tear/Ghost risk"),
        ],
        output_dir / "acceptance_panels" / f"p28_{sample_tag}_acceptance.png",
    )

    if args.save_before_after:
        make_panel_grid(
            [
                (review_blend, "Phase28 baseline fusion"),
                (anti_blend, "Anti-ghost fusion"),
                (alpha_view, "Anti-ghost alpha"),
                (clean_lwir_view, "Clean LWIR target"),
                (depth_view, "Raw depth support"),
                (clean_depth_view, "Clean depth support"),
                (risk_view, "Tear/Ghost risk"),
            ],
            output_dir / "before_after_panels" / f"p28_{sample_tag}_before_after.png",
            tile_size=(330, 245),
            columns=3,
        )

    crop_mask = (focus_mask | support | fusion_mask) if np.any(focus_mask | support | fusion_mask) else valid
    crop = foreground_crop(crop_mask, margin=96)
    if crop is not None:
        x0, y0, x1, y1 = crop
        make_six_panel(
            [
                (candidate.rgb[y0:y1, x0:x1], "RGB ROI"),
                (clean_depth_view[y0:y1, x0:x1], "Clean support ROI"),
                (clean_lwir_view[y0:y1, x0:x1], "Clean LWIR ROI"),
                (anti_blend[y0:y1, x0:x1], "Anti-ghost ROI"),
                (contour_overlay[y0:y1, x0:x1], f"Target edge {metrics['eval_target_lwir_to_mm5_aligned_t16_edge_distance']:.3f}px"),
                (risk_view[y0:y1, x0:x1], "Risk ROI"),
            ],
            output_dir / "roi_panels" / f"p28_{sample_tag}_roi.png",
            tile_size=(260, 205),
        )


def augment_phase28_summary(summary_rows: list[dict], metric_rows: list[dict]) -> None:
    grouped: dict[str, list[dict]] = {}
    for row in metric_rows:
        grouped.setdefault(str(row["candidate"]), []).append(row)
    numeric_keys = [
        "eval_lwir_to_mm5_aligned_t16_edge_distance",
        "eval_lwir_to_mm5_aligned_t16_ncc",
        "eval_rgb_to_mm5_aligned_rgb_ncc",
        "eval_target_lwir_to_mm5_aligned_t16_edge_distance",
        "eval_target_lwir_to_mm5_aligned_t16_ncc",
        "depth_canvas_valid_ratio",
        "depth_support_pixels",
        "depth_support_ratio",
        "anti_ghost_support_pixels",
        "anti_ghost_support_ratio",
        "anti_ghost_raw_pixels",
        "anti_ghost_depth_pixels",
        "anti_ghost_thermal_pixels",
        "rejected_background_pixels",
        "tear_risk_pixels",
        "ghost_risk_pixels",
        "ghost_edge_ratio",
        "alpha_edge_overlap",
        "depth_discontinuity_pixels",
        "target_support_pixels",
        "target_eval_pixels",
        "lwir_valid_ratio",
        "intersection_valid_ratio",
    ]
    constant_keys = [
        "allowed_for_generation",
        "phase28_class",
        "phase28_profile",
        "depth_registration_dx_px",
        "depth_registration_dy_px",
        "stabilization_kernel_px",
        "fill_depth_border",
        "uses_aligned_for_generation",
        "phase28_visual_mode",
        "anti_ghost_feather_px",
        "anti_ghost_erode_px",
        "max_support_area_frac",
        "calibration_file",
        "thermal_camera_calibration_file",
        "target_size",
    ]
    for summary in summary_rows:
        items = grouped.get(str(summary["candidate"]), [])
        if not items:
            continue
        for key in constant_keys:
            if key in items[0]:
                summary[key] = items[0][key]
        for key in numeric_keys:
            values = [float(item[key]) for item in items if key in item and np.isfinite(float(item[key]))]
            if values:
                summary[f"{key}_mean"] = float(np.mean(values))
                summary[f"{key}_min"] = float(np.min(values))
                summary[f"{key}_max"] = float(np.max(values))
        summary["phase28_sample_pass_count"] = int(sum(1 for item in items if bool(item.get("phase28_sample_pass"))))
        summary["phase28_sample_fail_count"] = int(sum(1 for item in items if not bool(item.get("phase28_sample_pass"))))


def write_report(output_dir: Path, payload: dict, metric_rows: list[dict]) -> None:
    best = payload["phase28_candidate"]
    failed = [row for row in metric_rows if not bool(row.get("phase28_sample_pass"))]
    lines = [
        "# Phase28 Calibration/Depth Acceptance Registration",
        "",
        "## Boundary",
        "- Generation uses calibration files, raw RGB/LWIR, and raw depth.",
        "- MM5 aligned RGB/T16 are evaluation-only and are not used to generate or choose the registered output.",
        "",
        "## Profile",
        f"- profile: `{payload['profile']}`",
        f"- aligned ids: `{payload['aligned_ids']}`",
        f"- metric gate required: `{payload['metric_gate_required']}`",
        f"- metric gate passed: `{payload['metric_gate_passed']}`",
        "",
        "## Calibration and Depth Parameters",
        f"- stereo calibration: `{best.get('calibration_file', '')}`",
        f"- thermal calibration: `{best.get('thermal_camera_calibration_file', '')}`",
        f"- target size: `{best.get('target_size', '')}`",
        f"- RGB crop offset: `{payload['context'].get('rgb_offset_xy')}`",
        f"- LWIR crop offset: `{payload['context'].get('lwir_offset_xy')}`",
        f"- board transform: `{payload['context'].get('board_transform_name')}`, RMSE `{payload['context'].get('board_transform_rmse_px'):.4f}px`",
        f"- residual shift: `dx={best.get('depth_registration_dx_px')}`, `dy={best.get('depth_registration_dy_px')}`",
        f"- stabilization kernel: `{best.get('stabilization_kernel_px')} px`",
        f"- depth border fill: `{best.get('fill_depth_border')}`",
        f"- visual mode: `{best.get('phase28_visual_mode', '')}`",
        f"- anti-ghost feather/erode: `{best.get('anti_ghost_feather_px', '')}` / `{best.get('anti_ghost_erode_px', '')}` px",
        f"- max visual support area fraction: `{best.get('max_support_area_frac', '')}`",
        "",
        "## Result",
        f"- LWIR edge distance mean/max: `{best['eval_lwir_to_mm5_aligned_t16_edge_distance_mean']:.4f}` / `{best['eval_lwir_to_mm5_aligned_t16_edge_distance_max']:.4f}` px",
        f"- LWIR NCC mean/min: `{best['eval_lwir_to_mm5_aligned_t16_ncc_mean']:.4f}` / `{best['eval_lwir_to_mm5_aligned_t16_ncc_min']:.4f}`",
        f"- RGB NCC mean/min: `{best['eval_rgb_to_mm5_aligned_rgb_ncc_mean']:.4f}` / `{best['eval_rgb_to_mm5_aligned_rgb_ncc_min']:.4f}`",
        f"- target edge distance mean/max: `{best.get('eval_target_lwir_to_mm5_aligned_t16_edge_distance_mean', float('nan')):.4f}` / `{best.get('eval_target_lwir_to_mm5_aligned_t16_edge_distance_max', float('nan')):.4f}` px",
        f"- depth support pixels mean/min/max: `{best.get('depth_support_pixels_mean', float('nan')):.1f}` / `{best.get('depth_support_pixels_min', float('nan')):.1f}` / `{best.get('depth_support_pixels_max', float('nan')):.1f}`",
        f"- anti-ghost support pixels mean/min/max: `{best.get('anti_ghost_support_pixels_mean', float('nan')):.1f}` / `{best.get('anti_ghost_support_pixels_min', float('nan')):.1f}` / `{best.get('anti_ghost_support_pixels_max', float('nan')):.1f}`",
        f"- rejected background pixels mean/max: `{best.get('rejected_background_pixels_mean', float('nan')):.1f}` / `{best.get('rejected_background_pixels_max', float('nan')):.1f}`",
        f"- tear risk pixels mean/max: `{best.get('tear_risk_pixels_mean', float('nan')):.1f}` / `{best.get('tear_risk_pixels_max', float('nan')):.1f}`",
        f"- ghost edge ratio mean/max: `{best.get('ghost_edge_ratio_mean', float('nan')):.4f}` / `{best.get('ghost_edge_ratio_max', float('nan')):.4f}`",
        f"- sample pass/fail: `{best.get('phase28_sample_pass_count', 0)}` / `{best.get('phase28_sample_fail_count', 0)}`",
        "",
        "## Visual Diagnostics",
        "- acceptance panels show generated RGB, cleaned depth support, cleaned LWIR target, anti-ghost fusion, RGB/LWIR/depth contours, and tear/ghost risk.",
        "- raw full-frame registered LWIR is still saved in `registered_lwir/` for traceability; the acceptance panel uses `clean_registered_lwir/` to avoid mistaking thermal reflection and sensor bloom for final fusion ghosting.",
        "- contour colors: RGB cyan, LWIR red, depth green, RGB/LWIR overlap white.",
        "- risk map colors: alpha boundary white, depth discontinuity green, ghost-risk edges magenta, tear-risk pixels red.",
        "",
        "## Per-Sample Failures",
    ]
    if not failed:
        lines.append("- none")
    else:
        for row in failed:
            lines.append(
                "- `{}`: edge `{:.4f}px`, LWIR NCC `{:.4f}`, tear `{}`, ghost ratio `{:.4f}`, rejected `{}`, panel `acceptance_panels/p28_s{:03d}_acceptance.png`".format(
                    row.get("sample_id", ""),
                    float(row.get("eval_lwir_to_mm5_aligned_t16_edge_distance", float("nan"))),
                    float(row.get("eval_lwir_to_mm5_aligned_t16_ncc", float("nan"))),
                    int(row.get("tear_risk_pixels", 0)),
                    float(row.get("ghost_edge_ratio", float("nan"))),
                    int(row.get("rejected_background_pixels", 0)),
                    int(str(row.get("sample_id", "000")).split("_", 1)[0]) if str(row.get("sample_id", "000")).split("_", 1)[0].isdigit() else 0,
                )
            )
    lines.extend(
        [
            "",
            "## Output Files",
            "- `registered_lwir/`: generated registered LWIR frames.",
            "- `clean_registered_lwir/`: acceptance-facing LWIR target views with background/reflection suppressed by the anti-ghost support mask.",
            "- `fusion_review/`: final anti-ghost visual registration review images.",
            "- `acceptance_panels/`: six-panel acceptance images with depth, contours, and tear/ghost evidence.",
            "- `roi_panels/`: target-focused acceptance crops.",
            "- `before_after_panels/`: baseline Phase28 fusion vs anti-ghost fusion comparisons.",
            "- `anti_ghost_support/`: cleaned visual support from calibration/depth/thermal consensus.",
            "- `alpha_masks/`: feathered anti-ghost alpha masks with rejected pixels.",
            "- `contour_overlays/`: RGB/LWIR/depth contour overlays for visual registration review.",
            "- `tear_ghost_maps/`: diagnostic maps for tearing and ghosting risk.",
            "- `edge_error_heatmaps/`: evaluation-only edge-distance heatmaps.",
            "- `metrics/p28_metrics.csv`, `metrics/p28_summary.csv`, `metrics/p28_best.json`.",
        ]
    )
    (output_dir / "reports").mkdir(parents=True, exist_ok=True)
    (output_dir / "reports" / "p28_acceptance_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def make_output_dirs(output_dir: Path) -> None:
    for child in (
        "metrics",
        "reports",
        "acceptance_panels",
        "roi_panels",
        "registered_lwir",
        "clean_registered_lwir",
        "fusion_preview",
        "fusion_review",
        "depth_support",
        "anti_ghost_support",
        "alpha_masks",
        "contour_overlays",
        "tear_ghost_maps",
        "before_after_panels",
        "edge_error_heatmaps",
    ):
        (output_dir / child).mkdir(parents=True, exist_ok=True)


def run_profile(args, profile: str, output_dir: Path) -> dict:
    local_args = copy.copy(args)
    local_args.current_profile = profile
    if not str(local_args.aligned_ids).strip():
        local_args.aligned_ids = PROFILE_IDS[profile]
    make_output_dirs(output_dir)

    prepared_rows, context = prepare_rows(local_args)
    metric_rows: list[dict] = []
    for item in prepared_rows:
        print(f"processing {sample_id(item['row'])} phase28 acceptance candidate [{profile}]")
        candidate = make_phase28_candidate(item, local_args)
        metrics = add_phase28_metadata(evaluate_candidate(item["row"], candidate, item["aligned_rgb"], item["aligned_lwir_u8"]), candidate)
        save_phase28_outputs(output_dir, item, candidate, metrics, local_args)
        metric_rows.append(metrics)

    summary_rows = summarize(metric_rows)
    augment_phase28_summary(summary_rows, metric_rows)
    summary_rows.sort(key=lambda row: float(row.get("eval_lwir_to_mm5_aligned_t16_edge_distance_mean", float("inf"))))
    best = summary_rows[0]
    metric_gate_passed = (
        float(best["eval_lwir_to_mm5_aligned_t16_edge_distance_mean"]) < float(local_args.edge_target_px)
        and float(best["eval_lwir_to_mm5_aligned_t16_edge_distance_max"]) < float(local_args.edge_target_px)
        and int(best.get("phase28_sample_fail_count", 0)) == 0
        and str(best.get("uses_aligned_for_generation", "True")).lower() == "false"
    )
    metric_gate_required = profile in {"core", "review"} or bool(str(args.aligned_ids).strip())

    write_csv(output_dir / "metrics" / "p28_metrics.csv", metric_rows, collect_fieldnames(metric_rows, []))
    write_csv(output_dir / "metrics" / "p28_summary.csv", summary_rows, collect_fieldnames(summary_rows, []))
    payload = {
        "profile": profile,
        "aligned_ids": str(local_args.aligned_ids),
        "context": context,
        "edge_target_px": float(local_args.edge_target_px),
        "metric_gate_required": bool(metric_gate_required),
        "metric_gate_passed": bool(metric_gate_passed),
        "required_acceptance_passed": bool(metric_gate_passed or not metric_gate_required),
        "aligned_usage": "evaluation_only_not_generation_or_bridge",
        "phase28_candidate": best,
        "candidate_summary": summary_rows,
        "per_sample": metric_rows,
    }
    (output_dir / "metrics" / "p28_best.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=json_default),
        encoding="utf-8",
    )
    write_report(output_dir, payload, metric_rows)

    print(f"Phase28 {profile} output: {output_dir}")
    print(
        "edge mean/max: "
        f"{best['eval_lwir_to_mm5_aligned_t16_edge_distance_mean']:.4f} / "
        f"{best['eval_lwir_to_mm5_aligned_t16_edge_distance_max']:.4f} px"
    )
    print(
        "LWIR NCC mean/min: "
        f"{best['eval_lwir_to_mm5_aligned_t16_ncc_mean']:.4f} / "
        f"{best['eval_lwir_to_mm5_aligned_t16_ncc_min']:.4f}"
    )
    print(f"metric gate passed: {metric_gate_passed}")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase28 calibration/depth assisted acceptance registration.")
    parser.add_argument("--index", default="mm5_calib_benchmark/outputs/mm5_benchmark/splits/index_with_splits.csv")
    parser.add_argument("--aligned-ids", default="")
    parser.add_argument("--limit", type=int, default=3)
    parser.add_argument("--splits", default="test,val")
    parser.add_argument("--calibration", default="calibration/def_stereocalib_THERM.yml")
    parser.add_argument("--thermal-camera-calibration", default="calibration/def_thermalcam_ori.yml")
    parser.add_argument("--calibration-root", default="")
    parser.add_argument("--target-size", default="640x480")
    parser.add_argument("--max-board-offset-rmse-px", type=float, default=12.0)
    parser.add_argument("--point-margin-px", type=int, default=80)
    parser.add_argument("--ransac-threshold-px", type=float, default=4.0)
    parser.add_argument("--dx-px", type=float, default=3.5)
    parser.add_argument("--dy-px", type=float, default=2.0)
    parser.add_argument("--stabilization-kernel-px", type=int, default=9)
    parser.add_argument("--no-depth-border-fill", action="store_true")
    parser.add_argument("--edge-target-px", type=float, default=3.0)
    parser.add_argument("--visual-mode", choices=["anti-ghost", "legacy"], default="anti-ghost")
    parser.add_argument("--anti-ghost-feather-px", type=float, default=3.0)
    parser.add_argument("--anti-ghost-erode-px", type=int, default=2)
    parser.add_argument("--max-support-area-frac", type=float, default=0.10)
    parser.add_argument("--save-before-after", dest="save_before_after", action="store_true", default=True)
    parser.add_argument("--no-save-before-after", dest="save_before_after", action="store_false")
    parser.add_argument("--run-profile", choices=["core", "review", "broad", "all"], default="core")
    parser.add_argument("--output", default=str(PHASE28_DIR / "outputs_visual_acceptance"))
    args = parser.parse_args()

    output_dir = Path(args.output)
    profiles = ["core", "review", "broad"] if args.run_profile == "all" and not str(args.aligned_ids).strip() else [args.run_profile if args.run_profile != "all" else "core"]
    payloads = {}
    for profile in profiles:
        profile_output = output_dir / profile if len(profiles) > 1 else output_dir
        payloads[profile] = run_profile(args, profile, profile_output)

    if len(payloads) > 1:
        (output_dir / "p28_all_profiles_summary.json").write_text(
            json.dumps(payloads, ensure_ascii=False, indent=2, default=json_default),
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
