from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np

METHOD_DIR = Path(__file__).resolve().parent
DARKLIGHT_DIR = METHOD_DIR.parent
if str(DARKLIGHT_DIR) not in sys.path:
    sys.path.insert(0, str(DARKLIGHT_DIR))
if str(METHOD_DIR) not in sys.path:
    sys.path.insert(0, str(METHOD_DIR))

from run_darklight import (  # noqa: E402
    auto_edges,
    collect_fieldnames,
    imread_unicode,
    load_rows,
    load_stereo_calibration,
    make_edge_overlay,
    make_five_panel,
    matrix_to_list,
    normalize_u8,
    parse_size_arg,
    select_dark_rows,
    write_csv,
)
from run_calibration_only import (  # noqa: E402
    CandidateOutput,
    crop_mask_with_border,
    crop_with_border,
    read_rectification_matrices,
    read_single_camera_model,
    sample_id,
)
from diagnose_aligned_canvas import evaluate_candidate, load_bridge_metrics, tuple2  # noqa: E402
from run_phase22_stereo_recalib import (  # noqa: E402
    add_metadata_to_metrics,
    default_calibration_root,
    make_phase21_ceiling_rule,
    summarize,
)
from run_phase23_lwir_board_offset import build_offset_candidates, collect_board_offsets  # noqa: E402
from run_phase24_lwir_board_affine import (  # noqa: E402
    BoardTransform,
    collect_board_correspondences,
    fit_transforms,
    make_candidate as make_phase24_candidate,
)


def choose_rows(index: str, aligned_ids: str, limit: int, splits: str):
    rows = load_rows(index, require_official=False, require_depth=True)
    if aligned_ids.strip():
        wanted = {int(x.strip()) for x in aligned_ids.split(",") if x.strip()}
        return sorted([row for row in rows if row.aligned_id in wanted], key=lambda row: row.aligned_id)
    return select_dark_rows(rows, limit, splits)


def matrix_json(matrix: np.ndarray) -> str:
    return json.dumps(matrix_to_list(matrix.astype(np.float64)))


def depth_project_lwir_to_rgb_canvas(
    raw_lwir_u8: np.ndarray,
    depth: np.ndarray,
    rgb_offset: tuple[int, int],
    target_size: tuple[int, int],
    calibration,
    thermal_model,
) -> tuple[np.ndarray, np.ndarray, dict]:
    target_w, target_h = target_size
    offset_x, offset_y = rgb_offset
    depth_h, depth_w = depth.shape[:2]

    x_raw = offset_x + np.arange(target_w, dtype=np.float32)
    y_raw = offset_y + np.arange(target_h, dtype=np.float32)
    xx, yy = np.meshgrid(x_raw, y_raw)
    in_depth = (xx >= 0) & (xx < depth_w) & (yy >= 0) & (yy < depth_h)

    z = np.zeros((target_h, target_w), dtype=np.float32)
    valid_y = yy[in_depth].astype(np.int32)
    valid_x = xx[in_depth].astype(np.int32)
    z[in_depth] = depth[valid_y, valid_x].astype(np.float32)
    valid = in_depth & np.isfinite(z) & (z > 0)

    raw_pixels = np.stack([xx.reshape(-1), yy.reshape(-1)], axis=1).astype(np.float32).reshape(-1, 1, 2)
    norm = cv2.undistortPoints(
        raw_pixels,
        calibration.rgb_K,
        calibration.rgb_D.reshape(-1, 1),
    ).reshape(target_h, target_w, 2)

    xyz_rgb = np.stack([norm[:, :, 0] * z, norm[:, :, 1] * z, z], axis=-1).reshape(-1, 3).astype(np.float32)
    xyz_lwir = (calibration.rgb_to_lwir_R @ xyz_rgb.T).T + calibration.rgb_to_lwir_T.reshape(1, 3)
    z_lwir = xyz_lwir[:, 2].reshape(target_h, target_w)
    valid &= z_lwir > 1e-6

    uv, _ = cv2.projectPoints(
        xyz_lwir.astype(np.float32),
        np.zeros(3),
        np.zeros(3),
        thermal_model.K,
        thermal_model.D.reshape(-1, 1),
    )
    uv = uv.reshape(target_h, target_w, 2)
    map_x = uv[:, :, 0].astype(np.float32)
    map_y = uv[:, :, 1].astype(np.float32)
    lwir_h, lwir_w = raw_lwir_u8.shape[:2]
    in_lwir = (map_x >= 0) & (map_x < lwir_w - 1) & (map_y >= 0) & (map_y < lwir_h - 1)
    valid &= in_lwir

    map_x = np.where(valid, map_x, -1).astype(np.float32)
    map_y = np.where(valid, map_y, -1).astype(np.float32)
    projected = cv2.remap(
        raw_lwir_u8,
        map_x,
        map_y,
        cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    debug = {
        "depth_valid_ratio": float((np.isfinite(z) & (z > 0)).mean()),
        "depth_project_valid_ratio": float(valid.mean()),
        "projection_camera": "rgb_depth_to_raw_lwir_using_thermal_ori_intrinsics",
    }
    return projected, valid, debug


def make_depth_projection_candidate(
    row,
    raw_rgb: np.ndarray,
    raw_lwir_u8: np.ndarray,
    depth: np.ndarray,
    rgb_offset: tuple[int, int],
    target_size: tuple[int, int],
    calibration,
    thermal_model,
) -> CandidateOutput:
    rgb_valid_full = np.ones(raw_rgb.shape[:2], dtype=bool)
    rgb = crop_with_border(raw_rgb, rgb_offset, target_size)
    rgb_valid = crop_mask_with_border(rgb_valid_full, rgb_offset, target_size)
    lwir, lwir_valid, debug = depth_project_lwir_to_rgb_canvas(
        raw_lwir_u8,
        depth,
        rgb_offset,
        target_size,
        calibration,
        thermal_model,
    )
    return CandidateOutput(
        name="phase25_depth_project_raw_lwir",
        rgb=rgb,
        lwir=lwir,
        rgb_valid=rgb_valid,
        lwir_valid=lwir_valid,
        metadata={
            "method": "depth_project_raw_lwir",
            "allowed_for_generation": True,
            "rgb_source": "phase21_fixed_rgb_canvas",
            "lwir_source": "raw_lwir_sampled_by_rgb_depth_projection",
            "rgb_crop_offset_xy": tuple2(rgb_offset),
            "lwir_crop_offset_xy": [],
            "rule_source": "raw depth image + RGB->LWIR stereo calibration; no MM5 aligned input",
            "depth_used": True,
            "depth_project_valid_ratio": debug["depth_project_valid_ratio"],
            "depth_valid_ratio": debug["depth_valid_ratio"],
            "sample": sample_id(row),
        },
    )


def make_holefill_candidate(
    phase24: CandidateOutput,
    depth_projection: CandidateOutput,
    *,
    name: str,
    method: str,
    use_union_valid: bool,
    transform: BoardTransform,
) -> CandidateOutput:
    fill_mask = (~phase24.lwir_valid.astype(bool)) & depth_projection.lwir_valid.astype(bool)
    lwir = phase24.lwir.copy()
    lwir[fill_mask] = depth_projection.lwir[fill_mask]
    lwir_valid = phase24.lwir_valid.astype(bool)
    if use_union_valid:
        lwir_valid = lwir_valid | fill_mask
    metadata = dict(phase24.metadata)
    metadata.update(
        {
            "method": method,
            "lwir_source": "phase24_affine_lwir_with_depth_projection_hole_fill",
            "depth_used": True,
            "depth_fill_pixels": int(np.count_nonzero(fill_mask)),
            "depth_fill_ratio": float(fill_mask.mean()),
            "depth_project_valid_ratio": float(depth_projection.lwir_valid.mean()),
            "depth_valid_policy": "union_phase24_and_depth" if use_union_valid else "keep_phase24_valid_mask",
            "board_transform_name": transform.name,
            "board_transform_matrix": matrix_to_list(transform.matrix),
            "board_transform_inliers": transform.inliers,
            "board_transform_mae_px": transform.board_mae_px,
            "board_transform_rmse_px": transform.board_rmse_px,
            "rule_source": (
                "Phase 24 affine geometry fitted from calibration-board corners; "
                "depth projection fills only pixels outside the Phase 24 valid support"
            ),
        }
    )
    return CandidateOutput(
        name=name,
        rgb=phase24.rgb,
        lwir=lwir,
        rgb_valid=phase24.rgb_valid,
        lwir_valid=lwir_valid,
        metadata=metadata,
    )


def make_depth_foreground_boundary(
    depth: np.ndarray,
    rgb_offset: tuple[int, int],
    target_size: tuple[int, int],
    foreground_threshold_mm: float,
) -> tuple[np.ndarray, np.ndarray, dict]:
    valid_full = np.isfinite(depth) & (depth > 0)
    depth_crop = crop_with_border(depth.astype(np.float32), rgb_offset, target_size)
    valid = crop_mask_with_border(valid_full, rgb_offset, target_size)
    near = valid & (depth_crop > 0) & (depth_crop < float(foreground_threshold_mm))
    near_u8 = near.astype(np.uint8)
    near_u8 = cv2.morphologyEx(near_u8, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8), iterations=1)
    near_u8 = cv2.morphologyEx(near_u8, cv2.MORPH_CLOSE, np.ones((11, 11), np.uint8), iterations=1)

    component_count, labels, stats, _centroids = cv2.connectedComponentsWithStats(near_u8, 8)
    kept = np.zeros_like(near_u8)
    for idx in range(1, component_count):
        if int(stats[idx, cv2.CC_STAT_AREA]) >= 2000:
            kept[labels == idx] = 1

    boundary = cv2.morphologyEx(kept, cv2.MORPH_GRADIENT, np.ones((7, 7), np.uint8)) > 0
    boundary[:15, :] = False
    boundary[-15:, :] = False
    boundary[:, :15] = False
    boundary[:, -15:] = False
    debug = {
        "depth_foreground_threshold_mm": float(foreground_threshold_mm),
        "depth_foreground_ratio": float(kept.mean()),
        "depth_boundary_pixels": int(np.count_nonzero(boundary)),
        "depth_valid_ratio": float(valid.mean()),
    }
    return boundary, kept.astype(bool), debug


def warp_translation(image: np.ndarray, valid: np.ndarray, dx: int, dy: int) -> tuple[np.ndarray, np.ndarray]:
    height, width = image.shape[:2]
    matrix = np.array([[1.0, 0.0, float(dx)], [0.0, 1.0, float(dy)]], dtype=np.float32)
    warped = cv2.warpAffine(
        image,
        matrix,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    warped_valid = (
        cv2.warpAffine(
            valid.astype(np.uint8) * 255,
            matrix,
            (width, height),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        > 0
    )
    return warped, warped_valid


def depth_boundary_distance(lwir: np.ndarray, valid: np.ndarray, boundary: np.ndarray) -> float:
    score_mask = valid.astype(bool) & boundary.astype(bool)
    if int(np.count_nonzero(score_mask)) < 100:
        return float("inf")
    moving_edges = np.where(valid, auto_edges(lwir), 0).astype(np.uint8)
    if int(np.count_nonzero(moving_edges)) < 100:
        return float("inf")
    distance = cv2.distanceTransform(255 - moving_edges, cv2.DIST_L2, 3)
    values = distance[score_mask]
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("inf")
    return float(np.mean(np.clip(values, 0, 50)))


def select_global_depth_translation(prepared_rows: list[dict], radius_px: int) -> tuple[dict, list[dict]]:
    score_rows: list[dict] = []
    best = {"mean_depth_boundary_distance": float("inf"), "dx_px": 0, "dy_px": 0}
    radius_px = int(max(0, radius_px))
    for dx in range(-radius_px, radius_px + 1):
        for dy in range(-radius_px, radius_px + 1):
            per_sample = []
            for item in prepared_rows:
                phase24 = item["phase24"]
                warped, warped_valid = warp_translation(phase24.lwir, phase24.lwir_valid, dx, dy)
                distance = depth_boundary_distance(warped, warped_valid, item["depth_boundary"])
                per_sample.append(distance)
                score_rows.append(
                    {
                        "sample": sample_id(item["row"]),
                        "dx_px": int(dx),
                        "dy_px": int(dy),
                        "depth_boundary_distance": distance,
                        "depth_boundary_pixels": int(item["depth_debug"]["depth_boundary_pixels"]),
                    }
                )
            finite = [value for value in per_sample if np.isfinite(value)]
            mean_distance = float(np.mean(finite)) if finite else float("inf")
            if mean_distance < float(best["mean_depth_boundary_distance"]):
                best = {
                    "mean_depth_boundary_distance": mean_distance,
                    "dx_px": int(dx),
                    "dy_px": int(dy),
                }
    baseline = [
        depth_boundary_distance(item["phase24"].lwir, item["phase24"].lwir_valid, item["depth_boundary"])
        for item in prepared_rows
    ]
    baseline_finite = [value for value in baseline if np.isfinite(value)]
    baseline_mean = float(np.mean(baseline_finite)) if baseline_finite else float("inf")
    best["baseline_mean_depth_boundary_distance"] = baseline_mean
    best["depth_boundary_distance_gain"] = baseline_mean - float(best["mean_depth_boundary_distance"])
    return best, score_rows


def make_depth_registered_candidate(
    phase24: CandidateOutput,
    depth_projection: CandidateOutput,
    selection: dict,
    *,
    fill_depth_border: bool,
    depth_debug: dict,
) -> CandidateOutput:
    dx = int(selection["dx_px"])
    dy = int(selection["dy_px"])
    lwir, lwir_valid = warp_translation(phase24.lwir, phase24.lwir_valid, dx, dy)
    fill_mask = np.zeros(lwir.shape[:2], dtype=bool)
    if fill_depth_border:
        fill_mask = (~lwir_valid.astype(bool)) & depth_projection.lwir_valid.astype(bool)
        lwir[fill_mask] = depth_projection.lwir[fill_mask]

    metadata = dict(phase24.metadata)
    metadata.update(
        {
            "method": (
                "depth_boundary_registered_global_translation_depth_fill"
                if fill_depth_border
                else "depth_boundary_registered_global_translation"
            ),
            "lwir_source": (
                "phase24_affine_lwir_shifted_by_depth_boundary_registration"
                if not fill_depth_border
                else "phase24_affine_lwir_shifted_by_depth_boundary_registration_with_depth_border_fill"
            ),
            "depth_used": True,
            "depth_registration_dx_px": dx,
            "depth_registration_dy_px": dy,
            "depth_registration_search_radius_px": int(selection.get("search_radius_px", 0)),
            "depth_registration_score": float(selection["mean_depth_boundary_distance"]),
            "depth_registration_baseline_score": float(selection["baseline_mean_depth_boundary_distance"]),
            "depth_registration_score_gain": float(selection["depth_boundary_distance_gain"]),
            "depth_foreground_threshold_mm": float(depth_debug["depth_foreground_threshold_mm"]),
            "depth_boundary_pixels": int(depth_debug["depth_boundary_pixels"]),
            "depth_foreground_ratio": float(depth_debug["depth_foreground_ratio"]),
            "depth_fill_pixels": int(np.count_nonzero(fill_mask)),
            "depth_fill_ratio": float(fill_mask.mean()),
            "depth_project_valid_ratio": float(depth_projection.lwir_valid.mean()),
            "depth_valid_policy": "shifted_phase24_valid_mask",
            "rule_source": (
                "Global residual translation selected by raw depth foreground-boundary distance "
                "to generated LWIR edges; MM5 aligned images are evaluation-only"
            ),
        }
    )
    return CandidateOutput(
        name=(
            "phase25_depth_registered_global_shift_depth_fill"
            if fill_depth_border
            else "phase25_depth_registered_global_shift"
        ),
        rgb=phase24.rgb,
        lwir=lwir,
        rgb_valid=phase24.rgb_valid,
        lwir_valid=lwir_valid,
        metadata=metadata,
    )


def make_median_control_candidate(phase24: CandidateOutput) -> CandidateOutput:
    lwir = phase24.lwir.copy()
    valid = phase24.lwir_valid.astype(bool)
    fill_value = int(np.median(lwir[valid])) if np.any(valid) else 0
    lwir[~valid] = fill_value
    metadata = dict(phase24.metadata)
    metadata.update(
        {
            "method": "phase24_affine_median_holefill_control",
            "lwir_source": "phase24_affine_lwir_with_constant_median_hole_fill",
            "depth_used": False,
            "depth_fill_pixels": 0,
            "depth_fill_ratio": 0.0,
            "depth_valid_policy": "keep_phase24_valid_mask",
            "rule_source": "control candidate; no depth, fills invalid border with median thermal value",
        }
    )
    return CandidateOutput(
        name="phase25_median_holefill_control",
        rgb=phase24.rgb,
        lwir=lwir,
        rgb_valid=phase24.rgb_valid,
        lwir_valid=phase24.lwir_valid,
        metadata=metadata,
    )


def add_phase25_metadata(metrics: dict, candidate: CandidateOutput) -> dict:
    out = add_metadata_to_metrics(metrics, candidate)
    for key in (
        "depth_used",
        "depth_fill_pixels",
        "depth_fill_ratio",
        "depth_project_valid_ratio",
        "depth_valid_ratio",
        "depth_valid_policy",
        "depth_registration_dx_px",
        "depth_registration_dy_px",
        "depth_registration_search_radius_px",
        "depth_registration_score",
        "depth_registration_baseline_score",
        "depth_registration_score_gain",
        "depth_foreground_threshold_mm",
        "depth_boundary_pixels",
        "depth_foreground_ratio",
    ):
        if key in candidate.metadata:
            out[key] = candidate.metadata[key]
    return out


def augment_phase25_summary(summary_rows: list[dict], metric_rows: list[dict]) -> None:
    grouped: dict[str, list[dict]] = {}
    for row in metric_rows:
        grouped.setdefault(str(row["candidate"]), []).append(row)

    constant_keys = [
        "depth_used",
        "depth_valid_policy",
        "depth_registration_dx_px",
        "depth_registration_dy_px",
        "depth_registration_search_radius_px",
        "depth_registration_score",
        "depth_registration_baseline_score",
        "depth_registration_score_gain",
        "depth_foreground_threshold_mm",
    ]
    numeric_keys = [
        "depth_fill_pixels",
        "depth_fill_ratio",
        "depth_project_valid_ratio",
        "depth_valid_ratio",
        "depth_boundary_pixels",
        "depth_foreground_ratio",
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


def save_panel(output_dir: Path, row, candidate: CandidateOutput, aligned_rgb: np.ndarray, aligned_lwir_u8: np.ndarray) -> None:
    panel_tags = {
        "phase24_affine_lmeds_baseline": "p24_base",
        "phase25_depth_holefill_keep_phase24_valid": "p25_keep",
        "phase25_depth_holefill_union_valid": "p25_union",
        "phase25_depth_project_raw_lwir": "p25_proj",
        "phase25_depth_registered_global_shift": "p25_reg",
        "phase25_depth_registered_global_shift_depth_fill": "p25_regfill",
        "phase25_median_holefill_control": "p25_med",
    }
    sample_tag = f"s{int(row.aligned_id):03d}"
    panel_tag = panel_tags.get(candidate.name, candidate.name)
    aligned_lwir_bgr = cv2.cvtColor(aligned_lwir_u8, cv2.COLOR_GRAY2BGR)
    make_five_panel(
        [
            (candidate.rgb, "Fixed RGB"),
            (candidate.lwir, "Phase25 LWIR"),
            (aligned_rgb, "MM5 RGB eval"),
            (aligned_lwir_bgr, "MM5 T16 eval"),
            (make_edge_overlay(candidate.rgb, candidate.lwir, candidate.rgb_valid & candidate.lwir_valid), "Generated edge check"),
        ],
        output_dir / "panels" / f"dl_p25_{sample_tag}_{panel_tag}.png",
        tile_size=(360, 270),
    )


def write_report(
    output_dir: Path,
    summary_rows: list[dict],
    transform_rows: list[dict],
    bridge_metrics: dict,
    point_count: int,
) -> None:
    baseline = next((row for row in summary_rows if row["candidate"] == "phase24_affine_lmeds_baseline"), None)
    promoted = next(
        (row for row in summary_rows if row["candidate"] == "phase25_depth_registered_global_shift_depth_fill"),
        None,
    )
    registered = next((row for row in summary_rows if row["candidate"] == "phase25_depth_registered_global_shift"), None)
    legacy_holefill = next((row for row in summary_rows if row["candidate"] == "phase25_depth_holefill_keep_phase24_valid"), None)
    union = next((row for row in summary_rows if row["candidate"] == "phase25_depth_holefill_union_valid"), None)
    depth_only = next((row for row in summary_rows if row["candidate"] == "phase25_depth_project_raw_lwir"), None)
    best = summary_rows[0] if summary_rows else None

    lines = [
        "# Phase 25 Depth-Assisted LWIR Refinement",
        "",
        "## Constraint",
        "RGB stays fixed to the Phase 21 calibration-derived canvas. The main LWIR geometry stays the Phase 24 calibration-board affine result. Depth is used only from the raw per-sample depth image and stereo calibration; MM5 aligned images are evaluation-only.",
        "",
        "## Depth Branch",
        "- `phase25_depth_registered_global_shift`: selects one global LWIR residual translation by minimizing raw depth foreground-boundary distance to generated LWIR edges.",
        "- `phase25_depth_registered_global_shift_depth_fill`: applies that depth-selected registration and fills only the new invalid border from the dense depth projection.",
        "- `phase25_depth_project_raw_lwir`: samples raw LWIR through RGB depth and RGB->LWIR calibration as a pure dense depth-registration diagnostic.",
        "- `phase25_depth_holefill_keep_phase24_valid`: old conservative depth-fill diagnostic, retained only for comparison.",
        "",
        "## Bridge Target",
    ]
    if bridge_metrics:
        lines.extend(
            [
                f"- retained bridge LWIR NCC mean: `{bridge_metrics.get('raw_lwir_to_official_lwir_ncc_mean', float('nan')):.4f}`",
                f"- retained bridge LWIR NCC min: `{bridge_metrics.get('raw_lwir_to_official_lwir_ncc_min', float('nan')):.4f}`",
            ]
        )
    lines.extend(
        [
            "",
            "## Calibration Geometry",
            f"- checkerboard correspondence points: `{point_count}`",
        ]
    )
    if transform_rows:
        first = transform_rows[0]
        lines.append(
            f"- best board transform: `{first['candidate']}`, board RMSE `{float(first['board_rmse_px']):.4f}px`"
        )
    if baseline:
        lines.extend(
            [
                "",
                "## Phase 24 Baseline",
                f"- LWIR NCC mean/min: `{baseline.get('eval_lwir_to_mm5_aligned_t16_ncc_mean', float('nan')):.4f}` / `{baseline.get('eval_lwir_to_mm5_aligned_t16_ncc_min', float('nan')):.4f}`",
                f"- RGB NCC mean/min: `{baseline.get('eval_rgb_to_mm5_aligned_rgb_ncc_mean', float('nan')):.4f}` / `{baseline.get('eval_rgb_to_mm5_aligned_rgb_ncc_min', float('nan')):.4f}`",
            ]
        )
    if promoted:
        lines.extend(
            [
                "",
                "## Promoted Phase 25 Candidate",
                "- candidate: `phase25_depth_registered_global_shift_depth_fill`",
                f"- LWIR NCC mean/min: `{promoted.get('eval_lwir_to_mm5_aligned_t16_ncc_mean', float('nan')):.4f}` / `{promoted.get('eval_lwir_to_mm5_aligned_t16_ncc_min', float('nan')):.4f}`",
                f"- RGB NCC mean/min: `{promoted.get('eval_rgb_to_mm5_aligned_rgb_ncc_mean', float('nan')):.4f}` / `{promoted.get('eval_rgb_to_mm5_aligned_rgb_ncc_min', float('nan')):.4f}`",
                f"- depth-selected residual shift: `dx={promoted.get('depth_registration_dx_px', '')}`, `dy={promoted.get('depth_registration_dy_px', '')}`",
                f"- depth boundary score gain: `{promoted.get('depth_registration_score_gain', float('nan'))}`",
                f"- valid policy: `{promoted.get('depth_valid_policy', '')}`",
            ]
        )
    if registered:
        lines.extend(
            [
                "",
                "## Depth Registration Without Fill",
                f"- LWIR NCC mean/min: `{registered.get('eval_lwir_to_mm5_aligned_t16_ncc_mean', float('nan')):.4f}` / `{registered.get('eval_lwir_to_mm5_aligned_t16_ncc_min', float('nan')):.4f}`",
            ]
        )
    if legacy_holefill:
        lines.extend(
            [
                "",
                "## Legacy Depth Fill Diagnostic",
                f"- LWIR NCC mean/min: `{legacy_holefill.get('eval_lwir_to_mm5_aligned_t16_ncc_mean', float('nan')):.4f}` / `{legacy_holefill.get('eval_lwir_to_mm5_aligned_t16_ncc_min', float('nan')):.4f}`",
            ]
        )
    if union:
        lines.extend(
            [
                "",
                "## Depth Valid-Union Diagnostic",
                f"- LWIR NCC mean/min: `{union.get('eval_lwir_to_mm5_aligned_t16_ncc_mean', float('nan')):.4f}` / `{union.get('eval_lwir_to_mm5_aligned_t16_ncc_min', float('nan')):.4f}`",
            ]
        )
    if depth_only:
        lines.extend(
            [
                "",
                "## Depth-Only Diagnostic",
                f"- LWIR NCC mean/min: `{depth_only.get('eval_lwir_to_mm5_aligned_t16_ncc_mean', float('nan')):.4f}` / `{depth_only.get('eval_lwir_to_mm5_aligned_t16_ncc_min', float('nan')):.4f}`",
            ]
        )
    if best:
        lines.extend(
            [
                "",
                "## Best Evaluated Candidate",
                f"- candidate: `{best['candidate']}`",
                f"- LWIR NCC mean/min: `{best.get('eval_lwir_to_mm5_aligned_t16_ncc_mean', float('nan')):.4f}` / `{best.get('eval_lwir_to_mm5_aligned_t16_ncc_min', float('nan')):.4f}`",
            ]
        )

    lines.extend(
        [
            "",
            "## Candidate Table",
            "",
            "| candidate | method | LWIR NCC mean | LWIR NCC min | edge mean | valid mean |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for row in summary_rows:
        lines.append(
            "| {candidate} | {method} | {lncc:.4f} | {lmin:.4f} | {ledge:.4f} | {valid:.4f} |".format(
                candidate=row["candidate"],
                method=row.get("method", ""),
                lncc=row.get("eval_lwir_to_mm5_aligned_t16_ncc_mean", float("nan")),
                lmin=row.get("eval_lwir_to_mm5_aligned_t16_ncc_min", float("nan")),
                ledge=row.get("eval_lwir_to_mm5_aligned_t16_edge_distance_mean", float("nan")),
                valid=row.get("lwir_valid_ratio_mean", float("nan")),
            )
        )
    report_dir = output_dir / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "dl_p25_report_p25.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 25 depth-assisted LWIR refinement on top of Phase 24.")
    parser.add_argument("--index", default="mm5_calib_benchmark/outputs/mm5_benchmark/splits/index_with_splits.csv")
    parser.add_argument("--aligned-ids", default="106,104,103")
    parser.add_argument("--limit", type=int, default=3)
    parser.add_argument("--splits", default="test,val")
    parser.add_argument("--calibration", default="calibration/def_stereocalib_THERM.yml")
    parser.add_argument("--thermal-camera-calibration", default="calibration/def_thermalcam_ori.yml")
    parser.add_argument("--calibration-root", default="")
    parser.add_argument("--target-size", default="640x480")
    parser.add_argument("--max-board-offset-rmse-px", type=float, default=12.0)
    parser.add_argument("--point-margin-px", type=int, default=80)
    parser.add_argument("--ransac-threshold-px", type=float, default=4.0)
    parser.add_argument("--depth-translation-radius-px", type=int, default=2)
    parser.add_argument("--depth-foreground-threshold-mm", type=float, default=1000.0)
    parser.add_argument("--output", default="darklight_mm5/calibration_only_method/outputs_phase25_depth_assisted")
    parser.add_argument("--bridge-metrics", default="darklight_mm5/outputs/metrics/dl_ref_met_reg_stage.csv")
    parser.add_argument("--save-panels", action="store_true", default=True)
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metrics").mkdir(parents=True, exist_ok=True)
    (output_dir / "panels").mkdir(parents=True, exist_ok=True)
    (output_dir / "reports").mkdir(parents=True, exist_ok=True)
    target_size = parse_size_arg(args.target_size)
    if target_size is None:
        raise ValueError("--target-size must be WxH")

    rows = choose_rows(args.index, args.aligned_ids, args.limit, args.splits)
    if not rows:
        raise RuntimeError("no rows selected; check --aligned-ids/--splits and depth paths")

    calibration = load_stereo_calibration(args.calibration)
    rectification = read_rectification_matrices(args.calibration)
    thermal_model = read_single_camera_model(args.thermal_camera_calibration, "thermal_ori")
    if thermal_model is None:
        raise FileNotFoundError(args.thermal_camera_calibration)

    first_rgb = imread_unicode(rows[0].raw_rgb1_path, cv2.IMREAD_COLOR)
    first_lwir_u8 = normalize_u8(imread_unicode(rows[0].raw_thermal16_path, cv2.IMREAD_UNCHANGED))
    phase21_rule = make_phase21_ceiling_rule(first_rgb, first_lwir_u8, calibration, rectification, target_size, thermal_model)
    rgb_offset = phase21_rule.rgb_offset

    calibration_root = Path(args.calibration_root.strip() or default_calibration_root(args.index))
    offset_observations = collect_board_offsets(
        calibration_root,
        rgb_offset,
        thermal_model,
        rectification,
        args.max_board_offset_rmse_px,
    )
    offset_candidates = build_offset_candidates(offset_observations, phase21_rule.lwir_offset)
    promoted_offset = next((row["offset_xy"] for row in offset_candidates if row["candidate"] == "board_all_median_floor"), None)
    if promoted_offset is None:
        raise RuntimeError("board_all_median_floor offset was not produced")

    source_points, target_points, correspondence_rows = collect_board_correspondences(
        calibration_root,
        rgb_offset,
        promoted_offset,
        thermal_model,
        rectification,
        args.max_board_offset_rmse_px,
        args.point_margin_px,
    )
    if len(source_points) < 8:
        raise RuntimeError("not enough checkerboard correspondences to fit residual transforms")

    transforms = fit_transforms(source_points, target_points, args.ransac_threshold_px)
    promoted_transform = next((item for item in transforms if item.name == "affine_lmeds"), transforms[0])
    transform_rows = [
        {
            "candidate": item.name,
            "transform_type": item.transform_type,
            "inliers": item.inliers,
            "board_mae_px": item.board_mae_px,
            "board_rmse_px": item.board_rmse_px,
            "matrix": matrix_json(item.matrix),
            "source": item.source,
        }
        for item in transforms
    ]
    write_csv(output_dir / "metrics" / "dl_p25_board_pts.csv", correspondence_rows, collect_fieldnames(correspondence_rows, []))
    write_csv(output_dir / "metrics" / "dl_p25_board_tf.csv", transform_rows, collect_fieldnames(transform_rows, []))

    prepared_rows: list[dict] = []
    for row in rows:
        print(f"preparing {sample_id(row)} depth registration inputs")
        raw_rgb = imread_unicode(row.raw_rgb1_path, cv2.IMREAD_COLOR)
        raw_lwir_u8 = normalize_u8(imread_unicode(row.raw_thermal16_path, cv2.IMREAD_UNCHANGED))
        depth = imread_unicode(row.raw_depth_tr_path, cv2.IMREAD_UNCHANGED)
        if depth is None:
            raise FileNotFoundError(row.raw_depth_tr_path)

        phase24 = make_phase24_candidate(
            raw_rgb,
            raw_lwir_u8,
            rgb_offset,
            promoted_offset,
            promoted_transform,
            thermal_model,
            rectification,
            target_size,
        )
        phase24.name = "phase24_affine_lmeds_baseline"
        phase24.metadata["method"] = "phase24_affine_lwir_only"
        phase24.metadata["depth_used"] = False
        phase24.metadata["depth_valid_policy"] = "phase24_valid_mask"

        depth_projection = make_depth_projection_candidate(
            row,
            raw_rgb,
            raw_lwir_u8,
            depth,
            rgb_offset,
            target_size,
            calibration,
            thermal_model,
        )
        depth_boundary, depth_foreground, depth_debug = make_depth_foreground_boundary(
            depth.astype(np.float32),
            rgb_offset,
            target_size,
            args.depth_foreground_threshold_mm,
        )
        prepared_rows.append(
            {
                "row": row,
                "raw_rgb": raw_rgb,
                "raw_lwir_u8": raw_lwir_u8,
                "depth": depth,
                "phase24": phase24,
                "depth_projection": depth_projection,
                "depth_boundary": depth_boundary,
                "depth_foreground": depth_foreground,
                "depth_debug": depth_debug,
            }
        )

    depth_selection, depth_score_rows = select_global_depth_translation(
        prepared_rows,
        args.depth_translation_radius_px,
    )
    depth_selection["search_radius_px"] = int(args.depth_translation_radius_px)
    for row in depth_score_rows:
        row["selected_dx_px"] = int(depth_selection["dx_px"])
        row["selected_dy_px"] = int(depth_selection["dy_px"])
        row["selected_mean_depth_boundary_distance"] = float(depth_selection["mean_depth_boundary_distance"])
        row["baseline_mean_depth_boundary_distance"] = float(depth_selection["baseline_mean_depth_boundary_distance"])
    write_csv(output_dir / "metrics" / "dl_p25_score_p25.csv", depth_score_rows, collect_fieldnames(depth_score_rows, []))

    metric_rows: list[dict] = []
    candidate_cache: dict[tuple[str, str], CandidateOutput] = {}
    for item in prepared_rows:
        row = item["row"]
        print(f"processing {sample_id(row)} phase25 depth-assisted candidates")
        aligned_rgb = imread_unicode(row.aligned_rgb1_path, cv2.IMREAD_COLOR)
        aligned_lwir_u8 = normalize_u8(imread_unicode(row.aligned_t16_path, cv2.IMREAD_UNCHANGED))
        phase24 = item["phase24"]
        depth_projection = item["depth_projection"]
        depth_registered = make_depth_registered_candidate(
            phase24,
            depth_projection,
            depth_selection,
            fill_depth_border=False,
            depth_debug=item["depth_debug"],
        )
        depth_registered_fill = make_depth_registered_candidate(
            phase24,
            depth_projection,
            depth_selection,
            fill_depth_border=True,
            depth_debug=item["depth_debug"],
        )
        depth_keep = make_holefill_candidate(
            phase24,
            depth_projection,
            name="phase25_depth_holefill_keep_phase24_valid",
            method="phase24_affine_depth_border_holefill",
            use_union_valid=False,
            transform=promoted_transform,
        )
        depth_union = make_holefill_candidate(
            phase24,
            depth_projection,
            name="phase25_depth_holefill_union_valid",
            method="phase24_affine_depth_border_holefill_union_valid",
            use_union_valid=True,
            transform=promoted_transform,
        )
        median_control = make_median_control_candidate(phase24)

        candidates = [
            phase24,
            depth_registered,
            depth_registered_fill,
            depth_projection,
            depth_keep,
            depth_union,
            median_control,
        ]
        for candidate in candidates:
            metrics = add_phase25_metadata(evaluate_candidate(row, candidate, aligned_rgb, aligned_lwir_u8), candidate)
            metric_rows.append(metrics)
            candidate_cache[(sample_id(row), candidate.name)] = candidate

    summary_rows = summarize(metric_rows)
    augment_phase25_summary(summary_rows, metric_rows)
    write_csv(output_dir / "metrics" / "dl_p25_met_p25.csv", metric_rows, collect_fieldnames(metric_rows, []))
    write_csv(output_dir / "metrics" / "dl_p25_sum_p25.csv", summary_rows, collect_fieldnames(summary_rows, []))
    payload = {
        "constraint": "phase24_geometry_plus_raw_depth_boundary_registration_aligned_eval_only",
        "rows": [sample_id(row) for row in rows],
        "calibration_root": str(calibration_root),
        "rgb_offset_xy": list(rgb_offset),
        "lwir_offset_xy": list(promoted_offset),
        "correspondence_point_count": int(len(source_points)),
        "promoted_transform": next((row for row in transform_rows if row["candidate"] == promoted_transform.name), None),
        "depth_registration_selection": depth_selection,
        "best_evaluated_candidate": summary_rows[0] if summary_rows else None,
        "promoted_candidate": next(
            (row for row in summary_rows if row["candidate"] == "phase25_depth_registered_global_shift_depth_fill"),
            None,
        ),
        "candidate_summary": summary_rows,
    }
    (output_dir / "metrics" / "dl_p25_sum_p25.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    if args.save_panels:
        top_names = {row["candidate"] for row in summary_rows}
        for row in rows:
            aligned_rgb = imread_unicode(row.aligned_rgb1_path, cv2.IMREAD_COLOR)
            aligned_lwir_u8 = normalize_u8(imread_unicode(row.aligned_t16_path, cv2.IMREAD_UNCHANGED))
            for name in top_names:
                candidate = candidate_cache.get((sample_id(row), name))
                if candidate is not None:
                    save_panel(output_dir, row, candidate, aligned_rgb, aligned_lwir_u8)

    write_report(
        output_dir,
        summary_rows,
        transform_rows,
        load_bridge_metrics(args.bridge_metrics),
        len(source_points),
    )
    print(f"best phase25 candidate: {summary_rows[0]['candidate'] if summary_rows else 'none'}")
    print("done")


if __name__ == "__main__":
    main()
