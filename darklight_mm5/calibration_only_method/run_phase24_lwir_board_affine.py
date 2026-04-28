from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
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
    collect_fieldnames,
    imread_unicode,
    load_stereo_calibration,
    make_edge_overlay,
    make_five_panel,
    matrix_to_list,
    normalize_u8,
    parse_size_arg,
    write_csv,
)
from run_calibration_only import (  # noqa: E402
    CandidateOutput,
    crop_mask_with_border,
    crop_with_border,
    read_rectification_matrices,
    read_single_camera_model,
    rectified_remap,
    sample_id,
)
from diagnose_aligned_canvas import evaluate_candidate, load_bridge_metrics, tuple2  # noqa: E402
from run_phase21_canvas_optimization import corner_order_variants, detect_chessboard  # noqa: E402
from run_phase22_stereo_recalib import (  # noqa: E402
    add_metadata_to_metrics,
    choose_rows,
    default_calibration_root,
    make_phase21_ceiling_rule,
    summarize,
)
from run_phase23_lwir_board_offset import (  # noqa: E402
    build_offset_candidates,
    collect_board_offsets,
)


@dataclass
class BoardTransform:
    name: str
    matrix: np.ndarray
    transform_type: str
    inliers: int
    board_mae_px: float
    board_rmse_px: float
    source: str


def offset_json(offset: tuple[int, int]) -> str:
    return json.dumps([int(offset[0]), int(offset[1])])


def matrix_json(matrix: np.ndarray) -> str:
    return json.dumps(matrix_to_list(matrix.astype(np.float64)))


def collect_board_correspondences(
    calibration_root: Path,
    rgb_offset: tuple[int, int],
    lwir_offset: tuple[int, int],
    thermal_model,
    rectification,
    max_offset_residual_rmse_px: float,
    point_margin_px: int,
) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    capture_root = calibration_root / "capture_THERM" / "1280x720"
    if not capture_root.exists():
        return np.empty((0, 2), np.float32), np.empty((0, 2), np.float32), []

    rgb_offset_arr = np.asarray(rgb_offset, dtype=np.float64)
    lwir_offset_arr = np.asarray(lwir_offset, dtype=np.float64)
    source_points: list[np.ndarray] = []
    target_points: list[np.ndarray] = []
    rows: list[dict] = []

    for left_path in sorted(capture_root.rglob("*_left.png")):
        right_path = Path(str(left_path).replace("_left.png", "_right.png"))
        if not right_path.exists():
            continue
        left_img = imread_unicode(left_path, cv2.IMREAD_UNCHANGED)
        right_img = imread_unicode(right_path, cv2.IMREAD_UNCHANGED)
        if left_img is None or right_img is None:
            continue
        left_corners = detect_chessboard(left_img)
        right_corners = detect_chessboard(right_img)
        if left_corners is None or right_corners is None:
            continue

        best: tuple[float, str, np.ndarray, np.ndarray] | None = None
        for orientation, right_variant in corner_order_variants(right_corners):
            thermal_rectified = cv2.undistortPoints(
                right_variant.reshape(-1, 1, 2),
                thermal_model.K,
                thermal_model.D.reshape(-1, 1),
                R=rectification.R2,
                P=rectification.P2,
            ).reshape(-1, 2)
            per_corner_offset = thermal_rectified - left_corners + rgb_offset_arr
            median_offset = np.median(per_corner_offset, axis=0)
            residual = per_corner_offset - median_offset
            rmse = float(np.sqrt(np.mean(np.sum(residual * residual, axis=1))))
            if best is None or rmse < best[0]:
                best = (rmse, orientation, thermal_rectified, left_corners)

        if best is None or best[0] > max_offset_residual_rmse_px:
            continue

        src = best[2] - lwir_offset_arr
        dst = best[3] - rgb_offset_arr
        margin = float(point_margin_px)
        mask = (
            (src[:, 0] > -margin)
            & (src[:, 0] < 640 + margin)
            & (src[:, 1] > -margin)
            & (src[:, 1] < 480 + margin)
            & (dst[:, 0] > -margin)
            & (dst[:, 0] < 640 + margin)
            & (dst[:, 1] > -margin)
            & (dst[:, 1] < 480 + margin)
        )
        src_kept = src[mask].astype(np.float32)
        dst_kept = dst[mask].astype(np.float32)
        if len(src_kept) < 4:
            continue
        source_points.append(src_kept)
        target_points.append(dst_kept)
        rows.append(
            {
                "capture": left_path.relative_to(capture_root).as_posix(),
                "orientation": best[1],
                "offset_residual_rmse_px": best[0],
                "kept_points": int(len(src_kept)),
            }
        )

    if not source_points:
        return np.empty((0, 2), np.float32), np.empty((0, 2), np.float32), rows
    return np.vstack(source_points).astype(np.float32), np.vstack(target_points).astype(np.float32), rows


def board_error(transform: np.ndarray, source: np.ndarray, target: np.ndarray) -> tuple[float, float]:
    if transform.shape == (2, 3):
        pred = cv2.transform(source.reshape(-1, 1, 2), transform).reshape(-1, 2)
    else:
        pred = cv2.perspectiveTransform(source.reshape(-1, 1, 2), transform).reshape(-1, 2)
    err = np.linalg.norm(pred - target, axis=1)
    return float(np.mean(err)), float(np.sqrt(np.mean(err * err)))


def fit_transforms(source: np.ndarray, target: np.ndarray, ransac_threshold_px: float) -> list[BoardTransform]:
    transforms: list[BoardTransform] = [
        BoardTransform(
            name="phase23_crop_only",
            matrix=np.eye(2, 3, dtype=np.float64),
            transform_type="identity_affine",
            inliers=len(source),
            board_mae_px=board_error(np.eye(2, 3, dtype=np.float64), source, target)[0],
            board_rmse_px=board_error(np.eye(2, 3, dtype=np.float64), source, target)[1],
            source="Phase 23 crop only; no residual transform",
        )
    ]

    fit_specs = []
    A, inliers = cv2.estimateAffinePartial2D(
        source,
        target,
        method=cv2.RANSAC,
        ransacReprojThreshold=ransac_threshold_px,
        maxIters=5000,
        confidence=0.99,
        refineIters=20,
    )
    fit_specs.append(("similarity_ransac", "affine", A, inliers, f"estimateAffinePartial2D RANSAC {ransac_threshold_px}px"))
    A, inliers = cv2.estimateAffine2D(
        source,
        target,
        method=cv2.RANSAC,
        ransacReprojThreshold=ransac_threshold_px,
        maxIters=5000,
        confidence=0.99,
        refineIters=20,
    )
    fit_specs.append(("affine_ransac", "affine", A, inliers, f"estimateAffine2D RANSAC {ransac_threshold_px}px"))
    A, inliers = cv2.estimateAffinePartial2D(source, target, method=cv2.LMEDS, refineIters=20)
    fit_specs.append(("similarity_lmeds", "affine", A, inliers, "estimateAffinePartial2D LMEDS"))
    A, inliers = cv2.estimateAffine2D(source, target, method=cv2.LMEDS, refineIters=20)
    fit_specs.append(("affine_lmeds", "affine", A, inliers, "estimateAffine2D LMEDS"))
    H, inliers = cv2.findHomography(source, target, cv2.RANSAC, ransac_threshold_px)
    fit_specs.append(("homography_ransac", "homography", H, inliers, f"findHomography RANSAC {ransac_threshold_px}px"))

    for name, transform_type, matrix, inliers, source_desc in fit_specs:
        if matrix is None:
            continue
        matrix = matrix.astype(np.float64)
        mae, rmse = board_error(matrix, source, target)
        transforms.append(
            BoardTransform(
                name=name,
                matrix=matrix,
                transform_type=transform_type,
                inliers=int(inliers.sum()) if inliers is not None else 0,
                board_mae_px=mae,
                board_rmse_px=rmse,
                source=source_desc,
            )
        )
    transforms.sort(key=lambda item: item.board_rmse_px)
    return transforms


def warp_lwir_and_mask(lwir: np.ndarray, valid: np.ndarray, transform: BoardTransform, target_size: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    if transform.name == "phase23_crop_only":
        return lwir, valid
    width, height = target_size
    valid_u8 = valid.astype(np.uint8) * 255
    if transform.transform_type == "homography":
        warped = cv2.warpPerspective(lwir, transform.matrix, (width, height), flags=cv2.INTER_LINEAR, borderValue=0)
        warped_valid = cv2.warpPerspective(valid_u8, transform.matrix, (width, height), flags=cv2.INTER_NEAREST, borderValue=0) > 0
    else:
        warped = cv2.warpAffine(lwir, transform.matrix, (width, height), flags=cv2.INTER_LINEAR, borderValue=0)
        warped_valid = cv2.warpAffine(valid_u8, transform.matrix, (width, height), flags=cv2.INTER_NEAREST, borderValue=0) > 0
    return warped, warped_valid


def make_candidate(
    raw_rgb: np.ndarray,
    raw_lwir_u8: np.ndarray,
    rgb_offset: tuple[int, int],
    lwir_offset: tuple[int, int],
    transform: BoardTransform,
    thermal_model,
    rectification,
    target_size: tuple[int, int],
) -> CandidateOutput:
    rgb_valid_full = np.ones(raw_rgb.shape[:2], dtype=bool)
    rgb = crop_with_border(raw_rgb, rgb_offset, target_size)
    rgb_valid = crop_mask_with_border(rgb_valid_full, rgb_offset, target_size)
    lwir_crop, lwir_valid = rectified_remap(
        raw_lwir_u8,
        thermal_model.K,
        thermal_model.D,
        rectification.R2,
        rectification.P2,
        lwir_offset,
        target_size,
        cv2.INTER_LINEAR,
    )
    lwir, valid = warp_lwir_and_mask(lwir_crop, lwir_valid, transform, target_size)
    return CandidateOutput(
        name=transform.name,
        rgb=rgb,
        lwir=lwir,
        rgb_valid=rgb_valid,
        lwir_valid=valid,
        metadata={
            "method": "board_affine_lwir_only" if transform.name != "phase23_crop_only" else "board_offset_lwir_only",
            "allowed_for_generation": True,
            "rgb_source": "phase21_fixed_rgb_canvas",
            "lwir_source": "phase23_crop_plus_board_residual_transform",
            "rgb_crop_offset_xy": tuple2(rgb_offset),
            "lwir_crop_offset_xy": tuple2(lwir_offset),
            "lwir_crop_mode": "board_all_median_floor",
            "transform_type": transform.transform_type,
            "board_transform_name": transform.name,
            "board_transform_matrix": matrix_to_list(transform.matrix),
            "board_transform_inliers": transform.inliers,
            "board_transform_mae_px": transform.board_mae_px,
            "board_transform_rmse_px": transform.board_rmse_px,
            "rule_source": transform.source,
        },
    )


def save_panel(output_dir: Path, row, candidate: CandidateOutput, aligned_rgb: np.ndarray, aligned_lwir_u8: np.ndarray) -> None:
    aligned_lwir_bgr = cv2.cvtColor(aligned_lwir_u8, cv2.COLOR_GRAY2BGR)
    make_five_panel(
        [
            (candidate.rgb, "Fixed RGB"),
            (candidate.lwir, "Board-affine LWIR"),
            (aligned_rgb, "MM5 RGB eval"),
            (aligned_lwir_bgr, "MM5 T16 eval"),
            (make_edge_overlay(candidate.rgb, candidate.lwir, candidate.rgb_valid & candidate.lwir_valid), "Generated edge check"),
        ],
        output_dir / "panels" / f"{sample_id(row)}_{candidate.name}.png",
        tile_size=(360, 270),
    )


def write_report(
    output_dir: Path,
    summary_rows: list[dict],
    transform_rows: list[dict],
    bridge_metrics: dict,
    point_count: int,
) -> None:
    baseline = next((row for row in summary_rows if row["candidate"] == "phase23_crop_only"), None)
    promoted = next((row for row in summary_rows if row["candidate"] == "affine_lmeds"), None)
    best = summary_rows[0] if summary_rows else None
    best_board = transform_rows[0] if transform_rows else None

    lines = [
        "# Phase 24 LWIR Board-Affine Optimization",
        "",
        "## Constraint",
        "RGB is fixed to the Phase 21 calibration-derived canvas. LWIR uses the Phase 23 board-derived crop plus a residual transform fitted only from calibration-board corners. MM5 aligned images are used only for evaluation.",
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
            "## Board Residual Calibration",
            f"- checkerboard correspondence points: `{point_count}`",
        ]
    )
    if best_board:
        lines.append(
            f"- best board-residual transform by RMSE: `{best_board['candidate']}`, board RMSE `{float(best_board['board_rmse_px']):.4f}px`"
        )
    if baseline:
        lines.extend(
            [
                "",
                "## Phase 23 Baseline",
                f"- RGB NCC mean/min: `{baseline.get('eval_rgb_to_mm5_aligned_rgb_ncc_mean', float('nan')):.4f}` / `{baseline.get('eval_rgb_to_mm5_aligned_rgb_ncc_min', float('nan')):.4f}`",
                f"- LWIR NCC mean/min: `{baseline.get('eval_lwir_to_mm5_aligned_t16_ncc_mean', float('nan')):.4f}` / `{baseline.get('eval_lwir_to_mm5_aligned_t16_ncc_min', float('nan')):.4f}`",
            ]
        )
    if promoted:
        lines.extend(
            [
                "",
                "## Promoted Calibration-Only Candidate",
                f"- candidate: `{promoted['candidate']}`",
                f"- RGB NCC mean/min: `{promoted.get('eval_rgb_to_mm5_aligned_rgb_ncc_mean', float('nan')):.4f}` / `{promoted.get('eval_rgb_to_mm5_aligned_rgb_ncc_min', float('nan')):.4f}`",
                f"- LWIR NCC mean/min: `{promoted.get('eval_lwir_to_mm5_aligned_t16_ncc_mean', float('nan')):.4f}` / `{promoted.get('eval_lwir_to_mm5_aligned_t16_ncc_min', float('nan')):.4f}`",
                f"- board RMSE: `{promoted.get('board_transform_rmse_px', float('nan')):.4f}px`",
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
            "## Top Candidates",
            "",
            "| candidate | method | board RMSE | LWIR NCC mean | LWIR NCC min | RGB NCC mean |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for row in summary_rows:
        lines.append(
            "| {candidate} | {method} | {brmse:.4f} | {lncc:.4f} | {lmin:.4f} | {rncc:.4f} |".format(
                candidate=row["candidate"],
                method=row.get("method", ""),
                brmse=float(row.get("board_transform_rmse_px", float("nan"))),
                lncc=row.get("eval_lwir_to_mm5_aligned_t16_ncc_mean", float("nan")),
                lmin=row.get("eval_lwir_to_mm5_aligned_t16_ncc_min", float("nan")),
                rncc=row.get("eval_rgb_to_mm5_aligned_rgb_ncc_mean", float("nan")),
            )
        )
    (output_dir / "phase24_lwir_board_affine_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 24 LWIR-only board-derived residual affine/projective optimization.")
    parser.add_argument("--index", default="mm5_calib_benchmark/outputs/mm5_benchmark/splits/index_with_splits.csv")
    parser.add_argument("--aligned-ids", default="106,104,103")
    parser.add_argument("--calibration", default="calibration/def_stereocalib_THERM.yml")
    parser.add_argument("--thermal-camera-calibration", default="calibration/def_thermalcam_ori.yml")
    parser.add_argument("--calibration-root", default="")
    parser.add_argument("--target-size", default="640x480")
    parser.add_argument("--max-board-offset-rmse-px", type=float, default=12.0)
    parser.add_argument("--point-margin-px", type=int, default=80)
    parser.add_argument("--ransac-threshold-px", type=float, default=4.0)
    parser.add_argument("--output", default="darklight_mm5/calibration_only_method/outputs_phase24_lwir_board_affine")
    parser.add_argument("--bridge-metrics", default="darklight_mm5/outputs/metrics/registration_stages.csv")
    parser.add_argument("--save-top", type=int, default=6)
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    target_size = parse_size_arg(args.target_size)
    if target_size is None:
        raise ValueError("--target-size must be WxH")

    rows = choose_rows(args.index, args.aligned_ids)
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
    write_csv(output_dir / "metrics" / "phase24_board_correspondences.csv", correspondence_rows, collect_fieldnames(correspondence_rows, []))
    write_csv(output_dir / "metrics" / "phase24_board_transforms.csv", transform_rows, collect_fieldnames(transform_rows, []))

    metric_rows: list[dict] = []
    candidate_cache: dict[tuple[str, str], CandidateOutput] = {}
    for row in rows:
        print(f"processing {sample_id(row)} phase24 LWIR residual candidates")
        raw_rgb = imread_unicode(row.raw_rgb1_path, cv2.IMREAD_COLOR)
        raw_lwir_u8 = normalize_u8(imread_unicode(row.raw_thermal16_path, cv2.IMREAD_UNCHANGED))
        aligned_rgb = imread_unicode(row.aligned_rgb1_path, cv2.IMREAD_COLOR)
        aligned_lwir_u8 = normalize_u8(imread_unicode(row.aligned_t16_path, cv2.IMREAD_UNCHANGED))
        for transform in transforms:
            candidate = make_candidate(
                raw_rgb,
                raw_lwir_u8,
                rgb_offset,
                promoted_offset,
                transform,
                thermal_model,
                rectification,
                target_size,
            )
            metrics = add_metadata_to_metrics(evaluate_candidate(row, candidate, aligned_rgb, aligned_lwir_u8), candidate)
            metric_rows.append(metrics)
            candidate_cache[(sample_id(row), candidate.name)] = candidate

    summary_rows = summarize(metric_rows)
    transform_lookup = {row["candidate"]: row for row in transform_rows}
    for row in summary_rows:
        transform_row = transform_lookup.get(row["candidate"])
        if not transform_row:
            continue
        row["board_transform_inliers"] = transform_row["inliers"]
        row["board_transform_mae_px"] = transform_row["board_mae_px"]
        row["board_transform_rmse_px"] = transform_row["board_rmse_px"]
        row["board_transform_matrix"] = transform_row["matrix"]
    write_csv(output_dir / "metrics" / "phase24_lwir_board_affine_metrics.csv", metric_rows, collect_fieldnames(metric_rows, []))
    write_csv(output_dir / "metrics" / "phase24_lwir_board_affine_summary.csv", summary_rows, collect_fieldnames(summary_rows, []))
    payload = {
        "constraint": "fixed_phase21_rgb_phase23_lwir_crop_board_residual_transform_aligned_eval_only",
        "calibration_root": str(calibration_root),
        "rgb_offset_xy": list(rgb_offset),
        "lwir_offset_xy": list(promoted_offset),
        "correspondence_point_count": int(len(source_points)),
        "best_board_transform": transform_rows[0] if transform_rows else None,
        "best_evaluated_candidate": summary_rows[0] if summary_rows else None,
        "promoted_candidate": next((row for row in summary_rows if row["candidate"] == "affine_lmeds"), None),
        "candidate_summary": summary_rows,
    }
    (output_dir / "metrics" / "phase24_lwir_board_affine_summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    top_names = {row["candidate"] for row in summary_rows[: max(1, args.save_top)]}
    top_names.add("phase23_crop_only")
    top_names.add("affine_lmeds")
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
    print(f"best phase24 candidate: {summary_rows[0]['candidate'] if summary_rows else 'none'}")
    print("done")


if __name__ == "__main__":
    main()
