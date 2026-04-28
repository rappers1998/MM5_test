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
    direct_alignment_metrics,
    gray_u8,
    imread_unicode,
    load_rows,
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
    crop_offset_center,
    crop_offset_rectified_principal,
    crop_with_border,
    read_rectification_matrices,
    read_single_camera_model,
    rectified_remap,
    sample_id,
)
from diagnose_aligned_canvas import (  # noqa: E402
    CanvasRule,
    build_canvas_rules,
    build_lwir_models,
    clamp_crop_offset,
    crop_offset_from_bbox_center,
    evaluate_candidate,
    load_bridge_metrics,
    rectified_full,
    tuple2,
)
from run_phase21_canvas_optimization import (  # noqa: E402
    PATTERN_SIZE,
    corner_order_variants,
    detect_chessboard,
)


@dataclass
class PoseObservation:
    capture: str
    reprojection_mae_px: float
    orientation: str
    R_rgb_to_lwir: np.ndarray
    T_rgb_to_lwir: np.ndarray


@dataclass
class RectifySpec:
    name: str
    R2: np.ndarray
    P2: np.ndarray
    lwir_offset: tuple[int, int]
    metadata: dict


def parse_float_list(text: str) -> list[float]:
    return [float(part.strip()) for part in str(text).split(",") if part.strip()]


def parse_size_list(text: str) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for part in str(text).split(","):
        part = part.strip()
        if not part:
            continue
        size = parse_size_arg(part)
        if size is None:
            raise ValueError(f"rectify size must be explicit WxH: {part}")
        out.append(size)
    return out


def choose_rows(index: str, aligned_ids: str):
    rows = load_rows(index, require_official=False, require_depth=False)
    wanted = {int(x.strip()) for x in aligned_ids.split(",") if x.strip()}
    return sorted([row for row in rows if row.aligned_id in wanted], key=lambda row: row.aligned_id)


def default_calibration_root(index: str) -> str:
    with Path(index).open("r", encoding="utf-8-sig", newline="") as f:
        first = next(csv.DictReader(f))
    return str(first.get("calibration_root", "")).strip()


def object_points(square_size_mm: float) -> np.ndarray:
    cols, rows = PATTERN_SIZE
    grid_x, grid_y = np.meshgrid(np.arange(cols), np.arange(rows))
    obj = np.zeros((cols * rows, 3), dtype=np.float32)
    obj[:, 0] = (grid_x.reshape(-1) * square_size_mm).astype(np.float32)
    obj[:, 1] = (grid_y.reshape(-1) * square_size_mm).astype(np.float32)
    return obj


def average_rotations(rotations: list[np.ndarray]) -> np.ndarray:
    mean_r = np.mean(np.stack(rotations, axis=0), axis=0)
    u, _, vt = np.linalg.svd(mean_r)
    r = u @ vt
    if np.linalg.det(r) < 0:
        u[:, -1] *= -1
        r = u @ vt
    return r


def solve_board_pose(points: np.ndarray, obj: np.ndarray, K: np.ndarray, D: np.ndarray) -> tuple[bool, np.ndarray, np.ndarray, float]:
    ok, rvec, tvec = cv2.solvePnP(obj, points, K, D.reshape(-1, 1), flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        return False, np.eye(3, dtype=np.float64), np.zeros((3, 1), dtype=np.float64), float("inf")
    projected, _ = cv2.projectPoints(obj, rvec, tvec, K, D.reshape(-1, 1))
    err = np.linalg.norm(projected.reshape(-1, 2) - points, axis=1)
    R, _ = cv2.Rodrigues(rvec)
    return True, R.astype(np.float64), tvec.astype(np.float64), float(np.mean(err))


def collect_pose_observations(
    calibration_root: Path,
    rgb_K: np.ndarray,
    rgb_D: np.ndarray,
    lwir_K: np.ndarray,
    lwir_D: np.ndarray,
    square_size_mm: float,
    max_reprojection_mae_px: float,
) -> list[PoseObservation]:
    capture_root = calibration_root / "capture_THERM" / "1280x720"
    if not capture_root.exists():
        return []

    obj = object_points(square_size_mm)
    observations: list[PoseObservation] = []
    for left_path in sorted(capture_root.rglob("*_left.png")):
        right_path = Path(str(left_path).replace("_left.png", "_right.png"))
        if not right_path.exists():
            continue
        left_img = imread_unicode(left_path, cv2.IMREAD_UNCHANGED)
        right_img = imread_unicode(right_path, cv2.IMREAD_UNCHANGED)
        if left_img is None or right_img is None:
            continue
        left_points = detect_chessboard(left_img)
        right_points = detect_chessboard(right_img)
        if left_points is None or right_points is None:
            continue

        ok1, R1, t1, err1 = solve_board_pose(left_points, obj, rgb_K, rgb_D)
        if not ok1:
            continue

        best: tuple[float, str, np.ndarray, np.ndarray] | None = None
        for orientation, right_variant in corner_order_variants(right_points):
            ok2, R2, t2, err2 = solve_board_pose(right_variant, obj, lwir_K, lwir_D)
            if not ok2:
                continue
            rel_R = R2 @ R1.T
            rel_T = t2 - rel_R @ t1
            reproj = (err1 + err2) / 2.0
            if best is None or reproj < best[0]:
                best = (float(reproj), orientation, rel_R, rel_T.reshape(3))
        if best is None or best[0] > max_reprojection_mae_px:
            continue

        observations.append(
            PoseObservation(
                capture=left_path.relative_to(capture_root).as_posix(),
                reprojection_mae_px=best[0],
                orientation=best[1],
                R_rgb_to_lwir=best[2],
                T_rgb_to_lwir=best[3],
            )
        )
    observations.sort(key=lambda obs: obs.reprojection_mae_px)
    return observations


def pose_sets(observations: list[PoseObservation]) -> dict[str, list[PoseObservation]]:
    sets: dict[str, list[PoseObservation]] = {
        "epnp_all": observations,
        "epnp_best8": observations[:8],
        "epnp_best16": observations[:16],
        "epnp_near90_mixed": [
            obs for obs in observations if obs.capture.startswith("0.90m/") or obs.capture.startswith("mixed/01")
        ],
    }
    for group in ("0.30m", "0.50m", "0.70m", "0.90m"):
        items = [obs for obs in observations if obs.capture.startswith(f"{group}/")]
        if len(items) >= 2:
            sets[f"epnp_{group.replace('.', '_')}"] = items
    return {name: items for name, items in sets.items() if len(items) >= 2}


def aggregate_pose(items: list[PoseObservation]) -> tuple[np.ndarray, np.ndarray]:
    R = average_rotations([obs.R_rgb_to_lwir for obs in items])
    T = np.median(np.stack([obs.T_rgb_to_lwir for obs in items], axis=0), axis=0).reshape(3, 1)
    return R, T


def roi_center_offset(roi: tuple[int, int, int, int], target_size: tuple[int, int]) -> tuple[int, int] | None:
    x, y, w, h = [int(v) for v in roi]
    if w <= 0 or h <= 0:
        return None
    target_w, target_h = target_size
    return int(round(x + w / 2.0 - target_w / 2.0)), int(round(y + h / 2.0 - target_h / 2.0))


def alpha_label(alpha: float) -> str:
    if alpha < 0:
        return "neg1"
    return str(alpha).replace(".", "_")


def make_phase21_ceiling_rule(
    raw_rgb: np.ndarray,
    raw_lwir_u8: np.ndarray,
    calibration,
    rectification,
    target_size: tuple[int, int],
    thermal_model,
) -> CanvasRule:
    lwir_model = build_lwir_models(calibration, raw_rgb.shape, raw_lwir_u8.shape, [], thermal_model)[0]
    rules = build_canvas_rules(raw_rgb, raw_lwir_u8, calibration, rectification, target_size, lwir_model, [])
    rule_map = {rule.name: rule for rule in rules}
    rgb_rule = rule_map.get("rectified_intersection_x_rgb_optimal_alpha1_y") or rule_map["shared_rgb_optimal_alpha0_0"]
    lwir_rule = rule_map["shared_rgb_optimal_alpha0_0"]
    return CanvasRule(
        name="phase21_ceiling_current",
        rgb_offset=rgb_rule.rgb_offset,
        lwir_offset=lwir_rule.lwir_offset,
        rgb_source=rgb_rule.rgb_source,
        lwir_source=lwir_rule.lwir_source,
        allowed_for_generation=True,
        source=f"Phase 21 ceiling: RGB from {rgb_rule.name}; LWIR from {lwir_rule.name}",
    )


def make_candidate_from_rectify_spec(
    raw_rgb: np.ndarray,
    raw_lwir_u8: np.ndarray,
    rgb_offset: tuple[int, int],
    spec: RectifySpec,
    thermal_model,
    target_size: tuple[int, int],
) -> CandidateOutput:
    rgb_valid_full = np.ones(raw_rgb.shape[:2], dtype=bool)
    rgb = crop_with_border(raw_rgb, rgb_offset, target_size)
    rgb_valid = crop_mask_with_border(rgb_valid_full, rgb_offset, target_size)
    lwir, lwir_valid = rectified_remap(
        raw_lwir_u8,
        thermal_model.K,
        thermal_model.D,
        spec.R2,
        spec.P2,
        spec.lwir_offset,
        target_size,
        cv2.INTER_LINEAR,
    )
    return CandidateOutput(
        name=spec.name,
        rgb=rgb,
        lwir=lwir,
        rgb_valid=rgb_valid,
        lwir_valid=lwir_valid,
        metadata={
            "method": "epnp_reestimated_stereo_rectification",
            "allowed_for_generation": True,
            "rgb_source": "phase21_calibration_derived_raw_crop",
            "lwir_source": "reestimated_epnp_rectified_lwir",
            "rgb_crop_offset_xy": tuple2(rgb_offset),
            "lwir_crop_offset_xy": tuple2(spec.lwir_offset),
            **spec.metadata,
        },
    )


def unique_offsets(offsets: list[tuple[str, tuple[int, int] | None]]) -> list[tuple[str, tuple[int, int]]]:
    seen: set[tuple[int, int]] = set()
    out: list[tuple[str, tuple[int, int]]] = []
    for name, offset in offsets:
        if offset is None:
            continue
        key = (int(offset[0]), int(offset[1]))
        if key in seen:
            continue
        seen.add(key)
        out.append((name, key))
    return out


def build_rectify_specs(
    sets: dict[str, list[PoseObservation]],
    calibration,
    thermal_model,
    raw_lwir_u8: np.ndarray,
    target_size: tuple[int, int],
    phase21_rgb_offset: tuple[int, int],
    phase21_lwir_offset: tuple[int, int],
    rectify_sizes: list[tuple[int, int]],
    alphas: list[float],
) -> tuple[list[RectifySpec], list[dict]]:
    specs: list[RectifySpec] = []
    pose_rows: list[dict] = []
    flags_options = [(cv2.CALIB_ZERO_DISPARITY, "zero_disparity"), (0, "free_principal")]

    for set_name, items in sets.items():
        R, T = aggregate_pose(items)
        pose_rows.append(
            {
                "pose_set": set_name,
                "observations": len(items),
                "reprojection_mae_px_mean": float(np.mean([obs.reprojection_mae_px for obs in items])),
                "reprojection_mae_px_min": float(np.min([obs.reprojection_mae_px for obs in items])),
                "reprojection_mae_px_max": float(np.max([obs.reprojection_mae_px for obs in items])),
                "T_rgb_to_lwir": json.dumps([float(x) for x in T.reshape(-1)]),
                "R_rgb_to_lwir": json.dumps(matrix_to_list(R)),
            }
        )
        for rectify_size in rectify_sizes:
            size_label = f"{rectify_size[0]}x{rectify_size[1]}"
            for alpha in alphas:
                for flags, flag_label in flags_options:
                    try:
                        _R1, R2, _P1, P2, _Q, _roi1, roi2 = cv2.stereoRectify(
                            calibration.rgb_K,
                            calibration.rgb_D.reshape(-1, 1),
                            thermal_model.K,
                            thermal_model.D.reshape(-1, 1),
                            rectify_size,
                            R,
                            T,
                            flags=flags,
                            alpha=float(alpha),
                            newImageSize=rectify_size,
                        )
                    except cv2.error:
                        continue

                    valid_center = None
                    try:
                        _, valid_full = rectified_full(
                            raw_lwir_u8,
                            thermal_model.K,
                            thermal_model.D,
                            R2,
                            P2,
                            rectify_size,
                            cv2.INTER_NEAREST,
                        )
                        valid_center = crop_offset_from_bbox_center(valid_full, target_size)
                        if valid_center is not None:
                            valid_center = clamp_crop_offset(valid_center, rectify_size, target_size)
                    except cv2.error:
                        pass

                    center = crop_offset_center((rectify_size[1], rectify_size[0]), target_size)
                    roi2_center = roi_center_offset(roi2, target_size)
                    if roi2_center is not None:
                        roi2_center = clamp_crop_offset(roi2_center, rectify_size, target_size)
                    offsets = unique_offsets(
                        [
                            ("phase21_lwir_alpha0", phase21_lwir_offset),
                            ("phase21_rgb_canvas", phase21_rgb_offset),
                            ("rectify_canvas_center", center),
                            ("p2_principal", crop_offset_rectified_principal(P2, target_size)),
                            ("roi2_center", roi2_center),
                            ("valid_bbox_center", valid_center),
                        ]
                    )
                    for offset_name, offset in offsets:
                        name = f"{set_name}_{size_label}_a{alpha_label(alpha)}_{flag_label}_{offset_name}"
                        specs.append(
                            RectifySpec(
                                name=name,
                                R2=R2,
                                P2=P2,
                                lwir_offset=offset,
                                metadata={
                                    "pose_set": set_name,
                                    "pose_observation_count": len(items),
                                    "rectify_size": list(rectify_size),
                                    "stereo_rectify_alpha": float(alpha),
                                    "stereo_rectify_flags": flag_label,
                                    "lwir_crop_mode": offset_name,
                                    "reprojection_mae_px_mean": float(
                                        np.mean([obs.reprojection_mae_px for obs in items])
                                    ),
                                },
                            )
                        )
    return specs, pose_rows


def summarize(rows: list[dict]) -> list[dict]:
    grouped: dict[str, list[dict]] = {}
    for row in rows:
        grouped.setdefault(str(row["candidate"]), []).append(row)
    summary: list[dict] = []
    numeric_keys = [
        "rgb_valid_ratio",
        "lwir_valid_ratio",
        "intersection_valid_ratio",
        "eval_rgb_to_mm5_aligned_rgb_ncc",
        "eval_lwir_to_mm5_aligned_t16_ncc",
        "eval_lwir_to_mm5_aligned_t16_edge_distance",
        "eval_cross_rgb_lwir_ncc",
    ]
    for name, items in grouped.items():
        out = {
            "candidate": name,
            "samples": len(items),
            "method": items[0].get("method", ""),
            "lwir_crop_offset_xy": items[0].get("lwir_crop_offset_xy", ""),
            "rule_source": items[0].get("rule_source", ""),
            "pose_set": items[0].get("pose_set", ""),
            "rectify_size": items[0].get("rectify_size", ""),
            "stereo_rectify_alpha": items[0].get("stereo_rectify_alpha", ""),
            "stereo_rectify_flags": items[0].get("stereo_rectify_flags", ""),
            "lwir_crop_mode": items[0].get("lwir_crop_mode", ""),
        }
        for key in numeric_keys:
            vals = [float(item[key]) for item in items if key in item and np.isfinite(float(item[key]))]
            out[f"{key}_mean"] = float(np.mean(vals)) if vals else float("nan")
            out[f"{key}_min"] = float(np.min(vals)) if vals else float("nan")
        summary.append(out)
    summary.sort(
        key=lambda row: (
            float(row.get("eval_lwir_to_mm5_aligned_t16_ncc_mean", float("nan")))
            if np.isfinite(float(row.get("eval_lwir_to_mm5_aligned_t16_ncc_mean", float("nan"))))
            else -1.0,
            float(row.get("eval_rgb_to_mm5_aligned_rgb_ncc_mean", float("nan")))
            if np.isfinite(float(row.get("eval_rgb_to_mm5_aligned_rgb_ncc_mean", float("nan"))))
            else -1.0,
        ),
        reverse=True,
    )
    return summary


def add_metadata_to_metrics(metrics: dict, candidate: CandidateOutput) -> dict:
    out = dict(metrics)
    for key in (
        "method",
        "rule_source",
        "pose_set",
        "rectify_size",
        "stereo_rectify_alpha",
        "stereo_rectify_flags",
        "lwir_crop_mode",
        "lwir_crop_offset_xy",
        "transform_type",
        "board_transform_name",
        "board_transform_inliers",
        "board_transform_mae_px",
        "board_transform_rmse_px",
        "board_transform_matrix",
    ):
        value = candidate.metadata.get(key)
        if value is None:
            continue
        out[key] = json.dumps(value) if isinstance(value, (list, tuple, dict)) else value
    return out


def save_panel(output_dir: Path, row, candidate: CandidateOutput, aligned_rgb: np.ndarray, aligned_lwir_u8: np.ndarray) -> None:
    aligned_lwir_bgr = cv2.cvtColor(aligned_lwir_u8, cv2.COLOR_GRAY2BGR)
    make_five_panel(
        [
            (candidate.rgb, "Generated RGB"),
            (candidate.lwir, "Generated LWIR"),
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
    pose_observations: list[PoseObservation],
    pose_set_rows: list[dict],
    bridge_metrics: dict,
    calibration_files: list[Path],
) -> None:
    baseline = next((row for row in summary_rows if row["candidate"] == "phase21_ceiling_current"), None)
    reestimated = [
        row
        for row in summary_rows
        if row.get("method") == "epnp_reestimated_stereo_rectification" or str(row.get("candidate", "")).startswith("epnp_")
    ]
    best_reestimated = reestimated[0] if reestimated else None
    best = summary_rows[0] if summary_rows else None
    metadata_names = ", ".join(path.name for path in calibration_files)

    lines = [
        "# Phase 22 Stereo Recalibration Probe",
        "",
        "## Constraint",
        "Generation candidates use calibration files, original calibration-board captures, output geometry, and raw inputs only. MM5 aligned images are used only for evaluation.",
        "",
        "## Calibration Inputs Checked",
        f"- files: `{metadata_names}`",
        "- no extra aligned-generation crop/canvas metadata was found beyond these calibration files and checkerboard captures.",
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
            "## Pose Observations",
            f"- accepted checkerboard pose observations: `{len(pose_observations)}`",
            f"- pose sets evaluated: `{len(pose_set_rows)}`",
        ]
    )
    if pose_observations:
        lines.append(f"- best board reprojection MAE: `{pose_observations[0].reprojection_mae_px:.4f}px`")

    lines.extend(["", "## Phase 21 Ceiling"])
    if baseline:
        lines.extend(
            [
                f"- candidate: `{baseline['candidate']}`",
                f"- RGB NCC mean/min: `{baseline.get('eval_rgb_to_mm5_aligned_rgb_ncc_mean', float('nan')):.4f}` / `{baseline.get('eval_rgb_to_mm5_aligned_rgb_ncc_min', float('nan')):.4f}`",
                f"- LWIR NCC mean/min: `{baseline.get('eval_lwir_to_mm5_aligned_t16_ncc_mean', float('nan')):.4f}` / `{baseline.get('eval_lwir_to_mm5_aligned_t16_ncc_min', float('nan')):.4f}`",
            ]
        )

    lines.extend(["", "## Best Re-Estimated Stereo Candidate"])
    if best_reestimated:
        lines.extend(
            [
                f"- candidate: `{best_reestimated['candidate']}`",
                f"- LWIR NCC mean/min: `{best_reestimated.get('eval_lwir_to_mm5_aligned_t16_ncc_mean', float('nan')):.4f}` / `{best_reestimated.get('eval_lwir_to_mm5_aligned_t16_ncc_min', float('nan')):.4f}`",
                f"- RGB NCC mean: `{best_reestimated.get('eval_rgb_to_mm5_aligned_rgb_ncc_mean', float('nan')):.4f}`",
                f"- crop mode: `{best_reestimated.get('lwir_crop_mode', '')}`",
            ]
        )
        if baseline and float(best_reestimated.get("eval_lwir_to_mm5_aligned_t16_ncc_mean", -1)) < float(
            baseline.get("eval_lwir_to_mm5_aligned_t16_ncc_mean", -1)
        ):
            lines.append("- result: re-estimating stereo from checkerboard captures does not beat the Phase 21 ceiling.")

    lines.extend(["", "## Current Best Overall"])
    if best:
        lines.append(
            f"- `{best['candidate']}` with LWIR NCC mean `{best.get('eval_lwir_to_mm5_aligned_t16_ncc_mean', float('nan')):.4f}`."
        )

    lines.extend(
        [
            "",
            "## Top Candidates",
            "",
            "| candidate | method | LWIR NCC mean | LWIR NCC min | RGB NCC mean | crop mode |",
            "|---|---|---:|---:|---:|---|",
        ]
    )
    for row in summary_rows[:12]:
        lines.append(
            "| {candidate} | {method} | {lncc:.4f} | {lmin:.4f} | {rncc:.4f} | {crop} |".format(
                candidate=row["candidate"],
                method=row.get("method", ""),
                lncc=row.get("eval_lwir_to_mm5_aligned_t16_ncc_mean", float("nan")),
                lmin=row.get("eval_lwir_to_mm5_aligned_t16_ncc_min", float("nan")),
                rncc=row.get("eval_rgb_to_mm5_aligned_rgb_ncc_mean", float("nan")),
                crop=row.get("lwir_crop_mode", ""),
            )
        )
    (output_dir / "phase22_stereo_recalib_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 22 strict calibration-only stereo recalibration probe.")
    parser.add_argument("--index", default="mm5_calib_benchmark/outputs/mm5_benchmark/splits/index_with_splits.csv")
    parser.add_argument("--aligned-ids", default="106,104,103")
    parser.add_argument("--calibration", default="calibration/def_stereocalib_THERM.yml")
    parser.add_argument("--thermal-camera-calibration", default="calibration/def_thermalcam_ori.yml")
    parser.add_argument("--calibration-root", default="")
    parser.add_argument("--target-size", default="640x480")
    parser.add_argument("--rectify-sizes", default="640x512,1280x720")
    parser.add_argument("--alphas", default="0,0.5,1")
    parser.add_argument("--square-size-mm", type=float, default=25.0)
    parser.add_argument("--max-reprojection-mae-px", type=float, default=1.5)
    parser.add_argument("--output", default="darklight_mm5/calibration_only_method/outputs_phase22_stereo_recalib")
    parser.add_argument("--bridge-metrics", default="darklight_mm5/outputs/metrics/registration_stages.csv")
    parser.add_argument("--save-top", type=int, default=4)
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

    calibration_root = Path(args.calibration_root.strip() or default_calibration_root(args.index))
    pose_observations = collect_pose_observations(
        calibration_root,
        calibration.rgb_K,
        calibration.rgb_D,
        thermal_model.K,
        thermal_model.D,
        args.square_size_mm,
        args.max_reprojection_mae_px,
    )
    sets = pose_sets(pose_observations)

    first_row = rows[0]
    first_rgb = imread_unicode(first_row.raw_rgb1_path, cv2.IMREAD_COLOR)
    first_lwir_u8 = normalize_u8(imread_unicode(first_row.raw_thermal16_path, cv2.IMREAD_UNCHANGED))
    phase21_rule = make_phase21_ceiling_rule(first_rgb, first_lwir_u8, calibration, rectification, target_size, thermal_model)
    rectify_specs, pose_set_rows = build_rectify_specs(
        sets,
        calibration,
        thermal_model,
        first_lwir_u8,
        target_size,
        phase21_rule.rgb_offset,
        phase21_rule.lwir_offset,
        parse_size_list(args.rectify_sizes),
        parse_float_list(args.alphas),
    )

    observation_rows = [
        {
            "capture": obs.capture,
            "reprojection_mae_px": obs.reprojection_mae_px,
            "orientation": obs.orientation,
            "T_rgb_to_lwir": json.dumps([float(x) for x in obs.T_rgb_to_lwir.reshape(-1)]),
            "R_rgb_to_lwir": json.dumps(matrix_to_list(obs.R_rgb_to_lwir)),
        }
        for obs in pose_observations
    ]
    write_csv(output_dir / "metrics" / "phase22_pose_observations.csv", observation_rows, collect_fieldnames(observation_rows, []))
    write_csv(output_dir / "metrics" / "phase22_pose_sets.csv", pose_set_rows, collect_fieldnames(pose_set_rows, []))

    metric_rows: list[dict] = []
    for row in rows:
        print(f"processing {sample_id(row)} phase22 candidates")
        raw_rgb = imread_unicode(row.raw_rgb1_path, cv2.IMREAD_COLOR)
        raw_lwir_u8 = normalize_u8(imread_unicode(row.raw_thermal16_path, cv2.IMREAD_UNCHANGED))
        aligned_rgb = imread_unicode(row.aligned_rgb1_path, cv2.IMREAD_COLOR)
        aligned_lwir_u8 = normalize_u8(imread_unicode(row.aligned_t16_path, cv2.IMREAD_UNCHANGED))

        sample_phase21_rule = make_phase21_ceiling_rule(raw_rgb, raw_lwir_u8, calibration, rectification, target_size, thermal_model)
        lwir_model = build_lwir_models(calibration, raw_rgb.shape, raw_lwir_u8.shape, [], thermal_model)[0]
        baseline = CandidateOutput(
            name=sample_phase21_rule.name,
            rgb=crop_with_border(raw_rgb, sample_phase21_rule.rgb_offset, target_size),
            lwir=rectified_remap(
                raw_lwir_u8,
                thermal_model.K,
                thermal_model.D,
                rectification.R2,
                rectification.P2,
                sample_phase21_rule.lwir_offset,
                target_size,
                cv2.INTER_LINEAR,
            )[0],
            rgb_valid=crop_mask_with_border(np.ones(raw_rgb.shape[:2], dtype=bool), sample_phase21_rule.rgb_offset, target_size),
            lwir_valid=rectified_remap(
                raw_lwir_u8,
                thermal_model.K,
                thermal_model.D,
                rectification.R2,
                rectification.P2,
                sample_phase21_rule.lwir_offset,
                target_size,
                cv2.INTER_NEAREST,
            )[1],
            metadata={
                "method": "phase21_strict_calibration_ceiling",
                "allowed_for_generation": True,
                "lwir_model": lwir_model.label,
                "rgb_crop_offset_xy": tuple2(sample_phase21_rule.rgb_offset),
                "lwir_crop_offset_xy": tuple2(sample_phase21_rule.lwir_offset),
                "rule_source": sample_phase21_rule.source,
            },
        )
        metric_rows.append(add_metadata_to_metrics(evaluate_candidate(row, baseline, aligned_rgb, aligned_lwir_u8), baseline))

        for spec in rectify_specs:
            try:
                candidate = make_candidate_from_rectify_spec(
                    raw_rgb,
                    raw_lwir_u8,
                    sample_phase21_rule.rgb_offset,
                    spec,
                    thermal_model,
                    target_size,
                )
            except cv2.error:
                continue
            metric_rows.append(add_metadata_to_metrics(evaluate_candidate(row, candidate, aligned_rgb, aligned_lwir_u8), candidate))

    summary_rows = summarize(metric_rows)
    write_csv(output_dir / "metrics" / "phase22_stereo_recalib_candidates.csv", metric_rows, collect_fieldnames(metric_rows, []))
    write_csv(output_dir / "metrics" / "phase22_stereo_recalib_summary.csv", summary_rows, collect_fieldnames(summary_rows, []))

    calibration_files = sorted(Path("calibration").glob("*"))
    payload = {
        "constraint": "calibration_board_and_raw_only_generation_aligned_eval_only",
        "calibration_root": str(calibration_root),
        "pose_observation_count": len(pose_observations),
        "pose_set_count": len(pose_set_rows),
        "rectify_candidate_count": len(rectify_specs),
        "best_candidate": summary_rows[0] if summary_rows else None,
        "candidate_summary": summary_rows,
    }
    (output_dir / "metrics" / "phase22_stereo_recalib_summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    top_names = {row["candidate"] for row in summary_rows[: max(args.save_top, 1)]}
    top_names.add("phase21_ceiling_current")
    for row in rows:
        raw_rgb = imread_unicode(row.raw_rgb1_path, cv2.IMREAD_COLOR)
        raw_lwir_u8 = normalize_u8(imread_unicode(row.raw_thermal16_path, cv2.IMREAD_UNCHANGED))
        aligned_rgb = imread_unicode(row.aligned_rgb1_path, cv2.IMREAD_COLOR)
        aligned_lwir_u8 = normalize_u8(imread_unicode(row.aligned_t16_path, cv2.IMREAD_UNCHANGED))
        sample_phase21_rule = make_phase21_ceiling_rule(raw_rgb, raw_lwir_u8, calibration, rectification, target_size, thermal_model)
        for spec in rectify_specs:
            if spec.name not in top_names:
                continue
            try:
                candidate = make_candidate_from_rectify_spec(
                    raw_rgb,
                    raw_lwir_u8,
                    sample_phase21_rule.rgb_offset,
                    spec,
                    thermal_model,
                    target_size,
                )
            except cv2.error:
                continue
            save_panel(output_dir, row, candidate, aligned_rgb, aligned_lwir_u8)

    write_report(
        output_dir,
        summary_rows,
        pose_observations,
        pose_set_rows,
        load_bridge_metrics(args.bridge_metrics),
        calibration_files,
    )
    print(f"best phase22 candidate: {summary_rows[0]['candidate'] if summary_rows else 'none'}")
    print("done")


if __name__ == "__main__":
    main()
