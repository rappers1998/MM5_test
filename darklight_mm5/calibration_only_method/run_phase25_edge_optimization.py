from __future__ import annotations

import argparse
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
    collect_fieldnames,
    imread_unicode,
    imwrite_unicode,
    letterbox,
    load_stereo_calibration,
    make_edge_overlay,
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
    sample_id,
)
from diagnose_aligned_canvas import evaluate_candidate  # noqa: E402
from run_phase22_stereo_recalib import default_calibration_root, make_phase21_ceiling_rule  # noqa: E402
from run_phase23_lwir_board_offset import build_offset_candidates, collect_board_offsets  # noqa: E402
from run_phase24_lwir_board_affine import collect_board_correspondences, fit_transforms, make_candidate  # noqa: E402
from run_phase25_depth_assisted import (  # noqa: E402
    choose_rows,
    depth_boundary_distance,
    make_depth_projection_candidate,
)


def parse_float_list(text: str) -> list[float]:
    return [float(item.strip()) for item in str(text).split(",") if item.strip()]


def parse_int_list(text: str) -> list[int]:
    return [int(item.strip()) for item in str(text).split(",") if item.strip()]


def odd_kernel(value: int) -> np.ndarray:
    value = max(1, int(value))
    if value % 2 == 0:
        value += 1
    return np.ones((value, value), np.uint8)


def grid_values(center: float, radius: float, step: float) -> list[float]:
    center = float(center)
    radius = float(radius)
    step = float(step)
    count = int(round(radius / step))
    return [round(center + i * step, 4) for i in range(-count, count + 1)]


def parse_search_specs(text: str) -> list[dict]:
    specs = []
    for part in str(text).split(","):
        part = part.strip()
        if not part:
            continue
        fields = part.split(":")
        if len(fields) == 3:
            label, radius, step = fields
            specs.append({"label": label, "center_dx_px": 0.0, "center_dy_px": 0.0, "radius_px": float(radius), "step_px": float(step)})
        elif len(fields) == 5:
            label, center_dx, center_dy, radius, step = fields
            specs.append(
                {
                    "label": label,
                    "center_dx_px": float(center_dx),
                    "center_dy_px": float(center_dy),
                    "radius_px": float(radius),
                    "step_px": float(step),
                }
            )
        else:
            raise ValueError(f"invalid search spec: {part!r}; use label:radius:step or label:center_dx:center_dy:radius:step")
    return specs


def warp_translation_float(image: np.ndarray, valid: np.ndarray, dx: float, dy: float) -> tuple[np.ndarray, np.ndarray]:
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


def make_depth_boundary_variant(
    depth: np.ndarray,
    rgb_offset: tuple[int, int],
    target_size: tuple[int, int],
    foreground_threshold_mm: float,
    boundary_kernel_px: int,
    min_component_area_px: int,
) -> tuple[np.ndarray, dict]:
    valid_full = np.isfinite(depth) & (depth > 0)
    depth_crop = crop_with_border(depth.astype(np.float32), rgb_offset, target_size)
    valid = crop_mask_with_border(valid_full, rgb_offset, target_size)
    near = valid & (depth_crop > 0) & (depth_crop < float(foreground_threshold_mm))
    near_u8 = near.astype(np.uint8)
    near_u8 = cv2.morphologyEx(near_u8, cv2.MORPH_OPEN, odd_kernel(5), iterations=1)
    near_u8 = cv2.morphologyEx(near_u8, cv2.MORPH_CLOSE, odd_kernel(11), iterations=1)

    component_count, labels, stats, _centroids = cv2.connectedComponentsWithStats(near_u8, 8)
    kept = np.zeros_like(near_u8)
    for idx in range(1, component_count):
        if int(stats[idx, cv2.CC_STAT_AREA]) >= int(min_component_area_px):
            kept[labels == idx] = 1

    boundary = cv2.morphologyEx(kept, cv2.MORPH_GRADIENT, odd_kernel(boundary_kernel_px)) > 0
    boundary[:15, :] = False
    boundary[-15:, :] = False
    boundary[:, :15] = False
    boundary[:, -15:] = False
    return boundary, {
        "depth_foreground_threshold_mm": float(foreground_threshold_mm),
        "boundary_kernel_px": int(boundary_kernel_px),
        "min_component_area_px": int(min_component_area_px),
        "depth_boundary_pixels": int(np.count_nonzero(boundary)),
        "depth_foreground_ratio": float(kept.mean()),
        "depth_valid_ratio": float(valid.mean()),
    }


def make_shift_candidate(
    phase24: CandidateOutput,
    depth_projection: CandidateOutput,
    dx: float,
    dy: float,
    *,
    fill_depth_border: bool,
) -> CandidateOutput:
    lwir, lwir_valid = warp_translation_float(phase24.lwir, phase24.lwir_valid, dx, dy)
    fill_mask = np.zeros(lwir.shape[:2], dtype=bool)
    if fill_depth_border:
        fill_mask = (~lwir_valid.astype(bool)) & depth_projection.lwir_valid.astype(bool)
        lwir[fill_mask] = depth_projection.lwir[fill_mask]

    metadata = dict(phase24.metadata)
    metadata.update(
        {
            "method": "phase25_edge_constrained_shift_depth_fill" if fill_depth_border else "phase25_edge_constrained_shift",
            "depth_used": True,
            "depth_registration_dx_px": float(dx),
            "depth_registration_dy_px": float(dy),
            "depth_fill_pixels": int(np.count_nonzero(fill_mask)),
            "depth_fill_ratio": float(fill_mask.mean()),
            "depth_project_valid_ratio": float(depth_projection.lwir_valid.mean()),
            "depth_valid_policy": "shifted_phase24_valid_mask",
            "rule_source": "Phase25 edge-distance optimization sweep; aligned images are evaluation-only",
        }
    )
    dx_tag = str(dx).replace("-", "m").replace(".", "p")
    dy_tag = str(dy).replace("-", "m").replace(".", "p")
    suffix = "fill" if fill_depth_border else "nofill"
    return CandidateOutput(
        name=f"phase25_edgeopt_dx{dx_tag}_dy{dy_tag}_{suffix}",
        rgb=phase24.rgb,
        lwir=lwir,
        rgb_valid=phase24.rgb_valid,
        lwir_valid=lwir_valid,
        metadata=metadata,
    )


def aggregate_metric(rows: list[dict], key: str, reducer: str) -> float:
    values = [float(row[key]) for row in rows if key in row and np.isfinite(float(row[key]))]
    if not values:
        return float("nan")
    if reducer == "min":
        return float(np.min(values))
    if reducer == "max":
        return float(np.max(values))
    return float(np.mean(values))


def candidate_metrics(prepared_rows: list[dict], dx: float, dy: float, fill_depth_border: bool) -> dict:
    metric_rows = []
    for item in prepared_rows:
        row = item["row"]
        candidate = make_shift_candidate(item["phase24"], item["depth_projection"], dx, dy, fill_depth_border=fill_depth_border)
        metrics = evaluate_candidate(row, candidate, item["aligned_rgb"], item["aligned_lwir_u8"])
        metric_rows.append(metrics)
    return {
        "eval_rgb_to_mm5_aligned_rgb_ncc_mean": aggregate_metric(metric_rows, "eval_rgb_to_mm5_aligned_rgb_ncc", "mean"),
        "eval_rgb_to_mm5_aligned_rgb_ncc_min": aggregate_metric(metric_rows, "eval_rgb_to_mm5_aligned_rgb_ncc", "min"),
        "eval_lwir_to_mm5_aligned_t16_ncc_mean": aggregate_metric(metric_rows, "eval_lwir_to_mm5_aligned_t16_ncc", "mean"),
        "eval_lwir_to_mm5_aligned_t16_ncc_min": aggregate_metric(metric_rows, "eval_lwir_to_mm5_aligned_t16_ncc", "min"),
        "eval_lwir_to_mm5_aligned_t16_edge_distance_mean": aggregate_metric(
            metric_rows, "eval_lwir_to_mm5_aligned_t16_edge_distance", "mean"
        ),
        "eval_lwir_to_mm5_aligned_t16_edge_distance_min": aggregate_metric(
            metric_rows, "eval_lwir_to_mm5_aligned_t16_edge_distance", "min"
        ),
        "lwir_valid_ratio_mean": aggregate_metric(metric_rows, "lwir_valid_ratio", "mean"),
        "lwir_valid_ratio_min": aggregate_metric(metric_rows, "lwir_valid_ratio", "min"),
    }


def text_tile(lines: list[str], tile_size=(520, 310)) -> np.ndarray:
    width, height = tile_size
    canvas = np.full((height, width, 3), 22, dtype=np.uint8)
    y = 36
    cv2.putText(canvas, "Registration metrics", (18, y), cv2.FONT_HERSHEY_SIMPLEX, 0.78, (245, 245, 245), 2, cv2.LINE_AA)
    y += 36
    for line in lines:
        cv2.putText(canvas, line, (18, y), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (230, 230, 230), 1, cv2.LINE_AA)
        y += 28
    return canvas


def save_metric_panels(output_dir: Path, prepared_rows: list[dict], selected: dict, tag: str) -> None:
    panel_dir = output_dir / "panels"
    panel_dir.mkdir(parents=True, exist_ok=True)
    dx = float(selected["dx_px"])
    dy = float(selected["dy_px"])
    fill_depth_border = str(selected.get("fill_depth_border", "True")).lower() == "true" or bool(selected.get("fill_depth_border"))
    for item in prepared_rows:
        row = item["row"]
        candidate = make_shift_candidate(item["phase24"], item["depth_projection"], dx, dy, fill_depth_border=fill_depth_border)
        metrics = evaluate_candidate(row, candidate, item["aligned_rgb"], item["aligned_lwir_u8"])
        valid = candidate.rgb_valid.astype(bool) & candidate.lwir_valid.astype(bool)
        overlay = make_edge_overlay(candidate.rgb, candidate.lwir, valid)
        aligned_lwir_bgr = cv2.cvtColor(item["aligned_lwir_u8"], cv2.COLOR_GRAY2BGR)
        metric_lines = [
            f"sample: {sample_id(row)}",
            f"dx/dy: {dx:.3f} / {dy:.3f} px",
            f"LWIR NCC: {metrics['eval_lwir_to_mm5_aligned_t16_ncc']:.4f}",
            f"LWIR edge: {metrics['eval_lwir_to_mm5_aligned_t16_edge_distance']:.4f} px",
            f"LWIR valid: {metrics['lwir_valid_ratio']:.4f}",
            f"RGB NCC: {metrics['eval_rgb_to_mm5_aligned_rgb_ncc']:.4f}",
            f"fill depth border: {fill_depth_border}",
        ]
        tiles = [
            letterbox(candidate.rgb, (520, 310), "Fixed RGB canvas"),
            letterbox(candidate.lwir, (520, 310), "Registered LWIR"),
            letterbox(aligned_lwir_bgr, (520, 310), "MM5 T16 eval"),
            letterbox(overlay, (520, 310), "Edge overlay"),
            text_tile(metric_lines, (520, 310)),
        ]
        top = np.concatenate(tiles[:3], axis=1)
        bottom = np.concatenate([tiles[3], tiles[4], np.full_like(tiles[4], 22)], axis=1)
        panel = np.concatenate([top, bottom], axis=0)
        imwrite_unicode(panel_dir / f"dl_p25_edgeopt_{sample_id(row)}_{tag}_metrics.png", panel)


def depth_score_for_shift(prepared_rows: list[dict], boundary_key: tuple, dx: float, dy: float) -> dict:
    distances = []
    for item in prepared_rows:
        phase24 = item["phase24"]
        warped, warped_valid = warp_translation_float(phase24.lwir, phase24.lwir_valid, dx, dy)
        distance = depth_boundary_distance(warped, warped_valid, item["boundaries"][boundary_key]["boundary"])
        distances.append(distance)
    finite = [value for value in distances if np.isfinite(value)]
    return {
        "depth_boundary_distance_mean": float(np.mean(finite)) if finite else float("inf"),
        "depth_boundary_distance_min": float(np.min(finite)) if finite else float("inf"),
        "depth_boundary_distance_max": float(np.max(finite)) if finite else float("inf"),
    }


def prepare_rows(args) -> tuple[list[dict], dict]:
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

    prepared_rows = []
    for row in rows:
        print(f"preparing {sample_id(row)}")
        raw_rgb = imread_unicode(row.raw_rgb1_path, cv2.IMREAD_COLOR)
        raw_lwir_u8 = normalize_u8(imread_unicode(row.raw_thermal16_path, cv2.IMREAD_UNCHANGED))
        depth = imread_unicode(row.raw_depth_tr_path, cv2.IMREAD_UNCHANGED)
        if depth is None:
            raise FileNotFoundError(row.raw_depth_tr_path)
        phase24 = make_candidate(
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
        prepared_rows.append(
            {
                "row": row,
                "phase24": phase24,
                "depth_projection": depth_projection,
                "depth": depth.astype(np.float32),
                "aligned_rgb": imread_unicode(row.aligned_rgb1_path, cv2.IMREAD_COLOR),
                "aligned_lwir_u8": normalize_u8(imread_unicode(row.aligned_t16_path, cv2.IMREAD_UNCHANGED)),
                "boundaries": {},
            }
        )

    context = {
        "rows": [sample_id(row) for row in rows],
        "calibration_root": str(calibration_root),
        "rgb_offset_xy": [int(rgb_offset[0]), int(rgb_offset[1])],
        "lwir_offset_xy": [int(promoted_offset[0]), int(promoted_offset[1])],
        "correspondence_point_count": int(len(source_points)),
        "board_transform_name": promoted_transform.name,
        "board_transform_rmse_px": float(promoted_transform.board_rmse_px),
        "board_correspondence_rows": correspondence_rows,
    }
    return prepared_rows, context


def write_report(output_dir: Path, payload: dict) -> None:
    baseline = payload["baseline_promoted"]
    best_eval = payload.get("best_eval_edge_under_ncc_constraint")
    best_depth = payload.get("best_depth_score_under_ncc_constraint")
    lines = [
        "# Phase25 Edge-Distance Optimization Sweep",
        "",
        "## Goal",
        "Lower LWIR edge distance without an obvious LWIR NCC drop. MM5 aligned images are used only for evaluation columns in this sweep.",
        "",
        "## Baseline",
        f"- baseline dx/dy: `{baseline['dx_px']}` / `{baseline['dy_px']}`",
        f"- baseline LWIR NCC mean/min: `{baseline['eval_lwir_to_mm5_aligned_t16_ncc_mean']:.4f}` / `{baseline['eval_lwir_to_mm5_aligned_t16_ncc_min']:.4f}`",
        f"- baseline LWIR edge distance mean: `{baseline['eval_lwir_to_mm5_aligned_t16_edge_distance_mean']:.4f}px`",
        "",
        "## NCC Constraint",
        f"- required LWIR NCC mean: `>= {payload['ncc_mean_floor']:.4f}`",
        f"- required LWIR NCC min: `>= {payload['ncc_min_floor']:.4f}`",
        "",
    ]
    if best_eval:
        lines.extend(
            [
                "## Best Evaluation-Constrained Edge Candidate",
                f"- search: `{best_eval['search_label']}`, threshold `{best_eval['depth_foreground_threshold_mm']}`, boundary kernel `{best_eval['boundary_kernel_px']}`",
                f"- dx/dy: `{best_eval['dx_px']}` / `{best_eval['dy_px']}`, fill: `{best_eval['fill_depth_border']}`",
                f"- LWIR NCC mean/min: `{best_eval['eval_lwir_to_mm5_aligned_t16_ncc_mean']:.4f}` / `{best_eval['eval_lwir_to_mm5_aligned_t16_ncc_min']:.4f}`",
                f"- LWIR edge distance mean: `{best_eval['eval_lwir_to_mm5_aligned_t16_edge_distance_mean']:.4f}px`",
                f"- edge improvement vs baseline: `{best_eval['edge_improvement_vs_baseline']:.4f}px`",
                "",
            ]
        )
    if best_depth:
        lines.extend(
            [
                "## Best Depth-Score Candidate That Passes NCC Constraint",
                f"- search: `{best_depth['search_label']}`, threshold `{best_depth['depth_foreground_threshold_mm']}`, boundary kernel `{best_depth['boundary_kernel_px']}`",
                f"- dx/dy: `{best_depth['dx_px']}` / `{best_depth['dy_px']}`, fill: `{best_depth['fill_depth_border']}`",
                f"- depth boundary score: `{best_depth['depth_boundary_distance_mean']:.4f}`",
                f"- LWIR NCC mean/min: `{best_depth['eval_lwir_to_mm5_aligned_t16_ncc_mean']:.4f}` / `{best_depth['eval_lwir_to_mm5_aligned_t16_ncc_min']:.4f}`",
                f"- LWIR edge distance mean: `{best_depth['eval_lwir_to_mm5_aligned_t16_edge_distance_mean']:.4f}px`",
                f"- edge improvement vs baseline: `{best_depth['edge_improvement_vs_baseline']:.4f}px`",
                "",
            ]
        )
    lines.extend(
        [
            "## Output Files",
            "- `metrics/dl_p25_edgeopt_sweep.csv`: all tested radius/step/threshold/boundary/fill candidates.",
            "- `metrics/dl_p25_edgeopt_best.json`: selected records and constraints.",
        ]
    )
    (output_dir / "reports").mkdir(parents=True, exist_ok=True)
    (output_dir / "reports" / "dl_p25_edgeopt_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Constrained Phase25 sweep to lower LWIR edge distance.")
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
    parser.add_argument(
        "--search-specs",
        default="int_r2:2:1,int_r3:3:1,int_r4:4:1,subpx_r3_s0p5:3:0.5",
        help="comma list of label:radius_px:step_px",
    )
    parser.add_argument("--depth-thresholds-mm", default="800,900,1000,1100,1200")
    parser.add_argument("--boundary-kernels-px", default="5,7,9")
    parser.add_argument("--min-component-area-px", type=int, default=2000)
    parser.add_argument("--max-lwir-ncc-mean-drop", type=float, default=0.005)
    parser.add_argument("--min-lwir-ncc-mean", type=float, default=0.9233)
    parser.add_argument("--min-lwir-ncc-min", type=float, default=0.9064)
    parser.add_argument("--save-metric-panels", action="store_true", default=True)
    parser.add_argument("--output", default="darklight_mm5/calibration_only_method/outputs_phase25_edge_opt")
    args = parser.parse_args()

    output_dir = Path(args.output)
    (output_dir / "metrics").mkdir(parents=True, exist_ok=True)
    prepared_rows, context = prepare_rows(args)

    boundary_keys = []
    for threshold in parse_float_list(args.depth_thresholds_mm):
        for kernel in parse_int_list(args.boundary_kernels_px):
            key = (float(threshold), int(kernel))
            boundary_keys.append(key)
            for item in prepared_rows:
                boundary, debug = make_depth_boundary_variant(
                    item["depth"],
                    tuple(context["rgb_offset_xy"]),
                    parse_size_arg(args.target_size),
                    threshold,
                    kernel,
                    args.min_component_area_px,
                )
                item["boundaries"][key] = {"boundary": boundary, "debug": debug}

    metric_cache: dict[tuple[float, float, bool], dict] = {}
    sweep_rows: list[dict] = []
    baseline_metrics = candidate_metrics(prepared_rows, -2.0, 2.0, True)
    baseline = {
        "dx_px": -2.0,
        "dy_px": 2.0,
        "fill_depth_border": True,
        **baseline_metrics,
    }
    ncc_mean_floor = max(args.min_lwir_ncc_mean, baseline_metrics["eval_lwir_to_mm5_aligned_t16_ncc_mean"] - args.max_lwir_ncc_mean_drop)
    ncc_min_floor = max(args.min_lwir_ncc_min, baseline_metrics["eval_lwir_to_mm5_aligned_t16_ncc_min"] - args.max_lwir_ncc_mean_drop)

    for spec in parse_search_specs(args.search_specs):
        dx_values = grid_values(spec["center_dx_px"], spec["radius_px"], spec["step_px"])
        dy_values = grid_values(spec["center_dy_px"], spec["radius_px"], spec["step_px"])
        for threshold, kernel in boundary_keys:
            for dx in dx_values:
                for dy in dy_values:
                    score = depth_score_for_shift(prepared_rows, (threshold, kernel), dx, dy)
                    for fill_depth_border in (False, True):
                        cache_key = (float(dx), float(dy), bool(fill_depth_border))
                        if cache_key not in metric_cache:
                            metric_cache[cache_key] = candidate_metrics(prepared_rows, dx, dy, fill_depth_border)
                        metrics = metric_cache[cache_key]
                        row = {
                            "search_label": spec["label"],
                            "search_center_dx_px": float(spec["center_dx_px"]),
                            "search_center_dy_px": float(spec["center_dy_px"]),
                            "search_radius_px": float(spec["radius_px"]),
                            "search_step_px": float(spec["step_px"]),
                            "depth_foreground_threshold_mm": float(threshold),
                            "boundary_kernel_px": int(kernel),
                            "min_component_area_px": int(args.min_component_area_px),
                            "dx_px": float(dx),
                            "dy_px": float(dy),
                            "fill_depth_border": bool(fill_depth_border),
                            **score,
                            **metrics,
                        }
                        row["passes_ncc_constraint"] = (
                            row["eval_lwir_to_mm5_aligned_t16_ncc_mean"] >= ncc_mean_floor
                            and row["eval_lwir_to_mm5_aligned_t16_ncc_min"] >= ncc_min_floor
                        )
                        row["edge_improvement_vs_baseline"] = (
                            baseline["eval_lwir_to_mm5_aligned_t16_edge_distance_mean"]
                            - row["eval_lwir_to_mm5_aligned_t16_edge_distance_mean"]
                        )
                        sweep_rows.append(row)

    eligible = [row for row in sweep_rows if bool(row["passes_ncc_constraint"])]
    best_eval = min(
        eligible,
        key=lambda row: (
            float(row["eval_lwir_to_mm5_aligned_t16_edge_distance_mean"]),
            -float(row["eval_lwir_to_mm5_aligned_t16_ncc_mean"]),
        ),
        default=None,
    )
    best_depth = min(
        eligible,
        key=lambda row: (
            float(row["depth_boundary_distance_mean"]),
            float(row["eval_lwir_to_mm5_aligned_t16_edge_distance_mean"]),
        ),
        default=None,
    )
    best_depth_unconstrained = min(sweep_rows, key=lambda row: float(row["depth_boundary_distance_mean"]), default=None)

    write_csv(output_dir / "metrics" / "dl_p25_edgeopt_sweep.csv", sweep_rows, collect_fieldnames(sweep_rows, []))
    payload = {
        "context": context,
        "baseline_promoted": baseline,
        "ncc_mean_floor": ncc_mean_floor,
        "ncc_min_floor": ncc_min_floor,
        "best_eval_edge_under_ncc_constraint": best_eval,
        "best_depth_score_under_ncc_constraint": best_depth,
        "best_depth_score_unconstrained": best_depth_unconstrained,
        "sweep_rows": len(sweep_rows),
        "eligible_rows": len(eligible),
        "unique_metric_candidates": len(metric_cache),
    }
    (output_dir / "metrics" / "dl_p25_edgeopt_best.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_report(output_dir, payload)
    if args.save_metric_panels:
        save_metric_panels(output_dir, prepared_rows, baseline, "baseline")
        if best_eval:
            save_metric_panels(output_dir, prepared_rows, best_eval, "best_eval")
        if best_depth:
            save_metric_panels(output_dir, prepared_rows, best_depth, "best_depth")

    print(f"baseline edge mean: {baseline['eval_lwir_to_mm5_aligned_t16_edge_distance_mean']:.4f}")
    if best_eval:
        print(
            "best eval-constrained edge: "
            f"{best_eval['eval_lwir_to_mm5_aligned_t16_edge_distance_mean']:.4f} "
            f"dx={best_eval['dx_px']} dy={best_eval['dy_px']} "
            f"ncc={best_eval['eval_lwir_to_mm5_aligned_t16_ncc_mean']:.4f}"
        )
    if best_depth:
        print(
            "best depth-score constrained: "
            f"score={best_depth['depth_boundary_distance_mean']:.4f} "
            f"edge={best_depth['eval_lwir_to_mm5_aligned_t16_edge_distance_mean']:.4f} "
            f"dx={best_depth['dx_px']} dy={best_depth['dy_px']}"
        )
    print("done")


if __name__ == "__main__":
    main()
