import argparse
import csv
import itertools
import json
import math
from pathlib import Path

import cv2
import numpy as np

from run_calibration_plane import plane_rgb_to_lwir_homography, process_sample_plane
from run_darklight import (
    adjusted_stereo_calibration,
    auto_edges,
    collect_fieldnames,
    direct_alignment_metrics,
    gray_u8,
    image_metric_row,
    imread_unicode,
    imwrite_unicode,
    load_annotation_mask,
    load_rows,
    load_stereo_calibration,
    make_five_panel,
    make_quad,
    metrics_for_csv,
    normalize_u8,
    parse_float_tuple,
    parse_size_arg,
    refine_lwir_translation,
    select_dark_rows,
    shift_image_and_mask,
    warp_image,
    write_csv,
    write_numeric_summary,
)


def sample_id(row):
    return f"{row.aligned_id:03d}_seq{row.sequence}"


def parse_float_list(text):
    return [float(x.strip()) for x in str(text).split(",") if x.strip()]


def parse_int_list(text):
    return [int(x.strip()) for x in str(text).split(",") if x.strip()]


def candidate_key(params, residual_shift=0):
    ox, oy = params["lwir_principal_offset"]
    return (
        f"d{params['plane_depth_mm']:.0f}_t{params['t_scale']:.2f}_"
        f"ox{ox:.0f}_oy{oy:.0f}_r{int(residual_shift)}"
    )


def fixed_feature_gray(raw_rgb):
    gray = cv2.cvtColor(raw_rgb, cv2.COLOR_BGR2GRAY)
    return cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)


def warp_candidate(row, calibration, params, residual_shift=0):
    raw_rgb = imread_unicode(row.raw_rgb1_path, cv2.IMREAD_COLOR)
    raw_lwir = imread_unicode(row.raw_thermal16_path, cv2.IMREAD_UNCHANGED)
    raw_lwir_u8 = normalize_u8(raw_lwir)
    effective = adjusted_stereo_calibration(
        calibration,
        raw_rgb.shape,
        raw_lwir_u8.shape,
        rgb_calib_size=params.get("rgb_calib_size"),
        lwir_calib_size=params.get("lwir_calib_size"),
        lwir_principal_offset=params["lwir_principal_offset"],
        t_scale=params["t_scale"],
    )
    H_rgb_to_lwir = plane_rgb_to_lwir_homography(effective, params["plane_depth_mm"])
    H_lwir_to_rgb = np.linalg.inv(H_rgb_to_lwir)
    H_lwir_to_rgb = H_lwir_to_rgb / H_lwir_to_rgb[2, 2]
    rgb_size = (raw_rgb.shape[1], raw_rgb.shape[0])
    lwir_on_rgb = warp_image(raw_lwir_u8, H_lwir_to_rgb, rgb_size)
    valid_on_rgb = warp_image(
        np.ones(raw_lwir_u8.shape[:2], dtype=np.uint8) * 255,
        H_lwir_to_rgb,
        rgb_size,
        interp=cv2.INTER_NEAREST,
    ) > 0
    anno_mask = load_annotation_mask(row.raw_rgb_anno_class_path, raw_rgb.shape)
    fixed_gray = fixed_feature_gray(raw_rgb)
    refine_info = {
        "accepted": False,
        "dx": 0,
        "dy": 0,
        "score_gain": 0.0,
        "edge_distance_gain": 0.0,
    }
    if residual_shift > 0:
        lwir_on_rgb, valid_on_rgb, refine_info = refine_lwir_translation(
            fixed_gray,
            lwir_on_rgb,
            valid_on_rgb,
            anno_mask,
            max_shift=residual_shift,
        )
    return {
        "raw_rgb": raw_rgb,
        "raw_lwir_u8": raw_lwir_u8,
        "lwir_on_rgb": lwir_on_rgb,
        "valid_on_rgb": valid_on_rgb,
        "anno_mask": anno_mask,
        "fixed_gray": fixed_gray,
        "H_lwir_to_rgb": H_lwir_to_rgb,
        "refine_info": refine_info,
    }


def safe_float(value, fallback=0.0):
    try:
        out = float(value)
    except (TypeError, ValueError):
        return fallback
    if not math.isfinite(out):
        return fallback
    return out


def score_roi(valid, anno_mask):
    roi = valid.astype(bool)
    if anno_mask is not None and np.count_nonzero(roi & anno_mask) > 1000:
        roi &= anno_mask.astype(bool)
    return roi


def boundary_metrics(fixed_gray, moving_gray, valid_mask, anno_mask=None, band_px=5):
    roi = score_roi(valid_mask, anno_mask)
    fixed_edges = (auto_edges(fixed_gray) > 0) & roi
    moving_edges = (auto_edges(moving_gray) > 0) & roi
    kernel = np.ones((max(1, int(band_px)), max(1, int(band_px))), np.uint8)
    fixed_band = cv2.dilate(fixed_edges.astype(np.uint8) * 255, kernel, iterations=1) > 0
    moving_band = cv2.dilate(moving_edges.astype(np.uint8) * 255, kernel, iterations=1) > 0
    moving_count = int(np.count_nonzero(moving_edges))
    fixed_count = int(np.count_nonzero(fixed_edges))
    precision = float(np.count_nonzero(moving_edges & fixed_band) / max(moving_count, 1))
    recall = float(np.count_nonzero(fixed_edges & moving_band) / max(fixed_count, 1))
    f1 = float((2.0 * precision * recall) / max(precision + recall, 1e-6))
    if fixed_count > 0 and moving_count > 0:
        dt = cv2.distanceTransform(255 - fixed_edges.astype(np.uint8) * 255, cv2.DIST_L2, 3)
        vals = dt[moving_edges]
        edge_distance = float(np.mean(np.clip(vals, 0, 50))) if vals.size else float("nan")
    else:
        edge_distance = float("nan")
    return {
        "boundary_precision": precision,
        "boundary_recall": recall,
        "boundary_f1": f1,
        "boundary_edge_distance": edge_distance,
        "boundary_fixed_edge_count": fixed_count,
        "boundary_moving_edge_count": moving_count,
        "fixed_edges": fixed_edges,
        "moving_edges": moving_edges,
        "roi": roi,
    }


def color_boundary_overlay(fixed_gray, fixed_edges, moving_edges, roi):
    bg = cv2.cvtColor(fixed_gray, cv2.COLOR_GRAY2BGR)
    bg = cv2.normalize(bg, None, 0, 180, cv2.NORM_MINMAX)
    out = bg.copy()
    out[roi] = np.maximum(out[roi], 28)
    out[fixed_edges] = (255, 210, 0)
    out[moving_edges] = (0, 40, 255)
    out[fixed_edges & moving_edges] = (255, 255, 255)
    return out


def color_distance_map(fixed_edges, moving_edges):
    if np.count_nonzero(fixed_edges) == 0:
        return np.zeros((*fixed_edges.shape, 3), dtype=np.uint8)
    dt = cv2.distanceTransform(255 - fixed_edges.astype(np.uint8) * 255, cv2.DIST_L2, 3)
    dt = np.clip(dt, 0, 25)
    heat = cv2.applyColorMap((dt * 255.0 / 25.0).astype(np.uint8), cv2.COLORMAP_TURBO)
    heat[~moving_edges] = (0, 0, 0)
    return heat


def write_boundary_diagnostic(out_dir, row, fixed_gray, moving_gray, valid_mask, anno_mask, label):
    sid = sample_id(row)
    metrics = boundary_metrics(fixed_gray, moving_gray, valid_mask, anno_mask)
    fixed_edges = metrics["fixed_edges"]
    moving_edges = metrics["moving_edges"]
    roi = metrics["roi"]
    overlay = color_boundary_overlay(fixed_gray, fixed_edges, moving_edges, roi)
    distance = color_distance_map(fixed_edges, moving_edges)
    rgb_edges = (fixed_edges.astype(np.uint8) * 255)
    lwir_edges = (moving_edges.astype(np.uint8) * 255)
    make_quad(
        [
            (rgb_edges, f"{sid} RGB boundary {label}"),
            (lwir_edges, "LWIR boundary"),
            (overlay, "edge overlay yellow=RGB red=LWIR"),
            (distance, "LWIR edge distance to RGB"),
        ],
        out_dir / f"{sid}_{label}_boundary_panel.png",
    )
    return {k: v for k, v in metrics.items() if not isinstance(v, np.ndarray)}


def reference_lwir(reference_dir, row):
    path = Path(reference_dir) / "samples" / sample_id(row) / "lwir_calibrated_to_rgb1_raw.png"
    if not path.exists():
        raise FileNotFoundError(path)
    return imread_unicode(path, cv2.IMREAD_GRAYSCALE)


def evaluate_candidate(row, calibration, params, reference_dir, residual_shift=0):
    data = warp_candidate(row, calibration, params, residual_shift=residual_shift)
    raw_metrics = direct_alignment_metrics(
        data["fixed_gray"],
        data["lwir_on_rgb"],
        data["valid_on_rgb"],
        data["anno_mask"],
    )
    ref = reference_lwir(reference_dir, row)
    target_metrics = direct_alignment_metrics(ref, data["lwir_on_rgb"], data["valid_on_rgb"], None)
    bmetrics = boundary_metrics(
        data["fixed_gray"],
        data["lwir_on_rgb"],
        data["valid_on_rgb"],
        data["anno_mask"],
    )
    return {
        "sample": sample_id(row),
        "aligned_id": row.aligned_id,
        "sequence": row.sequence,
        "target_ncc": target_metrics["ncc"],
        "target_mi": target_metrics["mi"],
        "target_edge_distance": target_metrics["edge_distance"],
        "target_valid_ratio": target_metrics["valid_ratio"],
        "raw_rgb_ncc": raw_metrics["ncc"],
        "raw_rgb_mi": raw_metrics["mi"],
        "raw_rgb_edge_distance": raw_metrics["edge_distance"],
        "valid_ratio": raw_metrics["valid_ratio"],
        "boundary_f1": bmetrics["boundary_f1"],
        "boundary_edge_distance": bmetrics["boundary_edge_distance"],
        "boundary_precision": bmetrics["boundary_precision"],
        "boundary_recall": bmetrics["boundary_recall"],
        "residual_dx": data["refine_info"]["dx"],
        "residual_dy": data["refine_info"]["dy"],
        "residual_accepted": data["refine_info"]["accepted"],
    }


def aggregate_candidate(rows):
    numeric = {}
    for key in rows[0].keys():
        vals = [safe_float(r.get(key), float("nan")) for r in rows]
        vals = [v for v in vals if math.isfinite(v)]
        if vals:
            numeric[f"{key}_mean"] = float(np.mean(vals))
            numeric[f"{key}_min"] = float(np.min(vals))
            numeric[f"{key}_max"] = float(np.max(vals))
    target_ncc = numeric.get("target_ncc_mean", -1.0)
    target_mi = numeric.get("target_mi_mean", 0.0)
    target_edge = numeric.get("target_edge_distance_mean", 50.0)
    raw_ncc_min = numeric.get("raw_rgb_ncc_min", -1.0)
    valid = numeric.get("valid_ratio_mean", 0.0)
    boundary_f1 = numeric.get("boundary_f1_mean", 0.0)
    penalty = 0.0
    if valid < 0.80:
        penalty += (0.80 - valid) * 3.0
    if raw_ncc_min <= 0:
        penalty += abs(raw_ncc_min) + 0.1
    numeric["selection_score"] = float(
        target_ncc + 0.05 * target_mi - 0.006 * target_edge + 0.05 * boundary_f1 - penalty
    )
    return numeric


def evaluate_param_set(rows, calibration, params, reference_dir, residual_shift=0):
    per_sample = [evaluate_candidate(r, calibration, params, reference_dir, residual_shift) for r in rows]
    summary = aggregate_candidate(per_sample)
    flat = {
        "candidate": candidate_key(params, residual_shift),
        "plane_depth_mm": params["plane_depth_mm"],
        "t_scale": params["t_scale"],
        "lwir_cx_offset": params["lwir_principal_offset"][0],
        "lwir_cy_offset": params["lwir_principal_offset"][1],
        "max_residual_shift": residual_shift,
        **summary,
    }
    return flat, per_sample


def sort_candidates(rows):
    return sorted(
        rows,
        key=lambda r: (
            safe_float(r.get("selection_score"), -999.0),
            safe_float(r.get("target_ncc_mean"), -999.0),
            -safe_float(r.get("target_edge_distance_mean"), 999.0),
        ),
        reverse=True,
    )


def build_param_grid(args):
    depths = parse_float_list(args.depths)
    t_scales = parse_float_list(args.t_scales)
    offset_xs = parse_float_list(args.offset_xs)
    offset_ys = parse_float_list(args.offset_ys)
    lwir_calib_size = parse_size_arg(args.lwir_calib_size)
    for depth, t_scale, ox, oy in itertools.product(depths, t_scales, offset_xs, offset_ys):
        yield {
            "rgb_calib_size": None,
            "lwir_calib_size": lwir_calib_size,
            "plane_depth_mm": float(depth),
            "t_scale": float(t_scale),
            "lwir_principal_offset": (float(ox), float(oy)),
        }


def make_evaluation_panel(out_dir, row, reference_dir, optimized_dir):
    sid = sample_id(row)
    baseline_dir = Path("darklight_mm5/outputs_calibration_plane/samples") / sid
    reference = reference_lwir(reference_dir, row)
    baseline = imread_unicode(baseline_dir / "lwir_calibrated_to_rgb1_raw.png", cv2.IMREAD_GRAYSCALE)
    optimized = imread_unicode(
        Path(optimized_dir) / "samples" / sid / "lwir_calibrated_to_rgb1_raw.png",
        cv2.IMREAD_GRAYSCALE,
    )
    raw_rgb = imread_unicode(Path(optimized_dir) / "samples" / sid / "rgb1_raw.png", cv2.IMREAD_COLOR)
    opt_edge = imread_unicode(Path(optimized_dir) / "samples" / sid / "edge_overlay.png", cv2.IMREAD_COLOR)
    make_five_panel(
        [
            (raw_rgb, f"{sid} RGB1 raw"),
            (reference, "retained official-reference LWIR"),
            (baseline, "current baseline plane"),
            (optimized, "optimized plane"),
            (opt_edge, "optimized edge check"),
        ],
        out_dir / f"{sid}_optimized_evaluation_panel.png",
    )


def write_evaluation_files(rows, output_dir, reference_dir):
    eval_rows = []
    for row in rows:
        sid = sample_id(row)
        sample_dir = Path(output_dir) / "samples" / sid
        raw_rgb = imread_unicode(sample_dir / "rgb1_raw.png", cv2.IMREAD_COLOR)
        raw_gray = fixed_feature_gray(raw_rgb)
        lwir = imread_unicode(sample_dir / "lwir_calibrated_to_rgb1_raw.png", cv2.IMREAD_GRAYSCALE)
        valid = imread_unicode(sample_dir / "valid_mask_calibration_plane.png", cv2.IMREAD_GRAYSCALE) > 0
        anno_mask = load_annotation_mask(row.raw_rgb_anno_class_path, raw_rgb.shape)
        ref = reference_lwir(reference_dir, row)
        target = direct_alignment_metrics(ref, lwir, valid, None)
        raw_metrics = direct_alignment_metrics(raw_gray, lwir, valid, anno_mask)
        alpha = imread_unicode(sample_dir / "fusion_alpha_mask.png", cv2.IMREAD_GRAYSCALE)
        fused_heat = imread_unicode(sample_dir / "fused_heat.png", cv2.IMREAD_COLOR)
        raw_metric = image_metric_row(sid, "raw_rgb1", raw_rgb, gray_u8(raw_rgb), lwir, valid, alpha)
        fused_metric = image_metric_row(sid, "fused_heat", fused_heat, gray_u8(raw_rgb), lwir, valid, alpha)
        eval_rows.append(
            {
                "sample": sid,
                "aligned_id": row.aligned_id,
                "sequence": row.sequence,
                "target_ncc": target["ncc"],
                "target_mi": target["mi"],
                "target_edge_distance": target["edge_distance"],
                "target_valid_ratio": target["valid_ratio"],
                "raw_rgb_ncc": raw_metrics["ncc"],
                "raw_rgb_edge_distance": raw_metrics["edge_distance"],
                "valid_ratio": raw_metrics["valid_ratio"],
                "fusion_entropy_gain_vs_raw_rgb": fused_metric["entropy"] - raw_metric["entropy"],
                "fusion_alpha_coverage": fused_metric["alpha_coverage"],
            }
        )
    write_csv(Path(output_dir) / "evaluation_against_reference.csv", eval_rows, collect_fieldnames(eval_rows, []))
    summary = {
        "strategy": "calibration_plane_optimized",
        "output": str(output_dir),
        "reference_output": str(reference_dir),
        "sample_count": len(eval_rows),
        "means": {},
        "per_sample": eval_rows,
    }
    for key in collect_fieldnames(eval_rows, []):
        vals = [safe_float(r.get(key), float("nan")) for r in eval_rows]
        vals = [v for v in vals if math.isfinite(v)]
        if vals:
            summary["means"][key] = float(np.mean(vals))
    Path(output_dir, "evaluation_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    panel_dir = Path(output_dir) / "evaluation_panels"
    panel_dir.mkdir(parents=True, exist_ok=True)
    for row in rows:
        make_evaluation_panel(panel_dir, row, reference_dir, output_dir)
    return summary


def run_final_output(rows, calibration, params, residual_shift, fusion_params, output_dir):
    output_dir = Path(output_dir)
    metric_rows = []
    registration_rows = []
    fusion_rows = []
    for row in rows:
        sample_summary, sample_registration, sample_fusion = process_sample_plane(
            row,
            output_dir,
            calibration,
            plane_depth_mm=params["plane_depth_mm"],
            rgb_calib_size=params.get("rgb_calib_size"),
            lwir_calib_size=params.get("lwir_calib_size"),
            lwir_principal_offset=params["lwir_principal_offset"],
            t_scale=params["t_scale"],
            max_residual_shift=residual_shift,
            fusion_params=fusion_params,
        )
        metric_rows.append(sample_summary)
        registration_rows.extend(sample_registration)
        fusion_rows.extend(sample_fusion)
    write_csv(output_dir / "metrics" / "per_sample.csv", metric_rows, collect_fieldnames(metric_rows, []))
    write_csv(
        output_dir / "metrics" / "registration_stages.csv",
        registration_rows,
        collect_fieldnames(registration_rows, []),
    )
    write_csv(output_dir / "metrics" / "fusion_metrics.csv", fusion_rows, collect_fieldnames(fusion_rows, []))
    write_numeric_summary(
        output_dir / "metrics" / "summary.json",
        metric_rows,
        {"split", "category", "subcategory", "strategy", "calibration_file", "quad", "five_panel", "edge_review"},
    )
    write_numeric_summary(output_dir / "metrics" / "registration_summary.json", registration_rows, {"sample", "pipeline", "stage"})
    write_numeric_summary(output_dir / "metrics" / "fusion_summary.json", fusion_rows, {"sample", "image"})


def parse_fusion_params(args):
    return {
        "saliency_sigma": args.fusion_saliency_sigma,
        "alpha_low_percentile": args.fusion_alpha_low,
        "alpha_high_percentile": args.fusion_alpha_high,
        "alpha_scale": args.fusion_alpha_scale,
        "alpha_max": args.fusion_alpha_max,
        "roi_dilate_px": args.fusion_roi_dilate_px,
    }


def main():
    parser = argparse.ArgumentParser(description="Optimize calibration-plane boundary and fusion parameters.")
    parser.add_argument("--index", default="mm5_calib_benchmark/outputs/mm5_benchmark/splits/index_with_splits.csv")
    parser.add_argument("--calibration", default="calibration/def_stereocalib_THERM.yml")
    parser.add_argument("--reference-output", default="darklight_mm5/outputs")
    parser.add_argument("--output", default="darklight_mm5/outputs_calibration_plane_opt")
    parser.add_argument("--diagnostic-output", default="darklight_mm5/outputs_calibration_plane_boundary")
    parser.add_argument("--aligned-ids", default="106,104,103")
    parser.add_argument("--limit", type=int, default=3)
    parser.add_argument("--splits", default="test,val")
    parser.add_argument("--lwir-calib-size", default="1280x720")
    parser.add_argument("--depths", default="325,350,375")
    parser.add_argument("--t-scales", default="1.40,1.45,1.50")
    parser.add_argument("--offset-xs", default="12,20,28")
    parser.add_argument("--offset-ys", default="-8,0,8")
    parser.add_argument("--residual-shifts", default="0,2,4,6,8")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--fusion-saliency-sigma", type=float, default=15.0)
    parser.add_argument("--fusion-alpha-low", type=float, default=40.0)
    parser.add_argument("--fusion-alpha-high", type=float, default=96.0)
    parser.add_argument("--fusion-alpha-scale", type=float, default=0.82)
    parser.add_argument("--fusion-alpha-max", type=float, default=0.72)
    parser.add_argument("--fusion-roi-dilate-px", type=int, default=3)
    args = parser.parse_args()

    rows = load_rows(args.index, require_official=False, require_depth=False)
    if args.aligned_ids.strip():
        wanted = {int(x.strip()) for x in args.aligned_ids.split(",") if x.strip()}
        rows = sorted([r for r in rows if r.aligned_id in wanted], key=lambda r: r.aligned_id)
    else:
        rows = select_dark_rows(rows, args.limit, args.splits)
    calibration = load_stereo_calibration(args.calibration)
    diagnostic_dir = Path(args.diagnostic_output)
    diagnostic_dir.mkdir(parents=True, exist_ok=True)

    baseline_params = {
        "rgb_calib_size": None,
        "lwir_calib_size": parse_size_arg(args.lwir_calib_size),
        "plane_depth_mm": 350.0,
        "t_scale": 1.45,
        "lwir_principal_offset": (20.0, 0.0),
    }
    boundary_rows = []
    for row in rows:
        data = warp_candidate(row, calibration, baseline_params, residual_shift=0)
        b = write_boundary_diagnostic(
            diagnostic_dir,
            row,
            data["fixed_gray"],
            data["lwir_on_rgb"],
            data["valid_on_rgb"],
            data["anno_mask"],
            "baseline",
        )
        boundary_rows.append({"sample": sample_id(row), **b})
    write_csv(diagnostic_dir / "boundary_metrics_baseline.csv", boundary_rows, collect_fieldnames(boundary_rows, []))

    coarse_rows = []
    per_candidate = {}
    params_by_name = {}
    for idx, params in enumerate(build_param_grid(args), start=1):
        flat, per_sample = evaluate_param_set(rows, calibration, params, args.reference_output, residual_shift=0)
        flat["stage"] = "coarse_plane"
        coarse_rows.append(flat)
        per_candidate[flat["candidate"]] = per_sample
        params_by_name[flat["candidate"]] = params
        if idx % 20 == 0:
            print(f"coarse candidates evaluated: {idx}")
    coarse_sorted = sort_candidates(coarse_rows)
    write_csv(diagnostic_dir / "optimization_candidates_coarse.csv", coarse_sorted, collect_fieldnames(coarse_sorted, []))

    residual_rows = []
    residual_shifts = parse_int_list(args.residual_shifts)
    for coarse in coarse_sorted[: max(1, args.top_k)]:
        base_params = params_by_name[coarse["candidate"]]
        for residual_shift in residual_shifts:
            flat, per_sample = evaluate_param_set(
                rows,
                calibration,
                base_params,
                args.reference_output,
                residual_shift=residual_shift,
            )
            flat["stage"] = "residual_refine"
            residual_rows.append(flat)
            per_candidate[flat["candidate"]] = per_sample
            params_by_name[flat["candidate"]] = base_params
    residual_sorted = sort_candidates(residual_rows)
    write_csv(diagnostic_dir / "optimization_candidates_residual.csv", residual_sorted, collect_fieldnames(residual_sorted, []))

    best = residual_sorted[0]
    best_params = params_by_name[best["candidate"]]
    best_residual = int(best["max_residual_shift"])
    optimized_boundary_rows = []
    for row in rows:
        data = warp_candidate(row, calibration, best_params, residual_shift=best_residual)
        b = write_boundary_diagnostic(
            diagnostic_dir,
            row,
            data["fixed_gray"],
            data["lwir_on_rgb"],
            data["valid_on_rgb"],
            data["anno_mask"],
            "optimized",
        )
        optimized_boundary_rows.append({"sample": sample_id(row), **b})
    write_csv(
        diagnostic_dir / "boundary_metrics_optimized.csv",
        optimized_boundary_rows,
        collect_fieldnames(optimized_boundary_rows, []),
    )

    fusion_params = parse_fusion_params(args)
    run_final_output(rows, calibration, best_params, best_residual, fusion_params, args.output)
    eval_summary = write_evaluation_files(rows, args.output, args.reference_output)

    summary = {
        "selected_candidate": best,
        "selected_parameters": {
            "lwir_calib_size": args.lwir_calib_size,
            "plane_depth_mm": best_params["plane_depth_mm"],
            "t_scale": best_params["t_scale"],
            "lwir_principal_offset": list(best_params["lwir_principal_offset"]),
            "max_residual_shift": best_residual,
            "fusion_params": fusion_params,
        },
        "baseline_metrics": {
            "target_ncc": 0.259933,
            "target_edge_distance": 5.6309466666666665,
            "valid_ratio": 0.8324159999999999,
            "fusion_alpha_coverage": 0.008846333333333333,
        },
        "optimized_evaluation": eval_summary,
        "notes": [
            "The retained official-reference output is used only for offline selection/evaluation.",
            "The original darklight_mm5/outputs_calibration_plane baseline is not overwritten.",
        ],
    }
    Path(args.output, "optimization_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary["selected_parameters"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
