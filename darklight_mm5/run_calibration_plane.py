import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from run_darklight import (
    adjusted_stereo_calibration,
    alignment_metrics,
    collect_fieldnames,
    direct_alignment_metrics,
    fit_cover_homography,
    gray_u8,
    image_metric_row,
    imread_unicode,
    imwrite_unicode,
    load_annotation_mask,
    load_rows,
    load_stereo_calibration,
    make_edge_overlay,
    make_five_panel,
    make_fusion,
    make_quad,
    matrix_to_list,
    metrics_for_csv,
    normalize_u8,
    parse_float_tuple,
    parse_size_arg,
    refine_lwir_translation,
    select_dark_rows,
    warp_image,
    write_csv,
    write_numeric_summary,
)


def plane_rgb_to_lwir_homography(calibration, plane_depth_mm):
    normal = np.array([[0.0], [0.0], [1.0]], dtype=np.float64)
    t = calibration.rgb_to_lwir_T.reshape(3, 1)
    H = calibration.lwir_K @ (
        calibration.rgb_to_lwir_R - (t @ normal.T) / float(max(plane_depth_mm, 1.0))
    ) @ np.linalg.inv(calibration.rgb_K)
    return H / H[2, 2]


def process_sample_plane(
    row,
    output_dir,
    calibration,
    plane_depth_mm,
    rgb_calib_size=None,
    lwir_calib_size=(1280, 720),
    lwir_principal_offset=(0.0, 0.0),
    t_scale=1.0,
    max_residual_shift=0,
    fusion_params=None,
):
    sid = f"{row.aligned_id:03d}_seq{row.sequence}"
    sample_dir = output_dir / "samples" / sid
    quad_dir = output_dir / "quads"
    five_dir = output_dir / "five_panels"
    edge_dir = output_dir / "edge_reviews"
    sample_dir.mkdir(parents=True, exist_ok=True)

    raw_rgb = imread_unicode(row.raw_rgb1_path, cv2.IMREAD_COLOR)
    raw_rgb_bright = imread_unicode(row.raw_rgb3_path, cv2.IMREAD_COLOR)
    raw_lwir = imread_unicode(row.raw_thermal16_path, cv2.IMREAD_UNCHANGED)
    raw_lwir_u8 = normalize_u8(raw_lwir)
    raw_rgb_gray = gray_u8(raw_rgb)
    effective = adjusted_stereo_calibration(
        calibration,
        raw_rgb.shape,
        raw_lwir_u8.shape,
        rgb_calib_size=rgb_calib_size,
        lwir_calib_size=lwir_calib_size,
        lwir_principal_offset=lwir_principal_offset,
        t_scale=t_scale,
    )
    H_rgb_to_lwir = plane_rgb_to_lwir_homography(effective, plane_depth_mm)
    H_lwir_to_rgb = np.linalg.inv(H_rgb_to_lwir)
    H_lwir_to_rgb = H_lwir_to_rgb / H_lwir_to_rgb[2, 2]
    rgb_size = (raw_rgb.shape[1], raw_rgb.shape[0])
    lwir_on_rgb_initial = warp_image(raw_lwir_u8, H_lwir_to_rgb, rgb_size)
    valid_initial = warp_image(
        np.ones(raw_lwir_u8.shape[:2], dtype=np.uint8) * 255,
        H_lwir_to_rgb,
        rgb_size,
        interp=cv2.INTER_NEAREST,
    ) > 0
    anno_mask = load_annotation_mask(row.raw_rgb_anno_class_path, raw_rgb.shape)
    fixed_for_refine = cv2.cvtColor(raw_rgb, cv2.COLOR_BGR2GRAY)
    fixed_for_refine = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(fixed_for_refine)
    if max_residual_shift > 0:
        lwir_on_rgb, valid_on_rgb, refine_info = refine_lwir_translation(
            fixed_for_refine,
            lwir_on_rgb_initial,
            valid_initial,
            anno_mask,
            max_shift=max_residual_shift,
        )
    else:
        final_metrics = direct_alignment_metrics(fixed_for_refine, lwir_on_rgb_initial, valid_initial, anno_mask)
        lwir_on_rgb, valid_on_rgb = lwir_on_rgb_initial, valid_initial
        refine_info = {
            "accepted": False,
            "dx": 0,
            "dy": 0,
            "score_gain": 0.0,
            "edge_distance_gain": 0.0,
            "final": final_metrics,
        }
    final_metrics = direct_alignment_metrics(fixed_for_refine, lwir_on_rgb, valid_on_rgb, anno_mask)
    initial_metrics = direct_alignment_metrics(fixed_for_refine, lwir_on_rgb_initial, valid_initial, anno_mask)
    raw_lwir_baseline_metrics = alignment_metrics(
        raw_rgb_gray,
        raw_lwir_u8,
        fit_cover_homography(raw_lwir_u8.shape, raw_rgb_gray.shape),
    )

    fusion_params = fusion_params or {}
    fused_heat, fused_intensity, alpha_mask, rgb_display = make_fusion(
        raw_rgb,
        lwir_on_rgb,
        valid_on_rgb,
        anno_mask,
        **fusion_params,
    )
    edge_overlay = make_edge_overlay(raw_rgb, lwir_on_rgb, valid_on_rgb)

    imwrite_unicode(sample_dir / "rgb1_raw.png", raw_rgb)
    imwrite_unicode(sample_dir / "rgb3_raw_bright_reference.png", raw_rgb_bright)
    imwrite_unicode(sample_dir / "rgb1_raw_display_enhanced.png", rgb_display)
    imwrite_unicode(sample_dir / "lwir_raw_norm.png", raw_lwir_u8)
    imwrite_unicode(sample_dir / "lwir_calibration_plane_initial_to_rgb1_raw.png", lwir_on_rgb_initial)
    imwrite_unicode(sample_dir / "lwir_calibrated_to_rgb1_raw.png", lwir_on_rgb)
    imwrite_unicode(sample_dir / "valid_mask_calibration_plane.png", valid_on_rgb.astype(np.uint8) * 255)
    imwrite_unicode(sample_dir / "fused_heat.png", fused_heat)
    imwrite_unicode(sample_dir / "fused_intensity.png", fused_intensity)
    imwrite_unicode(sample_dir / "fusion_alpha_mask.png", alpha_mask)
    imwrite_unicode(sample_dir / "edge_overlay.png", edge_overlay)
    np.save(sample_dir / "H_rgb_to_lwir_plane.npy", H_rgb_to_lwir)
    np.save(sample_dir / "H_lwir_to_rgb_plane.npy", H_lwir_to_rgb)

    transform_info = {
        "aligned_id": row.aligned_id,
        "sequence": row.sequence,
        "strategy": "calibration_plane",
        "calibration_file": calibration.path,
        "plane_depth_mm": float(plane_depth_mm),
        "rgb_calib_size": list(rgb_calib_size) if rgb_calib_size else None,
        "lwir_calib_size": list(lwir_calib_size) if lwir_calib_size else None,
        "lwir_principal_offset": [float(x) for x in lwir_principal_offset],
        "t_scale": float(t_scale),
        "max_residual_shift": int(max_residual_shift),
        "fusion_params": fusion_params,
        "H_rgb_to_lwir_plane": matrix_to_list(H_rgb_to_lwir),
        "H_lwir_to_rgb_plane": matrix_to_list(H_lwir_to_rgb),
        "initial_metrics": metrics_for_csv(initial_metrics),
        "refine_info": {
            "accepted": refine_info["accepted"],
            "dx": refine_info["dx"],
            "dy": refine_info["dy"],
            "score_gain": refine_info["score_gain"],
            "edge_distance_gain": refine_info["edge_distance_gain"],
        },
        "final_metrics": metrics_for_csv(final_metrics),
    }
    (sample_dir / "transform_info.json").write_text(
        json.dumps(transform_info, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    make_quad(
        [
            (raw_rgb, f"RGB1 Raw dark mean {row.dark_mean:.2f}"),
            (raw_lwir_u8, "LWIR Raw normalized"),
            (lwir_on_rgb, "LWIR -> RGB1 calibration plane"),
            (fused_heat, "Fused Result"),
        ],
        quad_dir / f"{sid}_quad.png",
    )
    make_five_panel(
        [
            (raw_rgb, f"RGB1 Raw dark/synced {row.dark_mean:.2f}"),
            (raw_rgb_bright, "RGB3 Raw bright reference"),
            (raw_lwir_u8, "LWIR Raw normalized"),
            (lwir_on_rgb, "LWIR -> RGB1 calibration plane"),
            (fused_heat, "Fused Result"),
        ],
        five_dir / f"{sid}_five_panel.png",
    )
    make_quad(
        [
            (lwir_on_rgb_initial, "Plane calibration initial"),
            (lwir_on_rgb, f"After residual dx={refine_info['dx']} dy={refine_info['dy']}"),
            (edge_overlay, "Edge Check cyan=RGB red=LWIR"),
            (alpha_mask, "Fusion Alpha Mask"),
        ],
        edge_dir / f"{sid}_edge_review.png",
    )

    registration_rows = [
        {
            "sample": sid,
            "aligned_id": row.aligned_id,
            "sequence": row.sequence,
            "pipeline": "raw_lwir_to_raw_rgb1",
            "stage": "fit_cover_baseline",
            **metrics_for_csv(raw_lwir_baseline_metrics),
        },
        {
            "sample": sid,
            "aligned_id": row.aligned_id,
            "sequence": row.sequence,
            "pipeline": "raw_lwir_to_raw_rgb1",
            "stage": "calibration_plane_initial",
            "plane_depth_mm": float(plane_depth_mm),
            "valid_ratio": float(valid_initial.mean()),
            **metrics_for_csv(initial_metrics),
        },
        {
            "sample": sid,
            "aligned_id": row.aligned_id,
            "sequence": row.sequence,
            "pipeline": "raw_lwir_to_raw_rgb1",
            "stage": "bounded_translation_refine",
            "residual_dx": refine_info["dx"],
            "residual_dy": refine_info["dy"],
            "refine_accepted": refine_info["accepted"],
            **metrics_for_csv(final_metrics),
        },
    ]
    fusion_rows = [
        image_metric_row(sid, "raw_rgb1", raw_rgb, raw_rgb_gray, lwir_on_rgb, valid_on_rgb, alpha_mask),
        image_metric_row(sid, "enhanced_rgb1", rgb_display, raw_rgb_gray, lwir_on_rgb, valid_on_rgb, alpha_mask),
        image_metric_row(sid, "fused_intensity", fused_intensity, raw_rgb_gray, lwir_on_rgb, valid_on_rgb, alpha_mask),
        image_metric_row(sid, "fused_heat", fused_heat, raw_rgb_gray, lwir_on_rgb, valid_on_rgb, alpha_mask),
    ]
    fused_heat_metrics = next(r for r in fusion_rows if r["image"] == "fused_heat")
    raw_rgb_metrics = next(r for r in fusion_rows if r["image"] == "raw_rgb1")
    sample_summary = {
        "aligned_id": row.aligned_id,
        "sequence": row.sequence,
        "split": row.split,
        "category": row.category,
        "subcategory": row.subcategory,
        "strategy": "calibration_plane",
        "dark_mean": row.dark_mean,
        "calibration_file": calibration.path,
        "lwir_calib_width": float(lwir_calib_size[0]) if lwir_calib_size else float("nan"),
        "lwir_calib_height": float(lwir_calib_size[1]) if lwir_calib_size else float("nan"),
        "plane_depth_mm": float(plane_depth_mm),
        "lwir_cx_offset": float(lwir_principal_offset[0]),
        "lwir_cy_offset": float(lwir_principal_offset[1]),
        "t_scale": float(t_scale),
        "valid_ratio": float(valid_on_rgb.mean()),
        "residual_dx": refine_info["dx"],
        "residual_dy": refine_info["dy"],
        "residual_refine_accepted": refine_info["accepted"],
        "raw_lwir_to_rgb_baseline_ncc": raw_lwir_baseline_metrics["ncc"],
        "raw_lwir_to_rgb_baseline_edge_distance": raw_lwir_baseline_metrics["edge_distance"],
        "raw_lwir_to_rgb_calibrated_ncc": final_metrics["ncc"],
        "raw_lwir_to_rgb_calibrated_edge_distance": final_metrics["edge_distance"],
        "raw_lwir_to_rgb_calibrated_mi": final_metrics["mi"],
        "raw_lwir_to_rgb_ncc_gain": final_metrics["ncc"] - raw_lwir_baseline_metrics["ncc"],
        "raw_lwir_to_rgb_edge_distance_gain": raw_lwir_baseline_metrics["edge_distance"] - final_metrics["edge_distance"],
        "fusion_entropy_gain_vs_raw_rgb": fused_heat_metrics["entropy"] - raw_rgb_metrics["entropy"],
        "fusion_avg_gradient_gain_vs_raw_rgb": fused_heat_metrics["average_gradient"] - raw_rgb_metrics["average_gradient"],
        "fusion_mi_with_lwir": fused_heat_metrics["mi_with_calibrated_lwir"],
        "fusion_mi_with_raw_rgb": fused_heat_metrics["mi_with_raw_rgb"],
        "fusion_mean_abs_change_vs_raw_rgb": fused_heat_metrics["mean_abs_change_vs_raw_rgb"],
        "fusion_alpha_mean": fused_heat_metrics["alpha_mean"],
        "fusion_alpha_coverage": fused_heat_metrics["alpha_coverage"],
        "quad": str(quad_dir / f"{sid}_quad.png"),
        "five_panel": str(five_dir / f"{sid}_five_panel.png"),
        "edge_review": str(edge_dir / f"{sid}_edge_review.png"),
    }
    return sample_summary, registration_rows, fusion_rows


def main():
    parser = argparse.ArgumentParser(description="Calibration-plane RGB1/LWIR fusion review.")
    parser.add_argument(
        "--index",
        default="mm5_calib_benchmark/outputs/mm5_benchmark/splits/index_with_splits.csv",
    )
    parser.add_argument("--output", default="darklight_mm5/outputs_calibration_plane")
    parser.add_argument("--calibration", default="calibration/def_stereocalib_THERM.yml")
    parser.add_argument("--plane-depth-mm", type=float, default=325.0)
    parser.add_argument("--rgb-calib-size", default="raw")
    parser.add_argument("--lwir-calib-size", default="1280x720")
    parser.add_argument("--lwir-principal-offset", default="14,0")
    parser.add_argument("--t-scale", type=float, default=1.45)
    parser.add_argument("--max-residual-shift", type=int, default=2)
    parser.add_argument("--fusion-saliency-sigma", type=float, default=15.0)
    parser.add_argument("--fusion-alpha-low", type=float, default=40.0)
    parser.add_argument("--fusion-alpha-high", type=float, default=96.0)
    parser.add_argument("--fusion-alpha-scale", type=float, default=0.82)
    parser.add_argument("--fusion-alpha-max", type=float, default=0.72)
    parser.add_argument("--fusion-roi-dilate-px", type=int, default=3)
    parser.add_argument("--limit", type=int, default=3)
    parser.add_argument("--splits", default="test,val")
    parser.add_argument("--aligned-ids", default="106,104,103")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = load_rows(args.index, require_official=False, require_depth=False)
    if args.aligned_ids.strip():
        wanted = {int(x.strip()) for x in args.aligned_ids.split(",") if x.strip()}
        selected = sorted([r for r in rows if r.aligned_id in wanted], key=lambda r: r.aligned_id)
    else:
        selected = select_dark_rows(rows, args.limit, args.splits)
    calibration = load_stereo_calibration(args.calibration)
    rgb_calib_size = parse_size_arg(args.rgb_calib_size)
    lwir_calib_size = parse_size_arg(args.lwir_calib_size)
    lwir_principal_offset = parse_float_tuple(args.lwir_principal_offset, 2)
    fusion_params = {
        "saliency_sigma": args.fusion_saliency_sigma,
        "alpha_low_percentile": args.fusion_alpha_low,
        "alpha_high_percentile": args.fusion_alpha_high,
        "alpha_scale": args.fusion_alpha_scale,
        "alpha_max": args.fusion_alpha_max,
        "roi_dilate_px": args.fusion_roi_dilate_px,
    }

    selected_rows = [
        {
            "aligned_id": r.aligned_id,
            "sequence": r.sequence,
            "split": r.split,
            "dark_mean": r.dark_mean,
            "raw_rgb1_path": r.raw_rgb1_path,
            "raw_rgb3_path": r.raw_rgb3_path,
            "raw_thermal16_path": r.raw_thermal16_path,
            "strategy": "calibration_plane",
        }
        for r in selected
    ]
    write_csv(
        output_dir / "selected_dark_samples.csv",
        selected_rows,
        ["aligned_id", "sequence", "split", "dark_mean", "raw_rgb1_path", "raw_rgb3_path", "raw_thermal16_path", "strategy"],
    )

    metric_rows = []
    registration_rows = []
    fusion_rows = []
    for row in selected:
        print(f"processing aligned_id={row.aligned_id} sequence={row.sequence} strategy=calibration_plane")
        sample_summary, sample_registration, sample_fusion = process_sample_plane(
            row,
            output_dir,
            calibration,
            plane_depth_mm=args.plane_depth_mm,
            rgb_calib_size=rgb_calib_size,
            lwir_calib_size=lwir_calib_size,
            lwir_principal_offset=lwir_principal_offset,
            t_scale=args.t_scale,
            max_residual_shift=args.max_residual_shift,
            fusion_params=fusion_params,
        )
        metric_rows.append(sample_summary)
        registration_rows.extend(sample_registration)
        fusion_rows.extend(sample_fusion)

    write_csv(output_dir / "metrics" / "per_sample.csv", metric_rows, collect_fieldnames(metric_rows, []))
    write_csv(output_dir / "metrics" / "registration_stages.csv", registration_rows, collect_fieldnames(registration_rows, []))
    write_csv(output_dir / "metrics" / "fusion_metrics.csv", fusion_rows, collect_fieldnames(fusion_rows, []))
    write_numeric_summary(output_dir / "metrics" / "summary.json", metric_rows, {"split", "category", "subcategory", "strategy", "calibration_file", "quad", "five_panel", "edge_review"})
    write_numeric_summary(output_dir / "metrics" / "registration_summary.json", registration_rows, {"sample", "pipeline", "stage"})
    write_numeric_summary(output_dir / "metrics" / "fusion_summary.json", fusion_rows, {"sample", "image"})
    print("done")


if __name__ == "__main__":
    main()
