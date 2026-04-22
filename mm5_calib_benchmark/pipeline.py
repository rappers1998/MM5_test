from __future__ import annotations

import csv
import importlib
import io
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .common import (
    compose_rows,
    compute_plane_homography,
    draw_text_block,
    ensure_bgr,
    ensure_dir,
    fit_image,
    load_opencv_yaml,
    read_image,
    save_opencv_yaml,
    to_gray_u8,
    valid_mask_from_homography,
    warp_image,
    warp_mask,
    write_image,
)
from .config import METHOD_CONFIGS, load_config, load_method_config
from .eval.geometry import summarize_alignment
from .eval.metrics import boundary_f1, compute_confusion, summarize_confusion, write_per_class_csv
from .viz.overlays import contour_overlay, error_heatmap, mar_like_overlay


METHOD_MODULES = {
    "m0": "mm5_calib_benchmark.methods.m0_mm5_official.run",
    "m1": "mm5_calib_benchmark.methods.m1_zhang.run",
    "m2": "mm5_calib_benchmark.methods.m2_su2025_xoftr.run",
    "m3": "mm5_calib_benchmark.methods.m3_jay2025_sgm.run",
    "m4": "mm5_calib_benchmark.methods.m4_muhovic_depthbridge.run",
    "m5": "mm5_calib_benchmark.methods.m5_epnp.run",
    "m6": "mm5_calib_benchmark.methods.m6_mar_edge_refine.run",
    "m7": "mm5_calib_benchmark.methods.m7_depth_guided_selfcal.run",
}

OUTPUT_NAMES = {
    ("m0", "thermal"): "method_M0_mm5_official_thermal",
    ("m0", "uv"): "method_M0_mm5_official_uv",
    ("m1", "thermal"): "method_M1_zhang_opencv_thermal",
    ("m1", "uv"): "method_M1_zhang_opencv_uv",
    ("m2", "thermal"): "method_M2_su2025_xoftr_thermal",
    ("m3", "uv"): "method_M3_jay2025_sgm_uv",
    ("m4", "thermal"): "method_M4_muhovic_depthbridge_thermal",
    ("m4", "uv"): "method_M4_muhovic_depthbridge_uv",
    ("m5", "thermal"): "method_M5_epnp_baseline_thermal",
    ("m5", "uv"): "method_M5_epnp_baseline_uv",
    ("m6", "thermal"): "method_M6_mar_edge_refine_thermal",
    ("m7", "thermal"): "method_M7_depth_guided_selfcal_thermal",
}

DEFAULT_SUITE = [
    ("m0", "thermal"),
    ("m0", "uv"),
    ("m1", "thermal"),
    ("m1", "uv"),
    ("m5", "thermal"),
    ("m5", "uv"),
    ("m2", "thermal"),
    ("m4", "thermal"),
    ("m4", "uv"),
    ("m3", "uv"),
    ("m6", "thermal"),
    ("m7", "thermal"),
]


@dataclass
class SceneRow:
    payload: dict[str, Any]

    def __getattr__(self, item: str) -> Any:
        try:
            return self.payload[item]
        except KeyError as exc:
            raise AttributeError(item) from exc

    def __getitem__(self, item: str) -> Any:
        return self.payload[item]


def _parse_bool(value: str) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def _read_csv_text(path: Path) -> str:
    for encoding in ("utf-8-sig", "utf-8", "gb18030"):
        try:
            return path.read_text(encoding=encoding)
        except UnicodeDecodeError:
            continue
    return path.read_text(encoding="latin-1")


def load_index_with_splits(config: dict[str, Any]) -> list[SceneRow]:
    csv_path = Path(config["outputs"]["index_csv"])
    reader = csv.DictReader(io.StringIO(_read_csv_text(csv_path)))
    rows: list[SceneRow] = []
    for row in reader:
        payload: dict[str, Any] = {}
        for key, value in row.items():
            value = value or ""
            if key in {"aligned_id", "sequence"}:
                payload[key] = int(value) if value else 0
            elif key.startswith("has_"):
                payload[key] = _parse_bool(value)
            else:
                payload[key] = value
        rows.append(SceneRow(payload))
    return rows


def load_class_names(config: dict[str, Any], rows: list[SceneRow] | None = None) -> list[str]:
    sample_rows = rows or load_index_with_splits(config)
    label_ids = {0}
    for row in sample_rows[:12]:
        for field in ("raw_rgb_anno_class_path", "raw_thermal_anno_class_path", "raw_uv_anno_class_path"):
            path = row.payload.get(field, "")
            if not path:
                continue
            image = read_image(path, cv2.IMREAD_UNCHANGED)
            if image is not None:
                label_ids.update(int(v) for v in np.unique(image))
    return ["background" if cls == 0 else f"class_{cls}" for cls in sorted(label_ids)]


def _read_scene_assets(row: SceneRow, track: str) -> dict[str, Any]:
    source_image = read_image(row.raw_rgb3_path, cv2.IMREAD_COLOR)
    source_mask = read_image(row.raw_rgb_anno_class_path, cv2.IMREAD_UNCHANGED)
    depth_image = read_image(row.raw_depth_tr_path, cv2.IMREAD_UNCHANGED)
    if source_image is None or source_mask is None:
        raise FileNotFoundError(f"Missing source assets for scene {row.sequence}")

    if track == "thermal":
        target_image = read_image(row.raw_thermal16_path, cv2.IMREAD_UNCHANGED)
        if target_image is None:
            target_image = read_image(row.raw_thermal_path, cv2.IMREAD_UNCHANGED)
        target_mask = read_image(row.raw_thermal_anno_class_path, cv2.IMREAD_UNCHANGED)
    elif track == "uv":
        target_image = read_image(row.raw_uv1_path, cv2.IMREAD_UNCHANGED)
        target_mask = read_image(row.raw_uv_anno_class_path, cv2.IMREAD_UNCHANGED)
    elif track == "ir":
        target_image = read_image(row.raw_ir16_path, cv2.IMREAD_UNCHANGED)
        target_mask = read_image(row.raw_thermal_anno_class_path, cv2.IMREAD_UNCHANGED)
    else:
        raise ValueError(f"Unsupported track: {track}")

    if target_image is None or target_mask is None:
        raise FileNotFoundError(f"Missing target assets for scene {row.sequence}, track {track}")

    return {
        "source_image": source_image,
        "source_mask": source_mask.astype(np.uint8),
        "target_image": target_image,
        "target_mask": target_mask.astype(np.uint8),
        "depth_image": depth_image.astype(np.float32) if depth_image is not None else None,
    }


def get_method_module(method_key: str):
    return importlib.import_module(METHOD_MODULES[method_key])


def output_dir_for(config: dict[str, Any], method_key: str, track: str) -> Path:
    return Path(config["outputs"]["root"]) / OUTPUT_NAMES[(method_key, track)]


def load_saved_stereo(config: dict[str, Any], method_key: str, track: str) -> dict[str, Any] | None:
    stereo_path = output_dir_for(config, method_key, track) / "calib" / "final_calibration.yml"
    if stereo_path.exists():
        return load_opencv_yaml(stereo_path)
    return None


def _runtime_config(base_config: dict[str, Any], method_key: str) -> dict[str, Any]:
    config = dict(base_config)
    config["method"] = load_method_config(method_key)
    return config


def _scene_metrics(result: dict[str, Any], class_names: list[str]) -> tuple[dict[str, float], np.ndarray]:
    assets = result["assets"]
    confusion = compute_confusion(result["pred_mask"], assets["target_mask"], result["valid_mask"], len(class_names))
    summary, _ = summarize_confusion(confusion, class_names)
    geom = summarize_alignment(result["warped_source"], assets["target_image"], result["pred_mask"], assets["target_mask"], result["valid_mask"])
    summary["boundary_f1"] = boundary_f1(result["pred_mask"], assets["target_mask"], result["valid_mask"])
    summary.update(geom)
    return summary, confusion


def _save_scene_outputs(out_dir: Path, row: SceneRow, result: dict[str, Any], summary: dict[str, float]) -> None:
    prefix = f"{int(row.sequence):03d}"
    target_bgr = ensure_bgr(result["assets"]["target_image"])
    warped_bgr = ensure_bgr(result["warped_source"])
    overlay = cv2.addWeighted(target_bgr, 0.55, warped_bgr, 0.45, 0.0)
    contour = contour_overlay(target_bgr, result["assets"]["target_mask"], (0, 255, 0))
    contour = contour_overlay(contour, result["pred_mask"], (255, 0, 0))
    heat = error_heatmap(result["pred_mask"], result["assets"]["target_mask"], result["valid_mask"])

    caption = "\n".join(
        [
            f"scene={int(row.sequence)}",
            f"mIoU={summary['mean_iou']:.4f}",
            f"PA={summary['pixel_accuracy']:.4f}",
            f"BF1={summary['boundary_f1']:.4f}",
        ]
    )
    overlay = draw_text_block(overlay, caption, (8, 8))

    write_image(out_dir / "warped" / f"{prefix}_warped.png", warped_bgr)
    write_image(out_dir / "warped" / f"{prefix}_overlay.png", overlay)
    write_image(out_dir / "masks" / f"{prefix}_pred_mask.png", result["pred_mask"].astype(np.uint8))
    write_image(out_dir / "masks" / f"{prefix}_gt_mask.png", result["assets"]["target_mask"].astype(np.uint8))
    write_image(out_dir / "masks" / f"{prefix}_error_map.png", heat)
    write_image(out_dir / "viz" / f"{prefix}_contours.png", contour)
    write_image(out_dir / "viz" / f"{prefix}_heatmap.png", heat)
    stage_images = dict(result.get("debug", {}).get("stage_images", {}))
    for stage_name, stage_image in stage_images.items():
        write_image(out_dir / "viz" / "stages" / f"{prefix}_{stage_name}.png", ensure_bgr(stage_image))


def _select_best_mid_worst(saved_rows: list[tuple[int, dict[str, Any], dict[str, float]]]) -> list[tuple[int, dict[str, Any], dict[str, float]]]:
    if len(saved_rows) <= 3:
        return saved_rows
    ordered = sorted(saved_rows, key=lambda item: float(item[2].get("mean_iou", 0.0)))
    indices = [0, len(ordered) // 2, len(ordered) - 1]
    unique_rows: list[tuple[int, dict[str, Any], dict[str, float]]] = []
    used = set()
    for idx in indices:
        sequence = int(ordered[idx][0])
        if sequence in used:
            continue
        unique_rows.append(ordered[idx])
        used.add(sequence)
    for candidate in ordered:
        sequence = int(candidate[0])
        if sequence in used:
            continue
        unique_rows.append(candidate)
        used.add(sequence)
        if len(unique_rows) == 3:
            break
    return unique_rows[:3]


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _make_qualitative_grid(out_dir: Path, saved_rows: list[tuple[int, dict[str, Any], dict[str, float]]]) -> None:
    if not saved_rows:
        return
    tiles: list[list[np.ndarray]] = []
    titles = ["Worst", "Median", "Best"]
    for title, (sequence, result, summary) in zip(titles, _select_best_mid_worst(saved_rows), strict=False):
        target = fit_image(ensure_bgr(result["assets"]["target_image"]), 360, 240)
        overlay = fit_image(cv2.addWeighted(ensure_bgr(result["assets"]["target_image"]), 0.55, ensure_bgr(result["warped_source"]), 0.45, 0.0), 360, 240)
        contour = fit_image(contour_overlay(ensure_bgr(result["assets"]["target_image"]), result["pred_mask"], (255, 0, 0)), 360, 240)
        target = draw_text_block(target, f"{title} scene {sequence}\nTarget", (8, 8))
        overlay = draw_text_block(overlay, f"Overlay\nmIoU={summary['mean_iou']:.4f}", (8, 8))
        contour = draw_text_block(contour, "Prediction contour", (8, 8))
        tiles.append([target, overlay, contour])
    grid = compose_rows(tiles, gap=10)
    write_image(out_dir / "viz" / "qualitative_grid_best_mid_worst.png", grid)


def run_method(config: dict[str, Any], method_key: str, track: str) -> dict[str, Any]:
    runtime_config = _runtime_config(config, method_key)
    rows = load_index_with_splits(runtime_config)
    eval_rows = [row for row in rows if str(row.payload.get("split", "")) == "test"]
    eval_rows = [row for row in eval_rows if _track_available(row, track)]
    max_scenes = int(runtime_config["runtime"].get("max_test_scenes", len(eval_rows)))
    eval_rows = eval_rows[:max_scenes]
    class_names = load_class_names(runtime_config, eval_rows or rows)

    out_dir = output_dir_for(runtime_config, method_key, track)
    for child in ("calib", "warped", "masks", "metrics", "viz"):
        ensure_dir(out_dir / child)

    module = get_method_module(method_key)
    context = module.calibrate(runtime_config, rows, track)
    stereo = context.get("stereo")
    if stereo is not None:
        save_opencv_yaml(out_dir / "calib" / "final_calibration.yml", stereo)

    total_confusion = np.zeros((len(class_names), len(class_names)), dtype=np.int64)
    per_scene_rows: list[dict[str, Any]] = []
    saved_results: list[tuple[int, dict[str, Any], dict[str, float]]] = []
    accumulators: dict[str, list[float]] = {
        "boundary_f1": [],
        "keypoint_transfer_error_px": [],
        "overall_region_error_px": [],
        "mutual_information": [],
        "ntg": [],
        "valid_warp_coverage": [],
    }

    for row in eval_rows:
        result = module.compute_scene_result(runtime_config, row, track, context)
        summary, confusion = _scene_metrics(result, class_names)
        total_confusion += confusion
        _save_scene_outputs(out_dir, row, result, summary)
        saved_results.append((int(row.sequence), result, summary))
        scene_row = {"sequence": int(row.sequence), "track": track, **summary}
        per_scene_rows.append(scene_row)
        for key in accumulators:
            value = summary.get(key)
            if value is not None and np.isfinite(float(value)):
                accumulators[key].append(float(value))

    confusion_summary, per_class_rows = summarize_confusion(total_confusion, class_names)
    summary = {
        "method_name": str(runtime_config["method"]["method_name"]),
        "track": track.upper(),
        "num_test_scenes": len(eval_rows),
        "reprojection_mae_px": float(context.get("calibration_metrics", {}).get("reprojection_mae_px", 0.0)),
        "checkerboard_corner_rmse_px": float(context.get("calibration_metrics", {}).get("checkerboard_corner_rmse_px", 0.0)),
        "keypoint_transfer_error_px": float(np.mean(accumulators["keypoint_transfer_error_px"])) if accumulators["keypoint_transfer_error_px"] else 0.0,
        "overall_region_error_px": float(np.mean(accumulators["overall_region_error_px"])) if accumulators["overall_region_error_px"] else 0.0,
        "ntg": float(np.mean(accumulators["ntg"])) if accumulators["ntg"] else 0.0,
        "mutual_information": float(np.mean(accumulators["mutual_information"])) if accumulators["mutual_information"] else 0.0,
        "valid_warp_coverage": float(np.mean(accumulators["valid_warp_coverage"])) if accumulators["valid_warp_coverage"] else 0.0,
        "mean_iou": float(confusion_summary["mean_iou"]),
        "freq_iou": float(confusion_summary["freq_iou"]),
        "pixel_accuracy": float(confusion_summary["pixel_accuracy"]),
        "mean_pixel_accuracy": float(confusion_summary["mean_pixel_accuracy"]),
        "boundary_f1": float(np.mean(accumulators["boundary_f1"])) if accumulators["boundary_f1"] else 0.0,
    }

    (out_dir / "metrics" / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _write_csv(out_dir / "metrics" / "per_scene.csv", per_scene_rows)
    write_per_class_csv(out_dir / "metrics" / "per_class.csv", per_class_rows)
    _make_qualitative_grid(out_dir, saved_results)
    return summary


def _track_available(row: SceneRow, track: str) -> bool:
    if track == "thermal":
        return bool(row.payload.get("has_thermal", False))
    if track == "uv":
        return bool(row.payload.get("has_uv", False))
    if track == "ir":
        return bool(row.payload.get("has_ir", False))
    return False


def run_default_suite(config: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    base_config = config or load_config()
    results = []
    for method_key, track in DEFAULT_SUITE:
        results.append(run_method(base_config, method_key, track))
    return results


def _load_or_build_context(base_config: dict[str, Any], method_key: str, track: str, rows: list[SceneRow]) -> dict[str, Any]:
    runtime_config = _runtime_config(base_config, method_key)
    stereo = load_saved_stereo(runtime_config, method_key, track)
    if stereo is not None:
        return {"stereo": stereo, "calibration_metrics": {}}
    module = get_method_module(method_key)
    return module.calibrate(runtime_config, rows, track)


def _read_json_if_exists(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _find_legacy_mar_root(project_root: Path) -> Path | None:
    candidates = [
        project_root.parent / "MAR_test",
        project_root / "MAR_test",
    ]
    for candidate in candidates:
        if (candidate / "backup_2600.py").exists() and (candidate / "runs").exists():
            return candidate
    return None


def _legacy_metrics_row(track: str, method: str, summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "track": track,
        "method": method,
        "mean_iou": float(summary.get("mean_iou", float("nan"))),
        "pixel_accuracy": float(summary.get("pixel_accuracy", float("nan"))),
        "mean_pixel_accuracy": float("nan"),
        "freq_iou": float("nan"),
        "boundary_f1": float("nan"),
        "mutual_information": float("nan"),
        "ntg": float("nan"),
        "valid_warp_coverage": float("nan"),
        "keypoint_transfer_error_px": float("nan"),
        "overall_region_error_px": float("nan"),
    }


def _load_legacy_mar_artifacts(project_root: Path) -> list[dict[str, Any]]:
    legacy_root = _find_legacy_mar_root(project_root)
    if legacy_root is None:
        return []

    runs_dir = legacy_root / "runs"
    specs = [
        ("282_seq282_engineered_best_acceptance", "MAR-engineered-full"),
        ("282_seq282_paper_final_acceptance", "MAR-paper-full"),
    ]
    artifacts: list[dict[str, Any]] = []
    for run_name, label in specs:
        run_dir = runs_dir / run_name
        if not run_dir.exists():
            continue
        artifacts.append(
            {
                "label": label,
                "run_name": run_name,
                "run_dir": run_dir,
                "raw_eval": _read_json_if_exists(run_dir / "eval_raw_gt.json"),
                "aligned_eval": _read_json_if_exists(run_dir / "eval_aligned_gt.json"),
                "diagnostics": _read_json_if_exists(run_dir / "align_diagnostics.json"),
                "rw_iou": _read_json_if_exists(run_dir / "rw_iou_report.json"),
                "raw_overlay_path": run_dir / "overlay_refined.png",
                "aligned_overlay_path": run_dir / "aligned_overlay_final.png",
            }
        )
    return artifacts


def _legacy_overlay_tile(image_path: Path, title: str, summary: dict[str, Any] | None) -> np.ndarray:
    image = ensure_bgr(read_image(image_path, cv2.IMREAD_COLOR))
    if image is None:
        image = np.full((240, 360, 3), 24, dtype=np.uint8)
    tile = fit_image(image, 360, 240)
    if summary:
        fg_iou = float(summary.get("binary_iou_foreground", float("nan")))
        tile = draw_text_block(
            tile,
            "\n".join(
                [
                    title,
                    f"PA={float(summary.get('pixel_accuracy', float('nan'))):.4f}",
                    f"mIoU={float(summary.get('mean_iou', float('nan'))):.4f}",
                    f"FG-IoU={fg_iou:.4f}",
                ]
            ),
            (8, 8),
            font_scale=0.55,
            line_height=22,
        )
    else:
        tile = draw_text_block(tile, title, (8, 8), font_scale=0.55, line_height=22)
    return tile


def _legacy_info_tile(label: str, diagnostics: dict[str, Any] | None, rw_iou: dict[str, Any] | None) -> np.ndarray:
    diagnostics = diagnostics or {}
    ratios = diagnostics.get("ratios", {})
    review = diagnostics.get("no_gt_review", {})
    roundtrip = diagnostics.get("rect_raw_roundtrip_error", {})
    rw_mean = (rw_iou or {}).get("mean_iou", {}).get("stage4_vs_final", float("nan"))
    tile = np.full((240, 360, 3), 24, dtype=np.uint8)
    tile = draw_text_block(
        tile,
        "\n".join(
            [
                label,
                "source=MAR_test/backup_2600.py",
                f"geom_ok={review.get('geometry_baseline_ok', 'n/a')}",
                f"realProj={float(ratios.get('real_projected_ratio_on_redistort', float('nan'))):.3f}",
                f"holeFill={float(ratios.get('hole_fill_ratio_on_redistort', float('nan'))):.3f}",
                f"rw(stage4->final)={float(rw_mean):.4f}",
                f"roundtrip={float(roundtrip.get('mean_px', float('nan'))):.4f}px",
            ]
        ),
        (10, 10),
        font_scale=0.55,
        line_height=22,
    )
    return tile


def _export_mar_history_artifacts(out_dir: Path, project_root: Path, metrics_rows: list[dict[str, Any]]) -> list[list[np.ndarray]]:
    legacy_artifacts = _load_legacy_mar_artifacts(project_root)
    if not legacy_artifacts:
        return []

    banner = np.full((96, 1100, 3), 28, dtype=np.uint8)
    banner = draw_text_block(
        banner,
        "Legacy MAR full pipeline imported from sibling MAR_test/backup_2600.py",
        (14, 18),
        font_scale=0.8,
        line_height=28,
    )
    panel_rows: list[list[np.ndarray]] = [[banner]]
    note_lines = [
        "# MAR History Note",
        "",
        "- The rows below are imported from the original `MAR_test/backup_2600.py` acceptance outputs.",
        "- They are not recomputed by the new benchmark code path; they are the stored full-pipeline historical results.",
        "",
    ]

    for artifact in legacy_artifacts:
        raw_summary = (artifact.get("raw_eval") or {}).get("summary", {})
        aligned_summary = (artifact.get("aligned_eval") or {}).get("summary", {})

        if raw_summary:
            metrics_rows.append(
                _legacy_metrics_row("thermal_legacy_raw", str(artifact["label"]), raw_summary)
            )
        if aligned_summary:
            metrics_rows.append(
                _legacy_metrics_row("thermal_legacy_aligned", str(artifact["label"]), aligned_summary)
            )

        raw_overlay_path = Path(artifact["raw_overlay_path"])
        aligned_overlay_path = Path(artifact["aligned_overlay_path"])
        if raw_overlay_path.exists():
            shutil.copy2(raw_overlay_path, out_dir / f"{artifact['run_name']}_overlay_refined.png")
        if aligned_overlay_path.exists():
            shutil.copy2(aligned_overlay_path, out_dir / f"{artifact['run_name']}_aligned_overlay_final.png")

        for json_name in ("eval_raw_gt.json", "eval_aligned_gt.json", "align_diagnostics.json", "rw_iou_report.json"):
            src = Path(artifact["run_dir"]) / json_name
            if src.exists():
                shutil.copy2(src, out_dir / f"{artifact['run_name']}_{json_name}")

        raw_tile = _legacy_overlay_tile(raw_overlay_path, f"{artifact['label']} raw", raw_summary)
        aligned_tile = _legacy_overlay_tile(aligned_overlay_path, f"{artifact['label']} aligned", aligned_summary)
        info_tile = _legacy_info_tile(str(artifact["label"]), artifact.get("diagnostics"), artifact.get("rw_iou"))
        panel_rows.append([raw_tile, aligned_tile, info_tile])

        note_lines.extend(
            [
                f"## {artifact['label']}",
                "",
                f"- Raw thermal GT: pixel_accuracy={float(raw_summary.get('pixel_accuracy', float('nan'))):.6f}, mean_iou={float(raw_summary.get('mean_iou', float('nan'))):.6f}, binary_iou_foreground={float(raw_summary.get('binary_iou_foreground', float('nan'))):.6f}",
                f"- Aligned GT: pixel_accuracy={float(aligned_summary.get('pixel_accuracy', float('nan'))):.6f}, mean_iou={float(aligned_summary.get('mean_iou', float('nan'))):.6f}, binary_iou_foreground={float(aligned_summary.get('binary_iou_foreground', float('nan'))):.6f}",
                f"- geometry_baseline_ok={artifact.get('diagnostics', {}).get('no_gt_review', {}).get('geometry_baseline_ok', 'n/a')}, real_projected_ratio_on_redistort={float(artifact.get('diagnostics', {}).get('ratios', {}).get('real_projected_ratio_on_redistort', float('nan'))):.6f}",
                "",
            ]
        )

    manual_metrics = _read_json_if_exists(project_root / "runs" / "manual_paper_engineered_metrics.json")
    figure_src = project_root / "runs" / "MAR_final_acceptance_vs_paper.png"
    if figure_src.exists():
        shutil.copy2(figure_src, out_dir / "scene_282_3_mar_history_source.png")
    if manual_metrics is not None:
        note_lines.extend(
            [
                "## Manual Binary Reference",
                "",
                f"- MAR-engineered-history: pixel_accuracy_total={float(manual_metrics.get('engineered_best', {}).get('vs_manual', {}).get('pixel_accuracy_total', float('nan'))):.6f}, foreground_iou={float(manual_metrics.get('engineered_best', {}).get('vs_manual', {}).get('foreground_iou', float('nan'))):.6f}",
                f"- MAR-paper-history: pixel_accuracy_total={float(manual_metrics.get('paper_final', {}).get('vs_manual', {}).get('pixel_accuracy_total', float('nan'))):.6f}, foreground_iou={float(manual_metrics.get('paper_final', {}).get('vs_manual', {}).get('foreground_iou', float('nan'))):.6f}",
                "",
                "- These manual binary numbers are a different metric definition from the multi-class benchmark table.",
                "",
            ]
        )

    legacy_panel = compose_rows(panel_rows, gap=10)
    write_image(out_dir / "scene_282_3_mar_history_panel.png", legacy_panel)
    (out_dir / "scene_282_3_mar_history_note.md").write_text("\n".join(note_lines), encoding="utf-8")
    return panel_rows


def _load_method_test_summary(config: dict[str, Any], method_key: str, track: str) -> dict[str, Any] | None:
    summary_path = output_dir_for(config, method_key, track) / "metrics" / "summary.json"
    return _read_json_if_exists(summary_path)


def _comparison_method_specs(config: dict[str, Any]) -> tuple[list[tuple[str, str]], dict[str, dict[str, Any]]]:
    ordered_specs = [
        ("m1", "M1"),
        ("m2", "M2"),
        ("m4", "M4"),
        ("m5", "M5"),
        ("m6", "M6"),
        ("m7", "M7"),
    ]
    summaries = {method_key: _load_method_test_summary(config, method_key, "thermal") or {} for method_key, _ in ordered_specs}
    return ordered_specs, summaries


def _finite_metric(value: Any, default: float = float("nan")) -> float:
    try:
        value_f = float(value)
    except (TypeError, ValueError):
        return default
    if not np.isfinite(value_f):
        return default
    return value_f


def _build_method_ranking_rows(test_summaries: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    ordered_specs = [
        ("m1", "M1"),
        ("m2", "M2"),
        ("m4", "M4"),
        ("m5", "M5"),
        ("m6", "M6"),
        ("m7", "M7"),
    ]
    rows: list[dict[str, Any]] = []
    for method_key, label in ordered_specs:
        summary = test_summaries.get(method_key) or {}
        if not summary:
            continue
        rows.append(
            {
                "method_key": method_key,
                "method": label,
                "mean_iou": _finite_metric(summary.get("mean_iou")),
                "pixel_accuracy": _finite_metric(summary.get("pixel_accuracy")),
                "boundary_f1": _finite_metric(summary.get("boundary_f1")),
                "keypoint_transfer_error_px": _finite_metric(summary.get("keypoint_transfer_error_px")),
                "overall_region_error_px": _finite_metric(summary.get("overall_region_error_px")),
                "valid_warp_coverage": _finite_metric(summary.get("valid_warp_coverage")),
                "num_test_scenes": int(summary.get("num_test_scenes", 0)),
            }
        )
    rows.sort(
        key=lambda row: (
            _finite_metric(row.get("overall_region_error_px"), float("inf")),
            -_finite_metric(row.get("mean_iou"), -1.0),
            -_finite_metric(row.get("pixel_accuracy"), -1.0),
            str(row.get("method", "")),
        )
    )
    return rows


def _draw_rank_bar_section(
    rows: list[dict[str, Any]],
    metric_key: str,
    title: str,
    *,
    width: int,
    row_height: int,
    higher_is_better: bool,
    max_value: float,
) -> np.ndarray:
    section_h = 72 + len(rows) * row_height + 26
    canvas = np.full((section_h, width, 3), 22, dtype=np.uint8)
    canvas = draw_text_block(canvas, title, (18, 14), font_scale=0.78, line_height=28)
    left_x = 220
    right_x = width - 140
    bar_w = right_x - left_x
    baseline_y = 64
    cv2.line(canvas, (left_x, baseline_y), (left_x, section_h - 18), (95, 95, 95), 1)
    cv2.line(canvas, (right_x, baseline_y), (right_x, section_h - 18), (95, 95, 95), 1)

    palette = [
        (50, 180, 90),
        (70, 165, 120),
        (90, 150, 165),
        (105, 130, 205),
        (110, 105, 230),
        (95, 85, 245),
    ]

    for idx, row in enumerate(rows):
        value = _finite_metric(row.get(metric_key), 0.0)
        y = baseline_y + idx * row_height + 22
        color = palette[idx % len(palette)]
        rank_label = f"{idx + 1}. {row['method']}"
        cv2.putText(canvas, rank_label, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (240, 240, 240), 1, cv2.LINE_AA)

        if metric_key in {"mean_iou", "pixel_accuracy", "boundary_f1", "valid_warp_coverage"}:
            value_text = f"{value:.4f}"
        else:
            value_text = f"{value:.2f}px"

        if higher_is_better:
            ratio = np.clip(value / max(max_value, 1e-6), 0.0, 1.0)
        else:
            ratio = np.clip(value / max(max_value, 1e-6), 0.0, 1.0)
        current_bar_w = int(round(bar_w * ratio))

        cv2.rectangle(canvas, (left_x, y - 14), (right_x, y + 8), (42, 42, 42), -1)
        cv2.rectangle(canvas, (left_x, y - 14), (left_x + current_bar_w, y + 8), color, -1)
        cv2.rectangle(canvas, (left_x, y - 14), (right_x, y + 8), (96, 96, 96), 1)
        cv2.putText(canvas, value_text, (right_x + 14, y), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (235, 235, 235), 1, cv2.LINE_AA)

    return canvas


def _export_method_ranking_chart(out_dir: Path, ranking_rows: list[dict[str, Any]]) -> None:
    if not ranking_rows:
        return

    width = 1360
    title = np.full((112, width, 3), 24, dtype=np.uint8)
    title = draw_text_block(
        title,
        "Thermal 方法排序图（按 test mean_iou 从高到低）\n当前参与排序的方法：M1、M2、M4、M5、M6、M7 | M3 仅有 UV 输出，因此不纳入 thermal 排序",
        (18, 16),
        font_scale=0.8,
        line_height=30,
    )

    mean_iou_section = _draw_rank_bar_section(
        ranking_rows,
        "mean_iou",
        "A. test mean_iou（按整体区域误差排序后的同序展示）",
        width=width,
        row_height=42,
        higher_is_better=True,
        max_value=1.0,
    )
    pixel_acc_section = _draw_rank_bar_section(
        ranking_rows,
        "pixel_accuracy",
        "B. 同一排序下的 test pixel_accuracy",
        width=width,
        row_height=42,
        higher_is_better=True,
        max_value=1.0,
    )
    region_section = _draw_rank_bar_section(
        ranking_rows,
        "overall_region_error_px",
        "C. 整体区域平均像素误差：overall_region_error_px（越低越好）",
        width=width,
        row_height=42,
        higher_is_better=False,
        max_value=max(_finite_metric(row.get("overall_region_error_px"), 0.0) for row in ranking_rows) * 1.10,
    )
    keypoint_section = _draw_rank_bar_section(
        ranking_rows,
        "keypoint_transfer_error_px",
        "D. 旧边界误差：keypoint_transfer_error_px（越低越好）",
        width=width,
        row_height=42,
        higher_is_better=False,
        max_value=max(_finite_metric(row.get("keypoint_transfer_error_px"), 0.0) for row in ranking_rows) * 1.10,
    )
    chart = compose_rows([[title], [region_section], [mean_iou_section], [pixel_acc_section], [keypoint_section]], gap=12)
    write_image(out_dir / "scene_282_3_thermal_method_ranking_bar.png", chart)
    write_image(out_dir / "scene_282_3_thermal_overall_region_ranking_bar.png", chart)

    ranking_csv_rows = [
        {
            "rank": idx + 1,
            "method": row["method"],
            "overall_region_error_px": row["overall_region_error_px"],
            "mean_iou": row["mean_iou"],
            "pixel_accuracy": row["pixel_accuracy"],
            "boundary_f1": row["boundary_f1"],
            "keypoint_transfer_error_px": row["keypoint_transfer_error_px"],
            "valid_warp_coverage": row["valid_warp_coverage"],
            "num_test_scenes": row["num_test_scenes"],
        }
        for idx, row in enumerate(ranking_rows)
    ]
    _write_csv(out_dir / "scene_282_3_thermal_method_ranking.csv", ranking_csv_rows)
    _write_csv(out_dir / "scene_282_3_thermal_overall_region_ranking.csv", ranking_csv_rows)


def _export_metric_explanation_note(out_dir: Path, thermal_shape: tuple[int, int], ranking_rows: list[dict[str, Any]]) -> None:
    height, width = thermal_shape
    total_pixels = int(height * width)
    assumed_wrong_px = int(round(total_pixels * 0.01))
    note_lines = [
        "# 指标说明",
        "",
        f"- 本说明按 thermal 图像尺寸 {width} x {height} = {total_pixels} 像素来举例。",
        "- `pixel_accuracy` 的含义是：在所有有效评测像素里，预测类别与目标 GT 完全一致的像素比例。",
        "- 如果 `pixel_accuracy = 0.99`，表示大约 99% 的有效像素分类是正确的，约 1% 是错误的。",
        f"- 如果整张图都进入评测，1% 的错误大约对应 {assumed_wrong_px} 个像素。",
        "- 但 `pixel_accuracy` 不是一个直接的几何位移像素量，它并不等价于 `0.01 px` 的校准偏差。",
        "- 这次新增的 `overall_region_error_px` 更偏向整体区域误差：它对整个前景区域做双向平均距离，而不是只看边界。",
        "- 原来的 `keypoint_transfer_error_px` 仍然保留，但它更偏向边界平均误差。",
        "- `mean_iou` 的含义是：先对每个类别计算 `IoU = intersection / union`，然后对各类取平均。它比 `pixel_accuracy` 更敏感于前景错位、类别混淆和边界问题。",
        "- 实际上，一个方法即使 `pixel_accuracy` 很高，也可能因为背景占比大而让 `mean_iou` 仍然一般，这通常意味着前景轮廓和局部区域还没有真正对齐好。",
        "",
        "## 当前排序结果（按 overall_region_error_px 从低到高，也就是从最好到最差）",
        "",
    ]
    for idx, row in enumerate(ranking_rows, start=1):
        note_lines.append(
            f"- {idx}. {row['method']}: overall_region_error_px={row['overall_region_error_px']:.2f}px, mean_iou={row['mean_iou']:.4f}, pixel_accuracy={row['pixel_accuracy']:.4f}, keypoint_transfer_error_px={row['keypoint_transfer_error_px']:.2f}px"
        )
    note_lines.extend(
        [
            "",
            "## 如何理解 `pixel_accuracy = 0.99`",
            "",
            "- 它表示 99% 的有效像素类别判断正确。",
            "- 它并不保证物体轮廓已经完全贴合。",
            "- 因此在看校准质量时，应该结合 `mean_iou`、`boundary_f1`、`overall_region_error_px` 和 `keypoint_transfer_error_px` 一起判断。",
        ]
    )
    (out_dir / "scene_282_3_metric_explanation.md").write_text("\n".join(note_lines), encoding="utf-8")


def generate_scene_2823_comparison(config: dict[str, Any] | None = None) -> Path:
    base_config = config or load_config()
    rows = load_index_with_splits(base_config)
    scene_row = next(row for row in rows if int(row.sequence) == 282)
    class_names = load_class_names(base_config, [scene_row])

    thermal_specs, test_summaries = _comparison_method_specs(base_config)
    out_dir = Path(base_config["outputs"]["root"]) / "scene_282_3_comparison"
    ensure_dir(out_dir)

    metrics_rows: list[dict[str, Any]] = []
    thermal_assets = _read_scene_assets(scene_row, "thermal")

    banner = np.full((100, 1100, 3), 24, dtype=np.uint8)
    banner = draw_text_block(
        banner,
        "Scene 282 thermal-only 对比图\n当前行顺序已固定为从上到下 M1 -> M7（仅 thermal 方法）",
        (14, 18),
        font_scale=0.82,
        line_height=28,
    )
    header_left = draw_text_block(fit_image(thermal_assets["source_image"], 360, 240), "RGB3 source", (8, 8))
    header_mid = draw_text_block(fit_image(ensure_bgr(thermal_assets["target_image"]), 360, 240), "Thermal target", (8, 8))
    header_right = draw_text_block(
        fit_image(mar_like_overlay(thermal_assets["target_image"], thermal_assets["target_mask"]), 360, 240),
        "Thermal GT",
        (8, 8),
    )
    all_rows: list[list[np.ndarray]] = [[banner], [header_left, header_mid, header_right]]

    for method_key, label in thermal_specs:
        runtime_config = _runtime_config(base_config, method_key)
        context = _load_or_build_context(base_config, method_key, "thermal", rows)
        result = get_method_module(method_key).compute_scene_result(runtime_config, scene_row, "thermal", context)
        summary, _ = _scene_metrics(result, class_names)
        metrics_rows.append({"track": "thermal", "method": label, **summary})
        test_summary = test_summaries.get(method_key, {})
        thermal_overlay = draw_text_block(
            fit_image(mar_like_overlay(result["assets"]["target_image"], result["pred_mask"]), 360, 240),
            f"{label} prediction\nsceneIoU={summary['mean_iou']:.4f}\ntestIoU={float(test_summary.get('mean_iou', summary['mean_iou'])):.4f}",
            (8, 8),
        )
        contour = contour_overlay(ensure_bgr(result["assets"]["target_image"]), result["assets"]["target_mask"], (0, 255, 0))
        contour = contour_overlay(contour, result["pred_mask"], (255, 0, 0))
        contour = draw_text_block(fit_image(contour, 360, 240), f"{label} contours\nBF1={summary['boundary_f1']:.4f}", (8, 8))
        heat = draw_text_block(fit_image(error_heatmap(result["pred_mask"], result["assets"]["target_mask"], result["valid_mask"]), 360, 240), f"{label} error\nCov={summary['valid_warp_coverage']:.3f}", (8, 8))
        all_rows.append([thermal_overlay, contour, heat])

    legacy_rows = _export_mar_history_artifacts(out_dir, Path(base_config["project_root"]), metrics_rows)
    if legacy_rows:
        all_rows.extend(legacy_rows)

    full_panel = compose_rows(all_rows, gap=10)
    write_image(out_dir / "scene_282_3_thermal_mar_style_panel.png", full_panel)
    write_image(out_dir / "scene_282_3_thermal_only_comparison.png", full_panel)
    write_image(out_dir / "scene_282_3_all_methods_comparison.png", full_panel)
    _write_csv(out_dir / "scene_282_3_metrics.csv", metrics_rows)
    ranking_rows = _build_method_ranking_rows(test_summaries)
    _export_method_ranking_chart(out_dir, ranking_rows)
    _export_metric_explanation_note(out_dir, thermal_assets["target_image"].shape[:2], ranking_rows)
    return out_dir
