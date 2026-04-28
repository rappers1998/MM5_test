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
    collect_fieldnames,
    direct_alignment_metrics,
    gray_u8,
    imread_unicode,
    load_rows,
    load_stereo_calibration,
    make_edge_overlay,
    make_five_panel,
    normalize_u8,
    parse_size_arg,
    write_csv,
)
from run_calibration_only import (  # noqa: E402
    crop_mask_with_border,
    crop_with_border,
    read_rectification_matrices,
    read_single_camera_model,
    sample_id,
)
from diagnose_aligned_canvas import (  # noqa: E402
    CanvasRule,
    build_canvas_rules,
    build_lwir_models,
    evaluate_candidate,
    load_bridge_metrics,
    make_candidate_from_rule,
    summarize,
    tuple2,
)


PATTERN_SIZE = (11, 8)


def choose_rows(index: str, aligned_ids: str):
    rows = load_rows(index, require_official=False, require_depth=False)
    wanted = {int(x.strip()) for x in aligned_ids.split(",") if x.strip()}
    return sorted([row for row in rows if row.aligned_id in wanted], key=lambda row: row.aligned_id)


def default_calibration_root(index: str) -> str:
    with Path(index).open("r", encoding="utf-8-sig", newline="") as f:
        first = next(csv.DictReader(f))
    return str(first.get("calibration_root", "")).strip()


def phase21_rules(rule_map: dict[str, CanvasRule]) -> list[CanvasRule]:
    out: list[CanvasRule] = []
    for key in (
        "shared_rgb_optimal_alpha0_0",
        "rectified_intersection_x_rgb_optimal_alpha1_y",
    ):
        if key in rule_map:
            out.append(rule_map[key])

    pair_specs = [
        ("phase21_sep_rgb_intersection_lwir_alpha0", "rectified_intersection_x_rgb_optimal_alpha1_y", "shared_rgb_optimal_alpha0_0"),
        ("phase21_sep_rgb_intersection_lwir_alpha0_5", "rectified_intersection_x_rgb_optimal_alpha1_y", "shared_rgb_optimal_alpha0_5"),
        ("phase21_sep_rgb_intersection_lwir_alpha1", "rectified_intersection_x_rgb_optimal_alpha1_y", "shared_rgb_optimal_alpha1_0"),
        ("phase21_sep_rgb_alpha0_5_lwir_alpha0", "shared_rgb_optimal_alpha0_5", "shared_rgb_optimal_alpha0_0"),
    ]
    for name, rgb_key, lwir_key in pair_specs:
        if rgb_key not in rule_map or lwir_key not in rule_map:
            continue
        rgb_rule = rule_map[rgb_key]
        lwir_rule = rule_map[lwir_key]
        out.append(
            CanvasRule(
                name=name,
                rgb_offset=rgb_rule.rgb_offset,
                lwir_offset=lwir_rule.lwir_offset,
                rgb_source=rgb_rule.rgb_source,
                lwir_source=lwir_rule.lwir_source,
                allowed_for_generation=True,
                source=f"RGB canvas from {rgb_key}; LWIR canvas from {lwir_key}. Both rules are calibration-derived.",
            )
        )
    return out


def read_image_unicode(path: Path) -> np.ndarray | None:
    data = np.fromfile(str(path), dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_UNCHANGED)


def to_gray(img: np.ndarray) -> np.ndarray:
    if img.ndim == 3 and img.shape[2] == 4:
        return cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
    if img.ndim == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


def detect_chessboard(img: np.ndarray) -> np.ndarray | None:
    gray = to_gray(img)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    variants = [gray, cv2.equalizeHist(gray), clahe.apply(gray), 255 - gray]
    flags_sb = cv2.CALIB_CB_EXHAUSTIVE + cv2.CALIB_CB_ACCURACY
    for candidate in variants:
        ok, corners = cv2.findChessboardCornersSB(candidate, PATTERN_SIZE, flags_sb)
        if ok:
            return corners.reshape(-1, 2).astype(np.float32)

    flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 1e-3)
    for candidate in variants:
        ok, corners = cv2.findChessboardCorners(candidate, PATTERN_SIZE, flags)
        if ok:
            corners = cv2.cornerSubPix(candidate, corners, (5, 5), (-1, -1), criteria)
            return corners.reshape(-1, 2).astype(np.float32)
    return None


def corner_order_variants(points: np.ndarray) -> list[tuple[str, np.ndarray]]:
    cols, rows = PATTERN_SIZE
    grid = points.reshape(rows, cols, 2)
    return [
        ("orig", grid.reshape(-1, 2)),
        ("rev", grid[::-1, ::-1].reshape(-1, 2)),
        ("flip_h", grid[:, ::-1].reshape(-1, 2)),
        ("flip_v", grid[::-1, :].reshape(-1, 2)),
    ]


def best_pair_homography(left_points: np.ndarray, right_points: np.ndarray) -> tuple[float, str, np.ndarray]:
    best: tuple[float, str, np.ndarray] | None = None
    for orientation, right_variant in corner_order_variants(right_points):
        H, _ = cv2.findHomography(right_variant, left_points, 0)
        if H is None:
            continue
        projected = cv2.perspectiveTransform(right_variant.reshape(-1, 1, 2), H).reshape(-1, 2)
        rmse = float(np.sqrt(np.mean(np.linalg.norm(projected - left_points, axis=1) ** 2)))
        if best is None or rmse < best[0]:
            best = (rmse, orientation, H / H[2, 2])
    if best is None:
        raise ValueError("failed to estimate pair homography")
    return best


def capture_homography_candidates(calibration_root: str, max_rmse: float, max_candidates: int) -> list[dict]:
    root = Path(calibration_root) / "capture_THERM" / "1280x720"
    if not root.exists():
        return []

    candidates = []
    for left_path in sorted(root.rglob("*_left.png")):
        right_path = Path(str(left_path).replace("_left.png", "_right.png"))
        if not right_path.exists():
            continue
        left_img = read_image_unicode(left_path)
        right_img = read_image_unicode(right_path)
        if left_img is None or right_img is None:
            continue
        left_points = detect_chessboard(left_img)
        right_points = detect_chessboard(right_img)
        if left_points is None or right_points is None:
            continue
        rmse, orientation, H = best_pair_homography(left_points, right_points)
        rel = left_path.relative_to(root).as_posix()
        candidates.append(
            {
                "capture": rel.replace("_left.png", ""),
                "board_rmse_px": rmse,
                "orientation": orientation,
                "accepted_by_board_rmse": bool(rmse <= max_rmse),
                "H_right_lwir_to_left_rgb": H,
            }
        )

    accepted = [row for row in candidates if row["accepted_by_board_rmse"]]
    accepted.sort(key=lambda row: float(row["board_rmse_px"]))
    return accepted[:max_candidates]


def evaluate_capture_homographies(capture_rows: list[dict], sample_rows, target_size: tuple[int, int], crop_offset: tuple[int, int]) -> list[dict]:
    metric_rows = []
    for cap in capture_rows:
        H = cap["H_right_lwir_to_left_rgb"]
        for row in sample_rows:
            raw_lwir = normalize_u8(imread_unicode(row.raw_thermal16_path, cv2.IMREAD_UNCHANGED))
            aligned_lwir = normalize_u8(imread_unicode(row.aligned_t16_path, cv2.IMREAD_UNCHANGED))
            warped = cv2.warpPerspective(raw_lwir, H, (1280, 720), flags=cv2.INTER_LINEAR, borderValue=0)
            valid_full = cv2.warpPerspective(
                np.ones(raw_lwir.shape[:2], dtype=np.uint8) * 255,
                H,
                (1280, 720),
                flags=cv2.INTER_NEAREST,
                borderValue=0,
            ) > 0
            candidate = crop_with_border(warped, crop_offset, target_size)
            valid = crop_mask_with_border(valid_full, crop_offset, target_size)
            metrics = direct_alignment_metrics(aligned_lwir, candidate, valid)
            metric_rows.append(
                {
                    "sample": sample_id(row),
                    "capture": cap["capture"],
                    "board_rmse_px": float(cap["board_rmse_px"]),
                    "orientation": cap["orientation"],
                    "crop_offset_xy": json.dumps(tuple2(crop_offset)),
                    "eval_lwir_to_mm5_aligned_t16_ncc": float(metrics["ncc"]),
                    "eval_lwir_to_mm5_aligned_t16_edge_distance": float(metrics["edge_distance"]),
                    "eval_lwir_to_mm5_aligned_t16_mi": float(metrics["mi"]),
                }
            )
    return metric_rows


def summarize_capture_probe(rows: list[dict]) -> list[dict]:
    grouped: dict[str, list[dict]] = {}
    for row in rows:
        grouped.setdefault(str(row["capture"]), []).append(row)
    summary = []
    for capture, items in grouped.items():
        ncc = [float(row["eval_lwir_to_mm5_aligned_t16_ncc"]) for row in items]
        edge = [float(row["eval_lwir_to_mm5_aligned_t16_edge_distance"]) for row in items]
        summary.append(
            {
                "capture": capture,
                "samples": len(items),
                "board_rmse_px": float(items[0]["board_rmse_px"]),
                "orientation": items[0]["orientation"],
                "eval_lwir_to_mm5_aligned_t16_ncc_mean": float(np.mean(ncc)),
                "eval_lwir_to_mm5_aligned_t16_ncc_min": float(np.min(ncc)),
                "eval_lwir_to_mm5_aligned_t16_edge_distance_mean": float(np.mean(edge)),
            }
        )
    summary.sort(key=lambda row: (float(row["board_rmse_px"]), -float(row["eval_lwir_to_mm5_aligned_t16_ncc_mean"])))
    return summary


def save_candidate_panel(output_dir: Path, row, candidate, aligned_rgb: np.ndarray, aligned_lwir_u8: np.ndarray) -> None:
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


def write_report(output_dir: Path, summary_rows: list[dict], capture_summary: list[dict], bridge_metrics: dict) -> None:
    best = summary_rows[0] if summary_rows else None
    baseline = next((row for row in summary_rows if row["candidate"] == "shared_rgb_optimal_alpha0_0"), None)
    capture_best = max(capture_summary, key=lambda row: float(row["eval_lwir_to_mm5_aligned_t16_ncc_mean"])) if capture_summary else None
    capture_board_selected = capture_summary[0] if capture_summary else None
    lines = [
        "# Phase 21 Calibration-Only Canvas Optimization",
        "",
        "## Constraint",
        "Generation candidates use calibration files, original calibration-board captures, output geometry, and raw inputs only. MM5 aligned images are used only for evaluation.",
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
    if baseline:
        lines.extend(
            [
                "",
                "## Baseline",
                f"- candidate: `{baseline['candidate']}`",
                f"- RGB NCC mean: `{baseline.get('eval_rgb_to_mm5_aligned_rgb_ncc_mean', float('nan')):.4f}`",
                f"- LWIR NCC mean: `{baseline.get('eval_lwir_to_mm5_aligned_t16_ncc_mean', float('nan')):.4f}`",
            ]
        )
    if best:
        lines.extend(
            [
                "",
                "## Best Phase 21 Canvas Candidate",
                f"- candidate: `{best['candidate']}`",
                f"- RGB NCC mean/min: `{best.get('eval_rgb_to_mm5_aligned_rgb_ncc_mean', float('nan')):.4f}` / `{best.get('eval_rgb_to_mm5_aligned_rgb_ncc_min', float('nan')):.4f}`",
                f"- LWIR NCC mean/min: `{best.get('eval_lwir_to_mm5_aligned_t16_ncc_mean', float('nan')):.4f}` / `{best.get('eval_lwir_to_mm5_aligned_t16_ncc_min', float('nan')):.4f}`",
                f"- cross RGB/LWIR NCC mean: `{best.get('eval_cross_rgb_lwir_ncc_mean', float('nan')):.4f}`",
                f"- rule source: `{best.get('rule_source', '')}`",
            ]
        )
    lines.extend(["", "## Capture-Homography Probe"])
    if capture_board_selected:
        lines.extend(
            [
                f"- best board-RMSE capture: `{capture_board_selected['capture']}`, board RMSE `{capture_board_selected['board_rmse_px']:.4f}px`, eval LWIR NCC mean `{capture_board_selected['eval_lwir_to_mm5_aligned_t16_ncc_mean']:.4f}`",
            ]
        )
    if capture_best:
        lines.extend(
            [
                f"- best eval capture: `{capture_best['capture']}`, board RMSE `{capture_best['board_rmse_px']:.4f}px`, eval LWIR NCC mean `{capture_best['eval_lwir_to_mm5_aligned_t16_ncc_mean']:.4f}`",
                "- This probe is kept as evidence only; it does not beat the rectification baseline.",
            ]
        )
    lines.extend(
        [
            "",
            "## Top Canvas Candidates",
            "",
            "| candidate | RGB NCC mean | LWIR NCC mean | cross NCC mean | source |",
            "|---|---:|---:|---:|---|",
        ]
    )
    for row in summary_rows:
        lines.append(
            "| {candidate} | {rgb:.4f} | {lwir:.4f} | {cross:.4f} | {source} |".format(
                candidate=row["candidate"],
                rgb=row.get("eval_rgb_to_mm5_aligned_rgb_ncc_mean", float("nan")),
                lwir=row.get("eval_lwir_to_mm5_aligned_t16_ncc_mean", float("nan")),
                cross=row.get("eval_cross_rgb_lwir_ncc_mean", float("nan")),
                source=row.get("rule_source", ""),
            )
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "phase21_canvas_optimization_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 21 strict calibration-only canvas optimization.")
    parser.add_argument("--index", default="mm5_calib_benchmark/outputs/mm5_benchmark/splits/index_with_splits.csv")
    parser.add_argument("--aligned-ids", default="106,104,103")
    parser.add_argument("--calibration", default="calibration/def_stereocalib_THERM.yml")
    parser.add_argument("--thermal-camera-calibration", default="calibration/def_thermalcam_ori.yml")
    parser.add_argument("--calibration-root", default="")
    parser.add_argument("--target-size", default="640x480")
    parser.add_argument("--output", default="darklight_mm5/calibration_only_method/outputs_phase21_canvas")
    parser.add_argument("--bridge-metrics", default="darklight_mm5/outputs/metrics/registration_stages.csv")
    parser.add_argument("--capture-max-rmse", type=float, default=2.2)
    parser.add_argument("--capture-max-candidates", type=int, default=12)
    args = parser.parse_args()

    target_size = parse_size_arg(args.target_size)
    if target_size is None:
        raise ValueError("--target-size must be WxH")

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = choose_rows(args.index, args.aligned_ids)
    calibration = load_stereo_calibration(args.calibration)
    rectification = read_rectification_matrices(args.calibration)
    thermal_model = read_single_camera_model(args.thermal_camera_calibration, "thermal_ori")
    if thermal_model is None:
        raise FileNotFoundError(args.thermal_camera_calibration)

    metric_rows = []
    candidate_cache = {}
    for row in rows:
        print(f"processing {sample_id(row)} phase21 canvas candidates")
        raw_rgb = imread_unicode(row.raw_rgb1_path, cv2.IMREAD_COLOR)
        raw_lwir_u8 = normalize_u8(imread_unicode(row.raw_thermal16_path, cv2.IMREAD_UNCHANGED))
        aligned_rgb = imread_unicode(row.aligned_rgb1_path, cv2.IMREAD_COLOR)
        aligned_lwir_u8 = normalize_u8(imread_unicode(row.aligned_t16_path, cv2.IMREAD_UNCHANGED))
        lwir_model = build_lwir_models(calibration, raw_rgb.shape, raw_lwir_u8.shape, [], thermal_model)[0]
        rules = build_canvas_rules(raw_rgb, raw_lwir_u8, calibration, rectification, target_size, lwir_model, [])
        rule_map = {rule.name: rule for rule in rules}
        for rule in phase21_rules(rule_map):
            candidate = make_candidate_from_rule(raw_rgb, raw_lwir_u8, rectification, target_size, lwir_model, rule)
            metrics = evaluate_candidate(row, candidate, aligned_rgb, aligned_lwir_u8)
            metric_rows.append(metrics)
            candidate_cache[(sample_id(row), candidate.name)] = candidate

    summary_rows = summarize(metric_rows)
    write_csv(output_dir / "metrics" / "phase21_canvas_candidates.csv", metric_rows, collect_fieldnames(metric_rows, []))
    write_csv(output_dir / "metrics" / "phase21_canvas_summary.csv", summary_rows, collect_fieldnames(summary_rows, []))

    calibration_root = args.calibration_root.strip() or default_calibration_root(args.index)
    capture_candidates = capture_homography_candidates(calibration_root, args.capture_max_rmse, args.capture_max_candidates)
    crop_offset = (296, 103)
    capture_metric_rows = evaluate_capture_homographies(capture_candidates, rows, target_size, crop_offset) if capture_candidates else []
    capture_summary = summarize_capture_probe(capture_metric_rows)
    capture_dump_rows = []
    for row in capture_summary:
        capture_dump_rows.append(dict(row))
    write_csv(output_dir / "metrics" / "capture_homography_probe.csv", capture_metric_rows, collect_fieldnames(capture_metric_rows, []))
    write_csv(output_dir / "metrics" / "capture_homography_summary.csv", capture_dump_rows, collect_fieldnames(capture_dump_rows, []))

    payload = {
        "bridge_metrics": load_bridge_metrics(args.bridge_metrics),
        "best_phase21_candidate": summary_rows[0] if summary_rows else None,
        "candidate_summary": summary_rows,
        "capture_homography_summary": capture_summary,
        "calibration_root": calibration_root,
    }
    (output_dir / "metrics" / "phase21_canvas_summary.json").parent.mkdir(parents=True, exist_ok=True)
    (output_dir / "metrics" / "phase21_canvas_summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    top_names = {row["candidate"] for row in summary_rows[:2]}
    for row in rows:
        aligned_rgb = imread_unicode(row.aligned_rgb1_path, cv2.IMREAD_COLOR)
        aligned_lwir_u8 = normalize_u8(imread_unicode(row.aligned_t16_path, cv2.IMREAD_UNCHANGED))
        for name in top_names:
            candidate = candidate_cache.get((sample_id(row), name))
            if candidate is not None:
                save_candidate_panel(output_dir, row, candidate, aligned_rgb, aligned_lwir_u8)

    write_report(output_dir, summary_rows, capture_summary, load_bridge_metrics(args.bridge_metrics))
    print(f"best phase21 candidate: {summary_rows[0]['candidate'] if summary_rows else 'none'}")
    print("done")


if __name__ == "__main__":
    main()
