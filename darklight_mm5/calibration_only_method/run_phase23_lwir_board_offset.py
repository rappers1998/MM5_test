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


@dataclass
class BoardOffsetObservation:
    capture: str
    residual_rmse_px: float
    orientation: str
    median_offset_xy: tuple[float, float]
    mean_offset_xy: tuple[float, float]


def offset_json(offset: tuple[int, int]) -> str:
    return json.dumps([int(offset[0]), int(offset[1])])


def collect_board_offsets(
    calibration_root: Path,
    rgb_offset: tuple[int, int],
    thermal_model,
    rectification,
    max_residual_rmse_px: float,
) -> list[BoardOffsetObservation]:
    capture_root = calibration_root / "capture_THERM" / "1280x720"
    if not capture_root.exists():
        return []

    rgb_offset_arr = np.asarray(rgb_offset, dtype=np.float64)
    observations: list[BoardOffsetObservation] = []
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

        best: tuple[float, str, np.ndarray, np.ndarray] | None = None
        for orientation, right_variant in corner_order_variants(right_points):
            thermal_rectified = cv2.undistortPoints(
                right_variant.reshape(-1, 1, 2),
                thermal_model.K,
                thermal_model.D.reshape(-1, 1),
                R=rectification.R2,
                P=rectification.P2,
            ).reshape(-1, 2)
            per_corner_offset = thermal_rectified - left_points + rgb_offset_arr
            median_offset = np.median(per_corner_offset, axis=0)
            residual = per_corner_offset - median_offset
            rmse = float(np.sqrt(np.mean(np.sum(residual * residual, axis=1))))
            if best is None or rmse < best[0]:
                best = (rmse, orientation, median_offset, np.mean(per_corner_offset, axis=0))

        if best is None or best[0] > max_residual_rmse_px:
            continue
        observations.append(
            BoardOffsetObservation(
                capture=left_path.relative_to(capture_root).as_posix(),
                residual_rmse_px=best[0],
                orientation=best[1],
                median_offset_xy=(float(best[2][0]), float(best[2][1])),
                mean_offset_xy=(float(best[3][0]), float(best[3][1])),
            )
        )
    observations.sort(key=lambda obs: obs.residual_rmse_px)
    return observations


def round_offset(values: np.ndarray, mode: str) -> tuple[int, int]:
    if mode == "floor":
        out = np.floor(values)
    elif mode == "ceil":
        out = np.ceil(values)
    else:
        out = np.round(values)
    return int(out[0]), int(out[1])


def add_offset_candidate(
    candidates: list[dict],
    seen: set[tuple[int, int]],
    name: str,
    offset: tuple[int, int],
    source: str,
) -> None:
    key = (int(offset[0]), int(offset[1]))
    if key in seen:
        return
    seen.add(key)
    candidates.append({"candidate": name, "offset_xy": key, "source": source})


def build_offset_candidates(
    observations: list[BoardOffsetObservation],
    phase21_lwir_offset: tuple[int, int],
) -> list[dict]:
    candidates: list[dict] = []
    seen: set[tuple[int, int]] = set()
    add_offset_candidate(candidates, seen, "phase21_ceiling_current", phase21_lwir_offset, "Phase 21 LWIR crop offset")
    if not observations:
        return candidates

    groups: list[tuple[str, list[BoardOffsetObservation]]] = [
        ("all", observations),
        ("best8", observations[:8]),
        ("best16", observations[:16]),
    ]
    for prefix in ("0.30m", "0.50m", "0.70m", "0.90m", "mixed"):
        items = [obs for obs in observations if obs.capture.startswith(f"{prefix}/")]
        if items:
            groups.append((prefix.replace(".", "_"), items))

    for group_name, items in groups:
        if not items:
            continue
        median_offsets = np.asarray([obs.median_offset_xy for obs in items], dtype=np.float64)
        weights = 1.0 / np.maximum(np.asarray([obs.residual_rmse_px for obs in items], dtype=np.float64), 1e-6)
        stats = {
            "median": np.median(median_offsets, axis=0),
            "mean": np.mean(median_offsets, axis=0),
            "weighted_mean": np.average(median_offsets, axis=0, weights=weights),
        }
        for stat_name, values in stats.items():
            for rounding in ("floor", "round", "ceil"):
                name = f"board_{group_name}_{stat_name}_{rounding}"
                offset = round_offset(values, rounding)
                source = f"{group_name} {stat_name} of rectified checkerboard offsets, integer {rounding}"
                add_offset_candidate(candidates, seen, name, offset, source)
    return candidates


def make_fixed_rgb_lwir_candidate(
    raw_rgb: np.ndarray,
    raw_lwir_u8: np.ndarray,
    rgb_offset: tuple[int, int],
    lwir_offset: tuple[int, int],
    candidate_name: str,
    offset_source: str,
    thermal_model,
    rectification,
    target_size: tuple[int, int],
) -> CandidateOutput:
    rgb_valid_full = np.ones(raw_rgb.shape[:2], dtype=bool)
    rgb = crop_with_border(raw_rgb, rgb_offset, target_size)
    rgb_valid = crop_mask_with_border(rgb_valid_full, rgb_offset, target_size)
    lwir, lwir_valid = rectified_remap(
        raw_lwir_u8,
        thermal_model.K,
        thermal_model.D,
        rectification.R2,
        rectification.P2,
        lwir_offset,
        target_size,
        cv2.INTER_LINEAR,
    )
    method = "phase21_strict_calibration_ceiling" if candidate_name == "phase21_ceiling_current" else "board_offset_lwir_only"
    return CandidateOutput(
        name=candidate_name,
        rgb=rgb,
        lwir=lwir,
        rgb_valid=rgb_valid,
        lwir_valid=lwir_valid,
        metadata={
            "method": method,
            "allowed_for_generation": True,
            "rgb_source": "phase21_fixed_rgb_canvas",
            "lwir_source": "stored_rectified_lwir_with_board_derived_crop",
            "rgb_crop_offset_xy": tuple2(rgb_offset),
            "lwir_crop_offset_xy": tuple2(lwir_offset),
            "lwir_crop_mode": candidate_name,
            "rule_source": offset_source,
        },
    )


def save_panel(output_dir: Path, row, candidate: CandidateOutput, aligned_rgb: np.ndarray, aligned_lwir_u8: np.ndarray) -> None:
    aligned_lwir_bgr = cv2.cvtColor(aligned_lwir_u8, cv2.COLOR_GRAY2BGR)
    make_five_panel(
        [
            (candidate.rgb, "Fixed RGB"),
            (candidate.lwir, "Board-offset LWIR"),
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
    observation_rows: list[dict],
    bridge_metrics: dict,
) -> None:
    baseline = next((row for row in summary_rows if row["candidate"] == "phase21_ceiling_current"), None)
    best = summary_rows[0] if summary_rows else None
    board_best = next((row for row in summary_rows if row.get("method") == "board_offset_lwir_only"), None)
    promoted = next((row for row in summary_rows if row["candidate"] == "board_all_median_floor"), board_best)

    lines = [
        "# Phase 23 LWIR Board-Offset Optimization",
        "",
        "## Constraint",
        "RGB is fixed to the Phase 21 calibration-derived canvas. LWIR crop offsets are derived from calibration-board captures only. MM5 aligned images are used only for evaluation.",
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
            "## Board Offset Calibration",
            f"- accepted checkerboard offset observations: `{len(observation_rows)}`",
            "- offset rule: rectified thermal checkerboard point minus raw RGB checkerboard point plus fixed RGB crop origin.",
        ]
    )

    if baseline:
        lines.extend(
            [
                "",
                "## Phase 21 Baseline",
                f"- RGB NCC mean/min: `{baseline.get('eval_rgb_to_mm5_aligned_rgb_ncc_mean', float('nan')):.4f}` / `{baseline.get('eval_rgb_to_mm5_aligned_rgb_ncc_min', float('nan')):.4f}`",
                f"- LWIR NCC mean/min: `{baseline.get('eval_lwir_to_mm5_aligned_t16_ncc_mean', float('nan')):.4f}` / `{baseline.get('eval_lwir_to_mm5_aligned_t16_ncc_min', float('nan')):.4f}`",
                f"- LWIR offset: `{baseline.get('lwir_crop_offset_xy', '')}`",
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
                f"- LWIR offset: `{promoted.get('lwir_crop_offset_xy', '')}`",
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
            "| candidate | method | LWIR NCC mean | LWIR NCC min | RGB NCC mean | LWIR offset | source |",
            "|---|---|---:|---:|---:|---|---|",
        ]
    )
    for row in summary_rows[:12]:
        lines.append(
            "| {candidate} | {method} | {lncc:.4f} | {lmin:.4f} | {rncc:.4f} | {offset} | {source} |".format(
                candidate=row["candidate"],
                method=row.get("method", ""),
                lncc=row.get("eval_lwir_to_mm5_aligned_t16_ncc_mean", float("nan")),
                lmin=row.get("eval_lwir_to_mm5_aligned_t16_ncc_min", float("nan")),
                rncc=row.get("eval_rgb_to_mm5_aligned_rgb_ncc_mean", float("nan")),
                offset=row.get("lwir_crop_offset_xy", ""),
                source=row.get("rule_source", ""),
            )
        )
    (output_dir / "phase23_lwir_board_offset_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 23 LWIR-only board-derived crop optimization.")
    parser.add_argument("--index", default="mm5_calib_benchmark/outputs/mm5_benchmark/splits/index_with_splits.csv")
    parser.add_argument("--aligned-ids", default="106,104,103")
    parser.add_argument("--calibration", default="calibration/def_stereocalib_THERM.yml")
    parser.add_argument("--thermal-camera-calibration", default="calibration/def_thermalcam_ori.yml")
    parser.add_argument("--calibration-root", default="")
    parser.add_argument("--target-size", default="640x480")
    parser.add_argument("--max-board-offset-rmse-px", type=float, default=12.0)
    parser.add_argument("--output", default="darklight_mm5/calibration_only_method/outputs_phase23_lwir_board_offset")
    parser.add_argument("--bridge-metrics", default="darklight_mm5/outputs/metrics/registration_stages.csv")
    parser.add_argument("--save-top", type=int, default=5)
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
    phase21_lwir_offset = phase21_rule.lwir_offset

    calibration_root = Path(args.calibration_root.strip() or default_calibration_root(args.index))
    observations = collect_board_offsets(
        calibration_root,
        rgb_offset,
        thermal_model,
        rectification,
        args.max_board_offset_rmse_px,
    )
    observation_rows = [
        {
            "capture": obs.capture,
            "residual_rmse_px": obs.residual_rmse_px,
            "orientation": obs.orientation,
            "median_offset_xy": json.dumps([float(obs.median_offset_xy[0]), float(obs.median_offset_xy[1])]),
            "mean_offset_xy": json.dumps([float(obs.mean_offset_xy[0]), float(obs.mean_offset_xy[1])]),
        }
        for obs in observations
    ]
    write_csv(output_dir / "metrics" / "phase23_board_offset_observations.csv", observation_rows, collect_fieldnames(observation_rows, []))

    offset_candidates = build_offset_candidates(observations, phase21_lwir_offset)
    offset_rows = [
        {"candidate": row["candidate"], "offset_xy": offset_json(row["offset_xy"]), "source": row["source"]}
        for row in offset_candidates
    ]
    write_csv(output_dir / "metrics" / "phase23_board_offset_candidates.csv", offset_rows, collect_fieldnames(offset_rows, []))

    metric_rows: list[dict] = []
    candidate_cache: dict[tuple[str, str], CandidateOutput] = {}
    for row in rows:
        print(f"processing {sample_id(row)} phase23 LWIR board-offset candidates")
        raw_rgb = imread_unicode(row.raw_rgb1_path, cv2.IMREAD_COLOR)
        raw_lwir_u8 = normalize_u8(imread_unicode(row.raw_thermal16_path, cv2.IMREAD_UNCHANGED))
        aligned_rgb = imread_unicode(row.aligned_rgb1_path, cv2.IMREAD_COLOR)
        aligned_lwir_u8 = normalize_u8(imread_unicode(row.aligned_t16_path, cv2.IMREAD_UNCHANGED))
        for cand in offset_candidates:
            candidate = make_fixed_rgb_lwir_candidate(
                raw_rgb,
                raw_lwir_u8,
                rgb_offset,
                cand["offset_xy"],
                cand["candidate"],
                cand["source"],
                thermal_model,
                rectification,
                target_size,
            )
            metrics = add_metadata_to_metrics(evaluate_candidate(row, candidate, aligned_rgb, aligned_lwir_u8), candidate)
            metric_rows.append(metrics)
            candidate_cache[(sample_id(row), candidate.name)] = candidate

    summary_rows = summarize(metric_rows)
    write_csv(output_dir / "metrics" / "phase23_lwir_board_offset_metrics.csv", metric_rows, collect_fieldnames(metric_rows, []))
    write_csv(output_dir / "metrics" / "phase23_lwir_board_offset_summary.csv", summary_rows, collect_fieldnames(summary_rows, []))
    payload = {
        "constraint": "fixed_phase21_rgb_board_derived_lwir_crop_aligned_eval_only",
        "calibration_root": str(calibration_root),
        "rgb_offset_xy": list(rgb_offset),
        "phase21_lwir_offset_xy": list(phase21_lwir_offset),
        "board_observation_count": len(observations),
        "best_candidate": summary_rows[0] if summary_rows else None,
        "promoted_candidate": next((row for row in summary_rows if row["candidate"] == "board_all_median_floor"), None),
        "candidate_summary": summary_rows,
    }
    (output_dir / "metrics" / "phase23_lwir_board_offset_summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    top_names = {row["candidate"] for row in summary_rows[: max(1, args.save_top)]}
    top_names.add("board_all_median_floor")
    top_names.add("phase21_ceiling_current")
    for row in rows:
        aligned_rgb = imread_unicode(row.aligned_rgb1_path, cv2.IMREAD_COLOR)
        aligned_lwir_u8 = normalize_u8(imread_unicode(row.aligned_t16_path, cv2.IMREAD_UNCHANGED))
        for name in top_names:
            candidate = candidate_cache.get((sample_id(row), name))
            if candidate is not None:
                save_panel(output_dir, row, candidate, aligned_rgb, aligned_lwir_u8)

    write_report(output_dir, summary_rows, observation_rows, load_bridge_metrics(args.bridge_metrics))
    print(f"best phase23 candidate: {summary_rows[0]['candidate'] if summary_rows else 'none'}")
    print("done")


if __name__ == "__main__":
    main()
