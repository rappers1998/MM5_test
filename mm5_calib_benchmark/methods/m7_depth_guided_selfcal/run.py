from __future__ import annotations

import json
from pathlib import Path

from ...pipeline import load_saved_stereo
from ..board import calibration_root_from_rows, evaluate_stereo_on_board, load_board_observations, official_stereo
from .core import compute_m7_scene_result, m7_output_dir, optimize_global_pose


def calibrate(config: dict, rows, track: str) -> dict:
    if track != "thermal":
        raise ValueError("M7 only supports thermal track")

    out_dir = m7_output_dir(config, track)
    cache_path = out_dir / "calib" / "pose_refine_cache.json"
    saved_stereo = load_saved_stereo(config, "m7", track)
    if saved_stereo is not None and cache_path.exists():
        cache_payload = json.loads(cache_path.read_text(encoding="utf-8"))
        return {
            "stereo": saved_stereo,
            "calibration_metrics": {
                "reprojection_mae_px": float(cache_payload.get("summary_final", {}).get("checkerboard_corner_rmse_px", 0.0)),
                "checkerboard_corner_rmse_px": float(cache_payload.get("summary_final", {}).get("checkerboard_corner_rmse_px", 0.0)),
            },
            "cache_payload": cache_payload,
        }

    board_payload = load_board_observations(calibration_root_from_rows(rows, config), track, float(config["method"].get("square_size_mm", 25.0)))
    base_stereo = load_saved_stereo(config, "m1", track) or official_stereo(config, track)
    refined_stereo, cache_payload = optimize_global_pose(config, rows, track, base_stereo, board_payload)
    board_metrics = evaluate_stereo_on_board(refined_stereo, board_payload)
    return {
        "stereo": refined_stereo,
        "calibration_metrics": board_metrics,
        "cache_payload": cache_payload,
    }


def compute_scene_result(config: dict, row, track: str, context: dict) -> dict:
    return compute_m7_scene_result(config, row, context["stereo"], context.get("cache_payload"))
