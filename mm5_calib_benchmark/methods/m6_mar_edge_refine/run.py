from __future__ import annotations

from ...mar_edge_refine import refine_alignment_result
from ...pipeline import load_saved_stereo
from ..board import calibration_root_from_rows, evaluate_stereo_on_board, load_board_observations, official_stereo
from ..m4_muhovic_depthbridge.run import compute_scene_result as compute_m4_scene_result


def calibrate(config: dict, rows, track: str) -> dict:
    stereo = load_saved_stereo(config, "m4", track) or load_saved_stereo(config, "m5", track) or official_stereo(config, track)
    board_payload = load_board_observations(calibration_root_from_rows(rows, config), track, float(config["method"].get("square_size_mm", 25.0)))
    metrics = evaluate_stereo_on_board(stereo, board_payload)
    return {"stereo": stereo, "calibration_metrics": metrics}


def compute_scene_result(config: dict, row, track: str, context: dict) -> dict:
    base_context = {"stereo": load_saved_stereo(config, "m4", track) or context["stereo"]}
    base_result = compute_m4_scene_result(config, row, track, base_context)
    refined = refine_alignment_result(
        base_result["pred_mask"],
        base_result["warped_source"],
        base_result["valid_mask"],
        base_result["assets"]["target_image"],
        track,
    )
    return {
        "assets": base_result["assets"],
        "pred_mask": refined["pred_mask"],
        "warped_source": refined["warped_source"],
        "valid_mask": refined["valid_mask"],
        "debug": {
            "base_method": "m4_depthbridge",
            **refined["debug"],
        },
    }
