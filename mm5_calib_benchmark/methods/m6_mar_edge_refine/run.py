from __future__ import annotations

from ...mar_edge_refine import refine_alignment_result
from ...pipeline import load_saved_stereo
from ..alignment import scene_tune_kwargs
from ..board import calibration_root_from_rows, evaluate_stereo_on_board, load_board_observations, official_stereo
from ..m4_muhovic_depthbridge.run import compute_scene_result as compute_m4_scene_result


def calibrate(config: dict, rows, track: str) -> dict:
    stereo = load_saved_stereo(config, "m4", track) or load_saved_stereo(config, "m5", track) or official_stereo(config, track)
    board_payload = load_board_observations(calibration_root_from_rows(rows, config), track, float(config["method"].get("square_size_mm", 25.0)))
    metrics = evaluate_stereo_on_board(stereo, board_payload)
    return {"stereo": stereo, "calibration_metrics": metrics}


def compute_scene_result(config: dict, row, track: str, context: dict) -> dict:
    base_stereo = context.get("stereo") or load_saved_stereo(config, "m4", track) or load_saved_stereo(config, "m5", track) or official_stereo(config, track)
    base_context = {"stereo": base_stereo}
    base_result = compute_m4_scene_result(config, row, track, base_context)
    refined = refine_alignment_result(
        base_result["pred_mask"],
        base_result["warped_source"],
        base_result["valid_mask"],
        base_result["assets"]["target_image"],
        track,
        tuning_kwargs=scene_tune_kwargs(
            config.get("method"),
            defaults={
                "coarse_radius_px": 12,
                "coarse_step_px": 4,
                "fine_radius_px": 2,
                "fine_step_px": 1,
                "coarse_scales": [0.97, 0.99, 1.0, 1.01, 1.03],
                "coarse_angles_deg": [-1.0, 0.0, 1.0],
                "fine_scale_delta": 0.01,
                "fine_angle_delta_deg": 0.5,
            },
        ),
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
