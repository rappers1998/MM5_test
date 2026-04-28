from __future__ import annotations

from ...pipeline import _read_scene_assets
from ..alignment import apply_scene_tuning, compute_homography_alignment, scene_tune_kwargs
from ..board import calibration_root_from_rows, evaluate_stereo_on_board, load_board_observations, official_stereo


def calibrate(config: dict, rows, track: str) -> dict:
    stereo = official_stereo(config, track)
    board_payload = load_board_observations(calibration_root_from_rows(rows, config), track, float(config["method"].get("square_size_mm", 25.0)))
    metrics = evaluate_stereo_on_board(stereo, board_payload)
    return {"stereo": stereo, "calibration_metrics": metrics}


def compute_scene_result(config: dict, row, track: str, context: dict) -> dict:
    assets = _read_scene_assets(row, track)
    result = compute_homography_alignment(assets, context["stereo"], float(config["runtime"]["plane_depth_mm"]))
    tuned = apply_scene_tuning(
        result,
        assets["target_image"],
        track,
        **scene_tune_kwargs(
            config.get("method"),
            defaults={
                "coarse_radius_px": 28,
                "coarse_step_px": 8,
                "fine_radius_px": 4,
                "fine_step_px": 2,
                "coarse_scales": [0.88, 0.94, 1.0, 1.06, 1.12],
                "coarse_angles_deg": [-1.5, 0.0, 1.5],
                "fine_scale_delta": 0.02,
                "fine_angle_delta_deg": 0.5,
            },
        ),
    )
    return {"assets": assets, **tuned}
