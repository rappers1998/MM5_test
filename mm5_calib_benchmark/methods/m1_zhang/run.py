from __future__ import annotations

from ...pipeline import _read_scene_assets
from ..alignment import apply_scene_tuning, compute_homography_alignment, scene_tune_kwargs
from ..board import calibrate_stereo_from_board, calibration_root_from_rows, load_board_observations, official_stereo


def calibrate(config: dict, rows, track: str) -> dict:
    board_payload = load_board_observations(calibration_root_from_rows(rows, config), track, float(config["method"].get("square_size_mm", 25.0)))
    calibrated = calibrate_stereo_from_board(board_payload)
    if calibrated is None:
        return {"stereo": official_stereo(config, track), "calibration_metrics": {"reprojection_mae_px": 0.0, "checkerboard_corner_rmse_px": 0.0}}
    return calibrated


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
                "coarse_radius_px": 18,
                "coarse_step_px": 6,
                "fine_radius_px": 3,
                "fine_step_px": 1,
                "coarse_scales": [0.94, 0.98, 1.0, 1.02, 1.06],
                "coarse_angles_deg": [-1.0, 0.0, 1.0],
                "fine_scale_delta": 0.02,
                "fine_angle_delta_deg": 0.5,
            },
        ),
    )
    return {"assets": assets, **tuned}
