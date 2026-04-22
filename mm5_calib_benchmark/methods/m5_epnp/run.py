from __future__ import annotations

from ...pipeline import _read_scene_assets
from ..alignment import apply_scene_tuning, compute_depth_alignment
from ..board import calibrate_stereo_from_board, calibration_root_from_rows, estimate_relative_pose_epnp, load_board_observations, official_stereo


def calibrate(config: dict, rows, track: str) -> dict:
    board_payload = load_board_observations(calibration_root_from_rows(rows, config), track, float(config["method"].get("square_size_mm", 25.0)))
    base_context = calibrate_stereo_from_board(board_payload)
    if base_context is None:
        return {"stereo": official_stereo(config, track), "calibration_metrics": {"reprojection_mae_px": 0.0, "checkerboard_corner_rmse_px": 0.0}}
    return estimate_relative_pose_epnp(base_context) or base_context


def compute_scene_result(config: dict, row, track: str, context: dict) -> dict:
    assets = _read_scene_assets(row, track)
    depth_result = compute_depth_alignment(
        assets,
        context["stereo"],
        float(config["runtime"]["plane_depth_mm"]),
        seed_with_homography=True,
        fill_holes=True,
        fill_distance_px=6.0,
        support_dilate_ksize=7,
        splat_radius=0,
    )
    tuned = apply_scene_tuning(
        depth_result,
        assets["target_image"],
        track,
        coarse_radius_px=14,
        coarse_step_px=4,
        fine_radius_px=2,
        fine_step_px=1,
        coarse_scales=[0.95, 0.98, 1.0, 1.02, 1.05],
    )
    return {"assets": assets, **tuned}
