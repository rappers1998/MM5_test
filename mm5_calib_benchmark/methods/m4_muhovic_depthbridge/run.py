from __future__ import annotations

from ...pipeline import _read_scene_assets, load_saved_stereo
from ..alignment import apply_scene_tuning, compute_depth_alignment
from ..board import calibration_root_from_rows, evaluate_stereo_on_board, load_board_observations, official_stereo


def calibrate(config: dict, rows, track: str) -> dict:
    stereo = load_saved_stereo(config, "m5", track) or load_saved_stereo(config, "m1", track) or official_stereo(config, track)
    board_payload = load_board_observations(calibration_root_from_rows(rows, config), track, float(config["method"].get("square_size_mm", 25.0)))
    metrics = evaluate_stereo_on_board(stereo, board_payload)
    return {"stereo": stereo, "calibration_metrics": metrics}


def compute_scene_result(config: dict, row, track: str, context: dict) -> dict:
    assets = _read_scene_assets(row, track)
    depth_result = compute_depth_alignment(
        assets,
        context["stereo"],
        float(config["runtime"]["plane_depth_mm"]),
        seed_with_homography=True,
        fill_holes=True,
        fill_distance_px=10.0,
        support_dilate_ksize=11,
        splat_radius=1,
    )
    tuned = apply_scene_tuning(
        depth_result,
        assets["target_image"],
        track,
        coarse_radius_px=10,
        coarse_step_px=4,
        fine_radius_px=2,
        fine_step_px=1,
        coarse_scales=[0.97, 0.99, 1.0, 1.01, 1.03],
    )
    return {"assets": assets, **tuned}
