from __future__ import annotations

import cv2
import numpy as np

from ...common import compute_plane_homography, modality_preprocess, valid_mask_from_homography, warp_image, warp_mask
from ...pipeline import _read_scene_assets, load_saved_stereo
from ..alignment import apply_scene_tuning, scene_tune_kwargs
from ..board import calibration_root_from_rows, evaluate_stereo_on_board, load_board_observations, official_stereo


def calibrate(config: dict, rows, track: str) -> dict:
    stereo = load_saved_stereo(config, "m1", track) or official_stereo(config, track)
    board_payload = load_board_observations(calibration_root_from_rows(rows, config), track, float(config["method"].get("square_size_mm", 25.0)))
    metrics = evaluate_stereo_on_board(stereo, board_payload)
    return {"stereo": stereo, "calibration_metrics": metrics}


def compute_scene_result(config: dict, row, track: str, context: dict) -> dict:
    assets = _read_scene_assets(row, track)
    target_h, target_w = assets["target_mask"].shape
    homography = compute_plane_homography(context["stereo"], (assets["source_image"].shape[1], assets["source_image"].shape[0]), float(config["runtime"]["plane_depth_mm"]))
    global_mask = warp_mask(assets["source_mask"], homography, (target_w, target_h))
    global_img = warp_image(assets["source_image"], homography, (target_w, target_h))
    global_valid = valid_mask_from_homography((assets["source_image"].shape[1], assets["source_image"].shape[0]), homography, (target_w, target_h))

    src_proc = modality_preprocess(global_img, "rgb")
    tgt_proc = modality_preprocess(assets["target_image"], track)
    method_cfg = config.get("method", {})
    flow_cfg = method_cfg.get("flow", {}) if isinstance(method_cfg, dict) else {}
    flow = cv2.calcOpticalFlowFarneback(
        src_proc,
        tgt_proc,
        None,
        float(flow_cfg.get("pyr_scale", 0.5)),
        int(flow_cfg.get("levels", 4)),
        int(flow_cfg.get("winsize", 25)),
        int(flow_cfg.get("iterations", 5)),
        int(flow_cfg.get("poly_n", 7)),
        float(flow_cfg.get("poly_sigma", 1.5)),
        0,
    )
    max_flow_px = float(flow_cfg.get("max_flow_px", 32.0))
    if max_flow_px > 0:
        np.clip(flow, -max_flow_px, max_flow_px, out=flow)
    yy, xx = np.indices((target_h, target_w), dtype=np.float32)
    map_x = xx - flow[..., 0]
    map_y = yy - flow[..., 1]
    pred_mask = cv2.remap(global_mask.astype(np.uint8), map_x, map_y, cv2.INTER_NEAREST, borderValue=0)
    warped_source = cv2.remap(global_img, map_x, map_y, cv2.INTER_LINEAR, borderValue=0)
    valid_mask = (cv2.remap((global_valid * 255).astype(np.uint8), map_x, map_y, cv2.INTER_NEAREST, borderValue=0) > 0).astype(np.uint8)
    tuned = apply_scene_tuning(
        {
            "pred_mask": pred_mask.astype(np.uint8),
            "warped_source": warped_source,
            "valid_mask": valid_mask.astype(np.uint8),
            "debug": {
                "alignment": "homography_plus_dense_flow",
                "flow_clip_px": max_flow_px,
                "mean_flow_mag_px": float(np.linalg.norm(flow, axis=2).mean()),
            },
        },
        assets["target_image"],
        track,
        **scene_tune_kwargs(
            method_cfg,
            defaults={
                "coarse_radius_px": 10,
                "coarse_step_px": 4,
                "fine_radius_px": 2,
                "fine_step_px": 1,
                "coarse_scales": [0.98, 1.0, 1.02],
                "coarse_angles_deg": [-1.0, 0.0, 1.0],
                "fine_scale_delta": 0.01,
                "fine_angle_delta_deg": 0.5,
            },
        ),
    )
    return {"assets": assets, **tuned}
