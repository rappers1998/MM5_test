from __future__ import annotations

import cv2
import numpy as np

from ...common import compute_plane_homography, modality_preprocess, valid_mask_from_homography, warp_image, warp_mask
from ...pipeline import _read_scene_assets, load_saved_stereo
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
    flow = cv2.calcOpticalFlowFarneback(src_proc, tgt_proc, None, 0.5, 4, 25, 5, 7, 1.5, 0)
    yy, xx = np.indices((target_h, target_w), dtype=np.float32)
    map_x = xx - flow[..., 0]
    map_y = yy - flow[..., 1]
    pred_mask = cv2.remap(global_mask.astype(np.uint8), map_x, map_y, cv2.INTER_NEAREST, borderValue=0)
    warped_source = cv2.remap(global_img, map_x, map_y, cv2.INTER_LINEAR, borderValue=0)
    valid_mask = (cv2.remap((global_valid * 255).astype(np.uint8), map_x, map_y, cv2.INTER_NEAREST, borderValue=0) > 0).astype(np.uint8)
    return {"assets": assets, "pred_mask": pred_mask.astype(np.uint8), "warped_source": warped_source, "valid_mask": valid_mask.astype(np.uint8)}
