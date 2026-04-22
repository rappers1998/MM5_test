from __future__ import annotations

import cv2
import numpy as np

from ...common import modality_preprocess
from ...pipeline import _read_scene_assets, load_saved_stereo
from ..alignment import apply_scene_tuning, compute_homography_alignment
from ..board import calibration_root_from_rows, evaluate_stereo_on_board, load_board_observations, official_stereo


def calibrate(config: dict, rows, track: str) -> dict:
    stereo = load_saved_stereo(config, "m1", track) or official_stereo(config, track)
    board_payload = load_board_observations(calibration_root_from_rows(rows, config), track, float(config["method"].get("square_size_mm", 25.0)))
    metrics = evaluate_stereo_on_board(stereo, board_payload)
    return {"stereo": stereo, "calibration_metrics": metrics}


def _feature_refine(base_img: np.ndarray, target_img: np.ndarray):
    src = modality_preprocess(base_img, "rgb")
    tgt = modality_preprocess(target_img, "thermal")
    try:
        detector = cv2.SIFT_create(nfeatures=800)
        norm = cv2.NORM_L2
    except AttributeError:
        detector = cv2.ORB_create(nfeatures=1000)
        norm = cv2.NORM_HAMMING

    key1, desc1 = detector.detectAndCompute(src, None)
    key2, desc2 = detector.detectAndCompute(tgt, None)
    if desc1 is None or desc2 is None or len(key1) < 8 or len(key2) < 8:
        return None

    matcher = cv2.BFMatcher(norm)
    if norm == cv2.NORM_L2:
        matches = matcher.knnMatch(desc1, desc2, k=2)
        good = [m for m, n in matches if m.distance < 0.78 * n.distance]
    else:
        good = sorted(matcher.match(desc1, desc2), key=lambda m: m.distance)[:64]
    if len(good) < 8:
        return None

    src_pts = np.float32([key1[m.queryIdx].pt for m in good])
    dst_pts = np.float32([key2[m.trainIdx].pt for m in good])
    affine, _ = cv2.estimateAffinePartial2D(src_pts, dst_pts, method=cv2.RANSAC, ransacReprojThreshold=6.0, maxIters=5000)
    if affine is None:
        return None
    sx = np.linalg.norm(affine[:, 0])
    sy = np.linalg.norm(affine[:, 1])
    tx = abs(float(affine[0, 2]))
    ty = abs(float(affine[1, 2]))
    if not (0.85 <= sx <= 1.15 and 0.85 <= sy <= 1.15 and tx <= 80.0 and ty <= 80.0):
        return None
    return affine


def compute_scene_result(config: dict, row, track: str, context: dict) -> dict:
    assets = _read_scene_assets(row, track)
    target_size = (assets["target_image"].shape[1], assets["target_image"].shape[0])
    result = compute_homography_alignment(assets, context["stereo"], float(config["runtime"]["plane_depth_mm"]))
    pred_mask = result["pred_mask"]
    warped_source = result["warped_source"]
    valid_mask = result["valid_mask"]

    affine = _feature_refine(warped_source, assets["target_image"])
    if affine is not None:
        pred_mask = cv2.warpAffine(pred_mask, affine, target_size, flags=cv2.INTER_NEAREST, borderValue=0)
        warped_source = cv2.warpAffine(warped_source, affine, target_size, flags=cv2.INTER_LINEAR, borderValue=0)
        valid_mask = (cv2.warpAffine((valid_mask * 255).astype(np.uint8), affine, target_size, flags=cv2.INTER_NEAREST, borderValue=0) > 0).astype(np.uint8)
    tuned = apply_scene_tuning(
        {
            "pred_mask": pred_mask.astype(np.uint8),
            "warped_source": warped_source,
            "valid_mask": valid_mask.astype(np.uint8),
            "debug": {
                "alignment": "homography_plus_feature_affine",
                "feature_refine_used": affine is not None,
            },
        },
        assets["target_image"],
        track,
        coarse_radius_px=12,
        coarse_step_px=4,
        fine_radius_px=2,
        fine_step_px=1,
        coarse_scales=[0.96, 0.99, 1.0, 1.01, 1.04],
    )
    return {"assets": assets, **tuned}
