from __future__ import annotations

import cv2
import numpy as np

from ...common import modality_preprocess
from ...pipeline import _read_scene_assets, load_saved_stereo
from ..alignment import apply_scene_tuning, compute_homography_alignment, scene_tune_kwargs
from ..board import calibration_root_from_rows, evaluate_stereo_on_board, load_board_observations, official_stereo


def calibrate(config: dict, rows, track: str) -> dict:
    stereo = load_saved_stereo(config, "m1", track) or official_stereo(config, track)
    board_payload = load_board_observations(calibration_root_from_rows(rows, config), track, float(config["method"].get("square_size_mm", 25.0)))
    metrics = evaluate_stereo_on_board(stereo, board_payload)
    return {"stereo": stereo, "calibration_metrics": metrics}


def _feature_refine_settings(method_cfg: dict | None) -> dict:
    cfg = method_cfg.get("feature_refine", {}) if isinstance(method_cfg, dict) else {}
    return {
        "use_clahe": bool(cfg.get("use_clahe", True)),
        "sift_nfeatures": int(cfg.get("sift_nfeatures", 1200)),
        "orb_nfeatures": int(cfg.get("orb_nfeatures", 1400)),
        "ratio_test": float(cfg.get("ratio_test", 0.76)),
        "max_matches": int(cfg.get("max_matches", 96)),
        "min_matches": int(cfg.get("min_matches", 10)),
        "ransac_reproj_threshold": float(cfg.get("ransac_reproj_threshold", 4.0)),
        "max_iters": int(cfg.get("max_iters", 8000)),
        "scale_min": float(cfg.get("scale_min", 0.90)),
        "scale_max": float(cfg.get("scale_max", 1.10)),
        "max_translation_px": float(cfg.get("max_translation_px", 60.0)),
    }


def _feature_refine(base_img: np.ndarray, target_img: np.ndarray, feature_cfg: dict):
    src = modality_preprocess(base_img, "rgb")
    tgt = modality_preprocess(target_img, "thermal")
    if feature_cfg["use_clahe"]:
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        src = clahe.apply(src)
        tgt = clahe.apply(tgt)
    try:
        detector = cv2.SIFT_create(nfeatures=feature_cfg["sift_nfeatures"])
        norm = cv2.NORM_L2
    except AttributeError:
        detector = cv2.ORB_create(nfeatures=feature_cfg["orb_nfeatures"])
        norm = cv2.NORM_HAMMING

    key1, desc1 = detector.detectAndCompute(src, None)
    key2, desc2 = detector.detectAndCompute(tgt, None)
    if desc1 is None or desc2 is None or len(key1) < feature_cfg["min_matches"] or len(key2) < feature_cfg["min_matches"]:
        return None, {"used": False, "reason": "insufficient_keypoints"}

    matcher = cv2.BFMatcher(norm)
    if norm == cv2.NORM_L2:
        matches = matcher.knnMatch(desc1, desc2, k=2)
        good = [m for m, n in matches if m.distance < feature_cfg["ratio_test"] * n.distance]
    else:
        good = sorted(matcher.match(desc1, desc2), key=lambda m: m.distance)[: feature_cfg["max_matches"]]
    if len(good) < feature_cfg["min_matches"]:
        return None, {"used": False, "reason": "insufficient_matches", "match_count": len(good)}

    src_pts = np.float32([key1[m.queryIdx].pt for m in good])
    dst_pts = np.float32([key2[m.trainIdx].pt for m in good])
    affine, inliers = cv2.estimateAffinePartial2D(
        src_pts,
        dst_pts,
        method=cv2.RANSAC,
        ransacReprojThreshold=feature_cfg["ransac_reproj_threshold"],
        maxIters=feature_cfg["max_iters"],
    )
    if affine is None:
        return None, {"used": False, "reason": "ransac_failed", "match_count": len(good)}
    sx = np.linalg.norm(affine[:, 0])
    sy = np.linalg.norm(affine[:, 1])
    tx = abs(float(affine[0, 2]))
    ty = abs(float(affine[1, 2]))
    if not (
        feature_cfg["scale_min"] <= sx <= feature_cfg["scale_max"]
        and feature_cfg["scale_min"] <= sy <= feature_cfg["scale_max"]
        and tx <= feature_cfg["max_translation_px"]
        and ty <= feature_cfg["max_translation_px"]
    ):
        return None, {
            "used": False,
            "reason": "affine_guard_reject",
            "match_count": len(good),
            "sx": float(sx),
            "sy": float(sy),
            "tx": tx,
            "ty": ty,
        }
    return affine, {
        "used": True,
        "match_count": len(good),
        "inlier_count": int(inliers.sum()) if inliers is not None else 0,
        "sx": float(sx),
        "sy": float(sy),
        "tx": tx,
        "ty": ty,
    }


def compute_scene_result(config: dict, row, track: str, context: dict) -> dict:
    assets = _read_scene_assets(row, track)
    target_size = (assets["target_image"].shape[1], assets["target_image"].shape[0])
    result = compute_homography_alignment(assets, context["stereo"], float(config["runtime"]["plane_depth_mm"]))
    pred_mask = result["pred_mask"]
    warped_source = result["warped_source"]
    valid_mask = result["valid_mask"]

    feature_cfg = _feature_refine_settings(config.get("method"))
    affine, feature_debug = _feature_refine(warped_source, assets["target_image"], feature_cfg)
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
                "feature_refine": feature_debug,
            },
        },
        assets["target_image"],
        track,
        **scene_tune_kwargs(
            config.get("method"),
            defaults={
                "coarse_radius_px": 12,
                "coarse_step_px": 4,
                "fine_radius_px": 2,
                "fine_step_px": 1,
                "coarse_scales": [0.96, 0.99, 1.0, 1.01, 1.04],
                "coarse_angles_deg": [-1.0, 0.0, 1.0],
                "fine_scale_delta": 0.015,
                "fine_angle_delta_deg": 0.5,
            },
        ),
    )
    return {"assets": assets, **tuned}
