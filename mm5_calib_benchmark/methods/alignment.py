from __future__ import annotations

import cv2
import numpy as np
from scipy import ndimage

from ..common import compute_plane_homography, modality_preprocess, mutual_information, valid_mask_from_homography, warp_image, warp_mask


SCENE_TUNE_DEFAULTS = {
    "coarse_radius_px": 24,
    "coarse_step_px": 8,
    "fine_radius_px": 4,
    "fine_step_px": 2,
    "coarse_scales": [0.92, 0.96, 1.00, 1.04, 1.08],
    "coarse_angles_deg": [0.0],
    "fine_scale_delta": 0.02,
    "fine_angle_delta_deg": 0.0,
    "edge_weight": 1.0,
    "mi_weight": 0.20,
    "coverage_weight": 0.50,
}


def _coerce_float_list(value, fallback: list[float]) -> list[float]:
    if value is None:
        return list(fallback)
    if isinstance(value, (list, tuple)):
        values = [float(item) for item in value]
        return values or list(fallback)
    if isinstance(value, str):
        raw = value.strip()
        if raw.startswith("[") and raw.endswith("]"):
            raw = raw[1:-1]
        values = [float(item.strip()) for item in raw.split(",") if item.strip()]
        return values or list(fallback)
    return [float(value)]


def scene_tune_kwargs(method_cfg: dict | None, defaults: dict | None = None) -> dict:
    merged = dict(SCENE_TUNE_DEFAULTS)
    if defaults:
        merged.update(defaults)
    if isinstance(method_cfg, dict):
        scene_cfg = method_cfg.get("scene_tune", {})
        if isinstance(scene_cfg, dict):
            merged.update(scene_cfg)
    return {
        "coarse_radius_px": int(merged["coarse_radius_px"]),
        "coarse_step_px": max(1, int(merged["coarse_step_px"])),
        "fine_radius_px": int(merged["fine_radius_px"]),
        "fine_step_px": max(1, int(merged["fine_step_px"])),
        "coarse_scales": _coerce_float_list(merged.get("coarse_scales"), SCENE_TUNE_DEFAULTS["coarse_scales"]),
        "coarse_angles_deg": _coerce_float_list(merged.get("coarse_angles_deg"), SCENE_TUNE_DEFAULTS["coarse_angles_deg"]),
        "fine_scale_delta": float(merged.get("fine_scale_delta", SCENE_TUNE_DEFAULTS["fine_scale_delta"])),
        "fine_angle_delta_deg": float(merged.get("fine_angle_delta_deg", SCENE_TUNE_DEFAULTS["fine_angle_delta_deg"])),
        "edge_weight": float(merged.get("edge_weight", SCENE_TUNE_DEFAULTS["edge_weight"])),
        "mi_weight": float(merged.get("mi_weight", SCENE_TUNE_DEFAULTS["mi_weight"])),
        "coverage_weight": float(merged.get("coverage_weight", SCENE_TUNE_DEFAULTS["coverage_weight"])),
    }


def compute_homography_alignment(assets: dict, stereo: dict, plane_depth_mm: float) -> dict:
    target_h, target_w = assets["target_mask"].shape
    source_h, source_w = assets["source_mask"].shape
    homography = compute_plane_homography(stereo, (source_w, source_h), plane_depth_mm)
    return {
        "pred_mask": warp_mask(assets["source_mask"], homography, (target_w, target_h)).astype(np.uint8),
        "warped_source": warp_image(assets["source_image"], homography, (target_w, target_h)),
        "valid_mask": valid_mask_from_homography((source_w, source_h), homography, (target_w, target_h)).astype(np.uint8),
        "debug": {
            "alignment": "homography",
        },
    }


def _target_edge_map(target_image: np.ndarray, modality: str) -> np.ndarray:
    gray = modality_preprocess(target_image, modality)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    canny = cv2.Canny(blur, 40, 120) > 0
    sx = cv2.Sobel(blur, cv2.CV_32F, 1, 0, ksize=3)
    sy = cv2.Sobel(blur, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(sx, sy)
    if np.any(mag > 0):
        threshold = float(np.percentile(mag[mag > 0], 80))
        sobel = mag >= threshold
    else:
        sobel = np.zeros_like(canny)
    edge = np.logical_or(canny, sobel).astype(np.uint8)
    edge = cv2.morphologyEx(edge * 255, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    return (edge > 0).astype(np.uint8)


def _foreground_boundary(mask: np.ndarray) -> np.ndarray:
    fg = (mask > 0).astype(np.uint8)
    dil = cv2.dilate(fg, np.ones((3, 3), np.uint8), iterations=1)
    ero = cv2.erode(fg, np.ones((3, 3), np.uint8), iterations=1)
    return ((dil > 0) ^ (ero > 0)).astype(np.uint8)


def _mask_center(mask: np.ndarray) -> tuple[float, float]:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return mask.shape[1] / 2.0, mask.shape[0] / 2.0
    return float(xs.mean()), float(ys.mean())


def _apply_affine_to_mask(mask: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    return cv2.warpAffine(mask, matrix, (mask.shape[1], mask.shape[0]), flags=cv2.INTER_NEAREST, borderValue=0)


def _apply_affine_to_image(image: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    return cv2.warpAffine(image, matrix, (image.shape[1], image.shape[0]), flags=cv2.INTER_LINEAR, borderValue=0)


def _apply_affine_to_valid(valid_mask: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    warped = cv2.warpAffine((valid_mask * 255).astype(np.uint8), matrix, (valid_mask.shape[1], valid_mask.shape[0]), flags=cv2.INTER_NEAREST, borderValue=0)
    return (warped > 0).astype(np.uint8)


def auto_scene_tune(
    pred_mask: np.ndarray,
    warped_source: np.ndarray,
    valid_mask: np.ndarray,
    target_image: np.ndarray,
    modality: str,
    coarse_radius_px: int = 24,
    coarse_step_px: int = 8,
    fine_radius_px: int = 4,
    fine_step_px: int = 2,
    coarse_scales: list[float] | None = None,
    coarse_angles_deg: list[float] | None = None,
    fine_scale_delta: float = 0.02,
    fine_angle_delta_deg: float = 0.0,
    edge_weight: float = 1.0,
    mi_weight: float = 0.20,
    coverage_weight: float = 0.50,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    edge = _target_edge_map(target_image, modality)
    edge_dist = cv2.distanceTransform((1 - edge).astype(np.uint8), cv2.DIST_L2, 3)
    center = _mask_center(pred_mask)
    base_valid_ratio = float(valid_mask.mean())

    def evaluate(matrix: np.ndarray) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
        mask_t = _apply_affine_to_mask(pred_mask, matrix)
        img_t = _apply_affine_to_image(warped_source, matrix)
        valid_t = _apply_affine_to_valid(valid_mask, matrix)
        boundary = _foreground_boundary(mask_t).astype(bool) & valid_t.astype(bool)
        if not boundary.any():
            return -1e9, mask_t, img_t, valid_t
        edge_score = -float(edge_dist[boundary].mean())
        mi_score = float(mutual_information(target_image, img_t, valid_t))
        coverage_penalty = abs(float(valid_t.mean()) - base_valid_ratio)
        score = float(edge_weight) * edge_score + float(mi_weight) * mi_score - float(coverage_weight) * coverage_penalty
        return score, mask_t, img_t, valid_t

    best = {
        "score": -1e9,
        "mask": pred_mask,
        "image": warped_source,
        "valid": valid_mask,
        "scale": 1.0,
        "angle_deg": 0.0,
        "dx": 0.0,
        "dy": 0.0,
    }

    if coarse_scales is None:
        coarse_scales = list(SCENE_TUNE_DEFAULTS["coarse_scales"])
    if coarse_angles_deg is None:
        coarse_angles_deg = list(SCENE_TUNE_DEFAULTS["coarse_angles_deg"])
    coarse_dx = range(-int(coarse_radius_px), int(coarse_radius_px) + 1, int(coarse_step_px))
    coarse_dy = range(-int(coarse_radius_px), int(coarse_radius_px) + 1, int(coarse_step_px))
    for angle_deg in coarse_angles_deg:
        for scale in coarse_scales:
            if scale <= 0:
                continue
            for dx in coarse_dx:
                for dy in coarse_dy:
                    matrix = cv2.getRotationMatrix2D(center, float(angle_deg), float(scale))
                    matrix[:, 2] += [dx, dy]
                    score, mask_t, img_t, valid_t = evaluate(matrix)
                    if score > best["score"]:
                        best.update(
                            {
                                "score": score,
                                "mask": mask_t,
                                "image": img_t,
                                "valid": valid_t,
                                "scale": scale,
                                "angle_deg": angle_deg,
                                "dx": dx,
                                "dy": dy,
                            }
                        )

    fine_dx = range(int(best["dx"]) - int(fine_radius_px), int(best["dx"]) + int(fine_radius_px) + 1, int(fine_step_px))
    fine_dy = range(int(best["dy"]) - int(fine_radius_px), int(best["dy"]) + int(fine_radius_px) + 1, int(fine_step_px))
    fine_scales = sorted({float(best["scale"] - fine_scale_delta), float(best["scale"]), float(best["scale"] + fine_scale_delta)})
    if float(fine_angle_delta_deg) > 0:
        fine_angles = sorted(
            {
                float(best["angle_deg"] - fine_angle_delta_deg),
                float(best["angle_deg"]),
                float(best["angle_deg"] + fine_angle_delta_deg),
            }
        )
    else:
        fine_angles = [float(best["angle_deg"])]
    for angle_deg in fine_angles:
        for scale in fine_scales:
            if scale <= 0:
                continue
            for dx in fine_dx:
                for dy in fine_dy:
                    matrix = cv2.getRotationMatrix2D(center, float(angle_deg), float(scale))
                    matrix[:, 2] += [dx, dy]
                    score, mask_t, img_t, valid_t = evaluate(matrix)
                    if score > best["score"]:
                        best.update(
                            {
                                "score": score,
                                "mask": mask_t,
                                "image": img_t,
                                "valid": valid_t,
                                "scale": scale,
                                "angle_deg": angle_deg,
                                "dx": dx,
                                "dy": dy,
                            }
                        )

    info = {
        "dx": float(best["dx"]),
        "dy": float(best["dy"]),
        "scale": float(best["scale"]),
        "angle_deg": float(best["angle_deg"]),
        "score": float(best["score"]),
        "edge_pixels": int(edge.sum()),
    }
    return best["mask"], best["image"], best["valid"], info


def _nearest_seed_fill(
    pred_mask: np.ndarray,
    warped_source: np.ndarray,
    valid_mask: np.ndarray,
    support_mask: np.ndarray,
    max_distance_px: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    seed_mask = (valid_mask > 0) & (pred_mask > 0)
    support_mask = support_mask.astype(bool)
    if not np.any(seed_mask):
        return pred_mask, warped_source, valid_mask, {"filled_pixels": 0, "used": False, "reason": "empty_seed"}

    distances, indices = ndimage.distance_transform_edt(~seed_mask, return_distances=True, return_indices=True)
    holes = support_mask & (~seed_mask) & (distances <= float(max_distance_px))
    if not np.any(holes):
        return pred_mask, warped_source, valid_mask, {"filled_pixels": 0, "used": False, "reason": "no_holes"}

    yy = indices[0][holes]
    xx = indices[1][holes]
    filled_mask = pred_mask.copy()
    filled_image = warped_source.copy()
    filled_valid = valid_mask.copy()
    filled_mask[holes] = pred_mask[yy, xx]
    filled_image[holes] = warped_source[yy, xx]
    filled_valid[holes] = 1
    return filled_mask, filled_image, filled_valid, {"filled_pixels": int(np.count_nonzero(holes)), "used": True}


def compute_depth_alignment(
    assets: dict,
    stereo: dict,
    plane_depth_mm: float,
    *,
    seed_with_homography: bool,
    fill_holes: bool,
    fill_distance_px: float,
    support_dilate_ksize: int,
    splat_radius: int,
) -> dict:
    base = compute_homography_alignment(assets, stereo, plane_depth_mm)
    if not seed_with_homography:
        pred_mask = np.zeros_like(base["pred_mask"], dtype=np.uint8)
        warped_source = np.zeros_like(base["warped_source"])
        valid_mask = np.zeros_like(base["valid_mask"], dtype=np.uint8)
    else:
        pred_mask = base["pred_mask"].copy()
        warped_source = base["warped_source"].copy()
        valid_mask = base["valid_mask"].copy()
    real_projected_mask = np.zeros_like(valid_mask, dtype=np.uint8)

    depth = assets.get("depth_image")
    if depth is None:
        base["debug"]["depth_projection"] = {"used": False, "reason": "no_depth"}
        return base

    target_h, target_w = assets["target_mask"].shape
    stereo_cm1 = np.asarray(stereo["CM1"], dtype=np.float64)
    stereo_cm2 = np.asarray(stereo["CM2"], dtype=np.float64)
    rotation = np.asarray(stereo["R"], dtype=np.float64)
    translation = np.asarray(stereo["T"], dtype=np.float64).reshape(3)

    fx1, fy1, cx1, cy1 = stereo_cm1[0, 0], stereo_cm1[1, 1], stereo_cm1[0, 2], stereo_cm1[1, 2]
    fx2, fy2, cx2, cy2 = stereo_cm2[0, 0], stereo_cm2[1, 1], stereo_cm2[0, 2], stereo_cm2[1, 2]

    valid_depth = np.isfinite(depth) & (depth > 0)
    if not np.any(valid_depth):
        base["debug"]["depth_projection"] = {"used": False, "reason": "empty_depth"}
        return base

    yy, xx = np.indices(depth.shape)
    z = depth[valid_depth].astype(np.float32)
    x = (xx[valid_depth] - cx1) * z / fx1
    y = (yy[valid_depth] - cy1) * z / fy1
    xyz = np.stack([x, y, z], axis=1)
    xyz2 = (rotation @ xyz.T).T + translation[None, :]
    z2 = xyz2[:, 2]
    keep = z2 > 1e-6
    if not np.any(keep):
        base["debug"]["depth_projection"] = {"used": False, "reason": "all_points_behind_camera"}
        return base

    xyz2 = xyz2[keep]
    z2 = z2[keep]
    src_x = xx[valid_depth][keep]
    src_y = yy[valid_depth][keep]

    u2 = np.round(fx2 * xyz2[:, 0] / z2 + cx2).astype(np.int32)
    v2 = np.round(fy2 * xyz2[:, 1] / z2 + cy2).astype(np.int32)
    in_bounds = (u2 >= 0) & (u2 < target_w) & (v2 >= 0) & (v2 < target_h)
    if not np.any(in_bounds):
        base["debug"]["depth_projection"] = {"used": False, "reason": "no_points_in_bounds"}
        return base

    u2 = u2[in_bounds]
    v2 = v2[in_bounds]
    z2 = z2[in_bounds]
    src_x = src_x[in_bounds]
    src_y = src_y[in_bounds]

    order = np.argsort(z2)
    z_buffer = np.full((target_h, target_w), np.inf, dtype=np.float32)
    projected_hits = 0
    radius = int(max(0, splat_radius))
    for idx in order:
        x_t = int(u2[idx])
        y_t = int(v2[idx])
        src_px = assets["source_image"][src_y[idx], src_x[idx]]
        src_label = int(assets["source_mask"][src_y[idx], src_x[idx]])
        if src_label <= 0:
            continue
        for oy in range(-radius, radius + 1):
            yy_t = y_t + oy
            if yy_t < 0 or yy_t >= target_h:
                continue
            for ox in range(-radius, radius + 1):
                xx_t = x_t + ox
                if xx_t < 0 or xx_t >= target_w:
                    continue
                depth_penalty = 0.2 * float(abs(ox) + abs(oy))
                score_depth = float(z2[idx]) + depth_penalty
                if score_depth <= float(z_buffer[yy_t, xx_t]):
                    z_buffer[yy_t, xx_t] = score_depth
                    pred_mask[yy_t, xx_t] = src_label
                    warped_source[yy_t, xx_t] = src_px
                    valid_mask[yy_t, xx_t] = 1
                    real_projected_mask[yy_t, xx_t] = 1
                    projected_hits += 1

    pre_fill_pred_mask = pred_mask.copy()
    pre_fill_warped_source = warped_source.copy()
    pre_fill_valid_mask = valid_mask.copy()
    support_mask = cv2.dilate(
        real_projected_mask,
        np.ones((int(max(1, support_dilate_ksize)), int(max(1, support_dilate_ksize))), np.uint8),
        iterations=1,
    )
    fill_info = {"used": False, "filled_pixels": 0}
    if fill_holes:
        pred_mask, warped_source, valid_mask, fill_info = _nearest_seed_fill(
            pred_mask,
            warped_source,
            valid_mask,
            support_mask > 0,
            float(fill_distance_px),
        )
    hole_fill_mask = ((valid_mask > 0) & (real_projected_mask == 0)).astype(np.uint8)
    projected_ratio = float(real_projected_mask.sum()) / max(float(support_mask.sum()), 1.0)

    return {
        "pred_mask": pred_mask.astype(np.uint8),
        "warped_source": warped_source,
        "valid_mask": valid_mask.astype(np.uint8),
        "real_projected_mask": real_projected_mask.astype(np.uint8),
        "support_mask": support_mask.astype(np.uint8),
        "hole_fill_mask": hole_fill_mask.astype(np.uint8),
        "pre_fill_pred_mask": pre_fill_pred_mask.astype(np.uint8),
        "pre_fill_warped_source": pre_fill_warped_source,
        "pre_fill_valid_mask": pre_fill_valid_mask.astype(np.uint8),
        "debug": {
            "alignment": "depth_projection",
            "seed_with_homography": bool(seed_with_homography),
            "splat_radius": radius,
            "projected_hits": int(projected_hits),
            "real_projected_pixels": int(real_projected_mask.sum()),
            "support_pixels": int(support_mask.sum()),
            "projected_ratio": projected_ratio,
            "fill": fill_info,
        },
    }


def apply_scene_tuning(
    result: dict,
    target_image: np.ndarray,
    modality: str,
    *,
    coarse_radius_px: int,
    coarse_step_px: int,
    fine_radius_px: int,
    fine_step_px: int,
    coarse_scales: list[float],
    coarse_angles_deg: list[float] | None = None,
    fine_scale_delta: float = 0.02,
    fine_angle_delta_deg: float = 0.0,
    edge_weight: float = 1.0,
    mi_weight: float = 0.20,
    coverage_weight: float = 0.50,
) -> dict:
    tuned_mask, tuned_image, tuned_valid, tune_info = auto_scene_tune(
        result["pred_mask"],
        result["warped_source"],
        result["valid_mask"],
        target_image,
        modality,
        coarse_radius_px=int(coarse_radius_px),
        coarse_step_px=int(coarse_step_px),
        fine_radius_px=int(fine_radius_px),
        fine_step_px=int(fine_step_px),
        coarse_scales=list(coarse_scales),
        coarse_angles_deg=list(coarse_angles_deg or [0.0]),
        fine_scale_delta=float(fine_scale_delta),
        fine_angle_delta_deg=float(fine_angle_delta_deg),
        edge_weight=float(edge_weight),
        mi_weight=float(mi_weight),
        coverage_weight=float(coverage_weight),
    )
    debug = dict(result.get("debug", {}))
    debug["scene_tune"] = tune_info
    return {
        "pred_mask": tuned_mask.astype(np.uint8),
        "warped_source": tuned_image,
        "valid_mask": tuned_valid.astype(np.uint8),
        "debug": debug,
    }
