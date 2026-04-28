from __future__ import annotations

import cv2
import numpy as np

from ..common import mutual_information, normalized_total_gradient


def _mask_boundary(mask: np.ndarray) -> np.ndarray:
    fg = (mask > 0).astype(np.uint8)
    kernel = np.ones((3, 3), dtype=np.uint8)
    return ((cv2.dilate(fg, kernel) > 0) ^ (cv2.erode(fg, kernel) > 0)).astype(np.uint8)


def keypoint_transfer_error_px(pred_mask: np.ndarray, gt_mask: np.ndarray, valid_mask: np.ndarray) -> float:
    pred_b = _mask_boundary(pred_mask) & valid_mask.astype(np.uint8)
    gt_b = _mask_boundary(gt_mask) & valid_mask.astype(np.uint8)
    if int(pred_b.sum()) == 0 or int(gt_b.sum()) == 0:
        return float("nan")
    distance = cv2.distanceTransform((1 - gt_b).astype(np.uint8), cv2.DIST_L2, 3)
    return float(distance[pred_b > 0].mean())


def overall_region_error_px(pred_mask: np.ndarray, gt_mask: np.ndarray, valid_mask: np.ndarray) -> float:
    pred_fg = ((pred_mask > 0) & valid_mask.astype(bool)).astype(np.uint8)
    gt_fg = ((gt_mask > 0) & valid_mask.astype(bool)).astype(np.uint8)
    if int(pred_fg.sum()) == 0 or int(gt_fg.sum()) == 0:
        return float("nan")

    dist_to_gt = cv2.distanceTransform((1 - gt_fg).astype(np.uint8), cv2.DIST_L2, 3)
    dist_to_pred = cv2.distanceTransform((1 - pred_fg).astype(np.uint8), cv2.DIST_L2, 3)
    pred_term = float(dist_to_gt[pred_fg > 0].mean())
    gt_term = float(dist_to_pred[gt_fg > 0].mean())
    return 0.5 * (pred_term + gt_term)


def normalized_overall_region_error(pred_mask: np.ndarray, gt_mask: np.ndarray, valid_mask: np.ndarray) -> float:
    error_px = overall_region_error_px(pred_mask, gt_mask, valid_mask)
    if not np.isfinite(float(error_px)):
        return float("nan")
    height, width = gt_mask.shape[:2]
    diagonal = float(np.hypot(width, height))
    return float(error_px) / max(diagonal, 1.0)


def summarize_alignment(
    warped_source: np.ndarray,
    target_image: np.ndarray,
    pred_mask: np.ndarray,
    gt_mask: np.ndarray,
    valid_mask: np.ndarray,
) -> dict[str, float]:
    coverage = float(valid_mask.mean())
    return {
        "mutual_information": mutual_information(target_image, warped_source, valid_mask),
        "ntg": normalized_total_gradient(target_image, warped_source, valid_mask),
        "valid_warp_coverage": coverage,
        "keypoint_transfer_error_px": keypoint_transfer_error_px(pred_mask, gt_mask, valid_mask),
        "overall_region_error_px": overall_region_error_px(pred_mask, gt_mask, valid_mask),
        "normalized_overall_region_error": normalized_overall_region_error(pred_mask, gt_mask, valid_mask),
    }
