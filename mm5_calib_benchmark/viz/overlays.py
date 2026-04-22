from __future__ import annotations

import cv2
import numpy as np

from ..common import ensure_bgr, to_gray_u8


def contour_overlay(base_image: np.ndarray, mask: np.ndarray, color: tuple[int, int, int]) -> np.ndarray:
    out = ensure_bgr(base_image)
    contours, _ = cv2.findContours((mask > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, contours, -1, color, 2)
    return out


def error_heatmap(pred_mask: np.ndarray, gt_mask: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    out = np.full((*pred_mask.shape, 3), 24, dtype=np.uint8)
    valid = valid_mask.astype(bool)
    tp = valid & (pred_mask == gt_mask)
    fp = valid & (pred_mask > 0) & (gt_mask == 0)
    fn = valid & (pred_mask == 0) & (gt_mask > 0)
    wrong_cls = valid & (pred_mask > 0) & (gt_mask > 0) & (pred_mask != gt_mask)
    out[tp] = (60, 130, 60)
    out[fp] = (40, 40, 220)
    out[fn] = (220, 90, 40)
    out[wrong_cls] = (255, 0, 255)
    out[~valid] = (0, 0, 0)
    return out


def mar_like_overlay(target_image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    gray = to_gray_u8(target_image)
    base = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    output = base.copy()
    num_components, labels = cv2.connectedComponents((mask > 0).astype(np.uint8))
    palette = [
        (255, 0, 255),
        (255, 255, 0),
        (0, 255, 255),
        (0, 165, 255),
        (0, 255, 0),
        (255, 80, 80),
    ]
    for comp_id in range(1, num_components):
        output[labels == comp_id] = palette[(comp_id - 1) % len(palette)]
    contours, _ = cv2.findContours((mask > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(output, contours, -1, (0, 0, 0), 2)
    return output
