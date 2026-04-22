from __future__ import annotations

import csv
from pathlib import Path

import cv2
import numpy as np


def compute_confusion(pred_mask: np.ndarray, gt_mask: np.ndarray, valid_mask: np.ndarray, num_classes: int) -> np.ndarray:
    valid = valid_mask.astype(bool)
    pred = pred_mask.astype(np.int64)[valid]
    gt = gt_mask.astype(np.int64)[valid]
    pred = np.clip(pred, 0, num_classes - 1)
    gt = np.clip(gt, 0, num_classes - 1)
    encoded = gt * num_classes + pred
    counts = np.bincount(encoded, minlength=num_classes * num_classes)
    return counts.reshape(num_classes, num_classes)


def summarize_confusion(confusion: np.ndarray, class_names: list[str]) -> tuple[dict[str, float], list[dict[str, float | int | str]]]:
    total = float(confusion.sum())
    num_classes = confusion.shape[0]
    rows: list[dict[str, float | int | str]] = []
    ious = []
    recalls = []
    weighted_iou = 0.0

    for cls in range(num_classes):
        tp = float(confusion[cls, cls])
        fp = float(confusion[:, cls].sum() - tp)
        fn = float(confusion[cls, :].sum() - tp)
        tn = total - tp - fp - fn
        precision = tp / max(tp + fp, 1.0)
        recall = tp / max(tp + fn, 1.0)
        iou = tp / max(tp + fp + fn, 1.0)
        f1 = 2.0 * precision * recall / max(precision + recall, 1e-9)
        class_pixels = int(confusion[cls, :].sum())
        rows.append(
            {
                "class_name": class_names[cls] if cls < len(class_names) else f"class_{cls}",
                "iou": iou,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "pixel_accuracy": recall,
                "class_pixels": class_pixels,
            }
        )
        if class_pixels > 0:
            ious.append(iou)
            recalls.append(recall)
            weighted_iou += (class_pixels / max(total, 1.0)) * iou

    pixel_accuracy = float(np.trace(confusion)) / max(total, 1.0)
    summary = {
        "mean_iou": float(np.mean(ious)) if ious else 0.0,
        "pixel_accuracy": pixel_accuracy,
        "mean_pixel_accuracy": float(np.mean(recalls)) if recalls else 0.0,
        "freq_iou": float(weighted_iou),
    }
    return summary, rows


def boundary_f1(pred_mask: np.ndarray, gt_mask: np.ndarray, valid_mask: np.ndarray, radius_px: int = 2) -> float:
    kernel = np.ones((3, 3), dtype=np.uint8)
    pred = ((cv2.dilate((pred_mask > 0).astype(np.uint8), kernel) > 0) ^ (cv2.erode((pred_mask > 0).astype(np.uint8), kernel) > 0)).astype(np.uint8)
    gt = ((cv2.dilate((gt_mask > 0).astype(np.uint8), kernel) > 0) ^ (cv2.erode((gt_mask > 0).astype(np.uint8), kernel) > 0)).astype(np.uint8)
    valid = valid_mask.astype(bool)
    pred = pred & valid
    gt = gt & valid

    pred_sum = int(pred.sum())
    gt_sum = int(gt.sum())
    if pred_sum == 0 and gt_sum == 0:
        return 1.0
    if pred_sum == 0 or gt_sum == 0:
        return 0.0

    dilate_kernel = np.ones((radius_px * 2 + 1, radius_px * 2 + 1), dtype=np.uint8)
    pred_match = pred & (cv2.dilate(gt.astype(np.uint8), dilate_kernel) > 0)
    gt_match = gt & (cv2.dilate(pred.astype(np.uint8), dilate_kernel) > 0)
    precision = float(pred_match.sum()) / max(pred_sum, 1.0)
    recall = float(gt_match.sum()) / max(gt_sum, 1.0)
    return 2.0 * precision * recall / max(precision + recall, 1e-9)


def write_per_class_csv(path: str | Path, rows: list[dict[str, float | int | str]]) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        out_path.write_text("", encoding="utf-8")
        return
    with out_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
