from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

METHOD_DIR = Path(__file__).resolve().parent
DARKLIGHT_DIR = METHOD_DIR.parent
REPO_ROOT = DARKLIGHT_DIR.parent
if str(DARKLIGHT_DIR) not in sys.path:
    sys.path.insert(0, str(DARKLIGHT_DIR))

from run_darklight import (  # noqa: E402
    StereoCalibration,
    adjusted_stereo_calibration,
    collect_fieldnames,
    direct_alignment_metrics,
    direct_metrics_for_csv,
    gray_u8,
    imread_unicode,
    imwrite_unicode,
    load_rows,
    load_stereo_calibration,
    make_edge_overlay,
    make_five_panel,
    matrix_to_list,
    metrics_for_csv,
    normalize_u8,
    parse_size_arg,
    select_dark_rows,
    warp_image,
    write_csv,
)


@dataclass
class RectificationMatrices:
    R1: np.ndarray | None
    R2: np.ndarray | None
    P1: np.ndarray | None
    P2: np.ndarray | None


@dataclass
class CameraModel:
    label: str
    K: np.ndarray
    D: np.ndarray


@dataclass
class CandidateOutput:
    name: str
    rgb: np.ndarray
    lwir: np.ndarray
    rgb_valid: np.ndarray
    lwir_valid: np.ndarray
    metadata: dict


def sample_id(row) -> str:
    return f"{row.aligned_id:03d}_seq{row.sequence}"


def parse_size_list(text: str) -> list[tuple[int, int] | None]:
    out = []
    for part in str(text).split(","):
        part = part.strip()
        if part:
            out.append(parse_size_arg(part))
    return out


def parse_float_list(text: str) -> list[float]:
    return [float(x.strip()) for x in str(text).split(",") if x.strip()]


def read_rectification_matrices(path: str | Path) -> RectificationMatrices:
    fs = cv2.FileStorage(str(path), cv2.FILE_STORAGE_READ)
    if not fs.isOpened():
        raise RuntimeError(f"failed to open calibration file: {path}")

    def mat(name: str) -> np.ndarray | None:
        node = fs.getNode(name)
        if node.empty():
            return None
        value = node.mat()
        if value is None:
            return None
        return value.astype(np.float64)

    out = RectificationMatrices(R1=mat("R1"), R2=mat("R2"), P1=mat("P1"), P2=mat("P2"))
    fs.release()
    return out


def read_yaml_array(fs: cv2.FileStorage, key: str) -> np.ndarray:
    node = fs.getNode(key)
    if node.empty():
        raise KeyError(f"missing calibration key: {key}")
    if node.isSeq():
        return np.array([node.at(i).real() for i in range(node.size())], dtype=np.float64)
    mat = node.mat()
    if mat is not None:
        return mat.astype(np.float64)
    raise ValueError(f"unsupported calibration key: {key}")


def read_single_camera_model(path: str | Path, label: str) -> CameraModel | None:
    p = Path(path)
    if not p.exists():
        return None
    fs = cv2.FileStorage(str(p), cv2.FILE_STORAGE_READ)
    if not fs.isOpened():
        raise RuntimeError(f"failed to open camera calibration file: {path}")
    K = read_yaml_array(fs, "CM").reshape(3, 3)
    D = read_yaml_array(fs, "D").reshape(-1)
    fs.release()
    return CameraModel(label=label, K=K, D=D)


def plane_rgb_to_lwir_homography(calibration: StereoCalibration, plane_depth_mm: float) -> np.ndarray:
    normal = np.array([[0.0], [0.0], [1.0]], dtype=np.float64)
    t = calibration.rgb_to_lwir_T.reshape(3, 1)
    H = calibration.lwir_K @ (
        calibration.rgb_to_lwir_R - (t @ normal.T) / float(max(plane_depth_mm, 1.0))
    ) @ np.linalg.inv(calibration.rgb_K)
    return H / H[2, 2]


def crop_offset_center(image_shape, target_size: tuple[int, int]) -> tuple[int, int]:
    target_w, target_h = target_size
    h, w = image_shape[:2]
    return int(round((w - target_w) / 2.0)), int(round((h - target_h) / 2.0))


def crop_offset_optical(K: np.ndarray, target_size: tuple[int, int]) -> tuple[int, int]:
    target_w, target_h = target_size
    return int(round(float(K[0, 2]) - target_w / 2.0)), int(round(float(K[1, 2]) - target_h / 2.0))


def crop_offset_rectified_principal(P: np.ndarray, target_size: tuple[int, int]) -> tuple[int, int]:
    target_w, target_h = target_size
    return int(round(float(P[0, 2]) - target_w / 2.0)), int(round(float(P[1, 2]) - target_h / 2.0))


def calibration_crop_modes(
    calibration: StereoCalibration,
    image_shape,
    target_size: tuple[int, int],
    include_optimal_new_camera: bool = True,
) -> dict[str, tuple[int, int]]:
    modes = {
        "center": crop_offset_center(image_shape, target_size),
        "rgb_optical": crop_offset_optical(calibration.rgb_K, target_size),
    }
    if include_optimal_new_camera:
        raw_size = (image_shape[1], image_shape[0])
        for alpha in (0.0, 0.5, 1.0):
            new_K, _ = cv2.getOptimalNewCameraMatrix(
                calibration.rgb_K,
                calibration.rgb_D.reshape(-1, 1),
                raw_size,
                alpha,
                raw_size,
                centerPrincipalPoint=False,
            )
            modes[f"rgb_optimal_alpha{str(alpha).replace('.', '_')}"] = crop_offset_optical(new_K, target_size)
    return modes


def crop_with_border(img: np.ndarray, offset: tuple[int, int], target_size: tuple[int, int], border=0) -> np.ndarray:
    x0, y0 = offset
    target_w, target_h = target_size
    if img.ndim == 2:
        out = np.full((target_h, target_w), border, dtype=img.dtype)
    else:
        out = np.full((target_h, target_w, img.shape[2]), border, dtype=img.dtype)

    src_x0 = max(0, x0)
    src_y0 = max(0, y0)
    src_x1 = min(img.shape[1], x0 + target_w)
    src_y1 = min(img.shape[0], y0 + target_h)
    if src_x1 <= src_x0 or src_y1 <= src_y0:
        return out

    dst_x0 = src_x0 - x0
    dst_y0 = src_y0 - y0
    out[dst_y0 : dst_y0 + (src_y1 - src_y0), dst_x0 : dst_x0 + (src_x1 - src_x0)] = img[src_y0:src_y1, src_x0:src_x1]
    return out


def crop_mask_with_border(mask: np.ndarray, offset: tuple[int, int], target_size: tuple[int, int]) -> np.ndarray:
    return crop_with_border(mask.astype(np.uint8) * 255, offset, target_size, border=0) > 0


def rectified_remap(
    img: np.ndarray,
    K: np.ndarray,
    D: np.ndarray,
    R: np.ndarray,
    P: np.ndarray,
    offset: tuple[int, int],
    target_size: tuple[int, int],
    interpolation: int,
) -> tuple[np.ndarray, np.ndarray]:
    P_crop = P.copy()
    P_crop[0, 2] -= float(offset[0])
    P_crop[1, 2] -= float(offset[1])
    new_K = P_crop[:, :3].astype(np.float64)
    map_x, map_y = cv2.initUndistortRectifyMap(
        K.astype(np.float64),
        D.reshape(-1, 1).astype(np.float64),
        R.astype(np.float64),
        new_K,
        target_size,
        cv2.CV_32FC1,
    )
    remapped = cv2.remap(img, map_x, map_y, interpolation, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    valid_src = np.ones(img.shape[:2], dtype=np.uint8) * 255
    valid = cv2.remap(valid_src, map_x, map_y, cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0) > 0
    return remapped, valid


def make_plane_candidates(
    raw_rgb: np.ndarray,
    raw_lwir_u8: np.ndarray,
    calibration: StereoCalibration,
    target_size: tuple[int, int],
    plane_depths: list[float],
    lwir_calib_sizes: list[tuple[int, int] | None],
    extra_lwir_models: list[CameraModel],
) -> list[CandidateOutput]:
    candidates = []
    rgb_size = (raw_rgb.shape[1], raw_rgb.shape[0])
    crop_modes = calibration_crop_modes(calibration, raw_rgb.shape, target_size)

    lwir_models: list[tuple[str, StereoCalibration, dict]] = []
    for lwir_calib_size in lwir_calib_sizes:
        effective = adjusted_stereo_calibration(
            calibration,
            raw_rgb.shape,
            raw_lwir_u8.shape,
            rgb_calib_size=None,
            lwir_calib_size=lwir_calib_size,
            lwir_principal_offset=(0.0, 0.0),
            t_scale=1.0,
        )
        size_label = "raw" if lwir_calib_size is None else f"{lwir_calib_size[0]}x{lwir_calib_size[1]}"
        lwir_models.append(
            (
                size_label,
                effective,
                {"lwir_calib_size": None if lwir_calib_size is None else list(lwir_calib_size)},
            )
        )
    for model in extra_lwir_models:
        lwir_models.append(
            (
                model.label,
                StereoCalibration(
                    path=calibration.path,
                    rgb_K=calibration.rgb_K.copy(),
                    rgb_D=calibration.rgb_D.copy(),
                    lwir_K=model.K.copy(),
                    lwir_D=model.D.copy(),
                    rgb_to_lwir_R=calibration.rgb_to_lwir_R.copy(),
                    rgb_to_lwir_T=calibration.rgb_to_lwir_T.copy(),
                ),
                {"lwir_camera_model": model.label},
            )
        )

    for size_label, effective, lwir_metadata in lwir_models:
        for depth in plane_depths:
            H_rgb_to_lwir = plane_rgb_to_lwir_homography(effective, depth)
            H_lwir_to_rgb = np.linalg.inv(H_rgb_to_lwir)
            H_lwir_to_rgb = H_lwir_to_rgb / H_lwir_to_rgb[2, 2]
            lwir_on_rgb = warp_image(raw_lwir_u8, H_lwir_to_rgb, rgb_size)
            lwir_valid_on_rgb = warp_image(
                np.ones(raw_lwir_u8.shape[:2], dtype=np.uint8) * 255,
                H_lwir_to_rgb,
                rgb_size,
                interp=cv2.INTER_NEAREST,
            ) > 0
            rgb_valid_full = np.ones(raw_rgb.shape[:2], dtype=bool)
            for crop_name, offset in crop_modes.items():
                name = f"plane_{size_label}_{int(round(depth))}mm_{crop_name}"
                candidates.append(
                    CandidateOutput(
                        name=name,
                        rgb=crop_with_border(raw_rgb, offset, target_size),
                        lwir=crop_with_border(lwir_on_rgb, offset, target_size),
                        rgb_valid=crop_mask_with_border(rgb_valid_full, offset, target_size),
                        lwir_valid=crop_mask_with_border(lwir_valid_on_rgb, offset, target_size),
                        metadata={
                            "method": "calibration_plane_crop",
                            **lwir_metadata,
                            "plane_depth_mm": float(depth),
                            "crop_mode": crop_name,
                            "crop_offset_xy": [int(offset[0]), int(offset[1])],
                            "H_rgb_to_lwir": matrix_to_list(H_rgb_to_lwir),
                            "H_lwir_to_rgb": matrix_to_list(H_lwir_to_rgb),
                        },
                    )
                )
    return candidates


def make_stored_rectification_candidates(
    raw_rgb: np.ndarray,
    raw_lwir_u8: np.ndarray,
    calibration: StereoCalibration,
    rectification: RectificationMatrices,
    target_size: tuple[int, int],
    lwir_calib_sizes: list[tuple[int, int] | None],
    extra_lwir_models: list[CameraModel],
) -> list[CandidateOutput]:
    if rectification.R1 is None or rectification.R2 is None or rectification.P1 is None or rectification.P2 is None:
        return []

    candidates = []
    crop_modes = calibration_crop_modes(calibration, raw_rgb.shape, target_size)
    crop_modes["p1_principal"] = crop_offset_rectified_principal(rectification.P1, target_size)
    lwir_models: list[tuple[str, StereoCalibration, dict]] = []
    for lwir_calib_size in lwir_calib_sizes:
        effective = adjusted_stereo_calibration(
            calibration,
            raw_rgb.shape,
            raw_lwir_u8.shape,
            rgb_calib_size=None,
            lwir_calib_size=lwir_calib_size,
            lwir_principal_offset=(0.0, 0.0),
            t_scale=1.0,
        )
        size_label = "raw" if lwir_calib_size is None else f"{lwir_calib_size[0]}x{lwir_calib_size[1]}"
        lwir_models.append(
            (
                size_label,
                effective,
                {"lwir_calib_size": None if lwir_calib_size is None else list(lwir_calib_size)},
            )
        )
    for model in extra_lwir_models:
        lwir_models.append(
            (
                model.label,
                StereoCalibration(
                    path=calibration.path,
                    rgb_K=calibration.rgb_K.copy(),
                    rgb_D=calibration.rgb_D.copy(),
                    lwir_K=model.K.copy(),
                    lwir_D=model.D.copy(),
                    rgb_to_lwir_R=calibration.rgb_to_lwir_R.copy(),
                    rgb_to_lwir_T=calibration.rgb_to_lwir_T.copy(),
                ),
                {"lwir_camera_model": model.label},
            )
        )

    for size_label, effective, lwir_metadata in lwir_models:
        for crop_name, offset in crop_modes.items():
            rgb_rect, rgb_valid = rectified_remap(
                raw_rgb,
                effective.rgb_K,
                effective.rgb_D,
                rectification.R1,
                rectification.P1,
                offset,
                target_size,
                cv2.INTER_LINEAR,
            )
            lwir_rect, lwir_valid = rectified_remap(
                raw_lwir_u8,
                effective.lwir_K,
                effective.lwir_D,
                rectification.R2,
                rectification.P2,
                offset,
                target_size,
                cv2.INTER_LINEAR,
            )
            name = f"yaml_rectify_{size_label}_{crop_name}"
            candidates.append(
                CandidateOutput(
                    name=name,
                    rgb=rgb_rect,
                    lwir=lwir_rect,
                    rgb_valid=rgb_valid,
                    lwir_valid=lwir_valid,
                    metadata={
                        "method": "stored_yaml_rectification",
                        **lwir_metadata,
                        "crop_mode": crop_name,
                        "crop_offset_xy": [int(offset[0]), int(offset[1])],
                    },
                )
            )
    return candidates


def make_hybrid_lwir_rectification_candidates(
    raw_rgb: np.ndarray,
    raw_lwir_u8: np.ndarray,
    calibration: StereoCalibration,
    rectification: RectificationMatrices,
    target_size: tuple[int, int],
    lwir_calib_sizes: list[tuple[int, int] | None],
    extra_lwir_models: list[CameraModel],
) -> list[CandidateOutput]:
    if rectification.R2 is None or rectification.P2 is None:
        return []

    crop_modes = calibration_crop_modes(calibration, raw_rgb.shape, target_size)
    lwir_models: list[tuple[str, StereoCalibration, dict]] = []
    for lwir_calib_size in lwir_calib_sizes:
        effective = adjusted_stereo_calibration(
            calibration,
            raw_rgb.shape,
            raw_lwir_u8.shape,
            rgb_calib_size=None,
            lwir_calib_size=lwir_calib_size,
            lwir_principal_offset=(0.0, 0.0),
            t_scale=1.0,
        )
        size_label = "raw" if lwir_calib_size is None else f"{lwir_calib_size[0]}x{lwir_calib_size[1]}"
        lwir_models.append(
            (
                size_label,
                effective,
                {"lwir_calib_size": None if lwir_calib_size is None else list(lwir_calib_size)},
            )
        )
    for model in extra_lwir_models:
        lwir_models.append(
            (
                model.label,
                StereoCalibration(
                    path=calibration.path,
                    rgb_K=calibration.rgb_K.copy(),
                    rgb_D=calibration.rgb_D.copy(),
                    lwir_K=model.K.copy(),
                    lwir_D=model.D.copy(),
                    rgb_to_lwir_R=calibration.rgb_to_lwir_R.copy(),
                    rgb_to_lwir_T=calibration.rgb_to_lwir_T.copy(),
                ),
                {"lwir_camera_model": model.label},
            )
        )

    candidates = []
    for size_label, effective, lwir_metadata in lwir_models:
        for crop_name, offset in crop_modes.items():
            lwir_rect, lwir_valid = rectified_remap(
                raw_lwir_u8,
                effective.lwir_K,
                effective.lwir_D,
                rectification.R2,
                rectification.P2,
                offset,
                target_size,
                cv2.INTER_LINEAR,
            )
            rgb_valid_full = np.ones(raw_rgb.shape[:2], dtype=bool)
            name = f"hybrid_raw_rgb_lwir_rectify_{size_label}_{crop_name}"
            candidates.append(
                CandidateOutput(
                    name=name,
                    rgb=crop_with_border(raw_rgb, offset, target_size),
                    lwir=lwir_rect,
                    rgb_valid=crop_mask_with_border(rgb_valid_full, offset, target_size),
                    lwir_valid=lwir_valid,
                    metadata={
                        "method": "hybrid_raw_rgb_crop_lwir_yaml_rectification",
                        **lwir_metadata,
                        "crop_mode": crop_name,
                        "crop_offset_xy": [int(offset[0]), int(offset[1])],
                    },
                )
            )
    return candidates


def make_raw_crop_candidates(
    raw_rgb: np.ndarray,
    raw_lwir_u8: np.ndarray,
    calibration: StereoCalibration,
    target_size: tuple[int, int],
) -> list[CandidateOutput]:
    candidates = []
    modes = calibration_crop_modes(calibration, raw_rgb.shape, target_size)
    for crop_name, offset in modes.items():
        # This is a geometry sanity baseline. It does not claim RGB/LWIR calibration.
        lwir_offset = crop_offset_center(raw_lwir_u8.shape, target_size)
        candidates.append(
            CandidateOutput(
                name=f"raw_crop_{crop_name}",
                rgb=crop_with_border(raw_rgb, offset, target_size),
                lwir=crop_with_border(raw_lwir_u8, lwir_offset, target_size),
                rgb_valid=crop_mask_with_border(np.ones(raw_rgb.shape[:2], dtype=bool), offset, target_size),
                lwir_valid=crop_mask_with_border(np.ones(raw_lwir_u8.shape[:2], dtype=bool), lwir_offset, target_size),
                metadata={
                    "method": "raw_crop_sanity_baseline",
                    "rgb_crop_mode": crop_name,
                    "rgb_crop_offset_xy": [int(offset[0]), int(offset[1])],
                    "lwir_crop_offset_xy": [int(lwir_offset[0]), int(lwir_offset[1])],
                },
            )
        )
    return candidates


def build_candidates(
    raw_rgb: np.ndarray,
    raw_lwir_u8: np.ndarray,
    calibration: StereoCalibration,
    rectification: RectificationMatrices,
    target_size: tuple[int, int],
    plane_depths: list[float],
    lwir_calib_sizes: list[tuple[int, int] | None],
    extra_lwir_models: list[CameraModel],
) -> list[CandidateOutput]:
    candidates = []
    candidates.extend(make_raw_crop_candidates(raw_rgb, raw_lwir_u8, calibration, target_size))
    candidates.extend(
        make_plane_candidates(raw_rgb, raw_lwir_u8, calibration, target_size, plane_depths, lwir_calib_sizes, extra_lwir_models)
    )
    candidates.extend(
        make_stored_rectification_candidates(
            raw_rgb,
            raw_lwir_u8,
            calibration,
            rectification,
            target_size,
            lwir_calib_sizes,
            extra_lwir_models,
        )
    )
    candidates.extend(
        make_hybrid_lwir_rectification_candidates(
            raw_rgb,
            raw_lwir_u8,
            calibration,
            rectification,
            target_size,
            lwir_calib_sizes,
            extra_lwir_models,
        )
    )
    return candidates


def metric_prefix(prefix: str, metrics: dict) -> dict:
    return {f"{prefix}_{k}": v for k, v in direct_metrics_for_csv(metrics).items()}


def evaluate_candidate(row, candidate: CandidateOutput, aligned_rgb: np.ndarray, aligned_lwir_u8: np.ndarray) -> dict:
    ref_rgb_gray = gray_u8(aligned_rgb)
    cand_rgb_gray = gray_u8(candidate.rgb)
    rgb_valid = candidate.rgb_valid.astype(bool)
    lwir_valid = candidate.lwir_valid.astype(bool)
    both_valid = rgb_valid & lwir_valid

    rgb_metrics = direct_alignment_metrics(ref_rgb_gray, cand_rgb_gray, rgb_valid)
    lwir_metrics = direct_alignment_metrics(aligned_lwir_u8, candidate.lwir, lwir_valid)
    cross_metrics = direct_alignment_metrics(cand_rgb_gray, candidate.lwir, both_valid)

    row_out = {
        "sample": sample_id(row),
        "aligned_id": row.aligned_id,
        "sequence": row.sequence,
        "candidate": candidate.name,
        "method": candidate.metadata.get("method", ""),
        "rgb_valid_ratio": float(rgb_valid.mean()),
        "lwir_valid_ratio": float(lwir_valid.mean()),
        "intersection_valid_ratio": float(both_valid.mean()),
    }
    row_out.update(metric_prefix("eval_rgb_to_mm5_aligned_rgb", rgb_metrics))
    row_out.update(metric_prefix("eval_lwir_to_mm5_aligned_t16", lwir_metrics))
    row_out.update(metric_prefix("eval_cross_rgb_lwir", cross_metrics))
    return row_out


def save_candidate_outputs(output_dir: Path, row, candidate: CandidateOutput, aligned_rgb: np.ndarray, aligned_lwir_u8: np.ndarray) -> None:
    sid = sample_id(row)
    cand_dir = output_dir / "samples" / sid / candidate.name
    imwrite_unicode(cand_dir / "rgb_calibration_only.png", candidate.rgb)
    imwrite_unicode(cand_dir / "lwir_calibration_only.png", candidate.lwir)
    imwrite_unicode(cand_dir / "rgb_valid_mask.png", candidate.rgb_valid.astype(np.uint8) * 255)
    imwrite_unicode(cand_dir / "lwir_valid_mask.png", candidate.lwir_valid.astype(np.uint8) * 255)
    imwrite_unicode(cand_dir / "edge_overlay_generated.png", make_edge_overlay(candidate.rgb, candidate.lwir, candidate.rgb_valid & candidate.lwir_valid))
    (cand_dir / "metadata.json").write_text(json.dumps(candidate.metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    panel_dir = output_dir / "panels"
    aligned_lwir_bgr = cv2.cvtColor(aligned_lwir_u8, cv2.COLOR_GRAY2BGR)
    make_five_panel(
        [
            (candidate.rgb, "Calibration-only RGB"),
            (candidate.lwir, "Calibration-only LWIR"),
            (aligned_rgb, "MM5 aligned RGB eval"),
            (aligned_lwir_bgr, "MM5 aligned T16 eval"),
            (make_edge_overlay(candidate.rgb, candidate.lwir, candidate.rgb_valid & candidate.lwir_valid), "Generated edge check"),
        ],
        panel_dir / f"{sid}_{candidate.name}_panel.png",
        tile_size=(360, 270),
    )


def summarize_candidates(rows: list[dict]) -> list[dict]:
    grouped: dict[str, list[dict]] = {}
    for row in rows:
        grouped.setdefault(str(row["candidate"]), []).append(row)
    summary = []
    numeric_keys = [
        "rgb_valid_ratio",
        "lwir_valid_ratio",
        "intersection_valid_ratio",
        "eval_rgb_to_mm5_aligned_rgb_ncc",
        "eval_rgb_to_mm5_aligned_rgb_edge_distance",
        "eval_lwir_to_mm5_aligned_t16_ncc",
        "eval_lwir_to_mm5_aligned_t16_edge_distance",
        "eval_lwir_to_mm5_aligned_t16_mi",
        "eval_cross_rgb_lwir_ncc",
        "eval_cross_rgb_lwir_edge_distance",
        "eval_cross_rgb_lwir_mi",
    ]
    for name, items in grouped.items():
        out = {"candidate": name, "samples": len(items), "method": items[0].get("method", "")}
        for key in numeric_keys:
            vals = [float(item[key]) for item in items if key in item and np.isfinite(float(item[key]))]
            out[f"{key}_mean"] = float(np.mean(vals)) if vals else float("nan")
            out[f"{key}_min"] = float(np.min(vals)) if vals else float("nan")
        summary.append(out)
    summary.sort(
        key=lambda r: (
            float(r.get("eval_lwir_to_mm5_aligned_t16_ncc_mean", float("nan"))),
            -float(r.get("eval_lwir_to_mm5_aligned_t16_edge_distance_mean", float("inf"))),
            float(r.get("eval_rgb_to_mm5_aligned_rgb_ncc_mean", float("nan"))),
        ),
        reverse=True,
    )
    return summary


def load_bridge_metrics(path: str | Path) -> dict:
    p = Path(path)
    if not p.exists():
        return {}
    rows = []
    with p.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            if row.get("pipeline") == "raw_lwir_to_official_lwir" and row.get("stage") in {"sift_ransac", "ecc_final"}:
                rows.append(row)
    if not rows:
        return {}
    ncc = [float(r["ncc"]) for r in rows if r.get("ncc")]
    edge = [float(r["edge_distance"]) for r in rows if r.get("edge_distance")]
    return {
        "source": str(p),
        "raw_lwir_to_official_lwir_ncc_mean": float(np.mean(ncc)) if ncc else float("nan"),
        "raw_lwir_to_official_lwir_ncc_min": float(np.min(ncc)) if ncc else float("nan"),
        "raw_lwir_to_official_lwir_edge_distance_mean": float(np.mean(edge)) if edge else float("nan"),
    }


def write_report(output_dir: Path, summary_rows: list[dict], bridge_metrics: dict, args) -> None:
    report = output_dir / "calibration_only_report.md"
    best = summary_rows[0] if summary_rows else {}
    lines = [
        "# Calibration-Only Method Report",
        "",
        "## Constraint",
        "Generation used only calibration data and raw RGB1/LWIR inputs. MM5 aligned images were loaded only after generation for evaluation.",
        "",
        "## Run Configuration",
        f"- target_size: `{args.target_size}`",
        f"- plane_depths_mm: `{args.plane_depths}`",
        f"- lwir_calib_sizes: `{args.lwir_calib_sizes}`",
        f"- thermal_camera_calibration: `{args.thermal_camera_calibration}`",
        "",
        "## MM5 Aligned Bridge Target",
    ]
    if bridge_metrics:
        lines.extend(
            [
                f"- bridge_lwir_ncc_mean: `{bridge_metrics.get('raw_lwir_to_official_lwir_ncc_mean', float('nan')):.4f}`",
                f"- bridge_lwir_ncc_min: `{bridge_metrics.get('raw_lwir_to_official_lwir_ncc_min', float('nan')):.4f}`",
                f"- bridge_lwir_edge_distance_mean: `{bridge_metrics.get('raw_lwir_to_official_lwir_edge_distance_mean', float('nan')):.4f}`",
            ]
        )
    else:
        lines.append("- bridge metrics were not found")

    lines.extend(["", "## Best Calibration-Only Candidate"])
    if best:
        lines.extend(
            [
                f"- candidate: `{best['candidate']}`",
                f"- method: `{best.get('method', '')}`",
                f"- lwir_to_aligned_t16_ncc_mean: `{best.get('eval_lwir_to_mm5_aligned_t16_ncc_mean', float('nan')):.4f}`",
                f"- lwir_to_aligned_t16_ncc_min: `{best.get('eval_lwir_to_mm5_aligned_t16_ncc_min', float('nan')):.4f}`",
                f"- lwir_to_aligned_t16_edge_distance_mean: `{best.get('eval_lwir_to_mm5_aligned_t16_edge_distance_mean', float('nan')):.4f}`",
                f"- rgb_to_aligned_rgb_ncc_mean: `{best.get('eval_rgb_to_mm5_aligned_rgb_ncc_mean', float('nan')):.4f}`",
                f"- cross_rgb_lwir_ncc_mean: `{best.get('eval_cross_rgb_lwir_ncc_mean', float('nan')):.4f}`",
            ]
        )

    lines.extend(["", "## Top Candidates", "", "| candidate | method | LWIR NCC mean | LWIR edge mean | RGB NCC mean | valid mean |", "|---|---|---:|---:|---:|---:|"])
    for row in summary_rows[:12]:
        lines.append(
            "| {candidate} | {method} | {lncc:.4f} | {ledge:.4f} | {rncc:.4f} | {valid:.4f} |".format(
                candidate=row["candidate"],
                method=row.get("method", ""),
                lncc=row.get("eval_lwir_to_mm5_aligned_t16_ncc_mean", float("nan")),
                ledge=row.get("eval_lwir_to_mm5_aligned_t16_edge_distance_mean", float("nan")),
                rncc=row.get("eval_rgb_to_mm5_aligned_rgb_ncc_mean", float("nan")),
                valid=row.get("lwir_valid_ratio_mean", float("nan")),
            )
        )
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")


def choose_rows(index: str, aligned_ids: str, limit: int, splits: str):
    rows = load_rows(index, require_official=False, require_depth=False)
    if aligned_ids.strip():
        wanted = {int(x.strip()) for x in aligned_ids.split(",") if x.strip()}
        return sorted([r for r in rows if r.aligned_id in wanted], key=lambda r: r.aligned_id)
    return select_dark_rows(rows, limit, splits)


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibration-only aligned-size RGB/LWIR reconstruction.")
    parser.add_argument("--index", default="mm5_calib_benchmark/outputs/mm5_benchmark/splits/index_with_splits.csv")
    parser.add_argument("--calibration", default="calibration/def_stereocalib_THERM.yml")
    parser.add_argument("--output", default="darklight_mm5/calibration_only_method/outputs")
    parser.add_argument("--aligned-ids", default="106,104,103")
    parser.add_argument("--limit", type=int, default=3)
    parser.add_argument("--splits", default="test,val")
    parser.add_argument("--target-size", default="640x480")
    parser.add_argument("--plane-depths", default="300,325,500,700")
    parser.add_argument("--lwir-calib-sizes", default="1280x720,1280x1024")
    parser.add_argument("--thermal-camera-calibration", default="calibration/def_thermalcam_ori.yml")
    parser.add_argument("--save-top", type=int, default=6)
    parser.add_argument("--bridge-metrics", default="darklight_mm5/outputs/metrics/registration_stages.csv")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    target_size = parse_size_arg(args.target_size)
    if target_size is None:
        raise ValueError("--target-size must be an explicit WxH size")
    plane_depths = parse_float_list(args.plane_depths)
    lwir_calib_sizes = parse_size_list(args.lwir_calib_sizes)
    rows = choose_rows(args.index, args.aligned_ids, args.limit, args.splits)
    calibration = load_stereo_calibration(args.calibration)
    rectification = read_rectification_matrices(args.calibration)
    thermal_model = read_single_camera_model(args.thermal_camera_calibration, "thermal_ori")
    extra_lwir_models = [thermal_model] if thermal_model is not None else []

    config = {
        "generation_constraint": "calibration_and_raw_only",
        "evaluation_only_references": "MM5 aligned RGB1/T16",
        "index": args.index,
        "calibration": args.calibration,
        "target_size": list(target_size),
        "plane_depths_mm": plane_depths,
        "lwir_calib_sizes": [None if s is None else list(s) for s in lwir_calib_sizes],
        "thermal_camera_calibration": args.thermal_camera_calibration if thermal_model is not None else None,
        "aligned_ids": args.aligned_ids,
    }
    (output_dir / "calibration_only_config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")

    all_metric_rows = []
    candidate_cache: dict[tuple[str, str], CandidateOutput] = {}
    for row in rows:
        sid = sample_id(row)
        print(f"processing {sid} calibration-only candidates")
        raw_rgb = imread_unicode(row.raw_rgb1_path, cv2.IMREAD_COLOR)
        raw_lwir = imread_unicode(row.raw_thermal16_path, cv2.IMREAD_UNCHANGED)
        raw_lwir_u8 = normalize_u8(raw_lwir)
        candidates = build_candidates(
            raw_rgb,
            raw_lwir_u8,
            calibration,
            rectification,
            target_size,
            plane_depths,
            lwir_calib_sizes,
            extra_lwir_models,
        )

        aligned_rgb = imread_unicode(row.aligned_rgb1_path, cv2.IMREAD_COLOR)
        aligned_lwir_u8 = normalize_u8(imread_unicode(row.aligned_t16_path, cv2.IMREAD_UNCHANGED))
        for candidate in candidates:
            metrics = evaluate_candidate(row, candidate, aligned_rgb, aligned_lwir_u8)
            all_metric_rows.append(metrics)
            candidate_cache[(sid, candidate.name)] = candidate

    summary_rows = summarize_candidates(all_metric_rows)
    write_csv(output_dir / "metrics" / "per_sample_candidates.csv", all_metric_rows, collect_fieldnames(all_metric_rows, []))
    write_csv(output_dir / "metrics" / "candidate_summary.csv", summary_rows, collect_fieldnames(summary_rows, []))
    bridge_metrics = load_bridge_metrics(args.bridge_metrics)
    summary_payload = {
        "bridge_metrics": bridge_metrics,
        "best_candidate": summary_rows[0] if summary_rows else None,
        "candidates": summary_rows,
    }
    (output_dir / "metrics" / "candidate_summary.json").parent.mkdir(parents=True, exist_ok=True)
    (output_dir / "metrics" / "candidate_summary.json").write_text(json.dumps(summary_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    top_names = {row["candidate"] for row in summary_rows[: max(1, int(args.save_top))]}
    for row in rows:
        sid = sample_id(row)
        aligned_rgb = imread_unicode(row.aligned_rgb1_path, cv2.IMREAD_COLOR)
        aligned_lwir_u8 = normalize_u8(imread_unicode(row.aligned_t16_path, cv2.IMREAD_UNCHANGED))
        for name in top_names:
            candidate = candidate_cache.get((sid, name))
            if candidate is not None:
                save_candidate_outputs(output_dir, row, candidate, aligned_rgb, aligned_lwir_u8)

    write_report(output_dir, summary_rows, bridge_metrics, args)
    print(f"best candidate: {summary_rows[0]['candidate'] if summary_rows else 'none'}")
    print("done")


if __name__ == "__main__":
    main()
