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
if str(DARKLIGHT_DIR) not in sys.path:
    sys.path.insert(0, str(DARKLIGHT_DIR))

from run_darklight import (  # noqa: E402
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
    normalize_u8,
    parse_size_arg,
    select_dark_rows,
    write_csv,
)
from run_calibration_only import (  # noqa: E402
    CameraModel,
    CandidateOutput,
    RectificationMatrices,
    adjusted_stereo_calibration,
    calibration_crop_modes,
    crop_mask_with_border,
    crop_offset_rectified_principal,
    crop_with_border,
    read_rectification_matrices,
    read_single_camera_model,
    rectified_remap,
    sample_id,
)


@dataclass(frozen=True)
class CanvasRule:
    name: str
    rgb_offset: tuple[int, int]
    lwir_offset: tuple[int, int]
    rgb_source: str
    lwir_source: str
    allowed_for_generation: bool
    source: str


@dataclass(frozen=True)
class LwirModel:
    label: str
    camera: CameraModel
    source: str


def parse_size_list(text: str) -> list[tuple[int, int] | None]:
    out = []
    for part in str(text).split(","):
        part = part.strip()
        if part:
            out.append(parse_size_arg(part))
    return out


def parse_probe_sizes(text: str) -> list[tuple[int, int]]:
    sizes = []
    for size in parse_size_list(text):
        if size is not None:
            sizes.append(size)
    return sizes


def tuple2(offset: tuple[int, int]) -> list[int]:
    return [int(offset[0]), int(offset[1])]


def clamp_crop_offset(offset: tuple[int, int], canvas_size: tuple[int, int], target_size: tuple[int, int]) -> tuple[int, int]:
    w, h = canvas_size
    tw, th = target_size
    x = int(round(offset[0]))
    y = int(round(offset[1]))
    return max(0, min(x, max(0, w - tw))), max(0, min(y, max(0, h - th)))


def bbox_from_mask(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    ys, xs = np.where(mask.astype(bool))
    if xs.size == 0 or ys.size == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def crop_offset_from_bbox_center(mask: np.ndarray, target_size: tuple[int, int]) -> tuple[int, int] | None:
    bbox = bbox_from_mask(mask)
    if bbox is None:
        return None
    x0, y0, x1, y1 = bbox
    tw, th = target_size
    return int(round((x0 + x1 - tw) / 2.0)), int(round((y0 + y1 - th) / 2.0))


def crop_offset_from_bbox_origin(mask: np.ndarray) -> tuple[int, int] | None:
    bbox = bbox_from_mask(mask)
    if bbox is None:
        return None
    return int(bbox[0]), int(bbox[1])


def rectified_full(
    img: np.ndarray,
    K: np.ndarray,
    D: np.ndarray,
    R: np.ndarray,
    P: np.ndarray,
    canvas_size: tuple[int, int],
    interpolation: int,
) -> tuple[np.ndarray, np.ndarray]:
    new_K = P[:, :3].astype(np.float64)
    map_x, map_y = cv2.initUndistortRectifyMap(
        K.astype(np.float64),
        D.reshape(-1, 1).astype(np.float64),
        R.astype(np.float64),
        new_K,
        canvas_size,
        cv2.CV_32FC1,
    )
    remapped = cv2.remap(img, map_x, map_y, interpolation, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    valid_src = np.ones(img.shape[:2], dtype=np.uint8) * 255
    valid = cv2.remap(valid_src, map_x, map_y, cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0) > 0
    return remapped, valid


def load_pv_device_crop_rules(
    device_jsons: list[Path],
    raw_rgb_shape,
    target_size: tuple[int, int],
) -> list[CanvasRule]:
    rules: list[CanvasRule] = []
    raw_w, raw_h = raw_rgb_shape[1], raw_rgb_shape[0]
    target_w, target_h = target_size
    for path in device_jsons:
        if not path.exists():
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        cameras = data.get("CalibrationInformation", {}).get("Cameras", [])
        for camera_index, camera in enumerate(cameras):
            if camera.get("Purpose") != "CALIBRATION_CameraPurposePhotoVideo":
                continue
            sensor_w = int(camera.get("SensorWidth", raw_w))
            sensor_h = int(camera.get("SensorHeight", raw_h))
            params = camera.get("Intrinsics", {}).get("ModelParameters", [])
            if len(params) < 4:
                continue

            # Azure/Kinect-style JSONs in this dataset appear normalized, but the
            # parameter order is not documented locally. Keep both plausible
            # interpretations as explicit diagnostics.
            interpretations = [
                ("pv_json_cxcyfxfy", float(params[0]), float(params[1])),
                ("pv_json_fxfycxcy", float(params[2]), float(params[3])),
            ]
            for label, cx_norm, cy_norm in interpretations:
                cx = cx_norm * sensor_w * raw_w / float(sensor_w)
                cy = cy_norm * sensor_h * raw_h / float(sensor_h)
                offset = clamp_crop_offset((round(cx - target_w / 2.0), round(cy - target_h / 2.0)), (raw_w, raw_h), target_size)
                stem = path.stem.replace("calib_", "")
                name = f"raw_rgb_{stem}_cam{camera_index}_{label}"
                rules.append(
                    CanvasRule(
                        name=name,
                        rgb_offset=offset,
                        lwir_offset=offset,
                        rgb_source="raw_crop",
                        lwir_source="rectified_lwir_thermal_ori",
                        allowed_for_generation=True,
                        source=f"{path.as_posix()} PV normalized intrinsics interpreted as {label}",
                    )
                )
    return rules


def build_lwir_models(
    calibration,
    raw_rgb_shape,
    raw_lwir_shape,
    lwir_calib_sizes: list[tuple[int, int] | None],
    thermal_model: CameraModel | None,
) -> list[LwirModel]:
    out: list[LwirModel] = []
    if thermal_model is not None:
        out.append(LwirModel("thermal_ori", thermal_model, "calibration/def_thermalcam_ori.yml"))
    for size in lwir_calib_sizes:
        effective = adjusted_stereo_calibration(
            calibration,
            raw_rgb_shape,
            raw_lwir_shape,
            rgb_calib_size=None,
            lwir_calib_size=size,
            lwir_principal_offset=(0.0, 0.0),
            t_scale=1.0,
        )
        label = "stereo_raw" if size is None else f"stereo_scaled_{size[0]}x{size[1]}"
        out.append(LwirModel(label, CameraModel(label, effective.lwir_K, effective.lwir_D), "def_stereocalib_THERM.yml CM2/D2"))
    return out


def add_rule(rules: list[CanvasRule], seen: set[tuple], rule: CanvasRule, canvas_size: tuple[int, int], target_size: tuple[int, int]) -> None:
    rgb_offset = clamp_crop_offset(rule.rgb_offset, canvas_size, target_size)
    lwir_offset = clamp_crop_offset(rule.lwir_offset, canvas_size, target_size)
    key = (rule.name, rgb_offset, lwir_offset, rule.rgb_source, rule.lwir_source)
    if key in seen:
        return
    seen.add(key)
    rules.append(
        CanvasRule(
            name=rule.name,
            rgb_offset=rgb_offset,
            lwir_offset=lwir_offset,
            rgb_source=rule.rgb_source,
            lwir_source=rule.lwir_source,
            allowed_for_generation=rule.allowed_for_generation,
            source=rule.source,
        )
    )


def build_canvas_rules(
    raw_rgb: np.ndarray,
    raw_lwir_u8: np.ndarray,
    calibration,
    rectification: RectificationMatrices,
    target_size: tuple[int, int],
    lwir_model: LwirModel,
    device_jsons: list[Path],
) -> list[CanvasRule]:
    canvas_size = (raw_rgb.shape[1], raw_rgb.shape[0])
    rules: list[CanvasRule] = []
    seen: set[tuple] = set()

    for crop_name, offset in calibration_crop_modes(calibration, raw_rgb.shape, target_size).items():
        add_rule(
            rules,
            seen,
            CanvasRule(
                name=f"shared_{crop_name}",
                rgb_offset=offset,
                lwir_offset=offset,
                rgb_source="raw_crop",
                lwir_source=f"rectified_lwir_{lwir_model.label}",
                allowed_for_generation=True,
                source=f"raw RGB geometry / CM1 getOptimalNewCameraMatrix crop mode {crop_name}",
            ),
            canvas_size,
            target_size,
        )

    if rectification.P1 is not None:
        offset = crop_offset_rectified_principal(rectification.P1, target_size)
        add_rule(
            rules,
            seen,
            CanvasRule(
                name="shared_p1_rectified_principal",
                rgb_offset=offset,
                lwir_offset=offset,
                rgb_source="raw_crop",
                lwir_source=f"rectified_lwir_{lwir_model.label}",
                allowed_for_generation=True,
                source="P1 principal point centered in the 640x480 output",
            ),
            canvas_size,
            target_size,
        )
    if rectification.P2 is not None:
        offset = crop_offset_rectified_principal(rectification.P2, target_size)
        add_rule(
            rules,
            seen,
            CanvasRule(
                name="shared_p2_rectified_principal",
                rgb_offset=offset,
                lwir_offset=offset,
                rgb_source="raw_crop",
                lwir_source=f"rectified_lwir_{lwir_model.label}",
                allowed_for_generation=True,
                source="P2 principal point centered in the 640x480 output",
            ),
            canvas_size,
            target_size,
        )

    if rectification.R2 is not None and rectification.P2 is not None:
        _, lwir_valid_full = rectified_full(
            raw_lwir_u8,
            lwir_model.camera.K,
            lwir_model.camera.D,
            rectification.R2,
            rectification.P2,
            canvas_size,
            cv2.INTER_LINEAR,
        )
        valid_offsets = {
            "lwir_valid_bbox_center": crop_offset_from_bbox_center(lwir_valid_full, target_size),
            "lwir_valid_bbox_origin": crop_offset_from_bbox_origin(lwir_valid_full),
        }
        for name, lwir_offset in valid_offsets.items():
            if lwir_offset is None:
                continue
            for rgb_name, rgb_offset in calibration_crop_modes(calibration, raw_rgb.shape, target_size).items():
                add_rule(
                    rules,
                    seen,
                    CanvasRule(
                        name=f"{name}_rgb_{rgb_name}",
                        rgb_offset=rgb_offset,
                        lwir_offset=lwir_offset,
                        rgb_source="raw_crop",
                        lwir_source=f"rectified_lwir_{lwir_model.label}",
                        allowed_for_generation=True,
                        source=f"{name} from full {lwir_model.label} rectification valid mask; RGB from {rgb_name}",
                    ),
                    canvas_size,
                    target_size,
                )

        if rectification.R1 is not None and rectification.P1 is not None:
            _, rgb_rect_valid_full = rectified_full(
                raw_rgb,
                calibration.rgb_K,
                calibration.rgb_D,
                rectification.R1,
                rectification.P1,
                canvas_size,
                cv2.INTER_LINEAR,
            )
            inter = rgb_rect_valid_full & lwir_valid_full
            inter_center = crop_offset_from_bbox_center(inter, target_size)
            inter_origin = crop_offset_from_bbox_origin(inter)
            for name, offset in {
                "rectified_valid_intersection_center": inter_center,
                "rectified_valid_intersection_origin": inter_origin,
            }.items():
                if offset is None:
                    continue
                add_rule(
                    rules,
                    seen,
                    CanvasRule(
                        name=name,
                        rgb_offset=offset,
                        lwir_offset=offset,
                        rgb_source="raw_crop",
                        lwir_source=f"rectified_lwir_{lwir_model.label}",
                        allowed_for_generation=True,
                        source="intersection of full RGB/LWIR rectification valid masks",
                    ),
                    canvas_size,
                    target_size,
                )

            crop_modes = calibration_crop_modes(calibration, raw_rgb.shape, target_size)
            if inter_center is not None and "rgb_optimal_alpha1_0" in crop_modes:
                mixed = (inter_center[0], crop_modes["rgb_optimal_alpha1_0"][1])
                add_rule(
                    rules,
                    seen,
                    CanvasRule(
                        name="rectified_intersection_x_rgb_optimal_alpha1_y",
                        rgb_offset=mixed,
                        lwir_offset=mixed,
                        rgb_source="raw_crop",
                        lwir_source=f"rectified_lwir_{lwir_model.label}",
                        allowed_for_generation=True,
                        source="x from RGB/LWIR valid-mask intersection, y from RGB getOptimalNewCameraMatrix alpha=1",
                    ),
                    canvas_size,
                    target_size,
                )

    for rule in load_pv_device_crop_rules(device_jsons, raw_rgb.shape, target_size):
        add_rule(
            rules,
            seen,
            CanvasRule(
                name=rule.name,
                rgb_offset=rule.rgb_offset,
                lwir_offset=rule.lwir_offset,
                rgb_source=rule.rgb_source,
                lwir_source=f"rectified_lwir_{lwir_model.label}",
                allowed_for_generation=rule.allowed_for_generation,
                source=rule.source,
            ),
            canvas_size,
            target_size,
        )

    return rules


def make_candidate_from_rule(
    raw_rgb: np.ndarray,
    raw_lwir_u8: np.ndarray,
    rectification: RectificationMatrices,
    target_size: tuple[int, int],
    lwir_model: LwirModel,
    rule: CanvasRule,
) -> CandidateOutput:
    rgb_valid_full = np.ones(raw_rgb.shape[:2], dtype=bool)
    rgb = crop_with_border(raw_rgb, rule.rgb_offset, target_size)
    rgb_valid = crop_mask_with_border(rgb_valid_full, rule.rgb_offset, target_size)

    if rule.lwir_source.startswith("rectified_lwir_"):
        if rectification.R2 is None or rectification.P2 is None:
            raise ValueError("R2/P2 are required for rectified LWIR rules")
        lwir, lwir_valid = rectified_remap(
            raw_lwir_u8,
            lwir_model.camera.K,
            lwir_model.camera.D,
            rectification.R2,
            rectification.P2,
            rule.lwir_offset,
            target_size,
            cv2.INTER_LINEAR,
        )
    elif rule.lwir_source == "raw_crop":
        lwir_valid_full = np.ones(raw_lwir_u8.shape[:2], dtype=bool)
        lwir = crop_with_border(raw_lwir_u8, rule.lwir_offset, target_size)
        lwir_valid = crop_mask_with_border(lwir_valid_full, rule.lwir_offset, target_size)
    else:
        raise ValueError(f"unsupported LWIR source: {rule.lwir_source}")

    return CandidateOutput(
        name=rule.name,
        rgb=rgb,
        lwir=lwir,
        rgb_valid=rgb_valid,
        lwir_valid=lwir_valid,
        metadata={
            "method": "phase20_canvas_rule",
            "allowed_for_generation": rule.allowed_for_generation,
            "rgb_source": rule.rgb_source,
            "lwir_source": rule.lwir_source,
            "lwir_model": lwir_model.label,
            "rgb_crop_offset_xy": tuple2(rule.rgb_offset),
            "lwir_crop_offset_xy": tuple2(rule.lwir_offset),
            "rule_source": rule.source,
        },
    )


def metric_prefix(prefix: str, metrics: dict) -> dict:
    return {f"{prefix}_{key}": value for key, value in direct_metrics_for_csv(metrics).items()}


def evaluate_candidate(row, candidate: CandidateOutput, aligned_rgb: np.ndarray, aligned_lwir_u8: np.ndarray) -> dict:
    rgb_valid = candidate.rgb_valid.astype(bool)
    lwir_valid = candidate.lwir_valid.astype(bool)
    both_valid = rgb_valid & lwir_valid
    ref_rgb_gray = gray_u8(aligned_rgb)
    cand_rgb_gray = gray_u8(candidate.rgb)

    rgb_metrics = direct_alignment_metrics(ref_rgb_gray, cand_rgb_gray, rgb_valid)
    lwir_metrics = direct_alignment_metrics(aligned_lwir_u8, candidate.lwir, lwir_valid)
    cross_metrics = direct_alignment_metrics(cand_rgb_gray, candidate.lwir, both_valid)
    out = {
        "sample": sample_id(row),
        "aligned_id": row.aligned_id,
        "sequence": row.sequence,
        "candidate": candidate.name,
        "allowed_for_generation": bool(candidate.metadata.get("allowed_for_generation", False)),
        "rgb_source": candidate.metadata.get("rgb_source", ""),
        "lwir_source": candidate.metadata.get("lwir_source", ""),
        "lwir_model": candidate.metadata.get("lwir_model", ""),
        "rgb_crop_offset_xy": json.dumps(candidate.metadata.get("rgb_crop_offset_xy", [])),
        "lwir_crop_offset_xy": json.dumps(candidate.metadata.get("lwir_crop_offset_xy", [])),
        "rule_source": candidate.metadata.get("rule_source", ""),
        "rgb_valid_ratio": float(rgb_valid.mean()),
        "lwir_valid_ratio": float(lwir_valid.mean()),
        "intersection_valid_ratio": float(both_valid.mean()),
    }
    out.update(metric_prefix("eval_rgb_to_mm5_aligned_rgb", rgb_metrics))
    out.update(metric_prefix("eval_lwir_to_mm5_aligned_t16", lwir_metrics))
    out.update(metric_prefix("eval_cross_rgb_lwir", cross_metrics))
    return out


def match_template_offset(search_img: np.ndarray, template_img: np.ndarray) -> tuple[tuple[int, int], float]:
    search = gray_u8(search_img)
    template = gray_u8(template_img)
    if search.shape[0] < template.shape[0] or search.shape[1] < template.shape[1]:
        raise ValueError("template is larger than search image")
    result = cv2.matchTemplate(search, template, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(result)
    return (int(max_loc[0]), int(max_loc[1])), float(max_val)


def make_oracle_candidates(
    raw_rgb: np.ndarray,
    raw_lwir_u8: np.ndarray,
    aligned_rgb: np.ndarray,
    aligned_lwir_u8: np.ndarray,
    rectification: RectificationMatrices,
    target_size: tuple[int, int],
    lwir_model: LwirModel,
) -> tuple[list[CandidateOutput], list[dict]]:
    canvas_size = (raw_rgb.shape[1], raw_rgb.shape[0])
    rgb_offset, rgb_template_score = match_template_offset(raw_rgb, aligned_rgb)
    rgb_offset = clamp_crop_offset(rgb_offset, canvas_size, target_size)

    oracle_rows = [
        {
            "oracle": "rgb_raw_template",
            "lwir_model": lwir_model.label,
            "offset_xy": json.dumps(tuple2(rgb_offset)),
            "template_ncc": rgb_template_score,
            "allowed_for_generation": False,
            "reason": "uses MM5 aligned RGB as a template",
        }
    ]

    candidates: list[CandidateOutput] = []
    if rectification.R2 is not None and rectification.P2 is not None:
        lwir_full, lwir_valid_full = rectified_full(
            raw_lwir_u8,
            lwir_model.camera.K,
            lwir_model.camera.D,
            rectification.R2,
            rectification.P2,
            canvas_size,
            cv2.INTER_LINEAR,
        )
        lwir_offset, lwir_template_score = match_template_offset(lwir_full, aligned_lwir_u8)
        lwir_offset = clamp_crop_offset(lwir_offset, canvas_size, target_size)
        lwir = crop_with_border(lwir_full, lwir_offset, target_size)
        lwir_valid = crop_mask_with_border(lwir_valid_full, lwir_offset, target_size)
        rgb = crop_with_border(raw_rgb, rgb_offset, target_size)
        rgb_valid = crop_mask_with_border(np.ones(raw_rgb.shape[:2], dtype=bool), rgb_offset, target_size)
        candidates.append(
            CandidateOutput(
                name=f"oracle_rgb_template_lwir_template_{lwir_model.label}",
                rgb=rgb,
                lwir=lwir,
                rgb_valid=rgb_valid,
                lwir_valid=lwir_valid,
                metadata={
                    "method": "aligned_template_oracle",
                    "allowed_for_generation": False,
                    "rgb_source": "raw_crop_template_oracle",
                    "lwir_source": "rectified_lwir_template_oracle",
                    "lwir_model": lwir_model.label,
                    "rgb_crop_offset_xy": tuple2(rgb_offset),
                    "lwir_crop_offset_xy": tuple2(lwir_offset),
                    "rule_source": "diagnostic-only template match against MM5 aligned RGB/T16",
                },
            )
        )
        oracle_rows.append(
            {
                "oracle": "lwir_rectified_template",
                "lwir_model": lwir_model.label,
                "offset_xy": json.dumps(tuple2(lwir_offset)),
                "template_ncc": lwir_template_score,
                "allowed_for_generation": False,
                "reason": "uses MM5 aligned T16 as a template",
            }
        )
    return candidates, oracle_rows


def summarize(rows: list[dict]) -> list[dict]:
    grouped: dict[tuple[str, str], list[dict]] = {}
    for row in rows:
        grouped.setdefault((str(row["candidate"]), str(row.get("lwir_model", ""))), []).append(row)

    numeric_keys = [
        "rgb_valid_ratio",
        "lwir_valid_ratio",
        "intersection_valid_ratio",
        "eval_rgb_to_mm5_aligned_rgb_ncc",
        "eval_rgb_to_mm5_aligned_rgb_edge_distance",
        "eval_rgb_to_mm5_aligned_rgb_mi",
        "eval_lwir_to_mm5_aligned_t16_ncc",
        "eval_lwir_to_mm5_aligned_t16_edge_distance",
        "eval_lwir_to_mm5_aligned_t16_mi",
        "eval_cross_rgb_lwir_ncc",
        "eval_cross_rgb_lwir_edge_distance",
        "eval_cross_rgb_lwir_mi",
    ]
    summary = []
    for (candidate, lwir_model), items in grouped.items():
        out = {
            "candidate": candidate,
            "lwir_model": lwir_model,
            "samples": len(items),
            "allowed_for_generation": bool(items[0].get("allowed_for_generation", False)),
            "rgb_source": items[0].get("rgb_source", ""),
            "lwir_source": items[0].get("lwir_source", ""),
            "rule_source": items[0].get("rule_source", ""),
        }
        for key in numeric_keys:
            vals = [float(item[key]) for item in items if key in item and np.isfinite(float(item[key]))]
            out[f"{key}_mean"] = float(np.mean(vals)) if vals else float("nan")
            out[f"{key}_min"] = float(np.min(vals)) if vals else float("nan")
        summary.append(out)

    summary.sort(
        key=lambda row: (
            bool(row.get("allowed_for_generation", False)),
            float(row.get("eval_lwir_to_mm5_aligned_t16_ncc_mean", float("nan"))),
            float(row.get("eval_rgb_to_mm5_aligned_rgb_ncc_mean", float("nan"))),
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
    ncc = [float(row["ncc"]) for row in rows if row.get("ncc")]
    edge = [float(row["edge_distance"]) for row in rows if row.get("edge_distance")]
    return {
        "source": str(p),
        "raw_lwir_to_official_lwir_ncc_mean": float(np.mean(ncc)) if ncc else float("nan"),
        "raw_lwir_to_official_lwir_ncc_min": float(np.min(ncc)) if ncc else float("nan"),
        "raw_lwir_to_official_lwir_edge_distance_mean": float(np.mean(edge)) if edge else float("nan"),
    }


def choose_rows(index: str, aligned_ids: str, limit: int, splits: str):
    rows = load_rows(index, require_official=False, require_depth=False)
    if aligned_ids.strip():
        wanted = {int(x.strip()) for x in aligned_ids.split(",") if x.strip()}
        return sorted([row for row in rows if row.aligned_id in wanted], key=lambda row: row.aligned_id)
    return select_dark_rows(rows, limit, splits)


def roi_center_offset(roi: tuple[int, int, int, int], target_size: tuple[int, int]) -> tuple[int, int]:
    x, y, w, h = [int(v) for v in roi]
    target_w, target_h = target_size
    return int(round(x + w / 2.0 - target_w / 2.0)), int(round(y + h / 2.0 - target_h / 2.0))


def stereo_rectify_alpha_probe(calibration, target_size: tuple[int, int], probe_sizes: list[tuple[int, int]]) -> list[dict]:
    rows: list[dict] = []
    for size in probe_sizes:
        for alpha in (-1.0, 0.0, 0.25, 0.5, 0.75, 1.0):
            R1, R2, P1, P2, _Q, roi1, roi2 = cv2.stereoRectify(
                calibration.rgb_K,
                calibration.rgb_D,
                calibration.lwir_K,
                calibration.lwir_D,
                size,
                calibration.rgb_to_lwir_R,
                calibration.rgb_to_lwir_T,
                alpha=float(alpha),
                newImageSize=size,
            )
            rows.append(
                {
                    "image_size": f"{size[0]}x{size[1]}",
                    "alpha": alpha,
                    "p1_principal_offset_xy": json.dumps(tuple2((round(P1[0, 2] - target_size[0] / 2.0), round(P1[1, 2] - target_size[1] / 2.0)))),
                    "p2_principal_offset_xy": json.dumps(tuple2((round(P2[0, 2] - target_size[0] / 2.0), round(P2[1, 2] - target_size[1] / 2.0)))),
                    "roi1": json.dumps([int(v) for v in roi1]),
                    "roi2": json.dumps([int(v) for v in roi2]),
                    "roi1_center_offset_xy": json.dumps(tuple2(roi_center_offset(roi1, target_size))),
                    "roi2_center_offset_xy": json.dumps(tuple2(roi_center_offset(roi2, target_size))),
                    "p1_fx": float(P1[0, 0]),
                    "p2_fx": float(P2[0, 0]),
                    "r1_trace": float(np.trace(R1)),
                    "r2_trace": float(np.trace(R2)),
                }
            )
    return rows


def save_panel(output_dir: Path, row, candidate: CandidateOutput, aligned_rgb: np.ndarray, aligned_lwir_u8: np.ndarray) -> None:
    sid = sample_id(row)
    aligned_lwir_bgr = cv2.cvtColor(aligned_lwir_u8, cv2.COLOR_GRAY2BGR)
    edge = make_edge_overlay(candidate.rgb, candidate.lwir, candidate.rgb_valid & candidate.lwir_valid)
    make_five_panel(
        [
            (candidate.rgb, "Generated RGB"),
            (candidate.lwir, "Generated LWIR"),
            (aligned_rgb, "MM5 RGB eval"),
            (aligned_lwir_bgr, "MM5 T16 eval"),
            (edge, "Generated edge check"),
        ],
        output_dir / "panels" / f"{sid}_{candidate.name}.png",
        tile_size=(360, 270),
    )


def write_report(
    output_dir: Path,
    summary_rows: list[dict],
    oracle_rows: list[dict],
    bridge_metrics: dict,
    stereo_probe_rows: list[dict],
    args,
) -> None:
    allowed_rows = [row for row in summary_rows if row.get("allowed_for_generation")]
    oracle_summary = [row for row in summary_rows if not row.get("allowed_for_generation")]
    best_allowed = allowed_rows[0] if allowed_rows else None
    best_oracle = oracle_summary[0] if oracle_summary else None
    lines = [
        "# Phase 20 Aligned Canvas Diagnostic",
        "",
        "## Constraint",
        "Allowed candidates are generated only from calibration files, device metadata, output geometry, and raw RGB/LWIR inputs. MM5 aligned images are used only for evaluation and for a separately labeled oracle diagnostic.",
        "",
        "## Inputs",
        f"- target_size: `{args.target_size}`",
        f"- full_canvas_size: `{args.full_canvas_size}`",
        f"- calibration: `{args.calibration}`",
        f"- thermal_camera_calibration: `{args.thermal_camera_calibration}`",
        f"- device_jsons: `{args.device_jsons}`",
        "",
        "## Bridge Target",
    ]
    if bridge_metrics:
        lines.extend(
            [
                f"- retained bridge LWIR NCC mean: `{bridge_metrics.get('raw_lwir_to_official_lwir_ncc_mean', float('nan')):.4f}`",
                f"- retained bridge LWIR NCC min: `{bridge_metrics.get('raw_lwir_to_official_lwir_ncc_min', float('nan')):.4f}`",
                f"- retained bridge edge distance mean: `{bridge_metrics.get('raw_lwir_to_official_lwir_edge_distance_mean', float('nan')):.4f}`",
            ]
        )
    else:
        lines.append("- bridge metrics were not found")

    lines.extend(["", "## Best Allowed Calibration-Derived Rule"])
    if best_allowed:
        lines.extend(
            [
                f"- candidate: `{best_allowed['candidate']}`",
                f"- lwir_model: `{best_allowed.get('lwir_model', '')}`",
                f"- LWIR NCC mean/min: `{best_allowed.get('eval_lwir_to_mm5_aligned_t16_ncc_mean', float('nan')):.4f}` / `{best_allowed.get('eval_lwir_to_mm5_aligned_t16_ncc_min', float('nan')):.4f}`",
                f"- RGB NCC mean/min: `{best_allowed.get('eval_rgb_to_mm5_aligned_rgb_ncc_mean', float('nan')):.4f}` / `{best_allowed.get('eval_rgb_to_mm5_aligned_rgb_ncc_min', float('nan')):.4f}`",
                f"- LWIR edge distance mean: `{best_allowed.get('eval_lwir_to_mm5_aligned_t16_edge_distance_mean', float('nan')):.4f}`",
                f"- rule source: `{best_allowed.get('rule_source', '')}`",
            ]
        )
    else:
        lines.append("- no allowed candidates were produced")

    lines.extend(["", "## Oracle Diagnostic, Not Allowed For Generation"])
    if best_oracle:
        lines.extend(
            [
                f"- oracle candidate: `{best_oracle['candidate']}`",
                f"- LWIR NCC mean/min: `{best_oracle.get('eval_lwir_to_mm5_aligned_t16_ncc_mean', float('nan')):.4f}` / `{best_oracle.get('eval_lwir_to_mm5_aligned_t16_ncc_min', float('nan')):.4f}`",
                f"- RGB NCC mean/min: `{best_oracle.get('eval_rgb_to_mm5_aligned_rgb_ncc_mean', float('nan')):.4f}` / `{best_oracle.get('eval_rgb_to_mm5_aligned_rgb_ncc_min', float('nan')):.4f}`",
            ]
        )
    if oracle_rows:
        lines.extend(["", "| oracle | lwir model | offset xy | template NCC | reason |", "|---|---|---:|---:|---|"])
        for row in oracle_rows:
            lines.append(
                f"| {row['oracle']} | {row.get('lwir_model', '')} | `{row.get('offset_xy', '')}` | {float(row.get('template_ncc', float('nan'))):.4f} | {row.get('reason', '')} |"
            )

    lines.extend(["", "## StereoRectify Alpha Probe"])
    if stereo_probe_rows:
        p1_offsets = [json.loads(row["p1_principal_offset_xy"]) for row in stereo_probe_rows]
        p2_offsets = [json.loads(row["p2_principal_offset_xy"]) for row in stereo_probe_rows]
        roi_offsets = [json.loads(row["roi1_center_offset_xy"]) for row in stereo_probe_rows] + [
            json.loads(row["roi2_center_offset_xy"]) for row in stereo_probe_rows
        ]
        lines.extend(
            [
                "- wrote `metrics/stereo_rectify_alpha_probe.csv`",
                f"- P1 principal offset x range: `{min(v[0] for v in p1_offsets)}..{max(v[0] for v in p1_offsets)}`; y range: `{min(v[1] for v in p1_offsets)}..{max(v[1] for v in p1_offsets)}`",
                f"- P2 principal offset x range: `{min(v[0] for v in p2_offsets)}..{max(v[0] for v in p2_offsets)}`; y range: `{min(v[1] for v in p2_offsets)}..{max(v[1] for v in p2_offsets)}`",
                f"- ROI-center offset x range: `{min(v[0] for v in roi_offsets)}..{max(v[0] for v in roi_offsets)}`; y range: `{min(v[1] for v in roi_offsets)}..{max(v[1] for v in roi_offsets)}`",
            ]
        )
    else:
        lines.append("- no stereoRectify probe rows were produced")

    lines.extend(
        [
            "",
            "## Top Allowed Rules",
            "",
            "| candidate | model | LWIR NCC mean | RGB NCC mean | LWIR edge mean | source |",
            "|---|---|---:|---:|---:|---|",
        ]
    )
    for row in allowed_rows[:12]:
        lines.append(
            "| {candidate} | {model} | {lncc:.4f} | {rncc:.4f} | {ledge:.4f} | {source} |".format(
                candidate=row["candidate"],
                model=row.get("lwir_model", ""),
                lncc=row.get("eval_lwir_to_mm5_aligned_t16_ncc_mean", float("nan")),
                rncc=row.get("eval_rgb_to_mm5_aligned_rgb_ncc_mean", float("nan")),
                ledge=row.get("eval_lwir_to_mm5_aligned_t16_edge_distance_mean", float("nan")),
                source=row.get("rule_source", ""),
            )
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "canvas_diagnostic_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose MM5 aligned canvas conventions without using aligned images for generation.")
    parser.add_argument("--index", default="mm5_calib_benchmark/outputs/mm5_benchmark/splits/index_with_splits.csv")
    parser.add_argument("--calibration", default="calibration/def_stereocalib_THERM.yml")
    parser.add_argument("--thermal-camera-calibration", default="calibration/def_thermalcam_ori.yml")
    parser.add_argument("--device-jsons", default="calibration/calib_device_0.json,calibration/calib_device_1.json")
    parser.add_argument("--output", default="darklight_mm5/calibration_only_method/outputs_phase20_canvas")
    parser.add_argument("--aligned-ids", default="106,104,103")
    parser.add_argument("--limit", type=int, default=3)
    parser.add_argument("--splits", default="test,val")
    parser.add_argument("--target-size", default="640x480")
    parser.add_argument("--full-canvas-size", default="1280x720")
    parser.add_argument("--lwir-calib-sizes", default="1280x720,1280x1024")
    parser.add_argument("--stereo-rectify-probe-sizes", default="1280x720,1280x1024,640x480,640x512")
    parser.add_argument("--bridge-metrics", default="darklight_mm5/outputs/metrics/registration_stages.csv")
    parser.add_argument("--save-top", type=int, default=4)
    args = parser.parse_args()

    target_size = parse_size_arg(args.target_size)
    full_canvas_size = parse_size_arg(args.full_canvas_size)
    if target_size is None or full_canvas_size is None:
        raise ValueError("--target-size and --full-canvas-size must be explicit WxH sizes")

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = choose_rows(args.index, args.aligned_ids, args.limit, args.splits)
    calibration = load_stereo_calibration(args.calibration)
    rectification = read_rectification_matrices(args.calibration)
    thermal_model = read_single_camera_model(args.thermal_camera_calibration, "thermal_ori")
    device_jsons = [Path(part.strip()) for part in args.device_jsons.split(",") if part.strip()]
    lwir_calib_sizes = parse_size_list(args.lwir_calib_sizes)

    config = {
        "generation_constraint": "calibration_metadata_and_raw_inputs_only",
        "aligned_usage": "evaluation_and_oracle_diagnostic_only",
        "index": args.index,
        "calibration": args.calibration,
        "thermal_camera_calibration": args.thermal_camera_calibration,
        "device_jsons": [str(path) for path in device_jsons],
        "target_size": list(target_size),
        "full_canvas_size": list(full_canvas_size),
        "aligned_ids": args.aligned_ids,
        "lwir_calib_sizes": [None if size is None else list(size) for size in lwir_calib_sizes],
        "stereo_rectify_probe_sizes": args.stereo_rectify_probe_sizes,
    }
    (output_dir / "canvas_diagnostic_config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")

    stereo_probe_rows = stereo_rectify_alpha_probe(calibration, target_size, parse_probe_sizes(args.stereo_rectify_probe_sizes))
    write_csv(output_dir / "metrics" / "stereo_rectify_alpha_probe.csv", stereo_probe_rows, collect_fieldnames(stereo_probe_rows, []))

    all_metric_rows: list[dict] = []
    all_oracle_rows: list[dict] = []
    candidate_cache: dict[tuple[str, str, str], CandidateOutput] = {}

    for row in rows:
        sid = sample_id(row)
        print(f"processing {sid} phase20 canvas rules")
        raw_rgb = imread_unicode(row.raw_rgb1_path, cv2.IMREAD_COLOR)
        raw_lwir_u8 = normalize_u8(imread_unicode(row.raw_thermal16_path, cv2.IMREAD_UNCHANGED))
        aligned_rgb = imread_unicode(row.aligned_rgb1_path, cv2.IMREAD_COLOR)
        aligned_lwir_u8 = normalize_u8(imread_unicode(row.aligned_t16_path, cv2.IMREAD_UNCHANGED))

        lwir_models = build_lwir_models(calibration, raw_rgb.shape, raw_lwir_u8.shape, lwir_calib_sizes, thermal_model)
        for lwir_model in lwir_models:
            rules = build_canvas_rules(
                raw_rgb,
                raw_lwir_u8,
                calibration,
                rectification,
                target_size,
                lwir_model,
                device_jsons,
            )
            rule_rows = [
                {
                    "sample": sid,
                    "lwir_model": lwir_model.label,
                    "rule": rule.name,
                    "allowed_for_generation": rule.allowed_for_generation,
                    "rgb_offset_xy": json.dumps(tuple2(rule.rgb_offset)),
                    "lwir_offset_xy": json.dumps(tuple2(rule.lwir_offset)),
                    "rgb_source": rule.rgb_source,
                    "lwir_source": rule.lwir_source,
                    "source": rule.source,
                }
                for rule in rules
            ]
            write_csv(
                output_dir / "rules" / f"{sid}_{lwir_model.label}_rules.csv",
                rule_rows,
                collect_fieldnames(rule_rows, []),
            )

            for rule in rules:
                candidate = make_candidate_from_rule(raw_rgb, raw_lwir_u8, rectification, target_size, lwir_model, rule)
                metrics = evaluate_candidate(row, candidate, aligned_rgb, aligned_lwir_u8)
                all_metric_rows.append(metrics)
                candidate_cache[(sid, candidate.name, lwir_model.label)] = candidate

            oracle_candidates, oracle_rows = make_oracle_candidates(
                raw_rgb,
                raw_lwir_u8,
                aligned_rgb,
                aligned_lwir_u8,
                rectification,
                target_size,
                lwir_model,
            )
            for oracle_row in oracle_rows:
                oracle_row = dict(oracle_row)
                oracle_row["sample"] = sid
                all_oracle_rows.append(oracle_row)
            for candidate in oracle_candidates:
                metrics = evaluate_candidate(row, candidate, aligned_rgb, aligned_lwir_u8)
                all_metric_rows.append(metrics)
                candidate_cache[(sid, candidate.name, lwir_model.label)] = candidate

    summary_rows = summarize(all_metric_rows)
    write_csv(output_dir / "metrics" / "per_sample_canvas_candidates.csv", all_metric_rows, collect_fieldnames(all_metric_rows, []))
    write_csv(output_dir / "metrics" / "canvas_candidate_summary.csv", summary_rows, collect_fieldnames(summary_rows, []))
    write_csv(output_dir / "metrics" / "oracle_offsets.csv", all_oracle_rows, collect_fieldnames(all_oracle_rows, []))

    bridge_metrics = load_bridge_metrics(args.bridge_metrics)
    payload = {
        "bridge_metrics": bridge_metrics,
        "best_allowed_candidate": next((row for row in summary_rows if row.get("allowed_for_generation")), None),
        "best_oracle_candidate": next((row for row in summary_rows if not row.get("allowed_for_generation")), None),
        "candidate_summary": summary_rows,
    }
    (output_dir / "metrics" / "canvas_candidate_summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    selected_for_panels = []
    for row in summary_rows:
        if row.get("allowed_for_generation") and len([r for r in selected_for_panels if r.get("allowed_for_generation")]) < args.save_top:
            selected_for_panels.append(row)
        if not row.get("allowed_for_generation") and len([r for r in selected_for_panels if not r.get("allowed_for_generation")]) < 1:
            selected_for_panels.append(row)

    for source_row in rows:
        sid = sample_id(source_row)
        aligned_rgb = imread_unicode(source_row.aligned_rgb1_path, cv2.IMREAD_COLOR)
        aligned_lwir_u8 = normalize_u8(imread_unicode(source_row.aligned_t16_path, cv2.IMREAD_UNCHANGED))
        for summary_row in selected_for_panels:
            candidate = candidate_cache.get((sid, summary_row["candidate"], summary_row.get("lwir_model", "")))
            if candidate is not None:
                save_panel(output_dir, source_row, candidate, aligned_rgb, aligned_lwir_u8)

    write_report(output_dir, summary_rows, all_oracle_rows, bridge_metrics, stereo_probe_rows, args)
    best_allowed = next((row for row in summary_rows if row.get("allowed_for_generation")), None)
    if best_allowed:
        print(f"best allowed candidate: {best_allowed['candidate']} ({best_allowed.get('lwir_model', '')})")
    print("done")


if __name__ == "__main__":
    main()
