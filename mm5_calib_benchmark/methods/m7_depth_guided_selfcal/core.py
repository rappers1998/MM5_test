from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from scipy import ndimage
from scipy.optimize import minimize
from skimage.segmentation import random_walker

from ...common import ensure_dir, modality_preprocess, mutual_information
from ...pipeline import _read_scene_assets
from ...viz.overlays import mar_like_overlay
from ..alignment import apply_scene_tuning, compute_depth_alignment, compute_homography_alignment
from ..board import build_stereo_dict, evaluate_stereo_on_board


M7_OUTPUT_NAME = "method_M7_depth_guided_selfcal_thermal"


def m7_output_dir(config: dict[str, Any], track: str) -> Path:
    if track != "thermal":
        raise ValueError("M7 is only supported for thermal track")
    return Path(config["outputs"]["root"]) / M7_OUTPUT_NAME


def _rotation_delta_from_degrees(rot_deg: np.ndarray) -> np.ndarray:
    rvec = np.deg2rad(np.asarray(rot_deg, dtype=np.float64).reshape(3))
    delta_r, _ = cv2.Rodrigues(rvec)
    return delta_r


def apply_pose_delta(base_stereo: dict[str, Any], source_size: tuple[int, int], delta: np.ndarray) -> dict[str, Any]:
    delta = np.asarray(delta, dtype=np.float64).reshape(6)
    delta_r = _rotation_delta_from_degrees(delta[:3])
    base_r = np.asarray(base_stereo["R"], dtype=np.float64)
    base_t = np.asarray(base_stereo["T"], dtype=np.float64).reshape(3)
    new_r = delta_r @ base_r
    new_t = base_t + delta[3:]
    return build_stereo_dict(
        np.asarray(base_stereo["CM1"], dtype=np.float64),
        np.asarray(base_stereo["D1"], dtype=np.float64).reshape(-1),
        np.asarray(base_stereo["CM2"], dtype=np.float64),
        np.asarray(base_stereo["D2"], dtype=np.float64).reshape(-1),
        new_r,
        new_t,
        source_size,
    )


def _sample_train_rows(rows: list[Any], train_count: int, seed: int) -> list[Any]:
    eligible = [
        row for row in rows
        if str(row.payload.get("split", "")) == "train"
        and bool(row.payload.get("has_thermal", False))
        and bool(row.payload.get("has_depth", False))
    ]
    if len(eligible) <= train_count:
        return eligible

    groups: dict[str, list[Any]] = {}
    for row in eligible:
        key = str(row.payload.get("category", "unknown"))
        groups.setdefault(key, []).append(row)

    rng = np.random.default_rng(seed)
    for group_rows in groups.values():
        rng.shuffle(group_rows)

    base_alloc: dict[str, int] = {key: min(2, len(group_rows)) for key, group_rows in groups.items()}
    remaining = max(0, train_count - sum(base_alloc.values()))

    leftovers: list[tuple[str, float, int]] = []
    total_available = float(sum(max(0, len(group_rows) - base_alloc[key]) for key, group_rows in groups.items()))
    for key, group_rows in groups.items():
        extra_cap = max(0, len(group_rows) - base_alloc[key])
        if extra_cap <= 0 or remaining <= 0 or total_available <= 0:
            continue
        exact = remaining * (extra_cap / total_available)
        alloc = min(extra_cap, int(np.floor(exact)))
        base_alloc[key] += alloc
        frac = exact - np.floor(exact)
        leftovers.append((key, frac, extra_cap - alloc))

    remaining = max(0, train_count - sum(base_alloc.values()))
    for key, _, cap_left in sorted(leftovers, key=lambda item: (-item[1], item[0])):
        if remaining <= 0:
            break
        if cap_left <= 0:
            continue
        base_alloc[key] += 1
        remaining -= 1

    sampled: list[Any] = []
    for key in sorted(groups.keys()):
        sampled.extend(groups[key][: base_alloc[key]])

    if len(sampled) < train_count:
        used_sequences = {int(row.sequence) for row in sampled}
        for row in eligible:
            if int(row.sequence) in used_sequences:
                continue
            sampled.append(row)
            used_sequences.add(int(row.sequence))
            if len(sampled) == train_count:
                break
    return sampled[:train_count]


def _target_edge_distance(target_image: np.ndarray) -> np.ndarray:
    gray = modality_preprocess(target_image, "thermal")
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    canny = cv2.Canny(blur, 40, 120) > 0
    sx = cv2.Sobel(blur, cv2.CV_32F, 1, 0, ksize=3)
    sy = cv2.Sobel(blur, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(sx, sy)
    if np.any(mag > 0):
        thresh = float(np.percentile(mag[mag > 0], 82))
        sobel = mag >= thresh
    else:
        sobel = np.zeros_like(canny)
    edge = np.logical_or(canny, sobel).astype(np.uint8)
    edge = cv2.morphologyEx(edge * 255, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    return cv2.distanceTransform((edge == 0).astype(np.uint8), cv2.DIST_L2, 3)


def _prepare_score_scene(row: Any, stride: int = 8, max_points: int = 16000) -> dict[str, Any] | None:
    assets = _read_scene_assets(row, "thermal")
    depth = assets.get("depth_image")
    if depth is None:
        return None

    source_gray = modality_preprocess(assets["source_image"], "rgb")
    target_gray = modality_preprocess(assets["target_image"], "thermal")
    edge_dist = _target_edge_distance(assets["target_image"])

    valid_depth = np.isfinite(depth) & (depth > 0)
    yy, xx = np.indices(depth.shape)
    sample_mask = valid_depth.copy()
    sample_mask[(yy % int(max(1, stride)) != 0) | (xx % int(max(1, stride)) != 0)] = False

    if int(sample_mask.sum()) == 0:
        return None

    sample_y = yy[sample_mask].astype(np.int32)
    sample_x = xx[sample_mask].astype(np.int32)
    sample_z = depth[sample_mask].astype(np.float32)
    sample_gray = source_gray[sample_mask].astype(np.uint8)

    if len(sample_z) > max_points:
        keep = np.linspace(0, len(sample_z) - 1, max_points, dtype=np.int32)
        sample_y = sample_y[keep]
        sample_x = sample_x[keep]
        sample_z = sample_z[keep]
        sample_gray = sample_gray[keep]

    return {
        "sequence": int(row.sequence),
        "target_shape": target_gray.shape,
        "target_gray": target_gray,
        "target_edge_dist": edge_dist,
        "sample_x": sample_x,
        "sample_y": sample_y,
        "sample_z": sample_z,
        "sample_gray": sample_gray,
    }


def _project_sparse_scene(scene_payload: dict[str, Any], stereo: dict[str, Any], support_dilate_ksize: int) -> dict[str, float]:
    k1 = np.asarray(stereo["CM1"], dtype=np.float64)
    k2 = np.asarray(stereo["CM2"], dtype=np.float64)
    rotation = np.asarray(stereo["R"], dtype=np.float64)
    translation = np.asarray(stereo["T"], dtype=np.float64).reshape(3)
    target_h, target_w = scene_payload["target_shape"]

    fx1, fy1, cx1, cy1 = k1[0, 0], k1[1, 1], k1[0, 2], k1[1, 2]
    fx2, fy2, cx2, cy2 = k2[0, 0], k2[1, 1], k2[0, 2], k2[1, 2]

    x = (scene_payload["sample_x"] - cx1) * scene_payload["sample_z"] / fx1
    y = (scene_payload["sample_y"] - cy1) * scene_payload["sample_z"] / fy1
    xyz = np.stack([x, y, scene_payload["sample_z"]], axis=1)
    xyz2 = (rotation @ xyz.T).T + translation[None, :]
    z2 = xyz2[:, 2]
    keep = z2 > 1e-6
    if not np.any(keep):
        return {"edge_score": 0.0, "mi_score": 0.0, "projected_ratio": 0.0}

    xyz2 = xyz2[keep]
    z2 = z2[keep]
    gray_vals = scene_payload["sample_gray"][keep]
    u2 = np.round(fx2 * xyz2[:, 0] / z2 + cx2).astype(np.int32)
    v2 = np.round(fy2 * xyz2[:, 1] / z2 + cy2).astype(np.int32)

    in_bounds = (u2 >= 0) & (u2 < target_w) & (v2 >= 0) & (v2 < target_h)
    if not np.any(in_bounds):
        return {"edge_score": 0.0, "mi_score": 0.0, "projected_ratio": 0.0}

    u2 = u2[in_bounds]
    v2 = v2[in_bounds]
    z2 = z2[in_bounds]
    gray_vals = gray_vals[in_bounds]
    flat = v2.astype(np.int64) * int(target_w) + u2.astype(np.int64)
    order = np.argsort(z2)
    first_indices = np.unique(flat[order], return_index=True)[1]
    keep_indices = order[first_indices]

    u_keep = u2[keep_indices]
    v_keep = v2[keep_indices]
    gray_keep = gray_vals[keep_indices]

    real_projected_mask = np.zeros((target_h, target_w), dtype=np.uint8)
    warped_gray = np.zeros((target_h, target_w), dtype=np.uint8)
    real_projected_mask[v_keep, u_keep] = 1
    warped_gray[v_keep, u_keep] = gray_keep

    support_mask = cv2.dilate(
        real_projected_mask,
        np.ones((int(max(1, support_dilate_ksize)), int(max(1, support_dilate_ksize))), np.uint8),
        iterations=1,
    )
    boundary = ((cv2.dilate(support_mask, np.ones((3, 3), np.uint8)) > 0) ^ (cv2.erode(support_mask, np.ones((3, 3), np.uint8)) > 0)).astype(bool)
    if not np.any(boundary):
        edge_score = 0.0
    else:
        mean_boundary_distance = float(scene_payload["target_edge_dist"][boundary].mean())
        edge_score = float(np.exp(-mean_boundary_distance / 6.0))

    mi = mutual_information(scene_payload["target_gray"], warped_gray, real_projected_mask)
    mi_score = float(np.clip(mi / 2.0, 0.0, 1.0))
    projected_ratio = float(real_projected_mask.sum()) / max(float(support_mask.sum()), 1.0)
    return {
        "edge_score": edge_score,
        "mi_score": mi_score,
        "projected_ratio": projected_ratio,
    }


def _candidate_vectors(rot_deg: float, trans_mm: float) -> list[np.ndarray]:
    candidates = [np.zeros(6, dtype=np.float64)]
    for idx in range(6):
        for sign in (-1.0, 1.0):
            vector = np.zeros(6, dtype=np.float64)
            vector[idx] = sign * (rot_deg if idx < 3 else trans_mm)
            candidates.append(vector)
    return candidates


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return np.asarray(value).tolist()
    if isinstance(value, (np.floating, float)):
        return float(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    return value


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _scene_stage_overlay(target_image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    return mar_like_overlay(target_image, mask.astype(np.uint8))


def _nearest_class_assign(original_mask: np.ndarray, fg_binary: np.ndarray) -> np.ndarray:
    if int((original_mask > 0).sum()) == 0:
        return np.zeros_like(original_mask, dtype=np.uint8)
    distances, indices = ndimage.distance_transform_edt(original_mask == 0, return_distances=True, return_indices=True)
    del distances
    nearest_classes = original_mask[indices[0], indices[1]]
    out = np.zeros_like(original_mask, dtype=np.uint8)
    out[fg_binary > 0] = nearest_classes[fg_binary > 0]
    return out


def boundary_snap_refine(mask: np.ndarray, target_image: np.ndarray, band_width_px: int) -> tuple[np.ndarray, dict[str, Any]]:
    working = mask.astype(np.uint8)
    fg = (working > 0).astype(np.uint8)
    if int(fg.sum()) == 0:
        return working, {"used": False, "reason": "empty_mask"}

    band_kernel = np.ones((int(max(3, band_width_px)), int(max(3, band_width_px))), np.uint8)
    band = ((cv2.dilate(fg, band_kernel) > 0) ^ (cv2.erode(fg, band_kernel) > 0)).astype(np.uint8)
    if int(band.sum()) == 0:
        return working, {"used": False, "reason": "empty_band"}

    inner_seed = cv2.erode(fg, np.ones((3, 3), np.uint8), iterations=1)
    outer_seed = (cv2.dilate(fg, np.ones((3, 3), np.uint8), iterations=1) == 0).astype(np.uint8)
    seeds = np.zeros_like(working, dtype=np.int32)
    seeds[outer_seed > 0] = 1
    seeds[inner_seed > 0] = 2
    stable = band == 0
    seeds[stable & (fg == 0)] = 1
    seeds[stable & (fg > 0)] = 2

    if len(np.unique(seeds[seeds > 0])) < 2:
        return working, {"used": False, "reason": "insufficient_seeds"}

    gray = modality_preprocess(target_image, "thermal").astype(np.float32) / 255.0
    try:
        rw = random_walker(gray, seeds, beta=120, mode="cg_j")
    except Exception as exc:
        return working, {"used": False, "reason": f"random_walker_failed:{type(exc).__name__}"}

    candidate_fg = (rw == 2).astype(np.uint8)
    candidate = _nearest_class_assign(working, candidate_fg)

    src_area = float((working > 0).sum())
    dst_area = float((candidate > 0).sum())
    area_ratio = dst_area / max(src_area, 1.0)
    inter = float(np.logical_and(working > 0, candidate > 0).sum())
    union = float(np.logical_or(working > 0, candidate > 0).sum())
    binary_iou = inter / max(union, 1.0)
    cc_src = int(cv2.connectedComponents((working > 0).astype(np.uint8))[0] - 1)
    cc_dst = int(cv2.connectedComponents((candidate > 0).astype(np.uint8))[0] - 1)

    if not (0.92 <= area_ratio <= 1.08) or binary_iou < 0.96 or cc_dst > cc_src + 1:
        return working, {
            "used": False,
            "reason": "guard_revert",
            "area_ratio": area_ratio,
            "binary_iou": binary_iou,
            "components_src": cc_src,
            "components_dst": cc_dst,
        }

    return candidate, {
        "used": True,
        "area_ratio": area_ratio,
        "binary_iou": binary_iou,
        "components_src": cc_src,
        "components_dst": cc_dst,
        "band_pixels": int(band.sum()),
    }


def optimize_global_pose(
    config: dict[str, Any],
    rows: list[Any],
    track: str,
    base_stereo: dict[str, Any],
    board_payload: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    out_dir = m7_output_dir(config, track)
    cache_path = out_dir / "calib" / "pose_refine_cache.json"
    final_stereo_path = out_dir / "calib" / "final_calibration.yml"
    score_csv_path = out_dir / "calib" / "train_refine_scores.csv"

    if cache_path.exists() and final_stereo_path.exists():
        cache_payload = json.loads(cache_path.read_text(encoding="utf-8"))
        return base_stereo, cache_payload

    sampled_rows = _sample_train_rows(
        rows,
        int(config["method"].get("train_refine_scene_count", 48)),
        int(config["runtime"].get("seed", 42)),
    )
    scene_payloads = [payload for payload in (_prepare_score_scene(row) for row in sampled_rows) if payload is not None]
    source_size = tuple(int(v) for v in board_payload.get("source_size", (1280, 720)))
    min_score_gain = float(config["method"].get("quality_gate_min_score_gain", 0.01))
    rot_deg = float(config["method"].get("pose_delta_deg", 2.0))
    trans_mm = float(config["method"].get("pose_delta_mm", 20.0))

    evaluation_rows: list[dict[str, Any]] = []

    def evaluate_delta(delta: np.ndarray, stage: str) -> tuple[float, dict[str, Any], dict[str, Any]]:
        stereo = apply_pose_delta(base_stereo, source_size, delta)
        board_metrics = evaluate_stereo_on_board(stereo, board_payload)
        reproj_rmse = float(board_metrics.get("checkerboard_corner_rmse_px", 0.0))
        reproj_score = float(np.exp(-reproj_rmse / 3.0))
        scene_scores = [_project_sparse_scene(scene_payload, stereo, int(config["method"].get("support_dilate_ksize", 9))) for scene_payload in scene_payloads]
        if scene_scores:
            edge_score = float(np.mean([score["edge_score"] for score in scene_scores]))
            mi_score = float(np.mean([score["mi_score"] for score in scene_scores]))
            projected_ratio = float(np.mean([score["projected_ratio"] for score in scene_scores]))
        else:
            edge_score = 0.0
            mi_score = 0.0
            projected_ratio = 0.0
        total_score = 0.45 * edge_score + 0.25 * mi_score + 0.20 * projected_ratio + 0.10 * reproj_score
        evaluation_rows.append(
            {
                "stage": stage,
                "rx_deg": float(delta[0]),
                "ry_deg": float(delta[1]),
                "rz_deg": float(delta[2]),
                "tx_mm": float(delta[3]),
                "ty_mm": float(delta[4]),
                "tz_mm": float(delta[5]),
                "score": float(total_score),
                "edge_score": edge_score,
                "mi_score": mi_score,
                "projected_ratio": projected_ratio,
                "reproj_score": reproj_score,
                "checkerboard_corner_rmse_px": reproj_rmse,
            }
        )
        summary = {
            "score": float(total_score),
            "edge_score": edge_score,
            "mi_score": mi_score,
            "projected_ratio": projected_ratio,
            "reproj_score": reproj_score,
            "checkerboard_corner_rmse_px": reproj_rmse,
        }
        return total_score, stereo, summary

    coarse_best = {"score": -1e9, "delta": np.zeros(6, dtype=np.float64), "stereo": base_stereo, "summary": {}}
    for candidate in _candidate_vectors(rot_deg, trans_mm):
        score, candidate_stereo, summary = evaluate_delta(candidate, "coarse")
        if score > float(coarse_best["score"]):
            coarse_best = {"score": score, "delta": candidate.copy(), "stereo": candidate_stereo, "summary": summary}

    def objective(delta: np.ndarray) -> float:
        score, _, _ = evaluate_delta(delta, "powell")
        return -score

    result = minimize(
        objective,
        coarse_best["delta"],
        method="Powell",
        bounds=[
            (-rot_deg, rot_deg),
            (-rot_deg, rot_deg),
            (-rot_deg, rot_deg),
            (-trans_mm, trans_mm),
            (-trans_mm, trans_mm),
            (-trans_mm, trans_mm),
        ],
        options={"maxiter": 24, "xtol": 1e-2, "ftol": 1e-3},
    )

    before_score, _, before_summary = evaluate_delta(np.zeros(6, dtype=np.float64), "final_before")
    after_score, refined_stereo, after_summary = evaluate_delta(np.asarray(result.x, dtype=np.float64), "final_after")
    use_refined = bool(after_score - before_score >= min_score_gain)
    final_stereo = refined_stereo if use_refined else base_stereo
    final_summary = after_summary if use_refined else before_summary
    final_delta = np.asarray(result.x, dtype=np.float64) if use_refined else np.zeros(6, dtype=np.float64)

    _write_csv(score_csv_path, evaluation_rows)
    cache_payload = {
        "base_stereo_source": "m1_saved_or_mm5_official",
        "sampled_sequences": [int(row.sequence) for row in sampled_rows],
        "train_scene_count": int(len(scene_payloads)),
        "score_before": float(before_score),
        "score_after": float(after_score),
        "score_gain": float(after_score - before_score),
        "used_refined_pose": use_refined,
        "quality_gate_min_score_gain": min_score_gain,
        "global_pose_delta": {
            "rx_deg": float(final_delta[0]),
            "ry_deg": float(final_delta[1]),
            "rz_deg": float(final_delta[2]),
            "tx_mm": float(final_delta[3]),
            "ty_mm": float(final_delta[4]),
            "tz_mm": float(final_delta[5]),
        },
        "summary_before": before_summary,
        "summary_after": after_summary,
        "summary_final": final_summary,
    }
    ensure_dir(cache_path.parent)
    cache_path.write_text(json.dumps(_json_ready(cache_payload), indent=2), encoding="utf-8")
    return final_stereo, cache_payload


def build_stage_images(assets: dict[str, Any], masks: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {name: _scene_stage_overlay(assets["target_image"], mask) for name, mask in masks.items()}


def compute_m7_scene_result(config: dict[str, Any], row: Any, stereo: dict[str, Any], cache_payload: dict[str, Any] | None = None) -> dict[str, Any]:
    assets = _read_scene_assets(row, "thermal")
    method_cfg = config["method"]
    homography_result = compute_homography_alignment(assets, stereo, float(config["runtime"]["plane_depth_mm"]))

    depth_result = compute_depth_alignment(
        assets,
        stereo,
        float(config["runtime"]["plane_depth_mm"]),
        seed_with_homography=False,
        fill_holes=True,
        fill_distance_px=float(method_cfg.get("hole_fill_max_dist_px", 8)),
        support_dilate_ksize=int(method_cfg.get("support_dilate_ksize", 9)),
        splat_radius=int(method_cfg.get("splat_radius", 1)),
    )

    projected_ratio = float(depth_result.get("debug", {}).get("projected_ratio", 0.0))
    if projected_ratio < float(method_cfg.get("quality_gate_min_projected_ratio", 0.90)):
        pred_mask = depth_result["pre_fill_pred_mask"].astype(np.uint8)
        warped_source = homography_result["warped_source"]
        valid_mask = homography_result["valid_mask"].astype(np.uint8)
        fallback_reason = "projected_ratio_guard"
    else:
        pred_mask = depth_result["pred_mask"].astype(np.uint8)
        warped_source = homography_result["warped_source"]
        valid_mask = homography_result["valid_mask"].astype(np.uint8)
        fallback_reason = ""

    tuned = apply_scene_tuning(
        {
            "pred_mask": pred_mask,
            "warped_source": warped_source,
            "valid_mask": valid_mask,
            "debug": {"alignment": "m7_depth_projection"},
        },
        assets["target_image"],
        "thermal",
        coarse_radius_px=10,
        coarse_step_px=4,
        fine_radius_px=2,
        fine_step_px=1,
        coarse_scales=[0.97, 0.99, 1.0, 1.01, 1.03],
    )

    snapped_mask, snap_info = boundary_snap_refine(tuned["pred_mask"], assets["target_image"], int(method_cfg.get("band_width_px", 7)))
    final_mask = snapped_mask.astype(np.uint8)
    final_source = tuned["warped_source"]
    final_valid = tuned["valid_mask"].astype(np.uint8)

    stage_images = build_stage_images(
        assets,
        {
            "01_depth_projection": depth_result["pre_fill_pred_mask"],
            "02_support_fill": pred_mask,
            "03_scene_tuned": tuned["pred_mask"],
            "04_boundary_snap": final_mask,
        },
    )

    debug = {
        "base_stereo_source": "m1_saved_or_mm5_official",
        "global_pose_delta": dict((cache_payload or {}).get("global_pose_delta", {})),
        "self_supervised_score_before": float((cache_payload or {}).get("score_before", 0.0)),
        "self_supervised_score_after": float((cache_payload or {}).get("score_after", 0.0)),
        "projected_ratio": projected_ratio,
        "hole_fill_ratio": float(depth_result["hole_fill_mask"].sum()) / max(float(depth_result["support_mask"].sum()), 1.0),
        "band_refine_changed_px": int(np.count_nonzero(final_mask != tuned["pred_mask"])),
        "fallback_reason": fallback_reason,
        "scene_tune": tuned["debug"].get("scene_tune", {}),
        "boundary_snap": snap_info,
        "stage_images": stage_images,
    }
    return {
        "assets": assets,
        "pred_mask": final_mask,
        "warped_source": final_source,
        "valid_mask": final_valid,
        "real_projected_mask": depth_result["real_projected_mask"],
        "support_mask": depth_result["support_mask"],
        "hole_fill_mask": depth_result["hole_fill_mask"],
        "debug": debug,
    }
