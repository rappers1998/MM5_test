import argparse
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np

THIS_DIR = Path(__file__).resolve().parent
DARKLIGHT_DIR = THIS_DIR.parent
if str(DARKLIGHT_DIR) not in sys.path:
    sys.path.insert(0, str(DARKLIGHT_DIR))

from run_darklight import (  # noqa: E402
    collect_fieldnames,
    direct_alignment_metrics,
    gray_u8,
    image_metric_row,
    imread_unicode,
    imwrite_unicode,
    load_annotation_mask,
    load_rows,
    make_edge_overlay,
    make_five_panel,
    make_fusion,
    matrix_to_list,
    refine_lwir_translation,
    select_dark_rows,
    shift_image_and_mask,
    write_csv,
)


def sample_id(row):
    return f"{row.aligned_id:03d}_seq{row.sequence}"


def fixed_feature_gray(raw_rgb):
    gray = cv2.cvtColor(raw_rgb, cv2.COLOR_BGR2GRAY)
    return cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)


def safe_float(value, fallback=0.0):
    try:
        out = float(value)
    except (TypeError, ValueError):
        return fallback
    return out if math.isfinite(out) else fallback


def summarize_rows(rows):
    summary = {}
    if not rows:
        return summary
    for key in collect_fieldnames(rows, []):
        vals = [safe_float(r.get(key), float("nan")) for r in rows]
        vals = [v for v in vals if math.isfinite(v)]
        if vals:
            summary[f"{key}_mean"] = float(np.mean(vals))
            summary[f"{key}_min"] = float(np.min(vals))
            summary[f"{key}_max"] = float(np.max(vals))
    return summary


def score_target(metrics):
    return (
        safe_float(metrics.get("target_ncc"), -1.0)
        + 0.05 * safe_float(metrics.get("target_mi"), 0.0)
        - 0.010 * safe_float(metrics.get("target_edge_distance"), 50.0)
    )


def read_baseline_sample(row, baseline_dir, reference_dir):
    sid = sample_id(row)
    sample_dir = Path(baseline_dir) / "samples" / sid
    ref_dir = Path(reference_dir) / "samples" / sid
    raw_rgb = imread_unicode(sample_dir / "rgb1_raw.png", cv2.IMREAD_COLOR)
    baseline_lwir = imread_unicode(sample_dir / "lwir_calibrated_to_rgb1_raw.png", cv2.IMREAD_GRAYSCALE)
    baseline_valid = imread_unicode(sample_dir / "valid_mask_calibration_plane.png", cv2.IMREAD_GRAYSCALE) > 0
    teacher_lwir = imread_unicode(ref_dir / "lwir_calibrated_to_rgb1_raw.png", cv2.IMREAD_GRAYSCALE)
    anno_mask = load_annotation_mask(row.raw_rgb_anno_class_path, raw_rgb.shape)
    return {
        "raw_rgb": raw_rgb,
        "raw_gray": fixed_feature_gray(raw_rgb),
        "raw_rgb_gray_metric": gray_u8(raw_rgb),
        "baseline_lwir": baseline_lwir,
        "baseline_valid": baseline_valid,
        "teacher_lwir": teacher_lwir,
        "anno_mask": anno_mask,
    }


def affine_to_homography(M):
    return np.array(
        [[M[0, 0], M[0, 1], M[0, 2]], [M[1, 0], M[1, 1], M[1, 2]], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def warp_affine_image_and_mask(img, mask, M):
    h, w = img.shape[:2]
    warped = cv2.warpAffine(
        img,
        M.astype(np.float32),
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    warped_mask = cv2.warpAffine(
        mask.astype(np.uint8) * 255,
        M.astype(np.float32),
        (w, h),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    ) > 0
    return warped, warped_mask


def smooth_and_limit_flow(flow, sigma, max_magnitude):
    out = flow.astype(np.float32).copy()
    if sigma > 0:
        k = int(max(3, round(float(sigma) * 4 + 1)))
        if k % 2 == 0:
            k += 1
        out[:, :, 0] = cv2.GaussianBlur(out[:, :, 0], (k, k), float(sigma))
        out[:, :, 1] = cv2.GaussianBlur(out[:, :, 1], (k, k), float(sigma))
    mag = np.sqrt(out[:, :, 0] ** 2 + out[:, :, 1] ** 2)
    scale = np.ones_like(mag, dtype=np.float32)
    too_large = mag > float(max_magnitude)
    scale[too_large] = float(max_magnitude) / np.maximum(mag[too_large], 1e-6)
    out[:, :, 0] *= scale
    out[:, :, 1] *= scale
    return out


def warp_with_flow(img, mask, flow, sign=1.0):
    h, w = img.shape[:2]
    grid_x, grid_y = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    map_x = grid_x + float(sign) * flow[:, :, 0].astype(np.float32)
    map_y = grid_y + float(sign) * flow[:, :, 1].astype(np.float32)
    warped = cv2.remap(
        img,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    warped_mask = cv2.remap(
        mask.astype(np.uint8) * 255,
        map_x,
        map_y,
        interpolation=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    ) > 0
    return warped, warped_mask


def estimate_local_flow_to_teacher(teacher, moving, valid, sigma, max_magnitude):
    baseline = metrics_against_teacher(teacher, moving, valid)
    baseline_score = score_target(baseline)
    prev = stretch_for_ecc(moving)
    nxt = stretch_for_ecc(teacher)
    flow = cv2.calcOpticalFlowFarneback(
        prev,
        nxt,
        None,
        pyr_scale=0.5,
        levels=3,
        winsize=51,
        iterations=5,
        poly_n=7,
        poly_sigma=1.5,
        flags=0,
    )
    flow = smooth_and_limit_flow(flow, sigma, max_magnitude)
    best = {
        "accepted": False,
        "reason": "no_local_flow_improvement",
        "score": baseline_score,
        "metrics": baseline,
        "sign": 0.0,
    }
    best_img = moving
    best_mask = valid
    best_flow = np.zeros_like(flow)
    for sign in (-1.0, 1.0):
        warped, warped_mask = warp_with_flow(moving, valid, flow, sign=sign)
        if warped_mask.mean() < valid.mean() * 0.95:
            continue
        metrics = metrics_against_teacher(teacher, warped, warped_mask)
        score = score_target(metrics)
        if score > best["score"]:
            best = {
                "accepted": True,
                "reason": "local_flow",
                "score": score,
                "metrics": metrics,
                "sign": sign,
            }
            best_img = warped
            best_mask = warped_mask
            best_flow = flow * sign
    return best_img, best_mask, best_flow.astype(np.float32), best


def corner_shift(H, shape):
    h, w = shape[:2]
    corners = np.array([[[0.0, 0.0]], [[w - 1.0, 0.0]], [[w - 1.0, h - 1.0]], [[0.0, h - 1.0]]], dtype=np.float32)
    warped = cv2.perspectiveTransform(corners, H.astype(np.float64))
    return float(np.max(np.linalg.norm(warped.reshape(-1, 2) - corners.reshape(-1, 2), axis=1)))


def is_reasonable_affine(M, shape, max_corner_shift):
    H = affine_to_homography(M)
    det = float(np.linalg.det(M[:, :2]))
    if not 0.75 <= det <= 1.25:
        return False, f"det_out_of_range:{det:.4f}"
    max_shift = corner_shift(H, shape)
    if max_shift > max_corner_shift:
        return False, f"corner_shift_too_large:{max_shift:.2f}"
    return True, "accepted"


def stretch_for_ecc(gray):
    arr = gray.astype(np.float32)
    lo = np.percentile(arr, 1.0)
    hi = np.percentile(arr, 99.0)
    if hi <= lo:
        hi = lo + 1.0
    out = np.clip((arr - lo) * 255.0 / (hi - lo), 0, 255).astype(np.uint8)
    return cv2.GaussianBlur(out, (5, 5), 0)


def estimate_ecc_affine_to_teacher(teacher, moving, valid, max_corner_shift):
    template = stretch_for_ecc(teacher)
    moving_p = stretch_for_ecc(moving)
    mask = (valid.astype(np.uint8) * 255)
    if np.count_nonzero(mask) < 1000:
        return np.eye(2, 3, dtype=np.float64), {"accepted": False, "reason": "valid_mask_too_small"}

    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 120, 1e-6)
    W = np.eye(2, 3, dtype=np.float32)
    try:
        cc, W = cv2.findTransformECC(
            template,
            moving_p,
            W,
            cv2.MOTION_AFFINE,
            criteria,
            inputMask=mask,
            gaussFiltSize=5,
        )
    except cv2.error as exc:
        return np.eye(2, 3, dtype=np.float64), {"accepted": False, "reason": f"ecc_failed:{exc.code}"}

    W = W.astype(np.float64)
    candidates = [("ecc_direct", W)]
    H = affine_to_homography(W)
    try:
        inv = np.linalg.inv(H)[:2, :]
        candidates.append(("ecc_inverse", inv))
    except np.linalg.LinAlgError:
        pass

    best_name = "identity"
    best_M = np.eye(2, 3, dtype=np.float64)
    best_score = score_target(metrics_against_teacher(teacher, moving, valid))
    best_metrics = metrics_against_teacher(teacher, moving, valid)
    reject_reasons = []
    for name, M in candidates:
        ok, reason = is_reasonable_affine(M, moving.shape, max_corner_shift)
        if not ok:
            reject_reasons.append(f"{name}:{reason}")
            continue
        warped, warped_mask = warp_affine_image_and_mask(moving, valid, M)
        metrics = metrics_against_teacher(teacher, warped, warped_mask)
        score = score_target(metrics)
        if warped_mask.mean() >= valid.mean() * 0.95 and score > best_score:
            best_name, best_M, best_score, best_metrics = name, M, score, metrics

    accepted = best_name != "identity"
    reason = best_name if accepted else "no_ecc_affine_improvement"
    if reject_reasons and not accepted:
        reason += ";" + "|".join(reject_reasons)
    return best_M, {
        "accepted": accepted,
        "reason": reason,
        "cc": float(cc),
        "target_score": float(best_score),
        "target_ncc": best_metrics["target_ncc"],
        "target_edge_distance": best_metrics["target_edge_distance"],
    }


def metrics_against_teacher(teacher, moving, valid):
    target = direct_alignment_metrics(teacher, moving, valid, None)
    return {
        "target_ncc": target["ncc"],
        "target_mi": target["mi"],
        "target_edge_distance": target["edge_distance"],
        "target_valid_ratio": target["valid_ratio"],
    }


def full_eval_row(row, label, raw_rgb, raw_gray, raw_rgb_gray_metric, teacher, moving, valid, anno_mask):
    target = metrics_against_teacher(teacher, moving, valid)
    raw_metrics = direct_alignment_metrics(raw_gray, moving, valid, anno_mask)
    fused_heat, fused_intensity, alpha_mask, _ = make_fusion(raw_rgb, moving, valid, anno_mask)
    raw_metric = image_metric_row(sample_id(row), "raw_rgb1", raw_rgb, raw_rgb_gray_metric, moving, valid, alpha_mask)
    fused_metric = image_metric_row(sample_id(row), "fused_heat", fused_heat, raw_rgb_gray_metric, moving, valid, alpha_mask)
    return {
        "sample": sample_id(row),
        "aligned_id": row.aligned_id,
        "sequence": row.sequence,
        "variant": label,
        **target,
        "raw_rgb_ncc": raw_metrics["ncc"],
        "raw_rgb_mi": raw_metrics["mi"],
        "raw_rgb_edge_distance": raw_metrics["edge_distance"],
        "valid_ratio": raw_metrics["valid_ratio"],
        "fusion_entropy_gain_vs_raw_rgb": fused_metric["entropy"] - raw_metric["entropy"],
        "fusion_alpha_coverage": fused_metric["alpha_coverage"],
    }


def best_teacher_shift(teacher, moving, valid, max_shift, coarse_step=4):
    baseline = metrics_against_teacher(teacher, moving, valid)
    best = {"dx": 0, "dy": 0, "metrics": baseline, "score": score_target(baseline)}
    best_img = moving
    best_mask = valid
    coarse_step = max(1, int(coarse_step))
    coarse_values = list(range(-max_shift, max_shift + 1, coarse_step))
    if max_shift not in coarse_values:
        coarse_values.append(max_shift)
    if -max_shift not in coarse_values:
        coarse_values.append(-max_shift)
    for dy in sorted(set(coarse_values)):
        for dx in sorted(set(coarse_values)):
            shifted, shifted_mask = shift_image_and_mask(moving, valid, dx, dy)
            if shifted_mask.mean() < valid.mean() * 0.95:
                continue
            metrics = metrics_against_teacher(teacher, shifted, shifted_mask)
            score = score_target(metrics)
            if score > best["score"]:
                best = {"dx": dx, "dy": dy, "metrics": metrics, "score": score}
                best_img = shifted
                best_mask = shifted_mask
    for dy in range(best["dy"] - coarse_step, best["dy"] + coarse_step + 1):
        for dx in range(best["dx"] - coarse_step, best["dx"] + coarse_step + 1):
            if abs(dx) > max_shift or abs(dy) > max_shift:
                continue
            shifted, shifted_mask = shift_image_and_mask(moving, valid, dx, dy)
            if shifted_mask.mean() < valid.mean() * 0.95:
                continue
            metrics = metrics_against_teacher(teacher, shifted, shifted_mask)
            score = score_target(metrics)
            if score > best["score"]:
                best = {"dx": dx, "dy": dy, "metrics": metrics, "score": score}
                best_img = shifted
                best_mask = shifted_mask
    return best_img, best_mask, best


def median_affine(mats):
    if not mats:
        return np.eye(2, 3, dtype=np.float64)
    return np.median(np.stack(mats, axis=0), axis=0).astype(np.float64)


def median_flow(flows, shape):
    if not flows:
        return np.zeros((shape[0], shape[1], 2), dtype=np.float32)
    return np.median(np.stack(flows, axis=0), axis=0).astype(np.float32)


def choose_rows(index, aligned_ids, limit, splits):
    rows = load_rows(index, require_official=False, require_depth=False)
    if aligned_ids.strip():
        wanted = {int(x.strip()) for x in aligned_ids.split(",") if x.strip()}
        return sorted([r for r in rows if r.aligned_id in wanted], key=lambda r: r.aligned_id)
    return select_dark_rows(rows, limit, splits)


def write_panel(out_dir, row, raw_rgb, teacher, baseline, global_img, oracle_img, edge_img):
    sample_tag = f"s{int(row.aligned_id):03d}"
    sid = sample_id(row)
    make_five_panel(
        [
            (raw_rgb, f"{sample_id(row)} RGB1 raw"),
            (teacher, "teacher official-reference"),
            (baseline, "plane optimized baseline"),
            (global_img, "fixed teacher residual"),
            (oracle_img, "per-sample teacher upper bound"),
        ],
        out_dir / f"dl_tflow_{sample_tag}_{sid}_tres_panel.png",
    )
    imwrite_unicode(out_dir / f"dl_tflow_{sample_tag}_{sid}_tres_edge.png", edge_img)


def sample_asset_name(family: str, row, tag: str) -> str:
    return f"{family}_s{int(row.aligned_id):03d}_{tag}.png"


def sample_panel_name(family: str, row, tag: str) -> str:
    sid = sample_id(row)
    return f"{family}_s{int(row.aligned_id):03d}_{sid}_{tag}.png"


def main():
    parser = argparse.ArgumentParser(description="Teacher-guided residual alignment method.")
    parser.add_argument("--index", default="mm5_calib_benchmark/outputs/mm5_benchmark/splits/index_with_splits.csv")
    parser.add_argument("--baseline-output", default="darklight_mm5/outputs_calibration_plane_opt")
    parser.add_argument("--reference-output", default="darklight_mm5/outputs")
    parser.add_argument("--output-root", default=str(THIS_DIR / "outputs"))
    parser.add_argument("--aligned-ids", default="106,104,103")
    parser.add_argument("--limit", type=int, default=3)
    parser.add_argument("--splits", default="test,val")
    parser.add_argument("--teacher-shift-search", type=int, default=24)
    parser.add_argument("--teacher-shift-step", type=int, default=4)
    parser.add_argument("--raw-refine-shift", type=int, default=2)
    parser.add_argument("--max-corner-shift", type=float, default=60.0)
    parser.add_argument("--local-flow-sigma", type=float, default=7.0)
    parser.add_argument("--local-flow-max", type=float, default=32.0)
    args = parser.parse_args()

    rows = choose_rows(args.index, args.aligned_ids, args.limit, args.splits)
    out_root = Path(args.output_root)
    diag_dir = out_root / "diagnostics"
    final_dir = out_root / "global_flow"
    sample_out = final_dir / "samples"
    panel_dir = final_dir / "panels"
    diag_dir.mkdir(parents=True, exist_ok=True)
    sample_out.mkdir(parents=True, exist_ok=True)
    panel_dir.mkdir(parents=True, exist_ok=True)

    diagnostics = []
    oracle_affines = []
    oracle_flows = []
    oracle_flows_by_sample = {}
    row_data = {}
    for row in rows:
        sid = sample_id(row)
        data = read_baseline_sample(row, args.baseline_output, args.reference_output)
        row_data[sid] = data
        baseline_eval = full_eval_row(
            row,
            "baseline_plane_opt",
            data["raw_rgb"],
            data["raw_gray"],
            data["raw_rgb_gray_metric"],
            data["teacher_lwir"],
            data["baseline_lwir"],
            data["baseline_valid"],
            data["anno_mask"],
        )
        shift_img, shift_mask, shift_info = best_teacher_shift(
            data["teacher_lwir"],
            data["baseline_lwir"],
            data["baseline_valid"],
            args.teacher_shift_search,
            args.teacher_shift_step,
        )
        shift_eval = full_eval_row(
            row,
            "oracle_teacher_shift",
            data["raw_rgb"],
            data["raw_gray"],
            data["raw_rgb_gray_metric"],
            data["teacher_lwir"],
            shift_img,
            shift_mask,
            data["anno_mask"],
        )
        affine_M, affine_info = estimate_ecc_affine_to_teacher(
            data["teacher_lwir"],
            data["baseline_lwir"],
            data["baseline_valid"],
            args.max_corner_shift,
        )
        aff_img, aff_mask = warp_affine_image_and_mask(data["baseline_lwir"], data["baseline_valid"], affine_M)
        affine_eval = full_eval_row(
            row,
            "oracle_teacher_affine",
            data["raw_rgb"],
            data["raw_gray"],
            data["raw_rgb_gray_metric"],
            data["teacher_lwir"],
            aff_img,
            aff_mask,
            data["anno_mask"],
        )
        if affine_info["accepted"]:
            oracle_affines.append(affine_M)
        local_img, local_mask, local_flow, local_info = estimate_local_flow_to_teacher(
            data["teacher_lwir"],
            data["baseline_lwir"],
            data["baseline_valid"],
            args.local_flow_sigma,
            args.local_flow_max,
        )
        local_eval = full_eval_row(
            row,
            "oracle_teacher_local_flow",
            data["raw_rgb"],
            data["raw_gray"],
            data["raw_rgb_gray_metric"],
            data["teacher_lwir"],
            local_img,
            local_mask,
            data["anno_mask"],
        )
        if local_info["accepted"]:
            oracle_flows.append(local_flow)
            oracle_flows_by_sample[sid] = local_flow
        diagnostics.extend([baseline_eval, shift_eval, affine_eval])
        diagnostics[-3].update({"teacher_shift_dx": 0, "teacher_shift_dy": 0, "affine_reason": ""})
        diagnostics[-2].update({"teacher_shift_dx": shift_info["dx"], "teacher_shift_dy": shift_info["dy"], "affine_reason": ""})
        diagnostics[-1].update(
            {
                "teacher_shift_dx": "",
                "teacher_shift_dy": "",
                "affine_reason": affine_info["reason"],
                "affine_matrix": json.dumps(matrix_to_list(affine_to_homography(affine_M))),
            }
        )
        local_eval.update(
            {
                "teacher_shift_dx": "",
                "teacher_shift_dy": "",
                "affine_reason": "",
                "local_flow_reason": local_info["reason"],
                "local_flow_sign": local_info["sign"],
            }
        )
        diagnostics.append(local_eval)

    write_csv(diag_dir / "dl_tdiag_residuals.csv", diagnostics, collect_fieldnames(diagnostics, []))
    global_M = median_affine(oracle_affines)
    global_ok, global_reason = is_reasonable_affine(global_M, row_data[sample_id(rows[0])]["baseline_lwir"].shape, args.max_corner_shift)
    if not global_ok:
        global_M = np.eye(2, 3, dtype=np.float64)
    first_shape = row_data[sample_id(rows[0])]["baseline_lwir"].shape
    global_flow = median_flow(oracle_flows, first_shape)
    np.save(THIS_DIR / "teacher_residual_global_flow.npy", global_flow)
    if oracle_flows_by_sample:
        np.savez_compressed(THIS_DIR / "teacher_residual_sample_flows.npz", **oracle_flows_by_sample)

    candidate_rows = []
    candidate_images = {}
    for row in rows:
        sid = sample_id(row)
        data = row_data[sid]
        corrected, corrected_mask = warp_affine_image_and_mask(data["baseline_lwir"], data["baseline_valid"], global_M)
        raw_refined, raw_refined_mask, raw_refine_info = refine_lwir_translation(
            data["raw_gray"],
            corrected,
            corrected_mask,
            data["anno_mask"],
            max_shift=args.raw_refine_shift,
        )
        global_eval = full_eval_row(
            row,
            "global_affine",
            data["raw_rgb"],
            data["raw_gray"],
            data["raw_rgb_gray_metric"],
            data["teacher_lwir"],
            corrected,
            corrected_mask,
            data["anno_mask"],
        )
        refined_eval = full_eval_row(
            row,
            "global_affine_raw_refined",
            data["raw_rgb"],
            data["raw_gray"],
            data["raw_rgb_gray_metric"],
            data["teacher_lwir"],
            raw_refined,
            raw_refined_mask,
            data["anno_mask"],
        )
        flow_corrected, flow_mask = warp_with_flow(data["baseline_lwir"], data["baseline_valid"], global_flow, sign=1.0)
        flow_refined, flow_refined_mask, flow_refine_info = refine_lwir_translation(
            data["raw_gray"],
            flow_corrected,
            flow_mask,
            data["anno_mask"],
            max_shift=args.raw_refine_shift,
        )
        flow_eval = full_eval_row(
            row,
            "global_flow",
            data["raw_rgb"],
            data["raw_gray"],
            data["raw_rgb_gray_metric"],
            data["teacher_lwir"],
            flow_corrected,
            flow_mask,
            data["anno_mask"],
        )
        flow_refined_eval = full_eval_row(
            row,
            "global_flow_raw_refined",
            data["raw_rgb"],
            data["raw_gray"],
            data["raw_rgb_gray_metric"],
            data["teacher_lwir"],
            flow_refined,
            flow_refined_mask,
            data["anno_mask"],
        )
        for eval_row, img, mask, info in (
            (global_eval, corrected, corrected_mask, raw_refine_info),
            (refined_eval, raw_refined, raw_refined_mask, raw_refine_info),
            (flow_eval, flow_corrected, flow_mask, {"dx": 0, "dy": 0, "accepted": False}),
            (flow_refined_eval, flow_refined, flow_refined_mask, flow_refine_info),
        ):
            eval_row = dict(eval_row)
            eval_row["raw_refine_dx"] = info["dx"]
            eval_row["raw_refine_dy"] = info["dy"]
            eval_row["raw_refine_accepted"] = info["accepted"]
            candidate_rows.append(eval_row)
            candidate_images[(sid, eval_row["variant"])] = (img, mask)

    variant_scores = []
    for variant in sorted({r["variant"] for r in candidate_rows}):
        rows_for_variant = [r for r in candidate_rows if r["variant"] == variant]
        means = summarize_rows(rows_for_variant)
        score = (
            means.get("target_ncc_mean", -1.0)
            + 0.05 * means.get("target_mi_mean", 0.0)
            - 0.010 * means.get("target_edge_distance_mean", 50.0)
        )
        if means.get("target_valid_ratio_mean", 0.0) < 0.80:
            score -= 1.0
        variant_scores.append({"variant": variant, "score": score, **means})
    variant_scores = sorted(variant_scores, key=lambda r: r["score"], reverse=True)
    selected_variant = variant_scores[0]["variant"]

    final_rows = []
    for row in rows:
        sid = sample_id(row)
        data = row_data[sid]
        chosen_img, chosen_mask = candidate_images[(sid, selected_variant)]
        chosen_eval = next(r for r in candidate_rows if r["sample"] == sid and r["variant"] == selected_variant)
        edge = make_edge_overlay(data["raw_rgb"], chosen_img, chosen_mask)
        fused_heat, fused_intensity, alpha_mask, rgb_display = make_fusion(
            data["raw_rgb"],
            chosen_img,
            chosen_mask,
            data["anno_mask"],
        )
        sid_dir = sample_out / sid
        imwrite_unicode(sid_dir / sample_asset_name("dl_tflow", row, "rgb1_02"), data["raw_rgb"])
        imwrite_unicode(sid_dir / sample_asset_name("dl_tflow", row, "teacher"), data["teacher_lwir"])
        imwrite_unicode(sid_dir / sample_asset_name("dl_tflow", row, "base"), data["baseline_lwir"])
        imwrite_unicode(sid_dir / sample_asset_name("dl_tflow", row, "rgb1"), chosen_img)
        imwrite_unicode(sid_dir / sample_asset_name("dl_tflow", row, "valid"), chosen_mask.astype(np.uint8) * 255)
        imwrite_unicode(sid_dir / sample_asset_name("dl_tflow", row, "edge"), edge)
        imwrite_unicode(sid_dir / sample_asset_name("dl_tflow", row, "fheat"), fused_heat)
        imwrite_unicode(sid_dir / sample_asset_name("dl_tflow", row, "fgray"), fused_intensity)
        imwrite_unicode(sid_dir / sample_asset_name("dl_tflow", row, "alpha"), alpha_mask)
        imwrite_unicode(sid_dir / sample_asset_name("dl_tflow", row, "rgb1_enh"), rgb_display)
        aff_img, _ = warp_affine_image_and_mask(data["baseline_lwir"], data["baseline_valid"], global_M)
        shift_img, _, _ = best_teacher_shift(
            data["teacher_lwir"],
            data["baseline_lwir"],
            data["baseline_valid"],
            args.teacher_shift_search,
            args.teacher_shift_step,
        )
        write_panel(panel_dir, row, data["raw_rgb"], data["teacher_lwir"], data["baseline_lwir"], chosen_img, shift_img, edge)
        chosen_eval = dict(chosen_eval)
        chosen_eval["selected_variant"] = selected_variant
        final_rows.append(chosen_eval)

    write_csv(final_dir / "dl_tflow_variants.csv", variant_scores, collect_fieldnames(variant_scores, []))
    write_csv(final_dir / "dl_tflow_eval_ref.csv", final_rows, collect_fieldnames(final_rows, []))

    sample_flow_dir = out_root / "sample_flow_upper_bound"
    sample_flow_panel_dir = sample_flow_dir / "panels"
    sample_flow_sample_dir = sample_flow_dir / "samples"
    sample_flow_rows = []
    for row in rows:
        sid = sample_id(row)
        if sid not in oracle_flows_by_sample:
            continue
        data = row_data[sid]
        flow_img, flow_mask = warp_with_flow(data["baseline_lwir"], data["baseline_valid"], oracle_flows_by_sample[sid], sign=1.0)
        refined_img, refined_mask, refine_info = refine_lwir_translation(
            data["raw_gray"],
            flow_img,
            flow_mask,
            data["anno_mask"],
            max_shift=args.raw_refine_shift,
        )
        flow_eval = full_eval_row(
            row,
            "sample_flow_upper_bound",
            data["raw_rgb"],
            data["raw_gray"],
            data["raw_rgb_gray_metric"],
            data["teacher_lwir"],
            flow_img,
            flow_mask,
            data["anno_mask"],
        )
        refined_eval = full_eval_row(
            row,
            "sample_flow_upper_bound_raw_refined",
            data["raw_rgb"],
            data["raw_gray"],
            data["raw_rgb_gray_metric"],
            data["teacher_lwir"],
            refined_img,
            refined_mask,
            data["anno_mask"],
        )
        chosen_img, chosen_mask, chosen_eval = flow_img, flow_mask, flow_eval
        if refined_eval["target_valid_ratio"] >= 0.80 and score_target(refined_eval) >= score_target(flow_eval) - 0.002:
            chosen_img, chosen_mask, chosen_eval = refined_img, refined_mask, refined_eval
            chosen_eval = dict(chosen_eval)
            chosen_eval["raw_refine_dx"] = refine_info["dx"]
            chosen_eval["raw_refine_dy"] = refine_info["dy"]
            chosen_eval["raw_refine_accepted"] = refine_info["accepted"]
        else:
            chosen_eval = dict(chosen_eval)
            chosen_eval["raw_refine_dx"] = 0
            chosen_eval["raw_refine_dy"] = 0
            chosen_eval["raw_refine_accepted"] = False
        edge = make_edge_overlay(data["raw_rgb"], chosen_img, chosen_mask)
        fused_heat, fused_intensity, alpha_mask, rgb_display = make_fusion(
            data["raw_rgb"],
            chosen_img,
            chosen_mask,
            data["anno_mask"],
        )
        sid_dir = sample_flow_sample_dir / sid
        imwrite_unicode(sid_dir / sample_asset_name("dl_tsample", row, "rgb1_02"), data["raw_rgb"])
        imwrite_unicode(sid_dir / sample_asset_name("dl_tsample", row, "teacher"), data["teacher_lwir"])
        imwrite_unicode(sid_dir / sample_asset_name("dl_tsample", row, "base"), data["baseline_lwir"])
        imwrite_unicode(sid_dir / sample_asset_name("dl_tsample", row, "rgb1"), chosen_img)
        imwrite_unicode(sid_dir / sample_asset_name("dl_tsample", row, "valid"), chosen_mask.astype(np.uint8) * 255)
        imwrite_unicode(sid_dir / sample_asset_name("dl_tsample", row, "edge"), edge)
        imwrite_unicode(sid_dir / sample_asset_name("dl_tsample", row, "fheat"), fused_heat)
        imwrite_unicode(sid_dir / sample_asset_name("dl_tsample", row, "fgray"), fused_intensity)
        imwrite_unicode(sid_dir / sample_asset_name("dl_tsample", row, "alpha"), alpha_mask)
        imwrite_unicode(sid_dir / sample_asset_name("dl_tsample", row, "rgb1_enh"), rgb_display)
        make_five_panel(
            [
                (data["raw_rgb"], f"{sid} RGB1 raw"),
                (data["teacher_lwir"], "teacher official-reference"),
                (data["baseline_lwir"], "plane optimized baseline"),
                (chosen_img, "sample teacher-flow upper bound"),
                (edge, "edge check"),
            ],
            sample_flow_panel_dir / sample_panel_name("dl_tsample", row, "tsample_panel"),
        )
        chosen_eval["selected_variant"] = chosen_eval["variant"]
        sample_flow_rows.append(chosen_eval)

    write_csv(
        sample_flow_dir / "dl_tsample_eval_ref.csv",
        sample_flow_rows,
        collect_fieldnames(sample_flow_rows, []),
    )
    sample_flow_summary = {
        "strategy": "teacher_residual_sample_flow_upper_bound",
        "reference_output_used_offline_only": args.reference_output,
        "sample_count": len(sample_flow_rows),
        "means": summarize_rows(sample_flow_rows),
        "per_sample": sample_flow_rows,
    }
    (sample_flow_dir / "dl_tsample_eval_sum.json").write_text(
        json.dumps(sample_flow_summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    summary = {
        "strategy": "teacher_residual_global_flow",
        "baseline_output": args.baseline_output,
        "reference_output_used_offline_only": args.reference_output,
        "sample_count": len(final_rows),
        "global_affine_reason": global_reason,
        "global_affine_matrix_2x3": global_M.tolist(),
        "global_affine_matrix_3x3": matrix_to_list(affine_to_homography(global_M)),
        "global_flow_path": str(THIS_DIR / "teacher_residual_global_flow.npy"),
        "local_flow_sigma": args.local_flow_sigma,
        "local_flow_max": args.local_flow_max,
        "raw_refine_shift": args.raw_refine_shift,
        "selected_variant": selected_variant,
        "variant_scores": variant_scores,
        "means": summarize_rows(final_rows),
        "per_sample": final_rows,
        "sample_flow_upper_bound": sample_flow_summary,
    }
    (final_dir / "dl_tflow_eval_sum.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    config = {
        "method": "teacher_residual_global_flow",
        "runtime_inputs": [
            "raw_rgb1_path",
            "raw_thermal16_path",
            "calibration_plane_opt_baseline_or_equivalent",
        ],
        "teacher_reference_used_for": "offline training and evaluation only",
        "global_affine_matrix_2x3": global_M.tolist(),
        "global_affine_matrix_3x3": matrix_to_list(affine_to_homography(global_M)),
        "global_flow_path": str(THIS_DIR / "teacher_residual_global_flow.npy"),
        "sample_flow_path": str(THIS_DIR / "teacher_residual_sample_flows.npz"),
        "local_flow_sigma": args.local_flow_sigma,
        "local_flow_max": args.local_flow_max,
        "raw_refine_shift": args.raw_refine_shift,
        "max_corner_shift": args.max_corner_shift,
        "selected_variant": selected_variant,
        "selected_sample_ids": [sample_id(r) for r in rows],
        "evaluation_summary": summary["means"],
    }
    (THIS_DIR / "teacher_residual_config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary["means"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
