import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass
class SampleRow:
    aligned_id: int
    sequence: int
    split: str
    category: str
    subcategory: str
    raw_rgb1_path: str
    raw_rgb3_path: str
    raw_thermal16_path: str
    raw_depth_tr_path: str
    aligned_rgb1_path: str
    aligned_t16_path: str
    raw_rgb_anno_class_path: str
    dark_mean: float = 0.0


@dataclass
class StereoCalibration:
    path: str
    rgb_K: np.ndarray
    rgb_D: np.ndarray
    lwir_K: np.ndarray
    lwir_D: np.ndarray
    rgb_to_lwir_R: np.ndarray
    rgb_to_lwir_T: np.ndarray


def parse_size_arg(text):
    if text is None:
        return None
    value = str(text).strip().lower()
    if not value or value in {"none", "raw", "off"}:
        return None
    for sep in ("x", ","):
        if sep in value:
            w, h = value.split(sep, 1)
            return int(w), int(h)
    raise ValueError(f"size must be WxH, raw, or none: {text}")


def parse_float_tuple(text, count):
    if text is None:
        return tuple(0.0 for _ in range(count))
    parts = [p.strip() for p in str(text).split(",") if p.strip()]
    if len(parts) != count:
        raise ValueError(f"expected {count} comma-separated values: {text}")
    return tuple(float(p) for p in parts)


def scale_camera_matrix(K, from_size, to_size):
    if from_size is None:
        return K.copy()
    from_w, from_h = from_size
    to_w, to_h = to_size
    sx = float(to_w) / float(from_w)
    sy = float(to_h) / float(from_h)
    out = K.copy()
    out[0, 0] *= sx
    out[0, 1] *= sx
    out[0, 2] *= sx
    out[1, 1] *= sy
    out[1, 2] *= sy
    return out


def adjusted_stereo_calibration(
    calibration,
    rgb_shape,
    lwir_shape,
    rgb_calib_size=None,
    lwir_calib_size=None,
    lwir_principal_offset=(0.0, 0.0),
    t_scale=1.0,
    t_offset=(0.0, 0.0, 0.0),
):
    rgb_size = (rgb_shape[1], rgb_shape[0])
    lwir_size = (lwir_shape[1], lwir_shape[0])
    rgb_K = scale_camera_matrix(calibration.rgb_K, rgb_calib_size, rgb_size)
    lwir_K = scale_camera_matrix(calibration.lwir_K, lwir_calib_size, lwir_size)
    lwir_K[0, 2] += float(lwir_principal_offset[0])
    lwir_K[1, 2] += float(lwir_principal_offset[1])
    T = calibration.rgb_to_lwir_T.astype(np.float64) * float(t_scale)
    T = T + np.array(t_offset, dtype=np.float64).reshape(3)
    return StereoCalibration(
        path=calibration.path,
        rgb_K=rgb_K,
        rgb_D=calibration.rgb_D.copy(),
        lwir_K=lwir_K,
        lwir_D=calibration.lwir_D.copy(),
        rgb_to_lwir_R=calibration.rgb_to_lwir_R.copy(),
        rgb_to_lwir_T=T,
    )


def imread_unicode(path, flags=cv2.IMREAD_UNCHANGED):
    data = np.fromfile(str(path), dtype=np.uint8)
    if data.size == 0:
        raise FileNotFoundError(path)
    img = cv2.imdecode(data, flags)
    if img is None:
        raise ValueError(f"failed to read image: {path}")
    return img


def imwrite_unicode(path, img):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ext = path.suffix if path.suffix else ".png"
    ok, buf = cv2.imencode(ext, img)
    if not ok:
        raise ValueError(f"failed to encode image: {path}")
    buf.tofile(str(path))


def _read_yaml_node_array(fs, key):
    node = fs.getNode(key)
    if node.empty():
        raise KeyError(f"missing calibration key: {key}")
    if node.isSeq():
        return np.array([node.at(i).real() for i in range(node.size())], dtype=np.float64)
    mat = node.mat()
    if mat is not None:
        return mat.astype(np.float64)
    raise ValueError(f"unsupported calibration node: {key}")


def load_stereo_calibration(path):
    path = Path(path)
    fs = cv2.FileStorage(str(path), cv2.FILE_STORAGE_READ)
    if not fs.isOpened():
        raise FileNotFoundError(path)
    try:
        rgb_K = _read_yaml_node_array(fs, "CM1").reshape(3, 3)
        lwir_K = _read_yaml_node_array(fs, "CM2").reshape(3, 3)
        rgb_D = _read_yaml_node_array(fs, "D1").reshape(-1, 1)
        lwir_D = _read_yaml_node_array(fs, "D2").reshape(-1, 1)
        R = _read_yaml_node_array(fs, "R").reshape(3, 3)
        T = _read_yaml_node_array(fs, "T").reshape(3)
    finally:
        fs.release()
    return StereoCalibration(
        path=str(path),
        rgb_K=rgb_K,
        rgb_D=rgb_D,
        lwir_K=lwir_K,
        lwir_D=lwir_D,
        rgb_to_lwir_R=R,
        rgb_to_lwir_T=T,
    )


def normalize_u8(img, low=1.0, high=99.0):
    if img.ndim == 3:
        if img.dtype == np.uint8:
            return img.copy()
        channels = [normalize_u8(img[:, :, c], low, high) for c in range(img.shape[2])]
        return cv2.merge(channels)
    if img.dtype == np.uint8:
        return img.copy()
    arr = img.astype(np.float32)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return np.zeros(img.shape, dtype=np.uint8)
    lo = np.percentile(finite, low)
    hi = np.percentile(finite, high)
    if hi <= lo:
        hi = lo + 1.0
    out = (arr - lo) / (hi - lo)
    return np.clip(out * 255.0, 0, 255).astype(np.uint8)


def gray_u8(img):
    img8 = normalize_u8(img)
    if img8.ndim == 2:
        return img8
    return cv2.cvtColor(img8, cv2.COLOR_BGR2GRAY)


def stretch_for_features(gray):
    arr = gray.astype(np.float32)
    lo = np.percentile(arr, 0.5)
    hi = np.percentile(arr, 99.5)
    if hi <= lo:
        hi = lo + 1.0
    out = np.clip((arr - lo) * 255.0 / (hi - lo), 0, 255).astype(np.uint8)
    out = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(out)
    return cv2.GaussianBlur(out, (3, 3), 0)


def enhance_lowlight_bgr(bgr):
    img = bgr.astype(np.float32) / 255.0
    p99 = max(float(np.percentile(img, 99.5)), 1.0 / 255.0)
    gain = min(8.0, max(1.0, 0.72 / p99))
    img = np.clip(img * gain, 0, 1)
    img = np.power(img, 0.72)
    lab = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(clipLimit=1.8, tileGridSize=(8, 8)).apply(l)
    return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)


def fit_cover_homography(moving_shape, fixed_shape):
    mh, mw = moving_shape[:2]
    fh, fw = fixed_shape[:2]
    scale = max(fw / float(mw), fh / float(mh))
    tx = (fw - mw * scale) * 0.5
    ty = (fh - mh * scale) * 0.5
    return np.array([[scale, 0.0, tx], [0.0, scale, ty], [0.0, 0.0, 1.0]], dtype=np.float64)


def safe_inv(H):
    try:
        inv = np.linalg.inv(H)
    except np.linalg.LinAlgError:
        inv = np.eye(3, dtype=np.float64)
    return inv


def warp_image(img, H, size, interp=cv2.INTER_LINEAR, border=0):
    return cv2.warpPerspective(
        img,
        H.astype(np.float64),
        size,
        flags=interp,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=border,
    )


def warp_mask(shape, H, size):
    mask = np.ones(shape[:2], dtype=np.uint8) * 255
    return warp_image(mask, H, size, interp=cv2.INTER_NEAREST, border=0) > 0


def auto_edges(gray):
    g = stretch_for_features(gray)
    med = float(np.median(g))
    lo = int(max(5, 0.66 * med))
    hi = int(min(255, max(lo + 20, 1.33 * med)))
    edges = cv2.Canny(g, lo, hi)
    return cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))


def masked_ncc(a, b, mask):
    m = mask.astype(bool)
    if m.sum() < 200:
        return float("nan")
    x = stretch_for_features(a)[m].astype(np.float32)
    y = stretch_for_features(b)[m].astype(np.float32)
    x -= x.mean()
    y -= y.mean()
    denom = float(np.sqrt(np.sum(x * x) * np.sum(y * y)))
    if denom < 1e-6:
        return float("nan")
    return float(np.sum(x * y) / denom)


def edge_distance_score(fixed_gray, warped_gray, mask):
    fixed_edges = auto_edges(fixed_gray)
    moving_edges = auto_edges(warped_gray)
    moving_edges = np.where(mask, moving_edges, 0).astype(np.uint8)
    if np.count_nonzero(moving_edges) < 50:
        return float("nan")
    dt = cv2.distanceTransform(255 - fixed_edges, cv2.DIST_L2, 3)
    vals = dt[moving_edges > 0]
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return float("nan")
    return float(np.mean(np.clip(vals, 0, 50)))


def alignment_metrics(fixed_gray, moving_gray, H):
    h, w = fixed_gray.shape[:2]
    warped = warp_image(moving_gray, H, (w, h), interp=cv2.INTER_LINEAR)
    valid = warp_mask(moving_gray.shape, H, (w, h))
    fixed_edges = auto_edges(fixed_gray)
    content = cv2.dilate(fixed_edges, np.ones((7, 7), np.uint8), iterations=1) > 0
    bright = fixed_gray > max(3, np.percentile(fixed_gray, 70))
    score_mask = valid & (content | bright)
    if score_mask.sum() < 300:
        score_mask = valid
    ncc = masked_ncc(fixed_gray, warped, score_mask)
    edge_dist = edge_distance_score(fixed_gray, warped, valid)
    return {
        "ncc": ncc,
        "edge_distance": edge_dist,
        "valid_ratio": float(valid.mean()),
        "score": score_value({"ncc": ncc, "edge_distance": edge_dist}),
        "warped": warped,
        "valid": valid,
    }


def direct_alignment_metrics(fixed_gray, moving_on_fixed, valid_mask, roi_mask=None):
    valid = valid_mask.astype(bool)
    if roi_mask is not None and np.count_nonzero(valid & roi_mask) > 1000:
        score_mask = valid & roi_mask.astype(bool)
    else:
        fixed_edges = auto_edges(fixed_gray)
        content = cv2.dilate(fixed_edges, np.ones((7, 7), np.uint8), iterations=1) > 0
        bright = fixed_gray > max(3, np.percentile(fixed_gray, 70))
        score_mask = valid & (content | bright)
    if np.count_nonzero(score_mask) < 300:
        score_mask = valid
    ncc = masked_ncc(fixed_gray, moving_on_fixed, score_mask)
    edge_dist = edge_distance_score(fixed_gray, moving_on_fixed, score_mask)
    mi = mutual_information(fixed_gray, moving_on_fixed, score_mask)
    ncc_for_score = ncc if np.isfinite(ncc) else -1.0
    edge_for_score = edge_dist if np.isfinite(edge_dist) else 50.0
    mi_for_score = mi if np.isfinite(mi) else 0.0
    return {
        "ncc": ncc,
        "edge_distance": edge_dist,
        "mi": mi,
        "valid_ratio": float(valid.mean()),
        "score": float(ncc_for_score + 0.15 * mi_for_score - 0.006 * edge_for_score),
    }


def score_value(metrics):
    ncc = metrics.get("ncc")
    edge = metrics.get("edge_distance")
    if not np.isfinite(ncc):
        ncc = -1.0
    if not np.isfinite(edge):
        edge = 50.0
    return float(ncc - 0.006 * edge)


def metrics_for_csv(metrics):
    return {
        "ncc": float(metrics.get("ncc", float("nan"))),
        "edge_distance": float(metrics.get("edge_distance", float("nan"))),
        "valid_ratio": float(metrics.get("valid_ratio", float("nan"))),
        "score": float(metrics.get("score", score_value(metrics))),
    }


def direct_metrics_for_csv(metrics):
    row = metrics_for_csv(metrics)
    row["mi"] = float(metrics.get("mi", float("nan")))
    return row


def image_entropy(gray, mask=None, bins=256):
    data = gray_u8(gray)
    if mask is not None:
        data = data[mask.astype(bool)]
    else:
        data = data.reshape(-1)
    if data.size == 0:
        return float("nan")
    hist = np.bincount(data.astype(np.uint8), minlength=bins).astype(np.float64)
    prob = hist / max(hist.sum(), 1.0)
    prob = prob[prob > 0]
    return float(-np.sum(prob * np.log2(prob)))


def average_gradient(gray, mask=None):
    data = gray_u8(gray).astype(np.float32)
    gx = cv2.Sobel(data, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(data, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx * gx + gy * gy)
    if mask is not None:
        vals = mag[mask.astype(bool)]
    else:
        vals = mag.reshape(-1)
    if vals.size == 0:
        return float("nan")
    return float(np.mean(vals))


def spatial_frequency(gray, mask=None):
    data = gray_u8(gray).astype(np.float32)
    rf = np.zeros_like(data, dtype=np.float32)
    cf = np.zeros_like(data, dtype=np.float32)
    rf[1:, :] = data[1:, :] - data[:-1, :]
    cf[:, 1:] = data[:, 1:] - data[:, :-1]
    sf = np.sqrt(rf * rf + cf * cf)
    if mask is not None:
        vals = sf[mask.astype(bool)]
    else:
        vals = sf.reshape(-1)
    if vals.size == 0:
        return float("nan")
    return float(np.sqrt(np.mean(vals * vals)))


def mutual_information(a, b, mask=None, bins=64):
    ag = gray_u8(a)
    bg = gray_u8(b)
    if ag.shape != bg.shape:
        raise ValueError("mutual_information inputs must have the same shape")
    if mask is not None:
        m = mask.astype(bool)
        av = ag[m].reshape(-1)
        bv = bg[m].reshape(-1)
    else:
        av = ag.reshape(-1)
        bv = bg.reshape(-1)
    if av.size < 10:
        return float("nan")
    hist, _, _ = np.histogram2d(av, bv, bins=bins, range=[[0, 255], [0, 255]])
    total = hist.sum()
    if total <= 0:
        return float("nan")
    pxy = hist / total
    px = pxy.sum(axis=1)
    py = pxy.sum(axis=0)
    nz = pxy > 0
    denom = px[:, None] * py[None, :]
    return float(np.sum(pxy[nz] * np.log(pxy[nz] / denom[nz])))


def image_metric_row(sample_key, image_name, image_bgr_or_gray, raw_rgb_gray, lwir_gray, valid_mask, alpha_mask):
    gray = gray_u8(image_bgr_or_gray)
    valid = valid_mask.astype(bool)
    alpha = alpha_mask.astype(np.float32) / 255.0
    return {
        "sample": sample_key,
        "image": image_name,
        "entropy": image_entropy(gray),
        "average_gradient": average_gradient(gray),
        "spatial_frequency": spatial_frequency(gray),
        "std": float(np.std(gray.astype(np.float32))),
        "mean_brightness": float(np.mean(gray.astype(np.float32))),
        "mi_with_raw_rgb": mutual_information(gray, raw_rgb_gray),
        "mi_with_calibrated_lwir": mutual_information(gray, lwir_gray, valid),
        "ncc_with_raw_rgb": masked_ncc(gray, raw_rgb_gray, np.ones(gray.shape, dtype=bool)),
        "ncc_with_calibrated_lwir": masked_ncc(gray, lwir_gray, valid),
        "mean_abs_change_vs_raw_rgb": float(np.mean(np.abs(gray.astype(np.float32) - raw_rgb_gray.astype(np.float32)))),
        "mean_abs_change_on_alpha": float(
            np.mean(np.abs(gray.astype(np.float32) - raw_rgb_gray.astype(np.float32))[alpha > 0.01])
        )
        if np.count_nonzero(alpha > 0.01) > 0
        else 0.0,
        "alpha_mean": float(np.mean(alpha)),
        "alpha_coverage": float(np.mean(alpha > 0.01)),
        "valid_ratio": float(np.mean(valid)),
    }


def sift_homography(fixed_gray, moving_gray):
    if not hasattr(cv2, "SIFT_create"):
        return None, {"reason": "sift_unavailable", "matches": 0, "inliers": 0}
    fixed_p = stretch_for_features(fixed_gray)
    moving_p = stretch_for_features(moving_gray)
    sift = cv2.SIFT_create(nfeatures=5000, contrastThreshold=0.005, edgeThreshold=12)
    kpf, desf = sift.detectAndCompute(fixed_p, None)
    kpm, desm = sift.detectAndCompute(moving_p, None)
    if desf is None or desm is None or len(kpf) < 8 or len(kpm) < 8:
        return None, {
            "reason": "not_enough_keypoints",
            "matches": 0,
            "inliers": 0,
            "fixed_kp": len(kpf),
            "moving_kp": len(kpm),
        }
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    knn = matcher.knnMatch(desm, desf, k=2)
    good = []
    for pair in knn:
        if len(pair) != 2:
            continue
        m, n = pair
        if m.distance < 0.78 * n.distance:
            good.append(m)
    if len(good) < 8:
        return None, {"reason": "not_enough_matches", "matches": len(good), "inliers": 0}
    pts_m = np.float32([kpm[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    pts_f = np.float32([kpf[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    H, inliers = cv2.findHomography(pts_m, pts_f, cv2.RANSAC, 4.0)
    if H is None or inliers is None:
        return None, {"reason": "homography_failed", "matches": len(good), "inliers": 0}
    inlier_count = int(inliers.ravel().sum())
    if inlier_count < 8:
        return None, {"reason": "too_few_inliers", "matches": len(good), "inliers": inlier_count}
    return H.astype(np.float64), {
        "reason": "sift_homography",
        "matches": len(good),
        "inliers": inlier_count,
    }


def refine_residual_ecc(fixed_gray, moving_gray, H):
    h, w = fixed_gray.shape[:2]
    current = warp_image(moving_gray, H, (w, h), interp=cv2.INTER_LINEAR)
    valid = warp_mask(moving_gray.shape, H, (w, h)).astype(np.uint8) * 255
    fixed_p = stretch_for_features(fixed_gray)
    current_p = stretch_for_features(current)
    before = alignment_metrics(fixed_gray, moving_gray, H)
    W = np.eye(2, 3, dtype=np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 80, 1e-6)
    try:
        cc, W = cv2.findTransformECC(
            fixed_p,
            current_p,
            W,
            cv2.MOTION_AFFINE,
            criteria,
            inputMask=valid,
            gaussFiltSize=5,
        )
    except cv2.error as exc:
        return H, {"accepted": False, "reason": f"ecc_failed:{exc.code}"}
    W3 = np.array(
        [[W[0, 0], W[0, 1], W[0, 2]], [W[1, 0], W[1, 1], W[1, 2]], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    candidate = safe_inv(W3) @ H
    after = alignment_metrics(fixed_gray, moving_gray, candidate)
    if score_value(after) >= score_value(before) - 0.002:
        return candidate, {"accepted": True, "reason": "ecc_residual", "cc": float(cc)}
    return H, {"accepted": False, "reason": "ecc_guard_rejected", "cc": float(cc)}


def estimate_same_modality_transform(fixed_img, moving_img):
    fixed_gray = gray_u8(fixed_img)
    moving_gray = gray_u8(moving_img)
    stage_rows = []
    H_fit = fit_cover_homography(moving_gray.shape, fixed_gray.shape)
    fit_metrics = alignment_metrics(fixed_gray, moving_gray, H_fit)
    stage_rows.append({"stage": "fit_cover", **metrics_for_csv(fit_metrics)})
    candidates = [("fit_cover", H_fit, {"reason": "fit_cover"}, fit_metrics)]
    H_sift, sift_info = sift_homography(fixed_gray, moving_gray)
    if H_sift is not None:
        sift_metrics = alignment_metrics(fixed_gray, moving_gray, H_sift)
        stage_rows.append({"stage": "sift_ransac", **metrics_for_csv(sift_metrics)})
        candidates.append(("sift", H_sift, sift_info, sift_metrics))
    else:
        stage_rows.append({"stage": "sift_ransac_failed", "failure_reason": sift_info["reason"]})
    name, H, info, metrics = max(candidates, key=lambda item: score_value(item[3]))
    H_refined, ecc_info = refine_residual_ecc(fixed_gray, moving_gray, H)
    final_metrics = alignment_metrics(fixed_gray, moving_gray, H_refined)
    stage_rows.append(
        {
            "stage": "ecc_final",
            "initial_used": name,
            "ecc_reason": ecc_info["reason"],
            **metrics_for_csv(final_metrics),
        }
    )
    info = {
        "initial": name,
        "sift": sift_info,
        "ecc": ecc_info,
        "ncc": final_metrics["ncc"],
        "edge_distance": final_metrics["edge_distance"],
        "valid_ratio": final_metrics["valid_ratio"],
        "score": final_metrics["score"],
    }
    return H_refined, final_metrics, info, stage_rows


def load_rows(index_path, require_official=True, require_depth=False):
    rows = []
    with Path(index_path).open("r", encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            thermal = r.get("raw_thermal16_path") or r.get("raw_thermal_path")
            bright_rgb = r.get("raw_rgb3_path") or r.get("raw_rgb1_path")
            paths = [
                r.get("raw_rgb1_path"),
                bright_rgb,
                thermal,
            ]
            if require_official:
                paths.extend([r.get("aligned_rgb1_path"), r.get("aligned_t16_path")])
            if require_depth:
                paths.append(r.get("raw_depth_tr_path"))
            if not all(paths) or not all(Path(p).exists() for p in paths):
                continue
            raw_rgb = imread_unicode(r["raw_rgb1_path"], cv2.IMREAD_COLOR)
            mean = float(cv2.cvtColor(raw_rgb, cv2.COLOR_BGR2GRAY).mean())
            rows.append(
                SampleRow(
                    aligned_id=int(r["aligned_id"]),
                    sequence=int(r["sequence"]),
                    split=r.get("split", ""),
                    category=r.get("category", ""),
                    subcategory=r.get("subcategory", ""),
                    raw_rgb1_path=r["raw_rgb1_path"],
                    raw_rgb3_path=bright_rgb,
                    raw_thermal16_path=thermal,
                    raw_depth_tr_path=r.get("raw_depth_tr_path", ""),
                    aligned_rgb1_path=r.get("aligned_rgb1_path", ""),
                    aligned_t16_path=r.get("aligned_t16_path", ""),
                    raw_rgb_anno_class_path=r.get("raw_rgb_anno_class_path", ""),
                    dark_mean=mean,
                )
            )
    return rows


def select_dark_rows(rows, limit, splits):
    allowed = {s.strip() for s in splits.split(",") if s.strip()}
    if allowed and "all" not in allowed:
        rows = [r for r in rows if r.split in allowed]
    rows = sorted(rows, key=lambda r: r.dark_mean)
    return rows[:limit]


def load_annotation_mask(path, shape):
    if not path or not Path(path).exists():
        return None
    mask = imread_unicode(path, cv2.IMREAD_GRAYSCALE)
    if mask.shape[:2] != shape[:2]:
        mask = cv2.resize(mask, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
    mask = mask > 0
    if mask.mean() < 0.0005:
        return None
    kernel = np.ones((19, 19), np.uint8)
    return cv2.dilate(mask.astype(np.uint8) * 255, kernel, iterations=1) > 0


def shift_image_and_mask(img, mask, dx, dy):
    h, w = img.shape[:2]
    M = np.array([[1.0, 0.0, float(dx)], [0.0, 1.0, float(dy)]], dtype=np.float32)
    shifted_img = cv2.warpAffine(
        img,
        M,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    shifted_mask = (
        cv2.warpAffine(
            mask.astype(np.uint8) * 255,
            M,
            (w, h),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        > 0
    )
    return shifted_img, shifted_mask


def refine_lwir_translation(fixed_gray, lwir_on_rgb, valid_mask, roi_mask=None, max_shift=10):
    baseline = direct_alignment_metrics(fixed_gray, lwir_on_rgb, valid_mask, roi_mask)
    best_img = lwir_on_rgb
    best_mask = valid_mask
    best_metrics = baseline
    best_dx = 0
    best_dy = 0
    coarse_values = range(-int(max_shift), int(max_shift) + 1, 2)
    for dy in coarse_values:
        for dx in coarse_values:
            shifted, shifted_mask = shift_image_and_mask(lwir_on_rgb, valid_mask, dx, dy)
            if shifted_mask.mean() < valid_mask.mean() * 0.95:
                continue
            metrics = direct_alignment_metrics(fixed_gray, shifted, shifted_mask, roi_mask)
            if metrics["score"] > best_metrics["score"]:
                best_img, best_mask, best_metrics = shifted, shifted_mask, metrics
                best_dx, best_dy = dx, dy
    for dy in range(best_dy - 2, best_dy + 3):
        for dx in range(best_dx - 2, best_dx + 3):
            if abs(dx) > max_shift or abs(dy) > max_shift:
                continue
            shifted, shifted_mask = shift_image_and_mask(lwir_on_rgb, valid_mask, dx, dy)
            if shifted_mask.mean() < valid_mask.mean() * 0.95:
                continue
            metrics = direct_alignment_metrics(fixed_gray, shifted, shifted_mask, roi_mask)
            if metrics["score"] > best_metrics["score"]:
                best_img, best_mask, best_metrics = shifted, shifted_mask, metrics
                best_dx, best_dy = dx, dy

    edge_before = baseline.get("edge_distance", float("nan"))
    edge_after = best_metrics.get("edge_distance", float("nan"))
    score_gain = best_metrics["score"] - baseline["score"]
    edge_gain = edge_before - edge_after if np.isfinite(edge_before) and np.isfinite(edge_after) else 0.0
    accepted = (best_dx != 0 or best_dy != 0) and (score_gain > 0.01 or edge_gain > 0.3)
    if not accepted:
        best_img = lwir_on_rgb
        best_mask = valid_mask
        best_metrics = baseline
        best_dx = 0
        best_dy = 0
    return best_img, best_mask, {
        "accepted": bool(accepted),
        "dx": int(best_dx),
        "dy": int(best_dy),
        "baseline": baseline,
        "final": best_metrics,
        "score_gain": float(score_gain if accepted else 0.0),
        "edge_distance_gain": float(edge_gain if accepted else 0.0),
    }


def make_fusion(
    raw_rgb_bgr,
    lwir_on_rgb_u8,
    valid_mask,
    anno_mask,
    saliency_sigma=17.0,
    alpha_low_percentile=45.0,
    alpha_high_percentile=97.0,
    alpha_scale=0.68,
    alpha_max=0.68,
    roi_dilate_px=0,
):
    base = enhance_lowlight_bgr(raw_rgb_bgr)
    thermal = lwir_on_rgb_u8.astype(np.float32) / 255.0
    blur = cv2.GaussianBlur(thermal, (0, 0), sigmaX=max(float(saliency_sigma), 0.1))
    sal = np.abs(thermal - blur)
    roi = valid_mask.copy()
    if anno_mask is not None:
        roi &= anno_mask
    if roi_dilate_px > 0:
        k = int(roi_dilate_px) * 2 + 1
        kernel = np.ones((k, k), np.uint8)
        roi = cv2.dilate(roi.astype(np.uint8) * 255, kernel, iterations=1) > 0
        roi &= valid_mask
    if roi.sum() < 500:
        roi = valid_mask
    vals = sal[roi]
    if vals.size < 100:
        alpha = np.zeros_like(thermal, dtype=np.float32)
    else:
        lo = float(np.percentile(vals, float(alpha_low_percentile)))
        hi = float(np.percentile(vals, float(alpha_high_percentile)))
        if hi <= lo:
            hi = lo + 1e-3
        alpha = np.clip((sal - lo) / (hi - lo), 0, 1)
    alpha *= roi.astype(np.float32)
    alpha = cv2.GaussianBlur(alpha, (0, 0), sigmaX=3)
    alpha = np.clip(alpha * float(alpha_scale), 0, float(alpha_max))
    heat = cv2.applyColorMap(lwir_on_rgb_u8, cv2.COLORMAP_TURBO)
    fused_heat = base.astype(np.float32) * (1 - alpha[..., None]) + heat.astype(np.float32) * alpha[..., None]
    hsv = cv2.cvtColor(base, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] + 85.0 * alpha, 0, 255)
    fused_intensity = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    return fused_heat.astype(np.uint8), fused_intensity, (alpha * 255).astype(np.uint8), base


def put_label(canvas, label):
    cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 30), (18, 18, 18), -1)
    cv2.putText(
        canvas,
        label,
        (10, 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        (245, 245, 245),
        1,
        cv2.LINE_AA,
    )


def letterbox(img, tile_size, label):
    tw, th = tile_size
    tile = np.full((th, tw, 3), 18, dtype=np.uint8)
    if img.ndim == 2:
        src = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    else:
        src = img
    body_h = th - 30
    h, w = src.shape[:2]
    scale = min(tw / float(w), body_h / float(h))
    nw = max(1, int(round(w * scale)))
    nh = max(1, int(round(h * scale)))
    resized = cv2.resize(src, (nw, nh), interpolation=cv2.INTER_AREA)
    x0 = (tw - nw) // 2
    y0 = 30 + (body_h - nh) // 2
    tile[y0 : y0 + nh, x0 : x0 + nw] = resized
    put_label(tile, label)
    return tile


def make_quad(panels, out_path, tile_size=(640, 360)):
    tiles = [letterbox(img, tile_size, label) for img, label in panels]
    top = np.concatenate(tiles[:2], axis=1)
    bottom = np.concatenate(tiles[2:4], axis=1)
    quad = np.concatenate([top, bottom], axis=0)
    imwrite_unicode(out_path, quad)


def make_five_panel(panels, out_path, tile_size=(520, 310)):
    top_tiles = [letterbox(img, tile_size, label) for img, label in panels[:3]]
    bottom_tile_size = (int(tile_size[0] * 1.5), tile_size[1])
    bottom_tiles = [letterbox(img, bottom_tile_size, label) for img, label in panels[3:5]]
    top = np.concatenate(top_tiles, axis=1)
    bottom = np.concatenate(bottom_tiles, axis=1)
    canvas = np.concatenate([top, bottom], axis=0)
    imwrite_unicode(out_path, canvas)


def make_edge_overlay(rgb_bgr, lwir_u8, valid_mask):
    bg = enhance_lowlight_bgr(rgb_bgr)
    gray_rgb = cv2.cvtColor(bg, cv2.COLOR_BGR2GRAY)
    edge_rgb = auto_edges(gray_rgb) > 0
    edge_lwir = (auto_edges(lwir_u8) > 0) & valid_mask
    out = bg.copy()
    out[edge_rgb] = (255, 210, 0)
    out[edge_lwir] = (0, 40, 255)
    both = edge_rgb & edge_lwir
    out[both] = (255, 255, 255)
    return out


def make_gray_edge_overlay(fixed_gray, moving_gray, valid_mask):
    bg = cv2.cvtColor(fixed_gray, cv2.COLOR_GRAY2BGR)
    bg = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(fixed_gray)
    out = cv2.cvtColor(bg, cv2.COLOR_GRAY2BGR)
    fixed_edges = auto_edges(fixed_gray) > 0
    moving_edges = (auto_edges(moving_gray) > 0) & valid_mask
    out[fixed_edges] = (255, 210, 0)
    out[moving_edges] = (0, 40, 255)
    out[fixed_edges & moving_edges] = (255, 255, 255)
    return out


def write_csv(path, rows, fieldnames):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def collect_fieldnames(rows, preferred):
    fields = list(preferred)
    seen = set(fields)
    for row in rows:
        for key in row.keys():
            if key not in seen:
                fields.append(key)
                seen.add(key)
    return fields


def write_numeric_summary(path, rows, ignored_fields):
    summary = {}
    if not rows:
        Path(path).write_text("{}", encoding="utf-8")
        return
    fieldnames = collect_fieldnames(rows, [])
    for key in fieldnames:
        if key in ignored_fields:
            continue
        vals = []
        for row in rows:
            try:
                val = float(row.get(key, float("nan")))
            except (TypeError, ValueError):
                continue
            if math.isfinite(val):
                vals.append(val)
        if vals:
            summary[f"{key}_mean"] = float(np.mean(vals))
            summary[f"{key}_min"] = float(np.min(vals))
            summary[f"{key}_max"] = float(np.max(vals))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


def matrix_to_list(H):
    return [[float(x) for x in row] for row in H]


def process_sample(row, output_dir):
    sid = f"{row.aligned_id:03d}_seq{row.sequence}"
    sample_dir = output_dir / "samples" / sid
    quad_dir = output_dir / "quads"
    five_dir = output_dir / "five_panels"
    edge_dir = output_dir / "edge_reviews"
    sample_dir.mkdir(parents=True, exist_ok=True)
    raw_rgb = imread_unicode(row.raw_rgb1_path, cv2.IMREAD_COLOR)
    raw_rgb_bright = imread_unicode(row.raw_rgb3_path, cv2.IMREAD_COLOR)
    raw_lwir = imread_unicode(row.raw_thermal16_path, cv2.IMREAD_UNCHANGED)
    aligned_rgb = imread_unicode(row.aligned_rgb1_path, cv2.IMREAD_COLOR)
    aligned_lwir = imread_unicode(row.aligned_t16_path, cv2.IMREAD_UNCHANGED)
    raw_lwir_u8 = normalize_u8(raw_lwir)
    aligned_lwir_u8 = normalize_u8(aligned_lwir)
    raw_rgb_gray = gray_u8(raw_rgb)

    H_rgb_to_aligned, rgb_metrics, rgb_info, rgb_stage_rows = estimate_same_modality_transform(
        aligned_rgb, raw_rgb
    )
    H_lwir_to_aligned, lwir_metrics, lwir_info, lwir_stage_rows = estimate_same_modality_transform(
        aligned_lwir_u8, raw_lwir_u8
    )
    H_lwir_to_raw_rgb = safe_inv(H_rgb_to_aligned) @ H_lwir_to_aligned

    rgb_size = (raw_rgb.shape[1], raw_rgb.shape[0])
    aligned_size = (aligned_rgb.shape[1], aligned_rgb.shape[0])
    rgb_to_official = warp_image(raw_rgb, H_rgb_to_aligned, aligned_size, interp=cv2.INTER_LINEAR)
    lwir_to_official = warp_image(raw_lwir_u8, H_lwir_to_aligned, aligned_size, interp=cv2.INTER_LINEAR)
    valid_lwir_official = warp_mask(raw_lwir_u8.shape, H_lwir_to_aligned, aligned_size)
    lwir_absdiff = cv2.absdiff(aligned_lwir_u8, lwir_to_official)
    lwir_official_edge = make_gray_edge_overlay(aligned_lwir_u8, lwir_to_official, valid_lwir_official)
    H_lwir_fit_to_raw_rgb = fit_cover_homography(raw_lwir_u8.shape, raw_rgb_gray.shape)
    raw_lwir_baseline_metrics = alignment_metrics(raw_rgb_gray, raw_lwir_u8, H_lwir_fit_to_raw_rgb)
    raw_lwir_calibrated_metrics = alignment_metrics(raw_rgb_gray, raw_lwir_u8, H_lwir_to_raw_rgb)
    lwir_on_rgb = warp_image(raw_lwir_u8, H_lwir_to_raw_rgb, rgb_size, interp=cv2.INTER_LINEAR)
    valid_on_rgb = warp_mask(raw_lwir_u8.shape, H_lwir_to_raw_rgb, rgb_size)
    anno_mask = load_annotation_mask(row.raw_rgb_anno_class_path, raw_rgb.shape)
    fused_heat, fused_intensity, alpha_mask, rgb_display = make_fusion(
        raw_rgb, lwir_on_rgb, valid_on_rgb, anno_mask
    )
    edge_overlay = make_edge_overlay(raw_rgb, lwir_on_rgb, valid_on_rgb)

    imwrite_unicode(sample_dir / "rgb1_raw.png", raw_rgb)
    imwrite_unicode(sample_dir / "rgb3_raw_bright_reference.png", raw_rgb_bright)
    imwrite_unicode(sample_dir / "rgb1_raw_display_enhanced.png", rgb_display)
    imwrite_unicode(sample_dir / "lwir_raw_norm.png", raw_lwir_u8)
    imwrite_unicode(sample_dir / "rgb1_raw_to_official_rgb1.png", rgb_to_official)
    imwrite_unicode(sample_dir / "lwir_raw_to_official_lwir.png", lwir_to_official)
    imwrite_unicode(sample_dir / "lwir_official_absdiff.png", lwir_absdiff)
    imwrite_unicode(sample_dir / "lwir_official_edge_check.png", lwir_official_edge)
    imwrite_unicode(sample_dir / "lwir_calibrated_to_rgb1_raw.png", lwir_on_rgb)
    imwrite_unicode(sample_dir / "fused_heat.png", fused_heat)
    imwrite_unicode(sample_dir / "fused_intensity.png", fused_intensity)
    imwrite_unicode(sample_dir / "fusion_alpha_mask.png", alpha_mask)
    imwrite_unicode(sample_dir / "edge_overlay.png", edge_overlay)

    np.save(sample_dir / "H_raw_rgb1_to_aligned_rgb1.npy", H_rgb_to_aligned)
    np.save(sample_dir / "H_raw_lwir_to_aligned_lwir.npy", H_lwir_to_aligned)
    np.save(sample_dir / "H_raw_lwir_to_raw_rgb1.npy", H_lwir_to_raw_rgb)
    transform_info = {
        "aligned_id": row.aligned_id,
        "sequence": row.sequence,
        "dark_mean": row.dark_mean,
        "H_raw_rgb1_to_aligned_rgb1": matrix_to_list(H_rgb_to_aligned),
        "H_raw_lwir_to_aligned_lwir": matrix_to_list(H_lwir_to_aligned),
        "H_raw_lwir_to_raw_rgb1": matrix_to_list(H_lwir_to_raw_rgb),
        "rgb_info": rgb_info,
        "lwir_info": lwir_info,
        "raw_lwir_fit_to_raw_rgb1": metrics_for_csv(raw_lwir_baseline_metrics),
        "raw_lwir_calibrated_to_raw_rgb1": metrics_for_csv(raw_lwir_calibrated_metrics),
    }
    (sample_dir / "transform_info.json").write_text(
        json.dumps(transform_info, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    make_quad(
        [
            (raw_rgb, f"RGB1 Raw dark mean {row.dark_mean:.2f}"),
            (raw_lwir_u8, "LWIR Raw normalized"),
            (lwir_on_rgb, "Calibrated LWIR to RGB1 Raw"),
            (fused_heat, "Fused Result"),
        ],
        quad_dir / f"{sid}_quad.png",
    )
    make_five_panel(
        [
            (raw_rgb, f"RGB1 Raw dark/synced {row.dark_mean:.2f}"),
            (raw_rgb_bright, "RGB3 Raw bright reference"),
            (raw_lwir_u8, "LWIR Raw normalized"),
            (lwir_on_rgb, "Calibrated LWIR to RGB1 Raw"),
            (fused_heat, "Fused Result"),
        ],
        five_dir / f"{sid}_five_panel.png",
    )
    aligned_lwir_bgr = cv2.cvtColor(aligned_lwir_u8, cv2.COLOR_GRAY2BGR)
    make_quad(
        [
            (aligned_rgb, "Official RGB1 aligned"),
            (rgb_to_official, "Raw RGB1 to Official"),
            (aligned_lwir_bgr, "Official LWIR aligned"),
            (cv2.cvtColor(lwir_to_official, cv2.COLOR_GRAY2BGR), "Raw LWIR to Official"),
        ],
        edge_dir / f"{sid}_official_check.png",
    )
    make_quad(
        [
            (aligned_lwir_bgr, "Official LWIR aligned"),
            (cv2.cvtColor(lwir_to_official, cv2.COLOR_GRAY2BGR), "Raw LWIR to Official"),
            (lwir_absdiff, "LWIR Official AbsDiff"),
            (lwir_official_edge, "LWIR Edge Check"),
        ],
        edge_dir / f"{sid}_lwir_calibration_check.png",
    )
    make_quad(
        [
            (aligned_rgb, "Official RGB1 aligned"),
            (aligned_lwir_bgr, "Official LWIR aligned"),
            (edge_overlay, "Edge Check cyan=RGB red=LWIR"),
            (alpha_mask, "Fusion Alpha Mask"),
        ],
        edge_dir / f"{sid}_edge_review.png",
    )

    registration_rows = []
    for stage in rgb_stage_rows:
        registration_rows.append(
            {
                "sample": sid,
                "aligned_id": row.aligned_id,
                "sequence": row.sequence,
                "pipeline": "raw_rgb1_to_official_rgb1",
                **stage,
            }
        )
    for stage in lwir_stage_rows:
        registration_rows.append(
            {
                "sample": sid,
                "aligned_id": row.aligned_id,
                "sequence": row.sequence,
                "pipeline": "raw_lwir_to_official_lwir",
                **stage,
            }
        )
    registration_rows.extend(
        [
            {
                "sample": sid,
                "aligned_id": row.aligned_id,
                "sequence": row.sequence,
                "pipeline": "raw_lwir_to_raw_rgb1",
                "stage": "fit_cover_baseline",
                **metrics_for_csv(raw_lwir_baseline_metrics),
            },
            {
                "sample": sid,
                "aligned_id": row.aligned_id,
                "sequence": row.sequence,
                "pipeline": "raw_lwir_to_raw_rgb1",
                "stage": "composed_calibrated",
                **metrics_for_csv(raw_lwir_calibrated_metrics),
            },
        ]
    )

    fusion_rows = [
        image_metric_row(sid, "raw_rgb1", raw_rgb, raw_rgb_gray, lwir_on_rgb, valid_on_rgb, alpha_mask),
        image_metric_row(sid, "enhanced_rgb1", rgb_display, raw_rgb_gray, lwir_on_rgb, valid_on_rgb, alpha_mask),
        image_metric_row(sid, "fused_intensity", fused_intensity, raw_rgb_gray, lwir_on_rgb, valid_on_rgb, alpha_mask),
        image_metric_row(sid, "fused_heat", fused_heat, raw_rgb_gray, lwir_on_rgb, valid_on_rgb, alpha_mask),
    ]
    fused_heat_metrics = next(r for r in fusion_rows if r["image"] == "fused_heat")
    raw_rgb_metrics = next(r for r in fusion_rows if r["image"] == "raw_rgb1")

    sample_summary = {
        "aligned_id": row.aligned_id,
        "sequence": row.sequence,
        "split": row.split,
        "category": row.category,
        "subcategory": row.subcategory,
        "dark_mean": row.dark_mean,
        "rgb_ncc_to_official": rgb_info["ncc"],
        "rgb_edge_distance_to_official": rgb_info["edge_distance"],
        "rgb_valid_ratio": rgb_info["valid_ratio"],
        "rgb_initial": rgb_info["initial"],
        "rgb_ecc": rgb_info["ecc"]["reason"],
        "lwir_ncc_to_official": lwir_info["ncc"],
        "lwir_edge_distance_to_official": lwir_info["edge_distance"],
        "lwir_valid_ratio": lwir_info["valid_ratio"],
        "lwir_initial": lwir_info["initial"],
        "lwir_ecc": lwir_info["ecc"]["reason"],
        "raw_lwir_to_rgb_baseline_ncc": raw_lwir_baseline_metrics["ncc"],
        "raw_lwir_to_rgb_baseline_edge_distance": raw_lwir_baseline_metrics["edge_distance"],
        "raw_lwir_to_rgb_calibrated_ncc": raw_lwir_calibrated_metrics["ncc"],
        "raw_lwir_to_rgb_calibrated_edge_distance": raw_lwir_calibrated_metrics["edge_distance"],
        "raw_lwir_to_rgb_ncc_gain": raw_lwir_calibrated_metrics["ncc"] - raw_lwir_baseline_metrics["ncc"],
        "raw_lwir_to_rgb_edge_distance_gain": raw_lwir_baseline_metrics["edge_distance"]
        - raw_lwir_calibrated_metrics["edge_distance"],
        "fusion_entropy_gain_vs_raw_rgb": fused_heat_metrics["entropy"] - raw_rgb_metrics["entropy"],
        "fusion_avg_gradient_gain_vs_raw_rgb": fused_heat_metrics["average_gradient"]
        - raw_rgb_metrics["average_gradient"],
        "fusion_mi_with_lwir": fused_heat_metrics["mi_with_calibrated_lwir"],
        "fusion_mi_with_raw_rgb": fused_heat_metrics["mi_with_raw_rgb"],
        "fusion_mean_abs_change_vs_raw_rgb": fused_heat_metrics["mean_abs_change_vs_raw_rgb"],
        "fusion_alpha_mean": fused_heat_metrics["alpha_mean"],
        "fusion_alpha_coverage": fused_heat_metrics["alpha_coverage"],
        "quad": str(quad_dir / f"{sid}_quad.png"),
        "five_panel": str(five_dir / f"{sid}_five_panel.png"),
        "edge_review": str(edge_dir / f"{sid}_edge_review.png"),
        "official_check": str(edge_dir / f"{sid}_official_check.png"),
        "lwir_calibration_check": str(edge_dir / f"{sid}_lwir_calibration_check.png"),
    }
    return sample_summary, registration_rows, fusion_rows


def main():
    parser = argparse.ArgumentParser(description="Official-reference dark-light MM5 RGB1+LWIR review.")
    parser.add_argument(
        "--index",
        default="mm5_calib_benchmark/outputs/mm5_benchmark/splits/index_with_splits.csv",
    )
    parser.add_argument("--output", default="darklight_mm5/outputs")
    parser.add_argument("--limit", type=int, default=3)
    parser.add_argument("--splits", default="test,val")
    parser.add_argument("--aligned-ids", default="")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = load_rows(
        args.index,
        require_official=True,
        require_depth=False,
    )
    if args.aligned_ids.strip():
        wanted = {int(x.strip()) for x in args.aligned_ids.split(",") if x.strip()}
        selected = [r for r in rows if r.aligned_id in wanted]
        selected = sorted(selected, key=lambda r: r.aligned_id)
    else:
        selected = select_dark_rows(rows, args.limit, args.splits)
    if not selected:
        raise RuntimeError("no valid MM5 samples selected")

    selected_rows = [
        {
            "aligned_id": r.aligned_id,
            "sequence": r.sequence,
            "split": r.split,
            "dark_mean": r.dark_mean,
            "raw_rgb1_path": r.raw_rgb1_path,
            "raw_rgb3_path": r.raw_rgb3_path,
            "raw_thermal16_path": r.raw_thermal16_path,
            "raw_depth_tr_path": r.raw_depth_tr_path,
            "aligned_rgb1_path": r.aligned_rgb1_path,
            "aligned_t16_path": r.aligned_t16_path,
            "strategy": "official_reference",
        }
        for r in selected
    ]
    write_csv(
        output_dir / "selected_dark_samples.csv",
        selected_rows,
        [
            "aligned_id",
            "sequence",
            "split",
            "dark_mean",
            "raw_rgb1_path",
            "raw_rgb3_path",
            "raw_thermal16_path",
            "raw_depth_tr_path",
            "aligned_rgb1_path",
            "aligned_t16_path",
            "strategy",
        ],
    )

    metric_rows = []
    registration_stage_rows = []
    fusion_metric_rows = []
    for row in selected:
        print(
            f"processing aligned_id={row.aligned_id} sequence={row.sequence} "
            f"mean={row.dark_mean:.2f} strategy=official_reference"
        )
        sample_summary, sample_registration, sample_fusion = process_sample(row, output_dir)
        metric_rows.append(sample_summary)
        registration_stage_rows.extend(sample_registration)
        fusion_metric_rows.extend(sample_fusion)

    per_sample_fields = collect_fieldnames(
        metric_rows,
        [
            "aligned_id",
            "sequence",
            "split",
            "category",
            "subcategory",
            "strategy",
            "dark_mean",
            "rgb_ncc_to_official",
            "lwir_ncc_to_official",
            "depth_valid_ratio",
            "reprojection_valid_ratio",
            "raw_lwir_to_rgb_baseline_ncc",
            "raw_lwir_to_rgb_calibrated_ncc",
            "raw_lwir_to_rgb_ncc_gain",
            "fusion_entropy_gain_vs_raw_rgb",
            "fusion_alpha_coverage",
            "five_panel",
        ],
    )
    registration_fields = collect_fieldnames(
        registration_stage_rows,
        [
            "sample",
            "aligned_id",
            "sequence",
            "pipeline",
            "stage",
            "ncc",
            "edge_distance",
            "mi",
            "valid_ratio",
            "score",
            "depth_valid_ratio",
            "reprojection_valid_ratio",
            "residual_dx",
            "residual_dy",
            "refine_accepted",
            "failure_reason",
            "initial_used",
            "ecc_reason",
        ],
    )
    fusion_fields = collect_fieldnames(
        fusion_metric_rows,
        [
            "sample",
            "image",
            "entropy",
            "average_gradient",
            "spatial_frequency",
            "std",
            "mean_brightness",
            "mi_with_raw_rgb",
            "mi_with_calibrated_lwir",
            "ncc_with_raw_rgb",
            "ncc_with_calibrated_lwir",
            "mean_abs_change_vs_raw_rgb",
            "mean_abs_change_on_alpha",
            "alpha_mean",
            "alpha_coverage",
            "valid_ratio",
        ],
    )
    write_csv(output_dir / "metrics" / "per_sample.csv", metric_rows, per_sample_fields)
    write_csv(output_dir / "metrics" / "registration_stages.csv", registration_stage_rows, registration_fields)
    write_csv(output_dir / "metrics" / "fusion_metrics.csv", fusion_metric_rows, fusion_fields)
    write_numeric_summary(
        output_dir / "metrics" / "summary.json",
        metric_rows,
        {
            "split",
            "category",
            "subcategory",
            "strategy",
            "calibration_file",
            "rgb_initial",
            "rgb_ecc",
            "lwir_initial",
            "lwir_ecc",
            "quad",
            "five_panel",
            "edge_review",
            "official_check",
            "lwir_calibration_check",
        },
    )
    write_numeric_summary(
        output_dir / "metrics" / "registration_summary.json",
        registration_stage_rows,
        {"sample", "pipeline", "stage", "failure_reason", "initial_used", "ecc_reason"},
    )
    write_numeric_summary(
        output_dir / "metrics" / "fusion_summary.json",
        fusion_metric_rows,
        {"sample", "image"},
    )
    print("done")
    for row in metric_rows:
        for key in ["quad", "five_panel", "edge_review", "official_check", "lwir_calibration_check"]:
            if row.get(key):
                print(row[key])


if __name__ == "__main__":
    main()
