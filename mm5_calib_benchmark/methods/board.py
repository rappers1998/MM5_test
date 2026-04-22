from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np

from ..common import load_opencv_yaml, modality_preprocess, read_image


_OBSERVATION_CACHE: dict[tuple[str, str, float], dict[str, Any]] = {}


def calibration_root_from_rows(rows: list[Any], config: dict[str, Any]) -> Path:
    for row in rows:
        root = Path(str(row.payload.get("calibration_root", "")).strip())
        if root.exists():
            return root
    return Path(config["paths"]["workspace_calibration"])


def official_stereo(config: dict[str, Any], track: str) -> dict[str, Any]:
    calib_dir = Path(config["paths"]["workspace_calibration"])
    filename = "def_stereocalib_THERM.yml" if track == "thermal" else "def_stereocalib_UV.yml"
    return load_opencv_yaml(calib_dir / filename)


def collect_board_pairs(calibration_root: Path, track: str) -> list[tuple[Path, Path]]:
    capture_name = "capture_THERM" if track == "thermal" else "capture_UV"
    capture_root = calibration_root / capture_name / "1280x720"
    pairs: list[tuple[Path, Path]] = []
    for left_path in sorted(capture_root.rglob("*_left.png")):
        right_path = Path(str(left_path).replace("_left.png", "_right.png"))
        if right_path.exists():
            pairs.append((left_path, right_path))
    return pairs


def _detect_chessboard(image: np.ndarray, modality: str, pattern_size: tuple[int, int]) -> np.ndarray | None:
    gray = modality_preprocess(image, modality)
    candidates = [gray, cv2.GaussianBlur(gray, (3, 3), 0)]
    if modality != "rgb":
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        candidates.extend([enhanced, 255 - enhanced])

    flags = cv2.CALIB_CB_EXHAUSTIVE + cv2.CALIB_CB_ACCURACY
    for candidate in candidates:
        ok, corners = cv2.findChessboardCornersSB(candidate, pattern_size, flags=flags)
        if ok:
            return corners.reshape(-1, 2).astype(np.float32)
    return None


def _choose_pattern(pairs: list[tuple[Path, Path]], track: str) -> tuple[int, int]:
    candidates = [(11, 8), (10, 7), (9, 6), (8, 5), (7, 6)]
    scores = {pattern: 0 for pattern in candidates}
    for left_path, right_path in pairs[: min(6, len(pairs))]:
        left_img = read_image(left_path, cv2.IMREAD_UNCHANGED)
        right_img = read_image(right_path, cv2.IMREAD_UNCHANGED)
        if left_img is None or right_img is None:
            continue
        for pattern in candidates:
            if _detect_chessboard(left_img, "rgb", pattern) is not None and _detect_chessboard(right_img, track, pattern) is not None:
                scores[pattern] += pattern[0] * pattern[1]
    return max(scores, key=scores.get)


def _object_points(pattern_size: tuple[int, int], square_size_mm: float) -> np.ndarray:
    cols, rows = pattern_size
    grid_x, grid_y = np.meshgrid(np.arange(cols), np.arange(rows))
    obj = np.zeros((cols * rows, 3), dtype=np.float32)
    obj[:, 0] = (grid_x.reshape(-1) * square_size_mm).astype(np.float32)
    obj[:, 1] = (grid_y.reshape(-1) * square_size_mm).astype(np.float32)
    return obj


def load_board_observations(calibration_root: Path, track: str, square_size_mm: float = 25.0) -> dict[str, Any]:
    cache_key = (str(calibration_root), track, float(square_size_mm))
    if cache_key in _OBSERVATION_CACHE:
        return _OBSERVATION_CACHE[cache_key]

    pairs = collect_board_pairs(calibration_root, track)
    pattern_size = _choose_pattern(pairs, track)
    obj_template = _object_points(pattern_size, square_size_mm)
    observations = []
    source_size = None
    target_size = None

    for left_path, right_path in pairs:
        left_img = read_image(left_path, cv2.IMREAD_UNCHANGED)
        right_img = read_image(right_path, cv2.IMREAD_UNCHANGED)
        if left_img is None or right_img is None:
            continue
        source_pts = _detect_chessboard(left_img, "rgb", pattern_size)
        target_pts = _detect_chessboard(right_img, track, pattern_size)
        if source_pts is None or target_pts is None:
            continue
        observations.append(
            {
                "source_points": source_pts,
                "target_points": target_pts,
                "object_points": obj_template.copy(),
                "source_path": str(left_path),
                "target_path": str(right_path),
            }
        )
        source_size = (left_img.shape[1], left_img.shape[0])
        target_size = (right_img.shape[1], right_img.shape[0])

    payload = {
        "pattern_size": pattern_size,
        "square_size_mm": square_size_mm,
        "observations": observations,
        "source_size": source_size,
        "target_size": target_size,
        "num_pairs": len(observations),
    }
    _OBSERVATION_CACHE[cache_key] = payload
    return payload


def build_stereo_dict(
    k1: np.ndarray,
    d1: np.ndarray,
    k2: np.ndarray,
    d2: np.ndarray,
    r: np.ndarray,
    t: np.ndarray,
    source_size: tuple[int, int],
) -> dict[str, Any]:
    stereo = {
        "CM1": np.asarray(k1, dtype=np.float64),
        "CM2": np.asarray(k2, dtype=np.float64),
        "D1": np.asarray(d1, dtype=np.float64).reshape(-1),
        "D2": np.asarray(d2, dtype=np.float64).reshape(-1),
        "R": np.asarray(r, dtype=np.float64),
        "T": np.asarray(t, dtype=np.float64).reshape(3),
    }
    try:
        r1, r2, p1, p2, q, _, _ = cv2.stereoRectify(
            stereo["CM1"],
            stereo["D1"],
            stereo["CM2"],
            stereo["D2"],
            tuple(int(v) for v in source_size),
            stereo["R"],
            stereo["T"],
            flags=cv2.CALIB_ZERO_DISPARITY,
            alpha=0.0,
        )
        stereo["R1"] = r1
        stereo["R2"] = r2
        stereo["P1"] = p1
        stereo["P2"] = p2
        stereo["Q"] = q
    except cv2.error:
        pass
    return stereo


def evaluate_stereo_on_board(stereo: dict[str, Any], board_payload: dict[str, Any]) -> dict[str, float]:
    observations = board_payload.get("observations", [])
    if not observations:
        return {"reprojection_mae_px": 0.0, "checkerboard_corner_rmse_px": 0.0}

    k1 = np.asarray(stereo["CM1"], dtype=np.float64)
    d1 = np.asarray(stereo["D1"], dtype=np.float64).reshape(-1, 1)
    k2 = np.asarray(stereo["CM2"], dtype=np.float64)
    d2 = np.asarray(stereo["D2"], dtype=np.float64).reshape(-1, 1)
    r = np.asarray(stereo["R"], dtype=np.float64)
    t = np.asarray(stereo["T"], dtype=np.float64).reshape(3, 1)

    errors = []
    for obs in observations:
        ok, rvec1, tvec1 = cv2.solvePnP(obs["object_points"], obs["source_points"], k1, d1, flags=cv2.SOLVEPNP_ITERATIVE)
        if not ok:
            continue
        r1, _ = cv2.Rodrigues(rvec1)
        r2 = r @ r1
        t2 = r @ tvec1 + t
        rvec2, _ = cv2.Rodrigues(r2)
        proj2, _ = cv2.projectPoints(obs["object_points"], rvec2, t2, k2, d2)
        diff = proj2.reshape(-1, 2) - obs["target_points"]
        errors.extend(np.linalg.norm(diff, axis=1).tolist())

    if not errors:
        return {"reprojection_mae_px": 0.0, "checkerboard_corner_rmse_px": 0.0}
    error_arr = np.asarray(errors, dtype=np.float64)
    return {
        "reprojection_mae_px": float(error_arr.mean()),
        "checkerboard_corner_rmse_px": float(np.sqrt(np.mean(error_arr**2))),
    }


def calibrate_stereo_from_board(board_payload: dict[str, Any]) -> dict[str, Any] | None:
    observations = board_payload.get("observations", [])
    if len(observations) < 4:
        return None

    object_points = [obs["object_points"] for obs in observations]
    source_points = [obs["source_points"] for obs in observations]
    target_points = [obs["target_points"] for obs in observations]
    source_size = tuple(int(v) for v in board_payload["source_size"])
    target_size = tuple(int(v) for v in board_payload["target_size"])

    _, k1, d1, _, _ = cv2.calibrateCamera(object_points, source_points, source_size, None, None)
    _, k2, d2, _, _ = cv2.calibrateCamera(object_points, target_points, target_size, None, None)
    stereo_rms, k1, d1, k2, d2, r, t, _, _ = cv2.stereoCalibrate(
        object_points,
        source_points,
        target_points,
        k1,
        d1,
        k2,
        d2,
        source_size,
        criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-6),
        flags=cv2.CALIB_FIX_INTRINSIC,
    )
    stereo = build_stereo_dict(k1, d1, k2, d2, r, t, source_size)
    metrics = evaluate_stereo_on_board(stereo, board_payload)
    metrics["stereo_rms"] = float(stereo_rms)
    return {"stereo": stereo, "calibration_metrics": metrics, "board_payload": board_payload}


def _average_rotations(rotations: list[np.ndarray]) -> np.ndarray:
    if not rotations:
        return np.eye(3, dtype=np.float64)
    mean_r = np.mean(np.stack(rotations, axis=0), axis=0)
    u, _, vt = np.linalg.svd(mean_r)
    r = u @ vt
    if np.linalg.det(r) < 0:
        u[:, -1] *= -1
        r = u @ vt
    return r


def estimate_relative_pose_epnp(base_context: dict[str, Any]) -> dict[str, Any] | None:
    board_payload = base_context.get("board_payload", {})
    observations = board_payload.get("observations", [])
    if not observations:
        return None

    stereo = base_context["stereo"]
    k1 = np.asarray(stereo["CM1"], dtype=np.float64)
    d1 = np.asarray(stereo["D1"], dtype=np.float64).reshape(-1, 1)
    k2 = np.asarray(stereo["CM2"], dtype=np.float64)
    d2 = np.asarray(stereo["D2"], dtype=np.float64).reshape(-1, 1)

    rotations = []
    translations = []
    reproj_errors = []
    for obs in observations:
        ok1, rvec1, tvec1 = cv2.solvePnP(obs["object_points"], obs["source_points"], k1, d1, flags=cv2.SOLVEPNP_EPNP)
        ok2, rvec2, tvec2 = cv2.solvePnP(obs["object_points"], obs["target_points"], k2, d2, flags=cv2.SOLVEPNP_EPNP)
        if not ok1 or not ok2:
            continue
        r1, _ = cv2.Rodrigues(rvec1)
        r2, _ = cv2.Rodrigues(rvec2)
        rotations.append(r2 @ r1.T)
        translations.append((tvec2 - (r2 @ r1.T) @ tvec1).reshape(3))
        proj1, _ = cv2.projectPoints(obs["object_points"], rvec1, tvec1, k1, d1)
        proj2, _ = cv2.projectPoints(obs["object_points"], rvec2, tvec2, k2, d2)
        reproj_errors.append(
            float(
                np.mean(
                    np.concatenate(
                        [
                            np.linalg.norm(proj1.reshape(-1, 2) - obs["source_points"], axis=1),
                            np.linalg.norm(proj2.reshape(-1, 2) - obs["target_points"], axis=1),
                        ]
                    )
                )
            )
        )

    if not rotations:
        return None

    avg_r = _average_rotations(rotations)
    avg_t = np.median(np.stack(translations, axis=0), axis=0)
    stereo_out = build_stereo_dict(k1, d1, k2, d2, avg_r, avg_t, tuple(int(v) for v in board_payload["source_size"]))
    metrics = evaluate_stereo_on_board(stereo_out, board_payload)
    metrics["epnp_pair_reprojection_mae_px"] = float(np.mean(reproj_errors)) if reproj_errors else 0.0
    return {"stereo": stereo_out, "calibration_metrics": metrics, "board_payload": board_payload}
