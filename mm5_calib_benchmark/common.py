from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np


def ensure_dir(path: str | Path) -> Path:
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out


def read_image(path: str | Path, flags: int = cv2.IMREAD_UNCHANGED) -> np.ndarray | None:
    file_path = Path(path)
    if not file_path.exists():
        return None
    data = np.fromfile(str(file_path), dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, flags)


def write_image(path: str | Path, image: np.ndarray) -> None:
    file_path = Path(path)
    ensure_dir(file_path.parent)
    suffix = file_path.suffix or ".png"
    ok, encoded = cv2.imencode(suffix, image)
    if not ok:
        raise RuntimeError(f"Failed to encode image for {file_path}")
    encoded.tofile(str(file_path))


def to_gray_u8(image: np.ndarray) -> np.ndarray:
    if image is None:
        raise ValueError("image is None")
    if image.ndim == 3 and image.shape[2] == 4:
        gray = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
    elif image.ndim == 3 and image.shape[2] == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    gray = np.asarray(gray)
    if gray.dtype == np.uint8:
        return gray

    gray = gray.astype(np.float32)
    finite = np.isfinite(gray)
    if not finite.any():
        return np.zeros(gray.shape, dtype=np.uint8)

    values = gray[finite]
    lo = float(np.percentile(values, 1.0))
    hi = float(np.percentile(values, 99.0))
    if hi <= lo:
        hi = lo + 1.0
    norm = np.clip((gray - lo) / (hi - lo), 0.0, 1.0)
    return (norm * 255.0).astype(np.uint8)


def ensure_bgr(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return cv2.cvtColor(to_gray_u8(image), cv2.COLOR_GRAY2BGR)
    if image.ndim == 3 and image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    if image.dtype != np.uint8:
        return cv2.cvtColor(to_gray_u8(image), cv2.COLOR_GRAY2BGR)
    return image.copy()


def modality_preprocess(image: np.ndarray, modality: str) -> np.ndarray:
    gray = to_gray_u8(image)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    if modality == "thermal":
        gray = clahe.apply(gray)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
    elif modality == "uv":
        gray = clahe.apply(gray)
    elif modality == "ir":
        gray = clahe.apply(gray)
    return gray


def label_to_color(mask: np.ndarray) -> np.ndarray:
    palette = np.array(
        [
            [0, 0, 0],
            [255, 200, 0],
            [255, 0, 255],
            [0, 255, 255],
            [0, 255, 0],
            [255, 0, 0],
            [0, 0, 255],
        ],
        dtype=np.uint8,
    )
    mask_u8 = mask.astype(np.uint8)
    color = np.zeros((*mask_u8.shape, 3), dtype=np.uint8)
    for cls in np.unique(mask_u8):
        color[mask_u8 == cls] = palette[int(cls) % len(palette)]
    return color


def load_opencv_yaml(path: str | Path) -> dict[str, Any]:
    file_path = str(Path(path))
    storage = cv2.FileStorage(file_path, cv2.FILE_STORAGE_READ)
    if not storage.isOpened():
        raise FileNotFoundError(file_path)

    def read_node(node) -> Any:
        if node.empty():
            return None
        try:
            mat = node.mat()
            if mat is not None and hasattr(mat, "size") and mat.size:
                return np.asarray(mat)
        except cv2.error:
            pass
        if node.isMap():
            return {key: read_node(node.getNode(key)) for key in node.keys()}
        if node.isSeq():
            values = [read_node(node.at(i)) for i in range(node.size())]
            if values and all(isinstance(v, (int, float, np.integer, np.floating)) for v in values):
                return np.asarray(values, dtype=np.float64)
            return values
        try:
            text = node.string()
            if text != "":
                return text
        except Exception:
            pass
        try:
            return float(node.real())
        except Exception:
            return None

    payload = {key: read_node(storage.getNode(key)) for key in storage.root().keys()}
    storage.release()
    return payload


def save_opencv_yaml(path: str | Path, payload: dict[str, Any]) -> None:
    file_path = Path(path)
    ensure_dir(file_path.parent)
    storage = cv2.FileStorage(str(file_path), cv2.FILE_STORAGE_WRITE)
    if not storage.isOpened():
        raise RuntimeError(f"Unable to open {file_path} for writing")

    for key, value in payload.items():
        if value is None:
            continue
        if isinstance(value, str):
            storage.write(key, value)
        elif isinstance(value, (int, float, np.integer, np.floating)):
            storage.write(key, float(value))
        else:
            storage.write(key, np.asarray(value))
    storage.release()


def stereo_baseline_mm(stereo: dict[str, Any]) -> float:
    t = np.asarray(stereo["T"], dtype=np.float64).reshape(3)
    return float(np.linalg.norm(t))


def compute_plane_homography(stereo: dict[str, Any], source_size: tuple[int, int], plane_depth_mm: float) -> np.ndarray:
    del source_size
    k1 = np.asarray(stereo["CM1"], dtype=np.float64)
    k2 = np.asarray(stereo["CM2"], dtype=np.float64)
    r = np.asarray(stereo["R"], dtype=np.float64)
    t = np.asarray(stereo["T"], dtype=np.float64).reshape(3, 1)
    normal = np.array([[0.0], [0.0], [1.0]], dtype=np.float64)
    h = k2 @ (r - (t @ normal.T) / float(max(plane_depth_mm, 1.0))) @ np.linalg.inv(k1)
    return h / max(h[2, 2], 1e-9)


def warp_image(image: np.ndarray, homography: np.ndarray, target_size: tuple[int, int]) -> np.ndarray:
    return cv2.warpPerspective(image, homography, target_size, flags=cv2.INTER_LINEAR, borderValue=0)


def warp_mask(mask: np.ndarray, homography: np.ndarray, target_size: tuple[int, int]) -> np.ndarray:
    return cv2.warpPerspective(mask, homography, target_size, flags=cv2.INTER_NEAREST, borderValue=0)


def valid_mask_from_homography(source_size: tuple[int, int], homography: np.ndarray, target_size: tuple[int, int]) -> np.ndarray:
    src_w, src_h = source_size
    white = np.full((src_h, src_w), 255, dtype=np.uint8)
    warped = cv2.warpPerspective(white, homography, target_size, flags=cv2.INTER_NEAREST, borderValue=0)
    return (warped > 0).astype(np.uint8)


def mutual_information(image_a: np.ndarray, image_b: np.ndarray, valid_mask: np.ndarray | None = None, bins: int = 64) -> float:
    a = to_gray_u8(image_a).astype(np.float32)
    b = to_gray_u8(image_b).astype(np.float32)
    if valid_mask is None:
        valid = np.ones_like(a, dtype=bool)
    else:
        valid = valid_mask.astype(bool)
    if int(valid.sum()) < 32:
        return 0.0
    a_vals = a[valid].ravel()
    b_vals = b[valid].ravel()
    hist, _, _ = np.histogram2d(a_vals, b_vals, bins=bins, range=[[0, 255], [0, 255]])
    pxy = hist / max(hist.sum(), 1.0)
    px = pxy.sum(axis=1, keepdims=True)
    py = pxy.sum(axis=0, keepdims=True)
    denom = px @ py
    valid_bins = pxy > 0
    return float(np.sum(pxy[valid_bins] * np.log((pxy[valid_bins] + 1e-12) / (denom[valid_bins] + 1e-12))))


def normalized_total_gradient(image_a: np.ndarray, image_b: np.ndarray, valid_mask: np.ndarray | None = None) -> float:
    a = modality_preprocess(image_a, "rgb").astype(np.float32)
    b = modality_preprocess(image_b, "thermal").astype(np.float32)
    if valid_mask is None:
        valid = np.ones_like(a, dtype=bool)
    else:
        valid = valid_mask.astype(bool)
    if int(valid.sum()) < 32:
        return 0.0

    ax = cv2.Sobel(a, cv2.CV_32F, 1, 0, ksize=3)
    ay = cv2.Sobel(a, cv2.CV_32F, 0, 1, ksize=3)
    bx = cv2.Sobel(b, cv2.CV_32F, 1, 0, ksize=3)
    by = cv2.Sobel(b, cv2.CV_32F, 0, 1, ksize=3)

    diff = np.abs(ax - bx) + np.abs(ay - by)
    mag = np.abs(ax) + np.abs(ay) + np.abs(bx) + np.abs(by)
    score = 1.0 - float(diff[valid].mean()) / float(mag[valid].mean() + 1e-6)
    return float(np.clip(score, 0.0, 1.0))


def fit_image(image: np.ndarray, width: int, height: int, background: tuple[int, int, int] = (18, 18, 18)) -> np.ndarray:
    canvas = np.full((height, width, 3), background, dtype=np.uint8)
    bgr = ensure_bgr(image)
    src_h, src_w = bgr.shape[:2]
    scale = min(width / max(src_w, 1), height / max(src_h, 1))
    new_w = max(1, int(round(src_w * scale)))
    new_h = max(1, int(round(src_h * scale)))
    resized = cv2.resize(bgr, (new_w, new_h), interpolation=cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR)
    x0 = (width - new_w) // 2
    y0 = (height - new_h) // 2
    canvas[y0 : y0 + new_h, x0 : x0 + new_w] = resized
    return canvas


def draw_text_block(
    image: np.ndarray,
    text: str,
    origin: tuple[int, int],
    color: tuple[int, int, int] = (255, 255, 255),
    bg_color: tuple[int, int, int] = (0, 0, 0),
    font_scale: float = 0.52,
    line_height: int = 20,
) -> np.ndarray:
    out = image.copy()
    x0, y0 = origin
    lines = [line for line in text.splitlines() if line]
    if not lines:
        return out
    widths = [cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)[0][0] for line in lines]
    block_w = max(widths) + 12
    block_h = len(lines) * line_height + 10
    x1 = min(out.shape[1], x0 + block_w)
    y1 = min(out.shape[0], y0 + block_h)
    cv2.rectangle(out, (x0, y0), (x1, y1), bg_color, -1)
    for idx, line in enumerate(lines):
        y = y0 + 18 + idx * line_height
        cv2.putText(out, line, (x0 + 6, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, 1, cv2.LINE_AA)
    return out


def compose_rows(rows: list[list[np.ndarray]], gap: int = 8, background: tuple[int, int, int] = (24, 24, 24)) -> np.ndarray:
    processed_rows = []
    row_width = 0
    for row in rows:
        height = max(img.shape[0] for img in row)
        padded = []
        for img in row:
            tile = img
            if tile.shape[0] != height:
                tile = fit_image(tile, tile.shape[1], height, background)
            padded.append(tile)
        width = sum(img.shape[1] for img in padded) + gap * max(len(padded) - 1, 0)
        row_width = max(row_width, width)
        processed_rows.append((padded, height, width))

    total_h = sum(height for _, height, _ in processed_rows) + gap * max(len(processed_rows) - 1, 0)
    canvas = np.full((total_h, row_width, 3), background, dtype=np.uint8)
    y = 0
    for padded, height, width in processed_rows:
        x = 0
        for img in padded:
            canvas[y : y + img.shape[0], x : x + img.shape[1]] = img
            x += img.shape[1] + gap
        y += height + gap
    return canvas
