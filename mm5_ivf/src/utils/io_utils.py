from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import cv2
import numpy as np


def ensure_dir(path: str | Path) -> Path:
    output_path = Path(path)
    output_path.mkdir(parents=True, exist_ok=True)
    return output_path


def read_image(path: str | Path, flags: int = cv2.IMREAD_UNCHANGED) -> np.ndarray | None:
    file_path = Path(path)
    if not file_path.exists():
        return None
    data = np.fromfile(str(file_path), dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, flags)


def write_image(path: str | Path, image: np.ndarray) -> None:
    output_path = Path(path)
    ensure_dir(output_path.parent)
    suffix = output_path.suffix or ".png"
    ok, encoded = cv2.imencode(suffix, image)
    if not ok:
        raise RuntimeError(f"Failed to encode image: {output_path}")
    encoded.tofile(str(output_path))


def read_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: str | Path, payload: Any) -> None:
    output_path = Path(path)
    ensure_dir(output_path.parent)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    output_path = Path(path)
    ensure_dir(output_path.parent)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def to_gray_u8(image: np.ndarray) -> np.ndarray:
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


def normalize_unit(image: np.ndarray) -> np.ndarray:
    gray = image.astype(np.float32)
    finite = np.isfinite(gray)
    if not finite.any():
        return np.zeros(gray.shape, dtype=np.float32)
    values = gray[finite]
    lo = float(values.min())
    hi = float(values.max())
    if hi <= lo:
        return np.zeros(gray.shape, dtype=np.float32)
    return np.clip((gray - lo) / (hi - lo), 0.0, 1.0)


def modality_preprocess(image: np.ndarray, modality: str) -> np.ndarray:
    gray = to_gray_u8(image)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    if modality in {"thermal", "lwir"}:
        gray = clahe.apply(gray)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
    elif modality in {"rgb", "visible"}:
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
    return gray


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


def draw_label(image: np.ndarray, text: str, origin: tuple[int, int] = (8, 22)) -> np.ndarray:
    output = image.copy()
    cv2.rectangle(output, (origin[0] - 4, origin[1] - 18), (origin[0] + 8 * max(len(text), 1), origin[1] + 6), (12, 12, 12), -1)
    cv2.putText(output, text, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.55, (240, 240, 240), 1, cv2.LINE_AA)
    return output


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)

