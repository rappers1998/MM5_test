from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path
from typing import Any

from ..utils.config_utils import load_yaml
from ..utils.io_utils import ensure_dir, load_opencv_yaml, write_json, write_jsonl


RAW_NAME_PATTERN = re.compile(
    r"(?P<scene>\d+)_(?P<frame>\d+)_(?P<date>\d{8})_(?P<hms>\d{6})_(?P<ms>\d+)"
)
DATASET_TOKENS = ("MM5_RAW", "MM5_ALIGNED", "MM5_CALIBRATION", "MM5_LABELSTUDIO")


def _read_csv_text(path: Path) -> str:
    for encoding in ("utf-8-sig", "utf-8", "gb18030"):
        try:
            return path.read_text(encoding=encoding)
        except UnicodeDecodeError:
            continue
    return path.read_text(encoding="latin-1")


def _read_index_rows(path: str | Path) -> list[dict[str, str]]:
    csv_path = Path(path)
    reader = csv.DictReader(_read_csv_text(csv_path).splitlines())
    return [dict(row) for row in reader]


def normalize_mm5_path(value: str | None, dataset_root: str | Path) -> str:
    """Map old mojibake MM5 roots to the real local dataset root."""
    if value is None:
        return ""
    text = str(value).strip()
    if not text:
        return ""
    normalized = text.replace("/", "\\")
    root = str(Path(dataset_root)).replace("/", "\\").rstrip("\\")
    for token in DATASET_TOKENS:
        marker = f"\\{token}\\"
        idx = normalized.find(marker)
        if idx >= 0:
            return root + normalized[idx:]
        if normalized.endswith(f"\\{token}"):
            return root + "\\" + token
    return text


def _parse_timestamp(raw_path: str) -> dict[str, Any]:
    match = RAW_NAME_PATTERN.search(Path(raw_path).stem)
    if not match:
        return {
            "frame_idx": None,
            "timestamp": None,
            "timestamp_or_frame_idx": Path(raw_path).stem,
        }
    timestamp = f"{match.group('date')}_{match.group('hms')}_{match.group('ms')}"
    return {
        "frame_idx": int(match.group("frame")),
        "timestamp": timestamp,
        "timestamp_or_frame_idx": f"{timestamp}|frame={match.group('frame')}",
    }


def _path_exists(value: str) -> bool:
    return bool(value) and Path(value).exists()


def _calibration_file_for_row(row: dict[str, str], fallback_path: str | Path, dataset_root: str | Path) -> Path:
    fallback = Path(fallback_path)
    if fallback.exists():
        return fallback
    root_text = normalize_mm5_path(row.get("calibration_root", ""), dataset_root)
    if not root_text:
        return fallback
    candidate = Path(root_text) / "def_stereocalib_THERM.yml"
    return candidate if candidate.exists() else fallback


def _row_path(row: dict[str, str], key: str, dataset_root: str | Path) -> str:
    return normalize_mm5_path(row.get(key, ""), dataset_root)


def _path_status(paths: dict[str, str]) -> dict[str, bool]:
    return {key: _path_exists(value) for key, value in paths.items()}


def build_manifest(config: dict[str, Any]) -> list[dict[str, Any]]:
    data_cfg = config["data"]
    path_cfg = config["paths"]
    dataset_root = path_cfg["dataset_root"]
    rows = _read_index_rows(path_cfg["existing_index_csv"])
    manifest_rows: list[dict[str, Any]] = []
    calib_cache: dict[str, dict[str, Any]] = {}

    for row in rows:
        calibration_file = _calibration_file_for_row(row, path_cfg["workspace_calibration"], dataset_root)
        calibration_key = str(calibration_file.resolve())
        if calibration_key not in calib_cache:
            calib_cache[calibration_key] = load_opencv_yaml(calibration_file)
        stereo = calib_cache[calibration_key]

        raw_rgb_path = _row_path(row, data_cfg["preferred_rgb_field"], dataset_root)
        raw_lwir_path = _row_path(row, data_cfg["preferred_lwir_field"], dataset_root)
        aligned_rgb_path = _row_path(row, data_cfg["aligned_rgb_field"], dataset_root)
        aligned_lwir_path = _row_path(row, data_cfg["aligned_lwir_field"], dataset_root)
        raw_depth_path = _row_path(row, "raw_depth_tr_path", dataset_root)
        raw_meta_path = _row_path(row, "raw_meta_path", dataset_root)
        aligned_meta_path = _row_path(row, "aligned_meta_path", dataset_root)
        calibration_root = _row_path(row, "calibration_root", dataset_root)
        timestamp_info = _parse_timestamp(raw_rgb_path or raw_lwir_path or aligned_rgb_path or aligned_lwir_path)
        path_status = _path_status(
            {
                "raw_rgb_path": raw_rgb_path,
                "raw_lwir_path": raw_lwir_path,
                "aligned_rgb_path": aligned_rgb_path,
                "aligned_lwir_path": aligned_lwir_path,
                "raw_depth_path": raw_depth_path,
                "raw_meta_path": raw_meta_path,
                "aligned_meta_path": aligned_meta_path,
                "calibration_root": calibration_root,
            }
        )

        manifest_rows.append(
            {
                "sample_id": int(row.get("aligned_id", "0") or 0),
                "scene_id": int(row.get("sequence", "0") or 0),
                "category": row.get("category", ""),
                "subcategory": row.get("subcategory", ""),
                "challenges": row.get("challenges", ""),
                "split": row.get("split", ""),
                "rgb_path": raw_rgb_path,
                "lwir_path": raw_lwir_path,
                "raw_rgb_path": raw_rgb_path,
                "raw_lwir_path": raw_lwir_path,
                "raw_depth_path": raw_depth_path,
                "aligned_rgb_path": aligned_rgb_path,
                "aligned_lwir_path": aligned_lwir_path,
                "raw_meta_path": raw_meta_path,
                "aligned_meta_path": aligned_meta_path,
                "calibration_root": calibration_root,
                "calibration_file": calibration_key,
                "calib_intrinsic_rgb": stereo["CM1"].tolist(),
                "calib_intrinsic_lwir": stereo["CM2"].tolist(),
                "calib_extrinsic_rgb_to_lwir": {
                    "R": stereo["R"].tolist(),
                    "T": stereo["T"].tolist(),
                },
                "timestamp_or_frame_idx": timestamp_info["timestamp_or_frame_idx"],
                "frame_idx": timestamp_info["frame_idx"],
                "timestamp": timestamp_info["timestamp"],
                "path_exists": path_status,
                "has_aligned_pair": path_status["aligned_rgb_path"] and path_status["aligned_lwir_path"],
                "has_raw_pair": path_status["raw_rgb_path"] and path_status["raw_lwir_path"],
                "has_depth": path_status["raw_depth_path"],
                "valid_modalities": row.get("valid_modalities", ""),
            }
        )
    return manifest_rows


def summarize_manifest(manifest_rows: list[dict[str, Any]]) -> dict[str, Any]:
    path_keys = sorted({key for row in manifest_rows for key in row.get("path_exists", {})})
    return {
        "count": len(manifest_rows),
        "splits": sorted({row["split"] for row in manifest_rows}),
        "raw_pairs": sum(int(row["has_raw_pair"]) for row in manifest_rows),
        "aligned_pairs": sum(int(row["has_aligned_pair"]) for row in manifest_rows),
        "depth_pairs": sum(int(row["has_depth"]) for row in manifest_rows),
        "path_exists": {
            key: sum(int(row.get("path_exists", {}).get(key, False)) for row in manifest_rows)
            for key in path_keys
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build an MM5 IVF manifest.")
    parser.add_argument("--config", default="mm5_ivf/configs/data_mm5.yaml")
    args = parser.parse_args()

    config = load_yaml(args.config)
    output_root = ensure_dir(Path(config["paths"]["output_root"]) / "manifests")
    manifest_rows = build_manifest(config)

    write_jsonl(output_root / "manifest.jsonl", manifest_rows)
    summary = summarize_manifest(manifest_rows)
    write_json(output_root / "manifest_summary.json", summary)
    print(f"Manifest written: {output_root / 'manifest.jsonl'}")
    print(summary)


if __name__ == "__main__":
    main()
