from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from ..utils.config_utils import load_yaml
from ..utils.geometry_utils import normalize_projected_thermal, resize_mask, thermal_to_rgb_homography, warp_thermal_to_rgb
from ..utils.io_utils import ensure_dir, read_image, read_jsonl, to_gray_u8, write_image, write_json


def _entry_dir(output_root: Path, row: dict[str, Any], source_mode: str) -> Path:
    return output_root / "canonical_pairs" / f"scene_{int(row['scene_id']):04d}_sample_{int(row['sample_id']):04d}_{source_mode}"


def _resize_image(image: np.ndarray, size: tuple[int, int], is_mask: bool = False) -> np.ndarray:
    interpolation = cv2.INTER_NEAREST if is_mask else cv2.INTER_AREA
    return cv2.resize(image, size, interpolation=interpolation)


def _write_pair(
    out_dir: Path,
    row: dict[str, Any],
    source_mode: str,
    rgb: np.ndarray,
    lwir: np.ndarray,
    mask: np.ndarray,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    ensure_dir(out_dir)
    rgb_path = out_dir / "rgb_canonical.png"
    lwir_path = out_dir / "lwir_canonical.png"
    mask_path = out_dir / "overlap_mask.png"
    meta_path = out_dir / "metadata.json"
    write_image(rgb_path, rgb)
    write_image(lwir_path, to_gray_u8(lwir))
    write_image(mask_path, (mask.astype(np.uint8) * 255))
    write_json(meta_path, metadata)

    output = dict(row)
    output.update(
        {
            "source_mode": source_mode,
            "rgb_canonical_path": str(rgb_path),
            "lwir_canonical_path": str(lwir_path),
            "overlap_mask_path": str(mask_path),
            "canonical_metadata_path": str(meta_path),
            "canonical_size": [int(rgb.shape[1]), int(rgb.shape[0])],
            "canonical_exists": {
                "rgb_canonical_path": True,
                "lwir_canonical_path": True,
                "overlap_mask_path": True,
            },
        }
    )
    return output


def _build_raw_pair(row: dict[str, Any], config: dict[str, Any], output_root: Path) -> dict[str, Any] | None:
    if not row.get("has_raw_pair"):
        return None
    raw_rgb = read_image(row["raw_rgb_path"], cv2.IMREAD_COLOR)
    raw_lwir = read_image(row["raw_lwir_path"], cv2.IMREAD_UNCHANGED)
    if raw_rgb is None or raw_lwir is None:
        return None

    runtime_cfg = config.get("runtime", {})
    target_size = tuple(int(v) for v in runtime_cfg.get("raw_target_size", [640, 512]))
    h = thermal_to_rgb_homography(row, runtime_cfg, raw_rgb.shape[:2], raw_lwir.shape[:2])
    projected, valid = warp_thermal_to_rgb(raw_lwir, h, raw_rgb.shape[:2])
    rgb_out = _resize_image(raw_rgb, target_size)
    lwir_out = _resize_image(normalize_projected_thermal(projected, valid), target_size)
    mask_out = resize_mask(valid, target_size)
    metadata = {
        "source_mode": "raw_homography",
        "raw_rgb_shape": [int(raw_rgb.shape[1]), int(raw_rgb.shape[0])],
        "raw_lwir_shape": [int(raw_lwir.shape[1]), int(raw_lwir.shape[0])],
        "target_size": [int(target_size[0]), int(target_size[1])],
        "homography_thermal_to_rgb": h.tolist(),
        "valid_coverage": float(mask_out.mean()),
    }
    output = _write_pair(_entry_dir(output_root, row, "raw_homography"), row, "raw_homography", rgb_out, lwir_out, mask_out, metadata)
    output["homography_thermal_to_rgb"] = h.tolist()
    output["valid_coverage"] = float(mask_out.mean())
    return output


def _build_aligned_pair(row: dict[str, Any], config: dict[str, Any], output_root: Path) -> dict[str, Any] | None:
    if not row.get("has_aligned_pair"):
        return None
    aligned_rgb = read_image(row["aligned_rgb_path"], cv2.IMREAD_COLOR)
    aligned_lwir = read_image(row["aligned_lwir_path"], cv2.IMREAD_UNCHANGED)
    if aligned_rgb is None or aligned_lwir is None:
        return None

    runtime_cfg = config.get("runtime", {})
    target_size = tuple(int(v) for v in runtime_cfg.get("aligned_target_size", [640, 480]))
    rgb_out = _resize_image(aligned_rgb, target_size)
    lwir_out = _resize_image(to_gray_u8(aligned_lwir), target_size)
    mask_out = np.ones((target_size[1], target_size[0]), dtype=np.uint8)
    metadata = {
        "source_mode": "aligned_reference",
        "aligned_rgb_shape": [int(aligned_rgb.shape[1]), int(aligned_rgb.shape[0])],
        "aligned_lwir_shape": [int(aligned_lwir.shape[1]), int(aligned_lwir.shape[0])],
        "target_size": [int(target_size[0]), int(target_size[1])],
        "valid_coverage": 1.0,
    }
    return _write_pair(_entry_dir(output_root, row, "aligned_reference"), row, "aligned_reference", rgb_out, lwir_out, mask_out, metadata)


def build_canonical_pairs(config: dict[str, Any]) -> dict[str, Any]:
    output_root = Path(config["paths"]["output_root"])
    manifest_path = output_root / "manifests" / "manifest.jsonl"
    manifest_rows = read_jsonl(manifest_path)
    split_rows: list[dict[str, Any]] = []
    aligned_rows: list[dict[str, Any]] = []

    for row in manifest_rows:
        raw_pair = _build_raw_pair(row, config, output_root)
        if raw_pair is not None:
            split_rows.append(raw_pair)
        aligned_pair = _build_aligned_pair(row, config, output_root)
        if aligned_pair is not None:
            aligned_rows.append(aligned_pair)

    split_dir = ensure_dir(output_root / "splits")
    write_json(split_dir / "split_real_raw.json", split_rows)
    write_json(split_dir / "split_aligned_reference.json", aligned_rows)
    summary = {
        "raw_homography": len(split_rows),
        "aligned_reference": len(aligned_rows),
        "splits": sorted({row.get("split", "") for row in split_rows}),
    }
    write_json(output_root / "canonical_summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Build MM5 canonical RGB/LWIR pairs.")
    parser.add_argument("--config", default="mm5_ivf/configs/data_mm5.yaml")
    args = parser.parse_args()

    config = load_yaml(args.config)
    summary = build_canonical_pairs(config)
    print(summary)


if __name__ == "__main__":
    main()
