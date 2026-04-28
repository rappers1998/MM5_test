from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from ..utils.config_utils import load_yaml
from ..utils.io_utils import ensure_dir, read_image, read_json, seed_everything, to_gray_u8, write_image, write_json


def _sample_affine(rng: np.random.Generator, size: tuple[int, int], cfg: dict[str, Any]) -> np.ndarray:
    width, height = size
    center = (width / 2.0, height / 2.0)
    angle = float(rng.uniform(-cfg["rot_deg"], cfg["rot_deg"]))
    scale = float(1.0 + rng.uniform(-cfg["scale_delta"], cfg["scale_delta"]))
    matrix = cv2.getRotationMatrix2D(center, angle, scale).astype(np.float32)
    shear = math.tan(math.radians(float(rng.uniform(-cfg["shear_deg"], cfg["shear_deg"]))))
    shear_matrix = np.array([[1.0, shear, -shear * center[1]], [0.0, 1.0, 0.0]], dtype=np.float32)
    affine3 = np.vstack([matrix, [0.0, 0.0, 1.0]])
    shear3 = np.vstack([shear_matrix, [0.0, 0.0, 1.0]])
    out = (shear3 @ affine3)[:2]
    out[:, 2] += [
        float(rng.uniform(-cfg["tx_px"], cfg["tx_px"])),
        float(rng.uniform(-cfg["ty_px"], cfg["ty_px"])),
    ]
    return out.astype(np.float32)


def _elastic_field(rng: np.random.Generator, shape: tuple[int, int], alpha: float, sigma: float) -> np.ndarray:
    height, width = shape
    dx = rng.normal(0.0, 1.0, (height, width)).astype(np.float32)
    dy = rng.normal(0.0, 1.0, (height, width)).astype(np.float32)
    dx = cv2.GaussianBlur(dx, (0, 0), float(sigma)) * float(alpha)
    dy = cv2.GaussianBlur(dy, (0, 0), float(sigma)) * float(alpha)
    return np.stack([dx, dy], axis=-1).astype(np.float32)


def _remap_with_flow(image: np.ndarray, flow: np.ndarray, interpolation: int, border_mode: int) -> np.ndarray:
    height, width = image.shape[:2]
    xx, yy = np.meshgrid(np.arange(width, dtype=np.float32), np.arange(height, dtype=np.float32))
    map_x = xx + flow[..., 0].astype(np.float32)
    map_y = yy + flow[..., 1].astype(np.float32)
    return cv2.remap(image, map_x, map_y, interpolation=interpolation, borderMode=border_mode, borderValue=0)


def _warp_affine(image: np.ndarray, affine: np.ndarray, interpolation: int, border_mode: int) -> np.ndarray:
    height, width = image.shape[:2]
    return cv2.warpAffine(image, affine, (width, height), flags=interpolation, borderMode=border_mode, borderValue=0)


def _make_sample(row: dict[str, Any], out_dir: Path, cfg: dict[str, Any], rng: np.random.Generator) -> dict[str, Any] | None:
    fixed_rgb = read_image(row["rgb_canonical_path"], cv2.IMREAD_COLOR)
    fixed_lwir = read_image(row["lwir_canonical_path"], cv2.IMREAD_UNCHANGED)
    overlap = read_image(row["overlap_mask_path"], cv2.IMREAD_GRAYSCALE)
    if fixed_rgb is None or fixed_lwir is None or overlap is None:
        return None

    fixed_lwir = to_gray_u8(fixed_lwir)
    overlap = (overlap > 0).astype(np.uint8)
    height, width = fixed_lwir.shape[:2]
    affine = _sample_affine(rng, (width, height), cfg)
    flow = _elastic_field(rng, (height, width), float(cfg["elastic_alpha"]), float(cfg["elastic_sigma"]))

    affined = _warp_affine(fixed_lwir, affine, cv2.INTER_LINEAR, cv2.BORDER_REFLECT101)
    moving = _remap_with_flow(affined, flow, cv2.INTER_LINEAR, cv2.BORDER_REFLECT101)

    mask0 = np.ones((height, width), dtype=np.uint8)
    forward_mask = _warp_affine(mask0, affine, cv2.INTER_NEAREST, cv2.BORDER_CONSTANT)
    forward_mask = _remap_with_flow(forward_mask, flow, cv2.INTER_NEAREST, cv2.BORDER_CONSTANT)
    inverse_affine = cv2.invertAffineTransform(affine)
    recovered = _remap_with_flow(forward_mask, -flow, cv2.INTER_NEAREST, cv2.BORDER_CONSTANT)
    recovered = _warp_affine(recovered, inverse_affine, cv2.INTER_NEAREST, cv2.BORDER_CONSTANT)
    reconstructible = ((recovered > 0) & (overlap > 0)).astype(np.uint8)

    ensure_dir(out_dir)
    fixed_rgb_path = out_dir / "fixed_rgb.png"
    fixed_lwir_path = out_dir / "fixed_lwir.png"
    moving_path = out_dir / "moving_lwir.png"
    mask_path = out_dir / "reconstructible_mask.png"
    overlap_path = out_dir / "overlap_mask.png"
    flow_path = out_dir / "elastic_flow.npy"
    meta_path = out_dir / "metadata.json"

    write_image(fixed_rgb_path, fixed_rgb)
    write_image(fixed_lwir_path, fixed_lwir)
    write_image(moving_path, moving)
    write_image(mask_path, reconstructible * 255)
    write_image(overlap_path, overlap * 255)
    np.save(flow_path, flow)
    metadata = {
        "scene_id": int(row["scene_id"]),
        "sample_id": int(row["sample_id"]),
        "affine_matrix": affine.tolist(),
        "elastic_alpha": float(cfg["elastic_alpha"]),
        "elastic_sigma": float(cfg["elastic_sigma"]),
        "reconstructible_coverage": float(reconstructible.mean()),
    }
    write_json(meta_path, metadata)

    sample = dict(row)
    sample.update(
        {
            "fixed_path": str(fixed_rgb_path),
            "fixed_rgb_path": str(fixed_rgb_path),
            "fixed_lwir_path": str(fixed_lwir_path),
            "moving_path": str(moving_path),
            "moving_lwir_path": str(moving_path),
            "reconstructible_mask_path": str(mask_path),
            "synthetic_overlap_mask_path": str(overlap_path),
            "elastic_flow_path": str(flow_path),
            "synthetic_metadata_path": str(meta_path),
            "affine_matrix": affine.tolist(),
            "elastic_alpha": float(cfg["elastic_alpha"]),
            "elastic_sigma": float(cfg["elastic_sigma"]),
            "reconstructible_coverage": float(reconstructible.mean()),
        }
    )
    return sample


def build_synthetic_pairs(config: dict[str, Any]) -> dict[str, Any]:
    seed_everything(int(config.get("runtime", {}).get("seed", 42)))
    rng = np.random.default_rng(int(config.get("runtime", {}).get("seed", 42)))
    output_root = Path(config["paths"]["output_root"])
    canonical_rows = read_json(output_root / "splits" / "split_real_raw.json")
    limit = int(config.get("runtime", {}).get("synthetic_limit", 0) or 0)
    if limit > 0:
        canonical_rows = canonical_rows[:limit]

    split_dir = ensure_dir(output_root / "splits")
    summary: dict[str, Any] = {}
    for regime, regime_cfg in config.get("perturbation", {}).items():
        rows: list[dict[str, Any]] = []
        for row in canonical_rows:
            out_dir = output_root / "synthetic_pairs" / f"scene_{int(row['scene_id']):04d}_sample_{int(row['sample_id']):04d}_{regime}"
            sample = _make_sample(row, out_dir, regime_cfg, rng)
            if sample is not None:
                sample["synthetic_regime"] = regime
                rows.append(sample)
        write_json(split_dir / f"split_synth_{regime}.json", rows)
        summary[regime] = len(rows)
    write_json(output_root / "synthetic_summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Build MM5 synthetic perturbation pairs.")
    parser.add_argument("--config", default="mm5_ivf/configs/data_mm5.yaml")
    args = parser.parse_args()

    config = load_yaml(args.config)
    summary = build_synthetic_pairs(config)
    print(summary)


if __name__ == "__main__":
    main()
