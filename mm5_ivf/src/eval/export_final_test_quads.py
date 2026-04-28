from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from ..utils.config_utils import load_yaml
from ..utils.geometry_utils import (
    estimate_depth_mm,
    feather_mask,
    normalize_projected_thermal,
    thermal_to_rgb_homography,
    warp_thermal_to_rgb,
)
from ..utils.io_utils import ensure_bgr, ensure_dir, read_image, to_gray_u8, write_json
from ..utils.viz_utils import save_panel


def _calibrated_overlay(rgb_raw: np.ndarray, thermal_proj: np.ndarray, feather: np.ndarray) -> np.ndarray:
    base = ensure_bgr(rgb_raw).astype(np.float32)
    thermal_bgr = cv2.cvtColor(thermal_proj, cv2.COLOR_GRAY2BGR).astype(np.float32)
    alpha = (0.42 * feather)[..., None]
    calibrated = base * (1.0 - alpha) + thermal_bgr * alpha
    return np.clip(calibrated, 0.0, 255.0).astype(np.uint8)


def _fused_rgb(rgb_raw: np.ndarray, thermal_proj: np.ndarray, feather: np.ndarray) -> np.ndarray:
    base = ensure_bgr(rgb_raw)
    thermal_smooth = cv2.GaussianBlur(thermal_proj, (0, 0), 2.2)
    thermal_low = cv2.GaussianBlur(thermal_smooth, (0, 0), 7.0)
    thermal_detail = np.clip(thermal_smooth.astype(np.float32) - thermal_low.astype(np.float32), -14.0, 14.0)

    ycrcb = cv2.cvtColor(base, cv2.COLOR_BGR2YCrCb)
    luminance = ycrcb[..., 0].astype(np.float32)
    alpha = 0.18 * feather
    fused_y = luminance * (1.0 - alpha) + thermal_smooth.astype(np.float32) * alpha
    fused_y = fused_y + thermal_detail * (0.28 * feather)
    ycrcb[..., 0] = np.clip(fused_y, 0.0, 255.0).astype(np.uint8)
    return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)


def _load_gaf_model(fusion_config_path: str | Path):
    try:
        import torch

        from ..models.busref import GAFFusion
    except Exception:
        return None, None
    cfg_path = Path(fusion_config_path)
    if not cfg_path.exists():
        return None, None
    config = load_yaml(cfg_path)
    checkpoint_path = Path(config.get("outputs", {}).get("fuse_dir", "")) / "best.pt"
    if not checkpoint_path.exists():
        return None, None
    model = GAFFusion()
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model, torch


def _gaf_fused_rgb(rgb_raw: np.ndarray, thermal_proj: np.ndarray, feather: np.ndarray, model, torch_module) -> np.ndarray:
    if model is None or torch_module is None:
        return _fused_rgb(rgb_raw, thermal_proj, feather)
    base = ensure_bgr(rgb_raw)
    visible_gray = to_gray_u8(base).astype(np.float32) / 255.0
    thermal = thermal_proj.astype(np.float32) / 255.0
    mask = np.clip(feather, 0.0, 1.0).astype(np.float32)
    with torch_module.no_grad():
        visible_t = torch_module.from_numpy(visible_gray)[None, None]
        thermal_t = torch_module.from_numpy(thermal)[None, None]
        mask_t = torch_module.from_numpy(mask)[None, None]
        fused_y = model(visible_t, thermal_t, mask_t).squeeze().numpy()
    ycrcb = cv2.cvtColor(base, cv2.COLOR_BGR2YCrCb)
    ycrcb[..., 0] = np.clip(fused_y * 255.0, 0.0, 255.0).astype(np.uint8)
    return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)


def _refine_with_ecc(rgb_raw: np.ndarray, thermal_proj: np.ndarray, valid_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    visible = cv2.equalizeHist(to_gray_u8(rgb_raw))
    moving = cv2.equalizeHist(to_gray_u8(thermal_proj))
    mask = (valid_mask > 0).astype(np.uint8) * 255
    warp = np.eye(2, 3, dtype=np.float32)
    debug: dict[str, Any] = {"used": False}
    try:
        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 40, 1e-5)
        cc, warp = cv2.findTransformECC(visible, moving, warp, cv2.MOTION_AFFINE, criteria, inputMask=mask, gaussFiltSize=5)
        refined = cv2.warpAffine(
            thermal_proj,
            warp,
            (thermal_proj.shape[1], thermal_proj.shape[0]),
            flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        refined_mask = cv2.warpAffine(
            valid_mask.astype(np.uint8) * 255,
            warp,
            (valid_mask.shape[1], valid_mask.shape[0]),
            flags=cv2.INTER_NEAREST | cv2.WARP_INVERSE_MAP,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        debug = {"used": True, "ecc": float(cc), "warp": warp.tolist()}
        return refined, (refined_mask > 0).astype(np.uint8), debug
    except cv2.error as exc:
        debug["reason"] = str(exc).splitlines()[0]
        return thermal_proj, valid_mask, debug


def _project_thermal(row: dict[str, Any], runtime_cfg: dict[str, Any], raw_rgb: np.ndarray, raw_lwir: np.ndarray, method: str) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    plane_depth = float(runtime_cfg.get("plane_depth_mm", 700.0))
    debug: dict[str, Any] = {"method": method, "plane_depth_mm": plane_depth}
    if method == "depth_guided":
        depth = read_image(row.get("raw_depth_path", ""), cv2.IMREAD_UNCHANGED) if row.get("raw_depth_path") else None
        plane_depth = estimate_depth_mm(depth, plane_depth)
        debug["plane_depth_mm"] = plane_depth
    h = thermal_to_rgb_homography(row, runtime_cfg, raw_rgb.shape[:2], raw_lwir.shape[:2], plane_depth)
    projected_raw_thermal, valid_mask = warp_thermal_to_rgb(raw_lwir, h, raw_rgb.shape[:2])
    thermal_proj = normalize_projected_thermal(projected_raw_thermal, valid_mask)
    debug["homography_thermal_to_rgb"] = h.tolist()
    if method == "busref_refined":
        thermal_proj, valid_mask, ecc_debug = _refine_with_ecc(raw_rgb, thermal_proj, valid_mask)
        debug["ecc_refine"] = ecc_debug
    return thermal_proj, valid_mask, debug


def _metric_values(rgb_raw: np.ndarray, thermal_proj: np.ndarray, valid_mask: np.ndarray) -> dict[str, float]:
    rgb = to_gray_u8(rgb_raw).astype(np.float32)
    thermal = thermal_proj.astype(np.float32)
    valid = valid_mask.astype(bool)
    if int(valid.sum()) < 32:
        return {"mse": 0.0, "ncc": 0.0, "edge_score": 0.0}
    rv = rgb[valid]
    tv = thermal[valid]
    mse = float(np.mean((rv / 255.0 - tv / 255.0) ** 2))
    rv0 = rv - float(rv.mean())
    tv0 = tv - float(tv.mean())
    ncc = float(np.sum(rv0 * tv0) / max(np.sqrt(np.sum(rv0**2) * np.sum(tv0**2)), 1e-6))
    rgx = cv2.Sobel(rgb, cv2.CV_32F, 1, 0)
    rgy = cv2.Sobel(rgb, cv2.CV_32F, 0, 1)
    tgx = cv2.Sobel(thermal, cv2.CV_32F, 1, 0)
    tgy = cv2.Sobel(thermal, cv2.CV_32F, 0, 1)
    edge_diff = np.abs(rgx - tgx) + np.abs(rgy - tgy)
    edge_mag = np.abs(rgx) + np.abs(rgy) + np.abs(tgx) + np.abs(tgy)
    edge_score = 1.0 - float(edge_diff[valid].mean()) / float(edge_mag[valid].mean() + 1e-6)
    return {"mse": mse, "ncc": ncc, "edge_score": float(np.clip(edge_score, 0.0, 1.0))}


def _write_metrics_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    fields = ["scene_id", "sample_id", "valid_coverage", "mse", "ncc", "edge_score", "ecc_used"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fields})


def export_test_quads(
    data_config: dict[str, Any],
    output_dir: Path,
    split_name: str,
    method: str,
    fusion: str,
    fusion_config: str | Path,
) -> dict[str, Any]:
    ensure_dir(output_dir)
    split_path = Path(data_config["paths"]["output_root"]) / "splits" / split_name
    rows = json.loads(split_path.read_text(encoding="utf-8"))
    test_rows = [row for row in rows if str(row.get("split", "")) == "test"]
    runtime_cfg = data_config.get("runtime", {})
    gaf_model, torch_module = _load_gaf_model(fusion_config) if fusion == "busref_gaf" else (None, None)
    samples: list[dict[str, Any]] = []
    metrics_rows: list[dict[str, Any]] = []

    for row in test_rows:
        raw_rgb = read_image(row["raw_rgb_path"], cv2.IMREAD_COLOR)
        raw_lwir = read_image(row["raw_lwir_path"], cv2.IMREAD_UNCHANGED)
        if raw_rgb is None or raw_lwir is None:
            raise FileNotFoundError(f"Missing raw sample for scene {row['scene_id']} sample {row['sample_id']}")

        thermal_proj, valid_mask, debug = _project_thermal(row, runtime_cfg, raw_rgb, raw_lwir, method)
        feather = feather_mask(valid_mask)
        calibrated = _calibrated_overlay(raw_rgb, thermal_proj, feather)
        fused = _gaf_fused_rgb(raw_rgb, thermal_proj, feather, gaf_model, torch_module) if fusion == "busref_gaf" else _fused_rgb(raw_rgb, thermal_proj, feather)
        metrics = _metric_values(raw_rgb, thermal_proj, valid_mask)

        panel_path = output_dir / f"scene_{int(row['scene_id']):04d}_sample_{int(row['sample_id']):04d}_quad.png"
        save_panel(
            panel_path,
            [
                ("rgb_raw", ensure_bgr(raw_rgb)),
                ("thermal_raw", ensure_bgr(raw_lwir)),
                ("calibrated", calibrated),
                ("fused", fused),
            ],
            tile_size=(640, 480),
            columns=2,
        )
        sample = {
            "scene_id": int(row["scene_id"]),
            "sample_id": int(row["sample_id"]),
            "quad_path": str(panel_path),
            "valid_coverage": float(valid_mask.mean()),
            "thermal_min": int(thermal_proj[valid_mask > 0].min()) if int(valid_mask.sum()) else 0,
            "thermal_max": int(thermal_proj[valid_mask > 0].max()) if int(valid_mask.sum()) else 0,
            "mse": metrics["mse"],
            "ncc": metrics["ncc"],
            "edge_score": metrics["edge_score"],
            "debug": debug,
        }
        samples.append(sample)
        metrics_rows.append(
            {
                "scene_id": sample["scene_id"],
                "sample_id": sample["sample_id"],
                "valid_coverage": sample["valid_coverage"],
                "mse": sample["mse"],
                "ncc": sample["ncc"],
                "edge_score": sample["edge_score"],
                "ecc_used": bool(debug.get("ecc_refine", {}).get("used", False)),
            }
        )

    metrics_dir = ensure_dir(output_dir / "metrics")
    _write_metrics_csv(metrics_dir / "per_scene.csv", metrics_rows)
    report = {
        "count": len(samples),
        "split": "test",
        "split_name": split_name,
        "method": method,
        "fusion": fusion,
        "gaf_checkpoint_used": bool(gaf_model is not None),
        "metrics_csv": str(metrics_dir / "per_scene.csv"),
        "samples": samples,
    }
    write_json(output_dir / "summary.json", report)
    markdown = [
        "# MM5 Final Test Quads",
        "",
        f"- count: {len(samples)}",
        "- split: test",
        f"- split_file: {split_name}",
        "- thermal_source: raw_lwir",
        f"- geometry: {method}",
        f"- fusion: {fusion}",
        f"- gaf_checkpoint_used: {bool(gaf_model is not None)}",
        f"- metrics: {metrics_dir / 'per_scene.csv'}",
        "",
        "## Files",
    ]
    markdown.extend([f"- {sample['quad_path']}" for sample in samples])
    (output_dir / "README.md").write_text("\n".join(markdown), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Export final MM5 test quads with selectable calibration and fusion.")
    parser.add_argument("--data-config", default="mm5_ivf/configs/data_mm5.yaml")
    parser.add_argument("--split-name", default="split_real_raw.json")
    parser.add_argument("--output-dir", default="mm5_ivf/outputs/final_test_quads_v2")
    parser.add_argument("--method", choices=["official_homography", "depth_guided", "busref_refined"], default="busref_refined")
    parser.add_argument("--fusion", choices=["luminance", "busref_gaf"], default="busref_gaf")
    parser.add_argument("--fusion-config", default="mm5_ivf/configs/busref_mm5_v2.yaml")
    args = parser.parse_args()

    data_config = load_yaml(args.data_config)
    report = export_test_quads(data_config, Path(args.output_dir), args.split_name, args.method, args.fusion, args.fusion_config)
    print(json.dumps({key: value for key, value in report.items() if key != "samples"}, indent=2))


if __name__ == "__main__":
    main()
