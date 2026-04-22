from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the original Scene 282 MAR full pipeline from sibling MAR_test.")
    parser.add_argument(
        "--mar-mode",
        choices=["engineered_best", "paper_final", "all"],
        default="all",
        help="Which historical MAR mode to reproduce.",
    )
    parser.add_argument(
        "--save-level",
        choices=["final_only", "paper_steps", "debug"],
        default="debug",
        help="Output level forwarded to backup_2600.py.",
    )
    return parser.parse_args()


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def legacy_root() -> Path:
    return project_root().parent / "MAR_test"


def legacy_python_executable(root: Path) -> str:
    candidate = root / ".venv" / "Scripts" / "python.exe"
    if candidate.exists():
        return str(candidate)
    return sys.executable


def scene282_asset_paths(root: Path) -> dict[str, Path]:
    assets = root / "scene282_assets"
    return {
        "rgb": assets / "rgb_282_3.png",
        "label": assets / "rgb_label_282.bmp",
        "target": assets / "thermal_282_1_lwir16.png",
        "depth": assets / "depth_282_1_tr.png",
        "stereo": assets / "def_stereocalib_THERM.yml",
        "target_cam": assets / "def_thermalcam_ori.yml",
        "rgb_cam": assets / "def_stereocalib_cam.yml",
    }


def build_command(root: Path, mar_mode: str, save_level: str, out_dir: Path) -> list[str]:
    paths = scene282_asset_paths(root)
    return [
        legacy_python_executable(root),
        str(root / "backup_2600.py"),
        "--mar-mode",
        mar_mode,
        "--rgb",
        str(paths["rgb"]),
        "--label",
        str(paths["label"]),
        "--target",
        str(paths["target"]),
        "--target-modality",
        "thermal",
        "--stereo",
        str(paths["stereo"]),
        "--target-cam",
        str(paths["target_cam"]),
        "--rgb-cam",
        str(paths["rgb_cam"]),
        "--depth",
        str(paths["depth"]),
        "--save-level",
        save_level,
        "--out",
        str(out_dir),
    ]


def main() -> None:
    args = parse_args()
    root = legacy_root()
    if not (root / "backup_2600.py").exists():
        raise FileNotFoundError(f"Legacy MAR root not found: {root}")

    modes = ["engineered_best", "paper_final"] if args.mar_mode == "all" else [args.mar_mode]
    outputs_root = project_root() / "mm5_calib_benchmark" / "outputs" / "mm5_benchmark" / "legacy_mar_scene282_reproduced"
    outputs_root.mkdir(parents=True, exist_ok=True)

    for mar_mode in modes:
        out_dir = outputs_root / mar_mode
        out_dir.mkdir(parents=True, exist_ok=True)
        cmd = build_command(root, mar_mode, args.save_level, out_dir)
        print(f"[legacy-mar] running {mar_mode}: {out_dir}")
        subprocess.run(cmd, cwd=str(root), check=True)


if __name__ == "__main__":
    main()
