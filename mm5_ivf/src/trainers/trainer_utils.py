from __future__ import annotations

import csv
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from ..utils.io_utils import ensure_dir, write_json


def seed_torch(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def device_from_config(config: dict[str, Any]) -> torch.device:
    requested = str(config.get("runtime", {}).get("device", "cpu"))
    if requested == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def image_size_from_config(config: dict[str, Any]) -> tuple[int, int]:
    width, height = config.get("runtime", {}).get("image_size", [160, 128])
    return int(width), int(height)


def make_loader(dataset, config: dict[str, Any], shuffle: bool) -> DataLoader:
    runtime = config.get("runtime", {})
    return DataLoader(
        dataset,
        batch_size=int(runtime.get("batch_size", 4)),
        shuffle=shuffle,
        num_workers=int(runtime.get("num_workers", 0)),
    )


def save_history(output_dir: str | Path, history: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    out = ensure_dir(output_dir)
    if history:
        keys = list(history[0].keys())
        with (out / "history.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=keys)
            writer.writeheader()
            writer.writerows(history)
    write_json(out / "summary.json", summary)


def strip_rows(batch: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in batch.items() if key != "row"}
