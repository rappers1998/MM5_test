from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch

from ..datasets.mm5_dataset_busref import BusReFReconDataset
from ..losses.image_losses import masked_mse, ssim_loss
from ..models.busref import BusReconstructor
from ..utils.config_utils import load_yaml
from ..utils.io_utils import ensure_dir
from .trainer_utils import device_from_config, image_size_from_config, make_loader, save_history, seed_torch, strip_rows


def _run_epoch(model: BusReconstructor, loader, device: torch.device, optimizer, config: dict[str, Any], train: bool) -> dict[str, float]:
    model.train(train)
    totals = {"total": 0.0, "mse": 0.0, "ssim": 0.0}
    count = 0
    for raw_batch in loader:
        batch = {key: value.to(device) for key, value in strip_rows(raw_batch).items()}
        inputs = torch.cat([batch["visible"], batch["thermal"]], dim=0)
        with torch.set_grad_enabled(train):
            recon, _features = model(inputs)
            mse = masked_mse(recon, inputs)
            ssim = ssim_loss(recon, inputs)
            loss = mse + float(config["training"].get("recon_ssim_weight", 0.2)) * ssim
            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
        n = int(inputs.shape[0])
        totals["total"] += float(loss.detach().cpu()) * n
        totals["mse"] += float(mse.detach().cpu()) * n
        totals["ssim"] += float(ssim.detach().cpu()) * n
        count += n
    return {key: value / max(count, 1) for key, value in totals.items()}


def train(config: dict[str, Any]) -> dict[str, Any]:
    seed_torch(int(config.get("runtime", {}).get("seed", 42)))
    device = device_from_config(config)
    image_size = image_size_from_config(config)
    runtime = config.get("runtime", {})
    data_cfg = config["data"]
    train_ds = BusReFReconDataset(data_cfg["real_split"], "train", image_size, int(runtime.get("train_limit", 48)))
    val_ds = BusReFReconDataset(data_cfg["real_split"], "val", image_size, int(runtime.get("val_limit", 16)))
    train_loader = make_loader(train_ds, config, shuffle=True)
    val_loader = make_loader(val_ds, config, shuffle=False)

    model = BusReconstructor().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config["training"].get("lr", 1e-4)))
    output_dir = ensure_dir(config["outputs"]["recon_dir"])
    history: list[dict[str, Any]] = []
    best = float("inf")

    for epoch in range(int(config["training"].get("epochs", 2))):
        train_metrics = _run_epoch(model, train_loader, device, optimizer, config, train=True)
        val_metrics = _run_epoch(model, val_loader, device, optimizer, config, train=False)
        row = {
            "epoch": epoch + 1,
            "train_total": train_metrics["total"],
            "val_total": val_metrics["total"],
            "val_mse": val_metrics["mse"],
            "val_ssim_loss": val_metrics["ssim"],
        }
        history.append(row)
        if val_metrics["total"] < best:
            best = val_metrics["total"]
            torch.save({"model": model.state_dict(), "config": config, "val_total": best}, output_dir / "best.pt")

    summary = {
        "stage": "busref_recon",
        "train_samples": len(train_ds),
        "val_samples": len(val_ds),
        "best_val_total": best,
        "checkpoint": str(Path(output_dir) / "best.pt"),
    }
    save_history(output_dir, history, summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the compact BusReF reconstructor.")
    parser.add_argument("--config", default="mm5_ivf/configs/busref_mm5_v2.yaml")
    args = parser.parse_args()
    summary = train(load_yaml(args.config))
    print(summary)


if __name__ == "__main__":
    main()
