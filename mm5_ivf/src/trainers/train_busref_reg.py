from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch

from ..datasets.mm5_dataset_busref import BusReFRegDataset
from ..losses.image_losses import flow_smoothness, masked_gradient_loss, masked_mse, masked_ncc
from ..models.busref import BusReFRegistration, BusReconstructor
from ..utils.config_utils import load_yaml
from ..utils.io_utils import ensure_dir
from .trainer_utils import device_from_config, image_size_from_config, make_loader, save_history, seed_torch, strip_rows


def _load_reconstructor(config: dict[str, Any], device: torch.device) -> BusReconstructor:
    model = BusReconstructor().to(device)
    checkpoint_path = Path(config["outputs"]["recon_dir"]) / "best.pt"
    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model"])
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    return model


def _run_epoch(reg_model, recon, loader, device, optimizer, config: dict[str, Any], train: bool) -> dict[str, float]:
    reg_model.train(train)
    totals = {"total": 0.0, "before_mmse": 0.0, "after_mmse": 0.0, "before_mncc": 0.0, "after_mncc": 0.0}
    count = 0
    weights = config["training"]
    for raw_batch in loader:
        batch = {key: value.to(device) for key, value in strip_rows(raw_batch).items()}
        fixed_visible = batch["fixed_visible"]
        fixed_thermal = batch["fixed_thermal"]
        moving = batch["moving_thermal"]
        mask = batch["mask"]
        with torch.no_grad():
            _rv, fixed_features = recon(fixed_visible)
            _rm, moving_features = recon(moving)
        with torch.set_grad_enabled(train):
            out = reg_model(fixed_visible, moving, fixed_features[-1], moving_features[-1])
            registered = out["registered"]
            after_mmse_loss = masked_mse(registered, fixed_thermal, mask)
            after_mncc = masked_ncc(registered, fixed_thermal, mask)
            grad = masked_gradient_loss(registered, fixed_thermal, mask)
            smooth = flow_smoothness(out["flow"])
            loss = (
                float(weights.get("reg_mmse_weight", 1.0)) * after_mmse_loss
                - float(weights.get("reg_mncc_weight", 1.0)) * after_mncc
                + float(weights.get("reg_grad_weight", 0.2)) * grad
                + float(weights.get("reg_smooth_weight", 0.02)) * smooth
            )
            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
        before_mmse = masked_mse(moving, fixed_thermal, mask)
        before_mncc = masked_ncc(moving, fixed_thermal, mask)
        n = int(fixed_visible.shape[0])
        totals["total"] += float(loss.detach().cpu()) * n
        totals["before_mmse"] += float(before_mmse.detach().cpu()) * n
        totals["after_mmse"] += float(after_mmse_loss.detach().cpu()) * n
        totals["before_mncc"] += float(before_mncc.detach().cpu()) * n
        totals["after_mncc"] += float(after_mncc.detach().cpu()) * n
        count += n
    return {key: value / max(count, 1) for key, value in totals.items()}


def train(config: dict[str, Any]) -> dict[str, Any]:
    seed_torch(int(config.get("runtime", {}).get("seed", 42)))
    device = device_from_config(config)
    image_size = image_size_from_config(config)
    runtime = config.get("runtime", {})
    data_cfg = config["data"]
    train_ds = BusReFRegDataset(data_cfg["synth_split"], "train", image_size, int(runtime.get("train_limit", 48)))
    val_ds = BusReFRegDataset(data_cfg["synth_split"], "val", image_size, int(runtime.get("val_limit", 16)))
    train_loader = make_loader(train_ds, config, shuffle=True)
    val_loader = make_loader(val_ds, config, shuffle=False)

    recon = _load_reconstructor(config, device)
    reg_model = BusReFRegistration().to(device)
    optimizer = torch.optim.AdamW(reg_model.parameters(), lr=float(config["training"].get("lr", 1e-4)))
    output_dir = ensure_dir(config["outputs"]["reg_dir"])
    history: list[dict[str, Any]] = []
    best = float("inf")

    for epoch in range(int(config["training"].get("epochs", 2))):
        train_metrics = _run_epoch(reg_model, recon, train_loader, device, optimizer, config, train=True)
        val_metrics = _run_epoch(reg_model, recon, val_loader, device, optimizer, config, train=False)
        row = {
            "epoch": epoch + 1,
            "train_total": train_metrics["total"],
            "val_total": val_metrics["total"],
            "before_mmse": val_metrics["before_mmse"],
            "after_mmse": val_metrics["after_mmse"],
            "before_mncc": val_metrics["before_mncc"],
            "after_mncc": val_metrics["after_mncc"],
        }
        history.append(row)
        if val_metrics["after_mmse"] < best:
            best = val_metrics["after_mmse"]
            torch.save({"model": reg_model.state_dict(), "config": config, "after_mmse": best}, output_dir / "best.pt")

    summary = {
        "stage": "busref_registration",
        "train_samples": len(train_ds),
        "val_samples": len(val_ds),
        "best_after_mmse": best,
        "last_before_mncc": history[-1]["before_mncc"] if history else None,
        "last_after_mncc": history[-1]["after_mncc"] if history else None,
        "checkpoint": str(Path(output_dir) / "best.pt"),
    }
    save_history(output_dir, history, summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Train compact BusReF registration.")
    parser.add_argument("--config", default="mm5_ivf/configs/busref_mm5_v2.yaml")
    args = parser.parse_args()
    summary = train(load_yaml(args.config))
    print(summary)


if __name__ == "__main__":
    main()
