from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch

from ..datasets.mm5_dataset_busref import BusReFRegDataset
from ..losses.image_losses import flow_smoothness, masked_gradient_loss, masked_mse
from ..models.bsr import BSRFlowNet, ipdg_tensor, pdg_tensor
from ..utils.config_utils import load_yaml
from .trainer_utils import device_from_config, image_size_from_config, make_loader, save_history, seed_torch, strip_rows


def _run_epoch(model, loader, device, optimizer, config: dict[str, Any], train: bool) -> dict[str, float]:
    model.train(train)
    totals = {"total": 0.0, "before_mse": 0.0, "after_mse": 0.0, "flow_abs": 0.0}
    count = 0
    n_pdg = int(config["training"].get("pdg_n", 2))
    for raw_batch in loader:
        batch = {key: value.to(device) for key, value in strip_rows(raw_batch).items()}
        visible = batch["fixed_visible"]
        moving = batch["moving_thermal"]
        target = batch["fixed_thermal"]
        mask = batch["mask"]
        pdg_seed = int(config.get("runtime", {}).get("seed", 42)) + count
        pdg_visible, ops = pdg_tensor(visible, n=n_pdg, seed=pdg_seed)
        pdg_moving = pdg_tensor(moving, n=n_pdg, seed=pdg_seed)[0]
        with torch.set_grad_enabled(train):
            out = model(moving, visible)
            pdg_out = model(pdg_moving, pdg_visible)
            pdg_back = ipdg_tensor(pdg_out["thermal_to_visible"], ops, n=n_pdg)
            after = out["thermal_to_visible"]
            align = masked_mse(after, target, mask)
            edge = masked_gradient_loss(after, visible, mask)
            branch = (after - pdg_back).abs().mean()
            smooth = flow_smoothness(out["phi_p"]) + flow_smoothness(out["phi_n"])
            loss = align + edge + float(config["training"].get("w1", 5.0)) * branch + float(config["training"].get("smooth_weight", 0.02)) * smooth
            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
        before = masked_mse(moving, target, mask)
        n = int(visible.shape[0])
        totals["total"] += float(loss.detach().cpu()) * n
        totals["before_mse"] += float(before.detach().cpu()) * n
        totals["after_mse"] += float(align.detach().cpu()) * n
        totals["flow_abs"] += float(out["phi_p"].abs().mean().detach().cpu()) * n
        count += n
    return {key: value / max(count, 1) for key, value in totals.items()}


def train(config: dict[str, Any]) -> dict[str, Any]:
    seed_torch(int(config.get("runtime", {}).get("seed", 42)))
    device = device_from_config(config)
    image_size = image_size_from_config(config)
    runtime = config.get("runtime", {})
    train_ds = BusReFRegDataset(config["data"]["synth_split"], "train", image_size, int(runtime.get("train_limit", 16)))
    val_ds = BusReFRegDataset(config["data"]["synth_split"], "val", image_size, int(runtime.get("val_limit", 8)))
    train_loader = make_loader(train_ds, config, shuffle=True)
    val_loader = make_loader(val_ds, config, shuffle=False)
    model = BSRFlowNet().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config["training"].get("lr", 1e-4)))
    output_dir = Path(config["outputs"]["root"])
    history: list[dict[str, Any]] = []
    best = float("inf")

    for epoch in range(int(config["training"].get("epochs", 1))):
        train_metrics = _run_epoch(model, train_loader, device, optimizer, config, train=True)
        val_metrics = _run_epoch(model, val_loader, device, optimizer, config, train=False)
        row = {
            "epoch": epoch + 1,
            "train_total": train_metrics["total"],
            "val_total": val_metrics["total"],
            "before_mse": val_metrics["before_mse"],
            "after_mse": val_metrics["after_mse"],
            "flow_abs": val_metrics["flow_abs"],
        }
        history.append(row)
        if val_metrics["after_mse"] < best:
            best = val_metrics["after_mse"]
            output_dir.mkdir(parents=True, exist_ok=True)
            torch.save({"model": model.state_dict(), "config": config, "after_mse": best}, output_dir / "best.pt")

    summary = {
        "stage": "bsr_v2_smoke",
        "train_samples": len(train_ds),
        "val_samples": len(val_ds),
        "best_after_mse": best,
        "checkpoint": str(output_dir / "best.pt"),
    }
    save_history(output_dir, history, summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a lightweight B-SR PDG/IPDG smoke training pass.")
    parser.add_argument("--config", default="mm5_ivf/configs/bsr_mm5_v2.yaml")
    args = parser.parse_args()
    summary = train(load_yaml(args.config))
    print(summary)


if __name__ == "__main__":
    main()
