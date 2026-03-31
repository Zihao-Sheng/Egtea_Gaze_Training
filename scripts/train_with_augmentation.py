#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.amp import GradScaler
import yaml
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from datasets.egtea_dataset import EgteaVideoDataset, build_split_records
from models.build_model import build_model
from scripts.train import NUM_CLASSES, merge_overrides, run_epoch, train


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and evaluate EgoVideo with configurable clip augmentations.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--train-max-samples", type=int, default=None)
    parser.add_argument("--val-max-samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--resume-from", type=Path, default=None)
    parser.add_argument("--smoke-test", action="store_true")
    return parser.parse_args()


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def evaluate_test(config: dict, checkpoint_path: Path) -> dict:
    device = torch.device(config["device"])
    model_bundle = build_model(config, num_classes=NUM_CLASSES)
    model_bundle.model.to(device)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model_bundle.model.load_state_dict(checkpoint["model_state_dict"])

    test_records = build_split_records(
        data_root=Path(config["data"]["root"]),
        split_id=int(config["data"]["split_id"]),
        split_name="test",
        max_samples=config["data"].get("val_max_samples") if config.get("smoke_test", False) else None,
        seed=int(config["seed"]) + 2,
    )
    test_dataset = EgteaVideoDataset(
        records=test_records,
        num_frames=int(config["model"]["num_frames"]),
        resize_size=int(config["data"]["resize_size"]),
        crop_size=int(config["data"]["crop_size"]),
        mean=list(config["data"]["mean"]),
        std=list(config["data"]["std"]),
        is_train=False,
        augmentation=config.get("augmentation"),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=int(config["training"]["batch_size"]),
        shuffle=False,
        num_workers=int(config["training"]["num_workers"]),
        pin_memory=torch.cuda.is_available(),
    )
    criterion = nn.CrossEntropyLoss()
    scaler = GradScaler(device=device.type, enabled=False)
    return run_epoch(
        model_bundle=model_bundle,
        loader=test_loader,
        criterion=criterion,
        optimizer=None,
        scaler=scaler,
        device=device,
        amp_enabled=bool(config["training"].get("amp", True) and device.type == "cuda"),
        config=config,
        epoch_index=-1,
        mode="val",
    )


def main() -> int:
    args = parse_args()
    config = merge_overrides(load_config(args.config), args)
    summary = train(config)
    output_dir = Path(config["output_dir"])

    val_metrics_path = output_dir / "metrics_val.json"
    if not val_metrics_path.exists():
        final_metrics_path = output_dir / "final_metrics.json"
        if final_metrics_path.exists():
            val_metrics_path.write_text(final_metrics_path.read_text(encoding="utf-8"), encoding="utf-8")

    test_metrics = evaluate_test(config, output_dir / "best.pt")
    (output_dir / "metrics_test.json").write_text(json.dumps(test_metrics, indent=2), encoding="utf-8")

    summary_payload = {
        "experiment_name": config.get("augmentation", {}).get("name", Path(config["output_dir"]).name),
        "augmentation": config.get("augmentation", {}).get("name", "baseline"),
        "val_top1": summary["final_val_top1"],
        "val_top5": summary["final_val_top5"],
        "test_top1": test_metrics["top1"],
        "test_top5": test_metrics["top5"],
        "train_time_sec": summary["train_total_time_sec"],
        "notes": "single-clip EgoVideo frozen-backbone augmentation ablation",
    }
    (output_dir / "experiment_summary.json").write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")
    print(json.dumps(summary_payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
