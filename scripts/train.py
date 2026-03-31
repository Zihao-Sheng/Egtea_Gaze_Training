#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import yaml
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from datasets.egtea_dataset import EgteaVideoDataset, build_split_records, export_split_manifest
from models.build_model import ModelBundle, build_model


NUM_CLASSES = 106


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train EGTEA RGB action models with a unified interface.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--train-max-samples", type=int, default=None)
    parser.add_argument("--val-max-samples", type=int, default=None)
    parser.add_argument("--resume-from", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--smoke-test", action="store_true")
    return parser.parse_args()


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def merge_overrides(config: dict, args: argparse.Namespace) -> dict:
    if args.output_dir is not None:
        config["output_dir"] = str(args.output_dir)
    if args.device is not None:
        config["device"] = args.device
    if args.epochs is not None:
        config["training"]["epochs"] = args.epochs
    if args.batch_size is not None:
        config["training"]["batch_size"] = args.batch_size
    if args.num_workers is not None:
        config["training"]["num_workers"] = args.num_workers
    if args.train_max_samples is not None:
        config["data"]["train_max_samples"] = args.train_max_samples
    if args.val_max_samples is not None:
        config["data"]["val_max_samples"] = args.val_max_samples
    if args.seed is not None:
        config["seed"] = args.seed
    if args.smoke_test:
        config["data"]["train_max_samples"] = 16
        config["data"]["val_max_samples"] = 8
        config["training"]["epochs"] = 1
    config["smoke_test"] = bool(args.smoke_test)
    config["resume_from"] = str(args.resume_from) if args.resume_from is not None else None
    return config


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_dataloaders(config: dict) -> tuple[DataLoader, DataLoader, list, list]:
    data_root = Path(config["data"]["root"])
    split_id = int(config["data"]["split_id"])
    train_split_name = str(config["data"].get("train_split_name", "train"))
    val_split_name = str(config["data"].get("val_split_name", "test"))
    train_split_file = config["data"].get("train_split_file")
    val_split_file = config["data"].get("val_split_file")
    train_records = build_split_records(
        data_root=data_root,
        split_id=split_id,
        split_name=train_split_name,
        max_samples=config["data"].get("train_max_samples"),
        seed=int(config["seed"]),
        split_file=Path(train_split_file) if train_split_file else None,
    )
    val_records = build_split_records(
        data_root=data_root,
        split_id=split_id,
        split_name=val_split_name,
        max_samples=config["data"].get("val_max_samples"),
        seed=int(config["seed"]) + 1,
        split_file=Path(val_split_file) if val_split_file else None,
    )

    export_split_manifest(train_records, Path(config["output_dir"]) / "train_manifest.json")
    export_split_manifest(val_records, Path(config["output_dir"]) / "val_manifest.json")

    train_dataset = EgteaVideoDataset(
        records=train_records,
        num_frames=int(config["model"]["num_frames"]),
        resize_size=int(config["data"]["resize_size"]),
        crop_size=int(config["data"]["crop_size"]),
        mean=list(config["data"]["mean"]),
        std=list(config["data"]["std"]),
        is_train=True,
    )
    val_dataset = EgteaVideoDataset(
        records=val_records,
        num_frames=int(config["model"]["num_frames"]),
        resize_size=int(config["data"]["resize_size"]),
        crop_size=int(config["data"]["crop_size"]),
        mean=list(config["data"]["mean"]),
        std=list(config["data"]["std"]),
        is_train=False,
    )

    batch_size = int(config["training"]["batch_size"])
    num_workers = int(config["training"]["num_workers"])
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return train_loader, val_loader, train_records, val_records


def accuracy_metrics(logits: torch.Tensor, targets: torch.Tensor) -> tuple[float, float]:
    topk = min(5, logits.shape[1])
    _, pred = logits.topk(topk, dim=1, largest=True, sorted=True)
    pred = pred.t()
    correct = pred.eq(targets.view(1, -1).expand_as(pred))

    top1 = correct[:1].reshape(-1).float().sum().item() * 100.0 / max(targets.numel(), 1)
    top5 = correct[:topk].reshape(-1).float().sum().item() * 100.0 / max(targets.numel(), 1)
    return top1, top5


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def append_jsonl(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload) + "\n")


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: AdamW,
    scheduler: CosineAnnealingLR,
    scaler: GradScaler,
    epoch: int,
    best_top1: float,
    config: dict,
) -> None:
    payload = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "epoch": epoch,
        "best_top1": best_top1,
        "config": config,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def load_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: AdamW,
    scheduler: CosineAnnealingLR,
    scaler: GradScaler,
    device: torch.device,
) -> tuple[int, float]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    scaler.load_state_dict(checkpoint["scaler_state_dict"])
    return int(checkpoint["epoch"]), float(checkpoint["best_top1"])


def run_epoch(
    model_bundle: ModelBundle,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: AdamW | None,
    scaler: GradScaler,
    device: torch.device,
    amp_enabled: bool,
    config: dict,
    epoch_index: int,
    mode: str,
) -> dict:
    is_train = mode == "train"
    model_bundle.model.train(is_train)

    total_loss = 0.0
    total_top1 = 0.0
    total_top5 = 0.0
    total_examples = 0

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    progress = tqdm(loader, desc=f"{mode.capitalize()} {epoch_index}", leave=True)
    for batch in progress:
        videos = batch["video"]
        labels = batch["label"].to(device, non_blocking=True)
        prepared_inputs = model_bundle.prepare_inputs(videos, config)

        if isinstance(prepared_inputs, list):
            prepared_inputs = [tensor.to(device, non_blocking=True) for tensor in prepared_inputs]
        else:
            prepared_inputs = prepared_inputs.to(device, non_blocking=True)

        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)

        with autocast(device_type=device.type, enabled=amp_enabled):
            logits = model_bundle.model(prepared_inputs)
            loss = criterion(logits, labels)

        if optimizer is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        batch_size = labels.size(0)
        top1, top5 = accuracy_metrics(logits.detach(), labels.detach())
        total_loss += loss.item() * batch_size
        total_top1 += top1 * batch_size
        total_top5 += top5 * batch_size
        total_examples += batch_size

        progress.set_postfix(
            loss=f"{total_loss / max(total_examples, 1):.4f}",
            top1=f"{total_top1 / max(total_examples, 1):.2f}",
            top5=f"{total_top5 / max(total_examples, 1):.2f}",
        )

    peak_memory_mb = 0.0
    if device.type == "cuda":
        peak_memory_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)

    return {
        "loss": total_loss / max(total_examples, 1),
        "top1": total_top1 / max(total_examples, 1),
        "top5": total_top5 / max(total_examples, 1),
        "peak_memory_mb": peak_memory_mb,
    }


def build_optimizer_and_scheduler(model: nn.Module, config: dict) -> tuple[AdamW, CosineAnnealingLR]:
    optimizer = AdamW(
        model.parameters(),
        lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=max(int(config["training"]["epochs"]), 1))
    return optimizer, scheduler


def maybe_autoscale_batch_size(
    config: dict,
    model_bundle: ModelBundle,
    device: torch.device,
    criterion: nn.Module,
) -> dict:
    if not config["training"].get("auto_scale_batch_size", False):
        return config

    while True:
        train_loader, _, _, _ = make_dataloaders(config)
        batch = next(iter(train_loader))
        videos = batch["video"]
        labels = batch["label"].to(device, non_blocking=True)
        prepared_inputs = model_bundle.prepare_inputs(videos, config)
        if isinstance(prepared_inputs, list):
            prepared_inputs = [tensor.to(device, non_blocking=True) for tensor in prepared_inputs]
        else:
            prepared_inputs = prepared_inputs.to(device, non_blocking=True)

        optimizer = torch.optim.SGD(model_bundle.model.parameters(), lr=1e-3)
        optimizer.zero_grad(set_to_none=True)
        try:
            with autocast(device_type=device.type, enabled=config["training"].get("amp", True)):
                logits = model_bundle.model(prepared_inputs)
                loss = criterion(logits, labels)
            loss.backward()
            optimizer.zero_grad(set_to_none=True)
            return config
        except RuntimeError as exc:
            if "out of memory" not in str(exc).lower() or device.type != "cuda":
                raise
            torch.cuda.empty_cache()
            current_batch_size = int(config["training"]["batch_size"])
            if current_batch_size <= 1:
                raise
            new_batch_size = max(current_batch_size // 2, 1)
            print(f"OOM during preflight. Reducing batch size from {current_batch_size} to {new_batch_size}.")
            config["training"]["batch_size"] = new_batch_size


def train(config: dict) -> dict:
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(int(config["seed"]))
    torch.hub.set_dir(str((ROOT / ".torch_cache").resolve()))

    device = torch.device(config["device"])
    print(f"Using device: {device}")

    model_bundle = build_model(config, num_classes=NUM_CLASSES)
    model_bundle.model.to(device)
    for note in model_bundle.notes:
        print(f"[model-note] {note}")

    criterion = nn.CrossEntropyLoss()
    config = maybe_autoscale_batch_size(config, model_bundle, device, criterion)
    train_loader, val_loader, train_records, val_records = make_dataloaders(config)

    optimizer, scheduler = build_optimizer_and_scheduler(model_bundle.model, config)
    scaler = GradScaler(device=device.type, enabled=bool(config["training"].get("amp", True) and device.type == "cuda"))

    start_epoch = 0
    best_top1 = -1.0
    resume_from = config.get("resume_from")
    if resume_from:
        start_epoch, best_top1 = load_checkpoint(
            Path(resume_from),
            model_bundle.model,
            optimizer,
            scheduler,
            scaler,
            device=device,
        )
        print(f"Resumed from {resume_from} at epoch {start_epoch}, best_top1={best_top1:.2f}")

    summary = {
        "model_name": config["model"]["name"],
        "split_id": int(config["data"]["split_id"]),
        "num_classes": NUM_CLASSES,
        "parameter_count": count_parameters(model_bundle.model),
        "macs_or_flops": "Not computed",
        "notes": list(model_bundle.notes),
        "train_samples": len(train_records),
        "val_samples": len(val_records),
        "effective_batch_size": int(config["training"]["batch_size"]),
        "clip_frames": int(config["model"]["num_frames"]),
        "crop_size": int(config["data"]["crop_size"]),
        "resize_size": int(config["data"]["resize_size"]),
    }

    total_start_time = time.perf_counter()
    epoch_durations: list[float] = []
    peak_memory_mb = 0.0
    best_checkpoint_path = output_dir / "best.pt"
    latest_checkpoint_path = output_dir / "latest.pt"
    log_path = output_dir / "train_val_log.jsonl"

    for epoch in range(start_epoch + 1, start_epoch + int(config["training"]["epochs"]) + 1):
        epoch_start_time = time.perf_counter()
        train_metrics = run_epoch(
            model_bundle=model_bundle,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            amp_enabled=bool(config["training"].get("amp", True) and device.type == "cuda"),
            config=config,
            epoch_index=epoch,
            mode="train",
        )
        with torch.no_grad():
            val_metrics = run_epoch(
                model_bundle=model_bundle,
                loader=val_loader,
                criterion=criterion,
                optimizer=None,
                scaler=scaler,
                device=device,
                amp_enabled=bool(config["training"].get("amp", True) and device.type == "cuda"),
                config=config,
                epoch_index=epoch,
                mode="val",
            )
        scheduler.step()

        peak_memory_mb = max(peak_memory_mb, train_metrics["peak_memory_mb"], val_metrics["peak_memory_mb"])
        epoch_duration = time.perf_counter() - epoch_start_time
        epoch_durations.append(epoch_duration)

        payload = {
            "epoch": epoch,
            "train": train_metrics,
            "val": val_metrics,
            "epoch_duration_sec": epoch_duration,
        }
        append_jsonl(log_path, payload)

        save_checkpoint(
            latest_checkpoint_path,
            model_bundle.model,
            optimizer,
            scheduler,
            scaler,
            epoch=epoch,
            best_top1=max(best_top1, val_metrics["top1"]),
            config=config,
        )

        if val_metrics["top1"] > best_top1:
            best_top1 = val_metrics["top1"]
            save_checkpoint(
                best_checkpoint_path,
                model_bundle.model,
                optimizer,
                scheduler,
                scaler,
                epoch=epoch,
                best_top1=best_top1,
                config=config,
            )

        print(
            f"Epoch {epoch} | "
            f"train_top1={train_metrics['top1']:.2f} | val_top1={val_metrics['top1']:.2f} | "
            f"val_top5={val_metrics['top5']:.2f} | epoch_time={epoch_duration:.1f}s"
        )

    total_duration = time.perf_counter() - total_start_time

    if best_checkpoint_path.exists():
        checkpoint = torch.load(best_checkpoint_path, map_location=device, weights_only=False)
        model_bundle.model.load_state_dict(checkpoint["model_state_dict"])

    with torch.no_grad():
        final_val_metrics = run_epoch(
            model_bundle=model_bundle,
            loader=val_loader,
            criterion=criterion,
            optimizer=None,
            scaler=scaler,
            device=device,
            amp_enabled=bool(config["training"].get("amp", True) and device.type == "cuda"),
            config=config,
            epoch_index=-1,
            mode="val",
        )

    summary.update(
        {
            "best_val_top1": best_top1,
            "final_val_top1": final_val_metrics["top1"],
            "final_val_top5": final_val_metrics["top5"],
            "train_total_time_sec": total_duration,
            "avg_epoch_time_sec": sum(epoch_durations) / max(len(epoch_durations), 1),
            "peak_memory_mb": peak_memory_mb,
        }
    )
    save_json(output_dir / "final_metrics.json", final_val_metrics)
    save_json(output_dir / "summary.json", summary)
    save_json(output_dir / "resolved_config.json", config)
    return summary


def main() -> int:
    args = parse_args()
    config = merge_overrides(load_config(args.config), args)
    summary = train(config)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
