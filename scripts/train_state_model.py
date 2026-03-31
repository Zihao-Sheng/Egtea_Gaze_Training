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
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from datasets.egtea_state_dataset import EgteaStateDataset
from models.causal_state_model import CausalStateModel
from models.state_classifier import StateClassifier


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train EGTEA state models from EgoVideo embeddings.")
    parser.add_argument("--dump-dir", type=Path, default=ROOT / "outputs" / "reranker" / "candidate_dumps")
    parser.add_argument("--taxonomy-version", type=str, choices=["v1", "v2"], default="v1")
    parser.add_argument("--mapping-path", type=Path, default=None)
    parser.add_argument("--train-split", type=str, default="train_internal")
    parser.add_argument("--val-split", type=str, default="val_internal")
    parser.add_argument("--task-mode", type=str, choices=["single", "causal"], default="single")
    parser.add_argument("--model-type", type=str, choices=["mlp", "gru", "tcn"], default="mlp")
    parser.add_argument("--history-len", type=int, default=3)
    parser.add_argument("--include-logits", action="store_true")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def append_jsonl(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload) + "\n")


def top1_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    return (logits.argmax(dim=1) == labels).float().mean().item() * 100.0


def build_model(args: argparse.Namespace, input_dim: int, num_states: int) -> nn.Module:
    if args.task_mode == "single":
        return StateClassifier(input_dim=input_dim, num_states=num_states)
    return CausalStateModel(input_dim=input_dim, num_states=num_states, model_type=args.model_type)


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: AdamW | None,
    device: torch.device,
    task_mode: str,
    epoch: int,
    split_name: str,
) -> dict:
    train_mode = optimizer is not None
    model.train(train_mode)
    total_loss = 0.0
    total_acc = 0.0
    total_examples = 0
    progress = tqdm(loader, desc=f"{split_name.capitalize()} {epoch}", leave=True)
    for batch in progress:
        labels = batch["state_label"].to(device=device, dtype=torch.long)
        if task_mode == "single":
            logits = model(batch["features"].to(device))
        else:
            logits = model(batch["features"].to(device), batch["mask"].to(device))
        loss = criterion(logits, labels)

        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_acc += top1_accuracy(logits.detach(), labels.detach()) * batch_size
        total_examples += batch_size
        progress.set_postfix(loss=f"{total_loss / max(total_examples, 1):.4f}", top1=f"{total_acc / max(total_examples, 1):.2f}")
    return {"loss": total_loss / max(total_examples, 1), "top1": total_acc / max(total_examples, 1)}


def main() -> int:
    args = parse_args()
    set_seed(args.seed)
    if args.mapping_path is None:
        args.mapping_path = ROOT / "outputs" / ("state_model_v2" if args.taxonomy_version == "v2" else "state_model") / (
            "action_to_state_v2.json" if args.taxonomy_version == "v2" else "action_to_state.json"
        )
    run_name = f"state_{args.task_mode}_{args.model_type}_h{args.history_len}"
    output_root = ROOT / "outputs" / ("state_model_v2" if args.taxonomy_version == "v2" else "state_model")
    output_dir = args.output_dir or (output_root / run_name)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_json(output_dir / "resolved_config.json", vars(args))

    history_len = 1 if args.task_mode == "single" else args.history_len
    train_dataset = EgteaStateDataset(
        dump_path=args.dump_dir / f"{args.train_split}.pt",
        mapping_path=args.mapping_path,
        history_len=history_len,
        task_mode=args.task_mode,
        include_logits=args.include_logits,
        taxonomy_version=args.taxonomy_version,
    )
    val_dataset = EgteaStateDataset(
        dump_path=args.dump_dir / f"{args.val_split}.pt",
        mapping_path=args.mapping_path,
        history_len=history_len,
        task_mode=args.task_mode,
        include_logits=args.include_logits,
        taxonomy_version=args.taxonomy_version,
    )
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    device = torch.device(args.device)
    model = build_model(args, input_dim=train_dataset.feature_dim, num_states=train_dataset.num_states).to(device)
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()

    best_top1 = -1.0
    best_epoch = 0
    epochs_without_improvement = 0
    total_start = time.perf_counter()

    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(model, train_loader, criterion, optimizer, device, args.task_mode, epoch, "train")
        with torch.no_grad():
            val_metrics = run_epoch(model, val_loader, criterion, None, device, args.task_mode, epoch, "val")
        append_jsonl(output_dir / "train_log.jsonl", {"epoch": epoch, "train": train_metrics, "val": val_metrics})
        torch.save({"model_state_dict": model.state_dict(), "epoch": epoch, "config": vars(args)}, output_dir / "latest.pth")
        if val_metrics["top1"] > best_top1:
            best_top1 = val_metrics["top1"]
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save({"model_state_dict": model.state_dict(), "epoch": epoch, "config": vars(args)}, output_dir / "best.pth")
        else:
            epochs_without_improvement += 1
        print(f"Epoch {epoch} | train_loss={train_metrics['loss']:.4f} | val_loss={val_metrics['loss']:.4f} | val_top1={val_metrics['top1']:.2f}")
        if epochs_without_improvement >= args.patience:
            print(f"Early stopping at epoch {epoch}.")
            break

    total_time = time.perf_counter() - total_start
    best_checkpoint = torch.load(output_dir / "best.pth", map_location="cpu", weights_only=False)
    model.load_state_dict(best_checkpoint["model_state_dict"])
    model.to(device)
    with torch.no_grad():
        best_val = run_epoch(model, val_loader, criterion, None, device, args.task_mode, -1, "val")

    summary = {
        "model": f"state_{args.task_mode}_{args.model_type}",
        "task_type": "state",
        "taxonomy_version": args.taxonomy_version,
        "history_len": history_len,
        "best_epoch": best_epoch,
        "val_top1": best_val["top1"],
        "val_loss": best_val["loss"],
        "train_time_sec": total_time,
        "num_states": train_dataset.num_states,
        "notes": f"include_logits={args.include_logits}",
    }
    save_json(output_dir / "metrics_val.json", best_val)
    save_json(output_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
