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

from datasets.egtea_transition_reranker_dataset import EgteaTransitionRerankerDataset
from models.transition_aware_reranker import LearnedTransitionAwareReranker


def compute_transition_scores(
    prior_log_probs: torch.Tensor,
    prev_action_ids: torch.Tensor,
    prev_action_mask: torch.Tensor,
    candidate_ids: torch.Tensor,
    prev_mode: str,
) -> torch.Tensor:
    batch_size, candidate_k = candidate_ids.shape
    scores = torch.zeros((batch_size, candidate_k), device=candidate_ids.device, dtype=torch.float32)
    if prev_mode == "prev1":
        prev_ids = prev_action_ids[:, -1]
        valid = prev_action_mask[:, -1]
        if valid.any():
            scores[valid] = prior_log_probs[prev_ids[valid]].gather(1, candidate_ids[valid])
        return scores

    valid_counts = prev_action_mask.sum(dim=1)
    for batch_index in range(batch_size):
        count = int(valid_counts[batch_index].item())
        if count <= 0:
            continue
        use_count = min(3, count)
        ids = prev_action_ids[batch_index, -use_count:]
        gathered = prior_log_probs[ids].gather(1, candidate_ids[batch_index].unsqueeze(0).expand(use_count, -1))
        scores[batch_index] = gathered.mean(dim=0)
    return scores


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a learned transition-aware Top-k reranker.")
    parser.add_argument("--dump-dir", type=Path, default=ROOT / "outputs" / "reranker" / "candidate_dumps")
    parser.add_argument("--prior-path", type=Path, default=ROOT / "outputs" / "transition_reranker" / "priors" / "train_internal_transition_priors.pt")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "transition_reranker" / "top5_learned")
    parser.add_argument("--train-split", type=str, default="train_internal")
    parser.add_argument("--val-split", type=str, default="val_internal")
    parser.add_argument("--candidate-k", type=int, default=5, choices=[5, 10])
    parser.add_argument("--history-len", type=int, default=5)
    parser.add_argument("--prev-mode", type=str, default="prev1", choices=["prev1", "prev3"])
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
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


def evaluate(model: LearnedTransitionAwareReranker, loader: DataLoader, prior_log_probs: torch.Tensor, candidate_k: int, prev_mode: str, device: torch.device) -> dict:
    model.eval()
    total_samples = 0
    total_hits = 0
    raw_top1_correct = 0
    reranked_top1_correct = 0
    reranked_top5_correct = 0
    corrected = 0
    worsened = 0
    corrected_examples: list[dict] = []
    worsened_examples: list[dict] = []

    with torch.no_grad():
        for batch in loader:
            history_embeddings = batch["history_embeddings"].to(device)
            history_mask = batch["history_mask"].to(device)
            current_embedding = batch["current_embedding"].to(device)
            candidate_ids = batch["candidate_ids"].to(device)
            candidate_scores = batch["candidate_scores"].to(device)
            candidate_probs = batch["candidate_probs"].to(device)
            labels = batch["label"].to(device=device, dtype=torch.long)
            target_pos = batch["target_pos"].to(device=device, dtype=torch.long)
            prev_action_ids = batch["prev_action_ids"].to(device=device, dtype=torch.long)
            prev_action_mask = batch["prev_action_mask"].to(device)
            prev_action_id = batch["prev_action_id"].to(device=device, dtype=torch.long)
            hit_mask = target_pos >= 0

            transition_scores = compute_transition_scores(
                prior_log_probs=prior_log_probs,
                prev_action_ids=prev_action_ids,
                prev_action_mask=prev_action_mask,
                candidate_ids=candidate_ids,
                prev_mode=prev_mode,
            )
            rerank_scores = model(
                history_embeddings=history_embeddings,
                history_mask=history_mask,
                current_embedding=current_embedding,
                candidate_ids=candidate_ids,
                candidate_scores=candidate_scores,
                candidate_probs=candidate_probs,
                prev_action_id=prev_action_id,
                prev_action_ids=prev_action_ids,
                prev_action_mask=prev_action_mask,
                prev_mode=prev_mode,
                transition_prior_scores=transition_scores,
            )
            reranked_order = rerank_scores.argsort(dim=1, descending=True)
            reranked_ids = candidate_ids.gather(1, reranked_order)
            final_predictions = torch.where(hit_mask, reranked_ids[:, 0], candidate_ids[:, 0])
            raw_top1 = candidate_ids[:, 0]

            total_samples += labels.size(0)
            total_hits += int(hit_mask.sum().item())
            raw_top1_correct += (raw_top1 == labels).sum().item()
            reranked_top1_correct += (final_predictions == labels).sum().item()
            reranked_top5_correct += (reranked_ids[:, : min(5, candidate_k)] == labels.unsqueeze(1)).any(dim=1).sum().item()

            for idx in range(labels.size(0)):
                raw_ok = int(raw_top1[idx].item() == labels[idx].item())
                rerank_ok = int(final_predictions[idx].item() == labels[idx].item())
                if rerank_ok and not raw_ok:
                    corrected += 1
                    if len(corrected_examples) < 10:
                        corrected_examples.append(
                            {
                                "clip_stem": batch["clip_stem"][idx],
                                "session_id": batch["session_id"][idx],
                                "clip_idx": int(batch["clip_idx"][idx]),
                                "label": int(labels[idx].item()),
                                "raw_top1": int(raw_top1[idx].item()),
                                "reranked_top1": int(final_predictions[idx].item()),
                            }
                        )
                if raw_ok and not rerank_ok:
                    worsened += 1
                    if len(worsened_examples) < 10:
                        worsened_examples.append(
                            {
                                "clip_stem": batch["clip_stem"][idx],
                                "session_id": batch["session_id"][idx],
                                "clip_idx": int(batch["clip_idx"][idx]),
                                "label": int(labels[idx].item()),
                                "raw_top1": int(raw_top1[idx].item()),
                                "reranked_top1": int(final_predictions[idx].item()),
                            }
                        )

    return {
        "coverage": total_hits * 100.0 / max(total_samples, 1),
        "raw_top1": raw_top1_correct * 100.0 / max(total_samples, 1),
        "raw_top5": total_hits * 100.0 / max(total_samples, 1),
        "reranked_top1": reranked_top1_correct * 100.0 / max(total_samples, 1),
        "reranked_top5": reranked_top5_correct * 100.0 / max(total_samples, 1),
        "gain_vs_raw": (reranked_top1_correct - raw_top1_correct) * 100.0 / max(total_samples, 1),
        "corrected_count": corrected,
        "worsened_count": worsened,
        "corrected_examples": corrected_examples,
        "worsened_examples": worsened_examples,
    }


def main() -> int:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    save_json(args.output_dir / "used_config.json", vars(args))
    save_json(args.output_dir / "resolved_config.json", vars(args))

    train_dataset = EgteaTransitionRerankerDataset(
        dump_path=args.dump_dir / f"{args.train_split}.pt",
        history_len=args.history_len,
        candidate_k=args.candidate_k,
        hit_only=True,
    )
    val_dataset = EgteaTransitionRerankerDataset(
        dump_path=args.dump_dir / f"{args.val_split}.pt",
        history_len=args.history_len,
        candidate_k=args.candidate_k,
        hit_only=False,
    )
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    prior_payload = torch.load(args.prior_path, map_location="cpu", weights_only=False)
    prior_log_probs = prior_payload["transition_log_probs"].float().to(device)

    model = LearnedTransitionAwareReranker(
        num_classes=train_dataset.num_classes,
        embedding_dim=train_dataset.embedding_dim,
    ).to(device)
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()

    save_json(
        args.output_dir / "dataset_summary.json",
        {
            "candidate_k": args.candidate_k,
            "history_len": args.history_len,
            "prev_mode": args.prev_mode,
            "train_coverage": train_dataset.coverage * 100.0,
            "val_coverage": val_dataset.coverage * 100.0,
        },
    )

    best_state = None
    best_epoch = 0
    best_val = -1.0
    no_improve = 0
    log_path = args.output_dir / "train_log.jsonl"
    started = time.perf_counter()

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total_examples = 0
        progress = tqdm(train_loader, desc=f"Train {epoch}", leave=True)
        for batch in progress:
            history_embeddings = batch["history_embeddings"].to(device)
            history_mask = batch["history_mask"].to(device)
            current_embedding = batch["current_embedding"].to(device)
            candidate_ids = batch["candidate_ids"].to(device)
            candidate_scores = batch["candidate_scores"].to(device)
            candidate_probs = batch["candidate_probs"].to(device)
            target_pos = batch["target_pos"].to(device=device, dtype=torch.long)
            prev_action_ids = batch["prev_action_ids"].to(device=device, dtype=torch.long)
            prev_action_mask = batch["prev_action_mask"].to(device)
            prev_action_id = batch["prev_action_id"].to(device=device, dtype=torch.long)

            transition_scores = compute_transition_scores(
                prior_log_probs=prior_log_probs,
                prev_action_ids=prev_action_ids,
                prev_action_mask=prev_action_mask,
                candidate_ids=candidate_ids,
                prev_mode=args.prev_mode,
            )

            optimizer.zero_grad(set_to_none=True)
            rerank_scores = model(
                history_embeddings=history_embeddings,
                history_mask=history_mask,
                current_embedding=current_embedding,
                candidate_ids=candidate_ids,
                candidate_scores=candidate_scores,
                candidate_probs=candidate_probs,
                prev_action_id=prev_action_id,
                prev_action_ids=prev_action_ids,
                prev_action_mask=prev_action_mask,
                prev_mode=args.prev_mode,
                transition_prior_scores=transition_scores,
            )
            loss = criterion(rerank_scores, target_pos)
            loss.backward()
            optimizer.step()

            batch_size = target_pos.size(0)
            total_loss += loss.item() * batch_size
            total_examples += batch_size
            progress.set_postfix(loss=f"{total_loss / max(total_examples, 1):.4f}")

        val_metrics = evaluate(
            model=model,
            loader=val_loader,
            prior_log_probs=prior_log_probs,
            candidate_k=args.candidate_k,
            prev_mode=args.prev_mode,
            device=device,
        )
        append_jsonl(log_path, {"epoch": epoch, "train_loss": total_loss / max(total_examples, 1), "val": val_metrics})
        torch.save({"model_state_dict": model.state_dict(), "epoch": epoch, "args": vars(args)}, args.output_dir / "latest.pth")

        if val_metrics["reranked_top1"] > best_val:
            best_val = val_metrics["reranked_top1"]
            best_epoch = epoch
            best_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
            torch.save({"model_state_dict": best_state, "epoch": epoch, "args": vars(args)}, args.output_dir / "best.pth")
            no_improve = 0
        else:
            no_improve += 1

        print(
            f"Epoch {epoch} | train_loss={total_loss / max(total_examples, 1):.4f} | "
            f"val_reranked_top1={val_metrics['reranked_top1']:.2f} | gain_vs_raw={val_metrics['gain_vs_raw']:.2f}"
        )
        if no_improve >= args.patience:
            print(f"Early stopping at epoch {epoch} after {no_improve} epochs without improvement.")
            break

    total_time = time.perf_counter() - started
    if best_state is not None:
        model.load_state_dict(best_state)
    final_val = evaluate(
        model=model,
        loader=val_loader,
        prior_log_probs=prior_log_probs,
        candidate_k=args.candidate_k,
        prev_mode=args.prev_mode,
        device=device,
    )
    summary = {
        "model": "transition_reranker_learned",
        "candidate_k": args.candidate_k,
        "context_K": args.history_len,
        "best_epoch": best_epoch,
        "train_time_sec": total_time,
        "val_top1": final_val["reranked_top1"],
        "val_top5": final_val["reranked_top5"],
        "raw_top1": final_val["raw_top1"],
        "raw_top5": final_val["raw_top5"],
        "gain_vs_raw": final_val["gain_vs_raw"],
        "coverage": final_val["coverage"],
        "notes": f"prev_mode={args.prev_mode}",
    }
    save_json(args.output_dir / "metrics_val.json", final_val)
    save_json(args.output_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
