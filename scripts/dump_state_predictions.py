#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.state_classifier import StateClassifier


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dump state predictions aligned with action candidate dumps.")
    parser.add_argument("--dump-dir", type=Path, default=ROOT / "outputs" / "reranker" / "candidate_dumps")
    parser.add_argument("--taxonomy-version", type=str, choices=["v1", "v2"], default="v1")
    parser.add_argument("--mapping-path", type=Path, default=None)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--splits", nargs="+", default=["train_internal", "val_internal", "test"])
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def main() -> int:
    args = parse_args()
    if args.mapping_path is None:
        args.mapping_path = ROOT / "outputs" / ("state_model_v2" if args.taxonomy_version == "v2" else "state_model") / (
            "action_to_state_v2.json" if args.taxonomy_version == "v2" else "action_to_state.json"
        )
    if args.checkpoint is None:
        args.checkpoint = ROOT / "outputs" / ("state_model_v2" if args.taxonomy_version == "v2" else "state_model") / "state_single_mlp_h3" / "best.pth"
    if args.output_dir is None:
        args.output_dir = ROOT / "outputs" / ("state_action_reranker_v2" if args.taxonomy_version == "v2" else "state_action_reranker") / "state_predictions"
    mapping_payload = json.loads(args.mapping_path.read_text(encoding="utf-8"))
    state_names = list(mapping_payload["state_names"])
    num_states = int(mapping_payload["num_states"])
    action_to_state = {int(action_id): int(info["state_id"]) for action_id, info in mapping_payload["action_to_state"].items()}

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    input_dim = int(checkpoint["model_state_dict"]["mlp.0.weight"].shape[1])
    device = torch.device(args.device)
    model = StateClassifier(input_dim=input_dim, num_states=num_states).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for split in args.splits:
        payload = torch.load(args.dump_dir / f"{split}.pt", map_location="cpu", weights_only=False)
        embeddings = payload["embeddings"].float()
        loader = DataLoader(TensorDataset(embeddings), batch_size=args.batch_size, shuffle=False)
        logits_chunks: list[torch.Tensor] = []
        probs_chunks: list[torch.Tensor] = []
        pred_chunks: list[torch.Tensor] = []
        with torch.no_grad():
            for (batch_embeddings,) in loader:
                batch_embeddings = batch_embeddings.to(device)
                logits = model(batch_embeddings)
                probs = torch.softmax(logits, dim=1)
                preds = probs.argmax(dim=1)
                logits_chunks.append(logits.cpu())
                probs_chunks.append(probs.cpu())
                pred_chunks.append(preds.cpu())
        state_logits = torch.cat(logits_chunks, dim=0)
        state_probs = torch.cat(probs_chunks, dim=0)
        pred_state_ids = torch.cat(pred_chunks, dim=0)
        true_state_labels = torch.tensor([action_to_state[int(label)] for label in payload["labels"].tolist()], dtype=torch.long)
        output_payload = {
            "split": split,
            "state_names": state_names,
            "num_states": num_states,
            "session_ids": list(payload["session_ids"]),
            "clip_indices": payload["clip_indices"].long(),
            "clip_stems": list(payload["clip_stems"]),
            "pred_state_ids": pred_state_ids.long(),
            "state_probs": state_probs.float(),
            "state_logits": state_logits.float(),
            "true_state_labels": true_state_labels,
            "true_action_labels": payload["labels"].long(),
        }
        output_path = args.output_dir / f"{split}.pt"
        torch.save(output_payload, output_path)
        save_json(
            args.output_dir / f"{split}_summary.json",
            {
                "split": split,
                "taxonomy_version": args.taxonomy_version,
                "num_samples": int(len(pred_state_ids)),
                "top1": float((pred_state_ids == true_state_labels).float().mean().item() * 100.0),
                "checkpoint": str(args.checkpoint),
                "output_path": str(output_path),
            },
        )
        print(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
