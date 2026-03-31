#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build action transition priors from a train split dump.")
    parser.add_argument("--dump-dir", type=Path, default=ROOT / "outputs" / "reranker" / "candidate_dumps")
    parser.add_argument("--split", type=str, default="train_internal")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "transition_reranker" / "priors")
    parser.add_argument("--smoothing", type=float, default=1.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    payload = torch.load(args.dump_dir / f"{args.split}.pt", map_location="cpu", weights_only=False)
    labels = payload["labels"].long()
    session_ids = list(payload["session_ids"])
    clip_indices = payload["clip_indices"].long()
    num_classes = int(payload["num_classes"])

    session_to_positions: dict[str, list[int]] = {}
    for idx, session_id in enumerate(session_ids):
        session_to_positions.setdefault(session_id, []).append(idx)
    for session_id, positions in session_to_positions.items():
        session_to_positions[session_id] = sorted(positions, key=lambda pos: int(clip_indices[pos]))

    counts = torch.zeros((num_classes, num_classes), dtype=torch.float32)
    for positions in session_to_positions.values():
        for prev_pos, curr_pos in zip(positions[:-1], positions[1:]):
            counts[int(labels[prev_pos].item()), int(labels[curr_pos].item())] += 1.0

    smoothed_counts = counts + float(args.smoothing)
    transition_probs = smoothed_counts / smoothed_counts.sum(dim=1, keepdim=True)
    transition_log_probs = transition_probs.log()

    class_names: dict[int, str] = {}
    cls_path = ROOT / "data" / "egtea_gaze_plus" / "raw_annotations" / "cls_label_index.csv"
    with cls_path.open("r", encoding="utf-8") as handle:
        for raw in handle:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            parts = [part.strip() for part in line.split(";")]
            class_names[int(parts[0])] = parts[1]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_pt = args.output_dir / f"{args.split}_transition_priors.pt"
    torch.save(
        {
            "split": args.split,
            "smoothing": float(args.smoothing),
            "order": 1,
            "counts": counts,
            "smoothed_counts": smoothed_counts,
            "transition_probs": transition_probs,
            "transition_log_probs": transition_log_probs,
            "class_names": class_names,
        },
        out_pt,
    )

    common_transitions: list[dict] = []
    flat = counts.flatten()
    topk = torch.topk(flat, k=min(20, flat.numel()))
    for flat_index, value in zip(topk.indices.tolist(), topk.values.tolist()):
        if value <= 0:
            continue
        prev_id = flat_index // num_classes
        curr_id = flat_index % num_classes
        common_transitions.append(
            {
                "count": int(value),
                "prev_id": prev_id,
                "prev_name": class_names[prev_id],
                "curr_id": curr_id,
                "curr_name": class_names[curr_id],
            }
        )

    summary = {
        "split": args.split,
        "smoothing": float(args.smoothing),
        "order": 1,
        "num_classes": num_classes,
        "num_sessions": len(session_to_positions),
        "num_transitions": int(counts.sum().item()),
        "common_transitions": common_transitions,
        "output_pt": str(out_pt),
    }
    (args.output_dir / f"{args.split}_transition_priors.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
