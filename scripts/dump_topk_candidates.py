#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from datasets.egtea_dataset import ClipRecord, EgteaVideoDataset
from datasets.egtea_sequence_manifest import build_sequence_records
from models.build_model import build_model


NUM_CLASSES = 106


DEFAULT_CONFIG = ROOT / "configs" / "egtea_egovideo_singleclip.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dump EgoVideo embeddings/logits/top-k candidates for reranking.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "reranker" / "candidate_dumps")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--split-id", type=int, default=1)
    parser.add_argument("--splits", nargs="+", default=["train_internal", "val_internal", "test"])
    return parser.parse_args()


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def build_dataset(records: list, config: dict) -> EgteaVideoDataset:
    clip_records = [
        ClipRecord(
            clip_stem=record.clip_stem,
            video_session=record.session_id,
            video_path=record.video_path,
            label_id=record.label,
        )
        for record in records
    ]
    return EgteaVideoDataset(
        records=clip_records,
        num_frames=int(config["model"]["num_frames"]),
        resize_size=int(config["data"]["resize_size"]),
        crop_size=int(config["data"]["crop_size"]),
        mean=list(config["data"]["mean"]),
        std=list(config["data"]["std"]),
        is_train=False,
    )


def dump_split(args: argparse.Namespace, config: dict, model_bundle, split_name: str) -> dict:
    records = build_sequence_records(
        data_root=Path(config["data"]["root"]),
        split_name=split_name,
        split_id=args.split_id,
        manifest_split_name=split_name,
    )
    dataset = build_dataset(records, config)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    device = torch.device(args.device)
    embeddings_list: list[torch.Tensor] = []
    logits_list: list[torch.Tensor] = []
    clip_stems: list[str] = []
    labels: list[int] = []

    with torch.no_grad():
        for batch in tqdm(loader, desc=f"Dump {split_name}", leave=True):
            videos = model_bundle.prepare_inputs(batch["video"], config).to(device, non_blocking=True)
            logits, embeddings = model_bundle.model.forward_with_features(videos)
            embeddings_list.append(embeddings.detach().cpu())
            logits_list.append(logits.detach().cpu())
            clip_stems.extend(list(batch["clip_stem"]))
            labels.extend([int(x) for x in batch["label"]])

    embeddings = torch.cat(embeddings_list, dim=0)
    logits = torch.cat(logits_list, dim=0)
    probabilities = logits.softmax(dim=1)
    top10_probs, top10_ids = probabilities.topk(10, dim=1)
    top10_scores = logits.gather(1, top10_ids)
    top5_ids = top10_ids[:, :5]
    top5_scores = top10_scores[:, :5]
    top5_probs = top10_probs[:, :5]

    stem_to_meta = {record.clip_stem: record for record in records}
    ordered_records = [stem_to_meta[clip_stem] for clip_stem in clip_stems]

    payload = {
        "split": split_name,
        "num_classes": NUM_CLASSES,
        "embeddings": embeddings,
        "logits": logits,
        "labels": torch.tensor(labels, dtype=torch.long),
        "clip_stems": clip_stems,
        "session_ids": [record.session_id for record in ordered_records],
        "clip_indices": torch.tensor([record.clip_idx for record in ordered_records], dtype=torch.long),
        "top1_ids": top10_ids[:, 0].clone(),
        "top5_ids": top5_ids,
        "top5_scores": top5_scores,
        "top5_probs": top5_probs,
        "top10_ids": top10_ids,
        "top10_scores": top10_scores,
        "top10_probs": top10_probs,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.output_dir / f"{split_name}.pt")

    labels_tensor = payload["labels"]
    top5_hit = (top5_ids == labels_tensor.unsqueeze(1)).any(dim=1).float().mean().item() * 100.0
    top10_hit = (top10_ids == labels_tensor.unsqueeze(1)).any(dim=1).float().mean().item() * 100.0
    top1 = (payload["top1_ids"] == labels_tensor).float().mean().item() * 100.0
    summary = {
        "split": split_name,
        "num_samples": len(labels),
        "top1": top1,
        "top5": top5_hit,
        "top10": top10_hit,
        "output_path": str(args.output_dir / f"{split_name}.pt"),
    }
    (args.output_dir / f"{split_name}_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    config["device"] = args.device

    device = torch.device(args.device)
    model_bundle = build_model(config, num_classes=NUM_CLASSES)
    model_bundle.model.to(device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model_bundle.model.load_state_dict(checkpoint["model_state_dict"])
    model_bundle.model.eval()

    summaries = [dump_split(args, config, model_bundle, split_name) for split_name in args.splits]
    print(json.dumps(summaries, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
