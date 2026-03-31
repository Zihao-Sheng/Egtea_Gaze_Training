#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from datasets.egtea_dataset import (
    apply_spatial_transform,
    decode_video_rgb_frames,
    normalize_video,
    sample_uniform_indices,
)
from models.build_model import build_model


DEFAULT_CONFIG = ROOT / "configs" / "augmentation_ablation" / "rrc_flip.yaml"
DEFAULT_CHECKPOINT = ROOT / "outputs" / "demo_ready" / "default_pipeline" / "action_encoder" / "best.pt"
NUM_CLASSES = 106


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict top-5 actions for one EGTEA clip using the raw single-clip model.")
    parser.add_argument("clip", type=str, help="Clip filename or path under data/egtea_gaze_plus/cropped_clips")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--json", action="store_true", help="Print full JSON instead of the compact 5-line format.")
    return parser.parse_args()


def load_action_names() -> list[str]:
    mapping_path = ROOT / "data" / "egtea_gaze_plus" / "raw_annotations" / "cls_label_index.csv"
    rows: list[tuple[int, str]] = []
    for raw in mapping_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = [part.strip() for part in line.split(";")]
        action_id = int(parts[0])
        rows.append((action_id, parts[1]))
    rows.sort(key=lambda item: item[0])
    names = [name for _, name in rows]
    return names


def resolve_clip_path(user_value: str | Path) -> Path:
    path = Path(user_value)
    if path.exists():
        return path.resolve()
    search_name = path.name if path.suffix else f"{path.name}.mp4"
    matches = list((ROOT / "data" / "egtea_gaze_plus" / "cropped_clips").rglob(search_name))
    if not matches:
        raise FileNotFoundError(f"Could not find clip '{user_value}' under data/egtea_gaze_plus/cropped_clips")
    if len(matches) > 1:
        raise RuntimeError(f"Found multiple clips named '{search_name}'. Please pass a fuller path.")
    return matches[0].resolve()


def resolve_session_dir(user_value: str | Path) -> Path:
    path = Path(user_value)
    if path.exists() and path.is_dir():
        return path.resolve()
    candidate = ROOT / "data" / "egtea_gaze_plus" / "cropped_clips" / str(user_value)
    if candidate.exists() and candidate.is_dir():
        return candidate.resolve()
    raise FileNotFoundError(f"Could not find clip folder '{user_value}'.")


def preprocess_clip(video_path: Path, config: dict) -> torch.Tensor:
    decoded_frames = decode_video_rgb_frames(video_path)
    num_frames = int(config["model"]["num_frames"])
    frame_indices = sample_uniform_indices(len(decoded_frames), num_frames)
    sampled_frames = [decoded_frames[i] for i in frame_indices]
    return preprocess_sampled_frames(sampled_frames, config)


def load_sampled_frames(video_path: Path, config: dict) -> list[torch.Tensor]:
    decoded_frames = decode_video_rgb_frames(video_path)
    num_frames = int(config["model"]["num_frames"])
    frame_indices = sample_uniform_indices(len(decoded_frames), num_frames)
    return [decoded_frames[i] for i in frame_indices]


def preprocess_sampled_frames(sampled_frames: list[torch.Tensor], config: dict) -> torch.Tensor:
    video = torch.stack(sampled_frames, dim=0).permute(0, 3, 1, 2).float() / 255.0
    video = apply_spatial_transform(
        frames=video,
        resize_size=int(config["data"]["resize_size"]),
        crop_size=int(config["data"]["crop_size"]),
        is_train=False,
        augmentation=None,
    )
    video = normalize_video(video, list(config["data"]["mean"]), list(config["data"]["std"]))
    return video.unsqueeze(0)


def load_raw_model(checkpoint_path: Path, config_path: Path, device: torch.device):
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config["device"] = str(device)
    model_bundle = build_model(config, num_classes=NUM_CLASSES)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model_bundle.model.load_state_dict(checkpoint["model_state_dict"])
    model_bundle.model.to(device)
    model_bundle.model.eval()
    return config, model_bundle


def predict_raw_probs_from_loaded(
    clip_path: Path,
    config: dict,
    model_bundle,
    device: torch.device,
) -> torch.Tensor:
    video = preprocess_clip(clip_path, config).to(device)
    with torch.no_grad():
        logits = model_bundle.model(model_bundle.prepare_inputs(video, config))
        probs = logits.softmax(dim=1)[0].cpu()
    return probs


def predict_raw_probs_from_sampled_frames(
    sampled_frames: list[torch.Tensor],
    config: dict,
    model_bundle,
    device: torch.device,
) -> torch.Tensor:
    video = preprocess_sampled_frames(sampled_frames, config).to(device)
    with torch.no_grad():
        logits = model_bundle.model(model_bundle.prepare_inputs(video, config))
        probs = logits.softmax(dim=1)[0].cpu()
    return probs


def predict_raw_probs(
    clip_path: Path,
    checkpoint_path: Path,
    config_path: Path,
    device: torch.device,
) -> tuple[torch.Tensor, list[str]]:
    config, model_bundle = load_raw_model(checkpoint_path=checkpoint_path, config_path=config_path, device=device)
    action_names = load_action_names()
    probs = predict_raw_probs_from_loaded(
        clip_path=clip_path,
        config=config,
        model_bundle=model_bundle,
        device=device,
    )
    return probs, action_names


def main() -> int:
    args = parse_args()
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    clip_path = resolve_clip_path(args.clip)
    probs, action_names = predict_raw_probs(
        clip_path=clip_path,
        checkpoint_path=args.checkpoint,
        config_path=args.config,
        device=device,
    )

    top5_probs, top5_ids = torch.topk(probs, k=5)
    rows = [
        {
            "rank": rank + 1,
            "action_id": int(top5_ids[rank].item()),
            "action_label": action_names[int(top5_ids[rank].item())],
            "probability": float(top5_probs[rank].item()),
        }
        for rank in range(5)
    ]

    if args.json:
        import json

        payload = {
            "clip": clip_path.name,
            "resolved_path": str(clip_path),
            "checkpoint": str(args.checkpoint),
            "top5": rows,
        }
        print(json.dumps(payload, indent=2))
        return 0

    for row in rows:
        print(f"{row['rank']}. {row['action_label']} ({row['probability']:.4f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
