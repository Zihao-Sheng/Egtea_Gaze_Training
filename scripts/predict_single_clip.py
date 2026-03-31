#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts._demo_ready_common import (
    BUNDLE_PATH,
    load_action_mapping,
    load_action_model,
    load_bundle,
    predict_action_logits_from_loaded,
    predict_action_logits_from_sampled_frames,
    resolve_clip_path,
)
from scripts.predict_single_clip_raw import (
    DEFAULT_CHECKPOINT,
    DEFAULT_CONFIG,
    load_action_names,
    load_raw_model,
    load_sampled_frames,
    predict_raw_probs_from_loaded,
    predict_raw_probs_from_sampled_frames,
)


def sync_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict top-5 actions for one EGTEA clip.")
    parser.add_argument("clip", type=str, help="Clip filename or path under data/egtea_gaze_plus/cropped_clips")
    parser.add_argument("--bundle", type=Path, default=BUNDLE_PATH)
    parser.add_argument("--json", action="store_true", help="Print full JSON instead of the compact 5-line format.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    clip_path = resolve_clip_path(args.clip)
    mode = "reranked" if args.bundle.exists() else "raw_fallback"
    if args.bundle.exists():
        bundle = load_bundle(args.bundle)
        action_names, _, _ = load_action_mapping()
        config, model_bundle = load_action_model(device=device, bundle=bundle)
        sampled_frames = load_sampled_frames(clip_path, config)
        sync_device(device)
        start_time = time.perf_counter()
        prediction = predict_action_logits_from_loaded(
            clip_path,
            device=device,
            config=config,
            model_bundle=model_bundle,
        )
        probs = prediction["probs"][0]
        sync_device(device)
        stream_start_time = time.perf_counter()
        stream_prediction = predict_action_logits_from_sampled_frames(
            sampled_frames,
            device=device,
            config=config,
            model_bundle=model_bundle,
        )
        probs = stream_prediction["probs"][0]
    else:
        raw_config, raw_model_bundle = load_raw_model(
            checkpoint_path=DEFAULT_CHECKPOINT,
            config_path=DEFAULT_CONFIG,
            device=device,
        )
        action_names = load_action_names()
        sampled_frames = load_sampled_frames(clip_path, raw_config)
        sync_device(device)
        start_time = time.perf_counter()
        probs = predict_raw_probs_from_loaded(
            clip_path=clip_path,
            config=raw_config,
            model_bundle=raw_model_bundle,
            device=device,
        )
    sync_device(device)
    elapsed_ms = (time.perf_counter() - start_time) * 1000.0
    if not args.bundle.exists():
        sync_device(device)
        stream_start_time = time.perf_counter()
        probs = predict_raw_probs_from_sampled_frames(
            sampled_frames=sampled_frames,
            config=raw_config,
            model_bundle=raw_model_bundle,
            device=device,
        )
    sync_device(device)
    stream_elapsed_ms = (time.perf_counter() - stream_start_time) * 1000.0
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
            "mode": mode,
            "elapsed_ms": round(elapsed_ms, 3),
            "stream_elapsed_ms": round(stream_elapsed_ms, 3),
            "top5": rows,
        }
        print(json.dumps(payload, indent=2))
        return 0
    for row in rows:
        print(f"{row['rank']}. {row['action_label']} ({row['probability']:.4f})")
    print(f"time_full: {elapsed_ms:.3f} ms")
    print(f"time_stream_only: {stream_elapsed_ms:.3f} ms")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
