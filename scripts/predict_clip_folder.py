#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts._demo_ready_common import (
    BUNDLE_PATH,
    load_bundle,
    load_reranker_runtime,
    rerank_sequence_predictions_from_loaded,
)
from scripts.predict_single_clip_raw import (
    DEFAULT_CHECKPOINT,
    DEFAULT_CONFIG,
    load_action_names,
    load_raw_model,
    load_sampled_frames,
    predict_raw_probs_from_loaded,
    predict_raw_probs_from_sampled_frames,
    resolve_session_dir,
)


def sync_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run sequential action prediction over one cropped_clips session folder.")
    parser.add_argument("folder", type=str, help="Session folder path or folder name under data/egtea_gaze_plus/cropped_clips")
    parser.add_argument("--bundle", type=Path, default=BUNDLE_PATH)
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--json", action="store_true", help="Print full JSON instead of the compact text format.")
    parser.add_argument(
        "--ultra-short",
        action="store_true",
        help="Print one line per clip with only the top-1 prediction.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    folder_path = resolve_session_dir(args.folder)
    clip_paths = sorted(folder_path.glob("*.mp4"))
    if not clip_paths:
        raise RuntimeError(f"No .mp4 clips found in {folder_path}")

    if args.bundle.exists():
        bundle = load_bundle(args.bundle)
        runtime = load_reranker_runtime(device=device, bundle=bundle)
        sync_device(device)
        total_start_time = time.perf_counter()
        rows = []
        for clip_path in tqdm(clip_paths, desc="Predicting clips", unit="clip"):
            sync_device(device)
            clip_start_time = time.perf_counter()
            clip_rows = rerank_sequence_predictions_from_loaded(clip_paths=[clip_path], device=device, runtime=runtime)
            sync_device(device)
            elapsed_ms = (time.perf_counter() - clip_start_time) * 1000.0
            row = clip_rows[0]
            row["elapsed_ms"] = round(elapsed_ms, 3)
            row["stream_elapsed_ms"] = round(elapsed_ms, 3)
            rows.append(row)
        mode = "reranked"
    else:
        action_names = load_action_names()
        raw_config, raw_model_bundle = load_raw_model(
            checkpoint_path=DEFAULT_CHECKPOINT,
            config_path=DEFAULT_CONFIG,
            device=device,
        )
        sync_device(device)
        total_start_time = time.perf_counter()
        stream_total_start_time = time.perf_counter()
        rows = []
        stream_total_elapsed_ms = 0.0
        for clip_path in tqdm(clip_paths, desc="Predicting clips", unit="clip"):
            sampled_frames = load_sampled_frames(clip_path, raw_config)
            sync_device(device)
            clip_start_time = time.perf_counter()
            probs = predict_raw_probs_from_loaded(
                clip_path=clip_path,
                config=raw_config,
                model_bundle=raw_model_bundle,
                device=device,
            )
            sync_device(device)
            elapsed_ms = (time.perf_counter() - clip_start_time) * 1000.0
            sync_device(device)
            stream_clip_start_time = time.perf_counter()
            stream_probs = predict_raw_probs_from_sampled_frames(
                sampled_frames=sampled_frames,
                config=raw_config,
                model_bundle=raw_model_bundle,
                device=device,
            )
            sync_device(device)
            stream_elapsed_ms = (time.perf_counter() - stream_clip_start_time) * 1000.0
            stream_total_elapsed_ms += stream_elapsed_ms
            top5_probs, top5_ids = torch.topk(probs, k=5)
            pred_id = int(top5_ids[0].item())
            rows.append(
                {
                    "clip": clip_path.name,
                    "pred_action_id": pred_id,
                    "pred_action_label": action_names[pred_id],
                    "pred_state_name": "raw_only",
                    "elapsed_ms": round(elapsed_ms, 3),
                    "stream_elapsed_ms": round(stream_elapsed_ms, 3),
                    "top5": [
                        {
                            "rank": rank + 1,
                            "action_id": int(top5_ids[rank].item()),
                            "action_label": action_names[int(top5_ids[rank].item())],
                            "probability": float(top5_probs[rank].item()),
                        }
                        for rank in range(5)
                    ],
                }
            )
        mode = "raw_fallback"
    sync_device(device)
    total_elapsed_ms = (time.perf_counter() - total_start_time) * 1000.0
    avg_elapsed_ms = total_elapsed_ms / max(len(rows), 1)
    if mode == "reranked":
        stream_total_elapsed_ms = total_elapsed_ms
    stream_avg_elapsed_ms = stream_total_elapsed_ms / max(len(rows), 1)
    payload = {
        "session_folder": str(folder_path),
        "num_clips": len(rows),
        "mode": mode,
        "total_elapsed_ms": round(total_elapsed_ms, 3),
        "avg_elapsed_ms": round(avg_elapsed_ms, 3),
        "stream_total_elapsed_ms": round(stream_total_elapsed_ms, 3),
        "stream_avg_elapsed_ms": round(stream_avg_elapsed_ms, 3),
        "results": rows,
    }
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    if args.json:
        print(json.dumps(payload, indent=2))
        return 0

    print(f"session: {folder_path.name}")
    print(f"clips: {len(rows)}")
    print(f"time_total: {total_elapsed_ms:.3f} ms")
    print(f"time_avg_per_clip: {avg_elapsed_ms:.3f} ms")
    print(f"time_stream_total: {stream_total_elapsed_ms:.3f} ms")
    print(f"time_stream_avg_per_clip: {stream_avg_elapsed_ms:.3f} ms")
    if args.ultra_short:
        for row in rows:
            print(
                f"{row['clip']}: {row['pred_action_label']} [{row['pred_state_name']}] "
                f"(full {row['elapsed_ms']:.3f} ms | stream {row['stream_elapsed_ms']:.3f} ms)"
            )
        return 0

    for row in rows:
        print(
            f"{row['clip']}: {row['pred_action_label']} [{row['pred_state_name']}] "
            f"(full {row['elapsed_ms']:.3f} ms | stream {row['stream_elapsed_ms']:.3f} ms)"
        )
        for top in row["top5"]:
            print(f"  {top['rank']}. {top['action_label']} ({top['probability']:.4f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
