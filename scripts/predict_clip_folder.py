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

from scripts._demo_ready_common import BUNDLE_PATH, load_bundle, rerank_sequence_predictions, resolve_session_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run sequential action prediction over one cropped_clips session folder.")
    parser.add_argument("folder", type=str, help="Session folder path or folder name under data/egtea_gaze_plus/cropped_clips")
    parser.add_argument("--bundle", type=Path, default=BUNDLE_PATH)
    parser.add_argument("--output-json", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bundle = load_bundle(args.bundle)
    folder_path = resolve_session_dir(args.folder)
    clip_paths = sorted(folder_path.glob("*.mp4"))
    if not clip_paths:
        raise RuntimeError(f"No .mp4 clips found in {folder_path}")

    rows = rerank_sequence_predictions(clip_paths=clip_paths, device=device, bundle=bundle)
    payload = {
        "session_folder": str(folder_path),
        "num_clips": len(rows),
        "results": rows,
    }
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
