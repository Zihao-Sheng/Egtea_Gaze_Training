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

from scripts._demo_ready_common import (
    BUNDLE_PATH,
    load_action_mapping,
    load_bundle,
    predict_action_logits,
    resolve_clip_path,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict top-5 actions for one EGTEA clip.")
    parser.add_argument("clip", type=str, help="Clip filename or path under data/egtea_gaze_plus/cropped_clips")
    parser.add_argument("--bundle", type=Path, default=BUNDLE_PATH)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bundle = load_bundle(args.bundle)
    clip_path = resolve_clip_path(args.clip)
    action_names, _, _ = load_action_mapping()
    prediction = predict_action_logits(clip_path, device=device, bundle=bundle)

    probs = prediction["probs"][0]
    top5_probs, top5_ids = torch.topk(probs, k=5)
    payload = {
        "clip": clip_path.name,
        "resolved_path": str(clip_path),
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
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
