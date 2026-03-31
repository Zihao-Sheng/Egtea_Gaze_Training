#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

BACKBONE_ROOT = ROOT / "third_party" / "EgoVideo-main" / "backbone"
CHECKPOINT_PATH = BACKBONE_ROOT / "ckpt_4frames.pth"
CHECKPOINT_URL = "https://drive.google.com/file/d/1k6f1eRdcL17IvXtdX_J8WxNbju2Ms3AW/view?usp=sharing"


def ensure_gdown() -> None:
    if importlib.util.find_spec("gdown") is not None:
        return
    subprocess.run([sys.executable, "-m", "pip", "install", "gdown"], check=True, cwd=ROOT)


def download_checkpoint() -> None:
    ensure_gdown()
    BACKBONE_ROOT.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            sys.executable,
            "-m",
            "gdown",
            "--fuzzy",
            CHECKPOINT_URL,
            "-O",
            str(CHECKPOINT_PATH),
        ],
        check=True,
        cwd=ROOT,
    )


def main() -> int:
    if not BACKBONE_ROOT.exists():
        raise RuntimeError(
            f"Missing EgoVideo backbone code at {BACKBONE_ROOT}. "
            "Please clone this repository with third_party/EgoVideo-main included."
        )

    if not CHECKPOINT_PATH.exists():
        download_checkpoint()

    summary = {
        "backbone_root": str(BACKBONE_ROOT),
        "checkpoint_path": str(CHECKPOINT_PATH),
        "checkpoint_exists": CHECKPOINT_PATH.exists(),
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
