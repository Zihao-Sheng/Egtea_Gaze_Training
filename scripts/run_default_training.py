#!/usr/bin/env python3
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts._demo_ready_common import (
    ACTION_RUN_DIR,
    BUNDLE_PATH,
    CANDIDATE_DUMP_DIR,
    DEFAULT_CONFIG,
    DEMO_ROOT,
    STATE_ACTION_DIR,
    STATE_MODEL_DIR,
    STATE_PRED_DIR,
    TRANSITION_PRIOR_DIR,
    TRANSITION_RUN_DIR,
    write_bundle,
)


def run_step(command: list[str], done_path: Path | None = None) -> None:
    if done_path is not None and done_path.exists():
        print(f"[skip] {done_path}")
        return
    print("[run]", " ".join(command))
    subprocess.run(command, check=True, cwd=ROOT)


def main() -> int:
    DEMO_ROOT.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    run_step([sys.executable, str(ROOT / "scripts" / "setup_dataset.py")])
    run_step([sys.executable, str(ROOT / "scripts" / "setup_egovideo.py")])
    run_step(
        [sys.executable, str(ROOT / "scripts" / "build_state_mapping.py"), "--taxonomy-version", "v1"],
        done_path=ROOT / "outputs" / "state_model" / "action_to_state.json",
    )

    run_step(
        [
            sys.executable,
            str(ROOT / "scripts" / "train_with_augmentation.py"),
            "--config",
            str(DEFAULT_CONFIG),
            "--output-dir",
            str(ACTION_RUN_DIR),
            "--epochs",
            "5",
            "--device",
            device,
        ],
        done_path=ACTION_RUN_DIR / "best.pt",
    )

    run_step(
        [
            sys.executable,
            str(ROOT / "scripts" / "dump_topk_candidates.py"),
            "--checkpoint",
            str(ACTION_RUN_DIR / "best.pt"),
            "--device",
            device,
            "--batch-size",
            "2",
            "--num-workers",
            "0",
            "--split-id",
            "1",
            "--splits",
            "train_internal",
            "val_internal",
            "test",
            "--output-dir",
            str(CANDIDATE_DUMP_DIR),
        ],
        done_path=CANDIDATE_DUMP_DIR / "test.pt",
    )

    run_step(
        [
            sys.executable,
            str(ROOT / "scripts" / "train_state_model.py"),
            "--dump-dir",
            str(CANDIDATE_DUMP_DIR),
            "--taxonomy-version",
            "v1",
            "--train-split",
            "train_internal",
            "--val-split",
            "val_internal",
            "--task-mode",
            "single",
            "--model-type",
            "mlp",
            "--history-len",
            "3",
            "--output-dir",
            str(STATE_MODEL_DIR),
            "--device",
            device,
            "--batch-size",
            "128",
            "--num-workers",
            "0",
        ],
        done_path=STATE_MODEL_DIR / "best.pth",
    )

    run_step(
        [
            sys.executable,
            str(ROOT / "scripts" / "dump_state_predictions.py"),
            "--dump-dir",
            str(CANDIDATE_DUMP_DIR),
            "--taxonomy-version",
            "v1",
            "--checkpoint",
            str(STATE_MODEL_DIR / "best.pth"),
            "--output-dir",
            str(STATE_PRED_DIR),
            "--device",
            device,
        ],
        done_path=STATE_PRED_DIR / "test.pt",
    )

    run_step(
        [
            sys.executable,
            str(ROOT / "scripts" / "build_transition_priors.py"),
            "--dump-dir",
            str(CANDIDATE_DUMP_DIR),
            "--split",
            "train_internal",
            "--output-dir",
            str(TRANSITION_PRIOR_DIR),
            "--smoothing",
            "1.0",
        ],
        done_path=TRANSITION_PRIOR_DIR / "train_internal_transition_priors.pt",
    )

    run_step(
        [
            sys.executable,
            str(ROOT / "scripts" / "train_transition_reranker.py"),
            "--dump-dir",
            str(CANDIDATE_DUMP_DIR),
            "--prior-path",
            str(TRANSITION_PRIOR_DIR / "train_internal_transition_priors.pt"),
            "--output-dir",
            str(TRANSITION_RUN_DIR),
            "--train-split",
            "train_internal",
            "--val-split",
            "val_internal",
            "--candidate-k",
            "10",
            "--history-len",
            "3",
            "--prev-mode",
            "prev3",
            "--epochs",
            "12",
            "--batch-size",
            "128",
            "--num-workers",
            "0",
            "--learning-rate",
            "0.001",
            "--weight-decay",
            "0.0001",
            "--patience",
            "3",
            "--device",
            device,
        ],
        done_path=TRANSITION_RUN_DIR / "best.pth",
    )

    run_step(
        [
            sys.executable,
            str(ROOT / "scripts" / "eval_state_constrained_reranker.py"),
            "--dump-dir",
            str(CANDIDATE_DUMP_DIR),
            "--state-pred-dir",
            str(STATE_PRED_DIR),
            "--mapping-path",
            str(ROOT / "outputs" / "state_model" / "action_to_state.json"),
            "--prior-path",
            str(TRANSITION_PRIOR_DIR / "train_internal_transition_priors.pt"),
            "--split",
            "val_internal",
            "--candidate-k",
            "10",
            "--history-len",
            "3",
            "--prev-mode",
            "prev3",
            "--state-mode",
            "soft",
            "--lambda-state",
            "0.5",
            "--transition-checkpoint",
            str(TRANSITION_RUN_DIR / "best.pth"),
            "--batch-size",
            "128",
            "--num-workers",
            "0",
            "--device",
            device,
            "--output-json",
            str(STATE_ACTION_DIR / "metrics_val.json"),
        ],
        done_path=STATE_ACTION_DIR / "metrics_val.json",
    )

    run_step(
        [
            sys.executable,
            str(ROOT / "scripts" / "eval_state_constrained_reranker.py"),
            "--dump-dir",
            str(CANDIDATE_DUMP_DIR),
            "--state-pred-dir",
            str(STATE_PRED_DIR),
            "--mapping-path",
            str(ROOT / "outputs" / "state_model" / "action_to_state.json"),
            "--prior-path",
            str(TRANSITION_PRIOR_DIR / "train_internal_transition_priors.pt"),
            "--split",
            "test",
            "--candidate-k",
            "10",
            "--history-len",
            "3",
            "--prev-mode",
            "prev3",
            "--state-mode",
            "soft",
            "--lambda-state",
            "0.5",
            "--transition-checkpoint",
            str(TRANSITION_RUN_DIR / "best.pth"),
            "--batch-size",
            "128",
            "--num-workers",
            "0",
            "--device",
            device,
            "--output-json",
            str(STATE_ACTION_DIR / "metrics_test.json"),
        ],
        done_path=STATE_ACTION_DIR / "metrics_test.json",
    )

    write_bundle()
    summary = {
        "bundle": str(BUNDLE_PATH.relative_to(ROOT)),
        "action_checkpoint": str((ACTION_RUN_DIR / "best.pt").relative_to(ROOT)),
        "state_checkpoint": str((STATE_MODEL_DIR / "best.pth").relative_to(ROOT)),
        "transition_checkpoint": str((TRANSITION_RUN_DIR / "best.pth").relative_to(ROOT)),
        "final_test_metrics": str((STATE_ACTION_DIR / "metrics_test.json").relative_to(ROOT)),
    }
    (DEMO_ROOT / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
