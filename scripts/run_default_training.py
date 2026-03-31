#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import csv
import subprocess
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

STATE_MAPPING_PATH = ROOT / "outputs" / "state_model" / "action_to_state.json"
DEFAULT_CONFIG = ROOT / "configs" / "augmentation_ablation" / "rrc_flip.yaml"
DEMO_ROOT = ROOT / "outputs" / "demo_ready" / "default_pipeline"
BUNDLE_PATH = DEMO_ROOT / "bundle.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the default demo-ready EGTEA training pipeline.")
    parser.add_argument("--device", type=str, default=None, help="Override device, e.g. cuda or cpu.")
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run a minimal end-to-end sanity check with tiny training splits and 1 epoch.",
    )
    parser.add_argument(
        "--force-rebuild",
        action="store_true",
        help="Ignore existing outputs and rerun all steps that this script launches.",
    )
    parser.add_argument(
        "--quick-train",
        action="store_true",
        help="Run the full pipeline but train the single-clip encoder for only 1 epoch.",
    )
    parser.add_argument(
        "--train-epochs",
        type=int,
        default=None,
        help="Override single-clip training epochs for the default pipeline.",
    )
    return parser.parse_args()


def ensure_state_mapping() -> None:
    if STATE_MAPPING_PATH.exists():
        return
    annotations_path = ROOT / "data" / "egtea_gaze_plus" / "raw_annotations" / "cls_label_index.csv"
    state_names = [
        "access_open",
        "acquire_take",
        "manipulate_process",
        "transfer_place_move",
        "close_finish",
        "other_misc",
    ]
    action_to_state: dict[str, dict] = {}
    with annotations_path.open("r", encoding="utf-8") as handle:
        reader = csv.reader(handle, delimiter=";")
        for row in reader:
            if not row:
                continue
            first = row[0].strip()
            if not first or first.startswith("#"):
                continue
            action_id = int(first)
            action_label = row[1].strip()
            verb = row[2].strip().lower()
            noun = row[3].strip().lower()
            if verb in {"open", "turn on"}:
                state_name = "access_open"
            elif verb == "take":
                state_name = "acquire_take"
            elif verb in {"close", "turn off"}:
                state_name = "close_finish"
            elif verb in {"put", "move around"}:
                state_name = "transfer_place_move"
            elif verb in {"cut", "mix", "divide/pull apart", "crack", "wash", "spread", "pour", "squeeze", "compress", "clean/wipe", "operate"}:
                state_name = "manipulate_process"
            else:
                state_name = "other_misc"
            action_to_state[str(action_id)] = {
                "action_id": action_id,
                "action_label": action_label,
                "verb_label": verb,
                "noun_label": noun,
                "state_id": state_names.index(state_name),
                "state_name": state_name,
            }
    STATE_MAPPING_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATE_MAPPING_PATH.write_text(
        json.dumps(
            {
                "taxonomy_version": "v1",
                "state_names": state_names,
                "num_states": len(state_names),
                "action_to_state": action_to_state,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def run_step(command: list[str], done_path: Path | None = None, force: bool = False) -> None:
    if (not force) and done_path is not None and done_path.exists():
        print(f"[skip] {done_path}")
        return
    print("[run]", " ".join(command))
    subprocess.run(command, check=True, cwd=ROOT)


def main() -> int:
    args = parse_args()
    DEMO_ROOT.mkdir(parents=True, exist_ok=True)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    singleclip_epochs = args.train_epochs
    if singleclip_epochs is None:
        if args.smoke_test:
            singleclip_epochs = 1
        elif args.quick_train:
            singleclip_epochs = 1
        else:
            singleclip_epochs = 5

    if args.smoke_test:
        variant_name = "smoke"
    elif args.quick_train:
        variant_name = f"quick_e{singleclip_epochs}"
    else:
        variant_name = "default"

    variant_root = DEMO_ROOT if variant_name == "default" else (ROOT / "outputs" / "demo_ready" / f"default_pipeline_{variant_name}")
    bundle_path = variant_root / "bundle.json"

    train_output_dir = variant_root / ("action_encoder_smoke" if args.smoke_test else "action_encoder")
    candidate_dump_dir = variant_root / ("candidate_dumps_smoke" if args.smoke_test else "candidate_dumps")
    state_model_dir = variant_root / "state_model" / ("state_single_mlp_h3_smoke" if args.smoke_test else "state_single_mlp_h3")
    state_pred_dir = variant_root / ("state_predictions_smoke" if args.smoke_test else "state_predictions")
    transition_prior_dir = variant_root / ("transition_priors_smoke" if args.smoke_test else "transition_priors")
    transition_run_dir = variant_root / "transition_reranker" / ("top10_h3_prev3_smoke" if args.smoke_test else "top10_h3_prev3")

    run_step([sys.executable, str(ROOT / "scripts" / "setup_dataset.py")], force=args.force_rebuild)
    run_step([sys.executable, str(ROOT / "scripts" / "setup_egovideo.py")], force=args.force_rebuild)
    ensure_state_mapping()

    train_command = [
        sys.executable,
        str(ROOT / "scripts" / "train_with_augmentation.py"),
        "--config",
        str(DEFAULT_CONFIG),
        "--output-dir",
        str(train_output_dir),
        "--device",
        device,
    ]
    if args.smoke_test:
        train_command.extend(["--smoke-test"])
    else:
        train_command.extend(["--epochs", str(singleclip_epochs)])
    run_step(
        train_command,
        done_path=train_output_dir / "best.pt",
        force=args.force_rebuild,
    )

    run_step(
        [
            sys.executable,
            str(ROOT / "scripts" / "dump_topk_candidates.py"),
            "--checkpoint",
            str(train_output_dir / "best.pt"),
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
            str(candidate_dump_dir),
        ],
        done_path=candidate_dump_dir / "test.pt",
        force=args.force_rebuild,
    )

    run_step(
        [
            sys.executable,
            str(ROOT / "scripts" / "train_state_model.py"),
            "--dump-dir",
            str(candidate_dump_dir),
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
            str(state_model_dir),
            "--device",
            device,
            "--batch-size",
            "128",
            "--num-workers",
            "0",
        ],
        done_path=state_model_dir / "best.pth",
        force=args.force_rebuild,
    )

    run_step(
        [
            sys.executable,
            str(ROOT / "scripts" / "dump_state_predictions.py"),
            "--dump-dir",
            str(candidate_dump_dir),
            "--taxonomy-version",
            "v1",
            "--checkpoint",
            str(state_model_dir / "best.pth"),
            "--output-dir",
            str(state_pred_dir),
            "--device",
            device,
        ],
        done_path=state_pred_dir / "test.pt",
        force=args.force_rebuild,
    )

    run_step(
        [
            sys.executable,
            str(ROOT / "scripts" / "build_transition_priors.py"),
            "--dump-dir",
            str(candidate_dump_dir),
            "--split",
            "train_internal",
            "--output-dir",
            str(transition_prior_dir),
            "--smoothing",
            "1.0",
        ],
        done_path=transition_prior_dir / "train_internal_transition_priors.pt",
        force=args.force_rebuild,
    )

    run_step(
        [
            sys.executable,
            str(ROOT / "scripts" / "train_transition_reranker.py"),
            "--dump-dir",
            str(candidate_dump_dir),
            "--prior-path",
            str(transition_prior_dir / "train_internal_transition_priors.pt"),
            "--output-dir",
            str(transition_run_dir),
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
        done_path=transition_run_dir / "best.pth",
        force=args.force_rebuild,
    )

    bundle_payload = {
        "name": f"egtea_demo_ready_{variant_name}",
        "description": "Smoke-check bundle" if args.smoke_test else f"RRC+Flip EgoVideo encoder + transition reranker + soft state prior ({singleclip_epochs} epoch single-clip)",
        "action_config": str(DEFAULT_CONFIG.relative_to(ROOT)),
        "action_checkpoint": str((train_output_dir / "best.pt").relative_to(ROOT)),
        "state_mapping": str((ROOT / "outputs" / "state_model" / "action_to_state.json").relative_to(ROOT)),
        "state_checkpoint": str((state_model_dir / "best.pth").relative_to(ROOT)),
        "transition_prior": str((transition_prior_dir / "train_internal_transition_priors.pt").relative_to(ROOT)),
        "transition_checkpoint": str((transition_run_dir / "best.pth").relative_to(ROOT)),
        "candidate_k": 10,
        "history_len": 3,
        "prev_mode": "prev3",
        "lambda_state": 0.5,
        "num_classes": 106,
        "smoke_test": bool(args.smoke_test),
        "quick_train": bool(args.quick_train),
        "singleclip_epochs": int(singleclip_epochs),
    }
    variant_root.mkdir(parents=True, exist_ok=True)
    bundle_path.write_text(json.dumps(bundle_payload, indent=2), encoding="utf-8")
    summary = {
        "bundle": str(bundle_path.relative_to(ROOT)),
        "smoke_test": bool(args.smoke_test),
        "quick_train": bool(args.quick_train),
        "singleclip_epochs": int(singleclip_epochs),
        "action_checkpoint": str((train_output_dir / "best.pt").relative_to(ROOT)),
        "state_checkpoint": str((state_model_dir / "best.pth").relative_to(ROOT)),
        "transition_checkpoint": str((transition_run_dir / "best.pth").relative_to(ROOT)),
    }
    (variant_root / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
