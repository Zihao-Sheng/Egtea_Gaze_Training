#!/usr/bin/env python3
from __future__ import annotations

import json
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
from models.state_classifier import StateClassifier
from models.transition_aware_reranker import LearnedTransitionAwareReranker


DATA_ROOT = ROOT / "data" / "egtea_gaze_plus"
DEFAULT_CONFIG = ROOT / "configs" / "augmentation_ablation" / "rrc_flip.yaml"
NUM_CLASSES = 106

DEMO_ROOT = ROOT / "outputs" / "demo_ready" / "default_pipeline"
ACTION_RUN_DIR = DEMO_ROOT / "action_encoder"
CANDIDATE_DUMP_DIR = DEMO_ROOT / "candidate_dumps"
STATE_MODEL_DIR = DEMO_ROOT / "state_model" / "state_single_mlp_h3"
STATE_PRED_DIR = DEMO_ROOT / "state_predictions"
TRANSITION_PRIOR_DIR = DEMO_ROOT / "transition_priors"
TRANSITION_RUN_DIR = DEMO_ROOT / "transition_reranker" / "top10_h3_prev3"
STATE_ACTION_DIR = DEMO_ROOT / "state_action_reranker" / "soft_top10_h3_prev3"
BUNDLE_PATH = DEMO_ROOT / "bundle.json"


def load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def bundle_payload() -> dict:
    return {
        "name": "egtea_demo_ready_default_pipeline",
        "description": "RRC+Flip EgoVideo encoder + transition reranker + soft state prior",
        "action_config": str(DEFAULT_CONFIG.relative_to(ROOT)),
        "action_checkpoint": str((ACTION_RUN_DIR / "best.pt").relative_to(ROOT)),
        "state_mapping": str((ROOT / "outputs" / "state_model" / "action_to_state.json").relative_to(ROOT)),
        "state_checkpoint": str((STATE_MODEL_DIR / "best.pth").relative_to(ROOT)),
        "transition_prior": str((TRANSITION_PRIOR_DIR / "train_internal_transition_priors.pt").relative_to(ROOT)),
        "transition_checkpoint": str((TRANSITION_RUN_DIR / "best.pth").relative_to(ROOT)),
        "candidate_k": 10,
        "history_len": 3,
        "prev_mode": "prev3",
        "lambda_state": 0.5,
        "num_classes": NUM_CLASSES,
    }


def write_bundle() -> None:
    DEMO_ROOT.mkdir(parents=True, exist_ok=True)
    BUNDLE_PATH.write_text(json.dumps(bundle_payload(), indent=2), encoding="utf-8")


def load_bundle(bundle_path: Path | None = None) -> dict:
    path = bundle_path or BUNDLE_PATH
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_repo_path(path_like: str | Path) -> Path:
    path = Path(path_like)
    return path if path.is_absolute() else (ROOT / path)


def load_action_mapping(mapping_path: Path | None = None) -> tuple[list[str], list[str], torch.Tensor]:
    path = mapping_path or resolve_repo_path(load_bundle()["state_mapping"])
    payload = json.loads(path.read_text(encoding="utf-8"))
    action_to_state = payload["action_to_state"]
    action_names = [action_to_state[str(i)]["action_label"] for i in range(len(action_to_state))]
    state_names = list(payload["state_names"])
    action_state_ids = torch.tensor([int(action_to_state[str(i)]["state_id"]) for i in range(len(action_to_state))], dtype=torch.long)
    return action_names, state_names, action_state_ids


def resolve_clip_path(user_value: str | Path) -> Path:
    path = Path(user_value)
    if path.exists():
        return path.resolve()
    search_name = path.name if path.suffix else f"{path.name}.mp4"
    matches = list((DATA_ROOT / "cropped_clips").rglob(search_name))
    if not matches:
        raise FileNotFoundError(f"Could not find clip '{user_value}' under {DATA_ROOT / 'cropped_clips'}")
    if len(matches) > 1:
        raise RuntimeError(f"Found multiple clips named '{search_name}'. Please pass a fuller path.")
    return matches[0].resolve()


def resolve_session_dir(user_value: str | Path) -> Path:
    path = Path(user_value)
    if path.exists() and path.is_dir():
        return path.resolve()
    candidate = DATA_ROOT / "cropped_clips" / str(user_value)
    if candidate.exists() and candidate.is_dir():
        return candidate.resolve()
    raise FileNotFoundError(f"Could not find clip folder '{user_value}'.")


def build_inference_config(bundle: dict | None = None) -> dict:
    bundle = bundle or load_bundle()
    config = load_yaml(resolve_repo_path(bundle["action_config"]))
    config["device"] = "cuda" if torch.cuda.is_available() else "cpu"
    config["output_dir"] = str(ACTION_RUN_DIR)
    return config


def preprocess_clip(video_path: Path, config: dict) -> torch.Tensor:
    decoded_frames = decode_video_rgb_frames(video_path)
    num_frames = int(config["model"]["num_frames"])
    frame_indices = sample_uniform_indices(len(decoded_frames), num_frames)
    sampled_frames = [decoded_frames[i] for i in frame_indices]
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


def compute_transition_scores(
    prior_log_probs: torch.Tensor,
    prev_action_ids: torch.Tensor,
    prev_action_mask: torch.Tensor,
    candidate_ids: torch.Tensor,
    prev_mode: str,
) -> torch.Tensor:
    batch_size, candidate_k = candidate_ids.shape
    scores = torch.zeros((batch_size, candidate_k), device=candidate_ids.device, dtype=torch.float32)
    if prev_mode == "prev1":
        prev_ids = prev_action_ids[:, -1]
        valid = prev_action_mask[:, -1]
        if valid.any():
            scores[valid] = prior_log_probs[prev_ids[valid]].gather(1, candidate_ids[valid])
        return scores

    valid_counts = prev_action_mask.sum(dim=1)
    for batch_index in range(batch_size):
        count = int(valid_counts[batch_index].item())
        if count <= 0:
            continue
        use_count = min(3, count)
        ids = prev_action_ids[batch_index, -use_count:]
        gathered = prior_log_probs[ids].gather(1, candidate_ids[batch_index].unsqueeze(0).expand(use_count, -1))
        scores[batch_index] = gathered.mean(dim=0)
    return scores


def load_action_model(device: torch.device, bundle: dict | None = None):
    bundle = bundle or load_bundle()
    config = build_inference_config(bundle)
    model_bundle = build_model(config, num_classes=NUM_CLASSES)
    checkpoint = torch.load(resolve_repo_path(bundle["action_checkpoint"]), map_location="cpu", weights_only=False)
    model_bundle.model.load_state_dict(checkpoint["model_state_dict"])
    model_bundle.model.to(device)
    model_bundle.model.eval()
    return config, model_bundle


def load_state_model(device: torch.device, bundle: dict | None = None) -> StateClassifier:
    bundle = bundle or load_bundle()
    _, state_names, _ = load_action_mapping(resolve_repo_path(bundle["state_mapping"]))
    model = StateClassifier(input_dim=768, num_states=len(state_names))
    checkpoint = torch.load(resolve_repo_path(bundle["state_checkpoint"]), map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model


def load_transition_model(device: torch.device, bundle: dict | None = None) -> LearnedTransitionAwareReranker:
    bundle = bundle or load_bundle()
    model = LearnedTransitionAwareReranker(num_classes=NUM_CLASSES, embedding_dim=768)
    checkpoint = torch.load(resolve_repo_path(bundle["transition_checkpoint"]), map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model


def predict_action_logits(video_path: Path, device: torch.device, bundle: dict | None = None) -> dict:
    config, model_bundle = load_action_model(device=device, bundle=bundle)
    video = preprocess_clip(video_path, config).to(device)
    with torch.no_grad():
        logits, embeddings = model_bundle.model.forward_with_features(model_bundle.prepare_inputs(video, config))
        probs = logits.softmax(dim=1)
    return {
        "logits": logits.cpu(),
        "probs": probs.cpu(),
        "embedding": embeddings.cpu(),
    }


def rerank_sequence_predictions(
    clip_paths: list[Path],
    device: torch.device,
    bundle: dict | None = None,
) -> list[dict]:
    bundle = bundle or load_bundle()
    action_names, state_names, action_state_ids = load_action_mapping(resolve_repo_path(bundle["state_mapping"]))
    config, action_bundle = load_action_model(device=device, bundle=bundle)
    state_model = load_state_model(device=device, bundle=bundle)
    transition_model = load_transition_model(device=device, bundle=bundle)
    prior_payload = torch.load(resolve_repo_path(bundle["transition_prior"]), map_location="cpu", weights_only=False)
    prior_log_probs = prior_payload["transition_log_probs"].float().to(device)

    history_len = int(bundle["history_len"])
    candidate_k = int(bundle["candidate_k"])
    lambda_state = float(bundle["lambda_state"])
    prev_mode = str(bundle["prev_mode"])

    history_embeddings: list[torch.Tensor] = []
    prev_raw_action_ids: list[int] = []
    rows: list[dict] = []

    with torch.no_grad():
        for clip_index, clip_path in enumerate(clip_paths):
            video = preprocess_clip(clip_path, config).to(device)
            logits, embedding = action_bundle.model.forward_with_features(action_bundle.prepare_inputs(video, config))
            probs = logits.softmax(dim=1)
            topk_probs, topk_ids = torch.topk(probs, k=candidate_k, dim=1)
            topk_scores = logits.gather(1, topk_ids)

            state_logits = state_model(embedding)
            state_probs = state_logits.softmax(dim=1)
            pred_state_id = int(state_probs.argmax(dim=1).item())

            history_embeddings.append(embedding.squeeze(0).cpu())
            valid_history = history_embeddings[-history_len:]
            if len(valid_history) < history_len:
                pad = [valid_history[0].clone() for _ in range(history_len - len(valid_history))]
                valid_history = pad + valid_history
            history_tensor = torch.stack(valid_history, dim=0).unsqueeze(0).to(device)
            history_mask = torch.zeros((1, history_len), dtype=torch.bool, device=device)
            history_mask[0, -min(len(history_embeddings), history_len):] = True

            prev_action_tensor = torch.full((1, history_len), NUM_CLASSES, dtype=torch.long, device=device)
            prev_action_mask = torch.zeros((1, history_len), dtype=torch.bool, device=device)
            if prev_raw_action_ids:
                use_prev = prev_raw_action_ids[-history_len:]
                prev_action_tensor[0, -len(use_prev):] = torch.tensor(use_prev, dtype=torch.long, device=device)
                prev_action_mask[0, -len(use_prev):] = True
                prev_action_id = prev_action_tensor[0, -1].unsqueeze(0)
            else:
                prev_action_id = torch.tensor([NUM_CLASSES], dtype=torch.long, device=device)

            transition_scores = compute_transition_scores(
                prior_log_probs=prior_log_probs,
                prev_action_ids=prev_action_tensor,
                prev_action_mask=prev_action_mask,
                candidate_ids=topk_ids.to(device),
                prev_mode=prev_mode,
            )
            base_scores = transition_model(
                history_embeddings=history_tensor,
                history_mask=history_mask,
                current_embedding=embedding.to(device),
                candidate_ids=topk_ids.to(device),
                candidate_scores=topk_scores.to(device),
                candidate_probs=topk_probs.to(device),
                prev_action_id=prev_action_id,
                prev_action_ids=prev_action_tensor,
                prev_action_mask=prev_action_mask,
                prev_mode=prev_mode,
                transition_prior_scores=transition_scores,
            )

            candidate_state_ids = action_state_ids[topk_ids.squeeze(0).cpu()].to(device)
            candidate_state_probs = state_probs[0, candidate_state_ids]
            final_scores = base_scores[0] + lambda_state * torch.log(candidate_state_probs.clamp_min(1e-8))
            final_order = final_scores.argsort(descending=True)
            final_ids = topk_ids[0, final_order].cpu()
            final_probs = topk_probs[0, final_order].cpu()

            raw_top1_id = int(topk_ids[0, 0].item())
            prev_raw_action_ids.append(raw_top1_id)
            top5_pairs = [
                {
                    "rank": rank + 1,
                    "action_id": int(final_ids[rank].item()),
                    "action_label": action_names[int(final_ids[rank].item())],
                    "probability": float(final_probs[rank].item()),
                }
                for rank in range(min(5, candidate_k))
            ]
            rows.append(
                {
                    "clip_idx": clip_index,
                    "clip_name": clip_path.name,
                    "predicted_action_id": int(final_ids[0].item()),
                    "predicted_action": action_names[int(final_ids[0].item())],
                    "predicted_state_id": pred_state_id,
                    "predicted_state": state_names[pred_state_id],
                    "top5": top5_pairs,
                }
            )

    return rows
