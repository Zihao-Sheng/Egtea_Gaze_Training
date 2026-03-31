from __future__ import annotations

import json
from pathlib import Path

import torch
from torch.utils.data import Dataset


class EgteaStateDataset(Dataset):
    """State dataset backed by the existing EgoVideo candidate dump files."""

    def __init__(
        self,
        dump_path: Path,
        mapping_path: Path,
        history_len: int = 1,
        task_mode: str = "single",
        include_logits: bool = False,
        taxonomy_version: str | None = None,
    ) -> None:
        if task_mode not in {"single", "causal"}:
            raise ValueError(f"Unsupported task_mode: {task_mode}")
        payload = torch.load(dump_path, map_location="cpu", weights_only=False)
        mapping_payload = json.loads(mapping_path.read_text(encoding="utf-8"))

        self.split = str(payload["split"])
        self.history_len = int(history_len)
        self.task_mode = task_mode
        self.include_logits = bool(include_logits)
        self.embeddings = payload["embeddings"].float()
        self.logits = payload["logits"].float()
        self.action_labels = payload["labels"].long()
        self.session_ids = list(payload["session_ids"])
        self.clip_stems = list(payload["clip_stems"])
        self.clip_indices = payload["clip_indices"].long()
        self.embedding_dim = int(self.embeddings.shape[1])
        self.logit_dim = int(self.logits.shape[1])
        self.feature_dim = self.embedding_dim + (self.logit_dim if self.include_logits else 0)
        self.state_names = list(mapping_payload["state_names"])
        self.num_states = int(mapping_payload["num_states"])
        self.taxonomy_version = str(mapping_payload.get("taxonomy_version", "v1"))
        if taxonomy_version is not None and str(taxonomy_version) != self.taxonomy_version:
            raise ValueError(
                f"Requested taxonomy_version={taxonomy_version}, but mapping payload reports {self.taxonomy_version}"
            )

        action_to_state = mapping_payload["action_to_state"]
        self.state_labels = torch.tensor(
            [int(action_to_state[str(int(action_id))]["state_id"]) for action_id in self.action_labels.tolist()],
            dtype=torch.long,
        )

        self.session_to_positions: dict[str, list[int]] = {}
        for position, session_id in enumerate(self.session_ids):
            self.session_to_positions.setdefault(session_id, []).append(position)
        for session_id, positions in self.session_to_positions.items():
            self.session_to_positions[session_id] = sorted(positions, key=lambda idx: int(self.clip_indices[idx]))

        self.samples: list[tuple[str, int]] = []
        for session_id, positions in self.session_to_positions.items():
            for session_offset in range(len(positions)):
                self.samples.append((session_id, session_offset))

    def __len__(self) -> int:
        return len(self.samples)

    def _compose_features(self, indices: list[int]) -> torch.Tensor:
        features = self.embeddings[indices]
        if self.include_logits:
            features = torch.cat([features, self.logits[indices]], dim=1)
        return features

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | int | str]:
        session_id, session_offset = self.samples[index]
        positions = self.session_to_positions[session_id]
        current_position = positions[session_offset]

        if self.task_mode == "single":
            feature = self._compose_features([current_position])[0]
            return {
                "features": feature,
                "state_label": int(self.state_labels[current_position]),
                "action_label": int(self.action_labels[current_position]),
                "clip_stem": self.clip_stems[current_position],
                "session_id": session_id,
                "clip_idx": int(self.clip_indices[current_position]),
            }

        start_offset = max(0, session_offset - self.history_len + 1)
        history_positions = positions[start_offset : session_offset + 1]
        sequence = self._compose_features(history_positions)
        valid_length = int(sequence.shape[0])
        if valid_length < self.history_len:
            pad = sequence[:1].repeat(self.history_len - valid_length, 1)
            sequence = torch.cat([pad, sequence], dim=0)
        mask = torch.zeros(self.history_len, dtype=torch.bool)
        mask[-valid_length:] = True
        return {
            "features": sequence,
            "mask": mask,
            "state_label": int(self.state_labels[current_position]),
            "action_label": int(self.action_labels[current_position]),
            "clip_stem": self.clip_stems[current_position],
            "session_id": session_id,
            "clip_idx": int(self.clip_indices[current_position]),
        }


def predict_state_for_clips(state_predictions_path: Path) -> dict[tuple[str, int], int]:
    payload = json.loads(state_predictions_path.read_text(encoding="utf-8"))
    return {(str(row["session_id"]), int(row["clip_idx"])): int(row["pred_state"]) for row in payload["predictions"]}


def load_state_predictions(state_predictions_path: Path) -> dict[tuple[str, int], int]:
    return predict_state_for_clips(state_predictions_path)
