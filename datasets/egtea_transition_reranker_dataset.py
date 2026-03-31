from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import Dataset


class EgteaTransitionRerankerDataset(Dataset):
    """Top-k reranker dataset with explicit causal previous-action signals."""

    def __init__(
        self,
        dump_path: Path,
        history_len: int = 5,
        candidate_k: int = 5,
        hit_only: bool = True,
    ) -> None:
        payload = torch.load(dump_path, map_location="cpu", weights_only=False)
        self.split = str(payload["split"])
        self.history_len = int(history_len)
        self.candidate_k = int(candidate_k)
        self.hit_only = bool(hit_only)

        self.embeddings = payload["embeddings"].float()
        self.labels = payload["labels"].long()
        self.session_ids = list(payload["session_ids"])
        self.clip_stems = list(payload["clip_stems"])
        self.clip_indices = payload["clip_indices"].long()
        self.topk_ids = payload[f"top{candidate_k}_ids"].long()
        self.topk_scores = payload[f"top{candidate_k}_scores"].float()
        self.topk_probs = payload[f"top{candidate_k}_probs"].float()
        self.raw_top1_ids = payload["top1_ids"].long()

        self.num_classes = int(payload["num_classes"])
        self.embedding_dim = int(self.embeddings.shape[1])
        self.padding_action_id = self.num_classes

        self.samples: list[int] = []
        self.target_positions: dict[int, int] = {}
        self.session_to_positions: dict[str, list[int]] = {}

        for position, session_id in enumerate(self.session_ids):
            self.session_to_positions.setdefault(session_id, []).append(position)
        for session_id, positions in self.session_to_positions.items():
            self.session_to_positions[session_id] = sorted(positions, key=lambda idx: int(self.clip_indices[idx]))

        hit_count = 0
        for position in range(len(self.labels)):
            label = int(self.labels[position])
            candidate_ids = self.topk_ids[position].tolist()
            try:
                target_pos = candidate_ids.index(label)
                hit_count += 1
            except ValueError:
                target_pos = -1
            if target_pos >= 0:
                self.target_positions[position] = target_pos
            if not self.hit_only or target_pos >= 0:
                self.samples.append(position)
        self.coverage = hit_count / max(len(self.labels), 1)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | int | str | bool]:
        position = self.samples[index]
        session_id = self.session_ids[position]
        session_positions = self.session_to_positions[session_id]
        offset = session_positions.index(position)

        history_start = max(0, offset - self.history_len + 1)
        history_positions = session_positions[history_start : offset + 1]
        history_embeddings = self.embeddings[history_positions]
        valid_history_length = history_embeddings.shape[0]
        if valid_history_length < self.history_len:
            pad = history_embeddings[:1].repeat(self.history_len - valid_history_length, 1)
            history_embeddings = torch.cat([pad, history_embeddings], dim=0)
        history_mask = torch.zeros(self.history_len, dtype=torch.bool)
        history_mask[-valid_history_length:] = True

        prev_positions = session_positions[max(0, offset - self.history_len) : offset]
        prev_action_ids = self.raw_top1_ids[prev_positions]
        prev_valid_length = prev_action_ids.shape[0]
        if prev_valid_length == 0:
            prev_action_ids = torch.full((self.history_len,), self.padding_action_id, dtype=torch.long)
            prev_action_mask = torch.zeros(self.history_len, dtype=torch.bool)
            prev_action_id = int(self.padding_action_id)
        else:
            if prev_valid_length < self.history_len:
                pad = torch.full((self.history_len - prev_valid_length,), self.padding_action_id, dtype=torch.long)
                prev_action_ids = torch.cat([pad, prev_action_ids], dim=0)
            prev_action_mask = torch.zeros(self.history_len, dtype=torch.bool)
            prev_action_mask[-prev_valid_length:] = True
            prev_action_id = int(prev_action_ids[-1].item())

        target_pos = self.target_positions.get(position, -1)
        return {
            "history_embeddings": history_embeddings,
            "history_mask": history_mask,
            "current_embedding": self.embeddings[position],
            "candidate_ids": self.topk_ids[position],
            "candidate_scores": self.topk_scores[position],
            "candidate_probs": self.topk_probs[position],
            "label": int(self.labels[position]),
            "target_pos": int(target_pos),
            "is_hit": bool(target_pos >= 0),
            "raw_top1_id": int(self.raw_top1_ids[position]),
            "prev_action_id": int(prev_action_id),
            "prev_action_ids": prev_action_ids,
            "prev_action_mask": prev_action_mask,
            "clip_stem": self.clip_stems[position],
            "session_id": session_id,
            "clip_idx": int(self.clip_indices[position]),
        }
