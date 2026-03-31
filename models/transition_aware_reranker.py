from __future__ import annotations

import torch
import torch.nn as nn

from models.causal_context_encoder import CausalContextEncoder


class LearnedTransitionAwareReranker(nn.Module):
    """Causal Top-k reranker with explicit transition-prior features."""

    def __init__(
        self,
        num_classes: int,
        embedding_dim: int,
        context_hidden_size: int = 256,
        candidate_id_dim: int = 128,
        prev_action_dim: int = 64,
        hidden_dim: int = 256,
    ) -> None:
        super().__init__()
        self.num_classes = int(num_classes)
        self.padding_action_id = self.num_classes
        self.context_encoder = CausalContextEncoder(
            input_size=embedding_dim,
            hidden_size=context_hidden_size,
            num_layers=2,
            dropout=0.3,
        )
        self.class_embedding = nn.Embedding(self.num_classes, candidate_id_dim)
        self.prev_action_embedding = nn.Embedding(self.num_classes + 1, prev_action_dim, padding_idx=self.padding_action_id)
        self.rank_embedding = nn.Embedding(32, 16)
        self.current_projection = nn.Linear(embedding_dim, 128)

        input_dim = context_hidden_size + candidate_id_dim + prev_action_dim + 16 + 128 + 5
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(
        self,
        history_embeddings: torch.Tensor,
        history_mask: torch.Tensor,
        current_embedding: torch.Tensor,
        candidate_ids: torch.Tensor,
        candidate_scores: torch.Tensor,
        candidate_probs: torch.Tensor,
        prev_action_id: torch.Tensor,
        transition_prior_scores: torch.Tensor,
        prev_action_ids: torch.Tensor | None = None,
        prev_action_mask: torch.Tensor | None = None,
        prev_mode: str = "prev1",
    ) -> torch.Tensor:
        batch_size, candidate_k = candidate_ids.shape
        context_vector = self.context_encoder(history_embeddings, history_mask)
        context_features = context_vector.unsqueeze(1).expand(-1, candidate_k, -1)
        candidate_embed = self.class_embedding(candidate_ids)
        prev_embed = self._build_prev_summary(
            prev_action_id=prev_action_id,
            prev_action_ids=prev_action_ids,
            prev_action_mask=prev_action_mask,
            prev_mode=prev_mode,
        ).unsqueeze(1).expand(-1, candidate_k, -1)
        rank_positions = torch.arange(candidate_k, device=candidate_ids.device).unsqueeze(0).expand(batch_size, -1)
        rank_embed = self.rank_embedding(rank_positions)
        current_features = self.current_projection(current_embedding).unsqueeze(1).expand(-1, candidate_k, -1)
        score_features = torch.stack(
            [
                candidate_scores,
                candidate_probs,
                candidate_scores - candidate_scores[:, :1],
                transition_prior_scores,
                candidate_scores + transition_prior_scores,
            ],
            dim=-1,
        )
        features = torch.cat(
            [context_features, candidate_embed, prev_embed, rank_embed, current_features, score_features],
            dim=-1,
        )
        return self.mlp(features).squeeze(-1)

    def _build_prev_summary(
        self,
        prev_action_id: torch.Tensor,
        prev_action_ids: torch.Tensor | None,
        prev_action_mask: torch.Tensor | None,
        prev_mode: str,
    ) -> torch.Tensor:
        if prev_mode == "prev1" or prev_action_ids is None or prev_action_mask is None:
            return self.prev_action_embedding(prev_action_id)

        prev_embeds = self.prev_action_embedding(prev_action_ids)
        summary = torch.zeros(
            (prev_embeds.size(0), prev_embeds.size(-1)),
            device=prev_embeds.device,
            dtype=prev_embeds.dtype,
        )
        valid_counts = prev_action_mask.sum(dim=1)
        for batch_index in range(prev_embeds.size(0)):
            count = int(valid_counts[batch_index].item())
            if count <= 0:
                continue
            use_count = min(3, count)
            valid_embeds = prev_embeds[batch_index, -use_count:]
            summary[batch_index] = valid_embeds.mean(dim=0)
        return summary
