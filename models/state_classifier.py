from __future__ import annotations

import torch
import torch.nn as nn


class StateClassifier(nn.Module):
    """Single-clip state classifier over EgoVideo embeddings."""

    def __init__(self, input_dim: int, num_states: int, hidden_dim: int = 256, dropout: float = 0.3) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_states),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.mlp(features)
