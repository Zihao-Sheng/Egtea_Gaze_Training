from __future__ import annotations

import torch
import torch.nn as nn


class CausalContextEncoder(nn.Module):
    """A lightweight causal GRU encoder over past-and-current clip embeddings."""

    def __init__(self, input_size: int, hidden_size: int = 256, num_layers: int = 2, dropout: float = 0.3) -> None:
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

    def forward(self, embeddings: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        outputs, hidden = self.gru(embeddings)
        if mask is None:
            return hidden[-1]

        lengths = mask.long().sum(dim=1).clamp(min=1)
        batch_indices = torch.arange(outputs.size(0), device=outputs.device)
        last_indices = lengths - 1
        return outputs[batch_indices, last_indices]
