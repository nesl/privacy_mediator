from __future__ import annotations

import torch
import torch.nn as nn


class PoseGRUFallClassifier(nn.Module):
    """Small temporal classifier for pose-keypoint fall detection."""

    def __init__(self, input_dim: int = 17 * 3, hidden_dim: int = 128, num_layers: int = 1, num_classes: int = 2, dropout: float = 0.2):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=True,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim * 2),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B,T,D]
        out, _ = self.gru(x)
        pooled = out.mean(dim=1)
        return self.head(pooled)
