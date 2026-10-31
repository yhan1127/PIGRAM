# src/pigram/models/heads.py
# -*- coding: utf-8 -*-

from __future__ import annotations

import torch
import torch.nn as nn


class PatchImportanceHead(nn.Module):
    """
    Patch-level importance prediction head.

    Input:
        patch_emb: [N_patches, hidden_dim]

    Output:
        patch_importance_logits: [N_patches]
    """

    def __init__(
        self,
        hidden_dim: int = 128,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, patch_emb: torch.Tensor) -> torch.Tensor:
        return self.mlp(patch_emb).squeeze(-1)