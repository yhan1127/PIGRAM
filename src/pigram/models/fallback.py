# src/pigram/models/fallback.py
# -*- coding: utf-8 -*-

from __future__ import annotations

import torch
import torch.nn as nn


class CrossAttentionBlock(nn.Module):
    """
    Bidirectional patch--motif cross-attention block.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        dropout: float,
    ) -> None:
        super().__init__()

        self.patch_to_motif = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.motif_to_patch = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.patch_ln1 = nn.LayerNorm(hidden_dim)
        self.patch_ln2 = nn.LayerNorm(hidden_dim)
        self.motif_ln1 = nn.LayerNorm(hidden_dim)
        self.motif_ln2 = nn.LayerNorm(hidden_dim)

        self.patch_ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )
        self.motif_ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        patch_x: torch.Tensor,
        motif_x: torch.Tensor,
    ):
        patch_attn, _ = self.patch_to_motif(
            query=patch_x,
            key=motif_x,
            value=motif_x,
        )
        patch_x = self.patch_ln1(patch_x + self.dropout(patch_attn))
        patch_x = self.patch_ln2(patch_x + self.dropout(self.patch_ffn(patch_x)))

        motif_attn, _ = self.motif_to_patch(
            query=motif_x,
            key=patch_x,
            value=patch_x,
        )
        motif_x = self.motif_ln1(motif_x + self.dropout(motif_attn))
        motif_x = self.motif_ln2(motif_x + self.dropout(self.motif_ffn(motif_x)))

        return patch_x, motif_x


class TransformerFallbackHead(nn.Module):
    """
    Transformer fallback head for patch--motif logits.

    Input:
        patch_latent: [N_patch, hidden_dim]
        motif_latent: [N_motif, hidden_dim]

    Output:
        logits: [N_patch, N_motif]
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        num_layers: int,
        dropout: float,
    ) -> None:
        super().__init__()

        self.blocks = nn.ModuleList(
            [
                CrossAttentionBlock(
                    hidden_dim=hidden_dim,
                    num_heads=num_heads,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )

        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    @staticmethod
    def compute_pair_feat(
        patch_x: torch.Tensor,
        motif_x: torch.Tensor,
    ) -> torch.Tensor:
        n_patches, hidden_dim = patch_x.shape
        n_motifs, _ = motif_x.shape

        patch_expand = patch_x.unsqueeze(1).expand(
            n_patches,
            n_motifs,
            hidden_dim,
        )
        motif_expand = motif_x.unsqueeze(0).expand(
            n_patches,
            n_motifs,
            hidden_dim,
        )

        return torch.cat(
            [
                patch_expand,
                motif_expand,
                patch_expand * motif_expand,
                torch.abs(patch_expand - motif_expand),
            ],
            dim=-1,
        )

    def forward(
        self,
        patch_latent: torch.Tensor,
        motif_latent: torch.Tensor,
    ) -> torch.Tensor:
        patch_x = patch_latent.unsqueeze(0)
        motif_x = motif_latent.unsqueeze(0)

        for block in self.blocks:
            patch_x, motif_x = block(patch_x, motif_x)

        patch_x = patch_x.squeeze(0)
        motif_x = motif_x.squeeze(0)

        pair_feat = self.compute_pair_feat(patch_x, motif_x)
        logits = self.mlp(pair_feat).squeeze(-1)

        return logits