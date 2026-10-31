# src/pigram/models/baseline.py
# -*- coding: utf-8 -*-

from __future__ import annotations

from typing import Dict, Type

import torch
import torch.nn as nn


def ensure_2d_latent(name: str, x: torch.Tensor) -> torch.Tensor:
    if x.dim() == 3 and x.shape[0] == 1:
        x = x.squeeze(0)

    if x.dim() == 3 and x.shape[1] == 1:
        x = x.squeeze(1)

    if x.dim() != 2:
        raise RuntimeError(
            f"{name} must be 2D after squeeze, got shape={tuple(x.shape)}"
        )

    return x


class CrossAttentionBlock(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        dropout: float,
    ) -> None:
        super().__init__()

        self.patch_to_motif = nn.MultiheadAttention(
            hidden_dim,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.motif_to_patch = nn.MultiheadAttention(
            hidden_dim,
            num_heads,
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


class SharedEncoderBackbone(nn.Module):
    def __init__(
        self,
        protein_encoder_cls: Type[nn.Module],
        motif_encoder_cls: Type[nn.Module],
        patch_importance_head_cls: Type[nn.Module],
        hidden_dim: int,
        dropout: float,
        protein_vocab_size: int = 32,
        rna_vocab_size: int = 16,
        motif_type_vocab_size: int = 8,
    ) -> None:
        super().__init__()

        self.hidden_dim = int(hidden_dim)

        self.protein_encoder = protein_encoder_cls(
            protein_vocab_size=protein_vocab_size,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )

        self.motif_encoder = motif_encoder_cls(
            rna_vocab_size=rna_vocab_size,
            motif_type_vocab_size=motif_type_vocab_size,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )

        self.patch_importance_head = patch_importance_head_cls(
            hidden_dim=hidden_dim,
            dropout=dropout,
        )

    def encode_all(self, batch: Dict[str, torch.Tensor]):
        patch_out = self.protein_encoder(
            protein_token_ids=batch["protein_token_ids"],
            patch_indices=batch["patch_indices"],
            patch_center_indices=batch["patch_center_indices"],
            patch_relative_distances=batch["patch_relative_distances"],
            precomputed_patch_attr=batch.get("patch_attr", None),
        )

        motif_out = self.motif_encoder(
            motif_token_ids_list=batch["motif_token_ids_list"],
            motif_type_ids=batch["motif_type_ids"],
            motif_lengths=batch["motif_lengths"],
            motif_pairing_ratios=batch["motif_pairing_ratios"],
            motif_gc_contents=batch["motif_gc_contents"],
            motif_edge_side_ids=batch["motif_edge_side_ids"],
            motif_is_terminal=batch["motif_is_terminal"],
            motif_center_positions=batch["motif_center_positions"],
            motif_is_dna=batch["motif_is_dna"],
            precomputed_motif_attr=batch.get("motif_attr", None),
        )

        return patch_out, motif_out

    def get_latents(self, batch: Dict[str, torch.Tensor]):
        patch_out, motif_out = self.encode_all(batch)

        z_patch = ensure_2d_latent("z_patch", patch_out["z_patch"])
        z_motif = ensure_2d_latent("z_motif", motif_out["z_motif"])

        return z_patch, z_motif

    def pair_feat_from_latent(
        self,
        z_patch: torch.Tensor,
        z_motif: torch.Tensor,
    ) -> torch.Tensor:
        n_patches, hidden_dim = z_patch.shape
        n_motifs, _ = z_motif.shape

        patch_expand = z_patch.unsqueeze(1).expand(
            n_patches,
            n_motifs,
            hidden_dim,
        )
        motif_expand = z_motif.unsqueeze(0).expand(
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

    def patch_logits(self, z_patch: torch.Tensor):
        return self.patch_importance_head(z_patch).squeeze(-1)


class ConcatMLPBaseline(SharedEncoderBackbone):
    def __init__(
        self,
        protein_encoder_cls: Type[nn.Module],
        motif_encoder_cls: Type[nn.Module],
        patch_importance_head_cls: Type[nn.Module],
        hidden_dim: int,
        dropout: float,
        pair_hidden_dim: int = 128,
    ) -> None:
        super().__init__(
            protein_encoder_cls=protein_encoder_cls,
            motif_encoder_cls=motif_encoder_cls,
            patch_importance_head_cls=patch_importance_head_cls,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )

        self.head = nn.Sequential(
            nn.Linear(hidden_dim * 4, pair_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(pair_hidden_dim, pair_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(pair_hidden_dim, 1),
        )

    def forward(self, batch: Dict[str, torch.Tensor]):
        z_patch, z_motif = self.get_latents(batch)
        pair_feat = self.pair_feat_from_latent(z_patch, z_motif)
        logits = self.head(pair_feat).squeeze(-1)

        return {
            "patch_motif_logits": logits,
            "patch_importance_logits": self.patch_logits(z_patch),
            "z_patch": z_patch,
            "z_motif": z_motif,
        }


class BilinearBaseline(SharedEncoderBackbone):
    def __init__(
        self,
        protein_encoder_cls: Type[nn.Module],
        motif_encoder_cls: Type[nn.Module],
        patch_importance_head_cls: Type[nn.Module],
        hidden_dim: int,
        dropout: float,
    ) -> None:
        super().__init__(
            protein_encoder_cls=protein_encoder_cls,
            motif_encoder_cls=motif_encoder_cls,
            patch_importance_head_cls=patch_importance_head_cls,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )

        self.patch_proj = nn.Linear(hidden_dim, hidden_dim)
        self.motif_proj = nn.Linear(hidden_dim, hidden_dim)
        self.pair_bias = nn.Parameter(torch.zeros(1))
        self.patch_bias = nn.Linear(hidden_dim, 1)
        self.motif_bias = nn.Linear(hidden_dim, 1)

    def forward(self, batch: Dict[str, torch.Tensor]):
        z_patch, z_motif = self.get_latents(batch)

        patch_proj = self.patch_proj(z_patch)
        motif_proj = self.motif_proj(z_motif)

        bilinear = patch_proj @ motif_proj.transpose(0, 1)
        patch_bias = self.patch_bias(z_patch)
        motif_bias = self.motif_bias(z_motif).transpose(0, 1)

        logits = bilinear + patch_bias + motif_bias + self.pair_bias

        return {
            "patch_motif_logits": logits,
            "patch_importance_logits": self.patch_logits(z_patch),
            "z_patch": z_patch,
            "z_motif": z_motif,
        }


class TransformerBaseline(SharedEncoderBackbone):
    def __init__(
        self,
        protein_encoder_cls: Type[nn.Module],
        motif_encoder_cls: Type[nn.Module],
        patch_importance_head_cls: Type[nn.Module],
        hidden_dim: int,
        dropout: float,
        num_heads: int,
        num_layers: int,
    ) -> None:
        super().__init__(
            protein_encoder_cls=protein_encoder_cls,
            motif_encoder_cls=motif_encoder_cls,
            patch_importance_head_cls=patch_importance_head_cls,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )

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

        self.head = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, batch: Dict[str, torch.Tensor]):
        z_patch, z_motif = self.get_latents(batch)

        patch_x = z_patch.unsqueeze(0)
        motif_x = z_motif.unsqueeze(0)

        for block in self.blocks:
            patch_x, motif_x = block(patch_x, motif_x)

        z_patch = patch_x.squeeze(0)
        z_motif = motif_x.squeeze(0)

        pair_feat = self.pair_feat_from_latent(z_patch, z_motif)
        logits = self.head(pair_feat).squeeze(-1)

        return {
            "patch_motif_logits": logits,
            "patch_importance_logits": self.patch_logits(z_patch),
            "z_patch": z_patch,
            "z_motif": z_motif,
        }


class PrototypeBaseline(SharedEncoderBackbone):
    def __init__(
        self,
        protein_encoder_cls: Type[nn.Module],
        motif_encoder_cls: Type[nn.Module],
        patch_importance_head_cls: Type[nn.Module],
        hidden_dim: int,
        dropout: float,
        num_proto_types: int = 7,
    ) -> None:
        super().__init__(
            protein_encoder_cls=protein_encoder_cls,
            motif_encoder_cls=motif_encoder_cls,
            patch_importance_head_cls=patch_importance_head_cls,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )

        self.num_proto_types = int(num_proto_types)

        self.patch_type_head = nn.Linear(hidden_dim, num_proto_types)
        self.motif_type_head = nn.Linear(hidden_dim, num_proto_types)
        self.compat = nn.Parameter(torch.randn(num_proto_types, num_proto_types) * 0.02)

    def forward(self, batch: Dict[str, torch.Tensor]):
        z_patch, z_motif = self.get_latents(batch)

        patch_type_logits = self.patch_type_head(z_patch)
        motif_type_logits = self.motif_type_head(z_motif)

        patch_type_probs = torch.softmax(patch_type_logits, dim=-1)
        motif_type_probs = torch.softmax(motif_type_logits, dim=-1)

        logits = torch.einsum(
            "ik,kq,jq->ij",
            patch_type_probs,
            self.compat,
            motif_type_probs,
        )

        return {
            "patch_motif_logits": logits,
            "patch_importance_logits": self.patch_logits(z_patch),
            "z_patch": z_patch,
            "z_motif": z_motif,
        }