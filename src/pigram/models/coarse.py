# src/pigram/models/coarse.py
# -*- coding: utf-8 -*-

from __future__ import annotations

from typing import Dict, List, Type

import torch
import torch.nn as nn


class DualHeadCoarseModel(nn.Module):
    """
    Coarse sign-separated grammar teacher.

    This class keeps the original architecture logic:
      - protein patch encoder
      - ssNA motif encoder
      - asymmetric sign-separated grammar head
      - patch importance head
      - Transformer fallback head
      - route MLP for stage-2 interpolation

    Component classes are injected so this module can be imported before we
    finish moving legacy encoders and grammar heads into src/pigram.
    """

    def __init__(
        self,
        protein_encoder_cls: Type[nn.Module],
        motif_encoder_cls: Type[nn.Module],
        grammar_head_cls: Type[nn.Module],
        patch_importance_head_cls: Type[nn.Module],
        fallback_head_cls: Type[nn.Module],
        protein_vocab_size: int = 32,
        rna_vocab_size: int = 16,
        motif_type_vocab_size: int = 8,
        hidden_dim: int = 128,
        num_heads: int = 8,
        num_layers: int = 3,
        num_neg_rules: int = 3,
        num_pos_rules: int = 4,
        rule_dim: int = 8,
        dropout: float = 0.1,
        route_bias_init: float = -3.0,
        neg_sigma_min: float = 0.10,
        neg_sigma_max: float = 0.80,
        pos_sigma_min: float = 1.20,
        pos_sigma_max: float = 2.40,
        sigma_residual_clip: float = 0.50,
    ) -> None:
        super().__init__()

        self.hidden_dim = int(hidden_dim)
        self.num_neg_rules = int(num_neg_rules)
        self.num_pos_rules = int(num_pos_rules)
        self.num_rules = self.num_neg_rules + self.num_pos_rules

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

        self.grammar_head = grammar_head_cls(
            patch_attr_dim=self.protein_encoder.patch_attr_dim,
            motif_attr_dim=self.motif_encoder.motif_attr_dim,
            num_neg_rules=num_neg_rules,
            num_pos_rules=num_pos_rules,
            rule_dim=rule_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
            neg_sigma_min=neg_sigma_min,
            neg_sigma_max=neg_sigma_max,
            pos_sigma_min=pos_sigma_min,
            pos_sigma_max=pos_sigma_max,
            sigma_residual_clip=sigma_residual_clip,
            use_linear_terms=True,
        )

        self.patch_importance_head = patch_importance_head_cls(
            hidden_dim=hidden_dim,
            dropout=dropout,
        )

        self.head_b = fallback_head_cls(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout,
        )

        route_dim = self.num_rules * 2 + 5

        self.route_mlp = nn.Sequential(
            nn.Linear(route_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 4, 1),
        )

        nn.init.constant_(self.route_mlp[-1].bias, route_bias_init)

    @staticmethod
    def _rule_stats(rule_contrib: torch.Tensor):
        abs_contrib = rule_contrib.abs() + 1e-8
        prob = abs_contrib / abs_contrib.sum(dim=-1, keepdim=True)

        top_rule_fraction = prob.max(dim=-1).values
        rule_entropy = -(prob * torch.log(prob + 1e-8)).sum(dim=-1)

        return top_rule_fraction, rule_entropy

    def _compute_route(
        self,
        patch_rule_activations: torch.Tensor,
        motif_rule_activations: torch.Tensor,
        grammar_logits: torch.Tensor,
        head_a_logits: torch.Tensor,
        top_rule_fraction: torch.Tensor,
        rule_entropy: torch.Tensor,
    ) -> torch.Tensor:
        u_pair = patch_rule_activations.unsqueeze(1).expand(
            -1,
            motif_rule_activations.shape[0],
            -1,
        )
        v_pair = motif_rule_activations.unsqueeze(0).expand(
            patch_rule_activations.shape[0],
            -1,
            -1,
        )

        grammar_abs = grammar_logits.abs().unsqueeze(-1)
        head_a_abs = head_a_logits.abs().unsqueeze(-1)
        top_frac = top_rule_fraction.unsqueeze(-1)
        entropy = rule_entropy.unsqueeze(-1)
        inv_entropy = (1.0 / (1.0 + rule_entropy)).unsqueeze(-1)

        feat = torch.cat(
            [
                u_pair,
                v_pair,
                grammar_abs,
                head_a_abs,
                top_frac,
                entropy,
                inv_entropy,
            ],
            dim=-1,
        )

        return torch.sigmoid(self.route_mlp(feat).squeeze(-1))

    def encode_all(
        self,
        protein_token_ids: torch.Tensor,
        patch_indices: torch.Tensor,
        patch_center_indices: torch.Tensor,
        patch_relative_distances: torch.Tensor,
        motif_token_ids_list: List[torch.Tensor],
        motif_type_ids: torch.Tensor,
        motif_lengths: torch.Tensor,
        motif_pairing_ratios: torch.Tensor,
        motif_gc_contents: torch.Tensor,
        motif_edge_side_ids: torch.Tensor,
        motif_is_terminal: torch.Tensor,
        motif_center_positions: torch.Tensor,
        motif_is_dna: torch.Tensor,
        patch_attr: torch.Tensor | None = None,
        motif_attr: torch.Tensor | None = None,
    ):
        patch_out = self.protein_encoder(
            protein_token_ids=protein_token_ids,
            patch_indices=patch_indices,
            patch_center_indices=patch_center_indices,
            patch_relative_distances=patch_relative_distances,
            precomputed_patch_attr=patch_attr,
        )

        motif_out = self.motif_encoder(
            motif_token_ids_list=motif_token_ids_list,
            motif_type_ids=motif_type_ids,
            motif_lengths=motif_lengths,
            motif_pairing_ratios=motif_pairing_ratios,
            motif_gc_contents=motif_gc_contents,
            motif_edge_side_ids=motif_edge_side_ids,
            motif_is_terminal=motif_is_terminal,
            motif_center_positions=motif_center_positions,
            motif_is_dna=motif_is_dna,
            precomputed_motif_attr=motif_attr,
        )

        return patch_out, motif_out

    def forward_head_a(
        self,
        patch_out: Dict,
        motif_out: Dict,
        schedule_progress: float = 1.0,
    ) -> Dict[str, torch.Tensor]:
        z_patch = patch_out["z_patch"]
        z_motif = motif_out["z_motif"]
        patch_attr = patch_out["patch_attr"]
        motif_attr = motif_out["motif_attr"]

        grammar_out = self.grammar_head(
            patch_attr=patch_attr,
            motif_attr=motif_attr,
            schedule_progress=schedule_progress,
        )

        grammar_logits = grammar_out["grammar_logits"]
        patch_importance_logits = self.patch_importance_head(z_patch).squeeze(-1)

        top_rule_fraction, rule_entropy = self._rule_stats(
            grammar_out["rule_contrib"]
        )

        return {
            "patch_motif_logits": grammar_logits,
            "head_a_logits": grammar_logits,
            "grammar_logits": grammar_logits,
            "neg_grammar_logits": grammar_out["neg_grammar_logits"],
            "pos_grammar_logits": grammar_out["pos_grammar_logits"],
            "patch_importance_logits": patch_importance_logits,
            "z_patch": z_patch,
            "z_motif": z_motif,
            "patch_attr": patch_attr,
            "motif_attr": motif_attr,
            "rule_contrib": grammar_out["rule_contrib"],
            "neg_rule_contrib": grammar_out["neg_rule_contrib"],
            "pos_rule_contrib": grammar_out["pos_rule_contrib"],
            "patch_rule_activations": grammar_out["patch_rule_activations"],
            "motif_rule_activations": grammar_out["motif_rule_activations"],
            "rbf_response": grammar_out["rbf_response"],
            "sigma": grammar_out["sigma"],
            "sigma_schedule": grammar_out["sigma_schedule"],
            "sigma_log_delta": grammar_out["sigma_log_delta"],
            "top_rule_fraction": top_rule_fraction,
            "rule_entropy": rule_entropy,
            "neg_rule_weight": grammar_out["neg_rule_weight"],
            "pos_rule_weight": grammar_out["pos_rule_weight"],
        }

    def forward_stage1(
        self,
        schedule_progress: float = 1.0,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        patch_out, motif_out = self.encode_all(**kwargs)

        return self.forward_head_a(
            patch_out,
            motif_out,
            schedule_progress=schedule_progress,
        )

    def forward_stage2(
        self,
        schedule_progress: float = 1.0,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        patch_out, motif_out = self.encode_all(**kwargs)

        head_a_out = self.forward_head_a(
            patch_out,
            motif_out,
            schedule_progress=schedule_progress,
        )

        head_b_logits = self.head_b(
            head_a_out["z_patch"],
            head_a_out["z_motif"],
        )

        route_r = self._compute_route(
            patch_rule_activations=head_a_out["patch_rule_activations"],
            motif_rule_activations=head_a_out["motif_rule_activations"],
            grammar_logits=head_a_out["grammar_logits"],
            head_a_logits=head_a_out["head_a_logits"],
            top_rule_fraction=head_a_out["top_rule_fraction"],
            rule_entropy=head_a_out["rule_entropy"],
        )

        final_logits = (
            (1.0 - route_r) * head_a_out["head_a_logits"]
            + route_r * head_b_logits
        )

        out = dict(head_a_out)
        out["patch_motif_logits"] = final_logits
        out["head_b_logits"] = head_b_logits
        out["route_r"] = route_r

        return out