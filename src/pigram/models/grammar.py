# src/pigram/models/grammar.py
# -*- coding: utf-8 -*-

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn


class FineRuleRefiner(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_fine_clusters: int,
        dropout: float,
    ) -> None:
        super().__init__()

        self.num_fine_clusters = int(num_fine_clusters)

        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.cls_head = nn.Linear(hidden_dim, num_fine_clusters)
        self.delta_head = nn.Linear(hidden_dim, num_fine_clusters)

    def forward(self, x: torch.Tensor):
        h = self.shared(x)
        cls_logits = self.cls_head(h)
        delta_values = self.delta_head(h)

        q = torch.softmax(cls_logits, dim=-1)
        refine = (q * delta_values).sum(dim=-1)

        return cls_logits, delta_values, refine


class UnifiedCoarseFineModelMixin:
    def _init_fine_heads(
        self,
        fine_rule_map: Dict[int, int],
        fine_refine_scale: float,
        fine_hidden_dim: int,
        dropout: float,
    ) -> None:
        self.fine_rule_map = {int(k): int(v) for k, v in fine_rule_map.items()}
        self.fine_refine_scale = float(fine_refine_scale)

        pair_input_dim = (
            4 * self.hidden_dim
            + self.protein_encoder.patch_attr_dim
            + self.motif_encoder.motif_attr_dim
            + 3
        )

        self.fine_heads = nn.ModuleDict()

        for rule_idx, num_fine in self.fine_rule_map.items():
            self.fine_heads[str(rule_idx)] = FineRuleRefiner(
                input_dim=pair_input_dim,
                hidden_dim=fine_hidden_dim,
                num_fine_clusters=num_fine,
                dropout=dropout,
            )

    def _build_pair_base_feat(
        self,
        z_patch: torch.Tensor,
        z_motif: torch.Tensor,
        patch_attr: torch.Tensor,
        motif_attr: torch.Tensor,
    ):
        n_patches, d = z_patch.shape
        n_motifs, _ = z_motif.shape

        patch_expand = z_patch.unsqueeze(1).expand(n_patches, n_motifs, d)
        motif_expand = z_motif.unsqueeze(0).expand(n_patches, n_motifs, d)

        pair_latent = torch.cat(
            [
                patch_expand,
                motif_expand,
                patch_expand * motif_expand,
                torch.abs(patch_expand - motif_expand),
            ],
            dim=-1,
        )

        patch_attr_pair = patch_attr.unsqueeze(1).expand(
            n_patches,
            n_motifs,
            patch_attr.shape[-1],
        )
        motif_attr_pair = motif_attr.unsqueeze(0).expand(
            n_patches,
            n_motifs,
            motif_attr.shape[-1],
        )

        return pair_latent, patch_attr_pair, motif_attr_pair

    def _apply_fine(
        self,
        head_a_out: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        pair_latent, patch_attr_pair, motif_attr_pair = self._build_pair_base_feat(
            z_patch=head_a_out["z_patch"],
            z_motif=head_a_out["z_motif"],
            patch_attr=head_a_out["patch_attr"],
            motif_attr=head_a_out["motif_attr"],
        )

        abs_contrib = head_a_out["rule_contrib"].abs()
        rule_resp = abs_contrib / (abs_contrib.sum(dim=-1, keepdim=True) + 1e-8)

        fine_total = torch.zeros_like(head_a_out["grammar_logits"])
        fine_cls_logits_by_rule = {}
        fine_refine_by_rule = {}

        grammar_scalar = head_a_out["grammar_logits"].unsqueeze(-1)

        for rule_idx, head in self.fine_heads.items():
            rule_id = int(rule_idx)

            extra = torch.cat(
                [
                    pair_latent,
                    patch_attr_pair,
                    motif_attr_pair,
                    rule_resp[..., rule_id:rule_id + 1],
                    head_a_out["rule_contrib"][..., rule_id:rule_id + 1],
                    grammar_scalar,
                ],
                dim=-1,
            )

            cls_logits, delta_values, refine = head(extra)

            fine_total = (
                fine_total
                + rule_resp[..., rule_id] * refine * self.fine_refine_scale
            )

            fine_cls_logits_by_rule[rule_idx] = cls_logits
            fine_refine_by_rule[rule_idx] = refine

        out = dict(head_a_out)
        out["fine_total"] = fine_total
        out["fine_head_a_logits"] = head_a_out["head_a_logits"] + fine_total
        out["fine_cls_logits_by_rule"] = fine_cls_logits_by_rule
        out["fine_refine_by_rule"] = fine_refine_by_rule

        return out

    def forward_stage1(self, schedule_progress: float = 1.0, **kwargs):
        patch_out, motif_out = self.encode_all(**kwargs)
        head_a_out = self.forward_head_a(
            patch_out,
            motif_out,
            schedule_progress=schedule_progress,
        )

        head_a_out = self._apply_fine(head_a_out)

        out = dict(head_a_out)
        out["patch_motif_logits"] = head_a_out["fine_head_a_logits"]
        out["head_a_logits"] = head_a_out["fine_head_a_logits"]

        return out

    def forward_stage2(self, schedule_progress: float = 1.0, **kwargs):
        patch_out, motif_out = self.encode_all(**kwargs)
        head_a_out = self.forward_head_a(
            patch_out,
            motif_out,
            schedule_progress=schedule_progress,
        )

        head_a_out = self._apply_fine(head_a_out)

        head_b_logits = self.head_b(
            head_a_out["z_patch"],
            head_a_out["z_motif"],
        )

        route_r = self._compute_route(
            patch_rule_activations=head_a_out["patch_rule_activations"],
            motif_rule_activations=head_a_out["motif_rule_activations"],
            grammar_logits=head_a_out["fine_head_a_logits"],
            head_a_logits=head_a_out["fine_head_a_logits"],
            top_rule_fraction=head_a_out["top_rule_fraction"],
            rule_entropy=head_a_out["rule_entropy"],
        )

        final_logits = (
            (1.0 - route_r) * head_a_out["fine_head_a_logits"]
            + route_r * head_b_logits
        )

        out = dict(head_a_out)
        out["patch_motif_logits"] = final_logits
        out["head_b_logits"] = head_b_logits
        out["route_r"] = route_r
        out["head_a_logits"] = head_a_out["fine_head_a_logits"]

        return out


def build_unified_coarse_fine_model_class(base_model_cls):
    class UnifiedCoarseFineModel(UnifiedCoarseFineModelMixin, base_model_cls):
        def __init__(
            self,
            fine_rule_map: Dict[int, int],
            fine_refine_scale: float = 0.5,
            fine_hidden_dim: int = 128,
            **kwargs,
        ) -> None:
            super().__init__(**kwargs)

            self._init_fine_heads(
                fine_rule_map=fine_rule_map,
                fine_refine_scale=fine_refine_scale,
                fine_hidden_dim=fine_hidden_dim,
                dropout=kwargs.get("dropout", 0.1),
            )

    return UnifiedCoarseFineModel