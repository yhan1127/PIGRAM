# src/pigram/models/grammar_head.py
# -*- coding: utf-8 -*-

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


class DualHeadAsymmetricSigmaRBFAttributeGrammar(nn.Module):
    """
    Sign-separated asymmetric RBF grammar head.

    This grammar head maps explicit patch/motif attributes into rule-specific
    vectors and scores patch--motif compatibility through RBF-like similarity.

    Output convention:
        negative rules are suppressive
        positive rules are favorable

    Returned keys match the coarse/grammar training pipeline:
        grammar_logits
        neg_grammar_logits
        pos_grammar_logits
        rule_contrib
        neg_rule_contrib
        pos_rule_contrib
        patch_rule_activations
        motif_rule_activations
        rbf_response
        sigma
        sigma_schedule
        sigma_log_delta
        neg_rule_weight
        pos_rule_weight
    """

    def __init__(
        self,
        patch_attr_dim: int,
        motif_attr_dim: int,
        num_neg_rules: int = 3,
        num_pos_rules: int = 4,
        rule_dim: int = 8,
        hidden_dim: int = 128,
        dropout: float = 0.1,
        neg_sigma_min: float = 0.10,
        neg_sigma_max: float = 0.80,
        pos_sigma_min: float = 1.20,
        pos_sigma_max: float = 2.40,
        sigma_residual_clip: float = 0.50,
        use_linear_terms: bool = True,
    ) -> None:
        super().__init__()

        self.patch_attr_dim = int(patch_attr_dim)
        self.motif_attr_dim = int(motif_attr_dim)
        self.num_neg_rules = int(num_neg_rules)
        self.num_pos_rules = int(num_pos_rules)
        self.num_rules = self.num_neg_rules + self.num_pos_rules
        self.rule_dim = int(rule_dim)

        self.neg_sigma_min = float(neg_sigma_min)
        self.neg_sigma_max = float(neg_sigma_max)
        self.pos_sigma_min = float(pos_sigma_min)
        self.pos_sigma_max = float(pos_sigma_max)
        self.sigma_residual_clip = float(sigma_residual_clip)
        self.use_linear_terms = bool(use_linear_terms)

        self.neg_patch_proj = nn.Sequential(
            nn.Linear(self.patch_attr_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, self.num_neg_rules * self.rule_dim),
        )
        self.neg_motif_proj = nn.Sequential(
            nn.Linear(self.motif_attr_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, self.num_neg_rules * self.rule_dim),
        )

        self.pos_patch_proj = nn.Sequential(
            nn.Linear(self.patch_attr_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, self.num_pos_rules * self.rule_dim),
        )
        self.pos_motif_proj = nn.Sequential(
            nn.Linear(self.motif_attr_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, self.num_pos_rules * self.rule_dim),
        )

        self.neg_rule_weight_raw = nn.Parameter(torch.zeros(self.num_neg_rules))
        self.pos_rule_weight_raw = nn.Parameter(torch.zeros(self.num_pos_rules))

        self.sigma_log_delta = nn.Parameter(torch.zeros(self.num_rules))

        if self.use_linear_terms:
            self.patch_linear = nn.Linear(self.patch_attr_dim, 1)
            self.motif_linear = nn.Linear(self.motif_attr_dim, 1)

    def _sigma_schedule(
        self,
        schedule_progress: float,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        progress = float(max(0.0, min(1.0, schedule_progress)))

        neg_sigma = self.neg_sigma_max + progress * (self.neg_sigma_min - self.neg_sigma_max)
        pos_sigma = self.pos_sigma_max + progress * (self.pos_sigma_min - self.pos_sigma_max)

        neg = torch.full(
            (self.num_neg_rules,),
            float(neg_sigma),
            device=device,
            dtype=dtype,
        )
        pos = torch.full(
            (self.num_pos_rules,),
            float(pos_sigma),
            device=device,
            dtype=dtype,
        )

        return torch.cat([neg, pos], dim=0)

    def _compute_sigma(
        self,
        schedule_progress: float,
        device: torch.device,
        dtype: torch.dtype,
    ):
        sigma_schedule = self._sigma_schedule(
            schedule_progress=schedule_progress,
            device=device,
            dtype=dtype,
        )

        delta = self.sigma_residual_clip * torch.tanh(
            self.sigma_log_delta.to(device=device, dtype=dtype)
        )

        sigma = sigma_schedule + delta
        sigma = torch.clamp(sigma, min=0.05, max=10.0)

        return sigma, sigma_schedule

    def _compute_branch(
        self,
        patch_attr: torch.Tensor,
        motif_attr: torch.Tensor,
        patch_proj: nn.Module,
        motif_proj: nn.Module,
        num_rules: int,
        sigma: torch.Tensor,
        rule_weight_raw: torch.Tensor,
        sign: float,
    ) -> Dict[str, torch.Tensor]:
        n_patches = patch_attr.shape[0]
        n_motifs = motif_attr.shape[0]

        patch_rule_vec = patch_proj(patch_attr).view(
            n_patches,
            num_rules,
            self.rule_dim,
        )
        motif_rule_vec = motif_proj(motif_attr).view(
            n_motifs,
            num_rules,
            self.rule_dim,
        )

        diff = (
            patch_rule_vec.unsqueeze(1)
            - motif_rule_vec.unsqueeze(0)
        )
        sq_dist = diff.pow(2).mean(dim=-1)

        sigma = sigma.view(1, 1, num_rules)
        rbf_response = torch.exp(
            -sq_dist / (2.0 * sigma.pow(2) + 1e-8)
        )

        rule_weight = F.softplus(rule_weight_raw)

        signed_rule_contrib = (
            float(sign)
            * rbf_response
            * rule_weight.view(1, 1, num_rules)
        )

        branch_logits = signed_rule_contrib.sum(dim=-1)

        patch_rule_activations = torch.norm(patch_rule_vec, dim=-1)
        motif_rule_activations = torch.norm(motif_rule_vec, dim=-1)

        return {
            "branch_logits": branch_logits,
            "rule_contrib": signed_rule_contrib,
            "rbf_response": rbf_response,
            "patch_rule_activations": patch_rule_activations,
            "motif_rule_activations": motif_rule_activations,
            "rule_weight": rule_weight,
        }

    def forward(
        self,
        patch_attr: torch.Tensor,
        motif_attr: torch.Tensor,
        schedule_progress: float = 1.0,
    ) -> Dict[str, torch.Tensor]:
        patch_attr = patch_attr.float()
        motif_attr = motif_attr.float()

        sigma, sigma_schedule = self._compute_sigma(
            schedule_progress=schedule_progress,
            device=patch_attr.device,
            dtype=patch_attr.dtype,
        )

        neg_sigma = sigma[: self.num_neg_rules]
        pos_sigma = sigma[self.num_neg_rules :]

        neg_out = self._compute_branch(
            patch_attr=patch_attr,
            motif_attr=motif_attr,
            patch_proj=self.neg_patch_proj,
            motif_proj=self.neg_motif_proj,
            num_rules=self.num_neg_rules,
            sigma=neg_sigma,
            rule_weight_raw=self.neg_rule_weight_raw,
            sign=-1.0,
        )
        pos_out = self._compute_branch(
            patch_attr=patch_attr,
            motif_attr=motif_attr,
            patch_proj=self.pos_patch_proj,
            motif_proj=self.pos_motif_proj,
            num_rules=self.num_pos_rules,
            sigma=pos_sigma,
            rule_weight_raw=self.pos_rule_weight_raw,
            sign=1.0,
        )

        neg_grammar_logits = neg_out["branch_logits"]
        pos_grammar_logits = pos_out["branch_logits"]
        grammar_logits = neg_grammar_logits + pos_grammar_logits

        if self.use_linear_terms:
            patch_term = self.patch_linear(patch_attr).expand(-1, motif_attr.shape[0])
            motif_term = self.motif_linear(motif_attr).transpose(0, 1).expand(patch_attr.shape[0], -1)
            grammar_logits = grammar_logits + patch_term + motif_term

        rule_contrib = torch.cat(
            [
                neg_out["rule_contrib"],
                pos_out["rule_contrib"],
            ],
            dim=-1,
        )

        patch_rule_activations = torch.cat(
            [
                neg_out["patch_rule_activations"],
                pos_out["patch_rule_activations"],
            ],
            dim=-1,
        )
        motif_rule_activations = torch.cat(
            [
                neg_out["motif_rule_activations"],
                pos_out["motif_rule_activations"],
            ],
            dim=-1,
        )

        rbf_response = torch.cat(
            [
                neg_out["rbf_response"],
                pos_out["rbf_response"],
            ],
            dim=-1,
        )

        return {
            "grammar_logits": grammar_logits,
            "neg_grammar_logits": neg_grammar_logits,
            "pos_grammar_logits": pos_grammar_logits,
            "rule_contrib": rule_contrib,
            "neg_rule_contrib": neg_out["rule_contrib"],
            "pos_rule_contrib": pos_out["rule_contrib"],
            "patch_rule_activations": patch_rule_activations,
            "motif_rule_activations": motif_rule_activations,
            "rbf_response": rbf_response,
            "sigma": sigma,
            "sigma_schedule": sigma_schedule,
            "sigma_log_delta": self.sigma_log_delta,
            "neg_rule_weight": neg_out["rule_weight"],
            "pos_rule_weight": pos_out["rule_weight"],
        }