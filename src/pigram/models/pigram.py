# src/pigram/models/pigram.py
# -*- coding: utf-8 -*-

from __future__ import annotations

from typing import Callable, Dict

import numpy as np
import torch
import torch.nn as nn


def compute_rule_confidence_features(
    out: Dict[str, torch.Tensor],
    num_rules: int,
) -> Dict[str, torch.Tensor]:
    h_a = out["head_a_logits"]
    grammar = out["grammar_logits"]

    if "fine_total" in out:
        fine_total = out["fine_total"]
    else:
        fine_total = h_a - grammar

    if "rule_contrib" in out:
        rule_contrib = out["rule_contrib"]
        abs_rule_contrib = rule_contrib.abs()

        top_vals, _ = torch.topk(abs_rule_contrib, k=min(2, num_rules), dim=-1)
        top1 = top_vals[..., 0]

        if top_vals.shape[-1] > 1:
            top2 = top_vals[..., 1]
        else:
            top2 = torch.zeros_like(top1)

        top_margin = (top1 - top2) / (top1 + 1e-8)

        prob = abs_rule_contrib / (abs_rule_contrib.sum(dim=-1, keepdim=True) + 1e-8)
        entropy = -(prob * (prob + 1e-8).log()).sum(dim=-1) / np.log(num_rules)
        top_rule_fraction = top1 / (abs_rule_contrib.sum(dim=-1) + 1e-8)
    else:
        top_margin = torch.zeros_like(h_a)
        entropy = torch.zeros_like(h_a)
        top_rule_fraction = torch.zeros_like(h_a)

    return {
        "fine_total": fine_total.detach(),
        "top_margin": top_margin.detach(),
        "entropy": entropy.detach(),
        "top_rule_fraction": top_rule_fraction.detach(),
    }


class BudgetedRescueRouter(nn.Module):
    def __init__(
        self,
        num_rules: int,
        hidden_dim: int,
        dropout: float,
        route_bias_init: float,
        route_max: float,
    ) -> None:
        super().__init__()

        self.num_rules = int(num_rules)
        self.route_max = float(route_max)

        in_dim = 2 * self.num_rules + 8

        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

        nn.init.constant_(self.net[-1].bias, route_bias_init)

    def forward(
        self,
        out: Dict[str, torch.Tensor],
        sigmoid_scale: float,
    ) -> torch.Tensor:
        h_a = out["head_a_logits"].detach()
        grammar = out["grammar_logits"].detach()

        n_p, n_m = h_a.shape

        if "patch_rule_activations" in out and "motif_rule_activations" in out:
            u = out["patch_rule_activations"].detach()
            v = out["motif_rule_activations"].detach()
        elif "rule_contrib" in out:
            rule_abs = out["rule_contrib"].detach().abs()
            u = rule_abs.mean(dim=1)
            v = rule_abs.mean(dim=0)
        else:
            u = torch.zeros(n_p, self.num_rules, device=h_a.device)
            v = torch.zeros(n_m, self.num_rules, device=h_a.device)

        u_pair = u.unsqueeze(1).expand(n_p, n_m, self.num_rules)
        v_pair = v.unsqueeze(0).expand(n_p, n_m, self.num_rules)

        conf = compute_rule_confidence_features(out, self.num_rules)

        fine_total = conf["fine_total"]
        top_margin = conf["top_margin"]
        entropy = conf["entropy"]
        top_rule_fraction = conf["top_rule_fraction"]

        p_a = torch.sigmoid(h_a / sigmoid_scale)
        prob_conf = (p_a - 0.5).abs()

        scalar_feats = torch.stack(
            [
                grammar.abs(),
                h_a.abs(),
                fine_total.abs(),
                top_margin,
                entropy,
                1.0 / (1.0 + entropy),
                top_rule_fraction,
                prob_conf,
            ],
            dim=-1,
        )

        feat = torch.cat([u_pair, v_pair, scalar_feats], dim=-1)

        raw_gate = torch.sigmoid(self.net(feat).squeeze(-1))
        route = self.route_max * raw_gate

        return route


class FrozenGrammarResidualRescue(nn.Module):
    def __init__(
        self,
        foundation_model: nn.Module,
        foundation_forward_fn: Callable[[nn.Module, Dict[str, torch.Tensor]], Dict[str, torch.Tensor]],
        delta_head_cls: type[nn.Module],
        hidden_dim: int,
        num_heads: int,
        num_layers: int,
        num_rules: int,
        dropout: float,
        delta_clip: float,
        route_bias_init: float,
        route_max: float,
    ) -> None:
        super().__init__()

        self.foundation = foundation_model
        self.foundation_forward_fn = foundation_forward_fn
        self.num_rules = int(num_rules)
        self.delta_clip = float(delta_clip)

        self.delta_head = delta_head_cls(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout,
        )

        self.router = BudgetedRescueRouter(
            num_rules=num_rules,
            hidden_dim=hidden_dim,
            dropout=dropout,
            route_bias_init=route_bias_init,
            route_max=route_max,
        )

    def frozen_forward(
        self,
        batch: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        with torch.no_grad():
            out = self.foundation_forward_fn(self.foundation, batch)

        return out

    def forward(
        self,
        batch: Dict[str, torch.Tensor],
        sigmoid_scale: float,
    ) -> Dict[str, torch.Tensor]:
        out = self.frozen_forward(batch)

        h_a = out["head_a_logits"].detach()
        grammar = out["grammar_logits"].detach()

        z_patch = out["z_patch"].detach()
        z_motif = out["z_motif"].detach()

        delta_raw = self.delta_head(z_patch, z_motif)

        if self.delta_clip is not None and self.delta_clip > 0:
            delta = self.delta_clip * torch.tanh(delta_raw / self.delta_clip)
        else:
            delta = delta_raw

        route = self.router(out, sigmoid_scale=sigmoid_scale)
        correction = route * delta
        final_logits = h_a + correction

        y = batch["patch_motif_structure_energy_v2"]

        p_a = torch.sigmoid(h_a / sigmoid_scale)
        p_final = torch.sigmoid(final_logits / sigmoid_scale)

        err_a = (p_a - y).abs().detach()
        err_final = (p_final - y).abs().detach()

        return {
            "foundation_out": out,
            "grammar_logits": grammar,
            "head_a_logits": h_a,
            "delta_logits": delta,
            "delta_raw_logits": delta_raw,
            "route_r": route,
            "correction_logits": correction,
            "patch_motif_logits": final_logits,
            "p_a": p_a,
            "p_final": p_final,
            "err_a": err_a,
            "err_final": err_final,
        }