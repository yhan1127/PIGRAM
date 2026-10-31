# src/pigram/losses/regularizer.py
# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import Any, Dict, Tuple
import numpy as np
import torch


def _zero_like_reference(reference: torch.Tensor) -> torch.Tensor:
    return reference.sum() * 0.0


def compute_route_regularizer(
    route_r: torch.Tensor,
    target_max: float,
) -> torch.Tensor:
    """
    Penalize route usage above a desired target.

    This is used in stage-2 coarse/grammar training.
    """
    return torch.relu(route_r.mean() - float(target_max))


def compute_positive_cv2_loss(
    outputs: Dict[str, torch.Tensor],
    batch: Dict[str, torch.Tensor],
    cfg: Any,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Balance positive-rule usage on high-target patch--motif pairs.

    The coarse grammar has suppressive and favorable rules. This regularizer
    looks only at favorable-rule contributions on semantically high target
    pairs and penalizes collapsed use of a single favorable rule.

    Required outputs:
        pos_rule_contrib: [N_patch, N_motif, N_pos_rules]

    Required batch:
        patch_motif_structure_energy_v2: [N_patch, N_motif]

    Config fields:
        semantic_high_target_threshold
        semantic_min_pairs
    """
    reference = outputs["patch_motif_logits"]

    if "pos_rule_contrib" not in outputs:
        zero = _zero_like_reference(reference)
        return zero, {
            "pos_cv2_loss": 0.0,
            "pos_cv2_selected_frac": 0.0,
            "pos_usage_max": 0.0,
            "pos_usage_entropy_norm": 0.0,
        }

    pos_rule_contrib = outputs["pos_rule_contrib"]
    y = batch["patch_motif_structure_energy_v2"].float()

    if pos_rule_contrib.ndim != 3:
        raise RuntimeError(
            f"pos_rule_contrib must be 3D, got {tuple(pos_rule_contrib.shape)}"
        )

    if y.shape != pos_rule_contrib.shape[:2]:
        raise RuntimeError(
            f"CV2 target shape mismatch: y={tuple(y.shape)}, "
            f"pos_rule_contrib={tuple(pos_rule_contrib.shape)}"
        )

    threshold = float(getattr(cfg, "semantic_high_target_threshold", 0.50))
    min_pairs = int(getattr(cfg, "semantic_min_pairs", 4))

    mask = y >= threshold
    selected_count = int(mask.sum().item())
    total_count = int(mask.numel())

    if selected_count < min_pairs:
        zero = _zero_like_reference(reference)
        return zero, {
            "pos_cv2_loss": 0.0,
            "pos_cv2_selected_frac": float(selected_count / max(total_count, 1)),
            "pos_usage_max": 0.0,
            "pos_usage_entropy_norm": 0.0,
        }

    selected = pos_rule_contrib[mask].abs()  # [N_selected, N_pos_rules]
    usage = selected.mean(dim=0)

    usage_mean = usage.mean()
    usage_var = usage.var(unbiased=False)
    cv2 = usage_var / (usage_mean.pow(2) + 1e-8)

    usage_prob = usage / (usage.sum() + 1e-8)
    entropy = -(usage_prob * (usage_prob + 1e-8).log()).sum()
    n_rules = int(usage_prob.numel())

    if n_rules > 1:
        entropy_norm = entropy / np.log(float(n_rules))
    else:
        entropy_norm = entropy * 0.0

    logs = {
        "pos_cv2_loss": float(cv2.detach().item()),
        "pos_cv2_selected_frac": float(selected_count / max(total_count, 1)),
        "pos_usage_max": float(usage_prob.max().detach().item()),
        "pos_usage_entropy_norm": float(entropy_norm.detach().item()),
    }

    return cv2, logs


def summarize_sigma(outputs: Dict[str, torch.Tensor]) -> Dict[str, float]:
    """
    Summarize sigma values from the explicit grammar head.

    Expected keys:
        sigma
        sigma_schedule
        sigma_log_delta
        neg_rule_contrib
        pos_rule_contrib
    """
    if "sigma" not in outputs:
        return {
            "sigma_neg_mean": 0.0,
            "sigma_pos_mean": 0.0,
            "sigma_sched_neg_mean": 0.0,
            "sigma_sched_pos_mean": 0.0,
            "sigma_delta_abs_mean": 0.0,
            "sigma_delta_neg_mean": 0.0,
            "sigma_delta_pos_mean": 0.0,
        }

    sigma = outputs["sigma"].detach().float().reshape(-1)

    if "sigma_schedule" in outputs:
        sigma_schedule = outputs["sigma_schedule"].detach().float().reshape(-1)
    else:
        sigma_schedule = torch.zeros_like(sigma)

    delta = sigma - sigma_schedule

    if "neg_rule_contrib" in outputs:
        n_neg = int(outputs["neg_rule_contrib"].shape[-1])
    else:
        n_neg = min(3, int(sigma.numel()))

    n_neg = max(0, min(n_neg, int(sigma.numel())))

    sigma_neg = sigma[:n_neg]
    sigma_pos = sigma[n_neg:]

    sched_neg = sigma_schedule[:n_neg]
    sched_pos = sigma_schedule[n_neg:]

    delta_neg = delta[:n_neg]
    delta_pos = delta[n_neg:]

    def mean_or_zero(x: torch.Tensor) -> float:
        if x.numel() == 0:
            return 0.0
        return float(x.mean().item())

    return {
        "sigma_neg_mean": mean_or_zero(sigma_neg),
        "sigma_pos_mean": mean_or_zero(sigma_pos),
        "sigma_sched_neg_mean": mean_or_zero(sched_neg),
        "sigma_sched_pos_mean": mean_or_zero(sched_pos),
        "sigma_delta_abs_mean": float(delta.abs().mean().item()) if delta.numel() > 0 else 0.0,
        "sigma_delta_neg_mean": mean_or_zero(delta_neg),
        "sigma_delta_pos_mean": mean_or_zero(delta_pos),
    }