# src/pigram/losses/router.py
# -*- coding: utf-8 -*-

from __future__ import annotations

from types import SimpleNamespace
from typing import Dict, Tuple

import torch
import torch.nn.functional as F


def masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if mask.sum().item() == 0:
        return torch.zeros((), device=x.device, dtype=x.dtype)

    return x[mask].mean()


def rescue_loss(
    out: Dict[str, torch.Tensor],
    batch: Dict[str, torch.Tensor],
    args: SimpleNamespace,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    y = batch["patch_motif_structure_energy_v2"]

    weight = args.err_floor + out["err_a"].pow(args.err_gamma)
    weight = weight / (weight.mean().detach() + 1e-8)

    pred_map = F.smooth_l1_loss(
        out["p_final"],
        y,
        beta=args.huber_delta,
        reduction="none",
    )
    pred_loss = (weight * pred_map).mean()

    sparse_loss = out["route_r"].mean()
    corr_l2 = out["correction_logits"].pow(2).mean()
    delta_l2 = out["delta_logits"].pow(2).mean()

    route_mean = out["route_r"].mean()
    budget_loss = F.relu(route_mean - args.route_budget).pow(2)

    route_cap_loss = F.relu(out["route_r"] - args.route_cap).pow(2).mean()

    err_final_for_grad = (out["p_final"] - y).abs()
    err_a = out["err_a"].detach()

    easy_threshold = torch.quantile(
        err_a.reshape(-1),
        args.easy_noharm_quantile,
    )
    easy_mask = err_a <= easy_threshold

    easy_noharm_map = F.relu(err_final_for_grad - err_a).pow(2)
    easy_noharm_loss = masked_mean(easy_noharm_map, easy_mask)

    total = (
        pred_loss
        + args.lambda_sparse * sparse_loss
        + args.lambda_corr_l2 * corr_l2
        + args.lambda_delta_l2 * delta_l2
        + args.lambda_budget * budget_loss
        + args.lambda_route_cap * route_cap_loss
        + args.lambda_easy_noharm * easy_noharm_loss
    )

    logs = {
        "total_loss": float(total.item()),
        "pred_loss": float(pred_loss.item()),
        "sparse_loss_raw": float(sparse_loss.item()),
        "corr_l2_raw": float(corr_l2.item()),
        "delta_l2_raw": float(delta_l2.item()),
        "budget_loss_raw": float(budget_loss.item()),
        "route_cap_loss_raw": float(route_cap_loss.item()),
        "easy_noharm_loss_raw": float(easy_noharm_loss.item()),
        "route_mean": float(out["route_r"].mean().item()),
        "route_max": float(out["route_r"].max().item()),
        "delta_abs_mean": float(out["delta_logits"].abs().mean().item()),
        "correction_abs_mean": float(out["correction_logits"].abs().mean().item()),
        "err_a_mean": float(out["err_a"].mean().item()),
        "err_final_mean": float(out["err_final"].mean().item()),
        "err_reduction_mean": float((out["err_a"] - out["err_final"]).mean().item()),
        "easy_threshold_batch": float(easy_threshold.item()),
    }

    return total, logs