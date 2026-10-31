# src/pigram/losses/grammar.py
# -*- coding: utf-8 -*-

from __future__ import annotations

from types import SimpleNamespace
from typing import Dict, Tuple

import torch
import torch.nn.functional as F


def cv2_scale(epoch: int, cfg: SimpleNamespace) -> float:
    if epoch < cfg.cv2_start_epoch:
        return 0.0

    if cfg.cv2_ramp_epochs <= 0:
        return 1.0

    return float(
        min(
            1.0,
            (epoch - cfg.cv2_start_epoch + 1) / float(cfg.cv2_ramp_epochs),
        )
    )


def compute_fine_cls_loss(
    outputs: Dict[str, torch.Tensor],
    batch: Dict[str, torch.Tensor],
    fine_rule_map: Dict[int, int],
) -> Tuple[torch.Tensor, int, Dict[str, int]]:
    zero = outputs["patch_motif_logits"].sum() * 0.0
    fine_loss_sum = zero
    total_count = 0

    fine_labels = batch["fine_label_matrix"]
    distill_mask = batch["distill_mask_matrix"]
    coarse_rule = batch["coarse_rule_matrix"]

    per_rule_counts = {}

    for rule_idx, _ in fine_rule_map.items():
        key = str(rule_idx)

        if key not in outputs["fine_cls_logits_by_rule"]:
            per_rule_counts[key] = 0
            continue

        logits = outputs["fine_cls_logits_by_rule"][key]
        mask = distill_mask & (coarse_rule == rule_idx) & (fine_labels >= 0)

        count = int(mask.sum().item())
        per_rule_counts[key] = count

        if count == 0:
            continue

        loss = F.cross_entropy(
            logits[mask],
            fine_labels[mask],
            reduction="sum",
        )
        fine_loss_sum = fine_loss_sum + loss
        total_count += count

    if total_count > 0:
        fine_loss = fine_loss_sum / total_count
    else:
        fine_loss = zero

    return fine_loss, total_count, per_rule_counts


def compute_route_regularizer(
    route_r: torch.Tensor,
    target_max: float,
) -> torch.Tensor:
    return torch.relu(route_r.mean() - target_max)


def compute_aux_grammar_loss(
    outputs: Dict[str, torch.Tensor],
    batch: Dict[str, torch.Tensor],
    cfg: SimpleNamespace,
    patch_motif_huber_loss_fn,
) -> torch.Tensor:
    return patch_motif_huber_loss_fn(
        logits=outputs["grammar_logits"],
        targets=batch["patch_motif_structure_energy_v2"],
        sigmoid_scale=cfg.sigmoid_scale,
        delta=cfg.huber_delta,
    )


def compute_stage1_loss(
    outputs: Dict[str, torch.Tensor],
    batch: Dict[str, torch.Tensor],
    cfg: SimpleNamespace,
    importance_pos_weight: float,
    fine_rule_map: Dict[int, int],
    epoch: int,
    total_loss_fn,
    patch_motif_huber_loss_fn,
    positive_cv2_loss_fn,
    sigma_summary_fn,
):
    interaction_loss = patch_motif_huber_loss_fn(
        logits=outputs["patch_motif_logits"],
        targets=batch["patch_motif_structure_energy_v2"],
        sigmoid_scale=cfg.sigmoid_scale,
        delta=cfg.huber_delta,
    )

    patch_loss_dict = total_loss_fn(
        patch_motif_logits=outputs["patch_motif_logits"],
        patch_motif_targets=batch["patch_motif_structure_energy_v2"],
        patch_importance_logits=outputs["patch_importance_logits"],
        patch_importance_labels=batch["patch_importance"],
        interaction_loss_weight=0.0,
        importance_loss_weight=cfg.importance_loss_weight,
        infonce_loss_weight=cfg.infonce_loss_weight,
        type_compat_loss_weight=0.0,
        patch_proto_div_loss_weight=0.0,
        motif_proto_div_loss_weight=0.0,
        importance_pos_weight=importance_pos_weight,
        sigmoid_scale=cfg.sigmoid_scale,
        target_weight_scale=cfg.target_weight_scale,
        patch_emb=outputs["z_patch"],
        motif_emb=outputs["z_motif"],
        infonce_pos_threshold=cfg.infonce_pos_threshold,
        infonce_temperature=cfg.infonce_temperature,
        patch_assign=None,
        motif_assign=None,
        compatibility_logits=None,
        patch_prototypes=None,
        motif_prototypes=None,
    )

    grammar_aux = compute_aux_grammar_loss(
        outputs=outputs,
        batch=batch,
        cfg=cfg,
        patch_motif_huber_loss_fn=patch_motif_huber_loss_fn,
    )

    pos_cv2_loss, pos_cv2_logs = positive_cv2_loss_fn(outputs, batch, cfg)
    fine_cls_loss, fine_count, _ = compute_fine_cls_loss(
        outputs,
        batch,
        fine_rule_map,
    )

    fine_l2 = outputs["fine_total"].pow(2).mean()

    total = (
        cfg.interaction_loss_weight * interaction_loss
        + patch_loss_dict["importance_loss"]
        + patch_loss_dict["infonce_loss"]
        + cfg.grammar_loss_weight * grammar_aux
        + cfg.fine_cls_loss_weight * fine_cls_loss
        + cfg.fine_refine_l2_weight * fine_l2
        + cv2_scale(epoch, cfg) * (cfg.pos_cv2_weight * pos_cv2_loss)
    )

    logs = {
        "total_loss": float(total.item()),
        "interaction_loss": float(interaction_loss.item()),
        "importance_loss": float(patch_loss_dict["importance_loss"].item()),
        "infonce_loss": float(patch_loss_dict["infonce_loss"].item()),
        "grammar_aux_loss": float(grammar_aux.item()),
        "fine_cls_loss": float(fine_cls_loss.item()),
        "fine_l2_loss": float(fine_l2.item()),
        "fine_count": int(fine_count),
        "fine_total_abs_mean": float(outputs["fine_total"].abs().mean().item()),
        "pos_cv2_loss": pos_cv2_logs["pos_cv2_loss"],
        "pos_cv2_selected_frac": pos_cv2_logs["pos_cv2_selected_frac"],
        "pos_usage_max": pos_cv2_logs["pos_usage_max"],
        "pos_usage_entropy_norm": pos_cv2_logs["pos_usage_entropy_norm"],
        "cv2_scale": cv2_scale(epoch, cfg),
        "head_b_aux_loss": 0.0,
        "route_reg_loss": 0.0,
        "route_mean": 0.0,
    }
    logs.update(sigma_summary_fn(outputs))

    return total, logs


def compute_stage2_loss(
    outputs: Dict[str, torch.Tensor],
    batch: Dict[str, torch.Tensor],
    cfg: SimpleNamespace,
    importance_pos_weight: float,
    fine_rule_map: Dict[int, int],
    epoch: int,
    total_loss_fn,
    patch_motif_huber_loss_fn,
    positive_cv2_loss_fn,
    sigma_summary_fn,
):
    interaction_loss = patch_motif_huber_loss_fn(
        logits=outputs["patch_motif_logits"],
        targets=batch["patch_motif_structure_energy_v2"],
        sigmoid_scale=cfg.sigmoid_scale,
        delta=cfg.huber_delta,
    )

    patch_loss_dict = total_loss_fn(
        patch_motif_logits=outputs["patch_motif_logits"],
        patch_motif_targets=batch["patch_motif_structure_energy_v2"],
        patch_importance_logits=outputs["patch_importance_logits"],
        patch_importance_labels=batch["patch_importance"],
        interaction_loss_weight=0.0,
        importance_loss_weight=cfg.importance_loss_weight,
        infonce_loss_weight=cfg.infonce_loss_weight,
        type_compat_loss_weight=0.0,
        patch_proto_div_loss_weight=0.0,
        motif_proto_div_loss_weight=0.0,
        importance_pos_weight=importance_pos_weight,
        sigmoid_scale=cfg.sigmoid_scale,
        target_weight_scale=cfg.target_weight_scale,
        patch_emb=outputs["z_patch"],
        motif_emb=outputs["z_motif"],
        infonce_pos_threshold=cfg.infonce_pos_threshold,
        infonce_temperature=cfg.infonce_temperature,
        patch_assign=None,
        motif_assign=None,
        compatibility_logits=None,
        patch_prototypes=None,
        motif_prototypes=None,
    )

    grammar_aux = compute_aux_grammar_loss(
        outputs=outputs,
        batch=batch,
        cfg=cfg,
        patch_motif_huber_loss_fn=patch_motif_huber_loss_fn,
    )

    head_b_aux = patch_motif_huber_loss_fn(
        logits=outputs["head_b_logits"],
        targets=batch["patch_motif_structure_energy_v2"],
        sigmoid_scale=cfg.sigmoid_scale,
        delta=cfg.huber_delta,
    )

    route_reg = compute_route_regularizer(
        outputs["route_r"],
        cfg.route_target_max,
    )

    pos_cv2_loss, pos_cv2_logs = positive_cv2_loss_fn(outputs, batch, cfg)
    fine_cls_loss, fine_count, _ = compute_fine_cls_loss(
        outputs,
        batch,
        fine_rule_map,
    )
    fine_l2 = outputs["fine_total"].pow(2).mean()

    total = (
        cfg.interaction_loss_weight * interaction_loss
        + patch_loss_dict["importance_loss"]
        + patch_loss_dict["infonce_loss"]
        + cfg.grammar_loss_weight * grammar_aux
        + cfg.head_b_aux_loss_weight * head_b_aux
        + cfg.route_reg_weight * route_reg
        + cfg.fine_cls_loss_weight * fine_cls_loss
        + cfg.fine_refine_l2_weight * fine_l2
        + cv2_scale(epoch, cfg) * (cfg.pos_cv2_weight * pos_cv2_loss)
    )

    logs = {
        "total_loss": float(total.item()),
        "interaction_loss": float(interaction_loss.item()),
        "importance_loss": float(patch_loss_dict["importance_loss"].item()),
        "infonce_loss": float(patch_loss_dict["infonce_loss"].item()),
        "grammar_aux_loss": float(grammar_aux.item()),
        "fine_cls_loss": float(fine_cls_loss.item()),
        "fine_l2_loss": float(fine_l2.item()),
        "fine_count": int(fine_count),
        "fine_total_abs_mean": float(outputs["fine_total"].abs().mean().item()),
        "pos_cv2_loss": pos_cv2_logs["pos_cv2_loss"],
        "pos_cv2_selected_frac": pos_cv2_logs["pos_cv2_selected_frac"],
        "pos_usage_max": pos_cv2_logs["pos_usage_max"],
        "pos_usage_entropy_norm": pos_cv2_logs["pos_usage_entropy_norm"],
        "cv2_scale": cv2_scale(epoch, cfg),
        "head_b_aux_loss": float(head_b_aux.item()),
        "route_reg_loss": float(route_reg.item()),
        "route_mean": float(outputs["route_r"].mean().item()),
    }
    logs.update(sigma_summary_fn(outputs))

    return total, logs