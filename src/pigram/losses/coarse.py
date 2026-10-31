# src/pigram/losses/coarse.py
# -*- coding: utf-8 -*-

from __future__ import annotations

from types import SimpleNamespace
from typing import Dict

import torch


def linear_ramp(epoch: int, start_epoch: int, ramp_epochs: int) -> float:
    if epoch < start_epoch:
        return 0.0

    if ramp_epochs <= 0:
        return 1.0

    return float(
        min(
            1.0,
            (epoch - start_epoch + 1) / float(ramp_epochs),
        )
    )


def cv2_scale(epoch: int, cfg: SimpleNamespace) -> float:
    return linear_ramp(
        epoch=epoch,
        start_epoch=cfg.cv2_start_epoch,
        ramp_epochs=cfg.cv2_ramp_epochs,
    )


def compute_route_regularizer(
    route_r: torch.Tensor,
    target_max: float,
) -> torch.Tensor:
    return torch.relu(route_r.mean() - target_max)


def compute_aux_grammar_loss(
    outputs: Dict,
    batch: Dict,
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
    outputs: Dict,
    batch: Dict,
    cfg: SimpleNamespace,
    importance_pos_weight: float,
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
    scale = cv2_scale(epoch, cfg)

    total = (
        cfg.interaction_loss_weight * interaction_loss
        + patch_loss_dict["importance_loss"]
        + patch_loss_dict["infonce_loss"]
        + cfg.grammar_loss_weight * grammar_aux
        + scale * (cfg.pos_cv2_weight * pos_cv2_loss)
    )

    logs = {
        "total_loss": float(total.item()),
        "interaction_loss": float(interaction_loss.item()),
        "importance_loss": float(patch_loss_dict["importance_loss"].item()),
        "infonce_loss": float(patch_loss_dict["infonce_loss"].item()),
        "grammar_aux_loss": float(grammar_aux.item()),
        "pos_cv2_loss": pos_cv2_logs["pos_cv2_loss"],
        "pos_cv2_selected_frac": pos_cv2_logs["pos_cv2_selected_frac"],
        "pos_usage_max": pos_cv2_logs["pos_usage_max"],
        "pos_usage_entropy_norm": pos_cv2_logs["pos_usage_entropy_norm"],
        "cv2_scale": scale,
        "head_b_aux_loss": 0.0,
        "route_reg_loss": 0.0,
        "route_mean": 0.0,
    }

    logs.update(sigma_summary_fn(outputs))

    return total, logs


def compute_stage2_loss(
    outputs: Dict,
    batch: Dict,
    cfg: SimpleNamespace,
    importance_pos_weight: float,
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
    scale = cv2_scale(epoch, cfg)

    total = (
        cfg.interaction_loss_weight * interaction_loss
        + patch_loss_dict["importance_loss"]
        + patch_loss_dict["infonce_loss"]
        + cfg.grammar_loss_weight * grammar_aux
        + cfg.head_b_aux_loss_weight * head_b_aux
        + cfg.route_reg_weight * route_reg
        + scale * (cfg.pos_cv2_weight * pos_cv2_loss)
    )

    logs = {
        "total_loss": float(total.item()),
        "interaction_loss": float(interaction_loss.item()),
        "importance_loss": float(patch_loss_dict["importance_loss"].item()),
        "infonce_loss": float(patch_loss_dict["infonce_loss"].item()),
        "grammar_aux_loss": float(grammar_aux.item()),
        "pos_cv2_loss": pos_cv2_logs["pos_cv2_loss"],
        "pos_cv2_selected_frac": pos_cv2_logs["pos_cv2_selected_frac"],
        "pos_usage_max": pos_cv2_logs["pos_usage_max"],
        "pos_usage_entropy_norm": pos_cv2_logs["pos_usage_entropy_norm"],
        "cv2_scale": scale,
        "head_b_aux_loss": float(head_b_aux.item()),
        "route_reg_loss": float(route_reg.item()),
        "route_mean": float(outputs["route_r"].mean().item()),
    }

    logs.update(sigma_summary_fn(outputs))

    return total, logs