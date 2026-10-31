# src/pigram/losses/baseline.py
# -*- coding: utf-8 -*-

from __future__ import annotations

from types import SimpleNamespace
from typing import Dict, Tuple

import torch


def check_pair_shape(
    preds: torch.Tensor,
    targets: torch.Tensor,
    z_patch: torch.Tensor,
    z_motif: torch.Tensor,
) -> None:
    if preds.shape != targets.shape:
        raise RuntimeError(
            f"Shape mismatch: preds={tuple(preds.shape)} "
            f"vs targets={tuple(targets.shape)}; "
            f"z_patch={tuple(z_patch.shape)}, "
            f"z_motif={tuple(z_motif.shape)}"
        )


def compute_single_stage_loss(
    outputs: Dict,
    batch: Dict,
    cfg: SimpleNamespace,
    importance_pos_weight: float,
    total_loss_fn,
    patch_motif_huber_loss_fn,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    preds = outputs["patch_motif_logits"]
    targets = batch["patch_motif_structure_energy_v2"]

    check_pair_shape(
        preds=preds,
        targets=targets,
        z_patch=outputs["z_patch"],
        z_motif=outputs["z_motif"],
    )

    interaction_loss = patch_motif_huber_loss_fn(
        logits=preds,
        targets=targets,
        sigmoid_scale=cfg.sigmoid_scale,
        delta=cfg.huber_delta,
    )

    patch_loss_dict = total_loss_fn(
        patch_motif_logits=preds,
        patch_motif_targets=targets,
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

    total = (
        cfg.interaction_loss_weight * interaction_loss
        + patch_loss_dict["importance_loss"]
        + patch_loss_dict["infonce_loss"]
    )

    logs = {
        "total_loss": float(total.item()),
        "interaction_loss": float(interaction_loss.item()),
        "importance_loss": float(patch_loss_dict["importance_loss"].item()),
        "infonce_loss": float(patch_loss_dict["infonce_loss"].item()),
    }

    return total, logs


def compute_proto_stage1_loss(
    outputs: Dict,
    batch: Dict,
    cfg: SimpleNamespace,
    importance_pos_weight: float,
    total_loss_fn,
    patch_motif_huber_loss_fn,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    total, logs = compute_single_stage_loss(
        outputs=outputs,
        batch=batch,
        cfg=cfg,
        importance_pos_weight=importance_pos_weight,
        total_loss_fn=total_loss_fn,
        patch_motif_huber_loss_fn=patch_motif_huber_loss_fn,
    )

    logs["head_b_aux_loss"] = 0.0
    logs["route_reg_loss"] = 0.0
    logs["route_mean"] = 0.0

    return total, logs


def compute_proto_stage2_loss(
    outputs: Dict,
    batch: Dict,
    cfg: SimpleNamespace,
    importance_pos_weight: float,
    total_loss_fn,
    patch_motif_huber_loss_fn,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    preds = outputs["patch_motif_logits"]
    targets = batch["patch_motif_structure_energy_v2"]

    check_pair_shape(
        preds=preds,
        targets=targets,
        z_patch=outputs["z_patch"],
        z_motif=outputs["z_motif"],
    )

    interaction_loss = patch_motif_huber_loss_fn(
        logits=preds,
        targets=targets,
        sigmoid_scale=cfg.sigmoid_scale,
        delta=cfg.huber_delta,
    )

    patch_loss_dict = total_loss_fn(
        patch_motif_logits=preds,
        patch_motif_targets=targets,
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

    head_b_preds = outputs["head_b_logits"]

    check_pair_shape(
        preds=head_b_preds,
        targets=targets,
        z_patch=outputs["z_patch"],
        z_motif=outputs["z_motif"],
    )

    head_b_aux = patch_motif_huber_loss_fn(
        logits=head_b_preds,
        targets=targets,
        sigmoid_scale=cfg.sigmoid_scale,
        delta=cfg.huber_delta,
    )

    route_reg = torch.relu(outputs["route_r"].mean() - cfg.route_target_max)

    total = (
        cfg.interaction_loss_weight * interaction_loss
        + patch_loss_dict["importance_loss"]
        + patch_loss_dict["infonce_loss"]
        + cfg.head_b_aux_loss_weight * head_b_aux
        + cfg.route_reg_weight * route_reg
    )

    logs = {
        "total_loss": float(total.item()),
        "interaction_loss": float(interaction_loss.item()),
        "importance_loss": float(patch_loss_dict["importance_loss"].item()),
        "infonce_loss": float(patch_loss_dict["infonce_loss"].item()),
        "head_b_aux_loss": float(head_b_aux.item()),
        "route_reg_loss": float(route_reg.item()),
        "route_mean": float(outputs["route_r"].mean().item()),
    }

    return total, logs