# src/pigram/losses/interaction.py
# -*- coding: utf-8 -*-

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn.functional as F


def _zero_like_reference(reference: torch.Tensor) -> torch.Tensor:
    return reference.sum() * 0.0


def _as_float_tensor(x: torch.Tensor) -> torch.Tensor:
    return x.float()


def compute_patch_motif_huber_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    sigmoid_scale: float = 2.0,
    delta: float = 0.10,
    reduction: str = "mean",
) -> torch.Tensor:
    """
    Structure-aware patch--motif regression loss.

    The model produces logits. The target is usually a bounded continuous
    structure-aware value in [0, 1]. We compare:

        sigmoid(logits / sigmoid_scale)  vs.  targets

    using SmoothL1 / Huber loss.

    Args:
        logits:
            Predicted patch--motif logits, shape [N_patch, N_motif].
        targets:
            Continuous targets, shape [N_patch, N_motif].
        sigmoid_scale:
            Temperature for converting logits to probabilities.
        delta:
            SmoothL1 beta / Huber transition point.
        reduction:
            "mean", "sum", or "none".

    Returns:
        Loss tensor.
    """
    logits = _as_float_tensor(logits)
    targets = _as_float_tensor(targets)

    if logits.shape != targets.shape:
        raise RuntimeError(
            f"compute_patch_motif_huber_loss shape mismatch: "
            f"logits={tuple(logits.shape)} targets={tuple(targets.shape)}"
        )

    prob = torch.sigmoid(logits / float(sigmoid_scale))

    return F.smooth_l1_loss(
        prob,
        targets,
        beta=float(delta),
        reduction=reduction,
    )


def compute_weighted_interaction_bce_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    sigmoid_scale: float = 2.0,
    target_weight_scale: float = 3.0,
) -> torch.Tensor:
    """
    Optional weighted BCE-style interaction loss.

    This is kept for compatibility with older training code. The current
    PIGRAM v2 target usually uses the Huber loss above, but older experiments
    sometimes used a target-weighted BCE objective.

    Args:
        logits:
            Patch--motif logits.
        targets:
            Continuous or binary targets in [0, 1].
        sigmoid_scale:
            Logit temperature.
        target_weight_scale:
            Positive-target emphasis factor.

    Returns:
        Scalar loss.
    """
    logits = _as_float_tensor(logits)
    targets = _as_float_tensor(targets)

    if logits.shape != targets.shape:
        raise RuntimeError(
            f"compute_weighted_interaction_bce_loss shape mismatch: "
            f"logits={tuple(logits.shape)} targets={tuple(targets.shape)}"
        )

    scaled_logits = logits / float(sigmoid_scale)
    weights = 1.0 + float(target_weight_scale) * targets.detach()

    loss_map = F.binary_cross_entropy_with_logits(
        scaled_logits,
        targets,
        reduction="none",
    )

    return (weights * loss_map).mean()


def compute_patch_importance_loss(
    patch_importance_logits: Optional[torch.Tensor],
    patch_importance_labels: Optional[torch.Tensor],
    importance_pos_weight: Optional[float] = None,
    reference: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Binary patch-importance loss.

    Args:
        patch_importance_logits:
            Logits with shape [N_patch].
        patch_importance_labels:
            Binary labels with shape [N_patch].
        importance_pos_weight:
            Optional positive-class weight.
        reference:
            Reference tensor used to create a differentiable zero if inputs are absent.

    Returns:
        Scalar BCE loss.
    """
    if patch_importance_logits is None or patch_importance_labels is None:
        if reference is None:
            return torch.zeros(())
        return _zero_like_reference(reference)

    logits = _as_float_tensor(patch_importance_logits).reshape(-1)
    labels = _as_float_tensor(patch_importance_labels).reshape(-1)

    if logits.shape != labels.shape:
        raise RuntimeError(
            f"Patch importance shape mismatch: "
            f"logits={tuple(logits.shape)} labels={tuple(labels.shape)}"
        )

    pos_weight_tensor = None
    if importance_pos_weight is not None:
        pos_weight_tensor = torch.tensor(
            float(importance_pos_weight),
            dtype=logits.dtype,
            device=logits.device,
        )

    return F.binary_cross_entropy_with_logits(
        logits,
        labels,
        pos_weight=pos_weight_tensor,
        reduction="mean",
    )


def compute_pairwise_infonce_loss(
    patch_emb: Optional[torch.Tensor],
    motif_emb: Optional[torch.Tensor],
    patch_motif_targets: torch.Tensor,
    pos_threshold: float = 0.5,
    temperature: float = 0.2,
) -> torch.Tensor:
    """
    Patch--motif InfoNCE-style representation loss.

    Positive pairs are selected by:

        patch_motif_targets >= pos_threshold

    For each positive pair (i, j), the patch embedding z_p[i] should be closer
    to motif embedding z_m[j] than to other motifs. A symmetric motif-to-patch
    term is also used.

    Args:
        patch_emb:
            Patch embeddings [N_patch, D].
        motif_emb:
            Motif embeddings [N_motif, D].
        patch_motif_targets:
            Continuous/binary target [N_patch, N_motif].
        pos_threshold:
            Positive-pair threshold.
        temperature:
            Softmax temperature.

    Returns:
        Scalar InfoNCE loss. Returns differentiable zero if no positive pairs exist.
    """
    targets = _as_float_tensor(patch_motif_targets)

    if patch_emb is None or motif_emb is None:
        return _zero_like_reference(targets)

    patch_emb = _as_float_tensor(patch_emb)
    motif_emb = _as_float_tensor(motif_emb)

    if patch_emb.ndim != 2 or motif_emb.ndim != 2:
        raise RuntimeError(
            f"patch_emb and motif_emb must be 2D, got "
            f"patch_emb={tuple(patch_emb.shape)} motif_emb={tuple(motif_emb.shape)}"
        )

    if targets.shape != (patch_emb.shape[0], motif_emb.shape[0]):
        raise RuntimeError(
            f"InfoNCE target shape mismatch: targets={tuple(targets.shape)} "
            f"expected={(patch_emb.shape[0], motif_emb.shape[0])}"
        )

    positive_mask = targets >= float(pos_threshold)

    if positive_mask.sum().item() == 0:
        return _zero_like_reference(patch_emb) + _zero_like_reference(motif_emb)

    patch_norm = F.normalize(patch_emb, dim=-1)
    motif_norm = F.normalize(motif_emb, dim=-1)

    logits = patch_norm @ motif_norm.transpose(0, 1)
    logits = logits / max(float(temperature), 1e-8)

    pos_indices = positive_mask.nonzero(as_tuple=False)
    patch_indices = pos_indices[:, 0]
    motif_indices = pos_indices[:, 1]

    # Patch -> motif: classify the positive motif for each positive pair.
    loss_patch_to_motif = F.cross_entropy(
        logits[patch_indices],
        motif_indices,
        reduction="mean",
    )

    # Motif -> patch: classify the positive patch for each positive pair.
    loss_motif_to_patch = F.cross_entropy(
        logits.transpose(0, 1)[motif_indices],
        patch_indices,
        reduction="mean",
    )

    return 0.5 * (loss_patch_to_motif + loss_motif_to_patch)


def compute_total_loss(
    patch_motif_logits: torch.Tensor,
    patch_motif_targets: torch.Tensor,
    patch_importance_logits: Optional[torch.Tensor] = None,
    patch_importance_labels: Optional[torch.Tensor] = None,
    interaction_loss_weight: float = 1.0,
    importance_loss_weight: float = 0.5,
    infonce_loss_weight: float = 0.5,
    type_compat_loss_weight: float = 0.0,
    patch_proto_div_loss_weight: float = 0.0,
    motif_proto_div_loss_weight: float = 0.0,
    importance_pos_weight: Optional[float] = None,
    sigmoid_scale: float = 2.0,
    target_weight_scale: float = 3.0,
    patch_emb: Optional[torch.Tensor] = None,
    motif_emb: Optional[torch.Tensor] = None,
    infonce_pos_threshold: float = 0.5,
    infonce_temperature: float = 0.2,
    patch_assign: Optional[torch.Tensor] = None,
    motif_assign: Optional[torch.Tensor] = None,
    compatibility_logits: Optional[torch.Tensor] = None,
    patch_prototypes: Optional[torch.Tensor] = None,
    motif_prototypes: Optional[torch.Tensor] = None,
) -> Dict[str, torch.Tensor]:
    """
    Unified loss helper used by baselines, coarse teacher, and grammar foundation.

    The return dictionary intentionally keeps names used by the older training
    scripts:
        interaction_loss
        importance_loss
        infonce_loss
        type_compat_loss
        patch_proto_div_loss
        motif_proto_div_loss
        total_loss

    Several prototype/type-compatibility arguments are accepted for API
    compatibility. If corresponding weights are zero or tensors are missing,
    their losses are differentiable zeros.

    Args:
        patch_motif_logits:
            Predicted pair logits [N_patch, N_motif].
        patch_motif_targets:
            Continuous targets [N_patch, N_motif].
        patch_importance_logits:
            Patch importance logits [N_patch].
        patch_importance_labels:
            Patch importance binary labels [N_patch].
        interaction_loss_weight:
            Weight for target-weighted interaction BCE compatibility loss.
            In current scripts this is often passed as 0.0 because they use
            compute_patch_motif_huber_loss separately.
        importance_loss_weight:
            Weight for patch importance BCE.
        infonce_loss_weight:
            Weight for representation InfoNCE.
        type_compat_loss_weight:
            Compatibility placeholder.
        patch_proto_div_loss_weight:
            Prototype regularization placeholder.
        motif_proto_div_loss_weight:
            Prototype regularization placeholder.

    Returns:
        Dict[str, torch.Tensor].
    """
    patch_motif_logits = _as_float_tensor(patch_motif_logits)
    patch_motif_targets = _as_float_tensor(patch_motif_targets)

    interaction_loss = compute_weighted_interaction_bce_loss(
        logits=patch_motif_logits,
        targets=patch_motif_targets,
        sigmoid_scale=sigmoid_scale,
        target_weight_scale=target_weight_scale,
    )

    importance_loss = compute_patch_importance_loss(
        patch_importance_logits=patch_importance_logits,
        patch_importance_labels=patch_importance_labels,
        importance_pos_weight=importance_pos_weight,
        reference=patch_motif_logits,
    )

    infonce_loss = compute_pairwise_infonce_loss(
        patch_emb=patch_emb,
        motif_emb=motif_emb,
        patch_motif_targets=patch_motif_targets,
        pos_threshold=infonce_pos_threshold,
        temperature=infonce_temperature,
    )

    # Compatibility placeholders for older scripts/configs. They are kept as
    # differentiable zeros to avoid changing callers.
    type_compat_loss = _zero_like_reference(patch_motif_logits)
    patch_proto_div_loss = _zero_like_reference(patch_motif_logits)
    motif_proto_div_loss = _zero_like_reference(patch_motif_logits)

    total_loss = (
        float(interaction_loss_weight) * interaction_loss
        + float(importance_loss_weight) * importance_loss
        + float(infonce_loss_weight) * infonce_loss
        + float(type_compat_loss_weight) * type_compat_loss
        + float(patch_proto_div_loss_weight) * patch_proto_div_loss
        + float(motif_proto_div_loss_weight) * motif_proto_div_loss
    )

    return {
        "total_loss": total_loss,
        "interaction_loss": interaction_loss,
        "importance_loss": float(importance_loss_weight) * importance_loss,
        "importance_loss_raw": importance_loss,
        "infonce_loss": float(infonce_loss_weight) * infonce_loss,
        "infonce_loss_raw": infonce_loss,
        "type_compat_loss": float(type_compat_loss_weight) * type_compat_loss,
        "patch_proto_div_loss": float(patch_proto_div_loss_weight) * patch_proto_div_loss,
        "motif_proto_div_loss": float(motif_proto_div_loss_weight) * motif_proto_div_loss,
    }