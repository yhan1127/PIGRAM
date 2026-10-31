# src/pigram/data/label.py
# -*- coding: utf-8 -*-

from __future__ import annotations

import math
from typing import Any, Dict, List, Tuple

import numpy as np


# =============================================================================
# Distance-based patch--motif supervision
# =============================================================================

def _aggregate_vector(
    values: np.ndarray,
    method: str = "topk_mean",
    top_k: int = 3,
) -> float:
    """
    Aggregate a 1D vector into a scalar.

    For distance-based labels, smaller distances indicate stronger contact.

    Args:
        values:
            1D numeric array.
        method:
            One of {"min", "mean", "topk_mean"}.
        top_k:
            Number of smallest values to average when method == "topk_mean".

    Returns:
        Aggregated scalar.
    """
    values = np.asarray(values, dtype=np.float32).reshape(-1)

    if values.size == 0:
        return 0.0

    if method == "min":
        return float(values.min())

    if method == "mean":
        return float(values.mean())

    if method == "topk_mean":
        k_eff = min(int(top_k), values.size)
        smallest = np.sort(values)[:k_eff]
        return float(smallest.mean())

    raise ValueError(f"Unsupported aggregation method: {method}")


def build_residue_to_motif_distance(
    contact_map: np.ndarray,
    motifs: List[Dict[str, Any]],
    residue_motif_agg: str = "topk_mean",
    residue_top_k: int = 3,
) -> np.ndarray:
    """
    Build residue-to-motif distance matrix.

    Args:
        contact_map:
            Protein residue--nucleotide distance/contact map with shape
            [L_protein, L_partner].
        motifs:
            List of motif dicts. Each motif must contain "indices".
        residue_motif_agg:
            Aggregation method over nucleotides in a motif.
        residue_top_k:
            Used when residue_motif_agg == "topk_mean".

    Returns:
        residue_to_motif_dist:
            Array with shape [L_protein, N_motifs].
    """
    contact_map = np.asarray(contact_map, dtype=np.float32)

    if contact_map.ndim != 2:
        raise ValueError(f"contact_map must be 2D, got shape={contact_map.shape}")

    num_residues = int(contact_map.shape[0])
    num_motifs = int(len(motifs))

    residue_to_motif_dist = np.zeros(
        (num_residues, num_motifs),
        dtype=np.float32,
    )

    for residue_idx in range(num_residues):
        for motif_idx, motif in enumerate(motifs):
            nt_indices = motif["indices"]
            values = contact_map[residue_idx, nt_indices]

            residue_to_motif_dist[residue_idx, motif_idx] = _aggregate_vector(
                values=values,
                method=residue_motif_agg,
                top_k=residue_top_k,
            )

    return residue_to_motif_dist


def build_patch_to_motif_distance(
    residue_to_motif_dist: np.ndarray,
    patch_indices: np.ndarray,
    patch_motif_agg: str = "topk_mean",
    patch_top_k: int = 3,
) -> np.ndarray:
    """
    Aggregate residue-to-motif distances into patch-to-motif distances.

    Args:
        residue_to_motif_dist:
            Array with shape [L_protein, N_motifs].
        patch_indices:
            Array with shape [N_patches, K].
        patch_motif_agg:
            Aggregation method over residues in a patch.
        patch_top_k:
            Used when patch_motif_agg == "topk_mean".

    Returns:
        patch_to_motif_dist:
            Array with shape [N_patches, N_motifs].
    """
    residue_to_motif_dist = np.asarray(residue_to_motif_dist, dtype=np.float32)
    patch_indices = np.asarray(patch_indices, dtype=np.int64)

    if residue_to_motif_dist.ndim != 2:
        raise ValueError(
            f"residue_to_motif_dist must be 2D, got shape={residue_to_motif_dist.shape}"
        )

    if patch_indices.ndim != 2:
        raise ValueError(f"patch_indices must be 2D, got shape={patch_indices.shape}")

    num_patches = int(patch_indices.shape[0])
    num_motifs = int(residue_to_motif_dist.shape[1])

    patch_to_motif_dist = np.zeros(
        (num_patches, num_motifs),
        dtype=np.float32,
    )

    for patch_idx in range(num_patches):
        residue_indices = patch_indices[patch_idx]

        for motif_idx in range(num_motifs):
            values = residue_to_motif_dist[residue_indices, motif_idx]
            patch_to_motif_dist[patch_idx, motif_idx] = _aggregate_vector(
                values=values,
                method=patch_motif_agg,
                top_k=patch_top_k,
            )

    return patch_to_motif_dist


def build_patch_motif_binary_label(
    patch_to_motif_dist: np.ndarray,
    threshold: float = 6.0,
) -> np.ndarray:
    """
    Build binary patch--motif contact labels from a distance threshold.

    Smaller distance means stronger contact.
    """
    patch_to_motif_dist = np.asarray(patch_to_motif_dist, dtype=np.float32)

    return (patch_to_motif_dist < float(threshold)).astype(np.float32)


def build_patch_motif_supervision(
    contact_map: np.ndarray,
    motifs: List[Dict[str, Any]],
    patch_indices: np.ndarray,
    residue_motif_agg: str = "topk_mean",
    patch_motif_agg: str = "topk_mean",
    threshold: float = 6.0,
    residue_top_k: int = 3,
    patch_top_k: int = 3,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Original distance-based patch--motif supervision pipeline.

    Returns:
        residue_to_motif_dist:
            [L_protein, N_motifs]
        patch_to_motif_dist:
            [N_patches, N_motifs]
        patch_motif_label:
            [N_patches, N_motifs]
    """
    residue_to_motif_dist = build_residue_to_motif_distance(
        contact_map=contact_map,
        motifs=motifs,
        residue_motif_agg=residue_motif_agg,
        residue_top_k=residue_top_k,
    )

    patch_to_motif_dist = build_patch_to_motif_distance(
        residue_to_motif_dist=residue_to_motif_dist,
        patch_indices=patch_indices,
        patch_motif_agg=patch_motif_agg,
        patch_top_k=patch_top_k,
    )

    patch_motif_label = build_patch_motif_binary_label(
        patch_to_motif_dist=patch_to_motif_dist,
        threshold=threshold,
    )

    return residue_to_motif_dist, patch_to_motif_dist, patch_motif_label


def derive_patch_importance_from_labels(
    patch_motif_label: np.ndarray,
) -> np.ndarray:
    """
    A patch is important if it is positive for at least one motif.
    """
    patch_motif_label = np.asarray(patch_motif_label, dtype=np.float32)

    if patch_motif_label.ndim != 2:
        raise ValueError(
            f"patch_motif_label must be 2D, got shape={patch_motif_label.shape}"
        )

    return (patch_motif_label.max(axis=1) > 0).astype(np.float32)


def build_patch_motif_energy_label(
    patch_to_motif_dist: np.ndarray,
    tau: float = 4.0,
) -> np.ndarray:
    """
    Smooth distance-decay energy label.

    Higher values indicate closer patch--motif contact.
    """
    patch_to_motif_dist = np.asarray(patch_to_motif_dist, dtype=np.float32)
    tau = max(float(tau), 1e-8)

    return np.exp(-patch_to_motif_dist / tau).astype(np.float32)


# =============================================================================
# Compatibility structure-aware label
# =============================================================================

def _motif_position_weights(
    length: int,
    mode: str = "center",
) -> np.ndarray:
    """
    Positional weights within a motif.

    mode:
        uniform:
            all positions equal.
        center:
            center positions get higher weight.
    """
    if length <= 0:
        return np.zeros((0,), dtype=np.float32)

    if mode == "uniform":
        weights = np.ones(length, dtype=np.float32)
        return weights / (weights.sum() + 1e-8)

    if mode == "center":
        idx = np.arange(length, dtype=np.float32)
        center = (length - 1) / 2.0
        sigma = max(length / 3.0, 1.0)
        weights = np.exp(-((idx - center) ** 2) / (2.0 * sigma * sigma)).astype(np.float32)
        return weights / (weights.sum() + 1e-8)

    raise ValueError(f"Unsupported position weight mode: {mode}")


def _motif_global_weight(
    motif: Dict[str, Any],
    motif_type_weights: Dict[str, float],
    pairing_weight_scale: float = 0.5,
    gc_weight_scale: float = 0.3,
) -> float:
    """
    Global motif multiplier based on motif type, pairing ratio, and GC content.
    """
    motif_type = str(motif.get("type", "unpaired"))
    pairing_ratio = float(motif.get("pairing_ratio", 0.0))
    gc_content = float(motif.get("gc_content", 0.0))

    type_weight = float(motif_type_weights.get(motif_type, 1.0))
    weight = type_weight * (
        1.0
        + pairing_weight_scale * pairing_ratio
        + gc_weight_scale * gc_content
    )

    return float(weight)


def _residue_to_single_motif_structure_aware_score(
    residue_to_nt_dist: np.ndarray,
    motif: Dict[str, Any],
    tau: float = 4.0,
    position_weight_mode: str = "center",
    motif_type_weights: Dict[str, float] | None = None,
    pairing_weight_scale: float = 0.5,
    gc_weight_scale: float = 0.3,
) -> float:
    """
    Compatibility structure-aware residue-to-motif score.

    This is kept because processed samples still store
    patch_motif_structure_energy in addition to the final v2 target.
    """
    if motif_type_weights is None:
        motif_type_weights = {
            "stem": 1.2,
            "loop": 1.0,
            "unpaired": 0.9,
        }

    residue_to_nt_dist = np.asarray(residue_to_nt_dist, dtype=np.float32).reshape(-1)

    if residue_to_nt_dist.size == 0:
        return 0.0

    pos_weights = _motif_position_weights(
        length=residue_to_nt_dist.size,
        mode=position_weight_mode,
    )
    dist_score = np.exp(
        -residue_to_nt_dist / max(float(tau), 1e-8)
    ).astype(np.float32)

    global_weight = _motif_global_weight(
        motif=motif,
        motif_type_weights=motif_type_weights,
        pairing_weight_scale=pairing_weight_scale,
        gc_weight_scale=gc_weight_scale,
    )

    return float(global_weight * float((pos_weights * dist_score).sum()))


def build_structure_aware_residue_to_motif_score(
    contact_map: np.ndarray,
    motifs: List[Dict[str, Any]],
    tau: float = 4.0,
    position_weight_mode: str = "center",
    motif_type_weights: Dict[str, float] | None = None,
    pairing_weight_scale: float = 0.5,
    gc_weight_scale: float = 0.3,
) -> np.ndarray:
    """
    Build compatibility structure-aware residue-to-motif score.

    Returns:
        residue_to_motif_score:
            [L_protein, N_motifs]
    """
    contact_map = np.asarray(contact_map, dtype=np.float32)

    if contact_map.ndim != 2:
        raise ValueError(f"contact_map must be 2D, got shape={contact_map.shape}")

    num_residues = int(contact_map.shape[0])
    num_motifs = int(len(motifs))

    residue_to_motif_score = np.zeros(
        (num_residues, num_motifs),
        dtype=np.float32,
    )

    for residue_idx in range(num_residues):
        for motif_idx, motif in enumerate(motifs):
            nt_indices = motif["indices"]
            residue_to_nt_dist = contact_map[residue_idx, nt_indices]

            residue_to_motif_score[residue_idx, motif_idx] = _residue_to_single_motif_structure_aware_score(
                residue_to_nt_dist=residue_to_nt_dist,
                motif=motif,
                tau=tau,
                position_weight_mode=position_weight_mode,
                motif_type_weights=motif_type_weights,
                pairing_weight_scale=pairing_weight_scale,
                gc_weight_scale=gc_weight_scale,
            )

    return residue_to_motif_score


def build_structure_aware_patch_to_motif_score(
    residue_to_motif_score: np.ndarray,
    patch_indices: np.ndarray,
    patch_agg: str = "topk_mean",
    patch_top_k: int = 3,
) -> np.ndarray:
    """
    Aggregate residue-to-motif scores into patch-to-motif scores.

    This score is high-is-better, unlike distance.
    """
    residue_to_motif_score = np.asarray(residue_to_motif_score, dtype=np.float32)
    patch_indices = np.asarray(patch_indices, dtype=np.int64)

    if residue_to_motif_score.ndim != 2:
        raise ValueError(
            f"residue_to_motif_score must be 2D, got shape={residue_to_motif_score.shape}"
        )

    if patch_indices.ndim != 2:
        raise ValueError(f"patch_indices must be 2D, got shape={patch_indices.shape}")

    num_patches = int(patch_indices.shape[0])
    num_motifs = int(residue_to_motif_score.shape[1])

    patch_to_motif_score = np.zeros(
        (num_patches, num_motifs),
        dtype=np.float32,
    )

    for patch_idx in range(num_patches):
        residue_indices = patch_indices[patch_idx]

        for motif_idx in range(num_motifs):
            values = residue_to_motif_score[residue_indices, motif_idx]

            if values.size == 0:
                patch_to_motif_score[patch_idx, motif_idx] = 0.0
            elif patch_agg == "mean":
                patch_to_motif_score[patch_idx, motif_idx] = float(values.mean())
            elif patch_agg == "max":
                patch_to_motif_score[patch_idx, motif_idx] = float(values.max())
            elif patch_agg == "topk_mean":
                k_eff = min(int(patch_top_k), values.size)
                largest = np.sort(values)[-k_eff:]
                patch_to_motif_score[patch_idx, motif_idx] = float(largest.mean())
            else:
                raise ValueError(f"Unsupported patch_agg: {patch_agg}")

    return patch_to_motif_score


def build_structure_aware_patch_motif_label(
    contact_map: np.ndarray,
    motifs: List[Dict[str, Any]],
    patch_indices: np.ndarray,
    tau: float = 4.0,
    position_weight_mode: str = "center",
    motif_type_weights: Dict[str, float] | None = None,
    pairing_weight_scale: float = 0.5,
    gc_weight_scale: float = 0.3,
    patch_agg: str = "topk_mean",
    patch_top_k: int = 3,
) -> np.ndarray:
    """
    Compatibility structure-aware patch-motif label.

    Kept in the unified label module because process_single_sample creates
    patch_motif_structure_energy for compatibility and analysis.
    """
    residue_to_motif_score = build_structure_aware_residue_to_motif_score(
        contact_map=contact_map,
        motifs=motifs,
        tau=tau,
        position_weight_mode=position_weight_mode,
        motif_type_weights=motif_type_weights,
        pairing_weight_scale=pairing_weight_scale,
        gc_weight_scale=gc_weight_scale,
    )

    patch_to_motif_score = build_structure_aware_patch_to_motif_score(
        residue_to_motif_score=residue_to_motif_score,
        patch_indices=patch_indices,
        patch_agg=patch_agg,
        patch_top_k=patch_top_k,
    )

    return patch_to_motif_score.astype(np.float32)


# =============================================================================
# Final structure-aware label v2
# =============================================================================

def softmax_np(
    x: np.ndarray,
    axis: int = -1,
) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    x_max = np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x - x_max)
    denom = np.sum(exp_x, axis=axis, keepdims=True) + 1e-12

    return exp_x / denom


def motif_type_group(motif_type: str) -> str:
    motif_type = str(motif_type).lower()

    if motif_type in {"stem_core", "stem_edge"}:
        return "stem"

    if motif_type in {"hairpin_loop", "internal_loop", "bulge", "multi_loop"}:
        return "loop"

    return "unpaired"


def motif_position_weight(
    normalized_center_position: float,
    mode: str = "center_mild",
) -> float:
    x = float(normalized_center_position)

    if mode == "none":
        return 1.0

    if mode == "center":
        return 1.0 - abs(x - 0.5)

    if mode == "terminal_favor":
        return abs(x - 0.5) * 2.0

    if mode == "center_mild":
        return 0.75 + 0.5 * (1.0 - abs(x - 0.5) * 2.0)

    return 1.0


def compute_motif_structure_multiplier_v2(
    motif: Dict[str, Any],
    motif_type_weights: Dict[str, float],
    pairing_weight_scale: float,
    gc_weight_scale: float,
    position_weight_mode: str,
    position_weight_scale: float,
) -> float:
    group = motif_type_group(motif["type"])
    type_weight = float(motif_type_weights.get(group, 1.0))

    pairing_ratio = float(motif.get("pairing_ratio", 0.0))
    gc_content = float(motif.get("gc_content", 0.0))
    center_pos = float(motif.get("normalized_center_position", 0.5))

    pairing_factor = 1.0 + pairing_weight_scale * pairing_ratio
    gc_factor = 1.0 + gc_weight_scale * gc_content

    pos_base = motif_position_weight(
        normalized_center_position=center_pos,
        mode=position_weight_mode,
    )
    position_factor = 1.0 + position_weight_scale * pos_base

    return float(type_weight * pairing_factor * gc_factor * position_factor)


def residue_to_motif_soft_score_v2(
    residue_nt_distances: np.ndarray,
    beta_nt: float,
    contact_distance_tau: float,
) -> float:
    d = np.asarray(residue_nt_distances, dtype=np.float64).reshape(-1)

    if d.size == 0:
        return 0.0

    weights = softmax_np(-float(beta_nt) * d, axis=0)
    effective_distance = float(np.sum(weights * d))
    score = math.exp(-effective_distance / max(float(contact_distance_tau), 1e-8))

    return float(score)


def patch_to_motif_soft_score_v2(
    residue_scores: np.ndarray,
    beta_patch: float,
) -> float:
    values = np.asarray(residue_scores, dtype=np.float64).reshape(-1)

    if values.size == 0:
        return 0.0

    weights = softmax_np(float(beta_patch) * values, axis=0)

    return float(np.sum(weights * values))


def build_structure_aware_patch_motif_label_v2(
    contact_map: np.ndarray,
    motifs: List[Dict[str, Any]],
    patch_indices: np.ndarray,
    beta_nt: float = 3.0,
    beta_patch: float = 6.0,
    contact_distance_tau: float = 4.0,
    motif_type_weights: Dict[str, float] | None = None,
    pairing_weight_scale: float = 0.5,
    gc_weight_scale: float = 0.25,
    position_weight_mode: str = "center_mild",
    position_weight_scale: float = 0.25,
    label_tau: float = 0.20,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build final structure-aware patch-motif target v2.

    Returns:
        residue_to_motif_score_v2_raw:
            [L_residue, N_motif]
        patch_motif_structure_energy_v2:
            [N_patch, N_motif]
    """
    if motif_type_weights is None:
        motif_type_weights = {
            "stem": 1.5,
            "loop": 1.1,
            "unpaired": 0.8,
        }

    contact_map = np.asarray(contact_map, dtype=np.float64)
    patch_indices = np.asarray(patch_indices, dtype=np.int64)

    if contact_map.ndim != 2:
        raise ValueError(f"contact_map must be 2D, got shape={contact_map.shape}")

    if patch_indices.ndim != 2:
        raise ValueError(f"patch_indices must be 2D, got shape={patch_indices.shape}")

    num_residues = int(contact_map.shape[0])
    num_patches = int(patch_indices.shape[0])
    num_motifs = int(len(motifs))

    residue_to_motif_score_v2_raw = np.zeros(
        (num_residues, num_motifs),
        dtype=np.float64,
    )
    patch_to_motif_raw_score_v2 = np.zeros(
        (num_patches, num_motifs),
        dtype=np.float64,
    )

    for motif_idx, motif in enumerate(motifs):
        motif_indices = motif["indices"]
        motif_block = contact_map[:, motif_indices]

        for residue_idx in range(num_residues):
            residue_to_motif_score_v2_raw[residue_idx, motif_idx] = residue_to_motif_soft_score_v2(
                residue_nt_distances=motif_block[residue_idx],
                beta_nt=beta_nt,
                contact_distance_tau=contact_distance_tau,
            )

    for patch_idx in range(num_patches):
        residue_indices = patch_indices[patch_idx]

        for motif_idx, motif in enumerate(motifs):
            residue_scores = residue_to_motif_score_v2_raw[residue_indices, motif_idx]

            patch_score = patch_to_motif_soft_score_v2(
                residue_scores=residue_scores,
                beta_patch=beta_patch,
            )

            structure_multiplier = compute_motif_structure_multiplier_v2(
                motif=motif,
                motif_type_weights=motif_type_weights,
                pairing_weight_scale=pairing_weight_scale,
                gc_weight_scale=gc_weight_scale,
                position_weight_mode=position_weight_mode,
                position_weight_scale=position_weight_scale,
            )

            patch_to_motif_raw_score_v2[patch_idx, motif_idx] = patch_score * structure_multiplier

    patch_motif_structure_energy_v2 = 1.0 - np.exp(
        -patch_to_motif_raw_score_v2 / max(float(label_tau), 1e-8)
    )

    patch_motif_structure_energy_v2 = np.clip(
        patch_motif_structure_energy_v2,
        0.0,
        1.0,
    ).astype(np.float32)

    return (
        residue_to_motif_score_v2_raw.astype(np.float32),
        patch_motif_structure_energy_v2,
    )


def build_label_v2_for_sample(
    sample: Dict[str, Any],
    beta_nt: float = 3.0,
    beta_patch: float = 6.0,
    contact_distance_tau: float = 4.0,
    stem_weight: float = 1.5,
    loop_weight: float = 1.1,
    unpaired_weight: float = 0.8,
    pairing_weight_scale: float = 0.5,
    gc_weight_scale: float = 0.25,
    position_weight_mode: str = "center_mild",
    position_weight_scale: float = 0.25,
    label_tau: float = 0.30,
) -> Dict[str, Any]:
    """
    Add final label-v2 fields to one processed sample.

    Note:
        The low-level v2 function keeps its original default label_tau=0.20.
        The training-data rebuild pipeline historically used label_tau=0.30,
        so this wrapper defaults to 0.30 and should normally be controlled
        by configs/data.yaml.
    """
    motif_type_weights = {
        "stem": float(stem_weight),
        "loop": float(loop_weight),
        "unpaired": float(unpaired_weight),
    }

    residue_to_motif_score_v2_raw, patch_motif_structure_energy_v2 = build_structure_aware_patch_motif_label_v2(
        contact_map=sample["contact_map"],
        motifs=sample["motifs"],
        patch_indices=sample["patch_indices"],
        beta_nt=float(beta_nt),
        beta_patch=float(beta_patch),
        contact_distance_tau=float(contact_distance_tau),
        motif_type_weights=motif_type_weights,
        pairing_weight_scale=float(pairing_weight_scale),
        gc_weight_scale=float(gc_weight_scale),
        position_weight_mode=str(position_weight_mode),
        position_weight_scale=float(position_weight_scale),
        label_tau=float(label_tau),
    )

    out = dict(sample)
    out["residue_to_motif_score_v2_raw"] = residue_to_motif_score_v2_raw
    out["patch_motif_structure_energy_v2"] = patch_motif_structure_energy_v2

    return out


def recompute_v2_labels_for_processed_split(
    processed_split: List[Dict[str, Any]],
    label_cfg: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """
    Recompute label-v2 for a processed split.
    """
    out_split = []

    for idx, sample in enumerate(processed_split, start=1):
        new_sample = build_label_v2_for_sample(
            sample=sample,
            beta_nt=float(label_cfg.get("beta_nt", 3.0)),
            beta_patch=float(label_cfg.get("beta_patch", 6.0)),
            contact_distance_tau=float(label_cfg.get("contact_distance_tau", 4.0)),
            stem_weight=float(label_cfg.get("stem_weight", 1.5)),
            loop_weight=float(label_cfg.get("loop_weight", 1.1)),
            unpaired_weight=float(label_cfg.get("unpaired_weight", 0.8)),
            pairing_weight_scale=float(label_cfg.get("pairing_weight_scale", 0.5)),
            gc_weight_scale=float(label_cfg.get("gc_weight_scale", 0.25)),
            position_weight_mode=str(label_cfg.get("position_weight_mode", "center_mild")),
            position_weight_scale=float(label_cfg.get("position_weight_scale", 0.25)),
            label_tau=float(label_cfg.get("label_tau", 0.30)),
        )

        out_split.append(new_sample)

        if idx % 50 == 0:
            print(f"Recomputed label v2 for {idx}/{len(processed_split)} samples")

    return out_split


def validate_label_v2_shapes(
    processed_split: List[Dict[str, Any]],
    split_name: str,
) -> Dict[str, Any]:
    """
    Validate that patch_motif_structure_energy_v2 exists and matches
    [num_patches, num_motifs].
    """
    bad_examples = []

    for idx, sample in enumerate(processed_split):
        if "patch_motif_structure_energy_v2" not in sample:
            bad_examples.append(f"{idx}: missing patch_motif_structure_energy_v2")
            continue

        y = np.asarray(sample["patch_motif_structure_energy_v2"])

        if y.ndim != 2:
            bad_examples.append(f"{idx}: label_v2 ndim={y.ndim}")
            continue

        num_patches = int(np.asarray(sample["patch_indices"]).shape[0])
        num_motifs = int(len(sample["motifs"]))

        if y.shape != (num_patches, num_motifs):
            bad_examples.append(
                f"{idx}: label_v2 shape={y.shape}, expected={(num_patches, num_motifs)}"
            )

    status = "ok" if len(bad_examples) == 0 else "failed"

    return {
        "split": split_name,
        "num_samples": len(processed_split),
        "status": status,
        "bad_examples": bad_examples[:10],
    }