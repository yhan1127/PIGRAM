# src/pigram/data/attr.py
# -*- coding: utf-8 -*-

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from pigram.utils.constants import AMINO_ACID_VOCAB
from pigram.utils.io import load_pickle, save_json, save_pickle


def _make_group_id_array(symbols: List[str]) -> np.ndarray:
    return np.asarray(
        [AMINO_ACID_VOCAB[symbol] for symbol in symbols],
        dtype=np.int64,
    )


AA_IDS_POS = _make_group_id_array(["K", "R", "H"])
AA_IDS_NEG = _make_group_id_array(["D", "E"])
AA_IDS_AROMATIC = _make_group_id_array(["F", "Y", "W"])
AA_IDS_HYDROPHOBIC = _make_group_id_array(["A", "V", "I", "L", "M", "F", "W", "Y"])
AA_IDS_POLAR = _make_group_id_array(["S", "T", "N", "Q", "C"])
AA_IDS_GLY = _make_group_id_array(["G"])
AA_IDS_PRO = _make_group_id_array(["P"])
AA_IDS_SMALL = _make_group_id_array(["A", "G", "S", "C", "P"])
AA_IDS_BULKY = _make_group_id_array(["F", "W", "Y", "R", "K", "L", "I", "M"])


# =============================================================================
# Patch attributes: 56D
# =============================================================================

def _ratio_for_group_np(
    patch_token_ids: np.ndarray,
    group_ids: np.ndarray,
) -> np.ndarray:
    mask = np.isin(patch_token_ids, group_ids).astype(np.float32)
    return mask.mean(axis=1, keepdims=True)


def _compute_patch_aa_composition_np(
    patch_token_ids: np.ndarray,
) -> np.ndarray:
    n_patches, _ = patch_token_ids.shape
    aa_comp = np.zeros((n_patches, 21), dtype=np.float32)

    clamped = np.clip(patch_token_ids, 0, 21)

    for aa_id in range(1, 22):
        aa_comp[:, aa_id - 1] = (clamped == aa_id).mean(axis=1).astype(np.float32)

    return aa_comp


def compute_patch_attr_from_processed_sample(sample: Dict[str, Any]) -> np.ndarray:
    """
    Compute 56D patch_attr from an already processed sample.

    Layout:
        0-20   amino-acid composition, 21D
        21-29  chemistry group ratios, 9D
        30-39  distance statistics, 10D
        40-48  sequence-topology statistics, 9D
        49-55  derived chemistry/topology proxies, 7D

    Total:
        21 + 9 + 10 + 9 + 7 = 56
    """
    protein_token_ids = np.asarray(sample["protein_token_ids"], dtype=np.int64)
    patch_indices = np.asarray(sample["patch_indices"], dtype=np.int64)
    patch_center_indices = np.asarray(sample["patch_center_indices"], dtype=np.int64)
    patch_relative_distances = np.asarray(sample["patch_relative_distances"], dtype=np.float32)

    patch_token_ids = protein_token_ids[patch_indices]
    center_idx = patch_center_indices[:, None]

    seq_offsets = patch_indices - center_idx
    abs_seq_offsets = np.abs(seq_offsets).astype(np.float32)

    dists = patch_relative_distances.astype(np.float32)

    is_center = patch_indices == center_idx
    non_center_mask = (~is_center).astype(np.float32)
    non_center_count = np.clip(
        non_center_mask.sum(axis=1, keepdims=True),
        a_min=1.0,
        a_max=None,
    )

    # 21D amino-acid composition.
    aa_comp = _compute_patch_aa_composition_np(patch_token_ids)

    # 9D chemistry group ratios.
    frac_pos = _ratio_for_group_np(patch_token_ids, AA_IDS_POS)
    frac_neg = _ratio_for_group_np(patch_token_ids, AA_IDS_NEG)
    frac_aromatic = _ratio_for_group_np(patch_token_ids, AA_IDS_AROMATIC)
    frac_hydrophobic = _ratio_for_group_np(patch_token_ids, AA_IDS_HYDROPHOBIC)
    frac_polar = _ratio_for_group_np(patch_token_ids, AA_IDS_POLAR)
    frac_gly = _ratio_for_group_np(patch_token_ids, AA_IDS_GLY)
    frac_pro = _ratio_for_group_np(patch_token_ids, AA_IDS_PRO)
    frac_small = _ratio_for_group_np(patch_token_ids, AA_IDS_SMALL)
    frac_bulky = _ratio_for_group_np(patch_token_ids, AA_IDS_BULKY)

    chemistry_feat = np.concatenate(
        [
            frac_pos,
            frac_neg,
            frac_aromatic,
            frac_hydrophobic,
            frac_polar,
            frac_gly,
            frac_pro,
            frac_small,
            frac_bulky,
        ],
        axis=-1,
    ).astype(np.float32)

    # 10D distance statistics.
    dist_mean = dists.mean(axis=1, keepdims=True)
    dist_std = dists.std(axis=1, keepdims=True)
    dist_min = dists.min(axis=1, keepdims=True)
    dist_max = dists.max(axis=1, keepdims=True)
    dist_median = np.median(dists, axis=1, keepdims=True)

    non_center_dists = dists * non_center_mask
    non_center_dist_mean = non_center_dists.sum(axis=1, keepdims=True) / non_center_count

    centered_non_center = (dists - non_center_dist_mean) * non_center_mask
    non_center_dist_std = np.sqrt(
        np.clip(
            (centered_non_center ** 2).sum(axis=1, keepdims=True) / non_center_count,
            a_min=1e-8,
            a_max=None,
        )
    )

    non_center_dist_max = non_center_dists.max(axis=1, keepdims=True)

    masked_for_min = np.where(non_center_mask > 0, dists, 1e9)
    non_center_dist_min = masked_for_min.min(axis=1, keepdims=True)
    non_center_dist_min = np.where(
        np.isfinite(non_center_dist_min),
        non_center_dist_min,
        0.0,
    ).astype(np.float32)

    frac_within_4a = (dists <= 4.0).astype(np.float32).mean(axis=1, keepdims=True)
    frac_within_8a = (dists <= 8.0).astype(np.float32).mean(axis=1, keepdims=True)

    distance_feat = np.concatenate(
        [
            dist_mean,
            dist_std,
            dist_min,
            dist_max,
            dist_median,
            non_center_dist_mean,
            non_center_dist_std,
            non_center_dist_min,
            non_center_dist_max,
            0.5 * (frac_within_4a + frac_within_8a),
        ],
        axis=-1,
    ).astype(np.float32)

    # 9D sequence-topology statistics.
    mean_abs_offset = abs_seq_offsets.mean(axis=1, keepdims=True)
    std_abs_offset = abs_seq_offsets.std(axis=1, keepdims=True)
    max_abs_offset = abs_seq_offsets.max(axis=1, keepdims=True)

    frac_local_2 = (abs_seq_offsets <= 2).astype(np.float32).mean(axis=1, keepdims=True)
    frac_local_4 = (abs_seq_offsets <= 4).astype(np.float32).mean(axis=1, keepdims=True)
    frac_local_8 = (abs_seq_offsets <= 8).astype(np.float32).mean(axis=1, keepdims=True)
    frac_long_16 = (abs_seq_offsets > 16).astype(np.float32).mean(axis=1, keepdims=True)

    signed_offset_mean = seq_offsets.astype(np.float32).mean(axis=1, keepdims=True)
    pos_frac = (seq_offsets > 0).astype(np.float32).mean(axis=1, keepdims=True)
    neg_frac = (seq_offsets < 0).astype(np.float32).mean(axis=1, keepdims=True)
    signed_balance = pos_frac - neg_frac

    topology_feat = np.concatenate(
        [
            mean_abs_offset,
            std_abs_offset,
            max_abs_offset,
            frac_local_2,
            frac_local_4,
            frac_local_8,
            frac_long_16,
            signed_offset_mean,
            signed_balance,
        ],
        axis=-1,
    ).astype(np.float32)

    # 7D derived features.
    net_charge = frac_pos - frac_neg
    charge_magnitude = frac_pos + frac_neg
    cation_pi_proxy = frac_pos * frac_aromatic
    aromatic_hydrophobic_packing = frac_aromatic * frac_hydrophobic
    compactness_proxy = 1.0 / (1.0 + non_center_dist_mean)
    locality_balance = frac_local_4 - frac_long_16
    asymmetry_strength = np.abs(signed_balance)

    derived_feat = np.concatenate(
        [
            net_charge,
            charge_magnitude,
            cation_pi_proxy,
            aromatic_hydrophobic_packing,
            compactness_proxy,
            locality_balance,
            asymmetry_strength,
        ],
        axis=-1,
    ).astype(np.float32)

    patch_attr = np.concatenate(
        [
            aa_comp,
            chemistry_feat,
            distance_feat,
            topology_feat,
            derived_feat,
        ],
        axis=-1,
    ).astype(np.float32)

    if patch_attr.shape[1] != 56:
        raise RuntimeError(f"patch_attr must be 56D, got shape={patch_attr.shape}")

    return patch_attr


# =============================================================================
# Motif attributes: 27D
# =============================================================================

def _normalize_nt_char(ch: str) -> str:
    ch = str(ch).upper()

    if ch == "T":
        return "U"

    if ch in {"A", "C", "G", "U"}:
        return ch

    return "N"


def build_pair_map(dot_bracket: str) -> List[int]:
    n = len(dot_bracket)
    pair_map = [-1] * n
    stack: List[int] = []

    for i, ch in enumerate(dot_bracket):
        if ch == "(":
            stack.append(i)
        elif ch == ")":
            if not stack:
                continue
            j = stack.pop()
            pair_map[i] = j
            pair_map[j] = i

    return pair_map


def _motif_nt_fractions(sequence: str) -> Dict[str, float]:
    seq = [_normalize_nt_char(ch) for ch in sequence]
    n = max(len(seq), 1)

    count_a = sum(ch == "A" for ch in seq)
    count_c = sum(ch == "C" for ch in seq)
    count_g = sum(ch == "G" for ch in seq)
    count_u = sum(ch == "U" for ch in seq)

    frac_a = count_a / n
    frac_c = count_c / n
    frac_g = count_g / n
    frac_u = count_u / n

    return {
        "frac_A": float(frac_a),
        "frac_C": float(frac_c),
        "frac_G": float(frac_g),
        "frac_U": float(frac_u),
        "frac_purine": float(frac_a + frac_g),
    }


def _nt_entropy_from_fractions(
    frac_a: float,
    frac_c: float,
    frac_g: float,
    frac_u: float,
) -> float:
    probs = [frac_a, frac_c, frac_g, frac_u]
    eps = 1e-8
    entropy = 0.0

    for prob in probs:
        if prob > 0:
            entropy -= prob * math.log(prob + eps)

    return float(entropy / math.log(4.0))


def _max_nt_fraction(
    frac_a: float,
    frac_c: float,
    frac_g: float,
    frac_u: float,
) -> float:
    return float(max(frac_a, frac_c, frac_g, frac_u))


def _max_run_fraction(sequence: str) -> float:
    seq = [_normalize_nt_char(ch) for ch in sequence]

    if len(seq) == 0:
        return 0.0

    best = 1
    current = 1

    for i in range(1, len(seq)):
        if seq[i] == seq[i - 1]:
            current += 1
            best = max(best, current)
        else:
            current = 1

    return float(best / len(seq))


def _compute_local_wobble_pair_fraction(
    sequence: str,
    dot_bracket: str,
) -> float:
    if len(sequence) == 0 or len(sequence) != len(dot_bracket):
        return 0.0

    seq = [_normalize_nt_char(ch) for ch in sequence]
    pair_map = build_pair_map(dot_bracket)

    visited = set()
    wobble_pairs = 0
    total_pairs = 0

    for i, j in enumerate(pair_map):
        if j < 0:
            continue

        if i in visited or j in visited:
            continue

        visited.add(i)
        visited.add(j)

        a = seq[i]
        b = seq[j]

        total_pairs += 1

        if (a == "G" and b == "U") or (a == "U" and b == "G"):
            wobble_pairs += 1

    if total_pairs == 0:
        return 0.0

    return float(wobble_pairs / total_pairs)


def ensure_motif_chemistry_features(sample: Dict[str, Any]) -> None:
    """
    Add motif chemistry features in-place.

    This is safe to call repeatedly.
    """
    na_source = str(sample.get("na_source", "unknown")).lower()
    is_dna = 1.0 if na_source == "dna" else 0.0

    for motif in sample["motifs"]:
        sequence = motif.get("sequence", "")
        dot_bracket = motif.get("dot_bracket", "")

        nt_fracs = _motif_nt_fractions(sequence)
        frac_a = nt_fracs["frac_A"]
        frac_c = nt_fracs["frac_C"]
        frac_g = nt_fracs["frac_G"]
        frac_u = nt_fracs["frac_U"]

        pairing_ratio = float(motif.get("pairing_ratio", 0.0))
        gc_content = float(motif.get("gc_content", 0.0))

        motif["frac_A"] = frac_a
        motif["frac_C"] = frac_c
        motif["frac_G"] = frac_g
        motif["frac_U"] = frac_u
        motif["frac_purine"] = nt_fracs["frac_purine"]
        motif["nt_entropy"] = _nt_entropy_from_fractions(frac_a, frac_c, frac_g, frac_u)
        motif["max_nt_fraction"] = _max_nt_fraction(frac_a, frac_c, frac_g, frac_u)
        motif["max_run_fraction"] = _max_run_fraction(sequence)
        motif["flexibility_proxy"] = 0.5 * (1.0 - pairing_ratio) + 0.5 * (1.0 - gc_content)
        motif["wobble_pair_fraction"] = _compute_local_wobble_pair_fraction(sequence, dot_bracket)
        motif["is_dna"] = is_dna


def compute_motif_attr_from_processed_sample(sample: Dict[str, Any]) -> np.ndarray:
    """
    Compute 27D motif_attr from motif dictionaries.

    Layout:
        0-7    motif type one-hot, 8D
        8-10   edge-side one-hot, 3D
        11     log motif length
        12     pairing ratio
        13     GC content
        14     terminal flag
        15     normalized motif center position
        16-26  chemistry-aware motif features, 11D

    Total:
        8 + 3 + 5 + 11 = 27
    """
    ensure_motif_chemistry_features(sample)

    motifs = sample["motifs"]
    num_motifs = len(motifs)

    motif_attr = np.zeros((num_motifs, 27), dtype=np.float32)

    motif_type_vocab_size = 8
    edge_side_vocab_size = 3

    for motif_idx, motif in enumerate(motifs):
        type_id = int(motif.get("type_id", 0))
        type_id = max(0, min(type_id, motif_type_vocab_size - 1))
        motif_attr[motif_idx, type_id] = 1.0

        edge_side = str(motif.get("edge_side", "none")).lower()

        if edge_side == "left":
            edge_id = 1
        elif edge_side == "right":
            edge_id = 2
        else:
            edge_id = 0

        motif_attr[motif_idx, motif_type_vocab_size + edge_id] = 1.0

        offset = motif_type_vocab_size + edge_side_vocab_size

        motif_attr[motif_idx, offset + 0] = np.log1p(float(motif["length"])) / 5.0
        motif_attr[motif_idx, offset + 1] = float(motif["pairing_ratio"])
        motif_attr[motif_idx, offset + 2] = float(motif["gc_content"])
        motif_attr[motif_idx, offset + 3] = float(motif["is_terminal"])
        motif_attr[motif_idx, offset + 4] = float(motif["normalized_center_position"])

        chemistry_offset = offset + 5

        motif_attr[motif_idx, chemistry_offset + 0] = float(motif.get("frac_A", 0.0))
        motif_attr[motif_idx, chemistry_offset + 1] = float(motif.get("frac_C", 0.0))
        motif_attr[motif_idx, chemistry_offset + 2] = float(motif.get("frac_G", 0.0))
        motif_attr[motif_idx, chemistry_offset + 3] = float(motif.get("frac_U", 0.0))
        motif_attr[motif_idx, chemistry_offset + 4] = float(motif.get("frac_purine", 0.0))
        motif_attr[motif_idx, chemistry_offset + 5] = float(motif.get("nt_entropy", 0.0))
        motif_attr[motif_idx, chemistry_offset + 6] = float(motif.get("max_nt_fraction", 0.0))
        motif_attr[motif_idx, chemistry_offset + 7] = float(motif.get("max_run_fraction", 0.0))
        motif_attr[motif_idx, chemistry_offset + 8] = float(motif.get("flexibility_proxy", 0.0))
        motif_attr[motif_idx, chemistry_offset + 9] = float(motif.get("wobble_pair_fraction", 0.0))
        motif_attr[motif_idx, chemistry_offset + 10] = float(motif.get("is_dna", 0.0))

    if motif_attr.shape[1] != 27:
        raise RuntimeError(f"motif_attr must be 27D, got shape={motif_attr.shape}")

    return motif_attr


# =============================================================================
# Split augmentation
# =============================================================================

def augment_processed_sample_with_precomputed_attrs(
    sample: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Add patch_attr (56D) and motif_attr (27D) to one processed sample.

    Also adds motif chemistry arrays for analysis/debugging.
    """
    out = dict(sample)

    out["patch_attr"] = compute_patch_attr_from_processed_sample(out)
    out["motif_attr"] = compute_motif_attr_from_processed_sample(out)

    motifs = out["motifs"]

    out["motif_frac_A"] = np.asarray([m.get("frac_A", 0.0) for m in motifs], dtype=np.float32)
    out["motif_frac_C"] = np.asarray([m.get("frac_C", 0.0) for m in motifs], dtype=np.float32)
    out["motif_frac_G"] = np.asarray([m.get("frac_G", 0.0) for m in motifs], dtype=np.float32)
    out["motif_frac_U"] = np.asarray([m.get("frac_U", 0.0) for m in motifs], dtype=np.float32)
    out["motif_frac_purine"] = np.asarray([m.get("frac_purine", 0.0) for m in motifs], dtype=np.float32)
    out["motif_nt_entropy"] = np.asarray([m.get("nt_entropy", 0.0) for m in motifs], dtype=np.float32)
    out["motif_max_nt_fraction"] = np.asarray([m.get("max_nt_fraction", 0.0) for m in motifs], dtype=np.float32)
    out["motif_max_run_fraction"] = np.asarray([m.get("max_run_fraction", 0.0) for m in motifs], dtype=np.float32)
    out["motif_flexibility_proxy"] = np.asarray([m.get("flexibility_proxy", 0.0) for m in motifs], dtype=np.float32)
    out["motif_wobble_pair_fraction"] = np.asarray([m.get("wobble_pair_fraction", 0.0) for m in motifs], dtype=np.float32)
    out["motif_is_dna"] = np.asarray([m.get("is_dna", 0.0) for m in motifs], dtype=np.float32)

    return out


def augment_processed_split_with_precomputed_attrs(
    processed_split: List[Dict[str, Any]],
    verbose: bool = True,
) -> List[Dict[str, Any]]:
    augmented = []

    for idx, sample in enumerate(processed_split):
        try:
            augmented.append(augment_processed_sample_with_precomputed_attrs(sample))

            if verbose and (idx + 1) % 50 == 0:
                print(f"Augmented attrs for {idx + 1}/{len(processed_split)} samples")

        except Exception as exc:
            print(
                f"[Warning] Failed to augment sample {idx}: "
                f"pdb={sample.get('pdb')} error={exc}"
            )

    return augmented


def summarize_attr_split(
    processed_split: List[Dict[str, Any]],
    split_name: str,
) -> Dict[str, Any]:
    if len(processed_split) == 0:
        return {
            "split": split_name,
            "num_samples": 0,
            "patch_attr_dim": 0,
            "motif_attr_dim": 0,
            "has_precomputed_patch_attr": False,
            "has_precomputed_motif_attr": False,
        }

    patch_dims = []
    motif_dims = []
    num_patches = []
    num_motifs = []
    bad_examples = []

    for idx, sample in enumerate(processed_split):
        if "patch_attr" not in sample:
            bad_examples.append(f"{idx}: missing patch_attr")
            continue

        if "motif_attr" not in sample:
            bad_examples.append(f"{idx}: missing motif_attr")
            continue

        patch_attr = np.asarray(sample["patch_attr"])
        motif_attr = np.asarray(sample["motif_attr"])

        if patch_attr.ndim != 2:
            bad_examples.append(f"{idx}: patch_attr ndim={patch_attr.ndim}")
        else:
            patch_dims.append(int(patch_attr.shape[1]))
            num_patches.append(int(patch_attr.shape[0]))

        if motif_attr.ndim != 2:
            bad_examples.append(f"{idx}: motif_attr ndim={motif_attr.ndim}")
        else:
            motif_dims.append(int(motif_attr.shape[1]))
            num_motifs.append(int(motif_attr.shape[0]))

    patch_dim = patch_dims[0] if patch_dims else 0
    motif_dim = motif_dims[0] if motif_dims else 0

    return {
        "split": split_name,
        "num_samples": len(processed_split),
        "avg_num_patches": float(np.mean(num_patches)) if num_patches else 0.0,
        "avg_num_motifs": float(np.mean(num_motifs)) if num_motifs else 0.0,
        "patch_attr_dim": int(patch_dim),
        "motif_attr_dim": int(motif_dim),
        "patch_attr_dim_values": sorted(set(patch_dims)),
        "motif_attr_dim_values": sorted(set(motif_dims)),
        "has_precomputed_patch_attr": len(patch_dims) == len(processed_split),
        "has_precomputed_motif_attr": len(motif_dims) == len(processed_split),
        "bad_examples": bad_examples[:10],
    }


def validate_attr_dims(
    processed_split: List[Dict[str, Any]],
    expected_patch_dim: int = 56,
    expected_motif_dim: int = 27,
    split_name: str = "split",
) -> Dict[str, Any]:
    summary = summarize_attr_split(processed_split, split_name=split_name)

    patch_ok = (
        summary["has_precomputed_patch_attr"]
        and summary["patch_attr_dim_values"] == [int(expected_patch_dim)]
    )
    motif_ok = (
        summary["has_precomputed_motif_attr"]
        and summary["motif_attr_dim_values"] == [int(expected_motif_dim)]
    )

    summary["expected_patch_attr_dim"] = int(expected_patch_dim)
    summary["expected_motif_attr_dim"] = int(expected_motif_dim)
    summary["patch_dim_ok"] = bool(patch_ok)
    summary["motif_dim_ok"] = bool(motif_ok)
    summary["status"] = "ok" if patch_ok and motif_ok else "failed"

    return summary


def augment_one_split(
    in_path: str | Path,
    out_path: str | Path,
    split_name: str,
    expected_patch_dim: int = 56,
    expected_motif_dim: int = 27,
    verbose: bool = True,
) -> Dict[str, Any]:
    in_path = Path(in_path)
    out_path = Path(out_path)

    print(f"\n[Attr] Loading {split_name}: {in_path}")
    data = load_pickle(in_path)

    print(f"[Attr] Augmenting {split_name}: n={len(data)}")
    augmented = augment_processed_split_with_precomputed_attrs(
        data,
        verbose=verbose,
    )

    summary = validate_attr_dims(
        processed_split=augmented,
        expected_patch_dim=expected_patch_dim,
        expected_motif_dim=expected_motif_dim,
        split_name=split_name,
    )

    if summary["status"] != "ok":
        raise RuntimeError(
            f"Attribute validation failed for split={split_name}:\n{summary}"
        )

    save_pickle(augmented, out_path)

    print(f"[Attr] Saved {split_name}: {out_path}")
    print(
        f"[Attr] patch_attr_dim={summary['patch_attr_dim']}, "
        f"motif_attr_dim={summary['motif_attr_dim']}"
    )

    return {
        "input": str(in_path),
        "output": str(out_path),
        **summary,
    }


def augment_processed_attrs(cfg: Dict[str, Any]) -> None:
    """
    Config-driven entrypoint used by scripts/data/attr.py.
    """
    attr_cfg = cfg["attr"]

    expected_patch_dim = int(attr_cfg.get("expected_patch_attr_dim", 56))
    expected_motif_dim = int(attr_cfg.get("expected_motif_attr_dim", 27))
    verbose = bool(attr_cfg.get("verbose", True))

    summaries = []

    summaries.append(
        augment_one_split(
            in_path=attr_cfg["train_in"],
            out_path=attr_cfg["train_out"],
            split_name="train",
            expected_patch_dim=expected_patch_dim,
            expected_motif_dim=expected_motif_dim,
            verbose=verbose,
        )
    )

    summaries.append(
        augment_one_split(
            in_path=attr_cfg["val_in"],
            out_path=attr_cfg["val_out"],
            split_name="val",
            expected_patch_dim=expected_patch_dim,
            expected_motif_dim=expected_motif_dim,
            verbose=verbose,
        )
    )

    summaries.append(
        augment_one_split(
            in_path=attr_cfg["test_in"],
            out_path=attr_cfg["test_out"],
            split_name="test",
            expected_patch_dim=expected_patch_dim,
            expected_motif_dim=expected_motif_dim,
            verbose=verbose,
        )
    )

    summary_payload = {
        "expected_patch_attr_dim": expected_patch_dim,
        "expected_motif_attr_dim": expected_motif_dim,
        "splits": summaries,
        "final_training_files": {
            "train": attr_cfg["train_out"],
            "val": attr_cfg["val_out"],
            "test": attr_cfg["test_out"],
        },
    }

    save_json(summary_payload, attr_cfg["summary_out"])

    print("\n[Attr] Saved summary:")
    print(attr_cfg["summary_out"])