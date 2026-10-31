# src/pigram/data/dataset.py
# -*- coding: utf-8 -*-

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np

from pigram.data.attr import augment_processed_sample_with_precomputed_attrs
from pigram.data.fold import normalize_rna_sequence, predict_secondary_structure
from pigram.data.label import (
    build_patch_motif_energy_label,
    build_patch_motif_supervision,
    build_structure_aware_patch_motif_label,
    derive_patch_importance_from_labels,
)
from pigram.data.motif import build_motifs_from_dotbracket
from pigram.data.patch import build_protein_patches
from pigram.utils.sequence import encode_protein_sequence, encode_rna_sequence


def infer_na_source(raw_sample: Dict[str, Any], default: str = "unknown") -> str:
    """
    Infer nucleic-acid source from raw sample metadata.

    Priority:
        1. explicit raw_sample["na_source"]
        2. raw_sample["kind"]
        3. sequence characters
        4. fallback default
    """
    if raw_sample.get("na_source") is not None:
        value = str(raw_sample["na_source"]).lower()
        if "dna" in value:
            return "dna"
        if "rna" in value:
            return "rna"
        return value

    kind = raw_sample.get("kind", None)
    if kind is not None:
        kind = str(kind).lower()
        if "dna" in kind:
            return "dna"
        if "rna" in kind:
            return "rna"

    partner_seq = str(raw_sample.get("partner_seq", "")).upper()
    has_t = "T" in partner_seq
    has_u = "U" in partner_seq

    if has_t and not has_u:
        return "dna"
    if has_u and not has_t:
        return "rna"

    return default


def _ensure_2d_contact_map(
    contact_map: Any,
    protein_len: int,
    partner_len: int,
) -> np.ndarray:
    contact_map = np.asarray(contact_map, dtype=np.float32)

    if contact_map.ndim != 2:
        raise ValueError(f"contact_map must be 2D, got shape={contact_map.shape}")

    expected = (protein_len, partner_len)
    if contact_map.shape != expected:
        raise ValueError(
            f"contact_map shape mismatch: got {contact_map.shape}, expected {expected}"
        )

    return contact_map


def _motif_arrays_from_motifs(motifs: List[Dict[str, Any]]) -> Dict[str, np.ndarray]:
    """
    Convert motif dictionaries into arrays consumed by model loaders/encoders.
    """
    motif_token_ids_list = []
    motif_type_ids = []
    motif_lengths = []
    motif_pairing_ratios = []
    motif_gc_contents = []
    motif_edge_side_ids = []
    motif_is_terminal = []
    motif_center_positions = []
    motif_is_dna = []

    for motif in motifs:
        motif_seq = str(motif.get("sequence", ""))
        motif_token_ids_list.append(
            np.asarray(encode_rna_sequence(motif_seq), dtype=np.int64)
        )

        motif_type_ids.append(int(motif.get("type_id", 0)))
        motif_lengths.append(float(motif.get("length", len(motif_seq))))
        motif_pairing_ratios.append(float(motif.get("pairing_ratio", 0.0)))
        motif_gc_contents.append(float(motif.get("gc_content", 0.0)))
        motif_edge_side_ids.append(int(motif.get("edge_side_id", 0)))
        motif_is_terminal.append(float(motif.get("is_terminal", 0.0)))
        motif_center_positions.append(float(motif.get("normalized_center_position", 0.0)))
        motif_is_dna.append(float(motif.get("is_dna", 0.0)))

    return {
        "motif_token_ids_list": motif_token_ids_list,
        "motif_type_ids": np.asarray(motif_type_ids, dtype=np.int64),
        "motif_lengths": np.asarray(motif_lengths, dtype=np.float32),
        "motif_pairing_ratios": np.asarray(motif_pairing_ratios, dtype=np.float32),
        "motif_gc_contents": np.asarray(motif_gc_contents, dtype=np.float32),
        "motif_edge_side_ids": np.asarray(motif_edge_side_ids, dtype=np.int64),
        "motif_is_terminal": np.asarray(motif_is_terminal, dtype=np.float32),
        "motif_center_positions": np.asarray(motif_center_positions, dtype=np.float32),
        "motif_is_dna": np.asarray(motif_is_dna, dtype=np.float32),
    }


def process_single_sample(
    raw_sample: Dict[str, Any],
    patch_k: int = 8,
    residue_motif_agg: str = "topk_mean",
    patch_motif_agg: str = "topk_mean",
    threshold: float = 6.0,
    residue_top_k: int = 3,
    patch_top_k: int = 3,
    energy_tau: float = 4.0,
    add_precomputed_attrs: bool = True,
) -> Dict[str, Any]:
    """
    Convert one raw protein--nucleic-acid sample into a PIGRAM patch--motif sample.

    Required raw_sample keys:
        protein_seq
        protein_coords
        partner_seq
        contact_map

    Output includes:
        protein patches
        ssNA motifs
        binary patch--motif label
        patch importance label
        distance-decay energy label
        structure-aware compatibility label
        optional precomputed patch_attr 56D and motif_attr 27D
    """
    protein_seq = str(raw_sample["protein_seq"])
    raw_partner_seq = str(raw_sample["partner_seq"])
    partner_seq = normalize_rna_sequence(raw_partner_seq)

    na_source = infer_na_source(raw_sample)

    protein_coords = np.asarray(raw_sample["protein_coords"], dtype=np.float32)

    if protein_coords.ndim != 3 or protein_coords.shape[-1] != 3:
        raise ValueError(
            f"protein_coords must have shape [L, A, 3], got {protein_coords.shape}"
        )

    if len(protein_seq) != protein_coords.shape[0]:
        raise ValueError(
            f"protein_seq length {len(protein_seq)} does not match "
            f"protein_coords length {protein_coords.shape[0]}"
        )

    contact_map = _ensure_2d_contact_map(
        contact_map=raw_sample["contact_map"],
        protein_len=len(protein_seq),
        partner_len=len(partner_seq),
    )

    protein_token_ids = np.asarray(
        encode_protein_sequence(protein_seq),
        dtype=np.int64,
    )

    secondary_structure, mfe = predict_secondary_structure(partner_seq)

    motifs = build_motifs_from_dotbracket(
        sequence=partner_seq,
        dot_bracket=secondary_structure,
        na_source=na_source,
    )

    if len(motifs) == 0:
        raise RuntimeError(
            f"No motifs generated for sample {raw_sample.get('pdb', '')}."
        )

    patch_info = build_protein_patches(
        protein_coords=protein_coords,
        k=patch_k,
    )

    residue_to_motif_dist, patch_to_motif_dist, patch_motif_label = build_patch_motif_supervision(
        contact_map=contact_map,
        motifs=motifs,
        patch_indices=patch_info["patch_indices"],
        residue_motif_agg=residue_motif_agg,
        patch_motif_agg=patch_motif_agg,
        threshold=threshold,
        residue_top_k=residue_top_k,
        patch_top_k=patch_top_k,
    )

    patch_importance = derive_patch_importance_from_labels(patch_motif_label)

    patch_motif_energy = build_patch_motif_energy_label(
        patch_to_motif_dist=patch_to_motif_dist,
        tau=energy_tau,
    )

    patch_motif_structure_energy = build_structure_aware_patch_motif_label(
        contact_map=contact_map,
        motifs=motifs,
        patch_indices=patch_info["patch_indices"],
        tau=energy_tau,
        position_weight_mode="center",
        motif_type_weights=None,
        pairing_weight_scale=0.5,
        gc_weight_scale=0.3,
        patch_agg=patch_motif_agg,
        patch_top_k=patch_top_k,
    )

    motif_arrays = _motif_arrays_from_motifs(motifs)

    processed = {
        # Metadata.
        "pdb": raw_sample.get("pdb", raw_sample.get("id", "")),
        "kind": raw_sample.get("kind", f"prot-{na_source}"),
        "na_source": na_source,
        "affinity": raw_sample.get("affinity", None),

        # Sequences and structures.
        "protein_seq": protein_seq,
        "partner_seq": partner_seq,
        "raw_partner_seq": raw_partner_seq,
        "secondary_structure": secondary_structure,
        "mfe": float(mfe),

        # Coordinates/contact map retained for label recomputation and analysis.
        "protein_coords": protein_coords,
        "contact_map": contact_map,

        # Protein tokens/patches.
        "protein_token_ids": protein_token_ids,
        "residue_centroids": patch_info["residue_centroids"],
        "patch_indices": patch_info["patch_indices"],
        "patch_center_indices": patch_info["patch_center_indices"],
        "patch_relative_distances": patch_info["patch_relative_distances"],

        # Motifs and motif arrays.
        "motifs": motifs,
        **motif_arrays,

        # Labels.
        "residue_to_motif_dist": residue_to_motif_dist,
        "patch_to_motif_dist": patch_to_motif_dist,
        "patch_motif_label": patch_motif_label,
        "patch_importance": patch_importance,
        "patch_motif_energy": patch_motif_energy,
        "patch_motif_structure_energy": patch_motif_structure_energy,
    }

    # Preserve optional fields useful for downstream / debugging.
    for key in [
        "target",
        "aptamer_id",
        "source_split",
        "split_index",
        "local_index",
        "expected_positive_count",
        "label_high_topN",
        "label_low_topN",
    ]:
        if key in raw_sample:
            processed[key] = raw_sample[key]

    if "partner_coords" in raw_sample:
        processed["partner_coords"] = np.asarray(raw_sample["partner_coords"], dtype=np.float32)

    if add_precomputed_attrs:
        processed = augment_processed_sample_with_precomputed_attrs(processed)

    return processed


def process_split(
    raw_split: List[Dict[str, Any]],
    patch_k: int = 8,
    residue_motif_agg: str = "topk_mean",
    patch_motif_agg: str = "topk_mean",
    threshold: float = 6.0,
    residue_top_k: int = 3,
    patch_top_k: int = 3,
    energy_tau: float = 4.0,
    add_precomputed_attrs: bool = True,
    verbose: bool = True,
) -> List[Dict[str, Any]]:
    """
    Process a raw split into PIGRAM patch--motif samples.

    Samples that fail processing are skipped with a warning.
    """
    processed_split = []

    for idx, raw_sample in enumerate(raw_split):
        try:
            processed = process_single_sample(
                raw_sample=raw_sample,
                patch_k=patch_k,
                residue_motif_agg=residue_motif_agg,
                patch_motif_agg=patch_motif_agg,
                threshold=threshold,
                residue_top_k=residue_top_k,
                patch_top_k=patch_top_k,
                energy_tau=energy_tau,
                add_precomputed_attrs=add_precomputed_attrs,
            )
            processed_split.append(processed)

        except Exception as exc:
            print(
                f"[Warning] Failed to process sample {idx}: "
                f"pdb={raw_sample.get('pdb', '')} error={exc}"
            )

        if verbose and (idx + 1) % 50 == 0:
            print(
                f"Processed {idx + 1}/{len(raw_split)} raw samples; "
                f"kept {len(processed_split)}"
            )

    return processed_split


def summarize_processed_split(processed_split: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Summarize a processed split.
    """
    if len(processed_split) == 0:
        return {
            "num_samples": 0,
            "avg_num_patches": 0.0,
            "avg_num_motifs": 0.0,
            "avg_positive_ratio": 0.0,
            "avg_energy_mean": 0.0,
            "avg_structure_energy_mean": 0.0,
            "avg_structure_energy_v2_mean": 0.0,
            "num_rna": 0,
            "num_dna": 0,
            "num_unknown": 0,
            "has_precomputed_patch_attr": False,
            "has_precomputed_motif_attr": False,
            "patch_attr_dim": 0,
            "motif_attr_dim": 0,
        }

    num_patches = []
    num_motifs = []
    positive_ratios = []
    energy_means = []
    structure_energy_means = []
    structure_energy_v2_means = []

    source_counts = {
        "rna": 0,
        "dna": 0,
        "unknown": 0,
    }

    patch_attr_dims = []
    motif_attr_dims = []

    for sample in processed_split:
        patch_motif_label = np.asarray(sample["patch_motif_label"], dtype=np.float32)
        patch_motif_energy = np.asarray(sample["patch_motif_energy"], dtype=np.float32)
        patch_motif_structure_energy = np.asarray(
            sample["patch_motif_structure_energy"],
            dtype=np.float32,
        )

        num_patches.append(int(patch_motif_label.shape[0]))
        num_motifs.append(int(patch_motif_label.shape[1]))
        positive_ratios.append(float(patch_motif_label.mean()))
        energy_means.append(float(patch_motif_energy.mean()))
        structure_energy_means.append(float(patch_motif_structure_energy.mean()))

        if "patch_motif_structure_energy_v2" in sample:
            structure_energy_v2_means.append(
                float(np.asarray(sample["patch_motif_structure_energy_v2"], dtype=np.float32).mean())
            )

        source = str(sample.get("na_source", "unknown")).lower()
        if source not in source_counts:
            source = "unknown"
        source_counts[source] += 1

        if "patch_attr" in sample:
            patch_attr_dims.append(int(np.asarray(sample["patch_attr"]).shape[1]))
        if "motif_attr" in sample:
            motif_attr_dims.append(int(np.asarray(sample["motif_attr"]).shape[1]))

    return {
        "num_samples": int(len(processed_split)),
        "avg_num_patches": float(np.mean(num_patches)),
        "avg_num_motifs": float(np.mean(num_motifs)),
        "avg_positive_ratio": float(np.mean(positive_ratios)),
        "avg_energy_mean": float(np.mean(energy_means)),
        "avg_structure_energy_mean": float(np.mean(structure_energy_means)),
        "avg_structure_energy_v2_mean": (
            float(np.mean(structure_energy_v2_means))
            if structure_energy_v2_means
            else 0.0
        ),
        "num_rna": int(source_counts["rna"]),
        "num_dna": int(source_counts["dna"]),
        "num_unknown": int(source_counts["unknown"]),
        "has_precomputed_patch_attr": len(patch_attr_dims) == len(processed_split),
        "has_precomputed_motif_attr": len(motif_attr_dims) == len(processed_split),
        "patch_attr_dim": int(patch_attr_dims[0]) if patch_attr_dims else 0,
        "motif_attr_dim": int(motif_attr_dims[0]) if motif_attr_dims else 0,
        "patch_attr_dim_values": sorted(set(patch_attr_dims)),
        "motif_attr_dim_values": sorted(set(motif_attr_dims)),
    }