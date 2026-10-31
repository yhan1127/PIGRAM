# src/pigram/data/__init__.py
# -*- coding: utf-8 -*-

from __future__ import annotations

from pigram.data.attr import (
    augment_processed_sample_with_precomputed_attrs,
    augment_processed_split_with_precomputed_attrs,
    compute_motif_attr_from_processed_sample,
    compute_patch_attr_from_processed_sample,
)
from pigram.data.dataset import (
    infer_na_source,
    process_single_sample,
    process_split,
    summarize_processed_split,
)
from pigram.data.fold import normalize_rna_sequence, predict_secondary_structure
from pigram.data.label import (
    build_label_v2_for_sample,
    build_patch_motif_energy_label,
    build_patch_motif_supervision,
    build_structure_aware_patch_motif_label,
    build_structure_aware_patch_motif_label_v2,
    derive_patch_importance_from_labels,
    recompute_v2_labels_for_processed_split,
    validate_label_v2_shapes,
)
from pigram.data.loader import (
    attach_annotation_tensors,
    estimate_pos_weight,
    make_dummy_annotation,
    move_sample_to_device,
    move_sample_to_device_with_ann,
    prepare_single_sample_tensors,
)
from pigram.data.motif import (
    MOTIF_TYPE_ID_TO_NAME,
    MOTIF_TYPE_TO_ID,
    build_motifs_from_dotbracket,
    build_pair_map,
    classify_unpaired_region,
    compute_normalized_center_position,
    infer_edge_side,
    is_stem_edge,
    merge_consecutive_labels,
    token_level_labels,
)
from pigram.data.patch import (
    build_knn_patches,
    build_protein_patches,
    compute_patch_relative_distances,
    compute_residue_centroids,
    pairwise_distances,
)

__all__ = [
    "normalize_rna_sequence",
    "predict_secondary_structure",
    "compute_residue_centroids",
    "pairwise_distances",
    "build_knn_patches",
    "compute_patch_relative_distances",
    "build_protein_patches",
    "MOTIF_TYPE_TO_ID",
    "MOTIF_TYPE_ID_TO_NAME",
    "build_pair_map",
    "is_stem_edge",
    "classify_unpaired_region",
    "token_level_labels",
    "merge_consecutive_labels",
    "infer_edge_side",
    "compute_normalized_center_position",
    "build_motifs_from_dotbracket",
    "build_patch_motif_supervision",
    "derive_patch_importance_from_labels",
    "build_patch_motif_energy_label",
    "build_structure_aware_patch_motif_label",
    "build_structure_aware_patch_motif_label_v2",
    "build_label_v2_for_sample",
    "recompute_v2_labels_for_processed_split",
    "validate_label_v2_shapes",
    "compute_patch_attr_from_processed_sample",
    "compute_motif_attr_from_processed_sample",
    "augment_processed_sample_with_precomputed_attrs",
    "augment_processed_split_with_precomputed_attrs",
    "prepare_single_sample_tensors",
    "attach_annotation_tensors",
    "move_sample_to_device",
    "move_sample_to_device_with_ann",
    "estimate_pos_weight",
    "make_dummy_annotation",
    "infer_na_source",
    "process_single_sample",
    "process_split",
    "summarize_processed_split",
]