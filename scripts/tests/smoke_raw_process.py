# scripts/tests/smoke_raw_process.py
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List

from pigram.data.dataset import process_single_sample
from pigram.data.label import build_label_v2_for_sample
from pigram.utils.io import load_pickle


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Smoke test: process one raw sample into PIGRAM processed sample."
    )
    parser.add_argument(
        "--raw_pkl",
        type=Path,
        default=Path("data/raw/prot_rna.pkl"),
        help="Path to raw pkl. Supports flat list or [train,val,test].",
    )
    parser.add_argument(
        "--sample_index",
        type=int,
        default=0,
        help="Sample index after flattening raw pkl.",
    )
    parser.add_argument(
        "--patch_k",
        type=int,
        default=8,
        help="Protein patch K.",
    )
    return parser.parse_args()


def flatten_raw_data(obj: Any) -> List[Dict[str, Any]]:
    if isinstance(obj, list) and len(obj) == 3 and all(isinstance(x, list) for x in obj):
        return obj[0] + obj[1] + obj[2]

    if isinstance(obj, list):
        return obj

    raise TypeError(f"Unsupported raw pkl format: {type(obj)}")


def main() -> None:
    args = parse_args()

    if not args.raw_pkl.exists():
        raise FileNotFoundError(f"Raw pkl not found: {args.raw_pkl}")

    print(f"[Load raw] {args.raw_pkl}")

    raw_obj = load_pickle(args.raw_pkl)
    raw_samples = flatten_raw_data(raw_obj)

    print(f"[Raw samples] n={len(raw_samples)}")

    if len(raw_samples) == 0:
        raise RuntimeError("Raw pkl is empty.")

    if args.sample_index < 0 or args.sample_index >= len(raw_samples):
        raise IndexError(f"sample_index out of range: {args.sample_index}")

    raw = raw_samples[args.sample_index]

    print("\n[Raw sample keys]")
    for key in sorted(raw.keys()):
        if key in {"pdb", "kind", "protein_seq", "partner_seq", "na_source"}:
            print(f"  {key}: {raw[key]}")
        elif key in {"protein_coords", "partner_coords", "contact_map"}:
            value = raw[key]
            print(f"  {key}: shape={getattr(value, 'shape', None)}")

    required = ["protein_seq", "protein_coords", "partner_seq", "contact_map"]
    missing = [key for key in required if key not in raw]
    if missing:
        raise KeyError(f"Raw sample missing required keys: {missing}")

    print("\n[Process single sample]")

    processed = process_single_sample(
        raw_sample=raw,
        patch_k=args.patch_k,
        residue_motif_agg="topk_mean",
        patch_motif_agg="topk_mean",
        threshold=6.0,
        residue_top_k=3,
        patch_top_k=3,
        energy_tau=4.0,
        add_precomputed_attrs=True,
    )

    print("  protein length:", len(processed["protein_seq"]))
    print("  partner length:", len(processed["partner_seq"]))
    print("  secondary_structure:", processed["secondary_structure"])
    print("  n_motifs:", len(processed["motifs"]))
    print("  patch_indices:", processed["patch_indices"].shape)
    print("  patch_motif_label:", processed["patch_motif_label"].shape)
    print("  patch_attr:", processed["patch_attr"].shape)
    print("  motif_attr:", processed["motif_attr"].shape)

    if processed["patch_attr"].shape[1] != 56:
        raise RuntimeError(f"patch_attr should be 56D, got {processed['patch_attr'].shape}")

    if processed["motif_attr"].shape[1] != 27:
        raise RuntimeError(f"motif_attr should be 27D, got {processed['motif_attr'].shape}")

    print("\n[Build label v2]")

    processed_v2 = build_label_v2_for_sample(
        sample=processed,
        beta_nt=3.0,
        beta_patch=6.0,
        contact_distance_tau=4.0,
        stem_weight=1.5,
        loop_weight=1.1,
        unpaired_weight=0.8,
        pairing_weight_scale=0.5,
        gc_weight_scale=0.25,
        position_weight_mode="center_mild",
        position_weight_scale=0.25,
        label_tau=0.30,
    )

    print("  residue_to_motif_score_v2_raw:", processed_v2["residue_to_motif_score_v2_raw"].shape)
    print("  patch_motif_structure_energy_v2:", processed_v2["patch_motif_structure_energy_v2"].shape)
    print(
        "  label_v2 min/max:",
        float(processed_v2["patch_motif_structure_energy_v2"].min()),
        float(processed_v2["patch_motif_structure_energy_v2"].max()),
    )

    if processed_v2["patch_motif_structure_energy_v2"].shape != processed_v2["patch_motif_label"].shape:
        raise RuntimeError(
            "label_v2 shape mismatch: "
            f"{processed_v2['patch_motif_structure_energy_v2'].shape} vs "
            f"{processed_v2['patch_motif_label'].shape}"
        )

    print("\nSMOKE TEST PASSED: raw sample can be processed into final PIGRAM schema.")


if __name__ == "__main__":
    main()