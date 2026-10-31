# scripts/tests/smoke_processed_forward.py
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from pigram.data.loader import prepare_single_sample_tensors
from pigram.losses.interaction import compute_patch_motif_huber_loss
from pigram.models.coarse import DualHeadCoarseModel
from pigram.models.encoders import MotifEncoder, ProteinPatchEncoder
from pigram.models.fallback import TransformerFallbackHead
from pigram.models.grammar_head import DualHeadAsymmetricSigmaRBFAttributeGrammar
from pigram.models.heads import PatchImportanceHead
from pigram.utils.io import load_pickle


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Smoke test: load processed pkl, build model, run forward pass."
    )
    parser.add_argument(
        "--pkl",
        type=Path,
        default=Path("data/processed/train.pkl"),
        help="Path to processed pkl.",
    )
    parser.add_argument(
        "--sample_index",
        type=int,
        default=0,
        help="Sample index to test.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="cuda or cpu.",
    )
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--num_layers", type=int, default=1)
    parser.add_argument("--num_neg_rules", type=int, default=3)
    parser.add_argument("--num_pos_rules", type=int, default=4)
    parser.add_argument("--rule_dim", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.1)
    return parser.parse_args()


def build_test_model(args: argparse.Namespace) -> DualHeadCoarseModel:
    return DualHeadCoarseModel(
        protein_encoder_cls=ProteinPatchEncoder,
        motif_encoder_cls=MotifEncoder,
        grammar_head_cls=DualHeadAsymmetricSigmaRBFAttributeGrammar,
        patch_importance_head_cls=PatchImportanceHead,
        fallback_head_cls=TransformerFallbackHead,
        protein_vocab_size=32,
        rna_vocab_size=16,
        motif_type_vocab_size=8,
        hidden_dim=args.hidden_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        num_neg_rules=args.num_neg_rules,
        num_pos_rules=args.num_pos_rules,
        rule_dim=args.rule_dim,
        dropout=args.dropout,
        route_bias_init=-3.0,
        neg_sigma_min=0.10,
        neg_sigma_max=0.80,
        pos_sigma_min=1.20,
        pos_sigma_max=2.40,
        sigma_residual_clip=0.50,
    )


def main() -> None:
    args = parse_args()

    if not args.pkl.exists():
        raise FileNotFoundError(f"Processed pkl not found: {args.pkl}")

    device = torch.device(
        args.device
        if torch.cuda.is_available() and args.device.lower() == "cuda"
        else "cpu"
    )

    print(f"[Device] {device}")
    print(f"[Load] {args.pkl}")

    samples = load_pickle(args.pkl)
    print(f"[Samples] n={len(samples)}")

    if len(samples) == 0:
        raise RuntimeError("Processed pkl is empty.")

    if args.sample_index < 0 or args.sample_index >= len(samples):
        raise IndexError(f"sample_index out of range: {args.sample_index}")

    sample = samples[args.sample_index]

    print("\n[Sample keys]")
    for key in sorted(sample.keys()):
        if key in {
            "protein_seq",
            "partner_seq",
            "secondary_structure",
            "pdb",
            "kind",
            "na_source",
        }:
            print(f"  {key}: {sample[key]}")
        elif key in {"patch_attr", "motif_attr", "patch_motif_structure_energy_v2"}:
            value = sample[key]
            print(f"  {key}: shape={getattr(value, 'shape', None)}")

    required = [
        "patch_attr",
        "motif_attr",
        "patch_motif_structure_energy_v2",
        "patch_motif_label",
        "patch_importance",
    ]
    missing = [key for key in required if key not in sample]
    if missing:
        raise KeyError(f"Missing required processed fields: {missing}")

    if sample["patch_attr"].shape[1] != 56:
        raise RuntimeError(f"patch_attr should be 56D, got {sample['patch_attr'].shape}")

    if sample["motif_attr"].shape[1] != 27:
        raise RuntimeError(f"motif_attr should be 27D, got {sample['motif_attr'].shape}")

    batch = prepare_single_sample_tensors(sample, device=device)

    print("\n[Tensor shapes]")
    print("  protein_token_ids:", tuple(batch["protein_token_ids"].shape))
    print("  patch_indices:", tuple(batch["patch_indices"].shape))
    print("  motif_type_ids:", tuple(batch["motif_type_ids"].shape))
    print("  patch_attr:", tuple(batch["patch_attr"].shape))
    print("  motif_attr:", tuple(batch["motif_attr"].shape))
    print("  y_v2:", tuple(batch["patch_motif_structure_energy_v2"].shape))

    model = build_test_model(args).to(device)
    model.eval()

    with torch.no_grad():
        out = model.forward_stage1(
            protein_token_ids=batch["protein_token_ids"],
            patch_indices=batch["patch_indices"],
            patch_center_indices=batch["patch_center_indices"],
            patch_relative_distances=batch["patch_relative_distances"],
            motif_token_ids_list=batch["motif_token_ids_list"],
            motif_type_ids=batch["motif_type_ids"],
            motif_lengths=batch["motif_lengths"],
            motif_pairing_ratios=batch["motif_pairing_ratios"],
            motif_gc_contents=batch["motif_gc_contents"],
            motif_edge_side_ids=batch["motif_edge_side_ids"],
            motif_is_terminal=batch["motif_is_terminal"],
            motif_center_positions=batch["motif_center_positions"],
            motif_is_dna=batch["motif_is_dna"],
            patch_attr=batch.get("patch_attr", None),
            motif_attr=batch.get("motif_attr", None),
            schedule_progress=1.0,
        )

        logits = out["patch_motif_logits"]
        target = batch["patch_motif_structure_energy_v2"]

        loss = compute_patch_motif_huber_loss(
            logits=logits,
            targets=target,
            sigmoid_scale=2.0,
            delta=0.10,
        )

    print("\n[Forward output]")
    print("  patch_motif_logits:", tuple(logits.shape))
    print("  grammar_logits:", tuple(out["grammar_logits"].shape))
    print("  rule_contrib:", tuple(out["rule_contrib"].shape))
    print("  patch_importance_logits:", tuple(out["patch_importance_logits"].shape))
    print("  loss:", float(loss.item()))

    if logits.shape != target.shape:
        raise RuntimeError(f"Logit/target shape mismatch: {logits.shape} vs {target.shape}")

    print("\nSMOKE TEST PASSED: processed pkl loads and model forward works.")


if __name__ == "__main__":
    main()