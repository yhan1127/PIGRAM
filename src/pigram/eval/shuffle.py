# src/pigram/eval/shuffle.py
# -*- coding: utf-8 -*-
from __future__ import annotations
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List
import numpy as np
import torch
from pigram.eval.metrics import safe_auprc, safe_auroc, safe_pearson, safe_spearman
from pigram.train.pigram import load_pigram_from_checkpoint, prepare_split
from pigram.utils.io import save_json


def cfg_to_args(cfg: Dict[str, Any]) -> SimpleNamespace:
    flat: Dict[str, Any] = {}
    for key, value in cfg.items():
        if isinstance(value, dict):
            flat.update(value)
        else:
            flat[key] = value
    return SimpleNamespace(**flat)


def clone_batch_with_shuffled_attrs(
    batch: Dict[str, Any],
    mode: str,
    generator: torch.Generator,
) -> Dict[str, Any]:
    if mode not in {"original", "patch", "motif", "both"}:
        raise ValueError(f"Unsupported shuffle mode: {mode}")

    out = dict(batch)

    if mode in {"patch", "both"} and "patch_attr" in batch:
        patch_attr = batch["patch_attr"]
        perm = torch.randperm(
            patch_attr.shape[0],
            generator=generator,
            device=patch_attr.device,
        )
        out["patch_attr"] = patch_attr[perm]

    if mode in {"motif", "both"} and "motif_attr" in batch:
        motif_attr = batch["motif_attr"]
        perm = torch.randperm(
            motif_attr.shape[0],
            generator=generator,
            device=motif_attr.device,
        )
        out["motif_attr"] = motif_attr[perm]

    return out


@torch.no_grad()
def evaluate_shuffle_mode(
    model,
    prepared_data: List[Dict[str, Any]],
    model_args: SimpleNamespace,
    mode: str,
    seed: int,
) -> Dict[str, float]:
    model.eval()

    if torch.cuda.is_available():
        generator = torch.Generator(device="cuda")
    else:
        generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))

    arrays = {
        "y": [],
        "ybin_struct": [],
        "final_logits": [],
        "head_a_logits": [],
        "grammar_logits": [],
        "route": [],
        "delta": [],
        "correction": [],
    }

    for batch in prepared_data:
        eval_batch = clone_batch_with_shuffled_attrs(
            batch=batch,
            mode=mode,
            generator=generator,
        )

        out = model(eval_batch, sigmoid_scale=model_args.sigmoid_scale)

        y = (
            batch["patch_motif_structure_energy_v2"]
            .detach()
            .cpu()
            .numpy()
            .reshape(-1)
        )
        ybin_struct = (y >= model_args.binary_threshold).astype(np.float32)

        arrays["y"].append(y)
        arrays["ybin_struct"].append(ybin_struct)
        arrays["final_logits"].append(
            out["patch_motif_logits"].detach().cpu().numpy().reshape(-1)
        )
        arrays["head_a_logits"].append(
            out["head_a_logits"].detach().cpu().numpy().reshape(-1)
        )
        arrays["grammar_logits"].append(
            out["grammar_logits"].detach().cpu().numpy().reshape(-1)
        )
        arrays["route"].append(
            out["route_r"].detach().cpu().numpy().reshape(-1)
        )
        arrays["delta"].append(
            out["delta_logits"].detach().cpu().numpy().reshape(-1)
        )
        arrays["correction"].append(
            out["correction_logits"].detach().cpu().numpy().reshape(-1)
        )

    arr = {
        key: np.concatenate(value, axis=0)
        for key, value in arrays.items()
    }

    return {
        "mode": mode,
        "energy_pearson": safe_pearson(arr["y"], arr["final_logits"]),
        "energy_spearman": safe_spearman(arr["y"], arr["final_logits"]),
        "struct_auroc": safe_auroc(arr["ybin_struct"], arr["final_logits"]),
        "struct_auprc": safe_auprc(arr["ybin_struct"], arr["final_logits"]),
        "head_a_only_pearson": safe_pearson(arr["y"], arr["head_a_logits"]),
        "grammar_only_pearson": safe_pearson(arr["y"], arr["grammar_logits"]),
        "route_r_mean": float(arr["route"].mean()),
        "route_r_p95": float(np.quantile(arr["route"], 0.95)),
        "delta_abs_mean": float(np.abs(arr["delta"]).mean()),
        "correction_abs_mean": float(np.abs(arr["correction"]).mean()),
    }


def run_shuffle_eval(cfg: Dict[str, Any]) -> None:
    args = cfg_to_args(cfg)

    device = torch.device(
        args.device
        if torch.cuda.is_available() and str(args.device).lower() == "cuda"
        else "cpu"
    )
    print(f"Using device: {device}")

    model, model_args, ckpt, fine_rule_map = load_pigram_from_checkpoint(
        checkpoint_path=args.pigram_ckpt,
        eval_args=args,
        device=device,
    )

    split = str(getattr(args, "split", "test")).lower()

    if split == "val":
        data = prepare_split(
            samples_path=args.val_pkl,
            anns_path=args.val_ann_pkl,
            device=device,
        )
    elif split == "test":
        data = prepare_split(
            samples_path=args.test_pkl,
            anns_path=args.test_ann_pkl,
            device=device,
        )
    else:
        raise ValueError(f"Unsupported split: {split}")

    modes = list(getattr(args, "modes", ["original", "patch", "motif", "both"]))

    results = {}

    for mode in modes:
        print(f"\nEvaluating shuffle mode: {mode}")
        metrics = evaluate_shuffle_mode(
            model=model,
            prepared_data=data,
            model_args=model_args,
            mode=mode,
            seed=int(args.seed),
        )
        results[mode] = metrics
        print(json.dumps(metrics, indent=2))

    summary = {
        "checkpoint": str(args.pigram_ckpt),
        "foundation_checkpoint": str(args.foundation_ckpt),
        "split": split,
        "fine_rule_map": {str(k): int(v) for k, v in fine_rule_map.items()},
        "note": (
            "struct_auroc and struct_auprc use "
            "patch_motif_structure_energy_v2 >= binary_threshold as the binary target. "
            "They are not directly comparable to the main benchmark binary AUPRC."
        ),
        "metrics": results,
    }

    out_dir = Path(args.shuffle_out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    save_json(summary, out_dir / "attribute_shuffle.json")

    print("\nSaved:")
    print(out_dir / "attribute_shuffle.json")