# src/pigram/eval/main.py
# -*- coding: utf-8 -*-
from __future__ import annotations
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict
import torch
from pigram.train.pigram import evaluate, load_pigram_from_checkpoint, prepare_split
from pigram.utils.io import save_json


def cfg_to_args(cfg: Dict[str, Any]) -> SimpleNamespace:
    flat: Dict[str, Any] = {}
    for key, value in cfg.items():
        if isinstance(value, dict):
            flat.update(value)
        else:
            flat[key] = value
    return SimpleNamespace(**flat)


def run_main_eval(cfg: Dict[str, Any]) -> None:
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

    val_data = prepare_split(
        samples_path=args.val_pkl,
        anns_path=args.val_ann_pkl,
        device=device,
    )
    test_data = prepare_split(
        samples_path=args.test_pkl,
        anns_path=args.test_ann_pkl,
        device=device,
    )

    print(f"Val samples:  {len(val_data)}")
    print(f"Test samples: {len(test_data)}")

    val_metrics = evaluate(model, val_data, model_args)
    test_metrics = evaluate(model, test_data, model_args)

    summary = {
        "checkpoint": str(args.pigram_ckpt),
        "foundation_checkpoint": str(args.foundation_ckpt),
        "fine_rule_map": {str(k): int(v) for k, v in fine_rule_map.items()},
        "checkpoint_epoch": ckpt.get("epoch", None),
        "checkpoint_best_val_pearson": ckpt.get("best_val_pearson", None),
        "checkpoint_best_selection_score": ckpt.get("best_selection_score", None),
        "val": val_metrics,
        "test": test_metrics,
    }

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_json(summary, out_dir / "metrics.json")

    print("\n=== Validation metrics ===")
    print(json.dumps(val_metrics, indent=2))

    print("\n=== Test metrics ===")
    print(json.dumps(test_metrics, indent=2))

    print("\nSaved:")
    print(out_dir / "metrics.json")