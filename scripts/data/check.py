# scripts/data/check.py
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch


def bootstrap_src_path() -> Path:
    repo_root = Path(__file__).resolve().parents[2]
    src_dir = repo_root / "src"

    if not src_dir.exists():
        raise FileNotFoundError(f"src directory not found: {src_dir}")

    src_str = str(src_dir)
    if src_str in sys.path:
        sys.path.remove(src_str)
    sys.path.insert(0, src_str)

    return repo_root


REPO_ROOT = bootstrap_src_path()

from pigram.data.loader import prepare_single_sample_tensors
from pigram.utils.config import load_config
from pigram.utils.io import load_pickle, save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check processed PIGRAM data schema."
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to data config YAML.",
    )
    parser.add_argument(
        "--max_samples_per_split",
        type=int,
        default=-1,
        help="Maximum samples to check per split. Use -1 to check all.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device for tensor conversion smoke check. Default: cpu.",
    )
    return parser.parse_args()


def get_nested(cfg: Dict[str, Any], key: str, default: Any = None) -> Any:
    if key in cfg:
        return cfg[key]
    for section in ["paths", "data", "processed", "train", "eval"]:
        value = cfg.get(section, {})
        if isinstance(value, dict) and key in value:
            return value[key]
    return default


def infer_processed_paths(cfg: Dict[str, Any]) -> Dict[str, Path]:
    train_pkl = get_nested(cfg, "train_pkl", "data/processed/train.pkl")
    val_pkl = get_nested(cfg, "val_pkl", "data/processed/val.pkl")
    test_pkl = get_nested(cfg, "test_pkl", "data/processed/test.pkl")

    return {
        "train": Path(train_pkl),
        "val": Path(val_pkl),
        "test": Path(test_pkl),
    }


def infer_output_path(cfg: Dict[str, Any]) -> Path:
    out_path = get_nested(cfg, "schema_check_json", None)
    if out_path is not None:
        return Path(out_path)

    processed_dir = get_nested(cfg, "processed_dir", "data/processed")
    return Path(processed_dir) / "schema_check.json"


def shape_of(x: Any) -> List[int] | None:
    if x is None:
        return None
    if torch.is_tensor(x):
        return list(x.shape)
    if hasattr(x, "shape"):
        return list(x.shape)
    try:
        return list(np.asarray(x).shape)
    except Exception:
        return None


def required_raw_or_processed_keys() -> List[str]:
    return [
        "protein_seq",
        "partner_seq",
        "patch_indices",
        "patch_motif_label",
        "patch_importance",
        "patch_motif_structure_energy_v2",
        "patch_attr",
        "motif_attr",
    ]


def check_sample_schema(
    sample: Dict[str, Any],
    split: str,
    sample_index: int,
    device: torch.device,
) -> Dict[str, Any]:
    errors: List[str] = []
    warnings: List[str] = []

    required = required_raw_or_processed_keys()
    missing = [key for key in required if key not in sample]
    if missing:
        errors.append(f"missing required keys: {missing}")

    patch_attr_shape = shape_of(sample.get("patch_attr"))
    motif_attr_shape = shape_of(sample.get("motif_attr"))
    y_shape = shape_of(sample.get("patch_motif_structure_energy_v2"))
    label_shape = shape_of(sample.get("patch_motif_label"))
    patch_importance_shape = shape_of(sample.get("patch_importance"))

    patch_attr_dim = None
    motif_attr_dim = None

    if patch_attr_shape is not None and len(patch_attr_shape) == 2:
        patch_attr_dim = int(patch_attr_shape[1])
        if patch_attr_dim != 56:
            errors.append(f"patch_attr dim should be 56, got {patch_attr_shape}")
    else:
        errors.append(f"patch_attr should be 2D, got {patch_attr_shape}")

    if motif_attr_shape is not None and len(motif_attr_shape) == 2:
        motif_attr_dim = int(motif_attr_shape[1])
        if motif_attr_dim != 27:
            errors.append(f"motif_attr dim should be 27, got {motif_attr_shape}")
    else:
        errors.append(f"motif_attr should be 2D, got {motif_attr_shape}")

    if y_shape is None or len(y_shape) != 2:
        errors.append(f"patch_motif_structure_energy_v2 should be 2D, got {y_shape}")

    if label_shape is None or len(label_shape) != 2:
        errors.append(f"patch_motif_label should be 2D, got {label_shape}")

    if y_shape is not None and label_shape is not None and y_shape != label_shape:
        errors.append(
            f"patch_motif_structure_energy_v2 shape {y_shape} != "
            f"patch_motif_label shape {label_shape}"
        )

    if patch_attr_shape is not None and y_shape is not None:
        if len(patch_attr_shape) == 2 and len(y_shape) == 2:
            if int(patch_attr_shape[0]) != int(y_shape[0]):
                errors.append(
                    f"num_patches mismatch: patch_attr={patch_attr_shape}, y={y_shape}"
                )

    if motif_attr_shape is not None and y_shape is not None:
        if len(motif_attr_shape) == 2 and len(y_shape) == 2:
            if int(motif_attr_shape[0]) != int(y_shape[1]):
                errors.append(
                    f"num_motifs mismatch: motif_attr={motif_attr_shape}, y={y_shape}"
                )

    if patch_importance_shape is not None and y_shape is not None:
        if len(patch_importance_shape) == 1 and len(y_shape) == 2:
            if int(patch_importance_shape[0]) != int(y_shape[0]):
                errors.append(
                    f"patch_importance length mismatch: "
                    f"patch_importance={patch_importance_shape}, y={y_shape}"
                )

    loader_ok = False
    batch_shapes: Dict[str, Any] = {}

    if not errors:
        try:
            batch = prepare_single_sample_tensors(sample, device=device)
            loader_ok = True
            for key in [
                "protein_token_ids",
                "patch_indices",
                "patch_center_indices",
                "patch_relative_distances",
                "motif_token_ids_list",
                "motif_type_ids",
                "motif_lengths",
                "motif_pairing_ratios",
                "motif_gc_contents",
                "motif_edge_side_ids",
                "motif_is_terminal",
                "motif_center_positions",
                "motif_is_dna",
                "patch_attr",
                "motif_attr",
                "patch_motif_label",
                "patch_importance",
                "patch_motif_structure_energy_v2",
            ]:
                if key not in batch:
                    warnings.append(f"loader output missing key: {key}")
                    continue

                if key == "motif_token_ids_list":
                    batch_shapes[key] = [list(x.shape) for x in batch[key]]
                else:
                    batch_shapes[key] = list(batch[key].shape)
        except Exception as exc:
            errors.append(f"loader conversion failed: {type(exc).__name__}: {exc}")

    return {
        "split": split,
        "sample_index": int(sample_index),
        "pdb": str(sample.get("pdb", "")),
        "kind": str(sample.get("kind", "")),
        "na_source": str(sample.get("na_source", "")),
        "patch_attr_shape": patch_attr_shape,
        "motif_attr_shape": motif_attr_shape,
        "patch_motif_structure_energy_v2_shape": y_shape,
        "patch_motif_label_shape": label_shape,
        "patch_importance_shape": patch_importance_shape,
        "patch_attr_dim": patch_attr_dim,
        "motif_attr_dim": motif_attr_dim,
        "loader_ok": bool(loader_ok),
        "batch_shapes": batch_shapes,
        "warnings": warnings,
        "errors": errors,
    }


def check_split(
    split: str,
    pkl_path: Path,
    max_samples: int,
    device: torch.device,
) -> Dict[str, Any]:
    if not pkl_path.exists():
        return {
            "split": split,
            "path": str(pkl_path),
            "exists": False,
            "status": "error",
            "errors": [f"file not found: {pkl_path}"],
        }

    samples = load_pickle(pkl_path)

    if not isinstance(samples, list):
        return {
            "split": split,
            "path": str(pkl_path),
            "exists": True,
            "status": "error",
            "errors": [f"expected list of samples, got {type(samples)}"],
        }

    n_total = len(samples)
    if max_samples is not None and max_samples > 0:
        samples_to_check = samples[: max_samples]
    else:
        samples_to_check = samples

    sample_reports = []
    num_errors = 0
    num_warnings = 0
    patch_attr_dims = []
    motif_attr_dims = []
    num_pairs = 0

    for i, sample in enumerate(samples_to_check):
        report = check_sample_schema(
            sample=sample,
            split=split,
            sample_index=i,
            device=device,
        )
        sample_reports.append(report)

        if report["errors"]:
            num_errors += 1
        if report["warnings"]:
            num_warnings += 1

        if report["patch_attr_dim"] is not None:
            patch_attr_dims.append(report["patch_attr_dim"])
        if report["motif_attr_dim"] is not None:
            motif_attr_dims.append(report["motif_attr_dim"])

        y_shape = report["patch_motif_structure_energy_v2_shape"]
        if y_shape is not None and len(y_shape) == 2:
            num_pairs += int(y_shape[0]) * int(y_shape[1])

    status = "ok" if num_errors == 0 else "error"

    return {
        "split": split,
        "path": str(pkl_path),
        "exists": True,
        "status": status,
        "num_samples_total": int(n_total),
        "num_samples_checked": int(len(samples_to_check)),
        "num_samples_with_errors": int(num_errors),
        "num_samples_with_warnings": int(num_warnings),
        "patch_attr_dims_seen": sorted(set(int(x) for x in patch_attr_dims)),
        "motif_attr_dims_seen": sorted(set(int(x) for x in motif_attr_dims)),
        "num_pairs_checked": int(num_pairs),
        "sample_reports": sample_reports[:20],
    }


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    device = torch.device(
        args.device
        if torch.cuda.is_available() and args.device.lower() == "cuda"
        else "cpu"
    )

    processed_paths = infer_processed_paths(cfg)
    out_path = infer_output_path(cfg)

    print(f"[Device] {device}")
    print("[Processed paths]")
    for split, path in processed_paths.items():
        print(f"  {split}: {path}")

    split_reports = {}
    for split, path in processed_paths.items():
        print(f"\n[Check] {split}: {path}")
        split_report = check_split(
            split=split,
            pkl_path=path,
            max_samples=args.max_samples_per_split,
            device=device,
        )
        split_reports[split] = split_report
        print(
            f"  status={split_report['status']} "
            f"n_total={split_report.get('num_samples_total', 0)} "
            f"n_checked={split_report.get('num_samples_checked', 0)} "
            f"errors={split_report.get('num_samples_with_errors', 0)}"
        )

    all_status_ok = all(report.get("status") == "ok" for report in split_reports.values())

    patch_dims = set()
    motif_dims = set()

    for report in split_reports.values():
        for dim in report.get("patch_attr_dims_seen", []):
            patch_dims.add(int(dim))
        for dim in report.get("motif_attr_dims_seen", []):
            motif_dims.add(int(dim))

    summary = {
        "status": "ok" if all_status_ok else "error",
        "patch_attr_dim": 56 if patch_dims == {56} else sorted(patch_dims),
        "motif_attr_dim": 27 if motif_dims == {27} else sorted(motif_dims),
        "expected_patch_attr_dim": 56,
        "expected_motif_attr_dim": 27,
        "config": str(args.config),
        "max_samples_per_split": int(args.max_samples_per_split),
        "device": str(device),
        "splits": split_reports,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_json(summary, out_path)

    print("\n=== Schema validation summary ===")
    print(json.dumps(
        {
            "status": summary["status"],
            "patch_attr_dim": summary["patch_attr_dim"],
            "motif_attr_dim": summary["motif_attr_dim"],
            "output": str(out_path),
        },
        indent=2,
    ))

    if summary["status"] != "ok":
        raise RuntimeError(f"Schema validation failed. See {out_path}")

    print("\nSchema validation passed.")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()