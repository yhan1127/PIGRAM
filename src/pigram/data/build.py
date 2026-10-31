# src/pigram/data/build.py
# -*- coding: utf-8 -*-

from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Dict, List, Tuple

from pigram.data.dataset import process_split, summarize_processed_split
from pigram.data.label import recompute_v2_labels_for_processed_split, validate_label_v2_shapes
from pigram.utils.io import load_pickle, save_json, save_pickle


def attach_source(
    split: List[Dict[str, Any]],
    na_source: str,
) -> List[Dict[str, Any]]:
    """
    Attach nucleic-acid source metadata to all samples in a split.
    """
    out = []

    for sample in split:
        item = dict(sample)
        item["na_source"] = str(na_source).lower()
        out.append(item)

    return out


def load_pairdisjoint_source(
    path: str | Path,
    na_source: str,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Load one pair-disjoint source file.

    Expected format:
        [train_split, val_split, test_split]
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Pair-disjoint split file not found: {path}")

    obj = load_pickle(path)

    if not isinstance(obj, list) or len(obj) != 3:
        raise ValueError(
            f"Expected [train, val, test] from {path}, got type={type(obj)}"
        )

    train, val, test = obj

    if not isinstance(train, list) or not isinstance(val, list) or not isinstance(test, list):
        raise TypeError(
            f"Expected all three splits to be lists in {path}."
        )

    train = attach_source(train, na_source)
    val = attach_source(val, na_source)
    test = attach_source(test, na_source)

    return train, val, test


def merge_rna_dna_splits(
    rna_splits: Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]],
    dna_splits: Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]],
    seed: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Merge RNA and DNA train/val/test splits and shuffle each split.
    """
    rna_train, rna_val, rna_test = rna_splits
    dna_train, dna_val, dna_test = dna_splits

    train = rna_train + dna_train
    val = rna_val + dna_val
    test = rna_test + dna_test

    rng = random.Random(seed)
    rng.shuffle(train)
    rng.shuffle(val)
    rng.shuffle(test)

    return train, val, test


def process_raw_split(
    raw_split: List[Dict[str, Any]],
    patch_cfg: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """
    Build base processed patch--motif samples.

    This includes:
        protein patches
        ssNA motifs
        binary patch_motif_label
        patch_importance
        patch_motif_energy
        patch_motif_structure_energy
        precomputed patch_attr/motif_attr if dataset.process_single_sample enables it
    """
    return process_split(
        raw_split=raw_split,
        patch_k=int(patch_cfg.get("patch_k", 8)),
        residue_motif_agg=str(patch_cfg.get("residue_motif_agg", "topk_mean")),
        patch_motif_agg=str(patch_cfg.get("patch_motif_agg", "topk_mean")),
        threshold=float(patch_cfg.get("threshold", 6.0)),
        residue_top_k=int(patch_cfg.get("residue_top_k", 3)),
        patch_top_k=int(patch_cfg.get("patch_top_k", 3)),
        energy_tau=float(patch_cfg.get("energy_tau", 4.0)),
        add_precomputed_attrs=bool(patch_cfg.get("add_precomputed_attrs", True)),
        verbose=bool(patch_cfg.get("verbose", True)),
    )


def save_split_and_summary(
    split: List[Dict[str, Any]],
    out_path: str | Path,
    split_name: str,
) -> Dict[str, Any]:
    """
    Save processed split and return its summary.
    """
    out_path = Path(out_path)
    save_pickle(split, out_path)

    summary = summarize_processed_split(split)
    summary["split"] = split_name
    summary["path"] = str(out_path)

    return summary


def build_processed_dataset(cfg: Dict[str, Any]) -> None:
    """
    Build final processed train/val/test datasets.

    Pipeline:
        data/split/prot_rna.pkl + data/split/prot_dna.pkl
        → merged raw train/val/test
        → patch--motif processed samples
        → recompute label v2
        → save data/processed/{train,val,test}.pkl
    """
    seed = int(cfg.get("seed", 42))
    random.seed(seed)

    build_cfg = cfg["build"]
    patch_cfg = cfg["patch"]
    label_cfg = cfg["label"]

    print("Loading pair-disjoint RNA splits...")
    rna_splits = load_pairdisjoint_source(
        path=build_cfg["rna_split"],
        na_source="rna",
    )

    print("Loading pair-disjoint DNA splits...")
    dna_splits = load_pairdisjoint_source(
        path=build_cfg["dna_split"],
        na_source="dna",
    )

    merged_train, merged_val, merged_test = merge_rna_dna_splits(
        rna_splits=rna_splits,
        dna_splits=dna_splits,
        seed=seed,
    )

    print("\nMerged raw split sizes:")
    print(f"train: {len(merged_train)}")
    print(f"val:   {len(merged_val)}")
    print(f"test:  {len(merged_test)}")

    print("\nStage 1: build base patch--motif processed dataset...")
    processed_train = process_raw_split(merged_train, patch_cfg)
    processed_val = process_raw_split(merged_val, patch_cfg)
    processed_test = process_raw_split(merged_test, patch_cfg)

    print("\nStage 2: recompute structure-aware label v2...")
    processed_train = recompute_v2_labels_for_processed_split(
        processed_train,
        label_cfg,
    )
    processed_val = recompute_v2_labels_for_processed_split(
        processed_val,
        label_cfg,
    )
    processed_test = recompute_v2_labels_for_processed_split(
        processed_test,
        label_cfg,
    )

    print("\nStage 3: validate label v2 shapes...")
    label_summaries = [
        validate_label_v2_shapes(processed_train, "train"),
        validate_label_v2_shapes(processed_val, "val"),
        validate_label_v2_shapes(processed_test, "test"),
    ]

    failed = [x for x in label_summaries if x["status"] != "ok"]
    if failed:
        raise RuntimeError(f"Label v2 validation failed: {failed}")

    print("\nStage 4: save processed datasets...")
    train_summary = save_split_and_summary(
        split=processed_train,
        out_path=build_cfg["train_out"],
        split_name="train",
    )
    val_summary = save_split_and_summary(
        split=processed_val,
        out_path=build_cfg["val_out"],
        split_name="val",
    )
    test_summary = save_split_and_summary(
        split=processed_test,
        out_path=build_cfg["test_out"],
        split_name="test",
    )

    meta = {
        "seed": seed,
        "rna_split": build_cfg["rna_split"],
        "dna_split": build_cfg["dna_split"],
        "outputs": {
            "train": build_cfg["train_out"],
            "val": build_cfg["val_out"],
            "test": build_cfg["test_out"],
        },
        "patch": dict(patch_cfg),
        "label": dict(label_cfg),
        "label_validation": label_summaries,
        "summary": {
            "train": train_summary,
            "val": val_summary,
            "test": test_summary,
        },
    }

    save_json(meta, build_cfg["summary_out"])

    print("\nSaved processed datasets:")
    print(build_cfg["train_out"])
    print(build_cfg["val_out"])
    print(build_cfg["test_out"])
    print(build_cfg["summary_out"])