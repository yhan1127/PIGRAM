# src/pigram/data/split.py
# -*- coding: utf-8 -*-

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Dict, List, Tuple

import random

from pigram.utils.io import load_pickle, save_json, save_pickle


def stable_hash(text: str | None) -> str:
    if text is None:
        text = ""

    return hashlib.md5(text.encode("utf-8")).hexdigest()


def pair_key(sample: Dict[str, Any]) -> str:
    protein_seq = str(sample.get("protein_seq", ""))
    partner_seq = str(sample.get("partner_seq", ""))

    return stable_hash(protein_seq + "||" + partner_seq)


def split_by_unique_pairs(
    samples: List[Dict[str, Any]],
    train_ratio: float = 0.87,
    val_ratio: float = 0.065,
    test_ratio: float = 0.065,
    seed: int = 42,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    pair_to_samples: Dict[str, List[Dict[str, Any]]] = {}

    for sample in samples:
        key = pair_key(sample)
        pair_to_samples.setdefault(key, []).append(sample)

    unique_pairs = list(pair_to_samples.keys())

    rng = random.Random(seed)
    rng.shuffle(unique_pairs)

    n_total = len(unique_pairs)
    n_train = int(round(n_total * train_ratio))
    n_val = int(round(n_total * val_ratio))

    if n_train + n_val > n_total:
        n_val = max(0, n_total - n_train)

    train_pairs = set(unique_pairs[:n_train])
    val_pairs = set(unique_pairs[n_train:n_train + n_val])
    test_pairs = set(unique_pairs[n_train + n_val:])

    train_split = []
    val_split = []
    test_split = []

    for key, grouped_samples in pair_to_samples.items():
        if key in train_pairs:
            train_split.extend(grouped_samples)
        elif key in val_pairs:
            val_split.extend(grouped_samples)
        else:
            test_split.extend(grouped_samples)

    summary = {
        "num_unique_pairs_total": n_total,
        "num_unique_pairs_train": len(train_pairs),
        "num_unique_pairs_val": len(val_pairs),
        "num_unique_pairs_test": len(test_pairs),
        "num_samples_train": len(train_split),
        "num_samples_val": len(val_split),
        "num_samples_test": len(test_split),
        "train_ratio": float(train_ratio),
        "val_ratio": float(val_ratio),
        "test_ratio": float(test_ratio),
        "seed": int(seed),
    }

    return train_split, val_split, test_split, summary


def count_overlap_pairs(
    split_a: List[Dict[str, Any]],
    split_b: List[Dict[str, Any]],
) -> int:
    pairs_a = {pair_key(sample) for sample in split_a}
    pairs_b = {pair_key(sample) for sample in split_b}

    return len(pairs_a & pairs_b)


def flatten_raw_data(data: Any) -> List[Dict[str, Any]]:
    # Supports either [train, val, test] or a flat list.
    if (
        isinstance(data, list)
        and len(data) == 3
        and all(isinstance(item, list) for item in data)
    ):
        return data[0] + data[1] + data[2]

    if isinstance(data, list):
        return data

    raise TypeError(
        "Expected raw data to be either a flat list or [train, val, test]. "
        f"Got: {type(data)}"
    )


def attach_source(
    samples: List[Dict[str, Any]],
    na_source: str,
) -> List[Dict[str, Any]]:
    tagged = []

    for sample in samples:
        item = dict(sample)
        item["na_source"] = na_source
        tagged.append(item)

    return tagged


def process_one_source(
    in_path: str | Path,
    out_path: str | Path,
    na_source: str,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> Dict[str, Any]:
    in_path = Path(in_path)
    out_path = Path(out_path)

    data = load_pickle(in_path)
    all_samples = flatten_raw_data(data)
    tagged_samples = attach_source(all_samples, na_source)

    train_split, val_split, test_split, split_summary = split_by_unique_pairs(
        tagged_samples,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=seed,
    )

    overlap_summary = {
        "train_val_pair_overlap": count_overlap_pairs(train_split, val_split),
        "train_test_pair_overlap": count_overlap_pairs(train_split, test_split),
        "val_test_pair_overlap": count_overlap_pairs(val_split, test_split),
    }

    out_obj = [train_split, val_split, test_split]
    save_pickle(out_obj, out_path)

    meta = {
        "input_path": str(in_path),
        "output_path": str(out_path),
        "na_source": na_source,
        "split_summary": split_summary,
        "overlap_summary": overlap_summary,
    }

    meta_path = out_path.with_name(out_path.stem + "_summary.json")
    save_json(meta, meta_path)

    print(f"\n=== {na_source.upper()} pair-disjoint split ===")
    print(meta)

    return meta


def build_pairdisjoint_splits(cfg: Dict[str, Any]) -> None:
    seed = int(cfg.get("seed", 42))
    split_cfg = cfg["split"]

    train_ratio = float(split_cfg.get("train_ratio", 0.87))
    val_ratio = float(split_cfg.get("val_ratio", 0.065))
    test_ratio = float(split_cfg.get("test_ratio", 0.065))

    rna_meta = process_one_source(
        in_path=split_cfg["rna_in"],
        out_path=split_cfg["rna_out"],
        na_source="rna",
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=seed,
    )

    dna_meta = process_one_source(
        in_path=split_cfg["dna_in"],
        out_path=split_cfg["dna_out"],
        na_source="dna",
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=seed,
    )

    print("\nSaved pair-disjoint split files:")
    print(rna_meta["output_path"])
    print(dna_meta["output_path"])