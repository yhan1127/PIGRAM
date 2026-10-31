# src/pigram/data/downstream.py
# -*- coding: utf-8 -*-

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from pigram.data.dataset import process_single_sample
from pigram.utils.io import load_pickle, save_json, save_pickle


def clean_seq(seq: Any) -> str:
    if seq is None:
        return ""

    seq_str = str(seq).strip().upper()
    seq_str = "".join(seq_str.split())
    seq_str = seq_str.replace("-", "")

    return seq_str


def infer_na_source(seq: str) -> str:
    seq = clean_seq(seq)

    has_t = "T" in seq
    has_u = "U" in seq

    if has_t and not has_u:
        return "dna"

    return "rna"


def safe_float(x: Any) -> float:
    try:
        value = float(x)

        if not np.isfinite(value):
            return float("nan")

        return value

    except Exception:
        return float("nan")


def coords_shape(x: Any) -> str:
    try:
        arr = np.asarray(x)
        return "x".join(map(str, arr.shape))
    except Exception:
        return ""


def target_from_path(path: Path, targets: List[str]) -> str:
    stem = path.stem.upper().replace("_EVAL", "")

    for target in targets:
        if target.upper() in stem:
            return target.upper()

    return stem


def make_dummy_contact_map(
    protein_seq: str,
    partner_seq: str,
    dummy_distance: float,
) -> np.ndarray:
    return np.full(
        (len(protein_seq), len(partner_seq)),
        fill_value=float(dummy_distance),
        dtype=np.float32,
    )


def add_v2_dummy_fields(sample: Dict[str, Any]) -> Dict[str, Any]:
    """
    The final PIGRAM rescue model expects patch_motif_structure_energy_v2
    in some forward/eval code paths.

    For downstream screening, this field is a compatibility placeholder.
    It is NOT used as the downstream screening label.
    """
    out = dict(sample)

    n_res = len(out["protein_seq"])
    n_patch = int(out["patch_indices"].shape[0])
    n_motif = int(len(out["motifs"]))

    out["residue_to_motif_score_v2_raw"] = np.zeros(
        (n_res, n_motif),
        dtype=np.float32,
    )
    out["patch_motif_structure_energy_v2"] = np.zeros(
        (n_patch, n_motif),
        dtype=np.float32,
    )

    return out


def process_one_candidate(
    target: str,
    protein_info: Dict[str, Any],
    candidate: Dict[str, Any],
    split_name: str,
    split_index: int,
    local_index: int,
    patch_k: int,
    dummy_distance: float,
    keep_coords: bool,
) -> Dict[str, Any]:
    """
    Convert one FAFormer downstream candidate into a PIGRAM-compatible sample.

    Important:
        The dummy contact_map and dummy label-v2 are only used to create a
        schema-compatible sample. They are not downstream labels.
    """
    protein_seq = str(protein_info["protein_seq"])
    protein_coords = np.asarray(protein_info["protein_coords"], dtype=np.float32)

    raw_partner_seq = clean_seq(candidate["partner_seq"])
    partner_seq = raw_partner_seq
    partner_coords = np.asarray(candidate.get("partner_coords", []), dtype=np.float32)

    aptamer_id = str(candidate.get("pdb", f"{target}_{split_index}_{local_index}"))
    affinity = safe_float(candidate.get("affinity", np.nan))
    na_source = infer_na_source(partner_seq)

    raw_sample: Dict[str, Any] = {
        "pdb": aptamer_id,
        "target": target,
        "kind": f"prot-{na_source}",
        "na_source": na_source,

        "protein_seq": protein_seq,
        "protein_coords": protein_coords,

        "partner_seq": partner_seq,

        # Downstream scoring should not use true contact labels.
        # This dummy map only lets process_single_sample create the expected schema.
        "contact_map": make_dummy_contact_map(
            protein_seq=protein_seq,
            partner_seq=partner_seq,
            dummy_distance=dummy_distance,
        ),

        "affinity": affinity,
    }

    if keep_coords and partner_coords.size > 0:
        raw_sample["partner_coords"] = partner_coords

    processed = process_single_sample(
        raw_sample=raw_sample,
        patch_k=patch_k,
        residue_motif_agg="topk_mean",
        patch_motif_agg="topk_mean",
        threshold=6.0,
        residue_top_k=3,
        patch_top_k=3,
        energy_tau=4.0,
        add_precomputed_attrs=True,
    )

    processed = add_v2_dummy_fields(processed)

    # Preserve downstream metadata.
    processed["target"] = target
    processed["aptamer_id"] = aptamer_id
    processed["source_split"] = split_name
    processed["split_index"] = int(split_index)
    processed["local_index"] = int(local_index)
    processed["affinity"] = affinity
    processed["raw_partner_seq"] = raw_partner_seq

    if keep_coords and partner_coords.size > 0:
        processed["protein_coords"] = protein_coords
        processed["partner_coords"] = partner_coords

    return processed


def attach_label_modes(
    samples: List[Dict[str, Any]],
    target: str,
    expected_positives: Dict[str, int],
) -> List[Dict[str, Any]]:
    """
    Keep both affinity directions because the downstream direction is selected
    on validation:

        label_high_topN: larger affinity values treated as positives
        label_low_topN: smaller affinity values treated as positives
    """
    n_positive = expected_positives.get(target, None)

    for sample in samples:
        sample["label_high_topN"] = np.nan
        sample["label_low_topN"] = np.nan

    if n_positive is None or n_positive <= 0 or len(samples) == 0:
        return samples

    valid = [
        (idx, safe_float(sample.get("affinity", np.nan)))
        for idx, sample in enumerate(samples)
    ]
    valid = [
        (idx, affinity)
        for idx, affinity in valid
        if np.isfinite(affinity)
    ]

    if len(valid) < n_positive:
        print(
            f"[Warning] {target}: valid affinity count {len(valid)} "
            f"< expected positives {n_positive}"
        )
        return samples

    high_pos = {
        idx
        for idx, _ in sorted(valid, key=lambda x: x[1], reverse=True)[:n_positive]
    }
    low_pos = {
        idx
        for idx, _ in sorted(valid, key=lambda x: x[1], reverse=False)[:n_positive]
    }

    for idx, sample in enumerate(samples):
        sample["label_high_topN"] = int(idx in high_pos)
        sample["label_low_topN"] = int(idx in low_pos)
        sample["expected_positive_count"] = int(n_positive)

    return samples


def load_target_raw(
    pkl_path: Path,
    targets: List[str],
) -> Tuple[str, Dict[str, Any], Dict[str, List[Dict[str, Any]]]]:
    """
    Expected FAFormer downstream pkl format:

        root[0] = protein dict
        root[1] = validation candidate list
        root[2] = test candidate list

    The original split is preserved.
    """
    target = target_from_path(pkl_path, targets)
    obj = load_pickle(pkl_path)

    if not isinstance(obj, list) or len(obj) < 3:
        raise ValueError(
            f"{pkl_path.name}: expected list [protein, val, test], "
            f"got {type(obj).__name__}"
        )

    protein_info = obj[0]

    if not isinstance(protein_info, dict):
        raise ValueError(f"{pkl_path.name}: root[0] should be protein dict.")

    splits: Dict[str, List[Dict[str, Any]]] = {}

    for split_index in range(1, len(obj)):
        if split_index == 1:
            split_name = "val"
        elif split_index == 2:
            split_name = "test"
        else:
            split_name = f"split{split_index}"

        split_obj = obj[split_index]

        if not isinstance(split_obj, list):
            print(f"[Warning] {target}: root[{split_index}] is not list, skip")
            continue

        splits[split_name] = split_obj

    return target, protein_info, splits


def process_target(
    pkl_path: Path,
    out_dir: Path,
    targets: List[str],
    expected_positives: Dict[str, int],
    patch_k: int,
    max_per_split: int,
    dummy_distance: float,
    keep_coords: bool,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    target, protein_info, splits = load_target_raw(pkl_path, targets)

    print(f"\n================ Process {target} ================")
    print("[Raw pkl]", pkl_path)
    print("[Protein length]", len(str(protein_info.get("protein_seq", ""))))
    print("[Protein coords]", coords_shape(protein_info.get("protein_coords", None)))

    manifest_rows = []
    index_rows = []

    all_target_samples_for_labels = []
    split_sample_refs: Dict[str, List[Dict[str, Any]]] = {}

    for split_name, candidates in splits.items():
        if max_per_split > 0:
            candidates = candidates[:max_per_split]

        print(f"\n[{target} / {split_name}] candidates: {len(candidates)}")
        start = time.perf_counter()

        processed = []

        for local_index, candidate in enumerate(candidates):
            if not isinstance(candidate, dict):
                continue

            if "partner_seq" not in candidate:
                continue

            try:
                processed_sample = process_one_candidate(
                    target=target,
                    protein_info=protein_info,
                    candidate=candidate,
                    split_name=split_name,
                    split_index=1 if split_name == "val" else 2 if split_name == "test" else -1,
                    local_index=local_index,
                    patch_k=patch_k,
                    dummy_distance=dummy_distance,
                    keep_coords=keep_coords,
                )
                processed.append(processed_sample)
                all_target_samples_for_labels.append(processed_sample)

            except Exception as exc:
                print(
                    f"[Warning] failed {target}/{split_name}/{local_index}: "
                    f"{repr(exc)}"
                )

            if (local_index + 1) % 100 == 0:
                print(
                    f"[Progress] {target}/{split_name}: "
                    f"{local_index + 1}/{len(candidates)}"
                )

        split_sample_refs[split_name] = processed
        elapsed = time.perf_counter() - start

        out_pkl = out_dir / f"{target}_downstream_{split_name}_processed.pkl"
        save_pickle(processed, out_pkl)

        manifest_rows.append(
            {
                "target": target,
                "split": split_name,
                "processed_pkl": str(out_pkl),
                "n_processed": len(processed),
                "elapsed_seconds": elapsed,
            }
        )

        print(f"[Saved] {out_pkl}")
        print(
            f"[Done] {target}/{split_name}: "
            f"processed={len(processed)}, elapsed={elapsed:.1f}s"
        )

    all_target_samples_for_labels = attach_label_modes(
        samples=all_target_samples_for_labels,
        target=target,
        expected_positives=expected_positives,
    )

    # Re-save splits after labels are attached.
    for split_name, processed in split_sample_refs.items():
        out_pkl = out_dir / f"{target}_downstream_{split_name}_processed.pkl"
        save_pickle(processed, out_pkl)

        for sample_index, sample in enumerate(processed):
            index_rows.append(
                {
                    "target": target,
                    "split": split_name,
                    "sample_index": sample_index,
                    "aptamer_id": sample.get("aptamer_id", ""),
                    "pdb": sample.get("pdb", ""),
                    "sequence": sample.get("partner_seq", ""),
                    "length": len(sample.get("partner_seq", "")),
                    "affinity": sample.get("affinity", np.nan),
                    "label_high_topN": sample.get("label_high_topN", np.nan),
                    "label_low_topN": sample.get("label_low_topN", np.nan),
                    "expected_positive_count": sample.get("expected_positive_count", np.nan),
                    "n_patches": int(sample["patch_indices"].shape[0]),
                    "n_motifs": int(len(sample["motifs"])),
                    "patch_attr_shape": (
                        "x".join(map(str, sample["patch_attr"].shape))
                        if "patch_attr" in sample
                        else ""
                    ),
                    "motif_attr_shape": (
                        "x".join(map(str, sample["motif_attr"].shape))
                        if "motif_attr" in sample
                        else ""
                    ),
                    "has_v2_dummy": "patch_motif_structure_energy_v2" in sample,
                    "secondary_structure": sample.get("secondary_structure", ""),
                    "mfe": sample.get("mfe", np.nan),
                }
            )

    return manifest_rows, index_rows


def parse_targets(raw_targets: Any, all_targets: List[str]) -> List[str]:
    if raw_targets is None:
        return all_targets

    if isinstance(raw_targets, list):
        return [str(item).upper() for item in raw_targets]

    raw = str(raw_targets).strip()

    if raw.lower() == "all":
        return all_targets

    return [
        item.strip().upper()
        for item in raw.split(",")
        if item.strip()
    ]


def preprocess_downstream(cfg: Dict[str, Any]) -> None:
    preprocess_cfg = cfg["preprocess"]
    paths_cfg = cfg["paths"]

    all_targets = [
        str(target).upper()
        for target in preprocess_cfg["targets"]
    ]
    selected_targets = set(
        parse_targets(
            preprocess_cfg.get("targets", all_targets),
            all_targets,
        )
    )

    expected_positives = {
        str(key).upper(): int(value)
        for key, value in preprocess_cfg["expected_positives"].items()
    }

    data_dir = Path(paths_cfg["raw_downstream_dir"])
    out_dir = Path(paths_cfg["processed_downstream_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    pkl_files = sorted(data_dir.glob("*_eval.pkl"))
    pkl_files = [
        path
        for path in pkl_files
        if target_from_path(path, all_targets) in selected_targets
    ]

    if len(pkl_files) == 0:
        raise FileNotFoundError(
            f"No *_eval.pkl files found under {data_dir} "
            f"for targets={sorted(selected_targets)}"
        )

    all_manifest_rows = []
    all_index_rows = []

    for pkl_path in pkl_files:
        manifest_rows, index_rows = process_target(
            pkl_path=pkl_path,
            out_dir=out_dir,
            targets=all_targets,
            expected_positives=expected_positives,
            patch_k=int(preprocess_cfg.get("patch_k", 8)),
            max_per_split=int(preprocess_cfg.get("max_per_split", 0)),
            dummy_distance=float(preprocess_cfg.get("dummy_distance", 99.0)),
            keep_coords=bool(preprocess_cfg.get("keep_coords", False)),
        )

        all_manifest_rows.extend(manifest_rows)
        all_index_rows.extend(index_rows)

    manifest = {
        "raw_downstream_dir": str(data_dir),
        "processed_downstream_dir": str(out_dir),
        "targets": sorted(selected_targets),
        "preserve_original_split": True,
        "note": (
            "FAFormer original validation/test split is preserved. "
            "No pair-disjoint resplitting is applied."
        ),
        "files": all_manifest_rows,
    }

    save_json(manifest, out_dir / "manifest.json")

    index_df = pd.DataFrame(all_index_rows)
    index_df.to_csv(out_dir / "index.csv", index=False)

    if not index_df.empty:
        summary_df = (
            index_df
            .groupby(["target", "split"], as_index=False)
            .agg(
                n_processed=("sample_index", "count"),
                n_labeled_high=("label_high_topN", "sum"),
                n_labeled_low=("label_low_topN", "sum"),
                mean_length=("length", "mean"),
                mean_affinity=("affinity", "mean"),
            )
        )
    else:
        summary_df = pd.DataFrame()

    summary_df.to_csv(out_dir / "summary.csv", index=False)

    print("\nSaved downstream preprocessing outputs:")
    print(out_dir / "manifest.json")
    print(out_dir / "index.csv")
    print(out_dir / "summary.csv")