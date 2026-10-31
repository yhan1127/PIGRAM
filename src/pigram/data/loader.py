# src/pigram/data/loader.py
# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np
import torch


AA_VOCAB = {
    "<PAD>": 0,
    "<UNK>": 1,
    "A": 2,
    "C": 3,
    "D": 4,
    "E": 5,
    "F": 6,
    "G": 7,
    "H": 8,
    "I": 9,
    "K": 10,
    "L": 11,
    "M": 12,
    "N": 13,
    "P": 14,
    "Q": 15,
    "R": 16,
    "S": 17,
    "T": 18,
    "V": 19,
    "W": 20,
    "Y": 21,
    "B": 22,
    "Z": 23,
    "X": 24,
    "U": 25,
    "O": 26,
}

NA_VOCAB = {
    "<PAD>": 0,
    "<UNK>": 1,
    "A": 2,
    "C": 3,
    "G": 4,
    "U": 5,
    "T": 6,
    "N": 7,
}

MOTIF_TYPE_VOCAB = {
    "unknown": 0,
    "stem_core": 1,
    "stem_edge": 2,
    "hairpin_loop": 3,
    "internal_loop": 4,
    "bulge": 5,
    "tail_unpaired": 6,
    "multi_loop": 7,
}

EDGE_SIDE_VOCAB = {
    "none": 0,
    "left": 1,
    "right": 2,
    "both": 3,
}


def _require_key(sample: Dict[str, Any], key: str) -> Any:
    if key not in sample:
        raise KeyError(f"Required sample key missing: {key}")
    return sample[key]


def _as_long_tensor(x: Any, device: torch.device) -> torch.Tensor:
    return torch.as_tensor(x, dtype=torch.long, device=device)


def _as_float_tensor(x: Any, device: torch.device) -> torch.Tensor:
    return torch.as_tensor(x, dtype=torch.float32, device=device)


def _encode_protein_sequence(seq: str) -> List[int]:
    seq = str(seq).upper()
    return [AA_VOCAB.get(ch, AA_VOCAB["<UNK>"]) for ch in seq]


def _encode_na_sequence(seq: str) -> List[int]:
    seq = str(seq).upper()
    return [NA_VOCAB.get(ch, NA_VOCAB["<UNK>"]) for ch in seq]


def _infer_num_motifs(sample: Dict[str, Any]) -> int:
    if "motif_attr" in sample:
        return int(np.asarray(sample["motif_attr"]).shape[0])
    if "motif_type_ids" in sample:
        return int(len(sample["motif_type_ids"]))
    if "motifs" in sample:
        return int(len(sample["motifs"]))
    if "patch_motif_structure_energy_v2" in sample:
        return int(np.asarray(sample["patch_motif_structure_energy_v2"]).shape[1])
    if "patch_motif_label" in sample:
        return int(np.asarray(sample["patch_motif_label"]).shape[1])
    raise KeyError("Cannot infer number of motifs from sample.")


def _infer_num_patches(sample: Dict[str, Any]) -> int:
    if "patch_attr" in sample:
        return int(np.asarray(sample["patch_attr"]).shape[0])
    if "patch_indices" in sample:
        return int(np.asarray(sample["patch_indices"]).shape[0])
    if "patch_motif_structure_energy_v2" in sample:
        return int(np.asarray(sample["patch_motif_structure_energy_v2"]).shape[0])
    if "patch_motif_label" in sample:
        return int(np.asarray(sample["patch_motif_label"]).shape[0])
    raise KeyError("Cannot infer number of patches from sample.")


def _motif_get(motif: Dict[str, Any], keys: List[str], default: Any = None) -> Any:
    for key in keys:
        if key in motif:
            return motif[key]
    return default


def _motif_start_end(motif: Dict[str, Any]) -> Tuple[int | None, int | None]:
    start = _motif_get(
        motif,
        ["start", "start_idx", "start_index", "begin", "left", "i"],
        None,
    )
    end = _motif_get(
        motif,
        ["end", "end_idx", "end_index", "stop", "right", "j"],
        None,
    )

    if start is None and "indices" in motif:
        indices = list(motif["indices"])
        if len(indices) > 0:
            start = min(indices)
            end = max(indices) + 1

    if start is None:
        return None, None

    start = int(start)

    if end is None:
        length = _motif_get(motif, ["length", "len"], None)
        if length is not None:
            end = start + int(length)
        else:
            end = start + 1
    else:
        end = int(end)

    if end <= start:
        end = start + 1

    return start, end


def _motif_sequence_from_dict(motif: Dict[str, Any], partner_seq: str) -> str:
    seq = _motif_get(
        motif,
        ["sequence", "seq", "motif_seq", "nucleotide_seq", "nt_seq"],
        None,
    )
    if seq is not None:
        if isinstance(seq, (list, tuple)):
            return "".join(str(x) for x in seq)
        return str(seq)

    start, end = _motif_start_end(motif)
    if start is None or end is None:
        return "N"

    partner_seq = str(partner_seq)
    start = max(0, min(start, len(partner_seq)))
    end = max(start + 1, min(end, len(partner_seq)))

    subseq = partner_seq[start:end]
    if len(subseq) == 0:
        return "N"
    return subseq


def _fallback_motif_sequences(sample: Dict[str, Any], n_motifs: int) -> List[str]:
    partner_seq = str(sample.get("partner_seq", ""))
    if len(partner_seq) == 0:
        return ["N" for _ in range(n_motifs)]

    if n_motifs <= 1:
        return [partner_seq]

    boundaries = np.linspace(0, len(partner_seq), n_motifs + 1).round().astype(int)
    seqs: List[str] = []

    for i in range(n_motifs):
        start = int(boundaries[i])
        end = int(boundaries[i + 1])
        if end <= start:
            end = min(start + 1, len(partner_seq))
        seqs.append(partner_seq[start:end] if start < len(partner_seq) else "N")

    return seqs


def _get_motif_sequences(sample: Dict[str, Any], n_motifs: int) -> List[str]:
    partner_seq = str(sample.get("partner_seq", ""))

    if "motifs" not in sample:
        return _fallback_motif_sequences(sample, n_motifs)

    motifs = list(sample["motifs"])
    seqs = [
        _motif_sequence_from_dict(motif, partner_seq)
        if isinstance(motif, dict)
        else "N"
        for motif in motifs
    ]

    if len(seqs) < n_motifs:
        seqs.extend(_fallback_motif_sequences(sample, n_motifs - len(seqs)))
    if len(seqs) > n_motifs:
        seqs = seqs[:n_motifs]

    return seqs


def _infer_motif_type_id(motif: Any) -> int:
    if not isinstance(motif, dict):
        return MOTIF_TYPE_VOCAB["unknown"]

    raw = _motif_get(
        motif,
        ["type", "motif_type", "name", "label"],
        "unknown",
    )
    raw = str(raw).lower()
    return MOTIF_TYPE_VOCAB.get(raw, MOTIF_TYPE_VOCAB["unknown"])


def _infer_edge_side_id(motif: Any) -> int:
    if not isinstance(motif, dict):
        return EDGE_SIDE_VOCAB["none"]

    raw = _motif_get(
        motif,
        ["edge_side", "side", "edge_state"],
        "none",
    )
    raw = str(raw).lower()
    return EDGE_SIDE_VOCAB.get(raw, EDGE_SIDE_VOCAB["none"])


def _infer_motif_start_end_list(sample: Dict[str, Any], n_motifs: int) -> List[Tuple[int, int]]:
    partner_seq = str(sample.get("partner_seq", ""))
    length = max(len(partner_seq), 1)

    if "motifs" in sample:
        pairs: List[Tuple[int, int]] = []
        for motif in sample["motifs"]:
            if isinstance(motif, dict):
                start, end = _motif_start_end(motif)
            else:
                start, end = None, None

            if start is None or end is None:
                pairs.append((0, 1))
            else:
                pairs.append((max(0, start), max(start + 1, end)))

        if len(pairs) < n_motifs:
            boundaries = np.linspace(0, length, n_motifs + 1).round().astype(int)
            for i in range(len(pairs), n_motifs):
                pairs.append(
                    (
                        int(boundaries[i]),
                        int(max(boundaries[i + 1], boundaries[i] + 1)),
                    )
                )

        return pairs[:n_motifs]

    boundaries = np.linspace(0, length, n_motifs + 1).round().astype(int)
    return [
        (
            int(boundaries[i]),
            int(max(boundaries[i + 1], boundaries[i] + 1)),
        )
        for i in range(n_motifs)
    ]


def _pairing_ratio_from_structure(structure: str, start: int, end: int) -> float:
    if not structure:
        return 0.0

    start = max(0, min(start, len(structure)))
    end = max(start + 1, min(end, len(structure)))

    sub = structure[start:end]
    if len(sub) == 0:
        return 0.0

    paired = sum(ch in {"(", ")", "[", "]", "{", "}", "<", ">"} for ch in sub)
    return float(paired / len(sub))


def _gc_content(seq: str) -> float:
    seq = str(seq).upper()
    valid = [ch for ch in seq if ch in {"A", "C", "G", "U", "T", "N"}]
    if len(valid) == 0:
        return 0.0
    return float(sum(ch in {"G", "C"} for ch in valid) / len(valid))


def _is_dna_sample(sample: Dict[str, Any]) -> bool:
    text = " ".join(
        str(sample.get(key, "")).lower()
        for key in ["kind", "na_source", "partner_kind", "molecule_type"]
    )
    return "dna" in text


def _build_missing_token_fields(sample: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(sample)

    if "protein_token_ids" not in out:
        protein_seq = _require_key(out, "protein_seq")
        out["protein_token_ids"] = np.asarray(
            _encode_protein_sequence(protein_seq),
            dtype=np.int64,
        )

    n_motifs = _infer_num_motifs(out)
    motif_sequences = _get_motif_sequences(out, n_motifs)
    motif_ranges = _infer_motif_start_end_list(out, n_motifs)

    if "motif_token_ids_list" not in out:
        out["motif_token_ids_list"] = [
            np.asarray(_encode_na_sequence(seq), dtype=np.int64)
            for seq in motif_sequences
        ]

    if "motif_type_ids" not in out:
        motifs = list(out.get("motifs", []))
        motif_type_ids = []
        for i in range(n_motifs):
            motif = motifs[i] if i < len(motifs) else {}
            motif_type_ids.append(_infer_motif_type_id(motif))
        out["motif_type_ids"] = np.asarray(motif_type_ids, dtype=np.int64)

    if "motif_lengths" not in out:
        out["motif_lengths"] = np.asarray(
            [max(1, len(seq)) for seq in motif_sequences],
            dtype=np.float32,
        )

    if "motif_pairing_ratios" not in out:
        secondary_structure = str(out.get("secondary_structure", ""))
        out["motif_pairing_ratios"] = np.asarray(
            [
                _pairing_ratio_from_structure(secondary_structure, start, end)
                for start, end in motif_ranges
            ],
            dtype=np.float32,
        )

    if "motif_gc_contents" not in out:
        out["motif_gc_contents"] = np.asarray(
            [_gc_content(seq) for seq in motif_sequences],
            dtype=np.float32,
        )

    if "motif_edge_side_ids" not in out:
        motifs = list(out.get("motifs", []))
        edge_ids = []
        for i in range(n_motifs):
            motif = motifs[i] if i < len(motifs) else {}
            edge_ids.append(_infer_edge_side_id(motif))
        out["motif_edge_side_ids"] = np.asarray(edge_ids, dtype=np.int64)

    if "motif_is_terminal" not in out:
        partner_len = max(len(str(out.get("partner_seq", ""))), 1)
        terminal = []
        for start, end in motif_ranges:
            terminal.append(float(start <= 0 or end >= partner_len))
        out["motif_is_terminal"] = np.asarray(terminal, dtype=np.float32)

    if "motif_center_positions" not in out:
        partner_len = max(len(str(out.get("partner_seq", ""))), 1)
        centers = []
        for start, end in motif_ranges:
            center = 0.5 * (float(start) + float(end))
            centers.append(center / float(partner_len))
        out["motif_center_positions"] = np.asarray(centers, dtype=np.float32)

    if "motif_is_dna" not in out:
        is_dna = float(_is_dna_sample(out))
        out["motif_is_dna"] = np.full((n_motifs,), is_dna, dtype=np.float32)

    if "patch_center_indices" not in out:
        patch_indices = np.asarray(_require_key(out, "patch_indices"))
        out["patch_center_indices"] = patch_indices[:, 0].astype(np.int64)

    if "patch_relative_distances" not in out:
        patch_indices = np.asarray(_require_key(out, "patch_indices"))
        out["patch_relative_distances"] = np.zeros_like(patch_indices, dtype=np.float32)

    if "patch_motif_structure_energy_v2" not in out:
        if "patch_motif_energy" in out:
            out["patch_motif_structure_energy_v2"] = out["patch_motif_energy"]
        elif "patch_motif_label" in out:
            out["patch_motif_structure_energy_v2"] = np.asarray(
                out["patch_motif_label"],
                dtype=np.float32,
            )
        else:
            raise KeyError("patch_motif_structure_energy_v2 missing and cannot be inferred.")

    if "patch_motif_label" not in out:
        y = np.asarray(out["patch_motif_structure_energy_v2"], dtype=np.float32)
        out["patch_motif_label"] = (y >= 0.5).astype(np.float32)

    if "patch_importance" not in out:
        label = np.asarray(out["patch_motif_label"], dtype=np.float32)
        out["patch_importance"] = (label.max(axis=1) > 0).astype(np.float32)

    return out


def prepare_single_sample_tensors(
    sample: Dict[str, Any],
    device: torch.device,
) -> Dict[str, Any]:
    sample = _build_missing_token_fields(sample)

    batch: Dict[str, Any] = {
        "protein_token_ids": _as_long_tensor(_require_key(sample, "protein_token_ids"), device),
        "patch_indices": _as_long_tensor(_require_key(sample, "patch_indices"), device),
        "patch_center_indices": _as_long_tensor(_require_key(sample, "patch_center_indices"), device),
        "patch_relative_distances": _as_float_tensor(_require_key(sample, "patch_relative_distances"), device),
        "motif_token_ids_list": [
            _as_long_tensor(x, device)
            for x in _require_key(sample, "motif_token_ids_list")
        ],
        "motif_type_ids": _as_long_tensor(_require_key(sample, "motif_type_ids"), device),
        "motif_lengths": _as_float_tensor(_require_key(sample, "motif_lengths"), device),
        "motif_pairing_ratios": _as_float_tensor(_require_key(sample, "motif_pairing_ratios"), device),
        "motif_gc_contents": _as_float_tensor(_require_key(sample, "motif_gc_contents"), device),
        "motif_edge_side_ids": _as_long_tensor(_require_key(sample, "motif_edge_side_ids"), device),
        "motif_is_terminal": _as_float_tensor(_require_key(sample, "motif_is_terminal"), device),
        "motif_center_positions": _as_float_tensor(_require_key(sample, "motif_center_positions"), device),
        "motif_is_dna": _as_float_tensor(_require_key(sample, "motif_is_dna"), device),
        "patch_motif_label": _as_float_tensor(_require_key(sample, "patch_motif_label"), device),
        "patch_importance": _as_float_tensor(_require_key(sample, "patch_importance"), device),
        "patch_motif_structure_energy_v2": _as_float_tensor(
            _require_key(sample, "patch_motif_structure_energy_v2"),
            device,
        ),
    }

    if "patch_attr" in sample:
        batch["patch_attr"] = _as_float_tensor(sample["patch_attr"], device)
    else:
        raise KeyError("patch_attr missing from sample.")

    if "motif_attr" in sample:
        batch["motif_attr"] = _as_float_tensor(sample["motif_attr"], device)
    else:
        raise KeyError("motif_attr missing from sample.")

    n_patches = _infer_num_patches(sample)
    n_motifs = _infer_num_motifs(sample)

    if batch["patch_attr"].shape[0] != n_patches:
        raise RuntimeError(
            f"patch_attr patch count mismatch: {tuple(batch['patch_attr'].shape)} vs {n_patches}"
        )

    if batch["motif_attr"].shape[0] != n_motifs:
        raise RuntimeError(
            f"motif_attr motif count mismatch: {tuple(batch['motif_attr'].shape)} vs {n_motifs}"
        )

    if batch["patch_motif_structure_energy_v2"].shape != batch["patch_motif_label"].shape:
        raise RuntimeError(
            "patch_motif_structure_energy_v2 and patch_motif_label shape mismatch: "
            f"{tuple(batch['patch_motif_structure_energy_v2'].shape)} vs "
            f"{tuple(batch['patch_motif_label'].shape)}"
        )

    return batch


def move_sample_to_device(
    sample: Dict[str, Any],
    device: torch.device,
) -> Dict[str, Any]:
    return prepare_single_sample_tensors(sample, device=device)


def make_dummy_annotation(sample: Dict[str, Any]) -> Dict[str, Any]:
    n_patches = _infer_num_patches(sample)
    n_motifs = _infer_num_motifs(sample)

    return {
        "fine_label_matrix": np.zeros((n_patches, n_motifs), dtype=np.int64),
        "distill_mask_matrix": np.zeros((n_patches, n_motifs), dtype=bool),
        "coarse_rule_matrix": np.zeros((n_patches, n_motifs), dtype=np.int64),
    }


def attach_annotation_tensors(
    batch: Dict[str, Any],
    annotation: Dict[str, Any] | None,
    device: torch.device | None = None,
    sample: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    out = dict(batch)

    if device is None:
        for value in out.values():
            if torch.is_tensor(value):
                device = value.device
                break
        if device is None:
            device = torch.device("cpu")

    if "patch_motif_structure_energy_v2" in out:
        n_patches, n_motifs = out["patch_motif_structure_energy_v2"].shape
    elif "patch_motif_label" in out:
        n_patches, n_motifs = out["patch_motif_label"].shape
    elif sample is not None:
        dummy = make_dummy_annotation(sample)
        n_patches, n_motifs = dummy["fine_label_matrix"].shape
    else:
        raise KeyError(
            "Cannot infer annotation shape. Need patch_motif_structure_energy_v2, "
            "patch_motif_label, or sample."
        )

    if annotation is None:
        annotation = {
            "fine_label_matrix": np.zeros((n_patches, n_motifs), dtype=np.int64),
            "distill_mask_matrix": np.zeros((n_patches, n_motifs), dtype=bool),
            "coarse_rule_matrix": np.zeros((n_patches, n_motifs), dtype=np.int64),
        }
    else:
        annotation = dict(annotation)

        if "fine_label_matrix" not in annotation:
            annotation["fine_label_matrix"] = np.zeros((n_patches, n_motifs), dtype=np.int64)
        if "distill_mask_matrix" not in annotation:
            annotation["distill_mask_matrix"] = np.zeros((n_patches, n_motifs), dtype=bool)
        if "coarse_rule_matrix" not in annotation:
            annotation["coarse_rule_matrix"] = np.zeros((n_patches, n_motifs), dtype=np.int64)

    out["fine_label_matrix"] = _as_long_tensor(annotation["fine_label_matrix"], device)
    out["distill_mask_matrix"] = torch.as_tensor(
        annotation["distill_mask_matrix"],
        dtype=torch.bool,
        device=device,
    )
    out["coarse_rule_matrix"] = _as_long_tensor(annotation["coarse_rule_matrix"], device)

    return out


def move_sample_to_device_with_ann(
    sample: Dict[str, Any],
    annotation: Dict[str, Any] | None,
    device: torch.device,
) -> Dict[str, Any]:
    batch = prepare_single_sample_tensors(sample, device=device)
    return attach_annotation_tensors(
        batch=batch,
        annotation=annotation,
        device=device,
        sample=sample,
    )


def estimate_pos_weight(labels: torch.Tensor) -> float:
    labels = labels.detach().float().reshape(-1)
    pos = float((labels > 0.5).sum().item())
    total = float(labels.numel())
    neg = max(total - pos, 0.0)

    if pos <= 0:
        return 1.0

    return float(neg / max(pos, 1.0))


__all__ = [
    "prepare_single_sample_tensors",
    "move_sample_to_device",
    "attach_annotation_tensors",
    "move_sample_to_device_with_ann",
    "make_dummy_annotation",
    "estimate_pos_weight",
]