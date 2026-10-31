# src/pigram/data/motif.py
# -*- coding: utf-8 -*-

from __future__ import annotations

import math
from typing import Dict, List

from pigram.utils.sequence import compute_gc_content, compute_pairing_ratio


MOTIF_TYPE_TO_ID = {
    "stem_core": 0,
    "stem_edge": 1,
    "hairpin_loop": 2,
    "internal_loop": 3,
    "bulge": 4,
    "multi_loop": 5,
    "tail_unpaired": 6,
}

MOTIF_TYPE_ID_TO_NAME = {
    value: key for key, value in MOTIF_TYPE_TO_ID.items()
}


def build_pair_map(dot_bracket: str) -> List[int]:
    """
    Build base-pair map from dot-bracket notation.

    pair_map[i] = paired index of i, or -1 if unpaired.
    """
    dot_bracket = str(dot_bracket)
    n = len(dot_bracket)

    pair_map = [-1] * n
    stack: List[int] = []

    for idx, char in enumerate(dot_bracket):
        if char == "(":
            stack.append(idx)
        elif char == ")":
            if not stack:
                continue
            paired_idx = stack.pop()
            pair_map[idx] = paired_idx
            pair_map[paired_idx] = idx

    return pair_map


def is_stem_edge(index: int, dot_bracket: str) -> bool:
    """
    A paired position is considered stem_edge if at least one neighbor is
    unpaired or at the sequence boundary.
    """
    dot_bracket = str(dot_bracket)
    n = len(dot_bracket)

    if index < 0 or index >= n:
        return False

    if dot_bracket[index] not in ("(", ")"):
        return False

    left_paired = index > 0 and dot_bracket[index - 1] in ("(", ")")
    right_paired = index < n - 1 and dot_bracket[index + 1] in ("(", ")")

    return not (left_paired and right_paired)


def classify_unpaired_region(
    start: int,
    end: int,
    dot_bracket: str,
    pair_map: List[int],
) -> str:
    """
    Classify a contiguous unpaired region [start, end).

    Returns one of:
        hairpin_loop
        internal_loop
        bulge
        multi_loop
        tail_unpaired
    """
    dot_bracket = str(dot_bracket)
    n = len(dot_bracket)

    left_paired = start > 0 and dot_bracket[start - 1] in ("(", ")")
    right_paired = end < n and dot_bracket[end] in ("(", ")")

    if not left_paired or not right_paired:
        return "tail_unpaired"

    left_partner = pair_map[start - 1] if left_paired else -1
    right_partner = pair_map[end] if right_paired else -1

    # Canonical hairpin-like closure.
    if left_partner == end and right_partner == start - 1:
        return "hairpin_loop"

    if left_paired and right_paired:
        left_neighbor_paired = start > 1 and dot_bracket[start - 2] in ("(", ")")
        right_neighbor_paired = end < n - 1 and dot_bracket[end + 1] in ("(", ")")

        if left_neighbor_paired != right_neighbor_paired:
            return "bulge"

        return "internal_loop"

    return "multi_loop"


def token_level_labels(dot_bracket: str) -> List[str]:
    """
    Assign fine-grained motif labels at nucleotide/token level.
    """
    dot_bracket = str(dot_bracket)
    n = len(dot_bracket)

    labels = ["tail_unpaired"] * n
    pair_map = build_pair_map(dot_bracket)

    for idx, char in enumerate(dot_bracket):
        if char in ("(", ")"):
            labels[idx] = "stem_edge" if is_stem_edge(idx, dot_bracket) else "stem_core"

    idx = 0
    while idx < n:
        if dot_bracket[idx] != ".":
            idx += 1
            continue

        start = idx
        while idx < n and dot_bracket[idx] == ".":
            idx += 1
        end = idx

        region_type = classify_unpaired_region(
            start=start,
            end=end,
            dot_bracket=dot_bracket,
            pair_map=pair_map,
        )

        for pos in range(start, end):
            labels[pos] = region_type

    return labels


def merge_consecutive_labels(labels: List[str]) -> List[Dict]:
    """
    Merge consecutive token labels into motif segments.
    """
    if len(labels) == 0:
        return []

    motifs = []
    start = 0
    current_type = labels[0]

    for idx in range(1, len(labels)):
        if labels[idx] != current_type:
            motifs.append(
                {
                    "start": start,
                    "end": idx,
                    "type": current_type,
                    "length": idx - start,
                    "indices": list(range(start, idx)),
                }
            )
            start = idx
            current_type = labels[idx]

    motifs.append(
        {
            "start": start,
            "end": len(labels),
            "type": current_type,
            "length": len(labels) - start,
            "indices": list(range(start, len(labels))),
        }
    )

    return motifs


def infer_edge_side(motif: Dict, sequence_length: int) -> str:
    """
    Infer left/right side for stem_edge motifs.
    For non-stem_edge motifs, return 'none'.
    """
    if motif["type"] != "stem_edge":
        return "none"

    start = int(motif["start"])
    end = int(motif["end"])

    motif_center = 0.5 * (start + end - 1)
    seq_center = 0.5 * max(sequence_length - 1, 0)

    if motif_center <= seq_center:
        return "left"

    return "right"


def compute_normalized_center_position(motif: Dict, sequence_length: int) -> float:
    start = int(motif["start"])
    end = int(motif["end"])

    motif_center = 0.5 * (start + end - 1)

    if sequence_length <= 1:
        return 0.0

    return float(motif_center / (sequence_length - 1))


def _normalize_nt_char(char: str) -> str:
    char = str(char).upper()

    if char == "T":
        return "U"

    if char in {"A", "C", "G", "U"}:
        return char

    return "N"


def _motif_nt_fractions(sequence: str) -> Dict[str, float]:
    seq = [_normalize_nt_char(char) for char in str(sequence)]
    n = max(len(seq), 1)

    count_a = sum(char == "A" for char in seq)
    count_c = sum(char == "C" for char in seq)
    count_g = sum(char == "G" for char in seq)
    count_u = sum(char == "U" for char in seq)

    frac_a = count_a / n
    frac_c = count_c / n
    frac_g = count_g / n
    frac_u = count_u / n

    return {
        "frac_A": float(frac_a),
        "frac_C": float(frac_c),
        "frac_G": float(frac_g),
        "frac_U": float(frac_u),
        "frac_purine": float(frac_a + frac_g),
    }


def _nt_entropy_from_fractions(
    frac_a: float,
    frac_c: float,
    frac_g: float,
    frac_u: float,
) -> float:
    probs = [frac_a, frac_c, frac_g, frac_u]
    eps = 1e-8
    entropy = 0.0

    for prob in probs:
        if prob > 0:
            entropy -= prob * math.log(prob + eps)

    return float(entropy / math.log(4.0))


def _max_nt_fraction(
    frac_a: float,
    frac_c: float,
    frac_g: float,
    frac_u: float,
) -> float:
    return float(max(frac_a, frac_c, frac_g, frac_u))


def _max_run_fraction(sequence: str) -> float:
    seq = [_normalize_nt_char(char) for char in str(sequence)]

    if len(seq) == 0:
        return 0.0

    best = 1
    current = 1

    for idx in range(1, len(seq)):
        if seq[idx] == seq[idx - 1]:
            current += 1
            best = max(best, current)
        else:
            current = 1

    return float(best / len(seq))


def _compute_local_wobble_pair_fraction(sequence: str, dot_bracket: str) -> float:
    """
    Motif-local wobble-pair proxy using motif substring and local dot-bracket.
    Count GU / UG among motif-internal paired bases.
    """
    sequence = str(sequence)
    dot_bracket = str(dot_bracket)

    if len(sequence) == 0 or len(sequence) != len(dot_bracket):
        return 0.0

    seq = [_normalize_nt_char(char) for char in sequence]
    pair_map = build_pair_map(dot_bracket)

    visited = set()
    wobble_pairs = 0
    total_pairs = 0

    for idx, paired_idx in enumerate(pair_map):
        if paired_idx < 0:
            continue

        if idx in visited or paired_idx in visited:
            continue

        visited.add(idx)
        visited.add(paired_idx)

        a = seq[idx]
        b = seq[paired_idx]

        total_pairs += 1

        if (a == "G" and b == "U") or (a == "U" and b == "G"):
            wobble_pairs += 1

    if total_pairs == 0:
        return 0.0

    return float(wobble_pairs / total_pairs)


def add_motif_features(
    motif: Dict,
    sequence: str,
    dot_bracket: str,
    sequence_length: int,
    na_source: str = "unknown",
) -> Dict:
    """
    Add type ids, positional, structure, and chemistry features to one motif.
    """
    out = dict(motif)

    motif_type = str(out["type"])
    start = int(out["start"])
    end = int(out["end"])

    motif_seq = str(sequence[start:end])
    motif_dot = str(dot_bracket[start:end])

    out["sequence"] = motif_seq
    out["dot_bracket"] = motif_dot

    out["type_id"] = int(MOTIF_TYPE_TO_ID.get(motif_type, 7))
    out["edge_side"] = infer_edge_side(out, sequence_length=sequence_length)
    out["edge_side_id"] = {"none": 0, "left": 1, "right": 2}.get(out["edge_side"], 0)
    out["is_terminal"] = int(start == 0 or end == sequence_length)
    out["normalized_center_position"] = compute_normalized_center_position(
        out,
        sequence_length=sequence_length,
    )

    out["gc_content"] = float(compute_gc_content(motif_seq))
    out["pairing_ratio"] = float(compute_pairing_ratio(motif_dot))

    nt_fracs = _motif_nt_fractions(motif_seq)
    frac_a = nt_fracs["frac_A"]
    frac_c = nt_fracs["frac_C"]
    frac_g = nt_fracs["frac_G"]
    frac_u = nt_fracs["frac_U"]

    out["frac_A"] = frac_a
    out["frac_C"] = frac_c
    out["frac_G"] = frac_g
    out["frac_U"] = frac_u
    out["frac_purine"] = nt_fracs["frac_purine"]
    out["nt_entropy"] = _nt_entropy_from_fractions(frac_a, frac_c, frac_g, frac_u)
    out["max_nt_fraction"] = _max_nt_fraction(frac_a, frac_c, frac_g, frac_u)
    out["max_run_fraction"] = _max_run_fraction(motif_seq)
    out["flexibility_proxy"] = 0.5 * (1.0 - out["pairing_ratio"]) + 0.5 * (1.0 - out["gc_content"])
    out["wobble_pair_fraction"] = _compute_local_wobble_pair_fraction(motif_seq, motif_dot)
    out["is_dna"] = 1.0 if str(na_source).lower() == "dna" else 0.0

    return out


def build_motifs_from_dotbracket(
    sequence: str,
    dot_bracket: str,
    na_source: str = "unknown",
) -> List[Dict]:
    """
    Convert a sequence and dot-bracket string into motif segments.

    Args:
        sequence:
            RNA-normalized sequence.
        dot_bracket:
            Dot-bracket secondary structure of the same length.
        na_source:
            "rna", "dna", or "unknown".

    Returns:
        List of motif dictionaries.
    """
    sequence = str(sequence)
    dot_bracket = str(dot_bracket)

    if len(sequence) != len(dot_bracket):
        raise ValueError(
            f"sequence and dot_bracket length mismatch: "
            f"{len(sequence)} vs {len(dot_bracket)}"
        )

    labels = token_level_labels(dot_bracket)
    motifs = merge_consecutive_labels(labels)

    enriched = [
        add_motif_features(
            motif=motif,
            sequence=sequence,
            dot_bracket=dot_bracket,
            sequence_length=len(sequence),
            na_source=na_source,
        )
        for motif in motifs
    ]

    return enriched