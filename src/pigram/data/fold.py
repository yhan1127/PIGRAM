# src/pigram/data/fold.py
# -*- coding: utf-8 -*-

from __future__ import annotations

from typing import Tuple


def normalize_rna_sequence(seq: str) -> str:
    """
    Normalize a nucleic-acid sequence for ViennaRNA folding.

    - uppercase
    - convert T to U
    - keep only A/U/G/C/N
    """
    if seq is None:
        return ""

    seq = str(seq).upper().replace("T", "U")
    valid_chars = set("AUGCN")

    return "".join(ch for ch in seq if ch in valid_chars)


def predict_secondary_structure(seq: str) -> Tuple[str, float]:
    """
    Predict RNA secondary structure using ViennaRNA.

    Args:
        seq:
            RNA or DNA sequence. DNA T is converted to RNA U.

    Returns:
        dot_bracket:
            Dot-bracket secondary structure.
        mfe:
            Minimum free energy from ViennaRNA.
    """
    try:
        import ViennaRNA as RNA
    except Exception:
        import RNA  # type: ignore

    seq = normalize_rna_sequence(seq)

    if len(seq) == 0:
        raise ValueError("Input RNA sequence is empty after normalization.")

    dot_bracket, mfe = RNA.fold(seq)

    return str(dot_bracket), float(mfe)