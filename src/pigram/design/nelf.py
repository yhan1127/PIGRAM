# src/pigram/design/nelf.py
# -*- coding: utf-8 -*-

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from pigram.analysis.runner import run_legacy_script


def _as_str(x: Any) -> str:
    return "" if x is None else str(x)


def run_nelf_design(cfg: Dict[str, Any]) -> None:
    legacy_cfg = cfg["legacy"]
    nelf_cfg = cfg["nelf"]

    script_path = Path(legacy_cfg["design_dir"]) / "design_nelf.py"

    argv: List[str] = [
        "--favorable_csv", _as_str(nelf_cfg.get("favorable_csv")),
        "--suppressive_csv", _as_str(nelf_cfg.get("suppressive_csv")),
        "--out_dir", _as_str(nelf_cfg.get("out_dir")),
        "--target", _as_str(nelf_cfg.get("target", "NELF")),
        "--top_candidates", _as_str(nelf_cfg.get("top_candidates", 4)),
        "--recommended_aptamers", _as_str(nelf_cfg.get("recommended_aptamers", "")),
        "--seed", _as_str(cfg.get("seed", 42)),
    ]

    run_legacy_script(
        script_path=script_path,
        argv=argv,
        extra_paths=[
            legacy_cfg["scripts_dir"],
            legacy_cfg["design_dir"],
        ],
    )