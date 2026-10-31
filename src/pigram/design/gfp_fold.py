# src/pigram/design/gfp_fold.py
# -*- coding: utf-8 -*-

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from pigram.analysis.runner import run_legacy_script


def _as_str(x: Any) -> str:
    return "" if x is None else str(x)


def run_gfp_fold_validation(cfg: Dict[str, Any]) -> None:
    legacy_cfg = cfg["legacy"]
    fold_cfg = cfg["gfp_fold"]

    script_path = Path(legacy_cfg["design_dir"]) / "fold_gfp.py"

    argv: List[str] = [
        "--design_csv", _as_str(fold_cfg.get("design_csv")),
        "--out_dir", _as_str(fold_cfg.get("out_dir")),
    ]

    run_legacy_script(
        script_path=script_path,
        argv=argv,
        extra_paths=[
            legacy_cfg["scripts_dir"],
            legacy_cfg["design_dir"],
        ],
    )