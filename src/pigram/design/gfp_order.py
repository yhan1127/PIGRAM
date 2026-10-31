# src/pigram/design/gfp_order.py
# -*- coding: utf-8 -*-

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from pigram.analysis.runner import run_legacy_script


def _as_str(x: Any) -> str:
    return "" if x is None else str(x)


def run_gfp_ordering(cfg: Dict[str, Any]) -> None:
    legacy_cfg = cfg["legacy"]
    order_cfg = cfg["gfp_order"]

    script_path = Path(legacy_cfg["design_dir"]) / "order_gfp.py"

    argv: List[str] = [
        "--design_csv", _as_str(order_cfg.get("design_csv")),
        "--validation_csv", _as_str(order_cfg.get("validation_csv")),
        "--out_dir", _as_str(order_cfg.get("out_dir")),
        "--aptamers", _as_str(order_cfg.get("aptamers", "default")),
    ]

    run_legacy_script(
        script_path=script_path,
        argv=argv,
        extra_paths=[
            legacy_cfg["scripts_dir"],
            legacy_cfg["design_dir"],
        ],
    )