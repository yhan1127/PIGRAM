# src/pigram/analysis/attr_route.py
# -*- coding: utf-8 -*-

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from pigram.analysis.runner import run_legacy_script


def _as_str(x: Any) -> str:
    return "" if x is None else str(x)


def run_attr_route_figure(cfg: Dict[str, Any]) -> None:
    legacy_cfg = cfg["legacy"]
    attr_cfg = cfg["attr_route"]

    script_path = Path(legacy_cfg["analysis_dir"]) / "make_attr_route.py"

    argv: List[str] = [
        "--signature_csv", _as_str(attr_cfg.get("signature_csv", "results/route/test/route_signature_table.csv")),
        "--out_dir", _as_str(attr_cfg.get("out_dir", "results/figures/attr_route")),
        "--top_k", _as_str(attr_cfg.get("top_k", 10)),
        "--focus_groups", _as_str(attr_cfg.get("focus_groups", "auto")),
    ]

    run_legacy_script(
        script_path=script_path,
        argv=argv,
        extra_paths=[
            legacy_cfg["scripts_dir"],
            legacy_cfg["analysis_dir"],
        ],
    )