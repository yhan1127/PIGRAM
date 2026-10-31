# src/pigram/analysis/route_fig.py
# -*- coding: utf-8 -*-

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from pigram.analysis.runner import run_legacy_script


def _as_str(x: Any) -> str:
    return "" if x is None else str(x)


def run_route_figures(cfg: Dict[str, Any]) -> None:
    legacy_cfg = cfg["legacy"]
    paths = cfg["paths"]
    fig_cfg = cfg["route_fig"]

    script_path = Path(legacy_cfg["analysis_dir"]) / "make_route_atlas_figures.py"

    out_dir = Path(paths["figure_out_dir"]) / "route"

    argv: List[str] = [
        "--atlas_dir", _as_str(paths["route_atlas_dir"]),
        "--out_dir", _as_str(out_dir),
        "--fine_min_support", _as_str(fig_cfg.get("fine_min_support", 1)),
        "--fine_sort", _as_str(fig_cfg.get("fine_sort", "grammar")),
    ]

    if bool(fig_cfg.get("annotate", False)):
        argv.append("--annotate")

    run_legacy_script(
        script_path=script_path,
        argv=argv,
        extra_paths=[
            legacy_cfg["scripts_dir"],
            legacy_cfg["analysis_dir"],
        ],
    )