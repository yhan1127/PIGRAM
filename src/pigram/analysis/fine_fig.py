# src/pigram/analysis/fine_fig.py
# -*- coding: utf-8 -*-

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from pigram.analysis.runner import run_legacy_script


def _as_str(x: Any) -> str:
    return "" if x is None else str(x)


def run_fine_refinement_figure(cfg: Dict[str, Any]) -> None:
    legacy_cfg = cfg["legacy"]
    paths = cfg["paths"]
    fig_cfg = cfg["fine_fig"]

    script_path = Path(legacy_cfg["analysis_dir"]) / "make_fine_refinement_figure.py"

    out_dir = Path(paths["figure_out_dir"]) / "fine"

    argv: List[str] = [
        "--coarse_csv", _as_str(paths["coarse_rule_csv"]),
        "--fine_csv", _as_str(paths["fine_rule_csv"]),
        "--outdir", _as_str(out_dir),
        "--split", _as_str(fig_cfg.get("split", "test")),
        "--rule_order", _as_str(fig_cfg.get("rule_order", "0,1,2,3,4,5,6")),
        "--output_prefix", _as_str(fig_cfg.get("output_prefix", "single_fine_refinement_composition")),
    ]

    run_legacy_script(
        script_path=script_path,
        argv=argv,
        extra_paths=[
            legacy_cfg["scripts_dir"],
            legacy_cfg["analysis_dir"],
        ],
    )