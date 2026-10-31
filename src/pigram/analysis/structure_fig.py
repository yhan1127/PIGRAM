# src/pigram/analysis/structure_fig.py
# -*- coding: utf-8 -*-

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any, Dict, List


def _load_module(script_path: Path):
    script_path = Path(script_path).resolve()

    if not script_path.exists():
        raise FileNotFoundError(f"Legacy structure figure script not found: {script_path}")

    script_dir = str(script_path.parent)
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)

    module_name = script_path.stem + "_structure_fig_runner"
    spec = importlib.util.spec_from_file_location(module_name, script_path)

    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load legacy script: {script_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return module


def run_structure_figure(cfg: Dict[str, Any]) -> None:
    legacy_cfg = cfg["legacy"]
    structure_cfg = cfg["structure"]

    script_path = Path(legacy_cfg["analysis_dir"]) / "make_structure_figure.py"
    module = _load_module(script_path)

    fig_root = Path(structure_cfg.get("pymol_out_dir", "results/structure/pymol"))
    out_png = Path(structure_cfg.get("output_png", "results/figures/structure/structure_projection.png"))
    out_pdf = Path(structure_cfg.get("output_pdf", "results/figures/structure/structure_projection.pdf"))

    figure_cases = structure_cfg.get("figure_cases", [])

    if not figure_cases:
        raise ValueError(
            "structure.figure_cases is empty. Add case_dir entries pointing to rendered PyMOL PNG folders."
        )

    cases: List[Dict[str, Any]] = []
    for item in figure_cases:
        cases.append(
            {
                "label": str(item["label"]),
                "case_dir": Path(item["case_dir"]),
            }
        )

    out_png.parent.mkdir(parents=True, exist_ok=True)

    module.FIG_ROOT = fig_root
    module.OUT_PNG = out_png
    module.OUT_PDF = out_pdf
    module.CASES = cases

    module.main()