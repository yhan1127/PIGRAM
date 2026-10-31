# src/pigram/design/gfp.py
# -*- coding: utf-8 -*-

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from pigram.analysis.runner import run_legacy_script


def _as_str(x: Any) -> str:
    return "" if x is None else str(x)


def run_gfp_design(cfg: Dict[str, Any]) -> None:
    legacy_cfg = cfg["legacy"]
    gfp_cfg = cfg["gfp"]

    design_dir = Path(legacy_cfg["design_dir"])

    if bool(gfp_cfg.get("run_analysis", True)):
        analyze_script = design_dir / "analyze_gfp.py"

        argv: List[str] = [
            "--processed_dir", _as_str(gfp_cfg.get("processed_dir")),
            "--candidate_csv", _as_str(gfp_cfg.get("candidate_csv")),
            "--target", _as_str(gfp_cfg.get("target", "GFP")),
            "--splits", _as_str(gfp_cfg.get("splits", "val,test")),
            "--score_col", _as_str(gfp_cfg.get("score_col", "grammar_HA_top50_prob")),
            "--top_candidates", _as_str(gfp_cfg.get("top_candidates", 50)),
            "--pair_top_k", _as_str(gfp_cfg.get("pair_top_k", 30)),
            "--motif_top_patch_k", _as_str(gfp_cfg.get("motif_top_patch_k", 8)),
            "--checkpoint_path", _as_str(gfp_cfg.get("checkpoint_path")),
            "--config_path", _as_str(gfp_cfg.get("config_path")),
            "--device", _as_str(cfg.get("device", "cuda")),
            "--out_dir", _as_str(gfp_cfg.get("analysis_out_dir")),
        ]

        run_legacy_script(
            script_path=analyze_script,
            argv=argv,
            extra_paths=[
                legacy_cfg["scripts_dir"],
                legacy_cfg["design_dir"],
            ],
        )

    if bool(gfp_cfg.get("run_design", True)):
        design_script = design_dir / "design_gfp.py"

        argv = [
            "--in_csv", _as_str(gfp_cfg.get("input_csv")),
            "--out_dir", _as_str(gfp_cfg.get("mutation_out_dir")),
            "--max_input_rows", _as_str(gfp_cfg.get("max_input_rows", 30)),
            "--max_recommended_designs", _as_str(gfp_cfg.get("max_recommended_designs", 8)),
            "--seed", _as_str(cfg.get("seed", 42)),
        ]

        run_legacy_script(
            script_path=design_script,
            argv=argv,
            extra_paths=[
                legacy_cfg["scripts_dir"],
                legacy_cfg["design_dir"],
            ],
        )