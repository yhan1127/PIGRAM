# src/pigram/analysis/rule_attr.py
# -*- coding: utf-8 -*-

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from pigram.analysis.runner import run_legacy_script


def _as_str(x: Any) -> str:
    return "" if x is None else str(x)


def run_rule_attr_figure(cfg: Dict[str, Any]) -> None:
    legacy_cfg = cfg["legacy"]
    rule_cfg = cfg["rule_attr"]

    script_path = Path(legacy_cfg["analysis_dir"]) / "make_rule_attr.py"

    argv: List[str] = [
        "--run_dir", _as_str(rule_cfg.get("run_dir", "runs/pigram")),
        "--checkpoint_name", _as_str(rule_cfg.get("checkpoint_name", "best_by_composite.pt")),
        "--config_json", _as_str(rule_cfg.get("config_json", "runs/pigram/config.json")),
        "--out_dir", _as_str(rule_cfg.get("out_dir", "results/figures/rule_attr")),
        "--device", _as_str(cfg.get("device", "cuda")),
        "--n_rules", _as_str(rule_cfg.get("n_rules", 7)),
        "--top_rules", _as_str(rule_cfg.get("top_rules", 7)),
        "--rule_ids", _as_str(rule_cfg.get("rule_ids", "0,1,2,3,4,5,6")),
        "--max_cases", _as_str(rule_cfg.get("max_cases", 0)),
    ]

    checkpoint_path = rule_cfg.get("checkpoint_path", "")
    if checkpoint_path:
        argv.extend(["--checkpoint_path", _as_str(checkpoint_path)])

    if bool(rule_cfg.get("require_forward", False)):
        argv.append("--require_forward")

    if bool(rule_cfg.get("annotate", False)):
        argv.append("--annotate")

    run_legacy_script(
        script_path=script_path,
        argv=argv,
        extra_paths=[
            legacy_cfg["scripts_dir"],
            legacy_cfg["analysis_dir"],
        ],
    )