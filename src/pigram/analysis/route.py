# src/pigram/analysis/route.py
# -*- coding: utf-8 -*-

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from pigram.analysis.runner import run_legacy_script


def _as_str(x: Any) -> str:
    return "" if x is None else str(x)


def run_route_analysis(cfg: Dict[str, Any]) -> None:
    legacy_cfg = cfg["legacy"]
    paths = cfg["paths"]
    route = cfg["route"]

    script_path = Path(legacy_cfg["analysis_dir"]) / "analyze_budgeted_rescue_route_atlas.py"

    argv: List[str] = [
        "--checkpoint_path", _as_str(paths["checkpoint_path"]),
        "--foundation_ckpt", _as_str(paths["foundation_ckpt"]),
        "--distill_plan_json", _as_str(paths["distill_plan_json"]),
        "--out_dir", _as_str(paths["route_out_dir"]),

        "--val_pkl", _as_str(paths["val_pkl"]),
        "--test_pkl", _as_str(paths["test_pkl"]),
        "--val_ann_pkl", _as_str(paths["val_ann_pkl"]),
        "--test_ann_pkl", _as_str(paths["test_ann_pkl"]),

        "--splits", _as_str(route.get("splits", "test,val")),
        "--sigmoid_scale", _as_str(route.get("sigmoid_scale", 2.0)),
        "--high_route_quantile", _as_str(route.get("high_route_quantile", 0.90)),
        "--hard_quantile", _as_str(route.get("hard_quantile", 0.75)),
        "--easy_quantile", _as_str(route.get("easy_quantile", 0.25)),
        "--min_rescue_delta", _as_str(route.get("min_rescue_delta", 0.01)),
        "--min_easy_harm_delta", _as_str(route.get("min_easy_harm_delta", 0.005)),
        "--top_n_pairs", _as_str(route.get("top_n_pairs", 200)),
        "--top_attr_n", _as_str(route.get("top_attr_n", 10)),
        "--device", _as_str(cfg.get("device", "cuda")),
    ]

    if route.get("patch_attr_names_json", ""):
        argv.extend(["--patch_attr_names_json", _as_str(route["patch_attr_names_json"])])

    if route.get("motif_attr_names_json", ""):
        argv.extend(["--motif_attr_names_json", _as_str(route["motif_attr_names_json"])])

    run_legacy_script(
        script_path=script_path,
        argv=argv,
        extra_paths=[
            legacy_cfg["scripts_dir"],
            legacy_cfg["analysis_dir"],
        ],
    )