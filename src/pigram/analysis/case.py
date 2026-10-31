# src/pigram/analysis/case.py
# -*- coding: utf-8 -*-

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from pigram.analysis.runner import run_legacy_script


def _as_str(x: Any) -> str:
    return "" if x is None else str(x)


def run_dense_case_figure(cfg: Dict[str, Any]) -> None:
    legacy_cfg = cfg["legacy"]
    paths = cfg["paths"]
    case_cfg = cfg["case"]

    script_path = Path(legacy_cfg["analysis_dir"]) / "make_dense_case.py"

    out_dir = Path(paths["figure_out_dir"]) / "case"

    argv: List[str] = [
        "--run_dir", _as_str(case_cfg.get("run_dir", "runs/pigram")),
        "--checkpoint_name", _as_str(case_cfg.get("checkpoint_name", "best_by_composite.pt")),
        "--config", _as_str(case_cfg.get("config_path", "runs/pigram/config.json")),
        "--summary", _as_str(case_cfg.get("summary_path", "runs/pigram/summary.json")),
        "--out_dir", _as_str(out_dir),

        "--foundation_ckpt", _as_str(paths["foundation_ckpt"]),
        "--distill_plan_json", _as_str(paths["distill_plan_json"]),
        "--test_pkl", _as_str(paths["test_pkl"]),
        "--test_ann_pkl", _as_str(paths["test_ann_pkl"]),

        "--easy_sample", _as_str(case_cfg.get("easy_sample", 234)),
        "--rescued_sample", _as_str(case_cfg.get("rescued_sample", 172)),
        "--sigmoid_scale", _as_str(case_cfg.get("sigmoid_scale", 0.0)),

        "--use_recommended_windows", _as_str(case_cfg.get("use_recommended_windows", 1)),
        "--keep_all_motifs", _as_str(case_cfg.get("keep_all_motifs", 1)),

        "--zoom_rows_easy", _as_str(case_cfg.get("zoom_rows_easy", 80)),
        "--zoom_rows_rescued", _as_str(case_cfg.get("zoom_rows_rescued", 80)),
        "--top_k_boxes", _as_str(case_cfg.get("top_k_boxes", 8)),

        "--easy_row_start", _as_str(case_cfg.get("easy_row_start", -1)),
        "--easy_row_end", _as_str(case_cfg.get("easy_row_end", -1)),
        "--easy_col_start", _as_str(case_cfg.get("easy_col_start", -1)),
        "--easy_col_end", _as_str(case_cfg.get("easy_col_end", -1)),

        "--rescued_row_start", _as_str(case_cfg.get("rescued_row_start", -1)),
        "--rescued_row_end", _as_str(case_cfg.get("rescued_row_end", -1)),
        "--rescued_col_start", _as_str(case_cfg.get("rescued_col_start", -1)),
        "--rescued_col_end", _as_str(case_cfg.get("rescued_col_end", -1)),

        "--route_vmax", _as_str(case_cfg.get("route_vmax", 0.0)),
        "--correction_vlim", _as_str(case_cfg.get("correction_vlim", 0.0)),
        "--device", _as_str(cfg.get("device", "cuda")),
    ]

    run_legacy_script(
        script_path=script_path,
        argv=argv,
        extra_paths=[
            legacy_cfg["scripts_dir"],
            legacy_cfg["analysis_dir"],
        ],
    )