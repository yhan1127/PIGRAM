# src/pigram/analysis/pymol.py
# -*- coding: utf-8 -*-

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from pigram.analysis.runner import run_legacy_script


def _as_str(x: Any) -> str:
    return "" if x is None else str(x)


def run_pymol_export(cfg: Dict[str, Any]) -> None:
    legacy_cfg = cfg["legacy"]
    paths = cfg["paths"]
    structure_cfg = cfg["structure"]

    script_path = Path(legacy_cfg["analysis_dir"]) / "export_structure_pymol.py"

    run_dir = Path(structure_cfg.get("run_dir", "runs/pigram"))

    argv: List[str] = [
        "--run_dir", _as_str(run_dir),
        "--checkpoint_name", _as_str(structure_cfg.get("checkpoint_name", "best_by_composite.pt")),
        "--config_json", _as_str(structure_cfg.get("config_json", "runs/pigram/config.json")),
        "--summary_json", _as_str(structure_cfg.get("summary_json", "runs/pigram/summary.json")),

        "--foundation_ckpt", _as_str(paths["foundation_ckpt"]),
        "--distill_plan_json", _as_str(paths["distill_plan_json"]),
        "--test_pkl", _as_str(paths["test_pkl"]),
        "--test_ann_pkl", _as_str(paths["test_ann_pkl"]),

        "--structures_dir", _as_str(structure_cfg.get("structures_dir", "structures")),
        "--out_dir", _as_str(structure_cfg.get("pymol_out_dir", "results/structure/pymol")),
        "--device", _as_str(cfg.get("device", "cuda")),
        "--sigmoid_scale", _as_str(structure_cfg.get("sigmoid_scale", -1.0)),

        "--top_motifs", _as_str(structure_cfg.get("top_motifs", 2)),
        "--route_vmax", _as_str(structure_cfg.get("route_vmax", -1.0)),
        "--correction_vlim", _as_str(structure_cfg.get("correction_vlim", -1.0)),
        "--image_width", _as_str(structure_cfg.get("image_width", 1800)),
        "--image_height", _as_str(structure_cfg.get("image_height", 1400)),
    ]

    checkpoint_path = structure_cfg.get("checkpoint_path", "")
    if checkpoint_path:
        argv.extend(["--checkpoint_path", _as_str(checkpoint_path)])

    motifs = structure_cfg.get("motifs", "")
    if motifs:
        argv.extend(["--motifs", _as_str(motifs)])

    cases = structure_cfg.get("cases", [])
    if cases:
        argv.append("--cases")
        argv.extend([_as_str(x) for x in cases])

    run_legacy_script(
        script_path=script_path,
        argv=argv,
        extra_paths=[
            legacy_cfg["scripts_dir"],
            legacy_cfg["analysis_dir"],
        ],
    )