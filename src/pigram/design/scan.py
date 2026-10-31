# src/pigram/design/scan.py
# -*- coding: utf-8 -*-

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any, Dict, List


def _load_module(script_path: Path):
    script_path = Path(script_path).resolve()

    if not script_path.exists():
        raise FileNotFoundError(f"Legacy design summary script not found: {script_path}")

    script_dir = str(script_path.parent)
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)

    module_name = script_path.stem + "_design_scan_runner"
    spec = importlib.util.spec_from_file_location(module_name, script_path)

    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load legacy script: {script_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return module


def run_design_scan_summary(cfg: Dict[str, Any]) -> None:
    legacy_cfg = cfg["legacy"]
    paths_cfg = cfg["paths"]
    scan_cfg = cfg["scan"]

    script_path = Path(legacy_cfg["design_dir"]) / "summarize_targets.py"
    module = _load_module(script_path)

    scan_root = Path(paths_cfg["all_target_scan_root"])
    out_dir = Path(paths_cfg["all_target_summary_dir"])
    targets: List[str] = [str(x).upper() for x in scan_cfg.get("targets", [])]

    out_dir.mkdir(parents=True, exist_ok=True)

    module.SCAN_ROOT = scan_root
    module.OUT_DIR = out_dir
    module.TARGETS = targets

    module.main()