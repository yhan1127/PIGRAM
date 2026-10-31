# src/pigram/analysis/runner.py
# -*- coding: utf-8 -*-

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import List


def add_path(path: str | Path) -> None:
    path = Path(path).resolve()

    if not path.exists():
        raise FileNotFoundError(f"Path not found: {path}")

    path_str = str(path)

    if path_str not in sys.path:
        sys.path.insert(0, path_str)


def run_legacy_script(
    script_path: str | Path,
    argv: List[str],
    extra_paths: List[str | Path] | None = None,
) -> None:
    script_path = Path(script_path).resolve()

    if not script_path.exists():
        raise FileNotFoundError(f"Legacy script not found: {script_path}")

    if extra_paths:
        for path in extra_paths:
            add_path(path)

    add_path(script_path.parent)

    module_name = script_path.stem + "_legacy_runner"
    spec = importlib.util.spec_from_file_location(module_name, script_path)

    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load legacy script: {script_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    if not hasattr(module, "main"):
        raise AttributeError(f"Legacy script has no main() function: {script_path}")

    old_argv = sys.argv[:]

    try:
        sys.argv = [str(script_path)] + argv
        module.main()
    finally:
        sys.argv = old_argv