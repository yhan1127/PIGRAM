# scripts/analysis/fine_fig.py
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
from pathlib import Path

from pigram.analysis.fine_fig import run_fine_refinement_figure
from pigram.utils.config import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Make fine-refinement composition figure.")
    parser.add_argument("--config", type=Path, default=Path("configs/analysis.yaml"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    run_fine_refinement_figure(cfg)


if __name__ == "__main__":
    main()