# scripts/analysis/case.py
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
from pathlib import Path

from pigram.analysis.case import run_dense_case_figure
from pigram.train.seed import set_seed
from pigram.utils.config import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Make dense matrix case-study figure.")
    parser.add_argument("--config", type=Path, default=Path("configs/analysis.yaml"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    set_seed(int(cfg.get("seed", 42)))
    run_dense_case_figure(cfg)


if __name__ == "__main__":
    main()