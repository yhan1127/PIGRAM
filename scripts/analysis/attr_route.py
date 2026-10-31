# scripts/analysis/attr_route.py
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
from pathlib import Path

from pigram.analysis.attr_route import run_attr_route_figure
from pigram.utils.config import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Make attribute-level route signature figure.")
    parser.add_argument("--config", type=Path, default=Path("configs/analysis.yaml"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    run_attr_route_figure(cfg)


if __name__ == "__main__":
    main()