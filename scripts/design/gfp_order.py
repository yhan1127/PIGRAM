# scripts/design/gfp_order.py
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
from pathlib import Path

from pigram.design.gfp_order import run_gfp_ordering
from pigram.utils.config import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build GFP final oligo ordering table.")
    parser.add_argument("--config", type=Path, default=Path("configs/design.yaml"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    run_gfp_ordering(cfg)


if __name__ == "__main__":
    main()