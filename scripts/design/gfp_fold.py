# scripts/design/gfp_fold.py
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
from pathlib import Path

from pigram.design.gfp_fold import run_gfp_fold_validation
from pigram.utils.config import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate GFP mutation designs with RNAfold.")
    parser.add_argument("--config", type=Path, default=Path("configs/design.yaml"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    run_gfp_fold_validation(cfg)


if __name__ == "__main__":
    main()