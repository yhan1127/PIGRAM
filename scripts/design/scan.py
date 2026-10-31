# scripts/design/scan.py
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
from pathlib import Path

from pigram.design.scan import run_design_scan_summary
from pigram.utils.config import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize all-target grammar-guided design scan.")
    parser.add_argument("--config", type=Path, default=Path("configs/design.yaml"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    run_design_scan_summary(cfg)


if __name__ == "__main__":
    main()