# scripts/design/nelf.py
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
from pathlib import Path

from pigram.design.nelf import run_nelf_design
from pigram.train.seed import set_seed
from pigram.utils.config import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate NELF favorable Rule-4 mutation designs.")
    parser.add_argument("--config", type=Path, default=Path("configs/design.yaml"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    set_seed(int(cfg.get("seed", 42)))
    run_nelf_design(cfg)


if __name__ == "__main__":
    main()