# scripts/eval/downstream.py
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
from pathlib import Path

from pigram.eval.downstream import run_downstream_eval
from pigram.train.seed import set_seed
from pigram.utils.config import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate PIGRAM on downstream candidate pools."
    )

    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/downstream.yaml"),
        help="Path to downstream config.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    set_seed(int(cfg.get("seed", 42)))
    run_downstream_eval(cfg)


if __name__ == "__main__":
    main()