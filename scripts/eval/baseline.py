# scripts/eval/baseline.py
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
from pathlib import Path

from pigram.eval.baseline import run_downstream_baseline_eval
from pigram.train.seed import set_seed
from pigram.utils.config import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate downstream baselines against PIGRAM."
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
    run_downstream_baseline_eval(cfg)


if __name__ == "__main__":
    main()