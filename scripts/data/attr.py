# scripts/data/attr.py
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
from pathlib import Path

from pigram.data.attr import augment_processed_attrs
from pigram.train.seed import set_seed
from pigram.utils.config import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Add precomputed 56D patch_attr and 27D motif_attr to processed datasets."
    )

    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/data.yaml"),
        help="Path to data config.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    set_seed(int(cfg.get("seed", 42)))
    augment_processed_attrs(cfg)


if __name__ == "__main__":
    main()