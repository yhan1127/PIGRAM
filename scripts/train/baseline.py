# scripts/train/baseline.py
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def bootstrap_src_path() -> Path:
    repo_root = Path(__file__).resolve().parents[2]
    src_dir = repo_root / "src"

    if not src_dir.exists():
        raise FileNotFoundError(f"src directory not found: {src_dir}")

    src_str = str(src_dir)

    if src_str in sys.path:
        sys.path.remove(src_str)

    sys.path.insert(0, src_str)
    return repo_root


REPO_ROOT = bootstrap_src_path()

from pigram.train.baseline import train_baselines
from pigram.utils.config import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train PIGRAM baseline models.")
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to baseline config YAML.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    train_baselines(cfg)


if __name__ == "__main__":
    main()