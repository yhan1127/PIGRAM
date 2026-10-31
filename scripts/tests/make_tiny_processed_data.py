# scripts/tests/make_tiny_processed_data.py
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import pickle
import random
import sys
from pathlib import Path
from typing import Any, Dict, List


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

from pigram.utils.io import load_pickle


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a small processed example dataset for supplementary release."
    )
    parser.add_argument("--train_pkl", type=Path, default=Path("data/processed/train.pkl"))
    parser.add_argument("--val_pkl", type=Path, default=Path("data/processed/val.pkl"))
    parser.add_argument("--test_pkl", type=Path, default=Path("data/processed/test.pkl"))
    parser.add_argument("--out_dir", type=Path, default=Path("data/example"))
    parser.add_argument("--n_train", type=int, default=80)
    parser.add_argument("--n_val", type=int, default=10)
    parser.add_argument("--n_test", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def sample_subset(samples: List[Dict[str, Any]], n: int, seed: int) -> List[Dict[str, Any]]:
    if n < 0 or n >= len(samples):
        return samples
    rng = random.Random(seed)
    idxs = list(range(len(samples)))
    rng.shuffle(idxs)
    idxs = sorted(idxs[:n])
    return [samples[i] for i in idxs]


def save_pickle(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def file_mb(path: Path) -> float:
    return path.stat().st_size / 1024 / 1024


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    train = load_pickle(args.train_pkl)
    val = load_pickle(args.val_pkl)
    test = load_pickle(args.test_pkl)

    train_tiny = sample_subset(train, args.n_train, args.seed)
    val_tiny = sample_subset(val, args.n_val, args.seed + 1)
    test_tiny = sample_subset(test, args.n_test, args.seed + 2)

    out_train = args.out_dir / "train_tiny.pkl"
    out_val = args.out_dir / "val_tiny.pkl"
    out_test = args.out_dir / "test_tiny.pkl"

    save_pickle(train_tiny, out_train)
    save_pickle(val_tiny, out_val)
    save_pickle(test_tiny, out_test)

    summary = {
        "train_tiny": {
            "path": str(out_train),
            "num_samples": len(train_tiny),
            "size_mb": round(file_mb(out_train), 3),
        },
        "val_tiny": {
            "path": str(out_val),
            "num_samples": len(val_tiny),
            "size_mb": round(file_mb(out_val), 3),
        },
        "test_tiny": {
            "path": str(out_test),
            "num_samples": len(test_tiny),
            "size_mb": round(file_mb(out_test), 3),
        },
    }

    summary_path = args.out_dir / "summary_tiny.json"
    summary_path.write_text(
        __import__("json").dumps(summary, indent=2),
        encoding="utf-8",
    )

    print("Saved tiny processed data:")
    print(summary)


if __name__ == "__main__":
    main()