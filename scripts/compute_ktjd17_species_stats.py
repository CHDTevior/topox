#!/usr/bin/env python3
"""Compute all-clip KTJD-17 mean/std for each biological species."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.ktjd17.species_stats import compute_species_stats  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--generation",
        type=Path,
        default=Path("dataset/ktjd17_pz_human312"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("dataset/ktjd17_pz_human312_species_stats"),
    )
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--shard-size", type=int, default=256)
    args = parser.parse_args()
    result = compute_species_stats(
        args.generation,
        args.output,
        workers=args.workers,
        shard_size=args.shard_size,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
