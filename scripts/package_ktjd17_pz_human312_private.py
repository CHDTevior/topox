#!/usr/bin/env python3
"""Create private Hugging Face tar shards for the full PZ/Human KTJD-17 data."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.ktjd17.pz_human_private_release import (  # noqa: E402
    PzHumanPrivateReleaseError,
    package_private_release,
)


def _relative(value: Path, label: str) -> Path:
    if value.is_absolute() or ".." in value.parts:
        raise PzHumanPrivateReleaseError(f"{label} must be repository-relative")
    return ROOT / value


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--generation", type=Path, default=Path("dataset/ktjd17_pz_human312"))
    parser.add_argument(
        "--species-stats",
        type=Path,
        default=Path("dataset/ktjd17_pz_human312_species_stats"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("dataset/ktjd17_pz_human312_private_release"),
    )
    parser.add_argument("--max-shard-mib", type=int, default=512)
    args = parser.parse_args()
    try:
        result = package_private_release(
            _relative(args.generation, "--generation"),
            _relative(args.species_stats, "--species-stats"),
            _relative(args.output, "--output"),
            max_shard_bytes=args.max_shard_mib * 1024 * 1024,
        )
    except PzHumanPrivateReleaseError as exc:
        parser.error(str(exc))
    display = dict(result)
    display["output_root"] = Path(result["output_root"]).relative_to(ROOT).as_posix()
    print(json.dumps(display, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
