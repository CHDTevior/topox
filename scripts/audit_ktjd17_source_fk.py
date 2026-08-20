#!/usr/bin/env python3
"""Run the KTJD-17 T03 numeric source-parser and source-FK gate."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.data.ktjd17.source_fk import (  # noqa: E402
    SourceFkConfig,
    run_source_fk_audit,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Decode the frozen T02 prototype source payloads in float64, run "
            "source-FK reproduction, and publish a new immutable manifest generation."
        )
    )
    parser.add_argument(
        "--manifest-root",
        type=Path,
        default=REPO_ROOT / "dataset" / "manifests",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=REPO_ROOT / "dataset" / "manifests",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Publish a new immutable generation and atomically replace the manifest symlink.",
    )
    args = parser.parse_args()
    summary = run_source_fk_audit(
        SourceFkConfig(
            manifest_root=args.manifest_root,
            output_root=args.output_root,
            overwrite=args.overwrite,
        )
    )
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
