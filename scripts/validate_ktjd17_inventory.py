#!/usr/bin/env python3
"""Validate all cross-artifact invariants of the KTJD-17 T02 inventory."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.data.ktjd17.inventory_validation import (  # noqa: E402
    validate_inventory_outputs,
    write_validation_report,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest-root", type=Path, default=REPO_ROOT / "dataset" / "manifests"
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=REPO_ROOT / "data" / "animo4d_L4TB_plus_human_v4b272neutral",
    )
    parser.add_argument(
        "--split-root", type=Path, default=REPO_ROOT / "data" / "holdout_splits_v1"
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=REPO_ROOT / "dataset" / "validation_reports" / "inventory_validation.json",
    )
    args = parser.parse_args()
    report = validate_inventory_outputs(
        args.manifest_root,
        dataset_root=args.dataset_root,
        split_root=args.split_root,
    )
    write_validation_report(
        report,
        args.report,
        immutable_manifest_root=args.manifest_root,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
