#!/usr/bin/env python3
"""Independently validate KTJD-17 T04 canonical skeleton evidence."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.data.ktjd17.canonical_skeleton_validation import (  # noqa: E402
    validate_canonical_skeleton_outputs,
    write_canonical_skeleton_validation_report,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest-root",
        type=Path,
        default=REPO_ROOT / "dataset" / "manifests",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=(
            REPO_ROOT
            / "dataset"
            / "validation_reports"
            / "canonical_skeleton_validation.json"
        ),
    )
    args = parser.parse_args()
    immutable_manifest = args.manifest_root.expanduser().resolve()
    report = validate_canonical_skeleton_outputs(immutable_manifest)
    write_canonical_skeleton_validation_report(
        report,
        args.report,
        immutable_manifest_root=immutable_manifest,
    )
    print(json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
