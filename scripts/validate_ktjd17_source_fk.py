#!/usr/bin/env python3
"""Independently reparse and validate KTJD-17 T03 source-FK evidence."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.data.ktjd17.source_fk_validation import (  # noqa: E402
    validate_source_fk_outputs,
    write_source_fk_validation_report,
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
        default=REPO_ROOT / "dataset" / "validation_reports" / "source_fk_validation.json",
    )
    args = parser.parse_args()
    report = validate_source_fk_outputs(args.manifest_root)
    write_source_fk_validation_report(
        report,
        args.report,
        immutable_manifest_root=args.manifest_root,
    )
    print(json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
