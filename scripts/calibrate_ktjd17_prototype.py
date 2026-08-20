#!/usr/bin/env python3
"""Run immutable train-only calibration over the accepted KTJD-17 prototype."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.ktjd17.calibration import run_prototype_calibration  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--prototype-root", type=Path, default=ROOT / "dataset/ktjd17_prototype"
    )
    parser.add_argument(
        "--fixed-qa-report",
        type=Path,
        default=ROOT / "scratch/ktjd17_t08_fixed_qa.json",
    )
    parser.add_argument("--output-root", type=Path, default=ROOT / "dataset")
    args = parser.parse_args()
    result = run_prototype_calibration(
        prototype_root=args.prototype_root,
        fixed_qa_report=args.fixed_qa_report,
        output_root=args.output_root,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
