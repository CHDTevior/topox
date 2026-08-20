#!/usr/bin/env python3
"""Run read-only, source-backed fixed QA on an immutable KTJD-17 prototype."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.ktjd17.fixed_qa import validate_prototype  # noqa: E402


def _write_json_atomic(path: Path, value: object) -> None:
    target = path.expanduser().absolute()
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
        parent_fd = os.open(target.parent, os.O_DIRECTORY)
        try:
            os.fsync(parent_fd)
        finally:
            os.close(parent_fd)
    finally:
        if temporary.exists():
            temporary.unlink()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--prototype-root", type=Path, default=ROOT / "dataset/ktjd17_prototype"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "scratch/ktjd17_t08_fixed_qa.json",
    )
    args = parser.parse_args()
    report = validate_prototype(args.prototype_root)
    _write_json_atomic(args.output, report)
    print(
        json.dumps(
            {
                key: report[key]
                for key in (
                    "status",
                    "generation_id",
                    "clip_count",
                    "pass_count",
                    "fail_count",
                    "calibration_eligible_pass_count",
                    "held_read_only_pass_count",
                    "skeleton_count",
                    "J_phys_max",
                    "T_max_observed",
                )
            },
            indent=2,
            sort_keys=True,
        )
    )
    print(f"report={args.output.expanduser().absolute()}")
    return 0 if report["status"] == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
