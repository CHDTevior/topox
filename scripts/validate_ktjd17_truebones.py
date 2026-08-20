#!/usr/bin/env python3
"""Run source-backed fixed QA over a full immutable KTJD-17 generation."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.ktjd17.fixed_qa import validate_prototype  # noqa: E402
from src.data.ktjd17.truebones_full_build import (  # noqa: E402
    verify_full_generation,
)


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
        descriptor = os.open(target.parent, os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    finally:
        if temporary.exists():
            temporary.unlink()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-root", type=Path, default=ROOT / "dataset/ktjd17_truebones"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "dataset/validation_reports/ktjd17_truebones_fixed_qa.json",
    )
    args = parser.parse_args()
    generation = verify_full_generation(args.dataset_root, require_complete=False)
    report = validate_prototype(args.dataset_root)
    report["artifact_kind"] = "full_truebones_dataset"
    report["full_build_version"] = generation["full_build_version"]
    report["full_generation_json_sha256"] = hashlib.sha256(
        (args.dataset_root.expanduser().resolve() / "generation.json").read_bytes()
    ).hexdigest()
    conversion_complete = generation.get("conversion_complete") is True
    report["full_conversion_gate"] = {
        "status": "pass" if conversion_complete else "fail",
        "conversion_complete": conversion_complete,
        "full_conversion_authorized": generation.get("full_conversion_authorized")
        is True,
        "required_source_safe_clip_count": 986,
        "observed_clip_count": report["clip_count"],
    }
    if not conversion_complete:
        report["status"] = "fail"
    _write_json_atomic(args.output, report)
    summary = {
        key: report[key]
        for key in (
            "status",
            "generation_id",
            "clip_count",
            "pass_count",
            "fail_count",
            "skeleton_count",
            "J_phys_max",
            "T_max_observed",
        )
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"report={args.output.expanduser().absolute()}")
    return 0 if report["status"] == "pass" and conversion_complete else 2


if __name__ == "__main__":
    raise SystemExit(main())
