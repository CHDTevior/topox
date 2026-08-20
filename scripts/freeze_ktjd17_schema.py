#!/usr/bin/env python3
"""Publish the reviewed immutable KTJD-17 schema and train-only gains."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.ktjd17.freeze import (  # noqa: E402
    FreezeConfig,
    default_freeze_config,
    run_freeze,
)


def main() -> int:
    defaults = default_freeze_config(ROOT)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prototype-root", type=Path, default=defaults.prototype_root)
    parser.add_argument("--fixed-qa-report", type=Path, default=defaults.fixed_qa_report)
    parser.add_argument("--calibration-root", type=Path, default=defaults.calibration_root)
    parser.add_argument("--visual-root", type=Path, default=defaults.visual_root)
    parser.add_argument("--codex-review", type=Path, default=defaults.codex_review)
    parser.add_argument("--codex-thread-id", default=defaults.codex_thread_id)
    parser.add_argument("--output-root", type=Path, default=defaults.output_root)
    parser.add_argument("--no-update-link", action="store_true")
    args = parser.parse_args()
    result = run_freeze(
        FreezeConfig(
            prototype_root=args.prototype_root,
            fixed_qa_report=args.fixed_qa_report,
            calibration_root=args.calibration_root,
            visual_root=args.visual_root,
            codex_review=args.codex_review,
            output_root=args.output_root,
            codex_thread_id=args.codex_thread_id,
            overwrite_link=not args.no_update_link,
        )
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
