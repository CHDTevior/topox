#!/usr/bin/env python3
"""Build the approved PZ-311 plus Human-1 KTJD-17 prototype or full corpus."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.ktjd17.pz_human312_build import (  # noqa: E402
    BuildConfig,
    run_build,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("prototype", "full"), default="prototype")
    parser.add_argument("--dataset-root", type=Path, default=Path("dataset"))
    parser.add_argument(
        "--freeze-root", type=Path, default=Path("dataset/ktjd17_freeze")
    )
    parser.add_argument("--output-root", type=Path, default=Path("dataset"))
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--source-rehash-workers", type=int, default=16)
    parser.add_argument("--visual-gate", type=Path)
    parser.add_argument(
        "--anomaly-allowlist",
        type=Path,
        help=(
            "optional read-only, content-addressed gpt-5.6-sol/xhigh-reviewed "
            "exact anomaly allowlist; omission means zero rejections"
        ),
    )
    parser.add_argument("--no-update-link", action="store_true")
    args = parser.parse_args()
    if args.mode == "full" and args.visual_gate is None:
        parser.error("--mode full requires --visual-gate")
    result = run_build(
        BuildConfig(
            dataset_root=args.dataset_root,
            freeze_root=args.freeze_root,
            output_root=args.output_root,
            mode=args.mode,
            workers=args.workers,
            source_rehash_workers=args.source_rehash_workers,
            visual_gate_path=args.visual_gate,
            anomaly_allowlist_path=args.anomaly_allowlist,
            update_link=not args.no_update_link,
        )
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
