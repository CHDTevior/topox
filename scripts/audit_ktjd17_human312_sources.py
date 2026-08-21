#!/usr/bin/env python3
"""Audit all MotionStreamer272 clips for the Human member of PZ+Human-312."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.ktjd17.human312_audit import (  # noqa: E402
    default_human_audit_config,
    run_human_source_audit,
    validate_active_human_audit,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--chunk-size", type=int, default=32)
    parser.add_argument(
        "--validate-active",
        action="store_true",
        help="validate the active generation/approval and re-hash every source",
    )
    parser.add_argument(
        "--no-update-link",
        action="store_true",
        help="publish an approved immutable generation without changing active links",
    )
    args = parser.parse_args()
    config = default_human_audit_config(args.repo_root)
    if args.validate_active:
        result = validate_active_human_audit(config.output_root, rehash_sources=True)
    else:
        config = config.__class__(
            **{
                **config.__dict__,
                "workers": args.workers,
                "chunk_size": args.chunk_size,
                "update_link": not args.no_update_link,
            }
        )
        result = run_human_source_audit(config)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
