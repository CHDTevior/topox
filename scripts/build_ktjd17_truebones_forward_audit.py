#!/usr/bin/env python3
"""Build the 66-rig source-backed KTJD-17 direction visual prototype."""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.ktjd17.truebones_forward_audit import (  # noqa: E402
    default_forward_audit_config,
    run_truebones_forward_audit,
)


def main() -> int:
    defaults = default_forward_audit_config(ROOT)
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest-root", type=Path, default=defaults.manifest_root)
    parser.add_argument("--freeze-root", type=Path, default=defaults.freeze_root)
    parser.add_argument("--output-root", type=Path, default=defaults.output_root)
    parser.add_argument("--active-cond", type=Path, default=defaults.active_cond_path)
    parser.add_argument("--legacy-cond", type=Path, default=defaults.legacy_cond_path)
    parser.add_argument("--no-update-link", action="store_true")
    args = parser.parse_args()
    config = dataclasses.replace(
        defaults,
        manifest_root=args.manifest_root,
        freeze_root=args.freeze_root,
        output_root=args.output_root,
        active_cond_path=args.active_cond,
        legacy_cond_path=args.legacy_cond,
        overwrite_link=not args.no_update_link,
    )
    result = run_truebones_forward_audit(config)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
