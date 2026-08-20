#!/usr/bin/env python3
"""Build the frozen source-safe Truebones KTJD-17 dataset."""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.ktjd17.truebones_full_build import (  # noqa: E402
    default_full_build_config,
    run_truebones_full_build,
)


def main() -> int:
    defaults = default_full_build_config(ROOT)
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest-root", type=Path, default=defaults.manifest_root)
    parser.add_argument("--freeze-root", type=Path, default=defaults.freeze_root)
    parser.add_argument(
        "--forward-audit-root", type=Path, default=defaults.forward_audit_root
    )
    parser.add_argument("--visual-root", type=Path, default=defaults.visual_root)
    parser.add_argument(
        "--visual-gate", type=Path, default=defaults.visual_gate_path
    )
    parser.add_argument("--output-root", type=Path, default=defaults.output_root)
    parser.add_argument("--active-cond", type=Path, default=defaults.active_cond_path)
    parser.add_argument("--legacy-cond", type=Path, default=defaults.legacy_cond_path)
    parser.add_argument("--no-update-link", action="store_true")
    args = parser.parse_args()
    config = dataclasses.replace(
        defaults,
        manifest_root=args.manifest_root,
        freeze_root=args.freeze_root,
        forward_audit_root=args.forward_audit_root,
        visual_root=args.visual_root,
        visual_gate_path=args.visual_gate,
        output_root=args.output_root,
        active_cond_path=args.active_cond,
        legacy_cond_path=args.legacy_cond,
        update_link=not args.no_update_link,
    )
    result = run_truebones_full_build(config)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["conversion_complete"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
