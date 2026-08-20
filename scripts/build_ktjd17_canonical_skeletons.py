#!/usr/bin/env python3
"""Build KTJD-17 T04 canonical skeletons without encoding motion."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.data.ktjd17.canonical_skeleton import (  # noqa: E402
    CanonicalSkeletonConfig,
    run_canonical_skeleton_audit,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Derive fixed per-rig canonical rests from the T03 source-FK scope, "
            "write immutable skeleton NPZs, and publish a manifest-authoritative T04 generation."
        )
    )
    parser.add_argument(
        "--manifest-root",
        type=Path,
        default=REPO_ROOT / "dataset" / "manifests",
        help=(
            "Immutable direct T03 manifest generation. After T04 is active, pass "
            "dataset/.ktjd17_manifest_generations/<T03_ID> explicitly."
        ),
    )
    parser.add_argument(
        "--skeleton-output-root",
        type=Path,
        default=REPO_ROOT / "dataset" / "skeletons",
    )
    parser.add_argument(
        "--manifest-output-root",
        type=Path,
        default=REPO_ROOT / "dataset" / "manifests",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help=(
            "Publish new immutable generations, replace the authoritative manifest "
            "symlink, then update the non-authoritative skeleton compatibility symlink."
        ),
    )
    args = parser.parse_args()
    summary = run_canonical_skeleton_audit(
        CanonicalSkeletonConfig(
            manifest_root=args.manifest_root,
            skeleton_output_root=args.skeleton_output_root,
            manifest_output_root=args.manifest_output_root,
            overwrite=args.overwrite,
        )
    )
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
