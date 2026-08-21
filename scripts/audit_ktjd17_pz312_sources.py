#!/usr/bin/env python3
"""Run the exhaustive resumable PlanetZoo 311-rig source audit."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.ktjd17.pz312_audit import (  # noqa: E402
    PzAuditConfig,
    default_pz_audit_config,
    run_pz_source_audit,
)


def main() -> None:
    defaults = default_pz_audit_config(REPO_ROOT)
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest-root", type=Path, default=Path("dataset/manifests")
    )
    parser.add_argument(
        "--pz-bvh-root", type=Path, default=Path("data/animo4d_anytop/bvhs")
    )
    parser.add_argument(
        "--active-cond",
        type=Path,
        default=Path("data/animo4d_L4TB_plus_human_v4b272neutral/cond.npy"),
    )
    parser.add_argument("--output-root", type=Path, default=Path("dataset"))
    parser.add_argument("--workers", type=int, default=defaults.workers)
    parser.add_argument("--chunk-size", type=int, default=defaults.chunk_size)
    parser.add_argument("--no-update-link", action="store_true")
    args = parser.parse_args()
    result = run_pz_source_audit(
        PzAuditConfig(
            manifest_root=args.manifest_root,
            pz_bvh_root=args.pz_bvh_root,
            active_cond_path=args.active_cond,
            output_root=args.output_root,
            workers=args.workers,
            chunk_size=args.chunk_size,
            update_link=not args.no_update_link,
        )
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
