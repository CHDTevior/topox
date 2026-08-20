#!/usr/bin/env python3
"""Build the KTJD-17 T02 live raw-source inventory.

This command is read-only with respect to the current BTJD/raw datasets.  It
only writes versioned inventory artifacts under ``dataset/manifests`` (or an
explicit ``--output-root``).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.data.ktjd17.inventory import (  # noqa: E402
    InventoryConfig,
    default_inventory_config,
    run_inventory,
)


def _parser() -> argparse.ArgumentParser:
    defaults = default_inventory_config(REPO_ROOT)
    parser = argparse.ArgumentParser(
        description=(
            "Inventory live BTJD-13 clips against BVH/MotionStreamer rotation "
            "authorities without decoding legacy 13D rotations."
        )
    )
    parser.add_argument("--dataset-root", type=Path, default=defaults.dataset_root)
    parser.add_argument("--split-root", type=Path, default=defaults.split_root)
    parser.add_argument("--pz-bvh-root", type=Path, default=defaults.pz_bvh_root)
    parser.add_argument(
        "--truebones-raw-root", type=Path, default=defaults.truebones_raw_root
    )
    parser.add_argument("--human272-root", type=Path, default=defaults.human272_root)
    parser.add_argument("--output-root", type=Path, default=defaults.output_root)
    parser.add_argument(
        "--human-builder-path", type=Path, default=defaults.human_builder_path
    )
    parser.add_argument(
        "--smpl-neutral-model-path",
        type=Path,
        default=defaults.smpl_neutral_model_path,
    )
    parser.add_argument(
        "--planetzoo-lineage-path",
        type=Path,
        default=defaults.planetzoo_lineage_path,
    )
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--prototype-min-train-clips", type=int, default=30)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Atomically replace prior inventory artifacts after deliberate review.",
    )
    return parser


def main() -> int:
    args = _parser().parse_args()
    config = InventoryConfig(
        dataset_root=args.dataset_root,
        split_root=args.split_root,
        pz_bvh_root=args.pz_bvh_root,
        truebones_raw_root=args.truebones_raw_root,
        human272_root=args.human272_root,
        output_root=args.output_root,
        human_builder_path=args.human_builder_path,
        smpl_neutral_model_path=args.smpl_neutral_model_path,
        planetzoo_lineage_path=args.planetzoo_lineage_path,
        workers=args.workers,
        overwrite=args.overwrite,
        prototype_min_train_clips=args.prototype_min_train_clips,
    )
    summary = run_inventory(config)
    print(json.dumps(summary["fresh_counts"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
