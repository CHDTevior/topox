#!/usr/bin/env python3
"""Render the reviewed 66-rig sample from the full KTJD-17 dataset."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.ktjd17.truebones_full_build import (  # noqa: E402
    default_full_build_config,
    reviewed_representative_clip_ids,
    validate_visual_gate,
    verify_full_generation,
)
from src.data.ktjd17.visual_qa import render_prototype_visual_qa  # noqa: E402


def main() -> int:
    defaults = default_full_build_config(ROOT)
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-root", type=Path, default=ROOT / "dataset/ktjd17_truebones"
    )
    parser.add_argument(
        "--forward-audit-root", type=Path, default=defaults.forward_audit_root
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT / "dataset/ktjd17_truebones_visual",
    )
    parser.add_argument("--max-gif-frames", type=int, default=24)
    parser.add_argument("--no-update-link", action="store_true")
    args = parser.parse_args()
    if args.max_gif_frames < 2:
        parser.error("--max-gif-frames must be at least 2")
    dataset_root = args.dataset_root.expanduser().resolve()
    verify_full_generation(dataset_root)
    forward_audit_root = args.forward_audit_root.expanduser().resolve()
    validate_visual_gate(
        gate_path=defaults.visual_gate_path,
        visual_root=defaults.visual_root,
        forward_audit_root=forward_audit_root,
    )
    clip_ids = reviewed_representative_clip_ids(forward_audit_root)
    result = render_prototype_visual_qa(
        prototype_root=dataset_root,
        calibration_root=None,
        output_root=args.output_root,
        clip_ids=clip_ids,
        max_gif_frames=args.max_gif_frames,
        update_link=not args.no_update_link,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
