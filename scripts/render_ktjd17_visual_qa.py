#!/usr/bin/env python3
"""Render synchronized source/direct/FK KTJD-17 prototype visual QA."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.ktjd17.visual_qa import render_prototype_visual_qa  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--prototype-root", type=Path, default=ROOT / "dataset/ktjd17_prototype"
    )
    parser.add_argument(
        "--calibration-root",
        type=Path,
        default=ROOT / "dataset/ktjd17_calibration_candidate",
    )
    parser.add_argument("--output-root", type=Path, default=ROOT / "dataset")
    parser.add_argument("--clip-id", action="append", default=[])
    parser.add_argument("--max-gif-frames", type=int, default=36)
    args = parser.parse_args()
    if args.max_gif_frames < 2:
        parser.error("--max-gif-frames must be at least 2")
    result = render_prototype_visual_qa(
        prototype_root=args.prototype_root,
        calibration_root=args.calibration_root,
        output_root=args.output_root,
        clip_ids=args.clip_id or None,
        max_gif_frames=args.max_gif_frames,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
