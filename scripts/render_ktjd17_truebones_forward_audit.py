#!/usr/bin/env python3
"""Render all per-rig Truebones KTJD-17 direction prototypes."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.ktjd17.visual_qa import render_prototype_visual_qa  # noqa: E402


def _clip_ids(root: Path) -> list[str]:
    result: list[str] = []
    with (root / "manifests/clips.jsonl").open("r", encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            result.append(str(record["clip_id"]))
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--audit-root",
        type=Path,
        default=ROOT / "dataset/ktjd17_truebones_forward_audit",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT / "dataset/ktjd17_truebones_forward_visual",
    )
    parser.add_argument("--max-gif-frames", type=int, default=24)
    parser.add_argument(
        "--no-update-link",
        action="store_true",
        help="publish an immutable generation without changing the compatibility link",
    )
    args = parser.parse_args()
    if args.max_gif_frames < 2:
        parser.error("--max-gif-frames must be at least 2")
    audit_root = args.audit_root.expanduser().resolve()
    result = render_prototype_visual_qa(
        prototype_root=audit_root,
        calibration_root=None,
        output_root=args.output_root,
        clip_ids=_clip_ids(audit_root),
        max_gif_frames=args.max_gif_frames,
        update_link=not args.no_update_link,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
