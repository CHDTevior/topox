#!/usr/bin/env python3
"""Tile all 312 three-path filmstrips into reviewable overview sheets."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

def _font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for candidate in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf",
    ):
        try:
            return ImageFont.truetype(candidate, size)
        except OSError:
            pass
    return ImageFont.load_default()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "visual_root",
        nargs="?",
        type=Path,
        default=Path("dataset/ktjd17_pz_human312_visual/ktjd17_visual_qa"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("scratch/ktjd17_pz_human312_review_sheets"),
    )
    parser.add_argument("--per-sheet", type=int, default=12)
    parser.add_argument("--columns", type=int, default=3)
    args = parser.parse_args()
    if args.per_sheet <= 0 or args.columns <= 0:
        parser.error("--per-sheet and --columns must be positive")
    visual_root = args.visual_root.expanduser().resolve()
    index = json.loads((visual_root / "visual_qa_index.json").read_text("utf-8"))
    records = sorted(index["clips"], key=lambda record: str(record["rig_id"]))
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    tile_width, tile_height, label_height = 450, 193, 24
    rows = math.ceil(args.per_sheet / args.columns)
    files: list[str] = []
    for start in range(0, len(records), args.per_sheet):
        batch = records[start : start + args.per_sheet]
        sheet = Image.new(
            "RGB",
            (tile_width * args.columns, (tile_height + label_height) * rows + 34),
            (5, 8, 13),
        )
        draw = ImageDraw.Draw(sheet)
        draw.text(
            (8, 7),
            (
                f"KTJD-17 312-rig visual QA {start + 1:03d}-"
                f"{start + len(batch):03d} | rows in every tile: "
                "source / position-direct / rotation-FK"
            ),
            fill=(245, 245, 245),
            font=_font(14),
        )
        for offset, record in enumerate(batch):
            row, column = divmod(offset, args.columns)
            x = column * tile_width
            y = 34 + row * (tile_height + label_height)
            label = str(record["rig_id"])
            draw.text(
                (x + 5, y + 3),
                label[:62],
                fill=(235, 235, 235),
                font=_font(12),
            )
            with Image.open(visual_root / record["filmstrip_relpath"]) as image:
                thumbnail = image.convert("RGB").resize(
                    (tile_width, tile_height), Image.Resampling.LANCZOS
                )
            sheet.paste(thumbnail, (x, y + label_height))
        path = output / f"sheet_{start // args.per_sheet:03d}.png"
        sheet.save(path)
        files.append(path.name)
    manifest = {
        "visual_generation_id": index["prototype_generation_id"],
        "visual_clip_count": len(records),
        "sheet_count": len(files),
        "per_sheet": args.per_sheet,
        "columns": args.columns,
        "files": files,
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
