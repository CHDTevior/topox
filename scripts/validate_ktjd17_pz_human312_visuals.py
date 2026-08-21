#!/usr/bin/env python3
"""Research-level decode and coverage check for the 312-rig visual QA output."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageChops


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.ktjd17.visual_qa import (  # noqa: E402
    COORDINATE_CONTRACT,
    verify_visual_generation,
)


EXPECTED_RIGS = 312


def _open_rgb(path: Path) -> Image.Image:
    with Image.open(path) as image:
        image.load()
        return image.convert("RGB")


def validate_visuals(root: Path) -> dict[str, object]:
    visual_root = root.expanduser().resolve()
    generation = verify_visual_generation(visual_root)
    index = json.loads((visual_root / "visual_qa_index.json").read_text("utf-8"))
    records = index.get("clips")
    if not isinstance(records, list):
        raise RuntimeError("visual_qa_index.json has no clip list")
    clip_ids = [str(record["clip_id"]) for record in records]
    rig_ids = [str(record["rig_id"]) for record in records]
    if (
        len(records) != EXPECTED_RIGS
        or len(set(clip_ids)) != EXPECTED_RIGS
        or len(set(rig_ids)) != EXPECTED_RIGS
    ):
        raise RuntimeError(
            f"visual coverage is {len(records)} clips/{len(set(rig_ids))} rigs, "
            f"expected {EXPECTED_RIGS}/{EXPECTED_RIGS}"
        )
    if (
        index.get("coordinate_contract") != COORDINATE_CONTRACT
        or index.get("required_paths")
        != ["source", "position-direct", "rotation-FK"]
        or index.get("perspective_camera") is not True
        or index.get("fixed_camera_across_frames_and_paths") is not True
        or index.get("frame_recenter_applied") is not False
        or index.get("ground_changed") is not False
        or index.get("face_direction_changed") is not False
    ):
        raise RuntimeError("visual coordinate/camera contract drifted")

    gif_frames: list[int] = []
    gif_durations: list[int] = []
    source_direct_errors: list[float] = []
    source_fk_errors: list[float] = []
    static_gifs: list[str] = []
    malformed: list[str] = []
    for record in records:
        clip_id = str(record["clip_id"])
        gif_path = visual_root / str(record["gif_relpath"])
        filmstrip_path = visual_root / str(record["filmstrip_relpath"])
        rest_path = visual_root / str(record["rest_relpath"])
        try:
            with Image.open(gif_path) as gif:
                frame_count = int(getattr(gif, "n_frames", 1))
                if gif.format != "GIF" or frame_count < 2 or gif.size != (1240, 342):
                    raise RuntimeError(
                        f"bad GIF format/shape/frames: {gif.format}/{gif.size}/{frame_count}"
                    )
                if frame_count != len(record["animation_frame_indices"]):
                    raise RuntimeError("GIF frame count does not match visual index")
                gif.seek(0)
                first = gif.convert("RGB")
                moving = False
                duration_values: list[int] = []
                for frame_index in range(frame_count):
                    gif.seek(frame_index)
                    current = gif.convert("RGB")
                    duration_values.append(int(gif.info.get("duration", 0)))
                    if ImageChops.difference(first, current).getbbox() is not None:
                        moving = True
                if not moving:
                    static_gifs.append(clip_id)
                gif_frames.append(frame_count)
                gif_durations.extend(duration_values)
            filmstrip = _open_rgb(filmstrip_path)
            expected_film_width = 310 * len(record["filmstrip_frame_indices"])
            if filmstrip.size != (expected_film_width, 798):
                raise RuntimeError(f"bad filmstrip size: {filmstrip.size}")
            rest = _open_rgb(rest_path)
            if rest.size != (620, 520):
                raise RuntimeError(f"bad rest image size: {rest.size}")
            film_array = np.asarray(filmstrip, dtype=np.float32)
            for row in range(3):
                row_pixels = film_array[48 + row * 250 : 48 + (row + 1) * 250]
                if float(np.std(row_pixels)) < 2.0:
                    raise RuntimeError(f"blank visual row {row}")
            source_direct_errors.append(float(record["source_direct_max_norm"]))
            source_fk_errors.append(float(record["source_fk_max_norm"]))
        except Exception as exc:  # noqa: BLE001
            malformed.append(f"{clip_id}: {type(exc).__name__}: {exc}")
    if malformed or static_gifs:
        raise RuntimeError(
            f"visual decode failed: malformed={malformed[:10]}, "
            f"static_gifs={static_gifs[:10]}"
        )
    result: dict[str, object] = {
        "status": "pass",
        "generation_id": generation["generation_id"],
        "clip_count": len(records),
        "rig_count": len(set(rig_ids)),
        "source_family_counts": {
            family: sum(record["source_family"] == family for record in records)
            for family in sorted({str(record["source_family"]) for record in records})
        },
        "coordinate_contract": COORDINATE_CONTRACT,
        "required_paths": ["source", "position-direct", "rotation-FK"],
        "gif_frame_count_min": min(gif_frames),
        "gif_frame_count_max": max(gif_frames),
        "gif_duration_ms_min": min(gif_durations),
        "gif_duration_ms_max": max(gif_durations),
        "source_direct_max_norm": max(source_direct_errors),
        "source_fk_max_norm": max(source_fk_errors),
        "malformed_asset_count": 0,
        "static_gif_count": 0,
    }
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "visual_root",
        nargs="?",
        type=Path,
        default=Path("dataset/ktjd17_pz_human312_visual/ktjd17_visual_qa"),
    )
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    result = validate_visuals(args.visual_root)
    payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(payload, encoding="utf-8")
    print(payload, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
