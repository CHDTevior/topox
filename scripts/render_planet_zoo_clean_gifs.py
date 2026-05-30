#!/usr/bin/env python3
"""Render per-object GIFs for Planet Zoo cleaned datasets."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.anytop_dataset import _recover_world_positions


EDGE = (210, 83, 45)
ROOT = (35, 112, 180)
TRAIL = (130, 130, 130)
GROUND = (226, 226, 226)
TEXT = (35, 35, 35)
AXIS_X = (210, 34, 34)
AXIS_Y = (30, 150, 55)
AXIS_Z = (30, 80, 210)


DEFAULT_OBJECTS = [
    "PZ_Indian_Peafowl_Male",
    "PZ_Indian_Peafowl_Female",
    "PZ_Tasmanian_Devil_Female",
    "PZ_Koala_Female",
    "PZ_Common_Wombat_Female",
    "PZ_Giant_Anteater_Female",
    "PZ_Aardvark_Female",
    "PZ_Grey_Seal_Female",
    "PZ_California_Sea_Lion_Female",
    "PZ_Coquerels_Sifaka_Female",
    "PZ_American_Alligator_Male",
    "PZ_Gharial_Female",
    "PZ_African_Penguin_Male",
    "PZ_Greater_Flamingo_Male",
    "PZ_Western_Chimpanzee_Female",
    "PZ_Red_Kangaroo_Female",
    "PZ_African_Elephant_Female",
    "PZ_Galapagos_Giant_Tortoise_Female",
]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--l1", default="/scratch/ts1v23/workspace/noKslot_clean/data/anytop_planet_zoo_clean_L1")
    ap.add_argument("--l2", default="/scratch/ts1v23/workspace/noKslot_clean/data/anytop_planet_zoo_clean_L2")
    ap.add_argument("--objects", nargs="*", default=DEFAULT_OBJECTS)
    ap.add_argument("--frames", type=int, default=32)
    ap.add_argument("--size", type=int, default=420)
    ap.add_argument("--duration-ms", type=int, default=80)
    ap.add_argument("--subdir", default="", help="Optional subdirectory under animations/")
    ap.add_argument("--manifest-name", default="clean_gif_manifest.json")
    ap.add_argument("--overwrite", action="store_true")
    return ap.parse_args()


def safe_name(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")
    return value or "unnamed"


def choose_motion(root: Path, obj: str) -> Path:
    paths = sorted((root / "motions").glob(f"{obj}_*.npy"))
    if not paths:
        raise FileNotFoundError(f"no motions for {obj} under {root}")
    preferred = [
        "runbase",
        "walktorun",
        "standtowalk",
        "walkbase",
        "swimbase",
        "climb",
        "jumpmid",
        "standturn",
    ]
    lower = [(p, p.name.lower()) for p in paths]
    for token in preferred:
        for p, name in lower:
            if token in name:
                return p
    return paths[0]


def view_uv(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = points[..., 0]
    y = points[..., 1]
    z = points[..., 2]
    return x - 0.36 * z, y - 0.22 * z


def compute_transform(point_sets: list[np.ndarray], size: tuple[int, int], pad: float = 0.12) -> tuple[float, float, float]:
    pts = np.concatenate([p.reshape(-1, 3) for p in point_sets if p.size], axis=0)
    u, v = view_uv(pts)
    umin, umax = float(u.min()), float(u.max())
    vmin, vmax = float(v.min()), float(v.max())
    scale = min(
        size[0] * (1.0 - 2.0 * pad) / max(umax - umin, 1e-6),
        size[1] * (1.0 - 2.0 * pad) / max(vmax - vmin, 1e-6),
    )
    return scale, (umin + umax) * 0.5, (vmin + vmax) * 0.5


def project(points: np.ndarray, transform: tuple[float, float, float], size: tuple[int, int]) -> np.ndarray:
    scale, umid, vmid = transform
    u, v = view_uv(points)
    return np.stack(
        [size[0] * 0.5 + (u - umid) * scale, size[1] * 0.5 - (v - vmid) * scale],
        axis=-1,
    )


def draw_axes(draw: ImageDraw.ImageDraw, size: tuple[int, int]) -> None:
    ox = int(size[0] * 0.12)
    oy = int(size[1] * 0.84)
    length = int(min(size) * 0.16)
    draw.line([(ox, oy), (ox + length, oy)], fill=AXIS_X, width=3)
    draw.line([(ox, oy), (ox, oy - length)], fill=AXIS_Y, width=3)
    draw.line([(ox, oy), (ox - int(0.45 * length), oy + int(0.65 * length))], fill=AXIS_Z, width=3)
    draw.text((ox + length + 3, oy - 8), "+X", fill=AXIS_X)
    draw.text((ox + 3, oy - length - 14), "+Y", fill=AXIS_Y)
    draw.text((ox - int(0.45 * length) - 20, oy + int(0.65 * length) - 4), "+Z", fill=AXIS_Z)


def draw_ground(draw: ImageDraw.ImageDraw, transform: tuple[float, float, float], size: tuple[int, int]) -> None:
    for v in np.linspace(-2.0, 2.0, 5):
        for line in (
            np.array([[-2.0, 0.0, v], [2.0, 0.0, v]], dtype=np.float32),
            np.array([[v, 0.0, -2.0], [v, 0.0, 2.0]], dtype=np.float32),
        ):
            pts = project(line, transform, size)
            draw.line([tuple(pts[0]), tuple(pts[1])], fill=GROUND, width=1)


def draw_frame(
    positions: np.ndarray,
    parents: np.ndarray,
    trail: np.ndarray,
    transform: tuple[float, float, float],
    size: tuple[int, int],
    title: str,
) -> Image.Image:
    image = Image.new("RGB", size, "white")
    draw = ImageDraw.Draw(image)
    draw_ground(draw, transform, size)
    if len(trail) > 1:
        trail_pts = project(trail, transform, size)
        draw.line([tuple(p) for p in trail_pts], fill=TRAIL, width=2)
    pts = project(positions, transform, size)
    for j in range(1, len(parents)):
        p = int(parents[j])
        if p >= 0:
            draw.line([tuple(pts[p]), tuple(pts[j])], fill=EDGE, width=2)
    root = tuple(pts[0])
    draw.ellipse((root[0] - 4, root[1] - 4, root[0] + 4, root[1] + 4), fill=ROOT)
    draw_axes(draw, size)
    draw.text((8, 8), title, fill=TEXT)
    return image


def sample_indices(length: int, frames: int) -> list[int]:
    if length <= 1:
        return [0]
    n = min(length, max(2, frames))
    return [int(round(v)) for v in np.linspace(0, length - 1, n)]


def render_one(
    root: Path,
    level: str,
    obj: str,
    frames: int,
    size: int,
    duration_ms: int,
    overwrite: bool,
    subdir: str = "",
) -> dict:
    cond = np.load(root / "cond.npy", allow_pickle=True).item()
    parents = np.asarray(cond[obj]["parents"], dtype=np.int64)
    motion_path = choose_motion(root, obj)
    out_dir = root / "animations"
    if subdir:
        out_dir = out_dir / safe_name(subdir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{safe_name(obj)}__{safe_name(motion_path.stem)}.gif"
    if out_path.exists() and not overwrite:
        return {
            "object": obj,
            "level": level,
            "motion": motion_path.name,
            "gif": str(out_path),
            "skipped": True,
        }
    motion = np.load(motion_path)
    positions = _recover_world_positions(motion)
    positions = positions.copy()
    positions[..., 1] -= positions[..., 1].min()
    roots = positions[:, 0].copy()
    idxs = sample_indices(len(positions), frames)
    frame_sets = []
    for idx in idxs:
        frame = positions[idx].copy()
        frame[:, 0] -= roots[idx, 0]
        frame[:, 2] -= roots[idx, 2]
        trail = roots.copy()
        trail[:, 0] -= roots[idx, 0]
        trail[:, 1] = 0.0
        trail[:, 2] -= roots[idx, 2]
        frame_sets.append((idx, frame, trail))
    transform = compute_transform([x for _, frame, trail in frame_sets for x in (frame, trail)], (size, size))
    images = [
        draw_frame(
            frame,
            parents,
            trail,
            transform,
            (size, size),
            f"{level} {obj} f{idx} J={len(parents)}",
        )
        for idx, frame, trail in frame_sets
    ]
    images[0].save(
        out_path,
        save_all=True,
        append_images=images[1:],
        duration=duration_ms,
        loop=0,
        optimize=True,
    )
    return {
        "object": obj,
        "level": level,
        "motion": motion_path.name,
        "shape": [int(v) for v in motion.shape],
        "frames_rendered": len(images),
        "gif": str(out_path),
        "skipped": False,
    }


def main() -> int:
    args = parse_args()
    roots = {"L1": Path(args.l1), "L2": Path(args.l2)}
    all_records = []
    for level, root in roots.items():
        for obj in args.objects:
            rec = render_one(
                root,
                level,
                obj,
                args.frames,
                args.size,
                args.duration_ms,
                args.overwrite,
                args.subdir,
            )
            all_records.append(rec)
            print(f"{level} rendered {obj}: {rec['gif']}", flush=True)
        manifest_dir = root / "animations"
        if args.subdir:
            manifest_dir = manifest_dir / safe_name(args.subdir)
        manifest_dir.mkdir(parents=True, exist_ok=True)
        (manifest_dir / args.manifest_name).write_text(
            json.dumps([r for r in all_records if r["level"] == level], indent=2)
        )
    print(f"rendered {len(all_records)} GIFs", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
