#!/usr/bin/env python3
"""Render visual QA sheets for Planet Zoo cleaned datasets.

Each sheet compares original / L1 / L2 for the same object and motion file.
It renders both rest pose and sampled motion frames from the AnyTop 13-channel
motion tensors, so no cleaned BVH export is required.
"""

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
]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="/scratch/ts1v23/workspace/noKslot_clean/data/anytop_planet_zoo")
    ap.add_argument("--l1", default="/scratch/ts1v23/workspace/noKslot_clean/data/anytop_planet_zoo_clean_L1")
    ap.add_argument("--l2", default="/scratch/ts1v23/workspace/noKslot_clean/data/anytop_planet_zoo_clean_L2")
    ap.add_argument("--out", default="/scratch/ts1v23/workspace/noKslot_clean/data/anytop_planet_zoo_clean_visual_qa")
    ap.add_argument("--objects", nargs="*", default=DEFAULT_OBJECTS)
    ap.add_argument("--frames", type=int, default=6)
    ap.add_argument("--cell-size", type=int, default=220)
    ap.add_argument("--overwrite", action="store_true")
    return ap.parse_args()


def safe_name(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")
    return value or "unnamed"


def load_cond(root: Path, obj: str) -> dict:
    cond = np.load(root / "cond.npy", allow_pickle=True).item()
    return cond[obj]


def rest_positions(offsets: np.ndarray, parents: np.ndarray) -> np.ndarray:
    pos = np.zeros_like(offsets, dtype=np.float32)
    for j, p in enumerate(parents.tolist()):
        if p >= 0:
            pos[j] = pos[int(p)] + offsets[j]
        else:
            pos[j] = offsets[j]
    return pos


def normalize_ground(points: np.ndarray) -> np.ndarray:
    points = points.copy()
    points[..., 1] -= points[..., 1].min()
    return points


def view_uv(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = points[..., 0]
    y = points[..., 1]
    z = points[..., 2]
    return x - 0.36 * z, y - 0.22 * z


def compute_transform(point_sets: list[np.ndarray], size: tuple[int, int], pad: float = 0.14) -> tuple[float, float, float]:
    pts = np.concatenate([p.reshape(-1, 3) for p in point_sets if p.size], axis=0)
    u, v = view_uv(pts)
    umin, umax = float(u.min()), float(u.max())
    vmin, vmax = float(v.min()), float(v.max())
    uspan = max(umax - umin, 1e-6)
    vspan = max(vmax - vmin, 1e-6)
    w, h = size
    scale = min(w * (1 - 2 * pad) / uspan, h * (1 - 2 * pad) / vspan)
    return scale, (umin + umax) * 0.5, (vmin + vmax) * 0.5


def project(points: np.ndarray, transform: tuple[float, float, float], size: tuple[int, int]) -> np.ndarray:
    scale, umid, vmid = transform
    w, h = size
    u, v = view_uv(points)
    return np.stack([w * 0.5 + (u - umid) * scale, h * 0.5 - (v - vmid) * scale], axis=-1)


def draw_axes(draw: ImageDraw.ImageDraw, size: tuple[int, int]) -> None:
    w, h = size
    ox = int(w * 0.12)
    oy = int(h * 0.82)
    length = int(min(w, h) * 0.16)
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


def draw_skeleton(
    draw: ImageDraw.ImageDraw,
    positions: np.ndarray,
    parents: np.ndarray,
    transform: tuple[float, float, float],
    size: tuple[int, int],
    *,
    title: str,
    trail: np.ndarray | None = None,
    axes: bool = False,
) -> None:
    draw_ground(draw, transform, size)
    if trail is not None and len(trail) > 1:
        trail_pts = project(trail, transform, size)
        draw.line([tuple(p) for p in trail_pts], fill=TRAIL, width=2)
    pts = project(positions, transform, size)
    for j in range(1, len(parents)):
        p = int(parents[j])
        if p >= 0:
            draw.line([tuple(pts[p]), tuple(pts[j])], fill=EDGE, width=2)
    root = tuple(pts[0])
    draw.ellipse((root[0] - 4, root[1] - 4, root[0] + 4, root[1] + 4), fill=ROOT)
    if axes:
        draw_axes(draw, size)
    draw.text((8, 8), title, fill=TEXT)


def sample_indices(length: int, count: int) -> list[int]:
    if length <= 1:
        return [0]
    return sorted(set(int(round(x)) for x in np.linspace(0, length - 1, count)))


def choose_motion(root: Path, obj: str) -> Path:
    paths = sorted((root / "motions").glob(f"{obj}_*.npy"))
    if not paths:
        raise FileNotFoundError(f"no motions for {obj} under {root}")
    preferred = [
        "runbase",
        "walktorun",
        "walkbase",
        "standtowalk",
        "swimbase",
        "climb",
    ]
    lower = [(p, p.name.lower()) for p in paths]
    for token in preferred:
        for p, name in lower:
            if token in name:
                return p
    return paths[0]


def render_rest_sheet(obj: str, roots: dict[str, Path], out_path: Path, cell: int) -> dict:
    panels = []
    stats = {}
    for label, root in roots.items():
        c = load_cond(root, obj)
        parents = np.asarray(c["parents"], dtype=np.int64)
        pos = normalize_ground(rest_positions(np.asarray(c["offsets"], dtype=np.float32), parents))
        pos[:, 0] -= pos[0, 0]
        pos[:, 2] -= pos[0, 2]
        panels.append((label, pos, parents))
        stats[label] = int(len(parents))
    transform = compute_transform([p[1] for p in panels], (cell, cell))
    img = Image.new("RGB", (cell * len(panels), cell), "white")
    for i, (label, pos, parents) in enumerate(panels):
        tile = Image.new("RGB", (cell, cell), "white")
        draw = ImageDraw.Draw(tile)
        draw_skeleton(draw, pos, parents, transform, (cell, cell), title=f"{label} rest J={len(parents)}", axes=(i == 0))
        img.paste(tile, (i * cell, 0))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path)
    return stats


def render_motion_sheet(obj: str, roots: dict[str, Path], out_path: Path, frames: int, cell: int) -> dict:
    src_motion = choose_motion(roots["orig"], obj)
    motion_name = src_motion.name
    rows = []
    stats = {"motion": motion_name}
    for label, root in roots.items():
        c = load_cond(root, obj)
        parents = np.asarray(c["parents"], dtype=np.int64)
        motion = np.load(root / "motions" / motion_name)
        pos = normalize_ground(_recover_world_positions(motion))
        idx = sample_indices(len(pos), frames)
        roots_xyz = pos[:, 0].copy()
        row = []
        for k in idx:
            frame = pos[k].copy()
            frame[:, 0] -= roots_xyz[k, 0]
            frame[:, 2] -= roots_xyz[k, 2]
            trail = roots_xyz.copy()
            trail[:, 0] -= roots_xyz[k, 0]
            trail[:, 1] = 0.0
            trail[:, 2] -= roots_xyz[k, 2]
            row.append((frame, trail))
        rows.append((label, parents, idx, row))
        stats[label] = {"shape": [int(v) for v in motion.shape], "indices": idx}
    all_points = []
    for _, _, _, row in rows:
        for frame, trail in row:
            all_points.extend([frame, trail])
    transform = compute_transform(all_points, (cell, cell))
    img = Image.new("RGB", (cell * frames, cell * len(rows)), "white")
    for r, (label, parents, idx, row) in enumerate(rows):
        for c, (frame, trail) in enumerate(row):
            tile = Image.new("RGB", (cell, cell), "white")
            draw = ImageDraw.Draw(tile)
            draw_skeleton(
                draw,
                frame,
                parents,
                transform,
                (cell, cell),
                title=f"{label} f{idx[c]} J={len(parents)}",
                trail=trail,
                axes=(r == 0 and c == 0),
            )
            img.paste(tile, (c * cell, r * cell))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path)
    return stats


def main() -> int:
    args = parse_args()
    roots = {"orig": Path(args.source), "L1": Path(args.l1), "L2": Path(args.l2)}
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    manifest = []
    for obj in args.objects:
        rest_path = out / "rest_compare" / f"{safe_name(obj)}.png"
        motion_path = out / "motion_compare" / f"{safe_name(obj)}.png"
        if not args.overwrite and rest_path.exists() and motion_path.exists():
            continue
        rest_stats = render_rest_sheet(obj, roots, rest_path, args.cell_size)
        motion_stats = render_motion_sheet(obj, roots, motion_path, args.frames, args.cell_size)
        manifest.append(
            {
                "object": obj,
                "rest_path": str(rest_path),
                "motion_path": str(motion_path),
                "rest_joints": rest_stats,
                "motion": motion_stats,
            }
        )
        print(f"rendered {obj}", flush=True)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"QA written to {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
