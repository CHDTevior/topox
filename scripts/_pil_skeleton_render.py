"""Shared PIL skeleton renderer — large oblique-projection, root-centered.

Single source of truth for the big-figure render style (learned from AnyTop
render_rot6d_pose_compare.py), so T2M (animate_denoiser) and VAE-recon QA render
identically without copy-paste:
  - oblique projection: u = x - 0.36*z, v = y - 0.22*z
  - per-frame root-centered (subject stays centered, not shrunk by trajectory)
  - 900x760 / panel, ground grid + axes + root trail, optional top prompt band

GEOMETRY / DRAWING ONLY. NO recover funcs live here — those stay in src
(_recover_world_positions / recover_from_bvh_rot_np / recover_rot6d_fk_positions_torch).
Copying recover code is exactly how the double-root-rotation bug spread (2026-06-03);
this module deliberately holds none.
"""
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT_COLOR = (18, 18, 18)
TRAIL_COLOR = (150, 150, 150)
GROUND_COLOR = (224, 224, 224)
AXIS_X = (210, 34, 34); AXIS_Y = (30, 150, 55); AXIS_Z = (30, 80, 210)
HEADER_BG = (245, 245, 245)
HEADER_FG = (20, 20, 20)


def view_uv(points):
    return points[..., 0] - 0.36 * points[..., 2], points[..., 1] - 0.22 * points[..., 2]


def compute_transform(point_sets, size, pad, zoom):
    points = np.concatenate([p.reshape(-1, 3) for p in point_sets if p.size], axis=0)
    u, v = view_uv(points)
    u_min, u_max = float(u.min()), float(u.max()); v_min, v_max = float(v.min()), float(v.max())
    width, height = size
    u_span = max(u_max - u_min, 1e-6); v_span = max(v_max - v_min, 1e-6)
    base = min(width * (1 - 2 * pad) / u_span, height * (1 - 2 * pad) / v_span)
    return base * zoom, (u_min + u_max) * 0.5, (v_min + v_max) * 0.5


def project(points, transform, size):
    scale, u_mid, v_mid = transform; width, height = size
    u, v = view_uv(points)
    px = width * 0.5 + (u - u_mid) * scale
    py = height * 0.54 - (v - v_mid) * scale
    return np.stack([px, py], axis=-1)


def get_font(size):
    for c in ["/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
              "/usr/share/fonts/dejavu/DejaVuSans.ttf",
              "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
              "/usr/share/fonts/liberation/LiberationSans-Regular.ttf"]:
        if Path(c).exists():
            return ImageFont.truetype(c, size)
    return ImageFont.load_default()


def draw_axes(draw, size):
    width, height = size; ox = int(width * 0.10); oy = int(height * 0.82)
    length = int(min(width, height) * 0.14)
    draw.line([(ox, oy), (ox + length, oy)], fill=AXIS_X, width=4)
    draw.line([(ox, oy), (ox, oy - length)], fill=AXIS_Y, width=4)
    draw.line([(ox, oy), (ox - int(0.45 * length), oy + int(0.65 * length))], fill=AXIS_Z, width=4)
    lf = get_font(18)
    draw.text((ox + length + 6, oy - 10), "+X", fill=AXIS_X, font=lf)
    draw.text((ox + 5, oy - length - 22), "+Y", fill=AXIS_Y, font=lf)
    draw.text((ox - int(0.45 * length) - 30, oy + int(0.65 * length) - 7), "+Z", fill=AXIS_Z, font=lf)


def draw_ground(draw, transform, size, radius):
    for value in np.linspace(-radius, radius, 7):
        for line in [np.array([[-radius, 0.0, value], [radius, 0.0, value]], dtype=float),
                     np.array([[value, 0.0, -radius], [value, 0.0, radius]], dtype=float)]:
            pts = project(line, transform, size)
            draw.line([tuple(pts[0]), tuple(pts[1])], fill=GROUND_COLOR, width=1)


def draw_skeleton(draw, positions, parents, transform, size, color, lw, jr, trail):
    if trail is not None and len(trail) > 1:
        t2d = project(trail, transform, size)
        draw.line([tuple(p) for p in t2d], fill=TRAIL_COLOR, width=max(2, lw - 1))
    pts = project(positions, transform, size)
    for j in range(1, len(parents)):
        p = int(parents[j])
        if p >= 0:
            draw.line([tuple(pts[p]), tuple(pts[j])], fill=color, width=lw)
    root = tuple(pts[0]); r = jr
    draw.ellipse((root[0]-r, root[1]-r, root[0]+r, root[1]+r), fill=ROOT_COLOR)


def render_panel(positions, parents, fi, transform, size, title, color, lw, jr, axes, static=False):
    """One panel. static=True → no root trail (e.g. input rest-pose skeleton)."""
    image = Image.new("RGB", size, "white"); draw = ImageDraw.Draw(image)
    radius = max(2.0, float(np.linalg.norm(positions.reshape(-1, 3)[:, [0, 2]], axis=-1).max()) * 1.05)
    draw_ground(draw, transform, size, radius)
    frame = positions[fi]
    trail = None if static else positions[:, 0].copy()
    if trail is not None:
        trail[:, 1] = 0.0
    draw_skeleton(draw, frame, parents, transform, size, color, lw, jr, trail)
    if axes:
        draw_axes(draw, size)
    draw.text((18, 16), title, fill=color, font=get_font(24))
    return image


def make_row_frame(panels, fi, transform, cell, lw, jr, header=None, header_h=0):
    """Render N panels side by side at frame fi, each per-frame root-centered.

    panels: list of dicts {positions:[T,J,3], parents:[J], title, color,
            axes:bool, static:bool}. Each panel is independently root-centered on
            its own current-frame root (subject stays centered).
    header: optional text band (e.g. T2M prompt) across the full top; header_h px.
    Returns a PIL.Image of size (W*N, H + header_h).
    """
    W, H = cell
    n = len(panels)
    canvas = Image.new("RGB", (W * n, H + header_h), "white")
    for k, p in enumerate(panels):
        s = p["positions"].copy()
        root = s[fi, 0].copy()                  # per-panel root-centered
        s[..., 0] -= root[0]; s[..., 2] -= root[2]
        panel = render_panel(s, p["parents"], fi, transform, cell, p["title"],
                             p["color"], lw, jr, p.get("axes", False), p.get("static", False))
        canvas.paste(panel, (W * k, header_h))
    if header and header_h > 0:
        draw = ImageDraw.Draw(canvas)
        draw.rectangle([0, 0, W * n, header_h], fill=HEADER_BG)
        hf = get_font(28)
        # wrap header to width
        words = header.split(); lines = []; cur = ""
        maxchars = max(20, int(W * n / 16))
        for w in words:
            if len(cur) + len(w) + 1 <= maxchars:
                cur = (cur + " " + w).strip()
            else:
                lines.append(cur); cur = w
        if cur:
            lines.append(cur)
        for li, ln in enumerate(lines[:max(1, header_h // 34)]):
            draw.text((20, 8 + li * 34), ln, fill=HEADER_FG, font=hf)
    return canvas


def save_gif(frames, out_path, fps):
    dur = int(round(1000.0 / max(fps, 1e-6)))
    frames[0].save(out_path, save_all=True, append_images=frames[1:], duration=dur, loop=0, optimize=True)


def sample_indices(length, max_frames):
    if length <= max_frames:
        return list(range(length))
    return sorted(set(int(round(v)) for v in np.linspace(0, length - 1, max_frames)))
