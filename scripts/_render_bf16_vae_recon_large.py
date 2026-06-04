"""bf16 VAE recon QA — 3-panel LARGE GIF: GT_RIC | PRED_RIC | PRED_FK.

Learns the AnyTop `render_rot6d_pose_compare.py` rendering style (user-requested
2026-06-03, "渲染太小，学 anytop 的大图方式"):
  - PIL oblique projection: u = x - 0.36*z, v = y - 0.22*z
  - per-frame root-centered (animal stays centered, not shrunk by trajectory)
  - 900x760 / panel, ground grid + axes + root trail, GIF (48 frames, 12 fps)

recover funcs IMPORTED FROM src — single source of truth. NO verbatim copy: that
is exactly how the rot6d-FK double-root-rotation bug crept back in 2026-06-03 (a
stale copy in the old QA script never got the 2026-06-01 src fix). codex 019e8e62.
  RIC (ch0:3, pos)  = src.data.anytop_dataset._recover_world_positions
  rot6d-FK (ch3:9)  = src.data.anytop_rot6d_fk.recover_from_bvh_rot_np (double-fix applied)

VAE forward (fp32 — the bf16-autocast-trained ckpt stores fp32 weights, so fp32
inference on CPU reproduces it; deterministic z=mu via sample=False):
  GT_RIC  = RIC(gt)   ch0:3 of the GT clip           (truth, grey)
  PRED_RIC= RIC(pred) recon position channel 0:3     (blue)
  PRED_FK = rot(pred) recon rot6d channel 3:9 + FK   (orange)

Run on compute node (offline OK, local data); default CPU so it never grabs a
training GPU:
  python scripts/_render_bf16_vae_recon_large.py --ckpt <bf16_vae>/best_model.pt
"""
import sys
import importlib.util
import argparse
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from PIL import Image, ImageDraw, ImageFont  # noqa: E402

# recover funcs from src — single source of truth (NEVER verbatim-copy; that's how
# the double-root-rotation bug reappeared 2026-06-03). recover_from_bvh_rot_np has
# the 2026-06-01 double-root fix; _recover_world_positions is the RIC (pos) path.
from src.data.anytop_rot6d_fk import recover_from_bvh_rot_np  # noqa: E402
from src.data.anytop_dataset import _recover_world_positions  # noqa: E402


# ===== PIL render (learned from AnyTop render_rot6d_pose_compare.py) =====
GT_COLOR = (90, 90, 90)        # GT_RIC   grey (truth)
POSE_COLOR = (35, 112, 180)    # PRED_RIC blue (recon pos 0:3)
ROTFK_COLOR = (210, 83, 45)    # PRED_FK  orange (recon rot6d-FK 3:9)
ROOT_COLOR = (18, 18, 18)
TRAIL_COLOR = (150, 150, 150)
GROUND_COLOR = (224, 224, 224)
AXIS_X = (210, 34, 34); AXIS_Y = (30, 150, 55); AXIS_Z = (30, 80, 210)


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


def render_panel(positions, parents, fi, transform, size, title, color, lw, jr, axes):
    image = Image.new("RGB", size, "white"); draw = ImageDraw.Draw(image)
    radius = max(2.0, float(np.linalg.norm(positions.reshape(-1, 3)[:, [0, 2]], axis=-1).max()) * 1.05)
    draw_ground(draw, transform, size, radius)
    frame = positions[fi]
    trail = positions[:, 0].copy(); trail[:, 1] = 0.0
    draw_skeleton(draw, frame, parents, transform, size, color, lw, jr, trail)
    if axes:
        draw_axes(draw, size)
    draw.text((18, 16), title, fill=color, font=get_font(24))
    return image


def make_frame_3panel(gt, pred_ric, pred_fk, parents, fi, transform, cell, lw, jr):
    W, H = cell
    image = Image.new("RGB", (W * 3, H), "white")
    series_list = [(gt, "GT_RIC (truth)", GT_COLOR, True),
                   (pred_ric, "PRED_RIC recon 0:3", POSE_COLOR, False),
                   (pred_fk, "PRED_FK recon 3:9", ROTFK_COLOR, False)]
    for k, (positions, title, color, axes) in enumerate(series_list):
        s = positions.copy()
        root = s[fi, 0].copy()          # per-panel root-centered (animal stays centered)
        s[..., 0] -= root[0]; s[..., 2] -= root[2]
        panel = render_panel(s, parents, fi, transform, cell, title, color, lw, jr, axes)
        image.paste(panel, (W * k, 0))
    return image


# ===== VAE recon =====
from src.data.anytop_dataset import AnyTopDataset, _STD_FLOOR, collate_fn  # noqa: E402
from src.models.graph_salad import GraphMotionBatch, GraphMotionVAE  # noqa: E402,F401
_spec = importlib.util.spec_from_file_location("aa13", str(ROOT / "scripts" / "animate_anytop13.py"))
aa = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(aa)


def sample_indices(length, max_frames):
    if length <= max_frames:
        return list(range(length))
    return sorted(set(int(round(v)) for v in np.linspace(0, length - 1, max_frames)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(ROOT / "runs/m1_bf16_anytop13_rot6dfk_w025f100t010_C128_8card_xnode_seed42/best_model.pt"))
    ap.add_argument("--out", default=str(ROOT / "runs/_bf16_vae_recon_large"))
    ap.add_argument("--data-root", default=str(ROOT / "data/anytop_planet_zoo_clean_L2"))
    ap.add_argument("--device", default="cpu")            # cpu = never grab a training GPU
    ap.add_argument("--cell-width", type=int, default=900)
    ap.add_argument("--cell-height", type=int, default=760)
    ap.add_argument("--max-frames", type=int, default=48)
    ap.add_argument("--fps", type=float, default=12.0)
    ap.add_argument("--zoom", type=float, default=1.15)
    ap.add_argument("--pad", type=float, default=0.06)
    ap.add_argument("--line-width", type=int, default=3)
    ap.add_argument("--joint-radius", type=int, default=4)
    ap.add_argument("--species", action="append", default=[])
    args = ap.parse_args()

    dev = torch.device(args.device)
    vae, ta = aa.load_anytop13_vae(args.ckpt, dev)
    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    ds = AnyTopDataset(split="val", val_frac=0.05, seed=42, data_root=args.data_root,
                       num_frames=64, max_joints=144, caption_emb_cache=None)
    want = args.species or ["PZ_Asian_Water_Monitor_Male", "PZ_Komodo_Dragon_Male",
                            "PZ_Saltwater_Crocodile_Male", "PZ_Grey_Seal_Male"]
    cell = (args.cell_width, args.cell_height)
    done = set()
    with torch.no_grad():
        for i in range(len(ds)):
            it = ds[i]; sp = it["object_type"]
            if sp not in want or sp in done:
                continue
            done.add(sp)
            J = int(it["num_joints"]); T = int(it["num_frames"])
            parents = np.asarray([int(p) for p in it["parent_indices"][:J]], dtype=int)
            offsets = np.asarray(it["rest_offsets"], np.float32)[:J].astype(np.float64)
            raw = collate_fn([it]); raw = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in raw.items()}
            batch = GraphMotionBatch.from_collate_dict(raw)
            out = vae(batch, sample=False)
            Tv = int(out["frame_mask_recovered"][0].sum().item()); T = min(T, Tv)
            std = raw["anytop_std"][0, :J].cpu().numpy(); mean = raw["anytop_mean"][0, :J].cpu().numpy()
            pred = (out["pred_motion"][0, :T, :J, :].cpu().numpy() * (std[None] + _STD_FLOOR) + mean[None]).astype(np.float64)
            gt = (np.transpose(np.asarray(it["anytop_x"], np.float32), (2, 0, 1))[:T, :J, :]
                  * (std[None] + _STD_FLOOR) + mean[None]).astype(np.float64)

            # recover via src (single source of truth — no verbatim copy)
            gt_ric = _recover_world_positions(gt).astype(np.float64)        # GT_RIC   (truth, grey)
            pred_ric = _recover_world_positions(pred).astype(np.float64)    # PRED_RIC (recon pos 0:3, blue)
            pred_fk = recover_from_bvh_rot_np(pred, parents, offsets).astype(np.float64)  # PRED_FK (recon 3:9 FK, orange)
            gt_rot = recover_from_bvh_rot_np(gt, parents, offsets).astype(np.float64)
            gtbbox = np.linalg.norm(gt_ric.reshape(-1, 3).max(0) - gt_ric.reshape(-1, 3).min(0))
            gtchk = np.linalg.norm(gt_ric - gt_rot, axis=-1).mean() / gtbbox * 100        # FK==RIC self-check
            recon_err = np.linalg.norm(gt_ric - pred_ric, axis=-1).mean() / gtbbox * 100  # GT vs PRED pose
            root_err = np.linalg.norm(gt_ric[:, 0] - pred_ric[:, 0], axis=-1).mean() / gtbbox * 100  # root traj (root-centered render hides this)
            print(f"{sp} J={J} T={T} | GT_selfchk={gtchk:.2f}% (<2 ok, renderer faithful) | "
                  f"recon_err(GT-vs-PRED_RIC)={recon_err:.2f}%bbox | root_traj_err={root_err:.2f}%bbox", flush=True)

            # ground-normalize each (y min -> 0) so all panels share the floor plane
            for arr in (gt_ric, pred_ric, pred_fk):
                arr[..., 1] -= arr[..., 1].min()

            idxs = sample_indices(T, args.max_frames)
            # unified transform across all 3 panels — feed each panel's root-centered
            # sampled frames (matches make_frame_3panel's per-panel root-centering)
            ps = []
            for arr in (gt_ric, pred_ric, pred_fk):
                c = arr.copy(); roots = c[:, 0].copy()
                c[..., 0] -= roots[:, None, 0]; c[..., 2] -= roots[:, None, 2]
                ps += [c[k] for k in idxs]
            transform = compute_transform(ps, cell, args.pad, args.zoom)

            frames = [make_frame_3panel(gt_ric, pred_ric, pred_fk, parents, k, transform, cell,
                                        args.line_width, args.joint_radius) for k in idxs]
            out_path = out_dir / f"{sp}_recon_3panel_large.gif"
            dur = int(round(1000.0 / max(args.fps, 1e-6)))
            frames[0].save(out_path, save_all=True, append_images=frames[1:], duration=dur, loop=0, optimize=True)
            print(f"  saved {out_path} ({len(frames)}f, {cell[0]*3}x{cell[1]})", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
