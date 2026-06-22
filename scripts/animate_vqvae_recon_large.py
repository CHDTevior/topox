"""LARGE per-joint-labeled recon QA for the Graph-VQVAE (GraphVQTokenizer).

2-panel GT | RECON, 900x760 / panel PIL oblique render (the "big" style from
scripts/_render_bf16_vae_recon_large.py), with EVERY joint drawn as a dot and its
joint index labeled (user 2026-06-13: "渲染太小，想要更大的渲染方式…把每个 joint
都给我标出来").

Single source of truth — NO verbatim copies of the bug-prone bits:
  * VQVAE model reconstruction reuses scripts/animate_vqvae_recon.py:load_vq_tokenizer
    (importlib) so the constructor never drifts from the train recipe.
  * rot6d FK recovery is src.data.anytop_rot6d_fk.recover_from_bvh_rot_np (has the
    2026-06-01 double-root fix). The project has been bitten twice by
    double-root-rotation from verbatim-copied FK — so it is IMPORTED, never copied.
  * Only the pure PIL projection/ground/axes helpers are copied from
    _render_bf16_vae_recon_large.py (geometry, no FK — safe to copy).

Forward mirrors animate_vqvae_recon.py exactly: model(batch, allow_collectives=False)
under the ckpt's bf16 autocast (rank-0-only eval, no cross-rank quantizer collectives),
out["pred_motion"] [B,T,J,13] -> de-norm -> recover_from_bvh_rot_np for BOTH GT and
pred (apples-to-apples).

Run on a spare GPU node (never a training card):
  python scripts/animate_vqvae_recon_large.py --ckpt <vqvae>/best_model.pt \
      --species Camel,Dragon,Horse --out runs/<run>/qa_recon_large
"""
import argparse
import importlib.util
import os
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.anytop_dataset import (  # noqa: E402
    AnyTopDataset, collate_fn, _recover_world_positions, _STD_FLOOR,
)
from src.data.anytop_rot6d_fk import recover_from_bvh_rot_np  # noqa: E402  official rot6d FK
from src.models.graph_salad.batch import GraphMotionBatch  # noqa: E402

# Reuse the VQVAE model reconstruction (single source of truth) from the small
# recon script — importlib because scripts/ is not a package and the module name
# has no __init__. Loading it runs only module-level imports (its main() is
# __main__-guarded), so this is side-effect-free here.
_spec = importlib.util.spec_from_file_location(
    "avr", str(ROOT / "scripts" / "animate_vqvae_recon.py"))
avr = importlib.util.module_from_spec(_spec)
# NOTE: exec'ing this module transitively imports scripts.animate, which does an
# os.chdir(PROJECT_ROOT) at import time (animate.py). That would silently rewrite
# the cwd and break relative --ckpt/--out. Save and restore the caller's cwd.
_cwd_before = os.getcwd()
_spec.loader.exec_module(avr)
os.chdir(_cwd_before)
load_vq_tokenizer = avr.load_vq_tokenizer

# ===== PIL render geometry (copied from _render_bf16_vae_recon_large.py — pure
# oblique projection / ground / axes; NOT the FK recovery) =====
GT_COLOR = (90, 90, 90)        # GT (truth)  grey
RECON_COLOR = (35, 112, 180)   # RECON       blue
ROOT_COLOR = (18, 18, 18)
TRAIL_COLOR = (150, 150, 150)
GROUND_COLOR = (224, 224, 224)
JOINT_FILL = (25, 25, 25)      # per-joint dot
LABEL_COLOR = (200, 30, 30)    # joint-index text (red, high-contrast)
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
    try:
        return ImageFont.load_default(size=size)   # Pillow >=10 honors size
    except TypeError:
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


def draw_skeleton_labeled(draw, positions, parents, transform, size, color, lw, jr,
                          trail, label, font):
    """Bones + a dot at EVERY joint + (optional) its joint index."""
    if trail is not None and len(trail) > 1:
        t2d = project(trail, transform, size)
        draw.line([tuple(p) for p in t2d], fill=TRAIL_COLOR, width=max(2, lw - 1))
    pts = project(positions, transform, size)
    # bones first (so joint dots/labels sit on top)
    for j in range(1, len(parents)):
        p = int(parents[j])
        if p >= 0:
            draw.line([tuple(pts[p]), tuple(pts[j])], fill=color, width=lw)
    # a dot at EVERY joint; root gets a bigger, darker marker
    for j in range(len(pts)):
        x, y = float(pts[j][0]), float(pts[j][1])
        r = jr + 2 if j == 0 else jr
        fill = ROOT_COLOR if j == 0 else JOINT_FILL
        draw.ellipse((x - r, y - r, x + r, y + r), fill=fill)
    if label:
        fs = getattr(font, "size", 12)  # truetype has .size; bitmap fallback may not
        for j in range(len(pts)):
            x, y = float(pts[j][0]), float(pts[j][1])
            draw.text((x + jr + 1, y - jr - fs), str(j), fill=LABEL_COLOR, font=font)


def render_panel(positions, parents, fi, transform, size, title, color, lw, jr,
                 axes, label, font):
    image = Image.new("RGB", size, "white"); draw = ImageDraw.Draw(image)
    radius = max(2.0, float(np.linalg.norm(positions.reshape(-1, 3)[:, [0, 2]], axis=-1).max()) * 1.05)
    draw_ground(draw, transform, size, radius)
    frame = positions[fi]
    trail = positions[:, 0].copy(); trail[:, 1] = 0.0
    draw_skeleton_labeled(draw, frame, parents, transform, size, color, lw, jr, trail, label, font)
    if axes:
        draw_axes(draw, size)
    draw.text((18, 16), title, fill=color, font=get_font(24))
    return image


def make_frame_2panel(gt, recon, parents, fi, transform, cell, lw, jr, label, font,
                      titles=("GT (truth)", "RECON (VQVAE)")):
    W, H = cell
    image = Image.new("RGB", (W * 2, H), "white")
    series = [(gt, titles[0], GT_COLOR, True),
              (recon, titles[1], RECON_COLOR, False)]
    for k, (positions, title, color, axes) in enumerate(series):
        s = positions.copy()
        root = s[fi, 0].copy()           # per-panel root-centered (animal stays centered)
        s[..., 0] -= root[0]; s[..., 2] -= root[2]
        panel = render_panel(s, parents, fi, transform, cell, title, color, lw, jr,
                             axes, label, font)
        image.paste(panel, (W * k, 0))
    return image


def sample_indices(length, max_frames):
    if length <= max_frames:
        return list(range(length))
    return sorted(set(int(round(v)) for v in np.linspace(0, length - 1, max_frames)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--split", default="val")
    ap.add_argument("--species", default="Camel,Dragon,Horse,Elephant,Lion",
                    help="comma-separated object_types to render")
    ap.add_argument("--n_per", type=int, default=1)
    ap.add_argument("--anytop_root", type=str, default=None,
                    help="defaults to the ckpt's training root if omitted")
    ap.add_argument("--cell_width", type=int, default=900)
    ap.add_argument("--cell_height", type=int, default=760)
    ap.add_argument("--max_frames", type=int, default=48)
    ap.add_argument("--fps", type=float, default=12.0)
    ap.add_argument("--zoom", type=float, default=1.15)
    ap.add_argument("--pad", type=float, default=0.06)
    ap.add_argument("--line_width", type=int, default=3)
    ap.add_argument("--joint_radius", type=int, default=5)
    ap.add_argument("--label", type=int, default=1, help="1=draw joint-index labels, 0=dots only")
    ap.add_argument("--font_size", type=int, default=13)
    ap.add_argument("--amp_dtype", choices=["fp32", "bf16"], default=None)
    ap.add_argument("--gt_fk_vs_ric", action="store_true",
                    help="DIAGNOSTIC (no VQVAE recon shown): render GT via rot6d->FK (left) vs GT via "
                         "RIC-position (right) — visualizes the GT rot6d-FK faithfulness / gt_fk_mismatch "
                         "floor. Forces render_from=fk for the GT-FK panel; right panel is GT-RIC.")
    ap.add_argument("--render_from", choices=["fk", "position"], default="fk",
                    help="'fk' = rot6d(ch3:9)->FK on bone offsets (recover_from_bvh_rot_np, "
                         "historical default); 'position' = RIC(ch0:3) world-position route "
                         "(_recover_world_positions) — SAME function the backbone gen QA uses, "
                         "so 'position' makes recon QA consistent with backbone + avoids the "
                         "human rot6d-FK self-check floor.")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        print("  [INFO] CUDA unavailable, falling back to CPU")
        args.device = "cpu"
    dev = torch.device(args.device)
    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)

    model, ta, ck = load_vq_tokenizer(args.ckpt, dev)
    amp_dtype = args.amp_dtype or ta.get("amp_dtype", "bf16")
    amp_enabled = (amp_dtype == "bf16") and dev.type == "cuda"
    print(f"Loaded GraphVQTokenizer: d_model={ta['d_model']} Q={ta['num_quantizers']} "
          f"num_codes={ta['num_codes']} temporal_stride={ta['temporal_stride']} "
          f"epoch={ck.get('epoch')} val_total={ck.get('val_total'):.4f} amp={amp_dtype}")

    # Mirror animate_vqvae_recon.py's dataset reconstruction EXACTLY (reproduce the
    # ckpt's training split: val_frac/seed/num_frames/max_joints from ckpt args,
    # captions off — the tokenizer takes no text).
    root = args.anytop_root or ta.get("anytop_root") or ck.get("data_root")
    ds = AnyTopDataset(
        split=args.split,
        num_frames=ta.get("max_frames", 64),
        max_joints=ta.get("max_joints", 64),
        val_frac=ta.get("val_frac", 0.05),
        seed=ta.get("seed", 42),
        load_captions=False,
        data_root=root,
    )
    print(f"  dataset: root={root} split={args.split} -> {len(ds)} clips")

    want = [s.strip() for s in args.species.split(",") if s.strip()]
    picked = {s: 0 for s in want}
    font = get_font(args.font_size)
    cell = (args.cell_width, args.cell_height)
    summary = []

    with torch.no_grad():
        for i in range(len(ds)):
            item = ds[i]
            sp = item["object_type"]
            if sp not in picked or picked[sp] >= args.n_per:
                continue
            raw = collate_fn([item])
            raw = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in raw.items()}
            batch = GraphMotionBatch.from_collate_dict(raw)

            if amp_enabled:
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    out = model(batch, allow_collectives=False)
            else:
                out = model(batch, allow_collectives=False)

            J = int(item["num_joints"])
            T_clip = int(item["num_frames"])
            T_valid = int(out["frame_mask_recovered"][0].sum().item())
            T = min(T_clip, T_valid)
            if T <= 0:
                print(f"  [SKIP] {sp}: no valid frames (T_clip={T_clip} T_valid={T_valid})", flush=True)
                continue

            std = raw["anytop_std"][0, :J].cpu().numpy()
            mean = raw["anytop_mean"][0, :J].cpu().numpy()
            pred_norm = out["pred_motion"][0, :T, :J, :].float().cpu().numpy()
            pred_raw = pred_norm * (std[None] + _STD_FLOOR) + mean[None]
            gt_norm = np.asarray(item["anytop_x"]).transpose(2, 0, 1)[:T, :J, :]
            gt_raw = gt_norm * (std[None] + _STD_FLOOR) + mean[None]

            parents = np.asarray([int(p) for p in item["parent_indices"][:J]], dtype=int)
            offsets = np.asarray(item["rest_offsets"])[:J]

            # --render_from: 'fk' = rot6d(ch3:9)->FK on bone offsets (recover_from_bvh_rot_np);
            # 'position' = RIC(ch0:3) world route (_recover_world_positions) for BOTH GT+recon,
            # the SAME function the backbone gen QA uses (consistency + no human FK floor).
            gt_ric = _recover_world_positions(gt_raw).astype(np.float64)        # RIC route (also self-check ref)
            if args.gt_fk_vs_ric:
                # DIAGNOSTIC: GT-FK (left) vs GT-RIC (right), independent of --render_from
                # (force the FK route for the LEFT panel; recon is not shown).
                gt_world = recover_from_bvh_rot_np(gt_raw, parents, offsets).astype(np.float64)
                pred_world = gt_ric.copy()
            elif args.render_from == "position":
                pred_world = _recover_world_positions(pred_raw).astype(np.float64)
                gt_world = gt_ric
            else:
                pred_world = recover_from_bvh_rot_np(pred_raw, parents, offsets).astype(np.float64)
                gt_world = recover_from_bvh_rot_np(gt_raw, parents, offsets).astype(np.float64)
            gtbbox = np.linalg.norm(gt_ric.reshape(-1, 3).max(0) - gt_ric.reshape(-1, 3).min(0))
            gt_selfcheck = float(np.linalg.norm(gt_world - gt_ric, axis=-1).mean())  # ~0 in position mode; =GT-FK-vs-RIC mismatch in gt_fk_vs_ric mode
            recon_l2 = float(np.linalg.norm(pred_world - gt_world, axis=-1).mean())

            # ground-normalize each (y min -> 0) so both panels share the floor.
            for arr in (gt_world, pred_world):
                arr[..., 1] -= arr[..., 1].min()

            idxs = sample_indices(T, args.max_frames)
            # shared transform across both panels (each panel root-centered, matching
            # make_frame_2panel) so GT and RECON render at the same scale.
            ps = []
            for arr in (gt_world, pred_world):
                c = arr.copy(); roots = c[:, 0].copy()
                c[..., 0] -= roots[:, None, 0]; c[..., 2] -= roots[:, None, 2]
                ps += [c[k] for k in idxs]
            transform = compute_transform(ps, cell, args.pad, args.zoom)

            _titles = (("GT rot6d->FK", "GT RIC-pos") if args.gt_fk_vs_ric
                       else ("GT (truth)", "RECON (VQVAE)"))
            frames = [make_frame_2panel(gt_world, pred_world, parents, k, transform, cell,
                                        args.line_width, args.joint_radius, bool(args.label), font,
                                        titles=_titles)
                      for k in idxs]
            _suffix = "GTfk_vs_GTric" if args.gt_fk_vs_ric else "gtvsrecon"
            out_path = out_dir / f"{sp}_clip{picked[sp]}_{_suffix}_large.gif"
            dur = int(round(1000.0 / max(args.fps, 1e-6)))
            frames[0].save(out_path, save_all=True, append_images=frames[1:],
                           duration=dur, loop=0, optimize=True)
            _metric = (f"GTfk_vs_GTric_L2={recon_l2:.4f}" if args.gt_fk_vs_ric
                       else f"recon_L2={recon_l2:.4f} GT_selfcheck_L2={gt_selfcheck:.2e}")
            line = (f"{sp} clip{picked[sp]}: J={J} T={T} {_metric} -> {out_path.name} "
                    f"({cell[0]*2}x{cell[1]}, {len(frames)}f)")
            print(line, flush=True)
            summary.append(line)
            picked[sp] += 1
            if all(picked[s] >= args.n_per for s in want):
                break

    missing = {s: args.n_per - picked[s] for s in want if picked[s] < args.n_per}
    (out_dir / "recon_large_summary.txt").write_text("\n".join(summary) + f"\nmissing={missing}\n")
    print(f"\nDONE {sum(picked.values())} gifs -> {out_dir}")
    print("PER-SPECIES picked:", picked)
    if missing:
        print(f"  [WARN] under-filled: {missing}")


if __name__ == "__main__":
    main()
