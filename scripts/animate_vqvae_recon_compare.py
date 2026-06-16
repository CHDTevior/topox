"""Multi-checkpoint recon-QA comparison for the Graph-VQVAE: one GIF per species
with panels  GT | RECON(ckpt_1) | RECON(ckpt_2) | ...  at MATCHED epoch, every
joint dotted + index-labeled (user 2026-06-13: compare 512 vs 1024 vs 2048 recon
quality at the same epoch).

Reuses, verbatim, the proven large per-joint-labeled renderer's helpers from
scripts/animate_vqvae_recon_large.py (geometry, draw_skeleton_labeled, render_panel,
sample_indices) and its load_vq_tokenizer / src FK recovery — so this script adds
ONLY the multi-ckpt forward loop and the N-panel composition. No FK is hand-rolled
(imported from src via the large module), no model constructor is duplicated.

Each ckpt is forwarded on the SAME GT clip; all recons share one transform with GT
so panels are directly comparable. Forward mirrors animate_vqvae_recon.py exactly
(allow_collectives=False, ckpt's bf16 autocast).

Run on a spare GPU node (never a training card):
  python scripts/animate_vqvae_recon_compare.py \
      --ckpts <512>/ep50_model.pt,<1024>/ep50_model.pt,<2048>/ep50_model.pt \
      --labels 512,1024,2048 --species Camel,Dragon,Horse --out runs/_cmp
"""
import argparse
import importlib.util
import os
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Reuse the large labeled renderer's helpers (single source of truth). Importing it
# runs its module-level importlib(avr) which transitively imports scripts.animate
# (os.chdir at import); that module already saves/restores cwd, but guard here too.
_cwd = os.getcwd()
_spec = importlib.util.spec_from_file_location(
    "avrl", str(ROOT / "scripts" / "animate_vqvae_recon_large.py"))
avrl = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(avrl)
os.chdir(_cwd)

from src.data.anytop_dataset import (  # noqa: E402
    AnyTopDataset, collate_fn, _recover_world_positions, _STD_FLOOR,
)
from src.data.anytop_rot6d_fk import recover_from_bvh_rot_np  # noqa: E402
from src.models.graph_salad.batch import GraphMotionBatch  # noqa: E402

load_vq_tokenizer = avrl.load_vq_tokenizer
compute_transform = avrl.compute_transform
render_panel = avrl.render_panel
sample_indices = avrl.sample_indices
get_font = avrl.get_font
GT_COLOR = avrl.GT_COLOR

# distinct colors for the recon panels (cycled if more ckpts than colors)
RECON_PALETTE = [(35, 112, 180), (210, 83, 45), (55, 140, 70), (150, 60, 160), (200, 140, 20)]


def forward_recon(model, raw, item, J, T, amp_enabled, dev):
    """One VQVAE forward -> recovered world positions [T,J,3]. Mirrors
    animate_vqvae_recon.py: de-norm + rot6d FK via src (imported, never copied)."""
    batch = GraphMotionBatch.from_collate_dict(raw)
    if amp_enabled:
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            out = model(batch, allow_collectives=False)
    else:
        out = model(batch, allow_collectives=False)
    std = raw["anytop_std"][0, :J].cpu().numpy()
    mean = raw["anytop_mean"][0, :J].cpu().numpy()
    pred_norm = out["pred_motion"][0, :T, :J, :].float().cpu().numpy()
    pred_raw = pred_norm * (std[None] + _STD_FLOOR) + mean[None]
    parents = np.asarray([int(p) for p in item["parent_indices"][:J]], dtype=int)
    offsets = np.asarray(item["rest_offsets"])[:J]
    pred_world = recover_from_bvh_rot_np(pred_raw, parents, offsets).astype(np.float64)
    T_valid = int(out["frame_mask_recovered"][0].sum().item())
    return pred_world, parents, offsets, T_valid


def make_frame_compare(gt, recons, labels, parents, fi, transform, cell, lw, jr, label, font):
    n = 1 + len(recons)
    W, H = cell
    image = Image.new("RGB", (W * n, H), "white")
    series = [(gt, "GT (truth)", GT_COLOR, True)]
    for k, (rc, lab) in enumerate(zip(recons, labels)):
        series.append((rc, f"RECON {lab}", RECON_PALETTE[k % len(RECON_PALETTE)], False))
    for k, (positions, title, color, axes) in enumerate(series):
        s = positions.copy()
        root = s[fi, 0].copy()
        s[..., 0] -= root[0]; s[..., 2] -= root[2]
        panel = render_panel(s, parents, fi, transform, cell, title, color, lw, jr,
                             axes, label, font)
        image.paste(panel, (W * k, 0))
    return image


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpts", required=True, help="comma-separated ckpt paths")
    ap.add_argument("--labels", required=True, help="comma-separated panel labels (match --ckpts order)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--split", default="val")
    ap.add_argument("--species", default="Camel,Dragon,Horse,Elephant")
    ap.add_argument("--n_per", type=int, default=1)
    ap.add_argument("--anytop_root", type=str, default=None)
    ap.add_argument("--cell_width", type=int, default=760)
    ap.add_argument("--cell_height", type=int, default=680)
    ap.add_argument("--max_frames", type=int, default=48)
    ap.add_argument("--fps", type=float, default=12.0)
    ap.add_argument("--zoom", type=float, default=1.15)
    ap.add_argument("--pad", type=float, default=0.06)
    ap.add_argument("--line_width", type=int, default=3)
    ap.add_argument("--joint_radius", type=int, default=5)
    ap.add_argument("--label", type=int, default=1)
    ap.add_argument("--font_size", type=int, default=13)
    ap.add_argument("--amp_dtype", choices=["fp32", "bf16"], default=None)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        print("  [INFO] CUDA unavailable, falling back to CPU")
        args.device = "cpu"
    dev = torch.device(args.device)
    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)

    ckpt_paths = [s.strip() for s in args.ckpts.split(",") if s.strip()]
    labels = [s.strip() for s in args.labels.split(",") if s.strip()]
    if len(ckpt_paths) != len(labels):
        raise SystemExit(f"--ckpts ({len(ckpt_paths)}) and --labels ({len(labels)}) length mismatch")

    models, tas, cks = [], [], []
    for p in ckpt_paths:
        m, ta, ck = load_vq_tokenizer(p, dev)
        models.append(m); tas.append(ta); cks.append(ck)
        print(f"  loaded {p}: epoch={ck.get('epoch')} num_codes={ta['num_codes']} "
              f"val_total={ck.get('val_total'):.4f}")

    # all ablation ckpts share the merged dataset; reconstruct the split from the
    # FIRST ckpt's training args (they are identical across the ablation).
    ta0 = tas[0]
    amp_dtype = args.amp_dtype or ta0.get("amp_dtype", "bf16")
    amp_enabled = (amp_dtype == "bf16") and dev.type == "cuda"
    root = args.anytop_root or ta0.get("anytop_root") or cks[0].get("data_root")
    ds = AnyTopDataset(
        split=args.split,
        num_frames=ta0.get("max_frames", 64),
        max_joints=ta0.get("max_joints", 64),
        val_frac=ta0.get("val_frac", 0.05),
        seed=ta0.get("seed", 42),
        load_captions=False,
        data_root=root,
    )
    print(f"  dataset: root={root} split={args.split} -> {len(ds)} clips  amp={amp_dtype}")

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
            J = int(item["num_joints"])
            T_clip = int(item["num_frames"])

            # GT world (rot6d FK), shared across panels
            std = raw["anytop_std"][0, :J].cpu().numpy()
            mean = raw["anytop_mean"][0, :J].cpu().numpy()
            gt_norm = np.asarray(item["anytop_x"]).transpose(2, 0, 1)[:T_clip, :J, :]
            gt_raw = gt_norm * (std[None] + _STD_FLOOR) + mean[None]
            parents = np.asarray([int(p) for p in item["parent_indices"][:J]], dtype=int)
            offsets = np.asarray(item["rest_offsets"])[:J]

            # forward every ckpt; intersect valid-T across all so panels align
            recons = []; T = T_clip
            for m in models:
                pw, _, _, T_valid = forward_recon(m, raw, item, J, T_clip, amp_enabled, dev)
                recons.append(pw)
                T = min(T, T_valid)
            if T <= 0:
                print(f"  [SKIP] {sp}: no valid frames", flush=True)
                continue
            recons = [rc[:T] for rc in recons]
            gt_world = recover_from_bvh_rot_np(gt_raw, parents, offsets).astype(np.float64)[:T]

            # renderer-faithfulness self-check (GT-FK vs GT-RIC)
            gt_ric = _recover_world_positions(gt_raw).astype(np.float64)[:T]
            gtbbox = np.linalg.norm(gt_ric.reshape(-1, 3).max(0) - gt_ric.reshape(-1, 3).min(0))
            gt_selfcheck = float(np.linalg.norm(gt_world - gt_ric, axis=-1).mean())
            recon_l2 = [float(np.linalg.norm(rc - gt_world, axis=-1).mean()) for rc in recons]

            # ground-normalize all (y min -> 0)
            for arr in [gt_world] + recons:
                arr[..., 1] -= arr[..., 1].min()

            idxs = sample_indices(T, args.max_frames)
            ps = []
            for arr in [gt_world] + recons:
                c = arr.copy(); roots = c[:, 0].copy()
                c[..., 0] -= roots[:, None, 0]; c[..., 2] -= roots[:, None, 2]
                ps += [c[k] for k in idxs]
            transform = compute_transform(ps, cell, args.pad, args.zoom)

            frames = [make_frame_compare(gt_world, recons, labels, parents, k, transform, cell,
                                         args.line_width, args.joint_radius, bool(args.label), font)
                      for k in idxs]
            out_path = out_dir / f"{sp}_clip{picked[sp]}_compare.gif"
            dur = int(round(1000.0 / max(args.fps, 1e-6)))
            frames[0].save(out_path, save_all=True, append_images=frames[1:],
                           duration=dur, loop=0, optimize=True)
            l2str = " ".join(f"{lab}={v:.4f}" for lab, v in zip(labels, recon_l2))
            line = (f"{sp} clip{picked[sp]}: J={J} T={T} recon_L2[{l2str}] "
                    f"GT_selfcheck_L2={gt_selfcheck:.2e} -> {out_path.name} "
                    f"({cell[0]*(1+len(recons))}x{cell[1]}, {len(frames)}f)")
            print(line, flush=True)
            summary.append(line)
            picked[sp] += 1
            if all(picked[s] >= args.n_per for s in want):
                break

    missing = {s: args.n_per - picked[s] for s in want if picked[s] < args.n_per}
    (out_dir / "compare_summary.txt").write_text("\n".join(summary) + f"\nmissing={missing}\n")
    print(f"\nDONE {sum(picked.values())} gifs -> {out_dir}")
    print("PER-SPECIES picked:", picked)
    if missing:
        print(f"  [WARN] under-filled: {missing}")


if __name__ == "__main__":
    main()
