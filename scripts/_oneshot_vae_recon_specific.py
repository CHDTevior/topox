"""One-shot VAE recon on a SPECIFIC motion (by motion_id substring).
Renders GT vs pred side-by-side. Use for debugging: if recon is already bad,
the visual problem is VAE-side (not denoiser). Caption-agnostic (recon only).
"""
from pathlib import Path
import sys
import os
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as anim

ROOT = Path("/scratch/ts1v23/workspace/noKslot_clean")
sys.path.insert(0, str(ROOT))

from src.data.anytop_dataset import (
    AnyTopDataset, collate_fn as anytop_collate_fn, _recover_world_positions,
    _STD_FLOOR,
)
from src.models.graph_salad.batch import GraphMotionBatch
from scripts.animate_denoiser import load_frozen_vae


def render_gtvspred(gt, pred, parents, out_path, fps=8, stride=1):
    """Side-by-side GT vs pred animated gif."""
    T, J, _ = gt.shape
    fig, axes = plt.subplots(1, 2, figsize=(10, 5), subplot_kw={"projection": "3d"})
    for ax in axes:
        ax.set_box_aspect([1, 1, 1])
    axes[0].set_title("GT")
    axes[1].set_title("VAE recon")

    # Determine bounds
    all_pts = np.concatenate([gt.reshape(-1, 3), pred.reshape(-1, 3)], axis=0)
    cmin = all_pts.min(0); cmax = all_pts.max(0)
    for ax in axes:
        ax.set_xlim(cmin[0], cmax[0])
        ax.set_ylim(cmin[1], cmax[1])
        ax.set_zlim(cmin[2], cmax[2])
        ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])

    bones = [(j, p) for j, p in enumerate(parents) if p >= 0]

    def draw_frame(t, ax, frame):
        ax.cla()
        ax.set_xlim(cmin[0], cmax[0])
        ax.set_ylim(cmin[1], cmax[1])
        ax.set_zlim(cmin[2], cmax[2])
        ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
        ax.scatter(frame[:, 0], frame[:, 1], frame[:, 2], c="r", s=15)
        for j, p in bones:
            ax.plot([frame[j, 0], frame[p, 0]],
                    [frame[j, 1], frame[p, 1]],
                    [frame[j, 2], frame[p, 2]], c="b", linewidth=1)
        ax.set_title(t)

    def update(i):
        idx = i * stride
        if idx >= T:
            return []
        draw_frame(f"GT  frame {idx}", axes[0], gt[idx])
        draw_frame(f"recon frame {idx}", axes[1], pred[idx])
        return []

    n_frames = T // stride
    ani = anim.FuncAnimation(fig, update, frames=n_frames, interval=1000 // fps)
    ani.save(out_path, writer="pillow", fps=fps)
    plt.close(fig)


def main() -> int:
    TARGET_MOTION = "Die_296"
    SPECIES = "Dragon"
    VAE_CKPT = ROOT / "runs/m1_7_anytop13_edge_segment_C96_fulldata_ddp2a100_seed42/last_model.pt"
    CAP_CACHE = ROOT / "data/anytop_caption_t5_1070_multi.npz"
    OUT = ROOT / "runs/m1_7_anytop13_edge_segment_C96_fulldata_ddp2a100_seed42/qa_dragon_die_recon.gif"

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42); np.random.seed(42)

    print(f"Loading VAE: {VAE_CKPT}")
    vae, ta = load_frozen_vae(str(VAE_CKPT), dev)

    print(f"Searching for {SPECIES} {TARGET_MOTION!r} in val + train...")
    found_item = None
    for split in ("val", "train", "all"):
        ds_kwargs = dict(
            split=split,
            num_frames=ta.get("max_frames", 64),
            max_joints=ta.get("max_joints", 143),
            caption_emb_cache=str(CAP_CACHE),
        )
        if ta.get("anytop_root"):
            ds_kwargs["data_root"] = ta["anytop_root"]
        ds = AnyTopDataset(**ds_kwargs)
        for i in range(len(ds)):
            s = ds.samples[i]
            if s["object_type"] != SPECIES:
                continue
            mid = s.get("motion_id", "") or s.get("source_file", "") or ""
            # check key match against TARGET_MOTION substring
            if TARGET_MOTION in mid:
                print(f"  found in split={split} idx={i} motion_id={mid!r}")
                found_item = ds[i]
                break
        if found_item is not None:
            break
    if found_item is None:
        # fallback: brute search across all
        print(f"  not found by motion_id; trying all by sample keys")
        return 1

    item = found_item
    raw = anytop_collate_fn([item])
    raw = {k: v.to(dev) if torch.is_tensor(v) else v for k, v in raw.items()}
    batch = GraphMotionBatch.from_collate_dict(raw)

    print(f"VAE encode-decode (deterministic, sample=False)")
    with torch.no_grad():
        enc = vae.encode(batch, sample=False)
        dec = vae.decode(enc, batch)
    pred_motion = dec["pred_motion"]

    J = int(item["num_joints"])
    T = int(item["num_frames"])
    std = raw["anytop_std"][0, :J].cpu().numpy()
    mean = raw["anytop_mean"][0, :J].cpu().numpy()
    pred_norm = pred_motion[0, :T, :J, :].cpu().numpy()
    pred_raw = pred_norm * (std[None] + _STD_FLOOR) + mean[None]
    pred_world = _recover_world_positions(pred_raw)
    gt_world = batch.motion_features[0, :T, :J, :3].cpu().numpy()
    parents = [int(p) for p in item["parent_indices"][:J]]

    print(f"  J={J} T={T}")
    g_spd = float(np.linalg.norm(np.diff(gt_world, axis=0), axis=-1).mean())
    p_spd = float(np.linalg.norm(np.diff(pred_world, axis=0), axis=-1).mean())
    print(f"  GT_speed={g_spd:.4f} PRED_speed={p_spd:.4f} ratio={p_spd/max(g_spd,1e-9):.3f}")

    render_gtvspred(gt_world, pred_world, parents, str(OUT), fps=8, stride=2)
    print(f"DONE -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
