"""DECOUPLER (read-only): is the motion-energy loss (fast targets come out
static) in the frozen VAE or in the diffusion sampling?

Take the SAME val clips the diffusion renders, run them through the frozen VAE
encode(sample=False, posterior MEAN) -> decode (NO diffusion), and measure
recon_speed vs GT_speed. If the VAE recon of the FAST Jaguar climb is also slow
(ratio<<1) -> energy is destroyed at the latent level (VAE ceiling, diffusion
can't recover it). If recon ratio ~1 (energy preserved) but the diffusion-sampled
ratio is <<1 -> the bottleneck is the diffusion. Reuses animate_denoiser's exact
de-norm + world-recovery + 4-panel(GT) render path; only z-source differs.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.animate import fk_rest_pose
from scripts.animate_denoiser import make_t2m_large_gif
from scripts.train_denoiser import load_frozen_vae
from src.data.anytop_dataset import (
    AnyTopDataset, collate_fn as anytop_collate_fn, _recover_world_positions, _STD_FLOOR,
)
from src.models.graph_salad.batch import GraphMotionBatch
from src.models.graph_salad.rot6d_fk_recovery import recover_rot6d_fk_positions_torch


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vae_ckpt", required=True)
    ap.add_argument("--caption_emb_cache", required=True)
    ap.add_argument("--anytop_root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--split", default="val")
    ap.add_argument("--species", required=True, help="comma-sep object_types")
    ap.add_argument("--n_per", type=int, default=1)
    ap.add_argument("--fps", type=int, default=8)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    dev = torch.device(args.device if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    vae, ta = load_frozen_vae(args.vae_ckpt, dev)
    temporal_stride = ta["temporal_stride"]
    print(f"VAE loaded: feat_mode={ta['feat_mode']} temporal_stride={temporal_stride}")

    ds = AnyTopDataset(
        split=args.split, num_frames=260, max_joints=ta.get("max_joints", 144),
        caption_emb_cache=args.caption_emb_cache, data_root=args.anytop_root,
    )
    want = [s.strip() for s in args.species.split(",") if s.strip()]
    want_set = set(want)
    match = [i for i, s in enumerate(ds.samples) if s.get("object_type") in want_set]
    picked = {s: 0 for s in want}
    summary = []

    for i in match:
        item = ds[i]
        sp = item["object_type"]
        if sp not in picked or picked[sp] >= args.n_per:
            if all(picked[s] >= args.n_per for s in want):
                break
            continue
        raw = anytop_collate_fn([item])
        raw = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in raw.items()}
        batch = GraphMotionBatch.from_collate_dict(raw)

        # VAE self-recon: REAL posterior-mean latent (no diffusion), then decode.
        with torch.no_grad():
            enc = vae.encode(batch, sample=False)
            dec = vae.decode(enc, batch)
        pred_motion = dec["pred_motion"]  # [B,T,J,13]

        # De-norm + world recovery — identical to animate_denoiser.
        frame_mask_lat = batch.frame_mask.view(
            1, batch.frame_mask.shape[1] // temporal_stride, temporal_stride
        ).all(dim=-1)
        J = int(item["num_joints"])
        T_clip = int(item["num_frames"])
        T_valid = int(frame_mask_lat[0].sum().item() * temporal_stride)
        T = min(T_clip, T_valid)
        std = raw["anytop_std"][0, :J].cpu().numpy()
        mean = raw["anytop_mean"][0, :J].cpu().numpy()
        pred_norm = pred_motion[0, :T, :J, :].cpu().numpy()
        pred_raw = pred_norm * (std[None] + _STD_FLOOR) + mean[None]
        pred_world = _recover_world_positions(pred_raw)
        gt_world = batch.motion_features[0, :T, :J, :3].cpu().numpy()
        parents = [int(p) for p in item["parent_indices"][:J]]
        rest_off = raw["rest_offsets"][0, :J].cpu().numpy()

        pred_raw_t = torch.from_numpy(pred_raw).float()[None]
        rest_off_t = torch.from_numpy(rest_off).float()[None]
        jmask_t = torch.ones(1, J, dtype=torch.bool)
        pred_world_fk = recover_rot6d_fk_positions_torch(
            pred_raw_t, [parents], rest_off_t, jmask_t
        )[0].cpu().numpy()

        g_spd = float(np.linalg.norm(np.diff(gt_world, axis=0), axis=-1).mean())
        p_spd = float(np.linalg.norm(np.diff(pred_world, axis=0), axis=-1).mean())
        pfk_spd = float(np.linalg.norm(np.diff(pred_world_fk, axis=0), axis=-1).mean())
        ratio = p_spd / max(g_spd, 1e-9)

        k = picked[sp]
        gif_path = out_dir / f"{sp}_clip{k}_vaerecon_large.gif"
        static_pose = fk_rest_pose(rest_off, parents)
        prompt_text = (item.get("caption") or "")
        make_t2m_large_gif(
            pred_world, pred_world_fk, static_pose, parents,
            f"[VAE-RECON] {prompt_text}", str(gif_path), fps=args.fps, gt=gt_world,
        )
        line = (f"{sp} clip{k}: J={J} T={T} GT_speed={g_spd:.4f} "
                f"RECON_pose_speed={p_spd:.4f} RECON_fk_speed={pfk_spd:.4f} "
                f"recon_ratio={ratio:.3f} -> {gif_path.name}")
        print(line)
        summary.append(line)
        picked[sp] += 1
        if all(picked[s] >= args.n_per for s in want):
            break

    (out_dir / "vae_recon_summary.txt").write_text("\n".join(summary) + "\n")
    print(f"\nDONE -> {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
