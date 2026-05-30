"""Diagnose v3 NaN: run the 32 samples that were in ep1343 batch 32 (combined
rank 0 + rank 1) through the VAE → noise → denoiser → loss path, looking for
the first NaN/Inf injection point.
"""
from pathlib import Path
import sys
import os
import numpy as np
import torch

ROOT = Path("/scratch/ts1v23/workspace/noKslot_clean")
sys.path.insert(0, str(ROOT))

from src.data.anytop_dataset import AnyTopDataset, collate_fn as anytop_collate_fn
from src.models.graph_salad.batch import GraphMotionBatch
from scripts.animate_denoiser import load_frozen_vae, load_denoiser

try:
    from diffusers import DDIMScheduler
except ImportError:
    raise SystemExit("diffusers not installed")


def main() -> int:
    VAE_CKPT = ROOT / "runs/m1_7_anytop13_edge_segment_C96_fulldata_ddp2a100_seed42/last_model.pt"
    DEN_CKPT = ROOT / "runs/m2_denoiser_v3_h200_ddp2_lr7e-4_4000ep_fulldata_seed42/last_model.pt"
    CAP_CACHE = ROOT / "data/anytop_caption_t5_1070_multi.npz"

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {dev}")
    print(f"VAE ckpt: {VAE_CKPT}")
    print(f"Denoiser ckpt: {DEN_CKPT}")

    vae, ta = load_frozen_vae(str(VAE_CKPT), dev)
    denoiser, dck = load_denoiser(str(DEN_CKPT), dev)
    print(f"denoiser ckpt epoch={dck.get('epoch')} val={dck.get('val_denoise')}")

    sched = DDIMScheduler(
        num_train_timesteps=1000, beta_start=0.00085, beta_end=0.012,
        beta_schedule="scaled_linear", prediction_type="v_prediction",
        clip_sample=False,
    )

    # Reproduce DistributedSampler set_epoch(1343) batch 32 — ALL 32 samples
    # (rank 0 + rank 1 combined since both ranks compute loss together via DDP)
    ds = AnyTopDataset(
        split="all", num_frames=64, max_joints=143,
        caption_emb_cache=str(CAP_CACHE),
    )
    N = len(ds)
    g = torch.Generator(); g.manual_seed(0 + 1343)
    indices = torch.randperm(N, generator=g).tolist()
    world_size = 2; batch_size = 16
    total = (N // world_size) * world_size
    indices = indices[:total]
    rank0 = indices[0::world_size][32*batch_size:33*batch_size]
    rank1 = indices[1::world_size][32*batch_size:33*batch_size]
    all_idx = rank0 + rank1

    # Walk each sample individually
    print(f"\n=== Probing {len(all_idx)} samples individually ===")
    nan_samples = []
    for i, idx in enumerate(all_idx):
        item = ds[idx]
        sp = item["object_type"]
        mid = item.get("motion_id", "?")
        J = int(item["num_joints"])
        T = int(item["num_frames"])
        raw = anytop_collate_fn([item])
        raw = {k: v.to(dev) if torch.is_tensor(v) else v for k, v in raw.items()}
        batch = GraphMotionBatch.from_collate_dict(raw)

        with torch.no_grad():
            enc = vae.encode(batch, sample=True)
        z0 = enc["z"]
        z0_nan = int(torch.isnan(z0).sum())
        z0_inf = int(torch.isinf(z0).sum())
        z0_max = float(z0.abs().max())

        # Random noise + timestep (deterministic for reproducibility)
        torch.manual_seed(42 + idx)  # per-sample seed
        noise = torch.randn_like(z0)
        timesteps = torch.randint(0, 1000, (1,), device=dev).long()
        z_t = sched.add_noise(z0, noise, timesteps)
        v_target = sched.get_velocity(z0, noise, timesteps)
        zt_nan = int(torch.isnan(z_t).sum())
        zt_max = float(z_t.abs().max())

        # Denoiser forward
        has_text = batch.has_text.to(dev)
        text_emb = batch.caption_emb.to(dev) * has_text[:, None].to(batch.caption_emb.dtype)
        with torch.no_grad():
            v_pred = denoiser(
                z_t=z_t, timesteps=timesteps, text=text_emb,
                adjacency=enc["pooled_adjacency"], geodesic_dist=enc["pooled_geodesic"],
                coarse_mask=enc["coarse_mask"], frame_mask=enc["frame_mask_lat"],
                pooled_skeleton_embeddings=enc["pooled_skeleton_embeddings"],
                has_text=has_text, validate_inputs=False,
            )
        vpred_nan = int(torch.isnan(v_pred).sum())
        vpred_inf = int(torch.isinf(v_pred).sum())
        vpred_max = float(v_pred.abs().max())

        # Compute single-sample loss
        mask = (enc["coarse_mask"][:, None, :, None] & enc["frame_mask_lat"][:, :, None, None]).to(v_pred.dtype)
        diff_sq = (v_pred - v_target).pow(2) * mask
        loss = diff_sq.sum() / (mask.sum() * v_pred.shape[-1]).clamp(min=1.0)
        loss_val = float(loss)
        loss_nan = (loss_val != loss_val) or (loss_val == float("inf"))

        tag = ""
        if z0_nan or z0_inf or zt_nan or vpred_nan or vpred_inf or loss_nan:
            tag = " ⚠ NAN/INF"
            nan_samples.append((idx, sp, mid, dict(
                z0_nan=z0_nan, z0_inf=z0_inf, z0_max=z0_max,
                zt_max=zt_max, vpred_nan=vpred_nan, vpred_inf=vpred_inf,
                vpred_max=vpred_max, loss=loss_val,
            )))
        print(f"  [{i:2d}] {sp}/{mid} J={J} T={T} | z0_max={z0_max:.2f} z0_nan={z0_nan} z0_inf={z0_inf} | "
              f"zt_max={zt_max:.2f} | vp_max={vpred_max:.2f} vp_nan={vpred_nan} vp_inf={vpred_inf} | loss={loss_val:.4f}{tag}")

    print(f"\n=== Summary ===")
    print(f"NaN/Inf-triggering samples: {len(nan_samples)} / {len(all_idx)}")
    for idx, sp, mid, info in nan_samples:
        print(f"  global_idx={idx} {sp}/{mid}: {info}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
