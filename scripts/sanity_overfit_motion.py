"""Sanity overfit experiment — verify GraphMotionVAE architecture CAN learn motion.

Per codex M1.5 frozen-pred audit (2026-05-21): all 4 variants output ~0% GT
motion speed. To diagnose whether this is an architecture issue or a
training-objective issue, we run the simplest possible setup:
  - 1 single clip (Trex, J=61) — no multi-topology, no batching variance
  - pool_type=none, temporal_stride=1 — minimal info bottleneck
  - w_kl=0 — no KL regularization (no posterior collapse pressure)
  - 200 epochs with same-clip train==val (overfit by design)

EXPECTED if architecture is sound: speed_ratio → 1.0, pos_l1 → 0, vel_l1 → 0
EXPECTED if architecture broken: speed_ratio stuck near 0 (frozen even
  on single-clip overfit)

This isolates: "model HAS the capacity to learn motion" vs "objective doesn't
incentivize motion".

Usage (on GPU node):
  python scripts/sanity_overfit_motion.py \\
      --species Trex --clip_idx 0 \\
      --epochs 200 --device cuda \\
      --out runs/sanity_overfit_Trex
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.unified_dataset import UnifiedMotionDataset, collate_fn
from src.models.graph_salad import (
    GraphMotionBatch,
    GraphMotionVAE,
    compute_total_loss,
)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--species", default="Trex")
    p.add_argument("--clip_idx", type=int, default=0,
                   help="index of the matching-species sample to pick")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--lr", type=float, default=4e-4)
    p.add_argument("--w_pos", type=float, default=1.0)
    p.add_argument("--w_vel", type=float, default=1.0)
    p.add_argument("--w_kl", type=float, default=0.0,
                   help="w_kl=0 disables KL regularization (no posterior collapse pressure)")
    p.add_argument("--d_model", type=int, default=384)
    p.add_argument("--n_heads", type=int, default=8)
    p.add_argument("--d_ff", type=int, default=1024)
    p.add_argument("--data_dir", default="data/cs_sparse2full_tgt")
    p.add_argument("--max_frames", type=int, default=64)
    p.add_argument("--max_joints", type=int, default=160)
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--temporal_stride", type=int, default=1,
                   help="1=no temporal compress (sanity default); 4=full config")
    p.add_argument("--pool_type", default="none",
                   choices=("none", "dynamic", "deterministic", "soft_deterministic"))
    p.add_argument("--max_clips", type=int, default=1,
                   help="0=all clips of species, N=first N clips. >1 means batched training")
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("[DEVICE FAIL] cuda requested but unavailable")
    dev = torch.device(args.device)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "sanity_overfit.log"
    metrics_path = out_dir / "sanity_metrics.jsonl"

    def log(msg: str) -> None:
        print(msg, flush=True)
        with open(log_path, "a") as f:
            f.write(msg + "\n")

    log(f"=== SANITY OVERFIT — species={args.species} clip={args.clip_idx} ===")
    log(f"Config: pool_type=none, temporal_stride=1, w_kl={args.w_kl}, lr={args.lr}, epochs={args.epochs}")

    # Load dataset, find one clip of the requested species
    ds = UnifiedMotionDataset(
        data_dirs=[args.data_dir], split="train",
        max_joints=args.max_joints, max_frames=args.max_frames,
        normalize=False,
    )
    # species == "*" means use ALL species in train (full multi-species sanity)
    if args.species == "*":
        matching_indices = list(range(len(ds.samples)))
        log(f"Multi-species mode: using ALL {len(matching_indices)} train clips")
    else:
        matching_indices = [i for i, s in enumerate(ds.samples)
                           if str(s.get("skeleton_id", "")) == args.species]
        if not matching_indices:
            raise RuntimeError(f"No {args.species} samples in train split")
    # Subset
    if args.max_clips > 0:
        if args.clip_idx + args.max_clips > len(matching_indices):
            args.max_clips = len(matching_indices) - args.clip_idx
        picked = matching_indices[args.clip_idx : args.clip_idx + args.max_clips]
    else:
        picked = matching_indices
    log(f"Picked {len(picked)} clips (species={args.species}, max_clips={args.max_clips})")

    class Subset(torch.utils.data.Dataset):
        def __init__(self, base, ids):
            self.base, self.ids = base, ids
        def __len__(self): return len(self.ids)
        def __getitem__(self, i): return self.base[self.ids[i]]

    subs = Subset(ds, picked)
    dl = DataLoader(subs, batch_size=args.batch_size, shuffle=True,
                    collate_fn=collate_fn, num_workers=0, drop_last=False)

    # Build VAE — pool_type/stride configurable for ablation
    max_coarse_use = 64 if args.pool_type != "none" else args.max_joints
    vae = GraphMotionVAE(
        pool_type=args.pool_type,
        pool_tau=1.0 if args.pool_type == "soft_deterministic" else None,
        d_model=args.d_model, n_heads=args.n_heads, d_ff=args.d_ff,
        n_graph_layers=2, n_enc_temporal_layers=1,
        n_cross_layers=1, n_dec_temporal_layers=1,
        n_treeik_layers=1,
        max_coarse=max_coarse_use,
        local_radius=8,
        temporal_stride=args.temporal_stride,
        temporal_kernel=9,
        dropout=0.0,
    ).to(dev)
    n_params = sum(pp.numel() for pp in vae.parameters())
    log(f"VAE params: {n_params:,}")

    opt = torch.optim.AdamW(vae.parameters(), lr=args.lr)

    loss_weights = {
        "pos": args.w_pos, "vel": args.w_vel,
        "vel_consistency": 0.0,  # codex finding: structurally inert anyway
        "kl": args.w_kl,
        "bone": 0.0,             # FK-degenerate
        "pool_aux": 0.0,         # no-pool, irrelevant
    }

    t0 = time.time()
    log(f"Iter per epoch: {len(dl)}")

    def make_batch(raw):
        return GraphMotionBatch.from_collate_dict({
            k: v.to(dev) if torch.is_tensor(v) else v for k, v in raw.items()
        })

    def eval_speed_ratio(batch):
        vae.eval()
        with torch.no_grad():
            out_e = vae(batch, sample=False)
            pp = out_e["pred_pos"]
            gp = batch.motion_features[..., :3]
            pp_d = (pp[:, 1:] - pp[:, :-1]).norm(dim=-1)
            gp_d = (gp[:, 1:] - gp[:, :-1]).norm(dim=-1)
            mask = batch.joint_mask.unsqueeze(1) & out_e["frame_mask_recovered"][:, 1:].unsqueeze(-1)
            mf = mask.to(pp.dtype)
            denom = mf.sum().clamp(min=1.0)
            pred_speed = ((pp_d * mf).sum() / denom).item()
            gt_speed = ((gp_d * mf).sum() / denom).item()
            ratio = pred_speed / max(gt_speed, 1e-8)
            pos_diff = (pp - gp).abs().sum(dim=-1)
            pos_mask = batch.joint_mask.unsqueeze(1) & out_e["frame_mask_recovered"].unsqueeze(-1)
            pos_l1 = ((pos_diff * pos_mask.to(pos_diff.dtype)).sum() / pos_mask.sum().clamp(min=1.0)).item()
        return ratio, pred_speed, gt_speed, pos_l1

    for epoch in range(args.epochs):
        vae.train()
        ep_losses = []
        for raw in dl:
            batch = make_batch(raw)
            out = vae(batch)
            gt_pos = batch.motion_features[..., :3]
            gt_vel = batch.motion_features[..., 3:6]
            rest_bones = torch.zeros(batch.batch_size, batch.max_joints, device=dev)
            for b in range(batch.batch_size):
                bls = batch.bone_lengths_rest[b]
                rest_bones[b, :len(bls)] = torch.tensor(bls, device=dev, dtype=torch.float32)
            eff_fm = out["frame_mask_recovered"]
            losses = compute_total_loss(
                pred_pos=out["pred_pos"], gt_pos=gt_pos,
                pred_vel=out["pred_vel"], gt_vel=gt_vel,
                mu=out["mu"], logvar=out["logvar"],
                pool_aux_outputs=out["pool_aux_outputs"],
                joint_mask=batch.joint_mask,
                frame_mask=eff_fm,
                coarse_mask=out["coarse_mask"],
                frame_mask_lat=out["frame_mask_lat"],
                rest_bone_lengths=rest_bones,
                parent_indices=batch.parent_indices,
                fps=batch.fps,
                weights=loss_weights,
            )
            opt.zero_grad()
            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(vae.parameters(), max_norm=10.0)
            opt.step()
            ep_losses.append({k: v.item() for k, v in losses.items()})

        # Eval every 10 ep
        if epoch % 10 == 0 or epoch == args.epochs - 1:
            # Eval on first batch to get speed ratio (representative)
            eval_raw = next(iter(dl))
            eval_batch = make_batch(eval_raw)
            ratio, pred_speed, gt_speed, pos_l1 = eval_speed_ratio(eval_batch)
            mean_total = sum(L["total"] for L in ep_losses) / len(ep_losses)
            mean_kl = sum(L["kl"] for L in ep_losses) / len(ep_losses)
            flag = "🥶FROZEN" if ratio < 0.1 else ("⚠LOW" if ratio < 0.5 else ("◐MOTION" if ratio < 0.9 else "✓OK"))
            log(f"[ep{epoch}] train_loss={mean_total:.4f} pos_l1={pos_l1:.4f} "
                f"ratio={ratio:.4f} {flag} (pred={pred_speed:.4f} gt={gt_speed:.4f}) "
                f"kl={mean_kl:.4f}")
            with open(metrics_path, "a") as f:
                f.write(json.dumps({
                    "epoch": epoch, "train_loss": mean_total,
                    "pos_l1": pos_l1, "speed_ratio": ratio,
                    "pred_speed": pred_speed, "gt_speed": gt_speed,
                }) + "\n")

    log(f"=== sanity overfit done in {time.time() - t0:.1f}s ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
