"""Smoke test for AnyTop integration (M1.7 iter 1.5).

Verifies:
  1. AnyTopDataset loads + collate produces GraphMotionBatch-compatible batch
  2. GraphMotionVAE (6ch path) consumes the batch, returns pred_pos/pred_vel
  3. Loss + backward + gradient flow (decoder grad is 0 at init by the
     identity-6D init scheme — expected; resolves after step 1)
  4. Renders 1 GT motion — world-space joint positions recovered from AnyTop's
     RIFKE encoding (motion_features[..., :3]) — to a GIF via animate_clip,
     for visual QA per cross-project "可视化 demo > metric" rule.

Run:
  python scripts/smoke_anytop.py --out runs/smoke_anytop --batch 4
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.anytop_dataset import AnyTopDataset, collate_fn  # noqa: E402
from src.models.graph_salad import GraphMotionBatch, GraphMotionVAE  # noqa: E402
from src.models.graph_salad.losses import compute_total_loss  # noqa: E402


def _print_shape_flow(batch, raw_batch, enc_out, dec_out):
    print("\n=== Shape flow ===")
    print(f"  batch.motion_features (pos+vel=6ch)  : {tuple(batch.motion_features.shape)}")
    if "anytop_x" in raw_batch:
        print(f"  raw.anytop_x (13ch J-first, normalized): {tuple(raw_batch['anytop_x'].shape)}")
    if "foot_contact_per_joint" in raw_batch:
        print(f"  raw.foot_contact_per_joint          : {tuple(raw_batch['foot_contact_per_joint'].shape)}")
    print(f"  batch.skeleton_features              : {tuple(batch.skeleton_features.shape)}")
    print(f"  batch.adjacency                      : {tuple(batch.adjacency.shape)}")
    print(f"  batch.geodesic_dist                  : {tuple(batch.geodesic_dist.shape)}")
    print(f"  batch.joint_mask  (n_joints={batch.num_joints.tolist()})")
    print(f"  batch.frame_mask  (n_frames={batch.num_frames.tolist()})")
    print(f"  enc.z (latent)                       : {tuple(enc_out['z'].shape)}")
    print(f"  enc.mu                                : {tuple(enc_out['mu'].shape)}")
    print(f"  enc.coarse_mask                       : {tuple(enc_out['coarse_mask'].shape)}")
    print(f"  enc.frame_mask_lat                    : {tuple(enc_out['frame_mask_lat'].shape)}")
    print(f"  dec.pred_pos                          : {tuple(dec_out['pred_pos'].shape)}")
    print(f"  dec.pred_vel                          : {tuple(dec_out['pred_vel'].shape)}")


def _render_gt_motion(sample, out_dir: Path, max_frames_render: int = 32):
    """Render the GT world-space positions of one sample as gif.

    Iter-1.5 change (codex P2 #11): motion_features now holds WORLD positions
    recovered from AnyTop's RIFKE encoding via _recover_world_positions. So
    we plot motion_features[..., :3] directly — no more de-normalize-and-slice
    of the RIFKE 0:3 channels (which previously made the gif visually
    meaningless for the root joint).
    """
    sys.path.insert(0, str(ROOT / "scripts"))
    from animate import animate_clip  # noqa: E402

    J_real = int(sample["num_joints"])
    T_real = int(sample["num_frames"])
    # World position is the first 3 channels of motion_features after recovery.
    pos = sample["motion_features"][:T_real, :J_real, :3].numpy()  # [T, J, 3]
    parents = sample["parent_indices"][:J_real]

    out_dir.mkdir(parents=True, exist_ok=True)
    T_use = min(T_real, max_frames_render)
    gif_path = out_dir / f"gt_{sample['object_type']}_{sample['motion_id']}.gif"
    title = (f"{sample['object_type']} {sample['motion_id']}  "
             f"J={J_real} T={T_use}  world pos (AnyTop RIFKE -> Cartesian recovery)")
    animate_clip(pos[:T_use], pos[:T_use], parents, str(gif_path),
                 title, stride=2, fps=8)
    print(f"  rendered: {gif_path}")
    return gif_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runs/smoke_anytop")
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--device", default="cpu",
                    help="cpu (default — smoke runs fine on CPU) or cuda")
    ap.add_argument("--pool_type", default="dynamic",
                    choices=("dynamic", "deterministic", "soft_deterministic", "none"))
    ap.add_argument("--pool_tau", type=float, default=None,
                    help="Required for --pool_type soft_deterministic")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- 1. Load dataset ----
    print("=== AnyTopDataset (val split, batch=4) ===")
    ds = AnyTopDataset(split="val", num_frames=64, max_joints=143)

    # ---- 2. Build batch via collate_fn ----
    items = [ds[i] for i in range(args.batch)]
    batch_raw_dict = collate_fn(items)
    batch = GraphMotionBatch.from_collate_dict(batch_raw_dict)
    print(f"  batch built OK — B={args.batch}, "
          f"unique object_types in batch: "
          f"{set(batch_raw_dict['skeleton_id'])}")

    # ---- 3. Build VAE (default 6ch path — same config as M1.5R dyn) ----
    print(f"\n=== GraphMotionVAE (pool={args.pool_type}) ===")
    if args.pool_type == "soft_deterministic" and args.pool_tau is None:
        args.pool_tau = 1.0
    vae = GraphMotionVAE(
        pool_type=args.pool_type, pool_tau=args.pool_tau,
        d_model=384, n_heads=8, d_ff=1024,
        n_graph_layers=4, n_enc_temporal_layers=2,
        n_cross_layers=3, n_dec_temporal_layers=2, n_treeik_layers=3,
        max_coarse=64, local_radius=8,
        motion_feat_dim=6, joint_feat_dim=9,
        temporal_kernel=9, temporal_stride=4, dropout=0.1,
    )
    vae.encoder.use_name_embed = True
    n_params = sum(p.numel() for p in vae.parameters())
    print(f"  VAE params: {n_params:,}")

    dev = torch.device(args.device)
    if args.device == "cuda" and not torch.cuda.is_available():
        print("  [INFO] CUDA unavailable, falling back to CPU")
        dev = torch.device("cpu")
    vae.to(dev)
    # Move batch tensors to device. GraphMotionBatch is a typed view, so we
    # move the underlying dict tensors then re-wrap.
    for k, v in list(batch_raw_dict.items()):
        if isinstance(v, torch.Tensor):
            batch_raw_dict[k] = v.to(dev)
    batch = GraphMotionBatch.from_collate_dict(batch_raw_dict)
    print(f"  device: {dev}")

    # ---- 4. Forward (train mode, sample=True) ----
    vae.train()
    out = vae(batch, sample=True)

    # ---- 5. Loss + backward ----
    # Minimal loss config — anti-frozen P3 weights (matches M1.5R best config).
    # Extract GT pos/vel from motion_features (channels 0:3 + 3:6 of the 6ch view).
    gt_pos = batch.motion_features[..., :3]
    gt_vel = batch.motion_features[..., 3:6]
    # Rest bone lengths: derive from rest_offsets (norm per joint).
    rest_bone = torch.linalg.norm(batch.rest_offsets, dim=-1)  # [B, J]
    loss_dict = compute_total_loss(
        pred_pos=out["pred_pos"],
        gt_pos=gt_pos,
        pred_vel=out["pred_vel"],
        gt_vel=gt_vel,
        mu=out["mu"],
        logvar=out["logvar"],
        pool_aux_outputs=out["pool_aux_outputs"],
        joint_mask=batch.joint_mask,
        frame_mask=batch.frame_mask,
        coarse_mask=out["coarse_mask"],
        frame_mask_lat=out["frame_mask_lat"],
        rest_bone_lengths=rest_bone,
        parent_indices=batch.parent_indices,
        fps=batch.fps,
        weights={"pos": 0.1, "vel": 10.0, "vel_normalized": 1.0,
                 "speed_mag": 10.0, "kl": 1e-3, "bone": 1.0, "pool_aux": 0.5},
    )
    total = loss_dict["total"]
    print(f"\n=== Loss (anti-frozen P3 weights) ===")
    for k, v in loss_dict.items():
        if isinstance(v, torch.Tensor):
            print(f"  {k}: {float(v):.4f}")

    total.backward()

    # ---- 6. Verify grads flow ----
    grad_summary = {"encoder": 0.0, "pool": 0.0, "decoder": 0.0, "treeik_head": 0.0}
    grad_counts = {k: 0 for k in grad_summary}
    for name, p in vae.named_parameters():
        if p.grad is None:
            continue
        gn = float(p.grad.norm())
        for key in grad_summary:
            if name.startswith(key + "."):
                grad_summary[key] += gn
                grad_counts[key] += 1
    print(f"\n=== Gradient norms (sum per module group) ===")
    for k in grad_summary:
        if grad_counts[k] > 0:
            print(f"  {k}: sum_grad_norm={grad_summary[k]:.4f} "
                  f"(over {grad_counts[k]} params)")
        else:
            print(f"  {k}: NO PARAMS / NO GRADS (likely intentional)")

    _print_shape_flow(batch, batch_raw_dict, out, out)

    # ---- 7. Render GT motion ----
    print(f"\n=== Visual QA (GT render, channels 0:3 denormalized) ===")
    _render_gt_motion(items[0], out_dir / "qa")

    # ---- 8. Summary ----
    print(f"\n=== Smoke PASS ===")
    print(f"  out_dir: {out_dir.resolve()}")
    print(f"  total loss: {float(total):.4f}")
    print(f"  encoder grad: {grad_summary['encoder']:.4f}")
    print(f"  decoder grad: {grad_summary['decoder']:.4f}")
    print(f"  treeik_head grad: {grad_summary['treeik_head']:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
