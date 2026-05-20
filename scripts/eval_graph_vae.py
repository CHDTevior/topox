#!/usr/bin/env python3
"""M1.5 evaluation script — recon evaluator for GraphMotionVAE.

Loads a trained ckpt, runs val/test split, reports per-species & overall
recon metrics (L1 pos, L1 vel, bone-length error). Outputs JSON for
downstream 3-way ablation comparison.

Usage:
  python scripts/eval_graph_vae.py \
    --ckpt runs/m1_5_dynamic/best_model.pt \
    --out runs/m1_5_dynamic/eval_val.json \
    --split val
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.unified_dataset import UnifiedMotionDataset, collate_fn
from src.models.graph_salad import GraphMotionBatch, GraphMotionVAE


def _per_sample_metric(
    pred: torch.Tensor, gt: torch.Tensor,
    joint_mask: torch.Tensor, frame_mask: torch.Tensor,
) -> torch.Tensor:
    """L1 per-sample averaged over valid (t, j). pred/gt: [B,T,J,3]."""
    B = pred.shape[0]
    diff = (pred - gt).abs().sum(dim=-1)  # [B,T,J]
    mask = (joint_mask.unsqueeze(1) & frame_mask.unsqueeze(-1)).to(pred.dtype)
    diff_m = diff * mask
    out = []
    for b in range(B):
        vc = mask[b].sum().clamp(min=1)
        out.append(diff_m[b].sum() / vc)
    return torch.stack(out)


def _per_sample_bone(
    pred_pos: torch.Tensor, rest_bones: torch.Tensor,
    parent_indices: list[list[int]], joint_mask: torch.Tensor,
    frame_mask: torch.Tensor,
) -> torch.Tensor:
    """Per-sample bone length error |||p_j - p_par|| - rest|."""
    B = pred_pos.shape[0]
    dev = pred_pos.device
    out = []
    for b in range(B):
        par_list = parent_indices[b]
        nj = len(par_list)
        if nj == 0:
            out.append(torch.tensor(0.0, device=dev))
            continue
        par_t = torch.tensor(par_list, device=dev, dtype=torch.long)
        valid_b = par_t >= 0
        child_idx = torch.arange(nj, device=dev, dtype=torch.long)
        child = child_idx[valid_b]
        par_v = par_t[valid_b]
        if child.numel() == 0:
            out.append(torch.tensor(0.0, device=dev))
            continue
        p_child = pred_pos[b, :, child, :]
        p_par = pred_pos[b, :, par_v, :]
        bone = (p_child - p_par).norm(dim=-1)  # [T, nj-1]
        rest = rest_bones[b, child].unsqueeze(0)  # [1, nj-1]
        err = (bone - rest).abs()  # [T, nj-1]
        fm = frame_mask[b].unsqueeze(-1).to(pred_pos.dtype)
        err_m = err * fm
        vc = (fm.expand_as(err)).sum().clamp(min=1)
        out.append(err_m.sum() / vc)
    return torch.stack(out)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--split", choices=("val", "test"), default="val")
    p.add_argument("--data_dir", default="data/cs_sparse2full_tgt")
    # max_frames/max_joints default = None → inherit from ckpt (codex M1.5 Medium)
    p.add_argument("--max_frames", type=int, default=None,
                   help="If None (default), inherit from ckpt args")
    p.add_argument("--max_joints", type=int, default=None,
                   help="If None (default), inherit from ckpt args")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    # Deterministic eval seed (codex M1.5 Low)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if not Path(args.ckpt).exists():
        raise RuntimeError(f"[CKPT FAIL] {args.ckpt} does not exist")

    print(f"Loading ckpt: {args.ckpt}")
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=True)
    train_args = ckpt.get("args", {})
    print(f"Train args (from ckpt): pool={train_args.get('pool_type')} "
          f"d_model={train_args.get('d_model')} C={train_args.get('max_coarse')}")

    # Inherit shape defaults from ckpt (codex M1.5 Medium: prevent drift)
    if args.max_frames is None:
        args.max_frames = train_args.get("max_frames", 64)
        print(f"  max_frames inherited from ckpt: {args.max_frames}")
    if args.max_joints is None:
        args.max_joints = train_args.get("max_joints", 160)
        print(f"  max_joints inherited from ckpt: {args.max_joints}")

    # Device — fail loud (codex M1.5 High)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "[DEVICE FAIL] --device cuda requested but torch.cuda.is_available()=False")
    dev = torch.device(args.device)
    print(f"device: {dev}")

    # Reconstruct VAE with ckpt's training hyperparams
    vae = GraphMotionVAE(
        pool_type=train_args["pool_type"],
        d_model=train_args["d_model"], n_heads=train_args["n_heads"],
        d_ff=train_args["d_ff"], n_graph_layers=train_args["n_graph_layers"],
        n_enc_temporal_layers=train_args["n_enc_temporal_layers"],
        n_cross_layers=train_args["n_cross_layers"],
        n_dec_temporal_layers=train_args["n_dec_temporal_layers"],
        n_treeik_layers=train_args["n_treeik_layers"],
        max_coarse=train_args["max_coarse"],
        local_radius=train_args["local_radius"],
        temporal_stride=train_args["temporal_stride"],
        temporal_kernel=train_args["temporal_kernel"],
        dropout=train_args["dropout"],
    ).to(dev)
    vae.load_state_dict(ckpt["model_state_dict"], strict=True)
    vae.eval()

    # Empty/missing split guard (codex M1.5 R2 High)
    split_file = Path(args.data_dir) / "splits" / f"{args.split}.txt"
    if not split_file.exists():
        raise RuntimeError(
            f"[SPLIT FAIL] {split_file} does not exist. "
            f"Available: {list((Path(args.data_dir) / 'splits').glob('*.txt'))}")
    ds = UnifiedMotionDataset(
        data_dirs=[args.data_dir], split=args.split,
        max_joints=args.max_joints, max_frames=args.max_frames,
        normalize=False,
    )
    print(f"{args.split} dataset: {len(ds)} motions")
    if len(ds) == 0:
        raise RuntimeError(f"[SPLIT FAIL] {args.split} split is empty")
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                    collate_fn=collate_fn, num_workers=2)

    per_species_pos = defaultdict(list)
    per_species_vel = defaultdict(list)
    per_species_bone = defaultdict(list)
    pool_diag = defaultdict(list)

    t0 = time.time()
    with torch.no_grad():
        for raw in dl:
            batch = GraphMotionBatch.from_collate_dict({
                k: v.to(dev) if torch.is_tensor(v) else v for k, v in raw.items()
            })
            # Deterministic eval: z = mu (codex M1.5 Critical)
            out = vae(batch, sample=False)
            gt_pos = batch.motion_features[..., :3]
            gt_vel = batch.motion_features[..., 3:6]
            rest_bones = torch.zeros(batch.batch_size, batch.max_joints, device=dev)
            for b in range(batch.batch_size):
                bls = batch.bone_lengths_rest[b]
                rest_bones[b, :len(bls)] = torch.tensor(bls, device=dev, dtype=torch.float32)

            # Codex M1.5 R3 P0: effective frame mask (stride-tail consistency)
            eff_fm = out["frame_mask_recovered"]
            psl = _per_sample_metric(out["pred_pos"], gt_pos,
                                     batch.joint_mask, eff_fm)
            vsl = _per_sample_metric(out["pred_vel"], gt_vel,
                                     batch.joint_mask, eff_fm)
            bsl = _per_sample_bone(out["pred_pos"], rest_bones,
                                   batch.parent_indices, batch.joint_mask,
                                   eff_fm)

            for i, sid in enumerate(batch.skeleton_id):
                per_species_pos[sid].append(psl[i].item())
                per_species_vel[sid].append(vsl[i].item())
                per_species_bone[sid].append(bsl[i].item())

            # Pool diagnostics
            if "coarse_mask" in out:
                active = out["coarse_mask"].sum(dim=-1).float()
                pool_diag["active_coarse"].extend(active.cpu().numpy().tolist())

    dt = time.time() - t0
    print(f"Eval done in {dt:.1f}s")

    # Aggregate
    overall_pos = float(np.mean([v for vals in per_species_pos.values() for v in vals]))
    overall_vel = float(np.mean([v for vals in per_species_vel.values() for v in vals]))
    overall_bone = float(np.mean([v for vals in per_species_bone.values() for v in vals]))

    species_stats = {}
    for sid in per_species_pos:
        n = len(per_species_pos[sid])
        species_stats[sid] = {
            "n": n,
            "pos_l1_mean": float(np.mean(per_species_pos[sid])),
            "pos_l1_std": float(np.std(per_species_pos[sid])),
            "pos_l1_sem": float(np.std(per_species_pos[sid]) / max(np.sqrt(n), 1.0)),
            "vel_l1_mean": float(np.mean(per_species_vel[sid])),
            "bone_l1_mean": float(np.mean(per_species_bone[sid])),
        }
    # Macro-species mean (less biased by sample count imbalance — codex M1.5 Medium)
    macro_pos = float(np.mean([s["pos_l1_mean"] for s in species_stats.values()]))
    macro_vel = float(np.mean([s["vel_l1_mean"] for s in species_stats.values()]))

    result = {
        "ckpt": args.ckpt,
        "split": args.split,
        "n_samples": sum(s["n"] for s in species_stats.values()),
        "pool_type": train_args.get("pool_type"),
        "overall_micro": {  # sample-weighted (small species under-represented)
            "pos_l1": overall_pos, "vel_l1": overall_vel, "bone_l1": overall_bone,
        },
        "overall_macro": {  # species-weighted (recommended for ablation ranking)
            "pos_l1": macro_pos, "vel_l1": macro_vel,
        },
        "per_species": species_stats,
        "pool_diag": {
            "active_coarse_mean": float(np.mean(pool_diag["active_coarse"]))
                if pool_diag["active_coarse"] else None,
            "active_coarse_min": float(np.min(pool_diag["active_coarse"]))
                if pool_diag["active_coarse"] else None,
            "active_coarse_max": float(np.max(pool_diag["active_coarse"]))
                if pool_diag["active_coarse"] else None,
        },
        "WARNING_bone_l1": (
            "bone_l1 is degenerate: FK constructs positions from fixed rest_offsets "
            "(treeik_decoder.py:79), so bone lengths ALWAYS match rest. Use as "
            "FK-constraint sanity check ONLY; do NOT use for ablation ranking."
        ),
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nResults written to {out_path}")
    print(f"\n=== Overall ({train_args.get('pool_type')}, n={result['n_samples']}) ===")
    print(f"  micro:  pos_l1={overall_pos:.4f}  vel_l1={overall_vel:.4f}")
    print(f"  macro:  pos_l1={macro_pos:.4f}  vel_l1={macro_vel:.4f}  ← USE FOR ABLATION RANKING")
    print(f"  (bone_l1={overall_bone:.4f} — DEGENERATE per FK rest_offsets; sanity check only)")
    print(f"\n=== Per-species ===")
    for sid in sorted(species_stats):
        s = species_stats[sid]
        print(f"  {sid:<20} n={s['n']:3d}  pos={s['pos_l1_mean']:.4f}±{s['pos_l1_std']:.3f}  "
              f"vel={s['vel_l1_mean']:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
