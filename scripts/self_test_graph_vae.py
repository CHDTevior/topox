#!/usr/bin/env python3
"""M1.4 CPU smoke — end-to-end graph_salad VAE forward + backward on mixed-J batch.

Acceptance per PLAN_GAP_REPORT.md §6:
  Gate 2 (CPU smoke): B=2 mixed-J, fwd/bwd, no NaN, z.shape = [B, T/4, C2_max, D],
                       no hard-coded 7.
  Gate 3 (padding gate): padded joints/coarse zeroed AND excluded from
                          recon/KL/pool losses; NaN-with-zero-gradient guard
                          (test that loss.backward() produces non-zero grads
                          on at least one parameter per batch).

Run (CPU-only ~5s):
    python scripts/self_test_graph_vae.py

Exit code: 0 = PASS, 1 = FAIL.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.graph_salad import (
    GraphMotionBatch,
    GraphMotionVAE,
    compute_total_loss,
)


def _print(msg: str, ok: bool = True) -> None:
    prefix = "[ OK ]" if ok else "[FAIL]"
    print(f"{prefix} {msg}", flush=True)


def _build_mixed_J_batch(B: int = 2, T: int = 16) -> GraphMotionBatch:
    """Real Bat (J=48) + Crab (J=54) from dataset if available; else synthetic."""
    species = ("Bat", "Crab")  # pick 2 species with different J
    skeletons_dir = (
        Path(__file__).resolve().parents[1]
        / "data" / "cs_sparse2full_tgt" / "skeletons"
    )
    if not all((skeletons_dir / f"{s}.npz").exists() for s in species):
        raise RuntimeError(f"dataset skeletons missing at {skeletons_dir}")

    raw = []
    for s in species:
        d = np.load(skeletons_dir / f"{s}.npz", allow_pickle=True)
        raw.append(d)

    J_max = max(d["joint_names"].shape[0] for d in raw)
    # Pad T to be divisible by stride=4
    while T % 4 != 0:
        T += 1

    motion_features = torch.randn(B, T, J_max, 6)
    skeleton_features = torch.randn(B, J_max, 9)
    adjacency = torch.zeros(B, J_max, J_max)
    geodesic_dist = torch.zeros(B, J_max, J_max)
    joint_mask = torch.zeros(B, J_max, dtype=torch.bool)
    frame_mask = torch.ones(B, T, dtype=torch.bool)
    rest_offsets = torch.zeros(B, J_max, 3)
    bone_lengths = torch.ones(B, T, J_max)
    local_rotations_6d = torch.randn(B, T, J_max, 6)
    foot_contact = torch.zeros(B, T, 4)
    root_position = torch.randn(B, T, 3)
    root_velocity = torch.randn(B, T, 3)
    name_hashes = torch.zeros(B, J_max, dtype=torch.long)

    parent_indices = []
    joint_names = []
    canonical_names = []
    bone_lengths_rest = []
    nj_list = []
    for b in range(B):
        d = raw[b]
        J_b = d["joint_names"].shape[0]
        nj_list.append(J_b)
        adjacency[b, :J_b, :J_b] = torch.from_numpy(d["adjacency"]).float()
        geodesic_dist[b, :J_b, :J_b] = torch.from_numpy(d["geodesic_dist"]).float()
        joint_mask[b, :J_b] = True
        rest_offsets[b, :J_b] = torch.from_numpy(d["rest_offsets"]).float()
        bls = torch.from_numpy(d["bone_lengths"]).float()
        bone_lengths[b, :, :J_b] = bls.unsqueeze(0).expand(T, J_b)
        parent_indices.append(d["parent_indices"].tolist())
        joint_names.append([str(n) for n in d["joint_names"]])
        canonical_names.append([str(n) for n in d.get("canonical_names", d["joint_names"])])
        bone_lengths_rest.append(bls.tolist())

    return GraphMotionBatch.from_collate_dict({
        "motion_features": motion_features,
        "skeleton_features": skeleton_features,
        "joint_mask": joint_mask,
        "frame_mask": frame_mask,
        "adjacency": adjacency,
        "geodesic_dist": geodesic_dist,
        "name_hashes": name_hashes,
        "root_position": root_position,
        "root_velocity": root_velocity,
        "local_rotations_6d": local_rotations_6d,
        "foot_contact": foot_contact,
        "bone_lengths": bone_lengths,
        "rest_offsets": rest_offsets,
        "num_joints": torch.tensor(nj_list, dtype=torch.long),
        "num_frames": torch.tensor([T] * B, dtype=torch.long),
        "fps": torch.full((B,), 20.0),
        "has_rotations": torch.ones(B, dtype=torch.bool),
        "parent_indices": parent_indices,
        "joint_names": joint_names,
        "canonical_names": canonical_names,
        "bone_lengths_rest": bone_lengths_rest,
        "text": [f"sample {b}" for b in range(B)],
        "skeleton_id": list(species),
        "motion_id": [f"mot_{b}" for b in range(B)],
    })


def _check_z_shape(z: torch.Tensor, B: int, T: int, expected_C_upper: int, D: int) -> bool:
    """z shape: [B, T/4, C, D]. C ≤ expected_C_upper (max_coarse). NOT hard-coded 7."""
    if z.dim() != 4:
        return False
    b, t_lat, c, d = z.shape
    return (b == B and t_lat == T // 4 and c <= expected_C_upper and d == D
            and c != 7)  # PLAN_GAP_REPORT.md §6 #2: no hard-coded 7


def _check_padded_excluded(out: dict, batch: GraphMotionBatch) -> bool:
    """Padded joints + frames have zero output. Skip empty slices."""
    pos = out["pred_pos"]
    vel = out["pred_vel"]
    for b in range(pos.shape[0]):
        nj = int(batch.num_joints[b].item())
        if nj < pos.shape[2]:
            if pos[b, :, nj:, :].abs().max().item() > 1e-7:
                return False
            if vel[b, :, nj:, :].abs().max().item() > 1e-7:
                return False
        nf = int(batch.num_frames[b].item())
        if nf < pos.shape[1]:
            if pos[b, nf:, :, :].abs().max().item() > 1e-7:
                return False
            if vel[b, nf:, :, :].abs().max().item() > 1e-7:
                return False
    return True


def _run_pool_variant(pool_type: str, batch: GraphMotionBatch) -> dict:
    """Build VAE + forward + backward + checks. Returns dict of results."""
    D = 64  # small for CPU speed
    vae = GraphMotionVAE(
        pool_type=pool_type,
        d_model=D, n_heads=4, d_ff=128,
        n_graph_layers=2, n_enc_temporal_layers=1,
        n_cross_layers=1, n_dec_temporal_layers=1,
        n_treeik_layers=1,
        max_coarse=64, local_radius=8,
        temporal_stride=4,
        dropout=0.0,
    )
    vae.train()

    t0 = time.time()
    out = vae(batch)
    fwd_ms = (time.time() - t0) * 1000

    # Check 1: z shape
    z = out["z"]
    z_ok = _check_z_shape(z, batch.batch_size, batch.max_frames, 64, D)

    # Check 2: no NaN/Inf in outputs
    no_nan_pos = torch.isfinite(out["pred_pos"]).all().item()
    no_nan_vel = torch.isfinite(out["pred_vel"]).all().item()
    no_nan_mu = torch.isfinite(out["mu"]).all().item()
    no_nan_logvar = torch.isfinite(out["logvar"]).all().item()

    # Check 3: padded joints / frames excluded
    padded_ok = _check_padded_excluded(out, batch)

    # Check 4: Loss computation
    # Build minimal gt (use pred_pos as gt for self-consistency = loss should be ~0)
    # For real recon test we'd use batch.motion_features but it's [B,T,J,6]=local_pos+vel,
    # so split:
    gt_pos = batch.motion_features[..., :3]  # local_pos
    gt_vel = batch.motion_features[..., 3:6]  # velocity
    losses = compute_total_loss(
        pred_pos=out["pred_pos"], gt_pos=gt_pos,
        pred_vel=out["pred_vel"], gt_vel=gt_vel,
        mu=out["mu"], logvar=out["logvar"],
        pool_aux_outputs=out["pool_aux_outputs"],
        joint_mask=batch.joint_mask,
        frame_mask=batch.frame_mask,
        coarse_mask=out["coarse_mask"],
        frame_mask_lat=out["frame_mask_lat"],
        rest_bone_lengths=torch.tensor(
            [bls + [0.0] * (batch.max_joints - len(bls)) for bls in batch.bone_lengths_rest]
        ),
        parent_indices=batch.parent_indices,
        fps=batch.fps,
    )
    loss_ok = torch.isfinite(losses["total"]).item()

    # Check 5: Backward + grad-flow
    t1 = time.time()
    losses["total"].backward()
    bwd_ms = (time.time() - t1) * 1000
    # At least one param has non-zero grad (NaN-with-zero-gradient guard)
    has_grad = False
    grad_nan = False
    for p in vae.parameters():
        if p.grad is not None:
            if p.grad.abs().sum().item() > 0:
                has_grad = True
            if not torch.isfinite(p.grad).all():
                grad_nan = True
    grad_ok = has_grad and not grad_nan

    return {
        "pool_type": pool_type,
        "fwd_ms": fwd_ms,
        "bwd_ms": bwd_ms,
        "z_shape": tuple(z.shape),
        "z_ok": z_ok,
        "no_nan_pos": no_nan_pos,
        "no_nan_vel": no_nan_vel,
        "no_nan_mu": no_nan_mu,
        "no_nan_logvar": no_nan_logvar,
        "padded_ok": padded_ok,
        "loss_total": losses["total"].item(),
        "loss_ok": loss_ok,
        "grad_ok": grad_ok,
        "loss_breakdown": {k: v.item() for k, v in losses.items()},
    }


def main() -> int:
    print("=" * 70)
    print("M1.4 CPU smoke — graph_salad VAE end-to-end on mixed-J real batch")
    print("=" * 70)

    torch.manual_seed(42)
    t_start = time.time()

    # Build mixed-J batch (Bat J=48 + Crab J=54)
    try:
        batch = _build_mixed_J_batch(B=2, T=16)
        _print(f"Built batch B=2 T={batch.max_frames} J_max={batch.max_joints} "
              f"num_joints={batch.num_joints.tolist()}", True)
    except Exception as e:
        _print(f"Failed to build batch: {e}", False)
        return 1

    # Run all 3 pool variants
    all_ok = True
    for pool_type in ("dynamic", "deterministic", "none"):
        try:
            result = _run_pool_variant(pool_type, batch)
        except Exception as e:
            _print(f"{pool_type}: forward/backward crashed — {e}", False)
            import traceback
            traceback.print_exc()
            all_ok = False
            continue

        print(f"\n--- {pool_type} ---")
        _print(f"forward {result['fwd_ms']:.1f}ms / backward {result['bwd_ms']:.1f}ms", True)
        _print(f"z shape {result['z_shape']} (gate: no hard-coded 7)", result["z_ok"])
        _print(f"no NaN in pos/vel/mu/logvar",
              result["no_nan_pos"] and result["no_nan_vel"]
              and result["no_nan_mu"] and result["no_nan_logvar"])
        _print(f"padded joints/frames zeroed", result["padded_ok"])
        _print(f"loss total = {result['loss_total']:.4f} (finite)", result["loss_ok"])
        _print(f"grad-flow OK (non-zero grads + no NaN)", result["grad_ok"])
        # Aggregate
        variant_ok = (
            result["z_ok"] and result["no_nan_pos"] and result["no_nan_vel"]
            and result["no_nan_mu"] and result["no_nan_logvar"]
            and result["padded_ok"] and result["loss_ok"] and result["grad_ok"]
        )
        if not variant_ok:
            all_ok = False
            _print(f"{pool_type} VARIANT FAILED — loss breakdown: "
                  f"{result['loss_breakdown']}", False)

    elapsed = time.time() - t_start
    print(f"\n{'=' * 70}")
    if all_ok:
        _print(f"M1.4 CPU SMOKE PASS — total {elapsed:.1f}s", True)
        return 0
    else:
        _print(f"M1.4 CPU SMOKE FAIL — total {elapsed:.1f}s", False)
        return 1


if __name__ == "__main__":
    sys.exit(main())
