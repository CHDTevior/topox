#!/usr/bin/env python3
"""M1.5 training script — GraphMotionVAE recon training (Phase 1, no denoiser).

3-way ablation via --pool_type:
  dynamic       — DynamicGraphPool (learned + MinCut)
  deterministic — DeterministicGraphPool (rule-based hard argmin)
  none          — No skeletal pool (temporal AvgPool only)

Acceptance gates wired in (PLAN_GAP_REPORT.md §6):
  Gate #2 — z shape verified per batch
  Gate #3 — padding gate + NaN-grad guard
  Gate #5 — train_diag (smoothed loss) and per-species recon tracking
  Gate #8 — pool diagnostics: active_coarse_count, mass_min_max_ratio,
            assignment_entropy, per_topology_recon (codex M1.4 caveats wired in)

Usage:
  python scripts/train_graph_vae.py --pool_type dynamic --out runs/m1_5_dynamic
  python scripts/train_graph_vae.py --pool_type deterministic --out runs/m1_5_det
  python scripts/train_graph_vae.py --pool_type none --out runs/m1_5_nopool

Cross-project rule: not in login node — wrap in srun or _deploy_train.sh.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.unified_dataset import UnifiedMotionDataset, collate_fn
from src.models.graph_salad import (
    GraphMotionBatch,
    GraphMotionVAE,
    compute_total_loss,
)


# ---------------------------------------------------------------------------
# Pool diagnostics (Gate #8 from PLAN_GAP_REPORT.md §6)
# ---------------------------------------------------------------------------

def compute_pool_diagnostics(out: dict, batch: GraphMotionBatch) -> dict:
    """Pool health metrics: active count, mass min/max, entropy, per-species recon."""
    P = out["assignment"]  # [B, J, C]
    coarse_mask = out["coarse_mask"]  # [B, C]
    joint_mask = batch.joint_mask  # [B, J]
    B = P.shape[0]

    diag = {}
    # Active coarse count per sample (mean/min/max)
    active = coarse_mask.sum(dim=-1).float()  # [B]
    diag["active_coarse_mean"] = active.mean().item()
    diag["active_coarse_min"] = int(active.min().item())
    diag["active_coarse_max"] = int(active.max().item())

    # Mass min/max ratio (catches all-to-one collapse)
    mass = P.sum(dim=1)  # [B, C], joint-sum per coarse
    # Per-sample valid mass stats
    mass_min = 1e9
    mass_max = 0.0
    for b in range(B):
        m = mass[b][coarse_mask[b]]
        if m.numel() > 0:
            mass_min = min(mass_min, m.min().item())
            mass_max = max(mass_max, m.max().item())
    diag["mass_min"] = mass_min if mass_min < 1e9 else 0.0
    diag["mass_max"] = mass_max
    diag["mass_max_to_min_ratio"] = (
        mass_max / max(mass_min, 1e-8) if mass_min > 0 else float("inf")
    )

    # Assignment entropy: mean H(P_row) over valid joints
    # H = -sum_c P_jc * log P_jc
    eps = 1e-8
    log_P = (P + eps).log()
    entropy = -(P * log_P).sum(dim=-1)  # [B, J]
    valid_ent = entropy[joint_mask]
    diag["assignment_entropy_mean"] = valid_ent.mean().item() if valid_ent.numel() else 0.0
    diag["assignment_entropy_max"] = valid_ent.max().item() if valid_ent.numel() else 0.0

    return diag


def compute_per_species_recon(
    pos_loss_per_sample: torch.Tensor,
    skeleton_ids: list[str],
) -> dict[str, float]:
    """Group per-sample pos loss by species, average."""
    out = defaultdict(list)
    for i, sid in enumerate(skeleton_ids):
        out[sid].append(pos_loss_per_sample[i].item())
    return {sid: float(np.mean(vals)) for sid, vals in out.items()}


def _per_sample_pos_loss(pred_pos: torch.Tensor, gt_pos: torch.Tensor,
                         joint_mask: torch.Tensor, frame_mask: torch.Tensor) -> torch.Tensor:
    """Per-sample L1 pos error averaged over valid (j, t)."""
    B = pred_pos.shape[0]
    mask = joint_mask.unsqueeze(1) & frame_mask.unsqueeze(-1)  # [B, T, J]
    diff = (pred_pos - gt_pos).abs().sum(dim=-1) * mask.to(pred_pos.dtype)  # [B, T, J]
    per_sample = []
    for b in range(B):
        vc = mask[b].sum().clamp(min=1)
        per_sample.append(diff[b].sum() / vc)
    return torch.stack(per_sample)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def main() -> int:
    p = argparse.ArgumentParser()
    # Pool ablation choice
    p.add_argument("--pool_type",
                   choices=("dynamic", "deterministic", "soft_deterministic", "none"),
                   required=True)
    p.add_argument("--pool_tau", type=float, default=None,
                   help="Required when --pool_type soft_deterministic (e.g., 0.5)")
    # Data
    p.add_argument("--data_dir", default="data/cs_sparse2full_tgt")
    p.add_argument("--max_frames", type=int, default=64)
    p.add_argument("--max_joints", type=int, default=160)
    # Model
    p.add_argument("--d_model", type=int, default=256)
    p.add_argument("--n_heads", type=int, default=4)
    p.add_argument("--d_ff", type=int, default=512)
    p.add_argument("--n_graph_layers", type=int, default=4)
    p.add_argument("--n_enc_temporal_layers", type=int, default=2)
    p.add_argument("--n_cross_layers", type=int, default=3)
    p.add_argument("--n_dec_temporal_layers", type=int, default=2)
    p.add_argument("--n_treeik_layers", type=int, default=3)
    p.add_argument("--max_coarse", type=int, default=64)
    p.add_argument("--local_radius", type=int, default=8)
    p.add_argument("--temporal_stride", type=int, default=4)
    p.add_argument("--temporal_kernel", type=int, default=9)
    p.add_argument("--dropout", type=float, default=0.1)
    # Train
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--save_every", type=int, default=10)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--init_ckpt", type=str, default=None,
                   help="Optional baseline ckpt to warm-start encoder+slot_norm+decoder")
    # Loss weights
    p.add_argument("--w_pos", type=float, default=1.0)
    p.add_argument("--w_vel", type=float, default=1.0)
    p.add_argument("--w_vel_normalized", type=float, default=0.0,
                   help="M1.5R decision #5: per-species vel norm weight (default 0=off for backward compat)")
    p.add_argument("--w_vel_consistency", type=float, default=0.5)
    p.add_argument("--w_speed_mag", type=float, default=0.0,
                   help="M1.5R B prong: anti-frozen speed magnitude weight (default 0=off)")
    p.add_argument("--w_kl", type=float, default=1e-3)
    p.add_argument("--w_bone", type=float, default=1.0)
    p.add_argument("--w_pool_aux", type=float, default=0.5)
    # M1.5R decision #4: enable name embedding (cross-species shared semantics)
    p.add_argument("--use_name_embed", action="store_true",
                   help="M1.5R decision #4: encoder.use_name_embed=True for cross-species transfer")
    # I/O
    p.add_argument("--out", required=True, help="Output dir for ckpts + logs")
    p.add_argument("--device", default="cuda")
    p.add_argument("--overwrite", action="store_true",
                   help="Overwrite non-empty --out dir (otherwise refuse)")
    p.add_argument("--smoke", action="store_true",
                   help="Run 5 iters smoke test (verifies wiring)")
    args = p.parse_args()

    # Seed (codex M1.5 Low: include CUDA seed for full determinism)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Device — fail loud if CUDA requested but unavailable (codex M1.5 High)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "[DEVICE FAIL] --device cuda requested but torch.cuda.is_available()=False. "
            "Use --device cpu explicitly or fix the CUDA env.")
    dev = torch.device(args.device)

    # Output dir — refuse non-empty unless --overwrite (codex M1.5 High)
    out_dir = Path(args.out)
    if out_dir.exists() and any(out_dir.iterdir()) and not args.overwrite:
        raise RuntimeError(
            f"[OUT FAIL] Output dir {out_dir} is non-empty. "
            "Use --overwrite or pick a fresh path.")
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "train.log"
    metrics_path = out_dir / "metrics.jsonl"
    diag_path = out_dir / "diagnostics.jsonl"

    def log(msg: str) -> None:
        print(msg, flush=True)
        with open(log_path, "a") as f:
            f.write(msg + "\n")

    # Capture git SHA for reproducibility (codex M1.5 Low)
    import subprocess
    try:
        git_sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parent.parent
        ).decode().strip()
    except Exception:
        git_sha = "unknown"

    log(f"=== M1.5 graph_salad VAE training — pool_type={args.pool_type} ===")
    log(f"git_sha: {git_sha}")
    log(f"device: {dev}")
    log(f"args: {vars(args)}")

    # Data
    log(f"Loading dataset from {args.data_dir} ...")
    ds_train = UnifiedMotionDataset(
        data_dirs=[args.data_dir], split="train",
        max_joints=args.max_joints, max_frames=args.max_frames,
        normalize=False,
    )
    ds_val = UnifiedMotionDataset(
        data_dirs=[args.data_dir], split="val",
        max_joints=args.max_joints, max_frames=args.max_frames,
        normalize=False,
    )
    log(f"train={len(ds_train)} val={len(ds_val)}")
    # Empty/too-small split guard (codex M1.5 Medium)
    if len(ds_train) < args.batch_size:
        raise RuntimeError(
            f"[DATA FAIL] train split has {len(ds_train)} samples but batch_size={args.batch_size}. "
            "Cannot form a single batch.")
    if len(ds_val) == 0:
        raise RuntimeError("[DATA FAIL] val split is empty.")

    # DataLoader tuning: workers=8 + pin_memory + persistent (codex-side-tuning for util>80%)
    dl_train = DataLoader(
        ds_train, batch_size=args.batch_size, shuffle=True,
        collate_fn=collate_fn, num_workers=8, drop_last=True,
        pin_memory=True, persistent_workers=True, prefetch_factor=4,
    )
    dl_val = DataLoader(
        ds_val, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=4, drop_last=False,
        pin_memory=True, persistent_workers=True, prefetch_factor=4,
    )

    # Model
    vae = GraphMotionVAE(
        pool_type=args.pool_type,
        d_model=args.d_model, n_heads=args.n_heads, d_ff=args.d_ff,
        n_graph_layers=args.n_graph_layers,
        n_enc_temporal_layers=args.n_enc_temporal_layers,
        n_cross_layers=args.n_cross_layers,
        n_dec_temporal_layers=args.n_dec_temporal_layers,
        n_treeik_layers=args.n_treeik_layers,
        max_coarse=args.max_coarse, local_radius=args.local_radius,
        temporal_stride=args.temporal_stride,
        temporal_kernel=args.temporal_kernel,
        dropout=args.dropout,
        pool_tau=args.pool_tau,
    ).to(dev)
    # M1.5R decision #4: enable name embedding in encoder
    if args.use_name_embed:
        vae.encoder.use_name_embed = True
        log(f"  [M1.5R #4] use_name_embed=True (cross-species shared semantics)")
    n_params = sum(p.numel() for p in vae.parameters())
    log(f"VAE params: {n_params:,}")

    # Warm-start from baseline ckpt — fail loud on missing/unexpected (codex M1.5 Critical)
    if args.init_ckpt is not None:
        if not Path(args.init_ckpt).exists():
            raise RuntimeError(f"[INIT_CKPT FAIL] {args.init_ckpt} does not exist.")
        log(f"Loading baseline ckpt: {args.init_ckpt}")
        ckpt = torch.load(args.init_ckpt, map_location=dev, weights_only=True)
        sd = ckpt.get("model_state_dict", ckpt)
        sd_filtered = {k: v for k, v in sd.items() if not k.startswith("slot_assignment.")}
        load_result = vae.load_state_dict(sd_filtered, strict=False)
        # Strict: no unexpected baseline keys allowed (after slot_assignment filter)
        if len(load_result.unexpected_keys) > 0:
            raise RuntimeError(
                f"[INIT_CKPT FAIL] unexpected baseline keys: "
                f"{load_result.unexpected_keys[:5]} (total {len(load_result.unexpected_keys)})")
        # Strict: missing keys must be confined to new-module prefixes
        allowed_missing_prefixes = ("pool.", "unpool.", "dist", "treeik_head.")
        bad_missing = [k for k in load_result.missing_keys
                       if not any(k.startswith(p) for p in allowed_missing_prefixes)]
        if bad_missing:
            raise RuntimeError(
                f"[INIT_CKPT FAIL] {len(bad_missing)} baseline keys NOT loaded "
                f"(not in allowed-missing prefixes): {bad_missing[:5]}")
        log(f"  loaded {len(sd_filtered)} keys; allowed-missing={len(load_result.missing_keys)} unexpected=0")

    # Pre-build loss weights (codex M1.5 High: use same in train and val)
    loss_weights = {
        "pos": args.w_pos, "vel": args.w_vel,
        "vel_normalized": args.w_vel_normalized,  # M1.5R #5
        "vel_consistency": args.w_vel_consistency,
        "speed_mag": args.w_speed_mag,            # M1.5R B
        "kl": args.w_kl, "bone": args.w_bone,
        "pool_aux": args.w_pool_aux,
    }
    # Gate #2 expected C dim depends on pool variant (codex M1.5 Critical)
    expected_C = args.max_joints if args.pool_type == "none" else args.max_coarse
    log(f"loss_weights: {loss_weights}")
    log(f"Gate #2 expected_C: {expected_C} ({'max_joints' if args.pool_type == 'none' else 'max_coarse'})")

    # Optimizer
    opt = torch.optim.AdamW(vae.parameters(), lr=args.lr)

    # Training loop
    n_iter = 0
    train_diag_smoothed = None  # EMA of train loss for gate #5
    best_val_loss = float("inf")
    # Codex M1.5 R2 Medium: recon-only val for ablation ranking
    # (total val loss includes pool aux which makes 3-way pool variants incomparable)
    best_val_recon = float("inf")
    smoke_iter_cap = 5 if args.smoke else None

    for epoch in range(args.epochs):
        vae.train()
        t0 = time.time()
        epoch_losses = defaultdict(list)
        for it, raw in enumerate(dl_train):
            if smoke_iter_cap is not None and it >= smoke_iter_cap:
                break

            batch = GraphMotionBatch.from_collate_dict({
                k: v.to(dev) if torch.is_tensor(v) else v for k, v in raw.items()
            })

            out = vae(batch)
            # Gate #2: z-shape contract verify + fp32 dtype assert (1st iter of every epoch)
            if it == 0:
                B, T_lat, C, D = out["mu"].shape
                assert C == expected_C, (
                    f"[GATE2 FAIL] z C={C} != expected_C={expected_C} "
                    f"(pool_type={args.pool_type})")
                assert D == args.d_model, (
                    f"[GATE2 FAIL] z D={D} != d_model={args.d_model}")
                assert out["mu"].dtype == torch.float32, (
                    f"[DTYPE FAIL] mu={out['mu'].dtype} not fp32")
                assert out["pred_pos"].dtype == torch.float32, (
                    f"[DTYPE FAIL] pred_pos={out['pred_pos'].dtype} not fp32")
                log(f"  [gate2 ok] z=[{B},{T_lat},{C},{D}] dtype=fp32")
            # Build gt
            gt_pos = batch.motion_features[..., :3]
            gt_vel = batch.motion_features[..., 3:6]
            # Build rest_bone_lengths from per-sample list
            rest_bones = torch.zeros(batch.batch_size, batch.max_joints, device=dev)
            for b in range(batch.batch_size):
                bls = batch.bone_lengths_rest[b]
                rest_bones[b, :len(bls)] = torch.tensor(bls, device=dev, dtype=torch.float32)
            # Codex M1.5 R3 P0: use frame_mask_recovered (⊆ batch.frame_mask) as effective
            # mask so stride-tail frames the decoder zeros are not penalized. Affects
            # 231/598 train and 46/112 val clips with actual_frames % stride != 0.
            effective_frame_mask = out["frame_mask_recovered"]
            if it == 0 and epoch == 0:
                dropped = (batch.frame_mask & ~effective_frame_mask).sum().item()
                log(f"  [stride-tail] frames dropped by stride={args.temporal_stride}: "
                    f"{dropped}/{batch.frame_mask.sum().item()}")
            losses = compute_total_loss(
                pred_pos=out["pred_pos"], gt_pos=gt_pos,
                pred_vel=out["pred_vel"], gt_vel=gt_vel,
                mu=out["mu"], logvar=out["logvar"],
                pool_aux_outputs=out["pool_aux_outputs"],
                joint_mask=batch.joint_mask,
                frame_mask=effective_frame_mask,
                coarse_mask=out["coarse_mask"],
                frame_mask_lat=out["frame_mask_lat"],
                rest_bone_lengths=rest_bones,
                parent_indices=batch.parent_indices,
                fps=batch.fps,
                weights=loss_weights,
            )

            # Gate #3: pre-backward loss finite check (codex M1.5 High)
            for k, v in losses.items():
                if not torch.isfinite(v):
                    log(f"[GATE3 FAIL] loss[{k}]={v.item()} non-finite at iter {n_iter}")
                    return 1
            opt.zero_grad()
            losses["total"].backward()
            # Gate #3: NaN-grad guard (per-param finite + clip with error_if_nonfinite=True)
            grad_max = 0.0
            for p_param in vae.parameters():
                if p_param.grad is not None:
                    g = p_param.grad
                    if not torch.isfinite(g).all():
                        log(f"[GATE3 FAIL] NaN/Inf grad at iter {n_iter}; halting epoch")
                        return 1
                    grad_max = max(grad_max, g.abs().max().item())
            torch.nn.utils.clip_grad_norm_(
                vae.parameters(), max_norm=10.0, error_if_nonfinite=True
            )
            opt.step()

            # Track losses
            for k, v in losses.items():
                epoch_losses[k].append(v.item())

            # EMA train diag (gate #5)
            cur_loss = losses["total"].item()
            train_diag_smoothed = (
                cur_loss if train_diag_smoothed is None
                else 0.99 * train_diag_smoothed + 0.01 * cur_loss
            )

            n_iter += 1

            # Periodic diagnostic log (every 50 iter or 1st of epoch) — Gate #8
            if it == 0 or n_iter % 50 == 0:
                diag = compute_pool_diagnostics(out, batch)
                # Assignment row-sum sanity (defense-in-depth)
                P_rowsum = out["assignment"].sum(dim=-1)  # [B, J]
                masked = batch.joint_mask
                valid_rowsum = P_rowsum[masked]
                rowsum_min = valid_rowsum.min().item() if valid_rowsum.numel() else 1.0
                rowsum_max = valid_rowsum.max().item() if valid_rowsum.numel() else 1.0
                diag["assignment_rowsum_min"] = rowsum_min
                diag["assignment_rowsum_max"] = rowsum_max
                # Pool aux finite check (fail-loud R12)
                for aux_dict in out["pool_aux_outputs"]:
                    for k, v in aux_dict.items():
                        if torch.is_tensor(v) and not torch.isfinite(v).all():
                            log(f"[GATE3 FAIL] pool aux {k}={v.item()} non-finite")
                            return 1
                log(f"[ep{epoch} it{it} n_iter={n_iter}] "
                    f"loss={cur_loss:.4f} diag={train_diag_smoothed:.4f} "
                    f"grad_max={grad_max:.3f} "
                    f"active_C={diag['active_coarse_mean']:.1f}({diag['active_coarse_min']}-{diag['active_coarse_max']}) "
                    f"mass_min={diag['mass_min']:.2f} ent={diag['assignment_entropy_mean']:.3f} "
                    f"rowsum=[{rowsum_min:.3f},{rowsum_max:.3f}]")
                # Persist diagnostics to JSON (codex M1.5 Medium)
                with open(diag_path, "a") as f:
                    f.write(json.dumps({
                        "epoch": epoch, "iter": it, "n_iter": n_iter,
                        "loss": cur_loss, "grad_max": grad_max,
                        **diag,
                    }) + "\n")

        epoch_dt = time.time() - t0
        train_loss_mean = float(np.mean(epoch_losses["total"]))
        log(f"=== epoch {epoch} done in {epoch_dt:.1f}s | "
            f"train_loss={train_loss_mean:.4f} train_diag={train_diag_smoothed:.4f} ===")

        # Save metrics
        rec = {"epoch": epoch, "train_loss": train_loss_mean,
               "train_diag": train_diag_smoothed,
               "epoch_dt_s": epoch_dt, "n_iter": n_iter,
               "loss_breakdown": {k: float(np.mean(v)) for k, v in epoch_losses.items()}}
        with open(metrics_path, "a") as f:
            f.write(json.dumps(rec) + "\n")

        # Val + per-species recon
        if (epoch + 1) % args.save_every == 0 or epoch == args.epochs - 1 or args.smoke:
            vae.eval()
            t_v = time.time()
            val_losses = defaultdict(list)
            # Per-species per-sample list (codex M1.5 Medium: no batch-mean double-averaging)
            per_species_pos = defaultdict(list)
            # Frozen-pred audit metrics (codex 2026-05-21 root cause finding)
            speed_ratios = []
            pred_speeds = []
            gt_speeds = []
            with torch.no_grad():
                for raw in dl_val:
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
                    effective_frame_mask_val = out["frame_mask_recovered"]
                    losses = compute_total_loss(
                        pred_pos=out["pred_pos"], gt_pos=gt_pos,
                        pred_vel=out["pred_vel"], gt_vel=gt_vel,
                        mu=out["mu"], logvar=out["logvar"],
                        pool_aux_outputs=out["pool_aux_outputs"],
                        joint_mask=batch.joint_mask,
                        frame_mask=effective_frame_mask_val,
                        coarse_mask=out["coarse_mask"],
                        frame_mask_lat=out["frame_mask_lat"],
                        rest_bone_lengths=rest_bones,
                        parent_indices=batch.parent_indices,
                        fps=batch.fps,
                        weights=loss_weights,  # codex M1.5 High: same weights as train
                    )
                    for k, v in losses.items():
                        val_losses[k].append(v.item())
                    # Per-species per-sample pos loss (codex M1.5 Medium)
                    psl = _per_sample_pos_loss(
                        out["pred_pos"], gt_pos, batch.joint_mask, effective_frame_mask_val
                    )
                    for i, sid in enumerate(batch.skeleton_id):
                        per_species_pos[sid].append(psl[i].item())
                    # Codex M1.5 frozen-pred audit (2026-05-21): speed-ratio metric
                    # to expose static-pose shortcut early. Per-frame mean displacement
                    # ratio (pred/gt). Frozen if ratio < 0.1.
                    pp = out["pred_pos"]  # [B, T, J, 3]
                    gp = gt_pos
                    if pp.shape[1] > 1:
                        pp_d = (pp[:, 1:] - pp[:, :-1]).norm(dim=-1)  # [B,T-1,J]
                        gp_d = (gp[:, 1:] - gp[:, :-1]).norm(dim=-1)
                        mask = batch.joint_mask.unsqueeze(1) & effective_frame_mask_val[:, 1:].unsqueeze(-1)
                        mask_f = mask.to(pp.dtype)
                        denom = mask_f.sum().clamp(min=1.0)
                        pred_meanspeed = (pp_d * mask_f).sum() / denom
                        gt_meanspeed = (gp_d * mask_f).sum() / denom
                        ratio = (pred_meanspeed / gt_meanspeed.clamp(min=1e-8)).item()
                        speed_ratios.append(ratio)
                        pred_speeds.append(pred_meanspeed.item())
                        gt_speeds.append(gt_meanspeed.item())

            val_loss_mean = float(np.mean(val_losses["total"]))
            # Recon-only val (3-way pool-fair ablation metric — codex M1.5 R2 Medium)
            # = w_pos*pos + w_vel*vel + w_vel_consistency*vel_consistency + w_bone*bone
            # Excludes pool_aux + KL (codex M1.5 R3 P1: weight components like train)
            # P3 codex review fix #1: include vel_normalized + speed_mag in val_recon
            # selection metric (otherwise best_recon_model.pt would prefer old-objective frozen
            # ckpts even when training optimizes the new terms).
            recon_keys = ("pos", "vel", "vel_normalized", "vel_consistency",
                          "speed_mag", "bone")
            val_recon_components_raw = {k: float(np.mean(val_losses[k]))
                                       for k in recon_keys
                                       if k in val_losses and loss_weights.get(k, 0.0) > 0.0}
            val_recon = sum(
                loss_weights[k] * v for k, v in val_recon_components_raw.items()
            )
            val_recon_components = {
                k: {"raw": v, "weighted": loss_weights[k] * v}
                for k, v in val_recon_components_raw.items()
            }
            # Frozen-pred audit (codex 2026-05-21): expose static-shortcut early
            mean_speed_ratio = float(np.mean(speed_ratios)) if speed_ratios else float("nan")
            mean_pred_speed = float(np.mean(pred_speeds)) if pred_speeds else float("nan")
            mean_gt_speed = float(np.mean(gt_speeds)) if gt_speeds else float("nan")
            frozen_flag = "🥶FROZEN" if mean_speed_ratio < 0.1 else ("⚠LOW" if mean_speed_ratio < 0.5 else "✓OK")
            val_dt = time.time() - t_v
            log(f"[val ep{epoch}] dt={val_dt:.1f}s total={val_loss_mean:.4f} "
                f"recon_only={val_recon:.4f} "
                f"speed_ratio={mean_speed_ratio:.4f} {frozen_flag} "
                f"(pred={mean_pred_speed:.4f} gt={mean_gt_speed:.4f})")
            for sid, vals in sorted(per_species_pos.items()):
                log(f"  per-species pos: {sid} n={len(vals)} "
                    f"mean={float(np.mean(vals)):.4f} std={float(np.std(vals)):.4f}")

            # Always save last_model.pt — weights_only-safe (codex M1.5 R2 High: no numpy)
            last_path = out_dir / "last_model.pt"
            torch.save({
                "epoch": epoch, "val_loss": val_loss_mean,
                "val_recon": val_recon,
                "model_state_dict": vae.state_dict(),
                "optimizer_state_dict": opt.state_dict(),
                "args": vars(args),
                "git_sha": git_sha,
            }, last_path)

            # Best-by-total checkpoint (kept for backward-compat / KL+pool monitoring)
            if val_loss_mean < best_val_loss:
                best_val_loss = val_loss_mean
                save_path = out_dir / "best_model.pt"
                torch.save({
                    "epoch": epoch, "val_loss": val_loss_mean,
                    "val_recon": val_recon,
                    "model_state_dict": vae.state_dict(),
                    "args": vars(args),
                    "git_sha": git_sha,
                }, save_path)
                log(f"  saved best (total) ckpt → {save_path}")

            # Best-by-recon checkpoint (USE THIS FOR 3-WAY ABLATION RANKING)
            if val_recon < best_val_recon:
                best_val_recon = val_recon
                save_path = out_dir / "best_recon_model.pt"
                torch.save({
                    "epoch": epoch, "val_loss": val_loss_mean,
                    "val_recon": val_recon,
                    "model_state_dict": vae.state_dict(),
                    "args": vars(args),
                    "git_sha": git_sha,
                }, save_path)
                log(f"  saved best (recon-only) ckpt → {save_path} "
                    f"[USE FOR ABLATION RANKING]")

            # Save metrics — per-sample stats (codex M1.5 Medium)
            macro_pos = float(np.mean([
                float(np.mean(vals)) for vals in per_species_pos.values()
            ]))
            with open(metrics_path, "a") as f:
                f.write(json.dumps({
                    "epoch": epoch, "val_loss": val_loss_mean,
                    "val_recon": val_recon,
                    "val_recon_components": val_recon_components,
                    "macro_per_species_pos": macro_pos,
                    # Frozen-pred audit (codex 2026-05-21):
                    "speed_ratio_mean": mean_speed_ratio,
                    "pred_speed_mean": mean_pred_speed,
                    "gt_speed_mean": mean_gt_speed,
                    "frozen_flag": frozen_flag,
                    "per_species_pos": {
                        sid: {
                            "n": len(vals),
                            "mean": float(np.mean(vals)),
                            "std": float(np.std(vals)),
                        }
                        for sid, vals in per_species_pos.items()
                    },
                }) + "\n")

        if args.smoke and epoch >= 0:
            log("=== SMOKE MODE: 1 epoch done, exit ===")
            break

    log("=== training complete ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
