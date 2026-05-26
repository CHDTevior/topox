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
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.unified_dataset import UnifiedMotionDataset, collate_fn
from src.data.anytop_dataset import AnyTopDataset, collate_fn as anytop_collate_fn
from src.models.graph_salad import (
    GraphMotionBatch,
    GraphMotionVAE,
    compute_total_loss,
    compute_total_loss_13ch,
)


def run_loss(out, batch, feat_mode, loss_weights, effective_frame_mask, dev):
    """Dispatch to the feat_mode-appropriate loss function.

    fk6      -> compute_total_loss (pred_pos/pred_vel vs motion_features).
    anytop13 -> compute_total_loss_13ch (pred_motion vs anytop_x, + contact BCE).
    """
    if feat_mode == "anytop13":
        if batch.anytop_x is None or batch.foot_contact_per_joint is None:
            raise ValueError(
                "run_loss(feat_mode=anytop13) requires batch.anytop_x and "
                "batch.foot_contact_per_joint; use --dataset anytop_truebones"
            )
        gt_motion = batch.anytop_x.permute(0, 3, 1, 2).contiguous()  # [B,T,J,13]
        return compute_total_loss_13ch(
            pred_motion=out["pred_motion"],
            gt_motion=gt_motion,
            foot_contact_per_joint=batch.foot_contact_per_joint,
            mu=out["mu"], logvar=out["logvar"],
            pool_aux_outputs=out["pool_aux_outputs"],
            joint_mask=batch.joint_mask,
            frame_mask=effective_frame_mask,
            coarse_mask=out["coarse_mask"],
            frame_mask_lat=out["frame_mask_lat"],
            weights=loss_weights,
        )
    # fk6
    gt_pos = batch.motion_features[..., :3]
    gt_vel = batch.motion_features[..., 3:6]
    rest_bones = torch.zeros(batch.batch_size, batch.max_joints, device=dev)
    for b in range(batch.batch_size):
        bls = batch.bone_lengths_rest[b]
        rest_bones[b, :len(bls)] = torch.tensor(bls, device=dev, dtype=torch.float32)
    return compute_total_loss(
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

def _ddp_setup() -> "tuple[bool, int, int, int, bool]":
    """Detect a torchrun DDP launch from the environment.

    Returns (is_ddp, rank, local_rank, world_size, is_main). A plain
    `python train_graph_vae.py` leaves WORLD_SIZE unset -> is_ddp=False and the
    single-GPU path runs unchanged. Under `torchrun --nproc_per_node=N` (N>1),
    WORLD_SIZE=N -> init the NCCL process group and pin this rank's CUDA device.
    """
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size <= 1:
        return False, 0, 0, 1, True
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    # device_id binds the process group to this rank's GPU — required by modern
    # torch to avoid the barrier() "using the device under current context" warning.
    dist.init_process_group(
        backend="nccl", device_id=torch.device("cuda", local_rank))
    return True, rank, local_rank, world_size, rank == 0


def main() -> int:
    p = argparse.ArgumentParser()
    # Pool ablation choice
    p.add_argument("--pool_type",
                   choices=("dynamic", "deterministic", "soft_deterministic",
                            "edge_segment", "none"),
                   required=True)
    p.add_argument("--pool_tau", type=float, default=None,
                   help="Required when --pool_type soft_deterministic (e.g., 0.5)")
    # Data
    p.add_argument("--dataset",
                   choices=("unified", "anytop_truebones"),
                   default="unified",
                   help="Dataset source. 'unified' = UnifiedMotionDataset (default, "
                        "M1.5/M1.5R 6ch path). 'anytop_truebones' = AnyTopDataset "
                        "reading AnyTop's pre-processed truebones (M1.7).")
    p.add_argument("--data_dir", default="data/cs_sparse2full_tgt",
                   help="For --dataset unified: dataset root. For --dataset "
                        "anytop_truebones: ignored (uses fixed AnyTop path) "
                        "unless --anytop_root passed.")
    p.add_argument("--anytop_root", type=str, default=None,
                   help="Override AnyTop processed-data root (default: AnyTop's "
                        ".../truebones/zoo/truebones_processed)")
    p.add_argument("--full_data_val_species", type=str, default=None,
                   help="(anytop_truebones only) Full-data training mode with "
                        "species-filtered val. When set: train uses split='all' "
                        "(all 1070 motions, no holdout); val uses split='all' "
                        "then filters to the listed comma-separated species "
                        "(e.g. 'Dragon,Monkey,Centipede,Horse' = 4 largest-J "
                        "skeletons). Train and val OVERLAP on those species — "
                        "intentional, val measures recon quality on the hardest "
                        "skeletons. Skips the per-species 80/20 split.")
    # AnyTop remove-joints augmentation (train split only; --dataset anytop_truebones)
    p.add_argument("--augment", action="store_true",
                   help="Enable AnyTop remove-joints augmentation on the train "
                        "split (drops non-foot end-effector joints)")
    p.add_argument("--augment_prob", type=float, default=0.3,
                   help="Per-sample probability of applying remove-joints aug")
    p.add_argument("--removal_rate", type=float, default=0.5,
                   help="Fraction of eligible end-effectors to drop per aug")
    # Optional text conditioning (--dataset anytop_truebones)
    p.add_argument("--use_text", action="store_true",
                   help="Enable text conditioning: inject precomputed T5 caption "
                        "embeddings into the decoder (optional — missing captions "
                        "fall back to a zero, no-op embedding)")
    p.add_argument("--caption_emb_cache", type=str, default=None,
                   help="Path to the .npz caption-embedding cache produced by "
                        "scripts/precompute_t5_captions.py (required for --use_text "
                        "to actually condition; without it captions are all no-op)")
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
    p.add_argument("--periodic_save_every", type=int, default=0,
                   help="Also save a PRESERVED ckpt named ep{N}_model.pt every "
                         "N epochs (in addition to last_model.pt overwrite). "
                         "0 disables. Mirrors train_denoiser.py (2026-05-25); "
                         "useful for long multi-cont VAE training where you "
                         "want sweeping ckpt history rather than only best+last.")
    p.add_argument("--val_frac", type=float, default=0.2,
                   help="AnyTopDataset object-stratified split val fraction. "
                         "Default 0.2 = 80/20 split (legacy). Pass 0.05 = 19:1 "
                         "for large datasets (PlanetZoo L1, 2026-05-26).")
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--init_ckpt", type=str, default=None,
                   help="Optional baseline ckpt to warm-start encoder+slot_norm+decoder")
    # feat_mode: fk6 (legacy 6ch+FK) vs anytop13 (AnyTop native 13ch end-to-end)
    p.add_argument("--feat_mode", choices=("fk6", "anytop13"), default="fk6",
                   help="fk6: 6ch local_pos+vel -> FK decoder. anytop13: AnyTop "
                        "native 13ch end-to-end (requires --dataset anytop_truebones)")
    # attn_mode: scalar (legacy) vs graphormer (AnyTop per-edge-type embedding bias)
    p.add_argument("--attn_mode", choices=("scalar", "graphormer"), default="scalar",
                   help="scalar: Linear(1,n_heads) geo/adj bias. graphormer: "
                        "AnyTop edge-type embedding bias (requires "
                        "--dataset anytop_truebones for graph_dist/joint_relations)")
    p.add_argument("--decoder_mode",
                   choices=("unpool_identity", "coarse_xattn", "graph_temporal"),
                   default=None,
                   help="Decoder structure (default if unset: anytop13 -> "
                        "coarse_xattn, fk6 -> unpool_identity). "
                        "unpool_identity: unpool to fine joints then decode with "
                        "identity assignment (cross-attn degenerate). coarse_xattn: "
                        "decode coarse slots directly with the real pool assignment "
                        "P (each joint attends its coarse anchors). graph_temporal: "
                        "coarse_xattn + AnyTop-style spatial+temporal refine layers "
                        "(anytop13 only).")
    p.add_argument("--n_graph_temporal_layers", type=int, default=4,
                   help="decoder_mode=graph_temporal: number of spatial+temporal "
                        "refine layers")
    # Loss weights
    p.add_argument("--w_pos", type=float, default=1.0)
    p.add_argument("--w_vel", type=float, default=1.0)
    p.add_argument("--w_rot", type=float, default=1.0,
                   help="anytop13: 6D-rotation channel (3:9) L1 weight")
    p.add_argument("--w_contact", type=float, default=0.1,
                   help="anytop13: foot-contact channel (12) BCE weight")
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

    # DDP: detect a torchrun launch (WORLD_SIZE>1). Single-process otherwise —
    # then is_ddp=False, is_main=True and every DDP branch below is a no-op.
    is_ddp, rank, local_rank, world_size, is_main = _ddp_setup()

    # feat_mode <-> dataset cross-check: anytop13 needs the AnyTop dataset
    # (only it emits anytop_x / foot_contact_per_joint).
    if args.feat_mode == "anytop13" and args.dataset != "anytop_truebones":
        raise RuntimeError(
            "[ARGS FAIL] --feat_mode anytop13 requires --dataset anytop_truebones "
            f"(got --dataset {args.dataset})")
    # decoder_mode default resolution: an unset --decoder_mode resolves by
    # feat_mode — anytop13 -> coarse_xattn (A/B-validated: coarse slots + real
    # pool assignment P), fk6 -> unpool_identity (its historical M1.5R behavior,
    # so fk6 stays reproducible). An explicit --decoder_mode always wins.
    if args.decoder_mode is None:
        args.decoder_mode = (
            "coarse_xattn" if args.feat_mode == "anytop13" else "unpool_identity")
        if is_main:
            print(f"[ARGS] --decoder_mode unset -> resolved to {args.decoder_mode!r} "
                  f"(feat_mode={args.feat_mode})")
    # graph_temporal needs the AnyTop graph tensors (graph_dist/joint_relations)
    # + the 13ch head — anytop13 + the AnyTop dataset only.
    if args.decoder_mode == "graph_temporal" and (
            args.feat_mode != "anytop13" or args.dataset != "anytop_truebones"):
        raise RuntimeError(
            "[ARGS FAIL] --decoder_mode graph_temporal requires --feat_mode "
            f"anytop13 and --dataset anytop_truebones (got --feat_mode "
            f"{args.feat_mode} --dataset {args.dataset})")
    # attn_mode <-> dataset cross-check: graphormer needs anytop_graph_dist +
    # anytop_joint_relations, emitted only by the AnyTop dataset.
    if args.attn_mode == "graphormer" and args.dataset != "anytop_truebones":
        raise RuntimeError(
            "[ARGS FAIL] --attn_mode graphormer requires --dataset anytop_truebones "
            f"(got --dataset {args.dataset})")
    # use_text needs caption_emb from the AnyTop dataset.
    if args.use_text and args.dataset != "anytop_truebones":
        raise RuntimeError(
            "[ARGS FAIL] --use_text requires --dataset anytop_truebones "
            f"(got --dataset {args.dataset})")
    # --use_text without a caption cache trains a text_proj that every sample
    # gates to zero (has_text=False) — text conditioning is then a silent no-op.
    if args.use_text and args.caption_emb_cache is None:
        if is_main:
            print("[ARGS WARN] --use_text set but --caption_emb_cache is None: "
                  "every sample will have has_text=False, so text conditioning is a "
                  "NO-OP. Pass --caption_emb_cache <npz> (scripts/precompute_t5_captions.py).")

    # Seed (codex M1.5 Low: include CUDA seed for full determinism)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Device — fail loud if CUDA requested but unavailable (codex M1.5 High).
    # Under DDP (torchrun) each rank pins its own GPU regardless of --device.
    if is_ddp:
        dev = torch.device("cuda", local_rank)
    else:
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
        # DDP: only rank 0 prints / writes the log — avoids N-way garbled output.
        if not is_main:
            return
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
    if args.dataset == "anytop_truebones":
        log(f"Loading AnyTop truebones (root={args.anytop_root or 'default'}) ...")
        atk = dict(num_frames=args.max_frames, max_joints=args.max_joints,
                   val_frac=args.val_frac)
        if args.anytop_root is not None:
            atk["data_root"] = args.anytop_root
        # Augmentation: train split only — ds_val never gets the aug args.
        # caption_emb_cache goes to BOTH (text condition is eval-relevant too).
        if args.caption_emb_cache is not None:
            atk["caption_emb_cache"] = args.caption_emb_cache
        if args.full_data_val_species is not None:
            # Full-data training mode: train=all 1070 (no holdout), val=species-filtered
            val_species_set = set(
                s.strip() for s in args.full_data_val_species.split(",") if s.strip()
            )
            if not val_species_set:
                raise SystemExit(
                    f"--full_data_val_species parsed to empty set from "
                    f"{args.full_data_val_species!r}"
                )
            # Codex P2 fail-loud (2026-05-23): AnyTopDataset internally forces
            # augment=False unless split=='train'. In full-data mode train uses
            # split='all' → --augment would silently no-op. Fail loud instead.
            if args.augment:
                raise SystemExit(
                    "[ARGS FAIL] --augment + --full_data_val_species combo is "
                    "currently a silent no-op (AnyTopDataset gates augment to "
                    "split=='train' only). Either drop --augment, or extend "
                    "AnyTopDataset to support augment in split='all' mode."
                )
            # Codex P1 fix (2026-05-23): split='all' default disables random
            # temporal crop on T>num_frames clips (731/1070 affected) → train
            # would see same first-64 frames each epoch. Pass random_crop=True
            # for train, False for val to preserve baseline data augmentation.
            ds_train = AnyTopDataset(
                split="all", augment=args.augment,
                augment_prob=args.augment_prob, removal_rate=args.removal_rate,
                random_crop=True,
                **atk,
            )
            ds_val = AnyTopDataset(split="all", random_crop=False, **atk)
            # Filter val samples in-place to the requested species (train+val
            # overlap on those species is INTENTIONAL — val measures recon
            # quality on the hardest skeletons, not OOD generalization).
            ds_val.samples = [s for s in ds_val.samples
                              if s["object_type"] in val_species_set]
            if len(ds_val.samples) == 0:
                raise SystemExit(
                    f"[DATA] val species filter {sorted(val_species_set)!r} matched "
                    f"0 motions. Check spelling against AnyTop cond.npy species keys."
                )
            present = sorted({s["object_type"] for s in ds_val.samples})
            missing = sorted(val_species_set - set(present))
            if missing:
                log(f"  [WARN] val species not in dataset (skipped): {missing}")
            log(f"  [FULL-DATA MODE] train=all 1070 ({len(ds_train)} samples), "
                f"val=species-filtered to {sorted(val_species_set)!r} "
                f"({len(ds_val)} samples). Train/val OVERLAP on these species "
                f"(intentional — val = recon quality on hard skeletons).")
        else:
            ds_train = AnyTopDataset(
                split="train", augment=args.augment,
                augment_prob=args.augment_prob, removal_rate=args.removal_rate,
                **atk,
            )
            ds_val = AnyTopDataset(split="val", **atk)
        active_collate_fn = anytop_collate_fn
    else:
        log(f"Loading UnifiedMotionDataset from {args.data_dir} ...")
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
        active_collate_fn = collate_fn
    log(f"train={len(ds_train)} val={len(ds_val)}")
    # Empty/too-small split guard (codex M1.5 Medium)
    if len(ds_train) < args.batch_size:
        raise RuntimeError(
            f"[DATA FAIL] train split has {len(ds_train)} samples but batch_size={args.batch_size}. "
            "Cannot form a single batch.")
    if len(ds_val) == 0:
        raise RuntimeError("[DATA FAIL] val split is empty.")

    # DataLoader tuning: workers=8 + pin_memory + persistent (codex-side-tuning for util>80%).
    # Under DDP the train loader is sharded by a DistributedSampler (one shard per
    # rank; drop_last so every rank gets an equal batch count — no padding/desync).
    # dl_val stays a plain full-set loader, iterated only by rank 0.
    train_sampler = (
        DistributedSampler(ds_train, shuffle=True, drop_last=True)
        if is_ddp else None
    )
    dl_train = DataLoader(
        ds_train, batch_size=args.batch_size,
        shuffle=(train_sampler is None), sampler=train_sampler,
        collate_fn=active_collate_fn, num_workers=8, drop_last=True,
        pin_memory=True, persistent_workers=True, prefetch_factor=4,
    )
    dl_val = DataLoader(
        ds_val, batch_size=args.batch_size, shuffle=False,
        collate_fn=active_collate_fn, num_workers=4, drop_last=False,
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
        feat_mode=args.feat_mode,
        attn_mode=args.attn_mode,
        use_text=args.use_text,
        decoder_mode=args.decoder_mode,
        n_graph_temporal_layers=args.n_graph_temporal_layers,
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
        if args.feat_mode == "anytop13":
            # A 6ch/fk6 baseline ckpt carries `encoder.motion_proj.*` (the shared
            # motion MLP); anytop13 replaces it with `encoder.motion_proj_root.*`
            # + `encoder.motion_proj_nonroot.*` (fresh-init, allow-listed in the
            # missing-keys check below). Drop the obsolete keys so they are not
            # treated as fatal-unexpected during warm-start (codex P1).
            sd_filtered = {k: v for k, v in sd_filtered.items()
                           if not k.startswith("encoder.motion_proj.")}
        if args.attn_mode == "graphormer":
            # A scalar-attn ckpt carries `.geodesic_bias.`/`.adjacency_bias.` in
            # each graph layer; graphormer replaces them with topo/edge embeddings
            # (fresh-init, allow-listed by substring below). q/k/v/o/norm/ff keys
            # are structurally identical and DO transfer.
            sd_filtered = {k: v for k, v in sd_filtered.items()
                           if ".geodesic_bias." not in k
                           and ".adjacency_bias." not in k}
        if args.pool_type == "edge_segment":
            # v2 EdgeSegmentPool has NO learnable pool.* state (rule-based +
            # hard assignment + segment mean). A v1 (dynamic/deterministic)
            # init_ckpt carries pool.q_proj/k_proj/etc — drop them, otherwise
            # they fire as unexpected on strict-load below. codex P1
            # 2026-05-23 (019e5693-2fcf-7612-adf4-7e920611e7b2).
            sd_filtered = {k: v for k, v in sd_filtered.items()
                           if not k.startswith("pool.")}
        load_result = vae.load_state_dict(sd_filtered, strict=False)
        # Strict: no unexpected baseline keys allowed (after slot_assignment filter)
        if len(load_result.unexpected_keys) > 0:
            raise RuntimeError(
                f"[INIT_CKPT FAIL] unexpected baseline keys: "
                f"{load_result.unexpected_keys[:5]} (total {len(load_result.unexpected_keys)})")
        # Strict: missing keys must be confined to new-module prefixes /
        # fresh-init graphormer edge-embedding suffixes.
        allowed_missing_prefixes = ("pool.", "unpool.", "dist", "treeik_head.",
                                    "anytop13_head.", "encoder.motion_proj_root.",
                                    "encoder.motion_proj_nonroot.", "text_proj.",
                                    "graph_temporal_layers.")
        allowed_missing_substrings = (".topo_q_emb.", ".topo_k_emb.",
                                      ".edge_q_emb.", ".edge_k_emb.")
        bad_missing = [
            k for k in load_result.missing_keys
            if not any(k.startswith(p) for p in allowed_missing_prefixes)
            and not any(s in k for s in allowed_missing_substrings)
        ]
        if bad_missing:
            raise RuntimeError(
                f"[INIT_CKPT FAIL] {len(bad_missing)} baseline keys NOT loaded "
                f"(not in allowed-missing prefixes): {bad_missing[:5]}")
        log(f"  loaded {len(sd_filtered)} keys; allowed-missing={len(load_result.missing_keys)} unexpected=0")

    # DDP wrap — AFTER warm-start (init_ckpt loads into the raw module) and the
    # use_name_embed setattr, which both need the unwrapped module. After this
    # `vae` is the DDP wrapper; `raw_vae` is the unwrapped module (state_dict).
    # find_unused_parameters=True is REQUIRED: MotionDecoder.output_proj is never
    # called (decoder runs return_features=True), and DynamicGraphUnpool is unused
    # in the coarse_xattn / graph_temporal decoder modes.
    if is_ddp:
        vae = DDP(vae, device_ids=[local_rank], find_unused_parameters=True)
    raw_vae = vae.module if is_ddp else vae

    # Pre-build loss weights (codex M1.5 High: use same in train and val).
    # feat_mode picks the loss schema — anytop13 has pos/rot/vel/contact groups,
    # fk6 has the legacy pos/vel/vel_consistency/bone terms.
    if args.feat_mode == "anytop13":
        loss_weights = {
            "pos": args.w_pos, "rot": args.w_rot, "vel": args.w_vel,
            "contact": args.w_contact, "kl": args.w_kl,
            "pool_aux": args.w_pool_aux,
        }
    else:
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
        if is_ddp:
            train_sampler.set_epoch(epoch)   # reshuffle shards each epoch
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
                _pred_key = "pred_motion" if args.feat_mode == "anytop13" else "pred_pos"
                assert out[_pred_key].dtype == torch.float32, (
                    f"[DTYPE FAIL] {_pred_key}={out[_pred_key].dtype} not fp32")
                log(f"  [gate2 ok] z=[{B},{T_lat},{C},{D}] dtype=fp32")
            # Codex M1.5 R3 P0: use frame_mask_recovered (⊆ batch.frame_mask) as effective
            # mask so stride-tail frames the decoder zeros are not penalized.
            effective_frame_mask = out["frame_mask_recovered"]
            if it == 0 and epoch == 0:
                dropped = (batch.frame_mask & ~effective_frame_mask).sum().item()
                log(f"  [stride-tail] frames dropped by stride={args.temporal_stride}: "
                    f"{dropped}/{batch.frame_mask.sum().item()}")
            losses = run_loss(out, batch, args.feat_mode, loss_weights,
                              effective_frame_mask, dev)

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
                # Persist diagnostics to JSON (codex M1.5 Medium) — rank 0 only
                if is_main:
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

        # Save metrics — rank 0 only
        if is_main:
            rec = {"epoch": epoch, "train_loss": train_loss_mean,
                   "train_diag": train_diag_smoothed,
                   "epoch_dt_s": epoch_dt, "n_iter": n_iter,
                   "loss_breakdown": {k: float(np.mean(v)) for k, v in epoch_losses.items()}}
            with open(metrics_path, "a") as f:
                f.write(json.dumps(rec) + "\n")

        # Val + per-species recon — under DDP rank 0 runs the full val set;
        # other ranks skip the body and wait at the barrier after it.
        do_val = ((epoch + 1) % args.save_every == 0
                  or epoch == args.epochs - 1 or args.smoke)
        if do_val and is_main:
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
                    # raw_vae (unwrapped) — val runs on rank 0 only; calling the
                    # DDP wrapper here would risk a one-sided buffer-broadcast
                    # collective if the model ever gains buffers (codex hardening).
                    out = raw_vae(batch, sample=False)
                    # Codex M1.5 R3 P0: effective frame mask (stride-tail consistency)
                    effective_frame_mask_val = out["frame_mask_recovered"]
                    losses = run_loss(out, batch, args.feat_mode, loss_weights,
                                      effective_frame_mask_val, dev)
                    for k, v in losses.items():
                        val_losses[k].append(v.item())
                    # Position prediction for diagnostics: fk6 -> pred_pos;
                    # anytop13 -> pred_motion channels 0:3 vs anytop_x 0:3.
                    if args.feat_mode == "anytop13":
                        pp = out["pred_motion"][..., :3]
                        gp = batch.anytop_x.permute(0, 3, 1, 2)[..., :3]
                    else:
                        pp = out["pred_pos"]  # [B, T, J, 3]
                        gp = batch.motion_features[..., :3]
                    # Per-species per-sample pos loss (codex M1.5 Medium)
                    psl = _per_sample_pos_loss(
                        pp, gp, batch.joint_mask, effective_frame_mask_val
                    )
                    for i, sid in enumerate(batch.skeleton_id):
                        per_species_pos[sid].append(psl[i].item())
                    # Codex M1.5 frozen-pred audit (2026-05-21): speed-ratio metric
                    # to expose static-pose shortcut early. Per-frame mean displacement
                    # ratio (pred/gt). Frozen if ratio < 0.1.
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
            if args.feat_mode == "anytop13":
                recon_keys = ("pos", "rot", "vel", "contact")
            else:
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
                "model_state_dict": raw_vae.state_dict(),
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
                    "model_state_dict": raw_vae.state_dict(),
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
                    "model_state_dict": raw_vae.state_dict(),
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

        # Periodic PRESERVED save (every periodic_save_every epochs, ep{N}_model.pt).
        # Mirrors train_denoiser.py (commit f407aec). For long multi-cont VAE
        # runs (e.g., PlanetZoo 300ep) keep sweeping ckpt history beyond best+last.
        # Rank 0 only; raw_vae unwrap consistent with last_model save pattern.
        if args.periodic_save_every > 0 and ((epoch + 1) % args.periodic_save_every) == 0:
            if is_main:
                periodic_path = out_dir / f"ep{epoch + 1:04d}_model.pt"
                torch.save({
                    "epoch": epoch, "val_loss": best_val_loss,
                    "val_recon": best_val_recon,
                    "model_state_dict": raw_vae.state_dict(),
                    "optimizer_state_dict": opt.state_dict(),
                    "args": vars(args),
                    "git_sha": git_sha,
                }, periodic_path)
                log(f"  saved periodic ckpt → {periodic_path}")

        # DDP: non-main ranks waited here while rank 0 ran validation.
        if do_val and is_ddp:
            dist.barrier()

        if args.smoke and epoch >= 0:
            log("=== SMOKE MODE: 1 epoch done, exit ===")
            break

    log("=== training complete ===")
    if is_ddp:
        dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    sys.exit(main())
