"""Phase-2 train_denoiser.py — train GraphSaladDenoiser on frozen Graph-VAE latents.

Loop per docs/phase2_diffusion_design.md §3:
  - Load Phase-1 VAE ckpt (use_text=False); freeze (vae.eval + requires_grad=False).
  - Init GraphSaladDenoiser + AdamW (lr=5e-4 default, wd=1e-6, betas=(0.9,0.99)).
  - Scheduler: DDIMScheduler (v_prediction, scaled_linear, 1000 train steps).
  - Per step: vae.encode(batch, sample=True) → z₀; sample noise + timesteps;
    add_noise → z_t; get_velocity → v_target; CFG drop has_text (10%);
    denoiser.forward → v_pred; masked MSE on valid (coarse_mask × frame_mask)
    slots; AdamW step with grad-norm clip 1.0.
  - Val: full sweep, val_denoise loss (same masked MSE). Best by val.

Usage:
  python scripts/train_denoiser.py \\
      --vae_ckpt runs/m1_7_anytop13_coarse_xattn_seed42/best_recon_model.pt \\
      --caption_emb_cache data/anytop_caption_t5_1070.npz \\
      --out runs/m2_denoiser_v1_seed42 \\
      --epochs 500 --batch_size 16 --lr 5e-4 --seed 42

  python scripts/train_denoiser.py --smoke ... (1 epoch)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, DistributedSampler

# Repo path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.anytop_dataset import AnyTopDataset, collate_fn as anytop_collate_fn
from src.models.graph_salad.batch import GraphMotionBatch
from src.models.graph_salad.vae import GraphMotionVAE
from src.models.graph_salad.denoiser import GraphSaladDenoiser

try:
    from diffusers import DDIMScheduler
except ImportError as e:
    raise SystemExit(
        "train_denoiser requires diffusers (pip install diffusers); "
        "DDIMScheduler is used for v_prediction + scaled_linear scheduling."
    ) from e


# ---------------------------------------------------------------------------
# VAE rebuild from ckpt args
# ---------------------------------------------------------------------------

def load_frozen_vae(ckpt_path: str, dev: torch.device) -> tuple[GraphMotionVAE, dict]:
    """Rebuild and load a frozen GraphMotionVAE from train_graph_vae.py ckpt.

    Phase-2 contract:
      - VAE MUST have use_text=False (denoiser owns its own text conditioning;
        otherwise decode would re-apply text → double conditioning).
    """
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    ta = ck.get("args", {})
    if not ta:
        raise SystemExit(f"vae ckpt {ckpt_path} missing 'args' key")
    if bool(ta.get("use_text", False)):
        raise SystemExit(
            f"Phase-2 VAE must have use_text=False (denoiser owns text conditioning); "
            f"ckpt args has use_text={ta.get('use_text')!r}"
        )
    vae = GraphMotionVAE(
        pool_type=ta["pool_type"],
        pool_tau=ta.get("pool_tau"),
        d_model=ta["d_model"], n_heads=ta["n_heads"], d_ff=ta["d_ff"],
        n_graph_layers=ta["n_graph_layers"],
        n_enc_temporal_layers=ta["n_enc_temporal_layers"],
        n_cross_layers=ta["n_cross_layers"],
        n_dec_temporal_layers=ta["n_dec_temporal_layers"],
        n_treeik_layers=ta["n_treeik_layers"],
        max_coarse=ta["max_coarse"], local_radius=ta["local_radius"],
        temporal_stride=ta["temporal_stride"],
        temporal_kernel=ta["temporal_kernel"],
        dropout=ta["dropout"],
        feat_mode=ta["feat_mode"],
        attn_mode=ta.get("attn_mode") or "scalar",
        use_text=False,
        decoder_mode=ta.get("decoder_mode") or "unpool_identity",
        n_graph_temporal_layers=ta.get("n_graph_temporal_layers", 4),
    ).to(dev)
    missing, unexpected = vae.load_state_dict(ck["model_state_dict"], strict=True)
    if missing or unexpected:
        raise SystemExit(
            f"VAE ckpt strict-load failed: missing={len(missing)} unexpected={len(unexpected)}"
        )
    vae.encoder.use_name_embed = bool(ta.get("use_name_embed", False))
    vae.eval()
    for p in vae.parameters():
        p.requires_grad_(False)
    return vae, ta


# ---------------------------------------------------------------------------
# Denoiser loss: masked MSE over valid (coarse × frame) slots
# ---------------------------------------------------------------------------

def masked_v_mse(v_pred: torch.Tensor, v_target: torch.Tensor,
                  coarse_mask: torch.Tensor, frame_mask: torch.Tensor) -> torch.Tensor:
    """Mean-squared error over valid (coarse_mask × frame_mask) positions,
    averaged over batch + valid slots + feature dim. Padded positions are
    ignored entirely.
    """
    # [B,T_lat,C,1] mask
    mask = (coarse_mask[:, None, :, None] & frame_mask[:, :, None, None])
    mask_f = mask.to(v_pred.dtype)
    diff_sq = (v_pred - v_target).pow(2) * mask_f
    # Denominator: total valid positions × feature dim
    denom = mask_f.sum() * v_pred.shape[-1]
    return diff_sq.sum() / denom.clamp(min=1.0)


# ---------------------------------------------------------------------------
# Preflight: caption coverage on train + val
# ---------------------------------------------------------------------------

def preflight_caption_coverage(ds_train, ds_val) -> None:
    """Fail loud if any sample is missing has_text=True. Phase-2 CFG requires
    100% caption coverage: cond_drop is explicit (forced False with prob 0.1).
    A silent has_text=False from missing cache → that sample is always uncond
    → CFG schedule breaks.

    Coverage is determined purely by membership: __getitem__ sets
    has_text=True iff `info["motion_id"]` has a non-empty entry in
    `ds.caption_embs_multi` (see AnyTopDataset.__getitem__). So the check is a
    pure in-memory dict lookup over ds.samples — NO need to materialize every
    sample via ds[i], which loads each motion .npy + runs geometry derivation
    (~30-60 min for the 80k clean-L2 set; the old loop did exactly that).
    """
    for split_name, ds in (("train", ds_train), ("val", ds_val)):
        cov = ds.caption_embs_multi
        n = len(ds.samples); n_has = 0; missing = []
        for s in ds.samples:
            mid = s["motion_id"]
            caps = cov.get(mid)
            if caps is not None and len(caps) > 0:
                n_has += 1
            else:
                missing.append(mid)
        if missing:
            raise SystemExit(
                f"PREFLIGHT FAIL: {split_name} split has {len(missing)} samples "
                f"without caption_emb (has_text=False). First 5: {missing[:5]}. "
                f"Run scripts/precompute_t5_captions.py to cover the full caption set."
            )
        print(f"  preflight {split_name}: {n_has}/{n} has_text=True [OK]")


# ---------------------------------------------------------------------------
# Train loop
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    # I/O
    ap.add_argument("--vae_ckpt", required=True, help="Phase-1 VAE ckpt path")
    ap.add_argument("--caption_emb_cache", required=True,
                    help="T5 caption .npz cache (data/anytop_caption_t5_1070.npz)")
    ap.add_argument("--out", required=True, help="output dir for ckpts + logs")
    ap.add_argument("--overwrite", action="store_true",
                    help="allow overwriting non-empty out dir")
    # Dataset
    ap.add_argument("--anytop_root", default=None,
                    help="override AnyTop processed-data root (default: take from VAE ckpt args)")
    ap.add_argument("--max_frames", type=int, default=260,
                    help="Max motion frames for denoiser (full-motion mode, "
                         "2026-05-25). Default 260 covers AnyTop bank max T=237 "
                         "with safety margin. MUST be divisible by VAE's "
                         "temporal_stride (260/4=65 latent frames). VAE itself "
                         "is unaffected — still trained at 64-window. This "
                         "param ONLY controls denoiser dataset crop/pad target, "
                         "eliminating file-level caption + random-64-crop "
                         "misalignment (see handoff design doc 20260525_221220).")
    ap.add_argument("--max_joints", type=int, default=143)
    ap.add_argument("--val_frac", type=float, default=0.05,
                    help="train/val split fraction; MUST match the Phase-1 VAE "
                         "(VAE used val_frac=0.05) so the diffusion val set is the "
                         "SAME per-object-stratified holdout — otherwise val is "
                         "incomparable and risks leakage (diffusion val motions the "
                         "VAE trained on). Ignored when --full_data_val_species set.")
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--full_data_val_species", type=str, default=None,
                    help="Full-data training mode with species-filtered val. "
                         "When set: train uses split='all' (all 1070 motions, "
                         "no holdout, random_caption=True for SALAD-style "
                         "multi-cap sampling); val uses split='all' filtered "
                         "to listed comma-separated species "
                         "(e.g. 'Dragon,Monkey,Centipede,Horse', random_caption"
                         "=False primary-only). Train and val OVERLAP on those "
                         "species — intentional, val measures denoise quality "
                         "on hardest skeletons. Mirrors train_graph_vae.py "
                         "--full_data_val_species (2026-05-24).")
    # Optim
    ap.add_argument("--epochs", type=int, default=500)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-6)
    ap.add_argument("--warmup_iters", type=int, default=2000)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    # Denoiser arch
    ap.add_argument("--n_layers", type=int, default=5)
    ap.add_argument("--d_ff", type=int, default=None,
                    help="default = 4 * d_model")
    ap.add_argument("--dropout", type=float, default=0.1)
    # Diffusion
    ap.add_argument("--num_train_timesteps", type=int, default=1000)
    ap.add_argument("--beta_start", type=float, default=0.00085)
    ap.add_argument("--beta_end", type=float, default=0.012)
    ap.add_argument("--beta_schedule", default="scaled_linear")
    ap.add_argument("--cond_drop_prob", type=float, default=0.1,
                    help="CFG cond-drop probability per sample")
    # Logging / checkpoint
    ap.add_argument("--val_every", type=int, default=5)
    ap.add_argument("--save_every", type=int, default=10)
    ap.add_argument("--periodic_save_every", type=int, default=0,
                    help="Also save a PRESERVED ckpt named ep{N}_model.pt every "
                         "N epochs (in addition to last_model.pt overwrite). "
                         "0 disables. Useful for long runs (e.g. 4000 ep) where "
                         "you want sweeping ckpt history rather than only "
                         "best+last.")
    # Resume / warm-start
    ap.add_argument("--init_ckpt", default=None,
                    help="warm-start the denoiser from this ckpt's "
                         "model_state_dict (loaded strict=True). Optimizer + "
                         "epoch state are NOT restored — fresh AdamW + epoch 0. "
                         "Use for continuation runs (e.g. ep1000 → ep3000). "
                         "Pass --warmup_iters 200 (or 0) since the model is "
                         "already past initial unstable regime.")
    # Misc
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--smoke", action="store_true",
                    help="1-epoch smoke + early exit")
    return ap.parse_args()


def _ddp_setup() -> tuple[bool, int, int, int, bool]:
    """Detect DDP from torchrun env (WORLD_SIZE/RANK/LOCAL_RANK). When ws>1,
    init nccl + set_device. Returns (is_ddp, rank, local_rank, world_size,
    is_main). Mirrors train_graph_vae.py DDP setup."""
    ws = int(os.environ.get("WORLD_SIZE", 1))
    if ws > 1:
        if not dist.is_initialized():
            dist.init_process_group(backend="nccl")
        rank = dist.get_rank()
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)
        return True, rank, local_rank, ws, (rank == 0)
    return False, 0, 0, 1, True


def main() -> int:
    args = parse_args()
    is_ddp, rank, local_rank, world_size, is_main = _ddp_setup()

    if args.device == "cuda" and not torch.cuda.is_available():
        if is_main:
            print("  [INFO] CUDA unavailable; falling back to CPU")
        args.device = "cpu"
    if is_ddp:
        dev = torch.device(f"cuda:{local_rank}")
    else:
        dev = torch.device(args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if dev.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    # Out dir guard runs on all ranks (same args → all raise → clean exit, no hang)
    out_dir = Path(args.out)
    if out_dir.exists() and any(out_dir.iterdir()):
        if not args.overwrite:
            raise SystemExit(
                f"out dir {out_dir} is non-empty; pass --overwrite to allow"
            )
    out_dir.mkdir(parents=True, exist_ok=True)
    # Codex P1 fix (2026-05-25): barrier here BEFORE rank-0 creates train.log,
    # so a slower non-main rank checking "non-empty out dir" doesn't see the
    # newly-created log file and falsely fail the guard.
    if is_ddp:
        dist.barrier()
    # Logging fns — rank 0 writes log file; other ranks no-op
    log_fp = open(out_dir / "train.log", "w") if is_main else None
    def log(msg: str) -> None:
        if is_main:
            print(msg); log_fp.write(msg + "\n"); log_fp.flush()
    log(f"=== M1.7 Phase-2 train_denoiser ===")
    log(f"git_sha: {os.popen('git rev-parse HEAD 2>/dev/null').read().strip() or 'unknown'}")
    log(f"device: {dev}")
    log(f"args: {vars(args)}")

    # ---- VAE (frozen) ----
    log(f"\nLoading frozen VAE: {args.vae_ckpt}")
    vae, ta = load_frozen_vae(args.vae_ckpt, dev)
    log(f"  pool_type={ta['pool_type']} feat_mode={ta['feat_mode']} "
        f"attn_mode={ta.get('attn_mode')} decoder_mode={ta.get('decoder_mode')} "
        f"d_model={ta['d_model']} max_coarse={ta['max_coarse']} max_frames={ta['max_frames']}")

    # Full-motion mode (2026-05-25): denoiser uses args.max_frames (default 260),
    # NOT inherited from VAE ckpt's max_frames (=64). This eliminates the random
    # 64-crop / file-level caption misalignment. VAE itself is unchanged — still
    # trained at 64-window but encodes arbitrary T at inference (Conv1d/AvgPool1d
    # + attention are shape-flexible). See handoff/20260525_221220.
    temporal_stride = ta["temporal_stride"]
    if args.max_frames % temporal_stride != 0:
        raise SystemExit(
            f"[ARGS FAIL] --max_frames {args.max_frames} not divisible by "
            f"VAE temporal_stride {temporal_stride}. Pick a multiple "
            f"(e.g. 260 with stride 4 → T_lat=65)."
        )
    T_lat_expected = args.max_frames // temporal_stride
    log(f"  denoiser_max_frames={args.max_frames}  vae_training_max_frames="
        f"{ta['max_frames']}  temporal_stride={temporal_stride}  "
        f"→ T_lat={T_lat_expected}")

    # ---- Dataset ----
    ds_kwargs = dict(
        num_frames=args.max_frames,
        max_joints=ta.get("max_joints", args.max_joints),
        caption_emb_cache=args.caption_emb_cache,
        val_frac=args.val_frac,
    )
    if args.anytop_root or ta.get("anytop_root"):
        ds_kwargs["data_root"] = args.anytop_root or ta["anytop_root"]
    # M1.7 Phase-2 P1 fix (2026-05-23): SALAD-style multi-caption random sample
    # for train (each __getitem__ picks one of the motion's 5-6 captions at random)
    # vs deterministic primary-only for val (keeps val_denoise loss stable across
    # epochs for the best-ckpt gate).
    if args.full_data_val_species is not None:
        # Full-data mode (2026-05-24): mirrors train_graph_vae.py. train=all
        # 1070 (no holdout), val=all 1070 filtered to listed species. Both use
        # multi-cap sampling rules: train random_caption=True (5498-cap pool),
        # val random_caption=False (primary-only deterministic).
        val_species_set = set(
            s.strip() for s in args.full_data_val_species.split(",") if s.strip()
        )
        if not val_species_set:
            raise SystemExit(
                f"--full_data_val_species parsed to empty set from "
                f"{args.full_data_val_species!r}"
            )
        # Full-motion mode (2026-05-25): random_crop=False on BOTH train + val
        # so denoiser sees pad-to-max_frames full motion + frame_mask, eliminating
        # random 64-crop / file-level caption misalignment. random_caption=True
        # for train (SALAD multi-cap) preserved; val deterministic primary-only.
        ds_train = AnyTopDataset(
            split="all", random_caption=True, random_crop=False, **ds_kwargs)
        ds_val = AnyTopDataset(
            split="all", random_caption=False, random_crop=False, **ds_kwargs)
        ds_val.samples = [s for s in ds_val.samples
                          if s["object_type"] in val_species_set]
        if len(ds_val.samples) == 0:
            raise SystemExit(
                f"[DATA] val species filter {sorted(val_species_set)!r} "
                f"matched 0 motions. Check spelling vs AnyTop cond.npy keys.")
        present = sorted({s["object_type"] for s in ds_val.samples})
        missing = sorted(val_species_set - set(present))
        if missing:
            log(f"  [WARN] val species not in dataset (skipped): {missing}")
        log(f"  [FULL-DATA MODE] train=all 1070 ({len(ds_train)} samples), "
            f"val=species-filtered to {sorted(val_species_set)!r} "
            f"({len(ds_val)} samples). Train/val OVERLAP on these species "
            f"(intentional — val = denoise quality on hard skeletons).")
    else:
        # Full-motion mode (2026-05-25): default split (855/215) also uses
        # random_crop=False for symmetry with full_data mode. random_caption=True
        # for train preserved (multi-cap diversity unchanged).
        ds_train = AnyTopDataset(
            split="train", random_caption=True, random_crop=False, **ds_kwargs)
        ds_val = AnyTopDataset(
            split="val", random_caption=False, random_crop=False, **ds_kwargs)
        log(f"  ds_train={len(ds_train)} (random_caption=True, random_crop=False)"
            f"  ds_val={len(ds_val)} (random_caption=False, primary only)")
    if len(ds_train) < args.batch_size:
        raise SystemExit(f"[DATA] train size {len(ds_train)} < batch_size {args.batch_size}")
    if len(ds_val) == 0:
        raise SystemExit("[DATA] val split is empty")

    # ---- Preflight: max_frames coverage of all source motions ----
    # Full-motion mode (2026-05-25): AnyTopDataset silently truncates if T_var
    # > num_frames. With max_frames=260 and AnyTop max T=237, no truncation
    # should ever fire. Verify by reading npy headers via mmap (~1s for 1070).
    log(f"\nPreflight: max_frames={args.max_frames} coverage check")
    violations = []
    for s in list(ds_train.samples) + list(ds_val.samples):
        T_raw = int(np.load(s["path"], mmap_mode="r").shape[0])
        if T_raw > args.max_frames:
            violations.append((s["object_type"], s.get("motion_id", "?"), T_raw))
    if violations:
        msg = "\n".join(f"    {sp}/{mid}: T={T}" for sp, mid, T in violations[:10])
        raise SystemExit(
            f"[DATA FAIL] {len(violations)} samples exceed --max_frames="
            f"{args.max_frames} (would be silently truncated, breaking "
            f"caption-motion alignment). Examples:\n{msg}\n"
            f"Bump --max_frames or filter dataset."
        )
    log(f"  preflight: 0 / {len(ds_train) + len(ds_val)} samples exceed "
        f"max_frames={args.max_frames}")

    # ---- Preflight: caption coverage + multi-cap cache check ----
    log(f"\nPreflight: T5 caption coverage check")
    preflight_caption_coverage(ds_train, ds_val)
    # Codex P1 (2026-05-23): a legacy single-cap cache would PASS the
    # has_text coverage check above while silently disabling SALAD-style
    # multi-cap random sampling. Train must see avg > 1 caption/motion.
    if ds_train.random_caption:
        n_cached_motions = len(ds_train.caption_embs_multi)
        n_total_caps = sum(len(v) for v in ds_train.caption_embs_multi.values())
        avg_caps = n_total_caps / max(n_cached_motions, 1)
        log(f"  multi-cap check: {n_total_caps} embeddings across "
            f"{n_cached_motions} motions (avg {avg_caps:.2f}/motion)")
        if avg_caps < 1.5:
            raise SystemExit(
                f"[PREFLIGHT FAIL] random_caption=True but avg captions per "
                f"motion = {avg_caps:.2f} (< 1.5). Cache "
                f"{args.caption_emb_cache!r} looks like a legacy single-cap "
                f"file. Re-run scripts/precompute_t5_captions.py to produce "
                f"the multi-cap cache (anytop_caption_t5_*_multi.npz), or pass "
                f"--cond_drop_prob 0 if you truly intend single-cap training."
            )

    if is_ddp:
        train_sampler = DistributedSampler(
            ds_train, shuffle=True, drop_last=True,
            num_replicas=world_size, rank=rank,
        )
    else:
        train_sampler = None
    dl_train = DataLoader(
        ds_train, batch_size=args.batch_size,
        sampler=train_sampler, shuffle=(train_sampler is None),
        collate_fn=anytop_collate_fn, num_workers=args.num_workers,
        drop_last=True, pin_memory=True,
        persistent_workers=(args.num_workers > 0),
        prefetch_factor=4 if args.num_workers > 0 else None,
    )
    dl_val = DataLoader(
        ds_val, batch_size=args.batch_size, shuffle=False,
        collate_fn=anytop_collate_fn, num_workers=max(1, args.num_workers // 2),
        drop_last=False, pin_memory=True,
        persistent_workers=(args.num_workers > 0),
        prefetch_factor=4 if args.num_workers > 0 else None,
    )

    # ---- Denoiser ----
    d_model = ta["d_model"]
    n_heads = ta["n_heads"]
    d_ff = args.d_ff if args.d_ff is not None else 4 * d_model
    denoiser = GraphSaladDenoiser(
        d_model=d_model, n_heads=n_heads, d_ff=d_ff,
        n_layers=args.n_layers, d_text=768, dropout=args.dropout,
    ).to(dev)
    n_params = sum(p.numel() for p in denoiser.parameters())
    log(f"\nDenoiser: n_layers={args.n_layers} d_model={d_model} d_ff={d_ff} "
        f"params={n_params:,}")

    # ---- Warm-start from --init_ckpt (continuation runs only) ----
    # Mirrors train_graph_vae.py's --init_ckpt pattern: only model weights are
    # restored; AdamW state + epoch counter + best_val + RNG all start fresh.
    # This is the conservative continuation pattern (you lose Adam moments but
    # avoid optimizer-state version skew). For ep1000 → ep3000 continuation,
    # pass --warmup_iters small (e.g. 200) since the model is past the
    # zero-init unstable regime.
    if args.init_ckpt is not None:
        if not Path(args.init_ckpt).exists():
            raise SystemExit(f"--init_ckpt {args.init_ckpt!r} does not exist")
        log(f"\nWarm-start denoiser from {args.init_ckpt}")
        ck = torch.load(args.init_ckpt, map_location="cpu", weights_only=False)
        sd = ck.get("model_state_dict", ck)
        missing, unexpected = denoiser.load_state_dict(sd, strict=True)
        if missing or unexpected:
            raise SystemExit(
                f"[INIT_CKPT FAIL] missing={len(missing)} unexpected={len(unexpected)}; "
                f"refusing to silently load partial weights"
            )
        prev_ep = ck.get("epoch", "?")
        prev_val = ck.get("val_denoise", "?")
        log(f"  loaded model_state_dict strict=True (prev epoch={prev_ep} val_denoise={prev_val})")
        log(f"  optimizer state + epoch counter + best_val are FRESH "
            f"(continuation pattern; pass --warmup_iters {args.warmup_iters} "
            f"to control re-warmup)")

    # ---- DDP wrap (AFTER warm-start: init_ckpt loads into raw module) ----
    # find_unused_parameters=True: GraphSaladDenoiser has CFG dropout (some text
    # samples gated to zero each batch). Text-projection weights are still
    # touched via masked tensor, but to be safe under varying cond mixes we
    # enable the conservative check. Mirrors train_graph_vae.py.
    if is_ddp:
        denoiser = nn.parallel.DistributedDataParallel(
            denoiser, device_ids=[local_rank], find_unused_parameters=True,
        )
    raw_denoiser = denoiser.module if is_ddp else denoiser

    # ---- Optimizer + scheduler + lr-warmup ----
    opt = torch.optim.AdamW(
        denoiser.parameters(), lr=args.lr,
        betas=(0.9, 0.99), weight_decay=args.weight_decay,
    )
    sched = DDIMScheduler(
        num_train_timesteps=args.num_train_timesteps,
        beta_start=args.beta_start, beta_end=args.beta_end,
        beta_schedule=args.beta_schedule,
        prediction_type="v_prediction",
        clip_sample=False,
    )

    def lr_for(it: int) -> float:
        if args.warmup_iters > 0 and it < args.warmup_iters:
            return args.lr * (it + 1) / args.warmup_iters
        return args.lr

    metrics_fp = open(out_dir / "metrics.jsonl", "w") if is_main else None
    best_val = float("inf")
    global_it = 0
    epochs = 1 if args.smoke else args.epochs
    log(f"\nTraining for {epochs} epochs (smoke={args.smoke})")
    log(f"steps per epoch: {len(dl_train)}"
        + (f" (per-rank, world_size={world_size})" if is_ddp else ""))

    for epoch in range(epochs):
        if is_ddp:
            train_sampler.set_epoch(epoch)
        denoiser.train()
        t_ep = time.time()
        ep_losses = []
        for batch_idx, raw in enumerate(dl_train):
            # device transfer
            raw = {k: v.to(dev) if torch.is_tensor(v) else v for k, v in raw.items()}
            batch = GraphMotionBatch.from_collate_dict(raw)

            # Encode (frozen VAE) — sample=True to use the training z distribution
            with torch.no_grad():
                enc = vae.encode(batch, sample=True)
            z0 = enc["z"]                                       # [B,T_lat,C,D]
            pooled_adj = enc["pooled_adjacency"]
            pooled_geo = enc["pooled_geodesic"]
            coarse_mask = enc["coarse_mask"]
            frame_mask = enc["frame_mask_lat"]
            pooled_skel = enc["pooled_skeleton_embeddings"]

            B = z0.shape[0]

            # CFG dropout: flip some has_text to False
            ht_in = batch.has_text.to(dev) if batch.has_text.device != dev else batch.has_text
            drop_mask = torch.rand(B, device=dev) < args.cond_drop_prob
            has_text = ht_in & (~drop_mask)
            # Caption emb: zero-gate when has_text=False (defense in depth;
            # denoiser will also gate, but pre-gating here keeps the loss step deterministic)
            text_emb = batch.caption_emb.to(dev) * has_text[:, None].to(batch.caption_emb.dtype)

            # Diffusion: noise + add_noise + v_target
            noise = torch.randn_like(z0)
            timesteps = torch.randint(0, args.num_train_timesteps, (B,), device=dev).long()
            z_t = sched.add_noise(z0, noise, timesteps)
            v_target = sched.get_velocity(z0, noise, timesteps)
            # Mask z_t + v_target at padded positions (defense in depth)
            mask_4d = (coarse_mask[:, None, :, None] & frame_mask[:, :, None, None]).to(z0.dtype)
            z_t = z_t * mask_4d
            v_target = v_target * mask_4d

            # Denoiser forward
            v_pred = denoiser(
                z_t=z_t, timesteps=timesteps, text=text_emb,
                adjacency=pooled_adj, geodesic_dist=pooled_geo,
                coarse_mask=coarse_mask, frame_mask=frame_mask,
                pooled_skeleton_embeddings=pooled_skel,
                has_text=has_text,
                # Validate on the first iter only (cold-start preflight)
                validate_inputs=(global_it == 0),
            )

            loss = masked_v_mse(v_pred, v_target, coarse_mask, frame_mask)

            # P3 fail-fast (2026-05-23): a NaN/Inf loss means upstream maths
            # diverged (bad lr / bad scheduler / nan input). Crashing here
            # surfaces the bad step + lr instead of training silently into
            # all-NaN ckpts.
            if not torch.isfinite(loss):
                raise SystemExit(
                    f"[FAIL] non-finite loss at global_it={global_it} "
                    f"epoch={epoch} batch_idx={batch_idx} lr={lr_for(global_it):.2e} "
                    f"loss={loss.item()!r}. Likely lr too high or input NaN; "
                    f"inspect last ckpt + first NaN batch before relaunching."
                )

            # LR warmup
            cur_lr = lr_for(global_it)
            for pg in opt.param_groups:
                pg["lr"] = cur_lr
            opt.zero_grad()
            loss.backward()
            # Codex P2 (2026-05-23): error_if_nonfinite=True (default in modern
            # PyTorch, set explicitly for version safety) — catches NaN grads
            # that loss-finite check would have missed.
            grad_norm = torch.nn.utils.clip_grad_norm_(
                denoiser.parameters(), args.grad_clip,
                error_if_nonfinite=True,
            )
            opt.step()

            ep_losses.append(loss.item())
            global_it += 1

        epoch_loss = float(np.mean(ep_losses))
        ep_dt = time.time() - t_ep
        log(f"\n=== epoch {epoch} done in {ep_dt:.1f}s | train_loss={epoch_loss:.4f} "
            f"lr={cur_lr:.2e} n_iter={len(ep_losses)} ===")

        # Val — only rank 0 runs (full val set, no metric all-reduce needed).
        # Other ranks wait at barrier OUTSIDE the is_main block (placed below
        # the periodic-save block so all rank-0-only IO is bounded together).
        if (epoch % args.val_every == 0) or (epoch == epochs - 1) or args.smoke:
            if is_main:
                denoiser.eval()
                t_v = time.time()
                # P1 (codex 2026-05-23): use a FIXED seed (not args.seed+epoch) so
                # the best-ckpt gate measures a static objective across epochs.
                # And create the generator ONCE before the batch loop so consecutive
                # batches don't reuse the same noise/timestep draw.
                g_val = torch.Generator(device=dev).manual_seed(args.seed)
                # Aggregate numerator + denominator separately so val_loss is a true
                # element-weighted mean over all valid positions, not a batch-mean
                # of per-batch element-weighted means (codex P2-3).
                val_num = 0.0
                val_den = 0.0
                with torch.no_grad():
                    for raw in dl_val:
                        raw = {k: v.to(dev) if torch.is_tensor(v) else v for k, v in raw.items()}
                        batch = GraphMotionBatch.from_collate_dict(raw)
                        enc = vae.encode(batch, sample=False)  # deterministic eval (z=mu)
                        z0 = enc["z"]
                        pooled_adj = enc["pooled_adjacency"]
                        pooled_geo = enc["pooled_geodesic"]
                        coarse_mask = enc["coarse_mask"]
                        frame_mask = enc["frame_mask_lat"]
                        pooled_skel = enc["pooled_skeleton_embeddings"]
                        B = z0.shape[0]
                        noise = torch.randn(z0.shape, generator=g_val, device=dev, dtype=z0.dtype)
                        timesteps = torch.randint(
                            0, args.num_train_timesteps, (B,),
                            generator=g_val, device=dev
                        ).long()
                        z_t = sched.add_noise(z0, noise, timesteps)
                        v_target = sched.get_velocity(z0, noise, timesteps)
                        mask = (coarse_mask[:, None, :, None] & frame_mask[:, :, None, None])
                        mask_f = mask.to(z0.dtype)
                        z_t = z_t * mask_f; v_target = v_target * mask_f
                        has_text = batch.has_text.to(dev) if batch.has_text.device != dev else batch.has_text
                        text_emb = batch.caption_emb.to(dev) * has_text[:, None].to(batch.caption_emb.dtype)
                        # Codex P2 fix (2026-05-25): rank-0-only val uses raw
                        # module, not DDP wrapper, to avoid any future one-sided
                        # collective if denoiser ever grows DDP-tracked buffers.
                        v_pred = raw_denoiser(
                            z_t=z_t, timesteps=timesteps, text=text_emb,
                            adjacency=pooled_adj, geodesic_dist=pooled_geo,
                            coarse_mask=coarse_mask, frame_mask=frame_mask,
                            pooled_skeleton_embeddings=pooled_skel,
                            has_text=has_text, validate_inputs=False,
                        )
                        diff_sq = (v_pred - v_target).pow(2) * mask_f
                        val_num += diff_sq.sum().item()
                        val_den += mask_f.sum().item() * v_pred.shape[-1]
                val_loss = val_num / max(val_den, 1.0)
                # Codex P2 (2026-05-23): fail-fast on non-finite val too.
                if not (val_loss == val_loss and val_loss != float("inf")
                        and val_loss != float("-inf")):
                    raise SystemExit(
                        f"[FAIL] non-finite val_loss={val_loss!r} at epoch={epoch}. "
                        f"Inspect last train iter for upstream divergence."
                    )
                log(f"[val ep{epoch}] dt={time.time()-t_v:.1f}s val_denoise={val_loss:.4f} "
                    f"n_valid_positions={int(val_den/v_pred.shape[-1])}")

                metrics_fp.write(json.dumps({
                    "epoch": epoch, "train_loss": epoch_loss, "val_denoise": val_loss,
                    "lr": cur_lr, "epoch_dt_s": ep_dt, "global_it": global_it,
                }) + "\n"); metrics_fp.flush()

                # Best ckpt — rank 0 only; unwrap DDP for clean state_dict
                if val_loss < best_val:
                    best_val = val_loss
                    best_path = out_dir / "best_model.pt"
                    torch.save({
                        "epoch": epoch, "val_denoise": val_loss, "train_loss": epoch_loss,
                        "model_state_dict": raw_denoiser.state_dict(),
                        "optimizer_state_dict": opt.state_dict(),
                        "args": vars(args),
                        "vae_ckpt_args": ta,
                    }, best_path)
                    log(f"  saved best ckpt → {best_path} (val_denoise={best_val:.4f})")
            # END val + best block (rank 0 only)

        # Periodic last save — rank 0 only
        if (epoch % args.save_every == 0) or (epoch == epochs - 1) or args.smoke:
            if is_main:
                last_path = out_dir / "last_model.pt"
                torch.save({
                    "epoch": epoch, "val_denoise": best_val, "train_loss": epoch_loss,
                    "model_state_dict": raw_denoiser.state_dict(),
                    "optimizer_state_dict": opt.state_dict(),
                    "args": vars(args), "vae_ckpt_args": ta,
                }, last_path)

        # Periodic PRESERVED save (every periodic_save_every epochs, ep{N}_model.pt)
        # Uses (epoch+1) so first save is at epoch=periodic_save_every-1 (i.e.
        # after 500 epochs done, save ep0500_model.pt). Rank 0 only.
        if args.periodic_save_every > 0 and ((epoch + 1) % args.periodic_save_every) == 0:
            if is_main:
                periodic_path = out_dir / f"ep{epoch + 1:04d}_model.pt"
                torch.save({
                    "epoch": epoch, "val_denoise": best_val, "train_loss": epoch_loss,
                    "model_state_dict": raw_denoiser.state_dict(),
                    "optimizer_state_dict": opt.state_dict(),
                    "args": vars(args), "vae_ckpt_args": ta,
                }, periodic_path)
                log(f"  saved periodic ckpt → {periodic_path}")

        # Barrier: re-sync all ranks after rank-0 val/save IO. Must be outside
        # any is_main block (otherwise rank!=0 never reaches → deadlock).
        if is_ddp:
            dist.barrier()

        if args.smoke:
            log(f"\n=== SMOKE MODE: 1 epoch done, exit ===")
            break

    log("\n=== training complete ===")
    if is_main:
        metrics_fp.close(); log_fp.close()
    if is_ddp:
        dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    sys.exit(main())
