#!/usr/bin/env python3
"""Train Graph-CodeFlow (level_a probe / graph_pscf formal backbone — rectified-flow over the FROZEN Graph-VQVAE
post-RVQ z_q) — handoff/20260609_graph_codeflow_rvq_backbone_plan.md +
handoff/20260609_0530_graph_codeflow_locked_recipe_and_state.md (LOCKED recipe).

SEPARATE from train_graph_vqvae.py / train_graph_vae.py / train_denoiser.py. Reads
the offline RVQ token cache (scripts/export_graph_vq_tokens.py), trains
GraphCodeFlow (flow-only loss), and logs the continuous-vs-snapped projection QA
(THE key gate). Mirrors train_graph_vqvae.py's DDP / bf16-autocast / resume /
logging patterns so the operational behavior is familiar.

LOCKED recipe defaults: batch 64, lr 1e-4, half_cosine, warmup 2000,
eta_min_ratio 0.01, wd 0.01, grad_clip 1.0, cond_drop_prob 0.1,
flow_loss_weight 1.0, terminal/clean 0.0, seed 42, empirical z_q norm,
terminal ID CE OFF.

Modes:
  --smoke        : a few train iters + 1 val pass (single proc OK).
  --mem_profile  : one fwd+bwd at --batch_size, report peak CUDA mem, exit (does
                   NOT launch the real run).
  (default)      : full training loop. DO NOT launch the real run from this task
                   (the frozen tokenizer is still training to ep300).

Usage (smoke, single GPU):
  python scripts/train_graph_codeflow.py --token_cache /tmp/cf_tokens \
    --frozen_vqvae_ckpt /tmp/vqvae_cur.pt --smoke --out /tmp/cf_smoke --overwrite
"""
from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import sys
import time
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.models.vq_model import GraphVQTokenizer, semantic_config_from_ckpt
from src.data.holdout_guard import guard_dataset
from src.data import provenance as prov
from src.models.graph_salad.losses import compute_world_geometry_terms
from scripts.train_denoiser import decoded_speed_loss
from src.models.CodeFlow_Model import GraphCodeFlow
from src.models.CodeFlow_Model.token_dataset import TokenCacheDataset, token_collate


def _ddp_setup():
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size <= 1:
        return False, 0, 0, 1, True
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    # 30-min PG timeout (default is 10): the rank-0-only val + online gen-eval block makes
    # other ranks wait at the post-epoch dist.barrier(); a slow gen-eval must not trip NCCL.
    dist.init_process_group(backend="nccl", device_id=torch.device("cuda", local_rank),
                            timeout=timedelta(minutes=30))
    return True, rank, local_rank, world_size, rank == 0


class HumanCurriculumSampler(torch.utils.data.Sampler):
    """DDP-aware train sampler with a LATE-PHASE human-upsampling curriculum.

    - epoch < start_epoch (or factor<=1): behaves EXACTLY like a shuffled, drop_last
      DistributedSampler — a per-epoch-seeded permutation, per-rank strided shard.
    - epoch >= start_epoch: EACH RANK independently draws its own num_samples indices
      by weighted sampling WITH REPLACEMENT (human/HML3D weight = factor, others = 1),
      using a PER-RANK seed (seed + epoch*1009 + rank). Independent per-rank draws avoid
      the systematic cross-rank duplication a shared-draw+stride would cause under
      replacement (which would over-count popular human indices across ranks and inflate
      the effective upweighting); the global per-step human share stays ~factor-upweighted.
    Deterministic per (seed, epoch[, rank]); reseeded each set_epoch. __len__ is equal on
    all ranks. With num_replicas=1/rank=0 it also serves the single-process (non-DDP) path.
    """
    def __init__(self, n, is_human, factor, start_epoch, num_replicas, rank, seed=42,
                 phase2_factor=1.0, phase2_start_epoch=-1):
        self.n = int(n)
        self.is_human = torch.as_tensor(list(is_human), dtype=torch.bool)
        self.factor = float(factor)
        self.start_epoch = int(start_epoch)
        self.phase2_factor = float(phase2_factor)
        self.phase2_start_epoch = int(phase2_start_epoch)
        self.num_replicas = max(1, int(num_replicas))
        self.rank = int(rank)
        self.seed = int(seed)
        self.epoch = 0
        self.num_samples = self.n // self.num_replicas          # drop_last
        self.total_size = self.num_samples * self.num_replicas

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __len__(self):
        return self.num_samples

    def _current_factor(self):
        # TWO-PHASE (VQVAE-style): phase2 takes precedence once epoch >= phase2_start_epoch,
        # else the phase-1 factor once epoch >= start_epoch, else 1.0 (OFF). phase2 defaults
        # (1.0 / -1) keep this byte-identical to the original single-phase behaviour.
        if self.phase2_factor > 1.0 and self.phase2_start_epoch >= 0 and self.epoch >= self.phase2_start_epoch:
            return self.phase2_factor
        if self.factor > 1.0 and self.start_epoch >= 0 and self.epoch >= self.start_epoch:
            return self.factor
        return 1.0

    def _active(self):
        return self._current_factor() > 1.0

    def __iter__(self):
        if not self._active():
            # OFF / pre-curriculum: IDENTICAL to DistributedSampler(shuffle=True, drop_last=True)
            # — one permutation SHARED across ranks (seed+epoch, no rank), then per-rank disjoint
            # strided shard. Ranks see disjoint indices (the no-duplicate invariant holds).
            g = torch.Generator(); g.manual_seed(self.seed + self.epoch)
            idx = torch.randperm(self.n, generator=g)[:self.total_size]
            idx = idx[self.rank:self.total_size:self.num_replicas]
        else:
            # active: each rank draws its OWN num_samples from the human-upweighted distribution
            # with a PER-RANK seed. NOT a shared draw + stride: with replacement, a shared draw would
            # place popular (human) indices at positions that fall to multiple ranks, systematically
            # duplicating them across ranks and inflating the effective upweighting beyond `factor`.
            # Independent per-rank draws keep the global per-step human share ~factor-upweighted;
            # random cross-rank repeats remain possible but are rare, not systematic. __len__ equal.
            g = torch.Generator(); g.manual_seed(self.seed + self.epoch * 1009 + self.rank)
            w = torch.ones(self.n, dtype=torch.float)
            w[self.is_human] = self._current_factor()
            idx = torch.multinomial(w, self.num_samples, replacement=True, generator=g)
        return iter(idx.tolist())


def load_frozen_tokenizer(ckpt_path: str, dev: torch.device) -> GraphVQTokenizer:
    """Rebuild + freeze the Graph-VQVAE tokenizer (for the snapped-decode QA +
    empirical-norm decode path). eval() + requires_grad_(False)."""
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    ta = ck["args"]
    model = GraphVQTokenizer(
        d_model=ta["d_model"], n_heads=ta["n_heads"], d_ff=ta["d_ff"],
        n_graph_layers=ta["n_graph_layers"],
        n_enc_temporal_layers=ta["n_enc_temporal_layers"],
        n_pre_vq_layers=ta["n_pre_vq_layers"], n_post_vq_layers=ta["n_post_vq_layers"],
        n_cross_layers=ta["n_cross_layers"],
        n_dec_temporal_layers=ta["n_dec_temporal_layers"],
        max_coarse=ta["max_coarse"], temporal_stride=ta["temporal_stride"],
        temporal_kernel=ta["temporal_kernel"], dropout=ta["dropout"],
        code_dim=ta["code_dim"], num_codes=ta["num_codes"],
        num_quantizers=ta["num_quantizers"], ema_mu=ta["ema_mu"],
        quantize_dropout_prob=ta["quantize_dropout_prob"],
        dead_code_threshold=ta["dead_code_threshold"],
        # Semantic ckpts carry encoder.clip_proj sized for the joint-semantics table;
        # rebuilding without it fails strict load (the dev-scan fix that never landed
        # in this checkout — caught by the 8-rank smoke).
        **semantic_config_from_ckpt(ck),
    ).to(dev)
    model.load_state_dict(ck["model_state_dict"], strict=True)
    model.eval()
    model.requires_grad_(False)
    return model, ta


def build_cond(b: dict, cond_drop_prob: float, training: bool,
               dtype: torch.dtype):
    """Assemble the GraphStructuredCodeFlow conditioning dict from a token batch.

    has_text starts from the dataset flag; during training we additionally CFG-
    drop (flip True->False) with cond_drop_prob (so the model learns the uncond
    branch). Float conditioning is cast to `dtype` (fp32 unless bf16 autocast
    wraps the forward — the model enforces dtype-match on the fp32 path).
    """
    has_text = b["has_text"].clone()
    if training and cond_drop_prob > 0.0:
        drop = torch.rand(has_text.shape[0], device=has_text.device) < cond_drop_prob
        has_text = has_text & ~drop
    return {
        "text_global": b["caption_emb"].to(dtype),
        "text_tokens": b["caption_token_emb"].to(dtype),
        "text_token_mask": b["caption_token_mask"],
        "has_text": has_text,
        "pooled_adjacency": b["pooled_adjacency"].to(dtype),
        "pooled_geodesic": b["pooled_geodesic"].to(dtype),
        "pooled_skeleton_embeddings": b["pooled_skeleton_embeddings"].to(dtype),
        "coarse_mask": b["coarse_mask"],
        "frame_mask_lat": b["frame_mask_lat"],
    }


def compute_empirical_stats(ds: TokenCacheDataset, D: int, dev: torch.device,
                            max_clips: int = 0, cache_path: str | None = None,
                            cache_key: dict | None = None,
                            is_ddp: bool = False, rank: int = 0):
    """Empirical z_q mean/std over VALID tokens of the train cache (LOCKED:
    empirical normalization over the FULL train set, not codebook-stat). Streamed
    sum / sumsq. max_clips<=0 (default) uses ALL train clips; a positive cap is for
    smoke/debug only (it would normalize on a PREFIX, not the full set).

    Startup acceleration (2026-06-09): (1) DISK CACHE — if cache_path exists and its
    stored cache_key matches, load mean/std instantly (skips the full scan; survives
    resume/restart/new experiments on the same cache). cache_key encodes the cache
    identity (manifest_md5 + full-content index_md5 + n/D/max_clips) so any re-export
    or clip-set/content change INVALIDATES it (a content hash, not a byte-size).
    (2) rank-0-only under DDP — only rank 0 scans (or loads) + writes the cache, then
    broadcasts mean/std to all ranks. Avoids every rank re-decompressing the full cache
    (the un-optimized path made a 6-rank/full-length scan take ~30min; this makes a cold
    scan ~6x faster and a warm/cached start instant)."""
    n = len(ds) if max_clips <= 0 else min(len(ds), max_clips)
    mean_t = std_t = None
    count = 0
    # Only rank 0 (or a single process) scans/loads + writes the cache.
    if not is_ddp or rank == 0:
        loaded = False
        if cache_path and Path(cache_path).exists():
            try:
                c = torch.load(cache_path, map_location="cpu", weights_only=False)
                if c.get("D") == D and c.get("cache_key") == cache_key:
                    mean_t, std_t = c["mean"].float(), c["std"].float()
                    count = int(c["count"])
                    # Explicit raise (not assert: survives `python -O`, which would
                    # strip an assert and silently feed a bad-shape mean/std). A bad
                    # shape here is caught by the enclosing except -> loaded=False ->
                    # full re-scan + cache rewrite (self-heals instead of using it).
                    if tuple(mean_t.shape) != (D,) or tuple(std_t.shape) != (D,):
                        raise RuntimeError(
                            f"cached empirical stats shape mismatch: "
                            f"mean{tuple(mean_t.shape)} std{tuple(std_t.shape)} != ({D},)")
                    loaded = True
            except Exception:
                loaded = False
        if not loaded:
            count = 0
            s = torch.zeros(D, dtype=torch.float64)
            s2 = torch.zeros(D, dtype=torch.float64)
            for i in range(n):
                it = ds[i]
                z = it["z_q"].reshape(-1, D).double()       # [T_lat*C, D]
                m = it["token_mask"].reshape(-1)
                zv = z[m]
                count += zv.shape[0]
                s += zv.sum(dim=0)
                s2 += zv.pow(2).sum(dim=0)
            if count == 0:
                raise RuntimeError("compute_empirical_stats: zero valid tokens in cache")
            mean_t = (s / count).float()
            std_t = ((s2 / count) - (s / count).pow(2)).clamp_min(1e-12).sqrt().float()
            if cache_path:
                try:
                    torch.save({"mean": mean_t, "std": std_t, "count": count,
                                "D": D, "cache_key": cache_key}, cache_path)
                except Exception:
                    pass
        mean_t, std_t = mean_t.to(dev), std_t.to(dev)
    else:
        mean_t = torch.zeros(D, dtype=torch.float32, device=dev)
        std_t = torch.ones(D, dtype=torch.float32, device=dev)
    # Broadcast rank-0's stats to all ranks (DDP); every rank ends with identical norm.
    if is_ddp:
        dist.broadcast(mean_t, src=0)
        dist.broadcast(std_t, src=0)
        cnt = torch.tensor([count], dtype=torch.long, device=dev)
        dist.broadcast(cnt, src=0)
        count = int(cnt.item())
    return mean_t, std_t, count


@torch.no_grad()
def projection_qa(flow, tokenizer, b: dict, cond: dict, dev: torch.device,
                  decode: bool = False):
    """THE key gate: compare continuous decode(z_hat) vs snapped decode(z_snap)
    and report projection_error = mse(z_hat, z_snap) over valid tokens.

    Here z_hat is the model's predicted CLEAN latent from a single flow eval at a
    fixed t (denormalized to raw RVQ space), NOT a full ODE sample (cheap, per-step
    diagnostic). Returns projection_error, per-q generated-code usage, and (if
    decode=True) the max abs decoded-motion gap continuous-vs-snapped.
    """
    z_q = b["z_q"].to(dev)
    token_mask = b["token_mask"].to(dev)
    B, T_lat, C, D = z_q.shape
    # One flow eval at t~U: predict v -> clean (in normalized space) -> denorm.
    x = flow.normalize(z_q) * token_mask.unsqueeze(-1).float()
    noise = torch.randn_like(x) * flow.noise_scale * token_mask.unsqueeze(-1).float()
    t = torch.rand(B, device=dev)
    t_view = t[:, None, None, None]
    z_t = (t_view * x + (1.0 - t_view) * noise) * token_mask.unsqueeze(-1).float()
    v = flow.predict_velocity(z_t, t, cond)
    clean = flow.predict_clean_from_velocity(z_t, t, v)
    z_hat = flow.denormalize(clean) * token_mask.unsqueeze(-1).float()

    proj = tokenizer.nearest_residual_ids(z_hat, token_mask)
    indices_hat = proj["indices_hat"]
    # per-q generated code usage (#unique codes used on valid tokens).
    usage = []
    for qi in range(indices_hat.shape[-1]):
        ids_q = indices_hat[..., qi][token_mask]
        usage.append(int(torch.unique(ids_q[ids_q >= 0]).numel()))
    out = {"projection_error": float(proj["projection_error"].item()),
           "code_usage_per_q": usage}
    if decode:
        skel_meta = {
            "s_j": b["s_j"].to(dev), "assignment": b["assignment"].to(dev),
            "coarse_mask": b["coarse_mask"].to(dev),
            "frame_mask_lat": b["frame_mask_lat"].to(dev),
            "pooled_adjacency": b["pooled_adjacency"].to(dev),
            "pooled_geodesic": b["pooled_geodesic"].to(dev),
        }
        fake_batch = SimpleNamespace(joint_mask=b["joint_mask"].to(dev))
        cont = tokenizer.decode(z_hat, skel_meta, fake_batch)["pred_motion"]
        snap = tokenizer.decode_from_indices(indices_hat, skel_meta, fake_batch)["pred_motion"]
        out["decode_cont_finite"] = bool(torch.isfinite(cont).all())
        out["decode_snap_finite"] = bool(torch.isfinite(snap).all())
        out["decode_cont_vs_snap_maxabs"] = float((cont - snap).abs().max().item())
    return out


def main() -> int:
    p = argparse.ArgumentParser()
    # data / tokenizer
    p.add_argument("--token_cache", required=True, help="dir from export_graph_vq_tokens.py")
    p.add_argument("--frozen_vqvae_ckpt", required=True)
    # model
    p.add_argument("--model_variant", choices=["level_a", "graph_pscf"],
                   default="graph_pscf",
                   help="level_a = GraphStructuredCodeFlow probe (compat/smoke); "
                        "graph_pscf = 287M formal backbone (DEFAULT for formal "
                        "training, spec §5.3)")
    p.add_argument("--text_dim", type=int, default=768,
                   help="caption embedding dim (768=T5 legacy, 4096=LLM2Vec); sets both "
                        "d_text and text_token_dim")
    p.add_argument("--text_input_norm", type=int, default=0,
                   help="LayerNorm on raw caption embeddings before both text projections "
                        "(required for LLM2Vec: per-dim RMS ~17x T5's)")
    p.add_argument("--use_sentence_token", type=int, default=0,
                   help="prepend the CFG-gated pooled projection to h_text as token 0")
    p.add_argument("--text_slot_xattn", type=int, default=0,
                   help="per-block TextCrossAttention: slots query text tokens directly")
    # ARDY-style decoded-geometry loss (L_dec): decode the gradient-carrying clean
    # prediction through the FROZEN tokenizer and supervise world geometry / root
    # trajectory / world speed against the no-grad decode of the GT z_q (the token
    # cache stores no GT 13ch motion; decode(z_q) is a lossless proxy per the v4b
    # recon evidence). All default None = unset sentinel (dropped from the config
    # digest so v2/v3 ckpts keep their resume contract); explicit values participate.
    # Validated v4b-era recipe: world/traj/speed @ 0.1 each (handoff
    # 20260608_2100_decode_loss_energy_collapse_result.md, -41% energy deviation).
    p.add_argument("--w_dec_world", type=float, default=None,
                   help="decoded-clean world-geometry L1 weight (None/0 = off)")
    p.add_argument("--w_dec_traj", type=float, default=None,
                   help="decoded-clean root-trajectory L1 weight (None/0 = off)")
    p.add_argument("--w_dec_speed", type=float, default=None,
                   help="decoded-clean world-speed log-Huber weight (None/0 = off)")
    p.add_argument("--dec_geom_t_min", type=float, default=None,
                   help="apply decoded geometry only to rows with t > this (RF: t=1 is "
                        "clean, so this gates to NEAR-CLEAN rows — polarity REVERSED vs "
                        "the diffusion-era timestep<t_max gate). None = 0.6")
    p.add_argument("--dec_geom_every", type=int, default=None,
                   help="compute decoded geometry every N train steps (None = 1)")
    p.add_argument("--dec_speed_floor", type=float, default=None,
                   help="skip GT speeds <= this; clamp both speeds >= floor before log "
                        "(None = 1e-4)")
    p.add_argument("--dec_speed_loss", choices=["log_huber", "log_l1", "raw_l1"],
                   default=None, help="decoded speed loss form (None = log_huber)")
    p.add_argument("--parameterization", choices=["v", "x"], default=None,
                   help="network output semantics: v = velocity (default), x = clean-latent "
                        "x0 prediction converted to velocity inside predict_velocity "
                        "(v-space loss, same effective time weighting). Default None = "
                        "unset sentinel: it is dropped from the config digest so ckpts "
                        "trained before this flag existed keep their resume contract; an "
                        "EXPLICIT value participates in the contract.")
    p.add_argument("--caption_sampling", choices=["fixed", "random"], default="fixed",
                   help="fixed = npz-baked primary caption (legacy). random = uniform draw over "
                        "ALL of a motion's captions from the LLM2Vec sidecar, reseeded per "
                        "epoch (train only; val is always fixed for determinism)")
    p.add_argument("--caption_sidecar", default=None,
                   help="LLM2Vec ragged sidecar prefix; REQUIRED for caption_sampling=random. "
                        "Byte-verified against the token cache's caption_provenance.")
    p.add_argument("--gen_eval_caption_cache", default=None,
                   help="LLM2Vec sidecar prefix for online gen-eval caption lookup "
                        "(required when --text_dim != 768)")
    p.add_argument("--code_dim", type=int, default=512)
    p.add_argument("--n_heads", type=int, default=8)
    p.add_argument("--d_ff", type=int, default=2048)
    p.add_argument("--n_layers", type=int, default=5,
                   help="level_a only (graph_pscf uses depth_double/depth_single)")
    p.add_argument("--depth_double", type=int, default=6,
                   help="graph_pscf double-stream depth (spec §5.3)")
    p.add_argument("--depth_single", type=int, default=12,
                   help="graph_pscf single-stream depth (spec §5.3)")
    p.add_argument("--hidden_size", type=int, default=512,
                   help="graph_pscf hidden size; spec A3 pins H==D==code_dim==512 "
                        "for v1 (asserted below)")
    p.add_argument("--mlp_ratio", type=float, default=4.0,
                   help="graph_pscf DiT MLP expansion ratio (spec §5.3)")
    p.add_argument("--max_T_lat", type=int, default=75,
                   help="frame_seed / positional capacity = ceil(T_fine_max/temporal_stride); "
                        "full-length T_fine_max=300 stride 4 -> 75. Must be >= the token "
                        "cache's T_lat")
    p.add_argument("--dropout", type=float, default=None,
                   help="None = per-variant default (graph_pscf 0.05, level_a 0.1); "
                        "an explicit value always wins")
    # train (LOCKED recipe)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--epochs", type=int, default=600)
    p.add_argument("--lr_scheduler", choices=["half_cosine", "none"], default="half_cosine")
    p.add_argument("--warmup_steps", type=int, default=2000)
    p.add_argument("--eta_min_ratio", type=float, default=0.01)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--grad_accum", type=int, default=1,
                   help="gradient accumulation micro-steps per optimizer step. Default 1 = "
                        "unchanged (step every micro-batch). Use >1 to preserve the same GLOBAL "
                        "batch (=per_gpu*world*grad_accum) / learning efficiency on fewer GPUs, "
                        "e.g. 2 GPUs x bs16 x accum2 = global 64 (same as 4 GPUs x bs16 x accum1). "
                        "n_iter/LR-schedule/logging count OPTIMIZER steps, so resume stays seamless.")
    p.add_argument("--cond_drop_prob", type=float, default=0.1)
    p.add_argument("--flow_loss_weight", type=float, default=1.0)
    p.add_argument("--terminal_loss_weight", type=float, default=0.0)
    p.add_argument("--clean_loss_weight", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--amp_dtype", choices=["fp32", "bf16"], default="bf16")
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--human_upsample_factor", type=float, default=1.0,
                   help="LATE-PHASE CURRICULUM: once epoch >= --human_upsample_start_epoch, upweight "
                        "human (HML3D) clips by this factor in the train sampler. 1.0 (default) = OFF "
                        "(byte-unchanged: plain DistributedSampler). factor=2 raises human per-batch "
                        "share from the dataset's ~25%% base to ~40%%.")
    p.add_argument("--human_upsample_start_epoch", type=int, default=-1,
                   help="epoch at which human upsampling activates (-1 = never). Only relevant if "
                        "--human_upsample_factor > 1.")
    p.add_argument("--human_upsample_phase2_factor", type=float, default=1.0,
                   help="TWO-PHASE CURRICULUM (VQVAE-style): once epoch >= --human_upsample_phase2_start_epoch "
                        "the human upweight switches from --human_upsample_factor to THIS (takes precedence). "
                        "1.0 (default) = OFF (single-phase, byte-unchanged). e.g. VQVAE used factor=3.0 start=0 "
                        "phase2_factor=4.5 phase2_start=50.")
    p.add_argument("--human_upsample_phase2_start_epoch", type=int, default=-1,
                   help="epoch at which phase-2 human upsampling activates (-1 = never / single-phase). Only "
                        "relevant if --human_upsample_phase2_factor > 1; should be > --human_upsample_start_epoch.")
    p.add_argument("--empirical_stats_max_clips", type=int, default=0,
                   help="0 (default) = use ALL train clips for the empirical z_q "
                        "norm (LOCKED: full train-set stats). A positive value caps "
                        "to a PREFIX — smoke/debug only, NOT the real run.")
    # eval / cfg
    p.add_argument("--eval_cond_scale", type=float, default=4.0,
                   help="CFG scale for sampling QA — SWEEP starting point, NOT a "
                        "fixed default (project energy-overshoot history; recipe "
                        "says do not hardcode 6.0).")
    p.add_argument("--eval_steps", type=int, default=50)
    # ---- ONLINE text->motion gen-eval (frozen evaluator; opt-in, rank-0, every N epochs) ----
    p.add_argument("--gen_eval", action="store_true",
                   help="enable periodic text->motion eval in a FROZEN evaluator's space "
                        "(requires --evaluator_ckpt). rank-0 only, inside the do_val window.")
    p.add_argument("--evaluator_ckpt", type=str, default=None,
                   help="frozen AnyTopT2MEvaluator best_model.pt (12ch contact-free).")
    p.add_argument("--gen_eval_manifest", type=str, default=None,
                   help="eval val manifest; default <gen_eval_data_root>/eval_splits/val_all.json")
    p.add_argument("--gen_eval_data_root", type=str, default=None,
                   help="eval dataset root; default = the evaluator ckpt's data_root.")
    p.add_argument("--gen_eval_every", type=int, default=50,
                   help="run the online gen-eval every N epochs (sparse; generation is costly).")
    p.add_argument("--gen_eval_n", type=int, default=256,
                   help="strided subset size per gen-eval (small to bound cost; <1024 => FID skipped).")
    p.add_argument("--gen_eval_steps", type=int, default=25,
                   help="ODE steps for the online gen-eval (cheaper than --eval_steps).")
    p.add_argument("--gen_eval_batch", type=int, default=8,
                   help="clips per batched flow.sample in the online gen-eval (proven-safe "
                        "on H100/H200 alongside the resident backbone; raise only if memory allows).")
    # logging / ckpt
    p.add_argument("--log_every", type=int, default=50)
    p.add_argument("--qa_every", type=int, default=200,
                   help="run the decode-based continuous-vs-snapped QA every N steps")
    p.add_argument("--save_every", type=int, default=10)
    p.add_argument("--out", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--resume", type=str, default=None)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--smoke_iters", type=int, default=4)
    p.add_argument("--mem_profile", action="store_true",
                   help="one fwd+bwd at --batch_size, report peak CUDA mem, exit")
    # --- held-out-topology protocol ---------------------------------------------------
    p.add_argument("--protocol", default="legacy",
                   choices=["legacy", "unseen_topology_v1"],
                   help="legacy (default) reproduces every pre-existing command exactly and does "
                        "not require the held-out artifact. unseen_topology_v1 enforces the "
                        "frozen pre-registration: retained splits, artifact SHA, and abort on any "
                        "held topology. Only a run under unseen_topology_v1 may back an "
                        "unseen-topology claim.")
    p.add_argument("--holdout_artifact", type=str, default=None,
                   help="frozen pre-registration (data/holdout_topologies_v1.json). Required "
                        "unless --allow_no_holdout. The token cache this stage reads must have "
                        "been exported from the RETAINED splits.")
    p.add_argument("--holdout_sha", type=str, default=None,
                   help="expected artifact sha256; abort if the freeze differs.")
    p.add_argument("--holdout_data_root", type=str, default=None,
                   help="corpus root holding cond.npy, for resolving canonical topologies. "
                        "Defaults to the anytop_root recorded in the token-cache manifest.")
    p.add_argument("--allow_no_holdout", action="store_true",
                   help="run WITHOUT the guard; logs loudly and must not back any unseen claim.")
    args = p.parse_args()

    is_ddp, rank, local_rank, world_size, is_main = _ddp_setup()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    if is_ddp:
        dev = torch.device("cuda", local_rank)
    else:
        if args.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("[DEVICE FAIL] --device cuda but CUDA unavailable.")
        dev = torch.device(args.device)

    # Restore + dropout resolution run FIRST — before ANY code reads an architecture or
    # text arg (the arch banner, tokenizer code_dim validation and the provenance stamp all
    # come later), so every consumer sees the EFFECTIVE args and the config digest matches
    # between a fresh run and its own resume (codex r5 B3).
    # Resume restore runs BEFORE the provenance stamp so the config digest hashes the args
    # that will actually take effect (codex r4 #3: a digest over pre-restore defaults would
    # false-mismatch every restored key).
    if args.resume is not None and Path(args.resume).exists():
        _rargs = torch.load(args.resume, map_location="cpu",
                            weights_only=False).get("args", {})
        for _k, _default in (("model_variant", "level_a"), ("code_dim", None),
                             ("n_heads", None), ("d_ff", None), ("n_layers", None),
                             ("depth_double", 6), ("depth_single", 12),
                             ("hidden_size", None), ("mlp_ratio", 4.0),
                             ("dropout", None), ("max_T_lat", 75),
                             # Text-architecture flags (codex chain review #4).
                             # use_sentence_token has NO parameters of its own, so a
                             # strict state-dict load canNOT catch it silently off —
                             # only restoring it from the ckpt can.
                             ("text_dim", 768), ("text_input_norm", 0),
                             ("use_sentence_token", 0), ("text_slot_xattn", 0)):
        # --parameterization is deliberately NOT restored from the ckpt: restoring it
        # before the provenance stamp would make the contract digest always match, so a
        # watchdog relaunch of an x-run that LOST the flag would be silently accepted as
        # x (codex xpred r1 #2). Leaving the CLI sentinel untouched makes the digest
        # fail-closed instead: x-ckpt + missing/wrong flag -> contract mismatch ->
        # REFUSED before the model is built. Resuming an x-run therefore REQUIRES
        # --parameterization x on the CLI (the launcher's PARAMETERIZATION env).
            _v = _rargs.get(_k, _default) if isinstance(_rargs, dict) \
                else getattr(_rargs, _k, _default)
            if _v is not None:
                setattr(args, _k, _v)
        if is_main:
            print(f"resume: rebuilding model from ckpt args model_variant={args.model_variant} "
                  f"depth_double={args.depth_double} depth_single={args.depth_single}", flush=True)
    # Resolve dropout per variant (spec §5.4: graph_pscf=0.05, level_a=0.1). None =
    # not set on the CLI; an explicit --dropout wins, and resume above may have restored
    # the ckpt's dropout (non-None -> takes precedence). Resolving HERE — before
    # vars(args) is saved into the ckpt — records the real dropout so resume rebuilds it.
    if args.dropout is None:
        args.dropout = 0.05 if args.model_variant == "graph_pscf" else 0.1

    out_dir = Path(args.out)
    resume_in_place = (args.resume is not None
                       and out_dir.resolve() == Path(args.resume).resolve().parent)
    if (out_dir.exists() and any(out_dir.iterdir())
            and not args.overwrite and not resume_in_place):
        raise RuntimeError(f"[OUT FAIL] {out_dir} non-empty. Use --overwrite or "
                           f"--resume to continue in place.")
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "train.log"
    metrics_path = out_dir / "metrics.jsonl"

    def log(msg: str) -> None:
        if not is_main:
            return
        print(msg, flush=True)
        with open(log_path, "a") as f:
            f.write(msg + "\n")

    import subprocess
    try:
        git_sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT).decode().strip()
    except Exception:
        git_sha = "unknown"
    log(f"=== Graph-CodeFlow [{args.model_variant}] rectified flow over post-RVQ z_q ===")
    log(f"git_sha: {git_sha}  device: {dev}  world_size: {world_size}")
    log(f"args: {vars(args)}")

    # ---- Frozen tokenizer (for snapped-decode QA + projection) ----
    tokenizer, ta = load_frozen_tokenizer(args.frozen_vqvae_ckpt, dev)
    D = ta["d_model"]
    if D != args.code_dim:
        raise RuntimeError(f"[CFG FAIL] tokenizer code_dim {D} != --code_dim {args.code_dim}")
    log(f"frozen tokenizer: code_dim={D} Q={ta['num_quantizers']} K={ta['num_codes']} "
        f"max_coarse={ta['max_coarse']}")

    # ---- Data (offline token cache) ----
    # Resolved before the dataset is built: under the protocol each payload's topology must be
    # checked against the corpus that DEFINES it, not only against the payload's own hash.
    _tc_man = json.loads((Path(args.token_cache) / "manifest.json").read_text())
    _hroot = args.holdout_data_root or _tc_man.get("anytop_root")
    strict = args.protocol == "unseen_topology_v1"
    if strict and not args.holdout_artifact:
        raise SystemExit("--protocol unseen_topology_v1 requires --holdout_artifact")
    # The backbone reads a token cache, not the corpus, so the guard checks the cache's own
    # index.jsonl motion_ids. A cache exported from the full splits would put held topologies
    # here with nothing else to notice.
    ds_train = TokenCacheDataset(args.token_cache, "train",
                                 authority_root=_hroot if strict else None,
                                 caption_sidecar=args.caption_sidecar,
                                 caption_sampling=args.caption_sampling,
                                 caption_seed=args.seed,
                                 # sidecar full verification (31 GB hash + exhaustive npz
                                 # scan) on rank 0 only: 8 ranks doing it concurrently
                                 # starved the shared fs past the NCCL PG timeout (smoke
                                 # 2026-08-05 SIGABRT). Rank 0 raising kills the whole
                                 # torchrun job, so every rank still trains only on a
                                 # verified artifact; other ranks keep the size pins.
                                 caption_verify=is_main)

    _PROV = prov.stamp(protocol=args.protocol, stage="codeflow",
                       holdout_artifact=args.holdout_artifact if strict else None,
                       data_root=_tc_man.get("anytop_root"),
                       upstream={"token_cache": str(args.token_cache),
                                 "token_cache_provenance": prov.read(_tc_man),
                                 "frozen_vqvae_ckpt": _tc_man.get("frozen_vqvae_ckpt")},
                       extra={"training_config_sha256":
                              prov.codeflow_training_config_sha256(args, world_size)})
    # The backbone never touches the corpus, so the token cache IS its data. A cache exported
    # from a full-corpus tokenizer is not made clean by the retained index it happens to carry.
    prov.verify_upstream(prov.read(_tc_man), protocol=args.protocol,
                         what=f"token cache {args.token_cache}",
                         expect_artifact_body_sha=_PROV.get("holdout_artifact_body_sha256"), log=log)
    if args.resume is not None:
        _rck_prov = prov.read(torch.load(args.resume, map_location="cpu", weights_only=False))
        prov.verify_upstream(
            _rck_prov,
            protocol=args.protocol, what=f"resume checkpoint {args.resume}",
            expect_artifact_body_sha=_PROV.get("holdout_artifact_body_sha256"), log=log)
        # Hard resume contract (codex r4 #3): protocol, holdout identity AND the full
        # training-config digest (text flags included, by exclusion) must match the ckpt.
        prov.verify_resume_contract(_rck_prov, _PROV,
                                    what=f"resume checkpoint {args.resume}", log=log)
    if strict and args.gen_eval and args.evaluator_ckpt:
        # The instrument must not have been trained on the rigs it will score.
        prov.verify_upstream(
            prov.read(torch.load(args.evaluator_ckpt, map_location="cpu", weights_only=False)),
            protocol=args.protocol, what=f"online evaluator {args.evaluator_ckpt}",
            expect_artifact_body_sha=_PROV.get("holdout_artifact_body_sha256"), log=log)
    guard_dataset(ds_train, data_root=_hroot,
                  artifact=args.holdout_artifact if strict else None,
                  expect_body_sha=args.holdout_sha, stage="codeflow:train",
                  allow_no_holdout=(not strict) or args.allow_no_holdout, log=log)
    try:
        ds_val = TokenCacheDataset(args.token_cache, "val",
                                   authority_root=_hroot if strict else None)
        guard_dataset(ds_val, data_root=_hroot,
                      artifact=args.holdout_artifact if strict else None,
                      expect_body_sha=args.holdout_sha, stage="codeflow:val",
                      allow_no_holdout=(not strict) or args.allow_no_holdout, log=log)
    except FileNotFoundError:
        ds_val = None
    log(f"token cache: train={len(ds_train)}" + (f" val={len(ds_val)}" if ds_val else " (no val)"))
    if len(ds_train) < args.batch_size and not (args.smoke or args.mem_profile):
        raise RuntimeError(f"[DATA FAIL] train {len(ds_train)} < batch {args.batch_size}")
    # Preflight (P2): graph_pscf's frame_seed capacity is max_T_lat; the cache's T_lat
    # must not exceed it, else GraphPSCFFlowNet.forward raises mid-run. Catch it HERE
    # (right after cache load, before empirical stats / first forward) by reading the
    # first clip's z_q[T_lat,C,D].
    if args.model_variant == "graph_pscf" and len(ds_train) > 0:
        _cache_T_lat = int(ds_train[0]["z_q"].shape[0])
        if _cache_T_lat > args.max_T_lat:
            raise RuntimeError(
                f"[CFG FAIL] token cache T_lat={_cache_T_lat} > --max_T_lat={args.max_T_lat}: "
                f"graph_pscf frame_seed capacity too small. Re-export with smaller "
                f"num_frames, or raise --max_T_lat to >= {_cache_T_lat}.")
        log(f"preflight: cache T_lat={_cache_T_lat} <= max_T_lat={args.max_T_lat} OK")

    # ---- Resume: rebuild the model from the CKPT's saved architecture args ----
    # The model is constructed BEFORE the resume state_dict load below, so when
    # resuming we must build the SAME architecture the ckpt was trained with (its
    # model_variant + depths), not the CLI default — otherwise load_state_dict
    # (strict=True) would mismatch. Old ckpts (pre-model_variant) default to
    # level_a so they still load. Only architecture-defining args are overridden;
    # train/schedule args stay from the CLI.


    # ---- Model ----
    # spec A3: graph_pscf pins H==D==512 for v1 (no "H!=D is fine" speculation).
    if args.hidden_size != args.code_dim:
        raise RuntimeError(
            f"[CFG FAIL] spec A3 requires hidden_size==code_dim (H==D==512 for v1), "
            f"got hidden_size={args.hidden_size} code_dim={args.code_dim}")
    flow = GraphCodeFlow(
        code_dim=args.code_dim, n_heads=args.n_heads, d_ff=args.d_ff,
        n_layers=args.n_layers, d_text=args.text_dim, text_token_dim=args.text_dim,
        dropout=args.dropout,
        text_input_norm=bool(args.text_input_norm),
        use_sentence_token=bool(args.use_sentence_token),
        text_slot_xattn=bool(args.text_slot_xattn),
        model_variant=args.model_variant, depth_double=args.depth_double,
        depth_single=args.depth_single, mlp_ratio=args.mlp_ratio,
        max_T_lat=args.max_T_lat,
        parameterization=(args.parameterization or "v"),
    ).to(dev)
    log(f"flow parameterization: {args.parameterization or 'v'}"
        + (" (x0 prediction, v-space loss)" if (args.parameterization or "v") == "x" else ""))
    # Empirical z_q normalization (LOCKED): mean/std over valid train tokens.
    # Startup acceleration: disk-cache the stats (keyed by cache identity from the
    # manifest, so a re-export invalidates it) + rank-0 scan + broadcast under DDP.
    _cache_path = None
    _cache_key = None
    try:
        import hashlib
        _tc = Path(args.token_cache)
        _man_text = (_tc / "manifest.json").read_text()
        _idx_bytes = (_tc / "train" / "index.jsonl").read_bytes()
        # Strong invalidation: full-manifest md5 (any export-config change) + a full
        # CONTENT md5 of train/index.jsonl (any clip-set/order/content change, even at
        # identical byte-size) + n/D/max_clips. A re-export rewrites manifest/index ->
        # key changes -> cache auto-invalidates. (md5 of ~24MB index is one-time
        # rank-0-only ~tens of ms, negligible vs the scan it guards.)
        _cache_key = {"manifest_md5": hashlib.md5(_man_text.encode()).hexdigest(),
                      "index_md5": hashlib.md5(_idx_bytes).hexdigest(),
                      "n": len(ds_train), "D": D,
                      "max_clips": args.empirical_stats_max_clips}
        _cache_path = str(_tc / "empirical_stats.pt")
    except Exception:
        _cache_path = None  # no manifest/index -> skip cache, fall back to full scan
    e_mean, e_std, n_stat = compute_empirical_stats(
        ds_train, D, dev, max_clips=args.empirical_stats_max_clips,
        cache_path=_cache_path, cache_key=_cache_key, is_ddp=is_ddp, rank=rank)
    flow.set_latent_stats(e_mean, e_std)
    log(f"empirical z_q norm over {n_stat} valid tokens: "
        f"mean|.|avg={e_mean.abs().mean().item():.4f} std.avg={e_std.mean().item():.4f}")
    n_params = sum(pp.numel() for pp in flow.parameters() if pp.requires_grad)
    log(f"GraphCodeFlow trainable params: {n_params:,}")

    amp_enabled = (args.amp_dtype == "bf16") and dev.type == "cuda"
    fwd_dtype = torch.float32  # conditioning/z_q dtype the model validates on the fp32 path
    amp_ctx = ((lambda: torch.autocast(device_type="cuda", dtype=torch.bfloat16))
               if amp_enabled else contextlib.nullcontext)
    log(f"AMP: amp_dtype={args.amp_dtype} (autocast around fp32 flow math)")

    # ---- Mem profile (one fwd+bwd, report peak, exit; NOT a real launch) ----
    if args.mem_profile:
        bs = min(args.batch_size, len(ds_train))
        dl = DataLoader(ds_train, batch_size=bs, shuffle=True,
                        collate_fn=token_collate, num_workers=0, drop_last=True)
        b = next(iter(dl))
        b = {k: v.to(dev) if torch.is_tensor(v) else v for k, v in b.items()}
        cond = build_cond(b, args.cond_drop_prob, training=True, dtype=fwd_dtype)
        torch.cuda.reset_peak_memory_stats(dev)
        with amp_ctx():
            r = flow.flow_loss(b["z_q"].to(fwd_dtype), b["token_mask"], cond,
                               validate_inputs=True)
        loss = args.flow_loss_weight * r["flow_loss"]
        loss.backward()
        peak = torch.cuda.max_memory_allocated(dev) / 1e9
        log(f"[MEM PROFILE] batch_size={bs} flow_loss={r['flow_loss'].item():.4f} "
            f"peak_cuda_mem={peak:.2f} GB  (NO real run launched)")
        if is_ddp:
            dist.destroy_process_group()
        return 0

    # fail-loud (mirrors the VQVAE guard): a factor>1 paired with a negative start_epoch would
    # SILENTLY never activate (_current_factor stays 1.0). Abort rather than train without the
    # intended curriculum. A phase must be BOTH factor>1 AND start_epoch>=0 to be "on".
    if args.human_upsample_factor > 1.0 and args.human_upsample_start_epoch < 0:
        raise SystemExit("[human-upsample] --human_upsample_factor>1 requires --human_upsample_start_epoch>=0 (else silent no-op)")
    if args.human_upsample_phase2_factor > 1.0 and args.human_upsample_phase2_start_epoch < 0:
        raise SystemExit("[human-upsample] --human_upsample_phase2_factor>1 requires --human_upsample_phase2_start_epoch>=0 (else silent no-op)")
    _p1_on = args.human_upsample_factor > 1.0 and args.human_upsample_start_epoch >= 0
    _p2_on = args.human_upsample_phase2_factor > 1.0 and args.human_upsample_phase2_start_epoch >= 0
    if _p1_on or _p2_on:                     # gate matches when _current_factor() can ever be >1
        _is_human = [str(r.get("motion_id", "")).upper().startswith("HML") for r in ds_train.rows]
        _nh, _nt = sum(_is_human), len(_is_human)
        def _share(f): return f * _nh / max(1, f * _nh + (_nt - _nh))
        _p1s = (f"phase1 factor={args.human_upsample_factor} start={args.human_upsample_start_epoch} "
                f"(~{100*_share(args.human_upsample_factor):.0f}%)") if _p1_on else "phase1 OFF"
        _p2s = (f"phase2 factor={args.human_upsample_phase2_factor} start={args.human_upsample_phase2_start_epoch} "
                f"(~{100*_share(args.human_upsample_phase2_factor):.0f}%)") if _p2_on else "phase2 OFF"
        log(f"[human-upsample] curriculum ON: {_p1s} -> {_p2s}; "
            f"human={_nh}/{_nt} ({100*_nh/max(1,_nt):.1f}% base)")
        train_sampler = HumanCurriculumSampler(
            len(ds_train), _is_human, args.human_upsample_factor, args.human_upsample_start_epoch,
            num_replicas=(world_size if is_ddp else 1), rank=(rank if is_ddp else 0), seed=args.seed,
            phase2_factor=args.human_upsample_phase2_factor, phase2_start_epoch=args.human_upsample_phase2_start_epoch)
    else:
        train_sampler = (DistributedSampler(ds_train, shuffle=True, drop_last=True)
                         if is_ddp else None)
    nw = max(0, args.num_workers)
    dl_train = DataLoader(
        ds_train, batch_size=args.batch_size, shuffle=(train_sampler is None),
        sampler=train_sampler, collate_fn=token_collate, num_workers=nw,
        drop_last=True, pin_memory=True,
        # persistent workers NEVER see set_caption_epoch (parent-only state; codex
        # caption-sampling review #1 reproduced the freeze) — random mode re-forks
        # workers each epoch so they inherit the updated epoch. Worker spawn is
        # seconds against a ~29-minute epoch. Fixed mode keeps the old behaviour.
        persistent_workers=(nw > 0 and args.caption_sampling != "random"),
        prefetch_factor=(4 if nw > 0 else None))
    dl_val = (DataLoader(ds_val, batch_size=args.batch_size, shuffle=False,
                         collate_fn=token_collate, num_workers=max(1, nw // 2),
                         drop_last=False, pin_memory=True)
              if ds_val is not None else None)

    if is_ddp:
        flow = DDP(flow, device_ids=[local_rank], find_unused_parameters=False)
    raw_flow = flow.module if is_ddp else flow

    # Resolve the decoded-geometry knobs from their None sentinels (unset -> off /
    # v4b-validated defaults). Done ONCE here so the hot loop reads plain floats.
    _dec_world_w = float(args.w_dec_world or 0.0)
    _dec_traj_w = float(args.w_dec_traj or 0.0)
    _dec_speed_w = float(args.w_dec_speed or 0.0)
    _dec_t_min = 0.6 if args.dec_geom_t_min is None else float(args.dec_geom_t_min)
    _dec_every = max(1, int(args.dec_geom_every or 1))
    _dec_speed_floor = 1e-4 if args.dec_speed_floor is None else float(args.dec_speed_floor)
    _dec_speed_mode = args.dec_speed_loss or "log_huber"
    if _dec_speed_w > 0 and _dec_speed_floor <= 0 and _dec_speed_mode != "raw_l1":
        raise RuntimeError(
            f"[CFG FAIL] --dec_speed_floor must be > 0 when the log-space speed loss is "
            f"active (got {_dec_speed_floor}): log(speed->0) produces non-finite loss")
    if (_dec_world_w + _dec_traj_w + _dec_speed_w) > 0:
        log(f"decoded-geometry loss ON: world={_dec_world_w} traj={_dec_traj_w} "
            f"speed={_dec_speed_w}({_dec_speed_mode}) t_min={_dec_t_min} "
            f"every={_dec_every} floor={_dec_speed_floor} (target=no-grad decode(z_q))")

    opt = torch.optim.AdamW(flow.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    # n_iter counts OPTIMIZER steps (one per grad_accum micro-batches), so total_steps for the
    # LR schedule must be in optimizer steps too — else grad_accum>1 stretches the cosine decay
    # and lr_at(n_iter) diverges from the accum=1 run on resume (codex catch).
    _accum = max(1, int(args.grad_accum))
    steps_per_epoch = max(1, len(dl_train) // _accum)
    total_steps = max(1, args.epochs * steps_per_epoch)

    def lr_at(step: int) -> float:
        """Linear warmup -> half-cosine decay to eta_min_ratio*lr (CodeFlow recipe)."""
        if args.warmup_steps > 0 and step < args.warmup_steps:
            return args.lr * (step + 1) / args.warmup_steps
        if args.lr_scheduler == "none":
            return args.lr
        prog = (step - args.warmup_steps) / max(1, total_steps - args.warmup_steps)
        prog = min(1.0, max(0.0, prog))
        cos = 0.5 * (1.0 + math.cos(math.pi * prog))
        return args.lr * (args.eta_min_ratio + (1.0 - args.eta_min_ratio) * cos)

    n_iter, start_epoch, best_val = 0, 0, float("inf")
    if args.resume is not None:
        if not Path(args.resume).exists():
            raise RuntimeError(f"[RESUME FAIL] {args.resume} missing.")
        rc = torch.load(args.resume, map_location=dev, weights_only=False)
        raw_flow.load_state_dict(rc["model_state_dict"], strict=True)
        opt.load_state_dict(rc["optimizer_state_dict"])
        for st in opt.state.values():
            for k, v in st.items():
                if torch.is_tensor(v):
                    st[k] = v.to(dev)
        start_epoch = int(rc["epoch"]) + 1
        n_iter = int(rc["global_step"])
        best_val = float(rc.get("best_val", float("inf")))
        log(f"resumed: start_epoch={start_epoch} n_iter={n_iter} best_val={best_val:.4f}")
        del rc

    # Resume-aware smoke bound (codex v2b r2 MAJOR-1): a bare `2` makes
    # range(start_epoch, 2) EMPTY whenever we smoke a resumed run (start_epoch=290 for the
    # v2->v2b continuation), so the smoke exits 0 having executed zero iterations — a
    # false PASS that verifies neither forward/backward, grad accumulation, nor NCCL.
    # Anchor the bound to where training actually starts, and never exceed the schedule.
    n_epochs = min(args.epochs, start_epoch + 2) if args.smoke else args.epochs
    smoke_cap = args.smoke_iters if args.smoke else None

    # ---- ONLINE gen-eval setup: load the FROZEN evaluator + T5 + eval dataset ONCE,
    # rank-0 only (opt-in via --gen_eval). Kept in gen_eval_ctx; None on other ranks /
    # when disabled, so the per-epoch hook below is naturally skipped there. ----
    gen_eval_ctx = None
    if args.gen_eval and is_main:
        # Whole setup is rank-0-only + guarded: a setup failure must DISABLE gen-eval and let
        # rank-0 fall through to training (so it can't desync the other ranks at the first DDP
        # collective). RNG save/restore wraps it because constructing the evaluator/T5 draws
        # from the global RNG before load_state_dict overwrites it (keeps training RNG identical
        # to a --gen_eval-off run).
        _rng_cpu = torch.get_rng_state()
        _rng_cuda = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        try:
            if not args.evaluator_ckpt:
                raise RuntimeError("--gen_eval requires --evaluator_ckpt")
            from src.data.anytop_t2m_eval_dataset import AnyTopT2MEvalDataset
            from src.models.graph_salad.t2m_evaluator import AnyTopT2MEvaluator
            from src.eval.codeflow_gen_eval import run_gen_eval as _run_gen_eval
            from transformers import T5EncoderModel, T5TokenizerFast

            def _resolve(pth):                                   # /scratch vs /iridisfs/scratch -> same tree
                try:
                    return str(Path(pth).resolve())
                except Exception:
                    return str(pth)

            _eck = torch.load(args.evaluator_ckpt, map_location="cpu")
            _ea = _eck["args"]
            _g = (lambda k, d: _ea.get(k, d)) if isinstance(_ea, dict) else (lambda k, d: getattr(_ea, k, d))
            _ev_nf, _ev_mj = int(_g("num_frames", 300)), int(_g("max_joints", 144))
            _ev_root = args.gen_eval_data_root or _g("data_root", None)
            if not _ev_root:
                raise RuntimeError("no data_root (pass --gen_eval_data_root or use an evaluator ckpt carrying data_root)")
            _vq_root = ta.get("anytop_root") or ta.get("data_root")
            if _vq_root and _resolve(_ev_root) != _resolve(_vq_root):
                raise RuntimeError(f"evaluator data_root {_ev_root} != tokenizer root {_vq_root} (eval-space would be invalid)")
            if int(args.text_dim) != 768 and not args.gen_eval_manifest:
                # The legacy default manifest embeds caption strings from the ORIGINAL
                # corpus; 18 of its primaries were removed in clean_v1 and the lookup
                # encoder hard-fails on them at the first scheduled eval (codex r3 #1).
                # Requiring the explicit clean manifest here fails at SETUP, not at
                # epoch N.
                raise RuntimeError(
                    "--text_dim != 768 requires --gen_eval_manifest (pass "
                    "eval_splits/val_all_clean_v1.json — the legacy default embeds "
                    "removed caption strings)")
            _manifest = args.gen_eval_manifest or str(Path(_ev_root) / "eval_splits" / "val_all.json")
            _core = AnyTopT2MEvaluator(
                coemb_dim=_g("coemb_dim", 512), text_tower=_g("text_tower", "distilbert"),
                distilbert_path=_g("distilbert_path", "checkpoints/text_encoders/distilbert-base-uncased"),
                text_max_length=_g("text_max_length", 64), n_heads=_g("n_heads", 8), d_ff=_g("d_ff", 2048),
                n_graph_layers=_g("n_graph_layers", 6), n_temporal_layers=_g("n_temporal_layers", 4),
                motion_feat_dim=_g("motion_feat_dim", 13), dropout=_g("dropout", 0.1),
                learnable_temperature=not _g("fixed_temperature", False), temperature=_g("temperature", 0.07))
            _miss, _unexp = _core.load_state_dict(_eck["model"], strict=False)
            _bad = [k for k in _miss if not k.startswith("text_distilbert.text_model.")]
            if _bad or _unexp:
                raise RuntimeError(f"evaluator load mismatch: missing={_bad[:6]} unexpected={list(_unexp)[:6]}")
            _core.to(dev).eval()
            if int(args.text_dim) == 768:
                _t5tok = T5TokenizerFast.from_pretrained("t5-base")
                _t5 = T5EncoderModel.from_pretrained("t5-base").to(dev).eval()

                @torch.no_grad()
                def _t5_encode_batch(texts):
                    enc = _t5tok(list(texts), return_tensors="pt", padding="max_length", truncation=True, max_length=64).to(dev)
                    hs = _t5(input_ids=enc.input_ids, attention_mask=enc.attention_mask).last_hidden_state
                    m = enc.attention_mask.bool()
                    gl = (hs * m.unsqueeze(-1).float()).sum(1) / m.sum(1, keepdim=True).clamp_min(1)
                    return gl.float(), hs.float(), m
            else:
                # LLM2Vec run: the text encoder is an 8B model and does NOT run on the
                # training GPUs. Every val caption is in the ragged sidecar the token
                # cache was built from, so "encoding" is a lookup — and byte-identical
                # to what training consumed. An unknown caption is a hard error: it
                # would mean the eval set and the cache disagree, and silently
                # re-encoding it differently is exactly the drift this forbids.
                if not args.gen_eval_caption_cache:
                    raise RuntimeError(
                        "--text_dim != 768 requires --gen_eval_caption_cache "
                        "(the LLM2Vec sidecar prefix; online gen-eval looks captions up "
                        "instead of running the 8B encoder on the training GPUs)")
                from src.eval.codeflow_gen_eval import make_caption_lookup_encoder
                _t5_encode_batch = make_caption_lookup_encoder(
                    args.gen_eval_caption_cache, _ev_root, int(args.text_dim), dev)

            _ds = AnyTopT2MEvalDataset(manifest_path=_manifest, data_root=_ev_root, caption_emb_cache=None,
                                       split="val", view="full", num_frames=_ev_nf, max_joints=_ev_mj)
            _nt = len(_ds)
            _ntake = _nt if (args.gen_eval_n <= 0 or args.gen_eval_n >= _nt) else args.gen_eval_n
            _idxs = list(range(_nt))[::max(1, _nt // _ntake)][:_ntake]
            gen_eval_ctx = {"run": _run_gen_eval, "core": _core, "t5_encode_batch": _t5_encode_batch,
                            "ds": _ds, "idxs": _idxs, "num_frames": _ev_nf, "stride": int(ta["temporal_stride"])}
            log(f"[gen-eval] online armed: evaluator ep={_eck.get('epoch')} root={_ev_root} "
                f"n={len(_idxs)}/{_nt} every {args.gen_eval_every}ep steps={args.gen_eval_steps} cfg={args.eval_cond_scale}")
        except Exception as _e:
            import traceback
            # LLM2Vec runs (text_dim != 768): a failed setup is a PROVENANCE failure
            # (bad sidecar, corpus drift, wrong prefix) — silently training without
            # eval is exactly the drift-hiding this path forbids (codex re-review #3).
            if int(args.text_dim) != 768:
                raise
            gen_eval_ctx = None
            log(f"[gen-eval] DISABLED — setup failed (training continues WITHOUT online eval): "
                f"{_e}\n{traceback.format_exc()}")
        finally:
            torch.set_rng_state(_rng_cpu)
            if _rng_cuda is not None:
                torch.cuda.set_rng_state_all(_rng_cuda)

    for epoch in range(start_epoch, n_epochs):
        if train_sampler is not None and hasattr(train_sampler, "set_epoch"):
            train_sampler.set_epoch(epoch)   # DistributedSampler + HumanCurriculumSampler both need this
        ds_train.set_caption_epoch(epoch)    # per-epoch caption re-draw (no-op in fixed mode)
        flow.train()
        t0 = time.time()
        run_sum, run_cnt = 0.0, 0
        accum = max(1, int(args.grad_accum))   # micro-steps per optimizer step (1 = unchanged)
        opt.zero_grad()                        # clean slate; leftover partial cycle at epoch end is discarded here next epoch
        micro = 0
        for it, b in enumerate(dl_train):
            if smoke_cap is not None and it >= smoke_cap:
                break
            b = {k: v.to(dev) if torch.is_tensor(v) else v for k, v in b.items()}
            cond = build_cond(b, args.cond_drop_prob, training=True, dtype=fwd_dtype)
            # DDP micro-batch sync control. PyTorch's no_sync() contract requires the
            # FORWARD inside the context — the reducer latches the flag at forward
            # time, so a backward-only wrapper still all-reduces every micro (codex
            # r3 comm-hook probe: 2 reductions for 2 micros). Setting
            # require_backward_grad_sync directly (exactly what no_sync() does)
            # covers forward+backward with no restructuring: False on non-final
            # micros accumulates grads locally; True on the final micro all-reduces
            # the accumulated total once per optimizer step.
            _sync_micro = ((micro + 1) % accum == 0)
            if is_ddp:
                flow.require_backward_grad_sync = _sync_micro
            with amp_ctx():
                r = flow(b["z_q"].to(fwd_dtype), b["token_mask"], cond,
                         validate_inputs=(it == 0 and epoch == start_epoch))
            loss = args.flow_loss_weight * r["flow_loss"]

            # ---- ARDY-style decoded-geometry loss (L_dec; v4b-validated recipe) ----
            # Decode the gradient-carrying clean prediction through the FROZEN
            # tokenizer in fp32 with autocast DISABLED (decoder Jacobian + cumsum
            # world recovery are precision-sensitive; landed pattern from
            # train_denoiser.py:976). Target = no-grad decode(GT z_q). Gated to
            # near-clean rows (t > t_min) and every-N steps for memory/time.
            dec_w = _dec_world_w + _dec_traj_w + _dec_speed_w
            loss_dec_world = loss_dec_traj = loss_dec_speed = None
            # Sync-micro-only decode: with grad accumulation the decoded loss runs on
            # the FINAL micro-batch of each optimizer step only (halves decode compute
            # at accum=2), with its weights scaled by `accum` below to cancel the
            # uniform /accum — expected gradient contribution per optimizer step is
            # unchanged (one full-weight dec term on a B*1 sample instead of the
            # B*accum average; slightly noisier, same scale). _sync_micro is set
            # above, before the DDP forward.
            if dec_w > 0 and (n_iter % _dec_every == 0) and _sync_micro:
                _t_rows = r["timesteps"]                       # [B] fp32
                _gate = _t_rows > _dec_t_min
                if bool(_gate.any()):
                    _idx = _gate.nonzero(as_tuple=True)[0]
                    _clean = r["clean_pred_norm"].float()[_idx]        # grad-carrying
                    _tm = b["token_mask"][_idx]
                    _z_hat = (raw_flow.denormalize(_clean)
                              * _tm.unsqueeze(-1).float())
                    _skel = {"s_j": b["s_j"][_idx], "assignment": b["assignment"][_idx],
                             "coarse_mask": b["coarse_mask"][_idx],
                             "frame_mask_lat": b["frame_mask_lat"][_idx],
                             "pooled_adjacency": b["pooled_adjacency"][_idx],
                             "pooled_geodesic": b["pooled_geodesic"][_idx]}
                    _fake_b = SimpleNamespace(joint_mask=b["joint_mask"][_idx])
                    with torch.autocast(device_type="cuda", enabled=False):
                        # Target FIRST (no-grad, freed of graph) so its transient
                        # activations don't overlap the prediction's retained
                        # autograd graph (codex decloss r1 #1 memory ordering).
                        with torch.no_grad():
                            _dt = tokenizer.decode(b["z_q"][_idx].float(), _skel, _fake_b)
                        _dp = tokenizer.decode(_z_hat, _skel, _fake_b)
                    _pred_m = _dp["pred_motion"].float()
                    _tgt_m = _dt["pred_motion"].float()
                    _fmask = _dp["frame_mask_recovered"].bool()
                    if _dec_world_w > 0 or _dec_traj_w > 0:
                        _terms = compute_world_geometry_terms(
                            pred_motion=_pred_m, gt_motion=_tgt_m,
                            anytop_mean=b["anytop_mean"][_idx],
                            anytop_std=b["anytop_std"][_idx],
                            joint_mask=b["joint_mask"][_idx], frame_mask=_fmask)
                        if _dec_world_w > 0:
                            loss_dec_world = _terms["world"]
                            loss = loss + (accum * _dec_world_w) * loss_dec_world
                        if _dec_traj_w > 0:
                            loss_dec_traj = _terms["traj"]
                            loss = loss + (accum * _dec_traj_w) * loss_dec_traj
                    if _dec_speed_w > 0:
                        loss_dec_speed = decoded_speed_loss(
                            _pred_m, _tgt_m, b["anytop_mean"][_idx],
                            b["anytop_std"][_idx], b["joint_mask"][_idx], _fmask,
                            _dec_speed_floor, _dec_speed_mode)
                        loss = loss + (accum * _dec_speed_w) * loss_dec_speed

            if it == 0 and epoch == start_epoch:
                B, T_lat, C, Dd = b["z_q"].shape
                Qd = b["indices"].shape[-1]
                log(f"  [gate ok] z_q=[{B},{T_lat},{C},{Dd}] indices Q={Qd} "
                    f"flow_loss={r['flow_loss'].item():.4f}")
            if not torch.isfinite(loss):
                log(f"[GATE FAIL] loss non-finite at iter {n_iter}")
                return 1
            # scale by 1/accum so accumulated grads = mean over the full global batch
            # (=per_gpu*world*accum); DDP averages across ranks each backward, summing to
            # the correct global-batch gradient after `accum` micro-steps.
            (loss / accum).backward()
            run_sum += float(r["flow_loss"].detach()); run_cnt += 1
            micro += 1
            if micro % accum != 0:
                continue                       # keep accumulating; step only every `accum` micro-batches
            grad_norm = torch.nn.utils.clip_grad_norm_(flow.parameters(), args.grad_clip)
            if not torch.isfinite(grad_norm):
                log(f"[GATE FAIL] non-finite grad norm at iter {n_iter}")
                return 1
            cur_lr = lr_at(n_iter)
            for pg in opt.param_groups:
                pg["lr"] = cur_lr
            opt.step()
            opt.zero_grad()

            do_log = (n_iter % args.log_every == 0) or (it == 0 and epoch == start_epoch)
            do_qa = (n_iter % args.qa_every == 0) or (args.smoke and it == 0)
            if do_log or do_qa:
                qa = None
                if do_qa and is_main:
                    qa = projection_qa(raw_flow, tokenizer, b,
                                       build_cond(b, 0.0, training=False, dtype=fwd_dtype),
                                       dev, decode=do_qa)
                if do_log:
                    _dec_str = ""
                    if (loss_dec_world is not None or loss_dec_traj is not None
                            or loss_dec_speed is not None):
                        _dec_str = (" | dec"
                                    + (f" world={loss_dec_world.item():.4f}" if loss_dec_world is not None else "")
                                    + (f" traj={loss_dec_traj.item():.4f}" if loss_dec_traj is not None else "")
                                    + (f" speed={loss_dec_speed.item():.4f}" if loss_dec_speed is not None else ""))
                    log(f"[ep{epoch} it{it} n_iter={n_iter}] flow_loss={r['flow_loss'].item():.5f} "
                        f"grad_norm={grad_norm.item():.3f} lr={cur_lr:.3e}" + _dec_str
                        + (f" | proj_err={qa['projection_error']:.4f} "
                           f"code_usage/q={qa['code_usage_per_q']}" if qa else ""))
                    if qa and "decode_cont_vs_snap_maxabs" in qa:
                        log(f"           [QA decode] cont_finite={qa['decode_cont_finite']} "
                            f"snap_finite={qa['decode_snap_finite']} "
                            f"cont_vs_snap_maxabs={qa['decode_cont_vs_snap_maxabs']:.4f}")
                    if is_main:
                        row = {"epoch": epoch, "iter": it, "n_iter": n_iter,
                               "flow_loss": r["flow_loss"].item(),
                               "grad_norm": grad_norm.item(), "lr": cur_lr}
                        if qa:
                            row.update(qa)
                        with open(metrics_path, "a") as f:
                            f.write(json.dumps(row) + "\n")
            n_iter += 1

        train_flow = run_sum / max(1, run_cnt)
        log(f"=== epoch {epoch} done in {time.time() - t0:.1f}s | train_flow={train_flow:.5f} ===")

        do_val = ((epoch + 1) % args.save_every == 0 or epoch == n_epochs - 1 or args.smoke)
        if do_val and is_main and dl_val is not None:
            raw_flow.eval()
            vlosses, vproj = [], []
            with torch.no_grad():
                for vit, vb in enumerate(dl_val):
                    if args.smoke and vit >= 2:
                        break
                    vb = {k: v.to(dev) if torch.is_tensor(v) else v for k, v in vb.items()}
                    vcond = build_cond(vb, 0.0, training=False, dtype=fwd_dtype)
                    with amp_ctx():
                        vr = raw_flow.flow_loss(vb["z_q"].to(fwd_dtype), vb["token_mask"], vcond)
                    vlosses.append(vr["flow_loss"].item())
                    vproj.append(projection_qa(raw_flow, tokenizer, vb, vcond, dev,
                                               decode=False)["projection_error"])
            val_flow = float(np.mean(vlosses)) if vlosses else float("nan")
            val_proj = float(np.mean(vproj)) if vproj else float("nan")
            log(f"  [val] flow_loss={val_flow:.5f} projection_error={val_proj:.4f}")
            if not args.smoke:
                hist_best = min(best_val, val_flow)
                ckpt = {prov.KEY: _PROV,
                        "model_state_dict": raw_flow.state_dict(),
                        "optimizer_state_dict": opt.state_dict(),
                        "epoch": epoch, "global_step": n_iter, "args": vars(args),
                        "val_flow": val_flow, "val_proj": val_proj, "best_val": hist_best,
                        "git_sha": git_sha, "frozen_vqvae_ckpt": args.frozen_vqvae_ckpt,
                        "latent_mean": raw_flow.latent_mean.cpu(),
                        "latent_std": raw_flow.latent_std.cpu()}
                # Atomic save: write to .tmp then os.replace so an alloc death mid-save
                # never leaves a truncated last_model.pt that the watchdog would resume into.
                _tmp = out_dir / "last_model.pt.tmp"
                torch.save(ckpt, _tmp)
                os.replace(_tmp, out_dir / "last_model.pt")
                if val_flow < best_val:
                    best_val = val_flow
                    _tmpb = out_dir / "best_model.pt.tmp"
                    torch.save(ckpt, _tmpb)
                    os.replace(_tmpb, out_dir / "best_model.pt")
                    log(f"  [ckpt] new best val_flow={val_flow:.5f}")
            raw_flow.train()
        # ---- ONLINE gen-eval (rank-0 only via gen_eval_ctx; sparse; NEVER aborts training) ----
        if gen_eval_ctx is not None and (epoch + 1) % args.gen_eval_every == 0:
            raw_flow.eval()
            try:
                _rep = gen_eval_ctx["run"](
                    flow=raw_flow, tokenizer=tokenizer, core=gen_eval_ctx["core"],
                    t5_encode_batch=gen_eval_ctx["t5_encode_batch"], ds=gen_eval_ctx["ds"],
                    idxs=gen_eval_ctx["idxs"], dev=dev, stride=gen_eval_ctx["stride"],
                    pool=32, steps=args.gen_eval_steps, cfg_scale=args.eval_cond_scale,
                    num_frames=gen_eval_ctx["num_frames"], gen_batch=args.gen_eval_batch, seed=args.seed, log=log)
                _o = _rep["overall"]; _rr = _o.get("rprec_text_to_gen") or {}
                _msg = (f"  [gen-eval] ep{epoch} overall R@1={_rr.get(1)} R@2={_rr.get(2)} "
                        f"R@3={_rr.get(3)} match={_o.get('matching_mean'):.3f}")
                for _s, _m in _rep.get("per_subset", {}).items():
                    _r2 = _m.get("rprec_text_to_gen") or {}
                    _msg += f" | {_s} R@1={_r2.get(1)}(n={_m.get('n')})"
                log(_msg)
                with open(metrics_path, "a") as f:
                    f.write(json.dumps({"epoch": epoch, "n_iter": n_iter, "gen_eval": _rep}) + "\n")
            except KeyError:
                # A caption missing from the sidecar mid-run = corpus drift under a
                # live training. Not a transient eval hiccup; stop rather than hide it.
                raise
            except Exception as _e:
                import traceback
                log(f"  [gen-eval] FAILED ep{epoch} (skipped, training continues): {_e}\n{traceback.format_exc()}")
            raw_flow.train()
        if is_ddp:
            dist.barrier()

    log("=== training loop complete ===")
    if is_ddp:
        dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
