"""Train the RawJointFlow ablation baseline.

Answers: does the Stage-1 tokenizer (graph pooling + shared RVQ codebook) buy anything,
or could the same text-conditioned rectified flow have run directly on the raw padded
per-joint tensor? Everything except the modelled space is held identical to the main
backbone: objective (rectified flow, masked MSE), text conditioning (dual_text, has_text
gated for CFG), graph conditioning, curriculum, optimizer and schedule.

Deliberately a SEPARATE script rather than a flag on train_graph_codeflow.py: that trainer
produced the released flagship run and is reproduced verbatim in the release repo, so it is
not modified here.

Single node:
  torchrun --nproc_per_node=2 scripts/train_raw_joint_baseline.py \
      --anytop_root data/animo4d_L4TB_plus_human_v4b272neutral \
      --caption_emb_cache   data/anytop_caption_t5_v4b272neutral_multi.npz \
      --caption_token_cache data/anytop_caption_t5_v4b272neutral_multi \
      --out runs/baseline_rawjoint --epochs 120 --batch_size 1 --grad_accum 8

Cross-allocation (same node, k allocations): pass --nnodes/--node_rank/--master_addr to
torchrun exactly as for the main run; nothing here assumes a single allocation.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.anytop_dataset import AnyTopDataset, collate_fn as anytop_collate_fn  # noqa: E402
from src.data.human_curriculum_sampler import HumanCurriculumSampler  # noqa: E402
from src.models.graph_salad.batch import GraphMotionBatch  # noqa: E402
from src.models.CodeFlow_Model.raw_joint_flow import RawJointFlow  # noqa: E402


def log(msg: str, rank: int = 0) -> None:
    if rank == 0:
        print(msg, flush=True)


def build_cond(batch, dev) -> dict:
    """Map a collated AnyTop batch onto the cond dict RawJointFlow expects."""
    return {
        "text_global": batch.caption_emb.to(dev),                      # [B,768]
        "text_tokens": batch.caption_token_emb.to(dev),                   # [B,L,768]
        "text_token_mask": batch.caption_token_mask.to(dev),           # [B,L] bool
        "has_text": batch.has_text.to(dev),                            # [B] bool
        "adjacency": batch.adjacency.to(dev),                          # [B,J,J]
        "geodesic_dist": batch.geodesic_dist.to(dev),                  # [B,J,J]
        "skeleton_features": batch.skeleton_features.to(dev),          # [B,J,9]
        "joint_mask": batch.joint_mask.to(dev),                        # [B,J] bool
        "frame_mask": batch.frame_mask.to(dev),                        # [B,T] bool
    }


def motion_and_mask(batch, dev):
    """anytop_x [B,J,13,T] -> x [B,T,J,13]; token_mask [B,T,J] = frame AND joint."""
    x = batch.anytop_x.to(dev).permute(0, 3, 1, 2).contiguous()        # [B,T,J,13]
    jm = batch.joint_mask.to(dev)                                      # [B,J]
    fm = batch.frame_mask.to(dev)                                      # [B,T]
    token_mask = fm.unsqueeze(-1) & jm.unsqueeze(1)                    # [B,T,J]
    return x, token_mask


def lr_at(step: int, args) -> float:
    """Half-cosine with linear warmup — identical schedule to the main run."""
    if step < args.warmup_steps:
        return args.lr * (step + 1) / max(1, args.warmup_steps)
    prog = (step - args.warmup_steps) / max(1, args.total_steps - args.warmup_steps)
    prog = min(1.0, max(0.0, prog))
    eta_min = args.lr * args.eta_min_ratio
    return eta_min + 0.5 * (args.lr - eta_min) * (1.0 + math.cos(math.pi * prog))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--anytop_root", required=True)
    ap.add_argument("--caption_emb_cache", required=True)
    ap.add_argument("--caption_token_cache", required=True)
    ap.add_argument("--out", required=True)
    # model — default is the largest raw-joint config that fits at T=300 on an 80GB GPU
    ap.add_argument("--d_model", type=int, default=512)
    ap.add_argument("--n_layers", type=int, default=13, help="must be odd (SALAD skip-transformer)")
    ap.add_argument("--n_heads", type=int, default=8)
    ap.add_argument("--d_ff", type=int, default=2048)
    # data
    ap.add_argument("--num_frames", type=int, default=300)
    ap.add_argument("--max_joints", type=int, default=144)
    # optimization — mirrors the main run
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--grad_accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=8e-5)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--warmup_steps", type=int, default=2000)
    ap.add_argument("--eta_min_ratio", type=float, default=0.01)
    ap.add_argument("--cond_drop_prob", type=float, default=0.1)
    ap.add_argument("--amp_dtype", default="bf16", choices=["bf16", "fp32"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--num_workers", type=int, default=4)
    # human up-sampling curriculum — same two phases as the main run
    ap.add_argument("--human_upsample_factor", type=float, default=3.0)
    ap.add_argument("--human_upsample_start_epoch", type=int, default=0)
    ap.add_argument("--human_upsample_phase2_factor", type=float, default=4.5)
    ap.add_argument("--human_upsample_phase2_start_epoch", type=int, default=50)
    # bookkeeping
    ap.add_argument("--log_every", type=int, default=50)
    ap.add_argument("--save_every", type=int, default=5)
    ap.add_argument("--resume", default=None)
    ap.add_argument("--smoke", action="store_true", help="a few iters only, then exit 0")
    ap.add_argument("--smoke_iters", type=int, default=4)
    args = ap.parse_args()

    if args.n_layers % 2 == 0:
        raise SystemExit(f"[ARGS FAIL] --n_layers must be odd, got {args.n_layers}")

    ddp = int(os.environ.get("WORLD_SIZE", "1")) > 1
    if ddp:
        dist.init_process_group(backend="nccl")
        rank = dist.get_rank()
        world = dist.get_world_size()
        local = int(os.environ.get("LOCAL_RANK", "0"))
        torch.cuda.set_device(local)
        dev = torch.device("cuda", local)
    else:
        rank, world, local = 0, 1, 0
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    torch.manual_seed(args.seed + rank)
    np.random.seed(args.seed + rank)
    out_dir = Path(args.out)
    if rank == 0:
        out_dir.mkdir(parents=True, exist_ok=True)

    log(f"=== RawJointFlow baseline (raw per-joint rectified flow) ===", rank)
    log(f"world_size={world}  device={dev}", rank)
    log(f"args: {vars(args)}", rank)

    # ---------------------------------------------------------------- data
    ds_kw = dict(data_root=args.anytop_root, num_frames=args.num_frames,
                 max_joints=args.max_joints, load_captions=True,
                 caption_emb_cache=args.caption_emb_cache,
                 caption_token_cache=args.caption_token_cache,
                 return_caption_tokens=True)
    ds_train = AnyTopDataset(split="train", **ds_kw)
    ds_val = AnyTopDataset(split="val", **ds_kw)
    log(f"train={len(ds_train)}  val={len(ds_val)}", rank)

    # Curriculum: which training clips count as human. This MUST match the main run
    # exactly or the two runs see different data distributions and the ablation is
    # confounded. train_graph_codeflow.py uses:
    #     str(r.get("motion_id", "")).upper().startswith("HML")   over ds_train.rows
    # AnyTopDataset exposes the same records as `.samples`.
    if not hasattr(ds_train, "samples"):
        raise SystemExit("[DATA FAIL] AnyTopDataset has no .samples; cannot build the curriculum")
    is_human = [str(s.get("motion_id", "")).upper().startswith("HML") for s in ds_train.samples]
    if len(is_human) != len(ds_train):
        raise SystemExit(
            f"[DATA FAIL] curriculum mask {len(is_human)} != dataset {len(ds_train)}")
    n_human = int(sum(is_human))
    log(f"[curriculum] human clips {n_human}/{len(ds_train)} "
        f"({100.0 * n_human / max(1, len(ds_train)):.1f}%) -> "
        f"phase1 x{args.human_upsample_factor}@{args.human_upsample_start_epoch}, "
        f"phase2 x{args.human_upsample_phase2_factor}@{args.human_upsample_phase2_start_epoch}", rank)
    if n_human == 0:
        raise SystemExit("[DATA FAIL] no human clips detected; curriculum would be a no-op")

    sampler = HumanCurriculumSampler(
        n=len(ds_train), is_upsampled=is_human,
        factor=args.human_upsample_factor, start_epoch=args.human_upsample_start_epoch,
        phase2_factor=args.human_upsample_phase2_factor,
        phase2_start_epoch=args.human_upsample_phase2_start_epoch,
        num_replicas=world, rank=rank, seed=args.seed)

    dl_train = DataLoader(ds_train, batch_size=args.batch_size, sampler=sampler,
                          collate_fn=anytop_collate_fn, num_workers=args.num_workers,
                          pin_memory=True, drop_last=True, persistent_workers=args.num_workers > 0)
    dl_val = DataLoader(ds_val, batch_size=args.batch_size, shuffle=False,
                        collate_fn=anytop_collate_fn, num_workers=max(1, args.num_workers // 2),
                        pin_memory=True, drop_last=False)

    # ---------------------------------------------------------------- model
    model = RawJointFlow(d_model=args.d_model, n_layers=args.n_layers,
                         n_heads=args.n_heads, d_ff=args.d_ff).to(dev)
    n_par = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log(f"RawJointFlow trainable params: {n_par:,}", rank)
    net = DDP(model, device_ids=[local]) if ddp else model
    raw = net.module if ddp else net

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    steps_per_epoch = max(1, len(dl_train) // args.grad_accum)
    args.total_steps = steps_per_epoch * args.epochs
    log(f"steps/epoch={steps_per_epoch}  total_steps={args.total_steps} "
        f"(global batch = {args.batch_size} x {args.grad_accum} x {world} = "
        f"{args.batch_size * args.grad_accum * world})", rank)

    start_epoch, gstep, best_val = 0, 0, float("inf")
    if args.resume and Path(args.resume).is_file():
        ck = torch.load(args.resume, map_location="cpu")
        raw.load_state_dict(ck["model_state_dict"])
        if "optimizer_state_dict" in ck:
            opt.load_state_dict(ck["optimizer_state_dict"])
        start_epoch = int(ck.get("epoch", -1)) + 1
        gstep = int(ck.get("global_step", 0))
        best_val = float(ck.get("best_val", float("inf")))
        log(f"resumed: start_epoch={start_epoch} global_step={gstep} best_val={best_val:.5f}", rank)

    amp = (args.amp_dtype == "bf16" and dev.type == "cuda")

    def save(tag: str, epoch: int, val: float) -> None:
        if rank != 0:
            return
        tmp = out_dir / f".{tag}.tmp"
        torch.save({"model_state_dict": raw.state_dict(),
                    "optimizer_state_dict": opt.state_dict(),
                    "epoch": epoch, "global_step": gstep, "val_flow": val,
                    "best_val": best_val, "args": vars(args)}, tmp)
        os.replace(tmp, out_dir / f"{tag}.pt")   # atomic: survives mid-write death

    # ---------------------------------------------------------------- loop
    for epoch in range(start_epoch, args.epochs):
        sampler.set_epoch(epoch)
        net.train()
        t0, run, seen = time.time(), 0.0, 0
        opt.zero_grad(set_to_none=True)
        for it, packed in enumerate(dl_train):
            batch = GraphMotionBatch.from_collate_dict(packed)
            x, tmask = motion_and_mask(batch, dev)
            cond = build_cond(batch, dev)
            # CFG: drop text on a fraction of samples (same probability as the main run)
            if args.cond_drop_prob > 0:
                drop = torch.rand(x.shape[0], device=dev) < args.cond_drop_prob
                cond["has_text"] = cond["has_text"] & (~drop)

            ctx = torch.autocast("cuda", dtype=torch.bfloat16) if amp else torch.enable_grad()
            with ctx:
                r = raw.flow_loss(x, tmask, cond, validate_inputs=(gstep == 0 and it == 0))
                loss = r["flow_loss"] / args.grad_accum
            loss.backward()

            if (it + 1) % args.grad_accum == 0:
                for g in opt.param_groups:
                    g["lr"] = lr_at(gstep, args)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                opt.step()
                opt.zero_grad(set_to_none=True)
                gstep += 1

            fl = float(r["flow_loss"].detach())
            if not math.isfinite(fl):
                raise SystemExit(f"[FAIL LOUD] non-finite flow_loss at epoch {epoch} it {it}")
            run += fl
            seen += 1
            if it % args.log_every == 0:
                log(f"[ep{epoch} it{it} step{gstep}] flow_loss={fl:.5f} "
                    f"lr={opt.param_groups[0]['lr']:.3e}", rank)
            if args.smoke and it + 1 >= args.smoke_iters:
                log(f"SMOKE OK: {it + 1} iters, mean flow_loss={run / seen:.5f}, "
                    f"peak {torch.cuda.max_memory_allocated() / 1e9:.1f} GB", rank)
                if ddp:
                    dist.destroy_process_group()
                return 0

        # ---- validation ----
        net.eval()
        vtot, vn = 0.0, 0
        with torch.no_grad():
            for packed in dl_val:
                batch = GraphMotionBatch.from_collate_dict(packed)
                x, tmask = motion_and_mask(batch, dev)
                cond = build_cond(batch, dev)
                ctx = torch.autocast("cuda", dtype=torch.bfloat16) if amp else torch.enable_grad()
                with ctx:
                    vr = raw.flow_loss(x, tmask, cond)
                vtot += float(vr["flow_loss"]); vn += 1
        val = vtot / max(1, vn)
        if ddp:   # average val across ranks so every rank agrees on "best"
            t = torch.tensor([val], device=dev)
            dist.all_reduce(t, op=dist.ReduceOp.SUM)
            val = float(t.item()) / world
        dt = time.time() - t0
        log(f"=== epoch {epoch} done in {dt:.1f}s | train_flow={run / max(1, seen):.5f} "
            f"val_flow={val:.5f} ===", rank)

        if val < best_val:
            best_val = val
            save("best_model", epoch, val)
            log(f"  [ckpt] new best val_flow={val:.5f}", rank)
        if (epoch + 1) % args.save_every == 0 or epoch == args.epochs - 1:
            save("last_model", epoch, val)

    log("=== training loop complete ===", rank)
    if ddp:
        dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
