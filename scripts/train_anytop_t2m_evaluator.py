"""Train the AnyTop T2M evaluator (M3) with DDP global symmetric false-negative-masked diagonal InfoNCE.

Design doc: handoff/20260614_anytop_t2m_evaluator_vq_codeflow_revision.md (§6, §6.1).

WHAT THIS TRAINS
----------------
`src.models.graph_salad.t2m_evaluator.AnyTopT2MEvaluator` — a standalone, frozen
*measuring instrument* for AnyTop T2M metrics. Trained ONLY on real training
motions (`AnyTopT2MEvalDataset` over train_main), NEVER on generated samples,
sharing NO weights with the VAE / VQVAE / CodeFlow generators (fresh SkeletonEncoder
motion tower; frozen DistilBERT text backbone + trainable head). See the model
docstring for the independent-frozen-evaluator contract.

LOSS — GLOBAL DDP false-negative-masked diagonal InfoNCE (spec §6)
--------------------------------------------------
Each rank encodes its local batch (bf16 autocast), then text_emb / motion_emb are
GRAD-SAFE all-gathered across ranks (torch.distributed.nn.all_gather — plain
all_gather would DETACH grads and silently train on local-only negatives) and the
symmetric InfoNCE is computed on the GLOBAL [WB,WB] logits in fp32. The false-negative
mask (motion_id ∪ source_motion_id ∪ caption_text) is built from
metadata all-gathered in rank order. Text is the RAW caption string (DistilBERT);
species/object/source_motion_id are mask/metadata only, never tower inputs.

GATE: smoke only here; full training is launched separately after user review.
Checkpoints are rank-0 only; safe single-process (WORLD_SIZE unset) for the smoke.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist
import torch.distributed.nn as dist_nn  # differentiable collectives (grad-safe all_gather)
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.anytop_t2m_eval_dataset import AnyTopT2MEvalDataset, collate_fn
from src.models.graph_salad.batch import GraphMotionBatch
from src.models.graph_salad.t2m_evaluator import (
    AnyTopT2MEvaluator,
    build_false_negative_mask,
    symmetric_infonce,
)


# ----------------------------------------------------------------------------- #
# DDP setup
# ----------------------------------------------------------------------------- #
def setup_ddp():
    """Init DDP from torchrun env. Returns (rank, world_size, local_rank, device, is_dist)."""
    ws = int(os.environ.get("WORLD_SIZE", "1"))
    if ws > 1:
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        # 30-min PG timeout (default is 10 min): the post-val dist.barrier() makes ranks
        # 1-3 wait while rank 0 runs the FULL val set (val_max_batches=0, ~119 batches);
        # generous headroom so an I/O-stalled rank-0 val never trips the NCCL watchdog
        # (TORCH_NCCL_ASYNC_ERROR_HANDLING=1 would otherwise abort the whole run).
        dist.init_process_group(backend=backend, timeout=timedelta(minutes=30))
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
            device = torch.device("cuda", local_rank)
        else:
            device = torch.device("cpu")
        return rank, ws, local_rank, device, True
    # single process
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return 0, 1, 0, device, False


def is_main(rank: int) -> bool:
    return rank == 0


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    # ---- data ----
    ap.add_argument("--manifest", required=True, help="M0 train_main.json manifest.")
    ap.add_argument("--data_root", required=True, help="AnyTop merged data root.")
    ap.add_argument("--caption_emb_cache", default=None,
                    help="T5 sidecar prefix; ONLY for --text_tower t5_cache. "
                         "Leave unset for the DistilBERT primary path.")
    ap.add_argument("--split", default="train", choices=["train", "val"])
    ap.add_argument("--view", default="full", choices=["full", "species_stripped"])
    # held-out val gate (rank-0; HumanML3D-style per-epoch val + best-by-val safety net).
    ap.add_argument("--val_manifest", default=None,
                    help="held-out val manifest (val_all.json); enables the per-epoch val gate.")
    ap.add_argument("--val_every", type=int, default=5, help="run val gate every N epochs.")
    ap.add_argument("--val_batch_size", type=int, default=32,
                    help="R-precision pool size per val batch (HumanML3D uses 32).")
    ap.add_argument("--val_max_batches", type=int, default=64,
                    help="cap val batches per gate for speed (0 = full val).")
    ap.add_argument("--num_frames", type=int, default=300)
    ap.add_argument("--max_joints", type=int, default=144)
    ap.add_argument("--num_workers", type=int, default=4)

    # ---- model ----
    ap.add_argument("--text_tower", default="distilbert", choices=["distilbert", "t5_cache"])
    ap.add_argument("--distilbert_path",
                    default="checkpoints/text_encoders/distilbert-base-uncased")
    ap.add_argument("--text_max_length", type=int, default=64)
    ap.add_argument("--coemb_dim", type=int, default=512)
    ap.add_argument("--n_heads", type=int, default=8)
    ap.add_argument("--d_ff", type=int, default=2048)
    ap.add_argument("--n_graph_layers", type=int, default=6)
    ap.add_argument("--n_temporal_layers", type=int, default=4)
    ap.add_argument("--motion_feat_dim", type=int, default=13, choices=[12, 13],
                    help="motion-tower input channels: 13=full AnyTop, 12=drop contact ch12 "
                         "(contact-free clean motion-semantics evaluator).")
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--temperature", type=float, default=0.07)
    ap.add_argument("--fixed_temperature", action="store_true")

    # ---- optim ----
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch_size", type=int, default=32, help="PER-RANK batch size.")
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--warmup_steps", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--bf16", action="store_true", help="bf16 autocast for encoder forward.")

    # ---- io ----
    ap.add_argument("--out", default=None)
    ap.add_argument("--save_every", type=int, default=10)
    ap.add_argument("--log_every", type=int, default=20)
    ap.add_argument("--max_steps", type=int, default=0, help="if >0, stop after N steps (smoke).")
    return ap.parse_args()


def build_dataset(args) -> AnyTopT2MEvalDataset:
    return AnyTopT2MEvalDataset(
        manifest_path=args.manifest,
        data_root=args.data_root,
        # DistilBERT primary path uses raw caption_text -> caption_emb_cache=None.
        caption_emb_cache=(args.caption_emb_cache if args.text_tower == "t5_cache" else None),
        split=args.split,
        view=args.view,
        num_frames=args.num_frames,
        max_joints=args.max_joints,
    )


def build_model(args) -> AnyTopT2MEvaluator:
    return AnyTopT2MEvaluator(
        coemb_dim=args.coemb_dim,
        text_tower=args.text_tower,
        distilbert_path=args.distilbert_path,
        text_max_length=args.text_max_length,
        n_heads=args.n_heads,
        d_ff=args.d_ff,
        n_graph_layers=args.n_graph_layers,
        n_temporal_layers=args.n_temporal_layers,
        motion_feat_dim=args.motion_feat_dim,
        dropout=args.dropout,
        learnable_temperature=not args.fixed_temperature,
        temperature=args.temperature,
    )


def _gather_meta(meta_lists, world_size: int):
    """All-gather three metadata string lists, concatenated in RANK ORDER.

    Rank order MUST match the embedding all_gather order so the false-negative
    mask aligns row-for-row with the gathered embeddings (⚠ ordering footgun).
    """
    if world_size == 1:
        return meta_lists
    gathered = [None] * world_size
    dist.all_gather_object(gathered, meta_lists)
    mid = [x for r in gathered for x in r[0]]
    smid = [x for r in gathered for x in r[1]]
    cap = [x for r in gathered for x in r[2]]
    return mid, smid, cap


def run_step(model, core, coll, device, world_size, is_dist, text_tower, autocast_dtype):
    """One forward + GLOBAL false-negative-masked diagonal InfoNCE. `model` is the (DDP) module;
    `core` is the unwrapped AnyTopT2MEvaluator (for logit_scale). The ENCODER forward
    runs under bf16 autocast; the gather + global logits + cross-entropy stay fp32
    (spec §6.1) — autocast is scoped to model(...) ONLY so the matmul/CE aren't bf16."""
    coll_dev = {k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                for k, v in coll.items()}
    batch = GraphMotionBatch.from_collate_dict(coll_dev)
    # text input: raw captions (DistilBERT primary) or T5 emb (fallback ablation).
    text_input = coll_dev["caption_text"] if text_tower == "distilbert" else coll_dev["caption_emb"]

    if autocast_dtype is not None and device.type == "cuda":
        with torch.autocast(device_type="cuda", dtype=autocast_dtype):
            out = model(batch, text_input)       # DDP forward -> local [b, D] (with grad)
    else:
        out = model(batch, text_input)
    # fp32 logits/CE: cast local embeddings out of bf16 BEFORE the gather/matmul.
    text_local = out.text_emb.float()
    motion_local = out.motion_emb.float()

    meta_local = (coll_dev["motion_id"], coll_dev["source_motion_id"], coll_dev["caption_text"])
    if is_dist and world_size > 1:
        # grad-safe differentiable all_gather (NOT torch.distributed.all_gather).
        text_all = torch.cat(dist_nn.all_gather(text_local), dim=0)     # [WB, D]
        motion_all = torch.cat(dist_nn.all_gather(motion_local), dim=0)  # [WB, D]
        mid, smid, cap = _gather_meta(meta_local, world_size)
    else:
        text_all, motion_all = text_local, motion_local
        mid, smid, cap = meta_local

    mask = build_false_negative_mask(mid, smid, cap)            # global [WB, WB]
    loss = symmetric_infonce(text_all, motion_all, core.logit_scale.float(), false_neg_mask=mask)
    return loss


@torch.no_grad()
def val_eval(core, val_loader, device, text_tower, max_batches):
    """Held-out val gate (rank-0 only; HumanML3D-style safety net). Per val batch:
    group-aware R@1/2/3 (a top-k hit if ANY same-group motion — diagonal OR a
    duplicate-aware match — is in the top-k) + matching score (mean true-pair cosine)
    + false-neg-masked InfoNCE. R-precision is pooled within each val batch
    (val_batch_size, HumanML3D uses 32). Averages over up to max_batches val batches."""
    core.eval()
    tot = {"loss": 0.0, "r1": 0.0, "r2": 0.0, "r3": 0.0, "match": 0.0}
    n = 0
    for bi, coll in enumerate(val_loader):
        if max_batches and bi >= max_batches:
            break
        coll = {k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                for k, v in coll.items()}
        batch = GraphMotionBatch.from_collate_dict(coll)
        text_input = coll["caption_text"] if text_tower == "distilbert" else coll["caption_emb"]
        out = core(batch, text_input)
        te, me = out.text_emb.float(), out.motion_emb.float()
        mask = build_false_negative_mask(
            coll["motion_id"], coll["source_motion_id"], coll["caption_text"])
        sim = te @ me.t()                                          # [B, B] cosine
        correct = mask.clone().to(sim.device)
        correct.fill_diagonal_(True)                               # acceptable (group-aware) matches
        order = sim.argsort(dim=1, descending=True)
        for k, key in ((1, "r1"), (2, "r2"), (3, "r3")):
            tot[key] += correct.gather(1, order[:, :k]).any(dim=1).float().mean().item()
        tot["match"] += sim.diagonal().mean().item()
        tot["loss"] += symmetric_infonce(
            te, me, core.logit_scale.float(), false_neg_mask=mask).item()
        n += 1
    core.train()
    return None if n == 0 else {k: v / n for k, v in tot.items()}


def main() -> int:
    args = parse_args()
    if args.text_tower == "t5_cache" and not args.caption_emb_cache:
        raise SystemExit(
            "--text_tower t5_cache requires --caption_emb_cache (T5 sidecar prefix); "
            "without it the dataset emits ZERO caption_emb and the fallback would train "
            "on empty text. Use --text_tower distilbert (primary) or pass the cache.")
    rank, world_size, local_rank, device, is_dist = setup_ddp()
    torch.manual_seed(args.seed + rank)

    dataset = build_dataset(args)
    if is_dist:
        sampler = DistributedSampler(dataset, shuffle=(args.split == "train"), drop_last=True)
        shuffle = False
    else:
        sampler = None
        shuffle = (args.split == "train")
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=shuffle, sampler=sampler,
        num_workers=args.num_workers, collate_fn=collate_fn, drop_last=True,
    )

    # Held-out val loader (rank-0 only) — the per-epoch val gate + best-by-val safety
    # net (HumanML3D-style). Full R@K/shuffle/source gates (M5) are still post-training.
    val_loader = None
    if args.val_manifest and is_main(rank):
        val_ds = AnyTopT2MEvalDataset(
            manifest_path=args.val_manifest, data_root=args.data_root,
            caption_emb_cache=(args.caption_emb_cache if args.text_tower == "t5_cache" else None),
            split="val", view=args.view, num_frames=args.num_frames, max_joints=args.max_joints,
        )
        val_loader = DataLoader(
            val_ds, batch_size=args.val_batch_size, shuffle=False, num_workers=2,
            collate_fn=collate_fn, drop_last=True,
        )
    best_val_r1 = -1.0

    core = build_model(args).to(device)
    if is_main(rank):
        n_train = sum(p.numel() for p in core.parameters() if p.requires_grad)
        n_total = sum(p.numel() for p in core.parameters())
        print(f"[evaluator] params trainable={n_train/1e6:.2f}M total={n_total/1e6:.2f}M "
              f"(coemb={args.coemb_dim} d_ff={args.d_ff} graph={args.n_graph_layers} "
              f"temporal={args.n_temporal_layers} text_tower={args.text_tower})")

    if is_dist:
        ddp_kwargs = {}
        if device.type == "cuda":
            ddp_kwargs["device_ids"] = [local_rank]
        # frozen DistilBERT + unused SkeletonEncoder branches produce no grad.
        model = DDP(core, find_unused_parameters=True, **ddp_kwargs)
    else:
        model = core

    optim = torch.optim.AdamW(
        [p for p in core.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.99),
    )
    steps_per_epoch = max(1, len(loader))
    total_steps = args.max_steps if args.max_steps > 0 else args.epochs * steps_per_epoch

    def lr_lambda(step: int) -> float:
        if step < args.warmup_steps:
            return (step + 1) / max(1, args.warmup_steps)
        prog = (step - args.warmup_steps) / max(1, total_steps - args.warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, prog)))
    sched = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda)

    out_dir = Path(args.out) if args.out else None
    if out_dir is not None and is_main(rank):
        out_dir.mkdir(parents=True, exist_ok=True)

    model.train()
    autocast_dtype = torch.bfloat16 if args.bf16 else None
    global_step = 0
    t0 = time.time()
    stop = False
    for epoch in range(args.epochs):
        if is_dist:
            sampler.set_epoch(epoch)
        for coll in loader:
            loss = run_step(model, core, coll, device, world_size, is_dist,
                            args.text_tower, autocast_dtype)
            optim.zero_grad(set_to_none=True)
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in core.parameters() if p.requires_grad], args.grad_clip)
            optim.step()
            sched.step()

            if is_main(rank) and global_step % args.log_every == 0:
                ls = float(core.logit_scale.detach())
                print(f"ep{epoch} step{global_step} loss {loss.item():.4f} "
                      f"logit_scale {ls:.2f} lr {sched.get_last_lr()[0]:.2e} "
                      f"({(time.time()-t0):.1f}s)", flush=True)
            global_step += 1
            if args.max_steps > 0 and global_step >= args.max_steps:
                stop = True
                break

        if (out_dir is not None and is_main(rank) and args.save_every > 0
                and (epoch + 1) % args.save_every == 0):
            ckpt = {"model": core.state_dict(), "epoch": epoch, "args": vars(args)}
            tmp = out_dir / f".evaluator_ep{epoch+1}.pt.tmp"
            torch.save(ckpt, tmp)
            os.replace(tmp, out_dir / f"evaluator_ep{epoch+1}.pt")
            (out_dir / "train.log").open("a").write(
                json.dumps({"epoch": epoch, "loss": loss.item()}) + "\n")

        # --- held-out val gate (rank-0 runs val_eval; all ranks barrier to stay in step) ---
        run_val = bool(args.val_manifest) and args.val_every > 0 and (epoch + 1) % args.val_every == 0
        if run_val and val_loader is not None and is_main(rank):
            vm = val_eval(core, val_loader, device, args.text_tower, args.val_max_batches)
            if vm is not None:
                print(f"[val] ep{epoch} loss {vm['loss']:.4f} R@1 {vm['r1']:.3f} "
                      f"R@2 {vm['r2']:.3f} R@3 {vm['r3']:.3f} match {vm['match']:.3f}", flush=True)
                if out_dir is not None:
                    (out_dir / "val.log").open("a").write(json.dumps({"epoch": epoch, **vm}) + "\n")
                    if vm["r1"] > best_val_r1:
                        best_val_r1 = vm["r1"]
                        tmp = out_dir / ".best_model.pt.tmp"
                        torch.save({"model": core.state_dict(), "epoch": epoch,
                                    "val": vm, "args": vars(args)}, tmp)
                        os.replace(tmp, out_dir / "best_model.pt")
                        print(f"[val] new best R@1={best_val_r1:.3f} -> best_model.pt", flush=True)
        if is_dist and run_val:
            dist.barrier()  # non-rank-0 ranks wait while rank-0 runs the val gate + saves
        if stop:
            break

    if is_main(rank):
        print(f"[evaluator] done — {global_step} steps in {(time.time()-t0):.1f}s")
    if is_dist:
        dist.barrier()
        dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
