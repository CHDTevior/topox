"""Train the AnyTop T2M evaluator (M2) with symmetric multi-positive InfoNCE.

Design doc: handoff/20260604_0121_anytop_t2m_evaluator_impl_proposal.md (§5-M2)
            handoff/20260604_0257_anytop_t2m_evaluator_impl_progress.md (M2 NEXT)

WHAT THIS TRAINS
----------------
`src.models.graph_salad.t2m_evaluator.AnyTopT2MEvaluator` — a standalone, frozen
*measuring instrument* for the AnyTop T2M metrics. It is trained ONLY on real
training motions (`AnyTopT2MEvalDataset` over the `train_main` manifest), NEVER on
generated samples, and shares NO weights with the VAE / diffusion generators (it
freshly initializes its own `SkeletonEncoder` motion tower). See the model
docstring for the full independent-frozen-evaluator contract (§1c-1/2/3).

LOSS
----
Symmetric (text->motion + motion->text) InfoNCE with the multi-positive
false-negative mask (§1c-8): off-diagonal pairs that share
`(motion_id) ∪ (source_motion_id) ∪ (caption_text)` are removed from the
denominator so duplicate captions / motions are not punished as hard negatives.
`source_motion_id` / `object_type` are used ONLY to build that mask — they are
never fed to either tower (§1c-7).

SCOPE NOTE (M2 = SMOKE ONLY)
----------------------------
Per the current handoff, the evaluator only needs to SMOKE-pass on CPU (no spare
GPU to actually train it). This script is the real training loop, but the M2
acceptance is the CPU smoke in scripts/_smoke_anytop_t2m_evaluator.py
(instantiate + forward + finite InfoNCE + backward + mask logic + tiny overfit).
The full 4-gate sanity (held-out R-precision, shuffled-caption control,
species_stripped sanity) and the group-aware R-precision / matching / FID metrics
are M3/M4 — intentionally NOT implemented here (Karpathy R2: no speculative code).

Checkpoint writes are guarded rank-0-only for forward-compatibility with a future
DDP launch; this script itself is single-process (CPU smoke).
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.anytop_t2m_eval_dataset import AnyTopT2MEvalDataset, collate_fn
from src.models.graph_salad.batch import GraphMotionBatch
from src.models.graph_salad.t2m_evaluator import (
    AnyTopT2MEvaluator,
    build_multi_positive_mask,
)


def _is_main() -> bool:
    """Rank-0 guard (single-process here; honors RANK if a launcher sets it)."""
    return int(os.environ.get("RANK", "0")) == 0


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    # ---- data ----
    ap.add_argument(
        "--manifest", required=True,
        help="M0 eval-split manifest JSON (train_main.json for real training).",
    )
    ap.add_argument(
        "--data_root", required=True,
        help="AnyTop processed data root (cond.npy + motions/ + splits/).",
    )
    ap.add_argument(
        "--caption_emb_cache", required=True,
        help="T5 sidecar cache prefix (<prefix>.embs.npy + <prefix>.keys.json).",
    )
    ap.add_argument(
        "--split", default="train", choices=["train", "val"],
        help="Underlying AnyTopDataset split (train for train_main).",
    )
    ap.add_argument(
        "--view", default="full", choices=["full", "species_stripped"],
        help="Caption view (full=cap0; species_stripped=animal-level cap).",
    )
    ap.add_argument("--num_frames", type=int, default=260)
    ap.add_argument("--max_joints", type=int, default=144)
    ap.add_argument("--num_workers", type=int, default=4)

    # ---- model ----
    ap.add_argument("--coemb_dim", type=int, default=384)
    ap.add_argument("--n_heads", type=int, default=8)
    ap.add_argument("--d_ff", type=int, default=1536)
    ap.add_argument("--n_graph_layers", type=int, default=4)
    ap.add_argument("--n_temporal_layers", type=int, default=2)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument(
        "--temperature", type=float, default=0.07,
        help="InfoNCE temperature (init for learnable, fixed otherwise).",
    )
    ap.add_argument(
        "--fixed_temperature", action="store_true",
        help="Use a fixed (non-learnable) temperature.",
    )

    # ---- optim ----
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-6)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--device", default="cpu",
        help="torch device (cpu for the CPU smoke; cuda only for a real run).",
    )

    # ---- io ----
    ap.add_argument(
        "--out", default=None,
        help="output dir for evaluator ckpts + train.log (rank-0 only).",
    )
    ap.add_argument("--save_every", type=int, default=10)
    ap.add_argument("--log_every", type=int, default=20)
    return ap.parse_args()


def build_dataset(args: argparse.Namespace) -> AnyTopT2MEvalDataset:
    return AnyTopT2MEvalDataset(
        manifest_path=args.manifest,
        data_root=args.data_root,
        caption_emb_cache=args.caption_emb_cache,
        split=args.split,
        view=args.view,
        num_frames=args.num_frames,
        max_joints=args.max_joints,
    )


def build_model(args: argparse.Namespace) -> AnyTopT2MEvaluator:
    return AnyTopT2MEvaluator(
        coemb_dim=args.coemb_dim,
        n_heads=args.n_heads,
        d_ff=args.d_ff,
        n_graph_layers=args.n_graph_layers,
        n_temporal_layers=args.n_temporal_layers,
        dropout=args.dropout,
        learnable_temperature=not args.fixed_temperature,
        temperature=args.temperature,
    )


def run_step(
    model: AnyTopT2MEvaluator,
    coll: dict,
    device: torch.device,
) -> torch.Tensor:
    """One forward + InfoNCE loss on a collated batch. Returns the scalar loss.

    Moves the collated tensors to `device`, builds the typed `GraphMotionBatch`
    (validates schema), runs both towers, builds the multi-positive mask from the
    metadata string lists, and returns the symmetric InfoNCE loss.
    """
    # Move tensors to device (lists / strings pass through untouched).
    coll_dev = {
        k: (v.to(device) if isinstance(v, torch.Tensor) else v)
        for k, v in coll.items()
    }
    batch = GraphMotionBatch.from_collate_dict(coll_dev)
    caption_emb = coll_dev["caption_emb"]  # [B, 768]

    out = model(batch, caption_emb)
    mask = build_multi_positive_mask(
        motion_id=coll_dev["motion_id"],
        source_motion_id=coll_dev["source_motion_id"],
        caption_text=coll_dev["caption_text"],
    )
    return model.contrastive_loss(out, false_neg_mask=mask)


def main() -> int:
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "device=cuda requested but CUDA is unavailable. The M2 evaluator "
            "smoke is CPU-only; pass --device cpu."
        )

    dataset = build_dataset(args)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=(args.split == "train"),
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        drop_last=True,
    )

    model = build_model(args).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    if _is_main():
        print(
            f"[evaluator] AnyTopT2MEvaluator params = {n_params/1e6:.2f}M "
            f"(coemb={args.coemb_dim}, d_ff={args.d_ff}, "
            f"graph={args.n_graph_layers}, temporal={args.n_temporal_layers})"
        )

    optim = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.99),
    )

    out_dir = Path(args.out) if args.out else None
    if out_dir is not None and _is_main():
        out_dir.mkdir(parents=True, exist_ok=True)

    model.train()
    global_step = 0
    t0 = time.time()
    for epoch in range(args.epochs):
        for coll in loader:
            loss = run_step(model, coll, device)
            optim.zero_grad(set_to_none=True)
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optim.step()

            if _is_main() and global_step % args.log_every == 0:
                ls = float(model.logit_scale.detach())
                print(
                    f"ep{epoch} step{global_step} loss {loss.item():.4f} "
                    f"logit_scale {ls:.2f} "
                    f"({(time.time()-t0):.1f}s)"
                )
            global_step += 1

        if (
            out_dir is not None and _is_main()
            and args.save_every > 0
            and (epoch + 1) % args.save_every == 0
        ):
            ckpt = {
                "model": model.state_dict(),
                "epoch": epoch,
                "args": vars(args),
            }
            torch.save(ckpt, out_dir / f"evaluator_ep{epoch+1}.pt")
            (out_dir / "train.log").open("a").write(
                json.dumps({"epoch": epoch, "loss": loss.item()}) + "\n"
            )

    if _is_main():
        print(f"[evaluator] done — {global_step} steps in {(time.time()-t0):.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
