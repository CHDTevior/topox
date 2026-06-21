"""Frozen-evaluator SANITY gates (M5): validate the trained AnyTop T2M evaluator
measures genuine text<->motion correspondence BEFORE it is used as a metric.

Loads the FROZEN evaluator checkpoint (best_model.pt), encodes val_all ONCE, then
reports four analyses — all pooled at `--pool` (=32, the training val protocol) so
the baseline is directly comparable to the training val gate (~0.956):

  1. BASELINE R@1/2/3 + matching (reproduce the training val number -> confirms the
     load is correct and the eval path is faithful).
  2. SHUFFLED-CAPTION R@1/2/3 (permute the ENCODED caption rows within each pool ==
     re-encoding permuted captions, since the text tower is per-caption independent;
     target = diagonal, NO group mask -> should collapse toward chance (R@k ~= k/pool).
     This is a text-PAIR-sensitivity control: it shows the aligned caption matters after
     encoding. It does NOT by itself rule out a species/source shortcut -> see the
     within-species SHUFFLED control below for the fine-grained version.
  3. WITHIN-SPECIES R@1/2/3 (retrieval pools drawn from a SINGLE object_type ->
     harder same-skeleton negatives; tests fine-grained discrimination).
  4. PER-SUBSET R@1/2/3 (animo4d vs truebones via `source` metadata, separately ->
     cross-data-source generalization).

Single-GPU, eval-only, no grad, no DDP. fp32 encoder forward (matches val_eval, which
runs core(...) WITHOUT autocast — needed to reproduce the 0.956 reference exactly).
Acceptance reading: baseline >> shuffled (shuffled ~= 1/pool), and per-subset/within-
species reported (not gated to a hard threshold — interpreted alongside visual QA).
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.anytop_t2m_eval_dataset import AnyTopT2MEvalDataset, collate_fn
from src.models.graph_salad.batch import GraphMotionBatch
from src.models.graph_salad.t2m_evaluator import (
    AnyTopT2MEvaluator,
    build_false_negative_mask,
)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", required=True, help="frozen evaluator .pt (best_model.pt).")
    ap.add_argument("--val_manifest", required=True)
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--pool", type=int, default=32, help="R-precision pool size (training used 32).")
    ap.add_argument("--num_frames", type=int, default=300)
    ap.add_argument("--max_joints", type=int, default=144)
    ap.add_argument("--encode_batch", type=int, default=64, help="batch size for the single encode pass.")
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--min_species", type=int, default=8,
                    help="skip within-species groups smaller than this (need a meaningful pool).")
    ap.add_argument("--seed", type=int, default=42, help="seed for the shuffled-caption permutation.")
    ap.add_argument("--out", default=None, help="optional JSON report path.")
    return ap.parse_args()


@torch.no_grad()
def rprecision_pool(te: torch.Tensor, me: torch.Tensor, mask: torch.Tensor | None, ks=(1, 2, 3)):
    """Group-aware R@K for ONE pool. te/me: [P, D] (rows aligned: text i <-> motion i).
    `mask`: [P, P] bool of acceptable (group-duplicate) matches, or None (diagonal-only).
    A query (text row) scores a hit@k if ANY acceptable motion is in its top-k by cosine."""
    sim = te @ me.t()                                    # [P, P], rows=text, cols=motion
    correct = torch.zeros_like(sim, dtype=torch.bool) if mask is None else mask.clone().to(sim.device)
    correct.fill_diagonal_(True)                         # the true pair is always acceptable
    order = sim.argsort(dim=1, descending=True)
    out = {}
    for k in ks:
        out[k] = correct.gather(1, order[:, :k]).any(dim=1).float().mean().item()
    return out


def avg_over_pools(te, me, meta, pool, masked: bool, shuffled: bool, gen: torch.Generator | None):
    """Chunk [N,D] embeddings into pools of `pool` (drop remainder, dataset order) and
    average R@K. masked -> build false-neg group mask per pool. shuffled -> permute the
    text rows within each pool (caption<->motion correspondence destroyed; diagonal-only)."""
    mids, smids, caps = meta["motion_id"], meta["source_motion_id"], meta["caption_text"]
    n = te.shape[0]
    npool = n // pool
    if npool == 0:
        return None, 0
    acc = {1: 0.0, 2: 0.0, 3: 0.0}
    for p in range(npool):
        s = slice(p * pool, (p + 1) * pool)
        te_p, me_p = te[s], me[s]
        if shuffled:
            perm = torch.randperm(pool, generator=gen)
            te_p = te_p[perm]                            # random caption for each motion slot
            m = None                                     # group mask meaningless once shuffled
        elif masked:
            m = build_false_negative_mask(mids[s], smids[s], caps[s])
        else:
            m = None
        r = rprecision_pool(te_p, me_p, m)
        for k in acc:
            acc[k] += r[k]
    return {k: acc[k] / npool for k in acc}, npool


@torch.no_grad()
def main() -> int:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(args.ckpt, map_location="cpu")
    m = ckpt.get("args", {})
    g = (lambda k, d: m.get(k, d)) if isinstance(m, dict) else (lambda k, d: getattr(m, k, d))
    core = AnyTopT2MEvaluator(
        coemb_dim=g("coemb_dim", 512), text_tower=g("text_tower", "distilbert"),
        distilbert_path=g("distilbert_path", "checkpoints/text_encoders/distilbert-base-uncased"),
        text_max_length=g("text_max_length", 64),
        n_heads=g("n_heads", 8), d_ff=g("d_ff", 2048),
        n_graph_layers=g("n_graph_layers", 6), n_temporal_layers=g("n_temporal_layers", 4),
        motion_feat_dim=g("motion_feat_dim", 13),  # 12ch (contact-free) ckpt rebuilds at 12; old 13ch -> 13
        dropout=g("dropout", 0.1),
        learnable_temperature=not g("fixed_temperature", False), temperature=g("temperature", 0.07),
    )
    missing, unexpected = core.load_state_dict(ckpt["model"], strict=False)
    # tolerate ONLY the frozen DistilBERT backbone (text_distilbert.text_model.*), which
    # may be reconstructed from the HF model. FAIL LOUD on any other missing key — incl.
    # the TRAINABLE text head text_distilbert.head.* — or any unexpected key.
    bad_missing = [k for k in missing if not k.startswith("text_distilbert.text_model.")]
    if bad_missing or unexpected:
        raise SystemExit(f"[sanity] state_dict mismatch: missing(non-backbone)={bad_missing[:8]} "
                         f"unexpected={list(unexpected)[:8]}")
    core.to(device).eval()
    text_tower = g("text_tower", "distilbert")
    print(f"[sanity] loaded {args.ckpt} (epoch={ckpt.get('epoch','?')} "
          f"saved_val={ckpt.get('val','?')}) text_tower={text_tower}", flush=True)

    ds = AnyTopT2MEvalDataset(
        manifest_path=args.val_manifest, data_root=args.data_root,
        caption_emb_cache=None, split="val", view="full",
        num_frames=args.num_frames, max_joints=args.max_joints,
    )
    loader = DataLoader(ds, batch_size=args.encode_batch, shuffle=False,
                        num_workers=args.num_workers, collate_fn=collate_fn, drop_last=False)

    # ---- single encode pass: collect embeddings + metadata in dataset order ----
    te_all, me_all = [], []
    meta = {"motion_id": [], "source_motion_id": [], "caption_text": [],
            "object_type": [], "source": []}
    for coll in loader:
        coll_dev = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in coll.items()}
        batch = GraphMotionBatch.from_collate_dict(coll_dev)
        text_input = coll_dev["caption_text"] if text_tower == "distilbert" else coll_dev["caption_emb"]
        out = core(batch, text_input)
        te_all.append(out.text_emb.float().cpu())
        me_all.append(out.motion_emb.float().cpu())
        for key in meta:
            meta[key].extend(coll[key])
    te_all = torch.cat(te_all, 0)
    me_all = torch.cat(me_all, 0)
    n = te_all.shape[0]
    print(f"[sanity] encoded {n} val motions (D={te_all.shape[1]})", flush=True)

    gen = torch.Generator().manual_seed(args.seed)
    report = {"n_total": n, "pool": args.pool, "ckpt": args.ckpt, "epoch": ckpt.get("epoch")}

    # (1) baseline + (2) shuffled — same dataset-order pools of `pool`
    base, npool = avg_over_pools(te_all, me_all, meta, args.pool, masked=True, shuffled=False, gen=None)
    shuf, _ = avg_over_pools(te_all, me_all, meta, args.pool, masked=False, shuffled=True, gen=gen)
    n_used = npool * args.pool
    chance = {k: min(1.0, k / args.pool) for k in (1, 2, 3)}
    report["chance"] = chance
    report["baseline"] = {"rprec": base, "n_pools": npool, "n_used": n_used, "n_dropped": n - n_used}
    report["shuffled_caption"] = {"rprec": shuf, "n_pools": npool}
    print(f"[sanity] BASELINE  R@1={base[1]:.3f} R@2={base[2]:.3f} R@3={base[3]:.3f} "
          f"({npool} pools of {args.pool}; used {n_used}/{n}, dropped {n-n_used})", flush=True)
    print(f"[sanity] SHUFFLED  R@1={shuf[1]:.3f} R@2={shuf[2]:.3f} R@3={shuf[3]:.3f} "
          f"(chance R@1/2/3={chance[1]:.3f}/{chance[2]:.3f}/{chance[3]:.3f})", flush=True)
    print(f"[sanity]   -> shuffled/baseline R@1 ratio={shuf[1]/max(base[1],1e-9):.3f}; "
          f"shuffled near chance SUPPORTS text-PAIR sensitivity (not full text-grounding proof)", flush=True)

    # FAIL-LOUD baseline gate: the sanity baseline MUST reproduce the training val gate
    # (identical protocol). Enforced only when --pool == the ckpt's val_batch_size (else
    # the pooling differs and the numbers legitimately won't match).
    saved = ckpt.get("val") or {}
    vbs = g("val_batch_size", args.pool)
    if args.pool != vbs:
        print(f"[sanity] NOTE: --pool {args.pool} != ckpt val_batch_size {vbs} -> skip baseline-vs-ckpt gate", flush=True)
    elif saved:
        for k, key in ((1, "r1"), (2, "r2"), (3, "r3")):
            if key in saved and abs(base[k] - saved[key]) > 0.02:
                raise SystemExit(f"[sanity] BASELINE MISMATCH R@{k}: computed {base[k]:.4f} vs ckpt['val'] "
                                 f"{saved[key]:.4f} (tol 0.02) -> eval path diverges from training val_eval; "
                                 f"FIX before trusting any gate.")
        print(f"[sanity] baseline reproduces ckpt['val'] within tol "
              f"(R@1 {base[1]:.4f} vs {saved.get('r1'):.4f}) OK", flush=True)

    # (3) per-subset (animo4d vs truebones) — filter then pool
    report["per_subset"] = {}
    srcs = sorted(set(meta["source"]))
    for src in srcs:
        idx = [i for i, s in enumerate(meta["source"]) if s == src]
        sub_meta = {k: [meta[k][i] for i in idx] for k in meta}
        rr, npp = avg_over_pools(te_all[idx], me_all[idx], sub_meta, args.pool, masked=True, shuffled=False, gen=None)
        used = npp * args.pool
        report["per_subset"][src] = {"n": len(idx), "n_pools": npp, "n_used": used,
                                     "n_dropped": len(idx) - used, "rprec": rr}
        if rr:
            print(f"[sanity] SUBSET {src:24s} n={len(idx):5d} R@1={rr[1]:.3f} R@2={rr[2]:.3f} R@3={rr[3]:.3f} "
                  f"({npp} pools, used {used}/{len(idx)}, dropped {len(idx)-used})", flush=True)
        else:
            print(f"[sanity] SUBSET {src:24s} n={len(idx):5d} (< pool size {args.pool}, skipped)", flush=True)

    # (4) within-species — group by object_type, pool WITHIN each species
    by_sp = defaultdict(list)
    for i, ot in enumerate(meta["object_type"]):
        by_sp[ot].append(i)
    sp_acc = {1: 0.0, 2: 0.0, 3: 0.0}
    sp_acc_shuf = {1: 0.0, 2: 0.0, 3: 0.0}
    sp_pools = 0
    n_species_used = 0
    n_species_total = len(by_sp)
    n_items_used = 0
    sp_gen = torch.Generator().manual_seed(args.seed + 1)
    for ot, idx in by_sp.items():
        if len(idx) < args.min_species:
            continue
        sub_meta = {k: [meta[k][i] for i in idx] for k in meta}
        pool_k = min(args.pool, len(idx))
        rr, npp = avg_over_pools(te_all[idx], me_all[idx], sub_meta, pool_k, masked=True, shuffled=False, gen=None)
        rs, _ = avg_over_pools(te_all[idx], me_all[idx], sub_meta, pool_k, masked=False, shuffled=True, gen=sp_gen)
        if rr and npp:
            n_species_used += 1
            sp_pools += npp
            n_items_used += npp * pool_k
            for k in sp_acc:
                sp_acc[k] += rr[k] * npp          # pool-weighted
                sp_acc_shuf[k] += rs[k] * npp
    within = {k: (sp_acc[k] / sp_pools if sp_pools else None) for k in sp_acc}
    within_shuf = {k: (sp_acc_shuf[k] / sp_pools if sp_pools else None) for k in sp_acc}
    report["within_species"] = {
        "rprec": within, "rprec_shuffled": within_shuf,
        "n_species_used": n_species_used, "n_species_total": n_species_total,
        "n_species_skipped": n_species_total - n_species_used,
        "n_pools": sp_pools, "n_items_used": n_items_used, "min_species": args.min_species}
    if sp_pools:
        print(f"[sanity] WITHIN-SPECIES R@1={within[1]:.3f} R@2={within[2]:.3f} R@3={within[3]:.3f} "
              f"| SHUFFLED R@1={within_shuf[1]:.3f} "
              f"({n_species_used}/{n_species_total} species >= {args.min_species}, {sp_pools} pools, "
              f"{n_items_used} items, pool-weighted)", flush=True)
    else:
        print(f"[sanity] WITHIN-SPECIES: no species with >= {args.min_species} items "
              f"(of {n_species_total} total)", flush=True)

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(report, indent=2))
        print(f"[sanity] report -> {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
