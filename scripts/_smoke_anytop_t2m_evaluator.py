"""Smoke for the AnyTop T2M evaluator (VQ/CodeFlow revision: DistilBERT raw-caption
primary tower + dim upgrade + DDP global false-negative-masked diagonal InfoNCE).

Spec: handoff/20260614_anytop_t2m_evaluator_vq_codeflow_revision.md §6.2 gates.

SINGLE-PROCESS (no WORLD_SIZE / WORLD_SIZE=1) runs gates:
  1. instantiate (frozen DistilBERT loads from local ckpt) + trainable/total param count.
  2. forward -> text_emb[B,512] + motion_emb[B,512], finite, L2-normalized.
  3. InfoNCE finite + backward: grads flow to ALL trainable params EXCEPT the disabled
     SkeletonEncoder clip/name branches; FROZEN DistilBERT params get no grad (requires_grad=False).
  4. false-negative mask logic (hand-built + real-batch brute-force match).
  5. tiny overfit on content-distinct real motions -> loss drops clearly.
  6. caption-shuffle: on the overfit batch, shuffling captions RAISES the loss
     (the model learned the correct text<->motion alignment).

DDP (launched via torchrun --nproc_per_node=2 ...) runs gate:
  7. global grad-safe all-gather false-negative-masked diagonal InfoNCE over 2 ranks: global batch == 2*local,
     loss finite + identical across ranks, grads flow to local trainable params (the grad-safe
     all_gather correctly backprops the global logits to each rank's local slice).

Run single-proc on a spare GPU step:
  srun --jobid=<own_alloc> --overlap --gres=gpu:1 --cpus-per-task=4 \
    python scripts/_smoke_anytop_t2m_evaluator.py
Run DDP gate (2 GPUs on one node):
  srun ... torchrun --nproc_per_node=2 scripts/_smoke_anytop_t2m_evaluator.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.anytop_t2m_eval_dataset import AnyTopT2MEvalDataset, collate_fn
from src.models.graph_salad.batch import GraphMotionBatch
from src.models.graph_salad.t2m_evaluator import (
    AnyTopT2MEvaluator,
    build_false_negative_mask,
    symmetric_infonce,
)

ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "data" / "animo4d_anytop_clean_L4_safe_plus_truebones"
SPLITS = DATA_ROOT / "eval_splits"
DISTILBERT = str(ROOT / "checkpoints" / "text_encoders" / "distilbert-base-uncased")
COEMB = 512
import os as _os
SMOKE_FRAMES = int(_os.environ.get('SMOKE_FRAMES', '64'))  # 64 = fast wiring smoke;
MAX_JOINTS = 144                          # 300-frame mem/throughput is validated separately
# via: torchrun --nproc_per_node=2 train_anytop_t2m_evaluator.py --num_frames 300 --max_steps 5 --bf16
TINY_B = 8

# Disabled SkeletonEncoder branches (use_clip / use_name_embed == False): trainable
# params that never enter the graph. FROZEN DistilBERT params have requires_grad=False
# so they are NOT in this set (the no-grad check only inspects requires_grad=True params).
_EXPECTED_NO_GRAD = {
    "motion_encoder.clip_proj.0.weight",
    "motion_encoder.clip_proj.0.bias",
    "motion_encoder.clip_proj.2.weight",
    "motion_encoder.clip_proj.2.bias",
    "motion_encoder.name_embedding.weight",
}

_results: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    _results.append((name, ok, detail))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" -- {detail}" if detail else ""))


def collate_indices(ds, idxs):
    return collate_fn([ds[i] for i in idxs])


def build_model(device, dropout=0.1):
    return AnyTopT2MEvaluator(
        coemb_dim=COEMB, text_tower="distilbert", distilbert_path=DISTILBERT,
        n_heads=8, d_ff=2048, n_graph_layers=6, n_temporal_layers=4, dropout=dropout,
        learnable_temperature=True, temperature=0.07,
    ).to(device)


# ============================================================================= #
# DDP gate (gate 7) — runs only under torchrun (WORLD_SIZE > 1)
# ============================================================================= #
def run_ddp_gate() -> int:
    import torch.distributed as dist
    import torch.distributed.nn as dist_nn
    ws = int(os.environ["WORLD_SIZE"])
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    dist.init_process_group(backend=backend)
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cpu")
    ok = True

    # Wrap in DDP so the gate tests the REAL trainer path (DDP forward + unused-param
    # handling + grad-safe all_gather), not just bare all_gather. `core` = unwrapped.
    from torch.nn.parallel import DistributedDataParallel as DDP
    core = build_model(device)
    core.train()
    ddp_kwargs = {"device_ids": [local_rank]} if torch.cuda.is_available() else {}
    model = DDP(core, find_unused_parameters=True, **ddp_kwargs)
    ds = AnyTopT2MEvalDataset(
        manifest_path=SPLITS / "val_all.json", data_root=DATA_ROOT,
        caption_emb_cache=None, split="val", view="full",
        num_frames=SMOKE_FRAMES, max_joints=MAX_JOINTS,
    )
    b = 4
    idxs = [(rank * b + k) % len(ds) for k in range(b)]
    coll = collate_indices(ds, idxs)
    coll = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in coll.items()}
    batch = GraphMotionBatch.from_collate_dict(coll)

    out = model(batch, coll["caption_text"])
    text_local = out.text_emb.float()
    motion_local = out.motion_emb.float()
    text_all = torch.cat(dist_nn.all_gather(text_local), dim=0)
    motion_all = torch.cat(dist_nn.all_gather(motion_local), dim=0)
    gathered = [None] * ws
    dist.all_gather_object(gathered, (coll["motion_id"], coll["source_motion_id"], coll["caption_text"]))
    mid = [x for r in gathered for x in r[0]]
    smid = [x for r in gathered for x in r[1]]
    cap = [x for r in gathered for x in r[2]]
    mask = build_false_negative_mask(mid, smid, cap)
    loss = symmetric_infonce(text_all, motion_all, core.logit_scale.float(), false_neg_mask=mask)
    loss.backward()

    # gate assertions (rank-0 prints)
    global_ok = (text_all.shape[0] == ws * b)
    finite_ok = bool(torch.isfinite(loss))
    # loss identical across ranks
    lt = loss.detach().clone()
    gl = [torch.zeros_like(lt) for _ in range(ws)]
    dist.all_gather(gl, lt)
    same_ok = all(bool(torch.allclose(x, gl[0], atol=1e-4)) for x in gl)
    # grads flow to local trainable params (motion encoder + text head + logit scale)
    grad_params = [p for n, p in core.named_parameters()
                   if p.requires_grad and n not in _EXPECTED_NO_GRAD]
    grad_ok = all(p.grad is not None and torch.isfinite(p.grad).all() for p in grad_params)
    ok = global_ok and finite_ok and same_ok and grad_ok
    if rank == 0:
        print(f"  [DDP gate ws={ws}] global_batch={text_all.shape[0]} (want {ws*b}) "
              f"loss={loss.item():.4f} finite={finite_ok} same_across_ranks={same_ok} "
              f"grads_flow={grad_ok}")
        print(f"  [{'PASS' if ok else 'FAIL'}] gate7 DDP global grad-safe InfoNCE")
    dist.barrier()
    dist.destroy_process_group()
    return 0 if ok else 1


# ============================================================================= #
# Single-process gates 1-6
# ============================================================================= #
def main() -> int:
    if int(os.environ.get("WORLD_SIZE", "1")) > 1:
        return run_ddp_gate()

    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 78)
    print(f"AnyTop T2M evaluator smoke (DistilBERT primary) on {device}")
    print(f"  data_root = {DATA_ROOT}")
    print(f"  distilbert = {DISTILBERT}")
    print("=" * 78)

    # (1) instantiate + param count -------------------------------------------- #
    print("\n[1] instantiate (frozen DistilBERT loads) + param count")
    model = build_model(device)
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    n_total = n_train + n_frozen
    print(f"    trainable={n_train/1e6:.2f}M frozen={n_frozen/1e6:.2f}M total={n_total/1e6:.2f}M")
    # frozen ~= DistilBERT backbone (~66M); trainable = motion encoder + text head + logit scale.
    check("frozen params ~= DistilBERT backbone (60-70M)", 60e6 <= n_frozen <= 70e6,
          f"frozen={n_frozen/1e6:.2f}M")
    check("trainable params present (>5M)", n_train > 5e6, f"trainable={n_train/1e6:.2f}M")

    ds = AnyTopT2MEvalDataset(
        manifest_path=SPLITS / "val_all.json", data_root=DATA_ROOT,
        caption_emb_cache=None, split="val", view="full",
        num_frames=SMOKE_FRAMES, max_joints=MAX_JOINTS,
    )
    coll = collate_indices(ds, list(range(TINY_B)))
    coll = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in coll.items()}

    # (2) forward shapes + finiteness ------------------------------------------ #
    print("\n[2] forward: text_emb[B,512] + motion_emb[B,512], finite, normalized")
    model.eval()
    batch = GraphMotionBatch.from_collate_dict(coll)
    with torch.no_grad():
        out = model(batch, coll["caption_text"])
    te, me = out.text_emb, out.motion_emb
    check("text_emb shape == (B,512)", tuple(te.shape) == (TINY_B, COEMB), f"got {tuple(te.shape)}")
    check("motion_emb shape == (B,512)", tuple(me.shape) == (TINY_B, COEMB), f"got {tuple(me.shape)}")
    check("text_emb finite", bool(torch.isfinite(te).all()))
    check("motion_emb finite", bool(torch.isfinite(me).all()))
    tn, mn = te.float().norm(dim=-1), me.float().norm(dim=-1)
    check("text_emb L2-normalized", bool(torch.allclose(tn, torch.ones_like(tn), atol=1e-3)),
          f"norms [{tn.min():.4f},{tn.max():.4f}]")
    check("motion_emb L2-normalized", bool(torch.allclose(mn, torch.ones_like(mn), atol=1e-3)),
          f"norms [{mn.min():.4f},{mn.max():.4f}]")

    # (3) InfoNCE finite + backward grad set ----------------------------------- #
    print("\n[3] InfoNCE finite + backward; grad set (frozen DistilBERT excluded)")
    model.train()
    out = model(batch, coll["caption_text"])
    mask = build_false_negative_mask(coll["motion_id"], coll["source_motion_id"], coll["caption_text"])
    loss = model.contrastive_loss(out, false_neg_mask=mask)
    check("InfoNCE loss finite", bool(torch.isfinite(loss)), f"loss={loss.item():.4f}")
    model.zero_grad(set_to_none=True)
    loss.backward()
    no_grad = {n for n, p in model.named_parameters() if p.requires_grad and p.grad is None}
    with_grad = {n for n, p in model.named_parameters() if p.requires_grad and p.grad is not None}
    n_trainable = sum(1 for p in model.parameters() if p.requires_grad)
    check("text head (masked-mean MLP) got grad",
          any(n.startswith("text_distilbert.head") and p.grad is not None
              for n, p in model.named_parameters()))
    check("frozen DistilBERT got NO grad (requires_grad=False)",
          all(not p.requires_grad for n, p in model.named_parameters()
              if n.startswith("text_distilbert.text_model")))
    check("no-grad trainable params == disabled clip/name encoder branches",
          no_grad == _EXPECTED_NO_GRAD, f"no_grad={sorted(no_grad)}")
    check("all other trainable params got grad",
          len(with_grad) == n_trainable - len(_EXPECTED_NO_GRAD),
          f"{len(with_grad)}/{n_trainable} trainable got grad")

    # (4) false-negative mask logic -------------------------------------------- #
    print("\n[4] false-negative mask logic")
    mid = ["mA", "mB", "mC", "mD", "mE", "mE"]
    smid = ["s0", "s9", "s0", "s7", "s4", "s5"]
    cap = ["c0", "cX", "c2", "cX", "c4", "c5"]
    m = build_false_negative_mask(mid, smid, cap)
    expected = {(0, 2), (2, 0), (1, 3), (3, 1), (4, 5), (5, 4)}
    got = {(i, j) for i in range(6) for j in range(6) if bool(m[i, j])}
    check("mask marks exactly shared off-diagonal pairs", got == expected,
          f"got {sorted(got)}")
    check("mask diagonal all False", not bool(m.diagonal().any()))
    torch.manual_seed(1)
    D = 16
    te6 = torch.nn.functional.normalize(torch.randn(6, D), dim=-1)
    me6 = torch.nn.functional.normalize(torch.randn(6, D), dim=-1)
    me6[2] = me6[0].clone(); te6[2] = te6[0].clone()
    scale = torch.tensor(20.0)
    lu = symmetric_infonce(te6, me6, scale, false_neg_mask=None)
    lm = symmetric_infonce(te6, me6, scale, false_neg_mask=m)
    check("masking duplicate pair lowers loss", float(lm) < float(lu),
          f"masked {lm:.4f} < unmasked {lu:.4f}")

    # (5) tiny overfit on content-distinct real motions ------------------------ #
    print("\n[5] tiny overfit (content-distinct real motions) -> loss drops")
    del out, loss, batch, model
    import gc
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    ofit = build_model(device, dropout=0.0)
    ofit.train()
    spread = [int(round(k * (len(ds) - 1) / (TINY_B - 1))) for k in range(TINY_B)]
    co = collate_indices(ds, spread)
    co = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in co.items()}
    fmask = build_false_negative_mask(co["motion_id"], co["source_motion_id"], co["caption_text"])
    check("overfit batch content-distinct (empty false-neg mask)", int(fmask.sum()) == 0,
          f"{int(fmask.sum())} shared pairs among {spread}")
    fixed = GraphMotionBatch.from_collate_dict(co)
    caps_fixed = list(co["caption_text"])
    # lr 2e-4 (NOT 2e-3/5e-4): the 512-d/6-graph/4-temporal model destabilizes at higher lr
    # (2e-3 collapses to the uniform ln(B) fixed point; 5e-4 oscillates over a short window).
    # lr 2e-4 (the real-run lr) overfits cleanly to ~0 (diag confirmed 2.04->0.000 over 200 steps,
    # no warmup here unlike the real run). 200 steps so the smoke window covers convergence.
    opt = torch.optim.AdamW([p for p in ofit.parameters() if p.requires_grad], lr=2e-4, weight_decay=0.0)
    losses = []
    for step in range(200):
        o = ofit(fixed, caps_fixed)
        l = ofit.contrastive_loss(o, false_neg_mask=fmask)
        opt.zero_grad(set_to_none=True)
        l.backward()
        torch.nn.utils.clip_grad_norm_([p for p in ofit.parameters() if p.requires_grad], 1.0)
        opt.step()
        losses.append(l.item())
        if step % 25 == 0 or step == 199:
            print(f"    step {step:3d}: loss {l.item():.4f}", flush=True)
    first, last, best = losses[0], losses[-1], min(losses)
    check("tiny-overfit loss strictly decreased", last < first, f"{first:.4f} -> {last:.4f}")
    check("tiny-overfit loss dropped >50% toward 0", best < 0.5 * first,
          f"best {best:.4f} < {0.5*first:.4f}")

    # (6) caption-shuffle raises the loss on the fitted batch ------------------ #
    print("\n[6] caption-shuffle on the overfit batch -> loss rises")
    ofit.eval()
    with torch.no_grad():
        o_true = ofit(fixed, caps_fixed)
        l_true = float(ofit.contrastive_loss(o_true, false_neg_mask=fmask))
        # derangement-ish shuffle (roll by 1 so no caption stays in place)
        caps_shuf = caps_fixed[1:] + caps_fixed[:1]
        o_shuf = ofit(fixed, caps_shuf)
        l_shuf = float(ofit.contrastive_loss(o_shuf, false_neg_mask=fmask))
    check("shuffled captions raise the fitted loss", l_shuf > l_true,
          f"true {l_true:.4f} -> shuffled {l_shuf:.4f}")

    print("\n" + "=" * 78)
    n_pass = sum(1 for _, ok, _ in _results if ok)
    print(f"SUMMARY: {n_pass}/{len(_results)} checks passed "
          f"(gate 7 DDP runs separately via torchrun --nproc_per_node=2)")
    if n_pass != len(_results):
        print("FAILURES:")
        for name, ok, detail in _results:
            if not ok:
                print(f"  - {name} :: {detail}")
    print("=" * 78)
    return 0 if n_pass == len(_results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
