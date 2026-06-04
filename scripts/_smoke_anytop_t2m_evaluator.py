"""M2 CPU smoke for the AnyTop T2M evaluator (no GPU, no VAE/denoiser ckpt).

Design doc: handoff/20260604_0121_anytop_t2m_evaluator_impl_proposal.md (§5-M2)
            handoff/20260604_0257_anytop_t2m_evaluator_impl_progress.md (M2 NEXT)

Verifies the 5 M2 smoke gates (user: SMOKE ONLY — no spare GPU to actually train):
  1. AnyTopT2MEvaluator instantiates + prints param count (~14M expected).
  2. tiny real batch (8 val_all samples, full view) -> text_emb[B,384] +
     motion_emb[B,384]; shapes correct, no NaN/Inf.
  3. symmetric multi-positive InfoNCE is finite + loss.backward() populates grads
     (params get non-None gradients).
  4. multi-positive false-negative mask logic: a hand-built batch with duplicate
     source_motion_id / caption_text masks exactly those off-diagonal pairs out of
     the negatives, keeps the diagonal, and removes them from the InfoNCE
     denominator (verified against an unmasked baseline). Also confirms any
     NATURAL duplicates in a real batch are masked.
  5. tiny overfit: 60 steps on one fixed batch of CONTENT-DISTINCT real motions
     (spread across species so the multi-positive mask is empty = full
     contrastive signal) drives the loss clearly down (the model can learn).
     NOTE: consecutive val_all motions are near-duplicates and the mask
     (correctly) removes them as negatives, leaving no signal — so the overfit
     MUST use distinct samples; this is asserted before the loop.

Run (CPU): /iridisfs/scratch/ts1v23/.conda/bin/python3.12 \
             scripts/_smoke_anytop_t2m_evaluator.py
On Iridis, run inside a managed Slurm step (e.g. `srun --jobid=<own_alloc>
--overlap --gres=gpu:0 --cpus-per-task=4 ...`): a bare ssh-spawned process gets
reaped mid-run by the node, whereas a managed step (taking ZERO gpu) runs to
completion without touching any training GPU.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.anytop_t2m_eval_dataset import AnyTopT2MEvalDataset, collate_fn
from src.models.graph_salad.batch import GraphMotionBatch
from src.models.graph_salad.t2m_evaluator import (
    AnyTopT2MEvaluator,
    build_multi_positive_mask,
    symmetric_infonce,
)

ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "data" / "anytop_planet_zoo_clean_L2"
SPLITS = DATA_ROOT / "eval_splits"
T5_CACHE = ROOT / "data" / "anytop_caption_t5_cleanL2_multi"
NUM_FRAMES, MAX_JOINTS = 260, 144
TINY_B = 8

_results: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    _results.append((name, ok, detail))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" -- {detail}" if detail else ""))


def collate_indices(ds: AnyTopT2MEvalDataset, idxs: list[int]) -> dict:
    return collate_fn([ds[i] for i in idxs])


def main() -> int:
    torch.manual_seed(0)
    device = torch.device("cpu")
    print("=" * 78)
    print("M2 smoke: AnyTopT2MEvaluator (CPU only)")
    print(f"  data_root = {DATA_ROOT}")
    print(f"  t5_cache  = {T5_CACHE}")
    print("=" * 78)

    # ------------------------------------------------------------------ #
    # (1) instantiate + param count
    # ------------------------------------------------------------------ #
    print("\n[1] instantiate AnyTopT2MEvaluator + param count")
    model = AnyTopT2MEvaluator(
        coemb_dim=384, n_heads=8, d_ff=1536,
        n_graph_layers=4, n_temporal_layers=2,
        learnable_temperature=True, temperature=0.07,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    n_text = sum(p.numel() for p in model.text_proj.parameters())
    n_motion = sum(p.numel() for p in model.motion_encoder.parameters())
    print(f"    total params = {n_params/1e6:.2f}M "
          f"(text_proj {n_text/1e6:.3f}M + motion_encoder {n_motion/1e6:.2f}M "
          f"+ logit_scale)")
    # ~14M target (design doc §1c-4 / M2 spec). Accept a generous band.
    check("param count in ~10-20M band", 10e6 <= n_params <= 20e6,
          f"got {n_params/1e6:.2f}M")

    # ------------------------------------------------------------------ #
    # dataset (val_all, full view) — tiny batch
    # ------------------------------------------------------------------ #
    print("\n[setup] AnyTopT2MEvalDataset(val_all, full view)")
    ds = AnyTopT2MEvalDataset(
        manifest_path=SPLITS / "val_all.json",
        data_root=DATA_ROOT,
        caption_emb_cache=T5_CACHE,
        split="val",
        view="full",
        num_frames=NUM_FRAMES,
        max_joints=MAX_JOINTS,
    )
    coll = collate_indices(ds, list(range(TINY_B)))

    # ------------------------------------------------------------------ #
    # (2) forward -> text_emb / motion_emb shapes + finiteness
    # ------------------------------------------------------------------ #
    print("\n[2] forward: text_emb[B,384] + motion_emb[B,384], finite")
    model.eval()
    batch = GraphMotionBatch.from_collate_dict(coll)
    with torch.no_grad():
        out = model(batch, coll["caption_emb"])
    te, me = out.text_emb, out.motion_emb
    check("text_emb shape == (B,384)", tuple(te.shape) == (TINY_B, 384),
          f"got {tuple(te.shape)}")
    check("motion_emb shape == (B,384)", tuple(me.shape) == (TINY_B, 384),
          f"got {tuple(me.shape)}")
    check("text_emb finite (no NaN/Inf)", bool(torch.isfinite(te).all()))
    check("motion_emb finite (no NaN/Inf)", bool(torch.isfinite(me).all()))
    # L2-normalized -> unit norm
    te_norm = te.norm(dim=-1)
    me_norm = me.norm(dim=-1)
    check("text_emb L2-normalized (||.||≈1)",
          bool(torch.allclose(te_norm, torch.ones_like(te_norm), atol=1e-4)),
          f"norms in [{te_norm.min():.4f},{te_norm.max():.4f}]")
    check("motion_emb L2-normalized (||.||≈1)",
          bool(torch.allclose(me_norm, torch.ones_like(me_norm), atol=1e-4)),
          f"norms in [{me_norm.min():.4f},{me_norm.max():.4f}]")

    # ------------------------------------------------------------------ #
    # (3) InfoNCE finite + backward populates grads
    # ------------------------------------------------------------------ #
    print("\n[3] InfoNCE finite + loss.backward() populates grads")
    model.train()
    out = model(batch, coll["caption_emb"])
    mask = build_multi_positive_mask(
        coll["motion_id"], coll["source_motion_id"], coll["caption_text"],
    )
    loss = model.contrastive_loss(out, false_neg_mask=mask)
    check("InfoNCE loss finite", bool(torch.isfinite(loss)),
          f"loss={loss.item():.4f}")
    model.zero_grad(set_to_none=True)
    loss.backward()
    # text_proj + motion_encoder + log_logit_scale must all receive grads.
    text_w = model.text_proj[0].weight
    motion_w = model.motion_encoder.joint_feat_proj[0].weight
    scale_g = model.log_logit_scale.grad
    check("text_proj.weight grad is not None + finite",
          text_w.grad is not None and bool(torch.isfinite(text_w.grad).all()))
    check("motion_encoder grad is not None + finite",
          motion_w.grad is not None and bool(torch.isfinite(motion_w.grad).all()))
    check("log_logit_scale grad is not None + finite",
          scale_g is not None and bool(torch.isfinite(scale_g).all()))
    # Every trainable param must receive a grad EXCEPT the SkeletonEncoder's
    # optional CLIP-name / hash-name branches, which are disabled by default
    # (encoder.use_clip / use_name_embed == False) and therefore never enter the
    # graph. This is inherited dead-weight from reusing SkeletonEncoder verbatim
    # (§1c-4 hard requirement: reuse it as-is; GraphMotionVAE carries the same
    # unused params). Asserting the no-grad set is EXACTLY those known-disabled
    # modules both confirms gradient flow everywhere it matters AND fails loud if
    # any *active* param silently detaches.
    _EXPECTED_NO_GRAD = {
        "motion_encoder.clip_proj.0.weight",
        "motion_encoder.clip_proj.0.bias",
        "motion_encoder.clip_proj.2.weight",
        "motion_encoder.clip_proj.2.bias",
        "motion_encoder.name_embedding.weight",
    }
    no_grad = {
        n for n, p in model.named_parameters()
        if p.requires_grad and p.grad is None
    }
    with_grad = {
        n for n, p in model.named_parameters()
        if p.requires_grad and p.grad is not None
    }
    n_total = sum(1 for p in model.parameters() if p.requires_grad)
    check(
        "no-grad params == exactly the disabled clip/name encoder branches",
        no_grad == _EXPECTED_NO_GRAD,
        f"no_grad={sorted(no_grad)}",
    )
    check(
        f"all active params got grads "
        f"({len(_EXPECTED_NO_GRAD)} disabled encoder branches excluded)",
        len(with_grad) == n_total - len(_EXPECTED_NO_GRAD),
        f"{len(with_grad)} with grad / {n_total} trainable, "
        f"{len(no_grad)} without",
    )

    # ------------------------------------------------------------------ #
    # (4) multi-positive mask logic
    # ------------------------------------------------------------------ #
    print("\n[4] multi-positive false-negative mask logic")
    # 4a — hand-built duplicate sets to pin the logic exactly.
    #   idx0,idx2 share source_motion_id ; idx1,idx3 share caption_text ;
    #   idx4,idx5 share motion_id (degenerate but allowed by the rule).
    mid = ["mA", "mB", "mC", "mD", "mE", "mE"]
    smid = ["s0", "s9", "s0", "s7", "s4", "s5"]   # s0: idx0&idx2
    cap = ["c0", "cX", "c2", "cX", "c4", "c5"]    # cX: idx1&idx3
    m = build_multi_positive_mask(mid, smid, cap)
    expected_pairs = {(0, 2), (2, 0), (1, 3), (3, 1), (4, 5), (5, 4)}
    got_pairs = {(i, j) for i in range(6) for j in range(6) if bool(m[i, j])}
    check("mask marks exactly the shared off-diagonal pairs",
          got_pairs == expected_pairs,
          f"got {sorted(got_pairs)} expected {sorted(expected_pairs)}")
    check("mask diagonal all False (positives kept)",
          not bool(m.diagonal().any()))

    # 4b — the mask actually removes those pairs from the InfoNCE denominator.
    #   Build embeddings where idx0 and idx2 (a masked false-neg pair) are
    #   identical: WITHOUT the mask that collision inflates the loss (idx2 is a
    #   hard negative for the idx0 text query); WITH the mask it is dropped, so
    #   the masked loss must be strictly lower.
    torch.manual_seed(1)
    D = 16
    te6 = torch.nn.functional.normalize(torch.randn(6, D), dim=-1)
    me6 = torch.nn.functional.normalize(torch.randn(6, D), dim=-1)
    me6[2] = me6[0].clone()          # motion2 == motion0 -> collision on row 0
    te6[2] = te6[0].clone()          # text2   == text0
    scale = torch.tensor(20.0)
    loss_unmasked = symmetric_infonce(te6, me6, scale, false_neg_mask=None)
    loss_masked = symmetric_infonce(te6, me6, scale, false_neg_mask=m)
    check("masking the duplicate pair lowers InfoNCE loss",
          float(loss_masked) < float(loss_unmasked),
          f"masked {loss_masked:.4f} < unmasked {loss_unmasked:.4f}")
    check("masked + unmasked losses both finite",
          bool(torch.isfinite(loss_masked)) and bool(torch.isfinite(loss_unmasked)))

    # 4c — natural duplicates in a real batch get masked. To make this a STRONG
    #   (non-trivial) test, deliberately choose a window of dataset indices known
    #   to contain a duplicate caption_text or source_motion_id. The metadata is
    #   read straight from the wrapper's plan (no motion load) so the selection is
    #   cheap; only the chosen indices are then collated (which reads motions).
    #   The mask's masked-entry count must match a brute-force recompute.
    meta = [
        (i, ds._plan[i][1]["motion_id"], ds._plan[i][1]["source_motion_id"],
         ds._plan[i][3])  # (idx, motion_id, source_motion_id, caption_text)
        for i in range(len(ds))
    ]
    # find the first caption_text (or source_motion_id) that repeats, take 2 of it
    from collections import defaultdict
    by_cap: dict[str, list[int]] = defaultdict(list)
    by_smid: dict[str, list[int]] = defaultdict(list)
    for i, _mid, smid, cap_t in meta:
        by_cap[cap_t].append(i)
        by_smid[smid].append(i)
    dup_idxs: list[int] = []
    dup_source = "none"
    for cap_t, idxs in by_cap.items():
        if len(idxs) >= 2:
            dup_idxs = idxs[:2]
            dup_source = f"caption_text={cap_t[:40]!r}"
            break
    if not dup_idxs:
        for smid, idxs in by_smid.items():
            if len(idxs) >= 2:
                dup_idxs = idxs[:2]
                dup_source = f"source_motion_id={smid!r}"
                break
    # Build a 16-sample window that includes the duplicate pair (pad with the
    # first distinct indices). If the val set somehow had zero duplicates, fall
    # back to the first 16 (test still validates mask == brute-force).
    sel = list(dict.fromkeys(dup_idxs + list(range(len(ds)))))[:16]
    bigcoll = collate_indices(ds, sel)
    bmask = build_multi_positive_mask(
        bigcoll["motion_id"], bigcoll["source_motion_id"], bigcoll["caption_text"],
    )
    n_masked = int(bmask.sum())
    B = len(sel)
    bf = 0
    for i in range(B):
        for j in range(B):
            if i == j:
                continue
            if (bigcoll["motion_id"][i] == bigcoll["motion_id"][j]
                    or bigcoll["source_motion_id"][i] == bigcoll["source_motion_id"][j]
                    or bigcoll["caption_text"][i] == bigcoll["caption_text"][j]):
                bf += 1
    print(f"    real {B}-sample batch (incl. dup {dup_source}): {n_masked} "
          f"off-diagonal false-negatives masked (brute-force {bf})")
    check("real-batch mask count == brute-force (with a real duplicate present)",
          n_masked == bf, f"mask {n_masked} vs brute {bf}")
    check("real-batch found >=1 natural duplicate to exercise the mask",
          n_masked >= 2 or dup_source == "none",
          f"dup_source={dup_source}, n_masked={n_masked}")

    # ------------------------------------------------------------------ #
    # (5) tiny overfit — loss drops on one fixed batch
    # ------------------------------------------------------------------ #
    n_overfit_steps = 60
    OFIT_FRAMES = 64  # short real clip (like the VAE's 64-frame train crop)
    print(f"\n[5] tiny overfit: {n_overfit_steps} steps on one fixed batch "
          f"(num_frames={OFIT_FRAMES}), loss drops")
    # Free the item-2/3 model + its retained autograd graph before allocating a
    # second model (keeps CPU peak memory ≈ one model's worth on the shared node).
    del out, loss, batch, model
    import gc
    gc.collect()
    # Build the overfit batch from REAL data but with a SHORT frame window
    # (num_frames=64 — the same crop length the VAE trains at). This goes through
    # the identical AnyTop preprocessing + the identical motion-encoder forward
    # path, just on fewer frames, so each CPU forward+backward is ~4x cheaper
    # (full T=260 overfit is needlessly heavy on CPU — items 1-4 already exercised
    # the full-resolution forward/backward).
    ds_ofit = AnyTopT2MEvalDataset(
        manifest_path=SPLITS / "val_all.json",
        data_root=DATA_ROOT,
        caption_emb_cache=T5_CACHE,
        split="val",
        view="full",
        num_frames=OFIT_FRAMES,
        max_joints=MAX_JOINTS,
    )
    # CRITICAL: overfit on CONTENT-DISTINCT samples. Consecutive val_all motions
    # are near-duplicates (same species, often the SAME caption), and the
    # multi-positive mask (correctly) removes shared-content pairs from the
    # negatives — leaving almost no contrastive signal, so the loss can't drop
    # (this is the mask working as designed, not the model failing to learn).
    # A faithful "can it learn" check therefore needs DISTINCT items: pick indices
    # spread across the dataset (different species + captions) and verify the
    # false-neg mask is empty (= every off-diagonal pair is a true negative). The
    # synthetic-tensor probe (scripts/_probe_overfit.py) separately confirms the
    # mechanism converges; here we confirm it on REAL distinct motions.
    spread = [int(round(k * (len(ds_ofit) - 1) / (TINY_B - 1))) for k in range(TINY_B)]
    coll_ofit = collate_indices(ds_ofit, spread)
    fixed_mask = build_multi_positive_mask(
        coll_ofit["motion_id"], coll_ofit["source_motion_id"],
        coll_ofit["caption_text"],
    )
    n_shared = int(fixed_mask.sum())
    check("overfit batch is content-distinct (empty false-neg mask)",
          n_shared == 0,
          f"{n_shared} shared-content off-diagonal pairs among spread indices "
          f"{spread}")
    # Fresh model so the prior backward state does not bias the run.
    ofit = AnyTopT2MEvaluator(
        coemb_dim=384, n_heads=8, d_ff=1536,
        n_graph_layers=4, n_temporal_layers=2,
        learnable_temperature=True, temperature=0.07, dropout=0.0,
    ).to(device)
    ofit.train()
    opt = torch.optim.AdamW(ofit.parameters(), lr=2e-3, weight_decay=0.0)
    fixed = GraphMotionBatch.from_collate_dict(coll_ofit)
    fixed_cap = coll_ofit["caption_emb"]
    losses: list[float] = []
    for step in range(n_overfit_steps):
        o = ofit(fixed, fixed_cap)
        l = ofit.contrastive_loss(o, false_neg_mask=fixed_mask)
        opt.zero_grad(set_to_none=True)
        l.backward()
        torch.nn.utils.clip_grad_norm_(ofit.parameters(), 1.0)
        opt.step()
        losses.append(l.item())
        if step % 5 == 0 or step == n_overfit_steps - 1:
            print(f"    step {step:2d}: loss {l.item():.4f}", flush=True)
    first, last = losses[0], losses[-1]
    best = min(losses)
    check("tiny-overfit loss strictly decreased",
          last < first, f"first {first:.4f} -> last {last:.4f}")
    # With 8 distinct items + identical fixed batch, the loss should fall well
    # below the log(B) random baseline (~2.08) — assert a clear drop (best seen),
    # proving the model genuinely fits the batch (not just minor noise).
    check("tiny-overfit loss dropped >50% of the way toward 0",
          best < 0.5 * first, f"best {best:.4f} < 0.5*first {0.5*first:.4f}")

    # ------------------------------------------------------------------ #
    # summary
    # ------------------------------------------------------------------ #
    print("\n" + "=" * 78)
    n_pass = sum(1 for _, ok, _ in _results if ok)
    n_total_c = len(_results)
    print(f"SUMMARY: {n_pass}/{n_total_c} checks passed")
    if n_pass != n_total_c:
        print("FAILURES:")
        for name, ok, detail in _results:
            if not ok:
                print(f"  - {name} :: {detail}")
    print("=" * 78)
    return 0 if n_pass == n_total_c else 1


if __name__ == "__main__":
    raise SystemExit(main())
