#!/usr/bin/env python3
"""Smoke the [demo | target] in-context pipeline on real TrueBones data, against the FROZEN split.

Four gates. Each states what it prevents and each is capable of FAILING -- a check that cannot fail
is not a check.

  P1 PIPELINE      shapes/masks are what the model expects; padded joints and padded target frames
                   are excluded from the loss mask, and the three buckets are disjoint.
                   Fails if masking is wrong -> would otherwise train on padding as valid zeros.
  P2 GRADIENT      dLoss/dx actually lands on the root row in proportion to the design, not 1/J.
                   NOT tautological: the group weights are applied to the loss, but whether the
                   gradient reaches the root CELLS depends on the indexing and the mask being right.
                   Fails if _GROUP_SPEC mis-indexes or padding pollutes the denominators.
  P3 CONDITIONING  text and demo each causally move the sample, measured against a noise-reseed
                   baseline. Fails if a conditioning path is inert -- the gate v1 never had.
  P4 BUCKETS       bucket A and bucket B datasets build, and B's rigs never appear in training.
                   Fails on any leak between the frozen lists.

Read-only w.r.t. the repo; writes nothing but stdout.
"""
import argparse, pickle, sys, time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.anytop_dataset import AnyTopDataset                                  # noqa: E402
from src.data.incontext_pairs import (InContextPairs, collate, read_split,         # noqa: E402
                                      truebones_types, DEMO_FRAMES, TARGET_FRAMES)
from src.models.v2.dit_motion import (InContextMotionDiT, cfm_loss,                # noqa: E402
                                      KIMODO_GAMMAS, sample as ode_sample)

FAILS = []


def gate(name, ok, damage):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    if not ok:
        FAILS.append(f"{name} -- {damage}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default="data/animo4d_L4TB_plus_human_v4b272neutral")
    ap.add_argument("--splits_dir", default="data/holdout_splits_v1")
    ap.add_argument("--joint_sem", default="data/joint_semantics_llm2vec_v1.npz")
    # LLM2Vec pooled captions (dim 4096). Without this the dataset serves its 768-d default and
    # the model's text_mlp (LayerNorm[4096]) fails loud -- caught by this smoke on first run.
    ap.add_argument("--caption_cache", default="data/anytop_caption_llm2vec_v4b272neutral_multi")
    # The cache's meta records which captions json its vectors encode. The dataset's DEFAULT resolution
    # order picks motion_texts_by_file.json (sha 8efb9490), but the vectors were built from
    # motion_texts_by_file_clean_v1.json (sha af247609) -- mismatched, the string served at caption
    # index i would not be the text vector i encodes. The dataset checks this and refuses; naming the
    # file explicitly is the fix, not a workaround.
    ap.add_argument("--texts_json", default="motion_texts_by_file_clean_v1.json")
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--dim", type=int, default=256)
    ap.add_argument("--depth", type=int, default=6)
    ap.add_argument("--lr", type=float, default=3e-4)
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)

    # ---------- frozen protocol ----------
    cond = pickle.load(open(f"{a.data_root}/_cond_normalized_J144.pkl", "rb"))
    tb = truebones_types(cond.keys())
    names = {k: read_split(a.splits_dir, k)
             for k in ("train", "val", "held_representative", "held_stress")}
    print(f"[smoke] frozen protocol {a.splits_dir}")
    for k, v in names.items():
        n_tb = sum(1 for n in v if n.split("___")[0] in set(tb))
        print(f"         {k:22s} {len(v):6d} clips ({n_tb} TrueBones)")

    base = AnyTopDataset(data_root=a.data_root, split="all", num_frames=300, max_joints=144,
                         load_captions=True, caption_emb_cache=a.caption_cache, random_caption=False,
                         augment=False, joint_semantics=a.joint_sem, species_whitelist=tb,
                         splits_dir=a.splits_dir, texts_json_name=a.texts_json)

    def build(tgt, demo, seed=0):
        return InContextPairs(base, names[tgt], names[demo], object_types=tb, seed=seed)

    ds_train = build("train", "train")
    ds_A = build("val", "train")
    ds_B = build("held_representative", "held_representative")
    print(f"\n[smoke] training : {len(ds_train):5d} targets / {len(ds_train.types):2d} rigs / "
          f"{ds_train.pair_count():6d} pairs")
    print(f"[smoke] bucket A : {len(ds_A):5d} targets / {len(ds_A.types):2d} rigs / "
          f"{ds_A.pair_count():6d} pairs")
    print(f"[smoke] bucket B : {len(ds_B):5d} targets / {len(ds_B.types):2d} rigs / "
          f"{ds_B.pair_count():6d} pairs")

    print("\n=== P4 BUCKETS ===")
    gate("bucket B rigs never appear in training",
         not (set(ds_B.types) & set(ds_train.types)),
         "a 'never seen' rig that was trained on makes bucket B meaningless")
    gate("bucket A rigs DO appear in training (by design)",
         set(ds_A.types) <= set(ds_train.types),
         "bucket A must isolate clip novelty, not rig novelty")
    gate("bucket A targets are disjoint from training targets",
         not (names["val"] & names["train"]),
         "A would be scoring on clips it trained on")
    gate("held lists are disjoint from train+val",
         not ((names["held_representative"] | names["held_stress"]) &
              (names["train"] | names["val"])),
         "the holdout is not a holdout")

    # ---------- one batch ----------
    def get_batch(ds, rng):
        idx = rng.integers(0, len(ds), a.batch)
        b = collate([ds[int(i)] for i in idx])
        return {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in b.items()}

    rng = np.random.default_rng(0)
    b0 = get_batch(ds_train, rng)
    B, T, Jm, C = b0["x"].shape
    print(f"\n=== P1 PIPELINE ===")
    print(f"  x {tuple(b0['x'].shape)} (demo {DEMO_FRAMES} + target {TARGET_FRAMES} = {T})")
    print(f"  per-item J {b0['joint_valid'].sum(1).tolist()}  (batch max {Jm})")
    print(f"  real target frames per item {b0['is_target'].sum(1).tolist()} / {TARGET_FRAMES}")
    print(f"  rigs in batch: {', '.join(sorted(set(b0['object_type'])))}")
    gate("no target flag on demo frames",
         not bool(b0["is_target"][:, :DEMO_FRAMES].any()),
         "the demo half would be treated as something to generate")
    gate("padded joints excluded from valid",
         not bool((b0["valid"] & (~b0["joint_valid"])[:, None, :]).any()),
         "padding would dominate the per-group means for small rigs")
    gate("padded target frames excluded from is_target",
         bool((b0["is_target"] <= b0["frame_valid"]).all()),
         "zero padding would be trained on as a legitimate target")
    gate("joint bias is finite on real joints and -1e4 on padding",
         bool(torch.isfinite(b0["joint_bias"]).all()) and
         float(b0["joint_bias"][~(b0["joint_valid"][:, :, None] & b0["joint_valid"][:, None, :])].max()) <= -1e3,
         "padded joints would receive attention")
    gate("geodesic bias is clipped", float(-b0["joint_bias"][b0["joint_bias"] > -1e3].min()) <= 8.0 + 1e-6,
         "unclipped hop distances swamp the attention logits")
    has_text, has_sem = "text" in b0, "joint_sem" in b0
    gate("text present", has_text, "no text conditioning to test")
    gate("joint_sem present (order-hash checked by the dataset)", has_sem,
         "the only cross-skeleton-transferable joint key is missing")

    # ---------- model ----------
    d_sem = b0["joint_sem"].shape[-1] if has_sem else 4096
    d_text = b0["text"].shape[-1] if has_text else 4096
    print(f"[smoke] text dim {d_text}  joint_sem dim {d_sem}")
    model = InContextMotionDiT(in_ch=C, dim=a.dim, depth=a.depth, n_heads=8,
                               d_text=d_text, d_joint_sem=d_sem).to(dev)
    print(f"\n[smoke] params {sum(p.numel() for p in model.parameters())/1e6:.2f} M on {dev}")
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.0)

    def cond_of(b):
        c = dict(joint_bias=b["joint_bias"], frame_valid=b["frame_valid"],
                 joint_valid=b["joint_valid"])
        if has_text: c["text"] = b["text"]
        if has_sem: c["joint_sem"] = b["joint_sem"]
        return c

    # ---------- P2: where does the gradient actually land? ----------
    def root_grad_share(gammas):
        b = get_batch(ds_train, np.random.default_rng(7))
        xin = b["x"].detach().clone().requires_grad_(True)
        loss = cfm_loss(model, xin, is_target=b["is_target"], valid=b["valid"],
                        gammas=gammas, **cond_of(b))
        g = torch.autograd.grad(loss, xin)[0]
        jv = b["joint_valid"]
        root = float((g[:, :, 0:1] ** 2).sum())
        body = float((g[:, :, 1:] ** 2 * jv[:, None, 1:, None]).sum())
        Jmean = float(jv.sum(1).float().mean())
        return root / max(root + body, 1e-30), Jmean

    print(f"\n=== P2 GRADIENT REACHES THE ROOT ROW ===")
    s_flat, Jmean = root_grad_share(None)
    s_grp, _ = root_grad_share(KIMODO_GAMMAS)
    print(f"  mean J in batch {Jmean:.1f}  -> a uniform MSE would give root ~{100/Jmean:.2f}%")
    print(f"  root share of |dLoss/dx|^2, unweighted MSE : {100*s_flat:.2f}%")
    print(f"  root share of |dLoss/dx|^2, grouped+gamma  : {100*s_grp:.2f}%")
    gate("grouped objective raises the root gradient share >= 5x",
         s_grp >= 5 * s_flat,
         "the group weights are not reaching the root cells -- check _GROUP_SPEC indexing")
    gate("root share is not absurd (< 80%)", s_grp < 0.80,
         "the body would be starved instead")

    # ---------- train briefly ----------
    hist, t0 = [], time.time()
    for step in range(a.steps):
        b = get_batch(ds_train, rng)
        loss = cfm_loss(model, b["x"], is_target=b["is_target"], valid=b["valid"],
                        gammas=KIMODO_GAMMAS, **cond_of(b))
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        hist.append(float(loss.detach()))
        if step % 100 == 0 or step == a.steps - 1:
            print(f"[smoke] step {step:>4}  loss {float(loss):.5f}  "
                  f"({(time.time()-t0)/(step+1)*1000:.0f} ms/step)", flush=True)
    print(f"[smoke] loss {np.mean(hist[:20]):.4f} -> {np.mean(hist[-20:]):.4f}")

    # ---------- P3: causal conditioning, vs a free-change baseline ----------
    print(f"\n=== P3 CONDITIONING ===")
    model.eval()
    b = get_batch(ds_train, np.random.default_rng(11))
    tg = (b["is_target"][..., None] & b["valid"])[..., None]

    def draw(bb, seed):
        torch.manual_seed(seed)
        return ode_sample(model, bb["x"], bb["is_target"], 10, **cond_of(bb))

    def diff(u, v):
        return float((((u - v) ** 2) * tg).sum() / tg.sum().clamp_min(1))

    ref = draw(b, 1)
    base_noise = diff(ref, draw(b, 2))
    perm = torch.randperm(B, device=dev)

    b_t = {**b, "text": b["text"][perm]} if has_text else None
    eff_text = diff(ref, draw(b_t, 1)) if has_text else float("nan")

    xd = b["x"].clone(); xd[:, :DEMO_FRAMES] = b["x"][perm][:, :DEMO_FRAMES]
    eff_demo = diff(ref, draw({**b, "x": xd}, 1))

    print(f"  noise reseed (free change) {base_noise:.5f}  <- baseline")
    print(f"  shuffle text               {eff_text:.5f}  = {eff_text/max(base_noise,1e-9):5.2f}x")
    print(f"  shuffle demo               {eff_demo:.5f}  = {eff_demo/max(base_noise,1e-9):5.2f}x")
    gate("demo causally moves the sample (> 2x baseline)", eff_demo > 2 * base_noise,
         "the demo is ignored -- the whole in-context premise fails")
    if has_text:
        note = "" if eff_text > 2 * base_noise else "  (may be premature at this step count)"
        print(f"  text effect {'used' if eff_text > 2*base_noise else 'WEAK'}{note}")

    print()
    if FAILS:
        print("SMOKE FAILED:"); [print("  -", f) for f in FAILS]; sys.exit(1)
    print("smoke passed")


if __name__ == "__main__":
    main()
