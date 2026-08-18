#!/usr/bin/env python3
"""Sanity ladder: raise difficulty one rung at a time so a failure localises to ONE cause.

The first attempt tested the hardest combination at once (8 clips + random 4096-d conditioning +
recovering a 16,640-d motion from a 256-d global modulation code) and failed without saying which
part was broken. Each rung below isolates a single capability:

  L1  1 clip, NO conditioning        -> can the network express one motion at all?
  L2  4 clips, ONE-HOT conditioning  -> can conditioning select among motions, given a clean signal?
  L3  4 clips, random 4096-d text    -> does it still work through the realistic text pathway?

A rung that fails while the previous passed names the culprit exactly.
"""
import sys, time
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
# Import the ONE sampler that lives with the model. This file previously carried its own
# copy, which stayed on the v-prediction update (x += v/steps) after the model moved to
# x-prediction -- so it integrated predicted CLEAN motion as if it were a velocity.
# Training was perfect (loss down 1475x, residual 0.0004) while sampling read relL2 0.69.
from src.models.v2.dit_motion import InContextMotionDiT, cfm_loss, sample   # noqa: E402


def batch(ds, idxs, T, dev):
    xs, jb, jv = [], [], []
    items = [(ds[i], int(ds[i]["num_joints"])) for i in idxs]
    Jm = max(J for _, J in items)
    for it, J in items:
        Tc = int(it["num_frames"])
        x = np.asarray(it["anytop_x"])[:J, :, :T]
        if Tc < T: x = np.concatenate([x, np.repeat(x[:, :, -1:], T - Tc, 2)], 2)
        xp = np.zeros((T, Jm, 13), np.float32); xp[:, :J] = x[:, :, :T].transpose(2, 0, 1)
        xs.append(xp)
        g = np.asarray(it["geodesic_dist"])[:J, :J]
        b = np.full((Jm, Jm), -1e4, np.float32); b[:J, :J] = -np.clip(g, 0, 8).astype(np.float32)
        jb.append(b); m = np.zeros((Jm,), bool); m[:J] = True; jv.append(m)
    return (torch.tensor(np.stack(xs), device=dev), torch.tensor(np.stack(jb), device=dev),
            torch.tensor(np.stack(jv), device=dev))


def rung(name, ds, n_clips, cond_kind, steps, dev, T=48, dim=192, depth=4, lr=5e-4):
    x, jb, jv = batch(ds, list(range(n_clips)), T, dev)
    B, T_, J, C = x.shape
    is_tgt = torch.zeros(B, T_, dtype=torch.bool, device=dev); is_tgt[:, T_ // 2:] = True
    d_text = 64
    if cond_kind == "none":
        text = torch.zeros(B, d_text, device=dev)
    elif cond_kind == "onehot":
        text = torch.zeros(B, d_text, device=dev)
        for b in range(B): text[b, b] = 10.0          # unmistakable, well-separated signal
    else:
        text = torch.randn(B, d_text, device=dev, generator=torch.Generator(dev).manual_seed(0))
    model = InContextMotionDiT(in_ch=C, dim=dim, depth=depth, n_heads=6,
                               d_text=d_text, d_blueprint=8, d_joint_sem=16).to(dev)
    cond = dict(text=text, joint_bias=jb, joint_valid=jv)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)
    vmask = jv[:, None, :].expand(B, T_, J)
    hist = []
    t0 = time.time()
    for s in range(steps):
        l = cfm_loss(model, x, is_target=is_tgt, valid=vmask, **cond)
        opt.zero_grad(set_to_none=True); l.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        hist.append(float(l))
    first, final = float(np.mean(hist[:50])), float(np.mean(hist[-50:]))
    model.eval()
    xs = sample(model, x, is_tgt, 20, **cond)
    tg = is_tgt[..., None, None] & jv[:, None, :, None]
    err = float(((xs - x) ** 2 * tg).sum() / tg.sum())
    var = float(((x - x.mean()) ** 2 * tg).sum() / tg.sum())
    rel = (err / max(var, 1e-9)) ** 0.5
    ok = (final * C / var < 0.05) and rel < 0.35
    print(f"  {name:<34} loss {first:.3f}->{final:.3f} ({first/max(final,1e-9):.0f}x)  "
          f"norm {final*C/var:.4f}  relL2 {rel:.3f}  {'PASS' if ok else 'FAIL'}  ({time.time()-t0:.0f}s)")
    return ok


if __name__ == "__main__":
    dev = torch.device("cuda")
    from src.data.anytop_dataset import AnyTopDataset
    ds = AnyTopDataset(data_root="data/anytop_truebones", split="all", num_frames=300,
                       max_joints=144, load_captions=False, caption_emb_cache=None,
                       random_caption=False, augment=False)
    print("=== SANITY LADDER (each rung isolates one capability) ===")
    r1 = rung("L1  1 clip, no conditioning", ds, 1, "none", 3000, dev)
    r2 = rung("L2  4 clips, one-hot cond", ds, 4, "onehot", 4000, dev)
    r3 = rung("L3  4 clips, random 64-d text", ds, 4, "rand", 4000, dev)
    print(f"\n  L1 {'PASS' if r1 else 'FAIL'} -> expressive power")
    print(f"  L2 {'PASS' if r2 else 'FAIL'} -> conditioning pathway")
    print(f"  L3 {'PASS' if r3 else 'FAIL'} -> realistic text pathway")
