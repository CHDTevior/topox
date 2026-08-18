#!/usr/bin/env python3
"""v2 prototype sanity gate: can the model OVERFIT a handful of clips?

This is deliberately the cheapest possible falsification. A correct conditional generative model
must be able to drive the loss on 8 clips to near zero and reproduce them at sampling time. If it
cannot, the architecture or the plumbing is broken and every downstream experiment is wasted -- which
is exactly what happened in v1, where 300 epochs were steered by a metric that could not see the
defect.

GATES (pre-registered, checked automatically):
  G1  loss falls by >= 100x from its first-100-step average          -> optimisation works at all
  G2  final loss < 1% of the variance of the data being fit          -> it really did fit
  G3  sampled motion reconstructs the held training clips:
        relative L2 < 0.15 in normalised space                       -> the sampler matches training
  G4  CONDITIONING IS USED: swapping in another clip's text/blueprint
        changes the sample by >> the noise-reseed difference          -> conditioning is not ignored

G4 is the one that matters most. v1's text path was dead for 300 epochs and nobody noticed; this
gate would have caught it in minutes.
"""
import argparse, json, sys, time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.models.v2.dit_motion import InContextMotionDiT, cfm_loss   # noqa: E402


def build_batch(ds, idxs, T, dev):
    """Pack a few clips into [B,T,J,13] plus skeleton bias / masks / joint semantics."""
    xs, sems, jb, jv = [], [], [], []
    Jm = 0
    items = []
    for i in idxs:
        it = ds[i]
        J = int(it["num_joints"]); Jm = max(Jm, J); items.append((it, J))
    for it, J in items:
        Tc = int(it["num_frames"])
        x = np.asarray(it["anytop_x"])[:J, :, :T]                    # [J,13,T] normalised
        if Tc < T:                                                    # pad short clips by repeating last
            x = np.concatenate([x, np.repeat(x[:, :, -1:], T - Tc, axis=2)], axis=2) if Tc > 0 else x
        x = x[:, :, :T].transpose(2, 0, 1)                            # [T,J,13]
        xp = np.zeros((T, Jm, 13), dtype=np.float32); xp[:, :J] = x
        xs.append(xp)
        s = np.asarray(it["joint_semantics"])[:J] if it.get("joint_semantics") is not None \
            else np.zeros((J, 4096), np.float32)
        sp = np.zeros((Jm, s.shape[-1]), np.float32); sp[:J] = s
        sems.append(sp)
        g = np.asarray(it["geodesic_dist"])[:J, :J]
        b = np.full((Jm, Jm), -1e4, np.float32)
        b[:J, :J] = -np.clip(g, 0, 8).astype(np.float32)              # closer joints attend more
        jb.append(b)
        m = np.zeros((Jm,), bool); m[:J] = True; jv.append(m)
    T_ = xs[0].shape[0]
    return (torch.tensor(np.stack(xs), device=dev),
            torch.tensor(np.stack(sems), device=dev),
            torch.tensor(np.stack(jb), device=dev),
            torch.tensor(np.stack(jv), device=dev),
            torch.ones(len(idxs), T_, dtype=torch.bool, device=dev))


@torch.no_grad()
def sample(model, x_ref, is_target, steps, **cond):
    """Euler ODE from noise to data on target frames; demo frames stay clean throughout."""
    x = x_ref.clone()
    noise = torch.randn_like(x)
    x = torch.where(is_target[..., None, None], noise, x_ref)
    for i in range(steps):
        t = torch.full((x.shape[0],), i / steps, device=x.device)
        v = model(x, t, is_target=is_target, **cond)
        x = torch.where(is_target[..., None, None], x + v / steps, x_ref)
    return x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default="data/anytop_truebones")
    ap.add_argument("--n_clips", type=int, default=8)
    ap.add_argument("--frames", type=int, default=64)
    ap.add_argument("--dim", type=int, default=256)
    ap.add_argument("--depth", type=int, default=6)
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--sample_steps", type=int, default=20)
    ap.add_argument("--out", default="scratch/v2_overfit")
    a = ap.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    from src.data.anytop_dataset import AnyTopDataset
    ds = AnyTopDataset(data_root=a.data_root, split="all", num_frames=300, max_joints=144,
                       load_captions=False, caption_emb_cache=None, random_caption=False,
                       augment=False, joint_semantics="data/joint_semantics_llm2vec_v1.npz")
    # pick clips long enough to hold a demo half and a target half
    idxs = [i for i in range(len(ds)) if int(ds.samples[i].get("n_frames", 999)) >= a.frames][:a.n_clips] \
        if hasattr(ds, "samples") else list(range(a.n_clips))
    idxs = idxs[:a.n_clips] or list(range(a.n_clips))
    print(f"[overfit] {len(idxs)} clips, T={a.frames}, dim={a.dim}, depth={a.depth}", flush=True)

    x, sem, jb, jv, fv = build_batch(ds, idxs, a.frames, dev)
    B, T, J, C = x.shape
    print(f"[overfit] batch {tuple(x.shape)}  (B,T,J,C)", flush=True)

    # first half = demo (given clean), second half = target (generated)
    is_target = torch.zeros(B, T, dtype=torch.bool, device=dev); is_target[:, T // 2:] = True
    # per-clip distinct conditioning: a fake "text" and "blueprint" that IDENTIFY the clip.
    # If the model ignores conditioning it cannot separate clips that share a demo prefix,
    # which is what gate G4 detects.
    g = torch.Generator(device="cpu").manual_seed(0)
    text = torch.randn(B, 4096, generator=g).to(dev)
    bp = torch.randn(B, 16, generator=g).to(dev)

    model = InContextMotionDiT(in_ch=C, dim=a.dim, depth=a.depth, n_heads=8,
                               d_text=4096, d_blueprint=16, d_joint_sem=sem.shape[-1]).to(dev)
    n_par = sum(p.numel() for p in model.parameters())
    print(f"[overfit] params {n_par/1e6:.2f} M", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.0)

    cond = dict(joint_sem=sem, text=text, blueprint=bp, joint_bias=jb,
                frame_valid=fv, joint_valid=jv)
    hist, t0 = [], time.time()
    for step in range(a.steps):
        loss = cfm_loss(model, x, is_target=is_target, valid=jv[:, None, :].expand(B, T, J), **cond)
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        hist.append(float(loss))
        if step % 100 == 0 or step == a.steps - 1:
            print(f"[overfit] step {step:>5}  loss {float(loss):.6f}", flush=True)

    first = float(np.mean(hist[:100])); final = float(np.mean(hist[-50:]))
    model.eval()
    xs = sample(model, x, is_target, a.sample_steps, **cond)
    tgt = is_target[..., None, None] & jv[:, None, :, None]
    err = ((xs - x) ** 2 * tgt).sum() / tgt.sum().clamp_min(1)
    var = ((x - x.mean()) ** 2 * tgt).sum() / tgt.sum().clamp_min(1)
    rel = float((err / var.clamp_min(1e-9)).sqrt())

    # G4: does conditioning matter? compare (a) rolling the conditioning across clips
    # against (b) merely reseeding the noise.
    xs_perm = sample(model, x, is_target, a.sample_steps,
                     **{**cond, "text": text.roll(1, 0), "blueprint": bp.roll(1, 0)})
    xs_reseed = sample(model, x, is_target, a.sample_steps, **cond)
    d_cond = float((((xs_perm - xs) ** 2) * tgt).sum() / tgt.sum().clamp_min(1))
    d_seed = float((((xs_reseed - xs) ** 2) * tgt).sum() / tgt.sum().clamp_min(1))
    ratio = d_cond / max(d_seed, 1e-12)

    g1 = first / max(final, 1e-12)
    g2 = final * C / float(var)   # loss is per-channel; var is not. Match them.
    res = {"params_M": n_par/1e6, "first100": first, "final50": final,
           "G1_loss_drop_x": g1, "G2_final_over_var": g2, "G3_rel_l2": rel,
           "G4_cond_vs_seed": ratio, "wall_s": time.time()-t0,
           "PASS": bool(g1 >= 100 and g2 < 0.01 and rel < 0.15 and ratio > 3.0)}
    Path(a.out).mkdir(parents=True, exist_ok=True)
    (Path(a.out)/"result.json").write_text(json.dumps(res, indent=1))
    print("\n=== GATES ===")
    print(f"  G1 loss drop        {g1:>10.1f}x     (need >=100)      {'PASS' if g1>=100 else 'FAIL'}")
    print(f"  G2 final/var        {g2:>10.5f}      (need <0.01)      {'PASS' if g2<0.01 else 'FAIL'}")
    print(f"  G3 sample rel-L2    {rel:>10.4f}      (need <0.15)      {'PASS' if rel<0.15 else 'FAIL'}")
    print(f"  G4 cond/seed effect {ratio:>10.2f}x     (need >3)         {'PASS' if ratio>3 else 'FAIL'}")
    print(f"\n  OVERALL: {'PASS' if res['PASS'] else 'FAIL'}   ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
