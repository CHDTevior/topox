#!/usr/bin/env python
"""Zero-training probe: how much does the generated WORLD motion move when only the skeleton's rest convention moves?

A rest convention is a choice of per-joint rest rotation R_rest_j; the same world motion is served as different rest
deltas (delta_j Q_j^T) with a different rest-normalised mean and a different 1-frame rest demo. A model that reads the
motion instead of the convention returns the same world motion. The perturbation is the R2 training channel itself
(src/data/ktjd17_augment.py, mode one_of with p=0 / rest_p=1: no op, every joint with full rotation rows gets a random
axis and an angle U(0, rest_deg); world motion, tree, bone offsets unchanged), served through the SAME dataset class the
arms train and render with, so the served item is exactly what the model would see on a differently-authored rig.

For each target (bucket A: val targets, train demo = the rest frame) and each seed: the plain item and the item under
each --degs convention are sampled from the SAME noise (torch.manual_seed before each call), decoded to world space with
their own rest (decode_ktjd17 direct + FK, no temporal integration), and compared:
    d_rot   = mean_{t,j} |gen_rot - gen_plain| / mean bone length         (same seed; the convention's effect)
    d_seed  = mean_{t,j} |gen_plain(seed b) - gen_plain(seed a)| / mbl     (the sampler's own spread; the yardstick)
    d_gt    = mean_{t,j} |gen_plain - GT| / mbl                            (how far a generation is from the clip)
each for the direct (position-channel) and FK (rotation-channel) recoveries, full world and root-relative (root
subtracted per frame). Fail-loud checks: the rotated item is the same clip with identical de-normalised GT world
positions, its transformed rest decodes the GT to the plain GT's FK positions, the drawn angles stay <= rest_deg.

Writes <out>/probe.json (per-item records + medians), <out>/summary.txt, and for the largest angle side-by-side
gifs [GT | GEN plain (fk) | GEN rotated (fk)] of the first --n_gif items at the first seed.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.incontext_pairs import InContextPairs, collate                     # noqa: E402
from src.data.anytop_dataset import _STD_FLOOR                                     # noqa: E402
from src.data.ktjd17_incontext import Ktjd17Base, ktjd17_split_names               # noqa: E402
from src.data.ktjd17_augment import AugConfig                                      # noqa: E402
from src.data.ktjd17.decoder import decode_ktjd17                                  # noqa: E402
from src.models.v2.dit_motion import sample                                        # noqa: E402
from scripts._eval_v2_gen_in_evalspace import load_gen_model                       # noqa: E402
from scripts.v2_render_incontext import render_gif, world_of_ktjd                  # noqa: E402


def so3_project(R):
    """Nearest rotation to each [3,3] (polar); the item stores the transformed rest in float32, the codec wants 1e-10."""
    U, _, Vt = np.linalg.svd(R.astype(np.float64))
    out = U @ Vt
    neg = np.linalg.det(out) < 0
    if np.any(neg):
        U[neg, :, -1] *= -1
        out = U @ Vt
    return out


def rest_local_of(par, Rg):
    loc = Rg.copy()
    loc[1:] = np.einsum("jba,jbc->jac", Rg[par[1:]], Rg[1:])          # R_par^T R_j
    return loc


def decode_with_rest(raw17, sk, Rg, strict_gt):
    par = np.asarray(sk["parents"], dtype=np.int64)[: raw17.shape[1]]
    dec = decode_ktjd17(raw17, parents=par, R_rest_global=Rg, R_rest_local=rest_local_of(par, Rg),
                        offset_parent_local=np.asarray(sk["offset_parent_local"])[: raw17.shape[1]],
                        rotation_source_kind=np.asarray(sk["rotation_source_kind"])[: raw17.shape[1]], strict_gt=strict_gt)
    return dec.positions_direct, dec.positions_fk


def denorm(seg18, item, J):
    mu = item["anytop_mean"].numpy()[:J, :17].astype(np.float64)
    sd = item["anytop_std"].numpy()[:J, :17].astype(np.float64)
    return seg18[:, :J, :17].astype(np.float64) * (sd[None] + _STD_FLOOR) + mu[None]


def dist(a, b, mbl):
    full = float(np.linalg.norm(a - b, axis=-1).mean() / mbl)
    rel = float(np.linalg.norm((a - a[:, :1]) - (b - b[:, :1]), axis=-1).mean() / mbl)
    return full, rel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--degs", default="10,20,30")
    ap.add_argument("--seeds", default="7,17")
    ap.add_argument("--n_items", type=int, default=24)
    ap.add_argument("--n_gif", type=int, default=6)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--cfg_text", type=float, default=2.0)
    ap.add_argument("--aug_seed", type=int, default=0, help="rng seed of the rest-convention draws (the dataset's seed)")
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    degs = [float(x) for x in a.degs.split(",") if x.strip()]
    seeds = [int(x) for x in a.seeds.split(",") if x.strip()]
    if len(seeds) < 2:
        raise SystemExit("[refuse] two seeds at least: the sampler's own spread is the yardstick")

    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    model, ca = load_gen_model(ck, dev)
    if str(ca.get("anchor", "none")) != "none":
        raise SystemExit("[refuse] anchored checkpoints are not wired here")
    if not bool(ca.get("demo_rest", False)) or int(ca.get("demo_frames", 0)) != 1:
        raise SystemExit("[refuse] this probe assumes the 1-frame rest demo (the demo itself follows the convention)")
    if str(ca.get("rep_norm", "percell")) != "rest":
        raise SystemExit("[refuse] this probe assumes rest normalisation (the served mean follows the convention)")
    base = Ktjd17Base(ca["ktjd_root"], caption_emb_cache=ca["caption_cache"], joint_semantics=ca["joint_sem"],
                      texts_json=ca["texts_json"], percell_stats=ca.get("ktjd_percell_stats", "data/ktjd17_percell_stats_v1.npz"),
                      exclude_clips=ca.get("exclude_clips") or None, normalization="rest")
    names = ktjd17_split_names(ca["ktjd_root"], exclude=ca.get("exclude_clips") or None)
    PK = dict(demo_rest=True, emit_ref_text=bool(ca.get("ref_text", False)),
              rest_demo_self_pairs=bool(ca.get("rest_demo_self_pairs", False)),
              demo_frames=1, target_frames=int(ca["target_frames"]), emit_fk_fields=True,
              emit_graph_v2=bool(ca.get("struct_feats", False)) or bool(ca.get("dir_bias", False)),
              struct_world_rest=bool(ca.get("struct_world_rest", False)),
              emit_spectral=(int(ca.get("spec_rope_k", 8)) if bool(ca.get("spec_rope", False)) else 0),
              spectral_hks=bool(ca.get("spec_rope_hks", False)), balance_skeletons=False, seed=a.aug_seed)
    ds0 = InContextPairs(base, names["val"], names["train"], **PK)
    aug_base = dict(mode="one_of", p=0.0, drop_mode="tips", bone_scale=0.1, rest_p=1.0)    # no op ever drawn (p=0); rest always
    dsD = {d: InContextPairs(base, names["val"], names["train"], augment=AugConfig(**aug_base, rest_deg=d), **PK) for d in degs}
    n = len(ds0)
    idx = [int(round(k)) for k in np.linspace(0, n - 1, min(a.n_items, n))]
    print(f"[probe] ckpt {a.ckpt} epoch {ck.get('epoch')} | bucket A {n} pairs, {len(idx)} probed | degs {degs} | seeds {seeds}", flush=True)

    def gen(item, seed):
        b = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in collate([item]).items()}
        kw = {k: b[k] for k in ("struct_feats", "updown", "spectral_feats", "demo_text") if k in b}
        rig = item["object_type"]
        cvj = torch.from_numpy(np.asarray(base.static_masks(rig)["channel_valid"], dtype=bool)).to(dev)
        cv = torch.zeros(1, b["x"].shape[2], 17, dtype=torch.bool, device=dev); cv[0, :cvj.shape[0]] = cvj
        if "channel_valid" in b and not torch.equal(b["channel_valid"][0, :cvj.shape[0]], cvj):
            raise SystemExit("[refuse] the served item's channel mask differs from the rig's static mask (a rest-only item keeps the tree)")
        torch.manual_seed(seed)
        with torch.no_grad():
            g = sample(model, b["x"][..., :17].contiguous(), b["is_target"], a.steps, cfg_text=a.cfg_text, demo_frames=1,
                       joint_bias=b["joint_bias"], frame_valid=b["frame_valid"], joint_valid=b["joint_valid"],
                       text=b["text"], joint_sem=b["joint_sem"], channel_valid=cv, heading_valid=b["x"][:, :, 0, 17] > 0.5, **kw)
        return g[0].float().cpu().numpy()

    recs = []
    for k, i in enumerate(idx):
        ds0._wrng_key = None
        it0 = ds0[i]
        rig, J = it0["object_type"], int(it0["n_joints"])
        sk = base.skeleton(rig)
        par = np.asarray(sk["parents"], dtype=np.int64)[:J]
        R0 = np.asarray(sk["R_rest_global"], dtype=np.float64)[:J]
        if np.abs(rest_local_of(par, R0) - np.asarray(sk["R_rest_local"])[:J]).max() > 1e-9:
            raise SystemExit(f"[refuse] {rig}: R_rest_local != R_rest_global[parent]^T R_rest_global (the probe's local rule is wrong)")
        mbl = float(np.linalg.norm(np.asarray(sk["offset_parent_local"])[1:J], axis=-1).mean())
        t_real = int(it0["frame_valid"][1:].sum())
        gt0 = it0["x"].numpy()[1:1 + t_real]
        gt_w, gt_fk = world_of_ktjd(gt0[:, :J], base, rig, strict_gt=True)
        items = {}
        for d in degs:
            dsD[d].seed = int(a.aug_seed) + 7919 * (i + 1)        # a fresh rest draw per (item, angle): resetting to the same
            dsD[d]._wrng_key = None                                 # stream gave every item the same Q sequence (reviewer P2-4)
            itd = dsD[d][i]
            if itd["motion_id"] != it0["motion_id"] or int(itd["n_joints"]) != J or "R_rest_global" not in itd:
                raise SystemExit(f"[refuse] item {i}: the {d} deg dataset served another clip / tree ({itd['motion_id']} vs {it0['motion_id']})")
            Rd = so3_project(itd["R_rest_global"].numpy()[:J])
            ang = np.degrees(np.arccos(np.clip((np.einsum("jii->j", np.einsum("jab,jcb->jac", Rd, R0)) - 1) / 2, -1, 1)))
            if ang.max() > d + 1e-3:
                raise SystemExit(f"[refuse] item {i}: drawn rest angle {ang.max():.2f} > {d}")
            gtd = itd["x"].numpy()[1:1 + t_real]
            gtd_w, gtd_fk = decode_with_rest(denorm(gtd, itd, J), sk, Rd, strict_gt=True)
            e_w, e_fk = np.abs(gtd_w - gt_w).max() / mbl, np.abs(gtd_fk - gt_fk).max() / mbl
            if e_w > 1e-3 or e_fk > 1e-3:
                raise SystemExit(f"[refuse] item {i} {d} deg: GT world positions moved under the convention (direct {e_w:.2e}, fk {e_fk:.2e} mbl)")
            items[d] = (itd, Rd, float(ang[ang > 1e-6].mean()) if np.any(ang > 1e-6) else 0.0, int((ang > 1e-6).sum()))
        gens = {s: gen(it0, s) for s in seeds}
        w0 = {s: world_of_ktjd(gens[s][1:1 + t_real, :J], base, rig, strict_gt=False) for s in seeds}
        cap = str(base[ds0.index[i][1]].get("caption", ""))                                # the pair item carries no caption
        rec = dict(i=i, rig=rig, motion_id=str(it0["motion_id"]), J=J, T=t_real, mbl=mbl, caption=cap,
                   d_seed=dict(direct=dist(w0[seeds[1]][0], w0[seeds[0]][0], mbl), fk=dist(w0[seeds[1]][1], w0[seeds[0]][1], mbl)),
                   d_gt={str(s): dict(direct=dist(w0[s][0], gt_w, mbl), fk=dist(w0[s][1], gt_fk, mbl)) for s in seeds}, rot={})
        for d in degs:
            itd, Rd, mean_ang, n_rot = items[d]
            rr = dict(mean_angle_deg=mean_ang, n_rotated_joints=n_rot)
            for s in seeds:
                gd = gen(itd, s)
                wd, fd = decode_with_rest(denorm(gd[1:1 + t_real], itd, J), sk, Rd, strict_gt=False)
                rr[str(s)] = dict(direct=dist(wd, w0[s][0], mbl), fk=dist(fd, w0[s][1], mbl),
                                  gt_direct=dist(wd, gt_w, mbl), gt_fk=dist(fd, gt_fk, mbl))
                if d == max(degs) and s == seeds[0] and k < a.n_gif:
                    render_gif(out / f"{k:02d}_{rig}__{str(it0['motion_id'])[:40]}_rot{int(d)}.gif",
                               [("gt", "TARGET GT", gt_fk), ("gen_ric", f"GEN plain rest (fk) s{s}", w0[s][1]),
                                ("gen_fk", f"GEN rest rotated <= {int(d)} deg (fk) s{s}", fd)],
                               [int(p) for p in par], rec["caption"], f"[A] {rig}", fps=30)
            rec["rot"][str(d)] = rr
        recs.append(rec)
        print(f"[probe] {k + 1}/{len(idx)} {rig} J{J} T{t_real}: d_seed fk {rec['d_seed']['fk'][0]:.3f} | "
              + " | ".join(f"{d:g}deg fk {np.mean([rec['rot'][str(d)][str(s)]['fk'][0] for s in seeds]):.3f}" for d in degs), flush=True)

    def med(vals):
        return float(np.median(vals)) if len(vals) else float("nan")
    summ = dict(n_items=len(recs), degs=degs, seeds=seeds, steps=a.steps, cfg_text=a.cfg_text, epoch=ck.get("epoch"), ckpt=a.ckpt,
                median=dict(d_seed={p: [med([r["d_seed"][p][q] for r in recs]) for q in (0, 1)] for p in ("direct", "fk")},
                            d_gt={p: [med([r["d_gt"][str(s)][p][q] for r in recs for s in seeds]) for q in (0, 1)] for p in ("direct", "fk")},
                            d_rot={str(d): {p: [med([r["rot"][str(d)][str(s)][p][q] for r in recs for s in seeds]) for q in (0, 1)]
                                            for p in ("direct", "fk")} for d in degs},
                            d_rot_gt={str(d): {p: [med([r["rot"][str(d)][str(s)]["gt_" + p][q] for r in recs for s in seeds]) for q in (0, 1)]
                                               for p in ("direct", "fk")} for d in degs},
                            mean_angle_deg={str(d): med([r["rot"][str(d)]["mean_angle_deg"] for r in recs]) for d in degs}))
    (out / "probe.json").write_text(json.dumps(dict(summary=summ, items=recs), indent=1))
    m = summ["median"]
    lines = [f"rest-convention probe  ckpt={a.ckpt} epoch={ck.get('epoch')} items={len(recs)} seeds={seeds} steps={a.steps} cfg_text={a.cfg_text}",
             "medians, mean per-joint world displacement in mean-bone-length units: full world / root-relative",
             f"{'':>22} {'direct':>17} {'fk':>17}",
             f"{'seed vs seed (plain)':>22} {m['d_seed']['direct'][0]:8.3f}/{m['d_seed']['direct'][1]:8.3f} {m['d_seed']['fk'][0]:8.3f}/{m['d_seed']['fk'][1]:8.3f}",
             f"{'gen vs GT (plain)':>22} {m['d_gt']['direct'][0]:8.3f}/{m['d_gt']['direct'][1]:8.3f} {m['d_gt']['fk'][0]:8.3f}/{m['d_gt']['fk'][1]:8.3f}"]
    for d in degs:
        r, g = m["d_rot"][str(d)], m["d_rot_gt"][str(d)]
        lines.append(f"{f'rot<={d:g}deg vs plain':>22} {r['direct'][0]:8.3f}/{r['direct'][1]:8.3f} {r['fk'][0]:8.3f}/{r['fk'][1]:8.3f}"
                     f"   (rot vs GT {g['direct'][0]:.3f}/{g['direct'][1]:.3f} {g['fk'][0]:.3f}/{g['fk'][1]:.3f}; mean drawn angle {m['mean_angle_deg'][str(d)]:.1f} deg)")
    (out / "summary.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)
    print(f"[probe] DONE -> {out}", flush=True)


if __name__ == "__main__":
    main()
