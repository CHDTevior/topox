#!/usr/bin/env python3
"""Render 4-panel skeleton GIFs from an in-context DiT checkpoint.

Layout per clip (GT stays the RED rightmost panel, per the established convention):
    DEMO (teal, ric) | GEN pos (orange, ric) | GEN fk (purple, rot6d-FK) | TARGET GT (red, ric)
The GENERATED motion is drawn under BOTH recoveries on purpose: H4 measured 4-19% disagreement
between the position family and the rotation family on generated output, and two panels make that
disagreement visible instead of hiding it behind whichever family we pick. GT needs one panel only
(the families agree to 0.000% on real data). One shared scale across all four panels.
True-speed playback: every real frame at 20 fps, no subsampling.

Read-only w.r.t. training state; writes GIFs + summary.txt (jitter for both recoveries).
"""
import argparse, json, pickle, sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.anytop_dataset import (AnyTopDataset, _STD_FLOOR,                 # noqa: E402
                                     _recover_world_positions)
from src.data.anytop_rot6d_fk import recover_from_bvh_rot_np                     # noqa: E402
from src.data.incontext_pairs import (InContextPairs, collate, read_split,      # noqa: E402
                                      truebones_types, pzh_types, DEMO_FRAMES, TARGET_FRAMES)
from src.models.v2.dit_motion import InContextMotionDiT, sample                  # noqa: E402

PANEL_W, PANEL_H, GAP, FOOT = 300, 360, 12, 66
# The GENERATED motion is drawn under BOTH recoveries: H4 measured a 4-19% disagreement between
# the position family (RIC) and the rotation family (FK) on generated output -- two panels make
# that disagreement visible to the eye instead of hiding it behind whichever family we pick.
# GT needs one panel only (the two families agree to 0.000% on real data).
COLS = {"demo": (13, 110, 100), "gen_ric": (176, 61, 8),
        "gen_fk": (91, 44, 184), "gt": (185, 28, 28)}


def world_of(norm_txjc, mean, std, recover="ric", parents=None, offsets=None):
    """World positions via either channel family. The two disagree exactly where the model is
    inconsistent, so rendering BOTH localises noise: ric reads per-frame positions (ch0:3),
    fk rebuilds them from rotations (ch3:9) over the bone chain.
    """
    raw = (norm_txjc * (std[None] + _STD_FLOOR) + mean[None]).astype(np.float64)
    if recover == "fk":
        return recover_from_bvh_rot_np(raw, parents, offsets)        # [T,J,3]
    return _recover_world_positions(raw)                             # [T,J,3]


def jitter_ratio(gen_w, gt_w):
    """Second-difference acceleration ratio gen/GT -- the milestone-comparable jitter number.
    Reported for all joints and for the root row separately (root mixes in velocity-integration
    noise). GT is the natural denominator: ~1.0 means as smooth as real motion."""
    def acc(w):
        if w.shape[0] < 3:
            return None
        a = w[2:] - 2 * w[1:-1] + w[:-2]
        return np.linalg.norm(a, axis=-1)
    ga, ta = acc(gen_w), acc(gt_w)
    if ga is None or ta is None:
        return float("nan"), float("nan"), float("nan"), float("nan")
    # the raw GT denominators ride along: a near-static GT makes the ratio arbitrarily large,
    # so the ratio is only interpretable next to its denominator.
    gt_all, gt_root = float(ta.mean()), float(ta[:, 0].mean())
    allr = float(ga.mean() / max(gt_all, 1e-9))
    rootr = float(ga[:, 0].mean() / max(gt_root, 1e-9))
    return allr, rootr, gt_all, gt_root


def draw_panel(img, xy, parents, col, x0):
    d = ImageDraw.Draw(img)
    for j, p in enumerate(parents):
        if p < 0:
            continue
        d.line([x0 + xy[j, 0], xy[j, 1], x0 + xy[p, 0], xy[p, 1]], fill=col, width=3)
    for j in range(len(parents)):
        r = 4 if j == 0 else 2
        d.ellipse([x0 + xy[j, 0] - r, xy[j, 1] - r, x0 + xy[j, 0] + r, xy[j, 1] + r], fill=col)


def project(seqs, yaw_deg=28.0):
    """Shared orthographic projection: x' = x cosA + z sinA, y' = y (up). One bbox over ALL
    sequences so panels share scale; returns list of [T,J,2] pixel coords."""
    a = np.radians(yaw_deg)
    flat = [np.stack([s[..., 0] * np.cos(a) + s[..., 2] * np.sin(a), s[..., 1]], axis=-1)
            for s in seqs]
    allp = np.concatenate([f.reshape(-1, 2) for f in flat], axis=0)
    lo, hi = allp.min(0), allp.max(0)
    span = float(max(hi[0] - lo[0], hi[1] - lo[1], 1e-6))
    sc = (min(PANEL_W, PANEL_H) - 46) / span
    out = []
    for f in flat:
        px = (f[..., 0] - (lo[0] + hi[0]) / 2) * sc + PANEL_W / 2
        py = PANEL_H - ((f[..., 1] - (lo[1] + hi[1]) / 2) * sc + PANEL_H / 2)
        out.append(np.stack([px, py], axis=-1))
    return out


def render_gif(out_path, panels, parents, caption, rig):
    """panels: list of (color_key, tag, seq [T,J,3]); shared scale across all of them."""
    names = [c for c, _, _ in panels]
    tags = [t for _, t, _ in panels]
    seqs = project([w for _, _, w in panels])
    T = max(s.shape[0] for s in seqs)
    W = PANEL_W * len(panels) + GAP * (len(panels) - 1)
    frames = []
    for t in range(T):
        img = Image.new("RGB", (W, PANEL_H + FOOT), (246, 248, 247))
        d = ImageDraw.Draw(img)
        for k, (name, seq) in enumerate(zip(names, seqs)):
            x0 = k * (PANEL_W + GAP)
            d.rectangle([x0, 0, x0 + PANEL_W - 1, PANEL_H - 1], outline=(216, 223, 225))
            tt = min(t, seq.shape[0] - 1)                    # shorter panels freeze on last frame
            draw_panel(img, seq[tt], parents, COLS[name], x0)
            d.text((x0 + 8, 6), tags[k], fill=COLS[name])
            d.text((x0 + 8, PANEL_H - 18), f"f{tt+1}/{seq.shape[0]}", fill=(116, 133, 150))
        d.text((8, PANEL_H + 8), rig, fill=(15, 23, 32))
        for li, line in enumerate([caption[i:i + 96] for i in range(0, min(len(caption), 192), 96)]):
            d.text((8, PANEL_H + 26 + 16 * li), line, fill=(64, 80, 94))
        frames.append(img)
    frames[0].save(out_path, save_all=True, append_images=frames[1:], duration=50, loop=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--rigs_A", default="Alligator,Trex")
    ap.add_argument("--rigs_B", default="BrownBear,Elephant")
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--corpus", choices=("truebones", "pzh"), default="truebones")
    ap.add_argument("--demo_frames", type=int, default=DEMO_FRAMES)
    ap.add_argument("--target_frames", type=int, default=TARGET_FRAMES)
    ap.add_argument("--all_targets", action="store_true",
                    help="render EVERY target clip of each requested rig (default: first only)")
    ap.add_argument("--data_root", default="data/animo4d_L4TB_plus_human_v4b272neutral")
    ap.add_argument("--splits_dir", default="data/holdout_splits_v1")
    ap.add_argument("--joint_sem", default="data/joint_semantics_llm2vec_v1.npz")
    ap.add_argument("--caption_cache", default="data/anytop_caption_llm2vec_v4b272neutral_multi")
    ap.add_argument("--texts_json", default="motion_texts_by_file_clean_v1.json")
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)  # our own ckpt; contains numpy RNG state, rejected by 2.6's weights_only default
    ca = ck["args"]
    model = InContextMotionDiT(in_ch=13, dim=ca["dim"], depth=ca["depth"], n_heads=ca["heads"],
                               d_text=4096, d_joint_sem=4096).to(dev)
    model.load_state_dict(ck["model"]); model.eval()
    ep = ck.get("epoch", -1)
    print(f"[render] ckpt {a.ckpt} (epoch {ep}) on {dev}", flush=True)

    cond = pickle.load(open(f"{a.data_root}/_cond_normalized_J144.pkl", "rb"))
    tb = truebones_types(cond.keys()) if a.corpus == "truebones" else pzh_types(cond.keys())
    names = {k: read_split(a.splits_dir, k) for k in ("train", "val", "held_representative")}
    base = AnyTopDataset(data_root=a.data_root, split="all", num_frames=300, max_joints=144,
                         load_captions=True, caption_emb_cache=a.caption_cache,
                         random_caption=False, augment=False, joint_semantics=a.joint_sem,
                         species_whitelist=tb, splits_dir=a.splits_dir,
                         texts_json_name=a.texts_json)
    PK = dict(demo_frames=a.demo_frames, target_frames=a.target_frames)
    dsA = InContextPairs(base, names["val"], names["train"], object_types=tb,
                         balance_skeletons=False, seed=a.seed, **PK)
    dsB = InContextPairs(base, names["held_representative"], names["held_representative"],
                         object_types=tb, balance_skeletons=False, seed=a.seed, **PK)

    jobs = []
    for bucket, rigs, ds in (("A", a.rigs_A, dsA), ("B", a.rigs_B, dsB)):
        for r in [x.strip() for x in rigs.split(",") if x.strip()]:
            if r not in ds.types:
                print(f"[render] SKIP {bucket}:{r} -- not in bucket "
                      f"({'A held?' if bucket == 'A' else 'train?'})", flush=True)
                continue
            positions = [i for i, (ot, _) in enumerate(ds.index) if ot == r]
            if not a.all_targets:
                positions = positions[:1]                     # default: first target, as before
            jobs += [(bucket, r, ds, pp) for pp in positions]
    lines = []
    for bucket, rig, ds, pos in jobs:
        # Stream reset PER ITEM: each target's demo/crop draw starts from the same rng origin, so a
        # given (rig, target) renders identically across invocations and epochs regardless of how
        # many other targets were rendered before it.
        ds._wrng_key = None
        item = ds[pos]
        b = {k: (v.to(dev) if torch.is_tensor(v) else v)
             for k, v in collate([item]).items()}
        J = int(item["n_joints"])
        t_real = int(item["frame_valid"][a.demo_frames:].sum())
        d_real = int(item["frame_valid"][:a.demo_frames].sum())

        torch.manual_seed(a.seed)
        with torch.no_grad():
            gen = sample(model, b["x"], b["is_target"], a.steps,
                         joint_bias=b["joint_bias"], frame_valid=b["frame_valid"],
                         joint_valid=b["joint_valid"], text=b["text"], joint_sem=b["joint_sem"])
        gen = gen[0].float().cpu().numpy()
        gt = b["x"][0].float().cpu().numpy()

        t_item = base[ds.index[pos][1]]
        mean = np.asarray(t_item["anytop_mean"])[:J]; std = np.asarray(t_item["anytop_std"])[:J]
        parents = [int(p) for p in t_item["parent_indices"][:J]]
        offsets = np.asarray(t_item["rest_offsets"])[:J]
        gseg = gen[a.demo_frames:a.demo_frames + t_real, :J]
        demo_w = world_of(gt[:d_real, :J], mean, std)
        gen_ric = world_of(gseg, mean, std)
        gen_fk = world_of(gseg, mean, std, recover="fk", parents=parents, offsets=offsets)
        gt_w = world_of(gt[a.demo_frames:a.demo_frames + t_real, :J], mean, std)
        jit_all, jit_root, gt_all, gt_root = jitter_ratio(gen_ric, gt_w)
        jfk_all, jfk_root, _, _ = jitter_ratio(gen_fk, gt_w)
        static_warn = "  [near-static GT, ratio inflated]" if gt_all < 1e-3 else ""

        cap = str(t_item.get("caption", ""))
        name = f"{bucket}_{rig}__{item['motion_id'][:48]}"
        render_gif(out / f"{name}.gif",
                   [("demo", f"DEMO {item['demo_id']}", demo_w),
                    ("gen_ric", f"GEN pos ep{ep} s{a.steps}", gen_ric),
                    ("gen_fk", f"GEN fk ep{ep} s{a.steps}", gen_fk),
                    ("gt", "TARGET GT", gt_w)],
                   parents, cap, f"[{bucket}] {rig}")
        print(f"[render] {name}.gif  (demo {d_real}f | target {t_real}f, J={J})  "
              f"jitter ric {jit_all:.2f}x fk {jfk_all:.2f}x (GT {gt_all:.4f}) "
              f"root ric {jit_root:.2f}x fk {jfk_root:.2f}x{static_warn}", flush=True)
        lines.append(f"{name}\tjitter_ric={jit_all:.3f}x jitter_fk={jfk_all:.3f}x(gt={gt_all:.5f}) "
                     f"root_ric={jit_root:.3f}x root_fk={jfk_root:.3f}x{static_warn}"
                     f"\tcaption: {cap}\tdemo={item['demo_id']} target={item['motion_id']}")
    (out / "summary.txt").write_text(
        f"ckpt={a.ckpt} epoch={ep} steps={a.steps} seed={a.seed} panels=demo|gen_ric|gen_fk|gt\n" + "\n".join(lines) + "\n")
    print(f"[render] DONE -> {out}", flush=True)


if __name__ == "__main__":
    main()
