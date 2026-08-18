#!/usr/bin/env python3
"""Render [demo | generated | GT] skeleton GIFs from an in-context DiT checkpoint.

Layout per clip (user's established convention: GT is the RED rightmost panel):
    DEMO (GT, teal)  |  GENERATED (orange)  |  TARGET GT (red)
True-speed playback: every real frame at 20 fps, no subsampling (a 135-frame target plays 6.75 s).
Positions come from the OFFICIAL RIC recovery (_recover_world_positions) on de-normalised 13ch --
the preflight checks it against rot6d-FK for internal consistency (rel mean err < 0.5%). One shared scale across the
three panels so sizes are comparable.

Read-only w.r.t. training state; writes GIFs + summary.txt under --out.
"""
import argparse, json, pickle, sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.anytop_dataset import (AnyTopDataset, _STD_FLOOR,                 # noqa: E402
                                     _recover_world_positions)
from src.data.incontext_pairs import (InContextPairs, collate, read_split,      # noqa: E402
                                      truebones_types, DEMO_FRAMES)
from src.models.v2.dit_motion import InContextMotionDiT, sample                  # noqa: E402

PANEL_W, PANEL_H, GAP, FOOT = 340, 380, 14, 66
COLS = {"demo": (13, 110, 100), "gen": (176, 61, 8), "gt": (185, 28, 28)}


def world_of(norm_txjc, mean, std):
    raw = norm_txjc * (std[None] + _STD_FLOOR) + mean[None]
    return _recover_world_positions(raw.astype(np.float64))          # [T,J,3]


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


def render_gif(out_path, demo_w, gen_w, gt_w, parents, caption, rig, tags):
    seqs = project([demo_w, gen_w, gt_w])
    T = max(s.shape[0] for s in seqs)
    W = PANEL_W * 3 + GAP * 2
    frames = []
    for t in range(T):
        img = Image.new("RGB", (W, PANEL_H + FOOT), (246, 248, 247))
        d = ImageDraw.Draw(img)
        for k, (name, seq) in enumerate(zip(("demo", "gen", "gt"), seqs)):
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
    tb = truebones_types(cond.keys())
    names = {k: read_split(a.splits_dir, k) for k in ("train", "val", "held_representative")}
    base = AnyTopDataset(data_root=a.data_root, split="all", num_frames=300, max_joints=144,
                         load_captions=True, caption_emb_cache=a.caption_cache,
                         random_caption=False, augment=False, joint_semantics=a.joint_sem,
                         species_whitelist=tb, splits_dir=a.splits_dir,
                         texts_json_name=a.texts_json)
    dsA = InContextPairs(base, names["val"], names["train"], object_types=tb,
                         balance_skeletons=False, seed=a.seed)
    dsB = InContextPairs(base, names["held_representative"], names["held_representative"],
                         object_types=tb, balance_skeletons=False, seed=a.seed)

    jobs = [("A", r.strip(), dsA) for r in a.rigs_A.split(",") if r.strip()] + \
           [("B", r.strip(), dsB) for r in a.rigs_B.split(",") if r.strip()]
    lines = []
    for bucket, rig, ds in jobs:
        if rig not in ds.types:
            print(f"[render] SKIP {bucket}:{rig} -- not in bucket ({'A held?' if bucket=='A' else 'train?'})",
                  flush=True)
            continue
        ds._wrng_key = None                                   # deterministic demo/crop per run
        pos = next(i for i, (ot, _) in enumerate(ds.index) if ot == rig)
        item = ds[pos]
        b = {k: (v.to(dev) if torch.is_tensor(v) else v)
             for k, v in collate([item]).items()}
        J = int(item["n_joints"])
        t_real = int(item["frame_valid"][DEMO_FRAMES:].sum())
        d_real = int(item["frame_valid"][:DEMO_FRAMES].sum())

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
        demo_w = world_of(gt[:d_real, :J], mean, std)
        gen_w = world_of(gen[DEMO_FRAMES:DEMO_FRAMES + t_real, :J], mean, std)
        gt_w = world_of(gt[DEMO_FRAMES:DEMO_FRAMES + t_real, :J], mean, std)

        cap = str(t_item.get("caption", ""))
        name = f"{bucket}_{rig}__{item['motion_id'][:48]}"
        render_gif(out / f"{name}.gif", demo_w, gen_w, gt_w, parents, cap, f"[{bucket}] {rig}",
                   (f"DEMO {item['demo_id']}", f"GEN ep{ep} s{a.steps}", "TARGET GT"))
        print(f"[render] {name}.gif  (demo {d_real}f | target {t_real}f, J={J})", flush=True)
        lines.append(f"{name}\tcaption: {cap}\tdemo={item['demo_id']} target={item['motion_id']}")
    (out / "summary.txt").write_text(
        f"ckpt={a.ckpt} epoch={ep} steps={a.steps} seed={a.seed}\n" + "\n".join(lines) + "\n")
    print(f"[render] DONE -> {out}", flush=True)


if __name__ == "__main__":
    main()
