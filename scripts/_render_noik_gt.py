#!/usr/bin/env python3
"""Ground-truth viewer for the merged corpus's visual gate.

Three views per rig, because a shape error that hides in one is obvious in another:
  SIDE   the project's own yaw-28 orthographic projection (same as v2_render_incontext)
  TOP    straight down the +Y axis: +X right, +Z down. Facing and left/right limb spread read
         here in a way they never do from the side.
  REST   the skeleton's own P_rest_global, in both of the above -- the pose every clip is a
         delta from, and the thing to check before trusting any motion.
Root joint is circled; the heading arrow is the ch15:17 facing direction (theta from +Z), carried
through the SAME projection as the skeleton rather than drawn separately.
"""
import json, os, sys
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.v2_render_incontext import draw_panel, PANEL_W, PANEL_H, GAP, FOOT

ROOT = Path(os.environ.get("GT_ROOT", "dataset/ktjd17_pzh312_noik_v1"))
QC = Path("data/animo4d_anytop_noik/processed/AniMo4D_AnyTop_Official_NoIK_v1/qc")
OUT = Path(sys.argv[1] if len(sys.argv) > 1 else "renders/noik_gt_gate")
YAW = np.radians(28.0)
SKEL, AXC, ROOTC, HEADC = (185, 28, 28), (16, 92, 168), (20, 40, 200), (224, 96, 0)

def flatten(P, view):
    """[T,J,3] world -> [T,J,2] plane. SIDE keeps Y up; TOP looks down +Y."""
    if view == "side":
        return np.stack([P[..., 0] * np.cos(YAW) + P[..., 2] * np.sin(YAW), P[..., 1]], -1)
    return np.stack([P[..., 0], -P[..., 2]], -1)          # top-down: +X right, +Z down

def fit(flat_list):
    allp = np.concatenate([f.reshape(-1, 2) for f in flat_list], 0)
    lo, hi = allp.min(0), allp.max(0)
    span = float(max(hi[0] - lo[0], hi[1] - lo[1], 1e-6))
    sc = (min(PANEL_W, PANEL_H) - 54) / span
    ctr = (lo + hi) / 2
    def to_px(f):
        return np.stack([(f[..., 0] - ctr[0]) * sc + PANEL_W / 2,
                         PANEL_H - ((f[..., 1] - ctr[1]) * sc + PANEL_H / 2)], -1)
    zero_y = PANEL_H - ((0.0 - ctr[1]) * sc + PANEL_H / 2)
    return to_px, float(zero_y), sc

def axes(d, x0, view, ox, oy, L=34):
    if view == "side":
        pairs = ((( np.cos(YAW)), 0.0, "+X"), (0.0, -1.0, "+Y"), ((np.sin(YAW)), 0.0, "+Z"))
    else:
        pairs = ((1.0, 0.0, "+X"), (0.0, 1.0, "+Z"))
    for dx, dy, lab in pairs:
        d.line([x0 + ox, oy, x0 + ox + dx * L, oy + dy * L], fill=AXC, width=2)
        d.text((x0 + ox + dx * L * 1.2 - 6, oy + dy * L * 1.2 - 6), lab, fill=AXC)
    if view == "top":
        d.text((x0 + ox - 4, oy + 46), "(looking down +Y)", fill=(140, 152, 164))

def panel(img, x0, xy_t, parents, view, zero_y, title, root_tip=None, hv_ok=True):
    d = ImageDraw.Draw(img)
    if view == "side":
        gy = min(max(zero_y, 0), PANEL_H - 1)
        d.rectangle([x0, gy, x0 + PANEL_W - 1, PANEL_H - 1], fill=(236, 240, 238))
        d.line([x0, gy, x0 + PANEL_W - 1, gy], fill=(170, 186, 178), width=2)
    d.rectangle([x0, 0, x0 + PANEL_W - 1, PANEL_H - 1], outline=(216, 223, 225))
    draw_panel(img, xy_t, parents, SKEL, x0)
    rx, ry = x0 + xy_t[0, 0], xy_t[0, 1]
    d.ellipse([rx - 7, ry - 7, rx + 7, ry + 7], outline=ROOTC, width=2)
    if root_tip is not None:
        if hv_ok:
            tx, ty = x0 + root_tip[0], root_tip[1]
            d.line([rx, ry, tx, ty], fill=HEADC, width=3)
            a = np.arctan2(ty - ry, tx - rx)
            for s in (+1, -1):
                d.line([tx, ty, tx + 11 * np.cos(a + s * 2.6), ty + 11 * np.sin(a + s * 2.6)],
                       fill=HEADC, width=3)
        else:
            d.text((rx + 10, ry - 6), "heading invalid", fill=(150, 90, 90))
    axes(d, x0, view, 40, PANEL_H - 40)
    d.text((x0 + 8, 6), title, fill=(15, 23, 32))

def render_motion(out_path, P, head, hv, parents, title, sub):
    L = float(np.ptp(P.reshape(-1, 3), 0).max()) * 0.35
    tip = P[:, 0, :].copy()
    tip[:, 0] += L * head[:, 1]        # sin(theta) -> +X
    tip[:, 2] += L * head[:, 0]        # cos(theta) -> +Z
    Q = np.concatenate([P, tip[:, None, :]], 1)
    views = {}
    for v in ("side", "top"):
        f = flatten(Q, v)
        to_px, zy, _ = fit([f])
        px = to_px(f)
        views[v] = (px[:, :-1, :], px[:, -1, :], zy)
    T = P.shape[0]
    W = PANEL_W * 2 + GAP
    frames = []
    for t in range(T):
        img = Image.new("RGB", (W, PANEL_H + FOOT), (246, 248, 247))
        for k, v in enumerate(("side", "top")):
            xy, tp, zy = views[v]
            panel(img, k * (PANEL_W + GAP), xy[t], parents, v, zy,
                  "SIDE (yaw 28)" if v == "side" else "TOP (down +Y)",
                  tp[t], bool(hv[min(t, len(hv) - 1)]))
        d = ImageDraw.Draw(img)
        d.text((8, PANEL_H + 8), title, fill=(15, 23, 32))
        d.text((8, PANEL_H + 26), f"{sub}   frame {t+1}/{T}", fill=(64, 80, 94))
        d.text((8, PANEL_H + 44), "blue circle = root joint    orange = facing (ch15:17)",
               fill=(120, 136, 150))
        frames.append(img)
    frames[0].save(out_path, save_all=True, append_images=frames[1:],
                   duration=round(1000.0 / 30), loop=0)

def render_rest(out_path, Prest, parents, title):
    Q = Prest[None]                                   # [1,J,3]
    img = Image.new("RGB", (PANEL_W * 2 + GAP, PANEL_H + FOOT), (246, 248, 247))
    for k, v in enumerate(("side", "top")):
        f = flatten(Q, v)
        to_px, zy, _ = fit([f])
        panel(img, k * (PANEL_W + GAP), to_px(f)[0], parents, v, zy,
              "REST side" if v == "side" else "REST top")
    d = ImageDraw.Draw(img)
    d.text((8, PANEL_H + 8), title, fill=(15, 23, 32))
    d.text((8, PANEL_H + 26), f"J={Prest.shape[0]}   Ymin={Prest[:,1].min():.3f} "
                              f"Ymax={Prest[:,1].max():.3f}", fill=(64, 80, 94))
    img.save(out_path)

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = {}
    mf = ROOT / "manifests" / "clips.jsonl"
    for l in open(mf):
        r = json.loads(l)
        r.setdefault("T_target", r.get("stored_frames"))
        rows[r["clip_id"]] = r
    rigs = ("HML3D_Human", "PZ_Cheetah_Male", "PZ_African_Elephant_Female",
            "PZ_Bairds_Tapir_Male", "PZ_Reticulated_Giraffe_Male",
            "PZ_Saltwater_Crocodile_Male", "PZ_West_African_Lion_Male",
            "PZ_Arctic_Fox_Male")
    byrig = {}
    for k, r in rows.items():
        byrig.setdefault(r["rig_id"], []).append((r.get("T_target") or 0, k))
    for rig in rigs:
        if rig not in byrig:
            print(f"[gt] SKIP {rig} (not in corpus)"); continue
        cid = max(byrig[rig])[1]
        with np.load(ROOT / "skeletons" / f"{rig}.npz", allow_pickle=True) as z:
            parents = np.asarray(z["parents"]); Prest = np.asarray(z["P_rest_global"], float)
        with np.load(ROOT / "motions" / f"{cid}.npz") as z:
            m = np.asarray(z["motion"], float); hv = np.asarray(z["heading_valid"], bool)
        P = m[..., 0:3].copy()
        P[..., 0] += m[:, 0, 13][:, None]
        P[..., 2] += m[:, 0, 14][:, None]
        render_rest(OUT / f"REST_{rig}.png", Prest, parents, f"REST POSE  {rig}")
        render_motion(OUT / f"MOTION_{rig}.gif", P, m[:, 0, 15:17], hv, parents,
                      f"{rig}", f"J={P.shape[1]} T={P.shape[0]} 30fps {cid[:12]}")
        print(f"[gt] {rig}  J={P.shape[1]} T={P.shape[0]}", flush=True)
    print(f"[gt] DONE -> {OUT}", flush=True)

if __name__ == "__main__":
    main()
