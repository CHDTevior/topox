"""Does each rig's rest pose stand the way its motion does?

Our 17-channel model (and UniMate) is conditioned on the rig's rest pose; with rep_norm=rest the rest frame is the
origin of the normalized space, so a rest pose exported lying down while the animation stands up (or the reverse)
forces the model to produce a large global rotation away from its own reference. The rest-vs-UniMate check
(_diag_rest_vs_unimate_tpose.py) cannot see this, since both pipelines read the same export rest arrays.

Per clip, on up to --max_frames evenly spaced frames: root-centre the rest pose P (skeletons/<rig>.npz P_rest_global)
and the frame's positions Q (motion channels 0:3, q_position), fit the best rotation R with P R^T ~ Q (Kabsch), and
record the tilt of the vertical axis, acos(R[1,1]). Also recorded: the fit residual and the unrotated distance, in
units of the rest pose's diameter, so a large tilt can be told apart from a pose that simply does not fit.
Frames whose rest pose is near-collinear (second singular value tiny) give no defined rotation and are skipped.
The REST VERDICT uses frame 0 only (most clips start in the rig's normal stance; later frames tilt because the
motion itself turns the body) and only when the rotation explains the difference (raw >= 2 x residual); tilt over
all frames is kept as a statistic but mixes in limb articulation and is not a verdict.

usage: python scripts/_diag_rest_vs_motion_orientation.py --out renders/rest_vs_motion_orientation [--render_top 12]
Read-only; writes clips.jsonl, summary.txt and PNGs (rest | frame 0 | middle frame) under --out.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[1]


def kabsch(A, B):
    """(R, s2/s1): R minimizes |A R^T - B| (no reflection); s2/s1 flags a near-collinear A."""
    H = A.T @ B
    U, S, Vt = np.linalg.svd(H)
    D = np.diag([1.0, 1.0, np.sign(np.linalg.det(Vt.T @ U.T))])
    return Vt.T @ D @ U.T, (S[1] / S[0] if S[0] > 0 else 0.0)


def clip_record(P, M, max_frames):
    """P [J,3] rest, M [T,J,3] positions -> dict of per-clip orientation statistics."""
    A = P - P[0]
    d = float(np.max(np.linalg.norm(A[:, None] - A[None], axis=-1)))
    T = len(M)
    idx = np.unique(np.linspace(0, T - 1, min(T, max_frames)).round().astype(int))
    tilt, resid, raw = [], [], []
    B0 = M[0] - M[0, 0]
    R0, c0 = kabsch(A, B0)
    f0 = {"tilt0": float(np.degrees(np.arccos(np.clip(R0[1, 1], -1, 1)))) if c0 >= 1e-3 else None,
          "resid0": float(np.linalg.norm(A @ R0.T - B0, axis=-1).mean() / d) if c0 >= 1e-3 else None,
          "raw0": float(np.linalg.norm(A - B0, axis=-1).mean() / d)}
    for t in idx:
        B = M[t] - M[t, 0]
        R, cond = kabsch(A, B)
        if cond < 1e-3:
            continue
        tilt.append(float(np.degrees(np.arccos(np.clip(R[1, 1], -1, 1)))))
        resid.append(float(np.linalg.norm(A @ R.T - B, axis=-1).mean() / d))
        raw.append(float(np.linalg.norm(A - B, axis=-1).mean() / d))
    if not tilt or f0["tilt0"] is None:
        return {"status": "degenerate", "T": T, **f0}
    return {"status": "ok", "T": T, "frames_used": len(tilt), "diameter": d, **f0,
            "tilt_med": float(np.median(tilt)), "tilt_max": float(np.max(tilt)),
            "resid_med": float(np.median(resid)), "raw_med": float(np.median(raw))}


def render(path, P, M, parents, title):
    sys.path.insert(0, str(REPO / "scripts"))
    from _pil_skeleton_render import compute_transform, render_panel
    from PIL import Image, ImageDraw
    cell = (520, 480)
    frames = [P - [P[0, 0], 0, P[0, 2]], M[0] - [M[0, 0, 0], 0, M[0, 0, 2]], M[len(M) // 2] - [M[len(M) // 2, 0, 0], 0, M[len(M) // 2, 0, 2]]]
    frames = [f - [0, f[:, 1].min(), 0] for f in frames]
    tr = compute_transform(frames, cell, 0.12, 1.0)
    names = ["rest (P_rest_global)", "motion frame 0", f"motion frame {len(M) // 2}"]
    cols = [(13, 110, 100), (20, 20, 20), (20, 20, 20)]
    canvas = Image.new("RGB", (cell[0] * 3, cell[1] + 40), "white")
    for k, (f, nm, c) in enumerate(zip(frames, names, cols)):
        canvas.paste(render_panel(f[None], parents, 0, tr, cell, nm, c, 4, 6, k == 0, True), (cell[0] * k, 40))
    ImageDraw.Draw(canvas).text((14, 10), title, fill=(20, 20, 20))
    canvas.save(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=REPO / "dataset/ktjd17_uniml3d_v2")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--max_frames", type=int, default=60)
    ap.add_argument("--render_top", type=int, default=12)
    ap.add_argument("--render_rigs", default="", help="comma-separated rig ids to render (first clip of each)")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    man = {}
    for line in open(a.root / "manifests/clips.jsonl"):
        r = json.loads(line)
        man[r["clip_id"]] = r
    sk_cache, rows = {}, []
    for cid, r in sorted(man.items()):
        rig = r["rig_id"]
        if rig not in sk_cache:
            z = np.load(a.root / "skeletons" / f"{rig}.npz")
            sk_cache[rig] = (np.asarray(z["P_rest_global"], np.float64), np.asarray(z["parents"]))
        P, par = sk_cache[rig]
        m = np.load(a.root / "motions" / f"{cid}.npz")
        if str(m["rig_id"]) != rig:
            raise SystemExit(f"[refuse] {cid}: motion file says rig {m['rig_id']}, manifest says {rig}")
        M = np.asarray(m["motion"], np.float64)[..., 0:3]
        if M.shape[1] != len(P):
            raise SystemExit(f"[refuse] {cid}: {M.shape[1]} joints in motion vs {len(P)} in skeleton")
        rec = clip_record(P, M, a.max_frames)
        rec.update(clip=cid, rig=rig, split=r["split"], official_id=r["official_id"], J=len(P),
                   caption=(r["captions"][0] if r["captions"] else ""))
        rows.append(rec)
    with open(a.out / "clips.jsonl", "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    ok = [r for r in rows if r["status"] == "ok"]
    rigs = {}
    for r in ok:
        rigs.setdefault(r["rig"], []).append(r)
    L = [f"clips {len(rows)}  ok {len(ok)}  degenerate {len(rows) - len(ok)}  rigs {len(rigs)}"]
    v = np.array([r["tilt_med"] for r in ok])
    L.append("[statistic, NOT a rest verdict -- includes articulation] clip median-tilt percentiles 50/90/95/99/max: "
             + " / ".join(f"{q:.1f}" for q in np.percentile(v, [50, 90, 95, 99, 100])))
    starts = lambda r, thr: r["tilt0"] > thr and r["raw0"] >= 2.0 * r["resid0"]
    L.append("[rest verdict] clip STARTS tilted: frame-0 tilt > thr AND the rotation explains it (raw0 >= 2 x resid0);"
             " rig counted only if EVERY clip starts tilted")
    for thr in (45, 60, 80):
        sc = [r for r in ok if starts(r, thr)]
        sr = [g for g, cl in rigs.items() if all(starts(r, thr) for r in cl)]
        ncl = sum(len(rigs[g]) for g in sr)
        L.append(f"  thr {thr} deg: clips {len(sc)} ({100 * len(sc) / len(ok):.1f}%, val {sum(r['split'] == 'val' for r in sc)}); "
                 f"rigs with every clip starting tilted {len(sr)} ({100 * len(sr) / len(rigs):.1f}%), multi-clip {sum(len(rigs[g]) > 1 for g in sr)}, "
                 f"their clips {ncl} (val {sum(r['split'] == 'val' for g in sr for r in rigs[g])})")
    L.append("largest frame-0 tilt among clips that start tilted (thr 60):")
    top = sorted([r for r in ok if starts(r, 60)], key=lambda r: -r["tilt0"])[:a.render_top]
    for r in top:
        L.append(f"  {r['rig']}  {r['official_id']}  split={r['split']}  J={r['J']}  tilt0={r['tilt0']:.1f}  "
                 f"raw0={r['raw0']:.3f}D resid0={r['resid0']:.3f}D  med={r['tilt_med']:.1f}  \"{r['caption']}\"")
    want = [r["rig"] for r in top] + [g for g in a.render_rigs.split(",") if g]
    L.append("requested rigs:")
    for g in [g for g in a.render_rigs.split(",") if g]:
        for r in rigs.get(g, []):
            L.append(f"  {g}  {r['official_id']}  split={r['split']}  tilt0={r['tilt0']:.1f}  raw0={r['raw0']:.3f}D  resid0={r['resid0']:.3f}D  "
                     f"starts tilted@60={starts(r, 60)}  med={r['tilt_med']:.1f}")
    (a.out / "summary.txt").write_text("\n".join(L) + "\n")
    print("\n".join(L))
    done = set()
    for g in want:
        if g in done or g not in rigs:
            continue
        done.add(g)
        r = max(rigs[g], key=lambda x: x["tilt0"])
        P, par = sk_cache[g]
        M = np.asarray(np.load(a.root / "motions" / f"{r['clip']}.npz")["motion"], np.float64)[..., 0:3]
        render(a.out / f"{g}__{r['clip']}.png", P, M, par,
               f"{g}  {r['official_id']}  split={r['split']}  frame-0 tilt {r['tilt0']:.1f} deg  raw0 {r['raw0']:.3f}D  resid0 {r['resid0']:.3f}D")
    print(f"rendered {len(done)} PNGs -> {a.out}")


if __name__ == "__main__":
    main()
