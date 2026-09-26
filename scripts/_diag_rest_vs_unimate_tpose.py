"""Compare our KTJD-17 rest pose (skeletons/<rig>.npz: P_rest_global -- with rep_norm=rest it is the origin of the
model's normalized space and the 1-frame demo tensor is exactly zero) with UniMate's T-pose condition
(cond.npy[object]['tpos_first_frame']) for every rig both corpora hold.

Both come from the same source file field (the export's rest_local_pos / rest_local_rot); each pipeline then
canonicalizes it (facing from the r_hip/l_hip pair, centering, grounding; UniMate also rescales to diameter 2).
So after removing translation and scale the two should coincide. Two numbers per rig, in units of the pose's
own diameter (max joint-to-joint distance):
  raw_err   -- mean joint distance with NO rotation: nonzero when the canonical frames differ (facing, up axis)
  shape_err -- mean joint distance after the best rotation (Kabsch): nonzero only when the POSE itself differs
plus the angle of that best rotation and how far it tilts the vertical axis (NaN when the pose is near-collinear
and the rotation is undefined), and the scale check |d_theirs / (scale_factor * d_ours) - 1| that the per-pose
normalization above cannot see.
Scope: both pipelines read the same export rest arrays, so a rest pose that is wrong IN THE SOURCE passes here with
error 0 -- see _diag_rest_vs_motion_orientation.py for that.

usage: python scripts/_diag_rest_vs_unimate_tpose.py --out renders/rest_vs_unimate_tpose [--render RIG,RIG,...]
Read-only on both corpora; writes metrics.jsonl, summary.txt and PNGs under --out.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
from _pil_skeleton_render import compute_transform, render_panel, draw_skeleton  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402

OURS = (20, 90, 160)
THEIRS = (190, 70, 30)


def normalize(P):
    """Root at the origin, max joint-to-joint distance 1 (no grounding)."""
    P = np.asarray(P, np.float64)
    d = float(np.max(np.linalg.norm(P[:, None] - P[None], axis=-1)))
    if not np.isfinite(d) or d <= 0:
        return None, d
    Q = (P - P[0]) / d
    return Q, d


def kabsch(A, B):
    """(R, s2/s1): R minimizes |A R^T - B| for root-centered A, B (no reflection); s2/s1 flags near-collinear A."""
    H = A.T @ B
    U, S, Vt = np.linalg.svd(H)
    D = np.diag([1.0, 1.0, np.sign(np.linalg.det(Vt.T @ U.T))])
    return Vt.T @ D @ U.T, (S[1] / S[0] if S[0] > 0 else 0.0)


def compare(ours_P, ours_names, ours_parents, th_P, th_names, th_parents, th_scale=None):
    on, tn = [str(x) for x in ours_names], [str(x) for x in th_names]
    rec = {"J_ours": len(on), "J_theirs": len(tn)}
    common = [n for n in on if n in set(tn)]
    rec["J_common"] = len(common)
    if len(common) < 2 or len(set(on)) != len(on) or len(set(tn)) != len(tn):
        rec["status"] = "names_unusable"
        return rec, None, None
    oi = [on.index(n) for n in common]
    ti = [tn.index(n) for n in common]
    # same tree? parent of each common joint must be the same named joint on both sides
    op = {on[j]: (on[int(ours_parents[j])] if int(ours_parents[j]) >= 0 else None) for j in range(len(on))}
    tp = {tn[j]: (tn[int(th_parents[j])] if int(th_parents[j]) >= 0 else None) for j in range(len(tn))}
    rec["tree_mismatch"] = int(sum(op[n] != tp[n] for n in common))
    A, dA = normalize(np.asarray(ours_P)[oi])
    B, dB = normalize(np.asarray(th_P)[ti])
    if A is None or B is None:
        rec["status"] = "degenerate"
        return rec, None, None
    rec["d_ours"], rec["d_theirs"] = dA, dB
    if th_scale is not None:
        rec["scale_ratio_err"] = float(abs(dB / (float(th_scale) * dA) - 1.0))
    rec["raw_err"] = float(np.linalg.norm(A - B, axis=-1).mean())
    rec["raw_err_max"] = float(np.linalg.norm(A - B, axis=-1).max())
    R, cond_ = kabsch(A, B)
    Ar = A @ R.T
    rec["shape_err"] = float(np.linalg.norm(Ar - B, axis=-1).mean())
    rec["shape_err_max"] = float(np.linalg.norm(Ar - B, axis=-1).max())
    ok_rot = cond_ >= 1e-3
    rec["rot_deg"] = float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1)))) if ok_rot else float("nan")
    rec["up_tilt_deg"] = float(np.degrees(np.arccos(np.clip(R[1, 1], -1, 1)))) if ok_rot else float("nan")
    rec["status"] = "ok"
    return rec, A, B


def ground(P):
    Q = P.copy()
    Q[:, 1] -= Q[:, 1].min()
    return Q


def render(rig, A, B, parents_common, rec, out):
    cell = (640, 600)
    Ag, Bg = ground(A), ground(B)
    tr = compute_transform([Ag, Bg], cell, 0.12, 1.0)
    p1 = render_panel(Ag[None], parents_common, 0, tr, cell, "ours: P_rest_global (demo)", OURS, 4, 6, True, True)
    p2 = render_panel(Bg[None], parents_common, 0, tr, cell, "UniMate: tpos_first_frame", THEIRS, 4, 6, False, True)
    p3 = render_panel(Ag[None], parents_common, 0, tr, cell, "overlay (blue ours / red UniMate)", (40, 40, 40), 4, 6, False, True)
    d3 = ImageDraw.Draw(p3)
    draw_skeleton(d3, Ag, parents_common, tr, cell, OURS, 4, 6, None)
    draw_skeleton(d3, Bg, parents_common, tr, cell, THEIRS, 3, 5, None)
    canvas = Image.new("RGB", (cell[0] * 3, cell[1] + 44), "white")
    for k, p in enumerate((p1, p2, p3)):
        canvas.paste(p, (cell[0] * k, 44))
    ImageDraw.Draw(canvas).text(
        (16, 10), f"{rig}  J={rec['J_common']}  raw_err={rec['raw_err']:.3f}  shape_err={rec['shape_err']:.3f}  "
        f"rot={rec['rot_deg']:.1f}deg  up_tilt={rec['up_tilt_deg']:.1f}deg  (units: pose diameter)", fill=(20, 20, 20))
    canvas.save(out / f"{rig}.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ours", type=Path, default=REPO / "dataset/ktjd17_uniml3d_v2/skeletons")
    ap.add_argument("--cond", type=Path, default=REPO / "outside_docs/UniMate/dataset/features/objaverse/cond.npy")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--render", default="", help="comma-separated rig ids (OBJ_...) to render")
    ap.add_argument("--render_worst", type=int, default=12, help="also render the N largest shape_err and raw_err")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    cond = np.load(a.cond, allow_pickle=True).item()
    rows, keep = [], {}
    want = {r for r in a.render.split(",") if r}
    for f in sorted(a.ours.glob("OBJ_*.npz")):
        rig = f.stem
        obj = rig[len("OBJ_"):]
        if obj not in cond:
            rows.append({"rig": rig, "status": "not_in_unimate"})
            continue
        z = np.load(f)
        c = cond[obj]
        rec, A, B = compare(z["P_rest_global"], z["joint_names"], z["parents"],
                            c["tpos_first_frame"], c["joint_names"], c["parents"], c.get("scale_factor"))
        rec["rig"] = rig
        rows.append(rec)
        if A is not None:
            names = [str(x) for x in z["joint_names"]]
            common = [n for n in names if n in set(str(x) for x in c["joint_names"])]
            par = [(-1 if int(z["parents"][names.index(n)]) < 0 else
                    common.index(names[int(z["parents"][names.index(n)])]))
                   if (int(z["parents"][names.index(n)]) < 0 or names[int(z["parents"][names.index(n)])] in common)
                   else -1 for n in common]
            keep[rig] = (A, B, np.array(par), rec)
    with open(a.out / "metrics.jsonl", "w") as fh:
        for r in rows:   # strict JSON: an undefined rotation is null, not a NaN token
            fh.write(json.dumps({k: (None if isinstance(v, float) and not np.isfinite(v) else v) for k, v in r.items()}) + "\n")
    ok = [r for r in rows if r.get("status") == "ok"]
    lines = [f"rigs ours={len(rows)}  in both={sum(r.get('status') != 'not_in_unimate' for r in rows)}  compared={len(ok)}",
             f"status counts: " + json.dumps({s: sum(r.get('status') == s for r in rows) for s in sorted({r.get('status') for r in rows})}),
             f"tree_mismatch>0: {sum(r['tree_mismatch'] > 0 for r in ok)}  J_common<J_ours: {sum(r['J_common'] < r['J_ours'] for r in ok)}  "
             f"J_common<J_theirs: {sum(r['J_common'] < r['J_theirs'] for r in ok)}"]
    sr = np.array([r["scale_ratio_err"] for r in ok if "scale_ratio_err" in r])
    lines.append(f"scale check |d_theirs/(scale_factor*d_ours)-1|: n={len(sr)}  max {sr.max():.3e}  median {np.median(sr):.3e}")
    lines.append(f"rotation undefined (near-collinear pose): {sum(np.isnan(r['rot_deg']) for r in ok)}")
    for key in ("raw_err", "shape_err", "rot_deg", "up_tilt_deg"):
        v = np.array([r[key] for r in ok]); v = v[np.isfinite(v)]
        q = np.percentile(v, [50, 90, 99, 100])
        lines.append(f"{key}: median {q[0]:.4f}  p90 {q[1]:.4f}  p99 {q[2]:.4f}  max {q[3]:.4f}")
    for thr in (0.01, 0.05, 0.1):
        lines.append(f"shape_err > {thr}: {sum(r['shape_err'] > thr for r in ok)}   raw_err > {thr}: {sum(r['raw_err'] > thr for r in ok)}")
    for key in ("shape_err", "raw_err"):
        lines.append(f"worst by {key}:")
        for r in sorted(ok, key=lambda r: -r[key])[:a.render_worst]:
            lines.append(f"  {r['rig']}  J={r['J_common']}  raw={r['raw_err']:.4f}  shape={r['shape_err']:.4f}  "
                         f"rot={r['rot_deg']:.1f}  tilt={r['up_tilt_deg']:.1f}")
            want.add(r["rig"])
    lines.append("requested renders:")
    for rig in sorted(set(a.render.split(",")) - {""}):
        r = next((x for x in rows if x["rig"] == rig), {"status": "missing_in_ours"})
        lines.append(f"  {rig}  " + (f"raw={r['raw_err']:.4f} shape={r['shape_err']:.4f} rot={r['rot_deg']:.1f} tilt={r['up_tilt_deg']:.1f}"
                                     if r.get("status") == "ok" else r.get("status", "?")))
    (a.out / "summary.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    for rig in sorted(want):
        if rig in keep:
            A, B, par, rec = keep[rig]
            render(rig, A, B, par, rec, a.out)
    print(f"rendered {sum(r in keep for r in want)} PNGs -> {a.out}")


if __name__ == "__main__":
    main()
