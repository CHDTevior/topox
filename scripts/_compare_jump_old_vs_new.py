#!/usr/bin/env python3
"""Same acceleration criterion on OLD vs NEW corpus. A number without its control is not evidence."""
import sys, glob, os
import numpy as np

def scan(root, n, key, prefix=None):
    fs = sorted(glob.glob(root + "/motions/*.npz"))
    # The old corpus is mostly HML3D_Human, whose motion is far smoother than any Planet Zoo rig.
    # Comparing the whole old corpus against a PZ-only new one measures the species mix, not the
    # IK defect. Restrict both sides to PZ.
    if prefix:
        fs = [f for f in fs if os.path.basename(f).startswith(prefix)]
    rng = np.random.default_rng(0)
    sel = [fs[i] for i in rng.choice(len(fs), min(n, len(fs)), replace=False)]
    out = []
    for f in sel:
        d = np.load(f, allow_pickle=True)
        if key not in d:
            continue
        P = d[key][..., 0:3].astype(np.float64)
        if P.ndim != 3 or P.shape[0] < 5:
            continue
        v = np.diff(P, axis=0); a = np.diff(v, axis=0)
        an = np.linalg.norm(a, axis=-1); vn = np.linalg.norm(v, axis=-1)
        med = np.median(vn[vn > 0]) if (vn > 0).any() else 0.0
        if med > 0:
            out.append(float(an.max() / med))
    return np.array(sorted(out)), len(fs)

NEWR = ("data/animo4d_anytop_noik/processed/AniMo4D_AnyTop_Official_NoIK_v1/data")
for label, root, key, pref in (
        ("OLD  PZ-only  ", "dataset/ktjd17_pz_human312", "motion", "PZ_"),
        ("OLD  Human-only", "dataset/ktjd17_pz_human312", "motion", "HML3D_"),
        ("NEW  no-IK PZ ", NEWR, "motion", None)):
    r, total = scan(root, 600, key, pref)
    if not len(r):
        print(f"{label}: no usable clips (key '{key}' missing?)"); continue
    n = len(r)
    print(f"{label}  [{total} clips, {n} sampled]  median={r[n//2]:6.1f}  p90={r[int(n*.9)]:7.1f}  "
          f"p99={r[int(n*.99)]:8.1f}  max={r[-1]:9.1f}   "
          f">100: {100*(r>100).mean():5.2f}%   >500: {100*(r>500).mean():5.2f}%   "
          f">2000: {100*(r>2000).mean():5.2f}%")
