#!/usr/bin/env python3
"""EXACT exhaustive normalized-magnitude bound, from the corpus's stored per-cell extrema.

The source stats artifact records `minimum`/`maximum` per (rig, joint, channel) over ALL frames of
ALL clips of that rig, so the largest normalized magnitude any frame can produce is
    max(|minimum - mean_eff|, |maximum - mean_eff|) / std_eff
This is a bound over the whole corpus, not a sample -- strictly stronger than walking the loader,
and it costs milliseconds instead of hours. Caveat recorded below: ch13/14 (smooth-root XZ) are
re-based per crop window, so their served values are NOT the stored ones; they are reported
separately and must be checked through the loader.
"""
import json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.anytop_dataset import _STD_FLOOR

SRC = np.load("data/pzh312_rig_stats_cut100.npz", allow_pickle=True)
ART = np.load("data/pzh312_norm_stats_v4.npz", allow_pickle=True)
rig = [str(r) for r in ART["rig_ids"]]
ch = [str(c) for c in SRC["channel_names"]]
mean, std = ART["mean"].astype(np.float64), ART["std"].astype(np.float64) + _STD_FLOOR
sup = ART["supervise_mask"]
lo, hi = SRC["minimum"], SRC["maximum"]

worst = np.maximum(np.abs(lo - mean), np.abs(hi - mean)) / std
REBASED = (13, 14)                      # served value is re-based per window; stored bound N/A
tgt = sup.copy()
tgt[:, :, list(REBASED)] = False
w = np.where(tgt, worst, 0.0)

print(f"supervised cells (excl. re-based ch13/14): {int(tgt.sum()):,}")
q = [50, 90, 99, 99.9, 99.99]
vals = worst[tgt]
print("exact upper bound on |normalized| per supervised cell:")
print("   " + "  ".join(f"p{x}={np.percentile(vals,x):.2f}" for x in q) + f"  max={vals.max():.2f}")
for t in (5, 10, 20, 30, 50, 100):
    n = int((vals > t).sum())
    print(f"   cells whose bound exceeds {t:4d}: {n:6d}  ({100*n/vals.size:.4f}%)")

order = np.dstack(np.unravel_index(np.argsort(-w, axis=None)[:25], w.shape))[0]
print("\ntop 25 cells by exact bound:")
for r, j, c in order:
    print(f"  {worst[r,j,c]:9.1f}  {rig[r]:34s} j{j:<3d} ch{c:<2d} {ch[c]:<18s} "
          f"raw[{lo[r,j,c]:+.4f},{hi[r,j,c]:+.4f}] mean={mean[r,j,c]:+.4f} std={std[r,j,c]:.4f}")

import collections
bych = collections.Counter()
for r, j, c in np.argwhere(tgt & (worst > 20)):
    bych[ch[c]] += 1
print("\ncells with bound > 20, by channel:", dict(bych.most_common()))
byrig = collections.Counter(rig[r] for r, j, c in np.argwhere(tgt & (worst > 20)))
print("cells with bound > 20, by rig (top 8):", byrig.most_common(8))

rep = {"percell_sha256": __import__("hashlib").sha256(
           Path("data/pzh312_norm_stats_v4.npz").read_bytes()).hexdigest(),
       "method": "exact bound from stored per-cell minimum/maximum over the full corpus",
       "excluded": "ch13/14 smooth-root XZ (re-based per crop window; check through the loader)",
       "n_cells": int(tgt.sum()),
       "percentiles": {f"p{x}": float(np.percentile(vals, x)) for x in q},
       "max": float(vals.max()),
       "n_over": {str(t): int((vals > t).sum()) for t in (5, 10, 20, 30, 50, 100)},
       "top": [{"bound": float(worst[r, j, c]), "rig": rig[r], "j": int(j), "ch": int(c),
                "channel": ch[c], "raw_min": float(lo[r, j, c]), "raw_max": float(hi[r, j, c]),
                "mean": float(mean[r, j, c]), "std_eff": float(std[r, j, c])}
               for r, j, c in order]}
Path("configs/pzh312_extrema_audit_exact.json").write_text(json.dumps(rep, indent=1))
print("\n[OK] wrote configs/pzh312_extrema_audit_exact.json")
