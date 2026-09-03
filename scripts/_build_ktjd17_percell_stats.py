#!/usr/bin/env python3
"""Old-style per-(rig, joint, channel) mean/std for KTJD-17 (user 2026-08-21).

WHY. KTJD's spec normalization is SCALE-ONLY (s_rig + frozen block gains) because it is designed
for lossless round-trip decode. As a LEARNING TARGET that leaves the rig's static pose inside the
signal: measured on all 671 train clips, a constant prediction captures 75.0% of the
gradient-weighted objective (61.2% from rig identity alone). AnyTop's 13ch pipeline -- the one that
produced coherent motion -- standardizes per (joint, channel) instead, and measures 45.2%/5.9%.
User's decision: use the old method.

COHORT: per species (rig), over ALL accepted clips of that rig -- the user's call: mean/std are a
property of the SPECIES (skeleton scale, typical pose spread), not of a particular clip subset, so
more clips give a steadier estimate. Recorded in the artifact so the choice is never implicit.
For held-out RIGS this is transductive (their own motion informs their normalization); that is
acceptable for the architecture test this artifact is built for, and is recorded here so no
downstream report can quietly treat those numbers as zero-shot.

CONVENTION: the repo's de-normalization is raw = x * (std + _STD_FLOOR) + mean, so the artifact
stores mean and (std - _STD_FLOOR); zero-variance cells (invalid channels, fixed-DOF rows,
constant contacts) get std = 1 so they normalize to exactly 0 and de-normalize back to `mean`.
"""
import json, hashlib, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.anytop_dataset import _STD_FLOOR
from src.data.ktjd17.loader import load_motion_npz

ROOT = Path("dataset/ktjd17_truebones")
OUT = Path("data/ktjd17_percell_stats_v1.npz")

rows = [json.loads(l) for l in open(ROOT / "manifests" / "clips.jsonl")]
rows = [r for r in rows if r.get("status") == "accept"]
by_rig = {}
for r in rows:
    by_rig.setdefault(str(r["rig_id"]), []).append(r)
print(f"{len(rows)} clips / {len(by_rig)} rigs")

means, stds, counts = {}, {}, {}
for i, (rig, rr) in enumerate(sorted(by_rig.items())):
    s1 = s2 = None
    n = 0
    for r in rr:
        m = np.asarray(load_motion_npz(ROOT / r["motion_relpath"],
                                       expected_fps_target=30.0)["motion"], dtype=np.float64)
        if s1 is None:
            s1 = np.zeros(m.shape[1:], np.float64); s2 = np.zeros(m.shape[1:], np.float64)
        s1 += m.sum(0); s2 += (m ** 2).sum(0); n += m.shape[0]
    mu = s1 / n
    var = np.maximum(s2 / n - mu ** 2, 0.0)
    sd = np.sqrt(var)
    sd[sd < 1e-6] = 1.0                     # constant cells: normalize to exactly 0
    means[rig] = mu.astype(np.float32)
    stds[rig] = (sd - _STD_FLOOR).astype(np.float32)      # repo convention
    counts[rig] = n
    if i % 20 == 0:
        print(f"  [{i:3d}] {rig:22s} frames={n:6d} |mu| max={np.abs(mu).max():7.3f} "
              f"sd range {sd.min():.4f}..{sd.max():.3f}")

payload = {f"mean__{k}": v for k, v in means.items()}
payload.update({f"std__{k}": v for k, v in stds.items()})
payload["__meta"] = json.dumps({
    "cohort": "per_rig_all_accepted_clips",
    "cohort_note": "ALL clips of each rig (user 2026-08-21), not train-only; transductive for "
                   "held-out rigs -- never report these as zero-shot without saying so",
    "convention": "raw = x * (std + _STD_FLOOR) + mean; stored std is (empirical_std - _STD_FLOOR)",
    "std_floor": float(_STD_FLOOR),
    "zero_variance_policy": "empirical_std < 1e-6 -> 1.0 (cell normalizes to exactly 0)",
    "n_rigs": len(means), "frames_per_rig": counts,
    "generation_id": json.loads((ROOT / "generation.json").read_text())["generation_id"],
})
OUT.parent.mkdir(exist_ok=True)
np.savez(OUT, **payload)
print(f"[OK] {OUT}  ({OUT.stat().st_size/1e6:.1f} MB)  sha256 "
      f"{hashlib.sha256(OUT.read_bytes()).hexdigest()[:16]}")
