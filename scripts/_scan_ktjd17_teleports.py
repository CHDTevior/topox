#!/usr/bin/env python3
"""Quantify per-frame teleports across the whole PZ+Human-312 corpus.

User verdict 2026-08-21, after watching the GT of the worst velocity outlier: "这种的就是出现了
突然的反转，感觉是经历了突变，要么就是源数据有问题。瞬移的都不能要按理说."

Criterion is physical and rig-relative: ch9:11 IS the 30 fps world-space position difference, so
one frame's displacement is |v| / fps. A rig's scale is the bounding-box diagonal of its rest pose.
A joint that moves more than alpha * scale in a single frame did not move -- it jumped.

Reports, for several alpha, how many frames / clips / rigs are affected, so the cut-off can be
chosen on evidence rather than taste. Writes the per-clip offender list so a cut can be executed.
"""
import json, sys, collections
from pathlib import Path
import numpy as np

D = Path("dataset/ktjd17_pz_human312")
ALPHAS = [0.02, 0.05, 0.10, 0.20, 0.50]
OUT = Path("configs/pzh312_teleport_scan.json")

scale = {}
for p in sorted((D / "skeletons").glob("*.npz")):
    sk = np.load(p, allow_pickle=True)
    P = np.asarray(sk["P_rest_global"], np.float64)
    scale[p.stem] = float(np.linalg.norm(P.max(0) - P.min(0)))

rows = [json.loads(l) for l in (D / "manifests/clips.jsonl").open()]
print(f"{len(rows)} clips, {len(scale)} rigs", flush=True)
n_frames = collections.Counter(); n_clips = collections.Counter()
rigs_hit = collections.defaultdict(set); worst_per_clip = {}
per_clip_hits = collections.defaultdict(dict)
tot_f = 0
for i, r in enumerate(rows):
    rig, cid = r["rig_id"], r["clip_id"]
    z = np.load(D / r["motion_relpath"], allow_pickle=True)
    m = z["motion"] if "motion" in z else z[list(z.keys())[0]]
    fps = float(r.get("fps_target", 30.0))
    disp = np.linalg.norm(np.asarray(m[:, :, 9:12], np.float64), axis=-1) / fps   # [T,J]
    rel = disp / max(scale[rig], 1e-9)
    mx = float(rel.max())
    worst_per_clip[cid] = mx
    tot_f += rel.shape[0]
    for a in ALPHAS:
        hit = rel > a
        nf = int(hit.any(axis=1).sum())            # frames where ANY joint jumped
        if nf:
            n_frames[a] += nf; n_clips[a] += 1; rigs_hit[a].add(rig)
            per_clip_hits[a][cid] = nf
    if i % 10000 == 0:
        print(f"  ...{i}/{len(rows)}", flush=True)

print(f"\ntotal frames = {tot_f:,} over {len(rows):,} clips / {len(scale)} rigs")
print(f"per-clip worst single-frame displacement, as a fraction of rig scale:")
w = np.array(list(worst_per_clip.values()))
print("   " + "  ".join(f"p{q}={np.percentile(w,q):.4f}" for q in (50, 90, 99, 99.9)) +
      f"  max={w.max():.4f}")
print(f"\n{'alpha':>6} {'frames':>10} {'frame%':>8} {'clips':>8} {'clip%':>8} {'rigs':>6}")
for a in ALPHAS:
    print(f"{a:6.2f} {n_frames[a]:10,} {100*n_frames[a]/tot_f:7.4f}% {n_clips[a]:8,} "
          f"{100*n_clips[a]/len(rows):7.3f}% {len(rigs_hit[a]):6d}")
rep = {"alphas": ALPHAS, "total_frames": tot_f, "total_clips": len(rows),
       "criterion": "max_j |v_j|/fps / rig_rest_bbox_diagonal > alpha",
       "per_alpha": {str(a): {"frames": n_frames[a], "clips": n_clips[a],
                              "rigs": sorted(rigs_hit[a])} for a in ALPHAS},
       "clip_hits": {str(a): per_clip_hits[a] for a in ALPHAS},
       "worst_per_clip": worst_per_clip}
OUT.write_text(json.dumps(rep))
print(f"\n[OK] wrote {OUT}")
