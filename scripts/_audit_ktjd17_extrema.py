#!/usr/bin/env python3
"""Exhaustive normalized-magnitude audit over the FULL train cohort (codex 2026-08-21 admission
gate before gamma calibration).

Two separations the earlier sampled scan got wrong:
  * DEMO context frames vs SUPERVISED TARGET frames. The loss never touches the demo slot, so a
    large value there is an input-scale defect, not an objective defect -- different severity,
    different fix.
  * clip-balanced, not rig-balanced. The 3,200-window rig-balanced scan reports how often a RIG
    is extreme, which is not how often a training BATCH is.
Every offending item is recorded with enough identity to replay it.
"""
import sys, json, collections
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.ktjd17_incontext import Ktjd17Base, ktjd17_split_names
from src.data.incontext_pairs import InContextPairs, collate
from torch.utils.data import DataLoader

R = "dataset/ktjd17_pz_human312"
THRESH = float(sys.argv[1]) if len(sys.argv) > 1 else 20.0
base = Ktjd17Base(R, caption_emb_cache="data/anytop_caption_llm2vec_v4b272neutral_multi",
                  joint_semantics="data/joint_semantics_llm2vec_pzh312_v1.npz",
                  percell_stats="data/pzh312_norm_stats_v1.npz")
names = ktjd17_split_names(R)
ds = InContextPairs(base, names["train"], names["train"], balance_skeletons=False, seed=0,
                    demo_rest=True, demo_frames=1)
g = torch.Generator(); g.manual_seed(0)
dl = DataLoader(ds, batch_size=16, shuffle=False, num_workers=24, collate_fn=collate, generator=g)
lut = {r: torch.from_numpy(base.static_masks(r)["channel_valid"]) for r in ds.types}
dm, tm = [], []
off_d, off_t = [], []
cell_t = collections.Counter()
for bi, b in enumerate(dl):
    x = b["x"][..., :17]; ist = b["is_target"]
    for k, ot in enumerate(b["object_type"]):
        cv = lut[ot]; J = cv.shape[0]
        m = (b["valid"][k][:, :J, None] & cv[None])
        xi = (x[k, :, :J] * m).abs()
        for role, sel, store, off in (("demo", ~ist[k], dm, off_d), ("target", ist[k], tm, off_t)):
            a = xi[sel]
            if not a.numel():
                continue
            v = float(a.max()); store.append(v)
            if v > THRESH:
                f, j, c = np.unravel_index(int(a.argmax()), tuple(a.shape))
                off.append((v, ot, int(j), int(c), int(f), b["motion_id"][k]))
                if role == "target":
                    cell_t[(ot, int(j), int(c))] += 1
    if bi % 500 == 0:
        print(f"  ...{bi*16} windows", flush=True)
dm, tm = np.array(dm), np.array(tm)
rep = {"threshold": THRESH, "windows": int(len(tm)),
       "percell_sha256": base.provenance["percell_sha256"],
       "generation_id": base.provenance["generation_id"]}
for nm, arr, off in (("demo", dm, off_d), ("target", tm, off_t)):
    rep[nm] = {"p50": float(np.percentile(arr, 50)), "p90": float(np.percentile(arr, 90)),
               "p99": float(np.percentile(arr, 99)), "p99.99": float(np.percentile(arr, 99.99)),
               "max": float(arr.max()), "n_over": len(off),
               "frac_over": len(off) / len(arr)}
    print(f"{nm:7s} |x|: p50={rep[nm]['p50']:6.2f} p90={rep[nm]['p90']:6.2f} "
          f"p99={rep[nm]['p99']:7.2f} p99.99={rep[nm]['p99.99']:8.2f} max={rep[nm]['max']:9.2f}  "
          f">{THRESH:g}: {len(off)} ({100*rep[nm]['frac_over']:.3f}%)")
off_t.sort(reverse=True); off_d.sort(reverse=True)
rep["target_top"] = [{"v": v, "rig": r, "j": j, "ch": c, "frame": f, "motion_id": mid}
                     for v, r, j, c, f, mid in off_t[:50]]
rep["demo_top"] = [{"v": v, "rig": r, "j": j, "ch": c, "frame": f, "motion_id": mid}
                   for v, r, j, c, f, mid in off_d[:50]]
rep["target_cells"] = [{"rig": r, "j": j, "ch": c, "n_windows": n}
                       for (r, j, c), n in cell_t.most_common(40)]
print("\nTARGET offenders, top 12 (value, rig, joint, channel, frame):")
for v, r, j, c, f, mid in off_t[:12]:
    print(f"  {v:9.2f}  {r:34s} j{j:<3d} ch{c:<2d} f{f:<4d} {mid}")
print("\nTARGET offending CELLS by window count:")
for (r, j, c), n in cell_t.most_common(10):
    print(f"  {n:6d} windows  {r:34s} j{j:<3d} ch{c}")
print("\nDEMO offenders, top 6:")
for v, r, j, c, f, mid in off_d[:6]:
    print(f"  {v:9.2f}  {r:34s} j{j:<3d} ch{c:<2d}")
Path("configs/pzh312_extrema_audit.json").write_text(json.dumps(rep, indent=1))
print("\n[OK] wrote configs/pzh312_extrema_audit.json")
