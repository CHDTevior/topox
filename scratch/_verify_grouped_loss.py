#!/usr/bin/env python3
"""Check the grouped loss: (a) the groups tile every (joint, channel) cell exactly once,
(b) the resulting gradient shares on REAL TrueBones data match what was predicted. Read-only."""
import sys, pickle
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.models.v2.dit_motion import _GROUP_SPEC, KIMODO_GAMMAS, grouped_loss

# ---- (a) tiling: every (j,c) covered exactly once, for a few J values ----
for J in (2, 5, 42, 144):
    cover = np.zeros((J, 13), dtype=int)
    for name, (js, cs) in _GROUP_SPEC.items():
        cover[js, :][:, cs] += 1
    ok = (cover == 1).all()
    print(f"J={J:3d}  每个 (关节,通道) 被覆盖恰好一次: {ok}   (min={cover.min()} max={cover.max()})")
    assert ok, f"tiling broken at J={J}"

# ---- (b) real-data shares ----
from src.data.anytop_dataset import AnyTopDataset, _STD_FLOOR
R = "data/animo4d_L4TB_plus_human_v4b272neutral"
tb = set(k for k in pickle.load(open(f"{R}/_cond_normalized_J144.pkl","rb")).keys()
         if not k.startswith("PZ_") and not k.startswith("HML3D"))
ds = AnyTopDataset(data_root=R, split="all", num_frames=300, max_joints=144,
                   load_captions=False, caption_emb_cache=None, random_caption=False, augment=False)
acc = {k: 0.0 for k in KIMODO_GAMMAS}; unw = {k: 0.0 for k in KIMODO_GAMMAS}; seen = 0
for i in range(len(ds)):
    try: item = ds[i]
    except Exception: continue
    ot = item.get("object_type") or str(item.get("motion_id","?")).split("___")[0]
    if ot not in tb: continue
    T, J = int(item["num_frames"]), int(item["num_joints"])
    x = torch.from_numpy(np.asarray(item["anytop_x"])[:J,:,:T].transpose(2,0,1)).float()[None]  # [1,T,J,13]
    m = torch.ones_like(x)
    # err2 = x^2 : the error a predict-zero model makes, i.e. the initial gradient landscape
    _, parts = grouped_loss(x**2, m, KIMODO_GAMMAS)
    for k, v in parts.items():
        acc[k] += float(v) * KIMODO_GAMMAS[k]; unw[k] += float(v)
    seen += 1
    if seen >= 300: break

print(f"\n在 {seen} 条真实 TrueBones clip 上 (err2 = x^2,即初始时的梯度地形):\n")
tw, tu = sum(acc.values()), sum(unw.values())
print(f"{'组':12s} {'gamma':>6s} {'加权后份额':>12s} {'未加权份额':>12s}")
for k in KIMODO_GAMMAS:
    print(f"{k:12s} {KIMODO_GAMMAS[k]:6.1f} {100*acc[k]/tw:11.2f}% {100*unw[k]/tu:11.2f}%")
r_new = 100*(acc['root_pos']+acc['root_rot'])/tw
r_old = 100*(unw['root_pos']+unw['root_rot'])/tu
print(f"\nroot 合计:  分组均值+gamma 加权 -> {r_new:.2f}%   (分组均值但不加 gamma -> {r_old:.2f}%)")
