#!/usr/bin/env python3
"""Per-(joint-role x channel-group) energy audit on TrueBones, in the NORMALISED space the loss
sees. An unweighted MSE weights each cell by its energy share, so this measures the weighting we
are implicitly applying today. Read-only.

The root row is singled out because in our layout root is just 1 of J joints, yet it carries global
translation (ch9,11) and heading (ch3:9), i.e. exactly the quantities reported as failing.
"""
import sys, pickle
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.anytop_dataset import AnyTopDataset, _STD_FLOOR

R = "data/animo4d_L4TB_plus_human_v4b272neutral"
tb = set(k for k in pickle.load(open(f"{R}/_cond_normalized_J144.pkl","rb")).keys()
         if not k.startswith("PZ_") and not k.startswith("HML3D"))
ds = AnyTopDataset(data_root=R, split="all", num_frames=300, max_joints=144,
                   load_captions=False, caption_emb_cache=None, random_caption=False, augment=False)

GROUPS = [("ric_pos", slice(0,3)), ("rot6d", slice(3,9)), ("vel", slice(9,12)), ("contact", slice(12,13))]
acc = {("root",g):0.0 for g,_ in GROUPS} | {("body",g):0.0 for g,_ in GROUPS}
n_root_cells = n_body_cells = 0
seen = 0; Js = []
for i in range(len(ds)):
    try: item = ds[i]
    except Exception: continue
    ot = item.get("object_type") or str(item.get("motion_id","?")).split("___")[0]
    if ot not in tb: continue
    T, J = int(item["num_frames"]), int(item["num_joints"])
    x = np.asarray(item["anytop_x"])[:J,:,:T].transpose(2,0,1)      # [T,J,13] normalised
    for g, sl in GROUPS:
        acc[("root",g)] += float((x[:,0:1,sl]**2).sum())
        acc[("body",g)] += float((x[:,1:,sl]**2).sum())
    n_root_cells += T*1; n_body_cells += T*(J-1); Js.append(J); seen += 1
    if seen >= 400: break

tot = sum(acc.values())
print(f"[audit2] {seen} clips, J 中位 {np.median(Js):.0f}\n")
print(f"{'':22s} {'能量占比':>10s} {'每单元均能':>12s}")
for role, cells in [("root", n_root_cells), ("body", n_body_cells)]:
    sub = sum(v for (r,_),v in acc.items() if r==role)
    print(f"--- {role} ({'1 个关节' if role=='root' else 'J-1 个关节'}) 合计 {100*sub/tot:.2f}% ---")
    for g,sl in GROUPS:
        v = acc[(role,g)]
        print(f"  {g:18s} {100*v/tot:9.2f}% {v/(cells*(sl.stop-sl.start)):12.4f}")
print(f"\n>>> root 行在未加权 MSE 下总共只拿到 {100*sum(v for (r,_),v in acc.items() if r=='root')/tot:.2f}% 的梯度")
