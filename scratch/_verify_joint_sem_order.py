#!/usr/bin/env python3
"""Does the joint_semantics table match the joint order the dataset actually serves?

The dataset BFS-reorders joints (new_to_old_perm) and then verifies the semantics table against a
sha256 of the served joint_names (anytop_dataset.py:1278-1283). Loading the npz by hand -- as the
new pipeline currently does -- bypasses that check, so a stale table would silently pair the wrong
embedding with each joint. This exercises the checked path on EVERY TrueBones rig. Read-only.
"""
import pickle, sys, traceback
from collections import defaultdict
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.anytop_dataset import AnyTopDataset

R = "data/animo4d_L4TB_plus_human_v4b272neutral"
SEM = "data/joint_semantics_llm2vec_v1.npz"
tb = sorted(k for k in pickle.load(open(f"{R}/_cond_normalized_J144.pkl","rb")).keys()
            if not k.startswith("PZ_") and not k.startswith("HML3D"))

ds = AnyTopDataset(data_root=R, split="all", num_frames=300, max_joints=144,
                   load_captions=False, caption_emb_cache=None, random_caption=False,
                   augment=False, joint_semantics=SEM, species_whitelist=tb)
print(f"[sem] dataset {len(ds)} clips over {len({s['object_type'] for s in ds.samples})} rigs\n")

first = {}
for i, s in enumerate(ds.samples):
    first.setdefault(s["object_type"], i)

ok, bad, missing = [], [], []
for ot, idx in sorted(first.items()):
    try:
        it = ds[idx]
    except KeyError as e:
        missing.append((ot, str(e)[:90])); continue
    except ValueError as e:
        bad.append((ot, str(e)[:160])); continue
    except Exception as e:
        bad.append((ot, f"{type(e).__name__}: {str(e)[:140]}")); continue
    sem = it.get("joint_semantics")
    J = int(it["num_joints"])
    if sem is None:
        missing.append((ot, "no joint_semantics key in item")); continue
    ok.append((ot, J, tuple(np.asarray(sem).shape)))

print(f"顺序校验通过 : {len(ok)} 个骨架")
print(f"顺序不一致   : {len(bad)}")
print(f"表里缺条目   : {len(missing)}")
for ot, m in bad[:6]:     print(f"   MISMATCH {ot}: {m}")
for ot, m in missing[:6]: print(f"   MISSING  {ot}: {m}")
if ok:
    print(f"\n样例(骨架, 真实关节数, joint_semantics 形状):")
    for ot, J, sh in ok[:6]:
        print(f"   {ot:16s} J={J:3d}  sem{sh}")
    # 关键:非零行数应等于真实关节数,不能是 padding 填出来的
    print("\n非零语义行数 vs 真实关节数(应相等):")
    for ot, idx in list(sorted(first.items()))[:6]:
        it = ds[idx]; J = int(it["num_joints"])
        sem = np.asarray(it["joint_semantics"])
        nz = int((np.abs(sem).sum(-1) > 0).sum())
        flag = "OK" if nz == J else "MISMATCH"
        print(f"   {ot:16s} 非零 {nz:3d} / J {J:3d}   {flag}")
print("\n" + ("全部通过 —— 数据顺序一致" if not bad and not missing else "有问题,见上"))
