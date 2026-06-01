"""Confirm what gt_fk_mismatch=0.2921 (train-mean) actually means, vs the
preflight's ~1-9% bbox numbers, vs the user's "random 5 ~1%" memory.

gt_fk_mismatch (losses.py) = masked L1(|FK(gt) - RIC(gt)|) over xyz, ABSOLUTE
  (same unit as pos loss). Uses ALL valid joints (joint_mask).
preflight = L2 norm / bbox-diagonal * 100, RELATIVE %, main-nonhelper.

This script prints, per object (mammal outliers + reptiles) and for a 60-clip
train sample: abs-L1 (gt_fk_mismatch unit), abs-L2, bbox-diag, L1/bbox%, L2/bbox%.
Run on rose11: python scripts/_check_gt_fk_units.py
"""
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data.anytop_dataset import AnyTopDataset, _recover_world_positions, _STD_FLOOR  # noqa
from src.data.anytop_rot6d_fk import recover_from_bvh_rot_np  # noqa

ds = AnyTopDataset(split="train", val_frac=0.05, seed=42,
                   data_root=str(ROOT / "data/anytop_planet_zoo_clean_L2"),
                   num_frames=64, max_joints=144, caption_emb_cache=None)


def compute(it):
    J = int(it["num_joints"]); T = int(it["num_frames"])
    ax = np.asarray(it["anytop_x"], np.float32)
    mean = np.asarray(it["anytop_mean"], np.float32); std = np.asarray(it["anytop_std"], np.float32)
    raw = np.transpose(ax, (2, 0, 1))[:T, :J, :] * (std[:J][None] + _STD_FLOOR) + mean[:J][None]
    parents = np.asarray([int(p) for p in it["parent_indices"][:J]], int)
    offsets = np.asarray(it["rest_offsets"], np.float32)[:J]
    ric = _recover_world_positions(raw)                  # [T,J,3]
    fk = recover_from_bvh_rot_np(raw, parents, offsets)  # [T,J,3]
    err_l1 = np.abs(fk - ric).sum(-1)                    # [T,J] L1 = gt_fk_mismatch unit
    err_l2 = np.linalg.norm(fk - ric, axis=-1)           # [T,J] L2 = preflight unit
    bb = ric.reshape(-1, 3)
    diag = float(np.linalg.norm(bb.max(0) - bb.min(0))) or 1e-9
    return float(err_l1.mean()), float(err_l2.mean()), diag, it["object_type"]


MAMMALS = ["Pronghorn", "Gorilla", "Saiga", "Rhino", "Bear", "Antelope", "Macaque"]
targets = ["Pronghorn", "Gorilla", "Saiga", "Rhino", "Sun_Bear",
           "Komodo", "Crocodile", "Seal", "Water_Monitor", "Alligator"]
print(f"{'object':42s} {'absL1':>8s} {'absL2':>8s} {'bbox':>7s} {'L1/bbox%':>9s} {'L2/bbox%':>9s}")
seen = set()
for i in range(len(ds)):
    it = ds[i]; o = it["object_type"]
    k = next((t for t in targets if t in o), None)
    if k and k not in seen:
        seen.add(k)
        l1, l2, diag, _ = compute(it)
        tag = "MAMMAL-outlier" if any(m in o for m in MAMMALS) else "reptile/other"
        print(f"{o:42s} {l1:8.4f} {l2:8.4f} {diag:7.3f} {100*l1/diag:9.2f} {100*l2/diag:9.2f}  {tag}")
    if len(seen) >= len(targets):
        break

idxs = np.linspace(0, len(ds) - 1, 60).astype(int)
l1s, l2pcts = [], []
for i in idxs:
    l1, l2, diag, _ = compute(ds[int(i)])
    l1s.append(l1); l2pcts.append(100 * l2 / diag)
l1s = np.array(l1s); l2pcts = np.array(l2pcts)
print(f"\n=== train 60-clip 平均 ===")
print(f"  abs-L1 (gt_fk_mismatch 口径): mean={l1s.mean():.4f} median={np.median(l1s):.4f} p95={np.percentile(l1s,95):.4f}")
print(f"     → 对照训练实测 gt_fk_mismatch = 0.2921 (train epoch mean)")
print(f"  L2/bbox% (preflight 口径):    mean={l2pcts.mean():.2f}% median={np.median(l2pcts):.2f}% p95={np.percentile(l2pcts,95):.2f}%")
print(f"     → 对照 preflight train main_nonhelper mean=9.63% median=3.44%")
