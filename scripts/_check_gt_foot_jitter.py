"""Investigate user's report: GT (motion_features) has a back foot that flails
during descent. Is it a DATA glitch (foot joint position erratic in the recovered
world-space motion) or rendering? For the rendered val clips, compute per-joint
frame-to-frame jitter, flag LEAF joints (end-effectors, likely feet) whose jitter
is anomalously high vs the body median."""
import sys
import numpy as np
import torch
sys.path.insert(0, ".")
from src.data.anytop_dataset import AnyTopDataset, collate_fn as anytop_collate_fn
from src.models.graph_salad.batch import GraphMotionBatch

DATA = "data/anytop_planet_zoo_clean_L2"
SPECIES = ["PZ_Koala_Female", "PZ_Jaguar_Female", "PZ_Proboscis_Monkey_Juvenile",
           "PZ_Siberian_Tiger_Juvenile", "PZ_Hippopotamus_Male"]

ds = AnyTopDataset(split="val", random_caption=False, random_crop=False,
                   num_frames=260, max_joints=144,
                   caption_emb_cache="data/anytop_caption_t5_cleanL2_multi.npz",
                   data_root=DATA, load_captions=False)
want = set(SPECIES)
match = [i for i, s in enumerate(ds.samples) if s.get("object_type") in want]

picked = set()
for i in match:
    item = ds[i]
    sp = item["object_type"]
    if sp in picked:
        continue
    picked.add(sp)
    raw = anytop_collate_fn([item])
    batch = GraphMotionBatch.from_collate_dict(raw)
    J = int(item["num_joints"]); T = int(item["num_frames"])
    parents = [int(p) for p in item["parent_indices"][:J]]
    gt = batch.motion_features[0, :T, :J, :3].cpu().numpy()      # [T,J,3] world pos
    # per-joint frame-to-frame displacement (jitter)
    jit = np.linalg.norm(np.diff(gt, axis=0), axis=-1).mean(axis=0)   # [J]
    # acceleration (jerk) per joint — flailing = high 2nd diff
    acc = np.linalg.norm(np.diff(gt, n=2, axis=0), axis=-1).mean(axis=0)  # [J]
    med = np.median(jit)
    # leaf joints = never appear as a parent
    has_child = set(parents)
    leaves = [j for j in range(J) if j not in has_child]
    # top-5 jitter joints
    order = np.argsort(jit)[::-1][:5]
    print(f"\n=== {sp}  J={J} T={T}  caption={item.get('caption','')[:50]!r}")
    print(f"  body median joint-jitter={med:.4f}   max={jit.max():.4f} (={jit.max()/max(med,1e-9):.1f}x median)")
    print(f"  #leaf(end-effector) joints={len(leaves)}")
    print(f"  top-5 jitter joints: " + ", ".join(
        f"j{j}{'(leaf)' if j in leaves else ''}={jit[j]:.3f}/{jit[j]/max(med,1e-9):.1f}x acc={acc[j]:.3f}" for j in order))
    # flag leaf joints with jitter > 3x median (candidate flailing feet)
    flail = [j for j in leaves if jit[j] > 3 * med]
    print(f"  ⚠ leaf joints with jitter>3x median (candidate flail): "
          + (", ".join(f"j{j}={jit[j]/med:.1f}x" for j in flail) if flail else "none"))
