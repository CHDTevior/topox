"""Build a MINIMAL animal subset of the cleaned v4b training dataset, with everything
needed to drive skinning, and VERIFY the representation is self-consistent.

Verification = the load-bearing part: FK(rot6d channels, rest offsets) must reproduce
the RIC position channels. If that holds, the representation can drive a rig.
"""
import json
import os
import pickle
import sys

import numpy as np

sys.path.insert(0, "/scratch/ts1v23/workspace/noKslot_clean")
from src.data.anytop_dataset import _recover_world_positions  # RIC/position route -> world
from src.data.anytop_rot6d_fk import recover_from_bvh_rot_np  # the FIXED FK (never re-derive)

R = "data/animo4d_L4TB_plus_human_v4b272neutral"
OUT = "scratch/animal_minipack"
os.makedirs(os.path.join(OUT, "clips"), exist_ok=True)

CLIPS = [
    # (filename, why)
    ("PZ_African_Elephant_Male_african_elephant_male__animationmotionextractedlocomotion_maniset28155cd8__african_elephant_male_runbase_3.npy",
     "PZ (Planet-Zoo-style rig) quadruped locomotion — the bulk of the animal data"),
    ("Deer___WalkForward_288.npy",
     "TrueBones (BVH-native) quadruped walk — the other animal source"),
    ("Bat___SlowFly_68.npy",
     "TrueBones non-quadruped (wings) — different topology, stresses the graph representation"),
]

cond = np.load(os.path.join(R, "cond.npy"), allow_pickle=True).item()
texts = json.load(open(os.path.join(R, "motion_texts_by_file.json")))


def obj_type_of(fn):
    """Resolve the clip's object_type by longest-prefix match against cond keys."""
    stem = fn[:-4]
    cands = [k for k in cond if stem.startswith(k)]
    if not cands:
        # TrueBones style: Deer___WalkForward_288 -> object_type is the leading token
        head = stem.split("___")[0].split("_")[0]
        cands = [k for k in cond if k == head or k.startswith(head)]
    if not cands:
        raise KeyError(f"no object_type for {fn}")
    return max(cands, key=len)


def cap_of(fn):
    v = texts.get(fn) or texts.get(fn[:-4])
    if isinstance(v, dict):
        for k in ("primary_caption", "caption", "text"):
            if k in v:
                x = v[k]
                return x[0] if isinstance(x, list) else x
        if "captions" in v and v["captions"]:
            return v["captions"][0]
    if isinstance(v, list) and v:
        return v[0]
    return str(v)


report = {"clips": [], "representation": {}}

for fn, why in CLIPS:
    p = os.path.join(R, "motions", fn)
    if not os.path.exists(p):
        print(f"[SKIP] missing {fn}")
        continue
    m = np.load(p)                       # [T, J, 13] raw (un-normalized) motion
    ot = obj_type_of(fn)
    c = cond[ot]
    parents = np.asarray(c["parents"])
    offsets = np.asarray(c["offsets"], dtype=np.float64)
    names = list(c["joints_names"])
    J = len(parents)
    T = m.shape[0]

    # ---- VERIFY: the two independent decode routes must agree ----
    # ROUTE A (rotation): FK on rot6d(ch3:9) + rest offsets  -> world  [this is what drives a rig]
    # ROUTE B (position): decode RIC(ch0:3)                  -> world  [ground truth]
    # If A == B, the rotation channels are valid and CAN drive skinning.
    mm = m[:, :J, :].astype(np.float64)
    fk = np.asarray(recover_from_bvh_rot_np(mm, parents, offsets), dtype=np.float64)[:, :J, :3]
    pos = np.asarray(_recover_world_positions(mm), dtype=np.float64)[:, :J, :3]

    err = np.linalg.norm(fk - pos, axis=-1)                    # [T,J] world-space abs error
    scale = float(np.linalg.norm(pos - pos[:, :1, :], axis=-1).mean()) + 1e-8  # mean bone extent
    rel = float(err.mean() / scale)

    rec = {
        "file": fn,
        "why": why,
        "object_type": ot,
        "num_frames": int(T),
        "num_joints": int(J),
        "motion_shape": list(m.shape),
        "motion_dtype": str(m.dtype),
        "caption": cap_of(fn),
        "fk_vs_ric_mean_abs_err": float(err.mean()),
        "fk_vs_ric_p95_abs_err": float(np.percentile(err, 95)),
        "fk_vs_ric_relative_err": rel,
        "self_consistent": bool(rel < 0.05),
    }
    report["clips"].append(rec)
    print(f"[{ot}] J={J} T={T} | FK-vs-RIC mean={err.mean():.6f} p95={np.percentile(err,95):.6f} "
          f"rel={rel:.4%} -> {'SELF-CONSISTENT' if rec['self_consistent'] else 'MISMATCH'}")

    # ---- pack ----
    base = fn[:-4]
    d = os.path.join(OUT, "clips", base)
    os.makedirs(d, exist_ok=True)
    np.save(os.path.join(d, "motion_13ch.npy"), m)            # raw, exactly as training reads it
    np.save(os.path.join(d, "fk_world_positions.npy"), fk.astype(np.float32))    # ROUTE A: FK(rot6d) -> world (drives the rig)
    np.save(os.path.join(d, "ric_world_positions.npy"), pos.astype(np.float32))  # ROUTE B: RIC decode -> world (ground truth)
    skel = {
        "object_type": ot,
        "num_joints": int(J),
        "joints_names": names,
        "parents": parents.tolist(),
        "offsets": offsets.tolist(),                            # rest-pose bone offsets (parent->child)
        "kinematic_chains": [list(map(int, ch)) for ch in c["kinematic_chains"]],
        "tpos_first_frame": np.asarray(c["tpos_first_frame"]).tolist(),
    }
    json.dump(skel, open(os.path.join(d, "skeleton.json"), "w"), indent=1)
    np.savez(os.path.join(d, "normalization.npz"),
             mean=np.asarray(c["mean"]), std=np.asarray(c["std"]))
    open(os.path.join(d, "caption.txt"), "w").write(rec["caption"] + "\n")
    json.dump(rec, open(os.path.join(d, "meta.json"), "w"), indent=1)

report["representation"] = {
    "layout": "[T, J, 13] float — per-frame, per-joint",
    "ch0:3": "RIC position (root-relative-ish world position route)",
    "ch3:9": "rot6d — ⚠ PER-PARENT convention: token[j].ch3:9 stores the rotation of joint j's PARENT, not of j itself",
    "ch9:12": "linear velocity",
    "ch12": "foot/ground contact flag",
    "normalization": "the on-disk .npy is RAW. training normalizes with (x - mean) / std from cond[object_type]",
    "fk": "use recover_from_bvh_rot_np(motion, offsets, parents) — shipped as fk.py. Do NOT re-derive: the per-parent convention + root handling cause a double-root-rotation bug if you write your own.",
}
json.dump(report, open(os.path.join(OUT, "REPORT.json"), "w"), indent=1)
print("\n=== wrote", OUT)
