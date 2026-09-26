"""One clip per ANIMAL species from the cleaned v4b training dataset, each packaged
with everything needed to drive a rig, and each VERIFIED (FK(rot6d) == RIC positions).

Selection per species: prefer a locomotion clip (walk/run/trot/gallop/...), longest
one under a frame cap; else the longest clip the species has.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, "/scratch/ts1v23/workspace/noKslot_clean")
from src.data.anytop_dataset import _recover_world_positions
from src.data.anytop_rot6d_fk import recover_from_bvh_rot_np

R = "data/animo4d_L4TB_plus_human_v4b272neutral"
OUT = "scratch/species_pack"
MAX_FRAMES = 300          # cap so a species with one giant clip doesn't bloat the pack
LOCOMOTION = ("walk", "run", "trot", "gallop", "canter", "locomotion", "move", "fly", "swim", "crawl")

os.makedirs(os.path.join(OUT, "clips"), exist_ok=True)

cond = np.load(os.path.join(R, "cond.npy"), allow_pickle=True).item()
texts = json.load(open(os.path.join(R, "motion_texts_by_file.json")))
all_files = sorted(os.listdir(os.path.join(R, "motions")))

# --- map every clip to its object_type via longest-prefix match against cond keys ---
keys_by_len = sorted(cond.keys(), key=len, reverse=True)
by_species = {}
unmapped = 0
for fn in all_files:
    stem = fn[:-4]
    ot = next((k for k in keys_by_len if stem.startswith(k)), None)
    if ot is None:
        unmapped += 1
        continue
    by_species.setdefault(ot, []).append(fn)

# SCOPE (user 2026-07-13): PZ_* only = the 311 AniMo4D / Planet-Zoo-rig species.
# (The merged animal side is 381 = 311 PZ + 70 TrueBones; human is 1 more.)
animal_species = [k for k in by_species if k.startswith("PZ_")]
n_tb = len([k for k in by_species if not k.startswith("PZ_") and "Human" not in k and "HML3D" not in k])
print(f"cond object_types={len(cond)} | mapped species={len(by_species)} | "
      f"PZ (packing)={len(animal_species)} | TrueBones (skipped)={n_tb} | unmapped clips={unmapped}")


def cap_of(fn):
    v = texts.get(fn) or texts.get(fn[:-4])
    if isinstance(v, dict):
        for k in ("primary_caption", "caption", "text"):
            if k in v:
                x = v[k]
                return x[0] if isinstance(x, list) else x
        if v.get("captions"):
            return v["captions"][0]
    if isinstance(v, list) and v:
        return v[0]
    return "" if v is None else str(v)


def pick(fns):
    """Prefer a locomotion clip; among candidates take the longest under MAX_FRAMES."""
    def nframes(f):
        try:
            return int(np.load(os.path.join(R, "motions", f), mmap_mode="r").shape[0])
        except Exception:
            return -1
    loco = [f for f in fns if any(w in f.lower() for w in LOCOMOTION)]
    for pool in (loco, fns):
        cands = [(nframes(f), f) for f in pool]
        cands = [(t, f) for t, f in cands if t > 0]
        if not cands:
            continue
        under = [(t, f) for t, f in cands if t <= MAX_FRAMES]
        return max(under or cands)[1]
    return None


rows, failures = [], []
for i, ot in enumerate(sorted(animal_species), 1):
    fn = pick(by_species[ot])
    if fn is None:
        failures.append({"object_type": ot, "error": "no readable clip"})
        continue
    c = cond[ot]
    parents = np.asarray(c["parents"], dtype=int)
    offsets = np.asarray(c["offsets"], dtype=np.float64)
    J = len(parents)
    m = np.load(os.path.join(R, "motions", fn))
    mm = m[:, :J, :].astype(np.float64)

    try:
        fk = np.asarray(recover_from_bvh_rot_np(mm, parents, offsets), dtype=np.float64)[:, :J, :3]
        pos = np.asarray(_recover_world_positions(mm), dtype=np.float64)[:, :J, :3]
        err = np.linalg.norm(fk - pos, axis=-1)
        mean_err, max_err = float(err.mean()), float(err.max())
        ok = mean_err < 1e-4
    except Exception as e:                                   # fail loud, keep going
        failures.append({"object_type": ot, "file": fn, "error": repr(e)[:200]})
        print(f"[{i:3d}/{len(animal_species)}] FAIL {ot}: {e}")
        continue

    if not ok:
        failures.append({"object_type": ot, "file": fn,
                         "fk_vs_ric_mean": mean_err, "fk_vs_ric_max": max_err})

    d = os.path.join(OUT, "clips", ot)
    os.makedirs(d, exist_ok=True)
    np.save(os.path.join(d, "motion_13ch.npy"), m)
    np.save(os.path.join(d, "fk_world_positions.npy"), fk.astype(np.float32))
    np.save(os.path.join(d, "ric_world_positions.npy"), pos.astype(np.float32))
    json.dump({
        "object_type": ot,
        "num_joints": int(J),
        "joints_names": list(c["joints_names"]),
        "parents": parents.tolist(),
        "offsets": offsets.tolist(),
        "kinematic_chains": [list(map(int, ch)) for ch in c["kinematic_chains"]],
        "tpos_first_frame": np.asarray(c["tpos_first_frame"]).tolist(),
    }, open(os.path.join(d, "skeleton.json"), "w"), indent=1)
    np.savez(os.path.join(d, "normalization.npz"),
             mean=np.asarray(c["mean"]), std=np.asarray(c["std"]))
    open(os.path.join(d, "caption.txt"), "w").write(cap_of(fn) + "\n")

    rows.append({
        "object_type": ot, "clip": fn, "source": "PZ" if ot.startswith("PZ_") else "TrueBones",
        "num_joints": int(J), "num_frames": int(m.shape[0]),
        "n_clips_available": len(by_species[ot]),
        "caption": cap_of(fn),
        "fk_vs_ric_mean": mean_err, "fk_vs_ric_max": max_err, "self_consistent": bool(ok),
    })
    if i % 40 == 0:
        print(f"[{i:3d}/{len(animal_species)}] ... {ot} J={J} T={m.shape[0]} err={mean_err:.2e}")

json.dump({"n_species": len(rows),
           "n_self_consistent": sum(r["self_consistent"] for r in rows),
           "n_failures": len(failures),
           "max_fk_vs_ric_mean_over_all_species": max((r["fk_vs_ric_mean"] for r in rows), default=None),
           "failures": failures,
           "species": rows},
          open(os.path.join(OUT, "SPECIES_TABLE.json"), "w"), indent=1)

with open(os.path.join(OUT, "SPECIES_TABLE.csv"), "w") as f:
    f.write("object_type,source,num_joints,num_frames,fk_vs_ric_mean,self_consistent,clip\n")
    for r in rows:
        f.write(f"{r['object_type']},{r['source']},{r['num_joints']},{r['num_frames']},"
                f"{r['fk_vs_ric_mean']:.3e},{r['self_consistent']},{r['clip']}\n")

print(f"\n=== packed {len(rows)} species | self-consistent {sum(r['self_consistent'] for r in rows)}"
      f" | failures {len(failures)}")
if rows:
    worst = max(rows, key=lambda r: r["fk_vs_ric_mean"])
    print(f"worst FK-vs-RIC: {worst['object_type']} mean={worst['fk_vs_ric_mean']:.3e}")
for x in failures[:10]:
    print("FAILURE:", x)
