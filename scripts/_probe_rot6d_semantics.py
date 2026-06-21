"""What does rot6d (ch3:9) actually MEAN in each convention?

Hypothesis: AnyTop's encode does reordered[j]=cont6d[parent[j]], so for a
branching joint p, ALL children of p carry the SAME rot6d (= p's BVH rotation).
HumanML3D's rot_data[j] is the per-bone IK rotation of joint j's incoming bone,
so siblings carry DIFFERENT rot6d. If true, that is the precise semantic gap:
AnyTop ch3:9 is shared-per-parent; HumanML3D ch3:9 is independent-per-bone.

Check both on REAL data: animal truebones motions vs converted human motions.
"""
import sys
from pathlib import Path
import numpy as np

REPO = "/iridisfs/scratch/ts1v23/workspace/noKslot_clean"
sys.path.insert(0, REPO)

ANIMAL = REPO + "/data/anytop_truebones"
HUMAN = REPO + "/data/humanml3d_anytop13"


def parents_of(cond_obj):
    return np.asarray(cond_obj["parents"], dtype=int)


def branching(parents):
    kids = {}
    for j, p in enumerate(parents):
        if p >= 0:
            kids.setdefault(int(p), []).append(j)
    return {p: cs for p, cs in kids.items() if len(cs) >= 2}


def check(name, raw, parents):
    """For each branching joint, are the children's ch3:9 identical across frames?"""
    br = branching(parents)
    print(f"\n[{name}]  shape={raw.shape}  branching joints: {list(br.keys())}")
    for p, cs in list(br.items())[:4]:
        # pairwise max abs diff of ch3:9 between the children, over all frames
        rots = [raw[:, c, 3:9] for c in cs]      # each [T,6]
        maxdiff = 0.0
        for a in range(len(rots)):
            for b in range(a + 1, len(rots)):
                maxdiff = max(maxdiff, float(np.abs(rots[a] - rots[b]).max()))
        verdict = "IDENTICAL (shared parent rot)" if maxdiff < 1e-5 else "DIFFERENT (independent per-bone)"
        print(f"  parent {p:>3} children {cs}:  max|ch3:9 diff between siblings| = {maxdiff:.4f}  -> {verdict}")


# ---- animal ----
acond = np.load(Path(ANIMAL) / "cond.npy", allow_pickle=True).item()
aobj = next(iter(acond))
amot = sorted((Path(ANIMAL) / "motions").glob("*.npy"))
# match a motion to its object: filename stem starts with object type
def load_animal():
    for f in amot:
        for ot in acond:
            if f.stem.startswith(ot) or ot.split("_")[0] in f.stem:
                return ot, np.load(f)
    return aobj, np.load(amot[0])

ot, araw = load_animal()
check(f"ANIMAL {ot}", araw, parents_of(acond[ot]))

# ---- human ----
hcond = np.load(Path(HUMAN) / "cond.npy", allow_pickle=True).item()["HML3D_Human"]
hraw = np.load(Path(HUMAN) / "motions" / "HML3D_Human_000000.npy")
check("HUMAN HML3D_Human_000000", hraw, parents_of(hcond))

print("\nIf animal siblings are IDENTICAL but human siblings DIFFER, that is the")
print("precise rot6d semantic gap: AnyTop ch3:9 = shared per-parent BVH rotation;")
print("HumanML3D ch3:9 = independent per-bone rotation. Our FK reindex assumes the")
print("former, so it discards human per-bone info at branches -> Gate C ~14%.")
