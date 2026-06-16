"""Export the rest/T-pose skeleton from cond.npy to BVH (hierarchy + 1 rest frame).
One-off (user 2026-06-07): a BVH to inspect our truebones skeletons in a viewer.
Writes HIERARCHY from offsets+parents+joint_names, MOTION = 1 frame, root@origin,
all rotations 0 (so FK gives exactly the rest pose = the matplotlib T-pose render).
Usage: python scripts/_export_tpose_bvh.py [Sp1 Sp2 ...]
"""
import os
import sys
import numpy as np

COND = "data/anytop_truebones/cond.npy"
OUTDIR = "runs/_qa_tpose/bvh"
species = sys.argv[1:] or ["Horse", "Spider", "Flamingo"]
c = np.load(COND, allow_pickle=True).item()
os.makedirs(OUTDIR, exist_ok=True)


def write_bvh(sp, v, path):
    off = np.asarray(v["offsets"], dtype=float)
    par = np.asarray(v["parents"]).astype(int)
    J = len(par)
    names = list(v.get("joints_names") or v.get("joint_names") or [f"j{i}" for i in range(J)])
    names = [str(n).replace(" ", "_") for n in names]
    children = {i: [] for i in range(J)}
    root = None
    for j in range(J):
        if par[j] < 0:
            root = j
        else:
            children[int(par[j])].append(j)
    assert root is not None, f"{sp}: no root (parent==-1)"

    lines = ["HIERARCHY"]

    def emit(j, depth, is_root):
        ind = "\t" * depth
        o = off[j]
        if is_root:
            lines.append(f"ROOT {names[j]}")
            lines.append(ind + "{")
            ci = "\t" * (depth + 1)
            lines.append(ci + "OFFSET 0.000000 0.000000 0.000000")  # root at origin
            lines.append(ci + "CHANNELS 6 Xposition Yposition Zposition Zrotation Xrotation Yrotation")
        else:
            lines.append(ind + f"JOINT {names[j]}")
            lines.append(ind + "{")
            ci = "\t" * (depth + 1)
            lines.append(ci + f"OFFSET {o[0]:.6f} {o[1]:.6f} {o[2]:.6f}")
            lines.append(ci + "CHANNELS 3 Zrotation Xrotation Yrotation")
        ch = children[j]
        if ch:
            for cc in ch:
                emit(cc, depth + 1, False)
        else:
            # leaf -> End Site (small tip so viewers draw the last bone)
            ci = "\t" * (depth + 1)
            lines.append(ci + "End Site")
            lines.append(ci + "{")
            lines.append("\t" * (depth + 2) + "OFFSET 0.000000 0.000000 0.000000")
            lines.append(ci + "}")
        lines.append(ind + "}")

    emit(root, 0, True)
    # MOTION: 1 rest frame; root 6 ch + 3*(J-1) joint ch, all zeros
    n_ch = 6 + 3 * (J - 1)
    lines.append("MOTION")
    lines.append("Frames: 1")
    lines.append("Frame Time: 0.033333")
    lines.append(" ".join(["0.000000"] * n_ch))
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    return J, n_ch


for sp in species:
    if sp not in c:
        print(f"SKIP {sp}: not in cond.npy"); continue
    p = os.path.join(OUTDIR, f"{sp}_Tpose.bvh")
    J, n_ch = write_bvh(sp, c[sp], p)
    print(f"WROTE {p}  J={J} channels={n_ch}")
