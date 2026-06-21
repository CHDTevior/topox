"""Feasibility probe for 'Option A' (re-encode HumanML3D rotations into AnyTop
rigid-BVH convention so Gate C passes).

The rigid-BVH FK route can place each joint c only at world_rot[parent[c]] @
offset[c] + pos[parent]. Because world_rot[p] is free per joint (local rot is
unconstrained), the BEST achievable per-bone error at parent p is the Kabsch
(orthogonal-Procrustes) residual fitting rest offsets {offset[c]} of p's
children to actual bone vectors {pos[c]-pos[p]}. Joints with <2 children fit
exactly; only branching joints (0 and 9 for the t2m skeleton) carry residual.

If the residual is ~0, a proper rigid-BVH re-encode would make Gate C pass.
If it is large, the human skeleton flexes non-rigidly at the branch and
rigid-BVH cannot represent it -> Gate C has an irreducible floor.

Read-only / diagnostic.
"""
import sys
import numpy as np
import torch

HM_ROOT = "/iridisfs/scratch/ts1v23/workspace/motion-latent-diffusion-main"
DATA = HM_ROOT + "/datasets/humanml3d/HumanML3D"
sys.path.insert(0, HM_ROOT)
from mld.data.humanml.scripts.motion_process import recover_from_ric
from mld.data.humanml.utils.paramUtil import t2m_kinematic_chain

J = 22
parents = np.full(J, -1, dtype=np.int64)
for chain in t2m_kinematic_chain:
    for k in range(1, len(chain)):
        parents[chain[k]] = chain[k - 1]
children = {p: [c for c in range(J) if parents[c] == p] for p in range(J)}
branch = {p: cs for p, cs in children.items() if len(cs) >= 2}
print("branching joints (>=2 children):", {p: cs for p, cs in branch.items()})


def kabsch_R(U, V):
    """best R minimizing ||R U^T - V^T|| ; U,V: [n,3]."""
    H = U.T @ V
    Uu, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ Uu.T))
    D = np.diag([1, 1, d])
    return Vt.T @ D @ Uu.T


# canonical offsets (same as gate probe)
from mld.data.humanml.common.skeleton import Skeleton
from mld.data.humanml.utils.paramUtil import t2m_raw_offsets
example = np.load(DATA + "/new_joints/000021.npy")
tgt = Skeleton(torch.from_numpy(t2m_raw_offsets.astype(np.float32)), t2m_kinematic_chain, "cpu")
offsets = tgt.get_offsets_joints(torch.from_numpy(example[0])).numpy().astype(np.float64)

CLIPS = ["000021", "000000", "000001", "012695", "004822"]
print(f"\n{'clip':>9} | {'rigid-BVH floor: per-bone resid (abs / %bbox)':>46}")
for cid in CLIPS:
    x = np.load(DATA + f"/new_joint_vecs/{cid}.npy")
    P = recover_from_ric(torch.from_numpy(x).float(), J).numpy()   # [T,22,3]
    T = P.shape[0]
    bbox = np.linalg.norm(P.reshape(-1, 3).max(0) - P.reshape(-1, 3).min(0))
    total_resid = []
    per_branch = {p: [] for p in branch}
    for t in range(T):
        for p, cs in branch.items():
            U = np.stack([offsets[c] for c in cs])              # rest bones [n,3]
            V = np.stack([P[t, c] - P[t, p] for c in cs])       # actual bones [n,3]
            R = kabsch_R(U, V)
            resid = np.linalg.norm((R @ U.T).T - V, axis=1)     # [n]
            total_resid.append(resid.mean())
            per_branch[p].append(resid.mean())
    tr = np.array(total_resid)
    bstr = " ".join(f"j{p}:{np.mean(per_branch[p]):.4f}" for p in branch)
    print(f"{cid:>9} | mean {tr.mean():.5f} ({100*tr.mean()/bbox:5.2f}%bbox)  max {tr.max():.4f}  [{bstr}]")

print("\nIf floor ~0 -> Option A (rigid-BVH re-encode) can make Gate C pass.")
print("If floor large -> rigid-BVH cannot represent HumanML3D at branches.")
