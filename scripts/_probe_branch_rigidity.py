"""Is a branching joint's child-bone fan RIGID across frames?

If yes, ONE rotation per frame reproduces all its children exactly -> AnyTop's
per-parent rot6d CAN represent the human motion losslessly (the earlier ~3-4%
Kabsch floor was an artifact of using simplified raw_offset directions as the
rest reference, NOT a true non-rigidity).

Test (per branching joint p, per frame t): fit the best rigid rotation R_t mapping
the FRAME-0 child bone vectors {b[0,c]} to the frame-t vectors {b[t,c]} via Kabsch.
Residual ~0 across all frames == rigid fan == losslessly representable.
Also report bone-length constancy and pairwise-angle constancy.
"""
import sys
from pathlib import Path
import numpy as np
import torch

HM = "/iridisfs/scratch/ts1v23/workspace/motion-latent-diffusion-main"
SRC = HM + "/datasets/humanml3d/HumanML3D"
sys.path.insert(0, HM)
from mld.data.humanml.scripts.motion_process import recover_from_ric
from mld.data.humanml.utils.paramUtil import t2m_kinematic_chain

J = 22
parents = np.full(J, -1, int)
for ch in t2m_kinematic_chain:
    for k in range(1, len(ch)):
        parents[ch[k]] = ch[k - 1]
kids = {}
for j, p in enumerate(parents):
    if p >= 0:
        kids.setdefault(int(p), []).append(j)
branch = {p: cs for p, cs in kids.items() if len(cs) >= 2}


def kabsch(U, V):
    H = U.T @ V
    Uu, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ Uu.T))
    return Vt.T @ np.diag([1, 1, d]) @ Uu.T


for cid in ["000021", "000000", "000005", "000006", "012695"]:
    x = np.load(SRC + f"/new_joint_vecs/{cid}.npy")
    P = recover_from_ric(torch.from_numpy(x).float(), J).numpy()
    T = P.shape[0]
    bbox = np.linalg.norm(P.reshape(-1, 3).max(0) - P.reshape(-1, 3).min(0))
    print(f"\n=== {cid}  T={T}  bbox={bbox:.3f} ===")
    for p, cs in branch.items():
        B = np.stack([P[:, c] - P[:, p] for c in cs], axis=1)   # [T, n, 3] child bones
        # bone length constancy
        L = np.linalg.norm(B, axis=2)                            # [T,n]
        len_var = float((L.std(0) / (L.mean(0) + 1e-9)).max())
        # pairwise-angle constancy (rigidity of the fan shape)
        U = B / (L[..., None] + 1e-9)                            # unit bones
        angs = []
        for a in range(len(cs)):
            for b in range(a + 1, len(cs)):
                angs.append(np.degrees(np.arccos(np.clip((U[:, a] * U[:, b]).sum(1), -1, 1))))
        angs = np.stack(angs, 1)                                 # [T, pairs]
        ang_range = float((angs.max(0) - angs.min(0)).max())
        # rigid fit frame0 -> frame t
        ref = B[0]                                               # [n,3]
        resid = []
        for t in range(T):
            R = kabsch(ref, B[t])
            resid.append(np.linalg.norm((R @ ref.T).T - B[t], axis=1).mean())
        resid = np.array(resid)
        print(f"  joint {p:>2} children {cs}: "
              f"bonelen_var={len_var*100:5.2f}%  pairangle_range={ang_range:6.2f}deg  "
              f"rigid-fit resid mean={resid.mean():.4f} ({100*resid.mean()/bbox:.2f}%bbox) max={resid.max():.4f}")

print("\nrigid-fit resid ~0 + small angle_range => fan is RIGID => AnyTop per-parent")
print("rot6d CAN represent it losslessly (re-encode from positions, not direct copy).")
