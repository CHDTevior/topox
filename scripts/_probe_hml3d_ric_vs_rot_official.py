"""Decisive probe: does HumanML3D's OWN ric route agree with its OWN rot route?

Compares, on the SAME 263 vector, three recoveries:
  A. official recover_from_ric(263)          -> positions from RIC channels
  B. official recover_from_rot(263, skel)    -> positions from rot6d via HumanML3D's
                                                OWN FK (Skeleton.forward_kinematics_cont6d)
  C. our recover_from_bvh_rot_np(13)         -> positions from rot6d via AnyTop's
                                                rigid-BVH FK (parent reindex)

If A==B (HumanML3D's own two routes agree) but A!=C, the Gate C gap is OUR FK
CONVENTION differing from HumanML3D's, not an inherent HumanML3D inconsistency.
If A!=B too, HumanML3D's ric and rot are inherently inconsistent and we inherited it.
"""
import sys
import numpy as np
import torch

HM = "/iridisfs/scratch/ts1v23/workspace/motion-latent-diffusion-main"
REPO = "/iridisfs/scratch/ts1v23/workspace/noKslot_clean"
SRC = HM + "/datasets/humanml3d/HumanML3D"
sys.path.insert(0, HM); sys.path.insert(0, REPO)

from mld.data.humanml.scripts.motion_process import recover_from_ric, recover_from_rot
from mld.data.humanml.common.skeleton import Skeleton
from mld.data.humanml.utils.paramUtil import t2m_raw_offsets, t2m_kinematic_chain
from src.data.anytop_dataset import _recover_world_positions
from src.data.anytop_rot6d_fk import recover_from_bvh_rot_np

J = 22
parents = np.full(J, -1, np.int64)
for ch in t2m_kinematic_chain:
    for k in range(1, len(ch)):
        parents[ch[k]] = ch[k - 1]

# HumanML3D skeleton with tgt(000021) offsets — used by recover_from_rot
ex = np.load(SRC + "/new_joints/000021.npy")
skel = Skeleton(torch.from_numpy(t2m_raw_offsets.astype(np.float32)), t2m_kinematic_chain, "cpu")
offsets = skel.get_offsets_joints(torch.from_numpy(ex[0]))   # sets skel._offset internally
offsets_np = offsets.numpy().astype(np.float64)


def conv13(x):
    from mld.data.humanml.common.quaternion import quaternion_to_cont6d_np
    x = x.astype(np.float64); T = x.shape[0]
    rrv = x[:, 0]; rxz = x[:, 1:3]; ry = x[:, 3]
    ric = x[:, 4:67].reshape(T, 21, 3); rot6 = x[:, 67:193].reshape(T, 21, 6)
    lv = x[:, 193:259].reshape(T, 22, 3); foot = x[:, 259:263]
    ang = np.zeros(T); ang[1:] = rrv[:-1]; ang = np.cumsum(ang)
    q = np.zeros((T, 4)); q[:, 0] = np.cos(ang); q[:, 2] = np.sin(ang)
    raw = np.zeros((T, 22, 13), np.float32)
    raw[:, 0, 0] = rrv; raw[:, 0, 1] = ry; raw[:, 0, 3:9] = quaternion_to_cont6d_np(q)
    raw[:, 1:, 0:3] = ric; raw[:, 1:, 3:9] = rot6; raw[:, :, 9:12] = lv
    raw[:, 0, 9] = rxz[:, 0]; raw[:, 0, 11] = rxz[:, 1]
    for b, jj in enumerate([7, 10, 8, 11]):
        raw[:, jj, 12] = foot[:, b]
    return raw


def relerr(a, b):
    d = np.linalg.norm(a - b, axis=-1)
    bbox = np.linalg.norm(a.reshape(-1, 3).max(0) - a.reshape(-1, 3).min(0))
    return d.mean(), 100 * d.mean() / bbox


print(f"{'clip':>8} {'T':>4} | {'A:ric':<6} | {'B官方rot-FK vs ric':>22} | {'C我们BVH-FK vs ric':>22}")
for cid in ["000021", "000000", "000005", "000006", "012695", "004822"]:
    x = np.load(SRC + f"/new_joint_vecs/{cid}.npy")
    A = recover_from_ric(torch.from_numpy(x).float(), J).numpy()           # ric
    B = recover_from_rot(torch.from_numpy(x).float(), J, skel).reshape(-1, J, 3).numpy()  # official rot FK
    raw = conv13(x)
    C = recover_from_bvh_rot_np(raw, parents, offsets_np)                  # our BVH FK
    # also confirm our RIC == official ric
    ours_ric = _recover_world_positions(raw.astype(np.float32))
    eAours = np.linalg.norm(A - ours_ric, axis=-1).max()
    mB, pB = relerr(A, B)
    mC, pC = relerr(A, C)
    print(f"{cid:>8} {x.shape[0]:>4} | ourRIC==offRIC max={eAours:.1e} | "
          f"{mB:7.4f} ({pB:5.2f}%) | {mC:7.4f} ({pC:5.2f}%)")

print("\nB = HumanML3D's OWN rot route (forward_kinematics_cont6d) vs its own ric route")
print("C = our AnyTop rigid-BVH FK route vs ric route (= Gate C)")
