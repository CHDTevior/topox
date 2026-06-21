"""Re-encode human rot6d (ch3:9) into AnyTop's per-parent BVH convention so that
our FK route reproduces the RIC positions (FK==RIC ~0, matching animals).

Method (the motion is a rigid retargeted skeleton, proven by _probe_branch_rigidity):
  1. world positions P[T,J,3] from RIC (exact).
  2. canonical offsets[j] = bone (P_ref[t0,j]-P_ref[t0,parent[j]]) from a reference
     frame; fan internal geometry is shared across all clips (uniform_skeleton).
  3. per joint p WITH children, per frame: WR[t,p] = Kabsch({offset[c]} -> {bone[c]}).
  4. local rot_q[i] = WR[parent[i]]^T @ WR[i]  (rot_q[0]=WR[0]).
  5. store token[j] (j>=1) = 6D(rot_q[parent[j]])  -> siblings identical (animal-style);
     keep root token[0] ch3:9 = r_rot (root facing, for root-position recovery).
  6. verify recover_from_bvh_rot_np(new, parents, offsets) ~= P.
"""
import sys
from pathlib import Path
import numpy as np
import torch

HM = "/iridisfs/scratch/ts1v23/workspace/motion-latent-diffusion-main"
REPO = "/iridisfs/scratch/ts1v23/workspace/noKslot_clean"
SRC = HM + "/datasets/humanml3d/HumanML3D"
sys.path.insert(0, HM); sys.path.insert(0, REPO)
from mld.data.humanml.scripts.motion_process import recover_from_ric
from mld.data.humanml.utils.paramUtil import t2m_kinematic_chain
from src.data.anytop_dataset import _recover_world_positions
from src.data.anytop_rot6d_fk import recover_from_bvh_rot_np

J = 22
parents = np.full(J, -1, int)
for ch in t2m_kinematic_chain:
    for k in range(1, len(ch)):
        parents[ch[k]] = ch[k - 1]
children = {p: [j for j in range(J) if parents[j] == p] for p in range(J)}


def kabsch(U, V):
    """best rotation R with R@U[i] ~= V[i]; U,V [n,3]."""
    H = U.T @ V
    Uu, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ Uu.T))
    return Vt.T @ np.diag([1, 1, d]) @ Uu.T


def mat_to_6d(R):
    """[...,3,3] -> [...,6] = first two COLUMNS (matches _rotation_6d_to_matrix_np)."""
    return np.concatenate([R[..., :, 0], R[..., :, 1]], axis=-1)


# canonical offsets from reference clip 000021 frame 0
Pref = recover_from_ric(torch.from_numpy(np.load(SRC + "/new_joint_vecs/000021.npy")).float(), J).numpy()
offsets = np.zeros((J, 3))
for j in range(1, J):
    offsets[j] = Pref[0, j] - Pref[0, parents[j]]


def reencode(raw13):
    """raw13 [T,J,13] (direct-copy) -> raw13 with ch3:9 re-encoded to AnyTop convention."""
    P = _recover_world_positions(raw13.astype(np.float32)).astype(np.float64)
    T = P.shape[0]
    WR = np.tile(np.eye(3), (T, J, 1, 1))
    for p in range(J):
        cs = children[p]
        if not cs:
            continue
        U = np.stack([offsets[c] for c in cs])
        for t in range(T):
            V = np.stack([P[t, c] - P[t, p] for c in cs])
            WR[t, p] = kabsch(U, V)
    rotq = np.tile(np.eye(3), (T, J, 1, 1))
    for i in range(J):
        if not children[i]:
            continue
        gp = parents[i]
        rotq[:, i] = WR[:, i] if gp < 0 else np.matmul(np.transpose(WR[:, gp], (0, 2, 1)), WR[:, i])
    new = raw13.astype(np.float64).copy()
    for j in range(1, J):
        new[:, j, 3:9] = mat_to_6d(rotq[:, parents[j]])
    # root token ch3:9 kept (root facing) for root-position recovery
    return new


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


print(f"{'clip':>8} {'T':>4} | {'OLD FK-vs-RIC':>14} | {'NEW FK-vs-RIC (re-encoded)':>26} | sib-diff")
for cid in ["000021", "000000", "000005", "000006", "012695", "004822", "000001"]:
    x = np.load(SRC + f"/new_joint_vecs/{cid}.npy")
    raw = conv13(x)
    ricP = _recover_world_positions(raw.astype(np.float32))
    old_fk = recover_from_bvh_rot_np(raw, parents, offsets)
    new = reencode(raw)
    new_fk = recover_from_bvh_rot_np(new, parents, offsets)
    eo = np.linalg.norm(old_fk - ricP, axis=-1)
    en = np.linalg.norm(new_fk - ricP, axis=-1)
    bbox = np.linalg.norm(ricP.reshape(-1, 3).max(0) - ricP.reshape(-1, 3).min(0))
    # sibling ch3:9 diff after re-encode (should be IDENTICAL like animals)
    sib = max(float(np.abs(new[:, 1, 3:9] - new[:, 2, 3:9]).max()),
              float(np.abs(new[:, 12, 3:9] - new[:, 13, 3:9]).max()))
    print(f"{cid:>8} {x.shape[0]:>4} | {eo.mean():7.4f}({100*eo.mean()/bbox:4.1f}%) | "
          f"NEW mean={en.mean():.6f} max={en.max():.6f} ({100*en.mean()/bbox:.3f}%) | {sib:.4f}")

print("\nGoal: NEW FK-vs-RIC ~0 (like animals) AND sib-diff ~0 (siblings now share parent rot).")
