"""Minimal empirical probe for HumanML3D->AnyTop13 conversion Gate B / Gate C.

Per user directive (2026-06-19): Gate C (FK==RIC) is a BLOCKING hard gate, and
must be front-loaded on a few clips to expose any offsets / FK-convention
mismatch BEFORE building the full converter. This script does NOT write any
dataset; it only converts a handful of clips in-memory and measures:

  Gate B: official HumanML3D recover_from_ric(263)  vs  our _recover_world_positions(13ch)
  Gate C: our recover_from_bvh_rot_np(13ch, parents, offsets)  vs  our _recover_world_positions(13ch)

Read-only against source data. Diagnostic only.
"""
import sys
import numpy as np
import torch

HM_ROOT = "/iridisfs/scratch/ts1v23/workspace/motion-latent-diffusion-main"
REPO = "/iridisfs/scratch/ts1v23/workspace/noKslot_clean"
DATA = HM_ROOT + "/datasets/humanml3d/HumanML3D"
sys.path.insert(0, HM_ROOT)
sys.path.insert(0, REPO)

from mld.data.humanml.scripts.motion_process import recover_from_ric          # official RIC
from mld.data.humanml.common.skeleton import Skeleton
from mld.data.humanml.common.quaternion import quaternion_to_cont6d_np
from mld.data.humanml.utils.paramUtil import t2m_raw_offsets, t2m_kinematic_chain

from src.data.anytop_dataset import _recover_world_positions                   # our RIC-route renderer
from src.data.anytop_rot6d_fk import recover_from_bvh_rot_np                    # our FK-route

J = 22

# parents from kinematic chains
parents = np.full(J, -2, dtype=np.int64)
parents[0] = -1
for chain in t2m_kinematic_chain:
    for k in range(1, len(chain)):
        parents[chain[k]] = chain[k - 1]
assert (parents >= -1).all(), parents
EXPECT = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19]
assert parents.tolist() == EXPECT, (parents.tolist(), EXPECT)
print("parents OK:", parents.tolist())

# canonical offsets from tgt skeleton 000021 first frame (HumanML3D standard)
example = np.load(DATA + "/new_joints/000021.npy")          # [T,22,3]
tgt_skel = Skeleton(torch.from_numpy(t2m_raw_offsets.astype(np.float32)), t2m_kinematic_chain, "cpu")
offsets = tgt_skel.get_offsets_joints(torch.from_numpy(example[0])).numpy().astype(np.float64)  # [22,3]
print("offsets shape", offsets.shape, "bone-len range",
      round(float(np.linalg.norm(offsets[1:], axis=1).min()), 4),
      round(float(np.linalg.norm(offsets[1:], axis=1).max()), 4))


def convert_263_to_13(x):
    """[T,263] HumanML3D -> [T,22,13] AnyTop raw, per conversion plan (direct-copy rot)."""
    x = x.astype(np.float64)
    T = x.shape[0]
    root_rot_vel = x[:, 0]
    root_vel_xz = x[:, 1:3]
    root_y = x[:, 3]
    ric = x[:, 4:4 + (J - 1) * 3].reshape(T, J - 1, 3)
    rot6d = x[:, 67:67 + (J - 1) * 6].reshape(T, J - 1, 6)
    local_vel = x[:, 193:193 + J * 3].reshape(T, J, 3)
    foot = x[:, 259:263]

    # integrate yaw exactly like HumanML3D recover_root_rot_pos
    r_rot_ang = np.zeros(T)
    r_rot_ang[1:] = root_rot_vel[:-1]
    r_rot_ang = np.cumsum(r_rot_ang)
    root_quat = np.zeros((T, 4))
    root_quat[:, 0] = np.cos(r_rot_ang)
    root_quat[:, 2] = np.sin(r_rot_ang)
    root6d = quaternion_to_cont6d_np(root_quat)                 # [T,6]

    raw = np.zeros((T, J, 13), dtype=np.float64)
    raw[:, 0, 0] = root_rot_vel
    raw[:, 0, 1] = root_y
    raw[:, 0, 2] = 0.0
    raw[:, 0, 3:9] = root6d
    raw[:, 1:, 0:3] = ric
    raw[:, 1:, 3:9] = rot6d
    raw[:, :, 9:12] = local_vel
    raw[:, 0, 9] = root_vel_xz[:, 0]
    raw[:, 0, 11] = root_vel_xz[:, 1]
    raw[:, 7, 12] = foot[:, 0]
    raw[:, 10, 12] = foot[:, 1]
    raw[:, 8, 12] = foot[:, 2]
    raw[:, 11, 12] = foot[:, 3]
    return raw


def err(a, b):
    d = np.linalg.norm(a - b, axis=-1)   # [T,J] per-joint L2
    bbox = np.linalg.norm(a.reshape(-1, 3).max(0) - a.reshape(-1, 3).min(0))
    return float(d.mean()), float(d.max()), float(d.mean() / (bbox + 1e-8))


CLIPS = ["000021", "000000", "000001", "012695", "004822", "M000000"]
print(f"\n{'clip':>9} {'T':>5} | {'B.mean':>9} {'B.max':>9} | {'C.mean':>9} {'C.max':>9} {'C.rel%':>8}")
for cid in CLIPS:
    p = DATA + f"/new_joint_vecs/{cid}.npy"
    try:
        x = np.load(p)
    except FileNotFoundError:
        print(f"{cid:>9}  (missing)")
        continue
    T = x.shape[0]
    raw = convert_263_to_13(x)

    # Gate B
    ric_off = recover_from_ric(torch.from_numpy(x).float(), J).numpy()   # [T,22,3]
    wpos = _recover_world_positions(raw.astype(np.float32))              # [T,22,3]
    bmean, bmax, _ = err(ric_off, wpos)

    # Gate C
    fk = recover_from_bvh_rot_np(raw, parents, offsets)                  # [T,22,3]
    cmean, cmax, crel = err(fk, wpos)

    print(f"{cid:>9} {T:>5} | {bmean:>9.5f} {bmax:>9.5f} | {cmean:>9.5f} {cmax:>9.5f} {100*crel:>7.2f}")

print("\nGate B target: ~1e-4 (near numeric tol). Gate C target: ~1e-3 (HARD).")
