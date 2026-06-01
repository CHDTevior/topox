"""AnyTop rot6d FK recovery: recover world-space joint positions from the 13ch
RIFKE encoding via the ROTATION channels (3:9) + bone offsets + parent chain,
i.e. the official `recover_from_bvh_rot_np` path — NOT the position channel
(0:3) RIC path used by `_recover_world_positions`.

This is a self-contained numpy port (official-derived, PATCHED 2026-06-01) of
the AnyTop/SALAD `recover_from_bvh_rot_np` (+ `recover_root_quat_and_pos_np`)
from the user's planetzoo-anytop-pipeline / truebones motion_process.py.

PATCH (2026-06-01): the official source applies a root correction
`rot_q[:,0] = -r_rot_quat * rot_q[:,0]` that DOUBLE-applies the root global
rotation (turn yaw twice). REMOVED here — the ch3:9 root/child tokens already
carry the global facing, and the parent reindex already puts the correct root
orientation in rot_q[:,0]. Verified: WITH the correction FK-vs-RIC absL1=0.65
(global-orient sweep ~1.98); WITHOUT it absL1=0.0000 (FK == RIC exactly) on
clean_L2 Saiga AND the 1070 old-truebones largest-rotation clips. The earlier
"<1% bbox" check that called this FK correct used near-idle clips (root barely
rotates) and missed the bug.

Why a self-contained port: the original lib needs numpy.core.umath_tests
(removed in new numpy) + a heavy BVH/Quaternions/Animation dependency chain that
won't import here. Every operator below re-implements the authoritative source
(Quaternions.__mul__/from_transforms/__neg__/transforms, Animation.positions_
global 4x4 matmul chain, rotation_6d_to_matrix_np) EXCEPT the removed root
correction noted in the PATCH above.

Inputs use the dataset's own per-item `rest_offsets` + `parent_indices`, which
share the same `new_to_old_perm` joint ordering as `anytop_x` (aligned).
"""
import numpy as np


def _rotation_6d_to_matrix_np(c):
    """utils/rotation_conversions.py:536 verbatim. [...,6]->[...,3,3] cols [x,y,z]."""
    x = c[..., 0:3] / np.linalg.norm(c[..., 0:3], axis=-1, keepdims=True)
    z = np.cross(x, c[..., 3:6], axis=-1)
    z = z / np.linalg.norm(z, axis=-1, keepdims=True)
    y = np.cross(z, x, axis=-1)
    return np.concatenate([x[..., None], y[..., None], z[..., None]], axis=-1)


def _quat_from_transforms(ts):
    """Quaternions.from_transforms verbatim (rotation matrix -> wxyz quat)."""
    d0, d1, d2 = ts[..., 0, 0], ts[..., 1, 1], ts[..., 2, 2]
    q0 = np.sqrt(((d0 + d1 + d2 + 1) / 4).clip(0, None))
    q1 = np.sqrt(((d0 - d1 - d2 + 1) / 4).clip(0, None))
    q2 = np.sqrt(((-d0 + d1 - d2 + 1) / 4).clip(0, None))
    q3 = np.sqrt(((-d0 - d1 + d2 + 1) / 4).clip(0, None))
    c0 = (q0 >= q1) & (q0 >= q2) & (q0 >= q3)
    c1 = (q1 >= q0) & (q1 >= q2) & (q1 >= q3)
    c2 = (q2 >= q0) & (q2 >= q1) & (q2 >= q3)
    c3 = (q3 >= q0) & (q3 >= q1) & (q3 >= q2)
    q1[c0] *= np.sign(ts[c0, 2, 1] - ts[c0, 1, 2]); q2[c0] *= np.sign(ts[c0, 0, 2] - ts[c0, 2, 0]); q3[c0] *= np.sign(ts[c0, 1, 0] - ts[c0, 0, 1])
    q0[c1] *= np.sign(ts[c1, 2, 1] - ts[c1, 1, 2]); q2[c1] *= np.sign(ts[c1, 1, 0] + ts[c1, 0, 1]); q3[c1] *= np.sign(ts[c1, 0, 2] + ts[c1, 2, 0])
    q0[c2] *= np.sign(ts[c2, 0, 2] - ts[c2, 2, 0]); q1[c2] *= np.sign(ts[c2, 1, 0] + ts[c2, 0, 1]); q3[c2] *= np.sign(ts[c2, 2, 1] + ts[c2, 1, 2])
    q0[c3] *= np.sign(ts[c3, 1, 0] - ts[c3, 0, 1]); q1[c3] *= np.sign(ts[c3, 2, 0] + ts[c3, 0, 2]); q2[c3] *= np.sign(ts[c3, 2, 1] + ts[c3, 1, 2])
    return np.stack([q0, q1, q2, q3], axis=-1)


def _quat_mul(s, o):
    """Quaternions.__mul__ (q*q) verbatim."""
    s, o = np.broadcast_arrays(s, o)
    q0, q1, q2, q3 = s[..., 0], s[..., 1], s[..., 2], s[..., 3]
    r0, r1, r2, r3 = o[..., 0], o[..., 1], o[..., 2], o[..., 3]
    return np.stack([
        r0 * q0 - r1 * q1 - r2 * q2 - r3 * q3,
        r0 * q1 + r1 * q0 - r2 * q3 + r3 * q2,
        r0 * q2 + r1 * q3 + r2 * q0 - r3 * q1,
        r0 * q3 - r1 * q2 + r2 * q1 + r3 * q0], axis=-1)


def _quat_neg(q):
    """Quaternions.__neg__ : conjugate."""
    return q * np.array([1.0, -1.0, -1.0, -1.0])


def _quat_mul_vec(q, v):
    """Quaternions.__mul__ (q*vec) verbatim: (q*(vs*-q)).imaginaries, vs=[0,v]."""
    vs = np.concatenate([np.zeros(v.shape[:-1] + (1,)), v], axis=-1)
    return _quat_mul(q, _quat_mul(vs, _quat_neg(q)))[..., 1:]


def _quat_to_matrix(q):
    """Quaternions.transforms() verbatim (wxyz -> 3x3)."""
    qw, qx, qy, qz = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    x2, y2, z2 = qx + qx, qy + qy, qz + qz
    xx, yy, wx, xy, yz = qx * x2, qy * y2, qw * x2, qx * y2, qy * z2
    wy, xz, zz, wz = qw * y2, qx * z2, qz * z2, qw * z2
    m = np.empty(q.shape[:-1] + (3, 3))
    m[..., 0, 0] = 1 - (yy + zz); m[..., 0, 1] = xy - wz; m[..., 0, 2] = xz + wy
    m[..., 1, 0] = xy + wz; m[..., 1, 1] = 1 - (xx + zz); m[..., 1, 2] = yz - wx
    m[..., 2, 0] = xz - wy; m[..., 2, 1] = yz + wx; m[..., 2, 2] = 1 - (xx + yy)
    return m


def _positions_global(rot_q, positions, parents):
    """Animation.positions_global verbatim: 4x4 local->global matmul chain.
    rot_q [F,J,4]; positions [F,J,3]; parents [J]. Returns global xyz [F,J,3]."""
    F, J = rot_q.shape[:2]
    R = _quat_to_matrix(rot_q)
    loc = np.zeros((F, J, 4, 4))
    loc[:, :, :3, :3] = R
    loc[:, :, :3, 3] = positions
    loc[:, :, 3, 3] = 1.0
    glob = np.zeros((F, J, 4, 4))
    glob[:, 0] = loc[:, 0]
    for i in range(1, J):
        glob[:, i] = np.matmul(glob[:, int(parents[i])], loc[:, i])
    p = glob[:, :, :, 3]
    return p[:, :, :3] / p[:, :, 3, None]


def _recover_root_quat_and_pos_np(data):
    """motion_process.py:700 verbatim. data [T,13] (root joint). Returns
    (r_rot_quat [T,4], r_pos [T,3]). Shared by both ric and rot paths."""
    r_rot_quat = _quat_from_transforms(_rotation_6d_to_matrix_np(data[:, 3:9]))
    r_pos = np.zeros(data.shape[:-1] + (3,))
    r_pos[..., 1:, [0, 2]] = data[..., :-1, [9, 11]]
    r_pos = _quat_mul_vec(_quat_neg(r_rot_quat), r_pos)
    r_pos = np.cumsum(r_pos, axis=-2)
    r_pos[..., 1] = data[..., 1]
    return r_rot_quat, r_pos


def recover_from_bvh_rot_np(data, parents, offsets):
    """Official `recover_from_bvh_rot_np` (motion_process.py:750) verbatim port.

    Recover world joint positions from the ROTATION channels (3:9) via FK on the
    parent chain with bone `offsets`, plus the integrated root translation.

    Args:
      data:    [T, J, 13] RAW (un-normalized) AnyTop motion (FK-ordered joints).
      parents: [J] int, FK order (parents[0] = -1, parents[j] < j).
      offsets: [J, 3] rest-pose bone offsets, SAME joint ordering as `data`.
    Returns:
      [T, J, 3] world-space joint positions.
    """
    data = np.asarray(data, dtype=np.float64)
    parents = np.asarray(parents, dtype=int)
    offsets = np.asarray(offsets, dtype=np.float64)
    T, J, _ = data.shape
    r_rot_quat, r_pos = _recover_root_quat_and_pos_np(data[:, 0])      # [T,4],[T,3]
    r_rot_mat = _quat_to_matrix(r_rot_quat)                            # [T,3,3]
    nonroot_mat = _rotation_6d_to_matrix_np(data[:, 1:, 3:9])          # [T,J-1,3,3]
    all_mat = np.concatenate([r_rot_mat[:, None], nonroot_mat], axis=1)  # [T,J,3,3]
    all_q_hml = _quat_from_transforms(all_mat)                         # [T,J,4]
    # parent reindex (motion_process.py:758-759): rotations[:,p] = hml[:,j]
    rot_q = np.zeros((T, J, 4)); rot_q[..., 0] = 1.0
    for j, p in enumerate(parents[1:], 1):
        rot_q[:, p] = all_q_hml[:, j]
    # NOTE: the official recover_from_bvh_rot_np applies a root correction here
    # (rot_q[:,0] = -r_rot_quat * rot_q[:,0]). REMOVED 2026-06-01 — it
    # DOUBLE-applies the root global rotation. Proof (RIC path = ground truth):
    # WITH it,   FK-vs-RIC absL1=0.6522 (global-orient sweep ratio ~1.98);
    # WITHOUT it, FK-vs-RIC absL1=0.0000 (FK == RIC exactly).
    # Verified on clean_L2 Saiga AND the 1070 old-truebones largest-rotation
    # clips (Parrot/Bird CircleFly 720/714 deg, Trex turn_180 396 deg, ...): all
    # absL1=0.0000 once removed. rot_q[:,0] keeps its reindexed value (the root
    # child's rotation from the parent reindex), which already carries the
    # correct root orientation — the correction was a spurious second apply.
    pos = np.repeat(offsets[None], T, axis=0).astype(np.float64)       # [T,J,3]
    pos[:, 0] = r_pos
    return _positions_global(rot_q, pos, parents).astype(np.float32)
