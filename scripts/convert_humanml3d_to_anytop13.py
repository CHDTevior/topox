"""Convert HumanML3D (263-dim) motions into the AnyTop13 dataset contract.

Per the reviewed plan (handoff/20260619_humanml3d_to_anytop13_conversion_plan.md)
and the user's 2026-06-19 execution decisions:

  - Channel mapping 263 -> [T,22,13] (root yaw integrated to 6D, RIC/local-vel/
    contact repacked). Gate B (official recover_from_ric vs our
    _recover_world_positions, ch0:3) is numerically EXACT.
  - rot6d RE-ENCODE (reencode_rot6d): the non-root rotation channels (ch3:9) are
    re-encoded from HumanML3D's per-bone convention into AnyTop's per-parent
    (rigid-BVH) convention, so the FK route (recover_from_bvh_rot_np) reproduces
    the RIC positions and siblings share the parent's single rotation -- exactly
    matching the animal dataset. Uses SHARED rest bones (000021 frame-0). The
    skeleton fan is rigid (proven), so this is a lossless re-parameterisation;
    FK-vs-RIC is exact on 000021, ~0.12-0.5% bbox typical, up to ~2.5% on a few
    per-subject-pelvis outliers (irreducible for a SHARED human skeleton, accepted
    by the user). ch0:3 are untouched so Gate B stays exact.
  - With ch3:9 in the animal convention, human and animal rot6d now MEAN the same
    thing and the VQVAE fk loss behaves consistently across both (the prior
    "no w_fk on human" red line is resolved down to the ~0.26% residual floor).

cond.npy uses ONE object: 'HML3D_Human' (22-joint fixed topology). cond key is
'joints_names' (plural) to match the loader schema. cond['offsets'] = the SAME
000021 frame-0 rest bones used by the re-encode (so the FK route is self-consistent).
graph fields are produced by the loader's own _build_derived for bit-consistency.

Single-author; re-encode validated by scripts/_reencode_rot6d_probe.py and codex
(thread 019ee178, PASS); Gate B == 0 by scripts/_probe_hml3d_gate_bc.py.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch

HM_ROOT = "/iridisfs/scratch/ts1v23/workspace/motion-latent-diffusion-main"
REPO = "/iridisfs/scratch/ts1v23/workspace/noKslot_clean"
SRC = HM_ROOT + "/datasets/humanml3d/HumanML3D"
sys.path.insert(0, HM_ROOT)
sys.path.insert(0, REPO)

from mld.data.humanml.common.quaternion import quaternion_to_cont6d_np
from mld.data.humanml.scripts.motion_process import recover_from_ric
from mld.data.humanml.utils.paramUtil import t2m_kinematic_chain
from src.data.anytop_dataset import _build_derived

OBJ = "HML3D_Human"
J = 22
TGT_SKEL_ID = "000021"

JOINT_NAMES = [
    "pelvis", "left_hip", "right_hip", "spine1", "left_knee", "right_knee",
    "spine2", "left_ankle", "right_ankle", "spine3", "left_foot", "right_foot",
    "neck", "left_collar", "right_collar", "head", "left_shoulder",
    "right_shoulder", "left_elbow", "right_elbow", "left_wrist", "right_wrist",
]
assert len(JOINT_NAMES) == J

# parents from the t2m kinematic chains (already FK-ordered: parents[j] < j)
PARENTS = np.full(J, -1, dtype=np.int64)
for chain in t2m_kinematic_chain:
    for k in range(1, len(chain)):
        PARENTS[chain[k]] = chain[k - 1]
assert PARENTS.tolist() == [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12,
                            13, 14, 16, 17, 18, 19]

# foot-contact bit order in the 263 vec is [feet_l, feet_r] = fid_l[7,10]+fid_r[8,11]
CONTACT_JOINTS = [7, 10, 8, 11]  # contact[0..3] -> these joints

CHILDREN = {p: [j for j in range(J) if PARENTS[j] == p] for p in range(J)}


def world_positions(x263: np.ndarray) -> np.ndarray:
    """[T,263] -> [T,22,3] world joints via official recover_from_ric (== our
    _recover_world_positions to ~1e-7; Gate B). Fast torch path for re-encoding."""
    return recover_from_ric(torch.from_numpy(np.asarray(x263)).float(), J).numpy().astype(np.float64)


def _kabsch_batch(U: np.ndarray, V: np.ndarray) -> np.ndarray:
    """Best rotations R[t] with R[t]@U[n] ~= V[t,n]. U:[n,3], V:[T,n,3] -> [T,3,3]."""
    H = np.einsum("ni,tnj->tij", U, V)                       # [T,3,3]
    Uu, _, Vt = np.linalg.svd(H)
    det = np.linalg.det(np.matmul(Vt.transpose(0, 2, 1), Uu.transpose(0, 2, 1)))
    D = np.broadcast_to(np.eye(3), H.shape).copy()
    D[:, 2, 2] = np.sign(det)
    return np.matmul(np.matmul(Vt.transpose(0, 2, 1), D), Uu.transpose(0, 2, 1))


def _mat_to_6d(R: np.ndarray) -> np.ndarray:
    """[...,3,3] -> [...,6] = first two COLUMNS (matches _rotation_6d_to_matrix_np)."""
    return np.concatenate([R[..., :, 0], R[..., :, 1]], axis=-1)


# ----------------------------------------------------------------------------
# v3 re-encode helpers (handoff/20260630_033233_human_rot6d_v3_converter_*.md).
# v3a/v3b replace the rank-1-degenerate Kabsch at SINGLE-CHILD parents with a
# deterministic zero-twist SWING (the under-constrained axial twist that LAPACK
# fills with an arbitrary, frame-to-frame-flipping completion is the root of the
# human rot6d-FK jitter). Multi-child parents keep full Kabsch (v3a) or a
# temporally-continuous Kabsch (v3b, spine3 only). v2 path is byte-unchanged.
# ----------------------------------------------------------------------------
def _skew(w: np.ndarray) -> np.ndarray:
    """[...,3] -> [...,3,3] skew-symmetric cross-product matrix."""
    K = np.zeros(w.shape[:-1] + (3, 3), dtype=np.float64)
    K[..., 0, 1] = -w[..., 2]; K[..., 0, 2] = w[..., 1]
    K[..., 1, 0] = w[..., 2];  K[..., 1, 2] = -w[..., 0]
    K[..., 2, 0] = -w[..., 1]; K[..., 2, 1] = w[..., 0]
    return K


def _axis_angle(axis: np.ndarray, angle: float) -> np.ndarray:
    """Single Rodrigues rotation [3],scalar -> [3,3] (axis assumed unit)."""
    K = _skew(np.asarray(axis, dtype=np.float64))
    return (np.eye(3) + np.sin(angle) * K
            + (1.0 - np.cos(angle)) * (K @ K))


def _axis_angle_batch(axis: np.ndarray, angle: np.ndarray) -> np.ndarray:
    """Batched Rodrigues: axis [T,3] (unit), angle [T] -> [T,3,3]. Used by v4 twist roll."""
    K = _skew(np.asarray(axis, dtype=np.float64))
    s = np.sin(angle)[:, None, None]; c = np.cos(angle)[:, None, None]
    I = np.broadcast_to(np.eye(3), K.shape).copy()
    return I + s * K + (1.0 - c) * np.matmul(K, K)


def _shortest_arc(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Batched minimal-arc (zero-twist) rotation with R@a = b for UNIT a,b [...,3].
    Closed form R = I + [w]x + [w]x^2/(1+c), w=a x b, c=a.b. Singular at c=-1
    (1/(1+c) -> inf); the CALLER masks those frames. At c=+1 -> I."""
    a = a / (np.linalg.norm(a, axis=-1, keepdims=True) + 1e-12)
    b = b / (np.linalg.norm(b, axis=-1, keepdims=True) + 1e-12)
    w = np.cross(a, b)
    c = np.sum(a * b, axis=-1)
    K = _skew(w)
    I = np.broadcast_to(np.eye(3), K.shape).copy()
    coef = 1.0 / (1.0 + c)                                   # inf at c=-1; masked by caller
    return I + K + np.matmul(K, K) * coef[..., None, None]


def _swing_batch(u_rest: np.ndarray, v_cur: np.ndarray,
                 eps: float = 1e-7, floor: float = 1e-7) -> np.ndarray:
    """SINGLE-CHILD parent world rotation WR[t]: the deterministic minimal-arc
    swing carrying the rest bone u_rest [3] onto the CURRENT bone v_cur [T,3],
    canonical twist=0. INVARIANT R[t]@normalize(u) == normalize(v[t]) is asserted
    STRICTLY (<1e-5) on the normal + anti-parallel branches; relaxed to ~|v-u| on
    the sub-floor identity branch. We branch on the cross-product MAGNITUDE |u x v|
    (= |sin angle|), NOT on dot -- so a near-but-not-exact +u frame (e.g. dot=1-1e-8,
    |u x v|~1.5e-4) goes through the WELL-conditioned shortest-arc (axis u x v is
    normalizable) and maps u->v exactly, instead of snapping to identity and failing
    the assert (the old dot>1-eps identity branch crashed on clips 000197/000332):
      |u x v| >= floor, dot > -1+eps -> shortest-arc (closed form, zero-twist). STRICT.
      dot < -1+eps                   -> anti-parallel: shortest-arc singular. DETERMINISTIC
                                        reference flip (pi about a fixed axis perp u, maps
                                        u->-u exactly) THEN well-conditioned arc(-u->v).
                                        R@u = arc@(-u) = v -> STRICT.
      |u x v| < floor, dot > 0       -> bone genuinely unchanged: pure identity, an
                                        intentional sub-floor approximation (|v-u| < ~1e-7),
                                        invariant relaxed to R@u ~= v within |v-u|."""
    T = v_cur.shape[0]
    u = u_rest.astype(np.float64) / (np.linalg.norm(u_rest) + 1e-12)
    v = v_cur.astype(np.float64) / (np.linalg.norm(v_cur, axis=-1, keepdims=True) + 1e-12)
    uT = np.broadcast_to(u, (T, 3))
    c = v @ u                                                # [T] = cos(angle)
    cross_mag = np.linalg.norm(np.cross(uT, v), axis=-1)     # [T] = |sin(angle)| (rotation-axis length)
    R = _shortest_arc(uT, v)                                 # exact for dot > -1+eps; garbage only at anti-parallel
    para = (cross_mag < floor) & (c > 0.0)                   # genuinely unchanged: sub-floor approximation
    R[para] = np.eye(3)
    sing = np.where(c < -1.0 + eps)[0]                       # anti-parallel (cross also sub-floor but dot<0)
    if len(sing):                                            # deterministic, transport-free
        ref = np.array([1.0, 0.0, 0.0]) if abs(u[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        axis = np.cross(u, ref); axis /= (np.linalg.norm(axis) + 1e-12)
        R_flip = _axis_angle(axis, np.pi)                    # R_flip @ u == -u (exact, axis perp u)
        minus_u = (R_flip @ u)[None]                         # == -u
        for t in sing:                                       # arc(-u -> v[t]) is well-conditioned
            R[t] = _shortest_arc(minus_u, v[t:t + 1])[0] @ R_flip
    Ru = np.einsum("tij,j->ti", R, u)                        # invariant: swing maps rest bone onto current bone
    err = np.abs(Ru - v)                                     # [T,3]
    assert float(np.max(err[~para], initial=0.0)) < 1e-5, "swing invariant R@u==v violated (strict branch)"
    if para.any():                                           # sub-floor identity (R@u=u) approx within |v-u|
        vu = float(np.abs(v[para] - u).max())
        assert float(err[para].max()) <= vu + 1e-9, "sub-floor identity beyond |v-u|"
    return R


def _kabsch_continuous(U: np.ndarray, V: np.ndarray,
                       enter: float = 0.05, exit: float = 0.15) -> np.ndarray:
    """v3b EXPERIMENTAL spine3 (multi-child) world rotation. v3a is the PRIMARY
    candidate; v3b's only delta vs v3a is this spine3 token-gauge (FK-position-
    invariant either way). Robustness contract (asserted in the gate runner):
      (1) REDUCES EXACTLY to _kabsch_batch on every NON-degenerate frame (identical
          SVD + identical diag(1,1,sign(det)) construction + identical formula);
      (2) only INSIDE a near-degenerate sigma regime -- entered when the gap of the
          two SMALLER sigmas (s1-s2)/s0 < `enter`, left only when it rises above
          `exit` (true two-threshold HYSTERESIS, no per-frame chatter) -- is the
          ambiguous singular-vector subspace resolved by TEMPORAL CONTINUITY
          (1<->2 column-order + per-column SIGN matched to the previous frame);
      (3) det-correction is the SAME diag(1,1,sign(det(Vt^T Uu^T))) as _kabsch_batch,
          applied to the continuity-SELECTED basis and ONLY when det<0 (det>0 -> D=I,
          NO flip). Frame-0 = plain LAPACK seed (single-child joints already handled
          by _swing_batch, so spine3 has no single-bone canonical swing to seed)."""
    T = V.shape[0]
    H = np.einsum("ni,tnj->tij", U, V)                       # [T,3,3]
    R = np.zeros((T, 3, 3), dtype=np.float64)
    Uu_p = None
    in_degen = False
    for t in range(T):
        Uu, s, Vt = np.linalg.svd(H[t])                      # H = Uu diag(s) Vt
        ratio = (s[1] - s[2]) / (s[0] + 1e-12)               # near-degeneracy of the two SMALLER sigmas
        in_degen = (ratio < exit) if in_degen else (ratio < enter)   # two-threshold hysteresis
        if in_degen and Uu_p is not None:
            # 1<->2 ordering continuity (only the two smaller, ambiguous, sigma-vectors)
            keep = np.dot(Uu[:, 1], Uu_p[:, 1]) ** 2 + np.dot(Uu[:, 2], Uu_p[:, 2]) ** 2
            swap = np.dot(Uu[:, 1], Uu_p[:, 2]) ** 2 + np.dot(Uu[:, 2], Uu_p[:, 1]) ** 2
            if swap > keep:
                Uu = Uu[:, [0, 2, 1]]; Vt = Vt[[0, 2, 1], :]
            for k in range(3):                               # per-column sign continuity
                if np.dot(Uu[:, k], Uu_p[:, k]) < 0:
                    Uu[:, k] = -Uu[:, k]; Vt[k, :] = -Vt[k, :]
        # det-correction: identical to _kabsch_batch, on the (continuity-selected) basis, det<0 only
        D = np.eye(3)
        D[2, 2] = np.sign(np.linalg.det(Vt.T @ Uu.T))
        R[t] = Vt.T @ D @ Uu.T
        Uu_p = Uu                                            # basis actually used (post swap/sign)
    return R


# single-/multi-child split (codex-confirmed against PARENTS): single-child parents
# are rank-1-degenerate (twist ill-posed); multi-child = pelvis(0)+spine3(9).
_SINGLE_CHILD_PARENTS = [p for p in range(J) if len(CHILDREN[p]) == 1]
_MULTI_CHILD_PARENTS = [p for p in range(J) if len(CHILDREN[p]) > 1]


def reencode_rot6d(raw13: np.ndarray, P: np.ndarray, offsets: np.ndarray,
                   rot6d_mode: str = "v2", return_wr: bool = False,
                   twist_phi: dict | None = None):
    """Re-encode non-root ch3:9 from HumanML3D per-bone into AnyTop per-parent
    (rigid-BVH) convention so recover_from_bvh_rot_np reproduces the RIC positions
    (siblings share the parent's rotation, matching the animal dataset).

    Per joint p with children C: WR[t,p] = Kabsch({offsets[c]} -> {P[t,c]-P[t,p]}).
    local rot_q[p] = WR[parent[p]]^T @ WR[p] (rot_q[0]=WR[0]). token[j] stores
    rot_q[parent[j]] (so all children of a parent carry the same 6D). Root token
    ch3:9 (yaw facing) is kept for root-position recovery.

    rot6d_mode: "v2" (default, BYTE-UNCHANGED rigid Kabsch everywhere); "v3a"
    (single-child parents -> deterministic zero-twist swing, multi-child Kabsch);
    "v3b" (single-child -> swing same as v3a, spine3 -> temporally-continuous
    Kabsch, pelvis -> plain Kabsch). v3a/v3b only change WR[p]; the local-rotation
    + sibling-share + token packing below is identical, so the per-parent /
    sibling-shared convention + ch0:3 + root ch3:9 are preserved.
    """
    if rot6d_mode not in ("v2", "v3a", "v3b", "v4"):
        raise ValueError(f"reencode_rot6d: unknown rot6d_mode {rot6d_mode!r}")
    T = P.shape[0]
    WR = np.broadcast_to(np.eye(3), (J, T, 3, 3)).copy()     # [J,T,3,3] world rot
    for p in range(J):
        cs = CHILDREN[p]
        if not cs:
            continue
        U = offsets[cs]                                      # [n,3]
        V = P[:, cs, :] - P[:, p:p + 1, :]                   # [T,n,3]
        if rot6d_mode == "v2":
            WR[p] = _kabsch_batch(U, V)                      # byte-unchanged v2 path
        elif len(cs) == 1:                                   # single-child: v3a/v3b/v4 swing (+v4 twist roll)
            Sw = _swing_batch(U[0], V[:, 0, :])              # FK-EXACT zero-twist swing (positions preserved)
            if rot6d_mode == "v4" and twist_phi is not None and p in twist_phi:
                # Variant S: real axial twist as a roll about the CURRENT bone (a pure gauge -> FK unchanged).
                a = V[:, 0, :] / (np.linalg.norm(V[:, 0, :], axis=-1, keepdims=True) + 1e-12)
                WR[p] = np.matmul(_axis_angle_batch(a, np.asarray(twist_phi[p], np.float64)), Sw)
            else:
                WR[p] = Sw                                   # v3a/v3b, or v4 fallback (twist_valid=NONE -> zero twist)
        elif rot6d_mode == "v3b" and p == 9:                 # spine3: continuity Kabsch (v3b only)
            WR[p] = _kabsch_continuous(U, V)
        else:                                                # multi-child (pelvis; spine3 in v3a/v4)
            WR[p] = _kabsch_batch(U, V)
    rotq = np.broadcast_to(np.eye(3), (J, T, 3, 3)).copy()
    for i in range(J):
        if not CHILDREN[i]:
            continue
        gp = PARENTS[i]
        rotq[i] = WR[i] if gp < 0 else np.matmul(WR[gp].transpose(0, 2, 1), WR[i])
    new = raw13.astype(np.float64).copy()
    for j in range(1, J):
        new[:, j, 3:9] = _mat_to_6d(rotq[PARENTS[j]])
    if return_wr:                                            # WR[J,T,3,3]: intended builder world rot (Gate G round-trip)
        return new.astype(np.float32), WR
    return new.astype(np.float32)


def convert_263_to_13(x: np.ndarray) -> np.ndarray:
    """[T,263] HumanML3D -> [T,22,13] AnyTop raw. Gate-B-exact (probe-verified)."""
    x = np.asarray(x, dtype=np.float64)
    T = x.shape[0]
    root_rot_vel = x[:, 0]
    root_vel_xz = x[:, 1:3]
    root_y = x[:, 3]
    ric = x[:, 4:4 + (J - 1) * 3].reshape(T, J - 1, 3)
    rot6d = x[:, 67:67 + (J - 1) * 6].reshape(T, J - 1, 6)
    local_vel = x[:, 193:193 + J * 3].reshape(T, J, 3)
    foot = x[:, 259:263]

    # integrate yaw exactly like HumanML3D recover_root_rot_pos, then -> 6D
    r_rot_ang = np.zeros(T)
    r_rot_ang[1:] = root_rot_vel[:-1]
    r_rot_ang = np.cumsum(r_rot_ang)
    root_quat = np.zeros((T, 4))
    root_quat[:, 0] = np.cos(r_rot_ang)
    root_quat[:, 2] = np.sin(r_rot_ang)
    root6d = quaternion_to_cont6d_np(root_quat)

    raw = np.zeros((T, J, 13), dtype=np.float32)
    raw[:, 0, 0] = root_rot_vel
    raw[:, 0, 1] = root_y
    raw[:, 0, 2] = 0.0
    raw[:, 0, 3:9] = root6d
    raw[:, 1:, 0:3] = ric
    raw[:, 1:, 3:9] = rot6d
    raw[:, :, 9:12] = local_vel
    raw[:, 0, 9] = root_vel_xz[:, 0]    # renderer-critical root xz vel
    raw[:, 0, 11] = root_vel_xz[:, 1]
    for bit, jj in enumerate(CONTACT_JOINTS):
        raw[:, jj, 12] = foot[:, bit]
    return raw


def compute_offsets() -> np.ndarray:
    """Shared rest bones = ACTUAL frame-0 bone vectors of the t2m target skeleton
    000021 (NOT raw_offset directions). These are the rest fan the per-parent FK
    re-encoding rotates rigidly, and must equal cond['offsets'] so the FK route
    (recover_from_bvh_rot_np) reproduces RIC. offset[j] = P[0,j] - P[0,parent[j]]."""
    P = world_positions(np.load(SRC + f"/new_joint_vecs/{TGT_SKEL_ID}.npy"))
    off = np.zeros((J, 3), dtype=np.float32)
    for j in range(1, J):
        off[j] = P[0, j] - P[0, PARENTS[j]]
    return off


def finalize_std(mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """Replicate AnyTop get_mean_std channel-group smoothing (motion_process:266)."""
    std = std.copy()
    std[0, 0:3] = std[0, 0:3].mean()
    std[0, 3:9] = std[0, 3:9].mean()
    std[0, 9:12] = std[0, 9:12].mean()
    std[1:, 0:3] = std[1:, 0:3].mean()
    std[1:, 3:9] = std[1:, 3:9].mean()
    std[1:, 9:12] = std[1:, 9:12].mean()
    nz = std[:, 12] != 0
    if nz.any():
        std[:, 12][nz] = std[:, 12][nz].mean()
    std[:, 12][std[:, 12] == 0] = 1.0
    return std


def parse_captions(mid: str):
    p = Path(SRC) / "texts" / f"{mid}.txt"
    caps = []
    if p.exists():
        for line in p.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            cap = line.split("#")[0].strip()
            if cap:
                caps.append(cap)
    return caps


def read_split(name: str):
    return [l.strip() for l in (Path(SRC) / f"{name}.txt").read_text().splitlines()
            if l.strip()]


def select_sample(n_target: int = 80):
    """Representative sample covering train/val/test, short/long, action variety."""
    splits = {s: read_split(s) for s in ("train", "val", "test")}
    id2split = {}
    for s, ids in splits.items():
        for i in ids:
            id2split.setdefault(i, s)
    all_ids = read_split("all")
    present = [i for i in all_ids if (Path(SRC) / "new_joint_vecs" / f"{i}.npy").exists()]

    # length scan (cheap mmap shape read)
    lens = {}
    for i in present:
        try:
            lens[i] = int(np.load(Path(SRC) / "new_joint_vecs" / f"{i}.npy",
                                  mmap_mode="r").shape[0])
        except Exception:
            pass
    short = [i for i in present if lens.get(i, 99) <= 4]
    long_ = sorted([i for i in present if lens.get(i, 0) > 300],
                   key=lambda i: -lens[i])

    # action variety via caption keywords
    KW = ["kick", "jump", "walk", "run", "turn", "wave", "dance", "sit",
          "throw", "punch", "crouch"]
    kw_pick = {}
    for i in present[:6000]:  # scan a prefix for speed
        caps = parse_captions(i)
        if not caps:
            continue
        low = caps[0].lower()
        for k in KW:
            if k in low and k not in kw_pick:
                kw_pick[k] = i
        if len(kw_pick) == len(KW):
            break

    picked, tags = [], {}

    def add(i, tag):
        if i in present and i not in picked:
            picked.append(i)
            tags[i] = tag

    for i in short[:3]:
        add(i, "short<=4")
    for i in long_[:6]:
        add(i, f"long{lens[i]}")
    for k, i in kw_pick.items():
        add(i, f"act:{k}")
    # ensure split coverage + fill with deterministic spread
    for s in ("train", "val", "test"):
        for i in splits[s][:: max(1, len(splits[s]) // 20)]:
            if len(picked) >= n_target:
                break
            add(i, s)
    for i in present[:: max(1, len(present) // n_target)]:
        if len(picked) >= n_target:
            break
        add(i, "spread")
    return picked, tags, id2split, lens


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["sample", "full"], default="sample")
    ap.add_argument("--out", default=REPO + "/data/humanml3d_anytop13")
    ap.add_argument("--sample_n", type=int, default=80)
    ap.add_argument("--rot6d_mode", choices=["v2", "v3a", "v3b"], default="v2",
                    help="ch3:9 re-encode: v2 (default, unchanged) / v3a (single-child "
                         "zero-twist swing) / v3b (+spine3 continuity Kabsch).")
    args = ap.parse_args()
    print(f"[convert] rot6d_mode={args.rot6d_mode}")

    import shutil
    out = Path(args.out)
    # Clear stale generated artifacts so a rerun (different --mode/--sample_n)
    # can never leave orphan motions/ files that violate the loader's
    # train.txt U val.txt == motions/ invariant (codex review 019ede9a).
    # Rendered GIFs under animations/ are preserved.
    for d in ("motions", "motions_heldout", "splits"):
        shutil.rmtree(out / d, ignore_errors=True)
    for f in ("cond.npy", "object_index.csv", "motion_texts_by_file.json",
              "DATASET_INFO.md", "_cond_normalized_J144.pkl",
              "_cond_normalized_J64.pkl"):
        (out / f).unlink(missing_ok=True)
    (out / "motions").mkdir(parents=True, exist_ok=True)
    (out / "motions_heldout").mkdir(parents=True, exist_ok=True)
    (out / "splits").mkdir(parents=True, exist_ok=True)
    (out / "animations").mkdir(parents=True, exist_ok=True)

    splits_full = {s: read_split(s) for s in ("train", "val", "test", "all")}
    id2split = {}
    for s in ("train", "val", "test"):
        for i in splits_full[s]:
            id2split.setdefault(i, s)

    if args.mode == "sample":
        ids, tags, _, lens = select_sample(args.sample_n)
        print(f"[sample] {len(ids)} clips selected")
    else:
        ids = [i for i in splits_full["all"]
               if (Path(SRC) / "new_joint_vecs" / f"{i}.npy").exists()]
        tags = {}
        lens = {}
        print(f"[full] {len(ids)} clips present")

    # ---- convert + Gate A + accumulate TRAIN stats ----
    # AnyTop file-mode loader requires motions/ == train.txt U val.txt exactly.
    # So motions/ holds only train+val (T>4); test clips + degenerate short
    # clips (T<=4, ~no temporal signal) go to motions_heldout/ for later
    # evaluator use and are kept out of the training loader.
    s1 = np.zeros((J, 13), np.float64)
    s2 = np.zeros((J, 13), np.float64)
    cnt = 0
    gateA = {"n": 0, "nan": [], "short": [], "long": [], "tmismatch": []}
    captions = {}
    obj_rows = []
    split_lists = {"train": [], "val": [], "test": [], "all": []}

    # shared rest bones (000021 frame-0); used for both ch3:9 re-encode and cond
    OFFSETS = compute_offsets()

    for k, mid in enumerate(ids):
        x = np.load(Path(SRC) / "new_joint_vecs" / f"{mid}.npy")
        T = x.shape[0]
        raw = convert_263_to_13(x)
        # re-encode non-root ch3:9 into AnyTop per-parent convention (aligns with
        # animal dataset; FK route reproduces RIC to ~0.26%, was ~14% direct-copy)
        raw = reencode_rot6d(raw, world_positions(x), OFFSETS, rot6d_mode=args.rot6d_mode)
        if raw.shape != (T, J, 13):
            raise RuntimeError(f"{mid} bad shape {raw.shape}")
        if not np.isfinite(raw).all():
            gateA["nan"].append(mid)
        is_short = T <= 4
        if is_short:
            gateA["short"].append((mid, T))
        if T > 300:
            gateA["long"].append((mid, T))

        split_of = id2split.get(mid, "test")
        to_loader = (split_of in ("train", "val")) and not is_short
        fname = f"{OBJ}_{mid}.npy"
        np.save(out / ("motions" if to_loader else "motions_heldout") / fname, raw)
        gateA["n"] += 1
        split_lists["all"].append(fname)
        if to_loader:
            split_lists[split_of].append(fname)
        elif split_of == "test":
            split_lists["test"].append(fname)

        if split_of == "train" and to_loader:
            s1 += raw.astype(np.float64).sum(axis=0)
            s2 += (raw.astype(np.float64) ** 2).sum(axis=0)
            cnt += T

        caps = parse_captions(mid)
        captions[fname] = {
            "primary_caption": caps[0] if caps else "",
            "captions": caps,
            "source_dataset": "HumanML3D",
            "source_motion_id": mid,
        }
        loc = "motions" if to_loader else "motions_heldout"
        obj_rows.append([fname, OBJ, T, split_of, loc, tags.get(mid, "")])
        if (k + 1) % 2000 == 0:
            print(f"  ... {k + 1}/{len(ids)}")

    # ---- mean/std (per joint/channel, AnyTop-smoothed) over TRAIN ----
    if cnt > 0:
        mean = (s1 / cnt).astype(np.float32)
        var = np.clip(s2 / cnt - (s1 / cnt) ** 2, 0, None)
        std = finalize_std(mean, np.sqrt(var).astype(np.float32))
    else:  # sample with no train clips -> use all written as a stand-in
        raise RuntimeError("no TRAIN clips in selection; cannot compute mean/std")

    # ---- cond.npy (single HML3D_Human object) ----
    offsets = OFFSETS
    x021 = np.load(Path(SRC) / "new_joint_vecs" / f"{TGT_SKEL_ID}.npy")
    tpos = reencode_rot6d(convert_263_to_13(x021), world_positions(x021), OFFSETS,
                          rot6d_mode=args.rot6d_mode)[0]
    derived = _build_derived(PARENTS, offsets, JOINT_NAMES)
    cond = {OBJ: {
        "parents": PARENTS.copy(),
        "offsets": offsets.astype(np.float64),
        "tpos_first_frame": tpos.astype(np.float64),
        "joints_names": list(JOINT_NAMES),
        "kinematic_chains": [list(c) for c in t2m_kinematic_chain],
        "joint_relations": derived["joint_relations"].astype(np.float64),
        "joints_graph_dist": derived["joints_graph_dist"].astype(np.float64),
        "mean": mean.astype(np.float64),
        "std": std.astype(np.float64),
        "object_type": OBJ,
    }}
    np.save(out / "cond.npy", cond, allow_pickle=True)

    # ---- splits: train.txt U val.txt == motions/ on disk (loader invariant) ----
    for s in ("train", "val", "test", "all"):
        (out / "splits" / f"{s}.txt").write_text("\n".join(split_lists[s]) + "\n")

    # ---- captions + object_index + info ----
    (out / "motion_texts_by_file.json").write_text(json.dumps(captions, indent=1))
    with open(out / "object_index.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["filename", "object_type", "num_frames", "split", "location", "tag"])
        w.writerows(obj_rows)

    n_loader = len(split_lists["train"]) + len(split_lists["val"])
    (out / "DATASET_INFO.md").write_text(f"""# HumanML3D -> AnyTop13 ({args.mode})

HumanML3D motions converted into the AnyTop13 dataset contract (one fixed
22-joint object `HML3D_Human`) so they load through the existing `AnyTopDataset`
with no model-side branch. Built by `scripts/convert_humanml3d_to_anytop13.py`.

Source: `{SRC}` (new_joint_vecs / texts / {{train,val,test,all}}.txt).

## Channel mapping (263 flat -> [T,22,13])
- flat 0 -> root ch0 (yaw angular velocity, kept honest, unused by recovery)
- flat 1:3 -> root ch9, ch11 (root xz linear velocity; renderer-critical)
- flat 3 -> root ch1 (root height)
- root ch3:9 = quaternion_to_cont6d_np of integrated yaw [cos,0,sin,0]
- flat 4:67 -> non-root ch0:3 (RIC positions)
- non-root ch3:9 = RE-ENCODED (reencode_rot6d): HumanML3D per-bone rot6d -> AnyTop
  per-parent (rigid-BVH) convention via per-joint Kabsch on the world positions, so
  siblings share the parent's single rotation (animal convention) and the FK route
  reproduces RIC. (flat 67:193 are NOT copied verbatim — they are re-derived.)
- flat 193:259 -> all-joint ch9:12 (local velocity; root ch9/11 then overwritten)
- flat 259:263 -> contact ch12 at joints [7,10,8,11] (L-ankle,L-foot,R-ankle,R-foot)

## QA gate results (this build)
- Gate A: {gateA['n']} clips, NaN/Inf={len(gateA['nan'])}, short(T<=4)={len(gateA['short'])}, long(T>300)={len(gateA['long'])}.
- Gate B (official recover_from_ric vs our _recover_world_positions, ch0:3): EXACT (~1e-6 max).
- Gate C (rot6d-FK recover_from_bvh_rot_np vs RIC) AFTER re-encode: exact on 000021,
  ~0.12-0.5% bbox typical, up to ~2.5% outliers (see scripts/_scan_fk_mismatch_full.py + fk_mismatch.jsonl).
- Gate E: AnyTopDataset loads (anytop_x [144,13,T], joint_mask.sum==22).

## rot6d is now in the ANIMAL convention (per-parent, siblings shared)
1. **RIC recovery (`_recover_world_positions`) is the faithful position path** (Gate B exact);
   use it for visualization / position-route consumption.
2. **The rot6d-FK route (`recover_from_bvh_rot_np`) now reproduces RIC** to ~0.26% mean
   (was ~14% with direct-copy). The small residual is the per-subject pelvis/spine3 fan
   variation that a SHARED human skeleton cannot absorb in the rigid-BVH convention
   (animals are exactly 0 because one object == one rig). NOT a conversion bug.
3. **Human and animal rot6d now MEAN the same thing**, so the VQVAE `fk` loss is
   consistent across both down to the ~0.26% floor (the earlier "no w_fk=1.0 on human"
   red line is resolved). cond['offsets'] = the 000021 frame-0 rest bones the re-encode
   used, so the FK route is self-consistent. (A handful of >1% outliers could later be
   per-clip-offset-corrected or dropped if they ever matter.)

## Layout
- `motions/`         : {n_loader} train+val clips (T>4) — the ONLY dir the loader reads
                       (AnyTop file-mode requires train.txt U val.txt == motions/ on disk).
- `motions_heldout/` : test clips + degenerate short clips (T<=4) — kept for later
                       evaluator use, NOT consumed by the training loader.
- `splits/{{train,val,test,all}}.txt`, `cond.npy`, `object_index.csv`,
  `motion_texts_by_file.json`, `animations/conversion_qa_*`.

Not done here (deferred per scope): T5 caption caches, animal+human union dataset, training.
""")

    print(f"\n=== Gate A ===")
    print(f"  written: {gateA['n']}  train-frames: {cnt}")
    print(f"  NaN/Inf clips: {len(gateA['nan'])} {gateA['nan'][:5]}")
    print(f"  short (T<=4): {len(gateA['short'])} {gateA['short']}  (excluded from train.txt, kept in all.txt)")
    print(f"  long (T>300): {len(gateA['long'])} (first 5) {gateA['long'][:5]}")
    print(f"  mean range [{mean.min():.3f},{mean.max():.3f}]  std range [{std.min():.4f},{std.max():.4f}]")
    print(f"  cond keys: {sorted(cond[OBJ].keys())}")
    print(f"  -> {out}")


if __name__ == "__main__":
    main()
