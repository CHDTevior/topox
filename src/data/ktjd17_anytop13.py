"""KTJD-17 <-> AnyTop-13 analytic conversion inside the 17-slot container (ablation item 2a, user GO 2026-09-07:
"新17转13的代码…有时间就一定要做（训完之后评估器当然还是用我们现在的）").

The OLD representation is this repo's AnyTop 13-channel format (external/planetzoo-anytop-pipeline/docs/
ANYTOP13_INTERMEDIATE_REPRESENTATION.md; encoder motion_process.py get_rifke / get_bvh_cont6d_params, decoder
src/data/anytop_rot6d_fk.py recover_from_bvh_rot_np), derived here from the frozen KTJD-17 clips instead of any old export:

  slot   non-root joint j                                        root (joint 0)
  0:3    RIC position  R_face[t] (P[t,j] - [Px_root, 0, Pz_root])   [0, root height, 0]
  3:9    6D of the LOCAL rotation of parent[j] (child slot)       6D of the facing rotation R_face[t] (world -> +Z)
  9:12   local velocity R_face[t+1] (P[t+1] - P[t]) * fps         [root vx, 0, root vz], same rule
  12     contact (KTJD per-joint contact, copied)                  copied
  13:17  exact zero (no root-track / heading channels; excluded as constants by the stats builder)

REPRESENTATION_ID names THIS container, not the native AnyTop file format: velocities are stored per SECOND (AnyTop stores
per-frame increments; the per-cell statistics builder floors small standard deviations at 0.05 in absolute units and a 30x
smaller velocity scale would have pushed most velocity cells under that floor). A native AnyTop decoder must divide slots
9:12 by fps first. The 6D encoding is the first two columns, as in both codecs.

Facts that make the conversion exact (checked against the two codecs): AnyTop's "world rotation of joint p relative to its
rest fan" (the Kabsch WR[p] of the human converter) is exactly KTJD-17's rest delta (global = delta @ R_rest_global), so the
per-parent local rotation is delta[gp]^T delta[p] and the root children's slot carries delta[0]; R_face = Ry(-atan2(fwd_x, fwd_z))
with the forward vector taken from the KTJD heading channel (cos, sin) = (fwd_z, fwd_x)/|h|; velocities use the NEXT frame's
facing (animal-native convention, motion_process.py:500), which makes the shift-by-one root recovery exact. The last frame
repeats the previous velocity (the release looked past the clip cut; recovery never reads it). Leaf joints have no slot for
their own rotation in the 13 format (inherent): the inverse fills a leaf's rest delta with its parent's.

INVERSE (evaluation of a 13-format generator with the frozen KTJD evaluator): root XZ track = shift-by-one integration of the
facing-un-rotated root velocities from the origin; height read from the root row; positions from RIC; rest deltas rebuilt down
the tree from the LAST child's slot (the native FK's last-child-wins rule); KTJD channels then use the corpus's own split, which
for the frozen PZ release is q_position = P - [root_x, 0, root_z] and slots 13:15 = root_xz - root_xz[0] (the release stores the
UN-smoothed root track; verified on the corpus, root q_position xz == 0 to float32 precision); world velocities = the predicted
local velocities un-rotated by the next frame's facing (model outputs, like the control arm's; the root's vertical velocity,
which has no slot in the 13 format, is the finite difference of the height track); heading from R_face^T z-hat.
Degenerate generated 6D cells: a degenerate facing holds the previous frame's facing (identity at t = 0) and marks the heading
invalid; a degenerate child slot means an identity local rotation for that joint and frame. Both are counted and returned.
An all-invalid heading input round-trips to valid headings (the facing is then the identity) -- declared information loss.
"""
from __future__ import annotations

import numpy as np

from src.data.ktjd17.codec import decode_column_cont6d, direct_decode_positions, encode_column_cont6d

REPRESENTATION_ID = "anytop13_vps_in_ktjd17_container_v1"      # vps: velocities per second (see the module docstring)
D6_EPS = 1e-6                                                   # Gram-Schmidt degeneracy threshold (the decoder's GT eps)


def _ry(phi: np.ndarray) -> np.ndarray:
    """[T] angles -> [T,3,3] right-handed rotations about +Y (x' = x cos + z sin, z' = -x sin + z cos)."""
    c, s = np.cos(phi), np.sin(phi)
    R = np.zeros(phi.shape + (3, 3), dtype=np.float64)
    R[..., 0, 0] = c; R[..., 0, 2] = s; R[..., 1, 1] = 1.0; R[..., 2, 0] = -s; R[..., 2, 2] = c
    return R


def validate_parents(parents, J: int) -> np.ndarray:
    par = np.asarray(parents, dtype=np.int64)
    if par.shape != (J,) or par[0] != -1 or (J > 1 and not (np.all(par[1:] >= 0) and np.all(par[1:] < np.arange(1, J)))):
        raise ValueError(f"parents must be [J={J}] in FK order (parents[0] = -1, 0 <= parents[j] < j), got shape {par.shape}")
    return par


def facing_from_heading(heading: np.ndarray, heading_valid: np.ndarray) -> np.ndarray:
    """KTJD heading [T,2] = (fwd_z, fwd_x)/|h| -> AnyTop facing R_face [T,3,3] (world -> canonical +Z).
    Invalid frames hold the nearest earlier valid heading (leading invalid frames take the first valid one); all-invalid -> identity."""
    h = np.asarray(heading, dtype=np.float64); valid = np.asarray(heading_valid, dtype=bool)
    T = h.shape[0]
    if not valid.any():
        return np.tile(np.eye(3), (T, 1, 1))
    theta = np.arctan2(h[:, 1], h[:, 0])                       # forward = (sin theta, cos theta) in (x, z)
    idx = np.maximum.accumulate(np.where(valid, np.arange(T), -1))
    idx[idx < 0] = int(np.argmax(valid))
    return _ry(-theta[idx])


def children_of(parents: np.ndarray) -> list[list[int]]:
    ch = [[] for _ in range(len(parents))]
    for j in range(1, len(parents)):
        ch[int(parents[j])].append(j)
    return ch


def safe_decode6d(d6: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Gram-Schmidt decode of generated 6D cells: returns (R [...,3,3], degenerate [...] bool). Degenerate cells (either
    Gram-Schmidt norm below D6_EPS) come back as the IDENTITY so a caller can apply its declared fallback."""
    d6 = np.asarray(d6, dtype=np.float64)
    a1, a2 = d6[..., :3], d6[..., 3:]
    n1 = np.linalg.norm(a1, axis=-1)
    b1 = a1 / np.maximum(n1[..., None], 1e-12)
    u2 = a2 - np.sum(b1 * a2, axis=-1, keepdims=True) * b1
    n2 = np.linalg.norm(u2, axis=-1)
    bad = (n1 < D6_EPS) | (n2 < D6_EPS)
    R = np.tile(np.eye(3), d6.shape[:-1] + (1, 1))
    if (~bad).any():
        R[~bad] = decode_column_cont6d(d6[~bad], strict=True)
    return R, bad


def ktjd17_to_anytop13(motion17: np.ndarray, heading_valid: np.ndarray, parents: np.ndarray, *, fps: float) -> np.ndarray:
    """Raw KTJD-17 [T,J,17] (float, clip frame) -> raw AnyTop-13 in the 17-slot container [T,J,17] float64."""
    x = np.asarray(motion17, dtype=np.float64)
    if x.ndim != 3 or x.shape[-1] != 17 or x.shape[0] < 1:
        raise ValueError(f"motion must be [T,J,17], got {x.shape}")
    T, J = x.shape[:2]
    par = validate_parents(parents, J)
    P = direct_decode_positions(x)                                # [T,J,3] world (clip frame)
    delta = decode_column_cont6d(x[..., 3:9])                     # [T,J,3,3] rest deltas (GT: strict)
    R_face = facing_from_heading(x[:, 0, 15:17], heading_valid)   # [T,3,3]
    out = np.zeros((T, J, 17), dtype=np.float64)
    rel = P.copy(); rel[..., 0] -= P[:, 0:1, 0]; rel[..., 2] -= P[:, 0:1, 2]
    out[..., 0:3] = np.einsum("tab,tjb->tja", R_face, rel)
    out[:, 0, 0:3] = 0.0; out[:, 0, 1] = P[:, 0, 1]               # root row: [0, height, 0] (exactly, as in AnyTop)
    if J > 1:
        local = np.empty_like(delta)                              # local[p] = delta[parent(p)]^T delta[p]; local[0] = delta[0]
        local[:, 0] = delta[:, 0]
        local[:, 1:] = np.matmul(np.swapaxes(delta[:, par[1:]], -1, -2), delta[:, 1:])
        out[:, 1:, 3:9] = encode_column_cont6d(local[:, par[1:]]) # child slot j carries the local rotation of parent[j]
    out[:, 0, 3:9] = encode_column_cont6d(R_face)
    vel = np.zeros_like(P)
    if T >= 2:
        vel[:-1] = np.einsum("tab,tjb->tja", R_face[1:], P[1:] - P[:-1]) * float(fps)
        vel[-1] = vel[-2]
    out[..., 9:12] = vel
    out[:, 0, 10] = 0.0                                           # root vertical velocity: unused slot in AnyTop
    out[..., 12] = x[..., 12]
    return out


def anytop13_to_ktjd17(x13: np.ndarray, parents: np.ndarray, *, fps: float, eps_h: float
                       ) -> tuple[np.ndarray, np.ndarray, dict]:
    """Raw AnyTop-13 (17-slot container) [T,J,17] -> (raw KTJD-17 [T,J,17] float64, heading_valid [T] bool, diagnostics).
    See the module docstring (INVERSE) for every rule; the split follows the frozen PZ release (un-smoothed root track)."""
    x = np.asarray(x13, dtype=np.float64)
    if x.ndim != 3 or x.shape[-1] != 17 or x.shape[0] < 1:
        raise ValueError(f"motion must be [T,J,17], got {x.shape}")
    T, J = x.shape[:2]
    par = validate_parents(parents, J)
    # facing: degenerate frames hold the previous facing (identity at t = 0) and are flagged
    R_face, bad_face = safe_decode6d(x[:, 0, 3:9])
    for t in range(T):
        if bad_face[t]:
            R_face[t] = R_face[t - 1] if t > 0 else np.eye(3)
    Rt = np.swapaxes(R_face, -1, -2)
    step = np.zeros((T, 3))
    step[1:, 0] = x[:-1, 0, 9] / float(fps); step[1:, 2] = x[:-1, 0, 11] / float(fps)
    root = np.cumsum(np.einsum("tab,tb->ta", Rt, step), axis=0)
    root[:, 1] = x[:, 0, 1]
    P = np.einsum("tab,tjb->tja", Rt, x[..., 0:3])
    P[..., 0] += root[:, None, 0]; P[..., 2] += root[:, None, 2]
    P[:, 0] = root
    ch = children_of(par)
    WR = np.empty((T, J, 3, 3)); n_bad_slots = 0
    for p in range(J):
        gp = int(par[p])
        if ch[p]:
            L, bad = safe_decode6d(x[:, ch[p][-1], 3:9])           # LAST child's slot: the native FK's last-child-wins rule
            n_bad_slots += int(bad.sum())                         # degenerate -> identity local rotation (safe_decode6d)
            WR[:, p] = L if gp < 0 else np.matmul(WR[:, gp], L)
        else:
            WR[:, p] = WR[:, gp] if gp >= 0 else np.tile(np.eye(3), (T, 1, 1))   # leaf: no slot of its own
    out = np.zeros((T, J, 17), dtype=np.float64)
    out[..., 0:3] = P
    out[..., 0] -= root[:, None, 0]; out[..., 2] -= root[:, None, 2]          # the release split: raw root track
    out[..., 3:9] = encode_column_cont6d(WR)
    vel = np.zeros_like(P)
    if T >= 2:
        vel[:-1] = np.einsum("tab,tjb->tja", Rt[1:], x[:-1, :, 9:12])         # predicted local velocities -> world, per second
        vel[:-1, 0, 1] = (x[1:, 0, 1] - x[:-1, 0, 1]) * float(fps)           # the root row has no vertical slot: from the height track
        vel[-1] = vel[-2]
    out[..., 9:12] = vel
    out[..., 12] = x[..., 12]
    out[:, 0, 13:15] = root[:, [0, 2]] - root[0:1, [0, 2]]
    fwd = Rt[:, :, 2]                                             # R_face^T z-hat = the world forward vector
    n = np.hypot(fwd[:, 0], fwd[:, 2]); valid = (n >= eps_h) & ~bad_face
    out[valid, 0, 15] = fwd[valid, 2] / n[valid]; out[valid, 0, 16] = fwd[valid, 0] / n[valid]
    return out, valid, {"degenerate_facing_frames": int(bad_face.sum()), "degenerate_child_slots": n_bad_slots}
