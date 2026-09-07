"""Skeleton-robustness augmentation for KTJD-17 in-context training (user 2026-09-07: the model itself must
become robust to rigs with other joint counts, joint names and rest conventions; code first, training later).

Four sampling-time perturbations of ONE served sample (target clip AND its demo, same transform):

  1. random sub-skeleton  -- drop up to `drop_max_frac` of the droppable joints (never the root = joint 0,
     never a joint that makes contact in the target clip); `drop_mode` "any" removes joints anywhere and re-parents
     every kept child to its nearest kept ancestor, "tips" prunes leaves only (iteratively), which keeps FK exact. KTJD-17 is invariant to joint removal (absolute positions, per-joint rest-delta rotations,
     per-joint velocity / contact), so the kept joints' channels are copied unchanged; only the tree
     (parents, bone offsets, hop distances, structural features, description rows) changes. A re-parented
     child's bone is the rest vector to its new parent expressed in that parent's rest frame; the forward
     kinematics of a virtual bone is exact only where the removed joint was rigid in the data.
  2. rest-convention perturbation -- every joint's rest rotation is replaced by Q_j R_rest_j with a random
     Q_j (angle <= `rest_deg`), fixed per sample. Global rotations are unchanged, so the served rest deltas
     become delta_j Q_j^T (the demo rest frame follows: identity -> Q_j^T), bone offsets are unchanged, and
     the rest positions are rebuilt by forward kinematics of the new rest rotations.
  3. description-embedding noise -- Gaussian noise on the joint description embeddings, scaled per row by
     the row's RMS, and with probability `sem_drop_p` the whole table of the sample is zeroed.
  4. statistics perturbation -- the per-cell mean and std the sample is normalised with are perturbed
     (mean += N(0, stats_shift) * std, std *= exp N(0, stats_logsd)) on the per-joint cells 0:13; the FK
     pack carries the perturbed statistics, so decoding stays consistent. (A uniform bone-length scaling
     with matched statistics is a no-op in normalised space, so proportion error is represented here.)
     Root cells 13:17 are never perturbed: the crop contract re-bases smooth-root XZ with the rig's stats.

Everything is a pure function of (rng, sample); nothing is cached per rig. `AugConfig.protocol()` is the
record the trainer pins into checkpoints and compares with the gamma calibration.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np

from src.data.anytop_dataset import _STD_FLOOR
from src.data.ktjd17.codec import decode_column_cont6d, encode_column_cont6d

AUG_VERSION = "ktjd17_skel_aug_v1"
STATS_Z_CLIP = 3.0            # standard-normal draws of the statistics perturbation are clipped to +-3 sigma
STATS_LOGSD_MAX = 1.0         # exp(+-3) at most: the effective scale stays within [e^-3, e^3] of the rig's
STATS_SHIFT_MAX = 3.0


@dataclass(frozen=True)
class AugConfig:
    p: float = 0.0               # probability that a training sample is augmented at all (0 = off)
    drop_max_frac: float = 0.0   # sub-skeleton: fraction of droppable joints removed, uniform in [0, max]
    drop_mode: str = "any"       # "any": any non-protected joint (children re-parented; FK through the virtual bone is inexact
    #                              where the removed joint articulated); "tips": prune leaves only, iteratively (FK stays exact)
    rest_deg: float = 0.0        # rest convention: max rotation angle per joint, degrees
    sem_noise: float = 0.0       # description embeddings: noise std relative to the row RMS
    sem_drop_p: float = 0.0      # description embeddings: probability of zeroing the whole table
    stats_logsd: float = 0.0     # statistics: std of the log-normal factor on per-cell std
    stats_shift: float = 0.0     # statistics: std of the mean shift, in units of the per-cell std

    def __post_init__(self):
        if self.drop_mode not in ("any", "tips"):
            raise ValueError(f"AugConfig.drop_mode must be 'any' or 'tips', got {self.drop_mode!r}")
        for k, v in asdict(self).items():
            if k == "drop_mode":
                continue
            if not np.isfinite(v) or v < 0:
                raise ValueError(f"AugConfig.{k} must be finite and >= 0, got {v!r}")
        if self.p > 1 or self.sem_drop_p > 1 or self.drop_max_frac > 1:
            raise ValueError("AugConfig probabilities / fractions must be <= 1")
        if self.stats_logsd > STATS_LOGSD_MAX or self.stats_shift > STATS_SHIFT_MAX:
            raise ValueError(f"AugConfig.stats_logsd <= {STATS_LOGSD_MAX} and stats_shift <= {STATS_SHIFT_MAX} are supported "
                             f"(draws are clipped to +-{STATS_Z_CLIP} sigma; beyond that the effective scale can vanish)")
        if self.p > 0 and not any((self.drop_max_frac, self.rest_deg, self.sem_noise, self.sem_drop_p,
                                   self.stats_logsd, self.stats_shift)):
            raise ValueError("AugConfig.p > 0 but every perturbation is 0 -- nothing to apply")

    @property
    def active(self) -> bool:
        return self.p > 0

    def protocol(self) -> dict | None:
        """None when off (so an unaugmented run matches a calibration that never heard of augmentation)."""
        return {"version": AUG_VERSION, **asdict(self), "stats_z_clip": STATS_Z_CLIP} if self.active else None


def hop_matrix(parents) -> np.ndarray:
    """[J,J] float32 tree hop distances, the served-order Floyd geodesic (same algorithm as Ktjd17Base._geodesic)."""
    par = [int(p) for p in parents]
    J = len(par)
    depth = np.zeros(J, np.int64)
    anc = []
    for j in range(J):
        if j:
            depth[j] = depth[par[j]] + 1
        c, k = {}, j
        while k >= 0:
            c[k] = depth[j] - depth[k]
            k = par[k]
        anc.append(c)
    g = np.zeros((J, J), np.float32)
    for i in range(J):
        for j in range(J):
            common = anc[i].keys() & anc[j].keys()
            l = max(common, key=lambda k: depth[k])
            g[i, j] = (depth[i] - depth[l]) + (depth[j] - depth[l])
    return g


def _random_rotation(rng, max_deg: float) -> np.ndarray:
    axis = rng.normal(size=3)
    axis /= max(np.linalg.norm(axis), 1e-12)
    ang = np.radians(float(rng.uniform(0.0, max_deg)))
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(ang) * K + (1 - np.cos(ang)) * (K @ K)


@dataclass
class SkeletonTransform:
    keep: np.ndarray            # [J'] original indices kept, ascending (keep[0] == 0)
    parents: np.ndarray         # [J'] new parents (compacted indices)
    offsets: np.ndarray         # [J',3] float64 rest bone offsets under the new parents (parent rest frame)
    R_rest: np.ndarray          # [J',3,3] float64 perturbed rest rotations Q_j R_rest_j
    P_rest: np.ndarray          # [J',3] float64 rest positions rebuilt from the new offsets and rest rotations
    Q: np.ndarray               # [J',3,3] the rest-convention rotations (identity where not applied)
    rot_rows: np.ndarray        # [J'] bool: rows whose rest-delta channels are valid and get Q
    mu: np.ndarray              # [J',17] perturbed normalisation mean
    sd: np.ndarray              # [J',17] perturbed normalisation std (minus the floor, serving convention)
    mu0: np.ndarray             # [J',17] the rig's mean on the kept rows (what the served arrays were normalised with)
    sd0: np.ndarray             # [J',17]
    channel_valid: np.ndarray   # [J',17] bool
    sem_noise: float
    sem_drop: bool


def make_transform(rng, cfg: AugConfig, *, parents, P_rest_global, R_rest_global, offset_parent_local,
                   channel_valid, mu, sd, contact_joints) -> SkeletonTransform:
    """One sample's transform. `contact_joints` [J] bool marks joints that touch the ground in the target clip
    (protected from dropping); `mu`/`sd` [J,17] are the rig's serving statistics."""
    par = np.asarray(parents, dtype=np.int64)
    J = len(par)
    if par[0] != -1 or np.any(par[1:] >= np.arange(1, J)):
        raise ValueError("parents must be in FK order with the root at 0")
    P = np.asarray(P_rest_global, dtype=np.float64)
    R = np.asarray(R_rest_global, dtype=np.float64)
    O = np.asarray(offset_parent_local, dtype=np.float64)
    cv = np.asarray(channel_valid, dtype=bool)
    # ---- 1. sub-skeleton ----
    keep_mask = np.ones(J, dtype=bool)
    if cfg.drop_max_frac > 0:
        protected = np.asarray(contact_joints, dtype=bool).copy()
        protected[0] = True
        droppable = np.where(~protected)[0]
        n_drop = int(round(float(rng.uniform(0.0, cfg.drop_max_frac)) * len(droppable)))
        if cfg.drop_mode == "any":
            if n_drop > 0:
                keep_mask[rng.choice(droppable, size=n_drop, replace=False)] = False
        else:
            children = [[] for _ in range(J)]
            for j in range(1, J):
                children[par[j]].append(j)
            for _ in range(n_drop):                  # prune from the tips: a dropped joint never has kept children
                leaves = [j for j in droppable if keep_mask[j] and not any(keep_mask[c] for c in children[j])]
                if not leaves:
                    break
                keep_mask[int(rng.choice(leaves))] = False
    keep = np.where(keep_mask)[0]
    new_index = -np.ones(J, dtype=np.int64)
    new_index[keep] = np.arange(len(keep))
    new_par = np.empty(len(keep), dtype=np.int64)
    new_off = np.empty((len(keep), 3), dtype=np.float64)
    for n, j in enumerate(keep):
        q = int(par[j])
        while q >= 0 and not keep_mask[q]:
            q = int(par[q])
        new_par[n] = new_index[q] if q >= 0 else -1
        if q >= 0:
            # bone to the nearest kept ancestor, in that ancestor's (original) rest frame:
            # positions_child = positions_parent + R_parent @ offset  (codec.fk_from_global_rotations)
            new_off[n] = R[q].T @ (P[j] - P[q])
            if q == par[j] and not np.allclose(new_off[n], O[j], atol=1e-4, rtol=1e-4):
                raise AssertionError(f"rest offset convention mismatch at joint {j}: recomputed {new_off[n]} "
                                     f"vs stored {O[j]}")
        else:
            new_off[n] = 0.0
    # ---- 2. rest convention ----
    cvk = cv[keep]
    rot_rows = cvk[:, 3:9].all(axis=1)
    Q = np.tile(np.eye(3), (len(keep), 1, 1))
    if cfg.rest_deg > 0:
        for n in np.where(rot_rows)[0]:
            Q[n] = _random_rotation(rng, cfg.rest_deg)
    R_new = Q @ R[keep]
    P_new = np.empty((len(keep), 3), dtype=np.float64)
    P_new[0] = P[keep[0]]
    for n in range(1, len(keep)):
        P_new[n] = P_new[new_par[n]] + R_new[new_par[n]] @ new_off[n]
    # ---- 4. statistics ----
    mu0 = np.asarray(mu, dtype=np.float32)[keep].copy()
    sd0 = np.asarray(sd, dtype=np.float32)[keep].copy()
    mu_n, sd_n = mu0.copy(), sd0.copy()
    if cfg.stats_logsd > 0 or cfg.stats_shift > 0:
        cells = cvk.copy()
        cells[:, 13:17] = False                          # root smooth-XZ / heading: crop re-base uses the rig's stats
        eff = sd0.astype(np.float64) + _STD_FLOOR
        z_mu = np.clip(rng.normal(0.0, 1.0, mu0.shape), -STATS_Z_CLIP, STATS_Z_CLIP) * cfg.stats_shift
        z_sd = np.clip(rng.normal(0.0, 1.0, sd0.shape), -STATS_Z_CLIP, STATS_Z_CLIP) * cfg.stats_logsd
        mu64 = np.where(cells, mu0.astype(np.float64) + z_mu * eff, mu0.astype(np.float64))
        sd64 = np.where(cells, eff * np.exp(z_sd) - _STD_FLOOR, sd0.astype(np.float64))
        mu_n, sd_n = mu64.astype(np.float32), sd64.astype(np.float32)
        if not (np.isfinite(mu_n).all() and np.isfinite(sd_n).all() and np.all(sd_n.astype(np.float64) + _STD_FLOOR > 0)):
            raise AssertionError("statistics perturbation produced a non-finite mean or a non-positive effective scale")
    # ---- 3. descriptions ----
    sem_drop = bool(cfg.sem_drop_p > 0 and rng.random() < cfg.sem_drop_p)
    return SkeletonTransform(keep=keep, parents=new_par, offsets=new_off, R_rest=R_new, P_rest=P_new, Q=Q,
                             rot_rows=rot_rows, mu=mu_n.astype(np.float32), sd=sd_n.astype(np.float32),
                             mu0=mu0, sd0=sd0, channel_valid=cvk, sem_noise=float(cfg.sem_noise),
                             sem_drop=sem_drop)


def apply_motion(x18: np.ndarray, tr: SkeletonTransform) -> np.ndarray:
    """[T,J,18] (or [J,18]) served-normalised motion -> [T,J',18] under the transform.
    Plane 17 (heading-valid flag) is copied; channels 0:17 are de-normalised with the rig's statistics,
    rotated (rest deltas -> delta Q^T on rows with valid rotation cells) and re-normalised with the
    perturbed statistics (unperturbed on cells outside channel_valid, which cfm_loss projects out anyway)."""
    squeeze = x18.ndim == 2
    x = x18[None] if squeeze else x18
    x = np.asarray(x, dtype=np.float32)[:, tr.keep]
    raw = x[..., :17].astype(np.float64) * (tr.sd0[None] + _STD_FLOOR) + tr.mu0[None]
    rows = np.where(tr.rot_rows & ~np.all(np.isclose(tr.Q, np.eye(3), atol=0.0), axis=(1, 2)))[0]
    if rows.size:
        d6 = raw[:, rows, 3:9]                                            # [T,r,6]
        a1, a2 = d6[..., :3], d6[..., 3:]
        n1 = np.linalg.norm(a1, axis=-1)
        b1 = a1 / np.maximum(n1[..., None], 1e-12)
        n2 = np.linalg.norm(a2 - np.sum(b1 * a2, axis=-1, keepdims=True) * b1, axis=-1)
        ok = (n1 > 1e-6) & (n2 > 1e-6)                                    # both Gram-Schmidt norms (decoder's GT eps)
        if ok.any():                                                      # degenerate cells keep their served six-vector
            M = decode_column_cont6d(d6[ok], strict=True)                 # [n,3,3]
            Qt = np.broadcast_to(np.swapaxes(tr.Q[rows], -1, -2)[None], (d6.shape[0], rows.size, 3, 3))[ok]
            d6n = d6.copy()
            d6n[ok] = encode_column_cont6d(np.matmul(M, Qt))
            raw[:, rows, 3:9] = d6n
    out = x.copy()
    out[..., :17] = ((raw - tr.mu[None]) / (tr.sd[None] + _STD_FLOOR)).astype(np.float32)
    return out[0] if squeeze else out


def apply_semantics(sem: np.ndarray, tr: SkeletonTransform, rng) -> np.ndarray:
    s = np.asarray(sem, dtype=np.float32)[tr.keep].copy()
    if tr.sem_drop:
        return np.zeros_like(s)
    if tr.sem_noise > 0:
        rms = np.sqrt(np.mean(s ** 2, axis=1, keepdims=True))
        s += rng.normal(0.0, 1.0, s.shape).astype(np.float32) * (tr.sem_noise * rms).astype(np.float32)
    return s
