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

Three KINEMATICS-PRESERVING operations (v2, 2026-09-13; after UniMate's augmentations, written for this
representation) change the body or the tree and then RE-ENCODE the sample from forward kinematics, so the served
direct positions, velocities and rotations stay exactly consistent (the corpus invariant the FK term supervises):

  5. bone-length perturbation -- every bone offset is scaled by an independent factor in [1-s, 1+s]; rotations are
     unchanged and every joint position is recomputed by FK from the (unchanged) root through the new bones.
  6. chain pooling -- a single-child interior joint is removed and its child re-parented to the grandparent with the
     rest vector as its new bone; positions are recomputed by FK through the virtual bone (where the removed joint
     articulated, the child's subtree moves rigidly with the grandparent from then on: the sample is a slightly
     different, self-consistent motion, not an FK-inconsistent one).
  7. joint addition -- one synthetic joint is inserted on a random bone at a fraction alpha in [0.3, 0.7] of its
     length, rigidly attached to the parent (its rest rotation is the parent's, so its rest-delta equals the parent's
     every frame); FK is preserved exactly. Its description embedding is the mean of the parent's and the child's
     (as UniMate does); its channel mask and statistics copy the child's row, except the rotation cells, which copy
     the parent's (they hold the parent's values); its contact flag is zero.

Re-encoding needs every joint's global rotation, delta_j R_rest_j (no rig of this library has fixed-DOF joints; a
degenerate six-vector on a served row is refused, since the emitted rotations could not reconstruct the positions --
the corpus has none), the world positions (served q_position plus the root's smooth track), velocities as the
codec's forward difference at the corpus frame rate, and -- under the rest normalisation -- a mean row for positions
that is the transformed rest pose, so the rest demo still normalises to zero on the position channels. Position and
velocity cells the statistics artifact excluded as exact constants (four rigs, near-root joints) re-enter the model
input and the loss of a re-encoded sample with the channel's scale from the rig's valid rows: they no longer hold
that constant. Contact flags of existing joints are kept (bone scaling and pooling move joints by small amounts);
root channels 13:17 are untouched.

Everything is a pure function of (rng, sample); nothing is cached per rig. `AugConfig.protocol()` is the
record the trainer pins into checkpoints and compares with the gamma calibration.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np

from src.data.anytop_dataset import _STD_FLOOR
from src.data.ktjd17.codec import decode_column_cont6d, encode_column_cont6d

AUG_VERSION = "ktjd17_skel_aug_v1"      # protocol record of the four v1 perturbations (pinned in the v1 checkpoints)
AUG_VERSION_KIN = "ktjd17_skel_aug_v2"  # + bone_scale / pool_frac / add_p (FK re-encoded) when any of the three is active
STATS_Z_CLIP = 3.0            # standard-normal draws of the statistics perturbation are clipped to +-3 sigma
STATS_LOGSD_MAX = 1.0         # exp(+-3) at most: the effective scale stays within [e^-3, e^3] of the rig's
STATS_SHIFT_MAX = 3.0
AUG_VERSION_ONE_OF = "ktjd17_skel_aug_v3.6_one_of"   # UniMate's rule: ONE kinematic operation per augmented sample; v3.1 =
#                                                       UniMate's rotation semantics for pool / add (ONE_OF_ROTATION_RULE);
#                                                       v3.2 = the rotation cells of recomposed rows re-enter (ONE_OF_REENTRY);
#                                                       v3.3 / v3.4 = contact flags pruned on the recomposed rows (ONE_OF_CONTACT);
#                                                       v3.5 = the foot-lock term keeps the pre-pruning pair count (ONE_OF_LOCK_DEN);
#                                                       v3.6 = the recomposed rows' contact cells re-enter too (ONE_OF_REENTRY)
ONE_OF_CONTACT = "pruned_on_recomposed_rows_where_displacement_grows"   # apply_motion, pool / add only: a served contact flag on a
#                                                       RECOMPOSED row (the synthetic joint's subtree, rows below a pooled joint)
#                                                       is cleared at frame t where the joint's re-encoded forward displacement
#                                                       exceeds its original one by more than ONE_OF_CONTACT_ABS mean bone
#                                                       lengths: a joint the recomposition swung is no longer a locked target
#                                                       of the foot-lock term. Every other flag is carried over: the scale op
#                                                       moves the joints by amounts the bone factors set (empirically its lock
#                                                       stays near the clip's own, the test's scale reproducer), and pruning it
#                                                       removed the planted feet and raised the term's mean (codex-r6). The
#                                                       corpus flags are not reproducible from the served channels (codex-r5
#                                                       diagnostic: 34% of cells disagree with the schema's height / speed rule),
#                                                       so they are never re-derived.
ONE_OF_CONTACT_ABS = 0.05      # in mean bone lengths per frame (1.5 bone lengths per second at 30 fps): a deleted or duplicated
#                                articulation swings a foot by several times that per frame, a bone-scale wobble by far less
ONE_OF_LOCK_DEN = "pre_pruning_pairs"   # the dataset serves the window's pre-pruning contact-pair count (lock_denominator) and the
#                                         foot-lock term (fk_torch.ktjd_dynamics_losses) divides by it, so pruning a pair removes its
#                                         demand without re-weighting the surviving pairs (codex-r6 / r7)
ONE_OF_REENTRY = "rotation_and_contact_cells_of_recomposed_rows"   # make_transform: a rotation or contact cell excluded as a
#                                                       constant on a row whose rest-delta is recomposed becomes valid with the
#                                                       channel's serving scale (its rotation is recomposed, its contact flag may
#                                                       be pruned; a cell the trainer projects to its constant would undo either)
ONE_OF_ROTATION_RULE = "unimate_local_slots"          # apply_motion: a pooled joint's articulation deleted, the added joint
#                                                       duplicates its parent's parent-relative delta (their slot copies);
#                                                       recorded so an artifact measured under the earlier rule cannot match
# UniMate's own constants (outside_docs/UniMate/unimate/dataset/mixture/{dataset.py,augmentations.py}): leaves removed at a
# rate uniform in [0.05, 0.15] of the leaves, at most 3; single-child joints pooled at a rate uniform in [0.1, 0.3], at least
# 1; both picks weighted by bone_length^-0.5 (shorter bones first); one joint added at alpha ~ U(0.3, 0.7) along a random
# bone plus a bone-aligned ellipsoid displacement (truncated half-normal radius, sigma 0.5 of the axial semi-axis = the bone
# length; lateral semi-axes 0.5 of it); every bone scaled by U[1-s, 1+s]. The op is drawn uniformly from the four; the
# "no-op" candidate of UniMate's five-way draw is the dataset's 1-p. Recorded in the protocol so a calibration or a
# checkpoint binds them.
ONE_OF = dict(remove_rate=[0.05, 0.15], remove_cap=3, pool_rate=[0.1, 0.3], pool_min=1, select_alpha=0.5,
              add_alpha=[0.3, 0.7], add_sigma=0.5, add_lateral=0.5)
ONE_OF_OPS = ("add", "remove", "pool", "scale")
# what this port keeps of OUR representation on top of UniMate's rule (recorded in the protocol so the arm binds them):
# the root and the target clip's contact joints are never removed or pooled; under the rest normalisation the position
# mean of a re-encoded sample is the transformed rest pose (the rest demo keeps normalising to zero; UniMate keeps the
# source rows' statistics); the world-velocity channels are re-encoded from the FK'd positions (UniMate leaves its
# velocity channels untouched); the added joint's rest rotation, mask and statistics follow apply_motion's synthetic-row
# rule.
ONE_OF_DEVIATIONS = ["protected_root_and_contact_joints", "rest_norm_mean_is_transformed_rest", "velocity_reencoded",
                     "contact_pruned_on_recomposed_rows", "synthetic_row_rule"]


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
    bone_scale: float = 0.0      # kinematics-preserving: per-bone length factor uniform in [1-bone_scale, 1+bone_scale]
    pool_frac: float = 0.0       # kinematics-preserving: fraction of single-child interior joints pooled away, uniform in [0, max]
    add_p: float = 0.0           # kinematics-preserving: probability of inserting one synthetic joint on a random bone
    mode: str = "joint"          # "joint": every enabled perturbation on every augmented sample; "one_of": UniMate's rule --
    #                              one of add / remove / pool / scale per augmented sample with UniMate's rates (ONE_OF),
    #                              bone_scale the magnitude of scale, drop_mode "tips", every other field 0

    def __post_init__(self):
        if self.drop_mode not in ("any", "tips"):
            raise ValueError(f"AugConfig.drop_mode must be 'any' or 'tips', got {self.drop_mode!r}")
        if self.mode not in ("joint", "one_of"):
            raise ValueError(f"AugConfig.mode must be 'joint' or 'one_of', got {self.mode!r}")
        for k, v in asdict(self).items():
            if k in ("drop_mode", "mode"):
                continue
            if not np.isfinite(v) or v < 0:
                raise ValueError(f"AugConfig.{k} must be finite and >= 0, got {v!r}")
        if self.p > 1 or self.sem_drop_p > 1 or self.drop_max_frac > 1:
            raise ValueError("AugConfig probabilities / fractions must be <= 1")
        if self.stats_logsd > STATS_LOGSD_MAX or self.stats_shift > STATS_SHIFT_MAX:
            raise ValueError(f"AugConfig.stats_logsd <= {STATS_LOGSD_MAX} and stats_shift <= {STATS_SHIFT_MAX} are supported "
                             f"(draws are clipped to +-{STATS_Z_CLIP} sigma; beyond that the effective scale can vanish)")
        if self.bone_scale >= 1.0 or self.pool_frac > 1.0 or self.add_p > 1.0:
            raise ValueError("AugConfig.bone_scale < 1, pool_frac <= 1 and add_p <= 1")
        if self.p > 0 and not any((self.drop_max_frac, self.rest_deg, self.sem_noise, self.sem_drop_p,
                                   self.stats_logsd, self.stats_shift, self.bone_scale, self.pool_frac, self.add_p)):
            raise ValueError("AugConfig.p > 0 but every perturbation is 0 -- nothing to apply")
        if self.mode == "one_of":
            if self.drop_mode != "tips":
                raise ValueError("AugConfig.mode 'one_of' removes leaves only (UniMate): drop_mode must be 'tips'")
            if self.bone_scale <= 0:
                raise ValueError("AugConfig.mode 'one_of' needs bone_scale > 0 (the magnitude of its scale operation)")
            if any((self.drop_max_frac, self.rest_deg, self.sem_noise, self.sem_drop_p, self.stats_logsd, self.stats_shift,
                    self.pool_frac, self.add_p)):
                raise ValueError("AugConfig.mode 'one_of' draws one of add / remove / pool / scale per sample with UniMate's "
                                 "own rates (ONE_OF); drop_max_frac, pool_frac, add_p and the rest / statistics / "
                                 "description fields must be 0")

    @property
    def active(self) -> bool:
        return self.p > 0

    def protocol(self) -> dict | None:
        """None when off (so an unaugmented run matches a calibration that never heard of augmentation). With the three
        kinematics-preserving operations at 0 the record is the v1 protocol, key for key (v1 checkpoints and calibrations
        keep matching); any of them active records the v2 protocol with the three fields."""
        if not self.active:
            return None
        d = asdict(self)
        mode = d.pop("mode")                              # absent from the v1 / v2 records (their checkpoints keep matching)
        if mode == "one_of":
            return {"version": AUG_VERSION_ONE_OF, "mode": mode, "p": d["p"], "drop_mode": d["drop_mode"],
                    "bone_scale": d["bone_scale"], "ops": list(ONE_OF_OPS), **ONE_OF, "rotation_rule": ONE_OF_ROTATION_RULE,
                    "reentry": ONE_OF_REENTRY, "contact": ONE_OF_CONTACT, "contact_abs_bl": ONE_OF_CONTACT_ABS,
                    "lock_denominator": ONE_OF_LOCK_DEN, "deviations": list(ONE_OF_DEVIATIONS)}
        kin = {k: d.pop(k) for k in ("bone_scale", "pool_frac", "add_p")}
        if any(kin.values()):
            return {"version": AUG_VERSION_KIN, **d, **kin, "stats_z_clip": STATS_Z_CLIP}
        return {"version": AUG_VERSION, **d, "stats_z_clip": STATS_Z_CLIP}


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


def _weighted_pick(rng, cands: np.ndarray, offsets: np.ndarray, n: int) -> list[int]:
    """UniMate's bone_length_weighted_sample: `n` of `cands` without replacement, weight bone_length^-select_alpha
    (shorter bones first); `offsets[j]` is joint j's rest bone to its parent."""
    cands = np.asarray(cands, dtype=np.int64)
    n = min(int(n), len(cands))
    if n <= 0:
        return []
    bl = np.maximum(np.linalg.norm(offsets[cands], axis=-1), 1e-8)
    w = bl ** (-float(ONE_OF["select_alpha"]))
    return [int(j) for j in rng.choice(cands, size=n, replace=False, p=w / w.sum())]


def _ellipsoid_displacement(rng, bone_offset: np.ndarray, sigma: float, lateral_ratio: float) -> np.ndarray:
    """UniMate's sample_ellipsoid_gaussian: a displacement inside the bone-aligned ellipsoid (axial semi-axis = the bone
    length, lateral semi-axes = lateral_ratio of it), a uniform direction on the sphere times a truncated half-normal
    radius (sigma, cut at 1)."""
    L = float(np.linalg.norm(bone_offset))
    if L < 1e-8:
        return np.zeros(3)
    while True:
        d = rng.uniform(-1.0, 1.0, size=3)
        nd = float(np.linalg.norm(d))
        if 0 < nd <= 1:
            d /= nd
            break
    while True:
        r = abs(float(rng.normal(0.0, sigma)))
        if r <= 1:
            break
    scaled = (r * d) * np.array([L, L * lateral_ratio, L * lateral_ratio])
    e1 = bone_offset / L
    ref = np.array([0.0, 1.0, 0.0])
    if abs(float(np.dot(e1, ref))) > 0.9:
        ref = np.array([1.0, 0.0, 0.0])
    e2 = np.cross(e1, ref)
    e2 /= np.linalg.norm(e2)
    e3 = np.cross(e1, e2)
    return np.column_stack([e1, e2, e3]) @ scaled


@dataclass
class SkeletonTransform:
    keep: np.ndarray            # [Jk] original indices of the kept ORIGINAL joints, ascending (keep[0] == 0)
    src: np.ndarray             # [J'] served joint -> original index, or -1 for a synthetic (added) joint
    synth_pc: np.ndarray        # [J',2] synthetic joint -> (original parent index, original child index); -1 elsewhere
    parents: np.ndarray         # [J'] new parents (compacted indices, FK order)
    offsets: np.ndarray         # [J',3] float64 rest bone offsets under the new parents (parent rest frame)
    R_rest: np.ndarray          # [J',3,3] float64 perturbed rest rotations Q_j R_rest_j
    R_rest_old: np.ndarray      # [J',3,3] float64 UNperturbed rest rotations (a synthetic joint: its parent's)
    P_rest: np.ndarray          # [J',3] float64 rest positions rebuilt from the new offsets and rest rotations
    Q: np.ndarray               # [J',3,3] the rest-convention rotations (identity where not applied)
    rot_rows: np.ndarray        # [J'] bool: rows whose rest-delta channels are valid and get Q
    mu: np.ndarray              # [J',17] perturbed normalisation mean
    sd: np.ndarray              # [J',17] perturbed normalisation std (minus the floor, serving convention)
    mu0: np.ndarray             # [J',17] the rig's mean on the served rows (what the served arrays were normalised with)
    sd0: np.ndarray             # [J',17]
    channel_valid: np.ndarray   # [J',17] bool
    sem_noise: float
    sem_drop: bool
    reencode: bool = False      # positions / velocities are recomputed by FK (bone scaling, pooling or addition happened)
    fps: float = 30.0
    op: str | None = None       # mode "one_of": the operation drawn for this sample (add / remove / pool / scale)
    unimate_rot: bool = False   # mode "one_of" pool / add: UniMate's rotation semantics -- apply_motion recomposes the served
    #                             rest-deltas through the new tree from every ORIGINAL joint's parent-relative delta (a pooled
    #                             joint's articulation is deleted, the added joint duplicates its parent's); needs the original
    #                             tree and the rig's statistics on every original row
    recomposed: np.ndarray | None = None   # [J'] bool (unimate_rot only): rows whose rest-delta the recomposition changes -- the
    #                                        synthetic joint and its subtree, rows with a pooled original ancestor
    par0: np.ndarray | None = None      # [J] original parents (unimate_rot only)
    mu_all: np.ndarray | None = None    # [J,17] the rig's serving statistics on every original joint (unimate_rot only)
    sd_all: np.ndarray | None = None

    @property
    def n_joints(self) -> int:
        return int(self.src.shape[0])

    @property
    def served_rows(self) -> np.ndarray:
        """[Jk] positions (in the J' order) of the kept original joints, in `keep` order."""
        return np.where(self.src >= 0)[0]


def _fk(parents, root_positions, G, offsets):
    """positions [T,J,3] = FK of global rotations G [T,J,3,3] through offsets (parent rest frame), root given."""
    T, J = G.shape[:2]
    pos = np.empty((T, J, 3), dtype=np.float64)
    pos[:, 0] = root_positions
    for j in range(1, J):
        p = int(parents[j])
        pos[:, j] = pos[:, p] + np.einsum("tij,j->ti", G[:, p], offsets[j])
    return pos


def make_transform(rng, cfg: AugConfig, *, parents, P_rest_global, R_rest_global, offset_parent_local,
                   channel_valid, mu, sd, contact_joints, fps: float = 30.0, rest_norm: bool = False) -> SkeletonTransform:
    """One sample's transform. `contact_joints` [J] bool marks joints that touch the ground in the target clip
    (protected from dropping and pooling); `mu`/`sd` [J,17] are the rig's serving statistics; `fps` is the corpus
    frame rate the velocity channels were encoded at; `rest_norm` says the serving mean is the rig's rest frame
    (Ktjd17Base normalization "rest"), so a re-encoded sample gets the transformed rest pose as its position mean.
    Mode "one_of" draws ONE of add / remove / pool / scale (UniMate's rule and rates, ONE_OF) instead of applying every
    enabled perturbation; the draw is recorded in the transform's `op`. In that mode a re-encoded sample keeps a served
    contact flag only where the joint moves no more than it did (apply_motion, ONE_OF_CONTACT; codex 2026-09-15 unimate r5 P1)."""
    par = np.asarray(parents, dtype=np.int64)
    J = len(par)
    if par[0] != -1 or np.any(par[1:] >= np.arange(1, J)):
        raise ValueError("parents must be in FK order with the root at 0")
    P = np.asarray(P_rest_global, dtype=np.float64)
    R = np.asarray(R_rest_global, dtype=np.float64)
    O = np.asarray(offset_parent_local, dtype=np.float64)
    cv = np.asarray(channel_valid, dtype=bool)
    protected = np.asarray(contact_joints, dtype=bool).copy()
    protected[0] = True                                   # the root also carries the heading / track channels
    children = [[] for _ in range(J)]
    for j in range(1, J):
        children[par[j]].append(j)
    reencode = False
    op = ONE_OF_OPS[int(rng.integers(len(ONE_OF_OPS)))] if cfg.mode == "one_of" else None   # UniMate: one op per sample
    do_scale = cfg.bone_scale > 0 and (op is None or op == "scale")
    # ---- 1. sub-skeleton ----
    keep_mask = np.ones(J, dtype=bool)
    if cfg.drop_max_frac > 0:
        droppable = np.where(~protected)[0]
        n_drop = int(round(float(rng.uniform(0.0, cfg.drop_max_frac)) * len(droppable)))
        if cfg.drop_mode == "any":
            if n_drop > 0:
                keep_mask[rng.choice(droppable, size=n_drop, replace=False)] = False
        else:
            for _ in range(n_drop):                  # prune from the tips: a dropped joint never has kept children
                leaves = [j for j in droppable if keep_mask[j] and not any(keep_mask[c] for c in children[j])]
                if not leaves:
                    break
                keep_mask[int(rng.choice(leaves))] = False
    elif op == "remove":
        # UniMate's apply_joint_removal: the CURRENT leaves only (no chain is eaten from the tip), a rate uniform in
        # remove_rate of them, at most remove_cap, shorter bones first; the root and the contact joints stay protected
        leaves0 = np.asarray([j for j in range(1, J) if not protected[j] and len(children[j]) == 0], dtype=np.int64)
        n_rm = min(int(round(len(leaves0) * float(rng.uniform(*ONE_OF["remove_rate"])))), int(ONE_OF["remove_cap"]))
        for j in _weighted_pick(rng, leaves0, O, n_rm):
            keep_mask[j] = False
    # ---- 6. chain pooling: single-child interior joints of the CONTRACTED tree (a dropped joint's children hang from its
    #         nearest kept ancestor; codex v2 r1 P2-3), re-parented through; the child lists follow every pooled joint ----
    if cfg.pool_frac > 0 or op == "pool":
        eff_par = np.full(J, -1, dtype=np.int64)          # nearest kept ancestor of every kept joint
        kids = [[] for _ in range(J)]
        for j in range(1, J):
            if keep_mask[j]:
                q = int(par[j])
                while q >= 0 and not keep_mask[q]:
                    q = int(par[q])
                eff_par[j] = q
                kids[q].append(j)
        cands = np.asarray([j for j in range(1, J) if keep_mask[j] and not protected[j] and len(kids[j]) == 1], dtype=np.int64)
        if op == "pool":
            # UniMate's apply_skeleton_pooling: a rate uniform in pool_rate of the candidates, at least pool_min, shorter
            # bones first (a candidate's parent is kept -- only leaves were removed -- so O[j] is its bone)
            n_pool = max(int(ONE_OF["pool_min"]), int(len(cands) * float(rng.uniform(*ONE_OF["pool_rate"])))) if len(cands) else 0
            picked = _weighted_pick(rng, cands, O, n_pool)
        else:
            n_pool = int(round(float(rng.uniform(0.0, cfg.pool_frac)) * len(cands)))
            picked = rng.permutation(cands)[:n_pool]
        for j in picked:
            j = int(j)
            if len(kids[j]) != 1:                          # a neighbour's pooling changed its degree
                continue
            c = kids[j][0]
            q = int(eff_par[j])
            keep_mask[j] = False
            eff_par[c] = q
            kids[q][kids[q].index(j)] = c
            kids[j] = []
            reencode = True
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
    # served-joint tables (J' rows): original joints first as a base, then a synthetic joint may be spliced in
    src = keep.copy()
    synth_pc = -np.ones((len(keep), 2), dtype=np.int64)
    R_old = R[keep].copy()
    cvk = cv[keep].copy()
    mu0 = np.asarray(mu, dtype=np.float32)[keep].copy()
    sd0 = np.asarray(sd, dtype=np.float32)[keep].copy()
    # ---- 7. joint addition: one synthetic joint on a random bone (joint mode: rigid with its parent; one_of: UniMate's
    #         insertion, whose rotation rule apply_motion applies through unimate_rot) ----
    if (cfg.add_p > 0 and len(keep) > 1 and float(rng.random()) < cfg.add_p) or (op == "add" and len(keep) > 1):
        c = int(rng.integers(1, len(keep)))                 # the child whose bone is split (new index)
        p = int(new_par[c])
        alpha = float(rng.uniform(0.3, 0.7))
        off_c = new_off[c].copy()
        ins = c                                             # the synthetic joint takes c's slot; c and later joints shift by one
        shift = lambda idx: idx + 1 if idx >= ins else idx  # noqa: E731
        par2 = np.array([shift(int(x)) if x >= 0 else -1 for x in new_par], dtype=np.int64)
        par2 = np.insert(par2, ins, p)                      # synthetic parent = p
        par2[ins + 1] = ins                                 # c (now at ins+1) hangs from the synthetic joint
        if op == "add":
            # UniMate's apply_joint_addition_ellipsoid: the split point is pushed off the bone axis, inside the ellipsoid,
            # and the two offsets still sum to the bone. The rotation rule is UniMate's as well (apply_motion, unimate_rot):
            # their insertion copies the parent's rotation slot into the new joint, which in their parent-local layout
            # duplicates the parent's local rotation at the new joint (their docstring says "identity local rotation";
            # the code does not do that), so the chain below bends -- codex 2026-09-15 unimate r2 P1-2
            off_new = alpha * off_c + _ellipsoid_displacement(rng, off_c, float(ONE_OF["add_sigma"]), float(ONE_OF["add_lateral"]))
            off_rem = off_c - off_new
        else:
            off_new, off_rem = alpha * off_c, (1.0 - alpha) * off_c
        off2 = np.insert(new_off, ins, off_new, axis=0)
        off2[ins + 1] = off_rem                             # same rest frame: the synthetic joint's rest rotation is p's
        src = np.insert(src, ins, -1)
        synth_pc = np.insert(synth_pc, ins, [int(keep[p]), int(keep[c])], axis=0)
        R_old = np.insert(R_old, ins, R_old[p], axis=0)
        cvk = np.insert(cvk, ins, cvk[c], axis=0); cvk[ins, 3:9] = cvk[p, 3:9]      # rotation cells hold the parent's
        mu0 = np.insert(mu0, ins, mu0[c], axis=0); mu0[ins, 3:9] = mu0[p, 3:9]      # values: its mask and statistics
        sd0 = np.insert(sd0, ins, sd0[c], axis=0); sd0[ins, 3:9] = sd0[p, 3:9]
        new_par, new_off = par2, off2
        reencode = True
    Jn = len(src)
    if new_par[0] != -1 or np.any(new_par[1:] >= np.arange(1, Jn)) or np.any(new_par[1:] < 0):
        raise AssertionError("augmented parents are not in FK order")
    cv_served = cvk.copy()                                # the mask the served rows were produced under
    if reencode or do_scale:
        # positions and velocities are recomputed for every row, so a cell the statistics artifact excluded as an exact
        # constant (Ktjd17Base.static_masks) no longer holds that constant: it re-enters the model input and the loss
        # (codex v2 r1 P1); the structural exclusions (root-only 13:17, fixed_dof rotations) are untouched
        cvk[:, 0:3] = True
        cvk[:, 9:12] = True
    recomposed = None
    if op in ("add", "pool") and reencode:
        # UniMate's rotation semantics (apply_motion, unimate_rot) recompose the rest-deltas of every row whose chain lost a
        # joint (below a pooled one) or gained one (the synthetic joint and its subtree). A rotation cell the statistics
        # artifact excluded as an exact constant on such a row -- the parent's mask copied onto the synthetic row, a constant
        # root cell -- no longer holds that constant, so it re-enters the model input and the loss with the channel's serving
        # scale, exactly like the position / velocity cells above (codex 2026-09-15 unimate r4 P1-1: a synthetic joint under
        # a root with an excluded rotation cell decoded to the constant after the trainer's mask projection, 0.065 bl FK gap)
        recomposed = np.zeros(Jn, dtype=bool)
        for n in range(Jn):
            if src[n] < 0:
                recomposed[n] = True
            else:
                q = int(par[src[n]])
                while q >= 0 and keep_mask[q]:
                    q = int(par[q])
                recomposed[n] = q >= 0                                   # an original ancestor was pooled away
            if not recomposed[n] and new_par[n] >= 0 and recomposed[new_par[n]]:
                recomposed[n] = True                                     # the synthetic joint's subtree
        cvk[recomposed, 3:9] = True
        cvk[recomposed, 12] = True                      # the pruned flag must survive the trainer's projection (codex r8)
    # ---- 5. bone-length perturbation ----
    if do_scale:
        new_off[1:] *= rng.uniform(1.0 - cfg.bone_scale, 1.0 + cfg.bone_scale, size=(Jn - 1, 1))
        reencode = True
    # ---- 2. rest convention ----
    rot_rows = cvk[:, 3:9].all(axis=1)
    Q = np.tile(np.eye(3), (Jn, 1, 1))
    if cfg.rest_deg > 0:
        for n in np.where(rot_rows)[0]:
            Q[n] = _random_rotation(rng, cfg.rest_deg)
    for n in np.where(src < 0)[0]:                        # a synthetic joint shares its parent's rest convention: its served
        Q[n] = Q[new_par[n]]                              # deltas are a copy of the parent's, so R_rest must be the parent's too
    R_new = Q @ R_old
    P_new = np.empty((Jn, 3), dtype=np.float64)
    P_new[0] = P[keep[0]]
    for n in range(1, Jn):
        P_new[n] = P_new[new_par[n]] + R_new[new_par[n]] @ new_off[n]
    # ---- 4. statistics ----
    mu_ref, sd_ref = mu0.copy(), sd0.copy()              # the new skeleton's reference statistics (de-normalisation keeps mu0 / sd0)
    reentered = cvk & ~cv_served
    for ch in np.unique(np.where(reentered)[1]):
        # an excluded cell's serving scale describes a constant (per-cell: ~0); the re-entered cell takes the channel's
        # scale on the rows that were valid (identical under the rest / scale-only normalisation, whose scale is per
        # channel), and that reference scale is what the perturbation below shifts and stretches (codex v2 r2 P2)
        ref = sd0[cv_served[:, ch], ch]
        if ref.size:
            sd_ref[reentered[:, ch], ch] = np.float32(np.median(ref))
    if reencode and rest_norm:
        # under the rest normalisation the mean is the rig's rest pose; the transformed skeleton's is that pose played
        # through the new bones (identity deltas = the UNperturbed rest rotations), FK'd from the mean's own root row
        # (Ktjd17Base._rest_raw17: the rest pose relative to its root XZ) -- exactly what apply_motion recomputes for
        # the rest demo, which therefore still normalises to zero on the position channels
        P_demo = _fk(new_par, mu0[0, 0:3].astype(np.float64)[None], R_old[None], new_off)[0]
        mu_ref[:, 0:3] = P_demo.astype(np.float32)
    mu_n, sd_n = mu_ref.copy(), sd_ref.copy()
    if cfg.stats_logsd > 0 or cfg.stats_shift > 0:
        cells = cvk.copy()
        cells[:, 13:17] = False                          # root smooth-XZ / heading: crop re-base uses the rig's stats
        eff = sd_ref.astype(np.float64) + _STD_FLOOR
        z_mu = np.clip(rng.normal(0.0, 1.0, mu0.shape), -STATS_Z_CLIP, STATS_Z_CLIP) * cfg.stats_shift
        z_sd = np.clip(rng.normal(0.0, 1.0, sd0.shape), -STATS_Z_CLIP, STATS_Z_CLIP) * cfg.stats_logsd
        mu64 = np.where(cells, mu_ref.astype(np.float64) + z_mu * eff, mu_ref.astype(np.float64))
        sd64 = np.where(cells, eff * np.exp(z_sd) - _STD_FLOOR, sd_ref.astype(np.float64))
        mu_n, sd_n = mu64.astype(np.float32), sd64.astype(np.float32)
        if not (np.isfinite(mu_n).all() and np.isfinite(sd_n).all() and np.all(sd_n.astype(np.float64) + _STD_FLOOR > 0)):
            raise AssertionError("statistics perturbation produced a non-finite mean or a non-positive effective scale")
    # ---- 3. descriptions ----
    sem_drop = bool(cfg.sem_drop_p > 0 and rng.random() < cfg.sem_drop_p)
    unimate_rot = op in ("add", "pool") and bool(reencode)          # a pool draw that pooled nothing leaves the sample as is
    return SkeletonTransform(keep=keep, src=src, synth_pc=synth_pc, parents=new_par, offsets=new_off, R_rest=R_new,
                             R_rest_old=R_old, P_rest=P_new, Q=Q, rot_rows=rot_rows,
                             mu=mu_n.astype(np.float32), sd=sd_n.astype(np.float32), mu0=mu0, sd0=sd0,
                             channel_valid=cvk, sem_noise=float(cfg.sem_noise), sem_drop=sem_drop,
                             reencode=bool(reencode), fps=float(fps), op=op, unimate_rot=unimate_rot,
                             recomposed=(recomposed.copy() if unimate_rot else None), par0=(par.copy() if unimate_rot else None),
                             mu_all=(np.asarray(mu, dtype=np.float32).copy() if unimate_rot else None),
                             sd_all=(np.asarray(sd, dtype=np.float32).copy() if unimate_rot else None))


def _rotate_rest_deltas(raw, tr, rows_of):
    """In place: on the rows of `raw` [T,J',17] listed by rows_of (J' indices whose Q is not the identity and whose
    rotation cells are valid), replace the six-vector by delta Q^T; degenerate cells keep their served six-vector."""
    rows = np.asarray([r for r in rows_of if tr.rot_rows[r] and not np.all(np.isclose(tr.Q[r], np.eye(3), atol=0.0))])
    if rows.size == 0:
        return
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


def _refuse_degenerate(d6: np.ndarray, joint_ids) -> None:
    """[T,n,6] six-vectors: ValueError naming the joints whose Gram-Schmidt norms vanish (the decoder's GT eps). A degenerate
    six-vector (zero or parallel columns) has no rotation the emitted representation could reconstruct the FK'd positions
    from, so it is refused rather than replaced (codex v2 r1 P2-4; the corpus has none -- runs/_aug_dev/excluded_cells_scan.json)."""
    a1, a2 = d6[..., :3], d6[..., 3:]
    n1 = np.linalg.norm(a1, axis=-1)
    b1 = a1 / np.maximum(n1[..., None], 1e-12)
    n2 = np.linalg.norm(a2 - np.sum(b1 * a2, axis=-1, keepdims=True) * b1, axis=-1)
    bad = ~((n1 > 1e-6) & (n2 > 1e-6))
    if bad.any():
        t_bad, r_bad = np.where(bad)
        raise ValueError(f"re-encoding refused: degenerate rotation six-vector on served joint(s) "
                         f"{sorted(set(np.asarray(joint_ids)[r_bad].tolist()))[:8]} at frame(s) {sorted(set(t_bad.tolist()))[:8]}")


def apply_motion(x18: np.ndarray, tr: SkeletonTransform) -> np.ndarray:
    """[T,J,18] (or [J,18]) served-normalised motion -> [T,J',18] under the transform (apply_motion_with_contact without
    the pre-pruning contact flags)."""
    return apply_motion_with_contact(x18, tr)[0]


def apply_motion_with_contact(x18: np.ndarray, tr: SkeletonTransform):
    """[T,J,18] (or [J,18]) served-normalised motion -> ([T,J',18] under the transform, the served contact flags [T,J'] bool
    BEFORE the one_of pruning (synthetic rows False) or None when nothing is pruned -- the dataset counts the window's
    pre-pruning pairs from them for the foot-lock term's denominator, ONE_OF_LOCK_DEN).
    Plane 17 (heading-valid flag) is copied (a synthetic joint takes its parent's); channels 0:17 are de-normalised
    with the rig's statistics, rotated (rest deltas -> delta Q^T on rows with valid rotation cells), re-encoded by FK
    when the transform changed the body or the tree, and re-normalised with the perturbed statistics (unperturbed
    on cells outside channel_valid, which cfm_loss projects out anyway)."""
    squeeze = x18.ndim == 2
    x = x18[None] if squeeze else x18
    contact_before = None
    xk = np.asarray(x, dtype=np.float32)[:, tr.keep]                                     # served kept originals
    T = xk.shape[0]; Jn = tr.n_joints; rows = tr.served_rows
    mu0k, sd0k = tr.mu0[rows], tr.sd0[rows]
    raw_k = xk[..., :17].astype(np.float64) * (sd0k[None] + _STD_FLOOR) + mu0k[None]
    raw = np.zeros((T, Jn, 17), dtype=np.float64)
    raw[:, rows] = raw_k
    p17 = np.zeros((T, Jn), dtype=np.float32)
    p17[:, rows] = xk[..., 17]
    if tr.reencode:
        if tr.unimate_rot:
            # UniMate's rotation semantics (mode one_of, pool / add). Every ORIGINAL joint's served rest-delta is read (a
            # pooled joint's included, through the rig's statistics) and turned into its parent-relative delta
            # Pi_k = Delta_par(k)^T Delta_k (Pi_root = Delta_root; identity at rest). The served rows are recomposed through
            # the NEW tree in FK order, Delta'_row = Delta'_newparent Pi_src(row): a pooled joint's articulation is skipped
            # (UniMate's collapse deletes the local rotation) and the added joint takes its parent's own parent-relative
            # delta (UniMate's insertion copies the parent's rotation slot, which in their parent-local layout duplicates
            # the parent's local rotation at the new joint). With identity rest rotations this is exactly UniMate's
            # local-rotation FK (codex 2026-09-15 unimate r2 P1-1 / P1-2).
            d6_all = (np.asarray(x, dtype=np.float32)[..., 3:9].astype(np.float64) * (tr.sd_all[None, :, 3:9] + _STD_FLOOR)
                      + tr.mu_all[None, :, 3:9])
            _refuse_degenerate(d6_all, np.arange(d6_all.shape[1]))
            D = decode_column_cont6d(d6_all, strict=True)                                  # [T,J,3,3] every original joint
            Pi = np.empty_like(D)
            for k in range(D.shape[1]):
                pk = int(tr.par0[k])
                Pi[:, k] = D[:, k] if pk < 0 else np.matmul(np.swapaxes(D[:, pk], -1, -2), D[:, k])
            Dn = np.empty((T, Jn, 3, 3), dtype=np.float64)
            for row in range(Jn):                                                          # FK order: parents precede children
                k = int(tr.src[row])
                Pi_row = Pi[:, k] if k >= 0 else Pi[:, int(tr.synth_pc[row, 0])]
                q = int(tr.parents[row])
                Dn[:, row] = Pi_row if q < 0 else np.matmul(Dn[:, q], Pi_row)
            raw[..., 3:9] = encode_column_cont6d(Dn)                                       # every served row, synthetic included
            G = np.matmul(Dn, tr.R_rest_old[None])
        else:
            # global rotations of the served originals from their served deltas
            d6 = raw_k[..., 3:9]
            _refuse_degenerate(d6, tr.keep)
            delta = decode_column_cont6d(d6, strict=True)
            G = np.zeros((T, Jn, 3, 3), dtype=np.float64)
            G[:, rows] = np.matmul(delta, tr.R_rest_old[rows][None])
            for j in np.where(tr.src < 0)[0]:                                              # synthetic: rigid with its parent
                G[:, j] = G[:, tr.parents[j]]
        track = raw[:, 0, 13:15]                                                           # the root's smooth XZ track
        root_world = raw[:, 0, 0:3] + np.stack([track[:, 0], np.zeros(T), track[:, 1]], axis=1)
        pos = _fk(tr.parents, root_world, G, tr.offsets)
        raw[..., 0:3] = pos - np.stack([track[:, 0], np.zeros(T), track[:, 1]], axis=1)[:, None]
        vel = np.zeros_like(pos)
        if T >= 2:
            vel[:-1] = (pos[1:] - pos[:-1]) * tr.fps
            vel[-1] = vel[-2]
        raw[..., 9:12] = vel
        if tr.unimate_rot and T >= 2:
            # mode one_of, pool / add: a served contact flag on a RECOMPOSED row survives only where the joint's re-encoded
            # forward displacement exceeds its original one by at most ONE_OF_CONTACT_ABS mean bone lengths; a joint the
            # recomposition swung is no longer a locked target of the foot-lock term (codex 2026-09-15 unimate r5 P1: a stale
            # flag under a re-articulated chain demanded motion and stillness at once). The flag at frame t governs the pair
            # (t, t+1), so it is judged on the forward displacement; the last frame keeps its served flag; every other row,
            # and every row of the scale op, keeps its served flags (codex r6: pruning the barely-moving planted feet of a
            # scaled skeleton raised the term's mean). Joint mode keeps every served flag (the trained arms).
            rr = np.where((tr.src >= 0) & tr.recomposed)[0]                                             # recomposed originals
            kk = np.searchsorted(rows, rr)                                                              # their kept-order index
            w0 = raw_k[:, kk, 0:3] + np.stack([track[:, 0], np.zeros(T), track[:, 1]], axis=1)[:, None]  # originals, world
            d_old = np.linalg.norm(w0[1:] - w0[:-1], axis=-1)                                           # [T-1,r]
            d_new = np.linalg.norm(pos[1:, rr] - pos[:-1, rr], axis=-1)
            bl = float(np.linalg.norm(tr.offsets[1:], axis=-1).mean()) if Jn > 1 else 1.0
            contact_before = raw[..., 12] > 0.5                                                       # synthetic rows: still 0
            con = raw[:, rr, 12] > 0.5
            con[:-1] &= d_new <= d_old + ONE_OF_CONTACT_ABS * bl
            raw[:, rr, 12] = con.astype(np.float64)
    _rotate_rest_deltas(raw, tr, rows)
    for j in np.where(tr.src < 0)[0]:                                                      # synthetic rows: the parent's deltas
        if not tr.unimate_rot:                                                             # (one_of: recomposed above)
            raw[:, j, 3:9] = raw[:, tr.parents[j], 3:9]
        raw[:, j, 12] = 0.0
        p17[:, j] = p17[:, tr.parents[j]]
    out = np.empty((T, Jn, 18), dtype=np.float32)
    out[..., :17] = ((raw - tr.mu[None]) / (tr.sd[None] + _STD_FLOOR)).astype(np.float32)
    out[..., 17] = p17
    if squeeze:
        return out[0], (contact_before[0] if contact_before is not None else None)
    return out, contact_before


def apply_semantics(sem: np.ndarray, tr: SkeletonTransform, rng) -> np.ndarray:
    sem = np.asarray(sem, dtype=np.float32)
    s = np.empty((tr.n_joints, sem.shape[1]), dtype=np.float32)
    s[tr.served_rows] = sem[tr.keep]
    for j in np.where(tr.src < 0)[0]:                                                      # synthetic: mean of parent and child
        pj, cj = tr.synth_pc[j]
        s[j] = 0.5 * (sem[pj] + sem[cj])
    if tr.sem_drop:
        return np.zeros_like(s)
    if tr.sem_noise > 0:
        rms = np.sqrt(np.mean(s ** 2, axis=1, keepdims=True))
        s += rng.normal(0.0, 1.0, s.shape).astype(np.float32) * (tr.sem_noise * rms).astype(np.float32)
    return s
