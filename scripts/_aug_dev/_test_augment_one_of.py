"""AugConfig.mode "one_of" (UniMate's rule: ONE of add / remove / pool / scale per augmented sample, UniMate's rates) and the
joint-mode parity of the module against its pre-change bytes (src/data/ktjd17_augment.py.bak_20260915). Run inside an
allocation: `srun ... /usr/bin/env python scripts/_aug_dev/_test_augment_one_of.py`."""
import sys, json, os, subprocess, importlib.util
from collections import Counter
from dataclasses import asdict, fields
import numpy as np, torch
sys.path.insert(0, ".")
import src.data.ktjd17_augment as AUGM
from src.data.ktjd17_incontext import Ktjd17Base, ktjd17_split_names
from src.data.incontext_pairs import InContextPairs
from src.data.ktjd17_augment import (AugConfig, make_transform, apply_motion, apply_motion_with_contact, apply_semantics, ONE_OF, ONE_OF_OPS,
                                     ONE_OF_DEVIATIONS, ONE_OF_ROTATION_RULE, ONE_OF_CONTACT, ONE_OF_LOCK_DEN, AUG_VERSION_ONE_OF,
                                     _weighted_pick, _ellipsoid_displacement)
from src.data.incontext_pairs import collate
from src.models.v2.fk_torch import ktjd_dynamics_losses, FK_SCALE_FRAC
from src.data.anytop_dataset import _STD_FLOOR
from src.data.ktjd17.codec import decode_column_cont6d, encode_column_cont6d, fk_from_global_rotations
from src.models.v2.fk_torch import fk_ktjd_consistency_loss

PREV = "src/data/ktjd17_augment.py.bak_20260915"
from importlib.machinery import SourceFileLoader
_ld = SourceFileLoader("ktjd17_augment_prev", PREV)                     # a .bak suffix needs an explicit loader
prev = importlib.util.module_from_spec(importlib.util.spec_from_loader("ktjd17_augment_prev", _ld))
sys.modules["ktjd17_augment_prev"] = prev                                # dataclasses resolve the module by name
_ld.exec_module(prev)

# UniMate's rates and constants as read from THEIR code -- outside_docs/UniMate/unimate/dataset/mixture/dataset.py:963-971
# (rates), augmentations.py:47-63 (bone-length weighting alpha), 203-229 / 315-393 (alpha, ellipsoid sigma / lateral),
# 167-202 (leaf removal cap 3), 394-429 (pooling min 1) -- so the module's ONE_OF is asserted against them, not read from them
UNI = dict(remove_rate=[0.05, 0.15], remove_cap=3, pool_rate=[0.1, 0.3], pool_min=1, select_alpha=0.5,
           add_alpha=[0.3, 0.7], add_sigma=0.5, add_lateral=0.5)

ROOT, CUT = "dataset/ktjd17_pzh312_noik_v2", "configs/heldout20_v1_exclusions.json"       # the held-out arms' training cut
KW = dict(caption_emb_cache="data/noik_caption_llm2vec_v1", joint_semantics="data/joint_semantics_llm2vec_pzh312_v1.npz",
          texts_json="data/noik_pzh312_motion_texts_v1.json", percell_stats="data/noik_norm_stats_v2.npz",
          exclude_clips=CUT, random_caption=False)
base = Ktjd17Base(ROOT, normalization="rest", **KW)
FPS = 30.0
TAUS = tuple(float(json.load(open(f"{ROOT}/schema.json"))["contact"][k]) for k in ("tau_h", "tau_v"))   # the schema's contact rule,
CR1 = (1.0,) + TAUS                                                                                     # used to FILL a fixture's flags
fails = []
def check(cond, msg):
    if not cond: fails.append(msg); print("FAIL:", msg)

def load(bi):
    it = base[int(bi)]
    J0, T = int(it["num_joints"]), int(it["num_frames"])
    x = np.asarray(it["anytop_x"])[:J0, :, :T].transpose(2, 0, 1)
    mu0, sd0 = np.asarray(it["anytop_mean"])[:J0, :17], np.asarray(it["anytop_std"])[:J0, :17]
    cv0 = base.static_masks(it["object_type"])["channel_valid"][:J0]
    raw0 = x[..., :17].astype(np.float64) * (sd0 + _STD_FLOOR) + mu0
    contact = (raw0[..., 12] > 0.5).any(0)
    sk = base.skeleton(it["object_type"])
    geom = dict(parents=np.asarray(sk["parents"])[:J0], P_rest_global=np.asarray(sk["P_rest_global"], dtype=np.float64)[:J0],
                R_rest_global=np.asarray(sk["R_rest_global"], dtype=np.float64)[:J0],
                offset_parent_local=np.asarray(sk["offset_parent_local"], dtype=np.float64)[:J0])
    return it, J0, T, x, mu0, sd0, cv0, raw0, contact, geom

def world(raw):
    w = raw[..., 0:3].copy(); w[..., 0] += raw[:, 0:1, 13]; w[..., 2] += raw[:, 0:1, 14]; return w
def G_of(raw, R_rest):
    d6 = raw[..., 3:9]; a1, a2 = d6[..., :3], d6[..., 3:]
    n1 = np.linalg.norm(a1, axis=-1); b1 = a1 / np.maximum(n1[..., None], 1e-12)
    n2 = np.linalg.norm(a2 - np.sum(b1 * a2, axis=-1, keepdims=True) * b1, axis=-1)
    ok = (n1 > 1e-6) & (n2 > 1e-6)
    D = np.tile(np.eye(3), d6.shape[:2] + (1, 1))
    if ok.any(): D[ok] = decode_column_cont6d(d6[ok], strict=True)
    return D @ R_rest[None]
def denorm(xn, tr):
    return xn[..., :17].astype(np.float64) * (tr.sd[None] + _STD_FLOOR) + tr.mu[None]
def official_fk_gap(xn, tr):
    x = torch.from_numpy(np.ascontiguousarray(xn[..., :17])).float().clone()
    cv = torch.from_numpy(tr.channel_valid); x[:, ~cv] = 0.0
    T, Jn = x.shape[:2]
    _, diag = fk_ktjd_consistency_loss(x[None], torch.from_numpy(tr.mu)[None].float(), torch.from_numpy(tr.sd)[None].float(), _STD_FLOOR,
                                       torch.from_numpy(tr.parents.astype(np.int64))[None], torch.from_numpy(tr.offsets)[None].float(),
                                       torch.from_numpy(tr.R_rest)[None].float(), torch.tensor([Jn]), torch.ones(1, T, dtype=torch.bool), want_diag=True)
    return float(diag)
def synth(parents):
    par = np.asarray(parents); J = len(par)
    O = np.zeros((J, 3)); O[1:, 0] = 1.0                                  # unit bones along x, identity rest rotations
    P = np.zeros((J, 3))
    for j in range(1, J): P[j] = P[par[j]] + O[j]
    return dict(parents=par, P_rest_global=P, R_rest_global=np.tile(np.eye(3), (J, 1, 1)), offset_parent_local=O,
                channel_valid=np.ones((J, 17), bool), mu=np.zeros((J, 17), np.float32), sd=np.ones((J, 17), np.float32))
def ellipsoid_ref(r, bone, sigma=0.5, lateral_ratio=0.5):
    """UniMate's sample_ellipsoid_gaussian, re-implemented from outside_docs/UniMate/.../augmentations.py:274 on numpy's Generator"""
    L = float(np.linalg.norm(bone))
    if L < 1e-8: return np.zeros(3)
    while True:
        d = r.uniform(-1, 1, size=3); n = float(np.linalg.norm(d))
        if 0 < n <= 1: d = d / n; break
    while True:
        rad = abs(float(r.normal(0, sigma)))
        if rad <= 1: break
    sphere = rad * d
    scaled = sphere * np.array([L, L * lateral_ratio, L * lateral_ratio])
    e1 = bone / L; ref = np.array([0.0, 1.0, 0.0])
    if abs(np.dot(e1, ref)) > 0.9: ref = np.array([1.0, 0.0, 0.0])
    e2 = np.cross(e1, ref); e2 = e2 / np.linalg.norm(e2); e3 = np.cross(e1, e2)
    return np.column_stack([e1, e2, e3]) @ scaled
def expected_G(raw0, R0, par0, tr):
    """UniMate's local-rotation FK written in our terms: Pi_k = Delta_par(k)^T Delta_k from every ORIGINAL joint's served
    delta, recomposed through the NEW tree (a pooled joint's Pi skipped; the synthetic joint gets its parent's Pi);
    returns the expected global rotations [T,J',3,3] of every served row"""
    D = decode_column_cont6d(raw0[..., 3:9], strict=True)
    Pi = np.empty_like(D)
    for k in range(D.shape[1]):
        pk = int(par0[k]); Pi[:, k] = D[:, k] if pk < 0 else np.swapaxes(D[:, pk], -1, -2) @ D[:, k]
    Dn = np.empty((D.shape[0], tr.n_joints, 3, 3))
    for row in range(tr.n_joints):
        k = int(tr.src[row]); q = int(tr.parents[row])
        P_row = Pi[:, k] if k >= 0 else Pi[:, int(tr.synth_pc[row, 0])]
        Dn[:, row] = P_row if q < 0 else Dn[:, q] @ P_row
    return Dn @ tr.R_rest_old[None]
def Rz(deg):
    c, s_ = np.cos(np.radians(deg)), np.sin(np.radians(deg)); return np.array([[c, -s_, 0], [s_, c, 0], [0, 0, 1]])
def fixture_clip(g, G, T=2):
    """[T,J,18] served clip of the synth fixture (mu 0, sd 1): global rotations G [J,3,3] (identity rest -> deltas = G),
    positions by FK, zero velocity / contact, root track 0, heading (1,0), plane 17 set"""
    J = len(g["parents"]); pos = np.zeros((J, 3))
    for j in range(1, J): pos[j] = pos[g["parents"][j]] + G[g["parents"][j]] @ g["offset_parent_local"][j]
    raw = np.zeros((J, 17)); raw[:, 0:3] = pos; raw[:, 3:9] = encode_column_cont6d(G); raw[0, 15] = 1.0
    x = np.zeros((T, J, 18), np.float32); x[:, :, :17] = raw[None]; x[:, :, 17] = 1.0
    return x, pos
def unimate_reference(par0, P0, D0, tr, root_world):
    """UniMate's own mechanics, literally, as an oracle: their slot layout (slot[j] = the local rotation of parents[j],
    slot[0] = the root's; augmentations.py:73-90 docstring, motion_utils.py:232-253), their collapse (slot[c] = slot[j],
    offsets summed, c re-parented, row j deleted; descending index), their insertion (the new slot is a copy of the child's,
    offsets split, parents shifted; augmentations.py:315-393), their slot->bvh conversion (a leaf's own rotation is never
    stored -> identity) and their FK (G_j = G_p L_j, pos_j = pos_p + G_p off_j) on offsets in THEIR identity-rest global
    frame (P_k - P_parent). Only the translation layer is ours: a joint's local rotation is its parent-relative rest-delta
    Delta_parent^T Delta_k (their locals with identity rest), and our parent-frame offset maps to their frame through the
    parent's rest rotation. Returns positions [T,J',3], global rotations [T,J',3,3], parents, row->original map, has-child."""
    T, J = D0.shape[:2]
    Pi = np.empty_like(D0)
    for k in range(J):
        pk = int(par0[k]); Pi[:, k] = D0[:, k] if pk < 0 else np.swapaxes(D0[:, pk], -1, -2) @ D0[:, k]
    slots = [Pi[:, 0].copy()] + [Pi[:, int(par0[j])].copy() for j in range(1, J)]
    par = [int(q) for q in par0]
    off = [np.zeros(3)] + [P0[k] - P0[int(par0[k])] for k in range(1, J)]
    idx = list(range(J))
    if tr.op == "pool":
        for j in sorted(set(range(J)) - set(tr.keep.tolist()), reverse=True):
            r = idx.index(j); kids = [i for i, q in enumerate(par) if q == r]; assert len(kids) == 1, (j, kids)
            c, q = kids[0], par[r]
            slots[c] = slots[r]; off[c] = off[r] + off[c]; par[c] = q
            del slots[r]; del off[r]; del idx[r]
            par = [v - 1 if v > r else v for v in par[:r] + par[r + 1:]]
    elif tr.op == "add":
        s = int(np.where(tr.src < 0)[0][0]); r = int(tr.src[s + 1]); pi = par[r]
        off_new = tr.R_rest_old[s] @ tr.offsets[s]
        off_rem = off[r] - off_new
        slots.insert(r, slots[r].copy()); off.insert(r, off_new); off[r + 1] = off_rem; idx.insert(r, -1)
        newp = [v + 1 if v >= r else v for v in par]; par = newp[:r] + [pi] + newp[r:]; par[r + 1] = r
    Jn = len(par)
    bvh = [np.tile(np.eye(3), (T, 1, 1)) for _ in range(Jn)]
    for j in range(1, Jn): bvh[par[j]] = slots[j]
    G = [None] * Jn; pos = np.empty((T, Jn, 3)); G[0] = bvh[0]; pos[:, 0] = root_world
    for j in range(1, Jn):
        q = par[j]; G[j] = G[q] @ bvh[j]; pos[:, j] = pos[:, q] + np.einsum("tij,j->ti", G[q], off[j])
    has_child = np.array([any(par[i] == j for i in range(Jn)) for j in range(Jn)])
    return pos, np.stack(G, axis=1), par, idx, has_child
def compare_with_unimate(tag, geom, raw0, tr, xn):
    """our served output vs the UniMate-literal reference: the tree, every row's world position, every non-leaf row's rest-delta"""
    D0 = decode_column_cont6d(raw0[..., 3:9], strict=True)
    rawn = denorm(xn, tr); wn = world(rawn); bl = np.linalg.norm(tr.offsets[1:], axis=-1).mean()
    posU, GU, parU, idxU, hasc = unimate_reference(geom["parents"], geom["P_rest_global"], D0, tr, world(raw0)[:, 0])
    check(parU == tr.parents.tolist() and idxU == tr.src.tolist(), tag + " UniMate slot mechanics: same tree and row map")
    e_pos = np.abs(posU - wn).max() / bl
    check(e_pos < 1e-4, tag + f" UniMate-literal FK positions == ours on every row (max {e_pos:.1e} bl)")
    Dn = G_of(rawn, tr.R_rest) @ np.swapaxes(tr.R_rest, -1, -2)[None]
    e_rot = np.abs(GU[:, hasc] - Dn[:, hasc]).max()
    check(e_rot < 1e-5, tag + f" UniMate-literal rotations == our rest-deltas on every non-leaf row (max {e_rot:.1e})")
def lock_term(xn, mean, std, offsets, contact_on=None, lock_denominator=None):
    """the trainer's foot-lock term (fk_torch.ktjd_dynamics_losses) with the prediction equal to the target: what the target
    itself demands; contact_on None = the served contact channel, else an explicit [T,J] bool; lock_denominator = the
    pre-pruning pair count the augmented sample carries (None = the surviving pairs)"""
    xt = torch.from_numpy(np.ascontiguousarray(xn[..., :17])).float()[None]
    mean_t, std_t = torch.from_numpy(np.asarray(mean))[None].float(), torch.from_numpy(np.asarray(std))[None].float()
    con = ((xt[..., 12] * (std_t[:, None, :, 12] + _STD_FLOOR) + mean_t[:, None, :, 12]) > 0.5) if contact_on is None else torch.from_numpy(np.asarray(contact_on))[None]
    T, Jn = xt.shape[1:3]
    den = None if lock_denominator is None else torch.tensor([float(lock_denominator)])
    _, lock, _, _ = ktjd_dynamics_losses(xt, xt, mean_t, std_t, _STD_FLOOR, torch.from_numpy(np.asarray(offsets))[None].float(),
                                         torch.tensor([Jn]), torch.ones(1, T, dtype=torch.bool), contact_on=con, want_diag=False,
                                         lock_denominator=den)
    return float(lock)
def pairs_of(cb):
    return int((cb[1:] & cb[:-1]).sum())
def lock_bound(raw0, tr, xn, A=0.05):
    """the foot-lock term the pruned target can demand at most: every retained pair of an unmoved row moves as before, every
    retained pair of a recomposed row at most A mean bone lengths farther, all measured in the loss's own scale on the new bones"""
    rawn = denorm(xn, tr); rows = tr.served_rows
    con = rawn[:, rows, 12] > 0.5; pair = con[1:] & con[:-1]
    w0k = world(raw0)[:, tr.keep]; d_old = np.linalg.norm(w0k[1:] - w0k[:-1], axis=-1)
    bl_new = float(np.linalg.norm(tr.offsets[1:], axis=-1).mean())
    scale_new = FK_SCALE_FRAC * (bl_new + 1e-3)
    allow = A * bl_new * (tr.recomposed[rows] if tr.recomposed is not None else np.zeros(len(rows), bool))
    per_pair = ((d_old + allow[None]) / scale_new) ** 2
    return float(per_pair[pair].mean()) if pair.any() else 0.0
def rand_rot(r):
    q = r.normal(size=4); q /= np.linalg.norm(q); w, x_, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x_ * y - z * w), 2 * (x_ * z + y * w)],
                     [2 * (x_ * y + z * w), 1 - 2 * (x_ * x_ + z * z), 2 * (y * z - x_ * w)],
                     [2 * (x_ * z - y * w), 2 * (y * z + x_ * w), 1 - 2 * (x_ * x_ + y * y)]])
def synth_rot(parents, r):
    """a fixture with random rest positions, random NON-identity rest rotations, parent-frame offsets consistent with them"""
    par = np.asarray(parents); J = len(par)
    P = r.normal(size=(J, 3)) * 2.0; R = np.stack([rand_rot(r) for _ in range(J)]); O = np.zeros((J, 3))
    for j in range(1, J): O[j] = R[par[j]].T @ (P[j] - P[par[j]])
    return dict(parents=par, P_rest_global=P, R_rest_global=R, offset_parent_local=O,
                channel_valid=np.ones((J, 17), bool), mu=np.zeros((J, 17), np.float32), sd=np.ones((J, 17), np.float32))
def fixture_clip_rot(g, D, root_world, contact_rule=None):
    """[T,J,18] served clip (mu 0, sd 1) of a synth_rot fixture with per-frame rest-deltas D [T,J,3,3] (root included) and
    root world positions [T,3]; positions by our FK, the root track = the root's XZ"""
    par, R, O = g["parents"], g["R_rest_global"], g["offset_parent_local"]; T, J = D.shape[:2]
    G = D @ R[None]; pos = np.empty((T, J, 3)); pos[:, 0] = root_world
    for j in range(1, J): pos[:, j] = pos[:, par[j]] + np.einsum("tij,j->ti", G[:, par[j]], O[j])
    raw = np.zeros((T, J, 17)); track = pos[:, 0][:, [0, 2]]
    raw[..., 0:3] = pos - np.stack([track[:, 0], np.zeros(T), track[:, 1]], 1)[:, None]
    raw[..., 3:9] = encode_column_cont6d(D); raw[:, 0, 13:15] = track; raw[:, 0, 15] = 1.0
    if T >= 2: raw[:-1, :, 9:12] = (pos[1:] - pos[:-1]) * FPS; raw[-1, :, 9:12] = raw[-2, :, 9:12]
    if contact_rule is not None:                                                     # the corpus rule on the fixture's own motion
        s_rig, tau_h, tau_v = contact_rule
        con = (pos[..., 1] / s_rig <= tau_h) & (np.linalg.norm(raw[..., 9:12], axis=-1) / s_rig <= tau_v)
        if T >= 2: con[-1] = con[-2]
        raw[..., 12] = con
    x = np.zeros((T, J, 18), np.float32); x[..., :17] = (raw - g["mu"][None]) / (g["sd"][None] + _STD_FLOOR); x[..., 17] = 1.0
    return x, raw
def same_transform(t_new, t_old):
    return all(np.array_equal(np.asarray(getattr(t_new, f.name)), np.asarray(getattr(t_old, f.name))) for f in fields(t_old))

# ---- 1. protocol records: v1 / v2 unchanged key for key, one_of carries UniMate's constants and the deviations, survives JSON ----
v1 = dict(p=0.5, drop_max_frac=0.3, drop_mode="tips", rest_deg=15.0, sem_noise=0.1, sem_drop_p=0.05, stats_logsd=0.2, stats_shift=0.3)
v2 = dict(v1, bone_scale=0.1, pool_frac=0.3, add_p=0.5)
for name, kw in (("v1", v1), ("v2", v2)):
    check(AugConfig(**kw).protocol() == prev.AugConfig(**kw).protocol(), f"protocol {name} unchanged")
    check("mode" not in AugConfig(**kw).protocol(), f"protocol {name} carries no mode key")
check(AugConfig().protocol() is None and prev.AugConfig().protocol() is None, "off -> protocol None")
one = AugConfig(p=0.8, bone_scale=0.1, drop_mode="tips", mode="one_of")
po = one.protocol()
check(po["version"] == AUG_VERSION_ONE_OF and po["mode"] == "one_of" and po["ops"] == list(ONE_OF_OPS) and po["p"] == 0.8
      and po["bone_scale"] == 0.1 and po["drop_mode"] == "tips" and po["deviations"] == list(ONE_OF_DEVIATIONS), "one_of protocol fields")
check(ONE_OF == UNI, f"ONE_OF equals UniMate's rates as read from their code ({ONE_OF} vs {UNI})")
check(all(po[k] == v for k, v in UNI.items()), "one_of protocol carries UniMate's constants")
check(po["rotation_rule"] == ONE_OF_ROTATION_RULE == "unimate_local_slots"
      and po["contact"] == ONE_OF_CONTACT == "pruned_on_recomposed_rows_where_displacement_grows" and po["contact_abs_bl"] == 0.05
      and po["lock_denominator"] == ONE_OF_LOCK_DEN == "pre_pruning_pairs" and po["reentry"] == "rotation_and_contact_cells_of_recomposed_rows"
      and "contact_slack" not in po and "v3.6" in po["version"],
      "one_of protocol versions the rotation, re-entry, contact and lock-denominator rules")
check(json.loads(json.dumps(po)) == po, "one_of protocol survives the JSON round trip of a calibration artifact")
check(po != json.loads(json.dumps(AugConfig(p=0.5, bone_scale=0.1, drop_mode="tips", mode="one_of").protocol())), "one_of protocol distinguishes p")
check(po != AugConfig(**dict(v2, p=0.8)).protocol(), "one_of protocol differs from the joint record")

# ---- 2. refusals ----
def refuses(**kw):
    try: AugConfig(**kw); return False
    except ValueError: return True
check(refuses(p=0.8, bone_scale=0.1, mode="one_of"), "one_of needs drop_mode tips (default 'any' refused)")
for bad in (dict(rest_deg=5.0), dict(drop_max_frac=0.1), dict(pool_frac=0.1), dict(add_p=0.5), dict(sem_noise=0.1),
            dict(sem_drop_p=0.05), dict(stats_logsd=0.1), dict(stats_shift=0.1)):
    check(refuses(p=0.8, bone_scale=0.1, drop_mode="tips", mode="one_of", **bad), f"one_of refuses {bad}")
check(refuses(p=0.8, bone_scale=0.0, drop_mode="tips", mode="one_of"), "one_of needs bone_scale > 0")
check(refuses(p=0.8, bone_scale=0.1, drop_mode="tips", mode="unimate"), "unknown mode refused")
check(not refuses(p=0.0, mode="joint") and not refuses(**v2), "joint configs still construct")

# ---- 3. joint-mode parity against the pre-change bytes: transform fields, RNG state, motion and semantics ----
rng = np.random.default_rng(11)
idxs = rng.choice(len(base), size=25, replace=False)
jcfgs = {"all_v2": AugConfig(**dict(v2, p=1.0)), "scale": AugConfig(p=1.0, bone_scale=0.1), "add": AugConfig(p=1.0, add_p=1.0),
         "pool": AugConfig(p=1.0, pool_frac=1.0), "v1_tips": AugConfig(**dict(v1, p=1.0)),
         "drop_any": AugConfig(p=1.0, drop_max_frac=1.0, drop_mode="any"), "stats_only": AugConfig(p=1.0, stats_logsd=0.2, stats_shift=0.3),
         "rest_only": AugConfig(p=1.0, rest_deg=10.0), "sem_only": AugConfig(p=1.0, sem_noise=0.1, sem_drop_p=0.5)}
def prev_cfg(cfg): return prev.AugConfig(**{k: v for k, v in asdict(cfg).items() if k != "mode" and k in {f.name for f in fields(prev.AugConfig)}})   # rest_p (R2) postdates the .bak
n_par = 0
for n, bi in enumerate(idxs):
    it, J0, T, x, mu0, sd0, cv0, raw0, contact, geom = load(bi)
    sem = np.asarray(it["joint_semantics"])[:J0]
    for name, cfg in jcfgs.items():
        seed = 500 + n
        r_new, r_old = np.random.default_rng(seed), np.random.default_rng(seed)
        t_new = make_transform(r_new, cfg, **geom, channel_valid=cv0, mu=mu0, sd=sd0, contact_joints=contact, fps=FPS, rest_norm=True)
        t_old = prev.make_transform(r_old, prev_cfg(cfg), **geom, channel_valid=cv0, mu=mu0, sd=sd0, contact_joints=contact, fps=FPS, rest_norm=True)
        tag = f"[parity/{name}/{it['object_type']}]"
        check(same_transform(t_new, t_old) and t_new.op is None, tag + " transform fields identical, op None")
        check(r_new.bit_generator.state == r_old.bit_generator.state, tag + " RNG state identical after the transform")
        check(np.array_equal(apply_motion(x, t_new), prev.apply_motion(x, t_old)), tag + " apply_motion byte-identical")
        check(np.array_equal(apply_semantics(sem, t_new, np.random.default_rng(seed)), prev.apply_semantics(sem, t_old, np.random.default_rng(seed))),
              tag + " apply_semantics byte-identical")
        n_par += 1
# small fixtures: a root-only rig, a chain, a branch; every stage enabled / disabled, 20 seeds each
fixtures = {"root_only": (synth([-1]), np.array([True])), "chain3": (synth([-1, 0, 1]), np.array([True, False, False])),
            "branch5": (synth([-1, 0, 1, 1, 2]), np.array([True, False, False, True, True]))}
n_fix = 0
for fname, (g, prot) in fixtures.items():
    for name, cfg in jcfgs.items():
        for seed in range(20):
            r_new, r_old = np.random.default_rng(seed), np.random.default_rng(seed)
            t_new = make_transform(r_new, cfg, **g, contact_joints=prot)
            t_old = prev.make_transform(r_old, prev_cfg(cfg), **g, contact_joints=prot)
            check(same_transform(t_new, t_old) and r_new.bit_generator.state == r_old.bit_generator.state, f"[parity/{fname}/{name}/seed{seed}] fields + RNG state")
            n_fix += 1
print(f"[parity] {n_par} real-rig and {n_fix} fixture transform pairs compared")

# ---- 4. the helpers against independent expectations ----
# 4a. bone-length-weighted selection: two candidate bones of length 1 and 4 -> weights 1 : 0.5 -> P(short) = 2/3
g2 = synth([-1, 0, 0]); off = g2["offset_parent_local"].copy(); off[2] *= 4.0
r = np.random.default_rng(3); short = sum(_weighted_pick(r, np.array([1, 2]), off, 1)[0] == 1 for _ in range(6000)) / 6000
check(abs(short - 2 / 3) < 0.03, f"_weighted_pick favours the shorter bone with weight L^-0.5 (P(short) {short:.3f}, expected 0.667)")
check(sorted(_weighted_pick(np.random.default_rng(0), np.array([1, 2]), off, 5)) == [1, 2] and _weighted_pick(np.random.default_rng(0), np.array([1, 2]), off, 0) == [],
      "_weighted_pick: without replacement, capped at the candidate count, empty for n = 0")
# 4b. the ellipsoid sampler: bounds, the truncated half-normal radius, direction symmetry, the rejection loops
L = 2.0; bone = np.array([0.0, 0.0, L]); r = np.random.default_rng(5)
D = np.stack([_ellipsoid_displacement(r, bone, ONE_OF["add_sigma"], ONE_OF["add_lateral"]) for _ in range(20000)])
ax = D @ (bone / L); lat = np.linalg.norm(D - ax[:, None] * (bone / L)[None], axis=-1)
rn = np.sqrt((ax / L) ** 2 + (lat / (ONE_OF["add_lateral"] * L)) ** 2)                       # the normalised ellipsoid radius
check(rn.max() <= 1 + 1e-9 and np.abs(ax).max() <= L + 1e-9 and lat.max() <= ONE_OF["add_lateral"] * L + 1e-9, "ellipsoid: every sample inside the ellipsoid")
from math import erf, sqrt
Phi = lambda z: 0.5 * (1 + erf(z / sqrt(2)))
p_gt_half = 2 * (Phi(1 / ONE_OF["add_sigma"]) - Phi(0.5 / ONE_OF["add_sigma"])) / (2 * Phi(1 / ONE_OF["add_sigma"]) - 1)   # P(0.5 < r <= 1 | r <= 1), half-normal sigma
check(abs(np.mean(rn > 0.5) - p_gt_half) < 0.015, f"ellipsoid: truncated half-normal radius (P(r>0.5) {np.mean(rn > 0.5):.3f}, expected {p_gt_half:.3f})")
check(np.linalg.norm(np.mean(D / np.linalg.norm(D, axis=-1, keepdims=True), axis=0)) < 0.03 and np.mean(lat > 1e-9) > 0.999, "ellipsoid: directions symmetric, off-axis")
check(np.allclose(_ellipsoid_displacement(np.random.default_rng(0), np.zeros(3), 0.5, 0.5), 0), "ellipsoid: zero-length bone -> no displacement")
check(all(np.allclose(_ellipsoid_displacement(np.random.default_rng(k), bone, 0.5, 0.5), ellipsoid_ref(np.random.default_rng(k), bone)) for k in range(500)),
      "ellipsoid: the module's sampler equals UniMate's code re-implemented, draw for draw (500 seeds)")
class Scripted:                                                                                # UniMate's rejection loops: the first uniform draw outside
    def __init__(s, u, n): s.u, s.n = list(u), list(n)                                          # the unit ball and the first radius > 1 are rejected
    def uniform(s, lo, hi, size=None): return np.asarray(s.u.pop(0), dtype=float)
    def normal(s, mu, sigma): return float(s.n.pop(0))
d = _ellipsoid_displacement(Scripted([[2.0, 2.0, 2.0], [0.0, 0.5, 0.0]], [1.5, 0.25]), bone, 0.5, 0.5)
e1 = bone / L; ref = np.array([1.0, 0.0, 0.0]) if abs(e1[1]) > 0.9 else np.array([0.0, 1.0, 0.0])
e2 = np.cross(e1, ref); e2 /= np.linalg.norm(e2); e3 = np.cross(e1, e2)
exp = np.column_stack([e1, e2, e3]) @ ((0.25 * np.array([0.0, 1.0, 0.0])) * np.array([L, 0.5 * L, 0.5 * L]))
check(np.allclose(d, exp), "ellipsoid: scripted rejection loops use the second draws, bone-aligned frame as UniMate's")

# ---- 5. one_of on real rigs ----
def replay(seed, n_leaves, n_cands):
    """UniMate's count rules on the module's RNG order (the op draw, then the rate draw): the expected count, independently"""
    r = np.random.default_rng(seed); op = ONE_OF_OPS[int(r.integers(len(ONE_OF_OPS)))]
    if op == "remove": return op, min(int(round(n_leaves * float(r.uniform(*UNI["remove_rate"])))), UNI["remove_cap"])
    if op == "pool": return op, (max(UNI["pool_min"], int(n_cands * float(r.uniform(*UNI["pool_rate"])))) if n_cands else 0)
    return op, None
idxs2 = rng.choice(len(base), size=300, replace=False)
cnt = Counter(); n_rm, n_pool, lat_all, ratios = [], [], [], []
n_adjacent = n_pool_under_root = n_add_under_root = 0
picked_len, cand_len = [], []
for n, bi in enumerate(idxs2):
    it, J0, T, x, mu0, sd0, cv0, raw0, contact, geom = load(bi)
    par0, O0, R0 = geom["parents"], geom["offset_parent_local"], geom["R_rest_global"]
    children = [[] for _ in range(J0)]
    for j in range(1, J0): children[par0[j]].append(j)
    protected = contact.copy(); protected[0] = True
    leaves0 = {j for j in range(1, J0) if not protected[j] and len(children[j]) == 0}
    cands0 = {j for j in range(1, J0) if not protected[j] and len(children[j]) == 1}
    seed = 900 + n
    tr = make_transform(np.random.default_rng(seed), one, **geom, channel_valid=cv0, mu=mu0, sd=sd0, contact_joints=contact, fps=FPS, rest_norm=True)
    tag = f"[one_of/{tr.op}/{it['object_type']}]"
    check(tr.op in ONE_OF_OPS, tag + " op recorded"); cnt[tr.op] += 1
    op_exp, n_exp = replay(seed, len(leaves0), len(cands0))
    check(op_exp == tr.op, tag + " op equals the replayed draw")
    Jn, keep, rows, synth_rows = tr.n_joints, tr.keep, tr.served_rows, np.where(tr.src < 0)[0]
    kept = set(keep.tolist()); removed = sorted(set(range(J0)) - kept)
    check(all(int(j) in kept for j in np.where(protected)[0]), tag + " protected joints kept")
    if tr.op == "remove":
        check(len(synth_rows) == 0 and len(removed) == n_exp and all(j in leaves0 for j in removed), tag + f" {len(removed)} initial leaves removed == replayed UniMate count {n_exp}")
        check(not tr.reencode and np.array_equal(tr.parents, np.array([-1] + [int(np.searchsorted(keep, par0[j])) for j in keep[1:]]))
              and np.allclose(tr.offsets[1:], O0[keep[1:]], atol=1e-4, rtol=1e-4),      # bones are rebuilt from the rest positions (module tolerance 1e-4)
              tag + " no re-encode, parents compacted, bones unscaled")
        n_rm.append(len(removed))
        if removed: picked_len += [np.linalg.norm(O0[j]) for j in removed]; cand_len += [np.linalg.norm(O0[j]) for j in leaves0]
    elif tr.op == "pool":
        check(len(synth_rows) == 0 and len(removed) == n_exp and all(j in cands0 for j in removed), tag + f" {len(removed)} single-child joints pooled == replayed UniMate count {n_exp}")
        check(tr.reencode == (len(removed) > 0), tag + " re-encode iff something was pooled")
        n_pool.append(len(removed))
        if removed: picked_len += [np.linalg.norm(O0[j]) for j in removed]; cand_len += [np.linalg.norm(O0[j]) for j in cands0]
    elif tr.op == "add":
        check(len(synth_rows) == 1 and Jn == J0 + 1 and not removed, tag + " exactly one synthetic joint, nothing removed")
        s = int(synth_rows[0]); c = s + 1; p = int(tr.parents[s])
        check(tr.parents[c] == s and tr.src[p] == par0[tr.src[c]], tag + " synthetic joint sits between the child and its original parent")
        off_c = O0[tr.src[c]]
        check(np.allclose(tr.offsets[s] + tr.offsets[c], off_c, atol=1e-4, rtol=1e-4), tag + " the two offsets sum to the bone (rest-rebuilt, tolerance 1e-4)")
        # the end-to-end oracle on the module's RNG order: op draw, the bone (uniform over the non-root joints), alpha ~ U(0.3, 0.7),
        # then UniMate's ellipsoid draws -- the split point must equal alpha * bone + displacement to the last bit
        r_o = np.random.default_rng(seed); assert ONE_OF_OPS[int(r_o.integers(len(ONE_OF_OPS)))] == "add"
        c_o = int(r_o.integers(1, J0)); alpha_o = float(r_o.uniform(*UNI["add_alpha"]))
        oc = int(tr.src[c]); off_c_rebuilt = R0[par0[oc]].T @ (geom["P_rest_global"][oc] - geom["P_rest_global"][par0[oc]])
        disp_o = ellipsoid_ref(r_o, off_c_rebuilt)
        check(c_o == oc and np.allclose(tr.offsets[s], alpha_o * off_c_rebuilt + disp_o, atol=1e-12, rtol=1e-12),
              tag + f" addition oracle: bone {oc} == {c_o}, split point == alpha {alpha_o:.3f} * bone + UniMate displacement")
        Lb = float(np.linalg.norm(off_c))
        if Lb > 1e-8:
            e1 = off_c / Lb; ax1 = float(np.dot(tr.offsets[s], e1)); la1 = float(np.linalg.norm(tr.offsets[s] - ax1 * e1))
            check(la1 <= ONE_OF["add_lateral"] * Lb + 1e-9 and -0.7 * Lb - 1e-9 <= ax1 <= 1.7 * Lb + 1e-9, tag + " split point inside UniMate's ellipsoid bounds")
            lat_all.append(la1 / Lb)
        check(tr.reencode and np.allclose(tr.R_rest[s], tr.R_rest[p]) and np.allclose(tr.R_rest_old[s], tr.R_rest_old[p]), tag + " synthetic joint rigid with its parent")
        sem = np.asarray(it["joint_semantics"])[:J0]
        sn = apply_semantics(sem, tr, np.random.default_rng(seed))
        check(np.allclose(sn[s], 0.5 * (sem[tr.src[p]] + sem[tr.src[c]])) and np.array_equal(sn[rows], sem[keep].astype(np.float32)),
              tag + " semantics: mean of parent and child, originals untouched")
    elif tr.op == "scale":
        check(Jn == J0 and len(synth_rows) == 0 and np.array_equal(tr.parents, par0), tag + " topology unchanged")
        nz = np.linalg.norm(O0[1:], axis=-1) > 1e-8
        ratio = np.linalg.norm(tr.offsets[1:], axis=-1)[nz] / np.linalg.norm(O0[1:], axis=-1)[nz]
        check(ratio.min() >= 1 - 0.1 - 1e-9 and ratio.max() <= 1 + 0.1 + 1e-9 and ratio.std() > 0.01, tag + " every bone scaled within U[0.9, 1.1], independently")
        check(tr.reencode, tag + " re-encoded"); ratios += ratio.tolist()
    check(np.allclose(tr.Q, np.eye(3)), tag + " no rest-convention rotation")
    check(np.array_equal(tr.mu[rows][:, 3:9], tr.mu0[rows][:, 3:9]) and np.array_equal(tr.sd[rows][:, 3:9], tr.sd0[rows][:, 3:9]), tag + " rotation statistics unperturbed")
    if not tr.reencode:
        check(np.array_equal(tr.mu, tr.mu0) and np.array_equal(tr.sd, tr.sd0), tag + " un-re-encoded sample keeps every statistic")
    check(tr.sem_noise == 0 and not tr.sem_drop, tag + " no description perturbation")
    xn = apply_motion(x, tr)
    check(xn.shape == (T, Jn, 18) and np.isfinite(xn).all(), tag + " motion finite")
    if not tr.reencode:
        check(np.array_equal(xn[:, rows], x[:, keep]), tag + " un-re-encoded sample: served values of kept joints identical")
    rawn = denorm(xn, tr); wn = world(rawn)
    bl = np.linalg.norm(tr.offsets[1:], axis=-1).mean()
    Gn = G_of(rawn, tr.R_rest); G0 = G_of(raw0, R0)
    check(tr.unimate_rot == (tr.op in ("add", "pool") and tr.reencode), tag + " unimate_rot set iff add / pool re-encoded")
    rec = np.zeros(Jn, bool)                     # rows whose rest-delta the port recomposes: the synthetic joint and its subtree,
    for n in range(Jn):                          # rows with a pooled original ancestor (independently derived)
        k = int(tr.src[n])
        if k < 0: rec[n] = True
        else:
            q = int(par0[k])
            while q >= 0 and q in kept: q = int(par0[q])
            rec[n] = q >= 0
        if not rec[n] and tr.parents[n] >= 0 and rec[tr.parents[n]]: rec[n] = True
    xn2, cb = apply_motion_with_contact(x, tr)
    check(np.array_equal(xn2, xn) and ((cb is None) == (not tr.unimate_rot)), tag + " apply_motion_with_contact: same output, pre-pruning flags iff pool / add re-encoded")
    if tr.unimate_rot:
        compare_with_unimate(tag, geom, raw0, tr, xn)
        check(tr.recomposed is not None and np.array_equal(tr.recomposed, rec), tag + " the transform records the recomposed rows")
        con_out = denorm(xn, tr)[..., 12] > 0.5
        check(cb.shape == (T, Jn) and np.array_equal(cb[:, rows], raw0[:, keep, 12] > 0.5) and not cb[:, synth_rows].any() and not (con_out & ~cb).any(),
              tag + " pre-pruning flags = the served flags on the kept rows, none on the synthetic row; pruning only clears")
        l_unp, l_den = lock_term(xn, tr.mu, tr.sd, tr.offsets, contact_on=cb), lock_term(xn, tr.mu, tr.sd, tr.offsets, lock_denominator=pairs_of(cb))
        check(l_den <= l_unp + 1e-6, tag + f" foot-lock with the pre-pruning denominator {l_den:.4f} <= unpruned {l_unp:.4f}")
        # every recomposed row must have all six rotation cells valid with a usable scale; every other row keeps the rig's mask
        check(tr.channel_valid[rec][:, 3:9].all() and (tr.sd[rec][:, 3:9] > 1e-3).all() and tr.channel_valid[rec][:, 12].all() and (tr.sd[rec][:, 12] > 1e-3).all(),
              tag + f" rotation and contact cells of the {int(rec.sum())} recomposed rows valid with usable scales")
        orig_unrec = np.where(~rec & (tr.src >= 0))[0]
        check(np.array_equal(tr.channel_valid[orig_unrec][:, 3:9], cv0[tr.src[orig_unrec]][:, 3:9]), tag + " rotation mask of the other rows unchanged")
        if tr.op == "pool" and any(par0[j] in removed for j in removed): n_adjacent += 1
        if tr.op == "pool" and any(par0[j] == 0 for j in removed): n_pool_under_root += 1
        if tr.op == "add" and tr.parents[int(synth_rows[0])] == 0: n_add_under_root += 1
    if tr.op in ("remove", "scale"):
        check(np.abs(Gn[:, rows] - G0[:, keep]).max() < 1e-5, tag + " global rotations of originals invariant")
    else:
        Ge = expected_G(raw0, R0, par0, tr)                       # UniMate's local-rotation FK: pooled articulation deleted / parent's duplicated
        check(np.abs(Gn - Ge).max() < 1e-5, tag + f" global rotations follow UniMate's rule on every served row (max {np.abs(Gn - Ge).max():.1e})")
    fk = fk_from_global_rotations(tr.parents, wn[:, 0], Gn, tr.offsets)
    gap = np.abs(fk - wn).max() / bl
    check(gap < 1e-5, tag + f" FK == direct (max {gap:.2e} bl)")
    if tr.reencode:
        og = official_fk_gap(xn, tr); check(og < 1e-4, tag + f" official FK mirror (mean {og:.2e} bl)")
        # contact: the scale op carries every served flag over; pool / add carry them over and clear, on the RECOMPOSED rows
        # only, the flag at frame t where the joint's forward displacement grew by more than 0.05 bone lengths; the last
        # frame keeps its flag; synthetic rows carry none
        con = rawn[..., 12] > 0.5
        check(not con[:, synth_rows].any(), tag + " synthetic rows carry no contact")
        if tr.op == "scale":
            check(np.array_equal(con[:, rows], raw0[:, keep, 12] > 0.5), tag + " scale: every served flag carried over")
        else:
            unm = np.where(~rec & (tr.src >= 0))[0]; rr_ = np.where(rec & (tr.src >= 0))[0]
            check(np.array_equal(con[:, unm], raw0[:, tr.src[unm], 12] > 0.5), tag + " flags of the unmoved rows unchanged")
            w0r = world(raw0)[:, tr.src[rr_]]
            d_old = np.linalg.norm(w0r[1:] - w0r[:-1], axis=-1); d_new = np.linalg.norm(wn[1:, rr_] - wn[:-1, rr_], axis=-1)
            exp_con = raw0[:, tr.src[rr_], 12] > 0.5; exp_con[:-1] &= d_new <= d_old + 0.05 * bl
            # the module decides on float64 FK positions, the test on the float32 served output: cells within 1e-4 bl of the
            # threshold are left out of the exact comparison
            margin = np.ones_like(exp_con); margin[:-1] = np.abs(d_new - (d_old + 0.05 * bl)) > 1e-4 * bl
            n_mis = int((con[:, rr_] != exp_con)[margin].sum())
            check(n_mis == 0, tag + f" recomposed rows: flags carried over and cleared where the joint moves farther ({int((raw0[:, tr.src[rr_], 12] > 0.5).sum() - exp_con.sum())} cleared, {n_mis} mismatches off the band)")
        vexp = np.zeros_like(wn)                                     # velocity = fps x forward difference of WORLD positions, last repeated
        if T >= 2: vexp[:-1] = (wn[1:] - wn[:-1]) * FPS; vexp[-1] = vexp[-2]
        verr = np.abs(rawn[..., 9:12] - vexp).max() / (np.abs(vexp).max() + 1e-6)
        check(verr < 1e-4, tag + f" velocities (all rows incl. synthetic) = finite differences, last frame repeated (rel {verr:.1e})")
    d = apply_motion(np.array(np.asarray(base.rest_frame_normalized(it["object_type"]))[:J0], dtype=np.float32), tr)
    check(np.abs(d[:, 0:3][tr.channel_valid[:, 0:3]]).max() < 1e-3, tag + " rest demo position channels normalise to ~0")
fr = {k: cnt[k] / len(idxs2) for k in ONE_OF_OPS}
check(all(0.15 <= f <= 0.35 for f in fr.values()), f"op frequencies near uniform: {fr}")
check(max(n_rm) >= 2 and max(n_pool) >= 4, f"the rates actually bite (max leaves removed {max(n_rm)}, max pooled {max(n_pool)})")
check(n_adjacent >= 5, f"adjacent pooled joints covered by the UniMate-literal reference on real rigs ({n_adjacent}; the root cases are the fixtures' job: "
                      f"a pooled child of the root {n_pool_under_root}, insertion below the root {n_add_under_root})")
check(len(picked_len) >= 100 and np.mean(picked_len) < np.mean(cand_len), f"shorter bones picked on aggregate (picked/candidate mean length {np.mean(picked_len) / np.mean(cand_len):.3f})")
ratios = np.asarray(ratios)
check(abs(ratios.mean() - 1.0) < 0.01 and abs(np.mean(ratios < 0.95) - 0.25) < 0.05 and abs(np.mean(ratios > 1.05) - 0.25) < 0.05,
      f"bone scale factors U[0.9,1.1] on aggregate (mean {ratios.mean():.4f}, P(<0.95) {np.mean(ratios < 0.95):.3f}, P(>1.05) {np.mean(ratios > 1.05):.3f})")
print(f"[one_of] ops {dict(cnt)}; leaves removed mean {np.mean(n_rm):.2f} max {max(n_rm)}; pooled mean {np.mean(n_pool):.2f} max {max(n_pool)}; "
      f"add lateral/L mean {np.mean(lat_all):.3f} max {max(lat_all):.3f}; picked/candidate bone length {np.mean(picked_len) / np.mean(cand_len):.3f}")

# ---- 5a. the masked-input reproducer (codex r4 P1-1): PZ_Red_River_Hog_Female's root has a rotation cell excluded as a constant;
#          a synthetic joint inserted under that root recomposes to Delta_root^2, whose cell varies -- it must re-enter ----
bi_rep = next((i for i, r in enumerate(base._rows) if str(r["clip_id"]) == "8a9f16a7b861f60ca8df"), None)
check(bi_rep is not None, "reproducer clip 8a9f16a7b861f60ca8df present in the training view")
if bi_rep is not None:
    it, J0, T, x, mu0, sd0, cv0, raw0, contact, geom = load(bi_rep)
    check(it["object_type"] == "PZ_Red_River_Hog_Female" and not cv0[0, 3:9].all(), "reproducer: the rig's root has an excluded rotation cell")
    found = None
    for seed in [34] + list(range(2000)):
        trp = make_transform(np.random.default_rng(seed), one, **geom, channel_valid=cv0, mu=mu0, sd=sd0, contact_joints=contact, fps=FPS, rest_norm=True)
        if trp.op == "add" and trp.parents[int(np.where(trp.src < 0)[0][0])] == 0: found = (seed, trp); break
    check(found is not None, "reproducer: an insertion under the root was drawn")
    if found is not None:
        seed, trp = found; sp = int(np.where(trp.src < 0)[0][0]); xp = apply_motion(x, trp)
        og = official_fk_gap(xp, trp)
        check(trp.channel_valid[sp, 3:9].all() and og < 1e-4, f"reproducer (seed {seed}): synthetic row's rotation cells valid, official FK gap after the mask projection {og:.2e} bl (was 0.065)")
        compare_with_unimate(f"[reproducer/seed{seed}]", geom, raw0, trp, xp)

# ---- 5a2. the stale-contact reproducer (codex r5 P1): PZ_Dingo_Juvenile clip 1fe162f85d02074ad9b8, seed 14 -- with the flags
#           carried over unpruned, the foot-lock term demanded stillness of joints the recomposition moved; pruned flags do not ----
bi_d = next((i for i, r in enumerate(base._rows) if str(r["clip_id"]) == "1fe162f85d02074ad9b8"), None)
check(bi_d is not None, "reproducer clip 1fe162f85d02074ad9b8 present in the training view")
if bi_d is not None:
    it, J0, T, x, mu0, sd0, cv0, raw0, contact, geom = load(bi_d)
    lock_orig = lock_term(x, mu0, sd0, geom["offset_parent_local"])
    cand = []
    for seed in [14] + list(range(300)):
        trd = make_transform(np.random.default_rng(seed), one, **geom, channel_valid=cv0, mu=mu0, sd=sd0, contact_joints=contact, fps=FPS, rest_norm=True)
        if not trd.unimate_rot: continue
        xd = apply_motion(x, trd)
        stale = np.zeros((T, trd.n_joints), bool); stale[:, trd.served_rows] = raw0[:, trd.keep, 12] > 0.5       # the flags carried over unpruned
        l_stale, l_new = lock_term(xd, trd.mu, trd.sd, trd.offsets, contact_on=stale), lock_term(xd, trd.mu, trd.sd, trd.offsets)
        bnd = lock_bound(raw0, trd, xd)
        cand.append((seed, trd.op, l_stale, l_new, bnd))
        if seed == 14 or l_stale > 4 * max(lock_orig, 1e-6): break
    check(bool(cand), "reproducer: a re-encoded pool / add draw exists")
    if cand:
        seed, opd, l_stale, l_new, bnd = cand[-1]
        check(l_new <= bnd + 1e-6 and l_new < 0.5 * l_stale,
              f"reproducer (seed {seed}, {opd}): foot-lock with the prediction equal to the target -- original clip {lock_orig:.3f}, carried-over flags {l_stale:.3f}, pruned flags {l_new:.3f} (exact bound {bnd:.3f})")
        print(f"[contact] Dingo reproducer seed {seed} {opd}: lock original {lock_orig:.3f} | stale flags {l_stale:.3f} | pruned {l_new:.3f} | bound {bnd:.3f}")

# ---- 5a2b. the scale reproducer (codex r6 P2-1): clip e917715115890a20c9d0 seed 1122 -- pruning the planted feet of a scaled
#            skeleton raised the term's mean (0.066 -> 0.151); the scale op now carries every flag over ----
bi_s = next((i for i, r in enumerate(base._rows) if str(r["clip_id"]) == "e917715115890a20c9d0"), None)
check(bi_s is not None, "reproducer clip e917715115890a20c9d0 present in the training view")
if bi_s is not None:
    it, J0, T, x, mu0, sd0, cv0, raw0, contact, geom = load(bi_s)
    trs_ = make_transform(np.random.default_rng(1122), one, **geom, channel_valid=cv0, mu=mu0, sd=sd0, contact_joints=contact, fps=FPS, rest_norm=True)
    check(trs_.op == "scale", f"reproducer: seed 1122 draws the scale op (got {trs_.op})")
    if trs_.op == "scale":
        xs_ = apply_motion(x, trs_); l_orig = lock_term(x, mu0, sd0, geom["offset_parent_local"]); l_aug = lock_term(xs_, trs_.mu, trs_.sd, trs_.offsets)
        check(np.array_equal(denorm(xs_, trs_)[..., 12] > 0.5, raw0[..., 12] > 0.5) and l_aug <= 1.25 * l_orig,
              f"scale reproducer: flags unchanged, foot-lock of the scaled target {l_aug:.4f} within 1.25x the clip's own {l_orig:.4f} (pruned it was 0.151)")
        print(f"[contact] scale reproducer: lock original {l_orig:.4f} | scaled target {l_aug:.4f}")

# ---- 5a2c. the denominator reproducer (codex r7 P2-1): PZ_Caracal_Juvenile clip cf606e52b7af99c6f419 seed 34 (add) -- pruning
#            lowered the numerator but the mean over the SURVIVING pairs rose (1.130 -> 1.358); with the window's pre-pruning
#            pair count as the denominator it cannot ----
bi_c = next((i for i, r in enumerate(base._rows) if str(r["clip_id"]) == "cf606e52b7af99c6f419"), None)
check(bi_c is not None, "reproducer clip cf606e52b7af99c6f419 present in the training view")
if bi_c is not None:
    it, J0, T, x, mu0, sd0, cv0, raw0, contact, geom = load(bi_c)
    trc = make_transform(np.random.default_rng(34), one, **geom, channel_valid=cv0, mu=mu0, sd=sd0, contact_joints=contact, fps=FPS, rest_norm=True)
    check(trc.op == "add" and trc.unimate_rot, f"reproducer: seed 34 draws add (got {trc.op})")
    if trc.unimate_rot:
        xc, cbc = apply_motion_with_contact(x, trc)
        l_o = lock_term(x, mu0, sd0, geom["offset_parent_local"])
        l_unp = lock_term(xc, trc.mu, trc.sd, trc.offsets, contact_on=cbc)
        l_surv = lock_term(xc, trc.mu, trc.sd, trc.offsets)
        l_den = lock_term(xc, trc.mu, trc.sd, trc.offsets, lock_denominator=pairs_of(cbc))
        n_after = pairs_of(denorm(xc, trc)[..., 12] > 0.5)
        check(l_den <= l_unp + 1e-9 and l_den < l_surv and abs(l_den - l_surv * n_after / pairs_of(cbc)) < 1e-4,
              f"Caracal reproducer: foot-lock original {l_o:.3f}, unpruned {l_unp:.3f}, surviving-pair mean {l_surv:.3f}, pre-pruning denominator {l_den:.3f} (pairs {pairs_of(cbc)} -> {n_after})")
        print(f"[contact] Caracal reproducer: lock original {l_o:.3f} | unpruned {l_unp:.3f} | surviving-pair mean {l_surv:.3f} | pre-pruning denominator {l_den:.3f} | pairs {pairs_of(cbc)} -> {n_after}")

# ---- 5a3. a stationary-contact chain: the root and joint 1 coincide, joint 1 counter-rotates the root's turn, so joints 2 and 3
#           stand still (every joint at ground height, all flagged); deleting joint 1's articulation (pool) or duplicating it
#           (add on bone 1->2) swings the foot -- its flag is pruned and the lock term stays zero ----
gs = synth([-1, 0, 1, 2]); gs["offset_parent_local"][1] = 0.0; gs["P_rest_global"][:] = [[0, 0, 0], [0, 0, 0], [1, 0, 0], [2, 0, 0]]
for turn in (30.0, 0.01):
    Ds = np.stack([np.tile(np.eye(3), (4, 1, 1)), np.stack([Rz(turn), np.eye(3), np.eye(3), np.eye(3)])])
    xs, raws = fixture_clip_rot(gs, Ds, np.zeros((2, 3)), contact_rule=CR1)
    check(np.allclose(raws[0, :, 0:3], raws[1, :, 0:3]) and (raws[..., 12] == 1).all(), f"fixture (turn {turn}): every joint stationary and flagged")
    gots = Counter()
    for seed in range(3000):
        trs = make_transform(np.random.default_rng(seed), one, **gs, contact_joints=np.array([True, False, False, False]))
        kind = "pool" if trs.op == "pool" and trs.keep.tolist() == [0, 2, 3] else "add" if trs.op == "add" and trs.src.tolist() == [0, 1, -1, 2, 3] else None
        if kind is None or gots[kind]: continue
        gots[kind] += 1
        xn_s = apply_motion(xs, trs); rawn_s = denorm(xn_s, trs); wn_s = world(rawn_s)
        moved = np.linalg.norm(wn_s[1] - wn_s[0], axis=-1) > 1e-6
        still = ~moved & (trs.src >= 0)                                           # a synthetic row carries no contact by design
        stale = np.ones((2, trs.n_joints), bool); stale[:, np.where(trs.src < 0)[0]] = False
        l_stale, l_new, bnd = lock_term(xn_s, trs.mu, trs.sd, trs.offsets, contact_on=stale), lock_term(xn_s, trs.mu, trs.sd, trs.offsets), lock_bound(raws, trs, xn_s)
        if turn >= 1.0:
            # a swing of a bone-length scale: the moved joints lose their flag on every frame but the last (the flag at t is judged
            # on the forward displacement, the pair t, t+1); the still originals keep theirs; the lock demands nothing
            check(moved.any() and not (rawn_s[:-1][:, moved, 12] > 0.5).any() and (rawn_s[:, still, 12] > 0.5).all() and not (rawn_s[:, trs.src < 0, 12] > 0.5).any(),
                  f"[static-chain/{kind}/turn{turn}] the moved joints lost their flag, the still originals kept it (moved rows {np.where(moved)[0].tolist()})")
            check(l_new < 1e-9 and l_stale > 0.5, f"[static-chain/{kind}/turn{turn}] foot-lock with unpruned flags {l_stale:.3f} vs pruned {l_new:.2e}")
        else:
            # a swing far below the threshold: every flag survives and the lock stays within the exact bound
            check(moved.any() and (rawn_s[:, trs.src >= 0, 12] > 0.5).all() and l_new <= bnd + 1e-9 and l_new > 0,
                  f"[static-chain/{kind}/turn{turn}] flags survive a sub-threshold swing, lock {l_new:.2e} within the bound {bnd:.2e}")
        compare_with_unimate(f"[static-chain/{kind}/turn{turn}]", gs, raws, trs, xn_s)
        if gots["pool"] and gots["add"]: break
    check(gots["pool"] == 1 and gots["add"] == 1, f"static chain (turn {turn}): both cases drawn {dict(gots)}")

# ---- 5a4. a contact cell excluded as a constant one (codex r8 P2): the trainer projects an invalid cell to zero and rebuilds it
#           from its mean, which would turn a pruned 0 back into 1 -- the recomposed rows' contact cells re-enter the mask ----
gc = synth([-1, 0, 1, 2]); gc["offset_parent_local"][1] = 0.0; gc["P_rest_global"][:] = [[0, 0, 0], [0, 0, 0], [1, 0, 0], [2, 0, 0]]
gc["channel_valid"][3, 12] = False; gc["mu"][3, 12] = 1.0; gc["sd"][3, 12] = 0.0                # joint 3's flag: a constant one
Dc = np.stack([np.tile(np.eye(3), (4, 1, 1)), np.stack([Rz(30), np.eye(3), np.eye(3), np.eye(3)])])
xc_, rawc = fixture_clip_rot(gc, Dc, np.zeros((2, 3)), contact_rule=CR1)
check((rawc[..., 12] == 1).all() and np.allclose(xc_[:, 3, 12], 0.0), "fixture: every flag on, joint 3's served as the excluded constant")
got_c = False
for seed in range(3000):
    trc_ = make_transform(np.random.default_rng(seed), one, **gc, contact_joints=np.array([True, False, False, False]))
    if not (trc_.op == "pool" and trc_.keep.tolist() == [0, 2, 3]): continue
    got_c = True
    xn_c = apply_motion(xc_, trc_)
    proj = xn_c.copy(); proj[..., :17][:, ~trc_.channel_valid] = 0.0                          # the trainer's state projection
    flags_served = denorm(xn_c, trc_)[..., 12] > 0.5; flags_proj = denorm(proj, trc_)[..., 12] > 0.5
    check(trc_.channel_valid[2, 12] and not flags_served[0, 2] and np.array_equal(flags_served, flags_proj),
          f"constant-one contact cell: re-entered on the recomposed row, the pruned flag survives the projection (served {flags_served[0].astype(int).tolist()}, projected {flags_proj[0].astype(int).tolist()})")
    check(lock_term(proj, trc_.mu, trc_.sd, trc_.offsets, lock_denominator=pairs_of(apply_motion_with_contact(xc_, trc_)[1])) < 1e-9,
          "constant-one contact cell: the foot-lock through the projected input is zero")
    break
check(got_c, "constant-one contact fixture: the pooling case was drawn")

# ---- 5b. upstream differential fixtures: chain 0->1->2->3, unit X bones, identity rest, a 90-degree articulation at joint 1 ----
gf = synth([-1, 0, 1, 2]); Gf = np.stack([np.eye(3), Rz(90), Rz(90), Rz(90)]); xf, posf = fixture_clip(gf, Gf)
check(np.allclose(posf, [[0, 0, 0], [1, 0, 0], [1, 1, 0], [1, 2, 0]]), "fixture: joint 2 at (1,1,0), joint 3 at (1,2,0) before pooling")
protf = np.array([True, False, False, False])
got_pool = got_add = False
for seed in range(400):
    trf = make_transform(np.random.default_rng(seed), one, **gf, contact_joints=protf)
    if trf.op == "pool" and trf.keep.tolist() == [0, 2, 3] and not got_pool:
        wf = world(denorm(apply_motion(xf, trf), trf))[0]
        check(np.allclose(wf, [[0, 0, 0], [2, 0, 0], [3, 0, 0]], atol=1e-6), f"pooling joint 1 deletes its articulation as UniMate's collapse does: joint 3 at {np.round(wf[2], 4)} (UniMate (3,0,0); global-rotation-preserving pooling would give (2,1,0))")
        got_pool = True
    if trf.op == "add" and trf.src.tolist() == [0, 1, -1, 2, 3] and not got_add:
        wf = world(denorm(apply_motion(xf, trf), trf))[0]
        exp_s = np.array([1, 0, 0]) + Rz(90) @ trf.offsets[2]; exp_2 = exp_s + Rz(180) @ trf.offsets[3]; exp_3 = exp_2 + Rz(180) @ np.array([1, 0, 0])
        check(np.allclose(wf, [[0, 0, 0], [1, 0, 0], exp_s, exp_2, exp_3], atol=1e-6),
              f"inserting on bone 1->2 duplicates joint 1's local rotation at the new joint as UniMate's insertion does: joint 2 at {np.round(wf[3], 4)} (expected {np.round(exp_2, 4)}; a rigid insertion would keep (1,1,0))")
        got_add = True
    if got_pool and got_add: break
check(got_pool and got_add, "fixture: both differential cases were drawn")

# ---- 5c. rotated fixtures against the UniMate-literal reference: random non-identity rest rotations and root rotation ----
rr = np.random.default_rng(21)
for fname, parents_f in (("chain12", [-1] + list(range(11))), ("fork6", [-1, 0, 0, 1, 2, 3])):
    gr = synth_rot(parents_f, rr); Jr = len(parents_f)
    Dr = np.stack([np.stack([rand_rot(rr) for _ in range(Jr)]) for _ in range(2)])         # 2 frames, every joint incl. the root
    xr, rawr = fixture_clip_rot(gr, Dr, rr.normal(size=(2, 3)) * 5.0)
    protr = np.zeros(Jr, bool); protr[0] = True
    seen = Counter()
    # chain12: candidates 1..10, 1-3 pooled per draw (adjacent pairs, joint 1 = a child of the root); fork6: the root's two
    # children are candidates and bones below the root are split
    want = ({"pool_adjacent": 2, "pool_under_root": 1, "pool": 3, "add_below_root": 1, "add": 4} if fname == "chain12"
            else {"pool_under_root": 2, "pool": 2, "add_below_root": 2, "add": 4})
    for seed in range(4000):
        trr = make_transform(np.random.default_rng(seed), one, **gr, contact_joints=protr)
        if not trr.unimate_rot: continue
        rem = sorted(set(range(Jr)) - set(trr.keep.tolist()))
        if trr.op == "pool":
            kind = ("pool_adjacent" if any(parents_f[j] in rem for j in rem) else "pool_under_root" if any(parents_f[j] == 0 for j in rem) else "pool")
        else:
            kind = "add_below_root" if trr.parents[int(np.where(trr.src < 0)[0][0])] == 0 else "add"
        if seen[kind] >= want.get(kind, 0): continue
        seen[kind] += 1
        xnr = apply_motion(xr, trr)
        compare_with_unimate(f"[fixture/{fname}/{kind}/seed{seed}]", gr, rawr, trr, xnr)
        if all(seen[k] >= v for k, v in want.items()): break
    check(all(seen[k] >= v for k, v in want.items()), f"fixture {fname}: every structural case drawn {dict(seen)} (wanted {want})")

# ---- 6. op frequency, 4000 draws on one rig ----
it, J0, T, x, mu0, sd0, cv0, raw0, contact, geom = load(idxs2[0])
c4 = Counter(make_transform(np.random.default_rng(10_000 + k), one, **geom, channel_valid=cv0, mu=mu0, sd=sd0, contact_joints=contact, fps=FPS, rest_norm=True).op
             for k in range(4000))
check(all(abs(c4[k] / 4000 - 0.25) < 0.03 for k in ONE_OF_OPS), f"4000 draws: {dict(c4)}")

# ---- 7. the dataset's five-way draw: p = 0.8 of the served samples are augmented, the four ops uniform among them ----
names = ktjd17_split_names(ROOT, exclude=CUT)
calls = []
_orig = AUGM.make_transform
def spy(rng_, cfg_, **kw):
    tr_ = _orig(rng_, cfg_, **kw); calls.append(tr_.op); return tr_
AUGM.make_transform = spy
ds = InContextPairs(base, names["train"], names["train"], demo_frames=1, target_frames=240, balance_skeletons=True, seed=0,
                    emit_fk_fields=True, emit_graph_v2=True, identity_p=0.0, emit_ref_text=False, demo_rest=True, augment=one)
N = 500
for i in range(N): ds[i]
AUGM.make_transform = _orig
frac = len(calls) / N; c5 = Counter(calls)
check(0.74 <= frac <= 0.86, f"dataset: {frac:.3f} of served samples augmented (expected 0.8)")
check(all(0.17 <= c5[k] / max(len(calls), 1) <= 0.33 for k in ONE_OF_OPS), f"dataset: ops uniform among augmented samples {dict(c5)}")
print(f"[dataset] augmented {len(calls)}/{N} = {frac:.3f}; ops {dict(c5)}")
# ---- 8. the trainer's calibration arm-model guard: legacy schema, modern strict, uniform strict, no record ----
spec = importlib.util.spec_from_file_location("trainer_mod", "scripts/train_v2_incontext.py"); trm = importlib.util.module_from_spec(spec); spec.loader.exec_module(trm)
full = dict(struct_feats=True, dir_bias=True, geo_bias=True, freeze_zero_joint_sem=False)
legacy = json.load(open("configs/pilot_animal_scaleonly_gamma_calibration_b32_v1.json"))
check(not any(k in legacy["protocol"]["verify"]["arm_model"] for k in full), "fixture: the scale-only artifact records the architecture only")
check(trm.calib_arm_model_drift(legacy, full, False) is None and trm.calib_arm_model_drift(legacy, dict(full, freeze_zero_joint_sem=True), False) is None,
      "trainer guard: a calibrated arm never compares the model record (legacy artifact accepted either way)")
artA = json.load(open("configs/pilot_animal_heldout_rest_gamma_calibration_b16_v1.json"))
check(trm.calib_arm_model_drift(artA, full, False) is None and trm.calib_arm_model_drift(artA, dict(full, geo_bias=False), False) is None, "trainer guard: calibrated arms are bound by the pinned artifact, not by this check")
flat_args = json.load(open("runs/v2_noik_pilot36m_mdmflat/args.json")); flat_art = json.load(open(flat_args["ktjd_gamma_calib"]))
want_flat = {k: bool(flat_args[k]) for k in full}
check(flat_art["protocol"]["verify"]["arm_model"]["struct_feats"] != want_flat["struct_feats"] and trm.calib_arm_model_drift(flat_art, want_flat, bool(flat_args["require_uniform_gammas"])) is None,
      "trainer guard: the flat arm's deliberate use of the nodesc control's artifact stays accepted")
norec = {"protocol": {"verify": {}}}
check(trm.calib_arm_model_drift(norec, full, False) is None and trm.calib_arm_model_drift(norec, full, True) is not None, "trainer guard: no record -> accepted unless the uniform arm asks")
artU = json.load(open("configs/pilot_animal_uniform_gamma_calibration_b16_v1.json"))
wantU = {k: bool(artU["protocol"]["verify"]["arm_model"][k]) for k in full}
check(trm.calib_arm_model_drift(artU, wantU, True) is None and trm.calib_arm_model_drift(artU, dict(wantU, dir_bias=not wantU["dir_bias"]), True) is not None, "trainer guard: uniform artifact compared strictly")
artU_stripped = json.loads(json.dumps(artU)); [artU_stripped["protocol"]["verify"]["arm_model"].pop(k) for k in full]
check(trm.calib_arm_model_drift(artU_stripped, wantU, True) is not None, "trainer guard: a uniform artifact without the four conditioning fields is refused (no legacy completion)")
# ---- 9. scripts/_calib_artifact_check.py: the complete augmentation record must equal the arm's AugConfig.protocol() ----
ENV = dict(os.environ, AUG_MODE="one_of", AUG_P="0.8", AUG_BONE_SCALE="0.1", AUG_DROP_MODE="tips", AUG_DROP_MAX_FRAC="0", AUG_REST_DEG="0",
           AUG_SEM_NOISE="0", AUG_SEM_DROP_P="0", AUG_STATS_LOGSD="0", AUG_STATS_SHIFT="0", AUG_POOL_FRAC="0", AUG_ADD_P="0",
           DIM="384", DEPTH="8", HEADS="6", QK_NORM="1", STRUCT_FEATS="1", DIR_BIAS="1", ARM_GEO_BIAS="1", ARM_FREEZE_ZERO_JOINT_SEM="0",
           GAMMA_SOLVE="kimodo", VERIFY_STEPS="30", EXPECT_CODE_SCRIPT="scripts/_measure_ktjd17_gamma_calibration_view_v2.py")
def check_rc(art, name):
    pth = f"runs/_heldout/_aug_dev_logs/_tmp_artifact_{name}.json"; json.dump(art, open(pth, "w"))
    r = subprocess.run([sys.executable, "scripts/_calib_artifact_check.py", pth], env=ENV, capture_output=True, text=True)
    os.remove(pth); return r.returncode, (r.stdout + r.stderr).strip()[-300:]
src_art = "configs/pilot_animal_heldout_unimate_gamma_calibration_b16_v1.json"
disk = json.load(open(src_art)) if os.path.exists(src_art) else json.load(open(sorted(__import__("glob").glob("runs/_heldout/_calib/superseded_*unimate_b16_artifact.json"))[-1]))
cur = json.loads(json.dumps(disk)); cur["protocol"]["augmentation"] = one.protocol()
rc, msg = check_rc(cur, "current"); check(rc == 0, f"artifact check accepts the current record ({msg})")
negs = {}
a = json.loads(json.dumps(cur)); a["protocol"]["augmentation"]["version"] = "ktjd17_skel_aug_v3.2_one_of"; a["protocol"]["augmentation"].pop("contact"); negs["old version without the contact rule"] = a
a = json.loads(json.dumps(cur)); a["protocol"]["augmentation"].pop("reentry"); negs["missing reentry"] = a
a = json.loads(json.dumps(cur)); a["protocol"]["augmentation"]["rotation_rule"] = "preserve_global_rotations"; negs["wrong rotation rule"] = a
a = json.loads(json.dumps(cur)); a["protocol"]["augmentation"]["p"] = 0.5; negs["wrong p"] = a
a = json.loads(json.dumps(cur)); a["protocol"]["augmentation"]["contact_abs_bl"] = 0.5; negs["wrong contact threshold"] = a
a = json.loads(json.dumps(cur)); a["protocol"]["gamma_solve"] = "uniform"; negs["uniform solve"] = a
a = json.loads(json.dumps(cur)); a["protocol"]["verify"]["arm_model"]["geo_bias"] = False; negs["wrong model"] = a
a = json.loads(json.dumps(cur)); a["protocol"]["verify"]["steps"] = 1; negs["one verify step"] = a
a = json.loads(json.dumps(cur)); a["hashes"]["code_script"] = "scripts/_measure_ktjd17_gamma_calibration_view.py"; negs["wrong producer"] = a
for name, art in negs.items():
    rc, msg = check_rc(art, name.replace(" ", "_")); check(rc != 0, f"artifact check refuses: {name} ({msg})")
if disk["protocol"]["augmentation"] != one.protocol():
    rc, msg = check_rc(disk, "disk"); check(rc != 0, f"artifact check refuses the on-disk artifact measured under an earlier record ({msg})")
# ---- 10. the foot-lock denominator in the loss: parity without it, the pre-pruning count with it, a fully pruned sample counts ----
Bn, Tn, Jn_ = 3, 4, 2
posn = np.zeros((Bn, Tn, Jn_, 3)); posn[:, :, 1, 0] = np.arange(Tn)[None] * 0.3          # joint 1 slides 0.3 per frame, joint 0 still
xl = np.zeros((Bn, Tn, Jn_, 17), np.float32); xl[..., 0:3] = posn
mean_l, std_l = torch.zeros(Bn, Jn_, 17), torch.ones(Bn, Jn_, 17) - _STD_FLOOR             # de-normalisation is the identity
off_l = torch.zeros(Bn, Jn_, 3); off_l[:, 1, 0] = 1.0
con_l = torch.zeros(Bn, Tn, Jn_, dtype=torch.bool)
con_l[0, :, :] = True                      # sample 0: every pair locked (3 pairs on each joint)
con_l[1, :, 1] = True                      # sample 1: joint 1 only (3 sliding pairs), as if joint 0's pairs were pruned
con_l[2, :, :] = False                     # sample 2: every pair pruned
xt_l = torch.from_numpy(xl)
args_l = (xt_l, xt_l, mean_l, std_l, _STD_FLOOR, off_l, torch.tensor([Jn_] * Bn), torch.ones(Bn, Tn, dtype=torch.bool))
_, l_none, _, _ = ktjd_dynamics_losses(*args_l, contact_on=con_l, want_diag=False)
_, l_zero, _, _ = ktjd_dynamics_losses(*args_l, contact_on=con_l, want_diag=False, lock_denominator=torch.zeros(Bn))
_, l_den3, _, _ = ktjd_dynamics_losses(*args_l, contact_on=con_l, want_diag=False, lock_denominator=torch.tensor([0.0, 6.0, 6.0]))
scale_l = FK_SCALE_FRAC * (1.0 + 1e-3); dp2 = (0.3 / scale_l) ** 2
check(abs(float(l_none) - float(l_zero)) < 1e-9, "loss: lock_denominator zeros == None (byte parity for every other arm)")
check(abs(float(l_none) - (dp2 / 2 + dp2) / 2) < 1e-4, f"loss without denominators: sample 0 mean over 6 pairs {dp2/2:.3f}, sample 1 over its 3 pairs {dp2:.3f}, sample 2 skipped -> {float(l_none):.4f}")
check(abs(float(l_den3) - (dp2 / 2 + 3 * dp2 / 6 + 0.0) / 3) < 1e-4, f"loss with pre-pruning counts: sample 1 keeps 6 as its denominator, the fully pruned sample 2 contributes 0 and counts -> {float(l_den3):.4f}")

# ---- 11. through the pair loader: the window's pre-pruning pair count, the truncated window's last flag, the collate ----
names = ktjd17_split_names(ROOT, exclude=CUT)
captured = []
_orig_amc = AUGM.apply_motion_with_contact
def spy_amc(x18, tr_):
    out_, cb_ = _orig_amc(x18, tr_)
    if x18.ndim == 3: captured.append((tr_, out_, cb_))
    return out_, cb_
AUGM.apply_motion_with_contact = spy_amc
ds2 = InContextPairs(base, names["train"], names["train"], demo_frames=1, target_frames=240, balance_skeletons=True, seed=5,
                     emit_fk_fields=True, emit_graph_v2=True, identity_p=0.0, emit_ref_text=False, demo_rest=True, augment=one)
n_aug = n_trunc = 0; item_pos = item_zero = None
for i in range(1500):
    n_before = len(captured)
    item = ds2[i]
    if len(captured) == n_before or captured[-1][2] is None:
        check(float(item["lock_denominator"]) == 0.0, f"loader item {i}: no pruning -> lock_denominator 0")
        if item_zero is None: item_zero = item
        continue
    tr_, out_, cb_ = captured[-1]; n_aug += 1
    if item_pos is None and float(item["lock_denominator"]) > 0 and float(item["lock_denominator"]) > pairs_of((out_[:, :, 12] * (tr_.sd[None, :, 12] + _STD_FLOOR) + tr_.mu[None, :, 12]) > 0.5):
        item_pos = item                                                            # a pruned item: pre-pruning pairs > surviving pairs
    xt_ = item["x"][item["is_target"]].numpy(); n_served = xt_.shape[0]; T_clip = cb_.shape[0]
    flags = (xt_[..., 12] * (tr_.sd[None, :, 12] + _STD_FLOOR) + tr_.mu[None, :, 12]) > 0.5
    out_flags = (out_[:n_served, :, 12] * (tr_.sd[None, :, 12] + _STD_FLOOR) + tr_.mu[None, :, 12]) > 0.5
    cbw = cb_[:n_served]
    check(float(item["lock_denominator"]) == pairs_of(cbw), f"loader item {i}: lock_denominator == the window's pre-pruning pairs ({pairs_of(cbw)})")
    if n_served < T_clip:
        n_trunc += 1
        check(np.array_equal(flags[:-1], out_flags[:-1]) and np.array_equal(flags[-1], cbw[-1]),
              f"loader item {i} (truncated {T_clip}->{n_served}): pruned flags served, the window's last frame restored to its served flag")
    else:
        check(np.array_equal(flags, out_flags), f"loader item {i}: pruned flags served")
    if n_aug >= 60 and n_trunc >= 3: break
AUGM.apply_motion_with_contact = _orig_amc
check(n_aug >= 60 and n_trunc >= 3, f"loader: {n_aug} pruned items seen, {n_trunc} of them truncated windows")
check(item_pos is not None and item_zero is not None, "loader: a pruned item with a positive denominator and a plain item were seen")
if item_pos is not None and item_zero is not None:
    # collate -> fk_pack_of -> the loss's own inputs (cfm_loss builds contact_on from x1 and forwards fk_pack["lock_denominator"])
    bt = collate([item_pos, item_zero]); pack = trm.fk_pack_of(bt)
    check(torch.equal(pack["lock_denominator"], torch.tensor([float(item_pos["lock_denominator"]), 0.0])) and float(item_pos["lock_denominator"]) > 0,
          f"collate + fk_pack_of carry the denominators ({float(item_pos['lock_denominator']):.0f}, 0)")
    x1 = bt["x"][..., :17]; real = bt["is_target"]
    con_b = (x1[..., 12] * (pack["anytop_std"][:, None, :, 12] + pack["std_floor"]) + pack["anytop_mean"][:, None, :, 12]) > 0.5
    _, l_b, _, _ = ktjd_dynamics_losses(x1, x1, pack["anytop_mean"], pack["anytop_std"], pack["std_floor"], pack["rest_offsets"], pack["n_joints"],
                                        frame_mask=real, contact_on=con_b, want_diag=False, lock_denominator=pack["lock_denominator"])
    # the same quantity by hand: per sample the sum of squared displacements over surviving pairs of the real target frames,
    # divided by the pre-pruning count for the pruned item and by the surviving count for the plain one, averaged over the two
    manual = []
    for k, it_ in enumerate((item_pos, item_zero)):
        J = int(pack["n_joints"][k]); xt_ = x1[k][real[k]][:, :J].numpy()
        mu_, sd_ = pack["anytop_mean"][k, :J, :17].numpy(), pack["anytop_std"][k, :J, :17].numpy()   # the pack's stats carry plane 17
        raw_ = xt_ * (sd_[None] + _STD_FLOOR) + mu_[None]; w_ = raw_[..., 0:3].copy(); w_[..., 0] += raw_[:, 0:1, 13]; w_[..., 2] += raw_[:, 0:1, 14]
        c_ = raw_[..., 12] > 0.5; pr_ = c_[1:] & c_[:-1]
        scale_ = FK_SCALE_FRAC * (np.linalg.norm(pack["rest_offsets"][k, 1:J].numpy(), axis=-1).mean() + 1e-3)
        num = float((((w_[1:] - w_[:-1]) / scale_) ** 2).sum(-1)[pr_].sum())
        den = float(it_["lock_denominator"]) if float(it_["lock_denominator"]) > 0 else max(int(pr_.sum()), 1)
        manual.append(num / den)
    check(abs(float(l_b) - float(np.mean(manual))) < 1e-3 * max(1.0, abs(float(np.mean(manual)))), f"collated batch through the loss == the hand computation ({float(l_b):.5f} vs {np.mean(manual):.5f})")
print(f"[loader] pruned items {n_aug}, truncated windows {n_trunc}")
print("PASS" if not fails else f"FAIL {len(fails)}")
sys.exit(1 if fails else 0)
