"""Tests for the kinematics-preserving operations of src/data/ktjd17_augment.py (bone_scale / pool_frac / add_p):
FK == direct positions after re-encoding, velocity = the codec's forward difference, the synthetic joint on its bone,
the rest demo at zero under rest normalisation, byte-identical v1 behaviour when the three fields are 0, and the
InContextPairs/collate/ktjd_prep/cfm_loss integration at J' = J+1.
usage: python scripts/_aug_dev/_test_augment_v2.py   (inside an allocation; ~2 min, CPU only)"""
import sys, os, json, importlib.util
import numpy as np, torch
sys.path.insert(0, os.getcwd())
from src.data.ktjd17_incontext import Ktjd17Base, ktjd17_split_names
from src.data.incontext_pairs import InContextPairs, collate
from src.data.ktjd17_augment import (AugConfig, make_transform, apply_motion, apply_semantics, hop_matrix,
                                     AUG_VERSION, AUG_VERSION_KIN)
from src.data.anytop_dataset import _STD_FLOOR
from src.data.ktjd17.codec import decode_column_cont6d, fk_from_global_rotations
from src.models.v2.fk_torch import fk_ktjd_consistency_loss

ROOT, CUT = "dataset/ktjd17_pzh312_noik_v2", "configs/pilot_animal_only_exclusions.json"
KW = dict(caption_emb_cache="data/noik_caption_llm2vec_v1", joint_semantics="data/joint_semantics_llm2vec_pzh312_v1.npz",
          texts_json="data/noik_pzh312_motion_texts_v1.json", percell_stats="data/noik_norm_stats_v2.npz",
          exclude_clips=CUT, random_caption=False)
bases = {"rest": Ktjd17Base(ROOT, normalization="rest", **KW), "percell": Ktjd17Base(ROOT, normalization="percell", **KW)}
names = ktjd17_split_names(ROOT, exclude=CUT)
FPS = 30.0
fails = []
def check(cond, msg):
    if not cond: fails.append(msg); print("FAIL:", msg)

def load(base, bi):
    it = base[int(bi)]
    J0, T = int(it["num_joints"]), int(it["num_frames"])
    x = np.asarray(it["anytop_x"])[:J0, :, :T].transpose(2, 0, 1)                       # [T,J,18]
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
    """global rotations from served deltas (identity where the six-vector is degenerate), as apply_motion decodes them"""
    d6 = raw[..., 3:9]; a1, a2 = d6[..., :3], d6[..., 3:]
    n1 = np.linalg.norm(a1, axis=-1); b1 = a1 / np.maximum(n1[..., None], 1e-12)
    n2 = np.linalg.norm(a2 - np.sum(b1 * a2, axis=-1, keepdims=True) * b1, axis=-1)
    ok = (n1 > 1e-6) & (n2 > 1e-6)
    D = np.tile(np.eye(3), d6.shape[:2] + (1, 1))
    if ok.any(): D[ok] = decode_column_cont6d(d6[ok], strict=True)
    return D @ R_rest[None]

def denorm(xn, tr):
    return xn[..., :17].astype(np.float64) * (tr.sd[None] + _STD_FLOOR) + tr.mu[None]

cfgs = {"scale": AugConfig(p=1.0, bone_scale=0.1),
        "pool": AugConfig(p=1.0, pool_frac=1.0),
        "add": AugConfig(p=1.0, add_p=1.0),
        "all": AugConfig(p=1.0, drop_max_frac=0.3, drop_mode="tips", rest_deg=15.0, sem_noise=0.1, stats_logsd=0.2, stats_shift=0.3,
                         bone_scale=0.1, pool_frac=0.3, add_p=1.0)}
rng = np.random.default_rng(7)
idxs = list(rng.choice(len(bases["rest"]), size=10, replace=False))
# the four rigs whose statistics artifact excludes near-root position / velocity cells as exact constants
# (runs/_aug_dev/excluded_cells_scan.json): one item of each, so the re-entered cells are exercised
_want = {"PZ_Blue_Wildebeest_Female", "PZ_Blue_Wildebeest_Male", "PZ_Bongo_Juvenile", "PZ_Red_River_Hog_Female"}
for i, r in enumerate(bases["rest"]._rows):
    if r["rig_id"] in _want:
        idxs.append(i); _want.discard(r["rig_id"])
assert not _want, _want

def official_fk_gap(xn, tr):
    """the trainer's own FK-consistency mirror (src/models/v2/fk_torch.fk_ktjd_consistency_loss) on the served sample AFTER the
    trainer's state projection (invalid cells held at exact zero, as ktjd_prep / cfm_loss do); returns the mean |FK - direct|
    in bone lengths"""
    x = torch.from_numpy(np.ascontiguousarray(xn[..., :17])).float().clone()
    cv = torch.from_numpy(tr.channel_valid)
    x[:, ~cv] = 0.0
    T, Jn = x.shape[:2]
    _, diag = fk_ktjd_consistency_loss(x[None], torch.from_numpy(tr.mu)[None].float(), torch.from_numpy(tr.sd)[None].float(), _STD_FLOOR,
                                       torch.from_numpy(tr.parents.astype(np.int64))[None], torch.from_numpy(tr.offsets)[None].float(),
                                       torch.from_numpy(tr.R_rest)[None].float(), torch.tensor([Jn]), torch.ones(1, T, dtype=torch.bool), want_diag=True)
    return float(diag)
stats = {k: {"fk": [], "fk_official": [], "orig_pos": [], "n_pooled": 0, "n_added": 0} for k in cfgs}
for norm, base in bases.items():
  for bi in idxs:
    it, J0, T, x, mu0, sd0, cv0, raw0, contact, geom = load(base, bi)
    par0, O0, R0 = geom["parents"], geom["offset_parent_local"], geom["R_rest_global"]
    children = [[] for _ in range(J0)]
    for j in range(1, J0): children[par0[j]].append(j)
    bl0 = np.linalg.norm(O0[1:], axis=-1).mean()
    sem = np.asarray(it["joint_semantics"])[:J0]
    for name, cfg in cfgs.items():
        tr = make_transform(rng, cfg, **geom, channel_valid=cv0, mu=mu0, sd=sd0, contact_joints=contact, fps=FPS, rest_norm=(norm == "rest"))
        Jn, keep, rows = tr.n_joints, tr.keep, tr.served_rows
        tag = f"[{norm}/{name}/{it['object_type']}]"
        # ---- structure ----
        check(tr.parents[0] == -1 and np.all(tr.parents[1:] < np.arange(1, Jn)) and np.all(tr.parents[1:] >= 0), tag + " FK order")
        check(keep[0] == 0 and np.all(np.diff(keep) > 0) and np.array_equal(tr.src[rows], keep), tag + " keep/src/served_rows consistent")
        synth = np.where(tr.src < 0)[0]
        check(len(synth) == (1 if cfg.add_p > 0 else 0), tag + f" synthetic count {len(synth)}")
        check(tr.reencode == (cfg.bone_scale > 0 or cfg.add_p > 0 or (cfg.pool_frac > 0 and Jn < J0)), tag + " reencode flag")
        check(all(int(j) in set(keep.tolist()) for j in np.where(contact)[0]), tag + " contact joints protected")
        check(tr.mu.shape == (Jn, 17) and tr.sd.shape == (Jn, 17) and tr.channel_valid.shape == (Jn, 17) and tr.R_rest.shape == (Jn, 3, 3)
              and tr.offsets.shape == (Jn, 3) and tr.Q.shape == (Jn, 3, 3) and tr.R_rest_old.shape == (Jn, 3, 3), tag + " table shapes")
        # ---- motion ----
        xn = apply_motion(x, tr)
        check(xn.shape == (T, Jn, 18) and np.isfinite(xn).all(), tag + " output shape / finite")
        check(np.array_equal(xn[:, rows, 17], x[:, keep, 17]), tag + " plane 17 copied on originals")
        for s in synth: check(np.array_equal(xn[:, s, 17], xn[:, tr.parents[s], 17]), tag + " plane 17 synthetic = parent's")
        rawn = denorm(xn, tr); wn = world(rawn); w0 = world(raw0)
        bl = np.linalg.norm(tr.offsets[1:], axis=-1).mean()
        Gn = G_of(rawn, tr.R_rest); G0 = G_of(raw0, R0)
        check(np.abs(Gn[:, rows] - G0[:, keep]).max() < 1e-5, tag + f" global rotations of originals invariant (max {np.abs(Gn[:, rows] - G0[:, keep]).max():.2e})")
        for s in synth: check(np.abs(Gn[:, s] - Gn[:, tr.parents[s]]).max() < 1e-5, tag + " synthetic global rotation = parent's")   # float32 rows
        fk = fk_from_global_rotations(tr.parents, wn[:, 0], Gn, tr.offsets)
        gap = np.abs(fk - wn).max() / bl; stats[name]["fk"].append(gap)
        check(gap < 1e-5, tag + f" FK == direct after re-encoding (max {gap:.2e} bl)")    # float32 serving: ~1e-6 bl
        og = official_fk_gap(xn, tr); stats[name]["fk_official"].append(og)
        check(og < 1e-4, tag + f" official FK mirror after the state projection (mean {og:.2e} bl)")
        if tr.reencode:
            check(tr.channel_valid[:, 0:3].all() and tr.channel_valid[:, 9:12].all(), tag + " re-encoded sample: position/velocity cells all valid")
            check((tr.sd[:, 0:3] > 1e-3).all() and (tr.sd[:, 9:12] > 1e-3).all(), tag + " re-encoded sample: usable position/velocity scales")
        else:
            check(np.array_equal(tr.channel_valid, cv0[keep]), tag + " untouched sample keeps the rig's mask")
        # velocity = the codec's forward difference of WORLD positions, last repeated (a re-encoded sample; the corpus itself
        # keeps a trimmed clip's last-frame velocity, so an untouched sample is compared with the served channels instead)
        if tr.reencode:
            vexp = np.zeros_like(wn)
            if T >= 2: vexp[:-1] = (wn[1:] - wn[:-1]) * FPS; vexp[-1] = vexp[-2]
            dv = np.abs(rawn[..., 9:12] - vexp).max()
            check(dv < 1e-5 * FPS, tag + f" velocity = forward difference (max {dv:.2e})")
        else:
            check(np.abs(rawn[:, rows, 9:12] - raw0[:, keep, 9:12]).max() < 2e-4 and np.abs(wn[:, rows] - w0[:, keep]).max() < 2e-4,
                  tag + " untouched sample keeps its served positions / velocities")
        # root track / heading / contact untouched; contact of synthetic zero
        check(np.abs(rawn[:, 0, 13:17] - raw0[:, 0, 13:17]).max() < 1e-4, tag + " root channels 13:17 unchanged")
        check(np.abs(rawn[:, rows, 12] - raw0[:, keep, 12]).max() < 2e-4, tag + " contact of originals unchanged")
        for s in synth: check(np.abs(rawn[:, s, 12]).max() < 2e-4, tag + " synthetic contact 0")
        # the originals' positions: unchanged unless a bone was scaled or a chain was pooled
        dpos = np.abs(wn[:, rows] - w0[:, keep]).max() / bl0; stats[name]["orig_pos"].append(dpos)
        if name == "add": check(dpos < 1e-5, tag + f" add-only: original positions unchanged (max {dpos:.2e} bl)")
        # ---- op-specific ----
        if name == "scale":
            check(np.array_equal(tr.parents, par0), tag + " scale-only keeps the tree")
            n_old = np.linalg.norm(O0[1:], axis=-1); nz = n_old > 1e-9                     # zero-length bones stay zero
            r = np.linalg.norm(tr.offsets[1:], axis=-1)[nz] / n_old[nz]
            check(r.min() >= 0.9 - 1e-9 and r.max() <= 1.1 + 1e-9 and r.std() > 1e-3, tag + f" bone factors in [0.9,1.1] (min {r.min():.3f} max {r.max():.3f})")
            check(np.all(np.linalg.norm(tr.offsets[1:], axis=-1)[~nz] == 0), tag + " zero-length bones stay zero")
            check(np.abs(wn[:, 0] - w0[:, 0]).max() < 1e-9, tag + " root position unchanged")
        if name == "pool":
            removed = sorted(set(range(J0)) - set(keep.tolist())); stats[name]["n_pooled"] += len(removed)
            check(all(j != 0 and not contact[j] and len(children[j]) == 1 for j in removed), tag + f" pooled joints are single-child interior non-contact ({len(removed)})")
        if name in ("add", "all"):
            s = int(synth[0]); p = int(tr.parents[s]); c = s + 1
            check(tr.parents[c] == s, tag + " child hangs from the synthetic joint")
            check(tr.synth_pc[s, 0] == keep[p] if p < s else True, tag + " synth_pc parent index")
            a = np.linalg.norm(tr.offsets[s]) / (np.linalg.norm(tr.offsets[s]) + np.linalg.norm(tr.offsets[c]))
            check(0.3 - 1e-9 <= a <= 0.7 + 1e-9, tag + f" alpha in [0.3,0.7] ({a:.3f})")
            d = np.abs((wn[:, s] - wn[:, p]) - a * (wn[:, c] - wn[:, p])).max() / bl
            check(d < 1e-5, tag + f" synthetic joint on the parent->child segment (max {d:.2e} bl)")
            check(np.allclose(tr.R_rest[s], tr.R_rest[p]) and np.allclose(tr.Q[s], tr.Q[p]) and np.allclose(tr.R_rest_old[s], tr.R_rest_old[p]),
                  tag + " synthetic rest rotation / Q = parent's")
            cvs, sds, mus = tr.channel_valid, tr.sd0, tr.mu0
            check(np.array_equal(cvs[s, :3], cvs[c, :3]) and np.array_equal(cvs[s, 9:], cvs[c, 9:]) and np.array_equal(cvs[s, 3:9], cvs[p, 3:9])
                  and np.array_equal(sds[s, :3], sds[c, :3]) and np.array_equal(sds[s, 9:], sds[c, 9:]) and np.array_equal(sds[s, 3:9], sds[p, 3:9])
                  and np.array_equal(mus[s, 3:9], mus[p, 3:9]), tag + " synthetic mask/stats: child's rows, parent's rotation cells")
            stats[name]["n_added"] += 1
        # ---- rest demo ----
        rest = base.rest_frame_normalized(it["object_type"])[:J0]
        rn = apply_motion(rest, tr); rraw = rn[:, :17].astype(np.float64) * (tr.sd + _STD_FLOOR) + tr.mu
        qr = np.where(tr.rot_rows)[0]
        check(np.abs(decode_column_cont6d(rraw[qr, 3:9]) - np.swapaxes(tr.Q[qr], -1, -2)).max() < 1e-5, tag + " rest demo rotations = Q^T")
        Pd = fk_from_global_rotations(tr.parents, rraw[None, 0, 0:3], tr.R_rest_old[None], tr.offsets)[0]
        check(np.abs(rraw[:, 0:3] - Pd).max() < 1e-6, tag + " rest demo positions = FK of the unperturbed rest through the new bones")
        if norm == "rest" and cfg.stats_shift == 0 and cfg.stats_logsd == 0:   # (a statistics shift moves the demo, as in v1)
            cvn = tr.channel_valid
            z = np.abs(rn[:, :3][cvn[:, :3]]).max()
            check(z < 1e-4, tag + f" rest demo positions normalise to zero under rest normalisation (max {z:.2e})")
            check(np.abs(rn[:, 9:13][cvn[:, 9:13]]).max() < 1e-6, tag + " rest demo velocity/contact zero")
        # ---- semantics ----
        sn = apply_semantics(sem, tr, np.random.default_rng(0))
        check(sn.shape == (Jn, sem.shape[1]), tag + " sem shape")
        if cfg.sem_noise == 0:
            check(np.array_equal(sn[rows], sem[keep]), tag + " originals' semantics copied")
            for s in synth:
                pj, cj = tr.synth_pc[s]
                check(np.allclose(sn[s], 0.5 * (sem[pj] + sem[cj])), tag + " synthetic semantics = mean(parent, child)")
        geo = hop_matrix(tr.parents); check(geo.shape == (Jn, Jn), tag + " hop matrix")
for k, v in stats.items():
    print(f"[{k}] FK gap max {max(v['fk']):.2e} bl | official mirror max {max(v['fk_official']):.2e} bl | original-position change max {max(v['orig_pos']):.2e} bl | pooled {v['n_pooled']} | added {v['n_added']}")

# ---------------- pooling on the contracted tree (codex v2 r1 P2-3) + degenerate rotations refused (P2-4) ----------------
def synth(parents):
    par = np.asarray(parents); J = len(par)
    O = np.zeros((J, 3)); O[1:, 0] = 1.0                                  # unit bones along x, identity rest rotations
    P = np.zeros((J, 3))
    for j in range(1, J): P[j] = P[par[j]] + O[j]
    return dict(parents=par, P_rest_global=P, R_rest_global=np.tile(np.eye(3), (J, 1, 1)), offset_parent_local=O,
                channel_valid=np.ones((J, 17), bool), mu=np.zeros((J, 17), np.float32), sd=np.ones((J, 17), np.float32))
# (i) codex's counter-example (round-3 fixture): joint 1 has children 2 and 3, joint 2 has child 4; dropping joint 2 ("any")
#     leaves joint 1 with ONE immediate kept child (3) but TWO contracted children (3 and 4) -- the original selector, which
#     counted immediate kept children, pooled it; it must stay (unprotected on purpose)
g = synth([-1, 0, 1, 1, 2]); prot = np.array([False, False, False, True, True])
n_case = 0
for seed in range(60):
    cfg_dp = AugConfig(p=1.0, drop_max_frac=1.0, drop_mode="any", pool_frac=1.0)
    tr = make_transform(np.random.default_rng(seed), cfg_dp, **g, contact_joints=prot)
    # the drop step draws first and identically whether or not pooling follows: the same seed with pooling off tells
    # which joints the DROP removed, so a joint kept there and gone here was POOLED
    tr_drop = make_transform(np.random.default_rng(seed), AugConfig(p=1.0, drop_max_frac=1.0, drop_mode="any"), **g, contact_joints=prot)
    pooled = sorted(set(tr_drop.keep.tolist()) - set(tr.keep.tolist()))
    if 2 not in tr_drop.keep and 1 in tr_drop.keep:               # exactly the counter-example: joint 2 dropped, joint 1 kept
        n_case += 1
        check(pooled == [], f"[pool/contracted seed {seed}] joint(s) {pooled} pooled although joint 1's contracted degree is two")
        check(np.array_equal(tr.parents, [-1, 0, 1, 1]), f"[pool/contracted seed {seed}] contracted parents {tr.parents.tolist()}")
    for j in pooled:                                              # any pooled joint had exactly one child in the contracted tree
        kept = tr_drop.keep.tolist(); pos = {int(v): i for i, v in enumerate(kept)}
        check(sum(1 for c in kept if c != 0 and tr_drop.parents[pos[c]] == pos[j]) == 1, f"[pool seed {seed}] pooled joint {j} was not single-child")
check(n_case > 0, "[pool/contracted] no seed dropped joint 2 while keeping joint 1 (precondition)")
# (ii) a chain: sequential pooling follows the child lists (every interior joint is a candidate; the tree stays a chain)
g = synth([-1, 0, 1, 2, 3, 4]); prot = np.array([False] * 6); prot[5] = True
n_pooled = []
for seed in range(40):
    tr = make_transform(np.random.default_rng(seed), AugConfig(p=1.0, pool_frac=1.0), **g, contact_joints=prot)
    n_pooled.append(6 - tr.n_joints)
    check(tr.parents.tolist() == [-1] + list(range(tr.n_joints - 1)), f"[pool/chain seed {seed}] not a chain: {tr.parents.tolist()}")
    check(0 in tr.keep and 5 in tr.keep, f"[pool/chain seed {seed}] root / protected leaf lost")
    if tr.reencode:
        # the pooled bones add up: the leaf stays 5 units from the root
        check(abs(np.linalg.norm(tr.offsets[1:], axis=-1).sum() - 5.0) < 1e-9, f"[pool/chain seed {seed}] bone lengths do not add up")
check(max(n_pooled) >= 3 and min(n_pooled) == 0, f"[pool/chain] pooled counts over seeds {sorted(set(n_pooled))} (expected 0..4)")
# (iii) a degenerate six-vector on a served row is refused when re-encoding, and passes through on the v1 path
g = synth([-1, 0, 1]); x = np.zeros((4, 3, 18), np.float32); x[..., 3] = 1.0; x[..., 7] = 1.0; x[..., 17] = 1.0   # identity deltas
x[2, 1, 3:9] = 0.0                                                                                         # zero six-vector, joint 1, frame 2
tr = make_transform(np.random.default_rng(0), AugConfig(p=1.0, bone_scale=0.1), **g, contact_joints=np.zeros(3, bool))
try:
    apply_motion(x, tr); check(False, "[degenerate] re-encoding accepted a zero six-vector")
except ValueError as e:
    check("degenerate" in str(e), f"[degenerate] wrong error: {e}")
tr1 = make_transform(np.random.default_rng(0), AugConfig(p=1.0, rest_deg=10.0), **g, contact_joints=np.zeros(3, bool))
check(np.array_equal(apply_motion(x, tr1)[2, 1, 3:9], x[2, 1, 3:9]), "[degenerate] v1 path passes the cell through unchanged")

# ---------------- v1 parity: the three fields at 0 reproduce the v1 module byte for byte ----------------
from importlib.machinery import SourceFileLoader
_ld = SourceFileLoader("aug_v1", "src/data/ktjd17_augment.py.bak_20260913")       # a .bak suffix needs an explicit loader
v1 = importlib.util.module_from_spec(importlib.util.spec_from_loader("aug_v1", _ld))
sys.modules["aug_v1"] = v1; _ld.exec_module(v1)                                        # dataclass looks the module up
p_v2 = AugConfig(p=1.0, drop_max_frac=0.4, rest_deg=15.0, sem_noise=0.1, sem_drop_p=0.2, stats_logsd=0.2, stats_shift=0.2)
p_v1 = v1.AugConfig(p=1.0, drop_max_frac=0.4, rest_deg=15.0, sem_noise=0.1, sem_drop_p=0.2, stats_logsd=0.2, stats_shift=0.2)
check(p_v2.protocol() == p_v1.protocol() and p_v2.protocol()["version"] == AUG_VERSION, "protocol identical to v1 with the three fields at 0")
pk = AugConfig(p=1.0, add_p=0.5).protocol()
check(pk["version"] == AUG_VERSION_KIN and pk["add_p"] == 0.5 and "bone_scale" in pk, "kinematic protocol carries the version and fields")
n_eq = 0
for norm, base in bases.items():
    for bi in idxs[:5]:
        it, J0, T, x, mu0, sd0, cv0, raw0, contact, geom = load(base, bi)
        sem = np.asarray(it["joint_semantics"])[:J0]
        for mode in ("any", "tips"):
            c2 = AugConfig(p=1.0, drop_max_frac=0.4, drop_mode=mode, rest_deg=15.0, sem_noise=0.1, sem_drop_p=0.2, stats_logsd=0.2, stats_shift=0.2)
            c1 = v1.AugConfig(p=1.0, drop_max_frac=0.4, drop_mode=mode, rest_deg=15.0, sem_noise=0.1, sem_drop_p=0.2, stats_logsd=0.2, stats_shift=0.2)
            for seed in (1, 2):
                t2 = make_transform(np.random.default_rng(seed), c2, **geom, channel_valid=cv0, mu=mu0, sd=sd0, contact_joints=contact, fps=FPS, rest_norm=(norm == "rest"))
                t1 = v1.make_transform(np.random.default_rng(seed), c1, **geom, channel_valid=cv0, mu=mu0, sd=sd0, contact_joints=contact)
                same = (np.array_equal(t2.keep, t1.keep) and np.array_equal(t2.parents, t1.parents) and np.array_equal(t2.offsets, t1.offsets)
                        and np.array_equal(t2.R_rest, t1.R_rest) and np.array_equal(t2.mu, t1.mu) and np.array_equal(t2.sd, t1.sd)
                        and t2.sem_drop == t1.sem_drop and not t2.reencode)
                same &= np.array_equal(apply_motion(x, t2), v1.apply_motion(x, t1))
                same &= np.array_equal(apply_semantics(sem, t2, np.random.default_rng(seed)), v1.apply_semantics(sem, t1, np.random.default_rng(seed)))
                rest = base.rest_frame_normalized(it["object_type"])[:J0]
                same &= np.array_equal(apply_motion(rest, t2), v1.apply_motion(rest, t1))
                n_eq += int(same)
                check(same, f"v1 parity [{norm}/{mode}/{it['object_type']}/seed{seed}]")
print(f"[parity] {n_eq}/40 transforms byte-identical to v1")

# ---------------- integration through the pair dataset at J' = J+1 (the aug arm's setting: rest normalisation) ----------------
base = bases["rest"]
ds = InContextPairs(base, names["train"], names["train"], demo_frames=1, target_frames=240, balance_skeletons=True, seed=0,
                    emit_fk_fields=True, emit_graph_v2=True, identity_p=0.0, emit_ref_text=False, demo_rest=True, augment=cfgs["all"])
items = [ds[i] for i in range(8)]
for it in items:
    J = it["n_joints"]
    check(it["channel_valid"].shape == (J, 17), "item channel_valid")
    check(it["parents"].shape[0] == J and it["rest_offsets"].shape == (J, 3) and it["R_rest_global"].shape == (J, 3, 3), "fk fields sized to J'")
    check(it["struct_feats"].shape == (J, 8) and it["updown"].shape == (J, J, 2) and it["joint_sem"].shape[0] == J, "graph/sem sized to J'")
    check(it["x"].shape[1] == J and it["geodesic"].shape == (J, J), "x/geodesic sized to J'")
    check(it["anytop_mean"].shape == (J, 18) and it["anytop_std"].shape == (J, 18), "stats sized to J'")
    check(torch.isfinite(it["x"]).all(), "item x finite")
# the served demo frame (rest) is zero on valid position cells under rest normalisation -- without the statistics
# perturbation, which shifts the demo by design (v1 semantics); the kinematic operations alone must leave it at zero
cfg_kin = AugConfig(p=1.0, drop_max_frac=0.3, drop_mode="tips", rest_deg=15.0, sem_noise=0.1, bone_scale=0.1, pool_frac=0.3, add_p=1.0)
ds_kin = InContextPairs(base, names["train"], names["train"], demo_frames=1, target_frames=240, balance_skeletons=True, seed=1,
                        emit_fk_fields=True, emit_graph_v2=True, identity_p=0.0, emit_ref_text=False, demo_rest=True, augment=cfg_kin)
for i in range(8):
    it = ds_kin[i]
    d = it["x"][~it["is_target"]][0]                                 # [J,C] the demo frame
    cvp = it["channel_valid"][:, :3]
    check(float(d[:, :3][cvp].abs().max()) < 1e-3, f"served rest demo ~0 on positions (max {float(d[:, :3][cvp].abs().max()):.2e})")
    check(it["n_joints"] == it["x"].shape[1] and it["parents"].shape[0] == it["n_joints"], "kin item sized to J'")
b = collate(items[:4])
check(b["channel_valid"].shape == (4, b["x"].shape[2], 17), "collated channel_valid")
spec = importlib.util.spec_from_file_location("trainer", "scripts/train_v2_incontext.py"); tr_mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(tr_mod)
lut = tr_mod.ktjd_channel_lut(base)
calib = json.load(open("configs/pilot_animal_rest_gamma_calibration_b16_v2.json"))
gammas = {k: float(v) for k, v in calib["gammas"].items()}
x17, kt = tr_mod.ktjd_prep(b, lut, gammas)
check(torch.equal(kt["channel_valid"], b["channel_valid"]), "ktjd_prep uses per-sample channel_valid")
from src.models.v2.dit_motion import InContextMotionDiT, cfm_loss
torch.manual_seed(0)
model = InContextMotionDiT(in_ch=17, dim=64, depth=1, n_heads=2, use_struct_feats=True, use_dir_bias=True, qk_norm=True)
loss = cfm_loss(model, x17, is_target=b["is_target"], valid=b["valid"], t_sampler="uniform", v_space=True, sigma_min=0.2, huber_delta=10.0,
                gamma_fk=0.07, gamma_vel=0.01, gamma_lock=0.01, gamma_acc=1.0, fk_pack=tr_mod.fk_pack_of(b), **kt, **tr_mod.cond_of(b))
check(torch.isfinite(loss).item(), f"cfm_loss finite ({loss.item()})")
loss.backward(); check(all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()), "grads finite")
print("[integration] loss", float(loss), "| Jm", b["x"].shape[2], "| n_joints", b["n_joints"].tolist())
print("RESULT:", "PASS" if not fails else f"FAIL ({len(fails)})")
sys.exit(1 if fails else 0)
