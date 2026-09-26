"""R1/R2 (2026-09-25) -- the world-frame rest descriptor in struct_feats (InContextPairs / InContextMotionDiT
struct_world_rest: 8 -> 14 columns) and the rest-convention channel of the one_of augmentation (AugConfig.rest_p,
make_transform skip_op / rest). The OFF path is checked BYTE-FOR-BYTE against the pre-change snapshot
(/iridisfs/scratch/ts1v23/workspace/noKslot_pre_r2_ref, read-only) through a `--dump` subprocess run inside that tree.
CPU only, inside an allocation:
  srun --jobid=<alloc> --overlap -N1 -n1 --cpus-per-task=2 /usr/bin/env CUDA_VISIBLE_DEVICES= SCRATCH_TMP=<shared dir> \\
       /iridisfs/scratch/ts1v23/.conda/bin/python scripts/_aug_dev/_test_struct_world_rest.py"""
import sys, os, json, subprocess, importlib.util, importlib.machinery, time
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, ".")
torch.set_num_threads(2)

MAIN = os.environ.get("SWR_TEST_MAIN", "/iridisfs/scratch/ts1v23/workspace/noKslot_pre_r2_ref")   # pre-change snapshot, never written
ROOT = f"{MAIN}/dataset/ktjd17_pzh312_noik_v2"
CUT = f"{MAIN}/configs/heldout20_v1_exclusions.json"
KW = dict(caption_emb_cache=f"{MAIN}/data/noik_caption_llm2vec_v1",
          joint_semantics=f"{MAIN}/data/joint_semantics_llm2vec_pzh312_v1.npz",
          texts_json=f"{MAIN}/data/noik_pzh312_motion_texts_v1.json",
          percell_stats=f"{MAIN}/data/noik_norm_stats_v2.npz", exclude_clips=CUT, random_caption=False)
DSK = dict(demo_frames=1, target_frames=240, balance_skeletons=True, seed=0, emit_fk_fields=True, emit_graph_v2=True,
           demo_rest=True, emit_spectral=8, spectral_hks=True)        # the H1 recipe's loader
KWM = dict(in_ch=17, dim=192, depth=2, n_heads=6, d_text=4096, d_joint_sem=4096, use_struct_feats=True, use_dir_bias=True,
           qk_norm=True, use_geo_bias=True, use_spec_rope=True, spec_rope_k=8, spec_rope_hks=True, use_temporal_rope=True)
ONE = dict(p=1.0, drop_mode="tips", bone_scale=0.1, mode="one_of")
N_ITEMS, N_AUG = 12, 6


def item_indices(n, k):
    return list(range(0, n, max(1, n // k)))[:k]


def dump_reference(out):
    """Run INSIDE the snapshot (cwd + sys.path[0] = MAIN): served items (plain and one_of-augmented), a collated batch and
    the model state dict of the pre-change code, through the APIs both trees share."""
    from src.data.ktjd17_incontext import Ktjd17Base, ktjd17_split_names
    from src.data.incontext_pairs import InContextPairs, collate
    from src.data.ktjd17_augment import AugConfig
    from src.models.v2.dit_motion import InContextMotionDiT
    base = Ktjd17Base(ROOT, normalization="rest", **KW)
    names = ktjd17_split_names(ROOT, exclude=CUT)
    ds = InContextPairs(base, names["train"], names["train"], **DSK)
    dsa = InContextPairs(base, names["train"], names["train"], augment=AugConfig(**ONE), **DSK)
    idx = item_indices(len(ds), N_ITEMS); idxa = item_indices(len(dsa), N_AUG)
    items = [ds[i] for i in idx]; itemsa = [dsa[i] for i in idxa]
    batch = collate(items[:3])
    torch.manual_seed(0); m = InContextMotionDiT(**KWM)
    torch.save({"idx": idx, "idxa": idxa, "n": len(ds), "items": items, "itemsa": itemsa, "batch": batch, "sd": m.state_dict(),
                "aug_protocol": AugConfig(**ONE).protocol()}, out)
    print(f"[dump] {len(items)} items, {len(itemsa)} augmented, batch, state dict -> {out}")


if len(sys.argv) > 2 and sys.argv[1] == "--dump":
    dump_reference(sys.argv[2]); sys.exit(0)

REPO = Path(".").resolve()
assert str(REPO) != MAIN, "run this from the main tree, not from the snapshot"
TMP = Path(os.environ.get("SCRATCH_TMP", str(REPO / "_aug_dev_tmp"))); TMP.mkdir(parents=True, exist_ok=True)
PY = sys.executable
from src.data.incontext_pairs import InContextPairs, collate, _world_rest_feats, _graph_v2_tables
from src.data.ktjd17_incontext import Ktjd17Base, ktjd17_split_names
from src.data.ktjd17_augment import AugConfig, make_transform, apply_motion, REST_RULE
from src.models.v2.dit_motion import InContextMotionDiT
from src.data.anytop_dataset import _STD_FLOOR
import scripts.train_v2_incontext as trm

fails = []
def check(cond, msg):
    if not cond: fails.append(msg); print("FAIL:", msg)
    else: print("ok:", msg)
T0 = time.time()


def same(a, b):
    if torch.is_tensor(a) or torch.is_tensor(b):
        return torch.is_tensor(a) and torch.is_tensor(b) and a.dtype == b.dtype and a.shape == b.shape and torch.equal(a, b)
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        return isinstance(a, np.ndarray) and isinstance(b, np.ndarray) and a.dtype == b.dtype and np.array_equal(a, b)
    if isinstance(a, dict):
        return isinstance(b, dict) and a.keys() == b.keys() and all(same(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return type(a) is type(b) and len(a) == len(b) and all(same(x, y) for x, y in zip(a, b))
    return a == b


def fk_rest(parents, offsets, R, root):
    P = np.zeros((len(parents), 3)); P[0] = root
    for j in range(1, len(parents)):
        P[j] = P[parents[j]] + R[parents[j]] @ offsets[j]
    return P


# ---------------- reference dump from the snapshot ----------------
REF = TMP / "_tmp_swr_reference.pt"
if REF.exists(): REF.unlink()
r = subprocess.run([PY, str(REPO / "scripts/_aug_dev/_test_struct_world_rest.py"), "--dump", str(REF)], cwd=MAIN,
                   env=dict(os.environ, PYTHONPATH=MAIN, CUDA_VISIBLE_DEVICES="", PYTHONDONTWRITEBYTECODE="1"),
                   capture_output=True, text=True)
check(r.returncode == 0 and REF.exists(), f"reference dump inside the snapshot ({(r.stdout + r.stderr).strip()[-160:]})")
ref = torch.load(REF, map_location="cpu", weights_only=False)

base = Ktjd17Base(ROOT, normalization="rest", **KW)
names = ktjd17_split_names(ROOT, exclude=CUT)
ds_off = InContextPairs(base, names["train"], names["train"], **DSK)
ds_on = InContextPairs(base, names["train"], names["train"], struct_world_rest=True, **DSK)
dsa_off = InContextPairs(base, names["train"], names["train"], augment=AugConfig(**ONE), **DSK)
check(len(ds_off) == ref["n"], f"same target count as the snapshot ({len(ds_off)})")

# ================= A. OFF path: byte-for-byte against the snapshot =================
items_off = [ds_off[i] for i in ref["idx"]]
check(all(same(a, b) for a, b in zip(items_off, ref["items"])), f"A1 plain items bitwise identical to the snapshot ({len(items_off)} items, every key)")
itemsa_off = [dsa_off[i] for i in ref["idxa"]]
check(all(same(a, b) for a, b in zip(itemsa_off, ref["itemsa"])), f"A2 one_of-augmented items (rest_p 0) bitwise identical ({len(itemsa_off)} items: same ops, same RNG stream)")
check(same(collate(items_off[:3]), ref["batch"]), "A3 collate(3) bitwise identical")
torch.manual_seed(0); m_off = InContextMotionDiT(**KWM)
check(same({k: v for k, v in m_off.state_dict().items()}, {k: v for k, v in ref["sd"].items()}), "A4 model (struct_world_rest off) state dict bitwise identical under seed 0")
check(AugConfig(**ONE).protocol() == ref["aug_protocol"], "A5 one_of protocol record without the rest channel unchanged (older artifacts keep matching)")

# ================= B. the descriptor =================
SKEL = sorted(Path(f"{ROOT}/skeletons").glob("*.npz"))
pick = [SKEL[i] for i in np.linspace(0, len(SKEL) - 1, 50).round().astype(int)]
okB = True; worst_u = 0.0; worst_p = 0.0
for f in pick:
    z = np.load(f); par = z["parents"].astype(np.int64); off = z["offset_parent_local"]; R = z["R_rest_global"]; P = z["P_rest_global"]
    w = _world_rest_feats(par, off, P)
    okB &= w.dtype == np.float32 and w.shape == (len(par), 6)
    u = w[:, :3].astype(np.float64); nrm = np.linalg.norm(u, axis=1)
    d = P[1:] - P[par[1:]]; dn = np.linalg.norm(d, axis=1); nz = dn > 1e-8
    okB &= abs(nrm[0]) < 1e-12 and np.abs(nrm[1:][nz] - 1).max() < 1e-5 and (nrm[1:][~nz] == 0).all()   # root / zero bones zero, others unit
    worst_u = max(worst_u, float(np.abs(u[1:][nz] - d[nz] / dn[nz, None]).max()))            # u = direction of the rest bone
    blen = np.linalg.norm(off, axis=-1); mean_bone = blen[1:].mean()
    cc = w[:, 3:6].astype(np.float64); rr = np.linalg.norm(cc, axis=1)                                   # undo the radial log1p
    back = cc * np.where(rr > 0, np.expm1(rr) / np.maximum(rr, 1e-12), 1.0)[:, None] * mean_bone + np.array([P[0, 0], 0, P[0, 2]])
    worst_p = max(worst_p, float(np.abs(back - P).max() / max(mean_bone, 1e-6)))
    okB &= abs(w[0, 3]) < 1e-6 and abs(w[0, 5]) < 1e-6                         # root XZ removed, height kept
check(okB and worst_u < 1e-6 and worst_p < 1e-4 and w[:, 3:6].__abs__().max() < 4.0, f"B1 50 rigs: [J,6] float32, root row (0,0,0 | 0,y,0), bones unit (zero-length bones zero; u vs rest bone direction max dev {worst_u:.1e}), positions invert to P_rest through expm1 (max {worst_p:.1e} bl), compressed columns < 4")
items_on = [ds_on[i] for i in ref["idx"]]
okB2 = True
for a, b in zip(items_on, ref["items"]):
    ka, kb = set(a.keys()), set(b.keys()); okB2 &= ka == kb
    okB2 &= tuple(a["struct_feats"].shape) == (int(a["n_joints"]), 14) and same(a["struct_feats"][:, :8], b["struct_feats"])
    okB2 &= all(same(a[k], b[k]) for k in ka if k != "struct_feats")
    # recompute from the item's own FK fields; the rest root sits at the mean's root row (rest normalisation: (0, y, 0))
    par = a["parents"].numpy(); R = a["R_rest_global"].numpy().astype(np.float64); off = a["rest_offsets"].numpy().astype(np.float64)
    Pfk = fk_rest(par, off, R, a["anytop_mean"].numpy()[0, :3].astype(np.float64)); w = _world_rest_feats(par, off, Pfk); got = a["struct_feats"][:, 8:].numpy().astype(np.float64)
    dev_u = float(np.abs(got[:, :3] - w[:, :3]).max()); dev_p = float(np.abs(got[:, 3:] - w[:, 3:]).max())
    if not (dev_u < 1e-4 and dev_p < 1e-4): print(f"   B2 item {a.get('motion_id')}: dev_u {dev_u:.2e} dev_p {dev_p:.2e}")
    okB2 &= dev_u < 1e-4 and dev_p < 1e-4          # float32 item fields: a short bone's direction rounds at ~1e-5
check(okB2, "B2 struct_world_rest items: 14 columns whose first 8 are the snapshot's bytes, every other key bitwise identical, columns 8:14 recomputed from the item's own parents / rest_offsets / R_rest_global and the mean's rest root")
try:
    InContextPairs(base, names["train"], names["train"], struct_world_rest=True, **{**DSK, "emit_graph_v2": False}); check(False, "B3 struct_world_rest without emit_graph_v2 refused")
except ValueError as e:
    check(True, f"B3 struct_world_rest without emit_graph_v2 refused ({str(e)[:50]}...)")
b_on = collate(items_on[:3]); check(tuple(b_on["struct_feats"].shape) == (3, max(int(a["n_joints"]) for a in items_on[:3]), 14), "B4 collate pads the 14-column table")

# ================= C. the rest channel =================
for bad, kw in [("one_of rest_deg without rest_p", dict(ONE, rest_deg=30.0)), ("one_of rest_p without rest_deg", dict(ONE, rest_p=0.5)),
                ("joint mode with rest_p", dict(p=1.0, rest_deg=20.0, rest_p=0.5)), ("rest_p > 1", dict(ONE, rest_deg=30.0, rest_p=1.5))]:
    try: AugConfig(**kw); check(False, f"C1 {bad} refused")
    except ValueError as e: check(True, f"C1 {bad} refused ({str(e)[:60]}...)")
c_r2 = AugConfig(**dict(ONE, p=0.8, rest_deg=30.0, rest_p=0.5))
rec = c_r2.protocol()
check(rec["rest_p"] == 0.5 and rec["rest_deg"] == 30.0 and rec["rest_rule"] == REST_RULE and {k: v for k, v in rec.items() if k not in ("rest_p", "rest_deg", "rest_rule", "p")} == {k: v for k, v in ref["aug_protocol"].items() if k != "p"},
      "C2 R2 protocol = the one_of record + rest_p / rest_deg / rest_rule (nothing else changes)")
check(AugConfig(**dict(ONE, p=0.0, rest_deg=30.0, rest_p=1.0)).active and not AugConfig(**dict(ONE, p=0.0)).active, "C3 active: the rest channel alone counts as augmentation")
# transform level: rest-only one_of == legacy joint-mode rest on the same RNG; rest=False leaves R_rest; skip_op only in one_of
z = np.load(pick[7]); par = z["parents"].astype(np.int64); J = len(par)
geom = dict(parents=par, P_rest_global=z["P_rest_global"], R_rest_global=z["R_rest_global"], offset_parent_local=z["offset_parent_local"])
sm = base.static_masks(pick[7].stem); cv = np.asarray(sm["channel_valid"], dtype=bool)[:J]
mu = np.zeros((J, 17), np.float32); sd = np.ones((J, 17), np.float32)
t_new = make_transform(np.random.default_rng(11), c_r2, **geom, channel_valid=cv, mu=mu, sd=sd, contact_joints=np.zeros(J, bool), rest_norm=True, skip_op=True, rest=True)
t_old = make_transform(np.random.default_rng(11), AugConfig(p=1.0, rest_deg=30.0), **geom, channel_valid=cv, mu=mu, sd=sd, contact_joints=np.zeros(J, bool), rest_norm=True)
rot = t_new.rot_rows
check(t_new.op is None and same(t_new.parents, t_old.parents) and np.array_equal(t_new.offsets, t_old.offsets) and np.array_equal(t_new.Q[rot], t_old.Q[rot])
      and t_new.rest_channel and not t_old.rest_channel and all(np.array_equal(t_new.Q[n], np.eye(3)) and np.array_equal(t_old.Q[n], np.eye(3)) for n in range(1, J) if not rot[n])
      and np.allclose(t_new.mu[:, :3], fk_rest(par, t_new.offsets, t_new.R_rest, mu[0, :3].astype(np.float64)).astype(np.float32), atol=1e-5) and np.array_equal(t_old.mu, mu)
      and np.array_equal(t_new.mu[:, 3:], mu[:, 3:]) and np.array_equal(t_new.sd, t_old.sd),
      f"C4 rest-only one_of transform (skip_op, rest) vs the legacy joint-mode rest perturbation on the same RNG: same op None / tree / bones and the same Q on the {int(rot.sum())}/{J} rotation rows; both keep Q = I on the {int((~rot[1:]).sum())} rows without rotation cells; the channel (rest_channel) rebuilds the mean's positions as FK of its rest (legacy: the source rig's mean untouched); other means and every std shared")
check(np.array_equal(t_new.parents, par) and np.allclose(t_new.offsets[1:], z["offset_parent_local"][1:], atol=1e-6) and not np.allclose(t_new.R_rest, z["R_rest_global"]) and np.allclose(fk_rest(par, t_new.offsets, t_new.R_rest, z["P_rest_global"][0]), t_new.P_rest)
      and not np.allclose(t_new.P_rest, z["P_rest_global"]) and np.allclose(np.linalg.norm(t_new.P_rest[1:] - t_new.P_rest[par[1:]], axis=1), np.linalg.norm(z["P_rest_global"][1:] - z["P_rest_global"][par[1:]], axis=1)),
      "C5 rest-only: same tree and bones (rows 1:), R_rest rotated, P_rest = FK of the new rest -- a different rest configuration with the same bone lengths")
t_no = make_transform(np.random.default_rng(11), c_r2, **geom, channel_valid=cv, mu=mu, sd=sd, contact_joints=np.zeros(J, bool), rest_norm=True, skip_op=True, rest=False)
check(np.array_equal(t_no.R_rest, z["R_rest_global"]) and np.array_equal(t_no.Q, np.tile(np.eye(3), (J, 1, 1))), "C6 rest=False: Q = I, R_rest untouched")
try: make_transform(np.random.default_rng(1), AugConfig(p=1.0, rest_deg=20.0), **geom, channel_valid=cv, mu=mu, sd=sd, contact_joints=np.zeros(J, bool), skip_op=True); check(False, "C7 skip_op in joint mode refused")
except ValueError as e: check(True, f"C7 skip_op in joint mode refused ({str(e)[:50]}...)")
# dataset level: rest channel alone (p 0, rest_p 1): same tree / bones / ops as the plain items, world positions unchanged, descriptor follows
DSK_IDX = {**DSK, "balance_skeletons": False}          # index-determined targets: the rest draw must not shift which clip is served
ds_plain = InContextPairs(base, names["train"], names["train"], struct_world_rest=True, **DSK_IDX)
ds_rest = InContextPairs(base, names["train"], names["train"], struct_world_rest=True, augment=AugConfig(**dict(ONE, p=0.0, rest_deg=30.0, rest_p=1.0)), **DSK_IDX)
okC = True; n_rot = 0; worst_w = 0.0; n_pairs = 0
for i in range(0, 600, 100):
    a, b = ds_plain[i], ds_rest[i]
    if a.get("motion_id") != b.get("motion_id"):
        okC = False; print("   C8 motion id differs at", i); continue
    n_pairs += 1
    okC &= same(b["parents"], a["parents"]) and np.allclose(b["rest_offsets"].numpy()[1:], a["rest_offsets"].numpy()[1:], atol=1e-6) and int(b["n_joints"]) == int(a["n_joints"])
    # rows 1: of the first 8 columns follow the (float-recomputed) offsets; the ROOT row differs by a PRE-EXISTING rule of the
    # augmentation path (the transform's root offset row is 0 -> direction / length 0, while the cached un-augmented table
    # reads the skeleton file's root row = the rest height -> direction (0,1,0), log-length ~2): every arm trained with the
    # one_of augmentation (control / S1 / H1) serves that pair of root rows, the rest-only item takes the augmented path's.
    okC &= np.allclose(b["struct_feats"][1:, :8].numpy(), a["struct_feats"][1:, :8].numpy(), atol=1e-5)
    okC &= bool(np.all(b["struct_feats"][0, :6].numpy() == 0) and b["struct_feats"][0, 7].item() == 0 and b["struct_feats"][0, 6].item() == a["struct_feats"][0, 6].item())
    Ra, Rb = a["R_rest_global"].numpy(), b["R_rest_global"].numpy(); n_rot += int(not np.allclose(Ra, Rb))
    par = b["parents"].numpy(); off = b["rest_offsets"].numpy().astype(np.float64)
    # the transform's rest is FK-consistent: R_rest[p] @ offset is the served rest bone, so u must be its direction
    u = b["struct_feats"][:, 8:11].numpy().astype(np.float64); v = np.einsum("jab,jb->ja", Rb[par[1:]].astype(np.float64), off[1:]); vn = np.linalg.norm(v, axis=1); nz = vn > 1e-8
    okC &= np.abs(u[1:][nz] - v[nz] / vn[nz, None]).max() < 1e-4 and not np.allclose(b["struct_feats"][:, 8:], a["struct_feats"][:, 8:])
    # world positions of the target frames: de-normalise each item with ITS OWN statistics (the rest-normalised mean moved)
    tf = a["is_target"].numpy().astype(bool) if "is_target" in a else None
    xa = a["x"].numpy()[..., :3] * (a["anytop_std"].numpy()[None, :, :3] + _STD_FLOOR) + a["anytop_mean"].numpy()[None, :, :3]
    xb = b["x"].numpy()[..., :3] * (b["anytop_std"].numpy()[None, :, :3] + _STD_FLOOR) + b["anytop_mean"].numpy()[None, :, :3]
    sel = tf if tf is not None else np.ones(xa.shape[0], bool)
    worst_w = max(worst_w, float(np.abs(xa[sel] - xb[sel]).max()))
check(okC and n_pairs == 6 and n_rot == 6 and worst_w < 1e-3, f"C8 dataset rest channel (p 0, rest_p 1): {n_pairs}/6 same-clip pairs, {n_rot} rotated, tree / bones / first 8 columns (rows 1:; root row = the augmentation path's pre-existing zero row) unchanged, descriptor follows the new rest, world target positions unchanged (max {worst_w:.1e})")
# C10 (reviewer 2026-09-25 P1): the rest-channel sample is what a rig AUTHORED in that convention serves -- its mean is the
# transformed rest pose (FK of the served R_rest from the mean's own rest root), its 1-frame rest demo is that rig's own rest
# frame (zero on every cell, plane 17 kept), the other means and every std are the rig's; no model-input cell carries Q
# except the target's deltas
okD = True; moved = 0.0; n_ok = 0
for i in range(0, 600, 100):
    a, b = ds_plain[i], ds_rest[i]
    if a.get("motion_id") != b.get("motion_id"):
        okD = False; continue
    n_ok += 1
    par = b["parents"].numpy(); Rb = b["R_rest_global"].numpy().astype(np.float64); off = b["rest_offsets"].numpy().astype(np.float64)
    mb, ma = b["anytop_mean"].numpy(), a["anytop_mean"].numpy()
    Pb = fk_rest(par, off, Rb, mb[0, :3].astype(np.float64))
    okD &= np.abs(mb[:, :3] - Pb).max() < 1e-4                                         # mean positions = the transformed rest pose
    moved = max(moved, float(np.abs(mb[:, :3] - ma[:, :3]).max()))
    okD &= np.array_equal(mb[:, 3:17], ma[:, 3:17]) and np.array_equal(b["anytop_std"].numpy()[:, :17], a["anytop_std"].numpy()[:, :17])
    dem, dem0 = b["x"].numpy()[0], a["x"].numpy()[0]                                     # demo_frames=1: row 0 is the rest demo
    okD &= bool((dem[:, :17] == 0).all() and (dem0[:, :17] == 0).all() and np.array_equal(dem[:, 17], dem0[:, 17]))
check(okD and n_ok == 6 and moved > 1e-3, f"C10 rest-channel sample = a rig authored in that convention: mean positions = FK of the served rest (moved up to {moved:.3f} vs the plain mean), other means and every std unchanged, rest demo all zero with plane 17 kept (plain demo all zero too), {n_ok}/6 pairs")
ds_r2 = InContextPairs(base, names["train"], names["train"], struct_world_rest=True, augment=c_r2, **DSK)
its = [ds_r2[i] for i in range(0, 400, 10)]
check(all(tuple(it["struct_feats"].shape) == (int(it["n_joints"]), 14) for it in its), f"C9 R2 loader (p 0.8, rest_p 0.5): {len(its)} items served with 14 columns")

# ================= C11. FK consistency of the served channel sample on the TRAINING corpus =================
# (reviewer 2026-09-26 P0-1 / P0-2): the FK loss's own comparison -- FK(served deltas with the invalid cells held at the mean,
# served R_rest_global, served offsets, root = direct + track) vs the served position channels -- for every one_of op x rest on
# three rigs with INTERIOR rows without full rotation cells (their delta is a held constant: any Q on them moves the subtree)
# and one rig whose rows all carry rotation cells (the add op's synthetic row must rotate too). Controls: the same ops without
# the rest draw. C8 / C10 could not see either defect (position channels are unchanged by construction; pzh312 rigs had no such rows).
from src.data.ktjd17_augment import _fk
from src.data.ktjd17.codec import decode_column_cont6d
_h1 = subprocess.run(["bash", "-c", "set -a; . configs/pilot36m_uniml3dv2common_bothrope_aug_d512_lr15_hks_2node_env.sh >/dev/null 2>&1; set +a; env"],
                     capture_output=True, text=True, env=dict(os.environ)).stdout
_e = dict(l.split("=", 1) for l in _h1.splitlines() if "=" in l and not l.startswith("BASH_FUNC"))
base2 = Ktjd17Base(_e["KTJD_ROOT"], caption_emb_cache=_e["CAPTION_CACHE"], joint_semantics=_e["JOINT_SEM"], texts_json=_e["TEXTS_JSON"],
                   percell_stats=_e["PERCELL"], exclude_clips=_e["CUT"], normalization="rest")
rig_first = {}
for _i, _r in enumerate(base2._rows):
    rig_first.setdefault(str(_r["rig_id"]), _i)
interior, fullrot = [], []
for rig in sorted(rig_first):
    if len(interior) >= 3 and len(fullrot) >= 1:
        break
    sk2 = base2.skeleton(rig); par2 = np.asarray(sk2["parents"], dtype=np.int64); J2 = len(par2)
    cv2 = np.asarray(base2.static_masks(rig)["channel_valid"], dtype=bool)[:J2]
    has_child = np.zeros(J2, bool); has_child[par2[1:]] = True
    nonrot = [n for n in range(1, J2) if has_child[n] and not cv2[n, 3:9].all()]
    if nonrot and len(interior) < 3:
        interior.append((rig, nonrot))
    elif not nonrot and cv2[:, 3:9].all() and len(fullrot) < 1:
        fullrot.append((rig, []))
def fk_gap(x18, tr):
    xa = apply_motion(x18, tr)
    raw = xa[..., :17].astype(np.float64) * (tr.sd[None].astype(np.float64) + _STD_FLOOR) + tr.mu[None].astype(np.float64)
    inv = ~tr.channel_valid
    raw[:, inv] = np.broadcast_to(tr.mu.astype(np.float64)[None], raw.shape)[:, inv]        # the trainer's projection: invalid cells at the mean
    G = np.matmul(decode_column_cont6d(raw[..., 3:9], strict=False), tr.R_rest[None])
    track = raw[:, 0, 13:15]; sh = np.stack([track[:, 0], np.zeros(len(track)), track[:, 1]], axis=1)
    pos = _fk(tr.parents, raw[:, 0, 0:3] + sh, G, tr.offsets)
    bl = float(np.linalg.norm(tr.offsets[1:], axis=-1).mean())
    return float(np.abs(pos - (raw[..., 0:3] + sh[:, None])).max() / bl)
okF = True; worst = {}; n_cases = 0
c_fk = AugConfig(**dict(ONE, p=1.0, rest_deg=30.0, rest_p=1.0))
for rig, nonrot in interior + fullrot:
    it2 = base2[rig_first[rig]]; J2 = int(it2["num_joints"]); T2 = min(int(it2["num_frames"]), 60)
    x18 = np.asarray(it2["anytop_x"])[:J2, :, :T2].transpose(2, 0, 1).astype(np.float32)
    sk2 = base2.skeleton(rig); cv2 = np.asarray(base2.static_masks(rig)["channel_valid"], dtype=bool)[:J2]
    mu2 = np.asarray(it2["anytop_mean"], dtype=np.float32)[:J2, :17]; sd2 = np.asarray(it2["anytop_std"], dtype=np.float32)[:J2, :17]
    con2 = ((x18[..., 12] * (sd2[None, :, 12] + _STD_FLOOR) + mu2[None, :, 12]) > 0.5).any(0)
    geom2 = dict(parents=np.asarray(sk2["parents"])[:J2], P_rest_global=np.asarray(sk2["P_rest_global"])[:J2], R_rest_global=np.asarray(sk2["R_rest_global"])[:J2],
                 offset_parent_local=np.asarray(sk2["offset_parent_local"])[:J2])
    seen = {}
    for seed in range(60):
        for rest_flag in (True, False):
            tr2 = make_transform(np.random.default_rng(seed), c_fk, **geom2, channel_valid=cv2, mu=mu2, sd=sd2, contact_joints=con2, fps=30.0, rest_norm=True,
                                 skip_op=False, rest=rest_flag)
            key = (tr2.op, rest_flag)
            if key in seen:
                continue
            seen[key] = fk_gap(x18, tr2); n_cases += 1
        if len(seen) == 8:
            break
    worst[rig[:16]] = {f"{op}{'+rest' if r else ''}": round(v, 7) for (op, r), v in sorted(seen.items(), key=lambda kv: (kv[0][0], not kv[0][1]))}
    okF &= len(seen) == 8 and max(seen.values()) < 1e-4
    print(f"   C11 {rig} J{J2} interior non-rot rows {nonrot[:6]}{'...' if len(nonrot) > 6 else ''}: " + " ".join(f"{k}={v:.1e}" for k, v in worst[rig[:16]].items()))
check(okF and len(interior) == 3 and len(fullrot) == 1 and n_cases == 32,
      f"C11 training-corpus FK consistency: {n_cases} cases (4 ops x rest on/off x 3 rigs with interior non-rotation rows + 1 full-rotation rig) all < 1e-4 bl (worst {max(max(w.values()) for w in worst.values()) if worst else float('nan'):.1e})")

# ================= D. the model =================
torch.manual_seed(0); m_on = InContextMotionDiT(**KWM, struct_world_rest=True)
check(m_on.struct_in == 14 and tuple(m_on.struct_mlp[0].weight.shape) == (KWM["dim"], 8) and tuple(m_on.struct_rest_in.weight.shape) == (KWM["dim"], 6)
      and m_on.struct_rest_in.bias is None and float(m_on.struct_mlp[-1].weight.abs().max()) == 0.0 and float(m_on.struct_mlp[-1].bias.abs().max()) == 0.0
      and m_off.struct_rest_in is None, "D1 struct_mlp keeps Linear(8, dim); struct_rest_in = Linear(6, dim, bias=False); last layer zero-initialised; off model has no struct_rest_in")
sd_on, sd_off = m_on.state_dict(), m_off.state_dict()
diff = [k for k in sd_off if k not in sd_on or not (sd_off[k].shape == sd_on[k].shape and torch.equal(sd_off[k], sd_on[k]))]
extra = sorted(set(sd_on) - set(sd_off))
check(diff == [] and extra == ["struct_rest_in.weight"], f"D2 under seed 0 every off-model tensor is bitwise identical in the on model (differs: {diff}); the on model adds exactly struct_rest_in.weight (extra: {extra})")
try: InContextMotionDiT(**{**KWM, "use_struct_feats": False}, struct_world_rest=True); check(False, "D3 struct_world_rest without use_struct_feats refused")
except ValueError as e: check(True, f"D3 struct_world_rest without use_struct_feats refused ({str(e)[:50]}...)")
for name, a_, b_, words in [("14-column weights into an 8-column model", m_off, m_on, ["Unexpected key", "struct_rest_in.weight"]), ("8-column weights into a 14-column model", m_on, m_off, ["Missing key", "struct_rest_in.weight"])]:
    try: a_.load_state_dict(b_.state_dict()); check(False, f"D4 strict load: {name} refused")
    except RuntimeError as e: check(all(w in str(e) for w in words), f"D4 strict load: {name} refused, names {words}")
b3 = collate(items_on[:3]); b3o = collate(items_off[:3])
def run(m, b):
    kw = {k: b[k] for k in ("joint_sem", "text", "joint_bias", "frame_valid", "joint_valid", "struct_feats", "updown", "spectral_feats") if k in b}
    with torch.no_grad(): return m(b["x"][..., :17], torch.full((b["x"].shape[0],), 0.5), is_target=b["is_target"], **kw)
m_on.eval(); m_off.eval()
y_on, y_off = run(m_on, b3), run(m_off, b3o)
check(torch.isfinite(y_on).all() and y_on.shape == y_off.shape and float((y_on - y_off).abs().max()) < 1e-5, f"D5 at init the 14-column model equals the 8-column model (zero last layer; max |diff| {float((y_on - y_off).abs().max()):.1e})")
try: run(m_on, b3o); check(False, "D6 8-column batch into a 14-column model refused")
except ValueError as e: check("struct_feats has 8 columns" in str(e), f"D6 8-column batch into a 14-column model refused ({str(e)[:60]}...)")
m_on.train(); m_on.zero_grad(); run.__wrapped__ = None
with torch.enable_grad():
    kw = {k: b3[k] for k in ("joint_sem", "text", "joint_bias", "frame_valid", "joint_valid", "struct_feats", "updown", "spectral_feats") if k in b3}
    m_on(b3["x"][..., :17], torch.full((3,), 0.5), is_target=b3["is_target"], **kw).square().mean().backward()
check(m_on.struct_mlp[0].weight.grad is not None and float(m_on.struct_mlp[0].weight.grad.abs().sum()) == 0.0 and m_on.struct_rest_in.weight.grad is not None
      and float(m_on.struct_rest_in.weight.grad.abs().sum()) == 0.0 and float(m_on.struct_mlp[-1].weight.grad.abs().sum()) > 0,
      "D7 gradient reaches the zero-initialised last layer (the pathway ramps in); first layer and struct_rest_in get none until then -- as for the 8-column arm")

# ================= E. wiring =================
h = subprocess.run([PY, "scripts/train_v2_incontext.py", "--help"], capture_output=True, text=True).stdout
check("--struct_world_rest" in h and "--aug_rest_p" in h, "E1 trainer: --struct_world_rest and --aug_rest_p in --help")
r = subprocess.run([PY, "scripts/train_v2_incontext.py", "--struct_world_rest"], capture_output=True, text=True, env=dict(os.environ, CUDA_VISIBLE_DEVICES=""))
check(r.returncode != 0 and "--struct_world_rest extends the struct_feats table" in (r.stdout + r.stderr), "E1 trainer refuses --struct_world_rest without --struct_feats")
src = open("scripts/train_v2_incontext.py").read()
check('"aug_rest_p", "struct_world_rest"' in src and src.count("struct_world_rest=a.struct_world_rest") == 4 and "rest_p=a.aug_rest_p" in src,
      "E1 trainer: both flags in the resume-critical list, passed to both datasets and both models, rest_p into AugConfig")
artH = json.load(open("configs/pilot_uniml3dv2common_bothrope_aug_d512_hks_gamma_calibration_b16_v1.json"))
def W(**upd):
    w = dict(artH["protocol"]["verify"]["arm_model"]); w.update(upd); return w
def A(**upd):
    a_ = json.loads(json.dumps(artH)); a_["protocol"]["verify"]["arm_model"].update(upd); return a_
drift = trm.calib_arm_model_drift
check(drift(artH, W(struct_world_rest=False), False) is None and drift(artH, W(struct_world_rest=False), True) is None, "E2 guard: the H1 artifact (no struct_world_rest field) accepted by a run without the rest input")
check(drift(artH, W(struct_world_rest=True), False) is not None and drift(artH, W(struct_world_rest=True), True) is not None, "E2 guard: ... and refused by a rest-input run")
artR = A(struct_world_rest=True)
check(drift(artR, W(struct_world_rest=True), False) is None and drift(artR, W(struct_world_rest=False), False) is not None, "E2 guard: a rest-input artifact accepted by the rest-input run, refused by the H1 run")
# artifact check with the arm's own environment (as the runner builds it from the config chain)
def cfg_env(cfg):
    out = subprocess.run(["bash", "-c", f"set -a; . {cfg}; set +a; env"], capture_output=True, text=True, env=dict(os.environ)).stdout
    e = dict(l.split("=", 1) for l in out.splitlines() if "=" in l and not l.startswith("BASH_FUNC"))
    m = {"DIM": e["DIM"], "DEPTH": e["DEPTH"], "HEADS": e["HEADS"], "QK_NORM": e["QK_NORM"], "STRUCT_FEATS": e["STRUCT_FEATS"], "DIR_BIAS": e["DIR_BIAS"],
         "ARM_SPEC_ROPE": e["SPEC_ROPE"], "ARM_SPEC_ROPE_K": e["SPEC_ROPE_K"], "ARM_SPEC_ROPE_HKS": e["SPEC_ROPE_HKS"], "ARM_TEMPORAL_ROPE": e["TEMPORAL_ROPE"],
         "ARM_TROPE_BASE": e["TROPE_BASE"], "ARM_STRUCT_WORLD_REST": e.get("STRUCT_WORLD_REST", "0"), "ARM_GEO_BIAS": "1", "ARM_FREEZE_ZERO_JOINT_SEM": "0",
         "VERIFY_STEPS": "30", "GAMMA_SOLVE": "kimodo", "EXPECT_CODE_SCRIPT": "scripts/_measure_ktjd17_gamma_calibration_view_v3.py"}
    m.update({k: e[k] for k in e if k.startswith("AUG_")})
    return m
def check_rc(art, name, cfg):
    pth = TMP / f"_tmp_swr_art_{name}.json"; json.dump(art, open(pth, "w"))
    r = subprocess.run([PY, "scripts/_calib_artifact_check.py", str(pth)], env=dict(os.environ, **cfg_env(cfg)), capture_output=True, text=True)
    return r.returncode, (r.stdout + r.stderr).strip()
H1CFG = "configs/pilot36m_uniml3dv2common_bothrope_aug_d512_lr15_hks_2node_env.sh"
R2CFG = "configs/pilot36m_uniml3dv2common_bothrope_aug_d512_lr15_hks_r2_2node_env.sh"
R1CFG = "configs/pilot36m_uniml3dv2common_bothrope_aug_d512_lr15_hks_r1_2node_env.sh"
rc, msg = check_rc(artH, "h1_for_h1", H1CFG); check(rc == 0, f"E3 artifact check: the H1 artifact still passes under the H1 config ({msg[-80:]})")
rc, msg = check_rc(artH, "h1_for_r1", R1CFG); check(rc == 1 and "struct_world_rest" in msg, f"E3 artifact check: the H1 artifact refused under the R1 config (rest input) ({msg[-100:]})")
artR2 = A(struct_world_rest=True); artR2["protocol"]["augmentation"] = AugConfig(**dict(ONE, p=0.8, rest_deg=30.0, rest_p=0.5)).protocol()
rc, msg = check_rc(artR2, "r2_for_r2", R2CFG); check(rc == 0, f"E3 artifact check: an R2-shaped artifact (rest input + rest channel) passes under the R2 config ({msg[-80:]})")
rc, msg = check_rc(artR, "r1_for_r2", R2CFG); check(rc == 1 and "augmentation" in msg, f"E3 artifact check: an R1-shaped artifact (no rest channel) refused under the R2 config ({msg[-100:]})")
rc, msg = check_rc(artR, "r1_for_r1", R1CFG); check(rc == 0, f"E3 artifact check: the R1-shaped artifact passes under the R1 config ({msg[-80:]})")
ms = open("scripts/_measure_ktjd17_gamma_calibration_view_v3.py").read()
check('ARM_STRUCT_WORLD_REST' in ms and 'AUG_REST_P' in ms and 'struct_world_rest=ARM["struct_world_rest"]' in ms and ms.count('struct_world_rest=ARM["struct_world_rest"]') == 2,
      "E4 measurer v3: ARM_STRUCT_WORLD_REST -> dataset + model, AUG_REST_P -> AugConfig, refused without ARM_STRUCT_FEATS")
ls = open("scripts/_launch_v2_ddp_2node_h200.sh").read()
check(subprocess.run(["bash", "-n", "scripts/_launch_v2_ddp_2node_h200.sh"]).returncode == 0 and "STRUCT_WORLD_REST=${STRUCT_WORLD_REST:-0}" in ls and "--struct_world_rest" in ls and "world_rest$STRUCT_WORLD_REST" in ls,
      "E5 launcher: STRUCT_WORLD_REST knob (guarded, in the command line and the banner), parses")
for f, n in (("scripts/v2_render_incontext.py", 2), ("scripts/_eval_v2_gen_in_evalspace.py", 3)):
    c = open(f).read().count('struct_world_rest=bool(ca.get("struct_world_rest", False))')
    check(c == n, f"E6 {f}: model + dataset rebuilt from the checkpoint args ({c}/{n} sites)")
for cfg, want in ((R2CFG, {"STRUCT_WORLD_REST": "1", "AUG_REST_DEG": "30", "AUG_REST_P": "0.5"}), (R1CFG, {"STRUCT_WORLD_REST": "1", "AUG_REST_DEG": "0"})):
    e = cfg_env(cfg); ex = subprocess.run(["bash", "-c", f"set -a; . {cfg}; set +a; echo \"$EXTRA\""], capture_output=True, text=True).stdout
    okE = all(e.get("ARM_STRUCT_WORLD_REST" if k == "STRUCT_WORLD_REST" else k) == v for k, v in want.items())
    okE &= ex.count("--aug_rest_deg") == 1 and f"--aug_rest_deg {want['AUG_REST_DEG']} " in ex + " " and ("--aug_rest_p 0.5" in ex) == (cfg == R2CFG) and "--balance_alpha 0.5" in ex
    check(okE, f"E7 config {Path(cfg).name}: variables and EXTRA agree ({want}; one --aug_rest_deg, balance_alpha kept)")
import scripts._eval_v2_gen_in_evalspace as ev
fp = ev.source_fingerprint("none", False, True, True)
check(isinstance(fp, str) and len(fp) == 64, "E8 generation fingerprint still computes for the spectral + temporal arm")

print(f"\n{'ALL PASS' if not fails else f'{len(fails)} FAIL'} ({time.time() - T0:.0f} s)")
for f in fails: print(" -", f)
sys.exit(1 if fails else 0)
