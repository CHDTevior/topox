"""H1 -- heat-kernel-signature coordinates for the spectral joint RoPE (in place of K Laplacian eigenvectors + SignNet):
the coordinate (src/data/skeleton_spectral.py heat_kernel_signature), the plain-MLP encoder (src/models/v2/spec_rope.py
HeatKernelSpectralEncoder), the DiT switch (src/models/v2/dit_motion.py spec_rope_hks), the pair loader (spectral_hks),
and the trainer / calibration / launcher / render / eval wiring. The OFF path is checked BYTE-FOR-BYTE against the main
tree (/iridisfs/scratch/ts1v23/workspace/noKslot_clean, read-only: its working files are the pre-change bytes, identical
to this worktree's before the H1 edit) through a `--dump` subprocess run inside that tree. CPU only, inside an
allocation:
  srun --jobid=<alloc> --overlap -N1 -n1 --cpus-per-task=2 /usr/bin/env CUDA_VISIBLE_DEVICES= \\
       /iridisfs/scratch/ts1v23/.conda/bin/python scripts/_aug_dev/_test_spec_rope_hks.py"""
import sys, os, json, subprocess, importlib.util, time, math
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, ".")
torch.set_num_threads(2)

MAIN = os.environ.get("HKS_TEST_MAIN", "/iridisfs/scratch/ts1v23/workspace/noKslot_clean")   # pre-change reference tree, never written (after the merge: a pre-merge snapshot)
ROOT = f"{MAIN}/dataset/ktjd17_pzh312_noik_v2"
CUT = f"{MAIN}/configs/heldout20_v1_exclusions.json"
KW = dict(caption_emb_cache=f"{MAIN}/data/noik_caption_llm2vec_v1",
          joint_semantics=f"{MAIN}/data/joint_semantics_llm2vec_pzh312_v1.npz",
          texts_json=f"{MAIN}/data/noik_pzh312_motion_texts_v1.json",
          percell_stats=f"{MAIN}/data/noik_norm_stats_v2.npz", exclude_clips=CUT, random_caption=False)
DSK = dict(demo_frames=1, target_frames=240, balance_skeletons=True, seed=0, emit_fk_fields=True, emit_graph_v2=True,
           demo_rest=True)
KWM = dict(in_ch=17, dim=192, depth=2, n_heads=6, d_text=4096, d_joint_sem=4096,
           use_struct_feats=True, use_dir_bias=True, qk_norm=True, use_geo_bias=True)
N_ITEMS, N_AUG = 12, 6


def item_indices(n, k):
    return list(range(0, n, max(1, n // k)))[:k]


def dump_reference(out):
    """Run INSIDE the main tree (cwd + sys.path[0] = MAIN): served items, a collated batch, and the model state dicts
    of the pre-change code, with only the APIs both trees share."""
    from src.data.ktjd17_incontext import Ktjd17Base, ktjd17_split_names
    from src.data.incontext_pairs import InContextPairs, collate
    from src.data.ktjd17_augment import AugConfig
    from src.models.v2.dit_motion import InContextMotionDiT
    base = Ktjd17Base(ROOT, normalization="rest", **KW)
    names = ktjd17_split_names(ROOT, exclude=CUT)
    ds8 = InContextPairs(base, names["train"], names["train"], emit_spectral=8, **DSK)
    ds0 = InContextPairs(base, names["train"], names["train"], emit_spectral=0, **DSK)
    aug = AugConfig(p=1.0, drop_mode="tips", bone_scale=0.1, mode="one_of")
    dsa = InContextPairs(base, names["train"], names["train"], emit_spectral=8, augment=aug, **DSK)
    idx = item_indices(len(ds8), N_ITEMS)
    items8 = [ds8[i] for i in idx]; items0 = [ds0[i] for i in idx]
    itemsa = [dsa[i] for i in item_indices(len(dsa), N_AUG)]
    batch = collate(items8[:3])
    torch.manual_seed(0); off = InContextMotionDiT(**KWM)
    torch.manual_seed(0); sig = InContextMotionDiT(**KWM, use_spec_rope=True, spec_rope_k=8)
    torch.save({"idx": idx, "n": len(ds8), "items8": items8, "items0": items0, "itemsa": itemsa, "batch": batch,
                "off_sd": off.state_dict(), "signet_sd": sig.state_dict()}, out)
    print(f"[dump] {len(items8)} items x2, {len(itemsa)} augmented, batch, 2 state dicts -> {out}")


if len(sys.argv) > 2 and sys.argv[1] == "--dump":
    dump_reference(sys.argv[2]); sys.exit(0)

REPO = Path(".").resolve()
assert str(REPO) != MAIN, "run this from the H1 worktree, not from the main tree"
TMP = Path(os.environ.get("SCRATCH_TMP", str(REPO / "_aug_dev_tmp"))); TMP.mkdir(parents=True, exist_ok=True)   # never the main tree
PY = sys.executable
from src.data.skeleton_spectral import laplacian_eigenvectors, heat_kernel_signature, _heat_kernel_signature64, hks_scales
from src.models.v2.spec_rope import SpectralJointRoPE, HeatKernelSpectralEncoder, SignNetSpectralEncoder
from src.models.v2.dit_motion import InContextMotionDiT
from src.data.ktjd17_incontext import Ktjd17Base, ktjd17_split_names
from src.data.incontext_pairs import InContextPairs, collate
from src.data.ktjd17_augment import AugConfig
import scripts.train_v2_incontext as trm

fails = []
def check(cond, msg):
    if not cond: fails.append(msg); print("FAIL:", msg)
    else: print("ok:", msg)
T0 = time.time()
rng = np.random.default_rng(0)
T8 = hks_scales(8)


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


def sym_laplacian(parents):
    n = len(parents); A = np.zeros((n, n))
    for j, p in enumerate(parents):
        if p >= 0 and p != j: A[j, p] = A[p, j] = 1.0
    d = A.sum(1); Dm = np.diag(1.0 / np.sqrt(np.where(d > 0, d, 1.0)))
    return np.eye(n) - Dm @ A @ Dm, d


def permute_tree(parents, order):
    """new tree whose joint i is old joint order[i]."""
    pos = np.argsort(order)
    return np.array([-1 if parents[o] < 0 or parents[o] == o else pos[parents[o]] for o in order], dtype=np.int64)


def chain(n): return np.array([-1] + list(range(n - 1)), dtype=np.int64)
def star(n): return np.array([-1] + [0] * (n - 1), dtype=np.int64)
ASYM = np.array([-1, 0, 1, 2, 3, 1, 5, 0, 7, 8, 9, 7, 11], dtype=np.int64)
QUAD = np.array([-1, 0, 1, 2, 0, 4, 5, 0, 7, 8, 1, 10, 11, 1, 13, 14], dtype=np.int64)   # mirror-symmetric limbs

# ---------------- real skeletons: 50 rigs of the no-IK corpus ----------------
SKEL = sorted(Path(f"{ROOT}/skeletons").glob("*.npz"))
pick = [SKEL[i] for i in np.linspace(0, len(SKEL) - 1, 50).round().astype(int)]
RIGS = {f.stem: np.load(f)["parents"].astype(np.int64) for f in pick}
check(len(RIGS) == 50 and all(int(((p < 0) | (p == np.arange(len(p)))).sum()) == 1 for p in RIGS.values()),
      f"50 real skeletons loaded ({min(len(p) for p in RIGS.values())} <= J <= {max(len(p) for p in RIGS.values())}), one root each")

# ================= A. OFF path: byte-for-byte against the main tree =================
_ld = importlib.machinery.SourceFileLoader("skel_spec_main", f"{MAIN}/src/data/skeleton_spectral.py")
spec_main = importlib.util.module_from_spec(importlib.util.spec_from_loader("skel_spec_main", _ld)); _ld.exec_module(spec_main)
check(not hasattr(spec_main, "heat_kernel_signature"), "main tree's skeleton_spectral.py is the pre-change module (no heat_kernel_signature)")
okA = True
for tag, par in [("chain5", chain(5)), ("chain2", chain(2)), ("star7", star(7)), ("asym13", ASYM), ("single", np.array([-1]))] + list(RIGS.items()):
    for K in (8, 4):
        v, l = laplacian_eigenvectors(par, K); v0, l0 = spec_main.laplacian_eigenvectors(par, K)
        okA &= same(v, v0) and same(l, l0)
check(okA, "A1 laplacian_eigenvectors: bitwise identical to the main tree on 5 fixtures + 50 rigs, K = 8 and 4")

REF = TMP / "_tmp_hks_reference.pt"
if REF.exists(): REF.unlink()
r = subprocess.run([PY, str(REPO / "scripts/_aug_dev/_test_spec_rope_hks.py"), "--dump", str(REF)], cwd=MAIN,
                   env=dict(os.environ, PYTHONPATH=MAIN, CUDA_VISIBLE_DEVICES="", PYTHONDONTWRITEBYTECODE="1"),
                   capture_output=True, text=True)
check(r.returncode == 0 and REF.exists(), f"A2 reference dump ran inside the main tree (rc {r.returncode}): {(r.stdout + r.stderr).strip()[-200:]}")
ref = torch.load(REF, weights_only=False); REF.unlink()

base = Ktjd17Base(ROOT, normalization="rest", **KW)
names = ktjd17_split_names(ROOT, exclude=CUT)
ds8 = InContextPairs(base, names["train"], names["train"], emit_spectral=8, **DSK)
ds0 = InContextPairs(base, names["train"], names["train"], emit_spectral=0, **DSK)
aug = AugConfig(p=1.0, drop_mode="tips", bone_scale=0.1, mode="one_of")
dsa = InContextPairs(base, names["train"], names["train"], emit_spectral=8, augment=aug, **DSK)
check(len(ds8) == ref["n"] and item_indices(len(ds8), N_ITEMS) == ref["idx"], f"A2 same served target index as the main tree ({len(ds8)} targets)")
items8 = [ds8[i] for i in ref["idx"]]; items0 = [ds0[i] for i in ref["idx"]]
itemsa = [dsa[i] for i in item_indices(len(dsa), N_AUG)]
check(all(same(a, b) for a, b in zip(items8, ref["items8"])), f"A2 emit_spectral=8 (flag off) items: every field bitwise identical to the main tree ({len(items8)} items, keys {sorted(items8[0])})")
check(all(same(a, b) for a, b in zip(items0, ref["items0"])), f"A2 emit_spectral=0 items: bitwise identical to the main tree ({len(items0)} items)")
check(all(same(a, b) for a, b in zip(itemsa, ref["itemsa"])), f"A2 augmented (one_of p=1) items with emit_spectral=8: bitwise identical to the main tree ({len(itemsa)} items)")
check(same(collate(items8[:3]), ref["batch"]), "A2 collate(3 items): bitwise identical to the main tree")
torch.manual_seed(0); off = InContextMotionDiT(**KWM)
torch.manual_seed(0); sig = InContextMotionDiT(**KWM, use_spec_rope=True, spec_rope_k=8)
torch.manual_seed(0); hk = InContextMotionDiT(**KWM, use_spec_rope=True, spec_rope_k=8, spec_rope_hks=True)
for m, key, nm in ((off, "off_sd", "slot-table model"), (sig, "signet_sd", "SignNet spectral model")):
    sd, sd0 = m.state_dict(), ref[key]
    check(list(sd) == list(sd0) and all(same(sd[k], sd0[k]) for k in sd), f"A2 {nm}: state_dict keys and every tensor bitwise identical to the main tree under seed 0 ({len(sd)} tensors)")

# ================= B. HKS invariances =================
check(np.array_equal(T8, np.array([0.25, 0.5, 1, 2, 4, 8, 16, 32])), f"B hks_scales(8) = {T8.tolist()}")
worst = 0.0
for tag, par in list(RIGS.items()) + [("asym13", ASYM), ("quad16", QUAD), ("star7", star(7))]:
    h0 = _heat_kernel_signature64(par, T8)
    for _ in range(3):
        order = rng.permutation(len(par))
        hp = _heat_kernel_signature64(permute_tree(par, order), T8)
        worst = max(worst, float(np.abs(hp - h0[order]).max()))
check(worst < 1e-9, f"B1 permutation: HKS(relabelled tree) == HKS(tree)[order], 53 trees x 3 permutations, float64 max dev {worst:.1e} (< 1e-9)")
worst32 = 0.0
for tag, par in RIGS.items():
    order = rng.permutation(len(par))
    worst32 = max(worst32, float(np.abs(heat_kernel_signature(permute_tree(par, order), T8) - heat_kernel_signature(par, T8)[order]).max()))
check(worst32 <= 1.2e-7, f"B1 the served float32 array agrees under relabelling to within one ulp (max dev {worst32:.1e})")

n_deg_any, n_deg_in8, worst_rot, worst_old_ctrl, n_ctrl = 0, 0, 0.0, 0.0, 0
for tag, par in list(RIGS.items()) + [("quad16", QUAD), ("star7", star(7))]:
    L, d = sym_laplacian(par); lam, V = np.linalg.eigh(L)
    groups, i = [], 1
    while i < len(lam):
        j = i
        while j + 1 < len(lam) and abs(lam[j + 1] - lam[i]) < 1e-9: j += 1
        if j > i: groups.append(list(range(i, j + 1)))
        i = j + 1
    Vr = V * (rng.integers(0, 2, size=(1, len(lam))) * 2 - 1)      # random signs on every column
    for g in groups:
        Q, _ = np.linalg.qr(rng.standard_normal((len(g), len(g))))
        Vr[:, g] = Vr[:, g] @ Q                                      # a random orthogonal basis inside the eigenspace
    h_rot = (Vr[:, 1:] ** 2) @ np.exp(-np.outer(np.maximum(lam[1:], 0), T8))
    worst_rot = max(worst_rot, float(np.abs(h_rot - _heat_kernel_signature64(par, T8)).max()))
    if tag in RIGS:
        n_deg_any += bool(groups); n_deg_in8 += any(min(g) <= 8 for g in groups)
    if any(min(g) <= 8 for g in groups):
        # control: the eigenvector coordinate DOES move under the same change of basis (after the best per-column sign)
        old = laplacian_eigenvectors(par, 8)[0].astype(np.float64); k = min(len(par) - 1, 8)
        rot = Vr[:, 1:1 + k]
        dev = max(min(np.abs(rot[:, c] - old[:, c]).max(), np.abs(rot[:, c] + old[:, c]).max()) for c in range(k))
        worst_old_ctrl = max(worst_old_ctrl, dev); n_ctrl += 1
check(worst_rot < 1e-12, f"B2 sign / basis: recomputed from randomly signed columns and a random orthogonal basis inside every repeated eigenvalue, HKS unchanged (max dev {worst_rot:.1e}); {n_deg_any}/50 rigs have a repeated eigenvalue, {n_deg_in8}/50 inside the first 8 non-trivial modes")
check(n_ctrl > 0 and worst_old_ctrl > 0.1, f"B2 control: the eigenvector coordinate moves under that same change of basis (max column dev {worst_old_ctrl:.2f} over {n_ctrl} trees with a repeated eigenvalue inside K=8)")
ok_rng, ok_lim, ok_expm = True, True, True
import scipy.linalg
for tag, par in list(RIGS.items()) + [("chain2", chain(2)), ("asym13", ASYM)]:
    h = _heat_kernel_signature64(par, T8); L, d = sym_laplacian(par); m = len(par) - 1
    ok_rng &= bool((h > 0).all() and (h < 1).all())
    ok_lim &= bool(np.abs(_heat_kernel_signature64(par, [1e-9])[:, 0] - (1 - d / (2 * m))).max() < 1e-6)
    for t in (0.25, 4.0, 32.0):
        ok_expm &= bool(np.abs(np.diag(scipy.linalg.expm(-t * L)) - d / (2 * m) - _heat_kernel_signature64(par, [t])[:, 0]).max() < 1e-10)
check(ok_rng, "B3 raw HKS: every value in (0, 1) on 52 trees")
check(ok_lim, "B3 t -> 0 limit: h_j -> 1 - d_j / (2m) (within 1e-6 at t = 1e-9)")
check(ok_expm, "B3 independent check: h_j(t) == diag(expm(-t L))_j - d_j / (2m) (scipy expm, t = 0.25 / 4 / 32, within 1e-10)")
check(np.array_equal(heat_kernel_signature([-1], T8), np.zeros((1, 8), np.float32)), "B3 single-joint tree: zeros [1, 8]")
ok_norm, ok_ratio = True, True
for tag, par in list(RIGS.items()) + [("chain2", chain(2)), ("asym13", ASYM)]:
    hn = heat_kernel_signature(par, T8).astype(np.float64); hr = _heat_kernel_signature64(par, T8)
    ok_norm &= bool(np.abs(hn.mean(axis=0) - 1.0).max() < 1e-5)
    ok_ratio &= bool(np.abs(hn - hr / hr.mean(axis=0, keepdims=True)).max() < 1e-6)
check(ok_norm, "B3 served HKS is trace-normalised: every column's mean over the rig's joints is 1 (within 1e-5, float32) on 52 trees")
check(ok_ratio, "B3 served HKS == raw HKS / its per-rig column mean (within 1e-6) on 52 trees")

# ================= C. leaf deletion: HKS vs eigenvector coordinate =================
def drop_leaf(par, leaf):
    keep = [j for j in range(len(par)) if j != leaf]; new = {o: i for i, o in enumerate(keep)}
    return np.array([-1 if par[o] < 0 or par[o] == o else new[par[o]] for o in keep], dtype=np.int64), keep
rel_h, rel_v, flips = [], [], 0
for tag, par in RIGS.items():
    children = np.zeros(len(par), int)
    for j, p in enumerate(par):
        if p >= 0 and p != j: children[p] += 1
    leaves = [j for j in range(len(par)) if children[j] == 0 and par[j] >= 0 and par[j] != j]
    leaf = int(rng.choice(leaves)); newp, keep = drop_leaf(par, leaf)
    h0, h1 = heat_kernel_signature(par, T8).astype(np.float64)[keep], heat_kernel_signature(newp, T8).astype(np.float64)
    v0, v1 = laplacian_eigenvectors(par, 8)[0].astype(np.float64)[keep], laplacian_eigenvectors(newp, 8)[0].astype(np.float64)
    rel_h.append(np.linalg.norm(h1 - h0) / np.linalg.norm(h0))
    dv = sum(min(np.linalg.norm(v1[:, c] - v0[:, c]), np.linalg.norm(v1[:, c] + v0[:, c])) ** 2 for c in range(8))
    rel_v.append(math.sqrt(dv) / np.linalg.norm(v0)); flips += rel_v[-1] > 0.5
mh, mv = float(np.median(rel_h)), float(np.median(rel_v))
print(f"C leaf deletion on 50 rigs, relative change of the surviving joints' coordinates: HKS median {mh:.4f} (mean {np.mean(rel_h):.4f}, p90 {np.percentile(rel_h, 90):.4f}, max {max(rel_h):.4f}); eigenvectors (best per-column sign) median {mv:.4f} (mean {np.mean(rel_v):.4f}, p90 {np.percentile(rel_v, 90):.4f}, max {max(rel_v):.4f}); eigenvector change > 0.5 on {flips}/50 rigs")
check(mh < mv, f"C removing one leaf changes the HKS of the surviving joints less than the eigenvector coordinate (median {mh:.4f} vs {mv:.4f}, ratio {mv / mh:.1f}x)")
# what the served (trace-normalised) values look like per scale (no assertion; the column mean is 1 by construction)
S = np.stack([heat_kernel_signature(p, T8).astype(np.float64) for p in RIGS.values()]) if len({len(p) for p in RIGS.values()}) == 1 else None
per_scale = {f"t={t:g}": (float(np.median([h[:, i].std() for h in (heat_kernel_signature(p, T8) for p in RIGS.values())])),
                          float(np.median([h[:, i].mean() for h in (heat_kernel_signature(p, T8) for p in RIGS.values())])))
             for i, t in enumerate(T8)}
print("C served HKS per scale over the 50 rigs -- median over rigs of (across-joint std, across-joint mean): " + ", ".join(f"{k}: ({s:.4f}, {m:.4f})" for k, (s, m) in per_scale.items()))

# ================= D. loader (HKS mode), shapes, collate =================
try:
    InContextPairs(base, names["train"], names["train"], emit_spectral=0, spectral_hks=True, **DSK); check(False, "D spectral_hks without emit_spectral refused")
except ValueError as e:
    check(True, f"D spectral_hks without emit_spectral refused ({str(e)[:50]}...)")
dsh = InContextPairs(base, names["train"], names["train"], emit_spectral=8, spectral_hks=True, **DSK)
itemsh = [dsh[i] for i in ref["idx"]]
check(all(set(a) == set(b) for a, b in zip(itemsh, items8)), "D HKS-mode items carry exactly the keys of the eigenvector-mode items")
check(all(same(a["spectral_feats"], torch.from_numpy(heat_kernel_signature(a["parents"].numpy(), T8))) and tuple(a["spectral_feats"].shape) == (int(a["n_joints"]), 8) for a in itemsh),
      f"D HKS-mode items: spectral_feats == heat_kernel_signature(served parents, 8 scales) bitwise, [n_joints, 8] ({len(itemsh)} items)")
check(all(same({k: v for k, v in a.items() if k != "spectral_feats"}, {k: v for k, v in b.items() if k != "spectral_feats"}) for a, b in zip(itemsh, items8)),
      "D HKS-mode items: every OTHER field bitwise identical to the eigenvector-mode item")
dsha = InContextPairs(base, names["train"], names["train"], emit_spectral=8, spectral_hks=True, augment=aug, **DSK)
itemsha = [dsha[i] for i in item_indices(len(dsha), N_AUG)]
check(all(same(a["spectral_feats"], torch.from_numpy(heat_kernel_signature(a["parents"].numpy(), T8))) for a in itemsha) and any(int(a["n_joints"]) != int(b["n_joints"]) for a, b in zip(itemsha, itemsa[:0] + [ds8[i] for i in item_indices(len(dsa), N_AUG)])),
      f"D augmented (one_of p=1) HKS-mode items: spectral_feats == HKS of the item's OWN served tree bitwise ({len(itemsha)} items; at least one sub-skeleton differs in J from the unaugmented rig)")
for n in range(2, 10):
    h = heat_kernel_signature(chain(n), T8)
    if not (h.shape == (n, 8) and (h > 0).all()):
        check(False, f"D chain J={n}: shape {h.shape}, all positive"); break
else:
    check(True, "D J = 2..9 trees: [J, 8] with every entry > 0 -- no zero padding along the scale axis, unlike the eigenvector coordinate (which pads K > J-1)")
byJ = {}
for i in range(0, len(dsh), max(1, len(dsh) // 40)):
    o = dsh[i]; byJ.setdefault(int(o["n_joints"]), o)
    if len(byJ) == 3: break
three = list(byJ.values()); bt = collate(three); Jm = bt["x"].shape[2]
check(tuple(bt["spectral_feats"].shape) == (3, Jm, 8) and all(same(bt["spectral_feats"][k, :int(b["n_joints"])], b["spectral_feats"]) and bool((bt["spectral_feats"][k, int(b["n_joints"]):] == 0).all()) for k, b in enumerate(three)),
      f"D collate of three HKS-mode rigs (J = {sorted(byJ)}): spectral_feats [3, {Jm}, 8], each rig's rows exact, zeros beyond n_joints")
c = trm.cond_of(bt); check("spectral_feats" in c and c["spectral_feats"] is bt["spectral_feats"], "D trainer cond_of forwards spectral_feats (unchanged)")

# ================= E. model =================
sd_off, sd_sig, sd_hk = off.state_dict(), sig.state_dict(), hk.state_dict()
shared = [k for k in sd_off if k in sd_hk]
check(all(same(sd_off[k], sd_hk[k]) for k in shared) and [k for k in sd_off if k not in sd_hk] == ["j_pos"], f"E1 HKS model: every backbone tensor bitwise identical to the slot-table model under seed 0 ({len(shared)} tensors); only j_pos is gone")
check(all(same(sd_sig[k], sd_hk[k]) for k in sd_sig if not k.startswith("spec_rope.")), "E1 HKS model: every non-spec_rope tensor bitwise identical to the SignNet model under seed 0")
hk_keys = sorted(k for k in sd_hk if k.startswith("spec_rope."))
want = sorted(["spec_rope.spectral_encoder.scales"] + [f"spec_rope.spectral_encoder.net.{i}.{w}" for i in (0, 2, 4) for w in ("weight", "bias")])
check(hk_keys == want, f"E1 HKS model's spec_rope keys are the MLP's + the scales buffer: {hk_keys}")
check(same(sd_hk["spec_rope.spectral_encoder.scales"], torch.tensor(T8, dtype=torch.float32)), "E1 the scales buffer carries hks_scales(8) in float32")
shp = {k: tuple(sd_hk[k].shape) for k in hk_keys if "net" in k}
check(shp == {"spec_rope.spectral_encoder.net.0.weight": (64, 8), "spec_rope.spectral_encoder.net.0.bias": (64,), "spec_rope.spectral_encoder.net.2.weight": (64, 64), "spec_rope.spectral_encoder.net.2.bias": (64,), "spec_rope.spectral_encoder.net.4.weight": (16, 64), "spec_rope.spectral_encoder.net.4.bias": (16,)}, f"E1 MLP shapes 8 -> 64 -> 64 -> head_dim/2 = 16: {shp}")
ok_init = True
for i in (0, 2, 4):
    W, bvec = sd_hk[f"spec_rope.spectral_encoder.net.{i}.weight"], sd_hk[f"spec_rope.spectral_encoder.net.{i}.bias"]
    bound = math.sqrt(6.0 / (W.shape[0] + W.shape[1])); mx = float(W.abs().max())
    ok_init &= mx <= bound + 1e-7 and mx > 0.6 * bound and float(bvec.abs().max()) == 0.0
check(ok_init, "E1 every MLP Linear carries the same xavier-uniform / zero-bias init pass as the SignNet (unimate_basic_init)")
for name, fn in (("spec_rope_hks without use_spec_rope", lambda: InContextMotionDiT(**KWM, spec_rope_hks=True)),):
    try: fn(); check(False, f"E2 {name} refused")
    except ValueError as e: check(True, f"E2 {name} refused ({str(e)[:60]}...)")
for a_, b_, name, words in ((hk, sig, "SignNet weights into an HKS model", ("net", "phi")), (sig, hk, "HKS weights into a SignNet model", ("net", "phi")),
                            (hk, off, "slot-table weights into an HKS model", ("j_pos", "spec_rope")), (off, hk, "HKS weights into a slot-table model", ("j_pos", "spec_rope"))):
    try: a_.load_state_dict(b_.state_dict()); check(False, f"E3 strict load: {name} refused")
    except RuntimeError as e: check(all(w in str(e) for w in words), f"E3 strict load: {name} refused, names {words}")

def hops(parents):
    n = len(parents); D = np.full((n, n), 99); np.fill_diagonal(D, 0)
    for j, p in enumerate(parents):
        if p >= 0: D[j, p] = D[p, j] = 1
    for m in range(n): D = np.minimum(D, D[:, m:m + 1] + D[m:m + 1, :])
    return D
def make_inputs(seed, parents, B=2, T=6):
    g = torch.Generator().manual_seed(seed); Jn = len(parents); geo = torch.from_numpy(hops(parents)).float()
    d = dict(x=torch.randn(B, T, Jn, 17, generator=g), t=torch.rand(B, generator=g), is_target=torch.zeros(B, T, dtype=torch.bool),
             joint_sem=torch.randn(B, Jn, 4096, generator=g), text=torch.randn(B, 4096, generator=g), joint_bias=(-geo.clamp(max=8))[None].repeat(B, 1, 1),
             frame_valid=torch.ones(B, T, dtype=torch.bool), joint_valid=torch.ones(B, Jn, dtype=torch.bool),
             struct_feats=torch.randn(B, Jn, 8, generator=g), updown=torch.randint(0, 16, (B, Jn, Jn, 2), generator=g),
             spectral_feats=torch.from_numpy(heat_kernel_signature(parents, T8))[None].repeat(B, 1, 1))
    d["is_target"][:, 1:] = True; return d
def run(model, d, **extra):
    kw = {k: v for k, v in d.items() if k not in ("x", "t")}; kw.update(extra); return model(d["x"], d["t"], **kw)
def permute(d, order):
    o = torch.as_tensor(order); e = dict(d)
    e["x"] = d["x"][:, :, o]; e["joint_sem"] = d["joint_sem"][:, o]; e["joint_bias"] = d["joint_bias"][:, o][:, :, o]
    e["struct_feats"] = d["struct_feats"][:, o]; e["updown"] = d["updown"][:, o][:, :, o]; e["spectral_feats"] = d["spectral_feats"][:, o]; e["joint_valid"] = d["joint_valid"][:, o]
    return e
D0 = make_inputs(1, ASYM); hk.eval()
with torch.no_grad(): o_hk = run(hk, D0)
check(tuple(o_hk.shape) == tuple(D0["x"].shape) and torch.isfinite(o_hk).all(), "E4 HKS forward (CPU, dim 192 / depth 2): runs, finite, right shape")
for name, fn in (("no spectral_feats", lambda: run(hk, {k: v for k, v in D0.items() if k != "spectral_feats"})), ("K mismatch", lambda: run(hk, D0, spectral_feats=D0["spectral_feats"][..., :4]))):
    try:
        with torch.no_grad(): fn()
        check(False, f"E4 refusal: {name}")
    except ValueError as e: check(True, f"E4 refusal: {name} raised ValueError")
with torch.no_grad():
    dlive = float((run(hk, D0, spectral_feats=D0["spectral_feats"][:, torch.randperm(13)]) - o_hk).abs().max())
check(dlive > 1e-5, f"E4 the HKS coordinates reach the output (max change {dlive:.2e} when shuffled)")
worst_eq = 0.0
with torch.no_grad():
    for _ in range(3):
        order = rng.permutation(13); worst_eq = max(worst_eq, float((run(hk, permute(D0, order)) - o_hk[:, :, order]).abs().max()))
check(worst_eq < 2e-5, f"E4 joint-permutation EQUIVARIANT: max |out(perm) - perm(out)| {worst_eq:.2e} (scale of out {float(o_hk.abs().max()):.2f})")
hk.train(); hk.grad_ckpt = False; hk.zero_grad(); o1 = run(hk, D0); o1.square().mean().backward()
g1 = {n: p.grad.clone() for n, p in hk.named_parameters() if p.grad is not None}
hk.grad_ckpt = True; hk.zero_grad(); o2 = run(hk, D0); o2.square().mean().backward()
g2 = {n: p.grad.clone() for n, p in hk.named_parameters() if p.grad is not None}; hk.grad_ckpt = False
gs = {n: float(g1[n].abs().max()) for n in g1 if n.startswith("spec_rope.")}
check(len(gs) == 6 and all(v > 0 for v in gs.values()), f"E4 every MLP parameter receives gradient (max |g| {min(gs.values()):.2e}..{max(gs.values()):.2e})")
check(g1.keys() == g2.keys() and torch.allclose(o1.detach(), o2.detach(), atol=1e-6) and all(torch.allclose(g1[n], g2[n], atol=1e-6) for n in g1), "E4 activation-checkpoint path == plain path (outputs and grads)")
hk.eval(); pth = TMP / "_tmp_hks_sd.pt"; torch.save(hk.state_dict(), pth)
torch.manual_seed(7); hk2 = InContextMotionDiT(**KWM, use_spec_rope=True, spec_rope_k=8, spec_rope_hks=True); hk2.load_state_dict(torch.load(pth)); hk2.eval(); pth.unlink()
with torch.no_grad(): check(torch.equal(run(hk2, D0), o_hk), "E4 state_dict round trip reproduces the output bitwise")
# E5 the scale-ladder guard: a checkpoint measured under another ladder is REFUSED, not silently corrected
_sd_bad = {k: v.clone() for k, v in hk.state_dict().items()}
_sk = [k for k in _sd_bad if k.endswith("spec_rope.spectral_encoder.scales")][0]
_sd_bad[_sk] = _sd_bad[_sk] * 2.0
try:
    hk2.load_state_dict(_sd_bad); check(False, "E5 tampered scale ladder refused")
except RuntimeError as e:
    check("scale ladder" in str(e), f"E5 tampered scale ladder refused with the ladder named ({str(e)[:60]}...)")
torch.manual_seed(7); hk4 = InContextMotionDiT(**KWM, use_spec_rope=True, spec_rope_k=4, spec_rope_hks=True)
try:
    hk4.load_state_dict(hk.state_dict()); check(False, "E5 K=8 checkpoint refused by a K=4 model")
except RuntimeError as e:
    check("scale ladder" in str(e), f"E5 K=8 checkpoint refused by a K=4 model, ladder named ({str(e)[:50]}...)")
with torch.no_grad():
    xin = bt["x"][..., :17]; o = hk(xin, torch.rand(3), is_target=bt["is_target"], **trm.cond_of(bt))
check(tuple(o.shape) == tuple(xin.shape) and torch.isfinite(o).all(), "E5 HKS model consumes a real collated HKS batch (padded rigs) through cond_of")

# ================= F. trainer / calibration / launcher / render / eval wiring =================
h = subprocess.run([PY, "scripts/train_v2_incontext.py", "--help"], capture_output=True, text=True).stdout
check("--spec_rope_hks" in h, "F1 trainer: --spec_rope_hks in --help")
r = subprocess.run([PY, "scripts/train_v2_incontext.py", "--spec_rope_hks"], capture_output=True, text=True, env=dict(os.environ, CUDA_VISIBLE_DEVICES=""))
check(r.returncode != 0 and "--spec_rope_hks needs --spec_rope" in (r.stdout + r.stderr), "F1 trainer refuses --spec_rope_hks without --spec_rope")
src = open("scripts/train_v2_incontext.py").read()
check('"spec_rope", "spec_rope_k", "spec_rope_hks",' in src, "F1 trainer: spec_rope_hks in the resume-critical list")
check(src.count("spectral_hks=a.spec_rope_hks") == 2 and src.count("spec_rope_hks=a.spec_rope_hks") == 2, "F1 trainer: both datasets and both model constructors receive the flag")
artA = json.load(open("configs/pilot_animal_heldout_rest_gamma_calibration_b16_v1.json"))
armA = artA["protocol"]["verify"]["arm_model"]
check(not any(k in armA for k in ("spec_rope", "spec_rope_k", "spec_rope_hks", "temporal_rope", "trope_base")), "F2 fixture: the rest-arm artifact records no rotary field (a legacy record)")
full4 = {k: armA[k] for k in ("struct_feats", "dir_bias", "geo_bias", "freeze_zero_joint_sem")}
def W(**kw):
    d = dict(full4, spec_rope=False, spec_rope_k=8, spec_rope_hks=False, temporal_rope=False, trope_base=700.0); d.update(kw); return d
def art(**upd):
    a_ = json.loads(json.dumps(artA)); a_["protocol"]["verify"]["arm_model"].update(upd); return a_
artS, artH, artH0 = art(spec_rope=True, spec_rope_k=8), art(spec_rope=True, spec_rope_k=8, spec_rope_hks=True), art(spec_rope=True, spec_rope_k=8, spec_rope_hks=False)
drift = trm.calib_arm_model_drift
check(drift(artA, W(), True) is None and drift(artA, W(), False) is None, "F2 guard: a legacy artifact (no rotary field) is accepted by a slot-table run, uniform AND calibrated (the completed record is what the strict path compares)")
check(drift(artS, W(spec_rope=True), True) is None and drift(artS, W(spec_rope=True), False) is None, "F2 guard: a SignNet artifact without the hks field == measured without HKS: accepted by a SignNet run")
check(drift(artS, W(spec_rope=True, spec_rope_hks=True), True) is not None and drift(artS, W(spec_rope=True, spec_rope_hks=True), False) is not None, "F2 guard: ... and refused by an HKS run (uniform and calibrated)")
check(drift(artH, W(spec_rope=True, spec_rope_hks=True), True) is None and drift(artH, W(spec_rope=True, spec_rope_hks=True), False) is None, "F2 guard: an HKS artifact is accepted by the HKS run")
check(all(drift(artH, w, u) is not None for w in (W(spec_rope=True), W()) for u in (True, False)), "F2 guard: an HKS artifact is refused by a SignNet run and by a slot-table run")
check(drift(artH0, W(spec_rope=True), False) is None and drift(artH0, W(spec_rope=True, spec_rope_hks=True), False) is not None, "F2 guard: an explicit spec_rope_hks=False record behaves as the legacy default")
check(drift(artA, W(spec_rope=True), True) is not None and drift(artS, W(), True) is not None, "F2 guard: the spectral pair still binds (legacy vs spec-rope run, spec-rope artifact vs slot-table run)")
artW = art(); artW["protocol"]["verify"]["arm_model"].pop("dir_bias")
check(drift(artW, W(), True) is not None and drift(artW, W(), False) is None, "F2 guard: the four conditioning fields are never completed -- still strict for the uniform arm, still free for a calibrated arm")
ENV = dict(os.environ, DIM="384", DEPTH="8", HEADS="6", QK_NORM="1", STRUCT_FEATS="1", DIR_BIAS="1", ARM_GEO_BIAS="1", ARM_FREEZE_ZERO_JOINT_SEM="0",
           GAMMA_SOLVE=str(artA["protocol"]["gamma_solve"]), VERIFY_STEPS=str(artA["protocol"]["verify"]["steps"]), EXPECT_CODE_SCRIPT=artA["hashes"]["code_script"])
for k in list(ENV):
    if k.startswith("AUG_") or k.startswith("ARM_SPEC") or k.startswith("ARM_TEMPORAL") or k.startswith("ARM_TROPE"): ENV.pop(k)
def check_rc(a_, name, **env):
    pth = TMP / f"_tmp_hks_art_{name}.json"; json.dump(a_, open(pth, "w"))
    r = subprocess.run([PY, "scripts/_calib_artifact_check.py", str(pth)], env=dict(ENV, **env), capture_output=True, text=True)
    pth.unlink(); return r.returncode, (r.stdout + r.stderr).strip()[-150:]
rc, msg = check_rc(artH, "hks", ARM_SPEC_ROPE="1", ARM_SPEC_ROPE_K="8", ARM_SPEC_ROPE_HKS="1"); check(rc == 0, f"F3 artifact check: HKS artifact accepted for the HKS arm ({msg})")
rc, msg = check_rc(artS, "signet_for_hks", ARM_SPEC_ROPE="1", ARM_SPEC_ROPE_K="8", ARM_SPEC_ROPE_HKS="1"); check(rc == 1, f"F3 artifact check: a SignNet artifact (no hks field) refused for the HKS arm ({msg})")
rc, msg = check_rc(artH, "hks_for_signet", ARM_SPEC_ROPE="1", ARM_SPEC_ROPE_K="8"); check(rc == 1, f"F3 artifact check: the HKS artifact refused for the SignNet arm (ARM_SPEC_ROPE_HKS unset = 0) ({msg})")
rc, msg = check_rc(artS, "signet", ARM_SPEC_ROPE="1", ARM_SPEC_ROPE_K="8"); check(rc == 0, f"F3 artifact check: the SignNet artifact still accepted for the SignNet arm ({msg})")
rc, msg = check_rc(artA, "legacy"); check(rc == 0, f"F3 artifact check: a legacy artifact accepted when no rotary variable is set ({msg})")
check(trm.rotary_code_files(True, False) == ["src/models/v2/spec_rope.py", "src/data/skeleton_spectral.py"], "F4 calibration code hash: the HKS code (spec_rope.py + skeleton_spectral.py) is inside the spectral arm's hashed files; formula unchanged")
ms = open("scripts/_measure_ktjd17_gamma_calibration_view_v3.py").read()
check('spec_rope_hks=os.environ.get("ARM_SPEC_ROPE_HKS", "0") == "1"' in ms and 'spectral_hks=ARM["spec_rope_hks"]' in ms and 'spec_rope_hks=ARM["spec_rope_hks"]' in ms and '"arm_model": ARM' in ms and 'ARM_SPEC_ROPE_HKS=1 needs ARM_SPEC_ROPE=1' in ms,
      "F5 measurer v3: ARM_SPEC_ROPE_HKS -> ARM['spec_rope_hks'] -> dataset spectral_hks + model spec_rope_hks, recorded in arm_model, refused without ARM_SPEC_ROPE")
ms2 = open("scripts/_measure_ktjd17_gamma_calibration_view_v2.py").read()
check("spec_rope_hks" not in ms2, "F5 measurer v2 untouched (frozen for the artifacts bound to it)")
la = open("scripts/_launch_v2_ddp_2node_h200.sh").read()
check("--spec_rope --spec_rope_k --spec_rope_hks" in la and '$([ "$SPEC_ROPE_HKS" = 1 ] && echo --spec_rope_hks)' in la and 'SPEC_ROPE_HKS=1 needs SPEC_ROPE=1' in la and "/hks$SPEC_ROPE_HKS" in la,
      "F6 launcher: SPEC_ROPE_HKS knob -> --spec_rope_hks, forbidden in EXTRA, needs SPEC_ROPE=1, in the banner")
check(subprocess.run(["bash", "-n", "scripts/_launch_v2_ddp_2node_h200.sh"]).returncode == 0, "F6 launcher parses (bash -n)")
for f in ("scripts/v2_render_incontext.py", "scripts/_eval_v2_gen_in_evalspace.py"):
    s_ = open(f).read()
    check('spec_rope_hks=bool(ca.get("spec_rope_hks", False))' in s_ and s_.count('spectral_hks=bool(ca.get("spec_rope_hks", False))') == (1 if "render" in f else 2),
          f"F7 {f}: model rebuilt with spec_rope_hks from ckpt args; every InContextPairs built with spectral_hks from ckpt args")
import hashlib
import scripts._eval_v2_gen_in_evalspace as ev
hh = hashlib.sha256()
for rel in ["scripts/_eval_v2_gen_in_evalspace.py", "src/models/v2/dit_motion.py", "src/data/incontext_pairs.py", "src/data/ktjd17_incontext.py", "src/data/ktjd17_anytop13.py", "src/models/v2/spec_rope.py", "src/data/skeleton_spectral.py"]:
    hh.update(rel.encode()); hh.update((REPO / rel).read_bytes())
check(ev.source_fingerprint("none", False, True, False) == hh.hexdigest(), "F8 generation fingerprint of a spectral checkpoint hashes the HKS code (skeleton_spectral.py + spec_rope.py) -- formula unchanged")
print("F8 live generation fingerprints of THIS worktree (to register at merge time; the main tree's are in the notes): " +
      "; ".join(f"flat={fl} spec={sr} trope={tr}: {ev.source_fingerprint('none', fl, sr, tr)[:16]}..." for fl, sr, tr in ((False, False, False), (False, False, True), (False, True, False), (False, True, True), (True, False, False))))

print(f"\n{'ALL PASS' if not fails else f'{len(fails)} FAIL'} ({time.time() - T0:.0f} s)")
for f in fails: print(" -", f)
sys.exit(1 if fails else 0)
