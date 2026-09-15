"""Spectral joint RoPE arm (UniMate's SignNet spectral RoPE in place of the joint-slot table j_pos): the Laplacian
coordinates (src/data/skeleton_spectral.py), the SignNet + rotary modules (src/models/v2/spec_rope.py), the DiT wiring
(src/models/v2/dit_motion.py) against its pre-change bytes (src/models/v2/dit_motion.py.bak_20260915 == HEAD), the pair
loader / collate emission, and the calibration guards. Run inside an allocation, on CPU:
`srun ... /usr/bin/env CUDA_VISIBLE_DEVICES= python scripts/_aug_dev/_test_spec_rope.py`."""
import sys, os, json, subprocess, importlib.util, copy, time
import numpy as np, torch
sys.path.insert(0, ".")
torch.set_num_threads(8)
from src.data.skeleton_spectral import laplacian_eigenvectors
from src.models.v2.spec_rope import SignNetSpectralEncoder, apply_rotary_pos_emb
from src.models.v2.dit_motion import InContextMotionDiT
from src.data.ktjd17_incontext import Ktjd17Base, ktjd17_split_names
from src.data.incontext_pairs import InContextPairs, collate
from src.data.ktjd17_augment import AugConfig
import scripts.train_v2_incontext as trm

PREV = "src/models/v2/dit_motion.py.bak_20260915"
from importlib.machinery import SourceFileLoader
_ld = SourceFileLoader("dit_motion_prev", PREV)
prev = importlib.util.module_from_spec(importlib.util.spec_from_loader("dit_motion_prev", _ld))
sys.modules["dit_motion_prev"] = prev
_ld.exec_module(prev)
assert "spec_rope" not in open(PREV).read(), "the backup must be the pre-change module"

fails = []
def check(cond, msg):
    if not cond: fails.append(msg); print("FAIL:", msg)
    else: print("ok:", msg)
T0 = time.time()

# ---------------- 1. Laplacian coordinates: fixtures + properties ----------------
def chain(n): return np.array([-1] + list(range(n - 1)), dtype=np.int64)
def star(n): return np.array([-1] + [0] * (n - 1), dtype=np.int64)
ASYM = np.array([-1, 0, 1, 2, 3, 1, 5, 0, 7, 8, 9, 7, 11], dtype=np.int64)     # 13 joints, no mirror symmetry

def sym_laplacian(parents):
    n = len(parents); A = np.zeros((n, n))
    for j, p in enumerate(parents):
        if p >= 0 and p != j: A[j, p] = A[p, j] = 1.0
    d = A.sum(1); Dm = np.diag(1.0 / np.sqrt(np.where(d > 0, d, 1.0)))
    return np.eye(n) - Dm @ A @ Dm, d

def lap_props(parents, K, tag):
    V, lam = laplacian_eigenvectors(parents, K)
    n = len(parents); k = min(n - 1, K)
    L, d = sym_laplacian(parents)
    check(V.shape == (n, K) and lam.shape == (K,) and V.dtype == np.float32 and lam.dtype == np.float32, f"{tag}: shapes [{n},{K}] / [{K}] float32")
    res = max((np.abs(L @ V[:, c].astype(np.float64) - lam[c] * V[:, c]).max() for c in range(k)), default=0.0)
    check(res < 2e-5, f"{tag}: L v = lambda v on the independently assembled L_sym (max residual {res:.2e})")
    G = V[:, :k].T.astype(np.float64) @ V[:, :k]
    gdev = float(np.abs(G - np.eye(k)).max()) if k else 0.0
    check(gdev < 2e-5, f"{tag}: kept columns orthonormal (max |V^T V - I| {gdev:.2e})")
    triv = np.abs(V[:, :k].T @ np.sqrt(d)).max() if k else 0.0
    check(k == 0 or (lam[0] > 1e-8 and triv < 2e-5), f"{tag}: trivial eigenvector excluded (lambda_1 {lam[0] if k else 0:.4f}, |V^T sqrt(d)| {triv:.1e})")
    check(np.all(lam[:k] >= 0) and np.all(np.diff(lam[:k]) >= -1e-7), f"{tag}: eigenvalues ascending and clamped >= 0")
    check(np.all(V[:, k:] == 0) and np.all(lam[k:] == 0), f"{tag}: zero padding beyond k={k}")
    return V, lam

for tag, par in (("chain5", chain(5)), ("chain2", chain(2)), ("star7", star(7)), ("asym13", ASYM), ("single", np.array([-1]))):
    lap_props(par, 8, tag)
V4, _ = laplacian_eigenvectors(chain(4), 8); check(np.count_nonzero(np.abs(V4).sum(0)) == 3, "chain4 with K=8 keeps 3 real columns and pads 5")
# permutation equivariance (rows follow the joints, columns equal up to sign) on a tree with distinct eigenvalues
Vo, lo = laplacian_eigenvectors(ASYM, 8)
gap = np.diff(np.linalg.eigvalsh(sym_laplacian(ASYM)[0])[:10]).min()
check(gap > 1e-6, f"asym13: first eigenvalues distinct (min gap {gap:.3e}) so the eigenvectors are unique up to sign")
rng = np.random.default_rng(0)
for trial in range(5):
    order = rng.permutation(len(ASYM)); pos = np.argsort(order)
    newp = np.array([-1 if ASYM[o] < 0 else pos[ASYM[o]] for o in order])
    Vn, ln = laplacian_eigenvectors(newp, 8)
    dev = max(min(np.abs(Vn[:, c] - Vo[order, c]).max(), np.abs(Vn[:, c] + Vo[order, c]).max()) for c in range(8))
    check(dev < 2e-5 and np.abs(ln - lo).max() < 1e-6, f"asym13 permutation {trial}: eigenvectors follow the joints up to column sign (dev {dev:.1e})")

# ---------------- 2. SignNet sign invariance (bitwise) + rotary identities ----------------
torch.manual_seed(0)
enc = SignNetSpectralEncoder(8, 64, 32)
v = torch.randn(3, 13, 8)
signs = (torch.randint(0, 2, (3, 1, 8)) * 2 - 1).float()
check(torch.equal(enc(v), enc(v * signs)), "SignNet: flipping eigenvector signs leaves the angles bitwise unchanged")
check(not torch.equal(enc(v), enc(v * 1.5)), "SignNet: (control) a scale change does change the angles")
dh = 64; J = 13
q = torch.randn(J, dh); k = torch.randn(J, dh); ang = torch.randn(J, dh // 2)
cs, sn = torch.cat([ang, ang], -1).cos(), torch.cat([ang, ang], -1).sin()
rq, rk = apply_rotary_pos_emb(q, k, cs, sn)
check(torch.allclose(rq.norm(dim=-1), q.norm(dim=-1), atol=1e-5), "rotary: norms preserved")
zc, zs = torch.ones_like(cs), torch.zeros_like(sn)
check(torch.equal(apply_rotary_pos_emb(q, k, zc, zs)[0], q), "rotary: zero angles = identity (bitwise)")
worst = 0.0
for i, j in ((0, 5), (3, 3), (12, 1), (7, 9)):
    lhs = float(rq[i] @ rk[j])
    d = ang[i] - ang[j]; dc, ds_ = torch.cat([d, d]).cos(), torch.cat([d, d]).sin()
    rhs = float(apply_rotary_pos_emb(q[i:i + 1], q[i:i + 1], dc, ds_)[0][0] @ k[j])
    worst = max(worst, abs(lhs - rhs))
check(worst < 1e-4, f"rotary: <R(a_i) q_i, R(a_j) k_j> == <R(a_i - a_j) q_i, k_j> (worst {worst:.1e}) -- relative angles only")

# ---------------- 3. model: off-path parity with the pre-change bytes, on-path structure ----------------
KWM = dict(in_ch=17, dim=384, depth=8, n_heads=6, d_text=4096, d_joint_sem=4096,
           use_struct_feats=True, use_dir_bias=True, qk_norm=True, use_geo_bias=True)

def hops(parents):
    n = len(parents); D = np.full((n, n), 99); np.fill_diagonal(D, 0)
    for j, p in enumerate(parents):
        if p >= 0: D[j, p] = D[p, j] = 1
    for m in range(n):
        D = np.minimum(D, D[:, m:m + 1] + D[m:m + 1, :])
    return D

def make_inputs(seed, parents, B=2, T=6):
    g = torch.Generator().manual_seed(seed); Jn = len(parents)
    geo = torch.from_numpy(hops(parents)).float()
    d = dict(x=torch.randn(B, T, Jn, 17, generator=g), t=torch.rand(B, generator=g),
             is_target=torch.zeros(B, T, dtype=torch.bool), joint_sem=torch.randn(B, Jn, 4096, generator=g),
             text=torch.randn(B, 4096, generator=g), joint_bias=(-geo.clamp(max=8))[None].repeat(B, 1, 1),
             frame_valid=torch.ones(B, T, dtype=torch.bool), joint_valid=torch.ones(B, Jn, dtype=torch.bool),
             struct_feats=torch.randn(B, Jn, 8, generator=g), updown=torch.randint(0, 16, (B, Jn, Jn, 2), generator=g),
             spectral_feats=torch.from_numpy(laplacian_eigenvectors(parents, 8)[0])[None].repeat(B, 1, 1))
    d["is_target"][:, 1:] = True
    return d

def run(model, d, **extra):
    kw = {k: v for k, v in d.items() if k not in ("x", "t")}; kw.update(extra)
    return model(d["x"], d["t"], **kw)

def build(mod, seed, **kw):
    torch.manual_seed(seed); return mod.InContextMotionDiT(**KWM, **kw)

off_prev = build(prev, 0); off = build(sys.modules["src.models.v2.dit_motion"], 0)
sp, sn_ = off_prev.state_dict(), off.state_dict()
check(list(sp.keys()) == list(sn_.keys()) and all(torch.equal(sp[k], sn_[k]) for k in sp), f"off-path: state_dict keys and every tensor bitwise identical to the pre-change module ({len(sp)} tensors)")
D0 = make_inputs(1, ASYM); D0off = {k: v for k, v in D0.items() if k != "spectral_feats"}
off_prev.eval(); off.eval()
with torch.no_grad():
    check(torch.equal(run(off_prev, D0off), run(off, D0off)), "off-path: eval forward bitwise identical to the pre-change module")
for m in (off_prev, off): m.train(); m.grad_ckpt = True
outs, grads = [], []
for m in (off_prev, off):
    m.zero_grad(); o = run(m, D0off); o.square().mean().backward(); outs.append(o.detach()); grads.append({n: p.grad.clone() for n, p in m.named_parameters() if p.grad is not None})
check(torch.equal(outs[0], outs[1]) and grads[0].keys() == grads[1].keys() and all(torch.equal(grads[0][n], grads[1][n]) for n in grads[0]), "off-path: train forward + grads through activation checkpointing bitwise identical to the pre-change module")
for m in (off_prev, off): m.grad_ckpt = False; m.eval()

on = build(sys.modules["src.models.v2.dit_motion"], 0, use_spec_rope=True, spec_rope_k=8)
so = on.state_dict()
extra = [k for k in so if k not in sn_]; missing = [k for k in sn_ if k not in so]
check(missing == ["j_pos"], f"on-path: the only key gone is j_pos (missing {missing})")
check(all(k.startswith("spec_rope.spectral_encoder.") for k in extra) and len(extra) == 8, f"on-path: new keys are the SignNet's ({extra})")
shared = [k for k in sn_ if k in so]
check(all(torch.equal(sn_[k], so[k]) for k in shared), f"on-path: every shared tensor bitwise identical to the off-path model under the same seed ({len(shared)} tensors; the dropped j_pos draw still consumed)")
shp = {k: tuple(so[k].shape) for k in extra}
want_shp = {"spec_rope.spectral_encoder.phi.0.weight": (64, 1), "spec_rope.spectral_encoder.phi.0.bias": (64,),
            "spec_rope.spectral_encoder.phi.2.weight": (64, 64), "spec_rope.spectral_encoder.phi.2.bias": (64,),
            "spec_rope.spectral_encoder.rho.0.weight": (64, 512), "spec_rope.spectral_encoder.rho.0.bias": (64,),
            "spec_rope.spectral_encoder.rho.2.weight": (32, 64), "spec_rope.spectral_encoder.rho.2.bias": (32,)}
check(shp == want_shp, f"on-path: SignNet shapes (K=8, hidden 64, head_dim 64 -> 32 angles): {shp}")
import math
ok_init, desc = True, []
for nm in ("phi.0", "phi.2", "rho.0", "rho.2"):
    W = so[f"spec_rope.spectral_encoder.{nm}.weight"]; bvec = so[f"spec_rope.spectral_encoder.{nm}.bias"]
    bound = math.sqrt(6.0 / (W.shape[0] + W.shape[1])); mx = float(W.abs().max())
    ok_init &= mx <= bound + 1e-7 and mx > 0.6 * bound and float(bvec.abs().max()) == 0.0 and float(W.std()) > 0.3 * bound
    desc.append(f"{nm} max|W| {mx:.3f} <= {bound:.3f}")
check(ok_init, "on-path: every SignNet Linear carries UniMate's EFFECTIVE init (denoiser-wide xavier-uniform / zero bias, base.py initialize_weights overrides rope.py's normal 0.02): " + "; ".join(desc))
n_off = sum(p.numel() for p in off.parameters()); n_on = sum(p.numel() for p in on.parameters())
print(f"params: off-path {n_off:,} on-path {n_on:,} (j_pos {160*384:,} removed, SignNet {n_on - n_off + 160*384:,} added)")
on.eval()
with torch.no_grad():
    o_on = run(on, D0)
check(tuple(o_on.shape) == tuple(D0["x"].shape) and torch.isfinite(o_on).all(), "on-path: eval forward runs, finite, right shape")
for name, fn in (("no spectral_feats", lambda: run(on, D0off)), ("K mismatch", lambda: run(on, D0, spectral_feats=D0["spectral_feats"][..., :4])),
                 ("off-path given spectral_feats", lambda: run(off, D0))):
    try:
        with torch.no_grad(): fn()
        check(False, f"refusal: {name} raised")
    except ValueError as e:
        check(True, f"refusal: {name} raised ValueError ({str(e)[:60]}...)")
for a, b, name in ((off, on, "on-path weights into an off-path model"), (on, off, "off-path weights into an on-path model")):
    try:
        a.load_state_dict(b.state_dict()); check(False, f"strict load: {name} refused")
    except RuntimeError as e:
        check("j_pos" in str(e) and "spec_rope" in str(e), f"strict load: {name} refused (names both j_pos and spec_rope)")
on.train(); on.grad_ckpt = False
on.zero_grad(); o1 = run(on, D0); o1.square().mean().backward(); g1 = {n: p.grad.clone() for n, p in on.named_parameters() if p.grad is not None}   # bp_mlp: no blueprint in the fixture
on.grad_ckpt = True
on.zero_grad(); o2 = run(on, D0); o2.square().mean().backward(); g2 = {n: p.grad.clone() for n, p in on.named_parameters() if p.grad is not None}
on.grad_ckpt = False
nograd = [n for n, p in on.named_parameters() if p.grad is None]
check(all(n.startswith("bp_mlp.") for n in nograd), f"on-path: every parameter except the unused blueprint MLP receives gradient (none: {nograd})")
eq = g1.keys() == g2.keys() and torch.equal(o1.detach(), o2.detach()) and all(torch.equal(g1[n], g2[n]) for n in g1)
close = torch.allclose(o1.detach(), o2.detach(), atol=1e-6) and all(torch.allclose(g1[n], g2[n], atol=1e-6) for n in g1)
check(close, f"on-path: activation-checkpoint path == plain path, outputs and grads (bitwise {eq})")
gs = {n: float(g1[n].abs().max()) for n in g1 if n.startswith("spec_rope.")}
check(all(v > 0 for v in gs.values()), f"on-path: every SignNet parameter receives gradient (max |g| {min(gs.values()):.2e}..{max(gs.values()):.2e})")
on.eval()
# checkpoint round trip
pth = os.path.join(os.environ.get("SCRATCH_TMP", "runs/_heldout/_aug_dev_logs"), "_tmp_specrope_sd.pt"); torch.save(on.state_dict(), pth)
on2 = build(sys.modules["src.models.v2.dit_motion"], 7, use_spec_rope=True, spec_rope_k=8); on2.load_state_dict(torch.load(pth)); on2.eval(); os.remove(pth)
with torch.no_grad():
    check(torch.equal(run(on2, D0), o_on), "on-path: state_dict round trip reproduces the output bitwise")
# the spectral path is live: another rig's coordinates change the output
with torch.no_grad():
    alt = D0["spectral_feats"][:, torch.randperm(13)]
    dlive = float((run(on, D0, spectral_feats=alt) - o_on).abs().max())
check(dlive > 1e-5, f"on-path: the spectral coordinates reach the output (max change {dlive:.2e} when another rig's are supplied)")

# ---------------- 4. joint-permutation equivariance: on-path exact, off-path broken (the slot table) ----------------
def permute(d, order):
    o = torch.as_tensor(order)
    e = dict(d); e["x"] = d["x"][:, :, o]; e["joint_sem"] = d["joint_sem"][:, o]; e["joint_bias"] = d["joint_bias"][:, o][:, :, o]
    e["struct_feats"] = d["struct_feats"][:, o]; e["updown"] = d["updown"][:, o][:, :, o]; e["spectral_feats"] = d["spectral_feats"][:, o]
    e["joint_valid"] = d["joint_valid"][:, o]
    return e
worst_on, worst_off = 0.0, 0.0
with torch.no_grad():
    base_on = run(on, D0); base_off = run(off, D0off)
    for trial in range(3):
        order = rng.permutation(13)
        Dp = permute(D0, order); Dpoff = {k: v for k, v in Dp.items() if k != "spectral_feats"}
        worst_on = max(worst_on, float((run(on, Dp) - base_on[:, :, order]).abs().max()))
        worst_off = max(worst_off, float((run(off, Dpoff) - base_off[:, :, order]).abs().max()))
check(worst_on < 1e-4, f"on-path: joint-permutation EQUIVARIANT (max |out(perm) - perm(out)| {worst_on:.2e}; scale of out {float(base_on.abs().max()):.2f})")
check(worst_off > 1e-3, f"off-path (slot table): NOT equivariant, as expected (max deviation {worst_off:.2e}) -- the mechanism this arm removes")

# ---------------- 5. the pair loader and collate ----------------
ROOT, CUT = "dataset/ktjd17_pzh312_noik_v2", "configs/heldout20_v1_exclusions.json"
KW = dict(caption_emb_cache="data/noik_caption_llm2vec_v1", joint_semantics="data/joint_semantics_llm2vec_pzh312_v1.npz",
          texts_json="data/noik_pzh312_motion_texts_v1.json", percell_stats="data/noik_norm_stats_v2.npz",
          exclude_clips=CUT, random_caption=False)
base = Ktjd17Base(ROOT, normalization="rest", **KW)
names = ktjd17_split_names(ROOT, exclude=CUT)
DSK = dict(demo_frames=1, target_frames=240, balance_skeletons=True, seed=0, emit_fk_fields=True, emit_graph_v2=True, demo_rest=True)
ds = InContextPairs(base, names["train"], names["train"], emit_spectral=8, **DSK)
ds0 = InContextPairs(base, names["train"], names["train"], emit_spectral=0, **DSK)
it0 = ds0[0]; it = ds[0]
check("spectral_feats" not in it0 and set(it.keys()) == set(it0.keys()) | {"spectral_feats"}, "loader: emit_spectral=0 emits no spectral_feats; =8 adds exactly that key")
seen, roots_ok, items = {}, True, []
step = max(1, len(ds) // 60)
for i in range(0, len(ds), step):
    o = ds[i]; par = o["parents"].numpy(); sf = o["spectral_feats"]
    items.append(o)
    ref = torch.from_numpy(laplacian_eigenvectors(par, 8)[0])
    if not (tuple(sf.shape) == (int(o["n_joints"]), 8) and torch.equal(sf, ref)):
        check(False, f"loader item {i}: spectral_feats == laplacian_eigenvectors(served parents) ({tuple(sf.shape)})"); break
    roots = int(((par < 0) | (par == np.arange(len(par)))).sum()); roots_ok &= roots == 1
    seen.setdefault(tuple(par.tolist()), i)
else:
    check(True, f"loader: {len(items)} items, spectral_feats == laplacian_eigenvectors(served parents) bitwise, [n_joints, 8]")
check(roots_ok, f"loader: every served tree has exactly one root ({len(seen)} distinct trees)")
for n, (par, i) in enumerate(list(seen.items())[:12]):
    lap_props(np.array(par), 8, f"rig#{n}(J={len(par)})")
aug = AugConfig(p=1.0, drop_mode="tips", bone_scale=0.1, mode="one_of")
dsa = InContextPairs(base, names["train"], names["train"], emit_spectral=8, augment=aug, **DSK)
changed, okall = 0, True
for i in range(0, len(dsa), max(1, len(dsa) // 24)):
    o = dsa[i]; par = o["parents"].numpy()
    okall &= torch.equal(o["spectral_feats"], torch.from_numpy(laplacian_eigenvectors(par, 8)[0])) and tuple(o["spectral_feats"].shape) == (int(o["n_joints"]), 8)
    j0 = int(np.asarray(base[ds.targets[i] if hasattr(ds, "targets") else 0]["parent_indices"]).shape[0]) if False else None
    changed += int(len(par) != len(items[0]["parents"]))
check(okall, "loader (augmented, one_of p=1): spectral_feats == laplacian_eigenvectors(the item's OWN served tree) bitwise")
bt = collate(items[:3]); Jm = bt["x"].shape[2]
check(tuple(bt["spectral_feats"].shape) == (3, Jm, 8) and all(torch.equal(bt["spectral_feats"][k, :int(b["n_joints"])], b["spectral_feats"]) and bool((bt["spectral_feats"][k, int(b["n_joints"]):] == 0).all()) for k, b in enumerate(items[:3])), f"collate: spectral_feats padded to [B,{Jm},8], zeros beyond n_joints")
c = trm.cond_of(bt)
check("spectral_feats" in c and c["spectral_feats"] is bt["spectral_feats"], "trainer cond_of forwards spectral_feats")
with torch.no_grad():
    dev_b = {k: (v.to("cpu") if torch.is_tensor(v) else v) for k, v in bt.items()}
    xin = dev_b["x"][..., :17]
    o = on(xin, torch.rand(3), is_target=dev_b["is_target"], **{k: v for k, v in trm.cond_of(dev_b).items()})
check(tuple(o.shape) == tuple(xin.shape) and torch.isfinite(o).all(), "on-path model consumes a real collated batch (padded rigs) through cond_of")

# ---------------- 6. trainer flags, calibration guards, artifact check ----------------
h = subprocess.run([sys.executable, "scripts/train_v2_incontext.py", "--help"], capture_output=True, text=True).stdout
check("--spec_rope " in h or "--spec_rope\n" in h or "--spec_rope]" in h, "trainer: --spec_rope in --help") ; check("--spec_rope_k" in h, "trainer: --spec_rope_k in --help")
src = open("scripts/train_v2_incontext.py").read()
check('"spec_rope", "spec_rope_k")' in src, "trainer: both flags pinned in the resume-critical list")
check("--spec_rope --spec_rope_k" in open("scripts/_launch_v2_ddp_2node_h200.sh").read(), "launcher: emits --spec_rope --spec_rope_k from SPEC_ROPE and forbids them in EXTRA")
artA = json.load(open("configs/pilot_animal_heldout_rest_gamma_calibration_b16_v1.json"))
full4 = {k: artA["protocol"]["verify"]["arm_model"][k] for k in ("struct_feats", "dir_bias", "geo_bias", "freeze_zero_joint_sem")}
check(trm.calib_arm_model_drift(artA, dict(full4, spec_rope=False, spec_rope_k=8), True) is None, "guard: an artifact without the spectral fields == measured without the spectral RoPE (legacy default) for the uniform arm")
check(trm.calib_arm_model_drift(artA, dict(full4, spec_rope=True, spec_rope_k=8), True) is not None, "guard: ... and is refused by a spec-rope run")
artS = json.loads(json.dumps(artA)); artS["protocol"]["verify"]["arm_model"].update(spec_rope=True, spec_rope_k=8)
check(trm.calib_arm_model_drift(artS, dict(full4, spec_rope=True, spec_rope_k=8), True) is None and trm.calib_arm_model_drift(artS, dict(full4, spec_rope=False, spec_rope_k=8), True) is not None and trm.calib_arm_model_drift(artS, dict(full4, spec_rope=True, spec_rope_k=4), True) is not None, "guard: a spec-rope artifact accepted by the matching run, refused by a slot-table run and by another K")
artL = json.loads(json.dumps(artS)); artL["protocol"]["verify"]["arm_model"].pop("dir_bias")
check(trm.calib_arm_model_drift(artL, dict(full4, spec_rope=True, spec_rope_k=8), True) is not None, "guard: the four older fields stay strict (no legacy completion)")
# a CALIBRATED arm (require_uniform False) -- the spectral fields still bind (codex specrope r1 P1), the four conditioning fields do not
check(trm.calib_arm_model_drift(artA, dict(full4, spec_rope=True, spec_rope_k=8), False) is not None, "guard (calibrated arm): a slot-table artifact is refused by a spec-rope run")
check(trm.calib_arm_model_drift(artS, dict(full4, spec_rope=True, spec_rope_k=8), False) is None, "guard (calibrated arm): the spec-rope artifact is accepted by the spec-rope run")
check(trm.calib_arm_model_drift(artS, dict(full4, spec_rope=True, spec_rope_k=4), False) is not None, "guard (calibrated arm): refused for another K")
check(trm.calib_arm_model_drift(artS, dict(full4, spec_rope=False, spec_rope_k=8), False) is not None, "guard (calibrated arm): a spec-rope artifact is refused by a slot-table run")
check(trm.calib_arm_model_drift(artA, dict(full4, spec_rope=False, spec_rope_k=8), False) is None, "guard (calibrated arm): a legacy artifact accepted by a slot-table run")
artW = json.loads(json.dumps(artA)); artW["protocol"]["verify"]["arm_model"]["dir_bias"] = not full4["dir_bias"]
check(trm.calib_arm_model_drift(artW, dict(full4, spec_rope=False, spec_rope_k=8), False) is None and trm.calib_arm_model_drift(artW, dict(full4, spec_rope=False, spec_rope_k=8), True) is not None, "guard: the four conditioning fields bind the uniform arm only (shared artifacts of calibrated arms stay allowed)")
# the calibration code hash: one formula for the trainer and the measurer, spectral files appended for the spectral arm only
import hashlib
from pathlib import Path
REPO = Path(".").resolve(); MS = REPO / "scripts/_measure_ktjd17_gamma_calibration_view_v2.py"
h0, h1 = trm.calib_code_sha256(REPO, MS, False), trm.calib_code_sha256(REPO, MS, True)
ref0 = hashlib.sha256((REPO / "src/models/v2/dit_motion.py").read_bytes() + MS.read_bytes()).hexdigest()
ref1 = hashlib.sha256((REPO / "src/models/v2/dit_motion.py").read_bytes() + MS.read_bytes() + (REPO / "src/models/v2/spec_rope.py").read_bytes()
                      + (REPO / "src/data/skeleton_spectral.py").read_bytes()).hexdigest()
check(h0 == ref0 and h1 == ref1 and h0 != h1, "calibration code hash: slot-table formula unchanged (dit_motion + script); spectral formula appends spec_rope.py + skeleton_spectral.py")
check("calib_code_sha256(Path(\".\").resolve(), Path(__file__), ARM[\"spec_rope\"])" in open(MS).read(), "measurer records the hash through the same function, keyed by its ARM spec_rope")
try:
    import scripts._eval_v2_gen_in_evalspace as ev
    f0, f1 = ev.source_fingerprint("none", False, False), ev.source_fingerprint("none", False, True)
    hh = hashlib.sha256()
    for rel in ["scripts/_eval_v2_gen_in_evalspace.py", "src/models/v2/dit_motion.py", "src/data/incontext_pairs.py", "src/data/ktjd17_incontext.py", "src/data/ktjd17_anytop13.py"]:
        hh.update(rel.encode()); hh.update((REPO / rel).read_bytes())
    check(f0 == hh.hexdigest() and f0 != f1, "generation fingerprint: slot-table checkpoints keep the old file list; a spectral checkpoint's fingerprint adds the spectral files")
    check("spec_rope=bool(ca.get(\"spec_rope\", False))" in open("scripts/_eval_v2_gen_in_evalspace.py").read(), "generation fingerprint keyed by the checkpoint's spec_rope arg")
except Exception as e:
    check(False, f"eval script fingerprint test raised {e!r}")
ENV = dict(os.environ, DIM="384", DEPTH="8", HEADS="6", QK_NORM="1", STRUCT_FEATS="1", DIR_BIAS="1", ARM_GEO_BIAS="1", ARM_FREEZE_ZERO_JOINT_SEM="0",
           GAMMA_SOLVE=str(artA["protocol"]["gamma_solve"]), VERIFY_STEPS=str(artA["protocol"]["verify"]["steps"]), EXPECT_CODE_SCRIPT=artA["hashes"]["code_script"])
for k in list(ENV):
    if k.startswith("AUG_") or k.startswith("ARM_SPEC"): ENV.pop(k)
def check_rc(art, name, **env):
    pth = f"runs/_heldout/_aug_dev_logs/_tmp_specrope_art_{name}.json"; json.dump(art, open(pth, "w"))
    r = subprocess.run([sys.executable, "scripts/_calib_artifact_check.py", pth], env=dict(ENV, **env), capture_output=True, text=True)
    os.remove(pth); return r.returncode, (r.stdout + r.stderr).strip()[-160:]
rc, msg = check_rc(artS, "specrope", ARM_SPEC_ROPE="1", ARM_SPEC_ROPE_K="8"); check(rc == 0, f"artifact check: spec-rope artifact accepted for the spec-rope arm ({msg})")
rc, msg = check_rc(artA, "legacy_for_specrope", ARM_SPEC_ROPE="1", ARM_SPEC_ROPE_K="8"); check(rc == 1, f"artifact check: an artifact whose model has no spectral RoPE is refused for the spec-rope arm ({msg})")
artK = json.loads(json.dumps(artS)); artK["protocol"]["verify"]["arm_model"]["spec_rope_k"] = 4
rc, msg = check_rc(artK, "k4", ARM_SPEC_ROPE="1", ARM_SPEC_ROPE_K="8"); check(rc == 1, f"artifact check: K=4 artifact refused for K=8 ({msg})")
artF = json.loads(json.dumps(artA)); artF["protocol"]["verify"]["arm_model"].update(spec_rope=False, spec_rope_k=8)
rc, msg = check_rc(artF, "v2_slot", ); check(rc == 0, f"artifact check: a v2-measured slot-table artifact accepted when ARM_SPEC_ROPE is unset (defaults 0 / 8) ({msg})")
rc, msg = check_rc(artS, "specrope_for_slot", ); check(rc == 1, f"artifact check: a spec-rope artifact refused for a slot-table arm ({msg})")

print(f"\n{'ALL PASS' if not fails else f'{len(fails)} FAIL'} ({time.time() - T0:.0f} s)")
for f in fails: print(" -", f)
sys.exit(1 if fails else 0)
