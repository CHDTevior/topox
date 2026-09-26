"""Arm E: sinusoidal temporal RoPE in place of the learned frame-position table. Checks the module's maths, the DiT
wiring against its pre-change bytes (src/models/v2/dit_motion.py.bak_20260916_trope), the checkpoint contract, and the
trainer / launcher / calibration guards. Run inside an allocation, on CPU:
`srun ... /usr/bin/env CUDA_VISIBLE_DEVICES= python scripts/_aug_dev/_test_temporal_rope.py`."""
import sys, os, json, math, subprocess, importlib.util, time
import numpy as np, torch
sys.path.insert(0, ".")
torch.set_num_threads(8)
from src.models.v2.temporal_rope import sinusoidal_cos_sin, unimate_auto_base
from src.models.v2.spec_rope import apply_rotary_pos_emb
from src.models.v2.dit_motion import InContextMotionDiT
import scripts.train_v2_incontext as trm

PREV = "src/models/v2/dit_motion.py.bak_20260916_trope"
from importlib.machinery import SourceFileLoader
_ld = SourceFileLoader("dit_motion_prev_t", PREV)
prev = importlib.util.module_from_spec(importlib.util.spec_from_loader("dit_motion_prev_t", _ld))
sys.modules["dit_motion_prev_t"] = prev
_ld.exec_module(prev)
assert "temporal_rope" not in open(PREV).read(), "the backup must be the pre-change module"

fails = []
def check(cond, msg):
    if not cond: fails.append(msg); print("FAIL:", msg)
    else: print("ok:", msg)
T0 = time.time()

# ---------------- 1. the module ----------------
check(unimate_auto_base(241) == 700.0 and unimate_auto_base(4096) == 10500.0,
      f"UniMate auto base rule (int(8L/pi)//100+1)*100: L=241 -> {unimate_auto_base(241)}, L=4096 -> {unimate_auto_base(4096)}")
dh, T, base = 64, 241, 700.0
cos, sin = sinusoidal_cos_sin(T, dh, base)
check(tuple(cos.shape) == (T, dh) and cos.dtype == torch.float32, f"cos/sin shapes {tuple(cos.shape)}")
# the table is UniMate's: inv_freq = base^(-2i/dh), concatenated with itself
inv = 1.0 / (base ** (torch.arange(0, dh, 2).float() / dh))
ref = torch.outer(torch.arange(T).float(), inv); ref = torch.cat([ref, ref], -1)
check(torch.allclose(cos, ref.cos()) and torch.allclose(sin, ref.sin()), "cos/sin == UniMate's RopeND table for this base")
check(torch.equal(cos[0], torch.ones(dh)) and torch.equal(sin[0], torch.zeros(dh)), "position 0 is the identity rotation")
q = torch.randn(2, 3, T, dh); k = torch.randn(2, 3, T, dh)
rq, rk = apply_rotary_pos_emb(q, k, cos[None, None], sin[None, None])
check(torch.allclose(rq.norm(dim=-1), q.norm(dim=-1), atol=1e-5), "rotation preserves the q/k norms")
worst = 0.0
for m, n in ((0, 0), (5, 3), (100, 7), (240, 1), (3, 200)):
    lhs = float(rq[0, 0, m] @ rk[0, 0, n]); d = m - n
    c_, s_ = sinusoidal_cos_sin(abs(d) + 1, dh, base)
    c_, s_ = (c_[abs(d)], s_[abs(d)] if d >= 0 else -s_[abs(d)])
    rhs = float(apply_rotary_pos_emb(q[0, 0, m:m + 1], q[0, 0, m:m + 1], c_[None], s_[None])[0][0] @ k[0, 0, n])
    worst = max(worst, abs(lhs - rhs))
check(worst < 1e-4, f"relative identity <R(m)q, R(n)k> == <R(m-n)q, k> (worst {worst:.1e}) -- the encoding is relative, not absolute")
slow = float((T - 1) * inv[-1] / (2 * math.pi))
check(0.03 < slow < 0.1, f"the slowest channel turns {slow:.3f} of a revolution across the window (UniMate's rule aims at ~1/16; nothing wraps)")
for bad, why in ((1.0, "base 1"), (0.0, "base 0")):
    try:
        sinusoidal_cos_sin(8, dh, bad); check(False, f"refusal: {why} raised")
    except ValueError: check(True, f"refusal: {why} raised ValueError")
try:
    sinusoidal_cos_sin(8, 63, base); check(False, "refusal: odd head_dim raised")
except ValueError: check(True, "refusal: odd head_dim raised ValueError")

# ---------------- 2. model: off-path parity, on-path structure ----------------
KWM = dict(in_ch=17, dim=384, depth=8, n_heads=6, d_text=4096, d_joint_sem=4096,
           use_struct_feats=True, use_dir_bias=True, qk_norm=True, use_geo_bias=True)

def hops(parents):
    n = len(parents); D = np.full((n, n), 99); np.fill_diagonal(D, 0)
    for j, p in enumerate(parents):
        if p >= 0: D[j, p] = D[p, j] = 1
    for m in range(n): D = np.minimum(D, D[:, m:m + 1] + D[m:m + 1, :])
    return D

ASYM = np.array([-1, 0, 1, 2, 3, 1, 5, 0, 7, 8, 9, 7, 11], dtype=np.int64)

def make_inputs(seed, parents, B=2, Tn=6):
    g = torch.Generator().manual_seed(seed); Jn = len(parents)
    geo = torch.from_numpy(hops(parents)).float()
    d = dict(x=torch.randn(B, Tn, Jn, 17, generator=g), t=torch.rand(B, generator=g),
             is_target=torch.zeros(B, Tn, dtype=torch.bool), joint_sem=torch.randn(B, Jn, 4096, generator=g),
             text=torch.randn(B, 4096, generator=g), joint_bias=(-geo.clamp(max=8))[None].repeat(B, 1, 1),
             frame_valid=torch.ones(B, Tn, dtype=torch.bool), joint_valid=torch.ones(B, Jn, dtype=torch.bool),
             struct_feats=torch.randn(B, Jn, 8, generator=g), updown=torch.randint(0, 16, (B, Jn, Jn, 2), generator=g))
    d["is_target"][:, 1:] = True
    return d

def run(model, d, **extra):
    kw = {k: v for k, v in d.items() if k not in ("x", "t")}; kw.update(extra)
    return model(d["x"], d["t"], **kw)

def build(mod, seed, **kw):
    torch.manual_seed(seed); return mod.InContextMotionDiT(**KWM, **kw)

off_prev, off = build(prev, 0), build(sys.modules["src.models.v2.dit_motion"], 0)
sp, sn_ = off_prev.state_dict(), off.state_dict()
check(list(sp.keys()) == list(sn_.keys()) and all(torch.equal(sp[k], sn_[k]) for k in sp),
      f"off-path: state_dict keys and every tensor bitwise identical to the pre-change module ({len(sp)} tensors)")
D0 = make_inputs(1, ASYM)
off_prev.eval(); off.eval()
with torch.no_grad():
    check(torch.equal(run(off_prev, D0), run(off, D0)), "off-path: eval forward bitwise identical to the pre-change module")
for m in (off_prev, off): m.train(); m.grad_ckpt = True
outs, grads = [], []
for m in (off_prev, off):
    m.zero_grad(); o = run(m, D0); o.square().mean().backward()
    outs.append(o.detach()); grads.append({n: p.grad.clone() for n, p in m.named_parameters() if p.grad is not None})
check(torch.equal(outs[0], outs[1]) and grads[0].keys() == grads[1].keys() and all(torch.equal(grads[0][n], grads[1][n]) for n in grads[0]),
      "off-path: train forward + grads through activation checkpointing bitwise identical")
for m in (off_prev, off): m.grad_ckpt = False; m.eval()

on = build(sys.modules["src.models.v2.dit_motion"], 0, use_temporal_rope=True, trope_base=700.0)
so = on.state_dict()
missing = [k for k in sn_ if k not in so]; extra = [k for k in so if k not in sn_]
check(missing == ["t_pos"], f"on-path: the only key gone is t_pos (missing {missing})")
check(extra == ["trope_base"] and float(so["trope_base"]) == 700.0,
      f"on-path: the only key added is the trope_base buffer ({extra}, value {float(so['trope_base']) if extra else None})")
shared = [k for k in sn_ if k in so]
check(all(torch.equal(sn_[k], so[k]) for k in shared),
      f"on-path: every shared tensor bitwise identical to the off-path model under the same seed ({len(shared)} tensors; the dropped t_pos draw still consumed)")
n_off = sum(p.numel() for p in off.parameters()); n_on = sum(p.numel() for p in on.parameters())
print(f"params: off-path {n_off:,} on-path {n_on:,} (t_pos {4096*384:,} removed, nothing added)")
check(n_off - n_on == 4096 * 384, f"on-path drops exactly the t_pos table ({n_off - n_on:,})")
on.eval()
with torch.no_grad(): o_on = run(on, D0)
check(tuple(o_on.shape) == tuple(D0["x"].shape) and torch.isfinite(o_on).all(), "on-path: eval forward runs, finite, right shape")
for a, b, name in ((off, on, "rotary weights into a learned-table model"), (on, off, "learned-table weights into a rotary model")):
    try:
        a.load_state_dict(b.state_dict()); check(False, f"strict load: {name} refused")
    except RuntimeError as e:
        check("t_pos" in str(e), f"strict load: {name} refused (names t_pos)")
# the base travels with the weights (persistent buffer) and a MATCHING rebuild reproduces the output exactly;
# a mismatched one is refused rather than silently corrected -- see the base-drift check further down
same = build(sys.modules["src.models.v2.dit_motion"], 7, use_temporal_rope=True, trope_base=700.0)
same.load_state_dict(so); same.eval()
with torch.no_grad(): check(torch.equal(run(same, D0), o_on), "on-path: a differently-seeded model with the SAME base reproduces the output bitwise after loading")
on.train(); on.grad_ckpt = False
on.zero_grad(); o1 = run(on, D0); o1.square().mean().backward(); g1 = {n: p.grad.clone() for n, p in on.named_parameters() if p.grad is not None}
on.grad_ckpt = True
on.zero_grad(); o2 = run(on, D0); o2.square().mean().backward(); g2 = {n: p.grad.clone() for n, p in on.named_parameters() if p.grad is not None}
on.grad_ckpt = False; on.eval()
check(g1.keys() == g2.keys() and torch.allclose(o1.detach(), o2.detach(), atol=1e-6) and all(torch.allclose(g1[n], g2[n], atol=1e-6) for n in g1),
      f"on-path: activation-checkpoint path == plain path (bitwise {torch.equal(o1.detach(), o2.detach())})")
nog = [n for n, p in on.named_parameters() if p.grad is None]
check(all(n.startswith("bp_mlp.") for n in nog), f"on-path: every parameter except the unused blueprint MLP receives gradient (none: {nog})")
# the temporal position actually reaches the output, and ONLY through the rotation
with torch.no_grad():
    d_rev = dict(D0); d_rev["x"] = D0["x"].flip(1); d_rev["is_target"] = D0["is_target"].flip(1)
    check(float((run(on, d_rev) - run(on, D0).flip(1)).abs().max()) > 1e-4,
          "on-path: the model is NOT time-symmetric -- the rotation carries frame order")
# past the learned table's length the rotary model still runs; the table model cannot
SMALL = dict(in_ch=17, dim=64, depth=1, n_heads=2, d_text=4096, d_joint_sem=4096, qk_norm=True)
torch.manual_seed(0); s_tab = InContextMotionDiT(**SMALL)
torch.manual_seed(0); s_rot = InContextMotionDiT(**SMALL, use_temporal_rope=True, trope_base=700.0)
long_T = s_tab.max_T + 8
Dl = make_inputs(3, ASYM[:4], B=1, Tn=long_T)
s_tab.eval(); s_rot.eval()
with torch.no_grad():
    try:
        run(s_tab, Dl); tab_ok = True
    except Exception as e:
        tab_ok = False; tab_err = type(e).__name__
    o_long = run(s_rot, Dl)
check(not tab_ok, f"beyond max_T={s_tab.max_T} the learned table cannot address the frames ({tab_err if not tab_ok else 'it ran'})")
check(torch.isfinite(o_long).all() and o_long.shape[1] == long_T, f"the rotary model runs at T={long_T} (finite output)")

# ---------------- 3. trainer / launcher / calibration guards ----------------
h = subprocess.run([sys.executable, "scripts/train_v2_incontext.py", "--help"], capture_output=True, text=True).stdout
check("--temporal_rope" in h and "--trope_base" in h, "trainer: --temporal_rope / --trope_base in --help")
src = open("scripts/train_v2_incontext.py").read()
check('"temporal_rope", "trope_base")' in src, "trainer: both flags pinned in the resume-critical list")
lsrc = open("scripts/_launch_v2_ddp_2node_h200.sh").read()
check('$([ "$TEMPORAL_ROPE" = 1 ] && echo --temporal_rope --trope_base $TROPE_BASE)' in lsrc,
      "launcher: emits --temporal_rope --trope_base from TEMPORAL_ROPE")
_dup = lsrc[lsrc.index("for dup in"):lsrc.index("done", lsrc.index("for dup in"))]
check("--temporal_rope" in _dup and "--trope_base" in _dup, "launcher: both flags are in the EXTRA duplicate guard")
artA = json.load(open("configs/pilot_animal_heldout_rest_gamma_calibration_b16_v1.json"))
full4 = {k: artA["protocol"]["verify"]["arm_model"][k] for k in ("struct_feats", "dir_bias", "geo_bias", "freeze_zero_joint_sem")}
base_want = dict(full4, spec_rope=False, spec_rope_k=8)
check(trm.calib_arm_model_drift(artA, dict(base_want, temporal_rope=False, trope_base=700.0), False) is None,
      "guard: an artifact without the temporal fields == measured without the temporal RoPE (legacy default)")
check(trm.calib_arm_model_drift(artA, dict(base_want, temporal_rope=True, trope_base=700.0), False) is not None,
      "guard: ... and is refused by a temporal-RoPE run")
artT = json.loads(json.dumps(artA)); artT["protocol"]["verify"]["arm_model"].update(temporal_rope=True, trope_base=700.0)
check(trm.calib_arm_model_drift(artT, dict(base_want, temporal_rope=True, trope_base=700.0), False) is None
      and trm.calib_arm_model_drift(artT, dict(base_want, temporal_rope=True, trope_base=10000.0), False) is not None
      and trm.calib_arm_model_drift(artT, dict(base_want, temporal_rope=False, trope_base=700.0), False) is not None,
      "guard: a temporal artifact is accepted by the matching run and refused by another base or by a table run")
from pathlib import Path
import hashlib
REPO = Path(".").resolve(); MS = REPO / "scripts/_measure_ktjd17_gamma_calibration_view_v2.py"
h00 = trm.calib_code_sha256(REPO, MS, False, False); h10 = trm.calib_code_sha256(REPO, MS, True, False)
h01 = trm.calib_code_sha256(REPO, MS, False, True); h11 = trm.calib_code_sha256(REPO, MS, True, True)
ref00 = hashlib.sha256((REPO / "src/models/v2/dit_motion.py").read_bytes() + MS.read_bytes()).hexdigest()
ref01 = hashlib.sha256((REPO / "src/models/v2/dit_motion.py").read_bytes() + MS.read_bytes()
                       + (REPO / "src/models/v2/spec_rope.py").read_bytes()
                       + (REPO / "src/models/v2/temporal_rope.py").read_bytes()).hexdigest()
check(h00 == ref00 and h01 == ref01 and len({h00, h10, h01, h11}) == 4,
      "calibration code hash: unchanged for a no-rotary arm; the temporal arm appends the shared rotation AND its own source")
ENV = dict(os.environ, DIM="384", DEPTH="8", HEADS="6", QK_NORM="1", STRUCT_FEATS="1", DIR_BIAS="1", ARM_GEO_BIAS="1",
           ARM_FREEZE_ZERO_JOINT_SEM="0", GAMMA_SOLVE=str(artA["protocol"]["gamma_solve"]),
           VERIFY_STEPS=str(artA["protocol"]["verify"]["steps"]), EXPECT_CODE_SCRIPT=artA["hashes"]["code_script"])
for k in list(ENV):
    if k.startswith("AUG_") or k.startswith("ARM_SPEC") or k.startswith("ARM_TROPE") or k.startswith("ARM_TEMPORAL"): ENV.pop(k)
def check_rc(art, name, **env):
    pth = f"runs/_heldout/_aug_dev_logs/_tmp_trope_art_{name}.json"; json.dump(art, open(pth, "w"))
    r = subprocess.run([sys.executable, "scripts/_calib_artifact_check.py", pth], env=dict(ENV, **env), capture_output=True, text=True)
    os.remove(pth); return r.returncode, (r.stdout + r.stderr).strip()[-160:]
artT2 = json.loads(json.dumps(artA)); artT2["protocol"]["verify"]["arm_model"].update(spec_rope=False, spec_rope_k=8, temporal_rope=True, trope_base=700.0)
rc, msg = check_rc(artT2, "trope", ARM_TEMPORAL_ROPE="1", ARM_TROPE_BASE="700"); check(rc == 0, f"artifact check: the temporal artifact is accepted for the temporal arm ({msg})")
rc, msg = check_rc(artT2, "trope_for_table"); check(rc == 1, f"artifact check: it is refused for a learned-table arm ({msg})")
artB2 = json.loads(json.dumps(artA)); artB2["protocol"]["verify"]["arm_model"].update(spec_rope=False, spec_rope_k=8, temporal_rope=True, trope_base=10000.0)
rc, msg = check_rc(artB2, "wrong_base", ARM_TEMPORAL_ROPE="1", ARM_TROPE_BASE="700"); check(rc == 1, f"artifact check: another base is refused ({msg})")
# ---- codex trope r1 fixes ----
# P2-6: the cos/sin table stays float32 whatever autocast does to the activations
class _Probe(torch.nn.Module):
    pass
with torch.autocast("cpu", dtype=torch.bfloat16):
    cs16, sn16 = sinusoidal_cos_sin(16, dh, base)
check(cs16.dtype == torch.float32 and sn16.dtype == torch.float32, "the table is float32 even inside autocast")
_bf = sinusoidal_cos_sin(241, dh, base)[0].to(torch.bfloat16).float()
_bs = sinusoidal_cos_sin(241, dh, base)[1].to(torch.bfloat16).float()
check(float((cos.float() ** 2 + sin.float() ** 2 - 1).abs().max()) < 1e-6 < float((_bf ** 2 + _bs ** 2 - 1).abs().max()),
      f"float32 stays on the unit circle ({float((cos**2+sin**2-1).abs().max()):.1e}) where bf16 would not ({float((_bf**2+_bs**2-1).abs().max()):.1e})")
_saw = []
_orig = sys.modules["src.models.v2.dit_motion"].sinusoidal_cos_sin
def _spy(n, d, b, *, device=None, dtype=None):
    _saw.append(dtype); return _orig(n, d, b, device=device, dtype=dtype)
sys.modules["src.models.v2.dit_motion"].sinusoidal_cos_sin = _spy
with torch.autocast("cpu", dtype=torch.bfloat16), torch.no_grad():
    run(on, D0)
sys.modules["src.models.v2.dit_motion"].sinusoidal_cos_sin = _orig
check(_saw and all(d == torch.float32 for d in _saw), f"the model asks for a float32 table under autocast (saw {_saw})")
# P1-2: a checkpoint whose base differs from the constructed one is refused, not silently honoured
other = build(sys.modules["src.models.v2.dit_motion"], 0, use_temporal_rope=True, trope_base=10000.0)
try:
    other.load_state_dict(so); check(False, "base drift: loading a base-700 checkpoint into a base-10000 model refused")
except RuntimeError as e:
    check("trope_base" in str(e) and "700" in str(e), f"base drift refused ({str(e)[:90]}...)")
check(torch.equal(build(sys.modules["src.models.v2.dit_motion"], 0, use_temporal_rope=True, trope_base=700.0).state_dict()["trope_base"],
                  so["trope_base"]), "... while the matching base still loads")
# P1-1: the rotation itself is hashed for BOTH rotary arms
check(trm.rotary_code_files(False, False) == [] and
      trm.rotary_code_files(True, False) == ["src/models/v2/spec_rope.py", "src/data/skeleton_spectral.py"] and
      trm.rotary_code_files(False, True) == ["src/models/v2/spec_rope.py", "src/models/v2/temporal_rope.py"],
      f"rotary_code_files: spec_rope.py (the rotation) is listed for either arm, the coordinate sources only for their own")
# P1-4: the flat baseline refuses a rotary flag. The trainer's CPU guard fires before this one in a real run, so the
# guard's own table is read out of the source and its refusal exercised directly on the same tuple.
_g = src[src.index("if a.flat_joints:"):src.index("if a.geo_bias:", src.index("if a.flat_joints:"))]
_pairs = _g[_g.index("for _bad, _why in ("):_g.index("):", _g.index("for _bad, _why in ("))]
check('"spec_rope"' in _pairs and '"temporal_rope"' in _pairs,
      f"trainer's flat-baseline guard rejects both rotary flags (table: {[t for t in ('two_stage','struct_feats','dir_bias','ref_text','spec_rope','temporal_rope') if chr(34)+t+chr(34) in _pairs]})")
check('raise SystemExit(f"[refuse] --flat_joints with --{_bad}' in _g, "... and raises rather than warning")
# REGRESSION (arm E's first launch died here): under torch.compile the forward must not read the base out of the
# tensor buffer -- on the first RECOMPILATION (a second sequence length) the traced value is not a real number and the
# table builder's range check raised. Two different T values through a compiled module is the reproducer.
_small = dict(in_ch=17, dim=64, depth=1, n_heads=2, d_text=4096, d_joint_sem=4096, qk_norm=True)
torch.manual_seed(0); _c = InContextMotionDiT(**_small, use_temporal_rope=True, trope_base=700.0).eval()
check(isinstance(getattr(_c, "trope_base_value", None), float) and _c.trope_base_value == 700.0,
      "the base is kept as a python float for the forward (and as a buffer for the checkpoint)")
try:
    _cc = torch.compile(_c, dynamic=True)
    with torch.no_grad():
        for _T in (6, 11):
            _o = run(_cc, make_inputs(5, ASYM[:4], B=1, Tn=_T))
            assert _o.shape[1] == _T and torch.isfinite(_o).all()
    check(True, "compiled forward survives a recompilation at a second sequence length (the crash reproducer)")
except Exception as _e:
    check(False, f"compiled forward raised at a second sequence length: {type(_e).__name__}: {str(_e)[:120]}")

# r2 P1: the flat consumers refuse a rotary flag BEFORE building FlatMotionDiT
for _f, _tag in (("scripts/_eval_v2_gen_in_evalspace.py", "eval"), ("scripts/v2_render_incontext.py", "render")):
    _t = open(_f).read()
    _i = _t.index("FlatMotionDiT(")
    _before = _t[:_i]
    check("a flat checkpoint records a rotary flag" in _before and
          _before.rindex("a flat checkpoint records a rotary flag") > _before.rindex("flat_joints") - 2000,
          f"{_tag}: a flat checkpoint recording a rotary flag is refused before FlatMotionDiT is built")
# r2 P2: a half-present rotary pair is malformed, not legacy
_artA = json.load(open("configs/pilot_animal_heldout_rest_gamma_calibration_b16_v1.json"))
_f4 = {k: _artA["protocol"]["verify"]["arm_model"][k] for k in ("struct_feats", "dir_bias", "geo_bias", "freeze_zero_joint_sem")}
_w = dict(_f4, spec_rope=False, spec_rope_k=8, temporal_rope=False, trope_base=700.0)
check(trm.calib_arm_model_drift(_artA, _w, False) is None, "legacy pair: an artifact missing BOTH members of each pair is completed")
for _k, _other in (("temporal_rope", "trope_base"), ("spec_rope", "spec_rope_k")):
    _half = json.loads(json.dumps(_artA)); _half["protocol"]["verify"]["arm_model"][_k] = True
    _msg = trm.calib_arm_model_drift(_half, dict(_w, **{_k: True}), False)
    check(_msg is not None and _other in _msg, f"legacy pair: {_k} without {_other} is refused as malformed")
    _pth = "runs/_heldout/_aug_dev_logs/_tmp_trope_half.json"; json.dump(_half, open(_pth, "w"))
    _r = subprocess.run([sys.executable, "scripts/_calib_artifact_check.py", _pth],
                        env=dict(ENV, ARM_SPEC_ROPE="1" if _k == "spec_rope" else "0", ARM_TEMPORAL_ROPE="1" if _k == "temporal_rope" else "0"),
                        capture_output=True, text=True)
    os.remove(_pth)
    check(_r.returncode == 1, f"artifact check: {_k} without {_other} is refused ({(_r.stdout + _r.stderr).strip()[-80:]})")
# P1-3: the arm's config pins the joint rotary off even against a hostile environment
_cfg = subprocess.run(["bash", "-c", "set -a; . configs/pilot36m_heldout_trope_2node_env.sh; set +a; echo $SPEC_ROPE/$TEMPORAL_ROPE"],
                      env=dict(os.environ, SPEC_ROPE="1"), capture_output=True, text=True).stdout.strip()
check(_cfg.endswith("0/1"), f"the arm config pins SPEC_ROPE=0 against an inherited SPEC_ROPE=1 (got {_cfg})")
# P2-5: every finished held-out report's fingerprint is live or audited
try:
    import scripts._eval_v2_gen_in_evalspace as ev
    f00, f01 = ev.source_fingerprint("none", False, False, False), ev.source_fingerprint("none", False, False, True)
    check(f00 != f01 and "temporal_rope=bool(ca.get(\"temporal_rope\", False))" in open("scripts/_eval_v2_gen_in_evalspace.py").read(),
          "generation fingerprint: a temporal-RoPE checkpoint adds temporal_rope.py, keyed by the checkpoint's arg")
    import glob
    live = {ev.source_fingerprint("none", fl, sr, tr) for fl in (0, 1) for sr in (0, 1) for tr in (0, 1)}
    miss = []
    for f in sorted(glob.glob("runs/_heldout/eval/geneval_*_pool64_strict*.json")):
        g = json.load(open(f))["protocol"]["generation"]
        for k in ("source_fingerprint", "scoring_source_fingerprint"):
            h = g.get(k)
            if h and h not in live and h not in ev.LEGACY_SOURCE_FINGERPRINTS:
                miss.append((os.path.basename(f), k, h[:16]))
    check(not miss, f"every finished held-out report's generation fingerprint is live or audited ({miss[:3]})")
except Exception as e:
    check(False, f"eval fingerprint test raised {e!r}")

print(f"\n{'ALL PASS' if not fails else f'{len(fails)} FAIL'} ({time.time() - T0:.0f} s)")
for f in fails: print(" -", f)
sys.exit(1 if fails else 0)
