"""Arms G/H: BOTH rotaries at once -- the spectral joint RoPE (arm D) and the sinusoidal temporal RoPE (arm E) in the
same model, so no learned position table survives on either axis. Each rotary was verified alone
(_test_spec_rope.py, _test_temporal_rope.py); this file checks only what the COMBINATION can break: the init draws of
both dropped tables, both rotations reaching their own attention, the two properties surviving each other, the
checkpoint contract with both keys gone and both added, and every guard that keys off the pair of flags.
Run inside an allocation, on CPU:
`srun ... /usr/bin/env CUDA_VISIBLE_DEVICES= python scripts/_aug_dev/_test_both_rope.py`."""
import sys, os, json, math, re, ast, shlex, argparse, textwrap, subprocess, time
from torch.utils.data import DistributedSampler
import numpy as np, torch
sys.path.insert(0, ".")
torch.set_num_threads(8)
from src.data.skeleton_spectral import laplacian_eigenvectors
from src.models.v2.dit_motion import InContextMotionDiT
import scripts.train_v2_incontext as trm

fails = []
def check(cond, msg):
    if not cond: fails.append(msg); print("FAIL:", msg)
    else: print("ok:", msg)
T0 = time.time()
rng = np.random.default_rng(0)
# the three text/AST assertions in section 8 read their file through these, so the mutation check below can point
# them at a deliberately broken copy and confirm they FAIL -- a check that survives deleting what it verifies is not
# a check (codex bothrope r1 P2).
LAUNCH_SH = os.environ.get("TEST_LAUNCHER", "scripts/_launch_v2_ddp_2node_h200.sh")
TRAIN_PY = os.environ.get("TEST_TRAINER", "scripts/train_v2_incontext.py")
CFG_G = os.environ.get("TEST_CFG_G", "configs/pilot36m_heldout_bothrope_2node_env.sh")
CFG_H = os.environ.get("TEST_CFG_H", "configs/pilot36m_heldout_bothrope_aug_2node_env.sh")
CALIB_G = os.environ.get("TEST_CALIB_G", "scripts/_calib_heldout_bothrope_b16_v1.sh")
CALIB_H = os.environ.get("TEST_CALIB_H", "scripts/_calib_heldout_bothrope_aug_b16_v1.sh")
EVAL_PY = os.environ.get("TEST_EVAL", "scripts/_eval_v2_gen_in_evalspace.py")

KWM = dict(in_ch=17, dim=384, depth=8, n_heads=6, d_text=4096, d_joint_sem=4096,
           use_struct_feats=True, use_dir_bias=True, qk_norm=True, use_geo_bias=True)
SPEC = dict(use_spec_rope=True, spec_rope_k=8)
TROPE = dict(use_temporal_rope=True, trope_base=700.0)

def hops(parents):
    n = len(parents); D = np.full((n, n), 99); np.fill_diagonal(D, 0)
    for j, p in enumerate(parents):
        if p >= 0: D[j, p] = D[p, j] = 1
    for m in range(n): D = np.minimum(D, D[:, m:m + 1] + D[m:m + 1, :])
    return D

ASYM = np.array([-1, 0, 1, 2, 3, 1, 5, 0, 7, 8, 9, 7, 11], dtype=np.int64)

def make_inputs(seed, parents, B=2, Tn=6, K=8):
    g = torch.Generator().manual_seed(seed); Jn = len(parents)
    geo = torch.from_numpy(hops(parents)).float()
    d = dict(x=torch.randn(B, Tn, Jn, 17, generator=g), t=torch.rand(B, generator=g),
             is_target=torch.zeros(B, Tn, dtype=torch.bool), joint_sem=torch.randn(B, Jn, 4096, generator=g),
             text=torch.randn(B, 4096, generator=g), joint_bias=(-geo.clamp(max=8))[None].repeat(B, 1, 1),
             frame_valid=torch.ones(B, Tn, dtype=torch.bool), joint_valid=torch.ones(B, Jn, dtype=torch.bool),
             struct_feats=torch.randn(B, Jn, 8, generator=g), updown=torch.randint(0, 16, (B, Jn, Jn, 2), generator=g),
             spectral_feats=torch.from_numpy(laplacian_eigenvectors(parents, K)[0])[None].repeat(B, 1, 1))
    d["is_target"][:, 1:] = True
    return d

def run(model, d, **extra):
    kw = {k: v for k, v in d.items() if k not in ("x", "t")}; kw.update(extra)
    return model(d["x"], d["t"], **kw)

def build(seed, **kw):
    torch.manual_seed(seed); return InContextMotionDiT(**KWM, **kw)

# ---------------- 1. the four models under ONE seed ----------------
# A = neither rotary (the baseline geometry), D = spectral only, E = temporal only, G = both.
mA, mD, mE, mG = build(0), build(0, **SPEC), build(0, **TROPE), build(0, **SPEC, **TROPE)
sA, sD, sE, sG = (m.state_dict() for m in (mA, mD, mE, mG))
gone = [k for k in sA if k not in sG]
added = [k for k in sG if k not in sA]
check(sorted(gone) == ["j_pos", "t_pos"], f"both models' tables are gone and nothing else ({sorted(gone)})")
check(sorted(added) == sorted(["trope_base"] + [k for k in sG if k.startswith("spec_rope.")]),
      f"exactly the SignNet weights and the trope_base buffer are added ({len(added)} keys)")
# THE load-bearing one: dropping TWO tables must not shift any other parameter's init draw. Both draws are still
# taken (dit_motion.py builds both tensors, then registers each only if its rotary is off), so every shared tensor
# must be bitwise what the baseline got under the same seed -- and the SignNet, built last, must match arm D's.
shared = [k for k in sA if k in sG]
check(all(torch.equal(sA[k], sG[k]) for k in shared),
      f"init parity: every shared tensor bitwise identical to the no-rotary model under the same seed "
      f"({len(shared)} tensors; BOTH dropped tables' normal draws still consumed, in order)")
sig = [k for k in sG if k.startswith("spec_rope.")]
check(all(torch.equal(sD[k], sG[k]) for k in sig),
      f"init parity: the SignNet is bitwise arm D's under the same seed ({len(sig)} tensors; dropping t_pos as well "
      f"did not move the draws it is built from)")
nA, nD, nE, nG = (sum(p.numel() for p in m.parameters()) for m in (mA, mD, mE, mG))
print(f"params: baseline {nA:,} | spectral {nD:,} | temporal {nE:,} | BOTH {nG:,}")
check(nA - nG == (nA - nD) + (nA - nE),
      f"the combined arm's parameter count is exactly the two deltas summed "
      f"({nA - nG:,} == {nA - nD:,} + {nA - nE:,}) -- no double count, nothing left behind")
n_tpos, n_jpos, n_sig = sA["t_pos"].numel(), sA["j_pos"].numel(), sum(sG[k].numel() for k in sig)
check(nG == nA - n_tpos - n_jpos + n_sig,
      f"...and equals baseline - t_pos({n_tpos:,}) - j_pos({n_jpos:,}) + SignNet({n_sig:,}) = {nG:,}")

# ---------------- 2. the combined forward ----------------
D0 = make_inputs(1, ASYM)
for m in (mA, mD, mE, mG): m.eval()
with torch.no_grad():
    oG = run(mG, D0); oD = run(mD, D0); oE = run(mE, {k: v for k, v in D0.items() if k != "spectral_feats"})
check(tuple(oG.shape) == tuple(D0["x"].shape) and torch.isfinite(oG).all(), "combined: eval forward runs, finite, right shape")
check(float((oG - oD).abs().max()) > 1e-4 and float((oG - oE).abs().max()) > 1e-4,
      f"combined: the output is neither arm's (|G-D| {float((oG-oD).abs().max()):.2e}, |G-E| {float((oG-oE).abs().max()):.2e})")
# both rotations are LIVE in the same model and each reaches only its own axis
with torch.no_grad():
    alt = D0["spectral_feats"][:, torch.randperm(13)]
    d_spec = float((run(mG, D0, spectral_feats=alt) - oG).abs().max())
check(d_spec > 1e-5, f"combined: the spectral coordinates still reach the output ({d_spec:.2e}) -- the temporal rotation did not shadow them")
mG2 = build(0, **SPEC, use_temporal_rope=True, trope_base=1300.0)
mG2.load_state_dict({k: v for k, v in sG.items() if k != "trope_base"}, strict=False); mG2.eval()
with torch.no_grad(): d_base = float((run(mG2, D0) - oG).abs().max())
check(d_base > 1e-5, f"combined: the temporal base still reaches the output ({d_base:.2e} at base 1300 vs 700) -- the spectral rotation did not shadow it")
with torch.no_grad():
    d_rev = dict(D0); d_rev["x"] = D0["x"].flip(1); d_rev["is_target"] = D0["is_target"].flip(1)
    d_tsym = float((run(mG, d_rev) - oG.flip(1)).abs().max())
check(d_tsym > 1e-4, f"combined: NOT time-symmetric ({d_tsym:.2e}) -- frame order still carried")

# ---------------- 3. joint-permutation equivariance survives the temporal rotation ----------------
def permute(d, order):
    o = torch.as_tensor(order)
    e = dict(d); e["x"] = d["x"][:, :, o]; e["joint_sem"] = d["joint_sem"][:, o]; e["joint_bias"] = d["joint_bias"][:, o][:, :, o]
    e["struct_feats"] = d["struct_feats"][:, o]; e["updown"] = d["updown"][:, o][:, :, o]; e["spectral_feats"] = d["spectral_feats"][:, o]
    e["joint_valid"] = d["joint_valid"][:, o]
    return e
worst_on, worst_off = 0.0, 0.0
D0off = {k: v for k, v in D0.items() if k != "spectral_feats"}
with torch.no_grad():
    base_off = run(mE, D0off)
    for _ in range(3):
        order = rng.permutation(13)
        Dp = permute(D0, order)
        worst_on = max(worst_on, float((run(mG, Dp) - oG[:, :, order]).abs().max()))
        worst_off = max(worst_off, float((run(mE, {k: v for k, v in Dp.items() if k != "spectral_feats"}) - base_off[:, :, order]).abs().max()))
check(worst_on < 1e-4, f"combined: joint-permutation EQUIVARIANT ({worst_on:.2e}; scale {float(oG.abs().max()):.2f}) -- arm D's property survives arm E")
check(worst_off > 1e-3, f"temporal-only (slot table): NOT equivariant ({worst_off:.2e}) -- the contrast the combination removes")

# ---------------- 4. training path, gradients, checkpoint contract ----------------
mG.train(); mG.grad_ckpt = False
mG.zero_grad(); o1 = run(mG, D0); o1.square().mean().backward()
g1 = {n: p.grad.clone() for n, p in mG.named_parameters() if p.grad is not None}
mG.grad_ckpt = True
mG.zero_grad(); o2 = run(mG, D0); o2.square().mean().backward()
g2 = {n: p.grad.clone() for n, p in mG.named_parameters() if p.grad is not None}
mG.grad_ckpt = False; mG.eval()
check(g1.keys() == g2.keys() and torch.allclose(o1.detach(), o2.detach(), atol=1e-6)
      and all(torch.allclose(g1[n], g2[n], atol=1e-6) for n in g1),
      f"combined: activation-checkpoint path == plain path (bitwise {torch.equal(o1.detach(), o2.detach())})")
nog = [n for n, p in mG.named_parameters() if p.grad is None]
check(all(n.startswith("bp_mlp.") for n in nog), f"combined: every parameter except the unused blueprint MLP receives gradient (none: {nog})")
gs = {n: float(g1[n].abs().max()) for n in g1 if n.startswith("spec_rope.")}
check(gs and all(v > 0 for v in gs.values()), f"combined: every SignNet parameter receives gradient ({min(gs.values()):.2e}..{max(gs.values()):.2e})")
# cross loads: the combined checkpoint must not be silently accepted by, or accept, a single-rotary model
for a, b, name, want in ((mG, mD, "spectral-only weights into a combined model", "t_pos"),
                         (mD, mG, "combined weights into a spectral-only model", "t_pos"),
                         (mG, mE, "temporal-only weights into a combined model", "j_pos"),
                         (mE, mG, "combined weights into a temporal-only model", "j_pos"),
                         (mG, mA, "baseline weights into a combined model", "j_pos"),
                         (mA, mG, "combined weights into a baseline model", "j_pos")):
    try:
        a.load_state_dict(b.state_dict()); check(False, f"strict load: {name} refused")
    except RuntimeError as e:
        check(want in str(e), f"strict load: {name} refused (names {want})")
pth = os.path.join(os.environ.get("SCRATCH_TMP", "runs/_heldout/_aug_dev_logs"), "_tmp_bothrope_sd.pt")
torch.save(sG, pth)
mR = build(7, **SPEC, **TROPE); mR.load_state_dict(torch.load(pth)); mR.eval(); os.remove(pth)
with torch.no_grad(): check(torch.equal(run(mR, D0), oG), "combined: state_dict round trip into a differently-seeded model reproduces the output bitwise")

# ---------------- 5. the crash reproducer, with both rotaries ----------------
SMALL = dict(in_ch=17, dim=64, depth=1, n_heads=2, d_text=4096, d_joint_sem=4096, qk_norm=True)
torch.manual_seed(0); mc = InContextMotionDiT(**SMALL, **SPEC, **TROPE).eval()
try:
    cc = torch.compile(mc, dynamic=True)
    with torch.no_grad():
        for Tn in (6, 11):
            o = run(cc, make_inputs(5, ASYM[:4], B=1, Tn=Tn))
            assert o.shape[1] == Tn and torch.isfinite(o).all()
    check(True, "combined: compiled forward survives a recompilation at a second sequence length (arm E's crash reproducer)")
except Exception as e:
    check(False, f"combined: compiled forward raised at a second sequence length: {type(e).__name__}: {str(e)[:140]}")

# ---------------- 6. the hashes and the artifact binding ----------------
# (the four-way calibration-hash separation, the trope_base buffer value, the base-drift refusal and the python-float
# storage all live in _test_temporal_rope.py and are not repeated here -- codex bothrope r2)
files_both = trm.rotary_code_files(True, True)
check(files_both == ["src/models/v2/spec_rope.py", "src/data/skeleton_spectral.py", "src/models/v2/temporal_rope.py"],
      f"rotary_code_files(True, True) lists all three sources once each, in order ({files_both})")
# an artifact measured on either single-rotary model must not certify the combined arm, and vice versa
artA = json.load(open("configs/pilot_animal_heldout_rest_gamma_calibration_b16_v1.json"))
f4 = {k: artA["protocol"]["verify"]["arm_model"][k] for k in ("struct_feats", "dir_bias", "geo_bias", "freeze_zero_joint_sem")}
want_both = dict(f4, spec_rope=True, spec_rope_k=8, temporal_rope=True, trope_base=700.0)
def art_with(**over):
    a = json.loads(json.dumps(artA))
    fields = dict(spec_rope=False, spec_rope_k=8, temporal_rope=False, trope_base=700.0)
    fields.update(over)
    a["protocol"]["verify"]["arm_model"].update(fields); return a
ok_art = art_with(spec_rope=True, temporal_rope=True)
check(trm.calib_arm_model_drift(ok_art, want_both, False) is None, "artifact binding: a both-rotary artifact certifies the combined arm")
for over, tag, wrong in (({"spec_rope": True}, "spectral-only artifact", "'temporal_rope': False"),
                         ({"temporal_rope": True}, "temporal-only artifact", "'spec_rope': False"),
                         ({}, "no-rotary artifact", "'spec_rope': False"),
                         ({"spec_rope": True, "temporal_rope": True, "trope_base": 10000.0}, "both-rotary artifact at another base", "'trope_base': 10000.0"),
                         ({"spec_rope": True, "temporal_rope": True, "spec_rope_k": 4}, "both-rotary artifact at another K", "'spec_rope_k': 4")):
    msg = trm.calib_arm_model_drift(art_with(**over), want_both, False)
    check(isinstance(msg, str) and wrong in msg,
          f"artifact binding: a {tag} is REFUSED for the combined arm, naming what it was measured on ({wrong}); got {msg!r:.120}")
check(trm.calib_arm_model_drift(ok_art, dict(f4, spec_rope=True, spec_rope_k=8, temporal_rope=False, trope_base=700.0), False) is not None,
      "artifact binding: the combined artifact is refused for the spectral-only arm (the binding is symmetric)")

# ---------------- 7. the generation fingerprint ----------------
import importlib.util as ilu
spec = ilu.spec_from_file_location("evalmod", "scripts/_eval_v2_gen_in_evalspace.py")
evm = ilu.module_from_spec(spec)
try:
    spec.loader.exec_module(evm); have_ev = True
except SystemExit:
    have_ev = False
except Exception as e:
    have_ev = False; print("  (eval module import skipped:", type(e).__name__, str(e)[:80], ")")
if have_ev and hasattr(evm, "source_fingerprint"):
    fp = {(s, t): evm.source_fingerprint(True, False, s, t) for s in (False, True) for t in (False, True)}
    check(len(set(fp.values())) == 4, f"the generation fingerprint separates all four arms ({len(set(fp.values()))} distinct)")
else:
    _t = open("scripts/_eval_v2_gen_in_evalspace.py").read()
    check("spec_rope" in _t and "temporal_rope" in _t and "rotary_code_files" in _t,
          "eval consumer: the fingerprint is driven by rotary_code_files over both flags (textual check; module not importable standalone)")

# ---------------- 8. the two configs and the launcher ----------------
# codex trope r1 P1-3: a switch must be PINNED, not inherited -- so source each config with the opposite values
# already exported and read back what actually survives. Arm H inherits its switches from arm G's config on purpose;
# only executing the file can show whether that inheritance is safe.
# every poison value is one the consumer would ACCEPT -- a value argparse rejects would fail loudly anyway, so it
# would not test the pin. AUG_MODE="joint" is arm B's augmentation strength: valid, and the wrong arm.
POISON = dict(SPEC_ROPE="0", SPEC_ROPE_K="4", TEMPORAL_ROPE="0", TROPE_BASE="10000", AUG_MODE="joint",
              AUG_P="0.5", AUG_DROP_MODE="any", AUG_BONE_SCALE="0.2")
def cfg_env(cfg):
    env = dict(os.environ); env.update(POISON)
    out = subprocess.run(["bash", "-c", f'set -a; source "{cfg}" >/dev/null 2>&1; '
                          'for v in SPEC_ROPE SPEC_ROPE_K TEMPORAL_ROPE TROPE_BASE AUG_MODE AUG_P AUG_DROP_MODE AUG_BONE_SCALE GPUS_PER OUT CALIB RDZV_PORT EXTRA; '
                          'do printf "%s=%s\n" "$v" "${!v}"; done'],
                         capture_output=True, text=True, env=env, cwd=".")
    return dict(l.split("=", 1) for l in out.stdout.strip().splitlines() if "=" in l)
for cfg, tag, aug in ((CFG_G, "spectral+temporal", False),
                      (CFG_H, "spectral+temporal+augmentation", True)):
    e = cfg_env(cfg)
    check(e.get("SPEC_ROPE") == "1" and e.get("TEMPORAL_ROPE") == "1",
          f"{tag} config: both rotary switches survive a hostile environment that pre-sets them to 0 "
          f"(SPEC_ROPE={e.get('SPEC_ROPE')}, TEMPORAL_ROPE={e.get('TEMPORAL_ROPE')})")
    check(e.get("SPEC_ROPE_K") == "8" and e.get("TROPE_BASE") == "700",
          f"{tag} config: both rotary parameters survive it too (K={e.get('SPEC_ROPE_K')}, base={e.get('TROPE_BASE')})")
    # what reaches the trainer is EXTRA, not the shell variables: the unaugmented arm must carry no --aug_ flag at
    # all (codex bothrope r1 P2 -- "AUG_MODE is not one_of" would also be true of joint/0.5), the augmented one must
    # carry arm C's four values, all four of them poisoned in this environment
    # tokenised, not substring-matched, and both spellings recognised: codex bothrope r2 slipped AUG_P=0.85 past
    # "--aug_p 0.8" and appended --aug_mode=joint past a regex that demanded a space.
    toks = shlex.split(e.get("EXTRA", ""))
    opts = {}
    i = 0
    while i < len(toks):
        t_ = toks[i]
        if t_.startswith("--"):
            if "=" in t_:
                k_, v_ = t_.split("=", 1); opts[k_] = v_; i += 1
            elif i + 1 < len(toks) and not toks[i + 1].startswith("--"):
                opts[t_] = toks[i + 1]; i += 2
            else:
                opts[t_] = True; i += 1
        else:
            i += 1
    aug_opts = {k_: v_ for k_, v_ in opts.items() if k_.startswith("--aug_")}
    if aug:
        want = {"--aug_mode": "one_of", "--aug_p": "0.8", "--aug_drop_mode": "tips", "--aug_bone_scale": "0.1"}
        bad = {k_: aug_opts.get(k_) for k_, v_ in want.items() if aug_opts.get(k_) != v_}
        check(not bad, f"{tag} config: EXTRA carries arm C's four values EXACTLY despite all four being poisoned "
                       f"(wrong: {bad or 'none'}; {len(aug_opts)} aug options in all)")
    else:
        check(aug_opts == {}, f"{tag} config: EXTRA carries NO augmentation option at all, in either spelling ({aug_opts})")
    check(e.get("CALIB", "").endswith("bothrope_aug_gamma_calibration_b16_v1.json" if aug else "bothrope_gamma_calibration_b16_v1.json"),
          f"{tag} config: points at its own calibration artifact ({e.get('CALIB')})")
# the two calibration runners must refuse to certify a model this arm does not build. Only the preflight is run:
# everything up to the srun that would actually measure (codex bothrope r1 P3 found H accepting a zero switch,
# because "0" is non-empty and only G had the equality checks).
absent = "runs/_heldout/_aug_dev_logs/_no_such_calibration_artifact.json"
assert not os.path.exists(absent), f"{absent} must not exist -- it is the redirect target for the refusal probes"
# the pair (0,0) is its own case: codex bothrope r3 wrapped the two refusals in `if SPEC_ROPE != TEMPORAL_ROPE`,
# which passes every one-at-a-time probe and then accepts a run with NEITHER rotary
for runner, poisons in ((CALIB_G, (("SPEC_ROPE", "0"), ("TEMPORAL_ROPE", "0"), ("SPEC_ROPE=0 TEMPORAL_ROPE", "0"))),
                        (CALIB_H, (("SPEC_ROPE", "0"), ("TEMPORAL_ROPE", "0"),
                                   ("SPEC_ROPE=0 TEMPORAL_ROPE", "0"), ("AUG_MODE", "joint")))):
    body = open(runner).read().split("\nsrun --jobid=", 1)[0]
    for knob, bad in poisons:
        code = body.replace('set -a; . "$CFG"; set +a', f'set -a; . "$CFG"; set +a; export {knob}={bad}')
        assert code != body, f"{runner}: could not place the poison after the source"
        # CALIB_OUT is redirected to a path that does not exist, so the "artifact already there" refusal further
        # down cannot stand in for the one under test; and the message must be the LAST line, so deleting only the
        # refusal's `exit 1` fails here (codex bothrope r2).
        code += f'\n[ "$CALIB_OUT" = {absent} ] || {{ echo "[refuse] CALIB_OUT was not redirected"; exit 1; }}\necho REACHED_END'
        code = code.replace('export CALIB_OUT="$CALIB"', f'export CALIB_OUT={absent}')
        r = subprocess.run(["bash", "-c", code], capture_output=True, text=True, env=dict(PATH="/usr/bin:/bin"))
        out = (r.stdout + r.stderr).strip().splitlines()
        last = out[-1] if out else ""
        # a poison may set more than one switch; the refusal must name at least one of them with its value, so the
        # two runners are free to check their switches in whichever order they like
        pairs = [kv if "=" in kv else f"{kv}={bad}" for kv in knob.split()]
        check(r.returncode != 0 and "[refuse]" in last and any(kv in last for kv in pairs),
              f"{os.path.basename(runner)}: {knob}={bad} is refused, terminally, before anything is measured "
              f"(rc {r.returncode}, last line {last[:70]!r})")
# codex bothrope r16: the suite never read the evaluation launchers at all, so swapping the augmented arm's
# checkpoint for the unaugmented one -- evaluating the wrong model under the augmented arm's label and report name --
# passed every check. Each launcher's queue spec is parsed and its checkpoint required to be ITS OWN arm's output
# directory, which the arm's config names; the label and the report must carry the same arm letter.
EVAL_SH = {"G": os.environ.get("TEST_EVAL_G", "runs/_heldout/eval/_launch_G.sh"),
           "H": os.environ.get("TEST_EVAL_H", "runs/_heldout/eval/_launch_H.sh")}
_ck_seen = {}
for _letter, _cfgp in (("G", CFG_G), ("H", CFG_H)):
    _txt = open(EVAL_SH[_letter]).read()
    _spec = re.search(r'"([^"]*\|[^"]*)"', _txt)
    check(_spec is not None, f"arm {_letter} evaluation launcher: its queue spec was found")
    if not _spec:
        continue
    _tag, _ck, _steps, _rep = (_spec.group(1).split("|") + ["", "", "", ""])[:4]
    _out = cfg_env(_cfgp)["OUT"]
    _ck_seen[_letter] = _ck
    check(_ck == f"{_out}/last_model.pt",
          f"arm {_letter} evaluation launcher: it evaluates ITS OWN arm's checkpoint "
          f"({_ck} vs the config's OUT {_out})")
    _lbl = re.search(r'_geneval_queue\.sh\s+"[^"]*"\s+(\S+)', _txt)
    check(_lbl is not None and _lbl.group(1) == f"heldout{_letter}"
          and _tag.startswith(f"heldout{_letter}_") and f"geneval_{_letter}_" in _rep,
          f"arm {_letter} evaluation launcher: label, shard tag and report all name arm {_letter} "
          f"(label {_lbl.group(1) if _lbl else None}, tag {_tag}, report {os.path.basename(_rep)})")
check(len(set(_ck_seen.values())) == 2,
      f"the two evaluation launchers read DIFFERENT checkpoints ({sorted(set(_ck_seen.values()))})")
# codex bothrope r17: the launcher checks never looked at EVAL_EXTRA, so dropping --strict_acceptance from one of
# them would score that arm under a laxer acceptance rule while its report still ended in _strict.json. The protocol
# arguments must be IDENTICAL to the finished arm whose reports these will be compared against -- arm D's launcher,
# which produced one of those reports -- because the comparison is only meaningful under one protocol.
def _eval_extra(path):
    m = re.search(r'EVAL_EXTRA="([^"]*)"', open(path).read())
    return shlex.split(m.group(1)) if m else None
_ref_extra = _eval_extra("runs/_heldout/eval/_launch_D.sh")
check(_ref_extra and "--strict_acceptance" in _ref_extra,
      f"the reference arm's evaluation protocol arguments were read ({_ref_extra})")
for _letter in ("G", "H"):
    _mine = _eval_extra(EVAL_SH[_letter])
    check(_mine == _ref_extra,
          f"arm {_letter} evaluation launcher: its protocol arguments are IDENTICAL to the finished arm's "
          f"({_mine} vs {_ref_extra})")
check(cfg_env(CFG_G)["RDZV_PORT"] != cfg_env(CFG_H)["RDZV_PORT"],
      "the two combined arms use different rendezvous ports (they can run at the same time)")
# codex bothrope r1 P2: whole-file substring searches survived deleting the very lines they claimed to verify --
# every rotary flag also appears in the duplicate-guard list, so "--temporal_rope is somewhere in the launcher" is
# true even with the emission gone. Each of the three below now reads the ONE construct it is about.
lt = open(LAUNCH_SH).read()
pre = lt.split('\nmkdir -p "$OUT"', 1)[0]
check(pre != lt and "for dup in" in pre, "launcher: the duplicate guard sits in the prefix that runs before OUT is touched")
# (a) the EFFECTIVE rotary settings of the run this launcher would start. Expanding the command was still not
# enough: codex bothrope r3 appended a second --trope_base (argparse takes the last) and commented the substitution
# out inside the quoted payload (shlex still saw the flags, the inner shell did not emit them). So the inner shell is
# EXECUTED with torchrun stubbed, the argv it really receives is captured, and that argv is parsed by the trainer's
# own parser -- obtained from the trainer itself, not re-declared here.
_orig_parse = argparse.ArgumentParser.parse_args
class _GrabbedParser(Exception):
    def __init__(self, ap): self.ap = ap
def _grab(self, args=None, namespace=None): raise _GrabbedParser(self)
argparse.ArgumentParser.parse_args = _grab
real_ap = None
try:
    trm.main()
except _GrabbedParser as g:
    real_ap = g.ap
except BaseException as _e:
    print("  (could not grab the trainer's parser:", type(_e).__name__, str(_e)[:80], ")")
finally:
    argparse.ArgumentParser.parse_args = _orig_parse
check(real_ap is not None and any(a.dest == "trope_base" for a in real_ap._actions),
      "the trainer's OWN argument parser was obtained (no re-declaration of the rotary options here)")
STUB = os.path.abspath(os.path.join(os.environ.get("SCRATCH_TMP", "runs/_heldout/_aug_dev_logs"), "_stub_bin"))
os.makedirs(STUB, exist_ok=True)
with open(os.path.join(STUB, "torchrun"), "w") as fh:
    fh.write('#!/bin/bash\nprintf "%s\\n" "$@"\n')
os.chmod(os.path.join(STUB, "torchrun"), 0o755)
fn = re.search(r"^run_rank\(\) \{.*?^\}", lt, re.S | re.M)
nccl = re.search(r'^NCCL_ENV="[^"]*"\n(?:\[ "\$SMOKE" = 1 \].*\n)?', lt, re.M)
check(fn is not None and nccl is not None, "launcher: the run_rank definition and the NCCL environment were found")
SRC = 'set -a; . "$CFG"; set +a'
check(pre.count(SRC) == 1, f"launcher: its own config source appears once in the prefix ({pre.count(SRC)})")
def prefix_with(cfg, inject=""):
    body = pre.replace(SRC, SRC + ("\n" + inject if inject else ""), 1)
    return f'CFG={cfg}\nRDZV_HOST=${{RDZV_HOST:-127.0.0.1}}\n' + body
def effective(cfg, **over):
    # srun's stub RUNS the payload it was given, exactly as srun does, so anything the inner shell decides -- a
    # comment, a second value, a nested test -- shows up in what torchrun actually receives
    # the launcher's OWN pre-OUT prefix runs first: codex bothrope r4 changed TROPE_BASE=${TROPE_BASE:-700} to 1300
    # there, and a probe that jumped straight from the config to run_rank still captured 700
    sh = ('srun() { while [ $# -gt 0 ] && [ "$1" != bash ]; do shift; done; "$@"; }\n'
          + prefix_with(cfg, "".join(f'export {k}={v}\n' for k, v in over.items())) + "\n"
          + (nccl.group(0) if nccl else "") + "\n"   # defined between the prefix and run_rank; read from the bytes
          + 'RES_ARG=""; SMOKE_ARGS=""\n'            # the only stubs: resume / smoke argv, no rotary content
          + (fn.group(0) if fn else "") + "\nrun_rank 1 0\n")
    r = subprocess.run(["bash", "-c", sh], capture_output=True, text=True,
                       env=dict(PATH=STUB + ":/usr/bin:/bin"))
    toks = [re.sub(r"^\[r\d+\] ", "", l) for l in r.stdout.splitlines()]
    if "scripts/train_v2_incontext.py" not in toks:
        return None, toks
    argv = toks[toks.index("scripts/train_v2_incontext.py") + 1:]
    try:
        ns = real_ap.parse_args(argv)     # STRICT: codex bothrope r4 mutated --trope_base to --trope_base_typo and
    except SystemExit:                    # parse_known_args silently dropped it while the trainer would exit 2
        return None, argv
    return ns, argv
ns_on, argv_on = effective(CFG_G)
check(ns_on is not None and bool(ns_on.spec_rope) and int(ns_on.spec_rope_k) == 8,
      f"the run arm G's config would start PARSES to spec_rope=True, K=8 "
      f"(got {getattr(ns_on, 'spec_rope', None)}/{getattr(ns_on, 'spec_rope_k', None)}; {len(argv_on)} argv tokens)")
check(ns_on is not None and bool(ns_on.temporal_rope) and float(ns_on.trope_base) == 700.0,
      f"...and to temporal_rope=True, base=700.0 "
      f"(got {getattr(ns_on, 'temporal_rope', None)}/{getattr(ns_on, 'trope_base', None)})")
ns_off, _ = effective(CFG_G, SPEC_ROPE="0", TEMPORAL_ROPE="0")
check(ns_off is not None and not ns_off.spec_rope and not ns_off.temporal_rope,
      f"with both switches 0 the same command parses to neither rotary "
      f"(got {getattr(ns_off, 'spec_rope', None)}/{getattr(ns_off, 'temporal_rope', None)})")
ns_h, _ = effective(CFG_H)
check(ns_h is not None and bool(ns_h.spec_rope) and int(ns_h.spec_rope_k) == 8
      and bool(ns_h.temporal_rope) and float(ns_h.trope_base) == 700.0
      and ns_h.aug_mode == "one_of" and abs(float(ns_h.aug_p) - 0.8) < 1e-9,
      f"the run arm H's config would start parses to both rotaries WITH their parameters AND one_of at p 0.8 "
      f"(got {getattr(ns_h, 'spec_rope', None)}/{getattr(ns_h, 'spec_rope_k', None)}/"
      f"{getattr(ns_h, 'temporal_rope', None)}/{getattr(ns_h, 'trope_base', None)}/"
      f"{getattr(ns_h, 'aug_mode', None)}/{getattr(ns_h, 'aug_p', None)})")
# (a2) the model the TRAINER ITSELF would build from those parsed arguments. codex bothrope r6: a one-line
# mutation of the trainer's constructor call, `use_temporal_rope=(a.temporal_rope and not a.spec_rope)`, breaks
# EXACTLY the combined arms and leaves both single-rotary arms intact -- and every check above still passed, because
# the parser probe stops at the arguments and the model checks build their own models. So the trainer's own
# construction block is extracted from its source and executed with the parsed namespace.
_tsrc = open(TRAIN_PY).read()
_tlines = _tsrc.splitlines(keepends=True)
def trainer_stmts(*wanted):
    """The trainer's own assignment statements, by target name, in source order."""
    out, seen = [], set()
    for _n in ast.walk(ast.parse(_tsrc)):
        if isinstance(_n, (ast.Assign, ast.ImportFrom)) and getattr(_n, "lineno", None):
            tgt = None
            if isinstance(_n, ast.Assign) and len(_n.targets) == 1 and isinstance(_n.targets[0], ast.Name):
                tgt = _n.targets[0].id
            if tgt in wanted and tgt not in seen:
                seen.add(tgt); out.append((_n.lineno, "".join(_tlines[_n.lineno - 1:_n.end_lineno])))
    return [t for _, t in sorted(out)], seen
_blk = None
for _n in ast.walk(ast.parse(_tsrc)):
    if isinstance(_n, ast.If):
        # by line range, so the FIRST line keeps its indentation and dedent sees one uniform prefix
        _seg = "".join(_tlines[_n.lineno - 1:_n.end_lineno])
        if "InContextMotionDiT(" in _seg and "TwoStageInContextDiT(" in _seg:
            _blk = textwrap.dedent(_seg); break
check(_blk is not None, "the trainer's own model-construction block was found in its source")
_inch = re.search(r"^\s*in_ch = (.+)$", _tsrc, re.M)
check(_inch is not None, f"the trainer's own in_ch rule was found ({_inch.group(1) if _inch else None})")
def trainer_builds(ns_args):
    g = dict(trm.__dict__)
    g["a"], g["dev"] = ns_args, "cpu"
    exec(compile(_inch.group(0).strip(), "<in_ch>", "exec"), g)
    exec(compile(_blk, TRAIN_PY, "exec"), g)
    return g["model"]
for _ns, _tag in ((ns_on, "spectral+temporal"), (ns_h, "spectral+temporal+augmentation")):
    if _ns is None:
        check(False, f"{_tag}: no parsed arguments to build from"); continue
    _m = trainer_builds(_ns)
    _sd = _m.state_dict()
    check(bool(getattr(_m, "use_spec_rope", False)) and int(getattr(_m, "spec_rope_k", 0)) == 8
          and bool(getattr(_m, "use_temporal_rope", False)) and float(getattr(_m, "trope_base_value", 0)) == 700.0,
          f"{_tag}: THE TRAINER'S OWN constructor builds a model with both rotaries live "
          f"(spec {getattr(_m, 'use_spec_rope', None)}/K{getattr(_m, 'spec_rope_k', None)}, "
          f"temporal {getattr(_m, 'use_temporal_rope', None)}/base{getattr(_m, 'trope_base_value', None)})")
    check("j_pos" not in _sd and "t_pos" not in _sd and "trope_base" in _sd
          and any(k.startswith("spec_rope.") for k in _sd),
          f"{_tag}: ...and that model carries NEITHER learned position table, plus the SignNet and the base buffer "
          f"(j_pos {'j_pos' in _sd}, t_pos {'t_pos' in _sd}, trope_base {'trope_base' in _sd})")

# (a3) the OPTIMIZER the trainer would hand that model to. codex bothrope r8: excluding the SignNet from the
# trainable list only when both rotaries are on freezes all 39,200 of its parameters while the backbone trains, and
# passed all 83 checks -- the gradient assertions above prove the SignNet RECEIVES gradient, never that the optimizer
# owns it or steps it. So the trainer's own `trainable = ...` and `opt = ...` statements are executed too.
# the schedule block: the `if a.warmup_steps > 0 ...` chain through the `for pg in opt.param_groups` write
_lrblk = None
_tt = ast.parse(_tsrc)
for _n in ast.walk(_tt):
    if isinstance(_n, ast.If) and "a.warmup_steps > 0 and gstep < a.warmup_steps" in \
            "".join(_tlines[_n.lineno - 1:_n.lineno]):
        _endfor = None
        for _m in ast.walk(_tt):
            if isinstance(_m, ast.For) and getattr(_m.target, "id", None) == "pg" \
                    and _m.lineno > _n.end_lineno and _m.lineno - _n.end_lineno < 12:
                _endfor = _m; break
        if _endfor is not None:
            _lrblk = textwrap.dedent("".join(_tlines[_n.lineno - 1:_endfor.end_lineno])); break
check(_lrblk is not None and "pg[\"lr\"]" in (_lrblk or ""),
      "the trainer's own lr-schedule block, through the write into the optimizer's groups, was found")
_stmts, _seen = trainer_stmts("aug_cfg", "base", "names", "types", "ds_tr")
check(_seen == {"aug_cfg", "base", "names", "types", "ds_tr"},
      f"the trainer's own corpus / augmentation / dataset statements were all found ({sorted(_seen)})")
def trainer_dataset(ns_args):
    g = dict(trm.__dict__); g["a"] = ns_args
    from src.data.ktjd17_incontext import Ktjd17Base, ktjd17_split_names
    g["Ktjd17Base"], g["ktjd17_split_names"] = Ktjd17Base, ktjd17_split_names
    for st in _stmts:
        exec(compile(textwrap.dedent(st), TRAIN_PY, "exec"), g)
    return g
_ns_g, _ns_h = trainer_dataset(ns_on), trainer_dataset(ns_h)
ds_g, ds_aug = _ns_g["ds_tr"], _ns_h["ds_tr"]
# the loader block, taken CONTIGUOUSLY from the generator that seeds it through the DataLoader itself, so the
# statement runs exactly as the trainer runs it (the loader references a generator built two lines above it)
_dlblk = None
_tt3 = ast.parse(_tsrc)
_gen_ln = next((n.lineno for n in ast.walk(_tt3) if isinstance(n, ast.Assign) and len(n.targets) == 1
                and getattr(n.targets[0], "id", None) == "dl_gen"), None)
_dl_end = next((n.end_lineno for n in ast.walk(_tt3) if isinstance(n, ast.Assign) and len(n.targets) == 1
                and getattr(n.targets[0], "id", None) == "dl_tr"), None)
if _gen_ln and _dl_end and _dl_end > _gen_ln:
    _dlblk = textwrap.dedent("".join(_tlines[_gen_ln - 1:_dl_end]))
check(_dlblk is not None and "DataLoader(" in (_dlblk or "") and "manual_seed" in (_dlblk or ""),
      "the trainer's own training-loader block was found, generator included")
_hstmts, _hseen = trainer_stmts("_decay_ep", "total_opt_steps")
check(_hseen == {"_decay_ep", "total_opt_steps"},
      f"the trainer's own decay-horizon statements were found ({sorted(_hseen)})")
_ostmts, _oseen = trainer_stmts("trainable", "opt")
check(_oseen == {"trainable", "opt"},
      f"the trainer's own trainable-parameter and optimizer statements were found ({sorted(_oseen)})")
def trainer_optimizes(ns_args, built):
    g = dict(trm.__dict__); g["a"], g["model"], g["raw_model"] = ns_args, built, built
    for st in _ostmts:
        exec(compile(textwrap.dedent(st), TRAIN_PY, "exec"), g)
    return g["opt"]
for _ns, _tag, _cfg2 in ((ns_on, "spectral+temporal", CFG_G), (ns_h, "spectral+temporal+augmentation", CFG_H)):
    if _ns is None:
        check(False, f"{_tag}: no parsed arguments to optimise from"); continue
    _m = trainer_builds(_ns)
    _opt = trainer_optimizes(_ns, _m)
    _owned = {id(q) for grp in _opt.param_groups for q in grp["params"]}
    _sig = [(n_, q) for n_, q in _m.named_parameters() if n_.startswith("spec_rope.")]
    check(_sig and all(id(q) in _owned for _, q in _sig),
          f"{_tag}: THE TRAINER'S OWN optimizer owns every SignNet parameter "
          f"({sum(id(q) in _owned for _, q in _sig)}/{len(_sig)} tensors, {sum(q.numel() for _, q in _sig):,} values)")
    # and a real step actually moves them: ownership without an update is the same failure
    _before = {n_: q.detach().clone() for n_, q in _sig}
    _m.train(); _m.grad_ckpt = False
    _dz = make_inputs(11, ASYM)
    _opt.zero_grad(); run(_m, _dz).square().mean().backward(); _opt.step()
    _moved = [n_ for n_, q in _sig if not torch.equal(q.detach(), _before[n_])]
    _m.eval()
    check(len(_moved) == len(_sig),
          f"{_tag}: ...and ONE optimizer step moves every one of them ({len(_moved)}/{len(_sig)})")
    # codex bothrope r10: constructing AdamW and stepping it says nothing about the SCHEDULE. A mutation writing
    # a.lr instead of lr_now for the combined arms starts them at 2e-4 instead of 5e-8 and never decays. The
    # trainer's own schedule block is executed at both endpoints and the optimizer's lr is read back.
    # codex bothrope r11: supplying my own total_opt_steps bypassed the horizon calculation, and a mutant using
    # a.epochs instead of a.lr_decay_epochs (120 instead of 40) tripled the decay length -- at the real horizon the
    # lr would be 1.6e-4 instead of 2e-6 -- while both endpoint assertions still passed. The horizon now comes from
    # the trainer's own statements, and is pinned to lr_decay_epochs by how it responds to each knob.
    # codex bothrope r13: a stubbed loader length cannot see a mutation of the loader itself -- doubling the
    # training batch size for the combined arms halves the steps per epoch (218 instead of 437) and the horizon
    # with it, while a.batch stays 16. The trainer's OWN dl_tr statement is executed over its OWN ds_tr, with a
    # sampler at this arm's launch geometry, and its length feeds the horizon.
    _world = int(cfg_env(_cfg2)["GPUS_PER"]) * 2          # two nodes, the geometry the config launches
    _ds_here = (_ns_g if _ns is ns_on else _ns_h)["ds_tr"]
    _dlns = dict(trm.__dict__)
    _a_dl = argparse.Namespace(**{**vars(_ns), "num_workers": 0})   # len() never iterates; keeps workers unspawned
    _dlns.update(a=_a_dl, ds_tr=_ds_here,
                 tr_sampler=DistributedSampler(_ds_here, num_replicas=_world, rank=0, shuffle=True, drop_last=True),
                 ddp=True)
    exec(compile(_dlblk, TRAIN_PY, "exec"), _dlns)
    _dl_real = _dlns["dl_tr"]
    check(int(_dl_real.batch_size) == int(_ns.batch),
          f"{_tag}: the trainer's own training loader carries THIS arm's batch size "
          f"({_dl_real.batch_size} vs --batch {_ns.batch})")
    def _horizon(**over):
        _g3 = dict(trm.__dict__)
        _g3.update(a=argparse.Namespace(**{**vars(_ns), **over}), dl_tr=_dl_real)
        for _st in _hstmts:
            exec(compile(textwrap.dedent(_st), TRAIN_PY, "exec"), _g3)
        return _g3["total_opt_steps"]
    _h0 = _horizon()
    # codex bothrope r12: the two knob invariants below do not constrain the ABSOLUTE scale -- a mutant that simply
    # doubles the horizon satisfies both. The absolute value is pinned against an EXTERNAL reference instead: the
    # decay length a finished arm of the same geometry printed into its own log. Arms A and E both run 40 decay
    # epochs over 120 at 8 ranks with grad_accum 1, which is exactly this arm's geometry.
    _ref_log = "runs/v2_noik_pilot36m_heldout_rest/orch_rank0.log"
    _ref = re.search(r"-> (\d+) decay steps \((\d+) ep of (\d+)\)", open(_ref_log).read()) if os.path.exists(_ref_log) else None
    check(_ref is not None, f"the reference arm's own printed decay length was found in {_ref_log}")
    if _ref:
        _same_recipe = (int(_ref.group(2)) == int(_ns.lr_decay_epochs) and int(_ref.group(3)) == int(_ns.epochs))
        check(_same_recipe and _h0 == int(_ref.group(1)),
              f"{_tag}: the decay horizon EQUALS the one the baseline arm printed for the same recipe "
              f"({_h0} vs {_ref.group(1)}; {_ref.group(2)} ep of {_ref.group(3)} there, "
              f"{_ns.lr_decay_epochs} of {_ns.epochs} here)")
    check(_horizon(lr_decay_epochs=_ns.lr_decay_epochs * 2) == _h0 * 2,
          f"{_tag}: the decay horizon scales with --lr_decay_epochs ({_h0} -> {_horizon(lr_decay_epochs=_ns.lr_decay_epochs * 2)})")
    check(_horizon(epochs=_ns.epochs * 3) == _h0,
          f"{_tag}: ...and does NOT move with --epochs ({_h0} -> {_horizon(epochs=_ns.epochs * 3)}; "
          f"lr_decay_epochs={_ns.lr_decay_epochs} of {_ns.epochs} epochs)")
    _lr_at = {}
    for _gs, _lbl in ((0, "first step"), (_h0, "at the decay horizon")):
        _g2 = dict(trm.__dict__)
        _g2.update(a=_ns, opt=_opt, gstep=_gs, total_opt_steps=_h0)
        exec(compile(_lrblk, TRAIN_PY, "exec"), _g2)
        _lr_at[_lbl] = _opt.param_groups[0]["lr"]
    _want_first = _ns.lr / _ns.warmup_steps
    _want_floor = _ns.lr * _ns.eta_min_ratio
    check(abs(_lr_at["first step"] - _want_first) < 1e-15 and abs(_lr_at["at the decay horizon"] - _want_floor) < 1e-15,
          f"{_tag}: THE TRAINER'S OWN SCHEDULE reaches the optimizer, at the first step and at ITS OWN horizon "
          f"(first {_lr_at['first step']:.3g} vs {_want_first:.3g}, at step {_h0} "
          f"{_lr_at['at the decay horizon']:.3g} vs floor {_want_floor:.3g})")

# (a4) the last handoff: what the checkpoint records, and what the EVALUATOR rebuilds from it. A checkpoint carries
# vars(a) verbatim, so the parsed namespace asserted above IS what the evaluator sees -- assert that verbatim-ness,
# then build through the evaluator's own constructor call with exactly those args.
# codex bothrope r9: checking the "args" entries that exist misses a payload that DROPS the key, so the payloads
# are identified first (a checkpoint dict is the one carrying gstep AND best_val) and args is then REQUIRED in each
_bad_args, _n_payload = [], 0
for _n in ast.walk(ast.parse(_tsrc)):
    if isinstance(_n, ast.Dict):
        _keys = {k.value for k in _n.keys if isinstance(k, ast.Constant)}
        if not {"gstep", "best_val"} <= _keys:
            continue
        _n_payload += 1
        _v = next((v for k, v in zip(_n.keys, _n.values)
                   if isinstance(k, ast.Constant) and k.value == "args"), None)
        ok = (_v is not None and isinstance(_v, ast.Call) and isinstance(_v.func, ast.Name) and _v.func.id == "vars"
              and len(_v.args) == 1 and isinstance(_v.args[0], ast.Name) and _v.args[0].id == "a")
        if not ok: _bad_args.append(_n.lineno)
check(_n_payload >= 4 and not _bad_args,
      f"all {_n_payload} checkpoint payloads in the trainer REQUIRE the parsed namespace verbatim as vars(a) "
      f"(offenders at lines {_bad_args})")
import importlib.util as _ilu
_espec = _ilu.spec_from_file_location("_evm_under_test", EVAL_PY)
_evm = _ilu.module_from_spec(_espec); sys.modules["_evm_under_test"] = _evm
try:
    _espec.loader.exec_module(_evm); _ev_ok = True
except BaseException as _e:
    _ev_ok = False; print("  (evaluator module did not import:", type(_e).__name__, str(_e)[:90], ")")
check(_ev_ok and callable(getattr(_evm, "load_gen_model", None)), "the evaluator module imports and exposes its loader")
_esrc = open(EVAL_PY).read(); _elines = _esrc.splitlines(keepends=True)
_ebuild = None
for _n in ast.walk(ast.parse(_esrc)):
    if isinstance(_n, ast.Assign) and len(_n.targets) == 1 and getattr(_n.targets[0], "id", None) == "model":
        _seg = "".join(_elines[_n.lineno - 1:_n.end_lineno])
        if "InContextMotionDiT(" in _seg and "use_temporal_rope=" in _seg:
            _ebuild = textwrap.dedent(_seg); break
check(_ebuild is not None, "the evaluator's own model-rebuild statement was found")
for _ns, _tag in ((ns_on, "spectral+temporal"), (ns_h, "spectral+temporal+augmentation")):
    if _ns is None or _ebuild is None:
        check(False, f"{_tag}: nothing to rebuild from"); continue
    _g = {"ca": vars(_ns), "dev": "cpu", "InContextMotionDiT": InContextMotionDiT}
    exec(compile(_ebuild, EVAL_PY, "exec"), _g)
    _em = _g["model"]; _esd = _em.state_dict()
    check(bool(_em.use_spec_rope) and int(_em.spec_rope_k) == 8 and bool(_em.use_temporal_rope)
          and float(_em.trope_base_value) == 700.0 and "j_pos" not in _esd and "t_pos" not in _esd,
          f"{_tag}: THE EVALUATOR'S OWN rebuild from this arm's recorded args restores both rotaries and neither "
          f"learned table (spec {_em.use_spec_rope}/K{_em.spec_rope_k}, temporal {_em.use_temporal_rope}/"
          f"base{_em.trope_base_value}, j_pos {'j_pos' in _esd}, t_pos {'t_pos' in _esd})")
    # codex bothrope r9: matching KEYS is not loading. The evaluator's real loader is called with a real payload --
    # the trainer-built model's own state_dict plus vars(a) -- so a shape change (its d_joint_sem 4096 -> 4095)
    # fails here the way it would fail on the real checkpoint.
    _tm = trainer_builds(_ns)
    _ckpt = {"model": _tm.state_dict(), "args": vars(_ns)}
    try:
        _lm, _lca = _evm.load_gen_model(_ckpt, "cpu"); _err = None
    except BaseException as _e:
        _lm, _err = None, f"{type(_e).__name__}: {str(_e)[:110]}"
    # codex bothrope r10: a mutant that SKIPS load_state_dict satisfies every attribute assertion while 112 of 166
    # tensors differ from the payload. Compare the tensors.
    _lsd = _lm.state_dict() if _lm is not None else {}
    _diff = [k for k, v in _ckpt["model"].items() if k not in _lsd or not torch.equal(_lsd[k], v)]
    check(_lm is not None and not _diff,
          f"{_tag}: ...and every one of the {len(_ckpt['model'])} checkpoint tensors is actually IN the returned "
          f"model ({len(_diff)} differ)")
    check(_lm is not None and bool(_lm.use_spec_rope) and bool(_lm.use_temporal_rope)
          and float(_lm.trope_base_value) == 700.0 and int(_lm.spec_rope_k) == 8,
          f"{_tag}: THE EVALUATOR'S REAL LOADER strictly loads the trainer's own payload and restores both rotaries "
          f"({_err or 'spec ' + str(_lm.use_spec_rope) + '/K' + str(_lm.spec_rope_k) + ', temporal ' + str(_lm.use_temporal_rope) + '/base' + str(_lm.trope_base_value)})")

# (a5) the TRAINING-LOSS handoff. codex bothrope r9: replacing the gammas with all-ones only when both rotaries are
# on changes the cropped diagnostic loss from 5.22 to 1.03 for one arm and 5.35 to 1.15 for the other, while their
# metadata still records calibrated weights -- and every check passed, because the optimizer step above uses the
# test's own squared-output loss. The trainer's own gamma assignment and its own ktjd_prep call are executed here,
# with ktjd_prep replaced by a capture, and the weights that actually arrive are compared with the artifact.
_gstmts, _gseen = trainer_stmts("ktjd_gammas")
check(_gseen == {"ktjd_gammas"}, f"the trainer's own gamma assignment was found ({sorted(_gseen)})")
# there are TWO ktjd_prep call sites -- the connectivity diagnostic and the training step -- and ast.walk reaches
# the diagnostic first, so the TRAINING one is selected by its assignment targets (x_in, kt_kw)
_call, _n_sites = None, 0
for _n in ast.walk(ast.parse(_tsrc)):
    if isinstance(_n, ast.Assign) and isinstance(_n.value, ast.Call) \
            and getattr(_n.value.func, "id", None) == "ktjd_prep":
        _n_sites += 1
        _tg = _n.targets[0]
        _names = [e.id for e in _tg.elts] if isinstance(_tg, ast.Tuple) else []
        if _names == ["x_in", "kt_kw"]:
            _call = textwrap.dedent("".join(_tlines[_n.lineno - 1:_n.end_lineno]))
check(_call is not None, f"the trainer's TRAINING-step ktjd_prep call site was found ({_n_sites} call sites in all)")
for _ns, _tag, _cfg in ((ns_on, "spectral+temporal", CFG_G), (ns_h, "spectral+temporal+augmentation", CFG_H)):
    if _ns is None or _call is None:
        check(False, f"{_tag}: nothing to weight"); continue
    _artj = json.load(open(cfg_env(_cfg)["CALIB"]))
    _seen_g = {}
    def _capture(b, lut, gammas, *rest, **kw):
        _seen_g.clear(); _seen_g.update(gammas); return None, {}
    _g = dict(trm.__dict__)
    _g.update(a=_ns, calib=_artj, ktjd_prep=_capture, b=None, ktjd_lut=object())
    for _st in _gstmts:
        exec(compile(textwrap.dedent(_st), TRAIN_PY, "exec"), _g)
    check(_g["ktjd_gammas"] == {k: float(v) for k, v in _artj["gammas"].items()},
          f"{_tag}: the trainer's own gamma assignment reproduces the artifact's weights exactly")
    exec(compile(_call, TRAIN_PY, "exec"), _g)
    _want_g = {k: float(v) for k, v in _artj["gammas"].items()}
    check(_seen_g == _want_g,
          f"{_tag}: THE WEIGHTS THAT REACH THE LOSS are this arm's calibrated ones, not uniform "
          f"(root_pos {_seen_g.get('root_pos')} vs {_want_g.get('root_pos')}, "
          f"{sum(1 for k in _want_g if _seen_g.get(k) != _want_g[k])} of {len(_want_g)} groups differ)")

# (b) the EXTRA duplicate guard must REFUSE, not merely list the flag: codex bothrope r2 deleted its `exit 1` and
# the membership check still passed. Only the launcher's prefix is run -- everything before it creates OUT or takes
# the lock -- and the refusal has to be the LAST thing printed, so a later unrelated failure cannot stand in for it.
def guard_says(extra):
    # the launcher demands its variables with ${VAR:?...}, so the arm's own config is sourced first, exactly as
    # runs/_heldout/_resume.sh does; EXTRA is then replaced with the probe
    # appended to the arm's OWN EXTRA, not replacing it: codex bothrope r4 deleted the leading * from the equals
    # pattern, which still caught a duplicate in first position and missed one after the real flags
    sh = prefix_with(CFG_G, f'EXTRA="$EXTRA {extra}"')
    r = subprocess.run(["bash", "-c", sh], capture_output=True, text=True, env=dict(PATH="/usr/bin:/bin"))
    out = (r.stdout + r.stderr).strip().splitlines()
    return r.returncode, (out[-1] if out else "")
rc0, last0 = guard_says("")     # the arm's own EXTRA, untouched
check(rc0 == 0, f"launcher: the prefix runs clean with the arm's own EXTRA (rc {rc0}: {last0[:70]}) -- so a refusal "
                f"below is the guard and not some unrelated preflight")
# both spellings: codex bothrope r3 deleted only the `|*" $dup="*` branch and every space-spelled probe still passed
for f, val in (("--spec_rope", ""), ("--spec_rope_k", " 8"), ("--temporal_rope", ""), ("--trope_base", " 700"),
               ("--spec_rope_k", "=4"), ("--trope_base", "=1300")):
    rc, last = guard_says(f"{f}{val}")
    check(rc != 0 and f"EXTRA must not set {f}" in last,
          f"launcher REFUSES {f!r}{val!r} in EXTRA, and that refusal is the last thing it says (rc {rc}: {last[:60]})")
# (c) the trainer's resume-critical tuple, read as the literal it is
tree = ast.parse(open(TRAIN_PY).read())
crit = [ [e.value for e in n.elts] for n in ast.walk(tree)
         if isinstance(n, ast.Tuple) and n.elts and all(isinstance(e, ast.Constant) and isinstance(e.value, str) for e in n.elts)
         and {"huber_delta", "data_root", "epochs"} <= {e.value for e in n.elts} ]
check(len(crit) == 1, f"found exactly one resume-critical tuple in the trainer ({len(crit)})")
for f in ("spec_rope", "spec_rope_k", "temporal_rope", "trope_base"):
    # membership only. That the trainer then COMPARES that tuple on resume is arm-independent behaviour every arm
    # has exercised; this file covers what the combination can break (codex bothrope r2).
    check(bool(crit) and f in crit[0], f"{f} is a member of the trainer's resume-critical tuple ({len(crit[0]) if crit else 0} keys)")


# ---------------- 9. arm H only: the augmentation EDITS the tree, so the eigenbasis must be recomputed ----------------
# The spectral coordinates are a GLOBAL function of the tree. arm C's augmentation adds, removes and pools joints, so a
# cached per-rig basis would silently describe a tree the model is not being shown. _test_spec_rope.py already checks
# that an augmented item's coordinates match its own served tree -- but that check passes vacuously if the
# augmentation never actually bit. This one fails unless the tree really changed AND the coordinates followed it.
# codex bothrope r7: section 9 used to build its OWN augmented dataset, so a trainer mutation that passes
# augment=None only when BOTH rotaries are on survived every check -- the combined augmented arm would have trained
# unaugmented under its augmented calibration. Everything below now comes from the TRAINER'S OWN source: its corpus
# construction, its AugConfig, its `types`, and its ds_tr assignment, executed with each arm's parsed arguments.
check(getattr(ds_g, "aug", "missing") is None or not getattr(ds_g.aug, "active", False),
      f"the unaugmented combined arm: the TRAINER'S OWN ds_tr carries no active augmentation ({ds_g.aug!r:.60})")
_art = json.load(open(cfg_env(CFG_H)["CALIB"]))
_want_proto = _art.get("protocol", {}).get("augmentation")
check(ds_aug.aug is not None and ds_aug.aug.active
      and ds_aug.aug.protocol() == _want_proto,
      f"the augmented combined arm: the TRAINER'S OWN ds_tr is augmented with exactly the protocol its calibration "
      f"artifact certifies ({ds_aug.aug.protocol() if ds_aug.aug else None!r} vs {_want_proto!r})")
def base_parents(ds, ot):
    t_item, _x, J, _T = ds._raw(ds.by_type[ot]["demos"][0])
    return np.asarray(t_item["parent_indices"][:J], dtype=np.int64)

N = 40
own_ok, n_edited, n_followed, n_intact, n_seen = True, 0, 0, 0, 0
for _ in range(N):
    a_ = ds_aug[0]
    pa, ot = a_["parents"].numpy(), a_["object_type"]
    own_ok &= torch.equal(a_["spectral_feats"], torch.from_numpy(laplacian_eigenvectors(pa, 8)[0]))
    n_seen += 1
    pb = base_parents(ds_aug, ot)
    base_sf = torch.from_numpy(laplacian_eigenvectors(pb, 8)[0])
    if len(pa) != len(pb) or not np.array_equal(pa, pb):
        n_edited += 1
        n_followed += int(a_["spectral_feats"].shape != base_sf.shape or not torch.equal(a_["spectral_feats"], base_sf))
    else:
        n_intact += int(a_["spectral_feats"].shape == base_sf.shape and torch.equal(a_["spectral_feats"], base_sf))
check(own_ok, f"augmented arm loader: every item's spectral_feats == laplacian_eigenvectors(its OWN served tree) bitwise ({n_seen} draws)")
check(n_edited >= max(3, N // 4),
      f"augmented arm loader: the augmentation really edits the served tree on {n_edited}/{n_seen} draws (compared "
      f"against each rig's own base tree from the corpus) -- without this the check above would pass vacuously")
check(n_followed == n_edited,
      f"augmented arm loader: EVERY edited tree got coordinates different from its rig's base ones ({n_followed}/{n_edited})")
# codex bothrope r10: capturing ktjd_prep's ARGUMENT is not capturing what reaches the loss -- a mutation that
# rewrites kt_kw["gammas"] during the expansion into cfm_loss leaves the prep call and the metadata untouched and
# makes all nine weights 1.0. The real preparation now runs on a real batch and cfm_loss itself is the capture.
from src.data.incontext_pairs import collate
_lut_st, _lut_seen = trainer_stmts("ktjd_lut")
check(_lut_seen == {"ktjd_lut"}, f"the trainer's own channel-LUT statement was found ({sorted(_lut_seen)})")
# codex bothrope r12: the conditioning dropout was never exercised -- a mutant forcing the text-drop probability to
# 1.0 only for the combined arms would train them with no text conditioning at all, while the configured 0.1 stayed
# in their args and artifacts. The trainer's own apply_cfg_drops is extracted and its observed drop rate compared
# with the probabilities this arm configures.
_dropfn = None
for _n in ast.walk(ast.parse(_tsrc)):
    if isinstance(_n, ast.FunctionDef) and _n.name == "apply_cfg_drops":
        _dropfn = textwrap.dedent("".join(_tlines[_n.lineno - 1:_n.end_lineno])); break
check(_dropfn is not None, "the trainer's own conditioning-dropout function was found")
# the trainer's own auxiliary-weight construction: `fk_kw = {}` through the gamma_fk ramp
_fkblk = None
_tt2 = ast.parse(_tsrc)
for _n in ast.walk(_tt2):
    if isinstance(_n, ast.Assign) and len(_n.targets) == 1 and getattr(_n.targets[0], "id", None) == "fk_kw" \
            and isinstance(_n.value, ast.Dict) and not _n.value.keys:
        _end = None
        for _m in ast.walk(_tt2):
            if isinstance(_m, ast.If) and _m.lineno > _n.lineno and _m.lineno - _n.lineno < 14 \
                    and "gamma_fk=a.gamma_fk * ramp" in "".join(_tlines[_m.lineno - 1:_m.end_lineno]):
                _end = _m; break
        if _end is not None:
            _fkblk = textwrap.dedent("".join(_tlines[_n.lineno - 1:_end.end_lineno])); break
check(_fkblk is not None and "gamma_acc" in (_fkblk or "") and "gamma_lock" in (_fkblk or ""),
      "the trainer's own auxiliary-weight block (fk_kw) was found, covering the vel / lock / acc / fk weights")
_lcall = None
for _n in ast.walk(ast.parse(_tsrc)):
    if isinstance(_n, ast.Assign) and len(_n.targets) == 1 and getattr(_n.targets[0], "id", None) == "loss" \
            and isinstance(_n.value, ast.Call) and getattr(_n.value.func, "id", None) == "cfm_loss":
        _seg = "".join(_tlines[_n.lineno - 1:_n.end_lineno])
        # selected on what a gamma mutation does NOT touch: the fk expansion and the sampler argument. Keying on
        # the literal "**kt_kw" let codex bothrope r10's mutant be caught only as "call not found" (it rewrites
        # exactly that token), which detects but misdiagnoses.
        if "**fk_kw" in _seg and "t_sampler=a.t_sampler" in _seg:
            _lcall = textwrap.dedent(_seg); break
check(_lcall is not None, "the trainer's own TRAINING cfm_loss call was found (the one expanding fk_kw)")
for _gns, _tag, _cfg in ((_ns_g, "spectral+temporal", CFG_G), (_ns_h, "spectral+temporal+augmentation", CFG_H)):
    if _lcall is None or _call is None:
        check(False, f"{_tag}: nothing to weigh"); continue
    _artj = json.load(open(cfg_env(_cfg)["CALIB"]))
    _b = collate([_gns["ds_tr"][0] for _ in range(2)])
    _g = dict(_gns)                       # carries the trainer's own a, base, aug_cfg, ds_tr
    _g["calib"] = _artj
    for _st in _lut_st + _gstmts:
        exec(compile(textwrap.dedent(_st), TRAIN_PY, "exec"), _g)
    _g["b"] = _b
    exec(compile(_call, TRAIN_PY, "exec"), _g)          # the REAL ktjd_prep, no stub
    _got = {}
    def _cap(model, x, **kw):
        _got.clear(); _got.update(kw)
        return torch.zeros((), requires_grad=False)
    # codex bothrope r11: fk_kw={} omitted four auxiliary weights that are live in this recipe, and a mutant
    # zeroing gamma_acc only for the combined arms passed. The trainer's own fk_kw block is executed instead, at a
    # gstep past the warmup so the fk ramp is complete.
    _g.update(cfm_loss=_cap, model=trainer_builds(_g["a"]), gstep=10 ** 7)
    exec(compile(_fkblk, TRAIN_PY, "exec"), _g)
    exec(compile(_lcall, TRAIN_PY, "exec"), _g)
    _aux_want = {k: float(getattr(_g["a"], k)) for k in ("gamma_vel", "gamma_lock", "gamma_acc", "gamma_fk")
                 if float(getattr(_g["a"], k)) > 0}
    _aux_got = {k: float(_got.get(k, float("nan"))) for k in _aux_want}
    check(_aux_want and _aux_got == _aux_want,
          f"{_tag}: THE AUXILIARY WEIGHTS THAT REACH cfm_loss are this arm's configured ones "
          f"(got {_aux_got}, want {_aux_want})")
    # codex bothrope r12: read at a post-warmup step only, a mutant pinning ramp=1.0 is invisible -- the first real
    # training step would get gamma_fk 0.07 instead of 1.4e-5, 5000x too strong. Read the ramp along the warmup.
    # codex bothrope r13: "increasing" is satisfied by any curve -- a sqrt ramp gives the first step 70x too much
    # weight and still finishes at the configured value. Sampled on an even grid, a LINEAR ramp has equal
    # increments; sqrt does not. The endpoint is checked separately.
    _W = int(_g["a"].fk_warmup_steps)
    _grid = [0, _W // 4, _W // 2, (3 * _W) // 4]
    _fk_at = []
    for _gs in _grid + [10 ** 7]:
        _h = dict(_g); _h["gstep"] = _gs; _got.clear()
        exec(compile(_fkblk, TRAIN_PY, "exec"), _h)
        exec(compile(_lcall, TRAIN_PY, "exec"), _h)
        _fk_at.append(float(_got.get("gamma_fk", float("nan"))))
    _d = [_fk_at[i + 1] - _fk_at[i] for i in range(3)]
    _lin = max(_d) - min(_d) < 1e-9 * max(1.0, max(_d)) + 1e-12
    check(_lin and _fk_at[0] < _fk_at[3] < _fk_at[4] and abs(_fk_at[4] - float(_g["a"].gamma_fk)) < 1e-12,
          f"{_tag}: the FK weight ramps LINEARLY over the warmup and reaches its configured value only after it "
          f"(grid {_grid} -> {[f'{v:.4g}' for v in _fk_at[:4]]}, after {_fk_at[4]:.3g} vs "
          f"{float(_g['a'].gamma_fk):.3g}; increments {[f'{v:.4g}' for v in _d]})")
    # codex bothrope r14: equal increments pin the SHAPE, not the slope or the duration -- a mutant doubling the
    # denominator keeps them equal, halves every weight and stretches the warmup to 2W. Pinning where the ramp
    # SATURATES fixes both: with linearity and the endpoint value already checked, "full strength exactly at the
    # configured warmup step, and not one step earlier" determines the whole curve.
    _sat = []
    for _gs in (_W - 2, _W - 1):
        _h = dict(_g); _h["gstep"] = _gs; _got.clear()
        exec(compile(_fkblk, TRAIN_PY, "exec"), _h)
        exec(compile(_lcall, TRAIN_PY, "exec"), _h)
        _sat.append(float(_got.get("gamma_fk", float("nan"))))
    _full = float(_g["a"].gamma_fk)
    check(abs(_sat[1] - _full) < 1e-12 and _sat[0] < _full - 1e-12,
          f"{_tag}: ...and it saturates EXACTLY at the configured warmup of {_W} steps "
          f"(step {_W - 2} {_sat[0]:.6g} < {_full:.6g}, step {_W - 1} {_sat[1]:.6g})")
    # codex bothrope r15: linearity plus a saturation point still leaves the INTERCEPT free -- `ramp = 0.5 + 0.5*ramp`
    # keeps the increments equal and saturates on time while giving the first step 2500x the weight. Three rounds of
    # property-only constraints each left a gap here, so the contract itself is stated: a linear warmup runs from ONE
    # STEP'S SHARE of the configured weight to the full weight over the configured number of steps. With the
    # increments and the saturation step already checked, this closes the curve.
    _want0 = _full / _W
    check(abs(_fk_at[0] - _want0) < 1e-12 * max(1.0, _full),
          f"{_tag}: ...and it STARTS at one step's share of the configured weight "
          f"({_fk_at[0]:.6g} vs {_full:.6g}/{_W} = {_want0:.6g})")
    if _dropfn is not None:
        _dg = dict(trm.__dict__); _dg.update(a=_g["a"], dev="cpu")
        exec(compile(_dropfn, TRAIN_PY, "exec"), _dg)
        _apply = _dg["apply_cfg_drops"]
        torch.manual_seed(20260916)
        _rows = _dropped = 0
        for _ in range(200):
            _bb = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in _b.items()}
            _t = _apply(_bb)["text"]
            _dropped += int((_t.abs().sum(-1) == 0).sum()); _rows += _t.shape[0]
        _rate = _dropped / max(1, _rows)
        _pt, _pb = float(_g["a"].p_drop_text), float(_g["a"].p_drop_both)
        _expect = _pt + _pb - _pt * _pb
        # codex bothrope r13: a tolerance wider than the probability itself accepts a rate of ZERO, i.e. a mutant
        # that silently trains the combined arms with text conditioning on every sample. The band is now relative
        # and excludes both 0 and 1 (400 rows, sd of the rate ~0.015, so +-0.5x the probability is ~3 sd).
        check(0 < _rate and abs(_rate - _expect) < 0.5 * _expect,
              f"{_tag}: THE CONDITIONING DROPOUT drops text at this arm's configured rate, and not at 0 or 1 "
              f"({_rate:.3f} over {_rows} samples vs p_text {_pt} | p_both {_pb} -> {_expect:.3f}, "
              f"band {_expect * 0.5:.3f}..{_expect * 1.5:.3f})")
    _want_g = {k: float(v) for k, v in _artj["gammas"].items()}
    _arrived = {k: float(v) for k, v in (_got.get("gammas") or {}).items()}
    check(_arrived == _want_g,
          f"{_tag}: THE WEIGHTS THAT REACH cfm_loss ITSELF are this arm's calibrated ones "
          f"(root_pos {_arrived.get('root_pos')} vs {_want_g.get('root_pos')}, "
          f"{sum(1 for k in _want_g if _arrived.get(k) != _want_g[k])} of {len(_want_g)} groups differ)")

# the complement, so the two together are exhaustive: at AUG_P=0.8 about a fifth of the draws are NOT augmented, and
# for those the rig's cached base basis IS the right answer. Every draw is therefore accounted for -- edited tree =>
# different coordinates, intact tree => exactly the base coordinates -- and neither case can be satisfied by a stale
# per-rig basis.
check(n_intact == n_seen - n_edited,
      f"arm H loader: every UNedited draw got exactly its rig's base coordinates ({n_intact}/{n_seen - n_edited}) "
      f"-- with the {n_edited} edited ones above, all {n_seen} draws are accounted for")

# and the combined model eats a real augmented batch end to end through the trainer's own cond_of
# the collate carries the spectral coordinates into the batch the model actually reads, and rigs of different sizes
# are padded together -- so a row that lost its own coordinates, or padding that is not zero, would rotate the wrong
# joints. Checked against each item's own values before the batch is fed to the model.
_items = [ds_aug[0] for _ in range(3)]
bt = collate(_items)
_Jm = bt["x"].shape[2]
_rowok = all(torch.equal(bt["spectral_feats"][k, :int(it["n_joints"])], it["spectral_feats"])
             and bool((bt["spectral_feats"][k, int(it["n_joints"]):] == 0).all())
             for k, it in enumerate(_items))
check(tuple(bt["spectral_feats"].shape) == (3, _Jm, 8) and _rowok,
      f"collate: every row's spectral coordinates are that item's own, zero beyond its joint count "
      f"(padded to [3,{_Jm},8] from rigs of {[int(it['n_joints']) for it in _items]} joints)")
c = trm.cond_of(bt)
with torch.no_grad():
    ob = mG(bt["x"][..., :17], torch.rand(bt["x"].shape[0]), is_target=bt["is_target"], **c)
check(tuple(ob.shape) == tuple(bt["x"][..., :17].shape) and torch.isfinite(ob).all(),
      f"arm H: the COMBINED model consumes a real augmented, padded batch through cond_of {tuple(ob.shape)}")

print(f"\n{'ALL PASS' if not fails else str(len(fails)) + ' FAILURES'} -- {time.time()-T0:.0f}s")
for f in fails: print("  -", f)
sys.exit(1 if fails else 0)
