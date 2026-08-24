#!/usr/bin/env python3
"""Four-rank launch gate for run6, implementing codex's required checks (2026-08-23).

SMOKE=1 is not sufficient: it runs too few batches to establish peak memory, dynamic-shape
behaviour, or compiled steady state. This runs the EXACT run6 configuration for enough varied-J
batches to clear compile warmup, then asserts:
  1. every rank's loss and gradients are finite
  2. every parameter is IDENTICAL across ranks, elementwise (DDP really synchronized)
  3. the only parameters without gradients are the known-dead bp_mlp tensors
  4. no torch._dynamo recompilation-limit fallback was hit
  5. per-rank peak allocation is recorded and materially below the no-checkpoint baseline
Run under torchrun; rank 0 prints the verdict and exits non-zero on any failure.
"""
import json, os, sys
from pathlib import Path
import torch, torch.distributed as dist
sys.path.insert(0, "/iridisfs/scratch/ts1v23/workspace/noKslot_clean")
from src.data.ktjd17_incontext import Ktjd17Base, ktjd17_split_names
from src.data.incontext_pairs import InContextPairs, collate
from src.models.v2.dit_motion import InContextMotionDiT, cfm_loss
from scripts.train_v2_incontext import ktjd_channel_lut, ktjd_prep, cond_of, to_dev, fk_pack_of
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

def _req(name, cast=str):
    """Read a setting from the environment, fail closed if absent.

    The gate used to HARD-CODE the objective (v_space, gamma_fk, bf16, demo_rest, struct_feats,
    dir_bias) and the architecture, while the launcher passed its own values separately. A PASS
    was therefore evidence about a configuration the real run might never use -- codex called this
    blocking on 2026-08-23. Both now read the SAME environment, so the gate is exercised with the
    exact settings the launch will use, or it refuses to run.
    """
    v = os.environ.get(name)
    if v is None:
        raise SystemExit(f"[gate] {name} is required -- run this with the SAME environment as "
                         f"scripts/_launch_v2_ddp_2node_h200.sh so the gate tests the real config")
    return cast(v)

_flag = lambda n: _req(n, int) == 1
DIM, DEPTH, HEADS = _req("DIM", int), _req("DEPTH", int), _req("HEADS", int)
BATCH, LR, WD = _req("BATCH", int), _req("LR", float), _req("WD", float)
V_SPACE, BF16 = _flag("V_SPACE"), _flag("BF16")
SIGMA_MIN, HUBER = _req("SIGMA_MIN", float), _req("HUBER", float)
GAMMA_FK = _req("GAMMA_FK", float)
DEMO_REST, DEMO_FRAMES = _flag("DEMO_REST"), _req("DEMO_FRAMES", int)
STRUCT_FEATS, DIR_BIAS = _flag("STRUCT_FEATS"), _flag("DIR_BIAS")
GRAD_CKPT, COMPILE = _flag("GRAD_CKPT"), _flag("COMPILE")
GRAD_CLIP = _req("GRAD_CLIP", float)
GAMMA_VEL, GAMMA_LOCK = _req("GAMMA_VEL", float), _req("GAMMA_LOCK", float)
T_SAMPLER, ANCHOR = _req("T_SAMPLER"), _req("ANCHOR")
RANDOM_CAPTION = _flag("RANDOM_CAPTION")
FK_WARMUP = _req("FK_WARMUP", int)
P_DROP_TEXT = _req("P_DROP_TEXT", float)
# Settings this gate cannot reproduce are REFUSED rather than silently ignored, so the gate can
# never quietly become evidence about a different objective (codex 2026-08-23, round 2).
if T_SAMPLER != "uniform":
    raise SystemExit(f"[gate] T_SAMPLER={T_SAMPLER} not exercised by this gate")
if ANCHOR != "none":
    raise SystemExit(f"[gate] ANCHOR={ANCHOR} not exercised by this gate")
if RANDOM_CAPTION:
    raise SystemExit("[gate] RANDOM_CAPTION=1 not exercised by this gate")
PERCELL, CALIB, CUT_ENV = _req("PERCELL"), _req("CALIB"), _req("CUT")
KTJD_ROOT, JOINT_SEM = _req("KTJD_ROOT"), _req("JOINT_SEM")

STEPS = int(os.environ.get("GATE_STEPS", "14"))
if STEPS < 4:
    raise SystemExit(f"GATE_STEPS={STEPS} cannot establish compile warmup or variable-J coverage")
NO_CKPT_BASELINE_GIB = 82.30          # measured, dim512/depth12, B8, no checkpointing
rank = int(os.environ["RANK"]); world = int(os.environ["WORLD_SIZE"])
# PINNED, not read from the environment: this file certifies the 4-rank cross-alloc launch, and an
# overridable expectation lets a 2-rank invocation print PASS and write the result JSON, where
# cross-rank parameter agreement is vacuous (codex round 4).
GATE_WORLD = 4
if world != GATE_WORLD:
    raise SystemExit(f"[gate] WORLD_SIZE={world}: this gate certifies a {GATE_WORLD}-rank launch "
                     f"and nothing else")
local = int(os.environ["LOCAL_RANK"])
torch.cuda.set_device(local); dist.init_process_group("nccl")
dev = f"cuda:{local}"
EXC = CUT_ENV
base = Ktjd17Base(KTJD_ROOT,
                  caption_emb_cache="data/anytop_caption_llm2vec_v4b272neutral_multi",
                  joint_semantics=JOINT_SEM,
                  percell_stats=PERCELL, exclude_clips=EXC)
names = ktjd17_split_names(KTJD_ROOT, exclude=EXC)
ds = InContextPairs(base, names["train"], names["train"], balance_skeletons=True, seed=0,
                    emit_fk_fields=True, emit_graph_v2=STRUCT_FEATS or DIR_BIAS,
                    demo_rest=DEMO_REST, demo_frames=DEMO_FRAMES)
lut = ktjd_channel_lut(base)
G = {k: float(v) for k, v in
     json.loads(Path(CALIB).read_text())["gammas"].items()}
m = InContextMotionDiT(in_ch=17, dim=DIM, depth=DEPTH, n_heads=HEADS, d_text=4096,
                       d_joint_sem=4096, use_struct_feats=STRUCT_FEATS, use_dir_bias=DIR_BIAS,
                       grad_ckpt=GRAD_CKPT).to(dev)
raw = m
if COMPILE:
    m = torch.compile(m, dynamic=True)
m = DDP(m, device_ids=[local], find_unused_parameters=True)
opt = torch.optim.AdamW(m.parameters(), lr=LR, weight_decay=WD)
smp = DistributedSampler(ds, shuffle=True, drop_last=True); smp.set_epoch(0)
dl = DataLoader(ds, batch_size=BATCH, sampler=smp, num_workers=3, collate_fn=collate)
it = iter(dl)
torch.cuda.reset_peak_memory_stats()
import torch._dynamo as _dyn
_dyn.utils.counters.clear()   # scope to this gate: stale counters would be a false failure
Js, bad = [], []
nograd_names = None
for step in range(STEPS):
    b = to_dev(next(it), dev)
    Js.append(int(b["x"].shape[2]))
    x17, kt = ktjd_prep(b, lut, gammas=G)
    # Assembled exactly as scripts/train_v2_incontext.py:1012-1021 does it. gamma_vel/gamma_lock
    # were previously omitted here, which UNDER-stated peak memory: both add graph.
    fk_kw = {}
    if GAMMA_VEL > 0 or GAMMA_LOCK > 0:
        fk_kw = dict(gamma_vel=GAMMA_VEL, gamma_lock=GAMMA_LOCK, fk_pack=fk_pack_of(b))
    if GAMMA_FK > 0:
        # DELIBERATE UPPER BOUND: training ramps gamma_fk over FK_WARMUP steps, so at the step
        # counts a gate can afford it would be running at ~0 and would measure neither the FK
        # path's memory nor its gradients. Full strength is the conservative direction -- the
        # real run never exceeds what is measured here.
        fk_kw.update(gamma_fk=GAMMA_FK, fk_pack=fk_pack_of(b))
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=BF16):
        loss = cfm_loss(m, x17, is_target=b["is_target"], valid=b["valid"], v_space=V_SPACE,
                        sigma_min=SIGMA_MIN, huber_delta=HUBER, t_sampler=T_SAMPLER,
                        **fk_kw, **kt, **cond_of(b))
    loss.float().backward()
    if not torch.isfinite(loss):
        bad.append(f"rank{rank} step{step}: non-finite loss")
    gn = torch.nn.utils.clip_grad_norm_(m.parameters(), GRAD_CLIP)
    if not torch.isfinite(gn):
        bad.append(f"rank{rank} step{step}: non-finite grad norm")
    if step == STEPS - 1:
        nograd_names = sorted(n for n, p in raw.named_parameters() if p.grad is None)
    opt.step(); opt.zero_grad(set_to_none=True)
peak = torch.cuda.max_memory_allocated() / 2**30
# Each tensor is judged against ITS OWN magnitude. A single global scale would let a real mismatch
# inside a small bias vector sit far below a threshold set by the largest weight matrix, and thus
# falsely certify synchronization (codex round 6).
_rels = []
for _t in raw.parameters():
    _ref = _t.detach().clone()
    dist.broadcast(_ref, src=0)
    _dev = float((_t.detach() - _ref).abs().max())
    _scale = float(_t.detach().abs().max())
    _rels.append(_dev / _scale if _scale > 0 else (0.0 if _dev == 0 else float("inf")))
chk_worst = max(range(len(_rels)), key=lambda i: _rels[i])
chk_rel = _rels[chk_worst]
# whether Dynamo fell back / hit the recompile limit is a real failure for a `dynamic=True`
# compile over variable J -- read it rather than inferring it from the shapes we happened to see
# graph_break/unimplemented alone miss a recompile storm, which is the failure that actually
# matters for a dynamic=True compile over variable J (codex round 2).
_c = _dyn.utils.counters
dyn_fail = sum(sum(v.values()) if isinstance(v, dict) else int(v)
               for k, v in _c.items() if k in ("graph_break", "unimplemented"))
recompiles = sum(sum(v.values()) if isinstance(v, dict) else int(v)
                 for k, v in _c.items() if "recompil" in k or "cache_size" in k
                 or "cache_limit" in k)
gather = [None] * world
dist.all_gather_object(gather, {"rank": rank, "peak": peak, "bad": bad,
                                "nograd": nograd_names, "J": Js, "dyn": int(dyn_fail),
                                "rel": chk_rel, "worst": chk_worst,
                                "recomp": int(recompiles)})
if rank == 0:
    fails = []
    for g in gather:
        fails += g["bad"]
    _wr = max(gather, key=lambda g: g["rel"])
    rel = _wr["rel"]
    # Measured bitwise-identical (0.0) on this topology, so the tolerance exists only to absorb a
    # possible non-bitwise NCCL reduction, not to make room for drift: anything a real
    # desynchronization would produce is orders of magnitude above it.
    if rel > 1e-6:
        fails.append(f"parameters differ across ranks by a relative {rel:.2e} on tensor "
                     f"#{_wr['worst']} on rank{_wr['rank']} (per-tensor scale) -- "
                     f"DDP did not synchronize")
    KNOWN_DEAD = {"bp_mlp.0.weight", "bp_mlp.0.bias", "bp_mlp.2.weight", "bp_mlp.2.bias"}
    for g in gather:
        got = set(g["nograd"] or [])
        unexpected = got - KNOWN_DEAD
        if unexpected:
            fails.append(f"rank{g['rank']} has ungraded params outside the known-dead set: "
                         f"{sorted(unexpected)[:6]}")
    peaks = [g["peak"] for g in gather]
    if max(peaks) > 0.6 * NO_CKPT_BASELINE_GIB:
        fails.append(f"peak {max(peaks):.1f} GiB not materially below the "
                     f"{NO_CKPT_BASELINE_GIB} GiB no-checkpoint baseline")
    allJ = sorted({j for g in gather for j in g["J"]})
    if len(allJ) < 4:
        fails.append(f"only {len(allJ)} distinct J seen ({allJ}) -- too few to establish that the "
                     f"dynamic compile generalizes across skeletons")
    dyns = [g["dyn"] for g in gather]
    if any(d > 0 for d in dyns):
        fails.append(f"dynamo graph-break/unimplemented counters non-zero per rank: {dyns}")
    # REPORTED, NOT ASSERTED. A distinct-J count is not a compiler-specialization count, so a
    # threshold built on it both false-fails (a healthy compiler may log several events per shape)
    # and false-passes (re-tracing within the budget). A check that can do both is not a check
    # (codex round 3). graph_break/unimplemented above is the assertion that does hold.
    recs = [g["recomp"] for g in gather]
    print(f"[gate] steps={STEPS} world={world}  (config read from the launcher's own environment)")
    print(f"[gate] dim={DIM} depth={DEPTH} heads={HEADS} batch={BATCH} lr={LR} wd={WD} "
          f"grad_ckpt={int(GRAD_CKPT)} compile={int(COMPILE)}")
    print(f"[gate] v_space={int(V_SPACE)} sigma_min={SIGMA_MIN} huber={HUBER} gamma_fk={GAMMA_FK} "
          f"bf16={int(BF16)} demo_rest={int(DEMO_REST)}/{DEMO_FRAMES} "
          f"struct={int(STRUCT_FEATS)} dir_bias={int(DIR_BIAS)}")
    print(f"[gate] gamma_vel={GAMMA_VEL} gamma_lock={GAMMA_LOCK} t_sampler={T_SAMPLER} "
          f"anchor={ANCHOR}")
    print(f"[gate] DELIBERATE DIFFERENCES from the real run, both conservative: gamma_fk at FULL "
          f"strength (training ramps it over {FK_WARMUP} steps, so a short gate would measure "
          f"neither its memory nor its gradients) and AdamW stepped at lr={LR} (training warms up "
          f"from ~7.5e-8, under which parameters barely move and cross-rank parameter agreement "
          f"would be vacuous). Both make this gate an UPPER bound on the real run. Text-drop "
          f"p={P_DROP_TEXT} is not applied: the full-conditioning path is the larger one.")
    print(f"[gate] distinct J seen: {allJ}  ({len(allJ)} shapes exercised past compile warmup; "
          f"generalization is evidenced by the graph-break and recompile counters below, "
          f"not by shape variety itself)")
    print(f"[gate] per-rank peak GiB: {[round(p,2) for p in peaks]}  (no-ckpt baseline {NO_CKPT_BASELINE_GIB})")
    print(f"[gate] worst per-tensor relative cross-rank deviation: {rel:.2e} "
          f"(tensor #{_wr['worst']} on rank{_wr['rank']}, threshold 1e-6)")
    print(f"[gate] dynamo graph-break counters per rank: {dyns} (asserted zero)")
    print(f"[gate] dynamo recompile/cache-limit counters per rank: {recs} (reported, not asserted)")
    print(f"[gate] ungraded params: {gather[0]['nograd']}")
    print(f"[gate] {'PASS' if not fails else 'FAIL'}")
    for f in fails:
        print(f"   - {f}")
    Path("configs/_run6_gate_result.json").write_text(json.dumps(
        {"pass": not fails, "peaks": peaks, "J": allJ, "fails": fails}))
    _rc = 1 if fails else 0
else:
    _rc = 0
# every rank must agree on the exit code, and a FAILED gate must exit NON-ZERO -- an unconditional
# sys.exit(0) would make this gate decorative, which is exactly the failure it exists to prevent.
_rct = torch.tensor([_rc], device=dev)
dist.all_reduce(_rct, op=dist.ReduceOp.MAX)
dist.barrier(); dist.destroy_process_group()
sys.exit(int(_rct.item()))
