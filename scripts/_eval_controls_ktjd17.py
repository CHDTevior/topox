#!/usr/bin/env python3
"""Evaluator-validation controls and uncertainty for the frozen gen-eval, computed on SAVED frozen-protocol samples.

Reviewers of the draft (2026-09-08) asked what the near-ceiling retrieval numbers of the skeleton-aware evaluator actually
measure. This tool re-scores one arm's saved shards (scripts/_eval_v2_gen_in_evalspace.py --save_gen) under the frozen protocol
(pool 64, dataset-order pools, group-aware R@K, FID in the evaluator space) and adds the controls that decide it:

  gen           the saved generations, converted into the evaluator's per-cell KTJD-17 space exactly as the merge stage does
  gt-ceiling    the dataset's own real-clip tensors (the text->GT ceiling of the protocol), scored identically
  fk            gen with its POSITION and VELOCITY channels replaced by the forward kinematics of its own ROTATION channels
                (official codec): does retrieval survive when the deployable FK output is what is scored?
  static        the real clip's first pose held for the whole clip, zero velocity (species + posture, no motion)
  rest          the rig's rest frame held for the whole clip (skeleton only)
  tshuffle      gen with its frames randomly permuted (content without temporal order)
  capshuffle    a WRONG caption at every target slot (text rows permuted within the pool, acceptance sets kept): the chance
                level of the protocol under its own multi-positive rule
  within-rig    retrieval pools made of ONE rig's validation clips (gen and gt), chance = 1/pool for each rig

Uncertainty: a 95% bootstrap interval over pools for every R@K (pools are the unit of the protocol), a bootstrap interval over
clips for FID(gen, GT), and the real-vs-real FID floor (random halves of the real clips).

  python scripts/_eval_controls_ktjd17.py --gen_ckpt CKPT --shards S0.npz ... --report gen_eval_X.json --out controls.json
  (the --report is the canonical merge of the same shards; --limit N >= pool runs a smoke without it). Scoring runs under the shards'
  recorded runtime. Note: static / rest hold the pose for the clip's real length, so duration is still visible to the evaluator.
"""
from __future__ import annotations
import argparse, hashlib, io, json, sys
from pathlib import Path
import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from scripts._eval_v2_gen_in_evalspace import load_evaluator, encode_split, fid, PROTOCOL_VAL_N, generation_plan, make_pairs  # noqa: E402
from src.data.ktjd17_incontext import ktjd17_split_names                                                  # noqa: E402
from scripts._eval_evaluator_sanity import rprecision_pool                                                # noqa: E402
from src.models.graph_salad.t2m_evaluator import build_false_negative_mask                                 # noqa: E402
from src.data.ktjd17_incontext import Ktjd17Base, _STD_FLOOR                                              # noqa: E402
from src.data.ktjd17_t2m_eval_dataset import Ktjd17T2MEvalDataset                                        # noqa: E402
from src.data.ktjd17.loader import load_motion_npz                                                        # noqa: E402
from src.data.ktjd17.decoder import decode_ktjd17                                                         # noqa: E402

EVAL_CKPT = "runs/evaluator_ktjd16_pz_v1/best_model.pt"


def wrong_caption_perm(caps):
    """A permutation of the pool's caption rows that gives every slot a DIFFERENT caption text whenever that is feasible
    (codex r3/r4 P2): slots are sorted by caption text so identical captions are contiguous, then rotated by the size of the
    largest caption group -- a rotation by k >= max_group moves every slot out of its own group when max_group <= n - max_group,
    i.e. whenever a zero-residual assignment exists at all; otherwise the residual 2*max_group - n is the minimum possible.
    No fixed point either (k >= 1). Deterministic. Returns (perm, n_identical)."""
    n = len(caps)
    order = sorted(range(n), key=lambda i: (caps[i], i))
    groups = {}
    for i in range(n):
        groups[caps[i]] = groups.get(caps[i], 0) + 1
    k = max(1, max(groups.values()))
    perm = torch.empty(n, dtype=torch.long)
    for r in range(n):
        perm[order[r]] = order[(r + k) % n]                 # slot order[r] receives the caption of slot order[r+k]
    bad = sum(1 for i in range(n) if caps[int(perm[i])] == caps[i])
    return perm, bad


def pooled_rprec(te, me, meta, pool, masked=True, shuffle=False, identical_out=None):
    """Per-pool group-aware R@K in dataset order (the protocol's pools); returns a list of {1,2,3} dicts."""
    mids, smids, caps = meta["motion_id"], meta["source_motion_id"], meta["caption_text"]
    out = []
    for p in range(te.shape[0] // pool):
        s = slice(p * pool, (p + 1) * pool)
        te_p, me_p = te[s], me[s]
        m = build_false_negative_mask(mids[s], smids[s], caps[s]) if masked else None
        if shuffle:
            # a WRONG caption at every target slot; the slot's own acceptance set stays (codex r2 P1): does a mismatched caption
            # still retrieve this slot's group? Exact assignment: identical caption text only where the pool's duplicates force it.
            perm, bad = wrong_caption_perm(caps[s])
            if identical_out is not None:
                identical_out.append(bad)
            te_p = te_p[perm]
        out.append(rprecision_pool(te_p, me_p, m))
    return out


def group_chance(meta, pool, masked=True):
    """Chance R@1 of a random top-1 under the acceptance rule: mean over queries of |acceptable(i)| / pool, averaged over pools."""
    mids, smids, caps = meta["motion_id"], meta["source_motion_id"], meta["caption_text"]
    ch = []
    for p in range(len(mids) // pool):
        s = slice(p * pool, (p + 1) * pool)
        if masked:
            m = build_false_negative_mask(mids[s], smids[s], caps[s]).bool() | torch.eye(pool, dtype=torch.bool)
            ch.append(float((m.sum(1).double() / pool).mean()))
        else:
            ch.append(1.0 / pool)
    return float(np.mean(ch)) if ch else None


def boot_mean(values, rng, n_boot):
    v = np.asarray(values, dtype=np.float64)
    idx = rng.integers(0, len(v), size=(n_boot, len(v)))
    b = v[idx].mean(axis=1)
    return float(v.mean()), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def summarize(pools, rng, n_boot):
    return {str(k): dict(zip(("mean", "lo95", "hi95"), boot_mean([p[k] for p in pools], rng, n_boot))) for k in (1, 2, 3)}


def within_rig(te, me, meta, rigs, min_pool, pool_cap, masked=True):
    """Pools made of one rig's clips (dataset order within the rig; chunked at pool_cap; rigs below min_pool skipped)."""
    by = {}
    for i, r in enumerate(rigs):
        by.setdefault(r, []).append(i)
    res, chance, n_pools, n_rigs = {1: [], 2: [], 3: []}, [], 0, 0
    mids, smids, caps = meta["motion_id"], meta["source_motion_id"], meta["caption_text"]
    for r, idx in by.items():
        if len(idx) < min_pool:
            continue
        n_rigs += 1
        for s in range(0, len(idx) - (len(idx) % min_pool if len(idx) < pool_cap else 0), pool_cap):
            ii = idx[s:s + pool_cap]
            if len(ii) < min_pool:
                continue
            ii_t = torch.as_tensor(ii)
            m = build_false_negative_mask([mids[i] for i in ii], [smids[i] for i in ii], [caps[i] for i in ii]) if masked else None
            rp = rprecision_pool(te[ii_t], me[ii_t], m)
            for k in res:
                res[k].append(rp[k])
            # chance of a random top-1 hit = mean over queries of (acceptable candidates incl. the true pair) / pool size (codex r1 P2)
            acc = torch.ones(len(ii), dtype=torch.float64) if m is None else (m.clone().bool() | torch.eye(len(ii), dtype=torch.bool)).sum(1).double()
            chance.append(float((acc / len(ii)).mean())); n_pools += 1
    return {"n_rigs": n_rigs, "n_pools": n_pools, "chance_R1_mean": float(np.mean(chance)) if chance else None,
            **{f"R{k}": float(np.mean(v)) if v else None for k, v in res.items()}}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gen_ckpt", required=True); ap.add_argument("--shards", nargs="+", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--eval_ckpt", default=EVAL_CKPT); ap.add_argument("--pool", type=int, default=64); ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n_boot", type=int, default=2000); ap.add_argument("--encode_batch", type=int, default=64)
    ap.add_argument("--min_rig_pool", type=int, default=8); ap.add_argument("--fid_halves", type=int, default=20); ap.add_argument("--fid_boot", type=int, default=200)
    ap.add_argument("--report", default=None, help="the canonical merge report of these shards (gen_eval_*.json); required unless --limit")
    ap.add_argument("--allow_variant", action="store_true", help="accept a non-frozen protocol (steps/cfg/pool/seed) -- labelled in the output")
    ap.add_argument("--limit", type=int, default=0, help="score only the first N clips in dataset order (smoke; not the protocol)")
    a = ap.parse_args()
    if a.limit and a.limit < a.pool:
        raise SystemExit(f"[refuse] --limit {a.limit} is smaller than one pool of {a.pool}")
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    rng = np.random.default_rng(a.seed)

    buf = Path(a.gen_ckpt).read_bytes(); gen_sha = hashlib.sha256(buf).hexdigest()
    ck = torch.load(io.BytesIO(buf), map_location="cpu", weights_only=False); del buf
    ca = ck["args"]; ca = vars(ca) if not isinstance(ca, dict) else ca
    root, excl = ca["ktjd_root"], (ca.get("exclude_clips") or None)
    base = Ktjd17Base(root, caption_emb_cache=ca["caption_cache"], joint_semantics=ca["joint_sem"], texts_json=ca["texts_json"],
                      percell_stats=ca.get("ktjd_percell_stats", "data/ktjd17_percell_stats_v1.npz"), exclude_clips=excl,
                      normalization=str(ca.get("rep_norm") or "percell"))
    live = {**base.provenance, "exclusion": base.provenance_exclusion}
    pins = ck.get("ktjd_pins") or {}
    drift = sorted(k for k, v in pins.items() if k in live and live[k] != v)
    if drift:
        raise SystemExit(f"[refuse] data drift vs the checkpoint's ktjd_pins: {drift}")
    prov_sha = hashlib.sha256(json.dumps(live, sort_keys=True, default=str).encode()).hexdigest()
    rep = ((getattr(base, "derivation", None) or {}).get("representation") or {})
    if ("representation" in live) != ("representation" in pins) or pins.get("representation") != live.get("representation"):
        raise SystemExit(f"[refuse] representation pin mismatch: live {live.get('representation')!r} vs checkpoint {pins.get('representation')!r}")
    if rep:
        from src.data.ktjd17_anytop13 import REPRESENTATION_ID, anytop13_to_ktjd17
        if str(rep.get("id")) != REPRESENTATION_ID:
            raise SystemExit(f"[refuse] unknown representation view {rep.get('id')!r}")
        parent_root, parent_stats = str(base.derivation["parent_root"]), str(rep["parent_norm_stats"]["path"])
        if hashlib.sha256(Path(parent_stats).read_bytes()).hexdigest() != str(rep["parent_norm_stats"]["sha256"]):
            raise SystemExit(f"[refuse] parent stats {parent_stats} do not match the sha pinned in the view's derivation.json")
        if hashlib.sha256((REPO / "src" / "data" / "ktjd17_anytop13.py").read_bytes()).hexdigest() != str(rep.get("converter_sha256")):
            raise SystemExit("[refuse] src/data/ktjd17_anytop13.py differs from the converter the view was built with")
        if hashlib.sha256((Path(parent_root) / "manifests" / "clips.jsonl").read_bytes()).hexdigest() != str(base.derivation.get("parent_manifest_sha256")):
            raise SystemExit("[refuse] the parent manifest differs from the one the view was derived from")
    else:
        parent_root, parent_stats = root, ca.get("ktjd_percell_stats", "data/ktjd17_percell_stats_v1.npz")
    base_eval = base if (parent_root == root and base.normalization == "percell") else Ktjd17Base(
        parent_root, caption_emb_cache=ca["caption_cache"], joint_semantics=ca["joint_sem"], texts_json=ca["texts_json"],
        percell_stats=parent_stats, exclude_clips=excl, normalization="percell")
    eval_ds = Ktjd17T2MEvalDataset(base_eval, "val", max_frames=240, exclude=excl)
    if len(eval_ds) != PROTOCOL_VAL_N:
        raise SystemExit(f"[refuse] eval val has {len(eval_ds)} clips, protocol pins {PROTOCOL_VAL_N}")
    sch = json.loads((Path(parent_root) / "schema.json").read_text()); fps, eps_h = float(sch["fps_target"]), float(sch["heading"]["eps_h"])
    rows = {str(r["clip_id"]): r for r in base_eval._rows}; T_of = {str(r["clip_id"]): int(r["T_target"]) for r in base._rows}
    target_frames = int(ca["target_frames"])

    # ---- shards: identity + one protocol ----
    PROTO = ("seed", "steps", "cfg_text", "nshards", "gen_batch", "ktjd_root", "exclude_clips", "base_provenance_sha256", "plan_sha256", "protocol_variant", "source_fingerprint")
    gens, proto, shard_sha, shard_idx = {}, None, {}, {}
    for p in a.shards:
        blob = Path(p).read_bytes()                                                 # ONE buffer: hashed and loaded (codex r3 P1)
        shard_sha[p] = hashlib.sha256(blob).hexdigest()
        z = np.load(io.BytesIO(blob), allow_pickle=False); meta = json.loads(str(z["__meta"])); del blob
        shard_idx[p] = int(meta["shard"])
        if str(meta.get("gen_ckpt")) != a.gen_ckpt or str(meta.get("gen_ckpt_sha256")) != gen_sha:
            raise SystemExit(f"[refuse] shard {p} is not from {a.gen_ckpt} (sha {gen_sha[:12]})")
        this = {k: meta.get(k) for k in PROTO}; this["runtime"] = meta.get("runtime")
        if proto is None:
            proto = this
        elif this != proto:
            raise SystemExit(f"[refuse] shard {p} does not share the first shard's protocol")
        if str(meta.get("base_provenance_sha256")) != prov_sha:
            raise SystemExit(f"[refuse] shard {p} provenance {str(meta.get('base_provenance_sha256'))[:12]} != live {prov_sha[:12]}")
        for k in z.files:
            if k.startswith("clip__"):
                mid = k[6:]
                if mid in gens:
                    raise SystemExit(f"[refuse] clip {mid} in two shards")
                gens[mid] = np.asarray(z[k], dtype=np.float32)
    nsh = proto.get("nshards")
    seen = sorted(shard_idx.values())
    if not (isinstance(nsh, int) and seen == list(range(nsh))):
        raise SystemExit(f"[refuse] shards given: indices {seen}, the pass has nshards={nsh!r}; pass every shard exactly once")
    # the CANONICAL merge validated the plan, the fingerprints and the runtime; these shards must be the ones it scored (codex r1 P1)
    if not a.limit:
        if not a.report:
            raise SystemExit("[refuse] --report <gen_eval_*.json of these shards> is required for a protocol run (use --limit for a smoke)")
        rp = json.loads(Path(a.report).read_text()); rg = rp["protocol"]["generation"]
        if (rp["protocol"].get("gen_ckpt_sha256") != gen_sha or rg.get("plan_sha256") != proto.get("plan_sha256")
                or rg.get("source_fingerprint") != proto.get("source_fingerprint") or rg.get("runtime") != proto.get("runtime")):
            raise SystemExit("[refuse] the shards are not the set the canonical report scored (ckpt sha / plan / fingerprint / runtime differ)")
        want = {int(sh["shard"]): str(sh["sha256"]) for sh in rg.get("shards", [])}
        have = {shard_idx[p]: shard_sha[p] for p in a.shards}
        if want != have:
            raise SystemExit(f"[refuse] shard bytes differ from the canonical report's shard hashes: report {sorted(want)} vs given {sorted(have)} / sha mismatch")
        if rp["protocol"].get("subset") not in (None, {}) or int(rp["protocol"].get("val_n", 0)) != PROTOCOL_VAL_N or int(rp["protocol"].get("pool", 0)) != a.pool:
            raise SystemExit(f"[refuse] the report is not a full-set pool-{a.pool} scoring (subset {rp['protocol'].get('subset')!r}, val_n {rp['protocol'].get('val_n')}, pool {rp['protocol'].get('pool')})")
        report_eval_sha = str(rp["protocol"].get("eval_ckpt_sha256"))
    frozen = (not a.limit and a.pool == 64 and proto.get("steps") == 20 and float(proto.get("cfg_text")) == 2.0 and proto.get("seed") == 42
              and a.seed == 42 and proto.get("protocol_variant") in (None, "None"))      # smoke runs and other scoring seeds are never "frozen" (codex r2 P2)
    if not a.limit and not frozen and not a.allow_variant:              # a smoke run is never a protocol claim; only protocol runs are gated
        raise SystemExit(f"[refuse] not the frozen protocol (pool 64 / 20 steps / cfg 2 / seed 42): pool {a.pool}, steps {proto.get('steps')}, "
                         f"cfg {proto.get('cfg_text')}, seed {proto.get('seed')}, variant {proto.get('protocol_variant')!r}; pass --allow_variant to score anyway")
    # score under the SAME runtime the shards were generated with (codex r1 P2)
    import os
    runtime_mismatch = {}
    if "NVIDIA_TF32_OVERRIDE" in os.environ:
        raise SystemExit("[refuse] NVIDIA_TF32_OVERRIDE is set; the recorded runtime cannot see it")
    rt = proto.get("runtime") or {}
    if rt:
        torch.backends.cuda.matmul.allow_tf32 = bool(rt.get("allow_tf32_matmul")); torch.backends.cudnn.allow_tf32 = bool(rt.get("allow_tf32_cudnn"))
        torch.set_float32_matmul_precision(str(rt.get("float32_matmul_precision", "highest")))
        now = {"torch": torch.__version__, "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version()}
        diff = {k: (rt.get(k), now[k]) for k in now if rt.get(k) != now[k]}
        runtime_mismatch = dict(diff)
        if diff:
            frozen = False                                                           # a different arithmetic is never the frozen protocol (codex r3 P2)
        if diff and not a.allow_variant and not a.limit:
            raise SystemExit(f"[refuse] scoring runtime differs from the shards' (torch/cuda/cudnn): {diff}; pass --allow_variant to score anyway (labelled)")
        elif diff:
            print(f"[controls] NOTE: scoring runtime differs from the shards' {diff} (smoke / variant run)", flush=True)

    # ---- build the variants in the evaluator's per-cell space ----
    def to_eval_space(raw, rig, J, Tv):
        mu_p, sd_p = base_eval._stats(rig); cv = base_eval.static_masks(rig)["channel_valid"][:J]
        gp = ((raw - mu_p[None, :J]) / (sd_p[None, :J] + _STD_FLOOR)).astype(np.float32); gp[:, ~cv] = 0.0
        out = np.zeros((target_frames, J, 17), np.float32); out[:Tv] = gp[:Tv]
        return out
    variants = {"gen": {}, "fk": {}, "static": {}, "rest": {}, "tshuffle": {}}
    rig_of, order = {}, []
    n_deg = 0
    for i in range(len(eval_ds)):
        it = eval_ds[i]; mid = str(it["motion_id"]); rig = str(it["object_type"]); order.append(mid); rig_of[mid] = rig
        if a.limit and i >= a.limit:
            continue
        if mid not in gens:
            raise SystemExit(f"[refuse] clip {mid} has no generation")
        sk = base_eval.skeleton(rig); J = len(sk["parents"]); g = gens[mid]
        if g.shape != (target_frames, J, 17):
            raise SystemExit(f"[refuse] clip {mid}: generated shape {g.shape} != ({target_frames}, {J}, 17)")
        mu_v, sd_v = base._stats(rig)
        raw = g.astype(np.float64) * (sd_v[None, :J] + _STD_FLOOR) + mu_v[None, :J]
        Tv = min(T_of[mid], raw.shape[0]); raw = raw[:Tv]
        if rep:
            raw, hv, dg = anytop13_to_ktjd17(raw, np.asarray(sk["parents"])[:J], fps=fps, eps_h=eps_h)
            raw = np.array(raw, dtype=np.float64); raw[~hv, 0, 15:17] = 0.0; n_deg += int(dg["degenerate_facing_frames"]) + int(dg["degenerate_child_slots"])
        # the merge's projection (constants restored) -- the same raw that the evaluator sees, decoded back to raw for the variants
        mu_p, sd_p = base_eval._stats(rig); cv = base_eval.static_masks(rig)["channel_valid"][:J]
        gp = ((raw - mu_p[None, :J]) / (sd_p[None, :J] + _STD_FLOOR)).astype(np.float32); gp[:, ~cv] = 0.0
        raw = gp.astype(np.float64) * (sd_p[None, :J] + _STD_FLOOR) + mu_p[None, :J]
        variants["gen"][mid] = to_eval_space(raw, rig, J, Tv)
        # fk: positions <- FK of the rotations (world), velocities <- forward differences; root track / heading / contact / rotations kept
        dec = decode_ktjd17(raw, parents=sk["parents"], R_rest_global=sk["R_rest_global"], R_rest_local=sk["R_rest_local"],
                            offset_parent_local=sk["offset_parent_local"], rotation_source_kind=sk["rotation_source_kind"], strict_gt=False)
        P = dec.positions_fk; r2 = raw.copy()
        r2[:, :, 0:3] = P; r2[:, :, 0] -= raw[:, 0, 13][:, None]; r2[:, :, 2] -= raw[:, 0, 14][:, None]
        if Tv >= 2:
            r2[:-1, :, 9:12] = (P[1:] - P[:-1]) * fps; r2[-1, :, 9:12] = r2[-2, :, 9:12]
        variants["fk"][mid] = to_eval_space(r2, rig, J, Tv)
        # static: the real clip's first pose held, zero velocity, root track frozen at frame 0
        pay = load_motion_npz(Path(parent_root) / rows[mid]["motion_relpath"], expected_fps_target=fps)
        gt = np.asarray(pay["motion"], dtype=np.float64)[:Tv]
        st = np.repeat(gt[:1], Tv, axis=0); st[:, :, 9:12] = 0.0
        variants["static"][mid] = to_eval_space(st, rig, J, Tv)
        # rest: the rig's rest frame held (skeleton only)
        rr = np.repeat(base_eval._rest_raw17(rig)[None].astype(np.float64), Tv, axis=0)
        variants["rest"][mid] = to_eval_space(rr, rig, J, Tv)
        # tshuffle: the generated frames in random temporal order
        Tw = min(Tv, eval_ds.Tt)                                   # the evaluator scores at most Tt frames: shuffle only that window (codex r4 P2)
        perm = rng.permutation(Tw); ts = variants["gen"][mid].copy(); ts[:Tw] = ts[:Tw][perm]
        variants["tshuffle"][mid] = ts
        if (i + 1) % 500 == 0:
            print(f"[controls] built {i + 1} clips", flush=True)
    keep = set(variants["gen"]) if a.limit else None
    # the pools are dataset-order chunks: bind the order (codex r4 P1). The shards' plan hash covers the rig set, each rig's clip
    # order and the gen_batch chunking; recompute it with the eval script's own generation_plan from THIS eval dataset and require
    # equality. The manifest digest and the ordered-id digest of the scored set are recorded (for a view the parent manifest digest
    # is also pinned by derivation.json and verified above).
    pairs = make_pairs(ca, base, ktjd17_split_names(root, exclude=excl), argparse.Namespace(seed=int(proto["seed"])))   # as the eval script builds it
    plan_now = generation_plan(pairs, base, argparse.Namespace(gen_batch=int(proto["gen_batch"]), nshards=int(nsh)))["plan_sha256"]
    if not a.limit and plan_now != str(proto.get("plan_sha256")):
        raise SystemExit(f"[refuse] the eval dataset's order does not reproduce the shards' generation plan ({plan_now[:12]} vs "
                         f"{str(proto.get('plan_sha256'))[:12]}): the manifest or split changed since the samples were generated")
    elif plan_now != str(proto.get("plan_sha256")):
        print(f"[controls] NOTE: recomputed plan {plan_now[:12]} != shards' {str(proto.get('plan_sha256'))[:12]} (smoke run)", flush=True)
    manifest_sha = hashlib.sha256((Path(parent_root) / "manifests" / "clips.jsonl").read_bytes()).hexdigest()

    core, eval_sha = load_evaluator(a.eval_ckpt, dev)
    if not a.limit and eval_sha != report_eval_sha:
        raise SystemExit(f"[refuse] evaluator {a.eval_ckpt} (sha {eval_sha[:12]}) is not the one the canonical report used ({report_eval_sha[:12]})")
    report = {"gen_ckpt": a.gen_ckpt, "gen_ckpt_sha256": gen_sha, "eval_ckpt_sha256": eval_sha, "protocol": proto, "pool": a.pool,
              "frozen_protocol": bool(frozen), "runtime_mismatch": runtime_mismatch, "report": a.report,
              "manifest_sha256": manifest_sha, "plan_sha256_recomputed": plan_now, "scoring_runtime": {"allow_tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
              "allow_tf32_cudnn": torch.backends.cudnn.allow_tf32, "float32_matmul_precision": torch.get_float32_matmul_precision(),
              "torch": torch.__version__, "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version()}, "smoke": bool(a.limit),
              "seed": a.seed, "n_boot": a.n_boot, "n_clips": len(variants["gen"]), "degenerate_6d_cells_converted": n_deg, "limit": a.limit,
              "gen_representation": str(rep.get("id")) if rep else "ktjd17", "gen_normalization": base.normalization, "variants": {}}
    te = me_gt = meta = None
    for name, gb in variants.items():
        te_, me_gt_, me_v, meta_ = encode_split(core, eval_ds, gb, dev, a, keep=keep)
        if te is None:
            te, me_gt, meta = te_, me_gt_, meta_
            # THE ORDER: the ids actually encoded, in encoding order -- the pools are chunks of exactly this list (codex r5 P3: the
            # digest must describe the scored set, not the whole split). The canonical scoring records the same digest, so an order
            # that differs anywhere -- a re-interleaved manifest that keeps the rig-grouped plan hash included -- is refused here
            # (codex r5 P1); score agreement alone cannot see a permutation that preserves the aggregates.
            order_sha = hashlib.sha256("\n".join(str(m) for m in meta["motion_id"]).encode()).hexdigest()
            report["eval_order_sha256"] = order_sha
            if not a.limit:
                canon_order = str((rp.get("protocol") or {}).get("eval_order_sha256") or "")
                if not canon_order:
                    raise SystemExit(f"[refuse] {a.report} predates the evaluation-order digest (protocol.eval_order_sha256): "
                                     f"re-run the canonical merge of these shards so the scored order is part of the artifact")
                if canon_order != order_sha:
                    raise SystemExit(f"[refuse] the scored order {order_sha[:12]} differs from the canonical report's "
                                     f"{canon_order[:12]}: the manifest or split changed since the report was written")
            rigs = [rig_of[m] for m in meta["motion_id"]]
            gt_pools = pooled_rprec(te, me_gt, meta, a.pool)                 # the dataset's own GT tensor = the text->GT ceiling
            report["gt_ceiling"] = {"rprec": summarize(gt_pools, rng, a.n_boot), "matching_text_cos": float((te * me_gt).sum(-1).mean()),
                                    "within_rig": within_rig(te, me_gt, meta, rigs, a.min_rig_pool, a.pool)}
            g = report["gt_ceiling"]["rprec"]; print(f"[controls] gt-ceiling R@1 {g['1']['mean']:.4f} [{g['1']['lo95']:.4f},{g['1']['hi95']:.4f}] R@3 {g['3']['mean']:.4f} | within-rig {report['gt_ceiling']['within_rig']}", flush=True)
        pools = pooled_rprec(te, me_v, meta, a.pool)
        rec = {"rprec": summarize(pools, rng, a.n_boot), "n_pools": len(pools),
               "matching_text_cos": float((te * me_v).sum(-1).mean()), "gt_cos": float((me_gt * me_v).sum(-1).mean()),
               "fid_vs_gt": fid(me_v, me_gt),
               "within_rig": within_rig(te, me_v, meta, rigs, a.min_rig_pool, a.pool)}
        if name == "gen":
            if not a.limit:
                # the pools are dataset-order chunks: the SAME order, shards and evaluator must reproduce the canonical report's
                # pooled R-precision (a re-interleaved order cannot) -- this binds the controls to the scoring artifact (codex r4 P1)
                canon = {str(k): float(v) for k, v in rp["text_to_gen"]["rprec"].items()}
                got = {k: float(rec["rprec"][k]["mean"]) for k in canon}
                dmax = max(abs(got[k] - canon[k]) for k in canon)
                report["canonical_reproduction"] = {"report_rprec": canon, "controls_rprec": got, "max_abs_diff": dmax,
                                                    "report_fid": float(rp["fid_gen_vs_gt"]), "controls_fid": rec["fid_vs_gt"]}
                if dmax > 1e-3:
                    raise SystemExit(f"[refuse] the gen variant does not reproduce the canonical report's pooled R-precision "
                                     f"(controls {got} vs report {canon}): the evaluation order or the encoding differs from the scoring the report records")
            ident = []
            rec["capshuffle"] = summarize(pooled_rprec(te, me_v, meta, a.pool, shuffle=True, identical_out=ident), rng, a.n_boot)
            rec["capshuffle_identical_caption_slots"] = int(sum(ident))     # slots that unavoidably kept their own caption text
            rec["capshuffle_chance_R1"] = group_chance(meta, a.pool)            # group-aware chance for the shuffled control
            n = me_v.shape[0]; boots = []
            for _ in range(a.fid_boot):
                ia = torch.as_tensor(rng.integers(0, n, n))                     # PAIRED resampling: one clip draw for both sides (codex r1 P2)
                boots.append(fid(me_v[ia], me_gt[ia]))
            # paired-bootstrap 2.5/97.5 percentiles: a RESAMPLING RANGE, not a coverage-calibrated interval -- FID is a biased
            # nonlinear estimator, so its percentile interval need not cover the population value (codex r5 P2); the real-vs-real
            # halves below give the estimator's resolution floor on this set
            rec["fid_vs_gt_paired_boot_range"] = [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]
            halves, cross = [], []
            for _ in range(a.fid_halves):
                pi = rng.permutation(n); h1, h2 = torch.as_tensor(pi[: n // 2]), torch.as_tensor(pi[n // 2:])
                halves.append(fid(me_gt[h1], me_gt[h2])); cross.append(fid(me_v[h1], me_gt[h2]))
            rec["fid_real_vs_real_halves"] = {"mean": float(np.mean(halves)), "std": float(np.std(halves))}
            rec["fid_gen_half_vs_real_half"] = {"mean": float(np.mean(cross)), "std": float(np.std(cross))}
        report["variants"][name] = rec
        r = rec["rprec"]
        print(f"[controls] {name:9s} R@1 {r['1']['mean']:.4f} [{r['1']['lo95']:.4f},{r['1']['hi95']:.4f}] R@3 {r['3']['mean']:.4f} | match {rec['matching_text_cos']:.3f} | FID {rec['fid_vs_gt']:.5f} | within-rig R@1 {rec['within_rig']['R1']} (chance {rec['within_rig']['chance_R1_mean']})", flush=True)
    if "capshuffle" in report["variants"]["gen"]:
        c = report["variants"]["gen"]["capshuffle"]; print(f"[controls] caption-shuffle R@1 {c['1']['mean']:.4f} (group-aware chance {report['variants']['gen']['capshuffle_chance_R1']:.4f}); FID gen-vs-GT paired-boot range {report['variants']['gen']['fid_vs_gt_paired_boot_range']}; real-vs-real halves {report['variants']['gen']['fid_real_vs_real_halves']}")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True); Path(a.out).write_text(json.dumps(report, indent=1))
    print(f"[controls] -> {a.out}")


if __name__ == "__main__":
    main()
