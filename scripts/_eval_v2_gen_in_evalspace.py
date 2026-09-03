"""Frozen-evaluator generation eval for the v2 in-context DiT (KTJD-17 PZ corpus).

Protocol (user 2026-08-29): FULL val (3,899 clips) + pool size 32, dataset-order pools --
identical chunking to _eval_evaluator_sanity.py, so text->GEN R@K is directly comparable
to that script's text->GT ceiling (0.975 for evaluator_ktjd16_pz_v1). Inference is the
deployment config: 20-step ODE, cfg_text=2, 1-frame rest demo.

Metrics: text->gen R@1/2/3 (group-aware, pool 32) | text->GT same-pool ceiling |
matching score (diagonal cos) | FID(gen, GT) in the 512-d evaluator space | gen<->GT cos.
Scores are PZ-only/16ch/T=240 -- NOT comparable to legacy 13ch evaluator numbers.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.incontext_pairs import InContextPairs, collate                     # noqa: E402
from src.data.ktjd17_incontext import Ktjd17Base, ktjd17_split_names             # noqa: E402
from src.data.ktjd17_t2m_eval_dataset import Ktjd17T2MEvalDataset, KEEP_CH, J_MAX  # noqa: E402
from src.data.anytop_t2m_eval_dataset import collate_fn as eval_collate          # noqa: E402
from src.models.graph_salad.batch import GraphMotionBatch                        # noqa: E402
from src.models.graph_salad.t2m_evaluator import AnyTopT2MEvaluator              # noqa: E402
from src.models.v2.dit_motion import InContextMotionDiT, sample                  # noqa: E402
from scripts._eval_evaluator_sanity import avg_over_pools                        # noqa: E402


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gen_ckpt", required=True, help="v2 DiT ckpt (best_model.pt).")
    ap.add_argument("--eval_ckpt", required=True, help="frozen evaluator ckpt.")
    ap.add_argument("--out", default=None, help="JSON report path.")
    ap.add_argument("--cfg_text", type=float, default=2.0)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--gen_batch", type=int, default=16)
    ap.add_argument("--encode_batch", type=int, default=64)
    ap.add_argument("--pool", type=int, default=32)
    # Exact multi-GPU split (2026-09-03): the 302M model needs ~7.3 h for the 3,899-clip val on one
    # H200. Generation is embarrassingly parallel over rigs and is reseeded per batch, so the
    # samples of a rig do not depend on which process/shard generates it. A shard generates its
    # rigs and saves them (--save_gen); --merge loads all shards, verifies they are one protocol
    # run (same gen ckpt sha, seed, steps, cfg, batch, shard count, every shard index exactly once,
    # every protocol clip exactly once) and scores them exactly as the single-process path would.
    ap.add_argument("--nshards", type=int, default=1, help="split the val rigs round-robin into N shards")
    ap.add_argument("--shard", type=int, default=0, help="which shard THIS process generates (0-based)")
    ap.add_argument("--save_gen", default=None,
                    help="write this shard's generated samples + metadata to an .npz and exit (no scoring)")
    ap.add_argument("--merge", default=None,
                    help="comma-separated shard .npz files to score instead of generating here")
    a = ap.parse_args()
    # The protocol is FROZEN (user 2026-08-29: full val + pool 32; deployment inference
    # 20-step/cfg2). Changing any of these is a protocol change -> edit this pin on purpose.
    if (a.steps, a.cfg_text, a.pool) != (20, 2.0, 32):
        raise SystemExit(f"[refuse] protocol pin is steps=20/cfg_text=2.0/pool=32; "
                         f"got {a.steps}/{a.cfg_text}/{a.pool}")
    if a.nshards < 1 or not 0 <= a.shard < a.nshards:
        raise SystemExit(f"[refuse] --shard {a.shard} must lie in [0, --nshards {a.nshards})")
    if a.merge and (a.save_gen or a.nshards > 1):
        raise SystemExit("[refuse] --merge scores existing shard files; it cannot be combined with --save_gen/--nshards")
    if a.nshards > 1 and not a.save_gen:
        raise SystemExit("[refuse] a shard generates a partial val set and can only be scored after --merge; "
                         "pass --save_gen to store it")
    return a


PROTOCOL_VAL_N = 3899   # frozen val size; a different corpus cut must re-pin on purpose


def hash_load(path):
    """sha256 + torch.load from ONE read of the file, so the reported hash is the hash
    of the weights actually evaluated (best_model.pt gets atomically replaced mid-training;
    codex fix3 observed epoch 344 -> 354 during one review)."""
    import hashlib
    import io
    raw = Path(path).read_bytes()
    sha = hashlib.sha256(raw).hexdigest()
    return torch.load(io.BytesIO(raw), map_location="cpu", weights_only=False), sha


def load_gen_model(ck, dev):
    ca = ck["args"]
    if str(ca.get("corpus")) != "ktjd17":
        raise SystemExit(f"[refuse] gen ckpt corpus={ca.get('corpus')!r}; this eval is KTJD-17 only")
    if bool(ca.get("two_stage", False)):
        raise SystemExit("[refuse] two_stage ckpts are not wired here")
    model = InContextMotionDiT(
        in_ch=17, dim=ca["dim"], depth=ca["depth"], n_heads=ca["heads"],
        d_text=4096, d_joint_sem=4096,
        use_struct_feats=bool(ca.get("struct_feats", False)),
        use_dir_bias=bool(ca.get("dir_bias", False)),
        qk_norm=bool(ca.get("qk_norm", False)),
        use_ref_text=bool(ca.get("ref_text", False))).to(dev)
    model.load_state_dict(ck["model"])
    model.eval()
    return model, ca


def load_evaluator(path, dev):
    ck, sha = hash_load(path)
    m = ck.get("args", {})
    g = (lambda k, d: m.get(k, d)) if isinstance(m, dict) else (lambda k, d: getattr(m, k, d))
    if int(g("motion_feat_dim", 13)) != 16:
        raise SystemExit(f"[refuse] evaluator motion_feat_dim={g('motion_feat_dim', 13)}; "
                         "this eval feeds 16ch KTJD tensors")
    core = AnyTopT2MEvaluator(
        coemb_dim=g("coemb_dim", 512), text_tower=g("text_tower", "distilbert"),
        distilbert_path=g("distilbert_path", "checkpoints/text_encoders/distilbert-base-uncased"),
        text_max_length=g("text_max_length", 64),
        n_heads=g("n_heads", 8), d_ff=g("d_ff", 2048),
        n_graph_layers=g("n_graph_layers", 6), n_temporal_layers=g("n_temporal_layers", 4),
        motion_feat_dim=16, dropout=g("dropout", 0.1),
        learnable_temperature=not g("fixed_temperature", False), temperature=g("temperature", 0.07),
        strict_frame_masking=g("strict_frame_masking", False))
    core.load_state_dict(ck["model"])
    core.to(dev).eval()
    print(f"[gen-eval] evaluator {path} (epoch={ck.get('epoch', '?')} "
          f"val={ck.get('val', {}).get('r1', '?')} sha256={sha[:16]})", flush=True)
    return core, sha


@torch.no_grad()
def generate_all(model, ca, base, names, dev, a):
    """One generated [T,J,17] (normalized space) per val target, keyed by motion_id."""
    PK = dict(demo_rest=bool(ca.get("demo_rest", False)),
              emit_ref_text=bool(ca.get("ref_text", False)),
              demo_frames=int(ca.get("demo_frames", 1)),
              target_frames=int(ca["target_frames"]),
              emit_graph_v2=bool(ca.get("struct_feats", False)) or bool(ca.get("dir_bias", False)))
    if PK["demo_rest"] and PK["demo_frames"] != 1:
        raise SystemExit("[refuse] demo_rest ckpt with demo_frames != 1")
    ds = InContextPairs(base, names["val"], names["train"], object_types=None,
                        balance_skeletons=False, seed=a.seed, **PK)
    anc_mode = str(ca.get("anchor", "none"))
    rest_lut = {}
    if anc_mode == "rest":
        from scripts.train_v2_incontext import ktjd_anchor  # noqa: F401 (used below)
    plan = generation_plan(ds, base, a)
    rigs_mine = plan["shard_rigs"][a.shard]
    n_mine = len(plan["shard_clips"][a.shard])
    if a.nshards > 1:
        print(f"[gen-eval] shard {a.shard}/{a.nshards}: {len(rigs_mine)} of {len(plan['rigs'])} rigs, "
              f"{n_mine} of {len(ds.index)} val clips (plan {plan['plan_sha256'][:12]})", flush=True)
    by_rig = plan["by_rig"]
    gen_by_clip: dict[str, np.ndarray] = {}
    df = PK["demo_frames"]
    n_done = 0
    for rig in rigs_mine:
        pos = by_rig[rig]
        cvj = torch.from_numpy(base.static_masks(rig)["channel_valid"]).to(dev)
        for s in range(0, len(pos), a.gen_batch):
            chunk = pos[s:s + a.gen_batch]
            items = []
            for p in chunk:
                ds._wrng_key = None          # per-item stream reset (render-script convention)
                items.append(ds[p])
                if str(items[-1]["motion_id"]) != plan["clip_of_pos"][p]:
                    raise SystemExit(f"[refuse] served item {items[-1]['motion_id']} != plan {plan['clip_of_pos'][p]} "
                                     f"at position {p}: the generation plan no longer matches the dataset")
            b = {k: (v.to(dev) if torch.is_tensor(v) else v)
                 for k, v in collate(items).items()}
            x_in = b["x"][..., :17].contiguous()
            g2kw = {k: b[k] for k in ("struct_feats", "updown") if k in b}
            cv = torch.zeros(x_in.shape[0], x_in.shape[2], 17, dtype=torch.bool, device=dev)
            cv[:, :cvj.shape[0]] = cvj
            g2kw["channel_valid"] = cv
            g2kw["heading_valid"] = b["x"][:, :, 0, 17] > 0.5
            if "demo_text" in b:
                g2kw["demo_text"] = b["demo_text"]
            if anc_mode != "none":
                from scripts.train_v2_incontext import ktjd_anchor
                if anc_mode == "rest" and rig not in rest_lut:
                    rest_lut[rig] = torch.from_numpy(base.rest_anchor_frame(rig))
                g2kw["anchor"] = ktjd_anchor(b, x_in, anc_mode,
                                             rest_lut if anc_mode == "rest" else None, df)
            torch.manual_seed(a.seed)
            gen = sample(model, x_in, b["is_target"], a.steps, cfg_text=a.cfg_text,
                         demo_frames=df,
                         joint_bias=b["joint_bias"], frame_valid=b["frame_valid"],
                         joint_valid=b["joint_valid"], text=b["text"],
                         joint_sem=b["joint_sem"], **g2kw)
            gen = gen.float().cpu().numpy()   # [B, df+Tt, J, 17]
            for bi, p in enumerate(chunk):
                mid = str(items[bi]["motion_id"])
                if mid in gen_by_clip:
                    raise SystemExit(f"[refuse] duplicate generated motion_id {mid}")
                gen_by_clip[mid] = gen[bi, df:].astype(np.float32)   # target frames only
                # fp32 kept on purpose: fp16 caching quantizes the samples that feed the
                # evaluator/FID (codex fix2); ~3GB RAM at 3,899 clips, fine on these nodes
            n_done += len(chunk)
        if n_done and (len(gen_by_clip) % 512 < a.gen_batch):
            print(f"[gen-eval] generated {n_done}/{n_mine}", flush=True)
    if len(gen_by_clip) != n_mine:
        raise SystemExit(f"[refuse] generated {len(gen_by_clip)} != this shard's targets {n_mine}")
    if a.nshards == 1 and len(gen_by_clip) != PROTOCOL_VAL_N:
        raise SystemExit(f"[refuse] protocol val is {PROTOCOL_VAL_N}, generated {len(gen_by_clip)}")
    print(f"[gen-eval] generated {len(gen_by_clip)} clips", flush=True)
    return gen_by_clip


def make_pairs(ds_args, base, names, a):
    ca = ds_args
    return InContextPairs(base, names["val"], names["train"], object_types=None, balance_skeletons=False, seed=a.seed,
                          demo_rest=bool(ca.get("demo_rest", False)), emit_ref_text=bool(ca.get("ref_text", False)),
                          demo_frames=int(ca.get("demo_frames", 1)), target_frames=int(ca["target_frames"]),
                          emit_graph_v2=bool(ca.get("struct_feats", False)) or bool(ca.get("dir_bias", False)))


def generation_plan(ds, base, a):
    """THE ordered plan both generation and --merge derive from the dataset: sorted rigs, each rig's val
    positions in dataset order, cut into gen_batch chunks (the per-batch reseed makes every chunk's noise a
    function of this plan only), round-robin shard assignment over the sorted rigs. Hashed as a whole and
    per shard; a shard file must reproduce its expected clip set exactly (codex 2026-09-03 P0-3)."""
    import hashlib
    by_rig: dict[str, list[int]] = {}
    for i, (rig, _) in enumerate(ds.index):
        by_rig.setdefault(rig, []).append(i)
    rigs = sorted(by_rig)
    clip_of_pos = {p: str(base._rows[ds.index[p][1]]["clip_id"]) for pos in by_rig.values() for p in pos}
    lines = []
    for rig in rigs:
        pos = by_rig[rig]
        for s in range(0, len(pos), a.gen_batch):
            lines.append(rig + ":" + ",".join(clip_of_pos[p] for p in pos[s:s + a.gen_batch]))
    shard_rigs = {k: [r for i, r in enumerate(rigs) if i % a.nshards == k] for k in range(a.nshards)}
    shard_clips = {k: [clip_of_pos[p] for r in shard_rigs[k] for p in by_rig[r]] for k in range(a.nshards)}
    return {"rigs": rigs, "by_rig": by_rig, "clip_of_pos": clip_of_pos,
            "plan_sha256": hashlib.sha256("\n".join(lines).encode()).hexdigest(),
            "shard_rigs": shard_rigs, "shard_clips": shard_clips,
            "shard_clips_sha256": {k: hashlib.sha256("\n".join(v).encode()).hexdigest() for k, v in shard_clips.items()}}


def source_fingerprint(anchor="none"):
    """sha256 over the code that turns (ckpt, data, seed) into samples: this script, the sampler/model, the
    pair dataset and the corpus adapter -- and the trainer module whenever the checkpoint's anchor mode makes
    generate_all() call its ktjd_anchor() (codex 2026-09-03 r2; anchor=none checkpoints never touch it)."""
    import hashlib
    repo = Path(__file__).resolve().parents[1]
    files = ["scripts/_eval_v2_gen_in_evalspace.py", "src/models/v2/dit_motion.py",
             "src/data/incontext_pairs.py", "src/data/ktjd17_incontext.py"]
    if str(anchor) != "none":
        files.append("scripts/train_v2_incontext.py")
    h = hashlib.sha256()
    for rel in files:
        h.update(rel.encode()); h.update((repo / rel).read_bytes())
    return h.hexdigest()


def runtime_fingerprint():
    """the sampling runtime a shard set must share (GPU MODEL is recorded but not required equal)."""
    return {"torch": torch.__version__, "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
            "allow_tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
            "allow_tf32_cudnn": torch.backends.cudnn.allow_tf32,
            "float32_matmul_precision": torch.get_float32_matmul_precision()}


def shard_meta(a, ca, gen_sha, base, plan):
    """What a shard file must agree on with every other shard of the same protocol run."""
    import hashlib
    prov = hashlib.sha256(json.dumps({**base.provenance, "exclusion": base.provenance_exclusion},
                                     sort_keys=True, default=str).encode()).hexdigest()
    return {"gen_ckpt_sha256": gen_sha, "gen_ckpt": a.gen_ckpt, "seed": a.seed, "steps": a.steps,
            "cfg_text": a.cfg_text, "gen_batch": a.gen_batch, "nshards": a.nshards, "shard": a.shard,
            "ktjd_root": str(ca["ktjd_root"]), "exclude_clips": str(ca.get("exclude_clips") or ""),
            "base_provenance_sha256": prov, "plan_sha256": plan["plan_sha256"],
            "shard_clips_sha256": plan["shard_clips_sha256"][a.shard], "shard_n_clips": len(plan["shard_clips"][a.shard]),
            "n_val_rigs": len(plan["rigs"]), "protocol_val_n": PROTOCOL_VAL_N,
            "source_fingerprint": source_fingerprint(ca.get("anchor", "none")), "runtime": runtime_fingerprint(),
            "rank_env": os.environ.get("RANK"),
            "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"}


def save_shard(path, gen_by_clip, meta, plan):
    want = plan["shard_clips"][meta["shard"]]
    if sorted(gen_by_clip) != sorted(want):
        raise SystemExit(f"[refuse] generated clip set differs from the plan for shard {meta['shard']}")
    payload = {f"clip__{mid}": np.ascontiguousarray(arr, dtype=np.float32) for mid, arr in gen_by_clip.items()}
    np.savez(path, __meta=json.dumps(meta), **payload)
    print(f"[gen-eval] shard {meta['shard']}/{meta['nshards']} -> {path} ({len(gen_by_clip)} clips)", flush=True)


def _strict_int(x):
    """an int and not a bool (True == 1 in Python, but a shard index of True is malformed metadata)."""
    return type(x) is int


def merge_shards(paths, want_meta, plan):
    """Load shard files, refuse anything that is not exactly one complete protocol run generated under
    THIS plan, source and runtime."""
    import hashlib
    gen_by_clip: dict[str, np.ndarray] = {}
    seen_shards, nshards, loaded, plan_sets = set(), None, [], None
    invariant = ("gen_ckpt_sha256", "seed", "steps", "cfg_text", "gen_batch", "ktjd_root", "exclude_clips",
                 "base_provenance_sha256", "plan_sha256", "n_val_rigs", "protocol_val_n", "source_fingerprint")
    for p in paths:
        raw = Path(p).read_bytes()
        fsha = hashlib.sha256(raw).hexdigest()
        import io
        with np.load(io.BytesIO(raw), allow_pickle=False) as z:
            if "__meta" not in z.files:
                raise SystemExit(f"[refuse] shard {p} has no __meta")
            try:
                meta = json.loads(str(z["__meta"]))
            except ValueError as e:
                raise SystemExit(f"[refuse] shard {p}: __meta is not valid JSON ({e})")
            if not isinstance(meta, dict):
                raise SystemExit(f"[refuse] shard {p}: __meta is a {type(meta).__name__}, not a JSON object")
            # exact metadata means exact TYPE and value: "42" is not the seed 42, 1.9 is not nshards 1, True is not 1
            # (codex 2026-09-03 r3); JSON round-trips int / float / str faithfully, so the writer's types are what we expect
            for k in invariant:
                if type(meta.get(k)) is not type(want_meta.get(k)) or meta.get(k) != want_meta.get(k):
                    raise SystemExit(f"[refuse] shard {p}: {k}={meta.get(k)!r:.40} differs from this run's "
                                     f"{want_meta.get(k)!r:.40}")
            if not isinstance(meta.get("runtime"), dict) or \
                    json.dumps(meta["runtime"], sort_keys=True) != json.dumps(want_meta["runtime"], sort_keys=True):
                raise SystemExit(f"[refuse] shard {p}: sampling runtime {meta.get('runtime')} != this run's {want_meta['runtime']}")
            if meta.get("rank_env") not in (None, "0"):
                raise SystemExit(f"[refuse] shard {p} was generated under RANK={meta.get('rank_env')}")
            if not _strict_int(meta.get("nshards")) or meta["nshards"] < 1 or not _strict_int(meta.get("shard")):
                raise SystemExit(f"[refuse] shard {p}: nshards={meta.get('nshards')!r} / shard={meta.get('shard')!r} "
                                 f"must be positive / non-negative integers")
            nshards = nshards or meta["nshards"]
            if meta["nshards"] != nshards:
                raise SystemExit(f"[refuse] shard {p} declares nshards={meta['nshards']}, others {nshards}")
            if nshards != want_meta["nshards"] and want_meta["nshards"] != 1:
                raise SystemExit(f"[refuse] shard count {nshards} != requested plan {want_meta['nshards']}")
            k_ = meta["shard"]
            if not 0 <= k_ < nshards:
                raise SystemExit(f"[refuse] shard {p} declares index {k_} outside [0, {nshards})")
            if k_ in seen_shards:
                raise SystemExit(f"[refuse] shard index {k_} appears twice ({p})")
            seen_shards.add(k_)
            keys = [k for k in z.files if k != "__meta"]
            bad = [k for k in keys if not k.startswith("clip__")]
            if bad:
                raise SystemExit(f"[refuse] shard {p} has unexpected keys {bad[:4]}")
            got = sorted(k[len("clip__"):] for k in keys)
            # the plan's assignment for this index, re-derived here, checked against THESE bytes (no re-open:
            # codex 2026-09-03 r2 P1); the meta's own count / sha must agree with it too
            plan_sets = plan_sets if plan_sets is not None else generation_plan_shards(plan, nshards)
            want_clips = plan_sets[k_]
            want_sha = hashlib.sha256("\n".join(want_clips).encode()).hexdigest()
            if got != sorted(want_clips):
                raise SystemExit(f"[refuse] shard {p} (index {k_}) holds {len(got)} clips that are not the plan's "
                                 f"assignment of {len(want_clips)} clips")
            if not _strict_int(meta.get("shard_n_clips")) or meta["shard_n_clips"] != len(want_clips) \
                    or not isinstance(meta.get("shard_clips_sha256"), str) or meta["shard_clips_sha256"] != want_sha:
                raise SystemExit(f"[refuse] shard {p} meta shard_n_clips={meta.get('shard_n_clips')} / "
                                 f"shard_clips_sha256={str(meta.get('shard_clips_sha256'))[:12]} != plan's "
                                 f"{len(want_clips)} / {want_sha[:12]}")
            for k in keys:
                arr = z[k]
                if arr.dtype != np.float32 or arr.ndim != 3 or arr.shape[-1] != 17 or not np.isfinite(arr).all():
                    raise SystemExit(f"[refuse] shard {p} sample {k} is not a finite float32 [T,J,17] array "
                                     f"(dtype {arr.dtype}, shape {arr.shape})")
                mid = k[len("clip__"):]
                if mid in gen_by_clip:
                    raise SystemExit(f"[refuse] clip {mid} appears in more than one shard ({p})")
                gen_by_clip[mid] = np.asarray(arr, dtype=np.float32)
            loaded.append({"path": p, "sha256": fsha, "shard": k_, "n_clips": len(got),
                           "device": meta.get("device"), "runtime": meta.get("runtime")})
            print(f"[gen-eval] merged shard {k_}/{nshards} from {p} ({len(got)} clips, {meta.get('device')}, sha {fsha[:12]})", flush=True)
    if seen_shards != set(range(nshards or 0)):
        raise SystemExit(f"[refuse] shards present {sorted(seen_shards)} != required {list(range(nshards or 0))}")
    if len(gen_by_clip) != PROTOCOL_VAL_N:
        raise SystemExit(f"[refuse] merged shards hold {len(gen_by_clip)} clips, protocol val is {PROTOCOL_VAL_N}")
    return gen_by_clip, nshards, loaded


def generation_plan_shards(plan, nshards):
    """re-derive the round-robin assignment for an arbitrary shard count from the plan's sorted rigs."""
    rigs = plan["rigs"]
    return {k: [plan["clip_of_pos"][p] for i, r in enumerate(rigs) if i % nshards == k for p in plan["by_rig"][r]]
            for k in range(nshards)}


def gen_to_anytop_x(gen_tjc: np.ndarray, J: int, T: int, Tt: int) -> torch.Tensor:
    """[T_target, J_serve, 17] normalized -> evaluator anytop_x [J_MAX, 16, Tt]."""
    x = gen_tjc[:T, :J, :].transpose(1, 2, 0)                  # [J, 17, T]
    x16 = np.zeros((J_MAX, 16, Tt), dtype=np.float32)
    x16[:J, :, :T] = x[:, KEEP_CH, :]
    return torch.from_numpy(x16)


@torch.no_grad()
def encode_split(core, eval_ds, gen_by_clip, dev, a):
    te, me_gt, me_gen = [], [], []
    meta = {"motion_id": [], "source_motion_id": [], "caption_text": []}
    missing = []
    buf = []

    def flush():
        if not buf:
            return
        coll = eval_collate([it for it, _ in buf])
        batch = GraphMotionBatch.from_collate_dict(
            {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in coll.items()})
        te.append(core.encode_text(coll["caption_text"]).float().cpu())
        me_gt.append(core.encode_motion(batch).float().cpu())
        gx = torch.stack([gx_ for _, gx_ in buf]).to(dev)
        import dataclasses
        me_gen.append(core.encode_motion(
            dataclasses.replace(batch, anytop_x=gx)).float().cpu())
        buf.clear()

    for i in range(len(eval_ds)):
        it = eval_ds[i]
        mid = str(it["motion_id"])
        if mid not in gen_by_clip:
            missing.append(mid)
            continue
        J = int(it["num_joints"]); T = int(it["num_frames"])
        buf.append((it, gen_to_anytop_x(gen_by_clip[mid], J, T, eval_ds.Tt)))
        meta["motion_id"].append(mid)
        meta["source_motion_id"].append(str(it["source_motion_id"]))
        meta["caption_text"].append(str(it["caption_text"]))
        if len(buf) >= a.encode_batch:
            flush()
        if (i + 1) % 1024 == 0:
            print(f"[gen-eval] encoded {i+1}/{len(eval_ds)}", flush=True)
    flush()
    if missing:
        raise SystemExit(f"[refuse] {len(missing)} val clips have no generation "
                         f"(e.g. {missing[:5]}); generation must cover the full protocol set")
    return (torch.nn.functional.normalize(torch.cat(te), dim=-1),
            torch.nn.functional.normalize(torch.cat(me_gt), dim=-1),
            torch.nn.functional.normalize(torch.cat(me_gen), dim=-1), meta)


def fid(a_emb: torch.Tensor, b_emb: torch.Tensor) -> float:
    from scipy import linalg
    x, y = a_emb.double().numpy(), b_emb.double().numpy()
    mu1, mu2 = x.mean(0), y.mean(0)
    s1 = np.cov(x, rowvar=False)
    s2 = np.cov(y, rowvar=False)
    covmean, _ = linalg.sqrtm(s1 @ s2, disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return float(((mu1 - mu2) ** 2).sum() + np.trace(s1 + s2 - 2.0 * covmean))


def main():
    a = parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck, gen_sha = hash_load(a.gen_ckpt)
    ca = ck["args"]
    model, ca = load_gen_model(ck, dev)
    print(f"[gen-eval] gen ckpt {a.gen_ckpt} (epoch {ck.get('epoch', -1)} "
          f"sha256={gen_sha[:16]}) cfg_text={a.cfg_text} steps={a.steps}", flush=True)

    root = ca["ktjd_root"]
    excl = ca.get("exclude_clips") or None
    base = Ktjd17Base(root, caption_emb_cache=ca["caption_cache"],
                      joint_semantics=ca["joint_sem"], texts_json=ca["texts_json"],
                      percell_stats=ca.get("ktjd_percell_stats",
                                           "data/ktjd17_percell_stats_v1.npz"),
                      exclude_clips=excl)
    pins_ck = ck.get("ktjd_pins") or {}
    live = {**base.provenance, "exclusion": base.provenance_exclusion}
    drift = sorted(k for k, v in pins_ck.items() if k in live and live[k] != v)
    if drift:
        raise SystemExit(f"[refuse] data drift vs gen-ckpt pins: {drift}")
    names = ktjd17_split_names(root, exclude=excl)

    # process-per-GPU sharding assumes the rank-0 RNG stream of InContextPairs ([seed, rank]); a
    # rank-setting launcher would silently pick other demos per shard (codex 2026-09-03 P0-2)
    if os.environ.get("RANK") not in (None, "0"):
        raise SystemExit(f"[refuse] RANK={os.environ.get('RANK')} is set; run one plain process per GPU (RANK unset or 0)")
    plan = generation_plan(make_pairs(ca, base, names, a), base, a)
    meta = shard_meta(a, ca, gen_sha, base, plan)
    gen_mode = {"mode": "single", "plan_sha256": plan["plan_sha256"], "source_fingerprint": meta["source_fingerprint"],
                "runtime": meta["runtime"], "device": meta["device"]}
    if a.merge:
        paths = [p.strip() for p in a.merge.split(",") if p.strip()]
        gen_by_clip, nsh, loaded = merge_shards(paths, meta, plan)
        gen_mode = {"mode": "sharded_merge", "nshards": nsh, "plan_sha256": plan["plan_sha256"],
                    "source_fingerprint": meta["source_fingerprint"], "runtime": meta["runtime"], "shards": loaded}
    else:
        gen_by_clip = generate_all(model, ca, base, names, dev, a)
        if a.save_gen:
            save_shard(a.save_gen, gen_by_clip, meta, plan)
            return                                  # scoring of saved samples happens at --merge only
    del model
    torch.cuda.empty_cache()

    core, eval_sha = load_evaluator(a.eval_ckpt, dev)
    # eval-side base: SAME root/percell/exclude as the generator ckpt, so both towers see
    # one normalization space and one val set.
    base_eval = Ktjd17Base(root, caption_emb_cache=ca["caption_cache"],
                           joint_semantics=ca["joint_sem"], texts_json=ca["texts_json"],
                           percell_stats=ca.get("ktjd_percell_stats",
                                                "data/ktjd17_percell_stats_v1.npz"),
                           exclude_clips=excl)
    eval_ds = Ktjd17T2MEvalDataset(base_eval, "val", max_frames=240, exclude=excl)
    if len(eval_ds) != PROTOCOL_VAL_N:
        raise SystemExit(f"[refuse] eval val has {len(eval_ds)} clips, protocol pins "
                         f"{PROTOCOL_VAL_N}")
    te, me_gt, me_gen, meta = encode_split(core, eval_ds, gen_by_clip, dev, a)

    gpool = torch.Generator().manual_seed(a.seed)
    n = te.shape[0]
    report = {"protocol": {"val_n": n, "pool": a.pool, "cfg_text": a.cfg_text,
                           "steps": a.steps, "seed": a.seed, "gen_batch": a.gen_batch,
                           "generation": gen_mode,
                           "gen_ckpt": a.gen_ckpt, "gen_epoch": int(ck.get("epoch", -1)),
                           "gen_ckpt_sha256": gen_sha,
                           "eval_ckpt": a.eval_ckpt, "eval_ckpt_sha256": eval_sha,
                           "note": "PZ-only/16ch/T=240; pools chunk dataset order like "
                                   "_eval_evaluator_sanity.py"}}
    for tag, me in (("text_to_gen", me_gen), ("text_to_gt_ceiling", me_gt)):
        rr, npool = avg_over_pools(te, me, meta, a.pool, masked=True, shuffled=False, gen=gpool)
        used = npool * a.pool
        report[tag] = {"rprec": rr, "n_pools": npool, "n_used": used}
        print(f"[gen-eval] {tag:<19} R@1={rr[1]:.3f} R@2={rr[2]:.3f} R@3={rr[3]:.3f} "
              f"({npool} pools of {a.pool}, used {used}/{n})", flush=True)
    report["matching"] = {
        "text_gen_cos": float((te * me_gen).sum(-1).mean()),
        "text_gt_cos": float((te * me_gt).sum(-1).mean()),
        "gen_gt_cos": float((me_gen * me_gt).sum(-1).mean())}
    report["fid_gen_vs_gt"] = fid(me_gen, me_gt)
    print(f"[gen-eval] matching text-gen={report['matching']['text_gen_cos']:.3f} "
          f"text-GT={report['matching']['text_gt_cos']:.3f} "
          f"gen-GT={report['matching']['gen_gt_cos']:.3f}", flush=True)
    print(f"[gen-eval] FID(gen, GT) = {report['fid_gen_vs_gt']:.4f}", flush=True)
    if a.out:
        Path(a.out).write_text(json.dumps(report, indent=2))
        print(f"[gen-eval] report -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
