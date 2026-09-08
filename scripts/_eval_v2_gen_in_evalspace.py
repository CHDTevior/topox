"""Frozen-evaluator generation eval for the v2 in-context DiT (KTJD-17 PZ corpus).

Protocol (user 2026-08-29, pool moved 32 -> 64 on 2026-09-07): FULL val (3,899 clips) + pool size 64, dataset-order pools --
identical chunking to _eval_evaluator_sanity.py, so text->GEN R@K is directly comparable
to that script's text->GT ceiling (0.975 at pool 32 / 0.964 at pool 64 for evaluator_ktjd16_pz_v1; that script still
defaults to pool 32, pass --pool 64 to compare). Inference is the
deployment config: 20-step ODE, cfg_text=2, 1-frame rest demo.

Metrics: text->gen R@1/2/3 (group-aware, pool 64) | text->GT same-pool ceiling |
matching score (diagonal cos) | FID(gen, GT) in the 512-d evaluator space | gen<->GT cos.
Scores are PZ-only/16ch/T=240 -- NOT comparable to legacy 13ch evaluator numbers.
"""
from __future__ import annotations

import argparse
import hashlib
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
    ap.add_argument("--tf32", action="store_true",
                    help="enable TF32 tensor-core matmul and cuDNN TF32 (user 2026-09-07: fastest path, small numeric change accepted). "
                         "Without it the process keeps PyTorch's defaults (matmul TF32 off, cuDNN TF32 on). Both flags are recorded in "
                         "protocol.runtime and --merge refuses shards whose runtime differs, so --merge must be run with the same flag")
    ap.add_argument("--encode_batch", type=int, default=64)
    ap.add_argument("--pool", type=int, default=64)   # protocol pin (user 2026-09-07: 主表暂定用 64; was 32 until then)
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
    ap.add_argument("--protocol_variant", default=None, choices=[None, "steps", "pool"],
                    help="EXPLICIT departure from the frozen protocol: 'steps' allows --steps != 20 (a sampling-cost sweep). "
                         "Every report and shard from such a run is stamped protocol.variant='steps' and must never be "
                         "placed in a table with frozen-protocol numbers without saying so (user 2026-09-06: efficiency "
                         "is measured, not a paper objective).")
    ap.add_argument("--score_subset", default=None,
                    help="--merge only: score ONLY the val clips listed in this JSON (format pz-clean-val-subset-v1, "
                         "scripts/_build_pz_clean_val_subset.py). Generation must still cover the full protocol set; the "
                         "report records the subset file, its sha256 and the scored count (codex 2026-09-06 P0-1: "
                         "duplicate asset exports leak most val clips into train; this scores the demonstrably unexposed ones).")
    ap.add_argument("--merge", default=None,
                    help="comma-separated shard .npz files to score instead of generating here")
    a = ap.parse_args()
    if os.environ.get("NVIDIA_TF32_OVERRIDE") is not None:
        # the driver-level override changes TF32 behaviour underneath PyTorch and is invisible to the runtime record below
        # (codex 2026-09-07 r3 P3): a shard set generated under it could be merged with one that was not
        raise SystemExit("[refuse] NVIDIA_TF32_OVERRIDE is set; unset it -- the runtime record cannot see it")
    if a.tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    # The protocol is FROZEN (user 2026-08-29: full val; deployment inference 20-step/cfg2; pool 32 until 2026-09-07, then the
    # user moved the main-table pool to 64 after the pool sweep -- 32 is now reached through --protocol_variant pool like 16/128).
    # Changing any of these is a protocol change -> edit this pin on purpose.
    if a.cfg_text != 2.0 or (a.pool != 64 and a.protocol_variant != "pool") or (a.steps != 20 and a.protocol_variant != "steps"):
        raise SystemExit(f"[refuse] protocol pin is steps=20/cfg_text=2.0/pool=64; "
                         f"got {a.steps}/{a.cfg_text}/{a.pool} (pass --protocol_variant steps for a sampling-cost sweep, "
                         f"--protocol_variant pool for a pool-size sweep over saved frozen-protocol samples)")
    if a.protocol_variant == "steps" and a.steps == 20:
        raise SystemExit("[refuse] --protocol_variant steps requires --steps != 20; the frozen protocol needs no variant flag")
    if a.protocol_variant == "pool":
        # A pool-size sweep is a SCORING-time change (user 2026-09-06: choose the pool on the large test set): the samples
        # stay the frozen-protocol ones, so it is only allowed on saved shards, alone (never stacked on a steps variant),
        # with a pool that differs from the pin and fits the protocol set.
        if a.pool == 64:
            raise SystemExit("[refuse] --protocol_variant pool requires --pool != 64; the frozen protocol needs no variant flag")
        if a.pool < 2 or a.pool > PROTOCOL_VAL_N:
            raise SystemExit(f"[refuse] pool={a.pool}: a retrieval pool needs 2..{PROTOCOL_VAL_N} candidates")
        if a.steps != 20:
            raise SystemExit("[refuse] one protocol variant at a time: a pool-size sweep scores frozen-protocol (20-step) samples")
        if not a.merge:
            raise SystemExit("[refuse] --protocol_variant pool re-scores saved frozen-protocol shards (--merge); it never generates")
    if a.seed != 42:
        raise SystemExit(f"[refuse] protocol pin is seed=42 (generation noise and pool shuffling); got {a.seed}. No variant covers "
                         "the seed: a different seed is a different sample set and cannot sit next to frozen-protocol numbers")
    if a.steps < 1:
        raise SystemExit(f"[refuse] steps={a.steps}: sample() with a non-positive step count returns the base noise without "
                         "calling the model (codex 2026-09-06 P2)")
    if a.nshards < 1 or not 0 <= a.shard < a.nshards:
        raise SystemExit(f"[refuse] --shard {a.shard} must lie in [0, --nshards {a.nshards})")
    if a.merge and (a.save_gen or a.nshards > 1):
        raise SystemExit("[refuse] --merge scores existing shard files; it cannot be combined with --save_gen/--nshards")
    if a.score_subset and not a.merge:
        raise SystemExit("[refuse] --score_subset scores saved shards; combine it with --merge")
    if a.nshards > 1 and not a.save_gen:
        raise SystemExit("[refuse] a shard generates a partial val set and can only be scored after --merge; "
                         "pass --save_gen to store it")
    return a


PROTOCOL_VAL_N = 3899   # frozen val size; a different corpus cut must re-pin on purpose

# Generation-code fingerprints of ALREADY SAVED shard sets that this script may still score. Every entry must be
# audited: the diff between that state and the current files must not change what generate_all / the sampler / the
# datasets PRODUCE for the checkpoints named -- scoring-only edits (argument parsing, evaluator encoding, report), or
# dataset edits whose default-off path is proven byte-identical (a served-item snapshot before/after, cited in the
# entry). Merge records which fingerprint a shard set carries so a report always names the generation code it came from
# (codex 2026-09-06 P1-1; exception wording codex 2026-09-07 r2 P3).
LEGACY_SOURCE_FINGERPRINTS = {
    "b09b82e020f9088ab24b6fa5093516c0dc42fe23343231ba2a465198919951fc":
        "state of the 2026-09-08 representation-view hook, before the rest-normalisation branch of src/data/ktjd17_incontext.py "
        "(a new normalization='rest' code path; percell/scale_only checkpoints serve byte-identical items) and before the "
        "geodesic-bias switch. Shards: the AnyTop-13 arm's ep25/50/75/100 passes (runs/_final_geneval/p36a13).",
    "8ef9d1a0e7e41fb10ed281214e0a3e3e8d9c78231edb76ebe7875908529eff22":
        "state before the 2026-09-08 geodesic-bias switch (InContextMotionDiT(use_geo_bias=True) default, a fail-loud buffer only "
        "when it is False, and this script passing the checkpoint's geo_bias flag): sampling is byte-identical for every checkpoint "
        "trained without --no_geo_bias. Shards: any pass generated between the rest-normalisation commit facfe28 and this edit.",
    "305059f48da8f7da8f7e71d2e6ad610e5308de342849145f6493c93520874a2d":
        "state with the --tf32 flag and the two 2026-09-07 legacy entries, before the representation-view hook of 2026-09-08 (which only "
        "adds conversion of samples from an AnyTop-13 view and payload-shape checks; generate_all / sampler / datasets untouched for "
        "KTJD-17 checkpoints). Shards: r12 (303M) ep200 TF32 (runs/_final_geneval/r12_ep200_tf32, generated 2026-09-07 21:05-22:32Z).",
    "c202792417bc13a958818dd5a394034af879d9c76ba07e2ea2bfb0c707ef9c6d":
        "state after the 2026-09-07 pool-pin move and before the 2026-09-07 skeleton-augmentation edit of src/data/incontext_pairs.py "
        "and src/data/ktjd17_incontext.py (InContextPairs(augment=None) default, per-sample channel_valid only when augmenting, new "
        "Ktjd17Base.skeleton() accessor, rest-demo clamp on a private copy): with augmentation off the served items and collated "
        "batches are byte-identical to this state (12-item / 3-batch snapshot runs/_aug_dev/baseline_items_pre.npz vs post, 93 fields); "
        "generate_all / the sampler / dit_motion.py untouched. Shards: p36so ep100 (generated 2026-09-07 17:22Z under this state).",
    "de4ac413950b9f8a6bbc835daa83e449735bae34ed1600351bf7c2230fff8a52":
        "state after the 2026-09-07 skeleton-augmentation edit (first version) and before its codex fixes of the same evening "
        "(incontext_pairs.py: the rest-demo clamp now runs on a private copy instead of in place on the cached frame -- same served "
        "values; this script: the legacy entries above and this one): generate_all / the sampler / dit_motion.py untouched. Shards: "
        "r12 (303M) ep200, generated from 2026-09-07 18:43Z under this state.",
    "52444b7a8f7217f7772be315ef74f8dd239758470d6366e7366662adfc59157b":
        "state after the 2026-09-06 representation-ablation edit and before the 2026-09-07 pool-pin move (32 -> 64): identical "
        "generate_all/sample/dataset code; the pin edit touched parse_args (default pool, pin check, pool-variant check) and nothing "
        "that produces samples. Shards: p36_ctrl ep050/075/100 and the p36so smoke conversion test, generated under this state.",
    "b341caa660df3e4d3d61a5fda561ad81d658f8a664cb3f2b1505a4afb9f3201c":
        "state after the 2026-09-06 --protocol_variant pool edit and before the representation-ablation edit of the same day "
        "(rep_norm plumbing in the view, the trainer, the calibration script, this script and the renderer): identical "
        "generate_all/sample/dataset code for percell checkpoints. Shards: poolsweep re-scores, 100M ep200 if generated before the edit.",
    "3ca49a80114872ffbc989d8e31a0bd74a93c112e8aea07bed80afd928c6e8d6a":
        "state after the 2026-09-06 --protocol_variant steps edit and before the --protocol_variant pool edit of the same day: "
        "identical generate_all/sample/dataset code; the pool edit touched parse_args (pin + pool variant), merge_shards (invariant remap "
        "for the scoring-only variant) and nothing that produces samples. Shards: p36_s10/p36_s5/p303_s10/p303_s5 variants, ablation_noacc.",
    "646c71e8d56603274f6e52d989094ef91ab72862307f0bd55d99c4d3d3102698":
        "state before the 2026-09-06 --score_subset edit: identical generate_all/sample/dataset code; only "
        "parse_args, encode_split(keep) and the report changed",
    "3ec31f2dcb218541e7c5c1751e7a53ebfb0d06e83d338c49776780db780bef0d":
        "state after the 2026-09-06 --score_subset edit (codex r4 PASS) and before the --protocol_variant edit of the same day; the p36_ep399_s20 shards were generated with it. Generation code (sampler, dataset, plan, seeding) identical to the current state: the edits touched argument parsing, shard/report metadata and the subset scoring path only",
}


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
        use_ref_text=bool(ca.get("ref_text", False)),
        use_geo_bias=bool(ca.get("geo_bias", True))).to(dev)
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
             "src/data/incontext_pairs.py", "src/data/ktjd17_incontext.py",
             "src/data/ktjd17_anytop13.py"]     # the representation-view inverse is scoring code (codex 2026-09-08 r2 #2)
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
            "protocol_variant": a.protocol_variant,
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


def merge_shards(paths, want_meta, plan, expect_shape=None):
    """Load shard files, refuse anything that is not exactly one complete protocol run generated under
    THIS plan, source and runtime."""
    import hashlib
    gen_by_clip: dict[str, np.ndarray] = {}
    seen_shards, nshards, loaded, plan_sets = set(), None, [], None
    invariant = ("gen_ckpt_sha256", "seed", "steps", "cfg_text", "gen_batch", "ktjd_root", "exclude_clips",
                 "base_provenance_sha256", "plan_sha256", "n_val_rigs", "protocol_val_n", "protocol_variant")
    # the pool variant changes scoring only: the shards it scores were generated under the frozen protocol (variant None)
    want_meta = {**want_meta, "protocol_variant": None} if want_meta.get("protocol_variant") == "pool" else want_meta
    fp_seen = set()
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
            fp = meta.get("source_fingerprint")
            if not isinstance(fp, str) or (fp != want_meta["source_fingerprint"] and fp not in LEGACY_SOURCE_FINGERPRINTS):
                raise SystemExit(f"[refuse] shard {p}: generation code fingerprint {str(fp)[:16]} is neither this script's "
                                 f"{want_meta['source_fingerprint'][:16]} nor an audited legacy fingerprint")
            fp_seen.add(fp)
            if len(fp_seen) > 1:
                raise SystemExit(f"[refuse] shards were generated under different code fingerprints {sorted(fp_seen)}")
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
                # the sample must be THIS clip's [target_frames, J_rig, 17] (a wrong-shaped payload would silently broadcast
                # into the evaluator tensor; codex 2026-09-08 r2 #4)
                if expect_shape is not None and tuple(arr.shape) != tuple(expect_shape[mid]):
                    raise SystemExit(f"[refuse] shard {p} sample {k} has shape {arr.shape}, expected {expect_shape[mid]}")
                if mid in gen_by_clip:
                    raise SystemExit(f"[refuse] clip {mid} appears in more than one shard ({p})")
                gen_by_clip[mid] = np.asarray(arr, dtype=np.float32)
            loaded.append({"path": p, "sha256": fsha, "shard": k_, "n_clips": len(got),
                           "device": meta.get("device"), "runtime": meta.get("runtime"),
                           "source_fingerprint": fp,
                           "legacy_fingerprint_note": LEGACY_SOURCE_FINGERPRINTS.get(fp) if fp != want_meta["source_fingerprint"] else None})
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
def encode_split(core, eval_ds, gen_by_clip, dev, a, keep=None):
    """keep: optional set of motion_ids to score (the clean subset); None = the full protocol set."""
    te, me_gt, me_gen = [], [], []
    meta = {"motion_id": [], "source_motion_id": [], "caption_text": []}
    missing = []
    buf = []
    seen_keep = set()

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
        if keep is not None and mid not in keep:
            continue
        if keep is not None:
            seen_keep.add(mid)
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
    if keep is not None and seen_keep != set(keep):
        absent = sorted(set(keep) - seen_keep)
        raise SystemExit(f"[refuse] {len(absent)} subset clips are not in the eval val set (e.g. {absent[:5]}); "
                         f"the subset must be built from the same corpus cut")
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
                      exclude_clips=excl, normalization=str(ca.get("rep_norm", "percell")))
    pins_ck = ck.get("ktjd_pins") or {}
    live = {**base.provenance, "exclusion": base.provenance_exclusion}
    drift = sorted(k for k, v in pins_ck.items() if k in live and live[k] != v)
    if drift:
        raise SystemExit(f"[refuse] data drift vs gen-ckpt pins: {drift}")
    if "representation" in live and pins_ck.get("representation") != live["representation"]:
        # a representation view's checkpoint must carry the pin; a missing pin is a mismatch, not a pass (codex 2026-09-08 r2 #3)
        raise SystemExit(f"[refuse] the live data is representation view {live['representation']!r} but the checkpoint pins "
                         f"{pins_ck.get('representation')!r}")
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
        _J_of = {str(r["rig_id"]): int(np.asarray(base.skeleton(str(r["rig_id"]))["parents"]).shape[0]) for r in base._rows}
        _expect = {str(r["clip_id"]): (int(ca["target_frames"]), _J_of[str(r["rig_id"])], 17) for r in base._rows}
        gen_by_clip, nsh, loaded = merge_shards(paths, meta, plan, expect_shape=_expect)
        shard_fp = loaded[0]["source_fingerprint"]        # merge_shards enforced one fingerprint across shards
        gen_mode = {"mode": "sharded_merge", "nshards": nsh, "plan_sha256": plan["plan_sha256"],
                    "source_fingerprint": shard_fp,        # the code that GENERATED the samples (codex r3 P2-1)
                    "scoring_source_fingerprint": meta["source_fingerprint"],
                    "legacy_fingerprint_note": LEGACY_SOURCE_FINGERPRINTS.get(shard_fp) if shard_fp != meta["source_fingerprint"] else None,
                    "runtime": meta["runtime"], "shards": loaded}
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
    # the evaluator was trained on per-cell-normalized KTJD channels (evaluator_ktjd16_pz_v1): its side is
    # ALWAYS percell, whatever normalization the generator checkpoint used
    # REPRESENTATION VIEW (ablation item 2a, 2026-09-07): a generator trained on a derived view whose payloads carry the old
    # AnyTop-13 representation (derivation.representation.id) is scored in the PARENT corpus's KTJD-17 per-cell space: its samples
    # are converted back (src/data/ktjd17_anytop13.anytop13_to_ktjd17, the release's own root split -- the frozen PZ release stores
    # the un-smoothed root track, so the split is exact) and normalised with the parent's stats; the control arm's samples are
    # untouched, so both arms meet the evaluator under the frozen protocol. Reports stamp protocol.gen_representation.
    from src.data.ktjd17_anytop13 import REPRESENTATION_ID, anytop13_to_ktjd17
    _rep = ((getattr(base, "derivation", None) or {}).get("representation") or {})
    gen_representation = str(_rep.get("id")) if _rep else "ktjd17"
    if _rep and gen_representation != REPRESENTATION_ID:
        raise SystemExit(f"[refuse] unknown representation view {gen_representation!r} (this script converts {REPRESENTATION_ID!r})")
    if _rep:
        eval_root = str(base.derivation["parent_root"])
        eval_stats = str(_rep["parent_norm_stats"]["path"])
        if hashlib.sha256(Path(eval_stats).read_bytes()).hexdigest() != str(_rep["parent_norm_stats"]["sha256"]):
            raise SystemExit(f"[refuse] parent stats {eval_stats} do not match the sha pinned in the view's derivation.json")
        # the view pins the converter bytes it was built with and the parent manifest it mirrors; both must still hold
        # (a changed converter means a rebuilt view; codex 2026-09-08 r2 #2)
        _conv_p = Path(__file__).resolve().parents[1] / "src" / "data" / "ktjd17_anytop13.py"
        if hashlib.sha256(_conv_p.read_bytes()).hexdigest() != str(_rep.get("converter_sha256")):
            raise SystemExit("[refuse] src/data/ktjd17_anytop13.py differs from the converter the view was built with -- rebuild the view")
        if hashlib.sha256((Path(eval_root) / "manifests" / "clips.jsonl").read_bytes()).hexdigest() != str(base.derivation.get("parent_manifest_sha256")):
            raise SystemExit("[refuse] the parent manifest differs from the one the view was derived from")
        _sch = json.loads((Path(eval_root) / "schema.json").read_text())
        _eps_h = float(_sch["heading"]["eps_h"]); _fps = float(_sch["fps_target"])
        _T_of = {str(r["clip_id"]): int(r["T_target"]) for r in base._rows}
    else:
        eval_root, eval_stats = root, ca.get("ktjd_percell_stats", "data/ktjd17_percell_stats_v1.npz")
    base_eval = Ktjd17Base(eval_root, caption_emb_cache=ca["caption_cache"],
                           joint_semantics=ca["joint_sem"], texts_json=ca["texts_json"],
                           percell_stats=eval_stats,
                           exclude_clips=excl, normalization="percell")
    eval_ds = Ktjd17T2MEvalDataset(base_eval, "val", max_frames=240, exclude=excl)
    if len(eval_ds) != PROTOCOL_VAL_N:
        raise SystemExit(f"[refuse] eval val has {len(eval_ds)} clips, protocol pins "
                         f"{PROTOCOL_VAL_N}")
    keep = None; subset_rec = None
    if a.score_subset:
        raw = Path(a.score_subset).read_bytes(); sub = json.loads(raw)
        if sub.get("format") != "pz-clean-val-subset-v1":
            raise SystemExit(f"[refuse] --score_subset format {sub.get('format')!r}, expected pz-clean-val-subset-v1")
        RULE = ("exact motion sha256 absent from all active train clips AND (rig_id, source_action_name) "
                "absent from active train")
        if sub.get("rule") != RULE:
            raise SystemExit(f"[refuse] subset rule {sub.get('rule')!r} is not the audited rule")
        # the subset must have been built from the checkpoint's exact corpus cut: same manifest bytes, same
        # exclusion bytes, same corpus generation (codex 2026-09-06 P1-2: names and paths do not identify content)
        # identity of the cut the subset was built from vs the cut this checkpoint pins: the live manifest bytes of the
        # checkpoint's root (the base corpus has no derivation.json, so provenance carries no manifest digest -- codex r2),
        # the corpus generation id and the exclusion artifact's sha256
        live_manifest_sha = hashlib.sha256((Path(root) / "manifests" / "clips.jsonl").read_bytes()).hexdigest()
        want_cut = {"manifest_sha256": live_manifest_sha,
                    "generation_id": base.provenance.get("generation_id"),
                    "exclusions_sha256": (base.provenance_exclusion or {}).get("sha256")}
        got_cut = {k: sub.get(k) for k in want_cut}
        if any(v is None for v in want_cut.values()) or got_cut != want_cut:
            raise SystemExit(f"[refuse] subset corpus identity {got_cut} != the gen ckpt's cut {want_cut}")
        clean = sub.get("clean")
        if not isinstance(clean, list) or not clean:
            raise SystemExit("[refuse] subset file lists no clips")
        import re as _re
        val_rows = {str(r["clip_id"]): r for r in base_eval._rows if str(r.get("split")) == "val"}
        for c in clean:
            if not (isinstance(c, dict) and isinstance(c.get("clip_id"), str) and isinstance(c.get("rig_id"), str) and c["rig_id"]
                    and isinstance(c.get("source_action_name"), str) and c["source_action_name"]
                    and _strict_int(c.get("T_target")) and c["T_target"] > 0
                    and isinstance(c.get("motion_sha256"), str) and _re.fullmatch(r"[0-9a-f]{64}", c["motion_sha256"])):
                raise SystemExit(f"[refuse] malformed subset record {str(c)[:120]}")
            r = val_rows.get(c["clip_id"])
            if r is None or str(r["rig_id"]) != c["rig_id"] or str(r["source_action_name"]) != c["source_action_name"] \
                    or int(r["T_target"]) != c["T_target"]:
                raise SystemExit(f"[refuse] subset record {c['clip_id']} does not match the active validation row "
                                 f"(rig/source/length): {str(c)[:100]}")
        keep = {c["clip_id"] for c in clean}
        ck_ = ("n_active_val", "n_clean", "n_clean_rigs", "n_val_exact_dup_in_train", "n_val_same_source_in_train",
               "n_val_both", "n_val_removed")
        counts = {k: sub.get(k) for k in ck_}
        if not all(_strict_int(v) and v >= 0 for v in counts.values()):
            raise SystemExit(f"[refuse] subset counts malformed {counts}")
        ok = (len(clean) == len(keep) == counts["n_clean"] and counts["n_active_val"] == PROTOCOL_VAL_N
              and counts["n_clean"] + counts["n_val_removed"] == counts["n_active_val"]
              and counts["n_val_both"] <= min(counts["n_val_exact_dup_in_train"], counts["n_val_same_source_in_train"])
              and counts["n_val_removed"] == counts["n_val_exact_dup_in_train"] + counts["n_val_same_source_in_train"] - counts["n_val_both"]
              and counts["n_clean_rigs"] == len({c["rig_id"] for c in clean}))
        if not ok:
            raise SystemExit(f"[refuse] subset counts inconsistent: listed {len(clean)}, unique {len(keep)}, {counts}")
        absent = sorted(keep - set(gen_by_clip))
        if absent:
            raise SystemExit(f"[refuse] {len(absent)} subset clips have no generation (e.g. {absent[:5]})")
        subset_rec = {"path": a.score_subset, "sha256": hashlib.sha256(raw).hexdigest(), "rule": sub["rule"],
                      "n_clean": sub["n_clean"], "n_clean_rigs": sub.get("n_clean_rigs"),
                      "n_active_val": sub.get("n_active_val"), "n_val_exact_dup_in_train": sub.get("n_val_exact_dup_in_train"),
                      "n_val_same_source_in_train": sub.get("n_val_same_source_in_train")}
        if len(keep) < a.pool:
            raise SystemExit(f"[refuse] subset has {len(keep)} clips, fewer than one retrieval pool of {a.pool}")
        print(f"[gen-eval] scoring the CLEAN SUBSET only: {len(keep)} of {len(gen_by_clip)} generated clips "
              f"({sub['rule']})", flush=True)
    if base.normalization != base_eval.normalization or _rep:
        # samples live in the generator's space (normalization and/or representation); move them into the evaluator's
        # per-cell KTJD-17 space through raw units (raw = x*(std+floor)+mean on both sides), re-zeroing invalid cells.
        # A representation view's samples are first cut to the clip's valid length (padding frames must not enter the root
        # integration) and converted with the release's exact root split; the padded frames are re-appended as zeros.
        clip2rig = {str(r["clip_id"]): str(r["rig_id"]) for r in base._rows}
        from src.data.ktjd17_incontext import _STD_FLOOR
        n_conv, n_degenerate = 0, 0
        for mid, g in list(gen_by_clip.items()):
            rig = clip2rig[mid]; J = g.shape[1]
            mu_v, sd_v = base._stats(rig); mu_p, sd_p = base_eval._stats(rig)
            cv = base_eval.static_masks(rig)["channel_valid"][:J]
            raw = (g.astype(np.float64) * (sd_v[None, :J] + _STD_FLOOR) + mu_v[None, :J])
            if _rep:
                Tv = min(_T_of[mid], raw.shape[0])
                conv, _hv, _dg = anytop13_to_ktjd17(raw[:Tv], np.asarray(base.skeleton(rig)["parents"])[:J], fps=_fps, eps_h=_eps_h)
                n_degenerate += _dg["degenerate_facing_frames"] + _dg["degenerate_child_slots"]
                raw = np.zeros_like(raw); raw[:Tv] = conv
                raw[:Tv][~_hv, 0, 15:17] = 0.0                                 # KTJD stores invalid headings as exact zero
            gp = ((raw - mu_p[None, :J]) / (sd_p[None, :J] + _STD_FLOOR)).astype(np.float32)
            gp[:, ~cv] = 0.0
            if _rep:
                gp[Tv:] = 0.0                                                  # padding stays inert
            gen_by_clip[mid] = gp; n_conv += 1
        print(f"[gen-eval] converted {n_conv} samples from the generator's space (normalization {base.normalization!r}, "
              f"representation {gen_representation!r}; degenerate generated 6D cells replaced: {n_degenerate}) into the "
              f"evaluator's per-cell KTJD-17 space", flush=True)
    te, me_gt, me_gen, meta = encode_split(core, eval_ds, gen_by_clip, dev, a, keep=keep)

    gpool = torch.Generator().manual_seed(a.seed)
    n = te.shape[0]
    # the pools are chunks of THIS order: recording its digest makes the evaluation order part of the scoring artifact, so a later
    # re-scoring (the evaluator-validation controls) can prove it scored the same clips in the same order (codex controls r5 P1)
    eval_order_sha = hashlib.sha256("\n".join(str(m) for m in meta["motion_id"]).encode()).hexdigest()
    report = {"protocol": {"val_n": n, "pool": a.pool, "cfg_text": a.cfg_text, "subset": subset_rec,
                           "eval_order_sha256": eval_order_sha,
                           "variant": a.protocol_variant,
                           "gen_normalization": base.normalization,
                           "gen_representation": gen_representation,
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
