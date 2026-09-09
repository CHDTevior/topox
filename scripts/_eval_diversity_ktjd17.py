"""Diversity for the frozen text-to-motion protocol.

Diversity is standard in this line of work and is not derivable from the retrieval report, so it is computed here from the SAME
frozen samples the canonical report scored, in the same evaluator space and under the same arithmetic: the mean L2 between the
embeddings of two index lists of 300 generated clips, each drawn without replacement, next to the same statistic on the real clips
of the validation set (the HumanML3D convention; a model that collapses to one motion scores far below the real value).

usage:
  python scripts/_eval_diversity_ktjd17.py --gen_ckpt CKPT --shards DIR/shard*.npz --report REPORT_bound.json --out OUT.json
"""
from __future__ import annotations
import argparse, hashlib, io, json, sys
from pathlib import Path
import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from scripts._eval_v2_gen_in_evalspace import load_evaluator, encode_split, PROTOCOL_VAL_N            # noqa: E402
from scripts._eval_controls_ktjd17 import EVAL_CKPT                                                    # noqa: E402
from src.data.ktjd17_incontext import Ktjd17Base, _STD_FLOOR                                           # noqa: E402
from src.data.ktjd17_t2m_eval_dataset import Ktjd17T2MEvalDataset                                      # noqa: E402


def mean_pair_l2(x: torch.Tensor, n_pairs: int, gen: torch.Generator) -> float:
    """Mean L2 between two index lists, each drawn WITHOUT replacement and allowed to overlap each other -- the HumanML3D
    reference implementation of diversity (text-to-motion, utils/metrics.py).

    The VALUE is not comparable with published HumanML3D numbers: this evaluator L2-normalises its embeddings, so the statistic
    is bounded by 2 and lives on a different scale from an unnormalised encoder's. It is read against the same statistic on the
    real clips of the same set, which is why that reference is computed and reported beside it."""
    n = x.shape[0]
    if n < 2 or n_pairs > n:
        raise SystemExit(f"[refuse] diversity needs at least {n_pairs} clips to draw its two lists, got {n}")
    i = torch.randperm(n, generator=gen)[:n_pairs]
    j = torch.randperm(n, generator=gen)[:n_pairs]
    return float((x[i] - x[j]).norm(dim=-1).mean())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gen_ckpt", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--shards", nargs="+", required=True, help="the canonical pass's shards")
    ap.add_argument("--report", required=True, help="the canonical merge report of those shards (binds ckpt, order and evaluator)")
    ap.add_argument("--eval_ckpt", default=EVAL_CKPT); ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n_pairs", type=int, default=300); ap.add_argument("--encode_batch", type=int, default=64)
    a = ap.parse_args()
    if a.n_pairs < 2:
        raise SystemExit(f"[refuse] --n_pairs {a.n_pairs} is not a usable sample size")
    if not torch.cuda.is_available():
        raise SystemExit("[refuse] no CUDA device: the canonical pass scored on a GPU and a CPU fallback is different arithmetic")
    dev = "cuda"
    gen = torch.Generator().manual_seed(a.seed)

    buf = Path(a.gen_ckpt).read_bytes(); gen_sha = hashlib.sha256(buf).hexdigest()
    ck = torch.load(io.BytesIO(buf), map_location="cpu", weights_only=False); del buf
    ca = ck["args"]; ca = vars(ca) if not isinstance(ca, dict) else ca
    root, excl = ca["ktjd_root"], (ca.get("exclude_clips") or None)
    # This script feeds the saved samples straight into the evaluator's encoder, which is correct only when the samples ALREADY
    # live in the evaluator's space: a per-cell KTJD-17 checkpoint. A representation view or another normalisation needs the merge's
    # conversion first, and duplicating that conversion here would be a second implementation of it (codex divmm r1 #2).
    if (Path(root) / "derivation.json").is_file() or str(ca.get("rep_norm") or "percell") != "percell":
        raise SystemExit(f"[refuse] {a.gen_ckpt} serves normalisation {ca.get('rep_norm')!r} from {root}; this script scores the "
                         f"evaluator's per-cell KTJD-17 space directly and does not convert. Score such an arm through "
                         f"scripts/_eval_v2_gen_in_evalspace.py --merge instead.")
    parent_root, parent_stats = root, ca.get("ktjd_percell_stats", "data/ktjd17_percell_stats_v1.npz")
    base = Ktjd17Base(parent_root, caption_emb_cache=ca["caption_cache"], joint_semantics=ca["joint_sem"],
                      texts_json=ca["texts_json"], percell_stats=parent_stats, exclude_clips=excl, normalization="percell")
    # ---- #6: the checkpoint pinned the data it trained on; a swapped statistics or caption artifact changes what a clip IS
    live = {**base.provenance, "exclusion": base.provenance_exclusion}
    pins = ck.get("ktjd_pins") or {}
    drift = sorted(k for k, v in pins.items() if k in live and live[k] != v)
    if drift:
        raise SystemExit(f"[refuse] data drift vs the checkpoint's ktjd_pins: {drift}")
    eval_ds = Ktjd17T2MEvalDataset(base, "val", max_frames=240, exclude=excl)
    if len(eval_ds) != PROTOCOL_VAL_N:
        raise SystemExit(f"[refuse] eval val has {len(eval_ds)} clips, protocol pins {PROTOCOL_VAL_N}")
    target_frames = int(ca["target_frames"]); rows = {str(r["clip_id"]): r for r in base._rows}

    rp = None
    if a.report:
        rp = json.loads(Path(a.report).read_text())
        if str(rp["protocol"].get("gen_ckpt_sha256")) != gen_sha:
            raise SystemExit(f"[refuse] {a.gen_ckpt} is not the checkpoint {a.report} scored")
        rt = (rp["protocol"].get("generation") or {}).get("runtime") or {}
        import os
        if "NVIDIA_TF32_OVERRIDE" in os.environ:
            raise SystemExit("[refuse] NVIDIA_TF32_OVERRIDE is set; the recorded runtime cannot see it")
        if rt:
            torch.backends.cuda.matmul.allow_tf32 = bool(rt.get("allow_tf32_matmul"))
            torch.backends.cudnn.allow_tf32 = bool(rt.get("allow_tf32_cudnn"))
            torch.set_float32_matmul_precision(str(rt.get("float32_matmul_precision", "highest")))
            now = {"torch": torch.__version__, "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version()}
            diff = {k: (rt.get(k), now[k]) for k in now if rt.get(k) != now[k]}
            if diff:
                raise SystemExit(f"[refuse] scoring runtime differs from the canonical pass's {diff}; a different arithmetic is a "
                                 f"different number under the same report identity (codex divmm r1 #7)")

    core, eval_sha = load_evaluator(a.eval_ckpt, dev)
    if rp is not None and eval_sha != str(rp["protocol"].get("eval_ckpt_sha256")):
        raise SystemExit(f"[refuse] evaluator {a.eval_ckpt} is not the one the report used")
    out = {"gen_ckpt": a.gen_ckpt, "gen_ckpt_sha256": gen_sha, "eval_ckpt_sha256": eval_sha, "report": a.report,
           "seed": a.seed, "n_pairs": a.n_pairs}

    # ---------------- diversity, on the canonical samples ----------------
    if a.shards:
        if rp is None:
            raise SystemExit("[refuse] --shards needs --report: diversity is a number about the canonical pass, and the report is "
                             "what says which bytes that pass scored")
        gens, proto, shard_sha, shard_idx = {}, None, {}, {}
        for p in a.shards:
            blob = Path(p).read_bytes()
            shard_sha[p] = hashlib.sha256(blob).hexdigest()
            z = np.load(io.BytesIO(blob), allow_pickle=False); meta = json.loads(str(z["__meta"])); del blob
            shard_idx[p] = int(meta["shard"])
            if str(meta.get("gen_ckpt_sha256")) != gen_sha:
                raise SystemExit(f"[refuse] shard {p} is not from this checkpoint")
            this = {k: meta.get(k) for k in ("seed", "steps", "cfg_text", "nshards", "plan_sha256", "source_fingerprint")}
            this["runtime"] = meta.get("runtime")
            proto = this if proto is None else proto
            if this != proto:
                raise SystemExit(f"[refuse] shard {p} does not share the first shard's protocol")
            for k in z.files:
                if k.startswith("clip__"):
                    mid = k[6:]
                    if mid in gens:
                        raise SystemExit(f"[refuse] clip {mid} appears in two shards")
                    gens[mid] = np.asarray(z[k], dtype=np.float32)
        rg = rp["protocol"]["generation"]
        want = {int(sh["shard"]): str(sh["sha256"]) for sh in rg.get("shards", [])}
        have = {shard_idx[p]: shard_sha[p] for p in a.shards}
        if want != have:
            raise SystemExit(f"[refuse] these are not the shard bytes the report scored: report {sorted(want)} vs given {sorted(have)}")
        for k in ("plan_sha256", "source_fingerprint"):
            if rg.get(k) != proto.get(k):
                raise SystemExit(f"[refuse] shard {k} differs from the report's")
        # the SCORING code the report ran under, not only the generation code (codex divmm r2 #8): this script encodes through
        # the same functions, so a change to them since the report is a change to the number
        from scripts._eval_v2_gen_in_evalspace import source_fingerprint
        now_scoring = source_fingerprint(ca.get("anchor", "none"))
        if rg.get("scoring_source_fingerprint") != now_scoring:
            raise SystemExit(f"[refuse] the scoring code has changed since the report, or the report does not record it "
                             f"({str(rg.get('scoring_source_fingerprint'))[:12]} vs {now_scoring[:12]}); re-merge the report first")
        out["scoring_source_fingerprint"] = now_scoring
        if not str(rp["protocol"].get("eval_order_sha256") or ""):
            raise SystemExit(f"[refuse] {a.report} predates the evaluation-order digest -- re-run its merge")
        if len(gens) != PROTOCOL_VAL_N:
            raise SystemExit(f"[refuse] shards hold {len(gens)} clips, protocol pins {PROTOCOL_VAL_N}")
        te, me_gt, me_gen, meta = encode_split(core, eval_ds, gens, dev, a, keep=None)
        order_sha = hashlib.sha256("\n".join(str(m) for m in meta["motion_id"]).encode()).hexdigest()
        if str(rp["protocol"]["eval_order_sha256"]) != order_sha:
            raise SystemExit(f"[refuse] the scored order {order_sha[:12]} differs from the report's "
                             f"{str(rp['protocol']['eval_order_sha256'])[:12]}")
        out["eval_order_sha256"] = order_sha
        out["diversity"] = {"generated": mean_pair_l2(me_gen, a.n_pairs, torch.Generator().manual_seed(a.seed)),
                            "real": mean_pair_l2(me_gt, a.n_pairs, torch.Generator().manual_seed(a.seed)),
                            "n_clips": int(me_gen.shape[0]), "protocol": proto}
        d = out["diversity"]
        print(f"[divmm] diversity generated {d['generated']:.4f} | real {d['real']:.4f} "
              f"({a.n_pairs} random pairs of {d['n_clips']} clips)", flush=True)

    Path(a.out).write_text(json.dumps(out, indent=1))
    print(f"[divmm] -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
