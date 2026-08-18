"""Graph-CodeFlow TEXT→MOTION generation quality in the FROZEN evaluator's space
(spec gate). The generated "answer" is the CONTINUOUS-decoded motion (user 2026-06-17:
backbone answer = continuous form, no codebook snap).

OFFLINE CLI wrapper. The generation+embedding+metrics loop lives in the SHARED helper
src/eval/codeflow_gen_eval.run_gen_eval (single source of truth — same code the backbone
trainer's ONLINE eval hook calls, so online == offline, no drift). This script only does:
arg parsing, model loading (flow / frozen VQVAE / frozen evaluator / T5), the data-contract
checks, building the strided index subset (+ optional --exclude_truebones), and reporting.

Per-subset is by motion_id (HML3D→human / PZ_→animal / else→truebones), done inside the
helper. SNAPSHOT eval (backbone may be mid-training). Single-GPU, frozen, no grad.
"""
from __future__ import annotations
import argparse, hashlib, importlib.util, json
from pathlib import Path
import torch
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.anytop_t2m_eval_dataset import AnyTopT2MEvalDataset
from src.models.graph_salad.t2m_evaluator import AnyTopT2MEvaluator
from src.eval.codeflow_gen_eval import run_gen_eval


def _imp(name):
    p = Path(__file__).resolve().parent / "animate_graph_codeflow.py"
    spec = importlib.util.spec_from_file_location("_agcf", p)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return getattr(mod, name)


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--flow_ckpt", required=True)
    ap.add_argument("--eval_ckpt", required=True)
    ap.add_argument("--caption_cache", default=None,
                    help="LLM2Vec ragged sidecar prefix; REQUIRED when the flow ckpt was "
                         "trained with text_dim != 768")
    ap.add_argument("--val_manifest", required=True)
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--n_samples", type=int, default=0, help="subset size (strided over the kept set for species spread); 0=full.")
    ap.add_argument("--balanced", action="store_true",
                    help="BALANCED subset: take --n_samples//2 animal (motion_id PZ_) + --n_samples//2 human "
                         "(HML), strided WITHIN each class, so the per-subset R@k are computed on equal, "
                         "larger samples (stable human R@k; fixes the natural-distribution n=69 human noise). "
                         "Requires --n_samples>0; a class with fewer clips than n_samples//2 contributes all it has.")
    ap.add_argument("--gt_baseline", action="store_true",
                    help="GT BASELINE/CEILING: skip generation; gallery = GT motion embedding so the reported "
                         "rprec_text_to_gen IS the text->GT R-precision (the upper bound a generator can reach). "
                         "Fast (no ODE sampling). Pair with --balanced --n_samples for the equal animal/human ceiling.")
    ap.add_argument("--exclude_truebones", action="store_true",
                    help="restrict eval to animo4d (motion_id startswith PZ_); drop truebones — the "
                         "evaluator is weak on the scarce truebones (sanity R@1 0.48 vs animo4d 0.96) "
                         "and 78 clips are too few for stable metrics. (visual QA still covers truebones.) "
                         "Obsolete for the L4safeHuman dataset (no truebones; use the animal/human subset split).")
    ap.add_argument("--pool", type=int, default=32)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--cfg_scale", type=float, default=4.0)
    ap.add_argument("--num_frames", type=int, default=300)
    ap.add_argument("--max_joints", type=int, default=144)
    ap.add_argument("--gen_batch", type=int, default=32,
                    help="clips per batched flow.sample (B). B>1 batches generation (collate pads J/T, "
                         "per-clip masks + gt_T truncation handle variable length).")
    ap.add_argument("--fid_min", type=int, default=1024)
    ap.add_argument("--max_div_pairs", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=None)
    ap.add_argument("--base_split", default=None, choices=["train", "val", "all"],
                    help="which partition the underlying dataset serves. Required as 'all' for a "
                         "HELD-OUT manifest: a held topology contributes its whole inventory, so "
                         "its clips come from both original partitions and the val list alone "
                         "cannot resolve them.")
    ap.add_argument("--moment_policy", default="own",
                    choices=["own", "estimated", "measured"],
                    help="where per-rig normalisation moments come from. own (default) = the "
                         "rig's own motion moments, which for a HELD-OUT rig is transductive and "
                         "must be reported as a disclosed upper bound. estimated = predicted from "
                         "the static rest pose, the only honest choice at k=0. measured = "
                         "computed from the k target clips one would actually have.")
    ap.add_argument("--moment_estimator", default=None,
                    help="estimator .npz, required for --moment_policy estimated")
    ap.add_argument("--moment_measured", default=None,
                    help="per-object measured moments .npz, required for --moment_policy measured")
    ap.add_argument("--emb_dump", default=None,
                    help="if set, also save the per-clip evaluator embeddings (text/gen/gt) and "
                         "motion_ids to this .pt so per-skeleton analysis can be done offline. "
                         "Does not change any reported metric.")
    return ap.parse_args()


@torch.no_grad()
def main():
    args = parse_args()
    if args.pool < 3:
        raise SystemExit(f"--pool must be >=3; got {args.pool}")
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    load_flow = _imp("load_flow")
    load_frozen_tokenizer = _imp("load_frozen_tokenizer")

    flow, fck = load_flow(args.flow_ckpt, 512, dev)
    if "latent_mean" in fck:                              # restore empirical-norm buffers
        flow.latent_mean = fck["latent_mean"].to(dev)
        flow.latent_std = fck["latent_std"].to(dev)
    vq_ckpt = fck.get("frozen_vqvae_ckpt")
    tokenizer, ta = load_frozen_tokenizer(vq_ckpt, dev)
    stride = int(ta["temporal_stride"])
    print(f"[gen-eval] flow {args.flow_ckpt} ep={fck.get('epoch')} val_flow={fck.get('val_flow')} | "
          f"VQVAE {vq_ckpt} stride={stride} | variant={getattr(flow,'model_variant','?')}", flush=True)

    # frozen evaluator
    eck = torch.load(args.eval_ckpt, map_location="cpu")
    ea = eck["args"]
    g = (lambda k, d: ea.get(k, d)) if isinstance(ea, dict) else (lambda k, d: getattr(ea, k, d))
    core = AnyTopT2MEvaluator(
        coemb_dim=g("coemb_dim", 512), text_tower=g("text_tower", "distilbert"),
        distilbert_path=g("distilbert_path", "checkpoints/text_encoders/distilbert-base-uncased"),
        text_max_length=g("text_max_length", 64), n_heads=g("n_heads", 8), d_ff=g("d_ff", 2048),
        n_graph_layers=g("n_graph_layers", 6), n_temporal_layers=g("n_temporal_layers", 4),
        motion_feat_dim=g("motion_feat_dim", 13),  # 12ch (contact-free) ckpt rebuilds at 12; old 13ch -> 13
        dropout=g("dropout", 0.1), learnable_temperature=not g("fixed_temperature", False),
        temperature=g("temperature", 0.07))
    miss, unexp = core.load_state_dict(eck["model"], strict=False)
    bad = [k for k in miss if not k.startswith("text_distilbert.text_model.")]
    if bad or unexp:
        raise SystemExit(f"[gen-eval] evaluator load mismatch: missing(non-backbone)={bad[:8]} unexpected={list(unexp)[:8]}")
    core.to(dev).eval()
    # data contract: evaluator + VQVAE must share data root + joints (eval-space validity).
    eval_root = g("data_root", None)
    if eval_root and args.data_root != eval_root:
        raise SystemExit(f"[gen-eval] --data_root {args.data_root} != evaluator data_root {eval_root}")
    vq_root = ta.get("anytop_root") or ta.get("data_root")   # VQVAE args use anytop_root; data_root may be None
    if vq_root and eval_root and vq_root != eval_root:
        raise SystemExit(f"[gen-eval] VQVAE root {vq_root} != evaluator data_root {eval_root} (normalization would differ)")
    if g("num_frames", None) and args.num_frames != int(g("num_frames", -1)):
        raise SystemExit(f"[gen-eval] --num_frames {args.num_frames} != evaluator num_frames {g('num_frames',None)} (must be full-length)")
    if g("max_joints", None) and args.max_joints != int(g("max_joints", -1)):
        raise SystemExit(f"[gen-eval] --max_joints {args.max_joints} != evaluator max_joints {g('max_joints',None)}")
    print(f"[gen-eval] evaluator {args.eval_ckpt} ep={eck.get('epoch')} loaded", flush=True)

    # Text encoder matches the FLOW CKPT's convention (codex chain review #5): T5 runtime
    # encoding for legacy 768 checkpoints; LLM2Vec sidecar LOOKUP for 4096 checkpoints (the
    # 8B encoder never runs here — identical to the online gen-eval convention).
    _fa = fck.get("args", {})
    _fget = (lambda k, d: _fa.get(k, d)) if isinstance(_fa, dict) else (lambda k, d: getattr(_fa, k, d))
    _tdim = int(_fget("text_dim", 768))
    if _tdim == 768:
        from transformers import T5EncoderModel, T5TokenizerFast
        t5tok = T5TokenizerFast.from_pretrained("t5-base")
        t5 = T5EncoderModel.from_pretrained("t5-base").to(dev).eval()

        @torch.no_grad()
        def t5_encode_batch(texts):
            # batched T5: list[str] -> global[b,768], tokens[b,64,768], mask[b,64] (t5-base/max64/masked-mean)
            enc = t5tok(list(texts), return_tensors="pt", padding="max_length", truncation=True, max_length=64).to(dev)
            hs = t5(input_ids=enc.input_ids, attention_mask=enc.attention_mask).last_hidden_state  # [b,64,768]
            m = enc.attention_mask.bool()                                                          # [b,64]
            gl = (hs * m.unsqueeze(-1).float()).sum(1) / m.sum(1, keepdim=True).clamp_min(1)        # [b,768]
            return gl.float(), hs.float(), m
    else:
        if not args.caption_cache:
            raise SystemExit(
                f"[gen-eval] flow ckpt has text_dim={_tdim}; pass --caption_cache "
                f"(LLM2Vec sidecar prefix) — offline eval looks captions up, it does not "
                f"run the 8B encoder")
        from src.eval.codeflow_gen_eval import make_caption_lookup_encoder
        t5_encode_batch = make_caption_lookup_encoder(
            args.caption_cache, args.data_root, _tdim, dev)

    _ms = None
    if args.moment_policy != "own":
        from src.data.moment_source import MomentSource
        _ms = MomentSource(args.moment_policy, estimator_path=args.moment_estimator,
                           measured_path=args.moment_measured)
        print(f"[gen-eval] moment policy: {args.moment_policy}", flush=True)
    ds = AnyTopT2MEvalDataset(moment_source=_ms, base_split=args.base_split,
                              manifest_path=args.val_manifest, data_root=args.data_root,
                              caption_emb_cache=None, split="val", view="full",
                              num_frames=args.num_frames, max_joints=args.max_joints)
    n_total = len(ds)
    cand = list(range(n_total))
    n_tb = sum(1 for i in cand if not str(ds._plan[i][1]["motion_id"]).startswith("PZ_"))
    if args.exclude_truebones:
        cand = [i for i in cand if str(ds._plan[i][1]["motion_id"]).startswith("PZ_")]
        print(f"[gen-eval] exclude_truebones: dropped {n_tb} non-PZ, kept {len(cand)} animo4d", flush=True)
    def _stride(lst, k):
        return lst if k >= len(lst) else lst[::max(1, len(lst) // k)][:k]   # even spread to ~k items
    if args.balanced:
        if not args.n_samples or args.n_samples <= 0:
            raise SystemExit("[gen-eval] --balanced requires --n_samples>0 (split half animal / half human)")
        per = args.n_samples // 2
        animal_cand = [i for i in cand if str(ds._plan[i][1]["motion_id"]).startswith("PZ_")]
        human_cand = [i for i in cand if str(ds._plan[i][1]["motion_id"]).startswith("HML")]
        a_sel, h_sel = _stride(animal_cand, per), _stride(human_cand, per)
        idxs = a_sel + h_sel
        print(f"[gen-eval] BALANCED: {len(a_sel)} animal + {len(h_sel)} human "
              f"(target {per}+{per}; avail {len(animal_cand)} animal/{len(human_cand)} human)", flush=True)
    else:
        n_take = len(cand) if (not args.n_samples or args.n_samples >= len(cand)) else args.n_samples
        idxs = cand[::max(1, len(cand) // n_take)][:n_take]                 # strided for species spread
    print(f"[gen-eval] val {n_total} (non-PZ={n_tb}); {'GT-baseline on' if args.gt_baseline else 'generating'} {len(idxs)} | "
          f"steps={args.steps} cfg={args.cfg_scale} nf={args.num_frames}", flush=True)

    sink = [] if args.emb_dump else None
    report = run_gen_eval(
        flow=flow, tokenizer=tokenizer, core=core, t5_encode_batch=t5_encode_batch,
        ds=ds, idxs=idxs, dev=dev, stride=stride, pool=args.pool, steps=args.steps,
        cfg_scale=args.cfg_scale, num_frames=args.num_frames, gen_batch=args.gen_batch,
        fid_min=args.fid_min, max_div_pairs=args.max_div_pairs, seed=args.seed,
        gt_baseline=args.gt_baseline, emb_sink=sink)
    from src.data import provenance as _prov
    report.update({"flow_ckpt": args.flow_ckpt, "flow_epoch": fck.get("epoch"), "eval_ckpt": args.eval_ckpt,
                   "gt_baseline": bool(args.gt_baseline), "balanced": bool(args.balanced),
                   # A number produced under one moment policy is not comparable to one produced
                   # under another, so the policy travels with the result rather than living in
                   # someone's memory of which command was run.
                   "moment_policy": args.moment_policy,
                   "moment_artifact": args.moment_estimator or args.moment_measured,
                   "moment_artifact_sha256": _prov.sha256_file(
                       args.moment_estimator or args.moment_measured)
                   if (args.moment_estimator or args.moment_measured) else None,
                   "flow_provenance": _prov.read(fck)})

    def _rr(m):
        rr = m.get("rprec_text_to_gen")
        return f"R@1={rr[1]:.3f} R@2={rr[2]:.3f} R@3={rr[3]:.3f}" if rr else "R@k=NA(n<pool)"

    def _n(x):
        return f"{x:.3f}" if x is not None else "NA"

    o = report["overall"]
    print(f"[gen-eval] OVERALL n={o['n']} text→gen {_rr(o)} match(mean/med)={_n(o['matching_mean'])}/{_n(o['matching_median'])} "
          f"FID={o['fid']} div(gen/gt)={_n(o['diversity_gen'])}/{_n(o['diversity_gt'])}", flush=True)
    for s, m in report["per_subset"].items():
        print(f"[gen-eval] SUBSET {s:9s} n={m['n']:5d} text→gen {_rr(m)} match={_n(m['matching_mean'])} "
              f"FID={m['fid']} ({m['fid_note']}) div(gen/gt)={_n(m['diversity_gen'])}/{_n(m['diversity_gt'])}", flush=True)

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(report, indent=2))
        print(f"[gen-eval] report -> {args.out}", flush=True)

    if args.emb_dump:
        # run_gen_eval appends exactly one payload; fail loud rather than silently dumping a
        # partial or empty file that would look like a successful measurement.
        if not sink or len(sink) != 1:
            raise RuntimeError(f"--emb_dump: expected 1 emb_sink payload, got {len(sink) if sink is not None else 'None'}")
        p = sink[0]
        n_emb, n_mid = p["gen_emb"].shape[0], len(p["motion_ids"])
        if not (n_emb == n_mid == p["text_emb"].shape[0] == p["gt_emb"].shape[0] == report["overall"]["n"]):
            raise RuntimeError(f"--emb_dump: row mismatch gen={n_emb} mids={n_mid} "
                               f"text={p['text_emb'].shape[0]} gt={p['gt_emb'].shape[0]} report_n={report['overall']['n']}")
        # Provenance is dumped so the analysis can PROVE which cohort it is reading rather than
        # trusting an independently supplied --data_root (codex P1 #3). `selection` records every
        # flag that changes which clips were evaluated; `overall_matching_mean` lets the analysis
        # recompute the mean from the embeddings and hard-fail if it disagrees.
        def _md5(path):
            try:
                h = hashlib.md5()
                with open(path, "rb") as fh:
                    for chunk in iter(lambda: fh.read(1 << 20), b""):
                        h.update(chunk)
                return h.hexdigest()
            except OSError as e:
                raise RuntimeError(f"--emb_dump: cannot hash {path} for provenance: {e}") from e
        splits = Path(args.data_root) / "splits"
        Path(args.emb_dump).parent.mkdir(parents=True, exist_ok=True)
        torch.save({"schema": 2, "gen_emb": p["gen_emb"], "text_emb": p["text_emb"], "gt_emb": p["gt_emb"],
                    "motion_ids": p["motion_ids"], "row_keys": p["row_keys"],
                    "flow_ckpt": args.flow_ckpt, "flow_epoch": fck.get("epoch"),
                    "eval_ckpt": args.eval_ckpt, "gt_baseline": bool(args.gt_baseline),
                    "cfg_scale": args.cfg_scale, "steps": args.steps, "seed": args.seed,
                    "data_root": str(Path(args.data_root).resolve()),
                    "val_manifest": str(Path(args.val_manifest).resolve()),
                    "val_manifest_md5": _md5(args.val_manifest),
                    "val_split_md5": _md5(splits / "val.txt"),
                    "train_split_md5": _md5(splits / "train.txt"),
                    "selection": {"n_samples": args.n_samples, "balanced": bool(args.balanced),
                                  "exclude_truebones": bool(args.exclude_truebones),
                                  "n_total": int(n_total), "n_evaluated": int(len(idxs)),
                                  "idxs": [int(i) for i in idxs]},
                    "overall_matching_mean": float(report["overall"]["matching_mean"]),
                    "num_frames": args.num_frames, "pool": args.pool,
                    "target_frames": p["target_frames"], "decoded_frames": p["decoded_frames"],
                    "n_soft_clamped": int(p["n_soft_clamped"])},
                   args.emb_dump)
        print(f"[gen-eval] embeddings ({n_emb} rows) -> {args.emb_dump}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
