"""Graph-CodeFlow TEXT→MOTION generation quality in the FROZEN evaluator's space
(spec gate). The generated "answer" is the CONTINUOUS-decoded motion (user 2026-06-17:
backbone answer = continuous form, no codebook snap).

Pipeline per val caption (driven by the evaluator's own val data = AnyTopT2MEvalDataset
/ val_all.json, so GT motion + raw caption + graph are exactly the evaluator's regime,
FULL-LENGTH 300):
  raw caption --T5(t5-base, training convention)--> caption_emb/token_emb/mask  (flow cond)
  tokenizer.prepare_skeleton_only(batch, T_lat) --> graph/skel cond
  flow.sample(cond, token_mask, T_lat, C, steps, cfg) --> z_hat   (ODE + CFG)
  tokenizer.decode(z_hat, meta, batch)["pred_motion"] --> CONTINUOUS motion [1,T,J,13]
  evaluator.encode_motion(gen) / encode_motion(GT) / encode_text(caption) --> [.,512] L2-norm
Metrics, OVERALL + per-subset (animo4d=PZ_*, truebones=else):
  - R-precision text→gen R@1/2/3 (pool=--pool, diagonal target) + matching = mean cos(text,gen)
  - FID(GT motion-emb set, gen motion-emb set) in eval space (best-effort, n>=fid_min)
  - diversity(gen) vs diversity(GT)  (mean pairwise L2)
Per-clip generation (B=1) with T_lat matched to the clip's GT length (animate convention).
SNAPSHOT eval (backbone may be mid-training). Single-GPU, frozen, no grad. NEW → codex review.
"""
from __future__ import annotations
import argparse, dataclasses, importlib.util, json, math
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.anytop_t2m_eval_dataset import AnyTopT2MEvalDataset, collate_fn
from src.models.graph_salad.batch import GraphMotionBatch
from src.models.graph_salad.t2m_evaluator import AnyTopT2MEvaluator


def _imp(name):
    p = Path(__file__).resolve().parent / "animate_graph_codeflow.py"
    spec = importlib.util.spec_from_file_location("_agcf", p)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return getattr(mod, name)


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--flow_ckpt", required=True)
    ap.add_argument("--eval_ckpt", required=True)
    ap.add_argument("--val_manifest", required=True)
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--n_samples", type=int, default=0, help="subset size (strided over the kept set for species spread); 0=full.")
    ap.add_argument("--exclude_truebones", action="store_true",
                    help="restrict eval to animo4d (motion_id startswith PZ_); drop truebones — the "
                         "evaluator is weak on the scarce truebones (sanity R@1 0.48 vs animo4d 0.96) "
                         "and 78 clips are too few for stable metrics. (visual QA still covers truebones.)")
    ap.add_argument("--pool", type=int, default=32)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--cfg_scale", type=float, default=4.0)
    ap.add_argument("--num_frames", type=int, default=300)
    ap.add_argument("--max_joints", type=int, default=144)
    ap.add_argument("--gen_batch", type=int, default=32,
                    help="clips per batched flow.sample (B). B=1 was the old per-clip slowness; "
                         "B>1 batches generation (collate pads J/T, per-clip masks + gt_T truncation handle variable length).")
    ap.add_argument("--fid_min", type=int, default=1024)
    ap.add_argument("--max_div_pairs", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=None)
    return ap.parse_args()


@torch.no_grad()
def rprec_pool(q, g, ks=(1, 2, 3)):
    sim = q @ g.t()
    correct = torch.zeros_like(sim, dtype=torch.bool); correct.fill_diagonal_(True)
    order = sim.argsort(dim=1, descending=True)
    return {k: correct.gather(1, order[:, :k]).any(dim=1).float().mean().item() for k in ks}


def pooled_rprec(q, g, pool):
    n = q.shape[0]; npool = n // pool
    if npool == 0:
        return None, 0
    acc = {1: 0.0, 2: 0.0, 3: 0.0}
    for p in range(npool):
        s = slice(p * pool, (p + 1) * pool)
        r = rprec_pool(q[s], g[s])
        for k in acc:
            acc[k] += r[k]
    return {k: acc[k] / npool for k in acc}, npool


def fid_score(x, y):
    try:
        from scipy.linalg import sqrtm
    except Exception:
        return None, "scipy unavailable"
    mu1, mu2 = x.mean(0), y.mean(0)
    c1, c2 = np.cov(x, rowvar=False), np.cov(y, rowvar=False)
    cov = sqrtm(c1 @ c2)
    if np.iscomplexobj(cov):
        cov = cov.real
    val = float((mu1 - mu2) @ (mu1 - mu2) + np.trace(c1 + c2 - 2.0 * cov))
    if not np.isfinite(val):
        return None, "non-finite"
    return max(val, 0.0), "ok"


def mean_pair_l2(x, max_pairs, gen):
    n = x.shape[0]
    if n < 2:
        return None
    if n * (n - 1) // 2 <= max_pairs:
        return float(torch.pdist(x).mean().item())
    i = torch.randint(0, n, (max_pairs,), generator=gen)
    j = torch.randint(0, n, (max_pairs,), generator=gen)
    keep = i != j
    return float((x[i[keep]] - x[j[keep]]).norm(dim=-1).mean().item())


def subset_metrics(text_emb, gen_emb, gt_emb, pool, fid_min, max_div_pairs, gen):
    n = gen_emb.shape[0]
    match = (text_emb * gen_emb).sum(-1)                 # cos(caption, gen) true pairs
    rr, npp = pooled_rprec(text_emb, gen_emb, pool)      # text query, gen gallery
    m = {"n": n, "matching_mean": float(match.mean().item()),
         "matching_median": float(match.median().item()),
         "rprec_text_to_gen": rr, "rprec_pools": npp,
         "diversity_gen": mean_pair_l2(gen_emb, max_div_pairs, gen),
         "diversity_gt": mean_pair_l2(gt_emb, max_div_pairs, gen)}
    if n >= fid_min:
        f, note = fid_score(gt_emb.numpy(), gen_emb.numpy())
        m["fid"], m["fid_note"] = f, note
    else:
        m["fid"], m["fid_note"] = None, f"skipped (n={n} < {fid_min})"
    return m


@torch.no_grad()
def main():
    args = parse_args()
    if args.pool < 3:
        raise SystemExit(f"--pool must be >=3; got {args.pool}")
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gen = torch.Generator().manual_seed(args.seed)
    torch.manual_seed(args.seed)                 # seed flow.sample's global torch.randn (reproducible generation)
    torch.cuda.manual_seed_all(args.seed)

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

    # T5 (training convention: t5-base, max_length=64), loaded ONCE.
    from transformers import T5EncoderModel, T5TokenizerFast
    t5tok = T5TokenizerFast.from_pretrained("t5-base")
    t5 = T5EncoderModel.from_pretrained("t5-base").to(dev).eval()

    def t5_encode_batch(texts):
        # batched T5: list[str] -> global[b,768], tokens[b,64,768], mask[b,64] (same t5-base/max64/masked-mean convention)
        enc = t5tok(list(texts), return_tensors="pt", padding="max_length", truncation=True, max_length=64).to(dev)
        hs = t5(input_ids=enc.input_ids, attention_mask=enc.attention_mask).last_hidden_state  # [b,64,768]
        m = enc.attention_mask.bool()                                                          # [b,64]
        gl = (hs * m.unsqueeze(-1).float()).sum(1) / m.sum(1, keepdim=True).clamp_min(1)        # [b,768]
        return gl.float(), hs.float(), m

    ds = AnyTopT2MEvalDataset(manifest_path=args.val_manifest, data_root=args.data_root,
                              caption_emb_cache=None, split="val", view="full",
                              num_frames=args.num_frames, max_joints=args.max_joints)
    n_total = len(ds)
    cand = list(range(n_total))
    n_tb = sum(1 for i in cand if not str(ds._plan[i][1]["motion_id"]).startswith("PZ_"))
    if args.exclude_truebones:
        cand = [i for i in cand if str(ds._plan[i][1]["motion_id"]).startswith("PZ_")]
        print(f"[gen-eval] exclude_truebones: dropped {n_tb} truebones, kept {len(cand)} animo4d", flush=True)
    n_take = len(cand) if (not args.n_samples or args.n_samples >= len(cand)) else args.n_samples
    idxs = cand[::max(1, len(cand) // n_take)][:n_take]                 # strided for species spread
    print(f"[gen-eval] val {n_total} (truebones={n_tb}); generating {len(idxs)} | steps={args.steps} cfg={args.cfg_scale} nf={args.num_frames}", flush=True)

    TE, GE, GTE, objs = [], [], [], []
    B = max(1, args.gen_batch)
    done = 0
    for bstart in range(0, len(idxs), B):                                    # BATCHED generation (was per-clip B=1)
        bidx = idxs[bstart:bstart + B]
        items = [ds[di] for di in bidx]
        caps = [it.get("caption_text") or "" for it in items]
        coll = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in collate_fn(items).items()}
        batch = GraphMotionBatch.from_collate_dict(coll)                     # collate pads J/T over the batch
        gl, tok, msk = t5_encode_batch(caps)                                 # [b,768],[b,64,768],[b,64]
        batch.caption_emb = gl; batch.caption_token_emb = tok; batch.caption_token_mask = msk
        batch.has_text = torch.ones(len(items), dtype=torch.bool, device=dev)
        gt_Ts = [min(args.num_frames, int(it["num_frames"])) for it in items]
        # one T_lat for the batch = max needed (per-clip frame_mask_lat from prepare_skeleton_only
        # marks each clip's valid latent frames, so shorter clips are masked, not mis-generated).
        T_lat = min(int(getattr(flow, "max_T_lat", 75)), max(1, max((g + stride - 1) // stride for g in gt_Ts)))
        meta = tokenizer.prepare_skeleton_only(batch, T_lat)
        C = meta["coarse_mask"].shape[1]
        # prepare_skeleton_only returns an ALL-TRUE frame_mask_lat (it assumes length==T_lat).
        # For a MIXED-LENGTH batch we MUST mask each clip's invalid latent frames, else a short
        # clip attends to its extra batch-max latent frames and its generated valid prefix
        # differs from B=1 (codex blocking fix). Build per-clip latent mask + rebuild token_mask.
        tlat_b = [min(T_lat, max(1, (g + stride - 1) // stride)) for g in gt_Ts]
        fml = torch.zeros(len(items), T_lat, dtype=torch.bool, device=dev)
        for bi, tl in enumerate(tlat_b):
            fml[bi, :tl] = True
        meta["frame_mask_lat"] = fml
        meta["token_mask"] = meta["coarse_mask"].unsqueeze(1) & fml.unsqueeze(-1)  # [B,T_lat,C]
        cond = {"text_global": batch.caption_emb.float(), "text_tokens": batch.caption_token_emb.float(),
                "text_token_mask": batch.caption_token_mask, "has_text": batch.has_text,
                "pooled_adjacency": meta["pooled_adjacency"].float(), "pooled_geodesic": meta["pooled_geodesic"].float(),
                "pooled_skeleton_embeddings": meta["pooled_skeleton_embeddings"].float(),
                "coarse_mask": meta["coarse_mask"], "frame_mask_lat": meta["frame_mask_lat"]}
        z_hat = flow.sample(cond, meta["token_mask"], T_lat, C, steps=args.steps, cfg_scale=args.cfg_scale)
        fake = type("B", (), {"joint_mask": batch.joint_mask})()
        cont = tokenizer.decode(z_hat, meta, fake)["pred_motion"].float()    # [b,Tc,J,13] CONTINUOUS
        gen_x = torch.zeros_like(batch.anytop_x)                             # [b,J,13,Tmax]
        for bi, gt_T in enumerate(gt_Ts):                                    # per-clip: place gen over its GT support
            if cont.shape[1] < gt_T:                                         # else zero tail masked-VALID -> biased
                raise SystemExit(f"[gen-eval] clip {bidx[bi]}: continuous decode T={cont.shape[1]} < gt_T={gt_T} "
                                 f"(T_lat={T_lat}, stride={stride}); would score zero frames as valid")
            gen_x[bi, :, :, :gt_T] = cont[bi, :gt_T].permute(1, 2, 0)        # [gt_T,J,13]->[J,13,gt_T]
        batch_gen = dataclasses.replace(batch, anytop_x=gen_x)              # frame_mask unchanged (per-clip gt_T valid)
        GE.append(core.encode_motion(batch_gen).float().cpu())
        GTE.append(core.encode_motion(batch).float().cpu())
        TE.append(core.encode_text(caps).float().cpu())
        objs.extend(str(it.get("object_type", "?")) for it in items)
        done += len(items)
        print(f"[gen-eval]   {done}/{len(idxs)} generated (B={len(items)} T_lat={T_lat})", flush=True)
    TE = torch.cat(TE); GE = torch.cat(GE); GTE = torch.cat(GTE); n = GE.shape[0]
    print(f"[gen-eval] embedded {n} (text/gen/gt)", flush=True)

    src = ["animo4d" if o.startswith("PZ_") else "truebones" for o in objs]
    report = {"n": n, "pool": args.pool, "steps": args.steps, "cfg_scale": args.cfg_scale,
              "flow_ckpt": args.flow_ckpt, "flow_epoch": fck.get("epoch"), "eval_ckpt": args.eval_ckpt,
              "num_frames": args.num_frames, "answer_form": "continuous"}

    def _rr(m):
        rr = m.get("rprec_text_to_gen")
        return f"R@1={rr[1]:.3f} R@2={rr[2]:.3f} R@3={rr[3]:.3f}" if rr else "R@k=NA(n<pool)"

    def _n(x):
        return f"{x:.3f}" if x is not None else "NA"

    report["overall"] = subset_metrics(TE, GE, GTE, args.pool, args.fid_min, args.max_div_pairs, gen)
    o = report["overall"]
    print(f"[gen-eval] OVERALL n={o['n']} text→gen {_rr(o)} match(mean/med)={_n(o['matching_mean'])}/{_n(o['matching_median'])} "
          f"FID={o['fid']} div(gen/gt)={_n(o['diversity_gen'])}/{_n(o['diversity_gt'])}", flush=True)
    report["per_subset"] = {}
    for s in ("animo4d", "truebones"):
        ii = [i for i, x in enumerate(src) if x == s]
        if not ii:
            continue
        t = torch.tensor(ii)
        m = subset_metrics(TE[t], GE[t], GTE[t], args.pool, args.fid_min, args.max_div_pairs, gen)
        report["per_subset"][s] = m
        print(f"[gen-eval] SUBSET {s:9s} n={m['n']:5d} text→gen {_rr(m)} match={_n(m['matching_mean'])} "
              f"FID={m['fid']} ({m['fid_note']}) div(gen/gt)={_n(m['diversity_gen'])}/{_n(m['diversity_gt'])}", flush=True)

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(report, indent=2))
        print(f"[gen-eval] report -> {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
