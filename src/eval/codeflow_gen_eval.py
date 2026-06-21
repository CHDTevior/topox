"""Shared Graph-CodeFlow TEXT->MOTION gen-eval in the FROZEN evaluator's space.

Single source of truth for the generation+embedding+metrics loop, imported by BOTH:
  - scripts/_eval_codeflow_gen_in_evalspace.py  (offline CLI snapshot eval)
  - the backbone trainer's ONLINE eval hook (train_graph_codeflow.py do_val block)
so online == offline (no drift). The generated "answer" is the CONTINUOUS-decoded
motion (user 2026-06-17: backbone answer = continuous, no codebook snap).

Differences vs an inline offline-only version (all required for the online hook):
  - per-subset split is by motion_id (HML3D->human / PZ_->animal / else->truebones),
    NOT the old object_type PZ_ heuristic (object_type is None for human clips).
  - NO SystemExit: a clip whose continuous decode is shorter than its GT length is
    SOFT-CLAMPED (score the available prefix + trim its frame_mask) and logged, never
    raised — an in-loop eval must never abort the training job.
  - RNG save/restore around the whole pass: flow.sample draws from the GLOBAL torch RNG
    (flow.py:268), so without this an in-loop eval would perturb the trainer's
    dropout/noise stream every time it runs.
"""
from __future__ import annotations
import dataclasses
import numpy as np
import torch

from src.data.anytop_t2m_eval_dataset import collate_fn
from src.models.graph_salad.batch import GraphMotionBatch


# ---- metrics (moved verbatim from scripts/_eval_codeflow_gen_in_evalspace.py) ----
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


def motion_id_bucket(mid: str) -> str:
    """Per-subset bucket from motion_id (object_type is None for human -> unusable).
    HML3D* -> human ; PZ_* -> animal ; else -> truebones (covers both L4safeHuman and
    the older mergedL4TB datasets)."""
    s = str(mid)
    if s.upper().startswith("HML3D"):
        return "human"
    if s.startswith("PZ_"):
        return "animal"
    return "truebones"


@torch.no_grad()
def run_gen_eval(*, flow, tokenizer, core, t5_encode_batch, ds, idxs, dev, stride,
                 pool=32, steps=25, cfg_scale=4.0, num_frames=300, gen_batch=32,
                 fid_min=1024, max_div_pairs=20000, seed=42, log=print):
    """Generate text->motion for ds[idxs], decode continuous, embed with the frozen
    evaluator, return a report dict {n, overall, per_subset{animal/human/...}}.

    flow: GraphCodeFlow (UNWRAPPED — caller passes raw_flow under DDP). tokenizer: frozen
    GraphVQTokenizer. core: frozen AnyTopT2MEvaluator. t5_encode_batch: callable
    list[str]->(global[b,768], tokens[b,L,768], mask[b,L]). ds: AnyTopT2MEvalDataset.
    Restores the global RNG state on exit so the caller's training stream is untouched."""
    cpu_state = torch.get_rng_state()
    cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    try:
        torch.manual_seed(seed)                       # reproducible flow.sample (global torch.randn)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        metric_gen = torch.Generator().manual_seed(seed)

        TE, GE, GTE, mids = [], [], [], []
        B = max(1, gen_batch)
        done = 0
        for bstart in range(0, len(idxs), B):
            bidx = idxs[bstart:bstart + B]
            items = [ds[di] for di in bidx]
            caps = [it.get("caption_text") or "" for it in items]
            coll = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in collate_fn(items).items()}
            batch = GraphMotionBatch.from_collate_dict(coll)                  # collate pads J/T over the batch
            gl, tok, msk = t5_encode_batch(caps)                             # [b,768],[b,L,768],[b,L]
            batch.caption_emb = gl; batch.caption_token_emb = tok; batch.caption_token_mask = msk
            batch.has_text = torch.ones(len(items), dtype=torch.bool, device=dev)  # all-True for CFG at eval
            gt_Ts = [min(num_frames, int(it["num_frames"])) for it in items]
            # one T_lat for the batch = max needed; per-clip frame_mask_lat masks shorter clips.
            T_lat = min(int(getattr(flow, "max_T_lat", 75)),
                        max(1, max((g + stride - 1) // stride for g in gt_Ts)))
            meta = tokenizer.prepare_skeleton_only(batch, T_lat)
            C = meta["coarse_mask"].shape[1]
            # prepare_skeleton_only returns ALL-TRUE frame_mask_lat (assumes length==T_lat).
            # For a MIXED-LENGTH batch we MUST mask each clip's invalid latent frames + rebuild
            # token_mask, else a short clip's generated prefix differs from B=1 (codex blocking fix).
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
            z_hat = flow.sample(cond, meta["token_mask"], T_lat, C, steps=steps, cfg_scale=cfg_scale)
            fake = type("B", (), {"joint_mask": batch.joint_mask})()
            cont = tokenizer.decode(z_hat, meta, fake)["pred_motion"].float()   # [b,Tc,J,13] CONTINUOUS
            gen_x = torch.zeros_like(batch.anytop_x)                            # [b,J,13,Tmax]
            clamp_fm = None                                                     # built lazily iff a clip is clamped
            for bi, gt_T in enumerate(gt_Ts):
                t_place = min(int(cont.shape[1]), gt_T)                         # SOFT CLAMP (never SystemExit)
                if t_place < gt_T:
                    if clamp_fm is None:
                        clamp_fm = batch.frame_mask.clone()
                    clamp_fm[bi, t_place:] = False                              # don't score zero tail as valid
                    log(f"[gen-eval] WARN clip {bidx[bi]}: decode T={cont.shape[1]} < gt_T={gt_T}; "
                        f"scoring first {t_place} frames (T_lat={T_lat}, stride={stride})")
                gen_x[bi, :, :, :t_place] = cont[bi, :t_place].permute(1, 2, 0)  # [t,J,13]->[J,13,t]
            repl = {"anytop_x": gen_x}
            if clamp_fm is not None:
                repl["frame_mask"] = clamp_fm                                   # gen frame validity trimmed for clamped clips
            batch_gen = dataclasses.replace(batch, **repl)
            GE.append(core.encode_motion(batch_gen).float().cpu())
            GTE.append(core.encode_motion(batch).float().cpu())
            TE.append(core.encode_text(caps).float().cpu())
            mids.extend(str(ds._plan[di][1]["motion_id"]) for di in bidx)
            done += len(items)
            log(f"[gen-eval]   {done}/{len(idxs)} generated (B={len(items)} T_lat={T_lat})")

        TE = torch.cat(TE); GE = torch.cat(GE); GTE = torch.cat(GTE)
        n = GE.shape[0]
        report = {"n": n, "pool": pool, "steps": steps, "cfg_scale": cfg_scale,
                  "num_frames": num_frames, "answer_form": "continuous"}
        report["overall"] = subset_metrics(TE, GE, GTE, pool, fid_min, max_div_pairs, metric_gen)
        buckets = [motion_id_bucket(m) for m in mids]
        report["per_subset"] = {}
        for s in ("animal", "human", "truebones"):
            ii = [i for i, x in enumerate(buckets) if x == s]
            if not ii:
                continue
            t = torch.tensor(ii)
            report["per_subset"][s] = subset_metrics(TE[t], GE[t], GTE[t], pool, fid_min, max_div_pairs, metric_gen)
        return report
    finally:
        torch.set_rng_state(cpu_state)
        if cuda_state is not None:
            torch.cuda.set_rng_state_all(cuda_state)
