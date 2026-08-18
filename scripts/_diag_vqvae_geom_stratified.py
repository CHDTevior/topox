#!/usr/bin/env python3
"""Evaluator-FREE geometric reconstruction diagnostic for the frozen Graph-VQVAE,
stratified by topology distance.

WHY THIS EXISTS (and why it is not the evaluator-space recon script):
  Hypothesis A -- "the frozen tokenizer cannot represent unseen topologies, so it caps
  what the flow can ever generate" -- is a GEOMETRIC claim. The project's own prior
  evidence for it is geometric (animal-only VQVAE on human: position L2 0.14 vs 0.027
  in-distribution, rot6d->FK 0.30 vs 0.064). Testing it through text->motion retrieval in
  a non-compliant evaluator's embedding space is the wrong instrument on three counts:
  a pool-32 caption retrieval still matches a Crab reconstructed as a twitching blob
  ("an animal moves forward"); recon is already known to be text-align-lossless in
  distribution (F ~ 0.99), so the metric is pinned at its ceiling with no downward range;
  and it routes a tokenizer question through the one component known to be broken.
  So: no evaluator here at all. Pure geometry, directly comparable to the prior 5x result.

STRATIFICATION: `motion_id_bucket` (PZ_* -> animal, else -> truebones) is a DATASET-SOURCE
  axis, not a topology-distance axis, and using it hides the question. held_stress has 580
  clips at dist>1.5 but only ~125 truebones-bucket clips, so >=455 of the hardest clips sit
  inside "animal" mixed with near-duplicates. This script buckets by
  `dist_to_nearest_retained` from protocol/holdout_topologies_v1.json (A<0.1, B 0.1-0.5,
  C 0.5-1.5, D >1.5) and also reports per-species, so the "no-excuse" A stratum
  (near-identical topology AND abundant data) is finally its own number.

INTEGRITY GATE: for every clip we recover world positions TWICE from the same GT -- once by
  rot6d FK and once from the RIC position channels. A correct de-norm + recovery makes these
  agree. If the aggregate self-check is not ~0 the de-norm/FK path is wrong and EVERY number
  in the report is meaningless, so we fail loudly rather than emit it.

Writes per-clip records (for bootstrap CIs over clips -- NOT pool-reshuffle noise) plus
per-stratum and per-species aggregates.
"""
import argparse, json, sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.anytop_dataset import _recover_world_positions, _STD_FLOOR   # noqa: E402
from src.data.anytop_rot6d_fk import recover_from_bvh_rot_np               # noqa: E402  official rot6d FK
from src.data.anytop_t2m_eval_dataset import AnyTopT2MEvalDataset, collate_fn  # noqa: E402
from src.models.graph_salad.batch import GraphMotionBatch                  # noqa: E402


def _load_vq():
    """Same loader the other eval scripts use: dynamically load animate_vqvae_recon.py and
    return its load_vq_tokenizer. src.models.vq_model exports GraphVQTokenizer /
    semantic_config_from_ckpt but NOT a ready-made loader, so importing one from there
    fails at runtime (codex r1 #2 on this script)."""
    import importlib.util
    q = Path(__file__).resolve().parent / "animate_vqvae_recon.py"
    spec = importlib.util.spec_from_file_location("_avr_loader", q)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod.load_vq_tokenizer


def strata_of(dist):
    if dist is None:   return "unknown"
    if dist < 0.1:     return "A_near_identical"
    if dist < 0.5:     return "B_close"
    if dist < 1.5:     return "C_mid"
    return "D_far"


def build_species_meta(holdout_json):
    """object_type -> {dist, J, train_clips, bucket, source}. Held topologies only;
    retained species resolve to dist=None -> stratum 'retained'."""
    ht = json.load(open(holdout_json))
    m = {}
    for t in ht["held_out_trees"]:
        for ot in t["object_types"]:
            m[ot] = {"dist": t.get("dist_to_nearest_retained"), "J": t["J"],
                     "train_clips": t["train_clips"], "bucket": t["bucket"],
                     "source": t["source"], "nearest": t.get("nearest_retained_example")}
    return m


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vqvae_ckpt", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--holdout_json", default="protocol/holdout_topologies_v1.json")
    ap.add_argument("--base_split", default="all", choices=["train", "val", "all"],
                    help="held_* manifests draw from BOTH original partitions -> must be 'all'.")
    ap.add_argument("--num_frames", type=int, default=300)
    ap.add_argument("--max_joints", type=int, default=None, help="default = VQVAE ckpt's max_joints")
    ap.add_argument("--encode_batch", type=int, default=8)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--max_clips", type=int, default=0, help="SMOKE: cap clips (0 = all)")
    ap.add_argument("--selfcheck_tol", type=float, default=1e-3,
                    help="abort if mean GT FK-vs-RIC disagreement exceeds this. Measured on the "
                         "current fixed path: 3.6e-08..5.9e-07 (renders/*/recon_summary.txt), so "
                         "1e-3 already allows ~1000x headroom; the original 5e-2 would have waved "
                         "through a 10^5x error in the de-norm/FK path.")
    ap.add_argument("--out", required=True)
    return ap.parse_args()


@torch.no_grad()
def main():
    args = parse_args()
    outp = Path(args.out).resolve()
    if "scratch" not in outp.parts:
        raise SystemExit(f"[geom] --out must live under scratch/ (disposable, outside any data "
                         f"root or run dir); got {outp}")
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    vqvae, ta, vck = _load_vq()(args.vqvae_ckpt, dev)
    stride = int(ta["temporal_stride"])
    # NOTE: max_coarse is the SPATIAL cap (coarse joint groups; 96 was chosen to keep Dragon's
    # 142 joints). It is NOT a frame budget -- capping time with max_coarse*stride is a category
    # error (codex). Full-length inference at 300 frames is the project's live assumption
    # (T_fine_max=300, stride 4 -> T_lat 75), and the token cache was exported at exactly that.
    nf = int(args.num_frames)
    mj = int(args.max_joints) if args.max_joints else int(ta.get("max_joints", 144))
    amp = (ta.get("amp_dtype", "bf16") == "bf16") and dev.type == "cuda"

    sem = ta.get("joint_semantics")
    if sem is not None and not Path(sem).exists():
        raise SystemExit(f"[geom] ckpt declares joint_semantics={sem} but the file is missing")

    ds = AnyTopT2MEvalDataset(manifest_path=args.manifest, data_root=args.data_root,
                              caption_emb_cache=None, split="val", base_split=args.base_split,
                              view="full", num_frames=nf, max_joints=mj,
                              **({"joint_semantics": sem} if sem is not None else {}))
    n_total = len(ds)
    print(f"[geom] VQVAE ep{vck.get('epoch','?')} | manifest {args.manifest} -> {n_total} clips "
          f"| base_split={args.base_split} nf={nf} mj={mj} semantics={sem}", flush=True)
    if args.max_clips and args.max_clips < n_total:
        ds = torch.utils.data.Subset(ds, list(range(args.max_clips)))
        print(f"[geom] SMOKE: capped to {len(ds)} clips", flush=True)

    loader = DataLoader(ds, batch_size=args.encode_batch, shuffle=False,
                        num_workers=args.num_workers, collate_fn=collate_fn, drop_last=False)
    meta = build_species_meta(args.holdout_json)
    OTS = sorted(meta, key=len, reverse=True)          # longest-prefix species match

    def species_of(mid):
        for ot in OTS:
            if str(mid).startswith(ot):
                return ot
        return None

    recs, done = [], 0
    for coll in loader:
        cd = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in coll.items()}
        batch = GraphMotionBatch.from_collate_dict(cd)

        def _recon():
            enc = vqvae.encode(batch)
            z = vqvae.nearest_residual_ids(enc["h_lat"], enc["token_mask"])["z_snap"]
            return vqvae.decode(z, enc, batch)
        if amp:
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                out = _recon()
        else:
            out = _recon()

        # Read everything off GraphMotionBatch (a dataclass with declared fields) rather than
        # guessing collate keys: batch.anytop_x is [B,J,13,T], num_joints not n_joints.
        for _f in ("anytop_x", "anytop_mean", "anytop_std"):
            if getattr(batch, _f, None) is None:
                raise SystemExit(f"[geom] batch.{_f} is None -- the dataset did not attach the "
                                 f"raw 13-channel motion / de-norm stats; every metric would be wrong.")
        pred_all = out["pred_motion"].float().cpu().numpy()                    # [B,T,J,13]
        eff = (batch.frame_mask & out["frame_mask_recovered"].bool()).cpu().numpy()
        gt_all = batch.anytop_x.float().cpu().numpy().transpose(0, 3, 1, 2)    # [B,J,13,T]->[B,T,J,13]
        mean_all = batch.anytop_mean.float().cpu().numpy()                     # [B,J,13]
        std_all = batch.anytop_std.float().cpu().numpy()
        rest_all = batch.rest_offsets.float().cpu().numpy()                    # [B,J,3]

        for b in range(pred_all.shape[0]):
            mid = str(batch.motion_id[b])
            T = int(eff[b].sum())
            J = int(batch.num_joints[b])
            if T < 4 or J < 2:
                continue
            mean, std = mean_all[b, :J], std_all[b, :J]
            pred_raw = pred_all[b, :T, :J, :] * (std[None] + _STD_FLOOR) + mean[None]
            gt_raw = gt_all[b, :T, :J, :] * (std[None] + _STD_FLOOR) + mean[None]
            parents = [int(p) for p in batch.parent_indices[b][:J]]
            offsets = rest_all[b, :J]

            pred_w = recover_from_bvh_rot_np(pred_raw, parents, offsets)       # [T,J,3]
            gt_w = recover_from_bvh_rot_np(gt_raw, parents, offsets)
            gt_ric = _recover_world_positions(gt_raw)

            selfcheck = float(np.linalg.norm(gt_w - gt_ric, axis=-1).mean())
            recon_l2 = float(np.linalg.norm(pred_w - gt_w, axis=-1).mean())
            pos_l2 = float(np.linalg.norm(pred_raw[..., 0:3] - gt_raw[..., 0:3], axis=-1).mean())
            rot_mse = float(((pred_raw[..., 3:9] - gt_raw[..., 3:9]) ** 2).mean())
            g_spd = float(np.linalg.norm(np.diff(gt_w, axis=0), axis=-1).mean())
            p_spd = float(np.linalg.norm(np.diff(pred_w, axis=0), axis=-1).mean())
            # jitter = mean |2nd difference|, i.e. frame-to-frame acceleration magnitude
            g_jit = float(np.linalg.norm(np.diff(gt_w, n=2, axis=0), axis=-1).mean())
            p_jit = float(np.linalg.norm(np.diff(pred_w, n=2, axis=0), axis=-1).mean())

            sp = species_of(mid)
            info = meta.get(sp) if sp else None
            recs.append({"motion_id": mid, "species": sp, "T": T, "J": J,
                         "dist": (info or {}).get("dist"),
                         "stratum": strata_of((info or {}).get("dist")) if info else "retained",
                         "train_clips": (info or {}).get("train_clips"),
                         "recon_l2": recon_l2, "pos_l2": pos_l2, "rot6d_mse": rot_mse,
                         "speed_ratio": p_spd / max(g_spd, 1e-9),
                         "jitter_ratio": p_jit / max(g_jit, 1e-9),
                         "gt_selfcheck_l2": selfcheck})
        done += pred_all.shape[0]
        if done % 200 < args.encode_batch:
            print(f"[geom]   {done}/{len(ds)}", flush=True)

    if not recs:
        raise SystemExit("[geom] no clips produced records")

    sc = float(np.mean([r["gt_selfcheck_l2"] for r in recs]))
    print(f"[geom] GT self-check (FK vs RIC) mean = {sc:.6f}", flush=True)
    if not np.isfinite(sc) or sc > args.selfcheck_tol:
        raise SystemExit(f"[geom] ABORT: GT self-check {sc:.4f} > tol {args.selfcheck_tol}. "
                         f"The de-norm/FK path is wrong -> every metric in this run is invalid.")

    def agg(rs):
        if not rs: return None
        a = {"n": len(rs)}
        for k in ("recon_l2", "pos_l2", "rot6d_mse", "speed_ratio", "jitter_ratio"):
            v = np.array([r[k] for r in rs], dtype=float)
            a[k] = {"mean": float(v.mean()), "median": float(np.median(v)),
                    "p90": float(np.percentile(v, 90)), "std": float(v.std(ddof=1)) if len(v) > 1 else 0.0}
        a["train_clips_median"] = float(np.median([r["train_clips"] for r in rs
                                                   if r["train_clips"] is not None])) \
            if any(r["train_clips"] is not None for r in rs) else None
        return a

    strata = sorted({r["stratum"] for r in recs})
    report = {"manifest": args.manifest, "vqvae_ckpt": args.vqvae_ckpt,
              "vqvae_epoch": vck.get("epoch"), "base_split": args.base_split,
              "n": len(recs), "num_frames": nf, "max_joints": mj,
              "joint_semantics": sem, "gt_selfcheck_mean": sc,
              "selfcheck_tol": args.selfcheck_tol,
              "note": ("evaluator-FREE geometric recon. recon_l2 = mean world-position L2 between "
                       "rot6d-FK(recon) and rot6d-FK(GT); speed/jitter ratios are pred/GT (1.0 = ideal). "
                       "Stratified by dist_to_nearest_retained, NOT by dataset source."),
              "OVERALL": agg(recs),
              "by_stratum": {s: agg([r for r in recs if r["stratum"] == s]) for s in strata},
              "by_species": {sp: agg([r for r in recs if r["species"] == sp])
                             for sp in sorted({r["species"] for r in recs if r["species"]})}}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2))
    Path(args.out + ".perclip.json").write_text(json.dumps(recs))
    print(f"[geom] report -> {args.out}  (+ .perclip.json for bootstrap CIs)", flush=True)
    for s in strata:
        a = report["by_stratum"][s]
        if a:
            print(f"[geom] {s:<18} n={a['n']:>5}  recon_L2={a['recon_l2']['mean']:.4f} "
                  f"pos_L2={a['pos_l2']['mean']:.4f} rot6d={a['rot6d_mse']['mean']:.4f} "
                  f"speed={a['speed_ratio']['mean']:.3f} jitter={a['jitter_ratio']['mean']:.3f}", flush=True)


if __name__ == "__main__":
    main()
