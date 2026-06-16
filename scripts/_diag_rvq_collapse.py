"""READ-ONLY diagnostic: is the ep300 Graph-VQVAE RVQ collapsed (stages 1-3 dead)
or is codebook_active=[512,0,0,0] a quantizer-dropout measurement artifact?

Loads the tokenizer in EVAL mode (full Q, NO dropout) and measures, on ~100-200
real L5 val clips:
  (1) per-stage UNIQUE-code counts + perplexity over all valid tokens
  (2) per-stage mean residual L2 norm (||r0||->||r1||->||r2||->||r3||->||r4_final||)
  (3) depth-1 vs full-Q world-pos recon L2 on the same clips

Mirrors scripts/export_graph_vq_tokens.py (eval(), encode(), quantizer with
allow_collectives=False) and scripts/animate_vqvae_recon.py (world-pos FK recon).
Does NOT modify any model/training code. Does NOT launch training.
"""
from __future__ import annotations

import argparse
import contextlib
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.anytop_dataset import (  # noqa: E402
    AnyTopDataset, collate_fn as anytop_collate_fn, _STD_FLOOR,
)
from src.models.graph_salad.batch import GraphMotionBatch  # noqa: E402
from src.models.vq_model import GraphVQTokenizer  # noqa: E402
from src.data.anytop_rot6d_fk import recover_from_bvh_rot_np  # noqa: E402 (official rot6d FK)


def load_vq(ckpt_path, dev):
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    ta = ck["args"]
    model = GraphVQTokenizer(
        d_model=ta["d_model"], n_heads=ta["n_heads"], d_ff=ta["d_ff"],
        n_graph_layers=ta["n_graph_layers"],
        n_enc_temporal_layers=ta["n_enc_temporal_layers"],
        n_pre_vq_layers=ta["n_pre_vq_layers"],
        n_post_vq_layers=ta["n_post_vq_layers"],
        n_cross_layers=ta["n_cross_layers"],
        n_dec_temporal_layers=ta["n_dec_temporal_layers"],
        max_coarse=ta["max_coarse"],
        temporal_stride=ta["temporal_stride"],
        temporal_kernel=ta["temporal_kernel"],
        dropout=ta["dropout"],
        code_dim=ta["code_dim"], num_codes=ta["num_codes"],
        num_quantizers=ta["num_quantizers"], ema_mu=ta["ema_mu"],
        quantize_dropout_prob=ta["quantize_dropout_prob"],
        dead_code_threshold=ta["dead_code_threshold"],
    ).to(dev)
    model.load_state_dict(ck["model_state_dict"], strict=True)
    model.eval()
    return model, ta, ck


@torch.no_grad()
def per_stage_rvq_stats(model, enc, dev):
    """Run the RVQ residual loop EXACTLY like the quantizer (no dropout, eval) and
    capture per-stage codes + residual norms over VALID tokens only."""
    h_lat = enc["h_lat"]
    token_mask = enc["token_mask"]
    B, T_lat, C, D = h_lat.shape
    z_flat = h_lat.reshape(-1, D).float()
    valid = token_mask.reshape(-1)
    residual = z_flat
    Q = model.num_quantizers
    out = {"stage_codes": [], "resid_norm_in": [], "z_snap": torch.zeros_like(z_flat)}
    # residual norm entering stage 0 = ||z_flat|| over valid tokens
    for qi, cb in enumerate(model.quantizer.codebooks):
        rn = residual[valid].pow(2).sum(dim=1).sqrt()  # per-valid-token L2 norm
        out["resid_norm_in"].append(rn.cpu())
        codes, q = cb.quantize(residual)  # [N],[N,D]
        out["stage_codes"].append(codes[valid].cpu())
        out["z_snap"] = out["z_snap"] + q
        residual = residual - q
    # final residual after all Q stages
    rn = residual[valid].pow(2).sum(dim=1).sqrt()
    out["resid_norm_in"].append(rn.cpu())  # index Q = post-stage-(Q-1)
    out["z_snap"] = out["z_snap"] * valid.unsqueeze(-1).to(out["z_snap"].dtype)
    out["n_valid"] = int(valid.sum().item())
    return out


@torch.no_grad()
def decode_at_depth(model, enc, batch, depth, dev, amp_dtype):
    """Decode using stage-0..depth-1 of the EVAL-mode RVQ (no dropout). depth in
    1..Q. Returns (pred_motion[B,T,J,13], frame_mask_recovered[B,T])."""
    h_lat = enc["h_lat"]
    token_mask = enc["token_mask"]
    B, T_lat, C, D = h_lat.shape
    z_flat = h_lat.reshape(-1, D).float()
    valid = token_mask.reshape(-1)
    residual = z_flat
    z_snap = torch.zeros_like(z_flat)
    for qi, cb in enumerate(model.quantizer.codebooks):
        if qi >= depth:
            break
        codes, q = cb.quantize(residual)
        z_snap = z_snap + q
        residual = residual - q
    z_snap = z_snap * valid.unsqueeze(-1).to(z_snap.dtype)
    z_q = z_snap.reshape(B, T_lat, C, D)
    amp_ctx = (torch.autocast("cuda", dtype=torch.bfloat16)
               if amp_dtype == "bf16" else contextlib.nullcontext())
    with amp_ctx:
        dec = model.decode(z_q, enc, batch)
    return dec["pred_motion"], dec["frame_mask_recovered"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--ckpt_ep89", default=None,
                    help="optional second ckpt for regression reality check")
    ap.add_argument("--anytop_root", default="data/animo4d_anytop_clean_L5")
    ap.add_argument("--split", default="val")
    ap.add_argument("--max_clips", type=int, default=160)
    ap.add_argument("--n_recon", type=int, default=12,
                    help="#clips for depth1-vs-fullQ recon comparison")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    dev = torch.device(args.device)
    model, ta, ck = load_vq(args.ckpt, dev)
    Q = model.num_quantizers
    K = model.quantizer.num_codes
    amp_dtype = ta.get("amp_dtype", "bf16")
    print(f"[ckpt] {args.ckpt}")
    print(f"  epoch={ck.get('epoch')} val_total={ck.get('val_total'):.4f} "
          f"codebook_active(FIELD)={ck.get('codebook_active')}  Q={Q} K={K} amp={amp_dtype}")
    print(f"  model.training={model.training}  quantize_dropout_prob={model.quantizer.quantize_dropout_prob}")

    ds = AnyTopDataset(split=args.split,
                       num_frames=ta["max_frames"], max_joints=ta["max_joints"],
                       val_frac=ta["val_frac"], load_captions=False,
                       data_root=args.anytop_root)
    n = min(len(ds), args.max_clips)
    print(f"  {args.split} dataset size={len(ds)}  using {n} clips")

    # accumulate per-stage code sets + residual norms + counts across clips
    stage_code_sets = [set() for _ in range(Q)]
    stage_counts = [defaultdict(int) for _ in range(Q)]  # code -> count for perplexity
    resid_norm_sums = [0.0 for _ in range(Q + 1)]
    total_valid = 0

    # recon L2 per depth, per clip (a subset for the recon comparison)
    recon_rows = []  # {sp, J, l2_depth1, l2_fullQ}

    for i in range(n):
        item = ds[i]
        raw = anytop_collate_fn([item])
        raw = {k: v.to(dev) if torch.is_tensor(v) else v for k, v in raw.items()}
        batch = GraphMotionBatch.from_collate_dict(raw)
        amp_ctx = (torch.autocast("cuda", dtype=torch.bfloat16)
                   if amp_dtype == "bf16" else torch.autocast("cuda", enabled=False))
        with torch.no_grad():
            with amp_ctx:
                enc = model.encode(batch)
        st = per_stage_rvq_stats(model, enc, dev)
        nv = st["n_valid"]
        total_valid += nv
        for qi in range(Q):
            codes = st["stage_codes"][qi].numpy()
            stage_code_sets[qi].update(codes.tolist())
            uniq, cnts = np.unique(codes, return_counts=True)
            for u, c in zip(uniq.tolist(), cnts.tolist()):
                stage_counts[qi][u] += c
            resid_norm_sums[qi] += float(st["resid_norm_in"][qi].sum().item())
        resid_norm_sums[Q] += float(st["resid_norm_in"][Q].sum().item())

        # recon L2 at depth 1 and full-Q for the first N_RECON clips (world-pos FK).
        # De-norm + FK EXACTLY as scripts/animate_vqvae_recon.py (apples-to-apples).
        if i < args.n_recon:
            J = int(item["num_joints"])
            sp = str(item["object_type"])
            parents = [int(p) for p in item["parent_indices"][:J]]
            offsets = np.asarray(item["rest_offsets"])[:J]
            std = raw["anytop_std"][0, :J].cpu().numpy()    # [J,13]
            mean = raw["anytop_mean"][0, :J].cpu().numpy()  # [J,13]
            # GT raw 13ch: anytop_x is [J,13,T] -> [T,J,13].
            gt_norm_full = np.asarray(item["anytop_x"]).transpose(2, 0, 1)  # [T,J,13]
            row = {"sp": sp, "J": J}
            for depth in (1, Q):
                pm, fmr = decode_at_depth(model, enc, batch, depth, dev, amp_dtype)
                T_clip = int(item["num_frames"])
                T_valid = int(fmr[0].sum().item())
                T = min(T_clip, T_valid)
                pred_norm = pm[0, :T, :J, :].float().cpu().numpy()  # [T,J,13]
                pred_raw = pred_norm * (std[None] + _STD_FLOOR) + mean[None]
                gt_raw = gt_norm_full[:T, :J, :] * (std[None] + _STD_FLOOR) + mean[None]
                pred_world = recover_from_bvh_rot_np(pred_raw, parents, offsets)
                gt_world = recover_from_bvh_rot_np(gt_raw, parents, offsets)
                row[f"l2_depth{depth}" if depth == 1 else "l2_fullQ"] = \
                    float(np.linalg.norm(pred_world - gt_world, axis=-1).mean())
            recon_rows.append(row)

    print("\n================ (1) PER-STAGE UNIQUE-CODE COUNTS + PERPLEXITY ================")
    print(f"  total valid tokens across {n} clips = {total_valid}")
    for qi in range(Q):
        uniq = len(stage_code_sets[qi])
        counts = np.array(list(stage_counts[qi].values()), dtype=np.float64)
        p = counts / counts.sum()
        ppl = float(np.exp(-(p * np.log(p + 1e-12)).sum()))
        # top-1 code share (collapse signature: ~1 code dominates)
        top1 = float(counts.max() / counts.sum()) if counts.size else 0.0
        print(f"  stage {qi}: unique_codes={uniq:4d}/{K}   perplexity={ppl:8.2f}   "
              f"top1_share={top1:.3f}")

    print("\n================ (2) PER-STAGE MEAN RESIDUAL L2 NORM (over valid tokens) =======")
    if total_valid > 0:
        for qi in range(Q):
            print(f"  ||r_in[{qi}]|| (entering stage {qi}) = {resid_norm_sums[qi]/total_valid:.4f}")
        print(f"  ||r_final||  (after stage {Q-1})       = {resid_norm_sums[Q]/total_valid:.4f}")

    print("\n================ (3) DEPTH-1 vs FULL-Q WORLD-POS RECON L2 ======================")
    print(f"  {'species':32s} {'J':>3s} {'L2_depth1':>10s} {'L2_fullQ':>10s} {'Δ(d1-fullQ)':>12s}")
    d1s, fqs = [], []
    for r in recon_rows:
        d1, fq = r["l2_depth1"], r["l2_fullQ"]  # noqa: F841
        d1s.append(d1); fqs.append(fq)
        print(f"  {r['sp'][:32]:32s} {r['J']:>3d} {d1:>10.4f} {fq:>10.4f} {d1-fq:>12.4f}")
    if d1s:
        print(f"  MEAN: depth1={np.nanmean(d1s):.4f}  fullQ={np.nanmean(fqs):.4f}  "
              f"improvement(fullQ vs depth1)={np.nanmean(d1s)-np.nanmean(fqs):.4f}")


if __name__ == "__main__":
    main()
