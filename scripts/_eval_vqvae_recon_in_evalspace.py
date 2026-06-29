"""Graph-VQVAE reconstruction quality in the FROZEN evaluator's embedding space
(spec gate: "VQVAE reconstruction in evaluator space ≈ GT").

Loads TWO independent frozen models (NO shared weights):
  • Graph-VQVAE  : GraphVQTokenizer via the canonical `load_vq_tokenizer` imported
                   from scripts/animate_vqvae_recon.py (single source of truth — the
                   VQVAE constructor is bug-prone, never copied). Use the n512 ckpt
                   trained on the SAME data root + max_joints=144 as the evaluator,
                   so the 13ch normalization aligns and the recon can be embedded
                   directly (no re-normalization).
  • Evaluator    : AnyTopT2MEvaluator.encode_motion (motion tower only; no text).

Per val motion (VQVAE regime: max_frames=64, J144, same data root): reconstruct
→ out['pred_motion'] [B,T,J,13] (NORMALIZED). Embed GT and recon via the evaluator
over the SAME effective frame support (GT frame_mask ∩ frame_mask_recovered) so the
comparison is apples-to-apples (the recon batch reuses every GT graph field via
dataclasses.replace; only anytop_x + frame_mask change). Report OVERALL + per-subset
(animo4d vs truebones — truebones is the weak subset, see the evaluator sanity gate):
  • pair cosine(GT_emb, recon_emb): mean / median (1 = recon identical in eval space)
  • recon→GT retrieval R@1/2/3 (pool=--pool, diagonal target — each val motion unique)
  • FID in eval space (best-effort; skipped when n < --fid_min, e.g. truebones n=78)
  • diversity: mean pairwise L2 within GT vs within recon (mode-collapse check)
  • norm-space recon MSE (masked) — cheap raw cross-check vs eval_graph_vae.py
Single-GPU, frozen, no grad. NEW code → codex review before trusting numbers.
"""

from __future__ import annotations

import argparse
import dataclasses
import importlib.util
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.anytop_dataset import AnyTopDataset, collate_fn, _STD_FLOOR
from src.models.graph_salad.batch import GraphMotionBatch
from src.models.graph_salad.t2m_evaluator import AnyTopT2MEvaluator
from src.models.graph_salad.world_recovery import recover_world_positions_torch
from src.models.graph_salad.rot6d_fk_recovery import recover_rot6d_fk_positions_torch


def _import_load_vq_tokenizer():
    """Import the canonical VQVAE loader from animate_vqvae_recon.py (has a __main__
    guard, so importing does NOT run its main)."""
    p = Path(__file__).resolve().parent / "animate_vqvae_recon.py"
    spec = importlib.util.spec_from_file_location("_avr_loader", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.load_vq_tokenizer


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--vqvae_ckpt", required=True, help="GraphVQTokenizer ckpt (n512, same data/J144 as evaluator).")
    ap.add_argument("--eval_ckpt", required=True, help="frozen AnyTopT2MEvaluator best_model.pt.")
    ap.add_argument("--split", default="val")
    ap.add_argument("--num_frames", type=int, default=None,
                    help="FULL-LENGTH eval regime: load/reconstruct/embed at this many frames "
                         "(default = the evaluator's training num_frames, the regime it MEASURES in). "
                         "The VQVAE is trained at max_frames=64 but RUN FULL-LENGTH at inference "
                         "(arch capacity = max_coarse * temporal_stride). Must not exceed that cap.")
    ap.add_argument("--pool", type=int, default=32, help="recon->GT retrieval pool size.")
    ap.add_argument("--encode_batch", type=int, default=32)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--amp_dtype", choices=["fp32", "bf16"], default=None,
                    help="VQVAE forward autocast; default = the ckpt's trained regime (bf16).")
    ap.add_argument("--exclude_truebones", action="store_true",
                    help="restrict to animo4d (object_type startswith PZ_); drop truebones — the "
                         "evaluator is weak on the scarce truebones (sanity R@1 0.48 vs animo4d 0.96).")
    ap.add_argument("--zero_contact", action="store_true",
                    help="DIAGNOSTIC: zero the contact channel (ch12) in BOTH GT and recon before "
                         "encode_motion -> isolate the MOTION-only (12ch) eval-space similarity. If cosine "
                         "recovers high, the low full-13ch cosine was contact-channel pollution, not motion.")
    ap.add_argument("--continuous_recon", action="store_true",
                    help="decode the PRE-VQ continuous latent h_lat (skip RVQ residual-snap) instead of "
                         "the quantized z_snap. The user's backbone answer is continuous, so this isolates "
                         "DECODER loss from QUANTIZATION loss: continuous>>quantized cos ⇒ the RVQ snap is the culprit.")
    ap.add_argument("--gt_as_recon", action="store_true",
                    help="CONTROL/calibration: use GT itself as the 'reconstruction' (perfect recon, "
                         "VQVAE bypassed) -> expect cosine~1.0, recon->GT R@1~1.0, FID~0, equal "
                         "diversity, norm-MSE 0. Validates the eval-space pipeline ceiling + calibrates "
                         "how far the real VQVAE recon is from perfect.")
    ap.add_argument("--mpjpe", action="store_true",
                    help="ALSO report MPJPE (mean per-joint position error) in metric/de-normalized "
                         "units, split human(HML*) vs animal(PZ*). De-normalizes both GT & recon 13ch "
                         "(raw=norm*(std+floor)+mean) then recovers world positions via the src torch "
                         "recovery (same path as the QA renderer); masked Euclidean over the effective "
                         "frame support ∩ valid joints. With --gt_as_recon this must be ~0 (self-check).")
    ap.add_argument("--fk", action="store_true",
                    help="With --mpjpe, ALSO report rot6d-FK MPJPE: recover joint positions by forward "
                         "kinematics from the 6D rotation channels (ch3:9) along the skeleton "
                         "(recover_rot6d_fk_positions_torch, parents+rest_offsets from the batch) instead "
                         "of the RIC/position route. Reports rot6d-FK pose vs TRUE GT positions "
                         "(rc_fk vs gt_w) PLUS the recon-independent FK-route floor (gt_fk vs gt_w). "
                         "rot6d is the weakest channel. Self-check: --gt_as_recon -> position ~0 and "
                         "fk == fk_floor (recon=GT makes rc_fk=gt_fk).")
    ap.add_argument("--fk_sibling_avg", action="store_true",
                    help="With --fk, ALSO compute rot6d-FK MPJPE using SIBLING-AVERAGED parent "
                         "rotation (mean 6D over a parent's duplicated child slots) instead of the "
                         "last-child slot the FK recovery normally uses, PLUS the sibling 6D "
                         "dispersion (GT ~0 verifies the duplicate convention; recon>0 = model "
                         "diverges siblings). Decisive test: if sibling-avg FK << last-child FK, the "
                         "sibling-duplicate convention is the dominant human rot6d-FK error source.")
    ap.add_argument("--fid_min", type=int, default=1024, help="skip FID for a (sub)set smaller than this.")
    ap.add_argument("--max_div_pairs", type=int, default=20000, help="cap pairwise-diversity sample.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=None, help="JSON report path.")
    return ap.parse_args()


@torch.no_grad()
def rprecision_pool(q: torch.Tensor, g: torch.Tensor, ks=(1, 2, 3)):
    """recon→GT retrieval R@k for ONE pool. q=query [P,D], g=gallery [P,D] (L2-normed,
    rows aligned: query i ↔ gallery i). Diagonal-only target (every val motion is unique,
    so the only acceptable match for recon i is its own GT i)."""
    sim = q @ g.t()                                      # [P,P]
    correct = torch.zeros_like(sim, dtype=torch.bool)
    correct.fill_diagonal_(True)
    order = sim.argsort(dim=1, descending=True)
    return {k: correct.gather(1, order[:, :k]).any(dim=1).float().mean().item() for k in ks}


def pooled_retrieval(q: torch.Tensor, g: torch.Tensor, pool: int):
    """Average recon→GT R@k over dataset-order pools of `pool` (drop remainder)."""
    n = q.shape[0]
    npool = n // pool
    if npool == 0:
        return None, 0
    acc = {1: 0.0, 2: 0.0, 3: 0.0}
    for p in range(npool):
        s = slice(p * pool, (p + 1) * pool)
        r = rprecision_pool(q[s], g[s])
        for k in acc:
            acc[k] += r[k]
    return {k: acc[k] / npool for k in acc}, npool


def fid_score(x: np.ndarray, y: np.ndarray):
    """Fréchet distance between two [N,D] embedding sets. Returns (fid, note)."""
    try:
        from scipy.linalg import sqrtm
    except Exception:
        return None, "scipy unavailable"
    mu1, mu2 = x.mean(0), y.mean(0)
    c1 = np.cov(x, rowvar=False)
    c2 = np.cov(y, rowvar=False)
    covmean = sqrtm(c1 @ c2)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    diff = mu1 - mu2
    val = float(diff @ diff + np.trace(c1 + c2 - 2.0 * covmean))
    if not np.isfinite(val):
        return None, "non-finite"
    return max(val, 0.0), "ok"   # clamp tiny negative numerical results to 0


def mean_pairwise_l2(x: torch.Tensor, max_pairs: int, gen: torch.Generator):
    """Mean L2 over a random sample of distinct pairs (diversity / spread)."""
    n = x.shape[0]
    if n < 2:
        return None
    n_all = n * (n - 1) // 2
    if n_all <= max_pairs:
        d = torch.pdist(x)
        return float(d.mean().item())
    i = torch.randint(0, n, (max_pairs,), generator=gen)
    j = torch.randint(0, n, (max_pairs,), generator=gen)
    keep = i != j
    i, j = i[keep], j[keep]
    d = (x[i] - x[j]).norm(dim=-1)
    return float(d.mean().item())


def subset_metrics(emb_gt, emb_rec, pool, fid_min, max_div_pairs, gen):
    """All eval-space metrics for one (sub)set of aligned GT/recon embeddings."""
    n = emb_gt.shape[0]
    cos = (emb_gt * emb_rec).sum(-1)                     # both L2-normed → cosine
    rr, npp = pooled_retrieval(emb_rec, emb_gt, pool)    # query=recon, gallery=GT
    m = {
        "n": n,
        "pair_cosine_mean": float(cos.mean().item()),
        "pair_cosine_median": float(cos.median().item()),
        "recon_to_gt_rprec": rr, "retrieval_pools": npp,
        "diversity_gt": mean_pairwise_l2(emb_gt, max_div_pairs, gen),
        "diversity_recon": mean_pairwise_l2(emb_rec, max_div_pairs, gen),
    }
    if n >= fid_min:
        f, note = fid_score(emb_gt.numpy(), emb_rec.numpy())
        m["fid"], m["fid_note"] = f, note
    else:
        m["fid"], m["fid_note"] = None, f"skipped (n={n} < fid_min={fid_min})"
    return m


@torch.no_grad()
def _sibling_avg_rot6d(raw, parent_indices, joint_mask):
    """Copy of raw [B,T,J,13] where, for every parent with >1 child, the 6D rotation (ch3:9)
    in ALL its child slots is set to the mean 6D over those siblings. AnyTop stores a parent's
    rotation duplicated into each child slot; FK recovery uses ONLY the LAST child slot, so if
    the model diverges the duplicates only one is used. Averaging makes siblings consistent."""
    out = raw.clone()
    B, T, J, _ = raw.shape
    for b in range(B):
        par = [int(p) for p in parent_indices[b]]
        Jb = min(len(par), J, int(joint_mask[b].sum().item()))
        kids = {}
        for j in range(1, Jb):
            p = par[j]
            if 0 <= p < Jb:
                kids.setdefault(p, []).append(j)
        for p, ks in kids.items():
            if len(ks) > 1:
                avg = out[b][:, ks, 3:9].mean(dim=1)            # [T,6]
                for k in ks:
                    out[b, :, k, 3:9] = avg
    return out


def _sibling_dispersion(raw, parent_indices, joint_mask, frame_valid):
    """Mean L2 deviation of each branching-parent child slot's 6D from its sibling-mean, over
    VALID (clip,frame,branch-child) — frame_valid [B,T] bool restricts to eff frames (codex fix:
    exclude padded/invalid-recovered frames). ~0 for GT (siblings duplicated); >0 if recon
    diverges them. Returns (sum, count)."""
    B, T, J, _ = raw.shape
    s = 0.0; c = 0
    for b in range(B):
        fm = frame_valid[b].bool()                              # [T]
        if int(fm.sum().item()) == 0:
            continue
        par = [int(p) for p in parent_indices[b]]
        Jb = min(len(par), J, int(joint_mask[b].sum().item()))
        kids = {}
        for j in range(1, Jb):
            p = par[j]
            if 0 <= p < Jb:
                kids.setdefault(p, []).append(j)
        for p, ks in kids.items():
            if len(ks) > 1:
                d6 = raw[b][:, ks, 3:9]                          # [T,K,6]
                dev = (d6 - d6.mean(dim=1, keepdim=True)).norm(dim=-1)  # [T,K]
                dev = dev[fm]                                   # [Tvalid,K] valid frames only
                s += float(dev.sum().item()); c += int(dev.numel())
    return s, c


def main() -> int:
    args = parse_args()
    if args.pool < 3:
        raise SystemExit(f"--pool must be >= 3 (R@3 needs a pool of >=3); got {args.pool}")
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gen = torch.Generator().manual_seed(args.seed)

    # ---- frozen VQVAE (canonical loader; rebuilds from ckpt args) ----
    load_vq_tokenizer = _import_load_vq_tokenizer()
    vqvae, ta, vck = load_vq_tokenizer(args.vqvae_ckpt, dev)
    amp_dtype = args.amp_dtype or ta.get("amp_dtype", "bf16")
    amp_enabled = (amp_dtype == "bf16") and dev.type == "cuda"
    print(f"[recon-eval] VQVAE {args.vqvae_ckpt} ep={vck.get('epoch')} "
          f"data_root={ta.get('data_root')} max_frames={ta.get('max_frames')} "
          f"max_joints={ta.get('max_joints')} num_codes={ta.get('num_codes')} amp={amp_dtype}", flush=True)
    vq_root = ta.get("anytop_root") or ta.get("data_root")

    # ---- frozen evaluator (motion tower) ----
    eck = torch.load(args.eval_ckpt, map_location="cpu")
    ea = eck.get("args", {})
    g = (lambda k, d: ea.get(k, d)) if isinstance(ea, dict) else (lambda k, d: getattr(ea, k, d))
    core = AnyTopT2MEvaluator(
        coemb_dim=g("coemb_dim", 512), text_tower=g("text_tower", "distilbert"),
        distilbert_path=g("distilbert_path", "checkpoints/text_encoders/distilbert-base-uncased"),
        text_max_length=g("text_max_length", 64),
        n_heads=g("n_heads", 8), d_ff=g("d_ff", 2048),
        n_graph_layers=g("n_graph_layers", 6), n_temporal_layers=g("n_temporal_layers", 4),
        motion_feat_dim=g("motion_feat_dim", 13),  # 12ch (contact-free) ckpt rebuilds at 12; old 13ch -> 13
        dropout=g("dropout", 0.1),
        learnable_temperature=not g("fixed_temperature", False), temperature=g("temperature", 0.07),
    )
    missing, unexpected = core.load_state_dict(eck["model"], strict=False)
    bad = [k for k in missing if not k.startswith("text_distilbert.text_model.")]
    if bad or unexpected:
        raise SystemExit(f"[recon-eval] evaluator load mismatch: missing(non-backbone)={bad[:8]} unexpected={list(unexpected)[:8]}")
    core.to(dev).eval()
    print(f"[recon-eval] evaluator {args.eval_ckpt} ep={eck.get('epoch')} loaded", flush=True)

    # FAIL-LOUD data/normalization contract: feeding the recon to the evaluator WITHOUT
    # re-normalizing is valid ONLY if the VQVAE and evaluator share the same data_root +
    # max_joints (=> identical per-joint 13ch normalization). Abort on mismatch, else the
    # gate numbers would be plausible-but-false.
    e_root, e_J = g("data_root", None), g("max_joints", None)
    if e_root is not None and vq_root != e_root:
        raise SystemExit(f"[recon-eval] DATA ROOT MISMATCH: vqvae '{vq_root}' vs evaluator "
                         f"'{e_root}' -> 13ch normalization differs; ABORT (pass matching ckpts).")
    if e_J is not None and int(ta.get("max_joints", -1)) != int(e_J):
        raise SystemExit(f"[recon-eval] max_joints MISMATCH: vqvae {ta.get('max_joints')} vs "
                         f"evaluator {e_J}; ABORT.")
    # FULL-LENGTH eval regime: the evaluator MEASURES at its training num_frames (300);
    # the VQVAE is run full-length at inference (trained at 64 but arch handles up to
    # max_coarse*temporal_stride frames). Load/reconstruct/embed at `nf` frames.
    nf = int(args.num_frames) if args.num_frames else int(g("num_frames", 300))
    vq_cap = int(ta.get("max_coarse", 96)) * int(ta.get("temporal_stride", 4))
    if nf > vq_cap:
        raise SystemExit(f"[recon-eval] num_frames {nf} exceeds VQVAE frame capacity "
                         f"(max_coarse {ta.get('max_coarse')} * temporal_stride {ta.get('temporal_stride')} "
                         f"= {vq_cap}); lower --num_frames or use a higher-capacity VQVAE.")
    print(f"[recon-eval] data contract OK: root={e_root} J={e_J} | FULL-LENGTH eval nf={nf} "
          f"(evaluator trained num_frames={g('num_frames', '?')}, VQVAE trained max_frames={ta.get('max_frames')}, "
          f"VQVAE inference cap={vq_cap})", flush=True)

    # ---- VQVAE val dataset (same val split via val_frac/seed; loaded at FULL length nf) ----
    ds = AnyTopDataset(
        split=args.split, num_frames=nf, max_joints=ta.get("max_joints", 144),
        val_frac=ta.get("val_frac", 0.05), seed=ta.get("seed", 42),
        load_captions=False, data_root=vq_root,
    )
    n_full = len(ds)
    if args.exclude_truebones:
        keep = [i for i in range(n_full) if str(ds.samples[i].get("object_type", "")).startswith("PZ_")]
        ds = Subset(ds, keep)
        print(f"[recon-eval] exclude_truebones: dropped {n_full - len(keep)} truebones, kept {len(keep)} animo4d", flush=True)
    loader = DataLoader(ds, batch_size=args.encode_batch, shuffle=False,
                        num_workers=args.num_workers, collate_fn=collate_fn, drop_last=False)
    print(f"[recon-eval] dataset root={vq_root} split={args.split} -> {len(ds)} clips (full {n_full})", flush=True)

    # ---- reconstruct + embed GT & recon over the SAME effective frame support ----
    EG, ER, objs = [], [], []
    mse_sum, mse_cnt = 0.0, 0
    CH = {"pos": (0, 3), "rot6d": (3, 9), "vel": (9, 12), "contact": (12, 13)}  # AnyTop 13ch groups
    ch_sum = {k: 0.0 for k in CH}
    ch_cnt = {k: 0 for k in CH}
    mpjpe_sum = {"animal": 0.0, "human": 0.0}   # de-norm world-pos L2 (sum over valid b,t,j)
    mpjpe_cnt = {"animal": 0.0, "human": 0.0}   # #valid (b,t,j)
    mpjpe_fk_sum = {"animal": 0.0, "human": 0.0}        # rot6d-FK pose vs TRUE GT pos: rc_fk vs gt_w (--fk)
    mpjpe_fk_cnt = {"animal": 0.0, "human": 0.0}
    mpjpe_fkfloor_sum = {"animal": 0.0, "human": 0.0}   # FK-route floor: gt_fk vs gt_w (--fk; recon-independent)
    mpjpe_fkfloor_cnt = {"animal": 0.0, "human": 0.0}
    mpjpe_fkavg_sum = {"animal": 0.0, "human": 0.0}     # sibling-AVERAGED FK vs gt_w (--fk_sibling_avg)
    mpjpe_fkavg_cnt = {"animal": 0.0, "human": 0.0}
    sibdisp = {"animal": [0.0, 0], "human": [0.0, 0]}      # recon sibling 6D dispersion [sum,cnt]
    sibdisp_gt = {"animal": [0.0, 0], "human": [0.0, 0]}   # GT sibling dispersion (self-check ~0)
    for coll in loader:
        coll_dev = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in coll.items()}
        batch = GraphMotionBatch.from_collate_dict(coll_dev)
        if args.gt_as_recon:
            # CONTROL: GT itself is the "recon" (VQVAE bypassed). recon == GT over the
            # full GT support -> the whole pipeline must yield cosine 1.0 / R@1 1.0 / FID 0.
            recon_x = batch.anytop_x.float()                    # [B,J,13,T] (GT)
            eff_mask = batch.frame_mask                         # [B,T]
        else:
            def _recon():
                # encode->decode directly (no quantizer.forward => collective-free, no EMA);
                # nearest_residual_ids replicates the quantizer's eval snap exactly.
                enc = vqvae.encode(batch)
                if args.continuous_recon:                       # PRE-VQ continuous latent h_lat (skip RVQ snap)
                    return vqvae.decode(enc["h_lat"], enc, batch)
                z_snap = vqvae.nearest_residual_ids(enc["h_lat"], enc["token_mask"])["z_snap"]
                return vqvae.decode(z_snap, enc, batch)         # quantized (= standard VQVAE recon)
            if amp_enabled:
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    out = _recon()
            else:
                out = _recon()
            pred = out["pred_motion"].float()                   # [B,T,J,13] normalized
            fmr = out["frame_mask_recovered"].bool()            # [B,T]
            eff_mask = batch.frame_mask & fmr                   # [B,T] apples-to-apples support
            recon_x = pred.permute(0, 2, 3, 1).contiguous()     # [B,T,J,13] -> [B,J,13,T]

        if args.mpjpe:
            # MPJPE: de-normalize GT & recon 13ch (raw = norm*(std+floor)+mean), recover joint world
            # positions, masked per-joint Euclidean over eff frames ∩ valid joints, per subset.
            # POSITION route (RIC->world, src torch recovery, = renderer's position path). With --fk,
            # ALSO the rot6d-FK route (6D rotations -> forward kinematics along the skeleton).
            # recon_x/anytop_x are [B,J,13,T] -> [B,T,J,13] for recovery.
            gtn = batch.anytop_x.permute(0, 3, 1, 2).float()        # [B,T,J,13] normalized GT
            rcn = recon_x.permute(0, 3, 1, 2).float()               # [B,T,J,13] normalized recon
            mean = batch.anytop_mean[:, None].float()               # [B,1,J,13]
            std = batch.anytop_std[:, None].float() + _STD_FLOOR    # [B,1,J,13]
            gt_raw = gtn * std + mean                               # [B,T,J,13] de-norm
            rc_raw = rcn * std + mean
            vmask = (eff_mask[:, :, None] & batch.joint_mask[:, None, :].bool())  # [B,T,J] bool
            gt_w = recover_world_positions_torch(gt_raw)            # [B,T,J,3] RIC/position route
            rc_w = recover_world_positions_torch(rc_raw)
            dpos = torch.where(vmask, (rc_w - gt_w).norm(dim=-1), torch.zeros(vmask.shape, device=vmask.device))
            ps_sum = dpos.sum(dim=(1, 2)); ps_cnt = vmask.float().sum(dim=(1, 2))   # [B]
            if args.fk:                                             # rot6d -> FK route
                gt_fk = recover_rot6d_fk_positions_torch(gt_raw, batch.parent_indices, batch.rest_offsets, batch.joint_mask)
                rc_fk = recover_rot6d_fk_positions_torch(rc_raw, batch.parent_indices, batch.rest_offsets, batch.joint_mask)
                z = torch.zeros(vmask.shape, device=vmask.device)
                dfk = torch.where(vmask, (rc_fk - gt_w).norm(dim=-1), z)    # recon FK pose vs TRUE GT pos
                dfl = torch.where(vmask, (gt_fk - gt_w).norm(dim=-1), z)    # FK-route floor (GT FK vs GT pos)
                fk_sum = dfk.sum(dim=(1, 2)); fl_sum = dfl.sum(dim=(1, 2))  # [B] (ps_cnt reused: same mask)
                if args.fk_sibling_avg:                                     # sibling-averaged parent rotation
                    rc_raw_sib = _sibling_avg_rot6d(rc_raw, batch.parent_indices, batch.joint_mask)
                    rc_fk_avg = recover_rot6d_fk_positions_torch(rc_raw_sib, batch.parent_indices, batch.rest_offsets, batch.joint_mask)
                    dfkavg = torch.where(vmask, (rc_fk_avg - gt_w).norm(dim=-1), z)
                    fkavg_sum = dfkavg.sum(dim=(1, 2))                      # [B]
            for bi, o in enumerate(coll["object_type"]):
                os = str(o).upper()
                if os.startswith("HML"):
                    sub = "human"
                elif os.startswith("PZ_"):
                    sub = "animal"
                else:
                    continue   # don't fold unknown skeletons (e.g. truebones) into a subset
                mpjpe_sum[sub] += float(ps_sum[bi].item())
                mpjpe_cnt[sub] += float(ps_cnt[bi].item())
                if args.fk:
                    mpjpe_fk_sum[sub] += float(fk_sum[bi].item())
                    mpjpe_fk_cnt[sub] += float(ps_cnt[bi].item())
                    if args.fk_sibling_avg:
                        mpjpe_fkavg_sum[sub] += float(fkavg_sum[bi].item())
                        mpjpe_fkavg_cnt[sub] += float(ps_cnt[bi].item())
                        _ds, _dc = _sibling_dispersion(rc_raw[bi:bi + 1], [batch.parent_indices[bi]], batch.joint_mask[bi:bi + 1], eff_mask[bi:bi + 1])
                        sibdisp[sub][0] += _ds; sibdisp[sub][1] += _dc
                        _gs, _gc = _sibling_dispersion(gt_raw[bi:bi + 1], [batch.parent_indices[bi]], batch.joint_mask[bi:bi + 1], eff_mask[bi:bi + 1])
                        sibdisp_gt[sub][0] += _gs; sibdisp_gt[sub][1] += _gc
                    mpjpe_fkfloor_sum[sub] += float(fl_sum[bi].item())
                    mpjpe_fkfloor_cnt[sub] += float(ps_cnt[bi].item())
        eff_nf = eff_mask.sum(dim=1).to(batch.num_frames.dtype)  # keep GraphMotionBatch invariant
        gt_x_emb, rec_x_emb = batch.anytop_x, recon_x
        if args.zero_contact:                                    # zero contact ch12 in BOTH before embed (motion-only)
            gt_x_emb = batch.anytop_x.clone(); gt_x_emb[:, :, 12, :] = 0.0
            rec_x_emb = recon_x.clone(); rec_x_emb[:, :, 12, :] = 0.0
        batch_gt = dataclasses.replace(batch, anytop_x=gt_x_emb, frame_mask=eff_mask, num_frames=eff_nf)
        batch_rec = dataclasses.replace(batch, anytop_x=rec_x_emb, frame_mask=eff_mask, num_frames=eff_nf)
        EG.append(core.encode_motion(batch_gt).float().cpu())
        ER.append(core.encode_motion(batch_rec).float().cpu())
        objs.extend(coll["object_type"])

        # norm-space masked recon MSE (cheap raw cross-check). GT anytop_x [B,J,13,T].
        gt_x = batch.anytop_x.float()
        valid = (eff_mask[:, None, None, :] & batch.joint_mask[:, :, None, None].bool())  # [B,J,1,T]
        diff2 = ((recon_x - gt_x) ** 2) * valid
        mse_sum += float(diff2.sum().item())
        mse_cnt += int(valid.sum().item()) * gt_x.shape[2]      # *13 channels
        nvalid = int(valid.sum().item())                       # #valid (b,j,t)
        for name, (a, b) in CH.items():                        # per-channel-group masked MSE
            ch_sum[name] += float((((recon_x[:, :, a:b, :] - gt_x[:, :, a:b, :]) ** 2) * valid).sum().item())
            ch_cnt[name] += nvalid * (b - a)
        if args.fk_sibling_avg:                                # 3 FK passes + clones fragment the
            torch.cuda.empty_cache()                           # allocator over many batches -> free per batch
    EG = torch.cat(EG, 0)
    ER = torch.cat(ER, 0)
    n = EG.shape[0]
    recon_mse_norm = mse_sum / max(mse_cnt, 1)
    per_channel_mse = {k: ch_sum[k] / max(ch_cnt[k], 1) for k in CH}
    print(f"[recon-eval] embedded {n} clips; norm-space masked recon MSE = {recon_mse_norm:.5f} "
          f"(recon={'CONTINUOUS' if args.continuous_recon else ('GT' if args.gt_as_recon else 'QUANTIZED')})", flush=True)
    print(f"[recon-eval] per-channel norm MSE: " + " ".join(f"{k}={per_channel_mse[k]:.4f}" for k in CH), flush=True)

    mpjpe = None
    if args.mpjpe:
        def _avg(s):
            return (mpjpe_sum[s] / mpjpe_cnt[s]) if mpjpe_cnt[s] > 0 else None
        tot_s = mpjpe_sum["animal"] + mpjpe_sum["human"]
        tot_c = mpjpe_cnt["animal"] + mpjpe_cnt["human"]
        mpjpe = {"animal": _avg("animal"), "human": _avg("human"),
                 "overall": (tot_s / tot_c) if tot_c > 0 else None,
                 "units": "ABSOLUTE world-position MPJPE, ROOT INCLUDED (not root-aligned); "
                          "de-normalized AnyTop world units (per-skeleton mean/std restored)"}
        def _f(x):
            return f"{x:.5f}" if isinstance(x, float) else "n/a"
        print(f"[recon-eval] MPJPE (ABSOLUTE world-pos, root-incl, de-norm units): animal={_f(mpjpe['animal'])} "
              f"human={_f(mpjpe['human'])} overall={_f(mpjpe['overall'])} "
              f"(n_valid animal={int(mpjpe_cnt['animal'])} human={int(mpjpe_cnt['human'])})", flush=True)
        if args.fk:
            def _avg2(sm, cn, s):
                return (sm[s] / cn[s]) if cn[s] > 0 else None
            def _ov(sm, cn):
                ts, tc = sm["animal"] + sm["human"], cn["animal"] + cn["human"]
                return (ts / tc) if tc > 0 else None
            mpjpe["fk"] = {"animal": _avg2(mpjpe_fk_sum, mpjpe_fk_cnt, "animal"),
                           "human": _avg2(mpjpe_fk_sum, mpjpe_fk_cnt, "human"),
                           "overall": _ov(mpjpe_fk_sum, mpjpe_fk_cnt),
                           "route": "recon rot6d->FK pose vs TRUE GT positions (rc_fk vs gt_w); "
                                    "absolute world pos, root-incl, de-norm units; INCLUDES the fk_floor"}
            mpjpe["fk_floor"] = {"animal": _avg2(mpjpe_fkfloor_sum, mpjpe_fkfloor_cnt, "animal"),
                                 "human": _avg2(mpjpe_fkfloor_sum, mpjpe_fkfloor_cnt, "human"),
                                 "overall": _ov(mpjpe_fkfloor_sum, mpjpe_fkfloor_cnt),
                                 "route": "FK-route inherent floor: GT rot6d->FK vs GT positions (gt_fk vs gt_w), "
                                          "recon-independent; subtract-in-quadrature-ish to gauge pure recon error"}
            print(f"[recon-eval] MPJPE-FK (recon rot6d->FK vs GT pos): animal={_f(mpjpe['fk']['animal'])} "
                  f"human={_f(mpjpe['fk']['human'])} overall={_f(mpjpe['fk']['overall'])}", flush=True)
            print(f"[recon-eval] FK-floor (GT rot6d->FK vs GT pos, recon-indep): animal={_f(mpjpe['fk_floor']['animal'])} "
                  f"human={_f(mpjpe['fk_floor']['human'])} overall={_f(mpjpe['fk_floor']['overall'])}", flush=True)
            if args.fk_sibling_avg:
                def _dsp(d, s):
                    return (d[s][0] / d[s][1]) if d[s][1] > 0 else None
                mpjpe["fk_sibling_avg"] = {"animal": _avg2(mpjpe_fkavg_sum, mpjpe_fkavg_cnt, "animal"),
                                           "human": _avg2(mpjpe_fkavg_sum, mpjpe_fkavg_cnt, "human"),
                                           "overall": _ov(mpjpe_fkavg_sum, mpjpe_fkavg_cnt),
                                           "route": "recon rot6d->FK with SIBLING-AVERAGED parent rotation vs gt_w"}
                mpjpe["sibling_dispersion"] = {"recon_animal": _dsp(sibdisp, "animal"), "recon_human": _dsp(sibdisp, "human"),
                                               "gt_animal": _dsp(sibdisp_gt, "animal"), "gt_human": _dsp(sibdisp_gt, "human"),
                                               "note": "mean L2 of branching-parent child-slot 6D from sibling-mean; GT~0 verifies duplicate convention, recon>0 = model diverges siblings"}
                print(f"[recon-eval] MPJPE-FK-SIBAVG (sibling-averaged parent rot): animal={_f(mpjpe['fk_sibling_avg']['animal'])} "
                      f"human={_f(mpjpe['fk_sibling_avg']['human'])} overall={_f(mpjpe['fk_sibling_avg']['overall'])}", flush=True)
                print(f"[recon-eval] sibling 6D dispersion: recon(animal={_f(_dsp(sibdisp,'animal'))} human={_f(_dsp(sibdisp,'human'))}) "
                      f"GT(animal={_f(_dsp(sibdisp_gt,'animal'))} human={_f(_dsp(sibdisp_gt,'human'))}) [GT~0 expected]", flush=True)

    # ---- subset split: animo4d (PZ_*) vs truebones (everything else) ----
    src = ["animo4d" if str(o).startswith("PZ_") else "truebones" for o in objs]
    report = {"n_total": n, "pool": args.pool, "vqvae_ckpt": args.vqvae_ckpt,
              "eval_ckpt": args.eval_ckpt, "vqvae_epoch": vck.get("epoch"),
              "eval_epoch": eck.get("epoch"), "recon_mse_norm": recon_mse_norm,
              "data_root": vq_root, "max_joints": ta.get("max_joints"),
              "eval_num_frames": nf, "vqvae_max_frames": ta.get("max_frames"),
              "num_codes": ta.get("num_codes"), "num_quantizers": ta.get("num_quantizers"),
              "recon_mode": ("continuous" if args.continuous_recon else ("gt" if args.gt_as_recon else "quantized")),
              "per_channel_mse": per_channel_mse,
              "mpjpe_worldpos": mpjpe}

    def _rr(m):  # robust "R@1/2/3" string (recon_to_gt_rprec may be None when n<pool)
        rr = m.get("recon_to_gt_rprec")
        return (f"R@1={rr[1]:.3f} R@2={rr[2]:.3f} R@3={rr[3]:.3f}" if rr else "R@k=NA(n<pool)")

    report["overall"] = subset_metrics(EG, ER, args.pool, args.fid_min, args.max_div_pairs, gen)
    o = report["overall"]
    print(f"[recon-eval] OVERALL n={o['n']} cos(mean/med)={o['pair_cosine_mean']:.3f}/{o['pair_cosine_median']:.3f} "
          f"recon→GT {_rr(o)} FID={o['fid']} div(gt/rec)={o['diversity_gt']:.3f}/{o['diversity_recon']:.3f}", flush=True)

    report["per_subset"] = {}
    for s in ("animo4d", "truebones"):
        idx = [i for i, x in enumerate(src) if x == s]
        if not idx:
            continue
        ii = torch.tensor(idx)
        m = subset_metrics(EG[ii], ER[ii], args.pool, args.fid_min, args.max_div_pairs, gen)
        report["per_subset"][s] = m
        print(f"[recon-eval] SUBSET {s:9s} n={m['n']:5d} cos(mean/med)={m['pair_cosine_mean']:.3f}/{m['pair_cosine_median']:.3f} "
              f"recon→GT {_rr(m)} FID={m['fid']} ({m['fid_note']}) div(gt/rec)={m['diversity_gt']:.3f}/{m['diversity_recon']:.3f}", flush=True)

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(report, indent=2))
        print(f"[recon-eval] report -> {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
