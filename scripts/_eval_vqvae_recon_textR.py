"""CORRECT text↔motion R-precision for a VQVAE reconstruction (the standard R@1/2/3:
text query -> rank candidate motions -> is the matching motion in top-k).

Distinct from scripts/_eval_vqvae_recon_in_evalspace.py, whose `recon_to_gt_rprec` is a
motion↔motion (recon-vs-GT) FIDELITY retrieval, NOT R-precision (it has no text). Lesson
2026-06-21: R-precision is ALWAYS text↔motion (SALAD/T2M/MotionMillion).

This script, per the evaluator's own regime (AnyTopT2MEvalDataset / val_all.json, DistilBERT
text tower — NO T5; T5 is only the flow's conditioning, not the evaluator), reconstructs each
val clip through the frozen VQVAE and reports, per subset (animal=PZ_* / human=HML3D_*):
  - text->RECON R@1/2/3   (does the reconstructed motion still get retrieved by its caption)
  - text->GT   R@1/2/3    (the ceiling = the evaluator's own discrimination on GT)
A recon that preserves text-alignment has text->recon ≈ text->GT. Frozen, no grad, single-GPU.
"""
from __future__ import annotations
import argparse, dataclasses, importlib.util, json
from pathlib import Path
import torch
from torch.utils.data import DataLoader
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.anytop_t2m_eval_dataset import AnyTopT2MEvalDataset, collate_fn
from src.models.graph_salad.batch import GraphMotionBatch
from src.models.graph_salad.t2m_evaluator import AnyTopT2MEvaluator
from src.eval.codeflow_gen_eval import pooled_rprec, motion_id_bucket


def _load_vq():
    p = Path(__file__).resolve().parent / "animate_vqvae_recon.py"
    spec = importlib.util.spec_from_file_location("_avr_loader", p)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod.load_vq_tokenizer


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--vqvae_ckpt", required=True)
    ap.add_argument("--eval_ckpt", required=True, help="frozen AnyTopT2MEvaluator best_model.pt (DistilBERT text tower).")
    ap.add_argument("--val_manifest", default=None, help="default <data_root>/eval_splits/val_all.json")
    ap.add_argument("--data_root", default=None, help="default = evaluator ckpt's data_root.")
    ap.add_argument("--pool", type=int, default=32)
    ap.add_argument("--num_frames", type=int, default=None, help="default = evaluator num_frames; capped at VQVAE max_coarse*stride.")
    ap.add_argument("--max_joints", type=int, default=None, help="default = evaluator max_joints.")
    ap.add_argument("--encode_batch", type=int, default=16)
    ap.add_argument("--amp_dtype", choices=["fp32", "bf16"], default=None,
                    help="VQVAE recon autocast; default = the ckpt's trained regime (bf16).")
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=None)
    return ap.parse_args()


@torch.no_grad()
def main():
    args = parse_args()
    if args.pool < 3:
        raise SystemExit(f"--pool must be >=3; got {args.pool}")
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- frozen VQVAE ----
    load_vq = _load_vq()
    vqvae, ta, vck = load_vq(args.vqvae_ckpt, dev)       # loader returns (model, ta, ckpt)
    stride = int(ta["temporal_stride"])
    vq_cap = int(ta.get("max_coarse", 96)) * stride
    amp_dtype = args.amp_dtype or ta.get("amp_dtype", "bf16")
    amp_enabled = (amp_dtype == "bf16") and dev.type == "cuda"

    # ---- frozen evaluator (DistilBERT text tower) ----
    eck = torch.load(args.eval_ckpt, map_location="cpu")
    ea = eck["args"]
    g = (lambda k, d: ea.get(k, d)) if isinstance(ea, dict) else (lambda k, d: getattr(ea, k, d))
    mfd = int(g("motion_feat_dim", 13))
    core = AnyTopT2MEvaluator(
        coemb_dim=g("coemb_dim", 512), text_tower=g("text_tower", "distilbert"),
        distilbert_path=g("distilbert_path", "checkpoints/text_encoders/distilbert-base-uncased"),
        text_max_length=g("text_max_length", 64), n_heads=g("n_heads", 8), d_ff=g("d_ff", 2048),
        n_graph_layers=g("n_graph_layers", 6), n_temporal_layers=g("n_temporal_layers", 4),
        motion_feat_dim=mfd, dropout=g("dropout", 0.1),
        learnable_temperature=not g("fixed_temperature", False), temperature=g("temperature", 0.07))
    miss, unexp = core.load_state_dict(eck["model"], strict=False)
    bad = [k for k in miss if not k.startswith("text_distilbert.text_model.")]
    if bad or unexp:
        raise SystemExit(f"[recon-textR] evaluator load mismatch: missing={bad[:6]} unexpected={list(unexp)[:6]}")
    core.to(dev).eval()

    eval_root = g("data_root", None)
    data_root = args.data_root or eval_root
    if not data_root:
        raise SystemExit("[recon-textR] no data_root")
    vq_root = ta.get("anytop_root") or ta.get("data_root")
    if vq_root and str(Path(vq_root).resolve()) != str(Path(data_root).resolve()):
        raise SystemExit(f"[recon-textR] VQVAE root {vq_root} != data_root {data_root} (eval-space invalid)")
    nf = int(args.num_frames) if args.num_frames else int(g("num_frames", 300))
    if nf > vq_cap:
        print(f"[recon-textR] WARN nf={nf} > VQVAE cap {vq_cap}; clamping to {vq_cap}", flush=True); nf = vq_cap
    mj = int(args.max_joints) if args.max_joints else int(g("max_joints", 144))
    vq_mj = ta.get("max_joints")                         # eval-space validity: VQVAE + evaluator must share max_joints
    if vq_mj is not None and int(vq_mj) != mj:
        raise SystemExit(f"[recon-textR] VQVAE max_joints {vq_mj} != evaluator/eval max_joints {mj}")
    manifest = args.val_manifest or str(Path(data_root) / "eval_splits" / "val_all.json")
    print(f"[recon-textR] VQVAE {args.vqvae_ckpt} ep={vck.get('epoch','?')} K={ta.get('num_codes')} | "
          f"evaluator ep={eck.get('epoch')} mfd={mfd} | nf={nf} mj={mj} root={data_root}", flush=True)

    ds = AnyTopT2MEvalDataset(manifest_path=manifest, data_root=data_root, caption_emb_cache=None,
                              split="val", view="full", num_frames=nf, max_joints=mj)
    loader = DataLoader(ds, batch_size=args.encode_batch, shuffle=False,
                        num_workers=args.num_workers, collate_fn=collate_fn, drop_last=False)
    print(f"[recon-textR] dataset {manifest} -> {len(ds)} clips", flush=True)

    TE, RE, GE, mids = [], [], [], []
    done = 0
    for coll in loader:
        cd = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in coll.items()}
        batch = GraphMotionBatch.from_collate_dict(cd)
        caps = [c or "" for c in coll["caption_text"]]
        # standard quantized VQVAE reconstruction (encode -> RVQ snap -> decode), in the
        # ckpt's trained autocast regime (bf16) to match scripts/_eval_vqvae_recon_in_evalspace.py.
        def _recon():
            enc = vqvae.encode(batch)
            z_snap = vqvae.nearest_residual_ids(enc["h_lat"], enc["token_mask"])["z_snap"]
            return vqvae.decode(z_snap, enc, batch)
        if amp_enabled:
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                out = _recon()
        else:
            out = _recon()
        pred = out["pred_motion"].float()                              # [B,T,J,13]
        eff_mask = batch.frame_mask & out["frame_mask_recovered"].bool()
        eff_nf = eff_mask.sum(dim=1).to(batch.num_frames.dtype)
        recon_x = pred.permute(0, 2, 3, 1).contiguous()                # [B,J,13,T]
        batch_rec = dataclasses.replace(batch, anytop_x=recon_x, frame_mask=eff_mask, num_frames=eff_nf)
        batch_gt = dataclasses.replace(batch, frame_mask=eff_mask, num_frames=eff_nf)  # GT over same support
        RE.append(core.encode_motion(batch_rec).float().cpu())
        GE.append(core.encode_motion(batch_gt).float().cpu())
        TE.append(core.encode_text(caps).float().cpu())
        mids.extend(str(m) for m in coll["motion_id"])     # collate passes motion_id per item, in order
        done += len(caps)
        print(f"[recon-textR]   {done}/{len(ds)} embedded", flush=True)
    TE = torch.cat(TE); RE = torch.cat(RE); GE = torch.cat(GE)
    assert len(mids) == TE.shape[0], f"motion_id count {len(mids)} != embeddings {TE.shape[0]}"
    buckets = [motion_id_bucket(m) for m in mids]

    def _rr(q, g_, idx=None):
        if idx is not None:
            t = torch.tensor(idx); q, g_ = q[t], g_[t]
        rr, npp = pooled_rprec(q, g_, args.pool)
        return rr, npp, q.shape[0]

    report = {"n": TE.shape[0], "pool": args.pool, "vqvae_ckpt": args.vqvae_ckpt,
              "vqvae_epoch": vck.get("epoch"), "eval_ckpt": args.eval_ckpt, "eval_epoch": eck.get("epoch"),
              "num_frames": nf, "note": "text->motion R-precision (standard); text->recon vs text->GT(ceiling)"}
    for name, idx in [("OVERALL", None)] + [(b, [i for i, x in enumerate(buckets) if x == b])
                                            for b in ("animal", "human", "truebones") if any(x == b for x in buckets)]:
        rec_rr, npp, ncnt = _rr(TE, RE, idx)
        gt_rr, _, _ = _rr(TE, GE, idx)
        report[name] = {"n": ncnt, "pools": npp, "text_to_recon_rprec": rec_rr, "text_to_gt_rprec": gt_rr}
        def s(rr): return f"R@1={rr[1]:.3f} R@2={rr[2]:.3f} R@3={rr[3]:.3f}" if rr else "NA(n<pool)"
        print(f"[recon-textR] {name:9s} n={ncnt:5d} | text->RECON {s(rec_rr)} | text->GT {s(gt_rr)}", flush=True)

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(report, indent=2))
        print(f"[recon-textR] report -> {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
