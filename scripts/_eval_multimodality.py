"""MultiModality (MModality) for the Graph-CodeFlow text->motion generator.

The one standard-protocol metric the full-protocol gen-eval was missing.

Definition (VERBATIM port of MotionMillion/T2M `calculate_multimodality`, see
outside_docs/MotionMillion-Codes/utils/eval_trans.py:1213):
  activation: [N_texts, R_gen_per_text, D] evaluator embeddings
  pick `mm_times` random generation-indices twice, L2 between the two picks, mean.

Protocol note: the reference generates R=30 per text. Doing that over the FULL val set
(5150 x 30 = 154.5k generations) is ~150 GPU-hours, so — exactly as the classic T2M /
MDM protocol does — MultiModality is computed on a TEXT SUBSET (`--mm_num_samples` per
subset, strided for spread), with the full R repeats each.

Generation reuses `run_gen_eval` (the SAME code path the protocol eval uses) via its
optional `emb_sink` out-param, so the mixed-length latent masking / soft-clamp logic is
not duplicated here. Each repeat uses a different seed => different flow noise.

Run on an IDLE gpu; never the training cards. Compute nodes have no internet:
    env HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python scripts/_eval_multimodality.py ...
"""
from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from scipy import linalg

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.anytop_t2m_eval_dataset import AnyTopT2MEvalDataset  # noqa: E402
from src.eval.codeflow_gen_eval import motion_id_bucket, run_gen_eval  # noqa: E402
from src.models.graph_salad.t2m_evaluator import AnyTopT2MEvaluator  # noqa: E402


def _imp(name):
    m = importlib.import_module("scripts.animate_graph_codeflow")
    return getattr(m, name)


def calculate_multimodality(activation, multimodality_times):
    """VERBATIM from MotionMillion utils/eval_trans.py:1213."""
    assert len(activation.shape) == 3
    assert activation.shape[1] > multimodality_times
    num_per_sent = activation.shape[1]
    first_dices = np.random.choice(num_per_sent, multimodality_times, replace=False)
    second_dices = np.random.choice(num_per_sent, multimodality_times, replace=False)
    dist = linalg.norm(activation[:, first_dices] - activation[:, second_dices], axis=2)
    return dist.mean()


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--flow_ckpt", required=True)
    ap.add_argument("--eval_ckpt", required=True)
    ap.add_argument("--val_manifest", required=True)
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--caption_cache", default=None,
                    help="LLM2Vec ragged sidecar prefix; REQUIRED when the flow ckpt was "
                         "trained with text_dim != 768")
    ap.add_argument("--mm_num_samples", type=int, default=100,
                    help="texts per subset (human/animal), strided over that subset for spread")
    ap.add_argument("--mm_num_repeats", type=int, default=30,
                    help="generations per text (reference uses 30; must exceed --mm_times)")
    ap.add_argument("--mm_times", type=int, default=10,
                    help="index picks per side in calculate_multimodality (reference uses 10)")
    ap.add_argument("--steps", type=int, default=25)
    ap.add_argument("--cfg_scale", type=float, default=4.0)
    ap.add_argument("--num_frames", type=int, default=300)
    ap.add_argument("--max_joints", type=int, default=144)
    ap.add_argument("--gen_batch", type=int, default=32)
    ap.add_argument("--pool", type=int, default=32)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=None)
    return ap.parse_args()


@torch.no_grad()
def main():
    args = parse_args()
    if args.mm_num_repeats <= args.mm_times:
        raise SystemExit(f"--mm_num_repeats ({args.mm_num_repeats}) must EXCEED "
                         f"--mm_times ({args.mm_times}) — calculate_multimodality asserts it")
    if args.mm_num_samples <= 0:
        raise SystemExit("--mm_num_samples must be > 0 (0 would divide by zero in the stride)")
    if args.mm_times <= 0:
        raise SystemExit("--mm_times must be > 0 (0 would make MultiModality an empty mean = NaN)")
    if args.pool < 3:
        raise SystemExit(f"--pool must be >= 3 (got {args.pool}); R@3 needs at least 3 candidates")
    # Compute nodes have NO internet. Without these, transformers tries to reach the HF hub and
    # dies mid-run (or hangs). Fail here, not 6 GPU-hours in.
    for _v in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"):
        if os.environ.get(_v) != "1":
            raise SystemExit(f"{_v}=1 is required (compute nodes have no internet); "
                             f"run under: env HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python ...")
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    load_flow = _imp("load_flow")
    load_frozen_tokenizer = _imp("load_frozen_tokenizer")

    flow, fck = load_flow(args.flow_ckpt, 512, dev)
    if "latent_mean" in fck:
        flow.latent_mean = fck["latent_mean"].to(dev)
        flow.latent_std = fck["latent_std"].to(dev)
    vq_ckpt = fck.get("frozen_vqvae_ckpt")
    tokenizer, ta = load_frozen_tokenizer(vq_ckpt, dev)
    stride = int(ta["temporal_stride"])
    print(f"[mm] flow {args.flow_ckpt} ep={fck.get('epoch')} val_flow={fck.get('val_flow')} | "
          f"VQVAE stride={stride}", flush=True)

    eck = torch.load(args.eval_ckpt, map_location="cpu")
    ea = eck["args"]
    g = (lambda k, d: ea.get(k, d)) if isinstance(ea, dict) else (lambda k, d: getattr(ea, k, d))
    core = AnyTopT2MEvaluator(
        coemb_dim=g("coemb_dim", 512), text_tower=g("text_tower", "distilbert"),
        distilbert_path=g("distilbert_path", "checkpoints/text_encoders/distilbert-base-uncased"),
        text_max_length=g("text_max_length", 64), n_heads=g("n_heads", 8), d_ff=g("d_ff", 2048),
        n_graph_layers=g("n_graph_layers", 6), n_temporal_layers=g("n_temporal_layers", 4),
        motion_feat_dim=g("motion_feat_dim", 13),
        dropout=g("dropout", 0.1), learnable_temperature=not g("fixed_temperature", False),
        temperature=g("temperature", 0.07))
    miss, unexp = core.load_state_dict(eck["model"], strict=False)
    bad = [k for k in miss if not k.startswith("text_distilbert.text_model.")]
    if bad or unexp:
        raise SystemExit(f"[mm] evaluator load mismatch: missing={bad[:8]} unexpected={list(unexp)[:8]}")
    core.to(dev).eval()

    # Same hard data contract as the protocol eval — a mismatch invalidates the eval space.
    eval_root = g("data_root", None)
    if eval_root and args.data_root != eval_root:
        raise SystemExit(f"[mm] --data_root {args.data_root} != evaluator data_root {eval_root}")
    vq_root = ta.get("anytop_root") or ta.get("data_root")
    if vq_root and eval_root and vq_root != eval_root:
        raise SystemExit(f"[mm] VQVAE root {vq_root} != evaluator data_root {eval_root}")
    if g("num_frames", None) and args.num_frames != int(g("num_frames", -1)):
        raise SystemExit(f"[mm] --num_frames {args.num_frames} != evaluator {g('num_frames', None)}")
    if g("max_joints", None) and args.max_joints != int(g("max_joints", -1)):
        raise SystemExit(f"[mm] --max_joints {args.max_joints} != evaluator {g('max_joints', None)}")

    # Text encoder follows the FLOW ckpt (codex r3 #6): LLM2Vec ckpts read the ragged
    # sidecar via the shared meta-driven lookup; only legacy 768 ckpts load T5.
    _fa_mm = fck.get("args", {})
    _fg_mm = (_fa_mm.get if isinstance(_fa_mm, dict) else lambda k, d=None: getattr(_fa_mm, k, d))
    _tdim_mm = int(_fg_mm("text_dim", 768) or 768)
    if _tdim_mm != 768:
        if not args.caption_cache:
            raise SystemExit(
                f"[mm] flow ckpt has text_dim={_tdim_mm}; pass --caption_cache "
                f"(LLM2Vec sidecar prefix)")
        from src.eval.codeflow_gen_eval import make_caption_lookup_encoder
        t5_encode_batch = make_caption_lookup_encoder(
            args.caption_cache, args.data_root, _tdim_mm, dev)
    else:
        from transformers import T5EncoderModel, T5TokenizerFast
        t5tok = T5TokenizerFast.from_pretrained("t5-base", local_files_only=True)
        t5 = T5EncoderModel.from_pretrained("t5-base", local_files_only=True).to(dev).eval()

        @torch.no_grad()
        def t5_encode_batch(texts):
            enc = t5tok(list(texts), return_tensors="pt", padding="max_length",
                        truncation=True, max_length=64).to(dev)
            hs = t5(input_ids=enc.input_ids, attention_mask=enc.attention_mask).last_hidden_state
            m = enc.attention_mask.bool()
            gl = (hs * m.unsqueeze(-1).float()).sum(1) / m.sum(1, keepdim=True).clamp_min(1)
            return gl.float(), hs.float(), m

    ds = AnyTopT2MEvalDataset(manifest_path=args.val_manifest, data_root=args.data_root,
                              caption_emb_cache=None, split="val", view="full",
                              num_frames=args.num_frames, max_joints=args.max_joints)

    # MM text subset: strided per subset so the sample spreads across species / actions.
    def _stride(lst, k):
        return lst if k >= len(lst) else lst[::max(1, len(lst) // k)][:k]

    by_bucket = {}
    for i in range(len(ds)):
        by_bucket.setdefault(motion_id_bucket(str(ds._plan[i][1]["motion_id"])), []).append(i)
    mm_idxs = []
    for s in ("human", "animal"):
        sel = _stride(by_bucket.get(s, []), args.mm_num_samples)
        mm_idxs += sel
        print(f"[mm] subset {s}: {len(sel)} texts (of {len(by_bucket.get(s, []))})", flush=True)
    if not mm_idxs:
        raise SystemExit("[mm] no texts selected")
    print(f"[mm] {len(mm_idxs)} texts x {args.mm_num_repeats} generations "
          f"= {len(mm_idxs) * args.mm_num_repeats} samples | steps={args.steps} cfg={args.cfg_scale}",
          flush=True)

    # R repeats over the SAME texts, different seed each => different flow noise.
    # run_gen_eval keeps idxs order, so row i of every repeat is the same text.
    sink = []
    for r in range(args.mm_num_repeats):
        run_gen_eval(flow=flow, tokenizer=tokenizer, core=core, t5_encode_batch=t5_encode_batch,
                     ds=ds, idxs=mm_idxs, dev=dev, stride=stride, pool=args.pool,
                     steps=args.steps, cfg_scale=args.cfg_scale, num_frames=args.num_frames,
                     gen_batch=args.gen_batch, seed=args.seed + r,
                     log=lambda *_a, **_k: None, emb_sink=sink)
        print(f"[mm] repeat {r + 1}/{args.mm_num_repeats} done", flush=True)

    # MM is only meaningful if row i is the SAME text in every repeat. motion_id can repeat
    # across caption rows, so key on (dataset_index, caption) and also check the row count.
    keys0 = [tuple(k) for k in sink[0]["row_keys"]]
    mids = sink[0]["motion_ids"]
    if len(keys0) != int(sink[0]["gen_emb"].shape[0]):
        raise SystemExit(f"[mm] repeat 0: {len(keys0)} row_keys vs {sink[0]['gen_emb'].shape[0]} "
                         f"embedding rows — refusing to compute MM")
    for k, rec in enumerate(sink):                       # fail loud on ANY row misalignment
        keys = [tuple(x) for x in rec["row_keys"]]
        if keys != keys0 or int(rec["gen_emb"].shape[0]) != len(keys0):
            raise SystemExit(f"[mm] repeat {k} row keys/count differ from repeat 0 — "
                             f"rows would not correspond; refusing to compute MM")
    act = torch.stack([rec["gen_emb"] for rec in sink], dim=1).numpy()   # [N, R, D]
    print(f"[mm] activation {act.shape} (texts, repeats, dim)", flush=True)

    np.random.seed(args.seed)
    report = {
        "n_texts": int(act.shape[0]), "mm_num_repeats": int(act.shape[1]),
        "mm_times": args.mm_times, "steps": args.steps, "cfg_scale": args.cfg_scale,
        "flow_ckpt": args.flow_ckpt, "epoch": fck.get("epoch"),
        # HONEST LABEL: this is a STRATIFIED-SUBSET MultiModality (equal texts per bucket via a
        # deterministic stride over the manifest), NOT natural-val-weighted over all 5150 clips.
        # Outcome-independent (so not cherry-picked) but manifest-order-sensitive.
        "sampling": "stratified-subset (deterministic stride, equal texts per bucket)",
        "mm_num_samples_per_bucket": args.mm_num_samples,
        "mm_idxs": [int(i) for i in mm_idxs],
        "row_keys": [[int(a), str(b)] for a, b in keys0],
        "multimodality": float(calculate_multimodality(act, args.mm_times)),
        "per_subset": {},
    }
    buckets = [motion_id_bucket(m) for m in mids]
    for s in ("human", "animal"):
        ii = [i for i, b in enumerate(buckets) if b == s]
        if len(ii) < 2:
            continue
        report["per_subset"][s] = {
            "n_texts": len(ii),
            "multimodality": float(calculate_multimodality(act[ii], args.mm_times)),
        }

    print(f"[mm] MultiModality OVERALL n={report['n_texts']} R={report['mm_num_repeats']} "
          f"-> {report['multimodality']:.4f}", flush=True)
    for s, v in report["per_subset"].items():
        print(f"[mm] MultiModality {s:8s} n={v['n_texts']} -> {v['multimodality']:.4f}", flush=True)

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        json.dump(report, open(args.out, "w"), indent=1)
        print(f"[mm] report -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
