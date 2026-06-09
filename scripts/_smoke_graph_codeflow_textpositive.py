#!/usr/bin/env python3
"""Text-positivity probe for the Graph-CodeFlow smoke (companion to
_smoke_graph_codeflow.py). Confirms the L5 val samples carry REAL text and that
the flow's text routes actually participate (not just are wired but dead).

Asserts, on the same 2 real L5 val clips the main smoke uses:
  A. caption_emb (global mean T5) is non-zero on the batch.
  B. caption_token_mask has > 0 valid tokens (token route has keys).
  C. caption_token_emb (token T5) is non-zero on valid tokens.
  D. has_text is True for the clips.
  E. BEHAVIORAL: predict_velocity with has_text=True vs has_text=False (CFG
     uncond view) yields a NON-TRIVIALLY DIFFERENT velocity on valid tokens.
     If the text route were dead (zeroed text, or projection collapsed), this
     delta would be ~0 and the probe FAILS. This is the catch-the-failure check.

DOES NOT launch training. Single-process, eval(), 2 real L5 samples.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data.anytop_dataset import AnyTopDataset, collate_fn as anytop_collate_fn
from src.models.graph_salad.batch import GraphMotionBatch
from src.models.vq_model import GraphVQTokenizer
from src.models.CodeFlow_Model import GraphCodeFlow

ROOT_DATA = "data/animo4d_anytop_clean_L5"


def fail(msg):
    print(f"[TEXT-SMOKE FAIL] {msg}")
    raise SystemExit(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frozen_vqvae_ckpt", type=str, required=True)
    ap.add_argument("--caption_cache", type=str,
                    default="data/anytop_caption_t5_cleanL5_multi")
    args = ap.parse_args()
    CAP_EMB = args.caption_cache + ".npz"
    CAP_TOK = args.caption_cache

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={dev}")
    ck = torch.load(args.frozen_vqvae_ckpt, map_location="cpu", weights_only=False)
    ta = ck["args"]
    model = GraphVQTokenizer(
        d_model=ta["d_model"], n_heads=ta["n_heads"], d_ff=ta["d_ff"],
        n_graph_layers=ta["n_graph_layers"],
        n_enc_temporal_layers=ta["n_enc_temporal_layers"],
        n_pre_vq_layers=ta["n_pre_vq_layers"], n_post_vq_layers=ta["n_post_vq_layers"],
        n_cross_layers=ta["n_cross_layers"],
        n_dec_temporal_layers=ta["n_dec_temporal_layers"],
        max_coarse=ta["max_coarse"], temporal_stride=ta["temporal_stride"],
        temporal_kernel=ta["temporal_kernel"], dropout=ta["dropout"],
        code_dim=ta["code_dim"], num_codes=ta["num_codes"],
        num_quantizers=ta["num_quantizers"], ema_mu=ta["ema_mu"],
        quantize_dropout_prob=ta["quantize_dropout_prob"],
        dead_code_threshold=ta["dead_code_threshold"],
    ).to(dev)
    model.load_state_dict(ck["model_state_dict"], strict=True)
    model.eval()
    model.requires_grad_(False)
    D = ta["d_model"]

    ds = AnyTopDataset(
        split="val", num_frames=ta.get("max_frames", 64),
        max_joints=ta.get("max_joints", 64), val_frac=ta.get("val_frac", 0.05),
        seed=ta.get("seed", 42), data_root=ROOT_DATA, load_captions=True,
        caption_emb_cache=CAP_EMB, caption_token_cache=CAP_TOK,
        return_caption_tokens=True, random_caption=False)
    items = [ds[0], ds[1]]
    raw = anytop_collate_fn(items)
    raw = {k: v.to(dev) if torch.is_tensor(v) else v for k, v in raw.items()}
    batch = GraphMotionBatch.from_collate_dict(raw)

    cemb = batch.caption_emb.float()                 # [B, 768]
    ctok = batch.caption_token_emb.float()           # [B, L, 768]
    ctmask = batch.caption_token_mask                # [B, L] bool
    htext = batch.has_text                           # [B] bool
    B = cemb.shape[0]
    print(f"2 real clips: species={[it['object_type'] for it in items]}")
    print(f"caption_emb={list(cemb.shape)} token_emb={list(ctok.shape)} "
          f"token_mask={list(ctmask.shape)} has_text={htext.tolist()}")

    # --- A. global caption_emb non-zero ---
    g_abs = cemb.abs()
    g_max = g_abs.max().item()
    g_per = g_abs.amax(dim=1)  # [B]
    if g_max == 0.0:
        fail("A: caption_emb is ALL-ZERO (no global text)")
    if not bool((g_per > 0).all()):
        fail(f"A: some clip has all-zero caption_emb (per-clip max {g_per.tolist()})")

    # --- B. token mask > 0 ---
    n_valid_tok = ctmask.sum(dim=1)  # [B]
    if int(ctmask.sum()) == 0:
        fail("B: caption_token_mask has ZERO valid tokens (token route dead)")
    if not bool((n_valid_tok > 0).all()):
        fail(f"B: some clip has 0 valid tokens (per-clip {n_valid_tok.tolist()})")

    # --- C. token emb non-zero on valid tokens ---
    valid_tok = ctmask.unsqueeze(-1).expand_as(ctok)
    t_max = ctok[valid_tok].abs().max().item() if valid_tok.any() else 0.0
    if t_max == 0.0:
        fail("C: caption_token_emb is ALL-ZERO on valid tokens")

    # --- D. has_text True ---
    if not bool(htext.all()):
        fail(f"D: has_text not all True ({htext.tolist()})")

    print(f"[TEXT A/B/C/D OK] caption_emb_max={g_max:.4f} per_clip_max={[round(x,3) for x in g_per.tolist()]} "
          f"| valid_tokens_per_clip={n_valid_tok.tolist()} token_emb_max={t_max:.4f} "
          f"| has_text={htext.tolist()}")

    # --- E. BEHAVIORAL: text route changes the predicted velocity ---
    with torch.no_grad():
        enc = model.encode(batch)
        vq = model.quantizer(enc["h_lat"], enc["token_mask"], allow_collectives=False)
        z_q = vq["quantized"].float()
        token_mask = enc["token_mask"]
        B2, T_lat, C, _ = z_q.shape

    flow = GraphCodeFlow(code_dim=D, n_heads=ta["n_heads"], d_ff=2 * D, n_layers=5,
                         d_text=768, text_token_dim=768).to(dev)
    flow.eval()
    zf = z_q.reshape(-1, D)[token_mask.reshape(-1)]
    flow.set_latent_stats(zf.mean(0), zf.std(0))

    base_cond = {
        "text_global": cemb,
        "text_tokens": ctok,
        "text_token_mask": ctmask,
        "has_text": htext.clone(),
        "pooled_adjacency": enc["pooled_adjacency"].float(),
        "pooled_geodesic": enc["pooled_geodesic"].float(),
        "pooled_skeleton_embeddings": enc["pooled_skeleton_embeddings"].float(),
        "coarse_mask": enc["coarse_mask"],
        "frame_mask_lat": enc["frame_mask_lat"],
    }
    # fixed z_t, t so the ONLY difference between the two passes is has_text.
    x = flow.normalize(z_q) * token_mask.unsqueeze(-1).float()
    t = torch.full((B2,), 0.5, device=dev)
    t_view = t[:, None, None, None]
    z_t = x * t_view  # deterministic (noise=0): isolates the text contribution

    with torch.no_grad():
        cond_on = dict(base_cond)
        cond_on["has_text"] = torch.ones(B2, dtype=torch.bool, device=dev)
        cond_off = dict(base_cond)
        cond_off["has_text"] = torch.zeros(B2, dtype=torch.bool, device=dev)
        v_on = flow.predict_velocity(z_t, t, cond_on, validate_inputs=True)
        v_off = flow.predict_velocity(z_t, t, cond_off, validate_inputs=True)

    vmask = token_mask.unsqueeze(-1).float()
    delta = ((v_on - v_off).abs() * vmask)
    delta_max = delta.max().item()
    v_on_scale = (v_on.abs() * vmask).max().item()
    rel = delta_max / max(v_on_scale, 1e-8)
    if not (torch.isfinite(v_on).all() and torch.isfinite(v_off).all()):
        fail("E: predict_velocity non-finite")
    if delta_max <= 1e-6:
        fail(f"E: text route is DEAD -- v(has_text=True) == v(has_text=False) "
             f"on valid tokens (max delta {delta_max:.3e}). Text does NOT participate.")
    print(f"[TEXT E OK] text route PARTICIPATES: |v_on - v_off|_max={delta_max:.4f} "
          f"(v_on scale {v_on_scale:.4f}, relative {rel:.3%}) -> non-trivial, text changes velocity")

    print("\n[TEXT-SMOKE] TEXT-POSITIVE CONFIRMED (A,B,C,D,E all pass)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
