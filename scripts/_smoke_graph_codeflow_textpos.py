#!/usr/bin/env python3
"""TEXT-POSITIVE smoke for Graph-CodeFlow L5 text conditioning (validation task).

Proves the dual-text conditioning path of GraphStructuredCodeFlow ACTUALLY
participates, on real L5 val samples that HAVE text (cleanL5 cache), with the
CURRENT frozen VQVAE ckpt.

Assertions (numbers reported):
  (a) caption_emb non-zero for the picked samples.
  (b) caption_token_mask.sum() > 0 for the picked samples.
  (c) BOTH text routes participate vs a text-zeroed control:
        - GLOBAL FiLM route: forward(real text_global) != forward(zeroed text_global)
          [token route held fixed real], measured by max|Δ| over valid tokens.
        - TOKEN cross-attn route: forward(real text_tokens) != forward(zeroed
          text_tokens) [global held fixed real], max|Δ| over valid tokens.
        - COMBINED: forward(real text) != forward(has_text=False) (both routes off).
  (d) CFG cond vs uncond branches differ on text samples: sample() with cond's
      real text vs a forced has_text=False uncond view -> velocity at one ODE step
      differs (max|Δ|).

NOTE on zero-init: GraphStructuredCodeFlow.output_proj, TextCrossAttention.o_proj,
and DenseFiLM are ZERO-INIT (flow-stable identity at init), so a *fresh* model
returns ~0 everywhere and is text-insensitive BY DESIGN. To prove the
ARCHITECTURE actually routes text to the output, we take the fresh model OUT of
its zero-init identity state by overwriting the zero-init output/text/FiLM
parameters with small random weights (explicit, logged). This tests wiring, not
a trained checkpoint. We then run forward() with real vs zeroed text and measure
the divergence on valid tokens.

Runs single-process, eval(), on one idle GPU. DOES NOT launch training. The
frozen VQVAE tokenizer is read-only; no model/tokenizer/training code is touched.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data.anytop_dataset import AnyTopDataset, collate_fn as anytop_collate_fn
from src.models.graph_salad.batch import GraphMotionBatch
from src.models.vq_model import GraphVQTokenizer
from src.models.CodeFlow_Model import GraphCodeFlow

ROOT_DATA = "data/animo4d_anytop_clean_L5"


def fail(msg):
    print(f"[TEXTPOS-SMOKE FAIL] {msg}")
    raise SystemExit(1)


def de_zero_init(flow: GraphCodeFlow, seed: int = 1234) -> int:
    """Overwrite the zero-init output / text-cross-attn-output / FiLM scale+shift
    params with small random weights so the network is NOT in its identity state.
    Returns count of tensors perturbed. Explicit + logged (tests wiring)."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    n = 0
    net = flow.net
    # output projection (zero-init -> v_pred ~ 0 at init)
    for p in (net.output_proj.weight, net.output_proj.bias):
        p.data = (torch.randn(p.shape, generator=g) * 0.02).to(p.device, p.dtype)
        n += 1
    # per-layer: TextCrossAttention.o_proj (zero-init), DenseFiLM scale/shift
    for layer in net.layers:
        for p in (layer.text_cross_attn.o_proj.weight,
                  layer.text_cross_attn.o_proj.bias):
            p.data = (torch.randn(p.shape, generator=g) * 0.05).to(p.device, p.dtype)
            n += 1
        # DenseFiLM commonly zero-inits the final scale/shift projection; perturb
        # whatever Linear params it exposes so FiLM is non-identity.
        for film in (layer.film_after_spatial, layer.film_after_temporal,
                     layer.film_after_text):
            for mod in film.modules():
                if isinstance(mod, nn.Linear):
                    mod.weight.data = (torch.randn(mod.weight.shape, generator=g)
                                       * 0.02).to(mod.weight.device, mod.weight.dtype)
                    if mod.bias is not None:
                        mod.bias.data = (torch.randn(mod.bias.shape, generator=g)
                                         * 0.02).to(mod.bias.device, mod.bias.dtype)
                    n += 1
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frozen_vqvae_ckpt", type=str,
                    default="runs/vqvae_L5_C50_d512_Q4_n512_b96_300ep_seed42/last_model.pt",
                    help="frozen Graph-VQVAE tokenizer ckpt (real L5 run dir's "
                         "last_model.pt); copy to torn-read-safe /tmp via "
                         "--frozen_vqvae_ckpt only if a training is writing it.")
    ap.add_argument("--caption_cache", type=str,
                    default="data/anytop_caption_t5_cleanL5_multi")
    ap.add_argument("--n_text_samples", type=int, default=2)
    ap.add_argument("--scan_limit", type=int, default=200)
    args = ap.parse_args()
    CKPT = args.frozen_vqvae_ckpt
    CAP_EMB = args.caption_cache + ".npz"
    CAP_TOK = args.caption_cache

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={dev}  ckpt={CKPT}")
    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
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
    model.eval(); model.requires_grad_(False)
    D, Q, K = ta["d_model"], ta["num_quantizers"], ta["num_codes"]
    print(f"tokenizer: D={D} Q={Q} K={K} epoch={ck.get('epoch')}")

    ds = AnyTopDataset(
        split="val", num_frames=ta.get("max_frames", 64),
        max_joints=ta.get("max_joints", 64), val_frac=ta.get("val_frac", 0.05),
        seed=ta.get("seed", 42), data_root=ROOT_DATA, load_captions=True,
        caption_emb_cache=CAP_EMB, caption_token_cache=CAP_TOK,
        return_caption_tokens=True, random_caption=False)
    print(f"val dataset: {len(ds)} clips")

    # ---- pick samples that HAVE text ----
    picks = []
    scan = min(args.scan_limit, len(ds))
    n_text_seen = 0
    for i in range(scan):
        it = ds[i]
        emb_nz = float(it["caption_emb"].abs().sum().item())
        tok_sum = int(it["caption_token_mask"].sum().item()) \
            if it.get("caption_token_mask") is not None else 0
        ht = bool(it["has_text"])
        if ht and emb_nz > 0 and tok_sum > 0:
            n_text_seen += 1
            if len(picks) < args.n_text_samples:
                picks.append((i, it, emb_nz, tok_sum))
    print(f"scanned {scan} val clips: {n_text_seen} have text "
          f"(coverage {n_text_seen/scan:.3f}); picked idx={[p[0] for p in picks]}")
    if len(picks) < args.n_text_samples:
        fail(f"only {len(picks)} text-positive val samples in first {scan}")

    items = [p[1] for p in picks]
    # ---- ASSERTION (a) + (b): per-sample non-zero emb + token mask ----
    print("\n=== ASSERTION (a) caption_emb non-zero + (b) token_mask.sum()>0 ===")
    for (i, it, emb_nz, tok_sum) in picks:
        otype = it["object_type"]
        cap = it.get("caption", "")[:60]
        print(f"  [idx {i}] species={otype} caption_emb.abs().sum()={emb_nz:.3f} "
              f"(>0 ✓) caption_token_mask.sum()={tok_sum} (>0 ✓) cap='{cap}...'")
        if not (emb_nz > 0):
            fail(f"(a) idx {i} caption_emb all-zero")
        if not (tok_sum > 0):
            fail(f"(b) idx {i} caption_token_mask empty")
    print("ASSERTION (a) PASS | ASSERTION (b) PASS")

    raw = anytop_collate_fn(items)
    raw = {k: v.to(dev) if torch.is_tensor(v) else v for k, v in raw.items()}
    batch = GraphMotionBatch.from_collate_dict(raw)

    with torch.no_grad():
        enc = model.encode(batch)
        vq = model.quantizer(enc["h_lat"], enc["token_mask"], allow_collectives=False)
        z_q = vq["quantized"].float()
        token_mask = enc["token_mask"]
        B, T_lat, C, _ = z_q.shape
    print(f"\nz_q={list(z_q.shape)} valid_tokens={int(token_mask.sum())}")

    # ---- build flow + de-zero-init so wiring is testable ----
    flow = GraphCodeFlow(code_dim=D, n_heads=ta["n_heads"], d_ff=2 * D, n_layers=5,
                         d_text=768, text_token_dim=768).to(dev)
    flow.eval(); flow.requires_grad_(False)
    n_pert = de_zero_init(flow)
    print(f"de_zero_init: perturbed {n_pert} zero-init tensors "
          f"(output_proj + text_cross o_proj + FiLM) so text routes can reach output")

    # conditioning with REAL text
    cond_real = {
        "text_global": batch.caption_emb.float(),
        "text_tokens": batch.caption_token_emb.float(),
        "text_token_mask": batch.caption_token_mask,
        "has_text": batch.has_text,
        "pooled_adjacency": enc["pooled_adjacency"].float(),
        "pooled_geodesic": enc["pooled_geodesic"].float(),
        "pooled_skeleton_embeddings": enc["pooled_skeleton_embeddings"].float(),
        "coarse_mask": enc["coarse_mask"],
        "frame_mask_lat": enc["frame_mask_lat"],
    }
    vmask = token_mask.unsqueeze(-1).float()
    # fixed t and z_t so the only thing varying is the text inputs
    t_fixed = torch.full((B,), 0.5, device=dev)
    z_t = (z_q * vmask)  # use the real (normalized-identity) latent as z_t probe

    def fwd(cond):
        return flow.predict_velocity(z_t, t_fixed, cond, validate_inputs=True)

    with torch.no_grad():
        v_real = fwd(cond_real)

        # --- GLOBAL FiLM route isolation: zero ONLY text_global ---
        cond_g0 = dict(cond_real)
        cond_g0["text_global"] = torch.zeros_like(cond_real["text_global"])
        v_g0 = fwd(cond_g0)
        d_global = ((v_real - v_g0).abs() * vmask).max().item()

        # --- TOKEN cross-attn route isolation: zero ONLY text_tokens ---
        cond_t0 = dict(cond_real)
        cond_t0["text_tokens"] = torch.zeros_like(cond_real["text_tokens"])
        v_t0 = fwd(cond_t0)
        d_token = ((v_real - v_t0).abs() * vmask).max().item()

        # --- COMBINED: both routes off via has_text=False ---
        cond_off = dict(cond_real)
        cond_off["has_text"] = torch.zeros_like(cond_real["has_text"])
        v_off = fwd(cond_off)
        d_both = ((v_real - v_off).abs() * vmask).max().item()

    print("\n=== ASSERTION (c) BOTH text routes participate (vs text-zeroed) ===")
    print(f"  GLOBAL FiLM route   max|Δv| (real vs zeroed text_global)  = {d_global:.6e}")
    print(f"  TOKEN cross-attn    max|Δv| (real vs zeroed text_tokens)  = {d_token:.6e}")
    print(f"  COMBINED off        max|Δv| (real vs has_text=False)      = {d_both:.6e}")
    TOL = 1e-6
    if not (d_global > TOL):
        fail(f"(c) GLOBAL route did NOT change output (Δ={d_global:.3e} <= {TOL})")
    if not (d_token > TOL):
        fail(f"(c) TOKEN route did NOT change output (Δ={d_token:.3e} <= {TOL})")
    if not (d_both > TOL):
        fail(f"(c) combined text off did NOT change output (Δ={d_both:.3e})")
    print("ASSERTION (c) PASS: both global FiLM and token cross-attn change the output")

    # ---- ASSERTION (d): CFG cond vs uncond branches differ ----
    # sample() internally builds cond_uncond = has_text all-False. To compare cond
    # vs uncond on the SAME z and t (apples-to-apples), call predict_velocity with
    # cond_real vs the same uncond view sample() uses, at one fixed step.
    print("\n=== ASSERTION (d) CFG cond vs uncond differ on text samples ===")
    with torch.no_grad():
        z0 = torch.randn(B, T_lat, C, D, device=dev) * flow.noise_scale * vmask
        t0 = torch.zeros(B, device=dev)
        cond_uncond = dict(cond_real)
        cond_uncond["has_text"] = torch.zeros_like(cond_real["has_text"])
        v_cond = flow.predict_velocity(z0, t0, cond_real, validate_inputs=True)
        v_uncond = flow.predict_velocity(z0, t0, cond_uncond, validate_inputs=True)
        d_cfg = ((v_cond - v_uncond).abs() * vmask).max().item()
        # also exercise the real sampler end-to-end with cfg!=1 (must be finite)
        z_hat = flow.sample(cond_real, token_mask, T_lat, C, steps=2,
                            cfg_scale=4.0, validate_inputs=True)
        finite = bool(torch.isfinite(z_hat).all())
    print(f"  CFG branch max|Δv| (cond vs uncond, same z & t)          = {d_cfg:.6e}")
    print(f"  sample(cfg=4.0, steps=2) z_hat finite                    = {finite}")
    if not (d_cfg > TOL):
        fail(f"(d) CFG cond==uncond (Δ={d_cfg:.3e} <= {TOL})")
    if not finite:
        fail("(d) sample() z_hat non-finite")
    print("ASSERTION (d) PASS: CFG cond and uncond velocities differ; sampler finite")

    print("\n[TEXTPOS-SMOKE] ALL 4 ASSERTIONS (a,b,c,d) PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
