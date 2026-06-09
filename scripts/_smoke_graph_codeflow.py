#!/usr/bin/env python3
"""Graph-CodeFlow 7-step end-to-end smoke (handoff/20260609_0500 review verdict
+ LOCKED recipe). Single-process, eval(), 2 real L5 samples, current VQVAE ckpt.

Steps:
  1. encode -> quantizer -> z_q/indices.
  2. RVQ identity: ids_to_embeddings(indices) ~= quantized on valid, 0 on padded.
  3. projection: z_hat = z_q + small_noise*mask -> nearest_residual_ids ->
     padded -1, z_snap=0, finite projection_error.
  4. decode both: decode(z_q) and decode_from_indices(indices_hat) -> finite [B,T,J,13].
  5. skeleton-only self-transfer (key): decode same z_q with encode() metadata vs
     prepare_skeleton_only() metadata -> assignment/pooled-graph identical, decoded
     motion matches.
  6. one flow step: masked z_t -> predict v -> masked-MSE backward (grads finite)
     -> one ODE update -> projection -> decode -> finite.
  7. (this script) runs on one idle GPU; caller frees it.

DOES NOT launch training.
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
    print(f"[SMOKE FAIL] {msg}")
    raise SystemExit(1)


def de_zero_init(flow, seed: int = 1234) -> int:
    """Take the flow net OUT of its zero-init identity state so the Q2 CFG block is
    meaningful (at init output_proj + all AdaLN-zero gates + coupling/FiLM projs are
    zero -> v_pred==0 everywhere, so cond/uncond cannot differ). Mirrors
    _smoke_graph_codeflow_textpos.de_zero_init's approach (overwrite zero-init
    params with small seeded randoms), generalized to BOTH variants: perturb
    output_proj and EVERY zero-init Linear (all-zero weight) — for graph_pscf these
    are exactly the AdaLNModulation gates + GraphFrameSlotCoupling o_proj/inject_proj
    + DenseFiLM finals; for level_a, output_proj + text-cross o_proj + FiLM. Returns
    #tensors perturbed. Tests WIRING only (a fresh, untrained net)."""
    import torch.nn as nn
    g = torch.Generator(device="cpu").manual_seed(seed)
    n = 0
    net = flow.net
    for p in (net.output_proj.weight, net.output_proj.bias):
        p.data = (torch.randn(p.shape, generator=g) * 0.02).to(p.device, p.dtype)
        n += 1
    for mod in net.modules():
        if isinstance(mod, nn.Linear) and mod is not net.output_proj:
            # only de-zero the zero-init Linears (gates / couplings / FiLM finals);
            # leave already-random Linears untouched.
            if float(mod.weight.abs().sum().item()) == 0.0:
                mod.weight.data = (torch.randn(mod.weight.shape, generator=g)
                                   * 0.02).to(mod.weight.device, mod.weight.dtype)
                if mod.bias is not None:
                    mod.bias.data = (torch.randn(mod.bias.shape, generator=g)
                                     * 0.02).to(mod.bias.device, mod.bias.dtype)
                n += 1
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--frozen_vqvae_ckpt", type=str,
        default="runs/vqvae_L5_C50_d512_Q4_n512_b96_300ep_seed42/last_model.pt",
        help="frozen Graph-VQVAE tokenizer ckpt (the real run dir's last_model.pt)")
    ap.add_argument(
        "--caption_cache", type=str,
        default="data/anytop_caption_t5_cleanL5_multi",
        help="caption cache prefix; emb cache = <prefix>.npz, token cache = <prefix>")
    ap.add_argument(
        "--model_variant", choices=["level_a", "graph_pscf"], default="level_a",
        help="which flow net to build/smoke (default level_a keeps existing "
             "behavior; graph_pscf = 287M formal backbone)")
    ap.add_argument(
        "--num_frames", type=int, default=None,
        help="override dataset num_frames (default: tokenizer ckpt's max_frames); "
             "lets the smoke run on longer clips")
    args = ap.parse_args()
    CKPT = args.frozen_vqvae_ckpt
    CAP_EMB = args.caption_cache + ".npz"
    CAP_TOK = args.caption_cache

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={dev}")
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
    model.eval()
    model.requires_grad_(False)
    D, Q, K = ta["d_model"], ta["num_quantizers"], ta["num_codes"]
    stride = ta["temporal_stride"]
    print(f"tokenizer: D={D} Q={Q} K={K} max_coarse={ta['max_coarse']} stride={stride} "
          f"epoch={ck.get('epoch')}")

    num_frames = args.num_frames if args.num_frames is not None else ta.get("max_frames", 64)
    ds = AnyTopDataset(
        split="val", num_frames=num_frames,
        max_joints=ta.get("max_joints", 64), val_frac=ta.get("val_frac", 0.05),
        seed=ta.get("seed", 42), data_root=ROOT_DATA, load_captions=True,
        caption_emb_cache=CAP_EMB, caption_token_cache=CAP_TOK,
        return_caption_tokens=True, random_caption=False)
    items = [ds[0], ds[1]]
    raw = anytop_collate_fn(items)
    raw = {k: v.to(dev) if torch.is_tensor(v) else v for k, v in raw.items()}
    batch = GraphMotionBatch.from_collate_dict(raw)
    print(f"2 real clips: species={[it['object_type'] for it in items]} "
          f"J={[int(it['num_joints']) for it in items]}")

    with torch.no_grad():
        # ---- STEP 1: encode -> quantizer ----
        enc = model.encode(batch)
        vq = model.quantizer(enc["h_lat"], enc["token_mask"], allow_collectives=False)
        z_q = vq["quantized"].float()
        indices = vq["indices"]
        token_mask = enc["token_mask"]
        B, T_lat, C, _ = z_q.shape
        assert indices.shape == (B, T_lat, C, Q), f"indices shape {tuple(indices.shape)}"
        if not torch.isfinite(z_q).all():
            fail("step1 z_q non-finite")
        print(f"[STEP 1 OK] z_q={list(z_q.shape)} indices={list(indices.shape)} "
              f"n_valid_tok={int(token_mask.sum())}")

        # ---- STEP 2: RVQ identity ----
        z_from_ids = model.ids_to_embeddings(indices, token_mask)
        valid = token_mask.unsqueeze(-1).expand_as(z_from_ids)
        id_err_valid = (z_from_ids[valid] - z_q[valid]).abs().max().item()
        pad_abs = z_from_ids[~valid].abs().max().item() if (~valid).any() else 0.0
        # valid IDs in [0,K-1], padded IDs == -1
        if token_mask.any():
            vi = indices[token_mask]
            if not ((vi >= 0).all() and (vi < K).all()):
                fail("step2 valid indices out of [0,K-1]")
        if (~token_mask).any():
            if not (indices[~token_mask] == -1).all():
                fail("step2 padded indices != -1")
        if id_err_valid > 1e-3:
            fail(f"step2 RVQ-identity valid err {id_err_valid:.3e} > 1e-3")
        if pad_abs != 0.0:
            fail(f"step2 padded z_q != 0 (max {pad_abs:.3e})")
        print(f"[STEP 2 OK] RVQ-identity valid_max_err={id_err_valid:.3e} "
              f"padded_max_abs={pad_abs:.1e} (valid ids in [0,{K-1}], padded ids=-1)")

        # ---- STEP 3: projection of z_hat = z_q + small noise ----
        zhat = z_q + 0.05 * torch.randn_like(z_q) * token_mask.unsqueeze(-1).float()
        proj = model.nearest_residual_ids(zhat, token_mask)
        ih, zsnap, pe = proj["indices_hat"], proj["z_snap"], proj["projection_error"]
        if (~token_mask).any() and not (ih[~token_mask] == -1).all():
            fail("step3 padded indices_hat != -1")
        if (~token_mask).any():
            zsnap_pad = zsnap[~token_mask.unsqueeze(-1).expand_as(zsnap)].abs().max().item()
            if zsnap_pad != 0.0:
                fail(f"step3 padded z_snap != 0 ({zsnap_pad:.3e})")
        if not torch.isfinite(pe):
            fail("step3 projection_error non-finite")
        # internal consistency: ids_to_embeddings(ih) == z_snap on valid
        z_consist = model.ids_to_embeddings(ih, token_mask)
        consist = (z_consist[valid] - zsnap[valid]).abs().max().item()
        if consist > 1e-3:
            fail(f"step3 ids_to_emb(ih)!=z_snap on valid ({consist:.3e})")
        print(f"[STEP 3 OK] projection_error={pe.item():.4f} (finite) padded ids=-1 "
              f"z_snap_pad=0 consistency_err={consist:.1e}")

        # ---- STEP 4: decode both ----
        dec_zq = model.decode(z_q, enc, batch)["pred_motion"]
        dec_snap = model.decode_from_indices(ih, enc, batch)["pred_motion"]
        if not (torch.isfinite(dec_zq).all() and torch.isfinite(dec_snap).all()):
            fail("step4 decode non-finite")
        if dec_zq.shape[-1] != 13 or dec_zq.dim() != 4:
            fail(f"step4 decode shape {tuple(dec_zq.shape)}")
        print(f"[STEP 4 OK] decode(z_q)={list(dec_zq.shape)} finite; "
              f"decode_from_indices(indices_hat)={list(dec_snap.shape)} finite")

        # ---- STEP 5: skeleton-only self-transfer (KEY) ----
        meta = model.prepare_skeleton_only(batch, T_lat)
        # assignment / pooled-graph identical (encode metadata vs skeleton-only).
        a_err = (enc["assignment"] - meta["assignment"]).abs().max().item()
        adj_err = (enc["pooled_adjacency"] - meta["pooled_adjacency"]).abs().max().item()
        geo_e = enc["pooled_geodesic"].clone(); geo_m = meta["pooled_geodesic"].clone()
        # compare finite entries (both share the +inf pattern for unreachable pairs)
        finite = torch.isfinite(geo_e) & torch.isfinite(geo_m)
        geo_err = (geo_e[finite] - geo_m[finite]).abs().max().item() if finite.any() else 0.0
        cm_eq = bool((enc["coarse_mask"] == meta["coarse_mask"]).all())
        skel_eq = (enc["pooled_skeleton_embeddings"]
                   - meta["pooled_skeleton_embeddings"]).abs().max().item()
        if a_err > 1e-5 or adj_err > 1e-5 or geo_err > 1e-5 or not cm_eq or skel_eq > 1e-5:
            fail(f"step5 metadata mismatch assign={a_err:.2e} adj={adj_err:.2e} "
                 f"geo={geo_err:.2e} cm_eq={cm_eq} skel={skel_eq:.2e}")
        # frame_mask_lat differs (skeleton-only is all-True to T_lat); to decode the
        # SAME z_q apples-to-apples we use encode()'s real frame_mask_lat by building
        # a meta-derived dict that swaps in the real frame mask (the geometry is what
        # we proved identical above). Decode same z_q with the skeleton-only geometry.
        meta_for_decode = dict(meta)
        meta_for_decode["frame_mask_lat"] = enc["frame_mask_lat"]
        dec_meta = model.decode(z_q, meta_for_decode, batch)["pred_motion"]
        dec_match = (dec_zq - dec_meta).abs().max().item()
        if dec_match > 1e-4:
            fail(f"step5 decoded motion mismatch (encode-meta vs skeleton-only-meta) "
                 f"max {dec_match:.3e}")
        print(f"[STEP 5 OK] skeleton-only self-transfer: assign/adj/geo/skel identical "
              f"(<=1e-5), coarse_mask equal, decoded motion match max_err={dec_match:.1e}")

    # ---- STEP 6: one flow step (needs grad) ----
    if args.model_variant == "graph_pscf":
        flow = GraphCodeFlow(code_dim=D, n_heads=ta["n_heads"], d_ff=4 * D,
                             model_variant="graph_pscf", depth_double=6,
                             depth_single=12, max_T_lat=75, mlp_ratio=4.0,
                             d_text=768, text_token_dim=768).to(dev)
    else:
        flow = GraphCodeFlow(code_dim=D, n_heads=ta["n_heads"], d_ff=2 * D, n_layers=5,
                             d_text=768, text_token_dim=768).to(dev)
    print(f"flow variant={args.model_variant} "
          f"params={sum(p.numel() for p in flow.parameters()):,}")
    flow.train()
    # empirical-ish stats from this batch's valid tokens (smoke; trainer uses the
    # full train cache).
    zf = z_q.reshape(-1, D)[token_mask.reshape(-1)]
    flow.set_latent_stats(zf.mean(0), zf.std(0))
    cond = {
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
    r = flow.flow_loss(z_q, token_mask, cond, validate_inputs=True)
    loss = r["flow_loss"]
    if not torch.isfinite(loss):
        fail("step6 flow_loss non-finite")
    loss.backward()
    gnorm = torch.nn.utils.clip_grad_norm_(flow.parameters(), 1.0)
    n_with_grad = sum(1 for p in flow.parameters() if p.grad is not None)
    all_grad_finite = all(torch.isfinite(p.grad).all() for p in flow.parameters()
                          if p.grad is not None)
    if not (torch.isfinite(gnorm) and all_grad_finite):
        fail("step6 grads non-finite")
    print(f"[STEP 6a OK] flow_loss={loss.item():.4f} backward grad_norm={gnorm.item():.3f} "
          f"({n_with_grad} params with finite grad)")

    # one ODE update + projection + decode (eval)
    flow.eval()
    with torch.no_grad():
        z_hat_ode = flow.sample(cond, token_mask, T_lat, C, steps=2,
                                cfg_scale=4.0, validate_inputs=True)
        if not torch.isfinite(z_hat_ode).all():
            fail("step6 ODE z_hat non-finite")
        proj2 = flow.normalize  # noqa: F841 (sanity that buffers exist)
        pr = model.nearest_residual_ids(z_hat_ode, token_mask)
        dec_ode = model.decode_from_indices(pr["indices_hat"], enc, batch)["pred_motion"]
        if not torch.isfinite(dec_ode).all():
            fail("step6 ODE decode non-finite")
    print(f"[STEP 6b OK] ODE(2 steps)+CFG z_hat finite, projection_error="
          f"{pr['projection_error'].item():.4f}, decode_from_indices finite "
          f"{list(dec_ode.shape)}")

    # ---- STEP 7: Q2 CFG smoke (uncond-invariance + cond-differs) ----
    # Only needs the flow net (no GPU-heavy decode). At init the net is in its
    # zero-init identity state (output_proj + all AdaLN-zero gates + coupling/FiLM
    # projs == 0 -> v_pred==0), so we de-zero those tensors first (same approach as
    # _smoke_graph_codeflow_textpos) to make cond/uncond distinguishable.
    flow.eval(); flow.requires_grad_(False)
    n_pert = de_zero_init(flow)
    print(f"[STEP 7] de_zero_init perturbed {n_pert} zero-init tensors "
          f"(output_proj + AdaLN/coupling/FiLM gates) so text/cond can reach output")
    vmask = token_mask.unsqueeze(-1).float()
    # FIXED z_t and t so the only thing varying is the text/has_text inputs.
    t_fixed = torch.full((B,), 0.5, device=dev)
    z_t_fixed = z_q * vmask
    # Two DIFFERENT text tensors ("dragon" vs "seal") — distinct seeded randoms of
    # the real text shapes — both run with has_text ALL-FALSE (uncond).
    g7 = torch.Generator(device="cpu").manual_seed(7)
    tg = batch.caption_emb.float()
    tt = batch.caption_token_emb.float()
    text_global_a = torch.randn(tg.shape, generator=g7).to(dev)   # "dragon"
    text_tokens_a = torch.randn(tt.shape, generator=g7).to(dev)
    text_global_b = torch.randn(tg.shape, generator=g7).to(dev)   # "seal"
    text_tokens_b = torch.randn(tt.shape, generator=g7).to(dev)
    has_text_off = torch.zeros_like(batch.has_text)
    has_text_on = torch.ones_like(batch.has_text)

    def make_cond(tg_, tt_, ht_):
        c = dict(cond)
        c["text_global"] = tg_
        c["text_tokens"] = tt_
        c["has_text"] = ht_
        return c

    with torch.no_grad():
        # (A) uncond invariance: two different text tensors, both has_text=False ->
        # IDENTICAL velocity (proves uncond truly drops text).
        v_unc_a = flow.predict_velocity(
            z_t_fixed, t_fixed, make_cond(text_global_a, text_tokens_a, has_text_off),
            validate_inputs=True)
        v_unc_b = flow.predict_velocity(
            z_t_fixed, t_fixed, make_cond(text_global_b, text_tokens_b, has_text_off),
            validate_inputs=True)
        d_uncond = ((v_unc_a - v_unc_b).abs() * vmask).max().item()
        # (B) cond differs: same text, has_text=True vs has_text=False -> velocity on
        # valid tokens differs non-trivially.
        v_cond_on = flow.predict_velocity(
            z_t_fixed, t_fixed, make_cond(text_global_a, text_tokens_a, has_text_on),
            validate_inputs=True)
        v_cond_off = flow.predict_velocity(
            z_t_fixed, t_fixed, make_cond(text_global_a, text_tokens_a, has_text_off),
            validate_inputs=True)
        d_cond = ((v_cond_on - v_cond_off).abs() * vmask).max().item()

    TOL = 1e-6
    print(f"[STEP 7] (A) uncond-invariance max|Δv| (two texts, has_text=False) = "
          f"{d_uncond:.3e}  -> {'PASS' if d_uncond <= TOL else 'FAIL'} (expect ~0)")
    print(f"[STEP 7] (B) cond-differs   max|Δv| (has_text True vs False)      = "
          f"{d_cond:.3e}  -> {'PASS' if d_cond > TOL else 'FAIL'} (expect > {TOL})")
    if d_uncond > TOL:
        fail(f"step7 (A) uncond NOT text-invariant (Δ={d_uncond:.3e} > {TOL}): "
             f"dropping text did not drop both streams")
    if not (d_cond > TOL):
        fail(f"step7 (B) cond==uncond (Δ={d_cond:.3e} <= {TOL}): text conditioning "
             f"does not change velocity")
    print("[STEP 7 OK] Q2 CFG: uncond text-invariant + cond differs from uncond")

    # ---- STEP 8: Gate-2 "graph is actually used" (graph-bias sensitivity) ----
    # Proves the pooled_geodesic graph bias REACHES the output (the slot stream's
    # GraphAttentionBlock), not just that the model runs. Reuses the SAME real batch
    # + the de_zero_init'd flow net from STEP 7 (so the output is not trivially zero
    # at init), with a FIXED z_t and t — the ONLY thing we vary is pooled_geodesic.
    # We zero the geodesic (cond_zero) and feed it with validate_inputs=False so the
    # deliberately-inconsistent geo does NOT trip GraphAttentionBlock's Floyd check;
    # we are probing sensitivity, not feeding a valid graph. If the geodesic bias is
    # wired through, real vs zeroed geo must change the velocity on valid tokens.
    # GraphAttentionBlock adds geo_bias AND adj_bias separately to the attention
    # scores (attention.py), so BOTH pooled_geodesic and pooled_adjacency must be
    # probed independently — zeroing only one would miss an unwired adjacency (or
    # unwired geodesic) and could misjudge an adj-only model as "graph unused".
    with torch.no_grad():
        v_real = flow.predict_velocity(
            z_t_fixed, t_fixed, cond, validate_inputs=False)
        cond_zg = dict(cond)
        cond_zg["pooled_geodesic"] = torch.zeros_like(cond["pooled_geodesic"])
        v_zg = flow.predict_velocity(z_t_fixed, t_fixed, cond_zg, validate_inputs=False)
        d_geo = ((v_real - v_zg).abs() * vmask).max().item()
        cond_za = dict(cond)
        cond_za["pooled_adjacency"] = torch.zeros_like(cond["pooled_adjacency"])
        v_za = flow.predict_velocity(z_t_fixed, t_fixed, cond_za, validate_inputs=False)
        d_adj = ((v_real - v_za).abs() * vmask).max().item()

    both_ok = d_geo > TOL and d_adj > TOL
    print(f"[STEP 8] graph-used max|Δv|: geo={d_geo:.3e} adj={d_adj:.3e} (valid tokens) "
          f"-> {'PASS' if both_ok else 'FAIL'} (each expect > {TOL})")
    if not (d_geo > TOL):
        fail(f"step8 geodesic NOT used (Δ={d_geo:.3e} <= {TOL}): zeroing pooled_geodesic "
             f"did not change velocity -> geodesic bias does not reach the output")
    if not (d_adj > TOL):
        fail(f"step8 adjacency NOT used (Δ={d_adj:.3e} <= {TOL}): zeroing pooled_adjacency "
             f"did not change velocity -> adjacency bias does not reach the output")
    print("[STEP 8 OK] Gate-2: BOTH pooled_geodesic and pooled_adjacency reach the output")

    print("\n[SMOKE] ALL 8 STEPS PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
