"""Phase-2 animate_denoiser.py — render samples from a trained denoiser.

Pipeline per docs/phase2_diffusion_design.md §4-5:
  1. Load frozen VAE (use_text=False) and trained denoiser ckpt.
  2. For each requested batch:
     - vae.encode_skeleton_only(batch) → coarse_mask / pooled_adj / pooled_geo /
       pooled_skeleton_embeddings / anchor_indices / hard_assignment / assignment / s_j
     - frame_mask_lat = batch.frame_mask.view(B, T_lat, stride).all(-1)
     - z_T = N(0, I) of shape [B, T_lat, C, D]
     - DDIM sampling loop (default 50 steps) with CFG (default cond_scale=1.5;
       ALWAYS pass --cond_scale explicitly when comparing renders):
         z2 = cat(z, z, dim=0); t2 = cat(t, t); has_text2 = cat(True, False);
         text2 = cat(text, text*0); other tensors all repeated to 2B
         v_2 = denoiser(z_2, ...)  → split into v_cond / v_uncond
         v = v_uncond + cond_scale * (v_cond - v_uncond)
         z = sched.step(v, t, z).prev_sample
     - Build fake_enc dict (z = denoised + other skeleton bits) → vae.decode
     - De-normalize anytop13 pred_motion → world positions → gif
  3. Render per-species GT vs. pred (visual QA primacy rule).

Usage:
  python scripts/animate_denoiser.py \\
      --vae_ckpt runs/m1_7_anytop13_coarse_xattn_seed42/best_recon_model.pt \\
      --denoiser_ckpt runs/m2_denoiser_v1_seed42/best_model.pt \\
      --caption_emb_cache data/anytop_caption_t5_1070.npz \\
      --species Alligator,Spider,Trex,Dragon --n_per 2 \\
      --out runs/m2_denoiser_v1_seed42/qa_sample
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import numpy as np
import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.animate import animate_clip, contact_sheet, animate_t2m_input_pred, fk_rest_pose
from src.data.anytop_dataset import (
    AnyTopDataset, collate_fn as anytop_collate_fn,
    _recover_world_positions, _STD_FLOOR,
)
from src.models.graph_salad.batch import GraphMotionBatch
from src.models.graph_salad.vae import GraphMotionVAE
from src.models.graph_salad.denoiser import GraphSaladDenoiser
from src.models.graph_salad.rot6d_fk_recovery import recover_rot6d_fk_positions_torch
from scripts.train_denoiser import load_frozen_vae

from diffusers import DDIMScheduler


def load_denoiser(ckpt_path: str, dev: torch.device) -> tuple[GraphSaladDenoiser, dict]:
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    da = ck.get("args", {})
    vae_ta = ck.get("vae_ckpt_args", {})
    d_model = vae_ta.get("d_model", 384)
    n_heads = vae_ta.get("n_heads", 8)
    d_ff = da.get("d_ff") or 4 * d_model
    # M2: rebuild with the ckpt's text_mode (mean ckpts → 'mean_additive' default;
    # token ckpts carry 'token_cross_attn' in args). Wrong mode ⇒ arch mismatch ⇒
    # strict-load fails loud below.
    text_mode = da.get("text_mode", "mean_additive")
    spatial_mode = da.get("spatial_mode", "graph")  # old ckpts (no key) → graph
    denoiser = GraphSaladDenoiser(
        d_model=d_model, n_heads=n_heads, d_ff=d_ff,
        n_layers=da.get("n_layers", 5),
        d_text=768, dropout=da.get("dropout", 0.1),
        text_mode=text_mode, text_token_dim=768,
        spatial_mode=spatial_mode,
    ).to(dev)
    missing, unexpected = denoiser.load_state_dict(ck["model_state_dict"], strict=True)
    if missing or unexpected:
        raise SystemExit(
            f"Denoiser ckpt strict-load failed: missing={len(missing)} unexpected={len(unexpected)}"
        )
    denoiser.eval()
    return denoiser, ck


@torch.no_grad()
def ddim_sample(
    denoiser: GraphSaladDenoiser,
    batch: GraphMotionBatch,
    skel: dict,
    frame_mask_lat: torch.Tensor,
    n_steps: int,
    cond_scale: float,
    sched_kwargs: dict,
    dev: torch.device,
    d_model: int,
) -> torch.Tensor:
    """Run DDIM sampling with classifier-free guidance.

    Returns z_0 [B, T_lat, C, D].
    """
    B = skel["pooled_adjacency"].shape[0]
    C = skel["pooled_adjacency"].shape[1]
    T_lat = frame_mask_lat.shape[1]

    sched = DDIMScheduler(**sched_kwargs)
    sched.set_timesteps(n_steps, device=dev)
    # Initialize z_T ~ N(0, I); mask padded positions
    z = torch.randn(B, T_lat, C, d_model, device=dev)
    mask_4d = (skel["coarse_mask"][:, None, :, None] & frame_mask_lat[:, :, None, None]).to(z.dtype)
    z = z * mask_4d

    # Repeat conditioning to 2B for CFG cond+uncond batching
    adj2 = skel["pooled_adjacency"].repeat(2, 1, 1)
    geo2 = skel["pooled_geodesic"].repeat(2, 1, 1)
    cm2 = skel["coarse_mask"].repeat(2, 1)
    fm2 = frame_mask_lat.repeat(2, 1)
    skel2 = skel["pooled_skeleton_embeddings"].repeat(2, 1, 1)
    has_text_cond = batch.has_text.to(dev)              # [B] bool
    has_text_uncond = torch.zeros_like(has_text_cond, dtype=torch.bool)
    has_text2 = torch.cat([has_text_cond, has_text_uncond], dim=0)  # [2B]
    # M2: mode-dependent text, repeated 2x for the CFG cond+uncond batch. The uncond
    # half's has_text=False zeroes the global add AND fully masks the token keys
    # (cross-attn → 0), so both streams CFG-drop together (dual_text).
    text_mode = getattr(denoiser, "text_mode", "mean_additive")
    text_tokens2 = None
    if text_mode == "dual_text":
        text2 = batch.caption_emb.to(dev).repeat(2, 1)                   # [2B, 768] global
        text_tokens2 = batch.caption_token_emb.to(dev).repeat(2, 1, 1)  # [2B, L, 768] tokens
        token_mask2 = batch.caption_token_mask.to(dev).repeat(2, 1)     # [2B, L]
    elif text_mode == "token_cross_attn":
        text2 = batch.caption_token_emb.to(dev).repeat(2, 1, 1)         # [2B, L, 768]
        token_mask2 = batch.caption_token_mask.to(dev).repeat(2, 1)     # [2B, L]
    else:  # mean_additive
        text2 = batch.caption_emb.to(dev).repeat(2, 1)                  # [2B, 768]
        token_mask2 = None

    first = True
    for t in sched.timesteps:
        # Build cond+uncond batch
        z2 = torch.cat([z, z], dim=0)                    # [2B, T_lat, C, D]
        t2 = torch.full((2 * B,), int(t.item()), device=dev, dtype=torch.long)
        v2 = denoiser(
            z_t=z2, timesteps=t2, text=text2,
            adjacency=adj2, geodesic_dist=geo2,
            coarse_mask=cm2, frame_mask=fm2,
            pooled_skeleton_embeddings=skel2,
            has_text=has_text2,
            text_token_mask=token_mask2,
            text_tokens=text_tokens2,
            validate_inputs=first,  # cold-start validate on first iter
        )
        first = False
        v_cond, v_uncond = v2.chunk(2, dim=0)
        v = v_uncond + cond_scale * (v_cond - v_uncond)
        z = sched.step(v, t, z).prev_sample
        # Re-mask padded after step (defense in depth)
        z = z * mask_4d
    return z


def make_fake_enc(z: torch.Tensor, skel: dict, frame_mask_lat: torch.Tensor) -> dict:
    """Build a dict shaped like vae.encode()'s output, with z replaced by the
    denoised sample. vae.decode() will pull s_j, assignment, coarse_mask,
    frame_mask_lat, z out of this dict.
    """
    return {
        "z": z,
        "s_j": skel["s_j"],
        "assignment": skel["assignment"],
        "coarse_mask": skel["coarse_mask"],
        "frame_mask_lat": frame_mask_lat,
        # Decode only needs the above 5; carry the rest for downstream completeness
        "pooled_adjacency": skel["pooled_adjacency"],
        "pooled_geodesic": skel["pooled_geodesic"],
        "pooled_skeleton_embeddings": skel["pooled_skeleton_embeddings"],
        "anchor_indices": skel["anchor_indices"],
        "hard_assignment": skel["hard_assignment"],
        "mu": z,
        "logvar": torch.zeros_like(z),
        "aux_losses": None,
    }


def make_t2m_large_gif(pred_ric, pred_fk, static_pose, parents, prompt, out_path,
                       max_frames=48, fps=12, cell=(900, 760), zoom=1.15, pad=0.06,
                       gt=None,
                       pred_labels=("PRED pose/RIC 0:3", "PRED rot6d-FK 3:9")):
    """Large-figure (PIL oblique, per-frame root-centered) T2M demo:
    input skeleton (static grey) | PRED pose/RIC (blue) | PRED rot6d-FK (green),
    with the prompt as a top header band. NO GT (T2M input = skeleton + prompt).
    recover already done by caller via src funcs; this is geometry/drawing only.

    Diagnostic (gt given, [T,J,3] world-space): the animated GT source motion is
    stitched on as the RIGHTMOST panel → input | PRED_RIC | PRED_FK | GT, so
    generated motion can be eyeballed against the real dataset clip. GT shares the
    panels' common scale, so a fast/janky PRED reads visually against a smooth GT."""
    import scripts._pil_skeleton_render as pr
    T = pred_ric.shape[0]
    static_T = np.repeat(np.asarray(static_pose)[None], T, axis=0)   # [J,3] -> [T,J,3]
    arrs = [(static_T, "input skeleton (rest)", (90, 90, 90), True, True),
            (pred_ric, pred_labels[0], (35, 112, 180), False, False),
            (pred_fk, pred_labels[1], (30, 150, 55), False, False)]
    if gt is not None:
        arrs.append((np.asarray(gt), "GT source 0:3", (200, 60, 60), False, False))
    norm = []
    for a, title, color, axes, static in arrs:
        aa = a.astype(np.float64).copy(); aa[..., 1] -= aa[..., 1].min()
        norm.append((aa, title, color, axes, static))
    idxs = pr.sample_indices(T, max_frames)
    ps = []
    for aa, *_ in norm:
        c = aa.copy(); roots = c[:, 0].copy()
        c[..., 0] -= roots[:, None, 0]; c[..., 2] -= roots[:, None, 2]
        ps += [c[k] for k in idxs]
    transform = pr.compute_transform(ps, cell, pad, zoom)
    frames = []
    for k in idxs:
        panels = [{"positions": aa, "parents": parents, "title": t,
                   "color": col, "axes": ax, "static": st}
                  for (aa, t, col, ax, st) in norm]
        frames.append(pr.make_row_frame(panels, k, transform, cell, 3, 4,
                                        header=prompt, header_h=84))
    pr.save_gif(frames, out_path, fps)


_SEX_SUFFIX = {"female", "male", "juvenile", "adult", "baby", "calf", "infant",
               "pup", "chick", "cub", "kit", "foal", "joey"}


def make_generic_caption(caption: str, object_type: str) -> str:
    """Replace the species subject with 'An animal', keep the action verb+rest.
    'A female aardvark runs forward while turning right.' (PZ_Aardvark_Female)
      -> 'An animal runs forward while turning right.'
    Locates the action by finding the species' last word (from object_type,
    minus PZ_ prefix + sex suffix) in the caption; action = everything after it."""
    parts = object_type.split("_")[1:]  # drop PZ_
    if parts and parts[-1].lower() in _SEX_SUFFIX:
        species = [p.lower() for p in parts[:-1]]
    else:
        species = [p.lower() for p in parts]
    if not species or not caption:
        return caption
    import re
    cap_low = caption.lower()
    # word-boundary match (full species phrase first, then last token); take the
    # LAST whole-word match so "cat" doesn't bite into "catches", "bear" not "bears".
    for phrase in (" ".join(species), species[-1]):
        last = None
        for m in re.finditer(r"\b" + re.escape(phrase) + r"\b", cap_low):
            last = m
        if last is not None:
            action = caption[last.end():].strip()
            return ("An animal " + action).strip() if action else caption
    return caption  # species not found as a whole word → leave as-is


_T5_CACHE: dict = {}


def _t5_encode(text: str, dev: torch.device) -> torch.Tensor:
    """Mean-pooled T5-base embedding [1,768] for one caption, mirroring
    precompute_t5_captions.py (the same pooling AnyTop T5Conditioner uses).
    T5-base loads from local HF cache (offline ok)."""
    if "model" not in _T5_CACHE:
        import os
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        from transformers import T5EncoderModel, T5TokenizerFast
        _T5_CACHE["tok"] = T5TokenizerFast.from_pretrained("t5-base", local_files_only=True)
        _T5_CACHE["model"] = T5EncoderModel.from_pretrained("t5-base", local_files_only=True).to(dev).eval()
        for p in _T5_CACHE["model"].parameters():
            p.requires_grad_(False)
    tok, model = _T5_CACHE["tok"], _T5_CACHE["model"]
    with torch.no_grad():
        enc = tok([text], padding=True, truncation=True, max_length=64,
                  return_tensors="pt").to(dev)
        hidden = model(input_ids=enc["input_ids"],
                       attention_mask=enc["attention_mask"]).last_hidden_state
        mask = enc["attention_mask"].unsqueeze(-1).to(hidden.dtype)
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-6)
    return pooled  # [1, 768]


def _t5_encode_tokens(text: str, dev: torch.device, max_len: int = 64
                      ) -> tuple[torch.Tensor, torch.Tensor]:
    """Token-level T5-base for ONE caption → (token_emb [1,L,768], mask [1,L]),
    mirroring scripts/precompute_t5_caption_tokens.py (padding='max_length',
    pad-token rows zeroed). For token_cross_attn custom/generic prompts — do NOT
    mean-pool in token mode (plan §3.7)."""
    if "model" not in _T5_CACHE:
        import os
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        from transformers import T5EncoderModel, T5TokenizerFast
        _T5_CACHE["tok"] = T5TokenizerFast.from_pretrained("t5-base", local_files_only=True)
        _T5_CACHE["model"] = T5EncoderModel.from_pretrained("t5-base", local_files_only=True).to(dev).eval()
        for p in _T5_CACHE["model"].parameters():
            p.requires_grad_(False)
    tok, model = _T5_CACHE["tok"], _T5_CACHE["model"]
    with torch.no_grad():
        enc = tok([text], padding="max_length", truncation=True, max_length=max_len,
                  return_tensors="pt").to(dev)
        hidden = model(input_ids=enc["input_ids"],
                       attention_mask=enc["attention_mask"]).last_hidden_state  # [1,L,768]
        amask = enc["attention_mask"].bool()                                    # [1,L]
        hidden = hidden * amask.unsqueeze(-1).to(hidden.dtype)
    return hidden.float(), amask  # [1,L,768], [1,L]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vae_ckpt", required=True)
    ap.add_argument("--denoiser_ckpt", required=True)
    ap.add_argument("--caption_emb_cache", required=True)
    ap.add_argument("--caption_token_cache", default=None,
                    help="token cache prefix (<prefix>.tokens.npy + .token_mask.npy "
                         "+ .keys.json); REQUIRED when the denoiser ckpt is "
                         "token_cross_attn.")
    ap.add_argument("--caption_token_max_len", type=int, default=64)
    ap.add_argument("--out", required=True)
    ap.add_argument("--anytop_root", default=None)
    ap.add_argument("--split", default="val")
    ap.add_argument("--species", default="Alligator,Spider,Trex,Dragon",
                    help="comma-separated species to render")
    ap.add_argument("--n_per", type=int, default=2)
    ap.add_argument("--n_ddim_steps", type=int, default=50)
    ap.add_argument("--cond_scale", type=float, default=1.5)
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--fps", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--large", action="store_true",
                    help="big PIL figures (input|PRED_RIC|PRED_FK + prompt) via _pil_skeleton_render")
    ap.add_argument("--generic_prompt", action="store_true",
                    help="replace species name with 'an animal' (keep action), re-encode via T5-base")
    ap.add_argument("--with_gt", action="store_true",
                    help="diagnostic: prepend the GT source-motion panel "
                         "(GT 0:3 | PRED_RIC | PRED_FK) so generated motion can be "
                         "eyeballed against the real dataset clip. --large only.")
    args = ap.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        print("  [INFO] CUDA unavailable; falling back to CPU"); args.device = "cpu"
    dev = torch.device(args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if dev.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # VAE
    print(f"Loading frozen VAE: {args.vae_ckpt}")
    vae, ta = load_frozen_vae(args.vae_ckpt, dev)
    feat_mode = ta["feat_mode"]
    if feat_mode != "anytop13":
        raise SystemExit(f"animate_denoiser supports feat_mode=anytop13 only, got {feat_mode}")
    d_model = ta["d_model"]
    temporal_stride = ta["temporal_stride"]

    # Denoiser
    print(f"Loading denoiser: {args.denoiser_ckpt}")
    denoiser, dck = load_denoiser(args.denoiser_ckpt, dev)
    da = dck.get("args", {})
    print(f"  denoiser params: {sum(p.numel() for p in denoiser.parameters()):,}")
    print(f"  denoiser ckpt epoch={dck.get('epoch', '?')} val_denoise={dck.get('val_denoise', '?')}")

    # Full-motion mode (2026-05-25): use denoiser ckpt's max_frames (NOT VAE's).
    # Old single-window ckpts default to 64; new full-motion ckpts use 260.
    # fail-loud divisibility check — bad ckpt or version skew would otherwise
    # crash later at frame_mask.view(B, T/stride, stride) with cryptic error.
    denoiser_max_frames = da.get("max_frames", 64)
    if denoiser_max_frames % temporal_stride != 0:
        raise SystemExit(
            f"[ARGS FAIL] denoiser ckpt max_frames={denoiser_max_frames} "
            f"not divisible by VAE temporal_stride={temporal_stride}. "
            f"Bad ckpt or version skew."
        )
    print(f"  denoiser ckpt max_frames={denoiser_max_frames} "
          f"→ T_lat={denoiser_max_frames // temporal_stride}")

    sched_kwargs = dict(
        num_train_timesteps=da.get("num_train_timesteps", 1000),
        beta_start=da.get("beta_start", 0.00085),
        beta_end=da.get("beta_end", 0.012),
        beta_schedule=da.get("beta_schedule", "scaled_linear"),
        prediction_type="v_prediction",
        clip_sample=False,
    )

    # Dataset
    cap_cache = args.caption_emb_cache
    anytop_root = args.anytop_root or ta.get("anytop_root")
    # M2: token ckpts need the token cache + return_caption_tokens so the dataset
    # emits caption_token_emb/mask aligned to the same caption idx as caption_emb.
    da_text_mode = da.get("text_mode", "mean_additive")
    use_tokens = da_text_mode in ("token_cross_attn", "dual_text")
    if use_tokens and not args.caption_token_cache:
        raise SystemExit(
            f"denoiser ckpt is {da_text_mode} but --caption_token_cache not "
            "given (need the token cache to sample dataset captions)."
        )
    ds_kwargs = dict(
        split=args.split,
        num_frames=denoiser_max_frames,
        max_joints=ta.get("max_joints", 143),
        caption_emb_cache=cap_cache,
        caption_token_cache=args.caption_token_cache,
        return_caption_tokens=use_tokens,
        caption_token_max_len=args.caption_token_max_len,
    )
    if anytop_root:
        ds_kwargs["data_root"] = anytop_root
    ds = AnyTopDataset(**ds_kwargs)

    # P2 (codex 2026-05-23): preflight caption coverage so we don't silently
    # render uncond samples for clips with missing T5 cache entries (the
    # AnyTopDataset zero-fills caption_emb + sets has_text=False on cache miss).
    n_missing = 0
    want_set = set(s.strip() for s in args.species.split(",") if s.strip())
    # Only touch clips of the requested species. object_type lives in the dataset
    # index (ds.samples[i]) and needs NO motion load, so a species-filtered render
    # iterates ~dozens of clips instead of walking the whole split (train=77882).
    match_indices = [i for i, s in enumerate(ds.samples)
                     if s.get("object_type") in want_set]
    for i in match_indices:
        it = ds[i]
        if it["object_type"] not in want_set:
            continue
        if args.generic_prompt:
            # generic mode re-encodes 'an animal ...' via T5 at runtime → only needs
            # a non-empty caption string, NOT the precomputed T5 cache / has_text.
            if not (it.get("caption") or "").strip():
                n_missing += 1
        elif not bool(it.get("has_text", False)):
            n_missing += 1
    if n_missing > 0:
        if args.generic_prompt:
            raise SystemExit(
                f"[animate preflight] {n_missing} requested-species clips have an "
                f"empty caption (--generic_prompt needs a caption string to genericize)."
            )
        raise SystemExit(
            f"[animate preflight] {n_missing} requested-species clips have "
            f"has_text=False (missing from T5 cache {cap_cache}). Re-run "
            f"precompute_t5_captions.py to cover them, or filter --species."
        )
    print(f"  [preflight] all requested-species clips have valid "
          f"{'caption strings (generic)' if args.generic_prompt else 'T5 caption emb'}")

    want = [s.strip() for s in args.species.split(",") if s.strip()]
    picked = {s: 0 for s in want}
    summary: list[str] = []

    print(f"\nSampling: DDIM {args.n_ddim_steps} steps, CFG cond_scale={args.cond_scale}")
    for i in match_indices:  # species-filtered (see preflight); no full-split walk
        item = ds[i]
        sp = item["object_type"]
        if sp not in picked or picked[sp] >= args.n_per:
            if all(picked[s] >= args.n_per for s in want):
                break  # all requested species collected → stop (don't walk all of train)
            continue
        raw = anytop_collate_fn([item])
        raw = {k: v.to(dev) if torch.is_tensor(v) else v for k, v in raw.items()}
        batch = GraphMotionBatch.from_collate_dict(raw)
        gen_caption = None
        if args.generic_prompt:
            gen_caption = make_generic_caption(item.get("caption") or "", sp)
            # dual_text re-encodes BOTH streams; token/mean re-encode only their own.
            if da_text_mode in ("token_cross_attn", "dual_text"):
                # encode token-level T5 (NOT mean-pool); override token emb + mask.
                te, tm = _t5_encode_tokens(gen_caption, dev, args.caption_token_max_len)
                batch.caption_token_emb = te                        # [1,L,768]
                batch.caption_token_mask = tm                       # [1,L] bool
            if da_text_mode in ("mean_additive", "dual_text"):
                batch.caption_emb = _t5_encode(gen_caption, dev)    # override [1,768]
            batch.has_text = torch.ones_like(batch.has_text)        # force conditioned

        # Skeleton conditioning (no motion needed)
        with torch.no_grad():
            skel = vae.encode_skeleton_only(batch)
        frame_mask_lat = batch.frame_mask.view(
            1, batch.frame_mask.shape[1] // temporal_stride, temporal_stride
        ).all(dim=-1)

        # DDIM sampling
        z = ddim_sample(
            denoiser, batch, skel, frame_mask_lat,
            n_steps=args.n_ddim_steps, cond_scale=args.cond_scale,
            sched_kwargs=sched_kwargs, dev=dev, d_model=d_model,
        )
        # Decode latent → motion
        fake_enc = make_fake_enc(z, skel, frame_mask_lat)
        with torch.no_grad():
            dec = vae.decode(fake_enc, batch)
        pred_motion = dec["pred_motion"]  # [B, T, J, 13]

        # De-normalize + recover world positions
        # P1 (codex 2026-05-23): T_vis must respect the stride-aware frame_mask.
        # Some val clips (67/215) have num_frames < 64 or num_frames not divisible
        # by temporal_stride=4 → the last (stride-incomplete) latent frame is
        # masked off in frame_mask_lat, so vae.decode zeros that range. Visualizing
        # item["num_frames"] would include those zeroed tails.
        J = int(item["num_joints"])
        T_clip = int(item["num_frames"])
        T_valid = int(frame_mask_lat[0].sum().item() * temporal_stride)
        T = min(T_clip, T_valid)
        std = raw["anytop_std"][0, :J].cpu().numpy()
        mean = raw["anytop_mean"][0, :J].cpu().numpy()
        pred_norm = pred_motion[0, :T, :J, :].cpu().numpy()
        pred_raw = pred_norm * (std[None] + _STD_FLOOR) + mean[None]
        pred_world = _recover_world_positions(pred_raw)          # pose/RIC route (ch0:3)
        gt_world = batch.motion_features[0, :T, :J, :3].cpu().numpy()
        parents = [int(p) for p in item["parent_indices"][:J]]
        rest_off = raw["rest_offsets"][0, :J].cpu().numpy()

        # rot6d-FK route (ch3:9 6D rot → FK): recover the SAME generated motion
        # via the rotation channels, to compare against the pose/RIC route above.
        # They should match if the generated 6D rot + local pos are self-consistent.
        pred_raw_t = torch.from_numpy(pred_raw).float()[None]    # [1,T,J,13]
        rest_off_t = torch.from_numpy(rest_off).float()[None]    # [1,J,3]
        jmask_t = torch.ones(1, J, dtype=torch.bool)
        pred_world_fk = recover_rot6d_fk_positions_torch(
            pred_raw_t, [parents], rest_off_t, jmask_t
        )[0].cpu().numpy()                                       # [T,J,3]

        k = picked[sp]
        gif_path = out_dir / f"{sp}_clip{k}_t2m.gif"
        actual_gif_path = (out_dir / f"{sp}_clip{k}_t2m_large.gif") if args.large else gif_path
        g_spd = float(np.linalg.norm(np.diff(gt_world, axis=0), axis=-1).mean())
        p_spd = float(np.linalg.norm(np.diff(pred_world, axis=0), axis=-1).mean())
        pfk_spd = float(np.linalg.norm(np.diff(pred_world_fk, axis=0), axis=-1).mean())
        ratio = p_spd / max(g_spd, 1e-9)
        # T2M demo layout (per cross-project rule feedback_t2m_gif_layout):
        # static input skeleton + prompt + pred animation (pose AND rot6d-FK), NO GT.
        # Static skeleton = T-pose via FK from rest_offsets (purely topology,
        # not a frame of GT motion — keeps "input-only" semantics).
        static_pose = fk_rest_pose(rest_off, parents)
        prompt_text = gen_caption if (args.generic_prompt and gen_caption) else (item.get("caption") or "")
        skel_label = (
            f"{sp} skeleton (J={J})\n"
            f"T={T}  cfg={args.cond_scale} steps={args.n_ddim_steps}\n"
            f"speed_ratio={ratio:.3f}"
        )
        if args.large:
            make_t2m_large_gif(
                pred_world, pred_world_fk, static_pose, parents, prompt_text,
                str(actual_gif_path), fps=args.fps,
                gt=(gt_world if args.with_gt else None),
            )
        else:
            animate_t2m_input_pred(
                pred_world, static_pose, parents, str(gif_path),
                prompt_text=prompt_text, stride=args.stride, fps=args.fps,
                skeleton_label=skel_label,
                pred_fk=pred_world_fk,
            )
        line = (f"{sp} clip{k}: J={J} T={T} prompt={prompt_text[:60]!r} "
                f"GT_speed={g_spd:.4f} PRED_pose_speed={p_spd:.4f} "
                f"PRED_fk_speed={pfk_spd:.4f} ratio={ratio:.3f} "
                f"-> {actual_gif_path.name}")
        print(line)
        summary.append(line)
        picked[sp] += 1
        if all(picked[s] >= args.n_per for s in want):
            break

    missing = {s: args.n_per - picked[s] for s in want if picked[s] < args.n_per}
    (out_dir / "animate_summary.txt").write_text(
        "\n".join(summary) + f"\nmissing={missing}\n"
    )
    print(f"\nDONE {sum(picked.values())} gifs -> {out_dir}")
    print("PER-SPECIES picked:", picked)
    if missing:
        raise RuntimeError(
            f"animate_denoiser under-filled split '{args.split}': {missing}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
