"""Phase-2 animate_denoiser.py — render samples from a trained denoiser.

Pipeline per docs/phase2_diffusion_design.md §4-5:
  1. Load frozen VAE (use_text=False) and trained denoiser ckpt.
  2. For each requested batch:
     - vae.encode_skeleton_only(batch) → coarse_mask / pooled_adj / pooled_geo /
       pooled_skeleton_embeddings / anchor_indices / hard_assignment / assignment / s_j
     - frame_mask_lat = batch.frame_mask.view(B, T_lat, stride).all(-1)
     - z_T = N(0, I) of shape [B, T_lat, C, D]
     - DDIM sampling loop (default 50 steps) with CFG (cond_scale=7.5):
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

from scripts.animate import animate_clip, contact_sheet
from src.data.anytop_dataset import (
    AnyTopDataset, collate_fn as anytop_collate_fn,
    _recover_world_positions, _STD_FLOOR,
)
from src.models.graph_salad.batch import GraphMotionBatch
from src.models.graph_salad.vae import GraphMotionVAE
from src.models.graph_salad.denoiser import GraphSaladDenoiser
from scripts.train_denoiser import load_frozen_vae

from diffusers import DDIMScheduler


def load_denoiser(ckpt_path: str, dev: torch.device) -> tuple[GraphSaladDenoiser, dict]:
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    da = ck.get("args", {})
    vae_ta = ck.get("vae_ckpt_args", {})
    d_model = vae_ta.get("d_model", 384)
    n_heads = vae_ta.get("n_heads", 8)
    d_ff = da.get("d_ff") or 4 * d_model
    denoiser = GraphSaladDenoiser(
        d_model=d_model, n_heads=n_heads, d_ff=d_ff,
        n_layers=da.get("n_layers", 5),
        d_text=768, dropout=da.get("dropout", 0.1),
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
    text_emb = batch.caption_emb.to(dev)                # [B, 768]
    text2 = text_emb.repeat(2, 1)                       # [2B, 768]
    has_text_cond = batch.has_text.to(dev)              # [B] bool
    has_text_uncond = torch.zeros_like(has_text_cond, dtype=torch.bool)
    has_text2 = torch.cat([has_text_cond, has_text_uncond], dim=0)  # [2B]

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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vae_ckpt", required=True)
    ap.add_argument("--denoiser_ckpt", required=True)
    ap.add_argument("--caption_emb_cache", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--anytop_root", default=None)
    ap.add_argument("--split", default="val")
    ap.add_argument("--species", default="Alligator,Spider,Trex,Dragon",
                    help="comma-separated species to render")
    ap.add_argument("--n_per", type=int, default=2)
    ap.add_argument("--n_ddim_steps", type=int, default=50)
    ap.add_argument("--cond_scale", type=float, default=7.5)
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--fps", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda")
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
    ds_kwargs = dict(
        split=args.split,
        num_frames=ta.get("max_frames", 64),
        max_joints=ta.get("max_joints", 143),
        caption_emb_cache=cap_cache,
    )
    if anytop_root:
        ds_kwargs["data_root"] = anytop_root
    ds = AnyTopDataset(**ds_kwargs)

    # P2 (codex 2026-05-23): preflight caption coverage so we don't silently
    # render uncond samples for clips with missing T5 cache entries (the
    # AnyTopDataset zero-fills caption_emb + sets has_text=False on cache miss).
    n_missing = 0
    want_set = set(s.strip() for s in args.species.split(",") if s.strip())
    for i in range(len(ds)):
        it = ds[i]
        if it["object_type"] not in want_set:
            continue
        if not bool(it.get("has_text", False)):
            n_missing += 1
    if n_missing > 0:
        raise SystemExit(
            f"[animate preflight] {n_missing} requested-species clips have "
            f"has_text=False (missing from T5 cache {cap_cache}). Re-run "
            f"precompute_t5_captions.py to cover them, or filter --species."
        )
    print(f"  [preflight] all requested-species clips have valid T5 caption emb")

    want = [s.strip() for s in args.species.split(",") if s.strip()]
    picked = {s: 0 for s in want}
    summary: list[str] = []

    print(f"\nSampling: DDIM {args.n_ddim_steps} steps, CFG cond_scale={args.cond_scale}")
    for i in range(len(ds)):
        item = ds[i]
        sp = item["object_type"]
        if sp not in picked or picked[sp] >= args.n_per:
            continue
        raw = anytop_collate_fn([item])
        raw = {k: v.to(dev) if torch.is_tensor(v) else v for k, v in raw.items()}
        batch = GraphMotionBatch.from_collate_dict(raw)

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
        pred_world = _recover_world_positions(pred_raw)
        gt_world = batch.motion_features[0, :T, :J, :3].cpu().numpy()
        parents = [int(p) for p in item["parent_indices"][:J]]

        k = picked[sp]
        gif_path = out_dir / f"{sp}_clip{k}_denoiser_gtvspred.gif"
        g_spd = float(np.linalg.norm(np.diff(gt_world, axis=0), axis=-1).mean())
        p_spd = float(np.linalg.norm(np.diff(pred_world, axis=0), axis=-1).mean())
        ratio = p_spd / max(g_spd, 1e-9)
        ttl = (
            f"{sp} clip{k} [denoiser cfg={args.cond_scale} steps={args.n_ddim_steps}] "
            f"J={J} T={T}  speed_ratio={ratio:.3f}"
        )
        animate_clip(pred_world, gt_world, parents, str(gif_path),
                     ttl, args.stride, args.fps)
        for elev, azim, tag in [(12, -70, "obl"), (75, -90, "top")]:
            contact_sheet(pred_world, gt_world, parents,
                          str(out_dir / f"{sp}_clip{k}_sheet_{tag}.png"),
                          ttl, elev=elev, azim=azim)
        line = (f"{sp} clip{k}: J={J} T={T} GT_speed={g_spd:.4f} "
                f"PRED_speed={p_spd:.4f} ratio={ratio:.3f} -> {gif_path.name}")
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
