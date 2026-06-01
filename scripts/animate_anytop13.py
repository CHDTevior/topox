"""Visual QA for the M1.7 AnyTop-native 13ch Graph-VAE (feat_mode=anytop13).

The anytop13 decoder outputs `pred_motion [B,T,J,13]` in AnyTop's NORMALIZED
RIFKE space — not directly plottable. This tool closes the loop:

  pred_motion (normalized 13ch)
    -> de-normalize  (× std + mean, AnyTop's per-joint stats)
    -> _recover_world_positions  (RIFKE -> Cartesian, AnyTop recovery)
    -> animate_clip  -> GT-vs-pred GIF

GT world positions are taken directly from `batch.motion_features[...,:3]` —
the dataset already recovered them from the raw clip, so GT needs no de-norm.

Per cross-project rule "可视化 demo 准确度 > metric": this is THE truth gate
for anytop13 — recon loss numbers must be cross-checked against visible motion
fidelity (frozen / jitter / collapse are invisible in metrics, visible here).

Usage:
  python scripts/animate_anytop13.py \\
      --ckpt runs/m1_7_anytop13_dynamic_seed42/best_recon_model.pt \\
      --species Alligator,Spider,Trex,Dragon --n_per 2 \\
      --out runs/m1_7_anytop13_dynamic_seed42/qa_animate
"""

import argparse
import sys
from pathlib import Path

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")

project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from scripts.animate import animate_clip, animate_clip_3col, contact_sheet  # noqa: E402  model-agnostic renderers
from src.data.anytop_dataset import (  # noqa: E402
    AnyTopDataset, collate_fn, _recover_world_positions, _STD_FLOOR,
)
from src.data.anytop_rot6d_fk import recover_from_bvh_rot_np  # noqa: E402  rot6d FK path
from src.models.graph_salad import GraphMotionBatch, GraphMotionVAE  # noqa: E402


def load_anytop13_vae(ckpt_path: str, dev: torch.device):
    """Reconstruct an anytop13 GraphMotionVAE from a train_graph_vae.py ckpt.

    Uses the ckpt's saved `args` to rebuild the exact model, then loads weights
    strict=True. attn_mode / use_text default safely for ckpts saved before
    those flags existed.
    """
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    ta = ck.get("args", {})
    if not ta:
        raise SystemExit(f"animate_anytop13: ckpt {ckpt_path} missing 'args' key")
    if ta.get("feat_mode") != "anytop13":
        raise SystemExit(
            f"animate_anytop13: ckpt feat_mode={ta.get('feat_mode')!r}, expected "
            f"'anytop13' — use scripts/animate_graph_vae.py for fk6 ckpts"
        )
    vae = GraphMotionVAE(
        pool_type=ta["pool_type"],
        pool_tau=ta.get("pool_tau"),
        d_model=ta["d_model"], n_heads=ta["n_heads"], d_ff=ta["d_ff"],
        n_graph_layers=ta["n_graph_layers"],
        n_enc_temporal_layers=ta["n_enc_temporal_layers"],
        n_cross_layers=ta["n_cross_layers"],
        n_dec_temporal_layers=ta["n_dec_temporal_layers"],
        n_treeik_layers=ta["n_treeik_layers"],
        max_coarse=ta["max_coarse"], local_radius=ta["local_radius"],
        temporal_stride=ta["temporal_stride"],
        temporal_kernel=ta["temporal_kernel"],
        dropout=ta["dropout"],
        feat_mode="anytop13",
        attn_mode=ta.get("attn_mode") or "scalar",
        use_text=bool(ta.get("use_text") or False),
        decoder_mode=ta.get("decoder_mode") or "unpool_identity",
        n_graph_temporal_layers=ta.get("n_graph_temporal_layers", 4),
    ).to(dev)
    vae.load_state_dict(ck["model_state_dict"], strict=True)
    vae.encoder.use_name_embed = bool(ta.get("use_name_embed", False))
    vae.eval()
    return vae, ta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--split", default="val")
    ap.add_argument("--species", default="Alligator,Spider,Trex,Dragon",
                    help="comma-separated object types to render")
    ap.add_argument("--n_per", type=int, default=2)
    ap.add_argument("--stride", type=int, default=2, help="frame subsample for gif")
    ap.add_argument("--fps", type=int, default=8)
    ap.add_argument("--caption_emb_cache", type=str, default=None,
                    help="T5 caption cache (only matters for use_text ckpts); "
                         "defaults to the ckpt's training cache if omitted")
    ap.add_argument("--anytop_root", type=str, default=None,
                    help="AnyTop processed-data root; defaults to the ckpt's "
                         "training root if omitted")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--render_mode", choices=("rot6d", "pos", "three_col"), default="rot6d",
                    help="rot6d (default): recover world via rotation channels "
                         "(3:9) + bone offsets + parent chain (official FK, bone "
                         "lengths exact). pos: legacy RIC path via position "
                         "channels (0:3). BOTH GT and pred use the same mode so "
                         "the comparison is apples-to-apples.")
    args = ap.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        print("  [INFO] CUDA unavailable, falling back to CPU")
        args.device = "cpu"
    dev = torch.device(args.device)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    vae, ta = load_anytop13_vae(args.ckpt, dev)
    print(f"Loaded anytop13 VAE: pool_type={ta['pool_type']} "
          f"attn_mode={ta.get('attn_mode') or 'scalar'} "
          f"use_text={bool(ta.get('use_text') or False)}")

    # Resolve caption cache + data root from the CLI, falling back to whatever
    # the ckpt was TRAINED with — so QA visuals reproduce the training-time
    # text condition / dataset rather than silently diverging.
    cap_cache = args.caption_emb_cache or ta.get("caption_emb_cache")
    if bool(ta.get("use_text") or False) and cap_cache is None:
        print("  [WARN] ckpt trained with use_text=True but no caption cache "
              "available — QA visuals run with has_text=False (text condition "
              "DISABLED); pass --caption_emb_cache to reproduce the conditioned output")
    anytop_root = args.anytop_root or ta.get("anytop_root")
    ds_kwargs = dict(
        split=args.split,
        num_frames=ta.get("max_frames", 64),
        max_joints=ta.get("max_joints", 143),
        caption_emb_cache=cap_cache,
        # Reproduce the ckpt's TRAINING val split. AnyTopDataset defaults
        # val_frac=0.2 seed=42, but these ckpts trained with val_frac=0.05 —
        # without this, QA renders a different (larger) val set that can include
        # clips the model trained on (leakage). Read from the ckpt's own args.
        val_frac=ta.get("val_frac", 0.2),
        seed=ta.get("seed", 42),
    )
    if anytop_root:
        ds_kwargs["data_root"] = anytop_root
    ds = AnyTopDataset(**ds_kwargs)
    want = [s.strip() for s in args.species.split(",") if s.strip()]
    picked = {s: 0 for s in want}
    summary = []

    with torch.no_grad():
        for i in range(len(ds)):
            item = ds[i]
            sp = item["object_type"]
            if sp not in picked or picked[sp] >= args.n_per:
                continue
            raw = collate_fn([item])
            raw = {k: v.to(dev) if torch.is_tensor(v) else v for k, v in raw.items()}
            batch = GraphMotionBatch.from_collate_dict(raw)
            out = vae(batch, sample=False)               # deterministic z=mu

            J = int(item["num_joints"])
            # Stride-aware T (2026-05-28, mirrors animate_denoiser.py codex P1
            # fix 2026-05-23): VAE temporal_stride=4 collapses every 4 frames
            # via .all() on encode, so a clip with T_var=67 has only
            # floor(67/4)*4=64 valid latent-recovered frames; the trailing 3
            # frames come back zero/garbage from decoder. Use frame_mask
            # _recovered to clip — otherwise GIF tail shows collapse/jitter
            # that LOOKS like recon failure but is just stride-incomplete tail.
            T_clip = int(item["num_frames"])
            T_valid = int(out["frame_mask_recovered"][0].sum().item())
            T = min(T_clip, T_valid)
            T_dropped = T_clip - T
            # De-normalize pred 13ch: raw = norm * (std + eps) + mean.
            # anytop_mean/std are de-norm stats — only needed for visualization,
            # not a model input, so they ride the raw collate dict (not the
            # typed GraphMotionBatch).
            std = raw["anytop_std"][0, :J].cpu().numpy()    # [J, 13]
            mean = raw["anytop_mean"][0, :J].cpu().numpy()  # [J, 13]
            pred_norm = out["pred_motion"][0, :T, :J, :].cpu().numpy()  # [T, J, 13]
            pred_raw = pred_norm * (std[None] + _STD_FLOOR) + mean[None]
            parents = [int(p) for p in item["parent_indices"][:J]]
            # Render mode: rot6d FK (default) or legacy pos/RIC. BOTH GT and pred
            # use the SAME recovery so the visual comparison is apples-to-apples.
            if args.render_mode == "rot6d":
                # GT raw 13ch (same de-norm as pred); offsets share the item's
                # new_to_old_perm ordering with anytop_x, so FK is aligned.
                # anytop_x is a [J,13,T] tensor -> numpy [T,J,13].
                gt_norm = np.asarray(item["anytop_x"]).transpose(2, 0, 1)[:T, :J, :]
                gt_raw = gt_norm * (std[None] + _STD_FLOOR) + mean[None]
                offsets = np.asarray(item["rest_offsets"])[:J]
                pred_world = recover_from_bvh_rot_np(pred_raw, parents, offsets)
                gt_world = recover_from_bvh_rot_np(gt_raw, parents, offsets)
            elif args.render_mode == "three_col":
                # 3-col QA: GT_RIC | PRED_RIC | PRED_FK. pred goes through BOTH the
                # RIC/position route AND the rot6d-FK route; their agreement is the
                # core rot6d_fk signal. speed_ratio reported on the RIC route.
                gt_norm = np.asarray(item["anytop_x"]).transpose(2, 0, 1)[:T, :J, :]
                gt_raw = gt_norm * (std[None] + _STD_FLOOR) + mean[None]
                offsets = np.asarray(item["rest_offsets"])[:J]
                gt_ric = _recover_world_positions(gt_raw)
                pred_ric = _recover_world_positions(pred_raw)
                pred_fk = recover_from_bvh_rot_np(pred_raw, parents, offsets)
                pred_world, gt_world = pred_ric, gt_ric
            else:  # pos (legacy RIC path)
                pred_world = _recover_world_positions(pred_raw)             # [T, J, 3]
                gt_world = batch.motion_features[0, :T, :J, :3].cpu().numpy()

            k = picked[sp]
            g_spd = float(np.linalg.norm(np.diff(gt_world, axis=0), axis=-1).mean())
            p_spd = float(np.linalg.norm(np.diff(pred_world, axis=0), axis=-1).mean())
            ratio = p_spd / max(g_spd, 1e-9)
            ttl = (f"{sp} clip{k} [anytop13/{ta['pool_type']}] J={J} T={T}  "
                   f"speed_ratio={ratio:.3f}")
            if args.render_mode == "three_col":
                gif = out_dir / f"{sp}_clip{k}_3col.gif"
                animate_clip_3col(gt_ric, pred_ric, pred_fk, parents, str(gif),
                                  ttl, args.stride, args.fps)
            else:
                gif = out_dir / f"{sp}_clip{k}_gtvspred.gif"
                animate_clip(pred_world, gt_world, parents, str(gif),
                             ttl, args.stride, args.fps)
                for elev, azim, tag in [(12, -70, "obl"), (75, -90, "top")]:
                    contact_sheet(pred_world, gt_world, parents,
                                  str(out_dir / f"{sp}_clip{k}_sheet_{tag}.png"),
                                  ttl, elev=elev, azim=azim)
            line = (f"{sp} clip{k}: J={J} T={T} effective_T={T} "
                    f"T_clip={T_clip} dropped_tail={T_dropped} "
                    f"GT_speed={g_spd:.4f} PRED_speed={p_spd:.4f} "
                    f"ratio={ratio:.3f} -> {gif.name}")
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
    # Fail loud on under-fill — a QA gate must not false-pass when requested
    # species are absent (matches scripts/animate_graph_vae.py behavior).
    if missing:
        raise RuntimeError(
            f"visual QA under-filled in split '{args.split}': {missing} "
            f"(requested {args.n_per}/species)"
        )


if __name__ == "__main__":
    main()
