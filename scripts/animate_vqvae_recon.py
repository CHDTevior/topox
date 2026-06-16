"""Reconstruction QA for the Graph-VQVAE tokenizer (GraphVQTokenizer).

Read-only visual QA: load an L5 clip -> build the AnyTop batch -> run
GraphVQTokenizer.forward(batch) (encode -> pool -> RVQ -> decode) -> take
out["pred_motion"] [B,T,J,13] -> de-normalize -> recover world positions via the
OFFICIAL rot6d FK (recover_from_bvh_rot_np) -> animate GT-vs-RECON side-by-side.

This is the VQVAE analogue of scripts/animate_anytop13.py (which targets the
Gaussian GraphMotionVAE). The ONLY differences from that VAE recon path:
  * model is GraphVQTokenizer (src/models/vq_model/graph_vq_tokenizer.py),
    reconstructed from the ckpt's saved `args`;
  * forward is model(batch, allow_collectives=False) — rank-0-only eval, so the
    quantizer must NOT fire cross-rank EMA/dead-code collectives (mirrors
    train_graph_vqvae.py:803 val pass);
  * out["pred_motion"] is read directly (the tokenizer forward already returns it).

Everything downstream (de-norm with anytop_mean/std + _STD_FLOOR, rot6d FK via
recover_from_bvh_rot_np for BOTH GT and pred so the comparison is
apples-to-apples, animate_clip GT|PRED gif) is reused VERBATIM from the existing
anytop13 path — no hand-rolled FK (the project has been bitten twice by
double-root-rotation from verbatim-copying FK code).

Self-check (renderer faithfulness gate): for each clip we also recover the GT
through the position/RIC route (_recover_world_positions on the de-normed GT
13ch) and report the per-clip L2 between GT-FK (rot6d) and GT-RIC. With a correct
de-norm + recovery this is ~0 (both are the same GT in world space, recovered two
ways), proving the renderer is faithful BEFORE the recon GIFs are trusted.

Usage:
  python scripts/animate_vqvae_recon.py \
      --ckpt /tmp/vqvae_last_qa.pt \
      --species PZ_Grey_Seal_Female,PZ_Caracal_Male,... --n_per 1 \
      --out runs/vqvae_.../qa_recon
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

from scripts.animate import animate_clip, contact_sheet  # noqa: E402  model-agnostic renderers
from src.data.anytop_dataset import (  # noqa: E402
    AnyTopDataset, collate_fn, _recover_world_positions, _STD_FLOOR,
)
from src.data.anytop_rot6d_fk import recover_from_bvh_rot_np  # noqa: E402  official rot6d FK
from src.models.graph_salad.batch import GraphMotionBatch  # noqa: E402
from src.models.vq_model import GraphVQTokenizer  # noqa: E402


def load_vq_tokenizer(ckpt_path: str, dev: torch.device):
    """Reconstruct GraphVQTokenizer from a train_graph_vqvae.py ckpt.

    Uses the ckpt's saved `args` to rebuild the exact model (same constructor
    call as train_graph_vqvae.py:477), then loads weights strict=True. The
    codebook buffers (embed / cluster_size / embed_avg / initted) are part of the
    state_dict, so the loaded quantizer reproduces the trained codebooks exactly.
    """
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    ta = ck.get("args", {})
    if not ta:
        raise SystemExit(f"animate_vqvae_recon: ckpt {ckpt_path} missing 'args' key "
                         f"(not produced by train_graph_vqvae.py?)")
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--split", default="val")
    ap.add_argument("--species",
                    default="PZ_Grey_Seal_Female,PZ_Caracal_Male,"
                            "PZ_West_African_Lion_Male,PZ_Red_Kangaroo_Female,"
                            "PZ_Indian_Elephant_Male,PZ_Fossa_Male",
                    help="comma-separated object_types to render")
    ap.add_argument("--n_per", type=int, default=1)
    ap.add_argument("--stride", type=int, default=2, help="frame subsample for gif")
    ap.add_argument("--fps", type=int, default=8)
    ap.add_argument("--anytop_root", type=str, default=None,
                    help="AnyTop processed-data root; defaults to the ckpt's "
                         "training root if omitted")
    ap.add_argument("--amp_dtype", choices=["fp32", "bf16"], default=None,
                    help="autocast dtype for the forward; defaults to the ckpt's "
                         "training amp_dtype (the model was trained under bf16 "
                         "autocast, so bf16 reproduces the trained behavior)")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        print("  [INFO] CUDA unavailable, falling back to CPU")
        args.device = "cpu"
    dev = torch.device(args.device)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    model, ta, ck = load_vq_tokenizer(args.ckpt, dev)
    print(f"Loaded GraphVQTokenizer: d_model={ta['d_model']} max_coarse={ta['max_coarse']} "
          f"Q={ta['num_quantizers']} num_codes={ta['num_codes']} "
          f"temporal_stride={ta['temporal_stride']}")
    print(f"  ckpt epoch={ck.get('epoch')} global_step={ck.get('global_step')} "
          f"val_total={ck.get('val_total'):.4f} codebook_active={ck.get('codebook_active')}")

    # autocast dtype: reproduce the trained regime (bf16) unless overridden.
    amp_dtype = args.amp_dtype or ta.get("amp_dtype", "bf16")
    amp_enabled = (amp_dtype == "bf16") and dev.type == "cuda"
    print(f"  autocast: amp_dtype={amp_dtype} (enabled={amp_enabled})")

    # Dataset: reproduce the ckpt's TRAINING split exactly (val_frac/seed/num_frames/
    # max_joints from the ckpt args, captions off — the tokenizer takes no text).
    anytop_root = args.anytop_root or ta.get("anytop_root") or ck.get("data_root")
    ds = AnyTopDataset(
        split=args.split,
        num_frames=ta.get("max_frames", 64),
        max_joints=ta.get("max_joints", 64),
        val_frac=ta.get("val_frac", 0.05),
        seed=ta.get("seed", 42),
        load_captions=False,
        data_root=anytop_root,
    )
    print(f"  dataset: root={anytop_root} split={args.split} "
          f"num_frames={ta.get('max_frames', 64)} max_joints={ta.get('max_joints', 64)} "
          f"val_frac={ta.get('val_frac', 0.05)} -> {len(ds)} clips")

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

            # rank-0-only eval: allow_collectives=False so the quantizer does NOT
            # block on the cross-rank EMA/dead-code collectives (mirrors the
            # train loop's val pass, train_graph_vqvae.py:803).
            if amp_enabled:
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    out = model(batch, allow_collectives=False)
            else:
                out = model(batch, allow_collectives=False)

            J = int(item["num_joints"])
            # Stride-aware valid T: temporal_stride collapses every `stride` frames
            # in the pool, so a clip with T_var not divisible by stride has a
            # trailing tail the decoder returns as padded-zero. Clip to the model's
            # frame_mask_recovered (intersect with the clip's actual T) so the GIF
            # tail shows real recon, not stride-incomplete garbage.
            T_clip = int(item["num_frames"])
            T_valid = int(out["frame_mask_recovered"][0].sum().item())
            T = min(T_clip, T_valid)
            T_dropped = T_clip - T

            # De-normalize: raw = norm * (std + eps) + mean (AnyTop per-joint stats).
            std = raw["anytop_std"][0, :J].cpu().numpy()    # [J, 13]
            mean = raw["anytop_mean"][0, :J].cpu().numpy()  # [J, 13]
            pred_norm = out["pred_motion"][0, :T, :J, :].float().cpu().numpy()  # [T,J,13]
            pred_raw = pred_norm * (std[None] + _STD_FLOOR) + mean[None]
            # GT raw 13ch (same de-norm as pred). anytop_x is [J,13,T] -> [T,J,13].
            gt_norm = np.asarray(item["anytop_x"]).transpose(2, 0, 1)[:T, :J, :]
            gt_raw = gt_norm * (std[None] + _STD_FLOOR) + mean[None]

            parents = [int(p) for p in item["parent_indices"][:J]]
            offsets = np.asarray(item["rest_offsets"])[:J]

            # Official rot6d FK for BOTH GT and pred (apples-to-apples).
            pred_world = recover_from_bvh_rot_np(pred_raw, parents, offsets)  # [T,J,3]
            gt_world = recover_from_bvh_rot_np(gt_raw, parents, offsets)      # [T,J,3]

            # --- renderer faithfulness self-check: GT-FK (rot6d) vs GT-RIC (pos) ---
            # Two independent recoveries of the SAME GT into world space; a correct
            # de-norm + recovery makes these ~0. (Reported per clip; the aggregate is
            # the renderer-fidelity gate.)
            gt_ric = _recover_world_positions(gt_raw)  # [T,J,3] from position channels
            gt_selfcheck_l2 = float(
                np.linalg.norm(gt_world - gt_ric, axis=-1).mean())

            # --- recon metric: mean world-position L2 (GT-FK vs RECON-FK) ---
            recon_l2 = float(np.linalg.norm(pred_world - gt_world, axis=-1).mean())

            k = picked[sp]
            g_spd = float(np.linalg.norm(np.diff(gt_world, axis=0), axis=-1).mean())
            p_spd = float(np.linalg.norm(np.diff(pred_world, axis=0), axis=-1).mean())
            ratio = p_spd / max(g_spd, 1e-9)
            ttl = (f"{sp} clip{k} [VQVAE ep{ck.get('epoch')}] J={J} T={T}  "
                   f"recon_L2={recon_l2:.4f} speed_ratio={ratio:.3f}")
            gif = out_dir / f"{sp}_clip{k}_gtvsrecon.gif"
            # animate_clip(pred, gt, ...) -> side-by-side GT|RECON gif.
            animate_clip(pred_world, gt_world, parents, str(gif),
                         ttl, args.stride, args.fps)
            for elev, azim, tag in [(12, -70, "obl"), (75, -90, "top")]:
                contact_sheet(pred_world, gt_world, parents,
                              str(out_dir / f"{sp}_clip{k}_sheet_{tag}.png"),
                              ttl, elev=elev, azim=azim)

            line = (f"{sp} clip{k}: J={J} T={T} T_clip={T_clip} dropped_tail={T_dropped} "
                    f"recon_L2={recon_l2:.4f} GT_selfcheck_L2={gt_selfcheck_l2:.2e} "
                    f"GT_speed={g_spd:.4f} RECON_speed={p_spd:.4f} ratio={ratio:.3f} "
                    f"-> {gif.name}")
            print(line)
            summary.append(line)
            picked[sp] += 1
            if all(picked[s] >= args.n_per for s in want):
                break

    missing = {s: args.n_per - picked[s] for s in want if picked[s] < args.n_per}
    (out_dir / "recon_summary.txt").write_text(
        "\n".join(summary) + f"\nmissing={missing}\n")
    print(f"\nDONE {sum(picked.values())} gifs -> {out_dir}")
    print("PER-SPECIES picked:", picked)
    if missing:
        raise RuntimeError(
            f"recon QA under-filled in split '{args.split}': {missing} "
            f"(requested {args.n_per}/species)")


if __name__ == "__main__":
    main()
