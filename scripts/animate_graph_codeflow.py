#!/usr/bin/env python3
"""Text-to-motion visual QA for Graph-CodeFlow (level_a / graph_pscf) — handoff/20260609_graph_
codeflow_rvq_backbone_plan.md §8 + LOCKED recipe (single-gif T2M layout).

Pipeline (inference, frozen tokenizer + trained flow):
  target skeleton + prompt
    -> tokenizer.prepare_skeleton_only(batch, T_lat)   (motion-independent graph)
    -> GraphCodeFlow.sample(cond, token_mask, ...)      (ODE + CFG -> z_hat)
    -> tokenizer.nearest_residual_ids(z_hat)            (residual snap -> indices)
    -> tokenizer.decode_from_indices(indices, meta, batch)  (frozen decoder)
    -> anytop13 motion [1,T,J,13]
    -> de-norm + rot6d-FK recovery -> single-gif T2M (static skeleton + prompt +
       pred; NO GT column — T2M inference takes only skeleton + prompt).

Read-only QA. Renders both the snapped-decode motion and (optionally) the
continuous-decode motion for the continuous-vs-snapped comparison (the key gate).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.animate import animate_t2m_input_pred, fk_rest_pose  # noqa: E402
from src.data.anytop_dataset import (  # noqa: E402
    AnyTopDataset, collate_fn as anytop_collate_fn, _STD_FLOOR,
)
from src.data.anytop_rot6d_fk import recover_from_bvh_rot_np  # noqa: E402
from src.models.graph_salad.batch import GraphMotionBatch  # noqa: E402
from src.models.vq_model import GraphVQTokenizer  # noqa: E402
from src.models.CodeFlow_Model import GraphCodeFlow  # noqa: E402


def load_frozen_tokenizer(ckpt_path: str, dev: torch.device):
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
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
    return model, ta


def load_flow(ckpt_path: str, code_dim: int, dev: torch.device):
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    a = ck["args"]
    flow = GraphCodeFlow(
        code_dim=a.get("code_dim", code_dim), n_heads=a.get("n_heads", 8),
        d_ff=a.get("d_ff", 2048), n_layers=a.get("n_layers", 5),
        d_text=768, text_token_dim=768, dropout=a.get("dropout", 0.1),
        # model_variant + graph_pscf arch args (old ckpts: level_a defaults so they
        # still rebuild; graph_pscf ckpts carry depth_double/depth_single/mlp_ratio).
        model_variant=a.get("model_variant", "level_a"),
        depth_double=a.get("depth_double", 6), depth_single=a.get("depth_single", 12),
        max_T_lat=a.get("max_T_lat", 75), mlp_ratio=a.get("mlp_ratio", 4.0),
    ).to(dev)
    flow.load_state_dict(ck["model_state_dict"], strict=True)
    flow.eval(); flow.requires_grad_(False)
    return flow, ck


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--flow_ckpt", required=True)
    ap.add_argument("--frozen_vqvae_ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--split", default="val")
    ap.add_argument("--species", default="PZ_Grey_Seal_Female,PZ_Caracal_Male,"
                    "PZ_West_African_Lion_Male,PZ_Red_Kangaroo_Female")
    ap.add_argument("--n_per", type=int, default=1)
    ap.add_argument("--cfg_scale", type=float, default=4.0,
                    help="CFG scale (SWEEP starting point — not hardcoded 6.0)")
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--T_lat", type=int, default=None,
                    help="latent frames to generate; defaults to num_frames/stride")
    ap.add_argument("--num_frames", type=int, default=None,
                    help="override the dataset num_frames used for rendering; None = "
                         "use ckpt max_frames. Set 300 to QA full-length clips")
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--fps", type=int, default=8)
    ap.add_argument("--anytop_root", type=str, default=None)
    ap.add_argument("--caption_emb_cache", type=str,
                    default="data/anytop_caption_t5_cleanL5_multi.npz")
    ap.add_argument("--caption_token_cache", type=str,
                    default="data/anytop_caption_t5_cleanL5_multi")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"
    dev = torch.device(args.device)
    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)

    tokenizer, ta = load_frozen_tokenizer(args.frozen_vqvae_ckpt, dev)
    flow, fck = load_flow(args.flow_ckpt, ta["d_model"], dev)
    stride = ta["temporal_stride"]
    # Render frame length: override the ckpt's max_frames when --num_frames is set
    # (full-length QA uses 300). T_lat defaults to num_frames/stride.
    num_frames = args.num_frames if args.num_frames is not None else ta.get("max_frames", 64)
    # graph_pscf trains at FULL length (T_lat up to max_T_lat=75); rendering at the
    # tokenizer's 64-frame default would mismatch the trained regime -> require an
    # explicit --num_frames (e.g. 300) instead of silently defaulting to 64.
    if getattr(flow, "model_variant", "level_a") == "graph_pscf" and args.num_frames is None:
        raise SystemExit("[QA FAIL] graph_pscf QA needs an explicit --num_frames "
                         "(full-length, e.g. 300); refusing the 64-frame default.")
    T_lat = args.T_lat or (num_frames // stride)
    T_full = T_lat * stride
    print(f"tokenizer code_dim={ta['d_model']} Q={ta['num_quantizers']} stride={stride}; "
          f"flow epoch={fck.get('epoch')} val_flow={fck.get('val_flow')}; "
          f"T_lat={T_lat} T_full={T_full} cfg={args.cfg_scale} steps={args.steps}")

    anytop_root = args.anytop_root or ta.get("anytop_root")
    ds = AnyTopDataset(
        split=args.split, num_frames=num_frames, max_joints=ta.get("max_joints", 64),
        val_frac=ta.get("val_frac", 0.05), seed=ta.get("seed", 42),
        data_root=anytop_root, load_captions=True,
        caption_emb_cache=args.caption_emb_cache,
        caption_token_cache=args.caption_token_cache,
        return_caption_tokens=True, random_caption=False)

    want = [s.strip() for s in args.species.split(",") if s.strip()]
    picked = {s: 0 for s in want}
    summary = []
    with torch.no_grad():
        for i in range(len(ds)):
            item = ds[i]
            sp = item["object_type"]
            if sp not in picked or picked[sp] >= args.n_per:
                continue
            raw = anytop_collate_fn([item])
            raw = {k: v.to(dev) if torch.is_tensor(v) else v for k, v in raw.items()}
            batch = GraphMotionBatch.from_collate_dict(raw)

            # Motion-independent graph metadata for the target skeleton.
            meta = tokenizer.prepare_skeleton_only(batch, T_lat)
            cond = {
                "text_global": batch.caption_emb.float(),
                "text_tokens": batch.caption_token_emb.float(),
                "text_token_mask": batch.caption_token_mask,
                "has_text": batch.has_text,
                "pooled_adjacency": meta["pooled_adjacency"].float(),
                "pooled_geodesic": meta["pooled_geodesic"].float(),
                "pooled_skeleton_embeddings": meta["pooled_skeleton_embeddings"].float(),
                "coarse_mask": meta["coarse_mask"],
                "frame_mask_lat": meta["frame_mask_lat"],
            }
            B, C = meta["coarse_mask"].shape
            # ODE + CFG sample -> continuous z_hat -> residual snap -> decode.
            z_hat = flow.sample(cond, meta["token_mask"], T_lat, C,
                                steps=args.steps, cfg_scale=args.cfg_scale,
                                validate_inputs=True)
            proj = tokenizer.nearest_residual_ids(z_hat, meta["token_mask"])
            indices_hat = proj["indices_hat"]
            fake_batch = type("B", (), {"joint_mask": batch.joint_mask})()
            snap = tokenizer.decode_from_indices(indices_hat, meta, fake_batch)["pred_motion"]
            cont = tokenizer.decode(z_hat, meta, fake_batch)["pred_motion"]

            J = int(item["num_joints"])
            std = raw["anytop_std"][0, :J].cpu().numpy()
            mean = raw["anytop_mean"][0, :J].cpu().numpy()
            parents = [int(p) for p in item["parent_indices"][:J]]
            offsets = np.asarray(item["rest_offsets"])[:J]

            def to_world(pred_motion):
                pn = pred_motion[0, :T_full, :J, :].float().cpu().numpy()
                pr = pn * (std[None] + _STD_FLOOR) + mean[None]
                return recover_from_bvh_rot_np(pr, parents, offsets)  # [T,J,3]

            snap_world = to_world(snap)
            cont_world = to_world(cont)
            static_pose = fk_rest_pose(offsets, parents)
            prompt_text = item.get("caption") or ""
            p_spd = float(np.linalg.norm(np.diff(snap_world, axis=0), axis=-1).mean())

            k = picked[sp]
            skel_label = (f"{sp} skeleton (J={J})\nT={T_full} cfg={args.cfg_scale} "
                          f"steps={args.steps}\nproj_err={proj['projection_error'].item():.3f} "
                          f"snap_speed={p_spd:.3f}")
            # Single-gif T2M: static input skeleton + prompt + SNAPPED pred (the
            # main generation path), with continuous decode as the 3rd "FK" panel
            # slot repurposed to show continuous-vs-snapped (diagnostic upper bound).
            gif = out_dir / f"{sp}_clip{k}_t2m.gif"
            animate_t2m_input_pred(
                snap_world, static_pose, parents, str(gif),
                prompt_text=prompt_text, stride=args.stride, fps=args.fps,
                skeleton_label=skel_label,
                pred_fk=cont_world, pred_label="snapped decode",
                pred_fk_label="continuous decode")
            line = (f"{sp} clip{k}: J={J} T={T_full} prompt={prompt_text[:50]!r} "
                    f"proj_err={proj['projection_error'].item():.4f} "
                    f"snap_speed={p_spd:.4f} cont_vs_snap_maxabs="
                    f"{(cont - snap).abs().max().item():.4f} -> {gif.name}")
            print(line)
            summary.append(line)
            picked[sp] += 1
            if all(picked[s] >= args.n_per for s in want):
                break

    (out_dir / "t2m_summary.txt").write_text("\n".join(summary) + "\n")
    print(f"\nDONE {sum(picked.values())} gifs -> {out_dir}")
    print("PER-SPECIES picked:", picked)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
