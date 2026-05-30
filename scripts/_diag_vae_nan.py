"""Diagnose WHERE the VAE recon diverges for high-J failure species.

For each (ckpt, species) it runs the SAME forward path animate_anytop13 uses,
then probes every intermediate tensor for NaN/Inf and prints per-stage stats:

  z(mu) -> pred_motion(normalized) -> pred_raw(de-normalized 13ch)
        -> rot6d channels (3:9) -> recovered world positions

This separates three hypotheses:
  (A) VAE latent/decoder genuinely outputs NaN  -> bad model
  (B) pred_motion finite but de-norm blows up    -> std/mean issue
  (C) pred_motion+raw finite but recover_world    -> 6D->matrix / cumsum amplifies
      explodes (rot6d near-degenerate -> inf)        a small rot error to inf

Read-only. No training touched. Usage:
  python scripts/_diag_vae_nan.py --ckpt <path> --species PZ_Grey_Seal_Female,...
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.animate_anytop13 import load_anytop13_vae  # reuse exact loader
from src.data.anytop_dataset import (
    AnyTopDataset, collate_fn, _recover_world_positions, _STD_FLOOR,
    _rotation_6d_to_matrix_np,
)
from src.models.graph_salad import GraphMotionBatch


def stat(name, arr):
    a = np.asarray(arr, dtype=np.float64)
    n_nan = int(np.isnan(a).sum())
    n_inf = int(np.isinf(a).sum())
    finite = a[np.isfinite(a)]
    amin = float(finite.min()) if finite.size else float("nan")
    amax = float(finite.max()) if finite.size else float("nan")
    aabs = float(np.abs(finite).max()) if finite.size else float("nan")
    flag = "  <<< NAN/INF" if (n_nan or n_inf) else ""
    print(f"    {name:28s} shape={str(a.shape):18s} "
          f"nan={n_nan} inf={n_inf} absmax={aabs:.4g} "
          f"[{amin:.4g},{amax:.4g}]{flag}")
    return n_nan, n_inf


def first_bad_frame(arr_TJC):
    """Return first frame index t where any value is nan/inf, else -1."""
    bad = ~np.isfinite(arr_TJC.reshape(arr_TJC.shape[0], -1)).all(axis=1)
    idx = np.where(bad)[0]
    return int(idx[0]) if idx.size else -1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--species", required=True)
    ap.add_argument("--anytop_root", default=None)
    args = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    vae, ta = load_anytop13_vae(args.ckpt, dev)
    root = args.anytop_root or ta.get("anytop_root")
    ds = AnyTopDataset(split="val", num_frames=ta.get("max_frames", 64),
                       max_joints=ta.get("max_joints", 143),
                       caption_emb_cache=None,
                       **({"data_root": root} if root else {}))
    want = [s.strip() for s in args.species.split(",") if s.strip()]
    print(f"\n#### CKPT {Path(args.ckpt).name}  (ep-tag from filename) ####")

    seen = {s: False for s in want}
    with torch.no_grad():
        for i in range(len(ds)):
            item = ds[i]
            sp = item["object_type"]
            if sp not in seen or seen[sp]:
                continue
            seen[sp] = True
            raw = collate_fn([item])
            raw = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in raw.items()}
            batch = GraphMotionBatch.from_collate_dict(raw)
            out = vae(batch, sample=False)

            J = int(item["num_joints"])
            T_valid = int(out["frame_mask_recovered"][0].sum().item())
            T = min(int(item["num_frames"]), T_valid)
            print(f"\n  == {sp}  J={J} T={T} ==")

            # Stage 0: latent mu
            mu = vae.encode(batch)["mu"][0].cpu().numpy()
            stat("z/mu (latent)", mu)
            # Stage 1: pred_motion (normalized 13ch)
            pm = out["pred_motion"][0, :T, :J, :].cpu().numpy()
            stat("pred_motion(norm 13ch)", pm)
            # Stage 2: de-normalize
            std = raw["anytop_std"][0, :J].cpu().numpy()
            mean = raw["anytop_mean"][0, :J].cpu().numpy()
            stat("anytop_std", std)
            pred_raw = pm * (std[None] + _STD_FLOOR) + mean[None]
            stat("pred_raw(de-norm 13ch)", pred_raw)
            # Stage 3: rot6d channels (3:9) of ROOT joint — feeds 6D->matrix
            rot6d_root = pred_raw[:, 0, 3:9]            # [T,6]
            stat("rot6d_root (ch3:9)", rot6d_root)
            # 6D->matrix on root, check determinant / norms
            try:
                Rm = _rotation_6d_to_matrix_np(rot6d_root)   # [T,3,3]
                stat("root_rot_matrix", Rm)
                dets = np.linalg.det(Rm)
                stat("root_rot_det", dets)
            except Exception as e:
                print(f"    6D->matrix RAISED: {e!r}")
            # Stage 4: recovered world positions (the cumsum/inv-rot stage)
            try:
                world = _recover_world_positions(pred_raw)   # [T,J,3]
                nn, ni = stat("recovered_world_pos", world)
                if nn or ni:
                    fb = first_bad_frame(world)
                    print(f"    >>> FIRST BAD FRAME in world pos: t={fb} / T={T}")
                    # show that frame's root rot6d to see if it's the trigger
                    if 0 <= fb < T:
                        print(f"    >>> rot6d_root[t={fb}] = {rot6d_root[fb]}")
            except Exception as e:
                print(f"    recover_world RAISED: {e!r}")
            # GT for reference
            gt = batch.motion_features[0, :T, :J, :3].cpu().numpy()
            stat("GT_world_pos (ref)", gt)

            if all(seen.values()):
                break
    miss = [s for s in want if not seen[s]]
    if miss:
        print(f"\n  [WARN] species not found in val: {miss}")
    print("\n#### DONE ####")


if __name__ == "__main__":
    main()
