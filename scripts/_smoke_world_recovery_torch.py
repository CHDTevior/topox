"""Smoke gate 1: torch differentiable world recovery == numpy/scipy version.

Loads a few REAL clips from the cleaned dataset, runs both:
  - numpy/scipy  src.data.anytop_dataset._recover_world_positions  (per [T,J,13])
  - torch        src.models.graph_salad.world_recovery.recover_world_positions_torch ([B,T,J,13])
and asserts max abs diff < 1e-4. Also checks autograd flows (grad finite).

Run: python scripts/_smoke_world_recovery_torch.py
"""
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.anytop_dataset import (  # noqa: E402
    AnyTopDataset, _recover_world_positions,
)
from src.models.graph_salad.world_recovery import (  # noqa: E402
    recover_world_positions_torch,
)

ANYTOP_ROOT = str(ROOT / "data" / "anytop_planet_zoo_clean_L2")


def main():
    # val split, a few long-chain clips (raw un-normalized 13ch needed)
    ds = AnyTopDataset(
        split="val", val_frac=0.05, seed=42, data_root=ANYTOP_ROOT,
        num_frames=64, max_joints=144, caption_emb_cache=None,
    )
    # find a few clips across species
    want = ["PZ_Asian_Water_Monitor_Male", "PZ_Komodo_Dragon_Male",
            "PZ_Saltwater_Crocodile_Male", "PZ_Grey_Seal_Male"]
    picked = []
    for i in range(len(ds)):
        it = ds[i]
        if it["object_type"] in want:
            picked.append((it["object_type"], it))
            want.remove(it["object_type"])
        if not want:
            break
    if not picked:
        # fallback: first 3 clips whatever species
        for i in range(min(3, len(ds))):
            it = ds[i]
            picked.append((it["object_type"], it))

    print(f"testing {len(picked)} clips", flush=True)
    max_diff_all = 0.0
    for sp, it in picked:
        # it["anytop_x"] is [J,13,T] NORMALIZED. De-normalize to raw 13ch [T,J,13].
        ax = it["anytop_x"]                       # [J,13,T] (np or tensor)
        ax = np.asarray(ax, dtype=np.float32)
        mean = np.asarray(it["anytop_mean"], dtype=np.float32)  # [J,13]
        std = np.asarray(it["anytop_std"], dtype=np.float32)    # [J,13]
        J = int(it["num_joints"])
        T = int(it["num_frames"])
        # de-normalize: raw = norm * std + mean ; anytop_x is [J,13,T] -> [T,J,13]
        norm_tjc = np.transpose(ax, (2, 0, 1))[:T, :J, :]       # [T,J,13]
        mean_jc = mean[:J]                                       # [J,13]
        std_jc = std[:J]
        # match dataset normalize EXACTLY: normed = (raw-mean)/(std+_STD_FLOOR),
        # so invert with the SAME 1e-6 floor (anytop_dataset._STD_FLOOR). Without
        # the floor this local denorm drifts vs the dataset's stored normed view.
        raw_tjc = norm_tjc * (std_jc[None] + 1e-6) + mean_jc[None]  # [T,J,13]

        # numpy/scipy reference
        world_np = _recover_world_positions(raw_tjc.astype(np.float32))  # [T,J,3]

        # torch differentiable
        x = torch.tensor(raw_tjc[None], dtype=torch.float32, requires_grad=True)  # [1,T,J,13]
        world_t = recover_world_positions_torch(x)              # [1,T,J,3]
        world_t_np = world_t.detach().numpy()[0]                # [T,J,3]

        diff = np.abs(world_np - world_t_np).max()
        max_diff_all = max(max_diff_all, diff)
        # autograd check
        loss = world_t.abs().mean()
        loss.backward()
        g = x.grad
        grad_finite = bool(torch.isfinite(g).all().item())
        grad_nonzero = float(g.abs().sum().item())
        print(f"  {sp:35s} J={J} T={T}  max|np-torch|={diff:.3e}  "
              f"grad_finite={grad_finite} grad_abs_sum={grad_nonzero:.3e}", flush=True)

    print(f"\nMAX DIFF across clips: {max_diff_all:.3e}", flush=True)
    GATE = 1e-4
    if max_diff_all < GATE:
        print(f"GATE1 PASS  (max_diff {max_diff_all:.3e} < {GATE})", flush=True)
    else:
        print(f"GATE1 FAIL  (max_diff {max_diff_all:.3e} >= {GATE})", flush=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
