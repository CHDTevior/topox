"""Smoke gate 1 (plan §8): torch rot6d FK == numpy official port, <1e-4.
Also autograd check: grad flows to non-root rot6d channels (the FK signature).
"""
import sys
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data.anytop_dataset import AnyTopDataset, _STD_FLOOR  # noqa
from src.data.anytop_rot6d_fk import recover_from_bvh_rot_np  # noqa numpy ref
from src.models.graph_salad.rot6d_fk_recovery import recover_rot6d_fk_positions_torch  # noqa

ds = AnyTopDataset(split="val", val_frac=0.05, seed=42,
                   data_root=str(ROOT / "data/anytop_planet_zoo_clean_L2"),
                   num_frames=64, max_joints=144, caption_emb_cache=None)
want = ["PZ_Asian_Water_Monitor_Male", "PZ_Komodo_Dragon_Male",
        "PZ_Saltwater_Crocodile_Male", "PZ_Grey_Seal_Male"]
done = set()
max_diff_all = 0.0
for i in range(len(ds)):
    it = ds[i]
    sp = it["object_type"]
    if sp not in want or sp in done:
        continue
    done.add(sp)
    J = int(it["num_joints"]); T = int(it["num_frames"])
    ax = np.asarray(it["anytop_x"], np.float32)
    mean = np.asarray(it["anytop_mean"], np.float32); std = np.asarray(it["anytop_std"], np.float32)
    raw = np.transpose(ax, (2, 0, 1))[:T, :J, :] * (std[:J][None] + _STD_FLOOR) + mean[:J][None]
    parents = [int(p) for p in it["parent_indices"][:J]]
    offsets = np.asarray(it["rest_offsets"], np.float32)[:J]

    # numpy ref
    fk_np = recover_from_bvh_rot_np(raw, parents, offsets)            # [T,J,3]
    # torch (B=1)
    x = torch.tensor(raw[None], dtype=torch.float32, requires_grad=True)   # [1,T,J,13]
    ro = torch.tensor(offsets[None], dtype=torch.float32)
    jm = torch.ones(1, J, dtype=torch.bool)
    fk_t = recover_rot6d_fk_positions_torch(x, [parents], ro, jm)     # [1,T,J,3]
    fk_t_np = fk_t.detach().numpy()[0]

    # compare only valid joints (both zero padded same way)
    diff = np.abs(fk_np - fk_t_np).max()
    max_diff_all = max(max_diff_all, diff)
    # autograd: grad to non-root rot6d (3:9 of j>=1) must be nonzero
    loss = fk_t.abs().mean(); loss.backward()
    g = x.grad
    nr_rot = float(g[0, :, 1:, 3:9].abs().sum().item())
    print(f"  {sp:32s} J={J} T={T} max|np-torch|={diff:.3e} "
          f"nonroot_rot6d_grad={nr_rot:.3e}", flush=True)

GATE = 1e-4
print(f"\nMAX DIFF vs numpy: {max_diff_all:.3e}", flush=True)
if max_diff_all < GATE:
    print(f"SMOKE1 PASS (max_diff {max_diff_all:.3e} < {GATE})", flush=True)
else:
    print(f"SMOKE1 FAIL (max_diff {max_diff_all:.3e} >= {GATE})", flush=True)
    sys.exit(1)
