"""GT-only sanity: render the SAME GT skeleton TWO ways and compare.

  RED  (ref) = _recover_world_positions — the CURRENT renderer, which uses
               channel 0:3 (root-relative POSITION) for non-root joints.
  BLUE (fk)  = rot6d FK — item['local_rotations_6d'] (raw local 6D rotation) +
               rest_offsets + parent chain, via treeik_decoder.fk_one.

Purpose (user asked "can we render from rot6d?"): show whether the rot6d
channels carry faithful skeleton geometry — i.e. whether rot6d FK reproduces the
same skeleton the 0:3-position path renders. Pure data-representation check, NO
model. If BLUE ~= RED -> rot6d is usable for rendering. If they differ -> rot6d
and 0:3-position are NOT redundant in this AnyTop representation.

Run (rose11, CPU ok): python scripts/_render_rot6d_fk_sanity.py
"""
import sys
import importlib.util
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.anytop_dataset import AnyTopDataset, _recover_world_positions  # noqa: E402
from src.models.treeik_decoder import fk_one  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "aa13", str(ROOT / "scripts" / "animate_anytop13.py"))
aa = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(aa)

ANYTOP_ROOT = str(ROOT / "data" / "anytop_planet_zoo_clean_L2")
OUT = ROOT / "runs" / "_rot6d_fk_sanity"
OUT.mkdir(parents=True, exist_ok=True)

ds = AnyTopDataset(split="val", val_frac=0.05, seed=42, data_root=ANYTOP_ROOT,
                   num_frames=64, max_joints=144, caption_emb_cache=None)
want = ["PZ_Asian_Water_Monitor_Male", "PZ_Komodo_Dragon_Male",
        "PZ_Saltwater_Crocodile_Male", "PZ_Grey_Seal_Male"]
done = set()
for i in range(len(ds)):
    it = ds[i]
    sp = it["object_type"]
    if sp not in want or sp in done:
        continue
    done.add(sp)
    J = int(it["num_joints"])
    T = int(it["num_frames"])
    # RED ref: current renderer path (0:3 position recovery), from raw 13ch
    ax = np.asarray(it["anytop_x"], np.float32)            # [Jm,13,Tm]
    mean = np.asarray(it["anytop_mean"], np.float32)
    std = np.asarray(it["anytop_std"], np.float32)
    raw = np.transpose(ax, (2, 0, 1))[:T, :J, :] * (std[:J][None] + 1e-6) + mean[:J][None]
    world03 = _recover_world_positions(raw)                # [T,J,3]

    # BLUE fk: rot6d FK from the item's raw local_rotations_6d field [T,Jm,6]
    lr6 = np.asarray(it["local_rotations_6d"], np.float32)[:T, :J, :]  # [T,J,6]
    rot6d = torch.tensor(lr6, dtype=torch.float32)
    root_local = torch.tensor(world03[:, 0, :], dtype=torch.float32)   # shared root xz/y
    rest = torch.tensor(np.asarray(it["rest_offsets"], np.float32)[:J], dtype=torch.float32)
    parents = [int(p) for p in it["parent_indices"][:J]]
    fk = fk_one(rot6d, root_local, parents, rest).numpy()  # [T,J,3]

    nr = slice(1, J)
    d = np.abs(world03[:, nr] - fk[:, nr])
    scale = float(np.abs(world03[:, nr]).mean())
    print(f"{sp} J={J} T={T} nonroot|0:3-vs-FK| mean={d.mean():.4f} max={d.max():.4f} "
          f"(coord_scale={scale:.4f} rel={d.mean()/max(scale,1e-9):.2f})", flush=True)
    ttl = f"{sp} RED=0:3pos BLUE=rot6dFK nr={d.mean():.3f}"
    aa.contact_sheet(fk, world03, parents, str(OUT / f"{sp}_sheet_obl.png"), ttl, elev=12, azim=-70)
    aa.contact_sheet(fk, world03, parents, str(OUT / f"{sp}_sheet_top.png"), ttl, elev=75, azim=-90)
    aa.animate_clip(fk, world03, parents, str(OUT / f"{sp}_gtvs.gif"), ttl, 2, 12)

print("DONE")
