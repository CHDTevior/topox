"""Re-render rot6d FK using the EXPLICIT per-frame real root translation+rotation
(it['root_position'] direct field) + non-root 6D FK, per user request.

Renders, for each long-chain species, GT(red, _recover_world_positions / 0:3
path) vs rot6d-FK(blue), where blue's root world position = it['root_position']
(direct per-frame field, NOT integrated) and root orientation = root 6D rot.

Also prints a diagnostic comparing TWO root-rotation conventions for the
non-root chain, because _recover_world_positions rotates channel-0:3 by the
INVERSE root rotation (Rr^-1) while fk_one chains the FORWARD root rotation (Rr):
  fwd  : standard fk_one (grot[0]=Rr, chain forward)
  inv  : fk_one but with root R replaced by Rr^T (so chain uses inverse root)
Whichever has smaller |GT - FK| is the convention AnyTop's 0:3 path uses.

Pure data, no model. Run on rose11: python scripts/_render_rot6d_fk_v2_realroot.py
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
from src.data.anytop_dataset import AnyTopDataset, _recover_world_positions  # noqa
from src.models.treeik_decoder import fk_one, rot6d_to_matrix  # noqa

_spec = importlib.util.spec_from_file_location(
    "aa13", str(ROOT / "scripts" / "animate_anytop13.py"))
aa = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(aa)

OUT = ROOT / "runs" / "_rot6d_fk_v2_realroot"
OUT.mkdir(parents=True, exist_ok=True)
ds = AnyTopDataset(split="val", val_frac=0.05, seed=42,
                   data_root=str(ROOT / "data/anytop_planet_zoo_clean_L2"),
                   num_frames=64, max_joints=144, caption_emb_cache=None)
want = ["PZ_Asian_Water_Monitor_Male", "PZ_Komodo_Dragon_Male",
        "PZ_Saltwater_Crocodile_Male", "PZ_Grey_Seal_Male"]
done = set()


def fk_with_root_R(rot6d_t, root_world_t, parents_l, rest_t, root_R):
    """fk_one but root orientation forced to root_R [T,3,3] (forward or inverse)."""
    Tn, Jn, _ = rot6d_t.shape
    R = rot6d_to_matrix(rot6d_t)            # [T,J,3,3]
    R = R.clone(); R[:, 0] = root_R         # override root orientation
    gpos = [None] * Jn; grot = [None] * Jn
    for j in range(Jn):
        p = parents_l[j]
        off = rest_t[j].view(1, 3, 1)
        if p < 0:
            grot[j] = R[:, j]; gpos[j] = root_world_t
        else:
            grot[j] = grot[p] @ R[:, j]
            gpos[j] = gpos[p] + (grot[p] @ off).squeeze(-1)
    return torch.stack(gpos, dim=1).numpy()


for i in range(len(ds)):
    it = ds[i]
    sp = it["object_type"]
    if sp not in want or sp in done:
        continue
    done.add(sp)
    J = int(it["num_joints"]); T = int(it["num_frames"])
    ax = np.asarray(it["anytop_x"], np.float32)
    mean = np.asarray(it["anytop_mean"], np.float32); std = np.asarray(it["anytop_std"], np.float32)
    raw = np.transpose(ax, (2, 0, 1))[:T, :J, :] * (std[:J][None] + 1e-6) + mean[:J][None]
    world03 = _recover_world_positions(raw)            # GT red (0:3 path)

    # EXPLICIT real per-frame root translation (direct field)
    root_world = np.asarray(it["root_position"], np.float32)[:T]   # [T,3]
    assert np.allclose(root_world, world03[:, 0], atol=1e-3), "root_position != recovered root"
    root_world_t = torch.tensor(root_world)

    lr6 = np.asarray(it["local_rotations_6d"], np.float32)[:T, :J, :]
    rot6d = torch.tensor(lr6)
    rest = torch.tensor(np.asarray(it["rest_offsets"], np.float32)[:J])
    parents = [int(p) for p in it["parent_indices"][:J]]
    Rr = rot6d_to_matrix(torch.tensor(raw[:, 0, 3:9]))            # root rot from ch3:9 [T,3,3]
    Rr_inv = Rr.transpose(-1, -2)

    fk_fwd = fk_with_root_R(rot6d, root_world_t, parents, rest, Rr)
    fk_inv = fk_with_root_R(rot6d, root_world_t, parents, rest, Rr_inv)

    nr = slice(1, J)
    e_fwd = np.abs(world03[:, nr] - fk_fwd[:, nr]).mean()
    e_inv = np.abs(world03[:, nr] - fk_inv[:, nr]).mean()
    scale = float(np.abs(world03[:, nr]).mean())
    best, fk = ("inv", fk_inv) if e_inv < e_fwd else ("fwd", fk_fwd)
    print(f"{sp} J={J} T={T} rootdisp={np.linalg.norm(root_world[-1]-root_world[0]):.3f} "
          f"fwd_err={e_fwd:.4f} inv_err={e_inv:.4f} best={best} "
          f"rel={min(e_fwd,e_inv)/max(scale,1e-9):.2f}", flush=True)
    ttl = f"{sp} RED=0:3pos BLUE=rot6dFK({best}root) err={min(e_fwd,e_inv):.3f}"
    aa.contact_sheet(fk, world03, parents, str(OUT / f"{sp}_sheet_obl.png"), ttl, elev=12, azim=-70)
    aa.contact_sheet(fk, world03, parents, str(OUT / f"{sp}_sheet_top.png"), ttl, elev=75, azim=-90)
    aa.animate_clip(fk, world03, parents, str(OUT / f"{sp}_gtvs.gif"), ttl, 2, 12)

print("DONE", flush=True)
