"""Static filmstrip (world-coords, NOT root-centered) of converted HumanML3D
motion using the AnyTop PIL renderer, so global trajectory + facing + upright
can be checked against the drawn axes. Examinable as a single PNG."""
import sys
from pathlib import Path
import numpy as np
import torch
from PIL import Image

HM = "/iridisfs/scratch/ts1v23/workspace/motion-latent-diffusion-main"
REPO = "/iridisfs/scratch/ts1v23/workspace/noKslot_clean"
OUT = REPO + "/data/humanml3d_anytop13"
sys.path.insert(0, HM); sys.path.insert(0, REPO)
from src.data.anytop_dataset import _recover_world_positions
import scripts._pil_skeleton_render as pr

OBJ = "HML3D_Human"
cond = np.load(Path(OUT) / "cond.npy", allow_pickle=True).item()[OBJ]
parents = [int(p) for p in cond["parents"]]
CELL = (760, 640)


def strip(mid, k=8):
    for d in ("motions", "motions_heldout"):
        p = Path(OUT) / d / f"{OBJ}_{mid}.npy"
        if p.exists():
            raw = np.load(p); break
    else:
        print("skip", mid); return
    pos = _recover_world_positions(raw.astype(np.float32)).astype(np.float64)
    pos[..., 1] -= pos[..., 1].min()                     # feet to ground (world XZ kept)
    T = pos.shape[0]
    idxs = [int(round(v)) for v in np.linspace(0, T - 1, k)]
    transform = pr.compute_transform([pos[i] for i in idxs], CELL, 0.06, 1.0)  # WORLD coords
    panels = []
    for j, fi in enumerate(idxs):
        img = pr.render_panel(pos, parents, fi, transform, CELL, f"f{fi}",
                              (35, 112, 180), 3, 4, axes=(j == 0), static=False)
        panels.append(img)
    W, H = CELL
    canvas = Image.new("RGB", (W * len(panels), H), "white")
    for j, im in enumerate(panels):
        canvas.paste(im, (W * j, 0))
    op = Path(OUT) / "animations" / "conversion_qa_20260619_large" / f"{mid}_filmstrip.png"
    canvas.save(op)
    print("ok", mid, "->", op.name)


if __name__ == "__main__":
    ids = sys.argv[1:] or ["000006", "000005", "000000"]
    for m in ids:
        strip(m)
