"""Compose the paper figures from paper-style skinned stills (PNG RGBA, transparent world) and skeleton line art.

  python paper/figures/compose_figs.py qual  <spec.json> <out.pdf>   # Fig 2: per rig  caption / real strip / generated strip
  python paper/figures/compose_figs.py strip <dir> <view> <out.png>   # one cropped filmstrip (debug)

spec.json for `qual`: {"rigs": [{"label": "Plains zebra (38 j.)", "caption": "...", "gt": "<dir with <view>_fNNNN.png>",
                                 "gen": "<dir>", "view": "threequarter"}, ...], "cols": 2}
Every strip is cropped to the union of its frames' alpha bounding boxes (same box for all frames of a strip, so the
animal's translation stays visible), padded, and composited on white. Text is set in the paper's serif face.
"""
import json, sys
from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import gridspec

plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Nimbus Roman", "DejaVu Serif"], "font.size": 8,
                     "pdf.fonttype": 42})
ACCENT = "#b5552f"; INK = "#222222"; MUTE = "#6b6b6b"

def load_stills(d, view):
    fs = sorted(Path(d).glob(f"{view}_f*.png"))
    if not fs: raise SystemExit(f"no stills {view}_f*.png in {d}")
    return [Image.open(f).convert("RGBA") for f in fs]

def union_bbox(ims, alpha_thr=200, pad=0.06, pad_bottom=0.16):
    """Box around the SUBJECT (alpha ~ 255); the soft contact shadow (partial alpha) is excluded from the
    measurement and given room by the larger bottom pad instead."""
    boxes = []
    for im in ims:
        a = np.asarray(im)[..., 3]
        ys, xs = np.where(a > alpha_thr)
        if len(xs): boxes.append((xs.min(), ys.min(), xs.max(), ys.max()))
    x0 = min(b[0] for b in boxes); y0 = min(b[1] for b in boxes); x1 = max(b[2] for b in boxes); y1 = max(b[3] for b in boxes)
    w, h = x1 - x0, y1 - y0; px, py = int(w * pad), int(h * pad)
    W, H = ims[0].size
    return (max(0, x0 - px), max(0, y0 - py), min(W, x1 + px), min(H, y1 + int(h * pad_bottom)))

def strip(ims, box=None, bg=(255, 255, 255, 255)):
    """Crop every frame to `box` (union bbox by default) and composite on white; returns a list of RGB arrays."""
    box = box or union_bbox(ims)
    out = []
    for im in ims:
        c = im.crop(box); b = Image.new("RGBA", c.size, bg); b.alpha_composite(c); out.append(np.asarray(b.convert("RGB")))
    return out

def qual(spec_path, out_path):
    spec = json.load(open(spec_path)); rigs = spec["rigs"]; cols = int(spec.get("cols", 2)); n_fr = int(spec.get("frames", 5))
    rows = int(np.ceil(len(rigs) / cols))
    fig_w = 6.75                                  # ICLR text width (in)
    cell_h = float(spec.get("cell_h", 1.6))         # label + wrapped caption + strips
    fig = plt.figure(figsize=(fig_w, cell_h * rows))
    outer = gridspec.GridSpec(rows, cols, figure=fig, wspace=0.05, hspace=0.42, left=0.005, right=0.995, top=0.985, bottom=0.01)
    for i, r in enumerate(rigs):
        view = r.get("view", "threequarter")
        gen = load_stills(r["gen"], view)
        gt = load_stills(r["gt"], view) if r.get("gt") else []          # optional: figures may show our generations only
        k = min([n_fr, len(gen)] + ([len(gt)] if gt else [])); gen = gen[:k]; gt = gt[:k]
        box = union_bbox(gt + gen)                # one box for real AND generated: same scale, same framing
        rows_ = ((gt, "real"), (gen, "generated")) if gt else ((gen, "generated"),)
        inner = gridspec.GridSpecFromSubplotSpec(len(rows_), k, subplot_spec=outer[i // cols, i % cols], wspace=0.02, hspace=0.04)
        for row, (ims, lab) in enumerate(rows_):
            arrs = strip(ims, box)
            for j, a in enumerate(arrs):
                ax = fig.add_subplot(inner[row, j]); ax.imshow(a); ax.axis("off")
                if j == 0 and len(rows_) > 1:
                    ax.text(-0.02, 0.5, lab, transform=ax.transAxes, rotation=90, va="center", ha="right", fontsize=6.5,
                            color=ACCENT if lab == "generated" else MUTE)
                if row == 0 and j == 0:
                    import textwrap
                    cap = "\n".join(textwrap.wrap(f"“{r['caption']}”", 62))
                    ax.set_title(f"{r['label']}\n{cap}", loc="left", fontsize=6.4, color=INK, pad=3, x=0.0, linespacing=1.15)
    fig.savefig(out_path, dpi=300, bbox_inches="tight", pad_inches=0.02)
    print(f"wrote {out_path}")

if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "qual": qual(sys.argv[2], sys.argv[3])
    elif mode == "strip":
        ims = load_stills(sys.argv[2], sys.argv[3]); arrs = strip(ims)
        Image.fromarray(np.concatenate(arrs, 1)).save(sys.argv[4]); print("wrote", sys.argv[4])
    else: raise SystemExit("mode: qual | strip")
