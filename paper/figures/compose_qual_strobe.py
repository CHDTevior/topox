"""Figure 2: motion strobes of generated clips (skinned, fixed camera, ground at the feet, trail + arrow) composed into one page-width
figure with the clips' own captions.  python paper/figures/compose_qual_strobe.py <renders_dir> <out.pdf>"""
import json, sys, glob, textwrap
from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = Path(sys.argv[1]); OUT = Path(sys.argv[2])
met = {o["clip_id"]: o for o in json.load(open("runs/_figs/val_action_metrics.json"))}
CLIPS = {  # panel name -> (rig display name, clip id)
    "cheetah_sprint": ("Cheetah", "6205049edc9f5d957562"), "moose_run": ("Alaskan moose", "d34d76e2c0a3e5b0d5e3"),
    "lion_pounce": ("African lion (juvenile)", "3b15bf396d2832ca1060"), "wolf_runturn": ("Arctic wolf", "ec7b75675f61b958ff03"),
    "lion_swim": ("African lion (juvenile)", "3532228af91450446d98"), "moose_fight": ("Alaskan moose", "f2d76b31c5b91611f152"),
    "elephant_turn": ("African elephant (juvenile)", "b3714bfdfb418c87c26a"),
}
ROWS = [  # (panel names, row height in inches)
    (["cheetah_sprint", "moose_run"], 0.92),
    (["lion_pounce", "wolf_runturn", "lion_swim"], 0.92),
    (["moose_fight", "elephant_turn"], 0.66),
]
plt.rcParams.update({"font.family": "serif", "font.serif": ["Nimbus Roman", "Times New Roman", "Times"], "font.size": 7, "pdf.fonttype": 42})

def load(name):
    fs = sorted(glob.glob(str(R / name / "strobe.png"))) or sorted(glob.glob(str(R / name / "row_f*.png")))
    ims = [Image.open(f).convert("RGBA") for f in fs]
    if len(ims) > 1:                                          # row mode: tile the frames with a thin gap
        w, h = ims[0].size; gap = 12; sheet = Image.new("RGBA", (w * len(ims) + gap * (len(ims) - 1), h), (255, 255, 255, 255))
        for k, im in enumerate(ims): sheet.alpha_composite(im, (k * (w + gap), 0))
        im = sheet
    else:
        im = ims[0]
    bg = Image.new("RGBA", im.size, (255, 255, 255, 255)); bg.alpha_composite(im); return bg.convert("RGB")

W = 5.5; CAP_H = 0.30; GAP = 0.06
H = sum(h + CAP_H for _, h in ROWS) + GAP * (len(ROWS) - 1)
fig = plt.figure(figsize=(W, H))
y = H
for names, rh in ROWS:
    ims = [load(n) for n in names]
    aspects = [im.width / im.height for im in ims]
    total = sum(a * rh for a in aspects); scale = min(1.0, (W - GAP * (len(names) - 1)) / total)   # shrink a row that is too wide
    rh_eff = rh * scale; x = 0.0
    y -= rh_eff
    for n, im, a in zip(names, ims, aspects):
        w = a * rh_eff
        ax = fig.add_axes([x / W, y / H, w / W, rh_eff / H]); ax.imshow(np.asarray(im)); ax.axis("off")
        rig, cid = CLIPS[n]; cap = met[cid]["caption"]
        label = f"{rig}: “{cap[0].lower() + cap[1:]}”"
        fig.text((x + 0.02) / W, (y - 0.05) / H, "\n".join(textwrap.wrap(label, width=int(w * 24))), ha="left", va="top", fontsize=7, color="#222222", linespacing=1.15)
        x += w + GAP
    y -= CAP_H + GAP
fig.savefig(OUT, dpi=300); fig.savefig(OUT.with_suffix(".png"), dpi=150); print("wrote", OUT, "size", W, round(H, 2), "in")
