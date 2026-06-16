#!/usr/bin/env python3
"""One-off QA helper for large T2M gifs (read-only on the gifs).

perclip (default): per gif, stack N frames VERTICALLY, each frame width-fit to
  --maxbox, so wide T2M panels stay legible under the 2000px image-read limit.
  Emits one PNG per gif: _qa_strip_<species>.png
grid: one row per gif (N frames side-by-side), all stacked into one montage.

Run: <python> scripts/_t2m_qa_filmstrip.py --dir <qa_out_dir> [--mode perclip] [--nframes 3]
"""
import argparse
import glob
import os

from PIL import Image, ImageSequence, ImageDraw

ap = argparse.ArgumentParser()
ap.add_argument("--dir", required=True)
ap.add_argument("--mode", choices=["perclip", "grid", "compare"], default="perclip")
ap.add_argument("--dir2", default=None, help="compare mode: 2nd dir (latest)")
ap.add_argument("--label1", default="best")
ap.add_argument("--label2", default="latest")
ap.add_argument("--nframes", type=int, default=3)
ap.add_argument("--maxbox", type=int, default=1880)
args = ap.parse_args()

gifs = sorted(glob.glob(os.path.join(args.dir, "*.gif")))
if not gifs:
    raise SystemExit(f"no gifs in {args.dir}")


def frames_of(g, k):
    im = Image.open(g)
    fr = [f.convert("RGB") for f in ImageSequence.Iterator(im)]
    n = len(fr)
    kk = min(k, n)
    idxs = [round(i * (n - 1) / (kk - 1)) for i in range(kk)] if kk > 1 else [0]
    return fr, n, idxs


def label(img, txt):
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, 8 * len(txt) + 6, 16], fill=(0, 0, 0))
    d.text((3, 3), txt, fill=(255, 255, 0))


if args.mode == "perclip":
    outs = []
    for g in gifs:
        name = os.path.basename(g).replace("_clip0_t2m_large.gif", "").replace("PZ_", "")
        fr, n, idxs = frames_of(g, args.nframes)
        cells = []
        for j in idxs:
            f = fr[j]
            w, h = f.size
            cells.append((f.resize((args.maxbox, int(h * args.maxbox / w))), j))
        H = sum(c.size[1] for c, _ in cells)
        mont = Image.new("RGB", (args.maxbox, H), (255, 255, 255))
        y = 0
        for c, j in cells:
            label(c, f"{name}  frame {j}/{n - 1}")
            mont.paste(c, (0, y))
            y += c.size[1]
        out = os.path.join(args.dir, f"_qa_strip_{name}.png")
        mont.save(out)
        outs.append(out)
        print(f"[perclip] {name}: {out} {mont.size}")
    print("OUTS=" + ",".join(outs))
elif args.mode == "compare":
    gb = {os.path.basename(g): g for g in gifs}
    gl = {os.path.basename(g): g for g in sorted(glob.glob(os.path.join(args.dir2, "*.gif")))}
    common = sorted(set(gb) & set(gl))
    if not common:
        raise SystemExit("compare: no common gifs between --dir and --dir2")
    for nm in common:
        sp = nm.replace("_recon_3panel_large.gif", "").replace("_clip0_t2m_large.gif", "").replace("PZ_", "")
        rows = []
        for tag, gp in [(args.label1, gb[nm]), (args.label2, gl[nm])]:
            fr, n, idxs = frames_of(gp, args.nframes)
            strip = [fr[j].resize((int(fr[j].size[0] * 300 / fr[j].size[1]), 300)) for j in idxs]
            row = Image.new("RGB", (sum(s.size[0] for s in strip), 300), (255, 255, 255))
            x = 0
            for s in strip:
                row.paste(s, (x, 0))
                x += s.size[0]
            label(row, f"{tag}  {sp}  f{idxs}")
            rows.append(row)
        mw, mh = max(r.size[0] for r in rows), sum(r.size[1] for r in rows)
        mont = Image.new("RGB", (mw, mh), (255, 255, 255))
        y = 0
        for r in rows:
            mont.paste(r, (0, y))
            y += r.size[1]
        sc = min(1.0, args.maxbox / mont.size[0], args.maxbox / mont.size[1])
        if sc < 1.0:
            mont = mont.resize((int(mont.size[0] * sc), int(mont.size[1] * sc)))
        out = os.path.join(args.dir, f"_cmp_{sp}.png")
        mont.save(out)
        print(f"[compare] {sp}: {out} {mont.size}")
else:
    rows = []
    for g in gifs:
        name = os.path.basename(g).replace("_clip0_t2m_large.gif", "").replace("PZ_", "")
        fr, n, idxs = frames_of(g, args.nframes)
        strip = [fr[j].resize((int(fr[j].size[0] * 240 / fr[j].size[1]), 240)) for j in idxs]
        row = Image.new("RGB", (sum(s.size[0] for s in strip), 240), (255, 255, 255))
        x = 0
        for s in strip:
            row.paste(s, (x, 0))
            x += s.size[0]
        label(row, f"{name} [{n}f {idxs}]")
        rows.append(row)
    mw, mh = max(r.size[0] for r in rows), sum(r.size[1] for r in rows)
    mont = Image.new("RGB", (mw, mh), (255, 255, 255))
    y = 0
    for r in rows:
        mont.paste(r, (0, y))
        y += r.size[1]
    sc = min(1.0, args.maxbox / mont.size[0], args.maxbox / mont.size[1])
    if sc < 1.0:
        mont = mont.resize((int(mont.size[0] * sc), int(mont.size[1] * sc)))
    out = os.path.join(args.dir, "_qa_filmstrip.png")
    mont.save(out)
    print(f"[grid] {len(gifs)} gifs -> {out} {mont.size}")
