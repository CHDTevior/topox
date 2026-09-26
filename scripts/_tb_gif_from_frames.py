"""Assemble a Blender frame folder into a labelled GIF at the clip's own frame rate.

Runs under the project's python (Blender's has no Pillow). The header carries what the reader needs
to judge the clip: the rig, the prompt it was generated from, and which colour is which.
  python scripts/_tb_gif_from_frames.py <render_dir> [<render_dir> ...]
"""
import json, sys, textwrap
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

GEN = (31, 111, 139)
GT = (181, 85, 47)
INK = (34, 34, 34)


def font(sz):
    for p in ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
              "/usr/share/fonts/dejavu/DejaVuSans.ttf"):
        if Path(p).is_file():
            return ImageFont.truetype(p, sz)
    return ImageFont.load_default()


def build(d: Path):
    meta = json.loads((d / "render.json").read_text())
    frames = sorted((d / "frames").glob("f*.png"))
    if not frames:
        print(f"[gif] {d}: no frames"); return
    w, h = Image.open(frames[0]).size
    head = int(h * 0.145)
    f_big, f_small = font(max(11, int(head * 0.30))), font(max(10, int(head * 0.24)))
    cap = " ".join(str(meta["caption"]).split())
    lines = textwrap.wrap(cap, width=max(30, int(w / (f_small.size * 0.52))))[:2]
    out = []
    for k, p in enumerate(frames):
        im = Image.open(p).convert("RGB")
        canvas = Image.new("RGB", (w, h + head), (255, 255, 255))
        canvas.paste(im, (0, head))
        dr = ImageDraw.Draw(canvas)
        dr.text((10, 4), f"{meta['rig']} - {meta['clip']}", font=f_big, fill=INK)
        y = 6 + f_big.size + 2
        for ln in lines:
            dr.text((10, y), "“" + ln + "”" if ln is lines[0] else ln, font=f_small, fill=(120, 120, 120))
            y += f_small.size + 1
        if meta.get("also_gt"):
            legend_y = head - f_small.size - 3
            dr.rectangle([w - 150, legend_y + 3, w - 140, legend_y + 11], fill=GEN)
            dr.text((w - 136, legend_y), "generated", font=f_small, fill=INK)
            dr.rectangle([w - 66, legend_y + 3, w - 56, legend_y + 11], fill=GT)
            dr.text((w - 52, legend_y), "real", font=f_small, fill=INK)
        dr.text((w - 46, h + head - f_small.size - 4), f"{k+1}/{len(frames)}", font=f_small,
                fill=(150, 150, 150))
        out.append(canvas.convert("P", palette=Image.ADAPTIVE))
    dur = int(round(1000.0 / float(meta["fps"])))
    gif = d / f"{d.name}.gif"
    out[0].save(gif, save_all=True, append_images=out[1:], duration=dur, loop=0, optimize=True)
    mb = gif.stat().st_size / 1e6
    print(f"[gif] {gif}  {len(out)} frames @ {meta['fps']:g} fps ({dur} ms)  {mb:.1f} MB", flush=True)


if __name__ == "__main__":
    for a in sys.argv[1:]:
        build(Path(a))
