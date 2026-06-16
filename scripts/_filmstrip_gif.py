"""Stitch a gif's frames into a grid filmstrip for ONE panel, so motion (not a
single frame) can be inspected. Usage: python _filmstrip_gif.py <gif> <panel> <out>
panel: 0=input 1=PRED_RIC 2=PRED_FK 3=GT (4-panel --with_gt layout, 900px each)."""
import sys
from PIL import Image, ImageSequence, ImageDraw

gif, panel, out = sys.argv[1], int(sys.argv[2]), sys.argv[3]
g = Image.open(gif)
frames = [f.copy().convert("RGB") for f in ImageSequence.Iterator(g)]
n = len(frames)
W, H = frames[0].size
pw = W // 4                                  # panel width (4 panels)
x0 = panel * pw
cells = []
for fr in frames:
    c = fr.crop((x0, 0, x0 + pw, H))         # one panel column, full height
    cells.append(c)

# grid: cols x rows
import math
cols = min(6, n)
rows = math.ceil(n / cols)
cell_w = 1900 // cols                          # fit within ~1900 px wide
cell_h = int(cell_w * H / pw)
grid = Image.new("RGB", (cols * cell_w, rows * cell_h), "white")
dr = ImageDraw.Draw(grid)
for i, c in enumerate(cells):
    c = c.resize((cell_w, cell_h))
    r, col = divmod(i, cols)
    grid.paste(c, (col * cell_w, r * cell_h))
    dr.text((col * cell_w + 4, r * cell_h + 2), f"f{i}", fill=(200, 0, 0))
# cap height to 2000
if grid.size[1] > 2000:
    grid = grid.resize((int(grid.size[0] * 2000 / grid.size[1]), 2000))
grid.save(out)
print(f"saved {out}  ({n} frames, panel {panel}, grid {cols}x{rows}, size {grid.size})")
