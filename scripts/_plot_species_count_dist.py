"""Visualize per-species motion-clip count distribution (473 object_types)."""
import re
from collections import Counter
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA = "data/anytop_planet_zoo_clean_L2"
counts = Counter()
for fn in ("train.txt", "val.txt"):
    for line in open(f"{DATA}/splits/{fn}"):
        s = line.strip()
        if s and not s.startswith("#"):
            counts[re.sub(r"(_[a-z][a-z0-9]*)+__.*", "", s)] += 1

items = counts.most_common()                       # (species, n) high→low
vals = np.array([n for _, n in items])
med, mean = np.median(vals), vals.mean()
print(f"{len(vals)} species, {vals.sum()} clips, min={vals.min()} max={vals.max()} "
      f"median={med:.0f} mean={mean:.0f}")

fig, (a1, a2) = plt.subplots(1, 2, figsize=(15, 5.5))

# left: ranked bar (long-tail shape)
a1.bar(range(len(vals)), vals, width=1.0, color="#4878a8")
a1.axhline(med, ls="--", c="r", alpha=.7, label=f"median {int(med)}")
a1.axhline(mean, ls="--", c="g", alpha=.7, label=f"mean {mean:.0f}")
a1.set_xlabel("species rank (most → fewest clips)")
a1.set_ylabel("# motion clips")
a1.set_title(f"Per-species motion-clip count — 473 species, {vals.sum()} clips")
a1.legend(); a1.grid(alpha=.3)
# annotate extremes
a1.annotate(f"Koala {vals[0]}", xy=(0, vals[0]), xytext=(30, vals[0] + 5), fontsize=8)
a1.annotate(f"Ring_Tailed_Lemur {vals[-1]}", xy=(len(vals) - 1, vals[-1]),
            xytext=(len(vals) - 180, vals[-1] + 18), fontsize=8,
            arrowprops=dict(arrowstyle="->", lw=.7))
# shade the ~180 "reliable-generation threshold" (from quality sweep)
a1.axhline(180, ls=":", c="purple", alpha=.6)
a1.text(len(vals) * 0.42, 188, "~180: gen-quality threshold (below = janky)", color="purple", fontsize=8)

# right: histogram
a2.hist(vals, bins=25, color="orange", edgecolor="k", alpha=.75)
a2.axvline(med, ls="--", c="r", alpha=.7, label=f"median {int(med)}")
a2.set_xlabel("# motion clips per species")
a2.set_ylabel("# species")
a2.set_title("Distribution (双峰: ~120 簇 + ~250 簇)")
a2.legend(); a2.grid(alpha=.3)

plt.tight_layout()
plt.savefig("runs/_species_count_dist.png", dpi=130, bbox_inches="tight")
print("saved runs/_species_count_dist.png")

# also dump the full ranked list to a text file the user can browse
with open("runs/_species_counts_ranked.txt", "w") as fh:
    for sp, n in items:
        fh.write(f"{n:4d}  {sp}\n")
print("saved runs/_species_counts_ranked.txt (full 473 ranked)")
