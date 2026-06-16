"""One-off: 3-way codebook-size ablation per-layer activation comparison @ ep39.
512 / 1024 / 2048 codes (Q4), merged L4_safe+truebones VQVAE. Data pulled from each
run's train.log at the matched epoch (ep39). Pure plotting, no data load."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

LAYERS = ["q1", "q2", "q3", "q4"]
RUNS = {
    "512":  dict(size=512,  active=[512, 512, 512, 512],      ppl=[424.1, 439.2, 429.6, 414.0], recon=2.015, color="#4C72B0"),
    "1024": dict(size=1024, active=[1022, 1022, 1024, 1024],  ppl=[814.7, 871.2, 857.0, 853.3], recon=1.852, color="#DD8452"),
    "2048": dict(size=2048, active=[2011, 2031, 2044, 2044],  ppl=[1519.9, 1638.4, 1636.7, 1602.6], recon=1.731, color="#55A868"),
}
x = np.arange(len(LAYERS)); w = 0.26

fig, axes = plt.subplots(1, 3, figsize=(18, 5.2))

# --- Panel 1: active codes per layer (with codebook-size reference) ---
ax = axes[0]
for i, (name, d) in enumerate(RUNS.items()):
    bars = ax.bar(x + (i - 1) * w, d["active"], w, label=f"{name} codes", color=d["color"])
    ax.axhline(d["size"], ls="--", lw=1, color=d["color"], alpha=0.5)
ax.set_xticks(x); ax.set_xticklabels(LAYERS)
ax.set_ylabel("active codes (used ≥1×)")
ax.set_title("Active codes per layer (dashed = codebook size)\nall three SATURATE, dead=[0,0,0,0]")
ax.legend(loc="upper left", fontsize=9)
ax.text(0.5, -0.16, "-> all full, dead=0: utilization alone cannot tell which size is enough",
        transform=ax.transAxes, ha="center", fontsize=9, color="#555")

# --- Panel 2: perplexity (effective #codes in use) per layer ---
ax = axes[1]
for i, (name, d) in enumerate(RUNS.items()):
    ax.bar(x + (i - 1) * w, d["ppl"], w, label=f"{name}", color=d["color"])
ax.set_xticks(x); ax.set_xticklabels(LAYERS)
ax.set_ylabel("perplexity (effective # codes)")
frac = {n: np.mean(d["ppl"]) / d["size"] * 100 for n, d in RUNS.items()}
ax.set_title("Perplexity per layer (effective # codes used)\nfraction of codebook ~constant: 512={:.0f}%  1024={:.0f}%  2048={:.0f}%".format(
    frac["512"], frac["1024"], frac["2048"]))
ax.legend(loc="upper left", fontsize=9)
ax.text(0.5, -0.16, "-> effective codes grow with size, but ~80% fraction stays constant: none clearly over-provisioned",
        transform=ax.transAxes, ha="center", fontsize=9, color="#555")

# --- Panel 3: recon @ ep39 (the actual quality signal) ---
ax = axes[2]
names = list(RUNS.keys()); recons = [RUNS[n]["recon"] for n in names]
cols = [RUNS[n]["color"] for n in names]
bars = ax.bar(names, recons, color=cols, width=0.55)
for b, r in zip(bars, recons):
    ax.text(b.get_x() + b.get_width() / 2, r + 0.01, f"{r:.3f}", ha="center", fontsize=11, fontweight="bold")
ax.set_ylabel("epoch-avg train recon total")
ax.set_ylim(1.5, 2.15)
ax.set_xlabel("codebook size")
ax.set_title("Train recon @ same epoch (ep39)\nlarger codebook -> monotonically lower recon (2.02 -> 1.85 -> 1.73)")
ax.text(0.5, -0.16, "-> THIS is the real signal for 'is 512 enough'; still train loss, final verdict = recon QA video",
        transform=ax.transAxes, ha="center", fontsize=9, color="#555")

fig.suptitle("Graph-VQVAE codebook-size ablation @ ep39 - L4_safe+TrueBones, Q=4, code_dim=512, b32 global64 lr6.65e-5 bf16",
             fontsize=13, y=1.02)
fig.tight_layout()
OUT = "/scratch/ts1v23/workspace/noKslot_clean/analysis_caption_wordcloud/codebook_ablation_ep39_3way.png"
import os
os.makedirs(os.path.dirname(OUT), exist_ok=True)
fig.savefig(OUT, dpi=130, bbox_inches="tight")
print("SAVED:", OUT)
print("\n=== ep39 per-layer table ===")
print(f"{'run':>6} {'size':>5} {'active(q1-q4)':>22} {'dead':>12} {'ppl(q1-q4)':>30} {'ppl/size%':>9} {'recon':>7}")
for n, d in RUNS.items():
    pf = np.mean(d["ppl"]) / d["size"] * 100
    print(f"{n:>6} {d['size']:>5} {str(d['active']):>22} {'[0,0,0,0]':>12} {str(d['ppl']):>30} {pf:>8.1f}% {d['recon']:>7.3f}")
