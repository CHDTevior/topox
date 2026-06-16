"""Plot the H100 mean-diffusion train_loss + val_denoise curves over the WHOLE
run: original (ep0→122, train_ep122_crashbak.log) + resumed (ep121→, train.log),
merged by epoch (resumed wins on overlap). Marks the crash/resume boundary."""
import re
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

D = "runs/m2_t2m_cleanL2_Bep79rot6dfk_d512C128_n11ff1536_h100x6_seed42"
ORIG = f"{D}/train_ep122_crashbak.log"   # original run (pre-crash)
RESUME = f"{D}/train.log"                # resumed run

RE_TRAIN = re.compile(r"epoch (\d+) done.*?train_loss=([\d.]+)")
RE_VAL = re.compile(r"val ep(\d+)\].*?val_denoise=([\d.]+)")


def parse(path):
    tr, va = {}, {}
    try:
        txt = open(path).read()
    except FileNotFoundError:
        return tr, va
    for m in RE_TRAIN.finditer(txt):
        tr[int(m.group(1))] = float(m.group(2))
    for m in RE_VAL.finditer(txt):
        va[int(m.group(1))] = float(m.group(2))
    return tr, va


tr_o, va_o = parse(ORIG)
tr_r, va_r = parse(RESUME)
# merge: original first, resumed overrides on overlapping epochs (it is the live continuation)
tr = {**tr_o, **tr_r}
va = {**va_o, **va_r}
resume_ep = min(tr_r) if tr_r else None

tr_x = sorted(tr); tr_y = [tr[e] for e in tr_x]
va_x = sorted(va); va_y = [va[e] for e in va_x]
print(f"train_loss: {len(tr_x)} pts ep{tr_x[0]}→{tr_x[-1]}  "
      f"{tr_y[0]:.4f}→{tr_y[-1]:.4f}")
print(f"val_denoise: {len(va_x)} pts ep{va_x[0]}→{va_x[-1]}  "
      f"{va_y[0]:.4f}→{va_y[-1]:.4f}")
print(f"resume boundary ≈ ep{resume_ep}")

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 8), sharex=True)

ax1.plot(tr_x, tr_y, "-o", ms=3, lw=1.3, color="#c0392b", label="train_loss (sample=True, +CFG-drop)")
ax1.set_ylabel("train_loss (v-MSE)")
ax1.set_title("H100 mean-diffusion (n11/d_ff1536, fp32, const lr 6.25e-4) — loss curves")
ax1.grid(alpha=0.3)
ax1.legend(loc="upper right", fontsize=9)

ax2.plot(va_x, va_y, "-o", ms=3, lw=1.3, color="#2471a3", label="val_denoise (sample=False, clean)")
ax2.set_ylabel("val_denoise (v-MSE)")
ax2.set_xlabel("epoch")
ax2.grid(alpha=0.3)
ax2.legend(loc="upper right", fontsize=9)

for ax in (ax1, ax2):
    if resume_ep is not None:
        ax.axvline(resume_ep, color="gray", ls="--", lw=1, alpha=0.7)
# annotate resume on top axis
if resume_ep is not None:
    ax1.annotate(f"crash@ep123 → full-resume@ep{resume_ep}",
                 xy=(resume_ep, tr_y[tr_x.index(resume_ep)] if resume_ep in tr_x else tr_y[-1]),
                 xytext=(resume_ep + 4, max(tr_y) - 0.004), fontsize=8, color="gray",
                 arrowprops=dict(arrowstyle="->", color="gray", lw=0.8))

# annotate the plateau values
ax1.annotate(f"plateau ≈ {tr_y[-1]:.4f}", xy=(tr_x[-1], tr_y[-1]),
             xytext=(tr_x[-1] - 45, tr_y[-1] + 0.006), fontsize=8, color="#c0392b")
ax2.annotate(f"plateau ≈ {va_y[-1]:.4f}", xy=(va_x[-1], va_y[-1]),
             xytext=(va_x[-1] - 45, va_y[-1] + 0.0035), fontsize=8, color="#2471a3")

plt.tight_layout()
OUT = "runs/_diffusion_loss_curves.png"
plt.savefig(OUT, dpi=130, bbox_inches="tight")
print(f"saved {OUT}")
