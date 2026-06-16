"""Test generation ability vs training clip-count: does the NEW T1 (H100 bf16-mean
conservative, currently training) generate jankier motion for RARE species (fewer
training clips)? Sweep ~40 species spanning the count range; per species' val
clip0, DDIM-generate + decode, measure speed_ratio (gen/GT mean ||Δpos||) and
jerk_ratio (gen/GT of mean ||Δ²pos||/mean||Δpos||); correlate vs clip count."""
import re
import sys
from collections import Counter
import numpy as np
import torch
sys.path.insert(0, ".")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from src.models.graph_salad.batch import GraphMotionBatch
from src.data.anytop_dataset import AnyTopDataset, collate_fn as anytop_collate_fn
from scripts.train_denoiser import load_frozen_vae
from scripts.animate_denoiser import (load_denoiser, ddim_sample, make_fake_enc,
                                      _recover_world_positions, _STD_FLOOR)

dev = torch.device("cuda"); torch.manual_seed(0)
VAE = "runs/m1_bf16_anytop13_rot6dfk_w025f100t010_C128_8card_xnode_seed42/best_recon_model.pt"
DEN = "runs/m2_t2m_cleanL2_bf16ep209MEAN_lr6.25e-5cos_h100x6_seed42/best_model.pt"  # NEW T1 H100 (ep50)
DATA = "data/anytop_planet_zoo_clean_L2"
N_SEL = 50

# 1. species -> clip count, select N_SEL evenly across the FULL count range
#    (linspace over the count-sorted rank so the RARE tail 23-133 is included).
counts = Counter()
for fn in ("train.txt", "val.txt"):
    for line in open(f"{DATA}/splits/{fn}"):
        s = line.strip()
        if s and not s.startswith("#"):
            counts[re.sub(r"(_[a-z][a-z0-9]*)+__.*", "", s)] += 1
ranked = [sp for sp, _ in counts.most_common()]  # high→low count
idxs = sorted(set(np.linspace(0, len(ranked) - 1, N_SEL).round().astype(int).tolist()))
sel = set(ranked[i] for i in idxs)

# 2. models + val dataset
vae, vta = load_frozen_vae(VAE, dev)
denoiser, ck = load_denoiser(DEN, dev)
da = ck["args"]; d_model = vta["d_model"]; stride = vta["temporal_stride"]
print(f"denoiser ep={ck.get('epoch')}  selected {len(sel)} species (count {min(counts[s] for s in sel)}-{max(counts[s] for s in sel)})")
sched_kwargs = dict(num_train_timesteps=da.get("num_train_timesteps", 1000),
    beta_start=da.get("beta_start", 0.00085), beta_end=da.get("beta_end", 0.012),
    beta_schedule=da.get("beta_schedule", "scaled_linear"),
    prediction_type="v_prediction", clip_sample=False)
ds = AnyTopDataset(split="val", random_caption=False, random_crop=False,
    num_frames=da.get("max_frames", 260), max_joints=144,
    caption_emb_cache="data/anytop_caption_t5_cleanL2_multi.npz", data_root=DATA)
match = [i for i, s in enumerate(ds.samples) if s.get("object_type") in sel]


def jit(m):
    if m.shape[0] < 3:
        return None, None
    spd = float(np.linalg.norm(np.diff(m, axis=0), axis=-1).mean())
    acc = float(np.linalg.norm(np.diff(m, n=2, axis=0), axis=-1).mean())
    return spd, acc / (spd + 1e-9)


rows = []; picked = set()
for i in match:
    item = ds[i]; sp = item["object_type"]
    if sp in picked:
        continue
    raw = anytop_collate_fn([item]); raw = {k: v.to(dev) if torch.is_tensor(v) else v for k, v in raw.items()}
    batch = GraphMotionBatch.from_collate_dict(raw)
    fml = batch.frame_mask.view(1, batch.frame_mask.shape[1] // stride, stride).all(dim=-1)
    J = int(item["num_joints"]); T = min(int(item["num_frames"]), int(fml[0].sum().item() * stride))
    if T < 16:
        continue
    with torch.no_grad():
        skel = vae.encode_skeleton_only(batch)
        z = ddim_sample(denoiser, batch, skel, fml, 50, 1.0, sched_kwargs, dev, d_model)
        dec = vae.decode(make_fake_enc(z, skel, fml), batch)["pred_motion"]
    std = raw["anytop_std"][0, :J].cpu().numpy(); mean = raw["anytop_mean"][0, :J].cpu().numpy()
    pred_raw = dec[0, :T, :J, :].cpu().numpy() * (std[None] + _STD_FLOOR) + mean[None]
    gen = _recover_world_positions(pred_raw)
    gt = batch.motion_features[0, :T, :J, :3].cpu().numpy()
    g_spd, g_acc = jit(gt); p_spd, p_acc = jit(gen)
    rows.append((sp, counts[sp], T, p_spd / max(g_spd, 1e-9), p_acc / max(g_acc, 1e-9)))
    picked.add(sp)

rows.sort(key=lambda r: r[1])
print(f"\n{'species':40s}{'cnt':>5s}{'T':>5s}{'spdR':>7s}{'jerkR':>7s}")
for sp, c, T, sR, aR in rows:
    print(f"{sp:40s}{c:5d}{T:5d}{sR:7.2f}{aR:7.2f}")

cnts = np.array([r[1] for r in rows]); spdR = np.array([r[3] for r in rows]); accR = np.array([r[4] for r in rows])
pear = lambda x, y: float((((x - x.mean()) / (x.std() + 1e-9)) * ((y - y.mean()) / (y.std() + 1e-9))).mean())
rs, ra = pear(cnts, spdR), pear(cnts, accR)
# robust: median |speed_ratio-1| (closer to 1 = better) vs count, + Spearman (rank)
def spearman(x, y):
    rx = np.argsort(np.argsort(x)); ry = np.argsort(np.argsort(y)); return pear(rx.astype(float), ry.astype(float))
sp_dev = np.abs(spdR - 1.0)  # distance from ideal (gen speed == GT)
print(f"\nPearson  corr(count, speed_ratio)={rs:.3f}  corr(count, jerk_ratio)={ra:.3f}")
print(f"Spearman corr(count, |speedR-1|)={spearman(cnts, sp_dev):.3f}  corr(count, jerk_ratio)={spearman(cnts, accR):.3f}")
print("  负相关 = clip 越多 → 越好(ratio近1/jerk低) = 稀有物种更鬼畜(假设成立)")
# binned means (averages out per-clip noise)
bins = [(0, 100), (100, 140), (140, 180), (180, 250), (250, 999)]
print(f"\n{'count-bin':>12s}{'n':>4s}{'med|spdR-1|':>12s}{'med jerkR':>11s}{'mean spdR':>11s}")
for lo, hi in bins:
    m = [(r[3], r[4]) for r in rows if lo <= r[1] < hi]
    if m:
        s = np.array([x[0] for x in m]); j = np.array([x[1] for x in m])
        print(f"{f'{lo}-{hi}':>12s}{len(m):4d}{np.median(np.abs(s-1)):12.2f}{np.median(j):11.2f}{s.mean():11.2f}")
fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5))
a1.scatter(cnts, spdR); a1.axhline(1, ls="--", c="g", alpha=.5); a1.grid(alpha=.3)
a1.set_xlabel("train clip count"); a1.set_ylabel("speed_ratio (gen/GT)"); a1.set_title(f"speed_ratio vs count (r={rs:.2f})")
a2.scatter(cnts, accR, c="orange"); a2.axhline(1, ls="--", c="g", alpha=.5); a2.grid(alpha=.3)
a2.set_xlabel("train clip count"); a2.set_ylabel("jerk_ratio (gen/GT)"); a2.set_title(f"jerk_ratio vs count (r={ra:.2f})")
fig.suptitle(f"NEW T1 (bf16-mean ep{ck.get('epoch')}) generation quality vs training clip-count, {len(rows)} val species")
plt.tight_layout(); plt.savefig("runs/_quality_vs_count.png", dpi=130, bbox_inches="tight")
print("saved runs/_quality_vs_count.png")
