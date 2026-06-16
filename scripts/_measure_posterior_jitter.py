"""KILL-GATE diagnostic (codex 019e9620 Q2b): on the bf16 VAE the NEW run uses,
does the VAE POSTERIOR NOISE (σε) alone already cause 'janky' motion? The
diffusion trains on sample=True latents z=mu+σε, so if decode(mu+σε) is already
jerky vs decode(mu), lower-LR diffusion CANNOT escape it → kill + redirect to a
mu-target / posterior-noise fix.

For N (non-short) train clips, decode mu (sample=False) vs K posterior draws
mu+σε (sample=True), via the SAME make_fake_enc path, and compare:
  - speed jitter  = mean_t ‖Δpos‖
  - accel jitter  = mean_t ‖Δ²pos‖ / (mean_t ‖Δpos‖ + eps)   ← jerkiness ('鬼畜') signature
posterior_noise_R = mean_k metric(post_k) / metric(mu).
  >> 1  → posterior noise itself is the jank → KILL new run, target mu.
  ~ 1   → posterior noise benign → jank is the diffusion → new run worth running.
"""
import random
import sys
import numpy as np
import torch
sys.path.insert(0, ".")
from src.models.graph_salad.batch import GraphMotionBatch
from src.data.anytop_dataset import AnyTopDataset, collate_fn as anytop_collate_fn
from scripts.train_denoiser import load_frozen_vae
from scripts.animate_denoiser import make_fake_enc

dev = torch.device("cuda")
VAE = "runs/m1_bf16_anytop13_rot6dfk_w025f100t010_C128_8card_xnode_seed42/best_recon_model.pt"
N_CLIPS = 16
K_POST = 3
MIN_T = 32

vae, vta = load_frozen_vae(VAE, dev)
stride = vta["temporal_stride"]
ds = AnyTopDataset(split="train", random_caption=False, random_crop=False,
                   num_frames=260, max_joints=144,
                   caption_emb_cache="data/anytop_caption_t5_cleanL2_multi.npz",
                   data_root="data/anytop_planet_zoo_clean_L2")


def jitters(motion, T):
    m = motion[0, :T, :, :3]                            # [T,J,3] normalised pos
    if T < 3:
        return None, None
    d1 = (m[1:] - m[:-1]).norm(dim=-1)                  # [T-1,J] speed
    d2 = (m[2:] - 2 * m[1:-1] + m[:-2]).norm(dim=-1)    # [T-2,J] accel
    spd = float(d1.mean().item())
    acc = float(d2.mean().item()) / (spd + 1e-9)        # jerkiness, speed-normalised
    return spd, acc


rng = random.Random(123)
order = rng.sample(range(len(ds)), len(ds))
rows = []
print(f"bf16 VAE posterior-noise kill-gate  (K={K_POST} draws, T>={MIN_T})")
print(f"{'species':30s} {'T':>4s} {'spd_mu':>7s} {'spd_pst':>7s} {'spdR':>5s}  "
      f"{'acc_mu':>7s} {'acc_pst':>7s} {'accR':>5s}")
for i in order:
    if len(rows) >= N_CLIPS:
        break
    item = ds[i]
    if int(item["num_frames"]) < MIN_T:
        continue
    raw = anytop_collate_fn([item])
    raw = {k: v.to(dev) if torch.is_tensor(v) else v for k, v in raw.items()}
    batch = GraphMotionBatch.from_collate_dict(raw)
    fml = batch.frame_mask.view(1, batch.frame_mask.shape[1] // stride, stride).all(dim=-1)
    T = int(fml[0].sum().item() * stride)
    with torch.no_grad():
        skel = vae.encode_skeleton_only(batch)
        z_mu = vae.encode(batch, sample=False)["z"].float()
        mot_mu = vae.decode(make_fake_enc(z_mu, skel, fml), batch)["pred_motion"]
        spd_mu, acc_mu = jitters(mot_mu, T)
        spds, accs = [], []
        for k in range(K_POST):
            z_p = vae.encode(batch, sample=True)["z"].float()
            mot_p = vae.decode(make_fake_enc(z_p, skel, fml), batch)["pred_motion"]
            s, a = jitters(mot_p, T)
            spds.append(s); accs.append(a)
    spd_p = float(np.median(spds)); acc_p = float(np.median(accs))
    spdR = spd_p / (spd_mu + 1e-9); accR = acc_p / (acc_mu + 1e-9)
    rows.append((spdR, accR))
    print(f"{item['object_type']:30s} {T:4d} {spd_mu:7.4f} {spd_p:7.4f} {spdR:5.2f}  "
          f"{acc_mu:7.4f} {acc_p:7.4f} {accR:5.2f}")

spdRs = [r[0] for r in rows]; accRs = [r[1] for r in rows]
print(f"\nMEAN  posterior_noise_R: speed={np.mean(spdRs):.2f}  accel(jerk)={np.mean(accRs):.2f}")
print(f"MEDIAN posterior_noise_R: speed={np.median(spdRs):.2f}  accel(jerk)={np.median(accRs):.2f}")
print("\nKILL GATE:")
print("  posterior_noise_R >> 1 (esp accel/jerk) → VAE posterior noise IS the jank →")
print("    KILL new run, redirect to mu-target / reduce-σ diffusion.")
print("  posterior_noise_R ~ 1 → posterior noise benign → jank is diffusion → keep new run.")
