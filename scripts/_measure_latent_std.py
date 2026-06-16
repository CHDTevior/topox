"""Diagnostic: measure std(z0) of the frozen-VAE latent that the diffusion is
trained on. v-prediction + the noise schedule assume the data ~ unit variance;
if std(z0) is far from 1, the per-timestep SNR is mis-calibrated and generation
quality suffers even on train. Read-only (no training touched)."""
import sys
import torch
sys.path.insert(0, ".")
from torch.utils.data import DataLoader
from scripts.train_denoiser import load_frozen_vae
from src.data.anytop_dataset import AnyTopDataset, collate_fn as anytop_collate_fn
from src.models.graph_salad.batch import GraphMotionBatch

VAE = "runs/m1_l2_anytop13_edgeseg_C128_rot6dfk_w025f100t010_g128_4card_seed42/best_model.pt"
dev = torch.device("cuda")
torch.manual_seed(0)

vae, ta = load_frozen_vae(VAE, dev)
print(f"VAE loaded. temporal_stride={ta['temporal_stride']} d_model={ta['d_model']} "
      f"max_coarse={ta['max_coarse']}")

ds = AnyTopDataset(
    split="train", random_caption=True, random_crop=False,
    num_frames=260, max_joints=144,
    caption_emb_cache="data/anytop_caption_t5_cleanL2_multi.npz",
    data_root="data/anytop_planet_zoo_clean_L2",
)
dl = DataLoader(ds, batch_size=10, shuffle=True, num_workers=4,
                collate_fn=anytop_collate_fn)

rows = []          # [N_valid, D] valid latent vectors
N_BATCHES = 20
for bi, raw in enumerate(dl):
    if bi >= N_BATCHES:
        break
    raw = {k: v.to(dev) if torch.is_tensor(v) else v for k, v in raw.items()}
    batch = GraphMotionBatch.from_collate_dict(raw)
    with torch.no_grad():
        enc = vae.encode(batch, sample=True)          # sample=True == training path
    z0 = enc["z"].float()                              # [B,T_lat,C,D]
    cm = enc["coarse_mask"]                            # [B,C] bool
    fm = enc["frame_mask_lat"]                         # [B,T_lat] bool
    mask_btc = (cm[:, None, :, None] & fm[:, :, None, None]).squeeze(-1)  # [B,T,C]
    rows.append(z0[mask_btc].cpu())                    # [n_valid_btc, D]
    print(f"  batch {bi}: z0 {tuple(z0.shape)}  valid_slots={int(mask_btc.sum())}")

allz = torch.cat(rows, dim=0)                          # [N, D]
N, D = allz.shape
flat = allz.reshape(-1)
per_dim_std = allz.std(dim=0)                          # [D]
per_dim_mean = allz.mean(dim=0)

print("\n========= LATENT z0 STATISTICS (valid positions) =========")
print(f"N_valid_vectors={N}  D={D}  total_scalars={flat.numel()}")
print(f"GLOBAL: mean={flat.mean():.4f}  std={flat.std():.4f}  "
      f"min={flat.min():.3f}  max={flat.max():.3f}")
print(f"TAILS:  |z|>3 frac={(flat.abs() > 3).float().mean():.4f}   "
      f"|z|>5 frac={(flat.abs() > 5).float().mean():.4f}   "
      f"|z|>10 frac={(flat.abs() > 10).float().mean():.5f}")
print(f"PER-DIM std: min={per_dim_std.min():.3f}  "
      f"median={per_dim_std.median():.3f}  mean={per_dim_std.mean():.3f}  "
      f"max={per_dim_std.max():.3f}")
print(f"PER-DIM mean: min={per_dim_mean.min():.3f}  "
      f"median={per_dim_mean.median():.3f}  max={per_dim_mean.max():.3f}")
# how many dims are "dead" (near-zero std → posterior collapse) vs "loud" (std>>1)
print(f"DIMS std<0.1 (near-collapsed)={int((per_dim_std < 0.1).sum())}/{D}   "
      f"std>2 (loud)={int((per_dim_std > 2).sum())}/{D}")
print("\nINTERPRETATION:")
print("  global std ~1.0 (0.7-1.5) → latent well-scaled, diffusion SNR OK.")
print("  global std >>1 or <<1     → mis-scaled; needs a latent scale_factor")
print("                              (divide z0 by std at train, multiply at sample).")
