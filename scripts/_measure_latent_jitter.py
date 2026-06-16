"""Diagnostic (read-only): localise the 'janky' generation — is it in the
LATENT (the diffusion produces temporally-jittery ẑ0) or in the DECODE (ẑ0 is
smooth but off-manifold and the VAE decoder amplifies it)?

For N train clips, compare TRUE z0 (encode, sample=False) vs SAMPLED ẑ0 (DDIM
text+skel conditioned) on TWO axes:
  - latent temporal jitter: mean over valid frames of per-slot ‖z[t+1]-z[t]‖
  - motion temporal jitter: decode BOTH via the SAME make_fake_enc path, then
    mean over valid frames/joints of ‖pos[t+1]-pos[t]‖ (ch0:3, normalised — the
    gen/recon RATIO is scale-invariant).

Read:
  latent_ratio = jitter(ẑ0)/jitter(z0) ; motion_ratio = jitter(gen)/jitter(recon)
  latent_ratio ≈ motion_ratio >> 1  → jank is IN THE LATENT (diffusion makes jittery ẑ0)
  latent_ratio ≈ 1 but motion_ratio >> 1 → ẑ0 smooth but off-manifold → DECODE amplifies
"""
import sys
import numpy as np
import torch
sys.path.insert(0, ".")
from src.models.graph_salad.batch import GraphMotionBatch
from src.data.anytop_dataset import AnyTopDataset, collate_fn as anytop_collate_fn
from scripts.train_denoiser import load_frozen_vae
from scripts.animate_denoiser import load_denoiser, ddim_sample, make_fake_enc

dev = torch.device("cuda")
torch.manual_seed(0)
VAE = "runs/m1_l2_anytop13_edgeseg_C128_rot6dfk_w025f100t010_g128_4card_seed42/best_model.pt"
DEN = "runs/m2_t2m_cleanL2_Bep79rot6dfk_d512C128_n11ff1536_h100x6_seed42/last_model.pt"
SPECIES = ["PZ_Dall_Sheep_Female", "PZ_Galapagos_Giant_Tortoise_Male",
           "PZ_Indian_Elephant_Female", "PZ_Siberian_Tiger_Male"]
N_DDIM = 50
CFG = 1.0

vae, vta = load_frozen_vae(VAE, dev)
denoiser, ck = load_denoiser(DEN, dev)
da = ck.get("args", {})
d_model = vta["d_model"]
stride = vta["temporal_stride"]
sched_kwargs = dict(
    num_train_timesteps=da.get("num_train_timesteps", 1000),
    beta_start=da.get("beta_start", 0.00085), beta_end=da.get("beta_end", 0.012),
    beta_schedule=da.get("beta_schedule", "scaled_linear"),
    prediction_type="v_prediction", clip_sample=False,
)
ds = AnyTopDataset(split="train", random_caption=False, random_crop=False,
                   num_frames=da.get("max_frames", 260), max_joints=144,
                   caption_emb_cache="data/anytop_caption_t5_cleanL2_multi.npz",
                   data_root="data/anytop_planet_zoo_clean_L2")
want = set(SPECIES)
match = [i for i, s in enumerate(ds.samples) if s.get("object_type") in want]


def lat_jitter(z, cm, fm):
    z0 = z[0]; c = cm[0]; f = fm[0]
    vals = []
    for t in range(z0.shape[0] - 1):
        if bool(f[t]) and bool(f[t + 1]):
            d = (z0[t + 1] - z0[t])[c]            # [Cvalid, D]
            vals.append(d.norm(dim=-1).mean().item())
    return float(np.mean(vals)) if vals else 0.0


def mot_jitter(motion, T):
    m = motion[0, :T, :, :3]                       # [T,J,3] normalised pos
    d = (m[1:] - m[:-1]).norm(dim=-1)              # [T-1,J]
    return float(d.mean().item())


print(f"denoiser ep={ck.get('epoch')}  cfg={CFG} ddim={N_DDIM}")
print(f"{'species':32s} {'lat_z0':>8s} {'lat_ẑ0':>8s} {'lat_R':>6s}  "
      f"{'mot_rec':>8s} {'mot_gen':>8s} {'mot_R':>6s}")
picked = {s: 0 for s in want}
agg = {"latR": [], "motR": []}
for i in match:
    item = ds[i]
    sp = item["object_type"]
    if picked.get(sp, 9) >= 1:
        continue
    raw = anytop_collate_fn([item])
    raw = {k: v.to(dev) if torch.is_tensor(v) else v for k, v in raw.items()}
    batch = GraphMotionBatch.from_collate_dict(raw)
    fml = batch.frame_mask.view(1, batch.frame_mask.shape[1] // stride, stride).all(dim=-1)
    with torch.no_grad():
        skel = vae.encode_skeleton_only(batch)
        enc_true = vae.encode(batch, sample=False)
        z0 = enc_true["z"].float()
        zhat = ddim_sample(denoiser, batch, skel, fml, N_DDIM, CFG, sched_kwargs, dev, d_model)
        dec_rec = vae.decode(make_fake_enc(z0, skel, fml), batch)["pred_motion"]
        dec_gen = vae.decode(make_fake_enc(zhat, skel, fml), batch)["pred_motion"]
    cm = skel["coarse_mask"]
    T = int(fml[0].sum().item() * stride)
    lat0 = lat_jitter(z0, cm, fml); lath = lat_jitter(zhat, cm, fml)
    mrec = mot_jitter(dec_rec, T); mgen = mot_jitter(dec_gen, T)
    latR = lath / max(lat0, 1e-9); motR = mgen / max(mrec, 1e-9)
    agg["latR"].append(latR); agg["motR"].append(motR)
    print(f"{sp:32s} {lat0:8.4f} {lath:8.4f} {latR:6.2f}  "
          f"{mrec:8.4f} {mgen:8.4f} {motR:6.2f}")
    picked[sp] += 1

print(f"\nMEAN  latent_ratio={np.mean(agg['latR']):.2f}  "
      f"motion_ratio={np.mean(agg['motR']):.2f}")
print("\nINTERPRETATION:")
print("  latent_ratio ≈ motion_ratio >> 1 → jank IN THE LATENT (diffusion makes jittery ẑ0)")
print("  latent_ratio ≈ 1, motion_ratio >> 1 → ẑ0 smooth but off-manifold → DECODE amplifies")
