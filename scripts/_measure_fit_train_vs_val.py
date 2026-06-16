"""Diagnostic (read-only): is the diffusion UNDERFITTING (train v-MSE ~ val v-MSE)
or GENERALIZING-poorly (train << val)? Replicates train_denoiser's val_denoise
computation EXACTLY (sample=False deterministic z=mu, no CFG dropout, fixed-seed
noise/timesteps, element-weighted mean) and runs it on a random subset of BOTH
the train and val split, plus a per-timestep-bucket breakdown.

  train v-MSE ~ val v-MSE  → model cannot fit train either = UNDERFIT
                             → optimization (const-lr plateau → LR decay worth a try)
                             or capacity ceiling.
  train v-MSE << val v-MSE → generalization gap = DATA SCARCITY
                             → LR decay won't fix val/generation quality.
"""
import random
import sys
import torch
sys.path.insert(0, ".")
from torch.utils.data import DataLoader, Subset
from diffusers import DDIMScheduler
from scripts.train_denoiser import load_frozen_vae
from scripts.animate_denoiser import load_denoiser
from src.data.anytop_dataset import AnyTopDataset, collate_fn as anytop_collate_fn
from src.models.graph_salad.batch import GraphMotionBatch

dev = torch.device("cuda")
VAE = "runs/m1_l2_anytop13_edgeseg_C128_rot6dfk_w025f100t010_g128_4card_seed42/best_model.pt"
DEN = "runs/m2_t2m_cleanL2_Bep79rot6dfk_d512C128_n11ff1536_h100x6_seed42/last_model.pt"
N_CLIPS = 256

vae, vta = load_frozen_vae(VAE, dev)
denoiser, ck = load_denoiser(DEN, dev)
da = ck.get("args", {})
print(f"denoiser ep={ck.get('epoch')} text_mode={da.get('text_mode', 'mean_additive')} "
      f"n_layers={da.get('n_layers')} d_ff={da.get('d_ff')}")
sched = DDIMScheduler(
    num_train_timesteps=da.get("num_train_timesteps", 1000),
    beta_start=da.get("beta_start", 0.00085), beta_end=da.get("beta_end", 0.012),
    beta_schedule=da.get("beta_schedule", "scaled_linear"),
    prediction_type="v_prediction", clip_sample=False,
)
MF = da.get("max_frames", 260)


def make_ds(split):
    return AnyTopDataset(
        split=split, random_caption=False, random_crop=False,
        num_frames=MF, max_joints=144,
        caption_emb_cache="data/anytop_caption_t5_cleanL2_multi.npz",
        data_root="data/anytop_planet_zoo_clean_L2",
    )


@torch.no_grad()
def vmse(ds, tag):
    rng = random.Random(123)
    idxs = rng.sample(range(len(ds)), min(N_CLIPS, len(ds)))
    dl = DataLoader(Subset(ds, idxs), batch_size=8, shuffle=False,
                    num_workers=4, collate_fn=anytop_collate_fn)
    g = torch.Generator(device=dev).manual_seed(42)   # same as val's fixed seed
    num = den = 0.0
    bnum = [0.0] * 5
    bden = [0.0] * 5                                   # 5 timestep buckets of 200
    for raw in dl:
        raw = {k: v.to(dev) if torch.is_tensor(v) else v for k, v in raw.items()}
        batch = GraphMotionBatch.from_collate_dict(raw)
        enc = vae.encode(batch, sample=False)          # deterministic z=mu (val path)
        z0 = enc["z"].float()
        cm = enc["coarse_mask"]; fm = enc["frame_mask_lat"]
        padj = enc["pooled_adjacency"].float()
        pgeo = enc["pooled_geodesic"].float()
        pskel = enc["pooled_skeleton_embeddings"].float()
        B = z0.shape[0]
        noise = torch.randn(z0.shape, generator=g, device=dev, dtype=z0.dtype)
        ts = torch.randint(0, da.get("num_train_timesteps", 1000), (B,),
                           generator=g, device=dev).long()
        z_t = sched.add_noise(z0, noise, ts)
        vtg = sched.get_velocity(z0, noise, ts)
        mask = (cm[:, None, :, None] & fm[:, :, None, None])
        mf = mask.to(z0.dtype)
        z_t = z_t * mf; vtg = vtg * mf
        ht = batch.has_text.to(dev)
        text_in = batch.caption_emb.to(dev) * ht[:, None].to(batch.caption_emb.dtype)
        vp = denoiser(
            z_t=z_t, timesteps=ts, text=text_in, adjacency=padj, geodesic_dist=pgeo,
            coarse_mask=cm, frame_mask=fm, pooled_skeleton_embeddings=pskel,
            has_text=ht, validate_inputs=False, text_token_mask=None,
        )
        d2 = (vp.float() - vtg).pow(2) * mf
        num += d2.sum().item(); den += mf.sum().item() * vp.shape[-1]
        for b in range(B):
            bk = min(int(ts[b].item()) // 200, 4)
            db = ((vp[b].float() - vtg[b]).pow(2) * mf[b]).sum().item()
            bnum[bk] += db; bden[bk] += mf[b].sum().item() * vp.shape[-1]
    overall = num / max(den, 1.0)
    print(f"\n[{tag}] n_clips={len(idxs)}  v-MSE={overall:.4f}")
    for k in range(5):
        if bden[k] > 0:
            print(f"    t[{k*200:4d}-{k*200+199}]  v-MSE={bnum[k]/bden[k]:.4f}")
    return overall


tr = vmse(make_ds("train"), "TRAIN")
va = vmse(make_ds("val"), "VAL")
print("\n========= VERDICT =========")
print(f"TRAIN v-MSE={tr:.4f}   VAL v-MSE={va:.4f}   gap={va - tr:+.4f}")
if abs(va - tr) < 0.03:
    print("→ train ≈ val: UNDERFIT (can't fit train either). Optimization (const-lr")
    print("  plateau → LR decay worth a try) or capacity ceiling — NOT data scarcity.")
elif tr < va - 0.03:
    print("→ train << val: GENERALIZATION gap (data scarcity). LR decay won't fix val.")
