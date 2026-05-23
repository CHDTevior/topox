"""[C] encode_skeleton_only structural + [D] ckpt re-eval, with EXACT production
config (max_coarse=64, attn_mode=graphormer, use_name_embed=True, etc.).

Run with CUDA_VISIBLE_DEVICES=2 on swarma1003.
"""
import os, sys
import torch

REPO = "/scratch/ts1v23/workspace/noKslot_clean"
os.chdir(REPO)
sys.path.insert(0, REPO)
torch.cuda.set_device(0)
device = torch.device("cuda")

from src.data.anytop_dataset import AnyTopDataset, collate_fn as anytop_collate_fn
from src.models.graph_salad.batch import GraphMotionBatch
from src.models.graph_salad.vae import GraphMotionVAE
from src.models.graph_salad.losses import compute_total_loss_13ch
from torch.utils.data import DataLoader
import numpy as np
from collections import defaultdict

# Production loss weights (from runs/m1_7_anytop13_coarse_xattn_seed42/train.log)
LOSS_WEIGHTS = dict(
    pos=1.0, vel=1.0, rot=1.0, contact=0.1,
    vel_normalized=0.0, vel_consistency=0.5, speed_mag=0.0,
    kl=0.001, bone=1.0, pool_aux=0.5,
)

def make_vae(decoder_mode):
    vae = GraphMotionVAE(
        pool_type="dynamic", pool_tau=None,
        d_model=384, n_heads=8, d_ff=1024,
        n_graph_layers=4, n_enc_temporal_layers=2, n_cross_layers=3,
        n_dec_temporal_layers=2, n_treeik_layers=3,
        max_coarse=64, local_radius=8,
        temporal_stride=4, temporal_kernel=9,
        dropout=0.1,
        motion_feat_dim=13, feat_mode="anytop13",
        attn_mode="graphormer",
        use_text=False,
        decoder_mode=decoder_mode,
        n_graph_temporal_layers=4,
    ).to(device).eval()
    vae.encoder.use_name_embed = True
    return vae

# ---------- [C] encode_skeleton_only ----------
print("=== [C] encode_skeleton_only structural check (anytop13 + dynamic pool, prod cfg) ===")
ds_at = AnyTopDataset(split="train", num_frames=64, max_joints=143)
items = [ds_at[i] for i in range(2)]
batch_dict = anytop_collate_fn(items)
raw = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch_dict.items()}
batch = GraphMotionBatch.from_collate_dict(raw)

vae = make_vae("coarse_xattn")

with torch.no_grad():
    skel_out = vae.encode_skeleton_only(batch)
    enc_out = vae.encode(batch, sample=False)

print(f"  B={batch.adjacency.shape[0]} J={batch.adjacency.shape[1]} C={vae.pool.max_coarse}")
for k, v in skel_out.items():
    print(f"    {k}: shape={tuple(v.shape)} dtype={v.dtype}")

all_ok = True
print("\n  Comparison vs encode():")
for k in ["pooled_adjacency", "pooled_geodesic", "coarse_mask",
          "pooled_skeleton_embeddings", "anchor_indices", "hard_assignment"]:
    a = skel_out[k]; b = enc_out[k]
    eq = torch.equal(a, b)
    if eq:
        print(f"    {k}: equal=True  [OK]")
    else:
        if a.dtype.is_floating_point:
            mx = (a - b).abs().max().item()
            print(f"    {k}: equal=False max_abs_diff={mx:.3e}")
            if mx > 1e-6:
                all_ok = False
        else:
            print(f"    {k}: equal=False (integer)")
            all_ok = False
if not all_ok:
    print("  [FAIL]"); sys.exit(4)
print("  [OK] all Phase-2 keys match encode() byte-for-byte\n")

# ---------- [D] ckpt re-eval ----------
print("=== [D] strict-load + re-eval ckpts (deterministic val sweep) ===")
ds_val = AnyTopDataset(split="val", num_frames=64, max_joints=143)

def reval(name, ckpt_path, decoder_mode, expected_total, expected_recon, B=16):
    print(f"\n  ---- {name} ----")
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    print(f"    saved: epoch={int(ck.get('epoch',-1))} val_loss={float(ck.get('val_loss',float('nan'))):.4f} val_recon={float(ck.get('val_recon',float('nan'))):.4f}")
    m = make_vae(decoder_mode)
    # ckpts saved without DDP wrapper (according to memory: DDP unwrap done at save)
    sd = ck["model_state_dict"]
    # Strip 'module.' prefix if present (defensive)
    if any(k.startswith("module.") for k in sd):
        sd = {k.removeprefix("module."): v for k, v in sd.items()}
    missing, unexpected = m.load_state_dict(sd, strict=True)
    if missing or unexpected:
        print(f"    LOAD ERROR: missing={len(missing)} unexpected={len(unexpected)}")
        if missing[:5]: print(f"      missing sample: {missing[:5]}")
        if unexpected[:5]: print(f"      unexpected sample: {unexpected[:5]}")
        return False
    print(f"    strict-load: missing=0 unexpected=0")
    dl = DataLoader(ds_val, batch_size=B, shuffle=False, collate_fn=anytop_collate_fn, num_workers=0)
    val_losses = defaultdict(list)
    with torch.no_grad():
        for bd in dl:
            raw_v = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in bd.items()}
            bv = GraphMotionBatch.from_collate_dict(raw_v)
            out = m(bv, sample=False)
            efm = out["frame_mask_recovered"]
            gt_motion = bv.anytop_x.permute(0, 3, 1, 2).contiguous()
            losses = compute_total_loss_13ch(
                pred_motion=out["pred_motion"], gt_motion=gt_motion,
                foot_contact_per_joint=bv.foot_contact_per_joint,
                mu=out["mu"], logvar=out["logvar"],
                pool_aux_outputs=out["pool_aux_outputs"],
                joint_mask=bv.joint_mask, frame_mask=efm,
                coarse_mask=out["coarse_mask"], frame_mask_lat=out["frame_mask_lat"],
                weights=LOSS_WEIGHTS,
            )
            for k, v in losses.items():
                val_losses[k].append(v.item())
    avg_t = float(np.mean(val_losses["total"]))
    recon_keys = ("pos", "rot", "vel", "contact")
    raw_means = {k: float(np.mean(val_losses[k])) for k in recon_keys if k in val_losses}
    avg_r = sum(LOSS_WEIGHTS[k] * v for k, v in raw_means.items())
    dt = avg_t - expected_total; dr = avg_r - expected_recon
    print(f"    RE-EVAL: val_total={avg_t:.4f} (saved {expected_total:.4f}, dt={dt:+.5f})")
    print(f"             val_recon={avg_r:.4f} (saved {expected_recon:.4f}, dr={dr:+.5f})")
    ok = abs(dt) < 1e-3 and abs(dr) < 1e-3
    print(f"             [{'OK MATCH' if ok else 'MISMATCH'}]")
    return ok

all_ok = True
all_ok &= reval("coarse_xattn ep829",
                "runs/m1_7_anytop13_coarse_xattn_seed42/best_recon_model.pt",
                "coarse_xattn", 2.4822, 2.0442)
all_ok &= reval("graph_temporal DDP ep299",
                "runs/m1_7_anytop13_graph_temporal_ddp2h100_seed42/best_recon_model.pt",
                "graph_temporal", 2.6767, 2.2781)
print(f"\n=== SUMMARY === all_match={all_ok}")
sys.exit(0 if all_ok else 5)
