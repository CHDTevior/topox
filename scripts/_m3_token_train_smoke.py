"""M3 token-train integration smoke (+ item G bf16). NOT a launch script — a
focused 5-iter check that mirrors train_denoiser.py's token code path on a few
real motions, verifying: finite loss, grads reach text_token_proj + cross-attn,
in BOTH fp32 and bf16 (item G = bf16 diffusion's first real run).

Run as a MANAGED srun step (never bare-ssh a GPU process):
  srun --jobid=<A100_JOB> --overlap --gres=gpu:1 --cpus-per-task=4 \
    python scripts/_m3_token_train_smoke.py
"""
from __future__ import annotations
import contextlib
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.anytop_dataset import AnyTopDataset, collate_fn as anytop_collate_fn
from src.models.graph_salad.batch import GraphMotionBatch
from src.models.graph_salad.denoiser import GraphSaladDenoiser
from scripts.train_denoiser import load_frozen_vae, masked_v_mse
from diffusers import DDIMScheduler

VAE_CKPT = "runs/m1_l2_anytop13_edgeseg_C128_rot6dfk_w025f100t010_g128_4card_seed42/best_model.pt"
ANYTOP_ROOT = "data/anytop_planet_zoo_clean_L2"
MEAN_CACHE = str(ROOT / ".aris/smoke_tok/m3mean.npz")
TOK_CACHE = str(ROOT / ".aris/smoke_tok/m3tok")
MIDS_JSON = str(ROOT / ".aris/smoke_tok/m3_mids.json")


def run(amp_dtype: str, dev: torch.device) -> bool:
    print(f"\n===== M3 token-train smoke (amp={amp_dtype}) =====")
    vae, ta = load_frozen_vae(VAE_CKPT, dev)
    d_model = ta["d_model"]; n_heads = ta["n_heads"]
    mids = set(json.load(open(MIDS_JSON)))
    ds = AnyTopDataset(
        data_root=ANYTOP_ROOT, split="all", num_frames=260,
        max_joints=ta.get("max_joints", 144), caption_emb_cache=MEAN_CACHE,
        random_caption=True, random_crop=False,
        caption_token_cache=TOK_CACHE, return_caption_tokens=True,
        caption_token_max_len=64,
    )
    ds.samples = [s for s in ds.samples if s["motion_id"] in mids]
    print(f"  smoke dataset: {len(ds.samples)} motions")

    denoiser = GraphSaladDenoiser(
        d_model=d_model, n_heads=n_heads, d_ff=1536, n_layers=11, d_text=768,
        dropout=0.1, text_mode="token_cross_attn", text_token_dim=768,
    ).to(dev)
    denoiser.train()
    opt = torch.optim.AdamW(denoiser.parameters(), lr=5e-4)
    sched = DDIMScheduler(num_train_timesteps=1000, beta_start=0.00085,
                          beta_end=0.012, beta_schedule="scaled_linear",
                          prediction_type="v_prediction", clip_sample=False)
    amp_enabled = (amp_dtype == "bf16")
    amp_ctx = ((lambda: torch.autocast(device_type="cuda", dtype=torch.bfloat16))
               if amp_enabled else contextlib.nullcontext)

    from torch.utils.data import DataLoader
    dl = DataLoader(ds, batch_size=4, shuffle=True, collate_fn=anytop_collate_fn,
                    num_workers=0, drop_last=True)
    losses = []
    grad_seen = {"text_token_proj": False, "text_cross_attn": False, "text_proj_unused": True}
    it = 0
    for raw in dl:
        if it >= 5:
            break
        raw = {k: v.to(dev) if torch.is_tensor(v) else v for k, v in raw.items()}
        batch = GraphMotionBatch.from_collate_dict(raw)
        with torch.no_grad(), amp_ctx():
            enc = vae.encode(batch, sample=True)
        z0 = enc["z"].float()
        pooled_adj = enc["pooled_adjacency"].float()
        pooled_geo = enc["pooled_geodesic"].float()
        coarse_mask = enc["coarse_mask"]; frame_mask = enc["frame_mask_lat"]
        pooled_skel = enc["pooled_skeleton_embeddings"].float()
        B = z0.shape[0]
        ht_in = batch.has_text.to(dev)
        drop = torch.rand(B, device=dev) < 0.1
        has_text = ht_in & (~drop)
        text_in = batch.caption_token_emb.to(dev)
        token_mask_in = batch.caption_token_mask.to(dev)
        noise = torch.randn_like(z0)
        ts = torch.randint(0, 1000, (B,), device=dev).long()
        z_t = sched.add_noise(z0, noise, ts)
        v_target = sched.get_velocity(z0, noise, ts)
        m4 = (coarse_mask[:, None, :, None] & frame_mask[:, :, None, None]).to(z0.dtype)
        z_t = z_t * m4; v_target = v_target * m4
        with amp_ctx():
            v_pred = denoiser(
                z_t=z_t, timesteps=ts, text=text_in,
                adjacency=pooled_adj, geodesic_dist=pooled_geo,
                coarse_mask=coarse_mask, frame_mask=frame_mask,
                pooled_skeleton_embeddings=pooled_skel, has_text=has_text,
                text_token_mask=token_mask_in, validate_inputs=(it == 0),
            )
            loss = masked_v_mse(v_pred, v_target, coarse_mask, frame_mask)
        if not torch.isfinite(loss):
            print(f"  [FAIL] non-finite loss at it={it}")
            return False
        opt.zero_grad()
        loss.backward()
        # grad reach check (token-mode params)
        if denoiser.text_token_proj.weight.grad is not None and \
                denoiser.text_token_proj.weight.grad.abs().sum() > 0:
            grad_seen["text_token_proj"] = True
        for lyr in denoiser.layers:
            g = lyr.text_cross_attn.q_proj.weight.grad
            if g is not None and g.abs().sum() > 0:
                grad_seen["text_cross_attn"] = True
                break
        torch.nn.utils.clip_grad_norm_(denoiser.parameters(), 1.0)
        opt.step()
        losses.append(loss.item())
        print(f"  it={it} loss={loss.item():.4f} "
              f"v_pred.dtype={v_pred.dtype} finite={torch.isfinite(v_pred).all().item()}")
        it += 1

    ok = (len(losses) == 5 and all(l == l for l in losses)
          and grad_seen["text_token_proj"] and grad_seen["text_cross_attn"])
    print(f"  losses={['%.4f' % l for l in losses]}")
    print(f"  grad reached text_token_proj={grad_seen['text_token_proj']} "
          f"text_cross_attn={grad_seen['text_cross_attn']}")
    print(f"  {'PASS' if ok else 'FAIL'} (amp={amp_dtype})")
    return ok


def main() -> int:
    if not torch.cuda.is_available():
        print("[M3 smoke] CUDA required (VAE encode); run as a managed srun step.")
        return 2
    dev = torch.device("cuda")
    torch.manual_seed(42)
    ok_fp32 = run("fp32", dev)
    ok_bf16 = run("bf16", dev)   # item G
    print(f"\n==== M3 SUMMARY: fp32={'PASS' if ok_fp32 else 'FAIL'} "
          f"bf16={'PASS' if ok_bf16 else 'FAIL'} ====")
    return 0 if (ok_fp32 and ok_bf16) else 1


if __name__ == "__main__":
    sys.exit(main())
