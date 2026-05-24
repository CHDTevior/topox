"""One-shot custom-prompt T2M demo. Picks a Dragon skeleton sample, encodes
arbitrary prompt via T5-base (same model + pooling as precompute_t5_captions.py),
runs DDIM 50-step CFG=7.5 sampling, renders gif via animate_t2m_input_pred.

Throwaway script — not for repo growth. Reproduces minimum slice of
animate_denoiser.py with caption embedding replaced by an inline-encoded one.
"""
from pathlib import Path
import sys
import numpy as np
import torch

ROOT = Path("/scratch/ts1v23/workspace/noKslot_clean")
sys.path.insert(0, str(ROOT))

from scripts.animate_denoiser import (
    load_frozen_vae, load_denoiser, ddim_sample, make_fake_enc, fk_rest_pose,
    animate_t2m_input_pred,
)
from src.data.anytop_dataset import (
    AnyTopDataset, collate_fn as anytop_collate_fn, _recover_world_positions,
    _STD_FLOOR,
)
from src.models.graph_salad.batch import GraphMotionBatch


def main() -> int:
    import os
    PROMPT = os.environ.get("PROMPT", "An animal energetically bucks, jumping and kicking its hind legs high into the air.")
    SPECIES = os.environ.get("SPECIES", "Horse")
    OUT_TAG = os.environ.get("OUT_TAG", "out")
    VAE_CKPT = ROOT / "runs/m1_7_anytop13_edge_segment_C96_fulldata_ddp2a100_seed42/last_model.pt"
    DEN_CKPT = ROOT / "runs/m2_denoiser_v2_edge_segment_C96_seed42/last_model.pt"
    CAP_CACHE = ROOT / "data/anytop_caption_t5_1070_multi.npz"
    OUT_DIR = ROOT / "runs/m2_denoiser_v2_edge_segment_C96_seed42/qa_custom_prompt"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    SEED = 42
    N_DDIM = 50
    CFG = 7.5
    STRIDE = 2
    FPS = 8
    T5_NAME = "t5-base"

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(SEED); np.random.seed(SEED)
    if dev.type == "cuda":
        torch.cuda.manual_seed_all(SEED)

    print(f"Loading VAE: {VAE_CKPT}")
    vae, ta = load_frozen_vae(str(VAE_CKPT), dev)
    d_model = ta["d_model"]; temporal_stride = ta["temporal_stride"]

    print(f"Loading denoiser: {DEN_CKPT}")
    denoiser, dck = load_denoiser(str(DEN_CKPT), dev)
    da = dck.get("args", {})

    sched_kwargs = dict(
        num_train_timesteps=da.get("num_train_timesteps", 1000),
        beta_start=da.get("beta_start", 0.00085),
        beta_end=da.get("beta_end", 0.012),
        beta_schedule=da.get("beta_schedule", "scaled_linear"),
        prediction_type="v_prediction",
        clip_sample=False,
    )

    print(f"Encoding prompt via {T5_NAME}: {PROMPT!r}")
    from transformers import T5EncoderModel, T5TokenizerFast
    tok = T5TokenizerFast.from_pretrained(T5_NAME)
    t5 = T5EncoderModel.from_pretrained(T5_NAME).to(dev).eval()
    enc = tok(PROMPT, return_tensors="pt", padding=True, truncation=True, max_length=128).to(dev)
    with torch.no_grad():
        out = t5(input_ids=enc.input_ids, attention_mask=enc.attention_mask).last_hidden_state
    mask = enc.attention_mask.unsqueeze(-1).float()
    pooled = (out * mask).sum(1) / mask.sum(1).clamp_min(1)
    text_emb_custom = pooled[0].to(dev)
    print(f"  prompt emb shape={tuple(text_emb_custom.shape)} norm={text_emb_custom.norm().item():.3f}")

    print("Loading Dragon sample for skeleton conditioning")
    ds_kwargs = dict(
        split="val",
        num_frames=ta.get("max_frames", 64),
        max_joints=ta.get("max_joints", 143),
        caption_emb_cache=str(CAP_CACHE),
    )
    if ta.get("anytop_root"):
        ds_kwargs["data_root"] = ta["anytop_root"]
    ds = AnyTopDataset(**ds_kwargs)
    dragon_idx = None
    for i in range(len(ds)):
        if ds[i]["object_type"] == SPECIES:
            dragon_idx = i; break
    if dragon_idx is None:
        raise SystemExit(f"No {SPECIES} sample found in val split")
    item = ds[dragon_idx]
    print(f"  picked {SPECIES} sample idx={dragon_idx} J={item['num_joints']} T={item['num_frames']}")

    raw = anytop_collate_fn([item])
    raw = {k: v.to(dev) if torch.is_tensor(v) else v for k, v in raw.items()}
    batch = GraphMotionBatch.from_collate_dict(raw)

    # Replace caption_emb with our custom prompt (in-place via tensor assign)
    batch.caption_emb = text_emb_custom.unsqueeze(0)  # [1, 768]
    batch.has_text = torch.tensor([True], device=dev)

    with torch.no_grad():
        skel = vae.encode_skeleton_only(batch)
    frame_mask_lat = batch.frame_mask.view(
        1, batch.frame_mask.shape[1] // temporal_stride, temporal_stride
    ).all(dim=-1)

    print(f"DDIM sample: {N_DDIM} steps, CFG={CFG}")
    z = ddim_sample(denoiser, batch, skel, frame_mask_lat,
                    n_steps=N_DDIM, cond_scale=CFG, sched_kwargs=sched_kwargs,
                    dev=dev, d_model=d_model)

    fake_enc = make_fake_enc(z, skel, frame_mask_lat)
    with torch.no_grad():
        dec = vae.decode(fake_enc, batch)
    pred_motion = dec["pred_motion"]

    J = int(item["num_joints"])
    T_clip = int(item["num_frames"])
    T_valid = int(frame_mask_lat[0].sum().item() * temporal_stride)
    T = min(T_clip, T_valid)
    std = raw["anytop_std"][0, :J].cpu().numpy()
    mean = raw["anytop_mean"][0, :J].cpu().numpy()
    pred_norm = pred_motion[0, :T, :J, :].cpu().numpy()
    pred_raw = pred_norm * (std[None] + _STD_FLOOR) + mean[None]
    pred_world = _recover_world_positions(pred_raw)
    parents = [int(p) for p in item["parent_indices"][:J]]

    rest_off = raw["rest_offsets"][0, :J].cpu().numpy()
    static_pose = fk_rest_pose(rest_off, parents)
    gif_path = OUT_DIR / f"{SPECIES}_{OUT_TAG}.gif"
    p_spd = float(np.linalg.norm(np.diff(pred_world, axis=0), axis=-1).mean())
    skel_label = (
        f"{SPECIES} skeleton (J={J})\n"
        f"T={T}  cfg={CFG} steps={N_DDIM}\n"
        f"pred_speed={p_spd:.4f}"
    )
    animate_t2m_input_pred(
        pred_world, static_pose, parents, str(gif_path),
        prompt_text=PROMPT, stride=STRIDE, fps=FPS,
        skeleton_label=skel_label,
    )
    print(f"DONE -> {gif_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
