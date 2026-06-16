"""场景2: cross-skeleton 文本控制 T2M demo (dual-path 三栏).

每个配对 (src_species, tgt_species): 取 src 物种某 train clip 的真实 caption 作 prompt,
配 tgt 相似物种(不同拓扑)的 skeleton → DDIM 生成动作 → pose + rot6d-FK 两路恢复 →
animate_t2m_input_pred 三栏(input skel + PRED_pose + PRED_FK)。

验证: 文本控制(src 动作描述)能否驱动 tgt 物种骨架生成合理动作 = 拓扑迁移 + 文本控制
(TopoSlots 核心目标)。用 old diffusion (cont_swarma1004 best 0.3721 + baseline VAE ep34)。

复用 animate_denoiser 的 load/ddim_sample/decode/dual-path 渲染 (不重写)。
"""
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path("/scratch/ts1v23/workspace/noKslot_clean")
sys.path.insert(0, str(ROOT))

from scripts.animate_denoiser import (
    load_frozen_vae, load_denoiser, ddim_sample, make_fake_enc, fk_rest_pose,
    animate_t2m_input_pred,
)
from src.data.anytop_dataset import (
    AnyTopDataset, collate_fn as anytop_collate_fn,
    _recover_world_positions, _STD_FLOOR,
)
from src.models.graph_salad.batch import GraphMotionBatch
from src.models.graph_salad.rot6d_fk_recovery import recover_rot6d_fk_positions_torch

# (src_species 提供 train caption/prompt, tgt_species 提供 skeleton). 相似拓扑配对.
PAIRS = [
    ("PZ_Jaguar_Male",      "PZ_Clouded_Leopard_Male"),          # 猫科四足
    ("PZ_Grey_Seal_Male",   "PZ_California_Sea_Lion_Juvenile"),  # 鳍足水生
    ("PZ_Giant_Otter_Male", "PZ_Honey_Badger_Male"),             # 鼬科
    ("PZ_Maned_Wolf_Female","PZ_Japanese_Raccoon_Dog_Female"),   # 犬科
]
VAE_CKPT = ROOT / "runs/_baseline_cleanL2_ep34_for_p1diag_compare/best_recon_model.pt"
DEN_CKPT = ROOT / "runs/m2_t2m_cleanL2_cont_swarma1004/best_model.pt"
CAP_CACHE = ROOT / "data/anytop_caption_t5_cleanL2_multi.npz"
ANYTOP_ROOT = ROOT / "data/anytop_planet_zoo_clean_L2"
OUT_DIR = ROOT / "runs/m2_t2m_cleanL2_cont_swarma1004/qa_cross_skeleton"
CFG = 7.5
N_DDIM = 50
STRIDE = 2
FPS = 8
SEED = 42
T5_NAME = "t5-base"


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    if dev.type == "cuda":
        torch.cuda.manual_seed_all(SEED)

    print(f"Loading VAE: {VAE_CKPT}")
    vae, ta = load_frozen_vae(str(VAE_CKPT), dev)
    d_model = ta["d_model"]
    temporal_stride = ta["temporal_stride"]
    print(f"Loading denoiser: {DEN_CKPT}")
    denoiser, dck = load_denoiser(str(DEN_CKPT), dev)
    da = dck.get("args", {})
    denoiser_max_frames = da.get("max_frames", 64)
    sched_kwargs = dict(
        num_train_timesteps=da.get("num_train_timesteps", 1000),
        beta_start=da.get("beta_start", 0.00085),
        beta_end=da.get("beta_end", 0.012),
        beta_schedule=da.get("beta_schedule", "scaled_linear"),
        prediction_type="v_prediction", clip_sample=False,
    )

    # train split: src caption + tgt skeleton 都从 train 取
    ds = AnyTopDataset(
        split="train", num_frames=denoiser_max_frames,
        max_joints=ta.get("max_joints", 143), caption_emb_cache=str(CAP_CACHE),
        data_root=str(ANYTOP_ROOT),
    )
    # species → first sample index (ds.samples carries object_type; no materialize)
    sp2idx: dict[str, int] = {}
    for i, s in enumerate(ds.samples):
        sp2idx.setdefault(s["object_type"], i)

    # NOTE: compute nodes have NO internet (can't download T5-base). Use the SRC
    # clip's CACHED caption_emb directly (precomputed by precompute_t5_captions.py
    # with the same T5-base + mean-pool) — codex fix-B, also more cache-consistent
    # than re-encoding inline. src_item["caption_emb"] and ["caption"] come from the
    # SAME randomly-picked cap (anytop_dataset __getitem__ picks one cap idx).
    n_done = 0
    for src_sp, tgt_sp in PAIRS:
        if src_sp not in sp2idx or tgt_sp not in sp2idx:
            print(f"SKIP {src_sp}->{tgt_sp}: species missing in train split")
            continue
        # src prompt: real train caption of the SOURCE species' clip (text + cached emb)
        src_item = ds[sp2idx[src_sp]]
        prompt = src_item.get("caption") or ""
        src_cap_emb = src_item.get("caption_emb")
        src_has_text = bool(src_item.get("has_text", False))
        if not prompt or src_cap_emb is None or not src_has_text:
            print(f"SKIP {src_sp}->{tgt_sp}: src clip missing caption/emb/has_text (cache miss?)")
            continue
        # tgt skeleton: a DIFFERENT (similar) species' clip provides the skeleton
        tgt_item = ds[sp2idx[tgt_sp]]
        # SRC caption embedding from CACHE (NOT inline T5 — compute node is offline)
        text_emb = torch.as_tensor(src_cap_emb).float().to(dev)

        raw = anytop_collate_fn([tgt_item])
        raw = {k: v.to(dev) if torch.is_tensor(v) else v for k, v in raw.items()}
        batch = GraphMotionBatch.from_collate_dict(raw)
        # CROSS: tgt skeleton conditioning + src-action prompt (replace caption_emb)
        batch.caption_emb = text_emb.unsqueeze(0)            # [1,768]
        batch.has_text = torch.tensor([True], device=dev)

        with torch.no_grad():
            skel = vae.encode_skeleton_only(batch)
        frame_mask_lat = batch.frame_mask.view(
            1, batch.frame_mask.shape[1] // temporal_stride, temporal_stride
        ).all(dim=-1)

        z = ddim_sample(denoiser, batch, skel, frame_mask_lat,
                        n_steps=N_DDIM, cond_scale=CFG, sched_kwargs=sched_kwargs,
                        dev=dev, d_model=d_model)
        fake_enc = make_fake_enc(z, skel, frame_mask_lat)
        with torch.no_grad():
            dec = vae.decode(fake_enc, batch)
        pred_motion = dec["pred_motion"]                     # [1,T,J,13]

        # length = TARGET skeleton clip's own length (stride-aware), since the
        # generated motion lives on the tgt skeleton.
        J = int(tgt_item["num_joints"])
        T_clip = int(tgt_item["num_frames"])
        T_valid = int(frame_mask_lat[0].sum().item() * temporal_stride)
        T = min(T_clip, T_valid)
        std = raw["anytop_std"][0, :J].cpu().numpy()
        mean = raw["anytop_mean"][0, :J].cpu().numpy()
        pred_norm = pred_motion[0, :T, :J, :].cpu().numpy()
        pred_raw = pred_norm * (std[None] + _STD_FLOOR) + mean[None]
        pred_world = _recover_world_positions(pred_raw)      # pose/RIC (ch0:3)
        parents = [int(p) for p in tgt_item["parent_indices"][:J]]
        rest_off = raw["rest_offsets"][0, :J].cpu().numpy()

        # rot6d-FK route (ch3:9), same de-normalized pred_raw
        pred_raw_t = torch.from_numpy(pred_raw).float()[None]
        rest_off_t = torch.from_numpy(rest_off).float()[None]
        jmask_t = torch.ones(1, J, dtype=torch.bool)
        pred_world_fk = recover_rot6d_fk_positions_torch(
            pred_raw_t, [parents], rest_off_t, jmask_t
        )[0].cpu().numpy()

        static_pose = fk_rest_pose(rest_off, parents)
        p_spd = float(np.linalg.norm(np.diff(pred_world, axis=0), axis=-1).mean())
        pfk_spd = float(np.linalg.norm(np.diff(pred_world_fk, axis=0), axis=-1).mean())
        gif_path = OUT_DIR / f"{src_sp}_PROMPT_on_{tgt_sp}_SKEL.gif"
        skel_label = (
            f"SKEL = {tgt_sp} (J={J})\n"
            f"PROMPT from = {src_sp}\n"
            f"T={T}  cfg={CFG}  pose_spd={p_spd:.3f} fk_spd={pfk_spd:.3f}"
        )
        animate_t2m_input_pred(
            pred_world, static_pose, parents, str(gif_path),
            prompt_text=prompt, stride=STRIDE, fps=FPS,
            skeleton_label=skel_label, pred_fk=pred_world_fk,
        )
        print(f"{src_sp}->{tgt_sp}: J={J} T={T} prompt={prompt[:55]!r} "
              f"pose_spd={p_spd:.4f} fk_spd={pfk_spd:.4f} -> {gif_path.name}")
        n_done += 1

    print(f"DONE {n_done}/{len(PAIRS)} cross-skeleton gifs -> {OUT_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
