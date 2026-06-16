"""One-shot SANITY probe: Graph-VQVAE long-frame (>64) reconstruction.

Read-only. Verifies the frozen 64-frame-trained Graph-VQVAE (ep280) can
tokenize + decode near-full-length clips (num_frames=300 -> T_lat up to ~73),
since the backbone will train on full-length motion (T_fine_max=300, stride 4,
T_lat_max=75) and the existing token cache is only 64-frame (T_lat=16).

NOT a model/training/anchor change. Reuses the AUDITED helpers from
animate_vqvae_recon.py (load_vq_tokenizer, official rot6d FK, animate_clip) —
no hand-rolled FK. Picks the N longest real clips and renders GT-vs-recon GIFs.
"""

import argparse
import glob
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import numpy as np  # noqa: E402
import torch  # noqa: E402

project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from scripts.animate_vqvae_recon import load_vq_tokenizer  # noqa: E402  audited loader
from scripts.animate import animate_clip  # noqa: E402  model-agnostic renderer
from src.data.anytop_dataset import (  # noqa: E402
    AnyTopDataset, collate_fn, _recover_world_positions, _STD_FLOOR,
)
from src.data.anytop_rot6d_fk import recover_from_bvh_rot_np  # noqa: E402  official FK
from src.models.graph_salad.batch import GraphMotionBatch  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--num_frames", type=int, default=300,
                    help="override dataset crop/pad target (trained was 64)")
    ap.add_argument("--n", type=int, default=4, help="render N longest clips")
    ap.add_argument("--split", default="all")
    ap.add_argument("--stride", type=int, default=3, help="gif frame subsample")
    ap.add_argument("--fps", type=int, default=10)
    ap.add_argument("--anytop_root", default=None)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    dev = torch.device(args.device if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    model, ta, ck = load_vq_tokenizer(args.ckpt, dev)
    root = args.anytop_root or ta.get("anytop_root") or ck.get("data_root")
    stride = ta["temporal_stride"]
    print(f"VQVAE ep{ck.get('epoch')} stride={stride} trained_max_frames={ta.get('max_frames')} "
          f"| num_frames(override)={args.num_frames}")

    ds = AnyTopDataset(
        split=args.split, num_frames=args.num_frames,
        max_joints=ta.get("max_joints", 64), val_frac=ta.get("val_frac", 0.05),
        seed=ta.get("seed", 42), load_captions=False, data_root=root,
    )
    print(f"dataset split={args.split} num_frames={args.num_frames} -> {len(ds)} clips")

    # ---- pre-scan raw lengths via mmap (header-only, fast) to pick longest ----
    name2len = {}
    for f in glob.glob(os.path.join(root, "motions", "*.npy")):
        try:
            name2len[os.path.basename(f)[:-4]] = int(
                np.load(f, mmap_mode="r", allow_pickle=True).shape[0])
        except Exception:
            pass
    matched = 0
    best_per_sp = {}  # species -> (L, idx): one longest clip per species (dedupe + diversity)
    for i, s in enumerate(ds.samples):
        mid = s.get("motion_id") or s.get("name") or ""
        L = name2len.get(mid, -1)
        if L > 0:
            matched += 1
        sp = s.get("object_type", "?")
        if L > best_per_sp.get(sp, (-1, -1))[0]:
            best_per_sp[sp] = (L, i)
    print(f"raw-length match: {matched}/{len(ds.samples)} samples; {len(best_per_sp)} species")
    if matched == 0:
        raise SystemExit("length match failed — inspect ds.samples[0] keys")
    ranked = sorted(best_per_sp.values(), reverse=True)  # cross-species longest clips
    picks = [(i, L) for L, i in ranked[:args.n] if L > 64]
    if not picks:
        raise SystemExit("no clips with raw length > 64 found")

    amp = (ta.get("amp_dtype", "bf16") == "bf16") and dev.type == "cuda"
    summary = []
    with torch.no_grad():
        for idx, rawL in picks:
            item = ds[idx]
            raw = collate_fn([item])
            raw = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in raw.items()}
            batch = GraphMotionBatch.from_collate_dict(raw)
            if amp:
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    out = model(batch, allow_collectives=False)
            else:
                out = model(batch, allow_collectives=False)

            J = int(item["num_joints"])
            T_clip = int(item["num_frames"])
            T_valid = int(out["frame_mask_recovered"][0].sum().item())
            T = min(T_clip, T_valid)
            T_lat = -(-T // stride)  # ceil

            std = raw["anytop_std"][0, :J].cpu().numpy()
            mean = raw["anytop_mean"][0, :J].cpu().numpy()
            pred_norm = out["pred_motion"][0, :T, :J, :].float().cpu().numpy()
            pred_raw = pred_norm * (std[None] + _STD_FLOOR) + mean[None]
            gt_norm = np.asarray(item["anytop_x"]).transpose(2, 0, 1)[:T, :J, :]
            gt_raw = gt_norm * (std[None] + _STD_FLOOR) + mean[None]

            parents = [int(p) for p in item["parent_indices"][:J]]
            offsets = np.asarray(item["rest_offsets"])[:J]
            pred_world = recover_from_bvh_rot_np(pred_raw, parents, offsets)
            gt_world = recover_from_bvh_rot_np(gt_raw, parents, offsets)

            gt_ric = _recover_world_positions(gt_raw)
            sc = float(np.linalg.norm(gt_world - gt_ric, axis=-1).mean())  # renderer self-check ~0
            l2 = float(np.linalg.norm(pred_world - gt_world, axis=-1).mean())
            g = float(np.linalg.norm(np.diff(gt_world, axis=0), axis=-1).mean())
            p = float(np.linalg.norm(np.diff(pred_world, axis=0), axis=-1).mean())
            ratio = p / max(g, 1e-9)

            sp = item["object_type"]
            ttl = (f"{sp} [VQVAE ep{ck.get('epoch')}] J={J} T={T} Tlat={T_lat} rawL={rawL} "
                   f"reconL2={l2:.3f} spd={ratio:.2f}")
            gif = out_dir / f"long_{T:03d}f_{sp}.gif"
            animate_clip(pred_world, gt_world, parents, str(gif), ttl, args.stride, args.fps)
            line = (f"{sp} T={T} Tlat={T_lat} rawL={rawL} reconL2={l2:.4f} "
                    f"GTselfcheck={sc:.2e} spd_ratio={ratio:.3f} -> {gif.name}")
            print(line)
            summary.append(line)

    (out_dir / "longframe_summary.txt").write_text("\n".join(summary) + "\n")
    print(f"DONE {len(picks)} long-frame recon gifs -> {out_dir}")


if __name__ == "__main__":
    main()
