"""Calibration (plan §6): on the frozen baseline VAE, over several real batches,
record base_total + each base term + world/fk/traj and their weighted
contributions at w_world=0.25, w_fk=0.25, w_traj=0.10. Apply plan §6 rule to
recommend weights so w_world*world ~= w_fk*fk and geometry not >60% / <10% of base.

Run on rose11: python scripts/_calibration_world_rot6d_fk.py
"""
import sys
import importlib.util
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data.anytop_dataset import AnyTopDataset, collate_fn  # noqa
from src.models.graph_salad import GraphMotionBatch, GraphMotionVAE  # noqa
spec = importlib.util.spec_from_file_location("tgv", str(ROOT / "scripts" / "train_graph_vae.py"))
tgv = importlib.util.module_from_spec(spec); spec.loader.exec_module(tgv)
run_loss = tgv.run_loss

BASE = str(ROOT / "runs/_baseline_cleanL2_ep34_for_p1diag_compare/best_recon_model.pt")
LW = {"pos": 1.0, "rot": 1.0, "vel": 1.0, "contact": 0.1, "kl": 1e-3, "pool_aux": 0.5}
W_WORLD, W_FK, W_TRAJ = 0.25, 0.25, 0.10
N_BATCH, BS = 5, 4


def build_vae(dev):
    ck = torch.load(BASE, map_location="cpu", weights_only=True); ta = ck["args"]
    vae = GraphMotionVAE(
        pool_type=ta["pool_type"], pool_tau=ta.get("pool_tau"), d_model=ta["d_model"],
        n_heads=ta["n_heads"], d_ff=ta["d_ff"], n_graph_layers=ta["n_graph_layers"],
        n_enc_temporal_layers=ta["n_enc_temporal_layers"], n_cross_layers=ta["n_cross_layers"],
        n_dec_temporal_layers=ta["n_dec_temporal_layers"], n_treeik_layers=ta["n_treeik_layers"],
        max_coarse=ta["max_coarse"], local_radius=ta["local_radius"], temporal_stride=ta["temporal_stride"],
        temporal_kernel=ta["temporal_kernel"], dropout=ta["dropout"], feat_mode="anytop13",
        attn_mode=ta.get("attn_mode") or "scalar", decoder_mode=ta.get("decoder_mode") or "unpool_identity",
        n_graph_temporal_layers=ta.get("n_graph_temporal_layers", 4)).to(dev)
    vae.load_state_dict(ck["model_state_dict"], strict=True)
    vae.encoder.use_name_embed = bool(ta.get("use_name_embed", False)); vae.eval()
    return vae


def main():
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds = AnyTopDataset(split="val", val_frac=0.05, seed=42,
                       data_root=str(ROOT / "data/anytop_planet_zoo_clean_L2"),
                       num_frames=64, max_joints=144, caption_emb_cache=None)
    vae = build_vae(dev)
    acc = {k: [] for k in ("base_total", "pos", "rot", "vel", "contact", "kl",
                           "pool_aux", "world", "fk", "traj", "gt_fk_mismatch")}
    idxs = np.linspace(0, len(ds) - 1, N_BATCH * BS).astype(int)
    with torch.no_grad():
        for bi in range(N_BATCH):
            items = [ds[int(idxs[bi * BS + j])] for j in range(BS)]
            d = collate_fn(items)
            d = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in d.items()}
            batch = GraphMotionBatch.from_collate_dict(d)
            out = vae(batch, sample=False)
            losses = run_loss(out, batch, "anytop13", LW, out["frame_mask_recovered"], dev,
                              loss_mode="anytop13_world_rot6d_fk",
                              w_world=W_WORLD, w_fk=W_FK, w_traj=W_TRAJ)
            # base_total = total minus the geometry adds
            geo = W_WORLD * losses["world"] + W_FK * losses["fk"] + W_TRAJ * losses["traj"]
            base_total = (losses["total"] - geo).item()
            acc["base_total"].append(base_total)
            for k in ("pos", "rot", "vel", "contact", "kl", "pool_aux"):
                acc[k].append(losses[k].item() if k in losses else float("nan"))
            for k in ("world", "fk", "traj", "gt_fk_mismatch"):
                acc[k].append(losses[k].item())

    m = {k: float(np.mean(v)) for k, v in acc.items()}
    bt = m["base_total"]
    ww, wf, wt = W_WORLD * m["world"], W_FK * m["fk"], W_TRAJ * m["traj"]
    print("=== RAW (mean over %d batches x %d) ===" % (N_BATCH, BS), flush=True)
    print(f"  base_total={bt:.4f}", flush=True)
    for k in ("pos", "rot", "vel", "contact", "kl", "pool_aux"):
        print(f"    {k:9s}={m[k]:.4f}", flush=True)
    print(f"  world={m['world']:.4f}  fk={m['fk']:.4f}  traj={m['traj']:.4f}  "
          f"gt_fk_mismatch={m['gt_fk_mismatch']:.4f}", flush=True)
    print(f"=== WEIGHTED @ w_world={W_WORLD} w_fk={W_FK} w_traj={W_TRAJ} ===", flush=True)
    print(f"  w_world*world={ww:.4f} ({ww/bt*100:.1f}% of base)", flush=True)
    print(f"  w_fk*fk      ={wf:.4f} ({wf/bt*100:.1f}% of base)", flush=True)
    print(f"  w_traj*traj  ={wt:.4f} ({wt/bt*100:.1f}% of base)", flush=True)
    geo_pct = (ww + wf + wt) / bt * 100
    print(f"  total geometry = {ww+wf+wt:.4f} ({geo_pct:.1f}% of base)", flush=True)
    # plan §6 rule
    print("=== PLAN §6 RECOMMENDATION ===", flush=True)
    if ww / bt < 0.10 and wf / bt < 0.10:
        print("  Both geometry terms <10% of base -> ALSO run stronger arm 0.5/0.5/0.25", flush=True)
    if ww / bt > 0.60 or wf / bt > 0.60:
        print("  A geometry term >60% of base -> REDUCE to 0.10/0.10/0.05", flush=True)
    # equalize w_world*world ~= w_fk*fk
    if m["fk"] > 1e-9:
        wf_eq = ww / m["fk"]   # weight that makes w_fk*fk == w_world*world
        print(f"  To equalize contributions (w_world*world==w_fk*fk): w_fk≈{wf_eq:.3f} "
              f"(world={m['world']:.4f} vs fk={m['fk']:.4f}, ratio={m['fk']/max(m['world'],1e-9):.2f}x)", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
