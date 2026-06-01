"""Smoke (Step3 wiring): run_loss with loss_mode default vs anytop13_world_geometry.

User-required 4 gates:
  G1. default loss_mode="anytop13" total == pre-change total (byte-identical).
      We approximate "pre-change" by calling compute_total_loss_13ch directly
      (the exact code path run_loss took before Step3) and comparing to
      run_loss(..., loss_mode="anytop13").  MUST be exactly equal.
  G2. loss_mode="anytop13_world_geometry": losses has world & traj keys, both >0,
      and total == default_total + w_world*world + w_traj*traj (exact).
  G3. backward() on the world_geometry total -> grads finite.
  G4. (path) recover_world_positions_torch runs on pred (already covered by
      gate1 smoke; here we just assert the world term is differentiable).

Uses a REAL batch + a REAL forward through an edge_segment VAE ckpt (the A/B
base architecture) so out["pred_motion"]/mu/logvar/coarse_mask are real shapes.

Run: python scripts/_smoke_step3_run_loss_wiring.py
"""
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.anytop_dataset import AnyTopDataset, collate_fn  # noqa: E402
from src.models.graph_salad.batch import GraphMotionBatch  # noqa: E402
from src.models.graph_salad.vae import GraphMotionVAE  # noqa: E402
from src.models.graph_salad.losses import compute_total_loss_13ch  # noqa: E402
# import the patched run_loss from the training script
import importlib.util  # noqa: E402
spec = importlib.util.spec_from_file_location(
    "train_graph_vae", str(ROOT / "scripts" / "train_graph_vae.py"))
tgv = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tgv)
run_loss = tgv.run_loss

ANYTOP_ROOT = str(ROOT / "data" / "anytop_planet_zoo_clean_L2")
BASE_CKPT = str(ROOT / "runs" / "_baseline_cleanL2_ep34_for_p1diag_compare" / "best_recon_model.pt")


def build_vae(dev):
    ck = torch.load(BASE_CKPT, map_location="cpu", weights_only=True)
    ta = ck["args"]
    vae = GraphMotionVAE(
        pool_type=ta["pool_type"], pool_tau=ta.get("pool_tau"),
        d_model=ta["d_model"], n_heads=ta["n_heads"], d_ff=ta["d_ff"],
        n_graph_layers=ta["n_graph_layers"],
        n_enc_temporal_layers=ta["n_enc_temporal_layers"],
        n_cross_layers=ta["n_cross_layers"],
        n_dec_temporal_layers=ta["n_dec_temporal_layers"],
        n_treeik_layers=ta["n_treeik_layers"],
        max_coarse=ta["max_coarse"], local_radius=ta["local_radius"],
        temporal_stride=ta["temporal_stride"], temporal_kernel=ta["temporal_kernel"],
        dropout=ta["dropout"], feat_mode="anytop13",
        attn_mode=ta.get("attn_mode") or "scalar",
        decoder_mode=ta.get("decoder_mode") or "unpool_identity",
        n_graph_temporal_layers=ta.get("n_graph_temporal_layers", 4),
    ).to(dev)
    vae.load_state_dict(ck["model_state_dict"], strict=True)
    vae.encoder.use_name_embed = bool(ta.get("use_name_embed", False))
    vae.eval()
    return vae, ta


def main():
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds = AnyTopDataset(split="val", val_frac=0.05, seed=42, data_root=ANYTOP_ROOT,
                       num_frames=64, max_joints=144, caption_emb_cache=None)
    d = collate_fn([ds[i] for i in range(4)])
    d = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in d.items()}
    batch = GraphMotionBatch.from_collate_dict(d)
    vae, ta = build_vae(dev)

    out = vae(batch, sample=False)
    eff_fm = out["frame_mask_recovered"]
    lw = {"pos": 1.0, "rot": 1.0, "vel": 1.0, "contact": 0.1, "kl": 1e-3, "pool_aux": 0.5}

    # ---- G1: default == direct compute_total_loss_13ch ----
    gt_motion = batch.anytop_x.permute(0, 3, 1, 2).contiguous()
    ref = compute_total_loss_13ch(
        pred_motion=out["pred_motion"], gt_motion=gt_motion,
        foot_contact_per_joint=batch.foot_contact_per_joint,
        mu=out["mu"], logvar=out["logvar"], pool_aux_outputs=out["pool_aux_outputs"],
        joint_mask=batch.joint_mask, frame_mask=eff_fm,
        coarse_mask=out["coarse_mask"], frame_mask_lat=out["frame_mask_lat"],
        weights=lw,
    )
    default = run_loss(out, batch, "anytop13", lw, eff_fm, dev,
                       loss_mode="anytop13", w_world=0.5, w_traj=0.25)
    g1_total = torch.equal(ref["total"], default["total"])
    g1_nokeys = ("world" not in default and "traj" not in default)
    print(f"G1 default total == direct: {g1_total}  | no world/traj keys: {g1_nokeys}", flush=True)
    print(f"   ref.total={ref['total'].item():.6f}  default.total={default['total'].item():.6f}", flush=True)

    # ---- G2: world_geometry adds terms + exact accumulation ----
    wg = run_loss(out, batch, "anytop13", lw, eff_fm, dev,
                  loss_mode="anytop13_world_geometry", w_world=0.5, w_traj=0.25)
    has_keys = ("world" in wg and "traj" in wg)
    world_pos = wg["world"].item() > 0
    traj_pos = wg["traj"].item() > 0
    expected_total = default["total"] + 0.5 * wg["world"] + 0.25 * wg["traj"]
    g2_acc = torch.allclose(wg["total"], expected_total, atol=1e-6)
    print(f"G2 world/traj keys: {has_keys}  world>0: {world_pos} ({wg['world'].item():.4f})  "
          f"traj>0: {traj_pos} ({wg['traj'].item():.4f})", flush=True)
    print(f"   accumulation exact (total == default + 0.5w + 0.25t): {g2_acc}", flush=True)
    print(f"   default.total={default['total'].item():.6f}  wg.total={wg['total'].item():.6f}", flush=True)

    # ---- G3: backward finite ----
    # need grad: re-run forward with grad enabled (vae frozen-eval still builds graph)
    out2 = vae(batch, sample=False)
    wg2 = run_loss(out2, batch, "anytop13", lw, out2["frame_mask_recovered"], dev,
                   loss_mode="anytop13_world_geometry", w_world=0.5, w_traj=0.25)
    wg2["total"].backward()
    grads = [p.grad for p in vae.parameters() if p.grad is not None]
    g3_finite = all(torch.isfinite(g).all().item() for g in grads)
    g3_nonzero = sum(float(g.abs().sum().item()) for g in grads) > 0
    print(f"G3 backward grads finite: {g3_finite}  nonzero: {g3_nonzero}", flush=True)

    ok = g1_total and g1_nokeys and has_keys and world_pos and traj_pos and g2_acc and g3_finite and g3_nonzero
    print(f"\nSTEP3_GATE {'PASS' if ok else 'FAIL'}", flush=True)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
