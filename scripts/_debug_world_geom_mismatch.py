"""Debug the check5 mismatch (denorm->recover vs dataset world GT, 0.297).
Execution-only (clean stdout). Localizes the bug without reading source.
"""
import sys
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.anytop_dataset import AnyTopDataset, collate_fn  # noqa
from src.models.graph_salad.batch import GraphMotionBatch  # noqa
from src.models.graph_salad.losses import _denorm_13ch  # noqa
from src.models.graph_salad.world_recovery import recover_world_positions_torch  # noqa

ANYTOP_ROOT = str(ROOT / "data" / "anytop_planet_zoo_clean_L2")


def main():
    ds = AnyTopDataset(split="val", val_frac=0.05, seed=42, data_root=ANYTOP_ROOT,
                       num_frames=64, max_joints=144, caption_emb_cache=None)
    items = [ds[0]]
    it = items[0]
    print("item keys:", sorted(it.keys()), flush=True)
    d = collate_fn(items)
    print("collate keys:", sorted(d.keys()), flush=True)
    batch = GraphMotionBatch.from_collate_dict(d)

    gt_motion = batch.anytop_x.permute(0, 3, 1, 2).contiguous()  # [B,T,J,13]
    B, T, J, _ = gt_motion.shape
    Jv = int(batch.joint_mask[0].sum().item())
    Tv = int(batch.frame_mask[0].sum().item())
    print(f"B={B} T={T} J={J}  valid J={Jv} valid T={Tv}", flush=True)

    # (a) denorm round-trip: renorm(denorm(anytop_x)) vs anytop_x  -> should be ~0
    gt_raw = _denorm_13ch(gt_motion, batch.anytop_mean, batch.anytop_std)
    mean = batch.anytop_mean.unsqueeze(1)
    std = batch.anytop_std.unsqueeze(1)
    renorm = (gt_raw - mean) / (std + 1e-6)
    mask = (batch.joint_mask.unsqueeze(1) & batch.frame_mask.unsqueeze(-1)).unsqueeze(-1).float()
    rt = ((renorm - gt_motion).abs() * mask).sum() / mask.sum().clamp(min=1e-8)
    print(f"(a) denorm roundtrip renorm vs anytop_x: {rt.item():.3e} (expect ~0)", flush=True)

    # (b) world from my path vs dataset, split root vs nonroot, valid only
    world_mine = recover_world_positions_torch(gt_raw)        # [B,T,J,3]
    world_ds = batch.motion_features[..., :3]                  # [B,T,J,3]
    # valid mask per joint/frame
    vm = (batch.joint_mask.unsqueeze(1) & batch.frame_mask.unsqueeze(-1))  # [B,T,J]
    vm3 = vm.unsqueeze(-1).float()
    # root joint 0 only (valid frames)
    fm = batch.frame_mask.float()  # [B,T]
    root_diff = ((world_mine[:, :, 0, :] - world_ds[:, :, 0, :]).abs().sum(-1) * fm).sum() / fm.sum().clamp(min=1e-8)
    # nonroot joints 1..
    nr_mask = vm.clone(); nr_mask[:, :, 0] = False
    nr3 = nr_mask.unsqueeze(-1).float()
    nr_diff = ((world_mine[:, :, 1:, :] - world_ds[:, :, 1:, :]).abs() * nr3[:, :, 1:, :]).sum() / nr3[:, :, 1:, :].sum().clamp(min=1e-8)
    print(f"(b) world diff  root={root_diff.item():.3e}  nonroot={nr_diff.item():.3e}", flush=True)

    # (c) is diff concentrated in late (padded) frames? compare first valid frame
    f0_diff = (world_mine[0, 0, :Jv, :] - world_ds[0, 0, :Jv, :]).abs().mean()
    fmid = Tv // 2
    fmid_diff = (world_mine[0, fmid, :Jv, :] - world_ds[0, fmid, :Jv, :]).abs().mean()
    flast = Tv - 1
    flast_diff = (world_mine[0, flast, :Jv, :] - world_ds[0, flast, :Jv, :]).abs().mean()
    print(f"(c) per-frame diff  f0={f0_diff.item():.3e}  fmid({fmid})={fmid_diff.item():.3e}  flast({flast})={flast_diff.item():.3e}", flush=True)

    # (d) magnitudes: typical world coord scale (so 0.297 is relative-to-what)
    ds_scale = (world_ds[0, :Tv, :Jv, :].abs().mean()).item()
    print(f"(d) dataset world coord mean|val|={ds_scale:.3e}  (diff 0.297 relative={0.297/max(ds_scale,1e-9):.2f})", flush=True)

    # (e) sample raw root channels: is channel 1 (height) / 9,11 (vel) sane after denorm?
    rr = gt_raw[0, :Tv, 0, :]  # [Tv,13] root raw
    print(f"(e) root raw ch1(height) range [{rr[:,1].min().item():.3f},{rr[:,1].max().item():.3f}]  "
          f"ch9(velx) range [{rr[:,9].min().item():.3f},{rr[:,9].max().item():.3f}]  "
          f"ch11(velz) range [{rr[:,11].min().item():.3f},{rr[:,11].max().item():.3f}]", flush=True)

    # (f) does dataset world GT first frame root == origin-ish? recovery starts cumsum at 0
    print(f"(f) world_ds root f0 = {world_ds[0,0,0,:].tolist()}  world_mine root f0 = {world_mine[0,0,0,:].tolist()}", flush=True)


if __name__ == "__main__":
    main()
