"""noKslot_clean / scripts/animate_graph_vae.py — visual QA for graph_salad VAE.

Adapted from scripts/animate.py (which targets the noKslot baseline Model +
TopoFKTreeIKDecoder). This version targets `GraphMotionVAE` ckpts produced
by `scripts/train_graph_vae.py` (M1.5).

Compatibility:
  - Reuses `animate_clip`, `contact_sheet` helpers VERBATIM from animate.py
    (pure numpy/matplotlib, model-agnostic).
  - Reuses `reach_metrics` from scripts.eval (distal diagnostics).
  - Replaces baseline Model+TopoFKTreeIKDecoder load+forward with
    `GraphMotionVAE` + deterministic forward (sample=False).

Per cross-project rule "可视化 demo 准确度 > metric": this is THE truth
gate for M1.5 — recon numbers (7.4-8.5) must be cross-checked against
visible motion fidelity.

Usage:
  python -u scripts/animate_graph_vae.py \\
      --ckpt runs/m1_5_graph_vae_dynamic_seed42/best_recon_model.pt \\
      --src_dir data/cs_sparse2full_tgt \\
      --tgt_dir data/cs_sparse2full_tgt \\
      --split val --species Bat,Crab,Trex,Dragon \\
      --n_per 2 --stride 2 \\
      --out runs/m1_5_graph_vae_dynamic_seed42/qa_animate
"""
import argparse
import sys
from pathlib import Path

import matplotlib
import numpy as np
import torch

matplotlib.use('Agg')

project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

# Reuse the rendering helpers VERBATIM (model-agnostic).
from scripts.animate import animate_clip, contact_sheet  # noqa: E402
from scripts.eval import reach_metrics  # noqa: E402
from src.data.unified_dataset import UnifiedMotionDataset, collate_fn  # noqa: E402
from src.models.graph_salad import GraphMotionBatch, GraphMotionVAE  # noqa: E402
from src.utils import to_dev  # noqa: E402


def load_graph_vae(ckpt_path: str, dev: torch.device):
    """Load GraphMotionVAE from a train_graph_vae.py-produced ckpt.

    Mirrors scripts/eval_graph_vae.py:127-149 (verified ckpt-compat path):
    reconstruct VAE with ckpt's training hparams + load weights with strict=True.
    """
    ck = torch.load(ckpt_path, map_location='cpu', weights_only=True)
    ta = ck.get('args', {})
    if not ta:
        raise SystemExit(f'animate_graph_vae: ckpt {ckpt_path} missing "args" '
                         f'key (not produced by train_graph_vae.py?)')
    vae = GraphMotionVAE(
        pool_type=ta['pool_type'],
        pool_tau=ta.get('pool_tau'),  # None for non-soft variants
        d_model=ta['d_model'], n_heads=ta['n_heads'],
        d_ff=ta['d_ff'], n_graph_layers=ta['n_graph_layers'],
        n_enc_temporal_layers=ta['n_enc_temporal_layers'],
        n_cross_layers=ta['n_cross_layers'],
        n_dec_temporal_layers=ta['n_dec_temporal_layers'],
        n_treeik_layers=ta['n_treeik_layers'],
        max_coarse=ta['max_coarse'],
        local_radius=ta['local_radius'],
        temporal_stride=ta['temporal_stride'],
        temporal_kernel=ta['temporal_kernel'],
        dropout=ta['dropout'],
    ).to(dev)
    vae.load_state_dict(ck['model_state_dict'], strict=True)
    vae.eval()
    return vae, ta


def forward_clip_graph_vae(vae, batch_dict, dev):
    """Single-path forward for graph_salad animation.

    Constructs a GraphMotionBatch from collate_fn output, runs the VAE with
    sample=False (deterministic z=mu so visuals are reproducible), and
    returns predictions in the SAME shape as animate.py's baseline forward:
    [B, T, J, 6] = cat(pred_pos, pred_vel).
    """
    batch = GraphMotionBatch.from_collate_dict({
        k: v.to(dev) if torch.is_tensor(v) else v for k, v in batch_dict.items()
    })
    out = vae(batch, sample=False)
    # Apply the effective frame mask (same as eval) so stride-tail frames
    # are zeroed in viz — matches what the model could actually produce.
    pred_pos = out['pred_pos']  # [B, T, J, 3]
    pred_vel = out['pred_vel']  # [B, T, J, 3]
    return torch.cat([pred_pos, pred_vel], dim=-1)  # [B, T, J, 6]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--src_dir', required=True,
                    help='unified dataset dir (graph_salad uses same data as '
                         'baseline; src and tgt typically the same path).')
    ap.add_argument('--tgt_dir', required=True)
    ap.add_argument('--max_frames', type=int, default=None,
                    help='If None, inherit from ckpt args')
    ap.add_argument('--max_joints', type=int, default=None,
                    help='If None, inherit from ckpt args')
    ap.add_argument('--species', default='Bat,Crab,Trex,Dragon',
                    help='Comma-separated species names to render. Defaults '
                         "cover small (Bat 48J), medium (Crab 54J), large "
                         "(Trex 61J, Dragon 143J) topologies.")
    ap.add_argument('--split', default='val')
    ap.add_argument('--n_per', type=int, default=2)
    ap.add_argument('--stride', type=int, default=2,
                    help='frame subsample for gif (visual smoothness)')
    ap.add_argument('--fps', type=int, default=8)
    ap.add_argument('--out', required=True)
    ap.add_argument('--device', default='cuda')
    args = ap.parse_args()

    if args.device == 'cuda' and not torch.cuda.is_available():
        raise SystemExit('[DEVICE FAIL] --device cuda requested but unavailable')
    dev = torch.device(args.device)

    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    want = [x.strip() for x in args.species.split(',') if x.strip()]

    # Load model first so we can inherit max_frames/joints from train_args
    vae, train_args = load_graph_vae(args.ckpt, dev)
    if args.max_frames is None:
        args.max_frames = train_args.get('max_frames', 64)
        print(f'  max_frames inherited from ckpt: {args.max_frames}')
    if args.max_joints is None:
        args.max_joints = train_args.get('max_joints', 160)
        print(f'  max_joints inherited from ckpt: {args.max_joints}')

    print(f'Loaded VAE: pool_type={train_args["pool_type"]}, '
          f'pool_tau={train_args.get("pool_tau")}, '
          f'd_model={train_args["d_model"]}')

    mk = lambda d: UnifiedMotionDataset(
        [d], args.split, max_frames=args.max_frames,
        max_joints=args.max_joints, normalize=False)
    src_va, tgt_va = mk(args.src_dir), mk(args.tgt_dir)
    if len(src_va) != len(tgt_va):
        raise SystemExit(f'pair len mismatch: src={len(src_va)} tgt={len(tgt_va)}')

    picked = {sp: 0 for sp in want}
    summary = []
    pool_type_tag = train_args['pool_type']
    if pool_type_tag == 'soft_deterministic':
        pool_type_tag = f"soft_det_t{train_args['pool_tau']}"

    with torch.no_grad():
        for i in range(len(src_va)):
            sp = str(tgt_va.samples[i].get('skeleton_id', '?'))
            if sp not in picked or picked[sp] >= args.n_per:
                continue
            # We only need ONE batch (src == tgt in our setting): use tgt
            t = to_dev(collate_fn([tgt_va[i]]), dev)
            pred = forward_clip_graph_vae(vae, t, dev)
            gt = t['motion_features']
            jm = t['joint_mask'].float(); fm = t['frame_mask'].float()
            Jv = int(jm[0].sum().item()); Tv = int(fm[0].sum().item())
            pp = pred[0, :Tv, :Jv, :3].cpu().numpy()
            gp = gt[0, :Tv, :Jv, :3].cpu().numpy()
            pv = pred[0, :Tv, :Jv, 3:].cpu().numpy()
            gv = gt[0, :Tv, :Jv, 3:].cpu().numpy()
            par = [int(x) for x in t['parent_indices'][0]][:Jv]
            tgt_cn = [str(x) for x in t['canonical_names'][0]][:Jv]
            src_cn = [str(x) for x in t['canonical_names'][0]]  # same here
            rm = (reach_metrics(pp, gp, pv, gv, par, tgt_cn, src_cn)
                  if len(tgt_cn) == Jv else
                  {'distal_n': 0, 'frozen_rate_clip': float('nan'),
                   'collapse_rate_clip': float('nan'),
                   'distal_vel_ratio_median_clip': float('nan'),
                   'distal_vel_corr_clip': float('nan')})
            diag = (f"distal_n={rm['distal_n']} "
                    f"frozen={rm['frozen_rate_clip']:.3f} "
                    f"collapse={rm['collapse_rate_clip']:.3f} "
                    f"distal_vratio={rm['distal_vel_ratio_median_clip']:.3f} "
                    f"distal_vcorr={rm['distal_vel_corr_clip']:.3f}")
            k = picked[sp]
            path = out_dir / f'{sp}_clip{k}_gtvspred.gif'
            ttl = (f'{sp} clip{k} [graph_salad/{pool_type_tag}] '
                   f'Jv={Jv} Tv={Tv}  {diag}')
            animate_clip(pp, gp, par, str(path), ttl, args.stride, args.fps)
            VIEWS = [(12, -70, 'obl'), (75, -90, 'top')]
            for elev, azim, tag in VIEWS:
                contact_sheet(pp, gp, par,
                              str(out_dir / f'{sp}_clip{k}_sheet_{tag}.png'),
                              ttl, elev=elev, azim=azim)
            g_spd = float(np.linalg.norm(np.diff(gp, axis=0), axis=-1).mean())
            p_spd = float(np.linalg.norm(np.diff(pp, axis=0), axis=-1).mean())
            line = (f'{sp} clip{k}: Jv={Jv} Tv={Tv} '
                    f'GT_meanspeed={g_spd:.4f} PRED_meanspeed={p_spd:.4f} '
                    f'ratio={p_spd / max(g_spd, 1e-9):.3f} | {diag} '
                    f'-> {path.name}')
            print(line); summary.append(line)
            picked[sp] += 1
            if all(picked[x] >= args.n_per for x in want):
                break

    missing = {sp: args.n_per - picked[sp] for sp in want
               if picked[sp] < args.n_per}
    (out_dir / 'animate_summary.txt').write_text(
        '\n'.join(summary) + f'\nmissing={missing}\n')
    print(f'\nDONE {sum(picked.values())} gifs -> {out_dir}')
    print('PER-SPECIES picked:', picked)
    if missing:
        raise RuntimeError(f'visual QA missing required clips: {missing}')


if __name__ == '__main__':
    main()
