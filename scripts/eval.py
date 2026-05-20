"""noKslot_clean / scripts/eval.py — single-path eval entry for the noKslot
reproducible baseline.

Loads a NoKslotModel + TopoFKTreeIKDecoder ckpt (from this repo's train.py OR
the source-produced `runs/baseline_noKslot_ep399/last_model.pt`), runs gate
metrics + REACH metrics + per-species breakdown, optionally auto-chains
scripts/animate.py for forced GT-vs-pred multi-frame gif + dual-view contact
sheet rendering (visual-QA-primacy hard gate).

Helpers (masked / depths_from_parents / pooled_corr / _empty_reach /
reach_metrics / _finite / reach_aggregate / reach_verdict / _san /
_rot_angle_deg / THRESH constants / REACH_THRESH / _render) are verbatim
ports from source motion_representation_study/scripts/eval_paired_gate.py.

main() is simplified for the single no_k_slot path:
  - no --head / --variant / --no_k_slot args (locked to topofk_treeik +
    no_k_slot=True; ckpt arg sniff removed since this repo has one path only)
  - --species default 'Bat,Crab,Horse' (noKslot baseline triad, vs source's
    Dragon/Spider/Trex which were the K-slot cross-species held-out triad)
  - tolerant ckpt schema: loads model state with strict=False (drops
    slot_assignment.* keys from source-produced baseline ckpt) and topofk
    state with strict=True; fk_state_dict (if present in source ckpt)
    is ignored.
  - reach gate is NOT enforced for noKslot baseline (src==tgt same-skeleton
    self-recon -> reach metric is trivially empty; reach numbers are still
    reported as diagnostic record).
  - animation gate IS enforced (visual-QA-primacy rule per
    feedback_visual_qa_primacy; rendering chain delegates to
    scripts/animate.py).

Run example:
  python -u scripts/eval.py \\
      --ckpt runs/baseline_noKslot_ep399/last_model.pt \\
      --src_dir data/cs_sparse2full_tgt \\
      --tgt_dir data/cs_sparse2full_tgt \\
      --ik_dir data/cs_sparse2full_ik_rot \\
      --split val --species Bat,Crab,Horse \\
      --out runs/eval_baseline_noKslot_ep399
"""
import argparse
import json
import os
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
import numpy as np
import torch

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.unified_dataset import UnifiedMotionDataset, collate_fn  # noqa: E402
from src.models.noKslot_model import NoKslotModel  # noqa: E402
from src.models.treeik_decoder import TopoFKTreeIKDecoder  # noqa: E402
from src.utils import fps_of, to_dev  # noqa: E402
from scripts.train import encode_decode_nok  # noqa: E402


# =========================================================================== #
# THRESH constants — verbatim values from source THRESH_TREX (which source eval
# used for the noKslot diagnostic baseline since `'cs_sparse2full' in tgt_dir`).
# Renamed THRESH_NOKSLOT_BASELINE here to make the application context explicit.
# =========================================================================== #
THRESH_NOKSLOT_BASELINE = {
    'pos_nrmse_extent': ('<', 0.10),
    'bone_len_rel_mean_pct': ('<', 5.0),
    'bone_len_rel_p95_pct': ('<', 10.0),
    'edge_ratio_p05': ('>', 0.75),
    'vel_corr': ('>', 0.90),
    'vel_nrmse': ('<', 0.30),
}

# codex C5 reach pass gate — verbatim from source. NOT enforced for the
# noKslot same-skeleton self-recon baseline (reach is trivially empty for
# src==tgt), but the aggregate is still computed + reported for diagnostic
# record.
REACH_THRESH = {
    'frozen_rate': ('<', 0.05),
    'collapse_rate': ('<', 0.01),
    'distal_vel_ratio_median': ('rng', (0.3, 3.0)),
    'distal_vel_corr': ('>', 0.40),
}
REACH_THRESH_STR = {'frozen_rate': '<0.05', 'collapse_rate': '<0.01',
                    'distal_vel_ratio_median': 'in(0.3,3.0)',
                    'distal_vel_corr': '>0.40'}


# =========================================================================== #
# Helper functions — VERBATIM from source eval_paired_gate.py:54-275.
# =========================================================================== #
def masked(x, jm, fm):
    m = (jm[:, None, :, None] * fm[:, :, None, None]).bool()
    return x[m.expand_as(x)]


def depths_from_parents(parents):
    d = np.zeros(len(parents), dtype=np.int64)
    for j, p in enumerate(parents):
        d[j] = 0 if p < 0 else d[p] + 1
    return d


def pooled_corr(a, b, eps=1e-9):
    a = a.reshape(-1); b = b.reshape(-1)
    a = a - a.mean(); b = b - b.mean()
    return float((a @ b) / (np.linalg.norm(a) * np.linalg.norm(b) + eps))


def _empty_reach(extra=None):
    """Zero/empty per-clip reach record (n_eval=0 -> contributes nothing to
    species pools; if a whole species is empty -> nan -> reach_verdict FAIL)."""
    d = {'distal_n': 0, 'n_frozen': 0, 'n_eval': 0, 'vel_ratios': [],
         'vc': {'n': 0, 'sa': 0.0, 'sb': 0.0, 'saa': 0.0, 'sbb': 0.0,
                'sab': 0.0},
         'n_bad_pairs': 0, 'n_tested_pairs': 0,
         'frozen_rate_clip': float('nan'), 'collapse_rate_clip': 0.0,
         'distal_vel_ratio_median_clip': float('nan'),
         'distal_vel_corr_clip': float('nan'), 'per_depth': {}}
    if extra:
        d.update(extra)
    return d


def reach_metrics(pp, gp, pv, gv, parents, tgt_names, src_names):
    """Per-joint/per-pair math is codex C5 VERBATIM. Only the RETURN changed
    (codex re-review FIX-AGAIN #3): emit RAW accumulable components so the
    species pass-gate aggregation is species-POOLED / COUNT-WEIGHTED, NOT
    mean-of-per-clip. pp/gp/pv/gv: [T,J,3], valid unpadded only."""
    extent = max(float(np.abs(gp).max()), 1e-6)
    src_set = set(map(str, src_names))
    tgt_names = list(map(str, tgt_names))
    distal = np.array([(n not in src_set) and parents[j] >= 0
                       for j, n in enumerate(tgt_names)], bool)

    gt_amp = np.sqrt(((gp - gp.mean(0, keepdims=True)) ** 2).sum(-1).mean(0))
    pr_amp = np.sqrt(((pp - pp.mean(0, keepdims=True)) ** 2).sum(-1).mean(0))
    gt_vrms = np.sqrt((gv ** 2).sum(-1).mean(0))
    pr_vrms = np.sqrt((pv ** 2).sum(-1).mean(0))

    moving = gt_amp > max(0.01 * extent, 0.02)
    eval_j = distal & moving                       # codex C5 verbatim

    amp_ratio = pr_amp / np.maximum(gt_amp, 1e-6)
    vel_ratio = pr_vrms / np.maximum(gt_vrms, 1e-6)

    # non-adjacent collapse: codex C5 verbatim; count bad / tested pairs
    n_bad = n_tested = 0
    J = gp.shape[1]
    for a in range(J):
        for b in range(a + 1, J):
            if parents[a] == b or parents[b] == a:
                continue
            gd = np.median(np.linalg.norm(gp[:, a] - gp[:, b], axis=-1))
            if gd > 0.05 * extent:
                n_tested += 1
                pd = np.median(np.linalg.norm(pp[:, a] - pp[:, b], axis=-1))
                if pd < max(0.01 * extent, 0.03):
                    n_bad += 1

    depths = depths_from_parents(parents)
    per_depth = {}
    for lo, hi, key in [(0, 2, "d0_2"), (3, 5, "d3_5"), (6, 999, "d6p")]:
        m = eval_j & (depths >= lo) & (depths <= hi)
        per_depth[key] = {
            "n": int(m.sum()),
            "frozen_rate": float(np.mean(amp_ratio[m] < 0.10)) if m.any() else np.nan,
            "vel_corr": pooled_corr(pv[:, m], gv[:, m]) if m.any() else np.nan,
            "vel_ratio_median": float(np.median(vel_ratio[m])) if m.any() else np.nan,
        }

    if not eval_j.any():
        return _empty_reach({'n_bad_pairs': n_bad, 'n_tested_pairs': n_tested,
                             'collapse_rate_clip': (n_bad / n_tested
                                                    if n_tested else 0.0),
                             'per_depth': per_depth})

    n_eval = int(eval_j.sum())
    n_frozen = int(np.sum(amp_ratio[eval_j] < 0.10))
    ratios = vel_ratio[eval_j].astype(np.float64).tolist()
    # raw moments for species-POOLED corr (concat all clips, center once ==
    # closed-form Pearson from n,Σa,Σb,Σa²,Σb²,Σab). float64 accumulation.
    A = pv[:, eval_j, :].reshape(-1).astype(np.float64)
    B = gv[:, eval_j, :].reshape(-1).astype(np.float64)
    vc = {'n': int(A.size), 'sa': float(A.sum()), 'sb': float(B.sum()),
          'saa': float((A * A).sum()), 'sbb': float((B * B).sum()),
          'sab': float((A * B).sum())}
    return {
        'distal_n': n_eval,
        'n_frozen': n_frozen,
        'n_eval': n_eval,
        'vel_ratios': ratios,                 # all per-distal-joint ratios
        'vc': vc,                             # pooled-corr accumulators
        'n_bad_pairs': n_bad,
        'n_tested_pairs': n_tested,
        # per-clip scalars: DIAGNOSTIC ONLY (reach_detail.json), NOT the gate
        'frozen_rate_clip': n_frozen / n_eval,
        'collapse_rate_clip': (n_bad / n_tested if n_tested else 0.0),
        'distal_vel_ratio_median_clip': float(np.median(vel_ratio[eval_j])),
        'distal_vel_corr_clip': pooled_corr(pv[:, eval_j], gv[:, eval_j]),
        'per_depth': per_depth,
    }


def _finite(vs):
    return [v for v in vs if v is not None and np.isfinite(v)]


def reach_aggregate(clips):
    """codex re-review FIX-AGAIN #3 EXACT spec — species-POOLED / COUNT-
    WEIGHTED, NOT mean-of-per-clip:
      frozen_rate = total frozen distal / total evaluated distal
      collapse_rate = total bad non-adj pairs / total tested non-adj pairs
      distal_vel_ratio_median = median over ALL evaluated distal ratios
      distal_vel_corr = ONE corr over all clips' distal (pv,gv) concatenated,
                        centered once (closed-form Pearson from pooled moments
                        == concatenate + center once + one corr)."""
    if not clips:
        return None
    n_frozen = sum(c['n_frozen'] for c in clips)
    n_eval = sum(c['n_eval'] for c in clips)
    n_bad = sum(c['n_bad_pairs'] for c in clips)
    n_tested = sum(c['n_tested_pairs'] for c in clips)
    ratios = [r for c in clips for r in c['vel_ratios']]

    n = sum(c['vc']['n'] for c in clips)
    sa = sum(c['vc']['sa'] for c in clips)
    sb = sum(c['vc']['sb'] for c in clips)
    saa = sum(c['vc']['saa'] for c in clips)
    sbb = sum(c['vc']['sbb'] for c in clips)
    sab = sum(c['vc']['sab'] for c in clips)
    if n > 1:
        cov = sab - sa * sb / n
        va_ = saa - sa * sa / n
        vb_ = sbb - sb * sb / n
        denom = (max(va_, 0.0) ** 0.5) * (max(vb_, 0.0) ** 0.5)
        dvc = float(cov / denom) if denom > 1e-12 else np.nan
    else:
        dvc = np.nan

    return {
        'n_clips': len(clips),
        'distal_n_total': int(n_eval),
        'n_frozen_total': int(n_frozen),
        'n_tested_pairs_total': int(n_tested),
        'frozen_rate': (n_frozen / n_eval if n_eval else np.nan),
        'collapse_rate': (n_bad / n_tested if n_tested else 0.0),
        'distal_vel_ratio_median': (float(np.median(ratios))
                                    if ratios else np.nan),
        'distal_vel_corr': dvc,
    }


def reach_verdict(agg):
    """(per-metric checks, all-pass). Undefined/nan reach => FAIL (conservative;
    never let an undefined reach silently pass the multi-topology gate)."""
    if agg is None:
        return {}, False
    checks = {}
    for k, (o, thr) in REACH_THRESH.items():
        v = agg.get(k)
        if v is None or not np.isfinite(v):
            checks[k] = False
        elif o == '<':
            checks[k] = bool(v < thr)
        elif o == '>':
            checks[k] = bool(v > thr)
        else:
            lo, hi = thr
            checks[k] = bool(lo < v < hi)
    return checks, bool(checks and all(checks.values()))


def _san(o):
    """Recursively replace non-finite floats with None (json allow_nan=False)."""
    if isinstance(o, dict):
        return {k: _san(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_san(v) for v in o]
    if isinstance(o, float) and not np.isfinite(o):
        return None
    return o


def _rot_angle_deg(pr6, ir6):
    """codex ROUTING ④: per-element geodesic-SO(3) angle in DEGREES between
    predicted and IK 6D rotations (same SO(3) metric as the training loss;
    head-to-head vs MoCapAnything V2 unseen 6.54deg). pr6/ir6 [...,6]."""
    from src.models.treeik_decoder import rot6d_to_matrix
    Rp = rot6d_to_matrix(pr6)
    Rt = rot6d_to_matrix(ir6)
    rel = Rp.transpose(-1, -2) @ Rt
    tr = rel[..., 0, 0] + rel[..., 1, 1] + rel[..., 2, 2]
    cos = ((tr - 1.0) / 2.0).clamp(-1.0 + 1e-6, 1.0 - 1e-6)
    return torch.rad2deg(torch.acos(cos))


# =========================================================================== #
# main()
# =========================================================================== #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--src_dir', default='data/cs_sparse2full_tgt',
                    help='source dataset (noKslot baseline: same as tgt)')
    ap.add_argument('--tgt_dir', default='data/cs_sparse2full_tgt')
    ap.add_argument('--ik_dir', default='data/cs_sparse2full_ik_rot',
                    help='IK rot targets dir for rot_l1_deg reporting')
    ap.add_argument('--max_frames', type=int, default=196)
    ap.add_argument('--max_joints', type=int, default=160)
    ap.add_argument('--n_orig', type=int, default=22,
                    help='source orig joint count (subdiv-only metrics; N/A '
                         'for noKslot baseline but kept for verbatim ports)')
    ap.add_argument('--out', default=None)
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--split', default='val',
                    help="dataset split for src/tgt (default 'val')")
    ap.add_argument('--species', default='Bat,Crab,Horse',
                    help='comma-list of target species to keep; default '
                         '== noKslot baseline triad. Source default was '
                         "'Dragon,Spider,Trex' for the K-slot held-out "
                         'protocol, which does not apply to noKslot baseline '
                         '(same-skeleton self-recon).')
    ap.add_argument('--n_per', type=int, default=3,
                    help='animation chain: n clips per species '
                         '(forwarded to scripts/animate.py)')
    ap.add_argument('--no_animate', action='store_true',
                    help='skip auto-chained animate.py (visual-QA-primacy '
                         'rule says NOT to skip; this flag is provided for '
                         'fast metric-only iteration only)')
    args = ap.parse_args()

    dev = torch.device(args.device)
    out_dir = Path(args.out or Path(args.ckpt).parent / 'eval_out')
    out_dir.mkdir(parents=True, exist_ok=True)

    mk = lambda d, sp: UnifiedMotionDataset(
        [d], sp, max_frames=args.max_frames,
        max_joints=args.max_joints, normalize=False)
    src_va = mk(args.src_dir, args.split)
    tgt_va = mk(args.tgt_dir, args.split)
    assert len(src_va) == len(tgt_va), 'pair len mismatch'

    # ----- species filter -----
    _keep_species = {x.strip() for x in args.species.split(',') if x.strip()}
    _keep_idx = [i for i in range(len(tgt_va))
                 if str(tgt_va.samples[i].get('skeleton_id', '?'))
                 in _keep_species]
    if len(_keep_idx) != len(tgt_va):
        src_va.samples = [src_va.samples[i] for i in _keep_idx]
        tgt_va.samples = [tgt_va.samples[i] for i in _keep_idx]
        assert len(src_va) == len(tgt_va)

    # ----- load ckpt (tolerant: source baseline may have slot_assignment.*
    # + fk_state_dict: None; new ckpt has neither) -----
    ck = torch.load(args.ckpt, map_location=dev, weights_only=False)
    ma = ck['args']; ma = ma if isinstance(ma, dict) else vars(ma)
    init_ckpt_path = ma.get('init_ckpt',
                            'runs/L6_anchor_h100_seed42/best_model.pt')
    base = torch.load(init_ckpt_path, map_location=dev, weights_only=False)
    bm = base['args']; bm = bm if isinstance(bm, dict) else vars(bm)
    model = NoKslotModel(
        d_model=bm['d_model'], n_heads=bm['n_heads'], d_ff=bm['d_ff'],
        n_graph_layers=bm['n_graph_layers'],
        n_enc_temporal_layers=bm['n_enc_temporal_layers'],
        n_cross_layers=bm['n_cross_layers'],
        n_dec_temporal_layers=bm['n_dec_temporal_layers'],
        temporal_kernel=bm.get('temporal_kernel', 9), dropout=0.0).to(dev)
    load_result = model.load_state_dict(ck['model_state_dict'], strict=False)
    unexp_slot = [k for k in load_result.unexpected_keys
                  if k.startswith('slot_assignment.')]
    unexp_other = [k for k in load_result.unexpected_keys
                   if not k.startswith('slot_assignment.')]
    print(f'eval ckpt load: {len(unexp_slot)} slot_assignment.* dropped '
          f'(expected for source-produced ckpt), {len(unexp_other)} other '
          f'unexpected: {unexp_other}, missing: {load_result.missing_keys}',
          flush=True)
    if unexp_other or load_result.missing_keys:
        raise SystemExit(
            f'eval ckpt schema mismatch (non-slot_assignment drift): '
            f'{load_result}')
    model.encoder.use_name_embed = bool(bm.get('use_name_embed', True))
    model.eval()

    topofk = TopoFKTreeIKDecoder(model.decoder, bm['d_model']).to(dev)
    topofk.load_state_dict(ck['topofk_state_dict'])
    topofk.eval()

    # ----- accumulators -----
    pe, op, mp, blr, er, ve, va, vc_num, vc_d1, vc_d2, ae = (
        [] for _ in range(11))
    ext_list = []
    reach_all = []
    rot_all = []; rot_sm_all = []
    per_sp = defaultdict(
        lambda: {'pe': [], 'ext': [], 'blr': [], 'er': [], 'vn': [],
                 'vd1': [], 'vd2': [], 'reach': [], 'rot': []})
    sp_render = defaultdict(int)
    n = len(src_va)

    with torch.no_grad():
        for i in range(n):
            s = to_dev(collate_fn([src_va[i]]), dev)
            t = to_dev(collate_fn([tgt_va[i]]), dev)
            slot, s_j, asg = encode_decode_nok(model, s, t)
            parents_list = [[int(x) for x in pl]
                            for pl in t['parent_indices']]
            pred, pred_r6 = topofk(
                slot, s_j, asg, t['joint_mask'], s['frame_mask'],
                parents_list, t['rest_offsets'].to(dev), fps_of(t),
                t['adjacency'], t['geodesic_dist'], return_rot=True)

            gt = t['motion_features']
            jm = t['joint_mask'].float(); fm = t['frame_mask'].float()
            Jv = int(jm[0].sum().item()); Tv = int(fm[0].sum().item())
            pp = pred[0, :Tv, :Jv, :3].cpu().numpy()
            gp = gt[0, :Tv, :Jv, :3].cpu().numpy()
            pv = pred[0, :Tv, :Jv, 3:].cpu().numpy()
            gv = gt[0, :Tv, :Jv, 3:].cpu().numpy()
            pe.append(np.sqrt(((pp - gp) ** 2).sum(-1).mean()))
            ext_list.append(float(np.abs(gp).max()))
            no = min(args.n_orig, Jv)
            op.append(np.sqrt(((pp[:, :no] - gp[:, :no]) ** 2).sum(-1).mean()))
            if Jv > no:
                mp.append(np.sqrt(
                    ((pp[:, no:] - gp[:, no:]) ** 2).sum(-1).mean()))
            par = [int(x) for x in t['parent_indices'][0]][:Jv]
            rest = t['rest_offsets'][0].cpu().numpy()[:Jv]
            for j, pj in enumerate(par):
                if pj < 0:
                    continue
                rl = float(np.linalg.norm(rest[j]))
                if rl < 0.005:
                    continue
                pl = np.linalg.norm(pp[:, j] - pp[:, pj], axis=-1)
                blr.append(np.abs(pl - rl) / rl * 100.0)
                er.append(pl / rl)
            ve.append(((pv - gv) ** 2).mean()); va.append((gv ** 2).mean())
            vc_num.append(
                (pv.flatten() - pv.mean()) @ (gv.flatten() - gv.mean())
                if pv.size == gv.size else 0.0)
            vc_d1.append(np.linalg.norm(pv.flatten() - pv.mean()))
            vc_d2.append(np.linalg.norm(gv.flatten() - gv.mean()))
            fps_t = fps_of(t)
            pa = np.zeros_like(pv); ga = np.zeros_like(gv)
            if pv.shape[0] > 1:
                pa[1:] = (pv[1:] - pv[:-1]) * fps_t
                ga[1:] = (gv[1:] - gv[:-1]) * fps_t
                pa[0] = pa[1]; ga[0] = ga[1]
            ae.append((((pa - ga) ** 2).mean(), (ga ** 2).mean()))

            sp = str(tgt_va.samples[i].get('skeleton_id', '?'))
            P = per_sp[sp]
            P['pe'].append(np.sqrt(((pp - gp) ** 2).sum(-1).mean()))
            P['ext'].append(float(np.abs(gp).max()))
            for j, pj in enumerate(par):
                if pj < 0:
                    continue
                rl = float(np.linalg.norm(rest[j]))
                if rl < 0.005:
                    continue
                pl = np.linalg.norm(pp[:, j] - pp[:, pj], axis=-1)
                P['blr'].append(np.abs(pl - rl) / rl * 100.0)
                P['er'].append(pl / rl)
            P['vn'].append(float((pv.flatten() - pv.mean())
                                 @ (gv.flatten() - gv.mean())))
            P['vd1'].append(float(np.linalg.norm(pv.flatten() - pv.mean())))
            P['vd2'].append(float(np.linalg.norm(gv.flatten() - gv.mean())))

            # reach metrics — trivially empty for src==tgt same-skeleton
            # baseline, but compute + record for diagnostic completeness.
            tgt_cn = [str(x) for x in t['canonical_names'][0]][:Jv]
            src_cn = [str(x) for x in s['canonical_names'][0]]
            if len(tgt_cn) == Jv:
                rm = reach_metrics(pp, gp, pv, gv, par, tgt_cn, src_cn)
            else:
                rm = _empty_reach(
                    {'name_len_mismatch': [len(tgt_cn), int(Jv)]})
            rm['skeleton_id'] = sp; rm['clip'] = i
            P['reach'].append(rm)
            reach_all.append(rm)

            # rot_l1_deg vs offline IK targets (always available for TreeIK)
            base_name = os.path.basename(
                str(tgt_va.samples[i]['motion_path']))
            ikp = os.path.join(args.ik_dir, base_name)
            if os.path.exists(ikp):
                dik = np.load(ikp, allow_pickle=True)
                ir = dik['ik_rot6d'].astype(np.float32)
                Tvi, Jvi = int(ir.shape[0]), int(ir.shape[1])
                if Tvi == Tv and Jvi == Jv:
                    nl = np.zeros(Jv, bool)
                    for pj in par:
                        if 0 <= pj < Jv:
                            nl[pj] = True
                    if nl.any():
                        pr = pred_r6[0, :Tv, :Jv]
                        it = torch.from_numpy(ir).to(pr.device)
                        ang = _rot_angle_deg(pr, it)
                        ang = ang[:, nl].detach().cpu().numpy()
                        rl1 = float(ang.mean())
                        sm = (float(np.abs(ang[1:] - ang[:-1]).mean())
                              if ang.shape[0] > 1 else 0.0)
                        rot_all.append(rl1); rot_sm_all.append(sm)
                        P['rot'].append(rl1)

            if sp_render[sp] < 3:
                _render(pp, gp, par,
                        out_dir / f'{sp}_val{sp_render[sp]}_pred_vs_gt.png',
                        sp_render[sp])
                sp_render[sp] += 1

    # ----- aggregate metrics (verbatim aggregation logic from source) -----
    blr_arr = np.concatenate(blr) if blr else np.array([np.nan])
    er_arr = np.concatenate(er) if er else np.array([np.nan])
    ve_s, va_s = float(np.mean(ve)), float(np.mean(va))
    ae_s = float(np.mean([a for a, _ in ae]))
    ag_s = float(np.mean([g for _, g in ae]))
    pos_rmse_abs = float(np.mean(pe))
    ext_mean = float(np.mean(ext_list)) if ext_list else 1.0
    metrics = {
        'pos_rmse_m': pos_rmse_abs,
        'pos_nrmse_extent': float(pos_rmse_abs / max(ext_mean, 1e-6)),
        'target_extent_mean_m': ext_mean,
        # subdiv-only orig/midpoint kept as None for noKslot baseline schema
        # compatibility with source gate_verdict.json.
        'orig_joint_rmse_m': None,
        'midpoint_rmse_m': None,
        'bone_len_rel_mean_pct': float(np.nanmean(blr_arr)),
        'bone_len_rel_p95_pct': float(np.nanpercentile(blr_arr, 95)),
        'edge_ratio_p05': float(np.nanpercentile(er_arr, 5)),
        'vel_nrmse': float(np.sqrt(ve_s / max(va_s, 1e-9))),
        'vel_corr': float(np.sum(vc_num) /
                          (np.sqrt(np.sum(np.square(vc_d1))) *
                           np.sqrt(np.sum(np.square(vc_d2))) + 1e-9)),
        'acc_nrmse': float(np.sqrt(ae_s / max(ag_s, 1e-9))),
        'rot_l1_deg': (float(np.mean(rot_all)) if rot_all else None),
        'rot_smooth_l1_deg': (float(np.mean(rot_sm_all))
                              if rot_sm_all else None),
        'rot_n_clips': len(rot_all),
    }
    THRESH = THRESH_NOKSLOT_BASELINE
    verdict = {}
    for k, (op_, thr) in THRESH.items():
        v = metrics.get(k)
        verdict[k] = None if v is None else (
            (v < thr) if op_ == '<' else (v > thr))
    passed = all(x for x in verdict.values() if x is not None)

    # gate_type label: keep source literal for gate_verdict.json schema
    # compatibility with the noKslot diagnostic baseline gate_verdict.json.
    gtype = 'sparse2full_crossspecies'

    # reach NOT enforced for noKslot same-skeleton baseline (trivially empty);
    # aggregate is still computed + reported as diagnostic record.
    enforce_reach = False

    per_species = {}
    for sp, P in per_sp.items():
        if not P['pe']:
            continue
        blrc = np.concatenate(P['blr']) if P['blr'] else np.array([np.nan])
        erc = np.concatenate(P['er']) if P['er'] else np.array([np.nan])
        pr_mean = float(np.mean(P['pe']))
        ex = float(np.mean(P['ext'])) or 1.0
        spm = {
            'n_clips': len(P['pe']),
            'pos_nrmse_extent': pr_mean / max(ex, 1e-6),
            'bone_len_rel_mean_pct': float(np.nanmean(blrc)),
            'bone_len_rel_p95_pct': float(np.nanpercentile(blrc, 95)),
            'edge_ratio_p05': float(np.nanpercentile(erc, 5)),
            'vel_corr': float(np.sum(P['vn']) /
                              (np.sqrt(np.sum(np.square(P['vd1'])))
                               * np.sqrt(np.sum(np.square(P['vd2'])))
                               + 1e-9)),
            'rot_l1_deg': (float(np.mean(P['rot'])) if P['rot'] else None),
        }
        spv = {k: ((spm[k] < t) if o == '<' else (spm[k] > t))
               for k, (o, t) in THRESH.items() if k in spm}
        ragg = reach_aggregate(P['reach'])
        rchecks, rpass = reach_verdict(ragg)
        all_pass = bool(all(spv.values())
                        and (rpass if enforce_reach else True))
        per_species[sp] = {'metrics': spm, 'per_metric_pass': spv,
                           'reach': _san(ragg), 'reach_per_metric': rchecks,
                           'reach_pass': bool(rpass),
                           'all_pass': all_pass}

    reach_overall = reach_aggregate(reach_all)
    rchecks_o, rpass_o = reach_verdict(reach_overall)
    reach_pass_all_species = (
        bool(per_species and all(v['reach_pass']
                                 for v in per_species.values()))
        if enforce_reach else None)

    # ----- auto-chained animation QA (visual-QA-primacy hard gate unless
    # --no_animate). Delegates to scripts/animate.py. -----
    enforce_anim = not args.no_animate
    anim_ok, anim_info = True, 'not_enforced'
    if enforce_anim:
        anim_dir = out_dir / 'qa_animate'
        cmd = [sys.executable, '-u',
               str(PROJECT_ROOT / 'scripts' / 'animate.py'),
               '--ckpt', str(args.ckpt),
               '--src_dir', args.src_dir, '--tgt_dir', args.tgt_dir,
               '--max_frames', str(args.max_frames),
               '--max_joints', str(args.max_joints),
               '--split', args.split,
               '--species', args.species,
               '--n_per', str(args.n_per),
               '--stride', '1',
               '--out', str(anim_dir),
               '--device', args.device]
        print(f'\n=== FORCED motion-animation QA (auto-chained, hard gate) '
              f'-> {anim_dir} ===', flush=True)
        rc = subprocess.run(cmd, cwd=str(PROJECT_ROOT)).returncode
        need = {sp.strip(): args.n_per for sp in args.species.split(',')
                if sp.strip()}
        have = {sp: len(list(anim_dir.glob(f'{sp}_clip*_gtvspred.gif')))
                for sp in need}
        sheet_obl = {sp: len(list(
            anim_dir.glob(f'{sp}_clip*_sheet_obl.png'))) for sp in need}
        sheet_top = {sp: len(list(
            anim_dir.glob(f'{sp}_clip*_sheet_top.png'))) for sp in need}
        gif_ok = all(have[sp] >= need[sp] for sp in need)
        png_ok = all(sheet_obl[sp] >= need[sp]
                     and sheet_top[sp] >= need[sp] for sp in need)
        cover_ok = bool(gif_ok and png_ok)
        anim_ok = (rc == 0 and cover_ok
                   and (anim_dir / 'animate_summary.txt').exists())
        anim_info = {'returncode': rc, 'gifs_per_species': have,
                     'sheet_obl_per_species': sheet_obl,
                     'sheet_top_per_species': sheet_top,
                     'required_per_species': need,
                     'summary_exists':
                         (anim_dir / 'animate_summary.txt').exists(),
                     'gif_coverage_ok': bool(gif_ok),
                     'multiview_png_coverage_ok': bool(png_ok),
                     'coverage_ok': bool(cover_ok)}
        if not anim_ok:
            print(f'=== ANIMATION QA HARD-FAIL (gate cannot PASS): '
                  f'{anim_info} ===', flush=True)

    gate_pass_all = bool(passed and per_species
                         and all(v['all_pass']
                                 for v in per_species.values())
                         and anim_ok)
    result = {
        'ckpt': args.ckpt, 'head': 'topofk_treeik', 'no_k_slot': True,
        'gate_type': gtype,
        'n_val': n, 'metrics': metrics,
        'thresholds': {k: f'{o}{t}' for k, (o, t) in THRESH.items()},
        'per_metric_pass': verdict,
        'reach_metrics_aggregate': _san(reach_overall),
        'reach_per_metric_aggregate': rchecks_o,
        'reach_thresholds': REACH_THRESH_STR,
        'reach_enforced': enforce_reach,
        'per_species': per_species,
        'animation_qa_enforced': enforce_anim,
        'animation_qa_ok': bool(anim_ok),
        'animation_qa_info': anim_info,
        'GATE_PASS': bool(passed),
        'GATE_PASS_all_species': gate_pass_all,
        'REACH_PASS_all_species': reach_pass_all_species,
    }
    json.dump(result, open(out_dir / 'gate_verdict.json', 'w'),
              indent=2, allow_nan=False)
    json.dump([_san(r) for r in reach_all],
              open(out_dir / 'reach_detail.json', 'w'), indent=2,
              allow_nan=False)
    print(json.dumps(result, indent=2))
    gpa = result['GATE_PASS_all_species']
    rpa = result['REACH_PASS_all_species']
    print(f'\n=== GATE agg={"PASS" if passed else "FAIL"} | '
          f'all_species={"PASS" if gpa else "FAIL"} | '
          f'reach_all_species={rpa} === -> {out_dir}/gate_verdict.json')


# =========================================================================== #
# _render helper — verbatim from source eval_paired_gate.py:755-768.
# =========================================================================== #
def _render(pp, gp, parents, path, idx):
    fi = pp.shape[0] // 2
    fig = plt.figure(figsize=(11, 5.5))
    for col, (P, name, c) in enumerate([(gp[fi], 'GT target', '#e74c3c'),
                                        (pp[fi], 'PRED target', '#2980b9')]):
        ax = fig.add_subplot(1, 2, col + 1, projection='3d')
        for j, pj in enumerate(parents):
            if 0 <= pj < P.shape[0]:
                ax.plot3D([P[j, 0], P[pj, 0]], [P[j, 2], P[pj, 2]],
                          [P[j, 1], P[pj, 1]], color='#888', lw=2)
        ax.scatter3D(P[:, 0], P[:, 2], P[:, 1], c=c, s=14)
        ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
        ax.set_title(f'val{idx} {name} f{fi}'); ax.view_init(12, -70)
    plt.savefig(path, dpi=70, bbox_inches='tight'); plt.close(fig)


if __name__ == '__main__':
    main()
