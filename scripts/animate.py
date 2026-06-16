"""noKslot_clean / scripts/animate.py — single-path visual QA for the noKslot
baseline. Renders per held-out species, side-by-side GT|PRED 3D skeleton
ANIMATIONS (gif) + dual-view contact-sheet PNGs (oblique + top-down) so
frozen / pinned / collapsed / jittery surplus DOF are visible against the
same scale.

Source: motion_representation_study/scripts/_qa_gate_animate.py (322 lines).
This version drops K-slot-only branching (no --head arg, locked to
topofk_treeik; no ckpt_args.no_k_slot sniff, always True) and uses
Model + strict=False ckpt load (tolerant of source-produced baseline
ckpts with slot_assignment.* keys + fk_state_dict: None).

Helpers (animate_clip / contact_sheet) are VERBATIM from source 128-202.
Rendering invariants preserved:
  - fixed equal cubic axis limits from GT extent (collapse/scale drift
    visible against same scale)
  - on-figure per-frame displacement readout (frozen PRED => disp~0 while
    GT moves)
  - dual camera views (oblique 12/-70 + top-down 75/-90) so depth-axis
    collapse a single view hides is exposed

Run example:
  python -u scripts/animate.py \\
      --ckpt runs/baseline_noKslot_ep399/last_model.pt \\
      --src_dir data/cs_sparse2full_tgt \\
      --tgt_dir data/cs_sparse2full_tgt \\
      --split val --species Bat,Crab,Horse \\
      --n_per 3 --stride 1 \\
      --out runs/eval_baseline_noKslot_ep399/qa_animate
"""
import argparse
import os
import sys
from pathlib import Path

import matplotlib
import numpy as np
import torch

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.animation import FuncAnimation, PillowWriter  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)   # self-anchor: ckpt args store RELATIVE init_ckpt;
                         # robust to srun cwd=home (§8C trap) — never depend
                         # on caller cwd for this QA script.

from src.data.unified_dataset import UnifiedMotionDataset, collate_fn  # noqa: E402
from src.models.model import Model  # noqa: E402
from src.models.treeik_decoder import TopoFKTreeIKDecoder  # noqa: E402
from src.utils import fps_of, to_dev  # noqa: E402
from scripts.train import encode_decode  # noqa: E402
from scripts.eval import reach_metrics  # noqa: E402  same reach logic as eval


def load_model_head(ckpt_path, dev):
    """noKslot single-path model+head load. Tolerant of source-produced
    baseline ckpt that has slot_assignment.* keys (strict=False) and may
    have fk_state_dict: None (ignored)."""
    ck = torch.load(ckpt_path, map_location=dev, weights_only=False)
    ma = ck['args']; ma = ma if isinstance(ma, dict) else vars(ma)
    base = torch.load(ma.get('init_ckpt',
                              'runs/L6_anchor_h100_seed42/best_model.pt'),
                      map_location=dev, weights_only=False)
    bm = base['args']; bm = bm if isinstance(bm, dict) else vars(bm)
    model = Model(
        d_model=bm['d_model'], n_heads=bm['n_heads'], d_ff=bm['d_ff'],
        n_graph_layers=bm['n_graph_layers'],
        n_enc_temporal_layers=bm['n_enc_temporal_layers'],
        n_cross_layers=bm['n_cross_layers'],
        n_dec_temporal_layers=bm['n_dec_temporal_layers'],
        temporal_kernel=bm.get('temporal_kernel', 9),
        dropout=0.0).to(dev)
    load_result = model.load_state_dict(ck['model_state_dict'], strict=False)
    unexp_other = [k for k in load_result.unexpected_keys
                   if not k.startswith('slot_assignment.')]
    if unexp_other or load_result.missing_keys:
        raise SystemExit(
            f'animate ckpt schema mismatch: {load_result}')
    model.encoder.use_name_embed = bool(bm.get('use_name_embed', True))
    model.eval()
    topofk = TopoFKTreeIKDecoder(model.decoder, bm['d_model']).to(dev)
    topofk.load_state_dict(ck['topofk_state_dict'])
    topofk.eval()
    return model, topofk


def forward_clip(model, topofk, s, t, dev):
    """Single-path forward for noKslot animation: encode_decode +
    TopoFKTreeIKDecoder. Mirrors scripts/eval.py forward exactly so the
    animation shows the SAME predictions the gate scores."""
    slot, s_j, asg = encode_decode(model, s, t)
    parents_list = [[int(x) for x in pl] for pl in t['parent_indices']]
    pred = topofk(
        slot, s_j, asg, t['joint_mask'], s['frame_mask'], parents_list,
        t['rest_offsets'].to(dev), fps_of(t),
        t['adjacency'], t['geodesic_dist'])
    return pred


# =========================================================================== #
# Rendering helpers — VERBATIM from source _qa_gate_animate.py:128-202.
# =========================================================================== #
def animate_clip(pp, gp, par, path, title, stride, fps):
    """Side-by-side GT|PRED 3D skeleton gif. Fixed equal axes from GT extent
    (collapse/explosion shows against the same scale). On-figure per-frame
    displacement (vs frame0) for GT & PRED — frozen pred => PRED≈0 vs GT>0."""
    T = pp.shape[0]
    idx = list(range(0, T, max(stride, 1)))
    if idx[-1] != T - 1:
        idx.append(T - 1)
    # equal cubic axis limits from GT over the whole clip
    allc = gp.reshape(-1, 3)
    ctr = allc.mean(0)
    rad = max(float(np.abs(allc - ctr).max()), 1e-3) * 1.05
    g0 = gp[0]; p0 = pp[0]
    fig = plt.figure(figsize=(12, 6))
    axes = [fig.add_subplot(1, 2, k + 1, projection='3d') for k in range(2)]

    def draw(ax, P, name, col):
        ax.clear()
        for j, pj in enumerate(par):
            if 0 <= pj < P.shape[0] and j < P.shape[0]:
                ax.plot3D([P[j, 0], P[pj, 0]], [P[j, 2], P[pj, 2]],
                          [P[j, 1], P[pj, 1]], color='#888', lw=1.4)
        ax.scatter3D(P[:, 0], P[:, 2], P[:, 1], c=col, s=12)
        ax.set_xlim(ctr[0] - rad, ctr[0] + rad)
        ax.set_ylim(ctr[2] - rad, ctr[2] + rad)
        ax.set_zlim(ctr[1] - rad, ctr[1] + rad)
        ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
        ax.view_init(12, -70)
        ax.set_title(name, fontsize=10)

    def update(fi):
        gd = float(np.linalg.norm(gp[fi] - g0, axis=-1).mean())
        pd = float(np.linalg.norm(pp[fi] - p0, axis=-1).mean())
        draw(axes[0], gp[fi], f'GT  f{fi}  disp={gd:.3f}', '#e74c3c')
        draw(axes[1], pp[fi], f'PRED f{fi}  disp={pd:.3f}', '#2980b9')
        fig.suptitle(f'{title}  (GT disp vs PRED disp — frozen=>PRED~0)',
                     fontsize=11)
        return axes

    ani = FuncAnimation(fig, update, frames=idx, blit=False)
    ani.save(path, writer=PillowWriter(fps=fps))
    plt.close(fig)


def animate_clip_3col(gt_ric, pred_ric, pred_fk, par, path, title, stride, fps):
    """3-col GT_RIC | PRED_RIC | PRED_FK 3D skeleton gif for rot6d_fk QA.

    Root-relative framing: subtract each clip's joint-0 (root) XZ translation per
    frame, keep Y. This matches the dataset QA renderers
    (render_planet_zoo_clean_gifs.py / render_anytop_vlm_previews.py) and fixes the
    "tiny skeleton" problem of whole-clip extent — with the root's horizontal travel
    removed the skeleton stays centered and FILLS the box, so bone/limb detail is
    visible. Shared cubic axes from GT_RIC's root-relative extent (collapse/explosion
    shows against the same scale across all three panels).

    PRED_RIC = pred via the RIC/position route (ch0:3, world_geometry-supervised);
    PRED_FK  = pred via the rot6d-FK route (ch3:9 rotations -> FK). PRED_RIC vs
    PRED_FK reveals whether the predicted rotations, when FK'd, agree with the
    position route — the core rot6d_fk signal. (Root-relative framing trades absolute
    root-drift visibility for clear local pose; the on-figure per-frame disp still
    reflects pose change.)"""
    def root_rel(P):  # [T,J,3] -> subtract joint-0 XZ per frame, keep Y
        Q = np.array(P, dtype=float, copy=True)
        Q[:, :, 0] -= P[:, 0:1, 0]
        Q[:, :, 2] -= P[:, 0:1, 2]
        return Q
    g, r, f = root_rel(gt_ric), root_rel(pred_ric), root_rel(pred_fk)
    T = g.shape[0]
    idx = list(range(0, T, max(stride, 1)))
    if idx[-1] != T - 1:
        idx.append(T - 1)
    # cubic axis limits from GT_RIC root-relative BOUNDING BOX (bbox center + half of
    # the largest dim). Fills the box better than mean+max-dist — matches the
    # fit-to-bounds scaling of the reference dataset renderers
    # (render_planet_zoo_clean_gifs.py compute_transform). Cubic (single rad) keeps
    # equal aspect so the skeleton is not distorted; collapse/explosion still shows.
    allc = g.reshape(-1, 3)
    lo, hi = allc.min(0), allc.max(0)
    ctr = (lo + hi) * 0.5
    rad = max(float((hi - lo).max()) * 0.5, 1e-3) * 1.10
    g0, r0, f0 = g[0], r[0], f[0]
    fig = plt.figure(figsize=(18, 6))
    axes = [fig.add_subplot(1, 3, k + 1, projection='3d') for k in range(3)]

    def draw(ax, P, name, col):
        ax.clear()
        for j, pj in enumerate(par):
            if 0 <= pj < P.shape[0] and j < P.shape[0]:
                ax.plot3D([P[j, 0], P[pj, 0]], [P[j, 2], P[pj, 2]],
                          [P[j, 1], P[pj, 1]], color='#888', lw=1.4)
        ax.scatter3D(P[:, 0], P[:, 2], P[:, 1], c=col, s=12)
        ax.set_xlim(ctr[0] - rad, ctr[0] + rad)
        ax.set_ylim(ctr[2] - rad, ctr[2] + rad)
        ax.set_zlim(ctr[1] - rad, ctr[1] + rad)
        ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
        ax.view_init(12, -70)
        ax.set_title(name, fontsize=10)

    def update(fi):
        gd = float(np.linalg.norm(g[fi] - g0, axis=-1).mean())
        rd = float(np.linalg.norm(r[fi] - r0, axis=-1).mean())
        fd = float(np.linalg.norm(f[fi] - f0, axis=-1).mean())
        draw(axes[0], g[fi], f'GT_RIC f{fi} d={gd:.3f}', '#e74c3c')
        draw(axes[1], r[fi], f'PRED_RIC f{fi} d={rd:.3f}', '#2980b9')
        draw(axes[2], f[fi], f'PRED_FK f{fi} d={fd:.3f}', '#27ae60')
        fig.suptitle(f'{title}  (GT_RIC | PRED_RIC | PRED_FK, root-relative — RIC vs FK route)',
                     fontsize=11)
        return axes

    ani = FuncAnimation(fig, update, frames=idx, blit=False)
    ani.save(path, writer=PillowWriter(fps=fps))
    plt.close(fig)


def fk_rest_pose(rest_offsets, parents):
    """T-pose world positions via FK at identity rotations.
    rest_offsets: [J, 3] local joint offsets in parent frame at rest.
    parents: list[int] length J; parents[0] = -1 (root).
    """
    J = len(parents)
    world_pos = np.zeros((J, 3), dtype=np.float32)
    world_pos[0] = rest_offsets[0]
    for j in range(1, J):
        p = parents[j]
        if p < 0:
            world_pos[j] = rest_offsets[j]
        else:
            world_pos[j] = world_pos[p] + rest_offsets[j]
    return world_pos


def animate_t2m_input_pred(pred, static_pose, par, path, prompt_text,
                            stride, fps, skeleton_label="input skeleton",
                            pred_fk=None, pred_label="pred pose/RIC",
                            pred_fk_label="pred rot6d-FK"):
    """T2M demo gif (per cross-project rule feedback_t2m_gif_layout):
    LEFT panel = static input skeleton (T-pose / rest pose, gray, no animation).
    MIDDLE panel = predicted motion via pose/RIC recovery (blue, animated).
    RIGHT panel (only if pred_fk given) = same motion via rot6d-FK recovery (green).
    TOP title = prompt text (wrapped, shown across all frames).

    The two pred panels compare the pose route (ch0:3 RIC) vs the rot6d-FK route
    (ch3:9 6D rotation → FK) of the SAME generated motion — they should match if
    the generated 6D rotations and local positions are self-consistent.

    NO GT panel — T2M inference takes only skeleton + prompt as input.

    Args:
        pred: [T, J, 3] predicted world positions (pose/RIC route)
        static_pose: [J, 3] static skeleton pose (e.g. T-pose from rest_offsets)
        par: list[int] parents
        path: output gif path
        prompt_text: caption string used as conditioning
        stride: frame subsample for gif
        fps: gif fps
        skeleton_label: text shown on the static skeleton panel
        pred_fk: [T, J, 3] world positions via rot6d-FK route, or None
                 (None → 2-panel legacy; given → 3-panel pose-vs-FK)
        pred_label / pred_fk_label: titles for the two pred panels
    """
    import textwrap
    T = pred.shape[0]
    idx = list(range(0, T, max(stride, 1)))
    if idx[-1] != T - 1:
        idx.append(T - 1)
    n_panel = 2 if pred_fk is None else 3
    # Equal cubic axis limits — UNION of all panels' extent so they share scale
    # and a frozen pred shows as ≈static.
    union_parts = [pred.reshape(-1, 3), static_pose.reshape(-1, 3)]
    if pred_fk is not None:
        union_parts.append(pred_fk.reshape(-1, 3))
    union = np.concatenate(union_parts, axis=0)
    ctr = union.mean(0)
    rad = max(float(np.abs(union - ctr).max()), 1e-3) * 1.05

    fig = plt.figure(figsize=(6 * n_panel, 6.5))
    axes = [fig.add_subplot(1, n_panel, k + 1, projection='3d') for k in range(n_panel)]

    def draw(ax, P, name, col):
        ax.clear()
        for j, pj in enumerate(par):
            if 0 <= pj < P.shape[0] and j < P.shape[0]:
                ax.plot3D([P[j, 0], P[pj, 0]], [P[j, 2], P[pj, 2]],
                          [P[j, 1], P[pj, 1]], color='#888', lw=1.4)
        ax.scatter3D(P[:, 0], P[:, 2], P[:, 1], c=col, s=12)
        ax.set_xlim(ctr[0] - rad, ctr[0] + rad)
        ax.set_ylim(ctr[2] - rad, ctr[2] + rad)
        ax.set_zlim(ctr[1] - rad, ctr[1] + rad)
        ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
        ax.view_init(12, -70)
        ax.set_title(name, fontsize=10)

    # Wrap long prompts so the suptitle stays readable.
    wrapped = "\n".join(textwrap.wrap(prompt_text or "(no prompt)", width=80))

    def update(fi):
        draw(axes[0], static_pose, skeleton_label, '#7f8c8d')          # gray, static
        draw(axes[1], pred[fi], f'{pred_label} f{fi}', '#2980b9')      # blue, pose
        if pred_fk is not None:
            draw(axes[2], pred_fk[fi], f'{pred_fk_label} f{fi}', '#27ae60')  # green, FK
        fig.suptitle(f'prompt: "{wrapped}"', fontsize=10, y=0.98)
        return axes

    ani = FuncAnimation(fig, update, frames=idx, blit=False)
    ani.save(path, writer=PillowWriter(fps=fps))
    plt.close(fig)


def contact_sheet(pp, gp, par, path, title, n_t=6, elev=12, azim=-70):
    """Read-able motion view: n_t evenly-spaced timepoints x (GT,PRED) grid in
    ONE static PNG. The Read tool only shows a gif's first frame, so this is
    how an agent/codex can actually SEE motion (frozen/collapse/jitter visible
    by comparing a joint's position DOWN the time-rows). Shared GT-extent axes."""
    T = pp.shape[0]
    ts = np.linspace(0, T - 1, min(n_t, T)).astype(int)
    allc = gp.reshape(-1, 3); ctr = allc.mean(0)
    rad = max(float(np.abs(allc - ctr).max()), 1e-3) * 1.05
    nr = len(ts)
    fig = plt.figure(figsize=(7, 2.6 * nr))
    for r, fi in enumerate(ts):
        for c, (P, nm, col) in enumerate([(gp[fi], 'GT', '#e74c3c'),
                                          (pp[fi], 'PRED', '#2980b9')]):
            ax = fig.add_subplot(nr, 2, r * 2 + c + 1, projection='3d')
            for j, pj in enumerate(par):
                if 0 <= pj < P.shape[0] and j < P.shape[0]:
                    ax.plot3D([P[j, 0], P[pj, 0]], [P[j, 2], P[pj, 2]],
                              [P[j, 1], P[pj, 1]], color='#999', lw=1.0)
            ax.scatter3D(P[:, 0], P[:, 2], P[:, 1], c=col, s=8)
            ax.set_xlim(ctr[0] - rad, ctr[0] + rad)
            ax.set_ylim(ctr[2] - rad, ctr[2] + rad)
            ax.set_zlim(ctr[1] - rad, ctr[1] + rad)
            ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
            ax.view_init(elev, azim)
            dsp = float(np.linalg.norm(P - gp[0], axis=-1).mean()) if c == 0 \
                else float(np.linalg.norm(P - pp[0], axis=-1).mean())
            ax.set_title(f'{nm} f{fi} d0={dsp:.3f}', fontsize=8)
    fig.suptitle(title + '  (motion = change DOWN the rows; PRED frozen if '
                 'rows identical while GT rows differ)', fontsize=9)
    plt.savefig(path, dpi=64, bbox_inches='tight'); plt.close(fig)


# =========================================================================== #
# main() — simplified single-path noKslot animation.
# =========================================================================== #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--src_dir', required=True)
    ap.add_argument('--tgt_dir', required=True)
    ap.add_argument('--max_frames', type=int, default=196)
    ap.add_argument('--max_joints', type=int, default=160)
    ap.add_argument('--species', default='Bat,Crab,Horse',
                    help='noKslot baseline triad; source default was '
                         "'Dragon,Spider,Trex' for K-slot held-out.")
    ap.add_argument('--split', default='val',
                    help="dataset split for src/tgt (default 'val')")
    ap.add_argument('--n_per', type=int, default=3)
    ap.add_argument('--stride', type=int, default=3,
                    help='frame subsample for gif')
    ap.add_argument('--fps', type=int, default=8)
    ap.add_argument('--out', required=True)
    ap.add_argument('--device', default='cuda')
    args = ap.parse_args()
    dev = torch.device(args.device)
    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    want = [x.strip() for x in args.species.split(',') if x.strip()]

    mk = lambda d: UnifiedMotionDataset(
        [d], args.split, max_frames=args.max_frames,
        max_joints=args.max_joints, normalize=False)
    src_va, tgt_va = mk(args.src_dir), mk(args.tgt_dir)
    assert len(src_va) == len(tgt_va), 'pair len mismatch'

    model, topofk = load_model_head(args.ckpt, dev)

    picked = {sp: 0 for sp in want}
    summary = []
    with torch.no_grad():
        for i in range(len(src_va)):
            sp = str(tgt_va.samples[i].get('skeleton_id', '?'))
            if sp not in picked or picked[sp] >= args.n_per:
                continue
            s = to_dev(collate_fn([src_va[i]]), dev)
            t = to_dev(collate_fn([tgt_va[i]]), dev)
            pred = forward_clip(model, topofk, s, t, dev)
            gt = t['motion_features']
            jm = t['joint_mask'].float(); fm = t['frame_mask'].float()
            Jv = int(jm[0].sum().item()); Tv = int(fm[0].sum().item())
            pp = pred[0, :Tv, :Jv, :3].cpu().numpy()
            gp = gt[0, :Tv, :Jv, :3].cpu().numpy()
            pv = pred[0, :Tv, :Jv, 3:].cpu().numpy()
            gv = gt[0, :Tv, :Jv, 3:].cpu().numpy()
            par = [int(x) for x in t['parent_indices'][0]][:Jv]
            # SAME reach logic as the gate — distal diagnostics (whole-skeleton
            # meanspeed can MASK frozen distal while torso/root moves; distal
            # is the actual reach failure).
            tgt_cn = [str(x) for x in t['canonical_names'][0]][:Jv]
            src_cn = [str(x) for x in s['canonical_names'][0]]
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
            ttl = (f'{sp} clip{k} [topofk_treeik+no_k_slot] '
                   f'Jv={Jv} Tv={Tv}  {diag}')
            animate_clip(pp, gp, par, str(path), ttl, args.stride, args.fps)
            # >=2 camera views (single view hides depth-axis collapse).
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
    # codex review#3: missing required visual coverage = HARD failure
    # (silently finishing on missing species/clips is a false-PASS risk).
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
