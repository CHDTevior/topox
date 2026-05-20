"""noKslot_clean / scripts/self_test.py — CPU-only invariant smoke test for
the no_k_slot path. Independent entry point (callable as
`python scripts/self_test.py` or via `python scripts/train.py --self_test`).

Adapted from source motion_representation_study/scripts/train_paired_gate.py:243-422
(_nokslot_self_test). Two test groups kept; one test dropped (see below).

KEPT (verbatim source spec):
  (1) IDENTITY UNPOOL — the EXISTING decoder's first op
      `einsum('bjk,btkd->btjd', A_id, slot)` recovers slot
      (== slot_norm(h_tj) on the no_k_slot path) BITWISE on valid joints
      and 0 on padded joints. Prove on synthetic CPU tensors AND on real
      Bat/Crab/Horse same-skeleton self-recon batches from
      data/cs_sparse2full_tgt (when available — SKIPPED if absent).

DROPPED (N/A in noKslot_clean):
  (2) "SlotAssignment params have grad=None on noK path" + forward-hook
      tripwire — this was a regression-protection invariant in source repo
      where SlotAssignment still existed and might accidentally rewire.
      noKslot_clean intentionally has NO SlotAssignment class anywhere, so
      there is nothing to register a forward hook on, nothing to track
      param grads for. The compile-time invariant (no SlotAssignment in
      NoKslotModel at all) is much stronger than the runtime tripwire.

Run examples:
  python -u scripts/self_test.py           # standalone entry
  python -u scripts/train.py --self_test   # forward from train.py
"""
import os
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.unified_dataset import UnifiedMotionDataset, collate_fn  # noqa: E402
from src.models.noKslot_model import NoKslotModel  # noqa: E402
from src.models.treeik_decoder import TopoFKTreeIKDecoder  # noqa: E402
from src.utils import recon_loss  # noqa: E402
from scripts.train import encode_decode_nok  # noqa: E402


def run_self_test():
    """CPU-only invariant smoke. Exits nonzero on any failure; prints a
    PASS report on success."""
    torch.manual_seed(0)
    dev = torch.device('cpu')
    D, B, T, J = 32, 2, 5, 7        # tiny; J incl. padded joints below
    model = NoKslotModel(d_model=D, n_heads=4, d_ff=4 * D,
                         n_graph_layers=2, n_enc_temporal_layers=1,
                         n_cross_layers=2, n_dec_temporal_layers=1,
                         motion_feat_dim=6, joint_feat_dim=9,
                         temporal_kernel=3, dropout=0.0).to(dev)
    model.encoder.use_name_embed = False

    # synthetic skeleton: single root at idx 0, strict parent-before-child
    # (TopoFK.validate_fk_tree precondition). Last 2 joints are PADDING
    # (joint_mask=False) so the masked-identity behaviour on pad is tested.
    n_real = J - 2
    parents = [-1, 0, 1, 0, 2]      # len == n_real, valid tree
    assert len(parents) == n_real
    jm = torch.zeros(B, J, dtype=torch.bool)
    jm[:, :n_real] = True
    fm = torch.ones(B, T, dtype=torch.bool)
    adj = torch.zeros(B, J, J)
    for j, pr in enumerate(parents):
        if pr >= 0:
            adj[:, j, pr] = 1.0
            adj[:, pr, j] = 1.0
    geo = torch.cdist(torch.arange(J).float().view(1, J, 1).expand(B, J, 1),
                      torch.arange(J).float().view(1, J, 1).expand(B, J, 1))
    batch = {
        'motion_features': torch.randn(B, T, J, 6),
        'skeleton_features': torch.randn(B, J, 9),
        'adjacency': adj,
        'geodesic_dist': geo,
        'joint_mask': jm,
        'frame_mask': fm,
        'rest_offsets': torch.randn(B, J, 3) * 0.3,
        'parent_indices': [list(parents) for _ in range(B)],
        'fps': torch.tensor(20.0),
    }
    src = tgt = batch                          # same-topo (in-distribution)

    # ---- (1) identity-unpool equality on synthetic batch -----------------
    slot, s_j, asg = encode_decode_nok(model, src, tgt)
    Js = src['joint_mask'].shape[1]
    assert slot.shape == (B, T, Js, D), f'slot {tuple(slot.shape)}'
    assert asg.shape == (B, Js, Js), f'A_id {tuple(asg.shape)}'
    # the EXACT first op MotionDecoder.forward performs on (asg, slot):
    with torch.no_grad():
        unpool = torch.einsum('bjk,btkd->btjd', asg, slot)  # [B,T,Js,D]
        valid = src['joint_mask'][:, None, :, None].to(slot.dtype)
        d_valid = float(((unpool - slot).abs() * valid).max())
        d_pad = float(((unpool * (1.0 - valid)).abs()).max())
    if d_valid != 0.0:
        raise SystemExit(
            f'NOKSLOT SELF-TEST FAIL: identity unpool does NOT recover '
            f'h_tj on valid joints (max|Δ|={d_valid:.3e}, need exactly 0)')
    if d_pad != 0.0:
        raise SystemExit(
            f'NOKSLOT SELF-TEST FAIL: identity unpool leaks onto PADDED '
            f'joints (max|val|={d_pad:.3e}, need exactly 0)')

    # ---- (1b) full forward through TopoFKTreeIK + recon backward sanity --
    # Replaces source test (2) tripwire on SlotAssignment (which doesn't
    # exist in noKslot_clean). This runs the FULL forward+backward path the
    # actual training loop takes, on synthetic data. Catches any wiring
    # regression that would silently break training (e.g. shape mismatch
    # in fk_persample / TopoFKTreeIKDecoder.forward / recon_loss).
    topofk = TopoFKTreeIKDecoder(model.decoder, D, n_layers=2,
                                 n_heads=4).to(dev)
    for p in model.parameters():
        p.grad = None
    slot, s_j, asg = encode_decode_nok(model, src, tgt)
    par = [[int(x) for x in pl] for pl in tgt['parent_indices']]
    pred, _r6 = topofk(slot, s_j, asg, tgt['joint_mask'],
                       src['frame_mask'], par,
                       tgt['rest_offsets'], 20.0,
                       tgt['adjacency'], tgt['geodesic_dist'],
                       return_rot=True)
    loss = recon_loss(pred, tgt)
    loss.backward()
    if torch.isnan(loss) or torch.isinf(loss):
        raise SystemExit(
            f'NOKSLOT SELF-TEST FAIL: synthetic forward produced '
            f'non-finite loss ({loss.item():.6f})')

    # ---- (2) REAL-DATA CPU smoke (covers actual cs_sparse2full_tgt data
    # path; verbatim spec from source 358-414; SKIPPED if data dir absent).
    _NOK_DIR = 'data/cs_sparse2full_tgt'
    _smoke = 'SKIPPED (data dir absent)'
    if os.path.isdir(os.path.join(_NOK_DIR, 'motions')):
        _MJ = 160
        _checked = []
        for _sp in ('Bat', 'Crab', 'Horse'):
            _ds = UnifiedMotionDataset([_NOK_DIR], 'train', max_frames=64,
                                       max_joints=_MJ, normalize=False)
            _ds.samples = [s for s in _ds.samples
                           if str(s.get('skeleton_id', '')) == _sp][:2]
            if not _ds.samples:
                continue
            # src_dir == tgt_dir == cs_sparse2full_tgt -> SAME file per
            # index -> src joint_mask MUST equal tgt joint_mask bitwise.
            _b = collate_fn([_ds[i] for i in range(len(_ds.samples))])
            _sm, _tm = _b['joint_mask'], _b['joint_mask']
            if not torch.equal(_sm.bool(), _tm.bool()):
                raise SystemExit(
                    f'NOKSLOT SELF-TEST FAIL (real-data): {_sp} src '
                    f'joint_mask != tgt joint_mask on the same-topo '
                    f'cs_sparse2full_tgt config (src_dir==tgt_dir).')
            _Jv = int(_sm[0].bool().sum())
            # real data path: encoder -> slot_norm -> masked-identity A_id.
            _src = {k: (v if torch.is_tensor(v) else v)
                    for k, v in _b.items()}
            with torch.no_grad():
                _slot, _sj, _asg = encode_decode_nok(model, _src, _src)
                _Js = _src['joint_mask'].shape[1]
                _unp = torch.einsum('bjk,btkd->btjd', _asg, _slot)
                _vm = _src['joint_mask'][:, None, :, None].to(_slot.dtype)
                _dv = float(((_unp - _slot).abs() * _vm).max())
                _dp = float(((_unp * (1.0 - _vm)).abs()).max())
            if _dv != 0.0 or _dp != 0.0:
                raise SystemExit(
                    f'NOKSLOT SELF-TEST FAIL (real-data): {_sp} identity '
                    f'unpool on REAL data does NOT recover slot_norm(h_tj) '
                    f'bitwise (max|Δ|valid={_dv:.3e}, '
                    f'max|val|pad={_dp:.3e}; need exactly 0/0).')
            _checked.append(f'{_sp}(J={_Jv})')
        if _checked:
            _smoke = ('PASS on real ' + ','.join(_checked)
                      + ' from ' + _NOK_DIR
                      + ' (src_dir==tgt_dir => src_mask==tgt_mask bitwise; '
                      + 'identity unpool recovers slot_norm(h_tj) bitwise '
                      + 'max|Δ|=0 on real data path)')
        else:
            _smoke = 'SKIPPED (no Bat/Crab/Horse train samples found)'

    print(f'NOKSLOT SELF-TEST PASS (CPU): '
          f'(1) synthetic identity unpool recovers h_tj BITWISE on valid '
          f'joints (max|Δ|={d_valid:.1e}) & 0 on padded '
          f'(max|val|={d_pad:.1e}); (1b) full forward + recon backward on '
          f'synthetic batch produced finite loss '
          f'({float(loss.item()):.6f}); (2) REAL-DATA smoke: {_smoke}. '
          f'NON-DECISIVE diagnostic.', flush=True)


if __name__ == '__main__':
    run_self_test()
