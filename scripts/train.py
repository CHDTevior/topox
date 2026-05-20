"""noKslot_clean / scripts/train.py — single-path training entry for the
no_k_slot reproducible baseline.

Trains: Model (encoder + slot_norm + decoder, NO SlotAssignment)
  + TopoFKTreeIKDecoder (TreeIK rot head + hard FK + IK rot supervision)
on the same-skeleton self-reconstruction task (src==tgt full->full).

Default hyperparameters are LOCKED to the noKslot diagnostic configuration
that produced runs/baseline_noKslot_ep399/last_model.pt — so re-running this
script with no CLI flags reproduces the baseline ckpt (modulo CUDA non-
determinism). See README.md "可复现基线" section for the full reproduction
contract.

Architecture is a surgical extract of the no_k_slot path from source
motion_representation_study/scripts/train_paired_gate.py:
  - encode_decode    : single-path version of source 191-231
                           (no_k_slot=True branch only; SlotAssignment K=24
                           Sinkhorn bottleneck is bypassed by masked-identity
                           assignment from _nok_identity_assignment).
  - _Composite           : forward routes encode + TopoFKTreeIKDecoder once,
                           DDP-wraps it once (shared decoder registered once).
  - main()               : argparse + DDP setup + paired dataset load + IK
                           retained filter + L6 init (strict=False) +
                           Model + TopoFKTreeIKDecoder instantiation +
                           freeze encoder.name_embedding + preflight (split
                           manifest + name policy + raw-rotation AST ban +
                           IK coverage + noK same-topo bitwise check) +
                           train loop (recon + vel-consistency + IK-rot
                           geodesic + acc-smoothness) + rank0 train-only
                           diagnostic + best/last ckpt save.

Run examples:
  Single-GPU (matches noKslot diagnostic single-GPU config):
    python -u scripts/train.py

  4-GPU DDP (replication / faster):
    torchrun --nnodes=1 --nproc_per_node=4 --rdzv_backend=c10d \\
        --rdzv_endpoint=127.0.0.1:29517 scripts/train.py \\
        --batch_size 1 --lr 4e-4 --epochs 1000

  CPU self-test (no GPU, no data needed):
    python -u scripts/train.py --self_test
"""

import argparse
import ast as _ast
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.unified_dataset import UnifiedMotionDataset, collate_fn  # noqa: E402
from src.models.model import Model  # noqa: E402
from src.models.treeik_decoder import (TopoFKTreeIKDecoder,  # noqa: E402
                                       rot_geodesic_loss)
from src.utils import (  # noqa: E402
    _ddp_abort_if_any_rank_failed,
    _ddp_barrier,
    _ddp_cleanup,
    _ddp_global_rank,
    _ddp_is_active,
    _ddp_is_main,
    _ddp_local_rank,
    _ddp_setup,
    _ddp_world_size,
    assert_name_policy,
    assert_no_crop_for_ik,
    build_non_leaf,
    fps_of,
    load_ik_batch,
    recon_loss,
    to_dev,
    write_preflight_manifest,
)


# =========================================================================== #
# no_k_slot bypass helpers — _nok_identity_assignment is verbatim from source
# train_paired_gate.py:173-188; encode_decode is a single-path
# simplification of source 191-231 keeping only the no_k_slot=True branch.
# =========================================================================== #
def _nok_identity_assignment(joint_mask):
    """codex NOKSLOT-DESIGN: build the joint_mask-masked IDENTITY assignment
    A_id [B,Jpad,Jpad] with K=Jpad. Row j of a valid joint is e_j; padded
    rows/cols are 0. Fed where the K=24 Sinkhorn `asg` [B,J,K] normally
    goes, so the EXISTING decoder einsum('bjk,btkd->btjd', A_id, h_tj)
    recovers h_tj exactly on valid joints (identity unpool) and the
    decoder/TopoFK path is reused VERBATIM. No SlotAssignment is involved:
    no learnable prototypes, no query/key projections, no Sinkhorn OT, no
    soft transport, no assignment losses."""
    B, Jpad = joint_mask.shape
    m = joint_mask.to(torch.float32)
    eye = torch.eye(Jpad, device=joint_mask.device,
                    dtype=torch.float32).unsqueeze(0)
    return eye * m[:, :, None] * m[:, None, :]


def encode_decode(model, src, tgt):
    """Single-path simplification of source train_paired_gate.py:191-231
    encode_decode(no_k_slot=True branch). Bypasses SlotAssignment entirely:
    encoder h_tj is normalized by slot_norm and a masked-identity assignment
    is constructed for use by the decoder einsum. Returns (slot, s_j, asg)
    suitable for TopoFKTreeIKDecoder.forward."""
    nh_s = src.get('name_hashes') if model.encoder.use_name_embed else None
    nh_t = tgt.get('name_hashes') if model.encoder.use_name_embed else None
    h_tj = model.encoder(
        src['motion_features'], src['skeleton_features'],
        src['adjacency'], src['geodesic_dist'],
        src['joint_mask'], src['frame_mask'], name_hashes=nh_s)
    slot = model.slot_norm(h_tj)                       # [B,T,Jpad,D]
    s_j = model.encoder.encode_skeleton(
        tgt['skeleton_features'], tgt['adjacency'],
        tgt['geodesic_dist'], tgt['joint_mask'], name_hashes=nh_t)
    asg = _nok_identity_assignment(src['joint_mask'])  # [B,Jpad,Jpad]
    return slot, s_j, asg


# =========================================================================== #
# Raw rotation AST guard — verbatim from source train_paired_gate.py:710-773,
# repointed at scripts/train.py via __file__.
# =========================================================================== #
def assert_no_raw_rotation_supervision():
    """The ONLY rotation target ever loaded is the offline IK ik_rot6d (via
    load_ik_batch in src/utils.py). batch['local_rotations_6d'] (raw GT
    rotations, present in cs_sparse2full_tgt motions) must NEVER be READ as
    supervision/oracle. We parse THIS module's AST and flag any ACTUAL
    value-access of the key 'local_rotations_6d': a subscript
    obj['local_rotations_6d'] or a .get('local_rotations_6d'...) /
    .pop(...) / .setdefault(...) call. Because we walk the parsed AST (not
    raw text), string literals inside docstrings/comments and this guard's
    own prose are NOT nodes and can never trigger it; a future edit that
    introduces a genuine read of the raw rotations IS a Subscript/Call node
    and aborts preflight before any GPU run. This guard's own access nodes
    are excluded by lineno so the tripwire never self-fails."""
    src = Path(__file__).read_text()
    tree = _ast.parse(src)
    KEY = 'local_rotations_6d'
    guard_lo = guard_hi = None
    for node in _ast.walk(tree):
        if (isinstance(node, _ast.FunctionDef)
                and node.name == 'assert_no_raw_rotation_supervision'):
            guard_lo = node.lineno
            guard_hi = getattr(node, 'end_lineno', node.lineno)
            break

    def _in_guard(n):
        ln = getattr(n, 'lineno', None)
        return (guard_lo is not None and ln is not None
                and guard_lo <= ln <= guard_hi)

    def _is_key_str(n):
        return isinstance(n, _ast.Constant) and n.value == KEY

    bad = []
    for node in _ast.walk(tree):
        if _in_guard(node):
            continue
        if isinstance(node, _ast.Subscript):
            sl = node.slice
            if isinstance(sl, _ast.Index):           # py<3.9 compat
                sl = sl.value
            if _is_key_str(sl):
                bad.append(('subscript', node.lineno))
        elif isinstance(node, _ast.Call):
            fn = node.func
            if (isinstance(fn, _ast.Attribute)
                    and fn.attr in ('get', 'pop', 'setdefault')
                    and node.args and _is_key_str(node.args[0])):
                bad.append((f'.{fn.attr}()', node.lineno))
    if bad:
        raise SystemExit(
            'PREFLIGHT ABORT (raw-rotation ban): scripts/train.py reads the '
            f"raw GT rotation key '{KEY}' as a possible target (AST nodes "
            f'{bad[:5]}) — only the offline IK ik_rot6d may supervise '
            f'rotation. Remove the raw-rotation read.')
    print('PREFLIGHT raw-rotation ban: AST scan found no '
          f"'{KEY}' value-access in scripts/train.py — OK", flush=True)


# =========================================================================== #
# main()
# =========================================================================== #
def main():
    p = argparse.ArgumentParser()
    # ----- data -----
    p.add_argument('--src_dir', default='data/cs_sparse2full_tgt',
                   help='source dataset (noKslot baseline: same as tgt for '
                        'same-skeleton self-recon)')
    p.add_argument('--tgt_dir', default='data/cs_sparse2full_tgt')
    p.add_argument('--ik_dir', default='data/cs_sparse2full_ik_rot',
                   help='offline IK rot targets dir (ik_rot6d npz + '
                        'retained_clips.txt)')
    # ----- init -----
    p.add_argument('--init_ckpt',
                   default='runs/L6_anchor_h100_seed42/best_model.pt',
                   help='L6 pre-trained init ckpt (loaded with strict=False; '
                        'slot_assignment.* keys intentionally dropped). Use '
                        '--from_scratch to skip.')
    p.add_argument('--from_scratch', action='store_true',
                   help='Random init instead of L6 fine-tune.')
    # ----- training (defaults LOCKED to noKslot reproducible baseline) -----
    p.add_argument('--epochs', type=int, default=400,
                   help='codex NOKSLOT-DESIGN predeclared budget (300-500 '
                        'range midpoint).')
    p.add_argument('--save_every', type=int, default=25,
                   help='save_every epochs + final epoch -> last_model.pt')
    p.add_argument('--lr', type=float, default=2e-4)
    p.add_argument('--batch_size', type=int, default=8,
                   help='per-GPU batch')
    p.add_argument('--max_frames', type=int, default=196)
    p.add_argument('--max_joints', type=int, default=160,
                   help='large zoo skeletons (up to ~150 joints)')
    p.add_argument('--seed', type=int, default=42)
    # ----- losses -----
    p.add_argument('--w_rot_ik', type=float, default=0.1,
                   help='IK-derived geodesic-SO(3) rot supervision (TreeIK ③)')
    p.add_argument('--w_acc', type=float, default=0.01,
                   help='Acceleration smoothness')
    p.add_argument('--w_vel_consistency', type=float, default=0.5,
                   help='Velocity-consistency between predicted pos '
                        'derivative and predicted vel channel')
    # ----- freeze -----
    p.add_argument('--freeze_name_embed', type=int, default=1,
                   help='1=freeze encoder.name_embedding (no_k_slot '
                        'equivalent of source --freeze_base_slot_name 1, '
                        'minus SlotAssignment freeze N/A here)')
    # ----- output -----
    p.add_argument('--out', default='runs/noKslot_baseline')
    p.add_argument('--device', default='cuda')
    p.add_argument('--self_test', action='store_true',
                   help='Run CPU self-test (no GPU, no data) and exit.')
    # ----- model defaults (only used when --from_scratch) -----
    p.add_argument('--d_model', type=int, default=256)
    p.add_argument('--n_heads', type=int, default=8)
    p.add_argument('--d_ff', type=int, default=1024)
    p.add_argument('--n_graph_layers', type=int, default=4)
    p.add_argument('--n_enc_temporal_layers', type=int, default=2)
    p.add_argument('--n_cross_layers', type=int, default=3)
    p.add_argument('--n_dec_temporal_layers', type=int, default=2)
    p.add_argument('--temporal_kernel', type=int, default=9)
    p.add_argument('--dropout', type=float, default=0.1)
    args = p.parse_args()

    if args.self_test:
        from scripts.self_test import run_self_test  # noqa: E402
        run_self_test()
        return

    # ----- DDP setup -----
    ddp_active = _ddp_setup()
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    if ddp_active:
        dev = torch.device(f'cuda:{_ddp_local_rank()}')
        args.device = f'cuda:{_ddp_local_rank()}'
    else:
        dev = torch.device(args.device)
    args.ddp = bool(ddp_active)
    args.world_size = _ddp_world_size()
    args.global_rank = _ddp_global_rank()
    if _ddp_is_main():
        Path(args.out).mkdir(parents=True, exist_ok=True)
    _ddp_barrier()
    if ddp_active and _ddp_is_main():
        print(f'[DDP] world={_ddp_world_size()} per-GPU batch='
              f'{args.batch_size} global batch='
              f'{args.batch_size * _ddp_world_size()}', flush=True)

    # ----- data load + paired alignment -----
    mk = lambda d, sp: UnifiedMotionDataset(
        [d], sp, max_frames=args.max_frames,
        max_joints=args.max_joints, normalize=False)
    src_tr, tgt_tr = mk(args.src_dir, 'train'), mk(args.tgt_dir, 'train')
    src_va, tgt_va = mk(args.src_dir, 'val'), mk(args.tgt_dir, 'val')
    assert len(src_tr) == len(tgt_tr) and len(src_va) == len(tgt_va), \
        'pair len mismatch'
    for nm, a, b in (('train', src_tr, tgt_tr), ('val', src_va, tgt_va)):
        for i in range(len(a)):
            sa = os.path.basename(str(a.samples[i].get('motion_path', '')))
            tb = os.path.basename(str(b.samples[i].get('motion_path', '')))
            assert sa == tb, f'{nm} pair misalign idx{i}: {sa} vs {tb}'
    if _ddp_is_main():
        print(f'paired train={len(src_tr)} val={len(src_va)} '
              f'(all basenames aligned)', flush=True)

    # ----- IK retained filter + IK npz coverage -----
    if args.w_rot_ik > 0:
        rc_path = os.path.join(args.ik_dir, 'retained_clips.txt')
        retained = set(Path(rc_path).read_text().split())

        def _ik_filter(src_ds, tgt_ds, label):
            n0 = len(tgt_ds.samples)
            keep = [i for i in range(n0)
                    if os.path.basename(
                        str(tgt_ds.samples[i]['motion_path'])) in retained]
            for i in keep:
                base = os.path.basename(
                    str(tgt_ds.samples[i]['motion_path']))
                assert os.path.exists(os.path.join(args.ik_dir, base)), \
                    f'retained {base} has no IK npz in {args.ik_dir}'
            src_ds.samples = [src_ds.samples[i] for i in keep]
            tgt_ds.samples = [tgt_ds.samples[i] for i in keep]
            if _ddp_is_main():
                print(f'  IK-retained {label}: {len(keep)}/{n0} kept',
                      flush=True)

        _ik_filter(src_tr, tgt_tr, 'train')
        _ik_filter(src_va, tgt_va, 'val')
        assert len(src_tr) == len(tgt_tr) and len(src_va) == len(tgt_va)
        for _ds, _lb in ((tgt_tr, 'tgt_tr'), (src_tr, 'src_tr'),
                         (tgt_va, 'tgt_va'), (src_va, 'src_va')):
            assert_no_crop_for_ik(_ds, args.max_frames, _lb)

    # ----- model: load L6 init (strict=False) + instantiate Model -----
    if args.from_scratch:
        model_kwargs = dict(
            d_model=args.d_model, n_heads=args.n_heads, d_ff=args.d_ff,
            n_graph_layers=args.n_graph_layers,
            n_enc_temporal_layers=args.n_enc_temporal_layers,
            n_cross_layers=args.n_cross_layers,
            n_dec_temporal_layers=args.n_dec_temporal_layers,
            temporal_kernel=args.temporal_kernel, dropout=args.dropout,
        )
        model = Model(**model_kwargs).to(dev)
        d_model_actual = args.d_model
        use_name_embed = True
        if _ddp_is_main():
            print('from_scratch: random init', flush=True)
    else:
        ckpt = torch.load(args.init_ckpt, map_location=dev,
                          weights_only=False)
        ma = ckpt['args']; ma = ma if isinstance(ma, dict) else vars(ma)
        model_kwargs = dict(
            d_model=ma['d_model'], n_heads=ma['n_heads'], d_ff=ma['d_ff'],
            n_graph_layers=ma['n_graph_layers'],
            n_enc_temporal_layers=ma['n_enc_temporal_layers'],
            n_cross_layers=ma['n_cross_layers'],
            n_dec_temporal_layers=ma['n_dec_temporal_layers'],
            temporal_kernel=ma.get('temporal_kernel', 9),
            dropout=args.dropout,
        )
        model = Model(**model_kwargs).to(dev)
        # strict=False because L6 ckpt contains slot_assignment.* keys that
        # this minimal model intentionally drops.
        load_result = model.load_state_dict(
            ckpt['model_state_dict'], strict=False)
        d_model_actual = ma['d_model']
        use_name_embed = bool(ma.get('use_name_embed', True))
        if _ddp_is_main():
            unexp_slot = [k for k in load_result.unexpected_keys
                          if k.startswith('slot_assignment.')]
            unexp_other = [k for k in load_result.unexpected_keys
                           if not k.startswith('slot_assignment.')]
            print(f'fine-tune init from {args.init_ckpt}')
            print(f'  missing keys (expect NONE): '
                  f'{load_result.missing_keys}')
            print(f'  unexpected keys: {len(unexp_slot)} slot_assignment.* '
                  f'(expected, dropped); {len(unexp_other)} other: '
                  f'{unexp_other}', flush=True)
            if unexp_other:
                raise SystemExit(
                    f'PREFLIGHT ABORT (ckpt load): unexpected non-'
                    f'slot_assignment keys in {args.init_ckpt}: '
                    f'{unexp_other}. The L6 ckpt schema must match '
                    f'Model except for slot_assignment.* keys.')
    model.encoder.use_name_embed = use_name_embed

    # ----- TreeIK head -----
    topofk = TopoFKTreeIKDecoder(model.decoder, d_model_actual).to(dev)

    # ----- freeze encoder.name_embedding (no_k_slot equivalent of source
    # --freeze_base_slot_name 1; SlotAssignment freeze is N/A here) -----
    frozen_ids = set()
    if args.freeze_name_embed:
        for nm, prm in model.named_parameters():
            if nm.startswith('encoder.name_embedding.'):
                prm.requires_grad_(False)
                frozen_ids.add(id(prm))
        if _ddp_is_main():
            print(f'FROZE encoder.name_embedding ({len(frozen_ids)} params, '
                  f'requires_grad=False & excluded from optimizer)',
                  flush=True)
    base_params = [prm for prm in model.parameters()
                   if id(prm) not in frozen_ids]
    extra = list(topofk.new_parameters())  # excludes self.base == model.decoder
    params = base_params + extra
    assert len({id(p) for p in params}) == len(params), \
        'duplicate optimizer params'
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=1e-4)

    # ----- preflight (rank0-only) -----
    _preflight_ok = True
    _preflight_where = 'preflight (not run)'
    if _ddp_is_main():
        try:
            write_preflight_manifest(
                args.out, args.src_dir, args.tgt_dir,
                src_tr, tgt_tr, src_va, tgt_va, args)
            assert_name_policy(src_tr, tgt_tr, src_va, tgt_va)
            assert_no_raw_rotation_supervision()
            if args.w_rot_ik > 0:
                miss = [os.path.basename(
                    str(tgt_tr.samples[k]['motion_path']))
                    for k in range(len(tgt_tr.samples))
                    if not os.path.exists(os.path.join(
                        args.ik_dir, os.path.basename(str(
                            tgt_tr.samples[k]['motion_path']))))]
                if miss:
                    raise SystemExit(
                        f'PREFLIGHT ABORT (IK coverage): {len(miss)} train '
                        f'clips lack ik_rot6d in {args.ik_dir}; e.g. '
                        f'{miss[:5]}')
                print(f'PREFLIGHT IK coverage: all {len(tgt_tr.samples)} '
                      f'train clips have ik_rot6d — OK', flush=True)
            _preflight_where = 'preflight PASSED'
        except BaseException as _e:                          # noqa: BLE001
            _preflight_ok = False
            _preflight_where = (f'rank0 preflight: '
                                f'{type(_e).__name__}: {_e}')
            print(f'PREFLIGHT ABORT (rank0): {_preflight_where}', flush=True)
    _ddp_abort_if_any_rank_failed(_preflight_ok, _preflight_where)
    _ddp_barrier()

    # ----- noK runtime preflight: src_jm == tgt_jm bitwise (source 1592-1629) -----
    _nb = 0
    for _bi in range(0, len(src_tr), args.batch_size):
        _ix = list(range(_bi, min(_bi + args.batch_size, len(src_tr))))
        _sm = collate_fn([src_tr[k] for k in _ix])['joint_mask']
        _tm = collate_fn([tgt_tr[k] for k in _ix])['joint_mask']
        if _sm.shape != _tm.shape or not torch.equal(
                _sm.bool(), _tm.bool()):
            _ex = [os.path.basename(str(src_tr.samples[k].get(
                'motion_path', ''))) for k in _ix[:4]]
            raise SystemExit(
                'PREFLIGHT ABORT (NOKSLOT same-topo): no_k_slot requires '
                'src joint_mask == tgt joint_mask bitwise (true same-'
                f'skeleton self-recon); batch {_bi}//{args.batch_size} '
                f'src={tuple(_sm.shape)} tgt={tuple(_tm.shape)} differ '
                f'(src_valid={int(_sm.bool().sum())} '
                f'tgt_valid={int(_tm.bool().sum())}); e.g. {_ex}.')
        _nb += 1
    if _ddp_is_main():
        print(f'PREFLIGHT NOKSLOT same-topo: src_jm == tgt_jm bitwise on '
              f'all {_nb} train batches ({len(src_tr)} clips)', flush=True)

    # ----- composite + DDP wrap -----
    class _Composite(nn.Module):
        """Single composite holding {model, topofk}; DDP wraps it ONCE so the
        shared decoder (== model.decoder == topofk.base) is registered
        exactly once in DDP's reducer (verbatim idiom from source 1423-
        1487)."""

        def __init__(self, m, tf):
            super().__init__()
            self.model = m
            self.topofk = tf

        def forward(self, s, t):
            slot, s_j, asg = encode_decode(self.model, s, t)
            parents_list = [[int(x) for x in pl]
                            for pl in t['parent_indices']]
            pred, r6 = self.topofk(
                slot, s_j, asg, t['joint_mask'], s['frame_mask'],
                parents_list,
                t['rest_offsets'].to(t['joint_mask'].device),
                fps_of(t), t['adjacency'], t['geodesic_dist'],
                return_rot=True)
            return pred, r6

    composite = _Composite(model, topofk).to(dev)
    ddp_wrapped = None
    if ddp_active:
        from torch.nn.parallel import DistributedDataParallel as _DDP
        ddp_wrapped = _DDP(composite, device_ids=[_ddp_local_rank()],
                           find_unused_parameters=True)
        _ddp_barrier()
        if _ddp_is_main():
            print(f'[DDP] wrapped composite (model + topofk); shared decoder '
                  f'registered once; find_unused_parameters=True',
                  flush=True)

    # ----- DDP sampler -----
    if ddp_active:
        _ws = _ddp_world_size()
        _rank = _ddp_global_rank()
        _n_global_steps = len(src_tr) // (args.batch_size * _ws)
        _n_drop = len(src_tr) - _n_global_steps * args.batch_size * _ws
        if _ddp_is_main():
            print(f'[DDP] sampler: {len(src_tr)} train clips, world={_ws}, '
                  f'per-GPU batch={args.batch_size}, {_n_global_steps} '
                  f'global steps/epoch, drop_last={_n_drop}', flush=True)
        if _n_global_steps < 1:
            raise SystemExit(
                f'DDP sampler: {len(src_tr)} train clips < one global batch '
                f'({args.batch_size}*{_ws}); cannot form a step.')

    # ----- train-only diagnostic split (no held-out leakage) -----
    _ntr = len(src_tr)
    _rng = np.random.RandomState(args.seed)
    _diag_idx = sorted(_rng.permutation(_ntr)[:max(1, _ntr // 10)].tolist())
    if _ddp_is_main():
        print(f'train-only diagnostic holdout: {len(_diag_idx)}/{_ntr} '
              f'TRAIN clips (seed={args.seed}); val NEVER read in train loop',
              flush=True)

    # ----- train loop -----
    best, logs = 1e9, []
    for ep in range(args.epochs):
        composite.train()
        tot = nb = 0
        if ddp_active:
            _g = np.random.default_rng(args.seed + ep)
            _gperm = _g.permutation(len(src_tr))
            _gperm = _gperm[:_n_global_steps * args.batch_size * _ws]
            _gperm = _gperm.reshape(_n_global_steps, _ws, args.batch_size)
            _rank_idx = _gperm[:, _rank, :]
            _step_iter = (list(_rank_idx[gs])
                          for gs in range(_n_global_steps))
        else:
            perm = np.random.permutation(len(src_tr))
            _step_iter = (perm[bi:bi + args.batch_size].tolist()
                          for bi in range(0, len(src_tr), args.batch_size))
        for idx in _step_iter:
            idx = [int(k) for k in idx]
            s = to_dev(collate_fn([src_tr[k] for k in idx]), dev)
            t = to_dev(collate_fn([tgt_tr[k] for k in idx]), dev)
            opt.zero_grad()
            _fwd = ddp_wrapped if ddp_active else composite
            pred, pred_r6 = _fwd(s, t)
            loss = recon_loss(pred, t)

            # velocity consistency (SlotAssignment-independent)
            if args.w_vel_consistency > 0:
                pp_vc = pred[..., :3]; pv_pred = pred[..., 3:]
                fps_vc = fps_of(t)
                pvc = torch.zeros_like(pp_vc)
                if pp_vc.shape[1] > 1:
                    pvc[:, 1:] = (pp_vc[:, 1:] - pp_vc[:, :-1]) * fps_vc
                    pvc[:, 0] = pvc[:, 1]
                mvc = (t['joint_mask'][:, None, :, None].float()
                       * t['frame_mask'][:, :, None, None].float())
                l_vc = (((pvc - pv_pred).abs() * mvc).sum()
                        / (mvc.sum() * 3).clamp(min=1.0))
                loss = loss + args.w_vel_consistency * l_vc

            # IK rot supervision (TreeIK ③)
            if args.w_rot_ik > 0:
                ik_r6 = load_ik_batch(
                    tgt_tr, idx, args.ik_dir,
                    args.max_frames, args.max_joints,
                    t['frame_mask'], t['joint_mask'], dev)
                par_l = [[int(x) for x in pl]
                         for pl in t['parent_indices']]
                non_leaf = build_non_leaf(par_l, args.max_joints,
                                          t['joint_mask'], dev)
                rot_mask = (t['frame_mask'][:, :, None].bool()
                            & t['joint_mask'][:, None, :].bool()
                            & non_leaf[:, None, :])
                loss = loss + args.w_rot_ik * rot_geodesic_loss(
                    pred_r6, ik_r6, rot_mask)

            # acc smoothness
            if args.w_acc > 0:
                pp_a = pred[..., :3]
                jm = t['joint_mask'].unsqueeze(1).unsqueeze(-1)
                fm = t['frame_mask'].unsqueeze(-1).unsqueeze(-1)
                if pp_a.shape[1] > 2:
                    acc = pp_a[:, 2:] - 2 * pp_a[:, 1:-1] + pp_a[:, :-2]
                    m = jm * fm[:, 2:]
                    loss = loss + args.w_acc * (
                        ((acc ** 2) * m).sum() / m.sum().clamp(min=1.0))

            if torch.isnan(loss) or torch.isinf(loss):
                raise RuntimeError(
                    f'NaN/Inf loss ep{ep} '
                    f'rank{_ddp_global_rank()} idx={idx}')

            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            tot += loss.item(); nb += 1
        avg = tot / max(nb, 1)

        # ----- rank0 train-only diagnostic + ckpt save -----
        if _ddp_is_main():
            composite.eval()
            with torch.no_grad():
                vnum = vden = 0.0
                for vb in range(0, len(_diag_idx), args.batch_size):
                    vi = _diag_idx[vb:vb + args.batch_size]
                    vs = to_dev(collate_fn([src_tr[k] for k in vi]), dev)
                    vt = to_dev(collate_fn([tgt_tr[k] for k in vi]), dev)
                    vp_pair = composite(vs, vt)
                    vp = vp_pair[0] if isinstance(vp_pair, tuple) else vp_pair
                    gt = vt['motion_features']
                    m = (vt['joint_mask'].unsqueeze(1).unsqueeze(-1)
                         * vt['frame_mask'].unsqueeze(-1).unsqueeze(-1))
                    vnum += float((((vp - gt) ** 2) * m).sum())
                    vden += float(m.sum())
                vloss = vnum / max(vden, 1.0)
            logs.append({'epoch': ep, 'train': avg, 'train_diag': vloss})
            if ep % 10 == 0 or ep == args.epochs - 1:
                print(f'ep{ep} train={avg:.6f} train_diag={vloss:.6f}'
                      + (f' [rank0; global_step/epoch={nb}]'
                         if ddp_active else ''), flush=True)
            ckpt_dict = {
                'model_state_dict': model.state_dict(),
                'topofk_state_dict': topofk.state_dict(),
                'args': vars(args), 'epoch': ep, 'train_diag': vloss,
            }
            if vloss < best:
                best = vloss
                torch.save(ckpt_dict, Path(args.out) / 'best_model.pt')
            if ep % args.save_every == 0 or ep == args.epochs - 1:
                torch.save(ckpt_dict, Path(args.out) / 'last_model.pt')
        _ddp_barrier()

    if _ddp_is_main():
        json.dump(logs, open(Path(args.out) / 'training_log.json', 'w'))
        print(f'DONE head=topofk_treeik no_k_slot=True '
              f'from_scratch={args.from_scratch} '
              f'best_train_diag={best:.6f} -> {args.out}', flush=True)
    _ddp_cleanup()


if __name__ == '__main__':
    main()
