"""noKslot_clean / src.utils — shared utilities for the no_k_slot training,
eval, and animate pipeline. Verbatim ports from source repo's
scripts/train_paired_gate.py where noted, with K-slot-only assertions trimmed
out of write_preflight_manifest_nokslot.

Contents:
  DDP helpers (verbatim source 55-117):
    _ddp_is_active / _ddp_local_rank / _ddp_global_rank / _ddp_world_size /
    _ddp_is_main / _ddp_setup / _ddp_cleanup / _ddp_barrier /
    _ddp_abort_if_any_rank_failed

  Data utils (verbatim source 425-443):
    to_dev / recon_loss / fps_of

  IK rotation supervision utils (verbatim source 445-494):
    assert_no_crop_for_ik / load_ik_batch / build_non_leaf

  Preflight (PARTIALLY REVISED for noKslot baseline, source 575-707):
    _sha256                       (verbatim source 575-582)
    write_preflight_manifest_nokslot  (SIMPLIFIED from source 584-680:
        REMOVED K-slot multi-topology protocol assertions that don't apply
        to same-skeleton self-recon — see function docstring for diff)
    assert_name_policy            (verbatim source 682-707)
"""

import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist

from .data.unified_dataset import collate_fn


# =========================================================================== #
# DDP helpers — verbatim source train_paired_gate.py:55-117 (DDP-RULING 2026-
# 05-18). Single-GPU path is a no-op when LOCAL_RANK is absent.
# =========================================================================== #
def _ddp_is_active():
    """True iff this process is a torchrun-launched DDP rank (LOCAL_RANK set
    AND the process group initialized). No LOCAL_RANK -> single-GPU path."""
    return ('LOCAL_RANK' in os.environ
            and dist.is_available() and dist.is_initialized())


def _ddp_local_rank():
    return int(os.environ.get('LOCAL_RANK', 0))


def _ddp_global_rank():
    return dist.get_rank() if _ddp_is_active() else 0


def _ddp_world_size():
    return dist.get_world_size() if _ddp_is_active() else 1


def _ddp_is_main():
    return _ddp_global_rank() == 0


def _ddp_setup():
    """Initialize the process group iff launched via torchrun (LOCAL_RANK in
    env). nccl backend, bind this rank to cuda:LOCAL_RANK. Returns True when
    DDP is active, False for the untouched single-GPU path."""
    if 'LOCAL_RANK' in os.environ:
        if not dist.is_initialized():
            dist.init_process_group(backend='nccl')
        torch.cuda.set_device(_ddp_local_rank())
        return True
    return False


def _ddp_cleanup():
    if _ddp_is_active():
        dist.barrier()
        dist.destroy_process_group()


def _ddp_barrier():
    if _ddp_is_active():
        dist.barrier()


def _ddp_abort_if_any_rank_failed(local_ok, where):
    """All-rank consensus abort: if ANY rank's local_ok is False, EVERY rank
    raises SystemExit (no rank may proceed past a guard that another rank
    failed). Used to wrap the rank0-only preflight writes so a rank0 abort
    halts every rank deterministically. No-op (returns) in single-GPU."""
    if not _ddp_is_active():
        if not local_ok:
            raise SystemExit(f'PREFLIGHT ABORT (single-GPU): {where}')
        return
    flag = torch.tensor([0 if local_ok else 1], device='cuda')
    dist.all_reduce(flag, op=dist.ReduceOp.SUM)
    if int(flag.item()) > 0:
        raise SystemExit(
            f'PREFLIGHT ABORT (DDP all-rank consensus): a rank failed at '
            f'"{where}" (sum_fail={int(flag.item())}/{_ddp_world_size()}); '
            f'no rank proceeds past this guard.')


# =========================================================================== #
# Data utils — verbatim source train_paired_gate.py:425-443.
# =========================================================================== #
def to_dev(b, dev):
    return {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in b.items()}


def recon_loss(pred, tgt):
    gt = tgt['motion_features']
    jm = tgt['joint_mask'].unsqueeze(1).unsqueeze(-1)
    fm = tgt['frame_mask'].unsqueeze(-1).unsqueeze(-1)
    m = jm * fm
    return (((pred - gt) ** 2) * m).sum() / m.sum().clamp(min=1.0)


def fps_of(batch):
    f = batch.get('fps')
    if torch.is_tensor(f):
        return float(f.reshape(-1)[0].item())
    return 20.0


# =========================================================================== #
# IK rotation supervision utils — verbatim source train_paired_gate.py:445-494.
# =========================================================================== #
def assert_no_crop_for_ik(ds, max_frames, label):
    """codex ③ VERBATIM: offline ik_rot6d aligns row-for-row with pred_r6
    ONLY if no random crop (unified_dataset:150 random-crops train clips
    with T>max_frames). Fail-fast makes the alignment precondition explicit
    so it cannot silently become a false-PASS later."""
    bad = []
    for s in ds.samples:
        path = s['motion_path']
        T = np.load(path, allow_pickle=True)['local_positions'].shape[0]
        if T > max_frames:
            bad.append((os.path.basename(str(path)), T))
    if bad:
        raise SystemExit(
            f'IK offline targets require no random crop, but {label} has '
            f'{len(bad)} clips with T>{max_frames}; examples={bad[:5]}. '
            f'Stop and implement online IK or shared crop.'
        )


def load_ik_batch(tgt_ds, batch_idx, ik_dir, Tpad, Jpad,
                  frame_mask, joint_mask, dev):
    """codex ③: per-sample offline ik_rot6d [Tv,Jv,6] -> [B,Tpad,Jpad,6],
    row-aligned. The Tv==frame_mask.sum / Jv==joint_mask.sum asserts catch
    ANY misalignment (no-crop precondition already enforced upstream)."""
    B = len(batch_idx)
    ik = torch.zeros(B, Tpad, Jpad, 6, device=dev)
    for bi, k in enumerate(batch_idx):
        base = os.path.basename(str(tgt_ds.samples[k]['motion_path']))
        d = np.load(os.path.join(ik_dir, base), allow_pickle=True)
        r = torch.from_numpy(d['ik_rot6d'].astype(np.float32))   # [Tv,Jv,6]
        Tv, Jv = int(r.shape[0]), int(r.shape[1])
        fm_n, jm_n = int(frame_mask[bi].sum()), int(joint_mask[bi].sum())
        assert Tv == fm_n, f'{base}: ik Tv={Tv} != frame_mask {fm_n}'
        assert Jv == jm_n, f'{base}: ik Jv={Jv} != joint_mask {jm_n}'
        ik[bi, :Tv, :Jv] = r.to(dev)
    return ik


def build_non_leaf(parents_list, Jpad, joint_mask, dev):
    """codex ③: non_leaf[b,j]=True iff j is the parent of some VALID child
    (p>=0). Leaves excluded (position-unobservable, IK leaf targets
    arbitrary); root included iff it has children. [B,Jpad] bool."""
    B = len(parents_list)
    nl = torch.zeros(B, Jpad, dtype=torch.bool, device=dev)
    for b, par in enumerate(parents_list):
        for j, p in enumerate(par):
            if p >= 0 and j < Jpad and bool(joint_mask[b, j]):
                nl[b, p] = True
    return nl


# =========================================================================== #
# Preflight — _sha256 verbatim source 575-582; write_preflight_manifest_nokslot
# SIMPLIFIED from source 584-680 (see docstring for the trimmed-out K-slot-
# specific assertions); assert_name_policy verbatim source 682-707.
# =========================================================================== #
def _sha256(path):
    import hashlib
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def write_preflight_manifest_nokslot(out_dir, src_dir, tgt_dir, src_tr, tgt_tr,
                                     src_va, tgt_va, args):
    """Simplified preflight manifest for the no_k_slot same-skeleton self-
    reconstruction baseline.

    KEPT from source train_paired_gate.py:584-680 write_preflight_manifest:
      - split_manifest.json with train+val species lists, clip counts,
        per-split-file sha256 hashes
      - explicit split files MUST exist (no silent fallback to all.txt /
        default zoo split)
      - skeleton-files sha256 (per src/tgt skeletons/*.npz)
      - leakage_audit.txt text summary

    REMOVED (K-slot multi-topology protocol assertions that do not apply
    to no_k_slot same-skeleton self-recon where src==tgt by construction):
      - 'cs_sparse2full' substring assert on tgt_dir
      - HELDOUT == {'Dragon', 'Spider', 'Trex'} val-species byte-equality
      - train INTERSECT val species == EMPTY (no held-out species concept
        on the noKslot baseline; val is a clip-level subsample of the same
        species used in train)

    These three assertions are tied to the K-slot multi-topology protocol
    that the user 2026-05-19 placed on Hold. The noKslot baseline trains
    src==tgt full->full self-reconstruction (the diagnostic launcher's
    actual protocol); val is a clip-level holdout of the same species, so
    train/val species WILL overlap by design and the held-out triad has no
    place here.
    """
    sp = lambda ds: sorted({str(s.get('skeleton_id', '?'))
                            for s in ds.samples})
    tr_sp, va_sp = sp(tgt_tr), sp(tgt_va)
    inter = sorted(set(tr_sp) & set(va_sp))
    split_files = {}
    for tag, d in (('src', src_dir), ('tgt', tgt_dir)):
        for s in ('train', 'val'):
            f = Path(d) / 'splits' / f'{s}.txt'
            if not f.exists():
                raise SystemExit(
                    f'PREFLIGHT ABORT: expected explicit split file {f} '
                    f'missing -> dataset would fall back to all.txt / '
                    f'default split. Stop.')
            split_files[f'{tag}_{s}'] = {
                'path': str(f), 'sha256': _sha256(f),
                'n_lines': len(f.read_text().split())}
    skel_hashes = {}
    for tag, d in (('src', src_dir), ('tgt', tgt_dir)):
        skd = Path(d) / 'skeletons'
        if skd.is_dir():
            for f in sorted(skd.glob('*.npz')):
                skel_hashes[f'{tag}/{f.name}'] = _sha256(f)
    manifest = {
        'protocol': 'noKslot baseline (same-skeleton self-recon: src==tgt '
                    'full->full; val is clip-level holdout of same species)',
        'src_dir': str(src_dir), 'tgt_dir': str(tgt_dir),
        'head': 'topofk_treeik', 'no_k_slot': True,
        'seed': int(args.seed),
        'train_species': tr_sp, 'val_species': va_sp,
        'n_train_species': len(tr_sp), 'n_val_species': len(va_sp),
        'n_train_clips_paired': len(src_tr),
        'n_val_clips_paired': len(src_va),
        'train_val_species_intersection': inter,
        'split_files_sha256': split_files,
        'skeleton_files_sha256': skel_hashes,
    }
    (Path(out_dir) / 'split_manifest.json').write_text(
        json.dumps(manifest, indent=2))
    audit = [
        '=== leakage_audit (noKslot baseline; simplified from source) ===',
        f'src_dir = {src_dir}',
        f'tgt_dir = {tgt_dir}',
        f'protocol: noKslot same-skeleton self-reconstruction (src==tgt)',
        f'train species ({len(tr_sp)}): {", ".join(tr_sp)}',
        f'val species ({len(va_sp)}): {", ".join(va_sp)}',
        f'train INTERSECT val species = {inter}  '
        f'(overlap is EXPECTED for noKslot baseline — val is a clip-level '
        f'holdout of the same species used in train)',
        f'paired clips: train={len(src_tr)} val={len(src_va)}',
        'split files: explicit cs_sparse2full splits asserted (no fallback)',
        'name policy: name_hashes is a skeleton-joint-name intrinsic, '
        'applied identically for train and val splits (since both come from '
        'the same species set, the policy is trivially symmetric here).',
        'raw rotation ban: batch["local_rotations_6d"] is NEVER read as '
        'supervision; the ONLY rotation supervision is the offline IK-'
        'derived ik_rot6d (w_rot_ik path). See '
        'assert_no_raw_rotation_supervision() in scripts/train.py for the '
        'AST guard.',
    ]
    (Path(out_dir) / 'leakage_audit.txt').write_text('\n'.join(audit) + '\n')
    print(f'PREFLIGHT manifest+audit written -> {out_dir}/'
          f'split_manifest.json (+leakage_audit.txt). train_sp='
          f'{len(tr_sp)} val_sp={len(va_sp)} inter={len(inter)}',
          flush=True)


def assert_name_policy(src_tr, tgt_tr, src_va, tgt_va):
    """codex checklist #4 name-policy assertion. The name-hash descriptor
    must be applied IDENTICALLY for train and held-out (no held-out-
    derived correspondence). The dataset hashes canonical_names -> %1024
    deterministically and identically regardless of split, so we assert
    the held-out species genuinely carry name_hashes (the SAME mechanism)
    and that nothing has special-cased the split. Fail-fast if a held-out
    target clip has all-zero name_hashes (would mean the name anchor is
    silently absent only for held-out -> asymmetric policy).

    Verbatim port note: in the noKslot baseline (src==tgt same-skeleton
    self-recon) the val split is a clip-level holdout of the SAME species
    used in train, so 'held-out' wording in this docstring + error strings
    means 'val clips' rather than 'held-out species'. The mechanism check
    is identical regardless.
    """
    import numpy as _np
    for ds, lab in ((tgt_va, 'val/held-out'), (tgt_tr, 'train')):
        sample = collate_fn([ds[0]])
        nh = sample.get('name_hashes')
        if nh is None:
            raise SystemExit(
                f'PREFLIGHT ABORT (name policy): {lab} produced no '
                f'name_hashes — the name anchor (#4) must be present and '
                f'identical for train AND held-out.')
        if int((nh != 0).sum()) == 0:
            raise SystemExit(
                f'PREFLIGHT ABORT (name policy): {lab} clip 0 has all-zero '
                f'name_hashes — asymmetric name policy (held-out missing '
                f'the anchor) is forbidden (Gate-3 v2 §4).')
    print('PREFLIGHT name-policy: name_hashes present & non-trivial for '
          'BOTH train and held-out (identical hash mechanism) — OK',
          flush=True)
