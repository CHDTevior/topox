"""One-time contract migration: recompute training_config_sha256 under the
grad_accum-excluded digest (provenance.py 2026-08-11 fix, incl. the
decoded_loss_global_batch derived field).

WHY: the config digest is exclusion-defined; adding "grad_accum" to
CODEFLOW_OPERATIONAL_KEYS (its experiment effect is captured by the normalised
global_batch; for decoded-loss runs the sync-micro decoded batch is separately
hashed as decoded_loss_global_batch) changes the digest of EVERY existing
checkpoint, so their stored resume contracts no longer match what the new code
computes — a legitimate cross-hardware resume (8xH100 B8 acc1 -> 4xH200 B8
acc2, same global 64) was refused live on 2026-08-11. This tool migrates
EXPLICITLY LISTED checkpoints only: it validates the operator-supplied
--world_size by reproducing the FROZEN LEGACY digest (grad_accum included, no
derived field) against the stored value, recomputes the current digest from the
ckpt's OWN args, stores the old value under a backup key, and rewrites the file
atomically (mode-preserving, fsynced). Auditable, opt-in, fail-closed: any
target that cannot be positively validated is a hard error.

Usage:
  python scripts/_migrate_contract_gradaccum.py --world_size 8 \
      runs/<run>/last_model.pt [more ckpts...]
  --dry_run prints what would change without writing.
"""
import argparse
import hashlib
import json
import os
import stat
import sys
import tempfile

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data import provenance as prov  # noqa: E402

BACKUP_KEY = "training_config_sha256_pre_gradaccum_migration"


def _legacy_digest(args: dict, world_size: int) -> str:
    """FROZEN pre-2026-08-11 digest: grad_accum participates, no derived
    decoded_loss_global_batch field. Used ONLY to validate --world_size against
    the stored stamp; must NOT track future provenance.py changes."""
    d = dict(args)
    d["global_batch"] = (int(d.get("batch_size", 0)) * int(world_size)
                         * int(d.get("grad_accum", 1) or 1))
    for k in ("parameterization", "w_dec_world", "w_dec_traj", "w_dec_speed",
              "dec_geom_t_min", "dec_geom_every", "dec_speed_floor", "dec_speed_loss"):
        if d.get(k) is None:
            d.pop(k, None)
    legacy_op = set(prov.CODEFLOW_OPERATIONAL_KEYS) - {"grad_accum"}
    keys = sorted(k for k in d if k not in legacy_op)
    body = json.dumps({k: d[k] for k in keys}, sort_keys=True, default=str)
    return hashlib.sha256(body.encode()).hexdigest()


def migrate(path: str, world_size: int, dry_run: bool) -> None:
    ck = torch.load(path, map_location="cpu", weights_only=False)
    p = prov.read(ck)
    if p is None:
        raise SystemExit(f"[FAIL] {path}: no provenance stamp — refusing (explicit "
                         f"target must be migratable)")
    stored = p.get("training_config_sha256")
    if stored is None:
        raise SystemExit(f"[FAIL] {path}: stamp has no training_config_sha256")
    args = ck.get("args")
    if args is None:
        raise SystemExit(f"[FAIL] {path}: no args in ckpt")
    args = dict(args)
    new = prov.codeflow_training_config_sha256(args, world_size)
    legacy = _legacy_digest(args, world_size)
    backup = p.get(BACKUP_KEY)

    if stored == new:
        print(f"[ok]   {path}: digest already current ({new[:16]}...)")
        return
    # Positive validation of --world_size (codex gradaccum r1 B1): the stored
    # value must be explainable as either the frozen legacy digest (first
    # migration) or, on a re-run, the backup must be that legacy digest.
    if stored == legacy:
        pass  # first migration, world_size validated
    elif backup is not None and backup == legacy:
        raise SystemExit(
            f"[FAIL] {path}: already migrated once (backup matches legacy) but the "
            f"stored digest {stored[:16]}... is not the current recomputation "
            f"{new[:16]}... — provenance/code out of sync; refusing to rewrite")
    else:
        raise SystemExit(
            f"[FAIL] {path}: stored digest {stored[:16]}... matches neither the "
            f"legacy digest {legacy[:16]}... for --world_size={world_size} nor an "
            f"expected state — wrong world_size or unknown history; refusing")

    print(f"[migrate] {path}:\n  old {stored}\n  new {new}  (world_size={world_size})")
    if dry_run:
        return
    if BACKUP_KEY not in p:
        p[BACKUP_KEY] = stored
    p["training_config_sha256"] = new
    ck[prov.KEY] = p
    # Atomic, mode-preserving, fsynced replace in the ckpt's own directory.
    mode = stat.S_IMODE(os.stat(path).st_mode)
    d = os.path.dirname(os.path.abspath(path))
    fd, tmp = tempfile.mkstemp(dir=d, prefix=".migrate_", suffix=".pt")
    try:
        with os.fdopen(fd, "wb") as f:
            torch.save(ck, f)
            f.flush()
            os.fsync(f.fileno())
        os.chmod(tmp, mode)
        os.replace(tmp, path)
        dfd = os.open(d, os.O_DIRECTORY)
        try:
            os.fsync(dfd)
        finally:
            os.close(dfd)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)
    print(f"[done] {path}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpts", nargs="+")
    ap.add_argument("--world_size", type=int, required=True,
                    help="world size the ckpt was TRAINED with (v2/v3/v4 backbone runs: 8)")
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()
    for c in args.ckpts:
        migrate(c, args.world_size, args.dry_run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
