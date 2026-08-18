"""One-time contract migration: re-stamp a checkpoint for a LONGER training schedule.

WHY: `epochs` participates in the CodeFlow config digest (it is not an operational key —
it changes the cosine lr schedule, hence the optimisation problem). The v2 run finished
its 300-epoch cosine at ep298 with lr decayed to 8.0e-07, i.e. 1% of the 8e-5 peak; its
val plateau from ep240 is schedule exhaustion, not model saturation. Continuing therefore
REQUIRES epochs > 300, which the resume contract correctly refuses — the refusal is the
system working, and this tool is the explicit, auditable way to say "yes, I mean it".

SAFETY MODEL (codex v2b r1). The operand is an irreplaceable training artefact, so:
  * PARENT-DIR DENY-LIST (BLOCKING-1): refuses any target whose canonical path lies inside
    a protected run directory. Migrate a branch-local COPY, never the original.
  * ONE-SHOT (MAJOR-1): refuses unconditionally, before computing anything, if the backup
    keys are already present — a migrated ckpt can never be migrated again, so the original
    pre-migration digest/epochs can never be overwritten.
  * CONCURRENCY + IDENTITY RECHECK (MAJOR-2): holds an exclusive lock for the whole
    read-modify-write, and re-verifies the source's (inode, size, mtime_ns) immediately
    before the atomic replace; if anything changed underneath, it aborts rather than
    clobbering a newer file with a stale in-memory snapshot.
  * WORLD-SIZE VALIDATION: the stored digest must be reproducible from the ckpt's OWN args
    at the supplied --world_size, otherwise the world_size is wrong or the history unknown.

The ckpt's args are rewritten (not just the stamp) because the trainer recomputes the
digest from the args it is handed at resume; leaving args['epochs'] stale would make the
stored stamp unreproducible from the file itself, which is worse than the mismatch we are
fixing.

DELIBERATELY NOT A GENERAL "CHANGE ANY KEY" TOOL: one key, one direction (longer schedule),
explicit target list. Anything else should get its own reviewed migration.

Usage:
  python scripts/_migrate_contract_epochs.py --world_size 8 \
      --old_epochs 300 --new_epochs 400 \
      runs/holdout_backbone_llm2vec_v2b_ep400/resume_seed_ep289.pt
  --dry_run prints what would change without writing.
"""
import argparse
import fcntl
import os
import stat
import sys
import tempfile

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from src.data import provenance as prov  # noqa: E402

BACKUP_DIGEST_KEY = "training_config_sha256_pre_epochs_migration"
BACKUP_EPOCHS_KEY = "epochs_pre_migration"

# realpath, not abspath: invoking this script through a symlink would otherwise move the
# protection root and let a parent-run ckpt slip past the deny-list (codex v2b r2 BLOCKING-1).
SCRIPT_REAL = os.path.realpath(__file__)
REPO = os.path.dirname(os.path.dirname(SCRIPT_REAL))
# Completed runs whose artefacts must never be rewritten by this tool. A branch that
# continues one of these copies the ckpt into its own OUT dir and migrates the copy.
PROTECTED_RUN_DIRS = (
    "runs/holdout_backbone_llm2vec_8card_v2",
    "runs/holdout_backbone_llm2vec_8card_v3_xpred",
    "runs/holdout_backbone_llm2vec_8card_v1",
    "runs/holdout_vqvae_semantic_8card_v1",
    "runs/codeflow_graph_pscf_v4b272neutral_n8192_b16g64_lr8e5_4xh200_seed42",
    "runs/vqvae_v4b272neutral_C96_J144_d512_Q4_n8192_b16g64_300ep_curric50to60_seed42",
    "runs/anytop_t2m_evaluator_distilbert_coemb512_gb128_lr1e-4_mfd12_v4b272_seed42",
)


def _assert_not_protected(path: str) -> None:
    real = os.path.realpath(path)
    for rel in PROTECTED_RUN_DIRS:
        prot = os.path.realpath(os.path.join(REPO, rel))
        if real == prot or real.startswith(prot + os.sep):
            raise SystemExit(
                f"[FAIL] {path}: target is inside PROTECTED run dir {rel} — refusing.\n"
                f"       Copy the ckpt into the continuation branch's OUT dir and migrate "
                f"the copy; the parent run's artefacts stay byte-identical.")


def migrate(path: str, world_size: int, old_epochs: int, new_epochs: int,
            dry_run: bool) -> None:
    _assert_not_protected(path)
    if new_epochs <= old_epochs:
        raise SystemExit(
            f"[FAIL] --new_epochs={new_epochs} must exceed --old_epochs={old_epochs}; "
            f"this tool only extends a schedule")

    # Hold the lock across read-modify-write so a concurrent trainer save or a second
    # migration cannot interleave (MAJOR-2). Lock file sits next to the target.
    lock_path = path + ".migrate.lock"
    lock_fd = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o644)
    try:
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            raise SystemExit(f"[FAIL] {path}: another migration holds the lock — refusing")

        st0 = os.stat(path)
        ck = torch.load(path, map_location="cpu", weights_only=False)
        p = prov.read(ck)
        if p is None:
            raise SystemExit(f"[FAIL] {path}: no provenance stamp — refusing")
        # One-shot: refuse before computing anything if this file was migrated before,
        # so the pre-migration backup can never be overwritten (MAJOR-1).
        if BACKUP_DIGEST_KEY in p or BACKUP_EPOCHS_KEY in p:
            raise SystemExit(
                f"[FAIL] {path}: already migrated (backup keys present: "
                f"epochs_pre_migration={p.get(BACKUP_EPOCHS_KEY)}) — refusing to migrate "
                f"twice; start from a fresh copy of the original if you need a different "
                f"schedule")
        stored = p.get("training_config_sha256")
        if stored is None:
            raise SystemExit(f"[FAIL] {path}: stamp has no training_config_sha256")
        args = ck.get("args")
        if args is None:
            raise SystemExit(f"[FAIL] {path}: no args in ckpt")
        args = dict(args)

        have = args.get("epochs")
        if have != old_epochs:
            raise SystemExit(
                f"[FAIL] {path}: ckpt records epochs={have}, but --old_epochs={old_epochs} "
                f"was given — refusing (wrong run or typo)")

        current = prov.codeflow_training_config_sha256(args, world_size)
        if stored != current:
            raise SystemExit(
                f"[FAIL] {path}: stored digest {stored[:16]}... is not reproducible from "
                f"the ckpt's args at --world_size={world_size} (recomputed "
                f"{current[:16]}...) — wrong world_size or unknown history; refusing")

        args["epochs"] = int(new_epochs)
        new = prov.codeflow_training_config_sha256(args, world_size)
        print(f"[migrate] {path}:\n  epochs {old_epochs} -> {new_epochs}"
              f"\n  old digest {stored}\n  new digest {new}  (world_size={world_size})")
        if dry_run:
            return

        p[BACKUP_DIGEST_KEY] = stored
        p[BACKUP_EPOCHS_KEY] = int(old_epochs)
        p["training_config_sha256"] = new
        ck["args"] = args
        ck[prov.KEY] = p

        d = os.path.dirname(os.path.abspath(path))
        fd, tmp = tempfile.mkstemp(dir=d, prefix=".migrate_ep_", suffix=".pt")
        try:
            with os.fdopen(fd, "wb") as f:
                torch.save(ck, f)
                f.flush()
                os.fsync(f.fileno())
            os.chmod(tmp, stat.S_IMODE(st0.st_mode))
            # Identity recheck immediately before the swap: if the source changed since we
            # read it, our in-memory copy is stale and replacing would destroy newer data.
            st1 = os.stat(path)
            if (st1.st_ino, st1.st_size, st1.st_mtime_ns) != \
               (st0.st_ino, st0.st_size, st0.st_mtime_ns):
                raise SystemExit(
                    f"[FAIL] {path}: source changed during migration "
                    f"(ino/size/mtime differ) — aborting without writing")
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
    finally:
        os.close(lock_fd)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpts", nargs="+")
    ap.add_argument("--world_size", type=int, required=True,
                    help="world size the ckpt was TRAINED with (v2 8-card run: 8)")
    ap.add_argument("--old_epochs", type=int, required=True)
    ap.add_argument("--new_epochs", type=int, required=True)
    ap.add_argument("--dry_run", action="store_true")
    a = ap.parse_args()
    for c in a.ckpts:
        migrate(c, a.world_size, a.old_epochs, a.new_epochs, a.dry_run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
