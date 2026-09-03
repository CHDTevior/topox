"""Carry the protocol down the chain, so a clean final split cannot be mistaken for a clean model.

Each stage of the pipeline guards its own input. That is not enough: the VQVAE can resume any
checkpoint, the export accepts any VQVAE, the backbone loads any tokenizer and any token cache,
and the online evaluator is whatever file was named. A backbone trained on a retained token cache
that was exported from a tokenizer trained on the full corpus is not unseen-clean, and nothing on
disk would say so.

So every artifact carries a stamp naming the protocol it was produced under and the SHA-256 of
everything it depended on, and every consumer verifies its upstream. Under `--protocol legacy`
the stamp is still written but nothing is enforced, which keeps every pre-existing command and
checkpoint working; under `unseen_topology_v1` a missing or mismatched upstream is fatal.

The rule that matters: **only a chain that is stamped `unseen_topology_v1` at every link may back
an unseen-topology claim.** A single legacy link anywhere makes the whole chain legacy.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Optional

PROTOCOL_LEGACY = "legacy"
PROTOCOL_UNSEEN = "unseen_topology_v1"
KEY = "_provenance"


def sha256_file(path: str | Path, chunk: int = 1 << 20) -> Optional[str]:
    p = Path(path)
    if not p.is_file():
        return None
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for c in iter(lambda: fh.read(chunk), b""):
            h.update(c)
    return h.hexdigest()


def artifact_body_sha256(path: str | Path) -> Optional[str]:
    """The self-hash an artifact carries: SHA-256 over its content with its own hash field
    removed. This is the identity of the pre-registration content; the file hash is not, because
    reformatting the file changes it without changing what it says."""
    p = Path(path)
    if not p.is_file():
        return None
    d = json.loads(p.read_text())
    body = json.dumps({k: v for k, v in d.items() if k != "artifact_sha256"},
                      indent=2, sort_keys=True)
    return hashlib.sha256(body.encode()).hexdigest()


# The fields that define WHICH EXPERIMENT a run is. A resume that differs on any of these is a
# different experiment wearing the same output directory, and continuing it silently would produce
# a checkpoint no one can describe. Kept separate from the rest of the stamp, which is descriptive.
# NOTE ON THE TWO HASHES. An artifact has a BODY hash (over its content excluding its own hash
# field — this is the identity of the pre-registration) and a FILE hash (over the bytes on disk).
# They are different values and calling both "the artifact sha" produced a live bug: the launcher
# passed the file hash where the body hash was expected, so a strict run would have aborted at
# startup. They are named apart everywhere now.
CONTRACT_KEYS = ("protocol", "holdout_artifact_body_sha256", "splits_sha256",
                 "joint_semantics_sha256", "semantic_dim", "semantic_enabled", "moment_policy",
                 "augment", "augment_prob", "removal_rate", "training_config_sha256")

# The arguments whose values ARE the experiment. A resume that differs on any of them is a
# different training run wearing the same output directory. This list exists because the ten
# data/protocol fields above were shown to be insufficient: a review changed batch 32->16,
# lr 6.65e-5->1e-3, seed 42->7, the human-upsampling curriculum and w_fk, and the contract
# comparison reported no mismatch at all. An automatic watchdog relaunching with the wrong flags
# would have been told everything was consistent.
#
# Deliberately EXCLUDED, because they describe how a run is being operated rather than what it is:
# out, resume, num_workers, log_every, qa_every, save_every, periodic_save_every, overwrite,
# device, smoke, smoke_iters. Changing those does not change the model being trained.
TRAINING_CONFIG_KEYS = (
    # data shape
    "anytop_root", "splits_dir", "max_frames", "max_joints", "max_coarse", "val_frac",
    # architecture
    "d_model", "n_heads", "d_ff", "n_graph_layers", "n_enc_temporal_layers", "n_pre_vq_layers",
    "n_post_vq_layers", "n_cross_layers", "n_dec_temporal_layers", "temporal_stride",
    "temporal_kernel", "dropout",
    # quantizer
    "code_dim", "num_codes", "num_quantizers", "ema_mu", "quantize_dropout_prob",
    "dead_code_threshold",
    # objective
    "w_pos", "w_rot", "w_vel", "w_contact", "w_world", "w_fk", "w_fk_smooth", "w_traj", "w_commit",
    # optimisation and the effective global batch (world size matters: the same per-rank batch on
    # a different number of ranks is a different optimisation problem)
    # global_batch, NOT batch_size or world_size. With the launcher deriving the per-rank batch
    # from a pinned global batch, 8 ranks x 8 and 2 ranks x 32 are the SAME optimisation problem
    # (there is no BatchNorm here, so averaged gradients are identical for a given global batch).
    # Including world size would have made the digest refuse a resume onto a different number of
    # cards -- which is exactly what an auto-resume across allocation boundaries has to do, and
    # refusing it would have bought no scientific protection.
    "lr", "warmup_steps", "epochs", "global_batch", "amp_dtype", "seed",
    # curriculum
    "human_upsample_factor", "human_upsample_start_epoch",
    "human_upsample_phase2_factor", "human_upsample_phase2_start_epoch",
)


def training_config_sha256(args: Any, world_size: int) -> str:
    """One hash over every value that defines the training run, for the resume contract.

    A missing key is recorded as the string "<absent>" rather than skipped: silently ignoring a key
    that one side has and the other does not would make an added or removed argument invisible to
    the comparison, which is the failure this hash exists to prevent.
    """
    d = args if isinstance(args, dict) else vars(args)
    d = dict(d)
    d["world_size"] = int(world_size)
    d["global_batch"] = int(d.get("batch_size", 0)) * int(world_size)
    body = json.dumps({k: (d[k] if k in d else "<absent>") for k in TRAINING_CONFIG_KEYS},
                      sort_keys=True, default=str)
    return hashlib.sha256(body.encode()).hexdigest()


# CodeFlow trainer args that describe HOW the run is operated, not WHAT is trained. The config
# digest is defined by EXCLUSION over these (a new experiment-defining argument is therefore
# hashed by default — forgetting to list it fails closed, not open). Path-valued args whose
# CONTENT is already pinned elsewhere are excluded too: the token cache and frozen tokenizer are
# byte-verified through the upstream provenance chain, and holdout artifacts through their body
# sha in CONTRACT_KEYS.
CODEFLOW_OPERATIONAL_KEYS = frozenset({
    "out", "resume", "overwrite", "num_workers", "log_every", "qa_every", "save_every",
    "periodic_save_every", "device", "smoke", "smoke_iters", "mem_profile",
    "gen_eval", "gen_eval_every", "gen_eval_n", "gen_eval_batch", "gen_eval_steps",
    "gen_eval_manifest", "gen_eval_data_root", "gen_eval_caption_cache", "evaluator_ckpt",
    "eval_cond_scale", "eval_steps",
    # empirical_stats_max_clips is DELIBERATELY HASHED (not listed here): capping the empirical
    # z_q scan changes the normalisation the model trains under, so a resume across different
    # caps is a different experiment. Real runs pin it to 0 everywhere (iron rule), so the
    # legitimate watchdog resume always matches.
    "token_cache", "frozen_vqvae_ckpt", "holdout_artifact", "holdout_sha", "allow_no_holdout",
    # caption_sidecar is a PATH whose content identity the token-dataset byte-verifies against
    # the cache manifest; the path itself is operational. caption_sampling is NOT listed —
    # fixed vs random IS the experiment (hashed by exclusion).
    "caption_sidecar",
    "batch_size",   # replaced by the world-size-normalised global_batch, as in the VQVAE digest
    # grad_accum: same argument as batch_size — its EXPERIMENT effect (effective global
    # batch) is fully captured by global_batch = batch_size*world*grad_accum; the raw key
    # is how the batch is scheduled, not what is trained. Without this exclusion a
    # legitimate cross-hardware resume (8xB8xacc1 -> 4xB8xacc2, same global 64) is
    # refused (hit live 2026-08-11 moving v3_xpred from 8xH100 to 4xH200).
    "grad_accum",
})


def codeflow_training_config_sha256(args: Any, world_size: int) -> str:
    """Exclusion-defined config digest for the CodeFlow trainer (codex r4 #3).

    Every argparse key not listed as operational participates, so newly added experiment
    arguments (the text-architecture flags being the motivating case) are covered without
    anyone remembering to extend a list. global_batch replaces batch_size x world_size for the
    same reason as the VQVAE digest: an auto-resume onto a different card count with the same
    global batch is the same optimisation problem.
    """
    d = dict(args if isinstance(args, dict) else vars(args))
    d["global_batch"] = int(d.get("batch_size", 0)) * int(world_size) * int(d.get("grad_accum", 1) or 1)
    # Decoded-geometry loss runs: the decoded loss executes on the SYNC micro-batch only
    # (weight scaled by accum), so its effective sample batch is batch_size*world — NOT
    # captured by global_batch. Hash it as a derived field whenever the decoded loss is
    # active, so 8xB4xacc2 (dec batch 32) and 8xB8xacc1 (dec batch 64) digest differently
    # (codex gradaccum r1 A). Dec-off runs add no field -> pre-existing digests unchanged.
    if any((d.get(k) or 0) > 0 for k in ("w_dec_world", "w_dec_traj", "w_dec_speed")):
        d["decoded_loss_global_batch"] = int(d.get("batch_size", 0)) * int(world_size)
    # None-sentinel experiment args: dropping them when unset keeps the digest
    # byte-identical to ckpts stamped before each flag existed (their resume
    # contract stays valid), while an EXPLICIT value participates and is therefore
    # contract-protected (a watchdog relaunch that loses the flag digests without
    # the key and is refused). Add NEW optional experiment args here — NEVER default
    # them to a concrete value in argparse, or every old ckpt's contract breaks.
    for _k in ("parameterization", "w_dec_world", "w_dec_traj", "w_dec_speed",
               "dec_geom_t_min", "dec_geom_every", "dec_speed_floor", "dec_speed_loss"):
        if d.get(_k) is None:
            d.pop(_k, None)
    keys = sorted(k for k in d if k not in CODEFLOW_OPERATIONAL_KEYS)
    body = json.dumps({k: d[k] for k in keys}, sort_keys=True, default=str)
    return hashlib.sha256(body.encode()).hexdigest()


def contract(prov: Optional[dict]) -> dict:
    """The experiment-defining subset of a stamp, with absent fields normalised to None so a
    missing key and an explicit null compare equal."""
    p = prov or {}
    return {k: p.get(k) for k in CONTRACT_KEYS}


def compare_contract(old: Optional[dict], new: dict) -> list[str]:
    a, b = contract(old), contract(new)
    return [f"{k}: was {a[k]!r}, now {b[k]!r}" for k in CONTRACT_KEYS if a[k] != b[k]]


def stamp(*, protocol: str, stage: str, holdout_artifact: Optional[str] = None,
          data_root: Optional[str] = None, splits_dir: Optional[str] = None,
          upstream: Optional[dict] = None, extra: Optional[dict] = None) -> dict:
    """Build the provenance record a stage writes into whatever it produces."""
    file_sha = sha256_file(holdout_artifact) if holdout_artifact else None
    body_sha = artifact_body_sha256(holdout_artifact) if holdout_artifact else None
    d: dict[str, Any] = {
        "protocol": protocol,
        "stage": stage,
        "holdout_artifact": str(holdout_artifact) if holdout_artifact else None,
        "holdout_artifact_body_sha256": body_sha,
        "holdout_artifact_file_sha256": file_sha,
        "data_root": str(data_root) if data_root else None,
        "splits_dir": str(splits_dir) if splits_dir else None,
        "split_sha256": {},
        "upstream": upstream or {},
    }
    if splits_dir:
        for name in ("train.txt", "val.txt"):
            s = sha256_file(Path(splits_dir) / name)
            if s:
                d["split_sha256"][name] = s
    if extra:
        # `extra` must not be able to overwrite a field the verifier trusts: a caller that passed
        # extra={"protocol": ...} could otherwise forge a stamp through the normal API.
        clash = sorted(set(extra) & set(d))
        if clash:
            raise ValueError(f"stamp(extra=...) would overwrite reserved provenance fields "
                             f"{clash}; those are set by this function, not by the caller")
        d.update(extra)
    # A single hash over the split files, so a contract comparison does not depend on dict order.
    d["splits_sha256"] = (hashlib.sha256(
        "".join(f"{k}:{v}" for k, v in sorted(d["split_sha256"].items())).encode()).hexdigest()
        if d["split_sha256"] else None)
    return d


def read(obj: Any) -> Optional[dict]:
    """Pull a provenance record out of a loaded checkpoint dict or a manifest dict."""
    if isinstance(obj, dict):
        return obj.get(KEY)
    return None


def verify_upstream(prov: Optional[dict], *, protocol: str, what: str,
                    expect_artifact_body_sha: Optional[str] = None, log=print) -> None:
    """Refuse an upstream artifact that cannot back the protocol this run claims.

    Under legacy this only reports, because pre-existing checkpoints carry no stamp at all and
    refusing them would break every command that worked yesterday.
    """
    if protocol != PROTOCOL_UNSEEN:
        if prov is None:
            log(f"[provenance] {what}: no stamp (legacy). This run cannot back an "
                f"unseen-topology claim.")
        return

    if prov is None:
        raise SystemExit(
            f"[provenance] REFUSED: {what} carries no provenance stamp, so there is no evidence "
            f"it was produced under {PROTOCOL_UNSEEN}. A clean split at this stage does not make "
            f"an upstream artifact clean. Re-produce it under the protocol, or run this stage "
            f"under --protocol legacy and do not use its output for an unseen-topology claim.")
    if prov.get("protocol") != PROTOCOL_UNSEEN:
        raise SystemExit(
            f"[provenance] REFUSED: {what} was produced under protocol "
            f"{prov.get('protocol')!r}, not {PROTOCOL_UNSEEN!r}. One legacy link makes the whole "
            f"chain legacy.")
    if expect_artifact_body_sha and \
            prov.get("holdout_artifact_body_sha256") != expect_artifact_body_sha:
        raise SystemExit(
            f"[provenance] REFUSED: {what} was produced against held-out artifact "
            f"{str(prov.get('holdout_artifact_body_sha256'))[:16]}..., this run uses "
            f"{expect_artifact_body_sha[:16]}.... Two different pre-registrations cannot be mixed in "
            f"one chain.")
    log(f"[provenance] {what}: OK, {PROTOCOL_UNSEEN} against artifact body "
        f"{str(prov.get('holdout_artifact_body_sha256'))[:16]}...")


def verify_resume_contract(old: Optional[dict], new: dict, *, what: str, log=print) -> None:
    """Refuse a resume whose experiment contract differs from the checkpoint's.

    This is the backstop for orchestration mistakes rather than for malice: a watchdog that
    restarts a job without the protocol flags would otherwise continue a strict run under legacy
    defaults, and the resulting checkpoint would carry a legacy stamp with strict weights inside.
    Nothing downstream could tell.
    """
    if old is None:
        if new.get("protocol") == PROTOCOL_UNSEEN:
            raise SystemExit(
                f"[provenance] REFUSED: resuming {what}, which carries no contract, into a "
                f"{PROTOCOL_UNSEEN} run. Start fresh or resume under --protocol legacy.")
        log(f"[provenance] {what}: no contract recorded (legacy checkpoint)")
        return
    diffs = compare_contract(old, new)
    if diffs:
        raise SystemExit(
            "[provenance] REFUSED: the resume target was trained under a DIFFERENT experiment "
            "contract:\n  " + "\n  ".join(diffs) +
            "\nContinuing would put weights from one experiment behind another experiment's "
            "label. Point --resume at a checkpoint from this configuration, or change the "
            "configuration to match.")
    log(f"[provenance] {what}: contract matches ({len(CONTRACT_KEYS)} fields)")


def summarise(prov: Optional[dict]) -> str:
    if not prov:
        return "unstamped (legacy)"
    return (f"{prov.get('protocol')} stage={prov.get('stage')} "
            f"artifact_body={str(prov.get('holdout_artifact_body_sha256'))[:12]}")
