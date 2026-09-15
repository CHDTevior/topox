#!/usr/bin/env python3
"""Train the in-context [demo | target] motion DiT on TrueBones (run-1 baseline, single GPU).

Everything here was fixed by the preflight/smoke chain, not by preference:
  data     frozen protocol splits (canonical-topology level); [demo 64 | target 240] = T 304;
           target slot covers the longest clip (237) so captions always describe the trained frames
  loss     CFM x0-prediction; grouped objective KIMODO_GAMMAS x sqrt(N_i/N_total) (root undiluted,
           gradient share independent of group size); `valid` mandatory
  sampling skeleton-balanced (Trex 72 clips must not outweigh Chicken 2)
  run-1    NO classifier-free guidance -- this run is the baseline the CFG run compares against

Validation = bucket A (seen rigs, unseen clips), with a RESET RNG stream each pass so every val
sees the identical demo/crop choices and the numbers are comparable across epochs.
Checkpoints are written atomically (tmp + rename); best is tracked on val flow loss.
"""
import argparse
import io, hashlib, json, math, os, pickle, sys, time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import contextlib
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.anytop_dataset import AnyTopDataset                               # noqa: E402
from src.data.ktjd17_augment import AugConfig                                    # noqa: E402
from src.data.incontext_pairs import (InContextPairs, collate, read_split,      # noqa: E402
                                      truebones_types, pzh_types, DEMO_FRAMES, TARGET_FRAMES)
from src.models.v2.dit_motion import (InContextMotionDiT, cfm_loss,             # noqa: E402
                                      KIMODO_GAMMAS, KTJD17_MASK_POLICY,
                                      _GROUP_SPEC_KTJD17)


def to_dev(b, dev):
    # parents/n_joints stay on CPU: the FK loss consumes them as python ints (.tolist()/int()),
    # and a GPU round-trip would force a sync per sample per step (codex 01a01939 fix 3).
    return {k: (v.to(dev, non_blocking=True) if torch.is_tensor(v) and k not in ("parents", "n_joints")
                else v) for k, v in b.items()}


def _rig_multiplicity(a):
    """`--rig_multiplicity "RIG:N,RIG:N"` -> {rig: N}. Applies to the TRAINING draw only; validation
    is unbalanced in every arm and stays so. A malformed spec refuses rather than silently training
    on the default mixture."""
    spec = str(getattr(a, "rig_multiplicity", "") or "").strip()
    if not spec:
        return None
    out = {}
    for tok in spec.split(","):
        tok = tok.strip()
        if not tok:
            raise SystemExit(f"[refuse] --rig_multiplicity {spec!r} has an empty token: expected "
                             f"RIG:N[,RIG:N], and a spec that parses to nothing would silently train "
                             f"the default mixture")
        if tok.count(":") != 1:
            raise SystemExit(f"[refuse] --rig_multiplicity token {tok!r}: expected RIG:N")
        rig, n = tok.split(":")
        rig = rig.strip()
        if not rig or not n.strip().isdigit() or int(n) < 1:
            raise SystemExit(f"[refuse] --rig_multiplicity token {tok!r}: RIG must be non-empty and N a "
                             f"positive integer")
        if rig in out:
            raise SystemExit(f"[refuse] --rig_multiplicity names {rig!r} twice")
        out[rig] = int(n)
    return out


def cond_of(b):
    d = dict(joint_bias=b["joint_bias"], frame_valid=b["frame_valid"],
             joint_valid=b["joint_valid"], text=b["text"], joint_sem=b["joint_sem"])
    if "demo_text" in b:                     # F5 reference-transcript analogue
        d["demo_text"] = b["demo_text"]
    for k in ("struct_feats", "updown"):     # graph-v2, present only when the dataset emits them
        if k in b:
            d[k] = b[k]
    return d


def fk_pack_of(b):
    """gamma_fk inputs, exactly the fields InContextPairs(emit_fk_fields=True) adds to the batch.
    _STD_FLOOR must be the SAME constant the dataset normalized with, or de-normalization drifts.
    An R_rest_global field marks a KTJD batch: cfm_loss then dispatches to the KTJD FK term
    (official decoder semantics) instead of the 13ch RIFKE recovery."""
    from src.data.anytop_dataset import _STD_FLOOR
    pack = dict(anytop_mean=b["anytop_mean"], anytop_std=b["anytop_std"], std_floor=_STD_FLOOR,
                parents=b["parents"], rest_offsets=b["rest_offsets"], n_joints=b["n_joints"])
    if "R_rest_global" in b:
        pack["kind"] = "ktjd17"
        pack["R_rest_global"] = b["R_rest_global"]
    if "lock_denominator" in b:
        # skeleton augmentation mode one_of: the pre-pruning contact-pair count of an augmented sample (0 = not augmented)
        pack["lock_denominator"] = b["lock_denominator"]
    return pack


# A validation this much worse than the best seen means the run has blown up, not merely
# regressed: healthy epoch-to-epoch movement here is a few percent. 5.0 was calibrated on run9's
# 50-100x blow-ups and then run10's ep34 damage -- a REAL 3.71x regression that froze the run --
# was stamped healthy under it. 2.0 still clears every healthy fluctuation ever observed (a few
# percent) by a wide margin.
HEALTH_RATIO = 2.0


def ktjd_channel_lut(base):
    """{rig: bool[J,17]} static channel-validity, assembled once (KTJD-17 only)."""
    rigs = {s["object_type"] for s in base.samples}
    return {r: torch.from_numpy(base.static_masks(r)["channel_valid"]) for r in rigs}


def ktjd_anchor(b, x17, mode, rest_lut, demo_frames):
    """[B,T,J,17] flow anchor. rest: per-rig frame broadcast over T. demo: the demo window's
    REAL frames tiled across the whole T axis (content prior; alignment-free by design)."""
    B, T, Jm, C = x17.shape
    if mode == "rest":
        a = torch.zeros(B, Jm, C, dtype=x17.dtype, device=x17.device)
        for k, ot in enumerate(b["object_type"]):
            r = rest_lut[ot]
            a[k, :r.shape[0]] = r.to(x17.device, x17.dtype)
        return a[:, None].expand(B, T, Jm, C)
    # demo: tile per sample by its real demo length
    anc = torch.zeros_like(x17)
    for k in range(B):
        d_real = int(b["frame_valid"][k, :demo_frames].sum())
        if d_real < 1:
            continue
        idx = torch.arange(T, device=x17.device) % d_real
        anc[k] = x17[k, idx]
    return anc


def ktjd_prep(b, lut, gammas):
    """Split the 18-plane KTJD batch: model sees x[...,:17]; plane 17 is the heading flag.
    Returns (x17, kwargs-for-cfm_loss). channel_valid is padded per batch from the rig LUT.
    gammas come from the versioned calibration artifact, never from in-code placeholders
    (codex round-S0); the trainer refuses to start without the artifact."""
    x = b["x"]
    x17 = x[..., :17].contiguous()
    heading = x[:, :, 0, 17] > 0.5                                  # [B,T]
    B, Jm = x.shape[0], x.shape[2]
    if "channel_valid" in b:
        # augmented batches carry their own per-sample masks (a sub-skeleton has no entry in the rig LUT)
        cv = b["channel_valid"].to(x.device)
    else:
        cv = torch.zeros(B, Jm, 17, dtype=torch.bool, device=x.device)
        for k, ot in enumerate(b["object_type"]):
            m = lut[ot]
            cv[k, :m.shape[0]] = m.to(x.device)
    return x17, dict(channel_valid=cv, heading_valid=heading,
                     gammas=gammas, group_spec=_GROUP_SPEC_KTJD17)


def atomic_save(obj, path: Path):
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    tmp.replace(path)                     # atomic on POSIX: no torn checkpoint on kill


def reset_val_stream(ds):
    """Make the next val pass replay the exact demo/crop choices of every previous pass."""
    ds._wrng_key = None


class fixed_torch_rng:
    """Deterministic torch RNG scope for validation.

    Resetting the DATA stream is not enough: cfm_loss draws t ~ rand and x0 ~ randn from the
    GLOBAL torch RNG, so two val passes over identical batches would still differ. Inside this
    scope the noise/t draws are identical every pass; global state is restored on exit so
    training randomness is untouched.
    """
    def __init__(self, seed): self.seed = seed
    def __enter__(self):
        self.cpu = torch.get_rng_state()
        self.gpu = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        torch.manual_seed(self.seed)
    def __exit__(self, *exc):
        torch.set_rng_state(self.cpu)
        if self.gpu is not None:
            torch.cuda.set_rng_state_all(self.gpu)


def connectivity_probe(model, b, demo_frames=DEMO_FRAMES, ktjd_lut=None, ktjd_gammas=None,
                       obj=None, probe_joint_sem=True):
    """P5: |dLoss/d input| per conditioning path. Zero = dead branch (v1's undetected failure).
    Magnitudes are comparable only against earlier probes of THIS run.

    `obj` carries the objective knobs (v_space/sigma_min/huber_delta) so the probe differentiates
    the loss the run is ACTUALLY training, not an unweighted MSE stand-in (codex 2026-08-22): with
    v_space on, an unweighted probe reports connectivity through a different objective than the
    optimizer sees, and the numbers would not be comparable across a sigma_min change either."""
    was_training = model.training
    model.train()
    c = cond_of(b)
    # `probe_joint_sem=False` is for an arm that BUILDS no joint-description pathway (the flat
    # baseline): differentiating through a tensor the graph never touched raises. Every other arm
    # keeps it in the list, so a per-joint model whose description branch went dead still raises
    # here instead of silently reporting a zero (codex 2026-09-10 #1).
    probed = ["text"] + (["joint_sem"] if probe_joint_sem else [])
    for k in probed:
        c[k] = c[k].detach().clone().requires_grad_(True)
    if ktjd_lut is not None:
        x_src, kt = ktjd_prep(b, ktjd_lut, ktjd_gammas)
    else:
        x_src, kt = b["x"], dict(gammas=KIMODO_GAMMAS)
    xin = x_src.detach().clone().requires_grad_(True)
    loss = cfm_loss(model, xin, is_target=b["is_target"], valid=b["valid"],
                    **(obj or {}), **kt, **c)
    grads = torch.autograd.grad(loss, [xin] + [c[k] for k in probed])
    g_x, g_t = grads[0], grads[1]
    model.zero_grad(set_to_none=True)
    if not was_training:
        model.eval()
    out = {"demo": float(g_x[:, :demo_frames].norm()),
           "target": float(g_x[:, demo_frames:].norm()), "text": float(g_t.norm())}
    out["joint_sem"] = float(grads[2].norm()) if probe_joint_sem else 0.0
    return out


def calib_arm_model_drift(calib: dict, want: dict, require_uniform: bool):
    """The uniform arm's artifact was measured ON A MODEL: it must be THIS arm's model, recorded with all four conditioning
    fields, or the mechanism check and the acceleration diagnostic certify a denoiser that still has the ingredients the arm
    removed (codex baseline r4 #2). Returns the refusal text, or None. Only the uniform arm compares: a calibrated arm is
    bound to its artifact by the pinned sha and by the calibration runner's own check, and some arms share another arm's
    artifact on purpose (the flat arm trains on the nodesc control's weights; codex 2026-09-15 unimate r4 P1-2), while
    artifacts older than the conditioning fields record the architecture only (codex unimate r3 P1)."""
    if not require_uniform:
        return None
    arm = (calib.get("protocol", {}).get("verify", {}) or {}).get("arm_model") or {}
    got = {k: arm.get(k) for k in want}
    if got != want:
        return f"was measured on a model with {got}, this run builds {want} -- re-measure it with the matching ARM_* conditioning"
    return None


def _calib_demo_drift(proto, run_demo_rest, run_demo_frames):
    """The mechanism check drives the arm model with a demo; the artifact certifies the objective only under
    that demo condition (codex 2026-09-04). Artifacts written before the field existed were all measured with
    the 1-frame rest demo, so absence means exactly that -- a 64-frame motion-demo run must recalibrate."""
    has_r, has_f = "demo_rest" in proto, "demo_frames" in proto
    if not has_r and not has_f:
        art = (True, 1)          # legacy artifact: both fields absent == measured with the 1-frame rest demo
    elif has_r and has_f:
        r, f = proto["demo_rest"], proto["demo_frames"]
        # exact types, no coercion (codex 2026-09-04 round 2): 64.9 / "64" / [] / 0 must not pass as 64 / False
        if (not isinstance(r, bool) or isinstance(f, bool) or not isinstance(f, int) or f < 1
                or (r and f != 1)):
            return [f"demo condition malformed in the calibration artifact (demo_rest={r!r}, "
                    f"demo_frames={f!r}); expected a bool and a positive int with rest => frames == 1 -- recalibrate"]
        art = (r, f)
    else:
        return ["demo condition half-recorded in the calibration artifact (only one of demo_rest / "
                "demo_frames is present) -- recalibrate"]
    # the run side is compared parser-native as well (codex 2026-09-04 round 3): --demo_rest is a store_true bool,
    # --demo_frames a non-bool positive int; anything else is an invocation error, not a drift to explain away
    if (not isinstance(run_demo_rest, bool) or isinstance(run_demo_frames, bool)
            or not isinstance(run_demo_frames, int) or run_demo_frames < 1
            or (run_demo_rest and run_demo_frames != 1)):
        raise SystemExit(f"[refuse] run demo condition malformed (demo_rest={run_demo_rest!r}, "
                         f"demo_frames={run_demo_frames!r}); expected a bool and a positive int with rest => frames == 1")
    run = (run_demo_rest, run_demo_frames)
    if art != run:
        return [f"demo condition (rest={art[0]}, frames={art[1]}) != run (rest={run[0]}, frames={run[1]}) "
                f"-- recalibrate with DEMO_REST={'1' if run[0] else '0'} DEMO_FRAMES={run[1]}"]
    return []


def _calib_batch_gate(proto, run_batch, calib_sha, calib_path, resume):
    """protocol.batch must be a positive integer (bool / float are not) equal to --batch, for every view:
    the grouped loss normalises per batch, so the mechanism check certifies the objective only at the
    batch it ran with (codex 2026-09-03). Single exception, announced not silent: a --resume whose
    checkpoint already trained at exactly this --batch under exactly this artifact (path AND byte sha
    pinned in ktjd_pins) -- the run predates the rule and continues without a batch-matched certificate.
    Returns the warning text when the exception is taken, None when the batch matches; refuses otherwise."""
    pb = proto.get("batch")
    if isinstance(pb, bool) or not isinstance(pb, int) or pb <= 0:
        raise SystemExit(f"[refuse] gamma calibration protocol.batch={pb!r} is not a positive integer; "
                         f"recalibrate with CALIB_BATCH={run_batch}")
    ra = rp = None
    if resume is not None:
        # the RAW loaded checkpoint: a non-mapping payload or non-mapping args / ktjd_pins is refused here,
        # whether or not the batch matches (codex 2026-09-03 r4)
        ra, rp = (resume.get("args"), resume.get("ktjd_pins")) if isinstance(resume, dict) else (None, None)
        if not isinstance(ra, dict) or not isinstance(rp, dict):
            raise SystemExit("[refuse] the checkpoint to resume carries malformed args / ktjd_pins metadata -- it "
                             "cannot certify what it trained under")
    if pb == run_batch:
        return None
    if resume is not None:
        rb = ra.get("batch")
        # type(rb) is int: 16.0 == 16 and True == 1 are Python truths but not the same training batch
        if (type(rb) is int and rb == run_batch and str(ra.get("ktjd_gamma_calib")) == str(calib_path)
                and str(rp.get("gamma_calib_sha256")) == str(calib_sha)):
            return (f"[calib] LEGACY: resuming a checkpoint that already trained at --batch {run_batch} under this "
                    f"exact artifact ({calib_path}, sha {calib_sha[:12]}, mechanism-checked at batch {pb}); the run "
                    f"predates the batch-matched protocol and carries NO batch-matched certificate")
    raise SystemExit(f"[refuse] gamma calibration was mechanism-checked at batch {pb} but this run uses "
                     f"--batch {run_batch}; recalibrate with CALIB_BATCH={run_batch}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default="data/animo4d_L4TB_plus_human_v4b272neutral")
    ap.add_argument("--splits_dir", default="data/holdout_splits_v1")
    ap.add_argument("--joint_sem", default="data/joint_semantics_llm2vec_v1.npz")
    ap.add_argument("--caption_cache", default="data/anytop_caption_llm2vec_v4b272neutral_multi")
    ap.add_argument("--texts_json", default="motion_texts_by_file_clean_v1.json")
    ap.add_argument("--out", default="runs/v2_incontext_run1")
    ap.add_argument("--dim", type=int, default=256)
    ap.add_argument("--depth", type=int, default=6)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--grad_accum", type=int, default=1,
                    help="micro-batches of --batch per optimizer step (gradient accumulation). Each micro-batch is "
                         "normalised on its own by cfm_loss, so R ranks x B x accum A weights every cell exactly as "
                         "R*A ranks x B do; the lr schedule, warmup, FK ramp, spike guard and resync count optimizer steps")
    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--val_every", type=int, default=5)
    ap.add_argument("--ckpt_every", type=int, default=25)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resume", default="")
    # ---- run-3 scale knobs (all defaults preserve run-1 behaviour bit-for-bit) ----
    ap.add_argument("--corpus", choices=("truebones", "pzh", "ktjd17"), default="truebones",
                    help="pzh = Planet-Zoo + HumanML3D, no TrueBones (312 rigs / 89.5k train clips)")
    ap.add_argument("--balance", choices=("rig", "clip"), default="rig",
                    help="rig = uniform-skeleton draws (run-1); clip = natural source proportions")
    ap.add_argument("--random_caption", action="store_true",
                    help="rotate ALL captions per clip (median 3) instead of cap0 only")
    ap.add_argument("--demo_frames", type=int, default=DEMO_FRAMES)
    ap.add_argument("--target_frames", type=int, default=TARGET_FRAMES)
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--wd", type=float, default=0.0)
    ap.add_argument("--warmup_steps", type=int, default=2000,
                    help="linear lr warmup over N optimizer steps. DEFAULT RAISED FROM 0 to 2000 "
                         "(2026-08-21): every run so far started at full lr, and the user's "
                         "requirement for this run is a stable, monotonically falling early loss. "
                         "Warmup is the standard, cheap way to get it and costs ~0.2 epochs on the "
                         "312-rig corpus.")
    ap.add_argument("--val_max_batches", type=int, default=0,
                    help=">0: cap validation at N batches (fixed deterministic subset) so peer "
                         "ranks are not parked behind a long rank-0 val")
    ap.add_argument("--val_every_steps", type=int, default=0,
                    help=">0: validate/checkpoint every N steps INSTEAD of every val_every epochs")
    ap.add_argument("--param_resync_steps", type=int, default=200,
                    help="broadcast rank 0's parameters every N optimizer steps. DDP syncs only "
                         "gradients and assumes identical updates; at extreme clip scaling that "
                         "assumption fails for the smallest-gradient tensors. 0 disables.")
    ap.add_argument("--qk_norm", action="store_true",
                    help="RMS-normalise q,k per head before the dot product (ViT-22B recipe). "
                         "Measured: without it run10's block-0 temporal attention ran at logit "
                         "1372-1483 while HEALTHY and 12392+ across the damage step.")
    ap.add_argument("--allow_calib_reswap", action="store_true",
                    help="Resume across a gamma-calibration ARTIFACT swap whose gammas and group "
                         "spec are BIT-IDENTICAL to the checkpoint's (verified here, not "
                         "trusted). The legitimate case: non-loss code (e.g. the sampler) "
                         "changed dit_motion.py, the calib code-hash guard demanded a "
                         "re-measurement, and the re-measured artifact is numerically the same "
                         "objective under a new file. Recorded in calib_history on every "
                         "checkpoint this run writes.")
    ap.add_argument("--allow_schedule_restart", action="store_true",
                    help="permit lr_scheduler/eta_min_ratio/lr_decay_epochs to differ from the "
                         "checkpoint ON RESUME, recording {old,new,epoch,gstep} into "
                         "schedule_history. Everything else in the crit list still refuses. "
                         "Motivated by run10: the model outgrew its own lr floor.")
    ap.add_argument("--allow_unhealthy_resume", action="store_true",
                    help="continue from a checkpoint written after a blow-up (refused by default)")
    ap.add_argument("--ckpt_snapshot_keep", type=int, default=20,
                    help="how many step snapshots to retain; older ones are deleted. They carry "
                         "optimizer state (~1 GB each), so an unbounded series fills the disk.")
    ap.add_argument("--ckpt_snapshot_steps", type=int, default=0,
                    help=">0: periodic epNNN-style snapshots every N steps instead of ckpt_every epochs")
    ap.add_argument("--t_sampler", choices=("uniform", "logitnormal"), default="uniform")
    ap.add_argument("--v_space", action="store_true", help="JiT velocity-space loss (clamped 1/(1-t)^2)")
    ap.add_argument("--p_drop_text", type=float, default=0.0)
    ap.add_argument("--p_drop_demo", type=float, default=0.0)
    ap.add_argument("--p_drop_both", type=float, default=0.0)
    ap.add_argument("--gamma_fk", type=float, default=0.0,
                    help="Kimodo Eq.1 gamma7 FK<->RIC consistency weight on the bone-scaled "
                         "residual (0.25 = slope-equivalent of Kimodo's raw-unit 5.0, the "
                         "calibrated launch value; 0 = off, the pre-2026-08-19 objective)")
    ap.add_argument("--fk_warmup_steps", type=int, default=5000,
                    help="linear ramp of gamma_fk over the first N global steps (hy273 recipe); "
                         "only read when gamma_fk > 0")
    ap.add_argument("--epoch_draws", type=int, default=0,
                    help="draws per training epoch (0 = the corpus size, the historical default). The "
                         "balanced sampler draws with replacement, so an epoch is a number of draws, not "
                         "a pass; stating it lets an arm trained on a LARGER cut keep the control's "
                         "steps per epoch -- and therefore the control's lr decay horizon, which is "
                         "written in epochs -- instead of silently taking more of both.")
    ap.add_argument("--rig_multiplicity", default="",
                    help="RIG:N[,RIG:N] -- draw a named rig as if it were N rigs under --balance rig. For a corpus "
                         "holding one topology with many rigs' worth of clips: uniform-over-rigs gives it 1/N of the "
                         "samples, uniform-over-clips lets it take a quarter of the batch. Empty = unchanged draws.")
    ap.add_argument("--flat_joints", type=int, default=0,
                    help="train the flat padded-joint baseline instead of the per-joint model: one token per FRAME "
                         "holding every joint's channels zero-padded to this many joints, no joint descriptions, no "
                         "skeleton attention bias, no structural features (src/models/v2/dit_flat.py)")
    ap.add_argument("--struct_feats", action="store_true",
                    help="graph-v2 knife 1: structural joint features (rest offset/bone/depth/"
                         "children/leaf) added to joint tokens")
    ap.add_argument("--dir_bias", action="store_true",
                    help="graph-v2 knife 2: learnable per-head directional (up/down LCA hop) "
                         "attention bias added to the fixed -geodesic scalar")
    ap.add_argument("--ktjd_root", default="dataset/ktjd17_truebones",
                    help="ktjd17 corpus root (only read when --corpus ktjd17)")
    ap.add_argument("--anchor", choices=("none", "rest", "demo"), default="none",
                    help="flow base = anchor + N(0,I) (UMO source-centered). rest = per-rig "
                         "rest-pose frame (variant A); demo = tiled demo window (variant B). "
                         "ktjd17 only.")
    ap.add_argument("--identity_p", type=float, default=0.0,
                    help="variant-B identity branch: prob that target IS the demo clip with a "
                         "ZEROED caption (UMO SOURCE_IDENTITY analogue)")
    ap.add_argument("--ktjd_training_authorized", action="store_true",
                    help="the KTJD release gate currently says ready_for_training=false; this "
                         "flag records an EXPLICIT user authorization to train anyway (user "
                         "2026-08-20: gate override (b), data artifact untouched). Without it, "
                         "ktjd17 training refuses to start.")
    ap.add_argument("--rep_norm", choices=("percell", "scale_only", "rest"), default="percell",
                    help="representation ablation (user 2026-09-06): per-cell mean/std (the method) or the KTJD "
                         "spec's scale-only normalization (the old representation); recorded in ktjd_pins as "
                         "target_centering/normalization and must match the gamma calibration's protocol")
    # skeleton-robustness augmentation (user 2026-09-07; src/data/ktjd17_augment.py). ktjd17 + train split only;
    # --aug_p 0 (default) = off and byte-identical batches. Recorded in ktjd_pins.augmentation and compared with
    # the gamma calibration's protocol.augmentation (absent there = measured without augmentation).
    ap.add_argument("--freeze_zero_joint_sem", action="store_true",
                    help="simplified-baseline arm (user 2026-09-08): zero the joint-description projection at init and freeze it, so the "
                         "model never receives joint semantics; the checkpoint stays loadable by every consumer (the projection is zero)")
    ap.add_argument("--require_uniform_gammas", action="store_true",
                    help="simplified-baseline arm (codex baseline r3 #3): refuse any calibration artifact whose group weights are "
                         "not all exactly 1.0, so an inherited or re-measured CALIB path cannot restore the calibrated weights")
    ap.add_argument("--no_geo_bias", dest="geo_bias", action="store_false",
                    help="simplified-baseline arm (codex baseline r1 #1): the spatial attention receives no geodesic-distance bias "
                         "(padding mask kept); recorded as geo_bias=False in the checkpoint args and enforced inside the model")
    ap.add_argument("--aug_p", type=float, default=0.0, help="probability a training sample is augmented (0 = off)")
    ap.add_argument("--aug_drop_max_frac", type=float, default=0.0,
                    help="sub-skeleton: max fraction of droppable joints removed (root / contact joints never)")
    ap.add_argument("--aug_drop_mode", choices=("any", "tips"), default="any",
                    help="sub-skeleton: 'any' joint (children re-parented) or 'tips' (prune leaves only; FK stays exact)")
    ap.add_argument("--aug_rest_deg", type=float, default=0.0, help="rest-convention: max per-joint rotation, degrees")
    ap.add_argument("--aug_sem_noise", type=float, default=0.0, help="description embeddings: noise std / row RMS")
    ap.add_argument("--aug_sem_drop_p", type=float, default=0.0, help="description embeddings: P(zero the whole table)")
    ap.add_argument("--aug_stats_logsd", type=float, default=0.0, help="statistics: log-normal std factor sigma")
    ap.add_argument("--aug_stats_shift", type=float, default=0.0, help="statistics: mean shift sigma (in stds)")
    ap.add_argument("--aug_bone_scale", type=float, default=0.0,
                    help="kinematics-preserving: per-bone length factor uniform in [1-s, 1+s]; positions/velocities re-encoded by FK")
    ap.add_argument("--aug_pool_frac", type=float, default=0.0,
                    help="kinematics-preserving: max fraction of single-child interior joints pooled away (child re-parented; FK re-encoded)")
    ap.add_argument("--aug_add_p", type=float, default=0.0,
                    help="kinematics-preserving: P(insert one synthetic joint on a random bone, rigid with its parent)")
    ap.add_argument("--aug_mode", choices=("joint", "one_of"), default="joint",
                    help="'joint': every enabled --aug_* perturbation on every augmented sample; 'one_of': UniMate's rule -- one of "
                         "add / remove / pool / scale per augmented sample with UniMate's own rates (src.data.ktjd17_augment.ONE_OF); "
                         "needs --aug_drop_mode tips and --aug_bone_scale > 0, every other --aug_* 0")
    ap.add_argument("--ktjd_gamma_calib", default="configs/ktjd17_gamma_calibration_v5.json",
                    help="versioned KTJD gamma calibration artifact (energies + gammas + hashes); "
                         "ktjd17 training REFUSES to start without it (codex round-S0)")
    ap.add_argument("--ktjd_auth_generation",
                    default="20260819T215405576671Z-2d04a8d85638",
                    help="the generation --ktjd_training_authorized was granted for (user "
                         "2026-08-20); a different corpus needs a fresh authorization")
    ap.add_argument("--allow_calib_code_drift", action="store_true",
                    help="proceed even though the loss code changed since gamma calibration "
                         "(recorded in args.json; use only when the change provably cannot move "
                         "group shares)")
    ap.add_argument("--grad_spike_reject", type=float, default=0.0,
                    help="skip (not clip) any post-warmup step whose PRE-clip gradient norm "
                         "exceeds this; 0 disables. Clipping keeps a garbage direction at full "
                         "step size, which is how six runs died.")
    ap.add_argument("--grad_clip", type=float, default=1.0,
                    help="clip_grad_norm_ threshold. NOTE the Kimodo gammas put the raw grad norm "
                         "around 150-400 on this objective, so the historical 1.0 renormalizes "
                         "EVERY step to unit length rather than clipping outliers -- which cancels "
                         "Adam's scale-invariance and makes every step the same size regardless of "
                         "how sharp the local landscape is. Set it above the norm distribution to "
                         "clip only genuine spikes.")
    ap.add_argument("--artic_min", type=float, default=0.30,
                    help="anti-collapse floor on the pose-relative articulation ratio (codex "
                         "round-S8): below this the body is effectively frozen. A checkpoint under "
                         "the floor can never be recorded as BEST, and N consecutive validations "
                         "under it abort the run. 0 disables (not recommended).")
    ap.add_argument("--artic_gate_after", type=int, default=30,
                    help="epoch from which the artic floor is enforced (an untrained model is "
                         "legitimately below it)")
    ap.add_argument("--artic_gate_strikes", type=int, default=3,
                    help="consecutive sub-floor validations that abort the run")
    ap.add_argument("--limit_train_clips", type=int, default=0,
                    help="OVERFIT RUNG (user 2026-08-21: 'at least it must be able to overfit'): "
                         ">0 restricts training to the first N train clips, deterministically. A "
                         "model that cannot drive the loss toward zero on a handful of clips has "
                         "an architecture or plumbing fault, and no amount of data will fix it -- "
                         "so this runs BEFORE any long run. Val is left untouched.")
    ap.add_argument("--demo_rest", action="store_true",
                    help="1-frame REST-POSE demo (user 2026-08-21): the demo slot carries the "
                         "rig's rest pose instead of a window of another clip, so it holds no "
                         "motion content. Requires --demo_frames 1. ktjd17 only.")
    ap.add_argument("--lr_scheduler", choices=("half_cosine", "none"), default="none",
                    help="PORTED from this project's locked CodeFlow recipe "
                         "(scripts/train_graph_codeflow.py:845): linear warmup -> half-cosine "
                         "decay to eta_min_ratio*lr, computed from the OPTIMIZER step so a resume "
                         "reproduces it exactly without storing scheduler state. Default 'none' "
                         "preserves the flat-lr behaviour of runs 1-5 -- which is what made every "
                         "one of them diverge: lr stayed at its peak forever, so as the model "
                         "sharpened, the fixed step size eventually exceeded what its accuracy "
                         "could tolerate. That is why crash time scaled with 1/lr and why wd and "
                         "sigma_min only postponed it.")
    ap.add_argument("--eta_min_ratio", type=float, default=0.01,
                    help="floor of the cosine decay as a fraction of --lr (CodeFlow default 0.01)")
    ap.add_argument("--lr_decay_epochs", type=int, default=0,
                    help="length of the cosine decay in EPOCHS; 0 = the full --epochs budget. "
                         "Decoupled from --epochs because a 500-epoch cosine is effectively FLAT "
                         "where this model actually fails: at epoch 18 (run5's death) it has "
                         "covered 3.6%% of the horizon and lr has fallen 0.27%% -- it cannot test "
                         "the hypothesis it exists to test (codex 2026-08-23 blocker 2). A shorter "
                         "horizon front-loads the decay; lr then holds at eta_min_ratio*lr for the "
                         "remainder, so pick eta_min_ratio high enough to keep learning after it.")
    ap.add_argument("--compile", action="store_true",
                    help="torch.compile(dynamic=True) on the model. MEASURED on this objective "
                         "(88M dim512/depth12, real varied-J batches, single H200): 11.9 -> 19.5 "
                         "items/s (+64%%), which more than repays activation checkpointing's 27%% "
                         "cost. dynamic=True is REQUIRED -- J varies per batch (87..96 observed in "
                         "one shuffled stream) and static compilation would re-trigger on every "
                         "new shape. First few batches cost ~76 s of compilation.")
    ap.add_argument("--grad_ckpt", action="store_true",
                    help="activation checkpointing on the transformer blocks: ~60-70%% less "
                         "activation memory for ~30%% more compute. Needed to raise dim/depth "
                         "while KEEPING global batch 32 -- shrinking the batch instead would "
                         "worsen the very tail-sample-count problem that drives the instability.")
    ap.add_argument("--sigma_min", type=float, default=0.05,
                    help="floor on (1-t) inside the v_space weight 1/(1-t)^2, i.e. the cap on how "
                         "much the near-data timesteps outweigh the rest. MEASURED on this "
                         "objective (run4 ep9, healthy): gradient norm by t-bin is 0.64 at "
                         "t in [0.2,0.5] but 34.66 at t in [0.95,1.0] -- a 54x concentration, of "
                         "which ~10x is this weight (capped at 400, unit-mean-normalized by "
                         "2/sigma_min-1 = 39) and ~5x is the higher parameter-sensitivity of the "
                         "near-data region itself. That concentration grows as the model improves "
                         "(the mid-range residual shrinks while t->1 does not), which is why all "
                         "four runs diverged at the same VAL level rather than at an lr threshold. "
                         "0.2 caps the weight at 25 (2.8x after normalization).")
    ap.add_argument("--huber_delta", type=float, default=0.0,
                    help="knee of a Huber on the per-cell target error, in normalized units. 0 "
                         "keeps the plain squared error. Below the knee the term is IDENTICAL to "
                         "the squared error, so calibrated gammas still describe the objective; "
                         "above it the per-cell gradient saturates at 2*delta, which is what stops "
                         "undetected source-data contamination from dominating a step. Measured "
                         "|normalized target| over 4.63e8 supervised cells: p99.9=6.65, "
                         "p99.99=11.45, p99.999=19.95.")
    ap.add_argument("--exclude_clips", default="",
                    help="JSON artifact listing clip_ids removed from every split (the frozen, "
                         "sha-pinned corpus cannot be edited, so the cut is applied at load time "
                         "and its sha is recorded in ktjd_pins). User 2026-08-21: source-animation "
                         "teleports are not to be trained on.")
    ap.add_argument("--ktjd_percell_stats", default="data/ktjd17_percell_stats_v1.npz",
                    help="old-style per-(rig,joint,channel) mean/std artifact")
    ap.add_argument("--ref_text", action="store_true",
                    help="F5-TTS reference-transcript analogue: feed the DEMO's caption too, "
                         "injected PER FRAME (demo caption over demo frames, request over target "
                         "frames) so the model can factor the demo's content out and keep its "
                         "style. Off = bit-identical to the pre-2026-08-20 arms.")
    ap.add_argument("--gamma_vel", type=float, default=0.0,
                    help="UMO clean_root/joint_velocity analogue (0.01 in their recipe): "
                         "physical-space frame-difference supervision a FROZEN output cannot "
                         "satisfy. ktjd17 only.")
    ap.add_argument("--gamma_lock", type=float, default=0.0,
                    help="UMO foot_lock analogue (0.01): zero displacement demanded only where GT "
                         "contact is on at both endpoints. ktjd17 only.")
    ap.add_argument("--gamma_acc", type=float, default=0.0,
                    help="acceleration-matching weight: MSE between the prediction's and GT's "
                         "temporal second difference on the normalized channels (anti-jitter, "
                         "plan-a 2026-08-28); 0 = off")
    ap.add_argument("--two_stage", action="store_true",
                    help="variant D: Kimodo/UMO two-stage denoiser (root tower -> parameter-free "
                         "bridge, detached in training -> body tower). ktjd17 only.")
    ap.add_argument("--root_dim", type=int, default=192,
                    help="two_stage root tower width (depth fixed at 4)")
    # ---- per-species LoRA fine-tuning (user 2026-09-02) ----
    ap.add_argument("--init_from", default="",
                    help="load ONLY the model weights of this checkpoint (no optimizer/epoch/pins): "
                         "the frozen backbone a LoRA run adapts")
    ap.add_argument("--lora_r", type=int, default=0, help="LoRA rank; 0 = no LoRA (full training)")
    ap.add_argument("--lora_alpha", type=float, default=64.0, help="LoRA scale numerator (scale = alpha / r)")
    ap.add_argument("--lora_dropout", type=float, default=0.0)
    ap.add_argument("--lora_targets", default="attn,ffn,cond",
                    help="comma list of src.models.v2.lora.TARGET_GROUPS keys")
    a = ap.parse_args()
    aug_cfg = AugConfig(p=a.aug_p, drop_max_frac=a.aug_drop_max_frac, drop_mode=a.aug_drop_mode, rest_deg=a.aug_rest_deg,
                        sem_noise=a.aug_sem_noise, sem_drop_p=a.aug_sem_drop_p,
                        stats_logsd=a.aug_stats_logsd, stats_shift=a.aug_stats_shift,
                        bone_scale=a.aug_bone_scale, pool_frac=a.aug_pool_frac, add_p=a.aug_add_p, mode=a.aug_mode)
    if aug_cfg.active and a.corpus != "ktjd17":
        raise SystemExit("[refuse] --aug_* is ktjd17-only (needs static_masks and the FK skeleton fields)")
    if aug_cfg.active and a.anchor == "rest":
        raise SystemExit("[refuse] --anchor rest keeps a per-rig rest anchor; a sub-skeleton sample has no entry "
                         "in that table -- use --anchor none/demo with --aug_p > 0")
    if a.lora_r > 0 and not a.init_from:
        raise SystemExit("[refuse] --lora_r needs --init_from <backbone ckpt>: a LoRA adapts a trained model")
    if a.lora_r > 0 and a.resume:
        raise SystemExit("[refuse] --resume is not supported for LoRA runs (they are short; restart from --init_from)")
    if a.lora_r > 0 and a.two_stage:
        raise SystemExit("[refuse] LoRA is wired for InContextMotionDiT only")
    # Objective-shaping knobs are validated UNCONDITIONALLY: they reach cfm_loss on every corpus,
    # so a guard inside the ktjd17 branch would let another corpus pass an invalid value straight
    # through (codex 2026-08-22 hygiene). 2/s-1 is only defined on (0, 1].
    # nan/inf must be rejected here, not just negatives: `nan != 0` passes the resume's
    # "0 -> positive" activation check and gets RECORDED as an activation, while `nan > threshold`
    # is always False -- so the lineage would claim a guard that is in fact disabled. inf is the
    # same defect with a different value (codex 2026-08-23 round 2).
    if a.ckpt_snapshot_steps > 0 and a.ckpt_snapshot_keep < 1:
        raise SystemExit(f"--ckpt_snapshot_keep must be >= 1 when snapshots are enabled "
                         f"(0 retains everything, negatives delete the file just written), got "
                         f"{a.ckpt_snapshot_keep}")
    if a.grad_accum < 1:
        raise SystemExit(f"[refuse] --grad_accum must be >= 1, got {a.grad_accum}")
    if not (np.isfinite(a.grad_spike_reject) and a.grad_spike_reject >= 0.0):
        raise SystemExit(f"[refuse] --grad_spike_reject must be finite and >= 0, got "
                         f"{a.grad_spike_reject}")
    if not (np.isfinite(a.sigma_min) and 0.0 < a.sigma_min <= 1.0):
        raise SystemExit(f"[refuse] --sigma_min must lie in (0, 1], got {a.sigma_min}")
    if not (np.isfinite(a.grad_clip) and a.grad_clip > 0):
        raise SystemExit(f"[refuse] --grad_clip must be finite and positive, got {a.grad_clip}")
    if not (0.0 <= a.eta_min_ratio <= 1.0):
        raise SystemExit(f"[refuse] --eta_min_ratio must lie in [0, 1], got {a.eta_min_ratio}")
    if a.corpus == "ktjd17" and a.joint_sem == ap.get_default("joint_sem"):
        # the legacy AnyTop table fails the KTJD order-hash on the first rig; select the KTJD
        # table when the user did not explicitly choose one (codex round-S0)
        a.joint_sem = "data/joint_semantics_llm2vec_ktjd17_v1.npz"
        print(f"[ktjd] --joint_sem defaulted to {a.joint_sem}", flush=True)
    assert torch.cuda.is_available(), "run-1 is a GPU run; refusing to silently train on CPU"
    # ---- DDP is opt-in via torchrun's env; absent WORLD_SIZE keeps the single-GPU path
    # bit-identical (run-1 and its crash-resume must not change behaviour). Cross-alloc same-node
    # specifics (static rendezvous, NCCL_P2P/SHM disable, IB socket) live in the LAUNCHER, not here.
    ddp = int(os.environ.get("WORLD_SIZE", "1")) > 1
    # A stray RANK/LOCAL_RANK without WORLD_SIZE must NOT leak into seeding or is_main:
    # single-GPU behaviour is pinned bit-identical to the pre-DDP trainer.
    rank = int(os.environ.get("RANK", "0")) if ddp else 0
    local_rank = int(os.environ.get("LOCAL_RANK", "0")) if ddp else 0
    if ddp:
        dist.init_process_group("nccl")
        torch.cuda.set_device(local_rank)
    dev = f"cuda:{local_rank}" if ddp else "cuda"
    is_main = rank == 0
    torch.manual_seed(a.seed + rank); np.random.seed(a.seed + rank)
    out = Path(a.out)
    if is_main:
        out.mkdir(parents=True, exist_ok=True)
    if ddp:
        dist.barrier()

    # ---------------- data ----------------
    if a.corpus == "ktjd17":
        # KTJD-17 (17 real channels + heading-flag plane 18 riding through the crop machinery;
        # see src/data/ktjd17_incontext.py). InContextPairs itself is reused unchanged.
        from src.data.ktjd17_incontext import Ktjd17Base, ktjd17_split_names
        base = Ktjd17Base(a.ktjd_root, caption_emb_cache=a.caption_cache,
                          joint_semantics=a.joint_sem, texts_json=a.texts_json,
                          percell_stats=a.ktjd_percell_stats,
                          exclude_clips=(a.exclude_clips or None),
                          random_caption=a.random_caption, normalization=a.rep_norm)
        names = ktjd17_split_names(a.ktjd_root, exclude=(a.exclude_clips or None))
        types = None                       # all KTJD rigs; splits already carve train/val/held
        # ---- external release gate (codex round-S0): optimization against this corpus is gated
        # by the DATA side, and the gate is checked against the EXACT generation the adapter
        # resolved. The override flag records user authorization without mutating the artifact.
        # Two corpus layouts carry the data-side release differently, so resolve which one this
        # corpus uses and reduce both to (gate_generation, ready, gate_desc):
        #   TrueBones     external dataset/KTJD17_TRUEBONES_RELEASE_GATE.json, ready_for_training
        #   PZ+Human 312  self-bound: generation.json.full_conversion_authorized, with the visual
        #                 gate pinned by sha inside the corpus (no external file exists)
        genj = json.loads((Path(a.ktjd_root) / "generation.json").read_text())
        # Layout detection by the artifact that actually exists, not by a key name: the TrueBones
        # generation.json ALSO carries `full_conversion_authorized`, so keying on it sent every
        # TrueBones-derived corpus (e.g. the per-species LoRA root) down the PZ branch and refused
        # it for a status string that layout never uses (2026-09-02).
        tb_gate_p = Path(a.ktjd_root).parent / "KTJD17_TRUEBONES_RELEASE_GATE.json"
        tb_layout = tb_gate_p.is_file() and str(json.loads(tb_gate_p.read_text())["generation"]
                                               ["generation_id"]) == str(genj.get("generation_id"))
        if not tb_layout:
            vg_p = Path(a.ktjd_root) / "evidence" / "visual_gate.json"
            vg_sha = hashlib.sha256(vg_p.read_bytes()).hexdigest()
            if vg_sha != str(genj.get("visual_gate_sha256")):
                raise SystemExit(f"[refuse] corpus generation.json pins visual gate "
                                 f"{genj.get('visual_gate_sha256')} but {vg_p} hashes {vg_sha}")
            verdict = str(json.loads(vg_p.read_text()).get("verdict", "")).lower()
            if verdict != "pass":
                raise SystemExit(f"[refuse] corpus visual gate verdict is {verdict!r}, not 'pass'")
            if str(genj.get("status")) != "full_numeric_pass_visual_gate_bound":
                raise SystemExit(f"[refuse] corpus status is {genj.get('status')!r}, expected "
                                 f"'full_numeric_pass_visual_gate_bound'")
            gate_generation = str(genj.get("generation_id"))
            # `is True`, not bool(): bool("false") and bool(0.0) both mislead here.
            ready = genj.get("full_conversion_authorized") is True
            gate_desc = "generation.json full_conversion_authorized (visual gate pass, sha-bound)"
        else:
            gate_p = tb_gate_p
            gate = json.loads(gate_p.read_text())
            gate_generation = str(gate["generation"]["generation_id"])
            ready = bool(gate.get("ready_for_training", False))
            gate_desc = "KTJD17_TRUEBONES_RELEASE_GATE.json ready_for_training"
        if gate_generation != base.generation_id:
            raise SystemExit(f"[refuse] release gate pins generation {gate_generation} but the "
                             f"corpus resolved {base.generation_id} -- gate and data disagree")
        if is_main:
            print(f"[ktjd] data-side release gate: {gate_desc} = {ready}", flush=True)
        if not ready:
            # fail-CLOSED-ish (codex round-S8 fail-open #1): the override must name the exact
            # generation it was granted for, so a flag cannot silently carry to a new corpus.
            if a.ktjd_training_authorized and a.ktjd_auth_generation != base.generation_id:
                raise SystemExit(f"[refuse] --ktjd_training_authorized was granted for generation "
                                 f"{a.ktjd_auth_generation!r}, corpus is {base.generation_id!r}")
            if not a.ktjd_training_authorized:
                raise SystemExit(f"[refuse] KTJD data-side release gate ({gate_desc}) is false. "
                                 f"Training needs either the data-side flag flip or "
                                 f"--ktjd_training_authorized (explicit user override, recorded "
                                 f"in args.json).")
            if is_main:
                print(f"[ktjd] GATE OVERRIDE: {gate_desc}=false, proceeding under "
                      "--ktjd_training_authorized (user 2026-08-20)", flush=True)
        # ---- gamma calibration artifact (codex round-S0): placeholders must not train ----
        calib_p = Path(a.ktjd_gamma_calib)
        if not calib_p.exists():
            raise SystemExit(f"[refuse] KTJD gamma calibration artifact {calib_p} missing -- "
                             f"run scripts/_measure_ktjd17_gamma_calibration.py first; "
                             f"placeholder gammas do not train (codex round-S0)")
        # ONE snapshot of the artifact: parsed, hashed for the batch gate and pinned into the checkpoint from the
        # same bytes (codex 2026-09-04 round 2: re-reading the file for each hash left a TOCTOU window)
        calib_bytes = calib_p.read_bytes()
        calib = json.loads(calib_bytes)
        calib_sha = hashlib.sha256(calib_bytes).hexdigest()
        if str(calib["generation_id"]) != base.generation_id:
            raise SystemExit(f"[refuse] gamma calibration measured on generation "
                             f"{calib['generation_id']}, corpus is {base.generation_id}")
        if str(calib.get("target_centering")) != base.provenance["target_centering"]:
            raise SystemExit(f"[refuse] gamma calibration was measured on target_centering="
                             f"{calib.get('target_centering')!r}, data now serves "
                             f"{base.provenance['target_centering']!r} -- the energies do not "
                             f"transfer across a re-parameterization of the target")
        if str(calib.get("percell_sha256")) != base.provenance["percell_sha256"]:
            raise SystemExit("[refuse] gamma calibration was measured against a different "
                             "per-cell stats artifact (sha mismatch) -- recalibrate")
        # the calibration measured gradient shares THROUGH the loss code; if that code changed,
        # the artifact no longer describes this objective (codex round-S8 fail-open #2). Recorded
        # but never compared was the defect.
        # exactly two measuring scripts may vouch for an artifact, anchored to THIS repo (the one
        # the trainer is imported from), not to the CWD: no absolute paths, no symlinks, no
        # resolve() outside the repo root (codex 2026-09-02 round 4)
        _REPO = Path(__file__).resolve().parents[1]
        _CALIB_SCRIPTS = {(_REPO / x).resolve() for x in ("scripts/_measure_ktjd17_gamma_calibration.py",
                                                          "scripts/_measure_ktjd17_gamma_calibration_view.py",
                                                          "scripts/_measure_ktjd17_gamma_calibration_view_v2.py")}
        _rel = str(calib["hashes"].get("code_script", "scripts/_measure_ktjd17_gamma_calibration.py"))
        _calib_script = _REPO / _rel
        if (Path(_rel).is_absolute() or ".." in Path(_rel).parts        # no escape-and-return paths
                or _calib_script.is_symlink() or not _calib_script.is_file()
                or _calib_script.resolve() not in _CALIB_SCRIPTS
                or _REPO not in _calib_script.resolve().parents):
            raise SystemExit(f"[refuse] gamma calibration names a measuring script outside the repo allowlist: "
                             f"{_rel!r}")
        _dit_src = _REPO / "src" / "models" / "v2" / "dit_motion.py"
        code_now = hashlib.sha256(_dit_src.read_bytes() + _calib_script.read_bytes()).hexdigest()
        if str(calib["hashes"].get("code_sha256")) != code_now and not a.allow_calib_code_drift:
            raise SystemExit("[refuse] gamma calibration was measured against different loss code "
                             "(dit_motion.py / the calibration script changed since). Recalibrate, "
                             "or pass --allow_calib_code_drift with a reason if the change provably "
                             "cannot affect group shares.")
        for hk in ("gains_sha256", "schema_sha256"):
            if calib["hashes"][hk] != base.provenance[hk]:
                raise SystemExit(f"[refuse] gamma calibration {hk} mismatch -- artifact was "
                                 f"measured against different data statistics")
        # Bind the calibration to the TRAINING VIEW, not just the generation (codex 2026-09-02 P1):
        # the same frozen generation serves many cuts (per-species LoRA), and a gamma set measured
        # on another cut / another manifest must not pass. Calibrations carrying these keys are
        # compared strictly; a derived view REQUIRES them (an old-format artifact cannot vouch).
        _train_ids_sha = hashlib.sha256("\n".join(sorted(names["train"])).encode()).hexdigest()
        _view_now = {"exclusion_sha256": (base.provenance_exclusion or {}).get("sha256") or "none",
                     "train_ids_sha256": _train_ids_sha,
                     "manifest_sha256": hashlib.sha256(
                         (Path(a.ktjd_root) / "manifests" / "clips.jsonl").read_bytes()).hexdigest(),
                     # the gammas were measured with one joint-description table; a pruned view ships
                     # another (codex 2026-09-02 r7 #5). Every artifact so far records this key.
                     "joint_sem_sha256": hashlib.sha256(Path(a.joint_sem).read_bytes()).hexdigest()}
        for hk, now in _view_now.items():
            have = calib["hashes"].get(hk)
            if have is None and getattr(base, "derivation", None) is not None:
                raise SystemExit(f"[refuse] this corpus is a derived training view but the gamma "
                                 f"calibration carries no hashes.{hk}; recalibrate on this view")
            if have is not None and str(have) != now:
                raise SystemExit(f"[refuse] gamma calibration hashes.{hk} mismatch: measured on a different "
                                 f"training view (cut / clip set / manifest) -- recalibrate")
        # The mechanism check certifies the calibration ONLY under the objective it ran with.
        # v_space reweights gradient energy by w(t)^2 and the residual profile is t-dependent,
        # so a check run under a different v_space/sigma_min/t_sampler certifies a DIFFERENT
        # loss (codex 2026-08-26 round 4; supersedes the share-invariance assumption that let
        # runs 7-10 pair v_space=False artifacts with v_space=True training).
        proto = calib.get("protocol", {})
        if "sigma_min" not in proto:
            raise SystemExit("[refuse] gamma calibration predates the objective-protocol record "
                             "(no protocol.sigma_min); its mechanism check did not run this "
                             "run's objective -- recalibrate with V_SPACE/SIGMA_MIN/T_SAMPLER")
        drift = []
        if bool(proto.get("v_space")) != bool(a.v_space):
            drift.append(f"v_space {proto.get('v_space')} != {a.v_space}")
        if abs(float(proto["sigma_min"]) - float(a.sigma_min)) > 1e-9:
            drift.append(f"sigma_min {proto['sigma_min']} != {a.sigma_min}")
        if str(proto.get("t_sampler")) != str(a.t_sampler):
            drift.append(f"t_sampler {proto.get('t_sampler')!r} != {a.t_sampler!r}")
        # gamma_acc: absent in pre-acc artifacts MEANS 0 (their mechanism ran without the term,
        # which is exactly the gamma_acc=0 objective) -- so absence certifies only acc-off runs
        if abs(float(proto.get("gamma_acc", 0.0)) - float(a.gamma_acc)) > 1e-9:
            drift.append(f"gamma_acc {proto.get('gamma_acc', 0.0)} != {a.gamma_acc}")
        # normalization: the energies were measured in the serving normalization; an artifact written before the
        # field existed was measured under per-cell mean/std (the only normalization that existed then)
        if str(proto.get("normalization", "percell")) != str(a.rep_norm):
            drift.append(f"normalization {proto.get('normalization', 'percell')!r} != {a.rep_norm!r}")
        # huber_delta shapes the gradient the mechanism check measured through -- a calibration
        # verified under the knee does not certify the knee-free (MSE) objective, and vice versa
        # (gap found 2026-08-28 when step-2 Huber->MSE landed; the three-key guard predated it)
        if "huber_delta" not in proto:
            raise SystemExit("[refuse] gamma calibration protocol lacks huber_delta -- "
                             "recalibrate with the HUBER env set")
        if abs(float(proto["huber_delta"]) - float(a.huber_delta)) > 1e-9:
            drift.append(f"huber_delta {proto['huber_delta']} != {a.huber_delta}")
        drift += _calib_demo_drift(proto, a.demo_rest, a.demo_frames)     # parser-native bool / int
        # augmentation changes the served distribution the energies were measured on; an artifact without
        # the field was measured without augmentation and certifies only an unaugmented run
        if proto.get("augmentation") != aug_cfg.protocol():
            drift.append(f"augmentation {proto.get('augmentation')!r} != {aug_cfg.protocol()!r}")
        if drift:
            raise SystemExit("[refuse] gamma calibration objective-protocol mismatch: "
                             + "; ".join(drift) + " -- recalibrate under this objective")
        # The grouped loss normalises per BATCH, so the mechanism check certifies the objective only
        # at the batch it ran with (codex 2026-09-03 r1/r2): EVERY view is bound and protocol.batch must
        # be a positive integer equal to --batch. The one announced exception is a --resume of a
        # checkpoint that already trained under exactly this (batch, artifact path+sha) pair -- a run
        # that predates the rule (run12: B16 on the B8 pilot artifact) continues, but never gains a
        # batch-matched certificate.
        # the raw checkpoint object goes to the gate untouched (mmap: tensors are not read); the gate
        # refuses a non-mapping payload / args / ktjd_pins instead of tripping over .get here
        _peek = torch.load(a.resume, map_location="cpu", weights_only=False, mmap=True) if a.resume else None
        _legacy = _calib_batch_gate(proto, int(a.batch), calib_sha, str(a.ktjd_gamma_calib), _peek)
        del _peek
        if _legacy:
            print(_legacy, flush=True)
        ktjd_gammas = {k: float(v) for k, v in calib["gammas"].items()}
        if sorted(ktjd_gammas) != sorted(_GROUP_SPEC_KTJD17):
            raise SystemExit(f"[refuse] calibration gamma groups {sorted(ktjd_gammas)} != "
                             f"group spec {sorted(_GROUP_SPEC_KTJD17)}")
        if not all(np.isfinite(v) and v > 0 for v in ktjd_gammas.values()):
            raise SystemExit(f"[refuse] calibration gammas must be finite and positive: "
                             f"{ktjd_gammas}")
        if a.require_uniform_gammas:
            # the simplified baseline's third removed ingredient IS the calibrated weighting: any artifact whose weights are not
            # all exactly 1.0 restores it, however validly it was measured (codex baseline r3 #3)
            _nonuni = {k: v for k, v in ktjd_gammas.items() if v != 1.0}
            if _nonuni:
                raise SystemExit(f"[refuse] --require_uniform_gammas: {a.ktjd_gamma_calib} weights are not all 1.0 ({_nonuni}); "
                                 f"this arm trains with fixed uniform group weights (measure the artifact with GAMMA_SOLVE=uniform)")
            if str(calib.get("protocol", {}).get("gamma_solve")) != "uniform":
                raise SystemExit(f"[refuse] --require_uniform_gammas: {a.ktjd_gamma_calib} records gamma_solve="
                                 f"{calib.get('protocol', {}).get('gamma_solve')!r}, not 'uniform'")
        elif "gamma_solve" in calib.get("protocol", {}) and str(calib["protocol"]["gamma_solve"]) != "kimodo":
            # the calibrated arms train with the Kimodo-implied shares; an artifact measured with GAMMA_SOLVE=uniform
            # (all-one gammas) would otherwise train a different objective under the same flags (codex 2026-09-15
            # unimate r1 P1: an inherited GAMMA_SOLVE reaches the measurer). Artifacts older than the field were all
            # measured before the uniform option existed.
            raise SystemExit(f"[refuse] {a.ktjd_gamma_calib} records gamma_solve={calib['protocol']['gamma_solve']!r}; "
                             "a calibrated arm needs 'kimodo' (the uniform arm passes --require_uniform_gammas)")
        _drift = calib_arm_model_drift(calib, {"struct_feats": bool(a.struct_feats), "dir_bias": bool(a.dir_bias),
                                               "geo_bias": bool(a.geo_bias), "freeze_zero_joint_sem": bool(a.freeze_zero_joint_sem)},
                                       bool(a.require_uniform_gammas))
        if _drift:
            raise SystemExit(f"[refuse] {a.ktjd_gamma_calib} {_drift}")
        calib_huber = float(calib.get("protocol", {}).get("huber_delta", 0.0))
        if abs(calib_huber - a.huber_delta) > 1e-9:
            raise SystemExit(f"[refuse] gamma calibration measured the objective at huber_delta="
                             f"{calib_huber}, this run sets {a.huber_delta}. The gammas describe "
                             f"per-group gradient shares THROUGH the loss; a different robustness "
                             f"knee is a different loss (codex 2026-08-21 (A)2).")
        if calib.get("protocol", {}).get("mask_policy_version") != KTJD17_MASK_POLICY:
            raise SystemExit(f"[refuse] calibration was measured under mask policy "
                             f"{calib.get('protocol', {}).get('mask_policy_version')!r}, "
                             f"code is {KTJD17_MASK_POLICY!r} -- the energies do not transfer")
        if int(base.provenance["caption_dim"]) != 4096:
            raise SystemExit(f"[refuse] caption embeddings are {base.provenance['caption_dim']}-d "
                             f"but the model's d_text is 4096")
        # Full pin surface (codex round-2): names alone are not selectors, and a path string is
        # not an artifact. Serialize the actual group SLICES, the calibration file's own SHA and
        # the code hash it was measured under, plus every data-payload hash the adapter resolved.
        spec_ser = {k: [[v[0].start, v[0].stop], list(v[1])]
                    for k, v in _GROUP_SPEC_KTJD17.items()}
        ktjd_pins = {**base.provenance,
                     "gamma_calib_version": str(calib.get("version", "?")),
                     "gamma_calib_sha256": calib_sha,
                     "gamma_calib_code_sha256": calib.get("hashes", {}).get("code_sha256"),
                     "gammas": ktjd_gammas,
                     "group_spec": spec_ser,
                     "in_ch": 17,
                     "mask_policy_version": KTJD17_MASK_POLICY,
                     # the exclusion list is part of "what this checkpoint was allowed to see":
                     # resuming or rendering against a different cut is a data change, and the
                     # drift check must catch it like any other payload hash.
                     "exclusion": base.provenance_exclusion,
                     "gate_override": bool(a.ktjd_training_authorized),
                     "augmentation": aug_cfg.protocol()}
    else:
        cond = pickle.load(open(f"{a.data_root}/_cond_normalized_J144.pkl", "rb"))
        types = truebones_types(cond.keys()) if a.corpus == "truebones" else pzh_types(cond.keys())
        names = {k: read_split(a.splits_dir, k) for k in ("train", "val")}
        base = AnyTopDataset(data_root=a.data_root, split="all", num_frames=300, max_joints=144,
                             load_captions=True, caption_emb_cache=a.caption_cache,
                             random_caption=a.random_caption, augment=False,
                             joint_semantics=a.joint_sem, species_whitelist=types,
                             splits_dir=a.splits_dir, texts_json_name=a.texts_json)
        ktjd_gammas, ktjd_pins = None, None
    if a.limit_train_clips > 0:
        # Round-robin over rigs, TWO clips per rig per round: an alphabetical prefix of this corpus
        # is all-human, and a rung that never sees an animal cannot answer "does the mixed corpus
        # fit". Two-at-a-time because InContextPairs drops any target whose only same-rig demo is
        # itself -- a one-clip rig contributes nothing.
        per_rig = {}
        for s_ in base.samples:
            nm = Path(s_["path"]).name.replace(".npy", "")
            if nm in names["train"]:
                per_rig.setdefault(s_["object_type"], []).append(nm)
        rigs = sorted(per_rig)
        for r in rigs:
            per_rig[r].sort()
        keep, i = [], 0
        while len(keep) < a.limit_train_clips and any(len(per_rig[r]) > i for r in rigs):
            for r in rigs:
                if len(per_rig[r]) > i + 1:               # a pair, or nothing
                    keep += per_rig[r][i:i + 2]
                if len(keep) >= a.limit_train_clips:
                    break
            i += 2
        del keep[a.limit_train_clips:]                   # an odd limit overshoots by one
        keep = set(keep)
        n_rigs = len({r for r in rigs if per_rig[r][0] in keep})
        print(f"[overfit-rung] training restricted to {len(keep)} clips over {n_rigs} rigs "
              f"(of {len(names['train'])}): {sorted(keep)[:3]} ...", flush=True)
        names = {**names, "train": keep}
    ds_tr = InContextPairs(base, names["train"], names["train"], object_types=types,
                           demo_frames=a.demo_frames, target_frames=a.target_frames,
                           balance_skeletons=(a.balance == "rig"), seed=a.seed,
                           rig_multiplicity=_rig_multiplicity(a), epoch_draws=a.epoch_draws,
                           emit_fk_fields=(a.gamma_fk > 0 or a.gamma_vel > 0 or a.gamma_lock > 0),
                           emit_graph_v2=(a.struct_feats or a.dir_bias),
                           identity_p=a.identity_p, emit_ref_text=a.ref_text,
                           demo_rest=a.demo_rest, augment=aug_cfg)
    ds_va = InContextPairs(base, names["val"], names["train"], object_types=types,
                           demo_frames=a.demo_frames, target_frames=a.target_frames,
                           balance_skeletons=False, seed=a.seed + 1,
                           emit_fk_fields=(a.gamma_fk > 0 or a.gamma_vel > 0 or a.gamma_lock > 0),
                           emit_graph_v2=(a.struct_feats or a.dir_bias),
                           emit_ref_text=a.ref_text, demo_rest=a.demo_rest)
    print(f"[train] {len(ds_tr)} targets / {len(ds_tr.types)} rigs / {ds_tr.pair_count()} pairs "
          f"| bucket-A val {len(ds_va)} targets / {len(ds_va.types)} rigs", flush=True)

    # Under DDP a DistributedSampler partitions the INDEX SPACE (734 -> ~183/rank -> 22 steps at
    # B8, 704 global draws vs 728 single-GPU). Balanced _pick ignores the indices themselves; the
    # sampler only meters how many batches each rank runs, and set_epoch() is a harmless no-op kept
    # for convention.
    tr_sampler = DistributedSampler(ds_tr, shuffle=True, drop_last=True) if ddp else None
    # EXPLICIT loader generator (codex 01a01b1a round-2): without one, DataLoader draws its
    # per-epoch base seed from the GLOBAL torch RNG, whose state at first iteration depends on
    # how many parameters the model construction consumed -- so runs differing only in optional
    # modules (gamma_fk/struct_feats/dir_bias arms) silently train on DIFFERENT shuffle and
    # worker streams. An own generator keyed to a.seed makes the data stream arm-independent.
    dl_gen = torch.Generator()
    dl_gen.manual_seed(a.seed + 7777)
    # Single-GPU path: dl_gen is RESEEDED to a.seed+7777+ep at every epoch top and workers are
    # NON-persistent, so the loader stream is a pure function of (seed, absolute epoch) -- a
    # resume at ANY restart boundary replays the exact uninterrupted stream (codex 01a01b1a
    # round-3: with persistent workers + a run-scoped generator, a restarted arm replayed the
    # epoch-0 stream and factorial arms stopped being data-paired). DDP keeps persistent workers:
    # its shuffle already comes from DistributedSampler.set_epoch(ep).
    dl_tr = DataLoader(ds_tr, batch_size=a.batch, shuffle=(tr_sampler is None),
                       sampler=tr_sampler, num_workers=a.num_workers,
                       collate_fn=collate, pin_memory=True, generator=dl_gen,
                       # drop_last: an "epoch" is INTENTIONALLY full batches only (734 targets ->
                       # 91x8 = 728 draws single-GPU; 22x8x4 = 704 under 4-rank DDP). Balanced
                       # _pick ignores the index, so no target is systematically excluded.
                       drop_last=True,
                       persistent_workers=(a.num_workers > 0) and (tr_sampler is not None))
    if len(dl_tr) == 0:
        raise SystemExit(f"[refuse] {len(ds_tr)} training targets < batch {a.batch}: with drop_last every "
                         f"epoch would have ZERO steps and checkpoints would be written for a model that "
                         f"never updated (codex 2026-09-02 P0-3)")
    dl_va = DataLoader(ds_va, batch_size=a.batch, shuffle=False, num_workers=0,
                       collate_fn=collate, pin_memory=True)

    ktjd_lut = ktjd_channel_lut(base) if a.corpus == "ktjd17" else None
    rest_lut = None
    if (a.corpus == "ktjd17" and a.artic_min > 0
            and a.gamma_vel <= 0 and a.gamma_lock <= 0):
        raise SystemExit("[refuse] the anti-collapse gate needs an articulation reading, and that "
                         "is only produced by the dynamics term -- set --gamma_vel/--gamma_lock "
                         "(0.01 each is the UMO recipe), or --artic_min 0 to disable the gate "
                         "knowingly. A gate that silently cannot fire is worse than none "
                         "(codex round-S9).")
    if a.demo_rest and (a.corpus != "ktjd17" or a.demo_frames != 1):
        raise SystemExit("[refuse] --demo_rest is ktjd17-only and needs --demo_frames 1")
    if a.corpus == "ktjd17" and a.anchor == "rest":
        rest_lut = {r: torch.from_numpy(base.rest_anchor_frame(r))
                    for r in {s_["object_type"] for s_ in base.samples}}

    # ---------------- model ----------------
    if a.corpus != "ktjd17" and (a.anchor != "none" or a.identity_p > 0):
        raise SystemExit("[refuse] --anchor/--identity_p are ktjd17-only in this integration")
    if a.corpus == "ktjd17" and a.gamma_fk > 0:
        # KTJD gamma7 exists (fk_ktjd_consistency_loss, official-decoder mirror) but has NO
        # fixed_dof override path -- fail loud if a future generation adds such rigs.
        for rig_ in {s_["object_type"] for s_ in base.samples}:
            if not base.static_masks(rig_)["rotation_supervised"].all():
                raise SystemExit(f"[refuse] rig {rig_!r} has fixed_dof joints; the KTJD gamma7 "
                                 f"FK mirror does not implement the fixed_dof override yet")
    if a.two_stage and a.corpus != "ktjd17":
        raise SystemExit("[refuse] --two_stage (variant D) is ktjd17-only in this integration")
    if a.corpus == "ktjd17" and a.anchor == "rest":
        # rest-centering (2026-08-20) makes the rest pose the ORIGIN of the target space, so a
        # rest anchor is now the zero tensor -- passing the pre-centering rest frame would place
        # the flow base at 2x rest. The centering subsumes what this arm was testing.
        raise SystemExit("[refuse] --anchor rest is redundant under rest-centering (the rest "
                         "pose IS the origin now); use --anchor none, or --anchor demo")
    in_ch = 17 if a.corpus == "ktjd17" else 13
    if a.flat_joints:
        # the adapted-baseline arm: everything outside the denoiser -- corpus, cut, split, captions,
        # rest-pose demonstration frame, objective, schedule, budget and the frozen evaluation -- is
        # the control's, which is what the works we are positioned against mean by a fair comparison
        for _bad, _why in (("two_stage", "a different denoiser"), ("struct_feats", "a skeleton input"),
                           ("dir_bias", "a skeleton input"), ("ref_text", "a per-frame text pathway")):
            if getattr(a, _bad):
                raise SystemExit(f"[refuse] --flat_joints with --{_bad}: the flat baseline has no place for {_why}")
        if a.geo_bias:
            raise SystemExit("[refuse] --flat_joints needs --no_geo_bias: there is no joint axis to bias")
        if a.freeze_zero_joint_sem:
            raise SystemExit("[refuse] --flat_joints with --freeze_zero_joint_sem: the flat baseline builds no "
                             "joint-description projection to zero and freeze")
        from src.models.v2.dit_flat import FlatMotionDiT
        model = FlatMotionDiT(in_ch=in_ch, max_joints=a.flat_joints, dim=a.dim, depth=a.depth,
                              n_heads=a.heads, d_text=4096, grad_ckpt=a.grad_ckpt,
                              qk_norm=a.qk_norm).to(dev)
    elif a.two_stage:
        from src.models.v2.dit_motion import TwoStageInContextDiT
        model = TwoStageInContextDiT(in_ch=in_ch, dim=a.dim, depth=a.depth, n_heads=a.heads,
                                     root_dim=a.root_dim, root_depth=4,
                                     d_text=4096, d_joint_sem=4096,
                                     use_struct_feats=a.struct_feats,
                                     use_dir_bias=a.dir_bias, grad_ckpt=a.grad_ckpt,
                                     use_ref_text=a.ref_text, qk_norm=a.qk_norm, use_geo_bias=a.geo_bias).to(dev)
    else:
        model = InContextMotionDiT(in_ch=in_ch, dim=a.dim, depth=a.depth, n_heads=a.heads,
                                   d_text=4096, d_joint_sem=4096,
                                   use_struct_feats=a.struct_feats, use_dir_bias=a.dir_bias, grad_ckpt=a.grad_ckpt,
                                   use_ref_text=a.ref_text, qk_norm=a.qk_norm, use_geo_bias=a.geo_bias).to(dev)
    # raw_model stays the UNCOMPILED module: it is what state_dict()/load_state_dict() use, so
    # checkpoints keep clean keys (a compiled wrapper prefixes everything with `_orig_mod.` and
    # every earlier checkpoint would fail to load).
    raw_model = model
    lora_paths = []
    init_from_sha256, init_from_epoch = None, None
    if a.init_from:
        # weights only, strict: the LoRA backbone must be EXACTLY the trained model (same arch args)
        # hash the bytes actually loaded: best_model.pt is replaced atomically by the still-running
        # backbone, so the path alone does not identify the initialization
        _init_bytes = Path(a.init_from).read_bytes()
        init_from_sha256 = hashlib.sha256(_init_bytes).hexdigest()
        ck0 = torch.load(io.BytesIO(_init_bytes), map_location="cpu", weights_only=False)
        del _init_bytes
        raw_model.load_state_dict(ck0["model"], strict=True)
        init_from_epoch = ck0.get("epoch", None)
        if is_main:
            print(f"[init_from] {a.init_from} (epoch {init_from_epoch}, sha256 {init_from_sha256[:16]}) -> "
                  f"model weights loaded, optimizer/epoch/pins NOT restored", flush=True)
        del ck0
    if a.freeze_zero_joint_sem:
        # no joint semantics: the projection is exactly zero and frozen (excluded from the optimiser below), so h += 0 for every
        # joint; a checkpoint of this arm loads into the unchanged architecture and decodes/evaluates with no special case.
        # Applied AFTER --init_from so an initialisation checkpoint cannot restore a non-zero projection (codex baseline r1 #2).
        with torch.no_grad():
            raw_model.joint_sem.weight.zero_(); raw_model.joint_sem.bias.zero_()
        for p_ in raw_model.joint_sem.parameters():
            p_.requires_grad_(False)
        if is_main:
            print("[model] --freeze_zero_joint_sem: joint-description projection zeroed and frozen (no joint semantics reach the model)", flush=True)
    if a.lora_r > 0:
        from src.models.v2.lora import inject_lora, freeze_non_lora
        lora_paths = inject_lora(raw_model, [g for g in a.lora_targets.split(",") if g],
                                 a.lora_r, a.lora_alpha, a.lora_dropout)
        n_lora, n_all = freeze_non_lora(raw_model)
        if is_main:
            print(f"[lora] r={a.lora_r} alpha={a.lora_alpha} targets={a.lora_targets}: "
                  f"{len(lora_paths)} Linear layers adapted, {n_lora/1e6:.2f}M trainable of "
                  f"{n_all/1e6:.2f}M ({100*n_lora/n_all:.2f}%)", flush=True)
    if a.compile:
        # compile BEFORE DDP wraps it. Order matters and the previous arrangement was a no-op:
        # rebinding `raw_model` after DDP already captured the module leaves DDP running the
        # uncompiled graph, so the flag would have looked enabled while changing nothing.
        model = torch.compile(model, dynamic=True)
        if is_main:
            print("[train] torch.compile(dynamic=True): first batches pay ~76s of compilation",
                  flush=True)
    if ddp:
        # find_unused_parameters: bp_mlp (the shelved blueprint pathway, param indices 17-20)
        # never enters the graph, and DDP's reducer otherwise waits forever for its gradients --
        # the H200 smoke caught exactly this. Costs a small per-step graph walk; excising bp_mlp
        # outright would break strict state_dict loading of every run-1 checkpoint.
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)
    n_par = sum(p.numel() for p in raw_model.parameters())
    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=a.lr, weight_decay=a.wd)

    def model_state_for_ckpt():
        """Plain backbone state_dict: with LoRA the adapters are FOLDED into the weights so every
        existing consumer (renderer, gen-eval, skinning generator) loads it unchanged."""
        if a.lora_r > 0:
            from src.models.v2.lora import merged_state_dict
            return merged_state_dict(raw_model)
        return raw_model.state_dict()

    def lora_extra():
        if a.lora_r <= 0:
            return {}
        from src.models.v2.lora import lora_state_dict
        return {"lora": lora_state_dict(raw_model),
                "lora_cfg": {"r": a.lora_r, "alpha": a.lora_alpha, "dropout": a.lora_dropout,
                             "targets": a.lora_targets, "paths": lora_paths, "init_from": a.init_from,
                             "init_from_sha256": init_from_sha256, "init_from_epoch": init_from_epoch}}
    start_ep, best_val, gstep = 0, float("inf"), 0
    resumed_strikes = 0
    # Initialised BEFORE the resume block, which overwrites it: placing it after would wipe the
    # history the checkpoint just restored, reintroducing the very misreporting this records.
    guard_history = []
    schedule_history = []
    calib_history = []
    # health of the most recent validation, so the periodic epoch/step writers can stamp it too
    last_health, last_vob = True, 1.0
    if a.resume:
        # Resume is a STATISTICAL continuation, not bit-exact: DataLoader worker streams re-derive
        # from a fresh base_seed after restart. What must NOT drift silently is the config -- a
        # resumed run with different dims/lr/data would corrupt the ckpt lineage, so critical
        # fields are hard-checked. Torch/numpy RNG states are restored best-effort on top.
        if Path(a.resume).name == "last_step_snapshot.pt":
            raise SystemExit(
                "[refuse] last_step_snapshot.pt is a MID-EPOCH snapshot for inspection only. "
                "Resume restarts at epoch+1 while keeping the saved gstep, so continuing from a "
                "mid-epoch file silently drops the rest of that epoch AND desynchronizes the lr "
                "schedule from an uninterrupted run. Resume from last_model.pt "
                "(epoch-boundary) instead.")
        ck = torch.load(a.resume, map_location="cpu", weights_only=False)  # our own ckpt; contains numpy RNG state, rejected by 2.6's weights_only default
        old_args = ck.get("args", {})
        if ck.get("healthy", True) is not True and not a.allow_unhealthy_resume:
            raise SystemExit(
                f"[refuse] {a.resume} was written AFTER a blow-up: its val {ck.get('val')} is "
                f"{ck.get('val_over_best', float('nan')):.1f}x the best {ck.get('best_val')}. "
                f"Resuming it continues training a wrecked model -- run7 burned eight hours that "
                f"way. Resume from best_model.pt, or pass --allow_unhealthy_resume to override.")
        if "at_epoch_end" not in ck:
            raise SystemExit(f"[refuse] {a.resume} predates the resume-safety stamp (written by a "
                             f"run before 2026-08-23); it cannot be proven epoch-aligned")
        if not bool(ck["at_epoch_end"]):
            raise SystemExit(f"[refuse] {a.resume} was written mid-epoch (at_epoch_end=False); "
                             f"only epoch-boundary checkpoints are resume-safe")
        crit = ("dim", "depth", "heads", "batch", "lr", "seed", "data_root", "splits_dir",
                "joint_sem", "caption_cache", "texts_json",
                # run-3 trajectory-defining knobs: silently flipping any of these mid-run would
                # change the objective or the data distribution under the same ckpt lineage.
                "corpus", "balance", "random_caption", "demo_frames", "target_frames",
                "t_sampler", "v_space", "p_drop_text", "p_drop_demo", "p_drop_both",
                "bf16", "warmup_steps", "wd", "gamma_fk", "fk_warmup_steps",
                "struct_feats", "dir_bias", "ktjd_root", "anchor", "identity_p",
                "two_stage", "flat_joints", "rig_multiplicity", "epoch_draws", "root_dim", "gamma_vel", "gamma_lock", "gamma_acc", "ref_text",
                "demo_rest", "ktjd_percell_stats", "qk_norm",
                # (codex 2026-08-21 (A)2) the robustness knee and the clip threshold BOTH define
                # the trajectory: resuming with a different one produces later epochs trained
                # under settings nothing in the lineage records.
                "huber_delta", "sigma_min", "grad_clip", "exclude_clips", "ktjd_gamma_calib",
                "lr_scheduler", "eta_min_ratio", "lr_decay_epochs", "grad_ckpt", "epochs", "grad_accum",
                "artic_min", "artic_gate_after", "artic_gate_strikes", "val_every",
                "aug_p", "aug_drop_max_frac", "aug_drop_mode", "aug_rest_deg", "aug_sem_noise", "aug_sem_drop_p",
                "aug_stats_logsd", "aug_stats_shift", "aug_bone_scale", "aug_pool_frac", "aug_add_p", "aug_mode",
                # simplified-baseline knobs (codex baseline r1 #2): both define the arm
                "geo_bias", "freeze_zero_joint_sem")
        core = ("dim", "depth", "heads", "batch", "lr", "seed", "data_root", "splits_dir",
                "joint_sem", "caption_cache", "texts_json")
        missing = [k for k in core if k not in old_args]
        if missing:
            raise SystemExit(f"[resume] ckpt args missing critical keys {missing} -- a legacy or "
                             f"malformed checkpoint must not bypass config validation")
        # New knobs may be absent in older ckpts; a missing key means the ckpt was trained with
        # the PARSER DEFAULT of its era, so it must be compared against ap.get_default -- comparing
        # against the runtime value would wave through exactly the drift this check exists to catch
        # (legacy ckpt + new flag => missing key silently "equals" the new flag).
        if type(old_args.get("batch")) is not int:
            raise SystemExit(f"[resume] ckpt args.batch={old_args.get('batch')!r} is not an int -- a float/bool "
                             f"would compare equal to --batch {a.batch} without being the same batch")
        bad = [k for k in crit if old_args.get(k, ap.get_default(k)) != getattr(a, k)]
        # restored BEFORE the append below -- appending first and restoring afterwards silently
        # discarded the very record the mechanism exists to keep (codex 2026-08-26)
        schedule_history = list(ck.get("schedule_history", []))
        calib_history = list(ck.get("calib_history", []))
        calib_path_changed = False
        SCHEDULE_KEYS = {"lr_scheduler", "eta_min_ratio", "lr_decay_epochs"}
        if bad and a.allow_schedule_restart and set(bad) <= SCHEDULE_KEYS:
            # a deliberate, recorded schedule change -- NOT silent drift: every checkpoint this
            # run writes will carry the change and where it was made
            schedule_history.append({
                "from_ckpt": str(a.resume),
                "old": {k: old_args.get(k, ap.get_default(k)) for k in bad},
                "new": {k: getattr(a, k) for k in bad},
                "at_epoch": int(ck["epoch"]) + 1, "at_gstep": int(ck.get("gstep", 0))})
            print(f"[resume] SCHEDULE RESTART recorded: "
                  f"{ {k: (old_args.get(k, ap.get_default(k)), getattr(a, k)) for k in bad} } "
                  f"from ep{int(ck['epoch'])+1} g{ck.get('gstep', 0)}", flush=True)
        elif bad == ["ktjd_gamma_calib"] and a.allow_calib_reswap:
            # allowed HERE only as a path change; the pins check below verifies the swapped
            # artifact before anything proceeds, and requires exactly this signal -- a swap
            # without a path change is a same-name regeneration, which stays refused
            calib_path_changed = True
        elif bad:
            raise SystemExit(f"[resume] config mismatch on {bad}: "
                             f"ckpt={[old_args.get(k, ap.get_default(k)) for k in bad]} "
                             f"vs now={[getattr(a, k) for k in bad]}"
                             f" -- refusing silent drift (change the ckpt or the flags; a "
                             f"schedule-only change may pass --allow_schedule_restart; an "
                             f"equivalent-calibration swap may pass --allow_calib_reswap)")
        if a.corpus == "ktjd17":
            # provenance pins: path strings alone cannot catch a retargeted symlink or a
            # regenerated artifact under the same name (codex round-S0)
            old_pins = ck.get("ktjd_pins")
            if old_pins is None:
                raise SystemExit("[resume] ckpt carries no ktjd_pins -- refusing to resume a "
                                 "pre-pinning KTJD checkpoint into the pinned lineage")
            if calib_path_changed:
                # BEFORE anything compares pins: the equivalence BASELINE (the old artifact on
                # disk) must itself be proven untampered against the checkpoint's byte pin --
                # a replaced old file could otherwise redefine what "equivalent" means
                # (codex r6/r7: integrity first, then judgement).
                oldc_p = Path(str(old_args.get("ktjd_gamma_calib")))
                if not oldc_p.is_file():
                    raise SystemExit(f"[resume] calib reswap refused: old artifact {oldc_p} is "
                                     f"gone -- behavioural equivalence cannot be verified")
                _old_bytes = oldc_p.read_bytes()
                _old_sha = hashlib.sha256(_old_bytes).hexdigest()
                _pin_sha = str((old_pins or {}).get("gamma_calib_sha256"))
                if _old_sha != _pin_sha:
                    raise SystemExit(f"[resume] calib reswap refused: the OLD artifact on disk "
                                     f"({oldc_p}) does not match the checkpoint's pinned sha "
                                     f"({_old_sha[:12]} != {_pin_sha[:12]}) -- the equivalence "
                                     f"baseline itself has been tampered with")
            drift = sorted(k for k in set(old_pins) | set(ktjd_pins)
                           if old_pins.get(k) != ktjd_pins.get(k))
            CALIB_ID_KEYS = {"gamma_calib_version", "gamma_calib_sha256",
                             "gamma_calib_code_sha256"}
            if calib_path_changed and set(drift) <= CALIB_ID_KEYS:
                # (codex 2026-08-27 round 2) Acceptance requires ALL of:
                #  - an actual PATH change (crit arm set the signal; a same-name regeneration
                #    never reaches here and stays refused),
                #  - the ckpt carries the complete calib-identity pin schema (a legacy ckpt
                #    without them cannot certify what it trained under -- refuse),
                #  - gammas/group_spec bit-identical (they are pins; a difference would be in
                #    `drift` and fail the subset test),
                #  - BEHAVIOURAL loss equivalence between the OLD and NEW artifacts: same
                #    preregistered protocol and, decisively, the same measured mechanism-check
                #    fields -- the recorded behaviour of the loss THROUGH its own code at
                #    measurement time. gammas alone prove grouped weighting, not objective
                #    semantics; this contract is what licenses the code-hash drift.
                missing_pins = sorted(k for k in CALIB_ID_KEYS
                                      if not old_pins.get(k) or not ktjd_pins.get(k))
                if missing_pins:
                    raise SystemExit(f"[resume] calib reswap refused: empty/missing identity "
                                     f"pins {missing_pins} (checked on BOTH lineages); an "
                                     f"uncertified calibration cannot take the hatch")
                import numpy as _np
                oldc = json.loads(_old_bytes.decode())
                def _flat(d, pre=""):
                    # a section that is not a mapping certifies nothing -- surface it as an
                    # empty flat dict, which the caller's nonempty check turns into ":keys"
                    if not isinstance(d, dict):
                        return {}
                    out = {}
                    for k, v in d.items():
                        if isinstance(v, dict):
                            out.update(_flat(v, pre + k + "."))
                        else:
                            out[pre + k] = v
                    return out
                sem_bad = []
                # leaves that are METADATA, not measurements: compared for exact equality after a
                # type check (codex baseline r2 #1 -- float() on a list marked every artifact,
                # including an artifact compared with itself, as malformed and blocked every reswap)
                # keyed by the LEAF name, so every section's metadata is covered (codex baseline r3 #1)
                _EXACT_LEAVES = {"asserted_groups": list, "skipped": bool, "statement": str,
                                 "scope": str, "rationale": str, "note": str}
                def _num_eq(sect, k, ov, nv):
                    want = _EXACT_LEAVES.get(k.rsplit(".", 1)[-1])
                    if want is not None:
                        if not (isinstance(ov, want) and isinstance(nv, want)):
                            sem_bad.append(f"{sect}.{k}:malformed")
                        elif ov != nv:
                            sem_bad.append(f"{sect}.{k}")
                        return
                    # every OTHER leaf of these sections is a measurement: text there is a malformed
                    # artifact, not a number to compare (codex r4/r5)
                    if isinstance(ov, str) or isinstance(nv, str):
                        sem_bad.append(f"{sect}.{k}:malformed")
                        return
                    # booleans float() silently (True == 1.0) but are not measurements
                    if isinstance(ov, bool) or isinstance(nv, bool):
                        sem_bad.append(f"{sect}.{k}:malformed"); return
                    # fail CLOSED on malformed or non-finite leaves: an artifact whose measured
                    # field cannot be read as a finite number certifies nothing (codex r3)
                    try:
                        ov, nv = float(ov), float(nv)
                    except (TypeError, ValueError):
                        sem_bad.append(f"{sect}.{k}:malformed"); return
                    if not (_np.isfinite(ov) and _np.isfinite(nv)):
                        sem_bad.append(f"{sect}.{k}:nonfinite"); return
                    if not _np.isclose(ov, nv, rtol=1e-3, atol=1e-9):
                        sem_bad.append(f"{sect}.{k}")
                # numeric sections AND mechanism_check: leaf sets must match EXACTLY -- a leaf
                # deleted from either side is a schema change, not a free pass
                # ALL of the artifact's evidence, not a subset: the mechanism check, the solve/skip record and the
                # acceleration diagnostic each certify part of the objective this checkpoint trained under, and a section
                # left out of the comparison is a section a re-measurement may silently change (codex baseline r3 #1)
                # residual_saturation_at_init is a top-level measurement, not a section (codex baseline r4 #3)
                _num_eq("root", "residual_saturation_at_init", oldc.get("residual_saturation_at_init"),
                        calib.get("residual_saturation_at_init"))
                for section in ("gammas", "energies", "counts", "target_family_shares",
                                "mechanism_check", "solve_consistency_check", "acc_diagnostic"):
                    ov_, nv_ = oldc.get(section), calib.get(section)
                    if ov_ is None and nv_ is None:
                        continue          # acc_diagnostic is null when the arm trains with gamma_acc=0 (codex baseline r5 #1)
                    if (ov_ is None) != (nv_ is None):
                        sem_bad.append(f"{section}:present_on_one_side")
                        continue
                    o, n = _flat(ov_), _flat(nv_)
                    if not o or not n or set(o) != set(n):
                        sem_bad.append(f"{section}:keys")
                        continue
                    for k in o:
                        _num_eq(section, k, o[k], n[k])
                for key in ("v_space", "sigma_min", "t_sampler", "huber_delta", "batch", "seed",
                            "mask_policy_version", "weighting", "cohort"):
                    ov = oldc.get("protocol", {}).get(key)
                    nv = calib.get("protocol", {}).get(key)
                    if ov is None or nv is None:
                        # both-missing must NOT compare equal: a required protocol key absent on
                        # either side means the artifact cannot certify its objective
                        sem_bad.append(f"protocol.{key}:missing")
                    elif ov != nv:
                        sem_bad.append(f"protocol.{key}")
                if sem_bad:
                    raise SystemExit(f"[resume] calib reswap refused: artifacts are NOT "
                                     f"behaviourally equivalent on {sorted(set(sem_bad))} -- "
                                     f"the code-hash drift is not license-able")
                calib_history.append({
                    "from_ckpt": str(a.resume),
                    "old": {k: old_pins.get(k) for k in sorted(CALIB_ID_KEYS)},
                    "new": {k: ktjd_pins.get(k) for k in sorted(CALIB_ID_KEYS)},
                    "old_path": str(old_args.get("ktjd_gamma_calib")),
                    "new_path": str(a.ktjd_gamma_calib),
                    "behavioural_equivalence": "gammas/energies/counts/shares/protocol/"
                                               "mechanism_check within rtol 1e-3",
                    "at_epoch": int(ck["epoch"]) + 1, "at_gstep": int(ck.get("gstep", 0))})
                print(f"[resume] CALIB RESWAP recorded (behavioural equivalence verified): "
                      f"{old_args.get('ktjd_gamma_calib')} -> {a.ktjd_gamma_calib} "
                      f"from ep{int(ck['epoch'])+1} g{ck.get('gstep', 0)}", flush=True)
            elif drift:
                raise SystemExit(
                    f"[resume] KTJD provenance drift on {drift}: "
                    f"ckpt={ {k: old_pins.get(k) for k in drift} } vs "
                    f"now={ {k: ktjd_pins.get(k) for k in drift} }")
        # GUARD LINEAGE. grad_spike_reject is deliberately NOT in `crit` (it is a numerical
        # stability guard like the non-finite skip, not a definition of the objective), which is
        # exactly why the checkpoint must carry its history: otherwise a final checkpoint records
        # only the CURRENT threshold and silently claims it applied to epochs that ran without it.
        guard_hist = list(ck.get("guard_history", []))
        old_guard = float(old_args.get("grad_spike_reject", ap.get_default("grad_spike_reject")))
        if old_guard != a.grad_spike_reject:
            # Only 0 -> positive is allowed: enabling a guard mid-run is a recoverable, reportable
            # event; changing or removing one silently rewrites what the later epochs mean.
            if old_guard != 0.0:
                raise SystemExit(
                    f"[resume] grad_spike_reject changes {old_guard} -> {a.grad_spike_reject}; "
                    f"only 0 -> positive (first activation) is permitted, so that a lineage never "
                    f"has a guard weakened or retuned underneath it")
            if a.grad_spike_reject <= 0:
                raise SystemExit(f"[resume] refusing to DISABLE grad_spike_reject "
                                 f"({old_guard} -> {a.grad_spike_reject})")
            guard_hist.append({"from_ckpt": str(a.resume), "old": old_guard,
                               "new": float(a.grad_spike_reject),
                               "activated_at_epoch": int(ck["epoch"]) + 1,
                               "activated_at_gstep": int(ck.get("gstep", 0))})
            print(f"[resume] grad-spike guard ACTIVATED at ep{int(ck['epoch'])+1} "
                  f"g{ck.get('gstep', 0)} (threshold {a.grad_spike_reject}); epochs 0-"
                  f"{ck['epoch']} ran WITHOUT it -- recorded in guard_history", flush=True)
        guard_history = guard_hist
        # Carry the health forward. Re-seeding to True would let a periodic checkpoint written
        # BEFORE the first validation of a resumed run -- or any checkpoint from an
        # --allow_unhealthy_resume continuation -- claim health it never had (codex 2026-08-24).
        last_health = bool(ck.get("healthy", True))
        last_vob = float(ck.get("val_over_best", 1.0))
        raw_model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"])
        start_ep, best_val = ck["epoch"] + 1, ck.get("best_val", float("inf"))
        gstep = ck.get("gstep", 0)
        # the strike counter is RUN state: resetting it on resume let a collapsing run dodge the
        # gate indefinitely by restarting (codex round-S9)
        resumed_strikes = int(ck.get("artic_strikes", 0))
        rng = ck.get("rng")
        # Only rank 0 restores the saved RNG: the ckpt carries ONE stream, and loading it on every
        # rank would make all ranks draw IDENTICAL noise/t after a DDP resume. Non-main ranks keep
        # their startup seeding (a.seed + rank) -- deterministic and rank-distinct.
        if rng is not None and (not ddp or is_main):
            torch.set_rng_state(rng["cpu"]); torch.cuda.set_rng_state_all(rng["cuda"])
            np.random.set_state(rng["np"])
        elif ddp and not is_main:
            rng = None
        if is_main:
            print(f"[resume] {a.resume} -> epoch {start_ep}, best_val {best_val:.5f} "
                  f"(rng {'restored' if rng else 'fresh'})", flush=True)
    if is_main:
        (out / "args.json").write_text(json.dumps(
            {**vars(a), "params": n_par, "trainable_params": sum(p.numel() for p in trainable),
             **({"lora_paths": lora_paths} if a.lora_r > 0 else {}),
             **({"init_from_sha256": init_from_sha256, "init_from_epoch": init_from_epoch} if a.init_from else {}),
             # EFFECTIVE objective config (codex round-S0: recording KIMODO_GAMMAS on a KTJD run
             # misdocumented the lineage): gammas actually applied + spec + mask policy + pins.
             "gammas": ktjd_gammas if a.corpus == "ktjd17" else KIMODO_GAMMAS,
             **({"evidence_class": "TRANSDUCTIVE_ARCHITECTURE_TEST",
                 "evidence_note": "per-cell normalization statistics cover ALL accepted clips of "
                                  "every rig, held-out rigs and val clips included. Never report "
                                  "results from this run as inductive or zero-shot.",
                 "group_spec": sorted(_GROUP_SPEC_KTJD17),
                 "mask_policy_version": KTJD17_MASK_POLICY,
                 "ktjd_pins": ktjd_pins} if a.corpus == "ktjd17" else {}),
             **({"world_size": int(os.environ["WORLD_SIZE"])} if ddp else {})},
            indent=2))
        _accum_note = f" (x{a.grad_accum} micro-batches of B{a.batch} per step)" if a.grad_accum > 1 else ""
        if ddp:
            print(f"[train] {n_par/1e6:.2f}M params | T={a.demo_frames}+{a.target_frames} | "
                  f"B{a.batch}x{os.environ['WORLD_SIZE']} lr{a.lr} | "
                  f"{len(dl_tr) // a.grad_accum} steps/epoch/rank{_accum_note}", flush=True)
        else:
            print(f"[train] {n_par/1e6:.2f}M params | T={a.demo_frames}+{a.target_frames} | "
                  f"B{a.batch} lr{a.lr} | {len(dl_tr) // a.grad_accum} steps/epoch{_accum_note}", flush=True)

    # ---------------- loop ----------------
    def apply_cfg_drops(b):
        """Per-sample conditioning dropout (F5-style: independent text/demo + joint). Dropped text
        becomes a zero vector (text_mlp's bias path acts as the learned null); a dropped demo is
        zeroed AND masked out of frame_valid so attention cannot see it. valid is rebuilt."""
        if not (a.p_drop_text or a.p_drop_demo or a.p_drop_both):
            return b
        B = b["x"].shape[0]
        db = torch.rand(B, device=dev) < a.p_drop_both
        dt = (torch.rand(B, device=dev) < a.p_drop_text) | db
        dm = (torch.rand(B, device=dev) < a.p_drop_demo) | db
        if dt.any() and "text" in b:
            b["text"] = b["text"].clone(); b["text"][dt] = 0
        if dm.any():
            b["x"] = b["x"].clone(); b["frame_valid"] = b["frame_valid"].clone()
            b["x"][dm, :a.demo_frames] = 0
            b["frame_valid"][dm, :a.demo_frames] = False
            if "demo_text" in b:
                # the demo's caption describes frames the model can no longer see -- drop it with
                # the demo, or the "demo-dropped" CFG branch still gets told what the demo did.
                b["demo_text"] = b["demo_text"].clone(); b["demo_text"][dm] = 0
            b["valid"] = b["frame_valid"][:, :, None] & b["joint_valid"][:, None, :]
        return b

    abort_msg = [None]

    def run_validation(ep, at_epoch_end=True):
        """rank-0 val + probe + best/last ckpt. Callable from the epoch boundary or mid-epoch
        (step cadence); barrier counts are 2/2 on both sides either way."""
        nonlocal best_val, artic_strikes, last_health, last_vob
        if ddp:
            dist.barrier()
        if not is_main:
            if ddp:
                dist.barrier()
            _collective_abort()
            return
        model.eval(); reset_val_stream(ds_va)
        if len(ds_va) == 0:
            # ALL-TRAIN fine-tune (user 2026-09-03: every clip of the rig in train): there is nothing
            # to validate, so no probe, no artic gate and no best_model.pt -- last_model.pt is the
            # deliverable. The barrier protocol below is kept intact for DDP peers.
            print(f"  [val] skipped -- no val targets (all-train run) | g{gstep}", flush=True)
            state = {"model": model_state_for_ckpt(), **lora_extra(), "opt": opt.state_dict(), "epoch": ep,
                     "gstep": gstep, "val": None, "best_val": best_val, "args": vars(a),
                     "rng": {"cpu": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all(),
                             "np": np.random.get_state()},
                     "ktjd_pins": ktjd_pins, "artic": None, "artic_windows": 0, "artic_strikes": artic_strikes,
                     "at_epoch_end": bool(at_epoch_end), "guard_history": guard_history,
                     "schedule_history": schedule_history, "calib_history": calib_history,
                     "val_over_best": 1.0, "healthy": True, "no_val": True}
            atomic_save(state, out / ("last_model.pt" if at_epoch_end else "last_step_snapshot.pt"))
            model.train()
            if ddp:
                dist.barrier()
            _collective_abort()
            return
        # Fixed captions for val: rotation would make the val text a moving target across passes.
        # ds_va shares `base`; the toggle is safe because the val loader is num_workers=0.
        rc_saved = base.random_caption
        base.random_caption = False
        vtot, vn, vfk, vfkd, vsr, vsrn, vacc = 0.0, 0, 0.0, 0.0, 0.0, 0, 0.0
        with fixed_torch_rng(10_000 + a.seed):
            with torch.no_grad():
                for vb in dl_va:
                    vb = to_dev(vb, dev)
                    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=a.bf16):
                        # Same objective as training (t_sampler/v_space) so val tracks what is
                        # optimized; fixed_torch_rng keeps the draws identical across passes.
                        # gamma_fk enters at FULL weight (no warmup ramp): val measures the
                        # objective being approached, and a step-dependent val is not comparable
                        # across the run.
                        if ktjd_lut is not None:
                            vx, vkt = ktjd_prep(vb, ktjd_lut, ktjd_gammas)
                            if a.anchor != "none":
                                vkt["anchor"] = ktjd_anchor(vb, vx, a.anchor, rest_lut,
                                                            a.demo_frames)
                        else:
                            vx, vkt = vb["x"], dict(gammas=KIMODO_GAMMAS)
                        vextra = {}
                        if a.gamma_fk > 0:
                            vextra.update(gamma_fk=a.gamma_fk, fk_pack=fk_pack_of(vb))
                        if a.gamma_acc > 0:
                            vextra.update(gamma_acc=a.gamma_acc)
                        if a.gamma_vel > 0 or a.gamma_lock > 0:
                            vextra.update(gamma_vel=a.gamma_vel, gamma_lock=a.gamma_lock,
                                          fk_pack=fk_pack_of(vb))
                        if vextra:
                            vloss, vp = cfm_loss(model, vx, is_target=vb["is_target"],
                                                 valid=vb["valid"],
                                                 t_sampler=a.t_sampler, v_space=a.v_space, sigma_min=a.sigma_min, huber_delta=a.huber_delta,
                                                 return_parts=True, **vextra, **vkt,
                                                 **cond_of(vb))
                            vfk += vp.get("fk_consist", 0.0); vfkd += vp.get("fk_dist", 0.0)
                            vacc += vp.get("acc_match", 0.0)
                            # speed_ratio is the ONLINE frozen-pose monitor (1.0 = GT speed)
                            # sum-of-ratios / count-of-windows, NEVER mean-of-batch-means:
                            # static-GT batches contribute no ratio at all (codex round-S3)
                            vsr += vp.get("speed_ratio", 0.0) * vp.get("speed_ratio_n", 0)
                            vsrn += vp.get("speed_ratio_n", 0)
                        else:
                            vloss = cfm_loss(model, vx, is_target=vb["is_target"],
                                             valid=vb["valid"],
                                             t_sampler=a.t_sampler, v_space=a.v_space, sigma_min=a.sigma_min, huber_delta=a.huber_delta,
                                             **vkt, **cond_of(vb))
                    vtot += float(vloss); vn += 1
                    if a.val_max_batches and vn >= a.val_max_batches:
                        break
            v = vtot / max(vn, 1)
            reset_val_stream(ds_va)
            probe_b = to_dev(next(iter(dl_va)), dev)
            p5 = connectivity_probe(raw_model, probe_b, a.demo_frames,
                                    obj=dict(t_sampler=a.t_sampler, v_space=a.v_space,
                                             sigma_min=a.sigma_min, huber_delta=a.huber_delta),
                                    ktjd_lut=ktjd_lut,
                                    ktjd_gammas=ktjd_gammas,
                                    probe_joint_sem=not a.flat_joints)
        base.random_caption = rc_saved
        fk_str = (f" | fk={vfk / max(vn, 1):.4f} fkdist={vfkd / max(vn, 1):.3f}bl"
                  if a.gamma_fk > 0 else "")
        # independent of fk_str: acc can be active with the FK term off
        if a.gamma_acc > 0:
            fk_str += f" acc={vacc / max(vn, 1):.4f}"
        if a.gamma_vel > 0 or a.gamma_lock > 0:
            # THE gate the ep250 collapse had no online equivalent of: 1.0 = GT speed, ~0 = frozen
            fk_str += (f" | artic={vsr / vsrn:.3f}xGT({vsrn}w)" if vsrn else " | artic=n/a")
        print(f"  [val] flow_loss={v:.5f}{fk_str} | g{gstep} | P5 demo={p5['demo']:.2e} "
              f"text={p5['text']:.2e} sem={p5['joint_sem']:.2e}", flush=True)
        # ANTI-COLLAPSE GATE (codex round-S8). The 2-epoch smoke showed articulation falling
        # 0.496 -> 0.271 while flow loss IMPROVED -- the exact signature of the collapse this
        # project has hit twice. Printing it was not enough: a collapsed checkpoint could still be
        # crowned "best" on flow loss alone, and the run would burn 500 epochs producing a frozen
        # animal. So the ratio now gates BOTH.
        artic_now = (vsr / vsrn) if vsrn else float("nan")
        # BEST-ELIGIBILITY IS GATED FROM THE FIRST VALIDATION (codex round-S10 blocker): the
        # epoch-30 grace period exists so an untrained model is not aborted, but it must not also
        # let epochs 5-30 crown a collapsed checkpoint as "best". Measured on this very config,
        # articulation sits at 0.30-0.49 in the first epochs -- squarely in the range that would
        # have been crowned. The grace period applies to STRIKE COUNTING only.
        artic_ok = True
        if a.artic_min > 0 and vsrn:
            artic_ok = artic_now >= a.artic_min
        # A FATAL articulation verdict must also mark the state unhealthy. Otherwise the gate stops
        # this process, the checkpoint still says "healthy", and the watchdog dutifully relaunches
        # exactly the collapsed model the gate refused -- turning a deliberate hard stop into an
        # infinite restart loop (codex 2026-08-24).
        # Compute the POST-update strike count exactly as the block below does, then ask whether
        # that aborts. Using the stale pre-reset count condemned a recovering validation: after an
        # --allow_unhealthy_resume from a fatal checkpoint, strikes sit at the threshold, and an
        # articulation-OK validation would still have been stamped unhealthy even though it resets
        # the count and does not abort (codex 2026-08-24).
        _next_strikes = 0 if artic_ok else artic_strikes + 1
        artic_fatal = bool(a.artic_min > 0 and ep >= a.artic_gate_after
                           and (not vsrn or _next_strikes >= a.artic_gate_strikes))
        if a.artic_min > 0 and ep >= a.artic_gate_after and not vsrn:
            raise SystemExit("[FATAL] the anti-collapse gate is enabled but this validation "
                             "produced NO articulation window -- the gate cannot judge, so it "
                             "refuses to wave the run through (codex round-S9).")
        if a.artic_min > 0 and vsrn and ep >= a.artic_gate_after:
            artic_strikes = 0 if artic_ok else artic_strikes + 1
            if not artic_ok:
                print(f"  [WARN] articulation {artic_now:.3f}xGT below floor {a.artic_min} "
                      f"(strike {artic_strikes}/{a.artic_gate_strikes})", flush=True)
            if artic_strikes >= a.artic_gate_strikes:
                # NOT raised here: rank 0 exiting before the closing barrier strands its peers
                # (codex round-S9). The decision is broadcast below so every rank aborts together.
                abort_msg[0] = (f"[FATAL] articulation stayed below {a.artic_min}xGT for "
                                f"{artic_strikes} consecutive validations (last {artic_now:.3f}) "
                                f"-- the body has collapsed to a near-static pose. Aborting "
                                f"instead of training a frozen model to {a.epochs} epochs.")
        rng_state = {"cpu": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all(),
                     "np": np.random.get_state()}
        state = {"model": model_state_for_ckpt(), **lora_extra(), "opt": opt.state_dict(), "epoch": ep,
                 "gstep": gstep, "val": v,
                 # only an ELIGIBLE checkpoint may lower the recorded best: otherwise a resume
                 # inherits a best score no saved file corresponds to, and every later legitimate
                 # best is suppressed by it (codex round-S9)
                 "best_val": min(best_val, v) if artic_ok else best_val, "args": vars(a),
                 "rng": rng_state, "ktjd_pins": ktjd_pins,
                 "artic": artic_now, "artic_windows": vsrn, "artic_strikes": artic_strikes,
                 # resume-safety marker: only epoch-boundary states may be continued
                 "at_epoch_end": bool(at_epoch_end),
                 # so a checkpoint never claims a guard applied to epochs that ran without it
                 "guard_history": guard_history, "schedule_history": schedule_history,
                 "calib_history": calib_history,
                 # HEALTH MARKER. A checkpoint written after a blow-up is still a valid file and a
                 # resume will happily continue from it -- run7 burned eight hours doing exactly
                 # that. Record the ratio against the best score so any consumer can tell a
                 # recoverable state from a wrecked one without re-running validation.
                 "val_over_best": (float(v) / best_val) if best_val not in (0.0, float("inf"))
                                  else 1.0,
                 # `nan > x` is False, so a naive comparison stamps a NaN validation HEALTHY --
                 # the worst possible state marked safe to resume (codex 2026-08-24). Require
                 # finiteness explicitly.
                 "healthy": bool(np.isfinite(v) and not artic_fatal
                                 and not (best_val not in (0.0, float("inf"))
                                          and v > HEALTH_RATIO * best_val))}
        # RESUME SAFETY (codex 2026-08-23 blocker 3): resume restarts at `epoch+1` while keeping
        # the saved gstep, so a MID-EPOCH checkpoint silently drops the remainder of its epoch and
        # carries a gstep that no longer matches the epoch counter -- which also desynchronizes the
        # lr schedule from an uninterrupted run. Only epoch-boundary checkpoints may be written to
        # last_model.pt; step-level validations still record a snapshot for inspection.
        last_health, last_vob = state["healthy"], state["val_over_best"]
        if at_epoch_end:
            atomic_save(state, out / "last_model.pt")
        else:
            atomic_save(state, out / "last_step_snapshot.pt")
        # a checkpoint that fails the articulation floor is never "best", however good its loss
        if v < best_val and artic_ok:
            best_val = v
            atomic_save(state, out / "best_model.pt")
            print(f"  [ckpt] new best val_flow={v:.5f}", flush=True)
        model.train()
        if ddp:
            dist.barrier()
        _collective_abort()

    def _collective_abort():
        """Every rank aborts or none does: an abort raised on rank 0 alone deadlocks the others."""
        flag = torch.tensor([1.0 if abort_msg[0] else 0.0], device=dev)
        if ddp:
            dist.all_reduce(flag, op=dist.ReduceOp.MAX)
        if flag.item() > 0:
            raise SystemExit(abort_msg[0] or "[FATAL] peer rank aborted on the articulation gate")

    nonfinite = 0
    spike_skips = 0
    artic_strikes = resumed_strikes
    # Cosine horizon in OPTIMIZER steps. len(dl_tr) is this rank's step count per epoch (the
    # DistributedSampler already partitioned the index), which is exactly what gstep counts, so
    # the two agree without a world-size factor. Fixed by --epochs: changing the epoch budget
    # mid-run would silently reshape the decay, which is why --epochs is resume-critical.
    _decay_ep = a.lr_decay_epochs if a.lr_decay_epochs > 0 else a.epochs
    if len(dl_tr) < a.grad_accum:
        raise SystemExit(f"[refuse] --grad_accum {a.grad_accum} exceeds the {len(dl_tr)} micro-batches of an epoch")
    total_opt_steps = max(1, _decay_ep * (len(dl_tr) // a.grad_accum))   # optimizer steps, not micro-batches
    if is_main:
        print(f"[train] lr schedule: {a.lr_scheduler} | warmup {a.warmup_steps} -> "
              f"{total_opt_steps} decay steps ({_decay_ep} ep of {a.epochs})"
              + (f" -> floor {a.lr * a.eta_min_ratio:.2e}"
                 if a.lr_scheduler == "half_cosine" else " (FLAT -- runs 1-5 all diverged here)"),
              flush=True)
    for ep in range(start_ep, a.epochs):
        if tr_sampler is not None:
            tr_sampler.set_epoch(ep)
        else:
            # loader stream = f(seed, absolute ep): restart-boundary-invariant (see dl_gen note)
            dl_gen.manual_seed(a.seed + 7777 + ep)
        model.train(); t0, tot, n = time.time(), 0.0, 0
        g_sum, g_max = 0.0, 0.0
        acc_i, acc_loss, window_bad = 0, 0.0, False   # gradient accumulation: micro-batches of the current step window, their mean loss, rejected?
        for b in dl_tr:
            b = to_dev(b, dev)
            # The anchor is BASE GEOMETRY, not a condition: UMO's source-centered base is
            # identical across every CFG branch and is NEVER dropped. It must therefore be built
            # from the UN-DROPPED batch. Built after apply_cfg_drops, a demo-dropped sample has
            # frame_valid[:demo]=False -> d_real=0 -> a ZERO anchor, so ~19% of arm-B training
            # saw base=N(0,I) while sampling always adds the demo anchor to every branch --
            # a train/inference base-distribution mismatch (codex round-2 blocker 1).
            anchor_kw = {}
            if ktjd_lut is not None and a.anchor != "none":
                anchor_kw["anchor"] = ktjd_anchor(b, b["x"][..., :17], a.anchor, rest_lut,
                                                  a.demo_frames)
            b = apply_cfg_drops(b)
            # lr schedule (manual: resumes correctly from the saved gstep, no scheduler state).
            # Warmup then half-cosine, both keyed to the OPTIMIZER step -- the CodeFlow recipe
            # notes that keying it to anything else makes a resume diverge from an uninterrupted
            # run. total_steps is fixed by --epochs, so extending a run's epoch count changes the
            # schedule and must not be done mid-run.
            if a.warmup_steps > 0 and gstep < a.warmup_steps:
                lr_now = a.lr * (gstep + 1) / a.warmup_steps
            elif a.lr_scheduler == "half_cosine":
                _prog = ((gstep - a.warmup_steps)
                         / max(1, total_opt_steps - a.warmup_steps))
                _prog = min(1.0, max(0.0, _prog))
                _cos = 0.5 * (1.0 + math.cos(math.pi * _prog))
                lr_now = a.lr * (a.eta_min_ratio + (1.0 - a.eta_min_ratio) * _cos)
            else:
                lr_now = a.lr
            # OUTSIDE the branch. It used to sit inside the `none` arm, so selecting half_cosine
            # computed a decayed lr and then never wrote it -- AdamW kept its constructor lr and
            # the run behaved exactly like the flat-lr ones (codex 2026-08-23 blocker 1).
            for pg in opt.param_groups:
                pg["lr"] = lr_now
            fk_kw = {}
            if a.gamma_vel > 0 or a.gamma_lock > 0:
                fk_kw = dict(gamma_vel=a.gamma_vel, gamma_lock=a.gamma_lock,
                             fk_pack=fk_pack_of(b))
            if a.gamma_acc > 0:
                # independent of the dynamics pair: pure normalized-channel term, no fk_pack
                fk_kw["gamma_acc"] = a.gamma_acc
            if a.gamma_fk > 0:
                # hy273 recipe: linear warmup of the consistency weight -- full-strength FK
                # penalties on the garbage x1_pred of the first steps destabilize more than they
                # teach. gstep-based, so a resume continues the ramp exactly where it stopped.
                ramp = min(1.0, (gstep + 1) / max(a.fk_warmup_steps, 1))
                fk_kw.update(gamma_fk=a.gamma_fk * ramp, fk_pack=fk_pack_of(b))
            if ktjd_lut is not None:
                x_in, kt_kw = ktjd_prep(b, ktjd_lut, ktjd_gammas)
                kt_kw.update(anchor_kw)          # built pre-drop, see the note above
            else:
                x_in, kt_kw = b["x"], dict(gammas=KIMODO_GAMMAS)
            # GRADIENT ACCUMULATION (2026-09-14, held-out study: 2 ranks x B16 x accum 2 must weight every cell as the
            # baseline's 4 ranks x B16 do). cfm_loss normalises each micro-batch on its own and DDP averages over
            # ranks, so summing loss / accum over the micro-batches of a step gives the same per-cell weighting as
            # accum x more ranks. A step is a FIXED window of a.grad_accum micro-batches: a window in which any
            # micro-batch produced a non-finite loss is rejected as a whole, its remaining micro-batches are consumed
            # without compute, and the attempted-step clock gstep advances once at the window boundary -- the same
            # budget of attempted steps per epoch as accum x more ranks (codex accum r1 P2-2). Every decision at the
            # window end (non-finite grad, spike, clip, step, resync, logging) is taken once per optimizer step.
            last_micro = acc_i == a.grad_accum - 1
            if window_bad:                                   # consuming the rest of a rejected window: no compute
                acc_i += 1
                if last_micro:
                    acc_i, acc_loss, window_bad = 0, 0.0, False
                    gstep += 1
                continue
            if acc_i == 0:
                opt.zero_grad(set_to_none=True)
            # DDP arms its reducer in the FORWARD, so no_sync has to cover forward and backward together (r1 P2-1);
            # only the window's last micro-batch synchronises.
            with (model.no_sync() if (ddp and not last_micro) else contextlib.nullcontext()):
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=a.bf16):
                    loss = cfm_loss(model, x_in, is_target=b["is_target"], valid=b["valid"],
                                    t_sampler=a.t_sampler, v_space=a.v_space, sigma_min=a.sigma_min, huber_delta=a.huber_delta,
                                    **kt_kw, **fk_kw, **cond_of(b))
                bad = (~torch.isfinite(loss.detach())).float()
                if ddp:
                    # The skip decision must be COLLECTIVE: one rank skipping backward while its peers
                    # run it deadlocks the DDP reducer at the next bucket sync. MAX-reduce the flag so
                    # every rank skips together whenever any rank saw a non-finite loss.
                    dist.all_reduce(bad, op=dist.ReduceOp.MAX)
                if bad.item() > 0:
                    # STABILITY GUARD: skip the step loudly; a silent NaN would poison the weights and
                    # every ckpt after it. Abort the run if it becomes a pattern.
                    nonfinite += 1
                    if ddp and last_micro and a.grad_accum > 1:
                        # the synchronised forward armed the reducer: give it the backward it expects, then discard
                        loss.backward()
                    opt.zero_grad(set_to_none=True)
                    window_bad = True                         # the micro-batches already accumulated go with it
                    if is_main:
                        print(f"[WARN] non-finite loss at g{gstep} ep{ep} (#{nonfinite}) -- step "
                              f"skipped on ALL ranks", flush=True)
                    if nonfinite >= 50:
                        raise SystemExit("[FATAL] 50 non-finite losses -- training unstable, aborting")
                    acc_i += 1
                    if last_micro:
                        acc_i, acc_loss, window_bad = 0, 0.0, False
                        gstep += 1
                    continue
                (loss / a.grad_accum if a.grad_accum > 1 else loss).backward()
            acc_loss += float(loss.detach()) / a.grad_accum
            if not last_micro:
                acc_i += 1
                continue
            acc_i, step_loss, acc_loss = 0, acc_loss, 0.0
            gn = torch.nn.utils.clip_grad_norm_(model.parameters(), a.grad_clip)
            # Gradient overflow can be non-finite even when the loss was finite (bf16 backward).
            # The decision must again be COLLECTIVE under DDP.
            gbad = (~torch.isfinite(gn)).float().to(dev)
            if ddp:
                dist.all_reduce(gbad, op=dist.ReduceOp.MAX)
            # The spike decision must be COLLECTIVE for the same reason the non-finite one is: one
            # rank skipping while its peers step deadlocks the reducer at the next bucket sync.
            # MAX over ranks, so any rank seeing a spike makes every rank skip.
            gnf_pre_t = torch.tensor([float(gn) if torch.isfinite(gn) else 0.0], device=dev)
            if ddp:
                dist.all_reduce(gnf_pre_t, op=dist.ReduceOp.MAX)
            gnf_pre = float(gnf_pre_t.item())
            if gbad.item() > 0:
                nonfinite += 1
                opt.zero_grad(set_to_none=True)
                if is_main:
                    print(f"[WARN] non-finite GRAD at g{gstep} ep{ep} (#{nonfinite}) -- step "
                          f"skipped on ALL ranks", flush=True)
                if nonfinite >= 50:
                    raise SystemExit("[FATAL] 50 non-finite events -- training unstable, aborting")
                gstep += 1
                continue
            # GRADIENT-SPIKE REJECTION (2026-08-23). Six runs died the same way: one step with a
            # gradient two to three orders of magnitude above normal, after which the model never
            # recovered. run6 g31800 grad=1.7 -> g32000 grad=2775 (loss 0.37 -> 43.5); run4 hit
            # grad=1075 at almost the same gstep (30800); run5 grad=8126. Clipping does NOT stop
            # this: it rescales the magnitude but keeps the direction, so a clipped garbage
            # gradient is still a full-size step in a garbage direction.
            # Measured separation on the ep9 weights (300 batches): median 2.86, p99 11.45, max
            # 15.69, none above 100; the worst normal value anywhere in training was 99.3 (ep1).
            # A threshold of 200 therefore sits 2x above anything healthy and 5x below the
            # smallest observed catastrophe. Only active after warmup, where early gradients are
            # legitimately large (run3 ep0 reached 483).
            if (a.grad_spike_reject > 0 and gstep >= a.warmup_steps
                    and gnf_pre > a.grad_spike_reject):
                spike_skips += 1
                opt.zero_grad(set_to_none=True)
                if is_main:
                    print(f"[SPIKE] grad {gnf_pre:.1f} > {a.grad_spike_reject} at g{gstep} ep{ep} "
                          f"(#{spike_skips}) -- step REJECTED on all ranks (clipping would keep "
                          f"the direction)", flush=True)
                gstep += 1
                continue
            opt.step()
            # PARAMETER RESYNC. DDP synchronises GRADIENTS, never parameters -- it relies on every
            # rank computing the identical update from the identical gradient. Measured on the 8-rank
            # 0.3B configuration that assumption breaks: with gradient norms of 2e3-4e4 the clip
            # coefficient falls to ~3e-5, and the two smallest-gradient tensors (t_mlp.2.bias,
            # text_mlp.3.bias) land at the edge of float32 resolution after rescaling. Their
            # parameters then drift apart monotonically (2.6e-4 by step 1, 1.9e-3 by step 6) while
            # every gradient stays bitwise equal. Broadcasting rank 0's parameters restores the
            # invariant the design already assumes; it is a no-op whenever the ranks agree.
            if ddp and a.param_resync_steps > 0 and gstep % a.param_resync_steps == 0:
                with torch.no_grad():
                    for _p in raw_model.parameters():
                        dist.broadcast(_p.data, src=0)
            gnf = float(gn)
            g_sum += gnf; g_max = max(g_max, gnf)
            tot += step_loss; n += 1
            gstep += 1
            if is_main and gstep % 200 == 0:
                print(f"[g{gstep}] ep{ep} loss={step_loss:.4f} grad={gnf:.3f} "
                      f"lr={opt.param_groups[0]['lr']:.2e}", flush=True)
            if a.val_every_steps > 0 and gstep % a.val_every_steps == 0:
                run_validation(ep, at_epoch_end=False)
            if a.ckpt_snapshot_steps > 0 and gstep % a.ckpt_snapshot_steps == 0 and is_main:
                atomic_save({"model": model_state_for_ckpt(), **lora_extra(), "opt": opt.state_dict(),
                             "epoch": ep, "gstep": gstep, "best_val": best_val, "args": vars(a),
                         "artic_strikes": artic_strikes,
                             # MID-epoch: resume must refuse this even if the file is renamed
                             "at_epoch_end": False, "guard_history": guard_history,
                             "schedule_history": schedule_history,
                             "calib_history": calib_history,
                             "healthy": last_health, "val_over_best": last_vob,
                             "ktjd_pins": ktjd_pins,
                             "rng": {"cpu": torch.get_rng_state(),
                                     "cuda": torch.cuda.get_rng_state_all(),
                                     "np": np.random.get_state()}},
                            out / f"g{gstep:07d}_model.pt")
                # Rolling window: these carry optimizer state and are ~1 GB each, so an
                # unbounded series fills the filesystem (42 GB in 1.5 h at 100-step spacing).
                # Keeping the most recent N still spans thousands of steps before any blow-up.
                snaps = sorted(q for q in out.glob("g[0-9][0-9][0-9][0-9][0-9][0-9][0-9]_model.pt")
                               if q.stem[1:-6].isdigit())
                for old_snap in snaps[:-a.ckpt_snapshot_keep]:
                    try:
                        old_snap.unlink()
                    except OSError:
                        pass
        if ddp:
            # g_sum rides in the SAME reduction so the printed mean divides a GLOBAL sum by the
            # GLOBAL step count (a local g_sum over a global n would understate the mean 4x).
            agg = torch.tensor([tot, float(n), g_sum], device=dev)
            dist.all_reduce(agg)
            gmax_t = torch.tensor([g_max], device=dev)
            dist.all_reduce(gmax_t, op=dist.ReduceOp.MAX)
            tot, n, g_sum, g_max = float(agg[0]), int(agg[1]), float(agg[2]), float(gmax_t[0])
        if acc_i > 0:                      # a trailing partial accumulation never becomes a step (deterministic step count)
            opt.zero_grad(set_to_none=True)
            if is_main and ep == start_ep:
                print(f"[train] {acc_i} trailing micro-batch(es) per epoch are dropped (epoch = {len(dl_tr) // a.grad_accum} steps)", flush=True)
            acc_i, acc_loss, window_bad = 0, 0.0, False
        if is_main:
            print(f"=== epoch {ep} done in {time.time()-t0:.1f}s | train_flow={tot/max(n,1):.5f} "
                  f"| grad mean={g_sum/max(n,1):.3f} max={g_max:.3f} ===", flush=True)

        if a.val_every_steps > 0 or (ep + 1) % a.val_every == 0 or ep == a.epochs - 1:
            # ALWAYS validate at the epoch boundary when a step cadence is active: that is the
            # only point whose checkpoint can be resumed without desynchronizing epoch/gstep.
            run_validation(ep, at_epoch_end=True)
        # RESUME POLICY: resume always restarts at epoch ck["epoch"]+1. For a MID-epoch snapshot
        # that discards the remainder of the interrupted epoch -- statistically harmless here
        # because draws are random (balanced or shuffled), while gstep/lr/warmup continue exactly.
        # no-val (all-train) runs keep last_model.pt only: periodic epNNNN snapshots would carry no
        # val/health fields and cost ~1.9 GB each (codex 2026-09-03)
        if (ep + 1) % a.ckpt_every == 0 and is_main and len(ds_va) > 0:
            atomic_save({"model": model_state_for_ckpt(), **lora_extra(), "opt": opt.state_dict(),
                         "epoch": ep, "gstep": gstep, "best_val": best_val, "args": vars(a),
                         "artic_strikes": artic_strikes,
                         "at_epoch_end": True,   # epoch boundary: resume-safe
                         "guard_history": guard_history, "schedule_history": schedule_history,
                 "calib_history": calib_history,
                         # carries the health of the most recent validation: an epoch checkpoint
                         # written after a blow-up must not look resume-safe (codex 2026-08-24)
                         "healthy": last_health, "val_over_best": last_vob,
                         "ktjd_pins": ktjd_pins,
                         "rng": {"cpu": torch.get_rng_state(),
                                 "cuda": torch.cuda.get_rng_state_all(),
                                 "np": np.random.get_state()}},
                        out / f"ep{ep+1:04d}_model.pt")
    if is_main:
        print("=== training loop complete ===", flush=True)
    if ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
