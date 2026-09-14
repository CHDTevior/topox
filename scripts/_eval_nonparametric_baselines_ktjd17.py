"""Non-parametric baselines for the frozen text-to-motion protocol: how far do real training clips alone get?

Three predictors, none of them a model, all scored by the SAME frozen evaluator, the same dataset-order pools and the same
false-negative masks as the canonical gen-eval:

  retrieval_text   for each validation clip, the TRAINING clip of the SAME rig whose caption embedding (the LLM2Vec vector the
                   generator conditions on) is closest to the validation caption's -- text-conditioned retrieval from the corpus.
  retrieval_random the same, with the training clip drawn uniformly at random (seeded): what rig identity alone is worth.
  rig_mean         the rig's per-frame mean over its training clips (frames beyond a clip's end do not contribute).

A clip shorter than the target holds its last frame. Everything is served through the merge-stage projection the evaluator sees,
so the only difference from a generated sample is where the motion came from. The evaluation order is bound to the canonical
report's protocol.eval_order_sha256, as in the evaluator-validation controls.

usage:
  python scripts/_eval_nonparametric_baselines_ktjd17.py --ckpt runs/v2_noik_pilot36m_r1acc/ep0100_model.pt \
      --report runs/v2_noik_pilot36m_r1acc/gen_eval_ep100_a100x2_s20_pool64_bound.json --out runs/_evalctrl/nonparam_p36.json
"""
from __future__ import annotations
import argparse, hashlib, io, json, sys
from pathlib import Path
import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from scripts._eval_v2_gen_in_evalspace import (load_evaluator, encode_split, fid, PROTOCOL_VAL_N,           # noqa: E402
                                               source_fingerprint, LEGACY_SOURCE_FINGERPRINTS)              # noqa: E402
from scripts._eval_controls_ktjd17 import pooled_rprec, within_rig, group_chance, EVAL_CKPT                  # noqa: E402
from src.data.ktjd17_incontext import Ktjd17Base, _STD_FLOOR                                                 # noqa: E402
from src.data.ktjd17_t2m_eval_dataset import Ktjd17T2MEvalDataset                                            # noqa: E402
from src.data.ktjd17.loader import load_motion_npz                                                           # noqa: E402

# THE COMMON EVALUATION PROTOCOL this script declares and pins (codex 2026-09-10 r3 #3). The canonical reports carry
# no acceptance digest, so there is no historical acceptance set to bind to; what is bound is THIS pass. Every arm --
# the generator re-scored from the report's own shards, and the three baselines -- goes through one acceptance
# construction in one process, and the ordered acceptance keys, the grouping rule and a digest of the acceptance
# metadata actually used are written into the report, so a re-run reproduces them and any later change is visible.
# Scoring every arm together removes the comparison against a stale recorded number; it does NOT by itself make a
# grouping change harmless -- regrouping moves the arms by different amounts -- which is why the grouping is pinned.
PROTOCOL_ID = "topx-nonparam-common-v1"
ACCEPTANCE_KEYS = ("motion_id", "source_motion_id", "caption_text")          # in this order
ACCEPTANCE_RULE = ("off-diagonal pairs sharing ANY of the ordered keys are accepted (union of the three); "
                   "the diagonal is the positive and is never masked")


def _paired_summary(pools, idx):
    """R@{1,2,3} with a bootstrap over ONE resample-index matrix shared by every arm (codex r3 #4).

    Drawing fresh indices per arm gives valid marginal intervals but not a paired comparison: two arms with
    identical per-pool scores came out with different limits in a probe. The shared indices make the intervals
    -- and the generated-minus-baseline deltas below -- paired."""
    out = {}
    for k in (1, 2, 3):
        v = np.asarray([p[k] for p in pools], dtype=np.float64)
        b = v[idx].mean(axis=1)
        out[str(k)] = {"mean": float(v.mean()), "lo95": float(np.percentile(b, 2.5)),
                       "hi95": float(np.percentile(b, 97.5))}
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True, help="any checkpoint of the arm whose protocol this scores: it fixes the corpus, the cut and the caption cache")
    ap.add_argument("--report", required=True, help="the canonical gen-eval report of that arm (binds the evaluation order and the evaluator)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--eval_ckpt", default=EVAL_CKPT); ap.add_argument("--pool", type=int, default=64)
    ap.add_argument("--seed", type=int, default=42); ap.add_argument("--n_boot", type=int, default=2000)
    ap.add_argument("--encode_batch", type=int, default=64); ap.add_argument("--min_rig_pool", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0, help="score only the first N clips in dataset order (smoke; not the protocol)")
    ap.add_argument("--subset", default="", help="a clean-subset json whose sha256 the report pins: score exactly its clips, "
                                                 "in dataset order, at the report's pool")
    ap.add_argument("--allow_manifest_after_ckpt", action="store_true",
                    help="score against a manifest written after the checkpoint; state the reason in the run log")
    ap.add_argument("--allow_runtime_drift", action="store_true",
                    help="score on a runtime other than the one the report records; state the reason in the run log")
    a = ap.parse_args()
    if a.limit and a.limit < a.pool:
        raise SystemExit(f"[refuse] --limit {a.limit} is smaller than one pool of {a.pool}")
    if a.limit and a.subset:
        raise SystemExit("[refuse] --limit is a smoke and --subset is a protocol run; they cannot be combined")
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    # the recorded runtime's precision flags do not exist on CPU, so a CPU fallback silently scores under
    # different arithmetic while every version string still matches (codex r2 P1 #4)
    if dev == "cpu":
        _cpu_ok = False   # set after the report is read, when we know what it was scored on
    rng = np.random.default_rng(a.seed)

    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    ca = ck["args"]; ca = vars(ca) if not isinstance(ca, dict) else ca
    root, excl = ca["ktjd_root"], (ca.get("exclude_clips") or None)
    # the baselines live in the EVALUATOR's per-cell space of the PARENT corpus: a representation view changes how a generator
    # is served, not what a real clip is, so a view checkpoint is scored against its parent's clips
    deriv = json.loads((Path(root) / "derivation.json").read_text()) if (Path(root) / "derivation.json").is_file() else {}
    parent_root = str(deriv.get("parent_root") or root)
    parent_stats = str(((deriv.get("representation") or {}).get("parent_norm_stats") or {}).get("path")
                       or ca.get("ktjd_percell_stats", "data/ktjd17_percell_stats_v1.npz"))
    # PROVENANCE FIRST, in the checkpoint's OWN space: the pins were written by a run whose normalization may be rest or
    # scale_only, and target_centering / the per-cell artifact are part of them (codex r2 #4). The view itself is verified before
    # its parent pins are trusted (codex r2 #1): the checkpoint pins the derivation bytes, so a swapped derivation.json that
    # re-points the parent statistics cannot pass by declaring itself.
    pins = ck.get("ktjd_pins") or {}
    if deriv:
        if str(pins.get("derivation_sha256")) != hashlib.sha256((Path(root) / "derivation.json").read_bytes()).hexdigest():
            raise SystemExit(f"[refuse] {root}/derivation.json is not the view the checkpoint was trained on "
                             f"(pinned {str(pins.get('derivation_sha256'))[:12]})")
        rep = (deriv.get("representation") or {})
        if rep and str(rep.get("parent_norm_stats", {}).get("sha256")) != hashlib.sha256(Path(parent_stats).read_bytes()).hexdigest():
            raise SystemExit(f"[refuse] {parent_stats} is not the parent statistics the view {root} was derived with")
        if str(deriv.get("parent_manifest_sha256")) != hashlib.sha256((Path(parent_root) / "manifests" / "clips.jsonl").read_bytes()).hexdigest():
            raise SystemExit(f"[refuse] the parent manifest differs from the one the view {root} was derived from")
    base_gen = Ktjd17Base(root, caption_emb_cache=ca["caption_cache"], joint_semantics=ca["joint_sem"],
                          texts_json=ca["texts_json"], percell_stats=ca.get("ktjd_percell_stats", "data/ktjd17_percell_stats_v1.npz"),
                          exclude_clips=excl, normalization=str(ca.get("rep_norm") or "percell"))
    live = {**base_gen.provenance, "exclusion": base_gen.provenance_exclusion}
    drift = sorted(k for k, v in pins.items() if k in live and live[k] != v)
    if drift:
        raise SystemExit(f"[refuse] data drift vs the checkpoint's ktjd_pins: {drift}")
    # the baselines are REAL clips, so they are served in the evaluator's own per-cell space of the parent corpus
    base = base_gen if (parent_root == root and base_gen.normalization == "percell") else Ktjd17Base(
        parent_root, caption_emb_cache=ca["caption_cache"], joint_semantics=ca["joint_sem"], texts_json=ca["texts_json"],
        percell_stats=parent_stats, exclude_clips=excl, normalization="percell")
    eval_ds = Ktjd17T2MEvalDataset(base, "val", max_frames=240, exclude=excl)
    if not a.limit and len(eval_ds) != PROTOCOL_VAL_N:
        raise SystemExit(f"[refuse] eval val has {len(eval_ds)} clips, protocol pins {PROTOCOL_VAL_N}")
    rows = {str(r["clip_id"]): r for r in base._rows}
    T_of = {str(r["clip_id"]): int(r["T_target"]) for r in base._rows}
    target_frames = int(ca["target_frames"])
    # the corpus' own frame rate: the velocity channels are per second, so the rig mean must be differenced with it
    FPS_TARGET = float(json.loads((Path(parent_root) / "schema.json").read_text())["fps_target"])

    rp = json.loads(Path(a.report).read_text())
    if (rp.get("protocol") or {}).get("eval_split") == "all":
        raise SystemExit("[refuse] this report scores a held-out cohort (--eval_split all): its clips are all evaluation targets, "
                         "so there is no same-rig training support for a non-parametric baseline")
    if int(rp["protocol"].get("pool", 0)) != a.pool:
        raise SystemExit(f"[refuse] the report pools {rp['protocol'].get('pool')}, this run pools {a.pool}")
    _rp_sub = rp["protocol"].get("subset")
    if bool(a.subset) != bool(_rp_sub not in (None, {})):
        raise SystemExit(f"[refuse] the report {'scores a subset' if _rp_sub else 'is a full-set scoring'} and this run "
                         f"{'names one' if a.subset else 'names none'}: --subset must match the report's cut")
    subset_ids = None
    if a.subset:
        _sp = Path(a.subset)
        _got = hashlib.sha256(_sp.read_bytes()).hexdigest()
        if _got != str(_rp_sub.get("sha256")):
            raise SystemExit(f"[refuse] {a.subset} ({_got[:12]}) is not the subset the report scored "
                             f"({str(_rp_sub.get('sha256'))[:12]})")
        _sj = json.loads(_sp.read_text())
        subset_ids = [str(c["clip_id"]) for c in _sj["clean"]]      # the rows carry the clip id and its payload digest
        if len(set(subset_ids)) != len(subset_ids):
            raise SystemExit(f"[refuse] {a.subset} lists a clip twice")
        if len(subset_ids) != int(rp["protocol"].get("val_n", -1)):
            raise SystemExit(f"[refuse] {a.subset} holds {len(subset_ids)} clips, the report scored "
                             f"{rp['protocol'].get('val_n')}")
        # THE ORDER BINDING FOR A DIGEST-LESS REPORT (codex 2026-09-10 #2): the pools chunk the dataset order, which
        # is the manifest's order, and reproducing the GT ceiling does not detect a reordering -- a probe held the
        # ceiling exactly while a baseline moved. The subset file pins the manifest and the exclusions it was cut
        # from, so bind to those.
        for _k, _path in (("manifest_sha256", Path(str(_sj["ktjd_root"])) / "manifests" / "clips.jsonl"),
                          ("exclusions_sha256", Path(str(_sj["exclusions"])))):
            _live = hashlib.sha256(_path.read_bytes()).hexdigest()
            if _live != str(_sj.get(_k)):
                raise SystemExit(f"[refuse] {_path} ({_live[:12]}) is not the {_k[:-7]} {a.subset} was cut from "
                                 f"({str(_sj.get(_k))[:12]}): the dataset order the pools chunk would differ")
        if str(_sj["ktjd_root"]) != root or str(_sj["exclusions"]) != str(excl):
            raise SystemExit(f"[refuse] {a.subset} was cut from {_sj['ktjd_root']!r} / {_sj['exclusions']!r}, the "
                             f"checkpoint trains on {root!r} / {str(excl)!r}")
    # THE SPACE THE SHARDS LIVE IN (codex r3 #1): a shard holds the sample in the GENERATOR's space. The canonical merge
    # converts a representation view's or a non-per-cell arm's samples into the evaluator's per-cell space before it
    # encodes them; this script re-scores the shards directly, so it is defined only for the arms that need no
    # conversion. Scoring a converted arm here would put its generator in a different space from the real clips it is
    # compared with, and neither the shape contract nor the GT ceiling would show it.
    _gn, _gr = str(rp["protocol"].get("gen_normalization") or ""), str(rp["protocol"].get("gen_representation") or "")
    _space_from = "report"
    if not _gn and not _gr:
        # a report written before the merge stamped those two fields. The shards were produced by THIS checkpoint
        # (its sha256 is checked above) sampling from THIS dataset, so the checkpoint's own verified pins say what
        # space they are in -- the same two quantities the merge stamps, read at their source.
        _gn = str(base_gen.normalization)
        _gr = str(((deriv.get("representation") or {}).get("id")) or "ktjd17")
        _space_from = "checkpoint"
    if (_gn, _gr) != ("percell", "ktjd17"):
        raise SystemExit(f"[refuse] the generator samples in normalization {_gn!r} / representation {_gr!r} (read from "
                         f"the {_space_from}); the samples would need the canonical merge's conversion into the "
                         f"evaluator's per-cell KTJD-17 space, which this script does not carry. Score such an arm "
                         f"with the canonical merge.")
    if str(rp["protocol"].get("gen_ckpt_sha256")) != hashlib.sha256(Path(a.ckpt).read_bytes()).hexdigest():
        raise SystemExit(f"[refuse] {a.ckpt} is not the checkpoint the report scored -- the report's data bindings would not apply")
    # score under the arithmetic the canonical pass used (codex r3 #4): the report records the runtime its samples and its scoring
    # ran with; TF32 on an A100 changes the evaluator's matmuls, so the same embeddings need the same flags
    import os
    if "NVIDIA_TF32_OVERRIDE" in os.environ:
        raise SystemExit("[refuse] NVIDIA_TF32_OVERRIDE is set; the recorded runtime cannot see it")
    _rt = (rp["protocol"].get("generation") or {}).get("runtime") or {}
    if not _rt:
        raise SystemExit(f"[refuse] {a.report} records no runtime -- it cannot certify the arithmetic this baseline "
                         f"would be scored under")
    # the recorded runtime is the whole runtime, not only its precision flags (codex r1 P0 #2): a different torch,
    # CUDA or cuDNN changes the evaluator's embeddings, and matching GT recalls does not rule that out
    _live = {"torch": torch.__version__, "cuda": torch.version.cuda,
             "cudnn": (torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None)}
    _rt_drift = {k: (_rt.get(k), _live[k]) for k in ("torch", "cuda", "cudnn")
                 if str(_rt.get(k)) != str(_live[k])}
    if dev == "cpu" and not a.allow_runtime_drift:
        raise SystemExit("[refuse] no CUDA device: the report was scored with the runtime it records, whose TF32 flags "
                         "have no meaning on CPU, so a CPU pass would answer it under different arithmetic "
                         "(pass --allow_runtime_drift with a reason to score on CPU anyway)")
    if _rt_drift and not a.allow_runtime_drift:
        raise SystemExit(f"[refuse] runtime differs from the scoring the report records: {_rt_drift} -- score this "
                         f"baseline on the canonical runtime, or pass --allow_runtime_drift with a reason")
    torch.backends.cuda.matmul.allow_tf32 = bool(_rt.get("allow_tf32_matmul"))
    torch.backends.cudnn.allow_tf32 = bool(_rt.get("allow_tf32_cudnn"))
    torch.set_float32_matmul_precision(str(_rt.get("float32_matmul_precision", "highest")))
    # the code that turns embeddings into the numbers (projection, encoding, FID) must be the code the report scored
    # with (codex r1 P0 #3): checkpoint hashes and clip order do not cover it
    _canon_fp = str((rp["protocol"].get("generation") or {}).get("scoring_source_fingerprint") or "")
    _live_fp = source_fingerprint(ca.get("anchor", "none"), flat=bool(ca.get("flat_joints", 0)))
    if not _canon_fp:
        raise SystemExit(f"[refuse] {a.report} records no scoring_source_fingerprint -- re-run its merge")
    if _canon_fp != _live_fp and _canon_fp not in LEGACY_SOURCE_FINGERPRINTS:
        raise SystemExit(f"[refuse] the scoring code changed since the report was written "
                         f"({_canon_fp[:16]} recorded, {_live_fp[:16]} live, not an audited legacy fingerprint)")
    canon_order = str(rp["protocol"].get("eval_order_sha256") or "")
    if not canon_order and not a.limit and not a.subset:
        raise SystemExit(f"[refuse] {a.report} predates the evaluation-order digest -- re-run its merge")
    # THE ACCEPTANCE BINDING (codex r1 P0 #1): the order digest fixes WHICH clips are scored, and reproducing the GT
    # ceiling does not fix WHICH clips accept each other -- a probe kept both while regrouping official ids and moved
    # a baseline's R@1 from 0.5 to 1.0. The corpus decides the groups, so pin the corpus the canonical shards ran on.
    _canon_prov = set()
    for _sh in (rp["protocol"].get("generation") or {}).get("shards") or []:
        _p = Path(str(_sh.get("path", "")))
        if not _p.is_file():
            raise SystemExit(f"[refuse] the report's shard {_p} is gone -- the acceptance sets cannot be bound to it")
        _got = hashlib.sha256(_p.read_bytes()).hexdigest()
        if str(_sh.get("sha256")) != _got:
            raise SystemExit(f"[refuse] shard {_p} does not match the report's sha256")
        _m = np.load(_p, allow_pickle=True)["__meta"]
        _m = json.loads(str(_m)) if _m.dtype.kind in "US" else _m.item()
        _canon_prov.add(str(_m.get("base_provenance_sha256")))
    if len(_canon_prov) != 1:
        raise SystemExit(f"[refuse] the report's shards disagree about the corpus they scored: {_canon_prov}")
    # generation hashes ITS OWN dataset (base_gen), not the parent per-cell one, and serialises with default=str
    # (codex r2 P1 #2: a scale-only or rest arm's shards carry their own digest, which the parent's would never match)
    _live_prov = hashlib.sha256(json.dumps({**base_gen.provenance, "exclusion": base_gen.provenance_exclusion},
                                           sort_keys=True, default=str).encode()).hexdigest()
    if _canon_prov.pop() != _live_prov:
        raise SystemExit(f"[refuse] the corpus here is not the corpus the report scored -- the acceptance groups, "
                         f"the captions and the training split would all differ")

    # ---- the support pool: every TRAINING clip of each rig, with its official caption embedding ----
    # WHAT DECIDES THE SUPPORT SET (codex r3 #2, r4 #1): a frozen parent corpus writes no manifest digest into its
    # provenance, so nothing checked above covers the split column that decides which clips the baseline may retrieve
    # from, and no record of the split as it stood when the checkpoint was written exists anywhere. Three things are
    # therefore checked and written down -- the manifest predates the checkpoint, its accepted rows carry only the two
    # splits the training cut uses, and its bytes are digested into the report -- and the report says in words what they
    # do and do not establish. A train/val pair swapped BEFORE the checkpoint was saved, with the canonical report
    # generated afterwards, passes all three: the counts, the provenance digest and the caption payload are unchanged.
    # This is a snapshot of the split as it stands, not a retrospective proof of the split the model trained under.
    _ck_mtime = Path(a.ckpt).stat().st_mtime
    manifest_binding = {}
    for _r in dict.fromkeys((root, parent_root)):          # the view's manifest, and the parent's when they differ
        _mp = Path(_r) / "manifests" / "clips.jsonl"
        _mt = _mp.stat().st_mtime
        if _mt > _ck_mtime and not a.allow_manifest_after_ckpt:
            raise SystemExit(f"[refuse] {_mp} was written after {a.ckpt} -- the split it declares is not the one the "
                             f"checkpoint trained under (pass --allow_manifest_after_ckpt with a reason)")
        _splits: dict[str, int] = {}
        for _line in _mp.open():
            _row = json.loads(_line)
            if str(_row.get("status")) != "accept":
                continue
            _splits[str(_row.get("split"))] = _splits.get(str(_row.get("split")), 0) + 1
        _extra = sorted(set(_splits) - {"train", "val"})
        if _extra:
            raise SystemExit(f"[refuse] {_mp} declares accepted clips in split(s) {_extra}: the training cut this "
                             f"baseline retrieves from is train/val alone, so a third split means the manifest is not "
                             f"the one the checkpoint trained under")
        manifest_binding[_r] = {"sha256": hashlib.sha256(_mp.read_bytes()).hexdigest(), "mtime": _mt,
                                "ckpt_mtime": _ck_mtime, "accepted_by_split": _splits}
    # the timestamp clause of the binding below must say what actually happened, not what the check would have
    # enforced: --allow_manifest_after_ckpt lets a newer manifest through (codex r4/r5)
    _newer = sorted(r for r, m in manifest_binding.items() if m["mtime"] > m["ckpt_mtime"])
    support_binding = (
        ("The manifest predates the checkpoint" if not _newer else
         f"The manifest of {', '.join(_newer)} was written AFTER the checkpoint and was accepted under "
         f"--allow_manifest_after_ckpt, so no timestamp evidence stands") +
        " and carries only train and val, its bytes are digested here, and the corpus provenance is pinned to the "
        "checkpoint. None of that records the split as it stood when the checkpoint was written, which no artifact "
        "of this corpus does: a swap of a train/val pair made before the checkpoint was saved would not be visible "
        "here. The support set is the training split this manifest declares, restricted to the checkpoint's view.")
    # THE SUPPORT SET IS THE TRAINING SPLIT THE MANIFEST DECLARES TODAY, restricted to the checkpoint's own view: a view can
    # train on fewer clips than the parent holds, and a baseline that retrieves from the parent's extra clips would be given
    # data the model never saw. What this does NOT establish is checkpoint-era membership -- see the binding below.
    gen_rows = {str(r["clip_id"]): r for r in base_gen._rows}
    train_by_rig: dict[str, list[str]] = {}
    missing_in_parent = []
    for cid, r in gen_rows.items():
        if str(r.get("split")) != "train":
            continue
        if cid not in rows:
            missing_in_parent.append(cid); continue
        if str(rows[cid]["rig_id"]) != str(r["rig_id"]):
            raise SystemExit(f"[refuse] clip {cid} is rig {r['rig_id']!r} in the checkpoint's view and {rows[cid]['rig_id']!r} in the parent")
        train_by_rig.setdefault(str(r["rig_id"]), []).append(cid)
    if missing_in_parent:
        raise SystemExit(f"[refuse] {len(missing_in_parent)} training clips of the checkpoint's view are absent from the parent corpus "
                         f"(e.g. {missing_in_parent[:3]}): the support set cannot be served")
    for rig in train_by_rig:
        train_by_rig[rig].sort()
    support_sha = hashlib.sha256("\n".join(f"{rig}\t" + ",".join(train_by_rig[rig])
                                           for rig in sorted(train_by_rig)).encode()).hexdigest()
    cap_of = lambda cid: np.asarray(base._cap_embs[base._cap_rows[cid][0]], dtype=np.float32)   # caption 0 = the official one

    def raw_of(cid: str, J: int) -> np.ndarray:
        pay = load_motion_npz(Path(parent_root) / rows[cid]["motion_relpath"], expected_fps_target=30.0)
        # the payload carries its own identity: a file that holds another clip's motion would let retrieval serve the query's
        # own motion under a training clip's name (codex r1 #4)
        if str(pay["clip_id"]) != cid or str(pay["rig_id"]) != str(rows[cid]["rig_id"]):
            raise SystemExit(f"[refuse] {rows[cid]['motion_relpath']} holds clip {pay['clip_id']!r} of rig {pay['rig_id']!r}, "
                             f"the manifest says {cid!r} / {rows[cid]['rig_id']!r}")
        return np.asarray(pay["motion"], dtype=np.float64)[:, :J, :17]

    def to_eval_space(raw: np.ndarray, rig: str, J: int, Tv: int) -> np.ndarray:
        mu, sd = base._stats(rig); cv = base.static_masks(rig)["channel_valid"][:J]
        g = ((raw - mu[None, :J]) / (sd[None, :J] + _STD_FLOOR)).astype(np.float32); g[:, ~cv] = 0.0
        out = np.zeros((target_frames, J, 17), np.float32); out[:Tv] = g[:Tv]
        return out

    def fill(src: np.ndarray, Tv: int) -> np.ndarray:
        """src [Ts,J,17] -> [Tv,J,17]: truncate, or hold the last frame with the velocity channels zeroed.

        A held pose does not move: repeating the last frame's world velocity (channels 9:12) would hand the evaluator a
        stationary body carrying a metre-per-second velocity, a contradiction no real clip contains (codex r2 #5). The last
        REAL frame is part of the stationary stretch too -- its forward difference to the first held frame is zero -- so its
        velocity is zeroed as well."""
        if src.shape[0] >= Tv:
            return src[:Tv]
        out = np.concatenate([src, np.repeat(src[-1:], Tv - src.shape[0], axis=0)], axis=0)
        out[src.shape[0] - 1:, :, 9:12] = 0.0
        return out

    variants = {"generated": {}, "retrieval_text": {}, "retrieval_random": {}, "rig_mean": {}}
    # The generator is re-scored HERE, from the canonical shards whose sha256 the report pins, rather than compared
    # with the number the report wrote down (codex r2: acceptance or scoring drift that nobody detects can move a
    # baseline against a saved score in either direction). Every arm then passes through one acceptance construction,
    # one evaluator and one pooling in a single process, and the re-scored generator is reported next to the report's
    # own number so any difference from it is visible. What that establishes is that no arm is compared against a
    # stale number -- NOT that the grouping is immaterial: a regrouping moves the arms by different amounts (a probe
    # merged the source ids of 64 items and took a baseline's R@1 from 0.0 to 1.0 while GT and the generator both
    # stayed at 1.0). The grouping rule and the acceptance metadata are therefore pinned in report["protocol"].
    for _sh in (rp["protocol"].get("generation") or {}).get("shards") or []:
        with np.load(Path(str(_sh["path"])), allow_pickle=False) as _z:
            for _k in _z.files:
                if not _k.startswith("clip__"):
                    continue
                _mid = _k[len("clip__"):]
                if _mid in variants["generated"]:
                    raise SystemExit(f"[refuse] clip {_mid} appears in more than one of the report's shards")
                variants["generated"][_mid] = np.asarray(_z[_k], dtype=np.float32)
    if not a.limit and len(variants["generated"]) != PROTOCOL_VAL_N:
        raise SystemExit(f"[refuse] the report's shards hold {len(variants['generated'])} samples, protocol val is "
                         f"{PROTOCOL_VAL_N} -- they are not the canonical generation of this arm")
    rig_of, order, picks = {}, [], {}
    mean_cache: dict[str, np.ndarray] = {}
    want = set(subset_ids) if subset_ids is not None else None
    for i in range(len(eval_ds)):
        it = eval_ds[i]; mid = str(it["motion_id"]); rig = str(it["object_type"]); order.append(mid); rig_of[mid] = rig
        if a.limit and i >= a.limit:
            continue
        if want is not None and mid not in want:
            continue
        # the SERVED length, which is what the evaluator's frame mask uses -- not the manifest's T_target (codex r3 #3)
        J = len(base.skeleton(rig)["parents"]); Tv = min(int(it["num_frames"]), target_frames)
        if int(it["num_frames"]) != min(int(T_of[mid]), target_frames):
            raise SystemExit(f"[refuse] clip {mid}: the evaluator serves {int(it['num_frames'])} frames, the manifest "
                             f"says {T_of[mid]} capped at {target_frames}")
        pool = train_by_rig.get(rig) or []
        if not pool:
            raise SystemExit(f"[refuse] rig {rig} has no training clip: a same-rig retrieval baseline is undefined for it")
        q = cap_of(mid); qn = q / (np.linalg.norm(q) + 1e-8)
        cands = np.stack([cap_of(c) for c in pool])
        cs = (cands / (np.linalg.norm(cands, axis=1, keepdims=True) + 1e-8)) @ qn
        best = pool[int(np.argmax(cs))]
        rnd = pool[int(rng.integers(len(pool)))]
        picks[mid] = {"rig": rig, "n_train": len(pool), "text": best, "text_cos": float(cs.max()), "random": rnd}
        variants["retrieval_text"][mid] = to_eval_space(fill(raw_of(best, J), Tv), rig, J, Tv)
        variants["retrieval_random"][mid] = to_eval_space(fill(raw_of(rnd, J), Tv), rig, J, Tv)
        if rig not in mean_cache:
            acc = np.zeros((target_frames, J, 17)); cnt = np.zeros(target_frames)
            for c in pool:
                m = raw_of(c, J)[:target_frames]
                acc[:m.shape[0]] += m; cnt[:m.shape[0]] += 1
            live = int((cnt > 0).sum())
            mu_r = np.zeros_like(acc); mu_r[:live] = acc[:live] / cnt[:live, None, None]
            if live < target_frames:                       # frames past the longest training clip hold the last mean frame
                mu_r[live:] = mu_r[live - 1]
            # The population changes whenever a shorter clip ends, so averaged velocities stop describing the averaged
            # positions (codex r1 P1 #5: one frame stored -0.184 where the position difference implies -13.118).
            # Recompute them from the trajectory this predictor actually serves.
            world = mu_r[:, :, 0:3].copy()
            world[:, :, 0] += mu_r[:, 0:1, 13]
            world[:, :, 2] += mu_r[:, 0:1, 14]
            vel = np.zeros_like(world)
            if target_frames >= 2:
                vel[:-1] = (world[1:] - world[:-1]) * FPS_TARGET
                vel[-1] = vel[-2]
            mu_r[:, :, 9:12] = vel
            if live < target_frames:
                mu_r[live - 1:, :, 9:12] = 0.0             # a held pose has no velocity
            mean_cache[rig] = mu_r
        variants["rig_mean"][mid] = to_eval_space(mean_cache[rig][:Tv], rig, J, Tv)
        _g = variants["generated"].get(mid)
        if _g is None:
            raise SystemExit(f"[refuse] clip {mid} is scored here but absent from the report's shards")
        if tuple(_g.shape) != (target_frames, J, 17):
            raise SystemExit(f"[refuse] the report's sample for {mid} has shape {_g.shape}, expected "
                             f"{(target_frames, J, 17)}")
        if (i + 1) % 500 == 0:
            print(f"[nonparam] built {i + 1} clips", flush=True)
    keep = set(variants["retrieval_text"]) if (a.limit or a.subset) else None
    if want is not None and keep != want:
        raise SystemExit(f"[refuse] {len(want - keep)} of the subset's clips are not in the evaluation cut "
                         f"(e.g. {sorted(want - keep)[:3]}): the subset must be built from the same corpus cut")

    core, eval_sha = load_evaluator(a.eval_ckpt, dev)
    if eval_sha != str(rp["protocol"].get("eval_ckpt_sha256")):
        raise SystemExit(f"[refuse] evaluator {a.eval_ckpt} ({eval_sha[:12]}) is not the report's ({str(rp['protocol'].get('eval_ckpt_sha256'))[:12]})")
    report = {"ckpt": a.ckpt, "report": a.report, "eval_ckpt_sha256": eval_sha, "pool": a.pool, "seed": a.seed,
              "n_boot": a.n_boot, "limit": a.limit, "smoke": bool(a.limit), "parent_root": parent_root,
              "parent_stats_sha256": hashlib.sha256(Path(parent_stats).read_bytes()).hexdigest(),
              "n_clips": len(variants["retrieval_text"]), "variants": {},
              "protocol": {"id": PROTOCOL_ID, "acceptance_keys": list(ACCEPTANCE_KEYS),
                           "acceptance_rule": ACCEPTANCE_RULE, "pool": a.pool,
                           "gen_normalization": _gn, "gen_representation": _gr, "space_read_from": _space_from,
                           "subset": ({"path": a.subset, "sha256": hashlib.sha256(Path(a.subset).read_bytes()).hexdigest(),
                                       "n": len(subset_ids), "rule": str((_rp_sub or {}).get("rule", ""))}
                                      if a.subset else None),
                           "manifest_binding": manifest_binding,
                           "support_clip_ids_sha256": support_sha,
                           "support_binding": support_binding,
                           "support_n_train_by_rig": {k: len(v) for k, v in sorted(train_by_rig.items())},
                           "allow_manifest_after_ckpt": bool(a.allow_manifest_after_ckpt),
                           "allow_runtime_drift": bool(a.allow_runtime_drift)},
              "report_says": {"rprec": {str(k): float(v) for k, v in rp["text_to_gen"]["rprec"].items()},
                              "fid": float(rp.get("fid_gen_vs_gt", float("nan")))}}
    te = me_gt = meta = None
    boot_idx = None            # ONE resample-index matrix, drawn once and shared by every arm (codex r3 #4)
    pool_scores: dict[str, np.ndarray] = {}
    for name, gb in variants.items():
        te_, me_gt_, me_v, meta_ = encode_split(core, eval_ds, gb, dev, a, keep=keep)
        if te is None:
            te, me_gt, meta = te_, me_gt_, meta_
            order_sha = hashlib.sha256("\n".join(str(m) for m in meta["motion_id"]).encode()).hexdigest()
            report["eval_order_sha256"] = order_sha
            if not a.limit and not a.subset and order_sha != canon_order:
                raise SystemExit(f"[refuse] the scored order {order_sha[:12]} differs from the canonical report's {canon_order[:12]}")
            # THE ORDER BINDING UNDER --subset (codex 2026-09-10 #1 of the second round): a digest-less report has
            # nothing to compare against, the manifest pins do not cover the evaluation dataset's own ordering, and
            # the scoring fingerprint does not include it -- a probe that sorted the dataset's index held the GT
            # ceiling exactly while a baseline moved. The subset file IS the order: its rows are the dataset order
            # restricted to its clips (verified 512/512 on this corpus), and the report pins its bytes, so the
            # scored sequence must equal it element for element.
            if a.subset:
                got = [str(m) for m in meta["motion_id"]]
                if got != subset_ids:
                    bad = next((k for k, (x, y) in enumerate(zip(got, subset_ids)) if x != y), min(len(got), len(subset_ids)))
                    raise SystemExit(f"[refuse] the scored order is not {a.subset}'s order (first difference at "
                                     f"position {bad}: scored {got[bad:bad + 1]}, pinned {subset_ids[bad:bad + 1]}); "
                                     f"the pools chunk this order, so they would not be the report's pools")
            rigs = [rig_of[m] for m in meta["motion_id"]]
            # the acceptance metadata THIS pass used, in evaluation order, under the pinned grouping rule
            report["protocol"]["acceptance_sha256"] = hashlib.sha256("\n".join(
                "\x1f".join(str(meta[k][i]) for k in ACCEPTANCE_KEYS)
                for i in range(len(meta["motion_id"]))).encode()).hexdigest()
            report["protocol"]["n_scored"] = len(meta["motion_id"])
            gt_pools = pooled_rprec(te, me_gt, meta, a.pool)
            boot_idx = np.random.default_rng(a.seed + 1).integers(0, len(gt_pools),
                                                                  size=(a.n_boot, len(gt_pools)))
            report["gt_ceiling"] = {"rprec": _paired_summary(gt_pools, boot_idx),
                                    "matching_text_cos": float((te * me_gt).sum(-1).mean()),
                                    "within_rig": within_rig(te, me_gt, meta, rigs, a.min_rig_pool, a.pool)}
            # THE SCORING BINDING (codex r3 #2): the text-to-GT ceiling depends on the captions, the acceptance groups
            # (official ids), the evaluator and its arithmetic -- everything the order digest alone does not cover. The
            # canonical report scored the same GT tensors in the same order, so it must reproduce exactly.
            # a subset report predates the order digest, so its binding is the ceiling: the same clips, the same
            # captions, the same acceptance groups and the same evaluator must give the number it recorded
            if not a.limit:
                canon_gt = {str(k): float(v) for k, v in rp["text_to_gt_ceiling"]["rprec"].items()}
                got_gt = {k: float(report["gt_ceiling"]["rprec"][k]["mean"]) for k in canon_gt}
                dmax = max(abs(got_gt[k] - canon_gt[k]) for k in canon_gt)
                report["canonical_gt_reproduction"] = {"report": canon_gt, "here": got_gt, "max_abs_diff": dmax}
                if dmax > 1e-3:
                    raise SystemExit(f"[refuse] the text-to-GT ceiling does not reproduce the canonical report "
                                     f"({got_gt} vs {canon_gt}): the captions, the acceptance groups or the evaluator arithmetic "
                                     f"differ from the scoring the report records")
            report["chance_R1"] = group_chance(meta, a.pool)
        pools = pooled_rprec(te, me_v, meta, a.pool)
        if len(pools) != boot_idx.shape[1]:
            raise SystemExit(f"[refuse] {name} formed {len(pools)} pools, the shared bootstrap was drawn over "
                             f"{boot_idx.shape[1]} -- the arms are not pooled alike")
        pool_scores[name] = np.asarray([p[1] for p in pools], dtype=np.float64)
        rec = {"rprec": _paired_summary(pools, boot_idx), "n_pools": len(pools),
               "matching_text_cos": float((te * me_v).sum(-1).mean()), "gt_cos": float((me_gt * me_v).sum(-1).mean()),
               "fid_vs_gt": fid(me_v, me_gt),
               "within_rig": within_rig(te, me_v, meta, rigs, a.min_rig_pool, a.pool)}
        report["variants"][name] = rec
        r1 = rec["rprec"]["1"]
        print(f"[nonparam] {name:<17} R@1 {r1['mean']:.4f} [{r1['lo95']:.4f},{r1['hi95']:.4f}] R@3 {rec['rprec']['3']['mean']:.4f} "
              f"| match {rec['matching_text_cos']:.3f} | FID {rec['fid_vs_gt']:.5f} | within-rig {rec['within_rig']}", flush=True)
    # the paper's own sentence is a DIFFERENCE, so it gets the paired interval the shared indices make possible
    _gen_v = pool_scores["generated"]
    report["paired_delta_R1_generated_minus"] = {}
    for name, v in pool_scores.items():
        if name == "generated":
            continue
        d = _gen_v - v
        b = d[boot_idx].mean(axis=1)
        report["paired_delta_R1_generated_minus"][name] = {
            "mean": float(d.mean()), "lo95": float(np.percentile(b, 2.5)), "hi95": float(np.percentile(b, 97.5))}
        print(f"[nonparam] generated - {name:<17} R@1 {d.mean():+.4f} "
              f"[{np.percentile(b, 2.5):+.4f},{np.percentile(b, 97.5):+.4f}] (paired)", flush=True)
    g = report["gt_ceiling"]["rprec"]
    print(f"[nonparam] gt-ceiling R@1 {g['1']['mean']:.4f} R@3 {g['3']['mean']:.4f} | chance {report['chance_R1']:.4f}", flush=True)
    # the retrieval picks are the evidence for what the baseline actually served
    report["picks_sample"] = {k: picks[k] for k in list(picks)[:20]}
    report["picks_same_clip_fraction"] = float(np.mean([picks[k]["text"] == picks[k]["random"] for k in picks])) if picks else 0.0
    Path(a.out).write_text(json.dumps(report, indent=1))
    print(f"[nonparam] -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
