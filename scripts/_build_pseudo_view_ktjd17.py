"""Pseudo-motion training view for TEST-TIME SELF-ADAPTATION on an unseen rig (zero-shot item 4, user 2026-09-07).

No real motion of the target rig is used AS A TRAINING TARGET (the served statistics may still depend on the rig's clips;
see "which statistics" below). For every usable clip of ONE TrueBones rig in a derived KTJD-17 view, the
frozen backbone generates a motion for that clip's caption under the deployment sampler (rest-frame demo, 20 Euler steps,
cfg 2), exactly as the zero-shot renders do (scripts/v2_render_incontext.py corpus-swap path; the sampling block below is
the one of scripts/_gen_ktjd17_clips.py). The samples are de-normalised to raw KTJD-17 units, optionally smoothed along
time (a temporal Gaussian on channels 0:12 -- positions, rest-delta rotations, velocities; contact / smooth-root /
heading untouched), and written as a SIBLING VIEW <dst> of <src>: every entry symlinked, except

  motions/            real directory: the rig's clips are the pseudo motions, every other clip a symlink to the original
  manifests/clips.jsonl  rows copied; pseudo rows get the new motion_sha256 / T_target and a `pseudo` record
  derivation.json     copied, derived_manifest_sha256 re-pinned, plus a `pseudo_generation` record
  pseudo_generation.json  ckpt / stats / sampler / smoothing / per-clip provenance

Splits, rig table, texts, caption cache, joint semantics and exclusion cuts are untouched (same clip ids), so the
per-rig LoRA launcher applies with KTJD_ROOT pointed here. The gamma CALIBRATION does NOT carry over: the trainer binds
it to the view's manifest sha (codex r1 BLOCKING 2), so measure a new artifact on this view with
scripts/_measure_ktjd17_gamma_calibration_view.py (same batch / demo / stats) and pass it as CALIB. heading_valid of a
pseudo clip is DERIVED from its generated rotations by the codec rule (carrier forward direction, schema eps_h), and
invalid frames get the heading sentinel 0 (codex r1 5). Manifest rows of pseudo clips carry the model-generated
provenance in the standard fields and keep the source declarations under `original_source` (codex r1 6).

WHICH STATISTICS MAKE IT "ZERO-MOTION" (codex r2): the served stats file decides the protocol. With the rig's OWN per-cell
stats (data/tb_norm_stats_v2_mainbody.npz, cohort = all of the rig's real clips) this is TRANSDUCTIVE adaptation: no real
motion is used as a training target, but the normalisation, sampler masks and de-normalisation carry the rig's real
statistics. The closest thing to a zero-real-motion arm uses BORROWED statistics (scripts/_build_borrowed_rig_stats.py
served through its sibling view, scripts/_derive_view_with_stats.py) -- pass that sibling view as --src and its stats as
--percell_stats. Even then the borrowed artifact keeps the rig's OWN supervise/constant masks and the mean/std of its
constant and unsupervised cells (an "oracle masks" arm, codex r3): no real motion is a training target, but which cells are
held constant, and at what value, still comes from the rig's clips. `pseudo_generation.stats_protocol` states which arm this
view is and what of the rig's real data it still depends on. A LoRA trained on this view sees the model's own outputs as targets: the flow term distils them
(smoothed or not), the gamma_fk term is FK(pred rotations) vs predicted positions -- self-consistency -- and the
dynamics terms anchor velocities to the pseudo targets. Render / evaluate the adapted model on the ORIGINAL view.

usage (inside a GPU alloc):
  python scripts/_build_pseudo_view_ktjd17.py --ckpt runs/lora_tb_init_run12_best_snapshot.pt --rig Buffalo \
     --src dataset/ktjd17_truebones_lora_v2_mainbody --dst dataset/ktjd17_truebones_lora_v2_mainbody_pseudo_buffalo_raw \
     --percell_stats data/tb_norm_stats_v2_mainbody.npz --exclude_clips configs/tb_lora_Buffalo_only_exclusions.json \
     [--smooth_sigma 0] [--splits train,val] [--steps 20 --cfg_text 2.0 --seed 7]
"""
from __future__ import annotations
import argparse, hashlib, json, os, sys
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.incontext_pairs import InContextPairs, collate                     # noqa: E402
from src.data.ktjd17_incontext import Ktjd17Base, ktjd17_split_names, _STD_FLOOR  # noqa: E402
from src.models.v2.dit_motion import InContextMotionDiT, sample                  # noqa: E402
from src.data.ktjd17.codec import decode_column_cont6d, encode_column_cont6d, heading_from_global_rotation  # noqa: E402
from src.data.caption_keys import ordered_captions                               # noqa: E402

SMOOTH_CH = list(range(0, 12))          # positions, rotations, velocities; never contact/smooth-root/heading


def sha256_file(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def gaussian_smooth_time(x: np.ndarray, sigma: float) -> np.ndarray:
    """x [T,J,C] -> same, centred Gaussian along T with edge replication (no phase shift)."""
    if sigma <= 0:
        return x
    r = int(np.ceil(3 * sigma))
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2); k /= k.sum()
    pad = np.concatenate([np.repeat(x[:1], r, 0), x, np.repeat(x[-1:], r, 0)], 0)
    out = np.zeros_like(x)
    for i, w in enumerate(k):
        out += w * pad[i:i + x.shape[0]]
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--rig", required=True)
    ap.add_argument("--src", required=True, help="derived KTJD-17 view whose clips (captions, lengths) define the pseudo set")
    ap.add_argument("--dst", required=True, help="sibling view to create")
    ap.add_argument("--percell_stats", required=True, help="stats the backbone is served with here (own or borrowed)")
    ap.add_argument("--exclude_clips", required=True, help="the rig-only cut declared by the view")
    ap.add_argument("--joint_sem", default="data/joint_semantics_llm2vec_ktjd17_v1_mainbody.npz")
    ap.add_argument("--caption_cache", default="data/tb_caption_llm2vec_pzstyle_v1")
    ap.add_argument("--texts_json", default="data/tb_motion_texts_pzstyle_v1.json")
    ap.add_argument("--splits", default="train,val", help="which splits of the rig to replace with pseudo motions")
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--cfg_text", type=float, default=2.0)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--smooth_sigma", type=float, default=0.0, help="temporal Gaussian sigma in frames on channels 0:12 (0 = off)")
    a = ap.parse_args()
    src, dst = Path(a.src), Path(a.dst)
    if not (src / "derivation.json").is_file():
        raise SystemExit(f"[refuse] {src} is not a derived view")
    if dst.exists():
        raise SystemExit(f"[refuse] {dst} exists -- remove it yourself if you mean to rebuild")
    if src.resolve().parent != dst.resolve().parent or src.resolve() == dst.resolve():
        raise SystemExit("[refuse] dst must be a sibling directory of src")
    if a.steps < 1:
        raise SystemExit("[refuse] steps must be >= 1")
    splits = [s for s in a.splits.split(",") if s]
    if not splits or any(s not in ("train", "val") for s in splits):
        raise SystemExit(f"[refuse] --splits must be a subset of train,val, got {a.splits!r}")

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    ckpt_sha = sha256_file(Path(a.ckpt))
    ca = ck["args"]
    if str(ca.get("corpus")) != "ktjd17" or bool(ca.get("two_stage", False)):
        raise SystemExit("[refuse] expects a single-stage KTJD-17 ckpt")
    if str(ca.get("rep_norm", "percell")) != "percell":
        # the protocol record below describes per-cell statistics; a scale_only backbone serves zero means and s_rig scaling
        # instead, which would make that record false (codex r4) -- restrict rather than mislabel
        raise SystemExit(f"[refuse] this tool is written for per-cell-normalised backbones; ckpt rep_norm={ca.get('rep_norm')!r}")
    model = InContextMotionDiT(in_ch=17, dim=ca["dim"], depth=ca["depth"], n_heads=ca["heads"],
                               d_text=4096, d_joint_sem=4096,
                               use_struct_feats=bool(ca.get("struct_feats", False)),
                               use_dir_bias=bool(ca.get("dir_bias", False)),
                               qk_norm=bool(ca.get("qk_norm", False)),
                               use_ref_text=bool(ca.get("ref_text", False))).to(dev)
    model.load_state_dict(ck["model"]); model.eval()
    # ZERO-SHOT corpus swap by construction: the backbone was trained elsewhere; no pin check against the ckpt here
    # (the swap is the point). The view's own derivation pins are still enforced by Ktjd17Base.
    base = Ktjd17Base(str(src), caption_emb_cache=a.caption_cache, joint_semantics=a.joint_sem, texts_json=a.texts_json,
                      percell_stats=a.percell_stats, exclude_clips=a.exclude_clips,
                      normalization=str(ca.get("rep_norm", "percell")))
    names = ktjd17_split_names(str(src), exclude=a.exclude_clips)
    targets = sorted(set(sum([list(names[s]) for s in splits], [])))
    PK = dict(demo_rest=bool(ca.get("demo_rest", False)), emit_ref_text=bool(ca.get("ref_text", False)),
              demo_frames=int(ca.get("demo_frames", 1)), target_frames=int(ca["target_frames"]),
              emit_graph_v2=bool(ca.get("struct_feats", False)) or bool(ca.get("dir_bias", False)))
    if not PK["demo_rest"]:
        raise SystemExit("[refuse] this tool assumes a rest-frame-demo backbone (a motion demo would need real clips of the rig)")
    # InContextPairs rejects overlapping target/demo lists unless the SAME object is passed (self-demo bucket); the
    # train targets therefore get the train list itself as the demo pool, the val targets the usual train pool (codex r1 BLOCKING 1)
    train_names = list(names["train"]); val_names = list(names["val"])
    dsets = []
    if "train" in splits:
        dsets.append(("train", InContextPairs(base, train_names, train_names, object_types=None, balance_skeletons=False, seed=a.seed, **PK)))
    if "val" in splits:
        dsets.append(("val", InContextPairs(base, val_names, train_names, object_types=None, balance_skeletons=False, seed=a.seed, **PK)))
    work = [(split, ds, i) for split, ds in dsets for i, (ot, _) in enumerate(ds.index) if ot == a.rig]
    if not work:
        raise SystemExit(f"[refuse] no targets of rig {a.rig} in splits {splits} under cut {a.exclude_clips}")
    anc_mode = str(ca.get("anchor", "none"))
    df = PK["demo_frames"]
    mu_r, sd_r = base._stats(a.rig)
    floor = float(_STD_FLOOR)
    sk = np.load(src / "skeletons" / f"{a.rig}.npz", allow_pickle=True)
    R_rest = np.asarray(sk["R_rest_global"], np.float64)                 # [J,3,3]; global = rest_delta @ R_rest
    carrier = int(sk["heading_carrier_joint"]); u_fwd = np.asarray(sk["u_forward_local"], np.float64)
    eps_h = float(json.loads((src / "schema.json").read_text())["heading"]["eps_h"])
    texts = json.loads(Path(a.texts_json).read_text())

    # ---- the sibling view skeleton: symlink everything, then materialise motions/ and manifests/ ----
    dst.mkdir()
    rel = os.path.relpath(src.resolve(), dst.resolve())
    for entry in sorted(src.iterdir()):
        if entry.name in ("derivation.json", "manifests", "motions", "pseudo_generation.json"):   # reserved outputs are never linked (codex r2 NIT)
            continue
        os.symlink(os.path.join(rel, entry.name), dst / entry.name)
    (dst / "motions").mkdir(); (dst / "manifests").mkdir()
    rel_m = os.path.relpath((src / "motions").resolve(), (dst / "motions").resolve())   # links live INSIDE dst/motions (codex r1 3)
    for f in sorted((src / "motions").iterdir()):
        os.symlink(os.path.join(rel_m, f.name), dst / "motions" / f.name)

    rows = [json.loads(l) for l in open(src / "manifests" / "clips.jsonl")]
    by_clip = {str(r["clip_id"]): r for r in rows}
    made = {}
    for n, (split, ds, p) in enumerate(work):
        ds._wrng_key = None
        item = ds[p]
        cid = str(item["motion_id"])
        row = by_clip.get(cid)
        if row is None or str(row.get("rig_id")) != a.rig:
            raise SystemExit(f"[refuse] target {cid} is not a manifest clip of {a.rig}")
        b = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in collate([item]).items()}
        x_in = b["x"][..., :17].contiguous()
        g2kw = {k: b[k] for k in ("struct_feats", "updown") if k in b}
        cvj = torch.from_numpy(base.static_masks(a.rig)["channel_valid"]).to(dev)
        cv = torch.zeros(1, x_in.shape[2], 17, dtype=torch.bool, device=dev); cv[0, :cvj.shape[0]] = cvj
        g2kw["channel_valid"] = cv
        g2kw["heading_valid"] = b["x"][:, :, 0, 17] > 0.5
        if "demo_text" in b:
            g2kw["demo_text"] = b["demo_text"]
        if anc_mode != "none":
            from scripts.train_v2_incontext import ktjd_anchor
            lut = {a.rig: torch.from_numpy(base.rest_anchor_frame(a.rig))} if anc_mode == "rest" else None
            g2kw["anchor"] = ktjd_anchor(b, x_in, anc_mode, lut, df)
        torch.manual_seed(a.seed + n)                    # one seed per clip; recorded per clip below
        with torch.no_grad():
            gen = sample(model, x_in, b["is_target"], a.steps, cfg_text=a.cfg_text, demo_frames=df,
                         joint_bias=b["joint_bias"], frame_valid=b["frame_valid"],
                         joint_valid=b["joint_valid"], text=b["text"], joint_sem=b["joint_sem"], **g2kw)
        J = int(item["n_joints"]); T = int(item["frame_valid"][df:].sum())
        arr = gen[0, df:df + T, :J, :].float().cpu().numpy()                       # [T,J,17] normalised
        raw = (arr.astype(np.float64) * (sd_r[:J][None] + floor) + mu_r[:J][None])   # KTJD units
        n_revert = 0
        if a.smooth_sigma > 0:
            sm = gaussian_smooth_time(raw[..., SMOOTH_CH], a.smooth_sigma)
            # 6D columns: re-orthonormalise the smoothed pair (Gram-Schmidt, the codec's own decode); a cell whose smoothed
            # columns are degenerate keeps its unsmoothed value (codex r1 4: averaging valid rotations can cancel a column)
            d6 = sm[..., 3:9]
            n1 = np.linalg.norm(d6[..., :3], axis=-1); b1 = d6[..., :3] / np.maximum(n1, 1e-12)[..., None]
            u2 = d6[..., 3:] - np.sum(b1 * d6[..., 3:], axis=-1, keepdims=True) * b1
            n2 = np.linalg.norm(u2, axis=-1)
            ok = (n1 > 1e-3) & (n2 > 1e-3)
            fixed = raw[..., 3:9].copy()
            fixed[ok] = encode_column_cont6d(decode_column_cont6d(d6[ok], strict=False))
            n_revert = int((~ok).sum())
            sm[..., 3:9] = fixed
            raw[..., SMOOTH_CH] = sm
        # the codec's strict decode is what the loader applies to every clip: fail here, loudly, not in training
        delta = decode_column_cont6d(raw[..., 3:9], strict=True)                     # [T,J,3,3]
        R_global = np.matmul(delta, R_rest[None, :J])
        heading, hvalid, _ = heading_from_global_rotation(R_global, carrier_joint=carrier, u_forward_local=u_fwd, eps_h=eps_h)
        raw[:, 0, 15:17] = 0.0
        raw[hvalid, 0, 15:17] = heading[hvalid]          # generated heading channels are replaced by the rule-derived ones
        if not np.isfinite(raw).all():
            raise SystemExit(f"[refuse] non-finite pseudo motion for {cid}")
        relp = str(row["motion_relpath"])
        orig = np.load(src / relp, allow_pickle=True)
        out_p = dst / relp
        if out_p.is_symlink():
            out_p.unlink()
        np.savez(out_p, motion=raw.astype(np.float32), heading_valid=np.asarray(hvalid, dtype=bool),
                 clip_id=np.array(cid), rig_id=np.array(a.rig), fps_target=np.asarray(orig["fps_target"]),
                 origin_xz=np.asarray(orig["origin_xz"], dtype=np.float64))
        sha = sha256_file(out_p)
        tinfo = texts.get(cid) or texts.get(cid + ".npy") or {}
        made[cid] = {"motion_sha256": sha, "T": T, "J": J, "seed": a.seed + n, "split": split,
                     "heading_valid_fraction": float(np.mean(hvalid)), "smoothing_cells_reverted": n_revert,
                     "captions": ordered_captions(tinfo) if tinfo else [], "original_motion_sha256": row.get("motion_sha256")}
        print(f"[pseudo] {a.rig} {cid} T={T} J={J} split={split} heading_valid={np.mean(hvalid):.2f} reverted={n_revert}", flush=True)

    # ---- manifest + derivation ----
    new_rows = []
    for r in rows:
        r2 = dict(r)
        cid = str(r["clip_id"])
        if cid in made:
            r2["original_source"] = {k: r.get(k) for k in ("motion_sha256", "T_target", "T_src", "rotation_authority", "audit_role",
                                                            "resample_mode", "fps_src") if k in r}
            r2["motion_sha256"] = made[cid]["motion_sha256"]
            r2["T_target"] = made[cid]["T"]
            r2["rotation_authority"] = "model_generated_rest_delta_6d"
            r2["audit_role"] = "pseudo_backbone_sample"
            r2["resample_mode"] = "none_generated_at_fps_target"
            r2["pseudo"] = {"kind": "backbone_sample_as_target", "ckpt_sha256": ckpt_sha, "seed": made[cid]["seed"],
                            "smooth_sigma": a.smooth_sigma, "heading_valid_fraction": made[cid]["heading_valid_fraction"]}
        new_rows.append(r2)
    man = dst / "manifests" / "clips.jsonl"
    man.write_text("".join(json.dumps(r) + "\n" for r in new_rows))
    deriv = json.loads((src / "derivation.json").read_text())
    deriv["derived_manifest_sha256"] = sha256_file(man)
    with np.load(a.percell_stats, allow_pickle=False) as z:
        stats_meta = json.loads(str(z["__meta"]))
    borrowed = stats_meta.get("borrowed_from") or {}
    if a.rig in borrowed:
        stats_protocol = {"kind": "borrowed_library_statistics_with_oracle_target_masks", "donor": borrowed[a.rig],
                          "real_motion_of_rig_used": "supervise/constant masks and the mean/std of constant and unsupervised cells only "
                                                     "(from the rig's own clips); all active cells' mean/std are the donor's",
                          "note": "no real motion of the rig is a training target; normalisation of active cells comes from a library "
                                  "species; the rig's own clips still decide which cells are constant and at what value (codex r3)"}
    else:
        stats_protocol = {"kind": "own_rig_statistics", "cohort": stats_meta.get("cohort"), "real_motion_of_rig_used": "statistics only",
                          "note": "TRANSDUCTIVE adaptation: no real motion is a training target, but normalisation / sampler masks / "
                                  "de-normalisation carry the rig's real per-cell statistics (codex r2)"}
    deriv["pseudo_generation"] = {"kind": "test-time self-adaptation view (scripts/_build_pseudo_view_ktjd17.py)",
                                  "stats_protocol": stats_protocol,
                                  "source_view": str(src), "source_derivation_sha256": sha256_file(src / "derivation.json"),
                                  "rig": a.rig, "n_pseudo": len(made), "splits": splits, "ckpt": a.ckpt, "ckpt_sha256": ckpt_sha,
                                  "ckpt_epoch": int(ck.get("epoch", -1)), "percell_stats": a.percell_stats,
                                  "percell_sha256": sha256_file(Path(a.percell_stats)), "steps": a.steps, "cfg_text": a.cfg_text,
                                  "seed": a.seed, "smooth_sigma": a.smooth_sigma, "smoothed_channels": SMOOTH_CH if a.smooth_sigma > 0 else [],
                                  "calibration_required": "the gamma calibration is bound to manifest_sha256/train_ids: measure a new artifact "
                                                          "on THIS view (scripts/_measure_ktjd17_gamma_calibration_view.py) before training",
                                  "note": "the rig's clip ids keep their captions/splits; their motions are the backbone's own zero-shot "
                                          "samples (no real motion of the rig is served). Train adapters here; render/evaluate on the source view."}
    (dst / "derivation.json").write_text(json.dumps(deriv, indent=1))
    (dst / "pseudo_generation.json").write_text(json.dumps({**deriv["pseudo_generation"], "clips": made}, indent=1))
    # self-check: the view must load with the same stats/cut and serve the pseudo rows
    chk = Ktjd17Base(str(dst), caption_emb_cache=a.caption_cache, joint_semantics=a.joint_sem, texts_json=a.texts_json,
                     percell_stats=a.percell_stats, exclude_clips=a.exclude_clips, normalization=str(ca.get("rep_norm", "percell")))
    served = {str(r["clip_id"]) for r in chk._rows if str(r.get("rig_id")) == a.rig}
    if not set(made) <= served:
        raise SystemExit(f"[refuse] self-check: {len(set(made) - served)} pseudo clips are not served by the new view")
    # and every pseudo clip must LOAD through the strict codec path the trainer uses (codex r1 4)
    n_loaded = 0
    for r in chk._rows:
        if str(r["clip_id"]) in made:
            it = chk[chk._rows.index(r)]                                   # strict codec decode happens inside
            if not np.isfinite(np.asarray(it["anytop_x"], dtype=np.float32)).all():
                raise SystemExit(f"[refuse] self-check: non-finite tensors when loading pseudo clip {r['clip_id']}")
            n_loaded += 1
    if n_loaded != len(made):
        raise SystemExit(f"[refuse] self-check loaded {n_loaded} of {len(made)} pseudo clips")
    print(f"[OK] {dst}: {len(made)} pseudo clips for {a.rig} (splits {splits}, smooth_sigma {a.smooth_sigma}); "
          f"manifest {deriv['derived_manifest_sha256'][:16]}")


if __name__ == "__main__":
    main()
