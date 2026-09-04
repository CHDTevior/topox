#!/usr/bin/env python3
"""Render 4-panel skeleton GIFs from an in-context DiT checkpoint.

Layout per clip (GT stays the RED rightmost panel, per the established convention):
    DEMO (teal, ric) | GEN pos (orange, ric) | GEN fk (purple, rot6d-FK) | TARGET GT (red, ric)
The GENERATED motion is drawn under BOTH recoveries on purpose: H4 measured 4-19% disagreement
between the position family and the rotation family on generated output, and two panels make that
disagreement visible instead of hiding it behind whichever family we pick. GT needs one panel only
(the families agree to 0.000% on real data). One shared scale across all four panels.
True-speed playback: every real frame at the CORPUS's own rate (13ch 20 fps, KTJD-17 30 fps),
no subsampling.

Read-only w.r.t. training state; writes GIFs + summary.txt (jitter for both recoveries).
"""
import argparse, hashlib, json, pickle, sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.anytop_dataset import (AnyTopDataset, _STD_FLOOR,                 # noqa: E402
                                     _recover_world_positions)
from src.data.anytop_rot6d_fk import recover_from_bvh_rot_np                     # noqa: E402
from src.data.incontext_pairs import (InContextPairs, collate, read_split,      # noqa: E402
                                      truebones_types, pzh_types, DEMO_FRAMES, TARGET_FRAMES)
from src.models.v2.dit_motion import InContextMotionDiT, sample                  # noqa: E402

PANEL_W, PANEL_H, GAP, FOOT = 300, 360, 12, 66
# The GENERATED motion is drawn under BOTH recoveries: H4 measured a 4-19% disagreement between
# the position family (RIC) and the rotation family (FK) on generated output -- two panels make
# that disagreement visible to the eye instead of hiding it behind whichever family we pick.
# GT needs one panel only (the two families agree to 0.000% on real data).
COLS = {"demo": (13, 110, 100), "gen_ric": (176, 61, 8),
        "gen_fk": (91, 44, 184), "gt": (185, 28, 28)}


def _sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def world_of(norm_txjc, mean, std, recover="ric", parents=None, offsets=None):
    """World positions via either channel family. The two disagree exactly where the model is
    inconsistent, so rendering BOTH localises noise: ric reads per-frame positions (ch0:3),
    fk rebuilds them from rotations (ch3:9) over the bone chain.
    """
    if norm_txjc.shape[-1] != 13:
        # S0c contract guard: the legacy AnyTop/RIFKE recovery must never consume KTJD tensors
        # (17/18 channels) -- that path would silently misread the channel layout.
        raise ValueError(f"legacy world_of expects 13 channels, got {norm_txjc.shape[-1]} -- "
                         f"KTJD tensors must go through world_of_ktjd")
    raw = (norm_txjc * (std[None] + _STD_FLOOR) + mean[None]).astype(np.float64)
    if recover == "fk":
        return recover_from_bvh_rot_np(raw, parents, offsets)        # [T,J,3]
    return _recover_world_positions(raw)                             # [T,J,3]


def world_of_ktjd(norm_seg18, base, rig, strict_gt):
    """KTJD-17 world positions via the OFFICIAL codec, both paths at once.

    norm_seg18 [T,J,18] normalized (plane 17 = heading flag, sliced off); de-normalized with the
    adapter's exact std trick, then decode_ktjd17 (direct + FK, NO temporal integration --
    velocity channels are never integrated, per the KTJD contract). strict_gt=False for model
    output (degenerate predicted 6D tolerated and reported by the codec).
    Returns (direct [T,J,3], fk [T,J,3]).
    """
    from src.data.ktjd17.decoder import decode_ktjd17
    sk = base._skeleton(rig)
    J = norm_seg18.shape[1]
    # per-cell mean/std (2026-08-21): the served/predicted tensor is standardized per
    # (rig, joint, channel), so de-normalization is the repo convention x*(std+floor)+mean with
    # BOTH taken from the stats artifact (codex round-S7 blocker 2: this used the superseded
    # scale-only path and called a function that no longer exists).
    mu_r, sd_r = base._pc[rig]
    mean, std = mu_r[:J, :17], sd_r[:J, :17]
    raw = (norm_seg18[..., :17] * (std[None] + _STD_FLOOR) + mean[None]).astype(np.float64)
    dec = decode_ktjd17(raw, parents=sk["parents"], R_rest_global=sk["R_rest_global"],
                        R_rest_local=sk["R_rest_local"],
                        offset_parent_local=sk["offset_parent_local"],
                        rotation_source_kind=sk["rotation_source_kind"], strict_gt=strict_gt)
    return dec.positions_direct, dec.positions_fk


def one_euro(x, fps=20.0, min_cutoff=1.5, beta=0.3, d_cutoff=1.0):
    """One-euro filter along axis 0 of x [T, ...] (Casiez et al. 2012), the game-industry
    standard adaptive low-pass: strong smoothing at low speeds (kills uniform micro-jitter on
    near-static joints -- the measured failure mode, see jitter_analysis_s32.txt), weak at high
    speeds (bursts stay sharp). Pure post-process on the MODEL OUTPUT channels; deployment
    candidate (user 2026-08-28 plan b)."""
    import numpy as _np
    def alpha(cutoff):
        tau = 1.0 / (2 * _np.pi * cutoff)
        te = 1.0 / fps
        return 1.0 / (1.0 + tau / te)
    y = x.copy()
    dx_prev = _np.zeros_like(x[0])
    for t in range(1, x.shape[0]):
        dx = (x[t] - y[t - 1]) * fps
        a_d = alpha(d_cutoff)
        dx_hat = a_d * dx + (1 - a_d) * dx_prev
        cutoff = min_cutoff + beta * _np.abs(dx_hat)
        a = alpha(cutoff)
        y[t] = a * x[t] + (1 - a) * y[t - 1]
        dx_prev = dx_hat
    return y


def jitter_ratio(gen_w, gt_w):
    """Second-difference acceleration ratio gen/GT -- the milestone-comparable jitter number.
    Reported for all joints and for the root row separately (root mixes in velocity-integration
    noise). GT is the natural denominator: ~1.0 means as smooth as real motion."""
    def acc(w):
        if w.shape[0] < 3:
            return None
        a = w[2:] - 2 * w[1:-1] + w[:-2]
        return np.linalg.norm(a, axis=-1)
    ga, ta = acc(gen_w), acc(gt_w)
    if ga is None or ta is None:
        return float("nan"), float("nan"), float("nan"), float("nan")
    # the raw GT denominators ride along: a near-static GT makes the ratio arbitrarily large,
    # so the ratio is only interpretable next to its denominator.
    gt_all, gt_root = float(ta.mean()), float(ta[:, 0].mean())
    allr = float(ga.mean() / max(gt_all, 1e-9))
    rootr = float(ga[:, 0].mean() / max(gt_root, 1e-9))
    return allr, rootr, gt_all, gt_root


def draw_panel(img, xy, parents, col, x0):
    d = ImageDraw.Draw(img)
    for j, p in enumerate(parents):
        if p < 0:
            continue
        d.line([x0 + xy[j, 0], xy[j, 1], x0 + xy[p, 0], xy[p, 1]], fill=col, width=3)
    for j in range(len(parents)):
        r = 4 if j == 0 else 2
        d.ellipse([x0 + xy[j, 0] - r, xy[j, 1] - r, x0 + xy[j, 0] + r, xy[j, 1] + r], fill=col)


def project(seqs, yaw_deg=28.0):
    """Shared orthographic projection: x' = x cosA + z sinA, y' = y (up). One bbox over ALL
    sequences so panels share scale; returns (list of [T,J,2] pixel coords, ground_row_px).

    The bbox is FORCED to include world y=0: with a pure-yaw projection the whole ground plane
    maps to one horizontal pixel row, so returning that row lets the caller draw a ground line --
    the reference the eye needs to see root-height drift ("每个物种有点飘", user 2026-08-20).
    The line is the ABSOLUTE world y=0 reference, not a promise that GT touches it: recovered
    rest-pose minima vary by rig (Pteranodon +1.31 aloft, Dragon -0.25 below zero; codex probe),
    so read it as a fixed altitude datum shared by all four panels -- GEN drifting relative to
    the GT panel's height above the line is the signal."""
    a = np.radians(yaw_deg)
    flat = [np.stack([s[..., 0] * np.cos(a) + s[..., 2] * np.sin(a), s[..., 1]], axis=-1)
            for s in seqs]
    allp = np.concatenate([f.reshape(-1, 2) for f in flat], axis=0)
    lo, hi = allp.min(0), allp.max(0)
    lo[1] = min(lo[1], 0.0)            # ground always in frame
    hi[1] = max(hi[1], 0.0)
    span = float(max(hi[0] - lo[0], hi[1] - lo[1], 1e-6))
    sc = (min(PANEL_W, PANEL_H) - 46) / span
    out = []
    for f in flat:
        px = (f[..., 0] - (lo[0] + hi[0]) / 2) * sc + PANEL_W / 2
        py = PANEL_H - ((f[..., 1] - (lo[1] + hi[1]) / 2) * sc + PANEL_H / 2)
        out.append(np.stack([px, py], axis=-1))
    ground_px = PANEL_H - ((0.0 - (lo[1] + hi[1]) / 2) * sc + PANEL_H / 2)
    return out, float(ground_px)


def render_gif(out_path, panels, parents, caption, rig, fps=20):
    """panels: list of (color_key, tag, seq [T,J,3]); shared scale across all of them."""
    names = [c for c, _, _ in panels]
    tags = [t for _, t, _ in panels]
    seqs, ground_px = project([w for _, _, w in panels])
    T = max(s.shape[0] for s in seqs)
    W = PANEL_W * len(panels) + GAP * (len(panels) - 1)
    frames = []
    for t in range(T):
        img = Image.new("RGB", (W, PANEL_H + FOOT), (246, 248, 247))
        d = ImageDraw.Draw(img)
        for k, (name, seq) in enumerate(zip(names, seqs)):
            x0 = k * (PANEL_W + GAP)
            # ground: shaded underground band + line at world y=0, drawn UNDER the skeleton
            gy = min(max(ground_px, 0), PANEL_H - 1)
            d.rectangle([x0, gy, x0 + PANEL_W - 1, PANEL_H - 1], fill=(236, 240, 238))
            d.line([x0, gy, x0 + PANEL_W - 1, gy], fill=(170, 186, 178), width=2)
            for tx in range(x0 + 10, x0 + PANEL_W - 4, 28):   # sparse ticks: reads as a floor
                d.line([tx, gy, tx - 6, gy + 5], fill=(198, 210, 202), width=1)
            d.rectangle([x0, 0, x0 + PANEL_W - 1, PANEL_H - 1], outline=(216, 223, 225))
            tt = min(t, seq.shape[0] - 1)                    # shorter panels freeze on last frame
            draw_panel(img, seq[tt], parents, COLS[name], x0)
            d.text((x0 + 8, 6), tags[k], fill=COLS[name])
            d.text((x0 + 8, PANEL_H - 18), f"f{tt+1}/{seq.shape[0]}", fill=(116, 133, 150))
        d.text((8, PANEL_H + 8), rig, fill=(15, 23, 32))
        for li, line in enumerate([caption[i:i + 96] for i in range(0, min(len(caption), 192), 96)]):
            d.text((8, PANEL_H + 26 + 16 * li), line, fill=(64, 80, 94))
        frames.append(img)
    # true speed = the CORPUS's own frame rate (13ch AnyTop 20fps, KTJD-17 30fps). Playing 30fps
    # data at 20fps is a 2/3 slow-motion that makes low-amplitude motion read as frozen.
    frames[0].save(out_path, save_all=True, append_images=frames[1:],
                   duration=round(1000.0 / fps), loop=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--rigs_A", default="Alligator,Trex")
    ap.add_argument("--rigs_B", default="BrownBear,Elephant")
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--smooth_mincutoff", type=float, default=0.0,
                    help="one-euro post-filter on the generated channels (0 = off); typical "
                         "1.5 mild / 0.8 strong at 30fps source")
    ap.add_argument("--smooth_beta", type=float, default=0.3)
    ap.add_argument("--cfg_text", type=float, default=1.0,
                    help="text-axis classifier-free guidance at inference; 1.0 = off (bare "
                         "conditional). The text-dropped branch was trained (p_drop_text 0.1), "
                         "the demo axis was NOT (p_drop_demo 0) -- so only cfg_text is exposed.")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--dump_world", action="store_true",
                    help="also write <name>.world.npz next to each gif: gen_ric / gen_fk / gt_w / demo_w world positions "
                         "[T,J,3], joint names, parents, fps -- the input of scripts/_compare_external_bvh_geometry.py")
    ap.add_argument("--corpus", choices=("truebones", "pzh", "ktjd17"), default="truebones")
    ap.add_argument("--ktjd_root", default="dataset/ktjd17_truebones")
    ap.add_argument("--demo_frames", type=int, default=DEMO_FRAMES)
    ap.add_argument("--target_frames", type=int, default=TARGET_FRAMES)
    ap.add_argument("--rigs_T", default="",
                    help="rigs rendered from the TRAIN bucket (seen rigs, SEEN clips -- the "
                         "memorisation upper bound; user 2026-08-20)")
    ap.add_argument("--caption_re", default="",
                    help="with --pick caption: python regex matched (case-insensitive) against "
                         "the target's caption; picks the FIRST matching clip per rig. The "
                         "turning-motion acceptance probe uses "
                         r"'turn|around|left|right|rotat|spin|circle'.")
    ap.add_argument("--pick", choices=("first", "energetic", "caption", "longest"),
                    default="first",
                    help="which target clip per rig: first (legacy) or the max-GT-motion-energy "
                         "one (user 2026-08-20: big actions read text-adherence best)")
    ap.add_argument("--all_targets", action="store_true",
                    help="render EVERY target clip of each requested rig (default: first only)")
    ap.add_argument("--data_root", default="data/animo4d_L4TB_plus_human_v4b272neutral")
    ap.add_argument("--splits_dir", default="data/holdout_splits_v1")
    ap.add_argument("--joint_sem", default="data/joint_semantics_llm2vec_v1.npz")
    ap.add_argument("--caption_cache", default="data/anytop_caption_llm2vec_v4b272neutral_multi")
    ap.add_argument("--texts_json", default="motion_texts_by_file_clean_v1.json")
    ap.add_argument("--allow_corpus_swap", action="store_true",
                    help="ktjd17: render a ckpt ZERO-SHOT on a corpus it was not trained on (different "
                         "generation/stats/captions). The pins drift is printed and written to "
                         "summary.txt instead of refusing; use only for deliberate OOD baselines.")
    ap.add_argument("--exclude_clips", default="",
                    help="ktjd17 + --allow_corpus_swap only: the swapped corpus' own clip-exclusion artifact "
                         "(the ckpt's cut names clips of another corpus)")
    ap.add_argument("--percell_stats", default="",
                    help="ktjd17: override the per-cell stats path stored in the ckpt (needed to run a "
                         "PZ-trained ckpt zero-shot on another corpus whose rigs the ckpt stats lack)")
    a = ap.parse_args()
    if a.exclude_clips and not a.allow_corpus_swap:
        raise SystemExit("[refuse] --exclude_clips is only honoured together with --allow_corpus_swap "
                         "(the ckpt's own cut is authoritative otherwise)")
    if a.corpus == "ktjd17" and a.joint_sem == ap.get_default("joint_sem"):
        # the legacy AnyTop table fails the KTJD order hash on the first rig (same defect codex
        # round-S0 found in the trainer); select the KTJD table when none was chosen explicitly
        a.joint_sem = "data/joint_semantics_llm2vec_ktjd17_v1.npz"
    if a.pick == "caption" and not a.caption_re:
        raise SystemExit("--pick caption requires --caption_re")
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)  # our own ckpt; contains numpy RNG state, rejected by 2.6's weights_only default
    ca = ck["args"]
    # The model width comes from the ckpt, so the CORPUS must too -- a default --corpus of
    # truebones against a KTJD ckpt would build the right width and then feed it the wrong
    # decode semantics (codex round-2). Refuse the mismatch instead of silently rendering.
    if str(ca.get("corpus", "truebones")) != a.corpus:
        raise SystemExit(f"[refuse] ckpt was trained on corpus {ca.get('corpus')!r} but "
                         f"--corpus is {a.corpus!r}; pass the matching corpus")
    mkw = dict(in_ch=17 if ca.get("corpus") == "ktjd17" else 13,
               dim=ca["dim"], depth=ca["depth"], n_heads=ca["heads"],
               d_text=4096, d_joint_sem=4096,
               use_struct_feats=bool(ca.get("struct_feats", False)),
               use_dir_bias=bool(ca.get("dir_bias", False)),
               qk_norm=bool(ca.get("qk_norm", False)),
               use_ref_text=bool(ca.get("ref_text", False)))
    if bool(ca.get("two_stage", False)):
        from src.models.v2.dit_motion import TwoStageInContextDiT
        model = TwoStageInContextDiT(root_dim=int(ca.get("root_dim", 192)), root_depth=4,
                                     **mkw).to(dev)
    else:
        model = InContextMotionDiT(**mkw).to(dev)
    model.load_state_dict(ck["model"]); model.eval()
    ep = ck.get("epoch", -1)
    print(f"[render] ckpt {a.ckpt} (epoch {ep}) on {dev}", flush=True)
    # --dump_world provenance (codex 2026-09-04): a dump must say which checkpoint produced it
    dump_ckpt_sha = _sha256_file(a.ckpt) if a.dump_world else ""

    corpus_swap_note = ""
    eff_pc, eff_excl = "", None          # effective per-cell stats / clip cut actually used (dumped with hashes)
    if a.corpus == "ktjd17":
        from src.data.ktjd17_incontext import Ktjd17Base, ktjd17_split_names
        pins_ck = ck.get("ktjd_pins") or {}
        # The CHECKPOINT names the data it was trained on. Unless this is an explicit zero-shot
        # corpus swap, every data argument is back-filled from the ckpt BEFORE the loader is built
        # (codex 2026-09-02 round 4: filling ktjd_root afterwards built a LoRA ckpt's PARENT view
        # and its derived-view pins were then skipped by the key-intersection drift check).
        if not a.allow_corpus_swap:
            for k_ in ("ktjd_root", "joint_sem", "caption_cache", "texts_json"):
                if k_ in ca and getattr(a, k_) != ca[k_]:
                    print(f"[render] {k_}: {getattr(a, k_)!r} -> {ca[k_]!r} (from ckpt)", flush=True)
                    setattr(a, k_, ca[k_])
        # the EFFECTIVE stats and cut, resolved ONCE and dumped as such (codex 2026-09-04 round 2: the dump used
        # to record the CLI value, which is empty for a LoRA ckpt whose cut comes from the checkpoint). The cut
        # comes from the CKPT, never a render-time default: rendering a model against data it was not trained
        # on is the drift this refuses -- except for an explicit zero-shot corpus swap, whose own cut applies.
        eff_pc = a.percell_stats or ca.get("ktjd_percell_stats", "data/ktjd17_percell_stats_v1.npz")
        # swap: the CLI cut verbatim (none given == no cut) -- a ckpt's cut names clips of ANOTHER corpus and must
        # never leak into a swap (codex 2026-09-04 round 3); no swap: the ckpt's own cut
        eff_excl = (a.exclude_clips or None) if a.allow_corpus_swap else (ca.get("exclude_clips") or None)
        base = Ktjd17Base(a.ktjd_root, caption_emb_cache=a.caption_cache,
                          joint_semantics=a.joint_sem, texts_json=a.texts_json,
                          percell_stats=eff_pc, exclude_clips=eff_excl)
        # `exclusion` is NOT in base.provenance -- it lives on base.provenance_exclusion -- so a
        # key-intersection drift check silently misses a swapped cut file at the same path, which
        # changes render/eval membership under an unchanged checkpoint (codex 2026-08-21 (A)4).
        live = {**base.provenance, "exclusion": base.provenance_exclusion}
        # BIDIRECTIONAL over the DATA pins: a data pin the ckpt carries but the live corpus lacks
        # (a derived view's manifest/derivation sha rendered against its parent) is drift, not a
        # skip. Objective pins the trainer adds to ktjd_pins (gammas, gamma_calib_*, group_spec,
        # gate_override, in_ch, ...) describe the run, not the data, and are not compared here.
        data_keys = set(live) | {"manifest_sha256", "derivation_sha256"}
        drift = sorted(k for k, v in pins_ck.items()
                       if k in data_keys and (k not in live or live[k] != v))
        if drift and a.allow_corpus_swap:
            corpus_swap_note = (f"ZERO-SHOT CORPUS SWAP: ckpt pins differ on {drift}; rendered on "
                                f"{a.ktjd_root} with stats {a.percell_stats or 'ckpt'} by explicit request")
            print(f"[render] {corpus_swap_note}", flush=True)
        elif drift:
            raise SystemExit(f"[refuse] render-time data does not match what this checkpoint was "
                             f"trained on: {drift}. Rendering a model against replaced artifacts "
                             f"produces a picture of nothing (pass --allow_corpus_swap for a "
                             f"deliberate zero-shot baseline on another corpus).")
        names = ktjd17_split_names(a.ktjd_root, exclude=eff_excl)
        tb = None
    else:
        cond = pickle.load(open(f"{a.data_root}/_cond_normalized_J144.pkl", "rb"))
        tb = truebones_types(cond.keys()) if a.corpus == "truebones" else pzh_types(cond.keys())
        names = {k: read_split(a.splits_dir, k) for k in ("train", "val", "held_representative")}
        base = AnyTopDataset(data_root=a.data_root, split="all", num_frames=300, max_joints=144,
                             load_captions=True, caption_emb_cache=a.caption_cache,
                             random_caption=False, augment=False, joint_semantics=a.joint_sem,
                             species_whitelist=tb, splits_dir=a.splits_dir,
                             texts_json_name=a.texts_json)
    # rest-demo runs must be rendered the way they were trained, and the checkpoint is the
    # authority -- deriving it from the ckpt stops a silent 64-frame-demo render of a 1-frame arm.
    # EVERY window-defining knob comes from the CHECKPOINT (codex round-S8 blocker 2): rendering a
    # different target window, corpus root or conditioning surface than the run was trained on
    # makes the visual acceptance meaningless, and visual acceptance is this project's only
    # quality gate.
    for k_ in ("target_frames",):                  # ktjd_root is back-filled BEFORE the loader above
        if k_ in ca and getattr(a, k_) != ca[k_]:
            print(f"[render] {k_}: {getattr(a, k_)!r} -> {ca[k_]!r} (from ckpt)", flush=True)
            setattr(a, k_, ca[k_])
    ck_rest = bool(ca.get("demo_rest", False))
    if ck_rest:
        if int(ca.get("demo_frames", 0)) != 1:
            raise SystemExit(f"[refuse] ckpt has demo_rest with demo_frames="
                             f"{ca.get('demo_frames')}; expected 1")
        a.demo_frames = 1
        print("[render] ckpt trained with --demo_rest: 1-frame rest demo", flush=True)
    PK = dict(demo_rest=ck_rest, emit_ref_text=bool(ca.get("ref_text", False)),
              demo_frames=a.demo_frames, target_frames=a.target_frames,
              emit_graph_v2=bool(ca.get("struct_feats", False)) or bool(ca.get("dir_bias", False)))
    dsA = InContextPairs(base, names["val"], names["train"], object_types=tb,
                         balance_skeletons=False, seed=a.seed, **PK)
    # the merged no-IK corpus ships only train/val; a missing held bucket renders nothing for B
    _held = names.get("held_representative", set())
    dsB = (InContextPairs(base, _held, _held, object_types=tb,
                          balance_skeletons=False, seed=a.seed, **PK) if _held else None)
    dsT = InContextPairs(base, names["train"], names["train"], object_types=tb,
                         balance_skeletons=False, seed=a.seed, **PK)

    jobs = []
    for bucket, rigs, ds in (("A", a.rigs_A, dsA), ("B", a.rigs_B, dsB), ("T", a.rigs_T, dsT)):
        if ds is None:
            continue
        for r in [x.strip() for x in rigs.split(",") if x.strip()]:
            if r not in ds.types:
                print(f"[render] SKIP {bucket}:{r} -- not in bucket "
                      f"({'A held?' if bucket == 'A' else 'train?'})", flush=True)
                continue
            positions = [i for i, (ot, _) in enumerate(ds.index) if ot == r]
            if not a.all_targets:
                if a.pick == "caption":
                    # Turning-motion acceptance probe: heading is the one axis with neither an
                    # augmentation nor a condition backing it (user 2026-08-20: no yaw aug; no
                    # c_dir), so the QA set must deliberately contain clips whose captions
                    # DESCRIBE a turn -- otherwise the known exposure goes unexamined.
                    import re
                    rx = re.compile(a.caption_re, re.I)
                    hit = [i for i in positions
                           if rx.search(str(base[ds.index[i][1]].get("caption", "")))]
                    if not hit:
                        print(f"[render] SKIP {bucket}:{r} -- no caption matches "
                              f"{a.caption_re!r}", flush=True)
                        continue
                    positions = hit[:1]
                elif a.pick == "longest" and len(positions) > 1:
                    # Short clips hide the failure modes that only appear over time (drift,
                    # accumulating jitter, a pose that is fine for a second then decays). `first`
                    # picked 28-51 frame clips (~1-1.7 s) while this corpus holds up to 374.
                    # Ranked on the GT's own length, capped by the model's target window.
                    def _len(ix):
                        return min(int(base[ds.index[ix][1]]["num_frames"]), ds.Tt)
                    positions = [max(positions, key=_len)]
                elif a.pick == "energetic" and len(positions) > 1:
                    # GT motion energy = mean frame-to-frame displacement of the DE-NORMALIZED
                    # RIC positions over the head window (codex 01a01b1a: measuring in the
                    # per-rig-normalized space distorts the ranking -- the winner changed for
                    # Horse/Anaconda/Monkey/Raptor). Physical space, GT-only, no RNG involved.
                    def _energy(ix):
                        gt_it = base[ds.index[ix][1]]
                        Jn = int(gt_it["num_joints"])
                        Tn = min(int(gt_it["num_frames"]), ds.Tt)
                        if Tn < 2:
                            return 0.0
                        xn = np.asarray(gt_it["anytop_x"])[:Jn, :, :Tn].transpose(2, 0, 1)
                        if a.corpus == "ktjd17":
                            xw, _ = world_of_ktjd(xn, base, ds.index[ix][0], strict_gt=True)
                        else:
                            mn = np.asarray(gt_it["anytop_mean"])[:Jn]
                            sd = np.asarray(gt_it["anytop_std"])[:Jn]
                            raw = (xn * (sd[None] + _STD_FLOOR) + mn[None]).astype(np.float64)
                            xw = _recover_world_positions(raw)   # WORLD, incl. root translation
                        return float(np.linalg.norm(np.diff(xw, axis=0), axis=-1).mean())
                    positions = [max(positions, key=_energy)]
                else:
                    positions = positions[:1]                 # legacy: first target
            jobs += [(bucket, r, ds, pp) for pp in positions]
    lines = []
    for bucket, rig, ds, pos in jobs:
        # Stream reset PER ITEM: each target's demo/crop draw starts from the same rng origin, so a
        # given (rig, target) renders identically across invocations and epochs regardless of how
        # many other targets were rendered before it.
        ds._wrng_key = None
        item = ds[pos]
        b = {k: (v.to(dev) if torch.is_tensor(v) else v)
             for k, v in collate([item]).items()}
        J = int(item["n_joints"])
        t_real = int(item["frame_valid"][a.demo_frames:].sum())
        d_real = int(item["frame_valid"][:a.demo_frames].sum())

        torch.manual_seed(a.seed)
        with torch.no_grad():
            g2kw = {k: b[k] for k in ("struct_feats", "updown") if k in b}
            if a.corpus == "ktjd17":
                cvj = torch.from_numpy(base.static_masks(rig)["channel_valid"]).to(dev)
                cv = torch.zeros(1, b["x"].shape[2], 17, dtype=torch.bool, device=dev)
                cv[0, :cvj.shape[0]] = cvj
                x_in = b["x"][..., :17].contiguous()
                g2kw["channel_valid"] = cv
                # demo-side heading validity (plane 17); sample() ignores it on target frames
                g2kw["heading_valid"] = b["x"][:, :, 0, 17] > 0.5
                if "demo_text" in b:
                    g2kw["demo_text"] = b["demo_text"]
                # anchored ckpts sample from anchor + N(0,I) -- rendering without the anchor
                # would sample from the WRONG base distribution
                anc_mode = str(ca.get("anchor", "none"))
                if anc_mode != "none":
                    from scripts.train_v2_incontext import ktjd_anchor
                    rest_lut = ({rig: torch.from_numpy(base.rest_anchor_frame(rig))}
                                if anc_mode == "rest" else None)
                    g2kw["anchor"] = ktjd_anchor(b, x_in, anc_mode, rest_lut, a.demo_frames)
            else:
                x_in = b["x"]
            gen = sample(model, x_in, b["is_target"], a.steps, cfg_text=a.cfg_text,
                         demo_frames=a.demo_frames,
                         joint_bias=b["joint_bias"], frame_valid=b["frame_valid"],
                         joint_valid=b["joint_valid"], text=b["text"], joint_sem=b["joint_sem"],
                         **g2kw)
        gen = gen[0].float().cpu().numpy()
        if a.smooth_mincutoff > 0:
            # smooth ONLY the target frames of the model-output channels; the demo frame is GT.
            # fps follows the corpus: KTJD-17 is 30fps, the legacy 13ch corpora are 20fps
            # (codex 2026-08-28: a hard-coded 30 would mistune the filter for those renders)
            _fps = 30.0 if a.corpus == "ktjd17" else 20.0
            gen[a.demo_frames:] = one_euro(gen[a.demo_frames:], fps=_fps,
                                           min_cutoff=a.smooth_mincutoff, beta=a.smooth_beta)
        gt = b["x"][0].float().cpu().numpy()

        t_item = base[ds.index[pos][1]]
        parents = [int(p) for p in t_item["parent_indices"][:J]]
        gseg = gen[a.demo_frames:a.demo_frames + t_real, :J]
        gtseg = gt[a.demo_frames:a.demo_frames + t_real, :J]
        if a.corpus == "ktjd17":
            demo_w, _ = world_of_ktjd(gt[:d_real, :J], base, rig, strict_gt=True)
            gen_ric, gen_fk = world_of_ktjd(gseg, base, rig, strict_gt=False)
            gt_w, _ = world_of_ktjd(gtseg, base, rig, strict_gt=True)
        else:
            mean = np.asarray(t_item["anytop_mean"])[:J]
            std = np.asarray(t_item["anytop_std"])[:J]
            offsets = np.asarray(t_item["rest_offsets"])[:J]
            demo_w = world_of(gt[:d_real, :J], mean, std)
            gen_ric = world_of(gseg, mean, std)
            gen_fk = world_of(gseg, mean, std, recover="fk", parents=parents, offsets=offsets)
            gt_w = world_of(gtseg, mean, std)
        jit_all, jit_root, gt_all, gt_root = jitter_ratio(gen_ric, gt_w)
        jfk_all, jfk_root, _, _ = jitter_ratio(gen_fk, gt_w)
        static_warn = "  [near-static GT, ratio inflated]" if gt_all < 1e-3 else ""

        cap = str(t_item.get("caption", ""))
        name = f"{bucket}_{rig}__{item['motion_id'][:48]}"
        if a.dump_world:
            _jn = [str(x) for x in base._skeleton(rig)["joint_names"]][:J] if a.corpus == "ktjd17" else [f"j{i}" for i in range(J)]
            _gen_id = (json.loads((Path(a.ktjd_root) / "generation.json").read_text())["generation_id"]
                       if a.corpus == "ktjd17" else "")
            np.savez(out / f"{name}.world.npz", gen_ric=gen_ric, gen_fk=gen_fk, gt_w=gt_w, demo_w=demo_w,
                     joint_names=np.array(_jn), parents=np.array(parents, dtype=np.int64),
                     fps=np.array(30.0 if a.corpus == "ktjd17" else 20.0), rig=np.array(rig),
                     motion_id=np.array(str(item["motion_id"])), caption=np.array(cap),
                     # provenance (codex 2026-09-04): the consumer must be able to tell WHICH model, sampler and
                     # corpus produced a dump, and refuse to pool dumps that disagree
                     dump_format=np.array("world-dump-v3"), ckpt=np.array(str(a.ckpt)),
                     ckpt_sha256=np.array(dump_ckpt_sha), epoch=np.array(int(ep)), steps=np.array(int(a.steps)),
                     cfg_text=np.array(float(a.cfg_text)), seed=np.array(int(a.seed)),
                     smooth_mincutoff=np.array(float(a.smooth_mincutoff)), smooth_beta=np.array(float(a.smooth_beta)),
                     corpus=np.array(a.corpus), ktjd_root=np.array(str(a.ktjd_root) if a.corpus == "ktjd17" else ""),
                     generation_id=np.array(_gen_id),
                     percell_stats=np.array(str(eff_pc or "")),
                     percell_sha256=np.array(_sha256_file(eff_pc) if eff_pc else ""),
                     exclude_clips=np.array(str(eff_excl or "")),
                     exclude_sha256=np.array(_sha256_file(eff_excl) if eff_excl else ""),
                     units=np.array("source-rig units of the KTJD skeleton; decode_ktjd17 direct/fk, no temporal integration"))
        render_gif(out / f"{name}.gif",
                   [("demo", f"DEMO {item['demo_id']}", demo_w),
                    ("gen_ric", f"GEN pos ep{ep} s{a.steps}", gen_ric),
                    ("gen_fk", f"GEN fk ep{ep} s{a.steps}", gen_fk),
                    ("gt", "TARGET GT", gt_w)],
                   parents, cap, f"[{bucket}] {rig}",
                   fps=30 if a.corpus == "ktjd17" else 20)
        print(f"[render] {name}.gif  (demo {d_real}f | target {t_real}f, J={J})  "
              f"jitter ric {jit_all:.2f}x fk {jfk_all:.2f}x (GT {gt_all:.4f}) "
              f"root ric {jit_root:.2f}x fk {jfk_root:.2f}x{static_warn}", flush=True)
        lines.append(f"{name}\tjitter_ric={jit_all:.3f}x jitter_fk={jfk_all:.3f}x(gt={gt_all:.5f}) "
                     f"root_ric={jit_root:.3f}x root_fk={jfk_root:.3f}x{static_warn}"
                     f"\tcaption: {cap}\tdemo={item['demo_id']} target={item['motion_id']}")
    (out / "summary.txt").write_text(
        f"ckpt={a.ckpt} epoch={ep} steps={a.steps} seed={a.seed} cfg_text={a.cfg_text} smooth_mincutoff={a.smooth_mincutoff} smooth_beta={a.smooth_beta} panels=demo|gen_ric|gen_fk|gt\n"
        + (f"{corpus_swap_note}\n" if a.corpus == "ktjd17" and corpus_swap_note else "")
        + "\n".join(lines) + "\n")
    print(f"[render] DONE -> {out}", flush=True)


if __name__ == "__main__":
    main()
