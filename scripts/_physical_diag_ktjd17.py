#!/usr/bin/env python3
"""Physical / geometric diagnostics of frozen-protocol generations against their ground-truth clips (KTJD-17 units).

The frozen gen-eval (scripts/_eval_v2_gen_in_evalspace.py) stores every generated clip in the GENERATOR's normalized space
(shard npz, key clip__<id>, [target_frames, J_rig, 17]). This tool de-normalizes them exactly as the merge stage does
(raw = x*(std+_STD_FLOOR)+mean with the generator's own stats; a representation view is cut to the clip's valid length and
converted with src.data.ktjd17_anytop13.anytop13_to_ktjd17), decodes gen and GT with the OFFICIAL codec (decode_ktjd17: direct
and FK positions, global/local rotations, no temporal integration) and reports, per clip and aggregated, quantities that
R-precision / FID do not resolve:

  travel      root XZ path length / bl          disp        root XZ end-start displacement / bl
  root_h      mean root height / bl              turn        total |d heading| over the GT-valid heading transitions (rad)
  heading_norm_err  mean | |heading| - 1 | over GT-valid frames (a unit vector when valid, per the KTJD contract)
  fk_gap      mean |direct - FK| / bl            bone_err    mean |bone length of direct positions - rest bone| / bl
  jitter      mean second-difference norm of direct positions / bl (all joints)      jitter_root  same, root row
  vel_incons  mean |stored world velocity - forward difference of direct positions| / bl (per second), NON-ROOT rows
              (the 13-view's inverse integrates the root from its velocity, so its root residual is 0 by construction)
  leaf_dev    mean deviation (deg) of animated LEAF joints' local rotation from rest   leaf_tstd  its std over time
  int_dev     same for animated internal joints                                       int_tstd   its std over time
  contact_frac  fraction of (supervised joint, frame) cells flagged in contact (ch12 > 0.5)
  slide       mean horizontal speed of joints while flagged in contact (bl/s)

bl = the rig's mean rest bone length + 1e-3 (root offset excluded), the trainer's fkdist denominator. Before decoding, every
arm's samples are projected through the PARENT corpus's per-cell space exactly as the eval does before scoring (constant cells
reset to their constant). Shards must carry this checkpoint's sha256 and one shared protocol (seed/steps/cfg/TF32/pins/plan). Leaf joints of the
13-channel view have no rotation slot: the inverse conversion fills them from the parent, so their leaf_dev is 0 by
construction -- the point of measuring it.

  python scripts/_physical_diag_ktjd17.py --gen_ckpt CKPT --shards S0.npz S1.npz ... --out diag.json [--limit N]
  python scripts/_physical_diag_ktjd17.py --compare A.json B.json [...]        # paired table over the common clips
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


def rot_angle_deg(R):
    tr = np.clip((np.trace(R, axis1=-2, axis2=-1) - 1.0) / 2.0, -1.0, 1.0)
    return np.degrees(np.arccos(tr))


def clip_metrics(raw17, sk, cv, fps, strict_gt, heading_support):
    """raw17 [T,J,17] float64 KTJD-17 raw units; heading_support [T] bool = the frames whose GROUND-TRUTH heading is valid.
    Heading metrics of gen and GT are both taken over this one support (turn over transitions with both frames in it): the
    support is a property of the clip, identical for every arm, so paired totals are comparable and the 13-view inverse's
    filled headings cannot inflate its total (codex r2 P2 / r3 P1); no magnitude cutoff -- a generated heading that is not a
    unit vector is scored as it is and its norm error is reported separately (codex r3 P2). Returns (dict, decode note)."""
    from src.data.ktjd17.decoder import decode_ktjd17
    from src.data.ktjd17.codec import Ktjd17CodecError
    T, J = raw17.shape[:2]
    parents = np.asarray(sk["parents"])[:J].astype(int)
    off = np.asarray(sk["offset_parent_local"], dtype=np.float64)[:J]
    kinds = np.asarray(sk["rotation_source_kind"]).astype(str)[:J]
    bl = float(np.linalg.norm(off[1:], axis=-1).mean()) + 1e-3 if J > 1 else 1.0   # the trainer's fkdist denominator (fk_torch.py)
    note = "strict"
    try:
        dec = decode_ktjd17(raw17, parents=sk["parents"], R_rest_global=sk["R_rest_global"], R_rest_local=sk["R_rest_local"],
                            offset_parent_local=sk["offset_parent_local"], rotation_source_kind=sk["rotation_source_kind"],
                            strict_gt=strict_gt)
    except Ktjd17CodecError:
        if strict_gt:
            note = "nonstrict_fallback"
            dec = decode_ktjd17(raw17, parents=sk["parents"], R_rest_global=sk["R_rest_global"], R_rest_local=sk["R_rest_local"],
                                offset_parent_local=sk["offset_parent_local"], rotation_source_kind=sk["rotation_source_kind"],
                                strict_gt=False)
        else:
            raise
    P, F = dec.positions_direct, dec.positions_fk                    # [T,J,3] world
    m = {}
    root = P[:, 0]
    step = np.diff(root[:, [0, 2]], axis=0)
    m["travel"] = float(np.linalg.norm(step, axis=-1).sum() / bl) if T > 1 else 0.0
    m["disp"] = float(np.linalg.norm(root[-1, [0, 2]] - root[0, [0, 2]]) / bl)
    m["root_h"] = float(root[:, 1].mean() / bl)
    h = raw17[:, 0, 15:17]
    sup = np.asarray(heading_support, dtype=bool)
    th = np.arctan2(h[:, 1], h[:, 0])                                 # KTJD heading = (fwd_z, fwd_x)
    both = sup[1:] & sup[:-1]
    d = (th[1:] - th[:-1] + np.pi) % (2 * np.pi) - np.pi
    m["turn"] = float(np.abs(d[both]).sum()) if both.any() else float("nan")   # no GT-valid heading transition -> unavailable
    m["heading_norm_err"] = float(np.abs(np.linalg.norm(h[sup], axis=-1) - 1.0).mean()) if sup.any() else float("nan")
    m["fk_gap"] = float(np.linalg.norm(P - F, axis=-1).mean() / bl)
    if J > 1:
        blen = np.linalg.norm(P[:, 1:] - P[:, parents[1:]], axis=-1)   # [T,J-1]
        m["bone_err"] = float(np.abs(blen - np.linalg.norm(off[1:], axis=-1)[None]).mean() / bl)
    else:
        m["bone_err"] = 0.0
    if T >= 3:
        acc = np.linalg.norm(P[2:] - 2 * P[1:-1] + P[:-2], axis=-1)
        m["jitter"], m["jitter_root"] = float(acc.mean() / bl), float(acc[:, 0].mean() / bl)
    else:
        m["jitter"] = m["jitter_root"] = float("nan")
    # stored world velocity vs the forward difference of the direct positions, NON-ROOT rows only: the 13-view's inverse
    # integrates the root track from its stored velocity and takes the vertical one from the height, so the root row's
    # residual is zero by construction for that arm (codex r1 P2); tail frame (repeat-last) excluded
    if T >= 2 and J > 1:
        v_fd = (P[1:, 1:] - P[:-1, 1:]) * fps
        m["vel_incons"] = float(np.linalg.norm(raw17[:-1, 1:, 9:12] - v_fd, axis=-1).mean() / bl)
    else:
        m["vel_incons"] = float("nan")
    # local rotation deviation from rest, animated joints only (fixed_dof rows are forced to rest by the codec; joints whose
    # rotation cells the stats artifact marks constant are excluded from the model and de-normalize to the constant)
    rest_local = np.asarray(sk["R_rest_local"], dtype=np.float64)[:J]
    dev = rot_angle_deg(np.matmul(np.swapaxes(rest_local[None], -1, -2), dec.local_rotations))   # [T,J]
    has_child = np.zeros(J, dtype=bool); has_child[parents[1:]] = True
    animated = (kinds == "animated_dof") & cv[:J, 3:9].any(axis=1); animated[0] = False
    leaf, inner = animated & ~has_child, animated & has_child
    for name, sel in (("leaf", leaf), ("int", inner)):
        if sel.any():
            m[f"{name}_dev"] = float(dev[:, sel].mean()); m[f"{name}_tstd"] = float(dev[:, sel].std(axis=0).mean())
        else:
            m[f"{name}_dev"] = m[f"{name}_tstd"] = float("nan")
    m["n_leaf_animated"], m["n_int_animated"] = int(leaf.sum()), int(inner.sum())
    csel = cv[:J, 12]
    if csel.any():
        c = raw17[:, csel, 12] > 0.5                                 # [T,Jc]
        m["contact_frac"] = float(c.mean())
        if T >= 2:
            sp = np.linalg.norm(np.diff(P[:, csel][:, :, [0, 2]], axis=0), axis=-1) * fps / bl   # [T-1,Jc]
            cc = c[:-1] & c[1:]
            m["slide"] = float(sp[cc].mean()) if cc.any() else float("nan")
        else:
            m["slide"] = float("nan")
    else:
        m["contact_frac"] = m["slide"] = float("nan")
    return m, note


def run(a):
    import torch
    from src.data.ktjd17_incontext import Ktjd17Base, _STD_FLOOR
    from src.data.ktjd17.loader import load_motion_npz
    from src.data.ktjd17_anytop13 import REPRESENTATION_ID, anytop13_to_ktjd17
    import hashlib, io
    buf = Path(a.gen_ckpt).read_bytes()                              # ONE byte buffer is hashed and deserialized (codex r3 P2)
    gen_sha = hashlib.sha256(buf).hexdigest()
    ck = torch.load(io.BytesIO(buf), map_location="cpu", weights_only=False); del buf
    ca = ck["args"]; ca = vars(ca) if not isinstance(ca, dict) else ca
    target_frames = int(ca["target_frames"])
    if str(ca.get("corpus")) != "ktjd17":
        raise SystemExit(f"[refuse] ckpt corpus={ca.get('corpus')!r}; KTJD-17 only")
    root = ca["ktjd_root"]; excl = ca.get("exclude_clips") or None
    base = Ktjd17Base(root, caption_emb_cache=ca["caption_cache"], joint_semantics=ca["joint_sem"], texts_json=ca["texts_json"],
                      percell_stats=ca.get("ktjd_percell_stats", "data/ktjd17_percell_stats_v1.npz"), exclude_clips=excl,
                      normalization=str(ca.get("rep_norm") or "percell"))
    # the live data must be what the checkpoint was trained on and what the shards were generated on (codex r2 P1): the
    # trainer's ktjd_pins vs base.provenance (bidirectional for the representation pin), and the eval's base_provenance_sha256
    live = {**base.provenance, "exclusion": base.provenance_exclusion}
    pins_ck = ck.get("ktjd_pins") or {}
    drift = sorted(k for k, v in pins_ck.items() if k in live and live[k] != v)
    if drift:
        raise SystemExit(f"[refuse] data drift vs the checkpoint's ktjd_pins: {drift}")
    if ("representation" in live) != ("representation" in pins_ck) or pins_ck.get("representation") != live.get("representation"):
        # bidirectional: a view's checkpoint must carry the pin and a pinned checkpoint must meet a view (codex r3 P2)
        raise SystemExit(f"[refuse] representation pin mismatch: live data {live.get('representation')!r} vs checkpoint {pins_ck.get('representation')!r}")
    prov_sha = hashlib.sha256(json.dumps(live, sort_keys=True, default=str).encode()).hexdigest()
    rep = ((getattr(base, "derivation", None) or {}).get("representation") or {})
    gen_rep = str(rep.get("id")) if rep else "ktjd17"
    if rep:
        if gen_rep != REPRESENTATION_ID:
            raise SystemExit(f"[refuse] unknown representation view {gen_rep!r}")
        parent_root = str(base.derivation["parent_root"]); parent_stats = str(rep["parent_norm_stats"]["path"])
        # the view's pins, as the eval checks them: parent stats bytes, converter bytes, parent manifest bytes
        if hashlib.sha256(Path(parent_stats).read_bytes()).hexdigest() != str(rep["parent_norm_stats"]["sha256"]):
            raise SystemExit(f"[refuse] parent stats {parent_stats} do not match the sha pinned in the view's derivation.json")
        if hashlib.sha256((REPO / "src" / "data" / "ktjd17_anytop13.py").read_bytes()).hexdigest() != str(rep.get("converter_sha256")):
            raise SystemExit("[refuse] src/data/ktjd17_anytop13.py differs from the converter the view was built with")
        if hashlib.sha256((Path(parent_root) / "manifests" / "clips.jsonl").read_bytes()).hexdigest() != str(base.derivation.get("parent_manifest_sha256")):
            raise SystemExit("[refuse] the parent manifest differs from the one the view was derived from")
    else:
        parent_root, parent_stats = root, ca.get("ktjd_percell_stats", "data/ktjd17_percell_stats_v1.npz")
    # GT / projection base = the PARENT corpus with PER-CELL normalisation, as the frozen evaluator constructs it -- also when the
    # generator is the scale-only arm on the same root (codex r5: reusing its base would project with scale-only stats)
    gt_base = base if (parent_root == root and base.normalization == "percell") else Ktjd17Base(
        parent_root, caption_emb_cache=ca["caption_cache"], joint_semantics=ca["joint_sem"], texts_json=ca["texts_json"],
        percell_stats=parent_stats, exclude_clips=excl, normalization="percell")
    sch = json.loads((Path(parent_root) / "schema.json").read_text())
    fps, eps_h = float(sch["fps_target"]), float(sch["heading"]["eps_h"])
    rows = {str(r["clip_id"]): r for r in gt_base._rows}
    T_of = {str(r["clip_id"]): int(r["T_target"]) for r in base._rows}
    # identity (codex r1 P1): every shard must come from THIS checkpoint (sha, not just the path) and all shards must share one
    # protocol -- seed / steps / cfg / TF32 runtime / data + view pins / plan; the same keys the eval's merge insists on
    PROTO_KEYS = ("seed", "steps", "cfg_text", "nshards", "ktjd_root", "exclude_clips", "base_provenance_sha256",
                  "plan_sha256", "protocol_variant", "source_fingerprint")
    shards_meta, gens, proto = [], {}, None
    for p in a.shards:
        z = np.load(p, allow_pickle=False)
        meta = json.loads(str(z["__meta"]))
        if str(meta.get("gen_ckpt")) != a.gen_ckpt or str(meta.get("gen_ckpt_sha256")) != gen_sha:
            raise SystemExit(f"[refuse] shard {p} was generated from {meta.get('gen_ckpt')!r} sha {str(meta.get('gen_ckpt_sha256'))[:12]}, "
                             f"not {a.gen_ckpt!r} sha {gen_sha[:12]}")
        this = {k: meta.get(k) for k in PROTO_KEYS}
        this["runtime"] = meta.get("runtime")                       # the WHOLE sampling runtime fingerprint (codex r2 P2)
        this["allow_tf32_matmul"] = (meta.get("runtime") or {}).get("allow_tf32_matmul")
        if proto is None:
            proto = this
        elif this != proto:
            raise SystemExit(f"[refuse] shard {p} does not share the protocol of the first shard: {this} vs {proto}")
        if str(meta.get("ktjd_root")) != str(root):
            raise SystemExit(f"[refuse] shard {p} was generated on {meta.get('ktjd_root')!r}, the checkpoint pins {root!r}")
        if str(meta.get("base_provenance_sha256")) != prov_sha:
            raise SystemExit(f"[refuse] shard {p} was generated on data with provenance {str(meta.get('base_provenance_sha256'))[:12]}, "
                             f"the live data has {prov_sha[:12]} (stats / captions / exclusion / schema changed)")
        shards_meta.append({"path": p, "shard": meta.get("shard")})
        for k in z.files:
            if k.startswith("clip__"):
                mid = k[len("clip__"):]
                if mid in gens:
                    raise SystemExit(f"[refuse] clip {mid} appears in two shards")
                gens[mid] = np.asarray(z[k], dtype=np.float32)
    mids = sorted(gens)
    if a.limit:
        mids = mids[:a.limit]
    recs, n_degenerate, n_nonstrict_gt = [], 0, 0
    for i, mid in enumerate(mids):
        if mid not in rows:
            raise SystemExit(f"[refuse] generated clip {mid} is not a row of {parent_root}")
        r = rows[mid]; rig = str(r["rig_id"]); sk = gt_base.skeleton(rig); cv = gt_base.static_masks(rig)["channel_valid"]
        g = gens[mid]; J = len(sk["parents"])
        if g.shape != (target_frames, J, 17):
            raise SystemExit(f"[refuse] clip {mid}: generated array has shape {g.shape}, the protocol stores ({target_frames}, {J}, 17) for rig {rig}")
        if not np.isfinite(g).all():
            raise SystemExit(f"[refuse] clip {mid}: non-finite generated values")
        mu_v, sd_v = base._stats(rig)
        raw = g.astype(np.float64) * (sd_v[None, :J] + _STD_FLOOR) + mu_v[None, :J]
        Tv = min(T_of[mid], raw.shape[0]); raw = raw[:Tv]
        if rep:
            raw, hv, dg = anytop13_to_ktjd17(raw, np.asarray(sk["parents"])[:J], fps=fps, eps_h=eps_h)
            raw = np.array(raw, dtype=np.float64); raw[~hv, 0, 15:17] = 0.0
            n_degenerate += int(dg["degenerate_facing_frames"]) + int(dg["degenerate_child_slots"])
        # the eval's projection into the PARENT per-cell space before scoring: cells the parent stats mark constant are reset
        # to their constant (normalized 0 -> mean); identical to what the evaluator sees (codex r1 P2). For a per-cell KTJD-17
        # arm this is the identity up to the 1e-6 std floor.
        mu_p, sd_p = gt_base._stats(rig)
        gp = ((raw - mu_p[None, :J]) / (sd_p[None, :J] + _STD_FLOOR)).astype(np.float32); gp[:, ~cv[:J]] = 0.0   # the merge's float32 boundary
        raw = gp.astype(np.float64) * (sd_p[None, :J] + _STD_FLOOR) + mu_p[None, :J]
        pay = load_motion_npz(Path(parent_root) / r["motion_relpath"], expected_fps_target=fps)
        if str(pay["clip_id"]) != mid or str(pay["rig_id"]) != rig:
            raise SystemExit(f"[refuse] GT payload identity mismatch for {mid}")
        gt = np.asarray(pay["motion"], dtype=np.float64); gt_hv = np.asarray(pay["heading_valid"], dtype=bool)
        if gt.shape[1] != J:
            raise SystemExit(f"[refuse] clip {mid}: GT J={gt.shape[1]} vs generated J={J}")
        if gt.shape[0] < Tv:
            raise SystemExit(f"[refuse] clip {mid}: GT has {gt.shape[0]} frames, fewer than the clip's T_target {Tv}")
        Tc = Tv
        support = gt_hv[:Tc]                                          # GT heading validity: the clip's own, arm-independent
        mg, _ = clip_metrics(raw[:Tc], sk, cv, fps, strict_gt=False, heading_support=support)
        mt, note = clip_metrics(gt[:Tc], sk, cv, fps, strict_gt=True, heading_support=support)
        n_nonstrict_gt += note != "strict"
        recs.append({"clip_id": mid, "rig": rig, "T": int(Tc), "J": int(J), "gen": mg, "gt": mt})
        if (i + 1) % 500 == 0:
            print(f"[diag] {i + 1}/{len(mids)} clips", flush=True)
    keys = [k for k, v in recs[0]["gen"].items() if isinstance(v, float)]
    summ = {}
    for k in keys:
        G = np.array([r["gen"][k] for r in recs], dtype=np.float64); Tt = np.array([r["gt"][k] for r in recs], dtype=np.float64)
        ok = np.isfinite(G) & np.isfinite(Tt)
        summ[k] = {"n": int(ok.sum()), "gen_mean": float(G[ok].mean()), "gt_mean": float(Tt[ok].mean()),
                   "diff_mean": float((G[ok] - Tt[ok]).mean()), "diff_median": float(np.median(G[ok] - Tt[ok])),
                   "absdiff_mean": float(np.abs(G[ok] - Tt[ok]).mean())}
    out = {"gen_ckpt": a.gen_ckpt, "gen_ckpt_sha256": gen_sha, "protocol": proto, "target_frames": target_frames,
           "base_provenance_sha256": prov_sha,
           "gen_epoch": int(ck.get("epoch", -1)), "gen_representation": gen_rep,
           "gen_normalization": base.normalization, "gen_root": str(root), "gt_root": str(parent_root), "fps": fps,
           "shards": shards_meta, "n_clips": len(recs), "degenerate_6d_cells_converted": n_degenerate,
           "gt_nonstrict_decodes": n_nonstrict_gt, "units": "bl = rig mean rest bone length; angles deg; turn rad; speeds per second",
           "summary": summ, "clips": recs}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=1))
    print(f"[diag] {len(recs)} clips ({gen_rep}, epoch {out['gen_epoch']}) -> {a.out}")
    for k in keys:
        s = summ[k]; print(f"  {k:14s} gen {s['gen_mean']:9.4f}  gt {s['gt_mean']:9.4f}  diff {s['diff_mean']:+9.4f}  |diff| {s['absdiff_mean']:8.4f}")


def compare(paths, allow_tf32_diff=False):
    runs = [json.loads(Path(p).read_text()) for p in paths]
    sets = [{c["clip_id"] for c in r["clips"]} for r in runs]
    if any(s_ != sets[0] for s_ in sets[1:]):
        raise SystemExit("[refuse] the arms were not measured on the same clip set: " + ", ".join(f"{Path(p).stem}={len(s_)}" for p, s_ in zip(paths, sets)))
    common = sets[0]
    # the arms must share the sampling protocol (codex r2 P1): seed / steps / cfg / variant / frame budget; representation and
    # normalisation are the intended differences; TF32 vs fp32 sampling is allowed only when said so, and then printed
    SAME = ("seed", "steps", "cfg_text", "protocol_variant", "plan_sha256")   # same plan = same per-clip noise (per-batch reseed)
    for r, p in zip(runs, paths):                                             # metadata must be PRESENT, not merely equal (codex r4)
        if not isinstance((r.get("protocol") or {}).get("plan_sha256"), str) or not (r.get("protocol") or {}).get("plan_sha256"):
            raise SystemExit(f"[refuse] {p}: no generation plan hash recorded")
        if not isinstance((r.get("protocol") or {}).get("allow_tf32_matmul"), bool):
            raise SystemExit(f"[refuse] {p}: TF32 sampling setting not recorded")
    for r, p in zip(runs[1:], paths[1:]):
        for k in SAME:
            if r["protocol"].get(k) != runs[0]["protocol"].get(k):
                raise SystemExit(f"[refuse] {p}: protocol {k}={r['protocol'].get(k)!r} differs from {runs[0]['protocol'].get(k)!r}")
        if r["target_frames"] != runs[0]["target_frames"]:
            raise SystemExit(f"[refuse] {p}: target_frames {r['target_frames']} vs {runs[0]['target_frames']}")
    tf = [r["protocol"].get("allow_tf32_matmul") for r in runs]
    if len(set(map(str, tf))) > 1 and not allow_tf32_diff:
        raise SystemExit(f"[refuse] arms were sampled under different TF32 settings {tf}; pass --allow_tf32_diff to compare anyway")
    fps_ = [str(r["protocol"].get("source_fingerprint"))[:12] for r in runs]
    print(f"[compare] {len(common)} clips (identical sets); arms: " + " | ".join(f"{Path(p).stem} ({r['gen_representation']}/{r['gen_normalization']} ep{r['gen_epoch']}, tf32={t})" for p, r, t in zip(paths, runs, tf)))
    if len(set(fps_)) > 1:
        print(f"[compare] NOTE: generation-code fingerprints differ across arms {fps_} (code state at sampling time; the eval's merge audited them)")
    by = [{c["clip_id"]: c for c in r["clips"] if c["clip_id"] in common} for r in runs]
    keys = [k for k, v in runs[0]["clips"][0]["gen"].items() if isinstance(v, float)]
    ids = sorted(common)
    # the arms must have been measured against the SAME ground truth (codex r1 P1): same GT corpus, fps, rig, frame count,
    # joint count and GT metric values per clip -- otherwise "closer to GT" compares against different targets
    for r, p in zip(runs[1:], paths[1:]):
        if r["gt_root"] != runs[0]["gt_root"] or r["fps"] != runs[0]["fps"]:
            raise SystemExit(f"[refuse] {p}: GT corpus/fps {r['gt_root']}@{r['fps']} differ from {runs[0]['gt_root']}@{runs[0]['fps']}")
    for m in ids:
        c0 = by[0][m]
        for b, p in zip(by[1:], paths[1:]):
            c = b[m]
            if (c["rig"], c["T"], c["J"]) != (c0["rig"], c0["T"], c0["J"]):
                raise SystemExit(f"[refuse] {p}: clip {m} rig/T/J {(c['rig'], c['T'], c['J'])} vs {(c0['rig'], c0['T'], c0['J'])}")
            for k in keys:
                x, y = c["gt"].get(k), c0["gt"].get(k)
                if not ((x is None and y is None) or (isinstance(x, float) and isinstance(y, float) and
                        ((np.isnan(x) and np.isnan(y)) or np.isclose(x, y, rtol=1e-9, atol=1e-12)))):
                    raise SystemExit(f"[refuse] {p}: clip {m} GT {k} = {x} differs from {y}")
    hdr = f"{'metric':14s} {'GT':>9s} " + " ".join(f"{'arm' + str(i):>9s}" for i in range(len(runs))) + "  |gen-GT| per arm" + ("   closer-to-GT win% (arm0 vs arm1)" if len(runs) >= 2 else "")
    print(hdr)
    for k in keys:
        gt = np.array([by[0][m]["gt"][k] for m in ids])                     # verified identical across arms above
        cols = [np.array([b[m]["gen"][k] for m in ids]) for b in by]
        ok = np.isfinite(gt)
        for c in cols:
            ok &= np.isfinite(c)
        if ok.sum() == 0:
            continue
        errs = [np.abs(c[ok] - gt[ok]) for c in cols]
        line = f"{k:14s} {gt[ok].mean():9.4f} " + " ".join(f"{c[ok].mean():9.4f}" for c in cols)
        line += "  " + " ".join(f"{e.mean():8.4f}" for e in errs)
        if len(runs) >= 2:
            line += f"   {100.0 * (errs[0] < errs[1]).mean():5.1f}% / {100.0 * (errs[1] < errs[0]).mean():5.1f}%"
        print(line)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gen_ckpt"); ap.add_argument("--shards", nargs="*", default=[]); ap.add_argument("--out")
    ap.add_argument("--limit", type=int, default=0, help="first N clips only (smoke)")
    ap.add_argument("--compare", nargs="*", default=None, help="per-arm diag JSONs to compare (same clip set, same sampling protocol)")
    ap.add_argument("--allow_tf32_diff", action="store_true", help="compare arms sampled under different TF32 settings (printed)")
    a = ap.parse_args()
    if a.compare:
        compare(a.compare, allow_tf32_diff=a.allow_tf32_diff); return
    if not (a.gen_ckpt and a.shards and a.out):
        raise SystemExit("need --gen_ckpt, --shards and --out (or --compare)")
    run(a)


if __name__ == "__main__":
    main()
