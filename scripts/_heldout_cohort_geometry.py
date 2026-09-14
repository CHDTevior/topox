#!/usr/bin/env python3
"""Geometry of the held-out-cohort samples of a gen-eval report against the cohort's real clips (2026-09-14).

Reads the merged report's shards (protocol.generation.shards, byte-verified), de-normalises every sample with the generator's
own serving statistics (the rig's rest normalisation) through the same float32 per-cell boundary as scripts/_rescore_fk_channels.py,
decodes it with the official codec (src/data/ktjd17/decoder.decode_ktjd17, strict_gt=False) and measures, on the played frames
min(T_target, 240):
  fk_pose_gap              mean |FK(rotations) - direct positions| / mean rest bone length (samples; real clips are ~0 by construction)
  bone_len_err             mean |direct bone length - rest bone length| / mean rest bone length (samples and real clips)
  leaf_global_rest_delta   mean rotation angle, degrees, of the GLOBAL rest delta G R_rest^T (ch 3:9) over LEAF joints -- a pose
                           measure that includes the body's global orientation, not local articulation (samples and real clips)
  foot_slide_bl_s          mean horizontal speed (bl/s) of direct positions over the (t, joint) pairs whose contact flag (ch 12)
                           exceeds 0.5 at frame t, for the interval t -> t+1; undefined (null) for a clip with no such pair; the
                           support (contact pairs, clips defined) is reported per side and the paired aggregate uses the clips
                           where BOTH sides are defined
  root_disp_bl             NET endpoint displacement of the smooth root track (ch 13:15, frame T-1 minus frame 0) / bl -- not a
                           path length or a speed; played lengths are reported beside it
Aggregates: clip-weighted means (every clip equal; the big rigs dominate), rig-equal means (every rig's mean weighted equally),
per rig, per tree class and over the cohort; per-clip records are kept in the JSON. Sample-side conventions follow
scripts/_rescore_fk_channels.py and the gen-eval's merge: samples leave the generator's serving space through the generator's own
statistics, are projected into the evaluator's PER-CELL space in float32 with the per-cell masks (the eval's boundary) and are
decoded from there; the codec, the played length min(T_target, 240) and the bone-length unit are the same.
  python scripts/_heldout_cohort_geometry.py --report runs/_heldout/eval/geneval_A_ep119_pool64_strict.json --out runs/_heldout/eval/geom_A_ep119.json
"""
import argparse, hashlib, json, os, sys
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, os.getcwd())
from src.data.ktjd17_incontext import Ktjd17Base, _STD_FLOOR                      # noqa: E402
from src.data.ktjd17.decoder import decode_ktjd17                                   # noqa: E402
EVAL_MAX_FRAMES = 240


def bone_length(sk, J):
    off = np.asarray(sk["offset_parent_local"], dtype=np.float64)[:J]
    return float(np.linalg.norm(off[1:], axis=-1).mean()) + 1e-3 if J > 1 else 1.0


def rot_angle_deg(d6):
    """[..., 6] rest-delta six-vector -> rotation angle in degrees (column Gram-Schmidt, as the codec decodes)."""
    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = a1 / np.maximum(np.linalg.norm(a1, axis=-1, keepdims=True), 1e-8)
    u2 = a2 - (b1 * a2).sum(-1, keepdims=True) * b1
    b2 = u2 / np.maximum(np.linalg.norm(u2, axis=-1, keepdims=True), 1e-8)
    b3 = np.cross(b1, b2)
    tr = b1[..., 0] + b2[..., 1] + b3[..., 2]                                        # trace of [b1 b2 b3]
    return np.degrees(np.arccos(np.clip((tr - 1.0) / 2.0, -1.0, 1.0)))


def measures(raw, sk, J, fps, bl, leaves, gen: bool):
    """raw [T,J,17] float64 KTJD-17 raw units on the played frames."""
    T = raw.shape[0]
    out = {}
    if gen:
        dec = decode_ktjd17(raw, parents=sk["parents"], R_rest_global=sk["R_rest_global"], R_rest_local=sk["R_rest_local"],
                            offset_parent_local=sk["offset_parent_local"], rotation_source_kind=sk["rotation_source_kind"], strict_gt=False)
        out["fk_pose_gap"] = float(np.linalg.norm(dec.positions_direct - dec.positions_fk, axis=-1).mean() / bl)
        out["degenerate_6d_cells"] = int(dec.model_d6_degenerate.sum())
    par = np.asarray(sk["parents"])[:J]
    world = raw[..., 0:3].copy(); world[..., 0] += raw[:, 0:1, 13]; world[..., 2] += raw[:, 0:1, 14]
    rest_len = np.linalg.norm(np.asarray(sk["offset_parent_local"], dtype=np.float64)[1:J], axis=-1)
    seg = np.linalg.norm(world[:, 1:] - world[:, par[1:]], axis=-1)                        # [T,J-1]
    out["bone_len_err"] = float(np.abs(seg - rest_len[None]).mean() / bl)
    out["leaf_global_rest_delta_deg"] = float(rot_angle_deg(raw[:, leaves, 3:9]).mean()) if len(leaves) else None
    contact = raw[..., 12] > 0.5                                                             # [T,J], flag at frame t
    n_pairs = int(contact[:-1].sum()) if T > 1 else 0                                        # (t, joint) pairs with an interval t -> t+1
    out["contact_pairs"] = n_pairs
    if n_pairs:
        sp = np.linalg.norm((world[1:, :, [0, 2]] - world[:-1, :, [0, 2]]) * fps, axis=-1)   # horizontal speed of the interval t -> t+1
        out["foot_slide_bl_s"] = float(sp[contact[:-1]].mean() / bl)
    else:
        out["foot_slide_bl_s"] = None
    out["root_disp_bl"] = float(np.linalg.norm(raw[-1, 0, 13:15] - raw[0, 0, 13:15]) / bl)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", required=True); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rep = json.loads(Path(a.report).read_text()); p = rep["protocol"]
    shards = p["generation"].get("shards")
    if not isinstance(shards, list):
        raise SystemExit("[refuse] the report was not produced from saved shards")
    ck = torch.load(p["gen_ckpt"], map_location="cpu", weights_only=False); ca = ck["args"]; ca = vars(ca) if not isinstance(ca, dict) else ca
    excl = p.get("cohort_exclude_clips") if p.get("cohort_override") else (ca.get("exclude_clips") or None)
    base = Ktjd17Base(ca["ktjd_root"], caption_emb_cache=ca["caption_cache"], joint_semantics=ca["joint_sem"], texts_json=ca["texts_json"],
                      percell_stats=ca.get("ktjd_percell_stats"), exclude_clips=excl, normalization=str(ca.get("rep_norm", "percell")))
    # the evaluator's per-cell space: the gen-eval projects every sample into it (float32, invalid cells zeroed) before scoring
    base_pc = base if base.normalization == "percell" else Ktjd17Base(
        ca["ktjd_root"], caption_emb_cache=ca["caption_cache"], joint_semantics=ca["joint_sem"], texts_json=ca["texts_json"],
        percell_stats=ca.get("ktjd_percell_stats"), exclude_clips=excl, normalization="percell")
    fps = float(json.loads((Path(ca["ktjd_root"]) / "schema.json").read_text())["fps_target"])
    rows = {str(r["clip_id"]): r for r in base._rows}; idx_of = {str(r["clip_id"]): i for i, r in enumerate(base._rows)}
    tree = (json.loads(Path(excl).read_text()).get("rigs") or {}) if excl else {}
    gen = {}
    for s in shards:
        raw_b = Path(s["path"]).read_bytes()
        if hashlib.sha256(raw_b).hexdigest() != s["sha256"]:
            raise SystemExit(f"[refuse] shard {s['path']} does not match the report's sha256")
        with np.load(s["path"], allow_pickle=False) as z:
            for k in z.files:
                if k.startswith("clip__"):
                    gen[k[len("clip__"):]] = z[k]
    if len(gen) != int(p["val_n"]):
        raise SystemExit(f"[refuse] {len(gen)} samples in the shards, the report scores {p['val_n']}")
    per_clip = {}
    leaves_of, bl_of = {}, {}
    for n_done, (mid, gs) in enumerate(sorted(gen.items()), 1):
        r = rows[mid]; rig = str(r["rig_id"]); sk = base.skeleton(rig); J = gs.shape[1]
        if rig not in leaves_of:
            par = np.asarray(sk["parents"])[:J]; has_child = np.zeros(J, bool); has_child[par[1:]] = True
            leaves_of[rig] = np.where(~has_child)[0]; bl_of[rig] = bone_length(sk, J)
        mu, sd = base._stats(rig); mu, sd = mu[:J], sd[:J]                                          # the generator's serving space
        mu_p, sd_p = base_pc._stats(rig); mu_p, sd_p = mu_p[:J], sd_p[:J]                             # the evaluator's per-cell space
        cv = base_pc.static_masks(rig)["channel_valid"][:J]
        Tv = min(int(r["T_target"]), gs.shape[0], EVAL_MAX_FRAMES)
        raw = gs.astype(np.float64) * (sd[None] + _STD_FLOOR) + mu[None]
        gp = ((raw - mu_p[None]) / (sd_p[None] + _STD_FLOOR)).astype(np.float32); gp[:, ~cv] = 0.0    # the eval's float32 per-cell boundary
        raw = (gp.astype(np.float64) * (sd_p[None] + _STD_FLOOR) + mu_p[None])[:Tv]
        it = base[idx_of[mid]]
        x = np.asarray(it["anytop_x"])[:J, :17, :Tv].transpose(2, 0, 1).astype(np.float64)              # the real clip, served space
        raw_gt = x * (np.asarray(it["anytop_std"])[:J, :17][None] + _STD_FLOOR) + np.asarray(it["anytop_mean"])[:J, :17][None]
        per_clip[mid] = {"rig": rig, "tree": tree.get(rig, "unknown"), "T_target": int(r["T_target"]), "played": Tv,
                         "gen": measures(raw, sk, J, fps, bl_of[rig], leaves_of[rig], True),
                         "gt": measures(raw_gt, sk, J, fps, bl_of[rig], leaves_of[rig], False)}
        if n_done % 1000 == 0:
            print(f"[geom] {n_done}/{len(gen)}", flush=True)
    keys_g = ["fk_pose_gap", "bone_len_err", "leaf_global_rest_delta_deg", "foot_slide_bl_s", "root_disp_bl"]
    keys_t = ["bone_len_err", "leaf_global_rest_delta_deg", "foot_slide_bl_s", "root_disp_bl"]
    def mean_or_null(vals):
        v = [x for x in vals if x is not None and np.isfinite(x)]
        return (float(np.mean(v)) if v else None), len(v)
    def agg(clips):
        """clip-weighted means (every clip equal) with the number of clips each mean stands on; the slide also paired"""
        o = {"n": len(clips)}
        for k in keys_g: o["gen_" + k], o["gen_" + k + "_n"] = mean_or_null([c["gen"][k] for c in clips])
        for k in keys_t: o["gt_" + k], o["gt_" + k + "_n"] = mean_or_null([c["gt"][k] for c in clips])
        both = [c for c in clips if c["gen"]["foot_slide_bl_s"] is not None and c["gt"]["foot_slide_bl_s"] is not None]
        o["paired_foot_slide"] = {"n": len(both), "gen": mean_or_null([c["gen"]["foot_slide_bl_s"] for c in both])[0],
                                  "gt": mean_or_null([c["gt"]["foot_slide_bl_s"] for c in both])[0]}
        o["contact_pairs"] = {"gen": int(sum(c["gen"]["contact_pairs"] for c in clips)), "gt": int(sum(c["gt"]["contact_pairs"] for c in clips))}
        o["played_frames"] = {"mean": float(np.mean([c["played"] for c in clips])), "min": int(min(c["played"] for c in clips)),
                              "max": int(max(c["played"] for c in clips)), "truncated_to_240": int(sum(c["T_target"] > c["played"] for c in clips))}
        o["gen_degenerate_6d_cells"] = int(sum(c["gen"]["degenerate_6d_cells"] for c in clips))
        return o
    by_rig = {}
    for c in per_clip.values(): by_rig.setdefault(c["rig"], []).append(c)
    rig_agg = {r: {"tree": v[0]["tree"], **agg(v)} for r, v in sorted(by_rig.items())}
    def rig_equal(rigs):
        """every rig's clip-weighted mean weighted equally; the per-side slide means keep their own (unequal) supports and are
        labelled so; the paired slide averages each rig's PAIRED mean over the rigs that have one (codex geom r2 P2)"""
        o = {"n_rigs": len(rigs)}
        for k in ["gen_" + x for x in keys_g] + ["gt_" + x for x in keys_t]:
            o[k] = mean_or_null([rig_agg[r][k] for r in rigs])[0]
        o["per_side_slide_note"] = "gen_/gt_foot_slide_bl_s average per-side rig means with unequal clip supports; compare paired_foot_slide"
        pr = [r for r in rigs if rig_agg[r]["paired_foot_slide"]["n"] > 0]
        o["paired_foot_slide"] = {"n_rigs": len(pr), "n_clips": int(sum(rig_agg[r]["paired_foot_slide"]["n"] for r in pr)),
                                  "gen": mean_or_null([rig_agg[r]["paired_foot_slide"]["gen"] for r in pr])[0],
                                  "gt": mean_or_null([rig_agg[r]["paired_foot_slide"]["gt"] for r in pr])[0]}
        return o
    trees = sorted({c["tree"] for c in per_clip.values()})
    out = {"report": a.report, "gen_ckpt": p["gen_ckpt"], "gen_epoch": p.get("gen_epoch"), "cohort_exclude_clips": excl,
           "normalization": base.normalization, "fps": fps, "played_frames": f"min(T_target, {EVAL_MAX_FRAMES})",
           "definitions": {"fk_pose_gap": "mean |FK(rotations) - direct| / mean rest bone length, codec decode_ktjd17 strict_gt=False (samples)",
                           "bone_len_err": "mean |direct bone length - rest bone length| / bl",
                           "leaf_global_rest_delta_deg": "mean rotation angle of the GLOBAL rest delta G R_rest^T over leaf joints (includes global orientation; not local articulation)",
                           "foot_slide_bl_s": "mean horizontal speed (bl/s) of direct positions over (t, joint) pairs with contact flag > 0.5 at t, interval t->t+1; null if no pair",
                           "root_disp_bl": "NET endpoint displacement of the smooth root track over the played frames / bl (not a path length or speed)",
                           "weighting": "clip_weighted = every clip equal (large rigs dominate); rig_equal = every rig's mean equal"},
           "cohort": {"clip_weighted": agg(list(per_clip.values())), "rig_equal": rig_equal(sorted(by_rig))},
           "by_tree": {t: {"clip_weighted": agg([c for c in per_clip.values() if c["tree"] == t]),
                           "rig_equal": rig_equal(sorted(r for r in by_rig if rig_agg[r]["tree"] == t))} for t in trees},
           "by_rig": rig_agg,
           "per_clip": per_clip}
    Path(a.out).write_text(json.dumps(out, indent=1))
    def line(tag, c, e):
        ps = c["paired_foot_slide"]
        print(f"[geom] {tag}: n={c['n']} clip-weighted | FK-pose gap {c['gen_fk_pose_gap']:.3f} bl | bone-length err gen {c['gen_bone_len_err']:.3f} vs data {c['gt_bone_len_err']:.3f} | "
              f"leaf global rest delta gen {c['gen_leaf_global_rest_delta_deg']:.1f} vs data {c['gt_leaf_global_rest_delta_deg']:.1f} deg | "
              f"foot slide (paired, n={ps['n']}) gen {ps['gen']:.2f} vs data {ps['gt']:.2f} bl/s | root net disp gen {c['gen_root_disp_bl']:.2f} vs data {c['gt_root_disp_bl']:.2f} bl | "
              f"played {c['played_frames']['mean']:.0f} frames ({c['played_frames']['truncated_to_240']} truncated) | degenerate 6D {c['gen_degenerate_6d_cells']}")
        pe = e["paired_foot_slide"]
        print(f"[geom] {tag}: rig-equal ({e['n_rigs']} rigs) | gap {e['gen_fk_pose_gap']:.3f} | bone err {e['gen_bone_len_err']:.3f}/{e['gt_bone_len_err']:.3f} | "
              f"leaf delta {e['gen_leaf_global_rest_delta_deg']:.1f}/{e['gt_leaf_global_rest_delta_deg']:.1f} | "
              f"foot slide (paired, {pe['n_rigs']} rigs / {pe['n_clips']} clips) {pe['gen']:.2f}/{pe['gt']:.2f} | root {e['gen_root_disp_bl']:.2f}/{e['gt_root_disp_bl']:.2f}")
    line("cohort", out["cohort"]["clip_weighted"], out["cohort"]["rig_equal"])
    for t, v in out["by_tree"].items():
        line(t, v["clip_weighted"], v["rig_equal"])
    print(f"[geom] wrote {a.out}")


if __name__ == "__main__":
    main()
