"""Zero-clip per-cell statistics for an unseen rig, BORROWED from the nearest library species.

Zero-shot item 1 (user 2026-09-07): our deploy convention supplies a new rig's per-(rig, joint, channel) mean/std with
its own clips (transductive). This tool builds the clip-free alternative: for a target rig T (TrueBones view) and a donor
rig D (library, per-cell stats of the training corpus), every joint of T is matched to the donor joint whose LLM2Vec joint
description is most similar (cosine), and T's mean/std are taken from D's matched joint. The root row is always matched
to the donor's root row (root-only channels 13:17 live there). Position-like channels (q_position 0:3, velocity 9:12,
smooth-root 13:15) are rescaled by s_rig(T)/s_rig(D), the KTJD per-rig scale (AABB diagonal of the rest pose), so the
borrowed amplitudes follow the target's body size; rotation deltas, contact and heading are scale-free and copied as is.
The rescaling acts on the EFFECTIVE divisor std + _STD_FLOOR (codex r1).

WHAT IS AND IS NOT BORROWED (codex r1): this is a "borrowed mean/std with ORACLE target masks" arm. supervise_mask /
was_constant / was_floored stay the TARGET's own (they come from the target's clips and decide which cells the sampler
holds at the constant), and cells the target holds CONSTANT keep the target's own mean/std too (the constant must decode
to itself: borrowing a donor value there would move a masked cell -- Gazelle's hand contact decoded to 0.70 instead of 0
before this fix). Where the DONOR cell is a constant-excluded placeholder (std 1, was_constant) but the target cell is
active, the donor value is meaningless; such cells take the donor rig's MEDIAN (mean, std) over its active cells of the
same channel, and their count is recorded per rig. Every non-target rig keeps its original statistics. The borrowing
table (joint pairs, cosines, scale ratio, fallback counts) is written next to the npz for the report.

usage: python scripts/_build_borrowed_rig_stats.py --target_stats data/tb_norm_stats_v2_mainbody.npz \
          --target_root dataset/ktjd17_truebones_lora_v2_mainbody --target_sem data/joint_semantics_llm2vec_ktjd17_v1_mainbody.npz \
          --donor_stats data/noik_norm_stats_v2.npz --donor_root dataset/ktjd17_pzh312_noik_v2 \
          --donor_sem data/joint_semantics_llm2vec_pzh312_v1.npz --pair Buffalo=PZ_African_Buffalo_Male \
          --pair Gazelle=PZ_Thomsons_Gazelle_Male --out data/tb_norm_stats_borrowed_v1.npz [--no_rescale]
"""
from __future__ import annotations
import argparse, json, hashlib
from pathlib import Path
import numpy as np

SCALED_CH = list(range(0, 3)) + list(range(9, 12))   # per-joint position-like channels
ROOT_SCALED_CH = [13, 14]                              # smooth-root x,z (root row only)


def load_stacked(path: Path):
    with np.load(path, allow_pickle=False) as z:
        meta = json.loads(str(z["__meta"]))
        d = {k: np.asarray(z[k]) for k in z.files if k != "__meta"}
    rids = [str(r) for r in d["rig_ids"]]
    return d, rids, meta


def s_rig_of(root: Path, rig: str) -> float:
    z = np.load(root / "skeletons" / f"{rig}.npz", allow_pickle=True)
    if "s_rig" not in z.files:
        raise SystemExit(f"[refuse] {root}/skeletons/{rig}.npz has no s_rig")
    s = float(z["s_rig"])
    if not (np.isfinite(s) and s > 0):
        raise SystemExit(f"[refuse] s_rig of {rig} is {s}")
    return s


def joint_names_of(root: Path, rig: str) -> list[str]:
    z = np.load(root / "skeletons" / f"{rig}.npz", allow_pickle=True)
    return [str(n) for n in z["joint_names"]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target_stats", required=True)
    ap.add_argument("--target_root", required=True)
    ap.add_argument("--target_sem", required=True)
    ap.add_argument("--donor_stats", required=True)
    ap.add_argument("--donor_root", required=True)
    ap.add_argument("--donor_sem", required=True)
    ap.add_argument("--pair", action="append", required=True, metavar="TARGET=DONOR")
    ap.add_argument("--out", required=True)
    ap.add_argument("--no_rescale", action="store_true", help="copy donor amplitudes without the s_rig ratio")
    a = ap.parse_args()
    out = Path(a.out)
    if out.exists():
        raise SystemExit(f"[refuse] {out} exists -- move it away first (never overwrite a stats artifact)")

    T, t_rids, t_meta = load_stacked(Path(a.target_stats))
    D, d_rids, d_meta = load_stacked(Path(a.donor_stats))
    for k in ("rig_ids", "joint_count", "mean", "std", "supervise_mask", "was_constant", "was_floored"):
        if k not in T:
            raise SystemExit(f"[refuse] target stats lack key {k}")
        if k not in D:
            raise SystemExit(f"[refuse] donor stats lack key {k}")
    if T["mean"].shape[-1] != 17 or D["mean"].shape[-1] != 17:
        raise SystemExit("[refuse] stats are not [R,J,17]")
    if float(t_meta.get("std_floor", -1)) != float(d_meta.get("std_floor", -2)):
        raise SystemExit(f"[refuse] std_floor differs: target {t_meta.get('std_floor')} donor {d_meta.get('std_floor')}")
    if str(t_meta.get("convention")) != str(d_meta.get("convention")):
        raise SystemExit("[refuse] normalization conventions differ between target and donor stats")
    t_sem = np.load(a.target_sem, allow_pickle=False)
    d_sem = np.load(a.donor_sem, allow_pickle=False)

    mean, std = T["mean"].copy(), T["std"].copy()
    table = {}
    for spec in a.pair:
        if "=" not in spec:
            raise SystemExit(f"[refuse] --pair wants TARGET=DONOR, got {spec!r}")
        trig, drig = spec.split("=", 1)
        if trig not in t_rids or drig not in d_rids:
            raise SystemExit(f"[refuse] unknown rig in pair {spec!r} (target known: {trig in t_rids}, donor known: {drig in d_rids})")
        if trig in table:
            raise SystemExit(f"[refuse] target {trig} listed twice")
        ti, di = t_rids.index(trig), d_rids.index(drig)
        Jt, Jd = int(T["joint_count"][ti]), int(D["joint_count"][di])
        et, ed = np.asarray(t_sem[f"emb__{trig}"], np.float64), np.asarray(d_sem[f"emb__{drig}"], np.float64)
        if et.shape[0] != Jt or ed.shape[0] != Jd:
            raise SystemExit(f"[refuse] description table / stats joint count mismatch: {trig} {et.shape[0]} vs {Jt}, "
                             f"{drig} {ed.shape[0]} vs {Jd}")
        en = et / np.linalg.norm(et, axis=1, keepdims=True)
        dn = ed / np.linalg.norm(ed, axis=1, keepdims=True)
        S = en @ dn.T
        match = S.argmax(1)
        match[0] = 0                                   # root row <- donor root row (root-only channels live there)
        cos = S[np.arange(Jt), match]
        ratio = 1.0 if a.no_rescale else s_rig_of(Path(a.target_root), trig) / s_rig_of(Path(a.donor_root), drig)
        floor = float(t_meta["std_floor"])
        d_mu_all = np.asarray(D["mean"][di, :Jd, :], np.float64)
        d_sd_all = np.asarray(D["std"][di, :Jd, :], np.float64)
        d_const = np.asarray(D["was_constant"][di, :Jd, :], bool)
        d_sup = np.asarray(D["supervise_mask"][di, :Jd, :], bool)
        d_active = d_sup & ~d_const                     # cells that carry real donor statistics (codex r2: structural placeholders are not a pool)
        t_const = np.asarray(T["was_constant"][ti, :Jt, :], bool)
        t_sup = np.asarray(T["supervise_mask"][ti, :Jt, :], bool)
        mu = d_mu_all[match, :].copy()
        sd = d_sd_all[match, :].copy()
        # donor placeholder cells (constant-excluded: std 1, mean = constant) landing on ACTIVE target cells carry no
        # amplitude information -> donor per-channel median over its own active cells (codex r1)
        donor_placeholder = (~d_active[match, :]) & t_sup & ~t_const
        n_fallback = 0
        for c in range(17):
            act = d_active[:, c]
            if donor_placeholder[:, c].any():
                if not act.any():
                    raise SystemExit(f"[refuse] donor {drig} has no active (supervised, non-constant) cell on channel {c} to fall back to")
                mu[donor_placeholder[:, c], c] = float(np.median(d_mu_all[act, c]))
                sd[donor_placeholder[:, c], c] = float(np.median(d_sd_all[act, c]))
                n_fallback += int(donor_placeholder[:, c].sum())
        for c in SCALED_CH:
            mu[:, c] *= ratio
            sd[:, c] = ratio * (sd[:, c] + floor) - floor
        for c in ROOT_SCALED_CH:
            mu[0, c] *= ratio
            sd[0, c] = ratio * (sd[0, c] + floor) - floor
        # target-constant / unsupervised cells keep the target's own (mean = the constant, std = placeholder): oracle masks
        keep_own = t_const | ~t_sup
        t_mu_own = np.asarray(T["mean"][ti, :Jt, :], np.float64)
        t_sd_own = np.asarray(T["std"][ti, :Jt, :], np.float64)
        mu[keep_own] = t_mu_own[keep_own]
        sd[keep_own] = t_sd_own[keep_own]
        n_borrowed = int((~keep_own).sum())
        # rows beyond the target's joint count stay padding (zeros / whatever the target file had)
        mean[ti, :Jt] = mu.astype(np.float32)
        std[ti, :Jt] = sd.astype(np.float32)
        tn, dnames = joint_names_of(Path(a.target_root), trig), joint_names_of(Path(a.donor_root), drig)
        table[trig] = {"donor": drig, "s_rig_ratio": float(ratio), "rescaled_channels": SCALED_CH + ROOT_SCALED_CH,
                       "arm": "borrowed mean/std with ORACLE target masks (target-constant / unsupervised cells keep the target's own values)",
                       "n_cells_borrowed": n_borrowed, "n_cells_kept_own": int(keep_own.sum()),
                       "n_cells_donor_placeholder_fallback_to_channel_median": n_fallback,
                       "cos_min": float(cos.min()), "cos_median": float(np.median(cos)),
                       "distinct_donor_joints": int(len(set(match.tolist()))), "J_target": Jt, "J_donor": Jd,
                       "pairs": [{"t": int(j), "t_name": tn[j], "d": int(match[j]), "d_name": dnames[int(match[j])],
                                  "cos": float(cos[j])} for j in range(Jt)]}
        print(f"[borrow] {trig}(J={Jt}) <- {drig}(J={Jd}): cos min/median {cos.min():.3f}/{np.median(cos):.3f}, "
              f"{len(set(match.tolist()))} distinct donor joints, s_rig ratio {ratio:.3f}; cells borrowed {n_borrowed}, "
              f"kept own (constant/unsupervised) {int(keep_own.sum())}, donor-placeholder fallbacks {n_fallback}")

    floor = float(t_meta["std_floor"])
    for ti, r in enumerate(t_rids):
        J = int(T["joint_count"][ti])
        m_, s_ = mean[ti, :J], std[ti, :J]
        if not (np.isfinite(m_).all() and np.isfinite(s_).all()) or float((s_ + floor).min()) <= 0.0:
            raise SystemExit(f"[refuse] borrowed stats for {r} are not finite / positive")
    meta = dict(t_meta)
    meta.update({"borrowed_from": {k: v["donor"] for k, v in table.items()},
                 "borrowed_note": "borrowed mean/std with ORACLE target masks: per-cell mean/std of the listed rigs' SUPERVISED, "
                                  "non-constant cells are the nearest library species' statistics matched joint-by-joint on LLM2Vec joint "
                                  "descriptions (root row -> donor root row), position-like channels rescaled by the s_rig ratio on the "
                                  "effective divisor; target-constant / unsupervised cells and supervise_mask/was_constant/was_floored "
                                  "are the target's own; donor placeholder cells fall back to the donor's per-channel median. Built by "
                                  "scripts/_build_borrowed_rig_stats.py",
                 "donor_stats_sha256": hashlib.sha256(Path(a.donor_stats).read_bytes()).hexdigest(),
                 "target_stats_sha256": hashlib.sha256(Path(a.target_stats).read_bytes()).hexdigest(),
                 "rescale": not a.no_rescale})
    payload = {k: v for k, v in T.items()}
    payload["mean"], payload["std"] = mean.astype(np.float32), std.astype(np.float32)
    payload["__meta"] = np.array(json.dumps(meta))
    np.savez(out, **payload)
    Path(str(out) + ".borrow_table.json").write_text(json.dumps(table, indent=1))
    print(f"[OK] wrote {out} ({len(table)} rigs borrowed) + {out}.borrow_table.json")


if __name__ == "__main__":
    main()
