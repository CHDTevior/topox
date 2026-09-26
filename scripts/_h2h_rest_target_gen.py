"""Same-data-id head-to-head, one renderer: UniMate's rest / target / generated vs ours, for rigs both corpora hold.

Three subcommands, because UniMate's feature decoders import its own `Quaternions` / `Animation` modules:

  export   (UniMate venv, cwd = UniMate repo root, PYTHONPATH = UniMate repo root)
           rest   = cond.npy[obj]['tpos_first_frame']                    (UniMate's T-pose condition)
           target = dataset/features/objaverse/motions/<official_id>-000.npz 'global_positions' (exact name), re-grounded
                    on the clip's own minimum Y at load when the run's config has dataset.ground_motion_height (the
                    convention UniMate's loader trains in; the saved npz is grounded over the whole source motion)
           gen    = <h2h_dir>/out/motions/<obj>-<tag>-rep_<r>-<i>.npy decoded with UniMate's
                    recover_unimate_joint_pos_from_ric (position channels) and _from_rot (FK, with the offsets
                    UniMate's own loader derives from the grounded T-pose)
           -> <h2h_dir>/unimate_positions.npz

  render / metrics   (our python; both go through load_case)
           ours from the renderer's --dump_world files (gt_w / gen_ric / gen_fk), located by CLIP id (a rig may hold
           several held-out clips); our rest pose from the rig's skeleton file (<ktjd_root>/skeletons/<rig>.npz), not
           the dump's demo_w: under --rep_norm rest the demo frame is all zeros and demo_w is its decode, which sets
           every exact-constant cell to the data's constant, so a joint that never moves lands at its animation pose.
           A case is refused unless our dump's clip maps (manifest official_id) to
           the same source clip as the UniMate case, the captions are identical, the joint name sets and trees
           agree, and UniMate's rest (divided by its scale_factor) equals our rest. UniMate positions are divided by
           scale_factor (it rescales each object to diameter 2) and reordered to our joints by name. UniMate turns
           every motion to face +Z at its own frame 0 while we keep the heading the motion has relative to its
           rest, so one yaw about +Y (plus a vertical offset), fitted on the two target GTs' frame 0, is applied to
           UniMate's target and generated sequences (never its rest); the case is refused if the fitted targets
           disagree on frame 0 or over all overlapping frames (a time offset from UniMate's static-frame trimming).
           render: both rows share one scale; a sequence shorter than the GIF holds its last frame, drawn grey and
           titled "(ended)". metrics: per-case numbers for both sides against their own GT over the HEAD window
           (min of UniMate's generated length and both GTs, 60 frames in practice) plus ours over its full length;
           CSV + summary with medians and win counts.

usage:
  PYTHONPATH=<UniMate> $V/bin/python <repo>/scripts/_h2h_rest_target_gen.py export --h2h_dir <dir> --cases <json> \
      --official_ids <json> --cond <cond.npy> --motions <dir> --unimate_config <run config json>
  python scripts/_h2h_rest_target_gen.py render  --h2h_dir <dir> --cases <json> --official_ids <json> \
      --ours_world <dir> --out <dir> [--rep 0]
  python scripts/_h2h_rest_target_gen.py metrics --h2h_dir <dir> --cases <json> --official_ids <json> \
      --ours_world <dir> --out <dir> [--rep 0]
"""
from __future__ import annotations
import argparse, csv, glob, json, re, sys
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[1]
ENDED = (178, 178, 178)


def load_cases(path):
    """{"<obj>-<tag>": caption} -> [(obj, tag, caption)]"""
    raw = json.loads(Path(path).read_text())
    out = []
    for key, cap in raw.items():
        obj, _, tag = key.partition("-")
        out.append((obj, tag, cap))
    tags = [t for _, t, _ in out]
    if len(set(tags)) != len(tags):
        raise SystemExit("[refuse] case tags must be unique (they name the output GIFs)")
    return out


def cmd_export(a):
    from unimate.utils.motion_utils import recover_unimate_joint_pos_from_ric, recover_unimate_joint_pos_from_rot
    from Animation import offsets_from_positions
    cond = np.load(a.cond, allow_pickle=True).item()
    oid = json.loads(Path(a.official_ids).read_text())            # {"<obj>-<tag>": official_id}
    # UniMate's loader re-grounds every clip on its own minimum Y when dataset.ground_motion_height is set
    # (unimate/dataset/mixture/dataset.py:585-587), so the model samples in that convention while the saved
    # npz targets are grounded once over the WHOLE source motion (before trimming / the 200-frame cut). The flag
    # is recorded here and applied to the target in load_case (review 2026-09-22 round 4).
    gmh = bool(json.loads(Path(a.unimate_config).read_text())["dataset"]["ground_motion_height"])
    store = {"__ground_motion_height": np.array(gmh), "__unimate_config": np.array(str(a.unimate_config))}
    for obj, tag, cap in load_cases(a.cases):
        c = cond[obj]
        key = f"{obj}-{tag}"
        gt_path = Path(a.motions) / f"{oid[key]}-000.npz"
        if not gt_path.is_file():
            raise SystemExit(f"[refuse] {key}: no UniMate clip {gt_path.name}")
        g = np.load(gt_path)
        gens = []
        for f in glob.glob(str(Path(a.h2h_dir) / "out" / "motions" / f"{key}-rep_*-*.npy")):
            m = re.fullmatch(re.escape(key) + r"-rep_(\d+)-\d+\.npy", Path(f).name)
            if m:
                gens.append((int(m.group(1)), f))
        gens.sort()
        if not gens:
            raise SystemExit(f"[refuse] no generated sample for {key}")
        if len({r for r, _ in gens}) != len(gens):
            raise SystemExit(f"[refuse] {key}: duplicate rep numbers among {[Path(f).name for _, f in gens]}")
        parents = np.asarray(c["parents"])
        # FK offsets exactly as UniMate's own loader builds them for sampling/visualization
        # (unimate/dataset/mixture/dataset.py:543-555): from the GROUNDED T-pose positions, not cond['offsets']
        tpos = np.asarray(c["tpos_first_frame"], np.float64).copy()
        tpos[..., 1] -= tpos[..., 1].min()
        offsets = offsets_from_positions(tpos, parents)
        ric, fk = [], []
        for _, f in gens:
            m = np.load(f)                                           # (T, J, 12), de-normalized features
            ric.append(recover_unimate_joint_pos_from_ric(m.copy()))
            fk.append(recover_unimate_joint_pos_from_rot(m.copy(), parents, offsets))
        store[f"{key}__rest"] = np.asarray(c["tpos_first_frame"], np.float64)
        store[f"{key}__target"] = np.asarray(g["global_positions"], np.float64)
        store[f"{key}__gen_ric"] = np.stack(ric)
        store[f"{key}__gen_fk"] = np.stack(fk)
        store[f"{key}__reps"] = np.array([r for r, _ in gens])
        store[f"{key}__names"] = np.array([str(x) for x in c["joint_names"]])
        store[f"{key}__parents"] = parents
        store[f"{key}__scale"] = np.array(float(c["scale_factor"]))
        store[f"{key}__fps"] = np.array(int(g["fps"]))
        store[f"{key}__clip"] = np.array(gt_path.name)
        print(f"{key}: target {store[key + '__target'].shape}  gen {store[key + '__gen_ric'].shape} reps "
              f"{[r for r, _ in gens]}  scale {float(c['scale_factor']):.4f}  clip {gt_path.name}", flush=True)
    np.savez(Path(a.h2h_dir) / "unimate_positions.npz", **store)


def ground_center(P):
    """[J,3] -> root XZ at 0, lowest joint at y=0."""
    return P - np.array([P[0, 0], P[:, 1].min(), P[0, 2]])


def yaw(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def open_side(a):
    """Everything the per-case loader needs that is shared by all cases."""
    U = np.load(Path(a.h2h_dir) / "unimate_positions.npz", allow_pickle=False)
    oid = json.loads(Path(a.official_ids).read_text())
    man = {}
    for line in open(a.manifest):
        r = json.loads(line)
        man[r["clip_id"]] = r["official_id"]
    clip_of = {o: c for c, o in man.items()}                      # official_id -> our clip id
    return U, oid, man, clip_of


def load_case(a, U, oid, man, clip_of, obj, tag, cap):
    """One case, both sides, in OUR frame and units. Refuses on every mismatch listed in the module docstring."""
    key = f"{obj}-{tag}"
    # our dump is located by CLIP id, not by rig: a rig may hold several held-out clips
    clip = clip_of.get(oid[key])
    if clip is None:
        raise SystemExit(f"[refuse] {key}: {oid[key]} is not in our manifest")
    hits = sorted(glob.glob(str(Path(a.ours_world) / f"*_OBJ_{obj}__{clip}.world.npz")))
    if len(hits) != 1:
        raise SystemExit(f"[refuse] {key}: expected one world dump for OBJ_{obj}__{clip}, found {len(hits)}")
    w = np.load(hits[0], allow_pickle=True)
    ours_oid = man.get(str(w["motion_id"]))
    if ours_oid != oid[key]:
        raise SystemExit(f"[refuse] {key}: our clip {w['motion_id']} is {ours_oid}, UniMate case is {oid[key]}")
    if str(w["caption"]) != cap:
        raise SystemExit(f"[refuse] {key}: our caption {str(w['caption'])!r} != case caption {cap!r}")
    names = [str(x) for x in w["joint_names"]]
    par = np.asarray(w["parents"])
    un = [str(x) for x in U[f"{key}__names"]]
    if sorted(un) != sorted(names):
        raise SystemExit(f"[refuse] {key}: joint name sets differ")
    perm = [un.index(n) for n in names]
    up = np.asarray(U[f"{key}__parents"])
    uname_parent = {un[i]: (un[int(up[i])] if int(up[i]) >= 0 else None) for i in range(len(un))}
    if any(uname_parent[names[j]] != (names[int(par[j])] if int(par[j]) >= 0 else None) for j in range(len(names))):
        raise SystemExit(f"[refuse] {key}: joint trees differ")
    s = float(U[f"{key}__scale"])
    reps = [int(r) for r in U[f"{key}__reps"]]
    if a.rep not in reps:
        raise SystemExit(f"[refuse] {key}: rep {a.rep} not among {reps}")
    ri = reps.index(a.rep)
    u_rest = ground_center(U[f"{key}__rest"][perm] / s)
    sk_path = REPO / str(w["ktjd_root"]) / "skeletons" / f"{w['rig']}.npz"      # our rest: the skeleton file (docstring)
    sk = np.load(sk_path, allow_pickle=True)
    if [str(x) for x in sk["joint_names"]] != names:
        raise SystemExit(f"[refuse] {key}: {sk_path.name} lists the joints in another order than our dump")
    o_rest = ground_center(np.asarray(sk["P_rest_global"], np.float64))
    diam = float(np.max(np.linalg.norm(o_rest[:, None] - o_rest[None], axis=-1)))
    if np.abs(u_rest - o_rest).max() > 1e-4 * diam:
        raise SystemExit(f"[refuse] {key}: rest poses differ by {np.abs(u_rest - o_rest).max() / diam:.2e} D")
    o_tgt = np.asarray(w["gt_w"], np.float64)
    u_tgt = U[f"{key}__target"][:, perm] / s
    if "__ground_motion_height" not in U.files:
        raise SystemExit("[refuse] unimate_positions.npz predates the grounding fix -- re-run export with --unimate_config")
    if bool(U["__ground_motion_height"]):
        # the convention the model was trained and sampled in: the saved clip's own lowest point at y=0
        u_tgt = u_tgt - np.array([0.0, float(u_tgt[..., 1].min()), 0.0])
    u_ric = U[f"{key}__gen_ric"][ri][:, perm] / s
    u_fk = U[f"{key}__gen_fk"][ri][:, perm] / s
    # UniMate's frame convention -> ours, fitted on the two target GTs' frame 0 (root-centred)
    A = u_tgt[0] - u_tgt[0, 0] * [1, 0, 1]
    B = o_tgt[0] - o_tgt[0, 0] * [1, 0, 1]
    th = float(np.arctan2(np.sum(B[:, 0] * A[:, 2] - B[:, 2] * A[:, 0]), np.sum(B[:, 0] * A[:, 0] + B[:, 2] * A[:, 2])))
    R = yaw(th)
    dy = float(np.mean(B[:, 1] - (A @ R.T)[:, 1]))
    fit = float(np.linalg.norm(A @ R.T + [0, dy, 0] - B, axis=-1).mean() / diam)
    if fit > a.max_fit:
        raise SystemExit(f"[refuse] {key}: target frame 0 still differs by {fit:.3f} D after the yaw fit")

    def to_ours(X):
        X0 = X - X[0, 0] * [1, 0, 1]
        return X0 @ R.T + [0, dy, 0] + o_tgt[0, 0] * [1, 0, 1]
    u_tgt, u_ric, u_fk = to_ours(u_tgt), to_ours(u_ric), to_ours(u_fk)
    # frame 0 alone cannot see a time offset (UniMate trims static lead-in frames): the fitted targets must agree
    # over every frame both have
    ov = min(len(u_tgt), len(o_tgt))
    full_fit = float(np.linalg.norm(u_tgt[:ov] - o_tgt[:ov], axis=-1).mean() / diam)
    if full_fit > a.max_fit:
        raise SystemExit(f"[refuse] {key}: targets differ by {full_fit:.3f} D over their {ov} overlapping frames "
                         f"(time offset or different clip)")
    fps = int(U[f"{key}__fps"])
    if abs(float(w["fps"]) - fps) > 1e-6:
        raise SystemExit(f"[refuse] {key}: fps differs (ours {float(w['fps'])}, UniMate {fps})")
    seqs = {"u_rest": u_rest[None], "u_tgt": u_tgt, "u_ric": u_ric, "u_fk": u_fk,
            "o_rest": o_rest[None], "o_tgt": o_tgt,
            "o_ric": np.asarray(w["gen_ric"], np.float64), "o_fk": np.asarray(w["gen_fk"], np.float64)}
    return dict(key=key, obj=obj, tag=tag, cap=cap, clip=clip, par=par, diam=diam, seqs=seqs,
                th=th, dy=dy, fit=fit, full_fit=full_fit, ov=ov, fps=fps)


def cmd_render(a):
    sys.path.insert(0, str(REPO / "scripts"))
    from _pil_skeleton_render import compute_transform, make_row_frame, save_gif
    from PIL import Image, ImageDraw
    U, oid, man, clip_of = open_side(a)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    for obj, tag, cap in load_cases(a.cases):
        c = load_case(a, U, oid, man, clip_of, obj, tag, cap)
        seqs, par, diam, fps = c["seqs"], c["par"], c["diam"], c["fps"]
        n = {k: len(v) for k, v in seqs.items()}
        T = min(max(n.values()), a.max_frames)
        stride = 2 if T > 120 else 1
        cell = (a.cell, int(a.cell * 0.94))
        # every sequence padded to T by holding its last frame, so one frame index drives all panels and each
        # panel keeps its full root trail
        seqs = {k: (v[:T] if len(v) >= T else np.concatenate([v, np.repeat(v[-1:], T - len(v), axis=0)]))
                for k, v in seqs.items()}
        # one scale for both rows: fit every panel as it will be drawn (root-centred per frame)
        cen = []
        for v in seqs.values():
            cc = v.copy(); cc[..., 0] -= cc[:, :1, 0]; cc[..., 2] -= cc[:, :1, 2]; cen.append(cc)
        tr = compute_transform(cen, cell, 0.10, 1.0)
        L = {k: (f"{T}/{v}f" if v > T else f"{v}f") for k, v in n.items()}   # shown length (of total, if cut)
        rows = [
            ("UniMate", [("rest (tpos)", "u_rest", (13, 110, 100), True), (f"target GT {L['u_tgt']}", "u_tgt", (20, 20, 20), False),
                         (f"generated ric {L['u_ric']}", "u_ric", (176, 61, 8), False), (f"generated fk {L['u_fk']}", "u_fk", (120, 40, 140), False)]),
            ("ours", [("rest (demo)", "o_rest", (13, 110, 100), True), (f"target GT {L['o_tgt']}", "o_tgt", (20, 20, 20), False),
                      (f"generated ric {L['o_ric']}", "o_ric", (176, 61, 8), False), (f"generated fk {L['o_fk']}", "o_fk", (120, 40, 140), False)]),
        ]
        frames = []
        for fi in range(0, T, stride):
            imgs = []
            for label, panels in rows:
                pl = []
                for title, k, col, static in panels:
                    ended = (not static) and fi >= n[k]
                    pl.append({"positions": seqs[k], "parents": par, "color": ENDED if ended else col, "axes": False,
                               "title": f"{label}: {title}" + (" (ended)" if ended else ""), "static": static})
                imgs.append(make_row_frame(pl, fi, tr, cell, 3, 5))
            W = imgs[0].width
            canvas = Image.new("RGB", (W, sum(i.height for i in imgs) + 40), "white")
            ImageDraw.Draw(canvas).text(
                (12, 10), f"{obj}  |  \"{cap}\"  |  frame {fi}  |  UniMate rows turned {np.degrees(c['th']):+.1f} deg about +Y "
                          f"to our heading (target frame-0 fit {c['fit']:.3f} D)", fill=(20, 20, 20))
            y = 40
            for i in imgs:
                canvas.paste(i, (0, y)); y += i.height
            frames.append(canvas)
        save_gif(frames, out / f"{tag}.gif", fps / stride)
        print(f"{tag}: {len(frames)} frames @ {fps / stride:g} fps  yaw {np.degrees(c['th']):+.1f} deg  dy {c['dy'] / diam:+.3f} D  "
              f"fit frame0 {c['fit']:.4f} D  all {c['ov']} overlapping frames {c['full_fit']:.4f} D  (ours target {n['o_tgt']}f gen {n['o_ric']}f, "
              f"UniMate target {n['u_tgt']}f gen {n['u_ric']}f rep {a.rep})  -> {out / (tag + '.gif')}", flush=True)


def kabsch_tilt(A, B):
    """Tilt of the vertical axis (deg) under the best rotation A -> B, both root-centred [J,3]; NaN if A is near-collinear."""
    H = A.T @ B
    Uu, S, Vt = np.linalg.svd(H)
    if S[0] <= 0 or S[1] / S[0] < 1e-3:
        return float("nan")
    D = np.diag([1.0, 1.0, np.sign(np.linalg.det(Vt.T @ Uu.T))])
    R = Vt.T @ D @ Uu.T
    return float(np.degrees(np.arccos(np.clip(R[1, 1], -1, 1))))


def side_metrics(gen, gt, par, T):
    """gen, gt [>=T, J, 3] world positions; the first T frames. Units: mean bone length of GT frame 0."""
    g, t = gen[:T], gt[:T]
    bl = float(np.linalg.norm(t[0, 1:] - t[0, par[1:]], axis=-1).mean()) if len(par) > 1 else 1.0
    acc = lambda w: np.linalg.norm(w[2:] - 2 * w[1:-1] + w[:-2], axis=-1)          # same as v2_render_incontext.jitter_ratio
    ga, ta = (acc(g), acc(t)) if T >= 3 else (None, None)
    jit = float(ga.mean() / max(ta.mean(), 1e-9)) if ga is not None else float("nan")
    gt_acc = float(ta.mean() / bl) if ta is not None else float("nan")
    low_diff = float(np.median(g[..., 1].min(axis=1)) - np.median(t[..., 1].min(axis=1))) / bl   # ground clearance
    root_diff = float(np.median(g[:, 0, 1]) - np.median(t[:, 0, 1])) / bl
    A = t[0] - t[0, 0]; B = g[0] - g[0, 0]
    tilt0 = kabsch_tilt(A, B)
    pose_err = float(np.linalg.norm(g - g[:, :1] - (t - t[:, :1]), axis=-1).mean()) / bl        # root-relative, per joint
    root_xz = float(np.linalg.norm(g[:, 0, [0, 2]] - t[:, 0, [0, 2]], axis=-1).mean()) / bl
    return {"T": T, "bl": bl, "jitter": jit, "gt_acc_bl": gt_acc, "lowest_diff_bl": low_diff, "root_h_diff_bl": root_diff,
            "tilt0_deg": tilt0, "pose_err_bl": pose_err, "root_xz_err_bl": root_xz}


def cmd_metrics(a):
    U, oid, man, clip_of = open_side(a)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    rows, refused = [], []
    for obj, tag, cap in load_cases(a.cases):
        try:
            c = load_case(a, U, oid, man, clip_of, obj, tag, cap)
        except SystemExit as e:
            # --skip_refused: a case that fails its own consistency check is listed with the reason (refused.json and
            # the summary), never dropped silently; a refusal that is not about this case still stops the run
            if not (a.skip_refused and str(e).startswith(f"[refuse] {obj}-{tag}:")):
                raise
            refused.append({"key": f"{obj}-{tag}", "reason": str(e)})
            print(f"[skip] {e}", file=sys.stderr)                 # loud at once: a wrong --manifest/--ours_world shows here
            continue
        sq, par = c["seqs"], c["par"]
        T_head = min(len(sq["u_ric"]), len(sq["u_tgt"]), len(sq["o_tgt"]), len(sq["o_ric"]))
        T_full = min(len(sq["o_ric"]), len(sq["o_tgt"]))
        rec = {"key": c["key"], "obj": obj, "clip": c["clip"], "caption": cap, "J": len(par), "yaw_deg": float(np.degrees(c["th"])),
               "full_fit_D": c["full_fit"], "T_head": T_head, "T_full": T_full}
        for side, ric, fk, gt in (("uni", sq["u_ric"], sq["u_fk"], sq["u_tgt"]), ("ours", sq["o_ric"], sq["o_fk"], sq["o_tgt"])):
            m = side_metrics(ric, gt, par, T_head)
            rec.update({f"{side}_{k}": v for k, v in m.items() if k != "T"})
            bl = m["bl"]
            rec[f"{side}_fk_ric_gap_bl"] = float(np.linalg.norm(fk[:T_head] - ric[:T_head], axis=-1).mean()) / bl
        mf = side_metrics(sq["o_ric"], sq["o_tgt"], par, T_full)
        rec.update({f"ours_full_{k}": v for k, v in mf.items() if k not in ("T", "bl")})
        rows.append(rec)
    if not rows:
        raise SystemExit("[refuse] every case was refused:\n" + "\n".join(r["reason"] for r in refused))
    keys = list(rows[0].keys())
    with open(out / "metrics.csv", "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=keys); wr.writeheader(); wr.writerows(rows)
    json.dump([{k: (None if isinstance(v, float) and not np.isfinite(v) else v) for k, v in r.items()} for r in rows],
              open(out / "metrics.json", "w"), indent=1)                                   # strict JSON: NaN tilt -> null
    if a.skip_refused:                                            # without the flag the outputs are exactly as before
        json.dump(refused, open(out / "refused.json", "w"), indent=1)
    T_gen = sorted({len(U[k + "__gen_ric"][0]) for k in {r["key"] for r in rows}})
    L = ([f"refused {len(refused)} case(s) (listed at the end)"] if a.skip_refused else []) + [
         f"cases {len(rows)}  head window T = {sorted({r['T_head'] for r in rows})[:5]}… frames (UniMate generates {T_gen} frames)",
         "metric (head window, both sides vs their own GT; last row = self-consistency)   UniMate median / mean      ours median / mean      ours closer to target (|.|, of n)"]
    def closer(name, target):
        u = np.array([r[f"uni_{name}"] for r in rows]); o = np.array([r[f"ours_{name}"] for r in rows])
        ok = np.isfinite(u) & np.isfinite(o)
        du, do = np.abs(u[ok] - target), np.abs(o[ok] - target)
        if name == "jitter":
            du, do = np.abs(np.log(np.maximum(u[ok], 1e-9))), np.abs(np.log(np.maximum(o[ok], 1e-9)))
        return u[ok], o[ok], int((do < du).sum()), int(ok.sum())
    for name, target, label in (("jitter", 1.0, "jitter ratio gen/GT (1 = as smooth as GT)"),
                                ("lowest_diff_bl", 0.0, "lowest-joint height - GT (bl)"),
                                ("root_h_diff_bl", 0.0, "root height - GT (bl)"),
                                ("tilt0_deg", 0.0, "frame-0 tilt vs GT (deg)"),
                                ("pose_err_bl", 0.0, "root-relative pose error (bl)"),
                                ("root_xz_err_bl", 0.0, "root XZ path error (bl)"),
                                ("fk_ric_gap_bl", 0.0, "FK vs position-channel gap (bl)")):
        u, o, wins, n = closer(name, target)
        L.append(f"{label:52s} {np.median(u):8.3f} / {u.mean():7.3f}   {np.median(o):8.3f} / {o.mean():7.3f}      {wins} / {n}")
    of = [r for r in rows]
    L.append("ours over its FULL length: " + "  ".join(          # nanmedian: tilt is NaN on near-collinear rigs
        f"{k} med {np.nanmedian([r['ours_full_' + k] for r in of]):.3f}" for k in ("jitter", "lowest_diff_bl", "root_h_diff_bl", "tilt0_deg", "pose_err_bl")))
    L.append("floating tail (lowest-joint height - GT > 0.3 bl): UniMate %d, ours(head) %d, ours(full) %d of %d" % (
        sum(r["uni_lowest_diff_bl"] > 0.3 for r in rows), sum(r["ours_lowest_diff_bl"] > 0.3 for r in rows),
        sum(r["ours_full_lowest_diff_bl"] > 0.3 for r in rows), len(rows)))
    L.append("orientation tail (frame-0 tilt vs GT > 45 deg): UniMate %d, ours %d of %d" % (
        sum(r["uni_tilt0_deg"] > 45 for r in rows if np.isfinite(r["uni_tilt0_deg"])),
        sum(r["ours_tilt0_deg"] > 45 for r in rows if np.isfinite(r["ours_tilt0_deg"])), len(rows)))
    L += [f"refused: {r['reason']}" for r in refused]
    (out / "summary.txt").write_text("\n".join(L) + "\n")
    print("\n".join(L))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("export")
    e.add_argument("--h2h_dir", required=True); e.add_argument("--cases", required=True)
    e.add_argument("--official_ids", required=True); e.add_argument("--cond", required=True); e.add_argument("--motions", required=True)
    e.add_argument("--unimate_config", required=True, help="the trained run's config JSON (dataset.ground_motion_height is read)")
    for name in ("render", "metrics"):
        r = sub.add_parser(name)
        r.add_argument("--h2h_dir", required=True); r.add_argument("--cases", required=True)
        r.add_argument("--official_ids", required=True)
        r.add_argument("--manifest", default=str(REPO / "dataset/ktjd17_uniml3d_v2/manifests/clips.jsonl"))
        r.add_argument("--ours_world", required=True); r.add_argument("--out", required=True)
        r.add_argument("--rep", type=int, default=0)
        r.add_argument("--max_fit", type=float, default=0.05, help="refuse when the yaw-fitted target frames differ by more (D)")
        if name == "render":
            r.add_argument("--max_frames", type=int, default=240); r.add_argument("--cell", type=int, default=380)
        else:
            r.add_argument("--skip_refused", action="store_true",
                           help="list a case that fails its consistency checks (with the reason) and go on, instead of stopping")
    a = ap.parse_args()
    {"export": cmd_export, "render": cmd_render, "metrics": cmd_metrics}[a.cmd](a)


if __name__ == "__main__":
    main()
