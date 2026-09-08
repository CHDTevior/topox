"""Tests of the KTJD-17 <-> AnyTop-13 conversion (src/data/ktjd17_anytop13.py) on real PZ clips plus synthetic edge cases.
Round trip KTJD -> 13 -> KTJD must reproduce the STORED release channels: q_position and the root track exactly (the frozen PZ release
stores the un-smoothed root XZ), velocities on every frame but the last (the release looked past the clip cut), contact, heading (valid
frames), non-leaf rest deltas; leaf rest deltas are reported (inherent loss). Independent checks: the native AnyTop FK decoder
(src/data/anytop_rot6d_fk.recover_from_bvh_rot_np) reproduces the world positions from the 13 container (velocities / fps), a known
facing direction maps to +Z, float32 storage round-trips, degenerate generated 6D cells follow the declared fallbacks, conflicting
sibling slots follow last-child-wins, and J = 1 / T = 1 clips convert.
usage: python scripts/_aug_dev/_test_anytop13_roundtrip.py [n_clips]"""
import sys, os, json
import numpy as np
sys.path.insert(0, os.getcwd())
from src.data.ktjd17.codec import decode_column_cont6d, direct_decode_positions, encode_column_cont6d
from src.data.ktjd17.loader import load_motion_npz
from src.data.ktjd17_anytop13 import (ktjd17_to_anytop13, anytop13_to_ktjd17, children_of, facing_from_heading, _ry,
                                      REPRESENTATION_ID)
from src.data.anytop_rot6d_fk import recover_from_bvh_rot_np
ROOT = "dataset/ktjd17_pzh312_noik_v2"
sch = json.load(open(f"{ROOT}/schema.json")); eps_h = float(sch["heading"]["eps_h"]); FPS = 30.0
rows = [json.loads(l) for l in open(f"{ROOT}/manifests/clips.jsonl")]
rows = [r for r in rows if r["status"] == "accept" and r["rig_id"].startswith("PZ_")]     # the animal arm's corpus (humans are cut)
rng = np.random.default_rng(0); pick = rng.choice(len(rows), size=int(sys.argv[1]) if len(sys.argv) > 1 else 40, replace=False)
fails, worst = [], {}
def Rf_expect(a, t):
    """R_face[t]^T decoded from the root slot of a 13 container (for un-rotation checks)."""
    return decode_column_cont6d(a[t, 0, 3:9]).T
def chk(name, ok, val):
    worst[name] = max(worst.get(name, 0.0), float(val))
    if not ok: fails.append(f"{name}: {val:.3e}")
print("representation:", REPRESENTATION_ID)
for i in pick:
    r = rows[int(i)]
    pl = load_motion_npz(f"{ROOT}/{r['motion_relpath']}", expected_fps_target=FPS)
    sk = np.load(f"{ROOT}/skeletons/{r['rig_id']}.npz", allow_pickle=True)
    m17 = np.asarray(pl["motion"], np.float64); hv = np.asarray(pl["heading_valid"], bool); par = np.asarray(sk["parents"], np.int64)
    T, J = m17.shape[:2]
    bl = np.linalg.norm(np.asarray(sk["offset_parent_local"])[1:], axis=-1).mean()
    a13 = ktjd17_to_anytop13(m17, hv, par, fps=FPS)
    assert np.all(a13[:, 1:, 13:17] == 0) and np.all(a13[:, 0, 13:17] == 0) and np.all(a13[:, 0, [0, 2]] == 0) and np.all(a13[:, 0, 10] == 0)
    a13_32 = a13.astype(np.float32).astype(np.float64)                                    # storage precision, as the view stores it
    back, hv2, diag = anytop13_to_ktjd17(a13_32, par, fps=FPS, eps_h=eps_h)
    chk("diag: degenerate cells on GT", diag["degenerate_facing_frames"] == 0 and diag["degenerate_child_slots"] == 0, diag["degenerate_facing_frames"] + diag["degenerate_child_slots"])
    # STORED channels (the release split is the raw root track: reproducible from the clip)
    chk("q_position (bl)", np.abs(back[..., 0:3] - m17[..., 0:3]).max() / bl < 1e-4, np.abs(back[..., 0:3] - m17[..., 0:3]).max() / bl)
    chk("root track ch13:15 (bl)", np.abs(back[:, 0, 13:15] - m17[:, 0, 13:15]).max() / bl < 1e-4, np.abs(back[:, 0, 13:15] - m17[:, 0, 13:15]).max() / bl)
    chk("velocity (frames :-1)", np.abs(back[:-1, :, 9:12] - m17[:-1, :, 9:12]).max() < 1e-3, np.abs(back[:-1, :, 9:12] - m17[:-1, :, 9:12]).max())
    worst["last-frame velocity dev (release looked past the cut)"] = max(worst.get("last-frame velocity dev (release looked past the cut)", 0.0), float(np.abs(back[-1, :, 9:12] - m17[-1, :, 9:12]).max()))
    chk("contact", np.abs(back[..., 12] - m17[..., 12]).max() < 1e-9, np.abs(back[..., 12] - m17[..., 12]).max())
    if hv.any():
        chk("heading (valid frames)", np.abs(back[hv, 0, 15:17] - m17[hv, 0, 15:17]).max() < 1e-5, np.abs(back[hv, 0, 15:17] - m17[hv, 0, 15:17]).max())
        chk("heading_valid kept on valid frames", bool(hv2[hv].all()), float((~hv2[hv]).sum()))
    ch = children_of(par); nonleaf = np.array([len(c) > 0 for c in ch]); leaf = ~nonleaf
    d0, d1 = decode_column_cont6d(m17[..., 3:9]), decode_column_cont6d(back[..., 3:9])
    chk("rest delta non-leaf", np.abs(d0[:, nonleaf] - d1[:, nonleaf]).max() < 1e-5, np.abs(d0[:, nonleaf] - d1[:, nonleaf]).max())
    ang = np.degrees(np.arccos(np.clip((np.einsum("tjab,tjab->tj", d0[:, leaf], d1[:, leaf]) - 1) / 2, -1, 1)))
    worst["leaf rotation error deg (mean over sampled clips)"] = worst.get("leaf rotation error deg (mean over sampled clips)", 0.0) + float(ang.mean()) / len(pick)
    worst["leaf fraction (mean over sampled clips)"] = worst.get("leaf fraction (mean over sampled clips)", 0.0) + float(leaf.mean()) / len(pick)
    # NATIVE AnyTop FK decoder on the 13 container (per-frame velocities; world rest bone vectors as offsets)
    Rrest, offl = np.asarray(sk["R_rest_global"], np.float64), np.asarray(sk["offset_parent_local"], np.float64)
    off_world = np.zeros((J, 3)); off_world[1:] = np.einsum("jab,jb->ja", Rrest[par[1:]], offl[1:])
    nat = a13[..., :13].copy(); nat[..., 9:12] /= FPS
    P_nat = recover_from_bvh_rot_np(nat, par, off_world)
    P0 = direct_decode_positions(m17); c = (P_nat - P0)[:, 0].mean(0)
    chk("native AnyTop FK == KTJD world positions (bl, up to one offset)", np.abs(P_nat - P0 - c).max() / bl < 1e-3, np.abs(P_nat - P0 - c).max() / bl)
    if int(i) == int(pick[0]):
        # generated-sample policies: degenerate cells and conflicting siblings
        g = a13_32.copy(); g[5, 0, 3:9] = 0.0                                  # degenerate facing at t=5 -> hold t=4, heading invalid
        b, hvg, dg = anytop13_to_ktjd17(g, par, fps=FPS, eps_h=eps_h)
        chk("degenerate facing: no raise, counted, held", dg["degenerate_facing_frames"] == 1 and not hvg[5] and np.isfinite(b).all(), dg["degenerate_facing_frames"])
        # the held facing (frame 4's) is what un-rotates frame 5's RIC positions; the heading channel of frame 5 is invalid -> exact zero
        P5 = direct_decode_positions(b)[5, 1:]
        exp5 = np.einsum("ab,jb->ja", Rf_expect(a13_32, 4), a13_32[5, 1:, 0:3]); exp5[:, 0] += b[5, 0, 13]; exp5[:, 2] += b[5, 0, 14]
        chk("degenerate facing: frame 5 un-rotated with frame 4's facing", np.abs(P5 - exp5).max() < 1e-4, np.abs(P5 - exp5).max())
        chk("degenerate facing: frame 5 heading stored as exact zero", not b[5, 0, 15:17].any(), float(np.abs(b[5, 0, 15:17]).max()))
        # velocities come from the PREDICTED local velocities, not from position differences
        g = a13_32.copy(); g[3, 2, 9:12] += np.array([1.0, 2.0, 3.0])
        b, _, _ = anytop13_to_ktjd17(g, par, fps=FPS, eps_h=eps_h)
        dv = b[3, 2, 9:12] - back[3, 2, 9:12]
        chk("velocity channels follow the predicted local velocity (un-rotated by the next facing)", np.abs(dv - Rf_expect(a13_32, 4) @ np.array([1.0, 2.0, 3.0])).max() < 1e-4 and np.abs(b[..., 0:3] - back[..., 0:3]).max() < 1e-9, np.abs(dv).max())
        p_multi = next((p for p in range(J) if len(ch[p]) >= 2), None)
        if p_multi is not None:
            g = a13_32.copy(); g[:, ch[p_multi][0], 3:9] = 0.0                # first child slot degenerate: the last child's is used
            b, _, dg = anytop13_to_ktjd17(g, par, fps=FPS, eps_h=eps_h)
            chk("degenerate FIRST child slot ignored under last-child-wins", dg["degenerate_child_slots"] == 0 and np.abs(b - back).max() < 1e-4, np.abs(b - back).max())
            g = a13_32.copy(); g[:, ch[p_multi][-1], 3:9] = np.array([1, 0, 0, 0, 1, 0])   # last child says identity: it wins
            b, _, _ = anytop13_to_ktjd17(g, par, fps=FPS, eps_h=eps_h)
            Lp = decode_column_cont6d(b[..., 3:9])[:, p_multi]; gp = int(par[p_multi])
            expect = decode_column_cont6d(back[..., 3:9])[:, gp] if gp >= 0 else np.tile(np.eye(3), (T, 1, 1))
            chk("conflicting siblings: last child wins", np.abs(Lp - expect).max() < 1e-6, np.abs(Lp - expect).max())
            g = a13_32.copy(); g[:, ch[p_multi][-1], 3:9] = 0.0                 # last child slot degenerate -> identity local
            b, _, dg = anytop13_to_ktjd17(g, par, fps=FPS, eps_h=eps_h)
            chk("degenerate LAST child slot -> identity local, counted", dg["degenerate_child_slots"] == T and np.isfinite(b).all(), dg["degenerate_child_slots"])
            Wp, Wgp = decode_column_cont6d(b[..., 3:9])[:, p_multi], (decode_column_cont6d(b[..., 3:9])[:, int(par[p_multi])] if par[p_multi] >= 0 else np.tile(np.eye(3), (T, 1, 1)))
            chk("degenerate LAST child slot: joint inherits its parent's rest delta (identity local)", np.abs(Wp - Wgp).max() < 1e-6, np.abs(Wp - Wgp).max())
# ---- synthetic: known facing direction, J=1, T=1 ----
fwd_x = np.array([[0.0, 1.0]])                                             # heading (fwd_z, fwd_x) = (0, 1): the animal faces +X
Rf = facing_from_heading(fwd_x, np.array([True]))[0]
chk("facing +X maps to +Z", np.abs(Rf @ np.array([1.0, 0, 0]) - np.array([0, 0, 1.0])).max() < 1e-12, np.abs(Rf @ np.array([1.0, 0, 0]) - np.array([0, 0, 1.0])).max())
chk("Ry sign convention: Ry(+90deg) sends +Z to +X", np.abs(_ry(np.array([np.pi / 2]))[0] @ np.array([0, 0, 1.0]) - np.array([1.0, 0, 0])).max() < 1e-12, 0.0)
one = np.zeros((7, 1, 17)); one[:, 0, 1] = 1.0; one[:, 0, 3:9] = [1, 0, 0, 0, 1, 0]; one[:, 0, 15] = 1.0
o13 = ktjd17_to_anytop13(one, np.ones(7, bool), np.array([-1]), fps=FPS); ob, ohv, _ = anytop13_to_ktjd17(o13, np.array([-1]), fps=FPS, eps_h=eps_h)
chk("J=1 round trip", np.abs(ob[..., 0:3] - one[..., 0:3]).max() < 1e-9 and ohv.all(), np.abs(ob[..., 0:3] - one[..., 0:3]).max())
r0 = rows[int(pick[0])]; pl0 = load_motion_npz(f"{ROOT}/{r0['motion_relpath']}", expected_fps_target=FPS); par0 = np.asarray(np.load(f"{ROOT}/skeletons/{r0['rig_id']}.npz", allow_pickle=True)["parents"], np.int64)
t1 = np.asarray(pl0["motion"], np.float64)[:1]; t13 = ktjd17_to_anytop13(t1, np.asarray(pl0["heading_valid"], bool)[:1], par0, fps=FPS); tb, _, _ = anytop13_to_ktjd17(t13, par0, fps=FPS, eps_h=eps_h)
chk("T=1 round trip (positions)", np.abs(tb[..., 0:3] - t1[..., 0:3]).max() < 1e-6, np.abs(tb[..., 0:3] - t1[..., 0:3]).max())
try:
    ktjd17_to_anytop13(t1, np.ones(1, bool), np.array([-1, -1, 0] + [1] * (t1.shape[1] - 3)), fps=FPS); chk("bad parents refused", False, 1)
except ValueError:
    pass
print("worst:", {k: round(v, 6) for k, v in worst.items()})
print("RESULT:", "PASS" if not fails else f"FAIL ({len(fails)}): " + "; ".join(fails[:8]))
sys.exit(1 if fails else 0)
