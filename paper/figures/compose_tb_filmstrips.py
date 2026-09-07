"""Appendix figure: filmstrips of the Truebones arms drawn from the world dumps of scripts/v2_render_incontext.py --dump_world.
For each rig one held-out clip, two rows (zero training / per-rig LoRA), six frames spread over the clip; in every panel the
directly predicted joint positions (clay, underneath) are overlaid by the forward kinematics of the predicted rotations
(teal, on top), so the FK--pose gap of Table lora shows up as separation of the two skeletons.  Only our generations are drawn.
  python paper/figures/compose_tb_filmstrips.py <out.pdf>"""
import sys, textwrap
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path(sys.argv[1])
POSE, FK, INK, MUTE = "#B5552F", "#1F6F8B", "#222222", "#9a9a9a"
ARMS = [("zero", "zerotrain"), ("LoRA", "lora")]                 # row tag -> renders/cmp_dump_<rig>_<arm>
RIGS = [  # (display name, dump rig key, clip id, anchor): anchor "ground" keeps the height above the floor, "root" follows the root
    ("Buffalo (30 j.)", "buffalo", "A_Buffalo__Buffalo___SleepUp_140", "ground"),
    ("Gazelle (29 j.)", "gazelle", "A_Gazelle__Gazelle___Attack1_381", "ground"),
    ("Dragon (95 j.)", "dragon", "A_Dragon__Dragon___Attack3_299", "root"),
    ("Spider (68 j.)", "spider", "A_Spider__Spider___Scramble_916", "ground"),
]
NF = 6
FLOOR_Y = 0.0        # KTJD world frame: the floor is the plane Y = 0 (+Y up, +Z forward, X lateral)
plt.rcParams.update({"font.family": "serif", "font.serif": ["Nimbus Roman", "Times New Roman", "Times"], "font.size": 7, "pdf.fonttype": 42})

# three-quarter camera of paper/figures/skeleton_art.py (azimuth 40 deg, elevation 18 deg) on KTJD world coordinates.
# Body coordinates are (forward, right, up) = (Z, X, Y).
_a, _e = np.radians(40.0), np.radians(18.0)
_cam = np.array([np.cos(_a) * np.cos(_e), np.sin(_a) * np.cos(_e), np.sin(_e)])
_fwd = -_cam; _up = np.array([0, 0, 1.0]); _right = np.cross(_fwd, _up); _right /= np.linalg.norm(_right); _u = np.cross(_right, _fwd)

def project(P):                                   # P (..., 3) world -> (..., 2) screen, (...,) depth (larger = farther from the camera)
    B = np.stack([P[..., 2], P[..., 0], P[..., 1]], -1)
    return np.stack([B @ _right, B @ _u], -1), B @ _fwd

def load(rig, arm, clip):
    z = np.load(f"renders/cmp_dump_{rig}_{arm}/{clip}.world.npz", allow_pickle=True)
    assert str(z["dump_format"]) == "world-dump-v3", z["dump_format"]
    assert str(z["motion_id"]) == clip.split("__", 1)[1], (z["motion_id"], clip)
    return z["gen_ric"], z["gen_fk"], z["parents"].astype(int), str(z["caption"])

def anchored(pos, fk, parents, anchor):
    """Per-frame anchoring on the root joint (its position is shared by the two decodes): horizontal always, vertical only
    for anchor='root'.  With anchor='ground' the floor stays the plane Y = FLOOR_Y."""
    root = int(np.where(parents < 0)[0][0])
    shift = pos[:, root:root + 1].copy()
    if anchor == "ground": shift[..., 1] = 0.0
    return pos - shift, fk - shift

def draw(ax, P, depth, parents, color, lw, alpha, z):
    """Bones of one skeleton, farthest bone first (bone depth = mid-point depth)."""
    j = np.where(parents >= 0)[0]; p = parents[j]
    for k in np.argsort(-(depth[j] + depth[p]) / 2):
        a, b = j[k], p[k]
        ax.plot([P[a, 0], P[b, 0]], [P[a, 1], P[b, 1]], color=color, lw=lw, alpha=alpha, solid_capstyle="round", zorder=z)

W, LAB, GAP, CAP_H, BLOCK_GAP = 5.5, 0.62, 0.04, 0.20, 0.08
PW = (W - LAB - GAP * (NF - 1)) / NF
blocks = []
for name, rig, clip, anchor in RIGS:
    arms = {tag: load(rig, arm, clip) for tag, arm in ARMS}
    (T, J), caption, parents = arms["zero"][0].shape[:2], arms["zero"][3], arms["zero"][2]
    for tag, (pos, fk, par, cap) in arms.items():          # both rows must show the same clip at the same time stamps
        assert pos.shape == fk.shape == (T, J, 3), (tag, pos.shape, fk.shape, T, J)
        assert cap == caption and np.array_equal(par, parents), tag
    idx = np.linspace(0, T - 1, NF).round().astype(int)
    rows = []
    for tag, (pos, fk, _, _) in arms.items():
        pos, fk = anchored(pos[idx], fk[idx], parents, anchor)
        (sp, dp), (sf, df) = project(pos), project(fk)
        pts = [sp.reshape(-1, 2), sf.reshape(-1, 2)]
        if anchor == "ground": pts.append(project(np.array([[0.0, FLOOR_Y, 0.0]]))[0])   # the floor point under the root stays in view
        pts = np.concatenate(pts)
        rows.append([tag, sp, dp, sf, df, pts.min(0), pts.max(0)])
    # one scale (inches per world unit) for both rows of a rig, set by the wider row, so the two arms are drawn at the same size;
    # each row gets its own vertical extent
    wmax = max(hi[0] - lo[0] for *_, lo, hi in rows) * 1.08
    for r in rows:
        lo, hi = r[5], r[6]; cx = (lo[0] + hi[0]) / 2; pad = 0.04 * wmax
        r.append((cx - wmax / 2, cx + wmax / 2, lo[1] - pad, hi[1] + pad))
        r.append(PW * (hi[1] - lo[1] + 2 * pad) / wmax)                     # panel height in inches
    blocks.append((name, clip, anchor, caption, parents, rows))

H = sum(sum(r[-1] + GAP for r in b[5]) + CAP_H + BLOCK_GAP for b in blocks)
fig = plt.figure(figsize=(W, H))
y = H
for name, clip, anchor, caption, parents, rows in blocks:
    for r, (tag, sp, dp, sf, df, _lo, _hi, lim, ph) in enumerate(rows):
        y -= ph
        fig.text(0.02 / W, (y + ph / 2) / H, f"{name}\n{tag}" if r == 0 else tag, ha="left", va="center", fontsize=7, color=INK, linespacing=1.2)
        for k in range(NF):
            ax = fig.add_axes([(LAB + k * (PW + GAP)) / W, y / H, PW / W, ph / H])
            if anchor == "ground":
                g = project(np.array([[0.0, FLOOR_Y, 0.0]]))[0][0, 1]
                ax.axhline(g, color="#d8d8d8", lw=0.5, zorder=1)
            draw(ax, sp[k], dp[k], parents, POSE, 1.1, 0.9, 2)
            draw(ax, sf[k], df[k], parents, FK, 0.7, 0.95, 3)
            ax.set_xlim(lim[0], lim[1]); ax.set_ylim(lim[2], lim[3]); ax.set_aspect("equal"); ax.axis("off")
        y -= GAP
    fig.text(LAB / W, (y - 0.02) / H, "\n".join(textwrap.wrap(f"“{caption}”", width=110)), ha="left", va="top", fontsize=7, color=MUTE)
    y -= CAP_H + BLOCK_GAP
fig.savefig(OUT, dpi=300); fig.savefig(OUT.with_suffix(".png"), dpi=150); print("wrote", OUT, "size", W, round(H, 2), "in")
