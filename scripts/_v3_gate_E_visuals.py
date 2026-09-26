"""Gate-E VISUALS for the v3a human rot6d re-encode (scratch-only, NO GPU, NO canonical writes).
  (1) TWIST-ANGLE-OVER-TIME plot: for the 3 worst-v2-jitter single-child joints, axial-twist angle
      (deg) vs frame, v2 (random zigzag) vs v3a (flat/canonical). One PNG per clip.
  (2) GT-FK POSE GIF: GT-FK positions reconstructed FROM the v3a re-encoded rotations (canonical
      recover_rot6d_fk -- the FK the converter assumes), side-by-side with an FK+GT-RIC overlay, to
      confirm v3 does NOT break the geometry. Prints the position self-checks.
Usage: python scripts/_v3_gate_E_visuals.py
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

REPO = "/iridisfs/scratch/ts1v23/workspace/noKslot_clean"
HM = "/iridisfs/scratch/ts1v23/workspace/motion-latent-diffusion-main"
SCRATCH = REPO + "/scratch"
sys.path.insert(0, HM); sys.path.insert(0, REPO)
import importlib.util as _u
def _load(name, path):
    s = _u.spec_from_file_location(name, path); m = _u.module_from_spec(s); s.loader.exec_module(m); return m
cv = _load("cv", REPO + "/scripts/convert_humanml3d_to_anytop13.py")
gr = _load("gr", REPO + "/scripts/_v3_gate_runner.py")               # _twist_angle_deg/_sixd_to_mat/_fk/_ric/SINGLE_CHILD_JOINTS

J = cv.J
PARENTS = cv.PARENTS
NAMES = cv.JOINT_NAMES
SC = gr.SINGLE_CHILD_JOINTS                                          # single-child token indices
HIGH_JITTER, LOCOMOTION = "002365", "019263"                        # worst_accel / locomotion strata


def encode(mid):
    x = np.load(Path(cv.SRC) / "new_joint_vecs" / f"{mid}.npy")
    P = cv.world_positions(x)
    raw0 = cv.convert_263_to_13(x)
    off = cv.compute_offsets()
    v2 = cv.reencode_rot6d(raw0, P, off, rot6d_mode="v2")
    v3 = cv.reencode_rot6d(raw0, P, off, rot6d_mode="v3a")
    return v2, v3, off


def twist_curves(v, off):                                           # -> theta[T,Jn] deg, u_axes[Jn,3]
    u = off[SC].astype(np.float64); u = u / (np.linalg.norm(u, axis=-1, keepdims=True) + 1e-12)
    R = gr._sixd_to_mat(v[:, SC, 3:9].astype(np.float64))
    return gr._twist_angle_deg(R, u), u


def per_joint_jitter(theta):                                        # 2nd-diff of twist angle, deg/frame^2, per joint
    dth = (np.diff(theta, axis=0) + 180.0) % 360.0 - 180.0          # wrapped velocity
    acc = np.diff(dth, axis=0)
    return np.abs(acc).mean(axis=0)                                 # [Jn]


def make_twist_plot(mid, kind):
    v2, v3, off = encode(mid)
    th2, _ = twist_curves(v2, off); th3, _ = twist_curves(v3, off)
    jit2 = per_joint_jitter(th2); jit3 = per_joint_jitter(th3)
    order = np.argsort(-jit2)[:3]                                   # 3 worst by v2 jitter
    T = th2.shape[0]
    fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True)
    rows = []
    for ax, k in zip(axes, order):
        j = SC[k]; p = PARENTS[j]
        lbl = f"{NAMES[p]}->{NAMES[j]}"
        ax.plot(range(T), th2[:, k], color="crimson", lw=1.4, label=f"v2 (random gauge)  jit={jit2[k]:.1f} deg/f^2")
        ax.plot(range(T), th3[:, k], color="royalblue", lw=1.8, label=f"v3a (canonical swing)  jit={jit3[k]:.2f} deg/f^2")
        ax.set_title(f"single-child twist: {lbl}", fontsize=10)
        ax.set_ylabel("axial twist (deg)"); ax.set_ylim(-185, 185); ax.grid(alpha=0.3); ax.legend(fontsize=8, loc="upper right")
        rows.append((lbl, float(jit2[k]), float(jit3[k])))
    axes[-1].set_xlabel("frame")
    fig.suptitle(f"Gate-E: axial-twist angle over time  [clip {mid} | {kind}]\n"
                 f"v2 = LAPACK-Kabsch null-space twist (random, zigzag)   vs   v3a = deterministic zero-twist swing (flat)",
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out = f"{SCRATCH}/gateE_twist_{kind}_{mid}.png"
    fig.savefig(out, dpi=120); plt.close(fig)
    return out, rows


def _draw(ax, pos, color, style, lw, label, marker):
    ax.plot([], [], [], color=color, lw=lw, label=label)            # legend proxy
    for j in range(1, J):
        p = PARENTS[j]
        ax.plot([pos[j, 0], pos[p, 0]], [pos[j, 2], pos[p, 2]], [pos[j, 1], pos[p, 1]],
                color=color, lw=lw, ls=style)
    ax.scatter(pos[:, 0], pos[:, 2], pos[:, 1], color=color, s=10, marker=marker)


def make_pose_gif(mid, n_frames=36):
    v2, v3, off = encode(mid)
    fk = gr._fk(v3, off)                                            # GT-FK from v3a rotations [T,J,3]
    fk2 = gr._fk(v2, off)                                           # GT-FK from v2 rotations (twist-invariance check)
    ric = gr._ric(v2)                                              # GT-RIC (ch0:3; v2==v3a here) [T,J,3]
    T = fk.shape[0]
    idx = np.linspace(0, T - 1, min(n_frames, T)).astype(int)
    # position self-checks (single-child joints + all joints)
    scj = SC
    d_v2v3 = float(np.abs(fk2[:, scj] - fk[:, scj]).max())          # twist-invariance: v2-FK == v3-FK
    d_fkric_sc = float(np.linalg.norm(fk[:, scj] - ric[:, scj], axis=-1).mean())
    d_fkric_all = float(np.linalg.norm(fk - ric, axis=-1).mean())
    allp = np.concatenate([fk[idx], ric[idx]], axis=0).reshape(-1, 3)
    lo, hi = allp.min(0), allp.max(0); ctr = (lo + hi) / 2; rad = float((hi - lo).max()) / 2 + 1e-3
    fig = plt.figure(figsize=(11, 5.5))
    axL = fig.add_subplot(1, 2, 1, projection="3d"); axR = fig.add_subplot(1, 2, 2, projection="3d")

    def setup(ax, title):
        ax.set_title(title, fontsize=10)
        ax.set_xlim(ctr[0] - rad, ctr[0] + rad); ax.set_ylim(ctr[2] - rad, ctr[2] + rad); ax.set_zlim(ctr[1] - rad, ctr[1] + rad)
        ax.set_xlabel("x"); ax.set_ylabel("z"); ax.set_zlabel("y(up)"); ax.view_init(elev=12, azim=-70)

    def update(fi):
        f = idx[fi]
        axL.cla(); axR.cla(); setup(axL, "GT-FK from v3a rotations"); setup(axR, "overlay: FK (blue) vs GT-RIC (red dashed)")
        _draw(axL, fk[f], "royalblue", "-", 2.0, "v3a-FK", "o")
        _draw(axR, fk[f], "royalblue", "-", 2.0, "v3a-FK", "o")
        _draw(axR, ric[f], "crimson", "--", 1.4, "GT-RIC", "^")
        axR.legend(fontsize=8, loc="upper left")
        fig.suptitle(f"Gate-E no-breakage: clip {mid}  frame {f}/{T-1}\n"
                     f"GT-FK is twist-invariant -> v2-FK==v3-FK (max delta {d_v2v3:.1e} m); this confirms v3 does NOT break geometry.\n"
                     f"FK-vs-RIC delta {d_fkric_all*100:.2f} cm = the accepted pre-existing FK-floor (NOT a v3 effect). "
                     f"Motion-smoothness payoff = model-recon matched-smoke (needs training).", fontsize=8)
        return []

    anim = FuncAnimation(fig, update, frames=len(idx), blit=False)
    out = f"{SCRATCH}/gateE_pose_{mid}.gif"
    anim.save(out, writer=PillowWriter(fps=12)); plt.close(fig)
    return out, dict(v2FK_vs_v3FK_singlechild_max_m=d_v2v3, FK_vs_RIC_singlechild_mean_m=d_fkric_sc,
                     FK_vs_RIC_alljoint_mean_m=d_fkric_all, n_frames=len(idx), T=T)


if __name__ == "__main__":
    print("=== (1) TWIST-ANGLE-OVER-TIME plots ===")
    p1, rows1 = make_twist_plot(HIGH_JITTER, "high-jitter")
    p2, rows2 = make_twist_plot(LOCOMOTION, "locomotion")
    for tag, p, rows in [("high-jitter " + HIGH_JITTER, p1, rows1), ("locomotion " + LOCOMOTION, p2, rows2)]:
        print(f"  [{tag}] -> {p}")
        for lbl, j2, j3 in rows:
            print(f"      {lbl:28s}  v2 twist-jit {j2:8.2f}  ->  v3a {j3:6.2f} deg/frame^2   ({j2/max(j3,1e-9):.0f}x lower)")
    print("=== (2) GT-FK POSE GIF ===")
    g, chk = make_pose_gif(LOCOMOTION)
    print(f"  gif -> {g}")
    print(f"  self-check (single-child): v2-FK vs v3-FK max delta = {chk['v2FK_vs_v3FK_singlechild_max_m']:.2e} m  (twist-invariance, ~0)")
    print(f"  FK vs RIC single-child mean = {chk['FK_vs_RIC_singlechild_mean_m']*100:.3f} cm | all-joint mean = {chk['FK_vs_RIC_alljoint_mean_m']*100:.3f} cm (accepted FK-floor)")
    print(f"  gif frames = {chk['n_frames']} (of T={chk['T']})")
    print("=== DONE ===")
