"""Rest-pose skeleton line art from a KTJD-17 skeleton npz (P_rest_global + parents), for the teaser figure.
draw_rest(ax, npz_path, view) draws bones as lines and joints as dots in an orthographic side / three-quarter view."""
import numpy as np
def load(npz_path):
    z = np.load(npz_path, allow_pickle=True)
    return z["P_rest_global"].astype(float), z["parents"].astype(int), [str(s) for s in z["joint_names"]]
def project(P, view="threequarter", contact_idx=()):
    # KTJD canonical rest frame (checked on 6 rigs 2026-09-06): hips->head = +Z (forward), feet = -Y (so +Y up),
    # lateral = X.  Body coords: x forward, y right, z up.
    C = P - P.mean(0)
    B = np.stack([C[:, 2], C[:, 0], C[:, 1]], 1)
    az = {"side": 90.0, "threequarter": 40.0, "front": 5.0}[view]
    el = {"side": 8.0, "threequarter": 18.0, "front": 12.0}[view]
    a, e = np.radians(az), np.radians(el)
    cam = np.array([np.cos(a) * np.cos(e), np.sin(a) * np.cos(e), np.sin(e)])     # camera position direction
    fwd = -cam; up = np.array([0, 0, 1.0]); right = np.cross(fwd, up); right /= np.linalg.norm(right); u = np.cross(right, fwd)
    return B @ right, B @ u, B @ fwd
def draw_rest(ax, npz_path, view="threequarter", color="#b5552f", lw=1.3, dot=4, label=None):
    P, par, names = load(npz_path)
    z = np.load(npz_path, allow_pickle=True); contact = z["contact_joint_indices"] if "contact_joint_indices" in z.files else []
    x, y, d = project(P, view, contact)
    order = np.argsort(d)                       # far bones first
    for j in order:
        p = par[j]
        if p < 0: continue
        ax.plot([x[j], x[p]], [y[j], y[p]], color=color, lw=lw, solid_capstyle="round", alpha=0.95, zorder=2)
    ax.scatter(x, y, s=dot, color="#2b2b2b", zorder=3, linewidths=0)
    ax.set_aspect("equal"); ax.axis("off")
    if label: ax.set_title(label, fontsize=9, pad=2, color="#333333")
    return len(names)
