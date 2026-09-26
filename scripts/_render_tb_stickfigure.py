"""Give a Truebones rig a body and render it, under one set of conditions for every rig.

The Truebones assets we hold are motion on a skeleton -- no mesh -- so a bone gets a tapered cylinder
and a joint a sphere, which is how AnyTop renders these same rigs
(outside_docs/AnyTop/visualization/visualize_stick_figure_blender.py, MIT; imported, not copied).

Everything that decides the look is fixed here and shared by every rig, so two rigs can be put side by
side: the three-quarter camera of the paper's figures (azimuth 40 deg, elevation 18 deg), a framing
computed once over the WHOLE clip so world translation stays visible, bone and joint radii in units of
the rig's own mean rest bone length, one floor plane at the world floor, one light rig, one resolution
and the corpus frame rate.

  blender -b -P scripts/_render_tb_stickfigure.py -- --npz <world.npz> --out <dir> [--key gen_ric]
          [--also_gt] [--samples 24] [--res 640] [--device CPU|GPU]

Writes <dir>/frames/*.png, <dir>/render.json (every setting, the checkpoint the dump names, the clip)
and, when Pillow is importable, <dir>/<name>.gif at the corpus frame rate.
"""
import argparse, json, math, os, sys
from pathlib import Path

import numpy as np
import bpy
from mathutils import Vector

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "outside_docs" / "AnyTop"))

# AnyTop's renderer imports scipy for one thing -- a 3x3 rotation matrix to a quaternion -- and
# Blender's bundled Python has no scipy. Stand in with Blender's own mathutils so that module is
# imported unmodified. scipy's as_quat() is (x, y, z, w); mathutils' to_quaternion() is (w, x, y, z).
if "scipy" not in sys.modules:
    import types
    from mathutils import Matrix as _M

    class _Rot:
        def __init__(self, q): self._q = q
        @staticmethod
        def from_matrix(m):
            q = _M([list(r) for r in np.asarray(m, dtype=float)]).to_quaternion()
            return _Rot(q)
        def as_quat(self):
            q = self._q
            return np.array([q.x, q.y, q.z, q.w], dtype=float)

    _sp = types.ModuleType("scipy"); _sptr = types.ModuleType("scipy.spatial")
    _tr = types.ModuleType("scipy.spatial.transform"); _tr.Rotation = _Rot
    _sptr.transform = _tr; _sp.spatial = _sptr
    sys.modules["scipy"] = _sp; sys.modules["scipy.spatial"] = _sptr
    sys.modules["scipy.spatial.transform"] = _tr

if "tqdm" not in sys.modules:                      # progress bar only; Blender's Python has none
    import types as _t
    _tq = _t.ModuleType("tqdm"); _tq.tqdm = lambda x, *a, **k: x
    sys.modules["tqdm"] = _tq

from visualization.visualize_stick_figure_blender import StickFigure          # noqa: E402  (MIT)

AZIM_DEG, ELEV_DEG = 40.0, 18.0        # the three-quarter camera of Figures 3 and 4
GEN_RGB = (0.122, 0.435, 0.545, 1.0)   # #1F6F8B, the paper's teal
GT_RGB = (0.710, 0.333, 0.184, 1.0)    # #B5552F, the paper's clay
FLOOR_RGB = (0.945, 0.937, 0.925, 1.0)
BONE_R, JOINT_R = 0.22, 0.28           # in mean rest bone lengths


def parse():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--key", default="gen_ric", choices=["gen_ric", "gen_fk", "gt_w"])
    ap.add_argument("--also_gt", action="store_true", help="a second body from gt_w, offset sideways")
    ap.add_argument("--samples", type=int, default=24); ap.add_argument("--res", type=int, default=640)
    ap.add_argument("--device", default="CPU", choices=["CPU", "GPU"])
    ap.add_argument("--max_frames", type=int, default=0, help="0 = the whole clip")
    return ap.parse_args(argv)


def ktjd_to_blender(P):
    """KTJD world (+Y up, +Z forward, X lateral) -> Blender (+Z up)."""
    return np.stack([P[..., 0], P[..., 2], P[..., 1]], -1)


def material(name, rgba):
    m = bpy.data.materials.new(name); m.use_nodes = True
    bsdf = m.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = rgba
    bsdf.inputs["Roughness"].default_value = 0.55
    return m


def build_scene(res, samples, device):
    bpy.ops.wm.read_homefile(use_empty=True)
    sc = bpy.context.scene
    sc.render.engine = "CYCLES"
    sc.cycles.samples = samples
    sc.cycles.use_denoising = True
    sc.cycles.device = device
    if device == "GPU":
        prefs = bpy.context.preferences.addons["cycles"].preferences
        prefs.compute_device_type = "CUDA"
        prefs.get_devices()
        for d in prefs.devices:
            d.use = (d.type == "CUDA")
    # Blender 4.x tone-maps with AgX by default, which desaturates a flat colour; the figures'
    # ink is a named hex value, so render it as that value
    sc.view_settings.view_transform = "Standard"
    sc.view_settings.look = "None"
    sc.render.resolution_x = sc.render.resolution_y = res
    sc.render.resolution_percentage = 100
    sc.render.film_transparent = False
    sc.world = bpy.data.worlds.new("W"); sc.world.use_nodes = True
    sc.world.node_tree.nodes["Background"].inputs[0].default_value = (1, 1, 1, 1)
    sc.world.node_tree.nodes["Background"].inputs[1].default_value = 1.0
    return sc


def add_floor(z, radius, mat):
    bpy.ops.mesh.primitive_plane_add(size=radius * 6, location=(0, 0, z))
    p = bpy.context.object; p.name = "floor"; p.data.materials.append(mat)
    return p


def add_lights(centre, radius):
    for key, (dx, dy, dz, e) in {"key": (1.4, -1.6, 2.2, 1.0), "fill": (-1.8, -0.8, 1.2, 0.35),
                                 "rim": (-0.6, 1.9, 1.8, 0.45)}.items():
        bpy.ops.object.light_add(type="AREA",
                                 location=Vector(centre) + Vector((dx, dy, dz)) * radius)
        L = bpy.context.object; L.name = f"light_{key}"
        L.data.energy = e * radius * radius * 12.0
        L.data.size = radius * 1.2
        d = Vector(centre) - L.location
        L.rotation_euler = d.to_track_quat("-Z", "Y").to_euler()


def place_camera(centre, radius):
    a, e = math.radians(AZIM_DEG), math.radians(ELEV_DEG)
    dirv = Vector((math.cos(a) * math.cos(e), math.sin(a) * math.cos(e), math.sin(e)))
    bpy.ops.object.camera_add(location=Vector(centre) + dirv * (radius * 3.1))
    cam = bpy.context.object
    cam.data.lens = 50.0
    cam.rotation_euler = (Vector(centre) - cam.location).to_track_quat("-Z", "Y").to_euler()
    bpy.context.scene.camera = cam
    return cam


def main():
    a = parse()
    z = np.load(a.npz, allow_pickle=True)
    parents = z["parents"].astype(int)
    rest = np.asarray(z["demo_w"][0], dtype=np.float64)
    j = np.where(parents >= 0)[0]
    bl = float(np.linalg.norm(rest[j] - rest[parents[j]], axis=-1).mean())    # the rig's own scale
    P = np.asarray(z[a.key], dtype=np.float64)
    if a.max_frames:
        P = P[:a.max_frames]
    bodies = [("gen", ktjd_to_blender(P) / bl, GEN_RGB)]
    if a.also_gt:
        G = np.asarray(z["gt_w"], dtype=np.float64)[:P.shape[0]]
        bodies.append(("gt", ktjd_to_blender(G) / bl, GT_RGB))
    T = min(b[1].shape[0] for b in bodies)
    bodies = [(n, Q[:T], c) for n, Q, c in bodies]

    # every body in one frame of reference, side by side when there are two
    allpts = np.concatenate([Q.reshape(-1, 3) for _, Q, _ in bodies])
    span = float(np.linalg.norm(allpts.max(0) - allpts.min(0)))
    if len(bodies) == 2:
        # separate the two along the CAMERA's right vector, not world X: at azimuth 40 deg a world-X
        # offset projects partly into depth and the two bodies overlap on screen
        ar = math.radians(AZIM_DEG)
        right = np.array([math.sin(ar), -math.cos(ar), 0.0])
        proj = allpts @ right
        gap = 0.80 * float(proj.max() - proj.min() + 1e-6)
        bodies = [(n, Q + right * (sgn * gap), c)
                  for (n, Q, c), sgn in zip(bodies, (-0.5, 0.5))]
        allpts = np.concatenate([Q.reshape(-1, 3) for _, Q, _ in bodies])
    centre = ((allpts.min(0) + allpts.max(0)) / 2.0)
    centre[2] = allpts[:, 2].min() + 0.45 * (allpts[:, 2].max() - allpts[:, 2].min())
    radius = max(float(np.linalg.norm(allpts.max(0) - allpts.min(0))) * 0.5, 1.0)

    sc = build_scene(a.res, a.samples, a.device)
    sc.frame_start, sc.frame_end = 1, T
    bones = [(int(c), int(parents[c])) for c in range(len(parents)) if parents[c] >= 0]
    for name, Q, rgba in bodies:
        StickFigure.visualize(joint_locations=Q, bone_list=bones, cylinder_radius=BONE_R,
                              sphere_radius=JOINT_R, joint_materials=None, unq_name=name)
        # AnyTop names a bone "cyl_<head>_<tail>_<unq>" and a joint "<index>_<unq>"
        m = material(f"mat_{name}", rgba)
        for ob in bpy.data.objects:
            if ob.type == "MESH" and ob.name.split(".")[0].endswith(f"_{name}"):
                ob.data.materials.clear(); ob.data.materials.append(m)
    add_floor(float(allpts[:, 2].min()) - 0.02, radius, material("mat_floor", FLOOR_RGB))
    add_lights(tuple(centre), radius)
    place_camera(tuple(centre), radius)

    out = Path(a.out); (out / "frames").mkdir(parents=True, exist_ok=True)
    sc.render.image_settings.file_format = "PNG"
    sc.render.filepath = str(out / "frames" / "f")
    bpy.ops.render.render(animation=True)

    fps = float(z["fps"])
    meta = {"npz": str(a.npz), "key": a.key, "also_gt": bool(a.also_gt), "frames": int(T),
            "fps": fps, "rig": str(z["rig"]), "clip": str(z["motion_id"]), "caption": str(z["caption"]),
            "ckpt": str(z["ckpt"]), "mean_rest_bone_length": bl, "azimuth_deg": AZIM_DEG,
            "elevation_deg": ELEV_DEG, "bone_radius_bl": BONE_R, "joint_radius_bl": JOINT_R,
            "samples": a.samples, "resolution": a.res, "device": a.device, "span_bl": span}
    (out / "render.json").write_text(json.dumps(meta, indent=1))
    try:
        from PIL import Image
        fr = sorted((out / "frames").glob("f*.png"))
        if fr:
            ims = [Image.open(p).convert("P", palette=Image.ADAPTIVE) for p in fr]
            ims[0].save(out / f"{out.name}.gif", save_all=True, append_images=ims[1:],
                        duration=int(round(1000.0 / fps)), loop=0, optimize=True)
            print(f"[render] {out / (out.name + '.gif')}  {len(fr)} frames @ {fps:g} fps", flush=True)
    except Exception as e:
        print(f"[render] frames written; gif skipped ({e})", flush=True)


if __name__ == "__main__":
    main()
