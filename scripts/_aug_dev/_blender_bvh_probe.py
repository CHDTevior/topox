"""Run inside Blender: import a BVH with Blender's own importer and dump every pose bone's world head per frame.
    blender -b --factory-startup --python scripts/_aug_dev/_blender_bvh_probe.py -- <in.bvh> <out.npz>
Blender's importer converts the BVH frame (Y up, -Z forward) to its own (Z up, Y forward): (x, y, z) -> (x, -z, y); the
dump is converted back so it compares directly with the BVH's own coordinates."""
import sys
import bpy
import numpy as np

bvh, out = sys.argv[sys.argv.index("--") + 1:][:2]
bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_anim.bvh(filepath=bvh, axis_forward="-Z", axis_up="Y", global_scale=1.0, rotate_mode="NATIVE",
                        update_scene_fps=True, update_scene_duration=True, use_fps_scale=False)
ob = [o for o in bpy.context.scene.objects if o.type == "ARMATURE"][0]
sc = bpy.context.scene
names = [pb.name for pb in ob.pose.bones]
pos = []
for f in range(sc.frame_start, sc.frame_end + 1):
    sc.frame_set(f)
    pos.append([[*(ob.matrix_world @ pb.head)] for pb in ob.pose.bones])
pos = np.asarray(pos, dtype=np.float64)
pos = np.stack([pos[..., 0], pos[..., 2], -pos[..., 1]], axis=-1)      # back to the BVH frame
np.savez(out, names=np.array(names), positions=pos, fps=np.array(sc.render.fps / sc.render.fps_base),
         frame_start=np.array(sc.frame_start), frame_end=np.array(sc.frame_end))
print(f"[blender] {len(names)} bones x {pos.shape[0]} frames -> {out}")
