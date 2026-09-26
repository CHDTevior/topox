"""Supplementary videos for the per-rig LoRA arm (paper Table tab:lora / Figure fig:tb_main).

Re-renders the world dumps that the figure already uses -- renders/cmp_dump_<rig>_lora/<clip>.world.npz, written by
scripts/v2_render_incontext.py --dump_world from the per-rig adapters on the 303M backbone snapshot of the paper's
extension study -- as clean H.264 mp4s:
rest pose (input) | generated (rotations played through FK), in the large-panel style of scripts/_pil_skeleton_render.py
(each panel root-centred, ground grid, root trail, prompt band on top). The real clip is not shown (user 2026-09-23:
"lora别放real"). No internal labels (checkpoint, epoch, sampler settings) are drawn. Read-only on the dumps; writes
<out>/<NN>_<Rig>__<clip>.mp4 and <out>/manifest.csv, NN being the clip's place in the full ordered list of dumps (the
numbers the candidates were reviewed under). Refuses unless every clip is a held-out ('val') clip of its rig in the
adaptation corpus and the adapter checkpoint on disk still has the sha256 the dump recorded.

usage (inside an alloc):  python scripts/_render_supp_lora_videos.py --out renders/supp_lora_tb_v2 --pick 1,5,6,7,9,11,15
"""
from __future__ import annotations
import argparse, csv, glob, hashlib, json, subprocess, sys
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
import matplotlib  # noqa: E402
from PIL import ImageFont  # noqa: E402
import _pil_skeleton_render as psr  # noqa: E402
from _pil_skeleton_render import compute_transform, make_row_frame  # noqa: E402

# The compute nodes have none of the system fonts psr.get_font looks for, so its fallback is PIL's tiny bitmap font.
# Use the DejaVu Sans that matplotlib ships, for this script only (the shared module is left as it is).
_FONT = Path(matplotlib.__file__).parent / "mpl-data/fonts/ttf/DejaVuSans.ttf"
psr.get_font = lambda size: ImageFont.truetype(str(_FONT), size)

RIGS = ["buffalo", "gazelle", "dragon", "spider"]          # the four rigs of Table tab:lora / Figure fig:tb_main
REST, GEN = (13, 110, 100), (91, 44, 184)
FFMPEG = sorted(glob.glob("/iridisfs/scratch/ts1v23/workspace/unimate_venv/lib/python3*/site-packages/"
                          "imageio_ffmpeg/binaries/ffmpeg-linux-x86_64-*"))
CELL, HEADER_H = (640, 560), 76                             # 2 x 640 = 1280 wide, 636 high: both even (yuv420p)


def hold(seq, T):
    """Pad to T frames by holding the last frame (the rest pose is one frame)."""
    return np.concatenate([seq, np.repeat(seq[-1:], T - len(seq), axis=0)]) if len(seq) < T else seq


def root_centred(seq):
    s = seq.copy()
    s[..., 0] -= s[:, :1, 0]
    s[..., 2] -= s[:, :1, 2]
    return s


def encode(frames, out_path, fps):
    w, h = frames[0].size
    cmd = [FFMPEG[-1], "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}",
           "-r", str(fps), "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18",
           "-movflags", "+faststart", str(out_path)]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for f in frames:
        p.stdin.write(np.asarray(f.convert("RGB"), dtype=np.uint8).tobytes())
    p.stdin.close()
    if p.wait() != 0:
        raise SystemExit(f"[refuse] ffmpeg failed on {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--pick", default="", help="comma-separated NN numbers to render (default: all)")
    a = ap.parse_args()
    pick = {int(v) for v in a.pick.split(",") if v.strip()}
    if not FFMPEG:
        raise SystemExit("[refuse] no ffmpeg binary found")
    a.out.mkdir(parents=True, exist_ok=True)
    rows, nn = [], 0
    for rig in RIGS:
        for path in sorted(glob.glob(str(REPO / f"renders/cmp_dump_{rig}_lora/*.world.npz"))):
            nn += 1
            if pick and nn not in pick:
                continue
            z = np.load(path, allow_pickle=True)
            clip, rig_name = str(z["motion_id"]), str(z["rig"])
            corpus = REPO / str(z["ktjd_root"])
            man = [json.loads(l) for l in open(corpus / "manifests/clips.jsonl")]
            hit = [r for r in man if r["clip_id"] == clip or r["clip_id"].endswith(clip)]
            if len(hit) != 1 or hit[0]["split"] != "val":
                raise SystemExit(f"[refuse] {clip}: not exactly one held-out ('val') clip in {corpus}")
            ckpt = REPO / str(z["ckpt"])
            if hashlib.sha256(ckpt.read_bytes()).hexdigest() != str(z["ckpt_sha256"]):
                raise SystemExit(f"[refuse] {ckpt}: sha256 differs from the one the dump recorded")
            parents = np.asarray(z["parents"]).astype(int)
            gen = np.asarray(z["gen_fk"], float)
            T = len(gen)
            rest = hold(np.asarray(z["demo_w"], float)[:1], T)
            transform = compute_transform([root_centred(s) for s in (rest, gen)], CELL, 0.12, 1.0)
            panels = [dict(positions=rest, parents=parents, title="rest pose (input)", color=REST, static=True),
                      dict(positions=gen, parents=parents, title="generated", color=GEN)]
            header = f"{rig_name} ({len(parents)} joints) · per-rig LoRA adapter · “{z['caption']}”"
            frames = [make_row_frame(panels, t, transform, CELL, 5, 7, header=header, header_h=HEADER_H)
                      for t in range(T)]
            name = f"{nn:02d}_{rig_name}__{clip}.mp4"
            fps = float(z["fps"])
            encode(frames, a.out / name, fps)
            rows.append(dict(file=name, sha256=hashlib.sha256((a.out / name).read_bytes()).hexdigest(),
                             width=frames[0].size[0], rig=rig_name, joints=len(parents), clip=clip, split="val", frames=T,
                             seconds=round(T / fps, 2), prompt=str(z["caption"]),
                             dump=str(Path(path).relative_to(REPO)), adapter_ckpt=str(z["ckpt"]),
                             adapter_ckpt_sha256=str(z["ckpt_sha256"])))
            print(f"[ok] {name}  T={T}")
    if pick and len(rows) != len(pick):
        raise SystemExit(f"[refuse] asked for {sorted(pick)}, rendered {[r['file'][:2] for r in rows]}")
    with open(a.out / "manifest.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"[done] {len(rows)} videos -> {a.out}")


if __name__ == "__main__":
    main()
