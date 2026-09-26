"""Assemble the supplementary video package: for every clip in the spec, copy the GT and generated skinned renders
(original H.264 mp4 from the render batches, no re-encoding) into <out>/videos/ with descriptive names, verify each
copy byte-for-byte and by a full decode to the frame count recorded in the batch's convert.json at 30 fps, cross-check
the batch's own records against the spec (rig, clip id, joint count, sampler settings), and write manifest.csv,
index.html (GT | generated side by side, prompt above, synchronised play/pause) and README.txt. The package is built in a
staging directory and swapped into place only when everything passed; the zip is written the same way.
  python scripts/_build_supp_videos.py --spec <spec.json> --out <dir>
spec.json: {"protocol": "<one paragraph>", "clips": [{"rig": ..., "clip_id": ..., "caption": ..., "joints": int,
            "gt_dir": <batch dir with videos/>, "gen_dir": <batch dir with videos/>, "views": ["side", ...]}]}
           optional "lora": {"protocol": "<one paragraph>", "dir": <out dir of scripts/_render_supp_lora_videos.py>}
Nothing in the output names people, machines or paths; species names come from the rig id."""
import argparse, csv, hashlib, html, json, os, shutil, sys, zipfile
import cv2
sys.path.insert(0, os.getcwd())
from src.data.caption_keys import ordered_captions   # the generator prompts with caption 0 of this ordering

SAMPLER = {"cfg_text": 2.0, "steps": 20, "seed": 42, "ckpt_epoch": 289,                              # what the protocol paragraph promises
           "ckpt_sha256": "9618342a45c6f1fdf2ee8b47aed1ae80b089f0df5e9d7f9db1a48578bc2eed84"}    # the paper's checkpoint (best_model.pt, epoch 289)
TEXTS_JSON = "data/noik_pzh312_motion_texts_v1.json"   # caption table the generator embeds from; pinned by sha in every generation manifest

ap = argparse.ArgumentParser()
ap.add_argument("--spec", required=True); ap.add_argument("--out", required=True)
a = ap.parse_args()
spec = json.load(open(a.spec)); root = a.out.rstrip("/")            # <root>/<build id>/{supplementary/, supplementary.zip}; <root>/current -> <build id>
import datetime, uuid
build_id = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
vdir = os.path.join(root, build_id); stage = os.path.join(vdir, "supplementary"); out = stage; vid_dir = os.path.join(stage, "videos")
texts = json.load(open(TEXTS_JSON))
os.makedirs(vid_dir)

def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""): h.update(b)
    return h.hexdigest()

def decode_all(path):
    """Decode every frame; returns (frames actually decoded, fps)."""
    cap = cv2.VideoCapture(path); fps = cap.get(cv2.CAP_PROP_FPS); n = 0
    while True:
        ok, _ = cap.read()
        if not ok: break
        n += 1
    cap.release(); return n, fps

def find_video(batch_dir, view):
    vd = os.path.join(batch_dir, "videos")
    hits = sorted(f for f in os.listdir(vd) if f.startswith(view + "_") and f.endswith(".mp4")) if os.path.isdir(vd) else []
    assert len(hits) == 1, (batch_dir, view, hits)
    path = os.path.join(vd, hits[0])
    # a render batch writes convert.json first, then re-renders videos/, then the _views_ok marker: a video older than
    # the conversion record is a leftover of an earlier render (an interrupted re-render), and is refused
    t_conv = os.path.getmtime(os.path.join(batch_dir, "convert.json")); t_ok = os.path.getmtime(os.path.join(batch_dir, "_views_ok")); t_vid = os.path.getmtime(path)
    assert t_conv < t_vid <= t_ok, ("video is not from this batch's render", path, t_conv, t_vid, t_ok)
    return path

def check_batch(c, kind):
    """The batch's own convert.json must agree with the spec; the generated batch must carry the promised sampler."""
    d = c[kind + "_dir"]; rec = json.load(open(os.path.join(d, "convert.json")))
    assert rec["rig"] == c["rig"] and rec["clip_id"] == c["clip_id"] and rec["ktjd_joints"] == c["joints"], (d, rec["rig"], rec["clip_id"], rec["ktjd_joints"])
    assert os.path.exists(os.path.join(d, "_views_ok")), ("render check marker missing", d)
    if kind == "gen":
        assert rec.get("source") == "generated", d
        g = rec["gen_manifest"]
        for k, v in SAMPLER.items(): assert g[k] == v, (d, k, g[k], v)
        assert g["clip_id"] == c["clip_id"], d
        # the prompt shown must be the prompt used: caption 0 of the pinned caption table (random_caption is off in the picker)
        assert g["ktjd_pins"]["texts_json_sha256"] == sha256(TEXTS_JSON), (d, "caption table differs from the one the generator embedded")
        assert ordered_captions(texts[c["clip_id"] + ".npy"])[0] == c["caption"], (d, c["caption"])
    else:
        assert str(rec.get("source", "")).startswith("gt"), (d, rec.get("source"))
        assert rec["caption"] == c["caption"], (d, rec["caption"], c["caption"])
    return rec

rows = []; total = 0
for i, c in enumerate(spec["clips"], 1):
    species = c["rig"].replace("PZ_", "")
    recs = {kind: check_batch(c, kind) for kind in ("gt", "gen")}
    for view in c["views"]:
        for kind in ("gt", "gen"):
            src = find_video(c[kind + "_dir"], view); expect = recs[kind]["frames"]
            name = f"{i:02d}_{species}_{c['clip_id'][:8]}_{kind}_{view}.mp4"; dst = os.path.join(vid_dir, name)
            shutil.copyfile(src, dst)
            assert sha256(src) == sha256(dst), ("copy differs", src)
            n, fps = decode_all(dst)
            assert n == expect and abs(fps - 30.0) < 0.1, (dst, n, expect, fps)
            total += os.path.getsize(dst)
            rows.append({"file": name, "clip": i, "species": species.replace("_", " "), "joints": c["joints"], "kind": "ground truth" if kind == "gt" else "generated",
                         "view": view, "frames": n, "seconds": round(n / 30.0, 2), "prompt": c["caption"]})

# optional second section (user 2026-09-23): the per-rig LoRA videos of scripts/_render_supp_lora_videos.py, skeleton
# renders of rest pose | generated. spec "lora": {"protocol": "<one paragraph>", "dir": <that script's --out>}. Every row
# of its manifest is re-checked here against its world dump, the adaptation corpus split and the adapter checkpoint.
lora = spec.get("lora"); lora_rows = []; ckpt_sha = {}
if lora:
    import numpy as np
    for k, r in enumerate(csv.DictReader(open(os.path.join(lora["dir"], "manifest.csv"))), 1):
        z = np.load(r["dump"], allow_pickle=True)
        assert str(z["motion_id"]) == r["clip"] and str(z["caption"]) == r["prompt"], (r["file"], "manifest row disagrees with its dump")
        split = [json.loads(l)["split"] for l in open(os.path.join(str(z["ktjd_root"]), "manifests/clips.jsonl"))
                 if json.loads(l)["clip_id"].endswith(r["clip"])]
        assert split == ["val"], (r["file"], "not exactly one held-out clip of its rig", split)
        if r["adapter_ckpt"] not in ckpt_sha: ckpt_sha[r["adapter_ckpt"]] = sha256(r["adapter_ckpt"])
        assert str(z["ckpt_sha256"]) == r["adapter_ckpt_sha256"] == ckpt_sha[r["adapter_ckpt"]], (r["file"], "adapter checkpoint changed")
        assert str(z["rig"]) == r["rig"] and len(z["parents"]) == int(r["joints"]), (r["file"], "rig or joint count disagrees with the dump")
        src = os.path.join(lora["dir"], r["file"])
        assert sha256(src) == r["sha256"], (src, "video differs from the one its manifest recorded")
        cap = cv2.VideoCapture(src); width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); cap.release()
        assert width == int(r["width"]) == 1280, (src, "not the two-panel (rest pose | generated) layout", width)
        name = f"L{k}_{r['rig']}_{r['clip'].split('___')[-1]}.mp4"; dst = os.path.join(vid_dir, name)
        shutil.copyfile(src, dst)
        assert sha256(src) == sha256(dst), ("copy differs", src)
        n, fps = decode_all(dst)
        assert n == int(r["frames"]) and abs(fps - 30.0) < 0.1, (dst, n, r["frames"], fps)
        total += os.path.getsize(dst)
        lora_rows.append({"file": name, "clip": f"L{k}", "species": f"{r['rig']} (Truebones)", "joints": int(r["joints"]),
                          "kind": "generated, per-rig LoRA adapter", "view": "skeleton", "frames": n, "seconds": round(n / 30.0, 2),
                          "prompt": r["prompt"]})
    rows += lora_rows

with open(os.path.join(stage, "manifest.csv"), "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

blocks = []
for i, c in enumerate(spec["clips"], 1):
    species = c["rig"].replace("PZ_", "").replace("_", " ")
    for view in c["views"]:
        gt = f"videos/{i:02d}_{c['rig'].replace('PZ_', '')}_{c['clip_id'][:8]}_gt_{view}.mp4"; gen = gt.replace("_gt_", "_gen_")
        ngt = next(r["frames"] for r in rows if r["file"] == os.path.basename(gt)); ngen = next(r["frames"] for r in rows if r["file"] == os.path.basename(gen))
        blocks.append(f"""
<section class="clip">
  <h2>{i:02d}. {html.escape(species)} <span class="meta">{c['joints']} joints &middot; {html.escape(view)} view</span></h2>
  <p class="prompt">&ldquo;{html.escape(c['caption'])}&rdquo;</p>
  <div class="pair">
    <figure><video src="{gt}" muted loop playsinline preload="metadata"></video><figcaption>real clip ({ngt} frames, 30 fps)</figcaption></figure>
    <figure><video src="{gen}" muted loop playsinline preload="metadata"></video><figcaption>generated ({ngen} frames, 30 fps)</figcaption></figure>
  </div>
  <button onclick="playPair(this)">play both from the start</button> <button onclick="pausePair(this)">pause</button>
</section>""")
intro = ("Each block shows one validation clip: the real clip on the left and the generated clip for the same prompt on the right, both "
         "skinned onto the same game mesh and rendered by the same automatic camera rule (the camera follows the animal's smoothed heading and "
         "sets its distance from the animal's extent, so the two views are framed alike but are not pixel-aligned), 30 fps. A generated clip is "
         "at most 240 frames long, so on longer clips it ends before the real one. The files are also listed in manifest.csv.")
lora_html = "" if not lora_rows else (
    "<h1>Rigs outside the training library: per-rig LoRA adapters</h1>\n<p>" + html.escape(lora["protocol"]) + "</p>" + "".join(f"""
<section class="clip">
  <h2>{r['clip']}. {html.escape(r['species'])} <span class="meta">{r['joints']} joints &middot; skeleton</span></h2>
  <p class="prompt">&ldquo;{html.escape(r['prompt'])}&rdquo;</p>
  <figure><video src="videos/{r['file']}" controls muted loop playsinline preload="metadata"></video><figcaption>rest pose (input) and generated motion ({r['frames']} frames, 30 fps)</figcaption></figure>
</section>""" for r in lora_rows))
page = f"""<!doctype html><html><head><meta charset="utf-8"><title>Supplementary videos</title>
<style>
body{{font-family:Georgia,serif;max-width:1100px;margin:2em auto;padding:0 1em;color:#222}}
h1{{font-size:1.4em}} h2{{font-size:1.1em;margin:1.6em 0 .2em}} .meta{{font-weight:normal;color:#666;font-size:.9em}}
.prompt{{margin:.2em 0 .6em;color:#444}} .pair{{display:flex;gap:12px}} figure{{margin:0;flex:1}} video{{width:100%;background:#111}}
figcaption{{font-size:.85em;color:#666;margin-top:.3em}} button{{margin-top:.5em;font:inherit}}
</style></head><body>
<h1>Rigs in the training library: text-to-motion samples played on the production meshes</h1>
<p>{html.escape(spec['protocol'])}</p>
<p>{html.escape(intro)}</p>
{''.join(blocks)}
{lora_html}
<script>
function playPair(btn){{const vs=btn.parentElement.querySelectorAll('video');vs.forEach(v=>{{v.pause();v.currentTime=0;}});
  vs.forEach(v=>{{const p=v.play();if(p&&p.catch)p.catch(()=>{{}});}});}}
function pausePair(btn){{btn.parentElement.querySelectorAll('video').forEach(v=>v.pause());}}
</script></body></html>"""
open(os.path.join(stage, "index.html"), "w").write(page)
open(os.path.join(stage, "BUILD_ID.txt"), "w").write(build_id + "\n")
open(os.path.join(stage, "README.txt"), "w").write(
    "Supplementary videos.\n\nOpen index.html in a browser (Chrome, Safari or Edge; the files are H.264 mp4) or play the files in videos/ directly.\n"
    "File name: <clip no>_<species>_<clip id>_<gt|gen>_<view>.mp4. manifest.csv lists the prompt, joint count and frame count of every file.\n\n"
    + spec["protocol"] + "\n\n" + intro + "\n"
    + ("" if not lora_rows else "\nRigs outside the training library (files L<no>_<rig>_<clip>.mp4): " + lora["protocol"] + "\n"))

# everything passed: the zip is built beside the folder inside the versioned directory, then ONE atomic symlink switch
# (<root>/current -> <build id>) publishes folder and zip together; earlier versions stay until removed by hand
zpath = os.path.join(vdir, "supplementary.zip")
with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
    for r_, _, files in os.walk(stage):
        for f in sorted(files):
            full = os.path.join(r_, f); z.write(full, os.path.join("supplementary", os.path.relpath(full, stage)))
with zipfile.ZipFile(zpath) as z:
    assert z.testzip() is None and z.read("supplementary/BUILD_ID.txt").decode().strip() == build_id
tmp_link = os.path.join(root, f".current.tmp.{build_id}"); os.symlink(build_id, tmp_link); os.replace(tmp_link, os.path.join(root, "current"))   # build-specific temp name: an interrupted run never blocks the next (codex r4)
print(f"{len(rows)} videos, {total/1e6:.1f} MB raw; published {root}/current -> {build_id}; zip {os.path.getsize(zpath)/1e6:.1f} MB")
