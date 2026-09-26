"""Tests of scripts/deploy_generate.py (single-rig inference without the corpus). GPU, inside an allocation, through
the per-card gate:
  srun --jobid=<alloc> --overlap -N1 -n1 --gres=gpu:<n> --cpus-per-task=4 --mem=64G /usr/bin/env HF_HUB_OFFLINE=1 \
       TRANSFORMERS_OFFLINE=1 python scripts/_gpu_gate_exec.py <idx> scripts/_aug_dev/_test_deploy_generate.py [--llm2vec]

P  parity: the 4 zero-shot clips of the H1 close-out (renders/h1_closeout/ood4_h1/ours_seed7/*.world.npz, made by
   scripts/v2_render_incontext.py --zero_shot) regenerated through deploy_generate from the skeleton npz, the cached
   caption embedding and the stored joint-semantic table: gen_ric / gen_fk must match (bitwise on the same GPU model).
C  rest convention: 24 corpus rigs, same text / joint semantics / noise, served with (a) the corpus's own rest frames,
   (b) the continuation rule deploy_generate applies to a BVH, (c) identity rest frames (a raw BVH); distances of the
   generated world motion from (a), with (a) seed 7 vs seed 17 as the yardstick.
B  BVH: the TrueBones chicken T-pose BVH (outside_docs/AnyTop/assets/Truebones_Chicken) imported from frame 0 under all
   24 axis pairs; the pair whose grounded rest pose matches the official KTJD-17 chicken skeleton
   (dataset/ktjd17_truebones/skeletons/Chicken.npz) up to scale must be unique with a tiny residual; generate and write
   the BVH (write_bvh reads it back and checks FK); also a corpus npz rig written as BVH.
L  (--llm2vec) LLM2Vec reproduction: the 4 captions and the 4 rigs' joint descriptions re-encoded here vs the cached
   caption rows / the stored table, and the P generations redone with the fresh embeddings (deviation in mean bone
   lengths vs the seed yardstick); then the chicken end to end from text.
"""
import argparse, glob, itertools, json, sys, time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, ".")
import scripts.deploy_generate as D                                                   # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", default="runs/_release/topox_h1_uniml3d73m_ep239_infer.pt")
ap.add_argument("--llm2vec", action="store_true")
ap.add_argument("--out", default="runs/_release/_deploy_test")
a = ap.parse_args()
OUT = Path(a.out); OUT.mkdir(parents=True, exist_ok=True)
dev = "cuda"
FAIL = []


def check(ok, msg):
    print(("ok: " if ok else "FAIL: ") + msg, flush=True)
    if not ok:
        FAIL.append(msg)


t0 = time.time()
model, ca, ck = D.load_model(a.ckpt, dev)
ROOT = "dataset/ktjd17_uniml3d_v2"
keys = json.load(open("data/uniml3d_caption_llm2vec_v2.keys.json"))
embs = np.load("data/uniml3d_caption_llm2vec_v2.embs.npy", mmap_mode="r")
row_of = {k: i for i, k in enumerate(keys)}
SEM = np.load("data/joint_semantics_llm2vec_uniml3d_v2corpus.npz", allow_pickle=False)


def cap_emb(clip):
    return np.asarray(embs[row_of[f"{clip}__cap0"]], dtype=np.float32)


def mbl(rig):
    par = np.asarray(rig["parents"], dtype=np.int64); P = np.asarray(rig["P_rest_global"], dtype=np.float64)
    return float(np.linalg.norm(P[1:] - P[par[1:]], axis=1).mean())


# ---------------- P. parity with scripts/v2_render_incontext.py --zero_shot ----------------
dumps = sorted(glob.glob("renders/h1_closeout/ood4_h1/ours_seed7/*.world.npz"))
check(len(dumps) == 4, f"P0 four zero-shot dumps found ({len(dumps)})")
par_cases = []
for f in dumps:
    z = np.load(f, allow_pickle=True)
    rigid, clip, T = str(z["rig"]), str(z["motion_id"]), int(z["gen_fk"].shape[0])
    rig = D.rig_from_npz(f"{ROOT}/skeletons/{rigid}.npz", reconvention=False)
    sem = SEM[f"emb__{rigid}"]
    seg, dec = D.generate(model, ca, rig, cap_emb(clip), sem, T, int(z["seed"]), int(z["steps"]), float(z["cfg_text"]), dev)
    e_r = float(np.abs(dec["positions_direct"] - z["gen_ric"]).max()); e_f = float(np.abs(dec["positions_fk"] - z["gen_fk"]).max())
    check(e_r == 0.0 and e_f == 0.0, f"P1 {rigid[:20]} J{len(rig['parents'])} T{T}: deploy == render dump (direct max |d| {e_r:.1e}, fk {e_f:.1e})")
    par_cases.append((rigid, clip, T, rig, sem, z, dec))
    err = D.write_bvh(OUT / f"P_{rigid}.bvh", rig, dec)
    check(err < 1e-4, f"P2 {rigid[:20]} npz rig written as BVH and read back: FK error {err:.1e} x s_rig")

# ---------------- C. rest convention ----------------
rows = [json.loads(l) for l in open(f"{ROOT}/manifests/clips.jsonl")]
excl = set(json.load(open("configs/uniml3d_v2_common_unimate_exclusions.json"))["clips"])     # {clip_id: reason}
by_rig = {}
for r in rows:
    if r["clip_id"] in excl or f"{r['clip_id']}__cap0" not in row_of:
        continue
    by_rig.setdefault(r["rig_id"], r["clip_id"])
rigs = sorted(by_rig)[::max(1, len(by_rig) // 24)][:24]
res = {"b_vs_a": [], "c_vs_a": [], "seed_a": [], "b_vs_a_direct": [], "c_vs_a_direct": [], "seed_a_direct": []}
for rigid in rigs:
    ra = D.rig_from_npz(f"{ROOT}/skeletons/{rigid}.npz", reconvention=False)
    rb = D.rig_from_npz(f"{ROOT}/skeletons/{rigid}.npz", reconvention=True)
    P = np.asarray(ra["P_rest_global"], dtype=np.float64)
    rc = D._finish_rig(ra["joint_names"], ra["parents"], P, np.tile(np.eye(3), (len(P), 1, 1)),
                       [str(k) for k in ra["rotation_source_kind"]], "identity")
    sem, emb, m = SEM[f"emb__{rigid}"], cap_emb(by_rig[rigid]), mbl(ra)
    out = {}
    for tag, rg, seed in (("a", ra, 7), ("a17", ra, 17), ("b", rb, 7), ("c", rc, 7)):
        out[tag] = D.generate(model, ca, rg, emb, sem, 120, seed, 20, 2.0, dev)[1]
    d = lambda x, y, k: float(np.linalg.norm(out[x][k] - out[y][k], axis=-1).mean() / m)
    res["b_vs_a"].append(d("b", "a", "positions_fk")); res["c_vs_a"].append(d("c", "a", "positions_fk")); res["seed_a"].append(d("a17", "a", "positions_fk"))
    res["b_vs_a_direct"].append(d("b", "a", "positions_direct")); res["c_vs_a_direct"].append(d("c", "a", "positions_direct")); res["seed_a_direct"].append(d("a17", "a", "positions_direct"))
med = {k: float(np.median(v)) for k, v in res.items()}
print(f"   C medians over {len(rigs)} rigs (mean per-joint distance, mean bone lengths): " + ", ".join(f"{k} {v:.3f}" for k, v in med.items()), flush=True)
json.dump({"rigs": rigs, "per_rig": res, "median": med}, open(OUT / "C_rest_convention.json", "w"), indent=1)
check(med["b_vs_a"] < med["c_vs_a"], f"C1 continuation-rule rest frames stay closer to the corpus frames than identity frames (fk {med['b_vs_a']:.3f} vs {med['c_vs_a']:.3f}; seed spread {med['seed_a']:.3f})")

# ---------------- B. BVH import / export ----------------
BVH = "outside_docs/AnyTop/assets/Truebones_Chicken/Chicken_TPOSE.bvh"
off = np.load("dataset/ktjd17_truebones/skeletons/Chicken.npz", allow_pickle=True)
on, oP = [str(x) for x in off["joint_names"]], np.asarray(off["P_rest_global"], dtype=np.float64)
fits = []
for up, fw in itertools.product(D.AXES, D.AXES):
    if abs(float(D.AXES[up] @ D.AXES[fw])) > 0.5:
        continue
    rig = D.rig_from_bvh(BVH, up, fw, 0)
    idx = [rig["joint_names"].index(n) for n in on]
    Q = np.asarray(rig["P_rest_global"])[idx]
    Qc, Oc = Q - Q[0], oP - oP[0]
    s = float((Qc * Oc).sum() / max((Qc * Qc).sum(), 1e-12))
    fits.append((float(np.abs(s * Qc - Oc).max() / np.ptp(Oc, axis=0).max()), up, fw, s))
fits.sort()
print("   B axis fits (residual / extent):", [(round(r, 4), u, f) for r, u, f, _ in fits[:4]], flush=True)
# the official TrueBones build (its own pipeline, source_to_canonical_C identical to this pair) differs from the T-pose file's
# direct FK by ~1.2% of the extent at the thighs / tail (every frame of the T-pose file alike): the threshold admits that,
# the uniqueness of the axis pair is the real test of the canonicalization
check(fits[0][0] < 0.03 and fits[1][0] > 10 * fits[0][0], f"B1 chicken BVH frame 0 under up {fits[0][1]} forward {fits[0][2]} (the official source_to_canonical_C) reproduces the official KTJD-17 rest pose up to scale (residual {fits[0][0]:.1e} of extent, next axis pair {fits[1][0]:.1e}; scale {fits[0][3]:.4g})")
UP, FW = fits[0][1], fits[0][2]
rig = D.rig_from_bvh(BVH, UP, FW, 0)
rig_o = D.rig_from_bvh(BVH, UP, FW, -1)
Pf, Po = np.asarray(rig["P_rest_global"]), np.asarray(rig_o["P_rest_global"])
d_off = float(np.abs((Po - Po[0]) - (Pf - Pf[0])).max() / np.ptp(Pf, axis=0).max())
check(d_off > 0.2, f"B2 the chicken's OFFSET-only pose is not its T-pose (differs by {d_off:.2f} of the extent) -- a 3ds Max Biped BVH keeps the rest pose in its first frame, hence --rest_frame 0 by default")
# A. axes are checked against the joint-name sides; the wrong forward axis is refused with the right one suggested
def axis_verdict(up, fw):
    r = D.rig_from_bvh(BVH, up, fw, 0)
    ag, n, f = D.side_check(r)
    cand = [k for k in D.AXES if abs(float(D.AXES[k] @ D.AXES[up])) < 0.5]
    return ag, n, max(cand, key=lambda k: float(D.AXES[k] @ (r["bvh"]["C"].T @ f)))
ag1, n1, s1 = axis_verdict(UP, FW)
ag2, n2, s2 = axis_verdict("+Y", "+Z")
check(ag1 == n1 and n1 >= 20 and ag2 < 0.8 * n2 and s1 == FW and s2 == FW,
      f"A1 side check: {ag1}/{n1} under +Y/{FW}, {ag2}/{n2} under +Y/+Z (refused), both suggest --forward {s2}")
import tempfile, os
tmpd = Path(tempfile.mkdtemp(dir=str(OUT)))
src_txt = Path(BVH).read_text()
SYN = """HIERARCHY
ROOT A
{{
  OFFSET 0 0 0
  CHANNELS {root}
  JOINT B
  {{
    OFFSET 0 1 0
    CHANNELS {child}
    End Site
    {{
      OFFSET 0 1 0
    }}
  }}
}}
MOTION
Frames: 2
Frame Time: 0.0333333
{rows}
"""
cases = [("two rotation axes", "6 Xposition Yposition Zposition Zrotation Xrotation Xrotation", "3 Zrotation Xrotation Yrotation",
          "only none or all three"),
         ("root without position channels", "3 Zrotation Xrotation Yrotation", "3 Zrotation Xrotation Yrotation",
          "no X/Y/Z position channels")]
refusals = []
for tag, rc, cc, why in cases:
    n = int(rc.split()[0]) + int(cc.split()[0])
    f = tmpd / f"{tag.replace(' ', '_')}.bvh"
    f.write_text(SYN.format(root=rc, child=cc, rows="\n".join(" ".join(["0"] * n) for _ in range(2))))
    try:
        D.rig_from_bvh(f, "+Y", "+Z", 0); refusals.append(f"{tag}: accepted")
    except SystemExit as e:
        if why not in str(e): refusals.append(f"{tag}: refused for another reason: {e}")
    except Exception as e:
        refusals.append(f"{tag}: crashed {type(e).__name__}: {e}")
check(not refusals, f"A2 unsupported BVH channels refused at import, before any generation ({refusals or 'two-axis rotation, root without positions'})")
lex = D.load_lexicon(D.DEFAULT_LEXICON)
# S. a key the corpus describes with a side the key function cannot see ('braco direito', Portuguese "right arm") takes
# the side from the geometry (+X = left), never from a learnt label: 4 corpus assets have names mirrored against geometry
names_s = ["Hips", "BracoDireito", "BracoEsquerdo", "MaoDireita", "MaoEsquerda"]
P_s = np.array([[0, 1.0, 0], [-0.20, 1.5, 0], [0.20, 1.5, 0], [-0.40, 1.5, 0], [0.40, 1.5, 0]])
rig_s = D._finish_rig(names_s, [-1, 0, 0, 1, 2], P_s, D.continuation_frames([-1, 0, 0, 1, 2], P_s), ["animated_dof"] * 5, "synthetic")
desc_s, src_s = D.describe_joints(rig_s, lex, None, None)
check(desc_s[1:] == ["Right Upper Arm joint.", "Left Upper Arm joint.", "Right Hand joint.", "Left Hand joint."],
      f"S1 foreign side words: sides read off the geometry ({desc_s[1:]})")
desc, src = D.describe_joints(rig, lex, None, None)
print("   B chicken descriptions:", {s: src.count(s) for s in set(src)}, desc[:6], flush=True)
cap_txt = json.load(open("data/uniml3d_motion_texts_v2.json"))          # {"<clip>.npy": {"primary_caption" (= cap0), "captions"}}
walk_clip = next(k[:-4] for k, v in sorted(cap_txt.items()) if "walks forward" in v["primary_caption"].lower() and f"{k[:-4]}__cap0" in row_of)
print(f"   B text: {cap_txt[walk_clip + '.npy']['primary_caption']!r}", flush=True)
sem_rows = np.stack([SEM[f"emb__{par_cases[0][0]}"][0]] * len(rig["parents"]))       # placeholder semantics for B only
seg, dec = D.generate(model, ca, rig, cap_emb(walk_clip), sem_rows, 90, 7, 20, 2.0, dev)
err = D.write_bvh(OUT / "B_chicken.bvh", rig, dec)
check(err < 1e-4, f"B3 chicken generation written into its own hierarchy (41 joints + End Sites, ZXY) and read back: FK error {err:.1e} x s_rig")
n2, fr2, _ = D.read_bvh(OUT / "B_chicken.bvh")
n1, _, _ = D.read_bvh(BVH)
rotcols = [i for i, c in enumerate(c for n in n2 for c in n.channels) if c.endswith("rotation")]
jump = float(np.abs(np.diff(fr2[:, rotcols], axis=0)).max())
check(jump < 180.0, f"W1 written rotation channels are continuous curves (largest frame-to-frame change {jump:.1f} deg)")
check([n.name for n in n2] == [n.name for n in n1] and all(np.allclose(x.offset, y.offset, atol=1e-5) for x, y in zip(n1, n2))
      and [x.channels for x in n1] == [x.channels for x in n2], "B4 written hierarchy = input hierarchy (names, End Sites, offsets, channel orders)")

# ---------------- L. LLM2Vec ----------------
if a.llm2vec:
    for rigid, clip, T, rig, sem, z, dec0 in par_cases:
        cap_text = cap_txt[clip + ".npy"]["primary_caption"]
        zz = np.load(f"{ROOT}/skeletons/{rigid}.npz", allow_pickle=False)
        descs = [str(x) for x in zz["joint_descriptions"]]
        cap, sem_new = D.encode_texts(cap_text, descs, dev)
        c0 = cap_emb(clip)
        cos_c = float(cap @ c0 / np.linalg.norm(cap) / np.linalg.norm(c0))
        cos_s = np.sum(sem_new * sem, 1) / np.linalg.norm(sem_new, axis=1) / np.linalg.norm(sem, axis=1)
        print(f"   L {rigid[:20]}: caption {cap_text[:50]!r} max|d| {np.abs(cap - c0).max():.3f} cos {cos_c:.5f}; joint semantics cos min {cos_s.min():.4f} median {np.median(cos_s):.4f}", flush=True)
        check(cos_c > 0.999, f"L1 {rigid[:20]} caption re-encoded like the cache (cos {cos_c:.5f})")
        check(float(cos_s.min()) > 0.95, f"L2 {rigid[:20]} joint descriptions re-encoded near the stored table (cos min {cos_s.min():.4f})")
        _, dec1 = D.generate(model, ca, rig, cap, sem_new, T, 7, 20, 2.0, dev)
        _, dec17 = D.generate(model, ca, rig, c0, sem, T, 17, 20, 2.0, dev)
        m = mbl(rig)
        dv = float(np.linalg.norm(dec1["positions_fk"] - dec0["positions_fk"], axis=-1).mean() / m)
        ds = float(np.linalg.norm(dec17["positions_fk"] - dec0["positions_fk"], axis=-1).mean() / m)
        print(f"   L {rigid[:20]}: fresh-embedding generation vs stored {dv:.3f} bl (seed 17 vs 7: {ds:.3f} bl)", flush=True)
    import subprocess
    e2e = OUT / "chicken_e2e_v2"
    r = subprocess.run([sys.executable, "scripts/deploy_generate.py", "--ckpt", a.ckpt, "--skeleton", BVH, "--up", UP, "--forward", FW,
                        "--text", "An object walks forward.", "--frames", "120", "--out", str(e2e), "--force"],
                       capture_output=True, text=True)
    print("\n".join(l for l in (r.stdout + r.stderr).splitlines() if ("[deploy]" in l and not l.startswith("    ")) or "Error" in l or "refuse" in l)[-3000:], flush=True)
    check(r.returncode == 0 and (e2e / "Chicken_TPOSE_s7.bvh").exists(), f"L3 chicken end to end from text (rc {r.returncode})")
    # the output can never replace the input skeleton; an existing output needs --force
    cp = tmpd / "rig.bvh"; cp.write_text(src_txt)
    r2 = subprocess.run([sys.executable, "scripts/deploy_generate.py", "--ckpt", a.ckpt, "--skeleton", str(cp), "--up", UP, "--forward", FW,
                         "--text_emb", str(e2e / "Chicken_TPOSE_s7.text_emb.npy"), "--joint_sem", str(e2e / "Chicken_TPOSE_s7.joint_sem.npy"),
                         "--out", str(tmpd), "--name", "rig"], capture_output=True, text=True)
    r3 = subprocess.run([sys.executable, "scripts/deploy_generate.py", "--ckpt", a.ckpt, "--skeleton", BVH, "--up", UP, "--forward", FW,
                         "--text_emb", str(e2e / "Chicken_TPOSE_s7.text_emb.npy"), "--joint_sem", str(e2e / "Chicken_TPOSE_s7.joint_sem.npy"),
                         "--out", str(e2e)], capture_output=True, text=True)
    r4 = subprocess.run([sys.executable, "scripts/deploy_generate.py", "--ckpt", a.ckpt, "--skeleton", BVH, "--up", "+Y", "--forward", "+Z",
                         "--describe_only", "--out", str(tmpd), "--name", "wrongaxis"], capture_output=True, text=True)
    check(r4.returncode == 0 and "WARNING axes" in r4.stdout and "mirrored" in r4.stdout and (tmpd / "wrongaxis.rest.gif").exists(),
          "D1 --describe_only with a wrong --forward warns (names imply +X, or mirrored names) and still writes the rest preview")
    check(r2.returncode != 0 and "overwrite the input skeleton" in r2.stdout + r2.stderr and cp.read_text() == src_txt
          and r3.returncode != 0 and "--force" in r3.stdout + r3.stderr,
          "O1 an output path equal to the input skeleton is refused (input untouched); an existing output needs --force")

print(f"{'ALL PASS' if not FAIL else str(len(FAIL)) + ' FAIL'} ({time.time() - t0:.0f} s)", flush=True)
