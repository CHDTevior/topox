#!/usr/bin/env python
"""Pick and package demo rigs for scripts/deploy_generate.py (user 2026-09-26: "打包的项目里多放一些比较好的物种例子，
训练的也可以，当作 demo"; the user chose the public branch and an automatic pick).

stage "score" (GPU): for candidate rigs of the UniML3D v2 common training set (per body plan, the rigs with the most
training clips), generate each rig's most energetic training captions through deploy_generate's npz path with the corpus
rest frames (--keep_rest_rotations: exactly the training setup), cached caption embedding and stored joint-semantic
table, seeds 7 and 17, and score against the clip's own ground truth (decoded like the renders):
    motion      joint motion relative to the root, gen / GT (0.7-1.4 = neither frozen nor over-excited)
    gap         max |position decode - rotation decode| / s_rig (the BVH plays the rotation decode)
    jitter      mean |acceleration| gen / GT
    float       median lowest-joint height, gen - GT, in mean bone lengths
-> runs/_release/_demos/scores.json
stage "licenses" (login node, network): the Sketchfab licence of every candidate rig -> sketchfab_licenses_all.json
stage "score2" (GPU): like "score" for the CC-BY / CC0 rigs of an extended pool -> scores2.json
stage "select": CLEAN, the ID-title rule and the curation below (NAMES / DROP / RELAX, decided by looking at the generated
animations) -> selection.json; the same scores give the same file
stage "relicense" (login node, network): each selected model's licence read again from Sketchfab, with its URL and the date
stage "package" (GPU): the chosen (rig, caption) pairs -> demo/<name>/ (skeleton.npz with the fields the deploy script reads,
joint_sem.npy, text_emb_<k>.npy, prompts.json, ref_<k>.{gif,bvh} made by the deploy CLI itself) + demo/README.md +
demo/run_demos.sh
"""
import argparse, csv, json, re, sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

R = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(R))
ROOT = R / "dataset/ktjd17_uniml3d_v2"
OUT = R / "runs/_release/_demos"


def candidates(per_plan, sketchfab_only=False, exclude=(), min_clips=1):
    cut = set(json.load(open(R / "configs/uniml3d_v2_common_unimate_exclusions.json"))["clips"])
    rows = [json.loads(l) for l in open(ROOT / "manifests/clips.jsonl")]
    train = [r for r in rows if r["split"] == "train" and r["clip_id"] not in cut]
    clips = defaultdict(list)
    for r in train:
        clips[r["rig_id"]].append(r)
    cen = {r["rig_id"]: r for r in csv.DictReader(open(ROOT / "analysis/rig_census.csv"))}
    by_plan = defaultdict(list)
    for rig, cl in clips.items():
        if rig in cen:
            by_plan[cen[rig]["body_plan"]].append((len(cl), rig))
    out = []
    for plan, lst in by_plan.items():
        if plan == "uncertain":
            continue
        lst = [(n, rig) for n, rig in lst if n >= min_clips and rig not in exclude
               and (not sketchfab_only or len(rig[4:]) == 32)]
        for n, rig in sorted(lst, reverse=True)[:per_plan.get(plan, 3)]:
            out.append({"rig": rig, "plan": plan, "n_train_clips": n, "clips": [r["clip_id"] for r in clips[rig]],
                        "rows": {r["clip_id"]: r for r in clips[rig]}})
    return out


def sketchfab_license(uid, cache):
    """Sketchfab's public model API (no auth): name, author, licence label/slug/URL, viewer URL and the date asked; cached
    in a JSON file. A failed lookup is not cached, so the next run asks again."""
    import subprocess, time
    if uid in cache:
        return cache[uid]
    r = subprocess.run(["curl", "-s", "--max-time", "20", f"https://api.sketchfab.com/v3/models/{uid}"], capture_output=True, text=True)
    try:
        d = json.loads(r.stdout)
    except Exception:
        d = {}
    lic = d.get("license") or {}
    if not lic.get("slug"):
        return {"license": None}
    cache[uid] = {"name": d.get("name"), "author": (d.get("user") or {}).get("displayName"),
                  "author_url": (d.get("user") or {}).get("profileUrl"), "license": lic.get("label"),
                  "license_slug": lic.get("slug"), "license_url": lic.get("url"), "url": d.get("viewerUrl"),
                  "retrieved": time.strftime("%Y-%m-%d", time.gmtime())}
    return cache[uid]


def licenses(a):
    """Login-node stage (network): licence of every Sketchfab candidate rig of the extended pool."""
    lic_p = OUT / "sketchfab_licenses_all.json"
    cache = json.loads(lic_p.read_text()) if lic_p.exists() else {}
    per_plan = {"bipedal": 60, "quadrupedal": 40, "insectoid": 20, "avian": 25, "marine": 15, "serpentine": 5}
    for c in candidates(per_plan, sketchfab_only=True):
        sketchfab_license(c["rig"][4:], cache)
    lic_p.write_text(json.dumps(cache, indent=1))
    ok = {k: v for k, v in cache.items() if v.get("license_slug") in ("by", "cc0")}
    print(f"[licenses] {len(cache)} Sketchfab rigs checked, {len(ok)} CC-BY / CC0 -> {lic_p}")


def score(a):
    import torch
    import scripts.deploy_generate as D
    from src.data.ktjd17.loader import load_motion_npz
    from src.data.ktjd17.decoder import decode_ktjd17
    OUT.mkdir(parents=True, exist_ok=True)
    keys = json.load(open(R / "data/uniml3d_caption_llm2vec_v2.keys.json"))
    embs = np.load(R / "data/uniml3d_caption_llm2vec_v2.embs.npy", mmap_mode="r")
    row_of = {k: i for i, k in enumerate(keys)}
    texts = json.load(open(R / "data/uniml3d_motion_texts_v2.json"))
    SEM = np.load(R / "data/joint_semantics_llm2vec_uniml3d_v2corpus.npz", allow_pickle=False)
    model, ca, _ = D.load_model(R / "runs/_release/topox_h1_uniml3d73m_ep239_infer.pt", "cuda")
    if a.stage == "score2":
        # the extended pool: Sketchfab rigs whose licence is CC-BY / CC0, not scored in the first pass
        cache = json.loads((OUT / "sketchfab_licenses_all.json").read_text())
        done = {r["rig"] for r in json.loads((OUT / "scores.json").read_text())}
        per_plan = {"bipedal": 14, "quadrupedal": 12, "insectoid": 6, "avian": 8, "marine": 5, "serpentine": 3}
        pool = [c for c in candidates({k: 999 for k in per_plan}, sketchfab_only=True, exclude=done)
                if (cache.get(c["rig"][4:]) or {}).get("license_slug") in ("by", "cc0")]
        cands, taken = [], Counter()
        for c in sorted(pool, key=lambda c: -c["n_train_clips"]):
            if taken[c["plan"]] < per_plan.get(c["plan"], 0):
                cands.append(c); taken[c["plan"]] += 1
        out_name = "scores2.json"
    else:
        per_plan = {"bipedal": 6, "quadrupedal": 6, "insectoid": 4, "avian": 5, "marine": 4, "serpentine": 3}
        cands, out_name = candidates(per_plan), "scores.json"
    res = []
    for c in cands:
        rig = D.rig_from_npz(ROOT / "skeletons" / f"{c['rig']}.npz", reconvention=False)
        par = np.asarray(rig["parents"], dtype=np.int64)
        P = np.asarray(rig["P_rest_global"], dtype=np.float64)
        mb = float(np.linalg.norm(P[1:] - P[par[1:]], axis=1).mean())
        sem = SEM[f"emb__{c['rig']}"]
        gts = []
        for clip in c["clips"]:
            if f"{clip}__cap0" not in row_of:
                continue
            pay = load_motion_npz(ROOT / c["rows"][clip]["motion_relpath"], expected_fps_target=30.0)
            raw = np.asarray(pay["motion"], dtype=np.float64)[:240, :, :17]
            gt = decode_ktjd17(raw, parents=rig["parents"], R_rest_global=rig["R_rest_global"], R_rest_local=rig["R_rest_local"],
                               offset_parent_local=rig["offset_parent_local"], rotation_source_kind=rig["rotation_source_kind"],
                               strict_gt=True).positions_direct
            gts.append((motion_energy(gt, mb), clip, gt))
        gts.sort(key=lambda t: -t[0])
        seen_caps = set()
        for _, clip, gt in gts:
            cap = texts[f"{clip}.npy"]["primary_caption"]
            if cap in seen_caps:
                continue
            seen_caps.add(cap)
            T = len(gt)
            emb = np.asarray(embs[row_of[f"{clip}__cap0"]], dtype=np.float32)
            for seed in (7, 17):
                _, dec = D.generate(model, ca, rig, emb, sem, T, seed, 20, 2.0, "cuda")
                res.append({"rig": c["rig"], "plan": c["plan"], "n_train_clips": c["n_train_clips"], "clip": clip,
                            "caption": cap, "T": T, "seed": seed, "J": len(par), **metrics(dec, gt, mb, rig)})
            if len(seen_caps) >= 3:
                break
        print(f"[score] {c['plan']:11s} {c['rig']} J{len(par)} {len(seen_caps)} captions", flush=True)
        (OUT / out_name).write_text(json.dumps(res, indent=1))
    print(f"[score] {len(res)} generations -> {OUT / out_name}")


def motion_energy(p, mb):
    rel = p - p[:, :1]
    return float(np.linalg.norm(np.diff(rel, axis=0), axis=-1).mean() / mb) if len(p) > 1 else 0.0


def metrics(dec, gt, mb, rig):
    f = dec["positions_fk"]
    acc = lambda x: float(np.linalg.norm(x[2:] - 2 * x[1:-1] + x[:-2], axis=-1).mean()) if len(x) > 2 else 0.0
    e_g, e_t = motion_energy(f, mb), motion_energy(gt, mb)
    return {"motion": e_g / max(e_t, 1e-9), "gt_motion_bl_per_frame": e_t,
            "gap": float(np.abs(dec["positions_direct"] - f).max() / rig["s_rig"]),
            "jitter": acc(f) / max(acc(gt), 1e-9),
            "float_bl": float(np.median(f[..., 1].min(1)) - np.median(gt[..., 1].min(1))) / mb}


CLEAN = dict(motion=(0.7, 1.4), gap=0.15, jitter=1.8, float_bl=0.4)


def clean(r):
    return (CLEAN["motion"][0] <= r["motion"] <= CLEAN["motion"][1] and r["gap"] <= CLEAN["gap"]
            and r["jitter"] <= CLEAN["jitter"] and abs(r["float_bl"]) <= CLEAN["float_bl"])


def badness(r):
    """lower is better: log-distance of the motion ratio from 1, the decode gap, jitter excess, floating"""
    return abs(np.log(max(r["motion"], 1e-6))) + 2 * r["gap"] + 0.3 * max(0.0, r["jitter"] - 1) + 0.5 * abs(r["float_bl"])


# Curation, decided by looking at the generated animations (2026-09-26). select() applies it, so re-running the stages
# rebuilds the same package; a rig that becomes eligible later stops select() until it has been looked at and named here.
NAMES = {"OBJ_d8b6a381f36c46f8b1d59ed6e0b57c65": "triceratops", "OBJ_4798d8c87a0e4ad8835217fe93ddf67b": "stag",
         "OBJ_c0baad2baee5467894087856cac1872b": "wolf_lowpoly", "OBJ_ffd55e32c04c498681ed11584bdd49a5": "bear",
         "OBJ_27ba717c173a40b7841d2f2c6a89d823": "mech_striker", "OBJ_cef34035d74c4dfdb9cad45fa36da294": "viking_worker",
         "OBJ_a87532a3e89947159cc1303008c06eaf": "crawling_human", "OBJ_237ce5d21a3c4bcab51f34a4ab451587": "spider",
         "OBJ_614ab66892894acabcda5cc4a94f87fe": "bat"}
DROP = {"OBJ_361c8d7745704879850b74308bc2549a": "whose training clip is only 15 frames",       # pass CLEAN, left out
        "OBJ_241619b4423148029ceecddbbe691af0": "whose training clip is only 15 frames",
        "OBJ_b1282c04dffd4a44a8d755171337e143": "whose source rest pose stands upright while its motion lies on its side"}
RELAX = {"OBJ_237ce5d21a3c4bcab51f34a4ab451587": ("c0eded99414f85b937fa", 17),    # just outside CLEAN, admitted for
         "OBJ_614ab66892894acabcda5cc4a94f87fe": ("c9fc6ff29aa4a5b62c89", 7)}     # body-plan variety: rig -> (clip, seed)
ID_TITLE = re.compile(r"\d{6,}")     # a long digit run in a model title reads as a person's ID number: such models are skipped


def outside(r):
    """the CLEAN bounds a generation misses, in words"""
    out = []
    if not CLEAN["motion"][0] <= r["motion"] <= CLEAN["motion"][1]:
        out.append(f"{r['motion']:.2f}x motion")
    if r["gap"] > CLEAN["gap"]:
        out.append(f"decode gap {r['gap']:.2f}")
    if r["jitter"] > CLEAN["jitter"]:
        out.append(f"{r['jitter']:.1f}x jitter")
    if abs(r["float_bl"]) > CLEAN["float_bl"]:
        out.append(f"{r['float_bl']:+.2f} bone lengths off the ground")
    return ", ".join(out)


def select(a):
    """-> runs/_release/_demos/selection.json: per rig the (caption, seed) generations that pass CLEAN, best first, at most
    2 captions per rig, rigs ranked by their best generation and capped per body plan, after DROP and the ID-title rule;
    then the RELAX picks. Every selected rig must be named in NAMES."""
    res = []
    for f in ("scores.json", "scores2.json"):
        if (OUT / f).exists():
            res += json.loads((OUT / f).read_text())
    cache = json.loads((OUT / "sketchfab_licenses_all.json").read_text())
    for k, v in json.loads((OUT / "sketchfab_licenses.json").read_text()).items():
        cache.setdefault(k[4:], v)
    ok = [r for r in res if len(r["rig"][4:]) == 32 and (cache.get(r["rig"][4:]) or {}).get("license_slug", "by" if (cache.get(r["rig"][4:]) or {}).get("license") == "CC Attribution" else None) in ("by", "cc0") and clean(r)]
    left_out = {}
    for r in ok:
        if r["rig"] in DROP:
            left_out[r["rig"]] = DROP[r["rig"]]
        elif ID_TITLE.search(cache[r["rig"][4:]].get("name") or ""):
            left_out[r["rig"]] = "whose model title contains an ID-like number"
    ok = [r for r in ok if r["rig"] not in left_out]
    per_rig = defaultdict(dict)
    for r in sorted(ok, key=badness):
        per_rig[r["rig"]].setdefault(r["caption"], r)             # best seed per caption
    rigs = sorted(per_rig, key=lambda k: min(badness(r) for r in per_rig[k].values()))
    cap = {"bipedal": a.max_bipedal, "quadrupedal": 4, "insectoid": 3, "avian": 3, "marine": 2, "serpentine": 2}
    taken, sel = Counter(), []
    for rig in rigs:
        rs = sorted(per_rig[rig].values(), key=badness)[:2]
        plan = rs[0]["plan"]
        if taken[plan] >= cap.get(plan, 2):
            continue
        taken[plan] += 1
        lic = cache[rig[4:]]
        sel.append({"rig": rig, "plan": plan, "license": lic, "picks": [{k: r[k] for k in ("clip", "caption", "seed", "T", "motion", "gap", "jitter", "float_bl")} for r in rs]})
    for rig, (clip, seed) in RELAX.items():
        assert rig not in {s["rig"] for s in sel}, f"{rig} is already selected on its own"
        assert rig not in DROP and not ID_TITLE.search(cache[rig[4:]].get("name") or ""), f"{rig} is excluded by DROP / ID_TITLE"
        r = next(r for r in res if r["rig"] == rig and r["clip"] == clip and r["seed"] == seed)
        sel.append({"rig": rig, "plan": r["plan"], "license": cache[rig[4:]], "relaxed": outside(r),
                    "picks": [{k: r[k] for k in ("clip", "caption", "seed", "T", "motion", "gap", "jitter", "float_bl")}]})
    unnamed = [s["rig"] for s in sel if s["rig"] not in NAMES]
    if unnamed:
        raise SystemExit(f"[refuse] newly eligible rigs {unnamed}: look at their animations, then name them in NAMES or add them to DROP")
    for s in sel:
        s["name"] = NAMES[s["rig"]]
    (OUT / "selection.json").write_text(json.dumps({"rigs": sel, "left_out": sorted(left_out.values())}, indent=1))
    for s in sel:
        print(f"{s['plan']:11s} {s['name']:15s} {(s['license'].get('name') or '')[:28]:28s} " + " | ".join(f"{p['caption'][:40]} s{p['seed']} m{p['motion']:.2f} g{p['gap']:.2f}" for p in s["picks"]))
    print(f"[select] {len(sel)} rigs, {sum(len(s['picks']) for s in sel)} demos; per plan {dict(taken)}; left out {len(left_out)}")


def relicense(a):
    """Login-node stage (network): each selected model's licence read again from Sketchfab, with the licence URL and the
    date, into selection.json; refuses a failed lookup or a licence other than CC BY / CC0."""
    p = OUT / "selection.json"
    sel = json.loads(p.read_text())
    for s in sel["rigs"]:
        lic = sketchfab_license(s["rig"][4:], {})
        if lic.get("license_slug") not in ("by", "cc0") or not lic.get("license_url"):
            raise SystemExit(f"[refuse] {s['name']}: licence lookup gave {lic}")
        s["license"] = lic
        print(f"[relicense] {s['name']:15s} {lic['license']:16s} {lic['license_url']} {lic['retrieved']}")
    p.write_text(json.dumps(sel, indent=1))


def cc_label(lic):
    """('CC BY 4.0', https link) from the licence URL Sketchfab returns"""
    u = (lic.get("license_url") or "").replace("http://", "https://")
    m = re.search(r"/licenses/([a-z-]+)/([0-9.]+)/", u) or re.search(r"/publicdomain/(zero)/([0-9.]+)/", u)
    if not m:
        raise SystemExit(f"[refuse] no Creative Commons licence URL for {lic.get('name')!r}: run the relicense stage")
    return ("CC0" if m.group(1) == "zero" else "CC " + m.group(1).upper()) + " " + m.group(2), u


UNIML3D_BIB = ["@article{mou2026unimate,",
               "  title   = {UniMate: One Unified Model to Animate Diverse Skeletons},",
               "  author  = {Mou, Linzhan and Lei, Jiahui and Dou, Zhiyang and Cai, Chenyue and Song, Chaoyue and Finkelstein, Adam and Rusinkiewicz, Szymon},",
               "  journal = {arXiv preprint arXiv:2609.05415},",
               "  year    = {2026}",
               "}"]


def package(a):
    """selection.json -> demo/<name>/ + demo/README.md + demo/run_demos.sh (GPU: the references are made by the deploy CLI)."""
    import shutil, subprocess
    selection = json.loads((OUT / "selection.json").read_text())
    sel, left_out = selection["rigs"], selection["left_out"]
    for s in sel:
        cc_label(s["license"])                           # refuse before anything is deleted
    DEMO = R / "demo"
    if DEMO.exists():
        shutil.rmtree(DEMO)
    DEMO.mkdir()
    keys = json.load(open(R / "data/uniml3d_caption_llm2vec_v2.keys.json"))
    embs = np.load(R / "data/uniml3d_caption_llm2vec_v2.embs.npy", mmap_mode="r")
    row_of = {k: i for i, k in enumerate(keys)}
    SEM = np.load(R / "data/joint_semantics_llm2vec_uniml3d_v2corpus.npz", allow_pickle=False)
    ckpt = R / "runs/_release/topox_h1_uniml3d73m_ep239_infer.pt"
    used, table, runs = set(), [], []
    for s in sel:
        lic = s["license"]
        name = s["name"]
        assert name not in used, name
        used.add(name)
        d = DEMO / name
        d.mkdir()
        z = np.load(ROOT / "skeletons" / f"{s['rig']}.npz", allow_pickle=False)
        np.savez(d / "skeleton.npz", **{k: z[k] for k in ("joint_names", "parents", "P_rest_global", "R_rest_global", "R_rest_local",
                                                          "offset_parent_local", "rotation_source_kind", "s_rig", "joint_descriptions")})
        np.save(d / "joint_sem.npy", np.asarray(SEM[f"emb__{s['rig']}"], dtype=np.float32))
        prompts = []
        for k, pk in enumerate(s["picks"], 1):
            np.save(d / f"text_emb_{k}.npy", np.asarray(embs[row_of[f"{pk['clip']}__cap0"]], dtype=np.float32))
            cmd = [sys.executable, str(R / "scripts/deploy_generate.py"), "--ckpt", str(ckpt), "--skeleton", str(d / "skeleton.npz"),
                   "--keep_rest_rotations", "--text_emb", str(d / f"text_emb_{k}.npy"), "--joint_sem", str(d / "joint_sem.npy"),
                   "--frames", str(pk["T"]), "--seed", str(pk["seed"]), "--out", str(d), "--name", f"ref_{k}", "--force"]
            r = subprocess.run(cmd, capture_output=True, text=True, cwd=str(R))
            if r.returncode != 0 or not (d / f"ref_{k}.bvh").exists():
                raise SystemExit(f"[refuse] {name} ref_{k}: deploy CLI rc {r.returncode}\n{(r.stdout + r.stderr)[-1500:]}")
            for extra in (d / f"ref_{k}.npz", d / f"ref_{k}.descriptions.json"):
                extra.unlink(missing_ok=True)           # large / redundant; the gif and the BVH are the references
            prompts.append({"text": pk["caption"], "text_emb": f"text_emb_{k}.npy", "seed": pk["seed"], "frames": pk["T"],
                            "reference": [f"ref_{k}.gif", f"ref_{k}.bvh"],
                            "scores_vs_training_clip": {kk: round(pk[kk], 3) for kk in ("motion", "gap", "jitter", "float_bl")}})
            runs.append((name, k, pk))
            print(f"[package] {name} ref_{k}: {pk['caption']}", flush=True)
        (d / "prompts.json").write_text(json.dumps({"rig": s["rig"], "body_plan": s["plan"], "joints": int(len(z["parents"])),
                                                    "source": {**lic, "object_id": s["rig"][4:]}, "prompts": prompts}, indent=1))
        table.append((name, s, lic, prompts, int(len(z["parents"]))))
    words = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six"}
    lo = [f"{words.get(n, n)} rig{'s' if n > 1 else ''} {reason}" for reason, n in Counter(left_out).items()]
    relaxed = [f"`{name}`: {s['relaxed']}" for name, s, _, _, _ in table if s.get("relaxed")]
    and_join = lambda xs: xs[0] if len(xs) == 1 else ", ".join(xs[:-1]) + " and " + xs[-1]
    lines = ["# Demo rigs", "",
             "Ready-to-run examples for `scripts/deploy_generate.py` with the H1 checkpoint: skeletons from the training set, "
             "their training captions, and the embeddings the model was trained with, so no text encoder (LLM2Vec / Llama-3-8B) is "
             "needed to run them. Each folder holds `skeleton.npz` (KTJD-17 skeleton, the fields the deploy script reads), "
             "`joint_sem.npy`, `text_emb_<k>.npy`, `prompts.json` and the reference output `ref_<k>.gif` / `ref_<k>.bvh` of each "
             "prompt (made on an H200 in fp32 by the same command; another GPU may differ in the last digits).", "",
             "Picked automatically among the training rigs whose Sketchfab licence is CC BY or CC0: for each rig, its most energetic "
             "training captions were generated with two seeds and compared with the training clip; a demo had to move "
             f"{CLEAN['motion'][0]}-{CLEAN['motion'][1]}x as much as the clip, keep the position and rotation decodes within "
             f"{CLEAN['gap']} x the rig size, stay under {CLEAN['jitter']}x its jitter and within {CLEAN['float_bl']} bone lengths of "
             "its ground clearance, and at most four rigs per body plan were taken, best first. "
             + (f"Left out although they pass: {and_join(lo)}. " if lo else "")
             + (f"For body-plan variety {words[len(relaxed)]} demo{'s were' if len(relaxed) > 1 else ' was'} admitted just outside "
                f"those bounds ({'; '.join(relaxed)}). " if relaxed else "")
             + "Every demo was looked at before it was kept; `prompts.json` records its scores.", "",
             "```bash", "bash demo/run_demos.sh            # CKPT=weights/topox_h1_uniml3d73m_ep239_infer.pt, outputs in out/demos/", "```", "",
             "| demo | UniML3D body plan | joints | prompt | seed | frames |", "|---|---|---|---|---|---|"]
    for name, s, lic, prompts, J in table:
        for pr in prompts:
            lines.append(f"| `{name}` | {s['plan']} | {J} | {pr['text']} | {pr['seed']} | {pr['frames']} |")
    lines += ["", "## Attribution and licences", "",
              "The skeletons come from 3D models published on Sketchfab and indexed by Objaverse-XL (Allen Institute for AI), taken "
              "from the UniML3D export of Objaverse-XL. Each model keeps its creator's licence, shown below as listed on Sketchfab on "
              f"{max(lic['retrieved'] for _, _, lic, _, _ in table)}; nothing here grants rights beyond it.", "",
              "The prompts are UniML3D captions, the joint descriptions in `skeleton.npz` are built from UniML3D's cleaned joint "
              "labels, and the body plans are UniML3D's categories. UniML3D (https://huggingface.co/datasets/Linzhan/UniML3D) offers "
              "these annotations under [ODC-BY 1.0](https://opendatacommons.org/licenses/by/1-0/); please cite", "",
              "```bibtex", *UNIML3D_BIB, "```", "",
              "Changes: from each model only the skeleton (joint hierarchy, rest pose and joint names) was extracted and converted to "
              "the KTJD-17 format; no mesh, texture or animation of the original model is included. Added to each skeleton: the "
              "UniML3D caption of each prompt, joint descriptions built from UniML3D's joint labels, and LLM2Vec embeddings of both "
              "(`text_emb_<k>.npy`, `joint_sem.npy`). If you are a rights holder and want a skeleton removed, open an issue on this "
              "repository.", "",
              "| demo | model | author | licence | source |", "|---|---|---|---|---|"]
    for name, s, lic, prompts, J in table:
        label, url = cc_label(lic)
        lines.append(f"| `{name}` | {lic.get('name')} | {lic.get('author')} | [{label}]({url}) | {lic.get('url')} |")
    (DEMO / "README.md").write_text("\n".join(lines) + "\n")
    sh = ["#!/bin/bash", "# Regenerate every demo of demo/README.md with the stored embeddings (no text encoder needed).",
          "set -euo pipefail", 'cd "$(dirname "$0")/.."', 'CKPT=${CKPT:-weights/topox_h1_uniml3d73m_ep239_infer.pt}',
          'OUT=${OUT:-out/demos}']
    for name, k, pk in runs:
        sh.append(f'python scripts/deploy_generate.py --ckpt "$CKPT" --skeleton demo/{name}/skeleton.npz --keep_rest_rotations '
                  f'--text_emb demo/{name}/text_emb_{k}.npy --joint_sem demo/{name}/joint_sem.npy --frames {pk["T"]} '
                  f'--seed {pk["seed"]} --out "$OUT/{name}" --name ref_{k} --force')
    (DEMO / "run_demos.sh").write_text("\n".join(sh) + "\n")
    print(f"[package] {len(table)} rigs, {len(runs)} demos -> {DEMO}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["score", "licenses", "score2", "select", "relicense", "package"])
    ap.add_argument("--max_bipedal", type=int, default=4)
    a = ap.parse_args()
    {"licenses": licenses, "select": select, "relicense": relicense, "package": package}.get(a.stage, score)(a)
