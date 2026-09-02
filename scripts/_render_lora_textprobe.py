#!/usr/bin/env python3
"""Text-generalisation probe for a per-species LoRA checkpoint (user 2026-09-02).

Question: does the Buffalo LoRA follow captions it never saw -- long PZ captions of RELATED species
(African / water buffalo, bison, highland cattle, wildebeest) -- on the TrueBones Buffalo rig?

For each requested PZ clip:
  * the TrueBones target item of --rig is built exactly as the renderer does (1-frame rest demo,
    joint semantics, structural features, channel validity) -- only two things change:
      - the text condition is REPLACED by the PZ clip's caption embedding (same LLM2Vec encoder,
        read from the PZ caption cache, key "<clip>__cap<k>")
      - the target window is RESIZED to min(PZ clip length, target_frames); frames beyond it are
        zeroed and marked invalid (no TrueBones GT content is left in the window)
  * panels: DEMO (rest) | GEN pos | GEN fk (TrueBones rig)  ||  PZ REFERENCE: the PZ source clip's
    own GT on ITS rig (different skeleton, drawn with its own parents). There is no GT for the
    generated motion; the reference shows what the caption looks like on a PZ animal.
  * summary.txt reports per clip: generated motion energy (mean frame-to-frame world displacement
    of all joints, canonical units, also divided by the rig's s_rig) vs the PZ reference's energy,
    root path length, and the TB rig's own val-clip energy for scale. Cross-rig energy is a
    DIAGNOSTIC of "how much it moves", not an action-fidelity metric (different s_rig / J / topology).

Lineage (codex 2026-09-02 P0): the LoRA was initialised from a backbone trained on the PZ corpus, so
a PZ caption + motion may be a SEEN training example of the final model even though the LoRA stage
never saw it. The probe walks the init_from chain, collects every ancestor's TRAIN clip ids and
caption strings, and refuses a probe whose clip or whose exact caption text was seen -- unless
--allow_seen, in which case the GIF and summary carry SEEN-BY-LINEAGE(ancestor|own) (post-LoRA
PZ->TrueBones transfer/retention, NOT unseen-text generalisation). The seen set is the union of every
verified ancestor's train set and this checkpoint's own LoRA-stage train set; each hop is verified
(parent file sha == child's lora_cfg.init_from_sha256, every view's ktjd_pins complete and matching).
`clip` alone auto-selects the first caption variant of that clip nobody in the lineage trained on;
`clip:k` forces variant k. Captions are enumerated with the canonical caption_keys.ordered_captions.

Data arguments are back-filled from the checkpoint (never a render-time default), as in
scripts/v2_render_incontext.py. Run inside an allocation on a GPU.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import hashlib
import numpy as np
import torch
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.v2_render_incontext import (COLS, FOOT, GAP, PANEL_H, PANEL_W,        # noqa: E402
                                          draw_panel, project, world_of_ktjd)
from src.data.caption_keys import ordered_captions                              # noqa: E402
from src.data.incontext_pairs import InContextPairs, collate                    # noqa: E402
from src.data.ktjd17_incontext import Ktjd17Base, ktjd17_split_names            # noqa: E402
from src.models.v2.dit_motion import InContextMotionDiT, sample                 # noqa: E402


def render_gif_multi(out_path, panels, caption, tag, fps=30):
    """panels: list of (color_key, label, seq [T,J,3], parents) -- skeletons MAY differ per panel."""
    seqs, ground_px = project([w for _, _, w, _ in panels])
    T = max(s.shape[0] for s in seqs)
    W = PANEL_W * len(panels) + GAP * (len(panels) - 1)
    frames = []
    for t in range(T):
        img = Image.new("RGB", (W, PANEL_H + FOOT), (246, 248, 247))
        d = ImageDraw.Draw(img)
        for k, ((name, label, _, parents), seq) in enumerate(zip(panels, seqs)):
            x0 = k * (PANEL_W + GAP)
            gy = min(max(ground_px, 0), PANEL_H - 1)
            d.rectangle([x0, gy, x0 + PANEL_W - 1, PANEL_H - 1], fill=(236, 240, 238))
            d.line([x0, gy, x0 + PANEL_W - 1, gy], fill=(170, 186, 178), width=2)
            d.rectangle([x0, 0, x0 + PANEL_W - 1, PANEL_H - 1], outline=(216, 223, 225))
            tt = min(t, seq.shape[0] - 1)
            draw_panel(img, seq[tt], parents, COLS[name], x0)
            d.text((x0 + 8, 6), label, fill=COLS[name])
            d.text((x0 + 8, PANEL_H - 18), f"f{tt + 1}/{seq.shape[0]}", fill=(116, 133, 150))
        d.text((8, PANEL_H + 8), tag, fill=(15, 23, 32))
        for li, line in enumerate([caption[i:i + 96] for i in range(0, min(len(caption), 192), 96)]):
            d.text((8, PANEL_H + 26 + 16 * li), line, fill=(64, 80, 94))
        frames.append(img)
    frames[0].save(out_path, save_all=True, append_images=frames[1:], duration=round(1000.0 / fps), loop=0)


def energy(w):
    """mean frame-to-frame displacement over all joints (canonical units); 0 for < 2 frames."""
    return float(np.linalg.norm(np.diff(w, axis=0), axis=-1).mean()) if w.shape[0] >= 2 else 0.0


def norm_text(c: str) -> str:
    return " ".join(str(c).lower().split())


VIEW_FIELDS = ("ktjd_root", "caption_cache", "joint_sem", "texts_json", "ktjd_percell_stats", "exclude_clips")


def view_identity(args: dict) -> tuple:
    """The FULL identity of a training view: root + every sidecar + the effective cut, as resolved paths.
    Two lineage nodes on the same root but different captions / semantics / stats / cut are different views
    (codex 2026-09-02 r3 P0-2)."""
    return tuple(str(Path(str(args[f])).resolve()) if args.get(f) else "" for f in VIEW_FIELDS)


def view_base(args: dict, bases: dict):
    """Ktjd17Base for a checkpoint's OWN training view (its args, its cut); cached by full view identity."""
    ident = view_identity(args)
    if ident not in bases:
        bases[ident] = (Ktjd17Base(str(args["ktjd_root"]), caption_emb_cache=args["caption_cache"],
                                   joint_semantics=args["joint_sem"], texts_json=args["texts_json"],
                                   percell_stats=args["ktjd_percell_stats"],
                                   exclude_clips=args.get("exclude_clips") or None), dict(args))
    return bases[ident][0]


def pins_drift(ck: dict, base) -> list[str]:
    """Fail-closed data-pin check: the checkpoint must carry EVERY data pin the live view exposes (a missing
    pin is drift, codex 2026-09-02 r3 P0-1), and each carried value must match the live view."""
    pins = ck.get("ktjd_pins") or {}
    if not pins:
        raise SystemExit("[refuse] checkpoint carries no ktjd_pins -- its training data cannot be verified")
    live = {**base.provenance, "exclusion": base.provenance_exclusion}
    data_keys = set(live) | {"manifest_sha256", "derivation_sha256"}
    missing = sorted(f"missing:{k}" for k in live if k not in pins)
    differing = sorted(k for k, v in pins.items() if k in data_keys and (k not in live or live[k] != v))
    return missing + differing


def train_text_of(base, args: dict) -> tuple[set, set, int]:
    """(train clip ids, canonical caption strings of those clips, n_train) of a checkpoint's view."""
    names = ktjd17_split_names(str(args["ktjd_root"]), exclude=args.get("exclude_clips") or None)
    texts = json.loads(Path(args["texts_json"]).read_text())
    ids, caps = set(), set()
    for cid in names.get("train", set()):
        ids.add(cid)
        for c in ordered_captions(texts.get(cid + ".npy") or {}):
            caps.add(norm_text(c))
    return ids, caps, len(names.get("train", set()))


def ancestor_seen_sets(ck: dict, bases: dict, max_hops: int = 16):
    """Walk the init_from chain fail-closed (codex 2026-09-02 r2): every hop's parent file must hash to the
    child's lora_cfg.init_from_sha256, every ancestor's ktjd_pins must match its live view, cycles and an
    over-long chain refuse. Returns (seen train clip ids, seen canonical caption strings, chain description)."""
    ids, caps, chain = set(), set(), []
    child, visited = ck, set()
    while child["args"].get("init_from"):
        p = Path(str(child["args"]["init_from"])).resolve()
        if p in visited:
            raise SystemExit(f"[refuse] init_from lineage cycles at {p}")
        visited.add(p)
        if len(visited) > max_hops:
            raise SystemExit(f"[refuse] init_from lineage longer than {max_hops} hops -- refusing to trust a partial walk")
        if not p.is_file():
            raise SystemExit(f"[refuse] ancestor checkpoint {p} is missing; the lineage cannot be checked")
        want = str(((child.get("lora_cfg") or {}).get("init_from_sha256")) or "")
        if not want:
            raise SystemExit(f"[refuse] {child['args'].get('out', 'child ckpt')} names init_from but records no "
                             f"lora_cfg.init_from_sha256 -- the parent it actually used cannot be verified")
        have = hashlib.sha256(p.read_bytes()).hexdigest()
        if have != want:
            raise SystemExit(f"[refuse] ancestor {p} hashes {have[:16]}, child recorded init_from_sha256 {want[:16]} -- "
                             f"not the parent this checkpoint was initialised from")
        anc = torch.load(p, map_location="cpu", weights_only=False, mmap=True)
        a_args = anc["args"]
        if str(a_args.get("corpus")) != "ktjd17":
            raise SystemExit(f"[refuse] ancestor {p} corpus {a_args.get('corpus')!r} unsupported by this probe")
        base = view_base(a_args, bases)
        drift = pins_drift(anc, base)
        if drift:
            raise SystemExit(f"[refuse] ancestor {p}: its training view on disk no longer matches its data pins {drift}")
        a_ids, a_caps, n_tr = train_text_of(base, a_args)
        ids |= a_ids; caps |= a_caps
        chain.append(f"{p.name} (sha {have[:12]}) <- {a_args['ktjd_root']} (train {n_tr}, cut {a_args.get('exclude_clips')})")
        child = anc
    return ids, caps, chain


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--rig", default="Buffalo", help="TrueBones rig whose skeleton receives the PZ captions")
    ap.add_argument("--pz_clips", required=True,
                    help="comma-separated PZ clip ids, optionally clip:k for caption variant k (default: the first "
                         "variant no ancestor trained on, else variant 0)")
    ap.add_argument("--allow_seen", action="store_true",
                    help="render probes whose clip / exact caption text the lineage trained on (labelled SEEN-BY-LINEAGE / TEXT-SEEN)")
    ap.add_argument("--pz_root", default="dataset/ktjd17_pzh312_noik_v2")
    ap.add_argument("--pz_percell", default="data/noik_norm_stats_v2.npz")
    ap.add_argument("--pz_caption_cache", default="data/noik_caption_llm2vec_v1")
    ap.add_argument("--pz_joint_sem", default="data/joint_semantics_llm2vec_pzh312_v1.npz")
    ap.add_argument("--pz_texts", default="data/noik_pzh312_motion_texts_v1.json")
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--cfg_text", type=float, default=2.0)
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    ca = ck["args"]
    if str(ca.get("corpus")) != "ktjd17":
        raise SystemExit(f"[refuse] this probe is for KTJD-17 checkpoints; ckpt corpus is {ca.get('corpus')!r}")
    if bool(ca.get("two_stage", False)) or str(ca.get("anchor", "none")) != "none":
        raise SystemExit("[refuse] two_stage / anchored checkpoints are not supported by this probe")
    model = InContextMotionDiT(in_ch=17, dim=ca["dim"], depth=ca["depth"], n_heads=ca["heads"],
                               d_text=4096, d_joint_sem=4096,
                               use_struct_feats=bool(ca.get("struct_feats", False)),
                               use_dir_bias=bool(ca.get("dir_bias", False)),
                               qk_norm=bool(ca.get("qk_norm", False)),
                               use_ref_text=bool(ca.get("ref_text", False))).to(dev)
    model.load_state_dict(ck["model"]); model.eval()
    ep = ck.get("epoch", -1)
    if bool(ca.get("ref_text", False)):
        raise SystemExit("[refuse] ref_text checkpoints carry a demo caption; this probe does not swap it")
    if not bool(ca.get("demo_rest", False)) or int(ca.get("demo_frames", 0)) != 1:
        raise SystemExit("[refuse] probe expects a demo_rest / 1-frame-demo checkpoint")
    Td, Tt = 1, int(ca["target_frames"])
    print(f"[probe] ckpt {a.ckpt} (epoch {ep}) rig {a.rig} target window {Tt}", flush=True)

    # ---- the checkpoint's own training view (back-filled, never a default) ----
    bases: dict = {}
    base = view_base(ca, bases)
    drift = pins_drift(ck, base)
    if drift:
        raise SystemExit(f"[refuse] the checkpoint's data pins do not match its view on disk: {drift}")
    names = ktjd17_split_names(ca["ktjd_root"], exclude=ca.get("exclude_clips") or None)
    PK = dict(demo_rest=True, emit_ref_text=False, demo_frames=Td, target_frames=Tt,
              emit_graph_v2=bool(ca.get("struct_feats", False)) or bool(ca.get("dir_bias", False)))
    ds = InContextPairs(base, names["val"], names["train"], object_types=[a.rig], balance_skeletons=False,
                        seed=a.seed, **PK)
    positions = [i for i, (ot, _) in enumerate(ds.index) if ot == a.rig]
    if not positions:
        raise SystemExit(f"[refuse] rig {a.rig} has no val target in the checkpoint's view")
    ds._wrng_key = None
    item = ds[positions[0]]                                   # carrier: skeleton, rest demo, semantics
    J = int(item["n_joints"])
    if "text" not in item:
        raise SystemExit("[refuse] carrier item has no caption embedding")
    tb_val_energy = energy(world_of_ktjd(item["x"][Td:Td + int(item["frame_valid"][Td:].sum()), :J].numpy(),
                                         base, a.rig, strict_gt=True)[0])
    tb_srig = float(base._skeleton(a.rig)["s_rig"])
    # seen = every ancestor's train set (verified lineage) PLUS this checkpoint's own LoRA-stage train set;
    # the two sources are kept apart so a refusal / label can say WHICH stage saw it (codex r3 P1)
    anc_ids, anc_caps, chain = ancestor_seen_sets(ck, bases)
    own_ids, own_caps, own_n = train_text_of(base, ca)
    seen_ids, seen_caps = anc_ids | own_ids, anc_caps | own_caps
    chain.append(f"{Path(a.ckpt).name} <- {ca['ktjd_root']} (train {own_n}, cut {ca.get('exclude_clips')}) [this checkpoint]")
    print(f"[probe] lineage: {' ; '.join(chain)} -> ancestors {len(anc_ids)} clips / {len(anc_caps)} captions, "
          f"own stage {len(own_ids)} / {len(own_caps)}, union {len(seen_ids)} / {len(seen_caps)}", flush=True)

    # ---- PZ side: caption embeddings + reference GT on the PZ rig ----
    # If the requested PZ root is one of the VERIFIED lineage views, the reference, the caption texts and the
    # caption embeddings all come from THAT view's pinned sidecars (its cut applies); CLI pz_* sidecars must
    # name the same files or the run is refused (codex r3 P0-3). Several verified views on one root = ambiguous.
    pz_root_res = str(Path(a.pz_root).resolve())
    matches = [ident for ident in bases if ident[0] == pz_root_res]
    if len(matches) > 1:
        raise SystemExit(f"[refuse] {len(matches)} verified lineage views share root {a.pz_root} with different sidecars/cuts; "
                         f"the reference view is ambiguous")
    if matches:
        pz, pz_args = bases[matches[0]]
        cli = {"pz_caption_cache": a.pz_caption_cache, "pz_texts": a.pz_texts, "pz_joint_sem": a.pz_joint_sem,
               "pz_percell": a.pz_percell}
        anc = {"pz_caption_cache": pz_args["caption_cache"], "pz_texts": pz_args["texts_json"],
               "pz_joint_sem": pz_args["joint_sem"], "pz_percell": pz_args["ktjd_percell_stats"]}
        bad = [k for k in cli if str(Path(cli[k]).resolve()) != str(Path(anc[k]).resolve())]
        if bad:
            raise SystemExit(f"[refuse] --{'/--'.join(bad)} differ from the verified lineage view's sidecars "
                             f"({', '.join(f'{k}={anc[k]}' for k in bad)}); the caption provenance would be unverified")
        cap_cache, texts_path = pz_args["caption_cache"], pz_args["texts_json"]
        pz_note = (f"reference view = verified lineage view of {a.pz_root} (cut {pz_args.get('exclude_clips')}); captions/"
                   f"embeddings from its pinned sidecars {cap_cache} / {texts_path}")
    else:
        pz = Ktjd17Base(a.pz_root, caption_emb_cache=a.pz_caption_cache, joint_semantics=a.pz_joint_sem,
                        texts_json=a.pz_texts, percell_stats=a.pz_percell, exclude_clips=None)
        cap_cache, texts_path = a.pz_caption_cache, a.pz_texts
        pz_note = f"reference view = {a.pz_root} built from pz_* args (no cut; NOT a lineage view, provenance unverified)"
    print(f"[probe] {pz_note}", flush=True)
    E = np.load(Path(cap_cache).with_suffix(".embs.npy"), mmap_mode="r")
    keys = json.load(open(Path(cap_cache).with_suffix(".keys.json")))
    row_of = {k: i for i, k in enumerate(keys)}
    texts = json.load(open(texts_path))
    idx_of = {str(r["clip_id"]): i for i, r in enumerate(pz._rows)}

    lines = []
    for spec in [c.strip() for c in a.pz_clips.split(",") if c.strip()]:
        cid, _, kstr = spec.partition(":")
        if cid not in idx_of:
            raise SystemExit(f"[refuse] PZ clip {cid}: not in the reference view ({pz_note}) -- missing or cut")
        caps = ordered_captions(texts.get(cid + ".npy") or {})   # canonical <clip>__cap<k> order (caption_keys.py)
        if not caps:
            raise SystemExit(f"[refuse] PZ clip {cid} has no captions")
        if kstr:
            k = int(kstr)
        else:                                                  # first variant nobody in the lineage trained on
            unseen = [i for i, c in enumerate(caps) if norm_text(c) not in seen_caps]
            k = unseen[0] if unseen else 0
        if not 0 <= k < len(caps):
            raise SystemExit(f"[refuse] PZ clip {cid}: caption variant {k} out of range ({len(caps)})")
        key = f"{cid}__cap{k}"
        if key not in row_of:
            raise SystemExit(f"[refuse] PZ clip {cid}: caption key {key!r} not in {cap_cache}")
        cap = caps[k]
        clip_by = [s for s, ids in (("ancestor", anc_ids), ("own", own_ids)) if cid in ids]
        text_by = [s for s, cs in (("ancestor", anc_caps), ("own", own_caps)) if norm_text(cap) in cs]
        clip_seen, text_seen = bool(clip_by), bool(text_by)
        if (clip_seen or text_seen) and not a.allow_seen:
            raise SystemExit(f"[refuse] PZ clip {cid} cap{k}: clip seen by {clip_by or 'nobody'}, caption text seen by "
                             f"{text_by or 'nobody'} in this checkpoint's lineage -- not an unseen-text probe (pass "
                             f"--allow_seen to render it as a retention/transfer example)")
        flag = (f"SEEN-BY-LINEAGE({'+'.join(clip_by)})" if clip_seen
                else (f"TEXT-SEEN({'+'.join(text_by)})" if text_seen else "UNSEEN-TEXT"))
        ref = pz[idx_of[cid]]
        pz_rig = str(pz._rows[idx_of[cid]]["rig_id"])
        Jp, Tp = int(ref["num_joints"]), int(ref["num_frames"])
        pz_srig = float(pz._skeleton(pz_rig)["s_rig"])
        T_new = min(Tp, Tt)
        emb = torch.as_tensor(np.asarray(E[row_of[key]])).float()

        # rebuild the carrier window: same demo, PZ caption, PZ length; nothing of the TB target survives
        x = item["x"].clone(); x[Td:] = 0.0
        is_t = torch.zeros_like(item["is_target"]); is_t[Td:Td + T_new] = True
        fv = torch.zeros_like(item["frame_valid"]); fv[:Td] = True; fv[Td:Td + T_new] = True
        it2 = dict(item); it2.update(x=x, is_target=is_t, frame_valid=fv, text=emb)
        b = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in collate([it2]).items()}

        torch.manual_seed(a.seed)
        with torch.no_grad():
            g2kw = {k: b[k] for k in ("struct_feats", "updown") if k in b}
            cvj = torch.from_numpy(base.static_masks(a.rig)["channel_valid"]).to(dev)
            cv = torch.zeros(1, b["x"].shape[2], 17, dtype=torch.bool, device=dev); cv[0, :cvj.shape[0]] = cvj
            g2kw["channel_valid"] = cv
            g2kw["heading_valid"] = b["x"][:, :, 0, 17] > 0.5
            gen = sample(model, b["x"][..., :17].contiguous(), b["is_target"], a.steps, cfg_text=a.cfg_text,
                         demo_frames=Td, joint_bias=b["joint_bias"], frame_valid=b["frame_valid"],
                         joint_valid=b["joint_valid"], text=b["text"], joint_sem=b["joint_sem"], **g2kw)
        gen = gen[0].float().cpu().numpy()
        gseg = gen[Td:Td + T_new, :J]
        demo_w, _ = world_of_ktjd(item["x"][:Td, :J].numpy(), base, a.rig, strict_gt=True)
        gen_ric, gen_fk = world_of_ktjd(gseg, base, a.rig, strict_gt=False)
        xr = np.asarray(ref["anytop_x"])[:Jp, :, :Tp].transpose(2, 0, 1)
        ref_w, _ = world_of_ktjd(xr, pz, pz_rig, strict_gt=True)
        tb_par = [int(p) for p in item_parents(base, a.rig, J)]
        pz_par = [int(p) for p in np.asarray(ref["parent_indices"])[:Jp]]

        e_gen, e_ref = energy(gen_ric), energy(ref_w[:T_new])
        root_path = float(np.linalg.norm(np.diff(gen_ric[:, 0], axis=0), axis=-1).sum())
        fk_gap = float(np.linalg.norm(gen_ric - gen_fk, axis=-1).mean())
        name = f"{a.rig}__{pz_rig}__{cid[:12]}_cap{k}"
        render_gif_multi(out / f"{name}.gif",
                         [("demo", f"DEMO rest {a.rig}", demo_w, tb_par),
                          ("gen_ric", f"GEN pos ep{ep} s{a.steps} [{flag}]", gen_ric, tb_par),
                          ("gen_fk", f"GEN fk ep{ep} s{a.steps}", gen_fk, tb_par),
                          ("gt", f"PZ SOURCE REF {pz_rig[3:][:16]} - NOT GT FOR GEN", ref_w[:T_new], pz_par)],
                         cap, f"[textprobe {flag}] {a.rig} (J={J}) <- caption {key} of {pz_rig} (J={Jp}), {T_new}f")
        print(f"[probe] {name}.gif  {flag}  T={T_new}  energy gen {e_gen:.4f} ({e_gen / tb_srig:.5f}/s_rig) / PZ ref "
              f"{e_ref:.4f} ({e_ref / pz_srig:.5f}/s_rig) / TB val-clip {tb_val_energy:.4f}  root path {root_path:.2f}  "
              f"pos-vs-fk gap {fk_gap:.3f}  | {cap}", flush=True)
        lines.append(f"{name}\t{flag}\tclip_seen_by={'+'.join(clip_by) or 'none'}\ttext_seen_by={'+'.join(text_by) or 'none'}"
                     f"\tpz_rig={pz_rig}\tpz_clip={cid}\tcaption_key={key}"
                     f"\tT={T_new}\tJ_gen={J}\tJ_ref={Jp}\ts_rig_gen={tb_srig:.4f}\ts_rig_ref={pz_srig:.4f}"
                     f"\tenergy_gen={e_gen:.4f}\tenergy_gen_over_s_rig={e_gen / tb_srig:.5f}\tenergy_ref={e_ref:.4f}"
                     f"\tenergy_ref_over_s_rig={e_ref / pz_srig:.5f}\tenergy_tbval={tb_val_energy:.4f}"
                     f"\troot_path={root_path:.3f}\tpos_fk_gap={fk_gap:.4f}\tcaption: {cap}")
    (out / "summary.txt").write_text(
        f"ckpt={a.ckpt} epoch={ep} rig={a.rig} steps={a.steps} seed={a.seed} cfg_text={a.cfg_text} "
        f"panels=demo|gen_ric|gen_fk|PZ_SOURCE_REFERENCE(other skeleton, NOT GT for the generation)\n"
        f"caption_source={a.pz_caption_cache} pz_root={a.pz_root} pz_generation={pz.generation_id} "
        f"gen_view={ca['ktjd_root']} gen_generation={base.generation_id}\n"
        f"{pz_note}\n"
        f"lineage (each hop sha-verified against lora_cfg.init_from_sha256, each view against its ktjd_pins): "
        f"{' ; '.join(chain)} -> ancestors {len(anc_ids)} clips / {len(anc_caps)} captions, own stage {len(own_ids)} / "
        f"{len(own_caps)}, union {len(seen_ids)} / {len(seen_caps)}; "
        f"flags: UNSEEN-TEXT = neither the clip nor the exact caption text was trained on by any ancestor or by this "
        f"checkpoint's own LoRA stage; TEXT-SEEN(who) = clip unseen but identical caption string trained on by <who>; "
        f"SEEN-BY-LINEAGE(who) = clip in the train set of <who> (ancestor / own)\n"
        f"energy = mean frame-to-frame world displacement over all joints (canonical units); cross-rig energy is a "
        f"how-much-it-moves diagnostic, NOT an action-fidelity metric (different s_rig / joint count / topology)\n"
        + "\n".join(lines) + "\n")
    print(f"[probe] DONE -> {out}", flush=True)


def item_parents(base, rig, J):
    sk = base._skeleton(rig)
    par = np.asarray(sk["parents"], dtype=np.int64)
    if len(par) != J:
        raise RuntimeError(f"{rig}: skeleton has {len(par)} joints, item has {J}")
    return par


if __name__ == "__main__":
    main()
