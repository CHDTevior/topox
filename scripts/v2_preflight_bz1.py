#!/usr/bin/env python3
"""Launch preflight for the in-context motion DiT: verify every conditioning input against its
SOURCE OF TRUTH, walk one bz=1 item variable by variable, hard-check the data processing
(FK==RIC), and prove the model handles every rig including the largest.

Sections (each prints PASS/FAIL lines; exit 1 on any FAIL):
  [1] asset provenance      joint_sem npz / caption cache / frozen splits all identify themselves
  [2] bz=1 item walkthrough every tensor printed and cross-checked against raw files:
                              - served text vector == the exact row of the LLM2Vec cache for the
                                served caption STRING (bitwise-level max|diff|)
                              - served joint_sem rows == the npz table rows (order-hash already
                                enforced inside the dataset; this re-checks values end-to-end)
                              - demo/target crops are VERBATIM windows of the base clips
                              - same-name joints across DIFFERENT rigs have near-identical
                                embeddings (proof the table is name-driven, i.e. "之前 LLM 处理过
                                的 joint name", not per-rig noise)
  [3] FK ~= RIC             de-normalised clips: positions recovered from rot6d (FK, official
                            port) vs positions carried in ch0:3 (RIC path) -- INTERNAL consistency
                            of the two channel families (rel mean err < 0.5%), not proof against
                            the original BVH world truth.
  [4] all-rigs sweep        for EVERY rig, bz=1 forward+backward on its LONGEST clip as target;
                            J up to 142 must run, loss/grad finite, peak memory recorded.
  [5] hyperparameters       the exact run-1 config, printed for sign-off.

Read-only. Run on a compute node GPU.
"""
import hashlib, itertools, json, pickle, sys, time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.anytop_dataset import (AnyTopDataset, _STD_FLOOR,                 # noqa: E402
                                     _recover_world_positions)
from src.data.anytop_rot6d_fk import recover_from_bvh_rot_np                    # noqa: E402
from src.data.incontext_pairs import (InContextPairs, collate, read_split,      # noqa: E402
                                      truebones_types, pzh_types, DEMO_FRAMES, TARGET_FRAMES)
from src.models.v2.dit_motion import (InContextMotionDiT, cfm_loss,             # noqa: E402
                                      KIMODO_GAMMAS)

R = "data/animo4d_L4TB_plus_human_v4b272neutral"
SPLITS = "data/holdout_splits_v1"
SEM = "data/joint_semantics_llm2vec_v1.npz"
CAP = "data/anytop_caption_llm2vec_v4b272neutral_multi"
TEXTS = "motion_texts_by_file_clean_v1.json"

FAILS = []
def gate(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  {detail}" if detail else ""))
    if not ok:
        FAILS.append(name)

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", choices=("truebones", "pzh"), default="truebones")
    ap.add_argument("--dim", type=int, default=256)
    ap.add_argument("--depth", type=int, default=6)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--demo_frames", type=int, default=DEMO_FRAMES)
    ap.add_argument("--target_frames", type=int, default=TARGET_FRAMES)
    ap.add_argument("--bf16", action="store_true", help="run the sweep/B8 under bf16 autocast, matching training")
    ap.add_argument("--mem_gate_gib", type=float, default=60.0)
    ap.add_argument("--worst_batch", type=int, default=8,
                    help="copies of the largest-J training clip in the worst-case memory gate")
    ap.add_argument("--random_caption", action="store_true")
    a = ap.parse_args()
    # This is a GPU LAUNCH gate: silently falling back to CPU would "verify" a configuration
    # that is not the one we launch. Refuse instead.
    assert torch.cuda.is_available(), "preflight must run on the launch GPU, not CPU"
    dev = "cuda"
    torch.manual_seed(0)

    # ================= [1] asset provenance =================
    print("=== [1] 资产与出处 ===")
    meta = json.load(open(f"{CAP}.meta.json"))
    print(f"  caption cache: encoder={meta['encoder']}")
    print(f"                 dim={meta['dim']}  rows={meta['n_rows']}  built from {meta['captions_json_name']}")
    gate("caption cache is LLM2Vec 4096-d", meta["dim"] == 4096 and "llm2vec" in meta["encoder"].lower())
    gate("cache was built from the json we will serve", meta["captions_json_name"] == TEXTS)
    sem_npz = np.load(SEM, allow_pickle=False)
    n_rigs_sem = sum(1 for k in sem_npz.files if k.startswith("emb__"))
    has_hash = "__order_hash" in sem_npz.files
    print(f"  joint_sem npz: {n_rigs_sem} rigs, dim={int(sem_npz['__dim'])}, order-hash table present={has_hash}")
    gate("joint_sem table is 4096-d with per-rig order hashes", int(sem_npz["__dim"]) == 4096 and has_hash)
    man = json.load(open(f"{SPLITS}/splits_manifest.json"))
    print(f"  frozen splits: artifact sha {man['artifact_sha256'][:16]}…")
    # artifact_sha256 is the artifact's SELF-hash: sha256 over the canonical dump of the json
    # MINUS its own artifact_sha256 field (_build_holdout_splits.py:58-63). Hashing the raw file
    # bytes instead compares apples to oranges and false-FAILs. Replicate the builder's formula,
    # and additionally pin the raw bytes against the SEAL record.
    art = json.loads(Path("data/holdout_topologies_v1.json").read_text())
    body = json.dumps({k: v for k, v in art.items() if k != "artifact_sha256"},
                      indent=2, sort_keys=True)
    self_hash = hashlib.sha256(body.encode()).hexdigest()
    gate("holdout artifact 自哈希一致(未被篡改)",
         self_hash == art.get("artifact_sha256") == man["artifact_sha256"])
    seal = json.load(open("protocol/SEAL.json"))
    seal_rec = seal.get("inline", {}).get("data/holdout_topologies_v1.json", {})
    raw_sha = hashlib.sha256(Path("data/holdout_topologies_v1.json").read_bytes()).hexdigest()
    gate("holdout artifact 原始字节与 SEAL 记录一致", raw_sha == seal_rec.get("sha256"))

    # ================= datasets =================
    cond = pickle.load(open(f"{R}/_cond_normalized_J144.pkl", "rb"))
    tb = truebones_types(cond.keys()) if a.corpus == "truebones" else pzh_types(cond.keys())
    print(f"  corpus={a.corpus}: {len(tb)} rigs | model dim{a.dim}x{a.depth}h{a.heads} | "
          f"slots {a.demo_frames}+{a.target_frames} | bf16={a.bf16}")
    names = {k: read_split(SPLITS, k) for k in ("train", "val", "held_representative", "held_stress")}
    gate("四个 split 名单两两不相交",
         all(not (names[u] & names[v]) for u, v in itertools.combinations(names, 2)))
    base = AnyTopDataset(data_root=R, split="all", num_frames=300, max_joints=144,
                         load_captions=True, caption_emb_cache=CAP, random_caption=a.random_caption,
                         augment=False, joint_semantics=SEM, species_whitelist=tb,
                         splits_dir=SPLITS, texts_json_name=TEXTS)
    PK = dict(demo_frames=a.demo_frames, target_frames=a.target_frames)
    dss = {k: InContextPairs(base, names[k] if k != "A" else names["val"],
                             names["train"] if k in ("train", "A") else names[k],
                             object_types=tb, balance_skeletons=False, seed=0, **PK)
           for k in ("train",)}
    dss["A"] = InContextPairs(base, names["val"], names["train"], object_types=tb,
                              balance_skeletons=False, seed=0, **PK)
    dss["B"] = InContextPairs(base, names["held_representative"], names["held_representative"],
                              object_types=tb, balance_skeletons=False, seed=0)
    dss["stress"] = InContextPairs(base, names["held_stress"], names["held_stress"],
                                   object_types=tb, balance_skeletons=False, seed=0)

    # ================= [2] bz=1 walkthrough =================
    ds = dss["train"]
    want_rig = "Alligator" if a.corpus == "truebones" else "PZ_Aardvark_Female"
    ex_rig = want_rig if want_rig in ds.types else ds.types[0]
    print(f"\n=== [2] bz=1 单样本逐变量(训练桶,{ex_rig}) ===")
    pos = next(i for i, (ot, _) in enumerate(ds.index) if ot == ex_rig)
    item = ds[pos]
    b = collate([item])
    ot = item["object_type"]
    t_idx = dict(ds.index)  # not unique per ot; recover the actual indices from the item ids
    tgt_idx = ds.index[pos][1]
    demo_idx = int(item["demo_id"])
    t_item = base[tgt_idx]; d_item = base[demo_idx]
    t_stem = str(t_item["motion_id"]); d_stem = str(d_item["motion_id"])
    print(f"  target = {t_stem}   demo = {d_stem}")
    gate("demo 与 target 同骨架且不同 clip",
         d_item["object_type"] == ot and t_stem != d_stem)

    x = b["x"][0]; J = int(item["n_joints"])
    print(f"  x {tuple(b['x'].shape)}  (demo {DEMO_FRAMES} + target {TARGET_FRAMES}, J={J})")
    d_real = int(item["frame_valid"][:DEMO_FRAMES].sum())
    t_real = int(item["frame_valid"][DEMO_FRAMES:].sum())
    dm = x[:d_real]; tg = x[DEMO_FRAMES:DEMO_FRAMES + t_real]
    print(f"  demo 段  真实 {d_real}/{DEMO_FRAMES} 帧  mean={dm.mean():+.4f} std={dm.std():.4f}")
    print(f"  target 段 真实 {t_real}/{TARGET_FRAMES} 帧  mean={tg.mean():+.4f} std={tg.std():.4f}")
    pad_frames = x[DEMO_FRAMES + t_real:]
    gate("padding 帧全零", pad_frames.numel() == 0 or float(pad_frames.abs().max()) == 0.0)
    gate("is_target: demo 全 False, 真实 target 全 True",
         not bool(b["is_target"][0, :DEMO_FRAMES].any())
         and int(b["is_target"][0].sum()) == t_real)
    gate("joint_valid 数 == J", int(b["joint_valid"][0].sum()) == J)
    bias = b["joint_bias"][0]
    real_bias = bias[:J, :J]
    print(f"  joint_bias 真实块 [{float(real_bias.min()):.1f}, {float(real_bias.max()):.1f}]  对角={float(real_bias.diagonal().abs().max()):.1f}")
    gate("bias: 对角 0 / 截断 -8 / padding 行与列均 -1e4",
         float(real_bias.diagonal().abs().max()) == 0.0
         and float(real_bias.min()) >= -8.0
         and (J == bias.shape[0] or (float(bias[J:, :].max()) <= -1e3
                                     and float(bias[:, J:].max()) <= -1e3)))
    fv = b["frame_valid"][0]; jv = b["joint_valid"][0]
    gate("valid == frame_valid ∧ joint_valid(逐元素)",
         bool(torch.equal(b["valid"][0], fv[:, None] & jv[None, :])))
    gate("非 valid 帧的 x 全零(demo 半与 target 半都算)",
         (not bool((~fv).any())) or float(b["x"][0][~fv].abs().max()) == 0.0)
    gate("padding 关节列的 x 全零",
         J == b["x"].shape[2] or float(b["x"][0][:, J:].abs().max()) == 0.0)
    ar = torch.arange(b["x"].shape[1])
    gate("is_target == frame_valid ∧ (t >= 64)(逐元素)",
         bool(torch.equal(b["is_target"][0], fv & (ar >= DEMO_FRAMES))))

    # ---- crop 是 base clip 的逐字窗口 ----
    def norm_motion(it):
        Jx, Tx = int(it["num_joints"]), int(it["num_frames"])
        return np.asarray(it["anytop_x"])[:Jx, :, :Tx].transpose(2, 0, 1).astype(np.float32)
    def find_window(full, crop, n_real):
        c = crop[:n_real]
        for s in range(full.shape[0] - n_real + 1):
            if np.array_equal(full[s:s + n_real], c):
                return s
        return -1
    dn, tn = norm_motion(d_item), norm_motion(t_item)
    s_d = find_window(dn, x[:DEMO_FRAMES].numpy(), d_real)
    s_t = find_window(tn, x[DEMO_FRAMES:].numpy(), t_real)
    gate(f"demo crop 是 base 的逐字窗口 (offset={s_d})", s_d >= 0)
    gate(f"target crop 是 base 的头部窗口 (offset={s_t})", s_t == 0)

    # ---- 文本:served 向量 == 该 caption 字符串在缓存里的那一行 ----
    cap_str = str(t_item.get("caption", ""))
    print(f"  caption(target): \"{cap_str[:90]}\"")
    caps_json = json.load(open(f"{R}/{TEXTS}"))
    entry = caps_json.get(t_stem) or caps_json.get(t_stem + ".npy")
    # entry is a DICT: caption variants live in entry["captions"] (list[str]); the dataset serves
    # entry["primary_caption"], which is captions[0] and maps to cache key <stem>__cap0.
    if isinstance(entry, dict):
        texts = [str(c) for c in entry.get("captions", [])]
    else:
        texts = [str(c) for c in (entry or [])]
    gate("caption 字符串存在于 captions json 的该条目里", cap_str in texts,
         f"(条目含 {len(texts)} 条)")
    if cap_str in texts:
        keys = json.load(open(f"{CAP}.keys.json"))
        key2row = {kk: i for i, kk in enumerate(keys)}
        embs = np.load(f"{CAP}.embs.npy", mmap_mode="r")
        served = np.asarray(b["text"][0], dtype=np.float32)
        # duplicate caption strings make a single index() ambiguous -- check EVERY row whose
        # string equals the served caption, so a positional mismatch cannot hide behind a twin.
        rows = [key2row[f"{t_stem}__cap{i}"] for i, t in enumerate(texts) if t == cap_str]
        diffs = [float(np.abs(embs[r].astype(np.float32) - served).max()) for r in rows]
        gate(f"served text 向量 == 该字符串对应的全部 {len(rows)} 行(重复也一致)",
             max(diffs) < 1e-5, f"max|diff|={max(diffs):.2e}")

    # ---- joint_sem:served 行 == npz 表行;且同名关节跨骨架一致 ----
    served_sem = np.asarray(b["joint_sem"][0], dtype=np.float32)
    tab = sem_npz[f"emb__{ot}"].astype(np.float32)
    diff = float(np.abs(served_sem[:J] - tab[:J]).max())
    gate("served joint_sem == npz 表(逐值)", diff < 1e-5, f"max|diff|={diff:.2e}")
    jn = list(t_item["joint_names"])[:J]
    print(f"  关节名样例: {jn[:3]} …  (共 {J})")
    # 跨骨架同名关节:证明表是"名字驱动"的
    _pref = ("Deer", "Horse", "Trex", "Raptor3") if a.corpus == "truebones" else tuple(dss["train"].types[:5])
    probe_rigs = [r for r in _pref if r != ot][:3]
    name2rows = {}
    for rig in [ot] + probe_rigs:
        it2 = base[dss["train"].by_type[rig]["demos"][0]] if rig in dss["train"].by_type else None
        if it2 is None:
            continue
        t2 = sem_npz[f"emb__{rig}"].astype(np.float32)
        for i, nm in enumerate(list(it2["joint_names"])[:int(it2["num_joints"])]):
            name2rows.setdefault(nm, []).append((rig, t2[i]))
    # pairs must come from two DISTINCT rigs (a duplicated name inside one rig must not
    # masquerade as a cross-rig match), and finding NO shared name is a FAIL, not a skip --
    # a silent skip would let the final PASS claim a proof that was never performed.
    pairs = []
    for nm, rs in name2rows.items():
        first_by_rig = {}
        for rig_, v in rs:
            first_by_rig.setdefault(rig_, v)
        if len(first_by_rig) >= 2:
            (r1, v1), (r2, v2) = list(first_by_rig.items())[:2]
            pairs.append((nm, r1, v1, r2, v2))
    gate("探针骨架间存在共享关节名(语义表可跨骨架验证)", len(pairs) > 0,
         f"{len(pairs)} 个共享名")
    if pairs:
        cs = []
        for nm, r1, v1, r2, v2 in pairs[:6]:
            c = float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-9))
            cs.append(c)
            print(f"    同名 '{nm[:28]}' @ {r1} vs {r2}: cos={c:.4f}")
        rng = np.random.default_rng(0)
        alln = [v for rs in name2rows.values() for _, v in rs]
        idx1, idx2 = rng.choice(len(alln), 40), rng.choice(len(alln), 40)
        rand_cos = float(np.mean([np.dot(alln[i], alln[j])
                                  / (np.linalg.norm(alln[i]) * np.linalg.norm(alln[j]) + 1e-9)
                                  for i, j in zip(idx1, idx2)]))
        print(f"    对照:随机关节对 cos 均值 = {rand_cos:.4f}")
        gate("同名关节跨骨架嵌入一致(名字驱动的语义表)", min(cs) > 0.99,
             f"min cos={min(cs):.4f} vs 随机 {rand_cos:.4f}")

    # ================= [3] FK == RIC =================
    print("\n=== [3] 数据处理:rot6d-FK 与 RIC 的内部表征一致性 ===")
    # NOTE: both sides derive from the same de-normalised anytop_x, so agreement proves the
    # two channel families are mutually consistent -- NOT correctness against the original
    # BVH world truth (that was established when the corpus was built, not re-proven here).
    for tag, it in (("target", t_item), ("demo", d_item)):
        Jx, Tx = int(it["num_joints"]), int(it["num_frames"])
        xn = np.asarray(it["anytop_x"])[:Jx, :, :Tx].transpose(2, 0, 1)
        mean = np.asarray(it["anytop_mean"])[:Jx]; std = np.asarray(it["anytop_std"])[:Jx]
        raw = (xn * (std[None] + _STD_FLOOR) + mean[None]).astype(np.float64)
        parents = [int(p) for p in it["parent_indices"][:Jx]]
        offsets = np.asarray(it["rest_offsets"])[:Jx]
        fk = recover_from_bvh_rot_np(raw, parents, offsets)
        ric = _recover_world_positions(raw)
        err = np.linalg.norm(fk - ric, axis=-1)
        extent = float(np.linalg.norm(ric - ric.mean(axis=(0, 1)), axis=-1).max())
        rel = float(err.mean()) / max(extent, 1e-9)
        print(f"  {tag:6s} [{Tx}f x {Jx}j]  |FK-RIC| mean={err.mean():.5f} p95={np.percentile(err,95):.5f}"
              f"  motion extent={extent:.3f}  相对均误差={100*rel:.3f}%")
        gate(f"{tag}: rot6d-FK 与 RIC 内部一致,相对均误差 < 0.5%", rel < 0.005)

    # ================= [4] all-rigs sweep =================
    print("\n=== [4] 全骨架 bz=1 前向+反向(每骨架取其最长 clip 当 target) ===")
    model = InContextMotionDiT(in_ch=13, dim=a.dim, depth=a.depth, n_heads=a.heads,
                               d_text=4096, d_joint_sem=4096).to(dev)
    n_par = sum(p.numel() for p in model.parameters())
    lengths = {}
    for i, s in enumerate(base.samples):
        T = int(np.load(s["path"], mmap_mode="r").shape[0])
        lengths.setdefault(s["object_type"], []).append((T, i))
    Tmax_corpus = max(t for v in lengths.values() for t, _ in v)
    gate(f"语料最长 clip ({Tmax_corpus}f) <= target 槽 {a.target_frames}", Tmax_corpus <= a.target_frames)
    rows, peak_all = [], 0.0
    all_types = sorted(set().union(*[set(d.types) for d in dss.values()]))
    for rig in all_types:
        ds_of = next(d for d in (dss["train"], dss["A"], dss["B"], dss["stress"]) if rig in d.types)
        tg_pool = set(ds_of.by_type[rig]["targets"])
        T_len, longest = max((tl, i) for tl, i in lengths[rig] if i in tg_pool)
        pos = ds_of.index.index((rig, longest))
        bb = collate([ds_of[pos]])
        bb = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in bb.items()}
        torch.cuda.reset_peak_memory_stats() if dev == "cuda" else None
        c = dict(joint_bias=bb["joint_bias"], frame_valid=bb["frame_valid"],
                 joint_valid=bb["joint_valid"], text=bb["text"], joint_sem=bb["joint_sem"])
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=a.bf16):
            loss = cfm_loss(model, bb["x"], is_target=bb["is_target"], valid=bb["valid"],
                            gammas=KIMODO_GAMMAS, **c)
        model.zero_grad(set_to_none=True); loss.backward()
        g = torch.sqrt(sum((p.grad ** 2).sum() for p in model.parameters() if p.grad is not None))
        mem = torch.cuda.max_memory_allocated() / 2**30 if dev == "cuda" else 0.0
        peak_all = max(peak_all, mem)
        ok = bool(torch.isfinite(loss)) and bool(torch.isfinite(g))
        cov = ("text" in bb) and ("joint_sem" in bb)
        rows.append((rig, int(bb["joint_valid"].sum()), min(T_len, TARGET_FRAMES), float(loss.detach()), ok, mem, cov))
        if not ok:
            gate(f"{rig}: loss/grad 有限", False)
    rows.sort(key=lambda r: -r[1])
    print(f"  模型 {n_par/1e6:.2f}M | 扫过 {len(rows)} 个骨架 | 峰值显存 {peak_all:.2f} GiB (bz=1)")
    print(f"  {'骨架':18s} {'J':>4s} {'T_tgt':>6s} {'loss':>9s} {'mem GiB':>8s}")
    for rig, j, tt, lo, ok, mem, cov in rows[:5]:
        print(f"  {rig:18s} {j:4d} {tt:6d} {lo:9.3f} {mem:8.2f}")
    print(f"  … (其余 {len(rows)-5} 个省略)")
    gate("全部骨架 forward+backward 有限", all(r[4] for r in rows) and len(rows) >= 60,
         f"{len(rows)} rigs, max J={rows[0][1]} ({rows[0][0]})")
    gate("最大骨架 J <= 模型上限 160", rows[0][1] <= 160)
    gate("每个骨架的 batch 都含 text 与 joint_sem", all(r[6] for r in rows),
         f"{sum(r[6] for r in rows)}/{len(rows)}")

    # ---- [4b] B8 worst case: the sweep proves bz=1 only; the trainer runs B=8. Build the worst
    # realistic training batch (8 items of the largest-J TRAINING rig, longest target) and measure.
    train_rigs = set(dss["train"].types)
    big_rig = max((r for r in rows if r[0] in train_rigs), key=lambda r: r[1])[0]
    tg_pool = set(dss["train"].by_type[big_rig]["targets"])
    _, longest = max((tl, i) for tl, i in lengths[big_rig] if i in tg_pool)
    pos8 = dss["train"].index.index((big_rig, longest))
    bb8 = collate([dss["train"][pos8] for _ in range(a.worst_batch)])
    bb8 = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in bb8.items()}
    torch.cuda.reset_peak_memory_stats()
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=a.bf16):
        loss8 = cfm_loss(model, bb8["x"], is_target=bb8["is_target"], valid=bb8["valid"],
                         gammas=KIMODO_GAMMAS,
                         **dict(joint_bias=bb8["joint_bias"], frame_valid=bb8["frame_valid"],
                                joint_valid=bb8["joint_valid"], text=bb8["text"],
                                joint_sem=bb8["joint_sem"]))
    model.zero_grad(set_to_none=True); loss8.backward()
    g8 = torch.sqrt(sum((q.grad ** 2).sum() for q in model.parameters() if q.grad is not None))
    print(f"  [4b] B{a.worst_batch} 最坏 batch = {a.worst_batch} x {big_rig} (训练桶最大 J): "
          f"loss={float(loss8.detach()):.3f}, |grad|={float(g8):.3e}")
    gate(f"B{a.worst_batch} 最坏情况 loss 与梯度有限", bool(torch.isfinite(loss8)) and bool(torch.isfinite(g8)))
    model.zero_grad(set_to_none=True)
    mem8 = torch.cuda.max_memory_allocated() / 2**30
    print(f"       峰值显存 {mem8:.2f} GiB")
    gate(f"B{a.worst_batch} 最坏情况显存 < {a.mem_gate_gib:.0f} GiB", mem8 < a.mem_gate_gib, f"{mem8:.2f} GiB")

    # ================= [5] hyperparameters =================
    print("\n=== [5] run-1 超参数(签核用) ===")
    hp = {
        "model": f"InContextMotionDiT dim=256 depth=6 heads=8 ({n_par/1e6:.2f}M), max_T=4096, max_J=160",
        "layout": f"[demo {DEMO_FRAMES} | target {TARGET_FRAMES}] = T 304; K=1 demo; target=clip head",
        "objective": "CFM x0-prediction; grouped loss KIMODO_GAMMAS x sqrt(N_i/N_total); valid 必填",
        "gammas": str(KIMODO_GAMMAS),
        "optim": "AdamW lr=3e-4 wd=0, grad-clip 1.0, bf16 关闭(fp32 起步), seed 0",
        "batch": "B=8 单卡 H100 (~445 ms/step), 骨架均衡采样; epoch=86 个满批=688 次均衡抽样(drop_last 弃 6 次)",
        "epochs": "400 (≈4–5 h) + 每 25 ep 存 ckpt + val(A 桶 flow_loss)",
        "eval": "A/B 桶: 10 步 ODE 采样; P5 梯度连通性随 val 打印",
        "cfg": "run-1 无 CFG(基线); CFG 双路 drop 为 run-2",
    }
    for k, v in hp.items():
        print(f"  {k:10s} {v}")

    print()
    if FAILS:
        print("PREFLIGHT FAILED:"); [print("  -", f) for f in FAILS]; sys.exit(1)
    print("preflight PASSED — 所有条件输入与源头一致,数据处理自洽,全骨架可跑")

if __name__ == "__main__":
    main()
