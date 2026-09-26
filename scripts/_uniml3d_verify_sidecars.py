#!/usr/bin/env python3
"""Acceptance harness for the four UniML3D training sidecars (2026-09-16).

Reads the frozen corpus, its OWN source statistics, the four sidecars and the exclusion artifact.
Constructs the real Ktjd17Base at --rep_norm rest and the real InContextPairs in the configuration
a launch would use, and checks the things a launch would otherwise discover at step 0.

EVERY EXPECTATION IS BUILT FROM A SOURCE THE ARTIFACT UNDER TEST DID NOT WRITE (codex r1 P2-4/5/6:
the previous harness reconstructed the supervision mask out of the artifact itself, never bound the
embeddings to the descriptions they claim to encode, and accepted a pair loader that silently
dropped 5,067 of 6,611 training clips):

  1  Ktjd17Base(normalization="rest") constructs; the provenance it would pin is printed; the
     served cut is exactly the split index of the exclusion artifact.
  2  the per-cell statistics artifact is re-derived FROM dataset/.../stats/rig_stats.npz -- source
     hash, generation, mask, constants, floor and means, plus the extrema test that a "constant"
     really is constant and that no mean sits outside the observed range.
  3  every served rig (5,263, not a sample): joint-order hash, supervise_mask vs channel_valid, and
     the serving divisor is the ANALYTIC s_rig/gains one, so no empirical std can reach a tensor.
  4  the semantics table is bound to the CURRENT descriptions by recomputing the canonical digest
     the builder stores, plus [J, __dim] shape, finiteness and no zero row for every served rig.
  5  sampled clips PLUS every partial-heading clip of the cut: caption string and vector;
     joint_sem rows; the excluded cells carry normalized zero and decoding AFTER zeroing them
     returns the raw payload; anytop_mean IS the skeleton rest frame on supervised cells; heading
     validity rides plane 17.  The two error reductions skip the root heading cells of
     heading-INVALID frames -- that cell holds a conditional constant (cos 1 while the heading is
     valid, 0 where it is not) and a frame-blind reduction fails a correct artifact (codex r2 P2-5:
     clip 7db2e5f8ae9ab49e4b17, 21 valid / 29 invalid frames, spurious rel err 0.362).  The raw
     invalid-heading cells are still separately asserted to be zero.
  6  the caption sidecars over the ENTIRE served cut, resolved INDEPENDENTLY: the order the loader
     serves is ordered_captions() -- primary first -- and must reproduce the manifest's list
     (codex r2 P2-3: checking `captions` alone accepted a wrong `primary_caption`); the canonical
     occurrence keys must all be present and carry the right text; and every served vector is
     resolved through the sha1-of-caption keyed ENCODER SHARDS rather than through the row the
     loader just read (codex r2 P2-4: swapping two real embedding rows still passed).
  7  InContextPairs with --demo_rest --rest_demo_self_pairs (the launch configuration): the target
     set is EXACTLY the split, nothing is dropped, and every rig is represented.
"""
import argparse, hashlib, json, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.anytop_dataset import _STD_FLOOR
from src.data.caption_keys import canonical_occurrences, ordered_captions
from src.data.incontext_pairs import InContextPairs
from src.data.ktjd17_incontext import Ktjd17Base, ktjd17_split_names

ap = argparse.ArgumentParser(allow_abbrev=False)
ap.add_argument("--root", default="dataset/ktjd17_uniml3d_v1")
ap.add_argument("--percell", default="data/uniml3d_norm_stats_v1.npz")
ap.add_argument("--joint_sem", default="data/joint_semantics_llm2vec_uniml3d_v3.npz")
ap.add_argument("--caption_cache", default="data/uniml3d_caption_llm2vec_v1")
ap.add_argument("--texts_json", default="data/uniml3d_motion_texts_v1.json")
ap.add_argument("--exclude_clips", default="configs/uniml3d_v1_visual_exclusions.json")
ap.add_argument("--rig_stats", default=None, help="default <root>/stats/rig_stats.npz")
ap.add_argument("--n_clips", type=int, default=20)
ap.add_argument("--n_pairs", type=int, default=8)
ap.add_argument("--self_demo_pairs", type=int, required=True, choices=(0, 1),
                help="rest_demo_self_pairs for section 7.  REQUIRED, with no default: this harness reports on "
                     "the configuration it is given, and a PASS obtained under an implicit 1 would read as "
                     "validating a launch that set SELF_DEMO_PAIRS=0 (codex uniml3d sidecars r2 item 1).  The "
                     "launch configuration is 1; 0 exists only to demonstrate that the exact-membership "
                     "assertion rejects the configuration that silently drops most training clips.")
a = ap.parse_args()

fail: list[str] = []
def check(ok: bool, msg: str):
    print(("  PASS  " if ok else "  FAIL  ") + msg, flush=True)
    if not ok:
        fail.append(msg)

print("=" * 88)
print("1. Ktjd17Base(normalization='rest')")
base = Ktjd17Base(a.root, caption_emb_cache=a.caption_cache, joint_semantics=a.joint_sem,
                  texts_json=a.texts_json, percell_stats=a.percell,
                  exclude_clips=(a.exclude_clips or None), random_caption=False,
                  normalization="rest")
rigs = sorted({s["object_type"] for s in base.samples})
print(f"  constructed: {len(base)} clips / {len(rigs)} rigs   caption_dim={base.caption_dim}   "
      f"sem_dim={base._sem_dim}")
print("  provenance:")
for k, v in base.provenance.items():
    print(f"    {k:28s} {v}")
names = ktjd17_split_names(a.root, exclude=(a.exclude_clips or None))
cut = set(names["train"]) | set(names["val"])
served = {s["path"] for s in base.samples}
print(f"  split sizes: " + "  ".join(f"{k}={len(v)}" for k, v in sorted(names.items())))
check(base.provenance["target_centering"] == "ktjd_rest_centered_scale_v1",
      "target_centering pins the rest-normalised objective")
check(served == cut,
      f"the served cut IS the split index ({len(served)} clips; "
      f"{len(cut - served)} split clips unserved, {len(served - cut)} served off-split)")

print()
print("=" * 88)
print("2. the per-cell statistics artifact, re-derived from the corpus's own rig_stats.npz")
src_p = Path(a.rig_stats) if a.rig_stats else Path(a.root) / "stats" / "rig_stats.npz"
src_sha = hashlib.sha256(src_p.read_bytes()).hexdigest()
z = np.load(a.percell, allow_pickle=False)
meta = json.loads(str(z["__meta"]))
zs = np.load(src_p, allow_pickle=False)
print(f"  source {src_p} sha256 {src_sha[:16]}  generation {str(zs['__generation_id'])}")
print(f"  artifact __meta: n_valid={meta['n_valid']:,} n_constant_excluded="
      f"{meta['n_constant_excluded']:,} n_floored={meta['n_floored']:,} "
      f"std_min={meta['std_min']} generation={meta['generation_id']}")
check(meta.get("source_sha256") == src_sha and meta.get("source_npz_sha256") == src_sha,
      "the artifact names the source file it was actually built from (sha256)")
check(str(zs["__generation_id"]) == base.generation_id == str(meta["generation_id"]),
      "source statistics, corpus and artifact agree on the generation id")
check([str(r) for r in z["rig_ids"]] == [str(r) for r in zs["rig_ids"]]
      and np.array_equal(np.asarray(z["joint_count"]), np.asarray(zs["joint_count"])),
      "rig order and joint counts are the source's")

STD_MIN = float(meta["std_min"])
mean_s = np.asarray(zs["mean"], np.float64); std_s = np.asarray(zs["std"], np.float64)
vm = np.asarray(zs["valid_mask"]); mn = np.asarray(zs["minimum"], np.float64)
mx = np.asarray(zs["maximum"], np.float64); jc = np.asarray(zs["joint_count"], np.int64)
const_e = vm & (std_s == 0.0)
sup_e = vm & ~const_e
tiny_e = vm & (std_s > 0.0) & (std_s < STD_MIN)
mean_e = np.where(vm, mean_s, 0.0).astype(np.float32)
std_e = (np.where(sup_e, np.maximum(std_s, STD_MIN), 1.0) - _STD_FLOOR).astype(np.float32)
check(np.array_equal(np.asarray(z["supervise_mask"]), sup_e),
      f"supervise_mask == source valid & not-exactly-constant ({int(sup_e.sum()):,} cells)")
check(np.array_equal(np.asarray(z["was_constant"]), const_e),
      f"was_constant == source valid & std==0 ({int(const_e.sum()):,} cells)")
check(np.array_equal(np.asarray(z["was_floored"]), tiny_e),
      f"was_floored == source valid & 0 < std < {STD_MIN} ({int(tiny_e.sum()):,} cells)")
check(np.array_equal(np.asarray(z["mean"]), mean_e),
      "every stored mean is the source population mean (0 off the valid mask)")
check(np.array_equal(np.asarray(z["std"]), std_e),
      f"every stored std is max(source std, {STD_MIN}) - _STD_FLOOR on supervised cells, else 1")
# EXTREMA: a restoration constant that the data never took is the failure mode a self-consistent
# artifact cannot show.  Ask the source's own min/max instead.
c = const_e
check(bool(np.array_equal(mn[c], mx[c])),
      f"every was_constant cell really is constant in the source (min == max, {int(c.sum()):,})")
mu_art = np.asarray(z["mean"]).astype(np.float64)
# relative, because the artifact stores float32 of a float64 population mean and these are physical
# units up to 6.7e4; 1e-6 is ~20x float32 resolution and still O(1) below any real corruption
dev_c = (float((np.abs(mu_art[c] - mn[c]) / np.maximum(np.abs(mn[c]), 1.0)).max())
         if c.any() else 0.0)
check(dev_c <= 1e-6, f"every restored constant IS that observed value (worst rel |d| {dev_c:.3e})")
tol = 1e-5 * np.maximum(np.abs(mn), np.abs(mx)) + 1e-9
out_of_range = int((vm & ((mu_art < mn - tol) | (mu_art > mx + tol))).sum())
check(out_of_range == 0,
      f"no stored mean lies outside the source's observed [min,max] ({out_of_range} do)")
pad_valid = int(sum(int(vm[i, int(J):].sum()) for i, J in enumerate(jc)))
check(pad_valid == 0, f"no valid cell beyond joint_count in the source ({pad_valid} found)")
check(float(meta["std_floor"]) == float(_STD_FLOOR)
      and "x * (std + _STD_FLOOR) + mean" in str(meta["convention"]),
      "the artifact declares this repo's floor and de-normalisation convention")

print()
print("=" * 88)
print("3. every served rig: order hash, supervise_mask reconciliation, analytic rest divisor")
pc_rigs = {str(r): i for i, r in enumerate(z["rig_ids"])}
was_const = np.asarray(z["was_constant"])
valid = np.asarray(z["supervise_mask"]) | was_const          # the source valid_mask, reconstructed
jc_a = np.asarray(z["joint_count"], np.int64)
drop_from_cv = 0          # structural cells that the mask removes, over SERVED rigs
const_served = 0          # cells the artifact removes from supervision, restricted to structural
struct_total = cv_total = 0
zero_sem_rows = 0
worst_div = 0.0
g = np.asarray(base.gains, np.float64)
for r in rigs:                       # _skeleton() raises on any order-hash mismatch
    m = base.static_masks(r)
    cv = m["channel_valid"]
    J = cv.shape[0]
    struct = np.zeros_like(cv)
    struct[:, :13] = True
    struct[0, 13:17] = True
    struct_total += int(struct.sum())
    cv_total += int(cv.sum())
    drop_from_cv += int((struct & ~cv).sum())
    i = pc_rigs[r]
    # a structural cell loses supervision if the artifact calls it exact-constant OR never saw a
    # valid observation there (the 8 rigs whose heading projects near-vertically every frame)
    const_served += int(((was_const[i, :J] | ~valid[i, :J]) & struct).sum())
    sem = base._sem[r]
    zero_sem_rows += int((np.abs(sem).sum(axis=1) == 0).sum())
    # THE DIVISOR: under `rest` it must be s_rig / frozen block gains, never an empirical std.
    s_rig = float(base.skeleton(r)["s_rig"])
    exp = np.ones((J, 18), dtype=np.float32)
    exp[:, 0:3] = s_rig / g[0]
    exp[:, 9:12] = s_rig / g[1]
    exp[0, 13:15] = s_rig / g[2]
    exp = (exp - _STD_FLOOR).astype(np.float32)[:, :17]
    worst_div = max(worst_div, float(np.abs(base._stats(r)[1] - exp).max()))
print(f"  rigs checked: {len(rigs)} (order hash verified for every one)")
print(f"  structural cells {struct_total:,} -> channel_valid {cv_total:,} "
      f"(dropped {drop_from_cv:,}, {drop_from_cv/struct_total*100:.2f}%)")
print(f"  artifact (was_constant | never-valid) on the same cells: {const_served:,}")
check(drop_from_cv == const_served,
      f"channel_valid drops exactly the cells the artifact unsupervises ({drop_from_cv:,})")
check(zero_sem_rows == 0, f"no all-zero joint_semantics row over {len(rigs)} rigs")
check(worst_div == 0.0,
      f"the serving divisor is analytic s_rig/gains on every rig, not an empirical std "
      f"(worst |d| {worst_div:.3e})")

print()
print("=" * 88)
print("4. the semantics table is bound to the descriptions the corpus carries NOW")
descs = {}
for p in sorted((Path(a.root) / "skeletons").glob("*.npz")):
    with np.load(p, allow_pickle=False) as zz:
        descs[p.stem] = [str(v) for v in zz["joint_descriptions"]]
want_dig = hashlib.sha256(json.dumps(descs, sort_keys=True,
                                     separators=(",", ":")).encode()).hexdigest()
with np.load(a.joint_sem, allow_pickle=False) as zt:
    got_dig = str(zt["__descriptions_sha256"])
    sem_dim = int(zt["__dim"])
    enc, jos = str(zt["__encoder"]), str(zt["__joint_order_source"])
    n_tables = sum(1 for k in zt.files if k.startswith("emb__"))
print(f"  table: {n_tables} rigs, dim {sem_dim}, encoder {enc}, joint order from {jos}")
print(f"  __descriptions_sha256 {got_dig[:16]}   recomputed from the corpus {want_dig[:16]}")
check(got_dig == want_dig,
      "__descriptions_sha256 is the digest of the corpus's CURRENT joint_descriptions")
check(sem_dim == base._sem_dim and sem_dim > 0, f"__dim {sem_dim} is what the loader took")
bad_shape = bad_finite = 0
for r in rigs:
    t = np.asarray(base._sem[r])
    if t.shape != (len(base.skeleton(r)["parents"]), sem_dim):
        bad_shape += 1
    elif not np.isfinite(t).all():
        bad_finite += 1
check(bad_shape == 0, f"every served rig's table is [J, {sem_dim}] ({bad_shape} are not)")
check(bad_finite == 0, f"every served rig's table is finite ({bad_finite} are not)")

print()
print("=" * 88)
print("5. probed clips: captions, joint_sem, de-normalisation, rest mean, heading")
texts = json.loads(Path(a.texts_json).read_text())
manifest = {str(json.loads(l)["clip_id"]): json.loads(l)
            for l in open(Path(a.root) / "manifests" / "clips.jsonl")}
keys = json.load(open(Path(a.caption_cache).with_suffix(".keys.json")))
embs = np.load(Path(a.caption_cache).with_suffix(".embs.npy"), mmap_mode="r")
key_row = {k: i for i, k in enumerate(keys)}
from src.data.ktjd17.loader import load_motion_npz

# THE INDEPENDENT ROUTE TO A CAPTION VECTOR.  scripts/_build_caption_llm2vec.py encodes each UNIQUE
# caption once into u_shardNNN.embs.npy keyed by sha1(text) and then fans the rows out to
# occurrence order.  Resolving a caption here through the shards -- rather than through the
# occurrence row the loader just read -- is the only way a swapped fan-out row can be seen.
_pfx = Path(a.caption_cache)
_metas = sorted(_pfx.parent.glob(f"{_pfx.name}.u_shard*.meta.json"))
if not _metas:
    raise SystemExit(f"REFUSED: no {_pfx.name}.u_shard*.meta.json -- the encoder shards the caption "
                     f"vectors must be resolved against are missing")
U, urow = [], {}
for _sn, _mp in enumerate(_metas):
    _tag = _mp.name[len(_pfx.name) + 1:-len(".meta.json")]
    U.append(np.load(f"{_pfx}.{_tag}.embs.npy", mmap_mode="r"))
    for _j, _k in enumerate(json.loads(Path(f"{_pfx}.{_tag}.keys.json").read_text())):
        urow[_k] = (_sn, _j)
print(f"  encoder shards: {len(_metas)} file(s), {len(urow):,} unique captions")

def shard_vec(cap: str):
    """The embedding the encoder produced for this exact string, or None if it never encoded it."""
    hit = urow.get(hashlib.sha1(cap.encode()).hexdigest())
    return None if hit is None else np.asarray(U[hit[0]][hit[1]], np.float32)

def manifest_caps(clip: str):
    return [str(t).strip() for t in (manifest[clip].get("captions") or []) if str(t).strip()]

rng = np.random.default_rng(0)
samp = sorted(rng.choice(len(base), size=min(a.n_clips, len(base)), replace=False).tolist())
# EVERY partial-heading clip of the cut, not a sample of them: the root heading cell is exactly the
# one whose round trip a frame-blind reduction gets wrong, and 20 random draws out of 6,880 hit
# none of the 106 clips that have one.
idx_of = {s["path"]: k for k, s in enumerate(base.samples)}
partial = sorted(c for c in cut
                 if float(manifest[c].get("qa", {}).get("heading_valid_fraction", 1.0)) < 1.0)
idxs = sorted(set(samp) | {idx_of[c] for c in partial})
print(f"  probing {len(idxs)} clips: {len(samp)} sampled + {len(partial)} with a partly invalid "
      f"heading (all of them)")
worst_rt = worst_rt_z = worst_rest = worst_excl = 0.0
cap_ok = sem_ok = head_ok = True
sem_zero = 0
samp_first4 = set(samp[:4])
blind_rt_z = blind_excl = 0.0        # report-only: what a frame-blind reduction would measure
for n, i in enumerate(idxs):
    it = base[i]
    rig, clip = it["object_type"], it["motion_id"]
    J, T = int(it["num_joints"]), int(it["num_frames"])
    x = np.asarray(it["anytop_x"]).transpose(2, 0, 1)             # [T,J,18]
    mu = np.asarray(it["anytop_mean"]); sd = np.asarray(it["anytop_std"])
    payload = load_motion_npz(Path(a.root) / manifest[clip]["motion_relpath"],
                              expected_fps_target=30.0)
    raw = np.asarray(payload["motion"])[:T, :J, :17].astype(np.float32)
    scale = np.maximum(np.abs(raw).max(), 1e-6)
    raw_back = x[:T, :, :17] * (sd[None, :, :17] + _STD_FLOOR) + mu[None, :, :17]
    worst_rt = max(worst_rt, float(np.abs(raw_back - raw).max() / scale))
    # THE DECODE THE SAMPLER ACTUALLY FEEDS: unsupervised cells leave the model input, so they
    # arrive at the decoder as normalized zero.  If a restoration constant is not the data's own
    # value, this is where it shows -- the old harness inverted BEFORE applying the mask and could
    # not see it.
    cv = np.asarray(base.static_masks(rig)["channel_valid"])      # [J,17]
    hv = np.asarray(payload["heading_valid"])[:T].astype(bool)
    # The root heading cells hold a CONDITIONAL constant: cos is 1 for every heading-valid frame
    # (hence exact-constant, hence unsupervised, hence restored as 1) and the payload stores 0
    # where the heading is invalid.  Both are correct; plane 17 is what tells a consumer which.
    # Measuring the excluded-cell and restoration errors over those frames fails a correct
    # artifact, so they are excluded from those two reductions only -- the raw cells are still
    # asserted to be zero below, and the unmasked round trip above still covers them.
    meas = np.ones((T, J, 17), dtype=bool)
    meas[~hv, 0, 15:17] = False
    xz = x[:T, :, :17].copy()
    if (~cv).any():
        sel = meas & ~cv[None]
        if sel.any():
            worst_excl = max(worst_excl, float(np.abs(xz[sel]).max()))
        xz[:, ~cv] = 0.0
    raw_back_z = xz * (sd[None, :, :17] + _STD_FLOOR) + mu[None, :, :17]
    rt_z = float(np.abs((raw_back_z - raw)[meas]).max() / scale)
    worst_rt_z = max(worst_rt_z, rt_z)
    blind_rt_z = max(blind_rt_z, float(np.abs(raw_back_z - raw).max() / scale))
    if (~cv).any():
        blind_excl = max(blind_excl, float(np.abs(x[:T, :, :17][:, ~cv]).max()))

    # caption: the order the LOADER serves (primary first) must be the manifest's list, and the
    # served VECTOR is compared against the encoder shards, not against the row it came from
    want_caps = manifest_caps(clip)
    loader_caps = ordered_captions(texts.get(f"{clip}.npy", {}))
    exp_vec = shard_vec(loader_caps[0]) if loader_caps else None
    if (f"{clip}__cap0" not in key_row or loader_caps != want_caps
            or it["caption"] != want_caps[0] or exp_vec is None
            or not np.array_equal(np.asarray(it["caption_emb"], np.float32), exp_vec)):
        cap_ok = False
        print(f"    caption MISMATCH on {clip}: item={it['caption']!r} "
              f"loader_order={loader_caps[:2]} manifest={want_caps[:2]} "
              f"shard_row={'absent' if exp_vec is None else 'present'}")

    sem = np.asarray(it["joint_semantics"])
    nz = int((np.abs(sem).sum(axis=1) == 0).sum())
    sem_zero += nz
    if sem.shape[0] != J or nz:
        sem_ok = False

    # heading validity rides plane 17 for every joint, and invalid frames carry no heading
    if not np.array_equal(x[:T, :, 17], np.broadcast_to(hv.astype(np.float32)[:, None], (T, J))):
        head_ok = False
        print(f"    heading plane MISMATCH on {clip}")
    elif (~hv).any() and float(np.abs(raw[~hv][:, 0, 15:17]).max()) != 0.0:
        head_ok = False
        print(f"    invalid-heading frames carry a nonzero heading on {clip}")

    # "rest" premise: anytop_mean == the skeleton-derived rest frame wherever the cell is supervised
    rest = base._rest_raw17(rig)
    sup = np.asarray(z["supervise_mask"])[pc_rigs[rig], :J]
    d = float(np.abs(mu[:, :17][sup] - rest[sup]).max()) if sup.any() else 0.0
    worst_rest = max(worst_rest, d)
    if n < 4 or (i in samp_first4):
        print(f"    {clip} rig={rig} J={J} T={T} cap={it['caption']!r}")
        print(f"      anytop_x{np.asarray(it['anytop_x']).shape} sem{sem.shape} "
              f"cap_emb{np.asarray(it['caption_emb']).shape} parents{it['parent_indices'].shape} "
              f"rest_off{it['rest_offsets'].shape} Rrest{it['R_rest_global'].shape} "
              f"s_rig={it['s_rig']:.4g}  denorm_rel_err={rt_z:.2e}  heading_valid={int(hv.sum())}/{T}")
check(cap_ok, f"served caption string and shard-resolved vector agree with the manifest on all "
              f"{len(idxs)} probed clips")
check(sem_ok, f"joint_semantics is [J,dim] with no zero row on all {len(idxs)} clips "
              f"({sem_zero} zero rows)")
check(head_ok, f"plane 17 is the payload's heading_valid and invalid frames carry no heading")
check(worst_rt < 1e-5, f"de-normalisation returns the raw payload (worst rel err {worst_rt:.2e})")
check(worst_excl < 1e-5,
      f"unsupervised cells are served at normalized zero, off the invalid-heading root cells "
      f"(worst |x| {worst_excl:.2e})")
check(worst_rt_z < 1e-5,
      f"decoding AFTER zeroing the unsupervised cells still returns the raw payload, off the "
      f"invalid-heading root cells (worst rel err {worst_rt_z:.2e})")
check(worst_rest == 0.0,
      f"anytop_mean == skeleton rest frame on every supervised cell (worst |d| {worst_rest:.2e})")
print(f"  (frame-blind reductions over the same clips would read |x| {blind_excl:.2e} and rel err "
      f"{blind_rt_z:.2e}: the invalid-heading root cells they fold in)")

print()
print("=" * 88)
print("6. the caption sidecars over the WHOLE served cut, resolved independently")
occ_all = dict(canonical_occurrences(texts))      # key -> caption: the law the cache rows follow
miss_txt, bad_order, miss_key, bad_occ, extra_occ = [], [], [], [], []
want_of: dict[str, list[str]] = {}
for c in sorted(cut):
    entry = texts.get(f"{c}.npy")
    if entry is None:
        miss_txt.append(c)
        continue
    want = manifest_caps(c)
    # ordered_captions() -- NOT entry["captions"] -- is what the loader hands the model, so a wrong
    # primary_caption is invisible to a check that only reads the captions list.
    got = ordered_captions(entry)
    if got != want:
        bad_order.append((c, got[:2], want[:2]))
        continue
    want_of[c] = want
    for i, capt in enumerate(want):
        k = f"{c}__cap{i}"
        if k not in key_row:
            miss_key.append(k)
        elif occ_all.get(k) != capt:
            bad_occ.append(k)
    if f"{c}__cap{len(want)}" in key_row:
        extra_occ.append(f"{c}__cap{len(want)}")
n_occ = sum(len(v) for v in want_of.values())
print(f"  cut {len(cut)} clips / {n_occ:,} caption occurrences: texts entries missing "
      f"{len(miss_txt)}, loader-order != manifest {len(bad_order)}, cache keys missing "
      f"{len(miss_key)}, key/text mismatches {len(bad_occ)}, extra occurrences {len(extra_occ)}")
check(not miss_txt, f"every clip of the cut has a texts_json entry ({len(miss_txt)} do not: "
                    f"{miss_txt[:3]})")
check(not bad_order,
      f"ordered_captions() -- primary first, the order the loader serves -- reproduces the "
      f"manifest's caption list for every clip ({len(bad_order)} do not: {bad_order[:2]})")
check(not miss_key and not bad_occ and not extra_occ,
      f"the canonical occurrence keys are all present, carry the manifest's text, and stop where "
      f"the caption list does ({len(miss_key)} missing {miss_key[:2]}, {len(bad_occ)} wrong "
      f"{bad_occ[:2]}, {len(extra_occ)} extra {extra_occ[:2]})")

# every served vector against the shard the ENCODER wrote, resolved by sha1 of the caption text
rows, exp_sn, exp_j, no_uni = [], [], [], []
for c in sorted(want_of):
    for i, capt in enumerate(want_of[c]):
        k = f"{c}__cap{i}"
        hit = urow.get(hashlib.sha1(capt.encode()).hexdigest())
        if hit is None:
            no_uni.append(k)
        elif k in key_row:
            rows.append(key_row[k]); exp_sn.append(hit[0]); exp_j.append(hit[1])
rows = np.asarray(rows, np.int64)
bad_vec = 0
for s0 in range(0, len(rows), 4096):
    sl = slice(s0, s0 + 4096)
    got_v = np.asarray(embs[rows[sl]], np.float32)
    want_v = np.stack([np.asarray(U[a_][b_], np.float32)
                       for a_, b_ in zip(exp_sn[sl], exp_j[sl])])
    bad_vec += int((~(got_v == want_v).all(axis=1)).sum())
print(f"  {len(rows):,} served rows resolved through the encoder shards; unencoded captions "
      f"{len(no_uni)}; rows differing from the shard vector {bad_vec}")
check(not no_uni, f"every served caption was actually encoded ({len(no_uni)} were not: "
                  f"{no_uni[:3]})")
check(bad_vec == 0,
      f"every served embedding row IS the vector the encoder produced for that exact caption "
      f"({bad_vec} of {len(rows):,} differ)")

print()
print("=" * 88)
SP = bool(a.self_demo_pairs)
print(f"7. InContextPairs (demo_rest=1, demo_frames=1, target_frames=240, "
      f"rest_demo_self_pairs={int(SP)}) as the launcher builds it")
ds_tr = InContextPairs(base, names["train"], names["train"], object_types=None,
                       demo_frames=1, target_frames=240, balance_skeletons=True, seed=0,
                       emit_fk_fields=True, emit_graph_v2=True, demo_rest=True,
                       rest_demo_self_pairs=SP)
ds_va = InContextPairs(base, names["val"], names["train"], object_types=None,
                       demo_frames=1, target_frames=240, balance_skeletons=False, seed=1,
                       emit_fk_fields=True, emit_graph_v2=True, demo_rest=True,
                       rest_demo_self_pairs=SP)
def targets_of(ds):
    return {base.samples[i]["path"] for ot in ds.by_type for i in ds.by_type[ot]["targets"]}
tg_tr, tg_va = targets_of(ds_tr), targets_of(ds_va)
print(f"  train: {len(ds_tr)} targets / {len(ds_tr.types)} rigs / {ds_tr.pair_count()} pairs; "
      f"dropped as self-only-demo: {ds_tr.n_dropped_self_only}")
print(f"  val  : {len(ds_va)} targets / {len(ds_va.types)} rigs; "
      f"dropped as self-only-demo: {ds_va.n_dropped_self_only}")
check(tg_tr == set(names["train"]),
      f"the train target set IS the train split ({len(names['train'])} clips; "
      f"{len(set(names['train']) - tg_tr)} dropped, {len(tg_tr - set(names['train']))} extra)")
check(tg_va == set(names["val"]),
      f"the val target set IS the val split ({len(names['val'])} clips; "
      f"{len(set(names['val']) - tg_va)} dropped, {len(tg_va - set(names['val']))} extra)")
check(ds_tr.n_dropped_self_only == 0 and ds_va.n_dropped_self_only == 0,
      f"no target is dropped for having only itself as a demo "
      f"({ds_tr.n_dropped_self_only} train, {ds_va.n_dropped_self_only} val)")
check(len(ds_tr.types) == len(rigs),
      f"every served rig has a train target ({len(ds_tr.types)} of {len(rigs)})")

want = ["x", "is_target", "frame_valid", "geodesic", "text", "joint_sem", "parents",
        "rest_offsets", "n_joints", "anytop_mean", "anytop_std", "R_rest_global"]
for k in range(min(a.n_pairs, len(ds_tr))):
    it = ds_tr[k]
    miss = [f for f in want if f not in it]
    if miss:
        check(False, f"pair item {k} is missing {miss}")
        break
    if k == 0:
        print("  field shapes:")
        for f in want:
            v = np.asarray(it[f])
            print(f"    {f:16s} {tuple(v.shape)!s:22s} {v.dtype}")
        # channel_valid rides the item only under skeleton augmentation; otherwise the trainer
        # fetches it per object_type from the base (ktjd17_incontext.py static_masks docstring)
        cvp = np.asarray(base.static_masks(it["object_type"])["channel_valid"])
        print(f"    channel_valid    {tuple(cvp.shape)!s:22s} {cvp.dtype} "
              f"(via base.static_masks, {int(cvp.sum())}/{cvp.size} true)")
    if not np.isfinite(np.asarray(it["x"], np.float32)).all():
        check(False, f"pair item {k} carries non-finite x")
        break
else:
    check(True, f"{min(a.n_pairs, len(ds_tr))} pair items built, all fields present and finite")

print()
print("=" * 88)
print(("ACCEPTANCE: PASS" if not fail else f"ACCEPTANCE: {len(fail)} FAILURE(S)"))
for f in fail:
    print("  -", f)
raise SystemExit(1 if fail else 0)
