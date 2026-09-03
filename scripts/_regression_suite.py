"""The checks that must pass before any multi-day run starts.

Written after a regression my own tests missed: adding `joint_semantics` to the REQUIRED collate
spec made every run without semantics die at `from_collate_dict`, while the bitwise test kept
passing because it only exercised `__getitem__`. A test that only covers the new path cannot show
the old one still works.

So the suite is organised by the failure classes actually hit in this project, not by module:

  A. bitwise identity of the default path — many samples, BOTH splits, all fields
  B. every optional field, present AND absent, through collate and batch construction
  C. the held-out guard's refusals — a guard that never fires is indistinguishable from a broken one
  D. provenance verification logic
  E. moment-source round trip — normalise then de-normalise must recover physical units
  F. the frozen artifacts still match their seal

    python scripts/_regression_suite.py            # all
    python scripts/_regression_suite.py --only A,C
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import traceback
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

DR = "data/animo4d_L4TB_plus_human_v4b272neutral"
ART = "data/holdout_topologies_v1.json"
SPLITS = "data/holdout_splits_v1"
SEM = "data/joint_semantics_llm2vec_v1.npz"
EST = "scratch/_moment_est_v2.npz"

_results: list[tuple[str, bool, str]] = []


def check(name: str):
    def deco(fn):
        def run():
            try:
                msg = fn() or ""
                _results.append((name, True, msg))
            except Exception as e:
                _results.append((name, False, f"{type(e).__name__}: {e}"))
                if "-v" in sys.argv:
                    traceback.print_exc()
        run.__name__ = fn.__name__
        run._group = name.split(".")[0]
        return run
    return deco


def _tensors_equal(a, b) -> bool:
    if torch.is_tensor(a):
        return torch.is_tensor(b) and a.shape == b.shape and a.dtype == b.dtype and torch.equal(a, b)
    if isinstance(a, np.ndarray):
        return (isinstance(b, np.ndarray) and a.shape == b.shape and a.dtype == b.dtype
                and np.array_equal(a.view(np.uint8), b.view(np.uint8)))
    return a == b


# ---------------------------------------------------------------- A: bitwise default path

@check("A.1 default path is bitwise identical, both splits, 120 samples")
def a1():
    from src.data.anytop_dataset import AnyTopDataset
    from src.data.moment_source import MomentSource
    n = 0
    for split in ("train", "val"):
        # random_crop must be pinned: the train split crops a random temporal window per call, so
        # two datasets can never be bitwise equal there no matter what the code does. Verified
        # directly — the same dataset returns different motion_features on two consecutive reads
        # of index 0 for train, and identical ones for val. Pinning it is what makes this a test
        # of the moment-source plumbing rather than a test of the RNG.
        k = dict(data_root=DR, split=split, num_frames=64, max_joints=144, load_captions=False,
                 random_crop=False)
        a = AnyTopDataset(**k)
        b = AnyTopDataset(**k, moment_source=MomentSource("own"))
        step = max(1, len(a) // 60)
        for i in range(0, len(a), step):
            x, y = a[i], b[i]
            if set(x) != set(y):
                raise AssertionError(f"{split}[{i}] key sets differ: {set(x) ^ set(y)}")
            for key in x:
                if not _tensors_equal(x[key], y[key]):
                    raise AssertionError(f"{split}[{i}] field {key!r} differs")
            n += 1
            if n >= 120:
                break
    return f"{n} samples, all fields, uint8-exact"


# ---------------------------------------------------------------- B: optional fields

@check("B.1 collate + batch works with NO optional fields")
def b1():
    from src.data.anytop_dataset import AnyTopDataset, collate_fn
    from src.models.graph_salad.batch import GraphMotionBatch
    d = AnyTopDataset(data_root=DR, split="val", num_frames=64, max_joints=144,
                      load_captions=False)
    g = GraphMotionBatch.from_collate_dict(collate_fn([d[i] for i in range(4)]))
    if g.joint_semantics is not None:
        raise AssertionError("joint_semantics should be None when the dataset does not emit it")
    return "joint_semantics absent -> None, no missing-key error"


@check("B.2 collate + batch works WITH joint_semantics")
def b2():
    from src.data.anytop_dataset import AnyTopDataset, collate_fn
    from src.models.graph_salad.batch import GraphMotionBatch
    d = AnyTopDataset(data_root=DR, split="val", num_frames=64, max_joints=144,
                      load_captions=False, joint_semantics=SEM)
    g = GraphMotionBatch.from_collate_dict(collate_fn([d[i] for i in range(4)]))
    if g.joint_semantics is None or g.joint_semantics.shape[:2] != (4, 144):
        raise AssertionError(f"bad joint_semantics {None if g.joint_semantics is None else tuple(g.joint_semantics.shape)}")
    j = int(g.joint_mask[0].sum())
    if not bool((g.joint_semantics[0, :j].abs().sum(-1) > 0).all()):
        raise AssertionError("a valid joint has an all-zero semantic row")
    if float(g.joint_semantics[0, j:].abs().sum()) != 0.0:
        raise AssertionError("padding rows are not zero")
    return f"shape {tuple(g.joint_semantics.shape)}, valid rows non-zero, padding zero"


@check("B.3 joint_semantics rejects a table built on a different joint order")
def b3():
    import json, tempfile, os
    from src.data.anytop_dataset import AnyTopDataset
    z = dict(np.load(SEM, allow_pickle=False))
    oh = json.loads(str(z["__order_hash"]))
    victim = sorted(oh)[0]
    oh[victim] = "0" * 64
    z["__order_hash"] = json.dumps(oh)
    f = tempfile.NamedTemporaryFile(suffix=".npz", delete=False)
    np.savez(f.name, **z)
    f.close()
    try:
        d = AnyTopDataset(data_root=DR, split="train", num_frames=64, max_joints=144,
                          load_captions=False, joint_semantics=f.name)
        for i in range(len(d)):
            try:
                d[i]
            except RuntimeError as e:
                if "joint-order hash mismatch" in str(e):
                    return f"caught on {victim}"
            except KeyError:
                pass
        raise AssertionError("a wrong joint order was NOT caught")
    finally:
        os.unlink(f.name)


# ---------------------------------------------------------------- C: guard refusals

@check("C.1 guard REFUSES the full-corpus split under the protocol")
def c1():
    from src.data.anytop_dataset import AnyTopDataset
    from src.data.holdout_guard import guard_dataset
    d = AnyTopDataset(data_root=DR, split="train", num_frames=64, max_joints=144,
                      load_captions=False)
    try:
        guard_dataset(d, data_root=DR, artifact=ART, stage="regression", log=lambda *a: None)
    except SystemExit as e:
        if "HELD-OUT topologies present" not in str(e):
            raise AssertionError(f"refused for the wrong reason: {e}")
        return "refused, as it must"
    raise AssertionError("the full split was ACCEPTED — the guard is not working")


@check("C.2 guard ACCEPTS the retained split")
def c2():
    from src.data.anytop_dataset import AnyTopDataset
    from src.data.holdout_guard import guard_dataset
    out = []
    d = AnyTopDataset(data_root=DR, split="train", num_frames=64, max_joints=144,
                      load_captions=False, splits_dir=SPLITS)
    guard_dataset(d, data_root=DR, artifact=ART, stage="regression", log=out.append)
    if "0 held-out topologies present" not in " ".join(out):
        raise AssertionError(f"unexpected guard output: {out}")
    return f"{len(d.samples)} clips accepted"


@check("C.3 guard REFUSES a wrong artifact hash")
def c3():
    from src.data.holdout_guard import verify_artifact
    try:
        verify_artifact(ART, DR, expect_body_sha="0" * 64)
    except SystemExit:
        return "refused"
    raise AssertionError("a wrong expected-sha was accepted")


@check("C.4 augmentation refuses to synthesise a held topology")
def c4():
    from src.data.anytop_dataset import AnyTopDataset
    targets = {"PZ_African_Buffalo_Female", "PZ_African_Buffalo_Male",
               "PZ_Black_Wildebeest_Female", "PZ_Plains_Zebra_Juvenile",
               "PZ_Plains_Zebra_Male", "PZ_Wild_Water_Buffalo_Juvenile"}
    d = AnyTopDataset(data_root=DR, split="train", num_frames=64, max_joints=144,
                      load_captions=False, splits_dir=SPLITS, augment=True, augment_prob=1.0,
                      removal_rate=0.3, holdout_artifact=ART, aug_seed=20260802)
    idxs = [i for i, s in enumerate(d.samples) if s["object_type"] in targets][:400]
    if not idxs:
        raise AssertionError("none of the known colliding rigs are in the retained split")
    for i in idxs:
        d[i]
    if d.n_aug_rejected_held == 0:
        raise AssertionError(
            "no augmentation was rejected over the rigs KNOWN to collide — either the guard is "
            "broken or the collision set has changed. A guard that never fires proves nothing.")
    return (f"{d.n_aug_rejected_held} rejected over {len(idxs)} draws from the colliding rigs "
            f"(seeded, so this count is reproducible)")


@check("C.8 the guard judges the topology the DATASET holds, and catches a divergent cache")
def c8():
    """The dataset serves motion against a reindexed cond cached in a pickle and trusted on mtime.
    Certifying cond.npy while the model consumes the pickle checks the wrong artifact. The guard
    now takes ds.cond as the authority and cross-checks the source; this test corrupts the
    effective topology and requires the guard to notice."""
    from src.data.anytop_dataset import AnyTopDataset
    from src.data.holdout_guard import guard_dataset
    d = AnyTopDataset(data_root=DR, split="train", num_frames=64, max_joints=144,
                      load_captions=False, splits_dir=SPLITS)
    guard_dataset(d, data_root=DR, artifact=ART, stage="regression", log=lambda *a: None)

    victim = sorted(d.cond)[0]
    par = np.asarray(d.cond[victim]["parents"]).ravel().astype(int).copy()
    if len(par) < 4:
        raise AssertionError("victim skeleton too small to perturb")
    for j in range(2, len(par)):          # re-parent one joint to the root
        if par[j] not in (-1, 0):
            par[j] = 0
            break
    else:
        raise AssertionError("could not perturb the victim topology")
    orig = d.cond[victim]["parents"]
    d.cond[victim]["parents"] = par
    try:
        guard_dataset(d, data_root=DR, artifact=ART, stage="regression", log=lambda *a: None)
    except SystemExit as e:
        if "effective topology disagrees" not in str(e):
            raise AssertionError(f"refused for the wrong reason: {e}")
        return f"a divergent effective topology on {victim} is caught"
    finally:
        d.cond[victim]["parents"] = orig
    raise AssertionError("a divergent effective topology was NOT caught — the guard is still "
                         "certifying the nominal source rather than what the model consumes")


# ---------------------------------------------------------------- C: semantics is causal

@check("C.5 joint semantics CHANGES the tokenizer output (the two arms are not identical)")
def c5():
    """The whole tokenizer-level ablation rests on the two arms differing. Wiring that looks
    right is not evidence: the first attempt fed semantics to GraphMotionVAE while the VQVAE
    trains GraphVQTokenizer, so the arms would have been byte-identical models and 282 GPU-hours
    would have bought two copies of one run. This test is what makes that impossible to repeat."""
    from src.data.anytop_dataset import AnyTopDataset, collate_fn
    from src.models.graph_salad.batch import GraphMotionBatch
    from src.models.vq_model import GraphVQTokenizer
    d = AnyTopDataset(data_root=DR, split="val", num_frames=64, max_joints=144,
                      load_captions=False, joint_semantics=SEM, random_crop=False)
    b = GraphMotionBatch.from_collate_dict(collate_fn([d[i] for i in range(2)]))
    dim = int(b.joint_semantics.shape[-1])

    # ONE tokenizer, two forwards. Building two with different semantic_dim changes the
    # parameter shapes and therefore the RNG draw, so their outputs differ for a trivial reason
    # and the check passes without the feature working at all — which is exactly what happened
    # on the first attempt.
    torch.manual_seed(0)
    on = GraphVQTokenizer(d_model=128, n_heads=4, d_ff=256, n_graph_layers=2,
                          n_enc_temporal_layers=1, max_coarse=32, code_dim=128,
                          num_codes=256, num_quantizers=2, semantic_dim=dim).eval()
    if not on.encoder.use_clip:
        raise AssertionError("semantic_dim>0 but the encoder's projection is still switched off")
    kw = dict(adjacency=b.adjacency, geodesic_dist=b.geodesic_dist, joint_mask=b.joint_mask,
              name_hashes=b.name_hashes, graph_dist=b.anytop_graph_dist,
              joint_relations=b.anytop_joint_relations)
    with torch.no_grad():
        s_on = on.encoder.encode_skeleton(b.skeleton_features,
                                          clip_embeddings=on._semantics(b), **kw)
        s_off = on.encoder.encode_skeleton(b.skeleton_features, clip_embeddings=None, **kw)
    delta = float((s_on - s_off).abs().max())
    if delta < 1e-6:
        raise AssertionError(
            "the semantic arm produces the SAME skeleton embedding as the control — the two "
            "ablation arms would be one run")

    # Permuting the semantics across joints must also change the output: identical output under
    # permutation would mean only a global bias is being read, not per-joint identity.
    with torch.no_grad():
        g = torch.Generator().manual_seed(1)
        perm = b.joint_semantics[:, torch.randperm(b.joint_semantics.shape[1], generator=g)]
        s_perm = on.encoder.encode_skeleton(b.skeleton_features, clip_embeddings=perm, **kw)
    dperm = float((s_on - s_perm).abs().max())
    if dperm < 1e-6:
        raise AssertionError("permuting semantics across joints changes nothing — per-joint "
                             "identity is not being used")

    # And the projection must actually receive gradient.
    on.train()
    out = on.encoder.encode_skeleton(b.skeleton_features,
                                     clip_embeddings=b.joint_semantics, **kw)
    out.sum().backward()
    g = on.encoder.clip_proj[0].weight.grad
    if g is None or not torch.isfinite(g).all() or float(g.abs().max()) == 0.0:
        raise AssertionError(f"semantic projection gradient is {'None' if g is None else 'zero/non-finite'}")
    return (f"dim {dim}; on-vs-off {delta:.3e}, permuted {dperm:.3e}, "
            f"grad max {float(g.abs().max()):.3e}")


@check("C.6 a wrong-dimension semantic table is refused, not silently projected")
def c6():
    from src.models.vq_model import GraphVQTokenizer
    tok = GraphVQTokenizer(d_model=64, n_heads=4, d_ff=128, n_graph_layers=1,
                           n_enc_temporal_layers=1, max_coarse=16, code_dim=64,
                           num_codes=64, num_quantizers=2, semantic_dim=4096)
    class B:
        joint_semantics = torch.zeros(2, 10, 768)
    try:
        tok._semantics(B())
    except ValueError as e:
        if "dimension" not in str(e):
            raise AssertionError(f"refused for the wrong reason: {e}")
        return "768-D table into a 4096-D tokenizer is refused"
    raise AssertionError("a wrongly-sized table was accepted")


@check("C.7 semantic_dim>0 with no table in the batch fails loudly")
def c7():
    from src.models.vq_model import GraphVQTokenizer
    tok = GraphVQTokenizer(d_model=64, n_heads=4, d_ff=128, n_graph_layers=1,
                           n_enc_temporal_layers=1, max_coarse=16, code_dim=64,
                           num_codes=64, num_quantizers=2, semantic_dim=4096)
    class B:
        joint_semantics = None
    try:
        tok._semantics(B())
    except ValueError:
        return "a semantic arm cannot silently degrade into the control"
    raise AssertionError("a missing table was accepted — the arm would silently be the control")


@check("C.9 a token payload whose recorded topology differs from its parents is rejected")
def c9():
    """The loader used to trust `payload_canonical_sha256`. A stored hash proves only that
    someone wrote a hash, so it is now recomputed from the parents the file actually saves."""
    import tempfile, os, json as _json, hashlib as _hl
    from src.models.CodeFlow_Model.token_dataset import TokenCacheDataset
    from src.data.holdout_guard import canonical_form as _cf
    src_dir = Path("data/codeflow_tokens_v4b272neutral_n8192_ep219_fulllen300/val")
    src = sorted(src_dir.glob("*.npz"))[0]
    row = _json.loads((src_dir / "index.jsonl").read_text().splitlines()[0])
    z = dict(np.load(src, allow_pickle=False))
    par = tuple(int(x) for x in np.asarray(z["parent_indices"]).ravel())
    z["payload_motion_id"] = np.array(row["motion_id"])
    z["payload_object_type"] = np.array(str(row["object_type"]))
    z["payload_canonical_sha256"] = np.array("0" * 64)          # a hash that does not match
    tmp = Path(tempfile.mkdtemp()) / "val"
    tmp.mkdir(parents=True)
    np.savez(tmp / "000000.npz", **z)
    (tmp / "index.jsonl").write_text(_json.dumps({**row, "file": "000000.npz"}) + "\n")
    try:
        ds = TokenCacheDataset(str(tmp.parent), "val")
        ds[0]
    except RuntimeError as e:
        if "not the topology in the file" not in str(e):
            raise AssertionError(f"rejected for the wrong reason: {e}")
        real = _hl.sha256(_cf(par).encode()).hexdigest()
        return f"a forged canonical hash is caught (real {real[:12]})"
    finally:
        import shutil; shutil.rmtree(tmp.parent, ignore_errors=True)
    raise AssertionError("a payload whose recorded topology contradicts its parents was accepted")


@check("C.14 a self-consistent payload carrying the WRONG topology is rejected as unauthoritative")
def c14():
    """Recomputing the canonical form from the saved parents catches a forged hash but not a
    payload that carries a consistent DIFFERENT tree: a reviewer built one with the correct motion
    and object identity, a valid alternative parent tree and a matching recomputed hash, and both
    the loader and the guard accepted it. Self-consistency is not authority, so the payload's
    topology must also equal the one the corpus defines for that object type."""
    import tempfile, json as _json, hashlib as _hl, shutil
    from src.models.CodeFlow_Model.token_dataset import TokenCacheDataset
    from src.data.holdout_guard import canonical_form as _cf
    src_dir = Path("data/codeflow_tokens_v4b272neutral_n8192_ep219_fulllen300/val")
    src = sorted(src_dir.glob("*.npz"))[0]
    row = _json.loads((src_dir / "index.jsonl").read_text().splitlines()[0])
    z = dict(np.load(src, allow_pickle=False))
    par = np.asarray(z["parent_indices"]).ravel().astype(np.int64)
    # A DIFFERENT but perfectly valid FK-ordered tree: re-attach the last joint to the root.
    forged = par.copy()
    j = int(np.max(np.nonzero(forged >= 0)[0]))
    if forged[j] == 0:
        forged[j] = 1 if j > 1 else 0
    else:
        forged[j] = 0
    if np.array_equal(forged, par):
        raise AssertionError("could not construct a differing valid tree for this rig")
    z["parent_indices"] = forged
    z["payload_motion_id"] = np.array(row["motion_id"])
    z["payload_object_type"] = np.array(str(row["object_type"]))
    # ...and a hash that MATCHES the forged tree, so the self-consistency check passes.
    z["payload_canonical_sha256"] = np.array(
        _hl.sha256(_cf(tuple(int(x) for x in forged)).encode()).hexdigest())
    tmp = Path(tempfile.mkdtemp()) / "val"
    tmp.mkdir(parents=True)
    np.savez(tmp / "000000.npz", **z)
    (tmp / "index.jsonl").write_text(_json.dumps({**row, "file": "000000.npz"}) + "\n")
    try:
        TokenCacheDataset(str(tmp.parent), "val")[0]            # no authority -> self-consistent
        ds = TokenCacheDataset(str(tmp.parent), "val", authority_root=DR)
        ds[0]
    except RuntimeError as e:
        if "not the topology the corpus defines" not in str(e):
            raise AssertionError(f"rejected for the wrong reason: {e}")
        return f"a self-consistent but unauthoritative tree is caught on {row['object_type']!r}"
    finally:
        shutil.rmtree(tmp.parent, ignore_errors=True)
    raise AssertionError("a payload carrying a different valid topology was accepted")


@check("C.10 a real two-split two-shard merge passes, and a divergent shard is caught")
def c10():
    """The shard-contract check initially looked for sidecars at the output root instead of
    inside each split directory, so it raised on every genuine multi-shard merge — a guard that
    fails on the normal case is worse than none. This exercises the actual layout."""
    import tempfile, json as _json, shutil, importlib.util
    spec = importlib.util.spec_from_file_location("_mes", "scripts/merge_export_shards.py")
    mes = importlib.util.module_from_spec(spec); spec.loader.exec_module(mes)
    root = Path(tempfile.mkdtemp())
    try:
        sd0 = {k: f"v_{k}" for k in mes.STATIC_MANIFEST_KEYS}
        for split in ("train", "val"):
            (root / split).mkdir(parents=True)
            for sh in range(2):
                (root / split / f"manifest_shard{sh:03d}.json").write_text(_json.dumps(sd0))
        n = mes.check_shard_contracts(root, ("train", "val"), 2, sd0)
        if n != 4:
            raise AssertionError(f"checked {n} sidecars, expected 4")
        bad = dict(sd0, frozen_vqvae_ckpt="a_different_tokenizer")
        (root / "val" / "manifest_shard001.json").write_text(_json.dumps(bad))
        try:
            mes.check_shard_contracts(root, ("train", "val"), 2, sd0)
        except RuntimeError as e:
            if "frozen_vqvae_ckpt" not in str(e):
                raise AssertionError(f"caught the wrong difference: {e}")
            return "4 sidecars across 2 splits verified; a divergent shard is caught"
        raise AssertionError("a shard from a different tokenizer was accepted")
    finally:
        shutil.rmtree(root, ignore_errors=True)


@check("C.11 the two arms share byte-identical initial weights; only one config field differs")
def c11():
    """Sizing the semantic projection from the table changes how many numbers the initialiser
    draws, so an arm built with semantic_dim=0 and one built with 4096 disagree on the initial
    value of every module constructed afterwards. A one-seed comparison between them would be
    confounded by unrelated initial weights rather than by semantics."""
    import torch
    from src.models.vq_model import GraphVQTokenizer
    K = dict(d_model=128, n_heads=4, d_ff=256, n_graph_layers=2, n_enc_temporal_layers=1,
             n_pre_vq_layers=1, n_post_vq_layers=1, n_cross_layers=1, n_dec_temporal_layers=1,
             code_dim=128, num_codes=64, num_quantizers=2, max_coarse=8, temporal_stride=4)

    def build(**kw):
        torch.manual_seed(42)
        return GraphVQTokenizer(**K, **kw)

    def n_differing(a, b):
        sa, sb = a.state_dict(), b.state_dict()
        common = [k for k in sa if k in sb and sa[k].shape == sb[k].shape]
        return len(common), sum(1 for k in common if not torch.equal(sa[k], sb[k]))

    _, bad = n_differing(build(semantic_dim=0), build(semantic_dim=4096))
    if bad == 0:
        raise AssertionError("the old sizing was expected to confound initialisation; it did not, "
                             "so this test no longer measures what it was written for")
    n, diff = n_differing(build(semantic_dim=4096, semantic_enabled=False),
                          build(semantic_dim=4096, semantic_enabled=True))
    if diff:
        raise AssertionError(f"{diff}/{n} common tensors still differ between the arms")
    on = build(semantic_dim=4096, semantic_enabled=True)
    off = build(semantic_dim=4096, semantic_enabled=False)
    if not (on.encoder.use_clip and not off.encoder.use_clip):
        raise AssertionError("semantic_enabled does not gate the projection")
    return f"old sizing confounded {bad} tensors; both arms now share all {n}"


@check("C.12 a semantic checkpoint strict-loads through the downstream exporter")
def c12():
    """The semantic arm could be trained but not used: every downstream loader rebuilt the
    tokenizer with the default 768-D projection, so strict-loading a 4096-D semantic checkpoint
    failed outright. Discovering that after a multi-day run would waste the run."""
    import torch, tempfile, shutil, importlib.util
    from src.models.vq_model import GraphVQTokenizer
    spec = importlib.util.spec_from_file_location("_exp", "scripts/export_graph_vq_tokens.py")
    exp = importlib.util.module_from_spec(spec); spec.loader.exec_module(exp)
    K = dict(d_model=128, n_heads=4, d_ff=256, n_graph_layers=2, n_enc_temporal_layers=1,
             n_pre_vq_layers=1, n_post_vq_layers=1, n_cross_layers=1, n_dec_temporal_layers=1,
             code_dim=128, num_codes=64, num_quantizers=2, max_coarse=8, temporal_stride=4,
             temporal_kernel=9, dropout=0.1, ema_mu=0.99, quantize_dropout_prob=0.1,
             dead_code_threshold=1.0)
    tmp = Path(tempfile.mkdtemp())
    try:
        out = []
        for sem_dim, enabled, label in ((4096, True, "semantic"), (4096, False, "control"),
                                        (0, False, "legacy")):
            kw = {} if sem_dim == 0 else {"semantic_dim": sem_dim, "semantic_enabled": enabled}
            m = GraphVQTokenizer(**K, **kw)
            ta = dict(K)
            ta["joint_semantics"] = None if sem_dim == 0 else "data/joint_semantics_llm2vec_v1.npz"
            ta["semantic_enabled"] = enabled
            f = tmp / f"{label}.pt"
            torch.save({"model_state_dict": m.state_dict(), "args": ta}, f)
            back, _, _ = exp.load_frozen_tokenizer(str(f), torch.device("cpu"))
            if back.semantic_dim != sem_dim or back.encoder.use_clip != (sem_dim > 0 and enabled):
                raise AssertionError(f"{label}: reloaded as semantic_dim={back.semantic_dim} "
                                     f"use_clip={back.encoder.use_clip}, expected "
                                     f"{sem_dim}/{sem_dim > 0 and enabled}")
            out.append(label)
        return "strict reload OK for " + ", ".join(out)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


@check("C.13 corrupting ANY parent-derived graph tensor is refused, field by field")
def c13():
    """The model does not attend over `parents`; it attends over adjacency, graph distance and
    joint relations, and batch validation checks those only for shape, dtype and finiteness. A
    reviewer zeroed one retained rig's `joint_relations` and the guard accepted it, so the
    certified topology and the consumed topology were different objects."""
    import numpy as np
    from src.data.anytop_dataset import AnyTopDataset
    from src.data.holdout_guard import guard_dataset
    d = AnyTopDataset(data_root=DR, split="train", num_frames=64, max_joints=144,
                      load_captions=False, splits_dir=SPLITS)
    guard_dataset(d, data_root=DR, artifact=ART, stage="c13-clean", log=lambda *a: None)
    fields = ("adjacency", "geodesic_dist", "joint_relations", "joints_graph_dist",
              "name_hashes", "skeleton_features")
    tgt = next(o for o in sorted(d.cond)
               if isinstance(d.cond[o], dict) and all(f in d.cond[o] for f in fields))
    missed = []
    for f in fields:
        keep = d.cond[tgt][f]
        d.cond[tgt][f] = np.zeros_like(np.asarray(keep))
        try:
            guard_dataset(d, data_root=DR, artifact=ART, stage="c13", log=lambda *a: None)
            missed.append(f)
        except SystemExit:
            pass
        finally:
            d.cond[tgt][f] = keep
    if missed:
        raise AssertionError(f"corruption of {missed} on {tgt!r} was accepted by the guard")
    guard_dataset(d, data_root=DR, artifact=ART, stage="c13-restored", log=lambda *a: None)
    return f"all {len(fields)} derived fields refused when corrupted on {tgt!r}; clean state passes"


# ---------------------------------------------------------------- D: provenance

@check("D.1 provenance accepts and refuses the right upstreams")
def d1():
    from src.data import provenance as prov
    good = prov.stamp(protocol=prov.PROTOCOL_UNSEEN, stage="t", holdout_artifact=ART)
    legacy = prov.stamp(protocol=prov.PROTOCOL_LEGACY, stage="t")
    other = dict(good, holdout_artifact_body_sha256="0" * 64)
    sha = good["holdout_artifact_body_sha256"]
    cases = [(None, prov.PROTOCOL_LEGACY, None, True),
             (None, prov.PROTOCOL_UNSEEN, None, False),
             (legacy, prov.PROTOCOL_UNSEEN, None, False),
             (other, prov.PROTOCOL_UNSEEN, sha, False),
             (good, prov.PROTOCOL_UNSEEN, sha, True)]
    for p, proto, exp, want_pass in cases:
        try:
            prov.verify_upstream(p, protocol=proto, what="t", expect_artifact_body_sha=exp,
                                 log=lambda *a: None)
            got = True
        except SystemExit:
            got = False
        if got != want_pass:
            raise AssertionError(f"case {prov.summarise(p)} under {proto}: expected "
                                 f"{'pass' if want_pass else 'refuse'}")
    return f"{len(cases)}/{len(cases)} cases"


@check("D.2 the artifact BODY hash and FILE hash are distinct and used consistently")
def d2():
    """These were both called 'the artifact sha'. The launcher passed the file hash where the
    body hash was expected, so every strict run would have aborted at startup. This pins the
    distinction so the confusion cannot come back."""
    from src.data import provenance as prov
    from src.data.holdout_guard import verify_artifact
    body = prov.artifact_body_sha256(ART)
    file_ = prov.sha256_file(ART)
    if body == file_:
        raise AssertionError("body and file hash are equal — this test can no longer detect the "
                             "confusion it exists for")
    verify_artifact(ART, DR, expect_body_sha=body)          # must pass
    try:
        verify_artifact(ART, DR, expect_body_sha=file_)      # must refuse
    except SystemExit:
        pass
    else:
        raise AssertionError("the FILE hash was accepted where the BODY hash is required")
    st = prov.stamp(protocol=prov.PROTOCOL_UNSEEN, stage="t", holdout_artifact=ART)
    if st["holdout_artifact_body_sha256"] != body or st["holdout_artifact_file_sha256"] != file_:
        raise AssertionError("the stamp records the two hashes under the wrong names")
    return f"body {body[:12]} != file {file_[:12]}, both recorded"


@check("D.3 stamp() refuses to let extra= overwrite a reserved field")
def d3():
    from src.data import provenance as prov
    try:
        prov.stamp(protocol=prov.PROTOCOL_LEGACY, stage="t", extra={"protocol": "forged"})
    except ValueError as e:
        if "reserved" not in str(e):
            raise AssertionError(f"refused for the wrong reason: {e}")
        return "a caller cannot forge protocol through extra="
    raise AssertionError("extra= overwrote a reserved field")


# ---------------------------------------------------------------- E: moment round trip

@check("E.1 normalise/de-normalise recovers physical units under every moment policy")
def e1():
    from src.data.anytop_dataset import AnyTopDataset, _STD_FLOOR
    from src.data.moment_source import MomentSource
    k = dict(data_root=DR, split="val", num_frames=64, max_joints=144, load_captions=False)
    ref = AnyTopDataset(**k)
    worst = 0.0
    for name, ms in (("own", MomentSource("own")),
                     ("estimated", MomentSource("estimated", estimator_path=EST))):
        d = AnyTopDataset(**k, moment_source=ms)
        for i in (0, 500, 2000):
            a, b = ref[i], d[i]
            J = int(np.asarray(b["joint_mask"]).sum())
            T = int(np.asarray(b["frame_mask"]).sum())
            xb = np.asarray(b["anytop_x"]).transpose(2, 0, 1)[:T, :J]
            mb, sb = np.asarray(b["anytop_mean"])[:J], np.asarray(b["anytop_std"])[:J]
            xa = np.asarray(a["anytop_x"]).transpose(2, 0, 1)[:T, :J]
            ma, sa = np.asarray(a["anytop_mean"])[:J], np.asarray(a["anytop_std"])[:J]
            err = np.abs((xb * (sb + _STD_FLOOR) + mb) - (xa * (sa + _STD_FLOOR) + ma)).max()
            worst = max(worst, float(err))
            if err > 1e-3:
                raise AssertionError(f"{name}[{i}]: round trip error {err:.4f} in physical units")
    return f"max error {worst:.2e} across policies"


# ---------------------------------------------------------------- F: seals

@check("F.1 protected baseline artifacts unchanged")
def f1():
    r = subprocess.run([sys.executable, "scripts/_protect_baseline.py", "--verify"],
                       capture_output=True, text=True)
    if r.returncode != 0:
        raise AssertionError(r.stdout.strip()[-300:])
    return r.stdout.strip().splitlines()[-1]


@check("F.2 sealed protocol DATA artifacts unchanged")
def f2():
    # Data drift is fatal here; builder-code drift is not, because the code is what is being
    # finalised. The launcher runs the same check with --strict-code, which makes code drift
    # fatal too: at launch the protocol must be sealed, not merely intact.
    r = subprocess.run([sys.executable, "scripts/_seal_protocol.py", "--verify"],
                       capture_output=True, text=True)
    if r.returncode != 0:
        raise AssertionError(r.stdout.strip()[-300:])
    return r.stdout.strip().splitlines()[-1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None, help="comma-separated groups, e.g. A,C")
    args = ap.parse_args()
    groups = set(args.only.split(",")) if args.only else None

    tests = [v for k, v in sorted(globals().items())
             if callable(v) and hasattr(v, "_group")]
    for t in tests:
        if groups and t._group not in groups:
            continue
        t()

    width = max(len(n) for n, _, _ in _results) if _results else 10
    print()
    for name, ok, msg in _results:
        print(f"  {'PASS' if ok else 'FAIL'}  {name:<{width}}  {msg}")
    n_fail = sum(1 for _, ok, _ in _results if not ok)
    print(f"\n{len(_results) - n_fail}/{len(_results)} passed")
    if n_fail:
        print("A multi-day run must not start with a failing regression.")
    return 1 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
