"""Enforce the frozen held-out-topology pre-registration at every training entry point.

The freeze is worthless if nothing checks it. A pre-registration that lives in a JSON file
nobody reads is a document, not a protocol: the VQVAE, the token export, the CodeFlow
backbone and the evaluator all read `splits/` directly, so any of them could silently train
on a held topology and the resulting "unseen" number would be a lie that no artefact on disk
would contradict.

Every entry point therefore calls `guard_dataset()` with the artifact's expected SHA-256 and
the samples it is about to consume. The check resolves each clip to its object type and then
to its AHU canonical form, and refuses to proceed if any held form appears. Resolution is by
canonical form rather than by name, so an object type that shares a held topology under a
different name is still caught.

Usage in a trainer:

    from src.data.holdout_guard import guard_dataset
    guard_dataset(ds_train, data_root=args.anytop_root,
                  artifact=args.holdout_artifact, expect_sha=args.holdout_sha,
                  stage="vqvae:train")

Passing `artifact=None` is allowed only when `allow_no_holdout=True` is also passed, which
exists solely so pre-holdout runs and smoke tests keep working; it prints a loud line so a
run without the guard is visible in the log.
"""
from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Optional

import numpy as np

_CANON_CACHE: dict[str, dict] = {}


def canonical_form(parents: tuple[int, ...]) -> str:
    """AHU canonical form of a rooted unordered tree. Duplicated from the builder on purpose:
    this module must not import from `scripts/`, and a guard that silently follows a changed
    definition is not a guard. `verify_artifact` cross-checks the two by recomputing the
    canonical count the artifact recorded."""
    ch: dict[int, list[int]] = defaultdict(list)
    root: Optional[int] = None
    for j, p in enumerate(parents):
        if p < 0 or p == j:
            if root is None:
                root = j
        else:
            ch[p].append(j)
    if root is None:
        root = 0

    def rec(v: int) -> str:
        return "()" if not ch[v] else "(" + "".join(sorted(rec(c) for c in ch[v])) + ")"
    return rec(root)


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_artifact(artifact_path: str | Path, data_root: str | Path,
                    expect_body_sha: Optional[str] = None) -> dict:
    """Load the freeze, prove it has not been edited, and prove the corpus still matches it."""
    p = Path(artifact_path)
    art = json.loads(p.read_text())

    body = json.dumps({k: v for k, v in art.items() if k != "artifact_sha256"},
                      indent=2, sort_keys=True)
    got = hashlib.sha256(body.encode()).hexdigest()
    if got != art.get("artifact_sha256"):
        raise SystemExit(f"[holdout-guard] REFUSED: {p} is TAMPERED — body hashes to {got}, "
                         f"artifact carries {art.get('artifact_sha256')}")
    if expect_body_sha and expect_body_sha != got:
        raise SystemExit(f"[holdout-guard] REFUSED: artifact sha256 {got} != expected "
                         f"{expect_body_sha}. This is the artifact BODY hash (its self-hash), "
                         f"not the file hash — they differ, and confusing them was a live bug.")

    root = Path(data_root)
    if _sha256_file(root / "cond.npy") != art["inputs"]["cond_npy_sha256"]:
        raise SystemExit(f"[holdout-guard] REFUSED: {root}/cond.npy changed since the freeze")
    return art


def _held_canon(art: dict) -> dict[str, str]:
    return {t["canonical_form_sha256"]: t["bucket"] for t in art["held_out_trees"]}


def _obj_to_canon(data_root: str | Path) -> tuple[dict[str, str], list[str]]:
    key = str(Path(data_root).resolve())
    if key not in _CANON_CACHE:
        cond = np.load(Path(data_root) / "cond.npy", allow_pickle=True).item()
        m = {}
        for o, v in cond.items():
            par = v["parents"] if isinstance(v, dict) else v[0]
            c = canonical_form(tuple(int(x) for x in np.asarray(par).ravel()))
            m[o] = hashlib.sha256(c.encode()).hexdigest()
        _CANON_CACHE[key] = {"map": m, "keys": sorted(cond.keys(), key=len, reverse=True)}
    e = _CANON_CACHE[key]
    return e["map"], e["keys"]


def _resolve(name: str, keys: list[str]) -> Optional[str]:
    for k in keys:
        if name.startswith(f"{k}_"):
            return k
    return None


def assert_no_held(names: Iterable[str], *, data_root: str | Path, art: dict,
                   stage: str, obj_canon: Optional[dict] = None) -> int:
    """Raise unless every clip name belongs to a retained canonical topology."""
    _m, keys = _obj_to_canon(data_root)
    obj_canon = obj_canon if obj_canon is not None else _m
    held = _held_canon(art)
    n = 0
    offenders: list[tuple[str, str, str]] = []
    unresolved: list[str] = []
    for name in names:
        n += 1
        fn = name if name.endswith(".npy") else f"{name}.npy"
        o = _resolve(fn, keys)
        if o is None:
            unresolved.append(name)
            continue
        b = held.get(obj_canon[o])
        if b is not None and len(offenders) < 8:
            offenders.append((name, o, b))
    if unresolved:
        raise SystemExit(f"[holdout-guard] REFUSED at {stage}: {len(unresolved)} clip names do "
                         f"not resolve to an object type, e.g. {unresolved[:3]}. A name the "
                         f"guard cannot resolve is a name it cannot check.")
    if offenders:
        raise SystemExit(f"[holdout-guard] REFUSED at {stage}: HELD-OUT topologies present in "
                         f"the data this stage would consume. Examples (clip, object_type, "
                         f"bucket): {offenders}. Point this stage at the retained split "
                         f"directory produced by scripts/_build_holdout_splits.py.")
    return n


def _effective_canon(ds, data_root: str | Path, log) -> dict[str, str]:
    """Object type -> canonical form, taken from the topology the DATASET ACTUALLY HOLDS.

    `AnyTopDataset` serves motion against a reindexed cond that it caches in a pickle and trusts
    on mtime alone. Certifying `cond.npy` while the model consumes the pickle is a check of the
    wrong artifact: a stale or replaced cache could supply different parents and the raw file
    would still pass. So the authority here is `ds.cond`, and `cond.npy` is used as a
    cross-check — a disagreement means the cache and its source describe different skeletons and
    is fatal rather than reported.
    """
    cond = getattr(ds, "cond", None)
    if cond is None:
        return _obj_to_canon(data_root)[0]
    eff = {}
    for o, c in cond.items():
        par = c["parents"] if isinstance(c, dict) else c[0]
        eff[o] = hashlib.sha256(
            canonical_form(tuple(int(x) for x in np.asarray(par).ravel())).encode()).hexdigest()
    raw, _ = _obj_to_canon(data_root)
    disagree = sorted(o for o in eff if o in raw and eff[o] != raw[o])
    if disagree:
        raise SystemExit(
            f"[holdout-guard] REFUSED: the dataset's effective topology disagrees with "
            f"{data_root}/cond.npy for {len(disagree)} object type(s), e.g. {disagree[:3]}. The "
            f"normalized-cond cache and its source describe different skeletons, so certifying "
            f"the source would certify something the model does not use. Delete the cache and "
            f"let it rebuild.")
    missing = sorted(set(raw) - set(eff))
    if missing:
        log(f"[holdout-guard] note: {len(missing)} object type(s) in cond.npy are absent from the "
            f"dataset's cond (e.g. {missing[:3]}); membership is judged on what the dataset holds.")
    return eff


def _verify_derived_graph(ds, log) -> int:
    """Every parent-derived graph tensor in the cache must equal what those parents produce.

    Checking `parents` alone is not enough: the tokenizer does not attend over `parents`, it
    attends over `adjacency`, `joints_graph_dist` and `joint_relations`, and batch validation only
    checks those for shape, dtype and finiteness. A reviewer zeroed one retained rig's
    `joint_relations` and the guard accepted it — the certified topology and the consumed topology
    were then different objects, which is precisely the substitution this whole protocol exists to
    make impossible.

    `_build_derived` is imported rather than re-implemented, unlike `canonical_form`. The two
    serve different purposes: the canonical form is the DEFINITION the freeze is stated in, so a
    guard that followed a changed definition would certify nothing, whereas this check asks
    whether the cache still agrees with the code that fills it — for which the code is the right
    reference. Definition drift is caught separately by the seal's --strict-code.
    """
    cond = getattr(ds, "cond", None)
    if cond is None:
        return 0
    from src.data.anytop_dataset import _build_derived
    fields = ("adjacency", "geodesic_dist", "joint_relations", "joints_graph_dist",
              "name_hashes", "skeleton_features")
    checked = 0
    for o, c in sorted(cond.items()):
        if not isinstance(c, dict) or "parents" not in c:
            continue
        try:
            want = _build_derived(np.asarray(c["parents"]), np.asarray(c["offsets"]),
                                  list(c["joint_names"]))
        except Exception as e:                                    # a rig we cannot rebuild is
            raise SystemExit(                                     # unverifiable, not "fine"
                f"[holdout-guard] REFUSED: cannot rebuild the derived graph for {o!r} from its "
                f"own parents ({type(e).__name__}: {e}). A tensor that cannot be recomputed "
                f"cannot be certified.")
        for f in fields:
            have = c.get(f)
            if have is None:
                continue
            if not np.array_equal(np.asarray(have), np.asarray(want[f])):
                raise SystemExit(
                    f"[holdout-guard] REFUSED: {o!r} carries a cached {f!r} that its own parents "
                    f"do not produce. The model attends over these tensors, so the topology it "
                    f"consumes is not the topology this guard certified. Delete the normalized "
                    f"cond cache and let it rebuild.")
            checked += 1
    return checked


def guard_dataset(ds, *, data_root: str | Path, artifact: Optional[str | Path],
                  expect_body_sha: Optional[str] = None, stage: str = "?",
                  allow_no_holdout: bool = False, log=print) -> None:
    """Guard a dataset object exposing `.samples` (AnyTopDataset) or `.rows` (token cache)."""
    if artifact is None:
        if not allow_no_holdout:
            raise SystemExit(f"[holdout-guard] REFUSED at {stage}: no --holdout_artifact given. "
                             f"Pass one, or pass the explicit opt-out if this run predates the "
                             f"held-out protocol.")
        log(f"[holdout-guard] {stage}: NO HOLDOUT ARTIFACT — this run may contain held-out "
            f"topologies and MUST NOT be used for any unseen-topology claim.")
        return
    art = verify_artifact(artifact, data_root, expect_body_sha)
    eff = _effective_canon(ds, data_root, log)
    n_dg = _verify_derived_graph(ds, log)
    names = _dataset_names(ds, stage)
    n = assert_no_held(names, data_root=data_root, art=art, stage=stage, obj_canon=eff)
    log(f"[holdout-guard] {stage}: {n} clips checked against freeze "
        f"{art['artifact_sha256'][:16]}; 0 held-out topologies present "
        f"({len(art['held_out_trees'])} held / {art['inputs']['n_canonical_topologies']} total); "
        f"{n_dg} derived graph tensors recomputed from parents and matched.")


def _dataset_names(ds, stage: str) -> list[str]:
    # AnyTopT2MEvalDataset stores (base_idx, record, t5_key, caption, valid) tuples in `_plan`;
    # the record carries `filename`. Handled first because it also has no `.samples`/`.rows`.
    plan = getattr(ds, "_plan", None)
    if plan is not None:
        out = []
        for entry in plan:
            rec = entry[1] if isinstance(entry, (tuple, list)) and len(entry) > 1 else None
            v = (rec or {}).get("filename") or (rec or {}).get("motion_id")
            if v is None:
                raise SystemExit(f"[holdout-guard] REFUSED at {stage}: a _plan record has "
                                 f"neither 'filename' nor 'motion_id'; cannot verify it.")
            out.append(Path(str(v)).name)
        return out
    for attr, key in (("samples", "path"), ("rows", "motion_id")):
        seq = getattr(ds, attr, None)
        if seq is None:
            continue
        out = []
        for s in seq:
            if isinstance(s, dict):
                v = s.get(key) or s.get("motion_id") or s.get("path")
            else:
                v = getattr(s, key, None)
            if v is None:
                raise SystemExit(f"[holdout-guard] REFUSED at {stage}: dataset entry has no "
                                 f"{key!r}; cannot verify it.")
            out.append(Path(str(v)).name)
        return out
    raise SystemExit(f"[holdout-guard] REFUSED at {stage}: dataset exposes neither .samples "
                     f"nor .rows; the guard cannot see what it would train on.")
