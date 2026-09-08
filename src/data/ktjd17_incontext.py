"""KTJD-17 -> in-context pipeline adapter.

Ktjd17Base exposes the SAME item surface AnyTopDataset gives InContextPairs (anytop_x layout
[J,C,T], num_joints/num_frames, caption_emb, joint_semantics with order-hash check, geodesic,
anytop_mean/std, parent_indices, rest_offsets), so InContextPairs/collate are reused with ZERO
changes. KTJD specifics ride along without touching the pairs code:

  - C is 18, not 17: plane 17 is the per-frame HEADING-VALID flag broadcast over joints, so the
    demo/target cropping machinery carries it for free. The trainer slices x[..., :17] for the
    model and reads the flag from x[..., 0, 17]. (heading_valid is time-dependent; every other
    mask is static per rig and is fetched per object_type at batch time, no dataset plumbing.)
  - normalization is KTJD's s_rig + frozen train-only block gains (loader.normalize_model_motion,
    applied inside build_model_view), THEN REST-CENTERED (see rest_centering; 2026-08-20
    frozen-pose fix): anytop_mean is the rig's rest frame in raw units and anytop_std is the
    EXACT per-cell de-normalization scale minus _STD_FLOOR, so the existing
    `x * (std + _STD_FLOOR) + mean` consumers (renderer/diag/gamma_fk packs) reproduce raw KTJD
    values to float32 round-off (~2e-6 relative; NOT bit-exact -- codex round-S0 measured the
    bound). Everything the model sees is a deviation from rest.
  - static per-rig masks (channel_valid [J,17] combining root-only ch13:17, fixed_dof rotation
    rows, contact supervision) are exposed via `static_masks(object_type)`.
  - velocity channels ch9:12 are supervision-only in KTJD (decode NEVER integrates them); root
    recovery is direct (ch13:15 smooth-root + origin_xz), enforced by using the official codec
    for any decode -- this adapter does not decode.

Split names come from the manifest's own `split` column (ktjd17_split_names), not from a
splits-dir file format assumption.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from src.data.anytop_dataset import _STD_FLOOR
from src.data.caption_keys import ordered_captions
from src.data.ktjd17.loader import build_model_view, load_motion_npz

HEADING_PLANE = 17          # x[..., 17] = heading-valid flag; model input is x[..., :17]
TARGET_CENTERING = "percell_meanstd_v3_masked_floor"   # see Ktjd17Base._percell; bump on any change


def load_exclusions(path: str | Path | None) -> set[str]:
    """clip_ids removed from every split by an exclusion artifact (teleport cut, 2026-08-21).

    The corpus is frozen and sha-pinned, so a rejected clip cannot be deleted from it. The cut is
    a separate hashable artifact applied at load time, and both the trainer and the renderer pin
    its sha, so a checkpoint always names the exact data it was allowed to see.
    """
    if not path:
        return set()
    d = json.loads(Path(path).read_text())
    clips = d["clips"]
    bad = sorted(c for c, v in clips.items() if v != "all")
    if bad:
        raise ValueError(f"frame-level exclusions are not supported by this loader "
                         f"({len(bad)} entries, e.g. {bad[:2]}); use mode=clip")
    return set(clips)


def ktjd17_split_names(root: str | Path, exclude: str | Path | None = None
                       ) -> dict[str, set[str]]:
    """{'train'|'val'|'held_representative'|'held_stress': {clip_id,...}} from clips.jsonl."""
    drop = load_exclusions(exclude)
    # train/val always present (possibly empty): an all-train cut (user 2026-09-03) leaves a rig
    # with no val clip, and every consumer indexes names["val"]
    out: dict[str, set[str]] = {"train": set(), "val": set()}
    for line in open(Path(root) / "manifests" / "clips.jsonl"):
        row = json.loads(line)
        if row.get("status") != "accept" or str(row["clip_id"]) in drop:
            continue
        out.setdefault(str(row["split"]), set()).add(str(row["clip_id"]))
    return out


class Ktjd17Base:
    def __init__(self, root: str | Path = "dataset/ktjd17_truebones", *,
                 caption_emb_cache: str | Path,
                 joint_semantics: str | Path,
                 texts_json: str | Path = "motion_texts_by_file_clean_v1.json",
                 percell_stats: str | Path = "data/ktjd17_percell_stats_v1.npz",
                 exclude_clips: str | Path | None = None,
                 random_caption: bool = False,
                 normalization: str = "percell"):
        self.root = Path(root)
        self.random_caption = bool(random_caption)
        # representation ablation (user 2026-09-06): "percell" = per-(rig, joint, channel) mean/std (the method);
        # "scale_only" = the KTJD spec's scale-only normalization (mean 0, std = s_rig/gain per channel family,
        # see _std_eff) -- the OLD representation, served through the same convention raw = x*(std+floor)+mean so
        # every consumer (trainer, calibration, renderer, evaluator conversion) stays unchanged.
        # "rest" (user 2026-09-08, deployability): the scale-only std with the rig's REST frame as the mean -- rest
        # positions relative to the root's XZ, identity rest-delta 6D, zero velocity / contact / root track, heading
        # [1, 0] -- everything derivable from the skeleton file alone, no motion statistics (the 2026-08-20
        # rest-centering fix, _rest_centering_SUPERSEDED, revived as a first-class serving normalization).
        if normalization not in ("percell", "scale_only", "rest"):
            raise ValueError(f"normalization must be 'percell', 'scale_only' or 'rest', got {normalization!r}")
        self.normalization = str(normalization)
        self._pc_so: dict = {}

        # ---- generation identity + frozen schema (codex round-S0: pin the EXACT generation and
        # let the official validator, not this adapter, define schema validity) ----
        from src.data.ktjd17.schema import validate_schema
        gen = json.loads((self.root / "generation.json").read_text())
        self.generation_id = str(gen["generation_id"])
        schema_p = self.root / "schema.json"
        schema = json.loads(schema_p.read_text())
        validate_schema(schema, expected_fps_target=30.0, require_frozen=True)

        gains_p = self.root / "stats" / "train_block_gains.npz"
        gz = np.load(gains_p, allow_pickle=True)
        if not bool(gz["frozen"]):
            raise RuntimeError("KTJD-17 block gains are not frozen; refusing to train against "
                               "an unfrozen calibration")
        # train-only provenance: the gains must come from the train split's own calibration --
        # a val/held-contaminated calibration would leak statistics across the protocol wall.
        if str(gz["split"]) != "train" or not str(gz["calibration_version"]).startswith(
                "ktjd17-train-only"):
            raise RuntimeError(f"gains provenance is not train-only: split={gz['split']!r} "
                               f"version={gz['calibration_version']!r}")
        # The freeze the gains belong to is named at the TOP LEVEL by the TrueBones build and
        # inside `freeze_binding` by the PZ+Human build (which binds explicitly to the same frozen
        # calibration). Accept either, but require one of them to match -- a gains artifact from
        # an unrelated freeze must still be refused.
        want_freeze = str(gen.get("freeze_generation_id")
                          or (gen.get("freeze_binding") or {}).get("generation_id"))
        if str(gz["freeze_generation_id"]) != want_freeze:
            raise RuntimeError(f"gains freeze {gz['freeze_generation_id']} != corpus freeze "
                               f"{want_freeze} -- the stats artifact belongs to a different "
                               f"calibration")
        self.gains = np.asarray(gz["gains"], dtype=np.float64)
        sg = np.asarray(schema["normalization"]["gains"], dtype=np.float64)
        if not np.array_equal(sg, self.gains):
            raise RuntimeError(f"schema gains {sg.tolist()} != gains artifact "
                               f"{self.gains.tolist()} -- frozen schema and stats disagree")

        # ---- OLD-STYLE per-(rig, joint, channel) mean/std (user 2026-08-21) ----
        # KTJD's spec normalization is scale-only, which leaves the rig's static pose in the
        # target: a constant prediction captured 75.0% of the objective (13ch, which standardizes
        # per cell, measures 45.2%). This restores the method that produced coherent motion.
        # The spec's s_rig/gains scaling is NOT applied on top -- per-cell std subsumes it.
        # Two artifact layouts are accepted. The PZ+Human 312-rig build ships ONE stacked array
        # per field plus a supervise_mask (exact-constant cells -- 5,219 of them never-touching
        # contact flags -- are dropped from supervision rather than taught as constant zeros, and
        # non-zero variances below 1e-4 are floored; user's numerical analysis 2026-08-21). The
        # TrueBones build stored one array per rig and no mask.
        self._pc, self._sup = {}, {}
        with np.load(percell_stats, allow_pickle=False) as z:
            self.percell_meta = json.loads(str(z["__meta"]))
            if "rig_ids" in z.files:                       # stacked layout
                rids = [str(r) for r in z["rig_ids"]]
                M, S = np.asarray(z["mean"]), np.asarray(z["std"])
                SUP = np.asarray(z["supervise_mask"])
                JC = np.asarray(z["joint_count"])
                for i, r in enumerate(rids):
                    J = int(JC[i])
                    self._pc[r] = (M[i, :J], S[i, :J])
                    self._sup[r] = SUP[i, :J]
            else:                                          # per-rig layout (TrueBones)
                for k in z.files:
                    if k.startswith("mean__"):
                        r = k[len("mean__"):]
                        self._pc[r] = (np.asarray(z[k]), np.asarray(z["std__" + r]))
        if str(self.percell_meta.get("generation_id")) != self.generation_id:
            raise RuntimeError(f"per-cell stats were built on generation "
                               f"{self.percell_meta.get('generation_id')}, corpus is "
                               f"{self.generation_id}")
        # VALIDATE, do not merely record (codex round-S8 fail-open #3). A stats artifact is the
        # affine map every tensor in the run passes through; a NaN, a non-positive std, a wrong
        # floor or a missing rig silently corrupts everything downstream.
        if float(self.percell_meta.get("std_floor", -1)) != float(_STD_FLOOR):
            raise RuntimeError(f"per-cell stats built with std_floor "
                               f"{self.percell_meta.get('std_floor')}, code uses {_STD_FLOOR}")
        conv = str(self.percell_meta.get("convention", ""))
        if "x * (std + _STD_FLOOR) + mean" not in conv:
            raise RuntimeError(f"per-cell stats declare an unexpected convention: {conv!r}")
        for r_, (mu_, sd_) in self._pc.items():
            if mu_.shape != sd_.shape or mu_.ndim != 2 or mu_.shape[1] != 17:
                raise RuntimeError(f"per-cell stats for {r_!r} have bad shape "
                                   f"{mu_.shape}/{sd_.shape}, expected [J,17]")
            if not (np.isfinite(mu_).all() and np.isfinite(sd_).all()):
                raise RuntimeError(f"per-cell stats for {r_!r} contain non-finite values")
            if float((sd_ + _STD_FLOOR).min()) <= 0.0:
                raise RuntimeError(f"per-cell stats for {r_!r} have a non-positive effective std")

        # A DERIVED TRAINING VIEW (e.g. the per-species LoRA corpus, 2026-09-02) keeps the parent's
        # frozen generation.json but ships its own manifest. It must declare itself: derivation.json
        # names the parent generation and the exact manifest bytes it serves, we verify both and
        # pin them, so a manifest swapped under an unchanged generation cannot pass a launch, a
        # resume or a render (codex 2026-09-02 P0-2).
        self.derivation = None
        deriv_p = self.root / "derivation.json"
        manifest_p = self.root / "manifests" / "clips.jsonl"
        manifest_sha = hashlib.sha256(manifest_p.read_bytes()).hexdigest()
        if deriv_p.is_file():
            self.derivation = json.loads(deriv_p.read_text())
            if str(self.derivation.get("parent_generation_id")) != self.generation_id:
                raise RuntimeError(f"derivation.json names parent generation "
                                   f"{self.derivation.get('parent_generation_id')}, corpus generation.json "
                                   f"says {self.generation_id}")
            if str(self.derivation.get("derived_manifest_sha256")) != manifest_sha:
                raise RuntimeError(f"derived manifest {manifest_p} hashes {manifest_sha[:16]}, derivation.json "
                                   f"pins {str(self.derivation.get('derived_manifest_sha256'))[:16]} -- the "
                                   f"training view was edited after it was declared")
            # the declared inputs must be the ones actually served: parent generation.json copy, the
            # rig table, and -- when this loader is handed them -- the stats / texts / exclusion files
            _want = {
                "generation.json": (self.root / "generation.json", self.derivation.get("parent_generation_json_sha256")),
                "rig_table": (self.root / "splits" / "lora_v1" / "rig_table.json",
                              (self.derivation.get("split") or {}).get("rig_table_sha256")),
                "norm_stats": (Path(percell_stats), (self.derivation.get("norm_stats") or {}).get("sha256")),
                "texts_json": (Path(texts_json), (self.derivation.get("texts_json") or {}).get("sha256")),
            }
            # a view that ships its own joint-semantics table (joint-pruned view, 2026-09-02) must be
            # served with exactly that table; the per-rig order hash alone would accept any table whose
            # names match (codex r7 #5). v1-style derivations that declare none keep the order-hash check only.
            if "joint_semantics" in self.derivation:
                _want["joint_semantics"] = (Path(joint_semantics), (self.derivation.get("joint_semantics") or {}).get("sha256"))
            if exclude_clips:
                # fail-closed: a derived view may only be cut by an exclusion artifact its
                # derivation declares (codex 2026-09-02 round 3) -- an undeclared cut is refused
                declared = self.derivation.get("exclusions") or {}
                if str(exclude_clips) not in declared:
                    raise RuntimeError(f"exclusion {exclude_clips} is not declared in {deriv_p} "
                                       f"(declared: {sorted(declared)[:6]}...) -- refusing an undeclared cut "
                                       f"on a derived view")
                _want["exclusion"] = (Path(exclude_clips), declared[str(exclude_clips)])
            for what, (path, want) in _want.items():
                if want is None or not path.is_file():
                    raise RuntimeError(f"derivation.json does not pin {what} ({path}) -- refusing an unverifiable view")
                have = hashlib.sha256(path.read_bytes()).hexdigest()
                if have != str(want):
                    raise RuntimeError(f"derived view {what} {path} hashes {have[:16]}, derivation.json pins "
                                       f"{str(want)[:16]}")
            self.derivation_sha256 = hashlib.sha256(deriv_p.read_bytes()).hexdigest()
        # a stats artifact built for another REPRESENTATION (payloads that are not KTJD-17 channels, e.g. the AnyTop-13 view)
        # may only serve a view whose derivation.json declares that same representation -- an incomplete or stripped view would
        # otherwise load as plain KTJD-17 with foreign statistics (codex 2026-09-08 r2 #3)
        _stats_rep = self.percell_meta.get("representation")
        _view_rep = ((self.derivation or {}).get("representation") or {}).get("id") if self.derivation is not None else None
        if (_stats_rep not in (None, "ktjd17") or _view_rep is not None) and str(_stats_rep) != str(_view_rep):
            raise RuntimeError(f"per-cell stats declare representation {_stats_rep!r} but the view declares {_view_rep!r} "
                               f"(derivation.json {'present' if self.derivation is not None else 'MISSING'}) -- refusing")
        self.representation = _view_rep
        rows = [json.loads(l) for l in open(manifest_p)]
        rows = [r for r in rows if r.get("status") == "accept"]
        _drop = load_exclusions(exclude_clips)
        if _drop:
            n0 = len(rows)
            rows = [r for r in rows if str(r["clip_id"]) not in _drop]
            self.provenance_exclusion = {
                "path": str(exclude_clips),
                "sha256": hashlib.sha256(Path(exclude_clips).read_bytes()).hexdigest(),
                "n_excluded": n0 - len(rows)}
            print(f"[ktjd] exclusion list drops {n0 - len(rows)} of {n0} accepted clips "
                  f"({Path(exclude_clips).name})", flush=True)
        else:
            self.provenance_exclusion = None
        self.samples = [{"object_type": str(r["rig_id"]), "path": str(r["clip_id"])}
                        for r in rows]
        self._rows = rows
        missing_pc = sorted({s_["object_type"] for s_ in self.samples} - set(self._pc))
        if missing_pc:
            raise RuntimeError(f"per-cell stats miss {len(missing_pc)} rigs the manifest serves, "
                               f"first: {missing_pc[:5]}")
        self._skel: dict[str, dict] = {}
        self._geo: dict[str, np.ndarray] = {}
        self._masks: dict[str, dict] = {}
        self._std: dict[str, np.ndarray] = {}

        # ---- joint semantics (KTJD order; hash-checked per rig on first use) ----
        with np.load(joint_semantics, allow_pickle=False) as z:
            self._sem_order = json.loads(str(z["__order_hash"]))
            self._sem_dim = int(z["__dim"])
            self._sem = {k[len("emb__"):]: np.asarray(z[k]) for k in z.files
                         if k.startswith("emb__")}

        # ---- caption embeddings: sidecar pair keyed "<clip_id>__capN" ----
        cache = Path(caption_emb_cache)
        embs = np.load(cache.with_suffix(".embs.npy"), mmap_mode="r")
        keys = json.load(open(cache.with_suffix(".keys.json")))
        want = {s["path"] for s in self.samples}
        per: dict[str, list[tuple[int, int]]] = {}
        for i, k in enumerate(keys):
            clip, _, cap = k.rpartition("__cap")
            if clip in want:
                per.setdefault(clip, []).append((int(cap), i))
        missing = sorted(want - set(per))
        if missing:
            # A clip with no caption cannot be trained on -- the text pathway would receive a zero
            # vector indistinguishable from CFG's dropped-text branch. DROP such clips, loudly and
            # with a hard ceiling: a handful is a data edge case (the PZ+Human build has exactly 2,
            # mirrored HumanML3D clips), thousands would mean the wrong cache is wired up.
            frac = len(missing) / max(len(want), 1)
            if frac > 0.01:
                raise RuntimeError(f"caption join incomplete for {len(missing)} of {len(want)} "
                                   f"clips ({frac*100:.2f}%) -- that is a wiring error, not an "
                                   f"edge case. First: {missing[:5]}")
            drop = set(missing)
            print(f"[ktjd] dropping {len(drop)} clip(s) with no caption embedding: "
                  f"{sorted(drop)[:5]}", flush=True)
            rows = [r for r in rows if str(r["clip_id"]) not in drop]
            self.samples = [s_ for s_ in self.samples if s_["path"] not in drop]
            want = want - drop
        # cap indices must be CONTIGUOUS 0..n-1 (codex round-S0): a gap would silently shift the
        # string<->vector pairing for every caption after it.
        for c, v in per.items():
            idxs = sorted(i for i, _ in v)
            if idxs != list(range(len(idxs))):
                raise RuntimeError(f"caption rows for {c!r} are not contiguous cap0..cap{{n-1}}: "
                                   f"{idxs[:8]}")
        self._rows = rows                      # may have been filtered just above
        self._cap_rows = {c: [i for _, i in sorted(v)] for c, v in per.items()}
        self._cap_embs = embs
        if embs.ndim != 2 or embs.shape[0] != len(keys):
            raise RuntimeError(f"caption cache malformed: embs {embs.shape} vs {len(keys)} keys")
        self.caption_dim = int(embs.shape[1])
        # Hash and finiteness-check EVERY row this corpus will actually serve (codex round-2:
        # a 32-row spot check and a byte count are not provenance). Only our rows are read, so
        # this costs ~80MB rather than the 4GB whole-cache scan, and it is the exact payload a
        # resume must not see change underneath it.
        used = np.array(sorted(i for rows in self._cap_rows.values() for i in rows),
                        dtype=np.int64)
        h = hashlib.sha256()
        for s0 in range(0, used.size, 4096):
            chunk = np.ascontiguousarray(embs[used[s0:s0 + 4096]], dtype=np.float32)
            if not np.isfinite(chunk).all():
                raise RuntimeError("caption cache contains non-finite embeddings")
            h.update(chunk.tobytes())
        self._cap_payload_sha = h.hexdigest()

        # ---- caption STRINGS: same JSON the cache was built from, same ordering law ----
        # The cache's meta.json declares its source captions JSON by sha256; binding strings from
        # any other file (or any other per-clip ordering than ordered_captions) desynchronizes
        # string[i] from the vector at "<clip>__cap<i>" (codex round-S0: the bare-id list lookup
        # joined 0/986 -- every clip silently got ""). Fail loud on every mismatch.
        tj = Path(texts_json)
        if not tj.exists():                      # legacy convention: a NAME under the old corpus root
            tj = Path("data/animo4d_L4TB_plus_human_v4b272neutral") / texts_json
        meta_p = cache.with_suffix(".meta.json")
        if not meta_p.exists():
            raise RuntimeError(f"caption cache {cache} has no .meta.json -- cannot verify which "
                               f"captions JSON produced it; refusing to bind strings blindly")
        cache_meta = json.loads(meta_p.read_text())
        want_sha = cache_meta.get("captions_json_sha256")
        if not want_sha:
            raise RuntimeError(f"{meta_p} declares no captions_json_sha256 -- refusing to bind "
                               f"caption strings to embeddings on unverifiable provenance")
        texts_bytes = tj.read_bytes()
        if hashlib.sha256(texts_bytes).hexdigest() != want_sha:
            raise RuntimeError(f"texts_json {tj} sha256 does not match the cache's declared "
                               f"source ({cache_meta.get('captions_json_name')}); the strings "
                               f"would not correspond to the embeddings")
        texts = json.loads(texts_bytes)
        self._cap_texts = {}
        for s in self.samples:
            caps = ordered_captions(texts.get(f"{s['path']}.npy", {}))
            if not caps:
                raise RuntimeError(f"no caption strings for KTJD clip {s['path']!r} in {tj}")
            if len(caps) != len(self._cap_rows[s["path"]]):
                raise RuntimeError(f"caption count mismatch for {s['path']!r}: {len(caps)} "
                                   f"strings vs {len(self._cap_rows[s['path']])} embedding rows")
            self._cap_texts[s["path"]] = caps
        self._rng = np.random.default_rng(0)

        # resume-pinning surface (trainer stores + re-checks these on every resume)
        self.provenance = {
            "generation_id": self.generation_id,
            "freeze_generation_id": str(gz["freeze_generation_id"]),
            "gains_sha256": hashlib.sha256(gains_p.read_bytes()).hexdigest(),
            "schema_sha256": hashlib.sha256(schema_p.read_bytes()).hexdigest(),
            "caption_keys_sha256": hashlib.sha256(
                cache.with_suffix(".keys.json").read_bytes()).hexdigest(),
            "caption_payload_sha256": self._cap_payload_sha,   # the served rows themselves
            "caption_dim": self.caption_dim,
            "texts_json_sha256": hashlib.sha256(texts_bytes).hexdigest(),
            "joint_sem_sha256": hashlib.sha256(Path(joint_semantics).read_bytes()).hexdigest(),
            # target parameterization: resuming a pre-centering ckpt into centered data (or the
            # reverse) trains against a different objective under one lineage -- pin it.
            # the normalisation variant rides in THIS existing key (no new pin key: every checkpoint written before the
            # variant existed must keep matching its view under the bidirectional pin check of the trainer / renderer / eval)
            "target_centering": {"percell": TARGET_CENTERING, "scale_only": "ktjd_spec_scale_only_v1",
                                 "rest": "ktjd_rest_centered_scale_v1"}[self.normalization],
            "percell_stats_cohort": str(self.percell_meta.get("cohort")),
            # pin the ARTIFACT, not just its description: an NPZ swapped under the same path must
            # not pass a fresh launch or a resume (codex round-S7 blocker 4)
            "percell_sha256": hashlib.sha256(Path(percell_stats).read_bytes()).hexdigest(),
            "percell_std_floor": float(self.percell_meta.get("std_floor", -1)),
            "percell_convention": str(self.percell_meta.get("convention")),
            # derived training views only (keys ABSENT for a frozen parent corpus, so existing
            # checkpoints' pins keep resuming unchanged)
            **({"manifest_sha256": manifest_sha, "derivation_sha256": self.derivation_sha256}
               if self.derivation is not None else {}),
            # representation views only (key ABSENT for KTJD-17 corpora and for views without a representation record)
            **({"representation": str(self.representation)} if self.representation else {}),
        }

    # ------------------------------------------------------------------ per-rig statics
    def _skeleton(self, rig: str) -> dict:
        if rig not in self._skel:
            z = np.load(self.root / "skeletons" / f"{rig}.npz", allow_pickle=True)
            sk = {k: z[k] for k in ("joint_names", "parents", "P_rest_global", "R_rest_global",
                                    "R_rest_local", "offset_parent_local",
                                    "rotation_source_kind", "s_rig")}
            names = [str(x) for x in sk["joint_names"]]
            want = hashlib.sha256("|".join(names).encode()).hexdigest()
            if self._sem_order.get(rig) != want:
                raise RuntimeError(f"joint_semantics order hash mismatch for {rig!r}: table was "
                                   f"built against a different KTJD joint ordering")
            self._skel[rig] = sk
        return self._skel[rig]

    def skeleton(self, rig: str) -> dict:
        """The rig's skeleton npz fields at their stored precision (float64); item fields are float32 copies."""
        return self._skeleton(rig)

    def _geodesic(self, rig: str) -> np.ndarray:
        if rig not in self._geo:
            par = [int(p) for p in self._skeleton(rig)["parents"]]
            J = len(par)
            depth = np.zeros(J, np.int64)
            anc = []
            for j in range(J):
                if j:
                    depth[j] = depth[par[j]] + 1
                c, k = {}, j
                while k >= 0:
                    c[k] = depth[j] - depth[k]
                    k = par[k]
                anc.append(c)
            g = np.zeros((J, J), np.float32)
            for i in range(J):
                for j in range(J):
                    common = anc[i].keys() & anc[j].keys()
                    l = max(common, key=lambda k: depth[k])
                    g[i, j] = (depth[i] - depth[l]) + (depth[j] - depth[l])
            self._geo[rig] = g
        return self._geo[rig]

    def static_masks(self, rig: str) -> dict:
        """channel_valid [J,17] (root-only 13:17; fixed_dof rows lose rot supervision 3:9)."""
        if rig not in self._masks:
            sk = self._skeleton(rig)
            J = len(sk["parents"])
            cv = np.zeros((J, 17), dtype=bool)
            cv[:, :13] = True
            cv[0, 13:17] = True
            fixed = np.asarray(sk["rotation_source_kind"]).astype(str) == "fixed_dof"
            cv[fixed, 3:9] = False
            # cells the stats artifact marks unsupervised (exact constants) leave the loss AND the
            # model input: they carry no information, and as constant-zero targets they only
            # dilute the group they sit in -- ch12 contact most of all.
            sup = self._sup.get(rig)
            if sup is not None:
                cv &= sup[:J]
            self._masks[rig] = {"channel_valid": cv,
                                "rotation_supervised": ~fixed}
        return self._masks[rig]

    def postcrop_window(self, w: np.ndarray, vm: np.ndarray, rig: str) -> np.ndarray:
        """KTJD crop contract (loader.py:99-122): every crop re-bases smooth-root XZ so the
        window's FIRST frame sits at the local origin. The generic InContextPairs._crop only
        slices, which leaves mid-clip windows with a nonzero ch13:15 start (codex S0: 28/40
        random demo windows violated the contract). Normalization is scale-only (mean=0), so
        Root is row 0 in KTJD serving order; padding frames stay exact zero.
        PER-CELL CORRECTION (codex round-S7 blocker 1): under the old scale-only normalization
        mean was zero, so subtracting the first frame's NORMALIZED value was the same as rebasing
        the raw one. With per-cell mean/std it is not. Writing n = (raw - mu)/sd, the rebased
        target is n' = (raw - raw0 - mu)/sd = n - n0 - mu/sd, i.e. the normalized first frame AND
        an extra mu/sd term. Omitting it left a constant root-XZ de-normalization error of up to
        0.357 in the tensors actually trained on -- which the adapter-level round-trip could not
        see, because it measured BEFORE the crop.
        """
        if vm.any():
            mu, sd = self._stats(rig)
            J = w.shape[1]
            f0 = int(np.argmax(vm))
            off = w[f0, 0, 13:15] + mu[0, 13:15] / (sd[0, 13:15] + _STD_FLOOR)
            if np.abs(off).any():
                w = w.copy()
                w[vm, 0, 13:15] -= off
        return w

    def _rest_raw17(self, rig: str) -> np.ndarray:
        """[J,17] the rig's REST frame in RAW KTJD-17 units, from the skeleton file alone: q_position of the rest pose
        (P_rest_global minus its own root XZ), identity rest-delta 6D [1,0,0,0,1,0], zero velocity / contact / root
        track, heading [1, 0] (canonical rest faces +Z)."""
        sk = self._skeleton(rig)
        J = len(sk["parents"])
        P = np.asarray(sk["P_rest_global"], dtype=np.float64)
        raw = np.zeros((J, 17), dtype=np.float64)
        raw[:, 0:3] = P
        raw[:, 0] -= P[0, 0]
        raw[:, 2] -= P[0, 2]
        raw[:, 3:9] = np.array([1, 0, 0, 0, 1, 0], dtype=np.float64)[None]
        raw[0, 15] = 1.0
        return raw.astype(np.float32)

    def _stats(self, rig: str) -> tuple[np.ndarray, np.ndarray]:
        """(mean [J,17], std [J,17]) of the serving normalization: raw = x * (std + _STD_FLOOR) + mean."""
        if self.normalization == "percell":
            return self._pc[rig]
        if rig not in self._pc_so:
            J = len(self._skeleton(rig)["parents"])
            # "rest": the mean is the rig's rest frame (skeleton-derived); "scale_only": zero
            mu = self._rest_raw17(rig).copy() if self.normalization == "rest" else np.zeros((J, 17), np.float32)
            # exact-constant cells leave supervision AND the model input (static_masks), so the sampler holds them at
            # normalized 0; decoding must restore the constant, exactly as per-cell decoding does -- the mean is therefore
            # the data's constant on those cells (codex 2026-09-06 r2 P1: 13 such cells on 4 rigs) for scale_only AND rest;
            # a deployed unseen rig has no such cells (its channel_valid is purely structural)
            sup = self._sup.get(rig)
            if sup is not None:
                mu_pc, _ = self._pc[rig]
                keep = ~np.asarray(sup[:J], dtype=bool)
                mu[keep] = np.asarray(mu_pc[:J], dtype=np.float32)[keep]
            self._pc_so[rig] = (mu, np.ascontiguousarray(self._std_eff(rig)[:, :17]).astype(np.float32))
        return self._pc_so[rig]

    def _std_eff(self, rig: str) -> np.ndarray:
        """[J,18] de-normalization scale minus _STD_FLOOR (plane 17 = identity scale)."""
        if rig not in self._std:
            sk = self._skeleton(rig)
            J = len(sk["parents"])
            s_rig = float(sk["s_rig"])
            g = self.gains
            scale = np.ones((J, 18), dtype=np.float32)
            scale[:, 0:3] = s_rig / g[0]
            scale[:, 9:12] = s_rig / g[1]
            scale[0, 13:15] = s_rig / g[2]
            self._std[rig] = (scale - _STD_FLOOR).astype(np.float32)
        return self._std[rig]

    def rest_anchor_frame(self, rig: str) -> np.ndarray:
        """[J,17] NORMALIZED rest-pose frame -- the variant-A flow anchor (UMO source-centered
        base with a content-free geometric prior). Analytic per the KTJD encoding:
          ch0:3   q_position of the rest pose = P_rest_global minus its own root XZ (smooth-root
                  of a static pose is its root trajectory), then * g_q/s_rig;
          ch3:9   delta rotation identity -> column-cont6d(I) = [1,0,0,0,1,0];
          ch9:12  zeros (static);   ch12 contact 0 (NOT a valid-motion claim -- the anchor is a
                  geometric base, not a decodable clip; contact is left neutral and documented);
          root 13:15 smooth-root at origin = 0;  root 15:17 heading identity = [1,0]
                  (canonical rest faces +Z; codec heading = [fwd_z, fwd_x]/|h|).
        Invalid cells (non-root 13:17, fixed_dof 3:9 stay AS THE DATA HAS THEM: fixed rows carry
        the identity delta too, which the codec treats as inherited) are zero where channel_valid
        is zero, matching the noise-zeroing contract."""
        if not hasattr(self, "_anchor"):
            self._anchor = {}
        if rig not in self._anchor:
            sk = self._skeleton(rig)
            J = len(sk["parents"])
            P = np.asarray(sk["P_rest_global"], dtype=np.float64)          # [J,3]
            q = P.copy()
            q[:, 0] -= P[0, 0]
            q[:, 2] -= P[0, 2]
            a = np.zeros((J, 17), dtype=np.float64)
            s_rig = float(sk["s_rig"])
            a[:, 0:3] = q * (self.gains[0] / s_rig)
            a[:, 3:9] = np.array([1, 0, 0, 0, 1, 0], dtype=np.float64)[None]
            a[0, 15] = 1.0                                                 # heading [1,0]
            cv = self.static_masks(rig)["channel_valid"]
            a[~cv] = 0.0
            self._anchor[rig] = a.astype(np.float32)
        return self._anchor[rig]

    def _rest_centering_SUPERSEDED(self, rig: str) -> tuple[np.ndarray, np.ndarray]:
        """(rest_norm [J,18], rest_raw [J,18]) -- the REST-CENTERING offset pair.

        WHY (2026-08-20, frozen-pose diagnosis): KTJD's model-space normalization is scale-only
        (s_rig + frozen block gains, loader.normalize_model_motion) -- unlike AnyTopDataset,
        which subtracts a per-(joint,channel) mean. Left uncentered, every joint's coordinate
        carries the rig's standing pose as a large constant, and MEASURED on all 671 train
        clips a constant prediction already removes 75.0% of the gradient-weighted objective
        (61.2% from rig identity alone) against 45.2%/5.9% for the 13ch corpus. Frozen-pose is
        then the objective's cheap optimum, which is exactly what the arms produced: 89-90% of
        their visible motion was rigid root translation and body articulation ran at 2-3% of GT.

        The offset is the rig's REST frame, not an empirical cohort mean, because rest is
        derivable from the skeleton alone -- available for UNSEEN rigs and at deployment, where
        no train statistics exist (the transductive trap). For rot6d the rest delta is exactly
        the identity, whose 6D form carries the entire 0.3333/element norm constant, so
        centering removes precisely that constant.

        The pair rides the EXISTING de-normalization convention: the item serves
        x_centered = raw/scale - rest_norm and reports anytop_mean = rest_raw, so every consumer
        of `x * (std + _STD_FLOOR) + mean` (renderer, gamma7 pack, diagnostics) inverts it
        exactly with no special case. Plane 17 (the heading-valid flag) is never centered.
        """
        if not hasattr(self, "_center"):
            self._center = {}
        if rig not in self._center:
            rn = np.zeros((len(self._skeleton(rig)["parents"]), 18), dtype=np.float32)
            rn[:, :17] = self.rest_anchor_frame(rig)
            scale = self._std_eff(rig) + _STD_FLOOR                       # [J,18]
            self._center[rig] = (rn, (rn * scale).astype(np.float32))
        return self._center[rig]

    def rest_frame_normalized(self, rig: str) -> np.ndarray:
        """[J,18] the rig's REST pose in the CURRENT normalized space -- the 1-frame demo.

        user 2026-08-21: "demo 我们就传 rest pose,就给 1 帧,我们简化来". This strips the demo of
        all motion content, leaving only the rig's static geometry, which the model already
        receives through joint_semantics / struct_feats / rest_offsets. That is the point: it
        removes the demo-interpretation variable entirely, so the run tests whether the raw-space
        DiT can produce coherent motion from text + skeleton alone. Plane 17 (heading-valid) is 1:
        a rest pose has a well-defined heading (canonical rest faces +Z).
        """
        if not hasattr(self, "_restf"):
            self._restf = {}
        if rig not in self._restf:
            sk = self._skeleton(rig)
            J = len(sk["parents"])
            raw = np.zeros((J, 17), dtype=np.float64)
            P = np.asarray(sk["P_rest_global"], dtype=np.float64)
            q = P.copy(); q[:, 0] -= P[0, 0]; q[:, 2] -= P[0, 2]
            raw[:, 0:3] = q
            raw[:, 3:9] = np.array([1, 0, 0, 0, 1, 0], dtype=np.float64)[None]
            raw[0, 15] = 1.0                                   # heading [1,0] = facing +Z
            cv = self.static_masks(rig)["channel_valid"]
            raw[~cv] = 0.0
            mu, sd = self._stats(rig)
            if self.representation or self.normalization == "rest":
                # a representation view stores excluded constants with an effective std of _STD_FLOOR: an excluded cell must hold
                # ITS constant here (not 0), or (0 - const) / 1e-6 explodes before the pair loader clamps it (codex 2026-09-08 r4);
                # under "rest" the mean of an excluded cell is likewise its constant, so the demo is exactly 0 there too
                raw[~cv[:J]] = np.asarray(mu[:J], dtype=np.float64)[~cv[:J]]
            n = (raw.astype(np.float32) - mu[:J]) / (sd[:J] + _STD_FLOOR)
            self._restf[rig] = np.concatenate(
                [n, np.ones((J, 1), np.float32)], axis=1).astype(np.float32)
        return self._restf[rig]

    # ------------------------------------------------------------------ item surface
    def __len__(self):
        return len(self._rows)

    def __getitem__(self, i: int) -> dict:
        r = self._rows[i]
        rig, clip = str(r["rig_id"]), str(r["clip_id"])
        sk = self._skeleton(rig)
        payload = load_motion_npz(self.root / r["motion_relpath"], expected_fps_target=30.0)
        # manifest<->payload identity (codex round-S0): a renamed/moved npz must not silently
        # serve another clip's frames under this row's caption and split membership.
        if str(payload["clip_id"]) != clip or str(payload["rig_id"]) != rig:
            raise RuntimeError(f"manifest/payload identity mismatch at {r['motion_relpath']}: "
                               f"row says {clip!r}/{rig!r}, payload says "
                               f"{payload['clip_id']!r}/{payload['rig_id']!r}")
        motion = np.asarray(payload["motion"])
        T, J = motion.shape[0], motion.shape[1]
        view = build_model_view(
            motion, np.asarray(payload["heading_valid"]),
            parents=sk["parents"], R_rest_global=sk["R_rest_global"],
            rotation_source_kind=sk["rotation_source_kind"],
            s_rig=float(sk["s_rig"]), gains=self.gains,
            T_max=T, J_max=J, crop_start=0, crop_length=T, yaw_radians=0.0)
        x = np.concatenate(
            [view.motion,
             np.broadcast_to(view.masks.heading_valid.astype(np.float32)[:, None, None],
                             (T, J, 1))], axis=-1)                     # [T,J,18]
        # OLD-STYLE per-cell standardization. build_model_view already applied the spec's
        # scale-only normalization, so undo it first and standardize the RAW values: the artifact's
        # mean/std were measured on raw payloads, and the repo-wide de-normalization convention
        # x*(std+_STD_FLOOR)+mean must return RAW for the official decoder to work.
        mu, sd = self._stats(rig)
        x = x.copy()
        raw17 = motion[:, :J, :17].astype(np.float32)            # the untouched payload
        x[..., :17] = (raw17 - mu[None, :J]) / (sd[None, :J] + _STD_FLOOR)

        rows = self._cap_rows[clip]
        cap_i = int(self._rng.integers(len(rows))) if (self.random_caption and len(rows) > 1) \
            else 0
        caps = self._cap_texts.get(clip) or [""]

        return {
            "object_type": rig,
            "motion_id": clip,
            "num_joints": J,
            "num_frames": int(view.T_valid),
            "anytop_x": np.ascontiguousarray(x.transpose(1, 2, 0)),   # [J,18,T]
            "geodesic_dist": self._geodesic(rig),
            "caption_emb": np.asarray(self._cap_embs[rows[cap_i]], dtype=np.float32),
            "caption": caps[min(cap_i, len(caps) - 1)],
            "joint_semantics": self._sem[rig],
            "anytop_mean": np.concatenate(
                [mu[:J], np.zeros((J, 1), np.float32)], axis=1),     # plane 17 carries no offset
            "anytop_std": np.concatenate(
                [sd[:J], np.ones((J, 1), np.float32) - _STD_FLOOR], axis=1),
            "parent_indices": np.asarray(sk["parents"], dtype=np.int64),
            "rest_offsets": np.asarray(sk["offset_parent_local"], dtype=np.float32),
            # KTJD extras (gamma7-17 / anchors consume these later)
            "R_rest_global": np.asarray(sk["R_rest_global"], dtype=np.float32),
            "P_rest_global": np.asarray(sk["P_rest_global"], dtype=np.float32),
            "s_rig": float(sk["s_rig"]),
        }
