"""TokenCacheDataset — reads the offline RVQ token cache produced by
scripts/export_graph_vq_tokens.py for Graph-CodeFlow training.

Each item is one exported clip: the post-RVQ z_q target + graph metadata + dual
text caption tensors. The CodeFlow trainer reads these instead of running the
frozen tokenizer encoder online every step (handoff §5.1).

Padding is along the C (coarse-slot) and T_lat axes and is ALREADY baked into the
export (token_mask/coarse_mask/frame_mask_lat). All exported clips share the same
[T_lat, C_max, D, Q] padded shape (from the frozen tokenizer's max_coarse /
temporal_stride), so the default collate stacks them directly — no ragged collate.
The pooled_geodesic sentinel (export GEO_INF_SENTINEL for +inf) is mapped back to
+inf here so GraphAttentionBlock sees its real unreachable-pair contract.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

GEO_INF_SENTINEL = 30000.0


class TokenCacheDataset(Dataset):
    _warned_no_identity = False

    def __init__(self, cache_dir: str, split: str,
                 geo_inf_sentinel: float = GEO_INF_SENTINEL,
                 authority_root: str | None = None,
                 caption_sidecar: str | None = None,
                 caption_sampling: str = "fixed",
                 caption_seed: int = 42,
                 caption_verify: bool = True) -> None:
        """`authority_root` is the corpus whose `cond.npy` DEFINES each object type's topology.

        Without it, a payload is only checked against itself: recomputing the canonical form from
        the saved parents catches a forged hash, but not a payload that carries a consistent
        DIFFERENT tree — a reviewer built one with the correct motion and object identity, a valid
        alternative parent tree and a matching recomputed hash, and both this loader and the
        holdout guard accepted it. Self-consistency is not authority. Default None keeps the
        existing behaviour so the protected baseline cache still loads unchanged.
        """
        self.split_dir = Path(cache_dir) / split
        idx_path = self.split_dir / "index.jsonl"
        if not idx_path.exists():
            raise FileNotFoundError(
                f"TokenCacheDataset: {idx_path} not found (run "
                f"scripts/export_graph_vq_tokens.py first)")
        self.rows = [json.loads(l) for l in idx_path.read_text().splitlines() if l.strip()]
        if not self.rows:
            raise RuntimeError(f"TokenCacheDataset: empty index {idx_path}")
        self.geo_inf_sentinel = float(geo_inf_sentinel)
        # ---- Multi-caption RANDOM sampling (user requirement, 2026-08-05) ----
        # "fixed" (default) serves the npz-baked primary caption — byte-identical legacy.
        # "random" re-reads the caption from the LLM2Vec RAGGED sidecar per __getitem__,
        # uniformly over ALL of the motion's captions, reseeded per (seed, epoch, index) via
        # set_caption_epoch so every epoch sees a fresh assignment and every rank/worker is
        # deterministic. No re-export needed: the sidecar already holds every occurrence, and
        # the npz's own caption is by construction sidecar row cap0.
        if caption_sampling not in ("fixed", "random"):
            raise ValueError(f"caption_sampling must be fixed|random, got {caption_sampling!r}")
        self.caption_sampling = caption_sampling
        self.caption_seed = int(caption_seed)
        self._caption_epoch = 0
        self._cap_embs = self._cap_toks = self._cap_offs = None
        self._cap_rows_of: dict[str, list[int]] | None = None
        if caption_sampling == "random":
            if not caption_sidecar:
                raise ValueError("caption_sampling=random requires caption_sidecar "
                                 "(the LLM2Vec ragged sidecar prefix)")
            import hashlib as _hl2
            pfx = Path(caption_sidecar)
            meta_p = Path(f"{pfx}.meta.json")
            if not meta_p.exists():
                raise FileNotFoundError(f"{meta_p} missing — refusing a caption sidecar "
                                        f"without provenance")
            # PROVENANCE, byte-level: the npz text in this cache was baked from a specific
            # corpus revision and caption cache; the sidecar we sample from must be the SAME
            # artifact, or index i's "other captions" could describe different motions.
            man_p = Path(cache_dir) / "manifest.json"
            man = json.loads(man_p.read_text()) if man_p.exists() else {}
            cp = man.get("caption_provenance") or {}
            want_meta_sha = cp.get("caption_cache_meta_sha256")
            got_meta_sha = _hl2.sha256(meta_p.read_bytes()).hexdigest()
            if want_meta_sha and got_meta_sha != want_meta_sha:
                raise RuntimeError(
                    f"caption sidecar {pfx} meta sha {got_meta_sha[:16]}... != the one this "
                    f"token cache was exported against ({want_meta_sha[:16]}...). Sampling "
                    f"from a different caption artifact than the baked text came from.")
            if not want_meta_sha:
                raise RuntimeError(
                    f"{man_p} carries no caption_provenance.caption_cache_meta_sha256 — this "
                    f"cache predates caption provenance and random sampling against it "
                    f"cannot be certified. Re-export, or use caption_sampling=fixed.")
            # PAYLOAD PINNING (codex caption review round-2 #1): a public deterministic
            # spot-check is bypassable by tampering that preserves the sampled rows, so
            # random mode REQUIRES full-content sha256 of all four payload files anchored
            # in the trusted manifest (written once by _pin_caption_sidecar_payloads.py
            # after a secret-seed authenticity audit) and verifies them here.
            pins = cp.get("payload_sha256")
            if not pins:
                raise RuntimeError(
                    f"{man_p} caption_provenance carries no payload_sha256 — the sidecar "
                    f"payloads are not content-pinned. Run "
                    f"scripts/_pin_caption_sidecar_payloads.py before random sampling.")
            # caption_verify=False (non-rank-0 DDP ranks): the FULL verification below
            # (hash + lattice + exhaustive scan + spot-checks) runs on rank 0 only — all
            # ranks read the same shared-fs files, and 8 ranks re-verifying concurrently
            # contended so hard the first NCCL collective timed out (smoke 2026-08-05
            # SIGABRT). Rank 0 failing kills the whole torchrun job, so skipping ranks
            # get no weaker guarantee about the artifact that is actually trained on.
            # Size pins stay checked on every rank (cheap stat, catches gross swaps).
            for suf in ("keys.json", "embs.npy", "tokens.npy", "offsets.npy"):
                fp = Path(f"{pfx}.{suf}")
                want = pins[suf]
                if fp.stat().st_size != int(want["bytes"]):
                    raise RuntimeError(f"{fp}: size {fp.stat().st_size} != pinned "
                                       f"{want['bytes']} — payload replaced or truncated.")
                if not caption_verify:
                    continue
                h = _hl2.sha256()
                with open(fp, "rb") as fh:
                    for blk in iter(lambda: fh.read(1 << 24), b""):
                        h.update(blk)
                if h.hexdigest() != want["sha256"]:
                    raise RuntimeError(
                        f"{fp}: sha256 {h.hexdigest()[:16]}... != anchored "
                        f"{want['sha256'][:16]}... — the payload is not the pinned artifact.")
            self._cap_embs = np.load(f"{pfx}.embs.npy", mmap_mode="r")     # [N, D] f32
            self._cap_toks = np.load(f"{pfx}.tokens.npy", mmap_mode="r")   # [total, D] f16
            self._cap_offs = np.load(f"{pfx}.offsets.npy")                 # [N+1]
            keys = json.loads(Path(f"{pfx}.keys.json").read_text())
            # Defense-in-depth lattice below (cheap; catches build bugs and drift even
            # if the pins themselves were mis-anchored):
            #   keys    == the canonical enumeration RE-DERIVED from the corpus json whose
            #             sha256 the manifest pins;
            #   offsets == meta.total_tokens (pinned) at [-1], len == n_rows+1, monotone;
            #   embs/tokens == spot-verified against 256 npz-baked captions.
            rows_of: dict[str, list[tuple[int, int]]] = {}
            for ri, k in enumerate(keys):
                stem, _, idx = k.rpartition("__cap")
                rows_of.setdefault(stem, []).append((int(idx), ri))
            self._cap_rows_of = {m: [ri for _, ri in sorted(v)] for m, v in rows_of.items()}
            if caption_verify:
                meta = json.loads(meta_p.read_text())
                tj_name = meta.get("captions_json_name")
                tj_sha = meta.get("captions_json_sha256")
                want_corpus_sha = cp.get("captions_json_sha256")
                if want_corpus_sha and tj_sha != want_corpus_sha:
                    raise RuntimeError(
                        f"sidecar corpus sha {str(tj_sha)[:16]}... != manifest corpus sha "
                        f"{want_corpus_sha[:16]}...")
                _root = man.get("anytop_root")
                if not (_root and tj_name):
                    raise RuntimeError("cannot re-derive caption keys: manifest lacks "
                                       "anytop_root or sidecar meta lacks captions_json_name")
                _tj = Path(_root) / tj_name
                if _hl2.sha256(_tj.read_bytes()).hexdigest() != tj_sha:
                    raise RuntimeError(f"{_tj} does not hash to the pinned corpus sha")
                from src.data.caption_keys import canonical_occurrences
                _occ = canonical_occurrences(json.loads(_tj.read_text()))
                if [k for k, _ in _occ] != keys:
                    raise RuntimeError(
                        "keys.json does not equal the canonical enumeration of the pinned "
                        "corpus — the sidecar keys were not built from this corpus")
                if len(self._cap_offs) != len(keys) + 1 or int(self._cap_offs[0]) != 0:
                    raise RuntimeError("offsets shape/origin mismatch vs keys")
                if int(self._cap_offs[-1]) != int(meta.get("total_tokens", -1)):
                    raise RuntimeError(
                        f"offsets[-1]={int(self._cap_offs[-1])} != pinned meta.total_tokens="
                        f"{meta.get('total_tokens')}")
                if (np.diff(self._cap_offs) <= 0).any():
                    raise RuntimeError("offsets not strictly increasing (empty/negative row)")
                if (self._cap_embs.shape[0] != len(keys)
                        or self._cap_toks.shape[0] != int(self._cap_offs[-1])):
                    raise RuntimeError("embs/tokens row counts disagree with keys/offsets")
                # EXHAUSTIVE identity + coverage scan (codex round-3): EVERY row's npz is
                # opened and checked — a divergent payload_motion_id or an uncovered
                # has_text=True row anywhere in the cache fails construction, not epoch N.
                # (__getitem__ enforces payload==index identity per served row; construction
                # must give the same guarantee for ALL rows, sampled or not.) IO-bound zip
                # reads amortised over a thread pool: ~1 min warm for 90k rows.
                from concurrent.futures import ThreadPoolExecutor

                def _scan_row(r):
                    d0 = np.load(self.split_dir / r["file"], allow_pickle=False)
                    pid = (str(d0["payload_motion_id"]) if "payload_motion_id" in d0
                           else None)
                    if pid is not None and pid != str(r.get("motion_id")):
                        raise RuntimeError(
                            f"{r['file']}: npz payload_motion_id={pid!r} != index motion_id="
                            f"{str(r.get('motion_id'))!r} — identity mismatch; __getitem__ "
                            f"would refuse this row, failing at construction instead.")
                    covered = r.get("motion_id") in self._cap_rows_of
                    if not covered and bool(d0["has_text"]):
                        raise RuntimeError(
                            f"{r['file']} ({r.get('motion_id')}) has_text=True but is "
                            f"absent from the caption sidecar — refusing to silently "
                            f"serve its baked primary instead of sampling.")
                    return covered

                with ThreadPoolExecutor(max_workers=8) as _tp:
                    _covered = list(_tp.map(_scan_row, self.rows, chunksize=64))
                n_missing = sum(1 for c in _covered if not c)
                missing = [r["motion_id"] for r, c in zip(self.rows, _covered) if not c][:5]
                print(f"[token-cache] exhaustive scan: {len(self.rows)} rows, identities "
                      f"verified; {n_missing} absent from sidecar"
                      + (f" (e.g. {missing}), all verified textless" if n_missing else "")
                      + ".")
                # embs/tokens payload spot-verification: 256 deterministic rows must reproduce
                # the npz-baked cap0 bytes exactly (fp16 round-trip is exact for bf16-sourced
                # values, proven at build time).
                import random as _rnd
                _pick = _rnd.Random(0).sample(range(len(self.rows)), min(256, len(self.rows)))
                _checked = 0
                for _i in _pick:
                    _r = self.rows[_i]
                    _d = np.load(self.split_dir / _r["file"], allow_pickle=False)
                    if ("payload_motion_id" in _d
                            and str(_d["payload_motion_id"]) != str(_r.get("motion_id"))):
                        raise RuntimeError(
                            f"{_r['file']}: payload/index identity mismatch found during "
                            f"sidecar spot-verification.")
                    _rows_c = self._cap_rows_of.get(_r.get("motion_id"))
                    if not _rows_c:
                        if bool(_d["has_text"]):
                            raise RuntimeError(
                                f"{_r['file']}: has_text=True but no sidecar rows (spot-check).")
                        continue
                    if not bool(_d["has_text"]):
                        continue
                    _ri0 = _rows_c[0]
                    _side = np.asarray(self._cap_embs[_ri0], np.float32).astype(np.float16)
                    if not np.array_equal(_side, _d["caption_emb"]):
                        raise RuntimeError(
                            f"sidecar embs row {_ri0} (cap0 of {_r.get('motion_id')}) != the "
                            f"npz-baked caption — the sidecar arrays are not the artifact "
                            f"this cache was exported from.")
                    _a, _b = int(self._cap_offs[_ri0]), int(self._cap_offs[_ri0 + 1])
                    if not np.array_equal(np.asarray(self._cap_toks[_a:_b]),
                                          _d["caption_token_emb"]):
                        raise RuntimeError(
                            f"sidecar tokens rows [{_a}:{_b}] != npz-baked tokens for "
                            f"{_r.get('motion_id')} — payload forgery or drift.")
                    _checked += 1
                print(f"[token-cache] caption sidecar payload verified: keys==canonical "
                      f"enumeration, offsets pinned, {_checked} npz cross-checks exact.")
        self.authority: dict[str, str] | None = None
        if authority_root:
            import hashlib as _hl
            from src.data.holdout_guard import canonical_form as _cf
            cond = np.load(Path(authority_root) / "cond.npy", allow_pickle=True).item()
            self.authority = {
                o: _hl.sha256(_cf(tuple(int(x) for x in
                                        np.asarray(c["parents"]).ravel())).encode()).hexdigest()
                for o, c in cond.items()}

    def set_caption_epoch(self, epoch: int) -> None:
        """Reseed the per-epoch caption assignment (called next to sampler.set_epoch)."""
        self._caption_epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int) -> dict:
        row = self.rows[i]
        d = np.load(self.split_dir / row["file"], allow_pickle=False)
        # Verify the PAYLOAD, not the sidecar. A cache whose index.jsonl disagrees with what the
        # npz files actually contain would pass a guard that only reads the index. Caches
        # exported before this field existed carry no identity and are only warned about, so the
        # protected baseline cache keeps working unchanged.
        if "payload_motion_id" in d:
            for key, col in (("payload_motion_id", "motion_id"),
                             ("payload_object_type", "object_type")):
                if key not in d:
                    raise RuntimeError(
                        f"{row['file']}: payload carries {list(d.keys())[:3]}... but not {key}; "
                        f"a partially-identified payload cannot be verified.")
                got, want = str(d[key]), str(row.get(col))
                if got != want:
                    raise RuntimeError(
                        f"token payload identity mismatch at {row['file']}: npz {key}={got!r}, "
                        f"index.jsonl {col}={want!r}. The index and the payload describe "
                        f"different clips.")
            # Recompute the canonical form from the saved parents rather than trusting the
            # recorded hash: a stored hash proves only that someone wrote a hash.
            import hashlib as _hl
            from src.data.holdout_guard import canonical_form as _cf
            _par = tuple(int(x) for x in np.asarray(d["parent_indices"]).ravel())
            _got = _hl.sha256(_cf(_par).encode()).hexdigest()
            if _got != str(d["payload_canonical_sha256"]):
                raise RuntimeError(
                    f"{row['file']}: the saved parents hash to {_got[:16]}... but the payload "
                    f"records {str(d['payload_canonical_sha256'])[:16]}.... The recorded topology "
                    f"is not the topology in the file.")
            if self.authority is not None:
                _obj = str(d["payload_object_type"])
                _auth = self.authority.get(_obj)
                if _auth is None:
                    raise RuntimeError(
                        f"{row['file']}: object type {_obj!r} is not in the authoritative "
                        f"cond.npy, so its topology cannot be certified against anything.")
                if _got != _auth:
                    raise RuntimeError(
                        f"{row['file']}: the payload's own parents are self-consistent but they "
                        f"are not the topology the corpus defines for {_obj!r} "
                        f"({_got[:16]}... vs {_auth[:16]}...). A payload that agrees with itself "
                        f"is not a payload that agrees with the freeze.")
        elif not TokenCacheDataset._warned_no_identity:
            TokenCacheDataset._warned_no_identity = True
            print(f"[token-cache] {self.split_dir}: payloads carry no embedded identity "
                  f"(exported before that field existed). The index is trusted; this cache "
                  f"cannot back an unseen-topology claim on its own.")
        geo = d["pooled_geodesic"].astype(np.float32)
        geo[geo >= self.geo_inf_sentinel] = np.inf
        # ---- caption override (random mode only; fixed mode = npz bytes untouched) ----
        cap_emb = d["caption_emb"]
        cap_tok = d["caption_token_emb"]
        cap_msk = d["caption_token_mask"]
        if (self.caption_sampling == "random" and bool(d["has_text"])):
            rows_c = self._cap_rows_of.get(str(d["payload_motion_id"])
                                           if "payload_motion_id" in d
                                           else row.get("motion_id"))
            if not rows_c:
                raise RuntimeError(
                    f"{row['file']}: has_text=True but no sidecar captions — construction "
                    f"checks should have caught this; refusing the silent primary fallback.")
            if rows_c:
                import random as _random
                k = _random.Random(
                    self.caption_seed * 1_000_003
                    + self._caption_epoch * 8191 + i).randrange(len(rows_c))
                ri = rows_c[k]
                a, b = int(self._cap_offs[ri]), int(self._cap_offs[ri + 1])
                # fp16 round-trip on the pooled row mirrors the npz bake (training must see
                # the identical quantisation whichever caption index is drawn).
                cap_emb = np.asarray(self._cap_embs[ri], np.float32).astype(np.float16)
                cap_tok = np.asarray(self._cap_toks[a:b])            # [n, D] f16 ragged
                cap_msk = np.ones((b - a,), dtype=bool)
        return {
            "z_q": torch.from_numpy(d["z_q"].astype(np.float32)),           # [T_lat,C,D]
            "indices": torch.from_numpy(d["indices"].astype(np.int64)),     # [T_lat,C,Q]
            "token_mask": torch.from_numpy(d["token_mask"].astype(np.bool_)),
            "coarse_mask": torch.from_numpy(d["coarse_mask"].astype(np.bool_)),
            "frame_mask_lat": torch.from_numpy(d["frame_mask_lat"].astype(np.bool_)),
            "pooled_adjacency": torch.from_numpy(d["pooled_adjacency"].astype(np.float32)),
            "pooled_geodesic": torch.from_numpy(geo),
            "pooled_skeleton_embeddings": torch.from_numpy(
                d["pooled_skeleton_embeddings"].astype(np.float32)),
            "assignment": torch.from_numpy(d["assignment"].astype(np.float32)),  # [J,C]
            "s_j": torch.from_numpy(d["s_j"].astype(np.float32)),               # [J,D]
            "joint_mask": torch.from_numpy(d["joint_mask"].astype(np.bool_)),
            "rest_offsets": torch.from_numpy(d["rest_offsets"].astype(np.float32)),
            "anytop_mean": torch.from_numpy(d["anytop_mean"].astype(np.float32)),
            "anytop_std": torch.from_numpy(d["anytop_std"].astype(np.float32)),
            "parent_indices": [int(p) for p in d["parent_indices"].tolist()],
            "num_joints": int(d["num_joints"]),
            "caption_emb": torch.from_numpy(cap_emb.astype(np.float32)),  # [d_text]
            "caption_token_emb": torch.from_numpy(
                cap_tok.astype(np.float32)),                     # [L,768] fixed or [n,dim] ragged
            "caption_token_mask": torch.from_numpy(cap_msk.astype(np.bool_)),
            "has_text": bool(d["has_text"]),
            "object_type": row["object_type"],
            "text": row.get("text", ""),
        }


_TEXT_RAGGED_KEYS = ("caption_token_emb", "caption_token_mask")


def token_collate(batch: list[dict]) -> dict:
    out: dict = {}
    keys = batch[0].keys()
    for k in keys:
        v0 = batch[0][k]
        if isinstance(v0, torch.Tensor):
            if (k in _TEXT_RAGGED_KEYS
                    and len({b[k].shape[0] for b in batch}) > 1):
                # Ragged caption rows (LLM2Vec cache stores true lengths): pad to
                # the batch max with zeros / False. Uniform-length batches — every
                # legacy T5 cache — never enter this branch, so the existing
                # fixed-L path stays byte-identical.
                Lm = max(b[k].shape[0] for b in batch)
                padded = []
                for b in batch:
                    t = b[k]
                    if t.shape[0] < Lm:
                        pad = torch.zeros((Lm - t.shape[0],) + tuple(t.shape[1:]),
                                          dtype=t.dtype)
                        t = torch.cat([t, pad], dim=0)
                    padded.append(t)
                out[k] = torch.stack(padded)
            else:
                out[k] = torch.stack([b[k] for b in batch])
        elif isinstance(v0, bool):
            out[k] = torch.tensor([b[k] for b in batch], dtype=torch.bool)
        elif isinstance(v0, int):
            out[k] = torch.tensor([b[k] for b in batch], dtype=torch.int64)
        else:
            out[k] = [b[k] for b in batch]
    return out
