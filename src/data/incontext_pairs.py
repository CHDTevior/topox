"""[demo | target] in-context pairs over a single skeleton, for the v2 motion DiT.

WHAT ONE ITEM IS
    Two DIFFERENT clips of the SAME skeleton, concatenated along time:
        x         = [ demo slot 64 | target slot 240 ]       T = 304
        is_target = [ False        | True on REAL target frames only ]
    The demo supplies "how THIS rig moves"; the text supplies "which action". Because demo and
    target are the same rig, the joint axis is aligned within an item for free -- no cross-skeleton
    joint correspondence is needed anywhere. That is why an unseen rig costs nothing structurally
    at inference: it walks the identical forward pass.

SLOT SIZES.
    target 240: covers the LONGEST TrueBones clip (measured max 237, Alligator___DIe_16), so no
    clip is ever truncated and the caption always describes exactly the frames being trained on --
    this deletes the prefix-generation caveat outright (previously 45% of clips >=96 lost their
    tail). The price is compute, not correctness: the median clip is 90 frames, so on average ~62%
    of the target slot is masked padding (frame_valid False -> excluded from attention keys and
    from every loss denominator). Length-bucketed batching can claw that back later if needed.
    demo 64: 3.2 s of context at 20 fps, a few gait cycles; 70.3% of clips fill it fully. The demo
    is context, not a target -- longer demos cost quadratic temporal attention for diminishing
    information (F5-TTS clones a voice from seconds of reference for the same reason).

K = 1 DEMO, and that is a data constraint, not a preference: the smallest rigs have only 2 clips
    (Chicken 2, Flamingo 3, Parrot2 3), so K=2 would drop rigs entirely. At inference the user may
    supply several demos -- run the sampler once per demo rather than changing the layout, because
    a different K at test time reintroduces the train/test mismatch this design exists to remove.

THE THREE BUCKETS, all built from the FROZEN protocol in data/holdout_splits_v1/ (sealed in
protocol/SEAL.json, split at canonical-topology level -- 179 trees, 35 held). Do NOT re-derive a
split here: a split by object_type would leak, because distinct object types can share one
canonical topology.

    training : target from train.txt              demo from train.txt        694 TB clips / 49 rigs
    bucket A : target from val.txt                demo from train.txt         55 TB clips / 49 rigs
               -> skeleton SEEN, target clip UNSEEN. Can text drive a new action?
    bucket B : target from held_representative    demo from the SAME file    150 TB clips /  8 rigs
               -> skeleton NEVER SEEN. Its demos must come from itself; that is the whole premise.
    stress   : same as B but held_stress.txt      125 TB clips / 7 rigs (Spider, Dragon, Crab, ...)

    A upper-bounds B. B - A is the generalisation cost, which the previous architecture could not
    measure because it had no demo path and the two failure modes were entangled.
"""
from __future__ import annotations

import os
import zlib
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from src.data.anytop_dataset import AnyTopDataset

DEMO_FRAMES = 64
REST_DEMO_CLAMP = 5.0     # rest-demo input clamp (codex 2026-08-21 (b))
TARGET_FRAMES = 240   # >= TrueBones max clip length 237: nothing truncated, caption always matches
GEODESIC_CLIP = 8.0      # hop distances reach ~20; an unclipped bias swamps the attention logits
PAD_BIAS = -1e4
UPDOWN_CLIP = 15         # graph-v2 directional tables: up/down hop indices 0..15 (vs the scalar
#                          bias's clip at 8 -- deep chains stay distinguishable twice as far)


def _graph_v2_tables(parents, offsets, geodesic_raw):
    """Per-rig graph-v2 statics: structural joint features [J,8] + LCA-decomposed directional
    hop matrix [J,J,2] (up-steps to the lowest common ancestor, then down-steps).

    SELF-CHECK: up + down must equal the served Floyd geodesic EXACTLY (pre-clip) -- one
    assert catches any joint-order or parent-table mismatch at rig-cache build time instead of
    letting a silently wrong topology signal train for 46k steps.

    Feature layout (all scale-free; mean bone length of THIS rig is the unit):
      0:3 rest offset direction (unit; zeros for root/zero-length)
      3   log1p(bone length / mean bone)
      4   depth from root in hops / 16 (uncapped rigs reach ~20; >1 is fine, it is a feature)
      5   log1p(physical path length to root / mean bone) / 4
      6   child count, clipped at 4, / 4
      7   is-leaf flag
    """
    J = len(parents)
    par = [int(p) for p in parents]
    depth = np.zeros(J, np.int64)
    for j in range(1, J):
        depth[j] = depth[par[j]] + 1
    anc = []
    for j in range(J):
        c, k = {}, j
        while k >= 0:
            c[k] = depth[j] - depth[k]      # up-steps from j to ancestor k
            k = par[k]
        anc.append(c)
    ud = np.zeros((J, J, 2), np.int64)
    for i in range(J):
        ai = anc[i]
        for j in range(J):
            aj = anc[j]
            best, up = -1, 0
            for k, u in ai.items():         # ancestors of i, nearest first is not guaranteed --
                if k in aj and depth[k] > best:   # pick the DEEPEST common ancestor (the LCA)
                    best, up = depth[k], u
            ud[i, j, 0] = up
            ud[i, j, 1] = depth[j] - best         # down-steps = depth(j) - depth(LCA)
    hops = ud.sum(-1)
    geo = np.asarray(geodesic_raw, dtype=np.int64)[:J, :J]
    if not np.array_equal(hops, geo):
        bad = int(np.abs(hops - geo).max())
        raise AssertionError(f"graph-v2 LCA decomposition disagrees with Floyd geodesic "
                             f"(max |up+down - geo| = {bad}); joint order or parents corrupted")
    off = np.asarray(offsets, dtype=np.float64)[:J]
    blen = np.linalg.norm(off, axis=-1)
    mean_bone = max(float(blen[1:].mean()) if J > 1 else 1.0, 1e-6)
    unit = np.zeros((J, 3))
    nz = blen > 1e-8
    unit[nz] = off[nz] / blen[nz, None]
    phys = np.zeros(J)
    for j in range(1, J):
        phys[j] = phys[par[j]] + blen[j]
    n_child = np.zeros(J)
    for j in range(1, J):
        n_child[par[j]] += 1
    feats = np.zeros((J, 8), np.float32)
    feats[:, 0:3] = unit
    feats[:, 3] = np.log1p(blen / mean_bone)
    feats[:, 4] = depth / 16.0
    feats[:, 5] = np.log1p(phys / mean_bone) / 4.0
    feats[:, 6] = np.clip(n_child, 0, 4) / 4.0
    feats[:, 7] = (n_child == 0).astype(np.float32)
    return feats, np.clip(ud, 0, UPDOWN_CLIP)


def read_split(splits_dir, name) -> set:
    """Clip names (no .npy) listed in one file of the frozen protocol.

    The '#' guard is load-bearing: every file starts with a provenance comment naming the artifact
    sha256, and without skipping it that line enters all four sets, making them appear to intersect.
    Matches how the dataset itself reads these files.
    """
    p = Path(splits_dir) / f"{name}.txt"
    return {ln.strip().replace(".npy", "") for ln in p.read_text().splitlines()
            if ln.strip() and not ln.lstrip().startswith("#")}


def truebones_types(cond_keys) -> list:
    """TrueBones object types = everything that is neither Planet-Zoo nor HumanML3D."""
    return sorted(k for k in cond_keys if not k.startswith("PZ_") and not k.startswith("HML3D"))


def pzh_types(cond_keys) -> list:
    """Planet-Zoo + HumanML3D object types (the run-3 scale corpus; no TrueBones).
    Measured 2026-08-19: 311 PZ rigs + 1 human = 312 types; train side 89,543 clips
    (PZ 71.5% / human 28.5%), every rig has >=59 clips, J max 102, clip T max 299."""
    return sorted(k for k in cond_keys if k.startswith("PZ_") or k.startswith("HML3D"))


class InContextPairs(Dataset):
    """Serves [demo | target] items. `base` must be an AnyTopDataset built with split="all" so both
    pools are visible; membership is decided here by the frozen name lists, never by re-splitting.

    Build `base` with joint_semantics=<npz> so the per-rig joint-order hash is CHECKED
    (anytop_dataset.py:1279-1283). Loading that table by hand bypasses the check and would pair
    embeddings with the wrong joints after any re-ordering, silently.
    """

    def __init__(self, base: AnyTopDataset, target_names, demo_names, *,
                 object_types=None, demo_frames=DEMO_FRAMES, target_frames=TARGET_FRAMES,
                 balance_skeletons=True, seed=0, emit_fk_fields=False, emit_graph_v2=False,
                 identity_p=0.0, emit_ref_text=False, demo_rest=False):
        self.base = base
        self.Td, self.Tt = int(demo_frames), int(target_frames)
        self.balance = bool(balance_skeletons)
        self.seed = int(seed)
        # gamma_fk (Kimodo Eq.1 term 7) needs the target rig's de-normalization stats + FK
        # skeleton in the SERVED joint order. The base item already carries all four exactly as
        # the preflight FK gate consumes them; flag-gated so pre-gamma7 runs see a byte-identical
        # batch dict.
        self.emit_fk_fields = bool(emit_fk_fields)
        # graph-v2 (2026-08-19): structural joint features + LCA-directional hop matrices,
        # computed once per RIG (statics; identical for every clip of a rig) and cached. Same
        # flag-gating contract: off = byte-identical batches.
        self.emit_graph_v2 = bool(emit_graph_v2)
        self._g2cache = {}
        # UMO SOURCE_IDENTITY analogue (manifest_dataset.py:289-304 mechanism): with prob p the
        # target IS the demo clip (different windows of the same clip) and the caption embedding
        # is ZEROED -- teaches "the demo/anchor carries content" without the (demo+text-demo)
        # guidance collapse. Off by default; the anti-shortcut demo!=target rule stays intact for
        # the normal branch (identity samples are explicit, flagged via is_identity).
        self.identity_p = float(identity_p)
        # F5-TTS reference-transcript analogue; flag-gated so an unset run is byte-identical.
        self.emit_ref_text = bool(emit_ref_text)
        # 1-FRAME REST DEMO (user 2026-08-21): the demo slot carries the rig's rest pose instead of
        # a window of another clip. The demo then holds NO motion content, so the run isolates
        # "can the raw-space DiT generate coherent motion from text + skeleton", with the
        # demo-interpretation variable removed. Requires demo_frames=1 and a base exposing
        # rest_frame_normalized().
        self.demo_rest = bool(demo_rest)
        if self.demo_rest and int(demo_frames) != 1:
            raise ValueError(f"demo_rest needs demo_frames=1, got {demo_frames}")
        # Corpus-specific post-crop hook (KTJD-17 crop contract, codex round-S0): KTJD requires
        # smooth-root XZ to be re-based at EVERY crop boundary (loader.py:99-122 is the normative
        # crop; generic _crop only slices). A base dataset that needs window-level fixups exposes
        # `postcrop_window(window, valid_mask) -> window`; absent hook = byte-identical batches.
        self._postcrop = getattr(base, "postcrop_window", None)
        keep = set(object_types) if object_types is not None else None

        tgt_pool, demo_pool = defaultdict(list), defaultdict(list)
        for i, s in enumerate(base.samples):
            ot = s["object_type"]
            if keep is not None and ot not in keep:
                continue
            nm = Path(s["path"]).name.replace(".npy", "")
            if nm in target_names:
                tgt_pool[ot].append(i)
            if nm in demo_names:
                demo_pool[ot].append(i)

        # A target is usable only if SOME demo differs from it. Keeping a target whose only demo is
        # itself would let __getitem__ fall back to demo == target: the clip becomes its own clean
        # demonstration, an identity-copy shortcut that would also make a demo-effect gate PASS for
        # the wrong reason. Filter per TARGET, not per rig.
        self.by_type = {}
        self.n_dropped_self_only = 0
        for ot, tg in tgt_pool.items():
            dm = sorted(demo_pool.get(ot, []))
            if not dm:
                continue
            legal = [t for t in sorted(tg) if any(d != t for d in dm)]
            self.n_dropped_self_only += len(tg) - len(legal)
            if legal:
                self.by_type[ot] = {"targets": legal, "demos": dm}

        self.types = sorted(self.by_type)
        self.index = [(ot, i) for ot in self.types for i in self.by_type[ot]["targets"]]
        # base is built with split="all", which SKIPS AnyTopDataset's own train/val/held
        # disjointness guards (anytop_dataset.py:661 takes the non-file branch). Re-assert here so a
        # mis-specified pair of name lists cannot silently score a model on clips it trained on.
        # ANY overlap is rejected -- a 30% leak disqualifies a bucket as surely as a 100% one. The
        # single legal exception is passing the SAME set object for both sides (bucket B: an unseen
        # rig's demos come from itself; that is the premise, not a leak).
        if target_names is not demo_names:
            overlap = set(target_names) & set(demo_names)
            if overlap:
                raise ValueError(
                    f"target/demo name lists overlap on {len(overlap)} clips "
                    f"(e.g. {sorted(overlap)[:3]}); a partially leaked bucket scores the model on "
                    f"clips its demo pool trained on. Pass the identical set object only for "
                    f"self-demo buckets.")

    # ---- introspection used by the smoke and by reports ----
    def pair_count(self):
        # |targets| x |demos| minus the self-pairs, counted directly: the nested form was
        # O(targets x demos), which on a 89.5k-clip corpus with one rig holding ~2e4 human clips
        # is ~4e8 Python iterations -- minutes of pure reporting before training starts.
        n = 0
        for v in self.by_type.values():
            dm = set(v["demos"])
            n += len(v["targets"]) * len(v["demos"]) - sum(1 for t in v["targets"] if t in dm)
        return n

    def __len__(self):
        return len(self.index)

    def _worker_rng(self):
        """Per-(rank, worker) RNG streams, full-width.

        Failure modes this replaces, in the order reviews forced them out:
          - one shared self.rng: every forked DataLoader worker replays the SAME stream;
          - self.seed + worker_id: repeats across DDP ranks and across restarts;
          - xor-then-&0xFFFFFFFF: truncates to 32 bits, so distinct (seed, rank) pairs can collide.
        np.random.default_rng accepts a SEQUENCE of ints as entropy and hashes it at full width, so
        seed with [info.seed, rank, self.seed]: info.seed is PyTorch's base_seed + worker_id (fresh
        per epoch for non-persistent workers), rank comes from the environment, nothing truncated.
        With persistent_workers=True the stream simply CONTINUES across epochs -- no reset, hence no
        repetition -- which is acceptable; pass an explicit epoch only if bit-reproducible per-epoch
        streams are ever needed. The no-worker path folds rank too, so single-process DDP ranks
        still differ.
        """
        rank = int(os.environ.get("RANK", "0"))
        info = torch.utils.data.get_worker_info()
        key = ("main", rank) if info is None else (info.id, int(info.seed), rank)
        if getattr(self, "_wrng_key", None) != key:
            self._wrng_key = key
            ent = [self.seed, rank] if info is None else [int(info.seed), rank, self.seed]
            self._wrng = np.random.default_rng(ent)
        return self._wrng

    def _pick(self, i, rng):
        """Balanced mode draws a rig uniformly first, so Trex (72 clips) cannot outweigh Hamster (6)
        12:1 -- we are training "any skeleton", not a Trex specialist. It makes __getitem__
        nondeterministic by design; an epoch is a fixed number of draws, not a cover of the index."""
        if not self.balance:
            return self.index[i]
        ot = self.types[int(rng.integers(len(self.types)))]
        tg = self.by_type[ot]["targets"]
        return ot, int(tg[int(rng.integers(len(tg)))])

    def _crop(self, x, n_valid, want, rng, random_window):
        """Take `want` frames; pad at the end if the clip is shorter.

        DEMO uses a random window: it only has to show how the rig moves, and 12% of training clips
        exceed 192 frames whose tails would otherwise never be seen.

        TARGET uses the HEAD (random_window=False). With the slot at 240 >= the TrueBones max
        (237) no clip is truncated, so head-vs-random is currently moot -- the head rule is kept as
        the fail-safe for any longer future data, because the caption describes the whole clip and
        the head is the one window guaranteed to start where the described action starts. Temporal
        resampling is deliberately NOT used as an alternative: ch9:12 are per-frame velocities and
        ch12 is a binary contact flag, both corrupted by a naive resample.
        """
        J, C = x.shape[1], x.shape[2]
        n = int(min(n_valid, x.shape[0]))
        out = np.zeros((want, J, C), dtype=np.float32)
        vm = np.zeros((want,), dtype=bool)
        if n >= want:
            s = int(rng.integers(0, n - want + 1)) if random_window else 0
            out[:] = x[s:s + want]; vm[:] = True
        else:
            out[:n] = x[:n]; vm[:n] = True
        return out, vm

    def _raw(self, idx):
        it = self.base[idx]
        J, T = int(it["num_joints"]), int(it["num_frames"])
        x = np.asarray(it["anytop_x"])[:J, :, :T].transpose(2, 0, 1)   # [T,J,13] normalised
        return it, x, J, T

    def __getitem__(self, i):
        rng = self._worker_rng()
        ot, tgt_idx = self._pick(i, rng)
        demos = [d for d in self.by_type[ot]["demos"] if d != tgt_idx]
        if not demos:      # cannot happen: targets without a distinct demo are dropped at build
            raise AssertionError(f"{ot}: target {tgt_idx} has no distinct demo")
        demo_idx = int(demos[int(rng.integers(len(demos)))])
        is_identity = self.identity_p > 0 and float(rng.random()) < self.identity_p
        if is_identity:
            demo_idx = tgt_idx          # same clip: demo window (random) vs target head window

        t_item, t_x, J, t_T = self._raw(tgt_idx)
        if (self.demo_rest and not self.emit_ref_text
                and not getattr(self.base, "random_caption", False)):
            # The demo slot carries the rig's REST POSE; the demo clip's motion is loaded and then
            # thrown away three lines down. That doubled this dataset's npz traffic -- a full pass
            # over the 89.5k-clip corpus paid for 89.5k reads nobody consumed -- to buy a
            # joint-count check that same-rig clips satisfy by construction (joint-count
            # augmentation is off for in-context pairs, and the check would still fire for the
            # non-rest path). d_item is only needed for the reference-transcript branch.
            # random_caption guard (codex 2026-08-21 (d)): Ktjd17Base.__getitem__ advances its own
            # caption RNG, so skipping the read is NOT stream-neutral when captions are drawn at
            # random -- my "_raw consumes no RNG" claim held only for this run's settings.
            d_item = t_item if is_identity else None
        else:
            d_item, d_x, dJ, d_T = self._raw(demo_idx)
            if dJ != J:
                raise ValueError(f"{ot}: demo has {dJ} joints, target {J} -- joint-count "
                                 f"augmentation must be off for in-context pairs")

        if self.demo_rest:
            d_crop = np.asarray(self.base.rest_frame_normalized(ot),
                                dtype=np.float32)[None, :J]           # [1,J,C]
            # The rest pose is a REFERENCE, not a motion sample, so it need not lie inside the
            # motion distribution its statistics describe. Four rigs have a root rot6d component
            # that is numerically zero on every stored frame, so the analytic rest identity
            # (ch3 = 1) normalized to 1/std -- 10,000 under the old 1e-4 floor, ~20 under 0.05.
            # The demo slot is pure context (cfm_loss masks it out of every term), so clamping it
            # costs no supervision, and it keeps an out-of-distribution token from dominating the
            # gradient of x_in, which is a bare nn.Linear with no preceding LayerNorm.
            # Range and placement per codex gpt-5.6-terra/xhigh 2026-08-21 (b).
            np.clip(d_crop, -REST_DEMO_CLAMP, REST_DEMO_CLAMP, out=d_crop)
            d_valid = np.ones(1, dtype=bool)
        else:
            d_crop, d_valid = self._crop(d_x, d_T, self.Td, rng, random_window=True)
        t_crop, t_valid = self._crop(t_x, t_T, self.Tt, rng, random_window=False)
        if self._postcrop is not None:
            # Each window is its own crop under the corpus contract (demo and target re-based
            # INDEPENDENTLY -- they are different clips, and even identity pairs use different
            # windows). The target head window is a no-op re-base today (full view is built with
            # crop_start=0), kept unconditional as the fail-safe for any future window policy.
            d_crop = self._postcrop(d_crop, d_valid, ot)
            t_crop = self._postcrop(t_crop, t_valid, ot)

        x = np.concatenate([d_crop, t_crop], axis=0)
        # is_target marks REAL target frames only: padding must not receive the mask token, and
        # must not be counted as a legitimate zero target (that is the direct route to a model
        # that emits the per-frame mean).
        is_target = np.concatenate([np.zeros(self.Td, bool), t_valid])
        frame_valid = np.concatenate([d_valid, t_valid])

        geo = np.asarray(t_item["geodesic_dist"])[:J, :J].astype(np.float32)
        out = {
            "x": torch.from_numpy(x),
            "is_target": torch.from_numpy(is_target),
            "frame_valid": torch.from_numpy(frame_valid),
            "geodesic": torch.from_numpy(np.clip(geo, 0.0, GEODESIC_CLIP)),
            "object_type": ot,
            "motion_id": str(t_item.get("motion_id", tgt_idx)),
            "demo_id": str(demo_idx),
            "n_joints": J,
        }
        # TARGET caption -> the global AdaLN text condition (unchanged).
        if t_item.get("caption_emb") is not None:
            emb = torch.as_tensor(np.asarray(t_item["caption_emb"])).float()
            out["text"] = torch.zeros_like(emb) if is_identity else emb
        # DEMO caption -> the F5-TTS reference-transcript analogue (2026-08-20, user: "和它对齐").
        # F5/E2 feed [ref_text + gen_text] alongside [ref_mel + masked span]: the model is TOLD what
        # the reference is saying, which is how it factors the reference's CONTENT out and keeps
        # only its STYLE (timbre). We had deliberately withheld this -- the old comment here feared
        # the text pathway would start describing the demo -- but F5 prevents exactly that by
        # POSITION, not by withholding: ref text sits over the reference frames, gen text over the
        # span to fill. The model consumes it per-frame on that layout (see InContextMotionDiT
        # ref_text), so the two captions can never be confused for one another.
        if self.emit_ref_text and d_item is not None and d_item.get("caption_emb") is not None:
            demb = torch.as_tensor(np.asarray(d_item["caption_emb"])).float()
            # an identity sample IS the target clip, so its "reference caption" is the request --
            # zero it with the target's, or the dropped text leaks back in through this door.
            out["demo_text"] = torch.zeros_like(demb) if is_identity else demb
        out["is_identity"] = bool(is_identity)
        sem = t_item.get("joint_semantics")          # order-hash checked inside the dataset
        if sem is not None:
            out["joint_sem"] = torch.as_tensor(np.asarray(sem))[:J].float()
        if self.emit_fk_fields:
            # Same fields, same [:J] slice, same source item as scripts/v2_preflight_bz1.py's
            # FK==RIC gate -- i.e. already in the served (permuted) joint order. Only the TARGET
            # item's rig matters: gamma_fk applies to target frames, and demo shares the rig.
            out["anytop_mean"] = torch.as_tensor(np.asarray(t_item["anytop_mean"])[:J]).float()
            out["anytop_std"] = torch.as_tensor(np.asarray(t_item["anytop_std"])[:J]).float()
            out["parents"] = torch.as_tensor(
                np.asarray(t_item["parent_indices"][:J], dtype=np.int64))
            out["rest_offsets"] = torch.as_tensor(np.asarray(t_item["rest_offsets"])[:J]).float()
            if "R_rest_global" in t_item:
                # KTJD-17 gamma7: the FK path composes cont6d deltas with the rig's global rest
                # rotations (decoder.py:70); the 13ch corpus has no such field, so flag-free.
                out["R_rest_global"] = torch.as_tensor(
                    np.asarray(t_item["R_rest_global"])[:J]).float()
        if self.emit_graph_v2:
            if ot not in self._g2cache:
                # geodesic_raw is the UN-clipped served-order Floyd matrix -- the self-check
                # inside _graph_v2_tables pins the LCA decomposition to it exactly.
                self._g2cache[ot] = _graph_v2_tables(
                    np.asarray(t_item["parent_indices"][:J], dtype=np.int64),
                    np.asarray(t_item["rest_offsets"])[:J],
                    np.asarray(t_item["geodesic_dist"])[:J, :J])
            feats, ud = self._g2cache[ot]
            out["struct_feats"] = torch.from_numpy(feats)
            out["updown"] = torch.from_numpy(ud)
        return out


def collate(batch):
    """Pad the joint axis to the batch max; frames are already fixed length.

    joint_valid MUST reach the loss. Padded joints are numerically zero, and counting them as valid
    would let padding dominate the per-group means: a J=9 rig in a batch whose max is 142 would be
    94% padding.
    """
    B, T, C = len(batch), batch[0]["x"].shape[0], batch[0]["x"].shape[2]
    Jm = max(b["n_joints"] for b in batch)

    x = torch.zeros(B, T, Jm, C)
    joint_valid = torch.zeros(B, Jm, dtype=torch.bool)
    bias = torch.full((B, Jm, Jm), PAD_BIAS)
    has_sem = "joint_sem" in batch[0]
    sem = torch.zeros(B, Jm, batch[0]["joint_sem"].shape[1]) if has_sem else None

    for k, b in enumerate(batch):
        J = b["n_joints"]
        x[k, :, :J] = b["x"]
        joint_valid[k, :J] = True
        bias[k, :J, :J] = -b["geodesic"]          # nearer joints attend more; padding stays at -1e4
        if has_sem:
            sem[k, :J] = b["joint_sem"]

    out = {
        "x": x,
        "joint_valid": joint_valid,
        "joint_bias": bias,
        "is_target": torch.stack([b["is_target"] for b in batch]),
        "frame_valid": torch.stack([b["frame_valid"] for b in batch]),
        "object_type": [b["object_type"] for b in batch],
        "motion_id": [b["motion_id"] for b in batch],
        "demo_id": [b["demo_id"] for b in batch],
    }
    if has_sem:
        out["joint_sem"] = sem
    if "text" in batch[0]:
        out["text"] = torch.stack([b["text"] for b in batch])
    if "demo_text" in batch[0]:
        out["demo_text"] = torch.stack([b["demo_text"] for b in batch])
    if "anytop_mean" in batch[0]:
        # gamma_fk fields, padded to Jm. Padded joints never reach the FK chain (the loss slices
        # [:n_joints] per sample), so zero mean/std/offsets and parent -1 are inert placeholders.
        # Channel count follows the corpus (13 for AnyTop, 18 for KTJD's plane-carrying stats).
        Cs = batch[0]["anytop_mean"].shape[1]
        am = torch.zeros(B, Jm, Cs); asd = torch.zeros(B, Jm, Cs)
        par = torch.full((B, Jm), -1, dtype=torch.long); ro = torch.zeros(B, Jm, 3)
        for k, b in enumerate(batch):
            J = b["n_joints"]
            am[k, :J] = b["anytop_mean"]; asd[k, :J] = b["anytop_std"]
            par[k, :J] = b["parents"]; ro[k, :J] = b["rest_offsets"]
        out.update(anytop_mean=am, anytop_std=asd, parents=par, rest_offsets=ro,
                   n_joints=torch.tensor([b["n_joints"] for b in batch], dtype=torch.long))
        if "R_rest_global" in batch[0]:
            rr = torch.zeros(B, Jm, 3, 3)
            for k, b in enumerate(batch):
                rr[k, :b["n_joints"]] = b["R_rest_global"]
            out["R_rest_global"] = rr
    if "struct_feats" in batch[0]:
        # graph-v2 fields, padded to Jm. Padded rows are zeros; padded PAIRS are irrelevant
        # because the -1e4 PAD_BIAS already excludes them from attention, and zero-index lookups
        # into the zero-initialised direction tables contribute exactly 0 at start anyway.
        sf = torch.zeros(B, Jm, batch[0]["struct_feats"].shape[1])
        ud = torch.zeros(B, Jm, Jm, 2, dtype=torch.long)
        for k, b in enumerate(batch):
            J = b["n_joints"]
            sf[k, :J] = b["struct_feats"]
            ud[k, :J, :J] = b["updown"]
        out.update(struct_feats=sf, updown=ud)
    out["valid"] = out["frame_valid"][:, :, None] & joint_valid[:, None, :]      # [B,T,Jm]
    return out
