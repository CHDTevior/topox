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
from src.data.anytop_dataset import _STD_FLOOR
from torch.utils.data import Dataset

from src.data.anytop_dataset import AnyTopDataset

DEMO_FRAMES = 64
REST_DEMO_CLAMP = 5.0     # rest-demo input clamp (codex 2026-08-21 (b))
TARGET_FRAMES = 240   # >= TrueBones max clip length 237: nothing truncated, caption always matches
GEODESIC_CLIP = 8.0      # hop distances reach ~20; an unclipped bias swamps the attention logits
PAD_BIAS = -1e4
UPDOWN_CLIP = 15         # graph-v2 directional tables: up/down hop indices 0..15 (vs the scalar
#                          bias's clip at 8 -- deep chains stay distinguishable twice as far)


def _world_rest_feats(parents, offsets, P_rest):
    """R1/R2 (2026-09-25): [J, 6] float32 world-frame rest descriptor appended to the graph-v2 struct features when
    struct_world_rest is on -- what the model never saw before (P_rest entered the FK loss only):
      0:3  u_j = (P_rest[j] - P_rest[parent(j)]) normalised: the world direction of bone j at rest (root and zero-length
           bones: zeros);
      3:6  P_rest[j] with the root's XZ subtracted, divided by the rig's mean bone length |offset| (height kept: the rest
           pose is grounded at the data level, so this is height above ground in bone lengths), then radially
           log-compressed, c -> c * log1p(|c|) / |c| (direction kept, invertible).
    Read off the rest positions themselves rather than R_rest @ offset: the skeleton files' parent-frame offsets are not
    FK-consistent with P_rest on every rig, and an augmented item's transform rebuilds P_rest by FK of its (possibly
    rotated) rest rotations, so on both paths P_rest IS the served rest pose and u follows it exactly."""
    par = np.asarray(parents, dtype=np.int64); J = len(par)
    off = np.asarray(offsets, dtype=np.float64)[:J]
    P = np.asarray(P_rest, dtype=np.float64)[:J]
    blen = np.linalg.norm(off, axis=-1)
    mean_bone = max(float(blen[1:].mean()) if J > 1 else 1.0, 1e-6)
    u = np.zeros((J, 3))
    if J > 1:
        d = P[1:] - P[par[1:]]
        n = np.linalg.norm(d, axis=1)
        ok = n > 1e-8
        u[1:][ok] = d[ok] / n[ok, None]
    c = (P - np.array([P[0, 0], 0.0, P[0, 2]])[None]) / mean_bone
    # radial log compression: a joint 20 bone lengths above the ground lands at 3.0 instead of 20, keeping these columns O(1)
    # like the graph-v2 table's log1p(...) columns (reviewer 2026-09-25 P2-2: max 21, median 11 on the UniML3D v2 rigs; the
    # bias-free struct_rest_in would have been dominated by them at init)
    r = np.linalg.norm(c, axis=1)
    c = c * np.where(r > 0, np.log1p(r) / np.maximum(r, 1e-12), 1.0)[:, None]
    return np.concatenate([u, c], axis=1).astype(np.float32)


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


def draw_cdf_for(draw_types, by_type, alpha):
    """Cumulative draw distribution over `draw_types` with weight (target count)^alpha per entry; None at alpha 0 so the
    caller keeps the original uniform draw (and its RNG consumption). The last cell is pinned to 1.0 so a uniform u in
    [0, 1) always lands on an entry."""
    alpha = float(alpha)
    if alpha < 0:
        raise ValueError(f"balance_alpha must be >= 0, got {alpha}")
    if alpha == 0.0:
        return None
    w = np.array([len(by_type[t]["targets"]) ** alpha for t in draw_types], dtype=np.float64)
    if not (np.isfinite(w).all() and w.sum() > 0):
        raise ValueError("balance_alpha: draw weights are not finite and positive")
    cdf = np.cumsum(w / w.sum())
    cdf[-1] = 1.0
    return cdf


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
                 rig_multiplicity=None, epoch_draws=None, balance_alpha=0.0,
                 identity_p=0.0, emit_ref_text=False, demo_rest=False, augment=None, emit_spectral=0,
                 spectral_hks=False, rest_demo_self_pairs=False, struct_world_rest=False):
        self.base = base
        # R1/R2 (2026-09-25): the world-frame rest descriptor (_world_rest_feats) is appended to struct_feats (8 -> 14).
        # Needs emit_graph_v2 (it extends that table). Off = byte-identical items.
        self.struct_world_rest = bool(struct_world_rest)
        if self.struct_world_rest and not emit_graph_v2:
            raise ValueError("struct_world_rest extends the graph-v2 struct_feats table: it needs emit_graph_v2=True")
        self._wrcache = {}
        # skeleton-robustness augmentation (src/data/ktjd17_augment.py, user 2026-09-07): per-sample
        # sub-skeleton / rest-convention / description-noise / statistics perturbations, applied to the
        # target AND its demo. KTJD-17 only (needs static_masks + the FK fields). None or p == 0 = off,
        # byte-identical items; active = every item also carries its own channel_valid [J,17].
        self.aug = augment if (augment is not None and augment.active) else None
        if self.aug is not None and not hasattr(base, "static_masks"):
            raise ValueError("skeleton augmentation needs a KTJD-17 base (static_masks / FK fields)")
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
        # spectral joint RoPE (2026-09-15): the K smallest non-trivial Laplacian eigenvectors of the SERVED tree,
        # [J,K] per item (src/data/skeleton_spectral.py, UniMate's compute_laplacian_eigenvectors), cached per rig
        # like the graph-v2 statics; an augmented item's tree is its own. 0 = off, byte-identical batches.
        self.emit_spectral = int(emit_spectral)
        # H1 (2026-09-24): with spectral_hks the served coordinates are the tree's heat-kernel signature at K scales
        # (skeleton_spectral.heat_kernel_signature: sign- and basis-invariant, every non-trivial mode kept, trace-normalised
        # per rig) instead of
        # its K eigenvectors -- same key, same [J,K] shape, same per-rig cache, an augmented item's tree still its own.
        # Off = byte-identical items.
        self.spectral_hks = bool(spectral_hks)
        if self.spectral_hks and self.emit_spectral <= 0:
            raise ValueError("spectral_hks needs emit_spectral=K > 0: it selects WHICH spectral coordinates are served")
        self._speccache = {}
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
        # A corpus of mostly ONE-CLIP rigs (UniML3D: 5,006 of 5,263 rigs hold a single motion) loses most of its
        # training set to the distinct-demo rule below -- 6,611 clips collapse to 1,544. Under the rest demo that rule
        # is buying nothing: the demo clip is not read at all, the demo slot carries the rig's rest pose, so a target
        # paired with ITSELF is not an identity-copy shortcut, it is the same rest frame every other target of that rig
        # gets. The flag is therefore legal only under exactly the conditions in which the demo clip is never opened
        # (user 2026-09-16: "就不需要配对器了，我们已经是用 rest pose 了"). Default off = byte-identical pair sets, and
        # on the active corpus the rule drops nothing anyway (measured: 0 targets on both the train and the val cut).
        self.rest_demo_self_pairs = bool(rest_demo_self_pairs)
        if self.rest_demo_self_pairs:
            if not self.demo_rest:
                raise ValueError("rest_demo_self_pairs needs demo_rest=True: without it the demo clip IS read and a "
                                 "self-demo would hand the target its own motion as the demonstration")
            if emit_ref_text:
                raise ValueError("rest_demo_self_pairs cannot be combined with emit_ref_text: the reference-transcript "
                                 "branch reads the demo clip, so a self-demo would leak the target's own caption")
            if getattr(base, "random_caption", False):
                raise ValueError("rest_demo_self_pairs cannot be combined with a random_caption base: its __getitem__ "
                                 "advances a caption RNG, so the demo clip is still opened")
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
            legal = sorted(tg) if self.rest_demo_self_pairs else [t for t in sorted(tg) if any(d != t for d in dm)]
            self.n_dropped_self_only += len(tg) - len(legal)
            if legal:
                self.by_type[ot] = {"targets": legal, "demos": dm}

        self.types = sorted(self.by_type)
        # BALANCED SAMPLING WITH ONE OVERSIZED TOPOLOGY. `balance_skeletons` draws a rig uniformly,
        # which is right while every rig holds a few hundred clips. A corpus that adds one rig with
        # a hundred rigs' worth of motion needs a middle ground: uniform-over-rigs gives that rig
        # 1/N of the samples and uniform-over-clips lets it take a quarter of every batch. A
        # multiplicity repeats a named rig in the DRAW list only -- `self.types`, `self.by_type` and
        # `self.index` are untouched, so nothing else in the dataset sees a duplicated rig and every
        # rig not named keeps exactly the share it had, up to the normalisation.
        self.rig_multiplicity = dict(rig_multiplicity or {})
        unknown = sorted(set(self.rig_multiplicity) - set(self.types))
        if unknown:
            raise ValueError(f"rig_multiplicity names rigs this cut does not serve: {unknown}")
        if any(int(v) < 1 for v in self.rig_multiplicity.values()):
            raise ValueError(f"rig_multiplicity must be >= 1: {self.rig_multiplicity}")
        self.draw_types = [t for t in self.types
                           for _ in range(int(self.rig_multiplicity.get(t, 1)))]
        # DRAW WEIGHTS (S1, user 2026-09-24). A rig's draw is weighted by (its target count)^alpha: alpha 0 is the
        # uniform-over-rigs rule above -- and consumes the RNG exactly as before, so every existing arm's batches stay
        # byte-identical -- alpha 1 is uniform over clips, alpha 0.5 is UniMate's sampler_alpha. On UniML3D 74% of the
        # training clips sit alone on their rig, so uniform-over-rigs starves the multi-clip rigs, the only place "same
        # rig, different text" is ever seen: on the common cut a 91-clip rig's clip is drawn 0.11x per epoch against
        # 10.2x for a lone clip; alpha 0.5 lifts it to 1.0x while lone clips stay at 9.7x. Multiplicity still applies
        # (a repeated entry is weighted once per repeat).
        self.balance_alpha = float(balance_alpha)
        self.draw_cdf = draw_cdf_for(self.draw_types, self.by_type, self.balance_alpha) if self.balance else None
        if self.balance_alpha != 0.0 and not self.balance:
            raise ValueError("balance_alpha needs balance_skeletons=True: unbalanced mode covers the index, it does not draw")
        # HOW LONG AN EPOCH IS. _pick's own rule -- "an epoch is a fixed number of draws, not a cover of
        # the index" -- leaves that number equal to the corpus size only by accident, so an arm trained on
        # a larger cut silently takes more optimizer steps per epoch and, because the lr decay horizon is
        # written in epochs, a longer schedule as well. epoch_draws states the number instead, which is
        # what lets the animal+human arm run the control's schedule exactly (codex 2026-09-10 r1 P1-1).
        # Balanced draws only: unbalanced mode indexes self.index[i], where a shorter length would not
        # shorten the epoch but amputate the corpus after the first epoch_draws clips.
        self.epoch_draws = None if epoch_draws in (None, 0) else int(epoch_draws)
        if self.epoch_draws is not None:
            if not self.balance:
                raise ValueError("epoch_draws needs balance_skeletons=True: unbalanced mode reads "
                                 "self.index[i], so a shorter length drops clips rather than draws")
            if self.epoch_draws < 1:
                raise ValueError(f"epoch_draws must be >= 1, got {self.epoch_draws}")
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
        return self.epoch_draws if self.epoch_draws is not None else len(self.index)

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
        if self.draw_cdf is None:                              # alpha 0: the original draw, RNG use unchanged
            ot = self.draw_types[int(rng.integers(len(self.draw_types)))]
        else:
            ot = self.draw_types[min(int(np.searchsorted(self.draw_cdf, rng.random(), side="right")),
                                     len(self.draw_types) - 1)]
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
        if not demos:
            if not self.rest_demo_self_pairs:   # otherwise unreachable: such targets are dropped at build
                raise AssertionError(f"{ot}: target {tgt_idx} has no distinct demo")
            demo_idx = tgt_idx                  # never opened under the rest demo; the slot carries the rest pose
        else:
            demo_idx = int(demos[int(rng.integers(len(demos)))])
        is_identity = self.identity_p > 0 and float(rng.random()) < self.identity_p
        if is_identity:
            demo_idx = tgt_idx          # same clip: demo window (random) vs target head window

        t_item, t_x, J, t_T = self._raw(tgt_idx)
        J0 = J                                    # the rig's joint count before any augmentation
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

        tr = None
        t_con_before = None
        # R2: the rest-convention channel is drawn AFTER the op draw and only when configured (rest_p > 0), so a run
        # without it consumes the RNG exactly as before; a rest-only sample gets a transform with no op (skip_op).
        do_op = self.aug is not None and float(rng.random()) < self.aug.p
        do_rest = self.aug is not None and self.aug.rest_p > 0 and float(rng.random()) < self.aug.rest_p
        if do_op or do_rest:
            from src.data.ktjd17_augment import make_transform, apply_motion, apply_motion_with_contact
            cv0 = np.asarray(self.base.static_masks(ot)["channel_valid"], dtype=bool)[:J0]
            mu0 = np.asarray(t_item["anytop_mean"], dtype=np.float32)[:J0, :17]
            sd0 = np.asarray(t_item["anytop_std"], dtype=np.float32)[:J0, :17]
            # joints that make contact in the target clip are never dropped (raw contact flag > 0.5)
            contact = ((t_x[:t_T, :, 12] * (sd0[None, :, 12] + _STD_FLOOR) + mu0[None, :, 12]) > 0.5).any(0)
            sk = self.base.skeleton(ot)                  # float64 skeleton (item fields are float32 copies)
            tr = make_transform(rng, self.aug, parents=np.asarray(sk["parents"])[:J0],
                                P_rest_global=np.asarray(sk["P_rest_global"])[:J0],
                                R_rest_global=np.asarray(sk["R_rest_global"])[:J0],
                                offset_parent_local=np.asarray(sk["offset_parent_local"])[:J0],
                                channel_valid=cv0, mu=mu0, sd=sd0, contact_joints=contact,
                                fps=30.0,                                   # the corpus rate (validate_schema pins fps_target 30)
                                rest_norm=(self.base.normalization == "rest"),
                                skip_op=not do_op,
                                rest=(do_rest if self.aug.mode == "one_of" else None))
            t_x, t_con_before = apply_motion_with_contact(t_x, tr)
            if not self.demo_rest:
                d_x = apply_motion(d_x, tr)
            J = tr.n_joints

        if self.demo_rest:
            # a PRIVATE copy: the clamp below used to run in place on the cached frame, which was harmless while the
            # clamp was the only consumer (idempotent) but would feed an already-clipped frame to the augmentation
            # (codex 2026-09-07 P1: a 20 -> 5 clipped rotation cell moved the transformed demo by 4 normalized units)
            d_crop = np.array(np.asarray(self.base.rest_frame_normalized(ot))[:J0], dtype=np.float32, copy=True)   # [J,C]
            if tr is not None:
                d_crop = apply_motion(d_crop, tr)
                if tr.rest_channel and tr.mu_rest is not None:
                    # the transformed rig's OWN rest frame: under the rest normalisation its mean IS that frame (make_transform,
                    # mu_rest), so the frame normalises to zero on every cell exactly as a real rig's does
                    # (Ktjd17Base.rest_frame_normalized); non-zero only under a statistics perturbation. apply_motion would
                    # instead serve the SOURCE rig's rest pose described in the new convention (positions relative to the new
                    # rest, deltas Q^T): a frame no rig serves, and one that hands the model every Q_j (reviewer 2026-09-25 P1)
                    d_crop[:, :17] = ((tr.mu_rest - tr.mu) / (tr.sd + _STD_FLOOR)).astype(np.float32)
            d_crop = d_crop[None]                                     # [1,J,C]
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

        lock_den = 0.0
        if t_con_before is not None:
            # skeleton augmentation mode one_of (ktjd17_augment): the contact flags of the recomposed rows were pruned on the
            # FULL clip, where the flag at frame t is judged on the displacement t -> t+1. The served window is the clip's head,
            # so when the clip was truncated its last frame was judged on a frame the model never sees: that frame keeps its
            # served flag. The foot-lock term of an augmented sample divides by the window's PRE-pruning pair count
            # (lock_denominator, fk_torch.ktjd_dynamics_losses), so a pruned pair removes its demand without re-weighting the
            # surviving pairs (codex 2026-09-15 unimate r6 / r7).
            n_served = int(t_valid.sum())
            cb = t_con_before[:n_served]                                                   # [n,J'] bool
            if n_served < t_T:
                on = cb[n_served - 1].astype(np.float32)
                t_crop[n_served - 1, :, 12] = (on - tr.mu[:, 12]) / (tr.sd[:, 12] + _STD_FLOOR)
            lock_den = float((cb[1:] & cb[:-1]).sum())
        x = np.concatenate([d_crop, t_crop], axis=0)
        # is_target marks REAL target frames only: padding must not receive the mask token, and
        # must not be counted as a legitimate zero target (that is the direct route to a model
        # that emits the per-frame mean).
        is_target = np.concatenate([np.zeros(self.Td, bool), t_valid])
        frame_valid = np.concatenate([d_valid, t_valid])

        if tr is not None:
            from src.data.ktjd17_augment import hop_matrix
            geo = hop_matrix(tr.parents)
        else:
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
            "lock_denominator": lock_den,
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
            if tr is not None:
                from src.data.ktjd17_augment import apply_semantics
                out["joint_sem"] = torch.from_numpy(apply_semantics(sem, tr, rng)).float()
            else:
                out["joint_sem"] = torch.as_tensor(np.asarray(sem))[:J].float()
        if self.aug is not None:
            # per-sample static channel mask: the trainer's per-rig LUT cannot describe a sub-skeleton
            cv_now = tr.channel_valid if tr is not None else \
                np.asarray(self.base.static_masks(ot)["channel_valid"], dtype=bool)[:J]
            out["channel_valid"] = torch.from_numpy(np.ascontiguousarray(cv_now))
        if self.emit_fk_fields and tr is not None:
            out["anytop_mean"] = torch.from_numpy(np.concatenate(
                [tr.mu, np.zeros((J, 1), np.float32)], axis=1)).float()          # plane 17 carries no offset
            out["anytop_std"] = torch.from_numpy(np.concatenate(
                [tr.sd, np.ones((J, 1), np.float32) - _STD_FLOOR], axis=1)).float()
            out["parents"] = torch.from_numpy(tr.parents.astype(np.int64))
            out["rest_offsets"] = torch.from_numpy(tr.offsets.astype(np.float32))
            out["R_rest_global"] = torch.from_numpy(tr.R_rest.astype(np.float32))
        elif self.emit_fk_fields:
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
        if self.emit_graph_v2 and tr is not None:
            feats, ud = _graph_v2_tables(tr.parents, tr.offsets, geo)   # this sample's tree, uncached
            if self.struct_world_rest:                                   # the transformed rest (Q_j R_rest_j, rebuilt P)
                feats = np.concatenate([feats, _world_rest_feats(tr.parents, tr.offsets, tr.P_rest)], axis=1)
            out["struct_feats"] = torch.from_numpy(feats)
            out["updown"] = torch.from_numpy(ud)
        elif self.emit_graph_v2:
            if ot not in self._g2cache:
                # geodesic_raw is the UN-clipped served-order Floyd matrix -- the self-check
                # inside _graph_v2_tables pins the LCA decomposition to it exactly.
                self._g2cache[ot] = _graph_v2_tables(
                    np.asarray(t_item["parent_indices"][:J], dtype=np.int64),
                    np.asarray(t_item["rest_offsets"])[:J],
                    np.asarray(t_item["geodesic_dist"])[:J, :J])
            feats, ud = self._g2cache[ot]
            if self.struct_world_rest:
                if ot not in self._wrcache:
                    self._wrcache[ot] = _world_rest_feats(
                        np.asarray(t_item["parent_indices"][:J], dtype=np.int64), np.asarray(t_item["rest_offsets"])[:J],
                        np.asarray(t_item["P_rest_global"])[:J])
                feats = np.concatenate([feats, self._wrcache[ot]], axis=1)
            out["struct_feats"] = torch.from_numpy(feats)
            out["updown"] = torch.from_numpy(ud)
        if self.emit_spectral > 0:
            if tr is not None:
                out["spectral_feats"] = torch.from_numpy(self._spectral_coords(tr.parents))
            else:
                if ot not in self._speccache:
                    self._speccache[ot] = self._spectral_coords(
                        np.asarray(t_item["parent_indices"][:J], dtype=np.int64))
                out["spectral_feats"] = torch.from_numpy(self._speccache[ot])
        return out

    def _spectral_coords(self, parents):
        """[J, K] float32 spectral coordinates of one served tree: its K eigenvectors, or with spectral_hks its
        trace-normalised heat-kernel signature at K scales."""
        from src.data.skeleton_spectral import laplacian_eigenvectors, heat_kernel_signature, hks_scales
        if self.spectral_hks:
            return heat_kernel_signature(parents, hks_scales(self.emit_spectral))
        return laplacian_eigenvectors(parents, self.emit_spectral)[0]


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
        # the pre-pruning contact-pair count of a one_of-augmented sample, 0 otherwise (fk_torch.ktjd_dynamics_losses)
        "lock_denominator": torch.tensor([float(b.get("lock_denominator", 0.0)) for b in batch]),
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
    if "channel_valid" in batch[0]:
        cvb = torch.zeros(B, Jm, batch[0]["channel_valid"].shape[1], dtype=torch.bool)
        for k, b in enumerate(batch):
            cvb[k, :b["n_joints"]] = b["channel_valid"]
        out["channel_valid"] = cvb
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
    if "spectral_feats" in batch[0]:
        # spectral RoPE coordinates padded to Jm with zeros: a padded joint gets the SignNet's angle for the zero vector
        # (one constant rotation), and PAD_BIAS already removes it from every softmax
        sp = torch.zeros(B, Jm, batch[0]["spectral_feats"].shape[1])
        for k, b in enumerate(batch):
            sp[k, :b["n_joints"]] = b["spectral_feats"]
        out["spectral_feats"] = sp
    out["valid"] = out["frame_valid"][:, :, None] & joint_valid[:, None, :]      # [B,T,Jm]
    return out
