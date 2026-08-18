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
TARGET_FRAMES = 240   # >= TrueBones max clip length 237: nothing truncated, caption always matches
GEODESIC_CLIP = 8.0      # hop distances reach ~20; an unclipped bias swamps the attention logits
PAD_BIAS = -1e4


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


class InContextPairs(Dataset):
    """Serves [demo | target] items. `base` must be an AnyTopDataset built with split="all" so both
    pools are visible; membership is decided here by the frozen name lists, never by re-splitting.

    Build `base` with joint_semantics=<npz> so the per-rig joint-order hash is CHECKED
    (anytop_dataset.py:1279-1283). Loading that table by hand bypasses the check and would pair
    embeddings with the wrong joints after any re-ordering, silently.
    """

    def __init__(self, base: AnyTopDataset, target_names, demo_names, *,
                 object_types=None, demo_frames=DEMO_FRAMES, target_frames=TARGET_FRAMES,
                 balance_skeletons=True, seed=0):
        self.base = base
        self.Td, self.Tt = int(demo_frames), int(target_frames)
        self.balance = bool(balance_skeletons)
        self.seed = int(seed)
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
        n = 0
        for v in self.by_type.values():
            for t in v["targets"]:
                n += sum(1 for d in v["demos"] if d != t)
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

        t_item, t_x, J, t_T = self._raw(tgt_idx)
        _, d_x, dJ, d_T = self._raw(demo_idx)
        if dJ != J:
            raise ValueError(f"{ot}: demo has {dJ} joints, target {J} -- joint-count augmentation "
                             f"must be off for in-context pairs")

        d_crop, d_valid = self._crop(d_x, d_T, self.Td, rng, random_window=True)
        t_crop, t_valid = self._crop(t_x, t_T, self.Tt, rng, random_window=False)

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
        # text is the TARGET's caption only; feeding the demo's caption would let the text pathway
        # learn to describe the demo instead of the request.
        if t_item.get("caption_emb") is not None:
            out["text"] = torch.as_tensor(np.asarray(t_item["caption_emb"])).float()
        sem = t_item.get("joint_semantics")          # order-hash checked inside the dataset
        if sem is not None:
            out["joint_sem"] = torch.as_tensor(np.asarray(sem))[:J].float()
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
    out["valid"] = out["frame_valid"][:, :, None] & joint_valid[:, None, :]      # [B,T,Jm]
    return out
