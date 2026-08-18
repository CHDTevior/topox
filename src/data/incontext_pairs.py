"""[demo | target] in-context pairs over a single skeleton, for the v2 motion DiT.

WHAT ONE ITEM IS
    Two DIFFERENT clips of the SAME skeleton, concatenated along time:
        x         = [ demo slot 64 | target slot 96 ]        T = 160
        is_target = [ False        | True on REAL target frames only ]
    The demo supplies "how THIS rig moves"; the text supplies "which action". Because demo and
    target are the same rig, the joint axis is aligned within an item for free -- no cross-skeleton
    joint correspondence is needed anywhere. That is why an unseen rig costs nothing structurally
    at inference: it walks the identical forward pass.

SLOT SIZES come from the measured length distribution of the TrueBones training pool (n=694):
    median 90 frames, p90 199, >=64 in 70.3% of clips, >=96 in 45.0%.
    target slot 96 -> the median clip fits whole;  demo slot 64 -> 70% of clips fill it.
    At 20 fps that is 3.2 s of demo and 4.8 s of target.

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

import zlib
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from src.data.anytop_dataset import AnyTopDataset

DEMO_FRAMES = 64
TARGET_FRAMES = 96
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
        self.rng = np.random.default_rng(seed)
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

        # A rig is usable only if it can supply a demo distinct from the target.
        self.by_type = {}
        for ot, tg in tgt_pool.items():
            dm = demo_pool.get(ot, [])
            if not dm:
                continue
            if len(dm) == 1 and len(tg) == 1 and dm[0] == tg[0]:
                continue                      # single clip that is both -> no legal pair
            self.by_type[ot] = {"targets": sorted(tg), "demos": sorted(dm)}

        self.types = sorted(self.by_type)
        self.index = [(ot, i) for ot in self.types for i in self.by_type[ot]["targets"]]

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
        """DataLoader workers fork the object, so a shared self.rng makes every worker emit the
        SAME demo/crop choices. Derive per-worker, per-call streams instead."""
        info = torch.utils.data.get_worker_info()
        if info is None:
            return self.rng
        if getattr(self, "_wrng_id", None) != info.id:
            self._wrng_id = info.id
            self._wrng = np.random.default_rng(self.seed * 1000003 + info.id)
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

    def _crop(self, x, n_valid, want, rng):
        """Random `want`-frame window from the clip's valid region; pad at the end if shorter.
        Random rather than always-from-zero because 12% of training clips exceed 192 frames and
        their tails would otherwise never be seen. Returns (cropped [want,J,C], valid [want])."""
        J, C = x.shape[1], x.shape[2]
        n = int(min(n_valid, x.shape[0]))
        out = np.zeros((want, J, C), dtype=np.float32)
        vm = np.zeros((want,), dtype=bool)
        if n >= want:
            s = int(rng.integers(0, n - want + 1))
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
        demo_idx = int(demos[int(rng.integers(len(demos)))]) if demos else tgt_idx

        t_item, t_x, J, t_T = self._raw(tgt_idx)
        _, d_x, dJ, d_T = self._raw(demo_idx)
        if dJ != J:
            raise ValueError(f"{ot}: demo has {dJ} joints, target {J} -- joint-count augmentation "
                             f"must be off for in-context pairs")

        d_crop, d_valid = self._crop(d_x, d_T, self.Td, rng)
        t_crop, t_valid = self._crop(t_x, t_T, self.Tt, rng)

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
