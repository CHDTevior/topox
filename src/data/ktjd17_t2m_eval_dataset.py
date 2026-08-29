"""KTJD-17 T2M evaluator dataset (PZ-animal corpus) -- the KTJD twin of
AnyTopT2MEvalDataset (user 2026-08-29: train an evaluator on the current pure-PZ data).

A THIN wrapper over Ktjd17Base.__getitem__ (all payload/identity/normalization
work stays there, A1-style). Representation deltas vs the 13ch evaluator
corpus, absorbed here:
  * channels: KTJD-17 per-(rig,joint,channel) standardized; CONTACT (ch12) is
    DROPPED at the dataset level -> 16ch served (the proven contact-free
    lesson; contact is mid-tensor here so it is an index_select, not the old
    tail slice). Kept: q_pos 0:3 | rot6d 3:9 | vel 9:12 | smooth_root 13:15 |
    heading 15:17. Plane 17 (heading flag) never leaves the base.
  * graph fields: derived per rig from FK-ordered parents/offsets/names via the
    SAME `_build_derived` the 13ch corpus used (skeleton_features[J,9],
    adjacency, Floyd geodesic, name_hashes, joint_relations 0..5,
    joints_graph_dist clamped 5), cached per rig.
  * captions: the base's raw caption string (DistilBERT primary; no T5 cache).
  * false-negative keys: motion_id = clip_id; caption_text as served (the key that
    actually fires: template captions repeat heavily). source_motion_id = official_id
    is a NO-OP on this corpus -- the active pure-PZ export has 77,894 clips with
    77,894 unique official_ids (verified 2026-08-29), i.e. no animation is exported
    across rigs here; the key is kept only as forward-compat for future mixed-source
    corpora where official_id does repeat.
"""
from __future__ import annotations

import numpy as np
import torch

from .ktjd17_incontext import Ktjd17Base, ktjd17_split_names
from .anytop_dataset import _build_derived

KEEP_CH = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16]   # 17 minus contact(12)
J_MAX = 102   # measured over all 311 PZ rigs (min 34); fixed padding for the stack-collate



class Ktjd17T2MEvalDataset(torch.utils.data.Dataset):
    def __init__(self, base: Ktjd17Base, split: str, *, max_frames: int = 240,
                 exclude: str | None = None):
        self.base = base
        self.Tt = int(max_frames)
        names = ktjd17_split_names(base.root, exclude=exclude)
        if split not in names:
            raise ValueError(f"split {split!r} not in corpus (has {sorted(names)})")
        wanted = names[split]
        self._idx, self._official = [], {}
        for i, r in enumerate(base._rows):
            cid = str(r["clip_id"])
            if cid in wanted:
                self._idx.append(i)
                self._official[cid] = str(r.get("official_id", cid))
        if not self._idx:
            raise ValueError(f"empty plan for split {split!r} (base rows may be cut differently)")
        self._derived_cache: dict[str, dict] = {}
        rigs = {str(base._rows[i]["rig_id"]) for i in self._idx}
        print(f"Ktjd17T2MEvalDataset[{split}]: {len(self._idx)} clips, {len(rigs)} rigs, "
              f"16ch (contact dropped)")

    def __len__(self):
        return len(self._idx)

    def _derived(self, rig: str) -> dict:
        if rig not in self._derived_cache:
            sk = self.base._skeleton(rig)
            self._derived_cache[rig] = _build_derived(
                np.asarray(sk["parents"], dtype=np.int64),
                np.asarray(sk["offset_parent_local"], dtype=np.float64),
                [str(x) for x in sk["joint_names"]])
        return self._derived_cache[rig]

    def __getitem__(self, idx: int) -> dict:
        item = self.base[self._idx[idx]]
        rig, clip = str(item["object_type"]), str(item["motion_id"])
        T = min(int(item["num_frames"]), self.Tt)
        J = int(item["num_joints"])
        x = np.asarray(item["anytop_x"])[:, :17, :T]                 # [J,17,T]
        x16 = np.zeros((J_MAX, 16, self.Tt), dtype=np.float32)      # fixed-size pad for collate
        x16[:J, :, :T] = x[:, KEEP_CH, :]
        d = self._derived(rig)
        def padJ(a, fill=0.0):
            a = np.asarray(a)
            out = np.full((J_MAX,) + a.shape[1:], fill, dtype=a.dtype)                 if a.ndim == 1 else np.full((J_MAX, J_MAX), fill, dtype=a.dtype)
            if a.ndim == 1:
                out[:J] = a
            else:
                out[:J, :J] = a
            return out
        skf = np.zeros((J_MAX, 9), dtype=np.float32)
        skf[:J] = np.asarray(d["skeleton_features"], dtype=np.float32)
        ro = np.zeros((J_MAX, 3), dtype=np.float32)
        ro[:J] = np.asarray(item["rest_offsets"], dtype=np.float32)
        joint_mask = np.zeros(J_MAX, dtype=bool); joint_mask[:J] = True
        frame_mask = np.zeros(self.Tt, dtype=bool); frame_mask[:T] = True
        return {
            "anytop_x": torch.from_numpy(x16),
            "num_joints": J,
            "num_frames": T,
            "joint_mask": torch.from_numpy(joint_mask),
            "frame_mask": torch.from_numpy(frame_mask),
            "skeleton_features": torch.from_numpy(skf),
            "adjacency": torch.from_numpy(padJ(np.asarray(d["adjacency"], dtype=np.float32))),
            "geodesic_dist": torch.from_numpy(
                padJ(np.asarray(d["geodesic_dist"], dtype=np.float32), fill=float(J_MAX))),
            "name_hashes": torch.from_numpy(padJ(np.asarray(d["name_hashes"], dtype=np.int64))),
            "anytop_joint_relations": torch.from_numpy(
                padJ(np.asarray(d["joint_relations"], dtype=np.float32))),
            "anytop_graph_dist": torch.from_numpy(
                padJ(np.asarray(d["joints_graph_dist"], dtype=np.float32), fill=5.0)),
            "caption_text": str(item.get("caption") or ""),
            "object_type": rig,   # sanity's within-"species" grouping key = per-rig here
            "motion_id": clip,
            "source_motion_id": self._official[clip],
            "source": "planetzoo",
            # ---- validation-only fillers -------------------------------------------------
            # GraphMotionBatch.from_collate_dict enforces the FULL graph-salad-VAE schema,
            # but AnyTopT2MEvaluator.encode_motion consumes ONLY anytop_x + the graph stack
            # above. These keys exist to satisfy the schema; none of them reaches the towers.
            "motion_features": torch.zeros(self.Tt, J_MAX, 6),
            "root_position": torch.zeros(self.Tt, 3),
            "root_velocity": torch.zeros(self.Tt, 3),
            "local_rotations_6d": torch.zeros(self.Tt, J_MAX, 6),
            "foot_contact": torch.zeros(self.Tt, 4),
            "bone_lengths": torch.from_numpy(np.linalg.norm(ro, axis=-1))[None].repeat(self.Tt, 1),  # [T,J] per the schema
            "rest_offsets": torch.from_numpy(ro),
            "parent_indices": [int(v) for v in np.asarray(item["parent_indices"])],
            "joint_names": [str(x) for x in self.base._skeleton(rig)["joint_names"]],
            "canonical_names": [str(x) for x in self.base._skeleton(rig)["joint_names"]],
            "bone_lengths_rest": [float(v) for v in np.linalg.norm(
                np.asarray(item["rest_offsets"], dtype=np.float32), axis=-1)],
            "text": str(item.get("caption") or ""),
            "skeleton_id": rig,
            "fps": 30.0,
            "has_rotations": True,
        }
