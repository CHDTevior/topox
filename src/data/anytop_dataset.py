"""AnyTop truebones_processed dataset adapter for Graph-VAE.

Reads AnyTop's pre-processed dataset at
  /iridisfs/scratch/ts1v23/workspace/Anytop/AnyTop/dataset/truebones/zoo/truebones_processed/
- motions/*.npy : per-clip RAW motion [T_var, J_i, 13], float64.
  channels: 0:3 RIFKE/relative pos | 3:9 6D rotation | 9:12 velocity | 12 contact
  IMPORTANT: for the ROOT joint (j=0), channels 0:3 are NOT positions — they are
  RIFKE root state (angular_vel_y, root_height_y, ???). Channels 9, 11 hold
  root xz linear velocity. Channel 1 holds root height. AnyTop's
  recover_root_quat_and_pos_np (motion_process.py:455) reconstructs the world-
  space root trajectory from these. We do the same in `_recover_world_positions`
  via scipy.spatial.transform.Rotation (no need to port AnyTop's Quaternions
  class — only inverse-quaternion and vector rotation are required, both
  trivially provided by scipy).
- cond.npy     : dict[object_type -> {parents, offsets, tpos_first_frame,
                  joint_relations, joints_graph_dist, joints_names,
                  kinematic_chains, mean, std}]
- motion_texts_by_file.json (optional) : caption per filename

Iter-1.5 contract (post-codex review of iter 1; semantics-correct):
  - motion_features [T, J, 6] holds WORLD-SPACE joint positions (channels 0:3)
    + world-space velocity (channels 3:6, numerical diff of pos × fps). Both
    derived from AnyTop's RIFKE encoding via `_recover_world_positions`. This
    is what the VAE's FK decoder is designed to predict.
  - skeleton_features [J, 9] built via the official SkeletonGraph class
    (src/data/skeleton_graph.py) — same recipe as UnifiedMotionDataset.
  - adjacency from parents; geodesic_dist Floyd-from-adjacency (true hops).
  - 6D rotation channels exposed as local_rotations_6d [T, J, 6] (raw, un-normalized
    — they live on a unit-like manifold already).
  - Foot contact exposed PER-JOINT as foot_contact_per_joint [T, J] (the
    AnyTop convention: channel 12 of every joint, not just root). The legacy
    [T, 4] foot_contact key is kept zero-filled for GraphMotionBatch schema
    compatibility; the new field is the source of truth for any contact loss.
  - Per-object stratified 80/20 split via `hashlib.md5(object_type).hexdigest()`
    seed (NOT Python's salted hash() — stable across processes).
  - Extra AnyTop-native passthrough keys for the future 13ch end-to-end path:
      anytop_x [J, 13, T]           : NORMALIZED 13ch view (AnyTop mean/std applied)
      anytop_graph_dist [J, J]      : AnyTop's CLAMPED-at-5 graph distance
      anytop_joint_relations [J, J] : 6-class edge type
      anytop_tpos_first_frame [J, 13] (normalized)
      anytop_mean [J, 13], anytop_std [J, 13] (raw, un-normalized)
      object_type str, caption str

NOT done in iter 1.5 (deferred):
  - 13ch end-to-end encoder/decoder (still 6ch path with FK head; pred_pos
    target is recovered world pos, which IS FK-compatible — semantic gap closed)
  - Contact BCE loss / rotation geodesic loss
  - Graphormer-style attention bias using anytop_graph_dist / anytop_joint_relations
  - T5 caption embedding (only raw string passed through)
"""

from __future__ import annotations

import hashlib
import json
import random
from collections import defaultdict, deque
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from scipy.spatial.transform import Rotation as _ScipyRotation
from torch.utils.data import Dataset

from .skeleton_graph import SkeletonGraph


# Local copy of AnyTop's processed truebones data (motions/ + cond.npy +
# motion_texts_by_file.json), copied into this project to decouple training
# from the external AnyTop repo path. The AnyTop source is read-only and
# never modified; this is an independent copy. Override with `data_root` /
# `--anytop_root` / `ANYTOP_ROOT` to point elsewhere (e.g. the AnyTop repo
# original at .../Anytop/AnyTop/dataset/truebones/zoo/truebones_processed).
_DEFAULT_ANYTOP_ROOT = (
    "/iridisfs/scratch/ts1v23/workspace/noKslot_clean/data/anytop_truebones"
)
_STD_FLOOR = 1e-6  # matches AnyTop's `std += 1e-6` stability constant


def _read_split_file(path: Path) -> list[str]:
    """Read a splits/{train,val}.txt list -- one .npy basename per line, skipping
    blank lines and '#' comments. Order preserved."""
    out: list[str] = []
    with path.open("r") as fh:
        for line in fh:
            s = line.strip()
            if s and not s.startswith("#"):
                out.append(s)
    return out


def _duplicates(names: list[str]) -> list[str]:
    """Return the distinct values that appear more than once in `names` (O(N))."""
    seen: set[str] = set()
    dup_seen: set[str] = set()
    dups: list[str] = []
    for n in names:
        if n in seen and n not in dup_seen:
            dup_seen.add(n)
            dups.append(n)
        seen.add(n)
    return dups


def _longest_prefix_match(fname: str, keys_sorted_desc: list[str]) -> Optional[str]:
    """Match a filename to its cond object_type by longest-prefix.

    AnyTop ships motions in two naming conventions:
      "Alligator___BigMouth_5.npy"     -> object_type "Alligator"
      "Cat_CAT_IdlePurr_195.npy"       -> object_type "Cat"
      "Fox_-_Attack1_361.npy"          -> object_type "Fox"
    so a plain `split("___")` misses 45 / 1070 files. `keys_sorted_desc` is
    cond.keys() sorted by len(key) descending so a "BrownBear" file resolves
    before a "Bear" prefix match would be tried.
    """
    for k in keys_sorted_desc:
        if fname.startswith(f"{k}_"):
            return k
    return None


def _derive_skeleton_features(
    parents: np.ndarray,
    offsets: np.ndarray,
    joint_names: list[str],
) -> np.ndarray:
    """Build [J, 9] skeleton features via the canonical SkeletonGraph recipe.

    Delegates to `SkeletonGraph.get_joint_features()` so this adapter stays
    bit-compatible with UnifiedMotionDataset (which is the contract pool /
    encoder were trained against). Reference impl:
    src/data/skeleton_graph.py:223 — norm_offsets(3) + norm_bones(1) +
    norm_depths(1) + norm_degrees(1) + side_onehot(3). Side tags inferred from
    the rich heuristic at skeleton_graph.py:103 (matches "left/right/lft/rgt",
    "_L"/"_R" suffix, "_l_"/"_r_" infix, "LHipJoint"/"RThumb" prefix patterns
    — strictly more than our previous "l_"/"r_" heuristic, which is codex
    P1 #7).
    """
    sg = SkeletonGraph(
        joint_names=[str(n) for n in joint_names],
        parent_indices=[int(p) for p in parents.tolist()],
        rest_offsets=offsets.astype(np.float32),
    )
    return sg.get_joint_features().astype(np.float32)


# ---------- AnyTop RIFKE -> world-position recovery ----------
def _rotation_6d_to_matrix_np(d6: np.ndarray) -> np.ndarray:
    """Continuous 6D rotation -> 3x3 rotation matrix (Zhou et al. 2019).

    d6: [..., 6]. First 3 = first column of R (after norm); next 3 normalized
    perpendicular to first; third column = cross. Returns [..., 3, 3] where
    output[..., :, k] is the k-th column. Matches AnyTop's
    utils.rotation_conversions.rotation_6d_to_matrix_np (verified equivalent
    by independent derivation; no proprietary code copied).
    """
    a1 = d6[..., :3]
    a2 = d6[..., 3:]
    b1 = a1 / (np.linalg.norm(a1, axis=-1, keepdims=True) + 1e-8)
    b2 = a2 - (np.sum(b1 * a2, axis=-1, keepdims=True)) * b1
    b2 = b2 / (np.linalg.norm(b2, axis=-1, keepdims=True) + 1e-8)
    b3 = np.cross(b1, b2)
    return np.stack([b1, b2, b3], axis=-1)  # [..., 3, 3]


def _create_topology_edge_relations(
    parents: np.ndarray, max_path_len: int = 5
) -> tuple[np.ndarray, np.ndarray]:
    """Port of AnyTop's create_topology_edge_relations (motion_process.py:284).

    Returns (edge_rel, topo_rel), both [J, J] float32:
      edge_rel  — edge type 0..5 (self/parent/child/sibling/no_relation/end_effector)
      topo_rel  — hop distance, clamped at max_path_len (5)
    Requires FK-ordered `parents` (parents[j] < j) — the topo recurrence reads
    topo_rel[i, parent_j] which is only filled if parent_j < j.
    """
    n = len(parents)
    topo_rel = np.zeros((n, n), dtype=np.float32)
    edge_rel = np.full((n, n), 4.0, dtype=np.float32)  # 4 = no_relation
    for i in range(n):
        parent_i = int(parents[i])
        is_ee = True
        for j in range(n):
            parent_j = int(parents[j])
            if i == j:
                edge_rel[i, j] = 0.0          # self
            elif parent_j == i:
                is_ee = False
                edge_rel[i, j] = 2.0          # child
            elif j == parent_i:
                edge_rel[i, j] = 1.0          # parent
            elif parent_j == parent_i:
                edge_rel[i, j] = 3.0          # sibling
            # topo (hop) distance
            if i == j:
                topo_rel[i, j] = 0.0
            elif j < i:
                topo_rel[i, j] = topo_rel[j, i]
            elif parent_j == i:
                topo_rel[i, j] = 1.0
            else:
                topo_rel[i, j] = topo_rel[i, parent_j] + 1.0
        if is_ee:
            edge_rel[i, i] = 5.0              # end_effector
    topo_rel[topo_rel > max_path_len] = max_path_len
    return edge_rel, topo_rel


def _build_derived(
    parents: np.ndarray, offsets: np.ndarray, joint_names: list[str]
) -> dict:
    """Derive all graph fields from an FK-ordered skeleton.

    Shared by `_normalize_cond_entry` (dataset construction) and
    `_remove_joints_aug` (augmentation) so both paths produce a bit-identical
    derived stack. `parents` must be FK-ordered numpy int64 (parents[0] == -1,
    parents[j] < j). Returns: skeleton_features [J,9], adjacency [J,J],
    geodesic_dist [J,J] (true Floyd hops), name_hashes [J], joint_relations
    [J,J], joints_graph_dist [J,J] (AnyTop clamped-at-5).
    """
    J = len(parents)
    skel_feats = _derive_skeleton_features(parents, offsets, joint_names)
    adjacency = _parents_to_adjacency(parents, J)
    # True Floyd hop count over the parents-derived adjacency — kept self-
    # consistent with adjacency (DynamicGraphPool validates floyd(adj) == geo).
    geodesic_floyd = _floyd_hops_numpy(adjacency)
    geodesic_floyd = np.where(
        np.isfinite(geodesic_floyd), geodesic_floyd, float(J)
    ).astype(np.float32)
    name_hashes = np.array(
        [int(hashlib.md5(n.encode()).hexdigest(), 16) % 1024 for n in joint_names],
        dtype=np.int64,
    )
    # AnyTop-style edge type + clamped hop distance (recomputed from the
    # FK-ordered topology — equivalent to AnyTop's cond.npy values, and
    # required for the augmentation path where the topology shrinks).
    joint_relations, joints_graph_dist = _create_topology_edge_relations(parents)
    return {
        "skeleton_features": skel_feats,
        "adjacency": adjacency,
        "geodesic_dist": geodesic_floyd,
        "name_hashes": name_hashes,
        "joint_relations": joint_relations,
        "joints_graph_dist": joints_graph_dist,
    }


def _remove_joints_aug(
    raw_motion: np.ndarray, sk: dict, removal_rate: float, rng: random.Random
) -> tuple[np.ndarray, dict]:
    """Port of AnyTop's remove_joints_augmentation (motion_process.py:580).

    Removes a random subset of NON-FOOT end-effector joints from an FK-ordered
    skeleton. End-effectors are leaves (no children); feet (joints that ever
    carry a contact flag) are excluded so locomotion stays intact. Joint count
    shrinks → fixed `max_joints` padding stays valid, and deleting a leaf keeps
    the `parents[j] < j` FK-ordering invariant.

    Args:
      raw_motion: [T, J, 13] FK-ordered RAW motion clip.
      sk: the FK-ordered cond dict (NOT mutated — local copies are made).
      removal_rate: fraction of eligible end-effectors to drop.
      rng: random.Random instance.
    Returns: (reduced_raw_motion, reduced_sk) — reduced_sk has the same keys
    `__getitem__` reads. If nothing is eligible, returns inputs unchanged.
    """
    parents = np.asarray(sk["parents"], dtype=np.int64)
    J = len(parents)
    # End-effectors = joints that are nobody's parent.
    has_child = set(int(p) for p in parents if p >= 0)
    ee = [j for j in range(1, J) if j not in has_child]  # exclude root (j=0)
    # Feet = joints that ever carry a contact flag (channel 12 > 0).
    feet = set(int(j) for j in np.unique(np.where(raw_motion[:, :, 12] > 0)[1]))
    removal_options = [j for j in ee if j not in feet]
    n_remove = int(np.floor(len(removal_options) * removal_rate))
    if n_remove <= 0:
        return raw_motion, sk
    remove = sorted(rng.sample(removal_options, n_remove), reverse=True)

    new_motion = np.delete(raw_motion, remove, axis=1)
    new_parents = np.delete(parents, remove, axis=0)
    # Decrement parent pointers above each removed index (descending order so
    # each decrement sees indices consistent with the prior step).
    for rj in remove:
        new_parents[new_parents > rj] -= 1
    new_offsets = np.delete(np.asarray(sk["offsets"], dtype=np.float32), remove, axis=0)
    new_tpos = np.delete(np.asarray(sk["tpos_first_frame"], dtype=np.float32), remove, axis=0)
    new_mean = np.delete(np.asarray(sk["mean"], dtype=np.float32), remove, axis=0)
    new_std = np.delete(np.asarray(sk["std"], dtype=np.float32), remove, axis=0)
    new_names = [n for k, n in enumerate(sk["joint_names"]) if k not in set(remove)]

    derived = _build_derived(new_parents, new_offsets, new_names)
    reduced_sk = {
        "n_joints": len(new_parents),
        "parents": new_parents,
        "joint_names": new_names,
        "offsets": new_offsets,
        "tpos_first_frame": new_tpos,
        "mean": new_mean,
        "std": new_std,
        **derived,
    }
    return new_motion, reduced_sk


def _recover_world_positions(motion_13ch: np.ndarray) -> np.ndarray:
    """Recover world-space [T, J, 3] joint positions from AnyTop RIFKE encoding.

    Mirrors AnyTop motion_process.recover_from_bvh_ric_np (line 493):
      1. Root rotation per frame from 6D rot at channels 3:9.
      2. Root xz position via cumulative sum of velocities at channels 9 & 11,
         applied AFTER inverse-rotating the per-frame velocity into the world
         frame (so cumsum acts in world space).
      3. Root y position from channel 1 (height stored directly, not integrated).
      4. Non-root joint positions: channels 0:3 are root-relative; rotate them
         by inverse root rotation per frame to go to world frame, then add
         root xz.

    Args:
      motion_13ch: [T, J, 13] raw (un-normalized) AnyTop motion encoding.
    Returns:
      [T, J, 3] world-space joint positions.
    """
    if motion_13ch.ndim != 3 or motion_13ch.shape[-1] != 13:
        raise ValueError(
            f"motion_13ch must be [T, J, 13], got {motion_13ch.shape}"
        )
    motion = motion_13ch.astype(np.float32)
    T, J, _ = motion.shape
    root = motion[:, 0, :]  # [T, 13]

    # 1. Root rotation per frame from 6D rot (channels 3:9).
    rot_mat = _rotation_6d_to_matrix_np(root[:, 3:9])  # [T, 3, 3]
    root_rot = _ScipyRotation.from_matrix(rot_mat)     # [T]

    # 2. Root xz integration: shift-by-1 vel (no motion at t=0), inverse-rotate
    #    per frame, cumsum. AnyTop's code uses indices 9 (x) and 11 (z); idx 10
    #    is NOT used in root recovery (it's per-joint vel_y elsewhere).
    rpos_local = np.zeros((T, 3), dtype=np.float32)
    rpos_local[1:, 0] = root[:-1, 9]   # vel_x at t-1
    rpos_local[1:, 2] = root[:-1, 11]  # vel_z at t-1
    # Apply inverse rotation per frame (no broadcasting in scipy; loop is cheap).
    inv_rot = root_rot.inv()
    rpos_world = np.zeros_like(rpos_local)
    for t in range(T):
        rpos_world[t] = inv_rot[t].apply(rpos_local[t])
    rpos_world = np.cumsum(rpos_world, axis=0)
    rpos_world[:, 1] = root[:, 1]  # root height directly from channel 1

    # 3. Non-root joints: rotate root-relative pos (channels 0:3) to world.
    if J > 1:
        rel = motion[:, 1:, :3].astype(np.float32)  # [T, J-1, 3]
        world_rel = np.zeros_like(rel)
        for t in range(T):
            world_rel[t] = inv_rot[t].apply(rel[t])  # [J-1, 3]
        # Add root xz (NOT root y — AnyTop encodes root y directly per frame
        # at root.channel_1; non-root joints carry their own y as part of
        # root-relative pos channels 0:3 -> after inverse-rotate, they're in
        # world frame already except for the missing root xz origin shift).
        world_rel[..., 0] += rpos_world[:, None, 0]
        world_rel[..., 2] += rpos_world[:, None, 2]
    else:
        world_rel = np.zeros((T, 0, 3), dtype=np.float32)

    # Concatenate root world pos at index 0
    world_positions = np.concatenate(
        [rpos_world[:, None, :], world_rel], axis=1
    )  # [T, J, 3]
    return world_positions.astype(np.float32)


def _parents_to_adjacency(parents: np.ndarray, J: int) -> np.ndarray:
    """Symmetric binary adjacency from parent_indices. Self-loops excluded."""
    A = np.zeros((J, J), dtype=np.float32)
    for j, p in enumerate(parents):
        if p >= 0 and p < J and j != p:
            A[j, int(p)] = 1.0
            A[int(p), j] = 1.0
    return A


def _floyd_hops_numpy(adjacency: np.ndarray) -> np.ndarray:
    """Floyd-Warshall hop-count shortest path on an undirected adjacency.

    Mirrors src/models/graph_salad/graph_utils.floyd_shortest_path (no_grad
    pure tensor op there) so we can compute it data-side without a
    model-module import. Output dtype float32; unreachable pairs -> +inf;
    diagonal -> 0.
    """
    J = adjacency.shape[0]
    INF = np.float32("inf")
    D = np.where(adjacency > 0, 1.0, INF).astype(np.float32)
    np.fill_diagonal(D, 0.0)
    # Floyd's loop (J usually ≤ 142 -> ~3M ops, sub-second).
    for k in range(J):
        D = np.minimum(D, D[:, k:k + 1] + D[k:k + 1, :])
    return D


def _normalize_parents_to_root_first(
    parents: np.ndarray, joint_names: list, **arrays
) -> tuple[np.ndarray, list, dict, np.ndarray]:
    """Reorder joints so parents[0] == -1 (root) and parents[j] < j for all j>0.

    AnyTop cond.parents has root sentinel -1 but root may not be at index 0
    (e.g., 'locator2' at index 0 with parents[0]=-1 but kinematic graph requires
    a topological re-ordering). We do a BFS from the root, mapping old index ->
    new index, and reindex parents + all per-joint arrays.

    Returns (new_parents, new_joint_names, reindexed_arrays, new_to_old_perm).
    `new_to_old_perm[new_idx] = old_idx` — used at __getitem__ time to reorder
    raw clip motion to match the FK-ordered skeleton arrays.
    """
    J = len(parents)
    root_candidates = np.where(parents == -1)[0]
    if len(root_candidates) != 1:
        raise ValueError(
            f"Expected exactly 1 root (parent == -1), got {len(root_candidates)}: "
            f"{root_candidates.tolist()}"
        )
    old_root = int(root_candidates[0])

    children = defaultdict(list)
    for j, p in enumerate(parents):
        if p >= 0:
            children[int(p)].append(j)
    old_to_new = {old_root: 0}
    queue = deque([old_root])
    next_new = 1
    while queue:
        u = queue.popleft()
        for v in sorted(children[u]):
            if v in old_to_new:
                continue
            old_to_new[v] = next_new
            next_new += 1
            queue.append(v)
    if len(old_to_new) != J:
        raise ValueError(
            f"BFS visited {len(old_to_new)} joints but skeleton has {J}; "
            f"disconnected graph?"
        )
    new_to_old = np.zeros(J, dtype=np.int64)
    for old, new in old_to_new.items():
        new_to_old[new] = old

    new_parents = np.full(J, -1, dtype=np.int64)
    for old, new in old_to_new.items():
        p_old = int(parents[old])
        new_parents[new] = -1 if p_old < 0 else old_to_new[p_old]
    if new_parents[0] != -1:
        raise ValueError(f"Post-reorder root not at 0: parents[0]={new_parents[0]}")
    for j in range(1, J):
        if new_parents[j] >= j:
            raise ValueError(
                f"Post-reorder parent[{j}]={new_parents[j]} >= j (not FK-ordered)"
            )

    new_joint_names = [str(joint_names[new_to_old[j]]) for j in range(J)]
    reindexed: dict[str, np.ndarray] = {}
    for name, arr in arrays.items():
        if arr is None:
            continue
        if arr.ndim == 1 and arr.shape[0] == J:
            reindexed[name] = arr[new_to_old]
        elif arr.ndim == 2 and arr.shape[0] == J and arr.shape[1] == J:
            reindexed[name] = arr[np.ix_(new_to_old, new_to_old)]
        elif arr.ndim == 2 and arr.shape[0] == J:
            reindexed[name] = arr[new_to_old]
        elif arr.ndim == 3 and arr.shape[1] == J:
            reindexed[name] = arr[:, new_to_old, :]
        else:
            reindexed[name] = arr
    return new_parents, new_joint_names, reindexed, new_to_old


class AnyTopDataset(Dataset):
    """AnyTop truebones_processed -> GraphMotionBatch-compatible samples.

    Args:
        data_root: path to truebones_processed dir (default: AnyTop's processed dir).
        split: 'train' | 'val' | 'all'. For 'train'/'val', if BOTH
            data_root/splits/{train,val}.txt exist and use_split_file is True,
            the split is READ from those files; otherwise it falls back to a
            per-object stratified holdout (md5-seeded, deterministic). 'all'
            returns every clip.
        num_frames: temporal crop/pad target (default 64 — matches our config).
        max_joints: spatial pad target (default 143 — user spec; dataset max is 142).
        load_captions: if True, parse motion_texts_by_file.json and attach primary_caption.
        val_frac: 0.2 default.
        seed: 42 default.
        augment: if True (train split only) randomly drop non-foot end-effector
            joints per AnyTop's remove_joints augmentation. NO-OP on val/all.
        augment_prob: per-sample probability of applying removal (default 0.3).
        removal_rate: fraction of eligible end-effectors to drop (default 0.5).
        use_split_file: if True (default), 'train'/'val' are read from
            data_root/splits/{train,val}.txt when both exist (else fall back to
            the stratified algorithm). Set False to FORCE the algorithm — used by
            scripts/_export_split_lists.py to (re)generate those files.
    """

    def __init__(
        self,
        data_root: str | Path = _DEFAULT_ANYTOP_ROOT,
        split: str = "train",
        num_frames: int = 64,
        max_joints: int = 143,
        target_fps: float = 20.0,
        load_captions: bool = True,
        val_frac: float = 0.2,
        seed: int = 42,
        augment: bool = False,
        augment_prob: float = 0.3,
        removal_rate: float = 0.5,
        caption_emb_cache: str | Path | None = None,
        random_caption: bool = False,
        random_crop: bool | None = None,
        use_split_file: bool = True,
    ) -> None:
        self.data_root = Path(data_root)
        self.split = split
        self.num_frames = num_frames
        self.max_joints = max_joints
        self.target_fps = target_fps
        # Augmentation is train-only — guard here so a val/all dataset built
        # with augment=True still never augments.
        self.augment = bool(augment) and split == "train"
        self.augment_prob = augment_prob
        self.removal_rate = removal_rate

        if not self.data_root.exists():
            raise FileNotFoundError(f"AnyTop data_root not found: {self.data_root}")
        cond_path = self.data_root / "cond.npy"
        if not cond_path.exists():
            raise FileNotFoundError(f"cond.npy not found at {cond_path}")
        motions_dir = self.data_root / "motions"
        if not motions_dir.exists():
            raise FileNotFoundError(f"motions/ dir not found at {motions_dir}")

        # ---- Load + per-object preprocess cond (with disk cache) ----
        # Cache rationale (2026-05-26): for PlanetZoo L1 (473 object types) the
        # pure-Python _normalize_cond_entry × _create_topology_edge_relations
        # O(J²) loop takes ~107s single-process,~25 min under 4-way DDP
        # contention. Cache normalized cond next to cond.npy so subsequent runs
        # (including DDP per-rank construct + cont chains) load in <1s.
        # Invalidation: cache filename includes max_joints (entries are skipped
        # if J > max_joints) so different max_joints get different caches.
        import pickle
        cache_path = self.data_root / f"_cond_normalized_J{self.max_joints}.pkl"
        if cache_path.exists() and cache_path.stat().st_mtime > cond_path.stat().st_mtime:
            with cache_path.open("rb") as f:
                self.cond: dict[str, dict] = pickle.load(f)
            print(f"  [AnyTopDataset] loaded normalized cond from cache "
                  f"({len(self.cond)} object types, {cache_path.name})")
        else:
            raw_cond = np.load(cond_path, allow_pickle=True).item()
            self.cond: dict[str, dict] = {}
            for obj_type, c in raw_cond.items():
                try:
                    normalized = self._normalize_cond_entry(c, obj_type)
                    if normalized["n_joints"] > self.max_joints:
                        print(
                            f"  [AnyTopDataset] WARNING: {obj_type} has J="
                            f"{normalized['n_joints']} > max_joints={self.max_joints}; "
                            f"clips of this type will be skipped"
                        )
                        continue
                    self.cond[obj_type] = normalized
                except (ValueError, KeyError) as e:
                    print(f"  [AnyTopDataset] WARNING: skip cond[{obj_type}]: {e}")
            # Save cache (atomic write via globally-unique tmp + rename to
            # handle DDP race). Codex P1 fix 2026-05-26: shared tmp suffix
            # between ranks was unsafe (multiple ranks open/truncate/write
            # same inode). PID alone is host-local — multi-node DDP could
            # still collide. Use tempfile.NamedTemporaryFile to get a
            # filesystem-unique name regardless of host/pid/rank; rename to
            # cache_path is atomic (POSIX), last rank wins, content
            # deterministic so no corruption.
            import tempfile
            with tempfile.NamedTemporaryFile(
                mode="wb", dir=cache_path.parent,
                prefix=cache_path.name + ".tmp.", delete=False
            ) as f:
                pickle.dump(self.cond, f, protocol=pickle.HIGHEST_PROTOCOL)
                tmp_path = Path(f.name)
            tmp_path.replace(cache_path)
            print(f"  [AnyTopDataset] saved normalized cond cache "
                  f"({len(self.cond)} object types → {cache_path.name})")

        # ---- Scan motions/, match prefix, build sample list ----
        keys_sorted = sorted(self.cond.keys(), key=lambda k: -len(k))
        all_samples: list[dict] = []
        skipped_unmatched = 0
        for fp in sorted(motions_dir.glob("*.npy")):
            fname = fp.name
            obj_type = _longest_prefix_match(fname, keys_sorted)
            if obj_type is None:
                skipped_unmatched += 1
                continue
            all_samples.append({"path": str(fp), "object_type": obj_type,
                                 "motion_id": fp.stem})
        if skipped_unmatched > 0:
            print(
                f"  [AnyTopDataset] {skipped_unmatched} clips unmatched to any "
                f"cond key (kept only matched: {len(all_samples)}/{len(all_samples)+skipped_unmatched})"
            )

        # ---- Split: prefer splits/{train,val}.txt, else per-object stratified ----
        # File mode (default): if BOTH data_root/splits/train.txt and val.txt exist
        # (and use_split_file), read the split from them -- a materialized, hand-
        # inspectable record of which clips train vs validate (generate/refresh via
        # scripts/_export_split_lists.py). If either file is missing (or the caller
        # forces use_split_file=False), fall back to the original per-object md5-
        # seeded stratified holdout, so datasets with no splits/ dir behave as before.
        if split == "all":
            self.samples = all_samples
        elif split not in ("train", "val"):
            raise ValueError(f"split must be 'train'/'val'/'all', got {split!r}")
        else:
            splits_dir = self.data_root / "splits"
            f_this = splits_dir / f"{split}.txt"
            f_other = splits_dir / ("val.txt" if split == "train" else "train.txt")
            if use_split_file and f_this.exists() and f_other.exists():
                # ---- File mode: split files are the source of truth, so ANY
                # inconsistency vs motions/ on disk is a HARD error (silent val
                # leakage or train-data exclusion otherwise). Refresh the files
                # via scripts/_export_split_lists.py after any data change.
                by_name = {Path(s["path"]).name: s for s in all_samples}
                want_this = _read_split_file(f_this)
                want_other = _read_split_file(f_other)
                if not want_this or not want_other:
                    raise ValueError(
                        f"empty split file: {f_this.name}={len(want_this)} "
                        f"{f_other.name}={len(want_other)} entries. "
                        f"Refresh _export_split_lists.py."
                    )
                dup_this, dup_other = _duplicates(want_this), _duplicates(want_other)
                if dup_this or dup_other:
                    raise ValueError(
                        f"duplicate entries in split files: {f_this.name}={dup_this[:3]} "
                        f"{f_other.name}={dup_other[:3]}. Refresh _export_split_lists.py."
                    )
                overlap = sorted(set(want_this) & set(want_other))
                if overlap:
                    raise ValueError(
                        f"{len(overlap)} clip(s) in BOTH train.txt and val.txt (val "
                        f"leakage); e.g. {overlap[:3]}. Refresh _export_split_lists.py."
                    )
                absent = [n for n in want_this + want_other if n not in by_name]
                if absent:
                    raise ValueError(
                        f"{len(absent)} clip(s) in the split files not found on disk "
                        f"(stale split file); e.g. {absent[:3]}. Refresh _export_split_lists.py."
                    )
                listed = set(want_this) | set(want_other)
                uncovered = [n for n in by_name if n not in listed]
                if uncovered:
                    raise ValueError(
                        f"{len(uncovered)} clip(s) on disk in NEITHER train.txt nor "
                        f"val.txt (excluded from training); e.g. {uncovered[:3]}. "
                        f"Refresh _export_split_lists.py."
                    )
                self.samples = [by_name[n] for n in want_this]
                print(f"  [AnyTopDataset] split='{split}' read from {f_this} "
                      f"({len(self.samples)} clips)")
            else:
                # ---- Fallback: per-object md5-seeded stratified holdout ----
                by_obj: dict[str, list[dict]] = defaultdict(list)
                for s in all_samples:
                    by_obj[s["object_type"]].append(s)
                train_set: list[dict] = []
                val_set: list[dict] = []
                for obj, lst in sorted(by_obj.items()):
                    # codex P1 #5: Python's hash() is PYTHONHASHSEED-salted -> non-
                    # deterministic across processes. Use a stable hashlib digest.
                    obj_seed_off = int(
                        hashlib.md5(obj.encode("utf-8")).hexdigest()[:8], 16
                    ) % 1000
                    rng = random.Random(seed + obj_seed_off)
                    ids = sorted(lst, key=lambda x: x["motion_id"])
                    rng.shuffle(ids)
                    n = len(ids)
                    n_val = max(1, round(n * val_frac)) if n >= 2 else 0
                    n_val = min(n_val, n - 1) if n >= 2 else 0
                    val_set.extend(ids[:n_val])
                    train_set.extend(ids[n_val:])
                self.samples = train_set if split == "train" else val_set
            self.samples.sort(key=lambda s: s["motion_id"])

        # ---- Captions (M1.7 Phase-2: multi-caption per motion, SALAD-style) ----
        # `self.captions` keeps the PRIMARY caption per motion (for display in
        # animate / log strings — backward compat with existing consumers).
        # `self.captions_multi` keeps the FULL list for future use (e.g.
        # animate captions on gif title selection).
        self.captions: dict[str, str] = {}
        self.captions_multi: dict[str, list[str]] = {}
        if load_captions:
            # Prefer the with_codex_drafts file if present (1070 covers full set);
            # fall back to the legacy file otherwise.
            for fn in ("motion_texts_by_file_with_codex_drafts.json",
                       "motion_texts_by_file.json"):
                cap_path = self.data_root / fn
                if cap_path.exists():
                    break
            else:
                cap_path = None
            if cap_path is not None:
                with cap_path.open("r") as f:
                    raw_caps = json.load(f)
                for fname, info in raw_caps.items():
                    if not isinstance(info, dict):
                        continue
                    primary = info.get("primary_caption") or ""
                    captions_list = info.get("captions") or []
                    stem = fname[:-4] if fname.endswith(".npy") else fname
                    self.captions[stem] = str(primary)
                    # Build ordered list: primary first, then de-duped rest
                    ordered = []
                    if primary:
                        ordered.append(str(primary))
                    for c in captions_list:
                        cs = str(c)
                        if cs and cs != primary:
                            ordered.append(cs)
                    if ordered:
                        self.captions_multi[stem] = ordered

        # ---- Caption T5 embeddings (M1.7 Phase-2: multi-caption cache) ----
        # New cache format (per scripts/precompute_t5_captions.py post-2026-05-23):
        #   .npz keys are '<motion_id>__cap<i>' (i=0 is primary_caption); we
        #   group by motion_id prefix → list[np.ndarray]. __getitem__ then
        #   random.choice when random_caption=True, else uses index 0
        #   (primary) for deterministic val.
        # Backward compat: old flat '{motion_id: [768]}' cache (single primary
        # caption per motion) is still loaded — each motion gets a single-element
        # list, so random.choice is degenerate but functional.
        self.caption_embs_multi: dict[str, list[np.ndarray]] = {}
        if caption_emb_cache is not None:
            cache_path = Path(caption_emb_cache)
            if not cache_path.exists():
                raise FileNotFoundError(
                    f"caption_emb_cache not found: {cache_path} "
                    f"(run scripts/precompute_t5_captions.py first)"
                )
            # Group by motion_id: collect (idx, emb) per motion, then sort by idx.
            raw_groups: dict[str, list[tuple[int, np.ndarray]]] = {}

            def _split_caption_key(key: str) -> tuple[str, int]:
                if "__cap" in key:
                    mid, _, idx_str = key.rpartition("__cap")
                    try:
                        return mid, int(idx_str)
                    except ValueError:
                        # Defensive: malformed key, treat as single-cap
                        return key, 0
                # Backward compat: legacy single-cap key
                return key, 0

            # Fast path (sidecar): a single .embs.npy [N,768] + .keys.json avoids
            # the per-key zip decompression of np.load(npz)[key], which is ~O(N^2)
            # for the 409970-key clean-L2 cache (~68 min). Sidecar load is seconds.
            # Build offline once via scripts/convert_caption_npz_to_npy.py.
            # Falls back to the legacy per-key npz path if sidecar is absent.
            sidecar_embs = cache_path.with_suffix(".embs.npy")
            sidecar_keys = cache_path.with_suffix(".keys.json")
            if sidecar_embs.exists() and sidecar_keys.exists():
                embs = np.load(sidecar_embs, mmap_mode="r")
                with open(sidecar_keys) as _kf:
                    keys = json.load(_kf)
                if len(keys) != embs.shape[0]:
                    raise ValueError(
                        f"caption sidecar length mismatch: {sidecar_keys} "
                        f"({len(keys)}) vs {sidecar_embs} ({embs.shape[0]})"
                    )
                for ki, key in enumerate(keys):
                    mid, idx = _split_caption_key(key)
                    raw_groups.setdefault(mid, []).append(
                        (idx, np.asarray(embs[ki], dtype=np.float32))
                    )
            else:
                with np.load(cache_path) as npz:
                    for key in npz.files:
                        vec = npz[key].astype(np.float32)
                        mid, idx = _split_caption_key(key)
                        raw_groups.setdefault(mid, []).append((idx, vec))
            for mid, lst in raw_groups.items():
                lst.sort(key=lambda x: x[0])
                self.caption_embs_multi[mid] = [emb for _, emb in lst]
            n_motions = len(self.caption_embs_multi)
            n_caps_total = sum(len(v) for v in self.caption_embs_multi.values())
            print(f"  [AnyTopDataset] loaded {n_caps_total} caption embeddings "
                  f"across {n_motions} motions (avg "
                  f"{n_caps_total/max(n_motions,1):.1f}/motion) from {cache_path}")
        self._caption_emb_dim = 768
        # random_caption: True (default) = per __getitem__ random.choice; False =
        # always idx 0 (primary). Train uses True (SALAD-style); val uses False
        # to keep val_denoise loss deterministic across epochs.
        self.random_caption = bool(random_caption)
        # random_crop: explicit override for the temporal crop policy on T>Tm
        # clips. None (default) = backward-compat: random crop when split=='train',
        # deterministic start=0 otherwise. True/False = force on/off regardless
        # of split. Use random_crop=True with split='all' to keep training-time
        # data augmentation (codex P1 2026-05-23: split='all' had silently
        # disabled random crop, hurting full-data training).
        self.random_crop = random_crop

        print(
            f"AnyTopDataset [{split}]: {len(self.samples)} motions, "
            f"{len({s['object_type'] for s in self.samples})} object types, "
            f"max_joints={self.max_joints}, num_frames={self.num_frames}"
        )

    def _normalize_cond_entry(self, c: dict, obj_type: str) -> dict:
        """Convert raw cond[type] to a typed dict with FK-ordered indexing.

        Also stashes `new_to_old_perm` so `__getitem__` can reorder its raw
        motion .npy J axis to match the FK-ordered skeleton arrays.
        """
        parents = np.asarray(c["parents"], dtype=np.int64)
        offsets = np.asarray(c["offsets"], dtype=np.float32)
        tpos = np.asarray(c["tpos_first_frame"], dtype=np.float32)
        jrel = np.asarray(c["joint_relations"], dtype=np.float32)
        jgd = np.asarray(c["joints_graph_dist"], dtype=np.float32)
        mean = np.asarray(c["mean"], dtype=np.float32)
        std = np.asarray(c["std"], dtype=np.float32)
        joint_names = list(c["joints_names"])
        J = parents.shape[0]
        if not (
            offsets.shape == (J, 3) and tpos.shape == (J, 13)
            and jrel.shape == (J, J) and jgd.shape == (J, J)
            and mean.shape == (J, 13) and std.shape == (J, 13)
            and len(joint_names) == J
        ):
            raise ValueError(
                f"cond[{obj_type}] shape mismatch: parents={parents.shape} "
                f"offsets={offsets.shape} tpos={tpos.shape} jrel={jrel.shape} "
                f"jgd={jgd.shape} mean={mean.shape} std={std.shape} "
                f"names={len(joint_names)}"
            )

        # jrel/jgd are read above only for the cond.npy schema/shape check;
        # the FK-ordered values are RE-DERIVED by `_build_derived` via
        # `_create_topology_edge_relations` (so the construction path and the
        # augmentation path share one derivation — and topology relations are a
        # pure function of `parents`, equivalent to AnyTop's cond.npy values).
        new_parents, new_joint_names, reindexed, new_to_old = (
            _normalize_parents_to_root_first(
                parents, joint_names,
                offsets=offsets, tpos=tpos, mean=mean, std=std,
            )
        )
        derived = _build_derived(
            new_parents, reindexed["offsets"], new_joint_names
        )
        return {
            "n_joints": J,
            "parents": new_parents,
            "joint_names": new_joint_names,
            "offsets": reindexed["offsets"],
            "tpos_first_frame": reindexed["tpos"],
            "mean": reindexed["mean"],
            "std": reindexed["std"],
            "new_to_old_perm": new_to_old,
            **derived,  # skeleton_features, adjacency, geodesic_dist,
                        # name_hashes, joint_relations, joints_graph_dist
        }

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        info = self.samples[idx]
        c = self.cond[info["object_type"]]
        Jm = self.max_joints
        Tm = self.num_frames

        # Reindex motion clip the same way we reindexed cond (BFS reorder).
        # cond["new_to_old_perm"] is built once at cond-normalize time and
        # stays cached across __getitem__ calls.
        if "new_to_old_perm" not in c:
            raise RuntimeError(
                "AnyTopDataset internal: cond entry missing 'new_to_old_perm'; "
                "this is a bug in cond normalization."
            )
        raw_motion = np.load(info["path"]).astype(np.float32)   # [T_var, J, 13]
        raw_motion = raw_motion[:, c["new_to_old_perm"], :]     # reorder J axis to FK order

        # ---------- Optional remove-joints augmentation (train only) ----------
        # `sk` is the effective skeleton dict for THIS sample: the shared cached
        # cond `c` when not augmenting, or a reduced-topology rebuild otherwise.
        # `_remove_joints_aug` never mutates `c` (works on local numpy copies).
        if self.augment and random.random() < self.augment_prob:
            raw_motion, sk = _remove_joints_aug(
                raw_motion, c, self.removal_rate, random.Random()
            )
        else:
            sk = c
        J_orig = sk["n_joints"]

        T_var = raw_motion.shape[0]

        # ---------- AnyTop normalized 13ch view (for the future end-to-end path) ----------
        mean = sk["mean"]               # [J_orig, 13] RAW (pre-normalize)
        std = sk["std"]                 # [J_orig, 13]
        std_safe = std + _STD_FLOOR
        normed_13 = (raw_motion - mean[None, :, :]) / std_safe[None, :, :]
        normed_13 = np.nan_to_num(normed_13).astype(np.float32)
        # tpos normalized for AnyTop extra key parity.
        tpos_norm = np.nan_to_num(
            ((sk["tpos_first_frame"] - mean) / std_safe).astype(np.float32)
        )

        # ---------- 6ch view: WORLD positions via AnyTop recovery (codex P1 #2) ----------
        # Recover from RAW 13ch (NOT normalized — AnyTop's recover assumes raw).
        world_pos = _recover_world_positions(raw_motion)        # [T_var, J_orig, 3]
        # World velocity: numerical diff × fps, zero-pad at t=0.
        world_vel = np.zeros_like(world_pos)
        if T_var >= 2:
            world_vel[1:] = (world_pos[1:] - world_pos[:-1]) * self.target_fps
            world_vel[0] = world_vel[1]
        # Stack into 6ch view in FK-ordered J axis.
        motion_pos_vel = np.concatenate([world_pos, world_vel], axis=-1)  # [T_var, J_orig, 6]

        # Per-joint contact (codex P1 #8): AnyTop channel 12 is per-joint
        # contact, NOT a single global flag. Pull the whole [T_var, J_orig].
        contact_per_joint_raw = raw_motion[:, :, 12].astype(np.float32)  # [T_var, J_orig]

        # ---------- Temporal crop/pad (shared across all derived fields) ----------
        if T_var > Tm:
            # Explicit random_crop override takes precedence; else fall back
            # to split-based default (random for train, deterministic for val/all).
            if self.random_crop is True:
                do_random = True
            elif self.random_crop is False:
                do_random = False
            else:
                do_random = (self.split == "train")
            if do_random:
                start = np.random.randint(0, T_var - Tm + 1)
            else:
                start = 0
            sl = slice(start, start + Tm)
            motion_pos_vel = motion_pos_vel[sl]
            contact_per_joint_raw = contact_per_joint_raw[sl]
            normed_13 = normed_13[sl]
            actual_T = Tm
        elif T_var < Tm:
            actual_T = T_var
            pad_pv = np.zeros((Tm - T_var, J_orig, 6), dtype=np.float32)
            pad_ct = np.zeros((Tm - T_var, J_orig), dtype=np.float32)
            pad_13 = np.zeros((Tm - T_var, J_orig, 13), dtype=np.float32)
            motion_pos_vel = np.concatenate([motion_pos_vel, pad_pv], axis=0)
            contact_per_joint_raw = np.concatenate([contact_per_joint_raw, pad_ct], axis=0)
            normed_13 = np.concatenate([normed_13, pad_13], axis=0)
        else:
            actual_T = Tm

        # ---------- Spatial pad to max_joints ----------
        motion_6ch = np.zeros((Tm, Jm, 6), dtype=np.float32)
        motion_6ch[:, :J_orig, :] = motion_pos_vel

        contact_per_joint_padded = np.zeros((Tm, Jm), dtype=np.float32)
        contact_per_joint_padded[:, :J_orig] = contact_per_joint_raw

        padded_normed_13 = np.zeros((Tm, Jm, 13), dtype=np.float32)
        padded_normed_13[:, :J_orig, :] = normed_13

        joint_mask = np.zeros(Jm, dtype=bool)
        joint_mask[:J_orig] = True
        frame_mask = np.zeros(Tm, dtype=bool)
        frame_mask[:actual_T] = True

        # ---- Static skeleton / graph fields (padded) — from `sk` (aug-aware) ----
        skel_feats_padded = np.zeros((Jm, 9), dtype=np.float32)
        skel_feats_padded[:J_orig] = sk["skeleton_features"]

        adjacency_padded = np.zeros((Jm, Jm), dtype=np.float32)
        adjacency_padded[:J_orig, :J_orig] = sk["adjacency"]

        geo_padded = np.zeros((Jm, Jm), dtype=np.float32)
        geo_padded[:J_orig, :J_orig] = sk["geodesic_dist"]
        anytop_gd_padded = np.zeros((Jm, Jm), dtype=np.float32)
        anytop_gd_padded[:J_orig, :J_orig] = sk["joints_graph_dist"]

        name_hashes_padded = np.zeros(Jm, dtype=np.int64)
        name_hashes_padded[:J_orig] = sk["name_hashes"]

        rest_offsets_padded = np.zeros((Jm, 3), dtype=np.float32)
        rest_offsets_padded[:J_orig] = sk["offsets"]

        # ---- Auxiliary derived fields ----
        # local_rotations_6d [T_max, J_max, 6] from raw channels 3:9 (raw is OK
        # — 6D rot doesn't carry RIFKE encoding ambiguity).
        rot6d_padded = np.zeros((Tm, Jm, 6), dtype=np.float32)
        # Take from un-padded part of padded_normed_13's *raw* — but we kept raw
        # already paged through crop in this branch. Use raw_motion sliced again
        # to keep the un-normalized 6D rot (safer for downstream FK loss).
        # We need raw motion sliced with the same temporal window. To avoid
        # re-slicing complexity, recompute now (cheap):
        raw_sliced = raw_motion  # before crop/pad
        # Apply same T-crop logic as above for raw_motion:
        if T_var > Tm:
            raw_sliced = raw_motion[start:start + Tm]
        elif T_var < Tm:
            pad_raw = np.zeros((Tm - T_var, J_orig, 13), dtype=np.float32)
            raw_sliced = np.concatenate([raw_motion, pad_raw], axis=0)
        rot6d_padded[:, :J_orig, :] = raw_sliced[:, :, 3:9]

        # bone_lengths [T_max, J_max] — constant per joint from offsets,
        # masked to valid frames.
        bone_per_joint = np.linalg.norm(sk["offsets"], axis=-1)  # [J_orig]
        bone_padded = np.zeros((Tm, Jm), dtype=np.float32)
        bone_padded[:, :J_orig] = bone_per_joint[None, :]
        bone_padded *= frame_mask.reshape(-1, 1).astype(np.float32)

        # foot_contact [T_max, 4] — legacy 4-leg schema kept ZERO for AnyTop
        # data (codex P1 #8). The per-joint signal lives in
        # `foot_contact_per_joint` below. Downstream contact loss should key
        # on the new field; the legacy 4-ch field is just schema-compat.
        contact_padded = np.zeros((Tm, 4), dtype=np.float32)

        # root_position / root_velocity now hold the RECOVERED root world
        # trajectory (channels 0:3 of motion_6ch at joint 0, which is root).
        # This is the same data the FK decoder will produce — exposing it
        # explicitly so downstream losses (e.g. root-trajectory regularization)
        # can target it directly.
        root_pos_padded = motion_6ch[:, 0, :3].copy()  # [Tm, 3]
        root_vel_padded = motion_6ch[:, 0, 3:6].copy()  # [Tm, 3]

        # ---- AnyTop-specific extra fields (passthrough for codex / future use) ----
        # anytop_x [J_max, 13, T_max] (J-first, T-last, per user spec).
        # Uses the NORMALIZED 13ch view (apply AnyTop's mean/std). This is the
        # tensor an AnyTop-style 13ch encoder would consume directly.
        anytop_x = np.zeros((Jm, 13, Tm), dtype=np.float32)
        anytop_x[:J_orig] = np.transpose(padded_normed_13[:, :J_orig, :], (1, 2, 0))
        # graph_dist padded [J_max, J_max] — same as geo_padded for now
        # joint_relations [J_max, J_max] padded (int-like 0..5)
        jrel_padded = np.zeros((Jm, Jm), dtype=np.float32)
        jrel_padded[:J_orig, :J_orig] = sk["joint_relations"]
        # tpos_first_frame [J_max, 13] padded (normalized)
        tpos_padded = np.zeros((Jm, 13), dtype=np.float32)
        tpos_padded[:J_orig] = tpos_norm
        # mean / std [J_max, 13] padded (RAW, un-normalized)
        mean_padded = np.zeros((Jm, 13), dtype=np.float32)
        mean_padded[:J_orig] = sk["mean"]
        std_padded = np.ones((Jm, 13), dtype=np.float32)  # ones to avoid div-by-0
        std_padded[:J_orig] = sk["std"]

        # Parent indices as Python list[int] of length J_orig (FK-ordered).
        parent_indices_list = [int(p) for p in sk["parents"]]
        joint_names_list = list(sk["joint_names"])
        canonical_names_list = joint_names_list[:]
        bone_lengths_rest_list = bone_per_joint.tolist()

        # M1.7 Phase-2: multi-caption SALAD-style. If `random_caption=True`
        # (train), pick a random caption per __getitem__; else (val) use
        # index 0 (= primary) for deterministic val_denoise across epochs.
        # `caption` (string, for display) tracks the SAME index as the embedding
        # so logs / animate titles reflect what the model actually saw.
        caps_emb_list = self.caption_embs_multi.get(info["motion_id"])
        caps_str_list = self.captions_multi.get(info["motion_id"])
        if caps_emb_list is not None and len(caps_emb_list) > 0:
            if self.random_caption and len(caps_emb_list) > 1:
                idx = random.randrange(len(caps_emb_list))
            else:
                idx = 0
            caption_emb = caps_emb_list[idx].astype(np.float32)
            has_text = True
            if caps_str_list is not None and idx < len(caps_str_list):
                caption = caps_str_list[idx]
            else:
                caption = self.captions.get(info["motion_id"], "")
        else:
            caption_emb = np.zeros(self._caption_emb_dim, dtype=np.float32)
            has_text = False
            caption = self.captions.get(info["motion_id"], "")

        return {
            # ---- GraphMotionBatch-compatible padded tensors ----
            "motion_features": torch.from_numpy(motion_6ch),               # [T, Jm, 6]
            "skeleton_features": torch.from_numpy(skel_feats_padded),      # [Jm, 9]
            "joint_mask": torch.from_numpy(joint_mask),                    # [Jm] bool
            "frame_mask": torch.from_numpy(frame_mask),                    # [Tm] bool
            "adjacency": torch.from_numpy(adjacency_padded),               # [Jm, Jm]
            "geodesic_dist": torch.from_numpy(geo_padded),                 # [Jm, Jm]
            "name_hashes": torch.from_numpy(name_hashes_padded),           # [Jm] int64
            "root_position": torch.from_numpy(root_pos_padded),            # [Tm, 3]
            "root_velocity": torch.from_numpy(root_vel_padded),            # [Tm, 3]
            "local_rotations_6d": torch.from_numpy(rot6d_padded),          # [Tm, Jm, 6]
            "foot_contact": torch.from_numpy(contact_padded),              # [Tm, 4]
            "bone_lengths": torch.from_numpy(bone_padded),                 # [Tm, Jm]
            "rest_offsets": torch.from_numpy(rest_offsets_padded),         # [Jm, 3]

            # ---- Batched scalars (collate -> [B] tensors) ----
            "num_joints": int(J_orig),
            "num_frames": int(actual_T),
            "fps": float(self.target_fps),
            "has_rotations": True,  # AnyTop always carries 6D rot in channels 3:9

            # ---- Per-sample lists ----
            "parent_indices": parent_indices_list,
            "joint_names": joint_names_list,
            "canonical_names": canonical_names_list,
            "bone_lengths_rest": bone_lengths_rest_list,

            # ---- Per-sample strings ----
            "text": caption,
            "skeleton_id": info["object_type"],
            "motion_id": info["motion_id"],

            # ---- AnyTop-specific extras (NOT validated by GraphMotionBatch) ----
            "anytop_x": torch.from_numpy(anytop_x),                        # [Jm, 13, Tm] NORMALIZED
            "anytop_graph_dist": torch.from_numpy(anytop_gd_padded),        # [Jm, Jm] (AnyTop clamped ≤5)
            "anytop_joint_relations": torch.from_numpy(jrel_padded),       # [Jm, Jm]
            "anytop_tpos_first_frame": torch.from_numpy(tpos_padded),      # [Jm, 13]
            "anytop_mean": torch.from_numpy(mean_padded),                  # [Jm, 13]
            "anytop_std": torch.from_numpy(std_padded),                    # [Jm, 13]
            # codex P1 #8: per-joint contact (AnyTop channel-12 source of truth)
            "foot_contact_per_joint": torch.from_numpy(contact_per_joint_padded),  # [Tm, Jm]
            # M1.7 Task 2: optional text condition (precomputed T5 caption embedding)
            "caption_emb": torch.from_numpy(caption_emb),                  # [768] f32
            "has_text": has_text,                                          # bool
            "object_type": info["object_type"],                            # str (alias of skeleton_id)
            "caption": caption,                                            # str
        }


def collate_fn(batch: list[dict]) -> dict:
    """Stack tensors, list everything else.

    Compatible with src/data/unified_dataset.py:collate_fn semantics so the
    resulting dict feeds GraphMotionBatch.from_collate_dict() unchanged.
    """
    result: dict = {}
    keys = batch[0].keys()
    for key in keys:
        v0 = batch[0][key]
        if key == "text":
            result[key] = [b[key] for b in batch]
        elif isinstance(v0, torch.Tensor):
            result[key] = torch.stack([b[key] for b in batch])
        elif isinstance(v0, bool):
            # bool ⊂ int in Python — must branch first or collate emits int64 ones.
            result[key] = torch.tensor([b[key] for b in batch], dtype=torch.bool)
        elif isinstance(v0, (int, float)):
            # Stay aligned with UnifiedMotionDataset.collate_fn: ints -> int64,
            # floats -> float32 via torch.tensor default rules.
            if isinstance(v0, int):
                result[key] = torch.tensor([b[key] for b in batch], dtype=torch.int64)
            else:
                result[key] = torch.tensor([b[key] for b in batch], dtype=torch.float32)
        else:
            result[key] = [b[key] for b in batch]
    return result


