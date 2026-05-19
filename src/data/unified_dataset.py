"""
Unified multi-skeleton motion dataset for TopoSlots (Scheme C).

Motion representation:
  Slot token input : per-joint [local_pos(3) + velocity(3)] = 6D  (cross-skeleton)
  Decoder GT       : per-joint local_rotations_6d               (skeleton-specific, FK supervision)
  Root track       : root_position(3) + root_velocity(3)         (separate)
  Auxiliary        : foot_contact(4), bone_lengths, accelerations (losses)
"""

import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import Optional


class UnifiedMotionDataset(Dataset):
    """
    Multi-skeleton motion dataset with unified format.

    Each sample returns:
        - motion_features: [T, J, D] padded to max_joints
        - skeleton_features: [J, D_skel] padded to max_joints
        - joint_mask: [J] boolean mask (True = valid joint)
        - adjacency: [J, J] padded adjacency matrix
        - geodesic_dist: [J, J] padded geodesic distances
        - text: str (empty if unavailable)
        - metadata: dict
    """

    def __init__(
        self,
        data_dirs: list[str | Path],
        split: str = 'train',
        max_joints: int = 128,
        max_frames: int = 196,
        target_fps: float = 20.0,
        motion_dim: int = 6,  # local_pos (3) + velocity (3)
        normalize: bool = True,
    ):
        self.data_dirs = [Path(d) for d in data_dirs]
        self.split = split
        self.max_joints = max_joints
        self.max_frames = max_frames
        self.target_fps = target_fps
        self.motion_dim = motion_dim
        self.do_normalize = normalize

        # Load all samples
        self.samples = []
        self.skeletons = {}  # skeleton_id -> skeleton data
        self.stats = {}  # skeleton_id -> normalization stats

        for data_dir in self.data_dirs:
            self._load_data_source(data_dir)

        print(f"UnifiedMotionDataset [{split}]: {len(self.samples)} motions, "
              f"{len(self.skeletons)} skeleton types")

    def _load_data_source(self, data_dir: Path):
        """Load one data source (e.g., processed/humanml3d)."""
        if not data_dir.exists():
            print(f"  Warning: {data_dir} not found, skipping")
            return

        # Load skeleton(s)
        skel_path = data_dir / 'skeleton.npz'
        if skel_path.exists():
            skel_data = dict(np.load(skel_path, allow_pickle=True))
            self.skeletons[data_dir.name] = skel_data

        # Load per-species skeletons (for Zoo-style datasets)
        skeletons_dir = data_dir / 'skeletons'
        if skeletons_dir.exists():
            for sp_file in skeletons_dir.glob('*.npz'):
                sp_name = sp_file.stem
                self.skeletons[sp_name] = dict(np.load(sp_file, allow_pickle=True))

        # Load stats
        stats_path = data_dir / 'stats.npz'
        if stats_path.exists():
            self.stats[data_dir.name] = dict(np.load(stats_path))

        # Load split
        split_path = data_dir / 'splits' / f'{self.split}.txt'
        if not split_path.exists():
            # Fall back to all.txt
            split_path = data_dir / 'splits' / 'all.txt'
            if not split_path.exists():
                print(f"  Warning: no split file for {data_dir.name}, skipping")
                return

        motion_ids = []
        with open(split_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    motion_ids.append(line)

        for mid in motion_ids:
            motion_path = data_dir / 'motions' / f'{mid}.npz'
            if motion_path.exists():
                # Determine skeleton_id: use species if available (Zoo), else dataset name
                skel_id = data_dir.name
                if skeletons_dir.exists():
                    # Try to infer species from motion file (e.g., "Dog_0001" → "Dog")
                    m_data = dict(np.load(motion_path, allow_pickle=True))
                    species = str(m_data.get('species', ''))
                    if species and species in self.skeletons:
                        skel_id = species
                self.samples.append({
                    'motion_path': str(motion_path),
                    'motion_id': mid,
                    'data_source': data_dir.name,
                    'skeleton_id': skel_id,
                })

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        sample_info = self.samples[idx]

        # Load motion data
        data = dict(np.load(sample_info['motion_path'], allow_pickle=True))

        # Get skeleton info
        skeleton_id = sample_info['skeleton_id']
        skel_data = self.skeletons.get(skeleton_id, {})

        # Extract motion features
        local_pos = data['local_positions']  # [T, J, 3]
        velocities = data['velocities']       # [T, J, 3]
        T, J, _ = local_pos.shape

        # Normalize if stats available and enabled
        if self.do_normalize and skeleton_id in self.stats:
            stats = self.stats[skeleton_id]
            std_floor = 0.01  # prevent division by tiny std
            local_pos = (local_pos - stats['local_pos_mean']) / np.maximum(stats['local_pos_std'], std_floor)
            velocities = (velocities - stats['velocity_mean']) / np.maximum(stats['velocity_std'], std_floor)

        # Concatenate motion features: [T, J, 6]
        motion_features = np.concatenate([local_pos, velocities], axis=-1)

        # Crop/pad temporal dimension
        if T > self.max_frames:
            # Random crop during training
            if self.split == 'train':
                start = np.random.randint(0, T - self.max_frames)
            else:
                start = 0
            motion_features = motion_features[start:start + self.max_frames]
            actual_frames = self.max_frames
        else:
            actual_frames = T
            # Pad with zeros
            pad = np.zeros(
                (self.max_frames - T, J, self.motion_dim),
                dtype=np.float32,
            )
            motion_features = np.concatenate([motion_features, pad], axis=0)

        # Pad joint dimension
        padded_motion = np.zeros(
            (self.max_frames, self.max_joints, self.motion_dim),
            dtype=np.float32,
        )
        padded_motion[:, :J, :] = motion_features

        # Joint mask
        joint_mask = np.zeros(self.max_joints, dtype=np.bool_)
        joint_mask[:J] = True

        # Frame mask
        frame_mask = np.zeros(self.max_frames, dtype=np.bool_)
        frame_mask[:actual_frames] = True

        # Skeleton features
        skeleton_features = np.zeros(
            (self.max_joints, 9), dtype=np.float32
        )
        # Canonical name hashes for name embedding
        name_hashes = np.zeros(self.max_joints, dtype=np.int64)
        if 'joint_names' in skel_data:
            from .skeleton_graph import SkeletonGraph
            sg = SkeletonGraph.from_dict(skel_data)
            skel_feats = sg.get_joint_features()  # [J_skel, 9]
            J_skel = min(skel_feats.shape[0], J)
            skeleton_features[:J_skel] = skel_feats[:J_skel]

            # Hash canonical names to vocab buckets
            canon_names = [str(n) for n in skel_data.get('canonical_names', skel_data['joint_names'])]
            import hashlib
            for j_idx in range(min(len(canon_names), J)):
                h = int(hashlib.md5(canon_names[j_idx].encode()).hexdigest(), 16)
                name_hashes[j_idx] = h % 1024

        # Adjacency and geodesic distance matrices
        adjacency = np.zeros(
            (self.max_joints, self.max_joints), dtype=np.float32
        )
        geodesic_dist = np.zeros(
            (self.max_joints, self.max_joints), dtype=np.float32
        )
        if 'adjacency' in skel_data:
            adj = skel_data['adjacency']
            Ja = min(adj.shape[0], J)
            adjacency[:Ja, :Ja] = adj[:Ja, :Ja]
        if 'geodesic_dist' in skel_data:
            gdist = skel_data['geodesic_dist']
            Jg = min(gdist.shape[0], J)
            geodesic_dist[:Jg, :Jg] = gdist[:Jg, :Jg]

        # Text
        text = ''
        if 'texts' in data:
            texts_str = str(data['texts'])
            if texts_str:
                text_list = texts_str.split('|||')
                if text_list and text_list[0]:
                    # Random text during training
                    if self.split == 'train':
                        text = text_list[np.random.randint(len(text_list))]
                    else:
                        text = text_list[0]

        # --- Root track (separate from slot tokens) ---
        root_pos = data.get('root_position', np.zeros((T, 3), dtype=np.float32))
        root_vel = data.get('root_velocity', np.zeros((T, 3), dtype=np.float32))
        padded_root_pos = np.zeros((self.max_frames, 3), dtype=np.float32)
        padded_root_vel = np.zeros((self.max_frames, 3), dtype=np.float32)
        padded_root_pos[:actual_frames] = root_pos[:actual_frames]
        padded_root_vel[:actual_frames] = root_vel[:actual_frames]

        # --- Foot contact: [T, 4] (l_heel, l_toe, r_heel, r_toe) ---
        fc_raw = data.get('foot_contact', np.zeros((T, 4), dtype=np.float32))
        if fc_raw.shape[-1] == 2:
            # Legacy 2-channel → duplicate into 4-channel
            fc_4ch = np.zeros((fc_raw.shape[0], 4), dtype=np.float32)
            fc_4ch[:, 0] = fc_4ch[:, 1] = fc_raw[:, 0]
            fc_4ch[:, 2] = fc_4ch[:, 3] = fc_raw[:, 1]
            fc_raw = fc_4ch
        padded_contact = np.zeros((self.max_frames, 4), dtype=np.float32)
        padded_contact[:actual_frames] = fc_raw[:actual_frames]

        # --- Decoder GT: local rotations 6D (skeleton-specific, for FK supervision) ---
        rot_6d = data.get('local_rotations_6d', None)
        if rot_6d is not None:
            # [T, J-1, 6] → pad to [T_max, J_max, 6]
            Jr = rot_6d.shape[1]  # J-1 (non-root)
            padded_rot = np.zeros((self.max_frames, self.max_joints, 6), dtype=np.float32)
            T_rot = min(rot_6d.shape[0], actual_frames)
            padded_rot[:T_rot, :Jr, :] = rot_6d[:T_rot]
            has_rotations = True
        else:
            padded_rot = np.zeros((self.max_frames, self.max_joints, 6), dtype=np.float32)
            has_rotations = False

        # --- Bone lengths [T, J] ---
        bone_raw = data.get('bone_lengths', np.zeros((T, J), dtype=np.float32))
        padded_bones = np.zeros((self.max_frames, self.max_joints), dtype=np.float32)
        padded_bones[:actual_frames, :J] = bone_raw[:actual_frames]

        # --- Skeleton structure for benchmark eval (Phase 1-5b: BoneLenRel + Contact F1) ---
        # These are kept variable-length (per-sample) and collate_fn returns them as
        # list[list[...]]. eval_benchmark.py uses them; training ignores them.
        skel_parent_indices = skel_data.get('parent_indices', None)
        if skel_parent_indices is not None:
            parent_indices_list = np.asarray(skel_parent_indices, dtype=np.int64).tolist()[:J]
        else:
            parent_indices_list = [-1] + list(range(J - 1)) if J > 0 else []  # linear chain fallback

        skel_joint_names = skel_data.get('joint_names', None)
        if skel_joint_names is not None:
            joint_names_list = [str(n) for n in skel_joint_names][:J]
        else:
            joint_names_list = [''] * J

        skel_canonical_names = skel_data.get('canonical_names', None)
        if skel_canonical_names is not None:
            canonical_names_list = [str(n) for n in skel_canonical_names][:J]
        else:
            canonical_names_list = joint_names_list[:]

        skel_bone_lengths_rest = skel_data.get('bone_lengths', None)
        if skel_bone_lengths_rest is not None:
            bone_lengths_rest_list = np.asarray(skel_bone_lengths_rest, dtype=np.float32).tolist()[:J]
        else:
            bone_lengths_rest_list = [0.0] * J

        # Padded rest offsets [J_max, 3] — needed by FK head / paired-gate eval
        # (codex 019e2cdb must-fix #1). Additive: no existing behavior changes.
        rest_raw = np.asarray(skel_data.get('rest_offsets', np.zeros((J, 3))),
                              dtype=np.float32)
        padded_rest = np.zeros((self.max_joints, 3), dtype=np.float32)
        Jr = min(rest_raw.shape[0], J, self.max_joints)
        padded_rest[:Jr] = rest_raw[:Jr]

        return {
            # Slot token input: per-joint [local_pos(3) + velocity(3)] = 6D
            'motion_features': torch.from_numpy(padded_motion),         # [T, J_max, 6]
            # Skeleton graph
            'skeleton_features': torch.from_numpy(skeleton_features),   # [J_max, 9]
            'joint_mask': torch.from_numpy(joint_mask),                 # [J_max]
            'frame_mask': torch.from_numpy(frame_mask),                 # [T_max]
            'adjacency': torch.from_numpy(adjacency),                   # [J_max, J_max]
            'geodesic_dist': torch.from_numpy(geodesic_dist),           # [J_max, J_max]
            'name_hashes': torch.from_numpy(name_hashes),                 # [J_max] int64
            # Root track (separate)
            'root_position': torch.from_numpy(padded_root_pos),         # [T_max, 3]
            'root_velocity': torch.from_numpy(padded_root_vel),         # [T_max, 3]
            # Decoder GT (skeleton-specific)
            'local_rotations_6d': torch.from_numpy(padded_rot),         # [T_max, J_max, 6]
            'has_rotations': has_rotations,
            # Auxiliary
            'foot_contact': torch.from_numpy(padded_contact),           # [T_max, 4]
            'bone_lengths': torch.from_numpy(padded_bones),             # [T_max, J_max]
            # Skeleton structure (variable-length lists for benchmark eval)
            'parent_indices': parent_indices_list,                       # list[int] length J
            'joint_names': joint_names_list,                             # list[str] length J
            'canonical_names': canonical_names_list,                     # list[str] length J
            'bone_lengths_rest': bone_lengths_rest_list,                 # list[float] length J
            'rest_offsets': torch.from_numpy(padded_rest),               # [J_max, 3]
            # Metadata
            'text': text,
            'num_joints': J,
            'num_frames': actual_frames,
            'skeleton_id': skeleton_id,
            'motion_id': sample_info['motion_id'],
            'fps': float(data['fps']) if 'fps' in data else float(self.target_fps),
        }


    def get_topology_weights(self) -> list[float]:
        """Compute per-sample weights for topology-balanced sampling.

        Uses sqrt(N_max / N_skel) so rare topologies are upweighted but not extreme.
        """
        from collections import Counter
        skel_counts = Counter(s['skeleton_id'] for s in self.samples)
        n_max = max(skel_counts.values())
        weights = []
        for s in self.samples:
            n = skel_counts[s['skeleton_id']]
            weights.append(float(np.sqrt(n_max / n)))
        return weights


def collate_fn(batch: list[dict]) -> dict:
    """Custom collate function for variable-length text."""
    result = {}
    for key in batch[0]:
        if key == 'text':
            result[key] = [b[key] for b in batch]
        elif isinstance(batch[0][key], torch.Tensor):
            result[key] = torch.stack([b[key] for b in batch])
        elif isinstance(batch[0][key], (int, float)):
            result[key] = torch.tensor([b[key] for b in batch])
        else:
            result[key] = [b[key] for b in batch]
    return result
