"""
Skeleton graph representation for topology-agnostic motion processing.

Each skeleton is represented as a directed tree graph with:
- Joint features: rest offset, bone length, depth, degree, parent type, side tag
- Edge features: parent-child relations, geodesic distances
- Joint name embeddings: CLIP-encoded semantic features

This is the foundation for the TopoSlots slot assignment module.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class SkeletonGraph:
    """Unified skeleton graph representation for arbitrary topologies."""

    # Basic structure
    joint_names: list[str]  # [J] joint names
    parent_indices: list[int]  # [J] parent index (-1 for root)
    rest_offsets: np.ndarray  # [J, 3] rest-pose offsets from parent

    # Derived features (computed in __post_init__)
    num_joints: int = 0
    bone_lengths: np.ndarray = field(default_factory=lambda: np.array([]))  # [J]
    depths: np.ndarray = field(default_factory=lambda: np.array([]))  # [J] tree depth
    degrees: np.ndarray = field(default_factory=lambda: np.array([]))  # [J] num children
    adjacency: np.ndarray = field(default_factory=lambda: np.array([]))  # [J, J] binary
    geodesic_dist: np.ndarray = field(default_factory=lambda: np.array([]))  # [J, J]
    side_tags: list[str] = field(default_factory=list)  # [J] 'left'/'right'/'center'
    symmetry_pairs: list[tuple[int, int]] = field(default_factory=list)

    # Semantic features (filled by encode_joint_names)
    name_embeddings: Optional[np.ndarray] = None  # [J, D_clip]

    def __post_init__(self):
        self.num_joints = len(self.joint_names)
        J = self.num_joints

        if J == 0:
            return

        # Bone lengths
        self.bone_lengths = np.linalg.norm(self.rest_offsets, axis=-1)  # [J]

        # Tree depth via BFS from root
        self.depths = np.zeros(J, dtype=np.int32)
        for j in range(J):
            d = 0
            p = self.parent_indices[j]
            while p >= 0:
                d += 1
                p = self.parent_indices[p]
            self.depths[j] = d

        # Degree (number of children)
        self.degrees = np.zeros(J, dtype=np.int32)
        for j in range(J):
            p = self.parent_indices[j]
            if p >= 0:
                self.degrees[p] += 1

        # Adjacency matrix (undirected)
        self.adjacency = np.zeros((J, J), dtype=np.float32)
        for j in range(J):
            p = self.parent_indices[j]
            if p >= 0:
                self.adjacency[j, p] = 1.0
                self.adjacency[p, j] = 1.0

        # Geodesic distances via BFS
        self.geodesic_dist = self._compute_geodesic_distances()

        # Side tags from joint names
        self.side_tags = self._infer_side_tags()

        # Symmetry pairs
        self.symmetry_pairs = self._find_symmetry_pairs()

    def _compute_geodesic_distances(self) -> np.ndarray:
        """BFS-based geodesic distance on skeleton tree."""
        J = self.num_joints
        dist = np.full((J, J), fill_value=J + 1, dtype=np.int32)
        np.fill_diagonal(dist, 0)

        for i in range(J):
            # BFS from joint i
            queue = [i]
            visited = {i}
            while queue:
                curr = queue.pop(0)
                for j in range(J):
                    if self.adjacency[curr, j] > 0 and j not in visited:
                        dist[i, j] = dist[i, curr] + 1
                        visited.add(j)
                        queue.append(j)

        return dist.astype(np.float32)

    def _infer_side_tags(self) -> list[str]:
        """Infer left/right/center from joint names.

        Handles diverse naming conventions:
          - 'left'/'right' anywhere: LeftArm, left_hip, RightFoot
          - '_L'/'_R' suffix: Shoulder_L, UpperArm_R
          - '_l_'/'_r_' infix: Bip01_L_Thigh, BN_Tai_R_01
          - 'L'/'R' prefix before uppercase: LHipJoint, RThumb
          - '_L'/'_R' suffix before numbers: Leg_L_30_
        """
        import re
        tags = []
        for name in self.joint_names:
            n = name  # preserve case for regex
            nl = name.lower()

            side = 'center'
            # Priority 1: explicit 'left'/'right'
            if 'left' in nl or 'lft' in nl:
                side = 'left'
            elif 'right' in nl or 'rgt' in nl:
                side = 'right'
            # Priority 2: _L / _R suffix (with optional trailing digits/underscores)
            elif re.search(r'[_]L(?:[_\d]|$)', n):
                side = 'left'
            elif re.search(r'[_]R(?:[_\d]|$)', n):
                side = 'right'
            # Priority 3: _l_ / _r_ infix
            elif '_l_' in nl or nl.startswith('l_'):
                side = 'left'
            elif '_r_' in nl or nl.startswith('r_'):
                side = 'right'
            # Priority 4: L/R prefix before uppercase (LHipJoint, RThumb)
            elif re.match(r'^L[A-Z]', n):
                side = 'left'
            elif re.match(r'^R[A-Z]', n):
                side = 'right'
            # Priority 5: prefix_L / prefix_R patterns
            # NPC_LLeg1, NPC_RArm1, BN_LWing01, BN_RWing01, ArmL, ArmR, LegR, etc.
            elif re.search(r'(?:NPC|BN|Bn)_L[A-Z]', n):
                side = 'left'
            elif re.search(r'(?:NPC|BN|Bn)_R[A-Z]', n):
                side = 'right'
            # Priority 6: {Word}L{Upper} or {Word}R{Upper} pattern
            # Catches: ArmLClaw, ElkLFemur, ElkRScapula, LegR_01_, etc.
            elif re.search(r'[a-z0-9]L[A-Z_]', n) or re.search(r'[a-z0-9]L$', n):
                side = 'left'
            elif re.search(r'[a-z0-9]R[A-Z_]', n) or re.search(r'[a-z0-9]R$', n):
                side = 'right'

            tags.append(side)

        return tags

    def _find_symmetry_pairs(self) -> list[tuple[int, int]]:
        """Find symmetric joint pairs based on naming conventions.

        Handles: Left↔Right, _L↔_R, _l_↔_r_, L-prefix↔R-prefix.
        """
        import re
        pairs = []
        used = set()

        # Build replacement rules: (pattern, left_repl, right_repl)
        swap_rules = [
            # Full words
            (r'(?i)left', 'left', 'right'),
            (r'(?i)right', 'right', 'left'),
            # _L / _R suffix (preserving case/context)
            (r'_L(?=[_\d]|$)', '_L', '_R'),
            (r'_R(?=[_\d]|$)', '_R', '_L'),
            # _l_ / _r_ infix
            (r'_l_', '_l_', '_r_'),
            (r'_r_', '_r_', '_l_'),
            # L/R prefix before uppercase
            (r'^L(?=[A-Z])', 'L', 'R'),
            (r'^R(?=[A-Z])', 'R', 'L'),
            # BN_L/BN_R prefix (BN_LWing01 ↔ BN_RWing01)
            (r'BN_L(?=[A-Z])', 'BN_L', 'BN_R'),
            (r'BN_R(?=[A-Z])', 'BN_R', 'BN_L'),
            (r'Bn_L(?=[A-Z])', 'Bn_L', 'Bn_R'),
            (r'Bn_R(?=[A-Z])', 'Bn_R', 'Bn_L'),
            # NPC_L/NPC_R prefix (NPC_LLeg1 ↔ NPC_RLeg1)
            (r'NPC_L(?=[A-Z_])', 'NPC_L', 'NPC_R'),
            (r'NPC_R(?=[A-Z_])', 'NPC_R', 'NPC_L'),
            # {word}L{Upper} ↔ {word}R{Upper} (ElkLFemur↔ElkRFemur, ArmL↔ArmR)
            (r'(?<=[a-z0-9])L(?=[A-Z_]|$)', 'L', 'R'),
            (r'(?<=[a-z0-9])R(?=[A-Z_]|$)', 'R', 'L'),
        ]

        for i, name_i in enumerate(self.joint_names):
            if i in used:
                continue

            mirror_name = None
            for pattern, from_str, to_str in swap_rules:
                if re.search(pattern, name_i):
                    mirror_name = re.sub(pattern, to_str, name_i, count=1)
                    break

            if mirror_name and mirror_name != name_i:
                mirror_lower = mirror_name.lower()
                for j, name_j in enumerate(self.joint_names):
                    if j != i and j not in used and name_j.lower() == mirror_lower:
                        pairs.append((i, j))
                        used.add(i)
                        used.add(j)
                        break

        return pairs

    def get_edge_list(self) -> list[tuple[int, int]]:
        """Return parent-child edge list."""
        edges = []
        for j in range(self.num_joints):
            p = self.parent_indices[j]
            if p >= 0:
                edges.append((p, j))
        return edges

    def get_joint_features(self) -> np.ndarray:
        """
        Compute per-joint feature vector for skeleton encoder input.
        Returns: [J, D] where D = 3 (offset) + 1 (bone_len) + 1 (depth) + 1 (degree) + 3 (side one-hot)
        """
        J = self.num_joints

        # Side tags as one-hot: [left, right, center]
        side_onehot = np.zeros((J, 3), dtype=np.float32)
        for j, tag in enumerate(self.side_tags):
            if tag == 'left':
                side_onehot[j, 0] = 1.0
            elif tag == 'right':
                side_onehot[j, 1] = 1.0
            else:
                side_onehot[j, 2] = 1.0

        # Normalize rest offsets
        max_offset = np.max(np.abs(self.rest_offsets)) + 1e-8
        norm_offsets = self.rest_offsets / max_offset

        # Normalize bone lengths
        max_bone = np.max(self.bone_lengths) + 1e-8
        norm_bones = (self.bone_lengths / max_bone).reshape(-1, 1)

        # Normalize depth/degree
        max_depth = max(self.depths.max(), 1)
        norm_depths = (self.depths / max_depth).reshape(-1, 1).astype(np.float32)
        max_degree = max(self.degrees.max(), 1)
        norm_degrees = (self.degrees / max_degree).reshape(-1, 1).astype(np.float32)

        features = np.concatenate([
            norm_offsets,     # [J, 3]
            norm_bones,       # [J, 1]
            norm_depths,      # [J, 1]
            norm_degrees,     # [J, 1]
            side_onehot,      # [J, 3]
        ], axis=-1)  # [J, 9]

        return features

    def to_dict(self) -> dict:
        """Serialize to dict for saving."""
        return {
            'joint_names': self.joint_names,
            'parent_indices': self.parent_indices,
            'rest_offsets': self.rest_offsets,
            'bone_lengths': self.bone_lengths,
            'depths': self.depths,
            'degrees': self.degrees,
            'adjacency': self.adjacency,
            'geodesic_dist': self.geodesic_dist,
            'side_tags': self.side_tags,
            'symmetry_pairs': self.symmetry_pairs,
            'name_embeddings': self.name_embeddings,
        }

    @classmethod
    def from_dict(cls, d: dict) -> 'SkeletonGraph':
        """Deserialize from dict."""
        sg = cls(
            joint_names=d['joint_names'],
            parent_indices=d['parent_indices'],
            rest_offsets=d['rest_offsets'],
        )
        if d.get('name_embeddings') is not None:
            sg.name_embeddings = d['name_embeddings']
        return sg


# ============================================================
# Joint name normalization (alias table)
# ============================================================

JOINT_NAME_ALIASES = {
    # Mixamo naming
    'mixamorig:hips': 'pelvis',
    'mixamorig:spine': 'spine',
    'mixamorig:spine1': 'spine1',
    'mixamorig:spine2': 'chest',
    'mixamorig:neck': 'neck',
    'mixamorig:head': 'head',
    'mixamorig:leftshoulder': 'left shoulder',
    'mixamorig:leftarm': 'left upper arm',
    'mixamorig:leftforearm': 'left forearm',
    'mixamorig:lefthand': 'left hand',
    'mixamorig:rightshoulder': 'right shoulder',
    'mixamorig:rightarm': 'right upper arm',
    'mixamorig:rightforearm': 'right forearm',
    'mixamorig:righthand': 'right hand',
    'mixamorig:leftupleg': 'left upper leg',
    'mixamorig:leftleg': 'left lower leg',
    'mixamorig:leftfoot': 'left foot',
    'mixamorig:lefttoebase': 'left toe',
    'mixamorig:rightupleg': 'right upper leg',
    'mixamorig:rightleg': 'right lower leg',
    'mixamorig:rightfoot': 'right foot',
    'mixamorig:righttoebase': 'right toe',
    # SMPL naming
    'pelvis': 'pelvis',
    'l_hip': 'left hip',
    'r_hip': 'right hip',
    'spine1': 'lower spine',
    'l_knee': 'left knee',
    'r_knee': 'right knee',
    'spine2': 'upper spine',
    'l_ankle': 'left ankle',
    'r_ankle': 'right ankle',
    'spine3': 'chest',
    'l_foot': 'left foot',
    'r_foot': 'right foot',
    'neck': 'neck',
    'l_collar': 'left collar',
    'r_collar': 'right collar',
    'head': 'head',
    'l_shoulder': 'left shoulder',
    'r_shoulder': 'right shoulder',
    'l_elbow': 'left elbow',
    'r_elbow': 'right elbow',
    'l_wrist': 'left wrist',
    'r_wrist': 'right wrist',
    # Common animal naming
    'hips': 'pelvis',
    'spine': 'spine',
    'chest': 'chest',
    'leftforeleg': 'left fore leg',
    'rightforeleg': 'right fore leg',
    'lefthindleg': 'left hind leg',
    'righthindleg': 'right hind leg',
    'tail': 'tail',
    'tail1': 'tail base',
    'tail2': 'tail mid',
    'tail3': 'tail tip',
}


def normalize_joint_name(name: str) -> str:
    """Normalize joint name to a canonical human-readable form."""
    key = name.lower().strip().replace(' ', '')
    if key in JOINT_NAME_ALIASES:
        return JOINT_NAME_ALIASES[key]

    # Fallback: split camelCase and add spaces
    import re
    result = re.sub(r'([a-z])([A-Z])', r'\1 \2', name)
    result = re.sub(r'[_:]', ' ', result)
    return result.lower().strip()
