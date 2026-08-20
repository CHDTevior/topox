"""Fixed-rig authority for Truebones KTJD-17 conversion.

The current BTJD corpus does not use every BVH joint's XYZ channel as a
time-varying bone translation.  Its physical rig comes from ``cond.npy``;
only the retained-root translation and the original BVH rotation channels are
dynamic authority.  This module makes that mixed provenance explicit and
keeps legacy 13D data out of the rotation/position construction path.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np


ACTIVE_COND_SHA256 = "161795a6507e24c2908f3837c9c999f19d411ecefd7943852f308942d8949bfb"
LEGACY_TRUEBONES_COND_SHA256 = "9dad7c833534edf90fa295e837d1c5e021306b9857aa40d8c3f88c17e5c33d02"
TRUEBONES_BTJD_MEAN_EDGE_TARGET = 0.2092142857142857
COND_GEOMETRY_TOL = 1e-8
# Coarse integrity fingerprint only.  The actual producer-vs-independent
# rotation acceptance gate is a direct full-array max-abs comparison at 1e-12;
# a coarser hash avoids false mismatches when equivalent evaluators straddle a
# decimal bin by a few ulps.
ROTATION_QUANTIZATION_STEP = 1e-8
RIGID_EDGE_MAX_NORM = 1e-4


class TruebonesFixedRigError(RuntimeError):
    """Conditioning geometry or fixed-rig algebra violates the contract."""


@dataclasses.dataclass(frozen=True)
class ForwardSpec:
    method: str
    anchor_names: tuple[str, ...]
    provenance: str
    legacy_anchor_indices: tuple[int, ...] | None = None


# Reviewed anatomy anchors are names, never fragile indices.  These remain a
# producer table; independent validators own a separate frozen copy.
TRUEBONES_FORWARD_SPECS: dict[str, ForwardSpec] = {
    "Alligator": ForwardSpec("lateral_pairs", ("R_momo", "L_momo", "R_hiji", "L_hiji"), "legacy_truebones_face_landmarks_reviewed_on_source_rest_t04"),
    "Anaconda": ForwardSpec("root_to_head", ("Hips", "BN_Tone_04"), "coiled_snake_root_to_tongue_endpoint_reviewed_t04"),
    "Bat": ForwardSpec("lateral_pairs", ("Bip01_R_Thigh", "Bip01_L_Thigh", "BN_R_UpperArm_01", "BN_L_UpperArm_01"), "legacy_truebones_face_landmarks_reviewed_on_source_rest_t04"),
    "Bird": ForwardSpec("lateral_pairs", ("Bip01_R_Thigh", "Bip01_L_Thigh", "BN_Forearm_R_01", "BN_Forearm_L_01"), "legacy_truebones_face_landmarks_reviewed_on_source_rest_t04"),
    "Buffalo": ForwardSpec("lateral_pairs", ("Bip01_R_Thigh", "Bip01_L_Thigh", "Bip01_R_Clavicle", "Bip01_L_Clavicle"), "legacy_truebones_face_landmarks_reviewed_on_source_rest_t04"),
    "Buzzard": ForwardSpec("lateral_pairs", ("Bip01_R_Thigh", "Bip01_L_Thigh", "BN_Wing_R_02", "BN_Wing_L_02"), "legacy_truebones_face_landmarks_reviewed_on_source_rest_t04"),
    "Cat": ForwardSpec("lateral_pairs", ("Bip01_R_Thigh", "Bip01_L_Thigh", "Bip01_R_Clavicle", "Bip01_L_Clavicle"), "legacy_truebones_face_landmarks_reviewed_on_source_rest_t04"),
    "Chicken": ForwardSpec("lateral_pairs", ("Bip01_R_Thigh", "Bip01_L_Thigh", "BN_Finger_R_01", "BN_Finger_L_01"), "legacy_truebones_face_landmarks_reviewed_on_source_rest_t04"),
    "Coyote": ForwardSpec("lateral_pairs", ("Bip01_R_Thigh", "Bip01_L_Thigh", "Bip01_R_Clavicle", "Bip01_L_Clavicle"), "legacy_truebones_face_landmarks_reviewed_on_source_rest_t04"),
    "Crocodile": ForwardSpec("lateral_pairs", ("Bip01_R_Thigh", "Bip01_L_Thigh", "Bip01_R_UpperArm", "Bip01_L_UpperArm"), "legacy_truebones_face_landmarks_reviewed_on_source_rest_t04"),
    "Dragon": ForwardSpec("lateral_pairs", ("Bip01_R_Thigh", "Bip01_L_Thigh", "Bip01_R_UpperArm", "Bip01_L_UpperArm"), "legacy_truebones_face_landmarks_reviewed_on_source_rest_t04"),
    "Eagle": ForwardSpec("lateral_pairs", ("Bip01_R_Thigh", "Bip01_L_Thigh", "BN_Wing_R_02", "BN_Wing_L_02"), "legacy_truebones_face_landmarks_reviewed_on_source_rest_t04"),
    "Flamingo": ForwardSpec("lateral_pairs", ("Bip01_R_Thigh", "Bip01_L_Thigh", "BN_Forearm_R_02", "BN_Forearm_L_02"), "legacy_truebones_face_landmarks_reviewed_on_source_rest_t04"),
    "Fox": ForwardSpec("lateral_pairs", ("Bip01_R_Thigh", "Bip01_L_Thigh", "Bip01_R_UpperArm", "Bip01_L_UpperArm"), "legacy_truebones_face_landmarks_reviewed_on_source_rest_t04"),
    "Gazelle": ForwardSpec("lateral_pairs", ("Bip01_R_Thigh", "Bip01_L_Thigh", "Bip01_R_UpperArm", "Bip01_L_UpperArm"), "legacy_truebones_face_landmarks_reviewed_on_source_rest_t04"),
    "Hamster": ForwardSpec("lateral_pairs", ("Bip01_R_Thigh", "Bip01_L_Thigh", "Bip01_R_UpperArm", "Bip01_L_UpperArm"), "legacy_truebones_face_landmarks_reviewed_on_source_rest_t04"),
    "HermitCrab": ForwardSpec("lateral_pairs", ("BN_Leg_R_09", "BN_Leg_L_09", "BN_Crab_pincers_R_02", "BN_Crab_pincers_L_02"), "legacy_truebones_face_landmarks_reviewed_on_source_rest_t04"),
    "Hippopotamus": ForwardSpec("lateral_pairs", ("Bip01_R_Thigh", "Bip01_L_Thigh", "Bip01_R_Clavicle", "Bip01_L_Clavicle"), "legacy_truebones_face_landmarks_reviewed_on_source_rest_t04"),
    "Horse": ForwardSpec("lateral_pairs", ("Bip01_R_Thigh", "Bip01_L_Thigh", "Bip01_R_UpperArm", "Bip01_L_UpperArm"), "legacy_truebones_face_landmarks_reviewed_on_source_rest_t04"),
    "Hound": ForwardSpec("lateral_pairs", ("Bip01_R_Thigh", "Bip01_L_Thigh", "Bip01_R_UpperArm", "Bip01_L_UpperArm"), "legacy_truebones_face_landmarks_reviewed_on_source_rest_t04"),
    "KingCobra": ForwardSpec("root_to_head", ("Hips", "BN_Tongue_02"), "snake_root_to_tongue_endpoint_reviewed_t04"),
    "Lion": ForwardSpec("lateral_pairs", ("Bip01_R_Thigh", "Bip01_L_Thigh", "Bip01_R_UpperArm", "Bip01_L_UpperArm"), "legacy_truebones_face_landmarks_reviewed_on_source_rest_t04"),
    "Lynx": ForwardSpec("lateral_pairs", ("Bip01_R_Thigh", "Bip01_L_Thigh", "Bip01_R_UpperArm", "Bip01_L_UpperArm"), "legacy_truebones_face_landmarks_reviewed_on_source_rest_t04"),
    "Mammoth": ForwardSpec("lateral_pairs", ("Bip01_R_Thigh", "Bip01_L_Thigh", "Bip01_R_UpperArm", "Bip01_L_UpperArm"), "legacy_truebones_face_landmarks_reviewed_on_source_rest_t04"),
    "Ostrich": ForwardSpec("lateral_pairs", ("Bip01_R_Thigh", "Bip01_L_Thigh", "BN_Forearm_R_02", "BN_Forearm_L_02"), "legacy_truebones_face_landmarks_reviewed_on_source_rest_t04"),
    "Parrot": ForwardSpec("lateral_pairs", ("Bip01_R_Thigh", "Bip01_L_Thigh", "BN_Wing_R_02", "BN_Wing_L_02"), "legacy_truebones_face_landmarks_reviewed_on_source_rest_t04"),
    "Parrot2": ForwardSpec("lateral_pairs", ("Bip01_R_Thigh", "Bip01_L_Thigh", "BN_Wing_R_02", "BN_Wing_L_02"), "legacy_truebones_face_landmarks_reviewed_on_source_rest_t04"),
    "Pteranodon": ForwardSpec("lateral_pairs", ("jt_Thigh_R", "jt_Thigh_L", "jt_Elbow_R", "jt_Elbow_L"), "legacy_truebones_face_landmarks_reviewed_on_source_rest_t04"),
    "Scorpion": ForwardSpec("lateral_pairs", ("Bip01_R_Thigh_4", "Bip01_L_Thigh1_4", "Bip01_R_Forearm", "Bip01_L_Forearm"), "legacy_truebones_face_landmarks_reviewed_on_source_rest_t04"),
    "SpiderG": ForwardSpec("lateral_pairs", ("Bip01_R_Thigh", "Bip01_L_Thigh", "Bip01_R_UpperArm", "Bip01_L_UpperArm"), "legacy_truebones_face_landmarks_reviewed_on_source_rest_t04"),
    "Tukan": ForwardSpec("lateral_pairs", ("R_momo", "L_momo", "R_kata", "L_kata"), "legacy_truebones_face_landmarks_reviewed_on_source_rest_t04"),
}


FULL_TRUEBONES_FORWARD_SPEC_VERSION = "ktjd17-truebones-forward-v1"
_FULL_FORWARD_PROVENANCE = (
    "legacy_truebones_face_joint_indices_resolved_to_frozen_cond_names_"
    "and_reaudited_before_full_conversion_v1"
)

# Full current-BTJD Truebones catalog.  The historical 31-rig table above is
# deliberately retained unchanged because it is pinned by the prototype and
# freeze evidence.  This catalog expands the same named-anchor convention to
# all 70 Truebones rigs.  The legacy indices are provenance only: conversion
# resolves and uses names, then checks that those names still occupy the
# recorded indices in the frozen conditioning payload.
_FULL_LEGACY_LATERAL_ANCHORS: dict[
    str, tuple[tuple[int, int, int, int], tuple[str, str, str, str]]
] = {
    "Alligator": ((8, 11, 17, 20), ("R_momo", "L_momo", "R_hiji", "L_hiji")),
    "Ant": ((9, 15, 23, 30), ("Bip01_R_Thigh", "Bip01_L_Thigh", "Bip01_R_UpperArm", "Bip01_L_UpperArm")),
    "Bat": ((6, 15, 26, 34), ("Bip01_R_Thigh", "Bip01_L_Thigh", "BN_R_UpperArm_01", "BN_L_UpperArm_01")),
    "Bear": ((8, 2, 36, 56), ("NPC_RLeg1", "NPC_LLeg1", "NPC_RArmCollarbone", "NPC_LArmCollarbone")),
    "Bird": ((15, 35, 6, 11), ("Bip01_R_Thigh", "Bip01_L_Thigh", "BN_Forearm_R_01", "BN_Forearm_L_01")),
    "BrownBear": ((2, 7, 15, 23), ("Bip01_R_Thigh", "Bip01_L_Thigh", "Bip01_R_UpperArm", "Bip01_L_UpperArm")),
    "Buffalo": ((6, 12, 20, 26), ("Bip01_R_Thigh", "Bip01_L_Thigh", "Bip01_R_Clavicle", "Bip01_L_Clavicle")),
    "Buzzard": ((7, 23, 41, 47), ("Bip01_R_Thigh", "Bip01_L_Thigh", "BN_Wing_R_02", "BN_Wing_L_02")),
    "Camel": ((9, 15, 32, 26), ("Bip01_R_Thigh", "Bip01_L_Thigh", "Bip01_R_Clavicle", "Bip01_L_Clavicle")),
    "Cat": ((6, 12, 21, 27), ("Bip01_R_Thigh", "Bip01_L_Thigh", "Bip01_R_Clavicle", "Bip01_L_Clavicle")),
    "Centipede": ((7, 2, 41, 47), ("BN_Thigh_R_01", "BN_Thigh_L_01", "Bip01_R_UpperArm", "Bip01_L_UpperArm")),
    "Chicken": ((5, 17, 30, 32), ("Bip01_R_Thigh", "Bip01_L_Thigh", "BN_Finger_R_01", "BN_Finger_L_01")),
    "Comodoa": ((11, 1, 33, 43), ("Bip01_R_Thigh", "Bip01_L_Thigh", "Bip01_R_Clavicle", "Bip01_L_Clavicle")),
    "Coyote": ((5, 11, 20, 27), ("Bip01_R_Thigh", "Bip01_L_Thigh", "Bip01_R_Clavicle", "Bip01_L_Clavicle")),
    "Crab": ((14, 20, 51, 47), ("BN_Leg_R_11", "BN_Leg_L_11", "BN_Arm_R_02", "BN_Arm_L_02")),
    "Cricket": ((20, 25, 32, 36), ("Bip01_R_Thigh", "Bip01_L_Thigh", "Bip01_R_UpperArm", "Bip01_L_UpperArm")),
    "Crocodile": ((7, 12, 21, 27), ("Bip01_R_Thigh", "Bip01_L_Thigh", "Bip01_R_UpperArm", "Bip01_L_UpperArm")),
    "Deer": ((4, 9, 29, 35), ("ElkRFemur", "ElkLFemur", "ElkRScapula", "ElkLScapula")),
    "Dragon": ((10, 23, 47, 83), ("Bip01_R_Thigh", "Bip01_L_Thigh", "Bip01_R_UpperArm", "Bip01_L_UpperArm")),
    "Eagle": ((7, 20, 35, 41), ("Bip01_R_Thigh", "Bip01_L_Thigh", "BN_Wing_R_02", "BN_Wing_L_02")),
    "Elephant": ((6, 10, 32, 36), ("Bip01_R_Thigh", "Bip01_L_Thigh", "Bip01_R_UpperArm", "Bip01_L_UpperArm")),
    "FireAnt": ((15, 19, 25, 29), ("Bip01_R_Thigh_Rear", "Bip01_L_Thigh_Rear", "Bip01_R_ForeArm", "Bip01_L_ForeArm")),
    "Flamingo": ((15, 22, 10, 6), ("Bip01_R_Thigh", "Bip01_L_Thigh", "BN_Forearm_R_02", "BN_Forearm_L_02")),
    "Fox": ((27, 33, 15, 8), ("Bip01_R_Thigh", "Bip01_L_Thigh", "Bip01_R_UpperArm", "Bip01_L_UpperArm")),
    "Gazelle": ((4, 10, 20, 26), ("Bip01_R_Thigh", "Bip01_L_Thigh", "Bip01_R_UpperArm", "Bip01_L_UpperArm")),
    "Giantbee": ((11, 16, 3, 1), ("Bip01_R_Thigh", "Bip01_L_Thigh", "BN_Wing_R_01", "BN_Wing_L_01")),
    "Goat": ((24, 19, 14, 8), ("Bip01_R_Thigh", "Bip01_L_Thigh", "Bip01_R_UpperArm", "Bip01_L_UpperArm")),
    "Hamster": ((3, 9, 19, 25), ("Bip01_R_Thigh", "Bip01_L_Thigh", "Bip01_R_UpperArm", "Bip01_L_UpperArm")),
    "HermitCrab": ((51, 46, 8, 12), ("BN_Leg_R_09", "BN_Leg_L_09", "BN_Crab_pincers_R_02", "BN_Crab_pincers_L_02")),
    "Hippopotamus": ((5, 11, 28, 34), ("Bip01_R_Thigh", "Bip01_L_Thigh", "Bip01_R_Clavicle", "Bip01_L_Clavicle")),
    "Horse": ((10, 16, 33, 41), ("Bip01_R_Thigh", "Bip01_L_Thigh", "Bip01_R_UpperArm", "Bip01_L_UpperArm")),
    "Hound": ((3, 9, 19, 25), ("Bip01_R_Thigh", "Bip01_L_Thigh", "Bip01_R_UpperArm", "Bip01_L_UpperArm")),
    "Isopetra": ((48, 55, 18, 26), ("Bip01_R_Thigh", "Bip01_L_Thigh", "Bip01_R_UpperArm", "Bip01_L_UpperArm")),
    "Jaguar": ((6, 12, 22, 28), ("Bip01_R_Thigh", "Bip01_L_Thigh", "Bip01_R_UpperArm", "Bip01_L_UpperArm")),
    "Leapord": ((7, 13, 25, 31), ("Bip01_R_Thigh", "Bip01_L_Thigh", "Bip01_R_Clavicle", "Bip01_L_Clavicle")),
    "Lion": ((6, 11, 19, 24), ("Bip01_R_Thigh", "Bip01_L_Thigh", "Bip01_R_UpperArm", "Bip01_L_UpperArm")),
    "Lynx": ((2, 8, 18, 24), ("Bip01_R_Thigh", "Bip01_L_Thigh", "Bip01_R_UpperArm", "Bip01_L_UpperArm")),
    "Mammoth": ((7, 11, 34, 38), ("Bip01_R_Thigh", "Bip01_L_Thigh", "Bip01_R_UpperArm", "Bip01_L_UpperArm")),
    "Monkey": ((9, 21, 56, 36), ("Bip01_R_Thigh", "Bip01_L_Thigh", "Bip01_R_UpperArm", "Bip01_L_UpperArm")),
    "Ostrich": ((6, 16, 36, 28), ("Bip01_R_Thigh", "Bip01_L_Thigh", "BN_Forearm_R_02", "BN_Forearm_L_02")),
    "Parrot": ((9, 25, 65, 43), ("Bip01_R_Thigh", "Bip01_L_Thigh", "BN_Wing_R_02", "BN_Wing_L_02")),
    "Parrot2": ((7, 23, 42, 48), ("Bip01_R_Thigh", "Bip01_L_Thigh", "BN_Wing_R_02", "BN_Wing_L_02")),
    "Pigeon": ((3, 4, 1, 6), ("RightLeg", "LeftLeg", "RightArm", "LeftArm")),
    "Pirrana": ((19, 20, 4, 5), ("harabireR", "harabireL", "munabireR", "munabireL")),
    "PolarBear": ((3, 9, 19, 25), ("Bip01_R_Thigh", "Bip01_L_Thigh", "Bip01_R_UpperArm", "Bip01_L_UpperArm")),
    "PolarBearB": ((3, 8, 17, 23), ("Bip01_R_Thigh", "Bip01_L_Thigh", "Bip01_R_UpperArm", "Bip01_L_UpperArm")),
    "Pteranodon": ((16, 5, 40, 35), ("jt_Thigh_R", "jt_Thigh_L", "jt_Elbow_R", "jt_Elbow_L")),
    "Puppy": ((5, 11, 20, 26), ("Bip01_R_Thigh", "Bip01_L_Thigh", "Bip01_R_UpperArm", "Bip01_L_UpperArm")),
    "Raindeer": ((3, 9, 18, 24), ("Bip01_R_Thigh", "Bip01_L_Thigh", "Bip01_R_UpperArm", "Bip01_L_UpperArm")),
    "Raptor": ((13, 19, 13, 19), ("Bip01_R_Thigh", "Bip01_L_Thigh", "Bip01_R_Thigh", "Bip01_L_Thigh")),
    "Raptor2": ((52, 40, 23, 13), ("jt_Thigh_R", "jt_Thigh_L", "jt_Shoulder_R", "jt_Shoulder_L")),
    "Raptor3": ((53, 41, 24, 14), ("jt_Thigh_R", "jt_Thigh_L", "jt_Shoulder_R", "jt_Shoulder_L")),
    "Rat": ((12, 15, 9, 6), ("RightUpLeg", "LeftUpLeg", "RightForeArm", "LeftForeArm")),
    "Rhino": ((5, 11, 21, 27), ("Bip01_R_Thigh", "Bip01_L_Thigh", "Bip01_R_Clavicle", "Bip01_L_Clavicle")),
    "Roach": ((2, 6, 29, 25), ("Bip01_R_Thigh01", "Bip01_L_Thigh01", "Bip01_R_UpperArm", "Bip01_L_UpperArm")),
    "SabreToothTiger": ((7, 2, 37, 51), ("Sabrecat_RightThigh_RThi_", "Sabrecat_LeftThigh_LThi_", "Sabrecat_RightClavicle_RClv_", "Sabrecat_LeftClavicle_LClv_")),
    "SandMouse": ((7, 13, 30, 34), ("Bip01_R_Thigh", "Bip01_L_Thigh", "Bip01_R_UpperArm", "Bip01_L_UpperArm")),
    "Scorpion": ((58, 29, 20, 25), ("Bip01_R_Thigh_4", "Bip01_L_Thigh1_4", "Bip01_R_Forearm", "Bip01_L_Forearm")),
    "Scorpion-2": ((55, 23, 48, 16), ("jt_HindLeg1_R", "jt_HindLeg1_L", "jt_FrontLeg2_R", "jt_FrontLeg2_L")),
    "Skunk": ((10, 15, 28, 32), ("Bip01_R_Thigh", "Bip01_L_Thigh", "Bip01_R_UpperArm", "Bip01_L_UpperArm")),
    "Spider": ((21, 27, 5, 9), ("Leg_R_30_", "Leg_L_30_", "ArmR_01_", "ArmL_01_")),
    "SpiderG": ((13, 19, 27, 33), ("Bip01_R_Thigh", "Bip01_L_Thigh", "Bip01_R_UpperArm", "Bip01_L_UpperArm")),
    "Stego": ((7, 12, 27, 21), ("Bip01_R_Thigh", "Bip01_L_Thigh", "Bip01_R_UpperArm", "Bip01_L_UpperArm")),
    "Trex": ((38, 50, 23, 15), ("jt_Thigh_R", "jt_Thigh_L", "jt_Shoulder_R", "jt_Shoulder_L")),
    "Tricera": ((6, 11, 24, 28), ("Bip01_R_Thigh", "Bip01_L_Thigh", "Bip01_R_UpperArm", "Bip01_L_UpperArm")),
    "Tukan": ((4, 6, 9, 11), ("R_momo", "L_momo", "R_kata", "L_kata")),
    "Turtle": ((31, 40, 12, 22), ("Bip01_R_Thigh", "Bip01_L_Thigh", "Bip01_R_UpperArm", "Bip01_L_UpperArm")),
    "Tyranno": ((7, 20, 37, 44), ("Bip01_R_Thigh", "Bip01_L_Thigh", "Bip01_R_UpperArm", "Bip01_L_UpperArm")),
}

TRUEBONES_FULL_FORWARD_SPECS: dict[str, ForwardSpec] = {
    rig_id: ForwardSpec(
        "lateral_pairs",
        anchor_names,
        _FULL_FORWARD_PROVENANCE,
        legacy_anchor_indices,
    )
    for rig_id, (legacy_anchor_indices, anchor_names) in (
        _FULL_LEGACY_LATERAL_ANCHORS.items()
    )
}
TRUEBONES_FULL_FORWARD_SPECS.update(
    {
        # The legacy Anaconda FACE_JOINTS row repeats (13,26).  The physical
        # axis is tail-tip -> tongue-tip, not Hips -> tongue-tip; the latter
        # misses the frozen conditioning forward by ~0.373.
        "Anaconda": ForwardSpec(
            "root_to_head",
            ("BN_Tail_13", "BN_Tone_04"),
            "legacy_truebones_tail13_to_tongue_axis_corrected_and_reaudited_v1",
            (13, 26),
        ),
        # KingCobra's paired neck landmarks do not define a longitudinal axis.
        # The reviewed physical root -> tongue endpoint is the whole-body axis.
        "KingCobra": ForwardSpec(
            "root_to_head",
            ("Hips", "BN_Tongue_02"),
            "kingcobra_physical_root_to_tongue_axis_reaudited_v1",
            None,
        ),
    }
)


def validate_full_forward_spec_catalog(
    joint_names_by_rig: Mapping[str, Sequence[str]],
) -> None:
    """Fail closed if the 70-rig named catalog drifts from frozen cond order."""
    expected = set(joint_names_by_rig)
    actual = set(TRUEBONES_FULL_FORWARD_SPECS)
    if actual != expected:
        raise TruebonesFixedRigError(
            "full forward catalog/conditioning rig mismatch: "
            f"missing={sorted(expected - actual)}, extra={sorted(actual - expected)}"
        )
    for rig_id, names_value in joint_names_by_rig.items():
        names = tuple(str(name) for name in names_value)
        spec = TRUEBONES_FULL_FORWARD_SPECS[rig_id]
        for name in spec.anchor_names:
            hits = [index for index, candidate in enumerate(names) if candidate == name]
            if len(hits) != 1:
                raise TruebonesFixedRigError(
                    f"{rig_id}: full forward anchor {name!r} resolves to {hits}"
                )
        if spec.legacy_anchor_indices is not None:
            observed = tuple(names[index] for index in spec.legacy_anchor_indices)
            if observed != spec.anchor_names:
                raise TruebonesFixedRigError(
                    f"{rig_id}: legacy index/name drift "
                    f"{observed} != {spec.anchor_names}"
                )


@dataclasses.dataclass(frozen=True)
class FixedRigGeometry:
    rig_id: str
    joint_names: tuple[str, ...]
    parents: np.ndarray
    offsets: np.ndarray
    rest_positions: np.ndarray
    ground_shift_y: float
    payload_sha256: str
    metrics: dict[str, float]


@dataclasses.dataclass(frozen=True)
class ConditioningCatalog:
    active_path: Path
    active_sha256: str
    legacy_path: Path
    legacy_sha256: str
    active_entries: Mapping[str, Mapping[str, Any]]
    legacy_rig_count: int
    active_rig_count: int

    def authority_record(self) -> dict[str, Any]:
        return {
            "authority_kind": "current_btjd_fixed_physical_rig_geometry_only",
            "active_cond_path": str(self.active_path),
            "active_cond_sha256": self.active_sha256,
            "expected_active_cond_sha256": ACTIVE_COND_SHA256,
            "legacy_truebones_cond_path": str(self.legacy_path),
            "legacy_truebones_cond_sha256": self.legacy_sha256,
            "expected_legacy_truebones_cond_sha256": LEGACY_TRUEBONES_COND_SHA256,
            "legacy_rig_count": self.legacy_rig_count,
            "active_rig_count": self.active_rig_count,
            "legacy_keys_present_in_active": True,
            "legacy_geometry_payloads_exact_in_active": True,
            "allowed_fields": [
                "joints_names",
                "parents",
                "offsets",
                "tpos_first_frame[:,0:3]",
            ],
            "forbidden_authority_fields": [
                "tpos_first_frame[:,3:9]",
                "mean",
                "std",
                "legacy_btjd_motion_channels",
            ],
        }

    def rig(
        self,
        rig_id: str,
        *,
        expected_names: Sequence[str],
        expected_parents: Sequence[int],
    ) -> FixedRigGeometry:
        try:
            entry = self.active_entries[rig_id]
        except KeyError as exc:
            raise TruebonesFixedRigError(
                f"{rig_id}: absent from active conditioning geometry"
            ) from exc
        return _fixed_geometry(
            rig_id,
            entry,
            expected_names=expected_names,
            expected_parents=expected_parents,
        )


@dataclasses.dataclass(frozen=True)
class FixedRigMotion:
    C: np.ndarray
    alpha: float
    o: np.ndarray
    P_rest_global: np.ndarray
    R_rest_global: np.ndarray
    R_rest_local: np.ndarray
    offset_parent_local: np.ndarray
    P_authoritative: np.ndarray
    R_global: np.ndarray
    source_forward: np.ndarray
    forward_anchor_indices: tuple[int, ...]
    metrics: dict[str, float]
    rotation_signatures: dict[str, Any]
    provenance: dict[str, Any]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _array_record(value: Any) -> dict[str, Any]:
    array = np.ascontiguousarray(np.asarray(value))
    return {
        "shape": list(array.shape),
        "dtype": array.dtype.str,
        "sha256": hashlib.sha256(array.tobytes(order="C")).hexdigest(),
    }


def conditioning_payload_sha256(entry: Mapping[str, Any]) -> str:
    payload = {
        "parents": _array_record(entry["parents"]),
        "offsets": _array_record(entry["offsets"]),
        "tpos_first_frame": _array_record(entry["tpos_first_frame"]),
        "joints_names": [str(value) for value in entry["joints_names"]],
    }
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _same_conditioning_payload(
    left: Mapping[str, Any], right: Mapping[str, Any]
) -> bool:
    if [str(x) for x in left["joints_names"]] != [
        str(x) for x in right["joints_names"]
    ]:
        return False
    for key in ("parents", "offsets", "tpos_first_frame"):
        if _array_record(left[key]) != _array_record(right[key]):
            return False
    return True


def load_conditioning_catalog(
    active_path: str | Path,
    *,
    expected_active_sha256: str,
    legacy_path: str | Path | None = None,
) -> ConditioningCatalog:
    active = Path(active_path).expanduser().resolve()
    legacy = (
        Path(legacy_path).expanduser().resolve()
        if legacy_path is not None
        else active.parent.parent.joinpath("anytop_truebones", "cond.npy").resolve()
    )
    if not active.is_file() or not legacy.is_file():
        raise TruebonesFixedRigError(
            f"conditioning authority files are missing: active={active}, legacy={legacy}"
        )
    active_sha = _sha256_file(active)
    legacy_sha = _sha256_file(legacy)
    if active_sha != expected_active_sha256 or active_sha != ACTIVE_COND_SHA256:
        raise TruebonesFixedRigError(
            f"active cond hash drifted: {active_sha} != {expected_active_sha256} "
            f"!= frozen {ACTIVE_COND_SHA256}"
        )
    if legacy_sha != LEGACY_TRUEBONES_COND_SHA256:
        raise TruebonesFixedRigError(
            f"legacy Truebones cond hash drifted: {legacy_sha}"
        )
    try:
        active_entries = np.load(active, allow_pickle=True).item()
        legacy_entries = np.load(legacy, allow_pickle=True).item()
    except Exception as exc:  # noqa: BLE001
        raise TruebonesFixedRigError(f"cannot load conditioning authority: {exc}") from exc
    if _sha256_file(active) != active_sha or _sha256_file(legacy) != legacy_sha:
        raise TruebonesFixedRigError("conditioning authority changed while loading")
    if not isinstance(active_entries, Mapping) or not isinstance(legacy_entries, Mapping):
        raise TruebonesFixedRigError("cond.npy must contain a rig mapping")
    missing = sorted(set(legacy_entries) - set(active_entries))
    if missing:
        raise TruebonesFixedRigError(
            f"active cond omits legacy Truebones rigs: {missing[:10]}"
        )
    drifted = [
        rig_id
        for rig_id in sorted(legacy_entries)
        if not _same_conditioning_payload(legacy_entries[rig_id], active_entries[rig_id])
    ]
    if drifted:
        raise TruebonesFixedRigError(
            f"legacy Truebones conditioning payloads drifted in active cond: {drifted[:10]}"
        )
    return ConditioningCatalog(
        active_path=active,
        active_sha256=active_sha,
        legacy_path=legacy,
        legacy_sha256=legacy_sha,
        active_entries=active_entries,
        legacy_rig_count=len(legacy_entries),
        active_rig_count=len(active_entries),
    )


def _fixed_geometry(
    rig_id: str,
    entry: Mapping[str, Any],
    *,
    expected_names: Sequence[str],
    expected_parents: Sequence[int],
) -> FixedRigGeometry:
    names = tuple(str(value) for value in entry["joints_names"])
    parents = np.asarray(entry["parents"], dtype=np.int64)
    offsets = np.asarray(entry["offsets"], dtype=np.float64)
    tpose = np.asarray(entry["tpos_first_frame"], dtype=np.float64)
    if names != tuple(str(value) for value in expected_names):
        raise TruebonesFixedRigError(f"{rig_id}: cond joint names differ from manifest")
    if not np.array_equal(parents, np.asarray(expected_parents, dtype=np.int64)):
        raise TruebonesFixedRigError(f"{rig_id}: cond parents differ from manifest")
    joint_count = len(names)
    if parents.shape != (joint_count,) or offsets.shape != (joint_count, 3):
        raise TruebonesFixedRigError(f"{rig_id}: invalid cond topology shapes")
    if tpose.ndim != 2 or tpose.shape[0] != joint_count or tpose.shape[1] < 3:
        raise TruebonesFixedRigError(f"{rig_id}: invalid cond tpos shape {tpose.shape}")
    if int(parents[0]) != -1 or any(
        not 0 <= int(parents[child]) < child for child in range(1, joint_count)
    ):
        raise TruebonesFixedRigError(f"{rig_id}: invalid cond parent-before-child tree")
    if not np.isfinite(offsets).all() or not np.isfinite(tpose[:, :3]).all():
        raise TruebonesFixedRigError(f"{rig_id}: non-finite cond geometry")

    raw_positions = tpose[:, :3].copy()
    cumulative = np.empty_like(raw_positions)
    cumulative[0] = raw_positions[0]
    for child in range(1, joint_count):
        cumulative[child] = cumulative[int(parents[child])] + offsets[child]
    delta = cumulative - raw_positions
    cumulative_max_abs = float(np.max(np.abs(delta)))
    cumulative_max_norm = float(np.max(np.linalg.norm(delta, axis=-1)))
    if max(cumulative_max_abs, cumulative_max_norm) > COND_GEOMETRY_TOL:
        raise TruebonesFixedRigError(
            f"{rig_id}: cond offsets do not reproduce tpos: "
            f"abs={cumulative_max_abs}, norm={cumulative_max_norm}"
        )

    ground_shift_y = -float(np.min(raw_positions[:, 1]))
    rest_positions = raw_positions.copy()
    rest_positions[:, 1] += ground_shift_y
    root_xz = float(np.max(np.abs(rest_positions[0, [0, 2]])))
    ground = abs(float(np.min(rest_positions[:, 1])))
    edges = np.asarray(
        [
            np.linalg.norm(rest_positions[child] - rest_positions[int(parents[child])])
            for child in range(1, joint_count)
        ],
        dtype=np.float64,
    )
    mean_edge = float(np.mean(edges))
    mean_edge_error = abs(mean_edge - TRUEBONES_BTJD_MEAN_EDGE_TARGET)
    if root_xz > COND_GEOMETRY_TOL or ground > COND_GEOMETRY_TOL:
        raise TruebonesFixedRigError(
            f"{rig_id}: cond root/ground gate failed: root_xz={root_xz}, ground={ground}"
        )
    if mean_edge_error > COND_GEOMETRY_TOL:
        raise TruebonesFixedRigError(
            f"{rig_id}: cond mean edge drifted: {mean_edge}"
        )
    return FixedRigGeometry(
        rig_id=rig_id,
        joint_names=names,
        parents=parents,
        offsets=offsets,
        rest_positions=rest_positions,
        ground_shift_y=ground_shift_y,
        payload_sha256=conditioning_payload_sha256(entry),
        metrics={
            "cond_offsets_to_tpos_max_abs": cumulative_max_abs,
            "cond_offsets_to_tpos_max_norm": cumulative_max_norm,
            "cond_ground_shift_y": ground_shift_y,
            "cond_ground_min_y_abs": ground,
            "cond_root_xz_max_abs": root_xz,
            "cond_mean_nonroot_edge_length": mean_edge,
            "cond_mean_edge_target_abs_error": mean_edge_error,
        },
    )


def forward_from_rest(
    joint_names: Sequence[str], positions: np.ndarray, spec: ForwardSpec
) -> tuple[np.ndarray, tuple[int, ...]]:
    lookup: dict[str, list[int]] = defaultdict(list)
    for index, name in enumerate(joint_names):
        lookup[str(name)].append(index)
    indices: list[int] = []
    for name in spec.anchor_names:
        hits = lookup.get(name, [])
        if len(hits) != 1:
            raise TruebonesFixedRigError(
                f"forward anchor {name!r} must resolve once, got {hits}"
            )
        indices.append(hits[0])
    points = np.asarray(positions, dtype=np.float64)
    if spec.method == "lateral_pairs":
        if len(indices) != 4:
            raise TruebonesFixedRigError("lateral_pairs requires four anchors")
        across = (
            points[indices[0]] - points[indices[1]]
            + points[indices[2]] - points[indices[3]]
        )
        forward = np.cross(np.asarray([0.0, 1.0, 0.0]), across)
    elif spec.method == "root_to_head":
        if len(indices) != 2:
            raise TruebonesFixedRigError("root_to_head requires two anchors")
        forward = points[indices[1]] - points[indices[0]]
    elif spec.method == "declared_plus_z":
        forward = np.asarray([0.0, 0.0, 1.0])
    else:
        raise TruebonesFixedRigError(f"unknown forward method {spec.method!r}")
    horizontal = np.asarray([forward[0], 0.0, forward[2]], dtype=np.float64)
    norm = float(np.linalg.norm(horizontal))
    scale = float(np.linalg.norm(np.ptp(points, axis=0)))
    if not math.isfinite(norm) or norm <= 1e-10 * scale:
        raise TruebonesFixedRigError(f"degenerate horizontal forward: {norm}")
    return horizontal / norm, tuple(indices)


def yaw_basis_to_plus_z(source_forward: np.ndarray) -> np.ndarray:
    forward = np.asarray(source_forward, dtype=np.float64)
    fx, fz = float(forward[0]), float(forward[2])
    norm = math.hypot(fx, fz)
    if not math.isfinite(norm) or norm <= 0.0:
        raise TruebonesFixedRigError("source forward has zero XZ norm")
    fx, fz = fx / norm, fz / norm
    C = np.asarray(
        [[fz, 0.0, -fx], [0.0, 1.0, 0.0], [fx, 0.0, fz]],
        dtype=np.float64,
    )
    orth = float(np.max(np.abs(C.T @ C - np.eye(3))))
    determinant = float(np.linalg.det(C))
    if orth > 1e-12 or abs(abs(determinant) - 1.0) > 1e-12:
        raise TruebonesFixedRigError(
            f"invalid source-to-canonical basis: orth={orth}, det={determinant}"
        )
    return C


def _rest_local_arrays(
    parents: np.ndarray, positions: np.ndarray, rotations: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    joint_count = len(parents)
    local_rotations = np.empty((joint_count, 3, 3), dtype=np.float64)
    offsets = np.zeros((joint_count, 3), dtype=np.float64)
    local_rotations[0] = rotations[0]
    for child in range(1, joint_count):
        parent = int(parents[child])
        parent_inverse = rotations[parent].T
        local_rotations[child] = parent_inverse @ rotations[child]
        offsets[child] = parent_inverse @ (positions[child] - positions[parent])
    return local_rotations, offsets


def _fixed_fk(
    parents: np.ndarray,
    root_positions: np.ndarray,
    global_rotations: np.ndarray,
    offsets: np.ndarray,
) -> np.ndarray:
    roots = np.asarray(root_positions, dtype=np.float64)
    rotations = np.asarray(global_rotations, dtype=np.float64)
    single = roots.ndim == 1
    if single:
        roots = roots[None]
        rotations = rotations[None]
    result = np.empty((len(roots), len(parents), 3), dtype=np.float64)
    result[:, 0] = roots
    for child in range(1, len(parents)):
        parent = int(parents[child])
        result[:, child] = result[:, parent] + np.einsum(
            "tij,j->ti", rotations[:, parent], offsets[child]
        )
    return result[0] if single else result


def quantized_rotation_sha256(rotations: np.ndarray) -> str:
    values = np.asarray(rotations, dtype=np.float64)
    if values.shape[-2:] != (3, 3) or not np.isfinite(values).all():
        raise TruebonesFixedRigError("rotation signature input must be finite [...,3,3]")
    quantized = np.ascontiguousarray(
        np.rint(values / ROTATION_QUANTIZATION_STEP).astype("<i8", copy=False)
    )
    digest = hashlib.sha256()
    digest.update(json.dumps(list(values.shape), separators=(",", ":")).encode("ascii"))
    digest.update(quantized.tobytes(order="C"))
    return digest.hexdigest()


def build_fixed_rig_motion(
    parsed: Any,
    fixed: FixedRigGeometry,
    spec: ForwardSpec,
) -> FixedRigMotion:
    """Build canonical fixed-rig motion from raw rotations and root only."""
    parents = np.asarray(parsed.parents, dtype=np.int64)
    raw_rest_positions = np.asarray(parsed.rest_global_positions, dtype=np.float64)
    raw_rest_rotations = np.asarray(parsed.rest_global_rotations, dtype=np.float64)
    source_forward, anchor_indices = forward_from_rest(
        parsed.joint_names, raw_rest_positions, spec
    )
    C = yaw_basis_to_plus_z(source_forward)
    raw_edges = np.asarray(
        [
            np.linalg.norm(
                raw_rest_positions[child] - raw_rest_positions[int(parents[child])]
            )
            for child in range(1, len(parents))
        ],
        dtype=np.float64,
    )
    raw_mean_edge = float(np.mean(raw_edges))
    if not math.isfinite(raw_mean_edge) or raw_mean_edge <= 0.0:
        raise TruebonesFixedRigError(f"invalid raw rest mean edge {raw_mean_edge}")
    alpha = TRUEBONES_BTJD_MEAN_EDGE_TARGET / raw_mean_edge
    P_rest_global = np.asarray(fixed.rest_positions, dtype=np.float64).copy()
    o = raw_rest_positions[0] - (C.T @ P_rest_global[0]) / alpha
    R_rest_global = np.einsum(
        "ab,jbc,dc->jad", C, raw_rest_rotations, C
    )
    R_rest_local, offset_parent_local = _rest_local_arrays(
        parents, P_rest_global, R_rest_global
    )
    rest_fk64 = _fixed_fk(
        parents, P_rest_global[0], R_rest_global, offset_parent_local
    )
    s_rig = float(np.linalg.norm(np.ptp(P_rest_global, axis=0)))
    rest_fk64_norm = float(
        np.max(np.linalg.norm(rest_fk64 - P_rest_global, axis=-1)) / s_rig
    )
    rest_fk32 = _fixed_fk(
        parents,
        P_rest_global[0].astype(np.float32).astype(np.float64),
        R_rest_global.astype(np.float32).astype(np.float64),
        offset_parent_local.astype(np.float32).astype(np.float64),
    )
    rest_fk32_norm = float(
        np.max(
            np.linalg.norm(
                rest_fk32 - P_rest_global.astype(np.float32).astype(np.float64),
                axis=-1,
            )
        )
        / s_rig
    )
    if rest_fk64_norm > 1e-10 or rest_fk32_norm > 1e-5:
        raise TruebonesFixedRigError(
            f"fixed-rest FK failed: f64={rest_fk64_norm}, f32={rest_fk32_norm}"
        )

    raw_global_rotations = np.asarray(parsed.global_rotations, dtype=np.float64)
    R_global = np.einsum("ab,tjbc,dc->tjad", C, raw_global_rotations, C)
    raw_root = np.asarray(parsed.source_positions[:, 0], dtype=np.float64)
    root_translation = np.asarray(parsed.root_translation, dtype=np.float64)
    root_source_error = float(np.max(np.abs(raw_root - root_translation)))
    if root_source_error > 1e-12:
        raise TruebonesFixedRigError(
            f"retained raw-root translation drifted: {root_source_error}"
        )
    authoritative_root = alpha * ((raw_root - o) @ C.T)
    P_authoritative = _fixed_fk(
        parents, authoritative_root, R_global, offset_parent_local
    )
    rest_lengths = np.asarray(
        [
            np.linalg.norm(P_rest_global[child] - P_rest_global[int(parents[child])])
            for child in range(1, len(parents))
        ],
        dtype=np.float64,
    )
    motion_lengths = np.stack(
        [
            np.linalg.norm(
                P_authoritative[:, child]
                - P_authoritative[:, int(parents[child])],
                axis=-1,
            )
            for child in range(1, len(parents))
        ],
        axis=-1,
    )
    rigid_edge_norm = float(np.max(np.abs(motion_lengths - rest_lengths)) / s_rig)
    if rigid_edge_norm > RIGID_EDGE_MAX_NORM:
        raise TruebonesFixedRigError(
            f"fixed-rig rigid-edge gate failed: {rigid_edge_norm}"
        )

    raw_xyz_canonical = alpha * (
        (np.asarray(parsed.source_positions, dtype=np.float64) - o) @ C.T
    )
    raw_xyz_discrepancy = np.linalg.norm(
        raw_xyz_canonical - P_authoritative, axis=-1
    )
    cond_forward, _ = forward_from_rest(parsed.joint_names, P_rest_global, spec)
    cond_forward_error = float(
        np.max(np.abs(cond_forward - np.asarray([0.0, 0.0, 1.0])))
    )
    if parsed.rest_status == "explicit_tpose_frame" and cond_forward_error > 1e-6:
        raise TruebonesFixedRigError(
            f"conditioning rest is not canonical +Z: {cond_forward_error}"
        )

    nonroot_count = int(
        parsed.diagnostics.get("nonroot_position_channel_joint_count", 0)
    )
    if nonroot_count <= 0:
        raise TruebonesFixedRigError(
            "Truebones audit did not enumerate ignored non-root XYZ channels"
        )
    metrics = {
        **fixed.metrics,
        "source_raw_rest_mean_nonroot_edge_length": raw_mean_edge,
        "source_to_canonical_alpha": float(alpha),
        "s_rig": s_rig,
        "rest_fk_float64_max_norm": rest_fk64_norm,
        "rest_fk_float32_max_norm": rest_fk32_norm,
        "motion_rigid_edge_max_norm": rigid_edge_norm,
        "raw_root_translation_max_abs": root_source_error,
        "ignored_nonroot_xyz_joint_count": float(nonroot_count),
        "ignored_nonroot_xyz_sample_count": float(
            parsed.diagnostics.get("nonroot_position_channel_sample_count", 0)
        ),
        "ignored_nonroot_xyz_max_frame_variation_norm": float(
            parsed.diagnostics.get("nonroot_position_channel_max_frame_variation_norm", 0.0)
        ),
        "raw_xyz_vs_authoritative_mpjpe_norm": float(
            np.mean(raw_xyz_discrepancy) / s_rig
        ),
        "raw_xyz_vs_authoritative_max_norm": float(
            np.max(raw_xyz_discrepancy) / s_rig
        ),
        "conditioning_forward_to_plus_z_max_abs": cond_forward_error,
    }
    signatures = {
        "quantization_step": ROTATION_QUANTIZATION_STEP,
        "local_rotation_sha256": quantized_rotation_sha256(parsed.local_rotations),
        "global_rotation_sha256": quantized_rotation_sha256(parsed.global_rotations),
        "rest_local_rotation_sha256": quantized_rotation_sha256(
            parsed.rest_local_rotations
        ),
        "rest_global_rotation_sha256": quantized_rotation_sha256(
            parsed.rest_global_rotations
        ),
    }
    provenance = {
        "position_authority": "cond_fixed_geometry_plus_raw_retained_root_xyz_only",
        "rotation_authority": "original_bvh_declared_rotation_channels_only",
        "rest_position_authority": "active_cond_tpos_first_frame_xyz_ground_shifted",
        "rest_rotation_authority": "selected_original_bvh_rest_rotation_channels",
        "authoritative_fk_formula": "P_child=P_parent+R_global_parent@offset_parent_local",
        "raw_nonroot_xyz_role": "diagnostic_only_never_rest_offset_or_motion_authority",
        "legacy_btjd_role": "optional_static_rig_witness_only",
        "forbidden_inputs_used": False,
        "conditioning_payload_sha256": fixed.payload_sha256,
        "ground_shift_y": fixed.ground_shift_y,
        "forward_method": spec.method,
        "forward_anchor_names": list(spec.anchor_names),
        "forward_spec_provenance": spec.provenance,
    }
    return FixedRigMotion(
        C=C,
        alpha=float(alpha),
        o=np.asarray(o, dtype=np.float64),
        P_rest_global=P_rest_global,
        R_rest_global=R_rest_global,
        R_rest_local=R_rest_local,
        offset_parent_local=offset_parent_local,
        P_authoritative=P_authoritative,
        R_global=R_global,
        source_forward=source_forward,
        forward_anchor_indices=anchor_indices,
        metrics=metrics,
        rotation_signatures=signatures,
        provenance=provenance,
    )
