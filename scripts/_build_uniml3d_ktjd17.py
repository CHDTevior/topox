#!/usr/bin/env python3
"""Convert selected UniML3D stage-1 Objaverse exports into the compact KTJD-17 contract.

Raw physical units are preserved. Training uses rest normalization, not per-cell whitening.
No Blender, IK, surrogate rotations, or changes to the frozen codec are involved.
"""
import argparse
import hashlib
import json
import shutil
import sys
import time
from collections import Counter, defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.ktjd17.canonical_skeleton import derive_rest_local_arrays
from src.data.ktjd17.codec import (Ktjd17CodecError, SmootherConfig, encode_ktjd17_channels,
    fk_from_global_rotations, local_to_global_rotations, resample_root_and_local_rotations,
    world_velocity)
from src.data.ktjd17.decoder import decode_ktjd17


class Reject(ValueError):
    """A source clip/rig cannot meet the requested corpus contract."""


# 数据清理门槛（单位=平均骨长），见 encode() 内的长注释。审查 2026-09-20 要求：实际生效值必须可追溯 ——
# converter_sha256 只钉得住源文件里的默认值，钉不住 env 覆盖后的值，于是 0.3 跑出来的语料和 0.5 跑出来的
# 从产物上无法区分。现改为 argparse（本脚本其它旋钮一律 argparse，env 曾是唯一例外），并写进 generation.json。
# 复审 2026-09-20：不留 env fallback。argparse 已是唯一旋钮，env 只制造"默认值本身可能被改过"的二阶歧义
# （generation.json 记的是实际生效值，所以本语料溯源完整，但别人不传 flag 重跑时就说不清了）；
# 而且 KTJD17_RIGIDIFY_MAX=abc 会在 **import 期** 炸掉每一个 import 本模块的脚本。
RIGIDIFY_MAX_DEFAULT = 0.5
# 单关节上限：均值口径会稀释"只有一两个关节错得离谱"的情形。审查实测全部 1,927 条被拒 clip：
# 纯 0.5 均值门槛放进来的那批里，单关节最差可达 8.08 骨长，而 user 实际看过的四条最差只有
# 0.080/0.564/0.967/1.361 —— 约 5% 被放行的比人眼审过的最差还差 6 倍。加这道上限代价 38/1767 = 2.2%。
RIGIDIFY_JOINT_CAP_DEFAULT = 2.0


def json_write(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def source_ready(raw,clip):
    path=raw/clip["source_npz"]
    return path.is_file() and path.stat().st_size==clip["source_bytes"]


def bfs_order(parents):
    p = np.asarray(parents)
    if p.ndim != 1 or p.dtype.kind not in "iu" or len(p) < 2 or len(p) > 142:
        raise Reject("invalid_joint_count_or_parent_array")
    roots = np.flatnonzero(p == -1)
    if len(roots) != 1:
        raise Reject("multiple_or_missing_roots")
    children = defaultdict(list)
    for j, parent in enumerate(p):
        if parent < -1 or parent >= len(p) or parent == j:
            raise Reject("invalid_parent_edge")
        children[int(parent)].append(j)
    order, queue = [], deque([int(roots[0])])
    while queue:
        j = queue.popleft()
        order.append(j)
        queue.extend(children[j])
    if len(order) != len(p):
        raise Reject("disconnected_cycle")
    perm = np.asarray(order)
    inverse = np.empty(len(p), int)
    inverse[perm] = np.arange(len(p))
    out = np.array([-1] + [inverse[p[j]] for j in perm[1:]], np.int32)
    return perm, out


def matrices(q):
    q = np.asarray(q, np.float64)
    if q.shape[-1:] != (4,) or not np.isfinite(q).all():
        raise Reject("invalid_quaternion")
    norm = np.linalg.norm(q, axis=-1)
    if np.max(np.abs(norm - 1)) > 1e-3:
        raise Reject("quaternion_not_unit")
    return Rotation.from_quat(q[..., [1, 2, 3, 0]].reshape(-1, 4)).as_matrix().reshape(q.shape[:-1] + (3, 3))


def read_source(path):
    with np.load(path, allow_pickle=False) as z:
        keys = ("names", "parents", "rest_local_pos", "rest_local_rot", "anim_local_pos",
                "anim_local_rot", "fps", "action_name")
        x = {k: np.array(z[k]) for k in keys}
    names = x["names"].astype(str)
    J = len(names)
    if len(set(names)) != J:
        raise Reject("duplicate_source_joint_names")
    if x["parents"].shape != (J,) or x["rest_local_pos"].shape != (J, 3) or x["rest_local_rot"].shape != (J, 4):
        raise Reject("invalid_rest_shape")
    T = len(x["anim_local_pos"])
    if T < 2 or x["anim_local_pos"].shape != (T, J, 3) or x["anim_local_rot"].shape != (T, J, 4):
        raise Reject("invalid_animation_shape")
    for k in ("rest_local_pos", "anim_local_pos"):
        if not np.isfinite(x[k]).all():
            raise Reject("nonfinite_positions")
    return x


def select_heading_carrier(skeleton, sources, C):
    """Use face-pair LCA when a stationary root anchors independently oriented body bones.

    A single carrier is selected per rig across its clips. Never use a flapping fin or leaf
    merely because its horizontal projection is nonzero.
    """
    names = list(skeleton["joint_names"])
    p = skeleton["parents"]
    def ancestors(j):
        result=[]
        while j >= 0:
            result.append(int(j)); j=p[j]
        return result
    r,l=[names.index(n) for n in skeleton["face_joint_names"]]
    left=set(ancestors(l))
    candidate=next(j for j in ancestors(r) if j in left)
    if candidate == 0:
        return 0, "face_pair_lca_is_root"
    u=skeleton["R_rest_global"][candidate].T @ [0.,0.,1.]
    root_u=skeleton["R_rest_global"][0].T @ [0.,0.,1.]
    all_valid=frames=0
    max_root_motion=max_difference=0.
    for source in sources:
        if set(source["names"]) != set(names):
            continue
        order=[list(source["names"]).index(n) for n in names]
        source_parents=[None if j == -1 else str(source["names"][j]) for j in source["parents"]]
        expected=[None if j == -1 else names[j] for j in p]
        if [source_parents[j] for j in order] != expected:
            continue
        try:
            local=matrices(source["anim_local_rot"][:,order])
        except Reject:
            continue  # Full conversion records this explicit source rejection later.
        local[:,0]=C @ local[:,0]
        G=local_to_global_rotations(p,local)
        max_root_motion=max(max_root_motion,float(np.max(np.abs(G[:,0]-G[:1,0]))))
        f=G[:,candidate] @ u; fr=G[:,0] @ root_u
        valid=np.linalg.norm(f[:,[0,2]],axis=-1)>=.05
        all_valid+=int(valid.sum());frames+=len(f)
        if valid.any():
            diff=np.arctan2(f[:,0],f[:,2])-np.arctan2(fr[:,0],fr[:,2])
            max_difference=max(max_difference,float(np.abs(np.arctan2(np.sin(diff),np.cos(diff)))[valid].max()))
    if frames and max_root_motion < 1e-5 and all_valid/frames >= .9 and max_difference > np.deg2rad(1):
        return candidate, "stationary_root_body_heading_on_face_pair_lca"
    return 0, "root_carries_heading_or_no_supported_alternative"


def make_skeleton(source, face, clean_mapping):
    perm, parents = bfs_order(source["parents"])
    names = source["names"].astype(str)[perm]
    offsets = source["rest_local_pos"][perm].astype(np.float64)
    rest_local_src = matrices(source["rest_local_rot"][perm])
    R = local_to_global_rotations(parents, rest_local_src[None])[0]
    P = fk_from_global_rotations(parents, offsets[0:1], R[None], offsets)[0]
    try:
        face_names = [face[k]["raw"] for k in ("r_hip", "l_hip")]
        r, l = [list(names).index(n) for n in face_names]
    except (KeyError, ValueError, TypeError) as exc:
        raise Reject("face_pair_not_in_source_skeleton") from exc
    axis = P[r] - P[l]
    forward = axis if face.get("body_axis", False) else np.cross([0., 1., 0.], axis)
    forward[1] = 0.
    if np.linalg.norm(forward) <= 1e-8 * max(np.linalg.norm(np.ptp(P, axis=0)), 1e-12):
        raise Reject("degenerate_rest_forward")
    theta = -np.arctan2(forward[0], forward[2])
    C = Rotation.from_euler("y", theta).as_matrix()
    origin = np.array([P[0, 0], P[:, 1].min(), P[0, 2]])
    Pc = (P - origin) @ C.T
    Rc = C @ R
    s = float(np.linalg.norm(np.ptp(Pc, axis=0)))
    if not np.isfinite(s) or s <= 1e-12:
        raise Reject("degenerate_rest_scale")
    # Root local rotation follows the world yaw; root offset follows our zero-offset convention.
    local, canonical_offsets = derive_rest_local_arrays(parents, Pc, Rc)
    if not np.allclose(canonical_offsets[1:], offsets[1:], atol=1e-9*s, rtol=1e-8):
        raise Reject("rest_offset_roundtrip_failed")
    if not np.allclose(local[1:], rest_local_src[1:], atol=1e-8, rtol=0):
        raise Reject("rest_rotation_roundtrip_failed")
    leaves = np.array([j for j in range(len(names)) if j not in set(parents)], np.int32)
    # Keep UNIQUE raw names; clean anatomical names often repeat and are descriptions, not IDs.
    descriptions = np.array([f"{clean_mapping[n]} joint." for n in names])
    skeleton = dict(joint_names=names, joint_descriptions=descriptions, parents=parents,
                    P_rest_global=Pc, R_rest_global=Rc, R_rest_local=local,
                    offset_parent_local=canonical_offsets,
                    rotation_source_kind=np.array(["animated_dof"] * len(names)),
                    contact_joint_indices=leaves, face_joint_names=np.array(face_names),
                    joint_order_sha256=np.array(hashlib.sha256("|".join(names).encode()).hexdigest()),
                    s_rig=np.array(s))
    return skeleton, C, origin, perm


def fk_positions(local_pos, local_rot, parents):
    """world[j] = world[p] + R_world[p] @ local_pos[j];  R_world[j] = R_world[p] @ local_rot[j].

    local_pos [T,J,3] 或 [J,3]（固定则广播），local_rot [T,J,3,3]，parents [J]（根 = -1）。
    按拓扑序推进而不假设 j > parents[j]：本模块的调用点传的是 `skeleton["parents"]`（bfs_order 产出，
    父必在子前，这里是死泛化），但 scripts/_diag_fk_approx_rejected.py 直接 import 本函数去算**源序** parents，
    那里没有这个保证。两处必须逐位同量，所以共用这一份（复审 2026-09-20：两份 FK 将来必然悄悄分叉）。
    """
    T, J = local_rot.shape[0], local_rot.shape[1]
    if local_pos.ndim == 2:
        local_pos = np.broadcast_to(local_pos[None], (T, J, 3))
    order, seen, pending = [], set(), list(range(J))
    while pending:
        nxt = [j for j in pending if parents[j] == -1 or parents[j] in seen]
        if not nxt:
            raise Reject("cycle_in_parents")
        order += nxt; seen.update(nxt)
        pending = [j for j in pending if j not in seen]
    wp = np.zeros((T, J, 3), np.float64)
    wr = np.zeros((T, J, 3, 3), np.float64)
    for j in order:
        q = int(parents[j])
        if q == -1:
            wp[:, j], wr[:, j] = local_pos[:, j], local_rot[:, j]
        else:
            wp[:, j] = wp[:, q] + np.einsum("tab,tb->ta", wr[:, q], local_pos[:, j])
            wr[:, j] = wr[:, q] @ local_rot[:, j]
    return wp


def encode(source, skeleton, C, origin, carrier=0,
           rigidify_max=RIGIDIFY_MAX_DEFAULT, rigidify_joint_cap=RIGIDIFY_JOINT_CAP_DEFAULT):
    names = list(source["names"].astype(str))
    order = np.array([names.index(n) for n in skeleton["joint_names"]])
    parent_names = [None if p == -1 else names[p] for p in source["parents"]]
    want = [None if p == -1 else str(skeleton["joint_names"][p]) for p in skeleton["parents"]]
    if [parent_names[j] for j in order] != want:
        raise Reject("clip_parent_tree_differs_from_rig")
    rest_pos = source["rest_local_pos"][order].astype(np.float64)
    local_pos = source["anim_local_pos"][:, order].astype(np.float64)
    s = float(skeleton["s_rig"])
    rigid = float(np.linalg.norm(local_pos[:, 1:] - rest_pos[None, 1:], axis=-1).max()/s)
    local = matrices(source["anim_local_rot"][:, order])
    # 数据清理（user 2026-09-20，看过四档 0.020/0.099/0.249/0.500 骨长的真值-vs-FK 并排渲染后定：
    # "我觉得差别不大，我们用fk后的数据替代pos那边的不正确数据，fk的数据肯定是正确且合理的，就说做了数据清理"）。
    #
    # 本函数**从来**没把源的逐帧非根偏移写进语料：位置一律由 `local`（旋转）+ skeleton 的固定
    # offset_parent_local 经 encode_ktjd17_channels 前向运动学算出，local_pos 只被用来取根轨迹
    # （下一行的 roots）与这里的判据。所以旧 `rigid > 1e-3` 不是在防"目标自相矛盾"（那由下方
    # numeric_codec_consistency 的 gap < 1e-6 钉死，结构上不可能发生），它只是一道**对原作保真度**的闸门。
    #
    # 换判据的理由：rigid 是"最大单关节偏移漂移/骨架尺度"的代理量，而真正要判的是这个漂移沿骨链
    # 累积之后的**位置误差**。实测（scripts/_diag_fk_approx_rejected.py，n=400 分层抽样）二者相关很弱：
    # rigid 1e-2~1e-1 档里 75.1% 的真实误差其实 < 0.25 骨长、38.4% < 0.10。
    # 放行量（审查按**全部 1,927 条被拒 clip** 复核，非抽样外推）：纯 0.5 均值门槛收回 1,767 条 = 91.7%；
    # 叠加单关节上限 2.0 后收回 1,729 条 = 89.7%。注意单位：被拒的 1,927 是 **clip**（分布在 1,237 个 rig 上），
    # v2 相对 v1 新增约 1,123 个全新 rig。0.5 是 user 逐条看过并判"差别不大"的最松那一档；再松未经人眼审。
    # 退化 rig（尺度趋零）的误差会去到 1e6 量级，仍被拒。
    fk_true = fk_positions(local_pos, local, skeleton["parents"])
    _rest_fixed = np.broadcast_to(rest_pos[None], local_pos.shape).copy()
    _rest_fixed[:, 0] = local_pos[:, 0]                       # 根平移保留（根有 smooth_root_xz 通道，合法）
    fk_rigidified = fk_positions(_rest_fixed, local, skeleton["parents"])
    _bone = float(np.linalg.norm(rest_pos[1:], axis=-1).mean()) if len(rest_pos) > 1 else 1.0
    if not np.isfinite(_bone) or _bone <= 0:
        raise Reject("degenerate_bone_scale")
    _e = np.linalg.norm(fk_true - fk_rigidified, axis=-1) / _bone
    rigidify_err = float(_e.mean(axis=1).max())               # 每帧关节均值，取时间最大 —— 与 user 看过的口径一致
    rigidify_joint_max = float(_e.max())                      # 单关节单帧最差 —— 均值口径稀释掉的那部分
    if rigidify_err > rigidify_max:
        raise Reject(f"rigidify_error:{rigidify_err:.9g}")
    if rigidify_joint_max > rigidify_joint_cap:
        raise Reject(f"rigidify_joint_error:{rigidify_joint_max:.9g}")
    roots = (local_pos[:, 0] - origin) @ C.T
    local[:, 0] = C @ local[:, 0]
    fps = float(source["fps"])
    sampled = resample_root_and_local_rotations(roots, local, fps_src=fps, fps_target=30.)
    params = dict(parents=skeleton["parents"], offset_parent_local=skeleton["offset_parent_local"],
                  R_rest_global=skeleton["R_rest_global"], s_rig=s, fps_target=30.,
                  smoother=SmootherConfig(), contact_tau_h=0.05, contact_tau_v=0.25,
                  heading_carrier_joint=carrier,
                  u_forward_local=skeleton["R_rest_global"][carrier].T @ np.array([0., 0., 1.]),
                  heading_eps_h=0.05)
    enc = encode_ktjd17_channels(root_positions=sampled.root_positions,
                               local_rotations=sampled.local_rotations, **params)
    decode_args = {k: skeleton[k] for k in ("parents", "R_rest_global", "R_rest_local",
                                          "offset_parent_local", "rotation_source_kind")}
    dec = decode_ktjd17(enc.motion, **decode_args)
    gap = float(np.linalg.norm(dec.positions_direct_minus_fk, axis=-1).max()/s)
    p = skeleton["parents"]
    bone = np.linalg.norm(skeleton["offset_parent_local"][1:], axis=-1)
    bone_err = float(np.abs(np.linalg.norm(dec.positions_direct[:, 1:] - dec.positions_direct[:, p[1:]], axis=-1)-bone).max()/s)
    vel_err = float(np.abs(world_velocity(dec.positions_direct, fps=30.)-enc.motion[..., 9:12]).max()/(s*30.))
    if max(gap, bone_err, vel_err) > 1e-6:
        raise Reject(f"numeric_codec_consistency:{gap},{bone_err},{vel_err}")
    if np.any(enc.motion[~enc.heading_valid, 0, 15:17] != 0):
        raise Reject("invalid_heading_not_zero")
    # Decode/re-encode is floating-point tolerant, not bitwise (6D Gram-Schmidt and filtering).
    roots2 = dec.positions_direct[:, 0].copy()
    roots2[:, [0, 2]] += enc.origin_xz
    roots2[:, 1] -= enc.ground_shift_y
    again = encode_ktjd17_channels(root_positions=roots2, local_rotations=dec.local_rotations, **params)
    scale = np.ones(17); scale[:3] = s; scale[9:12] = s*30.; scale[13:15] = s
    rt = float(np.max(np.abs(again.motion - enc.motion)/scale))
    if rt > 1e-6 or not np.array_equal(again.heading_valid, enc.heading_valid):
        raise Reject(f"roundtrip_error:{rt}")
    f32 = enc.motion.astype(np.float32)
    dec32 = decode_ktjd17(f32.astype(np.float64), **decode_args)
    gap32 = float(np.linalg.norm(dec32.positions_direct_minus_fk, axis=-1).max()/s)
    if not np.isfinite(f32).all() or gap32 > 2e-5:
        raise Reject(f"float32_decode_error:{gap32}")
    speed = np.linalg.norm(enc.motion[..., 9:12], axis=-1)/s
    acc = np.linalg.norm(np.diff(enc.motion[..., 9:12], axis=0)*30., axis=-1)/s
    step = np.linalg.norm(np.diff(enc.positions_clip, axis=0), axis=-1)/s
    qa = dict(source_rigid_offset_error=rigid, source_rigidify_error=rigidify_err,
              source_rigidify_joint_max=rigidify_joint_max, fk_direct_max_norm=gap,
              bone_length_max_norm=bone_err, velocity_max_norm=vel_err,
              roundtrip_max_scaled=rt, float32_fk_direct_max_norm=gap32,
              heading_valid_fraction=float(enc.heading_valid.mean()),
              contact_mean=float(enc.motion[..., 12].mean()),
              speed_p50=float(np.median(speed)), speed_p99=float(np.quantile(speed, .99)),
              acceleration_p99=float(np.quantile(acc, .99)),
              acceleration_p99_over_speed_median=float(np.quantile(acc, .99)/max(np.median(speed),1e-12)),
              max_step_rig_units=float(step.max()), ground_shift_y=enc.ground_shift_y,
              source_fps=fps, source_frames=len(local_pos),
              source_motion_sha256=None)
    return f32, enc.heading_valid, enc.origin_xz, qa


def convert_rig(row, raw, out, faces, clean_names, raw_names,
                rigidify_max=RIGIDIFY_MAX_DEFAULT, rigidify_joint_cap=RIGIDIFY_JOINT_CAP_DEFAULT):
    official = row["object_type"]
    rig = "OBJ_" + official
    accepted, rejected = [], []
    sk = C = origin = reference = None
    carrier, carrier_reason = 0, "root_default"
    reference_path = None
    clean_mapping = dict(zip(raw_names[official], clean_names[official], strict=True))
    for clip in row["clips"]:
        cid = hashlib.sha256(("uniml3d/objaverse/"+clip["clip"]).encode()).hexdigest()[:20]
        try:
            source = read_source(raw/clip["source_npz"])
            if sk is None:
                sk, C, origin, perm = make_skeleton(source, faces[official], clean_mapping)
                reference = source
                reference_path=clip["source_npz"]
                def rotations_only():
                    for other in row["clips"]:
                        with np.load(raw/other["source_npz"], allow_pickle=False) as z:
                            yield {k:np.array(z[k]) for k in ("names","parents","anim_local_rot")}
                carrier,carrier_reason=select_heading_carrier(sk,rotations_only(),C)
            if set(source["names"]) != set(reference["names"]):
                raise Reject("clip_joint_set_differs_from_rig")
            order = [list(source["names"]).index(n) for n in reference["names"]]
            if not np.allclose(source["rest_local_pos"][order], reference["rest_local_pos"],
                               rtol=1e-8, atol=float(sk["s_rig"])*1e-7):
                raise Reject("clip_rest_position_differs_from_rig")
            if not np.allclose(matrices(source["rest_local_rot"][order]), matrices(reference["rest_local_rot"]),
                               rtol=0, atol=1e-6):
                raise Reject("clip_rest_rotation_differs_from_rig")
            motion, hv, oxz, qa = encode(source, sk, C, origin, carrier,
                                         rigidify_max=rigidify_max, rigidify_joint_cap=rigidify_joint_cap)
            qa["source_motion_sha256"] = sha(raw/clip["source_npz"])
            np.savez_compressed(out/"motions"/(cid+".npz"), motion=motion, heading_valid=hv,
                                origin_xz=oxz, clip_id=np.array(cid), rig_id=np.array(rig),
                                fps_target=np.array(30., np.float64))
            captions = [clip["caption"]] if clip["caption"].strip() else []
            accepted.append(dict(clip_id=cid, rig_id=rig, motion_relpath=f"motions/{cid}.npz",
                skeleton_relpath=f"skeletons/{rig}.npz", split="train", status="accept", J_phys=None,
                T_target=len(motion), fps_target=30., official_id=clip["clip"],
                source_action_name=str(source["action_name"]), captions=captions,
                topology_family="objaverse", body_plan=row["category"],
                topology_distance_bucket="train_seen_topology", source_npz=clip["source_npz"], qa=qa))
        except (Reject, Ktjd17CodecError) as exc:
            rejected.append(dict(official_id=clip["clip"], rig_id=rig, reason=str(exc), body_plan=row["category"]))
    if accepted:
        # Stable rig-local split; singleton rigs necessarily have no val clip.
        order = sorted(accepted, key=lambda r: r["clip_id"])
        nval = min(len(order)-1, max(1, round(.05*len(order))))
        for entry in order[:nval]:
            entry["split"] = "val"
        np.savez_compressed(out/"skeletons"/(rig+".npz"), **sk)
        json_write(out/"source_metadata"/(rig+".json"), dict(
            official_object_id=official, body_plan=row["category"], source_to_canonical_yaw=C.tolist(),
            source_origin=origin.tolist(), heading_carrier_joint=carrier,
            heading_carrier_reason=carrier_reason,
            u_forward_local=(sk["R_rest_global"][carrier].T @ [0.,0.,1.]).tolist(),
            raw_joint_names=list(map(str,sk["joint_names"])),
            clean_joint_names=[clean_mapping[n] for n in sk["joint_names"]],
            joint_descriptions_status="upstream_clean_name_templates_pending_semantic_sidecars",
            license=row["license"], asset_url=row["asset_url"],
            rest_reference="rest arrays embedded in first clip; checked across all clips",
            first_source_npz=reference_path,
            annotations_face_pair=faces[official]))
    return accepted, rejected


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--raw-root", type=Path, default=Path("dataset/uniml3d"))
    p.add_argument("--output", type=Path, default=Path("dataset/ktjd17_uniml3d_v1"))
    p.add_argument("--reference-corpus", type=Path, default=Path("dataset/ktjd17_pzh312_noik_v2"))
    p.add_argument("--candidate-limit", type=int, default=0)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--rigidify-max", type=float, default=RIGIDIFY_MAX_DEFAULT,
                   help="数据清理门槛：FK-刚化相对源动画的位置误差（每帧关节均值取时间最大）上限，单位=平均骨长")
    p.add_argument("--rigidify-joint-cap", type=float, default=RIGIDIFY_JOINT_CAP_DEFAULT,
                   help="同上，但作用在**单关节单帧**最差误差上；防止均值口径稀释局部大错")
    p.add_argument("--wait-download-seconds", type=int, default=0,
                   help="Allow concurrent source download; bound each rig's wait (0: require all files now)")
    a = p.parse_args()
    converter_sha256=sha(Path(__file__))
    if a.output.exists():
        raise FileExistsError(f"Use a fresh output directory: {a.output}")
    data = json.loads((a.raw_root/"selection.json").read_text())
    rows = data["selected"][:a.candidate_limit or None]
    missing = [c["source_npz"] for r in rows for c in r["clips"] if not source_ready(a.raw_root,c)]
    if missing and not a.wait_download_seconds:
        raise FileNotFoundError(f"Download is incomplete: {len(missing)} files, e.g. {missing[:3]}")
    if a.wait_download_seconds:
        rows.sort(key=lambda r:(not all(source_ready(a.raw_root,c) for c in r["clips"]),
                                min(c["source_npz"] for c in r["clips"])))
    for name in ("motions", "skeletons", "manifests", "stats", "source_metadata", "analysis"):
        (a.output/name).mkdir(parents=True, exist_ok=True)
    for name in ("schema.json", "stats/train_block_gains.npz"):
        shutil.copyfile(a.reference_corpus/name, a.output/name)
    def read_ann(name):
        return json.loads((a.raw_root/"export/objaverse"/(name+".json")).read_text())
    faces, clean, names = [read_ann(n) for n in ("face_joint_names", "clean_joint_names", "joint_names")]
    accepted, rejected = [], []
    def work(row):
        deadline=time.monotonic()+a.wait_download_seconds
        while any(not source_ready(a.raw_root,c) for c in row["clips"]):
            if time.monotonic()>=deadline:
                raise FileNotFoundError(f"Source download wait expired for {row['object_type']}")
            time.sleep(2.)
        return convert_rig(row,a.raw_root,a.output,faces,clean,names,
                           rigidify_max=a.rigidify_max, rigidify_joint_cap=a.rigidify_joint_cap)
    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        results = pool.map(work, rows)
        for i, (good, bad) in enumerate(results,1):
            accepted.extend(good); rejected.extend(bad)
            if i % 20 == 0 or i == len(rows):
                print(f"rigs {i}/{len(rows)} accepted_clips={len(accepted)} rejected_clips={len(rejected)}",flush=True)
    accepted.sort(key=lambda x: (x["rig_id"], x["clip_id"]))
    for name, records in (("clips",accepted),("rejections",rejected)):
        with (a.output/"manifests"/(name+".jsonl")).open("w") as f:
            for row in records:
                f.write(json.dumps(row,allow_nan=False)+"\n")
    now = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    gen = now + "-" + sha(a.output/"manifests/clips.jsonl")[:12]
    json_write(a.output/"generation.json",dict(generation_id=gen,
        status="converted_numeric_checked_pending_external_visual_and_training_sidecars",
        freeze_binding={"generation_id":"20260819T192429040697Z-fe820492caaa",
                        "note":"block gains and schema carried unchanged"},
        coordinate_contract={"handedness":"right", "up":"+Y", "rest_forward":"+Z",
                             "ground":"XZ", "physical_units":"source Blender units; no scaling"},
        rep_norm="rest", clip_count=len(accepted), rig_count=len({x["rig_id"] for x in accepted}),
        # 审查 2026-09-20：实际生效的清理门槛必须可追溯 —— converter_sha256 钉不住命令行覆盖的值。
        cleaning={"rule":"rigidify: 非根关节的逐帧偏移钉回 rest，位置由旋转经 FK 重算（根平移保留）",
                  "rigidify_max_bone_lengths":a.rigidify_max,
                  "rigidify_joint_cap_bone_lengths":a.rigidify_joint_cap,
                  "metric":"‖FK(逐帧偏移)-FK(rest偏移)‖/平均骨长；max 口径为 mean-over-joints 的时间最大，cap 口径为单关节单帧最大",
                  "qa_fields":["source_rigidify_error","source_rigidify_joint_max","source_rigid_offset_error"],
                  "authority":"user 2026-09-20 看过 0.020/0.099/0.249/0.500 四档真值-vs-FK 并排渲染后定 0.5；cap 由审查加"},
        sources={"repo_id":data["repo_id"],"revision":data["revision"],"candidate_rigs":len(rows),
                 "download_root":str(a.raw_root), "rejection_counts":dict(Counter(x["reason"].split(":")[0] for x in rejected)),
                 "selection":"selection.json in source download; single tree and rigidity checked in conversion",
                 "per_asset_provenance":"source_metadata/<rig_id>.json; licenses absent from export indexes are marked unknown"},
        schema_sha256=sha(a.output/"schema.json"), gains_sha256=sha(a.output/"stats/train_block_gains.npz"),
        converter_sha256=converter_sha256,
        implementation_notes=["Root rest-local rotation includes canonical yaw; root offset is zero.",
                              "Tpose directory has PNG only; motion-embedded real rest is authoritative.",
                              "Round-trip uses float64 tolerance, not bitwise identity.",
                              "Joint descriptions are provisional anatomical templates for data handoff."]))
    print(f"DONE {a.output} generation={gen}",flush=True)


if __name__ == "__main__":
    main()
