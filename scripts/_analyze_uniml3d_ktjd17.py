#!/usr/bin/env python3
"""Population statistics, topology census and existing-pipeline motion previews for UniML3D."""
import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts._build_uniml3d_ktjd17 import json_write, sha
from src.data.ktjd17.codec import direct_decode_positions
from src.data.ktjd17.species_stats import CHANNEL_NAMES, ChannelMoments, RigCellMoments


def tree_signature(parents):
    """Unlabelled rooted-tree identity, independent of BFS sibling order and joint names."""
    children = defaultdict(list)
    for j, p in enumerate(parents):
        if p >= 0:
            children[int(p)].append(j)
    def visit(j):
        return "(" + "".join(sorted(visit(k) for k in children[j])) + ")"
    roots = np.flatnonzero(np.asarray(parents) == -1)
    return "|".join(sorted(visit(int(j)) for j in roots))


def quantiles(x):
    if not len(x):
        return {}
    return dict(zip(("min", "p25", "median", "p75", "p95", "max"),
                    map(float, np.quantile(x, [0, .25, .5, .75, .95, 1]))))


def analyze(root, reference):
    rows = [json.loads(l) for l in (root/"manifests/clips.jsonl").open()]
    byrig = defaultdict(list)
    for row in rows:
        byrig[row["rig_id"]].append(row)
    if not rows:
        raise RuntimeError("No accepted clips")
    totals = ChannelMoments()
    rest_totals = ChannelMoments()
    plans = defaultdict(ChannelMoments)
    cell_stats, census = [], []
    existing_trees = set()
    for file in (reference/"skeletons").glob("*.npz"):
        with np.load(file, allow_pickle=False) as z:
            existing_trees.add(tree_signature(z["parents"]))
    for i, (rig, clips) in enumerate(sorted(byrig.items())):
        moments = RigCellMoments(rig)
        plan = clips[0]["body_plan"]
        with np.load(root/"skeletons"/(rig+".npz"),allow_pickle=False) as sk:
            rest=np.asarray(sk["P_rest_global"],np.float64);s=float(sk["s_rig"])
        rest_frame=np.zeros((len(rest),17),np.float64)
        rest_frame[:,:3]=rest
        rest_frame[:,[0,2]]-=rest[0,[0,2]]
        rest_frame[:,3:9]=[1,0,0,0,1,0]
        rest_frame[0,15]=1.
        with np.load(root/"stats/train_block_gains.npz",allow_pickle=False) as z:
            gains=z["gains"]
        scale=np.ones((len(rest),17));scale[:,:3]=s/gains[0];scale[:,9:12]=s/gains[1];scale[0,13:15]=s/gains[2]
        rest_absmax=0.
        for row in clips:
            with np.load(root/row["motion_relpath"], allow_pickle=False) as z:
                m = z["motion"].astype(np.float64); hv = z["heading_valid"]
            moments.update(m, hv)
            normalized=(m-rest_frame)/scale
            rest_totals.update(normalized[...,:13].reshape(-1,13),slice(0,13))
            rest_totals.update(normalized[:,0,13:15],slice(13,15))
            rest_totals.update(normalized[hv,0,15:17],slice(15,17))
            rest_absmax=max(rest_absmax,float(np.abs(normalized[...,:15]).max()))
            if hv.any():rest_absmax=max(rest_absmax,float(np.abs(normalized[hv,0,15:17]).max()))
            for agg in (totals, plans[plan]):
                agg.update(m[..., :13].reshape(-1, 13), slice(0, 13))
                agg.update(m[:, 0, 13:15], slice(13, 15))
                agg.update(m[hv, 0, 15:17], slice(15, 17))
        final = moments.finalized()
        # Enforce mathematically exact constants from actual stored extrema, immune to mean roundoff.
        const = (final["count"] > 0) & (final["minimum"] == final["maximum"])
        final["std"][const] = 0.
        final["mean"][const] = final["minimum"][const]
        cell_stats.append(final)
        with np.load(root/"skeletons"/(rig+".npz"), allow_pickle=False) as sk:
            par = sk["parents"]; s = float(sk["s_rig"])
            depth = np.zeros(len(par), int)
            for j in range(1, len(par)):
                depth[j] = depth[par[j]]+1
            sig = tree_signature(par)
            leaves = len(set(range(len(par))) - set(par))
        census.append(dict(rig_id=rig, body_plan=plan, joint_count=len(par),
            clip_count=len(clips), frame_count=final["frame_count"], seconds=final["frame_count"]/30.,
            s_rig=s, max_depth=int(depth.max()), leaf_count=leaves,
            topology_signature=sig, topology_seen_in_pzh312=sig in existing_trees,
            heading_valid_fraction=final["heading_valid_frame_count"]/final["frame_count"],
            exact_constant_cells=int(const.sum()),
            rest_normalized_abs_max_before_constant_mask=rest_absmax,
            contact_mean=float(final["mean"][:,12].mean()),
            contact_all_zero=bool(np.all(final["maximum"][:,12] == 0)),
            split_counts=dict(Counter(r["split"] for r in clips))))
        if (i+1) % 200 == 0:
            print(f"stats {i+1}/{len(byrig)} rigs", flush=True)
    R, J = len(cell_stats), max(x["joint_count"] for x in cell_stats)
    payload = {}
    for key in ("count", "mean", "std", "minimum", "maximum"):
        buf = np.zeros((R, J, 17), dtype=np.int64 if key == "count" else np.float64)
        for i, st in enumerate(cell_stats):
            buf[i,:st["joint_count"]] = st[key]
        payload[key] = buf
    payload["valid_mask"] = payload["count"] > 0
    payload["rig_ids"] = np.array(sorted(byrig))
    payload["body_plans"] = np.array([x["body_plan"] for x in census])
    payload["channel_names"] = np.array(CHANNEL_NAMES)
    for key in ("joint_count", "clip_count", "frame_count", "heading_valid_frame_count"):
        payload[key] = np.array([x[key] for x in cell_stats], np.int64)
    gen = json.loads((root/"generation.json").read_text())
    payload["__generation_id"] = np.array(gen["generation_id"])
    np.savez_compressed(root/"stats/rig_stats.npz", **payload)
    # Reporting only: neither raw nor aggregated std is a training divisor under rep_norm=rest.
    plan_report = {}
    for plan, x in sorted(plans.items()):
        plan_report[plan] = dict(mean=x.mean.tolist(), std=x.population_std().tolist(), count=x.count.tolist())
    json_write(root/"stats/channel_stats.json", dict(channel_names=CHANNEL_NAMES, ddof=0,
        generation_id=gen["generation_id"], purpose="reporting; training uses skeleton-derived rest normalization",
        global_stats=dict(mean=totals.mean.tolist(), std=totals.population_std().tolist(), count=totals.count.tolist()),
        rest_normalized_before_constant_mask=dict(mean=rest_totals.mean.tolist(),std=rest_totals.population_std().tolist(),
            count=rest_totals.count.tolist(),minimum=rest_totals.minimum.tolist(),maximum=rest_totals.maximum.tolist()),
        by_body_plan=plan_report))
    json_write(root/"analysis/rig_census.json", census)
    with (root/"analysis/rig_census.csv").open("w") as f:
        writer = csv.DictWriter(f, fieldnames=list(census[0]))
        writer.writeheader(); writer.writerows(census)
    vm = payload["valid_mask"]
    low = vm & (payload["std"] < 1e-4)
    exact = vm & (payload["minimum"] == payload["maximum"])
    metrics = ("fk_direct_max_norm", "float32_fk_direct_max_norm", "bone_length_max_norm",
               "roundtrip_max_scaled", "source_rigid_offset_error", "speed_p99", "acceleration_p99",
               "acceleration_p99_over_speed_median", "max_step_rig_units", "heading_valid_fraction")
    rejections = [json.loads(l) for l in (root/"manifests/rejections.jsonl").open()]
    unique = set(x["topology_signature"] for x in census)
    summary = dict(generation_id=gen["generation_id"], accepted_rigs=R, accepted_clips=len(rows),
        frames=sum(r["T_target"] for r in rows), hours=sum(r["T_target"] for r in rows)/30/3600,
        rejected_clips=len(rejections), rejections=dict(Counter(x["reason"].split(":")[0] for x in rejections)),
        rigs_by_body_plan=dict(Counter(x["body_plan"] for x in census)),
        clips_by_body_plan=dict(Counter(x["body_plan"] for x in rows)),
        split_counts=dict(Counter(x["split"] for x in rows)),
        singleton_rigs=sum(x["clip_count"] == 1 for x in census),
        joint_count=quantiles([x["joint_count"] for x in census]),
        clip_duration_seconds=quantiles([x["T_target"]/30 for x in rows]),
        rest_scale_source_units=quantiles([x["s_rig"] for x in census]),
        rest_normalized_abs_max_before_constant_mask=quantiles([x["rest_normalized_abs_max_before_constant_mask"] for x in census]),
        unique_unlabelled_rooted_trees=len(unique), new_tree_shapes_vs_pzh312=len(unique-existing_trees),
        rigs_matching_pzh312_tree_shape=sum(x["topology_seen_in_pzh312"] for x in census),
        valid_cells=int(vm.sum()), exact_constant_cells=int(exact.sum()), low_std_cells=int(low.sum()),
        low_std_per_channel=low.sum((0,1)).tolist(), exact_constant_per_channel=exact.sum((0,1)).tolist(),
        contact_all_zero_rigs=sum(x["contact_all_zero"] for x in census),
        qa={k:quantiles([r["qa"][k] for r in rows]) for k in metrics},
        motion_payload_bytes=sum((root/r["motion_relpath"]).stat().st_size for r in rows),
        rig_stats_sha256=sha(root/"stats/rig_stats.npz"),
        interpretation=["Body plans are upstream labels, not biological species.",
                        "Different asset IDs can share a tree; new tree counts use rooted unlabeled isomorphism.",
                        "Rest scale is in arbitrary source units, not metres.",
                        "Statistics cover all accepted clips; not a zero-shot statistical fit.",
                        "Constant masks and semantic/caption embeddings remain with the training handoff owner.",
                        "Acceleration/speed ratio is descriptive and diverges near static median velocity."])
    json_write(root/"analysis/summary.json",summary)
    report=["# UniML3D → KTJD-17 数据统计", "",
        f"Generation: `{gen['generation_id']}`", "",
        f"接受 **{R:,} 个 rig / {len(rows):,} 条 clip / {summary['frames']:,} 帧 / {summary['hours']:.3f} 小时**。",
        f"转换拒绝 {len(rejections):,} 条 clip；详细原因见 `../manifests/rejections.jsonl`。", "",
        "## 类别与拓扑", "", "类别来自上游 body-plan 标注，不是生物物种分类。", "",
        "| Body plan | Rig | Clip | 小时 |", "|---|---:|---:|---:|"]
    for plan in sorted(plans):
        hours=sum(x['frame_count'] for x in census if x['body_plan']==plan)/108000
        report.append(f"| {plan} | {summary['rigs_by_body_plan'][plan]} | {summary['clips_by_body_plan'][plan]} | {hours:.3f} |")
    report += ["",f"按有根无标签树同构去重：**{len(unique):,} 种拓扑**；其中 **{len(unique-existing_trees):,} 种**未在现役 PZ/Human 骨架中出现。",
        f"单 clip rig：{summary['singleton_rigs']:,}；split 数量：`{summary['split_counts']}`。这是 rig 内划分，不是 OOD 留出协议。", "",
        "## 归一化与统计", "",
        f"有效格子 {int(vm.sum()):,}；exact-constant {int(exact.sum()):,}；std < 1e-4 共 {int(low.sum()):,}。",
        "这些 mean/std 是数据分析工件。训练按 `--rep_norm rest` 用骨架 rest 和尺度归一化，不以这些低方差 std 作除数。",
        "`rig_stats.npz` 保留逐 rig、逐 joint、逐 channel 的统计、count、mask、min/max；另有全库与 body-plan 分组统计。",
        "统计覆盖所有接受 clip，ddof=0。恒定格的监督 mask/常数恢复由侧文件阶段构建。", "",
        "## 分布与数值检查", "", "完整分位数、极值 clip 与逐 rig 明细见同目录 JSON/CSV。", "",
        "| 量 | min | median | p95 | max |", "|---|---:|---:|---:|---:|"]
    for name,key in (("关节数","joint_count"),("clip 秒数","clip_duration_seconds"),
                     ("rest 尺度（源单位）","rest_scale_source_units"),
                     ("rest 归一化绝对最大值（mask 前）","rest_normalized_abs_max_before_constant_mask")):
        q=summary[key];report.append(f"| {name} | {q['min']:.5g} | {q['median']:.5g} | {q['p95']:.5g} | {q['max']:.5g} |")
    report += ["", "通道独立验算：`channel_semantics_verification.json`；动态预览：`visuals/`；带 contact/velocity/FK 叠画：`channel_visuals/`。",
        "坐标：右手系，+Y 向上，rest 前向 +Z，XOZ 地面。Heading 是载体关节朝向，不强制等同行进方向。",
        "", "## 交接状态", "", "转换、统计和本地数值/可视化检查不等于训练侧文件已就绪。关节描述目前为上游 clean name 的模板，语义 embedding、caption embedding 和训练监督 mask 由规范 §4.8 的负责人接续。",
        "逐资产许可在提供的索引中缺失，已显式标为 unknown；数据保留本机。"]
    (root/"analysis/summary.md").write_text("\n".join(report)+"\n")
    with (root/"analysis/extreme_clips.json").open("w") as f:
        json.dump({key: sorted(rows,key=lambda x:x["qa"][key],reverse=True)[:10]
                   for key in ("speed_p99", "max_step_rig_units", "source_rigid_offset_error")},f,indent=2)
    print(json.dumps(summary, indent=2), flush=True)
    return rows, census


def render(root, rows, count):
    from scripts._render_noik_gt import render_motion, render_rest
    from PIL import Image, ImageDraw
    out = root/"analysis/visuals"; out.mkdir(exist_ok=True)
    byrig = defaultdict(list)
    for r in rows:
        byrig[r["rig_id"]].append(r)
    groups = defaultdict(list)
    for rig, rs in byrig.items():
        clip = max(rs, key=lambda r:min(r["T_target"],240))
        groups[clip["body_plan"]].append(clip)
    for rs in groups.values():
        rs.sort(key=lambda r:(abs(r["qa"]["speed_p99"]-.5),r["rig_id"]))
    chosen=[]
    while len(chosen)<count and any(groups.values()):
        for category in sorted(groups):
            if groups[category] and len(chosen)<count:
                chosen.append(groups[category].pop(0))
    for r in chosen:
        rig=r["rig_id"]
        with np.load(root/r["skeleton_relpath"],allow_pickle=False) as sk:
            par=sk["parents"]; rest=sk["P_rest_global"]
        with np.load(root/r["motion_relpath"],allow_pickle=False) as z:
            m=z["motion"][:240].astype(np.float64); hv=z["heading_valid"][:240]
        P=direct_decode_positions(m)
        render_rest(out/f"REST_{rig}.png",rest,par,rig)
        dest=out/f"MOTION_{rig}.gif"
        render_motion(dest,P,m[:,0,15:17],hv,par,rig,f"{r['body_plan']} {r['captions'][0] if r['captions'] else ''}")
        # Four time points plus rest; inspect full GIF for temporal quality as well.
        with Image.open(dest) as gif:
            tiles=[]
            for idx in np.linspace(0,gif.n_frames-1,4).astype(int):
                gif.seek(int(idx)); im=gif.convert("RGB");im.thumbnail((440,310));tiles.append(im.copy())
        sheet=Image.new("RGB",(880,660),"white")
        for i,tile in enumerate(tiles):sheet.paste(tile,((i%2)*440,(i//2)*310))
        ImageDraw.Draw(sheet).text((8,630),f"{r['body_plan']} {rig}",fill="black")
        sheet.save(out/f"SHEET_{rig}.png")
        print(f"rendered {rig} {len(m)} frames",flush=True)
    json_write(out/"selection.json",chosen)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root",type=Path,default=Path("dataset/ktjd17_uniml3d_v1"))
    p.add_argument("--reference",type=Path,default=Path("dataset/ktjd17_pzh312_noik_v2"))
    p.add_argument("--render-rigs",type=int,default=0)
    a=p.parse_args()
    rows,_=analyze(a.root,a.reference)
    if a.render_rigs:render(a.root,rows,a.render_rigs)


if __name__ == "__main__":main()
