"""被 animated_nonroot_translation 剔除的 rig：如果强行用 FK 近似，误差有多大。

user 2026-09-20 "有可能并不严重我们用fk算一下就可以呢？"。

问题的形状：KTJD-17 逐帧通道里没有非根平移这个自由度（schema channel_slices 只有
q_position/global_rest_delta_6d/world_velocity/contact/smooth_root_xz/heading），骨头偏移
offset_parent_local 是整架一份、不随帧变。源动画里非根关节偏移会漂的 rig 因此被
scripts/_build_uniml3d_ktjd17.py 的 encode() 拒掉（旧判据 animated_nonroot_translation，
2026-09-20 起换成 rigidify_error / rigidify_joint_error 两道，判在刚化后的真实位置误差上）。本脚本量的是"拒掉是否必要"：

  真值位置  = FK(逐帧 anim_local_pos, 逐帧 anim_local_rot)
  FK 近似   = FK(固定 rest_local_pos, 逐帧 anim_local_rot)     <- KTJD-17 唯一表达得了的
  误差      = 两者逐关节逐帧的位置差 / 平均骨长                <- 与 trainer 的 fkdist 同单位

判据（user 的直觉锐化成一个数）：我们训好的模型自己的 fkdist 就是 0.25 骨长。
若 FK 近似误差远小于 0.25，则我们是在为一个比自身误差还小的量丢掉三成语料。

自检：同一套 harness 跑在**被接受**的 rig 上，误差必须 ~0（它们 rigid < 1e-3 按定义）。
harness 若在已知近完美的案例上不给出近完美结果，它的数字一律不可信。

只读；不写语料、不碰训练。用法：
  python scripts/_diag_fk_approx_rejected.py --n_rejected 240 --n_accepted 60 --out <json>
"""
import argparse, json, sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _build_uniml3d_ktjd17 import read_source, matrices, Reject, fk_positions as fk   # noqa: E402
# 全部取转换器的原件，一份都不另写 —— 包括 FK：闸门与本脚本必须逐位同量，抄一份将来必然分叉（复审 2026-09-20）。


def measure(path):
    src = read_source(path)
    parents = src["parents"].astype(int)
    rest_pos = src["rest_local_pos"].astype(np.float64)          # [J,3]
    anim_pos = src["anim_local_pos"].astype(np.float64)          # [T,J,3]
    rot = matrices(src["anim_local_rot"].astype(np.float64))     # [T,J,3,3]
    J = len(parents)
    bone = float(np.linalg.norm(rest_pos[1:], axis=-1).mean()) if J > 1 else 1.0
    if not np.isfinite(bone) or bone <= 0:
        raise Reject("degenerate_bone_scale")
    true_w = fk(anim_pos, rot, parents)
    # FK 近似：根的平移保留（根平移是合法的、有 smooth_root_xz 通道），只把**非根**偏移钉回 rest
    approx_pos = np.broadcast_to(rest_pos[None], anim_pos.shape).copy()
    approx_pos[:, 0] = anim_pos[:, 0]
    approx_w = fk(approx_pos, rot, parents)
    err = np.linalg.norm(true_w - approx_w, axis=-1) / bone      # [T,J]，单位=平均骨长
    return dict(J=J, T=int(rot.shape[0]), bone=bone,
                err_mean=float(err.mean()), err_p95=float(np.percentile(err, 95)),
                err_max=float(err.max()),
                # 每帧所有关节的均值，再取时间上的最大 —— 最像 fkdist 的口径
                err_frame_mean_max=float(err.mean(axis=1).max()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="dataset/uniml3d")
    ap.add_argument("--ktjd", default="dataset/ktjd17_uniml3d_v1")
    ap.add_argument("--n_rejected", type=int, default=240)
    ap.add_argument("--n_accepted", type=int, default=60, help="自检样本：误差必须 ~0")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="runs/_heldout/fk_approx_rejected.json")
    ap.add_argument("--render_to", default="", help="非空则额外渲染：每档取一条，真值 vs FK 近似并排 gif")
    ap.add_argument("--render_levels", default="",
                    help="取最接近这些值的样本各渲一条；留空则按 --render_key 自动选档位 "
                         "（均值口径 0.02/0.10/0.25/0.50，单关节口径 0.5/1.0/1.5/2.0=cap 边界）。"
                         "复审 2026-09-20：按 err_max 挑却用均值档位，会挑到最温和的一端，正好不是要看的边界")
    ap.add_argument("--render_key", default="err_frame_mean_max",
                    choices=("err_frame_mean_max", "err_max"),
                    help="按哪个量挑样。审查 2026-09-20：按均值量挑必然挑到该档的典型条目，"
                         "挑不到被稀释的那条；要看闸门真正放行的边界就得按 err_max（单关节单帧最差）挑")
    ap.add_argument("--max_frames", type=int, default=48)
    a = ap.parse_args()
    raw, ktjd = Path(a.root), Path(a.ktjd)

    rej = [json.loads(l) for l in open(ktjd / "manifests" / "rejections.jsonl")]
    # 旧语料的 reason 是 animated_nonroot_translation:<rigid>，新语料是 rigidify_error / rigidify_joint_error:<误差>。
    # 只认旧名会在 v2 上筛出 0 条然后静默打印"无数据"并跳过渲染（复审 2026-09-20）—— 三种都认，认不出就 fail loud。
    _PREFIX = ("animated_nonroot_translation:", "rigidify_error:", "rigidify_joint_error:")
    rej = [r for r in rej if str(r.get("reason", "")).startswith(_PREFIX)]
    if not rej:
        raise SystemExit(f"[refuse] {ktjd}/manifests/rejections.jsonl 里没有任何刚化类拒绝记录；"
                         f"本脚本无事可做（是不是指错了语料？）")
    for r in rej:
        r["rigid"] = float(r["reason"].split(":")[1])
    acc = [json.loads(l) for l in open(ktjd / "manifests" / "clips.jsonl")]

    # official_id -> 源 npz。转换器用的是 clip["source_npz"]（相对 raw），accepted 清单里带着它；
    # 被拒的记录只有 official_id，所以按 official_id 在 export_flat 下定位，找不到就跳过并计数。
    def resolve(official_id):
        for sub in ("export_flat/objaverse/motions", "export/objaverse/motions"):
            p = raw / sub / f"{official_id}.npz"
            if p.exists():
                return p
            hits = list((raw / sub).glob(f"**/{official_id}.npz"))
            if hits:
                return hits[0]
        return None

    rng = np.random.default_rng(a.seed)

    def run(items, label, key, n):
        # 分层抽样：按 rigid 排序后等距取，覆盖整个量级范围而不是挤在中位
        items = sorted(items, key=key)
        idx = np.linspace(0, len(items) - 1, min(n, len(items))).astype(int)
        out, missing, failed = [], 0, 0
        for i in idx:
            it = items[int(i)]
            oid = it.get("official_id") or it.get("clip_id")
            p = resolve(oid)
            if p is None:
                missing += 1
                continue
            try:
                m = measure(p)
            except Exception as e:                       # 源缺陷是已知的，记下来不吞掉
                failed += 1
                print(f"  [{label}] {oid}: {type(e).__name__}: {e}", flush=True)
                continue
            m.update(official_id=oid, rigid=float(key(it)))
            out.append(m)
        print(f"[{label}] 量到 {len(out)} 条（源缺失 {missing}，读取失败 {failed}）", flush=True)
        return out

    res_acc = run(acc, "自检-被接受", lambda r: 0.0, a.n_accepted) if a.n_accepted else []
    res_rej = run(rej, "被剔除", lambda r: r["rigid"], a.n_rejected)

    def summarize(rows, name):
        if not rows:
            print(f"{name}: 无数据"); return {}
        for k in ("err_mean", "err_frame_mean_max", "err_max"):
            v = np.array([r[k] for r in rows])
            print(f"  {name} {k:20s} 中位 {np.median(v):.4f}  p90 {np.percentile(v,90):.4f}  最大 {v.max():.4f}  (骨长)")
        return {k: dict(median=float(np.median([r[k] for r in rows])),
                        p90=float(np.percentile([r[k] for r in rows], 90)),
                        max=float(max(r[k] for r in rows))) for k in ("err_mean", "err_frame_mean_max", "err_max")}

    print("\n=== 自检：被接受的 rig，FK 近似误差必须 ~0 ===")
    s_acc = summarize(res_acc, "接受")
    print("\n=== 被剔除的 rig ===")
    s_rej = summarize(res_rej, "剔除")

    if res_rej:
        print("\n=== 按 rigid 量级分档（err_frame_mean_max，与 fkdist 同口径）===")
        for lo, hi, lab in [(0, 1e-2, "rigid 1e-3~1e-2"), (1e-2, 1e-1, "rigid 1e-2~1e-1"), (1e-1, 1e99, "rigid >1e-1")]:
            g = [r["err_frame_mean_max"] for r in res_rej if lo <= r["rigid"] < hi]
            if g:
                print(f"  {lab:18s} n={len(g):3d}  中位 {np.median(g):.4f}  p90 {np.percentile(g,90):.4f}  最大 {max(g):.4f}")

    if a.render_to and res_rej:
        # 绘图用已有的共享渲染器（scripts/_pil_skeleton_render.py，纯几何/绘图），不另写一个
        import _pil_skeleton_render as R
        outdir = Path(a.render_to); outdir.mkdir(parents=True, exist_ok=True)
        _lv = a.render_levels or ("0.5,1.0,1.5,2.0" if a.render_key == "err_max" else "0.02,0.10,0.25,0.50")
        want = [float(x) for x in _lv.split(",") if x.strip()]
        picked = []
        for w in want:
            c = min(res_rej, key=lambda r: abs(r[a.render_key] - w))
            if c["official_id"] not in [p_["official_id"] for p_ in picked]:
                picked.append(c)
        print(f"\n=== 渲染 {len(picked)} 条（真值 | FK 近似）-> {outdir} ===", flush=True)
        for r in picked:
            src = read_source(resolve(r["official_id"]))
            parents = src["parents"].astype(int)
            rest_pos = src["rest_local_pos"].astype(np.float64)
            anim_pos = src["anim_local_pos"].astype(np.float64)
            rot = matrices(src["anim_local_rot"].astype(np.float64))
            true_w = fk(anim_pos, rot, parents)
            ap_pos = np.broadcast_to(rest_pos[None], anim_pos.shape).copy()
            ap_pos[:, 0] = anim_pos[:, 0]
            approx_w = fk(ap_pos, rot, parents)
            # 窗口必须**覆盖判据所指的位置**，且必须是原生帧率的连续窗口：
            # 抽帧到 48 帧虽然保住总时长，长 clip 会掉到 3~4 fps，画面卡顿 → 看不出形变与时序质量，
            # 而这正是要人眼判的东西（feedback_qa_window_must_cover_the_hypothesis / gif_realtime_length_fps）。
            _bone_r = float(np.linalg.norm(rest_pos[1:], axis=-1).mean())
            _err_t = (np.linalg.norm(true_w - approx_w, axis=-1) / _bone_r).max(axis=1)   # 每帧最差关节
            _c = int(_err_t.argmax())                                   # 误差峰值帧
            _half = a.max_frames // 2
            _lo = max(0, min(_c - _half, len(true_w) - a.max_frames))
            idx = list(range(_lo, min(_lo + a.max_frames, len(true_w))))
            tw, aw = true_w[idx], approx_w[idx]
            cell = (900, 760)
            tr = R.compute_transform([tw, aw], cell, 0.12, 1.0)
            hdr = (f'{r["official_id"]}  J={r["J"]}  '
                   f'均值口径 {r["err_frame_mean_max"]:.3f} / 单关节最差 {r["err_max"]:.3f} 骨长'
                   f'  ·  第 {idx[0]}-{idx[-1]} 帧（共 {len(true_w)}，以误差峰值帧 {_c} 为中心）@ {float(src["fps"]):.0f} fps')
            frames = [R.make_row_frame(
                [dict(positions=tw, parents=parents, title="真值（逐帧偏移）", color=(20, 20, 20), axes=True),
                 dict(positions=aw, parents=parents, title="FK 近似（偏移钉回 rest）", color=(181, 85, 47), axes=True)],
                fi, tr, cell, lw=5, jr=6, header=hdr, header_h=48) for fi in range(len(idx))]
            fps = float(src["fps"])                                    # 连续窗口 → 原生帧率即真·原速
            name = f'{a.render_key}{r[a.render_key]:.3f}_{r["official_id"][:40]}.gif'
            R.save_gif(frames, outdir / name, max(fps, 1.0))
            print(f'  {name}  ({len(idx)}/{len(true_w)} 帧 @ {fps:.1f} fps)', flush=True)

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(dict(accepted=res_acc, rejected=res_rej, summary_accepted=s_acc, summary_rejected=s_rej),
              open(a.out, "w"), indent=1)
    print(f"\n写入 {a.out}")


if __name__ == "__main__":
    main()
