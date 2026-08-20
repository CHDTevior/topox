# KTJD-17 Truebones 数据集使用说明

这份数据是当前 BTJD-13 Truebones 目录的 KTJD-17 转换结果。转换严格使用原始 BVH 声明的旋转通道；旧 BTJD-13 motion 只用于建立库存对应关系，未参与解码、IK 或旋转补造。

## 当前产物

- 稳定入口：`dataset/ktjd17_truebones`
- 固定 generation：`dataset/.ktjd17_truebones_generations/20260819T215405576671Z-2d04a8d85638`
- 大小：约 226 MiB
- motion：986 条，66 个可编码 rig，0 条新增转换失败
- skeleton：66 个，每个 rig 一份 canonical rest skeleton
- 帧率：30 FPS
- 最大形状：`T=237`、`J_phys=142`；文件本身不预先 pad
- 坐标系：右手系，`Y+` 向上，`Z+` 朝屏幕外/朝观察者
- TopoX 规格：commit `9181f5cccbad23e941bf94c2874daf36e7f288cf`

1070 条父库存中有 84 条保持上游拒绝：67 条 `SOURCE_LAYOUT_DRIFT`，17 条 `RAW_SOURCE_SEQUENCE_SPLIT_OVERLAP`。其中 Ant17、Crab10、Deer22、Jaguar14 没有任何 source-safe clip，因此没有伪造这四个 rig 的旋转。详见：

- `manifests/upstream_rejections.jsonl`
- `manifests/conversion_rejections.jsonl`（本 generation 为空）
- `manifests/unavailable_rigs.jsonl`

## 文件结构

```text
dataset/ktjd17_truebones/
├── generation.json
├── schema.json
├── motions/<clip_id>.npz
├── skeletons/<rig_id>.npz
├── manifests/clips.jsonl
├── manifests/upstream_rejections.jsonl
├── splits/holdout_splits_v1/{train,val,held_representative,held_stress}.txt
├── stats/train_block_gains.npz
└── qa/
```

每个 motion NPZ 使用 `allow_pickle=False`，包含：

- `motion`: `float32 [T,J_phys,17]`
- `heading_valid`: `bool [T]`
- `origin_xz`: `float64 [2]`
- `clip_id`, `rig_id`: Unicode scalar
- `fps_target`: `float64` scalar，固定为 30

17 个 channel：

| Channel | 含义 |
|---|---|
| `0:3` | `q_position=[Px-sx, Py, Pz-sz]` |
| `3:9` | column-cont6d 编码的 `R_global @ R_rest_global.T` |
| `9:12` | canonical 世界系关节速度，单位为 canonical length-unit/s |
| `12` | 逐关节 ground-support contact |
| `13:15` | smooth-root XZ，仅 root 行有效 |
| `15:17` | `[cos(theta), sin(theta)]`，仅 root 行有效 |

非 root 的 `13:17` 为精确零。不要把 canonical length-unit 擅自写成 metre；各 rig 的归一化尺度是 skeleton 中的 `s_rig`。

## 最小读取与双路径解码

在仓库根目录运行：

```python
from pathlib import Path
import json
import numpy as np

from src.data.ktjd17.codec import restore_origin_xz
from src.data.ktjd17.decoder import decode_ktjd17
from src.data.ktjd17.encoder import load_skeleton
from src.data.ktjd17.loader import load_motion_npz

root = Path("dataset/ktjd17_truebones").resolve()
rows = [json.loads(line) for line in (root / "manifests/clips.jsonl").read_text().splitlines()]
row = rows[0]

payload = load_motion_npz(root / row["motion_relpath"], expected_fps_target=30.0)
skeleton = load_skeleton(root / row["skeleton_relpath"])
motion64 = payload["motion"].astype(np.float64)

decoded = decode_ktjd17(
    motion64,
    parents=skeleton.parents,
    R_rest_global=skeleton.R_rest_global,
    R_rest_local=skeleton.R_rest_local,
    offset_parent_local=skeleton.offset_parent_local,
    rotation_source_kind=skeleton.rotation_source_kind,
    strict_gt=True,
)

# generation 内的 canonical clip 坐标；逐帧直接恢复，不做速度积分。
positions_clip = decoded.positions_direct
positions_fk_clip = decoded.positions_fk

# 恢复该 clip 的 canonical absolute XZ。Y 已在转换时统一 ground 到 min(Y)=0。
positions_absolute = restore_origin_xz(positions_clip, payload["origin_xz"])
positions_fk_absolute = restore_origin_xz(positions_fk_clip, payload["origin_xz"])

print(payload["clip_id"], payload["motion"].shape, skeleton.rig_id)
print("direct/FK max:", np.linalg.norm(positions_clip - positions_fk_clip, axis=-1).max())
```

不要用 `motion[...,9:12]` 积分来恢复位置；它是监督/诊断通道，不是解码依赖。直接位置恢复是：

```python
positions_clip = motion64[..., 0:3].copy()
positions_clip[..., 0] += motion64[:, 0, 13][:, None]
positions_clip[..., 2] += motion64[:, 0, 14][:, None]
```

## 构造训练输入

使用冻结的 train-only gains 和 loader 生成 padding、mask、crop 与 yaw augmentation：

```python
from src.data.ktjd17.loader import build_model_view

with np.load(root / "stats/train_block_gains.npz", allow_pickle=False) as stats:
    gains = stats["gains"].astype(np.float64)

view = build_model_view(
    payload["motion"],
    payload["heading_valid"],
    parents=skeleton.parents,
    R_rest_global=skeleton.R_rest_global,
    rotation_source_kind=skeleton.rotation_source_kind,
    s_rig=skeleton.s_rig,
    gains=gains,
    T_max=300,
    J_max=142,
    crop_start=0,
    crop_length=None,
    yaw_radians=0.0,
)

x = view.motion                         # float32 [300,142,17]
frame_mask = view.masks.frame_mask      # [300]
joint_mask = view.masks.joint_mask      # [142]
channel_mask = view.masks.channel_valid_mask  # [142,17]
heading_mask = view.masks.heading_valid
```

注意：

- `channel_valid_mask` 会排除非 root 的 `13:17`；不要让这些精确零参与 loss/statistics。
- `rotation_supervised` 只监督真实 animated DOF；fixed DOF 由 rest rotation 覆盖。
- invalid heading 的 `15:17` 是精确 `[0,0]`，必须配合 `heading_valid` mask。
- crop 只重置 smooth-root 的局部 XZ 原点，不改变 q/rotation/velocity/contact/heading 的语义。
- 当前 Graph-CodeFlow 合同是 `T_fine_max=300`、`temporal_stride=4`、`T_lat_max=75`，不要改回 64 帧/16 latent steps。

## Split

| Split | 数量 | 用途 |
|---|---:|---|
| `train` | 671 | 训练；冻结 gains 只来自既有 train calibration authority |
| `val` | 51 | seen-topology validation |
| `held_representative` | 149 | topology representative holdout |
| `held_stress` | 115 | topology stress holdout |

不要用三个非 train split 重新估计 gains、contact 阈值或 schema 参数。

## QA 与可视化

- 全量 fixed QA：`dataset/validation_reports/ktjd17_truebones_fixed_qa_20260819T215405576671Z-2d04a8d85638.json`
- 后构建视觉：`dataset/ktjd17_truebones_visual/ktjd17_visual_qa`
- 视觉等价证据：`dataset/ktjd17_truebones_visual_equivalence_20260819T215929053759Z-faf7ba07b6f5.json`
- 最终发布门：`dataset/KTJD17_TRUEBONES_RELEASE_GATE.json`

视觉目录对每个可编码 rig 提供：24 帧 GIF、6 个诊断时刻的 source/direct/FK 三行 filmstrip、canonical rest 图。198/198 个后构建图像与构建前经过人工和 gpt-5.5/xhigh 审核的图像逐字节一致。

## 重建与复验命令

```bash
PYTHONPATH=. python -B scripts/build_ktjd17_truebones.py
PYTHONPATH=. python -B scripts/validate_ktjd17_truebones.py
PYTHONPATH=. python -B scripts/render_ktjd17_truebones.py
```

构建会产生新的 immutable generation；不要覆盖现有 generation。消费者应先检查 `dataset/KTJD17_TRUEBONES_RELEASE_GATE.json` 的 `ready_to_test`，再通过稳定入口 `dataset/ktjd17_truebones` 读取。
