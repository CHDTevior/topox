# KTJD-17 v1: 面向多拓扑的 Kimodo-like TJD 动作表示

> 状态:设计提案，供 TopoX 实施 agent 使用；尚未宣称全量数据已经构建。  
> 核心接口:`motion[T,J_max,17]`。`J_max` 只包含物理关节，不包含 WORLD/control token。

## 1. Problem Anchor

- **Bottom-line problem**:为 TopoX 设计拓扑可变的 `[T,J,D]` 动作表示，同时保留
  Kimodo-like 表示的核心优点：显式全局轨迹与 heading、无需速度积分的逐帧位置恢复、
  世界系旋转和速度、稀疏控制接口，以及 position/FK 两条独立审计路径。
- **Must-solve bottleneck**:当前 legacy AnyTop13 的 root 行复用普通通道却改变语义，root XZ
  依赖速度积分，非 root 位置按逐帧 heading 转局部系，混用 FPS，并且现有旋转载体无法完整
  保留所有叶关节自身旋转。
- **Non-goals**:兼容旧 checkpoint；把所有动物重定向到一套人体 T-pose；从位置 IK 伪造
  原始旋转；在没有单位证据时声称数值单位是米；在本提案里另造 VQVAE/backbone 架构。
- **Constraints**:`T` 是时间轴；`J` 是 padding 后的最大物理 joint 数；`D` 是每个物理
  joint 节点的属性。现有 graph、parents、topology ID 和 held split 都必须继续只描述真实骨架。
- **Success condition**:任意有效帧只读该帧即可恢复全部关节位置；rotation-FK 与直接位置
  路径可独立比较；平移和全局 yaw 等变；新增拓扑只需提供 rig payload，不改变 D 和通道语义。

## 2. 一句话方案

采用 **KTJD-17**：前 13 个 channel 对全部物理关节完全同义，后 4 个 channel 仅在物理 root
行有效，用来承载 smooth-root XZ 和 heading；用显式 mask 区分结构有效性和逐帧 heading
有效性。

逻辑样本和 batch 形状分别为：

```text
single clip: [T_valid, J_phys, 17]
model view : [B, T_max, J_max, 17]
```

这里没有虚拟 WORLD 节点。模型内部可以把它无损拆成 `[B,T,4] + [B,T,J,13]` 两条流，
但数据契约和对外接口仍是严格 TJD。

## 3. 为什么不是 WORLD 节点

虚拟 WORLD 节点能让每行更整齐，但会把 `J` 从“最大物理关节数”改成“最大节点数”，还会污染
adjacency、geodesic distance、FK、topology canonical form 和 held-topology 协议。KTJD-17
选择声明式 root 特例：

- ch 0:13 在 root 与非 root 上不变义；
- ch 13:17 有固定名称，只是非 root 行结构无效；
- 非 root 的 0 由 mask 排除，不是训练真值；
- graph 和骨架接口完全不改含义。

这与 legacy13 的“同一 channel 在 root 行偷偷换语义”不是同一种特例。

## 4. 固定坐标与 rest-pose 契约

### 4.1 世界坐标

- 右手系；`Y+` 向上；`XZ` 为地面。
- canonical rest forward 为 `+Z`。
- 正 yaw 按右手规则绕 `+Y`。
- 不做逐帧 heading canonicalization。
- 不做首帧 heading canonicalization。
- 每个 rig 只允许一个固定的 source-to-canonical 变换；它对该 rig 全部 clip 相同。

### 4.2 Rest pose

不存在一套能覆盖所有拓扑的共享人体 T-pose。每个 **rig** 都有自己的 canonical rest pose：

```text
P_rest_global[J,3]
R_rest_global[J,3,3]
parents[J]
offset_parent_local[J,3]
```

`P_rest_global` 与 `R_rest_global` 必须来自同一 rest 帧。rig 进入数据集前，将其 rest 的 up
轴固定为 `+Y`、审定的 body forward 固定为 `+Z`、地面固定为 `Y=0`。这是 per-rig 常量
坐标校准，不是 per-clip 朝向归一化。

父局部 offset 定义为：

```text
offset[c] = R_rest_global[p].T
            @ (P_rest_global[c] - P_rest_global[p])
```

人体源可使用其已审核的 SMPL/SMPL-X canonical T-pose；动物、蛇、蜘蛛、翅膀拓扑使用各自
源 rig 的 rest pose。不能把动物强行套到人体 rest pose。

## 5. 17 个 channel 的严格定义

物理 root 在 FK 顺序中固定为 joint 0。布局如下：

| channel | 名称 | 数量 | 有效行 | 坐标/单位 |
|---|---:|---:|---|---|
| `0:3` | `q_position` | 3 | 全部物理 joint | 世界轴对齐、相对 smooth-root XZ |
| `3:9` | `global_rest_delta_6d` | 6 | 全部物理 joint | 全局 rest-delta，column-cont6d |
| `9:12` | `world_velocity` | 3 | 全部物理 joint | 世界系，length-unit/second |
| `12` | `contact` | 1 | 有监督的物理 joint | 二值 GT；模型输出 logit |
| `13:15` | `smooth_root_xz` | 2 | 仅物理 root | 世界 XZ 平面 |
| `15:17` | `heading` | 2 | 仅物理 root 且该帧有效 | `[cos(theta),sin(theta)]` |

### 5.1 ch 0:3:直接位置 `q_position`

令 `P_j(t)` 是 canonical 世界系关节位置，`s_xz(t)` 是平滑后的物理 root 水平轨迹：

```text
q_j(t) = [P_j.x(t)-s_x(t), P_j.y(t), P_j.z(t)-s_z(t)]
```

关键点：

- 不按 heading 旋转，始终与世界 XYZ 轴对齐；
- Y 是相对地面的绝对高度；
- root 行也使用完全相同公式，因此保存 pelvis 相对 smooth path 的 XZ 残差和 root 高度；
- 不是 parent-local position，也不是 velocity-integrated position。

逐帧直接恢复：

```text
P_j(t) = [q_j.x+s_x, q_j.y, q_j.z+s_z]
```

恢复不读取速度、不读取前一帧、不做累计和。

### 5.2 ch 3:9:全局 rest-delta 旋转

采用主动旋转、列向量、右手系：

```text
D_j(t) = R_global_j(t) @ R_rest_global_j.T
d6_j(t) = first_two_columns(D_j(t))
R_global_j(t) = decode_column_cont6d(d6_j(t)) @ R_rest_global_j
```

标准 rest pose 下所有 `D_j=I`。6D 解码只使用 Gram-Schmidt：先归一化第一列，将第二列对
第一列正交化并归一化，第三列为 `b1 cross b2`。

**Lossless 主 schema 的硬条件**：

- BVH 类源必须回到原始 BVH rotation channels；
- SMPL/MotionStreamer 类源必须回到其原始旋转通道和对应 rest skeleton；
- 禁止从 legacy AnyTop13、关节位置或 IK 生成主 schema 的旋转 GT；
- 有 rotation channel 的叶关节必须保留其自身旋转；
- source 明确定义为 fixed joint/end-site 的 identity local DOF 可以按源层级精确传播；
- 无法证明旋转来源的 clip/rig 不进入 lossless v1。若未来确需保留，另建带
  `leaf_rot_unobserved_mask` 的 `ktjd17-lossy-*`，不能与主数据混训而不声明。

### 5.3 ch 9:12:世界速度

统一按秒：

```text
v_j(t) = (P_j(t+1)-P_j(t)) * fps_target
v_j(T-1) = v_j(T-2), when T>=2
v_j(0) = 0, when T=1
```

速度是动态属性和监督信号，不参与位置解码。平滑、速度和 heading 在 full clip 上计算，之后
再 crop，避免窗口边界制造假速度。

### 5.4 ch 12:逐关节 contact

`contact[t,j]` 表示该物理关节在该帧是否与地面/环境发生静态支持接触。它不是 human-only
四足位，也不能只看 joint 名字：蛇的躯干、蜘蛛多足、翅膀端点都可能需要不同 eligibility。

- 有可信 source label 时原样保留；
- 否则在统一 FPS、ground 和 scale 后，从高度与世界速度派生；
- `contact_supervised[J]` 指定哪些 joint 可进入 BCE；
- 阈值按 `s_rig` 无量纲化，并由 prototype/full-train 直方图冻结，不能直接照搬人体脚阈值；
- 未监督槽存 0，但 mask 为 false，0 不是真值。

### 5.5 ch 13:15:smooth-root XZ

只平滑物理 root 的 XZ，不另存一份 Y：root 高度的唯一权威值是 `q_root.y`。这比固定
Kimodo273 少一个重复的 smooth-root Y。

建议使用 Kimodo ADMM 类平滑器，但先在尺度归一化空间处理：

```text
s_xz = Smooth(P_root_xz / s_rig; margin_norm) * s_rig
```

`margin_norm`、短 clip fallback 和滤波边界规则必须经 prototype 标定后写入 schema；在标定前
不得把某个人体米制阈值当成跨拓扑常量。

### 5.6 ch 15:17:heading

每个 rig payload 指定：

```text
heading_carrier_joint
u_forward_local[3]
```

在 rest pose 中应满足：

```text
R_rest_global[carrier] @ u_forward_local = [0,0,1]
```

运行时：

```text
f(t) = R_global_carrier(t) @ u_forward_local
n(t) = hypot(f.x, f.z)
theta(t) = atan2(f.x, f.z), if n(t) >= eps_h
h(t) = [cos(theta), sin(theta)]
```

`eps_h` 先由源数据退化直方图标定。前向接近竖直时 heading 在几何上没有定义，因此必须同时
存 `heading_valid[T]`：

- 正常帧：写真实 h，`heading_valid=1`；
- 有效帧之间的退化区间：数值上做 circular interpolation，但 mask 仍为 0；
- 首尾退化区间：数值取最近有效 heading，但 mask 仍为 0；
- 整段无有效 heading：数值写确定性 sentinel `[1,0]`，整段 mask 为 0；
- mask=0 的帧不进入 heading loss、统计或条件采样；推理接口必须显式支持“无 heading 条件”。

## 6. Mask 契约

至少需要以下 mask，不能只靠零 padding：

```text
frame_mask[T]                 # 时间 padding
joint_mask[J]                 # 物理 joint padding
channel_valid_mask[J,17]      # 静态结构有效性
heading_valid[T]              # 动态 heading 几何有效性
rotation_supervised[J]        # 旋转来源是否可审计
contact_supervised[J]         # contact 是否可监督
edge_mask[J]                  # 非 root 且父子都有效
```

静态 channel mask：

```text
valid physical root : [True]*17
valid non-root joint: [True]*13 + [False]*4
padded joint        : [False]*17
```

heading 最终 loss mask 是 `frame_mask & heading_valid`，且只取 root 的 ch 15:17。root-only
trajectory统计只取 `frame_mask` 下 root 的 ch 13:15。任何非 root 的 ch 13:17 零值都不能进入
loss、RMS、异常值统计或 codebook occupancy 解释。

## 7. 两条独立恢复路径

### 7.1 Position-direct

只用 ch 0:3 与 root ch 13:15：

```text
P_direct[...,0] = q[...,0] + smooth_x
P_direct[...,1] = q[...,1]
P_direct[...,2] = q[...,2] + smooth_z
```

### 7.2 Rotation-FK

先恢复全局旋转，再用 direct root 位置作为唯一 root translation：

```text
R_global[j] = decode(d6[j]) @ R_rest_global[j]
P_fk[root] = P_direct[root]
P_fk[c] = P_fk[p] + R_global[p] @ offset[c]
R_local[root] = R_global[root]
R_local[c] = R_global[p].T @ R_global[c]
```

两条路径都必须保留。`P_direct-P_fk` 是数据 QA、模型 loss 和重建诊断，不允许可视化时偷偷选择
看起来更好的一条。

## 8. 图和 topology 契约

- `J`、`joint_mask`、`parents`、adjacency、geodesic distance 全部只含物理关节；
- root parent 为 `-1`，parents 在 children 之前；
- ch 13:17 不创建任何新 graph edge；
- topology ID 和 held-topology descriptor 不读取动作值，只读取物理 parent tree/静态语义；
- 当前项目可继续用 `J_max=144` 作为 loader 配置，但实施前应从新 inventory 重新确认最大
  `J_phys`；`J_max` 是 padding 上界，不是 schema 的固有常数；
- joint name embedding、bone length、rest offset、拓扑语义是静态 sidecar，不塞进时间变化的 D。

## 9. 时间、尺度和归一化

### 9.1 FPS

一个训练语料只能有一个 `fps_target`。候选 v1 为 **30 FPS**，原因是它与现有 MotionFix、
HumanML3D、interaction/HOI Kimodo-like 数据一致，并保留 human 30 FPS；24 FPS 动物通过时间戳
SLERP/线性 root translation 重采样到 30 FPS，不改变物理时长。

但 30 FPS 只有在以下 prototype gate 通过后才冻结：

- 24->30 后 source-FK 与重采样-FK 的骨长和旋转误差在阈值内；
- acceleration/jitter 没有系统性放大；
- contact 与 heading 统计可稳定标定；
- 每个 source family 的真实 native FPS 已从文件/manifest 读取，而不是按目录猜测。

若 gate 否决 30，必须选择另一个全局 FPS 后重建全部数据；禁止 24/30 混存却只给一个 fps。

### 9.2 长度单位

raw artifact 必须记录：

```text
length_unit_id
source_unit_to_meter | null
canonical_scale_factor
s_rig
```

只有 `source_unit_to_meter` 可审计时才能把数值称为米。否则称为 TopoX canonical length unit，
不伪装物理尺度。

### 9.3 Model-view normalization

raw 文件保存未归一化语义值。loader 使用每 rig 的
`s_rig = rest-pose AABB diagonal` 和 train-only block scalar gain：

```text
q_model = g_q * q / s_rig
v_model = g_v * v / s_rig
s_model = g_s * smooth_root_xz / s_rig
d6, heading, contact unchanged
```

`g_q/g_v/g_s` 各为一个全局 scalar，按训练 split 的 valid entries 计算 RMS；不做 per-axis、
per-joint z-score，不减几何均值。`g_s` 只看 root-only ch 13:15。held topology、padding、非 root
无效 0 和 heading-invalid 帧不能进入统计。

## 10. 平移与 yaw 等变

crop 只把 crop 首帧的 smooth-root XZ 平移到原点：

```text
s_xz' = s_xz - s_xz[t0]
q, d6, v, contact, heading unchanged
```

不旋转首帧 heading。训练期可选整段随机 yaw `Y(phi)`：

```text
s_xz' = Y_xz(phi) @ s_xz
q' = Y(phi) @ q
v' = Y(phi) @ v
R_global' = Y(phi) @ R_global
d6' = encode(R_global' @ R_rest_global.T)
heading' = [cos(theta+phi), sin(theta+phi)]
contact' = contact
```

随机 heading 增强属于训练/loader，不属于离线转换；离线数据保留源动作的实际朝向分布。

## 11. 推荐预处理顺序

1. 重做 raw-source inventory，逐 rig/clip 证明 rotation、native FPS、单位和 joint map 来源。
2. 解析 root translation、local rotations、rest transforms，内部计算至少 float64。
3. 用 source parser 自己 FK，先复现 source positions；失败则不能继续编码。
4. 应用 per-rig 常量 source-to-canonical 轴系、forward、scale、ground 变换。
5. 按 timestamp 重采样 root translation 和 local SO(3) rotations，再重新 FK；禁止逐关节线性插值位置
   作为旋转路径。
6. 用一个 clip-constant Y translation 落地；禁止逐帧贴地。
7. 在 full clip 上计算 smooth-root XZ、direct q、世界速度、contact、heading 和 validity。
8. 对 full clip 做首帧 smooth-root XZ 平移归零；保留原 origin 到 manifest。
9. 保存未归一化 float32 TJD、masks、rig payload 和 provenance。
10. loader 在线统一 crop、可选 yaw、valid-only normalization、最后 padding T/J。

## 12. 数据工件

```text
dataset/
  schema.json
  motions/<clip_id>.npz
  skeletons/<rig_id>.npz
  manifests/clips.jsonl
  splits/<protocol>/*.txt
  stats/train_block_gains.npz
```

`motions/*.npz` 至少包含：

```text
motion[T,J_phys,17] float32
heading_valid[T] bool
rotation_supervised[J_phys] bool
contact_supervised[J_phys] bool
clip_id, rig_id, fps, origin_xz
```

`skeletons/*.npz` 至少包含：

```text
joint_names, parents
P_rest_global, R_rest_global, offset_parent_local
heading_carrier_joint, u_forward_local
source_to_canonical_transform
s_rig, unit metadata
contact eligibility and joint-map metadata
```

`schema.json` 冻结 channel slice、6D convention、坐标系、FPS、smoother/contact/heading 参数和
repr version。研究复现需要语义版本与配置记录，不把生产级防篡改当作科学 gate。

## 13. 训练 loss 最小契约

不能把 17 维直接做一个无 mask、等权 MSE。至少按 block：

- smooth-root XZ robust L1，只在 root ch 13:15；
- q position robust L1；
- rotation matrix chordal loss 或 SO(3) geodesic metric，使用 rotation mask；
- world velocity robust L1；
- contact `BCEWithLogits`，使用 contact mask；
- heading cosine/chordal + unit-circle penalty，使用 heading-valid mask；
- position-direct vs rotation-FK parent-edge consistency；
- 可选 finite-difference velocity consistency，但速度永不参与位置恢复。

每个 block 先按样本 valid count 归约再做 batch mean，并记录 count。空 heading/contact block 的 loss
为 0 且显式计数；空 frame/joint block 是数据错误。

## 14. QA 和视觉验收

先做 Human、普通四足、winged、蛇、蜘蛛/螃蟹、Dragon/大深度拓扑至少各一条 prototype：

1. raw parser FK 复现 source；
2. canonical rest 的全部 d6 解码为 identity rest-delta；
3. direct encode/decode 每帧 round trip；
4. column-cont6d 非 identity 随机旋转 round trip；
5. direct-position 与 independent rotation-FK 报 MPJPE、edge error、per-joint rotation error；
6. 叶关节 rotation source coverage 为 100%，否则主 schema 拒收；
7. velocity 与 `delta P * fps` 一致；
8. 随机平移/yaw 等变测试；
9. 单帧扰动不影响其它帧解码；
10. 全部数值 finite，所有 mask 的 shape/count 与 parents 一致。

视觉必须同步显示 source、position-direct、rotation-FK 三路，使用透视相机、地面网格、XYZ 轴、
root/smooth-root 轨迹、heading 箭头、heading-invalid 标记和 canonical rest-pose 图。渲染不能额外
recenter、改地面或改 face direction。prototype 通过后才做 102k 级全量构建。

## 15. 与已有表示的关系

| 性质 | legacy AnyTop13 | 固定 Kimodo273 | KTJD-17 |
|---|---|---|---|
| topology | 可变 | 固定人体 joint 数 | 可变物理 J |
| root XZ | 速度积分 | smooth root 直读 | root ch 13:15 直读 |
| joint position | heading-local/特殊 root | smooth-root 相对、世界轴 | 同 Kimodo-like，逐帧直读 |
| rotation | legacy carrier，叶旋转可能丢失 | 全局 6D | raw-source global rest-delta 6D |
| velocity | 部分承担恢复 | 特征 | 特征，不参与恢复 |
| heading | 隐式/legacy | 显式 cos/sin | root ch 15:17 + valid mask |
| contact | per-joint legacy | human 4 feet | 任意拓扑 per-joint |
| rest pose | rig metadata 混杂 | 固定人体 skeleton | 每 rig canonical rest payload |
| frame-local decode | 否 | 是 | 是 |
| position/FK 双审计 | 不完整 | 是 | 是，且为 acceptance gate |

## 16. 贡献边界与实施建议

KTJD-17 的核心贡献不是“多加 4 维”，而是把 **控制友好的世界状态** 与 **拓扑可变的物理
joint 图** 放进一个语义稳定的 TJD 契约：J 始终是真关节，位置不再依赖积分，旋转由各 rig 的
rest frame 统一，heading/trajectory 能直接做条件，且 direct/FK 能互相揭错。

第一实施里程碑应是 codec + 六类 rig 的数值/视觉 prototype。不要从 legacy13 就地改写，不要先
全量转换，也不要复用旧 tokenizer/checkpoint；表示、normalization、VQVAE、token cache、backbone
和 evaluator 都是破坏性不兼容的，需要在 prototype 通过后统一重建。

