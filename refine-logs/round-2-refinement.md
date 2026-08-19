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

若该常量变换的旋转、尺度和平移分别为 `C`、`alpha`、`o`，实现必须按主动列向量约定执行：

```text
P_canonical = alpha * C @ (P_source - o)
R_canonical = C @ R_source @ C.T
```

同一个 `C` 必须同时用于 motion rotation 与 rest rotation。只旋转位置却不共轭旋转矩阵，或只
修改 rest 而不修改 motion，都会破坏 rest-delta 和 FK。

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
| `15:17` | `heading` | 2 | 结构上仅物理 root | `[cos(theta),sin(theta)]`；逐帧有效性见 `heading_valid` |

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

采用主动旋转、列向量、右手系。以下 codec 是 schema 的逐元素定义，不允许换成 row-6D、SVD、
极分解或其它正交化器：

```text
D_j(t) = R_global_j(t) @ R_rest_global_j.T
encode(R) = [R00,R10,R20,R01,R11,R21]

a1 = d6[0:3]
a2 = d6[3:6]
b1 = a1 / max(norm(a1), 1e-8)
u2 = a2 - dot(b1,a2) * b1
b2 = u2 / max(norm(u2), 1e-8)
b3 = cross(b1,b2)
decode(d6) = stack_columns(b1,b2,b3)

d6_j(t) = encode(D_j(t))
R_global_j(t) = decode(d6_j(t)) @ R_rest_global_j
```

标准 rest pose 下所有 `D_j=I`。两个必须逐值相等的 gold case：

```text
encode(I) = [1,0,0, 0,1,0]
Y(+pi/2) = [[0,0,1], [0,1,0], [-1,0,0]]
encode(Y(+pi/2)) = [0,0,-1, 0,1,0]
```

GT 构建时若 `norm(a1)<1e-6` 或 `norm(u2)<1e-6` 直接 abort；模型输出只计退化率并报警，
不能让 NaN 进入 FK。构建内部使用 float64，最终 motion 存 float32。

**Lossless 主 schema 的硬条件**：

- BVH 类源必须回到原始 BVH rotation channels；
- SMPL/MotionStreamer 类源必须回到其原始旋转通道和对应 rest skeleton；
- 禁止从 legacy AnyTop13、关节位置或 IK 生成主 schema 的旋转 GT；
- 有 rotation channel 的叶关节必须保留其自身旋转；
- source 明确定义为 fixed joint/end-site 的 identity local DOF 可以按源层级精确传播；
- 无法证明旋转来源的 clip/rig 不进入 lossless v1。若未来确需保留，另建带
  `leaf_rot_unobserved_mask` 的 `ktjd17-lossy-*`，不能与主数据混训而不声明。

每个 joint 另存 `rotation_source_kind in {animated_dof,fixed_dof}`：

- `animated_dof` 包含真实 source rotation channel，进入 primary rotation loss；
- `fixed_dof` 的 d6 在数据中按 source hierarchy 精确写出，但不作为一条独立观察进入 primary
  rotation loss；生成/解码时由 parent global rotation 与该 joint 的 rest-local fixed transform
  硬重建，不能把模型随意输出当作自由 DOF；
- 缺失、IK 猜测或来源不明不是第三种可接收状态，而是主 schema reject。

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

v1 中 `contact[t,j]` 只表示 **joint-proxy ground support contact**，不泛指物体/自接触。全部
source family 使用同一条 canonical 后重算路径，禁止把 source label 与派生 label 混在 ch12：

```text
h_norm(t,j) = P_j.y(t) / s_rig
speed_norm(t,j) = norm(v_j(t)) / s_rig
contact(t,j) = (h_norm(t,j) <= tau_h)
               and (speed_norm(t,j) <= tau_v)
contact(T-1,j) = contact(T-2,j), when T>=2
```

- retained physical joints 全部 eligible，因此蛇的躯干不会被 human foot-name 规则清零；
- ground 无法审计的 clip 必须 reject，不能把其 contact 标成“未监督”后混入；
- `tau_h/tau_v` 只用 train split 的 prototype/full-train 分布标定，冻结后全量重建；
- 原始 source contact 仅可另存 `source_contact_audit` sidecar，用于测一致率，不进入 ch12；
- v1 中 `contact_supervised = joint_mask`，模型以 BCEWithLogits 学习这一统一定义。

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

payload provenance 必须是下列之一：

1. `explicit_reviewed`:源 rig/作者元数据明确给出 body forward 与 carrier，并通过 rest-pose 图复核；
2. `anchor_reviewed`:由左右锚点或 root-to-head/spine 轴产生两个正负候选，再用 source locomotion
   方向和 rest-pose 图人工确定 polarity；审核结果按 rig 固化。

carrier 默认只能选能代表整躯干刚体朝向的物理 root/pelvis/torso。自动候选必须在非退化运动帧
与独立位置锚点 forward 的 circular median 误差通过 train-only 标定门槛。缺 carrier、缺 polarity、
rest 等式不成立或审核不通过的 rig **fail-closed reject**；不得回退到世界 `+Z` 常量。

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
- 任意退化帧：写确定性 sentinel `[0,0]`，`heading_valid=0`；
- 整段无有效 heading：整段同样写 `[0,0]` 且 mask 全 0；
- mask=0 的帧不进入 heading loss、统计或条件采样；推理接口必须显式支持“无 heading 条件”。

## 6. Mask 契约

至少需要以下 mask，不能只靠零 padding：

```text
frame_mask[T]                 # 时间 padding
joint_mask[J]                 # 物理 joint padding
channel_valid_mask[J,17]      # 静态结构有效性
heading_valid[T]              # 动态 heading 几何有效性
rotation_supervised[J]        # animated_dof 才为 True
fixed_rotation_mask[J]        # fixed_dof 才为 True
contact_supervised[J]         # v1 对全部有效物理 joint 为 True
child_edge_valid[J]           # 以 child joint 下标表示的 parent-child 边
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

工件和 loader 的唯一派生规则如下。raw motion 文件不做 T/J padding，只存 `T_valid,J_phys`；
`heading_valid` 随 clip 存储，`rotation_source_kind` 随 skeleton 存储，其余由 loader 派生：

```text
frame_mask[t] = (t < T_valid)
joint_mask[j] = (j < J_phys)

channel_valid_mask[j,0:13] = joint_mask[j]
channel_valid_mask[0,13:17] = joint_mask[0]
channel_valid_mask[j>0,13:17] = False

rotation_supervised[j] = joint_mask[j] and source_kind[j] == animated_dof
fixed_rotation_mask[j] = joint_mask[j] and source_kind[j] == fixed_dof
contact_supervised[j] = joint_mask[j]

child_edge_valid[0] = False
child_edge_valid[c>0] = joint_mask[c] and joint_mask[parents[c]]
```

`parents` 只对 `0:J_phys` 有定义，root 必须为 0 且 `parents[0]=-1`。padding 后所有 motion 值和
mask 都补 0/False。任何实现若给非 root 的 ch13:17 计算 loss，即违反 schema。

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

对 `fixed_dof`，先按 `R_local[c]=R_rest_local[c]` 从 parent 递推其 `R_global[c]`，覆盖模型在该行
产生的自由 d6；对 `animated_dof` 才读取预测 d6。若下游需要相对 rest 的 local pose delta：

```text
R_local_delta[c] = R_local[c] @ R_rest_local[c].T
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

时间网格固定为：

```text
duration = (T_src-1) / fps_src
T_target = floor(duration * fps_target) + 1
t_target[k] = k / fps_target
```

只在源时间范围内采样，不外插；`fps_src==fps_target` 必须走逐值恒等旁路。禁止用
`linspace(0,T_src-1,T_target)` 拉伸首尾时间。root translation 线性插值，local SO(3) rotation
逐 joint SLERP，之后重新 FK。

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

`q_position` 的权威位置源也必须登记：BVH/只有 DOF 的源使用 raw rotation+translation FK；已有
经独立语义审核的 direct-position decoder 输出（例如同 FPS 的 human MotionStreamer272）可作为
`P_authoritative`，同时仍从独立 rotation stream 构建 d6。272 所需的速度积分只允许发生在离线
source decode；写成 KTJD 后不再参与训练期/推理期恢复。若需要改 FPS，rotation 路按 SLERP+FK，
position 路只在 source 明确定义连续位置时插值，并把两路差异纳入 §14 的 source-family gate。
禁止从 legacy AnyTop13 的在线积分结果直接生成主数据而不回到它的 raw source。

第 8 步之后存储的是 **clip-canonical world**，不是源文件的绝对世界平移。所有声称 absolute
source round-trip 的 QA 必须先执行：

```text
P_absolute[...,x] = P_decoded[...,x] + origin_xz[0]
P_absolute[...,z] = P_decoded[...,z] + origin_xz[1]
```

否则会把有意的 clip 平移归零误报成转换误差。

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
clip_id, rig_id, fps, origin_xz
```

以下字段不重复落盘，必须严格按 §6 的唯一公式派生：`frame_mask`、`joint_mask`、
`channel_valid_mask`、`rotation_supervised`、`fixed_rotation_mask`、`contact_supervised`、
`child_edge_valid`。

`skeletons/*.npz` 至少包含：

```text
joint_names, parents
P_rest_global, R_rest_global, R_rest_local, offset_parent_local
rotation_source_kind[J_phys]
heading_carrier_joint, u_forward_local
heading_payload_provenance
source_to_canonical_transform
s_rig, unit metadata
joint-map metadata
```

`schema.json` 至少逐字包含以下 key，不接受同义改名后静默加载：

```text
repr_version = "ktjd17-v1"
D = 17
root_index = 0
channel_slices = {
  q_position:[0,3], global_rest_delta_6d:[3,9],
  world_velocity:[9,12], contact:[12,13],
  smooth_root_xz:[13,15], heading:[15,17]
}
coordinate = {
  handedness:"right", up:"+Y", ground_plane:"XZ", rest_forward:"+Z",
  rotation_action:"active", vector_convention:"column"
}
rot6d = {
  order:"R00,R10,R20,R01,R11,R21", decoder:"gram_schmidt",
  eps:1e-8, third_axis:"b1_cross_b2"
}
fps_target
velocity = {difference:"forward", units:"length_per_second", tail:"repeat_last"}
smoother = {id, params, short_clip_rule}
heading = {eps_h, invalid_sentinel:[0,0], invalid_policy:"mask"}
contact = {definition:"joint_proxy_ground_support", tau_h, tau_v, tail:"repeat_last"}
normalization = {rig_scale:"rest_aabb_diagonal", gains:[g_q,g_v,g_s], center:false}
dtype = {build:"float64", storage:"float32"}
```

研究复现需要语义版本与配置记录，不把生产级防篡改当作科学 gate。

## 13. 训练 loss 最小契约

不能把 17 维直接做一个无 mask、等权 MSE。至少按 block：

- smooth-root XZ robust L1，只在 root ch 13:15；
- q position robust L1；
- rotation 默认用 decode 后 3x3 matrix chordal loss，SO(3) geodesic 只作报告指标，使用
  `rotation_supervised`；
- world velocity robust L1；
- contact `BCEWithLogits`，使用 `contact_supervised`；
- heading cosine/chordal + unit-circle penalty，使用 heading-valid mask；
- position-direct vs rotation-FK parent-edge consistency；
- 可选 finite-difference velocity consistency，但速度永不参与位置恢复。

每个 block 先按样本 valid count 归约再做 batch mean，并记录 count。空 heading/contact block 的 loss
为 0 且显式计数；空 frame/joint block 是数据错误。

## 14. QA 和视觉验收

验收分为固定 algebraic gate 和 train-only calibration gate，不能边看 held/test 边调阈值。

### 14.1 固定 gate

| gate | 判据 | fail 行为 |
|---|---|---|
| shape/tree | `D=17`；单 root；`parents[0]=-1`；`parents[c]<c`；mask 与 §6 逐位相等 | abort |
| finite | raw、rest、motion、stats 全部 finite；禁止 `nan_to_num` 掩盖错误 | abort，并报 clip/t/j/ch |
| 6D gold | identity 与 `Y(+pi/2)` 两个 gold case 逐值误差 `<=1e-12` in f64 | abort |
| 6D random | 10,000 个随机 SO(3) encode/decode Frobenius error `<=1e-10` in f64 | abort |
| rest identity | canonical rest 的 `max_abs(d6-encode(I)) <=1e-10` f64，存取 f32 后 `<=2e-6` | abort rig |
| direct round-trip | `max_abs(P_decode-P_encode) <=1e-10*s_rig` f64，f32 `<=1e-5*s_rig` | abort clip |
| velocity | 与 §5.3 forward difference 的 f32 误差 `<=1e-5*s_rig*fps_target` | abort clip |
| rigid edge | 对 rigid source，`max_t abs(length_t-length_rest)/s_rig <=1e-4` | reject clip |
| yaw equivariance | direct position误差 `<=1e-5*s_rig`，rotation Frobenius `<=2e-6`，heading `<=2e-6` | abort |
| locality | 改 frame t 的 TJD 后，解码其它帧必须逐位不变 | abort |
| rotation source | 每 joint 必须是 `animated_dof` 或可证明的 `fixed_dof`；不能 missing/IK | reject rig/clip |
| origin | 加回 `origin_xz` 后才与 absolute source 比较 | abort QA if omitted |

### 14.2 两遍 calibration/freeze

第一遍只用 train split，覆盖 Human、普通四足、winged、蛇、蜘蛛/螃蟹、Dragon/大深度拓扑，且
每个 source family 至少 30 clips，测：

```text
source-parser FK error / s_rig
P_direct vs independent P_fk MPJPE and p99 / s_rig
source global-rotation geodesic error
24->30 resample acceleration/jitter ratio
rotation-heading vs independent position-anchor circular error
heading degeneracy fraction/run length
normalized joint height/speed and contact positive rate
smooth-root residual / s_rig
```

对每个 source family 和 topology-distance bucket 保存完整分布。source-FK、P-vs-FK、resample
等 **误差 gate** 可先按 `max(engineering_floor, 1.5 * train_Q99.9)` 提议，再由六类可视化确认；
任何阈值若需要超过 `10x` 同类中位数，必须解释数据语义而不能直接放宽。contact 的
`tau_h/tau_v` 应用 train-only source-contact audit（若有）与人工视觉共同选定，不能套用 error
percentile 公式；`eps_h/margin_norm/fps_target` 分别按退化率、平滑残差和重采样审计选定。
随后把所有值写入 schema，冻结为第二遍正式全量构建配置。

第二遍使用冻结配置转换 train/val/held；val/held 不贡献 threshold 或 normalization。超阈值 clip
进入 explicit review/reject manifest，不能 clamp 后通过。正式 build 完成后重跑固定 gate 和冻结
gate，报告按 source family、rig 和 topology bucket 分层，不能只报全局平均。

视觉必须同步显示 source、position-direct、rotation-FK 三路，使用透视相机、地面网格、XYZ 轴、
root/smooth-root 轨迹、heading 箭头、heading-invalid 标记和 canonical rest-pose 图。渲染不能额外
recenter、改地面或改 face direction。prototype 通过后才做 102k 级全量构建。

现有 handoff 中的约 `102,438 clips / 382 rigs / 194 topology` 只是归档规划数，不是本设计轮次
重新生成的 inventory 证据；正式报告必须由新构建脚本重新计数。

## 15. 为什么它更适合控制

- **轨迹控制**：直接约束 root ch 13:15，不需要把目标路径先微分成速度再期待积分不漂移。
- **朝向控制**：直接约束 root ch 15:17；invalid mask 能区分“未指定/几何未定义”和真实 `+Z`。
- **稀疏 joint 控制**：给定目标世界点 `P*` 和 smooth path，直接转成
  `q*=[P*.x-s.x,P*.y,P*.z-s.z]`，对任意 topology 使用同一公式。
- **速度/contact 控制**：ch 9:12 与 ch12 是显式属性，不承担恢复职责，因此可单独 mask、编辑或
  加 consistency loss。
- **刚体可实现性**：direct position 满足控制目标，rotation-FK 路径检查骨长与关节旋转是否物理
  可实现；两者的差值直接暴露控制冲突。

推荐模型入口把 TJD 无损 factor 成 root-global 4ch 与 joint-common 13ch：root-global 经独立
projection 后 broadcast/FiLM/cross-attention 到物理 graph；graph pool 仍只处理 J 个真实 joint。
这是模型内部优化，不改变序列化格式。

## 16. 与已有表示的关系

| 性质 | legacy AnyTop13 | 固定 Kimodo273 | KTJD-17 |
|---|---|---|---|
| topology | 可变 | 固定人体 joint 数 | 可变物理 J |
| root XZ | 速度积分 | smooth root 直读 | root ch 13:15 直读 |
| joint position | heading-local/特殊 root | smooth-root 相对、世界轴 | 同 Kimodo-like，逐帧直读 |
| rotation | legacy carrier，叶旋转可能丢失 | 全局 6D | raw-source global rest-delta 6D |
| velocity | 部分承担恢复 | 特征 | 特征，不参与恢复 |
| heading | 隐式/legacy | 显式 cos/sin | root ch 15:17 + valid mask |
| contact | per-joint legacy | human 4 feet | 任意拓扑 joint-proxy ground support |
| rest pose | rig metadata 混杂 | 固定人体 skeleton | 每 rig canonical rest payload |
| frame-local decode | 否 | 是 | 是 |
| position/FK 双审计 | 不完整 | 是 | 是，且为 acceptance gate |

## 17. 贡献边界与实施建议

KTJD-17 的核心贡献不是“多加 4 维”，而是把 **控制友好的世界状态** 与 **拓扑可变的物理
joint 图** 放进一个语义稳定的 TJD 契约：J 始终是真关节，位置不再依赖积分，旋转由各 rig 的
rest frame 统一，heading/trajectory 能直接做条件，且 direct/FK 能互相揭错。

第一实施里程碑应是 codec + 六类 rig 的数值/视觉 prototype。不要从 legacy13 就地改写，不要先
全量转换，也不要复用旧 tokenizer/checkpoint；表示、normalization、VQVAE、token cache、backbone
和 evaluator 都是破坏性不兼容的，需要在 prototype 通过后统一重建。
