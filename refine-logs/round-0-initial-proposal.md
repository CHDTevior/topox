# Research Proposal: KTJD-15, a Kimodo-like Multi-topology Motion Representation

## Problem Anchor

- Bottom-line problem: design a topology-variable `[T,J,D]` motion representation for
  TopoX that preserves the useful properties of Kimodo-like 273D data: explicit global
  trajectory and heading, frame-local decoding without velocity integration, globally
  meaningful rotations and velocities, direct sparse control, and independent
  position/FK audit paths.
- Must-solve bottleneck: TopoX's current 13-channel AnyTop representation overloads
  the root row, reconstructs root XZ by integrating predicted velocity, rotates
  non-root positions into a per-frame heading frame, mixes FPS assumptions, and uses
  a rotation carrier layout that cannot faithfully retain every leaf rotation.
- Non-goals: preserve legacy checkpoint compatibility; represent facial/skin helper
  bones that are outside the selected kinematic skeleton; design a new VQVAE or
  diffusion architecture; claim physical metres when a source rig has no auditable
  unit conversion.
- Constraints: variable physical joint count (current project up to 144), multiple
  morphology families, existing graph masks/topology conditioning, current
  unseen-topology split, and raw PZ/TrueBones/Human rotation sources that must be
  re-inventoried before conversion.
- Success condition: a source motion can be encoded and decoded at any frame without
  temporal integration; direct-position and rotation-FK decoders agree; translation
  and yaw transforms are equivariant; all physical joint rows share exactly one
  semantic layout; held topologies can be loaded without changing D or channel rules.

## Method Thesis

Use one virtual WORLD node plus J homogeneous physical-joint nodes, yielding a single
logical tensor `X in R^[T,(J+1),15]`. The WORLD node is the sole carrier of smooth-root
XZ and heading; every physical node carries Kimodo-like root-relative/world-aligned
position, global rest-delta rotation, world velocity, and contact.

This is the smallest design that simultaneously avoids root-row overloading and avoids
broadcasting four global values J times. A split `[T,4] + [T,J,13]` storage form is
mathematically equivalent, but the canonical model interface is one TJD tensor.

## Fixed Coordinate and Time Contract

- Right-handed, `Y+` up, `XZ` ground.
- Canonical rest forward is `+Z`; positive yaw is right-hand rotation about `+Y`.
- Positions are stored in metres only when `unit_to_meter` is auditable. A source with
  unknown scale must provide a per-rig conversion before entering the metric schema.
- Target FPS is 30. PZ/TrueBones 24 FPS sources are resampled from root translation and
  local rotations using linear interpolation plus SLERP, then FK is rerun. Positions
  are never independently interpolated.
- No per-frame heading canonicalization and no first-frame heading canonicalization.
  A crop may be translated so its first smooth-root XZ is zero; it is not yaw-rotated.
- Full-clip features are computed before crop. Random yaw is training-time augmentation.

## Tensor Layout

Let `N = J_phys + 1`. Tensor node 0 is WORLD. Tensor nodes `1..J_phys` are physical
joints in FK order; tensor node 1 is the physical skeleton root.

Canonical layout:

```text
X.shape = [T, N, 15]

channel   block
0:3       position block
3:9       global rest-delta rotation, column-cont6d
9:12      world linear velocity, length units / second
12        per-joint contact
13:15     heading [cos(theta), sin(theta)]
```

WORLD node (`n=0`):

```text
X[t,0,0]      = smooth_root_x(t)
X[t,0,1]      = 0, invalid/reserved
X[t,0,2]      = smooth_root_z(t)
X[t,0,3:13]   = 0, invalid
X[t,0,13:15]  = [cos(theta(t)), sin(theta(t))]
```

Physical joint node (`n=j+1`):

```text
X[t,j+1,0:3]   = q_j(t)
X[t,j+1,3:9]   = d6_j(t)
X[t,j+1,9:12]  = v_j(t)
X[t,j+1,12]    = contact_j(t)
X[t,j+1,13:15] = 0, invalid
```

The static valid-channel mask is normative:

```text
WORLD: [1,0,1, 0,0,0,0,0,0, 0,0,0, 0, 1,1]
JOINT: [1,1,1, 1,1,1,1,1,1, 1,1,1, 1, 0,0]
```

Invalid entries are zero after normalization and are excluded from every loss and
statistic. Node type is a categorical embedding/metadata field, not a learned numeric
motion channel.

## Channel Semantics

### WORLD position: smooth root XZ

Let `P_root(t)` be the physical root trajectory. Smooth only its XZ coordinates on the
full clip; preserve no duplicate Y channel:

```text
s_xz(t) = Smooth(P_root[:,xz] / s_rig) * s_rig
```

Use the Kimodo ADMM smoother in scale-normalized coordinates with margin 0.03. For a
human-sized `s_rig ~= 2 m`, this recovers Kimodo's 0.06 m margin. For `T < 5`, use raw
root XZ deterministically rather than invoking the singular short-clip solve.

### Physical position block

Physical positions are world-axis aligned and relative only to smooth-root XZ:

```text
q_j(t) = [P_j.x(t)-s_x(t), P_j.y(t), P_j.z(t)-s_z(t)]
```

They are not rotated by heading. Y is ground-referenced world height. This follows the
useful Kimodo local-position convention while avoiding a second copy of root Y in the
WORLD node.

Direct recovery is frame-local:

```text
P_j(t) = [q_j.x+s_x, q_j.y, q_j.z+s_z]
```

No velocity, previous frame, or cumulative sum is read.

### Rotation block

Every retained physical joint, including root and leaves, stores a true global
rest-delta rotation:

```text
D_j(t) = R_global_j(t) @ R_rest_global_j.T
d6_j(t) = cont6d_columns(D_j(t))
R_global_j(t) = decode_cont6d(d6_j(t)) @ R_rest_global_j
```

At canonical rest, every `D_j = I`. The codec is active column-vector rotation; d6 is
the first two matrix columns, and decode uses Gram-Schmidt with `b3=b1 cross b2`.

Do not build this block from legacy AnyTop13 positions or IK. Conversion must return to
raw BVH/SMPL/MotionStreamer rotations. If a retained joint lacks a real rotation source,
either mark it explicitly unsupervised in a separately versioned lossy dataset or reject
it; never silently write identity.

### Velocity block

```text
v_j(t) = (P_j(t+1)-P_j(t)) * 30
v_j(T-1) = v_j(T-2); T=1 -> 0
```

Velocity is world-aligned and measured per second. It is an input/supervision feature,
never a reconstruction dependency.

### Contact block

`contact_j` is binary per physical joint. Source labels are preserved when their
meaning is auditable. Otherwise derive after resampling and grounding with morphology-
scaled thresholds and a per-rig eligibility mask. A starting detector is:

```text
height_j <= 0.075 * s_rig
speed_j  <= 0.050 * s_rig / second
```

Thresholds require prototype calibration before being frozen. Do not hard-code a human
foot list: snakes and other non-limb topologies can have meaningful body contacts.

### Heading block

Each rig has a reviewed `heading_carrier_joint` and a body-fixed forward axis
`u_fwd_local`. Canonical rest maps it to +Z. Per frame:

```text
f(t) = R_global_carrier(t) @ u_fwd_local
n(t) = hypot(f.x, f.z)
theta(t) = atan2(f.x, f.z), when n >= eps_h
h(t) = [cos(theta), sin(theta)]
```

Use `eps_h=0.05` as a prototype value. Invalid near-vertical runs are filled by circular
interpolation between nearest valid headings (nearest-value fill at clip boundaries)
so the tensor stays finite, but `heading_valid[t]=0` masks heading supervision and
heading-consistency loss there. An all-invalid clip is retained for pose learning with
the heading block masked; it is not deleted merely because heading is undefined.

Heading is intentionally redundant with rotation because it is a direct control and
conditioning interface. On valid frames, enforce agreement with carrier rotation.

## Rest-pose and Skeleton Payload

Every rig, not every topology class, owns a canonical skeleton payload:

```text
rig_id
topology_id
joint_names[J]
parents_fk[J]               # physical tree only, root parent=-1
P_rest_global[J,3]
R_rest_global[J,3,3]
offset_parent_local[J,3]
root_joint
heading_carrier_joint
u_fwd_local[3]
s_rig
unit_to_meter
contact_eligible[J]
rotation_supervised[J]
source_to_canonical_joint_map
source_to_canonical_world_transform
```

`P_rest_global` and `R_rest_global` must come from the same rest frame. The per-rig
constant world transform maps source up to +Y, source rest forward to +Z, and ground to
Y=0. Apply that same constant transform to every clip. This is rig canonicalization,
not per-clip heading canonicalization.

Define parent-local offsets by:

```text
offset[c] = R_rest_global[parent(c)].T
            @ (P_rest_global[c]-P_rest_global[parent(c)])
```

`s_rig` is the canonical rest AABB diagonal over retained physical joints. It must be
positive and is used for scale-equivariant smoothing and training normalization.

## Two Independent Decoders

Direct decoder:

```text
P_direct = decode(s_xz, q)
```

Rotation/FK decoder:

```text
P_fk[root] = P_direct[root]
R_global[j] = decode(d6[j]) @ R_rest_global[j]
P_fk[c] = P_fk[p] + R_global[p] @ offset[c]
```

Local rotations for BVH/skinning are derived exactly:

```text
R_local[root] = R_global[root]
R_local[c] = R_global[parent(c)].T @ R_global[c]
```

Direct positions expose sparse world constraints; FK rotations expose a rigid,
bone-consistent pose. Their disagreement is a first-class QA metric and training loss,
not hidden by choosing whichever rendering looks better.

## Graph Contract

- Model graph has N nodes, but FK has only J physical joints.
- WORLD is never a bone and never enters edge-length/FK losses.
- `fk_parents` remains length J with root=-1.
- `model_edges` adds a special WORLD relation. Recommended attention semantics:
  WORLD has a special bidirectional relation to every valid physical joint; physical
  parent-child edges retain their existing relation type. Do not include WORLD edges
  when computing topology IDs or held-topology descriptors.
- Existing `max_joints=144` becomes `max_nodes=145` if all 144 physical joints remain.
- WORLD cannot be removed by joint-drop augmentation. Physical root also remains.

## Preprocessing Order

1. Inventory raw sources and prove every retained joint rotation carrier exists.
2. Parse root translation and local rotation in float64 with native FPS/unit/axis data.
3. Apply the per-rig constant coordinate, metric-scale, rest-forward and ground transform.
4. Map once to canonical physical-joint order.
5. Resample root translation plus local rotations to 30 FPS; run FK once.
6. Ground with one clip-constant Y translation. Prefer source ground metadata; otherwise
   estimate from valid contact/support frames, with a robust low-quantile fallback.
   Never ground each frame independently.
7. Smooth full-clip root XZ in scale-normalized coordinates.
8. Encode q, global rest-delta d6, world velocity, contact and heading.
9. Store full length, unnormalized float32 plus masks and manifest.
10. Online: crop all blocks together, translate WORLD XZ by crop start, optionally yaw-
    augment, scale-normalize, then pad T and N last.

## Translation and Yaw Equivariance

Crop translation by `a_xz` changes only WORLD `[x,z]`:

```text
s_xz' = s_xz - a_xz
q, d6, v, contact, heading unchanged
```

For one rigid yaw `Y(phi)`:

```text
s_xz' = Y_xz(phi) s_xz
q'    = Y(phi) q
v'    = Y(phi) v
D_j'  = Y(phi) R_global_j R_rest_global_j.T
h'    = [cos(theta+phi), sin(theta+phi)]
contact' = contact
```

Do not yaw-canonicalize afterward; that would erase the augmentation and the desired
arbitrary-heading distribution.

## Normalization

Raw artifacts remain in metric units. The model view uses reversible, geometry-
preserving block scaling:

```text
s_norm = g_s * s_xz / s_rig
q_norm = g_p * q / s_rig
v_norm = g_v * v / s_rig
d6, heading, contact unchanged
```

`g_s,g_p,g_v` are three scalar train-split-only RMS gains. Do not apply per-axis or
per-joint z-score to geometric blocks; it breaks rotation equivariance and physical
distance ratios. Invalid channels are excluded from stats. Held topology data cannot
contribute to gains.

## Masks and Loader Output

Required sample fields:

```text
motion[T,N,15] float32
frame_mask[T] bool
node_mask[N] bool
physical_joint_mask[N] bool
feature_valid_mask[N,15] bool
heading_valid[T] bool
contact_supervised[J] bool
rotation_supervised[J] bool
fk_parents[J] int
rest_* / offsets / rig_id / topology_id / caption
```

Padding is zero and applied after normalization. Every reduction uses the appropriate
intersection of frame, node, feature and supervision masks.

## Minimal Loss Contract

- WORLD XZ trajectory loss.
- Joint direct-position robust L1.
- Global rest-delta rotation geodesic loss.
- World velocity robust L1.
- Contact BCE on contact-supervised joints.
- Heading cosine/unit-circle loss on heading-valid frames.
- Direct-position versus FK consistency, including parent-edge length error.
- Optional finite-difference consistency `Delta P * fps` versus predicted velocity;
  velocity still never enters decoding.

Losses must report per-block valid counts. Empty contact/heading blocks contribute zero
with an explicit counter; an empty frame block is a data error.

## Prototype and Acceptance Gates

Prototype at least one clip each from: Human, ordinary PZ quadruped, winged animal,
Anaconda/KingCobra, Crab/Spider, and a deep/large topology such as Dragon.

Numerical gates:

1. Source parser FK reproduces source positions before encoding.
2. Canonical rest encoded rotations are identity d6 for all joints.
3. Direct round trip max position error <= 1e-6 m in float64 and <= 1e-5 m after f32.
4. Rotation codec SO(3) and round-trip gates on non-identity random rotations.
5. Source versus decoded global rotation geodesic error is reported for every joint,
   including leaves.
6. Position-vs-FK MPJPE/edge error is below source-family thresholds frozen from the
   prototype, not guessed globally.
7. Velocity equals finite difference at 30 FPS; contact is binary.
8. Translation and yaw equivariance tests pass for direct and FK decoders.
9. Temporal locality test: perturbing frame t cannot alter decoded frames t+1...T.
10. All arrays finite; masks and N/J ordering consistent.

Visual gate for every prototype:

- source, direct-position decode and rotation-FK decode in three synchronized views;
- perspective camera, ground grid, XYZ axes, root/smooth-root trajectories;
- heading arrow and heading-valid state;
- rest-pose sheet with +Y/+Z labels;
- no render-only recentering or floor alignment.

Only after prototype gates pass should the full 102k-scale corpus be converted. The
repo's archived figures (102,438 clips / 382 rigs / 194 topologies) are planning
numbers from handoff docs, not fresh evidence in this clone; the implementation agent
must regenerate inventory evidence before using them as acceptance counts.

## Artifact Layout and Reproducibility

Use a simple research schema contract rather than production anti-tamper machinery:

```text
dataset/
  motions/<clip_id>.npz
  skeletons/<rig_id>.npz
  manifests/clips.jsonl
  splits/<protocol>/*.txt
  stats/train_block_gains.npz
  schema.json
```

Each artifact records `repr_version = ktjd15-v1`, preprocess config, source path/hash,
rig ID, FPS, units and joint-map version. Semantic/version mismatch must fail clearly;
cryptographic sealing is not a scientific requirement.

## Migration Recommendation

Do not incrementally reinterpret existing 13ch files. Build KTJD-15 from raw rotations,
retain the old dataset/checkpoints as a baseline, and retrain tokenizer, backbone and
evaluator. Preserve the existing unseen-topology split IDs where clips survive so the
representation comparison does not change the protocol at the same time.

The first implementation milestone is codec + six-rig visual/numeric prototype, not
full conversion and not model training.
