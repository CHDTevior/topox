# HumanML3D -> AnyTop13 Conversion Plan

Date: 2026-06-19

Goal: convert the existing HumanML3D representation into the same AnyTop-style
dataset contract used by the current arbitrary-topology pipeline, so human
motions can be mixed into later VAE / Graph-VQVAE / CodeFlow training without
special model-side branches.

This is a plan only. No source data or training code has been changed.

## 1. Source Facts Verified

### Current AnyTop contract

Current loader: `src/data/anytop_dataset.py`

- Raw motion files are `motions/*.npy` with shape `[T, J, 13]`.
- Channel contract is documented at `src/data/anytop_dataset.py:5`:
  `0:3 RIFKE/relative pos | 3:9 6D rotation | 9:12 velocity | 12 contact`.
- Root is special:
  - root channel `1` is root height.
  - root channels `9` and `11` are x/z root velocity.
  - root channels `3:9` are root orientation in 6D.
  - root channel `0` carries angular-y velocity style state; current renderer
    does not directly use it, but preserving it keeps the representation honest.
- Rendering/recovery path:
  - `_recover_world_positions(raw13)` at `src/data/anytop_dataset.py:307`
    recovers world joints from raw `[T,J,13]`.
  - It uses root `3:9` for orientation, integrates root velocity from root
    channels `9` and `11`, uses root channel `1` as height, then rotates non-root
    channels `0:3` from root-relative space into world space.
- Training loader path:
  - raw motion is normalized per object with `cond[obj]["mean"]` and
    `cond[obj]["std"]` at `src/data/anytop_dataset.py:1004`.
  - normalized `anytop_x` is emitted as `[J,13,T]` at
    `src/data/anytop_dataset.py:1131`.
  - graph fields are built from `parents/offsets/joint_names`, not from the
    motion vector itself. Derived graph fields are built in
    `_build_derived(...)` at `src/data/anytop_dataset.py:210`.

### HumanML3D source data

Source root:

`/iridisfs/scratch/ts1v23/workspace/motion-latent-diffusion-main/datasets/humanml3d/HumanML3D`

Verified files:

- `new_joint_vecs/*.npy`: 29226 clips, all shape `[T,263]`.
- `new_joints/*.npy`: 29226 files, intended shape `[T,22,3]`; 3 one-frame
  entries are stored as `[22,3]`.
- `Mean.npy` and `Std.npy`: both `[263]`.
- split sizes:
  - `train.txt`: 23384
  - `val.txt`: 1460
  - `test.txt`: 4382
  - `all.txt`: 29226
- length stats from `new_joint_vecs`:
  - min 1, median 149, mean 140.88, p95 199, p99 199, max 469.
  - 24 clips are longer than 300 frames.
  - 10260 clips are longer than 196 frames.
  - 6 clips have `T <= 4`.

HumanML3D text format:

`texts/<motion_id>.txt` lines are:

```text
caption#token/POS token/POS ...#start_time#end_time
```

For our caption JSON, use only the first field before the first `#`.

### HumanML3D 263-dim semantic layout

The authoritative parser is:

`/iridisfs/scratch/ts1v23/workspace/motion-latent-diffusion-main/mld/data/humanml/scripts/motion_process.py`

The source comments at lines 354-361 define the layout:

```text
root_rot_velocity         [1]
root_linear_velocity_xz   [2]
root_y                    [1]
ric_data                  [(J-1)*3]
rot_data                  [(J-1)*6]
local_velocity            [J*3]
foot_contact              [4]
```

For HumanML3D, `J=22`, so:

```text
0                 root angular-y velocity
1:3               root local x/z velocity
3                 root height y
4:67              non-root RIC positions, [21,3]
67:193            non-root 6D rotations, [21,6]
193:259           local joint velocities, [22,3]
259:263           foot contacts, 4 bits
```

This sums to `1 + 2 + 1 + 63 + 126 + 66 + 4 = 263`.

The HumanML3D recovery path:

- `recover_root_rot_pos(data)` at lines 362-385 integrates root angular velocity
  into a yaw quaternion and integrates root x/z velocity into root position.
- `recover_from_ric(data, joints_num)` then rotates RIC positions into world
  coordinates and concatenates the root.

### Human skeleton topology

HumanML3D uses 22 SMPL-like joints. Names are available in:

`/iridisfs/scratch/ts1v23/workspace/motion-latent-diffusion-main/mld/utils/joints.py:126`

Use these 22 names:

```text
pelvis, left_hip, right_hip, spine1,
left_knee, right_knee, spine2,
left_ankle, right_ankle, spine3,
left_foot, right_foot, neck,
left_collar, right_collar, head,
left_shoulder, right_shoulder,
left_elbow, right_elbow,
left_wrist, right_wrist
```

Topology comes from `t2m_kinematic_chain` at:

`/iridisfs/scratch/ts1v23/workspace/motion-latent-diffusion-main/mld/data/humanml/utils/paramUtil.py:55`

Derived parent array:

```python
parents = [
    -1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8,
    9, 9, 9, 12, 13, 14, 16, 17, 18, 19,
]
```

Foot contact order in HumanML3D is from `fid_l=[7,10]`, `fid_r=[8,11]`.
So the four contact bits map to:

```text
contact[0] -> left_ankle  joint 7
contact[1] -> left_foot   joint 10
contact[2] -> right_ankle joint 8
contact[3] -> right_foot  joint 11
```

## 2. Target Dataset Layout

Create a new independent converted dataset root, for example:

```text
data/humanml3d_anytop13/
  cond.npy
  object_index.csv
  DATASET_INFO.md
  motions/
    HML3D_Human_000000.npy
    ...
  texts/
    optional copied raw txts
  splits/
    train.txt
    val.txt
    test.txt
    all.txt
  motion_texts_by_file.json
```

Use one object type:

```text
HML3D_Human
```

This keeps HumanML3D as one fixed-topology skeleton inside the arbitrary-topology
framework. It should not be represented as 29226 separate skeletons.

Expected converted motion shape:

```text
raw HumanML3D vector: [T,263]
converted AnyTop raw: [T,22,13]
loader output anytop_x: [22,13,T_max] normalized
batched model tensor: [B,T_max,22,13] after permute in training scripts
```

## 3. Exact 263 -> 13ch Mapping

Let `x` be one HumanML3D `new_joint_vecs/*.npy`, shape `[T,263]`.

Slice it:

```python
J = 22
root_rot_vel = x[:, 0]                         # [T]
root_vel_xz  = x[:, 1:3]                       # [T,2]
root_y       = x[:, 3]                         # [T]
ric          = x[:, 4:4 + 21*3].reshape(T,21,3)
rot6d        = x[:, 67:67 + 21*6].reshape(T,21,6)
local_vel    = x[:, 193:193 + 22*3].reshape(T,22,3)
foot         = x[:, 259:263]                   # [T,4]
```

Pack to `raw13 = zeros([T,22,13])`:

```python
# Root RIFKE state.
raw13[:, 0, 0] = root_rot_vel
raw13[:, 0, 1] = root_y
raw13[:, 0, 2] = 0.0

# Root orientation: integrate HumanML3D yaw velocity exactly like
# recover_root_rot_pos(), then convert that quaternion to 6D.
yaw_state = np.zeros(T)
yaw_state[1:] = np.cumsum(root_rot_vel[:-1])
root_quat = [cos(yaw_state), 0, sin(yaw_state), 0]  # same convention as HML3D
raw13[:, 0, 3:9] = quaternion_to_cont6d_np(root_quat)

# Non-root RIC and non-root 6D rotations.
raw13[:, 1:, 0:3] = ric
raw13[:, 1:, 3:9] = rot6d

# Per-joint local velocity.
raw13[:, :, 9:12] = local_vel

# Renderer-critical root x/z velocity: enforce exact HumanML3D root fields.
raw13[:, 0, 9] = root_vel_xz[:, 0]
raw13[:, 0, 11] = root_vel_xz[:, 1]

# Contact: HumanML3D has only 4 foot bits, AnyTop13 stores per-joint contact.
raw13[:, 7, 12] = foot[:, 0]
raw13[:, 10, 12] = foot[:, 1]
raw13[:, 8, 12] = foot[:, 2]
raw13[:, 11, 12] = foot[:, 3]
```

Important: do not directly reshape `[T,263]` into `[T,J,13]`. The root fields
are in different locations, and root orientation must be reconstructed.

## 4. `cond.npy` Construction

Build one `cond.npy` entry:

```python
cond = {
  "HML3D_Human": {
    "parents": parents,                         # [22]
    "offsets": offsets,                         # [22,3]
    "tpos_first_frame": tpos_first_frame,        # [22,13]
    "joint_names": joint_names,                  # len 22
    "kinematic_chains": t2m_kinematic_chain,
    "mean": mean13,                              # [22,13]
    "std": std13,                                # [22,13]
    "joint_relations": joint_relations,          # [22,22]
    "joints_graph_dist": joints_graph_dist,      # [22,22]
  }
}
```

Recommended construction details:

- `joint_relations` and `joints_graph_dist`: reuse our local derivation path
  from `src/data/anytop_dataset.py:_build_derived(...)`, because that is what
  the loader trusts and validates.
- `offsets`:
  - Preferred: compute target offsets from HumanML3D's own canonical skeleton
    path, matching `Skeleton(n_raw_offsets, t2m_kinematic_chain).get_offsets_joints(...)`.
  - Practical source: use `new_joints/000021.npy` if present, since `000021`
    is the documented `t2m_tgt_skel_id`.
  - Gate: the FK route using these offsets must match the RIC route on sampled
    converted clips. If it does not, do not proceed to training.
- `tpos_first_frame`: use a representative converted first frame, preferably
  from `000021`, or a zero-motion rest-like frame if the implementation builds
  one. This field is currently mostly metadata, but it must have shape `[22,13]`.
- `mean/std`: recompute from the converted raw `[T,22,13]` over the training
  split only, per joint and per channel. Do not reuse the original flat
  HumanML3D `Mean.npy/Std.npy` directly, because the converted 13ch layout has
  a different axis structure.
- Apply a small std floor consistent with the existing AnyTop loader convention.

## 5. Text Conversion

For each `texts/<id>.txt`:

- Parse every non-empty line.
- Split by `#`.
- `caption = fields[0]`.
- Keep the original line metadata optionally, but do not feed POS tags as
  caption text.

Create:

```json
{
  "HML3D_Human_000000": {
    "primary_caption": "a man kicks something or someone with his left leg.",
    "captions": [
      "a man kicks something or someone with his left leg.",
      "the standing person kicks with their left foot before going back to their original stance.",
      ...
    ],
    "source_dataset": "HumanML3D",
    "source_motion_id": "000000"
  }
}
```

After conversion, build T5 caption caches with the same scripts currently used
for AnyTop:

- `scripts/precompute_t5_captions.py`
- `scripts/convert_caption_npz_to_npy.py`
- `scripts/precompute_t5_caption_tokens.py`

This lets the existing mean-text / token-text / dual-text branches consume the
converted human dataset without model changes.

## 6. Splits

Preserve HumanML3D's native split files:

```text
HumanML3D/train.txt -> data/humanml3d_anytop13/splits/train.txt
HumanML3D/val.txt   -> data/humanml3d_anytop13/splits/val.txt
HumanML3D/test.txt  -> data/humanml3d_anytop13/splits/test.txt
HumanML3D/all.txt   -> data/humanml3d_anytop13/splits/all.txt
```

Convert IDs to converted filenames:

```text
000000 -> HML3D_Human_000000.npy
```

The current `AnyTopDataset` only consumes `train.txt` and `val.txt`, but keeping
`test.txt` is useful for later evaluator work.

For mixing with animal datasets, build a separate union dataset root rather than
modifying this converted root in place.

## 7. Conversion QA Gates

These gates should run before any training uses the converted dataset.

### Gate A: shape and finite scan

For every converted motion:

- shape is `[T,22,13]`.
- no NaN/Inf.
- `T` matches the source `new_joint_vecs` length.
- report very short clips (`T <= 4`) and very long clips (`T > 300`).

Recommendation: keep the source files intact, but decide whether the 6 clips
with `T <= 4` should be excluded from train split. They carry almost no temporal
signal and can distort temporal losses.

### Gate B: RIC recovery equivalence

For sampled clips:

1. Run HumanML3D official `recover_from_ric(x, 22)` on the original `[T,263]`.
2. Run our `_recover_world_positions(raw13)` on the converted `[T,22,13]`.
3. Compare mean/max joint error.

Expected: near numeric tolerance. If this fails, root yaw integration or root
velocity packing is wrong.

### Gate C: FK route sanity

For sampled clips:

1. Run `_recover_world_positions(raw13)` as RIC reference.
2. Run `recover_from_bvh_rot_np(raw13, parents, offsets)`.
3. Compare mean/max joint error and bbox-relative error.

Expected: very small if `offsets` and root 6D reconstruction are correct.
If it fails while Gate B passes, the problem is `offsets` or FK convention, not
the 263->13 mapping.

### Gate D: visual QA

Render at least 15 GIFs:

- Use the current AnyTop-style renderer based on `_recover_world_positions`.
- Include:
  - locomotion,
  - kick/jump,
  - slow gesture,
  - long clip,
  - short clip,
  - several train/val/test examples.
- Put outputs under:

```text
data/humanml3d_anytop13/animations/conversion_qa_YYYYMMDD/
```

Visual check should compare:

- official HumanML3D recovered joints,
- converted AnyTop13 recovered joints.

Metric alone is not enough for this dataset.

### Gate E: loader smoke

Instantiate:

```python
AnyTopDataset(
    data_root="data/humanml3d_anytop13",
    split="train",
    num_frames=300,
    max_joints=144,
    random_crop=False,
    load_captions=True,
)
```

Verify:

- `anytop_x`: `[22,13,300]` before spatial pad or `[144,13,300]` after loader pad.
- `joint_mask.sum() == 22`.
- `frame_mask.sum() == T_clipped_or_padded`.
- `anytop_mean/std`: `[144,13]`.
- graph fields finite and Floyd-consistent.

## 8. Training Integration Strategy

Recommended staging:

1. Build standalone `data/humanml3d_anytop13`.
2. Run conversion QA and visual QA.
3. Build a union dataset root only after the standalone converted set passes.
4. For first mixed training, use `max_joints=144`; Human has only 22 joints and
   will not drive the max.
5. If using Graph-VQVAE / CodeFlow:
   - retrain tokenizer on the union if the human dataset is meant to influence
     the discrete latent vocabulary.
   - exporting tokens from an animal-only tokenizer for HumanML3D is possible
     as a diagnostic, but it is not the clean training setup.

## 9. Main Risks

1. Root orientation reconstruction is load-bearing. HumanML3D stores root
   angular velocity, while AnyTop13 renderer expects per-frame root 6D
   orientation. This must be reconstructed.
2. Root velocity channel locations differ. HumanML3D root x/z velocity is flat
   dims `1:3`; AnyTop13 renderer reads root joint channels `9` and `11`.
3. Offsets must match the HumanML3D 6D rotations. If offsets are approximate,
   RIC rendering can still look right, but FK-based QA/loss/rendering can fail.
4. Contact is only 4 foot bits in HumanML3D; AnyTop13 is per-joint contact.
   Mapping to four foot joints and zeroing the rest is the correct v1 behavior.
5. Very short clips exist. They are few, but they should be reported explicitly.

## 10. Implementation Checklist

Suggested new script:

```text
scripts/convert_humanml3d_to_anytop13.py
```

Core steps:

1. Read source root and output root.
2. Load `new_joint_vecs/*.npy`.
3. Convert each `[T,263]` to `[T,22,13]` with the mapping above.
4. Write `motions/HML3D_Human_<id>.npy`.
5. Build `cond.npy` with one `HML3D_Human` object.
6. Recompute train-split `mean/std` in converted 13ch space.
7. Write `object_index.csv`.
8. Convert split files.
9. Convert text files into AnyTop caption JSON.
10. Run Gates A-C in the script or a paired smoke script.
11. Render Gate D GIFs.
12. Run loader smoke Gate E.

Do not modify `AnyTopDataset` for v1 unless the standalone converted dataset
cannot satisfy the current schema. The current schema is already sufficient.

