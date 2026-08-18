# Human real-twist recovery from AMASS — code-grounded, PoC-proven plan

**Produced by:** multi-agent workflow (chain-trace + end-to-end injection PoC on real clips + pitfalls), 2026-07-02.
**Status:** method EMPIRICALLY PROVEN (2 real CMU clips); the remaining work + risk is DATA PLUMBING, not the method. Any implementation still goes through codex review + v3-discipline gates before a full run.
**One-line:** keep v3a's zero-twist swing (positions stay byte-exact) and inject the REAL AMASS SMPL axial twist as a roll about each current bone axis; twist is a gauge so positions don't move, and real AMASS twist is smooth so it does NOT reintroduce the v3 jitter.

---

# SYNTHESIS (build-ready plan)

# ANSWER — How AMASS→263→AnyTop works, and the correct way to give human data real twist as ground truth

This synthesizes three empirically-verified investigations (chain trace A, twist-injection PoC B, pitfalls C) into a build-ready answer. Where the three disagreed I take the more directly-tested claim and flag it (not averaged).

---

## (1) The verified full chain, with exact transforms and where twist dies

**Stage 1 — AMASS SMPL → joint positions.** `HM/datasets/humanml3d/raw_pose_processing.ipynb`:
- cell-5 loads SMPL-H `BodyModel` (`num_betas=10`, `num_dmpls=8`).
- cell-8 (`amass_to_pose`): `trans_matrix=[[1,0,0],[0,0,1],[0,1,0]]` (Y↔Z swap), `ex_fps=20`, `down_sample=int(fps/ex_fps)`. Builds `body_parms={root_orient=poses[:,:3], pose_body=poses[:,3:66], pose_hand=poses[:,66:], trans, betas[:10]}`, then `body=bm(**body_parms)` and takes **`pose_seq_np = body.Jtr`**.
  - **TWIST-LOSS POINT #1 (fatal):** only the 52 joint *positions* are kept. The full 3-DOF-per-joint SMPL axis-angle rotations — which carry axial twist — are discarded here and never saved to `pose_data/`.
- cell-18: per-dataset head-trims (Eyes_Japan/HDM05 60 frames, TotalCapture/MPI_Limits 20, Transitions 10 — all in 20fps frames), then `data[start:end]`, then **`data[...,0]*=-1`** (X-reflection), then `swap_left_right` (cell-16) makes the mirror; both `new_name` and `'M'+new_name` saved from the same index row.

**Stage 2 — positions → 263.** `HM/mld/data/humanml/scripts/motion_process.py`, driven by `motion_representation.ipynb` (`joints_num=22`, `face_joint_indx=[2,1,17,16]`, `example_id=000021`):
- `process_file` (L169): `uniform_skeleton`→`tgt_offsets` (L175), floor (L178), xz-origin (L185), face-Z+ (L193–213).
- `get_cont6d_params` (L283): `quat_params = skel.inverse_kinematics_np(positions, face_joint_indx, smooth_forward=True)` (L286) → `quaternion_to_cont6d_np` (L289).
- `skeleton.py inverse_kinematics_np` (L55): per bone `rot_u_v = qbetween_np(u, v)`, u=rest offset dir, v=current bone dir.
  - **TWIST-LOSS POINT #2 (structural):** `qbetween` is a minimal-arc **swing** → axial twist ≡ 0. Even if Stage 1 had kept rotations, this IK would zero the twist.
- 263 layout assembled (L344–349): `root(4)=[rot_vel, vel_xz(2), root_y]` + `ric[:-1](63)` + `rot6d[:-1](126)` + `local_vel(66)` + `feet(4)` = **263**; output length = pose-frames − 1.

**Stage 3 — 263 → AnyTop `[T,22,13]`.** `/scratch/ts1v23/workspace/noKslot_clean/scripts/convert_humanml3d_to_anytop13.py`:
- `convert_263_to_13` (L280): slices `root_rot_vel=x[:,0]`, `root_vel_xz=x[:,1:3]`, `root_y=x[:,3]`, `ric=x[:,4:67]`, `rot6d=x[:,67:193]`, `local_vel=x[:,193:259]`, `foot=x[:,259:263]`; integrates yaw→6D (L293–299); packs channels (L301–313). Gate B exact.
- `reencode_rot6d(...,"v3a")` (L230): rebuilds non-root ch3:9 in the animal per-parent convention. Single-child parents → `_swing_batch` (L261/L137, deterministic minimal-arc, rest bone `offsets[c]`→current bone); multi-child (pelvis 0, spine3 9) → `_kabsch_batch` (L265). Local `rotq[i]=WR[gp]ᵀ·WR[i]` (L271); token `new[:,j,3:9]=rotq[PARENTS[j]]` (L274). Rest offsets = `compute_offsets` (L316) = `000021` frame-0 bones.
  - **TWIST is deliberately 0 at v3a:** `_swing_batch` is twist-free by construction, so ch3:9 for single-child joints carries positions-exact swing and zero roll.

**Net:** twist is destroyed at Stage 1 (`body.Jtr`), re-zeroed at Stage 2 (`qbetween` swing IK), and left at zero by v3a at Stage 3. Positions (ch0:3 / RIC / FK) are exact throughout.

---

## (2) The correct method + PoC verdict

**Method (the gauge insight is correct).** Twist is a gauge for FK positions: rotating a bone about its own axis does not move its child. So keep v3a's swing (positions stay byte-exact) and post-multiply a real axial roll about each **current** bone axis:

```
WR_new[p] = R_twist(φ_p, a_p) @ S_p        # S_p = v3a swing, a_p = current bone axis
```

Because the converter recomputes every joint's swing from its own true world axis (not by frame inheritance), post-multiplying a roll at parent p leaves p's child and **all** descendants exactly in place. Inject **only** at single-child parents; keep Kabsch at multi-child pelvis 0 / spine3 9 (torso twist is already geometrically recovered there — injecting would double-count and is ill-defined with no single axis).

**The one non-obvious correction B discovered (this is load-bearing).** The naive `φ = twist-of-SMPL-relative-to-its-own-swing` is WRONG: it leaks global yaw into near-vertical bones (θ hits 80–170°) and blows up at the arms, where the `000021` rest (arms-down) vs SMPL rest (T-pose arms-out) differ by ~180° (jitter spikes to 130°). The correct φ is the roll of the **rest-aligned** true world rotation:
1. recover the clip's constant proper rotation `A` from bone directions (SMPL→HumanML3D frame; residual 0.18–0.50°);
2. per bone, a constant offset `C_p` = rotation-average of `A·G_smpl` vs the v3a swing (the 000021↔SMPL rest-pose gap; empirically ~5–50° legs/torso, ~80–94° arms — matching T-pose vs arms-down);
3. `φ_p(t)` = signed twist of `A·G_smpl·C_p` relative to the v3a swing about the actual current bone;
4. inject `R(φ_p, a_p) @ S_p`.

**PoC empirical verdict (numpy-only, 2 real CMU clips: `000043`←CMU/106/106_05, `000058`←CMU/132/132_02).** It works end-to-end:
- **(a) Positions preserved — EXACT.** FK(inject) vs FK(v3a) gauge diff = **0.13 / 0.27 µm** (numerical zero); FK-vs-RIC unchanged at the pre-existing 14.0 / 1.5 mm shared-skeleton floor.
- **(b) Round-trip — EXACT.** Injected roll re-read from packed ch3:9 = injected value to **4e-6°**.
- **(c) Twist real and dynamically correct.** Injected local rotation matches true SMPL local rotation to **<1°** for essentially all 15 single-child joints (clip2 all 0.05–0.98°; clip1 forearm worst 2.3°). Anatomical magnitudes physical: thighs 2–5°, shins 3–11°, upper arms 7–17°, forearms 5–11° (forearm/upper-arm largest = correct pronation/supination). v3a adds ≈0.
- **(d) Smoothness — identical to SMPL, nothing like v2.** Injected frame-delta == SMPL's to 2 decimals (~5–20°/frame) vs v2 random-Kabsch's ~90–180°/frame. Real AMASS twist does NOT reintroduce the v3 jitter.
- **Frame alignment PROVEN:** `T_263 = (end−start)−1` (000043 77→76, 000058 191→190); independent proof via joint-flexion cross-correlation agreeing at **lag 0** to 0.02–0.67° on both knees + both elbows. The SMPL→HumanML3D composite (`trans_matrix` det−1 ∘ x-flip det−1) is a **proper rotation (det=+1)**, so twist angles (rotation-invariant) transfer with correct sign — no reflection issue.

**Caveat the PoC itself flags:** `C_p` was estimated **per-clip** (it varied clip-to-clip, e.g. L_thigh 7.5° vs 40.4°), which flatters the <1° numbers. A real build must compute **one global per-bone `C_p`** from the SMPL-rest↔000021-rest poses and reuse it; per-clip residuals may then be marginally higher. Also the PoC used only low CMU ids where id→source identity happens to hold and `_poses.npz` exists — it did not exercise the id-remap or KIT-alignment blockers below.

---

## (3) End-to-end pipeline

1. **id → AMASS source (must remap, not identity).** Build `our_id → source_path` by matching `texts/{id}.txt` caption-set against `amass_annotations.json` (authoritative; its values carry the AMASS `path` + captions). Base ids `000000–014612`; mirrors = `base + 14613` (029226 clips = 14613×2). Fail-loud on no/ambiguous match. **Do not** assume `our_id == index.csv row` (see B2).
2. **Locate AMASS file & align frames.** Prefer original SMPL-H `_poses.npz` (CMU/EKUT: exact). Reconstruct the exact raw indices the original used so frame count == existing v3a length + 1; apply dataset head-trim (none for CMU/EKUT/KIT) + `[start:end]`; drop the last frame. If count can't be reconciled → skip (never emit silent garbage).
3. **Extract twist.** Take `poses[::ds][:,:66]` = root_orient(3)+pose_body(63) = joints 0–21 axis-angle (`poses[:,:3]==root_orient`, `poses[:,3:66]==pose_body` verified max-diff 0.0). Rotations only — no mesh, so betas/body-model not needed for the scalar twist. Recover clip `A` from bone dirs; get global per-bone `C_p`; compute `φ_p(t)` = signed roll of `A·G_smpl·C_p` vs v3a swing about the current axis; `np.unwrap` over time.
4. **Inject & rebuild rot6d.** `WR_new[p] = R(φ_p, a_p) @ S_p` at single-child parents only; keep v3a Kabsch at 0 and 9. Recompute local `rotq[i]=WR[gp]ᵀ·WR[i]`, repack `new[:,j,3:9]=rotq[PARENTS[j]]` exactly as L271–274.
5. **Positions untouched.** ch0:3 / RIC / local_vel / foot copied straight from v3a.
6. **Mirrors.** Do not re-align from AMASS; take base φ and apply L/R-swap + sign-negate (twist reverses under reflection): `φ_mirror[j'] = −φ_base[swap(j)]`, chains `left=[1,4,7,10,13,16,18,20]`↔`right=[2,5,8,11,14,17,19,21]`.
7. **Coverage tagging.** Emit per-clip `twist_valid` flag; uncovered clips fall back to v3a swing.

---

## (4) Pitfalls, ranked

**BLOCKING**
- **B2 — id remap (highest risk).** Our contiguous ids are NOT the standard/index ids: identity holds at 0/100/1000/5000 but BREAKS by 10000 (our `010000`="walks counterclockwise" vs annotations `010000`="hands on knees"); counts disagree (index.csv 14616 / annotations base 14614 / our base 14613). Naive `our_id==row` fetches the WRONG source → catastrophically wrong twist. **Mitigation:** caption-set match against `amass_annotations.json`, uniqueness-checked, fail-loud. *(This directly overrides investigation A's step-1 `row=int(id)`, whose verification only proved index.csv internal consistency, not that our built clip N came from row N — C's caption cross-check is the decisive test.)*
- **B1 — frame alignment for KIT (`_stageii` mismatch).** Local KIT is the SMPL-X 2021 re-release (`_stageii.npz`, `poses` 165, key `mocap_frame_rate`, 120fps) — not the 100fps release that built the data; naive `poses[::int(fps/20)]` gives wrong counts (97 vs 117 = the 5/6 ratio). Best sampling ds=5 length-matches only ~58% of KIT; the rest are tail-misaligned. **Note this is a KIT problem: CMU/EKUT have original `_poses.npz` and reproduce on-disk `new_joints` to err 0.00000 (A) and ran clean in the PoC (B).** **Mitigation:** reconcile per clip to the known target length (time-resample smooth φ if needed) or skip; fail-loud on the ~4 KIT files with framerate `-1`.
- **B4 — mirror sign (50% of data, position-invisible).** Wrong twist sign on mirrors corrupts half the dataset and is invisible to any FK/RIC check (only shows in skinning/mesh). **Mitigation:** apply swap+negate rule; validate the sign ONCE against a physically-mirrored SMPL pose, then apply to all 14.6k mirrors.
- **B3 — coverage ceiling (scope/resource decision for user).** Local AMASS = CMU/EKUT/KIT only → **7909/14616 ≈ 54%** index rows recoverable (CMU 2913, EKUT 351, KIT 4645). Of these, exact twist only for CMU+EKUT (**3264 rows**, ×2 mirrors) + approximate KIT (~2688 base). The 15 missing datasets (BMLmovi, Eyes_Japan, HDM05, …) need download from amass.is.tue.mpg.de. **humanact12 (1191) is positions-only / not AMASS — twist irrecoverable in principle.**

**MINOR / confirmed-safe**
- **M1 multi-child** (pelvis 0, spine3 9): keep v3a Kabsch, don't inject — 3 non-collinear children already constrain the 3-DOF; the v3 jitter came only from single-child rank-1 degeneracy.
- **M2 rest-agnostic injection is sound** (confirms the gauge insight; positions preserved by construction because each descendant swing is recomputed from its own true axis).
- **M3 continuity:** rot6d is matrix-level continuous across ±180°, and real SMPL twist is smooth (unlike v2's LAPACK sign-flips), so wrapped φ won't re-jitter; still `unwrap` + guard the genuine anti-parallel (swing≈180°) singularity via `_swing_batch`'s deterministic branch.
- **M4 betas:** twist axis is from current v3a positions (betas-independent up to bone length); use SMPL mean-shape rest for a betas-free zero-reference. No mesh needed.
- **M5 joint order:** `JOINT_NAMES`==SMPL first-22; `PARENTS`==`[-1,0,0,0,1,2,3,4,5,6,7,8,9,9,9,12,13,14,16,17,18,19]`; `poses[:,3:66]`==pose_body verified. No risk.
- **M6 frame edges:** replicate drop-last (263=frames−1); CMU/EKUT/KIT get no head-trim (only start:end + downsample + drop-last), so alignment for the recoverable subset is simple.

---

## (5) Validation gates (v3 discipline)

Per-clip, fail-loud:
1. **Positions-preserved:** FK(inject) vs FK(v3a) gauge diff < 1 µm; FK-vs-RIC unchanged vs v3a floor. (PoC: 0.13/0.27 µm.)
2. **Round-trip:** re-read roll from packed ch3:9 == injected φ (< 1e-4°).
3. **Twist-matches-AMASS:** injected local rotation vs true SMPL local rotation < ~1° (single-child); reject clips exceeding a threshold (surfaces mis-alignment).
4. **No-new-jitter:** injected frame-delta geodesic ≈ SMPL's (~5–20°/frame), and ≪ v2 (~90–180°). Enforce a per-frame-delta ceiling.
5. **Frame alignment:** independent cross-correlation of a joint-flexion angle (knee/elbow) between injected and SMPL peaks at lag 0 (< 1°).
6. **Visual (primary gate, per CV-QA-primacy):** GT-vs-injected side-by-side multi-frame render/animation for a handful of clips (walking, pronation-heavy, arm-cross) — send to user for the visual verdict; do NOT rely on metrics alone. Self-check the renderer first on a near-perfect known case.

Sequence: run global-`C_p` build → small sample (the CMU/EKUT exact-alignment clips first, then a few KIT to probe B1, then a few mirrors to probe B4) → all gates + user visual sign-off → **codex (gpt-5.5, xhigh, fresh thread) review of the converter change** → only then the full covered-subset build with `twist_valid` flags. New/changed code goes through codex review before any full run (iron rule).

---

## (6) Honest bottom line

**Feasible and correct — yes.** The core hypothesis is empirically confirmed: keep v3a's swing, inject real AMASS SMPL twist as a roll about each current bone. Positions stay byte-exact (twist is a true gauge), the injected twist is real, smooth, and faithful to SMPL to sub-degree, and it does **not** reintroduce the v3 jitter — real smooth AMASS twist (~5–20°/frame) is categorically different from v2's random-Kabsch (~90–180°/frame). So this **keeps the v3 jitter-fix intact** and makes the human data skinning-usable and twist-consistent with the (native-BVH-twist) animal data.

**Main risk is data plumbing, not the method.** In order:
1. **id remap (B2)** — must remap via `amass_annotations.json` captions, not `our_id==row`; getting this wrong silently injects the wrong clip's twist. This overrides A's identity assumption.
2. **coverage (B3)** — only ~54% recoverable locally, with humanact12 (1191) unrecoverable in principle; a full-coverage build needs the missing AMASS subsets downloaded (user resource decision).
3. **mirror sign (B4)** — 50% of data, position-invisible; validate once then apply.
4. **KIT `_stageii` alignment (B1)** — CMU/EKUT are exact via `_poses.npz`; KIT is ~58% alignable and approximate.
5. **global `C_p`** — the PoC's <1° used per-clip `C_p`; production must use one global per-bone offset, which may slightly raise residuals (still expected sub-few-degree).

**Recommended first deliverable:** the CMU+EKUT exact-alignment subset (3264 base rows ×2 mirrors, ~13k clips of real twist) with global `C_p`, `twist_valid` flags, all six gates + user visual sign-off + codex review — that is fully correct and unblocked today. Extend to KIT (approx) and to the downloaded subsets afterward; leave humanact12 at v3a.

Key files: chain `HM/datasets/humanml3d/raw_pose_processing.ipynb` (cell-8), `HM/mld/data/humanml/scripts/motion_process.py` (L283/L286, L344–349), `HM/mld/data/humanml/common/skeleton.py` (L55); converter `/scratch/ts1v23/workspace/noKslot_clean/scripts/convert_humanml3d_to_anytop13.py` (L230/L261/L265/L271/L274/L280/L316); source-of-truth map `HM/datasets/humanml3d/amass_annotations.json`; PoC `…/scratchpad/twist_poc.py`.

---

# APPENDIX A — full chain trace

I have everything needed, all empirically verified. Here is the complete DIMENSION A trace.

---

# DIMENSION A — Full-chain code trace: AMASS SMPL → HumanML3D 263 → AnyTop13, twist-loss points, and the per-clip AMASS-recovery recipe

All three transform stages traced against the actual code, with the recipe **empirically verified by end-to-end reproduction** (CMU/EKUT reproduce the on-disk `new_joints` to **err = 0.00000**). Along the way I found a hard blocker the task's "VERIFIED FACTS" did not anticipate: **the on-disk AMASS is a mixed release and only covers 3 of 18 datasets** — details in §4, which you must read before designing the injection.

Files (all absolute):
- `HM = /iridisfs/scratch/ts1v23/workspace/motion-latent-diffusion-main`
- raw stage: `HM/datasets/humanml3d/raw_pose_processing.ipynb`
- 263 stage: `HM/mld/data/humanml/scripts/motion_process.py`, `HM/mld/data/humanml/common/skeleton.py`, driver `HM/datasets/humanml3d/motion_representation.ipynb`
- AnyTop stage: `/scratch/ts1v23/workspace/noKslot_clean/scripts/convert_humanml3d_to_anytop13.py`
- index: `HM/datasets/humanml3d/index.csv`; AMASS: `HM/datasets/amass/motion_data/`

---

## 1. Stage 1 — AMASS SMPL → joint positions (this is where twist dies)

`raw_pose_processing.ipynb`:
- **cell-5**: loads SMPL-H `BodyModel` (male/female, `num_betas=10`, `num_dmpls=8`) from `./body_models/smplh/{male,female}/model.npz`.
- **cell-8** (`amass_to_pose`): the core:
  - `trans_matrix = [[1,0,0],[0,0,1],[0,1,0]]` (swaps Y↔Z), `ex_fps = 20`.
  - `down_sample = int(fps / ex_fps)`; `bdata_poses = bdata['poses'][::down_sample]`, `bdata_trans = bdata['trans'][::down_sample]`.
  - `body_parms = {root_orient=poses[:,:3], pose_body=poses[:,3:66], pose_hand=poses[:,66:], trans, betas=betas[:10]}`.
  - `body = bm(**body_parms)` → **`pose_seq_np = body.Jtr` ← TWIST LOST HERE**. Only the 52 joint *positions* are taken; the full 3-DOF-per-joint SMPL axis-angle rotations (which carry axial twist) are discarded and never saved.
  - `pose_seq_np_n = np.dot(pose_seq_np, trans_matrix)`; saved to `./pose_data/<DS>/…npy` (mirrors `./amass_data/<DS>/…`).
- **cell-18** (segment/mirror/relocate), per `index.csv` row, on the already-20fps `pose_data`:
  - dataset pre-trims (only these 5, applied in 20fps frames): `Eyes_Japan_Dataset` & `MPI_HDM05` drop `3*fps=60`; `TotalCapture` & `MPI_Limits` drop `1*fps=20`; `Transitions_mocap` drops `int(0.5*fps)=10`; `humanact12` gets none.
  - `data = data[start_frame:end_frame]`, then **`data[..., 0] *= -1`** (x reflection).
  - `data_m = swap_left_right(data)` (cell-16: another x-flip + L/R chain swap).
  - saves `new_name` (base) and `'M'+new_name` (mirror) — **same index row, no separate row for mirrors**.

## 2. Stage 2 — positions → 263 (twist stays lost: swing-only IK)

Driver `motion_representation.ipynb` cell-5 `__main__`: `joints_num=22`, `face_joint_indx=[2,1,17,16]`, `example_id=000021`, reads `./joints/`, calls `process_file(src, 0.002)` → saves `HumanML3D/new_joint_vecs/<id>.npy` (263) and `new_joints/<id>.npy` (positions via `recover_from_ric`).

`motion_process.py`:
- `process_file` (**L169**): `uniform_skeleton`→`tgt_offsets` (L175), floor (L178), xz-origin (L185), face-Z+ (L193–213).
- Rotations come from `get_cont6d_params` (**L283**): `quat_params = skel.inverse_kinematics_np(positions, face_joint_indx, smooth_forward=True)` (**L286**) → `quaternion_to_cont6d_np` (L289).
- **`skeleton.py` `inverse_kinematics_np` (L55)**: per bone, `rot_u_v = qbetween_np(u, v)` where `u`=rest raw-offset dir, `v`=current bone dir. **`qbetween` = minimal-arc SWING → axial twist ≡ 0.** This is the second twist-loss point (and would zero any twist even if Stage 1 had kept it).
- 263 layout assembled (L344–349): `root_data(4)[rot_vel, vel_xz(2), root_y]` + `ric[:-1]((J-1)*3=63)` + `rot6d[:-1]((J-1)*6=126)` + `local_vel(J*3=66)` + `feet_l/feet_r(4)` = **263**. Output length = (sliced pose frames) − 1.

## 3. Stage 3 — 263 → AnyTop13 `[T,22,13]`

`convert_humanml3d_to_anytop13.py`:
- `convert_263_to_13` (**L280**): slices exactly `root_rot_vel=x[:,0]`, `root_vel_xz=x[:,1:3]`, `root_y=x[:,3]`, `ric=x[:,4:67]`, `rot6d=x[:,67:193]`, `local_vel=x[:,193:259]`, `foot=x[:,259:263]`; integrates yaw→6D (L293–299); packs (L301–313). Gate B exact.
- `reencode_rot6d(..., "v3a")` (**L230**): rebuilds non-root ch3:9 in the animal per-parent convention. Single-child parents → `_swing_batch` (**L261/L137**, deterministic zero-twist minimal-arc, rest bone `offsets[c]`→current bone); multi-child (pelvis0, spine3=9) → `_kabsch_batch` (L265). Local `rotq[i]=WR[gp]ᵀ WR[i]` (L271); token `new[:,j,3:9]=rotq[PARENTS[j]]` (L274). **So v3a's twist is 0 by construction — positions (ch0:3) exact, twist unfilled.** Rest offsets = `compute_offsets` (L316) = `000021` frame-0 bones.

**Net:** twist is destroyed at Stage 1 (`body.Jtr`, positions only), re-confirmed zero at Stage 2 (`qbetween` swing IK), and v3a deliberately keeps it zero at Stage 3. Your gauge insight is correct: injecting `WR_new = R_twist(θ, current_bone_axis) @ v3a_swing` leaves ch0:3 / RIC / FK positions untouched and only writes the missing DOF.

## 4. The per-clip AMASS-recovery recipe — and the coverage blocker (verified)

**Recipe for a base clip (id N in 000000–014615):**
1. `row = int(id)`; read `index.csv` row N → `source_path, start_frame, end_frame, new_name` (`new_name == f"{N:06d}.npy"`, verified id==row exact end-to-end at rows 0,4,13,20,25,26,37,43,47,14601,14609,14612,14614).
2. AMASS file = `source_path` with `./pose_data/`→`HM/datasets/amass/motion_data/` and `_poses.npy`→(`_poses.npz` for CMU/EKUT, `_stageii.npz` for KIT).
3. `fps` from npz (`mocap_framerate` for CMU/EKUT SMPL+H; `mocap_frame_rate` for KIT SMPL-X); `ds = int(fps/20)`.
4. axis-angle twist source: SMPL+H `poses[::ds][:, :66]` = root_orient(3)+pose_body(63) = the 22-joint (root+21) rotations; SMPL-X stageii gives the same via keys `root_orient`/`pose_body`.
5. To land on the frames that match the existing 263: `poses[::ds]`, apply the same dataset pre-trim (§1; none for KIT/CMU/EKUT), then `[start_frame:end_frame]`. The resulting frame t corresponds to 263 frame t (drop the last frame, since 263 len = sliced−1).
6. The clip's positions were **x-reflected** (`[...,0]*=-1`) before the 263 was built, so the injected twist axis/sign must be carried through that reflection (rotation → M R M, M=diag(-1,1,1)); the SMPL twist is in the un-reflected SMPL frame.

**Mirror clips (id 014616–029225):** these are `swap_left_right` mirrors (re-numbered from the original `M`-prefixed ids by the user's `change_name.py`; on-disk all.txt/new_joint_vecs are plain-numbered 000000–029225, **no `M` prefix** — differs from the standard convention stated in the task). Do **not** try to re-align them from AMASS; recover the base twist and apply the L/R-swap + reflection transform to it.

**Blocker — data availability (this caps the whole effort):**
- On-disk `motion_data/` has **only KIT, CMU, EKUT** (6724 npz). The other 15 datasets referenced by `index.csv` are **absent**.
- Recoverable index rows = **7909 / 14616 (54%)**: KIT 4645, CMU 2913, EKUT 351. Unrecoverable: BMLmovi/Eyes_Japan/MPI_HDM05/BioMotionLab/ACCAD/DFaust/MPI_Limits/MPI_mosh/Transitions/TotalCapture/SFU/BMLhandball/HumanEva/SSM (5513 rows) + **humanact12 (1191 rows, which never had SMPL params — position-only in this pipeline, so twist is unrecoverable in principle)**.
- **The AMASS is a MIXED release** (surface conflict with the task's stated "156-dim/dmpls/mocap_framerate" SMPL+H facts):
  - **CMU/EKUT = canonical SMPL+H** (`_poses.npz`, `poses` 156, `mocap_framerate`, no root_orient key). Textbook recipe (`ds=int(fps/20)`: CMU 60→3 / 120→6, EKUT 100→5) reproduces the on-disk `new_joints` **EXACTLY (err 0.00000, all frames)**. **Twist here is exact and directly usable.** = **3264 rows** (CMU 2913 + EKUT 351).
  - **KIT = SMPL-X 2021 re-release** (`_stageii.npz`, `poses` 165, `mocap_frame_rate`, has root_orient/pose_body keys, 120fps, betas 16). This is **not** the release `index.csv` was built from. Best sampling is `ds=5` (length-matches 58% of KIT clips; ds=6 only 9%), giving a **flat ~1–4 cm position residual** — the residual is a body-model artifact (only the SMPL-H model is on disk, no SMPL-X body model, so I can't reproduce positions exactly), *not* a timing drift (error is flat across frames, so frames are aligned). Twist axis-angles are model-independent, so KIT twist at ds=5 is usable **but only for the ~58% of KIT clips whose durations still align**; the other ~42% are tail-misaligned because the SMPL-X re-release trimmed clips differently than `index.csv` expects.

**Bottom line for the injection:** you have exact, frame-aligned real twist for **CMU+EKUT base clips (3264 rows, ×2 with mirrors)** and approximate twist for **~2688 KIT base clips**. To get real twist for the majority of HumanML3D (BMLmovi, Eyes_Japan, HDM05, etc.), the corresponding **AMASS SMPL+H `_poses.npz` sub-datasets must be downloaded/untarred first** (some tarballs exist, e.g. `CMU.tar.bz2`, but the 15 missing datasets are not present at all). humanact12 twist is irrecoverable from this pipeline regardless.

---

# APPENDIX B — PoC detail

The PoC now works end-to-end for **all joints including arms**. Here is the complete result.

## VERDICT: The twist-injection PoC WORKS end-to-end (numerically verified on 2 real clips)

Twist is a gauge → positions are preserved **exactly**, real AMASS twist can be injected as ground truth, it is smooth (no v2-style jitter), and it dynamically matches the true SMPL twist to <1°. The task's core hypothesis is confirmed; I also found the one non-obvious correction needed to make it robust.

PoC script: `/tmp/claude-3565/-iridisfs-scratch-ts1v23-workspace-noKslot-clean/3b618710-38c6-4bfc-a19a-8bb6fa615348/scratchpad/twist_poc.py` (numpy-only, CPU, read-only inputs).

## Clips used (chosen because their SMPL-H `_poses.npz` still exists)
- `000043` ← `CMU/106/106_05_poses.npz`, female, 60 fps, slice[0:77]
- `000058` ← `CMU/132/132_02_poses.npz`, male, 120 fps, slice[0:191]

## Frame alignment — SOLVED and PROVEN (this was the risky part)
- The AMASS→HumanML3D map: `bdata['poses']` → downsample `[::int(fps/20)]` → dataset-trim (CMU/EKUT: none) → `[start_frame:end_frame]` → `process_file` drops the **last** frame. So `T_263 = (end-start) - 1`. Verified: 000043 77→76, 000058 191→190, both exact.
- **Independent proof of temporal alignment:** SMPL-derived joint-flexion angles vs HumanML3D (263-recovered) angles agree at **lag 0** to **0.02–0.67°** across both knees and both elbows. Alignment is exact, no off-by-one.
- IMPORTANT resolved gotcha: the current `datasets/amass/` has both SMPL-H `_poses.npz` (2431 files, the format that actually built HumanML3D) and newer SMPL-X `_stageii.npz` (4232). Many clips (e.g. 000000/KIT) now have **only** `_stageii`, which is re-fit at a **different fps/frame-count** (KIT 120fps/582 vs the old 100fps that HumanML3D used) → does NOT align. You must use `_poses.npz` clips.

## The sign/handedness question — resolved analytically and empirically
HumanML3D applies `trans_matrix` (swap Y/Z, det −1) then `data[...,0]*=-1` (det −1); the composite is a **proper rotation** (det = **+1.0000**, confirmed). Plus `uniform_skeleton` uses twist-free swing IK so bone **directions** are preserved. Net SMPL→HumanML3D is one constant proper rotation `A` per clip (recovered from bone directions with **0.18–0.50° residual**). Twist angles are rotation-invariant, so twist extracted in SMPL rotation space transfers with correct sign — no reflection issue.

## The 5 verified criteria (both clips)

**(a) Positions preserved — EXACT (twist is a gauge).** FK(inject)-vs-RIC == FK(v3a)-vs-RIC to the digit (14.026 mm clip1 / 1.512 mm clip2 — this is the pre-existing shared-skeleton FK floor, NOT from twist). FK(inject) vs FK(v3a) gauge diff = **0.13 / 0.27 µm** (numerical zero). Injecting any per-joint roll about the current bone moves nothing.

**(b) Round-trip — EXACT.** Injected roll re-measured from the packed ch3:9 tokens = injected value to **4×10⁻⁶ °**. The rot6d pack/unpack and sibling-share are bit-faithful.

**(c) Twist is now real and dynamically correct.** Injected stored-local rotation frame-to-frame geodesic matches the **true SMPL local rotation** to **<1°** for essentially all 15 single-child joints (clip2 all 0.05–0.98°; clip1 similar, forearm worst at 2.3°), and always ≤ v3a. v3a alone misses this twist. Anatomical local twist magnitudes are physical: thighs ~2–5°, shins ~3–11°, upper arms ~7–17°, forearms ~5–11° mean (forearm/upper-arm largest = correct pronation/supination). vs v3a ≈ 0 added twist.

**(d) Smoothness — identical to real SMPL, nothing like v2's jitter.** Injected stored-local geodesic frame-delta == SMPL's to 2 decimals (e.g. forearm 15.1==15.1, 3.8==3.8), and both are **~5–20°/frame vs v2 random-Kabsch's ~90–180°/frame**. Real AMASS twist does NOT reintroduce v3-era jitter, exactly as hypothesized.

## The one non-obvious correction I had to discover (fail-loud)
The task's proposed `WR_new = R_twist(θ, current_bone) @ v3a_swing` is correct, BUT the naive θ (twist-relative-to-swing, whether from the SMPL world or local rotation) is **wrong**:
- It leaks the body's global yaw into near-vertical bones (θ_world hits 80–170°), and
- it breaks at the arms, where 000021-rest (arms down) vs SMPL-rest (T-pose arms out) differ by ~180°, making the roll projection numerically unstable (jitter spikes to 130°).

The correct θ is the roll of the **rest-aligned** true world rotation: (1) recover clip rotation `A` from bone dirs; (2) per bone, a constant `C_p` = rotation-average of `A·G_smpl` vs the v3a swing (the 000021↔SMPL rest-pose offset, empirically ~5–50° for legs/torso, ~80–94° for arms — matching the T-pose/arms-down difference); (3) `φ_p(t)` = signed twist of `A·G_smpl·C_p` relative to the v3a swing about the actual current bone; (4) inject `R(φ_p, current_bone) @ v3a_swing`. This makes the residual small everywhere → stable → arms fixed, positions still exact.

## Productionization caveats (honest)
1. `C_p` is estimated **per-clip** here (it even varies clip-to-clip: L_thigh 7.5° vs 40.4°), which flatters the per-clip numbers. For skinning consistency a real build must compute **one global per-bone `C_p`** from the SMPL-rest ↔ 000021-rest poses and reuse it; the per-clip residual may then be marginally higher than the <1° shown.
2. Only clips with surviving `_poses.npz` are directly convertible; the `_stageii`-only clips need either the original `_poses` release or re-alignment by fps/frame-count.
3. The constant `C_p` per-bone offset is a rest-pose convention (absorbed by bind pose), not an error — but it must be applied consistently to be animal-consistent.

Bottom line: the method is feasible and verified — keep v3a's swing (exact positions), inject the rest-aligned real AMASS twist as a roll about each bone; positions stay exact, twist becomes real, smooth, and faithful to SMPL to sub-degree.

---

# APPENDIX C — pitfalls detail

I have thorough empirical grounding now. Here is the DIMENSION C analysis.

---

# DIMENSION C — Pitfalls & Edge Cases for Real-Twist Injection into Human AnyTop

All findings below are verified against the live data (`HM=/iridisfs/scratch/ts1v23/workspace/motion-latent-diffusion-main`, converter `/scratch/ts1v23/workspace/noKslot_clean/scripts/convert_humanml3d_to_anytop13.py`, built sets `data/humanml3d_anytop13_{v2,v3a}_shared_reencoded` = 24838 train+val / 4388 heldout = 29226 clips).

Two discoveries reshape the whole plan and were NOT in the VERIFIED FACTS: (a) the local AMASS is the newer **SMPL-X "stageii"** release, frame-misaligned to what built the dataset; (b) our clip ids are **renumbered 0–29225** and do NOT map to `index.csv`. These drive the blocking list.

## BLOCKING (must solve before any correct full-pipeline run)

### B1 — Local AMASS ≠ the release that built our data → per-frame gather is WRONG
The task assumed SMPL-H `poses` (T,156) with key `mocap_framerate`. The local files are SMPL-X stageii: `poses` (T,**165**), keys `root_orient/pose_body/pose_hand/pose_jaw/pose_eye`, framerate key is `mocap_frame_rate` (underscore), filenames `*_stageii.npz` (not `*_poses.npz`). Body layout is still safe (`poses[:,:3]==root_orient`, `poses[:,3:66]==pose_body`, verified max-diff 0.0), so joints 0–21 extract cleanly.
The killer is framerate/frame-count: for `KIT/3/kick_high_left02` the local file is 120fps → `poses[::int(120/20)=6]` = **97 frames**, but `index.csv` row 0 is `[0:117]` and the 263 vec has 116 frames. The mismatch is systematic and exactly the 5/6 ratio (97/117, 69/83, 76/91 all = 100/120): same raw take, but the local KIT is labelled **120fps vs the original 100fps**. CMU is mixed (120 and 60 both present), EKUT is 100, and ~4/196 KIT files have NO framerate key (`-1`).
Consequence: the naive `poses[::int(fps/20)]` gather produces the wrong number of twist frames, misaligned to v3a. Blocking.
Mitigation: do NOT trust the file's `mocap_frame_rate`. Reconcile per clip against the KNOWN target — reconstruct the exact raw indices the original used so the count matches (existing v3a length +1) and index `start:end`, OR time-resample θ (it's smooth, so linear-in-time interp onto the 263 frame times is safe) then drop the last frame (263 = frames−1). Add a fail-loud check: if raw-frame count can't be reconciled to the target length, skip that clip (don't emit silent garbage). Cleaner alternative worth a decision: rebuild positions+θ from the SAME SMPL-X forward pass for the covered subset, guaranteeing alignment by construction (cost: positions won't byte-match existing v3a, and only ~54% coverage — see B3).

### B2 — Our clip ids are renumbered; source lookup is NOT identity
Our ids are contiguous `000000–029225`, zero `M`-prefixed. `index.csv`/`amass_annotations.json` use standard `000000–014615` + `M000000–M014615`. I confirmed our `000000..014612` = base, `014613..029225` = mirror with a clean **mirror(i)=i+14613** offset (7/7 caption L/R-swap probes pass). BUT our base ids are NOT the standard ids: identity holds for id 0/100/1000/5000 then BREAKS by 10000 (our `010000`="walks counterclockwise" vs annotations `010000`="hands on knees"). Counts disagree three ways: `index.csv`=14616, `amass_annotations.json` base=14614, our base=14613. So `our_id==std_id` would fetch the WRONG AMASS source for most clips → catastrophically wrong twist. Blocking.
Mitigation: reconstruct our_id→source map by matching our `texts/{id}.txt` caption-set against `amass_annotations.json` (its value has `path`, the AMASS source, plus all captions). Validate uniqueness (caption-set is effectively unique; fail-loud on no/ambiguous match). `amass_annotations.json` is the authoritative source-of-truth here, not `index.csv` (the empty `pose_data/` means the index-path intermediates are gone anyway).

### B3 — Only 54% of clips are locally recoverable; humanact12 has NO twist source
Local AMASS = CMU/EKUT/KIT only (2088+349+4287 = 6724 npz). Mapping index rows → local files (with `_poses`→`_stageii` normalization): **7909/14616 = 54.1%** recoverable (CMU 2913/2913, EKUT 351/351, KIT 4645/4648). Everything else is absent: BMLmovi(1839), Eyes_Japan(1465), MPI_HDM05(771), BioMotionLab_NTroje(373), ACCAD(277), DFaust(135), MPI_Limits(132), MPI_mosh(122), Transitions(110), TotalCapture(74), SFU(68), BMLhandball(67), HumanEva(50), SSM(30), and **humanact12(1191)**. Applied to our clips (base+mirror), ~46% (~13.4k clips) get no real twist locally. humanact12 is worse: it is positions-only in HumanML3D and is NOT an AMASS dataset — there is no SMPL twist to recover for it from anywhere local. No other AMASS copy exists on the filesystem (checked). Blocking / decision point for the user.
Mitigation: (a) download the missing AMASS subsets from amass.is.tue.mpg.de (they exist, this is just data acquisition) for near-full coverage; (b) humanact12 stays zero-twist (v3a) or is sourced from the original HumanAct12 SMPL fits separately; (c) ship a mixed dataset with a per-clip `twist_valid` flag and keep v3a swing for uncovered clips. This is a user resource/scope decision (which subsets to fetch), not something to silently pick.

### B4 — Mirroring: 50% of clips need sign-correct twist, or half the data is corrupted
Mirrors are `base+14613`. HumanML3D mirror (raw_pose_processing cell 18 + `swap_left_right`) = X-flip + L/R joint-index swap, chains `left=[1,4,7,10,13,16,18,20]` ↔ `right=[2,5,8,11,14,17,19,21]`. Under reflection, axial twist REVERSES sign. So mirror θ[j'] = −θ[swap(j)]. Getting the sign wrong systematically corrupts half the dataset (and it's position-invisible, so it won't show up in any FK/RIC check — only in skinning/mesh). Blocking to get right.
Mitigation: derive mirror θ by (L/R-swap + negate); VALIDATE the sign empirically on a few clips against a physically-mirrored SMPL pose (SMPL flip: L/R-swap axis-angles + negate components 1,2 of each), re-extracting θ and checking it equals the swap+negate rule. Do this once; then apply the rule to all mirrors (cheaper than re-flipping SMPL for 14k clips).

## MINOR / already-handled / confirmed-safe

### M1 — Multi-child joints (pelvis 0, spine3 9): do NOT inject; keep v3a Kabsch
These have 3 non-collinear children (pelvis→hips+spine1; spine3→neck+both collars), which fully constrain the 3-DOF rotation, so their twist is already geometrically recovered by a well-conditioned Kabsch (the v3-jitter came only from rank-1 SINGLE-child degeneracy, not these). Injecting twist here would double-count and is ill-defined (no single bone axis). Torso twist is therefore already represented. Keep v3a's Kabsch at 0 and 9; inject ONLY at the single-child parents. Minor (correctly handled by keeping v3a there).

### M2 — Rest-agnostic injection is sound AND position-preserving (the KEY INSIGHT holds)
Confirmed by construction: the converter computes each `WR[j]` INDEPENDENTLY from world positions (`_swing_batch` maps rest bone → true current bone). Post-multiplying `WR_new[p] = R_twist(θ_p, a_p) @ S_p` about the current axis `a_p` leaves `WR_new[p]@u_rest = v_cur` unchanged (twist about `a_p` fixes `a_p`), so the single child stays put; and because each downstream joint's swing is recomputed from ITS own true axis (not inherited by frame composition), ALL descendant positions are preserved exactly — while every joint carries its real twist. This is precisely why it sidesteps the naive-gather failure: we discard SMPL's rest-dependent swing and transfer only the frame-invariant scalar twist. Requirement: extract θ_p as SMPL's twist BEYOND its own minimal-arc swing (i.e. `R_rel = WR_smpl[p]·S_smpl_p^{-1}`, take its rotation-about-bone angle) — a scalar, so rest/frame-agnostic on re-application. Not blocking; this is the correct method.

### M3 — Twist continuity: less dangerous than the v3-jitter lesson implies
rot6d is continuous across the ±180° wrap at the MATRIX level (`R(θ)` and `R(θ±2π)` identical; near ±180° nearly identical), and real SMPL twist is smooth — unlike v2's LAPACK sign-flips which were true matrix-level jumps. So a wrapped θ does not reintroduce jitter in the learned representation. Still recommend `np.unwrap(θ_j)` over time for cleanliness and a guard on the one genuine singularity: an anti-parallel bone (swing≈180°), where both swing and twist-axis are ill-defined — reuse `_swing_batch`'s deterministic anti-parallel branch. Minor.

### M4 — betas/shape: second-order only
The twist AXIS comes from current v3a positions (betas-independent, since `uniform_skeleton` IK→FK preserves bone DIRECTIONS and only rescales lengths). θ's zero-reference (SMPL rest bone direction) is betas-dependent only to second order; use the SMPL template (mean-shape) rest for a betas-free extraction, or the per-clip betas (present: 16-dim SMPL-X, `gender` present). Note the local `body_models` are SMPL-H (10 betas), but we need rotation-only FK (no mesh), so betas/body-model aren't required for the twist scalar at all. Minor.

### M5 — Joint order / correspondence: verified exact, no risk
`JOINT_NAMES` == SMPL first-22 kinematic order; converter's `PARENTS` assertion matches `[-1,0,0,0,1,2,3,4,5,6,7,8,9,9,9,12,13,14,16,17,18,19]`; `poses[:,3:66]` == `pose_body` (joints 1–21, same order) verified. The single-child parents needing injection are 1,2,3,4,5,6,7,8,10,11,12,13,14,15,16,17,18,19,20,21 minus leaves — i.e. index θ by the parent joint whose child count is 1. Not a risk.

### M6 — Frame-handling edge cases (mostly moot for the local 54%)
- `263 = frames−1`: drop the last θ frame — must replicate.
- Dataset-specific head-trims (Eyes_Japan 3s, HDM05 3s, TotalCapture 1s, MPI_Limits 1s, Transitions 0.5s) and the `start:end` slice + `data[...,0]*=-1` X-flip apply per raw_pose_processing. Importantly, NONE of the trim-rule subsets are local — CMU/EKUT/KIT get no head-trim, only `start:end` + downsample + drop-last — so alignment for the recoverable 54% is simpler. 
- humanact12 special-case (no X-flip, no `start:end` slice) is moot (no twist source).
- KIT files missing `mocap_frame_rate` (`-1`): fail-loud/skip.

## Priority order
Blocking: **B1** (frame alignment / stageii-vs-original) and **B2** (id→source remap) are prerequisites for ANY correct injection — without them you inject wrong-length or wrong-clip twist silently. **B4** (mirror sign) is co-equal blocking since it's 50% of data and position-invisible. **B3** (coverage) is a scope/resource decision that bounds how much of the dataset can be done at all.
Minor/confirmed: **M1** (keep Kabsch at pelvis/spine3), **M2** (method is sound), **M3/M4/M5/M6** are safe-with-standard-handling.

Net: the twist-injection method itself is correct (M2 confirms the gauge insight and position-preservation), and the multi-child/continuity/shape/joint-order concerns are minor. The real work and real risk are all in DATA PLUMBING against a mismatched local AMASS release: frame reconciliation (B1), id remapping via `amass_annotations.json` (B2), sign-correct mirroring (B4), and the hard 54% local coverage ceiling with humanact12 unrecoverable (B3).
