# Implementation Plan — v4 human real-twist recovery (`rot6d_mode='v4'`), cheap validation BEFORE any GPU

**Author:** Claude (Opus 4.8) · **Date:** 2026-07-01T23:57Z · **Status:** Rev 2 (SOUND-WITH-CHANGES fixes applied) — DRAFT for user REVIEW, then **PLAN-codex** (gpt-5.5 xhigh, fresh thread) adversarial audit of this plan/design **before any implementation begins**; a second **IMPLEMENTATION-codex** gate (gpt-5.5 xhigh, fresh thread) then adversarially reviews each written script/converter change **before it is run at scale / before the full build**.
**Basis:** `handoff/20260701_231029_human_twist_recovery_from_amass_plan.md` (PoC-proven method) + `handoff/20260630_190042_human_rot6d_data_encoding_lessons.md` (why v3a zeroed twist) + `handoff/20260630_033233_human_rot6d_v3_converter_implementation.md` (v3 gate discipline this reuses). All converter line-numbers below re-verified against the live source this session.
**One-line:** v4 = v3a swing (positions byte-exact) **+** the REAL AMASS SMPL axial twist injected as a roll about each current bone axis. Twist is a gauge → positions don't move; real AMASS twist is smooth → no return of the v2 jitter.

---

## 1. TL;DR + scope boundary

**⚠ SCOPE REFRAME (read first — defines what v4 is and is NOT).** v4 "real-twist preservation" is a **mesh / SKINNING-FACING human data branch**. It is an **additive parallel branch**, NOT a replacement for the skeletal-motion mainline. **v3a stays the skeletal T2M / VQVAE MAINLINE** (twist-zero is optimal there — it fixed the FK jitter, lessons doc §1). v4 **must NOT directly replace v3a** as the tokenizer/backbone training data — the only path by which v4 could ever become a mainline is if a **separate, independent tokenizer + backbone are later trained from scratch on v4 tokens** (a re-encode changes the rotation manifold → no token-cache/codebook transfer from v3a; §9). Until and unless that separate stack exists, v4 is a standalone mesh-facing dataset consumed by skinning, running in parallel to — never in place of — the v3a skeletal pipeline.

**Method is PoC-supported** on 2 real CMU clips (`scratchpad/twist_poc.py`) **under a per-clip `C_p`**; the production `C_p` (§3.3) and all headline numbers below are **TODO-verify** (§3.3, §7) — "confirmed" would overclaim what 2 clips under a per-clip `C_p` establish. PoC numbers (per-clip `C_p`): positions preserved to ~0.13/0.27 µm, injected twist matches SMPL <1° for the 15 single-child joints, smoothness identical to SMPL (~5–20°/frame) vs v2's ~90–180°/frame. **[TODO-verify: these numbers were NOT re-executed this pass — they require two ~87 MB SMPL-H body-model reads over iridisfs; the PoC computation was line-verified but not re-run. Re-run `twist_poc.py` on a compute node before trusting them as build gates.]**

**Why v4 exists:** v3a zeroed axial twist, which was correct + optimal for skeletal motion / topology-transfer / text-gen (it fixed the FK jitter, lessons doc §1). But **skinning/mesh** needs the real twist (forearm pronation, upper-arm roll). v4 restores it **without** disturbing anything v3a fixed.

**Additive-only invariant (Karpathy simplicity):** v4 must leave `v2` and `v3a` output **byte-identical**. v4 is a new `rot6d_mode` branch + one new twist-source kwarg + one new twist-extraction helper. No existing path changes.

**IN the first deliverable (fully local, verifiable today):**
- The **CMU + EKUT exact-alignment subset** — the only local AMASS with canonical SMPL-H `_poses.npz` (frame-exact reproduction of the built data, confirmed in the PoC).
- Coverage anchor: CMU = 2913 index rows, EKUT = 351 index rows recoverable-with-file (`coverage_census` anchor). **CMU+EKUT exact = 3264 index rows.** After L/R mirrors this is **3264 × 2 = 6528 clips (~6.5k)** — NOT ~13k. **[Corrects the plan doc's "~13k" arithmetic error: 3264×2=6528; ~13k would be the full local 7909 rows ×2 which includes APPROX KIT.]** The exact per-clip count after the id-remap may differ by a few (our base = 14613 vs index 14616, +3 skew) and **must be recomputed via the remap, not assumed** (`three_way_count_disagreement` anchor).
- Per-bone `C_p` calibration (default per-clip rest-based, §3.3), `twist_valid` provenance flags, all no-GPU gates (§7), codex review, user visual sign-off.

**OUT of the first deliverable (gated on user decisions):**
- **KIT** (4645 rows) — only SMPL-X `_stageii.npz` on disk (wrong release, 120 fps, key `mocap_frame_rate`) → **APPROX only**, needs fps reconciliation and per-clip acceptance (§4/B1).
- **The 15 missing AMASS subsets** (BMLmovi, Eyes_Japan, MPI_HDM05, …) — need download from amass.is.tue.mpg.de (§4/B3 table).
- **humanact12** (1191 rows) — NOT in the AMASS v4 first stage (not AMASS-sourced), but **NOT irrecoverable in principle**. Its real SMPL θ is recoverable via ACTOR/PHSPD's `humanact12poses.pkl` (poses `[N,72]`, 24-joint axis-angle, frame-aligned to the same Action2Motion clips; assert `||FK-joints3D|| < 1e-10`). This is a **SEPARATE small pipeline** (fit quality noisier than AMASS MoSh; needs clip-order + coord-convention reconciliation), out of scope for AMASS v4 stage 1.

Nothing v4 does touches the running VQVAE, the v2/v3a canonical datasets, the n8192 codebook, or the animal data.

---

## 2. Background — the twist-loss chain and the gauge insight

**Where twist dies (verified chain, plan doc §1):**
1. **AMASS SMPL → positions.** `raw_pose_processing.ipynb` cell-8 builds the SMPL body and keeps only `body.Jtr` (52 joint *positions*). The 3-DOF-per-joint axis-angle rotations — which carry axial twist — are discarded and never written to `pose_data/`. **Twist-loss point #1 (fatal).**
2. **positions → 263.** `motion_process.py`/`skeleton.py inverse_kinematics_np` uses `qbetween_np(u,v)` = minimal-arc **swing** → axial twist ≡ 0. **Twist-loss point #2 (structural)** — even if #1 had kept rotations, IK would zero the twist.
3. **263 → AnyTop 13ch.** `reencode_rot6d(...,'v3a')` uses `_swing_batch` for single-child joints = twist-free by construction. Twist stays 0.

**The gauge insight (why injection preserves positions):** rotating a bone about its own axis does not move its child (the child offset is parallel to the twist axis). And the converter recomputes **every** joint's world rotation `WR[p]` independently from that joint's own true world positions (`convert_humanml3d_to_anytop13.py:252-265`), **not** by frame-to-frame composition — so injecting a roll at parent `p` about `p`'s current bone axis leaves `p`'s child and **all** descendants exactly in place. Verified in the PoC: FK(inject) vs FK(v3a) = 0.13/0.27 µm (numerical zero). This is exactly the property that made the arbitrary v2 twist a *valid GT but unlearnable target* (lessons doc §3): the twist is a gauge that cancels along FK. v4 replaces the arbitrary gauge value with the **real, smooth** SMPL value — still position-invariant, but now a coherent learnable target usable for skinning.

See `handoff/20260630_190042_human_rot6d_data_encoding_lessons.md` for the full root-cause arc.

---

## 3. Design — `rot6d_mode='v4'`

### 3.1 Converter hook (single site, additive)

The only edit site is the `WR[p]` loop in `reencode_rot6d` (`convert_humanml3d_to_anytop13.py:252-265`). Current single-child branch:

```python
elif len(cs) == 1:                                   # single-child: v3a==v3b -> swing
    WR[p] = _swing_batch(U[0], V[:, 0, :])
```

v4 post-processes **only** the single-child branch, and **only when a twist source is supplied**:

```python
# v4: v3a swing + injected AMASS roll about the CURRENT world bone axis
elif len(cs) == 1 and rot6d_mode == "v4":
    S_p = _swing_batch(U[0], V[:, 0, :])             # [T,3,3] v3a swing (unchanged)
    if twist_phi is not None and twist_valid_p:      # phi_p [T] supplied for this joint
        a_p = _normalize(V[:, 0, :])                 # [T,3] CURRENT world bone axis
        WR[p] = _axis_angle_roll(a_p, twist_phi[p]) @ S_p   # LEFT-multiply
    else:
        WR[p] = S_p                                  # fallback == v3a (twist_valid False)
```

- **Signature change (additive):** `reencode_rot6d(raw13, P, offsets, rot6d_mode="v2", return_wr=False, twist_phi=None, twist_valid=None)`. Add `"v4"` to the L248-249 validation set (`("v2","v3a","v3b","v4")`) and to the CLI `--rot6d_mode` choices (L431) + both call sites (L492 build, L543 return_wr). **[Anchor discrepancy noted: `reencode_rot6d` today has NO twist input param and NO twist-extraction helper — both MUST be added. This is the gap the plan doc glossed.]**
- **In-scope vars at L260-261** (all VERIFIED present): `p` (parent idx), `cs=CHILDREN[p]`, `U=offsets[cs]` (`U[0]`=rest bone), `V=P[:,cs,:]-P[:,p:p+1,:]` (`V[:,0,:]`=current world bone vector per frame → axis `a_p=normalize(V[:,0,:])`), and the `S_p` swing about to be assigned. `twist_phi`/`twist_valid` come in as the new kwargs (a `[J,T]` array + `[J]` bool, or `None`).
- **Single-child only.** Inject at exactly `_SINGLE_CHILD_PARENTS = [1,2,3,4,5,6,7,8,12,13,14,16,17,18,19]` (15 joints, VERIFIED). **Keep Kabsch at multi-child `[0,9]`** (pelvis, spine3): 3 non-collinear children already constrain all 3 DOF there — injecting would double-count and has no single bone axis (M1). Leaves `[10,11,15,20,21]` keep identity WR, never injected.
- **Branch off v3a, never v2.** The v2 branch (L258-259) runs Kabsch for ALL parents including single-child; a v4 branch under v2 would be overwritten by degenerate Kabsch (open-risk item). v4 is its own top-level `rot6d_mode`, and the single-child post-mult happens only when `rot6d_mode=="v4"`.
- **No packing change.** Local rot `rotq[i]=WR[gp]ᵀ·WR[i]` (L266-271) and token `new[:,j,3:9]=_mat_to_6d(rotq[PARENTS[j]])` (L272-274) are reused verbatim — twist enters only via `WR[p]`, so siblings still share the parent rot and FK is unaffected (VERIFIED anchor).

### 3.2 Swing-twist decomposition + injection formula (EXACT — must be pinned; only lives in the PoC today)

For joint `p`, given the rest-aligned true world rotation `target_p(t)` (§3.3) and the v3a swing `S_p(t)`, both `[T,3,3]`, and the current world bone axis `a_p(t)=normalize(V[:,0,:])`:

```
resid = target_p @ S_pᵀ                       # [T,3,3], a rotation ~about a_p
w     = 0.5 * [ resid[2,1]-resid[1,2],
                resid[0,2]-resid[2,0],        # = sin(phi) * axis
                resid[1,0]-resid[0,1] ]
sin_s = w · a_p                               # per frame
cos_t = (trace(resid) - 1) / 2
phi_p = atan2(sin_s, clip(cos_t, -1, 1))      # signed roll about a_p, radians
phi_p = np.unwrap(phi_p)                       # temporal continuity (M3)
```

Injection (LEFT-multiply, world-frame roll about the world bone axis — **not** a body-frame post-multiply; the plan's word "post-multiplying" is imprecise, the math is left-mult and FK-preservation holds because `a_p` is the world axis `R_twist` fixes):

```
R_twist(phi_p, a_p) = Rodrigues(a_p * phi_p)   # [T,3,3]
WR_new[p]           = R_twist(phi_p, a_p) @ S_p
```

Helpers (Karpathy-simplicity — do NOT add a near-duplicate of what exists):
- **Reuse the existing `_axis_angle` (L116)** for the Rodrigues roll instead of adding `_axis_angle_roll`. It is already Rodrigues built on the batched `_skew` (L107); it can be batched over T by passing `axis=[T,3]` and reshaping the angle to `[T,1,1]` (`np.sin(angle)[...,None,None]*K + ...`). Only if that reshape proves awkward in-place should a thin `_axis_angle_roll` wrapper be added.
- **ADD `_signed_roll(resid, axis)`** (numpy, batched), mirroring the PoC `signed_twist`.
- **ADD `_normalize(v) = v / (np.linalg.norm(v, axis=-1, keepdims=True) + 1e-12)`** — used by both §3.1 (`a_p`) and here; it is NOT among the existing helpers, so it must be listed as an addition (or inlined).
Keep them tiny, in the converter, matching existing `_swing_batch` style. (The `_axis_angle_roll`/`_normalize` names in the §3.1 snippet refer to these: `_axis_angle_roll` = `_axis_angle` batched over T, `_normalize` = the new helper above.)

**Guard (REQUIRED, PoC lacks it):** `atan2/trace` is exact only when `resid` is (near) a pure rotation about `a_p`. Compute the off-axis residual `err = geodesic( resid, R_twist(phi_p,a_p) )`; if `err > tol` the extraction is mixing swing and twist → set `twist_valid_p=False` for that clip/joint and fall back to v3a. Fail-loud, do not silently emit a corrupted roll (open-risk item). **`tol` is NOT free to guess ("a few degrees" is a placeholder):** it is co-dependent with the `C_p` residual (a large `C_p` error inflates `err` and would reject legitimate clips, conflating `C_p` error with genuine swing/twist mixing). Pin `tol` **after** the `C_p` calibration (§8 step 5), from the measured off-axis-residual distribution, jointly with the G-match-phi (φ) threshold; report both `tol` and the resulting clip/joint rejection rate as part of the go/no-go.

### 3.3 rest-based per-bone offset `C_p` (why rest-aligned roll is required, else arms break)

The **naive** roll (twist of SMPL relative to its own swing) is WRONG: it leaks the body's global yaw into near-vertical bones (θ_world 80–170°) and blows up at the arms, where the AnyTop `000021` rest (arms-down) vs the SMPL rest (T-pose, arms-out) differ by ~180° → jitter spikes to ~130° (plan §2, VERIFIED reasoning). The correct roll is the twist of the **rest-aligned** true world rotation:

1. Recover the clip's constant proper rotation `A` (SMPL frame → HumanML3D frame) by Kabsch on **unit bone directions** (`P_smpl` vs `P_hml`). Residual 0.18–0.50°. **[Note: det(A)=+1 is Kabsch-imposed in the PoC via the SVD det-sign correction — it is NOT independent confirmation of properness. The independent evidence is the 0.18–0.50° bone-direction residual + the analytic argument that `trans_matrix`(det −1) ∘ x-flip(det −1) = det +1.]**
2. `Gtrue = A · G_smpl` (SMPL world rotations mapped into the HumanML3D frame).
3. `C_p` = per-bone constant offset absorbing the `000021`-rest ↔ SMPL-rest pose gap (empirically ~5–50° legs/torso, ~80–94° arms — matching T-pose vs arms-down).
4. `target_p = Gtrue[:,p] · C_p`; feed to §3.2.

**⚠ LOAD-BEARING GAP — the `C_p` choice is a METHOD-VALIDITY go/no-go, not a threshold tweak.** The PoC computed `C_p` **per-clip** as a frame-average of `Gtrueᵀ·WR_v3a` (`twist_poc.py:236`), a motion-dependent quantity that varied a lot clip-to-clip (L_thigh 7.5° vs 40.4°). Two sharper problems, both understated by "the <1° ceiling may need loosening":

1. **Absolute-twist loss (per-clip frame-average `C_p`).** A frame-average `C_p` absorbs each clip's **mean** twist into `C_p`, so the extracted φ is twist **relative to that clip's mean** — the absolute pronation/roll offset (exactly what skinning needs) is **discarded**. The reassuring "<1° matches SMPL" therefore measures only the **time-varying** part, not the absolute twist.
2. **Mass-rejection (frozen/motion-independent `C_p`).** A single frozen global `C_p` (or any motion-independent `C_p`) **re-injects** the clip-mean twist into `resid`, so `resid = target·Sᵀ` is no longer near-pure-about-axis. If that residual is large — the L_thigh 7.5°↔40.4° per-clip spread strongly suggests it will be — the §3.2 off-axis guard **trips and mass-rejects clips**, and v4-exact collapses to mostly v3a fallback, delivering ~no real twist.

So the naive per-clip average loses the absolute offset, and a naive frozen global one risks mass rejection. **[TODO-verify: measure the off-axis-residual distribution under the chosen `C_p` before trusting any fidelity number; the <1° gate ceiling is not the real risk — mass rejection / absolute-offset loss are.]**

**MAINLINE (the only extraction default): per-clip (per-unique-betas) REST-based `C_p`.** Compute `C_p` per clip, **motion-independently**, from that clip's own SMPL rest joints (betas-aware) as the closed-form `000021`-rest ↔ SMPL-rest offset — no frame-average, no calibration-set generalization assumption. This is both simpler in principle (closed-form from the rest pose already computed per-clip for `A` recovery, §5) and more correct: it **preserves absolute twist** and **absorbs subject betas variation**, avoiding both failure modes above. Clips sharing a subject share a betas vector, so `C_p` can be cached per unique betas — still per-clip in effect, never global.

**A single frozen GLOBAL `C_p` is DIAGNOSTIC / FAILURE-FALLBACK ONLY — never the mainline.** The old frozen calibration-average form (`proj_so3( mean over the CMU+EKUT calibration clips of Gtrue[:,p]ᵀ · WR_v3a[p] )`) may be computed as a diagnostic to characterize the residual distribution, or used as a last-resort fallback for a clip whose betas/rest cannot be recovered, but it is motion-dependent, loses the absolute offset, and MUST NOT be presented or used as the default extraction. Both the mainline per-clip `C_p` and the diagnostic global `C_p` are **TODO-verify**. G-match-phi's go/no-go criterion is the **fraction of single-child joints whose off-axis residual stays below the §3.2 guard tol** under the mainline per-clip `C_p` — NOT just an angle ceiling.

---

## 4. Data plumbing (the real work)

### B2 — id remap (HIGHEST RISK): our-clip → caption → `amass_annotations.json` → source npz

Our AnyTop human ids are **renumbered** and do **NOT** equal HumanML3D `index.csv`/`amass_annotations` ids. Identity holds at 0/5000 but BREAKS by 10000 (our `010000`="walks counterclockwise" vs annotations `010000`="hands on knees"). `three_way_count_disagreement`: index.csv=14616, annotations base(numeric)=14614, our base=14613 (VERIFIED). **`source_motion_id` in `motion_texts_by_file.json` is a TRAP** — it == our own renumbered id for all 29226 clips (VERIFIED), NOT a source pointer.

**Exact remap (VERIFIED unique+correct on 4 probes):**
1. our clip `HML3D_Human_{id}.npy` → `motion_texts_by_file.json[...]['captions']`.
2. For each caption, collect the set of numeric `amass_annotations.json` keys whose `annotations[].text` matches.
3. **Intersect** the sets across all captions → require a **unique** key. `annotations[key]['path']` → AMASS npz.
4. **FAIL-LOUD: reject the clip if the intersection is empty OR size>1** (no/ambiguous match) → `twist_valid=NONE`, keep v3a.
5. **Recover the source frame window from `index.csv` (fail-loud) — do NOT assume start=0.** The remapped numeric key == the `index.csv` `new_name` stem (VERIFIED: our000000→ann000000→`index.csv` `000000`=`KIT/3/kick_high_left`; `index.csv` columns = `source_path,start_frame,end_frame,new_name`). Read `start_frame`/`end_frame` from the `index.csv` row whose `new_name` stem == key. **A single AMASS file is segmented into multiple clips with per-segment windows, so `start` is not always 0** — the frame slice in §5 step 2 MUST use these, not `[0:len]`. Assert `basename(source_path)` is consistent with `annotations[key]['path']`; mismatch → fail-loud, `twist_valid=NONE`. (This step is the provenance the plan glossed; without it a wrong start-OFFSET silently injects real-but-wrong-motion twist into correct positions — see G-srcalign, §7.)

Ids: base `000000–014612`; **mirror(base_i) = base_i + 14613** (VERIFIED — NOT the standard +14616/M-prefix). `amass_annotations.json` numeric keys = 14614 over range 0..14615 (2 gaps).

**[TODO-verify: uniqueness was checked on only 4 of 29226 clips. Before relying on the remap, MEASURE how many clips have empty or size>1 caption intersections (generic captions like "a person walks" risk collisions) and report the count. Fail-loud is mandatory; the count tells us how much coverage we actually lose.]**

Locations (VERIFIED): index.csv and amass_annotations.json under `/iridisfs/scratch/ts1v23/workspace/motion-latent-diffusion-main/datasets/humanml3d/`.

### B3 — coverage census (recoverable index rows over 14616; VERIFIED `coverage_census`)

| tier | subsets | index rows | % of 14616 | twist quality | in v4 first deliverable? |
|---|---|---|---|---|---|
| EXACT | CMU (2913) + EKUT (351) | **3264** | 22.3% | real, frame-exact SMPL-H `_poses.npz` | **YES** |
| APPROX | KIT (`_stageii`, SMPL-X, 120 fps) | 4645 | 31.8% | fps-reconciled, ~58% length-align | NO (user gate, B1) |
| SEPARATE-pipeline | humanact12 | 1191 | 8.1% | recoverable via ACTOR/PHSPD `humanact12poses.pkl` (NOT irrecoverable), separate noisier pipeline | NO — out of AMASS v4 stage 1 |
| MISSING (need download) | 15 subsets: BMLmovi 1839, Eyes_Japan 1465, MPI_HDM05 771, BioMotionLab_NTroje 373, ACCAD 277, DFaust 135, MPI_Limits 132, MPI_mosh 122, Transitions 110, TotalCapture 74, SFU 68, BMLhandball 67, HumanEva 50, SSM 30 (=5513 listed + ~3 remainder) | **~5516** | **~37.7%** | real once downloaded | NO (user decision) |

Local recoverable-with-file total = **7909/14616 = 54.11%**. (Distinct from npz-files-on-disk = 6724 — do not conflate rows vs files.) 3 KIT index rows have no on-disk file → skip fail-loud. The "not-yet-covered" bucket = 14616 − 7909(recoverable) − 1191(humanact12) = 5516 (~37.7%); the four buckets sum to 14616 = 100.0%. (The earlier "~6707/~45.9%" was arithmetically impossible — it = 14616−7909 but that quantity already INCLUDES the 1191 humanact12 rows counted in their own row, so the buckets summed to 108.1%.)

**Stage framing (N3) — this is STAGE-1 LOCAL SCOPE, not a permanent ceiling.** Do NOT read the ~37.7% as "missing forever". **Stage 1 = the CMU+EKUT EXACT-LOCAL subset only** (the frame-exact SMPL-H `_poses.npz` we already hold). The ~5516 not-yet-local rows are an **extensible STAGE-2** item: if the AMASS download from amass.is.tue.mpg.de completes, those subsets become real-twist recoverable and fold into v4 with the same pipeline. Frame all coverage numbers as "stage-1 local scope, extensible to full AMASS in stage 2", never as a fixed shortfall.

### B1 — KIT fps handling (APPROX, deferred)

Local KIT is the SMPL-X 2021 re-release: `*_stageii.npz`, `poses` shape (T,165), key `mocap_frame_rate` (underscore) = 120 fps. The build release was SMPL-H 100 fps (`mocap_framerate`, no underscore). The build notebook `try: fps=bdata['mocap_framerate']` **KeyErrors and returns early (fps=0, no save)** on these files → proves on-disk KIT ≠ the release that built the 263 data (VERIFIED). Naive `poses[::int(120/20)=6]` gives the wrong frame count; ds=5 length-matches only **~58%** of KIT **[TODO-verify: the 58% figure was not re-counted this pass]**. Plan for KIT (only if user accepts APPROX): reconcile per-clip to the known target length (== v3a length + 1) by time-resampling the smooth φ, or skip; fail-loud on the ~4 KIT files with framerate −1.

### B4 — mirror sign flip (50% of data, position-invisible)

Mirrors are built by `swap_left_right` (`raw_pose_processing.ipynb:304-320`): `data[...,0]*=-1` (X-reflection) then swap `right_chain=[2,5,8,11,14,17,19,21]` ↔ `left_chain=[1,4,7,10,13,16,18,20]`. This is a **position-only** transform in HumanML3D (asserts last-dim==3); it never had twist. The proposed rule `φ_mirror[j'] = −φ_base[swap(j)]` (twist is a handedness-reversing pseudo-quantity under reflection) is an **analytical hypothesis, NOT a code fact** (VERIFIED — no twist term exists in the mirror code). **Midline coverage:** three injected single-child parents — 3 (spine1), 6 (spine2), 12 (neck) — are on the midline and appear in NEITHER `right_chain` nor `left_chain`, so `swap(j)` for them is otherwise undefined. They **self-map** (`swap(j)=j`) and are **still negated**: `φ_mirror[3]=−φ_base[3]`, `φ_mirror[6]=−φ_base[6]`, `φ_mirror[12]=−φ_base[12]`. With this the swap map covers all 15 injected joints. **Plan for mirrors:** do NOT re-extract from AMASS; take base φ, apply L/R swap + negate. **[TODO-verify: validate the negate sign ONCE against a physically-mirrored SMPL pose before applying to all ~14.6k mirrors — getting it wrong silently corrupts half the dataset and is invisible to any FK/RIC check (only shows in skinning).]**

---

## 5. AMASS twist extraction (per exact clip)

Rotations only — no mesh needed for the scalar twist. **The rest pose is computed PER CLIP (or once per unique betas vector) using that clip's own betas** — NOT a single betas-free mean-shape rest reused across a gender. The per-clip betas-aware rest joints are needed BOTH for `A` recovery AND for the per-clip rest-based `C_p` (§3.3, the mainline): the `000021`-rest ↔ SMPL-rest offset that `C_p` absorbs is subject-shape-dependent, so a betas-free rest would leak shape error into `C_p` and thus into the absolute twist. (Betas array on disk is (16,); the body model uses `betas[:10]`. Cache the rest by unique betas vector to avoid recomputation when many clips share a subject; a frozen global/mean-shape rest is a DIAGNOSTIC/fallback only, never the mainline.)

1. **id → source** via B2. Load the SMPL-H `_poses.npz` (`poses` (T,156), `mocap_framerate`, `betas`, `gender`).
2. **Frame-align to the built clip.** `ds = int(fps/20)` (CMU 60→3, 120→6; EKUT 100→5). **The slicing-vs-downsampling ORDER is NOT proven** and MUST be resolved by the Stage-0 gate (G-slicing, §7) before any build: `index.csv` `start_frame`/`end_frame` may be in ORIGINAL-fps coordinates (→ slice-then-downsample: `poses[start:end][::ds]`) or in DOWNSAMPLED coordinates (→ downsample-then-slice: `poses[::ds][start:end]`). **The PoC cannot distinguish them because it only tested `start=0` clips**, where both orders coincide. Adopt whichever order aligns to the stored v3a RIC positions (G-slicing, on NON-ZERO-start CMU/EKUT clips). `start`/`end` come from `index.csv[key]` (B2 step 5) — **NOT assumed 0**; per-segment windows exist. Then **drop the LAST frame** (`process_file` outputs frames−1). Assert `T_263 == (end-start)-1` (PoC: 000043 77→76, 000058 191→190 — both were `start=0` clips; do NOT generalize start=0). CMU/EKUT get **no head-trim**. Fail-loud on any length mismatch — but note the length assert catches a wrong LENGTH only, NOT a wrong OFFSET or a wrong slice-ORDER; both are caught by G-srcalign / G-slicing (§7). **Without the Stage-0 order gate, twist can be globally frame-SHIFTED: positions may partially pass while skinning breaks silently.** Never emit silent garbage.
3. **Per-joint local rotation.** `root_orient = poses[:,:3]` (joint 0), `pose_body = poses[:,3:66]` (joints 1..21 axis-angle). VERIFIED `poses[:,:3]==root_orient`, `poses[:,3:66]==pose_body` max-diff 0.0.
4. **SMPL→22 joint map = identity on first 22** (drop SMPL 22/23 = hands). `JOINT_NAMES`==SMPL first-22; `PARENTS`==`[-1,0,0,0,1,2,3,4,5,6,7,8,9,9,9,12,13,14,16,17,18,19]` (VERIFIED, asserted in converter).
5. **World rotations.** `Rloc = Rodrigues(local)`; `G[0]=Rloc[0]`, `G[j]=G[PARENTS[j]]@Rloc[j]`.
6. **Rest-align + extract roll** per single-child joint: recover `A` (§3.3.1), `Gtrue=A·G`, apply the **per-clip (per-unique-betas) rest-based `C_p`** computed closed-form from this clip's own betas-aware SMPL rest joints (§3.3, the mainline — NOT a frozen global `C_p`), then §3.2 → `φ_p(t)` + the off-axis guard → `twist_valid_p`. (A single frozen global `C_p` is only ever a diagnostic / failure fallback, never this extraction path.)
7. **Return** `twist_phi [J,T]` (0 at non-single-child rows) + `twist_valid [J]` to `reencode_rot6d(...,'v4', twist_phi=..., twist_valid=...)`.

EKUT format (SMPL-H `_poses.npz` at 100 fps) is claimed but **[TODO-verify: EKUT npz keys/shapes were not dumped this pass — confirm they match CMU's canonical `_poses.npz` before the build.]**

---

## 6. `twist_valid` provenance flags (stored alongside the data)

Every emitted clip carries a per-clip + per-joint provenance so downstream training and skinning know what is real vs zero-fallback:

- **`twist_provenance`** ∈ {`EXACT` (CMU/EKUT `_poses.npz`, frame-exact), `APPROX` (KIT fps-reconciled), `NONE` (remap failed / guard rejected / humanact12 / uncovered)}.
- **Per-joint `twist_valid [J]`** (length 22) bool: `False` at all non-single-child rows, and `False` at any of the 15 single-child joints failing the §3.2 off-axis guard (that joint falls back to v3a swing individually). This is the SAME array shape §5 step 7 returns and the §3.1 branch indexes as `twist_valid[p]` (p∈0..21) — do NOT store a length-15 variant, which would break the `[p]` indexing. (If a compact per-injected-joint view is wanted for storage, keep it as an explicit projection of the `[J]` array onto the 15 injected rows, clearly labelled — the canonical array stays `[J]`.)
- Stored in the clip's sidecar (extend `motion_texts_by_file.json` or a new `twist_manifest.json` in the v4 output dir) — do NOT overwrite v3a. A `NONE` clip is byte-identical to v3a (fallback path), so v4 degrades gracefully to v3a everywhere twist is unavailable.
- **Per-clip provenance record (N1, for debuggability) — store MORE than `twist_valid`/`twist_provenance`.** Each clip's sidecar entry MUST also record: `amass_key` (remapped numeric key), `source_path` (AMASS npz), `start`, `end` (the `index.csv` window actually used), `fps`, `ds`, `C_p_mode` (`per_clip_rest` mainline / `per_unique_betas` / `global_fallback`), `mirror_flag` (base vs mirrored + swap map applied), `off_axis_residual` (per-joint, from the §3.2 guard), `srcalign_mpjpe` (the G-srcalign per-joint mm), and `failure_reason` (empty, or which gate/step tagged it `NONE`). This makes every fallback/rejection traceable to its cause without re-running extraction.

---

## 7. Validation gates (cheap, no-GPU, on the CMU+EKUT subset)

Reuses v3 discipline (`_v3_gate_runner.py` style). Each gate has an explicit PASS criterion + baseline; all fail-loud.

- **G-slicing — slice/downsample ORDER (STAGE-0, MANDATORY, run FIRST of all gates).** Pick several CMU **and** EKUT sample clips **with NON-ZERO `start_frame`** (start=0 clips cannot distinguish the conventions). For each, extract the SMPL segment under BOTH conventions — **slice-then-downsample** (`poses[start:end][::ds]`) and **downsample-then-slice** (`poses[::ds][start:end]`) — FK the mean-shape joints and compare to the stored v3a RIC positions. **PASS bar (non-subjective — matches the thresholds now implemented in `scripts/_v4_framealign_probe.py`):** the winning convention must, on **EVERY** non-zero-start probe clip, (i) cross-correlate at **|lag| ≤ 1 frame**, (ii) **mean flexion-angle correlation > 0.95**, AND (iii) **per-frame similarity-Procrustes MPJPE < 30 mm** between the AMASS-FK joints and the stored v3a joints, AND (iv) be **CLEARLY better than the other convention** (the loser fails at least one of i–iii). Pin that convention for the whole build. If the winner does not clearly dominate — or NEITHER convention meets i–iii — the gate result is **INCONCLUSIVE and conversion MUST NOT proceed** (indicates a deeper index/fps mismatch). Rationale: without this, `index.csv` start/end could be in original-fps vs downsampled-fps coordinates and the whole dataset would be **globally frame-SHIFTED** — positions may partially pass while skinning breaks silently. G-srcalign below then re-checks alignment per-clip under the adopted convention.
- **G-srcalign — source↔clip positional alignment (PRECONDITION, catches OFFSET errors).** BEFORE trusting any extracted twist: FK the AMASS SMPL mean-shape joints for the sliced+downsampled segment (§5 step 2) and require **frame-exact per-joint agreement (per-joint mm)** with the clip's stored v3a RIC positions. This is the **only** gate that catches a wrong start-OFFSET: every twist-fidelity gate below (G-match-phi, G-framealign) compares injected-twist vs SMPL-twist, and BOTH derive from the same (possibly mis-aligned) segment, so they trivially agree at lag 0 even when the segment is offset from the clip's actual motion; the `T_263==(end-start)-1` assert catches wrong LENGTH only. A clip failing G-srcalign → fail-loud, `twist_valid=NONE`, v3a fallback. (Closes the silent-corruption hole: an off-by-N segment injects real-but-wrong-motion twist into correct positions, visible otherwise only in skinning.)
- **G-pos — positions preserved.** FK(v4) vs FK(v3a) gauge diff **< 1 µm** per joint; FK-vs-RIC unchanged vs the v3a shared-skeleton floor. Baseline/PoC: 0.13/0.27 µm gauge, 14.0/1.5 mm floor. **[TODO-verify: re-run on compute node.]**
- **G-roundtrip — WR round-trips.** Decode `WR_new[p]` from the packed ch3:9 tokens, extract its roll relative to the v3a swing, compare to the injected `φ_p` → **< 1e-4°** over all single-child joints/frames. PoC: 4e-6°.
- **G-match-phi — AXIAL ROLL φ agreement (HARD gate).** The **hard** criterion is on the scalar axial roll φ, all computed under the **SAME (mainline per-clip) `C_p`**: the injected φ (as re-extracted from the packed v4 tokens) must agree with (a) the extraction-time φ and (b) the SMPL-derived φ, per single-child joint, within the threshold set in §8 step 5 from the measured off-axis-residual distribution. Reject clips exceeding it. **Do NOT use "full local rotation vs SMPL local rotation < 1°" as a hard gate** — our AnyTop representation is **per-PARENT packing** while SMPL is **per-BONE local**, so the two are not necessarily term-by-term synonymous and a full-matrix diff can flag a benign packing-convention difference as failure. The full local-rotation (geodesic) comparison is **DIAGNOSTIC ONLY** (useful to eyeball gross mis-alignment). PoC (per-clip `C_p`): φ <1° for essentially all 15 joints; that number is TODO-verify and the φ threshold is re-set per §8 step 5.
- **G-nojitter — no return of v2 jitter.** Injected stored-local frame-delta geodesic ≈ SMPL's (~5–20°/frame) and **≪ v2** (~90–180°/frame); stays in the animal-continuous band (lessons doc: animal twist 2nd-diff median 0.13°, p95 1.5°). Enforce a per-frame-delta ceiling. Baseline animal vs human-v2 table in the v3 impl doc §4-D.
- **G-framealign — xcorr lag 0.** Independent cross-correlation of a joint-flexion angle (knee/elbow) between injected and SMPL peaks at **lag 0** (< 1°). PoC: 0.02–0.67° on both knees + both elbows.
- **G-mirror — mirror twist == negated base (HARD gate, N2).** Mirror sign is a **position-INVISIBLE** error (the mirror transform is position-only; a flipped twist sign is invisible to every FK/RIC check and shows up only in skinning). It is therefore a **HARD gate**, not a diagnostic: for a mirrored clip, injected φ must equal `−φ_base[swap(j)]` within tolerance, validated once against a physically-mirrored SMPL pose (§4/B4 hypothesis). **A mirrored clip whose twist sign is NOT validated MUST stay v3a — do NOT inject guessed twist** (tag `twist_valid=NONE`, `mirror_flag` recorded per N1). Half the dataset is mirrors, so an unvalidated sign silently corrupts ~50% of skinning.
- **G-visual — user verdict (BINDING, CV-primacy).** GT-vs-v4 **skinned mesh or bone-axis-arrow overlay** GIF (walking, pronation-heavy, arm-cross clips), multi-frame, side-by-side. Self-check the renderer first on a near-perfect known case. **Deliver to user for the visual verdict — do NOT self-judge or rely on metrics alone** (per the QA-primacy + deliver-to-user rules).

---

## 8. Step-by-step execution plan (each step → verify)

1. **User reviews THIS doc, then PLAN-codex** (gpt-5.5 xhigh, fresh thread) adversarial review of this plan/design **BEFORE any implementation begins**. → verify: user GO on scope (CMU+EKUT first; KIT/downloads deferred) + PLAN-codex PASS.
2. **Subset select** — enumerate CMU+EKUT exact clips via the id-remap (not by id). → verify: every selected clip's remap is unique + `_poses.npz` exists; report the exact count (expect ~3264 base ±few skew) and the # of empty/ambiguous remaps.
3. **id-remap + coverage census tool** (`_v4_id_remap.py`, fail-loud). → verify: 4 known probes reproduce (our 000000→ann000000 kick_high_left, our 010000→ann010002 WalkInCounterClockwiseCircle, etc.); count empty/ambiguous intersections.
4. **AMASS twist extraction** (`_v4_amass_twist.py`, §5, read-only + scratch). → verify on the 2 PoC clips: reproduces `twist_poc.py` φ per joint.
5. **`C_p` calibration** — compute the **per-clip (per-unique-betas) rest-based `C_p`** (the MAINLINE, §3.3; the frozen global `C_p` is diagnostic/fallback only), and measure the off-axis-residual distribution over the CMU+EKUT set. → verify: record the residual distribution; set **BOTH the §3.2 off-axis guard `tol` AND the G-match-phi (φ) threshold** from it; go/no-go = **fraction of single-child joints below the guard tol** (not an angle ceiling), plus the resulting clip/joint rejection rate. **(This closes the load-bearing gap — and specifically checks for the mass-rejection / absolute-offset-loss failure modes of §3.3, not just a ceiling.)**
6. **v4 converter mode** — add the `rot6d_mode='v4'` branch + `twist_phi`/`twist_valid` kwargs + `_signed_roll` + `_normalize` helpers (REUSE existing `_axis_angle` L116, batched over T — do not add a duplicate) + off-axis guard; default stays v2; assert v2/v3a bytes unchanged. → verify: `reencode_rot6d(...,'v2')` and `(...,'v3a')` outputs byte-identical to pre-change on a probe set (additive-only invariant).
7. **Run gates G-slicing → G-srcalign → G-framealign + G-mirror** on the subset (no GPU). → verify: **G-slicing (Stage-0) passes FIRST** to pin the slice/downsample order on non-zero-start clips, then G-srcalign (source↔clip offset) as the precondition; then each remaining gate PASSes at its stated threshold (G-mirror and G-match-phi are HARD gates); fail-loud eliminates offending clips (tagged `twist_valid=NONE`).
8. **IMPLEMENTATION-codex review** (gpt-5.5 xhigh, fresh thread) of each written script/converter change + extraction + gates + remap, **BEFORE it is run at scale / before the full build**. → verify: PASS (iron rule — no full build before IMPLEMENTATION-codex PASS; the v3 review caught 4+ real bugs, expect the same scrutiny here on the guard, unwrap, mirror sign, remap fail-loud, source-frame offset (G-srcalign), and the `C_p` choice). Note: the Stage-0 flow (grounding → write probe script → codex-review the script → run) already conforms to this IMPLEMENTATION-codex gate.
9. **User visual** (G-visual). → verify: user confirms twist looks anatomically right and no jitter returns (binding).
10. **Decision → extend.** Only after 8+9: build the full CMU+EKUT+mirrors v4 dataset with `twist_valid` flags. Then, per user decisions: (a) accept KIT APPROX (B1) → add KIT; (b) download the 15 missing subsets (B3) → extend. humanact12 stays v3a.

---

## 9. Risks + open decisions for the user

- **[DECISION] KIT APPROX?** KIT (4645 rows, 31.8%) is only recoverable approximately (fps reconciliation, ~58% clean align). Accept APPROX twist for KIT, or leave KIT at v3a until the correct 100 fps release is obtained?
- **[DECISION] Downloads?** The 15 missing subsets (~5516 rows, ~38%) need download from amass.is.tue.mpg.de (resource + time). Fetch now, or ship the CMU+EKUT exact ~6.5k first and extend later?
- **[DECISION] Skinning timeline.** v4 is only needed for skinning/mesh — skeletal motion/transfer/text-gen should stay on v3a (twist-zero is optimal there). Confirm v4 is a **separate mesh-facing dataset**, not a replacement for the v3a training data. A re-encode changes the rotation manifold → any codebook/backbone trained for skinning on v4 tokens would need its own token-cache rebuild (does not transfer from v3a).
- **[RISK] `C_p` choice is method-validity** (load-bearing) — see §3.3. A per-clip frame-average `C_p` discards the ABSOLUTE twist (what skinning needs); a frozen global `C_p` risks the off-axis guard MASS-REJECTING clips → collapse to v3a. **Mainline = per-clip (per-unique-betas) rest-based `C_p`; the frozen global `C_p` is diagnostic/fallback only.** Go/no-go = fraction of joints below the guard tol under the mainline `C_p` (not an angle ceiling).
- **[RISK] Source↔clip frame OFFSET** (silent-corruption, position-invisible) — start/end must come from `index.csv[key]` (B2 step 5), not assumed 0. A wrong offset injects real-but-wrong-motion twist into correct positions; the length assert does NOT catch it. **G-srcalign** (§7) is the only gate that does — must pass before trusting any extracted twist.
- **[RISK] Mirror negate sign** (50% of data, position-invisible) — **HARD gate** (G-mirror): a mirrored clip whose twist sign is not validated MUST stay v3a (do NOT inject guessed twist). Validate the negate sign once against a physically-mirrored SMPL pose before trusting the mirror half.
- **[RISK] Remap collisions** — generic captions may yield empty/ambiguous intersections; measured in step 2/3, lost clips fall back to v3a (`twist_valid=NONE`).
- **[RISK] humanact12 (1191) is a SEPARATE pipeline, not irrecoverable** — real SMPL θ recoverable via ACTOR/PHSPD `humanact12poses.pkl` (`[N,72]` 24-joint axis-angle, frame-aligned; assert `||FK-joints3D||<1e-10`), but fit quality is noisier than AMASS MoSh and it needs clip-order + coord-convention reconciliation. Out of scope for AMASS v4 stage 1; stays v3a until that separate small pipeline is built.

---

## 10. Pointers

- **Method plan (PoC-proven):** `handoff/20260701_231029_human_twist_recovery_from_amass_plan.md`.
- **Lessons (why v3a zeroed twist):** `handoff/20260630_190042_human_rot6d_data_encoding_lessons.md`.
- **v3 gate discipline (reused):** `handoff/20260630_033233_human_rot6d_v3_converter_implementation.md`.
- **PoC:** `scratchpad/twist_poc.py` (swing-twist math L61-107; `A`/`C_p`/φ L214-240; inject+repack L243-259; gate metrics L262-328). Numbers: gauge 0.13/0.27 µm, twist<1°, smooth 5–20°/frame, xcorr lag 0 — **all TODO-verify (re-run on compute node).**
- **Converter (edit site):** `scripts/convert_humanml3d_to_anytop13.py` — `reencode_rot6d` L230-277 (hook L260-261; single/multi lists L226-227; `_swing_batch` L137; packing L266-274; `compute_offsets` L316; CLI L431/L492/L543).
- **FK:** `src/data/anytop_rot6d_fk.py` (`recover_from_bvh_rot_np`).
- **Data maps (authoritative):** `/iridisfs/scratch/ts1v23/workspace/motion-latent-diffusion-main/datasets/humanml3d/amass_annotations.json` (caption→source), `index.csv`; AMASS `datasets/amass/motion_data/{CMU,EKUT,KIT}`; mirror code `raw_pose_processing.ipynb:304-320`.
- **Our v3a data (input):** `data/humanml3d_anytop13_v3a_shared_reencoded/` (motions/, motions_heldout/, `motion_texts_by_file.json`, `object_index.csv`).

**Iron-rule compliance:** additive-only (v2/v3a bytes unchanged); no GPU until gates pass; **PLAN-codex** (gpt-5.5 xhigh, fresh thread) audits this plan/design before any implementation begins, then **IMPLEMENTATION-codex** (gpt-5.5 xhigh, fresh thread) reviews each written script/converter change before any full build; visual QA is the binding gate delivered to the user; v3a datasets preserved (v4 writes a new dir) → full rollback intact.

---

## 11. Execution plan — STAGE 0 / STAGE 1 only (for now)

Scoped to what is buildable on local data today; full-AMASS (stage 2) is explicitly deferred. v3a remains the skeletal mainline throughout (§1 scope reframe). **Two codex gates apply: PLAN-codex** (gpt-5.5 xhigh, fresh thread) adversarially reviews this plan/design **before any implementation begins** — it precedes step 1; **IMPLEMENTATION-codex** (gpt-5.5 xhigh, fresh thread) adversarially reviews each written script/converter change **before it is run at scale / before the full build** — it precedes any at-scale conversion (step 3). The Stage-0 flow below (grounding → write probe script → codex-review the script → run) already conforms to the IMPLEMENTATION-codex gate.

1. **id remap + NON-ZERO-start frame-align probe.** Build the our-clip→amass_key remap (§4/B2, fail-loud, measure empty/ambiguous count), and run the **Stage-0 G-slicing probe** on CMU+EKUT clips **with non-zero `start`** to fix the slice/downsample ORDER (§5 step 2, §7 G-slicing). → verify: remap unique on probes; one slicing convention meets the §7 G-slicing PASS bar (|lag| ≤ 1 frame, mean flexion-corr > 0.95, Procrustes MPJPE < 30 mm on EVERY non-zero-start clip) AND clearly dominates the other, pinned for the build; otherwise the gate is INCONCLUSIVE → do NOT proceed.
2. **Per-clip `C_p` calibration.** Compute the mainline **per-clip (per-unique-betas) rest-based `C_p`** (§3.3); measure the off-axis-residual distribution; set the §3.2 guard `tol` and the G-match-phi (φ) threshold from it. → verify: rejection-rate + residual distribution recorded (global `C_p` only as a diagnostic).
3. **CMU+EKUT subset v4 conversion.** Add the `rot6d_mode='v4'` branch + helpers + off-axis guard (v2/v3a bytes unchanged), extract twist, inject, write the v4 dir with the N1 provenance sidecar. → verify: additive-only byte check passes.
4. **Gates.** Run **G-srcalign → G-roundtrip → G-match-phi → G-mirror → G-visual (user)** (plus G-pos/G-nojitter/G-framealign, §7). G-slicing (step 1) and G-srcalign are preconditions; G-mirror and G-match-phi are HARD gates; unvalidated clips fall back to v3a (`twist_valid=NONE`). → verify: each gate PASSes at its threshold; user gives the binding visual verdict.
5. **Only after all gates + user PASS → consider full-AMASS (stage 2) extension.** KIT APPROX, the 15 downloadable subsets, and the separate humanact12 pipeline are stage-2 decisions for the user (§9) — not part of stage 1.
