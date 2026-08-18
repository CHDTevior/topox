# Implementation Plan — v3 human rot6d converter (cheap validation BEFORE any GPU)

**Author:** Claude (Opus 4.8) · **Date:** 2026-06-30T03:32Z · **Status:** DRAFT for reviewer (codex gpt-5.5 xhigh) audit before any code runs.
**Rev 2 (2026-06-30, user round-2 feedback — verdict SOUND-WITH-CHANGES):** (1) Gates A–I now **per-candidate** pass/fail, ≥1 pass → smoke (not "both must pass"); (2) v3b **frame-0 seed = canonical zero-twist swing** (not LAPACK basis); (3) Gate C now **same-subset relative** comparison; (4) **stratified subset sampling** rule (worst-accel / top-fk-mismatch / 180°-stress motions / length spread / random locomotion); (5) v3a **rest-frame alignment = two reported numbers** (cosine + geodesic delta), claim "= native" only if both align; (6) matched-smoke **per-encoding mean/std recompute + raw-space metrics**; (7) Gate D reports **both 6D-accel AND SO(3)-geodesic-accel**.
**Rev 2.1 (2026-06-30, user GO + 2 clarifications):** v3a 180° guard rewritten as a **3-way branch on `u·v`** (`+1`→identity no-op, `−1`→reference-axis fallback, else shortest-arc) — fixes the `|u·v|>1-eps` over-catch; Gate-D2 **SO(3) geodesic angular-accel given a fixed `omega=log(Rᵀ R')` formula** shared by v2/v3. User verdict: **GO** — codex-audit plan → implement v3a/v3b (default v2) → codex-audit code → subset Gates A-I → backup; NO full re-convert/retrain yet.
**Basis:** user feedback on `handoff/20260630_031328_human_rot6d_encoding_analysis.md`. This doc operationalizes that feedback into a concrete, gated implementation. It does **NOT** touch the running smooth VQVAE (that continues to ep300 independently as the fk_smooth baseline).

---

## 0. What the user decided (the contract this implements)

- **Root cause accepted:** human FK jitter is an **ill-conditioned rotation TARGET** at the converter, not a VQVAE loss-weight problem. v2 is **geometrically correct** (GT rot6d→FK reproduces RIC) but writes a **gauge freedom (the under-constrained axial twist) into the training target**.
- **Direction accepted:** fix it at the **converter** (a v3 re-encode), **not** by adding more loss. `fk_smooth` is confirmed insufficient (plateaus) — that plateau IS the evidence to move to the converter.
- **Validate cheaply FIRST (no GPU):** build a **small-subset v3**, export **TWO candidates in parallel** (`v3a_zero_swing`, `v3b_svd_continuity`), gate them, then a **matched short smoke**, and only then a full re-convert + retrain.
- **NO target multiplier promised.** Do not claim 1×/2×/3×. The achievable human-FK smoothness is **measured** (Gate D + matched smoke), not asserted.
- **Wording discipline:** write "**≈0 within probe tolerance**", never "0.0000 mm absolute".

---

## 1. Scope & invariants (what must NOT change — Gate-A enforced)

Only the computation of **non-root human `ch3:9`** inside `reencode_rot6d` (`scripts/convert_humanml3d_to_anytop13.py:99-127`) changes. Everything else is held fixed and gate-checked identical:

- `ch0:3` (RIC position), `ch9:12` (velocity), `ch12` (contact), **root `ch3:9`** (yaw facing) — untouched.
- The **per-parent, sibling-shared** convention; the FK recovery (`recover_from_bvh_rot_np` / `src/data/anytop_rot6d_fk.py`); the 13ch format; the VQVAE arch + the shared n8192 RVQ codebook; the animal data — all unchanged.
- **No GPU is touched until Gates A–I pass on the small subset.**

---

## 2. The two candidate encodings (both keep per-parent/sibling-shared + the unchanged downstream)

The change is localized to the `WR[p]` world-rotation loop (`:111-117`). Today: `WR[p] = _kabsch_batch(offsets[cs] → P[:,cs]-P[:,p])` (`:117`); for a **single-child** parent this Kabsch is rank-1 → swing determined, **twist = LAPACK's arbitrary completion** → frame-to-frame flips.

### v3a — deterministic zero-twist swing for single-child joints (+ reference-axis guard)  ·  **PRIMARY first version (v3a)**
- **Branch on `len(CHILDREN[p])`:**
  - `== 1` (the **15 single-child parents, codex-confirmed from `PARENTS`: `[1,2,3,4,5,6,7,8,12,13,14,16,17,18,19]`** — arm/leg/spine chains): set `WR[p][t]` = the **minimal-arc swing** mapping rest bone `u = normalize(offsets[c])` onto current `v[t] = normalize(P[t,c]-P[t,p])`, with **canonical twist (0 about the bone axis)**. Impl: `qbetween(u, v)`→matrix, or `axis = normalize(u×v)`, `angle = arccos(clamp(u·v, -1, 1))`.
  - `> 1` (the **2 multi-child parents: pelvis/root `0` + spine3 `9`**): **keep `_kabsch_batch` UNCHANGED** (v3a does not touch multi-child). (leaves `[10,11,15,20,21]` have no children → no WR.)
- The existing local-rotation + sibling-share + token code (`:118-127`) is reused verbatim — v3a only changes how `WR[p]` is built for single-child `p`.
- **Rank 1b reference-axis guard — REQUIRED, not optional polish — 3-way branch on `dot = u·v` (do NOT use `|u·v|>1-eps`, which wrongly catches the +1 no-op case too):**
  ```
  dot = clip(u·v, -1, 1)
  if   dot >  1 - eps:   R = I                      # bone direction ~unchanged -> identity / no-op
  elif dot < -1 + eps:   R = reference-axis / parallel-transport fallback   # ~180° reversal: u×v ill-defined
  else:                  R = shortest-arc swing (axis=normalize(u×v), angle=arccos(dot))
  ```
  **Why the split matters (the must-fix):** at `dot≈+1` the bone barely moved, so the correct rotation is **identity** — propagating the previous frame's frame here would drag a stale twist gauge in and *create* drift. Only `dot≈−1` is the genuinely unstable case (cross-product axis undefined), and only there do we fall back to a **temporally-continuous reference axis** (parallel-transport the previous frame's frame, re-aligned to `v[t]` by the minimal update). **Explicitly test on clips with deep knee/elbow flexion, kicks, sitting** — do not assume 180° is rare for HumanML3D.
- **Rest-frame alignment check — report TWO numbers (before any "= native qbetween" claim):**
  - **(i)** cosine between AnyTop shared-000021 rest offset directions and the rest bone directions HumanML3D's native `qbetween` uses (per single-child joint);
  - **(ii)** geodesic delta between v3a's single-child swing rotation and HumanML3D's native `ch3:9` rotation (per joint, over the subset).
  - If (i)≈1 and (ii)≈0 → v3a *is* the native qbetween swing; if not, **that's fine** but the report must say "**canonical zero-twist swing**", NOT "equals native". The fix's validity does not depend on the equality — it's a labeling/claim-accuracy guard.
- Batchable over T except the guard fix-up at singular frames (sequential there).

### v3b — temporal SVD-basis continuity  ·  **PARALLEL CANDIDATE (v3b)**
- Keep the **full-rank Kabsch for ALL parents**, but resolve the **degenerate singular-vector subspace** (rank-1 twist for single-child; near-rank-2 for spine3's near-coplanar fan) by **temporal continuity** rather than LAPACK's per-frame tie-break.
- Impl: a `_kabsch_batch_continuous` that walks frames **sequentially**: per `t`, after SVD, detect degeneracy (σ-ratio threshold); for the degenerate subspace pick the orthonormal completion **closest to frame `t-1`'s** (align / Gram-Schmidt against the determined directions), preserving the det-correction. Kills the flips.
- **frame-0 SEED — REQUIRED, don't inherit LAPACK's arbitrary basis:** initialize the degenerate subspace at `t=0` from the **canonical zero-twist swing** (the v3a value), NOT from LAPACK's raw Kabsch completion. Otherwise the whole sequence smoothly inherits an **arbitrary gauge** — better than per-frame flipping, but not the cleanest canonical twist. With a canonical seed + nearest-to-prev propagation, v3b ≈ v3a on single-child joints and additionally smooths the spine3 near-degenerate fan.
- **DETERMINISM spec — REQUIRED so the continuity itself doesn't reintroduce discontinuities (codex):**
  - **single-child parents:** do NOT rely on SVD threshold-detection at all — use the **direction-anchored construction** (the v3a swing) directly. SVD-threshold for a rank-1 case is fragile; the anchored swing is unambiguous.
  - **multi-child (spine3) only:** apply the continuity resolution, and pin down (a) **singular-vector SIGN** (fix sign by aligning to prev frame, SVD sign is arbitrary), (b) **σ-ORDERING swaps** between frames (track by continuity, not by magnitude order, when σ's are close — else the basis label swaps), (c) **det-correction AFTER** the continuity choice (re-apply `det=+1` fix to the continuity-selected basis, not before), (d) **threshold HYSTERESIS** (two thresholds to enter/exit the degenerate branch, so frames hovering at the σ-ratio boundary don't flip between branches).
- **Covers single-child AND spine3 in one mechanism** (more general than v3a; higher implementation risk; sequential-over-T so slower — fine for an offline converter). Net: on single-child v3b == v3a (both anchored); v3b's only delta vs v3a is the spine3 continuity.

Both candidates write only non-root `ch3:9`; both flow through the **same unchanged** `rotq`/token/FK path.

---

## 3. Build artifacts (no GPU)

- Add a **v3 mode** to the converter (e.g. `--rot6d_mode {v2,v3a_zero_swing,v3b_svd_continuity}` selecting the `WR[p]` builder), or sibling `reencode_rot6d_v3a` / `reencode_rot6d_v3b`. Default stays **v2** (canonical dataset untouched).
- A **small-subset driver**: for ~100–200 human clips, emit v2, v3a, v3b `ch3:9` (carrying `ch0:3/9:12/12/root` from v2) into a **scratch dir** — never overwrite `data/animo4d_anytop_clean_L4_safe_plus_humanml3d`.
- **Subset SAMPLING — stratified, NOT pure-random** (random risks missing the cases that trigger the 180° fallback / worst jitter). The subset MUST include: (a) the clips with the **highest v2 rot6d 2nd-diff accel**; (b) the **top v2 `gt_fk_mismatch`** clips; (c) motions with **deep knee/elbow flexion, kicks, sitting, jumps, sharp turns** (the 180°-swing stress cases for the 1b guard); (d) a spread of **short / medium / long** clip lengths; (e) a handful of **ordinary locomotion** clips as the easy baseline. Record the chosen motion-ids so v2-vs-v3 comparisons (Gate C, §5) are on the **identical** clips.
- Probes: extend `scripts/_scan_fk_mismatch_full.py` (Gate C) + a per-joint **rot6d 2nd-diff accel** probe (Gate D) + a GT-FK / GT-RIC render (Gate E).

---

## 4. GATES (small subset, NO GPU) — evaluated PER-CANDIDATE (NOT "both must pass")

- **A — parity (fail-loud):** `ch0:3`, `ch9:12`, `ch12`, **root `ch3:9`** byte-identical to v2; ONLY non-root `ch3:9` changed. Abort if anything else moved.
- **B — validity:** non-root `ch3:9` finite; 6D→matrix orthonormal (`detR ≈ +1`) within tolerance.
- **C — FK-floor not worse (SAME-SUBSET relative comparison, NOT a full-v2 absolute threshold):** compute `gt_fk_mismatch` (GT rot6d→FK vs RIC) for v2 AND the candidate **on the identical subset clips**, and require: (i) `candidate_fk_floor ≤ v2_fk_floor + tol` on that subset; (ii) at **single-child** joints the **FK POSITION** delta v3-vs-v2 is **≈ 0** (codex-probed ≈1.7e-7 m) — NOTE this is POSITION invariance, NOT token invariance: the local rot6d TOKENS legitimately CHANGE (that IS the fix); the recovered FK positions cancel back. So Gate C checks FK positions, not raw ch3:9 values; (iii) overall **mean / p90 / p95 not worse than the SAME-SUBSET v2** (do NOT compare a possibly-hard/easy subset against the full-dataset ~6.8 mm / p95 ~2.3 % — that gives false pass/fail either way). Report all three.
- **D — conditioning improved (the direct proof) — report TWO accel metrics, not just 6D:**
  - **(D1) rot6d 2nd-diff accel** (matches the training `rot` 6D-L1) — single-child **CLEARLY DOWN** from same-subset v2 (v2 ≈ 2.32 → much lower).
  - **(D2) SO(3) geodesic angular accel — FIXED definition (v2 and v3 use the IDENTICAL formula; don't let the implementer improvise):**
    ```
    R_t      = rotation matrix from ch3:9 (6D -> matrix, same as FK)
    omega_t  = log( R_t^T @ R_{t+1} )     # SO(3) angular velocity as axis-angle vector (rad)
    accel_t  = omega_t - omega_{t-1}
    metric   = masked mean || accel_t ||  (over valid joints/frames)
    ```
    Must ALSO drop. **Rationale:** 6D-accel falling while SO(3)-accel does not = only the representation *coordinates* got smoother, not the actual rotation → NOT a real win. (A stricter tangent-transport variant is possible but unnecessary for v1; the key requirement is ONE fixed definition shared by v2 and v3.)
  - **Report the measured numbers; promise no target multiple.** Report **per-token** (so spine3 tokens 12–14 residual is visible → drives §7's deferral decision). Report the **DISTRIBUTION** (mean / median / p95 / p99 / frac>10 / frac>30), NOT just the mean, and give **D2 in BOTH rad and deg** (the gate computes rad; the reference baselines below are in deg).
  - **REFERENCE BASELINES (user-measured, 2026-06-30, single-child parent tokens; encode as constants in the gate report) — these turn Gate D from a weak "v3<v2" into a real "did it reach the animal regime":**

    | metric (single-child) | **animal L4_safe** (250 clips, 10707 tok) | **human v2 (bad)** (250 clips, 3750 tok) |
    |---|---|---|
    | axial-twist 2nd-diff median (deg/frame²) | **0.13** | **115** |
    | axial-twist 2nd-diff p95 | **1.5** | **134** |
    | frac > 10 / > 30 | 0.36% / 0.21% | 100% / 99.95% |
    | SO(3) angular-accel median (deg) | **0.9** | **129** |
    | SO(3) angular-accel p95 | **6.6** | **172** |

    Key finding: animal twist is **NOT random** — its main distribution is smooth/continuous (median 0.13°, only 0.36% > 10°), its few outliers LOCAL/distal (toe/claw, climb/turn cleaning-residue). human v2 is **systematic whole-body** noise (shoulder→elbow→wrist, spine1→2→3; 100% of tokens > 10°). So v3's job = pull human single-child rot targets back into the **animal-continuous regime**, NOT chase a nonexistent "perfect twist".
  - **Gate-D PASS criterion (strengthened, per user):** v3 must move human single-child SO(3)/twist accel OUT of the random-gauge regime toward animal: concretely **SO(3) median (deg) ≥ ~10× below the human-v2 ~129° level** (clearly << 100°, into the animal-continuous band) AND **frac>30 collapses from ~100% to a small %**. Aspirational (not required): within animal p95 (6.6°). NOT required: hitting animal median. Smoke already lands v3 at D2=0.066 rad = **3.8°** (within animal p95 6.6°) — the full-subset distribution must confirm. If v3 only nudges ~129°→~120°, that's a **FAIL** (root cause not fixed). **Unit reconciliation:** gate D2 (rad) × 57.3 = deg; smoke v2 D2 = 2.23 rad = 128° ≈ user's human-v2 median 129° (same metric → the gate measures the real thing).
- **E — visual not bad (binding, CV-primacy):** render GT-FK + GT-RIC from the re-encoded rotations on a few clips → pose correct + smooth; **deliver to user** for the visual verdict.
- **F — sibling-shared storage preserved (codex):** all children of each parent carry the **identical** `ch3:9` 6D (the per-parent convention the FK + animal codebook assume). Assert per parent.
- **G — telescoping / reindex round-trip (codex — FK uses last-child-wins for multi-child, so position-floor alone can hide breakage):** compose the FK local rotations from the v3 tokens and verify the **recovered global `WR[p]` matches the intended builder `WR[p]`** (per joint, within tolerance) — at single-child AND multi-child parents.
- **H — root recovery unaffected (codex):** root `ch3:9` yaw byte-identical (Gate A) is necessary but NOT sufficient — the FK ROOT local rotation is sourced from CHILD tokens (`anytop_rot6d_fk.py:147`), which v3 changes. Verify the recovered root/global frame + root-trajectory FK is unchanged within tolerance.
- **I — dataset/loader hygiene for the scratch v3 (codex):** `cond['offsets']` in the v3 output equals the offsets the converter used (FK self-consistency); and the loader's normalized-cond cache (`src/data/anytop_dataset.py:554`, `_cond_normalized_J*.pkl`) is absent/regenerated for the scratch v3 dir — a stale v2 cache must NOT be silently reused.

> Gate B alone proves smoothness, not correctness — it only discriminates a good encode **ANDed with Gate A+C** (a smooth-but-wrong encode passes B alone). The five gates A–I are evaluated **together** for a single candidate.
>
> **Per-candidate decision (NOT "both must pass"):** run Gates A–I **independently for each of v3a and v3b**. A candidate that fails any gate is **eliminated**; a candidate that passes all of them advances. **As long as ≥1 candidate passes, proceed to the matched short smoke (§5) with the passing candidate(s).** v3b is the more complex / higher-risk candidate — a v3b failure must NOT block v3a (or vice-versa). If both pass, both go to §5 and are compared there.

---

## 5. Matched short smoke (ONLY after Gates A–I pass) — the matched control

- Train **two short smokes, SAME config / epochs / seed, `fk_smooth` held FIXED** (OFF preferred, or identical weight — held constant so the encode's effect is not confounded with the loss):
  - (i) old **v2** subset (baseline control — required, not optional);
  - (ii) the **v3** subset(s).
- **NORMALIZATION (mean/std) — recompute per-encoding, never cross-use (v3 changes the `ch3:9` distribution):** v2 and v3 smokes use the **same motion-ids / same split**, but **each recomputes `cond['mean']`/`cond['std']` from its OWN raw motion** (v2 stats for the v2 smoke, v3 stats for the v3 smoke). Do NOT train v3 with v2 stats (or vice-versa) — that silently corrupts the comparison. Verify the stats provenance in each smoke's log.
- Measure recon-FK human jitter with `scripts/_fk_mpjpe_diag.py --continuous` (encoder-vs-RVQ split). The **v3-minus-v2 delta at matched training** is the real signal. **Report RAW-space (de-normalized, meters) metrics too**, not just normalized loss — so a normalization change cannot mask (or fake) an improvement.
- Pick the winning candidate (`v3a` vs `v3b`) by FK-floor + rot-accel (D1+D2) + smoke jitter + **visual**.

---

## 6. Full re-convert + retrain — ONLY if the matched smoke's VISUAL confirms

- Commit GPU to a full human re-convert + token-cache rebuild + full VQVAE retrain **only if** the matched short smoke shows human FK jitter **clearly improved on the GT-vs-recon GIF** (per QA-primacy; not on metrics alone).
- Note the coupling cost: a re-encode changes the rotation manifold → the n8192 **token cache must be rebuilt** and any backbone trained on old tokens **will not transfer**.

---

## 7. Explicitly deferred / not-now (per user priorities)

- **spine3 regularized Kabsch:** NOT a v3a blocker. If Gate-D per-token shows spine3 (tokens 12–14) is the dominant residual after v3a, then either rely on v3b (svd-continuity already covers it) or add a **bounded** regularized fit (position-error budget Gate-C-checked). **Leave pelvis alone** (well-conditioned).
- **Rank 3 rotation-space accel loss:** only on a **v3 clean target**, never on v2 (on v2 it would chase the noisy twist). Optional complement post-v3.
- **Rank 4 geodesic / leverage-weighted rot loss:** polish, low priority, after the converter.
- **Rank 5 position-route-for-human:** demo/stopgap only — cross-topology transfer needs rotations, so this is not the research fix.
- **Rank 6 latent/token temporal coherence:** only if post-v3 residual is shown RVQ-dominated.
- **Rank 7/8 (post-filter / more `w_fk_smooth`):** not mainline; `fk_smooth` already shown insufficient.

---

## 8. No target promise (explicit)

Do **not** commit to 1×/2×/3×. The achievable human-FK smoothness is set by **recon-error × leverage** and is **unknown until measured** — by Gate D (target conditioning), the matched smoke (§5), and the physical-floor simulation (analysis-doc Gate C, fixed per critic to inject calibrated noise onto **re-encoded twist-zero GT**, not onto v2). Report measured numbers only. (Animals reach ~1.1× despite proximal leverage, so a low floor is **not** precluded — but it is not promised either.)

---

## 8.5 Execution order (the operational sequence)

1. **codex audits THIS plan.**
2. **Implement v3a + v3b** in the converter; **default stays v2** (canonical dataset untouched).
3. **codex audits the implementation.**
4. **Run subset Gates A–I** (no GPU): each candidate **independent** pass/fail; **≥1 pass → proceed** (§4).
5. **Matched short smoke:** v2 subset baseline + each passing v3 candidate; **same seed / epochs / loss config / fk_smooth fixed**, **per-encoding mean/std**, raw-space metrics (§5).
6. **Human eyes on the GT-vs-recon-FK GIF** (binding, CV-primacy).
7. **Only after the visual confirms improvement → full re-convert + token-cache rebuild + full retrain** (§6).

## 9. Iron-rule compliance

- All converter + probe code changes **MUST pass codex (gpt-5.5 xhigh)** before running.
- **No GPU** until Gates A–I pass; the smoke uses a **matched v2 control**; **visual QA is the binding gate** at §5 and §6.
- Does **not** touch the running smooth VQVAE (independent; continues to ep300; remains the fk_smooth baseline for comparison).
- v2 dataset preserved (v3 writes to a new scratch dir) — full rollback intact.

**Key code anchors:** `scripts/convert_humanml3d_to_anytop13.py` — `_kabsch_batch:84-91` (the rank-1 twist null-space), `reencode_rot6d:99-127` (the `WR[p]` loop `:111-117` = the only edit site; sibling-share `:125-126`). `src/data/anytop_rot6d_fk.py:147` (FK back-fills parent rotation from a child token → last-child-wins for multi-child; relevant only if a later step changes multi-child). `src/models/graph_salad/losses.py:764` (`fk_smooth` — NOT extended here). Probes: `scripts/_scan_fk_mismatch_full.py` (Gate C), `scripts/_fk_mpjpe_diag.py --continuous` (§5).
