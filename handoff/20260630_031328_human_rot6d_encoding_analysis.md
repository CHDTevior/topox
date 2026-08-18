# Human rot6d-FK Jitter — Root-Cause Analysis & Re-Encode Decision

**Produced by:** multi-agent research workflow (6 parallel investigations -> synthesis -> adversarial critic), 2026-06-30. All file:line anchors verified against the live tree by the agents.
**Reader guide:** §0 = post-critic corrections (apply these OVER the body). Body = the synthesis. Appendix = the adversarial critic verbatim.

---

## 0. Post-critic revisions (READ FIRST — override the body where they conflict)

The synthesis body below is strong and code-grounded; an adversarial critic (full text in Appendix) flagged the following. Apply these over the body:

1. **The smoothness TARGET is UNKNOWN, not "~2-3x".** Body §4 asserts a ~2-3x ceiling and "1x precluded by topology." The critic is right that this is decided on incomplete evidence: animals reach 1.1x DESPITE comparable proximal leverage (tails / quadruped legs / necks were never measured), so the FK-jitter floor is set by *recon-error x leverage* and the re-encode pulls exactly the recon-error lever -> 1x is NOT proven precluded. **Treat the floor as UNKNOWN until measured by Gate C/D; do not promise a number.**
2. **Add Gate 0 — downstream jitter REQUIREMENT.** The whole effort optimizes a number with no spec for what cross-topology transfer / text-gen actually TOLERATES. Before any GPU: decide whether 7.8x is already acceptable for the consumer, or how tight it must be. Don't retrain to an unspecified target.
3. **A strong candidate the body underweights — TEMPORAL SVD-BASIS CONTINUITY.** Keep the full-rank Kabsch but enforce frame-to-frame continuity of the degenerate singular vectors (kill the LAPACK tie-break flips). ONE mechanism that fixes BOTH the single-child twist (Rank 1) AND the near-degenerate spine3 fan (Rank 2). Evaluate alongside the zero-twist swing.
4. **Rank 3 (rotation-space accel hinge) does NOT help on the CURRENT encoding.** A hinge ceilinged at GT's own (noisy ~2.32) accel cannot push below GT noise; it only becomes well-posed on a RE-ENCODED (clean) target. Drop the "~3-5x on current encoding" claim.
5. **Validation gates need tightening:** Gate B (rot6d-accel) proves smoothness only when ANDed with Gate A (FK fidelity) — a smooth-but-wrong encode passes B alone. Gate C is ill-posed as written (calibrating rotation noise from the POSITION channel is a category mismatch, and it must inject onto re-encoded twist-zero GT or it inherits the very twist noise it claims to exclude) -> treat its number as rough. Gate D needs a MATCHED same-config smoke train on the OLD encoding as baseline.
6. **"Zero-twist swing == native qbetween" assumes AnyTop rest offsets match HumanML3D rest bone directions** — verify rest-frame alignment before relying on "no native-data dependency."
7. **The 73/27 encoder/RVQ split and the 15x/11x/7.8x multipliers mix baselines** (denominators 5.60 vs ~6.8; pre- vs post-fk_smooth). Acceleration NORMS are not linearly decomposable -> treat 73/27 and the "<=27% cap" as rough; hold fk_smooth config FIXED (ideally OFF) when comparing across the re-encode.
8. **Cost the HYBRID option** (model emits position-route positions + analytic-swing rotations DERIVED from those positions) — could give smooth rotations without a full re-convert + retrain; the body dismissed it wholesale under Rank 5.

**What stands:** the ROOT CAUSE (§2 — rank-1 Kabsch twist null-space -> LAPACK discontinuity -> position-invariant gauge that cancels in GT-FK but breaks under independent RVQ snap, leverage-amplified; explicitly NOT a conversion bug) is solid and critic-endorsed. The RECOMMENDED DIRECTION (deterministically re-encode human rotations at the converter, gate cheaply BEFORE the expensive retrain) stands. Only the TARGET number and some validation details are softened per the above.

---

# Body — multi-agent synthesis

All file:line anchors verified against the actual source. The six findings are cross-consistent on mechanism; they diverge on the achievable target (twist-conditioning's "approach animal 1.1×" vs realistic-target's "~2–3× physical floor"), which I surface rather than average. Below is the synthesized document.

---

# Human rot6d→FK Jitter in the Shared-Codebook Graph-VQVAE: Root-Cause Analysis and Re-Encode Decision

**Scope:** Why human rotation reconstruction jitters under FK while animals do not, what the cause mechanically is, and whether/how to fix it by re-encoding human `ch3:9`. The 13ch format, VQVAE architecture, and shared n8192 RVQ codebook are held FIXED; only how human `ch3:9` is *computed in the converter* is on the table.
**Status:** Decision-ready. Synthesizes six independent investigations (encoding-mechanics, native-rotation-path, twist-conditioning, rotation-repr-literature, alternative-fixes, realistic-target), all code-grounded against the live tree.

---

## 1. Problem statement (grounded in the measured data)

The Graph-VQVAE reconstructs AnyTop 13ch motion: `ch0:3` RIC world position, `ch3:9` per-parent 6D rotation, `ch9:12` velocity, `ch12` contact. There are two ways to turn a reconstruction into joint positions: the **POSITION route** (read `ch0:3` directly) and the **FK route** (run forward kinematics on the reconstructed rotations `ch3:9`). Measured per-joint acceleration jitter `||x[t+1]−2x[t]+x[t−1]||` (mm/frame²):

| signal | human | animal |
|---|---|---|
| GT-pos jitter | 5.60 | — (smooth) |
| **GT-FK** (GT rot6d → FK) | **5.60 (smooth)** | smooth |
| recon-POSITION route | 8.29 = **1.48× GT** | ~animal-smooth |
| **recon-FK route** | **43.70 = 7.8× GT** (post-`fk_smooth` plateau) | **1.1× GT** |

Three facts pin the problem precisely:

1. **The jitter is 100% route-specific and species-specific.** Human recon-FK is 7.8× GT; human recon-POSITION is already near-animal-smooth (1.48×); animal recon-FK is 1.1×. So the defect lives entirely in **how the model reconstructs HUMAN rotations**, surfaced only through FK.
2. **GT is smooth.** GT rot6d → FK = 5.60 = GT-pos. The converter's *output* round-trips to smooth positions. **This is not a conversion bug** (expanded in §2).
3. **A temporal-smoothness loss plateaus.** `fk_smooth` (matches recon-FK accel to GT accel, `losses.py:689,768`, `w_fk_smooth=2.0`) cut jitter from ~15× → ~7.8× and then stalled — it cannot reach the animal 1.1×.

A second, finer decomposition (handoff `20260628_211915…:23`, measured pre-`fk_smooth`): human recon-FK jitter **quantized = 102.4 (≈15×)** vs **continuous pre-VQ = 74.3 (≈11×)** → **~73% of the jitter originates in the encoder/decoder, ~27% in the RVQ snap.** (Note the measurement-context split: 15×/11× are the *pre-smoothness-loss* numbers; 7.8× = 43.70 is the *post-`fk_smooth`-plateau* number. They describe the same model family at different loss configs — see §7 measurement caveat. Either way the route/species asymmetry is robust.)

**Goal.** Make human rotations FK-reconstruct as smoothly as animals, primarily via a better human `ch3:9` encoding (re-do the conversion), weighed against alternative fixes — and be honest about whether "animal-equal" is even physically attainable.

---

## 2. Root cause — mechanically precise, code-grounded

### 2.1 How human `ch3:9` is produced today

`scripts/convert_humanml3d_to_anytop13.py`, per clip: `convert_263_to_13` (`:335`) writes HumanML3D's native per-bone rot6d into `ch3:9` (`:138` reads `x[:,67:67+(J-1)*6]`; `:157` stores it) — and then `reencode_rot6d` (`:338`) **overwrites all non-root `ch3:9`**, discarding the native rotation. The re-encode:

- For each parent `p` with children `cs`, computes a single world rotation `WR[p] = _kabsch_batch(U, V)` (`:117`) where `U = offsets[cs]` (rest bones) and `V = P[:,cs]−P[:,p]` (current bones), `P` = official `recover_from_ric` positions. `_kabsch_batch` (`:84–91`) is `H = Σ uᵀv`, SVD, det-correct, `WR = Vtᵀ D Uuᵀ`.
- Decomposes to local rotations `rotq[i] = WR[parent[i]]ᵀ WR[i]` (`:119–123`), column-major 6D (`_mat_to_6d`, `:94–96`), and stores **per-parent, sibling-shared**: every non-root token `j` carries `rotq[parents[j]]` (`:125–126`), so all children of one parent hold the identical 6D. This matches the animal BVH convention exactly.

### 2.2 Why this is CORRECT-but-hard, not a defect — and why GT-FK stays smooth

The topology (`PARENTS`) has only **two** multi-child parents — pelvis→{hips, spine1} and spine3→{neck, collars}. **All 15 other rotation-carrying joints are single-child** (the entire arm, leg, and lower-spine chains).

For a single-child parent, `H = u⊗v` is **rank-1**: SVD gives σ₁=|u||v| and σ₂=σ₃=0. The first singular pair pins the **swing** (the 2-DOF rotation carrying the rest bone onto the current bone — fully determined by positions). The remaining singular vectors — the **twist** about the bone axis (the 3rd DOF) — are an **arbitrary orthonormal completion** chosen by LAPACK's tie-break among the degenerate zero singular values, plus a reflection-fix `D[2,2]=sign(det)` taken on a `det≈0` quantity. **This twist is not a continuous function of `v(t)`; it flips frame-to-frame.**

This is the textbook **rotation→position information loss** (HybrIK, CVPR 2021/2023): positions determine swing analytically but **carry zero information about axial twist**, because a bone's offset lies *on* its own twist axis and rotating about it does not move the child. The under-constrained DOF is intrinsic to recovering rotation from position — confirmed by Zhou (CVPR 2019, 6D is the right *format*) and Geist (2024, *"changing the loss does not fix discontinuities"* — a representation/determinacy problem cannot be cured downstream).

**Measured (probes on real clips):** at single-child joints the twist 2nd-difference jitter is **≈109 deg/frame² (≈1.5 rad/frame², near the random ceiling)** vs **≈0.5 deg/frame² at the well-determined multi-child joints — a 200–1000× gap.** Expressed as rot6d-target acceleration: single-child tokens **2.32** under the current Kabsch encode vs **0.033** under a deterministic swing — **~70× noisier**, up to **145–184×** on the worst joints.

**Why GT-FK is nonetheless smooth (5.60).** The arbitrary twist is a **gauge DOF that cancels along the chain**: `rotq[child] = WR[p]ᵀ WR[child]` exactly undoes `WR[p]`'s twist (`anytop_rot6d_fk.py:107–108,149` telescope the locals back to the independent `WR[i]`), and the child's offset is parallel to the bone the twist rotates about → position-invariant. **Verified directly:** replacing the SVD twist with zero-twist changes FK positions by **exactly 0.0000 mm**. The noise is purely in the *representation*, not the geometry — which is exactly why GT-FK jitter = GT-pos jitter.

**Therefore the current encoding is "correct" (lossless round-trip of GT to smooth positions, in the animal-matching per-parent convention) but "hard" (the learning target is ill-conditioned).** This is the distinction the task asks for: the *intrinsic* part (FK-floor, §2.4) is not a defect; the *avoidable* part (filling the twist null-space with LAPACK's discontinuous tie-break instead of a deterministic value) is the true, fixable encoding defect.

### 2.3 Why the MODEL jitters (and why `fk_smooth` plateaus)

Three mechanisms turn the position-invariant target noise back into position jitter on the model's output:

1. **The target is unlearnable noise.** The twist channels at the 15 single-child joints have no temporal structure (~per-frame random). `rot` is a naive per-element L1 (`losses.py:560`) on a noisy target → the encoder/RVQ/decoder can only emit a smoothed/averaged approximation; capacity is spent fitting noise.
2. **Quantization breaks the exact cancellation.** GT smoothness requires `WR[p]`'s twist and the compensating term in `rotq[child]` to be *perfectly anti-correlated*. RVQ snaps each joint's token **independently**, destroying that fine anti-correlation. The leftover uncancelled twist error rotates the whole descendant sub-chain. (This is the ~27% RVQ contribution.)
3. **Long-limb leverage amplifies it.** An uncancelled twist at the shoulder/hip swings the long distal segment through an arc ∝ remaining-limb-length × Δθ. Animals escape on both counts: their `ch3:9` is **native BVH rotation** (coherent, smooth twist — never position-derived), and their segments are short (low leverage).

`fk_smooth` is a *position-space* accel-match: its gradient to the rotation channels passes through the FK Jacobian, which has a **near-flat direction exactly along the under-constrained twist**. So it cannot pin twist, and the rot-MSE target it would otherwise lean on is itself noise. It damps *leaked* jitter (15×→7.8×) but cannot supply a clean twist target — **the noise floor is baked into the encoded `ch3:9`.** This plateau is the fingerprint of the encoding defect.

### 2.4 The FK-floor (~6.8 mm) is a SEPARATE, non-jitter issue

A single rigid `WR[p]` can fit *all* of `p`'s child bones only if the fan is rigid. Animals are exactly 0 (one BVH rig). Humans articulate *within* the pelvis and spine3 fans (hip-line vs spine bend; neck vs collars), so the rank-3 Kabsch leaves a small residual — the only place a residual appears (single-child = 0.0000). Accumulated, this is the ~6.8 mm / `gt_fk_mismatch` ≈0.46% mean / p95 2.3% **static-accuracy** floor. It is temporally **smooth** (jitter ~0.005) → **not the jitter source**, and intrinsic to the rigid-fan/sibling-shared convention. Any single-child re-encode leaves it unchanged.

### 2.5 Convergent verdict (R7 — surface, don't average)

Four of six investigations (encoding-mechanics, native-rotation-path, twist-conditioning, alternative-fixes) independently converge: **the dominant jitter driver is the rank-1 SVD twist null-space at the 15 single-child limb joints, amplified by long-limb FK leverage.** The literature dimension frames the same thing as the HybrIK swing/twist info-loss and prescribes a *representation* fix over a smoothing loss.

**One correction the literature needs (do not propagate the misconception):** the literature recommendation "restore the *native* twist" assumes HumanML3D's native rotation carries real anatomical twist. **It does not.** HumanML3D's native `ch3:9` is produced by `inverse_kinematics_np(..., smooth_forward=True)` using `qbetween` — the **shortest-arc swing, twist≡0 by construction** (`mld/.../skeleton.py:84–101`, `quaternion.py:387–397`). Real 3-DOF twist exists only upstream in AMASS SMPL pose, *before* retarget/floor/face-Z+/mirroring. So "use native" and "use a deterministic zero-twist swing" are the **same fix** for single-child joints — and the in-place swing computation is the lower-risk way to obtain it (no native-data plumbing, no index/rest/mirror alignment).

---

## 3. Candidate solutions — RANKED

All keep 13ch / arch / shared codebook fixed; only human `ch3:9` computation or auxiliary losses change.

### Rank 1 (RECOMMENDED CORE) — Swing-twist re-encode: deterministic zero-twist swing at single-child joints
**Mechanism.** In `reencode_rot6d`, for single-child parents only, replace the rank-1 Kabsch with the **minimal geodesic (swing) rotation** carrying the rest bone onto the current bone (twist=0); keep the existing local-rotation/sibling-share/token code (`:118–127`) untouched. Multi-child parents (pelvis, spine3) keep the full-rank Kabsch. **Mathematically identical to the native qbetween rotation for these 15 joints**, so no HumanML3D-native dependency is needed.
**Expected jitter reduction.** Removes the dominant target noise at the source: rot6d-target accel 2.32→~0.03 (single-child), twist jitter 109→~0 deg/frame². Should break the 7.8× plateau and move toward the physical floor (§4). **Not to 1× alone** (swing-amplification floor remains).
**Root vs symptom.** **Root.** Removes the ill-posed target rather than damping its leakage.
**Effort.** Low — ~5–10 lines + one branch in `reencode_rot6d`; then a full human re-convert + VQVAE retrain (the expensive part) only after the cheap pre-train gates (§6) pass.
**Risk.** (i) 180° bone-reversal singularity in the geodesic swing (axis `u×v` undefined) — rare for human limbs but must fall back to a transported/reference axis (Rank-1b) or it injects *new* discontinuities. (ii) Does **not** fix spine3 (Rank-2). (iii) Token cache + backbone must be rebuilt.
**Interaction with fixed constraints.** Perfect: stays in the per-parent sibling-shared animal convention, preserves Gate B/C for single-child (FK-lossless, verified 0.0000 mm), leaves the codebook untouched.

### Rank 1b (REQUIRED COMPANION) — reference-axis / parallel-transport twist as the singularity-robust variant
**Mechanism.** Instead of bare zero-twist, define twist relative to a smoothly varying reference (parent-bone-transported frame or projected global up). Same smoothness, but well-defined through the 180° flip. Use as the fallback inside Rank 1.
**Everything else** as Rank 1. This is not optional polish — it is the correctness guard for Rank 1.

### Rank 2 (REQUIRED for the residual) — bounded regularized fit for spine3 (the near-degenerate multi-child fan)
**Mechanism.** spine3's fan (neck + two near-coplanar short collars) is near-degenerate, so even its *swing* is poorly conditioned → tokens 12–14 measure ~2.0 rot6d-accel in CUR (vs pelvis ~0.014, already fine). Zero-twist swing does **not** apply (the multi-child rotation is position-determining). Needs a temporally-coherent / regularized fit with a **bounded position-error budget** (Gate B/C must not regress beyond the ~6.8 mm floor).
**Expected effect.** Cleans 6/21 tokens that Rank 1 leaves jittery. Secondary magnitude vs the 15 single-child joints, but real.
**Root vs symptom.** Root (for those 6 tokens). **Effort** Med. **Risk** Med — trades a small position error for smoothness; must be bounded and Gate-checked. **Interaction:** stays in convention; pelvis must be left alone (it is well-conditioned).

### Rank 3 (BEST COMPLEMENT, not a standalone) — rotation-space temporal-smoothness loss (excess-jitter hinge)
**Mechanism.** Add a 2nd-difference penalty **directly on `ch3:9`** (masked, hinged to GT accel — same `_masked_accel_l1` form already used by `fk_smooth`, which crucially targets GT's *own* accel, not zero), bypassing the FK Jacobian's flat-twist direction that stalls `fk_smooth`.
**Expected effect.** On the *current* encoding, plausibly breaks the 7.8× plateau toward ~3–5× (speculative; still tracks an arbitrary GT-twist target so bounded). **On a re-encoded target it becomes well-posed** and enforces temporal coherence where the gradient is now meaningful. **Best paired with Rank 1.**
**Root vs symptom.** Symptom — but the *right-space* symptom; closest non-re-encode lever.
**Effort** Low. **Risk** Med on current encoding (GT twist is arbitrary → use the excess-jitter hinge: penalize only `|accel(rot6d_pred)| > |accel(rot6d_gt)|`). 6D≠SO(3) is a minor approximation. **Interaction:** none with codebook; pure loss add.

### Rank 4 (POLISH) — SO(3)-geodesic + FK-leverage-weighted rotation loss
**Mechanism.** Replace the naive per-element 6D L1 (`losses.py:560`) with a geodesic loss on rotation matrices (AnyTop uses one), weighting each joint by its descendant-arm length to directly target long-limb amplification. Optionally switch derived-rotation orthonormalization from Gram-Schmidt to 9D+SVD/Procrustes (Geist 2024: removes the 6D-GSO first-column gradient asymmetry).
**Expected effect.** Modest magnitude-error reduction; indirect on jitter; best on a now-well-conditioned target. **Root vs symptom** partial root (metric + amplification), does **not** fix conditioning. **Effort** Med. **Risk** Med (leverage-weight tuning; SVD-orthonorm must stay consistent with `rot6d_fk_recovery.py`'s Gram-Schmidt recovery). Complements Rank 1, does not replace it.

### Rank 5 (ESCAPE HATCH, goal-incompatible) — per-species route selection
**Mechanism.** Consume human via the already-smooth POSITION route (1.48×), animals via FK. **Completely** removes human FK jitter for *position* output, zero work.
**Why it is not the answer.** Cross-topology **transfer retargets rotations** onto a different skeleton — you cannot transfer world positions across topologies. The project headline (multi-topology transfer + text-control gen) **needs** human rotations. **Use only as a demo/render stopgap**, never the research fix. (It is, however, the correct framing of the *decision*: if a given downstream consumer only needs human RIC positions, that consumer's jitter is already solved with no re-encode — see §4/§5.)

### Rank 6 (≤27% cap) — latent / token temporal-coherence constraint
**Mechanism.** Penalize pre-snap latent acceleration / token-transition smoothness to attack the RVQ-snap component. **Hard-capped at ~27%** of the jitter (the encoder/decoder owns ~73%): 7.8×→~5.7× at the theoretical best, which it won't reach. **Effort** Med-High (touches the locked latent/RVQ path). **Risk** Med (codebook-fixed invariant). **Defer** until §6 shows residual jitter is RVQ-dominated.

### Rank 7 (cosmetic, fail-loud risk) — post-hoc temporal filtering
Savitzky-Golay/one-euro on decoder output. Arbitrarily low jitter, trivial effort, but **cosmetic only**: lag, damps real fast motion, does nothing for raw-token consumers (backbone/transfer), and **hides jitter from QA/metrics** (violates fail-loud / QA-primacy). Acceptable only as final-render polish, never as the reported fix.

### Rank 8 (exhausted) — raise `w_fk_smooth`
The stall is conditioning, not weight. Marginal/negative (over-smooths real motion). Lowest.

---

## 4. Realistic expected outcome — encoding-fixable vs physically-fixed

The realistic-target investigation computed lever arms from the actual HML3D 000021 skeleton. **Amplification R = subtree-reach / immediate-bone-length** (meters of distal position error per radian of local rotation error):

| joint | bone (m) | subtree reach (m) | amplification |
|---|---|---|---|
| l_hip | 0.103 | 0.875 | **8.5×** |
| l_collar | 0.137 | 0.548 | 4.0× |
| l_shoulder | 0.132 | 0.510 | 3.9× |
| spine1 | 0.132 | 0.493 | 3.7× |
| l_knee | 0.394 | 0.482 | 1.2× |
| l_elbow | 0.257 | 0.266 | 1.0× |

The dangerous joints are **proximal** (short bone, long subtree). A 0.57° error at the hip sweeps the foot ~8.8 mm — larger than the entire human GT-pos signal (5.60).

**The split:**
- **Encoding-fixable (the dominant chunk):** the *twist* contribution. The Kabsch twist is arbitrary/high-entropy/off-manifold → large recon error → amplified ×3.7–8.5 at proximal joints. Re-encoding (Rank 1) collapses this toward swing-recon levels.
- **Physically fixed (the floor):** the *swing* contribution. Swing is anchored by child positions, which the model already reconstructs at **1.48×** (position route). But residual swing error still pays the same proximal amplification tax. **No reparameterization within the fixed 13ch per-parent convention lowers R/bone — it is anatomy.**

**Honest target after a good re-encode:**
- **NOT 1.0–1.1× (animal).** Precluded by *topology*, not coverage: the human hip's 8.5× swing-amplification has no animal analogue. Targeting "human == animal FK" chases a physically unavailable number.
- **NOT 1.48× (position route).** That is the no-leverage floor; the FK route always pays the swing-amplification tax.
- **Realistic ≈ 2–3×** (position-route recon noise ~1.48× × proximal amplification averaging ~2–4×), **best case ~1.5–2×** only if re-encode is combined with the human-upsampling curriculum sharpening the codebook's human region.

**This directly contradicts the twist-conditioning investigation's claim that Rank 1 could "approach the animal ~1.1× floor."** I side with the realistic-target analysis: that claim conflates "twist jitter → 0" (true, and decisive) with "FK jitter → animal" (false — the residual swing error is still leverage-amplified). **Report ~2–3× as the honest ceiling, not 1×.** The §6 physical-floor simulation exists precisely to nail this number before spending GPU.

**Strategic corollary.** The human position route is *already* animal-smooth (1.48×). Re-encoding is a full re-convert + expensive VQVAE retrain whose best outcome (~2–3×) is **still worse than the position route you already have.** So re-encoding is justified **only because rotations are load-bearing for the project's cross-topology-transfer + text-gen headline** (you cannot transfer positions across topologies) — not because there is no cheaper way to get smooth *human positions*. Keep this honest in any writeup.

---

## 5. RECOMMENDED PATH

**Re-encode human single-child joints to a deterministic, singularity-robust swing (Rank 1 + Rank 1b), add Rank 2 for spine3, and pair with the rotation-space excess-jitter hinge (Rank 3). Gate the expensive retrain behind the §6 cheap checks. Target ~2–3×, not 1×.**

Rationale:
1. **It attacks the root** (the ill-conditioned twist target), which §2 shows is the only thing that can move the plateau — every downstream/loss-only fix (`fk_smooth`, latent coherence, post-hoc filter) is structurally capped above the floor.
2. **It is the convergent recommendation** of four independent investigations and the literature, and it is **provably FK-lossless** (0.0000 mm) for single-child joints, so it cannot regress Gate B/C there.
3. **It stays inside every fixed constraint** — per-parent sibling-shared convention, 13ch format, untouched shared codebook — and needs **no HumanML3D-native-data plumbing** (the in-place swing equals native qbetween, avoiding index/rest/mirror-alignment risk).
4. **Rank 1b and Rank 2 are not optional** — the 180° singularity and the spine3 near-degenerate fan are the two ways a naive Rank-1 implementation silently fails.
5. **Do NOT pursue the AMASS-SMPL "true twist" path** unless a downstream consumer is later proven to need physically-faithful twist — the jitter data does not require it, and it reintroduces retarget/mirror/floor/hand-trim alignment risk for no jitter benefit.
6. **Keep human-upsampling separate** — it addresses the animal-dominant-codebook confound (suspect 3), is already in progress, and is what may close the last gap from ~2–3× toward ~1.5–2×. Do not conflate its effect with the re-encode's.

---

## 6. VALIDATION PLAN — prove the re-encode is better BEFORE a full re-convert + retrain

Three cheap, decisive pre-train gates (no GPU), then a gated smoke train. **Run all measurements with `fk_smooth` held at a fixed config** (ideally OFF) so the re-encode's effect is not confounded by the loss (see §7 measurement caveat).

**Gate A — FK-floor / Gate B/C preserved (cheap, decisive).** Re-encode a held-out human subset; run `scripts/_scan_fk_mismatch_full.py`. Expect single-child FK-vs-RIC **unchanged (Δ ≈ 0.0000 mm**, verified for zero-twist swing); overall `gt_fk_mismatch` stays at the ~6.8 mm / p95 2.3% floor. **Fail loud** if spine3's Rank-2 fit pushes the floor materially higher than budget.

**Gate B — rotation-target conditioning improved (cheap, decisive).** On the re-encoded subset, measure accel of `ch3:9` per joint group (probe scripts already written). Expect single-child rot6d-target accel **2.3 → ~0.03** (≈native), twist jitter **109 → ~0 deg/frame²**. This is the *direct* proof the target noise is gone — independent of any training.

**Gate C — physical-floor simulation (cheap, sets the honest ceiling).** Take GT rotations, inject per-channel noise calibrated to the model's *position-channel* recon fidelity (~1.48× level), FK, measure jitter. This isolates `leverage × well-conditioned-recon-noise` with no twist pathology and no training → tells you whether the irreducible floor is ~2×, ~3×, or higher, i.e. the re-encode's true headroom (current 7.8× → floor). **Decision input:** if the simulated floor is already near 7.8×, re-encode is not worth the retrain; if it is ~2–3×, headroom is real.

**Gate D — short smoke train (only if A/B/C pass).** Short finetune/retrain on the re-encoded human set. Measure recon-FK human jitter via `scripts/_fk_mpjpe_diag.py` **with `--continuous`** to get the encoder-vs-RVQ split. Expect the jitter to fall from 7.8× **toward the Gate-C floor (~2–3×)**, with most improvement in the encoder/decoder (~73%) component.

**Gate E — visual QA (the binding gate, per CV-visual-primacy).** Side-by-side **multi-frame GT-vs-recon-FK human gif** (twist-jitter is invisible in single frames and in scalar metrics). Per project rule, this gif goes to the user for the visual verdict; numbers alone do not pass.

**Commit to the full re-convert + full VQVAE retrain only if A, B, C, D, and E all pass.** A/B/C cost minutes and can kill a bad re-encode before any GPU spend; D/E confirm the training-time payoff.

---

## 7. Risks & unknowns — what could be wrong

- **The ~2–3× target is an estimate, not a measurement.** It comes from lever-arm geometry × position-route fidelity. Gate C is designed to replace it with a measured floor; until then, treat 2–3× as provisional and do not promise animal-equal smoothness.
- **Conflicting target claims (surfaced, not averaged).** twist-conditioning says "approach 1.1×"; realistic-target says "~2–3×, 1× precluded." I chose the latter (it computed the lever arms). If Gate C shows the swing-amplification floor is actually low, the optimistic claim could partially hold — but plan for ~2–3×.
- **Re-encode may be necessary but not sufficient.** The 73/27 encoder/RVQ split and long-limb leverage are independent of the target-noise fix. Rank 1 removes the unlearnable target; the leverage tax and the RVQ-snap residual remain. Rank 3 (rotation-space hinge) and possibly Rank 6 (latent coherence, ≤27%) may be needed to reach the floor.
- **180° singularity could inject new jitter.** If Rank 1b's reference-axis fallback is implemented sloppily, the geodesic swing's flip at `v=−u` becomes a *new* discontinuity. Must be explicitly tested (rare in human limbs, but present in some clips).
- **spine3 is not fixed by the core fix.** Tokens 12–14 (~2.0 in CUR) stay jittery unless Rank 2 is done, and Rank 2 trades position error for smoothness — its budget must be Gate-A-bounded. Pelvis must be left alone (well-conditioned; touching it would regress).
- **Sibling-shared last-child-wins under the MODEL.** Recovery reindex (`anytop_rot6d_fk.py:149`) takes pelvis rot from token 3 and spine3 rot from token 14; with GT all siblings agree (lossless), but the model reconstructs siblings independently and disagrees, so the recovered multi-child rotation is whatever the model produced for one arbitrarily-chosen child, and effort on discarded siblings is wasted/unconstrained. Relevant if Rank 2 changes multi-child handling.
- **Codebook / token-cache / backbone coupling.** A re-encode changes the rotation manifold → the n8192 token cache must be rebuilt and any backbone trained on old tokens will not transfer. This is a real cost beyond the VQVAE retrain.
- **Measurement-context confound (43.70 vs 102.4).** 7.8× is post-`fk_smooth`; 15×/11× and the 73/27 split are pre-loss. Comparisons across the re-encode must hold the loss config fixed, or the re-encode's effect is conflated with `fk_smooth`'s damping. Gate-measure with `fk_smooth` OFF.
- **6D-GSO gradient asymmetry (Geist 2024).** The first 6D column dominates the gradient; 9D+SVD orthonormalization is more robust. Out of scope for the minimal fix, but if the re-encode's target is still imperfectly learned, this is a cheap robustness lever — provided it stays consistent with the Gram-Schmidt recovery in `rot6d_fk_recovery.py`.
- **Animal-dominant codebook (suspect 3) is untouched by the re-encode.** Re-encode moves human rotations onto a smoother manifold but does not change minority coverage; the human-upsampling curriculum is the separate lever. If after re-encode + Rank 3 the jitter sits at the high end of 2–3×, coverage — not encoding — may be the remaining bottleneck.
- **Literature caveat (fail-loud).** AnyTop's exact source-of-rotation (native BVH vs derived) and per-channel dims were inferred as field-standard + the paper's geodesic-loss mention, not extracted verbatim; the "RP6JR is smoother" claim (Jin & Haworth 2025) is abstract-level. Confirm against AnyTop code/§method if either becomes load-bearing in a paper.

---

**Key code anchors (verified against the live tree this session):** `scripts/convert_humanml3d_to_anytop13.py` — `_kabsch_batch` `:84–91` (rank-1/det≈0 twist null-space), `reencode_rot6d` `:99–127` (fix branch at `:111–117`; sibling-share `:125–126`), native rot6d read `:138`/stored `:157`/**overwritten** by `reencode_rot6d` call `:338`, FK-floor docstring `~:447`. `src/data/anytop_rot6d_fk.py:107–108,149` (telescoping FK + last-child-wins reindex). `src/models/graph_salad/rot6d_fk_recovery.py` (per-parent sibling-shared FK, Gram-Schmidt 6D→matrix). `src/models/graph_salad/losses.py:560` (naive 6D rot L1 — Rank-4 target), `:689` (`_masked_accel_l1`, GT-accel-hinged), `:764–768` (`fk_smooth`). Diagnostics: `scripts/_fk_mpjpe_diag.py` (`--continuous` = encoder/RVQ split), `scripts/_scan_fk_mismatch_full.py` (Gate A). Literature: HybrIK (arXiv 2304.05690), Zhou CVPR2019 (1812.07035), Geist 2024 (2404.11735), Jin & Haworth 2025 (2512.04499), AnyTop (2502.17327).

---

## Appendix: Adversarial critic (verbatim)

Overall: this is a strong, sophisticated analysis. The core mechanism (rank-1 Kabsch twist null-space → LAPACK tie-break discontinuity → position-invariant gauge that cancels in GT-FK but breaks under independent RVQ snap, amplified by limb leverage) is sound and internally consistent. The "correct-but-hard, not a bug" line is drawn correctly (intrinsic FK-floor vs avoidable twist-fill), and §2.5's "native HML3D rotation is already twist≡0 by qbetween construction → 'use native' == 'zero-twist swing'" is a genuinely valuable correction that also defuses the transfer-twist-loss worry. Gate E as the binding visual gate and the R7 surfacing of the target dispute are right. The issues below are mostly about overclaimed magnitudes, one self-inconsistent solution, and validation gates that don't fully discriminate.

Highest-value issues:

- **The 2–3× vs 1.1× verdict is decided on incomplete evidence.** The "1× precluded by topology" claim rests on the assertion that the human hip's 8.5× leverage "has no animal analogue" — but no animal lever arms were computed. TrueBones quadrupeds/dinosaurs/tails have proximal joints driving long subtrees with comparable or worse amplification, yet animals hit 1.1×. That implies the floor is set by *recon-error × leverage*, and animals win on the recon-error term (coherent native twist + good coverage) **despite** leverage. Since the re-encode pulls exactly that recon-error lever, 1× is *not* obviously precluded. The doc sided with "realistic-target" because "it computed lever arms," but lever arms alone don't set the floor. Honest statement: floor unknown until Gate C/D; do not assert 2–3× as a ceiling.

- **No downstream jitter requirement is ever defined.** The entire effort optimizes a number with no spec for what cross-topology-transfer / text-gen actually *needs*. §4 admits best-case (~2–3×) is worse than the already-available position route (1.48×) and justifies the retrain only by "rotations are load-bearing" — but never asks whether 7.8× is actually insufficient for the consumer. Add a requirement-analysis gate (what jitter does transfer/gen tolerate?) *before* committing GPU; otherwise the expensive retrain may be premature or under/over-targeted.

- **Inconsistent ×-denominators across the doc.** 7.8× (43.70) and 1.48× (8.29) are ÷5.60. But 102.4="≈15×" and 74.3="≈11×" only reconcile against ÷~6.8 (102.4/5.60=18.3×, not 15×; /6.8=15.1×). So the headline "15× → 7.8× (fk_smooth)" mixes baselines and overstates/confuses the loss's effect. State the pre-config GT-FK jitter baseline and put all multipliers on one denominator.

- **Rank 3 is internally inconsistent.** An excess-jitter hinge ceilinged at GT's *own* rot6d accel (2.32, noisy) cannot push single-child jitter below GT-noise on the current encoding — pred either already sits below 2.32 (hinge inactive) or gets pulled back to 2.32 (still noisy). So the "~3–5× on current encoding" claim contradicts the mechanism the doc specifies. The hinge only does real work on *re-encoded* data or against a smoothed/zero target. Drop the current-encoding improvement claim.

- **73/27 split (and Rank-6 "≤27% hard cap") treats a non-additive metric as linearly decomposable.** 102.4 (quantized) and 74.3 (continuous) are acceleration *norms*; the difference isn't a clean "RVQ contribution" because errors can be correlated/anti-correlated. The "hard-capped at ~27%" for Rank 6 is not rigorous — present as a rough indication, not a ceiling.

Validation gates that don't discriminate:

- **Gate B proves smoothness, not correctness.** Any smooth re-encode — including one that FKs to *wrong* positions — trivially passes a rot6d-accel check. It only discriminates a good fix when ANDed with Gate A (FK fidelity). The doc frames B as independent "direct proof"; reword to "necessary, not sufficient; binds only with A."

- **Gate C is ill-posed / partly circular.** Rotation noise is calibrated from the *position-channel* (ch0:3) recon jitter — a category mismatch, and the position→rotation conversion *is* the leverage map you're trying to measure. Also it injects noise on top of GT rotations that (under the current encode) still carry the Kabsch twist pathology — must use re-encoded twist-zero GT or it inherits the very noise it claims to exclude. As written it won't yield a trustworthy ceiling.

- **Gate D has no matched control.** Smoke-training only on re-encoded data can't separate the re-encode's effect from training-config/init differences. Requires a same-length, same-config smoke train on the *old* encoding as baseline (the §7 "hold fk_smooth fixed" caveat doesn't cover this).

Missed / underspecified options:

- **Temporal SVD-basis continuity propagation** is not considered: keep the full-rank Kabsch but enforce frame-to-frame continuity of the degenerate eigenvectors (kill the LAPACK tie-break flips), which removes the discontinuity *and* — unlike zero-twist swing — also handles the near-degenerate multi-child spine3 fan. This is a single mechanism covering both Rank 1 and Rank 2.

- **Hybrid output** (model emits both; use the already-smooth position-route positions + analytic-swing rotations derived from them) is dismissed wholesale under Rank 5, but could deliver smooth rotations without a full re-convert + retrain. At least cost it as an option.

- **"Mathematically identical to native qbetween"** requires the AnyTop rest offsets to match HML3D's rest bone directions. If rest poses differ, the swings are not identical — the claim of "no native dependency, provably equal" is glossed; verify rest-frame alignment.

Lower-priority precision/overclaim:

- "Convergence of four independent investigations" overweights correlated analyses (same model, same code-reading) as independent corroboration — not strong evidence per se.
- Rank 2 (spine3) is marked **REQUIRED** but the method is vague ("temporally-coherent / regularized fit") and its own magnitude is called "secondary" — resolve the priority/spec tension; either give a concrete method or demote to optional pending Gate-B per-token evidence.
- "Animal segments are short (low leverage)" is unsupported (tails/necks/quadruped legs); the real reason animals are smooth is native coherent twist + coverage. The leverage half of that claim should be dropped or measured.
- Arithmetic/exactness slips: 109 deg/frame² ≈ 1.9 rad, not "≈1.5 rad"; the cited 109-vs-0.5 gap is ~218×, not "200–1000×"; "exactly 0.0000 mm" overstates float exactness (state the tolerance).
- "180° bone-reversal singularity rare for human limbs" is likely underestimated for HumanML3D's diverse motions (deep knee/elbow flexion, kicks, sitting) — fine that Rank 1b is mandatory regardless, but don't let "rare" license under-testing the fallback.
