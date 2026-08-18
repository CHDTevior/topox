# Plan — Fix human rot6d-FK via FK-leverage reweighting (raise `w_fk`)

**Author:** Claude (Opus 4.8) · **Date:** 2026-06-28T02:08Z · **Status:** DRAFT for reviewer (codex gpt-5.5 xhigh) audit BEFORE any code change / launch.
**Decision context:** user chose option **B** (prepare the w_fk-heavy config now, run as a clean A/B vs the current curriculum run; train both concurrently *if* cards allow). This doc is the pre-implementation plan to be audited.

---

## 1. Problem & root cause (established, codex-verified)

Human rot6d→FK reconstruction is poor (~53–63mm MPJPE) while human position-route recon is good (~35mm) and animals are fine on both routes.

Decisive evidence (all in `[[project_next_n8192_vqvae_balanced_curriculum]]`):
- **FROZEN converged n8192 (ep239, NO curriculum): human rot6d-FK MPJPE 63.4mm vs human position 34.7mm; animal FK 28.7 ≈ animal position 23.8.** → structural, not curriculum, not mid-training.
- **Sibling-duplicate convention REFUTED** as the cause: sibling-averaged-parent FK == last-child FK (frozen 53.7→53.5, curric 50.7→50.7 on 12 human clips, 288f). GT sibling dispersion 4.5e-8 (premise valid); recon dispersion ~0.025 but averaging it gives no FK gain.
- **By elimination → the cause is the rotation objective:** `rot` loss is direct normalized **6D-L1** (per-element), which is **NOT FK-leverage-weighted and NOT SO(3)-geodesic**. Small rotation errors on high-leverage **proximal** joints (shoulder/hip) amplify into large distal FK position errors along human long limbs; in 6D-L1 these errors look as light as distal-joint errors, so the optimizer has no incentive to prioritize them. Animals are immune (short levers + native rot6d ⇒ FK≈RIC).

We DO have an FK-position loss (`fk` = masked L1 of rot6d-FK(pred) vs RIC(gt)) that IS leverage-aware, but it is dominated by the raw `rot` term in the total.

### Loss-magnitude evidence (per-20ep MIN, curric run)
| term | weight (current) | typical value | weighted contribution |
|---|---|---|---|
| `rot` (6D-L1) | w_rot=1.0 | ~0.45 | **0.45** |
| `fk` (FK-pos L1) | w_fk=1.0 | ~0.04 | **0.04** |
| `pos` (RIC) | 1.0 | ~0.11 | 0.11 |
| `vel` | 1.0 | ~0.18 | 0.18 |
| `world` | 0.25 | ~0.04 | 0.01 |

The FK-position objective currently carries **~9%** of the gradient weight that the raw 6D-L1 objective does. The fix is to rebalance.

---

## 2. Intervention — raise `w_fk` (primary), single-knob

**Primary (clean single-variable A/B): `w_fk` 1.0 → 5.0**, everything else identical to the current curriculum run.
- Rationale: at w_fk=5, weighted fk ≈ 0.20 (~44% of rot's 0.45) — FK-position correctness becomes a first-class objective WITHOUT crushing the `rot` term that still supervises the rot6d output channels (ch3:9) the human renderer reads.
- This is a starting point chosen by loss-magnitude balance; the early-epoch fk-vs-rot balance + human FK MPJPE should be watched and the weight retuned if needed.

**Alternative (more aggressive, only if w_fk=5 underdelivers): `w_fk`=10 + `w_rot`=0.5.** Listed for the reviewer; NOT launched unless the moderate one plateaus high.

**Out of scope for this round (heavier change, deferred):** a true SO(3)-geodesic or explicit per-joint FK-leverage-weighted rot loss (new loss code). If `w_fk` reweighting proves insufficient, that becomes the next plan. Keeping this round surgical (loss weights only, no new loss math) per Karpathy "simplicity / surgical".

### Why this should work + the explicit trade-off it accepts
- `fk` directly supervises rot6d→FK positions vs RIC(gt). Raising its weight pushes the decoder to emit rotations whose **FK lands on the right position**, which is exactly the failing objective. Room is huge (human FK 53mm vs floor 8.6mm).
- **TRADE-OFF (must be measured, not assumed):** for human, GT's own rot6d→FK deviates from RIC by ~8.6mm (the Kabsch re-encode floor). So pushing pred-FK toward RIC pulls pred-rot6d *away* from GT-rot6d → the **rot6d CHANNEL** (ch3:9) MSE may *worsen* for human even as FK-positions improve. For the human renderer (which uses the FK route), FK-position is what matters, so the trade is likely worth it — but the ablation MUST report per-channel rot6d MSE alongside FK MPJPE so we see the cost. The FK route also can't beat the 8.6mm floor for human (ceiling).
- Animals: `fk` target is clean (animal FK==RIC, floor 0), so raising w_fk helps/neutral for animals too. Global w_fk raise is fine.

---

## 3. Exact code changes (all MUST pass codex before launch)

### 3.1 `scripts/_launch_graph_vqvae_2node_h200.sh` — forward W_FK/W_ROT
- Add env defaults near the other knobs (NUM_CODES/BATCH_SIZE/LR): `W_FK="${W_FK:-1.0}"`, `W_ROT="${W_ROT:-1.0}"`.
- Append `--w_fk "$W_FK" --w_rot "$W_ROT"` to the torchrun → train_graph_vqvae.py argument list.
- Echo them in the run-config log line.
- Backward-compatible: default 1.0/1.0 = current behavior (the curric baseline run is unaffected).

### 3.2 `scripts/_watchdog_h200_vqvae.sh` — preserve W_FK/W_ROT on resume (CRITICAL — this is the exact class of bug we already hit with curriculum)
- Add env defaults at top: `W_FK="${W_FK:-1.0}"`, `W_ROT="${W_ROT:-1.0}"`.
- Add `W_FK=$W_FK W_ROT=$W_ROT` to the `setsid nohup env ...` resume block (line ~194), alongside the HUMAN_UPSAMPLE_* forwarding.
- Add a START/resume log line echoing `w_fk=$W_FK w_rot=$W_ROT`.
- **Fail-loud guard (mirrors the curriculum guard):** if this watchdog instance is meant to manage the w_fk-heavy run (W_FK>1) but resume would launch with W_FK=1 (env lost), ABORT loudly rather than silently revert.

### 3.3 `scripts/_launch_graph_vqvae.sh` (base/single-node launcher) — same W_FK/W_ROT forwarding
- For smoke-test parity. Same env→arg pattern.

### 3.4 `scripts/train_graph_vqvae.py` — NO CHANGE
- `--w_fk` / `--w_rot` args already exist (lines 307/311, default 1.0); the loss dict already wires them (lines 586-587). Verified.

---

## 4. Experimental design — clean A/B

| run | codebook | curriculum | loss weights | from-scratch | role |
|---|---|---|---|---|---|
| **curric-baseline** (already running) | n8192 | two-phase 50→60% | w_fk=1, w_rot=1 (default) | yes | BASELINE |
| **curric-wfk5** (NEW) | n8192 | two-phase 50→60% (identical) | **w_fk=5**, w_rot=1 | yes | TREATMENT |

- Only `w_fk` differs ⇒ isolates the FK-reweight effect. Same seed (42), same data, same arch, same curriculum, same lr/batch/epochs.
- New OUT: `runs/vqvae_L4safeHuman_C72_J144_d512_Q4_n8192_b16g64_300ep_curric50to60_wfk5_seed42` (OVERWRITE=1, new dir).
- Compare at **matched epochs** (during) and at **ep300** (final): human rot6d-FK MPJPE + position MPJPE + per-channel rot6d MSE + QA visual (the real gate).

---

## 5. Concurrency & resources (the "train both if cards allow" part — has a real infra catch)

**Catch:** the current cross-alloc H200 watchdog (`_watchdog_h200_vqvae.sh`) **auto-discovers "exactly one dual_h200 + one quad_h200"** and uses fixed singleton lock/pid files (`.aris/meta/.watchdog_h200_vqvae.lock`, `.vqvae_h200_orch.pid`). A second concurrent cross-node H200 DDP job would (a) collide on those lock/pid files and (b) have its watchdog grab the *same* alloc pair as the baseline's watchdog. So we cannot just launch a second copy.

**Options for concurrency (pick per reviewer + actual availability — user controls allocs, I cannot self-submit):**
- **(C1) Different node-type for the new run.** Run curric-wfk5 on **4×H100 cross-alloc** (swarm_h100) or 4×A100, leaving the H200 pair for the baseline. Needs a separate, *pinned* watchdog (distinct lock/pid/OUT, allocs passed explicitly, NOT auto-discover) — a parametrized copy of the watchdog. Blocked today by MocapAnything occupying most H100/A100 (do not touch).
- **(C2) Second H200 alloc-pair.** Needs a 2nd dual+quad pair RUNNING. Requires the auto-discovery watchdog to be generalized to bind a *specific* pair (else the two watchdogs fight over allocs). More infra work.
- **(C3) Sequential (no concurrency).** Let curric-baseline finish ep300, free the H200 pair, then launch curric-wfk5 reusing it with the existing (unmodified-topology) watchdog. Zero new watchdog infra; just the W_FK env. Slower wall-clock but lowest risk.

**Recommendation:** if a clean second 4-GPU set is genuinely free → C1 with a pinned watchdog (1 small new watchdog variant, codex-reviewed). If not → C3 (sequential), which still answers the question and needs the least new code. Either way the **launcher/watchdog W_FK/W_ROT plumbing (§3) is required first.**

---

## 6. Smoke plan (before any full launch)
1. After §3 edits: `py_compile`/`bash -n` the launcher + watchdog.
2. Single-GPU (or 1-node) short smoke with `W_FK=5`: confirm the train log shows `w_fk=5.0` in the resolved config AND the early `fk`/`rot` component magnitudes shift as expected (weighted fk contribution rises toward ~0.2).
3. If concurrency via a new/pinned watchdog: 5-case guard test (as done for the curriculum watchdog) + verify it binds ONLY its own allocs and its own lock/pid (no collision with the baseline watchdog).
4. cross-node DDP smoke (WORLD_SIZE=4, via IB, rc=0) before the real run, per the cross-alloc DDP rules.

---

## 7. Acceptance gates (QA-visual is the real gate, not metric alone)
- **PASS if:** curric-wfk5 human rot6d-FK MPJPE is meaningfully **lower** than curric-baseline at matched epoch AND at ep300 (target: close a large fraction of the 53→8.6mm gap), **AND** the human rot6d-FK QA GIF (GT vs recon, multi-frame) is visibly less distorted — visual verdict delivered to user (per QA-primacy: visual > metric).
- **No-regress guards:** animal FK MPJPE and BOTH species' position MPJPE must not regress materially. Per-channel rot6d MSE is **reported** (expected to rise somewhat for human — the accepted trade-off in §2); flag if it rises catastrophically.
- **Health signals (not gates):** loss descending, codebook active count climbing, no NaN.

---

## 8. Risks / open questions for the reviewer
1. **Weight value:** is w_fk=5 the right first step, or start higher (8–10) given the 9% current contribution? Trade single-variable cleanliness vs getting a decisive signal in one run.
2. **rot6d-channel degradation:** acceptable trade for the renderer, but does anything downstream consume the rot6d channel fidelity directly (beyond FK rendering)? The backbone uses z_q latents, not raw rot6d — so likely fine, but confirm.
3. **w_fk alone vs geodesic loss:** if reweighting plateaus high, is the SO(3)-geodesic / explicit FK-leverage-weighted rot loss the necessary next step? (Deferred this round.)
4. **Concurrency infra:** C1 (pinned watchdog on H100/A100) vs C3 (sequential on the freed H200). Is the pinned-watchdog variant worth building now, or is sequential acceptable?
5. **FK-floor ceiling:** human FK can't beat ~8.6mm (Kabsch re-encode). Is that floor acceptable for the paper, or do we also need to revisit the HumanML3D→AnyTop13 rot6d encoding? (Separate, larger question.)

---

## 9. Iron-rule compliance
- All code changes (§3 + any pinned-watchdog variant) **MUST pass codex (gpt-5.5 xhigh) before launch.**
- Baseline curric run is **not touched** (changes are env-gated, default 1.0 = current behavior).
- No self-submit/cancel of Slurm; concurrency depends on allocs the **user** provides; watchdog auto-resume is the only authorized exception.
- Do not touch MocapAnything's GPUs (swarma1003 4×A100 + several swarmh1002 H100).
- Smoke before real run (cross-alloc DDP rendezvous).
