Review a watchdog patch for correctness. Reply with an explicit verdict: PASS or NEEDS-FIX, plus enumerated issues.

## Context
We stopped the old n8192 L4safeTB VQVAE run and are starting a NEW VQVAE run on a DIFFERENT dataset (L4-safe + HumanML3D), with MAX_COARSE=72 (NOT 96). The cross-node H200 auto-resume watchdog `scripts/_watchdog_h200_vqvae.sh` was hardcoded to the OLD run:
- OUT_REL was hardcoded to the old `...L4safeTB_C96..._n8192...` run dir.
- Its resume() relaunch command passed ONLY BATCH_SIZE/LR/NUM_CODES to the 2node launcher, NOT ANYTOP_ROOT or MAX_COARSE. Because `scripts/_launch_graph_vqvae_2node_h200.sh` defaults ANYTOP_ROOT to the L4safeTB root and MAX_COARSE to 96, an alloc-expiry auto-resume would silently relaunch the NEW run with the WRONG dataset + wrong coarse-slot count, corrupting it.

## The patch just applied to `scripts/_watchdog_h200_vqvae.sh`
1. Parameterized OUT_REL / ANYTOP_ROOT / MAX_COARSE / MAX_JOINTS / MAX_FRAMES / NUM_CODES / BATCH_SIZE / LR as env-overridable (`${VAR:-default}`) with L4safeHuman defaults (root=data/animo4d_anytop_clean_L4_safe_plus_humanml3d, MAX_COARSE=72, MAX_JOINTS=144, MAX_FRAMES=64, NUM_CODES=8192).
2. Added a fail-fast guard near the top: if OUT_REL contains "L4safeHuman" and (ANYTOP_ROOT != the humanml3d root OR MAX_COARSE != 72) then echo + exit 1.
3. resume() relaunch command now also passes ANYTOP_ROOT / MAX_COARSE / MAX_JOINTS / MAX_FRAMES (in addition to BATCH_SIZE / LR / NUM_CODES) into the `setsid nohup env ... bash scripts/_launch_graph_vqvae_2node_h200.sh` invocation.

## Files to read
- `scripts/_watchdog_h200_vqvae.sh` (the patched watchdog — this is the file under review)
- `scripts/_launch_graph_vqvae_2node_h200.sh` (the launcher resume() calls; confirm it consumes ANYTOP_ROOT/MAX_COARSE/etc. from env and forwards them)
- `scripts/_launch_graph_vqvae.sh` (the inner per-node launcher; confirm it forwards --anytop_root/--max_coarse/etc. to train_graph_vqvae.py)

## Verdict must cover
(a) Does the resume now correctly relaunch the NEW L4safeHuman dataset + MAX_COARSE=72, NOT the old L4safeTB/C96 defaults? Trace the env through: watchdog resume() -> ssh -> setsid nohup env -> _launch_graph_vqvae_2node_h200.sh -> per-node inner _launch_graph_vqvae.sh -> train_graph_vqvae.py CLI flags.
(b) Is the fail-fast guard correct, and does it actually prevent a mismatched resume (i.e. would it catch the exact corruption mode described above)? Is its placement (before flock / main loop) appropriate?
(c) Any bug introduced by the patch: shell quoting, env var expansion inside the double-quoted ssh string, the guard boolean logic, or any regression vs the proven original watchdog logic (discovery, scoped kills, flock, never-scancel)?
(d) Will every newly-added env var actually survive the ssh -> setsid nohup env -> launcher chain (no word-splitting / quoting loss)?

Be concrete and cite line content. If PASS, say so plainly. If NEEDS-FIX, enumerate each fix.
