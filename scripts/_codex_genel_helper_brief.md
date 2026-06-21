Review a refactor for correctness. Reply with explicit verdict: PASS or NEEDS-FIX + enumerated issues.

## Goal
Extract the Graph-CodeFlow text->motion gen-eval loop from the offline CLI into a SHARED helper so the backbone trainer's upcoming ONLINE eval hook calls the SAME code (no drift). Plus 3 required behavior changes for the online use-case. This is step (a) of a plan reviewed/approved by the user.

## Files
- NEW `src/eval/codeflow_gen_eval.py` — the shared helper (metric fns moved verbatim + `run_gen_eval(...)` + `motion_id_bucket`).
- CHANGED `scripts/_eval_codeflow_gen_in_evalspace.py` — offline CLI now imports + calls `run_gen_eval`; keeps arg parsing, model loading (flow/frozen VQVAE/frozen evaluator/T5), data-contract checks, strided idxs (+ --exclude_truebones), reporting.
- Reference (the PRE-refactor version is in git HEAD `1322b15` if you want to diff): the loop + metrics were moved VERBATIM; only the 3 changes below differ.

## The 3 intended behavior changes (verify each is correct + safe)
1. **Per-subset split** was `object_type.startswith("PZ_") -> animo4d/truebones`. object_type is None for human clips, so it's now by **motion_id** via `motion_id_bucket(mid)`: HML3D*->human, PZ_*->animal, else->truebones. Per-subset keys are now {animal, human, truebones} (whichever present). For the L4safeHuman dataset this yields animal+human; for the old mergedL4TB it yields animal+truebones. Confirm motion_id is the right key (it is collected via `ds._plan[di][1]["motion_id"]`, the same field the offline script already used for --exclude_truebones).
2. **No SystemExit** in the loop. The old code did `raise SystemExit` when continuous-decode T < a clip's gt_T (would crash a training job if reused online). Now it SOFT-CLAMPS: `t_place=min(cont.shape[1], gt_T)`, places the available prefix, and if clamped also trims that clip's `frame_mask` (lazily cloned `clamp_fm`) so zero tail frames are NOT scored as valid, + logs a WARN. With num_frames=300 / T_lat=75 / stride=4 -> decode T=300 >= gt_T (<=300) so this never fires, but it must be safe if it ever does. Verify: (a) the clamp math; (b) that trimming `frame_mask[bi, t_place:]=False` keeps it a contiguous-True-prefix (GraphMotionBatch requires frame_mask be a contiguous prefix — see src/models/graph_salad/batch.py from_collate_dict validation); (c) that `dataclasses.replace(batch, anytop_x=gen_x, frame_mask=clamp_fm)` only overrides gen's frame_mask while GT (`batch`) keeps its own; (d) the common no-clamp path leaves frame_mask shared (no needless copy).
3. **RNG save/restore**: `flow.sample` draws from the GLOBAL torch RNG (src/models/CodeFlow_Model/flow.py:268). The helper saves `torch.get_rng_state()`+`torch.cuda.get_rng_state_all()`, seeds for reproducible sampling, and restores both in a `finally` (so an in-loop eval never perturbs the trainer's dropout/noise RNG stream, even on exception). Verify the save/restore is complete + correct (CPU + all CUDA devices; restored on every path).

## Verdict must cover
(a) Is the refactor behavior-PRESERVING for the OVERALL metrics (metric fns rprec_pool/pooled_rprec/fid_score/mean_pair_l2/subset_metrics + the generation loop incl the per-clip frame_mask_lat/token_mask rebuild — the prior codex blocking fix — moved verbatim)? Any accidental semantic change?
(b) Are the 3 changes each correct + safe (esp. 2b/2c frame_mask handling + contiguous-prefix invariant, and 3 RNG completeness)?
(c) Offline CLI correctness: does it still build idxs (+ exclude_truebones), pass the right args to run_gen_eval, restore latent_mean/std, run the data-contract checks, and report overall+per_subset? Any arg dropped/mismatched vs the helper signature?
(d) Any bug introduced: import paths (src.eval.codeflow_gen_eval, collate_fn, GraphMotionBatch), the `@torch.no_grad()` on run_gen_eval + t5_encode_batch, the dataclasses.replace usage, the finally block, the metric_gen generator.

Read both files (and src/models/graph_salad/batch.py for the frame_mask invariant, src/models/CodeFlow_Model/flow.py:241-274 for flow.sample). Be concrete with line refs. If PASS say so plainly.
