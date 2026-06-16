Reading additional input from stdin...
OpenAI Codex v0.136.0
--------
workdir: /iridisfs/scratch/ts1v23/workspace/noKslot_clean
model: gpt-5.5
provider: openai
approval: never
sandbox: danger-full-access
reasoning effort: xhigh
reasoning summaries: none
session id: 019e9b03-8595-7211-a776-c6b342f519d7
--------
user
You are an independent adversarial code reviewer with no prior context. Review ONLY the spatial_mode/use_graph_bias additions in this local repo diff, and ignore unrelated latent-dynamics loss and dual_text changes even if they appear in the same files.

Scope files:
- src/models/graph_salad/attention.py
- src/models/graph_salad/denoiser.py
- scripts/train_denoiser.py
- scripts/animate_denoiser.py
- scripts/_launch_diffusion_t2m.sh
- scripts/_launch_diffusion_t2m_4card.sh
- scripts/_launch_token_diffusion_8card_a100.sh

Claims to verify adversarially:
1. GraphAttentionBlock.use_graph_bias: when False, geodesic_bias/adjacency_bias modules are not instantiated and _compute skips adding topo bias. When True default, behavior is byte-equivalent to before. node_mask masking, fp32 softmax, FFN, residuals unchanged in both modes. Plain path is valid masked plain self-attention.
2. Wiring: DenoiserLayer takes spatial_mode and passes use_graph_bias=(spatial_mode=="graph") to GraphAttentionBlock. GraphSaladDenoiser validates graph|plain, stores it, and passes to every layer. Default graph should keep old checkpoints strict-loadable with no new keys in graph mode.
3. train_denoiser.py: --spatial_mode {graph,plain} default graph; passed to denoiser build; saved in ckpt args via vars(args); resume and init_ckpt assert ckpt spatial_mode equals CLI spatial_mode before strict load.
4. animate_denoiser.py: load_denoiser reads spatial_mode from ckpt args default graph and rebuilds denoiser with it so strict load succeeds for graph and plain ckpts.
5. Launchers: SPATIAL_MODE env default graph is threaded through inner _launch_diffusion_t2m.sh, 4-card orchestrator, and 8-card orchestrator. 4-card orchestrator also adds CAPTION_TOKEN_CACHE/CAPTION_TOKEN_MAX_LEN to COMMON_ENV. With SPATIAL_MODE unset, default graph reaches train_denoiser and behavior is unchanged for graph-mode runs sourcing these launchers.
6. Semantics: graph vs plain differ only by 2*n_heads*n_layers bias params; ablation cleanly removes only spatial graph bias while retaining pooled_skeleton additive, node_mask, temporal, text, masks.

Be specifically adversarial about:
- Any path that still references geodesic_bias/adjacency_bias when use_graph_bias=False (AttributeError)
- validate_inputs still validating adjacency/geodesic in plain mode: harmless or could fail?
- default-graph byte-equivalence for running graph runs
- resume/init spatial_mode guard quality
- shell env quoting/propagation problems in launchers

Output exactly:
PASS or NEEDS-FIX
Then a numbered list of findings. Cite file:line evidence. Include any non-blocking notes separately.
hook: SessionStart
hook: SessionStart Completed
codex
I’ll inspect only the requested files and separate the `spatial_mode` / `use_graph_bias` changes from unrelated edits before judging the claims.
exec
/usr/bin/bash -lc 'git status --short' in /iridisfs/scratch/ts1v23/workspace/noKslot_clean
 succeeded in 0ms:
 M REPO_AUDIT.md
 M scripts/_launch_diffusion_t2m.sh
 M scripts/_launch_diffusion_t2m_6card.sh
 M scripts/_launch_token_diffusion_8card_a100.sh
 M scripts/animate.py
 M scripts/animate_denoiser.py
 M scripts/train_denoiser.py
 M src/data/anytop_dataset.py
 M src/models/graph_salad/attention.py
 M src/models/graph_salad/denoiser.py
?? REPO_AUDIT_spatial_plain_20260606.md
?? handoff/20260602_2038_prism_diffusion_backbone_iteration_notes.md
?? handoff/20260602_2220_backbone_diffusion_plan.md
?? handoff/20260602_2233_backbone_diffusion_plan_review.md
?? handoff/20260603_0255_todo_token_level_text_conditioning.md
?? handoff/20260603_0330_bf16_vae_progress.md
?? handoff/20260603_0410_bf16_vae_8card_running.md
?? handoff/20260603_2200_session_handoff_t2m_render_generic.md
?? handoff/20260604_0015_anytop_t2m_evaluator_split_plan.md
?? handoff/20260604_0121_anytop_t2m_evaluator_impl_proposal.md
?? handoff/20260604_0257_anytop_t2m_evaluator_impl_progress.md
?? handoff/20260604_2043_token_level_text_conditioning_impl_plan.md
?? handoff/20260605_0615_autonomous_bf16lowlr_launch.md
?? handoff/20260605_latent_temporal_dynamics_loss_experiment.md
?? handoff/20260606_latent_dynamics_loss_results.md
?? scripts/_check_gt_foot_jitter.py
?? scripts/_codex_animate_dualpath_brief.md
?? scripts/_codex_cross_skeleton_brief.md
?? scripts/_codex_latdyn_brief.md
?? scripts/_codex_t2m_bf16_brief.md
?? scripts/_codex_t2m_n11_brief.md
?? scripts/_filmstrip_gif.py
?? scripts/_launch_diffusion_t2m_4card.sh
?? scripts/_measure_fit_train_vs_val.py
?? scripts/_measure_latent_jitter.py
?? scripts/_measure_latent_std.py
?? scripts/_measure_posterior_jitter.py
?? scripts/_measure_quality_vs_count.py
?? scripts/_measure_vae_recon_energy.py
?? scripts/_pil_skeleton_render.py
?? scripts/_plot_diffusion_loss.py
?? scripts/_plot_species_count_dist.py
?? scripts/_render_cap_traj.sh
?? scripts/_render_count_contrast.sh
?? scripts/_render_ep100_all20.sh
?? scripts/_render_ep100_bmu.sh
?? scripts/_render_ep100_dualA.sh
?? scripts/_render_ep100_latdyn.sh
?? scripts/_render_newT1_qa.sh
?? scripts/_render_tokenB_qa.sh
?? scripts/_render_train4_withgt.sh
?? scripts/_smoke_latdyn.sh
?? scripts/_t2m_cross_skeleton_dual.py
?? scripts/_t2m_qa_filmstrip.py

exec
/usr/bin/bash -lc 'git diff -- src/models/graph_salad/attention.py src/models/graph_salad/denoiser.py scripts/train_denoiser.py scripts/animate_denoiser.py scripts/_launch_diffusion_t2m.sh scripts/_launch_diffusion_t2m_4card.sh scripts/_launch_token_diffusion_8card_a100.sh' in /iridisfs/scratch/ts1v23/workspace/noKslot_clean
 succeeded in 0ms:
diff --git a/scripts/_launch_diffusion_t2m.sh b/scripts/_launch_diffusion_t2m.sh
index 8304d86..e2b2995 100644
--- a/scripts/_launch_diffusion_t2m.sh
+++ b/scripts/_launch_diffusion_t2m.sh
@@ -43,6 +43,16 @@ INIT_CKPT="${INIT_CKPT:-}"
 RESUME_CKPT="${RESUME_CKPT:-}"
 CVD="${CVD:-0,1}"
 AMP_DTYPE="${AMP_DTYPE:-fp32}"          # bf16 now bf16-safe (fp32-forced softmax); default fp32
+LR_SCHEDULE="${LR_SCHEDULE:-constant}"  # constant (default, unchanged) | cosine (warmup→cosine→lr_min)
+LR_MIN="${LR_MIN:-0.0}"                 # cosine floor (only used when LR_SCHEDULE=cosine)
+SPECIES_WHITELIST="${SPECIES_WHITELIST:-}"  # comma-sep object_types (capacity probe); empty = full 473
+TRAIN_SPLIT="${TRAIN_SPLIT:-train}"     # train (default) | all (train on all whitelisted clips incl val)
+# M2 latent temporal dynamics loss (handoff 20260605); ALL 0 = byte-equivalent.
+W_LAT_DZ="${W_LAT_DZ:-0}"               # weight on latent velocity loss ||Δz0_hat-Δz0||²
+W_LAT_DDZ="${W_LAT_DDZ:-0}"            # weight on latent acceleration loss ||Δ²z0_hat-Δ²z0||²
+W_LAT_X0="${W_LAT_X0:-0}"              # weight on direct latent loss ||z0_hat-z0||² (keep 0 first run)
+LATENT_DYN_TARGET="${LATENT_DYN_TARGET:-sample}"  # sample (default) | mu
+SPATIAL_MODE="${SPATIAL_MODE:-graph}"             # graph (default) | plain (no_graph_spatial ablation)
 # M2 token-level text conditioning (default mean_additive = current behavior).
 TEXT_MODE="${TEXT_MODE:-mean_additive}"
 CAPTION_TOKEN_CACHE="${CAPTION_TOKEN_CACHE:-}"   # required when TEXT_MODE=token_cross_attn
@@ -98,11 +108,12 @@ else
 fi
 
 echo "[t2m] $(date '+%F %T %Z') host=$(hostname) CVD=$CVD nnodes=$NNODES nproc_per_node=$NPROC_PER_NODE node_rank=$NODE_RANK"
-echo "[t2m] VAE=$VAE_CKPT (B rot6d_fk ep79)"
+echo "[t2m] VAE=$VAE_CKPT (frozen)"
 echo "[t2m] cap_cache=$CAPCACHE anytop_root=$ANYTOP_ROOT"
-echo "[t2m] per_gpu=$PER_GPU_BATCH global=$GLOBAL(=${PER_GPU_BATCH}x${NNODES}x${NPROC_PER_NODE}) lr=$LR | smoke=$SMOKE epochs=$EPOCHS warmup=$WARMUP_ITERS"
+echo "[t2m] per_gpu=$PER_GPU_BATCH global=$GLOBAL(=${PER_GPU_BATCH}x${NNODES}x${NPROC_PER_NODE}) lr=$LR sched=$LR_SCHEDULE lr_min=$LR_MIN | smoke=$SMOKE epochs=$EPOCHS warmup=$WARMUP_ITERS"
 echo "[t2m] master=${MASTER_ADDR:-<standalone>}:$MASTER_PORT nccl_ifname=${NCCL_SOCKET_IFNAME:-<n/a>} out=$OUT"
 echo "[t2m] text_mode=$TEXT_MODE amp=$AMP_DTYPE token_cache=${CAPTION_TOKEN_CACHE:-<none>} L=$CAPTION_TOKEN_MAX_LEN"
+echo "[t2m] latdyn: w_lat_dz=$W_LAT_DZ w_lat_ddz=$W_LAT_DDZ w_lat_x0=$W_LAT_X0 target=$LATENT_DYN_TARGET | spatial_mode=$SPATIAL_MODE"
 
 torchrun $RDZV_ARGS scripts/train_denoiser.py \
   --vae_ckpt "$VAE_CKPT" \
@@ -110,7 +121,12 @@ torchrun $RDZV_ARGS scripts/train_denoiser.py \
   --anytop_root "$ANYTOP_ROOT" \
   --max_frames 260 --max_joints 144 \
   --batch_size "$PER_GPU_BATCH" --lr "$LR" --epochs "$EPOCHS" \
-  --warmup_iters "$WARMUP_ITERS" \
+  --warmup_iters "$WARMUP_ITERS" --lr_schedule "$LR_SCHEDULE" --lr_min "$LR_MIN" \
+  --train_split "$TRAIN_SPLIT" \
+  --w_lat_dz "$W_LAT_DZ" --w_lat_ddz "$W_LAT_DDZ" --w_lat_x0 "$W_LAT_X0" \
+  --latent_dyn_target "$LATENT_DYN_TARGET" \
+  --spatial_mode "$SPATIAL_MODE" \
+  ${SPECIES_WHITELIST:+--species_whitelist "$SPECIES_WHITELIST"} \
   ${INIT_CKPT:+--init_ckpt "$INIT_CKPT"} \
   ${RESUME_CKPT:+--resume "$RESUME_CKPT"} \
   --n_layers "$N_LAYERS" --d_ff "$D_FF" --dropout 0.1 \
diff --git a/scripts/_launch_token_diffusion_8card_a100.sh b/scripts/_launch_token_diffusion_8card_a100.sh
index ce3fa7b..e69aa24 100644
--- a/scripts/_launch_token_diffusion_8card_a100.sh
+++ b/scripts/_launch_token_diffusion_8card_a100.sh
@@ -29,7 +29,20 @@ SMOKE="${SMOKE:-0}"
 # smoke-tune up. Goyal: global = PER_GPU_BATCH * 8, lr = 5e-4 * global / 48.
 PER_GPU_BATCH="${PER_GPU_BATCH:-8}"
 LR="${LR:-$(awk "BEGIN{printf \"%.3e\", 5e-4 * ($PER_GPU_BATCH*8) / 48}")}"
+LR_SCHEDULE="${LR_SCHEDULE:-constant}" # constant (default) | cosine (warmup→cosine→lr_min)
+LR_MIN="${LR_MIN:-0.0}"                # cosine floor (only used when LR_SCHEDULE=cosine)
+SPECIES_WHITELIST="${SPECIES_WHITELIST:-}"  # comma-sep object_types (capacity probe); empty=full 473
+TRAIN_SPLIT="${TRAIN_SPLIT:-train}"    # train (default) | all (train on all whitelisted clips incl val)
+# M2 latent temporal dynamics loss (handoff 20260605); ALL 0 = byte-equivalent.
+W_LAT_DZ="${W_LAT_DZ:-0}"
+W_LAT_DDZ="${W_LAT_DDZ:-0}"
+W_LAT_X0="${W_LAT_X0:-0}"
+LATENT_DYN_TARGET="${LATENT_DYN_TARGET:-sample}"
+SPATIAL_MODE="${SPATIAL_MODE:-graph}"  # graph (default) | plain (no_graph_spatial ablation)
+WARMUP_ITERS="${WARMUP_ITERS:-4000}"
+EPOCHS="${EPOCHS:-500}"
 OUT="${OUT:-runs/m2_token_cleanL2_bf16ep209_d512C128_n11ff1536_a100x8_seed42}"
+RESUME_CKPT="${RESUME_CKPT:-}"           # full crash/walltime resume (model+opt+epoch+global_it); inner passes --resume. cosine resume re-passes same lr_schedule/epochs.
 AMP_DTYPE="${AMP_DTYPE:-bf16}"
 TEXT_MODE="${TEXT_MODE:-token_cross_attn}"
 CAPTION_TOKEN_CACHE="${CAPTION_TOKEN_CACHE:-data/anytop_caption_t5_cleanL2_multi}"
@@ -50,11 +63,11 @@ flock -n 9 || { echo "[token-8card] ABORT: already running"; exit 0; }
 # same-node cross-alloc cgroup isolation and would route intra-node collectives
 # through slow host/NET). Matches the proven xnode VAE launcher. IB_HCA=mlx5_0
 # (ibdev2netdev: mlx5_0->ib0 Up on both nodes).
-COMMON_ENV="NNODES=2 NPROC_PER_NODE=4 MASTER_ADDR=$RDZV_HOST MASTER_PORT=$RDZV_PORT CVD=0,1,2,3 NCCL_P2P_DISABLE=0 NCCL_SHM_DISABLE=0 NCCL_IB_HCA=mlx5_0 PER_GPU_BATCH=$PER_GPU_BATCH LR=$LR OUT=$OUT SMOKE=$SMOKE AMP_DTYPE=$AMP_DTYPE TEXT_MODE=$TEXT_MODE CAPTION_TOKEN_CACHE=$CAPTION_TOKEN_CACHE CAPTION_TOKEN_MAX_LEN=$CAPTION_TOKEN_MAX_LEN VAE_CKPT=$VAE_CKPT"
+COMMON_ENV="NNODES=2 NPROC_PER_NODE=4 MASTER_ADDR=$RDZV_HOST MASTER_PORT=$RDZV_PORT CVD=0,1,2,3 NCCL_P2P_DISABLE=0 NCCL_SHM_DISABLE=0 NCCL_IB_HCA=mlx5_0 PER_GPU_BATCH=$PER_GPU_BATCH LR=$LR LR_SCHEDULE=$LR_SCHEDULE LR_MIN=$LR_MIN WARMUP_ITERS=$WARMUP_ITERS EPOCHS=$EPOCHS SPECIES_WHITELIST=$SPECIES_WHITELIST TRAIN_SPLIT=$TRAIN_SPLIT W_LAT_DZ=$W_LAT_DZ W_LAT_DDZ=$W_LAT_DDZ W_LAT_X0=$W_LAT_X0 LATENT_DYN_TARGET=$LATENT_DYN_TARGET SPATIAL_MODE=$SPATIAL_MODE OUT=$OUT RESUME_CKPT=$RESUME_CKPT SMOKE=$SMOKE AMP_DTYPE=$AMP_DTYPE TEXT_MODE=$TEXT_MODE CAPTION_TOKEN_CACHE=$CAPTION_TOKEN_CACHE CAPTION_TOKEN_MAX_LEN=$CAPTION_TOKEN_MAX_LEN VAE_CKPT=$VAE_CKPT"
 
 echo "[token-8card] $(date '+%F %T %Z') cross-node 8-card A100 DDP: $JOB_A(1004,r0)+$JOB_B(1001,r1) via $RDZV_HOST:$RDZV_PORT smoke=$SMOKE"
 echo "[token-8card] text_mode=$TEXT_MODE amp=$AMP_DTYPE vae=$VAE_CKPT token_cache=$CAPTION_TOKEN_CACHE L=$CAPTION_TOKEN_MAX_LEN"
-echo "[token-8card] global=$(( PER_GPU_BATCH*8 )) (8xbs$PER_GPU_BATCH) lr=$LR out=$OUT"
+echo "[token-8card] global=$(( PER_GPU_BATCH*8 )) (8xbs$PER_GPU_BATCH) lr=$LR sched=$LR_SCHEDULE/lr_min=$LR_MIN warmup=$WARMUP_ITERS epochs=$EPOCHS out=$OUT"
 
 # One torchrun group per alloc; static rendezvous joins them into 8 global ranks.
 # Explicit --gres/--cpus so each srun step gets its alloc's 4 GPUs + CPU for 4 ranks
diff --git a/scripts/animate_denoiser.py b/scripts/animate_denoiser.py
index c9b60fd..779088f 100644
--- a/scripts/animate_denoiser.py
+++ b/scripts/animate_denoiser.py
@@ -7,7 +7,8 @@ Pipeline per docs/phase2_diffusion_design.md §4-5:
        pooled_skeleton_embeddings / anchor_indices / hard_assignment / assignment / s_j
      - frame_mask_lat = batch.frame_mask.view(B, T_lat, stride).all(-1)
      - z_T = N(0, I) of shape [B, T_lat, C, D]
-     - DDIM sampling loop (default 50 steps) with CFG (cond_scale=7.5):
+     - DDIM sampling loop (default 50 steps) with CFG (default cond_scale=1.5;
+       ALWAYS pass --cond_scale explicitly when comparing renders):
          z2 = cat(z, z, dim=0); t2 = cat(t, t); has_text2 = cat(True, False);
          text2 = cat(text, text*0); other tensors all repeated to 2B
          v_2 = denoiser(z_2, ...)  → split into v_cond / v_uncond
@@ -65,11 +66,13 @@ def load_denoiser(ckpt_path: str, dev: torch.device) -> tuple[GraphSaladDenoiser
     # token ckpts carry 'token_cross_attn' in args). Wrong mode ⇒ arch mismatch ⇒
     # strict-load fails loud below.
     text_mode = da.get("text_mode", "mean_additive")
+    spatial_mode = da.get("spatial_mode", "graph")  # old ckpts (no key) → graph
     denoiser = GraphSaladDenoiser(
         d_model=d_model, n_heads=n_heads, d_ff=d_ff,
         n_layers=da.get("n_layers", 5),
         d_text=768, dropout=da.get("dropout", 0.1),
         text_mode=text_mode, text_token_dim=768,
+        spatial_mode=spatial_mode,
     ).to(dev)
     missing, unexpected = denoiser.load_state_dict(ck["model_state_dict"], strict=True)
     if missing or unexpected:
@@ -116,17 +119,20 @@ def ddim_sample(
     has_text_cond = batch.has_text.to(dev)              # [B] bool
     has_text_uncond = torch.zeros_like(has_text_cond, dtype=torch.bool)
     has_text2 = torch.cat([has_text_cond, has_text_uncond], dim=0)  # [2B]
-    # M2: mode-dependent text. token_cross_attn repeats tokens + token mask; the
-    # uncond half's has_text=False fully masks the text keys (cross-attn → 0).
-    token_mode = (getattr(denoiser, "text_mode", "mean_additive") == "token_cross_attn")
-    if token_mode:
-        ttok = batch.caption_token_emb.to(dev)          # [B, L, 768]
-        tmask = batch.caption_token_mask.to(dev)        # [B, L] bool
-        text2 = ttok.repeat(2, 1, 1)                    # [2B, L, 768]
-        token_mask2 = tmask.repeat(2, 1)               # [2B, L] (uncond gated by has_text)
-    else:
-        text_emb = batch.caption_emb.to(dev)            # [B, 768]
-        text2 = text_emb.repeat(2, 1)                  # [2B, 768]
+    # M2: mode-dependent text, repeated 2x for the CFG cond+uncond batch. The uncond
+    # half's has_text=False zeroes the global add AND fully masks the token keys
+    # (cross-attn → 0), so both streams CFG-drop together (dual_text).
+    text_mode = getattr(denoiser, "text_mode", "mean_additive")
+    text_tokens2 = None
+    if text_mode == "dual_text":
+        text2 = batch.caption_emb.to(dev).repeat(2, 1)                   # [2B, 768] global
+        text_tokens2 = batch.caption_token_emb.to(dev).repeat(2, 1, 1)  # [2B, L, 768] tokens
+        token_mask2 = batch.caption_token_mask.to(dev).repeat(2, 1)     # [2B, L]
+    elif text_mode == "token_cross_attn":
+        text2 = batch.caption_token_emb.to(dev).repeat(2, 1, 1)         # [2B, L, 768]
+        token_mask2 = batch.caption_token_mask.to(dev).repeat(2, 1)     # [2B, L]
+    else:  # mean_additive
+        text2 = batch.caption_emb.to(dev).repeat(2, 1)                  # [2B, 768]
         token_mask2 = None
 
     first = True
@@ -141,6 +147,7 @@ def ddim_sample(
             pooled_skeleton_embeddings=skel2,
             has_text=has_text2,
             text_token_mask=token_mask2,
+            text_tokens=text_tokens2,
             validate_inputs=first,  # cold-start validate on first iter
         )
         first = False
@@ -176,17 +183,25 @@ def make_fake_enc(z: torch.Tensor, skel: dict, frame_mask_lat: torch.Tensor) ->
 
 
 def make_t2m_large_gif(pred_ric, pred_fk, static_pose, parents, prompt, out_path,
-                       max_frames=48, fps=12, cell=(900, 760), zoom=1.15, pad=0.06):
+                       max_frames=48, fps=12, cell=(900, 760), zoom=1.15, pad=0.06,
+                       gt=None):
     """Large-figure (PIL oblique, per-frame root-centered) T2M demo:
     input skeleton (static grey) | PRED pose/RIC (blue) | PRED rot6d-FK (green),
     with the prompt as a top header band. NO GT (T2M input = skeleton + prompt).
-    recover already done by caller via src funcs; this is geometry/drawing only."""
+    recover already done by caller via src funcs; this is geometry/drawing only.
+
+    Diagnostic (gt given, [T,J,3] world-space): the animated GT source motion is
+    stitched on as the RIGHTMOST panel → input | PRED_RIC | PRED_FK | GT, so
+    generated motion can be eyeballed against the real dataset clip. GT shares the
+    panels' common scale, so a fast/janky PRED reads visually against a smooth GT."""
     import scripts._pil_skeleton_render as pr
     T = pred_ric.shape[0]
     static_T = np.repeat(np.asarray(static_pose)[None], T, axis=0)   # [J,3] -> [T,J,3]
     arrs = [(static_T, "input skeleton (rest)", (90, 90, 90), True, True),
             (pred_ric, "PRED pose/RIC 0:3", (35, 112, 180), False, False),
             (pred_fk, "PRED rot6d-FK 3:9", (30, 150, 55), False, False)]
+    if gt is not None:
+        arrs.append((np.asarray(gt), "GT source 0:3", (200, 60, 60), False, False))
     norm = []
     for a, title, color, axes, static in arrs:
         aa = a.astype(np.float64).copy(); aa[..., 1] -= aa[..., 1].min()
@@ -309,7 +324,7 @@ def main() -> int:
                     help="comma-separated species to render")
     ap.add_argument("--n_per", type=int, default=2)
     ap.add_argument("--n_ddim_steps", type=int, default=50)
-    ap.add_argument("--cond_scale", type=float, default=7.5)
+    ap.add_argument("--cond_scale", type=float, default=1.5)
     ap.add_argument("--stride", type=int, default=2)
     ap.add_argument("--fps", type=int, default=8)
     ap.add_argument("--seed", type=int, default=42)
@@ -318,6 +333,10 @@ def main() -> int:
                     help="big PIL figures (input|PRED_RIC|PRED_FK + prompt) via _pil_skeleton_render")
     ap.add_argument("--generic_prompt", action="store_true",
                     help="replace species name with 'an animal' (keep action), re-encode via T5-base")
+    ap.add_argument("--with_gt", action="store_true",
+                    help="diagnostic: prepend the GT source-motion panel "
+                         "(GT 0:3 | PRED_RIC | PRED_FK) so generated motion can be "
+                         "eyeballed against the real dataset clip. --large only.")
     args = ap.parse_args()
 
     if args.device == "cuda" and not torch.cuda.is_available():
@@ -375,10 +394,10 @@ def main() -> int:
     # M2: token ckpts need the token cache + return_caption_tokens so the dataset
     # emits caption_token_emb/mask aligned to the same caption idx as caption_emb.
     da_text_mode = da.get("text_mode", "mean_additive")
-    use_tokens = (da_text_mode == "token_cross_attn")
+    use_tokens = da_text_mode in ("token_cross_attn", "dual_text")
     if use_tokens and not args.caption_token_cache:
         raise SystemExit(
-            "denoiser ckpt is token_cross_attn but --caption_token_cache not "
+            f"denoiser ckpt is {da_text_mode} but --caption_token_cache not "
             "given (need the token cache to sample dataset captions)."
         )
     ds_kwargs = dict(
@@ -399,7 +418,12 @@ def main() -> int:
     # AnyTopDataset zero-fills caption_emb + sets has_text=False on cache miss).
     n_missing = 0
     want_set = set(s.strip() for s in args.species.split(",") if s.strip())
-    for i in range(len(ds)):
+    # Only touch clips of the requested species. object_type lives in the dataset
+    # index (ds.samples[i]) and needs NO motion load, so a species-filtered render
+    # iterates ~dozens of clips instead of walking the whole split (train=77882).
+    match_indices = [i for i, s in enumerate(ds.samples)
+                     if s.get("object_type") in want_set]
+    for i in match_indices:
         it = ds[i]
         if it["object_type"] not in want_set:
             continue
@@ -429,10 +453,12 @@ def main() -> int:
     summary: list[str] = []
 
     print(f"\nSampling: DDIM {args.n_ddim_steps} steps, CFG cond_scale={args.cond_scale}")
-    for i in range(len(ds)):
+    for i in match_indices:  # species-filtered (see preflight); no full-split walk
         item = ds[i]
         sp = item["object_type"]
         if sp not in picked or picked[sp] >= args.n_per:
+            if all(picked[s] >= args.n_per for s in want):
+                break  # all requested species collected → stop (don't walk all of train)
             continue
         raw = anytop_collate_fn([item])
         raw = {k: v.to(dev) if torch.is_tensor(v) else v for k, v in raw.items()}
@@ -440,13 +466,13 @@ def main() -> int:
         gen_caption = None
         if args.generic_prompt:
             gen_caption = make_generic_caption(item.get("caption") or "", sp)
-            if use_tokens:
-                # token mode: encode token-level T5 (NOT mean-pool); override both
-                # caption_token_emb + mask (plan §3.7).
+            # dual_text re-encodes BOTH streams; token/mean re-encode only their own.
+            if da_text_mode in ("token_cross_attn", "dual_text"):
+                # encode token-level T5 (NOT mean-pool); override token emb + mask.
                 te, tm = _t5_encode_tokens(gen_caption, dev, args.caption_token_max_len)
                 batch.caption_token_emb = te                        # [1,L,768]
                 batch.caption_token_mask = tm                       # [1,L] bool
-            else:
+            if da_text_mode in ("mean_additive", "dual_text"):
                 batch.caption_emb = _t5_encode(gen_caption, dev)    # override [1,768]
             batch.has_text = torch.ones_like(batch.has_text)        # force conditioned
 
@@ -520,6 +546,7 @@ def main() -> int:
             make_t2m_large_gif(
                 pred_world, pred_world_fk, static_pose, parents, prompt_text,
                 str(actual_gif_path), fps=args.fps,
+                gt=(gt_world if args.with_gt else None),
             )
         else:
             animate_t2m_input_pred(
diff --git a/scripts/train_denoiser.py b/scripts/train_denoiser.py
index aa21f85..ddcef88 100644
--- a/scripts/train_denoiser.py
+++ b/scripts/train_denoiser.py
@@ -26,6 +26,7 @@ import json
 import os
 import sys
 import contextlib
+import math
 import time
 from collections import defaultdict
 from pathlib import Path
@@ -130,6 +131,59 @@ def masked_v_mse(v_pred: torch.Tensor, v_target: torch.Tensor,
     return diff_sq.sum() / denom.clamp(min=1.0)
 
 
+# ---------------------------------------------------------------------------
+# M2 latent temporal dynamics loss (handoff/20260605_latent_temporal_dynamics_loss_experiment.md):
+# penalise the temporal derivatives of the v-implied clean latent z0_hat so the
+# sampled latent SEQUENCE moves through time like the real latent (the v-MSE only
+# supervises one-step velocity, not the cross-time z0 trajectory). All extra
+# losses are computed in fp32 with the SAME valid (coarse × frame) semantics as
+# masked_v_mse, and reduce to a no-op when all weights are 0.
+# ---------------------------------------------------------------------------
+
+def predict_z0_from_v(z_t: torch.Tensor, v_pred: torch.Tensor,
+                      timesteps: torch.Tensor, scheduler) -> torch.Tensor:
+    """v-prediction → implied clean latent: z0 = √ᾱ_t·z_t − √(1−ᾱ_t)·v."""
+    alphas = scheduler.alphas_cumprod.to(device=z_t.device, dtype=z_t.dtype)
+    a = alphas[timesteps].sqrt().view(-1, 1, 1, 1)
+    b = (1.0 - alphas[timesteps]).sqrt().view(-1, 1, 1, 1)
+    return a * z_t - b * v_pred
+
+
+def masked_latent_mse(pred: torch.Tensor, target: torch.Tensor,
+                      mask: torch.Tensor) -> torch.Tensor:
+    """fp32 masked MSE over valid positions × feature dim (mask is [...,1])."""
+    mask_f = mask.float()
+    diff_sq = (pred.float() - target.float()).pow(2) * mask_f
+    return diff_sq.sum() / (mask_f.sum() * pred.shape[-1]).clamp(min=1.0)
+
+
+def masked_latent_dz_mse(z0_hat: torch.Tensor, z0_target: torch.Tensor,
+                         coarse_mask: torch.Tensor, frame_mask: torch.Tensor) -> torch.Tensor:
+    """Latent velocity (first temporal difference) MSE; valid = both frames valid."""
+    dz_p = z0_hat[:, 1:] - z0_hat[:, :-1]
+    dz_t = z0_target[:, 1:] - z0_target[:, :-1]
+    m = (
+        coarse_mask[:, None, :, None]
+        & frame_mask[:, 1:, None, None]
+        & frame_mask[:, :-1, None, None]
+    )
+    return masked_latent_mse(dz_p, dz_t, m)
+
+
+def masked_latent_ddz_mse(z0_hat: torch.Tensor, z0_target: torch.Tensor,
+                          coarse_mask: torch.Tensor, frame_mask: torch.Tensor) -> torch.Tensor:
+    """Latent acceleration (second temporal difference) MSE; valid = all 3 frames valid."""
+    ddz_p = z0_hat[:, 2:] - 2.0 * z0_hat[:, 1:-1] + z0_hat[:, :-2]
+    ddz_t = z0_target[:, 2:] - 2.0 * z0_target[:, 1:-1] + z0_target[:, :-2]
+    m = (
+        coarse_mask[:, None, :, None]
+        & frame_mask[:, 2:, None, None]
+        & frame_mask[:, 1:-1, None, None]
+        & frame_mask[:, :-2, None, None]
+    )
+    return masked_latent_mse(ddz_p, ddz_t, m)
+
+
 # ---------------------------------------------------------------------------
 # Preflight: caption coverage on train + val
 # ---------------------------------------------------------------------------
@@ -210,12 +264,45 @@ def parse_args() -> argparse.Namespace:
                          "species — intentional, val measures denoise quality "
                          "on hardest skeletons. Mirrors train_graph_vae.py "
                          "--full_data_val_species (2026-05-24).")
+    ap.add_argument("--species_whitelist", type=str, default=None,
+                    help="comma-separated object_types; restrict BOTH train and val "
+                         "to this subset (e.g. a 20-species capacity probe). Applied "
+                         "after the normal per-species split, so train/val stay the "
+                         "same per-species holdout, just narrowed to these species.")
+    ap.add_argument("--train_split", default="train", choices=["train", "all"],
+                    help="which split ds_train uses (default-mode only). 'train' "
+                         "(default) = the per-species train holdout; 'all' = ALL "
+                         "clips of the (whitelisted) species incl. the val subset "
+                         "(pure-fitting / capacity probe: train on all, eval on val).")
     # Optim
     ap.add_argument("--epochs", type=int, default=500)
     ap.add_argument("--batch_size", type=int, default=16)
     ap.add_argument("--lr", type=float, default=5e-4)
     ap.add_argument("--weight_decay", type=float, default=1e-6)
     ap.add_argument("--warmup_iters", type=int, default=2000)
+    ap.add_argument("--lr_schedule", default="constant",
+                    choices=["constant", "cosine"],
+                    help="post-warmup LR schedule. 'constant' (default, unchanged): "
+                         "hold args.lr. 'cosine': decay args.lr -> lr_min over "
+                         "[warmup_iters, epochs*steps_per_epoch].")
+    ap.add_argument("--lr_min", type=float, default=0.0,
+                    help="cosine floor LR (only used when --lr_schedule cosine).")
+    # M2 latent temporal dynamics loss (handoff/20260605_latent_temporal_dynamics_loss_experiment.md):
+    # extra penalty on the implied clean latent z0_hat's temporal derivatives.
+    # ALL default 0.0 -> loss path byte-equivalent to the existing masked_v_mse
+    # (full-data runs unaffected; the only variable is the extra loss term).
+    ap.add_argument("--w_lat_dz", type=float, default=0.0,
+                    help="weight on latent velocity loss ||Δz0_hat - Δz0||² over "
+                         "latent time (0 = off, current behavior).")
+    ap.add_argument("--w_lat_ddz", type=float, default=0.0,
+                    help="weight on latent acceleration loss ||Δ²z0_hat - Δ²z0||² (0 = off).")
+    ap.add_argument("--w_lat_x0", type=float, default=0.0,
+                    help="weight on direct latent loss ||z0_hat - z0||² (0 = off; "
+                         "keep 0 in the first dynamics run per the handoff).")
+    ap.add_argument("--latent_dyn_target", default="sample", choices=["sample", "mu"],
+                    help="latent-dynamics loss target: 'sample' (z0 ~ posterior, "
+                         "matches the v-target; default) or 'mu' (posterior mean, "
+                         "less noisy fallback if sample makes the loss unstable).")
     ap.add_argument("--grad_clip", type=float, default=1.0)
     # Denoiser arch
     ap.add_argument("--n_layers", type=int, default=5)
@@ -230,12 +317,20 @@ def parse_args() -> argparse.Namespace:
     # M2 token-level text conditioning (optional). Default mean_additive keeps the
     # current behavior + old-ckpt strict-load. token_cross_attn requires the token
     # cache built by scripts/precompute_t5_caption_tokens.py.
-    ap.add_argument("--text_mode", choices=["mean_additive", "token_cross_attn"],
+    ap.add_argument("--text_mode",
+                    choices=["mean_additive", "token_cross_attn", "dual_text"],
                     default="mean_additive",
                     help="mean_additive (default): mean-pooled T5 additive broadcast "
                          "(byte-equiv to current; old ckpts strict-load). "
                          "token_cross_attn: per-layer cross-attention over token-level "
-                         "T5 (needs --caption_token_cache).")
+                         "T5. dual_text: BOTH streams — global mean-add + token "
+                         "cross-attn (both CFG-gated by has_text). "
+                         "token_cross_attn/dual_text need --caption_token_cache.")
+    ap.add_argument("--spatial_mode", choices=["graph", "plain"], default="graph",
+                    help="backbone spatial attention. 'graph' (default): graph-aware "
+                         "(adjacency+geodesic bias); old ckpts strict-load. 'plain': "
+                         "no_graph_spatial ablation — plain slot self-attn (no topo "
+                         "bias), still node-masked + pooled_skeleton additive kept.")
     ap.add_argument("--caption_token_cache", default=None,
                     help="prefix of the token cache (<prefix>.tokens.npy + "
                          ".token_mask.npy + .keys.json). REQUIRED when "
@@ -367,13 +462,17 @@ def main() -> int:
     # ---- Dataset ----
     # M2: token mode needs the offline token cache + return_caption_tokens on both
     # train + val datasets (val uses primary caption idx 0 deterministically).
-    use_tokens = (args.text_mode == "token_cross_attn")
+    use_tokens = args.text_mode in ("token_cross_attn", "dual_text")
     if use_tokens and args.caption_token_cache is None:
         raise SystemExit(
-            "--text_mode token_cross_attn requires --caption_token_cache "
+            f"--text_mode {args.text_mode} requires --caption_token_cache "
             "(<prefix>.tokens.npy + .token_mask.npy + .keys.json from "
             "scripts/precompute_t5_caption_tokens.py)."
         )
+    species_whitelist = (
+        [s.strip() for s in args.species_whitelist.split(",") if s.strip()]
+        if args.species_whitelist else None
+    )
     ds_kwargs = dict(
         num_frames=args.max_frames,
         max_joints=ta.get("max_joints", args.max_joints),
@@ -382,6 +481,7 @@ def main() -> int:
         caption_token_cache=args.caption_token_cache,
         return_caption_tokens=use_tokens,
         caption_token_max_len=args.caption_token_max_len,
+        species_whitelist=species_whitelist,
     )
     if args.anytop_root or ta.get("anytop_root"):
         ds_kwargs["data_root"] = args.anytop_root or ta["anytop_root"]
@@ -429,7 +529,7 @@ def main() -> int:
         # random_crop=False for symmetry with full_data mode. random_caption=True
         # for train preserved (multi-cap diversity unchanged).
         ds_train = AnyTopDataset(
-            split="train", random_caption=True, random_crop=False, **ds_kwargs)
+            split=args.train_split, random_caption=True, random_crop=False, **ds_kwargs)
         ds_val = AnyTopDataset(
             split="val", random_caption=False, random_crop=False, **ds_kwargs)
         log(f"  ds_train={len(ds_train)} (random_caption=True, random_crop=False)"
@@ -513,10 +613,11 @@ def main() -> int:
         d_model=d_model, n_heads=n_heads, d_ff=d_ff,
         n_layers=args.n_layers, d_text=768, dropout=args.dropout,
         text_mode=args.text_mode, text_token_dim=768,
+        spatial_mode=args.spatial_mode,
     ).to(dev)
     n_params = sum(p.numel() for p in denoiser.parameters())
     log(f"\nDenoiser: n_layers={args.n_layers} d_model={d_model} d_ff={d_ff} "
-        f"text_mode={args.text_mode} params={n_params:,}")
+        f"text_mode={args.text_mode} spatial_mode={args.spatial_mode} params={n_params:,}")
 
     # ---- Full resume (--resume): restore MODEL here (before DDP wrap); optimizer
     # + epoch + best_val + global_it are restored further below. Seamless crash
@@ -539,6 +640,13 @@ def main() -> int:
                 f"--text_mode {args.text_mode!r}. Rebuild with the matching "
                 f"text_mode (token/mean arch differ)."
             )
+        ck_spatial_mode = resume_ck.get("args", {}).get("spatial_mode", "graph")
+        if ck_spatial_mode != args.spatial_mode:
+            raise SystemExit(
+                f"[RESUME FAIL] ckpt spatial_mode={ck_spatial_mode!r} != CLI "
+                f"--spatial_mode {args.spatial_mode!r}. graph/plain arch differ "
+                f"(plain drops adjacency/geodesic bias params)."
+            )
         missing, unexpected = denoiser.load_state_dict(resume_ck["model_state_dict"], strict=True)
         if missing or unexpected:
             raise SystemExit(
@@ -566,6 +674,12 @@ def main() -> int:
                 f"[INIT_CKPT FAIL] ckpt text_mode={ck_text_mode!r} != CLI "
                 f"--text_mode {args.text_mode!r}. Rebuild with matching text_mode."
             )
+        ck_spatial_mode = ck.get("args", {}).get("spatial_mode", "graph")
+        if ck_spatial_mode != args.spatial_mode:
+            raise SystemExit(
+                f"[INIT_CKPT FAIL] ckpt spatial_mode={ck_spatial_mode!r} != CLI "
+                f"--spatial_mode {args.spatial_mode!r}. graph/plain arch differ."
+            )
         sd = ck.get("model_state_dict", ck)
         missing, unexpected = denoiser.load_state_dict(sd, strict=True)
         if missing or unexpected:
@@ -613,9 +727,30 @@ def main() -> int:
         clip_sample=False,
     )
 
+    # Total per-rank optimizer steps over the whole run (cosine horizon). len(dl_train)
+    # is per-rank steps/epoch under the DistributedSampler (drop_last=True), and
+    # global_it is the per-rank counter, so both live in the same (per-rank) iter
+    # space. Honour --smoke (1 epoch) so a cosine smoke decays over its real
+    # one-epoch horizon (codex 019e95f0 #1). NOTE for cosine --resume: global_it is
+    # rebuilt from start_epoch*len(dl_train), so a manual resume MUST re-pass the
+    # SAME --lr_schedule/--lr_min/--epochs (+batch/world/data) or the phase shifts
+    # (codex #3). The orchestrator re-passes all of these via COMMON_ENV, so a
+    # relaunch through it is safe.
+    total_iters = (1 if args.smoke else args.epochs) * len(dl_train)
+
     def lr_for(it: int) -> float:
+        # Linear warmup (unchanged): ramp 0 -> args.lr over warmup_iters.
         if args.warmup_iters > 0 and it < args.warmup_iters:
             return args.lr * (it + 1) / args.warmup_iters
+        # Post-warmup: constant (default, byte-identical to before) or cosine decay.
+        if args.lr_schedule == "cosine":
+            # Endpoint-exact: the last ACTIVE step is it=total_iters-1 (lr_for runs
+            # before global_it increments), so denom uses total_iters-1 → progress=1
+            # → lr_min lands on the final step, not a virtual one (codex #2).
+            denom = max(1, total_iters - 1 - args.warmup_iters)
+            progress = min(1.0, max(0.0, (it - args.warmup_iters) / denom))
+            return args.lr_min + 0.5 * (args.lr - args.lr_min) * (
+                1.0 + math.cos(math.pi * progress))
         return args.lr
 
     metrics_fp = open(out_dir / "metrics.jsonl", "w") if is_main else None
@@ -642,6 +777,9 @@ def main() -> int:
     log(f"\nTraining for {epochs} epochs (smoke={args.smoke})")
     log(f"steps per epoch: {len(dl_train)}"
         + (f" (per-rank, world_size={world_size})" if is_ddp else ""))
+    log(f"LR schedule: {args.lr_schedule} (peak={args.lr:.3e} warmup={args.warmup_iters}"
+        + (f" → cosine → lr_min={args.lr_min:.3e} over total_iters={total_iters}"
+           if args.lr_schedule == "cosine" else " then constant") + ")")
 
     for epoch in range(start_epoch, epochs):
         if is_ddp:
@@ -649,6 +787,9 @@ def main() -> int:
         denoiser.train()
         t_ep = time.time()
         ep_losses = []
+        # M2 latent-dynamics loss: track components separately for logging.
+        lat_active = bool(args.w_lat_dz > 0 or args.w_lat_ddz > 0 or args.w_lat_x0 > 0)
+        ep_v_mse, ep_lat_dz, ep_lat_ddz, ep_lat_x0 = [], [], [], []
         for batch_idx, raw in enumerate(dl_train):
             # device transfer
             raw = {k: v.to(dev) if torch.is_tensor(v) else v for k, v in raw.items()}
@@ -673,13 +814,21 @@ def main() -> int:
             ht_in = batch.has_text.to(dev) if batch.has_text.device != dev else batch.has_text
             drop_mask = torch.rand(B, device=dev) < args.cond_drop_prob
             has_text = ht_in & (~drop_mask)
-            if use_tokens:
-                # token mode: pass token hidden states [B,L,768] + raw token mask
-                # [B,L]. The denoiser builds key_padding_mask = ~(mask & has_text),
-                # so has_text=False rows are fully masked → cross-attn output 0
-                # (CFG-uncond). Mask drives the gate (plan §3.6); no zero-multiply
-                # of the embedding is needed (and would be wrong — softmax over
-                # all-(-1e9) handled by TextCrossAttention's zero-output path).
+            # Text inputs by mode. dual_text = BOTH streams, CFG-dropped together
+            # (global pre-gated by has_text; token gated via key_padding_mask).
+            text_tokens_in = None
+            if args.text_mode == "dual_text":
+                # global stream via `text` (pre-gated like mean mode) + token stream
+                # via `text_tokens` (masked-gated like token mode).
+                text_in = batch.caption_emb.to(dev) * has_text[:, None].to(batch.caption_emb.dtype)
+                text_tokens_in = batch.caption_token_emb.to(dev)  # [B,L,768]
+                token_mask_in = batch.caption_token_mask.to(dev)  # [B,L] bool
+            elif use_tokens:
+                # token mode: tokens via `text` [B,L,768] + raw token mask [B,L]. The
+                # denoiser builds key_padding_mask = ~(mask & has_text), so
+                # has_text=False rows are fully masked → cross-attn output 0
+                # (CFG-uncond). Mask drives the gate; no zero-multiply of the
+                # embedding (softmax over all-(-1e9) → TextCrossAttention zero path).
                 text_in = batch.caption_token_emb.to(dev)         # [B,L,768]
                 token_mask_in = batch.caption_token_mask.to(dev)  # [B,L] bool
             else:
@@ -710,10 +859,28 @@ def main() -> int:
                     pooled_skeleton_embeddings=pooled_skel,
                     has_text=has_text,
                     text_token_mask=token_mask_in,
+                    text_tokens=text_tokens_in,
                     # Validate on the first iter only (cold-start preflight)
                     validate_inputs=(global_it == 0),
                 )
-                loss = masked_v_mse(v_pred, v_target, coarse_mask, frame_mask)
+                loss_v = masked_v_mse(v_pred, v_target, coarse_mask, frame_mask)
+                loss = loss_v
+                # M2 latent temporal dynamics loss (handoff 20260605). Gated on
+                # weights>0 → zero-weight path is byte-identical (loss == loss_v).
+                loss_dz = loss_ddz = loss_x0 = None
+                if args.w_lat_dz > 0 or args.w_lat_ddz > 0 or args.w_lat_x0 > 0:
+                    z0_hat = predict_z0_from_v(z_t.float(), v_pred.float(), timesteps, sched)
+                    z0_dyn_target = (enc["mu"].float() * mask_4d
+                                     if args.latent_dyn_target == "mu" else z0)
+                    if args.w_lat_x0 > 0:
+                        loss_x0 = masked_latent_mse(z0_hat, z0_dyn_target, mask_4d.bool())
+                        loss = loss + args.w_lat_x0 * loss_x0
+                    if args.w_lat_dz > 0:
+                        loss_dz = masked_latent_dz_mse(z0_hat, z0_dyn_target, coarse_mask, frame_mask)
+                        loss = loss + args.w_lat_dz * loss_dz
+                    if args.w_lat_ddz > 0:
+                        loss_ddz = masked_latent_ddz_mse(z0_hat, z0_dyn_target, coarse_mask, frame_mask)
+                        loss = loss + args.w_lat_ddz * loss_ddz
 
             # P3 fail-fast (2026-05-23): a NaN/Inf loss means upstream maths
             # diverged (bad lr / bad scheduler / nan input). Crashing here
@@ -743,11 +910,26 @@ def main() -> int:
             opt.step()
 
             ep_losses.append(loss.item())
+            if lat_active:
+                ep_v_mse.append(loss_v.item())
+                if loss_dz is not None:
+                    ep_lat_dz.append(loss_dz.item())
+                if loss_ddz is not None:
+                    ep_lat_ddz.append(loss_ddz.item())
+                if loss_x0 is not None:
+                    ep_lat_x0.append(loss_x0.item())
             global_it += 1
 
         epoch_loss = float(np.mean(ep_losses))
         ep_dt = time.time() - t_ep
-        log(f"\n=== epoch {epoch} done in {ep_dt:.1f}s | train_loss={epoch_loss:.4f} "
+        # M2 latent-dynamics component means (0.0 when inactive / term off).
+        epoch_v_mse = float(np.mean(ep_v_mse)) if ep_v_mse else epoch_loss
+        epoch_lat_dz = float(np.mean(ep_lat_dz)) if ep_lat_dz else 0.0
+        epoch_lat_ddz = float(np.mean(ep_lat_ddz)) if ep_lat_ddz else 0.0
+        epoch_lat_x0 = float(np.mean(ep_lat_x0)) if ep_lat_x0 else 0.0
+        comp_str = (f" v_mse={epoch_v_mse:.4f} lat_dz={epoch_lat_dz:.4f} "
+                    f"lat_ddz={epoch_lat_ddz:.4f}") if lat_active else ""
+        log(f"\n=== epoch {epoch} done in {ep_dt:.1f}s | train_loss={epoch_loss:.4f}{comp_str} "
             f"lr={cur_lr:.2e} n_iter={len(ep_losses)} ===")
 
         # Val — only rank 0 runs (full val set, no metric all-reduce needed).
@@ -767,6 +949,9 @@ def main() -> int:
                 # of per-batch element-weighted means (codex P2-3).
                 val_num = 0.0
                 val_den = 0.0
+                # M2 latent-dynamics component logging (diagnostic only; the
+                # best-ckpt gate stays val_denoise). Batch-mean of per-batch means.
+                vdz_list, vddz_list, vx0_list = [], [], []
                 with torch.no_grad():
                     for raw in dl_val:
                         raw = {k: v.to(dev) if torch.is_tensor(v) else v for k, v in raw.items()}
@@ -791,7 +976,12 @@ def main() -> int:
                         mask_f = mask.to(z0.dtype)
                         z_t = z_t * mask_f; v_target = v_target * mask_f
                         has_text = batch.has_text.to(dev) if batch.has_text.device != dev else batch.has_text
-                        if use_tokens:
+                        text_tokens_in = None
+                        if args.text_mode == "dual_text":
+                            text_in = batch.caption_emb.to(dev) * has_text[:, None].to(batch.caption_emb.dtype)
+                            text_tokens_in = batch.caption_token_emb.to(dev)
+                            token_mask_in = batch.caption_token_mask.to(dev)
+                        elif use_tokens:
                             text_in = batch.caption_token_emb.to(dev)
                             token_mask_in = batch.caption_token_mask.to(dev)
                         else:
@@ -808,11 +998,26 @@ def main() -> int:
                                 pooled_skeleton_embeddings=pooled_skel,
                                 has_text=has_text, validate_inputs=False,
                                 text_token_mask=token_mask_in,
+                                text_tokens=text_tokens_in,
                             )
                         diff_sq = (v_pred.float() - v_target).pow(2) * mask_f
                         val_num += diff_sq.sum().item()
                         val_den += mask_f.sum().item() * v_pred.shape[-1]
+                        if lat_active:
+                            z0_hat = predict_z0_from_v(z_t.float(), v_pred.float(), timesteps, sched)
+                            z0_dyn_target = (enc["mu"].float() * mask_f
+                                             if args.latent_dyn_target == "mu" else z0)
+                            vdz_list.append(masked_latent_dz_mse(
+                                z0_hat, z0_dyn_target, coarse_mask, frame_mask).item())
+                            vddz_list.append(masked_latent_ddz_mse(
+                                z0_hat, z0_dyn_target, coarse_mask, frame_mask).item())
+                            if args.w_lat_x0 > 0:
+                                vx0_list.append(masked_latent_mse(
+                                    z0_hat, z0_dyn_target, mask_f.bool()).item())
                 val_loss = val_num / max(val_den, 1.0)
+                val_lat_dz = float(np.mean(vdz_list)) if vdz_list else 0.0
+                val_lat_ddz = float(np.mean(vddz_list)) if vddz_list else 0.0
+                val_lat_x0 = float(np.mean(vx0_list)) if vx0_list else 0.0
                 # Codex P2 (2026-05-23): fail-fast on non-finite val too.
                 if not (val_loss == val_loss and val_loss != float("inf")
                         and val_loss != float("-inf")):
@@ -820,13 +1025,28 @@ def main() -> int:
                         f"[FAIL] non-finite val_loss={val_loss!r} at epoch={epoch}. "
                         f"Inspect last train iter for upstream divergence."
                     )
-                log(f"[val ep{epoch}] dt={time.time()-t_v:.1f}s val_denoise={val_loss:.4f} "
+                val_comp_str = (f" val_lat_dz={val_lat_dz:.4f} val_lat_ddz={val_lat_ddz:.4f}"
+                                if lat_active else "")
+                log(f"[val ep{epoch}] dt={time.time()-t_v:.1f}s val_denoise={val_loss:.4f}{val_comp_str} "
                     f"n_valid_positions={int(val_den/v_pred.shape[-1])}")
 
-                metrics_fp.write(json.dumps({
+                metrics_row = {
                     "epoch": epoch, "train_loss": epoch_loss, "val_denoise": val_loss,
                     "lr": cur_lr, "epoch_dt_s": ep_dt, "global_it": global_it,
-                }) + "\n"); metrics_fp.flush()
+                }
+                if lat_active:
+                    metrics_row.update({
+                        "train_v_mse": epoch_v_mse,
+                        "train_lat_dz": epoch_lat_dz,
+                        "train_lat_ddz": epoch_lat_ddz,
+                        "train_total": epoch_loss,
+                        "val_lat_dz": val_lat_dz,
+                        "val_lat_ddz": val_lat_ddz,
+                    })
+                    if args.w_lat_x0 > 0:
+                        metrics_row["train_lat_x0"] = epoch_lat_x0
+                        metrics_row["val_lat_x0"] = val_lat_x0
+                metrics_fp.write(json.dumps(metrics_row) + "\n"); metrics_fp.flush()
 
                 # Best ckpt — rank 0 only; unwrap DDP for clean state_dict
                 if val_loss < best_val:
diff --git a/src/models/graph_salad/attention.py b/src/models/graph_salad/attention.py
index 2c978c1..038864a 100644
--- a/src/models/graph_salad/attention.py
+++ b/src/models/graph_salad/attention.py
@@ -78,6 +78,7 @@ class GraphAttentionBlock(nn.Module):
         n_heads: int,
         d_ff: int,
         dropout: float = 0.1,
+        use_graph_bias: bool = True,
     ) -> None:
         super().__init__()
         if d_model <= 0 or n_heads <= 0:
@@ -97,6 +98,7 @@ class GraphAttentionBlock(nn.Module):
         self.d_model = d_model
         self.n_heads = n_heads
         self.d_head = d_model // n_heads
+        self.use_graph_bias = use_graph_bias
 
         # Q/K/V/O projections
         self.q_proj = nn.Linear(d_model, d_model)
@@ -104,10 +106,15 @@ class GraphAttentionBlock(nn.Module):
         self.v_proj = nn.Linear(d_model, d_model)
         self.o_proj = nn.Linear(d_model, d_model)
 
-        # Edge bias projections (scalar → per-head)
-        # Matches encoder.py:41-42 formulation.
-        self.geodesic_bias = nn.Linear(1, n_heads, bias=False)
-        self.adjacency_bias = nn.Linear(1, n_heads, bias=False)
+        # Edge bias projections (scalar → per-head). Matches encoder.py:41-42.
+        # Only the graph-aware variant adds adjacency/geodesic bias to the scores.
+        # The no_graph_spatial ablation (use_graph_bias=False) is a plain slot
+        # self-attention: it drops these two tiny projections (~2*n_heads params/
+        # block, negligible vs d_model² Q/K/V/O+FFN) and skips the bias in _compute,
+        # but keeps node_mask + the rest of the block byte-identical (param-aligned).
+        if use_graph_bias:
+            self.geodesic_bias = nn.Linear(1, n_heads, bias=False)
+            self.adjacency_bias = nn.Linear(1, n_heads, bias=False)
 
         # Norms (pre-norm)
         self.norm1 = nn.LayerNorm(d_model)
@@ -350,12 +357,15 @@ class GraphAttentionBlock(nn.Module):
         # affects unmasked-but-disconnected pairs (rare; deferred to a later
         # learnable "unreachable" bucket per lit survey if it shows up in
         # generation eval). NaN/-Inf were rejected above.
-        geo = geodesic_dist.clone()
-        geo[torch.isinf(geo)] = 0.0
-        geo_bias = self.geodesic_bias(geo.unsqueeze(-1))         # [B, N, N, H]
-        adj_bias = self.adjacency_bias(adjacency.unsqueeze(-1))  # [B, N, N, H]
-        topo_bias = (geo_bias + adj_bias).permute(0, 3, 1, 2)    # [B, H, N, N]
-        scores = scores + topo_bias
+        # Graph-aware variant only; the no_graph_spatial ablation skips the topo
+        # bias entirely → plain slot self-attention (still node-masked below).
+        if self.use_graph_bias:
+            geo = geodesic_dist.clone()
+            geo[torch.isinf(geo)] = 0.0
+            geo_bias = self.geodesic_bias(geo.unsqueeze(-1))         # [B, N, N, H]
+            adj_bias = self.adjacency_bias(adjacency.unsqueeze(-1))  # [B, N, N, H]
+            topo_bias = (geo_bias + adj_bias).permute(0, 3, 1, 2)    # [B, H, N, N]
+            scores = scores + topo_bias
 
         # Mask invalid nodes (key side). Use large finite negative for softmax
         # numerical safety; matches encoder.py:84-85.
diff --git a/src/models/graph_salad/denoiser.py b/src/models/graph_salad/denoiser.py
index edf92ec..324e376 100644
--- a/src/models/graph_salad/denoiser.py
+++ b/src/models/graph_salad/denoiser.py
@@ -215,17 +215,27 @@ class GraphSaladDenoiserLayer(nn.Module):
         d_t: int,
         dropout: float = 0.1,
         text_mode: str = "mean_additive",
+        spatial_mode: str = "graph",
     ) -> None:
         super().__init__()
         self.text_mode = text_mode
-        self.spatial = GraphAttentionBlock(d_model, n_heads, d_ff, dropout=dropout)
+        self.spatial_mode = spatial_mode
+        # spatial_mode="graph" (default): graph-aware spatial attn (adjacency+geodesic
+        # bias). "plain": no_graph_spatial ablation — plain slot self-attn (no topo
+        # bias), still node-masked; pooled_skeleton_embeddings additive (top-level
+        # input_proj) is unchanged, so the model still knows "what segment" each slot is.
+        self.spatial = GraphAttentionBlock(
+            d_model, n_heads, d_ff, dropout=dropout,
+            use_graph_bias=(spatial_mode == "graph"))
         self.temporal = TemporalSelfAttention(d_model, n_heads, dropout=dropout)
         self.film_after_spatial = DenseFiLM(d_t, d_model)
         self.film_after_temporal = DenseFiLM(d_t, d_model)
         self.film_after_text = DenseFiLM(d_t, d_model)
-        # Token cross-attn sub-block exists only in token mode (so mean-mode
-        # state_dict is byte-identical to old ckpts — strict-load preserved).
-        if text_mode == "token_cross_attn":
+        # Token cross-attn sub-block exists in token_cross_attn AND dual_text (so
+        # mean-mode state_dict is byte-identical to old ckpts — strict-load
+        # preserved). dual_text additionally uses the (always-present) global
+        # text_proj path, so its per-layer params == token mode's.
+        if text_mode in ("token_cross_attn", "dual_text"):
             self.text_cross_attn = TextCrossAttention(d_model, n_heads, dropout=dropout)
 
     def forward(
@@ -277,14 +287,17 @@ class GraphSaladDenoiserLayer(nn.Module):
         x = self.film_after_temporal(x, t_emb)
 
         # --- 5. Text conditioning (mode-dependent) ---
-        if self.text_mode == "token_cross_attn":
+        # dual_text runs BOTH sub-blocks (token cross-attn THEN global add) in the
+        # SAME per-layer slot — the spatial/temporal FiLM ordering is unchanged, so
+        # mean_additive / token_cross_attn behave byte-identically.
+        if self.text_mode in ("token_cross_attn", "dual_text"):
             # Motion tokens [B,T*C,D] cross-attend text tokens [B,L,D]. CFG-uncond
             # rows (all text keys masked) get zero output (TextCrossAttention).
             q = x.reshape(B, T_lat * C, D)
             ca = self.text_cross_attn(q, text_tokens, text_key_padding_mask)
             x = x + ca.reshape(B, T_lat, C, D)
-        else:
-            # mean_additive (default): broadcast-add projected mean text (gated).
+        if self.text_mode in ("mean_additive", "dual_text"):
+            # broadcast-add projected mean/global text (gated by has_text → CFG).
             text_gated = text_cond * has_text[:, None].to(text_cond.dtype)  # [B, D]
             x = x + text_gated[:, None, None, :]
 
@@ -321,12 +334,18 @@ class GraphSaladDenoiser(nn.Module):
         dropout: float = 0.1,
         text_mode: str = "mean_additive",
         text_token_dim: int = 768,
+        spatial_mode: str = "graph",
     ) -> None:
         super().__init__()
-        if text_mode not in ("mean_additive", "token_cross_attn"):
+        if text_mode not in ("mean_additive", "token_cross_attn", "dual_text"):
             raise ValueError(
-                f"text_mode must be 'mean_additive' or 'token_cross_attn', "
-                f"got {text_mode!r}"
+                f"text_mode must be 'mean_additive', 'token_cross_attn' or "
+                f"'dual_text', got {text_mode!r}"
+            )
+        if spatial_mode not in ("graph", "plain"):
+            raise ValueError(
+                f"spatial_mode must be 'graph' (graph-aware spatial attn) or "
+                f"'plain' (no_graph_spatial ablation), got {spatial_mode!r}"
             )
         if n_layers % 2 == 0:
             raise ValueError(
@@ -348,6 +367,7 @@ class GraphSaladDenoiser(nn.Module):
         self.d_t = d_t
         self.text_mode = text_mode
         self.text_token_dim = text_token_dim
+        self.spatial_mode = spatial_mode
 
         # --- Timestep embedding (shared across all layers' FiLMs) ---
         self.t_sin = SinusoidalTimestepEmbedding(d_t)
@@ -361,9 +381,10 @@ class GraphSaladDenoiser(nn.Module):
         # Per design §2.3: denoiser owns its own text_proj (NOT reusing VAE's).
         # mean_additive: projects the [B,768] mean-pooled caption.
         self.text_proj = nn.Linear(d_text, d_model)
-        # token_cross_attn: separate projection for token-level T5 [B,L,768]→[B,L,D]
-        # (exists only in token mode → mean-mode ckpts stay byte-identical).
-        if text_mode == "token_cross_attn":
+        # token_cross_attn / dual_text: separate projection for token-level T5
+        # [B,L,768]→[B,L,D] (exists only in these modes → mean-mode ckpts stay
+        # byte-identical). dual_text uses BOTH text_proj (global) and text_token_proj.
+        if text_mode in ("token_cross_attn", "dual_text"):
             self.text_token_proj = nn.Linear(text_token_dim, d_model)
 
         # --- Input projection: latent z + slot conditioning ---
@@ -376,7 +397,7 @@ class GraphSaladDenoiser(nn.Module):
         self.layers = nn.ModuleList(
             [
                 GraphSaladDenoiserLayer(d_model, n_heads, d_ff, d_t, dropout=dropout,
-                                        text_mode=text_mode)
+                                        text_mode=text_mode, spatial_mode=spatial_mode)
                 for _ in range(n_layers)
             ]
         )
@@ -409,9 +430,16 @@ class GraphSaladDenoiser(nn.Module):
         has_text: torch.Tensor | None = None,
         validate_inputs: bool = False,
         text_token_mask: torch.Tensor | None = None,
+        text_tokens: torch.Tensor | None = None,
     ) -> torch.Tensor:
         """Returns v_pred [B, T_lat, C, D].
 
+        text/text_tokens contract by text_mode:
+          - mean_additive:    text=[B,768] global; text_tokens=None.
+          - token_cross_attn: text=[B,L,768] tokens; text_tokens=None.
+          - dual_text:        text=[B,768] global AND text_tokens=[B,L,768] tokens
+                              (+ text_token_mask). Both streams gated by has_text.
+
         See module docstring for input/output contracts.
 
         Args:
@@ -448,36 +476,59 @@ class GraphSaladDenoiser(nn.Module):
                 f"has_text must be [B={B}] bool, got {tuple(has_text.shape)} "
                 f"dtype {has_text.dtype}"
             )
+        def _check_token_mask(L: int) -> None:
+            if (text_token_mask is None or text_token_mask.shape != (B, L)
+                    or text_token_mask.dtype != torch.bool):
+                raise ValueError(
+                    f"{self.text_mode} requires text_token_mask [B={B}, L={L}] bool, "
+                    f"got {None if text_token_mask is None else tuple(text_token_mask.shape)}"
+                    f"{'' if text_token_mask is None else ' ' + str(text_token_mask.dtype)}"
+                )
+            if text_token_mask.device != z_t.device:
+                raise ValueError(
+                    f"text_token_mask.device {text_token_mask.device} != "
+                    f"z_t.device {z_t.device}"
+                )
         if self.text_mode == "mean_additive":
             if text.dim() != 2 or text.shape[0] != B or text.shape[1] != self.d_text:
                 raise ValueError(
                     f"text must be [B={B}, d_text={self.d_text}] for mean_additive "
                     f"(mean-pooled cache); got {tuple(text.shape)}."
                 )
-            if text_token_mask is not None:
+            if text_token_mask is not None or text_tokens is not None:
+                raise ValueError(
+                    "mean_additive: text_token_mask AND text_tokens must both be None."
+                )
+        elif self.text_mode == "dual_text":
+            # global stream: text = mean-pooled [B,768] (like mean_additive)
+            if text.dim() != 2 or text.shape[0] != B or text.shape[1] != self.d_text:
+                raise ValueError(
+                    f"dual_text: text (global mean) must be [B={B}, d_text={self.d_text}]; "
+                    f"got {tuple(text.shape)}."
+                )
+            # token stream: text_tokens = token-level [B,L,768] (separate keyword arg)
+            if (text_tokens is None or text_tokens.dim() != 3
+                    or text_tokens.shape[0] != B
+                    or text_tokens.shape[2] != self.text_token_dim):
                 raise ValueError(
-                    "text_token_mask must be None in mean_additive mode."
+                    f"dual_text: text_tokens must be [B={B}, L, "
+                    f"text_token_dim={self.text_token_dim}]; got "
+                    f"{None if text_tokens is None else tuple(text_tokens.shape)}."
                 )
-        else:  # token_cross_attn
+            _check_token_mask(text_tokens.shape[1])
+        else:  # token_cross_attn (single token stream via `text`)
             if (text.dim() != 3 or text.shape[0] != B
                     or text.shape[2] != self.text_token_dim):
                 raise ValueError(
                     f"text must be [B={B}, L, text_token_dim={self.text_token_dim}] "
                     f"for token_cross_attn; got {tuple(text.shape)}."
                 )
-            L = text.shape[1]
-            if (text_token_mask is None or text_token_mask.shape != (B, L)
-                    or text_token_mask.dtype != torch.bool):
-                raise ValueError(
-                    f"token_cross_attn requires text_token_mask [B={B}, L={L}] bool, "
-                    f"got {None if text_token_mask is None else tuple(text_token_mask.shape)}"
-                    f"{'' if text_token_mask is None else ' ' + str(text_token_mask.dtype)}"
-                )
-            if text_token_mask.device != z_t.device:
+            if text_tokens is not None:
                 raise ValueError(
-                    f"text_token_mask.device {text_token_mask.device} != "
-                    f"z_t.device {z_t.device}"
+                    "token_cross_attn: tokens go via `text`; text_tokens must be None "
+                    "(text_tokens is the dual_text global+token split)."
                 )
+            _check_token_mask(text.shape[1])
         if timesteps.shape != (B,):
             raise ValueError(f"timesteps must be [B={B}], got {tuple(timesteps.shape)}")
         if coarse_mask.shape != (B, C) or coarse_mask.dtype != torch.bool:
@@ -529,6 +580,13 @@ class GraphSaladDenoiser(nn.Module):
                 raise ValueError(
                     f"GraphSaladDenoiser: {name}.dtype {t.dtype} != z_t.dtype {z_t.dtype}"
                 )
+        if text_tokens is not None and (text_tokens.device != ref_device
+                                        or text_tokens.dtype != z_t.dtype):
+            raise ValueError(
+                f"GraphSaladDenoiser: text_tokens device/dtype "
+                f"({text_tokens.device}/{text_tokens.dtype}) must match z_t "
+                f"({ref_device}/{z_t.dtype})."
+            )
 
         # --- Timestep embedding (shared by all FiLMs) ---
         t_emb = self.t_mlp(self.t_sin(timesteps))         # [B, D_t]
@@ -538,13 +596,23 @@ class GraphSaladDenoiser(nn.Module):
         # token_cross_attn: project tokens [B,L,768]→[B,L,D] + build the per-layer
         #   key_padding_mask (True = ignore): pad-token OR has_text=False. An
         #   all-masked (uncond) row is handled by TextCrossAttention (output→0).
+        # dual_text: BOTH — global text_cond (from `text`) + token tok_emb (from the
+        #   separate `text_tokens` arg). One shared key_padding_mask gates the token
+        #   stream; has_text gates the global stream → both CFG-drop together.
+        # NOTE: local projected tokens are `tok_emb` (NOT `text_tokens`) so the
+        # dual_text `text_tokens` forward ARG is not clobbered.
         text_cond = None
-        text_tokens = None
+        tok_emb = None
         text_key_padding_mask = None
         if self.text_mode == "mean_additive":
             text_cond = self.text_proj(text)              # [B, D]
-        else:
-            text_tokens = self.text_token_proj(text)      # [B, L, D]
+        elif self.text_mode == "dual_text":
+            text_cond = self.text_proj(text)              # global [B, D]
+            tok_emb = self.text_token_proj(text_tokens)   # tokens [B, L, D]
+            valid = text_token_mask & has_text[:, None]   # [B, L] bool
+            text_key_padding_mask = ~valid                # [B, L] True=mask
+        else:  # token_cross_attn (tokens via `text`)
+            tok_emb = self.text_token_proj(text)          # [B, L, D]
             # valid key = token present AND has_text=True. key_padding_mask is
             # the inverse (True ⇒ mask). has_text=False ⇒ whole row masked.
             valid = text_token_mask & has_text[:, None]   # [B, L] bool
@@ -570,7 +638,7 @@ class GraphSaladDenoiser(nn.Module):
                 x, t_emb, text_cond, has_text,
                 adjacency, geodesic_dist, coarse_mask, frame_mask,
                 validate_inputs=validate_inputs,
-                text_tokens=text_tokens,
+                text_tokens=tok_emb,
                 text_key_padding_mask=text_key_padding_mask,
             )
             enc_outputs.append(x)
@@ -580,7 +648,7 @@ class GraphSaladDenoiser(nn.Module):
             x, t_emb, text_cond, has_text,
             adjacency, geodesic_dist, coarse_mask, frame_mask,
             validate_inputs=validate_inputs,
-            text_tokens=text_tokens,
+            text_tokens=tok_emb,
             text_key_padding_mask=text_key_padding_mask,
         )
 
@@ -594,7 +662,7 @@ class GraphSaladDenoiser(nn.Module):
                 x, t_emb, text_cond, has_text,
                 adjacency, geodesic_dist, coarse_mask, frame_mask,
                 validate_inputs=validate_inputs,
-                text_tokens=text_tokens,
+                text_tokens=tok_emb,
                 text_key_padding_mask=text_key_padding_mask,
             )
 

codex
The first diff includes unrelated latent-dynamics and dual-text edits, so I’m narrowing to the symbol paths and line-numbered code that affect graph/plain behavior and launcher propagation.
exec
/usr/bin/bash -lc "nl -ba src/models/graph_salad/attention.py | sed -n '70,145p;300,405p;405,455p'" in /iridisfs/scratch/ts1v23/workspace/noKslot_clean
 succeeded in 0ms:
    70	
    71	    Forward returns:
    72	        [B, N, d_model]
    73	    """
    74	
    75	    def __init__(
    76	        self,
    77	        d_model: int,
    78	        n_heads: int,
    79	        d_ff: int,
    80	        dropout: float = 0.1,
    81	        use_graph_bias: bool = True,
    82	    ) -> None:
    83	        super().__init__()
    84	        if d_model <= 0 or n_heads <= 0:
    85	            raise ValueError(f"d_model and n_heads must be > 0, got {d_model}, {n_heads}")
    86	        if d_model % n_heads != 0:
    87	            raise ValueError(
    88	                f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
    89	            )
    90	        if d_ff <= 0:
    91	            raise ValueError(f"d_ff must be > 0, got {d_ff}")
    92	        # Dropout contract is [0, 1) per nn.Dropout docs; p=1 zeros all outputs
    93	        # (everything dropped) which corrupts gradient flow silently. Codex M1.2
    94	        # round 1 R12 fix.
    95	        if not (0.0 <= dropout < 1.0):
    96	            raise ValueError(f"dropout must be in [0, 1), got {dropout}")
    97	
    98	        self.d_model = d_model
    99	        self.n_heads = n_heads
   100	        self.d_head = d_model // n_heads
   101	        self.use_graph_bias = use_graph_bias
   102	
   103	        # Q/K/V/O projections
   104	        self.q_proj = nn.Linear(d_model, d_model)
   105	        self.k_proj = nn.Linear(d_model, d_model)
   106	        self.v_proj = nn.Linear(d_model, d_model)
   107	        self.o_proj = nn.Linear(d_model, d_model)
   108	
   109	        # Edge bias projections (scalar → per-head). Matches encoder.py:41-42.
   110	        # Only the graph-aware variant adds adjacency/geodesic bias to the scores.
   111	        # The no_graph_spatial ablation (use_graph_bias=False) is a plain slot
   112	        # self-attention: it drops these two tiny projections (~2*n_heads params/
   113	        # block, negligible vs d_model² Q/K/V/O+FFN) and skips the bias in _compute,
   114	        # but keeps node_mask + the rest of the block byte-identical (param-aligned).
   115	        if use_graph_bias:
   116	            self.geodesic_bias = nn.Linear(1, n_heads, bias=False)
   117	            self.adjacency_bias = nn.Linear(1, n_heads, bias=False)
   118	
   119	        # Norms (pre-norm)
   120	        self.norm1 = nn.LayerNorm(d_model)
   121	        self.norm2 = nn.LayerNorm(d_model)
   122	
   123	        # Feedforward block
   124	        self.ff = nn.Sequential(
   125	            nn.Linear(d_model, d_ff),
   126	            nn.GELU(),
   127	            nn.Dropout(dropout),
   128	            nn.Linear(d_ff, d_model),
   129	            nn.Dropout(dropout),
   130	        )
   131	        self.dropout = nn.Dropout(dropout)
   132	
   133	    def forward(
   134	        self,
   135	        x: torch.Tensor,
   136	        adjacency: torch.Tensor,
   137	        geodesic_dist: torch.Tensor,
   138	        node_mask: torch.Tensor,
   139	        validate_inputs: bool = True,
   140	    ) -> torch.Tensor:
   141	        if not validate_inputs:
   142	            # Caller asserts inputs already validated (hot-path, e.g. denoiser
   143	            # timestep loop where adjacency / geodesic / mask are static across
   144	            # the loop). Forward straight to compute.
   145	            return self._compute(x, adjacency, geodesic_dist, node_mask)
   300	                f"batch element(s) {bad} (no valid nodes; attention undefined)"
   301	            )
   302	
   303	        # --- R12 fail-loud: adj/geo cross-consistency (codex M1.2 round 6) ---
   304	        # geodesic_dist must equal floyd_shortest_path(adjacency, node_mask).
   305	        # Without this, a bounded-but-wrong geo (e.g. correct adj, geo[0,3]=2
   306	        # instead of 3 on a 4-node line) silently corrupts the additive topology
   307	        # bias and skews attention. Costs O(B·N^3) per forward — acceptable at
   308	        # N≤160 (~4M ops, <1ms on GPU).
   309	        expected_geo = floyd_shortest_path(adjacency, node_mask)
   310	        both_valid = node_mask[:, :, None] & node_mask[:, None, :]
   311	        # Pattern check: reachability (finite/+Inf) must match on valid pairs.
   312	        finite_actual = torch.isfinite(geodesic_dist) & both_valid
   313	        finite_expected = torch.isfinite(expected_geo) & both_valid
   314	        if not torch.equal(finite_actual, finite_expected):
   315	            raise ValueError(
   316	                "GraphAttentionBlock: geodesic_dist reachability pattern "
   317	                "inconsistent with adjacency (Floyd-recomputed)"
   318	            )
   319	        # Value check on entries that are finite in BOTH:
   320	        compare_mask = finite_actual & finite_expected
   321	        if not torch.allclose(
   322	            geodesic_dist[compare_mask], expected_geo[compare_mask],
   323	            atol=1e-6, rtol=0.0,
   324	        ):
   325	            raise ValueError(
   326	                "GraphAttentionBlock: geodesic_dist values inconsistent with "
   327	                "shortest-path over adjacency (Floyd-recomputed)"
   328	            )
   329	
   330	        return self._compute(x, adjacency, geodesic_dist, node_mask)
   331	
   332	    def _compute(
   333	        self,
   334	        x: torch.Tensor,
   335	        adjacency: torch.Tensor,
   336	        geodesic_dist: torch.Tensor,
   337	        node_mask: torch.Tensor,
   338	    ) -> torch.Tensor:
   339	        """Pure compute path; assumes inputs already validated."""
   340	        B, N, _ = x.shape
   341	        # --- Pre-norm + self-attn ---
   342	        residual = x
   343	        x_norm = self.norm1(x)
   344	
   345	        q = self.q_proj(x_norm).view(B, N, self.n_heads, self.d_head).permute(0, 2, 1, 3)
   346	        k = self.k_proj(x_norm).view(B, N, self.n_heads, self.d_head).permute(0, 2, 1, 3)
   347	        v = self.v_proj(x_norm).view(B, N, self.n_heads, self.d_head).permute(0, 2, 1, 3)
   348	        # q/k/v: [B, H, N, d_head]
   349	
   350	        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_head)
   351	        # [B, H, N, N]
   352	
   353	        # Topology biases. geodesic_dist may contain +inf for legitimate
   354	        # unreachable pairs (from floyd_shortest_path). Substitute +inf with
   355	        # 0.0 BEFORE projecting — this gives a neutral additive bias on those
   356	        # pairs. The key-mask masks out padded keys, so the neutral bias only
   357	        # affects unmasked-but-disconnected pairs (rare; deferred to a later
   358	        # learnable "unreachable" bucket per lit survey if it shows up in
   359	        # generation eval). NaN/-Inf were rejected above.
   360	        # Graph-aware variant only; the no_graph_spatial ablation skips the topo
   361	        # bias entirely → plain slot self-attention (still node-masked below).
   362	        if self.use_graph_bias:
   363	            geo = geodesic_dist.clone()
   364	            geo[torch.isinf(geo)] = 0.0
   365	            geo_bias = self.geodesic_bias(geo.unsqueeze(-1))         # [B, N, N, H]
   366	            adj_bias = self.adjacency_bias(adjacency.unsqueeze(-1))  # [B, N, N, H]
   367	            topo_bias = (geo_bias + adj_bias).permute(0, 3, 1, 2)    # [B, H, N, N]
   368	            scores = scores + topo_bias
   369	
   370	        # Mask invalid nodes (key side). Use large finite negative for softmax
   371	        # numerical safety; matches encoder.py:84-85.
   372	        mask = node_mask.unsqueeze(1).unsqueeze(2)  # [B, 1, 1, N]
   373	        scores = scores.masked_fill(~mask, -1e9)
   374	
   375	        # Softmax. The earlier `all-False node_mask` per-sample guard ensures
   376	        # at least one valid key per batch element, so no row of `scores` is
   377	        # entirely -1e9 → no NaN in softmax output → no nan_to_num needed.
   378	        # Padded-query rows still compute attention (over valid keys); their
   379	        # output is zeroed downstream by the caller's joint_mask multiplication.
   380	        # softmax in fp32 for bf16-safety (sentinel + reduction precision). On the
   381	        # fp32 path scores.float() is a no-op and .to(scores.dtype) returns fp32, so
   382	        # behavior is byte-for-byte unchanged; on the bf16 path softmax runs in fp32
   383	        # then casts the probabilities back to bf16 for the attn@v matmul.
   384	        attn = F.softmax(scores.float(), dim=-1).to(scores.dtype)
   385	        attn = self.dropout(attn)
   386	
   387	        out = torch.matmul(attn, v)  # [B, H, N, d_head]
   388	        out = out.permute(0, 2, 1, 3).contiguous().view(B, N, self.d_model)
   389	        out = self.o_proj(out)
   390	        x = residual + self.dropout(out)
   391	
   392	        # --- Pre-norm + FFN ---
   393	        x = x + self.ff(self.norm2(x))
   394	
   395	        return x

exec
/usr/bin/bash -lc 'rg -n "use_graph_bias|geodesic_bias|adjacency_bias|spatial_mode|validate_inputs|adjacency|geodesic_dist" src/models/graph_salad/attention.py src/models/graph_salad/denoiser.py scripts/train_denoiser.py scripts/animate_denoiser.py scripts/_launch_diffusion_t2m.sh scripts/_launch_diffusion_t2m_4card.sh scripts/_launch_token_diffusion_8card_a100.sh' in /iridisfs/scratch/ts1v23/workspace/noKslot_clean
 succeeded in 0ms:
scripts/animate_denoiser.py:69:    spatial_mode = da.get("spatial_mode", "graph")  # old ckpts (no key) → graph
scripts/animate_denoiser.py:75:        spatial_mode=spatial_mode,
scripts/animate_denoiser.py:102:    B = skel["pooled_adjacency"].shape[0]
scripts/animate_denoiser.py:103:    C = skel["pooled_adjacency"].shape[1]
scripts/animate_denoiser.py:114:    adj2 = skel["pooled_adjacency"].repeat(2, 1, 1)
scripts/animate_denoiser.py:145:            adjacency=adj2, geodesic_dist=geo2,
scripts/animate_denoiser.py:151:            validate_inputs=first,  # cold-start validate on first iter
scripts/animate_denoiser.py:174:        "pooled_adjacency": skel["pooled_adjacency"],
scripts/_launch_diffusion_t2m.sh:116:echo "[t2m] latdyn: w_lat_dz=$W_LAT_DZ w_lat_ddz=$W_LAT_DDZ w_lat_x0=$W_LAT_X0 target=$LATENT_DYN_TARGET | spatial_mode=$SPATIAL_MODE"
scripts/_launch_diffusion_t2m.sh:128:  --spatial_mode "$SPATIAL_MODE" \
scripts/train_denoiser.py:329:    ap.add_argument("--spatial_mode", choices=["graph", "plain"], default="graph",
scripts/train_denoiser.py:331:                         "(adjacency+geodesic bias); old ckpts strict-load. 'plain': "
scripts/train_denoiser.py:616:        spatial_mode=args.spatial_mode,
scripts/train_denoiser.py:620:        f"text_mode={args.text_mode} spatial_mode={args.spatial_mode} params={n_params:,}")
scripts/train_denoiser.py:643:        ck_spatial_mode = resume_ck.get("args", {}).get("spatial_mode", "graph")
scripts/train_denoiser.py:644:        if ck_spatial_mode != args.spatial_mode:
scripts/train_denoiser.py:646:                f"[RESUME FAIL] ckpt spatial_mode={ck_spatial_mode!r} != CLI "
scripts/train_denoiser.py:647:                f"--spatial_mode {args.spatial_mode!r}. graph/plain arch differ "
scripts/train_denoiser.py:648:                f"(plain drops adjacency/geodesic bias params)."
scripts/train_denoiser.py:677:        ck_spatial_mode = ck.get("args", {}).get("spatial_mode", "graph")
scripts/train_denoiser.py:678:        if ck_spatial_mode != args.spatial_mode:
scripts/train_denoiser.py:680:                f"[INIT_CKPT FAIL] ckpt spatial_mode={ck_spatial_mode!r} != CLI "
scripts/train_denoiser.py:681:                f"--spatial_mode {args.spatial_mode!r}. graph/plain arch differ."
scripts/train_denoiser.py:805:            pooled_adj = enc["pooled_adjacency"].float()
scripts/train_denoiser.py:857:                    adjacency=pooled_adj, geodesic_dist=pooled_geo,
scripts/train_denoiser.py:864:                    validate_inputs=(global_it == 0),
scripts/train_denoiser.py:962:                        pooled_adj = enc["pooled_adjacency"].float()
scripts/train_denoiser.py:996:                                adjacency=pooled_adj, geodesic_dist=pooled_geo,
scripts/train_denoiser.py:999:                                has_text=has_text, validate_inputs=False,
src/models/graph_salad/attention.py:6:  adjacency / geodesic bias
src/models/graph_salad/attention.py:10:adjacency + geodesic. Matches encoder.py::GraphAttentionBlock formulation
src/models/graph_salad/attention.py:42:    """Graph-aware multi-head self-attention with adjacency + geodesic bias.
src/models/graph_salad/attention.py:52:        adjacency:     [B, N, N]        — binary-or-soft in [0, 1], symmetric,
src/models/graph_salad/attention.py:56:                                           skeleton-or-pooled adjacency; values
src/models/graph_salad/attention.py:59:        geodesic_dist: [B, N, N]        — non-negative finite hop-count distances
src/models/graph_salad/attention.py:65:        validate_inputs: bool           — when True (default), runs the full
src/models/graph_salad/attention.py:81:        use_graph_bias: bool = True,
src/models/graph_salad/attention.py:101:        self.use_graph_bias = use_graph_bias
src/models/graph_salad/attention.py:110:        # Only the graph-aware variant adds adjacency/geodesic bias to the scores.
src/models/graph_salad/attention.py:111:        # The no_graph_spatial ablation (use_graph_bias=False) is a plain slot
src/models/graph_salad/attention.py:115:        if use_graph_bias:
src/models/graph_salad/attention.py:116:            self.geodesic_bias = nn.Linear(1, n_heads, bias=False)
src/models/graph_salad/attention.py:117:            self.adjacency_bias = nn.Linear(1, n_heads, bias=False)
src/models/graph_salad/attention.py:136:        adjacency: torch.Tensor,
src/models/graph_salad/attention.py:137:        geodesic_dist: torch.Tensor,
src/models/graph_salad/attention.py:139:        validate_inputs: bool = True,
src/models/graph_salad/attention.py:141:        if not validate_inputs:
src/models/graph_salad/attention.py:143:            # timestep loop where adjacency / geodesic / mask are static across
src/models/graph_salad/attention.py:145:            return self._compute(x, adjacency, geodesic_dist, node_mask)
src/models/graph_salad/attention.py:159:        if adjacency.shape != (B, N, N) or geodesic_dist.shape != (B, N, N):
src/models/graph_salad/attention.py:161:                f"adjacency/geodesic_dist must be [B={B}, N={N}, N={N}], "
src/models/graph_salad/attention.py:162:                f"got {tuple(adjacency.shape)} and {tuple(geodesic_dist.shape)}"
src/models/graph_salad/attention.py:183:        for name, t in (("x", x), ("adjacency", adjacency), ("geodesic_dist", geodesic_dist)):
src/models/graph_salad/attention.py:203:        # adjacency contract: finite, non-negative, symmetric, zero diagonal.
src/models/graph_salad/attention.py:205:        # variants that may emit soft-weighted pooled adjacency, but the
src/models/graph_salad/attention.py:207:        if not torch.isfinite(adjacency).all():
src/models/graph_salad/attention.py:209:                "GraphAttentionBlock: adjacency contains NaN or Inf"
src/models/graph_salad/attention.py:211:        if (adjacency < 0).any():
src/models/graph_salad/attention.py:213:                "GraphAttentionBlock: adjacency contains negative values "
src/models/graph_salad/attention.py:216:        if (adjacency > 1.0).any():
src/models/graph_salad/attention.py:218:                "GraphAttentionBlock: adjacency contains values > 1.0; "
src/models/graph_salad/attention.py:226:            adjacency, adjacency.transpose(-2, -1), atol=1e-6, rtol=0.0
src/models/graph_salad/attention.py:229:                "GraphAttentionBlock: adjacency is not symmetric "
src/models/graph_salad/attention.py:232:        if (adjacency.diagonal(dim1=-2, dim2=-1) != 0).any():
src/models/graph_salad/attention.py:234:                "GraphAttentionBlock: adjacency has non-zero diagonal "
src/models/graph_salad/attention.py:237:        # geodesic_dist contract: no NaN, no -Inf (+Inf is legitimate per Floyd
src/models/graph_salad/attention.py:240:        if torch.isnan(geodesic_dist).any():
src/models/graph_salad/attention.py:242:                "GraphAttentionBlock: geodesic_dist contains NaN"
src/models/graph_salad/attention.py:244:        if (geodesic_dist == float("-inf")).any():
src/models/graph_salad/attention.py:246:                "GraphAttentionBlock: geodesic_dist contains -Inf "
src/models/graph_salad/attention.py:250:        finite_geo = geodesic_dist[torch.isfinite(geodesic_dist)]
src/models/graph_salad/attention.py:253:                "GraphAttentionBlock: geodesic_dist has negative finite entries "
src/models/graph_salad/attention.py:261:                f"GraphAttentionBlock: geodesic_dist has finite entries > {N - 1} "
src/models/graph_salad/attention.py:269:        gt = geodesic_dist.transpose(-2, -1)
src/models/graph_salad/attention.py:270:        finite_g = torch.isfinite(geodesic_dist)
src/models/graph_salad/attention.py:274:                "GraphAttentionBlock: geodesic_dist finite/+Inf pattern is not "
src/models/graph_salad/attention.py:279:            geodesic_dist[both_finite], gt[both_finite], atol=1e-6, rtol=0.0
src/models/graph_salad/attention.py:282:                "GraphAttentionBlock: geodesic_dist is not symmetric on finite entries"
src/models/graph_salad/attention.py:285:        diag = geodesic_dist.diagonal(dim1=-2, dim2=-1)  # [B, N]
src/models/graph_salad/attention.py:288:                "GraphAttentionBlock: geodesic_dist has non-zero diagonal "
src/models/graph_salad/attention.py:304:        # geodesic_dist must equal floyd_shortest_path(adjacency, node_mask).
src/models/graph_salad/attention.py:309:        expected_geo = floyd_shortest_path(adjacency, node_mask)
src/models/graph_salad/attention.py:312:        finite_actual = torch.isfinite(geodesic_dist) & both_valid
src/models/graph_salad/attention.py:316:                "GraphAttentionBlock: geodesic_dist reachability pattern "
src/models/graph_salad/attention.py:317:                "inconsistent with adjacency (Floyd-recomputed)"
src/models/graph_salad/attention.py:322:            geodesic_dist[compare_mask], expected_geo[compare_mask],
src/models/graph_salad/attention.py:326:                "GraphAttentionBlock: geodesic_dist values inconsistent with "
src/models/graph_salad/attention.py:327:                "shortest-path over adjacency (Floyd-recomputed)"
src/models/graph_salad/attention.py:330:        return self._compute(x, adjacency, geodesic_dist, node_mask)
src/models/graph_salad/attention.py:335:        adjacency: torch.Tensor,
src/models/graph_salad/attention.py:336:        geodesic_dist: torch.Tensor,
src/models/graph_salad/attention.py:353:        # Topology biases. geodesic_dist may contain +inf for legitimate
src/models/graph_salad/attention.py:362:        if self.use_graph_bias:
src/models/graph_salad/attention.py:363:            geo = geodesic_dist.clone()
src/models/graph_salad/attention.py:365:            geo_bias = self.geodesic_bias(geo.unsqueeze(-1))         # [B, N, N, H]
src/models/graph_salad/attention.py:366:            adj_bias = self.adjacency_bias(adjacency.unsqueeze(-1))  # [B, N, N, H]
src/models/graph_salad/denoiser.py:16:    for Phase-2 coarse-node self-attn over `pooled_adjacency` / `pooled_geodesic`
src/models/graph_salad/denoiser.py:33:  adjacency     [B, C, C] — pooled_adjacency
src/models/graph_salad/denoiser.py:34:  geodesic_dist [B, C, C] — pooled_geodesic
src/models/graph_salad/denoiser.py:44:Hot-path note: `validate_inputs=False` is passed to GraphAttentionBlock on every
src/models/graph_salad/denoiser.py:46:`validate_inputs=True` separately to catch graph-contract violations.
src/models/graph_salad/denoiser.py:218:        spatial_mode: str = "graph",
src/models/graph_salad/denoiser.py:222:        self.spatial_mode = spatial_mode
src/models/graph_salad/denoiser.py:223:        # spatial_mode="graph" (default): graph-aware spatial attn (adjacency+geodesic
src/models/graph_salad/denoiser.py:229:            use_graph_bias=(spatial_mode == "graph"))
src/models/graph_salad/denoiser.py:252:        validate_inputs: bool = False,
src/models/graph_salad/denoiser.py:270:        # `validate_inputs=False` for hot-path; preflight callers pass True once.
src/models/graph_salad/denoiser.py:272:            x_sp_in, adj_exp, geo_exp, cmask_exp, validate_inputs=validate_inputs
src/models/graph_salad/denoiser.py:337:        spatial_mode: str = "graph",
src/models/graph_salad/denoiser.py:345:        if spatial_mode not in ("graph", "plain"):
src/models/graph_salad/denoiser.py:347:                f"spatial_mode must be 'graph' (graph-aware spatial attn) or "
src/models/graph_salad/denoiser.py:348:                f"'plain' (no_graph_spatial ablation), got {spatial_mode!r}"
src/models/graph_salad/denoiser.py:370:        self.spatial_mode = spatial_mode
src/models/graph_salad/denoiser.py:400:                                        text_mode=text_mode, spatial_mode=spatial_mode)
src/models/graph_salad/denoiser.py:423:        adjacency: torch.Tensor,
src/models/graph_salad/denoiser.py:424:        geodesic_dist: torch.Tensor,
src/models/graph_salad/denoiser.py:431:        validate_inputs: bool = False,
src/models/graph_salad/denoiser.py:446:            validate_inputs: passed to inner GraphAttentionBlock calls. Use True
src/models/graph_salad/denoiser.py:448:                contract checks on pooled adjacency/geodesic/mask; thereafter
src/models/graph_salad/denoiser.py:549:        if adjacency.shape != (B, C, C):
src/models/graph_salad/denoiser.py:551:                f"adjacency must be [B={B}, C={C}, C={C}], got {tuple(adjacency.shape)}"
src/models/graph_salad/denoiser.py:553:        if geodesic_dist.shape != (B, C, C):
src/models/graph_salad/denoiser.py:555:                f"geodesic_dist must be [B={B}, C={C}, C={C}], got {tuple(geodesic_dist.shape)}"
src/models/graph_salad/denoiser.py:558:            ("adjacency", adjacency),
src/models/graph_salad/denoiser.py:559:            ("geodesic_dist", geodesic_dist),
src/models/graph_salad/denoiser.py:574:            ("adjacency", adjacency),
src/models/graph_salad/denoiser.py:575:            ("geodesic_dist", geodesic_dist),
src/models/graph_salad/denoiser.py:639:                adjacency, geodesic_dist, coarse_mask, frame_mask,
src/models/graph_salad/denoiser.py:640:                validate_inputs=validate_inputs,
src/models/graph_salad/denoiser.py:649:            adjacency, geodesic_dist, coarse_mask, frame_mask,
src/models/graph_salad/denoiser.py:650:            validate_inputs=validate_inputs,
src/models/graph_salad/denoiser.py:663:                adjacency, geodesic_dist, coarse_mask, frame_mask,
src/models/graph_salad/denoiser.py:664:                validate_inputs=validate_inputs,

exec
/usr/bin/bash -lc "nl -ba src/models/graph_salad/denoiser.py | sed -n '205,310p;330,410p;430,590p;590,675p'" in /iridisfs/scratch/ts1v23/workspace/noKslot_clean
 succeeded in 0ms:
   205	      - "token_cross_attn": motion tokens cross-attend projected text tokens
   206	        [B,L,D] (key-padding-masked; CFG-uncond rows zeroed). The additive mean
   207	        path is NOT used in this mode.
   208	    """
   209	
   210	    def __init__(
   211	        self,
   212	        d_model: int,
   213	        n_heads: int,
   214	        d_ff: int,
   215	        d_t: int,
   216	        dropout: float = 0.1,
   217	        text_mode: str = "mean_additive",
   218	        spatial_mode: str = "graph",
   219	    ) -> None:
   220	        super().__init__()
   221	        self.text_mode = text_mode
   222	        self.spatial_mode = spatial_mode
   223	        # spatial_mode="graph" (default): graph-aware spatial attn (adjacency+geodesic
   224	        # bias). "plain": no_graph_spatial ablation — plain slot self-attn (no topo
   225	        # bias), still node-masked; pooled_skeleton_embeddings additive (top-level
   226	        # input_proj) is unchanged, so the model still knows "what segment" each slot is.
   227	        self.spatial = GraphAttentionBlock(
   228	            d_model, n_heads, d_ff, dropout=dropout,
   229	            use_graph_bias=(spatial_mode == "graph"))
   230	        self.temporal = TemporalSelfAttention(d_model, n_heads, dropout=dropout)
   231	        self.film_after_spatial = DenseFiLM(d_t, d_model)
   232	        self.film_after_temporal = DenseFiLM(d_t, d_model)
   233	        self.film_after_text = DenseFiLM(d_t, d_model)
   234	        # Token cross-attn sub-block exists in token_cross_attn AND dual_text (so
   235	        # mean-mode state_dict is byte-identical to old ckpts — strict-load
   236	        # preserved). dual_text additionally uses the (always-present) global
   237	        # text_proj path, so its per-layer params == token mode's.
   238	        if text_mode in ("token_cross_attn", "dual_text"):
   239	            self.text_cross_attn = TextCrossAttention(d_model, n_heads, dropout=dropout)
   240	
   241	    def forward(
   242	        self,
   243	        x: torch.Tensor,                  # [B, T_lat, C, D]
   244	        t_emb: torch.Tensor,              # [B, D_t]
   245	        text_cond: torch.Tensor | None,   # [B, D] projected mean text (mean_additive)
   246	        has_text: torch.Tensor,           # [B] bool
   247	        pooled_adj: torch.Tensor,         # [B, C, C]
   248	        pooled_geo: torch.Tensor,         # [B, C, C]
   249	        coarse_mask: torch.Tensor,        # [B, C] bool
   250	        frame_mask: torch.Tensor,         # [B, T_lat] bool
   251	        *,
   252	        validate_inputs: bool = False,
   253	        text_tokens: torch.Tensor | None = None,       # [B, L, D] (token_cross_attn)
   254	        text_key_padding_mask: torch.Tensor | None = None,  # [B, L] bool, True=mask
   255	    ) -> torch.Tensor:
   256	        B, T_lat, C, D = x.shape
   257	
   258	        # --- 1. Spatial graph self-attn (per frame, over C slots) ---
   259	        # Reshape [B, T_lat, C, D] -> [B*T_lat, C, D]; expand graph tensors along T_lat.
   260	        x_sp_in = x.reshape(B * T_lat, C, D)
   261	        adj_exp = (
   262	            pooled_adj.unsqueeze(1).expand(B, T_lat, C, C).reshape(B * T_lat, C, C)
   263	        )
   264	        geo_exp = (
   265	            pooled_geo.unsqueeze(1).expand(B, T_lat, C, C).reshape(B * T_lat, C, C)
   266	        )
   267	        cmask_exp = (
   268	            coarse_mask.unsqueeze(1).expand(B, T_lat, C).reshape(B * T_lat, C)
   269	        )
   270	        # `validate_inputs=False` for hot-path; preflight callers pass True once.
   271	        x_sp = self.spatial(
   272	            x_sp_in, adj_exp, geo_exp, cmask_exp, validate_inputs=validate_inputs
   273	        )
   274	        x = x_sp.reshape(B, T_lat, C, D)
   275	
   276	        # --- 2. FiLM after spatial ---
   277	        x = self.film_after_spatial(x, t_emb)
   278	
   279	        # --- 3. Temporal self-attn (per slot, over T_lat frames) ---
   280	        # [B, T_lat, C, D] -> [B*C, T_lat, D]
   281	        x_t_in = x.permute(0, 2, 1, 3).contiguous().reshape(B * C, T_lat, D)
   282	        fmask_exp = frame_mask.unsqueeze(1).expand(B, C, T_lat).reshape(B * C, T_lat)
   283	        x_t = self.temporal(x_t_in, fmask_exp)
   284	        x = x_t.reshape(B, C, T_lat, D).permute(0, 2, 1, 3).contiguous()
   285	
   286	        # --- 4. FiLM after temporal ---
   287	        x = self.film_after_temporal(x, t_emb)
   288	
   289	        # --- 5. Text conditioning (mode-dependent) ---
   290	        # dual_text runs BOTH sub-blocks (token cross-attn THEN global add) in the
   291	        # SAME per-layer slot — the spatial/temporal FiLM ordering is unchanged, so
   292	        # mean_additive / token_cross_attn behave byte-identically.
   293	        if self.text_mode in ("token_cross_attn", "dual_text"):
   294	            # Motion tokens [B,T*C,D] cross-attend text tokens [B,L,D]. CFG-uncond
   295	            # rows (all text keys masked) get zero output (TextCrossAttention).
   296	            q = x.reshape(B, T_lat * C, D)
   297	            ca = self.text_cross_attn(q, text_tokens, text_key_padding_mask)
   298	            x = x + ca.reshape(B, T_lat, C, D)
   299	        if self.text_mode in ("mean_additive", "dual_text"):
   300	            # broadcast-add projected mean/global text (gated by has_text → CFG).
   301	            text_gated = text_cond * has_text[:, None].to(text_cond.dtype)  # [B, D]
   302	            x = x + text_gated[:, None, None, :]
   303	
   304	        # --- 6. FiLM after text ---
   305	        x = self.film_after_text(x, t_emb)
   306	
   307	        # --- 7. Padded re-mask: padded slots/frames must be zero after each layer ---
   308	        cm = coarse_mask[:, None, :, None].to(x.dtype)   # [B, 1, C, 1]
   309	        fm = frame_mask[:, :, None, None].to(x.dtype)    # [B, T_lat, 1, 1]
   310	        x = x * cm * fm
   330	        d_ff: int | None = None,
   331	        n_layers: int = 5,
   332	        d_text: int = 768,
   333	        d_t: int | None = None,
   334	        dropout: float = 0.1,
   335	        text_mode: str = "mean_additive",
   336	        text_token_dim: int = 768,
   337	        spatial_mode: str = "graph",
   338	    ) -> None:
   339	        super().__init__()
   340	        if text_mode not in ("mean_additive", "token_cross_attn", "dual_text"):
   341	            raise ValueError(
   342	                f"text_mode must be 'mean_additive', 'token_cross_attn' or "
   343	                f"'dual_text', got {text_mode!r}"
   344	            )
   345	        if spatial_mode not in ("graph", "plain"):
   346	            raise ValueError(
   347	                f"spatial_mode must be 'graph' (graph-aware spatial attn) or "
   348	                f"'plain' (no_graph_spatial ablation), got {spatial_mode!r}"
   349	            )
   350	        if n_layers % 2 == 0:
   351	            raise ValueError(
   352	                f"n_layers must be odd for SALAD skip-transformer "
   353	                f"(enc + mid + dec); got {n_layers}"
   354	            )
   355	        if d_model % n_heads != 0:
   356	            raise ValueError(f"d_model ({d_model}) must divide n_heads ({n_heads})")
   357	        if d_ff is None:
   358	            d_ff = 4 * d_model
   359	        if d_t is None:
   360	            d_t = d_model
   361	
   362	        self.d_model = d_model
   363	        self.n_heads = n_heads
   364	        self.d_ff = d_ff
   365	        self.n_layers = n_layers
   366	        self.d_text = d_text
   367	        self.d_t = d_t
   368	        self.text_mode = text_mode
   369	        self.text_token_dim = text_token_dim
   370	        self.spatial_mode = spatial_mode
   371	
   372	        # --- Timestep embedding (shared across all layers' FiLMs) ---
   373	        self.t_sin = SinusoidalTimestepEmbedding(d_t)
   374	        self.t_mlp = nn.Sequential(
   375	            nn.Linear(d_t, d_t * 4),
   376	            nn.SiLU(),
   377	            nn.Linear(d_t * 4, d_t),
   378	        )
   379	
   380	        # --- Text projection (T5-base 768 → d_model); shared across layers ---
   381	        # Per design §2.3: denoiser owns its own text_proj (NOT reusing VAE's).
   382	        # mean_additive: projects the [B,768] mean-pooled caption.
   383	        self.text_proj = nn.Linear(d_text, d_model)
   384	        # token_cross_attn / dual_text: separate projection for token-level T5
   385	        # [B,L,768]→[B,L,D] (exists only in these modes → mean-mode ckpts stay
   386	        # byte-identical). dual_text uses BOTH text_proj (global) and text_token_proj.
   387	        if text_mode in ("token_cross_attn", "dual_text"):
   388	            self.text_token_proj = nn.Linear(text_token_dim, d_model)
   389	
   390	        # --- Input projection: latent z + slot conditioning ---
   391	        self.input_proj = nn.Linear(d_model, d_model)
   392	
   393	        # --- Skip-transformer stack ---
   394	        # n_layers = 2*depth + 1; depth pairs (enc[i], dec[i]) + 1 middle.
   395	        # For n_layers=5: depth=2, layers = [enc0, enc1, mid, dec0, dec1] with
   396	        # skip(enc1->dec0) and skip(enc0->dec1).
   397	        self.layers = nn.ModuleList(
   398	            [
   399	                GraphSaladDenoiserLayer(d_model, n_heads, d_ff, d_t, dropout=dropout,
   400	                                        text_mode=text_mode, spatial_mode=spatial_mode)
   401	                for _ in range(n_layers)
   402	            ]
   403	        )
   404	        # Skip mergers: one per decoder layer
   405	        self.depth = (n_layers - 1) // 2
   406	        self.skip_mergers = nn.ModuleList(
   407	            [nn.Linear(2 * d_model, d_model) for _ in range(self.depth)]
   408	        )
   409	
   410	        # --- Output projection: D → D (predicts v_pred at same dim as z_t) ---
   430	        has_text: torch.Tensor | None = None,
   431	        validate_inputs: bool = False,
   432	        text_token_mask: torch.Tensor | None = None,
   433	        text_tokens: torch.Tensor | None = None,
   434	    ) -> torch.Tensor:
   435	        """Returns v_pred [B, T_lat, C, D].
   436	
   437	        text/text_tokens contract by text_mode:
   438	          - mean_additive:    text=[B,768] global; text_tokens=None.
   439	          - token_cross_attn: text=[B,L,768] tokens; text_tokens=None.
   440	          - dual_text:        text=[B,768] global AND text_tokens=[B,L,768] tokens
   441	                              (+ text_token_mask). Both streams gated by has_text.
   442	
   443	        See module docstring for input/output contracts.
   444	
   445	        Args:
   446	            validate_inputs: passed to inner GraphAttentionBlock calls. Use True
   447	                on the very first iter / sampling step to run the full ~14
   448	                contract checks on pooled adjacency/geodesic/mask; thereafter
   449	                pass False to keep the hot-path lean. (Per design §2.1 spatial
   450	                block bullet — "冷启动验,热路径关".)
   451	        """
   452	        if z_t.dim() != 4:
   453	            raise ValueError(f"z_t must be [B, T_lat, C, D], got {tuple(z_t.shape)}")
   454	        B, T_lat, C, D = z_t.shape
   455	        if D != self.d_model:
   456	            raise ValueError(f"z_t last dim {D} != d_model {self.d_model}")
   457	        if pooled_skeleton_embeddings is None:
   458	            raise ValueError(
   459	                "pooled_skeleton_embeddings is required (additive slot "
   460	                "conditioning per design §2.4). Pass enc['pooled_skeleton_embeddings'] "
   461	                "from vae.encode(...) or vae.encode_skeleton_only(...)."
   462	            )
   463	        if pooled_skeleton_embeddings.shape != (B, C, D):
   464	            raise ValueError(
   465	                f"pooled_skeleton_embeddings must be [B, C, D]=[{B},{C},{D}], "
   466	                f"got {tuple(pooled_skeleton_embeddings.shape)}"
   467	            )
   468	        if has_text is None:
   469	            raise ValueError(
   470	                "has_text is required for CFG gating. Pass a [B] bool tensor; "
   471	                "the trainer is responsible for cond_drop (flipping True→False "
   472	                "with cond_drop_prob)."
   473	            )
   474	        if has_text.shape != (B,) or has_text.dtype != torch.bool:
   475	            raise ValueError(
   476	                f"has_text must be [B={B}] bool, got {tuple(has_text.shape)} "
   477	                f"dtype {has_text.dtype}"
   478	            )
   479	        def _check_token_mask(L: int) -> None:
   480	            if (text_token_mask is None or text_token_mask.shape != (B, L)
   481	                    or text_token_mask.dtype != torch.bool):
   482	                raise ValueError(
   483	                    f"{self.text_mode} requires text_token_mask [B={B}, L={L}] bool, "
   484	                    f"got {None if text_token_mask is None else tuple(text_token_mask.shape)}"
   485	                    f"{'' if text_token_mask is None else ' ' + str(text_token_mask.dtype)}"
   486	                )
   487	            if text_token_mask.device != z_t.device:
   488	                raise ValueError(
   489	                    f"text_token_mask.device {text_token_mask.device} != "
   490	                    f"z_t.device {z_t.device}"
   491	                )
   492	        if self.text_mode == "mean_additive":
   493	            if text.dim() != 2 or text.shape[0] != B or text.shape[1] != self.d_text:
   494	                raise ValueError(
   495	                    f"text must be [B={B}, d_text={self.d_text}] for mean_additive "
   496	                    f"(mean-pooled cache); got {tuple(text.shape)}."
   497	                )
   498	            if text_token_mask is not None or text_tokens is not None:
   499	                raise ValueError(
   500	                    "mean_additive: text_token_mask AND text_tokens must both be None."
   501	                )
   502	        elif self.text_mode == "dual_text":
   503	            # global stream: text = mean-pooled [B,768] (like mean_additive)
   504	            if text.dim() != 2 or text.shape[0] != B or text.shape[1] != self.d_text:
   505	                raise ValueError(
   506	                    f"dual_text: text (global mean) must be [B={B}, d_text={self.d_text}]; "
   507	                    f"got {tuple(text.shape)}."
   508	                )
   509	            # token stream: text_tokens = token-level [B,L,768] (separate keyword arg)
   510	            if (text_tokens is None or text_tokens.dim() != 3
   511	                    or text_tokens.shape[0] != B
   512	                    or text_tokens.shape[2] != self.text_token_dim):
   513	                raise ValueError(
   514	                    f"dual_text: text_tokens must be [B={B}, L, "
   515	                    f"text_token_dim={self.text_token_dim}]; got "
   516	                    f"{None if text_tokens is None else tuple(text_tokens.shape)}."
   517	                )
   518	            _check_token_mask(text_tokens.shape[1])
   519	        else:  # token_cross_attn (single token stream via `text`)
   520	            if (text.dim() != 3 or text.shape[0] != B
   521	                    or text.shape[2] != self.text_token_dim):
   522	                raise ValueError(
   523	                    f"text must be [B={B}, L, text_token_dim={self.text_token_dim}] "
   524	                    f"for token_cross_attn; got {tuple(text.shape)}."
   525	                )
   526	            if text_tokens is not None:
   527	                raise ValueError(
   528	                    "token_cross_attn: tokens go via `text`; text_tokens must be None "
   529	                    "(text_tokens is the dual_text global+token split)."
   530	                )
   531	            _check_token_mask(text.shape[1])
   532	        if timesteps.shape != (B,):
   533	            raise ValueError(f"timesteps must be [B={B}], got {tuple(timesteps.shape)}")
   534	        if coarse_mask.shape != (B, C) or coarse_mask.dtype != torch.bool:
   535	            raise ValueError(
   536	                f"coarse_mask must be [B={B}, C={C}] bool, got "
   537	                f"{tuple(coarse_mask.shape)} {coarse_mask.dtype}"
   538	            )
   539	        if frame_mask.shape != (B, T_lat) or frame_mask.dtype != torch.bool:
   540	            raise ValueError(
   541	                f"frame_mask must be [B={B}, T_lat={T_lat}] bool, got "
   542	                f"{tuple(frame_mask.shape)} {frame_mask.dtype}"
   543	            )
   544	        # P1 (codex 2026-05-23): explicit shape/device/dtype for graph tensors.
   545	        # Without these, a [1, C, C] graph would broadcast-expand into the
   546	        # B-axis via `.unsqueeze().expand()` and silently condition multiple
   547	        # samples on the wrong topology.
   548	        ref_device = z_t.device
   549	        if adjacency.shape != (B, C, C):
   550	            raise ValueError(
   551	                f"adjacency must be [B={B}, C={C}, C={C}], got {tuple(adjacency.shape)}"
   552	            )
   553	        if geodesic_dist.shape != (B, C, C):
   554	            raise ValueError(
   555	                f"geodesic_dist must be [B={B}, C={C}, C={C}], got {tuple(geodesic_dist.shape)}"
   556	            )
   557	        for name, t in (
   558	            ("adjacency", adjacency),
   559	            ("geodesic_dist", geodesic_dist),
   560	            ("coarse_mask", coarse_mask),
   561	            ("frame_mask", frame_mask),
   562	            ("text", text),
   563	            ("pooled_skeleton_embeddings", pooled_skeleton_embeddings),
   564	            ("has_text", has_text),
   565	            ("timesteps", timesteps),
   566	        ):
   567	            if t.device != ref_device:
   568	                raise ValueError(
   569	                    f"GraphSaladDenoiser: {name}.device {t.device} != z_t.device {ref_device}"
   570	                )
   571	        # float tensors must match z_t.dtype (denoiser is fp32-only — same
   572	        # contract as GraphAttentionBlock)
   573	        for name, t in (
   574	            ("adjacency", adjacency),
   575	            ("geodesic_dist", geodesic_dist),
   576	            ("text", text),
   577	            ("pooled_skeleton_embeddings", pooled_skeleton_embeddings),
   578	        ):
   579	            if t.dtype != z_t.dtype:
   580	                raise ValueError(
   581	                    f"GraphSaladDenoiser: {name}.dtype {t.dtype} != z_t.dtype {z_t.dtype}"
   582	                )
   583	        if text_tokens is not None and (text_tokens.device != ref_device
   584	                                        or text_tokens.dtype != z_t.dtype):
   585	            raise ValueError(
   586	                f"GraphSaladDenoiser: text_tokens device/dtype "
   587	                f"({text_tokens.device}/{text_tokens.dtype}) must match z_t "
   588	                f"({ref_device}/{z_t.dtype})."
   589	            )
   590	
   590	
   591	        # --- Timestep embedding (shared by all FiLMs) ---
   592	        t_emb = self.t_mlp(self.t_sin(timesteps))         # [B, D_t]
   593	
   594	        # --- Text prep (mode-dependent) ---
   595	        # mean_additive: project the [B,768] mean caption → [B,D], gated per-layer.
   596	        # token_cross_attn: project tokens [B,L,768]→[B,L,D] + build the per-layer
   597	        #   key_padding_mask (True = ignore): pad-token OR has_text=False. An
   598	        #   all-masked (uncond) row is handled by TextCrossAttention (output→0).
   599	        # dual_text: BOTH — global text_cond (from `text`) + token tok_emb (from the
   600	        #   separate `text_tokens` arg). One shared key_padding_mask gates the token
   601	        #   stream; has_text gates the global stream → both CFG-drop together.
   602	        # NOTE: local projected tokens are `tok_emb` (NOT `text_tokens`) so the
   603	        # dual_text `text_tokens` forward ARG is not clobbered.
   604	        text_cond = None
   605	        tok_emb = None
   606	        text_key_padding_mask = None
   607	        if self.text_mode == "mean_additive":
   608	            text_cond = self.text_proj(text)              # [B, D]
   609	        elif self.text_mode == "dual_text":
   610	            text_cond = self.text_proj(text)              # global [B, D]
   611	            tok_emb = self.text_token_proj(text_tokens)   # tokens [B, L, D]
   612	            valid = text_token_mask & has_text[:, None]   # [B, L] bool
   613	            text_key_padding_mask = ~valid                # [B, L] True=mask
   614	        else:  # token_cross_attn (tokens via `text`)
   615	            tok_emb = self.text_token_proj(text)          # [B, L, D]
   616	            # valid key = token present AND has_text=True. key_padding_mask is
   617	            # the inverse (True ⇒ mask). has_text=False ⇒ whole row masked.
   618	            valid = text_token_mask & has_text[:, None]   # [B, L] bool
   619	            text_key_padding_mask = ~valid                # [B, L] True=mask
   620	
   621	        # --- Input projection: latent + slot conditioning (additive) ---
   622	        x = self.input_proj(z_t)                          # [B, T_lat, C, D]
   623	        x = x + pooled_skeleton_embeddings.unsqueeze(1).expand(-1, T_lat, -1, -1)
   624	        # Apply input padded re-mask (defense in depth)
   625	        cm = coarse_mask[:, None, :, None].to(x.dtype)
   626	        fm = frame_mask[:, :, None, None].to(x.dtype)
   627	        x = x * cm * fm
   628	
   629	        # --- Skip-transformer: enc → mid → dec with paired skips ---
   630	        # depth = (n_layers - 1) // 2 (e.g. 2 for n_layers=5)
   631	        # layout: layers[0..depth-1] = encoders, layers[depth] = mid,
   632	        #         layers[depth+1..2*depth] = decoders (in order)
   633	        # skip pairing: dec[i] consumes encoder output enc[depth-1-i] (i.e.
   634	        # symmetric around the middle).
   635	        enc_outputs: list[torch.Tensor] = []
   636	        for i in range(self.depth):
   637	            x = self.layers[i](
   638	                x, t_emb, text_cond, has_text,
   639	                adjacency, geodesic_dist, coarse_mask, frame_mask,
   640	                validate_inputs=validate_inputs,
   641	                text_tokens=tok_emb,
   642	                text_key_padding_mask=text_key_padding_mask,
   643	            )
   644	            enc_outputs.append(x)
   645	
   646	        # middle layer
   647	        x = self.layers[self.depth](
   648	            x, t_emb, text_cond, has_text,
   649	            adjacency, geodesic_dist, coarse_mask, frame_mask,
   650	            validate_inputs=validate_inputs,
   651	            text_tokens=tok_emb,
   652	            text_key_padding_mask=text_key_padding_mask,
   653	        )
   654	
   655	        # decoder layers with skip
   656	        for i in range(self.depth):
   657	            dec_layer = self.layers[self.depth + 1 + i]
   658	            skip = enc_outputs[self.depth - 1 - i]    # reverse pair
   659	            x = torch.cat([x, skip], dim=-1)          # [B, T_lat, C, 2D]
   660	            x = self.skip_mergers[i](x)               # [B, T_lat, C, D]
   661	            x = dec_layer(
   662	                x, t_emb, text_cond, has_text,
   663	                adjacency, geodesic_dist, coarse_mask, frame_mask,
   664	                validate_inputs=validate_inputs,
   665	                text_tokens=tok_emb,
   666	                text_key_padding_mask=text_key_padding_mask,
   667	            )
   668	
   669	        # --- Output: pre-norm + zero-init linear (init v_pred ≈ 0) ---
   670	        x = self.output_norm(x)
   671	        v_pred = self.output_proj(x)
   672	        # Final padded re-mask
   673	        v_pred = v_pred * cm * fm
   674	        return v_pred

codex
I found the plain attention path itself gates the two bias projections, but validation and denoiser-level shape checks still require graph tensors even in plain mode. I’m checking whether that is just an API contract or a real failure mode for the claimed ablation.
exec
/usr/bin/bash -lc "nl -ba src/models/graph_salad/attention.py | sed -n '145,330p'" in /iridisfs/scratch/ts1v23/workspace/noKslot_clean
 succeeded in 0ms:
   145	            return self._compute(x, adjacency, geodesic_dist, node_mask)
   146	
   147	        if x.dim() != 3 or x.shape[-1] != self.d_model:
   148	            raise ValueError(
   149	                f"x must be [B, N, {self.d_model}], got {tuple(x.shape)}"
   150	            )
   151	        B, N, _ = x.shape
   152	        # Reject empty batches loud: zero-sized B/N silently produce zero-sized
   153	        # output, which downstream pool/VAE consumes as "everything padded" and
   154	        # cascades into shape-only-OK-but-meaningless errors (codex M1.2 round 2 R12).
   155	        if B <= 0 or N <= 0:
   156	            raise ValueError(
   157	                f"GraphAttentionBlock: batch B={B} and node count N={N} must be > 0"
   158	            )
   159	        if adjacency.shape != (B, N, N) or geodesic_dist.shape != (B, N, N):
   160	            raise ValueError(
   161	                f"adjacency/geodesic_dist must be [B={B}, N={N}, N={N}], "
   162	                f"got {tuple(adjacency.shape)} and {tuple(geodesic_dist.shape)}"
   163	            )
   164	        if node_mask.shape != (B, N) or node_mask.dtype != torch.bool:
   165	            raise ValueError(
   166	                f"node_mask must be [B={B}, N={N}] bool, got "
   167	                f"shape {tuple(node_mask.shape)} dtype {node_mask.dtype}"
   168	            )
   169	
   170	        # --- R12 fail-loud: dtype consistency (codex M1.2 round 5) ---
   171	        # All float tensors must (a) be fp32 or fp64 (fp16/bf16 overflow at
   172	        # softmax with -1e9 mask sentinel and at large bias terms), and
   173	        # (b) match the module's parameter dtype (mixed-dtype matmul crashes
   174	        # opaquely deep in attention compute).
   175	        # bf16-safe (2026-06-03): bf16 IS allowed — its 8-bit exponent (range ±3e38,
   176	        # same as fp32) does NOT overflow the -1e9 softmax sentinel or the additive
   177	        # topology bias; softmax is forced to fp32 in _compute below. fp16 is STILL
   178	        # rejected (5-bit exponent overflows at -1e9). The fp32/fp64 path is
   179	        # byte-for-byte unchanged. Under autocast(bf16), x may be bf16 while module
   180	        # weights stay fp32 — a valid autocast pattern (matmul casts internally), so
   181	        # the strict x.dtype==weight.dtype check is enforced ONLY on the fp32/64 path.
   182	        expected_dtype = self.q_proj.weight.dtype
   183	        for name, t in (("x", x), ("adjacency", adjacency), ("geodesic_dist", geodesic_dist)):
   184	            if t.dtype not in (torch.float32, torch.float64, torch.bfloat16):
   185	                raise ValueError(
   186	                    f"GraphAttentionBlock: {name}.dtype must be float32/float64/bfloat16, "
   187	                    f"got {t.dtype} (fp16 unsupported: 5-bit exponent overflows the "
   188	                    f"-1e9 softmax sentinel + additive bias)"
   189	                )
   190	            if t.dtype in (torch.float32, torch.float64) and t.dtype != expected_dtype:
   191	                raise ValueError(
   192	                    f"GraphAttentionBlock: {name}.dtype {t.dtype} != module dtype "
   193	                    f"{expected_dtype} (cast inputs OR module to match)"
   194	                )
   195	
   196	        # --- R12 fail-loud: finite + topology semantic checks ---
   197	        # (codex M1.2 round 1 + 3)
   198	        # x: all entries must be finite.
   199	        if not torch.isfinite(x).all():
   200	            raise ValueError(
   201	                "GraphAttentionBlock: x contains NaN or Inf"
   202	            )
   203	        # adjacency contract: finite, non-negative, symmetric, zero diagonal.
   204	        # We allow weighted edges (not only binary {0,1}) to support pool
   205	        # variants that may emit soft-weighted pooled adjacency, but the
   206	        # geometric meaning must remain: undirected graph with no self-loops.
   207	        if not torch.isfinite(adjacency).all():
   208	            raise ValueError(
   209	                "GraphAttentionBlock: adjacency contains NaN or Inf"
   210	            )
   211	        if (adjacency < 0).any():
   212	            raise ValueError(
   213	                "GraphAttentionBlock: adjacency contains negative values "
   214	                "(edges must be non-negative weights)"
   215	            )
   216	        if (adjacency > 1.0).any():
   217	            raise ValueError(
   218	                "GraphAttentionBlock: adjacency contains values > 1.0; "
   219	                "contract is binary {0,1} or soft [0,1] (large magnitudes "
   220	                "would dominate the additive bias projection)"
   221	            )
   222	        # Symmetry with rtol=0 so large absolute asymmetry (e.g. 1e6 vs 1e6+1)
   223	        # cannot slip past allclose's default relative tolerance (codex M1.2
   224	        # round 4 R12 fix).
   225	        if not torch.allclose(
   226	            adjacency, adjacency.transpose(-2, -1), atol=1e-6, rtol=0.0
   227	        ):
   228	            raise ValueError(
   229	                "GraphAttentionBlock: adjacency is not symmetric "
   230	                "(undirected graph required)"
   231	            )
   232	        if (adjacency.diagonal(dim1=-2, dim2=-1) != 0).any():
   233	            raise ValueError(
   234	                "GraphAttentionBlock: adjacency has non-zero diagonal "
   235	                "(self-loops not permitted)"
   236	            )
   237	        # geodesic_dist contract: no NaN, no -Inf (+Inf is legitimate per Floyd
   238	        # unreachable-pair contract), non-negative on finite entries, symmetric
   239	        # on finite entries, zero diagonal at valid nodes.
   240	        if torch.isnan(geodesic_dist).any():
   241	            raise ValueError(
   242	                "GraphAttentionBlock: geodesic_dist contains NaN"
   243	            )
   244	        if (geodesic_dist == float("-inf")).any():
   245	            raise ValueError(
   246	                "GraphAttentionBlock: geodesic_dist contains -Inf "
   247	                "(bug; only +Inf is legitimate for unreachable pairs)"
   248	            )
   249	        # Negative finite distances are nonsense (Floyd output is hop count ≥ 0).
   250	        finite_geo = geodesic_dist[torch.isfinite(geodesic_dist)]
   251	        if (finite_geo < 0).any():
   252	            raise ValueError(
   253	                "GraphAttentionBlock: geodesic_dist has negative finite entries "
   254	                "(distances must be ≥ 0)"
   255	            )
   256	        # Floyd hop-count upper bound: max hops ≤ N-1 (line-graph case on N
   257	        # nodes). Larger finite values would dominate the additive bias and
   258	        # silently corrupt attention. Codex M1.2 round 5 R12 fix.
   259	        if (finite_geo > (N - 1)).any():
   260	            raise ValueError(
   261	                f"GraphAttentionBlock: geodesic_dist has finite entries > {N - 1} "
   262	                f"(max hop-count on {N} nodes); not a valid Floyd shortest-path output"
   263	            )
   264	        # Symmetry: two-stage check (codex M1.2 round 4 R12 fix).
   265	        # (1) The finite/+Inf pattern must be symmetric — i.e. cell is finite
   266	        #     iff its transpose is. Otherwise asymmetric reachability slips
   267	        #     through e.g. geo[0,1]=+Inf, geo[1,0]=1.0.
   268	        # (2) Where both sides are finite, values must allclose with rtol=0.
   269	        gt = geodesic_dist.transpose(-2, -1)
   270	        finite_g = torch.isfinite(geodesic_dist)
   271	        finite_gt = torch.isfinite(gt)
   272	        if not torch.equal(finite_g, finite_gt):
   273	            raise ValueError(
   274	                "GraphAttentionBlock: geodesic_dist finite/+Inf pattern is not "
   275	                "symmetric (asymmetric reachability)"
   276	            )
   277	        both_finite = finite_g & finite_gt
   278	        if not torch.allclose(
   279	            geodesic_dist[both_finite], gt[both_finite], atol=1e-6, rtol=0.0
   280	        ):
   281	            raise ValueError(
   282	                "GraphAttentionBlock: geodesic_dist is not symmetric on finite entries"
   283	            )
   284	        # Diagonal of geodesic at valid nodes must be zero (i->i distance).
   285	        diag = geodesic_dist.diagonal(dim1=-2, dim2=-1)  # [B, N]
   286	        if ((diag != 0) & node_mask).any():
   287	            raise ValueError(
   288	                "GraphAttentionBlock: geodesic_dist has non-zero diagonal "
   289	                "at valid nodes (i→i distance must be 0)"
   290	            )
   291	
   292	        # Per-sample sanity: at least one valid node per batch element.
   293	        # This MUST run before adj/geo cross-consistency below, since Floyd on
   294	        # an all-False mask sample produces all-+Inf which would falsely trip
   295	        # the reachability-pattern check.
   296	        if (~node_mask.any(dim=1)).any():
   297	            bad = (~node_mask.any(dim=1)).nonzero(as_tuple=False).flatten().tolist()
   298	            raise ValueError(
   299	                f"GraphAttentionBlock: node_mask has all-False rows for "
   300	                f"batch element(s) {bad} (no valid nodes; attention undefined)"
   301	            )
   302	
   303	        # --- R12 fail-loud: adj/geo cross-consistency (codex M1.2 round 6) ---
   304	        # geodesic_dist must equal floyd_shortest_path(adjacency, node_mask).
   305	        # Without this, a bounded-but-wrong geo (e.g. correct adj, geo[0,3]=2
   306	        # instead of 3 on a 4-node line) silently corrupts the additive topology
   307	        # bias and skews attention. Costs O(B·N^3) per forward — acceptable at
   308	        # N≤160 (~4M ops, <1ms on GPU).
   309	        expected_geo = floyd_shortest_path(adjacency, node_mask)
   310	        both_valid = node_mask[:, :, None] & node_mask[:, None, :]
   311	        # Pattern check: reachability (finite/+Inf) must match on valid pairs.
   312	        finite_actual = torch.isfinite(geodesic_dist) & both_valid
   313	        finite_expected = torch.isfinite(expected_geo) & both_valid
   314	        if not torch.equal(finite_actual, finite_expected):
   315	            raise ValueError(
   316	                "GraphAttentionBlock: geodesic_dist reachability pattern "
   317	                "inconsistent with adjacency (Floyd-recomputed)"
   318	            )
   319	        # Value check on entries that are finite in BOTH:
   320	        compare_mask = finite_actual & finite_expected
   321	        if not torch.allclose(
   322	            geodesic_dist[compare_mask], expected_geo[compare_mask],
   323	            atol=1e-6, rtol=0.0,
   324	        ):
   325	            raise ValueError(
   326	                "GraphAttentionBlock: geodesic_dist values inconsistent with "
   327	                "shortest-path over adjacency (Floyd-recomputed)"
   328	            )
   329	
   330	        return self._compute(x, adjacency, geodesic_dist, node_mask)

exec
/usr/bin/bash -lc "nl -ba scripts/train_denoiser.py | sed -n '300,345p;590,690p;690,735p;840,870p;985,1005p;1030,1055p'" in /iridisfs/scratch/ts1v23/workspace/noKslot_clean
 succeeded in 0ms:
   300	                    help="weight on direct latent loss ||z0_hat - z0||² (0 = off; "
   301	                         "keep 0 in the first dynamics run per the handoff).")
   302	    ap.add_argument("--latent_dyn_target", default="sample", choices=["sample", "mu"],
   303	                    help="latent-dynamics loss target: 'sample' (z0 ~ posterior, "
   304	                         "matches the v-target; default) or 'mu' (posterior mean, "
   305	                         "less noisy fallback if sample makes the loss unstable).")
   306	    ap.add_argument("--grad_clip", type=float, default=1.0)
   307	    # Denoiser arch
   308	    ap.add_argument("--n_layers", type=int, default=5)
   309	    ap.add_argument("--d_ff", type=int, default=None,
   310	                    help="default = 4 * d_model")
   311	    ap.add_argument("--dropout", type=float, default=0.1)
   312	    # Diffusion
   313	    ap.add_argument("--num_train_timesteps", type=int, default=1000)
   314	    ap.add_argument("--beta_start", type=float, default=0.00085)
   315	    ap.add_argument("--beta_end", type=float, default=0.012)
   316	    ap.add_argument("--beta_schedule", default="scaled_linear")
   317	    # M2 token-level text conditioning (optional). Default mean_additive keeps the
   318	    # current behavior + old-ckpt strict-load. token_cross_attn requires the token
   319	    # cache built by scripts/precompute_t5_caption_tokens.py.
   320	    ap.add_argument("--text_mode",
   321	                    choices=["mean_additive", "token_cross_attn", "dual_text"],
   322	                    default="mean_additive",
   323	                    help="mean_additive (default): mean-pooled T5 additive broadcast "
   324	                         "(byte-equiv to current; old ckpts strict-load). "
   325	                         "token_cross_attn: per-layer cross-attention over token-level "
   326	                         "T5. dual_text: BOTH streams — global mean-add + token "
   327	                         "cross-attn (both CFG-gated by has_text). "
   328	                         "token_cross_attn/dual_text need --caption_token_cache.")
   329	    ap.add_argument("--spatial_mode", choices=["graph", "plain"], default="graph",
   330	                    help="backbone spatial attention. 'graph' (default): graph-aware "
   331	                         "(adjacency+geodesic bias); old ckpts strict-load. 'plain': "
   332	                         "no_graph_spatial ablation — plain slot self-attn (no topo "
   333	                         "bias), still node-masked + pooled_skeleton additive kept.")
   334	    ap.add_argument("--caption_token_cache", default=None,
   335	                    help="prefix of the token cache (<prefix>.tokens.npy + "
   336	                         ".token_mask.npy + .keys.json). REQUIRED when "
   337	                         "--text_mode token_cross_attn. The cache MUST reuse the "
   338	                         "mean cache's keys.json order (idx-align law).")
   339	    ap.add_argument("--caption_token_max_len", type=int, default=64,
   340	                    help="L: token sequence length of the token cache (must match "
   341	                         "the cache's --max_length).")
   342	    ap.add_argument("--cond_drop_prob", type=float, default=0.1,
   343	                    help="CFG cond-drop probability per sample")
   344	    ap.add_argument("--amp_dtype", choices=["fp32", "bf16"], default="fp32",
   345	                    help="fp32 (default). bf16 is now bf16-SAFE: GraphAttentionBlock, "
   590	    else:
   591	        train_sampler = None
   592	    dl_train = DataLoader(
   593	        ds_train, batch_size=args.batch_size,
   594	        sampler=train_sampler, shuffle=(train_sampler is None),
   595	        collate_fn=anytop_collate_fn, num_workers=args.num_workers,
   596	        drop_last=True, pin_memory=True,
   597	        persistent_workers=(args.num_workers > 0),
   598	        prefetch_factor=4 if args.num_workers > 0 else None,
   599	    )
   600	    dl_val = DataLoader(
   601	        ds_val, batch_size=args.batch_size, shuffle=False,
   602	        collate_fn=anytop_collate_fn, num_workers=max(1, args.num_workers // 2),
   603	        drop_last=False, pin_memory=True,
   604	        persistent_workers=(args.num_workers > 0),
   605	        prefetch_factor=4 if args.num_workers > 0 else None,
   606	    )
   607	
   608	    # ---- Denoiser ----
   609	    d_model = ta["d_model"]
   610	    n_heads = ta["n_heads"]
   611	    d_ff = args.d_ff if args.d_ff is not None else 4 * d_model
   612	    denoiser = GraphSaladDenoiser(
   613	        d_model=d_model, n_heads=n_heads, d_ff=d_ff,
   614	        n_layers=args.n_layers, d_text=768, dropout=args.dropout,
   615	        text_mode=args.text_mode, text_token_dim=768,
   616	        spatial_mode=args.spatial_mode,
   617	    ).to(dev)
   618	    n_params = sum(p.numel() for p in denoiser.parameters())
   619	    log(f"\nDenoiser: n_layers={args.n_layers} d_model={d_model} d_ff={d_ff} "
   620	        f"text_mode={args.text_mode} spatial_mode={args.spatial_mode} params={n_params:,}")
   621	
   622	    # ---- Full resume (--resume): restore MODEL here (before DDP wrap); optimizer
   623	    # + epoch + best_val + global_it are restored further below. Seamless crash
   624	    # continuation — unlike --init_ckpt, Adam moments + epoch counter are kept. ----
   625	    resume_ck = None
   626	    if args.resume is not None:
   627	        if args.init_ckpt is not None:
   628	            raise SystemExit("--resume and --init_ckpt are mutually exclusive")
   629	        if not Path(args.resume).exists():
   630	            raise SystemExit(f"--resume {args.resume!r} does not exist")
   631	        log(f"\nFULL RESUME from {args.resume}")
   632	        resume_ck = torch.load(args.resume, map_location="cpu", weights_only=False)
   633	        # M2: the denoiser was built from CLI --text_mode; the ckpt arch must match
   634	        # (token vs mean state_dicts differ by 134 keys → strict-load would fail
   635	        # cryptically). Assert text_mode agreement FIRST for a clear error.
   636	        ck_text_mode = resume_ck.get("args", {}).get("text_mode", "mean_additive")
   637	        if ck_text_mode != args.text_mode:
   638	            raise SystemExit(
   639	                f"[RESUME FAIL] ckpt text_mode={ck_text_mode!r} != CLI "
   640	                f"--text_mode {args.text_mode!r}. Rebuild with the matching "
   641	                f"text_mode (token/mean arch differ)."
   642	            )
   643	        ck_spatial_mode = resume_ck.get("args", {}).get("spatial_mode", "graph")
   644	        if ck_spatial_mode != args.spatial_mode:
   645	            raise SystemExit(
   646	                f"[RESUME FAIL] ckpt spatial_mode={ck_spatial_mode!r} != CLI "
   647	                f"--spatial_mode {args.spatial_mode!r}. graph/plain arch differ "
   648	                f"(plain drops adjacency/geodesic bias params)."
   649	            )
   650	        missing, unexpected = denoiser.load_state_dict(resume_ck["model_state_dict"], strict=True)
   651	        if missing or unexpected:
   652	            raise SystemExit(
   653	                f"[RESUME FAIL] model missing={len(missing)} unexpected={len(unexpected)}"
   654	            )
   655	        log(f"  loaded model_state_dict strict=True (ckpt epoch={resume_ck.get('epoch')} "
   656	            f"val_denoise={resume_ck.get('val_denoise')})")
   657	
   658	    # ---- Warm-start from --init_ckpt (continuation runs only) ----
   659	    # Mirrors train_graph_vae.py's --init_ckpt pattern: only model weights are
   660	    # restored; AdamW state + epoch counter + best_val + RNG all start fresh.
   661	    # This is the conservative continuation pattern (you lose Adam moments but
   662	    # avoid optimizer-state version skew). For ep1000 → ep3000 continuation,
   663	    # pass --warmup_iters small (e.g. 200) since the model is past the
   664	    # zero-init unstable regime.
   665	    if args.init_ckpt is not None:
   666	        if not Path(args.init_ckpt).exists():
   667	            raise SystemExit(f"--init_ckpt {args.init_ckpt!r} does not exist")
   668	        log(f"\nWarm-start denoiser from {args.init_ckpt}")
   669	        ck = torch.load(args.init_ckpt, map_location="cpu", weights_only=False)
   670	        # M2: warm-start arch must match CLI --text_mode (see resume note above).
   671	        ck_text_mode = ck.get("args", {}).get("text_mode", "mean_additive")
   672	        if ck_text_mode != args.text_mode:
   673	            raise SystemExit(
   674	                f"[INIT_CKPT FAIL] ckpt text_mode={ck_text_mode!r} != CLI "
   675	                f"--text_mode {args.text_mode!r}. Rebuild with matching text_mode."
   676	            )
   677	        ck_spatial_mode = ck.get("args", {}).get("spatial_mode", "graph")
   678	        if ck_spatial_mode != args.spatial_mode:
   679	            raise SystemExit(
   680	                f"[INIT_CKPT FAIL] ckpt spatial_mode={ck_spatial_mode!r} != CLI "
   681	                f"--spatial_mode {args.spatial_mode!r}. graph/plain arch differ."
   682	            )
   683	        sd = ck.get("model_state_dict", ck)
   684	        missing, unexpected = denoiser.load_state_dict(sd, strict=True)
   685	        if missing or unexpected:
   686	            raise SystemExit(
   687	                f"[INIT_CKPT FAIL] missing={len(missing)} unexpected={len(unexpected)}; "
   688	                f"refusing to silently load partial weights"
   689	            )
   690	        prev_ep = ck.get("epoch", "?")
   690	        prev_ep = ck.get("epoch", "?")
   691	        prev_val = ck.get("val_denoise", "?")
   692	        log(f"  loaded model_state_dict strict=True (prev epoch={prev_ep} val_denoise={prev_val})")
   693	        log(f"  optimizer state + epoch counter + best_val are FRESH "
   694	            f"(continuation pattern; pass --warmup_iters {args.warmup_iters} "
   695	            f"to control re-warmup)")
   696	
   697	    # ---- DDP wrap (AFTER warm-start: init_ckpt loads into raw module) ----
   698	    # find_unused_parameters=True: GraphSaladDenoiser has CFG dropout (some text
   699	    # samples gated to zero each batch). Text-projection weights are still
   700	    # touched via masked tensor, but to be safe under varying cond mixes we
   701	    # enable the conservative check. Mirrors train_graph_vae.py.
   702	    if is_ddp:
   703	        denoiser = nn.parallel.DistributedDataParallel(
   704	            denoiser, device_ids=[local_rank], find_unused_parameters=True,
   705	        )
   706	    raw_denoiser = denoiser.module if is_ddp else denoiser
   707	
   708	    # ---- Optimizer + scheduler + lr-warmup ----
   709	    opt = torch.optim.AdamW(
   710	        denoiser.parameters(), lr=args.lr,
   711	        betas=(0.9, 0.99), weight_decay=args.weight_decay,
   712	    )
   713	    if resume_ck is not None:
   714	        opt.load_state_dict(resume_ck["optimizer_state_dict"])
   715	        # optimizer state tensors were loaded on CPU (map_location) → move to the
   716	        # training device, else AdamW.step() hits a cpu/cuda device mismatch.
   717	        for st in opt.state.values():
   718	            for k, v in st.items():
   719	                if torch.is_tensor(v):
   720	                    st[k] = v.to(dev)
   721	        log("  loaded optimizer_state_dict (Adam moments restored + moved to device)")
   722	    sched = DDIMScheduler(
   723	        num_train_timesteps=args.num_train_timesteps,
   724	        beta_start=args.beta_start, beta_end=args.beta_end,
   725	        beta_schedule=args.beta_schedule,
   726	        prediction_type="v_prediction",
   727	        clip_sample=False,
   728	    )
   729	
   730	    # Total per-rank optimizer steps over the whole run (cosine horizon). len(dl_train)
   731	    # is per-rank steps/epoch under the DistributedSampler (drop_last=True), and
   732	    # global_it is the per-rank counter, so both live in the same (per-rank) iter
   733	    # space. Honour --smoke (1 epoch) so a cosine smoke decays over its real
   734	    # one-epoch horizon (codex 019e95f0 #1). NOTE for cosine --resume: global_it is
   735	    # rebuilt from start_epoch*len(dl_train), so a manual resume MUST re-pass the
   840	            # Diffusion: noise + add_noise + v_target
   841	            noise = torch.randn_like(z0)
   842	            timesteps = torch.randint(0, args.num_train_timesteps, (B,), device=dev).long()
   843	            z_t = sched.add_noise(z0, noise, timesteps)
   844	            v_target = sched.get_velocity(z0, noise, timesteps)
   845	            # Mask z_t + v_target at padded positions (defense in depth)
   846	            mask_4d = (coarse_mask[:, None, :, None] & frame_mask[:, :, None, None]).to(z0.dtype)
   847	            z_t = z_t * mask_4d
   848	            v_target = v_target * mask_4d
   849	
   850	            # Denoiser forward + loss under bf16 autocast (inputs stay fp32; autocast
   851	            # casts internal matmuls to bf16). v_pred is bf16; masked_v_mse's
   852	            # (v_pred - v_target) promotes to fp32 (v_target is fp32) so the loss +
   853	            # backward are fp32 — no GradScaler needed.
   854	            with amp_ctx():
   855	                v_pred = denoiser(
   856	                    z_t=z_t, timesteps=timesteps, text=text_in,
   857	                    adjacency=pooled_adj, geodesic_dist=pooled_geo,
   858	                    coarse_mask=coarse_mask, frame_mask=frame_mask,
   859	                    pooled_skeleton_embeddings=pooled_skel,
   860	                    has_text=has_text,
   861	                    text_token_mask=token_mask_in,
   862	                    text_tokens=text_tokens_in,
   863	                    # Validate on the first iter only (cold-start preflight)
   864	                    validate_inputs=(global_it == 0),
   865	                )
   866	                loss_v = masked_v_mse(v_pred, v_target, coarse_mask, frame_mask)
   867	                loss = loss_v
   868	                # M2 latent temporal dynamics loss (handoff 20260605). Gated on
   869	                # weights>0 → zero-weight path is byte-identical (loss == loss_v).
   870	                loss_dz = loss_ddz = loss_x0 = None
   985	                            text_in = batch.caption_token_emb.to(dev)
   986	                            token_mask_in = batch.caption_token_mask.to(dev)
   987	                        else:
   988	                            text_in = batch.caption_emb.to(dev) * has_text[:, None].to(batch.caption_emb.dtype)
   989	                            token_mask_in = None
   990	                        # Codex P2 fix (2026-05-25): rank-0-only val uses raw
   991	                        # module, not DDP wrapper, to avoid any future one-sided
   992	                        # collective if denoiser ever grows DDP-tracked buffers.
   993	                        with amp_ctx():
   994	                            v_pred = raw_denoiser(
   995	                                z_t=z_t, timesteps=timesteps, text=text_in,
   996	                                adjacency=pooled_adj, geodesic_dist=pooled_geo,
   997	                                coarse_mask=coarse_mask, frame_mask=frame_mask,
   998	                                pooled_skeleton_embeddings=pooled_skel,
   999	                                has_text=has_text, validate_inputs=False,
  1000	                                text_token_mask=token_mask_in,
  1001	                                text_tokens=text_tokens_in,
  1002	                            )
  1003	                        diff_sq = (v_pred.float() - v_target).pow(2) * mask_f
  1004	                        val_num += diff_sq.sum().item()
  1005	                        val_den += mask_f.sum().item() * v_pred.shape[-1]
  1030	                log(f"[val ep{epoch}] dt={time.time()-t_v:.1f}s val_denoise={val_loss:.4f}{val_comp_str} "
  1031	                    f"n_valid_positions={int(val_den/v_pred.shape[-1])}")
  1032	
  1033	                metrics_row = {
  1034	                    "epoch": epoch, "train_loss": epoch_loss, "val_denoise": val_loss,
  1035	                    "lr": cur_lr, "epoch_dt_s": ep_dt, "global_it": global_it,
  1036	                }
  1037	                if lat_active:
  1038	                    metrics_row.update({
  1039	                        "train_v_mse": epoch_v_mse,
  1040	                        "train_lat_dz": epoch_lat_dz,
  1041	                        "train_lat_ddz": epoch_lat_ddz,
  1042	                        "train_total": epoch_loss,
  1043	                        "val_lat_dz": val_lat_dz,
  1044	                        "val_lat_ddz": val_lat_ddz,
  1045	                    })
  1046	                    if args.w_lat_x0 > 0:
  1047	                        metrics_row["train_lat_x0"] = epoch_lat_x0
  1048	                        metrics_row["val_lat_x0"] = val_lat_x0
  1049	                metrics_fp.write(json.dumps(metrics_row) + "\n"); metrics_fp.flush()
  1050	
  1051	                # Best ckpt — rank 0 only; unwrap DDP for clean state_dict
  1052	                if val_loss < best_val:
  1053	                    best_val = val_loss
  1054	                    best_path = out_dir / "best_model.pt"
  1055	                    torch.save({

exec
/usr/bin/bash -lc "nl -ba scripts/_launch_diffusion_t2m_4card.sh | sed -n '1,220p'" in /iridisfs/scratch/ts1v23/workspace/noKslot_clean
 succeeded in 0ms:
     1	#!/bin/bash
     2	# Cross-alloc 4-card H100 DDP launcher (2 same-node allocs on swarmh1002).
     3	# DIAGNOSTIC B-mu: latent temporal dynamics loss with --latent_dyn_target mu
     4	# (deterministic posterior-mean latent trajectory as the dz/ddz reference) vs the
     5	# main B run's `sample` target. Reuses the proven cross-alloc pattern (mem: 6-card
     6	# same-node + 8-card xnode): each alloc's srun runs the SAME _launch_diffusion_t2m.sh
     7	# with NNODES=2 + shared MASTER + explicit NODE_RANK; static rendezvous over IB.
     8	#
     9	# Adapted from _launch_diffusion_t2m_6card.sh (3-alloc) -> 2-alloc; COMMON_ENV adds
    10	# the 20-species capacity whitelist + train_split=all + latent-dynamics-loss flags
    11	# (the inner launcher already threads SPECIES_WHITELIST/TRAIN_SPLIT/W_LAT_*/LATENT_DYN_TARGET,
    12	# both codex-PASSED 019e98dc / 019e9a10).
    13	#
    14	# Usage (SMOKE FIRST -- TRUE 4-rank, verify 2-alloc rendezvous + IB NCCL + bs no-OOM,
    15	# 1 epoch; MUST pass before the real run):
    16	#   SMOKE=1 NCCL_DEBUG=INFO bash scripts/_launch_diffusion_t2m_4card.sh 2>&1 | tee scripts/_smoke_t2m_4card.log
    17	# Usage (real, DURABLE -- orchestrator ON a compute node so PPID=1):
    18	#   ssh swarmh1002 "cd /scratch/ts1v23/workspace/noKslot_clean && setsid nohup bash scripts/_launch_diffusion_t2m_4card.sh > scripts/_train_latdyn_mu_4card.log 2>&1 < /dev/null &"
    19	set -uo pipefail
    20	P=/scratch/ts1v23/workspace/noKslot_clean
    21	cd "$P" || exit 1
    22	
    23	JOB_A="${JOB_A:-944460}"                 # swarmh1002, NODE_RANK 0 (master, starts TCPStore)
    24	JOB_B="${JOB_B:-944461}"                 # swarmh1002, NODE_RANK 1
    25	RDZV_HOST="${RDZV_HOST:-swarmh1002-ib0}"
    26	RDZV_PORT="${RDZV_PORT:-29541}"          # distinct from 29501(6card)/29511(cap)/29531(B)
    27	SMOKE="${SMOKE:-0}"
    28	# 4-card global = PER_GPU_BATCH*4. Goyal off the pz20 low-LR line (global64->6.667e-5):
    29	# global40 (4xbs10) -> lr 4.17e-5. bs10 smoke-tested no-OOM @64.8GB on H100 (6-card mem).
    30	PER_GPU_BATCH="${PER_GPU_BATCH:-10}"
    31	LR="${LR:-4.17e-5}"
    32	LR_SCHEDULE="${LR_SCHEDULE:-cosine}"
    33	LR_MIN="${LR_MIN:-0.0}"
    34	WARMUP_ITERS="${WARMUP_ITERS:-400}"
    35	EPOCHS="${EPOCHS:-1500}"
    36	AMP_DTYPE="${AMP_DTYPE:-bf16}"
    37	TEXT_MODE="${TEXT_MODE:-mean_additive}"
    38	# 20-species capacity probe set (== B run); train on all clips, eval on val.
    39	SPECIES_WHITELIST="${SPECIES_WHITELIST:-PZ_Koala_Female,PZ_Jaguar_Female,PZ_Siberian_Tiger_Juvenile,PZ_Ocelot_Female,PZ_Amur_Leopard_Juvenile,PZ_Cougar_Male,PZ_Bush_Dog_Male,PZ_Raccoon_Juvenile,PZ_Sun_Bear_Female,PZ_Formosan_Black_Bear_Male,PZ_Red_Panda_Male,PZ_Proboscis_Monkey_Juvenile,PZ_Hamadryas_Baboon_Male,PZ_Western_Chimpanzee_Male,PZ_Bonobo_Juvenile,PZ_Siamang_Male,PZ_Japanese_Macaque_Juvenile,PZ_King_Penguin_Male,PZ_Little_Penguin_Male,PZ_Hippopotamus_Male}"
    40	TRAIN_SPLIT="${TRAIN_SPLIT:-all}"
    41	# latent temporal dynamics loss -- B-mu: SAME weights as B, target=mu (vs B's sample).
    42	W_LAT_DZ="${W_LAT_DZ:-0.05}"
    43	W_LAT_DDZ="${W_LAT_DDZ:-0.02}"
    44	W_LAT_X0="${W_LAT_X0:-0}"
    45	LATENT_DYN_TARGET="${LATENT_DYN_TARGET:-mu}"
    46	SPATIAL_MODE="${SPATIAL_MODE:-graph}"    # graph (default) | plain (no_graph_spatial ablation)
    47	CAPTION_TOKEN_CACHE="${CAPTION_TOKEN_CACHE:-}"        # required when TEXT_MODE=token_cross_attn/dual_text
    48	CAPTION_TOKEN_MAX_LEN="${CAPTION_TOKEN_MAX_LEN:-64}"
    49	VAE_CKPT="${VAE_CKPT:-runs/m1_bf16_anytop13_rot6dfk_w025f100t010_C128_8card_xnode_seed42/best_recon_model.pt}"
    50	OUT="${OUT:-runs/m2_capacity_pz20_latdyn_dz005_ddz002_MUtarget_h100x4_lr4.17e-5cos_seed42}"
    51	RESUME_CKPT="${RESUME_CKPT:-}"           # full crash-resume (model+opt+epoch+best_val+global_it); inner launcher passes --resume. cosine resume re-passes same lr_schedule/epochs (above).
    52	
    53	# Single-instance lock (per-launch pgrep guard disabled for NNODES>1).
    54	mkdir -p .aris/meta
    55	exec 9>".aris/meta/.t2m4card.lock"
    56	flock -n 9 || { echo "[t2m-4card] ABORT: already running"; exit 0; }
    57	
    58	# NNODES=2 triggers the static-rendezvous branch in _launch_diffusion_t2m.sh;
    59	# CVD=0,1 = each alloc's 2 local H100s. Same-node cross-cgroup -> inner launcher
    60	# disables P2P/SHM + forces IB (proven on the 6-card same-node run).
    61	COMMON_ENV="NNODES=2 NPROC_PER_NODE=2 MASTER_ADDR=$RDZV_HOST MASTER_PORT=$RDZV_PORT CVD=0,1 PER_GPU_BATCH=$PER_GPU_BATCH LR=$LR LR_SCHEDULE=$LR_SCHEDULE LR_MIN=$LR_MIN WARMUP_ITERS=$WARMUP_ITERS EPOCHS=$EPOCHS AMP_DTYPE=$AMP_DTYPE TEXT_MODE=$TEXT_MODE SPECIES_WHITELIST=$SPECIES_WHITELIST TRAIN_SPLIT=$TRAIN_SPLIT W_LAT_DZ=$W_LAT_DZ W_LAT_DDZ=$W_LAT_DDZ W_LAT_X0=$W_LAT_X0 LATENT_DYN_TARGET=$LATENT_DYN_TARGET SPATIAL_MODE=$SPATIAL_MODE CAPTION_TOKEN_CACHE=$CAPTION_TOKEN_CACHE CAPTION_TOKEN_MAX_LEN=$CAPTION_TOKEN_MAX_LEN VAE_CKPT=$VAE_CKPT OUT=$OUT SMOKE=$SMOKE RESUME_CKPT=$RESUME_CKPT"
    62	
    63	echo "[t2m-4card] $(date '+%F %T %Z') cross-alloc 4-card H100 DDP: $JOB_A(r0)+$JOB_B(r1) via $RDZV_HOST:$RDZV_PORT smoke=$SMOKE"
    64	echo "[t2m-4card] global=$(( PER_GPU_BATCH*4 )) (4xbs$PER_GPU_BATCH) lr=$LR sched=$LR_SCHEDULE/lr_min=$LR_MIN warmup=$WARMUP_ITERS epochs=$EPOCHS amp=$AMP_DTYPE"
    65	echo "[t2m-4card] latdyn: w_lat_dz=$W_LAT_DZ w_lat_ddz=$W_LAT_DDZ w_lat_x0=$W_LAT_X0 TARGET=$LATENT_DYN_TARGET (B-mu diagnostic)"
    66	echo "[t2m-4card] VAE=$VAE_CKPT text=$TEXT_MODE split=$TRAIN_SPLIT out=$OUT"
    67	
    68	# One torchrun group per alloc; static rendezvous joins them into 4 global ranks.
    69	run_alloc() {
    70	    local tag="$1" job="$2" noderank="$3"
    71	    srun --jobid="$job" --overlap --nodes=1 --ntasks=1 \
    72	      --gres=gpu:2 --cpus-per-task=16 --no-kill \
    73	      bash -c "cd '$P' && NODE_RANK=$noderank $COMMON_ENV bash scripts/_launch_diffusion_t2m.sh" \
    74	      2>&1 | stdbuf -oL sed "s/^/[$tag] /"
    75	}
    76	# allocA = node_rank 0 (master, starts the TCPStore); allocB = 1.
    77	run_alloc allocA "$JOB_A" 0 & PID_A=$!
    78	run_alloc allocB "$JOB_B" 1 & PID_B=$!
    79	
    80	wait "$PID_A"; RC_A=$?
    81	wait "$PID_B"; RC_B=$?
    82	echo "[t2m-4card] $(date '+%F %T %Z') EXITED rc_A=$RC_A rc_B=$RC_B"
    83	if [ "$RC_A" -ne 0 ] || [ "$RC_B" -ne 0 ]; then exit 1; fi
    84	exit 0

exec
/usr/bin/bash -lc "nl -ba scripts/_launch_diffusion_t2m.sh | sed -n '35,132p'" in /iridisfs/scratch/ts1v23/workspace/noKslot_clean
 succeeded in 0ms:
    35	MASTER_ADDR="${MASTER_ADDR:-}"
    36	MASTER_PORT="${MASTER_PORT:-29501}"
    37	SMOKE="${SMOKE:-0}"
    38	EPOCHS="${EPOCHS:-500}"
    39	WARMUP_ITERS="${WARMUP_ITERS:-4000}"   # review-recommended (global 96)
    40	N_LAYERS="${N_LAYERS:-11}"             # user 2026-06-03: 5enc+1mid+5dec ~63.5M; n17/96.6M was 97% full @bs8 → 11 for bz headroom; must be odd
    41	D_FF="${D_FF:-1536}"                   # user 2026-06-02: < default 4*d_model=2048, cuts FFN act/params (~96.6M)
    42	INIT_CKPT="${INIT_CKPT:-}"
    43	RESUME_CKPT="${RESUME_CKPT:-}"
    44	CVD="${CVD:-0,1}"
    45	AMP_DTYPE="${AMP_DTYPE:-fp32}"          # bf16 now bf16-safe (fp32-forced softmax); default fp32
    46	LR_SCHEDULE="${LR_SCHEDULE:-constant}"  # constant (default, unchanged) | cosine (warmup→cosine→lr_min)
    47	LR_MIN="${LR_MIN:-0.0}"                 # cosine floor (only used when LR_SCHEDULE=cosine)
    48	SPECIES_WHITELIST="${SPECIES_WHITELIST:-}"  # comma-sep object_types (capacity probe); empty = full 473
    49	TRAIN_SPLIT="${TRAIN_SPLIT:-train}"     # train (default) | all (train on all whitelisted clips incl val)
    50	# M2 latent temporal dynamics loss (handoff 20260605); ALL 0 = byte-equivalent.
    51	W_LAT_DZ="${W_LAT_DZ:-0}"               # weight on latent velocity loss ||Δz0_hat-Δz0||²
    52	W_LAT_DDZ="${W_LAT_DDZ:-0}"            # weight on latent acceleration loss ||Δ²z0_hat-Δ²z0||²
    53	W_LAT_X0="${W_LAT_X0:-0}"              # weight on direct latent loss ||z0_hat-z0||² (keep 0 first run)
    54	LATENT_DYN_TARGET="${LATENT_DYN_TARGET:-sample}"  # sample (default) | mu
    55	SPATIAL_MODE="${SPATIAL_MODE:-graph}"             # graph (default) | plain (no_graph_spatial ablation)
    56	# M2 token-level text conditioning (default mean_additive = current behavior).
    57	TEXT_MODE="${TEXT_MODE:-mean_additive}"
    58	CAPTION_TOKEN_CACHE="${CAPTION_TOKEN_CACHE:-}"   # required when TEXT_MODE=token_cross_attn
    59	CAPTION_TOKEN_MAX_LEN="${CAPTION_TOKEN_MAX_LEN:-64}"
    60	
    61	# VAE = B's rot6d_fk ep79 best (frozen). Review-confirmed: load_frozen_vae() LOAD_OK
    62	# (full Phase-2 loading path, strict rebuild+load, not just key count).
    63	VAE_CKPT="${VAE_CKPT:-runs/m1_l2_anytop13_edgeseg_C128_rot6dfk_w025f100t010_g128_4card_seed42/best_model.pt}"
    64	CAPCACHE="data/anytop_caption_t5_cleanL2_multi.npz"
    65	ANYTOP_ROOT="$P/data/anytop_planet_zoo_clean_L2"
    66	
    67	# H2/C4: GLOBAL = PER_GPU × NNODES × NPROC_PER_NODE (NOT PER_GPU × WORLD_SIZE).
    68	REF_GLOBAL=48
    69	GLOBAL=$(( PER_GPU_BATCH * NNODES * NPROC_PER_NODE ))
    70	LR="${LR:-$(awk "BEGIN{printf \"%.3e\", 5e-4 * $GLOBAL / $REF_GLOBAL}")}"
    71	
    72	OUT="${OUT:-runs/m2_t2m_cleanL2_Bep79rot6dfk_d512C128_n11ff1536_h100x6_seed42}"
    73	
    74	# H1/C3: smoke adds ONLY --smoke, keeps full nproc/nnodes (true 6-rank validation).
    75	SMOKE_FLAG=""
    76	if [ "$SMOKE" = 1 ]; then
    77	    SMOKE_FLAG="--smoke"
    78	    OUT="${OUT}_smoke"
    79	fi
    80	
    81	# Guard: never double-launch the real run (single-alloc only; the cross-alloc
    82	# 6-card run is managed by its orchestrator, and same-node pgrep would otherwise
    83	# false-match a peer alloc's rank → self-abort).
    84	if [ "$SMOKE" != 1 ] && [ "$NNODES" -le 1 ] && pgrep -f "train_denoiser.py.*Bep79rot6dfk" >/dev/null 2>&1; then
    85	    echo "[t2m] ABORT: this diffusion run already training"; exit 0
    86	fi
    87	mkdir -p "$OUT"
    88	
    89	export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
    90	export CUDA_VISIBLE_DEVICES="$CVD"
    91	
    92	# torchrun mode: standalone (single alloc) vs static rendezvous (cross-alloc).
    93	# Static (not c10d) because same-node agents' hostname != IB rdzv host → c10d
    94	# auto-host election fails (cross-alloc memory; verified on the 4-card rot6d_fk run).
    95	if [ "$NNODES" -gt 1 ]; then
    96	    [ -z "$MASTER_ADDR" ] && { echo "[t2m] FAIL: NNODES>1 requires MASTER_ADDR"; exit 2; }
    97	    # same-node cross-cgroup: Slurm isolates P2P/SHM between allocs → disable + force IB.
    98	    export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
    99	    export NCCL_SHM_DISABLE="${NCCL_SHM_DISABLE:-1}"
   100	    export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-ib0}"
   101	    export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-0}"
   102	    export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
   103	    export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"
   104	    export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
   105	    RDZV_ARGS="--nnodes=$NNODES --node_rank=$NODE_RANK --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT --nproc_per_node=$NPROC_PER_NODE"
   106	else
   107	    RDZV_ARGS="--standalone --nnodes=1 --nproc_per_node=$NPROC_PER_NODE"
   108	fi
   109	
   110	echo "[t2m] $(date '+%F %T %Z') host=$(hostname) CVD=$CVD nnodes=$NNODES nproc_per_node=$NPROC_PER_NODE node_rank=$NODE_RANK"
   111	echo "[t2m] VAE=$VAE_CKPT (frozen)"
   112	echo "[t2m] cap_cache=$CAPCACHE anytop_root=$ANYTOP_ROOT"
   113	echo "[t2m] per_gpu=$PER_GPU_BATCH global=$GLOBAL(=${PER_GPU_BATCH}x${NNODES}x${NPROC_PER_NODE}) lr=$LR sched=$LR_SCHEDULE lr_min=$LR_MIN | smoke=$SMOKE epochs=$EPOCHS warmup=$WARMUP_ITERS"
   114	echo "[t2m] master=${MASTER_ADDR:-<standalone>}:$MASTER_PORT nccl_ifname=${NCCL_SOCKET_IFNAME:-<n/a>} out=$OUT"
   115	echo "[t2m] text_mode=$TEXT_MODE amp=$AMP_DTYPE token_cache=${CAPTION_TOKEN_CACHE:-<none>} L=$CAPTION_TOKEN_MAX_LEN"
   116	echo "[t2m] latdyn: w_lat_dz=$W_LAT_DZ w_lat_ddz=$W_LAT_DDZ w_lat_x0=$W_LAT_X0 target=$LATENT_DYN_TARGET | spatial_mode=$SPATIAL_MODE"
   117	
   118	torchrun $RDZV_ARGS scripts/train_denoiser.py \
   119	  --vae_ckpt "$VAE_CKPT" \
   120	  --caption_emb_cache "$CAPCACHE" \
   121	  --anytop_root "$ANYTOP_ROOT" \
   122	  --max_frames 260 --max_joints 144 \
   123	  --batch_size "$PER_GPU_BATCH" --lr "$LR" --epochs "$EPOCHS" \
   124	  --warmup_iters "$WARMUP_ITERS" --lr_schedule "$LR_SCHEDULE" --lr_min "$LR_MIN" \
   125	  --train_split "$TRAIN_SPLIT" \
   126	  --w_lat_dz "$W_LAT_DZ" --w_lat_ddz "$W_LAT_DDZ" --w_lat_x0 "$W_LAT_X0" \
   127	  --latent_dyn_target "$LATENT_DYN_TARGET" \
   128	  --spatial_mode "$SPATIAL_MODE" \
   129	  ${SPECIES_WHITELIST:+--species_whitelist "$SPECIES_WHITELIST"} \
   130	  ${INIT_CKPT:+--init_ckpt "$INIT_CKPT"} \
   131	  ${RESUME_CKPT:+--resume "$RESUME_CKPT"} \
   132	  --n_layers "$N_LAYERS" --d_ff "$D_FF" --dropout 0.1 \

exec
/usr/bin/bash -lc "nl -ba scripts/_launch_token_diffusion_8card_a100.sh | sed -n '20,120p'" in /iridisfs/scratch/ts1v23/workspace/noKslot_clean
 succeeded in 0ms:
    20	P=/scratch/ts1v23/workspace/noKslot_clean
    21	cd "$P" || exit 1
    22	
    23	JOB_A="${JOB_A:-944455}"                 # swarma1004, NODE_RANK 0 (master)
    24	JOB_B="${JOB_B:-944456}"                 # swarma1001, NODE_RANK 1
    25	RDZV_HOST="${RDZV_HOST:-swarma1004-ib0}" # master = alloc A's IB host (verify reachable)
    26	RDZV_PORT="${RDZV_PORT:-29511}"          # distinct from the H100 6-card run (29501)
    27	SMOKE="${SMOKE:-0}"
    28	# token cross-attn adds activation (scores [B,heads,T_lat*C,L]); start conservative,
    29	# smoke-tune up. Goyal: global = PER_GPU_BATCH * 8, lr = 5e-4 * global / 48.
    30	PER_GPU_BATCH="${PER_GPU_BATCH:-8}"
    31	LR="${LR:-$(awk "BEGIN{printf \"%.3e\", 5e-4 * ($PER_GPU_BATCH*8) / 48}")}"
    32	LR_SCHEDULE="${LR_SCHEDULE:-constant}" # constant (default) | cosine (warmup→cosine→lr_min)
    33	LR_MIN="${LR_MIN:-0.0}"                # cosine floor (only used when LR_SCHEDULE=cosine)
    34	SPECIES_WHITELIST="${SPECIES_WHITELIST:-}"  # comma-sep object_types (capacity probe); empty=full 473
    35	TRAIN_SPLIT="${TRAIN_SPLIT:-train}"    # train (default) | all (train on all whitelisted clips incl val)
    36	# M2 latent temporal dynamics loss (handoff 20260605); ALL 0 = byte-equivalent.
    37	W_LAT_DZ="${W_LAT_DZ:-0}"
    38	W_LAT_DDZ="${W_LAT_DDZ:-0}"
    39	W_LAT_X0="${W_LAT_X0:-0}"
    40	LATENT_DYN_TARGET="${LATENT_DYN_TARGET:-sample}"
    41	SPATIAL_MODE="${SPATIAL_MODE:-graph}"  # graph (default) | plain (no_graph_spatial ablation)
    42	WARMUP_ITERS="${WARMUP_ITERS:-4000}"
    43	EPOCHS="${EPOCHS:-500}"
    44	OUT="${OUT:-runs/m2_token_cleanL2_bf16ep209_d512C128_n11ff1536_a100x8_seed42}"
    45	RESUME_CKPT="${RESUME_CKPT:-}"           # full crash/walltime resume (model+opt+epoch+global_it); inner passes --resume. cosine resume re-passes same lr_schedule/epochs.
    46	AMP_DTYPE="${AMP_DTYPE:-bf16}"
    47	TEXT_MODE="${TEXT_MODE:-token_cross_attn}"
    48	CAPTION_TOKEN_CACHE="${CAPTION_TOKEN_CACHE:-data/anytop_caption_t5_cleanL2_multi}"
    49	CAPTION_TOKEN_MAX_LEN="${CAPTION_TOKEN_MAX_LEN:-64}"
    50	# Frozen VAE = bf16 ep209 best (val_recon 1.3983), archived to main.
    51	VAE_CKPT="${VAE_CKPT:-runs/m1_bf16_anytop13_rot6dfk_w025f100t010_C128_8card_xnode_seed42/best_recon_model.pt}"
    52	
    53	# Single-instance lock (cross-alloc: per-launch pgrep guard is disabled for NNODES>1).
    54	mkdir -p .aris/meta
    55	exec 9>".aris/meta/.token8card.lock"
    56	flock -n 9 || { echo "[token-8card] ABORT: already running"; exit 0; }
    57	
    58	# Shared env every alloc's launch inherits. NNODES=2 → static-rendezvous branch in
    59	# _launch_diffusion_t2m.sh; CVD=0,1,2,3 = each alloc's 4 local A100s.
    60	# NCCL (codex 019e94d2 P1): this is TRUE cross-NODE (not same-node cross-cgroup).
    61	# Each node's 4 A100-SXM4 are NV4 NVLink → ENABLE intra-node P2P/SHM, overriding
    62	# the _launch_diffusion_t2m.sh NNODES>1 defaults (P2P/SHM=disabled, which were for
    63	# same-node cross-alloc cgroup isolation and would route intra-node collectives
    64	# through slow host/NET). Matches the proven xnode VAE launcher. IB_HCA=mlx5_0
    65	# (ibdev2netdev: mlx5_0->ib0 Up on both nodes).
    66	COMMON_ENV="NNODES=2 NPROC_PER_NODE=4 MASTER_ADDR=$RDZV_HOST MASTER_PORT=$RDZV_PORT CVD=0,1,2,3 NCCL_P2P_DISABLE=0 NCCL_SHM_DISABLE=0 NCCL_IB_HCA=mlx5_0 PER_GPU_BATCH=$PER_GPU_BATCH LR=$LR LR_SCHEDULE=$LR_SCHEDULE LR_MIN=$LR_MIN WARMUP_ITERS=$WARMUP_ITERS EPOCHS=$EPOCHS SPECIES_WHITELIST=$SPECIES_WHITELIST TRAIN_SPLIT=$TRAIN_SPLIT W_LAT_DZ=$W_LAT_DZ W_LAT_DDZ=$W_LAT_DDZ W_LAT_X0=$W_LAT_X0 LATENT_DYN_TARGET=$LATENT_DYN_TARGET SPATIAL_MODE=$SPATIAL_MODE OUT=$OUT RESUME_CKPT=$RESUME_CKPT SMOKE=$SMOKE AMP_DTYPE=$AMP_DTYPE TEXT_MODE=$TEXT_MODE CAPTION_TOKEN_CACHE=$CAPTION_TOKEN_CACHE CAPTION_TOKEN_MAX_LEN=$CAPTION_TOKEN_MAX_LEN VAE_CKPT=$VAE_CKPT"
    67	
    68	echo "[token-8card] $(date '+%F %T %Z') cross-node 8-card A100 DDP: $JOB_A(1004,r0)+$JOB_B(1001,r1) via $RDZV_HOST:$RDZV_PORT smoke=$SMOKE"
    69	echo "[token-8card] text_mode=$TEXT_MODE amp=$AMP_DTYPE vae=$VAE_CKPT token_cache=$CAPTION_TOKEN_CACHE L=$CAPTION_TOKEN_MAX_LEN"
    70	echo "[token-8card] global=$(( PER_GPU_BATCH*8 )) (8xbs$PER_GPU_BATCH) lr=$LR sched=$LR_SCHEDULE/lr_min=$LR_MIN warmup=$WARMUP_ITERS epochs=$EPOCHS out=$OUT"
    71	
    72	# One torchrun group per alloc; static rendezvous joins them into 8 global ranks.
    73	# Explicit --gres/--cpus so each srun step gets its alloc's 4 GPUs + CPU for 4 ranks
    74	# x dataloaders; --no-kill so one rank's transient failure does not tear down the step.
    75	run_alloc() {
    76	    local tag="$1" job="$2" noderank="$3"
    77	    srun --jobid="$job" --overlap --nodes=1 --ntasks=1 \
    78	      --gres=gpu:4 --cpus-per-task=32 --no-kill \
    79	      bash -c "cd '$P' && NODE_RANK=$noderank $COMMON_ENV bash scripts/_launch_diffusion_t2m.sh" \
    80	      2>&1 | stdbuf -oL sed "s/^/[$tag] /"
    81	}
    82	# allocA = NODE_RANK 0 (master, starts the TCPStore); allocB = 1.
    83	run_alloc allocA "$JOB_A" 0 & PID_A=$!
    84	run_alloc allocB "$JOB_B" 1 & PID_B=$!
    85	
    86	wait "$PID_A"; RC_A=$?
    87	wait "$PID_B"; RC_B=$?
    88	echo "[token-8card] $(date '+%F %T %Z') EXITED rc_A=$RC_A rc_B=$RC_B"
    89	if [ "$RC_A" -ne 0 ] || [ "$RC_B" -ne 0 ]; then exit 1; fi
    90	exit 0

exec
/usr/bin/bash -lc "nl -ba scripts/animate_denoiser.py | sed -n '60,82p;95,155p'" in /iridisfs/scratch/ts1v23/workspace/noKslot_clean
 succeeded in 0ms:
    60	    da = ck.get("args", {})
    61	    vae_ta = ck.get("vae_ckpt_args", {})
    62	    d_model = vae_ta.get("d_model", 384)
    63	    n_heads = vae_ta.get("n_heads", 8)
    64	    d_ff = da.get("d_ff") or 4 * d_model
    65	    # M2: rebuild with the ckpt's text_mode (mean ckpts → 'mean_additive' default;
    66	    # token ckpts carry 'token_cross_attn' in args). Wrong mode ⇒ arch mismatch ⇒
    67	    # strict-load fails loud below.
    68	    text_mode = da.get("text_mode", "mean_additive")
    69	    spatial_mode = da.get("spatial_mode", "graph")  # old ckpts (no key) → graph
    70	    denoiser = GraphSaladDenoiser(
    71	        d_model=d_model, n_heads=n_heads, d_ff=d_ff,
    72	        n_layers=da.get("n_layers", 5),
    73	        d_text=768, dropout=da.get("dropout", 0.1),
    74	        text_mode=text_mode, text_token_dim=768,
    75	        spatial_mode=spatial_mode,
    76	    ).to(dev)
    77	    missing, unexpected = denoiser.load_state_dict(ck["model_state_dict"], strict=True)
    78	    if missing or unexpected:
    79	        raise SystemExit(
    80	            f"Denoiser ckpt strict-load failed: missing={len(missing)} unexpected={len(unexpected)}"
    81	        )
    82	    denoiser.eval()
    95	    dev: torch.device,
    96	    d_model: int,
    97	) -> torch.Tensor:
    98	    """Run DDIM sampling with classifier-free guidance.
    99	
   100	    Returns z_0 [B, T_lat, C, D].
   101	    """
   102	    B = skel["pooled_adjacency"].shape[0]
   103	    C = skel["pooled_adjacency"].shape[1]
   104	    T_lat = frame_mask_lat.shape[1]
   105	
   106	    sched = DDIMScheduler(**sched_kwargs)
   107	    sched.set_timesteps(n_steps, device=dev)
   108	    # Initialize z_T ~ N(0, I); mask padded positions
   109	    z = torch.randn(B, T_lat, C, d_model, device=dev)
   110	    mask_4d = (skel["coarse_mask"][:, None, :, None] & frame_mask_lat[:, :, None, None]).to(z.dtype)
   111	    z = z * mask_4d
   112	
   113	    # Repeat conditioning to 2B for CFG cond+uncond batching
   114	    adj2 = skel["pooled_adjacency"].repeat(2, 1, 1)
   115	    geo2 = skel["pooled_geodesic"].repeat(2, 1, 1)
   116	    cm2 = skel["coarse_mask"].repeat(2, 1)
   117	    fm2 = frame_mask_lat.repeat(2, 1)
   118	    skel2 = skel["pooled_skeleton_embeddings"].repeat(2, 1, 1)
   119	    has_text_cond = batch.has_text.to(dev)              # [B] bool
   120	    has_text_uncond = torch.zeros_like(has_text_cond, dtype=torch.bool)
   121	    has_text2 = torch.cat([has_text_cond, has_text_uncond], dim=0)  # [2B]
   122	    # M2: mode-dependent text, repeated 2x for the CFG cond+uncond batch. The uncond
   123	    # half's has_text=False zeroes the global add AND fully masks the token keys
   124	    # (cross-attn → 0), so both streams CFG-drop together (dual_text).
   125	    text_mode = getattr(denoiser, "text_mode", "mean_additive")
   126	    text_tokens2 = None
   127	    if text_mode == "dual_text":
   128	        text2 = batch.caption_emb.to(dev).repeat(2, 1)                   # [2B, 768] global
   129	        text_tokens2 = batch.caption_token_emb.to(dev).repeat(2, 1, 1)  # [2B, L, 768] tokens
   130	        token_mask2 = batch.caption_token_mask.to(dev).repeat(2, 1)     # [2B, L]
   131	    elif text_mode == "token_cross_attn":
   132	        text2 = batch.caption_token_emb.to(dev).repeat(2, 1, 1)         # [2B, L, 768]
   133	        token_mask2 = batch.caption_token_mask.to(dev).repeat(2, 1)     # [2B, L]
   134	    else:  # mean_additive
   135	        text2 = batch.caption_emb.to(dev).repeat(2, 1)                  # [2B, 768]
   136	        token_mask2 = None
   137	
   138	    first = True
   139	    for t in sched.timesteps:
   140	        # Build cond+uncond batch
   141	        z2 = torch.cat([z, z], dim=0)                    # [2B, T_lat, C, D]
   142	        t2 = torch.full((2 * B,), int(t.item()), device=dev, dtype=torch.long)
   143	        v2 = denoiser(
   144	            z_t=z2, timesteps=t2, text=text2,
   145	            adjacency=adj2, geodesic_dist=geo2,
   146	            coarse_mask=cm2, frame_mask=fm2,
   147	            pooled_skeleton_embeddings=skel2,
   148	            has_text=has_text2,
   149	            text_token_mask=token_mask2,
   150	            text_tokens=text_tokens2,
   151	            validate_inputs=first,  # cold-start validate on first iter
   152	        )
   153	        first = False
   154	        v_cond, v_uncond = v2.chunk(2, dim=0)
   155	        v = v_uncond + cond_scale * (v_cond - v_uncond)

codex
The launcher propagation path is the riskiest area now: both orchestrators interpolate `COMMON_ENV` through `bash -c`, so I’m checking whether values containing commas or paths are safe and whether unset defaults still flow as graph.
exec
/usr/bin/bash -lc "nl -ba scripts/train_denoiser.py | sed -n '1050,1118p;1118,1165p'" in /iridisfs/scratch/ts1v23/workspace/noKslot_clean
 succeeded in 0ms:
  1050	
  1051	                # Best ckpt — rank 0 only; unwrap DDP for clean state_dict
  1052	                if val_loss < best_val:
  1053	                    best_val = val_loss
  1054	                    best_path = out_dir / "best_model.pt"
  1055	                    torch.save({
  1056	                        "epoch": epoch, "val_denoise": val_loss, "train_loss": epoch_loss,
  1057	                        "model_state_dict": raw_denoiser.state_dict(),
  1058	                        "optimizer_state_dict": opt.state_dict(),
  1059	                        "args": vars(args),
  1060	                        "vae_ckpt_args": ta,
  1061	                    }, best_path)
  1062	                    log(f"  saved best ckpt → {best_path} (val_denoise={best_val:.4f})")
  1063	            # END val + best block (rank 0 only)
  1064	
  1065	        # Periodic last save — rank 0 only
  1066	        if (epoch % args.save_every == 0) or (epoch == epochs - 1) or args.smoke:
  1067	            if is_main:
  1068	                last_path = out_dir / "last_model.pt"
  1069	                torch.save({
  1070	                    "epoch": epoch, "val_denoise": best_val, "train_loss": epoch_loss,
  1071	                    "model_state_dict": raw_denoiser.state_dict(),
  1072	                    "optimizer_state_dict": opt.state_dict(),
  1073	                    "args": vars(args), "vae_ckpt_args": ta,
  1074	                }, last_path)
  1075	
  1076	        # Periodic PRESERVED save (every periodic_save_every epochs, ep{N}_model.pt)
  1077	        # Uses (epoch+1) so first save is at epoch=periodic_save_every-1 (i.e.
  1078	        # after 500 epochs done, save ep0500_model.pt). Rank 0 only.
  1079	        if args.periodic_save_every > 0 and ((epoch + 1) % args.periodic_save_every) == 0:
  1080	            if is_main:
  1081	                periodic_path = out_dir / f"ep{epoch + 1:04d}_model.pt"
  1082	                torch.save({
  1083	                    "epoch": epoch, "val_denoise": best_val, "train_loss": epoch_loss,
  1084	                    "model_state_dict": raw_denoiser.state_dict(),
  1085	                    "optimizer_state_dict": opt.state_dict(),
  1086	                    "args": vars(args), "vae_ckpt_args": ta,
  1087	                }, periodic_path)
  1088	                log(f"  saved periodic ckpt → {periodic_path}")
  1089	
  1090	        # Barrier: re-sync all ranks after rank-0 val/save IO. Must be outside
  1091	        # any is_main block (otherwise rank!=0 never reaches → deadlock).
  1092	        if is_ddp:
  1093	            dist.barrier()
  1094	
  1095	        if args.smoke:
  1096	            log(f"\n=== SMOKE MODE: 1 epoch done, exit ===")
  1097	            break
  1098	
  1099	    log("\n=== training complete ===")
  1100	    if is_main:
  1101	        metrics_fp.close(); log_fp.close()
  1102	    if is_ddp:
  1103	        dist.destroy_process_group()
  1104	    return 0
  1105	
  1106	
  1107	if __name__ == "__main__":
  1108	    sys.exit(main())

exec
/usr/bin/bash -lc "git show HEAD:src/models/graph_salad/attention.py | nl -ba | sed -n '75,125p;345,375p'" in /iridisfs/scratch/ts1v23/workspace/noKslot_clean
 succeeded in 0ms:
    75	    def __init__(
    76	        self,
    77	        d_model: int,
    78	        n_heads: int,
    79	        d_ff: int,
    80	        dropout: float = 0.1,
    81	    ) -> None:
    82	        super().__init__()
    83	        if d_model <= 0 or n_heads <= 0:
    84	            raise ValueError(f"d_model and n_heads must be > 0, got {d_model}, {n_heads}")
    85	        if d_model % n_heads != 0:
    86	            raise ValueError(
    87	                f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
    88	            )
    89	        if d_ff <= 0:
    90	            raise ValueError(f"d_ff must be > 0, got {d_ff}")
    91	        # Dropout contract is [0, 1) per nn.Dropout docs; p=1 zeros all outputs
    92	        # (everything dropped) which corrupts gradient flow silently. Codex M1.2
    93	        # round 1 R12 fix.
    94	        if not (0.0 <= dropout < 1.0):
    95	            raise ValueError(f"dropout must be in [0, 1), got {dropout}")
    96	
    97	        self.d_model = d_model
    98	        self.n_heads = n_heads
    99	        self.d_head = d_model // n_heads
   100	
   101	        # Q/K/V/O projections
   102	        self.q_proj = nn.Linear(d_model, d_model)
   103	        self.k_proj = nn.Linear(d_model, d_model)
   104	        self.v_proj = nn.Linear(d_model, d_model)
   105	        self.o_proj = nn.Linear(d_model, d_model)
   106	
   107	        # Edge bias projections (scalar → per-head)
   108	        # Matches encoder.py:41-42 formulation.
   109	        self.geodesic_bias = nn.Linear(1, n_heads, bias=False)
   110	        self.adjacency_bias = nn.Linear(1, n_heads, bias=False)
   111	
   112	        # Norms (pre-norm)
   113	        self.norm1 = nn.LayerNorm(d_model)
   114	        self.norm2 = nn.LayerNorm(d_model)
   115	
   116	        # Feedforward block
   117	        self.ff = nn.Sequential(
   118	            nn.Linear(d_model, d_ff),
   119	            nn.GELU(),
   120	            nn.Dropout(dropout),
   121	            nn.Linear(d_ff, d_model),
   122	            nn.Dropout(dropout),
   123	        )
   124	        self.dropout = nn.Dropout(dropout)
   125	
   345	
   346	        # Topology biases. geodesic_dist may contain +inf for legitimate
   347	        # unreachable pairs (from floyd_shortest_path). Substitute +inf with
   348	        # 0.0 BEFORE projecting — this gives a neutral additive bias on those
   349	        # pairs. The key-mask masks out padded keys, so the neutral bias only
   350	        # affects unmasked-but-disconnected pairs (rare; deferred to a later
   351	        # learnable "unreachable" bucket per lit survey if it shows up in
   352	        # generation eval). NaN/-Inf were rejected above.
   353	        geo = geodesic_dist.clone()
   354	        geo[torch.isinf(geo)] = 0.0
   355	        geo_bias = self.geodesic_bias(geo.unsqueeze(-1))         # [B, N, N, H]
   356	        adj_bias = self.adjacency_bias(adjacency.unsqueeze(-1))  # [B, N, N, H]
   357	        topo_bias = (geo_bias + adj_bias).permute(0, 3, 1, 2)    # [B, H, N, N]
   358	        scores = scores + topo_bias
   359	
   360	        # Mask invalid nodes (key side). Use large finite negative for softmax
   361	        # numerical safety; matches encoder.py:84-85.
   362	        mask = node_mask.unsqueeze(1).unsqueeze(2)  # [B, 1, 1, N]
   363	        scores = scores.masked_fill(~mask, -1e9)
   364	
   365	        # Softmax. The earlier `all-False node_mask` per-sample guard ensures
   366	        # at least one valid key per batch element, so no row of `scores` is
   367	        # entirely -1e9 → no NaN in softmax output → no nan_to_num needed.
   368	        # Padded-query rows still compute attention (over valid keys); their
   369	        # output is zeroed downstream by the caller's joint_mask multiplication.
   370	        # softmax in fp32 for bf16-safety (sentinel + reduction precision). On the
   371	        # fp32 path scores.float() is a no-op and .to(scores.dtype) returns fp32, so
   372	        # behavior is byte-for-byte unchanged; on the bf16 path softmax runs in fp32
   373	        # then casts the probabilities back to bf16 for the attn@v matmul.
   374	        attn = F.softmax(scores.float(), dim=-1).to(scores.dtype)
   375	        attn = self.dropout(attn)

exec
/usr/bin/bash -lc 'git diff --unified=0 -- src/models/graph_salad/attention.py src/models/graph_salad/denoiser.py scripts/train_denoiser.py scripts/animate_denoiser.py scripts/_launch_diffusion_t2m.sh scripts/_launch_diffusion_t2m_4card.sh scripts/_launch_token_diffusion_8card_a100.sh | rg -n "spatial_mode|use_graph_bias|geodesic_bias|adjacency_bias|CAPTION_TOKEN_CACHE|COMMON_ENV|--spatial_mode|model_state_dict|args"' in /iridisfs/scratch/ts1v23/workspace/noKslot_clean
 succeeded in 0ms:
23:+echo "[t2m] latdyn: w_lat_dz=$W_LAT_DZ w_lat_ddz=$W_LAT_DDZ w_lat_x0=$W_LAT_X0 target=$LATENT_DYN_TARGET | spatial_mode=$SPATIAL_MODE"
30:+  --spatial_mode "$SPATIAL_MODE" \
52:-COMMON_ENV="NNODES=2 NPROC_PER_NODE=4 MASTER_ADDR=$RDZV_HOST MASTER_PORT=$RDZV_PORT CVD=0,1,2,3 NCCL_P2P_DISABLE=0 NCCL_SHM_DISABLE=0 NCCL_IB_HCA=mlx5_0 PER_GPU_BATCH=$PER_GPU_BATCH LR=$LR OUT=$OUT SMOKE=$SMOKE AMP_DTYPE=$AMP_DTYPE TEXT_MODE=$TEXT_MODE CAPTION_TOKEN_CACHE=$CAPTION_TOKEN_CACHE CAPTION_TOKEN_MAX_LEN=$CAPTION_TOKEN_MAX_LEN VAE_CKPT=$VAE_CKPT"
53:+COMMON_ENV="NNODES=2 NPROC_PER_NODE=4 MASTER_ADDR=$RDZV_HOST MASTER_PORT=$RDZV_PORT CVD=0,1,2,3 NCCL_P2P_DISABLE=0 NCCL_SHM_DISABLE=0 NCCL_IB_HCA=mlx5_0 PER_GPU_BATCH=$PER_GPU_BATCH LR=$LR LR_SCHEDULE=$LR_SCHEDULE LR_MIN=$LR_MIN WARMUP_ITERS=$WARMUP_ITERS EPOCHS=$EPOCHS SPECIES_WHITELIST=$SPECIES_WHITELIST TRAIN_SPLIT=$TRAIN_SPLIT W_LAT_DZ=$W_LAT_DZ W_LAT_DDZ=$W_LAT_DDZ W_LAT_X0=$W_LAT_X0 LATENT_DYN_TARGET=$LATENT_DYN_TARGET SPATIAL_MODE=$SPATIAL_MODE OUT=$OUT RESUME_CKPT=$RESUME_CKPT SMOKE=$SMOKE AMP_DTYPE=$AMP_DTYPE TEXT_MODE=$TEXT_MODE CAPTION_TOKEN_CACHE=$CAPTION_TOKEN_CACHE CAPTION_TOKEN_MAX_LEN=$CAPTION_TOKEN_MAX_LEN VAE_CKPT=$VAE_CKPT"
66:+    spatial_mode = da.get("spatial_mode", "graph")  # old ckpts (no key) → graph
68:+        spatial_mode=spatial_mode,
138:+            if all(picked[s] >= args.n_per for s in want):
151:+                gt=(gt_world if args.with_gt else None),
212:@@ -212,0 +267,10 @@ def parse_args() -> argparse.Namespace:
223:@@ -218,0 +283,23 @@ def parse_args() -> argparse.Namespace:
227:+                         "hold args.lr. 'cosine': decay args.lr -> lr_min over "
247:@@ -233 +320,2 @@ def parse_args() -> argparse.Namespace:
251:@@ -238 +326,8 @@ def parse_args() -> argparse.Namespace:
256:+    ap.add_argument("--spatial_mode", choices=["graph", "plain"], default="graph",
262:-    use_tokens = (args.text_mode == "token_cross_attn")
263:+    use_tokens = args.text_mode in ("token_cross_attn", "dual_text")
266:+            f"--text_mode {args.text_mode} requires --caption_token_cache "
269:+        [s.strip() for s in args.species_whitelist.split(",") if s.strip()]
270:+        if args.species_whitelist else None
275:-            split="train", random_caption=True, random_crop=False, **ds_kwargs)
276:+            split=args.train_split, random_caption=True, random_crop=False, **ds_kwargs)
278:+        spatial_mode=args.spatial_mode,
280:-        f"text_mode={args.text_mode} params={n_params:,}")
281:+        f"text_mode={args.text_mode} spatial_mode={args.spatial_mode} params={n_params:,}")
283:+        ck_spatial_mode = resume_ck.get("args", {}).get("spatial_mode", "graph")
284:+        if ck_spatial_mode != args.spatial_mode:
286:+                f"[RESUME FAIL] ckpt spatial_mode={ck_spatial_mode!r} != CLI "
287:+                f"--spatial_mode {args.spatial_mode!r}. graph/plain arch differ "
291:+        ck_spatial_mode = ck.get("args", {}).get("spatial_mode", "graph")
292:+        if ck_spatial_mode != args.spatial_mode:
294:+                f"[INIT_CKPT FAIL] ckpt spatial_mode={ck_spatial_mode!r} != CLI "
295:+                f"--spatial_mode {args.spatial_mode!r}. graph/plain arch differ."
305:+    # (codex #3). The orchestrator re-passes all of these via COMMON_ENV, so a
307:+    total_iters = (1 if args.smoke else args.epochs) * len(dl_train)
310:+        # Linear warmup (unchanged): ramp 0 -> args.lr over warmup_iters.
313:+        if args.lr_schedule == "cosine":
317:+            denom = max(1, total_iters - 1 - args.warmup_iters)
318:+            progress = min(1.0, max(0.0, (it - args.warmup_iters) / denom))
319:+            return args.lr_min + 0.5 * (args.lr - args.lr_min) * (
322:+    log(f"LR schedule: {args.lr_schedule} (peak={args.lr:.3e} warmup={args.warmup_iters}"
323:+        + (f" → cosine → lr_min={args.lr_min:.3e} over total_iters={total_iters}"
324:+           if args.lr_schedule == "cosine" else " then constant") + ")")
327:+        lat_active = bool(args.w_lat_dz > 0 or args.w_lat_ddz > 0 or args.w_lat_x0 > 0)
340:+            if args.text_mode == "dual_text":
361:+                if args.w_lat_dz > 0 or args.w_lat_ddz > 0 or args.w_lat_x0 > 0:
364:+                                     if args.latent_dyn_target == "mu" else z0)
365:+                    if args.w_lat_x0 > 0:
367:+                        loss = loss + args.w_lat_x0 * loss_x0
368:+                    if args.w_lat_dz > 0:
370:+                        loss = loss + args.w_lat_dz * loss_dz
371:+                    if args.w_lat_ddz > 0:
373:+                        loss = loss + args.w_lat_ddz * loss_ddz
400:+                        if args.text_mode == "dual_text":
411:+                                             if args.latent_dyn_target == "mu" else z0)
416:+                            if args.w_lat_x0 > 0:
443:+                    if args.w_lat_x0 > 0:
452:+        use_graph_bias: bool = True,
454:+        self.use_graph_bias = use_graph_bias
458:-        self.geodesic_bias = nn.Linear(1, n_heads, bias=False)
459:-        self.adjacency_bias = nn.Linear(1, n_heads, bias=False)
462:+        # The no_graph_spatial ablation (use_graph_bias=False) is a plain slot
466:+        if use_graph_bias:
467:+            self.geodesic_bias = nn.Linear(1, n_heads, bias=False)
468:+            self.adjacency_bias = nn.Linear(1, n_heads, bias=False)
472:-        geo_bias = self.geodesic_bias(geo.unsqueeze(-1))         # [B, N, N, H]
473:-        adj_bias = self.adjacency_bias(adjacency.unsqueeze(-1))  # [B, N, N, H]
478:+        if self.use_graph_bias:
481:+            geo_bias = self.geodesic_bias(geo.unsqueeze(-1))         # [B, N, N, H]
482:+            adj_bias = self.adjacency_bias(adjacency.unsqueeze(-1))  # [B, N, N, H]
490:+        spatial_mode: str = "graph",
493:+        self.spatial_mode = spatial_mode
494:+        # spatial_mode="graph" (default): graph-aware spatial attn (adjacency+geodesic
500:+            use_graph_bias=(spatial_mode == "graph"))
522:+        spatial_mode: str = "graph",
532:+        if spatial_mode not in ("graph", "plain"):
534:+                f"spatial_mode must be 'graph' (graph-aware spatial attn) or "
535:+                f"'plain' (no_graph_spatial ablation), got {spatial_mode!r}"
537:+        self.spatial_mode = spatial_mode
548:+                                        text_mode=text_mode, spatial_mode=spatial_mode)

exec
/usr/bin/bash -lc "git show HEAD:src/models/graph_salad/denoiser.py | nl -ba | sed -n '210,235p;330,400p'" in /iridisfs/scratch/ts1v23/workspace/noKslot_clean
 succeeded in 0ms:
   210	    def __init__(
   211	        self,
   212	        d_model: int,
   213	        n_heads: int,
   214	        d_ff: int,
   215	        d_t: int,
   216	        dropout: float = 0.1,
   217	        text_mode: str = "mean_additive",
   218	    ) -> None:
   219	        super().__init__()
   220	        self.text_mode = text_mode
   221	        self.spatial = GraphAttentionBlock(d_model, n_heads, d_ff, dropout=dropout)
   222	        self.temporal = TemporalSelfAttention(d_model, n_heads, dropout=dropout)
   223	        self.film_after_spatial = DenseFiLM(d_t, d_model)
   224	        self.film_after_temporal = DenseFiLM(d_t, d_model)
   225	        self.film_after_text = DenseFiLM(d_t, d_model)
   226	        # Token cross-attn sub-block exists only in token mode (so mean-mode
   227	        # state_dict is byte-identical to old ckpts — strict-load preserved).
   228	        if text_mode == "token_cross_attn":
   229	            self.text_cross_attn = TextCrossAttention(d_model, n_heads, dropout=dropout)
   230	
   231	    def forward(
   232	        self,
   233	        x: torch.Tensor,                  # [B, T_lat, C, D]
   234	        t_emb: torch.Tensor,              # [B, D_t]
   235	        text_cond: torch.Tensor | None,   # [B, D] projected mean text (mean_additive)
   330	            )
   331	        if n_layers % 2 == 0:
   332	            raise ValueError(
   333	                f"n_layers must be odd for SALAD skip-transformer "
   334	                f"(enc + mid + dec); got {n_layers}"
   335	            )
   336	        if d_model % n_heads != 0:
   337	            raise ValueError(f"d_model ({d_model}) must divide n_heads ({n_heads})")
   338	        if d_ff is None:
   339	            d_ff = 4 * d_model
   340	        if d_t is None:
   341	            d_t = d_model
   342	
   343	        self.d_model = d_model
   344	        self.n_heads = n_heads
   345	        self.d_ff = d_ff
   346	        self.n_layers = n_layers
   347	        self.d_text = d_text
   348	        self.d_t = d_t
   349	        self.text_mode = text_mode
   350	        self.text_token_dim = text_token_dim
   351	
   352	        # --- Timestep embedding (shared across all layers' FiLMs) ---
   353	        self.t_sin = SinusoidalTimestepEmbedding(d_t)
   354	        self.t_mlp = nn.Sequential(
   355	            nn.Linear(d_t, d_t * 4),
   356	            nn.SiLU(),
   357	            nn.Linear(d_t * 4, d_t),
   358	        )
   359	
   360	        # --- Text projection (T5-base 768 → d_model); shared across layers ---
   361	        # Per design §2.3: denoiser owns its own text_proj (NOT reusing VAE's).
   362	        # mean_additive: projects the [B,768] mean-pooled caption.
   363	        self.text_proj = nn.Linear(d_text, d_model)
   364	        # token_cross_attn: separate projection for token-level T5 [B,L,768]→[B,L,D]
   365	        # (exists only in token mode → mean-mode ckpts stay byte-identical).
   366	        if text_mode == "token_cross_attn":
   367	            self.text_token_proj = nn.Linear(text_token_dim, d_model)
   368	
   369	        # --- Input projection: latent z + slot conditioning ---
   370	        self.input_proj = nn.Linear(d_model, d_model)
   371	
   372	        # --- Skip-transformer stack ---
   373	        # n_layers = 2*depth + 1; depth pairs (enc[i], dec[i]) + 1 middle.
   374	        # For n_layers=5: depth=2, layers = [enc0, enc1, mid, dec0, dec1] with
   375	        # skip(enc1->dec0) and skip(enc0->dec1).
   376	        self.layers = nn.ModuleList(
   377	            [
   378	                GraphSaladDenoiserLayer(d_model, n_heads, d_ff, d_t, dropout=dropout,
   379	                                        text_mode=text_mode)
   380	                for _ in range(n_layers)
   381	            ]
   382	        )
   383	        # Skip mergers: one per decoder layer
   384	        self.depth = (n_layers - 1) // 2
   385	        self.skip_mergers = nn.ModuleList(
   386	            [nn.Linear(2 * d_model, d_model) for _ in range(self.depth)]
   387	        )
   388	
   389	        # --- Output projection: D → D (predicts v_pred at same dim as z_t) ---
   390	        self.output_norm = nn.LayerNorm(d_model)
   391	        self.output_proj = nn.Linear(d_model, d_model)
   392	        # Zero-init output_proj weights so initial v_pred ≈ 0 (training-stable
   393	        # for diffusion; common practice — e.g. DiT, U-Net).
   394	        nn.init.zeros_(self.output_proj.weight)
   395	        nn.init.zeros_(self.output_proj.bias)
   396	
   397	    def forward(
   398	        self,
   399	        z_t: torch.Tensor,
   400	        timesteps: torch.Tensor,

codex
The checkpoint path records `vars(args)` at every save site and the resume/init guards compare checkpoint mode to CLI before strict load. I’m doing one small runtime smoke on constructor keys and plain forward to catch missed attribute references.
exec
/usr/bin/bash -lc "nl -ba scripts/_launch_diffusion_t2m.sh | sed -n '118,155p'" in /iridisfs/scratch/ts1v23/workspace/noKslot_clean
 succeeded in 0ms:
   118	torchrun $RDZV_ARGS scripts/train_denoiser.py \
   119	  --vae_ckpt "$VAE_CKPT" \
   120	  --caption_emb_cache "$CAPCACHE" \
   121	  --anytop_root "$ANYTOP_ROOT" \
   122	  --max_frames 260 --max_joints 144 \
   123	  --batch_size "$PER_GPU_BATCH" --lr "$LR" --epochs "$EPOCHS" \
   124	  --warmup_iters "$WARMUP_ITERS" --lr_schedule "$LR_SCHEDULE" --lr_min "$LR_MIN" \
   125	  --train_split "$TRAIN_SPLIT" \
   126	  --w_lat_dz "$W_LAT_DZ" --w_lat_ddz "$W_LAT_DDZ" --w_lat_x0 "$W_LAT_X0" \
   127	  --latent_dyn_target "$LATENT_DYN_TARGET" \
   128	  --spatial_mode "$SPATIAL_MODE" \
   129	  ${SPECIES_WHITELIST:+--species_whitelist "$SPECIES_WHITELIST"} \
   130	  ${INIT_CKPT:+--init_ckpt "$INIT_CKPT"} \
   131	  ${RESUME_CKPT:+--resume "$RESUME_CKPT"} \
   132	  --n_layers "$N_LAYERS" --d_ff "$D_FF" --dropout 0.1 \
   133	  --num_train_timesteps 1000 --beta_start 0.00085 --beta_end 0.012 \
   134	  --beta_schedule scaled_linear --cond_drop_prob 0.1 --amp_dtype "$AMP_DTYPE" \
   135	  --text_mode "$TEXT_MODE" --caption_token_max_len "$CAPTION_TOKEN_MAX_LEN" \
   136	  ${CAPTION_TOKEN_CACHE:+--caption_token_cache "$CAPTION_TOKEN_CACHE"} \
   137	  --val_every 5 --save_every 10 --periodic_save_every 100 \
   138	  --seed 42 --out "$OUT" --overwrite $SMOKE_FLAG
   139	rc=$?
   140	echo "[t2m] $(date '+%F %T %Z') torchrun EXITED rc=$rc"
   141	exit "$rc"

exec
/usr/bin/bash -lc "python - <<'PY'
import torch
from src.models.graph_salad.attention import GraphAttentionBlock
from src.models.graph_salad.denoiser import GraphSaladDenoiser

torch.manual_seed(0)
for use in (True, False):
    b = GraphAttentionBlock(16, 4, 32, dropout=0.0, use_graph_bias=use)
    print('block', use, 'has_geo', hasattr(b, 'geodesic_bias'), 'has_adj', hasattr(b, 'adjacency_bias'), 'keys', [k for k in b.state_dict().keys() if 'bias' in k and ('geodesic' in k or 'adjacency' in k)])
    x = torch.randn(2,3,16)
    adj = torch.tensor([[[0,1,0],[1,0,1],[0,1,0]], [[0,1,0],[1,0,0],[0,0,0]]], dtype=torch.float32)
    geo = torch.tensor([[[0,1,2],[1,0,1],[2,1,0]], [[0,1,float('inf')],[1,0,float('inf')],[float('inf'),float('inf'),0]]], dtype=torch.float32)
    mask = torch.tensor([[1,1,1],[1,1,0]], dtype=torch.bool)
    y = b(x, adj, geo, mask, validate_inputs=True)
    print('forward', use, tuple(y.shape), torch.isfinite(y).all().item())

g = GraphSaladDenoiser(d_model=16,n_heads=4,d_ff=32,n_layers=3,spatial_mode='graph')
p = GraphSaladDenoiser(d_model=16,n_heads=4,d_ff=32,n_layers=3,spatial_mode='plain')
gkeys=set(g.state_dict())
pkeys=set(p.state_dict())
print('denoiser bias key diff', sorted(gkeys-pkeys)[:10], 'count', len(gkeys-pkeys), 'reverse', len(pkeys-gkeys))
PY" in /iridisfs/scratch/ts1v23/workspace/noKslot_clean
 succeeded in 5780ms:
block True has_geo True has_adj True keys ['geodesic_bias.weight', 'adjacency_bias.weight']
forward True (2, 3, 16) True
block False has_geo False has_adj False keys []
forward False (2, 3, 16) True
denoiser bias key diff ['layers.0.spatial.adjacency_bias.weight', 'layers.0.spatial.geodesic_bias.weight', 'layers.1.spatial.adjacency_bias.weight', 'layers.1.spatial.geodesic_bias.weight', 'layers.2.spatial.adjacency_bias.weight', 'layers.2.spatial.geodesic_bias.weight'] count 6 reverse 0

codex
The smoke confirms plain blocks have no bias attributes or bias state keys and can run a masked forward without AttributeError. I’m doing syntax checks now; then I’ll return the review verdict in the requested format.
exec
/usr/bin/bash -lc 'bash -n scripts/_launch_diffusion_t2m.sh scripts/_launch_diffusion_t2m_4card.sh scripts/_launch_token_diffusion_8card_a100.sh' in /iridisfs/scratch/ts1v23/workspace/noKslot_clean
 succeeded in 0ms:
exec
/usr/bin/bash -lc 'python -m py_compile src/models/graph_salad/attention.py src/models/graph_salad/denoiser.py scripts/train_denoiser.py scripts/animate_denoiser.py' in /iridisfs/scratch/ts1v23/workspace/noKslot_clean
 succeeded in 114ms:
codex
PASS

1. Graph bias gating is correct: bias modules are only instantiated under `if use_graph_bias` and only referenced under `if self.use_graph_bias`, so plain mode has no unguarded `geodesic_bias` / `adjacency_bias` access. Evidence: `src/models/graph_salad/attention.py:115`, `src/models/graph_salad/attention.py:362`.

2. Graph-mode state/load compatibility looks preserved: graph mode still creates the same bias module names, while `spatial_mode` / `use_graph_bias` are plain attributes, not state keys. Denoiser validates `graph|plain`, stores the mode, and threads it to every layer. Evidence: `src/models/graph_salad/attention.py:101`, `src/models/graph_salad/attention.py:115`, `src/models/graph_salad/denoiser.py:345`, `src/models/graph_salad/denoiser.py:370`, `src/models/graph_salad/denoiser.py:397`.

3. Train wiring and checkpoint guards are present: CLI default is `graph`, construction passes `args.spatial_mode`, checkpoints save `vars(args)`, and both resume/init check checkpoint mode before strict load. Evidence: `scripts/train_denoiser.py:329`, `scripts/train_denoiser.py:612`, `scripts/train_denoiser.py:643`, `scripts/train_denoiser.py:677`, `scripts/train_denoiser.py:1055`.

4. Animation rebuilds from checkpoint mode with old-checkpoint fallback to `graph`, then strict-loads. Evidence: `scripts/animate_denoiser.py:68`, `scripts/animate_denoiser.py:70`, `scripts/animate_denoiser.py:77`.

5. Launcher propagation is correct for the scoped files: inner launcher defaults `SPATIAL_MODE=graph` and passes `--spatial_mode`; 4-card and 8-card orchestrators default and thread `SPATIAL_MODE` through `COMMON_ENV`. 4-card also includes caption token cache envs. Evidence: `scripts/_launch_diffusion_t2m.sh:55`, `scripts/_launch_diffusion_t2m.sh:128`, `scripts/_launch_diffusion_t2m_4card.sh:46`, `scripts/_launch_diffusion_t2m_4card.sh:61`, `scripts/_launch_token_diffusion_8card_a100.sh:41`, `scripts/_launch_token_diffusion_8card_a100.sh:66`.

Non-blocking notes:

1. Plain mode still requires `adjacency` and `geodesic_dist` tensors and, on validated calls, still checks their graph semantics. That can fail on malformed/missing graph metadata even though plain compute does not use topology bias. In the current train/animate paths this is harmless because those tensors are still always supplied. Evidence: `src/models/graph_salad/denoiser.py:549`, `src/models/graph_salad/attention.py:159`, `scripts/train_denoiser.py:855`, `scripts/animate_denoiser.py:143`.

2. The orchestrator `COMMON_ENV` strings are not shell-escaped for arbitrary values, but the scoped `SPATIAL_MODE=graph|plain`, cache paths, and comma-separated whitelist values are safe under the current usage. Evidence: `scripts/_launch_diffusion_t2m_4card.sh:61`, `scripts/_launch_token_diffusion_8card_a100.sh:66`.

3. I smoke-checked constructor/forward behavior: plain blocks have no graph-bias attributes or graph-bias state keys, plain masked forward runs, and graph-vs-plain denoiser state keys differ only by per-layer adjacency/geodesic bias weights.
tokens used
84,491
PASS

1. Graph bias gating is correct: bias modules are only instantiated under `if use_graph_bias` and only referenced under `if self.use_graph_bias`, so plain mode has no unguarded `geodesic_bias` / `adjacency_bias` access. Evidence: `src/models/graph_salad/attention.py:115`, `src/models/graph_salad/attention.py:362`.

2. Graph-mode state/load compatibility looks preserved: graph mode still creates the same bias module names, while `spatial_mode` / `use_graph_bias` are plain attributes, not state keys. Denoiser validates `graph|plain`, stores the mode, and threads it to every layer. Evidence: `src/models/graph_salad/attention.py:101`, `src/models/graph_salad/attention.py:115`, `src/models/graph_salad/denoiser.py:345`, `src/models/graph_salad/denoiser.py:370`, `src/models/graph_salad/denoiser.py:397`.

3. Train wiring and checkpoint guards are present: CLI default is `graph`, construction passes `args.spatial_mode`, checkpoints save `vars(args)`, and both resume/init check checkpoint mode before strict load. Evidence: `scripts/train_denoiser.py:329`, `scripts/train_denoiser.py:612`, `scripts/train_denoiser.py:643`, `scripts/train_denoiser.py:677`, `scripts/train_denoiser.py:1055`.

4. Animation rebuilds from checkpoint mode with old-checkpoint fallback to `graph`, then strict-loads. Evidence: `scripts/animate_denoiser.py:68`, `scripts/animate_denoiser.py:70`, `scripts/animate_denoiser.py:77`.

5. Launcher propagation is correct for the scoped files: inner launcher defaults `SPATIAL_MODE=graph` and passes `--spatial_mode`; 4-card and 8-card orchestrators default and thread `SPATIAL_MODE` through `COMMON_ENV`. 4-card also includes caption token cache envs. Evidence: `scripts/_launch_diffusion_t2m.sh:55`, `scripts/_launch_diffusion_t2m.sh:128`, `scripts/_launch_diffusion_t2m_4card.sh:46`, `scripts/_launch_diffusion_t2m_4card.sh:61`, `scripts/_launch_token_diffusion_8card_a100.sh:41`, `scripts/_launch_token_diffusion_8card_a100.sh:66`.

Non-blocking notes:

1. Plain mode still requires `adjacency` and `geodesic_dist` tensors and, on validated calls, still checks their graph semantics. That can fail on malformed/missing graph metadata even though plain compute does not use topology bias. In the current train/animate paths this is harmless because those tensors are still always supplied. Evidence: `src/models/graph_salad/denoiser.py:549`, `src/models/graph_salad/attention.py:159`, `scripts/train_denoiser.py:855`, `scripts/animate_denoiser.py:143`.

2. The orchestrator `COMMON_ENV` strings are not shell-escaped for arbitrary values, but the scoped `SPATIAL_MODE=graph|plain`, cache paths, and comma-separated whitelist values are safe under the current usage. Evidence: `scripts/_launch_diffusion_t2m_4card.sh:61`, `scripts/_launch_token_diffusion_8card_a100.sh:66`.

3. I smoke-checked constructor/forward behavior: plain blocks have no graph-bias attributes or graph-bias state keys, plain masked forward runs, and graph-vs-plain denoiser state keys differ only by per-layer adjacency/geodesic bias weights.
