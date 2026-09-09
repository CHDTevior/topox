# TopX information flow — ASCII version, every line grep-able (2026-09-06)

Source of truth: `src/models/v2/dit_motion.py` (InContextMotionDiT; run12 args.json: two_stage=False, dim 896, depth 14, heads 14,
qk_norm, struct_feats, dir_bias, demo_rest, demo_frames 1, target_frames 240, p_drop_text 0.1) and `src/data/incontext_pairs.py`.

```
INPUT (one rig, one caption, one flow time t)
  x[T=1+240, J, 17]      frame 0 = REST POSE of the rig, clean, never noised            incontext_pairs.py:191-198
                         frames 1..240 = x_t = (1-t)*x0 + t*x1, x0 ~ N(0,I)               dit_motion.py:655 (train) / :802 (sample)
                         17 = position 3 | rotation-delta 6 | velocity 3 | contact 1 | root row: smooth XZ 2 + heading 2
                         every (rig, joint, channel) cell standardised by its own mean/std   (PERCELL npz)

TOKEN  h[f, j] in R^D, D = 896  ==  a SUM of six terms                                     dit_motion.py:291-310
  h =  W_in x[f, j]                     linear 17 -> D                        "the channels"            :291
     + P_frame[f]                       learned table [4096, D]                "which frame"             :292
     + P_slot[j]                        learned table [160, D]                 "which slot"              :292
     + m * 1[f is a target frame]       ONE learned vector, target frames only "generate this frame"     :293
     + W_sem LLM2Vec(desc_j)            frozen 4096-d text embedding of the joint's DESCRIPTION -> linear  "which joint"  :305
     + MLP_struct(s_j), s_j in R^8      8 scale-free numbers read off THIS rig's kinematic tree:              "where in the tree" :310
                                        rest-offset direction (3), log bone length, depth/16,
                                        log path-length to root, #children/4, leaf flag                incontext_pairs.py:73-80
                                        MLP = Linear(8->D) SiLU Linear(D->D), last layer zero-init    :242-244
  terms 5-6 depend on the joint only (same for every frame); terms 2 and 4 on the frame only.

GLOBAL CONDITION  c in R^D, one per clip                                                  dit_motion.py:321-323
  c =  MLP_t( sinusoidal(1000 t) )      Linear(D->4D) SiLU Linear(4D->D)                                  :215
     + MLP_text( LLM2Vec(caption) )     LN(4096) Linear(4096->D) SiLU Linear(D->D); pooled, frozen        :216
  The caption enters the network ONLY through c (no text tokens, no cross-attention); the caption
  embedding is zeroed with p = 0.1 in training, which is what classifier-free guidance uses at sampling.

BLOCK x14 (same structure, own weights)                                                    dit_motion.py:104-166
  [shift, scale, gate] x 3 sub-layers = Linear( SiLU(c) ) in R^{9D}      one AdaLN projection per block   :117, :139-140
  each sub-layer f:   h <- h + gate * f( LN(h) * (1 + scale) + shift )                                      :145-165
    1 temporal attention   tokens = the 241 frames of ONE joint                                             :145-148
    2 spatial attention    tokens = the J joints of ONE frame; logits += B[i,k]                            :150-158
                           B[i,k] = -min(hops(i,k), 8) + E_up[up(i,k)] + E_down[down(i,k)] per head       :313-316, incontext_pairs.py:468
    3 feed-forward         Linear(D->4D) GELU Linear(4D->D)                                                 :114, :165
  q, k RMS-normalised per head                                                                              :76-77, :93

HEAD (once, after block 14)                                                                dit_motion.py:336-338
  x1_hat[f, j] = W_out ( LN(h) * (1 + scale_o) + shift_o ),  [shift_o, scale_o] = Linear(SiLU(c)),  D -> 17
  = the model's prediction of the CLEAN motion (x-prediction), all 17 channels of every token

TRAINING                                                                                    dit_motion.py:595-712
  calibrated grouped Huber( x1_hat - x1 ) on target frames, 9 channel groups, weight 1/max(1-t, sigma_min)^2,
  plus FK-consistency / velocity / foot-lock / acceleration terms

SAMPLING  20 Euler steps, s = 2                                                            dit_motion.py:802-879
  x <- [ rest | N(0, I) ]
  for i = 0..19:  t = i/20
      x1_hat = x1_hat(no caption) + 2 * ( x1_hat(caption) - x1_hat(no caption) )      two passes; "no caption" = embedding zeroed
      x[target] <- x[target] + ( x1_hat - x )[target] / (20 - i)                      rest frame never touched
  FIXED through the loop: rest frame, P_frame, P_slot, m, W_sem LLM2Vec(desc), MLP_struct(s), tree bias, caption embedding
  CHANGING: x on the target frames, t (-> c -> every AdaLN)
  after step 20: x = the clean motion; positions used directly, rotations -> FK on the rig -> skinned mesh
```

What the figure must therefore show (each module labelled by its INPUT and by WHERE its output goes):
- token = ⊕ of six terms, with the ⊕ drawn (not a "tower"); "MLP" is never bare — it is "8 tree numbers → MLP → +token";
- caption and t reach the model only as c → AdaLN (shift, scale, gate) at every sub-layer and the head; the mechanism written once;
- ×14 encloses only the three sub-layers; the head sits outside and is named: x̂₁, the predicted clean motion;
- the loop has an update node: x ← x + (x̂₁ − x)/(20 − i) on target frames; what returns is x, not the head's output.
