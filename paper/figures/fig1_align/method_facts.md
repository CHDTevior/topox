# TopX — method facts for Figure 1 (verified against code and paper/sections/method.tex, 2026-09-06)

Relationships a reader must recover, in order:
1. A rig is an INPUT: kinematic tree (34–102 joints; 311 rigs in the library, or a rig never seen), rest pose, and one
   natural-language description per joint ("the head", "the left front foot", "the left claw on the fifth digit of the front foot").
2. Any rig converts to the same per-joint representation (KTJD-17): one token per (frame, joint) with 17 channels = position (3),
   6D rotation delta from the joint's rest orientation (6), velocity (3), contact (1); root row also carries smoothed root trajectory
   (2) and heading (2). Every (rig, joint, channel) cell is standardised with its own mean/std. No template skeleton, no retargeting.
3. The sequence fed to the network is [ rest frame | 240 target frames ]: the rest frame is the rig's rest pose, CLEAN, never noised
   (the skeleton prompt); the target frames carry the flow interpolant x_t = (1−t)·x_0 + t·x_1, x_0 ~ N(0, I); a mask token marks
   target frames. Only target frames are denoised and scored.
4. Token embedding: Linear(17→D) + learned frame position + learned joint-slot position + joint semantics (LLM2Vec embedding of the
   joint description, 4096-d, frozen encoder → Linear → ADDED to every token of that joint) + 8-d structural descriptor of the joint
   (rest-offset direction, log bone length, depth, root-path length, child count, leaf flag) through a zero-initialised MLP.
5. Conditioning vector c = MLP(timestep embedding of t) + MLP(LayerNorm(pooled LLM2Vec caption embedding)); c drives AdaLN
   (scale, shift, gate) of every sub-layer of every block and of the output layer. The caption reaches the model ONLY through this
   modulation; it never enters the token stream.
6. Block ×14 (D = 896, 14 heads, 303M params; a 36M pilot: D = 384, 8 blocks): temporal self-attention (tokens = frames of one
   joint) → spatial self-attention (tokens = joints of one frame, attention bias −min(g_ij, 8) from the geodesic hop distance on the
   kinematic tree + learned per-head tables indexed by hops up/down to the lowest common ancestor) → 4× MLP. Queries/keys RMS-normalised.
7. Output: LayerNorm + AdaLN → Linear(D→17) = x̂_1, the predicted CLEAN motion (x-prediction). Training loss: calibrated grouped
   Huber loss over 9 channel groups (root/body × position/rotation/velocity, contact, smoothed root, heading), weights
   γ_g·sqrt(N_g/N) with γ solved from a pre-registered share profile and measured energies; auxiliary FK-consistency, foot-lock,
   acceleration terms.
8. Sampling: 20 Euler steps x ← x + (x̂_1 − x)/(S − i), classifier-free guidance on the caption only (s = 2); each step updates the
   target frames only (the rest frame stays clean); invalid cells re-zeroed every step.
9. Decoding: positions directly; rotations → forward kinematics on the rig → skinned character (game mesh).
10. A rig never seen: the SAME forward pass once its rest pose, joint descriptions and per-cell statistics are supplied (zero
    training); with ~10 clips, a per-rig LoRA (rank 64) folded back into the weights.

Facts the figure states: 311 rigs; 34–102 joints; 17 channels; [1 | 240] frames; ×14 blocks; D = 896; 303M; 20 Euler steps; s = 2;
LLM2Vec 4096-d frozen; −min(hops, 8) tree bias; ~10 clips → LoRA.
Left to the caption: Huber δ, the velocity-space weight w(t), σ_min, the learning-rate schedule, the evaluator.

Real assets that may be embedded as raster crops (already rendered from the model's own output on a moose run-to-walk clip):
- paper/figures/framework/glyphs/rest_tags.pdf (rest pose with three description tags), rest.pdf, rest_gharial.pdf, rest_lemur.pdf
- paper/figures/framework/glyphs/clean_{0..3}.pdf (predicted clean frames), noisy_{0..3}.pdf (noise-mixed frames), motion_strip.pdf
- paper/figures/framework/glyphs/bias.pdf (the real −min(hops,8) matrix of that rig), skinned_moose.png (skinned render)
Palette of the paper: skeleton/structure path teal #1F6F8B, caption path clay #B5552F, ink #2B2B2B, greys.
Composition sources (arrangement only): composition_source_tikz_v15.png (our current TikZ draft) and composition_source_codex_draft.png.
