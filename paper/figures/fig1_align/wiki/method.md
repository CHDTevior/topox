# method.md — what Figure 1 must let a reader recover (TopX)
Relationships, in reading order:
1. A rig is an input: a kinematic tree, a rest pose and one description per joint; three visibly different rigs feed one model.
2. Every rig becomes one token grid: a token per (frame, joint) with 17 standardised channels; the FIRST column is the rig's rest
   pose, clean; the following 240 target columns carry the noise-mixed motion x_t.
3. The description of joint j (through the frozen text encoder and a linear map) is added to every token of row j.
4. The caption and the flow time, each through the same frozen text encoder / an MLP, are summed into c, which modulates every
   sub-layer (AdaLN); text never enters the tokens.
5. One transformer, 14 blocks: temporal attention (a joint over frames) → spatial attention (a frame over joints, biased by tree
   distance) → MLP.
6. The network predicts the clean motion x̂₁ for the target frames; 20 Euler steps with caption guidance s = 2 update the target
   frames only, the rest column stays clean.
7. The result is the motion of this rig; rotations → forward kinematics → the skinned character.
8. A rig never seen takes the same pass (rest pose, descriptions, statistics); optionally LoRA from ~10 clips.
Facts stated: 311 rigs, 34–102 joints; 17 channels; [rest | 240] frames; ×14; D = 896, 303M; 20 Euler steps, s = 2; LLM2Vec
frozen; tree-distance bias; ~10 clips → LoRA.
Left to the caption: loss weights and Huber knee, the velocity-space weight, σ_min, the schedule, the evaluator, the 36M pilot.
