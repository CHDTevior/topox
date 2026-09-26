# TopoX — text-to-motion for arbitrary skeletons (H1 release branch)

This branch (`release/h1-uniml3d-73m`) carries the code that trained and runs the **H1** model: a 73M-parameter
in-context motion diffusion transformer over the KTJD-17 representation, trained on the UniML3D v2 "common" cut
(6,747 clips / 5,487 rigs, the split shared with the UniMate comparison). Recipe: heat-kernel-signature spectral
joint RoPE + temporal RoPE, UniMate-rule skeleton augmentation, rig sampling weight = clips^0.5, rest-pose
normalisation, 1-frame rest-pose demo, LLM2Vec text conditioning.

Newer arms (R2: world-frame rest descriptor + rest-convention augmentation; R1: descriptor only) are present in the
code behind flags that default OFF; with the flags off every served item and every model tensor is byte-identical
to the H1 code.

## Weights

Private HuggingFace repo `Tevior/topox-h1-uniml3d-73m` (ask the owner for access):

| file | what | sha256 |
|---|---|---|
| `topox_h1_uniml3d73m_ep239_infer.pt` | H1 checkpoint, optimizer state stripped (model state dict + training args + data bindings), 293 MB | `282b077c8374551ce59b395ab370687c943f56c80b919638e6ff5d3df6004b26` |
| `pilot_uniml3dv2common_bothrope_aug_d512_hks_gamma_calibration_b16_v1.json` | the run's loss-weight calibration artifact (training only) | — |

Load it with `scripts/_eval_v2_gen_in_evalspace.py::load_gen_model(torch.load(path, map_location="cpu", weights_only=False), device)`
(builds `InContextMotionDiT` from the checkpoint's own args and loads strictly).

## Environment

Python 3.12, PyTorch 2.10 (CUDA 12.8), numpy 2.2, Pillow; text encoding needs `llm2vec` + `transformers` + `peft`
and the LLM2Vec checkpoints `McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp` (+ `-supervised`) on top of
`meta-llama/Meta-Llama-3-8B-Instruct` (gated; ~16 GB in bf16). `requirements.txt` is the older baseline list;
the exact environment is the maintainer's conda env, described in `docs/` where present.

## Layout

- `src/models/v2/dit_motion.py` — the model (`InContextMotionDiT`) and the sampler (`sample`).
- `src/models/v2/spec_rope.py`, `temporal_rope.py`, `src/data/skeleton_spectral.py` — the two rotaries and the spectral / heat-kernel coordinates.
- `src/data/ktjd17/` — the KTJD-17 representation: schema, codec (encode / decode, FK), skeleton construction.
- `src/data/ktjd17_incontext.py`, `src/data/incontext_pairs.py` — corpus adapter and the pair dataset (what the model sees).
- `src/data/ktjd17_augment.py` — skeleton augmentation (UniMate rule; rest-convention channel).
- `scripts/train_v2_incontext.py`, `scripts/_launch_v2_ddp_2node_h200.sh`, `configs/pilot36m_uniml3dv2common_bothrope_aug_d512_lr15_hks_2node_env.sh` — training (the H1 config is the standing default).
- `scripts/v2_render_incontext.py` — generation + GIF rendering on corpus rigs (seen or zero-shot).
- `scripts/_probe_rest_convention.py` — zero-training probe: how much the generated world motion moves when the rig's rest convention is rotated.

## Generating on corpus rigs

Needs the KTJD-17 corpus artifacts the checkpoint was trained on (skeleton files, manifests, per-rig statistics,
LLM2Vec caption cache, joint-description embeddings; not in this repo — ask the owner). Example (the H1 close-out
renders, `renders/h1_closeout/_render.sh`):

```bash
python scripts/v2_render_incontext.py --ckpt <ckpt> --out <dir> --corpus ktjd17 --steps 20 --cfg_text 2.0 \
    --rigs_A <rig ids> --rigs_B= --rigs_T= --pick energetic --seed 7
```

## Generating on your own skeleton (deploy)

`scripts/deploy_generate.py` needs only the checkpoint, a skeleton and a prompt -- no training corpus.

```bash
# 1. environment: Python 3.10+, PyTorch 2.x (CUDA), numpy, scipy, Pillow, and for text encoding
pip install llm2vec transformers==4.44.2 peft
huggingface-cli login            # needs access to meta-llama/Meta-Llama-3-8B-Instruct (gated) for LLM2Vec
# 2. weights (private repo, ask the owner for access)
huggingface-cli download Tevior/topox-h1-uniml3d-73m topox_h1_uniml3d73m_ep239_infer.pt --local-dir weights/
# 3. check the rig first: prints the joint descriptions and writes out/my_rig_s7.rest.gif (the rest pose in the model's frame)
python scripts/deploy_generate.py --ckpt weights/topox_h1_uniml3d73m_ep239_infer.pt \
    --skeleton my_rig.bvh --up +Y --forward +Z --describe_only --out out/
# 4. generate (4 s at 30 fps); writes out/my_rig_s7.{npz,gif,bvh}
python scripts/deploy_generate.py --ckpt weights/topox_h1_uniml3d73m_ep239_infer.pt \
    --skeleton my_rig.bvh --up +Y --forward +Z --text "An object walks forward." --frames 120 --seed 7 --out out/
```

- **Rest pose.** The BVH's frame `--rest_frame` (default 0) is the rest pose: export the rig with its T-pose / rest pose as
  the first frame. `--rest_frame -1` uses the OFFSETs with zero rotations (the rest pose of a Blender export, but NOT of a
  3ds Max Biped export, whose OFFSETs are not the rest pose). Look at `<name>.rest.gif` before generating.
- **Axes.** `--up` is the BVH axis pointing up, `--forward` the axis the creature faces (Blender BVH exports are usually
  `--up +Y`; the TrueBones animals are `--up +Y --forward +X`). The model's frame is +Y up, +Z forward, +X = the
  creature's left. The script checks the axes against the joint names: joints named Left/Right (.L, _R, ...) must lie on
  their named side; below 80% agreement it refuses and prints the `--forward` the names imply.
- **Prompts.** Training captions are phrased "An object walks forward.", "An object flaps its wings." -- use that
  phrasing ("An object ..."), short and about the motion. A prompt like "A chicken walks forward." is out of the training
  distribution (measured on the TrueBones chicken: the position and rotation decodes disagreed by 0.60 x rig size with
  "A chicken ...", 0.08 with "An object ...").
- **Joint descriptions** come from the joint names (a lexicon learnt from the names of the corpus's 6,360 rig files --
  training, validation and excluded rigs alike: on a held-out 10% of rigs, 95% of the names are covered and 99% of those
  reproduce the corpus description) and, for names without anatomy, from the skeleton's
  geometry. Override any of them with `--descriptions my.json` (`{"joint name": "Left Thigh joint."}`).
- **Outputs** (`out/<skeleton>_s<seed>.*`; an existing output needs `--force`, the input skeleton is never overwritten).
  `.bvh` = your hierarchy (same joints, offsets, End Sites, channel orders) animated with the generated
  rotations and root path, in your axes and units, verified by reading it back. `.npz` = world positions from both
  decodes (`positions_direct` from the position channels, `positions_fk` from the rotations -- the BVH plays the latter),
  in the model's frame. `.gif` = rest | position decode | rotation decode.
- **Validation** (`scripts/_aug_dev/_test_deploy_generate.py`): through this script the 4 zero-shot clips of the H1
  close-out regenerate bit-for-bit; re-encoding captions / joint descriptions with LLM2Vec reproduces the training
  embeddings (cos 1.0000 / >= 0.9999; generation moves 0.002-0.02 bone lengths vs 0.25-1.7 between seeds). Each joint's
  rest frame is rebuilt in the training rigs' convention (local +Y along the bone to the primary child): on 24 training
  rigs this keeps the generated motion 0.62 bone lengths from the rigs' own convention, where identity rest frames (a raw
  BVH) end 1.64 away (seed-to-seed spread 1.06).
