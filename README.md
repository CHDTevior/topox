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

`scripts/deploy_generate.py` — in progress on this branch: BVH skeleton (or a KTJD-17 skeleton `.npz`) + a text
prompt → generated motion (positions `.npz`, GIF, BVH). It rebuilds every conditioning tensor from the skeleton file
alone (graph features, spectral coordinates, joint-description embeddings) and encodes the prompt with LLM2Vec.
A rig without motion statistics uses a documented fallback for the per-channel normalisation scale.
