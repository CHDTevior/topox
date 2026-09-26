# Demo rigs

Ready-to-run examples for `scripts/deploy_generate.py` with the H1 checkpoint: skeletons from the training set, their training captions, and the embeddings the model was trained with, so no text encoder (LLM2Vec / Llama-3-8B) is needed to run them. Each folder holds `skeleton.npz` (KTJD-17 skeleton, the fields the deploy script reads), `joint_sem.npy`, `text_emb_<k>.npy`, `prompts.json` and the reference output `ref_<k>.gif` / `ref_<k>.bvh` of each prompt (made on an H200 in fp32 by the same command; another GPU may differ in the last digits).

Picked automatically among the training rigs whose Sketchfab licence is CC BY or CC0: for each rig, its most energetic training captions were generated with two seeds and compared with the training clip; a demo had to move 0.7-1.4x as much as the clip, keep the position and rotation decodes within 0.15 x the rig size, stay under 1.8x its jitter and within 0.4 bone lengths of its ground clearance, and at most four rigs per body plan were taken, best first. Left out although they pass: one rig whose model title contains an ID-like number, one rig whose source rest pose stands upright while its motion lies on its side and two rigs whose training clip is only 15 frames. For body-plan variety two demos were admitted just outside those bounds (`spider`: decode gap 0.16; `bat`: 2.3x jitter). Every demo was looked at before it was kept; `prompts.json` records its scores.

```bash
bash demo/run_demos.sh            # CKPT=weights/topox_h1_uniml3d73m_ep239_infer.pt, outputs in out/demos/
```

| demo | UniML3D body plan | joints | prompt | seed | frames |
|---|---|---|---|---|---|
| `triceratops` | quadrupedal | 29 | An object walks in place. | 7 | 61 |
| `triceratops` | quadrupedal | 29 | An object stomps its feet in place. | 7 | 130 |
| `stag` | quadrupedal | 28 | An object rears up and lowers its head. | 7 | 92 |
| `wolf_lowpoly` | quadrupedal | 37 | An object crouches and then stands while flapping its arms. | 7 | 113 |
| `crawling_human` | marine | 44 | An object swims in place with arms and legs moving. | 17 | 149 |
| `mech_striker` | bipedal | 16 | An object walks forward. | 17 | 31 |
| `mech_striker` | bipedal | 16 | An object kicks with one leg. | 17 | 26 |
| `viking_worker` | bipedal | 30 | An object falls backward and lies on its back. | 17 | 68 |
| `viking_worker` | bipedal | 30 | An object crouches and falls forward. | 17 | 34 |
| `bear` | quadrupedal | 20 | An object walks in place. | 17 | 200 |
| `spider` | insectoid | 24 | An object scuttles in place with legs moving. | 17 | 31 |
| `bat` | avian | 48 | An object flaps its wings in place. | 7 | 60 |

## Attribution and licences

The skeletons come from 3D models published on Sketchfab and indexed by Objaverse-XL (Allen Institute for AI), taken from the UniML3D export of Objaverse-XL. Each model keeps its creator's licence, shown below as listed on Sketchfab on 2026-09-26; nothing here grants rights beyond it.

The prompts are UniML3D captions, the joint descriptions in `skeleton.npz` are built from UniML3D's cleaned joint labels, and the body plans are UniML3D's categories. UniML3D (https://huggingface.co/datasets/Linzhan/UniML3D) offers these annotations under [ODC-BY 1.0](https://opendatacommons.org/licenses/by/1-0/); please cite

```bibtex
@article{mou2026unimate,
  title   = {UniMate: One Unified Model to Animate Diverse Skeletons},
  author  = {Mou, Linzhan and Lei, Jiahui and Dou, Zhiyang and Cai, Chenyue and Song, Chaoyue and Finkelstein, Adam and Rusinkiewicz, Szymon},
  journal = {arXiv preprint arXiv:2609.05415},
  year    = {2026}
}
```

Changes: from each model only the skeleton (joint hierarchy, rest pose and joint names) was extracted and converted to the KTJD-17 format; no mesh, texture or animation of the original model is included. Added to each skeleton: the UniML3D caption of each prompt, joint descriptions built from UniML3D's joint labels, and LLM2Vec embeddings of both (`text_emb_<k>.npy`, `joint_sem.npy`). If you are a rights holder and want a skeleton removed, open an issue on this repository.

| demo | model | author | licence | source |
|---|---|---|---|---|
| `triceratops` | Triceratops occultatum | Miguelangelo Rosario | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) | https://sketchfab.com/3d-models/triceratops-occultatum-d8b6a381f36c46f8b1d59ed6e0b57c65 |
| `stag` | Roaring Stag ( deepdreamed ) | Miguelangelo Rosario | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) | https://sketchfab.com/3d-models/roaring-stag-deepdreamed-4798d8c87a0e4ad8835217fe93ddf67b |
| `wolf_lowpoly` | Low poly Snake - Wolf Animated | imitate | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) | https://sketchfab.com/3d-models/low-poly-snake-wolf-animated-c0baad2baee5467894087856cac1872b |
| `crawling_human` | Crawling mutated human | Elisey | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) | https://sketchfab.com/3d-models/crawling-mutated-human-a87532a3e89947159cc1303008c06eaf |
| `mech_striker` | Medium Mech Striker | MSGDI | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) | https://sketchfab.com/3d-models/medium-mech-striker-27ba717c173a40b7841d2f2c6a89d823 |
| `viking_worker` | Viking Framps Worker | RG3D | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) | https://sketchfab.com/3d-models/viking-framps-worker-cef34035d74c4dfdb9cad45fa36da294 |
| `bear` | Bear Walk | tiikeri | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) | https://sketchfab.com/3d-models/bear-walk-ffd55e32c04c498681ed11584bdd49a5 |
| `spider` | Blackarachnia | PCIXOPAT | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) | https://sketchfab.com/3d-models/blackarachnia-237ce5d21a3c4bcab51f34a4ab451587 |
| `bat` | Bat Bay04 | thanhnhan | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) | https://sketchfab.com/3d-models/bat-bay04-614ab66892894acabcda5cc4a94f87fe |
