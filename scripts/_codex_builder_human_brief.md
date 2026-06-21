Review a small change to scripts/build_anytop_t2m_eval_splits.py. Reply with an explicit verdict: PASS or NEEDS-FIX + enumerated issues.

## Context
This M0 builder generates the AnyTop T2M evaluator eval-split manifests (eval_splits/*.json) from splits/{train,val}.txt + a caption JSON. We are about to run it on the NEW merged dataset data/animo4d_anytop_clean_L4_safe_plus_humanml3d (312 obj = 311 animal + 1 human; train 94170 / val 5190; human rows have source_dataset='HumanML3D' and motion_id like 'HML3D_Human_XXXXXX'; animal rows have NO source_dataset key so build_record sets source='unknown'). Measured split: train human 23378 / animal 70792; val human 1460 / animal 3730.

## The change I made
The prior per-source val split used a literal substring filter `"truebones" in source` -> val_truebones.json / val_animo4d.json. On THIS dataset that is meaningless (no truebones; val_truebones would be empty, val_animo4d would be everything). I REPLACED it with a human-vs-animal split:
- def is_human(r): return ("humanml3d" in str(r["source"]).lower()) or str(r["motion_id"]).upper().startswith("HML3D")
- val_human = [r for r in val_recs if is_human(r)] ; val_animal = [r for r in val_recs if not is_human(r)]
- dump("val_human", val_human) ; dump("val_animal", val_animal)  (replacing the val_animo4d/val_truebones dumps)
- audit dict + final print updated to val_human / val_animal counts; canonical_action_key string updated to mention HumanML3D grouping.
No other logic changed (record schema, caption blacklist, t5_keys validation, train/val integrity asserts, val_action_clean/overlap by source_motion_id all unchanged).

## Intended run command
python3 scripts/build_anytop_t2m_eval_splits.py --data_root data/animo4d_anytop_clean_L4_safe_plus_humanml3d --expect_train 94170 --expect_val 5190 --cap_json data/animo4d_anytop_clean_L4_safe_plus_humanml3d/motion_texts_by_file.json --t5_keys data/anytop_caption_t5_l4safe_human_multi.keys.json

## Verdict must cover
(a) Is the is_human detection correct + exhaustive for this dataset (human=HumanML3D source OR HML3D_ motion_id prefix; animal=everything else incl source='unknown')? Any way a row is mis-classified or dropped (every val rec must land in exactly one of val_human/val_animal)?
(b) Does replacing val_animo4d/val_truebones with val_human/val_animal break anything? (the trainer/launcher wires only val_all.json; the per-source manifests are for separate post-hoc reporting — confirm nothing else in the repo hard-reads val_animo4d.json/val_truebones.json such that removing them breaks a consumer. If you cannot verify consumers, say so.)
(c) Will the builder run cleanly on the new dataset with the intended command — esp.: caption JSON coverage (caps must be a superset of every train+val filename, else build_record hard-fails), the t5_keys validation (every emitted f'{stem}__cap{i}' must exist in data/anytop_caption_t5_l4safe_human_multi.keys.json), and the human records' fields (source_motion_id = real HumanML3D id which CAN repeat across train/val -> affects val_action_clean/overlap; object_type=None for human; species_stripped ANIMAL_RE won't match human captions so has_species_stripped=False for human — all acceptable since we train --view full and skip species_stripped in v1).
(d) Any bug in my edit (python correctness, the boolean, the dumps, the audit keys)?

Read scripts/build_anytop_t2m_eval_splits.py. Be concrete with line references. If PASS, say so plainly.
