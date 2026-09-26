"""codex NEEDS-FIX #2/#3/#4/#5 on scripts/_eval_multimodality.py:
  - HF-offline must FAIL FAST (compute nodes have no internet) + local_files_only=True
  - early guards: mm_num_samples>0, mm_times>0, pool>=3
  - row correspondence must key on (dataset_index, caption), not motion_id, + row-count check
  - record the selected mm_idxs / row_keys and LABEL the sample as a stratified subset
"""
import os

p = "scripts/_eval_multimodality.py"
s = open(p).read()

# --- guards + HF-offline fail-fast, right after parse_args() ---
old = """    args = parse_args()
    if args.mm_num_repeats <= args.mm_times:
        raise SystemExit(f"--mm_num_repeats ({args.mm_num_repeats}) must EXCEED "
                         f"--mm_times ({args.mm_times}) — calculate_multimodality asserts it")
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")"""
new = """    args = parse_args()
    if args.mm_num_repeats <= args.mm_times:
        raise SystemExit(f"--mm_num_repeats ({args.mm_num_repeats}) must EXCEED "
                         f"--mm_times ({args.mm_times}) — calculate_multimodality asserts it")
    if args.mm_num_samples <= 0:
        raise SystemExit("--mm_num_samples must be > 0 (0 would divide by zero in the stride)")
    if args.mm_times <= 0:
        raise SystemExit("--mm_times must be > 0 (0 would make MultiModality an empty mean = NaN)")
    if args.pool < 3:
        raise SystemExit(f"--pool must be >= 3 (got {args.pool}); R@3 needs at least 3 candidates")
    # Compute nodes have NO internet. Without these, transformers tries to reach the HF hub and
    # dies mid-run (or hangs). Fail here, not 6 GPU-hours in.
    for _v in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"):
        if os.environ.get(_v) != "1":
            raise SystemExit(f"{_v}=1 is required (compute nodes have no internet); "
                             f"run under: env HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python ...")
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")"""
assert s.count(old) == 1, "parse_args anchor not unique"
s = s.replace(old, new)

# --- local_files_only on both from_pretrained ---
old = """    t5tok = T5TokenizerFast.from_pretrained("t5-base")
    t5 = T5EncoderModel.from_pretrained("t5-base").to(dev).eval()"""
new = """    t5tok = T5TokenizerFast.from_pretrained("t5-base", local_files_only=True)
    t5 = T5EncoderModel.from_pretrained("t5-base", local_files_only=True).to(dev).eval()"""
assert s.count(old) == 1, "t5 anchor not unique"
s = s.replace(old, new)

# --- row correspondence on (dataset_index, caption) + row count, not motion_id ---
old = """    mids = sink[0]["motion_ids"]
    for k, rec in enumerate(sink):                       # fail loud on any row misalignment
        if rec["motion_ids"] != mids:
            raise SystemExit(f"[mm] repeat {k} motion_id order differs from repeat 0 — "
                             f"rows would not correspond; refusing to compute MM")
    act = torch.stack([rec["gen_emb"] for rec in sink], dim=1).numpy()   # [N, R, D]"""
new = """    # MM is only meaningful if row i is the SAME text in every repeat. motion_id can repeat
    # across caption rows, so key on (dataset_index, caption) and also check the row count.
    keys0 = [tuple(k) for k in sink[0]["row_keys"]]
    mids = sink[0]["motion_ids"]
    if len(keys0) != int(sink[0]["gen_emb"].shape[0]):
        raise SystemExit(f"[mm] repeat 0: {len(keys0)} row_keys vs {sink[0]['gen_emb'].shape[0]} "
                         f"embedding rows — refusing to compute MM")
    for k, rec in enumerate(sink):                       # fail loud on ANY row misalignment
        keys = [tuple(x) for x in rec["row_keys"]]
        if keys != keys0 or int(rec["gen_emb"].shape[0]) != len(keys0):
            raise SystemExit(f"[mm] repeat {k} row keys/count differ from repeat 0 — "
                             f"rows would not correspond; refusing to compute MM")
    act = torch.stack([rec["gen_emb"] for rec in sink], dim=1).numpy()   # [N, R, D]"""
assert s.count(old) == 1, "rowcheck anchor not unique"
s = s.replace(old, new)

# --- record the sample + label it honestly ---
old = """    report = {
        "n_texts": int(act.shape[0]), "mm_num_repeats": int(act.shape[1]),
        "mm_times": args.mm_times, "steps": args.steps, "cfg_scale": args.cfg_scale,
        "flow_ckpt": args.flow_ckpt, "epoch": fck.get("epoch"),
        "multimodality": float(calculate_multimodality(act, args.mm_times)),
        "per_subset": {},
    }"""
new = """    report = {
        "n_texts": int(act.shape[0]), "mm_num_repeats": int(act.shape[1]),
        "mm_times": args.mm_times, "steps": args.steps, "cfg_scale": args.cfg_scale,
        "flow_ckpt": args.flow_ckpt, "epoch": fck.get("epoch"),
        # HONEST LABEL: this is a STRATIFIED-SUBSET MultiModality (equal texts per bucket via a
        # deterministic stride over the manifest), NOT natural-val-weighted over all 5150 clips.
        # Outcome-independent (so not cherry-picked) but manifest-order-sensitive.
        "sampling": "stratified-subset (deterministic stride, equal texts per bucket)",
        "mm_num_samples_per_bucket": args.mm_num_samples,
        "mm_idxs": [int(i) for i in mm_idxs],
        "row_keys": [[int(a), str(b)] for a, b in keys0],
        "multimodality": float(calculate_multimodality(act, args.mm_times)),
        "per_subset": {},
    }"""
assert s.count(old) == 1, "report anchor not unique"
s = s.replace(old, new)

# --- os import (used by the offline guard) ---
if "\nimport os\n" not in s:
    s = s.replace("import json\n", "import json\nimport os\n", 1)

open(p, "w").write(s)
print("patched", p)
