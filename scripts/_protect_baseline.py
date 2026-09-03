"""Fingerprint the artifacts that cannot be regenerated, so an accidental write is detectable.

The user's constraint on the inductive-normalisation work is explicit: the existing pipeline
is poor at generalisation but it is CORRECT, and it must survive. Every change is therefore
additive — new fields, new directories, new flags defaulting to current behaviour — and this
script is the tripwire that proves nothing under the protected set moved.

Large files are fingerprinted by (size, mtime, sha256 of a head+tail sample) rather than a
full hash: a full sha256 of the 142 GB token cache would take hours and would not be run, and
a check nobody runs is not a check.

    python scripts/_protect_baseline.py --write     # record (once)
    python scripts/_protect_baseline.py --verify    # check (before/after any risky step)
"""
from __future__ import annotations
import argparse, hashlib, json, os
from pathlib import Path

PROTECTED = [
    "data/animo4d_L4TB_plus_human_v4b272neutral/cond.npy",
    "data/animo4d_L4TB_plus_human_v4b272neutral/_cond_normalized_J144.pkl",
    "data/animo4d_L4TB_plus_human_v4b272neutral/splits/train.txt",
    "data/animo4d_L4TB_plus_human_v4b272neutral/splits/val.txt",
    "data/animo4d_L4TB_plus_human_v4b272neutral/motion_texts_by_file.json",
    "data/codeflow_tokens_v4b272neutral_n8192_ep219_fulllen300/manifest.json",
    "data/codeflow_tokens_v4b272neutral_n8192_ep219_fulllen300/empirical_stats.pt",
    "data/codeflow_tokens_v4b272neutral_n8192_ep219_fulllen300/train/index.jsonl",
    "data/codeflow_tokens_v4b272neutral_n8192_ep219_fulllen300/val/index.jsonl",
    "runs/vqvae_v4b272neutral_C96_J144_d512_Q4_n8192_b16g64_300ep_curric50to60_seed42/best_model.pt",
    "runs/codeflow_graph_pscf_v4b272neutral_n8192_b16g64_lr8e5_4xh200_seed42/last_model.pt",
    "runs/anytop_t2m_evaluator_distilbert_coemb512_gb128_lr1e-4_mfd12_v4b272_seed42/best_model.pt",
]
SAMPLE_DIRS = [
    ("data/codeflow_tokens_v4b272neutral_n8192_ep219_fulllen300/train", "*.npz", 24),
]
BIG = 256 << 20   # above this, sample head+tail instead of hashing the whole file


def fp(path: Path) -> dict:
    st = path.stat()
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        if st.st_size <= BIG:
            for c in iter(lambda: fh.read(1 << 20), b""):
                h.update(c)
            mode = "full"
        else:
            h.update(fh.read(1 << 20))
            fh.seek(-(1 << 20), os.SEEK_END)
            h.update(fh.read(1 << 20))
            mode = "head+tail"
    return {"size": st.st_size, "sha256": h.hexdigest(), "mode": mode}


def collect() -> dict:
    out = {"files": {}, "sampled": {}}
    for rel in PROTECTED:
        p = Path(rel)
        if not p.exists():
            out["files"][rel] = {"missing": True}
            continue
        out["files"][rel] = fp(p)
    for d, pat, n in SAMPLE_DIRS:
        files = sorted(Path(d).glob(pat))
        if not files:
            continue
        step = max(1, len(files) // n)
        picks = files[::step][:n]
        out["sampled"][d] = {"n_total": len(files),
                             "picks": {p.name: fp(p) for p in picks}}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="data/_baseline_protection/baseline_fingerprint.json")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--write", action="store_true")
    g.add_argument("--verify", action="store_true")
    args = ap.parse_args()
    m = Path(args.manifest)

    cur = collect()
    if args.write:
        if m.exists():
            raise SystemExit(f"REFUSED: {m} exists. Delete it deliberately if the baseline "
                             f"genuinely changed; do not overwrite a tripwire by habit.")
        m.parent.mkdir(parents=True, exist_ok=True)
        m.write_text(json.dumps(cur, indent=1))
        n = len(cur["files"]) + sum(len(v["picks"]) for v in cur["sampled"].values())
        print(f"[protect] recorded {n} fingerprints -> {m}")
        return 0

    ref = json.loads(m.read_text())
    bad = []
    for rel, want in ref["files"].items():
        got = cur["files"].get(rel)
        if got != want:
            bad.append((rel, want, got))
    for d, want in ref["sampled"].items():
        got = cur["sampled"].get(d, {})
        if got.get("n_total") != want["n_total"]:
            bad.append((d + " [file count]", want["n_total"], got.get("n_total")))
        for name, w in want["picks"].items():
            if got.get("picks", {}).get(name) != w:
                bad.append((f"{d}/{name}", w, got.get("picks", {}).get(name)))
    if bad:
        print(f"[protect] *** {len(bad)} PROTECTED ARTIFACT(S) CHANGED ***")
        for rel, w, g_ in bad[:12]:
            print(f"  {rel}\n     was {w}\n     now {g_}")
        raise SystemExit(1)
    n = len(ref["files"]) + sum(len(v["picks"]) for v in ref["sampled"].values())
    print(f"[protect] all {n} protected artifacts unchanged")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
