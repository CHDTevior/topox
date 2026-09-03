"""Seal the unseen-topology protocol into a tracked, hash-verifiable bundle.

The pre-registration was self-hashing but lived entirely under `data/`, which is gitignored, so
nothing outside the working directory recorded what the protocol actually was. A freeze that only
exists next to the thing it constrains is not a freeze: an artifact and its own hash can be
rewritten together.

This writes `protocol/` — tracked by git — containing the small artifacts verbatim and a SHA-256
manifest for everything too large to commit. Committing that directory is what makes the
protocol external. Verification re-hashes the live files and refuses on any drift.

    python scripts/_seal_protocol.py --seal      # write protocol/, then git add + commit it
    python scripts/_seal_protocol.py --verify    # prove the live artifacts still match the seal
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

# Copied verbatim: small, and the exact content is what the protocol IS.
INLINE = [
    "data/holdout_topologies_v1.json",
    "data/holdout_splits_v1/splits_manifest.json",
    "data/holdout_eval_splits_v1/manifest_report.json",
    "data/holdout_eval_splits_v1/README.txt",
]
# Hashed only: too large to commit, but drift must still be detectable.
HASHED = [
    "data/holdout_splits_v1/train.txt",
    "data/holdout_splits_v1/val.txt",
    "data/holdout_splits_v1/held_representative.txt",
    "data/holdout_splits_v1/held_stress.txt",
    "data/holdout_eval_splits_v1/train_main_retained.json",
    "data/holdout_eval_splits_v1/val_all_retained.json",
    "data/holdout_eval_splits_v1/held_representative.json",
    "data/holdout_eval_splits_v1/held_stress.json",
    "data/joint_descriptions_v1.json",
    "data/joint_semantics_llm2vec_v1.npz",
    "data/joint_semantics_v1.npz",
    "data/animo4d_L4TB_plus_human_v4b272neutral/cond.npy",
    "data/animo4d_L4TB_plus_human_v4b272neutral/splits/train.txt",
    "data/animo4d_L4TB_plus_human_v4b272neutral/splits/val.txt",
]
# The code that produced them: a hash of the artifact is worthless if the builder can drift.
CODE = [
    "scripts/_build_holdout_trees.py",
    "scripts/_build_holdout_splits.py",
    "scripts/_build_holdout_eval_manifests.py",
    "scripts/_build_joint_descriptions.py",
    "scripts/_build_joint_semantic_embeddings.py",
    "scripts/_fit_restpose_moment_estimator.py",
    "scripts/_probe_joint_descriptions.py",
    "src/data/holdout_guard.py",
    "src/data/moment_source.py",
    "src/data/text_encoders.py",
    # The code that CONSUMES them. A seal covering only the builders certifies how the freeze was
    # produced and says nothing about what the run actually did with it: the guard could be sound
    # and the trainer could still enable augmentation, feed a different table, or pick different
    # moments. These are the files whose behaviour defines the experiment.
    "src/data/provenance.py",
    "src/data/anytop_dataset.py",
    "src/models/vq_model/graph_vq_tokenizer.py",
    "src/models/encoder.py",
    "src/models/CodeFlow_Model/token_dataset.py",
    "scripts/train_graph_vqvae.py",
    "scripts/export_graph_vq_tokens.py",
    "scripts/merge_export_shards.py",
    "scripts/_launch_holdout_vqvae.sh",
    # The orchestrator and the controller. Both can change what runs: one decides the world size
    # and therefore the per-rank batch, the other decides when and from which checkpoint the run
    # resumes. Leaving them outside the seal would leave the two files that operate the experiment
    # free to drift while everything they operate stays frozen.
    "scripts/_launch_holdout_vqvae_8card.sh",
    "scripts/_watchdog_holdout_vqvae.sh",
    # The suite is what turns "these guards exist" into "these guards were shown to fire".
    "scripts/_regression_suite.py",
]


def sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for c in iter(lambda: fh.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def collect() -> dict:
    def entry(rel):
        p = Path(rel)
        if not p.exists():
            return {"missing": True}
        return {"sha256": sha(p), "bytes": p.stat().st_size}
    try:
        git = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                             text=True).stdout.strip() or None
    except Exception:
        git = None
    return {"protocol": "unseen_topology_v1", "git_sha_at_seal": git,
            "inline": {r: entry(r) for r in INLINE},
            "hashed": {r: entry(r) for r in HASHED},
            "code": {r: entry(r) for r in CODE}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="protocol")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--seal", action="store_true")
    g.add_argument("--verify", action="store_true")
    ap.add_argument("--strict-code", action="store_true", dest="strict_code",
                    help="treat drift in the BUILDER CODE as fatal too. The launcher passes this: "
                         "at launch the code must be final. During development the code is the "
                         "thing being finalised, so its drift is reported, not fatal — while the "
                         "DATA artifacts must never drift from the moment they are frozen.")
    a = ap.parse_args()
    d = Path(a.dir)
    man = d / "SEAL.json"

    cur = collect()
    missing = [k for sec in ("inline", "hashed", "code")
               for k, v in cur[sec].items() if v.get("missing")]
    if missing:
        raise SystemExit(f"REFUSED: {len(missing)} artifact(s) missing: {missing[:5]}")

    if a.seal:
        if man.exists():
            raise SystemExit(f"REFUSED: {man} exists. A sealed protocol is not re-sealed by "
                             f"habit — delete it deliberately if the protocol genuinely changed, "
                             f"and say so in the paper.")
        d.mkdir(parents=True, exist_ok=True)
        for rel in INLINE:
            dst = d / Path(rel).name
            shutil.copy2(rel, dst)
        man.write_text(json.dumps(cur, indent=2))
        n = sum(len(cur[s]) for s in ("inline", "hashed", "code"))
        print(f"[seal] {n} artifacts sealed -> {d}/")
        print(f"[seal] inline copies: {[Path(r).name for r in INLINE]}")
        print(f"[seal] NOT yet committed. Make the protocol external with:")
        print(f"[seal]   git add {d} .gitignore {' '.join(CODE)} && \\")
        print(f"[seal]   git commit -m 'protocol: seal unseen_topology_v1 pre-registration'")
        return 0

    ref = json.loads(man.read_text())
    data_bad, code_bad = [], []
    for sec in ("inline", "hashed", "code"):
        for rel, want in ref[sec].items():
            got = cur[sec].get(rel, {})
            if got.get("sha256") != want.get("sha256"):
                (code_bad if sec == "code" else data_bad).append(
                    (sec, rel, want.get("sha256", "")[:12], got.get("sha256", "")[:12]))

    # DATA drift is always fatal: those artifacts define the pre-registration, and once frozen
    # they must never move. CODE drift means the protocol has not been sealed for launch yet,
    # which is the normal state while the code is still being finalised — so it is fatal only
    # under --strict-code, which is what the launcher passes.
    if data_bad:
        print(f"[seal] *** {len(data_bad)} DATA artifact(s) DRIFTED from the seal ***")
        for sec, rel, w, g_ in data_bad:
            print(f"  [{sec}] {rel}\n      sealed {w}...  now {g_}...")
        raise SystemExit(1)
    if code_bad:
        print(f"[seal] {len(code_bad)} builder file(s) changed since the seal — the protocol is "
              f"NOT sealed for launch:")
        for sec, rel, w, g_ in code_bad:
            print(f"  [{sec}] {rel}\n      sealed {w}...  now {g_}...")
        if a.strict_code:
            print("[seal] --strict-code: re-seal before launching (delete SEAL.json "
                  "deliberately, then --seal).")
            raise SystemExit(1)
        print("[seal] data artifacts are intact; expected while the code is being finalised.")
        return 0
    n = sum(len(ref[s]) for s in ("inline", "hashed", "code"))
    print(f"[seal] all {n} sealed artifacts match")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
