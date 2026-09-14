"""sha256 of every motion file of a KTJD-17 library (for duplicate-aware held-out splits).
  srun --jobid=<alloc> --overlap --gres=gpu:0 --cpus-per-task=16 --mem=32G -N1 -n1 /usr/bin/env python scripts/_hash_motions_v2.py <ktjd_root> <out.json> [workers]
Writes {clip_id: sha256} over manifests/clips.jsonl; read-only apart from the output file."""
import hashlib, json, os, sys
from concurrent.futures import ProcessPoolExecutor
root, out = sys.argv[1], sys.argv[2]; workers = int(sys.argv[3]) if len(sys.argv) > 3 else 16
import socket
if not os.environ.get("SLURM_JOB_ID") or os.environ.get("SLURMD_NODENAME", "") != socket.gethostname().split(".")[0]:
    raise SystemExit("[refuse] run this inside a Slurm step on the allocation's compute node (srun --jobid=<alloc> --overlap "
                     "--gres=gpu:0 --cpus-per-task=16 ...): hashing the whole corpus is heavy work and never runs on a login node "
                     "(a job id alone is not proof -- the allocation-owning shell may be a login shell)")
recs = [json.loads(l) for l in open(os.path.join(root, "manifests/clips.jsonl"))]
def h(rec):
    p = os.path.join(root, rec["motion_relpath"]); s = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""): s.update(b)
    return rec["clip_id"], s.hexdigest()
res = {}
with ProcessPoolExecutor(workers) as ex:
    for i, (cid, d) in enumerate(ex.map(h, recs, chunksize=64), 1):
        res[cid] = d
        if i % 10000 == 0: print(i, flush=True)
json.dump(res, open(out, "w")); print("done", len(res), "clips ->", out)
