#!/usr/bin/env python3
"""Turn the teleport scan into a pinned exclusion artifact at a chosen threshold K."""
import json, sys, hashlib, collections
from pathlib import Path
import numpy as np

SCAN = Path("configs/pzh312_teleport_local_scan.json")
K = float(sys.argv[1])
MODE = sys.argv[2] if len(sys.argv) > 2 else "clip"        # "clip" | "frame"
OUT = Path(sys.argv[3] if len(sys.argv) > 3 else f"configs/pzh312_teleport_exclusions_K{K:g}_{MODE}.json")
scan = json.loads(SCAN.read_text())
if str(K) not in scan["clip_hits"] and f"{K:g}" not in scan["clip_hits"]:
    raise SystemExit(f"scan has no K={K}; available {list(scan['clip_hits'])}. "
                     f"Frame-level exclusion needs a rescan that records frame indices.")
hits = scan["clip_hits"].get(str(K)) or scan["clip_hits"][f"{K:g}"]
if MODE == "frame":
    raise SystemExit("frame mode needs per-frame indices; rerun the scan with --record-frames")
rep = {"criterion": scan["criterion"], "K": K, "floor": scan["floor"], "mode": MODE,
       "validated_on": scan["validated_on"],
       "scan_sha256": hashlib.sha256(SCAN.read_bytes()).hexdigest(),
       "n_clips": len(hits), "total_clips": scan["total_clips"],
       "frac_clips": len(hits) / scan["total_clips"],
       "clips": {c: "all" for c in sorted(hits)}}
OUT.write_text(json.dumps(rep, indent=1))
print(f"[OK] {OUT}: {len(hits)} clips ({100*rep['frac_clips']:.3f}%) excluded at K={K:g}")
print(f"     sha {hashlib.sha256(OUT.read_bytes()).hexdigest()[:16]}")
