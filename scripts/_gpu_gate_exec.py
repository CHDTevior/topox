"""Take one GPU by its UUID, prove nothing else is on it, hold it, and exec a script on it.

Written after three rounds of the same review on a shell version (codex 2026-09-10): in bash the census kept
failing open in ways that had nothing to do with GPUs -- a here-string whose temporary file could not be written
skipped the validation loop, `mapfile` did not propagate its producer's exit status, and an `exec` redirection
onto an inherited descriptor locked the wrong file. Here every command's exit status is checked, every line of
every census is parsed before it is trusted, and the lock is an explicit descriptor made inheritable so the
work itself holds it.

usage:  python scripts/_gpu_gate_exec.py <gpu-index-within-this-step> <script.py> [args...]

exit 3 the census could not describe the card   4 the card is busy   5 the card is held by another launcher
"""
from __future__ import annotations
import fcntl, os, re, subprocess, sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
UUID = r"GPU-[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}"
GPU_ROW = re.compile(rf"^(\d+),\s*({UUID})$")
APP_ROW = re.compile(rf"^({UUID}),\s*(\d+),\s*(.+)$")


def refuse(code: int, msg: str) -> "NoReturn":                                 # noqa: F821
    print(f"[refuse] {os.uname().nodename}: {msg}", file=sys.stderr, flush=True)
    raise SystemExit(code)


def census(fields: str, code: int) -> list[str]:
    """One nvidia-smi query, or a refusal: a query that cannot run and a query that says nothing are different
    answers, and neither of them is 'the card is free'."""
    try:
        p = subprocess.run(["nvidia-smi", f"--query-{fields}", "--format=csv,noheader"],
                           capture_output=True, text=True)
    except OSError as e:
        refuse(code, f"nvidia-smi --query-{fields} could not be run ({e}): no census, no launch")
    if p.returncode != 0:
        refuse(code, f"nvidia-smi --query-{fields} exited {p.returncode}: no census, no launch\n{p.stderr.strip()}")
    return [ln for ln in p.stdout.splitlines() if ln.strip()]


def main() -> "NoReturn":                                                      # noqa: F821
    if len(sys.argv) < 3:
        refuse(2, f"usage: {sys.argv[0]} <gpu-index> <script.py> [args...]")
    want, script, args = sys.argv[1], sys.argv[2], sys.argv[3:]
    if not re.fullmatch(r"\d+", want):
        refuse(2, f"the GPU index must be a non-negative integer, got {want!r}")

    rows = census("gpu=index,uuid", 3)
    if not rows:
        refuse(3, "the GPU list is empty: no census, no launch")
    seen: list[tuple[str, str]] = []
    for ln in rows:
        m = GPU_ROW.match(ln.strip())
        if not m:
            refuse(3, f"the GPU list does not parse: {ln!r}")
        seen.append((m.group(1), m.group(2)))
    hits = sorted({u for i, u in seen if i == want})
    if len(hits) != 1:
        refuse(3, f"index {want} resolves to {len(hits)} device(s) in this step: {seen}")
    uuid = hits[0]

    # hold the card BEFORE looking at it, so a second launcher of mine cannot pass the same empty census;
    # the descriptor is inheritable, so after the exec the work itself is the holder for its whole run
    lock_dir = REPO / ".aris" / "meta"
    lock_dir.mkdir(parents=True, exist_ok=True)
    lock = lock_dir / f".gpu_pin.{os.uname().nodename}.{uuid}"
    try:
        fd = os.open(lock, os.O_CREAT | os.O_RDWR, 0o644)
    except OSError as e:
        refuse(5, f"cannot open the pin lock {lock}: {e}")
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        refuse(5, f"another launcher of mine already holds {uuid} ({lock})")
    os.set_inheritable(fd, True)

    apps = census("compute-apps=gpu_uuid,pid,used_memory", 4)   # an empty list here IS the good case
    busy = []
    for ln in apps:
        m = APP_ROW.match(ln.strip())
        if not m:
            refuse(4, f"a process row does not say which device it is on: {ln!r}")
        if m.group(1) == uuid:
            busy.append((m.group(2), m.group(3)))
    if busy:
        refuse(4, f"{uuid} already has {len(busy)} compute process(es), not pinning onto it: {busy}")
    print(f"[gate] {os.uname().nodename} index {want} = {uuid}, held and free", flush=True)

    os.chdir(REPO)
    os.execve(sys.executable, [sys.executable, script, *args], {**os.environ, "CUDA_VISIBLE_DEVICES": uuid})


if __name__ == "__main__":
    main()
