"""Preflight: prove the run directory can actually take a full-size checkpoint write.

WHY (codex v2b r3 MAJOR-2): the trainer skips checkpoint saving entirely under SMOKE, so a
green smoke says nothing about whether a ~3.7GB save will succeed. On this cluster the
first real save lands 10 epochs in (~7h of 4xH100); discovering "no space" / "read-only" /
"quota" / "torch.save dies on this fs" at that point costs the whole window.

This exercises the same sequence the trainer uses — write a temp file in the TARGET
directory, fsync it, os.replace onto the final name, fsync the directory, then read the
result back — using a payload of the same order of magnitude as the real checkpoint, and
cleans up afterwards. It does NOT touch any existing file: the probe name is unique and
removed at the end.

Usage:
  python scripts/_preflight_ckpt_write.py --out_dir runs/<branch> --like <ref_ckpt>
    --like: an existing checkpoint whose byte size sets the probe size (the real save is
            about this big). Falls back to --bytes if not given.
"""
import argparse
import os
import shutil
import sys
import tempfile
import time


def human(n: int) -> str:
    return f"{n / (1 << 30):.2f} GiB"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--like", default=None,
                    help="reference ckpt; its size is used as the probe payload size")
    ap.add_argument("--bytes", type=int, default=3_700_000_000)
    a = ap.parse_args()

    out = os.path.abspath(a.out_dir)
    os.makedirs(out, exist_ok=True)
    size = os.path.getsize(a.like) if a.like else a.bytes
    st = os.statvfs(out)
    free = st.f_bavail * st.f_frsize
    print(f"[preflight-ckpt] dir={out}")
    print(f"[preflight-ckpt] probe size={human(size)}  free={human(free)}")
    # The trainer keeps last + best + periodic snapshots; require headroom for 3 of them
    # so a successful probe does not immediately precede an out-of-space real save.
    need = size * 3
    if free < need:
        print(f"[preflight-ckpt] FAIL: need >= {human(need)} free (3x ckpt) but have "
              f"{human(free)}")
        return 1

    final = os.path.join(out, ".preflight_ckpt_probe.pt")
    fd, tmp = tempfile.mkstemp(dir=out, prefix=".preflight_", suffix=".pt")
    t0 = time.time()
    try:
        with os.fdopen(fd, "wb") as f:
            if a.like:
                with open(a.like, "rb") as src:
                    shutil.copyfileobj(src, f, length=16 << 20)
            else:
                chunk = b"\0" * (16 << 20)
                written = 0
                while written < size:
                    n = min(len(chunk), size - written)
                    f.write(chunk[:n])
                    written += n
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, final)
        dfd = os.open(out, os.O_DIRECTORY)
        try:
            os.fsync(dfd)
        finally:
            os.close(dfd)
        dt = time.time() - t0
        got = os.path.getsize(final)
        if got != size:
            print(f"[preflight-ckpt] FAIL: wrote {got} bytes, expected {size}")
            return 1
        # Read back the head and tail to prove the file is actually readable after replace.
        with open(final, "rb") as f:
            head = f.read(1 << 20)
            f.seek(-(1 << 20), os.SEEK_END)
            tail = f.read(1 << 20)
        if len(head) != (1 << 20) or len(tail) != (1 << 20):
            print("[preflight-ckpt] FAIL: readback short")
            return 1
        print(f"[preflight-ckpt] OK: wrote+fsync+replace+readback {human(got)} in {dt:.1f}s "
              f"({got / dt / (1 << 20):.0f} MiB/s)")
        return 0
    finally:
        for p in (tmp, final):
            try:
                if os.path.exists(p):
                    os.unlink(p)
            except OSError:
                pass


if __name__ == "__main__":
    raise SystemExit(main())
