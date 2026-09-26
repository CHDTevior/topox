#!/usr/bin/env python3
"""Copy the allowlisted paper files into a private snapshot directory, refusing anything that is not a plain
file reached without following a symlink.

Every path component below SRC is opened with O_NOFOLLOW relative to the previous one, the open descriptor is
fstat'ed (regular file, link count 1) and the copy reads from THAT descriptor, so no rename or symlink swap
between check and copy can publish a file from outside the paper directory (codex 2026-09-12 r5 P1 #1/#2).
Usage: _sync_snapshot.py SRC DST FILE...   (exit 1 with a message on the first refusal)
"""
import errno
import os
import shutil
import stat
import sys


def main() -> int:
    src, dst, files = sys.argv[1], sys.argv[2], sys.argv[3:]
    root = os.open(src, os.O_RDONLY | os.O_DIRECTORY)  # SRC itself is the configured root; a stable symlink above it is fine
    root_path = os.readlink(f"/proc/self/fd/{root}")
    for f in files:
        parts = f.split("/")
        fd = root
        try:
            for i, part in enumerate(parts):
                last = i == len(parts) - 1
                # O_NONBLOCK: a FIFO in the paper directory would otherwise block this open forever, holding the
                # sync lock (codex r12); it has no effect on a regular file, which fstat then requires
                flags = os.O_RDONLY | os.O_NOFOLLOW | (os.O_NONBLOCK if last else os.O_DIRECTORY)
                nfd = os.open(part, flags, dir_fd=fd)
                if fd != root:
                    os.close(fd)
                fd = nfd
        except OSError as e:
            why = {errno.ELOOP: "a symlink in its path", errno.ENOENT: "missing (a partial paper is never pushed)",
                   errno.ENOTDIR: "a path component is not a directory"}.get(e.errno, e.strerror)
            print(f"[sync] {f}: {why} -- refusing", file=sys.stderr)
            return 1
        st = os.fstat(fd)
        # the descriptor is pinned to one inode; make sure that inode still sits inside SRC (a directory renamed
        # away between the opens would otherwise let a file from elsewhere through -- codex r7 #1)
        where = os.readlink(f"/proc/self/fd/{fd}")
        if not where.startswith(root_path + "/"):
            print(f"[sync] {f} resolved to {where}, outside {root_path} -- refusing", file=sys.stderr)
            return 1
        if not stat.S_ISREG(st.st_mode):
            print(f"[sync] {f} is not a regular file -- refusing", file=sys.stderr)
            return 1
        if st.st_nlink != 1:
            print(f"[sync] {f} has {st.st_nlink} hard links -- refusing", file=sys.stderr)
            return 1
        out = os.path.join(dst, f)
        os.makedirs(os.path.dirname(out), exist_ok=True)
        with os.fdopen(fd, "rb") as r, open(out, "wb") as w:
            shutil.copyfileobj(r, w)
        os.utime(out, ns=(st.st_atime_ns, st.st_mtime_ns))
    return 0


if __name__ == "__main__":
    sys.exit(main())
