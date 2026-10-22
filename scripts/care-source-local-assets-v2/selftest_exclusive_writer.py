#!/usr/bin/env python3
"""Prove that a second process cannot acquire the source-cache writer lock."""

import fcntl
import subprocess
import sys
import tempfile
from pathlib import Path


def main() -> None:
    with tempfile.TemporaryDirectory() as name:
        path = Path(name) / ".exclusive_writer.lock"
        with path.open("a+") as first:
            fcntl.flock(first.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            code = (
                "import fcntl,sys; "
                f"f=open({str(path)!r},'a+'); "
                "\ntry: fcntl.flock(f.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)"
                "\nexcept BlockingIOError: sys.exit(73)"
                "\nsys.exit(0)"
            )
            result = subprocess.run([sys.executable, "-c", code], check=False)
        if result.returncode != 73:
            raise RuntimeError(f"exclusive writer self-test failed: {result.returncode}")
    print("EXCLUSIVE_WRITER_SELFTEST_PASS")


if __name__ == "__main__":
    main()
