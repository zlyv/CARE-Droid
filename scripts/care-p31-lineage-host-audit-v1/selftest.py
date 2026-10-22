#!/usr/bin/env python3
from __future__ import annotations

from common import BIT_BYTES, sketch, sketch_jaccard


def main():
    a = sketch(["a", "b", "c"])
    b = sketch(["a", "b", "c"])
    c = sketch(["x", "y", "z"])
    assert len(__import__("base64").b64decode(a["bits_b64"])) == BIT_BYTES
    assert sketch_jaccard(a, b) == 1.0
    assert 0.0 <= sketch_jaccard(a, c) < 1.0
    print("CARE_P31_LINEAGE_HOST_AUDIT_SELFTEST_PASS", flush=True)


if __name__ == "__main__":
    main()
