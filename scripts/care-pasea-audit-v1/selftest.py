#!/usr/bin/env python3
from __future__ import annotations

from common import OUTPUT, ensure_output, normalize_owner, sha256, truthy


def main() -> None:
    assert normalize_owner("FIRST_PARTY") == "APP"
    assert normalize_owner("third-party") == "TPL"
    assert normalize_owner("unresolved") == "UNKNOWN"
    assert truthy("true") and truthy(1) and not truthy("false")
    ensure_output()
    path = OUTPUT / "checkpoints/selftest.sha"
    path.write_bytes(b"abc")
    assert sha256(path) == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    path.unlink()
    print("PASEA_AUDIT_SELFTEST_PASS")


if __name__ == "__main__":
    main()

