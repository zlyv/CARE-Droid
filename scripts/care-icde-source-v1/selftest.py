#!/usr/bin/env python3
from __future__ import annotations

import tempfile
import zipfile
from pathlib import Path

from audit_stage_a import parse_adb_devices
from icde_common import norm_sha, sha256_file


def main() -> None:
    assert norm_sha("A" * 64) == "a" * 64
    assert norm_sha("bad") == ""
    parsed = parse_adb_devices({
        "returncode": 0,
        "stdout": "List of devices attached\nemulator-5554 device product:sdk\nABC123 device product:phone\nemulator-5556 offline\n",
    })
    assert parsed["emulators"] == ["emulator-5554"]
    assert parsed["physical_or_other"] == ["ABC123"]
    assert parsed["offline"] == ["emulator-5556"]
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "sample.apk"
        with zipfile.ZipFile(path, "w") as archive:
            archive.writestr("AndroidManifest.xml", b"binary")
            archive.writestr("classes.dex", b"dex")
        assert len(sha256_file(path)) == 64
    print("ICDE_STAGE_A_SELFTEST_PASS")


if __name__ == "__main__":
    main()

