#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

from common import OUTPUT, dump_json, file_record


def main():
    before = json.loads((OUTPUT / "diagnostic_asset_manifest_before.json").read_text(encoding="utf-8"))
    after = [file_record(Path(row["path"]), row["purpose"]) for row in before["assets"]]
    changed = [a["path"] for a, b in zip(after, before["assets"]) if a["sha256"] != b["sha256"]]
    dump_json(OUTPUT / "diagnostic_asset_manifest_after.json", {"status": "CAPTURED", "assets": after})
    dump_json(OUTPUT / "diagnostic_integrity_check.json", {"status": "PASS" if not changed else "FAIL", "changed": changed})
    if changed: raise RuntimeError("BLOCKED_BY_CONTRACT: frozen assets changed: " + repr(changed))
    print(f"CCBC_DIAGNOSTIC_INTEGRITY_PASS assets={len(after)}")


if __name__ == "__main__": main()
