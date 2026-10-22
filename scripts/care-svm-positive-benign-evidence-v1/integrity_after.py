#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

from common import OUTPUT, asset_record, dump_json, read_json


def main():
    before = read_json(OUTPUT / "baseline_asset_manifest_before.json"); after = []; changed = []
    for old in before:
        current = asset_record(Path(old["absolute_path"]), old["purpose"]); after.append(current)
        if current["sha256"] != old["sha256"] or current["size_bytes"] != old["size_bytes"]:
            changed.append({"path": old["absolute_path"], "before": old["sha256"], "after": current["sha256"]})
    dump_json(OUTPUT / "baseline_asset_manifest_after.json", after)
    dump_json(OUTPUT / "baseline_integrity_check.json", {"status": "PASS" if not changed else "FAIL",
        "asset_count": len(after), "changed": changed})
    if changed:
        raise RuntimeError("BLOCKED_BY_CONTRACT: baseline asset changed")
    print(f"SVM_POSITIVE_BENIGN_BASELINE_INTEGRITY_PASS assets={len(after)}")


if __name__ == "__main__":
    main()

