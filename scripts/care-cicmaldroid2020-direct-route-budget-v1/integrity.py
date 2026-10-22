#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

from common import OUTPUT, dump_json, file_record, read_json


def main():
    before = read_json(OUTPUT / "baseline_asset_manifest_before.json")
    after = []
    missing = []
    for record in before:
        path = Path(record["absolute_path"])
        if not path.is_file():
            missing.append(str(path))
            continue
        after.append(file_record(path, record["purpose"]))
    dump_json(OUTPUT / "baseline_asset_manifest_after.json", after)
    after_by_path = {row["absolute_path"]: row for row in after}
    changed = []
    for old in before:
        new = after_by_path.get(old["absolute_path"])
        if new is None or new["sha256"] != old["sha256"] or new["size_bytes"] != old["size_bytes"]:
            changed.append({"before": old, "after": new})
    status = "PASS" if not changed and not missing else "FAIL"
    result = {"status": status, "asset_count": len(before), "changed": changed, "missing": missing}
    dump_json(OUTPUT / "baseline_integrity_check.json", result)
    if status != "PASS":
        raise RuntimeError("BLOCKED_BY_CONTRACT: frozen formal assets changed")
    print(f"CICMALDROID2020_BASELINE_INTEGRITY_PASS assets={len(before)}", flush=True)


if __name__ == "__main__":
    main()
