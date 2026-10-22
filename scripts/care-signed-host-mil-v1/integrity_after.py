#!/usr/bin/env python3
from pathlib import Path

from common import OUTPUT, asset_record, dump_json, read_json


def main():
    before = read_json(OUTPUT / "baseline_asset_manifest_before.json"); after = []; missing = []
    for record in before:
        path = Path(record["absolute_path"])
        if path.is_file(): after.append(asset_record(path, record["purpose"]))
        else: missing.append(str(path))
    dump_json(OUTPUT / "baseline_asset_manifest_after.json", after)
    lookup = {row["absolute_path"]: row for row in after}; changed = []
    for old in before:
        new = lookup.get(old["absolute_path"])
        if new is None or new["sha256"] != old["sha256"] or new["size_bytes"] != old["size_bytes"]:
            changed.append({"before": old, "after": new})
    status = "PASS" if not changed and not missing else "FAIL"
    dump_json(OUTPUT / "baseline_integrity_check.json", {"status": status, "asset_count": len(before),
        "changed": changed, "missing": missing})
    decision = read_json(OUTPUT / "final_decision.json"); decision["formal_integrity_pending"] = False
    decision["formal_integrity"] = status; dump_json(OUTPUT / "final_decision.json", decision)
    if status != "PASS": raise RuntimeError("BLOCKED_BY_CONTRACT: formal assets changed")
    print("SIGNED_HOST_MIL_BASELINE_INTEGRITY_PASS", flush=True)


if __name__ == "__main__": main()

