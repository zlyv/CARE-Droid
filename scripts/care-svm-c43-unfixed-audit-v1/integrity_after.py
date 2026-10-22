#!/usr/bin/env python3
from common import OUTPUT, SOURCE, asset_record, dump_json, read_json


def main():
    before = read_json(OUTPUT / "baseline_asset_manifest_before.json")
    after = [asset_record(SOURCE, "frozen SVM/C43 source OOF predictions")]
    changed = []
    old = {row["absolute_path"]: row for row in before}
    for row in after:
        previous = old.get(row["absolute_path"])
        if previous is None or previous["sha256"] != row["sha256"] or previous["size_bytes"] != row["size_bytes"]:
            changed.append(row["absolute_path"])
    status = "PASS" if not changed else "FAIL"
    dump_json(OUTPUT / "baseline_asset_manifest_after.json", after)
    dump_json(OUTPUT / "baseline_integrity_check.json", {"status": status, "asset_count": len(after), "changed": changed})
    decision = read_json(OUTPUT / "final_decision.json")
    decision["baseline_integrity"] = status
    dump_json(OUTPUT / "final_decision.json", decision)
    if changed: raise RuntimeError("BLOCKED_BY_CONTRACT: formal source prediction asset changed")
    print("SVM_C43_UNFIXED_BASELINE_INTEGRITY_PASS", flush=True)


if __name__ == "__main__": main()
