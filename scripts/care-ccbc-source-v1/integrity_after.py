#!/usr/bin/env python3
from __future__ import annotations

from common import OUTPUT, dump_json, file_record, read_json


def main() -> None:
    before = read_json(OUTPUT / "baseline_asset_manifest_before.json")
    after_assets = []
    changed = []; missing = []
    for row in before.get("assets", []):
        from pathlib import Path
        path = Path(row["path"])
        if not path.is_file():
            missing.append(str(path)); continue
        current = file_record(path, row.get("purpose", "formal asset"))
        after_assets.append(current)
        if current["sha256"] != row["sha256"] or current["bytes"] != row["bytes"]:
            changed.append({"path": str(path), "before": row, "after": current})
    after = {
        "status": "CAPTURED", "project": before.get("project"),
        "output": before.get("output"), "git_used": False, "assets": after_assets,
    }
    check = {
        "status": "PASS" if not changed and not missing else "FAIL",
        "checked": len(before.get("assets", [])), "changed": changed,
        "missing": missing, "formal_assets_unchanged": not changed and not missing,
    }
    dump_json(OUTPUT / "baseline_asset_manifest_after.json", after)
    dump_json(OUTPUT / "baseline_integrity_check.json", check)
    if check["status"] != "PASS":
        raise RuntimeError("BLOCKED_BY_CONTRACT: formal baseline assets changed or disappeared")
    print(f"CCBC_BASELINE_INTEGRITY_PASS assets={check['checked']}")


if __name__ == "__main__":
    main()
