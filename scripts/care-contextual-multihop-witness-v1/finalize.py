#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

from common import OUTPUT, dump_json, file_record


def main() -> None:
    before = json.loads((OUTPUT / "baseline_asset_manifest_before.json").read_text(encoding="utf-8"))["assets"]
    after = [file_record(Path(row["absolute_path"]), row["purpose"]) for row in before]
    dump_json(OUTPUT / "baseline_asset_manifest_after.json", {"git_used": False, "assets": after})
    old = {row["absolute_path"]: row for row in before}; new = {row["absolute_path"]: row for row in after}
    changed = [{"path": path, "before": old.get(path), "after": new.get(path)} for path in sorted(set(old) | set(new)) if old.get(path) != new.get(path)]
    status = "PASS" if not changed else "FAIL"
    dump_json(OUTPUT / "baseline_integrity_check.json", {"status": status, "asset_count": len(after), "changed": changed})
    print("CMW_BASELINE_INTEGRITY_" + status, flush=True)
    if changed:
        raise RuntimeError("formal baseline assets changed")


if __name__ == "__main__":
    main()
