#!/usr/bin/env python3
from __future__ import annotations

import json

from common import OUTPUT, assets, dump_json, file_record


def main() -> None:
    found = assets(); after = [file_record(found[name], name) for name in ("formal", "endpoint", "attention")]
    dump_json(OUTPUT / "baseline_asset_manifest_after.json", {"assets": after})
    before = json.loads((OUTPUT / "baseline_asset_manifest_before.json").read_text(encoding="utf-8"))["assets"]
    left = {item["absolute_path"]: item for item in before}; right = {item["absolute_path"]: item for item in after}
    changed = [path for path in sorted(set(left) | set(right)) if path not in left or path not in right or left[path]["sha256"] != right[path]["sha256"]]
    status = "PASS" if not changed else "FAIL"
    dump_json(OUTPUT / "baseline_integrity_check.json", {"status": status, "asset_count": len(after), "changed": changed})
    print(f"SCS_BASELINE_INTEGRITY_{status} assets={len(after)}", flush=True)
    if changed:
        raise RuntimeError("BLOCKED_BY_CONTRACT: formal assets changed " + repr(changed))


if __name__ == "__main__":
    main()
