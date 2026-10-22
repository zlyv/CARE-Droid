#!/usr/bin/env python3
from __future__ import annotations

import json

from common import OUTPUT, SCS_OUTPUT, dump_json, file_record, formal_path


def main() -> None:
    paths = [(formal_path(), "formal_predictions"), (SCS_OUTPUT / "final_decision.json", "SCS_final_decision"),
             (SCS_OUTPUT / "coverage_gate.json", "SCS_coverage_gate"), (SCS_OUTPUT / "direction_probe.json", "SCS_direction_probe")]
    after = [file_record(path, purpose) for path, purpose in paths]; dump_json(OUTPUT / "baseline_asset_manifest_after.json", {"assets": after})
    before = json.loads((OUTPUT / "baseline_asset_manifest_before.json").read_text(encoding="utf-8"))["assets"]
    left = {row["absolute_path"]: row["sha256"] for row in before}; right = {row["absolute_path"]: row["sha256"] for row in after}
    changed = [path for path in sorted(set(left) | set(right)) if left.get(path) != right.get(path)]
    status = "PASS" if not changed else "FAIL"; dump_json(OUTPUT / "baseline_integrity_check.json", {"status": status, "asset_count": len(after), "changed": changed})
    print(f"FPR_BASELINE_INTEGRITY_{status} assets={len(after)}", flush=True)
    if changed:
        raise RuntimeError("BLOCKED_BY_CONTRACT: frozen assets changed")


if __name__ == "__main__":
    main()
