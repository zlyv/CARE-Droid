#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

from common import OUTPUT, dump_json, read_json


def require(path: Path, label: str) -> None:
    if not path.is_file() or not path.stat().st_size:
        raise RuntimeError(f"BLOCKED_BY_CONTRACT: missing/empty {label}: {path}")


def main() -> None:
    required = {
        "asset_manifest_before": OUTPUT / "baseline_asset_manifest_before.json",
        "baseline_reproduction": OUTPUT / "baseline_reproduction.json",
        "contracts": OUTPUT / "cache/contracts_2017.jsonl.gz",
        "component_scope": OUTPUT / "cache/component_scope_2017.jsonl.gz",
        "formal_attention": OUTPUT / "cache/formal_attention.npz",
        "mapping_audit": OUTPUT / "component_mapping_audit.json",
    }
    for label, path in required.items():
        require(path, label)
    baseline = read_json(required["baseline_reproduction"])
    mapping = read_json(required["mapping_audit"])
    if baseline.get("status") != "PASS":
        raise RuntimeError("BLOCKED_BY_CONTRACT: formal baseline has not reproduced")
    if int(mapping.get("source_rows", 0)) != 799 or int(mapping.get("future_rows_loaded", -1)) != 0:
        raise RuntimeError("BLOCKED_BY_CONTRACT: component scope lineage mismatch")
    status = {
        "status": "PASS", "source_rows": 799, "future_rows_loaded": 0,
        "baseline": "P31-RF/RF-DI25/A0-C43-RQ", "max_delta": 2.25,
        "component_entry_mapping_ratio": mapping.get("entry_mapping_ratio"),
        "host_assignment_ratio": mapping.get("host_assignment_ratio"),
    }
    dump_json(OUTPUT / "preflight.json", status)
    print("CCBC_PREFLIGHT_PASS")
    print(status)


if __name__ == "__main__":
    main()
