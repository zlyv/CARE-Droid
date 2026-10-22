#!/usr/bin/env python3
from __future__ import annotations

import importlib.metadata
import json
import sys

from common import OUTPUT, discover_assets, endpoint_hosts, ensure_output_boundary, formal_rows, load_apk_map, load_attention, load_endpoint_rows


def main() -> None:
    ensure_output_boundary()
    from androguard.core.apk import APK  # noqa: F401
    from androguard.core.dex import DEX  # noqa: F401
    import yaml  # noqa: F401

    assets = discover_assets()
    rows = formal_rows(assets["formal"])
    routed = [row for row in rows if row["routed"]]
    wanted = {row["sha256"] for row in routed}
    apk_map, apk_audit = load_apk_map(wanted)
    endpoints = load_endpoint_rows(assets["endpoint"], wanted)
    attention = load_attention(assets["attention"])
    if len(apk_map) != 200:
        raise RuntimeError(f"BLOCKED_BY_CONTRACT: routed APK paths {len(apk_map)}/200")
    if len(endpoints) != 200:
        raise RuntimeError(f"BLOCKED_BY_CONTRACT: endpoint rows {len(endpoints)}/200")
    if len(wanted & set(attention)) != 200:
        raise RuntimeError("BLOCKED_BY_CONTRACT: attention rows not 200")
    fold_counts = {fold: sum(row["fold"] == fold for row in routed) for fold in range(1, 6)}
    if list(fold_counts.values()) != [40] * 5:
        raise RuntimeError(f"route fold sizes changed: {fold_counts}")
    mapping_text = assets["mapping"].read_text(encoding="utf-8")
    if "mapping_id: \"tasa_droid_semantic_mapping_v1\"" not in mapping_text:
        raise RuntimeError("semantic mapping identity mismatch")
    result = {
        "status": "PASS", "python": sys.executable,
        "androguard_version": importlib.metadata.version("androguard"),
        "source_rows": len(rows), "routed_rows": len(routed), "fold_counts": fold_counts,
        "apk_paths": len(apk_map), "endpoint_rows": len(endpoints),
        "future_rows_loaded": 0, "labels_used": False,
        "host_rows": sum(len(endpoint_hosts(endpoints[sid])) for sid in wanted),
        "assets": {key: str(value) for key, value in assets.items()},
        "apk_path_audit": apk_audit,
        "output": str(OUTPUT),
    }
    print("DEX_TRANSPARENT_WRAPPER_PREFLIGHT_PASS")
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
