#!/usr/bin/env python3
from __future__ import annotations

import json

from common import OUTPUT, discover_assets, dump, formal_rows, load_apk_map, read_csv


def main():
    assets = discover_assets()
    formal = formal_rows(assets["formal"])
    assignments = read_csv(assets["group_assignments"])
    if len(assignments) != 799:
        raise RuntimeError(f"BLOCKED_BY_CONTRACT: group assignments rows={len(assignments)}")
    sizes = {}
    for row in assignments:
        sizes[int(row["group"])] = int(row["group_size"])
    multi = {row["sha256"].strip().casefold() for row in assignments if sizes[int(row["group"])] > 1}
    routed = {row["sha256"] for row in formal if row["routed"]}
    wanted = multi | routed
    apk_map, audit = load_apk_map(wanted)
    largest = max(sizes.values())
    largest_group = next(group for group, size in sizes.items() if size == largest)
    largest_ids = {row["sha256"].strip().casefold() for row in assignments if int(row["group"]) == largest_group}
    graph_dir = assets["full_graph_dir"]
    graph_coverage = sum((graph_dir / f"{sid}.json.gz").is_file() for sid in routed)
    record = {
        "status": "PASS" if largest_ids <= set(apk_map) and len(apk_map) == len(wanted) and graph_coverage == len(routed) else "BLOCKED",
        "source_rows": len(formal), "multi_group_samples": len(multi), "routed_samples": len(routed),
        "wanted_union": len(wanted), "apk_paths_resolved": len(apk_map), "largest_group": largest,
        "largest_group_apks_resolved": len(largest_ids & apk_map.keys()), "routed_full_graphs": graph_coverage,
        "apk_resolution_audit": audit, "future_rows_loaded": 0,
        "labels_used_for_fingerprint_extraction": False,
    }
    dump(OUTPUT / "preflight.json", record)
    if record["status"] != "PASS":
        raise RuntimeError("BLOCKED_BY_CONTRACT: lineage/Host audit inputs incomplete: " + json.dumps(record, sort_keys=True))
    print("CARE_P31_LINEAGE_HOST_AUDIT_PREFLIGHT_PASS", flush=True)
    print(json.dumps({key: record[key] for key in ("source_rows", "multi_group_samples", "apk_paths_resolved", "largest_group", "routed_full_graphs")}), flush=True)


if __name__ == "__main__":
    main()
