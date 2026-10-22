#!/usr/bin/env python3
from __future__ import annotations

from common import discover_assets, endpoint_hosts, load_apk_map, load_endpoint_rows, svm_route_rows


def main() -> None:
    assets = discover_assets()
    source = svm_route_rows(assets["source_member"])
    routed = [row for row in source if row["routed"]]
    wanted = {row["sha256"] for row in routed}
    apk_map, audit = load_apk_map(wanted)
    endpoints = load_endpoint_rows(assets["endpoint"], wanted)
    if len(routed) != 200 or len(apk_map) != 200 or len(endpoints) != 200:
        raise RuntimeError(f"BLOCKED_BY_CONTRACT: route={len(routed)} apk={len(apk_map)} endpoint={len(endpoints)}")
    zero_host = sum(not endpoint_hosts(endpoints[row["sha256"]]) for row in routed)
    import androguard, yaml  # noqa: F401
    print("SENSITIVE_GUARD_CONTEXT_PREFLIGHT_PASS")
    print({"source": len(source), "routed": len(routed), "apk": len(apk_map), "endpoint": len(endpoints),
           "zero_host_routed_apks": zero_host, "labels_loaded": False,
           "future_rows_loaded": 0, "apk_path_audit": audit})


if __name__ == "__main__":
    main()
