#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from build_component_scope import aliases_for_node, reachable, reconstruct_hosts
from common import OUTPUT, discover_role_manifest, dump_json, norm, write_csv, write_jsonl_gz
from diagnostic_assets import (
    a0_region_cache_path, discover_future_apks, future_cohort,
    load_future_endpoint_rows,
)
from extract_contracts import extract_one


def main() -> None:
    cohort = future_cohort(); ids = [sid for sid, _ in cohort]; wanted = set(ids)
    year_by_sha = dict(cohort)
    apk_map, apk_audit = discover_future_apks()
    dump_json(OUTPUT / "future_apk_path_audit.json", {
        "wanted": len(wanted), "found": len(apk_map), "sources": apk_audit,
        "missing_preview": sorted(wanted - set(apk_map))[:20],
    })
    if len(apk_map) != 1989:
        raise RuntimeError(
            f"BLOCKED_BY_CONTRACT: future APK path coverage {len(apk_map)}/1989; "
            "set CCBC_FUTURE_APK_INDEX or CCBC_FUTURE_APK_ROOTS"
        )
    shard_dir = OUTPUT / "cache/future_contracts"; shard_dir.mkdir(parents=True, exist_ok=True)
    failed = 0
    for index, sid in enumerate(ids, 1):
        target = shard_dir / f"{sid}.json"
        if target.is_file() and target.stat().st_size:
            try:
                if json.loads(target.read_text(encoding="utf-8")).get("status") == "OK": continue
            except Exception: pass
        try: row = extract_one(sid, Path(apk_map[sid]))
        except Exception as exc:
            failed += 1; row = {"sha256": sid, "status": "ERROR", "error_type": type(exc).__name__, "error": str(exc)}
        target.write_text(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n", encoding="utf-8")
        if index <= 5 or index % 100 == 0 or index == len(ids):
            print(f"CCBC_FUTURE_CONTRACT_EXTRACTION {index}/{len(ids)} failed={failed}", flush=True)
    contracts = {sid: json.loads((shard_dir / f"{sid}.json").read_text(encoding="utf-8")) for sid in ids}
    bad = [sid for sid, row in contracts.items() if row.get("status") != "OK"]
    if bad: raise RuntimeError(f"BLOCKED_BY_CONTRACT: future contract failures={len(bad)} first={bad[:5]}")
    endpoints = load_future_endpoint_rows(wanted)
    if len(endpoints) != 1989:
        raise RuntimeError(f"BLOCKED_BY_CONTRACT: future endpoint coverage {len(endpoints)}/1989")
    manifest = json.loads(discover_role_manifest().read_text(encoding="utf-8"))
    role_mapping = np.asarray(manifest["payload_to_model_role"], int)
    if role_mapping.shape != (31,) or sorted(role_mapping.tolist()) != list(range(31)):
        raise RuntimeError("BLOCKED_BY_CONTRACT: role mapping changed")
    with np.load(a0_region_cache_path(), allow_pickle=False) as z:
        compact_ids = [norm(value) for value in z["sha256"]]
        real_count = np.asarray(z["real_host_count"], int)
    real_by_sha = dict(zip(compact_ids, real_count.tolist()))
    if any(sid not in real_by_sha for sid in ids):
        raise RuntimeError("BLOCKED_BY_CONTRACT: A0 Host cache misses future SHA")
    rows = []; apk_rows = []; total_hosts = assigned_hosts = entry_total = entry_matched = 0
    for index, sid in enumerate(ids, 1):
        started = time.perf_counter(); payload = endpoints[sid]
        node_ids, hosts, adjacency, edge_count = reconstruct_hosts(payload, role_mapping)
        if len(hosts) != int(real_by_sha[sid]):
            raise RuntimeError(f"BLOCKED_BY_CONTRACT: future Host count mismatch {sid}: {len(hosts)} != {real_by_sha[sid]}")
        alias_index = {}
        for node, value in enumerate(node_ids):
            for alias in aliases_for_node(value): alias_index.setdefault(alias, node)
        components = []; host_to_components = [[] for _ in hosts]
        host_position = {row["node"]: pos for pos, row in enumerate(hosts)}
        local_total = local_matched = 0
        for component in contracts[sid]["components"]:
            seeds = set(); callbacks = []
            for entry in component.get("entry_methods", []):
                local_total += 1; entry_total += 1; matched = None
                for alias in entry.get("aliases", []):
                    if str(alias) in alias_index: matched = alias_index[str(alias)]; break
                    if norm(alias) in alias_index: matched = alias_index[norm(alias)]; break
                if matched is not None:
                    seeds.add(int(matched)); local_matched += 1; entry_matched += 1; callbacks.append(entry.get("callback", "unknown"))
            visited = reachable(adjacency, seeds) if seeds else set()
            positions = sorted(host_position[node] for node in visited if node in host_position)
            for position in positions: host_to_components[position].append(int(component["component_id"]))
            components.append({
                "component_id": int(component["component_id"]), "type": component["type"],
                "tokens": component["tokens"], "weak_tokens": component.get("weak_tokens", []),
                "exported": component.get("exported", "implicit"),
                "entry_callbacks": sorted(set(callbacks)), "entry_node_count": len(seeds),
                "reachable_host_positions": positions,
                "reachable_roles": sorted({role for position in positions for role in hosts[position]["roles"]}),
            })
        host_rows = []
        for position, host in enumerate(hosts):
            component_ids = sorted(set(host_to_components[position])); assigned_hosts += int(bool(component_ids))
            host_rows.append({
                "host_position": position, "host_id": host["host_id"], "roles": host["roles"],
                "role_counts": host["role_counts"], "reachable_component_ids": component_ids,
            })
        total_hosts += len(hosts)
        rows.append({
            "sha256": sid, "status": "OK", "components": components, "hosts": host_rows,
            "weak_permissions": contracts[sid].get("weak_permissions", []),
            "uses_features": contracts[sid].get("uses_features", []),
            "graph_node_count": len(node_ids), "graph_edge_count": edge_count,
        })
        apk_rows.append({
            "sha256": sid, "year": year_by_sha[sid],
            "component_count": len(components), "entry_method_count": local_total,
            "entry_node_matched": local_matched, "entry_mapping_ratio": local_matched / max(local_total, 1),
            "host_count": len(hosts), "assigned_host_count": sum(bool(value) for value in host_to_components),
            "host_assignment_ratio": sum(bool(value) for value in host_to_components) / max(len(hosts), 1),
            "graph_node_count": len(node_ids), "graph_edge_count": edge_count,
            "scope_seconds": time.perf_counter() - started,
        })
        if index <= 5 or index % 100 == 0 or index == len(ids): print(f"CCBC_FUTURE_COMPONENT_SCOPE {index}/{len(ids)}", flush=True)
    write_jsonl_gz(OUTPUT / "cache/component_scope_future.jsonl.gz", rows)
    write_csv(OUTPUT / "future_component_mapping_by_apk.csv", apk_rows)
    summary = {
        "rows": len(rows), "entry_methods": entry_total, "entry_nodes_matched": entry_matched,
        "entry_mapping_ratio": entry_matched / max(entry_total, 1), "real_hosts": total_hosts,
        "assigned_hosts": assigned_hosts, "host_assignment_ratio": assigned_hosts / max(total_hosts, 1),
        "positional_matching_used": False, "fuzzy_class_matching_used": False,
        "new_call_graph_constructed": False, "future_rows_loaded": 1989,
    }
    dump_json(OUTPUT / "future_component_mapping_audit.json", summary)
    print("CCBC_FUTURE_SCOPE_DONE", flush=True); print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__": main()
