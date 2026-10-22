#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict, deque
from pathlib import Path

import numpy as np

from assets import load_endpoint_rows
from common import (
    OUTPUT, ROLE_COUNT, discover_compact_cache, discover_role_manifest, dump_json,
    norm, read_jsonl_gz, write_jsonl_gz,
)


def aliases_for_node(value) -> set[str]:
    values = {str(value), norm(value)}
    try:
        integer = int(value)
        values.add(str(integer)); values.add(f"{integer:016x}")
    except Exception:
        text = str(value).strip().casefold()
        if len(text) == 16 and all(ch in "0123456789abcdef" for ch in text):
            values.add(str(int(text, 16)))
    return values


def reconstruct_hosts(payload: dict, role_mapping: np.ndarray):
    ids = list(payload.get("node_ids") or [])
    base = np.asarray(payload.get("node_base_features") or [], np.float32)
    role_rows = payload.get("node_role_ids") or []
    edges = payload.get("edge_index") or [[], []]
    if base.size == 0:
        base = np.empty((0, 15), np.float32)
    if base.ndim != 2 or len(ids) != len(base) or len(role_rows) != len(base) or len(edges) != 2:
        raise RuntimeError("endpoint node schema mismatch")
    src = np.asarray(edges[0], int); dst = np.asarray(edges[1], int)
    if len(src) != len(dst):
        raise RuntimeError("endpoint edge schema mismatch")
    roles = [
        sorted({int(role_mapping[int(value)]) for value in (row or []) if 0 <= int(value) < ROLE_COUNT})
        for row in role_rows
    ]
    endpoints = set(np.flatnonzero(base[:, 4] > .5).astype(int).tolist())
    targets: dict[int, set[int]] = defaultdict(set)
    for left, right in zip(src, dst):
        if int(right) in endpoints:
            targets[int(left)].add(int(right))
    hosts = []
    for node in sorted(targets):
        counts = np.zeros(ROLE_COUNT, np.float32)
        for endpoint in targets[node]:
            for role in roles[endpoint]:
                counts[role] += 1
        if np.any(counts > 0):
            hosts.append({
                "node": int(node), "host_id": str(ids[node]),
                "roles": np.flatnonzero(counts > 0).astype(int).tolist(),
                "role_counts": counts.astype(float).tolist(),
            })
    adjacency = [[] for _ in ids]
    for left, right in zip(src, dst):
        left = int(left); right = int(right)
        if 0 <= left < len(ids) and 0 <= right < len(ids) and left != right:
            adjacency[left].append(right)
    return ids, hosts, adjacency, len(src)


def reachable(adjacency: list[list[int]], seeds: set[int]) -> set[int]:
    seen = set(seeds); queue = deque(seeds)
    while queue:
        node = queue.popleft()
        for nxt in adjacency[node]:
            if nxt not in seen:
                seen.add(nxt); queue.append(nxt)
    return seen


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    contracts = {row["sha256"]: row for row in read_jsonl_gz(OUTPUT / "cache/contracts_2017.jsonl.gz")}
    ids = sorted(contracts)
    if args.limit:
        ids = ids[:args.limit]
    endpoint = load_endpoint_rows(set(ids))
    manifest = json.loads(discover_role_manifest().read_text(encoding="utf-8"))
    role_mapping = np.asarray(manifest["payload_to_model_role"], int)
    if role_mapping.shape != (31,) or sorted(role_mapping.tolist()) != list(range(31)):
        raise RuntimeError("BLOCKED_BY_CONTRACT: role mapping changed")
    with np.load(discover_compact_cache(), allow_pickle=False) as z:
        compact_ids = [norm(value) for value in z["sha256"]]
        real_count = np.asarray(z["real_host_count"], int)
    real_by_sha = dict(zip(compact_ids, real_count.tolist()))
    rows = []
    apk_audit = []
    total_hosts = assigned_hosts = entry_total = entry_matched = 0
    for index, sid in enumerate(ids, 1):
        apk_started = time.perf_counter()
        payload = endpoint[sid]
        node_ids, hosts, adjacency, edge_count = reconstruct_hosts(payload, role_mapping)
        expected = int(real_by_sha[sid])
        if len(hosts) != expected:
            raise RuntimeError(f"BLOCKED_BY_CONTRACT: Host count mismatch {sid}: {len(hosts)} != {expected}")
        alias_index: dict[str, int] = {}
        for node, value in enumerate(node_ids):
            for alias in aliases_for_node(value):
                alias_index.setdefault(alias, node)
        components = []
        host_to_components: list[list[int]] = [[] for _ in hosts]
        host_position = {row["node"]: pos for pos, row in enumerate(hosts)}
        local_entry_total = local_entry_matched = 0
        for component in contracts[sid]["components"]:
            seeds = set()
            entry_callbacks = []
            for entry in component.get("entry_methods", []):
                local_entry_total += 1; entry_total += 1
                matched = None
                for alias in entry.get("aliases", []):
                    if str(alias) in alias_index:
                        matched = alias_index[str(alias)]; break
                    if norm(alias) in alias_index:
                        matched = alias_index[norm(alias)]; break
                if matched is not None:
                    seeds.add(int(matched)); local_entry_matched += 1; entry_matched += 1
                    entry_callbacks.append(entry.get("callback", "unknown"))
            visited = reachable(adjacency, seeds) if seeds else set()
            reachable_positions = sorted(host_position[node] for node in visited if node in host_position)
            for position in reachable_positions:
                host_to_components[position].append(int(component["component_id"]))
            reachable_roles = sorted({role for position in reachable_positions for role in hosts[position]["roles"]})
            components.append({
                "component_id": int(component["component_id"]), "type": component["type"],
                "tokens": component["tokens"], "weak_tokens": component.get("weak_tokens", []),
                "exported": component.get("exported", "implicit"),
                "entry_callbacks": sorted(set(entry_callbacks)),
                "entry_node_count": len(seeds), "reachable_host_positions": reachable_positions,
                "reachable_roles": reachable_roles,
            })
        host_rows = []
        for position, host in enumerate(hosts):
            component_ids = sorted(set(host_to_components[position]))
            if component_ids:
                assigned_hosts += 1
            host_rows.append({
                "host_position": position, "host_id": host["host_id"],
                "roles": host["roles"], "role_counts": host["role_counts"],
                "reachable_component_ids": component_ids,
            })
        total_hosts += len(hosts)
        row = {
            "sha256": sid, "status": "OK", "components": components, "hosts": host_rows,
            "weak_permissions": contracts[sid].get("weak_permissions", []),
            "uses_features": contracts[sid].get("uses_features", []),
            "graph_node_count": len(node_ids), "graph_edge_count": edge_count,
        }
        rows.append(row)
        apk_audit.append({
            "sha256": sid, "component_count": len(components), "entry_method_count": local_entry_total,
            "entry_node_matched": local_entry_matched,
            "entry_mapping_ratio": local_entry_matched / max(local_entry_total, 1),
            "host_count": len(hosts), "assigned_host_count": sum(bool(v) for v in host_to_components),
            "host_assignment_ratio": sum(bool(v) for v in host_to_components) / max(len(hosts), 1),
            "graph_node_count": len(node_ids), "graph_edge_count": edge_count,
            "scope_seconds": time.perf_counter() - apk_started,
        })
        if index <= 5 or index % 25 == 0 or index == len(ids):
            print(f"CCBC_COMPONENT_SCOPE {index}/{len(ids)}", flush=True)
    write_jsonl_gz(OUTPUT / "cache/component_scope_2017.jsonl.gz", rows)
    from common import write_csv
    write_csv(OUTPUT / "component_mapping_by_apk.csv", apk_audit)
    summary = {
        "source_rows": len(rows), "entry_methods": entry_total,
        "entry_nodes_matched": entry_matched,
        "entry_mapping_ratio": entry_matched / max(entry_total, 1),
        "real_hosts": total_hosts, "assigned_hosts": assigned_hosts,
        "host_assignment_ratio": assigned_hosts / max(total_hosts, 1),
        "positional_matching_used": False,
        "fuzzy_class_matching_used": False,
        "new_call_graph_constructed": False,
        "future_rows_loaded": 0,
    }
    dump_json(OUTPUT / "component_mapping_audit.json", summary)
    print("CCBC_COMPONENT_SCOPE_DONE")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

