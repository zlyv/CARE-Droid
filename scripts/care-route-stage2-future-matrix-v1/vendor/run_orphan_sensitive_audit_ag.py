#!/usr/bin/env python3
"""Audit sensitive call-graph nodes not reachable from manifest entries.

For each APK, the call graph and raw method signatures exist only in memory.
The script writes compact counts, role totals, and reachability coverage.  It
does not write a call graph, DEX bytes, method signatures, component names, or
paths.  This is an audit: it neither reads labels for computation nor trains a
model.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import logging
import shutil
import sys
from collections import Counter, deque
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from run_semantic_mapping_audit_ag import load_yaml  # noqa: E402
from run_static_coverage_audit_ag import read_csv, suppress_logs  # noqa: E402
from tasa_droid.p2.graph_representation import GraphPolicy, build_compact_graph_summary  # noqa: E402


def suppress_loguru() -> None:
    logging.disable(logging.INFO)
    try:
        from loguru import logger
        logger.remove()
    except Exception:
        pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def role_tiers(priority: dict) -> dict[str, int]:
    result: dict[str, int] = {}
    for tier, value in (priority.get("tiers", {}) or {}).items():
        try:
            tier_value = int(tier)
        except (TypeError, ValueError):
            tier_value = 4
        for role in (value or {}).get("roles", []):
            result[str(role)] = tier_value
    return result


def reachable_distances(
    entries: set[str], edges: list[dict], max_states: int
) -> tuple[dict[str, int], bool]:
    """Multi-source directed BFS over the canonical in-memory graph."""
    adjacency: dict[str, list[str]] = {}
    for edge in edges:
        source, target = edge.get("source"), edge.get("target")
        if source and target:
            adjacency.setdefault(str(source), []).append(str(target))
    for source in adjacency:
        adjacency[source].sort()

    distances = {entry: 0 for entry in entries}
    queue = deque(sorted(entries))
    explored = 0
    while queue:
        if explored >= max_states:
            return distances, True
        source = queue.popleft()
        explored += 1
        next_distance = distances[source] + 1
        for target in adjacency.get(source, []):
            if target not in distances:
                distances[target] = next_distance
                queue.append(target)
    return distances, False


def weak_component_count(nodes: set[str], edges: list[dict]) -> tuple[int, int]:
    """Return compact weak-component and induced-edge counts for a node set."""
    if not nodes:
        return 0, 0
    parent = {node: node for node in nodes}

    def find(node: str) -> str:
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    def union(left: str, right: str) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    induced_edges = 0
    for edge in edges:
        source, target = str(edge.get("source", "")), str(edge.get("target", ""))
        if source in nodes and target in nodes:
            induced_edges += 1
            union(source, target)
    return len({find(node) for node in nodes}), induced_edges


def feature_fields(roles: list[str]) -> list[str]:
    return [
        "sha256", "source_id", "metadata_year", "status",
        "entry_method_count", "sensitive_node_count",
        "path_retained_sensitive_node_count",
        "path_excluded_sensitive_node_count",
        "entry_reachable_sensitive_node_count",
        "entry_reachable_within_h_sensitive_node_count",
        "entry_reachable_beyond_h_sensitive_node_count",
        "entry_unreachable_sensitive_node_count",
        "entry_unreachable_sensitive_node_ratio",
        "entry_reachability_complete", "entry_reachability_state_cap_hit",
        "orphan_sensitive_component_count", "orphan_sensitive_induced_edge_count",
    ] + [f"orphan_role__{role}" for role in roles] + [
        f"orphan_tier__{tier}" for tier in range(5)
    ] + ["error_type", "error_message"]


def analyze_one(
    row: dict[str, str], rules: list[dict], priority: dict, edge_rules: list[dict],
    roles: list[str], policy: GraphPolicy, reachability_max_states: int,
) -> dict[str, object]:
    result: dict[str, object] = {
        "sha256": row.get("sha256", ""),
        "source_id": row.get("source_id", ""),
        "metadata_year": row.get("metadata_year", ""),
        "status": "FAILED",
        "error_type": "",
        "error_message": "",
    }
    for name in feature_fields(roles):
        result.setdefault(name, 0)
    apk = dalvik = analysis = graph = None
    try:
        from androguard.misc import AnalyzeAPK

        apk, dalvik, analysis = AnalyzeAPK(row["absolute_path"])
        graph = analysis.get_call_graph(no_isolated=False)
        summary = build_compact_graph_summary(
            apk, graph, rules, priority, policy, edge_rules, include_internal=True
        )
        internal = summary.pop("_internal")
        node_records = internal["node_records"]
        edges = internal["edge_records"]
        entries = set(internal["entry_signatures"])
        path_nodes = set(internal["path_node_signatures"])
        sensitive = {
            signature for signature, record in node_records.items()
            if record.get("roles")
        }
        distances, cap_hit = reachable_distances(
            entries, edges, max(100, reachability_max_states)
        )
        reachable = sensitive & set(distances)
        reachable_within_h = {
            signature for signature in reachable if distances[signature] <= policy.H
        }
        reachable_beyond_h = reachable - reachable_within_h
        # When the traversal reaches its state cap, a node absent from
        # ``distances`` cannot be called unreachable with confidence.
        orphan = set() if cap_hit else sensitive - reachable
        orphan_components, orphan_edges = weak_component_count(orphan, edges)
        tiers = role_tiers(priority)
        orphan_roles = Counter(
            role
            for signature in orphan
            for role in node_records[signature].get("roles", [])
        )
        orphan_tiers = Counter(
            min((tiers.get(role, 4) for role in node_records[signature].get("roles", [])), default=4)
            for signature in orphan
        )
        result.update({
            "status": "OK",
            "entry_method_count": len(entries),
            "sensitive_node_count": len(sensitive),
            "path_retained_sensitive_node_count": len(sensitive & path_nodes),
            "path_excluded_sensitive_node_count": len(sensitive - path_nodes),
            "entry_reachable_sensitive_node_count": len(reachable),
            "entry_reachable_within_h_sensitive_node_count": len(reachable_within_h),
            "entry_reachable_beyond_h_sensitive_node_count": len(reachable_beyond_h),
            "entry_unreachable_sensitive_node_count": len(orphan),
            "entry_unreachable_sensitive_node_ratio": len(orphan) / max(1, len(sensitive)),
            "entry_reachability_complete": int(not cap_hit),
            "entry_reachability_state_cap_hit": int(cap_hit),
            "orphan_sensitive_component_count": orphan_components,
            "orphan_sensitive_induced_edge_count": orphan_edges,
        })
        for role in roles:
            result[f"orphan_role__{role}"] = orphan_roles[role]
        for tier in range(5):
            result[f"orphan_tier__{tier}"] = orphan_tiers[tier]
    except Exception as exc:
        result["error_type"] = type(exc).__name__
        result["error_message"] = str(exc)[:500]
    finally:
        graph = None
        analysis = None
        dalvik = None
        apk = None
        gc.collect()
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lineage", required=True, type=Path)
    parser.add_argument("--mapping", required=True, type=Path)
    parser.add_argument("--priority", required=True, type=Path)
    parser.add_argument("--edge-semantics", required=True, type=Path)
    parser.add_argument("--failure-csv", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--limit", type=int, default=200, help="0 means all APKs")
    parser.add_argument("--H", type=int, default=4)
    parser.add_argument("--K-out", type=int, default=8)
    parser.add_argument("--P-cap", type=int, default=8)
    parser.add_argument("--E-cap", type=int, default=2000)
    parser.add_argument("--max-states", type=int, default=20000)
    parser.add_argument("--reachability-max-states", type=int, default=100000)
    parser.add_argument("--min-free-gb", type=float, default=1.0)
    parser.add_argument("--progress-every", type=int, default=25)
    args = parser.parse_args()

    suppress_loguru()
    suppress_logs()
    mapping = load_yaml(args.mapping)
    priority = load_yaml(args.priority)
    edge_mapping = load_yaml(args.edge_semantics)
    rules = list(mapping.get("rules", []))
    roles = list(mapping.get("roles", []))
    edge_rules = list(edge_mapping.get("edge_rules", []))
    excluded = set()
    if args.failure_csv and args.failure_csv.exists():
        excluded = {
            row.get("sha256", "").casefold()
            for row in read_csv(args.failure_csv)
            if row.get("sha256")
        }
    rows = [
        row for row in read_csv(args.lineage)
        if row.get("sha256", "").casefold() not in excluded
    ]
    # SHA256 ordering gives a deterministic, label-independent sample.
    rows.sort(key=lambda row: row.get("sha256", "").casefold())
    if args.limit > 0:
        rows = rows[: args.limit]
    if not rows:
        raise SystemExit("no APK selected after exclusions")

    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    policy = GraphPolicy(
        H=max(0, args.H), K_out=max(1, args.K_out), P_cap=max(1, args.P_cap),
        E_cap=max(1, args.E_cap), max_states=max(100, args.max_states),
    )
    fields = feature_fields(roles)
    result_rows = []
    statuses = Counter()
    print(f"orphan-sensitive audit started: {len(rows)} APKs", flush=True)
    for index, row in enumerate(rows, 1):
        free_gb = shutil.disk_usage(output).free / (1024 ** 3)
        if free_gb < args.min_free_gb:
            raise SystemExit(
                f"disk guard stopped run: free_gb={free_gb:.3f} < min_free_gb={args.min_free_gb:.3f}"
            )
        result = analyze_one(
            row, rules, priority, edge_rules, roles, policy, args.reachability_max_states
        )
        result_rows.append(result)
        statuses[str(result["status"])] += 1
        if index % max(1, args.progress_every) == 0 or index == len(rows):
            print(
                f"processed {index}/{len(rows)} ok={statuses['OK']} failed={statuses['FAILED']}",
                flush=True,
            )

    summary_path = output / "orphan_sensitive_summary.csv"
    with summary_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(result_rows)
    successful = [row for row in result_rows if row["status"] == "OK"]
    total_sensitive = sum(int(row["sensitive_node_count"]) for row in successful)
    total_orphan = sum(int(row["entry_unreachable_sensitive_node_count"]) for row in successful)
    total_beyond_h = sum(int(row["entry_reachable_beyond_h_sensitive_node_count"]) for row in successful)
    total_path_excluded = sum(int(row["path_excluded_sensitive_node_count"]) for row in successful)
    audit = {
        "schema_version": "1.0.0",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "phase": "P24_ORPHAN_SENSITIVE_NODE_REACHABILITY_AUDIT",
        "formal_training_enabled": False,
        "sample_selection": "sha256_sorted_label_independent",
        "requested_apk_count": len(rows),
        "status_counts": dict(statuses),
        "reachability_state_cap_hit_count": sum(
            int(row["entry_reachability_state_cap_hit"]) for row in successful
        ),
        "aggregate": {
            "sensitive_node_count": total_sensitive,
            "entry_unreachable_sensitive_node_count": total_orphan,
            "entry_unreachable_sensitive_node_ratio": total_orphan / max(1, total_sensitive),
            "entry_reachable_beyond_h_sensitive_node_count": total_beyond_h,
            "path_excluded_sensitive_node_count": total_path_excluded,
        },
        "policy": {
            "H": policy.H, "K_out": policy.K_out, "P_cap": policy.P_cap,
            "E_cap": policy.E_cap, "path_max_states": policy.max_states,
            "reachability_max_states": max(100, args.reachability_max_states),
        },
        "outputs": {"summary": str(summary_path)},
        "notes": [
            "No labels are used for sampling, reachability, or aggregation.",
            "An entry-unreachable count is emitted only when the reachability traversal completes.",
            "Raw call graphs, method signatures, paths, and DEX bytes are never persisted.",
        ],
    }
    (output / "orphan_sensitive_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
