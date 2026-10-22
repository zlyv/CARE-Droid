"""Deterministic bounded graph view centred on sensitive API endpoints.

The entry-path view intentionally keeps only high-confidence paths rooted at
Manifest component callbacks.  This module provides the complementary view:
it starts from every mapped sensitive endpoint and retains a small caller and
callee neighbourhood even when no Manifest entry can reach the endpoint.

Raw method signatures are accepted only as transient in-memory keys.  The
returned payload contains stable hashes, integer indexes, role IDs, and small
numeric arrays; it is therefore safe to cache without serialising a call
graph or method-signature list.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from typing import Mapping, Sequence


@dataclass(frozen=True)
class EndpointViewPolicy:
    upstream_hops: int = 2
    downstream_hops: int = 1
    K_in: int = 8
    K_out: int = 4
    endpoint_cap: int = 128
    node_cap: int = 2048
    edge_cap: int = 4096
    reachability_max_states: int = 100000


def _stable_id(value: str) -> int:
    digest = hashlib.sha256(str(value).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False)


def _payload_hash(payload: Mapping) -> str:
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _package_descriptor_prefix(package_name: str) -> str:
    value = str(package_name or "").strip().casefold().replace(".", "/")
    return f"l{value}/" if value else ""


def _provenance(record: Mapping, package_prefix: str) -> int:
    """Return 0=package-aligned, 1=bundled non-package, 2=external/framework."""

    if bool(record.get("external")):
        return 2
    class_name = str(record.get("class_name", "")).casefold()
    if package_prefix and class_name.startswith(package_prefix):
        return 0
    return 1


def _edge_key(edge: Mapping) -> tuple:
    return (
        str(edge.get("source", "")),
        str(edge.get("target", "")),
        str(edge.get("edge_type", "DIRECT_CALL")),
        bool(edge.get("icc_any")),
        bool(edge.get("reflection_any")),
    )


def entry_reachable_distances(
    entry_signatures: set[str],
    edge_records: Sequence[Mapping],
    max_states: int,
) -> tuple[dict[str, int], bool]:
    """Compute directed entry reachability without making absence claims on cap hit."""

    adjacency: dict[str, set[str]] = defaultdict(set)
    for edge in edge_records:
        source = str(edge.get("source", ""))
        target = str(edge.get("target", ""))
        if source and target:
            adjacency[source].add(target)
    ordered_adjacency = {
        source: sorted(targets) for source, targets in adjacency.items()
    }
    distances = {entry: 0 for entry in entry_signatures}
    queue = deque(sorted(entry_signatures))
    explored = 0
    while queue:
        if explored >= max(1, max_states):
            return distances, True
        source = queue.popleft()
        explored += 1
        distance = distances[source] + 1
        for target in ordered_adjacency.get(source, []):
            if target in distances:
                continue
            distances[target] = distance
            queue.append(target)
    return distances, False


def _select_endpoints(
    candidates: set[str],
    node_records: Mapping[str, Mapping],
    role_tier: Mapping[str, int],
    cap: int,
) -> list[str]:
    """Select endpoints deterministically while preserving role coverage."""

    def tier(signature: str) -> int:
        record = node_records[signature]
        return min(
            (int(role_tier.get(role, 4)) for role in record.get("roles", [])),
            default=4,
        )

    ordered = sorted(candidates, key=lambda value: (tier(value), _stable_id(value)))
    all_roles = sorted(
        {role for value in candidates for role in node_records[value].get("roles", [])},
        key=lambda role: (int(role_tier.get(role, 4)), str(role)),
    )
    selected: list[str] = []
    selected_set: set[str] = set()
    # The first pass guarantees that a rare mapped role is not removed merely
    # because a large APK contains many endpoints from a common role.
    for role in all_roles:
        match = next(
            (
                value for value in ordered
                if value not in selected_set
                and role in node_records[value].get("roles", [])
            ),
            None,
        )
        if match is not None:
            selected.append(match)
            selected_set.add(match)
            if len(selected) >= cap:
                return selected
    for value in ordered:
        if value in selected_set:
            continue
        selected.append(value)
        selected_set.add(value)
        if len(selected) >= cap:
            break
    return selected


def build_endpoint_centered_structure(
    node_records: Mapping[str, Mapping],
    edge_records: Sequence[Mapping],
    entry_signatures: set[str],
    role_vocabulary: Sequence[str],
    role_tier: Mapping[str, int],
    package_name: str,
    policy: EndpointViewPolicy,
) -> dict:
    """Build a compact endpoint-centred graph payload for one APK."""

    if min(
        policy.K_in, policy.K_out, policy.endpoint_cap,
        policy.node_cap, policy.edge_cap, policy.reachability_max_states,
    ) < 1:
        raise ValueError("endpoint view caps must be positive")
    if policy.upstream_hops < 0 or policy.downstream_hops < 0:
        raise ValueError("endpoint view hop counts must be nonnegative")

    role_to_id = {role: index for index, role in enumerate(sorted(role_vocabulary))}
    package_prefix = _package_descriptor_prefix(package_name)
    sensitive_all = {
        signature
        for signature, record in node_records.items()
        if record.get("roles")
    }
    endpoint_limit = min(policy.endpoint_cap, policy.node_cap)
    selected_endpoints = _select_endpoints(
        sensitive_all, node_records, role_tier, endpoint_limit
    )
    endpoint_set = set(selected_endpoints)

    canonical_edges: dict[tuple, dict] = {}
    outgoing: dict[str, list[tuple[str, tuple]]] = defaultdict(list)
    incoming: dict[str, list[tuple[str, tuple]]] = defaultdict(list)
    for raw_edge in edge_records:
        key = _edge_key(raw_edge)
        source, target = key[0], key[1]
        if not source or not target or source not in node_records or target not in node_records:
            continue
        canonical_edges[key] = {
            "source": source,
            "target": target,
            "edge_type": key[2],
            "icc_any": key[3],
            "reflection_any": key[4],
        }
        outgoing[source].append((target, key))
        incoming[target].append((source, key))

    def neighbour_key(signature: str) -> tuple:
        record = node_records[signature]
        tier = min(
            (int(role_tier.get(role, 4)) for role in record.get("roles", [])),
            default=4,
        )
        return (
            _provenance(record, package_prefix),
            0 if record.get("roles") else 1,
            tier,
            _stable_id(signature),
        )

    for source in outgoing:
        outgoing[source].sort(key=lambda item: neighbour_key(item[0]))
    for target in incoming:
        incoming[target].sort(key=lambda item: neighbour_key(item[0]))

    upstream_distance = {endpoint: 0 for endpoint in selected_endpoints}
    downstream_distance = {endpoint: 0 for endpoint in selected_endpoints}
    upstream_edges: set[tuple] = set()
    downstream_edges: set[tuple] = set()

    queue = deque((endpoint, 0) for endpoint in selected_endpoints)
    while queue:
        current, distance = queue.popleft()
        if distance >= policy.upstream_hops:
            continue
        for caller, edge_key in incoming.get(current, [])[: policy.K_in]:
            upstream_edges.add(edge_key)
            next_distance = distance + 1
            previous = upstream_distance.get(caller)
            if previous is None or next_distance < previous:
                upstream_distance[caller] = next_distance
                queue.append((caller, next_distance))

    queue = deque((endpoint, 0) for endpoint in selected_endpoints)
    while queue:
        current, distance = queue.popleft()
        if distance >= policy.downstream_hops:
            continue
        for callee, edge_key in outgoing.get(current, [])[: policy.K_out]:
            downstream_edges.add(edge_key)
            next_distance = distance + 1
            previous = downstream_distance.get(callee)
            if previous is None or next_distance < previous:
                downstream_distance[callee] = next_distance
                queue.append((callee, next_distance))

    candidate_nodes = set(upstream_distance) | set(downstream_distance)

    def active_node_key(signature: str) -> tuple:
        record = node_records[signature]
        distance = min(
            upstream_distance.get(signature, 10**9),
            downstream_distance.get(signature, 10**9),
        )
        tier = min(
            (int(role_tier.get(role, 4)) for role in record.get("roles", [])),
            default=4,
        )
        return (
            0 if signature in endpoint_set else 1,
            distance,
            _provenance(record, package_prefix),
            tier,
            _stable_id(signature),
        )

    active = sorted(candidate_nodes, key=active_node_key)[: policy.node_cap]
    active_set = set(active)
    node_truncated = len(candidate_nodes) > len(active)

    induced_edges = [
        (key, edge)
        for key, edge in canonical_edges.items()
        if edge["source"] in active_set and edge["target"] in active_set
    ]

    def active_distance(signature: str) -> int:
        return min(
            upstream_distance.get(signature, 10**9),
            downstream_distance.get(signature, 10**9),
        )

    induced_edges.sort(
        key=lambda item: (
            0 if item[1]["source"] in endpoint_set or item[1]["target"] in endpoint_set else 1,
            0 if item[1]["edge_type"] != "DIRECT_CALL" else 1,
            min(active_distance(item[1]["source"]), active_distance(item[1]["target"])),
            _stable_id("|".join(map(str, item[0]))),
        )
    )
    retained_edges = induced_edges[: policy.edge_cap]
    edge_truncated = len(induced_edges) > len(retained_edges)

    entry_distances, reachability_cap_hit = entry_reachable_distances(
        set(entry_signatures), edge_records, policy.reachability_max_states
    )
    node_index = {signature: index for index, signature in enumerate(active)}
    payload = {
        "node_ids": [_stable_id(signature) for signature in active],
        "node_role_ids": [
            [
                role_to_id[role]
                for role in node_records[signature].get("roles", [])
                if role in role_to_id
            ]
            for signature in active
        ],
        "node_base_features": [],
        "edge_index": [[], []],
        "edge_features": [],
        "edge_type_ids": [],
    }
    provenance_counts = Counter()
    reachability_counts = Counter()
    for signature in active:
        record = node_records[signature]
        provenance = _provenance(record, package_prefix)
        provenance_counts[provenance] += 1
        if signature in entry_distances:
            reachable, unreachable, unknown = 1.0, 0.0, 0.0
            reachability_counts["ENTRY_REACHABLE"] += 1
            entry_distance_score = 1.0 / (1.0 + float(entry_distances[signature]))
        elif reachability_cap_hit:
            reachable, unreachable, unknown = 0.0, 0.0, 1.0
            reachability_counts["REACHABILITY_UNKNOWN"] += 1
            entry_distance_score = 0.0
        else:
            reachable, unreachable, unknown = 0.0, 1.0, 0.0
            reachability_counts["ENTRY_UNREACHABLE"] += 1
            entry_distance_score = 0.0
        endpoint_distance = min(
            upstream_distance.get(signature, 10**9),
            downstream_distance.get(signature, 10**9),
        )
        endpoint_distance_score = (
            1.0 / (1.0 + float(endpoint_distance))
            if endpoint_distance < 10**9 else 0.0
        )
        tier = min(
            (int(role_tier.get(role, 4)) for role in record.get("roles", [])),
            default=4,
        )
        payload["node_base_features"].append([
            float(bool(record.get("external"))),
            float(bool(record.get("roles"))),
            float(tier) / 4.0,
            float(signature in entry_signatures),
            float(signature in endpoint_set),
            float(provenance == 0),
            float(provenance == 1),
            float(provenance == 2),
            reachable,
            unreachable,
            unknown,
            entry_distance_score,
            float(signature in upstream_distance),
            float(signature in downstream_distance),
            endpoint_distance_score,
        ])

    for key, edge in retained_edges:
        source = edge["source"]
        target = edge["target"]
        payload["edge_index"][0].append(node_index[source])
        payload["edge_index"][1].append(node_index[target])
        payload["edge_features"].append([
            float(bool(edge.get("icc_any"))),
            float(bool(edge.get("reflection_any"))),
            float(edge.get("edge_type") != "DIRECT_CALL"),
            float(target in endpoint_set),
            float(source in endpoint_set),
            float(key in upstream_edges),
            float(key in downstream_edges),
        ])
        payload["edge_type_ids"].append(_stable_id(str(edge.get("edge_type"))))

    selected_role_counts = Counter(
        role
        for signature in selected_endpoints
        for role in node_records[signature].get("roles", [])
    )
    reachable_endpoints = sum(
        1 for signature in selected_endpoints if signature in entry_distances
    )
    unknown_endpoints = (
        len(selected_endpoints) - reachable_endpoints if reachability_cap_hit else 0
    )
    unreachable_endpoints = (
        0 if reachability_cap_hit else len(selected_endpoints) - reachable_endpoints
    )
    return {
        "schema_version": "1.0.0",
        "view": "SENSITIVE_ENDPOINT_CENTERED_BOUNDED_GRAPH",
        "node_count": len(active),
        "edge_count": len(retained_edges),
        "base_node_feature_count": 15,
        "edge_feature_count": 7,
        "role_vocabulary_size": len(role_to_id),
        "tensor_sha256": _payload_hash(payload),
        "summary": {
            "sensitive_endpoint_count_total": len(sensitive_all),
            "selected_endpoint_count": len(selected_endpoints),
            "endpoint_truncated": len(sensitive_all) > len(selected_endpoints),
            "candidate_node_count": len(candidate_nodes),
            "node_truncated": node_truncated,
            "candidate_induced_edge_count": len(induced_edges),
            "edge_truncated": edge_truncated,
            "entry_reachable_selected_endpoint_count": reachable_endpoints,
            "entry_unreachable_selected_endpoint_count": unreachable_endpoints,
            "reachability_unknown_selected_endpoint_count": unknown_endpoints,
            "entry_reachability_state_cap_hit": reachability_cap_hit,
            "selected_endpoint_role_counts": dict(sorted(selected_role_counts.items())),
            "active_node_provenance_counts": {
                "PACKAGE_ALIGNED": provenance_counts[0],
                "BUNDLED_NON_PACKAGE": provenance_counts[1],
                "EXTERNAL_OR_FRAMEWORK": provenance_counts[2],
            },
            "active_node_reachability_counts": dict(sorted(reachability_counts.items())),
        },
        "payload": payload,
    }
