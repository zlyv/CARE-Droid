#!/usr/bin/env python3
from __future__ import annotations

import numpy as np

ROLE_COUNT = 31

# Fixed semantic groups from the frozen model role taxonomy.  They are used only
# to collapse directed Host-to-Host relations; no outcome is used to define them.
EXECUTION = frozenset((0, 1, 2, 3, 4, 5))
COLLECTION = frozenset((8, 9, 10, 11, 12, 13, 14, 15, 16, 28, 29))
TRANSPORT = frozenset((17, 18, 25, 26, 27))
PERSISTENCE = frozenset((6, 7, 19, 20, 21, 22, 23, 24))


def safe_ratio(numerator, denominator):
    return float(numerator) / float(denominator) if denominator else 0.0


def entropy(values):
    values = np.asarray(values, float)
    total = float(values.sum())
    if total <= 0:
        return 0.0
    probability = values[values > 0] / total
    return float(-(probability * np.log(probability)).sum() / np.log(ROLE_COUNT))


def payload_arrays(payload, role_remap):
    base = np.asarray(payload.get("node_base_features") or [], float)
    if base.size == 0:
        base = np.zeros((0, 15), float)
    if base.ndim != 2 or base.shape[1] != 15:
        raise ValueError("node_base_features must have shape [N,15]")
    roles = []
    for row in payload.get("node_role_ids") or []:
        roles.append(frozenset(int(role_remap[int(value)]) for value in (row or [])))
    edge_index = payload.get("edge_index") or [[], []]
    if len(edge_index) != 2:
        raise ValueError("edge_index must have two rows")
    source = np.asarray(edge_index[0], int); target = np.asarray(edge_index[1], int)
    edge_feature = np.asarray(payload.get("edge_features") or [], float)
    if edge_feature.size == 0:
        edge_feature = np.zeros((len(source), 7), float)
    if len(base) != len(roles) or len(source) != len(target) or edge_feature.shape != (len(source), 7):
        raise ValueError("endpoint payload dimensions disagree")
    return base, roles, source, target, edge_feature


def provenance(prefix, base, indices, output):
    value = base[list(indices)] if len(indices) else np.empty((0, 15), float)
    denominator = len(value)
    output[prefix + "package_aligned_ratio"] = safe_ratio(value[:, 5].sum(), denominator)
    output[prefix + "bundled_nonpackage_ratio"] = safe_ratio(value[:, 6].sum(), denominator)
    output[prefix + "external_framework_ratio"] = safe_ratio(value[:, 7].sum(), denominator)


def extract_features(payload, role_remap):
    base, roles, source, target, edge_feature = payload_arrays(payload, role_remap)
    endpoints = set(map(int, np.flatnonzero(base[:, 4] > .5)))
    targets = {}
    for src, dst in zip(source, target):
        if int(dst) in endpoints:
            targets.setdefault(int(src), set()).add(int(dst))
    host_roles = {}
    for host, values in targets.items():
        merged = frozenset().union(*(roles[index] for index in values))
        if merged:
            host_roles[host] = merged
    hosts = set(host_roles)
    host_edges = {(int(src), int(dst)) for src, dst in zip(source, target)
                  if int(src) in hosts and int(dst) in hosts and int(src) != int(dst)}

    output = {}
    provenance("endpoint_", base, endpoints, output)
    provenance("host_", base, hosts, output)
    endpoint_base = base[list(endpoints)] if endpoints else np.empty((0, 15), float)
    output["endpoint_entry_reachable_ratio"] = safe_ratio(endpoint_base[:, 8].sum(), len(endpoint_base))
    output["endpoint_entry_unreachable_ratio"] = safe_ratio(endpoint_base[:, 9].sum(), len(endpoint_base))
    output["endpoint_reachability_unknown_ratio"] = safe_ratio(endpoint_base[:, 10].sum(), len(endpoint_base))
    output["endpoint_entry_distance_score_mean"] = float(endpoint_base[:, 11].mean()) if len(endpoint_base) else 0.0

    role_count = np.zeros(ROLE_COUNT, float)
    for index in endpoints:
        for role in roles[index]:
            role_count[role] += 1.0
    output["endpoint_role_entropy"] = entropy(role_count)
    output["endpoint_dominant_role_ratio"] = safe_ratio(role_count.max(initial=0.0), role_count.sum())
    output["endpoint_role_coverage_ratio"] = safe_ratio(np.count_nonzero(role_count), ROLE_COUNT)
    output["host_multirole_ratio"] = safe_ratio(sum(len(value) > 1 for value in host_roles.values()), len(host_roles))

    degree = {host: 0 for host in hosts}
    same = cross = 0
    unique_pairs = set()
    motif = {"execution_to_transport": 0, "collection_to_transport": 0,
             "persistence_to_collection": 0, "persistence_to_transport": 0}
    for src, dst in host_edges:
        degree[src] += 1; degree[dst] += 1
        left, right = host_roles[src], host_roles[dst]
        pairs = {(a, b) for a in left for b in right}
        unique_pairs.update(pairs)
        if any(a == b for a, b in pairs): same += 1
        if any(a != b for a, b in pairs): cross += 1
        motif["execution_to_transport"] += int(bool(left & EXECUTION) and bool(right & TRANSPORT))
        motif["collection_to_transport"] += int(bool(left & COLLECTION) and bool(right & TRANSPORT))
        motif["persistence_to_collection"] += int(bool(left & PERSISTENCE) and bool(right & COLLECTION))
        motif["persistence_to_transport"] += int(bool(left & PERSISTENCE) and bool(right & TRANSPORT))
    output["host_isolated_ratio"] = safe_ratio(sum(value == 0 for value in degree.values()), len(degree))
    output["host_same_role_edge_ratio"] = safe_ratio(same, len(host_edges))
    output["host_cross_role_edge_ratio"] = safe_ratio(cross, len(host_edges))
    output["host_relation_pair_diversity"] = safe_ratio(len(unique_pairs), min(max(len(host_edges), 1), ROLE_COUNT * ROLE_COUNT))
    for name, value in motif.items():
        output[name + "_ratio"] = safe_ratio(value, len(host_edges))

    incoming_endpoint = np.asarray([index for index, dst in enumerate(target) if int(dst) in endpoints], int)
    selected_edges = edge_feature[incoming_endpoint] if len(incoming_endpoint) else np.empty((0, 7), float)
    output["endpoint_icc_edge_ratio"] = safe_ratio(selected_edges[:, 0].sum(), len(selected_edges))
    output["endpoint_reflection_edge_ratio"] = safe_ratio(selected_edges[:, 1].sum(), len(selected_edges))
    output["endpoint_nondirect_edge_ratio"] = safe_ratio(selected_edges[:, 2].sum(), len(selected_edges))

    # Counts are never part of the semantic score; they are retained only for
    # shortcut/coverage auditing.
    audit = {
        "node_count": len(base), "edge_count": len(source),
        "endpoint_count": len(endpoints), "host_count": len(hosts),
        "host_edge_count": len(host_edges),
    }
    return output, audit


FEATURE_GROUPS = {
    "PROVENANCE": (
        "endpoint_package_aligned_ratio", "endpoint_bundled_nonpackage_ratio",
        "endpoint_external_framework_ratio", "host_package_aligned_ratio",
        "host_bundled_nonpackage_ratio", "host_external_framework_ratio"),
    "TRIGGER_CONTEXT": (
        "endpoint_entry_reachable_ratio", "endpoint_entry_unreachable_ratio",
        "endpoint_reachability_unknown_ratio", "endpoint_entry_distance_score_mean",
        "endpoint_icc_edge_ratio", "endpoint_reflection_edge_ratio",
        "endpoint_nondirect_edge_ratio"),
    "SEMANTIC_ORGANIZATION": (
        "endpoint_role_entropy", "endpoint_dominant_role_ratio",
        "endpoint_role_coverage_ratio", "host_multirole_ratio",
        "host_isolated_ratio", "host_same_role_edge_ratio",
        "host_cross_role_edge_ratio", "host_relation_pair_diversity",
        "execution_to_transport_ratio", "collection_to_transport_ratio",
        "persistence_to_collection_ratio", "persistence_to_transport_ratio"),
}
FEATURE_GROUPS["ALL_SEMANTIC"] = sum(FEATURE_GROUPS.values(), ())


def centroid_score(train_x, train_y, query_x):
    mean = np.mean(train_x, axis=0)
    std = np.std(train_x, axis=0)
    std[std < 1e-6] = 1.0
    train = (train_x - mean) / std
    query = (query_x - mean) / std
    benign = np.mean(train[train_y == 0], axis=0)
    malware = np.mean(train[train_y == 1], axis=0)
    d_benign = np.sqrt(np.mean((query - benign) ** 2, axis=1))
    d_malware = np.sqrt(np.mean((query - malware) ** 2, axis=1))
    return d_malware - d_benign  # positive means closer to benign controls
