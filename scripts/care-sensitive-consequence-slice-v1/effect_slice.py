#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import random
from collections import Counter, defaultdict, deque
from typing import Any

import numpy as np

from common import endpoint_hosts

MAX_DEPTH = 4
GROUPS = {
    "TRANSFER": {"NETWORK", "SMS_CALL", "IPC"},
    "PERSIST": {"FILE_WRITE", "COMPONENT_STATE"},
    "EXECUTE": {"PROCESS_EXEC", "DYNAMIC_LOAD"},
    "VISIBLE": {"UI_OUTPUT"},
    "SENSITIVE_ACTION": {"AUTH_ACCOUNT", "MEDIA_CAPTURE"},
}
VARIANTS = ("REAL", "LOCAL_ONLY", "ONE_HOP", "EFFECT_SHUFFLE", "EDGE_SHUFFLE")


def seeded(sha256: str, variant: str) -> random.Random:
    value = int.from_bytes(hashlib.sha256(f"{sha256}:{variant}:20261008".encode()).digest()[:8], "big")
    return random.Random(value)


def method_package(method: dict[str, Any]) -> str:
    value = str(method.get("class", ""))
    return value.rsplit("/", 1)[0] if "/" in value else value


def edge_pairs(graph: dict[str, Any], variant: str, sha256: str) -> list[tuple[int, int]]:
    pairs = [(int(row[0]), int(row[1])) for row in graph.get("edges", []) if len(row) >= 2]
    if variant != "EDGE_SHUFFLE" or len(pairs) < 2:
        return pairs
    targets = [target for _, target in pairs]; seeded(sha256, variant).shuffle(targets)
    return [(pairs[index][0], targets[index]) for index in range(len(pairs))]


def group_targets(graph: dict[str, Any], variant: str, sha256: str) -> dict[str, set[int]]:
    bundles = [tuple(method.get("external_tags") or ()) for method in graph["methods"]]
    if variant == "EFFECT_SHUFFLE" and len(bundles) > 1:
        bundles = list(bundles); seeded(sha256, variant).shuffle(bundles)
    output = {group: set() for group in GROUPS}
    for index, tags in enumerate(bundles):
        tags = set(tags)
        for group, accepted in GROUPS.items():
            if tags & accepted:
                output[group].add(index)
    return output


def reverse_nearest(method_count: int, pairs: list[tuple[int, int]], targets: set[int], max_depth: int) -> tuple[dict[int, int], dict[int, int], dict[int, int]]:
    reverse: dict[int, list[int]] = defaultdict(list)
    for source, target in pairs:
        if 0 <= source < method_count and 0 <= target < method_count:
            reverse[target].append(source)
    distance = {node: 0 for node in sorted(targets)}
    endpoint = {node: node for node in sorted(targets)}
    next_hop: dict[int, int] = {}
    queue = deque(sorted(targets))
    while queue:
        current = queue.popleft()
        if distance[current] >= max_depth:
            continue
        for predecessor in sorted(reverse.get(current, [])):
            candidate = distance[current] + 1
            if predecessor not in distance:
                distance[predecessor] = candidate; endpoint[predecessor] = endpoint[current]
                next_hop[predecessor] = current; queue.append(predecessor)
            elif candidate == distance[predecessor] and endpoint[current] < endpoint[predecessor]:
                endpoint[predecessor] = endpoint[current]; next_hop[predecessor] = current
    return distance, endpoint, next_hop


def boundary(methods: list[dict[str, Any]], source: int, target: int) -> str:
    left, right = methods[source], methods[target]
    if left.get("class") == right.get("class"):
        return "SAME_CLASS"
    if method_package(left) == method_package(right):
        return "SAME_PACKAGE"
    return "CROSS_PACKAGE"


def extract_variant(graph: dict[str, Any], hosts: list[dict[str, Any]], weights: np.ndarray,
                    critical: int, sha256: str, variant: str) -> tuple[dict[str, float], list[dict[str, Any]], dict[str, Any]]:
    methods = graph["methods"]; method_count = len(methods)
    index_by_id = {int(method["node_id"]): index for index, method in enumerate(methods)}
    effective_variant = variant if variant in {"EFFECT_SHUFFLE", "EDGE_SHUFFLE"} else "REAL"
    pairs = edge_pairs(graph, effective_variant, sha256)
    targets = group_targets(graph, effective_variant, sha256)
    maximum = 0 if variant == "LOCAL_ONLY" else 1 if variant == "ONE_HOP" else MAX_DEPTH
    states = {}
    for group in GROUPS:
        states[group] = reverse_nearest(method_count, pairs, targets[group], maximum)

    host_rows = []; slice_nodes = set(); endpoint_frequency = Counter()
    for order, host in enumerate(hosts):
        node = index_by_id.get(int(host["node_id"])); row = {"host_order": order, "aligned": int(node is not None),
            "roles": list(host["roles"]), "attention": float(weights[order]), "critical": int(order == critical), "groups": {}}
        if node is not None:
            for group, (distances, endpoints, next_hops) in states.items():
                distance = distances.get(node); target = endpoints.get(node)
                if distance is None or distance > maximum:
                    row["groups"][group] = {"reachable": 0, "distance": None, "boundary": "NONE", "target": None}
                    continue
                relation = boundary(methods, node, target)
                row["groups"][group] = {"reachable": 1, "distance": int(distance), "boundary": relation, "target": int(target)}
                endpoint_frequency[(group, int(target))] += 1
                current = node; slice_nodes.add(current)
                while current in next_hops:
                    current = next_hops[current]; slice_nodes.add(current)
                    if current == target:
                        break
        else:
            for group in GROUPS:
                row["groups"][group] = {"reachable": 0, "distance": None, "boundary": "NONE", "target": None}
        host_rows.append(row)

    features: dict[str, float] = {}
    mean_weights = np.full(len(hosts), 1 / max(1, len(hosts)))
    for prefix, local_weights in (("attn", weights), ("mean", mean_weights)):
        any_reach = []
        for row in host_rows:
            any_reach.append(any(value["reachable"] for value in row["groups"].values()))
        features[f"{prefix}_any_effect"] = float(sum(weight * int(value) for weight, value in zip(local_weights, any_reach)))
        for group in GROUPS:
            values = [row["groups"][group] for row in host_rows]
            features[f"{prefix}_{group}_reach"] = float(sum(weight * value["reachable"] for weight, value in zip(local_weights, values)))
            features[f"{prefix}_{group}_inverse_distance"] = float(sum(weight / (1 + value["distance"]) for weight, value in zip(local_weights, values) if value["reachable"]))
            for depth_name, predicate in (("d0", lambda d: d == 0), ("d1", lambda d: d == 1),
                                          ("d2", lambda d: d == 2), ("d3_4", lambda d: d is not None and 3 <= d <= 4)):
                features[f"{prefix}_{group}_{depth_name}"] = float(sum(weight for weight, value in zip(local_weights, values) if value["reachable"] and predicate(value["distance"])))
            for relation in ("SAME_CLASS", "SAME_PACKAGE", "CROSS_PACKAGE"):
                features[f"{prefix}_{group}_{relation}"] = float(sum(weight for weight, value in zip(local_weights, values) if value["boundary"] == relation))
            features[f"{prefix}_{group}_unique_target"] = float(sum(weight for weight, value in zip(local_weights, values)
                if value["reachable"] and endpoint_frequency[(group, value["target"])] == 1))
            features[f"{prefix}_{group}_shared_target"] = float(sum(weight for weight, value in zip(local_weights, values)
                if value["reachable"] and endpoint_frequency[(group, value["target"])] > 1))

    for role in range(31):
        for group in GROUPS:
            value = 0.0
            for weight, row in zip(weights, host_rows):
                if role in row["roles"] and row["groups"][group]["reachable"]:
                    value += weight / max(1, len(row["roles"]))
            features[f"role_{role:02d}_{group}"] = float(value)

    critical_row = host_rows[critical] if 0 <= critical < len(host_rows) else None
    features["critical_any_effect"] = float(bool(critical_row and any(value["reachable"] for value in critical_row["groups"].values())))
    for group in GROUPS:
        features[f"critical_{group}"] = float(bool(critical_row and critical_row["groups"][group]["reachable"]))
    features["slice_node_ratio"] = float(len(slice_nodes) / max(1, method_count))
    features["slice_reduction"] = float(1 - len(slice_nodes) / max(1, method_count))
    features["effect_group_diversity"] = float(sum(any(row["groups"][group]["reachable"] for row in host_rows) for group in GROUPS) / len(GROUPS))
    audit = {"aligned_hosts": sum(row["aligned"] for row in host_rows), "host_count": len(host_rows),
             "any_effect_hosts": sum(any(value["reachable"] for value in row["groups"].values()) for row in host_rows),
             "attention_effect_mass": features["attn_any_effect"], "critical_effect": features["critical_any_effect"],
             "slice_node_count": len(slice_nodes), "slice_node_ratio": features["slice_node_ratio"],
             "effect_method_count": len(set().union(*targets.values())),
             "effect_tag_count": sum(len(value) for value in targets.values()),
             "graph_method_count": method_count, "graph_edge_count": len(pairs)}
    return features, host_rows, audit


def extract(graph: dict[str, Any], payload: dict[str, Any], attention_info: dict[str, Any]) -> dict[str, Any]:
    sha256 = str(graph.get("sha256") or "unknown")
    hosts = endpoint_hosts(payload)
    # The frozen attention cache uses a single padding value for the two APKs
    # whose endpoint payload contains no Host nodes.  Padding is not evidence:
    # retain the APK with an empty Host set and an all-zero representation.
    if not hosts:
        weights = np.asarray([], float); critical = -1
    elif len(attention_info["attention"]) != len(hosts):
        raise RuntimeError(f"attention/Host mismatch attention={len(attention_info['attention'])} hosts={len(hosts)}")
    else:
        weights = np.asarray(attention_info["attention"], float); critical = int(attention_info["critical"])
    if len(weights) and weights.sum() > 0:
        weights = weights / weights.sum()
    elif len(weights):
        weights = np.full(len(weights), 1 / len(weights))
    variants = {}; real_hosts = []; real_audit = None
    for variant in VARIANTS:
        features, rows, audit = extract_variant(graph, hosts, weights, critical, sha256, variant)
        variants[variant] = features
        if variant == "REAL":
            real_hosts, real_audit = rows, audit
    count_only = {key: float(real_audit[key]) for key in ("graph_method_count", "graph_edge_count", "host_count", "effect_method_count", "effect_tag_count")}
    path_only = {key: value for key, value in variants["REAL"].items() if any(token in key for token in ("any_effect", "inverse_distance", "d0", "d1", "d2", "d3_4", "slice_node_ratio", "slice_reduction"))}
    variants["COUNT_ONLY"] = count_only; variants["PATH_ONLY"] = path_only
    real_audit["empty_host_payload"] = int(not hosts)
    return {"sha256": sha256, "schema_version": "1.0.1", "variants": variants,
            "audit": real_audit, "hosts": real_hosts,
            "interpretation_boundary": "call reachability and shortest-path projection; not data flow, execution proof, or user-intent proof"}
