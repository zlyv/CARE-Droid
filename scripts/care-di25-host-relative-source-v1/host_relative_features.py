#!/usr/bin/env python3
"""Label-free, APK-relative Host-V2 feature construction."""
from __future__ import annotations

import math
from collections import defaultdict
import numpy as np

ROLE_COUNT = 31
ARMS = ("HM_R", "HC_R", "HMC_R")


def normalized_entropy(counts):
    x = np.asarray(counts, dtype=np.float64)
    total = float(x.sum())
    if total <= 0:
        return 0.0
    p = x[x > 0] / total
    return float(-np.sum(p * np.log(p)) / math.log(ROLE_COUNT))


def centered_midrank(values):
    """Tie-aware ranks with exact zero mean, computed inside one APK."""
    x = np.asarray(values, dtype=np.float64)
    if x.ndim != 1:
        raise ValueError("centered_midrank expects one dimension")
    n = len(x)
    if n <= 1 or np.all(x == x[0]):
        return np.zeros(n, dtype=np.float32)
    order = np.argsort(x, kind="mergesort")
    ranked = np.empty(n, dtype=np.float64)
    start = 0
    while start < n:
        stop = start + 1
        while stop < n and x[order[stop]] == x[order[start]]:
            stop += 1
        # Average of one-based ranks [start+1, ..., stop].
        ranked[order[start:stop]] = 0.5 * ((start + 1) + stop)
        start = stop
    result = ranked / (n + 1.0) - 0.5
    result -= result.mean()
    return result.astype(np.float32)


def reconstruct_hosts(payload, payload_to_model_role):
    ids = list(payload.get("node_ids") or [])
    mapping = np.asarray(payload_to_model_role, dtype=np.int64)
    if mapping.shape != (ROLE_COUNT,) or sorted(mapping.tolist()) != list(range(ROLE_COUNT)):
        raise RuntimeError("payload_to_model_role must be a 31-role permutation")
    roles = [
        tuple(sorted({int(mapping[int(v)]) for v in (row or []) if 0 <= int(v) < ROLE_COUNT}))
        for row in (payload.get("node_role_ids") or [])
    ]
    base = np.asarray(payload.get("node_base_features") or [], dtype=np.float32)
    if base.size == 0:
        base = np.empty((0, 15), dtype=np.float32)
    edge_index = payload.get("edge_index") or [[], []]
    edge_features = np.asarray(payload.get("edge_features") or [], dtype=np.float32)
    if len(edge_index) != 2:
        raise RuntimeError("endpoint edge_index schema mismatch")
    src = np.asarray(edge_index[0], dtype=np.int64)
    dst = np.asarray(edge_index[1], dtype=np.int64)
    if edge_features.size == 0:
        edge_features = np.zeros((len(src), 7), dtype=np.float32)
    if base.ndim != 2 or base.shape[1] != 15 or len(ids) != len(base) or len(roles) != len(base):
        raise RuntimeError("endpoint node schema mismatch")
    if len(src) != len(dst) or edge_features.shape != (len(src), 7):
        raise RuntimeError("endpoint edge feature schema mismatch")

    endpoints = set(map(int, np.flatnonzero(base[:, 4] > 0.5)))
    targets = defaultdict(set)
    endpoint_edge_rows = defaultdict(list)
    for edge_row, (left, right) in enumerate(zip(src, dst)):
        left, right = int(left), int(right)
        if right in endpoints:
            targets[left].add(right)
            endpoint_edge_rows[left].append(edge_row)
    host_roles = {}
    for host, endpoint_nodes in targets.items():
        merged = tuple(sorted({role for endpoint in endpoint_nodes for role in roles[endpoint]}))
        if merged:
            host_roles[host] = merged
    host_nodes = sorted(host_roles)
    host_set = set(host_nodes)
    host_edges = {
        (int(left), int(right)) for left, right in zip(src, dst)
        if int(left) in host_set and int(right) in host_set and int(left) != int(right)
    }
    incoming = {host: set() for host in host_nodes}
    outgoing = {host: set() for host in host_nodes}
    for left, right in host_edges:
        outgoing[left].add(right)
        incoming[right].add(left)

    records = []
    for host in host_nodes:
        endpoint_nodes = sorted(targets[host])
        counts = np.zeros(ROLE_COUNT, dtype=np.float32)
        multi_role = []
        for endpoint in endpoint_nodes:
            endpoint_roles = roles[endpoint]
            multi_role.append(len(endpoint_roles) >= 2)
            for role in endpoint_roles:
                counts[role] += 1.0
        edge_rows = endpoint_edge_rows[host]
        context = edge_features[edge_rows].mean(axis=0) if edge_rows else np.zeros(7, dtype=np.float32)
        in_degree = len(incoming[host]); out_degree = len(outgoing[host])
        reciprocal = sum(1 for other in outgoing[host] if host in outgoing.get(other, set()))
        records.append({
            "host_node": host,
            "host_id": ids[host],
            "presence": (counts > 0).astype(np.float32),
            "endpoint_counts": counts,
            "unique_endpoint_count": float(len(endpoint_nodes)),
            "role_assignment_count": float(counts.sum()),
            "role_diversity": float(np.count_nonzero(counts) / ROLE_COUNT),
            "role_entropy": normalized_entropy(counts),
            "multi_role_endpoint_ratio": float(np.mean(multi_role)) if multi_role else 0.0,
            "incoming_degree": float(in_degree),
            "outgoing_degree": float(out_degree),
            "direction_balance": float((out_degree - in_degree) / (out_degree + in_degree + 1.0)),
            "reciprocal_ratio": float(reciprocal / max(out_degree, 1)),
            "icc_ratio": float(context[0]),
            "reflection_ratio": float(context[1]),
            "non_direct_ratio": float(context[2]),
        })
    return records


def relative_blocks(records):
    n = len(records)
    if n == 0:
        return np.empty((0, 31), np.float32), np.empty((0, 12), np.float32)
    endpoint_counts = np.stack([r["endpoint_counts"] for r in records]).astype(np.float32)
    multiplicity = np.stack([
        centered_midrank(np.log1p(endpoint_counts[:, role])) for role in range(ROLE_COUNT)
    ], axis=1)
    context_raw = np.asarray([[r[key] for key in (
        "unique_endpoint_count", "role_assignment_count", "role_diversity", "role_entropy",
        "multi_role_endpoint_ratio", "incoming_degree", "outgoing_degree", "direction_balance",
        "reciprocal_ratio", "icc_ratio", "reflection_ratio", "non_direct_ratio",
    )] for r in records], dtype=np.float32)
    context = np.empty_like(context_raw)
    # Count/degree channels use ranks; already-bounded semantic ratios are centered.
    for column in (0, 1, 5, 6):
        context[:, column] = centered_midrank(np.log1p(np.maximum(context_raw[:, column], 0.0)))
    for column in (2, 3, 4, 7, 8, 9, 10, 11):
        context[:, column] = context_raw[:, column] - context_raw[:, column].mean()
    if not np.isfinite(multiplicity).all() or not np.isfinite(context).all():
        raise RuntimeError("non-finite relative Host feature")
    return multiplicity.astype(np.float32), context.astype(np.float32)


def build_arm_values(old_values, records, arm):
    if arm not in ARMS:
        raise KeyError(arm)
    old = np.asarray(old_values, dtype=np.float32)
    if old.shape != (len(records), 74):
        raise RuntimeError(f"Host row mismatch old={old.shape} reconstructed={len(records)}")
    presence = np.stack([r["presence"] for r in records]).astype(np.float32)
    if not np.allclose(old[:, :31], presence, atol=1e-6, rtol=0):
        raise RuntimeError("Host order or payload-to-model role mapping mismatch")
    multiplicity, context = relative_blocks(records)
    if arm == "HM_R":
        result = np.concatenate([presence, multiplicity, old[:, 62:74]], axis=1)
    elif arm == "HC_R":
        result = np.concatenate([old[:, :62], context], axis=1)
    else:
        result = np.concatenate([presence, multiplicity, context], axis=1)
    if result.shape != old.shape or not np.isfinite(result).all():
        raise RuntimeError("Host-V2 feature contract failed")
    return result.astype(np.float32)
