#!/usr/bin/env python3
"""Fixed-width directed one-hop semantic compression for Host-only CARE."""
from __future__ import annotations

from collections import defaultdict
import numpy as np

ROLE_COUNT = 31


def reconstruct_host_relations(payload, payload_to_model_role):
    """Return Host records in the canonical sorted-node order.

    A Host is a caller with at least one retained mapped sensitive endpoint.
    Host-to-Host edges are raw retained call edges; no path contraction is used.
    """
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
    if len(edge_index) != 2:
        raise RuntimeError("endpoint edge_index schema mismatch")
    src = np.asarray(edge_index[0], dtype=np.int64)
    dst = np.asarray(edge_index[1], dtype=np.int64)
    if base.ndim != 2 or base.shape[1] != 15 or len(ids) != len(base) or len(roles) != len(base):
        raise RuntimeError("endpoint node schema mismatch")
    if len(src) != len(dst):
        raise RuntimeError("endpoint edge length mismatch")

    endpoints = set(map(int, np.flatnonzero(base[:, 4] > 0.5)))
    targets = defaultdict(set)
    for left, right in zip(src, dst):
        left, right = int(left), int(right)
        if right in endpoints:
            targets[left].add(right)

    host_counts = {}
    for host, endpoint_nodes in targets.items():
        counts = np.zeros(ROLE_COUNT, dtype=np.float32)
        for endpoint in endpoint_nodes:
            for role in roles[endpoint]:
                counts[role] += 1.0
        if np.any(counts > 0):
            host_counts[int(host)] = counts

    host_nodes = sorted(host_counts)
    host_set = set(host_nodes)
    incoming = {host: set() for host in host_nodes}
    outgoing = {host: set() for host in host_nodes}
    for left, right in zip(src, dst):
        left, right = int(left), int(right)
        if left in host_set and right in host_set and left != right:
            outgoing[left].add(right)
            incoming[right].add(left)

    normalized_roles = {}
    for host, counts in host_counts.items():
        total = float(counts.sum())
        normalized_roles[host] = counts / total if total > 0 else np.zeros(ROLE_COUNT, np.float32)

    records = []
    for host in host_nodes:
        inc = np.sum([normalized_roles[n] for n in incoming[host]], axis=0) if incoming[host] else np.zeros(ROLE_COUNT, np.float32)
        out = np.sum([normalized_roles[n] for n in outgoing[host]], axis=0) if outgoing[host] else np.zeros(ROLE_COUNT, np.float32)
        # Positive = outgoing-to-role; negative = incoming-from-role.
        relation = np.tanh(np.log1p(out) - np.log1p(inc)).astype(np.float32)
        records.append({
            "host_node": host,
            "host_id": ids[host],
            "endpoint_counts": host_counts[host],
            "presence": (host_counts[host] > 0).astype(np.float32),
            "incoming_role_mass": inc.astype(np.float32),
            "outgoing_role_mass": out.astype(np.float32),
            "directed_neighbor_role": relation,
            "incoming_neighbor_count": len(incoming[host]),
            "outgoing_neighbor_count": len(outgoing[host]),
        })
    return records


def build_hostrel74(old_values, records):
    old = np.asarray(old_values, dtype=np.float32)
    if old.shape != (len(records), 74):
        raise RuntimeError(f"Host row mismatch old={old.shape} reconstructed={len(records)}")
    presence = np.stack([r["presence"] for r in records]).astype(np.float32)
    relation = np.stack([r["directed_neighbor_role"] for r in records]).astype(np.float32)
    if not np.allclose(old[:, :31], presence, atol=1e-6, rtol=0):
        raise RuntimeError("Host order or role mapping mismatch")
    result = np.concatenate([presence, relation, old[:, 62:74]], axis=1)
    if result.shape != old.shape or not np.isfinite(result).all():
        raise RuntimeError("HostRel74 contract failed")
    return result.astype(np.float32)
