"""A3-MVP in-memory graph tensor contract.

The builder consumes the normalized graph objects while they are still in
memory. It emits stable hashed node IDs, bounded edge indices, semantic role
IDs, and reliability-scaled node features. Raw method signatures are never
returned by this public tensor object.
"""

from __future__ import annotations

import hashlib
import json
from typing import Mapping, Sequence


A3_MVP_COMPONENTS = ("D", "T", "one_minus_F")


def stable_id(value: str) -> int:
    """Return a deterministic unsigned 64-bit ID without persisting the value."""

    digest = hashlib.sha256(str(value).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False)


def compute_reliability(
    components: Mapping[str, float],
    weights: Mapping[str, float],
    r_min: float,
    r_max: float,
    active_components: Sequence[str] | None = None,
) -> float:
    """Compute reliability from the explicitly selected protocol components."""

    names = tuple(A3_MVP_COMPONENTS if active_components is None else active_components)
    if not names or len(set(names)) != len(names):
        raise ValueError("active reliability components must be non-empty and unique")
    unknown = [name for name in names if name not in A3_MVP_COMPONENTS]
    if unknown:
        raise KeyError(f"unknown A3-MVP reliability components: {unknown}")
    missing = [name for name in names if name not in components]
    if missing:
        raise KeyError(f"missing A3-MVP reliability components: {missing}")
    missing_weights = [name for name in names if name not in weights]
    if missing_weights:
        raise KeyError(f"missing A3-MVP reliability weights: {missing_weights}")
    if not (0.0 <= r_min <= r_max <= 1.0):
        raise ValueError("reliability bounds must satisfy 0 <= r_min <= r_max <= 1")
    total_weight = sum(float(weights[name]) for name in names)
    if total_weight <= 0.0 or any(float(weights[name]) < 0.0 for name in names):
        raise ValueError("A3-MVP weights must be nonnegative and have positive sum")
    raw = sum(
        float(components[name]) * float(weights[name])
        for name in names
    ) / total_weight
    return max(r_min, min(r_max, raw))


def _tensor_hash(payload: dict) -> str:
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def build_a3_mvp_structure(
    node_records: Mapping[str, Mapping],
    edge_records: Sequence[Mapping],
    entry_signatures: set[str],
    active_node_signatures: set[str],
    role_vocabulary: Sequence[str],
) -> dict:
    """Build the unscaled A3-MVP structure before training-fold reliability.

    This is intentionally separate from ``build_a3_mvp_tensor``.  It is safe
    to cache or audit because it contains only hashed IDs and bounded indexes;
    D/T/one-minus-F are attached later using training-fold evidence.
    """

    role_to_id = {role: index for index, role in enumerate(sorted(role_vocabulary))}
    active = sorted(
        signature for signature in active_node_signatures if signature in node_records
    )
    node_index = {signature: index for index, signature in enumerate(active)}
    payload = {
        "node_ids": [stable_id(signature) for signature in active],
        "node_role_ids": [
            [role_to_id[role] for role in node_records[signature].get("roles", []) if role in role_to_id]
            for signature in active
        ],
        "node_base_features": [
            [
                float(bool(node_records[signature].get("external"))),
                float(bool(node_records[signature].get("roles"))),
                float(node_records[signature].get("tier", 4)) / 4.0,
                float(signature in entry_signatures),
            ]
            for signature in active
        ],
        "edge_index": [[], []],
        "edge_features": [],
        "edge_type_ids": [],
    }
    for edge in edge_records:
        source = edge.get("source")
        target = edge.get("target")
        if source not in node_index or target not in node_index:
            continue
        payload["edge_index"][0].append(node_index[source])
        payload["edge_index"][1].append(node_index[target])
        payload["edge_features"].append(
            [
                float(bool(edge.get("icc_any"))),
                float(bool(edge.get("reflection_any"))),
                float(edge.get("edge_type", "DIRECT_CALL") != "DIRECT_CALL"),
            ]
        )
        payload["edge_type_ids"].append(stable_id(edge.get("edge_type", "DIRECT_CALL")))
    return {
        "schema_version": "1.0.0",
        "view": "A3_MVP_UNSCALED_STRUCTURE",
        "node_count": len(payload["node_ids"]),
        "edge_count": len(payload["edge_type_ids"]),
        "role_vocabulary_size": len(role_to_id),
        "base_node_feature_count": 4,
        "edge_feature_count": 3,
        "reliability_status": "REQUIRES_TRAINING_FOLD_COMPONENTS",
        "tensor_sha256": _tensor_hash(payload),
        "payload": payload,
    }


def build_a3_mvp_tensor(
    node_records: Mapping[str, Mapping],
    edge_records: Sequence[Mapping],
    entry_signatures: set[str],
    active_node_signatures: set[str],
    role_vocabulary: Sequence[str],
    reliability_by_signature: Mapping[str, Mapping[str, float]],
    reliability_weights: Mapping[str, float],
    r_min: float,
    r_max: float,
) -> dict:
    """Build an in-memory A3-MVP tensor object from one bounded graph view.

    ``active_node_signatures`` must already be bounded by the P2 path policy.
    The input maps contain raw signatures only transiently; the returned object
    contains hashed IDs and integer indexes instead.
    """

    role_to_id = {role: index for index, role in enumerate(sorted(role_vocabulary))}
    active = sorted(
        signature for signature in active_node_signatures if signature in node_records
    )
    node_index = {signature: index for index, signature in enumerate(active)}
    node_ids = [stable_id(signature) for signature in active]
    node_role_ids = []
    node_features = []
    reliabilities = []
    for signature in active:
        record = node_records[signature]
        roles = [role_to_id[role] for role in record.get("roles", []) if role in role_to_id]
        components = reliability_by_signature.get(signature)
        if components is None:
            raise KeyError(f"missing reliability record for node: {signature}")
        reliability = compute_reliability(
            components, reliability_weights, r_min, r_max
        )
        node_role_ids.append(roles)
        node_features.append(
            [
                float(bool(record.get("external"))),
                float(bool(record.get("roles"))),
                float(record.get("tier", 4)) / 4.0,
                float(signature in entry_signatures),
                reliability,
            ]
        )
        reliabilities.append(reliability)

    edge_index = [[], []]
    edge_features = []
    edge_type_ids = []
    for edge in edge_records:
        source = edge.get("source")
        target = edge.get("target")
        if source not in node_index or target not in node_index:
            continue
        edge_index[0].append(node_index[source])
        edge_index[1].append(node_index[target])
        edge_features.append(
            [
                float(bool(edge.get("icc_any"))),
                float(bool(edge.get("reflection_any"))),
                float(edge.get("edge_type", "DIRECT_CALL") != "DIRECT_CALL"),
            ]
        )
        edge_type_ids.append(stable_id(edge.get("edge_type", "DIRECT_CALL")))

    payload = {
        "node_ids": node_ids,
        "node_role_ids": node_role_ids,
        "node_features": node_features,
        "edge_index": edge_index,
        "edge_features": edge_features,
        "edge_type_ids": edge_type_ids,
    }
    return {
        "schema_version": "1.0.0",
        "view": "A3_MVP",
        "node_count": len(node_ids),
        "edge_count": len(edge_type_ids),
        "role_vocabulary_size": len(role_to_id),
        "reliability_components": list(A3_MVP_COMPONENTS),
        "reliability_min": min(reliabilities) if reliabilities else None,
        "reliability_max": max(reliabilities) if reliabilities else None,
        "reliability_mean": (
            sum(reliabilities) / len(reliabilities) if reliabilities else None
        ),
        "tensor_sha256": _tensor_hash(payload),
        "payload": payload,
    }


def build_a3_mvp_tensor_from_cached_payload(
    cached_payload: Mapping,
    reliability_by_node_id: Mapping[str, Mapping[str, float]],
    reliability_weights: Mapping[str, float],
    r_min: float,
    r_max: float,
) -> dict:
    """Attach training-fold reliability to a cached bounded payload.

    This is the no-signature path used after the APK cache has been built.
    Missing node components raise instead of silently assigning a default.
    The returned payload contains only hashed IDs, role IDs, indexes, compact
    features, and reliability values.
    """

    node_ids = list(cached_payload.get("node_ids") or [])
    base_features = list(cached_payload.get("node_base_features") or [])
    if len(node_ids) != len(base_features):
        raise ValueError("cached node_ids and node_base_features length mismatch")
    node_features = []
    reliabilities = []
    for node_id, base in zip(node_ids, base_features):
        key = str(node_id)
        components = reliability_by_node_id.get(key)
        if components is None:
            raise KeyError(f"missing reliability record for cached node: {key}")
        reliability = compute_reliability(
            components, reliability_weights, r_min, r_max
        )
        node_features.append([float(value) for value in base] + [reliability])
        reliabilities.append(reliability)

    payload = {
        "node_ids": node_ids,
        "node_role_ids": list(cached_payload.get("node_role_ids") or []),
        "node_features": node_features,
        "edge_index": [
            list(cached_payload.get("edge_index", [[], []])[0]),
            list(cached_payload.get("edge_index", [[], []])[1]),
        ],
        "edge_features": list(cached_payload.get("edge_features") or []),
        "edge_type_ids": list(cached_payload.get("edge_type_ids") or []),
    }
    return {
        "schema_version": "1.0.0",
        "view": "A3_MVP_FROM_CACHED_PAYLOAD",
        "node_count": len(node_ids),
        "edge_count": len(payload["edge_type_ids"]),
        "reliability_components": list(A3_MVP_COMPONENTS),
        "reliability_min": min(reliabilities) if reliabilities else None,
        "reliability_max": max(reliabilities) if reliabilities else None,
        "reliability_mean": (
            sum(reliabilities) / len(reliabilities) if reliabilities else None
        ),
        "tensor_sha256": _tensor_hash(payload),
        "payload": payload,
    }
