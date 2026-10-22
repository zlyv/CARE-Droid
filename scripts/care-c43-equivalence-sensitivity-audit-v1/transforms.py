#!/usr/bin/env python3
from __future__ import annotations

import copy
import hashlib
from typing import Any

import numpy as np


def _rng(sid: str, name: str, repeat: int):
    raw = hashlib.sha256(f"{sid}|{name}|{repeat}|20261012".encode()).digest()
    return np.random.default_rng(int.from_bytes(raw[:8], "little"))


def _stable_number(*parts) -> int:
    raw = hashlib.sha256("|".join(map(str, parts)).encode()).digest()
    return int.from_bytes(raw[:8], "little")


def _validate(payload: dict[str, Any]) -> tuple[int, int]:
    node_ids = list(payload.get("node_ids") or [])
    roles = list(payload.get("node_role_ids") or [])
    base = list(payload.get("node_base_features") or [])
    edge = payload.get("edge_index") or [[], []]
    if len(edge) != 2 or len(node_ids) != len(roles) or len(node_ids) != len(base):
        raise RuntimeError("endpoint payload schema mismatch")
    e = len(edge[0])
    if len(edge[1]) != e:
        raise RuntimeError("edge source/target mismatch")
    for key in ("edge_features", "edge_type_ids"):
        value = payload.get(key)
        if value is not None and len(value) not in (0, e):
            raise RuntimeError(f"{key} length mismatch")
    return len(node_ids), e


def _ensure_optional_arrays(payload: dict[str, Any]) -> None:
    """Materialize optional empty arrays used by sparse/zero-edge endpoint rows."""
    payload.setdefault("node_ids", [])
    payload.setdefault("node_role_ids", [])
    payload.setdefault("node_base_features", [])
    payload.setdefault("edge_index", [[], []])


def _node_fields(payload, n):
    result = []
    for key, value in payload.items():
        if key.startswith("node_") and isinstance(value, list) and len(value) == n:
            result.append(key)
    return result


def _edge_fields(payload, e):
    result = []
    for key, value in payload.items():
        if key.startswith("edge_") and key != "edge_index" and isinstance(value, list) and len(value) == e:
            result.append(key)
    return result


def _neutral_node(payload, source, new_id):
    n, _ = _validate(payload); values = {}
    for key in _node_fields(payload, n):
        if key == "node_ids": values[key] = int(new_id)
        elif key == "node_role_ids": values[key] = []
        elif key == "node_base_features":
            row = list(copy.deepcopy(payload[key][source]))
            if len(row) > 4: row[4] = 0.0
            values[key] = row
        else: values[key] = copy.deepcopy(payload[key][source])
    return values


def _append_node(payload, source, new_id):
    n, _ = _validate(payload); values = _neutral_node(payload, source, new_id)
    for key in _node_fields(payload, n): payload[key].append(values[key])
    return n


def _edge_value(payload, key, source_edge=None):
    values = payload.get(key) or []
    if source_edge is not None and source_edge < len(values): return copy.deepcopy(values[source_edge])
    if values:
        sample = values[0]
        if isinstance(sample, list): return [0 for _ in sample]
        if isinstance(sample, (int, float)): return type(sample)(0)
        return copy.deepcopy(sample)
    return None


def _append_edge(payload, source, target, source_edge=None):
    _, e = _validate(payload); fields = _edge_fields(payload, e)
    payload["edge_index"][0].append(int(source)); payload["edge_index"][1].append(int(target))
    for key in fields: payload[key].append(_edge_value(payload, key, source_edge))


def _remove_edges(payload, indices):
    _, e = _validate(payload); fields = _edge_fields(payload, e); drop = set(indices)
    keep = [i for i in range(e) if i not in drop]
    payload["edge_index"] = [[payload["edge_index"][0][i] for i in keep], [payload["edge_index"][1][i] for i in keep]]
    for key in fields: payload[key] = [payload[key][i] for i in keep]


def endpoint_nodes(payload):
    return {i for i, row in enumerate(payload.get("node_base_features") or []) if len(row) > 4 and float(row[4]) > .5}


def host_edges(payload):
    endpoints = endpoint_nodes(payload)
    edge = payload.get("edge_index") or [[], []]
    return [i for i, target in enumerate(edge[1]) if int(target) in endpoints]


def order_permutation(payload, sid, repeat=0):
    out = copy.deepcopy(payload); _ensure_optional_arrays(out); n, e = _validate(out); rng = _rng(sid, "T0_ORDER", repeat)
    if n == 0:
        return out, {"edited_nodes": 0, "edited_edges": 0, "applicable": False, "reason": "empty_node_graph"}
    permutation = rng.permutation(n); inverse = np.empty(n, int); inverse[permutation] = np.arange(n)
    for key in _node_fields(out, n): out[key] = [out[key][int(old)] for old in permutation]
    out["edge_index"] = [[int(inverse[int(x)]) for x in out["edge_index"][0]], [int(inverse[int(x)]) for x in out["edge_index"][1]]]
    # Also permute edge storage; graph semantics remain identical.
    edge_perm = rng.permutation(e)
    out["edge_index"] = [[out["edge_index"][0][int(i)] for i in edge_perm], [out["edge_index"][1][int(i)] for i in edge_perm]]
    for key in _edge_fields(out, e): out[key] = [out[key][int(i)] for i in edge_perm]
    return out, {"edited_nodes": n, "edited_edges": e, "applicable": True}


def wrapper(payload, sid, repeat=0, fraction=.20, sensitive=True):
    out = copy.deepcopy(payload); _ensure_optional_arrays(out); n, e = _validate(out); rng = _rng(sid, "T1" if sensitive else "N1", repeat)
    candidates = host_edges(out) if sensitive else [i for i in range(e) if i not in set(host_edges(out))]
    if not candidates: return out, {"edited_nodes": 0, "edited_edges": 0, "applicable": False}
    count = max(1, int(round(len(candidates) * fraction))); chosen = sorted(rng.choice(candidates, size=min(count, len(candidates)), replace=False).tolist())
    original = [(i, int(out["edge_index"][0][i]), int(out["edge_index"][1][i])) for i in chosen]
    _remove_edges(out, chosen)
    for serial, (edge_index, source, target) in enumerate(original):
        new = _append_node(out, source, 8_000_000_000_000_000_000 + (_stable_number(sid, repeat, sensitive, serial) % 900_000_000_000_000_000))
        _append_edge(out, source, new, None); _append_edge(out, new, target, None)
    return out, {"edited_nodes": len(original), "edited_edges": 2 * len(original), "replaced_edges": len(original), "applicable": True}


def dead_subgraph(payload, sid, repeat=0, fraction=.05):
    out = copy.deepcopy(payload); _ensure_optional_arrays(out); n, _ = _validate(out); rng = _rng(sid, "T2", repeat)
    if n == 0:
        return out, {"edited_nodes": 0, "edited_edges": 0, "applicable": False, "reason": "empty_node_graph"}
    count = max(2, min(32, int(round(max(n, 1) * fraction))))
    added = []
    for serial in range(count):
        source = int(rng.integers(0, max(n, 1))) if n else 0
        added.append(_append_node(out, source, 7_000_000_000_000_000_000 + serial + repeat * 1000))
    for a, b in zip(added[:-1], added[1:]): _append_edge(out, a, b, None)
    return out, {"edited_nodes": len(added), "edited_edges": max(0, len(added)-1), "applicable": True}


def dead_edge_near_host(payload, sid, repeat=0, fraction=.20):
    out = copy.deepcopy(payload); _ensure_optional_arrays(out); n, _ = _validate(out); rng = _rng(sid, "T3", repeat)
    hosts = sorted({int(out["edge_index"][0][i]) for i in host_edges(out)})
    if not hosts: return out, {"edited_nodes": 0, "edited_edges": 0, "applicable": False}
    count = max(1, int(round(len(hosts) * fraction))); chosen = rng.choice(hosts, size=min(count, len(hosts)), replace=False)
    for serial, source in enumerate(chosen):
        new = _append_node(out, int(source), 6_000_000_000_000_000_000 + serial + repeat * 1000)
        _append_edge(out, int(source), new, None)
    return out, {"edited_nodes": len(chosen), "edited_edges": len(chosen), "applicable": True}


def apply(payload, sid, name, repeat=0):
    if name == "T0_ORDER": return order_permutation(payload, sid, repeat)
    if name == "T1_WRAPPER_SENSITIVE": return wrapper(payload, sid, repeat, sensitive=True)
    if name == "T2_DEAD_SUBGRAPH": return dead_subgraph(payload, sid, repeat)
    if name == "T3_DEAD_EDGE_NEAR_HOST": return dead_edge_near_host(payload, sid, repeat)
    if name == "N1_WRAPPER_FAR": return wrapper(payload, sid, repeat, sensitive=False)
    raise KeyError(name)


def apply_record(record, sid, name, repeat=0):
    """Apply a graph transform without discarding the cache record envelope."""
    if not isinstance(record, dict):
        raise RuntimeError("endpoint cache record is not a mapping")
    if isinstance(record.get("payload"), dict):
        output = copy.deepcopy(record)
        output["payload"], audit = apply(record["payload"], sid, name, repeat)
        return output, {**audit, "record_container": "payload"}
    if any(key in record for key in ("node_ids", "node_base_features", "edge_index")):
        output, audit = apply(record, sid, name, repeat)
        return output, {**audit, "record_container": "direct"}
    raise RuntimeError("endpoint cache record has neither payload nor graph fields")
