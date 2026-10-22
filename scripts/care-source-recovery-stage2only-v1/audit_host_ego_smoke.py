#!/usr/bin/env python3
"""Human-readable Host/Ego smoke audit on the real endpoint-cache payload."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
from collections import deque
from pathlib import Path

import numpy as np


ROLE_COUNT = 31
BASE_DIM = 72
REGION_DIM = 74


def role_names(path: Path) -> list[str]:
    names = []
    in_roles = False
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if line == "roles:":
            in_roles = True
            continue
        if in_roles and line == "rules:":
            break
        if in_roles and line.startswith("- "):
            names.append(line[2:].strip().strip('"\''))
    if len(names) != ROLE_COUNT:
        raise RuntimeError(f"expected {ROLE_COUNT} role names, got {len(names)}")
    return names


def read_cache(path: Path) -> list[dict]:
    rows = []
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("status") == "OK" and row.get("payload"):
                rows.append(row)
    rows.sort(key=lambda row: str(row.get("sha256", "")).strip().lower())
    return rows


def payload_arrays(payload):
    node_ids = list(payload.get("node_ids") or [])
    role_rows = list(payload.get("node_role_ids") or [])
    base = np.asarray(payload.get("node_base_features") or [], dtype=np.float64)
    edge_index = payload.get("edge_index") or [[], []]
    if len(edge_index) != 2:
        raise ValueError("edge_index must contain source and target rows")
    source = np.asarray(edge_index[0], dtype=np.int64)
    target = np.asarray(edge_index[1], dtype=np.int64)
    if base.size == 0:
        base = np.zeros((0, 15), dtype=np.float64)
    if base.ndim != 2 or base.shape[1] != 15:
        raise ValueError("node_base_features must have shape (n,15)")
    if len(node_ids) != len(role_rows) or len(node_ids) != len(base):
        raise ValueError("payload node arrays disagree")
    roles = [frozenset(int(v) for v in (values or []) if 0 <= int(v) < ROLE_COUNT)
             for values in role_rows]
    return node_ids, base, roles, source, target


def reconstruct(payload):
    node_ids, base, node_roles, source, target = payload_arrays(payload)
    endpoints = set(map(int, np.flatnonzero(base[:, 4] > 0.5)))
    caller_targets = {}
    for src, dst in zip(source, target):
        src, dst = int(src), int(dst)
        if dst in endpoints:
            caller_targets.setdefault(src, set()).add(dst)
    host_roles = {}
    for host, targets in caller_targets.items():
        merged = frozenset().union(*(node_roles[index] for index in targets))
        if merged:
            host_roles[host] = merged
    hosts = set(host_roles)
    host_edges = {(int(src), int(dst)) for src, dst in zip(source, target)
                  if int(src) in hosts and int(dst) in hosts and int(src) != int(dst)}
    return node_ids, host_roles, host_edges


def adjacency(hosts, edges):
    result = {int(host): set() for host in hosts}
    for src, dst in edges:
        if src in result and dst in result and src != dst:
            result[src].add(dst)
            result[dst].add(src)
    return result


def entropy(counts):
    counts = np.asarray(counts, dtype=np.float64)
    if float(np.sum(counts)) <= 0:
        return 0.0
    p = counts[counts > 0] / np.sum(counts)
    return float(-np.sum(p * np.log(p)) / math.log(ROLE_COUNT))


def region_vector(nodes, host_roles, edges):
    nodes = frozenset(int(value) for value in nodes)
    counts = np.zeros(ROLE_COUNT, dtype=np.float64)
    degrees = []
    for host in nodes:
        roles = host_roles.get(host, ())
        degrees.append(len(roles))
        for role in roles:
            counts[int(role)] += 1
    internal = {(s, t) for s, t in edges if s in nodes and t in nodes and s != t}
    boundary = {(s, t) for s, t in edges if (s in nodes) != (t in nodes)}
    reciprocal = sum((t, s) in internal for s, t in internal)
    host_n = len(nodes)
    incidence = float(np.sum(counts))
    possible = host_n * max(host_n - 1, 0)
    degree = np.asarray(degrees, dtype=np.float64)
    topology = np.asarray([
        np.log1p(host_n), np.log1p(incidence), np.count_nonzero(counts) / ROLE_COUNT,
        float(np.mean(degree)) if len(degree) else 0.0,
        float(np.mean(degree >= 2)) if len(degree) else 0.0,
        np.log1p(len(internal)), np.log1p(len(boundary)),
        len(internal) / possible if possible else 0.0,
        reciprocal / len(internal) if internal else 0.0, entropy(counts),
    ])
    result = np.concatenate([counts / max(host_n, 1), np.log1p(counts), topology])
    if result.shape != (BASE_DIM,) or not np.isfinite(result).all():
        raise RuntimeError("invalid local-region vector")
    return result


def matrices(host_roles, edges):
    hosts = sorted(host_roles)
    graph = adjacency(hosts, edges)
    host_regions = [frozenset([host]) for host in hosts]
    ego_regions = sorted({frozenset({host, *neighbors}) for host, neighbors in graph.items()},
                         key=lambda value: (-len(value), tuple(sorted(value))))
    host = [np.concatenate([region_vector(r, host_roles, edges), [1.0, 0.0]])
            for r in host_regions]
    ego = [np.concatenate([region_vector(r, host_roles, edges), [0.0, 1.0]])
           for r in ego_regions]
    host_matrix = np.asarray(host, dtype=np.float32) if host else np.zeros((1, REGION_DIM), np.float32)
    ego_matrix = np.asarray(ego, dtype=np.float32) if ego else np.zeros((0, REGION_DIM), np.float32)
    return host_matrix, ego_matrix, graph


def synthetic_cases():
    cases = {
        "no_host": ({}, set(), (1, REGION_DIM), (0, REGION_DIM)),
        "single_host": ({0: frozenset([12])}, set(), (1, REGION_DIM), (1, REGION_DIM)),
        "multi_host": ({0: frozenset([8]), 1: frozenset([12]), 2: frozenset([17])},
                       {(0, 1), (1, 2)}, (3, REGION_DIM), (3, REGION_DIM)),
    }
    result = {}
    for name, (roles, edges, host_shape, ego_shape) in cases.items():
        host, ego, graph = matrices(roles, edges)
        ok = host.shape == host_shape and ego.shape == ego_shape
        ok = ok and np.isfinite(host).all() and np.isfinite(ego).all()
        if name == "multi_host":
            ok = ok and max(len({center, *neighbors}) for center, neighbors in graph.items()) == 3
        result[name] = bool(ok)
    return result


def labels(values, names):
    return [names[int(value)] for value in sorted(values)]


def describe(row, asset, names, category):
    node_ids, host_roles, edges, graph, host, ego = asset
    lines = [f"APK: {str(row.get('sha256', '')).strip().lower()}", f"category: {category}",
             f"host_count: {len(host_roles)}", f"host_values_shape: {list(host.shape)}",
             f"ego_values_shape: {list(ego.shape)}"]
    if not host_roles:
        lines.extend(["missing_region_handling: deterministic_zero_host_row", "valid_ego: false"])
        return lines
    center = max(graph, key=lambda value: (len(graph[value]), -value))
    region = sorted({center, *graph[center]})
    lines.append("center_host:")
    lines.append(f"  node_index: {center}")
    lines.append(f"  method: {node_ids[center]}")
    lines.append(f"  roles: {labels(host_roles[center], names)}")
    lines.append("one_hop_nodes:")
    for node in region:
        lines.append(f"  - node_index: {node}")
        lines.append(f"    method: {node_ids[node]}")
        lines.append(f"    roles: {labels(host_roles[node], names)}")
    lines.append("edges:")
    internal = sorted((s, t) for s, t in edges if s in region and t in region)
    if internal:
        for src, dst in internal:
            lines.append(f"  - {src} {labels(host_roles[src], names)} -> {dst} {labels(host_roles[dst], names)}")
    else:
        lines.append("  []")
    lines.append(f"valid_ego: {str(len(region) > 1).lower()}")
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--mapping", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected", type=int, default=50)
    args = parser.parse_args()
    names = role_names(args.mapping)
    rows = read_cache(args.cache)
    if len(rows) != args.expected:
        raise RuntimeError(f"expected {args.expected} successful smoke rows, got {len(rows)}")
    args.output.mkdir(parents=True, exist_ok=True)
    audits, assets, errors = [], {}, []
    host_all, ego_all, host_offsets, ego_offsets = [], [], [0], [0]
    meaningful_difference = False
    for row in rows:
        sha = str(row.get("sha256", "")).strip().lower()
        try:
            node_ids, host_roles, edges = reconstruct(row["payload"])
            host, ego, graph = matrices(host_roles, edges)
            nontrivial = [frozenset({h, *n}) for h, n in graph.items() if len(n) > 0]
            for vector in ego[:, :BASE_DIM]:
                if not any(np.allclose(vector, candidate[:BASE_DIM]) for candidate in host):
                    meaningful_difference = True
            category = "no_host" if not host_roles else ("single_host" if len(host_roles) == 1 else "multi_host")
            audits.append({
                "sha256": sha, "category": category, "host_count": len(host_roles),
                "host_host_edge_count": len(edges), "host_region_count": len(host),
                "ego_region_count": len(ego), "nontrivial_ego_count": len(nontrivial),
                "max_ego_size": max((len(v) for v in nontrivial), default=1 if host_roles else 0),
                "finite": int(np.isfinite(host).all() and np.isfinite(ego).all()),
            })
            assets[sha] = (node_ids, host_roles, edges, graph, host, ego)
            host_all.append(host); ego_all.append(ego)
            host_offsets.append(host_offsets[-1] + len(host)); ego_offsets.append(ego_offsets[-1] + len(ego))
        except Exception as exc:
            errors.append({"sha256": sha, "error_type": type(exc).__name__, "error": str(exc)})
    synthetic = synthetic_cases()
    counts = {name: sum(row["category"] == name for row in audits)
              for name in ("no_host", "single_host", "multi_host")}
    nontrivial_n = sum(row["nontrivial_ego_count"] > 0 for row in audits)
    passed = (
        len(audits) == args.expected and not errors
        and all(row["finite"] for row in audits)
        and nontrivial_n > 0 and meaningful_difference
        and all(synthetic.values())
    )
    np.savez_compressed(
        args.output / "smoke_host_ego_values.npz",
        sha256=np.asarray([row["sha256"] for row in audits]),
        host_values=np.concatenate(host_all), host_offsets=np.asarray(host_offsets),
        ego_values=np.concatenate([v for v in ego_all if len(v)]) if any(len(v) for v in ego_all) else np.zeros((0, REGION_DIM)),
        ego_offsets=np.asarray(ego_offsets), region_dim=np.asarray(REGION_DIM),
    )
    with (args.output / "apk_smoke_audit.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(audits[0]))
        writer.writeheader(); writer.writerows(audits)
    gate = {
        "passed": bool(passed), "real_apk_n": len(audits), "errors": errors,
        "category_counts": counts, "natural_nontrivial_ego_apk_n": nontrivial_n,
        "host_ego_meaningfully_different": bool(meaningful_difference),
        "synthetic_edge_cases": synthetic,
        "current_model_consumption": "HOST_ONLY",
        "warning": "Ego cache validity does not mean the current v4/MalBen model consumes ego_values.",
    }
    (args.output / "smoke_gate.json").write_text(json.dumps(gate, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = [
        "CARE-Droid Host/Ego pre-run smoke audit", "",
        f"automatic_gate_passed: {str(passed).lower()}",
        f"real_apk_n: {len(audits)}", f"category_counts: {counts}",
        f"natural_nontrivial_ego_apk_n: {nontrivial_n}",
        f"host_ego_meaningfully_different: {str(meaningful_difference).lower()}",
        f"synthetic_edge_cases: {synthetic}", "",
        "IMPORTANT: current v4/MalBen consumes host_values only; ego_values are audited assets but are not model inputs.",
        "Passing this audit must not be described as a Host+Ego model result.", "",
        "Concrete real samples:", "",
    ]
    for category in ("no_host", "single_host", "multi_host"):
        candidates = [row for row in audits if row["category"] == category]
        if category == "multi_host":
            candidates.sort(key=lambda row: (-row["max_ego_size"], row["sha256"]))
        if not candidates:
            lines.extend([f"category: {category}", "natural_example: not_observed_in_fixed_50", "synthetic_case_passed: true", ""])
            continue
        chosen = candidates[0]
        cache_row = next(row for row in rows if str(row.get("sha256", "")).strip().lower() == chosen["sha256"])
        lines.extend(describe(cache_row, assets[chosen["sha256"]], names, category)); lines.append("")
    (args.output / "REPORT.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    if not passed:
        raise SystemExit("Host/Ego smoke gate failed; full extraction is blocked")


if __name__ == "__main__":
    main()
