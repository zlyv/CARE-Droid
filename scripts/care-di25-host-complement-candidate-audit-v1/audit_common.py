#!/usr/bin/env python3
from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import Counter, defaultdict, deque
from pathlib import Path

import numpy as np

ROOT = Path("artifacts/care-di25-host-complement-candidate-audit-v1")
PREFLIGHT = ROOT / "preflight"
FEATURES = ROOT / "features"
PREDICTIONS = ROOT / "predictions"
REPORTS = ROOT / "reports"
AUDIT = ROOT / "audit"
THRESHOLD = 0.5
SEED = 20260920
BOOTSTRAPS = 10000
YEARS = (2018, 2019, 2020, 2021)
FOLDS = (1, 2, 3, 4, 5)


def norm(value):
    return str(value or "").strip().lower()


def read_csv(path):
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_text(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8")


def ensure_dirs():
    for path in (ROOT, PREFLIGHT, FEATURES, PREDICTIONS, REPORTS, AUDIT):
        path.mkdir(parents=True, exist_ok=True)


def hash_files(paths):
    return {str(Path(path)): sha256(Path(path)) for path in paths}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def entropy_from_counts(counts, normalizer):
    values = np.asarray(counts, dtype=float)
    values = values[values > 0]
    if not len(values) or values.sum() <= 0 or normalizer <= 1:
        return 0.0
    probability = values / values.sum()
    return float(-np.sum(probability * np.log(probability)) / np.log(normalizer))


def aggregate(values, prefix):
    array = np.asarray(values, dtype=float)
    if not len(array):
        array = np.asarray([0.0])
    return {
        f"{prefix}__mean": float(np.mean(array)),
        f"{prefix}__median": float(np.median(array)),
        f"{prefix}__std": float(np.std(array)),
        f"{prefix}__max": float(np.max(array)),
    }


def _short_reach(start, adjacency, hosts, max_depth=2):
    reached = set()
    frontier = {start}
    for _ in range(max_depth):
        following = set()
        for node in frontier:
            following.update(adjacency[node])
        following.discard(start)
        reached.update(following)
        frontier = {node for node in following if node not in hosts}
        if not frontier:
            break
    return reached


def reconstruct_frozen_graph(payload, payload_to_model_role):
    node_ids = [str(value) for value in (payload.get("node_ids") or [])]
    base = np.asarray(payload.get("node_base_features") or [], dtype=float)
    if base.size == 0:
        base = np.empty((0, 15), dtype=float)
    raw_role_rows = payload.get("node_role_ids") or []
    edge_index = payload.get("edge_index") or [[], []]
    if (base.ndim != 2 or base.shape[1] != 15 or len(node_ids) != len(base)
            or len(raw_role_rows) != len(base) or len(edge_index) != 2):
        raise RuntimeError("BLOCKED_BY_CONTRACT: frozen endpoint payload schema mismatch")
    role_rows = []
    for values in raw_role_rows:
        roles = []
        for value in values or []:
            index = int(value)
            if not 0 <= index < len(payload_to_model_role):
                raise RuntimeError("BLOCKED_BY_CONTRACT: payload role ID outside frozen taxonomy")
            roles.append(int(payload_to_model_role[index]))
        role_rows.append(tuple(sorted(set(roles))))
    source = [int(value) for value in edge_index[0]]
    target = [int(value) for value in edge_index[1]]
    if len(source) != len(target):
        raise RuntimeError("BLOCKED_BY_CONTRACT: edge rows disagree")
    node_n = len(node_ids)
    edges = {(left, right) for left, right in zip(source, target)
             if 0 <= left < node_n and 0 <= right < node_n and left != right}
    endpoints = set(map(int, np.flatnonzero(base[:, 4] > 0.5)))
    caller_targets = defaultdict(set)
    for left, right in edges:
        if right in endpoints:
            caller_targets[left].add(right)
    host_role_multisets = {}
    for host, targets in caller_targets.items():
        roles = [role for endpoint in sorted(targets) for role in role_rows[endpoint]]
        if roles:
            host_role_multisets[host] = tuple(roles)
    hosts = set(host_role_multisets)
    host_role_sets = {host: tuple(sorted(set(values))) for host, values in host_role_multisets.items()}
    ordinary = set(range(node_n)) - hosts - endpoints
    outgoing = {node: set() for node in range(node_n)}
    incoming = {node: set() for node in range(node_n)}
    for left, right in edges:
        outgoing[left].add(right)
        incoming[right].add(left)
    return {
        "node_ids": node_ids, "node_count": node_n, "edge_count": len(edges), "edges": edges,
        "endpoints": endpoints, "hosts": hosts, "ordinary": ordinary,
        "host_role_multisets": host_role_multisets, "host_role_sets": host_role_sets,
        "outgoing": outgoing, "incoming": incoming,
    }


def extract_candidate_features(graph, role_count):
    hosts = sorted(graph["hosts"])
    host_n = len(hosts)
    role_sets = graph["host_role_sets"]
    role_multisets = graph["host_role_multisets"]
    pair_denom = max(role_count * (role_count - 1) // 2, 1)

    distinct = [len(role_sets[host]) for host in hosts]
    role_entropies = []
    pair_counts = []
    observed_pairs = set()
    signatures = set()
    pair_vector = np.zeros(pair_denom, dtype=float)
    pair_index = {}
    cursor = 0
    for left in range(role_count):
        for right in range(left + 1, role_count):
            pair_index[(left, right)] = cursor
            cursor += 1
    for host in hosts:
        counts = Counter(role_multisets[host])
        role_entropies.append(entropy_from_counts(list(counts.values()), role_count))
        roles = role_sets[host]
        signatures.add(roles)
        local_pairs = [(roles[i], roles[j]) for i in range(len(roles)) for j in range(i + 1, len(roles))]
        pair_counts.append(len(local_pairs))
        for pair in local_pairs:
            observed_pairs.add(pair)
            pair_vector[pair_index[pair]] += 1.0
    if host_n:
        pair_vector /= host_n
    a = {
        "A__multi_role_host_fraction": float(np.mean(np.asarray(distinct) >= 2)) if host_n else 0.0,
        "A__threeplus_role_host_fraction": float(np.mean(np.asarray(distinct) >= 3)) if host_n else 0.0,
        "A__mean_distinct_roles_per_host_norm": float(np.mean(distinct) / role_count) if host_n else 0.0,
        "A__std_distinct_roles_per_host_norm": float(np.std(distinct) / role_count) if host_n else 0.0,
        "A__max_distinct_roles_per_host_norm": float(np.max(distinct) / role_count) if host_n else 0.0,
        "A__mean_role_entropy_per_host": float(np.mean(role_entropies)) if host_n else 0.0,
        "A__max_role_entropy_per_host": float(np.max(role_entropies)) if host_n else 0.0,
        "A__mean_role_pair_count_per_host_norm": float(np.mean(pair_counts) / pair_denom) if host_n else 0.0,
        "A__unique_host_role_set_ratio": len(signatures) / host_n if host_n else 0.0,
        "A__role_pair_diversity": len(observed_pairs) / pair_denom,
    }

    per_host = defaultdict(list)
    structural_types = Counter()
    raw_structure = []
    context_nodes = graph["hosts"] | graph["ordinary"]
    for host in hosts:
        # Structural position intentionally excludes sensitive endpoint nodes.
        # Otherwise every Host has an endpoint successor by definition and a
        # sink-like Host becomes structurally impossible.
        in_all = graph["incoming"][host] & context_nodes
        out_all = graph["outgoing"][host] & context_nodes
        in_host, out_host = in_all & graph["hosts"], out_all & graph["hosts"]
        in_ord, out_ord = in_all & graph["ordinary"], out_all & graph["ordinary"]
        if not in_all and out_all:
            structural_types["source"] += 1
        elif in_all and not out_all:
            structural_types["sink"] += 1
        elif in_all and out_all:
            structural_types["bridge"] += 1
        else:
            structural_types["isolated"] += 1
        upstream = _short_reach(host, graph["incoming"], graph["hosts"], 2) & context_nodes
        downstream = _short_reach(host, graph["outgoing"], graph["hosts"], 2) & context_nodes
        host_in, host_out = len(in_host), len(out_host)
        ord_in, ord_out = len(in_ord), len(out_ord)
        all_in, all_out = len(in_all), len(out_all)
        up_n, down_n = len(upstream), len(downstream)
        per_host["ordinary_degree_total"].append(math.log1p(ord_in + ord_out))
        per_host["host_degree_total"].append(math.log1p(host_in + host_out))
        per_host["all_degree_total"].append(math.log1p(all_in + all_out))
        per_host["host_in_out_asymmetry"].append((host_out - host_in) / (host_out + host_in + 1e-12))
        per_host["ordinary_in_out_asymmetry"].append((ord_out - ord_in) / (ord_out + ord_in + 1e-12))
        per_host["short_reach_total"].append(math.log1p(up_n + down_n))
        per_host["short_reach_asymmetry"].append((down_n - up_n) / (down_n + up_n + 1e-12))
        raw_structure.append((ord_in, ord_out, host_in, host_out, all_in, all_out, up_n, down_n))
    b = {}
    for name in ("ordinary_degree_total", "host_degree_total", "all_degree_total", "host_in_out_asymmetry",
                 "ordinary_in_out_asymmetry", "short_reach_total", "short_reach_asymmetry"):
        b.update(aggregate(per_host[name], f"B__{name}"))
    for name in ("source", "sink", "bridge", "isolated"):
        b[f"B__fraction_{name}_like"] = structural_types[name] / host_n if host_n else 0.0

    relations = {0: set(), 1: set(), 2: set()}
    path_counts = {0: Counter(), 1: Counter(), 2: Counter()}
    for source_host in hosts:
        for target_host in graph["outgoing"][source_host] & graph["hosts"]:
            if source_host != target_host:
                relations[0].add((source_host, target_host)); path_counts[0][(source_host, target_host)] += 1
        for middle in graph["outgoing"][source_host] & graph["ordinary"]:
            for target_host in graph["outgoing"][middle] & graph["hosts"]:
                if source_host != target_host:
                    relations[1].add((source_host, target_host)); path_counts[1][(source_host, target_host)] += 1
            for middle2 in graph["outgoing"][middle] & graph["ordinary"]:
                for target_host in graph["outgoing"][middle2] & graph["hosts"]:
                    if source_host != target_host:
                        relations[2].add((source_host, target_host)); path_counts[2][(source_host, target_host)] += 1
    possible = host_n * max(host_n - 1, 0)
    all_relations = set().union(*relations.values())
    coord_out = {host: set() for host in hosts}
    coord_in = {host: set() for host in hosts}
    for left, right in all_relations:
        coord_out[left].add(right); coord_in[right].add(left)
    transition_counts = {gap: np.zeros((role_count, role_count), dtype=float) for gap in (0, 1, 2)}
    for gap in (0, 1, 2):
        for left, right in relations[gap]:
            for source_role in role_sets[left]:
                for target_role in role_sets[right]:
                    transition_counts[gap][source_role, target_role] += 1.0
    total_relations = sum(len(relations[gap]) for gap in (0, 1, 2))
    transition_total = sum(transition_counts.values())
    c = {
        "C__direct_coordination_density": len(relations[0]) / possible if possible else 0.0,
        "C__gap1_coordination_density": len(relations[1]) / possible if possible else 0.0,
        "C__gap2_coordination_density": len(relations[2]) / possible if possible else 0.0,
        "C__any_coordination_density": len(all_relations) / possible if possible else 0.0,
        "C__gap1_fraction_among_coordination": len(relations[1]) / total_relations if total_relations else 0.0,
        "C__gap2_fraction_among_coordination": len(relations[2]) / total_relations if total_relations else 0.0,
        "C__coordination_out_degree_mean_norm": float(np.mean([len(coord_out[h]) for h in hosts]) / max(host_n - 1, 1)) if host_n else 0.0,
        "C__coordination_out_degree_max_norm": float(max([len(coord_out[h]) for h in hosts], default=0) / max(host_n - 1, 1)),
        "C__coordination_in_degree_mean_norm": float(np.mean([len(coord_in[h]) for h in hosts]) / max(host_n - 1, 1)) if host_n else 0.0,
        "C__coordination_in_degree_max_norm": float(max([len(coord_in[h]) for h in hosts], default=0) / max(host_n - 1, 1)),
        "C__coordination_bridge_host_fraction": float(np.mean([bool(coord_in[h]) and bool(coord_out[h]) for h in hosts])) if host_n else 0.0,
        "C__coordination_source_host_fraction": float(np.mean([not coord_in[h] and bool(coord_out[h]) for h in hosts])) if host_n else 0.0,
        "C__coordination_sink_host_fraction": float(np.mean([bool(coord_in[h]) and not coord_out[h] for h in hosts])) if host_n else 0.0,
        "C__coordination_role_transition_diversity": float(np.count_nonzero(transition_total) / (role_count**2)),
        "C__gap1_role_transition_diversity": float(np.count_nonzero(transition_counts[1]) / (role_count**2)),
        "C__gap2_role_transition_diversity": float(np.count_nonzero(transition_counts[2]) / (role_count**2)),
        "C__coordination_transition_entropy": entropy_from_counts(transition_total.ravel(), role_count**2),
    }
    transition_vector = np.concatenate([transition_counts[gap].ravel() for gap in (0, 1, 2)])
    if total_relations:
        transition_vector /= total_relations
    audit = {
        "host_count": host_n,
        "method_count": graph["node_count"],
        "graph_node_count": graph["node_count"],
        "graph_edge_count": graph["edge_count"],
        "direct_relation_count": len(relations[0]),
        "gap1_relation_count": len(relations[1]),
        "gap2_relation_count": len(relations[2]),
        "any_relation_pair_count": len(all_relations),
        "direct_raw_path_count": sum(path_counts[0].values()),
        "gap1_raw_path_count": sum(path_counts[1].values()),
        "gap2_raw_path_count": sum(path_counts[2].values()),
        "duplicate_coordination_path_count": sum(max(0, count - 1) for gap in path_counts.values() for count in gap.values()),
        "self_coordination_count": sum(left == right for gap in relations.values() for left, right in gap),
        "ordinary_node_count": len(graph["ordinary"]),
        "endpoint_node_count": len(graph["endpoints"]),
        "raw_structure_rows": raw_structure,
    }
    return a, b, c, pair_vector, transition_vector, audit


def binomial_cdf(k, n, p):
    if k < 0: return 0.0
    if k >= n: return 1.0
    return sum(math.comb(n, i) * p**i * (1.0-p)**(n-i) for i in range(k+1))


def _solve_cdf(k, n, target):
    low, high = 0.0, 1.0
    for _ in range(100):
        middle = (low + high) / 2
        if binomial_cdf(k, n, middle) > target: low = middle
        else: high = middle
    return (low + high) / 2


def clopper_pearson(k, n, alpha=0.05):
    if n <= 0: return float("nan"), float("nan")
    low = 0.0 if k == 0 else _solve_cdf(k-1, n, 1-alpha/2)
    high = 1.0 if k == n else _solve_cdf(k, n, alpha/2)
    return low, high


def rankdata(values):
    values = np.asarray(values, float); order = np.argsort(values, kind="mergesort"); ranks = np.empty(len(values), float)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]: end += 1
        ranks[order[start:end]] = (start + 1 + end) / 2
        start = end
    return ranks


def binary_auc(y, score):
    y = np.asarray(y, int); score = np.asarray(score, float); pos = int(np.sum(y == 1)); neg = int(np.sum(y == 0))
    if not pos or not neg: return float("nan")
    ranks = rankdata(score)
    return float((np.sum(ranks[y == 1]) - pos*(pos+1)/2) / (pos*neg))


def average_precision(y, score):
    y = np.asarray(y, int); score = np.asarray(score, float); positives = int(np.sum(y == 1))
    if not positives: return float("nan")
    order = np.argsort(-score, kind="mergesort"); yy = y[order]
    cumulative = np.cumsum(yy)
    return float(np.sum((cumulative / np.arange(1, len(yy)+1)) * yy) / positives)


def classification_metrics(y, probability):
    y = np.asarray(y, int); probability = np.asarray(probability, float); pred = probability >= THRESHOLD
    tp = int(np.sum((y == 1) & pred)); tn = int(np.sum((y == 0) & ~pred)); fp = int(np.sum((y == 0) & pred)); fn = int(np.sum((y == 1) & ~pred))
    recall = tp / (tp + fn) if tp + fn else float("nan"); specificity = tn / (tn + fp) if tn + fp else float("nan")
    precision = tp / (tp + fp) if tp + fp else 0.0
    return {"N": len(y), "positive_n": int(np.sum(y)), "prevalence": float(np.mean(y)), "ROC_AUC": binary_auc(y, probability),
            "PR_AUC": average_precision(y, probability), "BA": (recall + specificity)/2, "F1": 2*precision*recall/(precision+recall) if precision+recall else 0.0,
            "Recall_wrong": recall, "Specificity_correct": specificity, "TN": tn, "FP": fp, "FN": fn, "TP": tp}


def spearman(x, y):
    rx, ry = rankdata(x), rankdata(y)
    if np.std(rx) == 0 or np.std(ry) == 0: return 0.0
    return float(np.corrcoef(rx, ry)[0, 1])
