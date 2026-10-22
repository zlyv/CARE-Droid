#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import math
import random
from collections import defaultdict, deque
from typing import Any

import numpy as np

from common import SEED, endpoint_hosts, stable_id
from graph_builder import SINK_TAGS, type_closure
from taxonomy import classify

MAX_DEPTH = 8
DOWNSTREAM_DEPTH = 3
CONTEXT_K = 3
MAX_CONTEXTS_PER_NODE_CATEGORY = 16


def seed_for(sha256: str, variant: str) -> int:
    return int.from_bytes(hashlib.sha256(f"{SEED}:{sha256}:{variant}".encode()).digest()[:8], "big")


def roots(graph: dict[str, Any]) -> list[dict[str, Any]]:
    memo: dict[str, set[str]] = {}
    output = []
    for index, method in enumerate(graph["methods"]):
        rule = classify(type_closure(method["class"], graph["classes"], memo), method["name"], method["descriptor"])
        if rule:
            output.append({"node": index, "category": rule["category"], "family": rule["family"]})
    return output


def adjacency(graph: dict[str, Any], variant: str, sha256: str) -> dict[int, list[tuple[int, int]]]:
    edges = [(int(row[0]), int(row[1]), int(row[3])) for row in graph["edges"]]
    targets = [row[1] for row in edges]
    if variant == "CALLSITE_SHUFFLE":
        # Matched path null: retain every caller out-degree, every callee in-degree,
        # call-site count and call-site token, but destroy the real caller→callee
        # composition by assigning the observed target multiset to other call sites.
        shuffled = list(targets)
        random.Random(seed_for(sha256, variant)).shuffle(shuffled)
        targets = shuffled
    result: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for (source, _, token), target in zip(edges, targets):
        result[source].append((target, token))
    for source in result:
        result[source].sort()
    return dict(result)


def root_variant(raw: list[dict[str, Any]], variant: str, sha256: str) -> list[dict[str, Any]]:
    output = [dict(row) for row in raw]
    if variant == "ROOT_SHUFFLE":
        categories = [row["category"] for row in output]
        random.Random(seed_for(sha256, variant)).shuffle(categories)
        for row, category in zip(output, categories):
            row["category"] = category
    return output


def propagate(graph: dict[str, Any], variant: str) -> dict[str, Any]:
    sha256 = graph["sha256"]
    graph_variant = "REAL" if variant == "ONE_HOP" else variant
    depth_limit = 1 if variant == "ONE_HOP" else MAX_DEPTH
    graph_adj = adjacency(graph, graph_variant, sha256)
    raw_roots = root_variant(roots(graph), graph_variant, sha256)
    cats = ("USER", "AUTO", "NEUTRAL")
    contexts = {cat: defaultdict(set) for cat in cats}
    distance = {cat: {} for cat in cats}
    predecessors = {cat: defaultdict(set) for cat in cats}
    queue = deque()
    for root in sorted(raw_roots, key=lambda row: (row["node"], row["category"], row["family"])):
        cat = root["category"]
        token = stable_id("ROOT:" + root["family"])
        context = (token,)
        if context not in contexts[cat][root["node"]]:
            contexts[cat][root["node"]].add(context); distance[cat][root["node"]] = 0
            queue.append((root["node"], cat, context, 0))
    while queue:
        node, cat, context, depth = queue.popleft()
        if depth >= depth_limit:
            continue
        for target, callsite in graph_adj.get(node, []):
            new_context = (context + (callsite,))[-(CONTEXT_K + 1):]
            bucket = contexts[cat][target]
            if new_context in bucket:
                continue
            if len(bucket) >= MAX_CONTEXTS_PER_NODE_CATEGORY:
                continue
            bucket.add(new_context); predecessors[cat][target].add(node)
            old = distance[cat].get(target)
            distance[cat][target] = depth + 1 if old is None else min(old, depth + 1)
            queue.append((target, cat, new_context, depth + 1))
    return {"roots": raw_roots, "adjacency": graph_adj, "contexts": contexts,
            "depth_limit": depth_limit,
            "distance": distance, "predecessors": predecessors}


def downstream_tags(graph: dict[str, Any], graph_adj: dict[int, list[tuple[int, int]]], start: int) -> set[str]:
    found, seen = set(), {start}; queue = deque([(start, 0)])
    while queue and len(seen) <= 5000:
        node, depth = queue.popleft()
        found.update(graph["methods"][node].get("external_tags", []))
        if depth >= DOWNSTREAM_DEPTH:
            continue
        for target, _ in graph_adj.get(node, []):
            if target not in seen:
                seen.add(target); queue.append((target, depth + 1))
    return found


def entropy(counts: list[int]) -> float:
    values = np.asarray(counts, float); total = float(values.sum())
    if total <= 0:
        return 0.0
    probability = values[values > 0] / total
    return float(-np.sum(probability * np.log(probability)) / math.log(max(2, len(counts))))


def aggregate_variant(graph: dict[str, Any], hosts: list[dict[str, Any]], weights: np.ndarray,
                      critical: int, variant: str, keep_hosts: bool) -> tuple[dict[str, float], list[dict[str, Any]]]:
    state = propagate(graph, variant); contexts, distances, predecessors = state["contexts"], state["distance"], state["predecessors"]
    index_by_id = {int(method["node_id"]): index for index, method in enumerate(graph["methods"])}
    host_rows, derived = [], []
    for order, host in enumerate(hosts):
        node = index_by_id.get(int(host["node_id"]))
        counts = {cat: (len(contexts[cat].get(node, ())) if node is not None else 0) for cat in ("USER", "AUTO", "NEUTRAL")}
        pred = {cat: (set(predecessors[cat].get(node, ())) if node is not None else set()) for cat in ("USER", "AUTO", "NEUTRAL")}
        user, auto, neutral = counts["USER"] > 0, counts["AUTO"] > 0, counts["NEUTRAL"] > 0
        splittable = bool(user and auto and pred["USER"] and pred["AUTO"] and pred["USER"].isdisjoint(pred["AUTO"]))
        tags = downstream_tags(graph, state["adjacency"], node) if node is not None else set()
        total_contexts = sum(counts.values())
        row = {"host_order": order, "host_node_id": int(host["node_id"]), "aligned": int(node is not None),
               "roles": list(host["roles"]), "attention": float(weights[order]), "critical": int(order == critical),
               "user_contexts": counts["USER"], "auto_contexts": counts["AUTO"], "neutral_contexts": counts["NEUTRAL"],
               "user_distance": distances["USER"].get(node) if node is not None else None,
               "auto_distance": distances["AUTO"].get(node) if node is not None else None,
               "neutral_distance": distances["NEUTRAL"].get(node) if node is not None else None,
               "user_predecessors": len(pred["USER"]), "auto_predecessors": len(pred["AUTO"]),
               "user_exclusive": int(user and not auto), "auto_exclusive": int(auto and not user),
               "both": int(user and auto), "splittable": int(splittable),
               "mixed_unresolved": int(user and auto and not splittable),
               "unreachable": int(not user and not auto and not neutral),
               "context_entropy": entropy(list(counts.values())), "downstream_tags": sorted(tags),
               "user_share": counts["USER"] / max(1, total_contexts), "auto_share": counts["AUTO"] / max(1, total_contexts),
               "user_confinement": 1 / max(1, len(pred["USER"])), "auto_confinement": 1 / max(1, len(pred["AUTO"]))}
        derived.append(row)
        if keep_hosts:
            host_rows.append(row)

    features: dict[str, float] = {}
    for prefix, local_weights in (("attn", weights), ("mean", np.full(len(hosts), 1 / max(1, len(hosts))))):
        def avg(name: str) -> float:
            return float(sum(float(w) * float(row[name]) for w, row in zip(local_weights, derived)))
        for name in ("aligned", "user_exclusive", "auto_exclusive", "both", "splittable", "mixed_unresolved", "unreachable",
                     "context_entropy", "user_share", "auto_share", "user_confinement", "auto_confinement"):
            features[f"{prefix}_{name}"] = avg(name)
        features[f"{prefix}_user_reachable"] = float(sum(w for w, row in zip(local_weights, derived) if row["user_contexts"] > 0))
        features[f"{prefix}_auto_reachable"] = float(sum(w for w, row in zip(local_weights, derived) if row["auto_contexts"] > 0))
        features[f"{prefix}_neutral_reachable"] = float(sum(w for w, row in zip(local_weights, derived) if row["neutral_contexts"] > 0))
        features[f"{prefix}_user_inverse_distance"] = float(sum(w / (1 + row["user_distance"]) for w, row in zip(local_weights, derived) if row["user_distance"] is not None))
        features[f"{prefix}_auto_inverse_distance"] = float(sum(w / (1 + row["auto_distance"]) for w, row in zip(local_weights, derived) if row["auto_distance"] is not None))
        for category in ("user", "auto"):
            features[f"{prefix}_{category}_depth_1"] = float(sum(w for w, row in zip(local_weights, derived) if row[f"{category}_distance"] == 1))
            features[f"{prefix}_{category}_depth_2_4"] = float(sum(w for w, row in zip(local_weights, derived) if row[f"{category}_distance"] is not None and 2 <= row[f"{category}_distance"] <= 4))
            features[f"{prefix}_{category}_depth_5_8"] = float(sum(w for w, row in zip(local_weights, derived) if row[f"{category}_distance"] is not None and 5 <= row[f"{category}_distance"] <= 8))
    for role in range(31):
        role_rows = [int(role in row["roles"]) for row in derived]
        features[f"role_{role:02d}_user"] = float(sum(w * present * row["user_share"] for w, present, row in zip(weights, role_rows, derived)))
        features[f"role_{role:02d}_auto"] = float(sum(w * present * row["auto_share"] for w, present, row in zip(weights, role_rows, derived)))
    for tag in SINK_TAGS:
        key = tag.casefold()
        features[f"attn_user_{key}"] = float(sum(w * row["user_share"] for w, row in zip(weights, derived) if tag in row["downstream_tags"]))
        features[f"attn_auto_{key}"] = float(sum(w * row["auto_share"] for w, row in zip(weights, derived) if tag in row["downstream_tags"]))
        features[f"attn_userexclusive_{key}"] = float(sum(w * row["user_exclusive"] for w, row in zip(weights, derived) if tag in row["downstream_tags"]))
        features[f"attn_autoexclusive_{key}"] = float(sum(w * row["auto_exclusive"] for w, row in zip(weights, derived) if tag in row["downstream_tags"]))
    features.update({"graph_method_count": float(graph["stats"]["method_count"]), "graph_edge_count": float(graph["stats"]["edge_count"]),
                     "root_count": float(len(state["roots"])), "host_count": float(len(hosts)),
                     "critical_directional": float(bool(derived) and (derived[critical]["user_exclusive"] or derived[critical]["auto_exclusive"] or derived[critical]["splittable"])) if 0 <= critical < len(derived) else 0.0})
    return features, host_rows


def extract(graph: dict[str, Any], payload: dict[str, Any], attention_info: dict[str, Any] | None) -> dict[str, Any]:
    hosts = endpoint_hosts(payload)
    if hosts:
        if attention_info is None or len(attention_info["attention"]) != len(hosts):
            raise RuntimeError(f"attention/Host mismatch: attention={None if attention_info is None else len(attention_info['attention'])} hosts={len(hosts)}")
        weights = np.asarray(attention_info["attention"], float)
        weights = weights / max(float(weights.sum()), 1e-12)
        critical = int(attention_info["critical"])
    else:
        weights = np.asarray([], float); critical = -1
    variants, host_rows = {}, []
    for variant in ("REAL", "ONE_HOP", "ROOT_SHUFFLE", "CALLSITE_SHUFFLE"):
        features, rows = aggregate_variant(graph, hosts, weights, critical, variant, keep_hosts=variant == "REAL")
        variants[variant] = features
        if rows:
            host_rows = rows
    return {"sha256": graph["sha256"], "host_count": len(hosts), "critical_index": critical,
            "variants": variants, "hosts": host_rows, "schema": {"max_depth": MAX_DEPTH,
            "downstream_depth": DOWNSTREAM_DEPTH, "context_k": CONTEXT_K,
            "max_contexts_per_node_category": MAX_CONTEXTS_PER_NODE_CATEGORY}}
