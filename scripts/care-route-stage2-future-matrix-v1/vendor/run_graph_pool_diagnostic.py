#!/usr/bin/env python3
"""Diagnose the A3 failure using cache-only pooled graph features.

This deliberately removes message passing and trains a regularized logistic
classifier on deterministic graph-level summaries.  It answers three
questions on the same outer split:

1. Does the bounded graph cache contain useful APK-level signal?
2. Is the current two-layer message-passing classifier the bottleneck?
3. Is the main failure raw probability-threshold calibration?

The selected threshold is learned from grouped inner validation only.  The
additional ``0.5`` and training-median metrics are diagnostics; the latter
uses only outer-training scores and is never used to claim a final result.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from run_pilot_compact_eval import fit_logistic, metrics  # noqa: E402
from run_coverage_aware_eval import add_derived_features  # noqa: E402
from tasa_droid.p3.a3_graph_model import build_cached_graph  # noqa: E402
from tasa_droid.p3.a3_mvp import A3_MVP_COMPONENTS, stable_id  # noqa: E402
from tasa_droid.p3.reliability import (  # noqa: E402
    fit_training_fold_components,
    read_cache,
    validate_active_components,
)


AUDIT_COLUMNS = {
    "absolute_path", "certificate_sha256", "error_message", "error_type",
    "lineage_id", "metadata_label", "metadata_year", "package_name",
    "sha256", "source_id", "status", "version_code",
}

EDGE_TYPE_NAMES = (
    "DIRECT_CALL",
    "ICC_COMPONENT_DISPATCH",
    "ICC_CONTENT_PROVIDER",
    "ICC_PENDING_INTENT",
    "REFLECTION_INVOKE",
)
EDGE_TYPE_IDS = np.asarray(
    [stable_id(name) for name in EDGE_TYPE_NAMES], dtype=np.uint64
)

BOUNDARY_NAMES = (
    "INTERNAL_TO_INTERNAL",
    "INTERNAL_TO_EXTERNAL",
    "EXTERNAL_TO_INTERNAL",
    "EXTERNAL_TO_EXTERNAL",
)


def bounded_path_features(graph, max_hops=4) -> np.ndarray:
    """Extract entry-to-tier reachability features from the bounded graph.

    The cache contains only bounded node/edge IDs and compact flags.  This
    computes a small APK-level vector in memory and never persists paths.
    Tier 0/1/2 are treated as sensitive target tiers; tier 3/4 are context.
    """

    x = np.asarray(graph.x, dtype=np.float64)
    n_nodes = len(x)
    if n_nodes == 0:
        return np.zeros(15, dtype=np.float64)
    source = np.asarray(graph.source, dtype=np.int64)
    target = np.asarray(graph.target, dtype=np.int64)
    adjacency = [[] for _ in range(n_nodes)]
    for source_value, target_value in zip(source, target):
        adjacency[int(source_value)].append(int(target_value))
    entry_nodes = np.flatnonzero(x[:, 3] > 0.5).tolist()
    if not entry_nodes:
        return np.zeros(15, dtype=np.float64)

    tier_codes = np.rint(np.clip(x[:, 2], 0.0, 1.0) * 4.0).astype(np.int64)
    per_tier_counts = [[] for _ in range(3)]
    per_tier_hops = [[] for _ in range(3)]
    per_tier_reach = [[] for _ in range(3)]
    per_tier_internal = [[] for _ in range(3)]
    any_sensitive_reach = []
    any_sensitive_count = []
    any_sensitive_internal = []

    for entry in entry_nodes:
        distances = {int(entry): 0}
        frontier = [int(entry)]
        for _hop in range(1, max_hops + 1):
            next_frontier = []
            for node in frontier:
                for neighbor in adjacency[node]:
                    if neighbor not in distances:
                        distances[neighbor] = _hop
                        next_frontier.append(neighbor)
            frontier = next_frontier
            if not frontier:
                break

        sensitive_nodes = []
        for tier in range(3):
            nodes = [
                node for node, distance in distances.items()
                if node != entry and tier_codes[node] == tier
            ]
            count = len(nodes)
            per_tier_counts[tier].append(float(count))
            per_tier_hops[tier].append(
                float(min((distances[node] for node in nodes), default=max_hops + 1))
                / float(max_hops + 1)
            )
            per_tier_reach[tier].append(float(bool(nodes)))
            per_tier_internal[tier].append(
                float(np.mean(x[nodes, 0] < 0.5)) if nodes else 0.0
            )
            sensitive_nodes.extend(nodes)
        any_sensitive_reach.append(float(bool(sensitive_nodes)))
        any_sensitive_count.append(float(len(sensitive_nodes)))
        any_sensitive_internal.append(
            float(np.mean(x[sensitive_nodes, 0] < 0.5))
            if sensitive_nodes else 0.0
        )

    values = []
    for tier in range(3):
        values.extend([
            float(np.log1p(np.mean(per_tier_counts[tier]))),
            float(np.mean(per_tier_hops[tier])),
            float(np.mean(per_tier_reach[tier])),
            float(np.mean(per_tier_internal[tier])),
        ])
    values.extend([
        float(np.mean(any_sensitive_reach)),
        float(np.log1p(np.mean(any_sensitive_count))),
        float(np.mean(any_sensitive_internal)),
    ])
    return np.asarray(values, dtype=np.float64)


def discriminative_node_features(graph, scores) -> np.ndarray:
    """Summarize train-only class-contrast scores without node identities."""

    x = np.asarray(graph.x, dtype=np.float64)
    scores = np.asarray(scores, dtype=np.float64)
    if scores.shape != (len(x),):
        raise ValueError("node contrast scores must align with graph nodes")
    if not len(scores):
        return np.zeros(12, dtype=np.float64)

    positive = np.maximum(scores, 0.0)
    negative = np.maximum(-scores, 0.0)

    def masked_mean(mask):
        return float(scores[mask].mean()) if np.any(mask) else 0.0

    mapped = x[:, 1] > 0.5
    external = x[:, 0] > 0.5
    internal = ~external
    entry = x[:, 3] > 0.5
    return np.asarray([
        float(scores.mean()),
        float(scores.std()),
        float(scores.max()),
        float(scores.min()),
        float(np.mean(scores > 0.0)),
        float(np.mean(scores < 0.0)),
        float(np.log1p(positive.sum())),
        float(np.log1p(negative.sum())),
        masked_mean(mapped),
        masked_mean(external),
        masked_mean(internal),
        masked_mean(entry),
    ], dtype=np.float64)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return [
            {str(key).lstrip("\ufeff"): value for key, value in row.items()}
            for row in csv.DictReader(handle)
        ]


def read_split_rows(path: Path) -> dict[str, dict[str, list[dict]]]:
    result = defaultdict(lambda: {"train": [], "test": []})
    for row in read_csv_rows(path):
        split = row.get("split_name", "")
        partition = row.get("partition", "").casefold()
        sha256 = row.get("sha256", "").casefold()
        if split and partition in {"train", "test"} and sha256:
            result[split][partition].append(
                {"sha256": sha256, "blocking_group": row.get("blocking_group") or sha256}
            )
    return dict(result)


def label_of(row: dict) -> int:
    value = str(row.get("metadata_label", row.get("label", ""))).casefold()
    if value in {"malicious", "malware", "1", "true"}:
        return 1
    if value in {"benign", "0", "false"}:
        return 0
    raise ValueError(f"unknown label: {value!r}")


def group_folds(rows: list[dict], n_splits: int) -> list[list[int]]:
    groups = defaultdict(list)
    positives = defaultdict(int)
    for index, row in enumerate(rows):
        group = str(row["blocking_group"])
        groups[group].append(index)
        positives[group] += label_of(row)
    ordered = sorted(
        groups.items(), key=lambda item: (-len(item[1]), -positives[item[0]], item[0])
    )
    actual = min(max(2, n_splits), len(ordered))
    if actual < 2:
        raise ValueError("not enough blocking groups for inner validation")
    folds = [[] for _ in range(actual)]
    totals = [0] * actual
    positive_totals = [0] * actual
    target_total = len(rows) / actual
    target_positive = sum(positives.values()) / actual
    for fold, (group, indices) in enumerate(ordered[:actual]):
        folds[fold].extend(indices)
        totals[fold] += len(indices)
        positive_totals[fold] += positives[group]
    for group, indices in ordered[actual:]:
        chosen = min(
            range(actual),
            key=lambda fold: (
                abs(positive_totals[fold] + positives[group] - target_positive),
                abs(totals[fold] + len(indices) - target_total),
                fold,
            ),
        )
        folds[chosen].extend(indices)
        totals[chosen] += len(indices)
        positive_totals[chosen] += positives[group]
    if any(not fold for fold in folds):
        raise ValueError("grouped inner split produced an empty fold")
    return folds


def summary_vector(
    graph, graph_summary_mode="full", graph_pool_mode="global",
    role_vocabulary_size=31, node_prevalence_weights=None,
    node_discriminative_scores=None,
) -> np.ndarray:
    """Create a bounded graph-level vector without retaining graph structure."""

    x = np.asarray(graph.x, dtype=np.float64)
    if node_prevalence_weights is None:
        node_weights = np.ones(len(x), dtype=np.float64)
    else:
        node_weights = np.asarray(node_prevalence_weights, dtype=np.float64)
        if node_weights.shape != (len(x),):
            raise ValueError("node prevalence weights must align with graph nodes")
        if np.any(~np.isfinite(node_weights)) or np.any(node_weights <= 0.0):
            raise ValueError("node prevalence weights must be finite and positive")

    def weighted_mean(values, weights):
        if len(values) == 0:
            return np.zeros(values.shape[1], dtype=np.float64)
        return np.average(values, axis=0, weights=weights)

    def weighted_std(values, weights, mean):
        if len(values) == 0:
            return np.zeros(values.shape[1], dtype=np.float64)
        variance = np.average((values - mean) ** 2, axis=0, weights=weights)
        return np.sqrt(np.maximum(variance, 0.0))

    if len(x):
        node_mean = weighted_mean(x, node_weights)
        node_max = x.max(axis=0)
        node_std = weighted_std(x, node_weights, node_mean)
    else:
        node_mean = np.zeros(x.shape[1], dtype=np.float64)
        node_max = np.zeros(x.shape[1], dtype=np.float64)
        node_std = np.zeros(x.shape[1], dtype=np.float64)
    if graph_summary_mode == "full":
        edge_features = np.asarray(graph.edge_features, dtype=np.float64)
        edge_mean = (
            edge_features.mean(axis=0)
            if len(edge_features)
            else np.zeros(3, dtype=np.float64)
        )
    elif graph_summary_mode == "no_edge_summary":
        edge_mean = np.zeros(0, dtype=np.float64)
    else:
        raise ValueError(f"unknown graph_summary_mode: {graph_summary_mode}")
    graph_stats = np.asarray(
        [np.log1p(len(x)), np.log1p(len(graph.source))], dtype=np.float64
    )
    vector = np.concatenate([node_mean, node_max, node_std, edge_mean, graph_stats])
    if graph_pool_mode == "global":
        return vector
    if graph_pool_mode not in {
        "role_stratified", "role_edge_stratified", "role_edge_type_stratified",
        "role_edge_type_boundary_stratified",
        "role_edge_type_boundary_path_stratified",
    }:
        raise ValueError(f"unknown graph_pool_mode: {graph_pool_mode}")

    # x layout is [4 base node features, role one-hot features, optional reliability].
    role_start = 4
    role_end = min(role_start + role_vocabulary_size, x.shape[1])
    role_matrix = x[:, role_start:role_end]
    role_parts = []
    for role_index in range(role_matrix.shape[1]):
        mask = role_matrix[:, role_index] > 0.0
        role_weights = node_weights[mask]
        support = float(role_weights.sum())
        role_parts.append(np.log1p(support))
        if support:
            role_parts.extend(
                weighted_mean(x[mask, :4], role_weights).tolist()
            )
            if graph.reliability_values is not None:
                role_parts.append(float(np.average(
                    graph.reliability_values[mask], weights=role_weights
                )))
        else:
            role_parts.extend([0.0] * 4)
            if graph.reliability_values is not None:
                role_parts.append(0.0)
    role_vector = np.asarray(role_parts, dtype=np.float64)
    if graph_pool_mode == "role_stratified":
        return np.concatenate([vector, role_vector])

    # Cache-only role/edge statistics.  They preserve semantic connectivity
    # without retaining raw method signatures or raw graph objects.
    role_count = role_matrix.shape[1]
    outgoing_count = np.zeros(role_count, dtype=np.float64)
    incoming_count = np.zeros(role_count, dtype=np.float64)
    flagged_outgoing_count = np.zeros(role_count, dtype=np.float64)
    flagged_incoming_count = np.zeros(role_count, dtype=np.float64)
    tier_transitions = np.zeros((5, 5, 4), dtype=np.float64)

    source = np.asarray(graph.source, dtype=np.int64)
    target = np.asarray(graph.target, dtype=np.int64)
    edge_features = np.asarray(graph.edge_features, dtype=np.float64)
    if len(source):
        if edge_features.ndim != 2 or edge_features.shape[1] < 3:
            raise ValueError("cached edge_features must have at least three columns")
        source_roles = role_matrix[source]
        target_roles = role_matrix[target]
        edge_weights = np.sqrt(node_weights[source] * node_weights[target])
        outgoing_count = (source_roles * edge_weights[:, None]).sum(axis=0)
        incoming_count = (target_roles * edge_weights[:, None]).sum(axis=0)
        flagged = (edge_features[:, :3].max(axis=1) > 0.0).astype(np.float64)
        flagged_weight = flagged * edge_weights
        flagged_outgoing_count = (
            source_roles * flagged_weight[:, None]
        ).sum(axis=0)
        flagged_incoming_count = (
            target_roles * flagged_weight[:, None]
        ).sum(axis=0)

        source_tier = np.rint(np.clip(x[source, 2], 0.0, 1.0) * 4.0).astype(np.int64)
        target_tier = np.rint(np.clip(x[target, 2], 0.0, 1.0) * 4.0).astype(np.int64)
        for edge_index, (source_value, target_value) in enumerate(
            zip(source_tier, target_tier)
        ):
            weight = edge_weights[edge_index]
            tier_transitions[source_value, target_value, 0] += weight
            tier_transitions[source_value, target_value, 1:] += edge_features[
                edge_index, :3
            ] * weight
    else:
        edge_weights = np.zeros(0, dtype=np.float64)

    outgoing_flag_ratio = flagged_outgoing_count / np.maximum(outgoing_count, 1.0)
    incoming_flag_ratio = flagged_incoming_count / np.maximum(incoming_count, 1.0)
    role_edge_vector = np.stack(
        [
            np.log1p(outgoing_count),
            np.log1p(incoming_count),
            outgoing_flag_ratio,
            incoming_flag_ratio,
        ],
        axis=1,
    ).reshape(-1)
    tier_transition_vector = np.log1p(tier_transitions).reshape(-1)
    if graph_pool_mode == "role_edge_stratified":
        return np.concatenate(
            [vector, role_vector, role_edge_vector, tier_transition_vector]
        )

    # The cache already stores stable edge-type IDs, but the previous pool
    # collapsed them into only three boolean flags. Add bounded per-type
    # counts, proportions, and sensitive-flag proportions without retaining
    # raw edges or signatures.
    edge_type_vector = []
    edge_type_ids = np.asarray(
        graph.edge_type_ids if graph.edge_type_ids is not None else [],
        dtype=np.uint64,
    )
    total_edges = max(float(edge_weights.sum()), 1.0)
    for edge_type_id in EDGE_TYPE_IDS:
        mask = edge_type_ids == edge_type_id
        count = float(edge_weights[mask].sum())
        edge_type_vector.append(np.log1p(count))
        edge_type_vector.append(count / total_edges)
        if count:
            edge_type_vector.append(
                float(np.average(
                    np.max(edge_features[mask, :3], axis=1),
                    weights=edge_weights[mask],
                ))
            )
        else:
            edge_type_vector.append(0.0)
    edge_type_vector = np.asarray(edge_type_vector, dtype=np.float64)
    if graph_pool_mode == "role_edge_type_stratified":
        return np.concatenate(
            [vector, role_vector, role_edge_vector, tier_transition_vector,
             edge_type_vector]
        )

    # Split edge behavior by whether the source and target are external
    # library/API nodes. This prevents a large amount of library traffic from
    # being treated as equivalent to internal application behavior.
    source_external = x[source, 0] > 0.5 if len(source) else np.zeros(0, dtype=bool)
    target_external = x[target, 0] > 0.5 if len(target) else np.zeros(0, dtype=bool)
    boundary_index = source_external.astype(np.int64) * 2 + target_external.astype(
        np.int64
    )
    boundary_vector = []
    for boundary_id in range(len(BOUNDARY_NAMES)):
        mask = boundary_index == boundary_id
        count = float(edge_weights[mask].sum())
        boundary_vector.append(np.log1p(count))
        boundary_vector.append(count / total_edges)
        if count:
            boundary_vector.append(
                float(np.average(
                    np.max(edge_features[mask, :3], axis=1),
                    weights=edge_weights[mask],
                ))
            )
        else:
            boundary_vector.append(0.0)
    if graph_pool_mode == "role_edge_type_boundary_stratified":
        return np.concatenate(
            [vector, role_vector, role_edge_vector, tier_transition_vector,
             edge_type_vector, np.asarray(boundary_vector, dtype=np.float64)]
        )

    path_vector = bounded_path_features(graph)
    result = np.concatenate(
        [vector, role_vector, role_edge_vector, tier_transition_vector,
         edge_type_vector, np.asarray(boundary_vector, dtype=np.float64),
         path_vector]
    )
    if node_discriminative_scores is not None:
        result = np.concatenate([
            result,
            discriminative_node_features(graph, node_discriminative_scores),
        ])
    return result


def graph_matrix(rows, cache, evidence, args, include_reliability):
    active_components = tuple(
        getattr(args, "reliability_components", A3_MVP_COMPONENTS)
    )
    weights = {name: 1.0 for name in active_components}
    values = []
    labels = []
    prevalence_mode = getattr(args, "graph_prevalence_mode", "none")
    if prevalence_mode not in {"none", "train_idf"}:
        raise ValueError(f"unknown graph prevalence mode: {prevalence_mode}")
    training_count = max(int(evidence.get("training_row_count", 0)), 1)
    document_frequency = evidence.get("document_frequency_by_node_id", {})
    idf_floor = float(getattr(args, "graph_idf_floor", 0.05))
    if not (0.0 < idf_floor <= 1.0):
        raise ValueError("graph IDF floor must be in (0, 1]")
    idf_denominator = max(np.log1p(training_count), 1e-12)
    discriminative_mode = getattr(args, "graph_discriminative_mode", "none")
    if discriminative_mode not in {"none", "train_smoothed_log_odds"}:
        raise ValueError(
            f"unknown graph discriminative mode: {discriminative_mode}"
        )
    contrast_alpha = float(getattr(args, "graph_contrast_alpha", 2.0))
    contrast_kappa = float(getattr(args, "graph_contrast_kappa", 10.0))
    contrast_clip = float(getattr(args, "graph_contrast_clip", 4.0))
    if contrast_alpha <= 0.0 or contrast_kappa <= 0.0 or contrast_clip <= 0.0:
        raise ValueError("graph contrast parameters must be positive")
    class_df = evidence.get("class_document_frequency_by_node_id", {})
    class_count = evidence.get("training_class_row_count", {})
    benign_count = max(int(class_count.get("benign", 0)), 1)
    malicious_count = max(int(class_count.get("malicious", 0)), 1)

    def node_log_odds(node_id):
        counts = class_df.get(str(node_id))
        if not counts:
            return 0.0
        benign_df = float(counts.get("benign", 0))
        malicious_df = float(counts.get("malicious", 0))
        benign_rate = (benign_df + contrast_alpha) / (
            benign_count + 2.0 * contrast_alpha
        )
        malicious_rate = (malicious_df + contrast_alpha) / (
            malicious_count + 2.0 * contrast_alpha
        )
        benign_logit = np.log(benign_rate / max(1.0 - benign_rate, 1e-12))
        malicious_logit = np.log(
            malicious_rate / max(1.0 - malicious_rate, 1e-12)
        )
        shrinkage = (benign_df + malicious_df) / (
            benign_df + malicious_df + contrast_kappa
        )
        return float(np.clip(
            (malicious_logit - benign_logit) * shrinkage,
            -contrast_clip,
            contrast_clip,
        ))
    for row in rows:
        cache_row = cache[row["sha256"]]
        payload = cache_row["payload"]
        if prevalence_mode == "train_idf":
            node_prevalence_weights = np.asarray([
                max(
                    idf_floor,
                    np.log(
                        (training_count + 1.0)
                        / (float(document_frequency.get(str(node_id), 0)) + 1.0)
                    ) / idf_denominator,
                )
                for node_id in (payload.get("node_ids") or [])
            ], dtype=np.float64)
        else:
            node_prevalence_weights = None
        if discriminative_mode == "train_smoothed_log_odds":
            node_ids = payload.get("node_ids") or []
            base = payload.get("node_base_features") or []
            roles = payload.get("node_role_ids") or []
            node_discriminative_scores = np.asarray([
                node_log_odds(node_id)
                if (bool(role_ids) or bool(base_features[0]))
                else 0.0
                for node_id, base_features, role_ids in zip(node_ids, base, roles)
            ], dtype=np.float64)
        else:
            node_discriminative_scores = None
        graph = build_cached_graph(
            payload,
            evidence["components_by_node_id"],
            evidence["global_training_prior"],
            weights,
            args.r_min,
            args.r_max,
            args.role_vocabulary_size,
            include_reliability=include_reliability,
            reliability_components=active_components,
        )
        if args.graph_edge_mode == "direct_only" and len(graph.source):
            edge_features = np.asarray(graph.edge_features, dtype=np.float64)
            direct_mask = edge_features[:, 2] == 0.0
            graph.source = graph.source[direct_mask]
            graph.target = graph.target[direct_mask]
            graph.edge_features = edge_features[direct_mask]
            if graph.edge_type_ids is not None:
                graph.edge_type_ids = graph.edge_type_ids[direct_mask]
        values.append(
            summary_vector(
                graph,
                args.graph_summary_mode,
                args.graph_pool_mode,
                args.role_vocabulary_size,
                node_prevalence_weights=node_prevalence_weights,
                node_discriminative_scores=node_discriminative_scores,
            )
        )
        labels.append(label_of(cache_row))
    return np.asarray(values, dtype=np.float64), np.asarray(labels, dtype=np.int64)


def compact_matrix(
    train_rows, eval_rows, feature_by_sha, schema, feature_mode="base"
):
    base_names = [
        name for name in schema.get("feature_fields", []) if name not in AUDIT_COLUMNS
    ]
    fit_table = [
        add_derived_features(feature_by_sha[row["sha256"]], base_names, feature_mode)
        for row in train_rows
    ]
    eval_table = [
        add_derived_features(feature_by_sha[row["sha256"]], base_names, feature_mode)
        for row in eval_rows
    ]
    candidates = sorted(fit_table[0]) if fit_table else []
    names = []
    for name in candidates:
        try:
            values = np.asarray([float(row[name]) for row in fit_table])
        except (KeyError, TypeError, ValueError):
            continue
        if np.all(np.isfinite(values)) and np.unique(values).size > 1:
            names.append(name)
    x_fit = np.asarray([[float(row[name]) for name in names] for row in fit_table])
    x_eval = np.asarray([[float(row[name]) for name in names] for row in eval_table])
    mean = x_fit.mean(axis=0)
    scale = x_fit.std(axis=0)
    scale[scale < 1e-12] = 1.0
    return (x_fit - mean) / scale, (x_eval - mean) / scale, len(names)


def standardize_pair(x_fit, x_eval):
    mean = x_fit.mean(axis=0)
    scale = x_fit.std(axis=0)
    scale[scale < 1e-12] = 1.0
    return (x_fit - mean) / scale, (x_eval - mean) / scale


def evaluate_variant(
    variant_name, include_reliability, fuse_compact, train_rows, test_rows,
    cache, feature_by_sha, schema, args, graph_weight=1.0,
):
    folds = group_folds(train_rows, args.inner_folds)
    threshold_scores = {
        threshold: {"accuracy": [], "balanced_accuracy": []}
        for threshold in args.thresholds
    }
    for fold_index, validation_indices in enumerate(folds):
        validation_set = set(validation_indices)
        fit_rows = [row for i, row in enumerate(train_rows) if i not in validation_set]
        validation_rows = [train_rows[i] for i in validation_indices]
        evidence = fit_training_fold_components(
            cache,
            {row["sha256"] for row in fit_rows},
            args.definition,
            active_components=args.reliability_components,
        )
        x_fit, y_fit = graph_matrix(
            fit_rows, cache, evidence, args, include_reliability
        )
        x_validation, y_validation = graph_matrix(
            validation_rows, cache, evidence, args, include_reliability
        )
        if fuse_compact:
            c_fit, c_validation, _feature_count = compact_matrix(
                fit_rows, validation_rows, feature_by_sha, schema,
                feature_mode=args.compact_feature_mode,
            )
            x_fit, x_validation = standardize_pair(x_fit, x_validation)
            x_fit = np.concatenate([graph_weight * x_fit, c_fit], axis=1)
            x_validation = np.concatenate(
                [graph_weight * x_validation, c_validation], axis=1
            )
        else:
            x_fit, x_validation = standardize_pair(x_fit, x_validation)
        weights, bias = fit_logistic(
            x_fit, y_fit, args.selection_seed + fold_index,
            iterations=args.iterations,
            learning_rate=args.learning_rate,
            l2=args.l2,
        )
        probability = 1.0 / (1.0 + np.exp(-np.clip(x_validation @ weights + bias, -40, 40)))
        for threshold in threshold_scores:
            result = metrics(y_validation, probability, threshold=threshold)
            threshold_scores[threshold]["accuracy"].append(result["accuracy"])
            threshold_scores[threshold]["balanced_accuracy"].append(result["balanced_accuracy"])
    selected_threshold = max(
        threshold_scores,
        key=lambda threshold: (
            float(np.mean(threshold_scores[threshold]["balanced_accuracy"])),
            float(np.mean(threshold_scores[threshold]["accuracy"])),
            -abs(threshold - 0.5),
            -threshold,
        ),
    )

    evidence = fit_training_fold_components(
        cache,
        {row["sha256"] for row in train_rows},
        args.definition,
        active_components=args.reliability_components,
    )
    x_train, y_train = graph_matrix(
        train_rows, cache, evidence, args, include_reliability
    )
    x_test, y_test = graph_matrix(
        test_rows, cache, evidence, args, include_reliability
    )
    feature_count = 0
    if fuse_compact:
        c_train, c_test, feature_count = compact_matrix(
            train_rows, test_rows, feature_by_sha, schema,
            feature_mode=args.compact_feature_mode,
        )
        x_train, x_test = standardize_pair(x_train, x_test)
        x_train = np.concatenate([graph_weight * x_train, c_train], axis=1)
        x_test = np.concatenate([graph_weight * x_test, c_test], axis=1)
    else:
        x_train, x_test = standardize_pair(x_train, x_test)

    seed_results = []
    for seed in args.seeds:
        weights, bias = fit_logistic(
            x_train, y_train, seed,
            iterations=args.iterations,
            learning_rate=args.learning_rate,
            l2=args.l2,
        )
        train_probability = 1.0 / (1.0 + np.exp(-np.clip(x_train @ weights + bias, -40, 40)))
        test_probability = 1.0 / (1.0 + np.exp(-np.clip(x_test @ weights + bias, -40, 40)))
        train_median = float(np.median(train_probability))
        seed_results.append(
            {
                "selected_threshold": metrics(y_test, test_probability, selected_threshold),
                "threshold_0_5": metrics(y_test, test_probability, 0.5),
                "training_median_threshold": metrics(y_test, test_probability, train_median),
                "train_median_threshold": train_median,
                "test_probability_min": float(test_probability.min()),
                "test_probability_max": float(test_probability.max()),
                "test_probability_mean": float(test_probability.mean()),
            }
        )
    def mean_metric(key, metric):
        values = [row[key][metric] for row in seed_results]
        return float(np.mean(values))

    summary = {
        "variant": variant_name,
        "graph_weight": float(graph_weight),
        "train_count": len(train_rows),
        "test_count": len(test_rows),
        "feature_count": int(x_train.shape[1]),
        "compact_feature_count": feature_count,
        "selected_threshold": float(selected_threshold),
        "inner_threshold_scores": {
            str(threshold): {
                "accuracy": float(np.mean(value["accuracy"])),
                "balanced_accuracy": float(np.mean(value["balanced_accuracy"])),
            }
            for threshold, value in threshold_scores.items()
        },
        "selected_threshold_accuracy_mean": mean_metric("selected_threshold", "accuracy"),
        "selected_threshold_balanced_accuracy_mean": mean_metric("selected_threshold", "balanced_accuracy"),
        "threshold_0_5_accuracy_mean": mean_metric("threshold_0_5", "accuracy"),
        "threshold_0_5_balanced_accuracy_mean": mean_metric("threshold_0_5", "balanced_accuracy"),
        "training_median_accuracy_mean": mean_metric("training_median_threshold", "accuracy"),
        "training_median_balanced_accuracy_mean": mean_metric("training_median_threshold", "balanced_accuracy"),
        "training_median_threshold_mean": float(np.mean([row["train_median_threshold"] for row in seed_results])),
        "auc_mean": float(np.mean([row["selected_threshold"]["auc"] for row in seed_results])),
        "test_probability_min_mean": float(np.mean([row["test_probability_min"] for row in seed_results])),
        "test_probability_max_mean": float(np.mean([row["test_probability_max"] for row in seed_results])),
        "test_probability_mean": float(np.mean([row["test_probability_mean"] for row in seed_results])),
        "test_rows_used_for_training_or_selection": False,
    }
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", required=True, type=Path)
    parser.add_argument("--features", required=True, type=Path)
    parser.add_argument("--schema", required=True, type=Path)
    parser.add_argument("--splits", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--split", action="append", dest="split_names")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--inner-folds", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=1500)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--l2", type=float, default=1e-3)
    parser.add_argument("--thresholds", nargs="+", type=float, default=[round(0.05 + i * 0.05, 2) for i in range(19)])
    parser.add_argument("--selection-seed", type=int, default=20260806)
    parser.add_argument("--alpha-prior", type=float, default=1.0)
    parser.add_argument("--kappa-support", type=float, default=20.0)
    parser.add_argument("--kappa-context", type=float, default=20.0)
    parser.add_argument("--r-min", type=float, default=0.10)
    parser.add_argument("--r-max", type=float, default=0.95)
    parser.add_argument("--role-vocabulary-size", type=int, default=31)
    parser.add_argument(
        "--graph-weights",
        nargs="+",
        type=float,
        default=[1.0],
        help="graph-block weights for fused variants; choose by inner validation",
    )
    parser.add_argument(
        "--reliability-components",
        nargs="+",
        choices=list(A3_MVP_COMPONENTS),
        default=list(A3_MVP_COMPONENTS),
    )
    parser.add_argument(
        "--compact-feature-mode",
        choices=[
            "base", "coverage", "coverage_interactions",
            "size_density_additive", "size_density_only",
        ],
        default="base",
        help="compact feature set used by fused variants; derived values are fit-filtered on training rows",
    )
    parser.add_argument(
        "--graph-summary-mode",
        choices=["full", "no_edge_summary"],
        default="full",
        help="whether pooled graph summaries include the three edge-feature means",
    )
    parser.add_argument(
        "--graph-edge-mode",
        choices=["all", "direct_only"],
        default="all",
        help="retain all cached edges or only direct-call edges for message passing",
    )
    parser.add_argument(
        "--graph-pool-mode",
        choices=[
            "global", "role_stratified", "role_edge_stratified",
            "role_edge_type_stratified", "role_edge_type_boundary_stratified",
            "role_edge_type_boundary_path_stratified",
        ],
        default="global",
        help="global, role-stratified, or role-and-edge-stratified pooling",
    )
    args = parser.parse_args()
    args.reliability_components = validate_active_components(args.reliability_components)

    from tasa_droid.p3.reliability import PilotReliabilityDefinition

    args.definition = PilotReliabilityDefinition(
        alpha_prior=args.alpha_prior,
        kappa_support=args.kappa_support,
        kappa_context=args.kappa_context,
        r_min=args.r_min,
        r_max=args.r_max,
    )
    args.definition.validate()
    paths = {
        "cache": args.cache.expanduser().resolve(strict=True),
        "features": args.features.expanduser().resolve(strict=True),
        "schema": args.schema.expanduser().resolve(strict=True),
        "splits": args.splits.expanduser().resolve(strict=True),
    }
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    cache = read_cache(paths["cache"])
    feature_rows = read_csv_rows(paths["features"])
    feature_by_sha = {row["sha256"].casefold(): row for row in feature_rows}
    schema = json.loads(paths["schema"].read_text(encoding="utf-8"))
    split_rows = read_split_rows(paths["splits"])
    if args.split_names:
        unknown = set(args.split_names) - set(split_rows)
        if unknown:
            raise SystemExit(f"unknown split names: {sorted(unknown)}")
        split_rows = {name: split_rows[name] for name in args.split_names}
    summaries = []
    for split_name in sorted(split_rows):
        train_rows = split_rows[split_name]["train"]
        test_rows = split_rows[split_name]["test"]
        for row in train_rows + test_rows:
            if row["sha256"] not in cache or row["sha256"] not in feature_by_sha:
                raise SystemExit(f"missing cache or compact row: {row['sha256']}")
            row["metadata_label"] = cache[row["sha256"]].get("metadata_label", "")
        for name, include_reliability, fuse_compact in (
            ("graph_pool_reliability_logistic", True, False),
            ("graph_pool_no_reliability_logistic", False, False),
            ("graph_pool_compact_no_reliability_logistic", False, True),
            ("graph_pool_compact_fused_logistic", True, True),
        ):
            weights = args.graph_weights if fuse_compact else [1.0]
            for graph_weight in weights:
                variant_name = (
                    f"{name}__gw{graph_weight:g}" if fuse_compact else name
                )
                print(f"running {variant_name} on {split_name}", flush=True)
                summary = evaluate_variant(
                    variant_name, include_reliability, fuse_compact,
                    train_rows, test_rows, cache, feature_by_sha, schema, args,
                    graph_weight=graph_weight,
                )
                summary["split_name"] = split_name
                summaries.append(summary)
    audit = {
        "schema_version": "1.0.0",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "phase": "P3_GRAPH_POOL_CLASSIFIER_DIAGNOSTIC",
        "formal_training_enabled": False,
        "strict_test_isolation": True,
        "cache": str(paths["cache"]),
        "cache_sha256": sha256_file(paths["cache"]),
        "features": str(paths["features"]),
        "features_sha256": sha256_file(paths["features"]),
        "schema": str(paths["schema"]),
        "schema_sha256": sha256_file(paths["schema"]),
        "split_manifest": str(paths["splits"]),
        "split_manifest_sha256": sha256_file(paths["splits"]),
        "seeds": list(args.seeds),
        "inner_folds": args.inner_folds,
        "iterations": args.iterations,
        "learning_rate": args.learning_rate,
        "l2": args.l2,
        "thresholds": list(args.thresholds),
        "graph_weights": list(args.graph_weights),
        "compact_feature_mode": args.compact_feature_mode,
        "graph_summary_mode": args.graph_summary_mode,
        "graph_edge_mode": args.graph_edge_mode,
        "graph_pool_mode": args.graph_pool_mode,
        "reliability_components": list(args.reliability_components),
        "summaries": summaries,
        "outputs": {"audit": str(output / "graph_pool_diagnostic_audit.json")},
        "blocking_reasons": [
            "pilot_reliability_definition_not_resolved_by_nested_validation",
            "formal_training_config_not_resolved",
            "coverage_aware_feature_mode_is_exploratory_and_test_error_informed",
            "edge_summary_ablation_is_diagnostic_not_a_locked_protocol",
            "direct_only_edge_ablation_is_diagnostic_not_a_locked_protocol",
            "role_stratified_pooling_is_an_exploratory_representation_extension",
        ],
        "notes": [
            "Graph summaries contain pooled bounded-cache statistics only; raw graphs are not persisted.",
            "Selected thresholds use grouped inner validation only.",
            "Threshold 0.5 and training-median results are diagnostics, not model-selection results.",
            "Coverage-derived compact features are an exploratory extension and are not locked by this run.",
            "This diagnostic is not a final formal result.",
        ],
    }
    path = output / "graph_pool_diagnostic_audit.json"
    path.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
