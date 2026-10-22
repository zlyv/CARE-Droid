#!/usr/bin/env python3
"""Evaluate an entry-independent sensitive-behaviour combination hypergraph.

All selected sensitive endpoint methods are mapped to semantic roles.  Normal
methods are not represented as hypergraph vertices.  Two bounded hyperedge
relations are constructed from the existing endpoint cache:

* shared_caller: roles of sensitive endpoints invoked by one caller method;
* compressed_reachability: roles connected after ordinary intermediate
  methods are collapsed, stopping at the next sensitive endpoint.

The representation records high-order role sets and hypergraph topology, not
raw method identities or ordinary paths.  Entry context remains optional and
is evaluated only through the separate existing entry-view representation.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from run_dual_view_2017_rolling_eval import (  # noqa: E402
    ACTIVE_RELIABILITY_COMPONENTS,
    ROLE_VOCABULARY_SIZE,
    attach_audit_metadata,
    entry_model_args,
    sha256_file,
)
from run_endpoint_motif_2017_eval import _payload_arrays  # noqa: E402
from run_graph_pool_diagnostic import (  # noqa: E402
    compact_matrix,
    graph_matrix,
    read_csv_rows,
    read_split_rows,
    standardize_pair,
)
from run_pilot_compact_eval import metrics  # noqa: E402
from run_uncertainty_gated_dual_view_2017_eval import predict_logistic  # noqa: E402
from tasa_droid.p3.reliability import (  # noqa: E402
    PilotReliabilityDefinition,
    fit_training_fold_components,
    read_cache,
)


THRESHOLD = 0.5
ENTRY_GRAPH_WEIGHT = 0.5
MAX_COMPRESSED_HOPS = 4
PATH_STATE_CAP = 20000
ORDER_BUCKETS = 5  # 2, 3, 4, 5, >=6
SPECTRAL_WIDTH = 8
ROLE_WIDTH = 3
RELATION_GLOBAL_WIDTH = ORDER_BUCKETS + 8 + SPECTRAL_WIDTH + 6
RELATION_DIMENSION = ROLE_VOCABULARY_SIZE * ROLE_WIDTH + RELATION_GLOBAL_WIDTH
HYPERGRAPH_DIMENSION = 2 * RELATION_DIMENSION


def _roles(values) -> frozenset[int]:
    return frozenset(
        int(value) for value in (values or [])
        if 0 <= int(value) < ROLE_VOCABULARY_SIZE
    )


def _entropy(values: np.ndarray) -> float:
    total = float(values.sum())
    if total <= 0.0:
        return 0.0
    probabilities = values[values > 0.0] / total
    return float(-(probabilities * np.log(probabilities)).sum())


def _components(adjacency: np.ndarray) -> int:
    active = np.flatnonzero(adjacency.sum(axis=1) > 0.0)
    unseen = set(map(int, active))
    count = 0
    while unseen:
        count += 1
        stack = [unseen.pop()]
        while stack:
            node = stack.pop()
            neighbors = set(map(int, np.flatnonzero(adjacency[node] > 0.0)))
            new = neighbors & unseen
            unseen -= new
            stack.extend(new)
    return count


def _relation_vector(
    hyperedges: list[tuple[frozenset[int], int]],
) -> np.ndarray:
    """Summarize high-order incidence and topology for one relation type."""

    role_degree = np.zeros(ROLE_VOCABULARY_SIZE, dtype=np.float64)
    high_order_degree = np.zeros(ROLE_VOCABULARY_SIZE, dtype=np.float64)
    role_order_sum = np.zeros(ROLE_VOCABULARY_SIZE, dtype=np.float64)
    cooccurrence = np.zeros(
        (ROLE_VOCABULARY_SIZE, ROLE_VOCABULARY_SIZE), dtype=np.float64
    )
    order_hist = np.zeros(ORDER_BUCKETS, dtype=np.float64)
    provenance = np.zeros(3, dtype=np.float64)
    pattern_counts: Counter[tuple[int, ...]] = Counter()

    for roles, provenance_id in hyperedges:
        if len(roles) < 2:
            continue
        order = len(roles)
        order_hist[min(order, 6) - 2] += 1.0
        provenance[int(np.clip(provenance_id, 0, 2))] += 1.0
        pattern_counts[tuple(sorted(roles))] += 1
        role_list = sorted(roles)
        for role in role_list:
            role_degree[role] += 1.0
            role_order_sum[role] += order
            if order >= 3:
                high_order_degree[role] += 1.0
        for left in role_list:
            for right in role_list:
                if left != right:
                    cooccurrence[left, right] += 1.0

    mean_order_by_role = role_order_sum / np.maximum(role_degree, 1.0)
    role_vector = np.stack([
        np.log1p(role_degree),
        np.log1p(high_order_degree),
        mean_order_by_role / float(ROLE_VOCABULARY_SIZE),
    ], axis=1).reshape(-1)

    degree = cooccurrence.sum(axis=1)
    normalizer = np.sqrt(np.outer(np.maximum(degree, 1.0), np.maximum(degree, 1.0)))
    normalized = cooccurrence / normalizer
    eigenvalues = np.linalg.eigvalsh((normalized + normalized.T) / 2.0)
    spectral = np.zeros(SPECTRAL_WIDTH, dtype=np.float64)
    selected = np.sort(np.abs(eigenvalues))[::-1][:SPECTRAL_WIDTH]
    spectral[:len(selected)] = selected
    binary_adjacency = (cooccurrence > 0.0).astype(np.float64)
    possible_pairs = ROLE_VOCABULARY_SIZE * (ROLE_VOCABULARY_SIZE - 1)
    pattern_values = np.asarray(list(pattern_counts.values()), dtype=np.float64)
    edge_orders = np.asarray([len(roles) for roles, _ in hyperedges], dtype=np.float64)
    total_hyperedges = float(len(hyperedges))
    global_vector = np.concatenate([
        np.log1p(order_hist),
        np.asarray([
            np.log1p(total_hyperedges),
            np.log1p(len(pattern_counts)),
            float(edge_orders.mean()) if len(edge_orders) else 0.0,
            float(edge_orders.max() / ROLE_VOCABULARY_SIZE) if len(edge_orders) else 0.0,
            _entropy(pattern_values),
            *(
                provenance / max(provenance.sum(), 1.0)
            ).tolist(),
        ], dtype=np.float64),
        spectral,
        np.asarray([
            float(binary_adjacency.sum() / max(possible_pairs, 1)),
            float(_components(binary_adjacency) / ROLE_VOCABULARY_SIZE),
            _entropy(np.maximum(eigenvalues, 0.0)),
            float(np.mean(degree)),
            float(np.std(degree)),
            float(np.max(degree)) if len(degree) else 0.0,
        ], dtype=np.float64),
    ])
    result = np.concatenate([role_vector, global_vector])
    if result.shape != (RELATION_DIMENSION,):
        raise ValueError(
            f"unexpected relation dimension {result.shape}; "
            f"expected {(RELATION_DIMENSION,)}"
        )
    return result


def sensitive_behavior_hypergraph_vector(payload: dict) -> tuple[np.ndarray, dict]:
    node_ids, base, role_rows, source, target = _payload_arrays(payload)
    endpoint_mask = base[:, 4] > 0.5
    endpoint_nodes = set(map(int, np.flatnonzero(endpoint_mask)))
    node_roles = [_roles(values) for values in role_rows]
    provenance = (
        np.argmax(base[:, 5:8], axis=1)
        if len(base) else np.zeros(0, dtype=np.int64)
    )
    outgoing: dict[int, list[int]] = defaultdict(list)
    caller_targets: dict[int, set[int]] = defaultdict(set)
    for src_value, dst_value in zip(source, target):
        src = int(src_value)
        dst = int(dst_value)
        outgoing[src].append(dst)
        if dst in endpoint_nodes:
            caller_targets[src].add(dst)

    # A behavior-bearing host is an application/framework method that directly
    # invokes at least one selected sensitive endpoint.  Sensitive API methods
    # are terminal call targets in the usual call-graph direction, so the
    # compressed relation must traverse between their hosts, not start from the
    # endpoint API vertices themselves.
    host_roles = {
        caller: frozenset().union(*(node_roles[index] for index in endpoints))
        for caller, endpoints in caller_targets.items()
    }
    host_roles = {caller: roles for caller, roles in host_roles.items() if roles}

    shared_caller_hyperedges = []
    for caller, roles in host_roles.items():
        if len(roles) >= 2:
            shared_caller_hyperedges.append((roles, int(provenance[caller])))

    compressed_hyperedges = []
    path_states = 0
    path_state_cap_hit = False
    for start in sorted(host_roles):
        reached_hosts = set()
        stack = [(start, 0, frozenset({start}))]
        while stack and not path_state_cap_hit:
            node, distance, visited = stack.pop()
            if distance >= MAX_COMPRESSED_HOPS:
                continue
            for neighbor in outgoing.get(node, []):
                if neighbor in visited:
                    continue
                path_states += 1
                if path_states > PATH_STATE_CAP:
                    path_state_cap_hit = True
                    break
                # The API target is evidence attached to its caller host; it is
                # not an intermediate vertex in the compressed behavior graph.
                if neighbor in endpoint_nodes:
                    continue
                if neighbor in host_roles and neighbor != start:
                    reached_hosts.add(neighbor)
                    continue
                stack.append((neighbor, distance + 1, visited | {neighbor}))
        roles = set(host_roles[start])
        for host in reached_hosts:
            roles.update(host_roles[host])
        if len(roles) >= 2:
            compressed_hyperedges.append((frozenset(roles), int(provenance[start])))
        if path_state_cap_hit:
            break

    vector = np.concatenate([
        _relation_vector(shared_caller_hyperedges),
        _relation_vector(compressed_hyperedges),
    ])
    if vector.shape != (HYPERGRAPH_DIMENSION,):
        raise ValueError("unexpected sensitive behavior hypergraph dimension")
    audit = {
        "node_count": len(node_ids),
        "selected_sensitive_endpoint_count": len(endpoint_nodes),
        "behavior_host_method_count": len(host_roles),
        "shared_caller_hyperedge_count": len(shared_caller_hyperedges),
        "shared_caller_high_order_count": sum(
            len(roles) >= 3 for roles, _ in shared_caller_hyperedges
        ),
        "compressed_reachability_hyperedge_count": len(compressed_hyperedges),
        "compressed_reachability_high_order_count": sum(
            len(roles) >= 3 for roles, _ in compressed_hyperedges
        ),
        "path_state_count": path_states,
        "path_state_cap_hit": path_state_cap_hit,
        "feature_dimension": HYPERGRAPH_DIMENSION,
    }
    return vector, audit


def hypergraph_matrix(rows, cache):
    vectors = []
    audits = []
    for row in rows:
        vector, audit = sensitive_behavior_hypergraph_vector(
            cache[row["sha256"]]["payload"]
        )
        audit["sha256"] = row["sha256"]
        vectors.append(vector)
        audits.append(audit)
    return np.asarray(vectors, dtype=np.float64), audits


def write_csv(path: Path, fields: list[str], rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})



# ============================================================
# Matched 70-dimensional Compact representations
#
# raw_count70:
#   保留原始数量，使用原训练折标准化，删除与allratio70相同的6维。
#
# logcount70:
#   46维保留语义数量先做log1p，再用训练折标准化；
#   其余24维与raw_count70相同。
#
# allratio70:
#   完全复用已经冻结的allratio70实现。
# ============================================================

def compact_quantity_matrices(
    train_rows,
    test_rows,
    feature_by_sha,
    schema,
):
    import run_stable70_vs_allratio70_2017 as ratio_exp

    # --------------------------------------------------------
    # 原始76维Compact。
    # original_compact_matrix_named已经使用训练折均值/方差标准化。
    # --------------------------------------------------------

    (
        raw_train_76,
        raw_test_76,
        names_76,
    ) = ratio_exp.original_compact_matrix_named(
        train_rows,
        test_rows,
        feature_by_sha,
        schema,
    )

    names_76 = list(
        names_76
    )

    if len(names_76) != 76:
        raise RuntimeError(
            "expected 76 original Compact features, got "
            + str(len(names_76))
        )

    drop_features = set(
        ratio_exp.DROP_FEATURES
    )

    found_drop_features = {
        name
        for name in names_76
        if name in drop_features
    }

    if found_drop_features != drop_features:
        raise RuntimeError(
            "missing DROP_FEATURES: "
            + repr(
                sorted(
                    drop_features
                    - found_drop_features
                )
            )
        )

    keep_indices = [
        index
        for index, name in enumerate(
            names_76
        )
        if name not in drop_features
    ]

    names_70 = [
        names_76[index]
        for index in keep_indices
    ]

    if len(names_70) != 70:
        raise RuntimeError(
            "expected 70 matched Compact features, got "
            + str(len(names_70))
        )

    # ========================================================
    # A. raw_count70
    #
    # 删除和allratio70相同的6维，保留其余原始数量特征。
    # ========================================================

    raw_train = (
        raw_train_76[
            :,
            keep_indices,
        ]
        .copy()
    )

    raw_test = (
        raw_test_76[
            :,
            keep_indices,
        ]
        .copy()
    )

    # ========================================================
    # B. logcount70
    #
    # 从原始76维副本开始，只替换保留的46维语义数量。
    # 不能对已经标准化的数值做log1p，必须重新读取原始计数。
    # ========================================================

    log_train_76 = (
        raw_train_76.copy()
    )

    log_test_76 = (
        raw_test_76.copy()
    )

    semantic_names = [
        name
        for name in names_76
        if (
            (
                name.startswith(
                    "role_node__"
                )
                or name.startswith(
                    "role_edge__"
                )
            )
            and name not in drop_features
        )
    ]

    if len(semantic_names) != 46:
        raise RuntimeError(
            "expected 46 retained semantic features, got "
            + str(len(semantic_names))
        )

    def raw_log_column(
        rows,
        feature_name,
    ):
        values = []

        for row in rows:
            sha = str(
                row["sha256"]
            ).casefold()

            if sha not in feature_by_sha:
                raise RuntimeError(
                    "missing feature row for "
                    + sha
                )

            source = feature_by_sha[
                sha
            ]

            try:
                value = float(
                    source.get(
                        feature_name,
                        0.0,
                    )
                )
            except (
                TypeError,
                ValueError,
            ):
                value = 0.0

            if not np.isfinite(
                value
            ):
                raise RuntimeError(
                    "non-finite count for "
                    + sha
                    + " "
                    + feature_name
                    + ": "
                    + repr(value)
                )

            if value < 0.0:
                raise RuntimeError(
                    "negative count for "
                    + sha
                    + " "
                    + feature_name
                    + ": "
                    + repr(value)
                )

            values.append(
                value
            )

        return np.log1p(
            np.asarray(
                values,
                dtype=np.float64,
            )
        )

    for feature_name in semantic_names:
        column_index = (
            names_76.index(
                feature_name
            )
        )

        fit_values = raw_log_column(
            train_rows,
            feature_name,
        )

        test_values = raw_log_column(
            test_rows,
            feature_name,
        )

        mean = float(
            fit_values.mean()
        )

        scale = float(
            fit_values.std()
        )

        if (
            not np.isfinite(
                scale
            )
            or scale < 1e-12
        ):
            scale = 1.0

        log_train_76[
            :,
            column_index,
        ] = (
            fit_values
            - mean
        ) / scale

        log_test_76[
            :,
            column_index,
        ] = (
            test_values
            - mean
        ) / scale

    log_train = (
        log_train_76[
            :,
            keep_indices,
        ]
        .copy()
    )

    log_test = (
        log_test_76[
            :,
            keep_indices,
        ]
        .copy()
    )

    # ========================================================
    # C. allratio70
    #
    # 完全复用冻结实现。
    # ========================================================

    (
        ratio_train,
        ratio_test,
        ratio_names,
    ) = ratio_exp.compact_matrix_allratio70(
        train_rows,
        test_rows,
        feature_by_sha,
        schema,
    )

    ratio_names = list(
        ratio_names
    )

    if ratio_names != names_70:
        raise RuntimeError(
            "feature order mismatch between "
            "Stable70 and allratio70"
        )

    matrices = {
        "raw_count70": (
            raw_train,
            raw_test,
        ),
        "logcount70": (
            log_train,
            log_test,
        ),
        "allratio70": (
            ratio_train,
            ratio_test,
        ),
    }

    for mode, (
        fit_matrix,
        test_matrix,
    ) in matrices.items():

        if (
            fit_matrix.shape[1] != 70
            or test_matrix.shape[1] != 70
        ):
            raise RuntimeError(
                mode
                + " dimension mismatch: "
                + repr(
                    (
                        fit_matrix.shape,
                        test_matrix.shape,
                    )
                )
            )

        if (
            not np.all(
                np.isfinite(
                    fit_matrix
                )
            )
            or not np.all(
                np.isfinite(
                    test_matrix
                )
            )
        ):
            raise RuntimeError(
                mode
                + " contains non-finite values"
            )

    return (
        matrices,
        names_70,
    )

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--entry-cache", required=True, type=Path)
    parser.add_argument("--endpoint-cache", required=True, type=Path)
    parser.add_argument("--features", required=True, type=Path)
    parser.add_argument("--schema", required=True, type=Path)
    parser.add_argument("--splits", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--iterations", type=int, default=1500)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--l2", type=float, default=0.03)
    args = parser.parse_args()

    paths = {
        "entry_cache": args.entry_cache.expanduser().resolve(strict=True),
        "endpoint_cache": args.endpoint_cache.expanduser().resolve(strict=True),
        "features": args.features.expanduser().resolve(strict=True),
        "schema": args.schema.expanduser().resolve(strict=True),
        "splits": args.splits.expanduser().resolve(strict=True),
    }
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    entry_cache = read_cache(paths["entry_cache"])
    endpoint_cache = read_cache(paths["endpoint_cache"])
    feature_rows = read_csv_rows(paths["features"])
    feature_by_sha = {row["sha256"].casefold(): row for row in feature_rows}
    schema = json.loads(paths["schema"].read_text(encoding="utf-8"))
    split_rows = read_split_rows(paths["splits"])
    attach_audit_metadata(entry_cache, feature_by_sha)
    attach_audit_metadata(endpoint_cache, feature_by_sha)
    selected = {
        row["sha256"] for groups in split_rows.values()
        for partition in ("train", "test") for row in groups[partition]
    }
    missing = {
        "entry_cache": len(selected - set(entry_cache)),
        "endpoint_cache": len(selected - set(endpoint_cache)),
        "features": len(selected - set(feature_by_sha)),
    }
    if any(missing.values()):
        raise SystemExit(f"missing selected rows: {missing}")
    years = {str(feature_by_sha[sha].get("metadata_year", "")) for sha in selected}
    if years != {"2017"}:
        raise SystemExit(f"2017 development rows required; observed={sorted(years)}")
    for groups in split_rows.values():
        for partition in ("train", "test"):
            for row in groups[partition]:
                row["metadata_label"] = feature_by_sha[row["sha256"]].get("metadata_label", "")

    definition = PilotReliabilityDefinition(
        alpha_prior=1.0, kappa_support=20.0, kappa_context=20.0,
        r_min=0.10, r_max=0.95,
    )
    entry_args = entry_model_args(definition)
    variants = (
        "raw_count70_only",
        "raw_count70_plus_shared_caller",
        "logcount70_only",
        "logcount70_plus_shared_caller",
        "allratio70_only",
        "allratio70_plus_shared_caller",
    )

    results = {variant: [] for variant in variants}
    predictions = []
    audit_rows = []

    for split_name in sorted(split_rows):
        train_rows = split_rows[split_name]["train"]
        test_rows = split_rows[split_name]["test"]
        evidence = fit_training_fold_components(
            entry_cache, {row["sha256"] for row in train_rows}, definition,
            active_components=ACTIVE_RELIABILITY_COMPONENTS,
        )
        entry_train, y_train = graph_matrix(train_rows, entry_cache, evidence, entry_args, True)
        entry_test, y_test = graph_matrix(test_rows, entry_cache, evidence, entry_args, True)
        (
            compact_modes,
            compact_names,
        ) = compact_quantity_matrices(
            train_rows,
            test_rows,
            feature_by_sha,
            schema,
        )

        compact_count = len(
            compact_names
        )
        hyper_train, train_audit = hypergraph_matrix(train_rows, endpoint_cache)
        hyper_test, test_audit = hypergraph_matrix(test_rows, endpoint_cache)
        entry_train, entry_test = standardize_pair(entry_train, entry_test)
        hyper_train, hyper_test = standardize_pair(hyper_train, hyper_test)
        shared_train = hyper_train[:, :RELATION_DIMENSION]
        shared_test = hyper_test[:, :RELATION_DIMENSION]
        compressed_train = hyper_train[:, RELATION_DIMENSION:]
        compressed_test = hyper_test[:, RELATION_DIMENSION:]
        entry_train *= ENTRY_GRAPH_WEIGHT
        entry_test *= ENTRY_GRAPH_WEIGHT
        matrices = {
            "raw_count70_only": (
                compact_modes[
                    "raw_count70"
                ][0],
                compact_modes[
                    "raw_count70"
                ][1],
            ),

            "raw_count70_plus_shared_caller": (
                np.concatenate(
                    [
                        compact_modes[
                            "raw_count70"
                        ][0],
                        shared_train,
                    ],
                    axis=1,
                ),
                np.concatenate(
                    [
                        compact_modes[
                            "raw_count70"
                        ][1],
                        shared_test,
                    ],
                    axis=1,
                ),
            ),

            "logcount70_only": (
                compact_modes[
                    "logcount70"
                ][0],
                compact_modes[
                    "logcount70"
                ][1],
            ),

            "logcount70_plus_shared_caller": (
                np.concatenate(
                    [
                        compact_modes[
                            "logcount70"
                        ][0],
                        shared_train,
                    ],
                    axis=1,
                ),
                np.concatenate(
                    [
                        compact_modes[
                            "logcount70"
                        ][1],
                        shared_test,
                    ],
                    axis=1,
                ),
            ),

            "allratio70_only": (
                compact_modes[
                    "allratio70"
                ][0],
                compact_modes[
                    "allratio70"
                ][1],
            ),

            "allratio70_plus_shared_caller": (
                np.concatenate(
                    [
                        compact_modes[
                            "allratio70"
                        ][0],
                        shared_train,
                    ],
                    axis=1,
                ),
                np.concatenate(
                    [
                        compact_modes[
                            "allratio70"
                        ][1],
                        shared_test,
                    ],
                    axis=1,
                ),
            ),
        }

        relation_feature_counts = {
            "raw_count70_only":
                0,

            "raw_count70_plus_shared_caller":
                RELATION_DIMENSION,

            "logcount70_only":
                0,

            "logcount70_plus_shared_caller":
                RELATION_DIMENSION,

            "allratio70_only":
                0,

            "allratio70_plus_shared_caller":
                RELATION_DIMENSION,
        }

        for partition, rows in (("train", train_audit), ("test", test_audit)):
            for audit in rows:
                audit.update({"split_name": split_name, "partition": partition})
                audit_rows.append(audit)
        seed_metrics = {variant: [] for variant in variants}
        for seed in args.seeds:
            for variant, (x_train, x_test) in matrices.items():
                probability = predict_logistic(
                    x_train, y_train, x_test, seed,
                    args.iterations, args.learning_rate, args.l2,
                )
                seed_metrics[variant].append(metrics(y_test, probability, THRESHOLD))
                for row, label, value in zip(test_rows, y_test, probability):
                    predictions.append({
                        "variant": variant, "split_name": split_name,
                        "sha256": row["sha256"], "seed": seed,
                        "label": int(label), "probability": float(value),
                        "threshold": THRESHOLD,
                        "predicted_label": int(value >= THRESHOLD),
                    })
        for variant in variants:
            values = seed_metrics[variant]
            result = {
                "variant": variant, "split_name": split_name,
                "train_count": len(train_rows), "test_count": len(test_rows),
                "compact_feature_count": int(compact_count),
                "hypergraph_feature_count": relation_feature_counts[variant],
                "entry_graph_weight": ENTRY_GRAPH_WEIGHT,
                "threshold": THRESHOLD,
            }
            for metric_name in ("accuracy", "balanced_accuracy", "precision", "recall", "f1", "auc"):
                result[f"{metric_name}_mean"] = float(np.mean([
                    item[metric_name] for item in values
                ]))
            result["test_rows_used_for_training_or_selection"] = False
            results[variant].append(result)
        print(f"completed {split_name}", flush=True)

    ranked = []
    for variant in variants:
        rows = results[variant]
        bas = np.asarray([row["balanced_accuracy_mean"] for row in rows])
        ranked.append({
            "rank": 0, "variant": variant,
            "mean_accuracy": float(np.mean([row["accuracy_mean"] for row in rows])),
            "mean_balanced_accuracy": float(bas.mean()),
            "minimum_split_balanced_accuracy": float(bas.min()),
            "std_balanced_accuracy": float(bas.std()),
            "stability_score": float(bas.mean() - bas.std()),
            "mean_auc": float(np.mean([row["auc_mean"] for row in rows])),
        })
    ranked.sort(key=lambda row: (-row["mean_balanced_accuracy"], -row["mean_auc"], row["variant"]))
    for rank, row in enumerate(ranked, 1):
        row["rank"] = rank

    summary_path = output / "sensitive_behavior_hypergraph_2017_summary.csv"
    per_split_path = output / "sensitive_behavior_hypergraph_2017_per_split.csv"
    prediction_path = output / "sensitive_behavior_hypergraph_2017_predictions.csv"
    hypergraph_audit_path = output / "sensitive_behavior_hypergraph_structure_audit.csv"
    write_csv(summary_path, list(ranked[0]), ranked)
    write_csv(per_split_path, list(results[variants[0]][0]), [row for variant in variants for row in results[variant]])
    write_csv(prediction_path, list(predictions[0]), predictions)
    write_csv(hypergraph_audit_path, list(audit_rows[0]), audit_rows)
    audit = {
        "schema_version": "1.0.0",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "phase": "P46_ENTRY_INDEPENDENT_SENSITIVE_BEHAVIOR_COMBINATION_HYPERGRAPH_2017",
        "formal_training_enabled": False,
        "strict_test_isolation": True,
        "label_independent_hypergraph_construction": True,
        "threshold": THRESHOLD,
        "entry_graph_weight": ENTRY_GRAPH_WEIGHT,
        "hyperedge_relations": ["shared_direct_caller", "compressed_sensitive_reachability"],
        "hypergraph_feature_count": HYPERGRAPH_DIMENSION,
        "ordinary_method_vertices_retained": False,
        "raw_method_identifiers_persisted": False,
        "entry_required_for_hypergraph": False,
        "same_class_relation_status": "NOT_AVAILABLE_IN_EXISTING_CACHE",
        "shared_state_relation_status": "REQUIRES_BYTECODE_DATA_DEPENDENCE_EXTRACTION",
        "endpoint_cache_limitations": {
            "upstream_hops": 2, "downstream_hops": 1,
            "K_in": 8, "K_out": 4,
        },
        "seeds": list(args.seeds), "iterations": args.iterations,
        "learning_rate": args.learning_rate, "l2": args.l2,
        "entry_cache": str(paths["entry_cache"]),
        "entry_cache_sha256": sha256_file(paths["entry_cache"]),
        "endpoint_cache": str(paths["endpoint_cache"]),
        "endpoint_cache_sha256": sha256_file(paths["endpoint_cache"]),
        "features": str(paths["features"]),
        "features_sha256": sha256_file(paths["features"]),
        "splits": str(paths["splits"]),
        "splits_sha256": sha256_file(paths["splits"]),
        "variants": list(variants), "ranked_variants": ranked,
        "outputs": {
            "summary": str(summary_path), "per_split": str(per_split_path),
            "predictions": str(prediction_path),
            "structure_audit": str(hypergraph_audit_path),
        },
        "blocking_reasons": [
            "same_class_hyperedges_not_available_without_cache_rebuild",
            "shared_state_hyperedges_not_available_without_data_dependence_extraction",
            "method_level_endpoint_cache_is_not_call_site_level",
            "candidate_not_locked",
        ],
        "notes": [
            "This MVP uses method-level sensitive endpoints because the existing cache does not preserve individual call sites.",
            "A positive result justifies a second extraction stage for call-site, same-class, and shared-state hyperedges.",
            "A negative result rejects these two inexpensive hyperedge relations but does not test shared-state relations.",
            "Classification threshold is fixed at 0.5.",
        ],
    }
    audit_path = output / "sensitive_behavior_hypergraph_2017_audit.json"
    audit_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("rank variant mean_BA min_BA std_BA stability AUC", flush=True)
    for row in ranked:
        print(
            row["rank"], row["variant"],
            f'{row["mean_balanced_accuracy"]:.4f}',
            f'{row["minimum_split_balanced_accuracy"]:.4f}',
            f'{row["std_balanced_accuracy"]:.4f}',
            f'{row["stability_score"]:.4f}',
            f'{row["mean_auc"]:.4f}', flush=True,
        )


if __name__ == "__main__":
    main()
