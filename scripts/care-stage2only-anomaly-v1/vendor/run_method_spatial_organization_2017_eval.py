#!/usr/bin/env python3
"""Pilot method-level sensitive-behaviour spatial organization features.

This experiment reuses the bounded P25 endpoint cache and never opens an APK.
It asks how semantic roles are distributed across observed caller methods:
role concentration/dispersion, pairwise method co-location, and hashed
high-order method co-location.  It does not claim call reachability or data
dependence.  The existing endpoint cache retains at most K_in callers per
sensitive endpoint, so this is a feasibility pilot rather than a complete
method-coverage measurement.

Only chronological 2017 development folds are accepted.  Labels are used only
for model fitting/evaluation, never for spatial feature construction.  The
classification threshold is fixed at 0.5.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from itertools import combinations
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
ROLE_STAT_WIDTH = 8
PAIR_STAT_WIDTH = 2
TRIPLE_HASH_WIDTH = 128
GLOBAL_WIDTH = 14
ROLE_STAT_DIMENSION = ROLE_VOCABULARY_SIZE * ROLE_STAT_WIDTH
PAIR_COUNT = ROLE_VOCABULARY_SIZE * (ROLE_VOCABULARY_SIZE - 1) // 2
PAIR_DIMENSION = PAIR_COUNT * PAIR_STAT_WIDTH
DISTRIBUTION_DIMENSION = ROLE_STAT_DIMENSION + GLOBAL_WIDTH
ORGANIZATION_DIMENSION = (
    DISTRIBUTION_DIMENSION + PAIR_DIMENSION + TRIPLE_HASH_WIDTH
)


def _roles(values) -> frozenset[int]:
    return frozenset(
        int(value)
        for value in (values or [])
        if 0 <= int(value) < ROLE_VOCABULARY_SIZE
    )


def _normalized_entropy(counts: np.ndarray) -> float:
    total = float(counts.sum())
    if total <= 0.0 or len(counts) <= 1:
        return 0.0
    probabilities = counts[counts > 0.0] / total
    entropy = float(-(probabilities * np.log(probabilities)).sum())
    return entropy / float(np.log(len(counts)))


def _triple_bucket(triple: tuple[int, int, int]) -> int:
    token = ":".join(map(str, triple)).encode("ascii")
    digest = hashlib.blake2b(token, digest_size=8).digest()
    return int.from_bytes(digest, "big") % TRIPLE_HASH_WIDTH


def method_spatial_organization_vector(payload: dict) -> tuple[np.ndarray, dict]:
    """Summarize observed semantic-role organization across caller methods."""

    node_ids, base, role_rows, source, target = _payload_arrays(payload)
    endpoint_nodes = set(map(int, np.flatnonzero(base[:, 4] > 0.5)))
    node_roles = [_roles(values) for values in role_rows]

    # A host method is observed when a retained caller->sensitive-endpoint edge
    # exists.  Counts are endpoint-method assignments, not individual call-site
    # counts, because P25 deliberately does not persist call sites.
    host_role_counts: dict[int, Counter[int]] = defaultdict(Counter)
    endpoint_with_host = set()
    for src_value, dst_value in zip(source, target):
        caller = int(src_value)
        endpoint = int(dst_value)
        if endpoint not in endpoint_nodes or not node_roles[endpoint]:
            continue
        endpoint_with_host.add(endpoint)
        for role in node_roles[endpoint]:
            host_role_counts[caller][role] += 1

    hosts = sorted(host_role_counts)
    host_count = len(hosts)
    role_hosts: list[set[int]] = [set() for _ in range(ROLE_VOCABULARY_SIZE)]
    role_method_counts: list[list[float]] = [
        [] for _ in range(ROLE_VOCABULARY_SIZE)
    ]
    provenance_counts = np.zeros(
        (ROLE_VOCABULARY_SIZE, 3), dtype=np.float64
    )
    host_role_sets: dict[int, frozenset[int]] = {}

    for host in hosts:
        counts = host_role_counts[host]
        roles = frozenset(counts)
        host_role_sets[host] = roles
        provenance = int(np.argmax(base[host, 5:8]))
        for role, count in counts.items():
            role_hosts[role].add(host)
            role_method_counts[role].append(float(count))
            provenance_counts[role, provenance] += 1.0

    role_features = np.zeros(
        (ROLE_VOCABULARY_SIZE, ROLE_STAT_WIDTH), dtype=np.float64
    )
    denominator = max(host_count, 1)
    for role in range(ROLE_VOCABULARY_SIZE):
        counts = np.asarray(role_method_counts[role], dtype=np.float64)
        method_count = len(counts)
        if method_count == 0:
            continue
        probabilities = counts / max(float(counts.sum()), 1.0)
        provenance_share = provenance_counts[role] / float(method_count)
        role_features[role] = np.asarray([
            np.log1p(method_count) / np.log1p(denominator + 1.0),
            method_count / float(denominator),
            float(probabilities.max()),
            float(np.square(probabilities).sum()),
            _normalized_entropy(counts),
            *provenance_share.tolist(),
        ])

    pair_features = np.zeros(
        (PAIR_COUNT, PAIR_STAT_WIDTH), dtype=np.float64
    )
    pair_index = 0
    nonzero_pairs = 0
    for left in range(ROLE_VOCABULARY_SIZE):
        for right in range(left + 1, ROLE_VOCABULARY_SIZE):
            intersection = len(role_hosts[left] & role_hosts[right])
            union = len(role_hosts[left] | role_hosts[right])
            minimum = min(len(role_hosts[left]), len(role_hosts[right]))
            if intersection:
                nonzero_pairs += 1
            pair_features[pair_index] = (
                intersection / float(union) if union else 0.0,
                intersection / float(minimum) if minimum else 0.0,
            )
            pair_index += 1

    triple_features = np.zeros(TRIPLE_HASH_WIDTH, dtype=np.float64)
    triple_pattern_count = 0
    triple_occurrence_count = 0
    seen_triples = set()
    for roles in host_role_sets.values():
        if len(roles) < 3:
            continue
        for triple in combinations(sorted(roles), 3):
            triple_features[_triple_bucket(triple)] += 1.0
            seen_triples.add(triple)
            triple_occurrence_count += 1
    triple_pattern_count = len(seen_triples)
    triple_features /= float(denominator)

    host_diversities = np.asarray(
        [len(host_role_sets[host]) for host in hosts], dtype=np.float64
    )
    host_assignment_counts = np.asarray(
        [sum(host_role_counts[host].values()) for host in hosts],
        dtype=np.float64,
    )
    diversity_hist = np.zeros(5, dtype=np.float64)
    for diversity in host_diversities:
        diversity_hist[min(int(diversity), 5) - 1] += 1.0
    diversity_hist /= float(denominator)
    assignment_probabilities = (
        host_assignment_counts / max(float(host_assignment_counts.sum()), 1.0)
        if host_count else np.zeros(0, dtype=np.float64)
    )
    global_features = np.asarray([
        np.log1p(host_count),
        len(endpoint_with_host) / float(max(len(endpoint_nodes), 1)),
        float(host_diversities.mean()) if host_count else 0.0,
        (
            float(host_diversities.max()) / ROLE_VOCABULARY_SIZE
            if host_count else 0.0
        ),
        float(np.mean(host_diversities >= 2.0)) if host_count else 0.0,
        *diversity_hist.tolist(),
        float(np.square(assignment_probabilities).sum()),
        _normalized_entropy(host_assignment_counts),
        nonzero_pairs / float(max(PAIR_COUNT, 1)),
        triple_pattern_count / float(max(triple_occurrence_count, 1)),
    ])
    if global_features.shape != (GLOBAL_WIDTH,):
        raise ValueError(f"unexpected global feature shape: {global_features.shape}")

    distribution = np.concatenate([role_features.reshape(-1), global_features])
    organization = np.concatenate([
        distribution,
        pair_features.reshape(-1),
        triple_features,
    ])
    if distribution.shape != (DISTRIBUTION_DIMENSION,):
        raise ValueError("unexpected method distribution feature dimension")
    if organization.shape != (ORGANIZATION_DIMENSION,):
        raise ValueError("unexpected method organization feature dimension")
    audit = {
        "node_count": len(node_ids),
        "selected_sensitive_endpoint_count": len(endpoint_nodes),
        "endpoint_with_observed_host_count": len(endpoint_with_host),
        "endpoint_observed_host_ratio": (
            len(endpoint_with_host) / float(max(len(endpoint_nodes), 1))
        ),
        "observed_behavior_host_method_count": host_count,
        "multi_role_host_method_count": int(np.sum(host_diversities >= 2.0)),
        "high_order_host_method_count": int(np.sum(host_diversities >= 3.0)),
        "nonzero_role_pair_count": nonzero_pairs,
        "triple_pattern_count": triple_pattern_count,
        "distribution_feature_dimension": DISTRIBUTION_DIMENSION,
        "organization_feature_dimension": ORGANIZATION_DIMENSION,
    }
    return organization, audit


def organization_matrix(rows, cache):
    vectors = []
    audits = []
    for row in rows:
        vector, audit = method_spatial_organization_vector(
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
        row["sha256"]
        for groups in split_rows.values()
        for partition in ("train", "test")
        for row in groups[partition]
    }
    missing = {
        "entry_cache": len(selected - set(entry_cache)),
        "endpoint_cache": len(selected - set(endpoint_cache)),
        "features": len(selected - set(feature_by_sha)),
    }
    if any(missing.values()):
        raise SystemExit(f"missing selected rows: {missing}")
    years = {
        str(feature_by_sha[sha].get("metadata_year", "")) for sha in selected
    }
    if years != {"2017"}:
        raise SystemExit(f"2017 development rows required; observed={sorted(years)}")
    for groups in split_rows.values():
        for partition in ("train", "test"):
            for row in groups[partition]:
                row["metadata_label"] = feature_by_sha[row["sha256"]].get(
                    "metadata_label", ""
                )

    definition = PilotReliabilityDefinition(
        alpha_prior=1.0,
        kappa_support=20.0,
        kappa_context=20.0,
        r_min=0.10,
        r_max=0.95,
    )
    entry_args = entry_model_args(definition)
    variants = (
        "compact_only",
        "compact_plus_entry",
        "compact_plus_method_distribution",
        "compact_plus_method_organization",
        "compact_plus_entry_plus_method_organization",
    )
    results = {variant: [] for variant in variants}
    predictions = []
    audit_rows = []

    for split_name in sorted(split_rows):
        train_rows = split_rows[split_name]["train"]
        test_rows = split_rows[split_name]["test"]
        evidence = fit_training_fold_components(
            entry_cache,
            {row["sha256"] for row in train_rows},
            definition,
            active_components=ACTIVE_RELIABILITY_COMPONENTS,
        )
        entry_train, y_train = graph_matrix(
            train_rows, entry_cache, evidence, entry_args, True
        )
        entry_test, y_test = graph_matrix(
            test_rows, entry_cache, evidence, entry_args, True
        )
        compact_train, compact_test, compact_count = compact_matrix(
            train_rows,
            test_rows,
            feature_by_sha,
            schema,
            feature_mode="base",
        )
        organization_train, train_audit = organization_matrix(
            train_rows, endpoint_cache
        )
        organization_test, test_audit = organization_matrix(
            test_rows, endpoint_cache
        )
        distribution_train = organization_train[:, :DISTRIBUTION_DIMENSION]
        distribution_test = organization_test[:, :DISTRIBUTION_DIMENSION]
        entry_train, entry_test = standardize_pair(entry_train, entry_test)
        distribution_train, distribution_test = standardize_pair(
            distribution_train, distribution_test
        )
        organization_train, organization_test = standardize_pair(
            organization_train, organization_test
        )
        entry_train *= ENTRY_GRAPH_WEIGHT
        entry_test *= ENTRY_GRAPH_WEIGHT

        matrices = {
            "compact_only": (compact_train, compact_test),
            "compact_plus_entry": (
                np.concatenate([compact_train, entry_train], axis=1),
                np.concatenate([compact_test, entry_test], axis=1),
            ),
            "compact_plus_method_distribution": (
                np.concatenate([compact_train, distribution_train], axis=1),
                np.concatenate([compact_test, distribution_test], axis=1),
            ),
            "compact_plus_method_organization": (
                np.concatenate([compact_train, organization_train], axis=1),
                np.concatenate([compact_test, organization_test], axis=1),
            ),
            "compact_plus_entry_plus_method_organization": (
                np.concatenate(
                    [compact_train, entry_train, organization_train], axis=1
                ),
                np.concatenate(
                    [compact_test, entry_test, organization_test], axis=1
                ),
            ),
        }
        for partition, rows in (("train", train_audit), ("test", test_audit)):
            for audit in rows:
                audit.update({"split_name": split_name, "partition": partition})
                audit_rows.append(audit)

        seed_metrics = {variant: [] for variant in variants}
        for seed in args.seeds:
            for variant, (x_train, x_test) in matrices.items():
                probability = predict_logistic(
                    x_train,
                    y_train,
                    x_test,
                    seed,
                    args.iterations,
                    args.learning_rate,
                    args.l2,
                )
                seed_metrics[variant].append(
                    metrics(y_test, probability, THRESHOLD)
                )
                for row, label, value in zip(test_rows, y_test, probability):
                    predictions.append({
                        "variant": variant,
                        "split_name": split_name,
                        "sha256": row["sha256"],
                        "seed": seed,
                        "label": int(label),
                        "probability": float(value),
                        "threshold": THRESHOLD,
                        "predicted_label": int(value >= THRESHOLD),
                    })

        for variant in variants:
            values = seed_metrics[variant]
            result = {
                "variant": variant,
                "split_name": split_name,
                "train_count": len(train_rows),
                "test_count": len(test_rows),
                "compact_feature_count": int(compact_count),
                "method_distribution_feature_count": DISTRIBUTION_DIMENSION,
                "method_organization_feature_count": ORGANIZATION_DIMENSION,
                "entry_graph_weight": ENTRY_GRAPH_WEIGHT,
                "threshold": THRESHOLD,
            }
            for metric_name in (
                "accuracy",
                "balanced_accuracy",
                "precision",
                "recall",
                "f1",
                "auc",
            ):
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
            "rank": 0,
            "variant": variant,
            "mean_accuracy": float(np.mean([
                row["accuracy_mean"] for row in rows
            ])),
            "mean_balanced_accuracy": float(bas.mean()),
            "minimum_split_balanced_accuracy": float(bas.min()),
            "std_balanced_accuracy": float(bas.std()),
            "stability_score": float(bas.mean() - bas.std()),
            "mean_auc": float(np.mean([row["auc_mean"] for row in rows])),
        })
    ranked.sort(
        key=lambda row: (
            -row["mean_balanced_accuracy"], -row["mean_auc"], row["variant"]
        )
    )
    for rank, row in enumerate(ranked, 1):
        row["rank"] = rank

    summary_path = output / "method_spatial_organization_2017_summary.csv"
    per_split_path = output / "method_spatial_organization_2017_per_split.csv"
    prediction_path = output / "method_spatial_organization_2017_predictions.csv"
    structure_path = output / "method_spatial_organization_structure_audit.csv"
    write_csv(summary_path, list(ranked[0]), ranked)
    write_csv(
        per_split_path,
        list(results[variants[0]][0]),
        [row for variant in variants for row in results[variant]],
    )
    write_csv(prediction_path, list(predictions[0]), predictions)
    write_csv(structure_path, list(audit_rows[0]), audit_rows)

    audit = {
        "schema_version": "1.0.0",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "phase": "P49_METHOD_LEVEL_SENSITIVE_BEHAVIOR_SPATIAL_ORGANIZATION_2017",
        "formal_training_enabled": False,
        "strict_test_isolation": True,
        "label_independent_spatial_feature_construction": True,
        "threshold": THRESHOLD,
        "entry_graph_weight": ENTRY_GRAPH_WEIGHT,
        "spatial_levels": ["OBSERVED_CALLER_METHOD"],
        "spatial_feature_families": [
            "role_method_dispersion",
            "role_method_concentration",
            "caller_provenance_distribution",
            "pairwise_method_colocation_jaccard",
            "pairwise_method_colocation_overlap",
            "hashed_high_order_method_colocation",
        ],
        "distribution_feature_count": DISTRIBUTION_DIMENSION,
        "organization_feature_count": ORGANIZATION_DIMENSION,
        "triple_hash_width": TRIPLE_HASH_WIDTH,
        "call_reachability_claimed": False,
        "data_dependence_claimed": False,
        "raw_method_identifiers_persisted": False,
        "seeds": list(args.seeds),
        "iterations": args.iterations,
        "learning_rate": args.learning_rate,
        "l2": args.l2,
        "entry_cache": str(paths["entry_cache"]),
        "entry_cache_sha256": sha256_file(paths["entry_cache"]),
        "endpoint_cache": str(paths["endpoint_cache"]),
        "endpoint_cache_sha256": sha256_file(paths["endpoint_cache"]),
        "features": str(paths["features"]),
        "features_sha256": sha256_file(paths["features"]),
        "splits": str(paths["splits"]),
        "splits_sha256": sha256_file(paths["splits"]),
        "variants": list(variants),
        "ranked_variants": ranked,
        "outputs": {
            "summary": str(summary_path),
            "per_split": str(per_split_path),
            "predictions": str(prediction_path),
            "structure_audit": str(structure_path),
        },
        "blocking_reasons": [
            "existing_endpoint_cache_retains_at_most_8_callers_per_endpoint",
            "existing_endpoint_cache_is_method_level_not_call_site_level",
            "class_and_package_levels_require_lightweight_cache_rebuild",
            "candidate_not_locked",
        ],
        "notes": [
            "Spatial co-location is an observed code-organization relation, not a data-flow or reachability claim.",
            "This pilot tests method-level feasibility before any APK re-extraction.",
            "Pairwise co-location uses normalized Jaccard and overlap statistics rather than raw Cartesian products.",
            "High-order role triples use a fixed label-independent hash space.",
            "Classification threshold is fixed at 0.5.",
        ],
    }
    audit_path = output / "method_spatial_organization_2017_audit.json"
    audit_path.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print("rank variant mean_BA min_BA std_BA stability AUC", flush=True)
    for row in ranked:
        print(
            row["rank"],
            row["variant"],
            f'{row["mean_balanced_accuracy"]:.4f}',
            f'{row["minimum_split_balanced_accuracy"]:.4f}',
            f'{row["std_balanced_accuracy"]:.4f}',
            f'{row["stability_score"]:.4f}',
            f'{row["mean_auc"]:.4f}',
            flush=True,
        )


if __name__ == "__main__":
    main()
