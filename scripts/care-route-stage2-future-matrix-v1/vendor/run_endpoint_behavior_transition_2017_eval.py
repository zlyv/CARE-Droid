#!/usr/bin/env python3
"""Evaluate semantic behaviour transitions around all sensitive endpoints.

This is the first exact four-group experiment for the orphan-sensitive-node
hypothesis.  It uses the existing bounded endpoint cache, maps nodes to
semantic role IDs (plus one ``normal_application_function`` bucket), and
counts role transitions without retaining method names.

The current cache policy is upstream=2 and downstream=1.  Therefore this
first experiment is a 2-hop/1-hop feasibility test; a 3-hop/2-hop result
requires rebuilding the endpoint cache.

Compared groups:
  1. compact_only
  2. compact_plus_entry
  3. compact_plus_endpoint_behavior
  4. compact_plus_entry_plus_endpoint_behavior

Only chronological 2017 development rows are accepted.  The classification
threshold is fixed at 0.5 and all standardization/reliability fitting uses
the corresponding training partition only.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
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
NORMAL_ROLE = ROLE_VOCABULARY_SIZE
ROLE_COUNT = ROLE_VOCABULARY_SIZE + 1
STAGES = (
    "caller_1hop_to_endpoint",
    "caller_2hop_to_endpoint",
    "endpoint_to_callee_1hop",
)
PAIR_FAMILIES = (
    "shared_caller_endpoint_pair",
    "collapsed_sensitive_path_pair",
)
MAX_COLLAPSED_PATH_HOPS = 4
PATH_STATE_CAP = 20000
BEHAVIOR_DIMENSION = (
    len(STAGES) * ROLE_COUNT
    + ROLE_COUNT
    + len(PAIR_FAMILIES) * ROLE_COUNT * ROLE_COUNT
    + MAX_COLLAPSED_PATH_HOPS
    + 8
)


def _roles(role_row) -> list[int]:
    values = sorted({
        int(value) for value in (role_row or [])
        if 0 <= int(value) < ROLE_VOCABULARY_SIZE
    })
    return values or [NORMAL_ROLE]


def _add_histogram(target: np.ndarray, values: list[int]) -> None:
    for value in values:
        target[int(value)] += 1.0


def _add_pair_matrix(target: np.ndarray, left: list[int], right: list[int]) -> None:
    for left_value in left:
        for right_value in right:
            target[int(left_value), int(right_value)] += 1.0


def endpoint_behavior_vector(payload: dict) -> tuple[np.ndarray, dict]:
    """Build name-free role transitions and return reachability audit data."""

    node_ids, base, roles, source, target = _payload_arrays(payload)
    role_rows = [_roles(row) for row in roles]
    endpoint_mask = base[:, 4] > 0.5
    unreachable_mask = base[:, 9] > 0.5
    reachable_mask = base[:, 8] > 0.5
    unknown_mask = base[:, 10] > 0.5

    incoming: dict[int, list[int]] = defaultdict(list)
    outgoing: dict[int, list[int]] = defaultdict(list)
    for edge_index, (src, dst) in enumerate(zip(source, target)):
        incoming[int(dst)].append(edge_index)
        outgoing[int(src)].append(edge_index)

    caller_hist = [np.zeros(ROLE_COUNT, dtype=np.float64) for _ in STAGES]
    endpoint_hist = np.zeros(ROLE_COUNT, dtype=np.float64)
    shared_pairs = np.zeros((ROLE_COUNT, ROLE_COUNT), dtype=np.float64)
    collapsed_path_pairs = np.zeros((ROLE_COUNT, ROLE_COUNT), dtype=np.float64)
    collapsed_path_lengths = np.zeros(MAX_COLLAPSED_PATH_HOPS, dtype=np.float64)

    selected_endpoints = [
        int(index) for index in np.flatnonzero(endpoint_mask)
    ]
    unreachable_endpoints = {
        index for index in selected_endpoints if unreachable_mask[index]
    }
    reachable_endpoints = {
        index for index in selected_endpoints if reachable_mask[index]
    }
    unknown_endpoints = {
        index for index in selected_endpoints if unknown_mask[index]
    }

    endpoint_by_caller: dict[int, set[int]] = defaultdict(set)
    for endpoint in selected_endpoints:
        _add_histogram(endpoint_hist, role_rows[endpoint])
        for edge_index in incoming.get(endpoint, []):
            caller = int(source[edge_index])
            endpoint_by_caller[caller].add(endpoint)
            _add_histogram(caller_hist[0], role_rows[caller])
            for second_index in incoming.get(caller, []):
                grandparent = int(source[second_index])
                _add_histogram(caller_hist[1], role_rows[grandparent])
        for edge_index in outgoing.get(endpoint, []):
            callee = int(target[edge_index])
            _add_histogram(caller_hist[2], role_rows[callee])

    for endpoint_set in endpoint_by_caller.values():
        endpoint_list = sorted(endpoint_set)
        for left in endpoint_list:
            for right in endpoint_list:
                if left == right:
                    continue
                _add_pair_matrix(
                    shared_pairs, role_rows[left], role_rows[right]
                )

    # Collapse bounded paths between sensitive endpoints.  Intermediate nodes
    # may be unmapped normal functions or mapped support roles.  We stop at the
    # first downstream endpoint, so a path is not counted through another
    # sensitive API.  The existing 2-upstream/1-downstream endpoint cache can
    # retain endpoint-to-endpoint paths of up to four edges through the union
    # of both endpoint neighbourhoods.
    path_state_count = 0
    path_state_cap_hit = False
    collapsed_path_count = 0
    for start in selected_endpoints:
        stack = [(start, 0, frozenset({start}))]
        while stack and not path_state_cap_hit:
            node, distance, visited = stack.pop()
            if distance >= MAX_COLLAPSED_PATH_HOPS:
                continue
            for edge_index in outgoing.get(node, []):
                next_node = int(target[edge_index])
                if next_node in visited:
                    continue
                path_state_count += 1
                if path_state_count > PATH_STATE_CAP:
                    path_state_cap_hit = True
                    break
                next_distance = distance + 1
                if endpoint_mask[next_node]:
                    _add_pair_matrix(
                        collapsed_path_pairs,
                        role_rows[start],
                        role_rows[next_node],
                    )
                    collapsed_path_lengths[next_distance - 1] += 1.0
                    collapsed_path_count += 1
                    continue
                stack.append((
                    next_node,
                    next_distance,
                    visited | {next_node},
                ))
        if path_state_cap_hit:
            break

    values = [
        np.log1p(hist).reshape(-1) for hist in caller_hist
    ]
    values.append(np.log1p(endpoint_hist))
    values.extend([
        np.log1p(shared_pairs).reshape(-1),
        np.log1p(collapsed_path_pairs).reshape(-1),
    ])
    values.append(np.log1p(collapsed_path_lengths))
    values.append(np.asarray([
        np.log1p(len(selected_endpoints)),
        np.log1p(len(unreachable_endpoints)),
        np.log1p(len(reachable_endpoints)),
        np.log1p(len(unknown_endpoints)),
        np.log1p(len(endpoint_by_caller)),
        np.log1p(sum(len(value) for value in endpoint_by_caller.values())),
        np.log1p(len(source)),
        float(np.mean(unreachable_mask[endpoint_mask]))
        if np.any(endpoint_mask) else 0.0,
    ], dtype=np.float64))
    vector = np.concatenate(values).astype(np.float64, copy=False)
    if vector.shape != (BEHAVIOR_DIMENSION,):
        raise ValueError(
            f"unexpected behavior dimension {vector.shape}; expected {BEHAVIOR_DIMENSION}"
        )
    audit = {
        "sensitive_endpoint_count": len(selected_endpoints),
        "entry_unreachable_endpoint_count": len(unreachable_endpoints),
        "entry_reachable_endpoint_count": len(reachable_endpoints),
        "reachability_unknown_endpoint_count": len(unknown_endpoints),
        "nonzero_shared_caller_pair_count": int(np.count_nonzero(shared_pairs)),
        "nonzero_collapsed_path_pair_count": int(
            np.count_nonzero(collapsed_path_pairs)
        ),
        "collapsed_sensitive_path_count": collapsed_path_count,
        "collapsed_path_state_count": path_state_count,
        "collapsed_path_state_cap_hit": path_state_cap_hit,
        "unreachable_endpoint_ratio": (
            float(len(unreachable_endpoints) / len(selected_endpoints))
            if selected_endpoints else 0.0
        ),
        "example_pair_feature_space": "role_id_to_role_id_without_method_names",
    }
    return vector, audit


def behavior_matrix(rows: list[dict], cache: dict) -> tuple[np.ndarray, list[dict]]:
    vectors = []
    audits = []
    for row in rows:
        vector, audit = endpoint_behavior_vector(
            cache[row["sha256"]]["payload"]
        )
        vectors.append(vector)
        audit["sha256"] = row["sha256"]
        audits.append(audit)
    if not vectors:
        return np.zeros((0, BEHAVIOR_DIMENSION), dtype=np.float64), audits
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
    parser.add_argument("--iterations", type=int, default=800)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--l2", type=float, default=0.03)
    args = parser.parse_args()
    if args.l2 <= 0.0:
        raise SystemExit("--l2 must be positive")

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

    all_rows = [
        row
        for partitions in split_rows.values()
        for partition in ("train", "test")
        for row in partitions[partition]
    ]
    selected_shas = {row["sha256"] for row in all_rows}
    missing = {
        "entry_cache": len(selected_shas - set(entry_cache)),
        "endpoint_cache": len(selected_shas - set(endpoint_cache)),
        "features": len(selected_shas - set(feature_by_sha)),
    }
    if any(missing.values()):
        raise SystemExit(f"missing selected rows: {missing}")
    years = {
        str(feature_by_sha[sha].get("metadata_year", ""))
        for sha in selected_shas
    }
    if years != {"2017"}:
        raise SystemExit(
            f"this development evaluator accepts only 2017 rows; observed={sorted(years)}"
        )
    for row in all_rows:
        feature = feature_by_sha[row["sha256"]]
        row["metadata_label"] = feature.get("metadata_label", "")
        row["metadata_year"] = feature.get("metadata_year", "")

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
        "compact_plus_endpoint_behavior",
        "compact_plus_entry_plus_endpoint_behavior",
    )
    split_results = {variant: [] for variant in variants}
    prediction_rows = []
    reachability_rows = []

    for split_name in sorted(split_rows):
        train_rows = split_rows[split_name]["train"]
        validation_rows = split_rows[split_name]["test"]
        train_shas = {row["sha256"] for row in train_rows}
        evidence = fit_training_fold_components(
            entry_cache,
            train_shas,
            definition,
            active_components=ACTIVE_RELIABILITY_COMPONENTS,
        )
        entry_train, y_train = graph_matrix(
            train_rows, entry_cache, evidence, entry_args, True
        )
        entry_validation, y_validation = graph_matrix(
            validation_rows, entry_cache, evidence, entry_args, True
        )
        compact_train, compact_validation, compact_count = compact_matrix(
            train_rows,
            validation_rows,
            feature_by_sha,
            schema,
            feature_mode="base",
        )
        behavior_train, train_audit = behavior_matrix(train_rows, endpoint_cache)
        behavior_validation, validation_audit = behavior_matrix(
            validation_rows, endpoint_cache
        )
        entry_train, entry_validation = standardize_pair(
            entry_train, entry_validation
        )
        behavior_train, behavior_validation = standardize_pair(
            behavior_train, behavior_validation
        )
        entry_train = ENTRY_GRAPH_WEIGHT * entry_train
        entry_validation = ENTRY_GRAPH_WEIGHT * entry_validation
        matrices = {
            "compact_only": (compact_train, compact_validation),
            "compact_plus_entry": (
                np.concatenate([compact_train, entry_train], axis=1),
                np.concatenate([compact_validation, entry_validation], axis=1),
            ),
            "compact_plus_endpoint_behavior": (
                np.concatenate([compact_train, behavior_train], axis=1),
                np.concatenate([compact_validation, behavior_validation], axis=1),
            ),
            "compact_plus_entry_plus_endpoint_behavior": (
                np.concatenate([compact_train, entry_train, behavior_train], axis=1),
                np.concatenate([
                    compact_validation, entry_validation, behavior_validation
                ], axis=1),
            ),
        }
        for audit in train_audit:
            audit["split_name"] = split_name
            audit["partition"] = "train"
            reachability_rows.append(audit)
        for audit in validation_audit:
            audit["split_name"] = split_name
            audit["partition"] = "test"
            reachability_rows.append(audit)

        seed_metrics = {variant: [] for variant in variants}
        for seed in args.seeds:
            scores = {}
            for variant, (x_train, x_validation) in matrices.items():
                scores[variant] = predict_logistic(
                    x_train,
                    y_train,
                    x_validation,
                    seed,
                    args.iterations,
                    args.learning_rate,
                    args.l2,
                )
                seed_metrics[variant].append(
                    metrics(y_validation, scores[variant], threshold=THRESHOLD)
                )
                for row, label, value in zip(
                    validation_rows, y_validation, scores[variant]
                ):
                    prediction_rows.append({
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
            summary = {
                "variant": variant,
                "split_name": split_name,
                "train_count": len(train_rows),
                "validation_count": len(validation_rows),
                "compact_feature_count": int(compact_count),
                "behavior_feature_count": BEHAVIOR_DIMENSION,
                "entry_graph_feature_count": int(entry_train.shape[1]),
                "threshold": THRESHOLD,
            }
            for name in (
                "accuracy", "balanced_accuracy", "precision", "recall", "f1", "auc"
            ):
                summary[f"{name}_mean"] = float(np.mean([
                    row[name] for row in seed_metrics[variant]
                ]))
            split_results[variant].append(summary)
        print(f"completed {split_name}", flush=True)

    ranked = []
    for variant in variants:
        rows = split_results[variant]
        ba = np.asarray([row["balanced_accuracy_mean"] for row in rows])
        ranked.append({
            "variant": variant,
            "mean_accuracy": float(np.mean([row["accuracy_mean"] for row in rows])),
            "mean_balanced_accuracy": float(ba.mean()),
            "minimum_split_balanced_accuracy": float(ba.min()),
            "std_balanced_accuracy": float(ba.std()),
            "stability_score": float(ba.mean() - ba.std()),
            "mean_auc": float(np.mean([row["auc_mean"] for row in rows])),
        })
    ranked.sort(key=lambda row: (
        -row["stability_score"], -row["mean_balanced_accuracy"], row["variant"]
    ))
    for rank, row in enumerate(ranked, 1):
        row["rank"] = rank

    summary_path = output / "endpoint_behavior_transition_2017_summary.csv"
    write_csv(
        summary_path,
        [
            "rank", "variant", "mean_accuracy", "mean_balanced_accuracy",
            "minimum_split_balanced_accuracy", "std_balanced_accuracy",
            "stability_score", "mean_auc",
        ],
        ranked,
    )
    per_split_path = output / "endpoint_behavior_transition_2017_per_split.csv"
    write_csv(
        per_split_path,
        [
            "variant", "split_name", "train_count", "validation_count",
            "compact_feature_count", "behavior_feature_count",
            "entry_graph_feature_count", "threshold", "accuracy_mean",
            "balanced_accuracy_mean", "precision_mean", "recall_mean",
            "f1_mean", "auc_mean",
        ],
        [row for variant in variants for row in split_results[variant]],
    )
    write_csv(
        output / "endpoint_behavior_reachability_audit.csv",
        [
            "split_name", "partition", "sha256", "sensitive_endpoint_count",
            "entry_unreachable_endpoint_count", "entry_reachable_endpoint_count",
            "reachability_unknown_endpoint_count", "unreachable_endpoint_ratio",
            "nonzero_shared_caller_pair_count",
            "nonzero_collapsed_path_pair_count", "collapsed_sensitive_path_count",
            "collapsed_path_state_count", "collapsed_path_state_cap_hit",
            "example_pair_feature_space",
        ],
        reachability_rows,
    )
    write_csv(
        output / "endpoint_behavior_transition_2017_predictions.csv",
        [
            "variant", "split_name", "sha256", "seed", "label", "probability",
            "threshold", "predicted_label",
        ],
        prediction_rows,
    )

    audit = {
        "schema_version": "1.0.0",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "phase": "P29B_ENDPOINT_BEHAVIOR_TRANSITION_2017_DEVELOPMENT",
        "formal_training_enabled": False,
        "protocol_status": "DEVELOPMENT_ONLY_NOT_OUTER_TESTED",
        "strict_future_year_isolation": True,
        "observed_years": sorted(years),
        "fixed_threshold": THRESHOLD,
        "threshold_tuning_performed": False,
        "endpoint_cache_hops": {"upstream_hops": 2, "downstream_hops": 1},
        "requested_hops": {"upstream_hops": "2_or_3", "downstream_hops": "1_or_2"},
        "semantic_role_vocabulary_size": ROLE_VOCABULARY_SIZE,
        "normal_application_function_role_id": NORMAL_ROLE,
        "collapsed_path_max_hops": MAX_COLLAPSED_PATH_HOPS,
        "collapsed_path_state_cap": PATH_STATE_CAP,
        "behavior_feature_dimension": BEHAVIOR_DIMENSION,
        "behavior_feature_definition": {
            "caller_histograms": list(STAGES),
            "pair_families": list(PAIR_FAMILIES),
            "pair_examples": [
                "device_identifier_access -> network_transmission_or_upload",
                "sms_send_receive -> network_transmission_or_upload",
                "command_execution -> file_read",
            ],
            "method_names_persisted": False,
            "direct_runtime_execution_claimed": False,
        },
        "variants": list(variants),
        "entry_graph_weight": ENTRY_GRAPH_WEIGHT,
        "l2": args.l2,
        "seeds": args.seeds,
        "iterations": args.iterations,
        "learning_rate": args.learning_rate,
        "entry_cache": str(paths["entry_cache"]),
        "entry_cache_sha256": sha256_file(paths["entry_cache"]),
        "endpoint_cache": str(paths["endpoint_cache"]),
        "endpoint_cache_sha256": sha256_file(paths["endpoint_cache"]),
        "features": str(paths["features"]),
        "features_sha256": sha256_file(paths["features"]),
        "schema": str(paths["schema"]),
        "schema_sha256": sha256_file(paths["schema"]),
        "splits": str(paths["splits"]),
        "splits_sha256": sha256_file(paths["splits"]),
        "ranked_variants": ranked,
        "outputs": {
            "summary": str(summary_path),
            "per_split": str(per_split_path),
            "predictions": str(output / "endpoint_behavior_transition_2017_predictions.csv"),
            "reachability_audit": str(output / "endpoint_behavior_reachability_audit.csv"),
        },
        "blocking_reasons": [
            "candidate_not_locked",
            "3_hop_2_hop_cache_not_built",
            "formal_dataset_not_evaluated",
        ],
        "notes": [
            "All endpoint neighbourhood statistics are label-independent graph summaries.",
            "Role IDs and the normal-function bucket are used; method names are not features.",
            "The endpoint cache currently retains two upstream and one downstream hop.",
            "A higher score would support the value of entry-unreachable endpoint context but would not prove runtime execution.",
        ],
    }
    audit_path = output / "endpoint_behavior_transition_2017_audit.json"
    audit_path.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print("rank variant mean_BA min_BA std_BA stability AUC", flush=True)
    for row in ranked:
        print(
            row["rank"], row["variant"],
            f'{row["mean_balanced_accuracy"]:.4f}',
            f'{row["minimum_split_balanced_accuracy"]:.4f}',
            f'{row["std_balanced_accuracy"]:.4f}',
            f'{row["stability_score"]:.4f}',
            f'{row["mean_auc"]:.4f}',
            flush=True,
        )


if __name__ == "__main__":
    main()
