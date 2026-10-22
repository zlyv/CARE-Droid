#!/usr/bin/env python3
"""Identify which information in the current Entry view adds value.

The existing Entry cache does not retain Android component types such as
Activity, Service, Receiver, or Provider.  It does retain entry-node flags,
semantic roles, bounded graph edges, and compact node/edge attributes.  This
script therefore separates only information that is genuinely available:

* entry presence/coverage statistics;
* global bounded-subgraph structure;
* sensitive-role support in the entry-guided subgraph;
* role-node/entry-node overlap;
* tier-level entry-to-sensitive path statistics;
* role-level entry-to-sensitive reachability statistics;
* the complete current Entry representation.

The classification threshold is fixed at 0.5.  Feature scaling, reliability,
and fitting use training rows only.  Long-form predictions, seed-averaged
per-APK comparisons, correction/regression counts, and group profiles are
written for post-hoc interpretation.  No APK is opened.
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
from run_dual_view_error_attribution import compact_matrix_named  # noqa: E402
from run_graph_pool_diagnostic import (  # noqa: E402
    bounded_path_features,
    graph_matrix,
    read_csv_rows,
    read_split_rows,
    standardize_pair,
)
from run_locked_dual_view_module_ablation import predict_logistic  # noqa: E402
from run_pilot_compact_eval import metrics  # noqa: E402
from tasa_droid.p3.a3_graph_model import build_cached_graph  # noqa: E402
from tasa_droid.p3.reliability import (  # noqa: E402
    PilotReliabilityDefinition,
    fit_training_fold_components,
    read_cache,
)


THRESHOLD = 0.5
ROLE_REACH_WIDTH = 4


def write_csv(path: Path, fields: list[str], rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def entry_layout(
    role_vocabulary_size: int = ROLE_VOCABULARY_SIZE,
    include_reliability: bool = True,
) -> dict[str, np.ndarray]:
    """Return exact column groups for role-stratified Entry pooling."""

    node_width = 4 + role_vocabulary_size + int(include_reliability)
    global_width = 3 * node_width + 3 + 2
    role_width = 5 + int(include_reliability)
    full_width = global_width + role_vocabulary_size * role_width

    entry_presence = np.asarray([
        3,
        node_width + 3,
        2 * node_width + 3,
    ], dtype=np.int64)
    role_support = []
    for statistic_offset in (0, node_width, 2 * node_width):
        role_support.extend(
            statistic_offset + 4 + role
            for role in range(role_vocabulary_size)
        )
    role_support.extend(
        global_width + role * role_width
        for role in range(role_vocabulary_size)
    )
    role_support = np.asarray(role_support, dtype=np.int64)
    role_context = np.asarray([
        global_width + role * role_width + offset
        for role in range(role_vocabulary_size)
        for offset in (1, 2, 3, 5)
    ], dtype=np.int64)
    role_node_entry_overlap = np.asarray([
        global_width + role * role_width + 4
        for role in range(role_vocabulary_size)
    ], dtype=np.int64)
    global_structure = []
    for statistic_offset in (0, node_width, 2 * node_width):
        global_structure.extend(
            statistic_offset + base_offset
            for base_offset in (0, 1, 2, node_width - 1)
        )
    global_structure.extend(range(3 * node_width, global_width))
    global_structure = np.asarray(global_structure, dtype=np.int64)
    return {
        "entry_presence": entry_presence,
        "entry_global_structure": global_structure,
        "sensitive_role_support": role_support,
        "sensitive_role_context": role_context,
        "role_node_entry_overlap": role_node_entry_overlap,
        "full_entry": np.arange(full_width, dtype=np.int64),
        "node_width": np.asarray([node_width], dtype=np.int64),
        "global_width": np.asarray([global_width], dtype=np.int64),
        "role_width": np.asarray([role_width], dtype=np.int64),
        "full_width": np.asarray([full_width], dtype=np.int64),
    }


def entry_feature_names(role_names: list[str]) -> list[str]:
    if len(role_names) != ROLE_VOCABULARY_SIZE:
        raise ValueError("semantic role name count does not match cache vocabulary")
    node_names = [
        "node_external",
        "node_mapped_sensitive",
        "node_tier",
        "node_is_entry",
        *[f"node_role__{role}" for role in role_names],
        "node_reliability",
    ]
    names = []
    for statistic in ("mean", "max", "std"):
        names.extend(f"global__{statistic}__{name}" for name in node_names)
    names.extend([
        "global__edge_mean__icc",
        "global__edge_mean__reflection",
        "global__edge_mean__non_direct",
        "global__log1p_node_count",
        "global__log1p_edge_count",
    ])
    for role in role_names:
        names.extend([
            f"role__{role}__support",
            f"role__{role}__external_mean",
            f"role__{role}__mapped_mean",
            f"role__{role}__tier_mean",
            f"role__{role}__entry_overlap_mean",
            f"role__{role}__reliability_mean",
        ])
    if len(names) != 299:
        raise ValueError(f"unexpected Entry feature name count: {len(names)}")
    return names


def entry_role_reachability_features(
    graph,
    role_vocabulary_size: int = ROLE_VOCABULARY_SIZE,
    max_hops: int = 4,
) -> np.ndarray:
    """Summarize entry-to-role reachability without claiming data flow."""

    x = np.asarray(graph.x, dtype=np.float64)
    result = np.zeros(
        (role_vocabulary_size, ROLE_REACH_WIDTH), dtype=np.float64
    )
    if len(x) == 0:
        return result.reshape(-1)
    source = np.asarray(graph.source, dtype=np.int64)
    target = np.asarray(graph.target, dtype=np.int64)
    adjacency = [[] for _ in range(len(x))]
    for src, dst in zip(source, target):
        adjacency[int(src)].append(int(dst))
    entries = list(map(int, np.flatnonzero(x[:, 3] > 0.5)))
    if not entries:
        return result.reshape(-1)
    role_matrix = x[:, 4:4 + role_vocabulary_size] > 0.5

    reachable_counts = np.zeros((len(entries), role_vocabulary_size))
    minimum_hops = np.full(
        (len(entries), role_vocabulary_size), max_hops + 1, dtype=np.float64
    )
    internal_ratios = np.zeros((len(entries), role_vocabulary_size))
    for entry_index, entry in enumerate(entries):
        distances = {entry: 0}
        frontier = [entry]
        for hop in range(1, max_hops + 1):
            next_frontier = []
            for node in frontier:
                for neighbor in adjacency[node]:
                    if neighbor not in distances:
                        distances[neighbor] = hop
                        next_frontier.append(neighbor)
            frontier = next_frontier
            if not frontier:
                break
        for role in range(role_vocabulary_size):
            nodes = [
                node for node, distance in distances.items()
                if node != entry and role_matrix[node, role]
            ]
            if not nodes:
                continue
            reachable_counts[entry_index, role] = len(nodes)
            minimum_hops[entry_index, role] = min(distances[node] for node in nodes)
            internal_ratios[entry_index, role] = float(
                np.mean(x[nodes, 0] < 0.5)
            )

    for role in range(role_vocabulary_size):
        present = reachable_counts[:, role] > 0.0
        result[role] = (
            float(np.mean(present)),
            float(
                np.mean(minimum_hops[present, role]) / (max_hops + 1)
            ) if np.any(present) else 0.0,
            float(np.log1p(np.mean(reachable_counts[:, role]))),
            float(np.mean(internal_ratios[present, role]))
            if np.any(present) else 0.0,
        )
    return result.reshape(-1)


def build_graphs_and_derived(
    rows: list[dict],
    cache: dict,
    evidence: dict,
    definition: PilotReliabilityDefinition,
) -> tuple[np.ndarray, np.ndarray]:
    path_rows = []
    reachability_rows = []
    reliability_weights = {
        name: 1.0 for name in ACTIVE_RELIABILITY_COMPONENTS
    }
    for row in rows:
        payload = cache[row["sha256"]]["payload"]
        graph = build_cached_graph(
            payload,
            evidence["components_by_node_id"],
            evidence["global_training_prior"],
            reliability_weights,
            definition.r_min,
            definition.r_max,
            ROLE_VOCABULARY_SIZE,
            include_reliability=True,
            reliability_components=ACTIVE_RELIABILITY_COMPONENTS,
        )
        path_rows.append(bounded_path_features(graph))
        reachability_rows.append(entry_role_reachability_features(graph))
    return (
        np.asarray(path_rows, dtype=np.float64),
        np.asarray(reachability_rows, dtype=np.float64),
    )


def transition_name(label: int, compact_pred: int, fusion_pred: int) -> str:
    compact_correct = compact_pred == label
    fusion_correct = fusion_pred == label
    if compact_correct and fusion_correct:
        return "STILL_CORRECT"
    if not compact_correct and fusion_correct:
        return "CORRECTED"
    if compact_correct and not fusion_correct:
        return "REGRESSED"
    return "STILL_WRONG"


def metric_summary(rows: list[dict]) -> list[dict]:
    result = []
    variants = sorted({row["variant"] for row in rows})
    for variant in variants:
        selected = [row for row in rows if row["variant"] == variant]
        ba = np.asarray([row["balanced_accuracy_mean"] for row in selected])
        result.append({
            "variant": variant,
            "mean_accuracy": float(np.mean([
                row["accuracy_mean"] for row in selected
            ])),
            "mean_balanced_accuracy": float(ba.mean()),
            "minimum_split_balanced_accuracy": float(ba.min()),
            "std_balanced_accuracy": float(ba.std()),
            "stability_score": float(ba.mean() - ba.std()),
            "mean_auc": float(np.mean([row["auc_mean"] for row in selected])),
        })
    result.sort(
        key=lambda row: (
            -row["mean_balanced_accuracy"], -row["mean_auc"], row["variant"]
        )
    )
    for rank, row in enumerate(result, 1):
        row["rank"] = rank
    return result


def load_role_names(mapping_path: Path) -> list[str]:
    lines = mapping_path.read_text(encoding="utf-8").splitlines()
    roles = []
    in_roles = False
    for line in lines:
        if line.strip() == "roles:":
            in_roles = True
            continue
        if in_roles and line.startswith("  - "):
            roles.append(line.split("-", 1)[1].strip())
            continue
        if in_roles and line and not line.startswith(" "):
            break
    return sorted(roles)


def paired_feature_effects(
    rows: list[dict],
    transition_by_key: dict[tuple[str, str], str],
    role_names: list[str],
) -> list[dict]:
    comparisons = (
        ("CORRECTED_VS_STILL_WRONG", "CORRECTED", "STILL_WRONG"),
        ("REGRESSED_VS_STILL_CORRECT", "REGRESSED", "STILL_CORRECT"),
    )
    result = []
    by_split = defaultdict(list)
    for row in rows:
        by_split[row["split_name"]].append(row)
    full_names = entry_feature_names(role_names)
    path_names = [
        f"path__tier{tier}__{metric}"
        for tier in range(3)
        for metric in ("mean_count_log", "mean_min_hop", "reach_rate", "internal_rate")
    ] + [
        "path__any_sensitive_reach",
        "path__any_sensitive_count_log",
        "path__any_sensitive_internal_rate",
    ]
    reach_names = [
        f"entry_role__{role}__{metric}"
        for role in role_names
        for metric in ("reach_rate", "mean_min_hop", "mean_count_log", "internal_rate")
    ]
    feature_spaces = {
        "full_entry": full_names,
        "path_statistics": path_names,
        "entry_role_reachability": reach_names,
    }
    for split_name, split_rows in sorted(by_split.items()):
        year = split_rows[0]["year"] if split_rows else ""
        for feature_group, names in feature_spaces.items():
            group_rows = [row for row in split_rows if row["group"] == feature_group]
            if not group_rows:
                continue
            for comparison, error_transition, reference_transition in comparisons:
                error_values = [
                    np.asarray(json.loads(row["values_json"]), dtype=np.float64)
                    for row in group_rows
                    if transition_by_key[(row["split_name"], row["sha256"])]
                    == error_transition
                ]
                reference_values = [
                    np.asarray(json.loads(row["values_json"]), dtype=np.float64)
                    for row in group_rows
                    if transition_by_key[(row["split_name"], row["sha256"])]
                    == reference_transition
                ]
                if not error_values or not reference_values:
                    continue
                error_matrix = np.vstack(error_values)
                reference_matrix = np.vstack(reference_values)
                delta = error_matrix.mean(axis=0) - reference_matrix.mean(axis=0)
                for index, feature in enumerate(names):
                    result.append({
                        "split_name": split_name,
                        "year": year,
                        "comparison": comparison,
                        "feature_group": feature_group,
                        "feature": feature,
                        "error_n": len(error_values),
                        "reference_n": len(reference_values),
                        "error_mean": float(error_matrix[:, index].mean()),
                        "reference_mean": float(reference_matrix[:, index].mean()),
                        "standardized_mean_delta": float(delta[index]),
                    })
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--entry-cache", required=True, type=Path)
    parser.add_argument("--features", required=True, type=Path)
    parser.add_argument("--schema", required=True, type=Path)
    parser.add_argument(
        "--mapping",
        type=Path,
        default=PROJECT_ROOT / "configs" / "protocol" / "semantic_mapping_v1.yaml",
    )
    parser.add_argument("--splits", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--iterations", type=int, default=1500)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--l2", type=float, default=0.03)
    parser.add_argument(
        "--entry-weights", nargs="+", type=float,
        default=[0.25, 0.5, 1.0, 2.0],
    )
    args = parser.parse_args()
    if not args.seeds or not args.entry_weights:
        raise SystemExit("seeds and entry weights must be non-empty")
    if any(weight <= 0.0 for weight in args.entry_weights):
        raise SystemExit("entry weights must be positive")

    paths = {
        "entry_cache": args.entry_cache.expanduser().resolve(strict=True),
        "features": args.features.expanduser().resolve(strict=True),
        "schema": args.schema.expanduser().resolve(strict=True),
        "mapping": args.mapping.expanduser().resolve(strict=True),
        "splits": args.splits.expanduser().resolve(strict=True),
    }
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    cache = read_cache(paths["entry_cache"])
    feature_rows = read_csv_rows(paths["features"])
    feature_by_sha = {row["sha256"].casefold(): row for row in feature_rows}
    schema = json.loads(paths["schema"].read_text(encoding="utf-8"))
    role_names = load_role_names(paths["mapping"])
    if len(role_names) != ROLE_VOCABULARY_SIZE:
        raise SystemExit(
            f"expected {ROLE_VOCABULARY_SIZE} semantic roles; observed={len(role_names)}"
        )
    split_rows = read_split_rows(paths["splits"])
    attach_audit_metadata(cache, feature_by_sha)

    selected_shas = {
        row["sha256"]
        for groups in split_rows.values()
        for partition in ("train", "test")
        for row in groups[partition]
    }
    missing = {
        "entry_cache": len(selected_shas - set(cache)),
        "features": len(selected_shas - set(feature_by_sha)),
    }
    if any(missing.values()):
        raise SystemExit(f"missing selected rows: {missing}")
    observed_years = sorted({
        str(feature_by_sha[sha].get("metadata_year", ""))
        for sha in selected_shas
    })

    definition = PilotReliabilityDefinition(
        alpha_prior=1.0,
        kappa_support=20.0,
        kappa_context=20.0,
        r_min=0.10,
        r_max=0.95,
    )
    entry_args = entry_model_args(definition)
    layout = entry_layout()
    variants = [
        "compact_only",
        "entry_full_only",
        "compact_plus_entry_presence",
        "compact_plus_entry_global_structure",
        "compact_plus_sensitive_role_support",
        "compact_plus_sensitive_role_context",
        "compact_plus_role_node_entry_overlap",
        "compact_plus_path_statistics",
        "compact_plus_entry_role_reachability",
    ] + [
        f"compact_plus_full_entry_w{weight:g}"
        for weight in args.entry_weights
    ]

    split_results = []
    prediction_rows = []
    group_value_rows = []
    dimension_rows = []

    for split_name in sorted(split_rows):
        train_rows = split_rows[split_name]["train"]
        test_rows = split_rows[split_name]["test"]
        if not train_rows or not test_rows:
            raise SystemExit(f"empty partition: {split_name}")
        for row in train_rows + test_rows:
            row["metadata_label"] = feature_by_sha[row["sha256"]].get(
                "metadata_label", ""
            )
        train_shas = {row["sha256"] for row in train_rows}
        evidence = fit_training_fold_components(
            cache,
            train_shas,
            definition,
            active_components=ACTIVE_RELIABILITY_COMPONENTS,
        )
        full_train, y_train = graph_matrix(
            train_rows, cache, evidence, entry_args, True
        )
        full_test, y_test = graph_matrix(
            test_rows, cache, evidence, entry_args, True
        )
        expected_width = int(layout["full_width"][0])
        if full_train.shape[1] != expected_width:
            raise SystemExit(
                f"unexpected Entry width {full_train.shape[1]}; "
                f"expected role-stratified width {expected_width}"
            )
        compact_train, compact_test, compact_names = compact_matrix_named(
            train_rows, test_rows, feature_by_sha, schema
        )
        path_train, reach_train = build_graphs_and_derived(
            train_rows, cache, evidence, definition
        )
        path_test, reach_test = build_graphs_and_derived(
            test_rows, cache, evidence, definition
        )
        full_train, full_test = standardize_pair(full_train, full_test)
        path_train, path_test = standardize_pair(path_train, path_test)
        reach_train, reach_test = standardize_pair(reach_train, reach_test)

        group_matrices = {
            "entry_presence": (
                full_train[:, layout["entry_presence"]],
                full_test[:, layout["entry_presence"]],
            ),
            "entry_global_structure": (
                full_train[:, layout["entry_global_structure"]],
                full_test[:, layout["entry_global_structure"]],
            ),
            "sensitive_role_support": (
                full_train[:, layout["sensitive_role_support"]],
                full_test[:, layout["sensitive_role_support"]],
            ),
            "sensitive_role_context": (
                full_train[:, layout["sensitive_role_context"]],
                full_test[:, layout["sensitive_role_context"]],
            ),
            "role_node_entry_overlap": (
                full_train[:, layout["role_node_entry_overlap"]],
                full_test[:, layout["role_node_entry_overlap"]],
            ),
            "path_statistics": (path_train, path_test),
            "entry_role_reachability": (reach_train, reach_test),
            "full_entry": (full_train, full_test),
        }
        matrices = {
            "compact_only": (compact_train, compact_test),
            "entry_full_only": (full_train, full_test),
        }
        for group_name in (
            "entry_presence",
            "entry_global_structure",
            "sensitive_role_support",
            "sensitive_role_context",
            "role_node_entry_overlap",
            "path_statistics",
            "entry_role_reachability",
        ):
            group_train, group_test = group_matrices[group_name]
            matrices[f"compact_plus_{group_name}"] = (
                np.concatenate([compact_train, group_train], axis=1),
                np.concatenate([compact_test, group_test], axis=1),
            )
        for weight in args.entry_weights:
            matrices[f"compact_plus_full_entry_w{weight:g}"] = (
                np.concatenate([compact_train, weight * full_train], axis=1),
                np.concatenate([compact_test, weight * full_test], axis=1),
            )

        dimension_rows.append({
            "split_name": split_name,
            "compact_feature_count": len(compact_names),
            **{
                f"{name}_feature_count": int(values[0].shape[1])
                for name, values in group_matrices.items()
            },
        })
        for group_name, (_, test_values) in group_matrices.items():
            for row, values in zip(test_rows, test_values):
                group_value_rows.append({
                    "split_name": split_name,
                    "sha256": row["sha256"],
                    "year": feature_by_sha[row["sha256"]].get(
                        "metadata_year", ""
                    ),
                    "group": group_name,
                    "l1_mean": float(np.mean(np.abs(values))),
                    "l2_norm": float(np.linalg.norm(values)),
                    "nonzero_fraction": float(np.mean(np.abs(values) > 1e-12)),
                    "values_json": json.dumps(values.tolist(), separators=(",", ":")),
                })

        seed_metrics = {variant: [] for variant in variants}
        for seed in args.seeds:
            for variant in variants:
                x_train, x_test = matrices[variant]
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
                    metrics(y_test, probability, threshold=THRESHOLD)
                )
                for row, label, value in zip(test_rows, y_test, probability):
                    prediction_rows.append({
                        "variant": variant,
                        "split_name": split_name,
                        "sha256": row["sha256"],
                        "year": feature_by_sha[row["sha256"]].get(
                            "metadata_year", ""
                        ),
                        "seed": seed,
                        "label": int(label),
                        "probability": float(value),
                        "threshold": THRESHOLD,
                        "predicted_label": int(value >= THRESHOLD),
                    })
        for variant in variants:
            values = seed_metrics[variant]
            row = {
                "variant": variant,
                "split_name": split_name,
                "train_count": len(train_rows),
                "test_count": len(test_rows),
                "feature_count": int(matrices[variant][0].shape[1]),
                "threshold": THRESHOLD,
            }
            for metric_name in (
                "accuracy", "balanced_accuracy", "precision", "recall", "f1", "auc"
            ):
                row[f"{metric_name}_mean"] = float(np.mean([
                    value[metric_name] for value in values
                ]))
            split_results.append(row)
        print(f"completed {split_name}", flush=True)

    # Average seeds before paired per-APK comparison.
    probability_groups: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    metadata = {}
    for row in prediction_rows:
        key = (row["split_name"], row["sha256"], row["variant"])
        probability_groups[key].append(float(row["probability"]))
        metadata[(row["split_name"], row["sha256"])] = (
            row["year"], int(row["label"])
        )
    control_weight = 0.5 if 0.5 in args.entry_weights else args.entry_weights[0]
    fusion_variant = f"compact_plus_full_entry_w{control_weight:g}"
    paired_rows = []
    for split_name, sha256 in sorted(metadata):
        year, label = metadata[(split_name, sha256)]
        compact_probability = float(np.mean(
            probability_groups[(split_name, sha256, "compact_only")]
        ))
        entry_probability = float(np.mean(
            probability_groups[(split_name, sha256, "entry_full_only")]
        ))
        fusion_probability = float(np.mean(
            probability_groups[(split_name, sha256, fusion_variant)]
        ))
        compact_pred = int(compact_probability >= THRESHOLD)
        entry_pred = int(entry_probability >= THRESHOLD)
        fusion_pred = int(fusion_probability >= THRESHOLD)
        paired_rows.append({
            "split_name": split_name,
            "year": year,
            "sha256": sha256,
            "label": label,
            "compact_probability": compact_probability,
            "compact_predicted_label": compact_pred,
            "entry_probability": entry_probability,
            "entry_predicted_label": entry_pred,
            "fusion_variant": fusion_variant,
            "fusion_probability": fusion_probability,
            "fusion_predicted_label": fusion_pred,
            "transition": transition_name(label, compact_pred, fusion_pred),
            "probability_delta_fusion_minus_compact": (
                fusion_probability - compact_probability
            ),
        })

    transition_by_key = {
        (row["split_name"], row["sha256"]): row["transition"]
        for row in paired_rows
    }
    correction_counter = Counter(
        (row["split_name"], row["year"], row["transition"], row["label"])
        for row in paired_rows
    )
    correction_rows = [
        {
            "split_name": split_name,
            "year": year,
            "transition": transition,
            "label": label,
            "sample_count": count,
        }
        for (split_name, year, transition, label), count
        in sorted(correction_counter.items())
    ]
    profile_groups: dict[tuple, list[dict]] = defaultdict(list)
    for row in group_value_rows:
        transition = transition_by_key[(row["split_name"], row["sha256"])]
        profile_groups[(
            row["split_name"], row["year"], transition, row["group"]
        )].append(row)
    profile_rows = []
    for (split_name, year, transition, group), rows in sorted(profile_groups.items()):
        profile_rows.append({
            "split_name": split_name,
            "year": year,
            "transition": transition,
            "group": group,
            "sample_count": len(rows),
            "l1_mean": float(np.mean([row["l1_mean"] for row in rows])),
            "l2_norm_mean": float(np.mean([row["l2_norm"] for row in rows])),
            "nonzero_fraction_mean": float(np.mean([
                row["nonzero_fraction"] for row in rows
            ])),
        })

    feature_effect_rows = paired_feature_effects(
        group_value_rows, transition_by_key, role_names
    )

    ranked = metric_summary(split_results)
    summary_path = output / "entry_increment_ablation_summary.csv"
    split_path = output / "entry_increment_ablation_per_split.csv"
    prediction_path = output / "entry_increment_ablation_predictions.csv"
    paired_path = output / "entry_increment_apk_comparison.csv"
    correction_path = output / "entry_increment_correction_summary.csv"
    profile_path = output / "entry_increment_group_profiles.csv"
    dimension_path = output / "entry_increment_dimensions.csv"
    feature_effect_path = output / "entry_increment_feature_effects.csv"
    write_csv(summary_path, list(ranked[0]), ranked)
    write_csv(split_path, list(split_results[0]), split_results)
    write_csv(prediction_path, list(prediction_rows[0]), prediction_rows)
    write_csv(paired_path, list(paired_rows[0]), paired_rows)
    write_csv(correction_path, list(correction_rows[0]), correction_rows)
    write_csv(profile_path, list(profile_rows[0]), profile_rows)
    write_csv(dimension_path, list(dimension_rows[0]), dimension_rows)
    if feature_effect_rows:
        write_csv(
            feature_effect_path,
            list(feature_effect_rows[0]),
            feature_effect_rows,
        )

    audit = {
        "schema_version": "1.0.0",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "phase": "P50_ENTRY_INCREMENT_INFORMATION_ABLATION",
        "formal_training_enabled": False,
        "strict_test_isolation": True,
        "observed_years": observed_years,
        "protocol_status": (
            "DEVELOPMENT_2017_ONLY" if observed_years == ["2017"]
            else "RETROSPECTIVE_FIXED2017_ERROR_ANALYSIS_NOT_FOR_SELECTION"
        ),
        "fixed_threshold": THRESHOLD,
        "threshold_tuning_performed": False,
        "classifier": {
            "type": "custom_l2_logistic",
            "seeds": args.seeds,
            "iterations": args.iterations,
            "learning_rate": args.learning_rate,
            "l2": args.l2,
        },
        "entry_weights": args.entry_weights,
        "entry_feature_availability": {
            "android_component_entry_type": "NOT_RETAINED_IN_CURRENT_CACHE",
            "entry_presence": "AVAILABLE",
            "entry_guided_sensitive_role_support": "AVAILABLE",
            "tier_path_statistics": "DERIVED_FROM_CURRENT_CACHE",
            "entry_role_reachability": "DERIVED_FROM_CURRENT_CACHE",
            "explicit_trigger_type_by_role_pair": "NOT_RETAINED_IN_CURRENT_CACHE",
        },
        "current_full_entry_representation": {
            "pool_mode": "role_stratified",
            "explicit_bounded_path_features_included": False,
            "explicit_entry_role_reachability_included": False,
            "interpretation": (
                "Roles are pooled over an entry-guided bounded subgraph; "
                "the current model does not explicitly encode Android trigger type."
            ),
        },
        "ranked_variants": ranked,
        "inputs": {
            name: {"path": str(path), "sha256": sha256_file(path)}
            for name, path in paths.items()
        },
        "outputs": {
            "summary": str(summary_path),
            "per_split": str(split_path),
            "predictions": str(prediction_path),
            "apk_comparison": str(paired_path),
            "correction_summary": str(correction_path),
            "group_profiles": str(profile_path),
            "dimensions": str(dimension_path),
            "feature_effects": str(feature_effect_path),
        },
        "notes": [
            "Entry block scaling changes effective L2 regularization for this logistic classifier.",
            "The weight sweep tests scale sensitivity; it is not a Random Forest weight experiment.",
            "Per-APK probabilities are averaged across seeds before correction/regression assignment.",
            "Future-year outputs are retrospective explanation only and must not select a new protocol.",
            "No APK, raw graph, raw signature, or test label is used during feature construction.",
        ],
    }
    audit_path = output / "entry_increment_ablation_audit.json"
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
