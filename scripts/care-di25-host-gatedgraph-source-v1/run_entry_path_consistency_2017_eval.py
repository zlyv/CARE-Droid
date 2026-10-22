#!/usr/bin/env python3
"""Evaluate layered semantic path consistency on the existing entry-view cache.

The module keeps only semantic role IDs, hop layers, edge categories, and
bounded aggregate counts.  It never stores method names or raw paths.  For
each APK it builds a directed entry-reachable subgraph and summarizes:

* role support at hop 0..H;
* role-to-role transitions between successive reachable layers;
* boundary and edge-type transitions by hop;
* sensitive-tier reachability and path concentration statistics.

The evaluation compares four fixed-threshold variants on chronological 2017
rolling development splits:

1. compact_only
2. compact_plus_entry
3. compact_plus_path_consistency
4. compact_plus_entry_plus_path_consistency

All graph reliability and feature standardization are fit on each training
partition only.  The path module itself is label-independent.
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
from run_graph_pool_diagnostic import (  # noqa: E402
    compact_matrix,
    graph_matrix,
    read_csv_rows,
    read_split_rows,
    standardize_pair,
)
from run_pilot_compact_eval import metrics  # noqa: E402
from run_uncertainty_gated_dual_view_2017_eval import predict_logistic  # noqa: E402
from tasa_droid.p3.a3_mvp import stable_id  # noqa: E402
from tasa_droid.p3.reliability import (  # noqa: E402
    PilotReliabilityDefinition,
    fit_training_fold_components,
    read_cache,
)


THRESHOLD = 0.5
ENTRY_GRAPH_WEIGHT = 0.5
DEFAULT_MAX_HOPS = 4
NORMAL_ROLE = ROLE_VOCABULARY_SIZE
ROLE_COUNT = ROLE_VOCABULARY_SIZE + 1
EDGE_TYPE_NAMES = (
    "DIRECT_CALL",
    "ICC_COMPONENT_DISPATCH",
    "ICC_CONTENT_PROVIDER",
    "ICC_PENDING_INTENT",
    "REFLECTION_INVOKE",
)
EDGE_TYPE_IDS = np.asarray([stable_id(name) for name in EDGE_TYPE_NAMES], dtype=np.uint64)


def _roles(values) -> list[int]:
    result = sorted({
        int(value) for value in (values or [])
        if 0 <= int(value) < ROLE_VOCABULARY_SIZE
    })
    return result or [NORMAL_ROLE]


def _entropy(hist: np.ndarray) -> float:
    total = float(hist.sum())
    if total <= 0.0:
        return 0.0
    probabilities = hist[hist > 0.0] / total
    return float(-(probabilities * np.log(probabilities)).sum())


def entry_path_consistency_vector(
    payload: dict,
    max_hops: int = DEFAULT_MAX_HOPS,
    role_vocabulary_size: int = ROLE_VOCABULARY_SIZE,
) -> tuple[np.ndarray, dict]:
    """Return a name-free, layered semantic path vector and an audit record."""

    node_ids = list(payload.get("node_ids") or [])
    base = np.asarray(payload.get("node_base_features") or [], dtype=np.float64)
    roles = list(payload.get("node_role_ids") or [])
    edge_index = payload.get("edge_index") or [[], []]
    source = np.asarray(edge_index[0], dtype=np.int64)
    target = np.asarray(edge_index[1], dtype=np.int64)
    edge_features = np.asarray(payload.get("edge_features") or [], dtype=np.float64)
    edge_type_ids = np.asarray(payload.get("edge_type_ids") or [], dtype=np.uint64)
    n_nodes = len(node_ids)
    if n_nodes == 0:
        dimension = (max_hops + 1) * ROLE_COUNT + ROLE_COUNT * ROLE_COUNT
        dimension += max_hops * (3 + 4 + len(EDGE_TYPE_NAMES) + 3) + 12
        return np.zeros(dimension, dtype=np.float64), {
            "node_count": 0, "entry_count": 0, "reachable_node_count": 0,
            "reachable_sensitive_node_count": 0, "path_edge_count": 0,
        }
    if base.shape != (n_nodes, 4):
        raise ValueError("cached node_base_features must have shape [node_count, 4]")
    if len(roles) != n_nodes:
        raise ValueError("cached node_role_ids length mismatch")
    if source.shape != target.shape:
        raise ValueError("cached edge_index source/target length mismatch")
    if len(edge_features) != len(source) or len(edge_type_ids) != len(source):
        raise ValueError("cached edge arrays have inconsistent lengths")

    adjacency = [[] for _ in range(n_nodes)]
    for src, dst in zip(source, target):
        if 0 <= int(src) < n_nodes and 0 <= int(dst) < n_nodes:
            adjacency[int(src)].append(int(dst))
    entries = np.flatnonzero(base[:, 3] > 0.5).astype(int).tolist()
    depths: dict[int, int] = {}
    for entry in entries:
        if entry in depths and depths[entry] == 0:
            continue
        depths[entry] = 0
        frontier = [entry]
        for hop in range(1, max_hops + 1):
            next_frontier = []
            for node in frontier:
                for neighbor in adjacency[node]:
                    if neighbor not in depths:
                        depths[neighbor] = hop
                        next_frontier.append(neighbor)
            frontier = next_frontier
            if not frontier:
                break

    level_roles = np.zeros((max_hops + 1, ROLE_COUNT), dtype=np.float64)
    for node, depth in depths.items():
        if 0 <= depth <= max_hops:
            for role in _roles(roles[node]):
                level_roles[depth, role] += 1.0

    transitions = np.zeros((ROLE_COUNT, ROLE_COUNT), dtype=np.float64)
    tier_by_hop = np.zeros((max_hops, 3), dtype=np.float64)
    boundary_by_hop = np.zeros((max_hops, 4), dtype=np.float64)
    edge_type_by_hop = np.zeros((max_hops, len(EDGE_TYPE_NAMES)), dtype=np.float64)
    flag_by_hop = np.zeros((max_hops, 3), dtype=np.float64)
    path_edge_count = 0
    forward_edge_count = 0
    back_or_same_edge_count = 0
    role_adjacent_edge_count = 0
    sensitive_nodes = 0

    for node, depth in depths.items():
        if depth <= max_hops and int(round(np.clip(base[node, 2], 0.0, 1.0) * 4.0)) < 3:
            sensitive_nodes += 1

    for edge_number, (src_value, dst_value) in enumerate(zip(source, target)):
        src = int(src_value)
        dst = int(dst_value)
        if src not in depths or dst not in depths:
            continue
        src_depth = depths[src]
        dst_depth = depths[dst]
        if dst_depth <= src_depth or dst_depth > max_hops:
            back_or_same_edge_count += 1
            continue
        hop = dst_depth - 1
        if hop < 0 or hop >= max_hops:
            continue
        path_edge_count += 1
        forward_edge_count += 1
        source_roles = _roles(roles[src])
        target_roles = _roles(roles[dst])
        if source_roles != [NORMAL_ROLE] or target_roles != [NORMAL_ROLE]:
            role_adjacent_edge_count += 1
        for left in source_roles:
            for right in target_roles:
                transitions[left, right] += 1.0
        tier = int(round(np.clip(base[dst, 2], 0.0, 1.0) * 4.0))
        if tier < 3:
            tier_by_hop[hop, tier] += 1.0
        external_boundary = int(base[src, 0] > 0.5) * 2 + int(base[dst, 0] > 0.5)
        boundary_by_hop[hop, external_boundary] += 1.0
        for edge_type, edge_type_id in enumerate(EDGE_TYPE_IDS):
            if edge_type_ids[edge_number] == edge_type_id:
                edge_type_by_hop[hop, edge_type] += 1.0
                break
        if edge_features.shape[1] >= 3:
            flag_by_hop[hop] += (edge_features[edge_number, :3] > 0.0).astype(float)

    reachable = np.asarray(list(depths), dtype=np.int64)
    reachable_role_hist = level_roles.sum(axis=0)
    terminal_role_hist = level_roles[max_hops].copy()
    total_role_transitions = float(transitions.sum())
    nonnormal_role_transitions = float(
        transitions.sum() - transitions[NORMAL_ROLE, NORMAL_ROLE]
    )
    semantic_transition_ratio = (
        nonnormal_role_transitions / total_role_transitions
        if total_role_transitions else 0.0
    )
    path_features = np.concatenate([
        np.log1p(level_roles).reshape(-1),
        np.log1p(transitions).reshape(-1),
        np.log1p(tier_by_hop).reshape(-1),
        np.log1p(boundary_by_hop).reshape(-1),
        np.log1p(edge_type_by_hop).reshape(-1),
        np.log1p(flag_by_hop).reshape(-1),
        np.asarray([
            np.log1p(len(node_ids)),
            np.log1p(len(source)),
            np.log1p(len(entries)),
            np.log1p(len(reachable)),
            np.log1p(sensitive_nodes),
            np.log1p(path_edge_count),
            np.log1p(forward_edge_count),
            np.log1p(back_or_same_edge_count),
            semantic_transition_ratio,
            _entropy(reachable_role_hist),
            _entropy(terminal_role_hist),
            float(sensitive_nodes / max(len(reachable), 1)),
        ], dtype=np.float64),
    ])
    expected_dimension = (
        (max_hops + 1) * ROLE_COUNT
        + ROLE_COUNT * ROLE_COUNT
        + max_hops * (3 + 4 + len(EDGE_TYPE_NAMES) + 3)
        + 12
    )
    if path_features.shape != (expected_dimension,):
        raise ValueError(
            f"unexpected path feature dimension {path_features.shape}; "
            f"expected {(expected_dimension,)}"
        )
    audit = {
        "node_count": n_nodes,
        "edge_count": int(len(source)),
        "entry_count": len(entries),
        "reachable_node_count": int(len(reachable)),
        "reachable_sensitive_node_count": int(sensitive_nodes),
        "path_edge_count": int(path_edge_count),
        "back_or_same_edge_count": int(back_or_same_edge_count),
        "nonzero_role_transition_count": int(np.count_nonzero(transitions)),
        "semantic_transition_ratio": float(semantic_transition_ratio),
        "path_feature_dimension": int(expected_dimension),
        "max_hops": int(max_hops),
    }
    return path_features.astype(np.float64, copy=False), audit


def path_matrix(rows: list[dict], cache: dict, max_hops: int) -> tuple[np.ndarray, list[dict]]:
    vectors = []
    audits = []
    for row in rows:
        vector, audit = entry_path_consistency_vector(
            cache[row["sha256"]]["payload"], max_hops=max_hops
        )
        audit["sha256"] = row["sha256"]
        vectors.append(vector)
        audits.append(audit)
    if not vectors:
        return np.zeros((0, 0), dtype=np.float64), audits
    return np.asarray(vectors, dtype=np.float64), audits


def _write_csv(path: Path, fields: list[str], rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", required=True, type=Path)
    parser.add_argument("--features", required=True, type=Path)
    parser.add_argument("--schema", required=True, type=Path)
    parser.add_argument("--splits", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--iterations", type=int, default=1500)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--l2", type=float, default=0.03)
    parser.add_argument("--max-hops", type=int, default=DEFAULT_MAX_HOPS)
    args = parser.parse_args()
    if args.max_hops < 1:
        raise SystemExit("--max-hops must be positive")

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
    attach_audit_metadata(cache, feature_by_sha)
    selected = {
        row["sha256"]
        for partitions in split_rows.values()
        for partition in ("train", "test")
        for row in partitions[partition]
    }
    missing_cache = selected - set(cache)
    missing_features = selected - set(feature_by_sha)
    if missing_cache or missing_features:
        raise SystemExit(
            "missing selected rows: "
            f"cache={len(missing_cache)}, features={len(missing_features)}"
        )
    for row in [
        item
        for partitions in split_rows.values()
        for partition in ("train", "test")
        for item in partitions[partition]
    ]:
        row["metadata_label"] = feature_by_sha[row["sha256"]].get("metadata_label", "")
        row["metadata_year"] = feature_by_sha[row["sha256"]].get("metadata_year", "")
    observed_years = {
        str(feature_by_sha[sha].get("metadata_year", "")) for sha in selected
    }
    if observed_years != {"2017"}:
        raise SystemExit(
            f"this development evaluator accepts only 2017 rows; observed={sorted(observed_years)}"
        )

    definition = PilotReliabilityDefinition(
        alpha_prior=1.0, kappa_support=20.0, kappa_context=20.0,
        r_min=0.10, r_max=0.95,
    )
    entry_args = entry_model_args(definition)
    variants = (
        "compact_only",
        "compact_plus_entry",
        "compact_plus_path_consistency",
        "compact_plus_entry_plus_path_consistency",
    )
    split_results = {variant: [] for variant in variants}
    prediction_rows = []
    audit_rows = []

    for split_name in sorted(split_rows):
        train_rows = split_rows[split_name]["train"]
        validation_rows = split_rows[split_name]["test"]
        train_shas = {row["sha256"] for row in train_rows}
        evidence = fit_training_fold_components(
            cache, train_shas, definition,
            active_components=ACTIVE_RELIABILITY_COMPONENTS,
        )
        entry_train, y_train = graph_matrix(
            train_rows, cache, evidence, entry_args, True
        )
        entry_validation, y_validation = graph_matrix(
            validation_rows, cache, evidence, entry_args, True
        )
        compact_train, compact_validation, compact_count = compact_matrix(
            train_rows, validation_rows, feature_by_sha, schema, feature_mode="base"
        )
        path_train, train_audit = path_matrix(train_rows, cache, args.max_hops)
        path_validation, validation_audit = path_matrix(
            validation_rows, cache, args.max_hops
        )
        entry_train, entry_validation = standardize_pair(entry_train, entry_validation)
        path_train, path_validation = standardize_pair(path_train, path_validation)
        entry_train *= ENTRY_GRAPH_WEIGHT
        entry_validation *= ENTRY_GRAPH_WEIGHT
        matrices = {
            "compact_only": (compact_train, compact_validation),
            "compact_plus_entry": (
                np.concatenate([compact_train, entry_train], axis=1),
                np.concatenate([compact_validation, entry_validation], axis=1),
            ),
            "compact_plus_path_consistency": (
                np.concatenate([compact_train, path_train], axis=1),
                np.concatenate([compact_validation, path_validation], axis=1),
            ),
            "compact_plus_entry_plus_path_consistency": (
                np.concatenate([compact_train, entry_train, path_train], axis=1),
                np.concatenate([compact_validation, entry_validation, path_validation], axis=1),
            ),
        }
        for audit in train_audit:
            audit.update({"split_name": split_name, "partition": "train"})
            audit_rows.append(audit)
        for audit in validation_audit:
            audit.update({"split_name": split_name, "partition": "validation"})
            audit_rows.append(audit)

        for variant, (x_train, x_validation) in matrices.items():
            metrics_by_seed = []
            for seed in args.seeds:
                probability = predict_logistic(
                    x_train, y_train, x_validation, seed,
                    args.iterations, args.learning_rate, args.l2,
                )
                result = metrics(y_validation, probability, threshold=THRESHOLD)
                metrics_by_seed.append(result)
                for row, label, value in zip(validation_rows, y_validation, probability):
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
            summary = {
                "variant": variant,
                "split_name": split_name,
                "train_count": len(train_rows),
                "validation_count": len(validation_rows),
                "compact_feature_count": compact_count,
                "path_feature_count": int(path_train.shape[1]),
                "entry_graph_weight": ENTRY_GRAPH_WEIGHT,
                "threshold": THRESHOLD,
                "accuracy_mean": float(np.mean([m["accuracy"] for m in metrics_by_seed])),
                "balanced_accuracy_mean": float(np.mean([m["balanced_accuracy"] for m in metrics_by_seed])),
                "precision_mean": float(np.mean([m["precision"] for m in metrics_by_seed])),
                "recall_mean": float(np.mean([m["recall"] for m in metrics_by_seed])),
                "f1_mean": float(np.mean([m["f1"] for m in metrics_by_seed])),
                "auc_mean": float(np.mean([m["auc"] for m in metrics_by_seed])),
                "test_rows_used_for_training_or_selection": False,
            }
            split_results[variant].append(summary)
            print(
                f"{variant} {split_name} "
                f"BA={summary['balanced_accuracy_mean']:.4f} "
                f"AUC={summary['auc_mean']:.4f}",
                flush=True,
            )

    summary_rows = []
    for variant in variants:
        rows = split_results[variant]
        bas = np.asarray([row["balanced_accuracy_mean"] for row in rows])
        summary_rows.append({
            "rank": 0,
            "variant": variant,
            "mean_accuracy": float(np.mean([row["accuracy_mean"] for row in rows])),
            "mean_balanced_accuracy": float(bas.mean()),
            "minimum_split_balanced_accuracy": float(bas.min()),
            "std_balanced_accuracy": float(bas.std()),
            "stability_score": float(bas.mean() - bas.std()),
            "mean_auc": float(np.mean([row["auc_mean"] for row in rows])),
        })
    summary_rows.sort(
        key=lambda row: (-row["mean_balanced_accuracy"], -row["mean_auc"], row["variant"])
    )
    for rank, row in enumerate(summary_rows, 1):
        row["rank"] = rank

    _write_csv(
        output / "entry_path_consistency_2017_summary.csv",
        list(summary_rows[0]), summary_rows,
    )
    _write_csv(
        output / "entry_path_consistency_2017_per_split.csv",
        list(split_results[variants[0]][0]),
        [row for variant in variants for row in split_results[variant]],
    )
    _write_csv(
        output / "entry_path_consistency_2017_predictions.csv",
        list(prediction_rows[0]), prediction_rows,
    )
    _write_csv(
        output / "entry_path_consistency_2017_path_audit.csv",
        list(audit_rows[0]), audit_rows,
    )
    audit = {
        "schema_version": "1.0.0",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "phase": "P44_ENTRY_REACHABLE_LAYERED_SEMANTIC_PATH_CONSISTENCY_2017_ROLLING",
        "formal_training_enabled": False,
        "strict_test_isolation": True,
        "cache": str(paths["cache"]),
        "cache_sha256": sha256_file(paths["cache"]),
        "features": str(paths["features"]),
        "features_sha256": sha256_file(paths["features"]),
        "splits": str(paths["splits"]),
        "splits_sha256": sha256_file(paths["splits"]),
        "threshold": THRESHOLD,
        "entry_graph_weight": ENTRY_GRAPH_WEIGHT,
        "max_hops": args.max_hops,
        "role_feature_count": ROLE_COUNT,
        "path_feature_count": int(
            (args.max_hops + 1) * ROLE_COUNT
            + ROLE_COUNT * ROLE_COUNT
            + args.max_hops * (3 + 4 + len(EDGE_TYPE_NAMES) + 3)
            + 12
        ),
        "seeds": list(args.seeds),
        "iterations": args.iterations,
        "learning_rate": args.learning_rate,
        "l2": args.l2,
        "variants": list(variants),
        "summary_csv": str(output / "entry_path_consistency_2017_summary.csv"),
        "per_split_csv": str(output / "entry_path_consistency_2017_per_split.csv"),
        "predictions_csv": str(output / "entry_path_consistency_2017_predictions.csv"),
        "path_audit_csv": str(output / "entry_path_consistency_2017_path_audit.csv"),
        "blocking_reasons": [
            "formal_first_seen_utc_not_joined",
            "candidate_path_consistency_module_requires_nested_2017_validation",
        ],
        "notes": [
            "Path features use only bounded hashed cache IDs, semantic role IDs, hop layers, edge types, and compact counts.",
            "No method names, raw paths, labels, or test values are used to construct path features.",
            "Standardization is fit separately on each rolling training partition.",
            "Classification threshold is fixed at 0.5.",
            "This is a candidate module evaluation, not a formal confirmatory result.",
        ],
    }
    (output / "entry_path_consistency_2017_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({"summary": summary_rows, "audit": audit}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
