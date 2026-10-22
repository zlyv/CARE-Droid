#!/usr/bin/env python3
"""Evaluate entry and endpoint graph views on 2017 development folds only.

This script never opens an APK.  It consumes the existing bounded entry cache,
the P25 endpoint-centred cache, and the already extracted compact feature CSV.
The classification threshold is unconditionally fixed at 0.5.  Candidate
comparison is allowed only on chronological 2017 development folds; the
script aborts if a selected row belongs to another year.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from run_graph_pool_diagnostic import (  # noqa: E402
    compact_matrix,
    graph_matrix,
    read_csv_rows,
    read_split_rows,
    standardize_pair,
)
from run_pilot_compact_eval import fit_logistic, metrics  # noqa: E402
from tasa_droid.p3.a3_mvp import compute_reliability, stable_id  # noqa: E402
from tasa_droid.p3.reliability import (  # noqa: E402
    PilotReliabilityDefinition,
    fit_training_fold_components,
    read_cache,
)


THRESHOLD = 0.5
ROLE_VOCABULARY_SIZE = 31
ACTIVE_RELIABILITY_COMPONENTS = ("D", "one_minus_F")
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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def binary_label(row: dict) -> int:
    value = str(row.get("metadata_label", row.get("label", ""))).casefold()
    if value in {"malicious", "malware", "1", "true"}:
        return 1
    if value in {"benign", "0", "false"}:
        return 0
    raise ValueError(f"unknown label: {value!r}")


def attach_audit_metadata(cache: dict[str, dict], feature_by_sha: dict[str, dict]) -> None:
    """Attach labels/years in memory for train-fold fitting; never alter cache files."""

    for sha256, cache_row in cache.items():
        feature_row = feature_by_sha.get(sha256)
        if feature_row is None:
            continue
        cache_row["metadata_label"] = feature_row.get("metadata_label", "")
        cache_row["metadata_year"] = feature_row.get("metadata_year", "")


def _weighted_mean(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    if len(values) == 0:
        return np.zeros(values.shape[1], dtype=np.float64)
    return np.average(values, axis=0, weights=weights)


def _weighted_std(
    values: np.ndarray, weights: np.ndarray, mean: np.ndarray
) -> np.ndarray:
    if len(values) == 0:
        return np.zeros(values.shape[1], dtype=np.float64)
    variance = np.average((values - mean) ** 2, axis=0, weights=weights)
    return np.sqrt(np.maximum(variance, 0.0))


def endpoint_summary_vector(
    payload: dict,
    evidence: dict,
    role_vocabulary_size: int = ROLE_VOCABULARY_SIZE,
) -> np.ndarray:
    """Pool one P25 endpoint payload into provenance-aware semantic features."""

    node_ids = list(payload.get("node_ids") or [])
    base = np.asarray(payload.get("node_base_features") or [], dtype=np.float64)
    if not node_ids:
        base = np.zeros((0, 15), dtype=np.float64)
    if base.ndim != 2 or base.shape != (len(node_ids), 15):
        raise ValueError("endpoint node_base_features must have shape [N, 15]")
    role_ids = list(payload.get("node_role_ids") or [])
    if len(role_ids) != len(node_ids):
        raise ValueError("endpoint node role rows do not align with node IDs")
    role_matrix = np.zeros((len(node_ids), role_vocabulary_size), dtype=np.float64)
    reliability = np.zeros(len(node_ids), dtype=np.float64)
    weights = {name: 1.0 for name in ACTIVE_RELIABILITY_COMPONENTS}
    for index, (node_id, roles) in enumerate(zip(node_ids, role_ids)):
        for role_id in roles:
            role_id = int(role_id)
            if 0 <= role_id < role_vocabulary_size:
                role_matrix[index, role_id] = 1.0
        components = evidence["components_by_node_id"].get(
            str(node_id), evidence["global_training_prior"]
        )
        reliability[index] = compute_reliability(
            components,
            weights,
            0.10,
            0.95,
            active_components=ACTIVE_RELIABILITY_COMPONENTS,
        )
    pool_weights = np.maximum(reliability, 0.10)

    if len(base):
        base_mean = _weighted_mean(base, pool_weights)
        base_max = base.max(axis=0)
        base_std = _weighted_std(base, pool_weights, base_mean)
    else:
        base_mean = np.zeros(15, dtype=np.float64)
        base_max = np.zeros(15, dtype=np.float64)
        base_std = np.zeros(15, dtype=np.float64)

    edge_index = payload.get("edge_index") or [[], []]
    source = np.asarray(edge_index[0], dtype=np.int64)
    target = np.asarray(edge_index[1], dtype=np.int64)
    if source.shape != target.shape:
        raise ValueError("endpoint edge index source/target mismatch")
    edge_features = np.asarray(payload.get("edge_features") or [], dtype=np.float64)
    if len(source) == 0:
        edge_features = np.zeros((0, 7), dtype=np.float64)
    if edge_features.ndim != 2 or edge_features.shape != (len(source), 7):
        raise ValueError("endpoint edge_features must have shape [E, 7]")
    edge_type_ids = np.asarray(payload.get("edge_type_ids") or [], dtype=np.uint64)
    if len(edge_type_ids) != len(source):
        raise ValueError("endpoint edge type IDs do not align with edges")
    if len(source) and (
        source.min() < 0 or target.min() < 0
        or source.max() >= len(node_ids) or target.max() >= len(node_ids)
    ):
        raise ValueError("endpoint edge index is out of bounds")

    edge_mean = (
        edge_features.mean(axis=0)
        if len(edge_features) else np.zeros(7, dtype=np.float64)
    )
    global_vector = np.concatenate([
        base_mean,
        base_max,
        base_std,
        edge_mean,
        np.asarray([
            np.log1p(len(node_ids)),
            np.log1p(len(source)),
            float(reliability.mean()) if len(reliability) else 0.0,
        ]),
    ])

    # Per-role values deliberately emphasize endpoint state and provenance:
    # support, endpoint support, reachable/unreachable/unknown, three code
    # origins, and closeness to a sensitive endpoint.
    role_parts: list[float] = []
    for role_index in range(role_vocabulary_size):
        mask = role_matrix[:, role_index] > 0.0
        if not np.any(mask):
            role_parts.extend([0.0] * 9)
            continue
        local_weights = pool_weights[mask]
        local = base[mask]
        role_parts.extend([
            float(np.log1p(local_weights.sum())),
            float(np.log1p(np.sum(local_weights * local[:, 4]))),
            float(np.average(local[:, 8], weights=local_weights)),
            float(np.average(local[:, 9], weights=local_weights)),
            float(np.average(local[:, 10], weights=local_weights)),
            float(np.average(local[:, 5], weights=local_weights)),
            float(np.average(local[:, 6], weights=local_weights)),
            float(np.average(local[:, 7], weights=local_weights)),
            float(np.average(local[:, 14], weights=local_weights)),
        ])

    outgoing = np.zeros(role_vocabulary_size, dtype=np.float64)
    incoming = np.zeros(role_vocabulary_size, dtype=np.float64)
    flagged_outgoing = np.zeros(role_vocabulary_size, dtype=np.float64)
    flagged_incoming = np.zeros(role_vocabulary_size, dtype=np.float64)
    if len(source):
        edge_weights = np.sqrt(pool_weights[source] * pool_weights[target])
        source_roles = role_matrix[source]
        target_roles = role_matrix[target]
        outgoing = (source_roles * edge_weights[:, None]).sum(axis=0)
        incoming = (target_roles * edge_weights[:, None]).sum(axis=0)
        flagged = (edge_features[:, :3].max(axis=1) > 0.0).astype(np.float64)
        flagged_outgoing = (
            source_roles * (flagged * edge_weights)[:, None]
        ).sum(axis=0)
        flagged_incoming = (
            target_roles * (flagged * edge_weights)[:, None]
        ).sum(axis=0)
    role_edge_vector = np.stack([
        np.log1p(outgoing),
        np.log1p(incoming),
        flagged_outgoing / np.maximum(outgoing, 1.0),
        flagged_incoming / np.maximum(incoming, 1.0),
    ], axis=1).reshape(-1)

    edge_type_parts: list[float] = []
    total_edges = max(float(len(source)), 1.0)
    for edge_type_id in EDGE_TYPE_IDS:
        mask = edge_type_ids == edge_type_id
        count = int(mask.sum())
        edge_type_parts.extend([
            float(np.log1p(count)),
            float(count / total_edges),
            float(edge_features[mask, :3].mean()) if count else 0.0,
        ])
    return np.concatenate([
        global_vector,
        np.asarray(role_parts, dtype=np.float64),
        role_edge_vector,
        np.asarray(edge_type_parts, dtype=np.float64),
    ])


def endpoint_matrix(rows: list[dict], cache: dict, evidence: dict):
    values = []
    labels = []
    for row in rows:
        cache_row = cache[row["sha256"]]
        values.append(endpoint_summary_vector(cache_row["payload"], evidence))
        labels.append(binary_label(cache_row))
    return np.asarray(values, dtype=np.float64), np.asarray(labels, dtype=np.int64)


def entry_model_args(definition: PilotReliabilityDefinition):
    args = type("EntryModelArgs", (), {})()
    args.r_min = definition.r_min
    args.r_max = definition.r_max
    args.role_vocabulary_size = ROLE_VOCABULARY_SIZE
    args.reliability_components = ACTIVE_RELIABILITY_COMPONENTS
    args.graph_edge_mode = "all"
    args.graph_summary_mode = "full"
    args.graph_pool_mode = "role_stratified"
    args.graph_prevalence_mode = "none"
    args.graph_idf_floor = 0.05
    args.graph_discriminative_mode = "none"
    args.graph_contrast_alpha = 2.0
    args.graph_contrast_kappa = 10.0
    args.graph_contrast_clip = 4.0
    return args


def write_predictions(path: Path, rows: list[dict]) -> None:
    fields = [
        "variant", "split_name", "sha256", "seed", "label",
        "probability", "threshold", "predicted_label",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--entry-cache", required=True, type=Path)
    parser.add_argument("--endpoint-cache", required=True, type=Path)
    parser.add_argument("--features", required=True, type=Path)
    parser.add_argument("--schema", required=True, type=Path)
    parser.add_argument("--splits", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--endpoint-weights", nargs="+", type=float, default=[0.25, 0.5, 1.0])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--iterations", type=int, default=1500)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--l2", type=float, default=0.03)
    args = parser.parse_args()

    if any(weight <= 0.0 for weight in args.endpoint_weights):
        raise SystemExit("endpoint weights must be positive")
    if not args.seeds:
        raise SystemExit("at least one seed is required")
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

    selected_shas = {
        row["sha256"]
        for partitions in split_rows.values()
        for partition in ("train", "test")
        for row in partitions[partition]
    }
    missing = {
        "entry_cache": sorted(selected_shas - set(entry_cache)),
        "endpoint_cache": sorted(selected_shas - set(endpoint_cache)),
        "features": sorted(selected_shas - set(feature_by_sha)),
    }
    if any(missing.values()):
        counts = {name: len(values) for name, values in missing.items()}
        raise SystemExit(f"missing selected rows: {counts}")
    selected_years = {
        str(feature_by_sha[sha].get("metadata_year", "")) for sha in selected_shas
    }
    if selected_years != {"2017"}:
        raise SystemExit(
            "this evaluator is restricted to 2017 development rows; "
            f"observed years={sorted(selected_years)}"
        )

    definition = PilotReliabilityDefinition(
        alpha_prior=1.0,
        kappa_support=20.0,
        kappa_context=20.0,
        r_min=0.10,
        r_max=0.95,
    )
    entry_args = entry_model_args(definition)
    variants = [
        {"name": "entry_compact_control", "entry": True, "endpoint": False, "endpoint_weight": 0.0},
        {"name": "endpoint_compact", "entry": False, "endpoint": True, "endpoint_weight": 1.0},
    ] + [
        {
            "name": f"dual_concat_endpoint_w{weight:g}",
            "entry": True,
            "endpoint": True,
            "endpoint_weight": float(weight),
        }
        for weight in args.endpoint_weights
    ]

    per_variant_split: dict[str, list[dict]] = {
        variant["name"]: [] for variant in variants
    }
    prediction_rows: list[dict] = []
    feature_dimensions: dict[str, Counter] = {
        variant["name"]: Counter() for variant in variants
    }

    for split_name in sorted(split_rows):
        train_rows = split_rows[split_name]["train"]
        validation_rows = split_rows[split_name]["test"]
        if not train_rows or not validation_rows:
            raise SystemExit(f"empty 2017 development partition: {split_name}")
        train_shas = {row["sha256"] for row in train_rows}
        entry_evidence = fit_training_fold_components(
            entry_cache,
            train_shas,
            definition,
            active_components=ACTIVE_RELIABILITY_COMPONENTS,
        )
        endpoint_evidence = fit_training_fold_components(
            endpoint_cache,
            train_shas,
            definition,
            active_components=ACTIVE_RELIABILITY_COMPONENTS,
        )
        entry_train, y_train = graph_matrix(
            train_rows, entry_cache, entry_evidence, entry_args, True
        )
        entry_validation, y_validation = graph_matrix(
            validation_rows, entry_cache, entry_evidence, entry_args, True
        )
        endpoint_train, endpoint_y_train = endpoint_matrix(
            train_rows, endpoint_cache, endpoint_evidence
        )
        endpoint_validation, endpoint_y_validation = endpoint_matrix(
            validation_rows, endpoint_cache, endpoint_evidence
        )
        if not (
            np.array_equal(y_train, endpoint_y_train)
            and np.array_equal(y_validation, endpoint_y_validation)
        ):
            raise SystemExit(f"cache label mismatch in {split_name}")
        compact_train, compact_validation, compact_count = compact_matrix(
            train_rows,
            validation_rows,
            feature_by_sha,
            schema,
            feature_mode="base",
        )
        entry_train, entry_validation = standardize_pair(
            entry_train, entry_validation
        )
        endpoint_train, endpoint_validation = standardize_pair(
            endpoint_train, endpoint_validation
        )

        for variant in variants:
            train_parts = []
            validation_parts = []
            if variant["entry"]:
                train_parts.append(0.5 * entry_train)
                validation_parts.append(0.5 * entry_validation)
            if variant["endpoint"]:
                weight = variant["endpoint_weight"]
                train_parts.append(weight * endpoint_train)
                validation_parts.append(weight * endpoint_validation)
            train_parts.append(compact_train)
            validation_parts.append(compact_validation)
            x_train = np.concatenate(train_parts, axis=1)
            x_validation = np.concatenate(validation_parts, axis=1)
            feature_dimensions[variant["name"]][int(x_train.shape[1])] += 1

            seed_metrics = []
            for seed in args.seeds:
                weights, bias = fit_logistic(
                    x_train,
                    y_train,
                    seed,
                    iterations=args.iterations,
                    learning_rate=args.learning_rate,
                    l2=args.l2,
                )
                probability = 1.0 / (
                    1.0 + np.exp(
                        -np.clip(x_validation @ weights + bias, -40.0, 40.0)
                    )
                )
                result = metrics(y_validation, probability, threshold=THRESHOLD)
                seed_metrics.append(result)
                for row, label, score in zip(
                    validation_rows, y_validation, probability
                ):
                    prediction_rows.append({
                        "variant": variant["name"],
                        "split_name": split_name,
                        "sha256": row["sha256"],
                        "seed": seed,
                        "label": int(label),
                        "probability": float(score),
                        "threshold": THRESHOLD,
                        "predicted_label": int(score >= THRESHOLD),
                    })
            summary = {
                "split_name": split_name,
                "train_count": len(train_rows),
                "validation_count": len(validation_rows),
                "feature_count": int(x_train.shape[1]),
                "compact_feature_count": int(compact_count),
                "threshold": THRESHOLD,
            }
            for metric_name in (
                "accuracy", "balanced_accuracy", "precision", "recall", "f1", "auc"
            ):
                summary[f"{metric_name}_mean"] = float(np.mean([
                    item[metric_name] for item in seed_metrics
                ]))
            per_variant_split[variant["name"]].append(summary)
        print(f"completed {split_name}", flush=True)

    variant_summaries = []
    variant_by_name = {variant["name"]: variant for variant in variants}
    for name, split_summaries in per_variant_split.items():
        ba_values = np.asarray(
            [row["balanced_accuracy_mean"] for row in split_summaries]
        )
        auc_values = np.asarray([row["auc_mean"] for row in split_summaries])
        accuracy_values = np.asarray([row["accuracy_mean"] for row in split_summaries])
        variant_summaries.append({
            "variant": name,
            "entry_view": variant_by_name[name]["entry"],
            "endpoint_view": variant_by_name[name]["endpoint"],
            "entry_weight": 0.5 if variant_by_name[name]["entry"] else 0.0,
            "endpoint_weight": variant_by_name[name]["endpoint_weight"],
            "feature_counts": sorted(feature_dimensions[name]),
            "mean_accuracy": float(accuracy_values.mean()),
            "mean_balanced_accuracy": float(ba_values.mean()),
            "minimum_fold_balanced_accuracy": float(ba_values.min()),
            "std_balanced_accuracy": float(ba_values.std()),
            "stability_score": float(ba_values.mean() - ba_values.std()),
            "mean_auc": float(auc_values.mean()),
            "split_summaries": split_summaries,
        })
    ranked = sorted(
        variant_summaries,
        key=lambda row: (
            -row["stability_score"],
            -row["mean_balanced_accuracy"],
            row["variant"],
        ),
    )
    for rank, row in enumerate(ranked, 1):
        row["rank"] = rank

    prediction_path = output / "dual_view_2017_predictions.csv"
    write_predictions(prediction_path, prediction_rows)
    summary_path = output / "dual_view_2017_summary.csv"
    summary_fields = [
        "rank", "variant", "entry_view", "endpoint_view", "entry_weight",
        "endpoint_weight", "mean_accuracy", "mean_balanced_accuracy",
        "minimum_fold_balanced_accuracy", "std_balanced_accuracy",
        "stability_score", "mean_auc",
    ]
    with summary_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=summary_fields)
        writer.writeheader()
        for row in ranked:
            writer.writerow({name: row.get(name, "") for name in summary_fields})

    audit = {
        "schema_version": "1.0.0",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "phase": "P26_DUAL_GRAPH_VIEW_2017_ROLLING_DEVELOPMENT_EVALUATION",
        "formal_training_enabled": False,
        "protocol_status": "DEVELOPMENT_ONLY_NOT_OUTER_TESTED",
        "strict_future_year_isolation": True,
        "observed_years": sorted(selected_years),
        "fixed_threshold": THRESHOLD,
        "threshold_tuning_performed": False,
        "entry_cache": str(paths["entry_cache"]),
        "entry_cache_sha256": sha256_file(paths["entry_cache"]),
        "endpoint_cache": str(paths["endpoint_cache"]),
        "endpoint_cache_sha256": sha256_file(paths["endpoint_cache"]),
        "features": str(paths["features"]),
        "features_sha256": sha256_file(paths["features"]),
        "schema": str(paths["schema"]),
        "schema_sha256": sha256_file(paths["schema"]),
        "split_manifest": str(paths["splits"]),
        "split_manifest_sha256": sha256_file(paths["splits"]),
        "classifier": {
            "type": "custom_l2_logistic",
            "seeds": args.seeds,
            "iterations": args.iterations,
            "learning_rate": args.learning_rate,
            "l2": args.l2,
        },
        "entry_view": {
            "pool_mode": "role_stratified",
            "weight": 0.5,
            "reliability_components": list(ACTIVE_RELIABILITY_COMPONENTS),
        },
        "endpoint_view": {
            "pool_mode": "role_provenance_reachability_edge_stratified",
            "weights_compared": args.endpoint_weights,
            "reliability_components": list(ACTIVE_RELIABILITY_COMPONENTS),
        },
        "candidate_selection_metric": "mean_balanced_accuracy_minus_fold_std",
        "ranked_variants": ranked,
        "selected_development_candidate": ranked[0]["variant"],
        "entry_control_reproduction_target_mean_ba": 0.9195,
        "outputs": {
            "summary": str(summary_path),
            "predictions": str(prediction_path),
        },
        "blocking_reasons": [
            "dual_view_candidate_not_locked",
            "uncertainty_gate_not_implemented",
            "formal_dataset_not_evaluated",
            "legacy_future_years_must_not_drive_this_candidate_selection",
        ],
        "notes": [
            "Only chronological 2017 development rows are accepted by this script.",
            "Each fold fits reliability, filtering, and normalization on its training partition only.",
            "The 2017 validation partitions are used for development candidate comparison.",
            "No 2018--2021 row is read or evaluated.",
            "The classification threshold is fixed at 0.5 for every variant and fold.",
        ],
    }
    audit_path = output / "dual_view_2017_audit.json"
    audit_path.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print("rank variant mean_BA min_BA std_BA stability AUC", flush=True)
    for row in ranked:
        print(
            row["rank"],
            row["variant"],
            f'{row["mean_balanced_accuracy"]:.4f}',
            f'{row["minimum_fold_balanced_accuracy"]:.4f}',
            f'{row["std_balanced_accuracy"]:.4f}',
            f'{row["stability_score"]:.4f}',
            f'{row["mean_auc"]:.4f}',
            flush=True,
        )


if __name__ == "__main__":
    main()
