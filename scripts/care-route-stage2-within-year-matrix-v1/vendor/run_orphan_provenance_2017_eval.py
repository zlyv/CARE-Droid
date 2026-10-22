#!/usr/bin/env python3
"""Evaluate orphan-sensitive provenance calibration on chronological 2017 folds.

The module summarizes all selected sensitive endpoints by semantic role,
entry-reachability state, and code provenance (package-aligned, bundled, or
external).  It deliberately does not retain method names or local paths.

Compared variants:
  compact_only
  compact_plus_entry
  compact_plus_orphan_provenance
  compact_plus_entry_plus_orphan_provenance

The endpoint cache is read only; no APK is reopened.  Threshold is fixed at
0.5 and all model fitting/standardization is performed on each training
partition.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
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
PROVENANCE_COUNT = 3
REACHABILITY_COUNT = 3
ROLE_FEATURE_COUNT = 10
GLOBAL_FEATURE_COUNT = 16
ORPHAN_PROVENANCE_DIMENSION = (
    ROLE_VOCABULARY_SIZE * PROVENANCE_COUNT * REACHABILITY_COUNT
    + ROLE_VOCABULARY_SIZE * ROLE_FEATURE_COUNT
    + GLOBAL_FEATURE_COUNT
)


def _role_ids(role_row) -> list[int]:
    values = sorted({
        int(value) for value in (role_row or [])
        if 0 <= int(value) < ROLE_VOCABULARY_SIZE
    })
    return values


def _entropy(values: np.ndarray) -> float:
    total = float(values.sum())
    if total <= 0.0:
        return 0.0
    probabilities = values[values > 0.0] / total
    return float(-(probabilities * np.log(probabilities)).sum())


def orphan_provenance_vector(payload: dict) -> tuple[np.ndarray, dict]:
    """Return fixed semantic/provenance/reachability summary features."""

    node_ids, base, roles, _source, _target = _payload_arrays(payload)
    endpoint_mask = base[:, 4] > 0.5
    endpoints = np.flatnonzero(endpoint_mask)
    provenance = (
        np.argmax(base[:, 5:8], axis=1)
        if len(base) else np.zeros(0, dtype=np.int64)
    )
    reachability = (
        np.argmax(base[:, 8:11], axis=1)
        if len(base) else np.zeros(0, dtype=np.int64)
    )

    joint = np.zeros(
        (ROLE_VOCABULARY_SIZE, PROVENANCE_COUNT, REACHABILITY_COUNT),
        dtype=np.float64,
    )
    role_support = np.zeros(ROLE_VOCABULARY_SIZE, dtype=np.float64)
    role_provenance = np.zeros(
        (ROLE_VOCABULARY_SIZE, PROVENANCE_COUNT), dtype=np.float64
    )
    role_reachability = np.zeros(
        (ROLE_VOCABULARY_SIZE, REACHABILITY_COUNT), dtype=np.float64
    )
    provenance_counts = np.zeros(PROVENANCE_COUNT, dtype=np.float64)
    reachability_counts = np.zeros(REACHABILITY_COUNT, dtype=np.float64)

    for index in endpoints:
        index = int(index)
        p = int(np.clip(provenance[index], 0, PROVENANCE_COUNT - 1))
        r = int(np.clip(reachability[index], 0, REACHABILITY_COUNT - 1))
        provenance_counts[p] += 1.0
        reachability_counts[r] += 1.0
        for role in _role_ids(roles[index]):
            joint[role, p, r] += 1.0
            role_support[role] += 1.0
            role_provenance[role, p] += 1.0
            role_reachability[role, r] += 1.0

    role_parts = []
    for role in range(ROLE_VOCABULARY_SIZE):
        support = role_support[role]
        role_parts.extend([
            float(np.log1p(support)),
            float(role_reachability[role, 0] / max(support, 1.0)),
            float(role_reachability[role, 1] / max(support, 1.0)),
            float(role_reachability[role, 2] / max(support, 1.0)),
            float(role_provenance[role, 0] / max(support, 1.0)),
            float(role_provenance[role, 1] / max(support, 1.0)),
            float(role_provenance[role, 2] / max(support, 1.0)),
            float(role_reachability[role, 1] / max(
                role_reachability[role].sum(), 1.0
            )),
            float(role_provenance[role, 0] / max(
                role_provenance[role].sum(), 1.0
            )),
            float(_entropy(role_provenance[role])),
        ])

    endpoint_count = float(len(endpoints))
    global_parts = [
        float(np.log1p(endpoint_count)),
        float(reachability_counts[0] / max(endpoint_count, 1.0)),
        float(reachability_counts[1] / max(endpoint_count, 1.0)),
        float(reachability_counts[2] / max(endpoint_count, 1.0)),
        float(provenance_counts[0] / max(endpoint_count, 1.0)),
        float(provenance_counts[1] / max(endpoint_count, 1.0)),
        float(provenance_counts[2] / max(endpoint_count, 1.0)),
        float(_entropy(provenance_counts)),
        float(_entropy(reachability_counts)),
        float(np.count_nonzero(role_support)),
        float(_entropy(role_support)),
        float(np.log1p(base[:, 1].sum()) if len(base) else 0.0),
        float(np.log1p(base[:, 0].sum()) if len(base) else 0.0),
        float(np.log1p(len(node_ids))),
        float(np.log1p(len(_source))),
        float(reachability_counts[1] / max(reachability_counts[0] + reachability_counts[1], 1.0)),
    ]
    vector = np.concatenate([
        np.log1p(joint).reshape(-1),
        np.asarray(role_parts, dtype=np.float64),
        np.asarray(global_parts, dtype=np.float64),
    ])
    if vector.shape != (ORPHAN_PROVENANCE_DIMENSION,):
        raise ValueError(
            f"unexpected orphan feature dimension {vector.shape}; "
            f"expected {ORPHAN_PROVENANCE_DIMENSION}"
        )
    audit = {
        "sensitive_endpoint_count": int(endpoint_count),
        "entry_reachable_endpoint_count": int(reachability_counts[0]),
        "entry_unreachable_endpoint_count": int(reachability_counts[1]),
        "reachability_unknown_endpoint_count": int(reachability_counts[2]),
        "package_aligned_endpoint_count": int(provenance_counts[0]),
        "bundled_endpoint_count": int(provenance_counts[1]),
        "external_endpoint_count": int(provenance_counts[2]),
        "unreachable_endpoint_ratio": float(
            reachability_counts[1] / max(endpoint_count, 1.0)
        ),
        "package_aligned_ratio": float(
            provenance_counts[0] / max(endpoint_count, 1.0)
        ),
    }
    return vector, audit


def orphan_matrix(rows: list[dict], cache: dict):
    vectors = []
    audits = []
    for row in rows:
        vector, audit = orphan_provenance_vector(cache[row["sha256"]]["payload"])
        vectors.append(vector)
        audit["sha256"] = row["sha256"]
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
        row for partitions in split_rows.values()
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
        raise SystemExit(f"2017 development rows required; observed={sorted(years)}")
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
        "compact_plus_orphan_provenance",
        "compact_plus_entry_plus_orphan_provenance",
    )
    split_results = {variant: [] for variant in variants}
    prediction_rows = []
    audit_rows = []

    for split_name in sorted(split_rows):
        train_rows = split_rows[split_name]["train"]
        test_rows = split_rows[split_name]["test"]
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
        entry_test, y_test = graph_matrix(
            test_rows, entry_cache, evidence, entry_args, True
        )
        compact_train, compact_test, compact_count = compact_matrix(
            train_rows, test_rows, feature_by_sha, schema, feature_mode="base"
        )
        orphan_train, train_audit = orphan_matrix(train_rows, endpoint_cache)
        orphan_test, test_audit = orphan_matrix(test_rows, endpoint_cache)
        entry_train, entry_test = standardize_pair(entry_train, entry_test)
        orphan_train, orphan_test = standardize_pair(orphan_train, orphan_test)
        entry_train *= ENTRY_GRAPH_WEIGHT
        entry_test *= ENTRY_GRAPH_WEIGHT
        matrices = {
            "compact_only": (compact_train, compact_test),
            "compact_plus_entry": (
                np.concatenate([compact_train, entry_train], axis=1),
                np.concatenate([compact_test, entry_test], axis=1),
            ),
            "compact_plus_orphan_provenance": (
                np.concatenate([compact_train, orphan_train], axis=1),
                np.concatenate([compact_test, orphan_test], axis=1),
            ),
            "compact_plus_entry_plus_orphan_provenance": (
                np.concatenate([compact_train, entry_train, orphan_train], axis=1),
                np.concatenate([compact_test, entry_test, orphan_test], axis=1),
            ),
        }
        for partition, rows in (("train", train_audit), ("test", test_audit)):
            for audit in rows:
                audit["split_name"] = split_name
                audit["partition"] = partition
                audit_rows.append(audit)

        seed_metrics = {variant: [] for variant in variants}
        for seed in args.seeds:
            for variant, (x_train, x_test) in matrices.items():
                probability = predict_logistic(
                    x_train, y_train, x_test, seed,
                    args.iterations, args.learning_rate, args.l2,
                )
                seed_metrics[variant].append(
                    metrics(y_test, probability, threshold=THRESHOLD)
                )
                for row, label, value in zip(test_rows, y_test, probability):
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
                "validation_count": len(test_rows),
                "compact_feature_count": int(compact_count),
                "orphan_provenance_feature_count": ORPHAN_PROVENANCE_DIMENSION,
                "entry_graph_feature_count": int(entry_train.shape[1]),
                "threshold": THRESHOLD,
            }
            for name in (
                "accuracy", "balanced_accuracy", "precision", "recall", "f1", "auc"
            ):
                summary[f"{name}_mean"] = float(np.mean([
                    item[name] for item in seed_metrics[variant]
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

    summary_path = output / "orphan_provenance_2017_summary.csv"
    write_csv(
        summary_path,
        [
            "rank", "variant", "mean_accuracy", "mean_balanced_accuracy",
            "minimum_split_balanced_accuracy", "std_balanced_accuracy",
            "stability_score", "mean_auc",
        ],
        ranked,
    )
    per_split_path = output / "orphan_provenance_2017_per_split.csv"
    write_csv(
        per_split_path,
        [
            "variant", "split_name", "train_count", "validation_count",
            "compact_feature_count", "orphan_provenance_feature_count",
            "entry_graph_feature_count", "threshold", "accuracy_mean",
            "balanced_accuracy_mean", "precision_mean", "recall_mean",
            "f1_mean", "auc_mean",
        ],
        [row for variant in variants for row in split_results[variant]],
    )
    write_csv(
        output / "orphan_provenance_reachability_audit.csv",
        [
            "split_name", "partition", "sha256", "sensitive_endpoint_count",
            "entry_reachable_endpoint_count", "entry_unreachable_endpoint_count",
            "reachability_unknown_endpoint_count", "package_aligned_endpoint_count",
            "bundled_endpoint_count", "external_endpoint_count",
            "unreachable_endpoint_ratio", "package_aligned_ratio",
        ],
        audit_rows,
    )
    prediction_path = output / "orphan_provenance_2017_predictions.csv"
    write_csv(
        prediction_path,
        [
            "variant", "split_name", "sha256", "seed", "label", "probability",
            "threshold", "predicted_label",
        ],
        prediction_rows,
    )
    audit = {
        "schema_version": "1.0.0",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "phase": "P30_ORPHAN_SENSITIVE_PROVENANCE_CALIBRATION_2017_DEVELOPMENT",
        "formal_training_enabled": False,
        "protocol_status": "DEVELOPMENT_ONLY_NOT_OUTER_TESTED",
        "observed_years": sorted(years),
        "fixed_threshold": THRESHOLD,
        "threshold_tuning_performed": False,
        "feature_dimension": ORPHAN_PROVENANCE_DIMENSION,
        "role_vocabulary_size": ROLE_VOCABULARY_SIZE,
        "joint_axes": ["semantic_role", "provenance", "entry_reachability"],
        "provenance_categories": [
            "package_aligned", "bundled", "external"
        ],
        "reachability_categories": [
            "entry_reachable", "entry_unreachable", "unknown"
        ],
        "method_names_persisted": False,
        "label_independent_feature_construction": True,
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
            "reachability_audit": str(output / "orphan_provenance_reachability_audit.csv"),
            "predictions": str(prediction_path),
        },
        "blocking_reasons": [
            "candidate_not_locked",
            "formal_dataset_not_evaluated",
        ],
        "notes": [
            "The module summarizes all selected sensitive endpoints, including entry-unreachable endpoints.",
            "Provenance and reachability are represented as aggregate statistics, not method identities.",
            "This experiment reuses the existing endpoint cache and does not reopen APKs.",
        ],
    }
    audit_path = output / "orphan_provenance_2017_audit.json"
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
