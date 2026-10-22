#!/usr/bin/env python3
"""Run strict grouped nested validation for the compact pilot features.

The outer annual test partition is never used for model selection.  For each
outer split and seed, hyperparameters and the decision threshold are selected
from grouped inner OOF predictions made only on the outer training partition.
The fixed schema supplies candidate column names; finite/constant filtering,
normalization, fitting, and threshold selection are all training-fold local.
This remains a pilot compact-feature evaluator, not the final graph model.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

try:
    import numpy as np
except ModuleNotFoundError as exc:  # pragma: no cover - environment dependent
    raise SystemExit("NumPy is required; use /home/boyuanjie/envs/tif/bin/python") from exc

from run_pilot_compact_eval import (
    AUDIT_COLUMNS,
    fit_logistic,
    label_value,
    metrics,
    read_csv,
    sha256_file,
    standardize,
)


SCHEMA_VERSION = "1.0.0"
DEFAULT_L2 = (1e-4, 1e-3, 1e-2)
DEFAULT_LEARNING_RATES = (0.02, 0.05)
DEFAULT_THRESHOLDS = tuple(round(0.30 + 0.05 * index, 2) for index in range(9))


def select_features(rows, candidates):
    usable = []
    for name in candidates:
        try:
            values = np.asarray([float(row[name]) for row in rows], dtype=float)
        except (KeyError, TypeError, ValueError):
            continue
        if np.all(np.isfinite(values)) and np.unique(values).size > 1:
            usable.append(name)
    return usable


def matrices(train_rows, eval_rows, features):
    x_train = np.asarray(
        [[float(row[name]) for name in features] for row in train_rows], dtype=float
    )
    x_eval = np.asarray(
        [[float(row[name]) for name in features] for row in eval_rows], dtype=float
    )
    if not np.all(np.isfinite(x_train)) or not np.all(np.isfinite(x_eval)):
        raise ValueError("non-finite value in a selected feature")
    return standardize(x_train, x_eval)


def group_folds(rows, n_splits=5):
    """Greedily distribute whole blocking groups while balancing class counts."""
    by_group = defaultdict(list)
    for index, row in enumerate(rows):
        group = str(row.get("blocking_group", "")).strip()
        if not group:
            raise ValueError("inner-validation row has an empty blocking_group")
        by_group[group].append(index)

    groups = []
    for group, indices in by_group.items():
        labels = [label_value(rows[index]["metadata_label"]) for index in indices]
        groups.append((group, indices, sum(label == 1 for label in labels), len(indices)))
    groups.sort(key=lambda item: (-item[3], -item[2], item[0]))

    actual_splits = min(n_splits, len(groups))
    if actual_splits < 2:
        raise ValueError("at least two blocking groups are required for inner validation")

    folds = [[] for _ in range(actual_splits)]
    fold_total = [0] * actual_splits
    fold_positive = [0] * actual_splits
    total_positive = sum(item[2] for item in groups)
    target_positive = total_positive / max(1, actual_splits)
    target_total = len(rows) / max(1, actual_splits)

    # Seed every fold with a distinct group.  The previous greedy-only
    # allocator could repeatedly choose fold 0 for highly imbalanced groups.
    for fold, (_group, indices, positives, size) in enumerate(groups[:actual_splits]):
        folds[fold].extend(indices)
        fold_total[fold] += size
        fold_positive[fold] += positives

    for _group, indices, positives, size in groups[actual_splits:]:
        chosen = min(
            range(actual_splits),
            key=lambda fold: (
                abs((fold_positive[fold] + positives) - target_positive),
                abs((fold_total[fold] + size) - target_total),
                fold,
            ),
        )
        folds[chosen].extend(indices)
        fold_total[chosen] += size
        fold_positive[chosen] += positives

    if any(not fold for fold in folds):
        raise ValueError("grouped inner split produced an empty fold")
    return folds


def inner_select(train_rows, candidates, seed, l2_grid, learning_rate_grid, thresholds,
                 inner_folds=5, iterations=800):
    folds = group_folds(train_rows, inner_folds)
    all_indices = set(range(len(train_rows)))
    candidates_report = []
    for l2 in l2_grid:
        for learning_rate in learning_rate_grid:
            oof_scores = np.full(len(train_rows), np.nan, dtype=float)
            for fold_index, validation_indices in enumerate(folds):
                validation_set = set(validation_indices)
                fit_indices = sorted(all_indices - validation_set)
                fit_rows = [train_rows[index] for index in fit_indices]
                validation_rows = [train_rows[index] for index in validation_indices]
                features = select_features(fit_rows, candidates)
                if not features:
                    raise ValueError("inner fold has no usable features")
                x_fit, x_validation = matrices(fit_rows, validation_rows, features)
                y_fit = np.asarray([label_value(row["metadata_label"]) for row in fit_rows])
                weights, bias = fit_logistic(
                    x_fit, y_fit, seed + fold_index,
                    iterations=iterations, learning_rate=learning_rate, l2=l2,
                )
                oof_scores[validation_indices] = 1.0 / (
                    1.0 + np.exp(-np.clip(x_validation @ weights + bias, -40.0, 40.0))
                )

            y_train = np.asarray([label_value(row["metadata_label"]) for row in train_rows])
            if not np.all(np.isfinite(oof_scores)):
                raise ValueError("inner OOF predictions are incomplete")
            best = None
            for threshold in thresholds:
                metric = metrics(y_train, oof_scores, threshold=threshold)
                candidate = {
                    "l2": l2,
                    "learning_rate": learning_rate,
                    "threshold": threshold,
                    "accuracy": metric["accuracy"],
                    "balanced_accuracy": metric["balanced_accuracy"],
                }
                rank = (
                    candidate["accuracy"],
                    candidate["balanced_accuracy"],
                    -abs(threshold - 0.5),
                    -l2,
                    -learning_rate,
                )
                if best is None or rank > best[0]:
                    best = (rank, candidate)
            candidates_report.append(best[1])

    selected = max(
        candidates_report,
        key=lambda item: (
            item["accuracy"], item["balanced_accuracy"],
            -abs(item["threshold"] - 0.5), -item["l2"], -item["learning_rate"],
        ),
    )
    return selected, candidates_report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", required=True, type=Path)
    parser.add_argument("--splits", required=True, type=Path)
    parser.add_argument("--schema", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--inner-folds", type=int, default=5)
    parser.add_argument("--inner-iterations", type=int, default=800)
    parser.add_argument("--outer-iterations", type=int, default=1500)
    parser.add_argument("--reference-mean", type=float, default=0.9074)
    args = parser.parse_args()

    feature_path = args.features.expanduser().resolve(strict=True)
    split_path = args.splits.expanduser().resolve(strict=True)
    schema_path = args.schema.expanduser().resolve(strict=True)
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)

    feature_rows = read_csv(feature_path)
    split_rows = read_csv(split_path)
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    candidates = [
        name for name in schema.get("feature_fields", [])
        if name not in AUDIT_COLUMNS
    ]
    if not feature_rows or not split_rows or not candidates:
        raise SystemExit("features, split manifest, or schema candidates are empty")
    feature_by_sha = {str(row["sha256"]).strip().lower(): row for row in feature_rows}

    result_rows = []
    prediction_rows = []
    summaries = []
    for split_name in sorted({row["split_name"] for row in split_rows}):
        current = [row for row in split_rows if row["split_name"] == split_name]
        train_manifest = [row for row in current if row["partition"] == "train"]
        test_manifest = [row for row in current if row["partition"] == "test"]
        train_ids = [row["sha256"].strip().lower() for row in train_manifest]
        test_ids = [row["sha256"].strip().lower() for row in test_manifest]
        train_rows = [feature_by_sha[sha] | {"blocking_group": manifest["blocking_group"]}
                      for sha, manifest in zip(train_ids, train_manifest)]
        test_rows = [feature_by_sha[sha] for sha in test_ids]
        if not train_rows or not test_rows:
            raise SystemExit(f"empty outer partition in {split_name}")

        split_results = []
        for seed in args.seeds:
            selected, selection_candidates = inner_select(
                train_rows, candidates, seed,
                DEFAULT_L2, DEFAULT_LEARNING_RATES, DEFAULT_THRESHOLDS,
                inner_folds=args.inner_folds, iterations=args.inner_iterations,
            )
            outer_features = select_features(train_rows, candidates)
            x_train, x_test = matrices(train_rows, test_rows, outer_features)
            y_train = np.asarray([label_value(row["metadata_label"]) for row in train_rows])
            y_test = np.asarray([label_value(row["metadata_label"]) for row in test_rows])
            weights, bias = fit_logistic(
                x_train, y_train, seed,
                iterations=args.outer_iterations,
                learning_rate=selected["learning_rate"],
                l2=selected["l2"],
            )
            scores = 1.0 / (1.0 + np.exp(-np.clip(x_test @ weights + bias, -40.0, 40.0)))
            predicted_metric = metrics(y_test, scores, threshold=selected["threshold"])
            predicted_labels = (scores >= selected["threshold"]).astype(int)
            for manifest_row, true_label, score, predicted_label in zip(
                test_manifest, y_test, scores, predicted_labels
            ):
                prediction_rows.append(
                    {
                        "split_name": split_name,
                        "seed": seed,
                        "sha256": manifest_row["sha256"],
                        "year": manifest_row["year"],
                        "label": "malicious" if int(true_label) else "benign",
                        "y_true": int(true_label),
                        "score": float(score),
                        "threshold": selected["threshold"],
                        "y_pred": int(predicted_label),
                    }
                )
            result = {
                "split_name": split_name,
                "seed": seed,
                "model": "compact_logistic_nested_pilot",
                "train_count": len(train_rows),
                "test_count": len(test_rows),
                "feature_count": len(outer_features),
                "selected_l2": selected["l2"],
                "selected_learning_rate": selected["learning_rate"],
                "selected_threshold": selected["threshold"],
                "inner_accuracy": selected["accuracy"],
                "inner_balanced_accuracy": selected["balanced_accuracy"],
                **predicted_metric,
            }
            result_rows.append(result)
            split_results.append(result)

        summary = {
            "split_name": split_name,
            "train_count": len(train_rows),
            "test_count": len(test_rows),
            "accuracy_mean": float(np.mean([row["accuracy"] for row in split_results])),
            "accuracy_std": float(np.std([row["accuracy"] for row in split_results])),
            "balanced_accuracy_mean": float(np.mean([row["balanced_accuracy"] for row in split_results])),
            "f1_mean": float(np.mean([row["f1"] for row in split_results])),
            "auc_mean": float(np.mean([row["auc"] for row in split_results if row["auc"] is not None])),
            "feature_count_mean": float(np.mean([row["feature_count"] for row in split_results])),
            "thresholds": sorted({row["selected_threshold"] for row in split_results}),
            "l2_values": sorted({row["selected_l2"] for row in split_results}),
            "learning_rate_values": sorted({row["selected_learning_rate"] for row in split_results}),
        }
        summaries.append(summary)

    result_path = output / "nested_pilot_results.csv"
    with result_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(result_rows[0]))
        writer.writeheader()
        writer.writerows(result_rows)
    prediction_path = output / "nested_pilot_predictions.csv"
    with prediction_path.open("w", encoding="utf-8", newline="") as handle:
        prediction_fields = [
            "split_name", "seed", "sha256", "year", "label",
            "y_true", "score", "threshold", "y_pred",
        ]
        writer = csv.DictWriter(handle, fieldnames=prediction_fields)
        writer.writeheader()
        writer.writerows(prediction_rows)

    annual_accuracy = [row["accuracy_mean"] for row in summaries]
    audit = {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "phase": "P1_GROUPED_NESTED_PILOT_EVALUATION",
        "formal_training_enabled": False,
        "model": "compact_logistic_nested_pilot",
        "feature_source": "FIXED_SCHEMA_COLUMNS_WITHIN_INNER_AND_OUTER_TRAIN_FILTERING",
        "strict_test_feature_isolation": True,
        "features_csv": str(feature_path),
        "features_sha256": sha256_file(feature_path),
        "split_manifest_csv": str(split_path),
        "split_manifest_sha256": sha256_file(split_path),
        "schema_json": str(schema_path),
        "schema_sha256": sha256_file(schema_path),
        "seeds": list(args.seeds),
        "inner_folds": args.inner_folds,
        "inner_iterations": args.inner_iterations,
        "outer_iterations": args.outer_iterations,
        "l2_grid": list(DEFAULT_L2),
        "learning_rate_grid": list(DEFAULT_LEARNING_RATES),
        "threshold_grid": list(DEFAULT_THRESHOLDS),
        "reference_mean": args.reference_mean,
        "reference_metric": "accuracy_mean_assumed_from_baseline_table",
        "split_summaries": summaries,
        "four_year_mean_accuracy": float(np.mean(annual_accuracy)),
        "exceeds_reference_point_estimate": bool(np.mean(annual_accuracy) > args.reference_mean),
        "outputs": {
            "results_csv": str(result_path),
            "predictions_csv": str(prediction_path),
        },
        "blocking_reasons": [
            "audited_first_seen_utc_not_joined",
            "compact_summary_is_not_formal_graph_model",
            "tif_implementation_and_metric_definition_not_reproduced",
            "resolved_formal_training_config_not_generated",
        ],
        "notes": [
            "Outer test years are never used for hyperparameter or threshold selection.",
            "Inner folds keep blocking groups intact.",
            "All filtering and normalization are fit on the corresponding training fold only.",
            "This is still a compact-feature pilot, not the final TASA-Droid graph model.",
        ],
    }
    (output / "nested_pilot_evaluation_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
