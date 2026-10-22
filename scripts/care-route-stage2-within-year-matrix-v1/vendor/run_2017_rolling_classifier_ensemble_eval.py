#!/usr/bin/env python3
"""Compare classifiers and probability blends on grouped rolling 2017 folds.

All representation fitting, reliability evidence, feature filtering, scaling,
and model fitting are fold-training-only.  The decision threshold is fixed at
0.5.  No 2018--2021 row is required or read by this evaluator.
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
from types import SimpleNamespace

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss

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
from tasa_droid.p3.reliability import (  # noqa: E402
    PilotReliabilityDefinition,
    fit_training_fold_components,
    read_cache,
)


THRESHOLD = 0.5
GRAPH_WEIGHT = 0.5
RELIABILITY_COMPONENTS = ("D", "one_minus_F")
SEEDS = (0, 1, 2, 3, 4)
HGB_BLEND_WEIGHTS = (0.25, 0.50, 0.75)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def score(labels: np.ndarray, probability: np.ndarray) -> dict[str, float]:
    result = metrics(labels, probability, threshold=THRESHOLD)
    result["brier_score"] = float(brier_score_loss(labels, probability))
    result["log_loss"] = float(
        log_loss(labels, np.clip(probability, 1e-7, 1.0 - 1e-7))
    )
    result["predicted_positive_rate"] = float(
        np.mean(probability >= THRESHOLD)
    )
    return result


def mean_seed_metrics(rows: list[dict]) -> dict[str, float]:
    names = (
        "accuracy",
        "balanced_accuracy",
        "precision",
        "recall",
        "f1",
        "auc",
        "brier_score",
        "log_loss",
        "predicted_positive_rate",
    )
    return {
        f"{name}_mean": float(np.mean([row[name] for row in rows]))
        for name in names
    } | {
        f"{name}_std_across_seeds": float(np.std([row[name] for row in rows]))
        for name in names
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", required=True, type=Path)
    parser.add_argument("--features", required=True, type=Path)
    parser.add_argument("--schema", required=True, type=Path)
    parser.add_argument("--splits", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

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
    split_map = read_split_rows(paths["splits"])
    if len(split_map) < 2:
        raise SystemExit("rolling evaluation requires at least two temporal folds")

    definition = PilotReliabilityDefinition(
        alpha_prior=1.0,
        kappa_support=20.0,
        kappa_context=20.0,
        r_min=0.1,
        r_max=0.95,
    )
    definition.validate()
    model_args = SimpleNamespace(
        r_min=definition.r_min,
        r_max=definition.r_max,
        role_vocabulary_size=31,
        reliability_components=RELIABILITY_COMPONENTS,
        graph_edge_mode="all",
        graph_summary_mode="full",
        graph_pool_mode="role_stratified",
    )

    split_summaries = []
    prediction_rows = []
    for split_name in sorted(split_map):
        train_rows = split_map[split_name]["train"]
        validation_rows = split_map[split_name]["test"]
        if not train_rows or not validation_rows:
            raise SystemExit(f"empty rolling partition: {split_name}")
        for row in train_rows + validation_rows:
            sha256 = row["sha256"]
            if sha256 not in cache or sha256 not in feature_by_sha:
                raise SystemExit(f"missing cache or feature row: {sha256}")
            row["metadata_label"] = cache[sha256].get("metadata_label", "")
            row["metadata_year"] = cache[sha256].get("metadata_year", "")

        evidence = fit_training_fold_components(
            cache,
            {row["sha256"] for row in train_rows},
            definition,
            active_components=RELIABILITY_COMPONENTS,
        )
        graph_train, y_train = graph_matrix(
            train_rows, cache, evidence, model_args, include_reliability=True
        )
        graph_validation, y_validation = graph_matrix(
            validation_rows, cache, evidence, model_args, include_reliability=True
        )
        compact_train, compact_validation, compact_count = compact_matrix(
            train_rows,
            validation_rows,
            feature_by_sha,
            schema,
            feature_mode="base",
        )
        graph_train, graph_validation = standardize_pair(
            graph_train, graph_validation
        )
        x_train = np.concatenate(
            [GRAPH_WEIGHT * graph_train, compact_train], axis=1
        )
        x_validation = np.concatenate(
            [GRAPH_WEIGHT * graph_validation, compact_validation], axis=1
        )

        variant_seed_metrics: dict[str, list[dict]] = defaultdict(list)
        for seed in SEEDS:
            custom_weights, custom_bias = fit_logistic(
                x_train,
                y_train,
                seed,
                iterations=1500,
                learning_rate=0.05,
                l2=0.001,
            )
            custom_probability = 1.0 / (
                1.0
                + np.exp(
                    -np.clip(
                        x_validation @ custom_weights + custom_bias,
                        -40.0,
                        40.0,
                    )
                )
            )
            sklearn_logistic = LogisticRegression(
                C=1.0,
                solver="liblinear",
                max_iter=5000,
                random_state=seed,
            )
            sklearn_logistic.fit(x_train, y_train)
            sklearn_probability = sklearn_logistic.predict_proba(x_validation)[:, 1]

            extra_trees = ExtraTreesClassifier(
                n_estimators=500,
                max_features="sqrt",
                min_samples_leaf=2,
                class_weight="balanced",
                n_jobs=-1,
                random_state=seed,
            )
            extra_trees.fit(x_train, y_train)
            extra_probability = extra_trees.predict_proba(x_validation)[:, 1]

            histgb = HistGradientBoostingClassifier(
                learning_rate=0.05,
                max_iter=300,
                max_leaf_nodes=15,
                min_samples_leaf=10,
                l2_regularization=1.0,
                early_stopping=False,
                random_state=seed,
            )
            histgb.fit(x_train, y_train)
            histgb_probability = histgb.predict_proba(x_validation)[:, 1]

            probabilities = {
                "custom_l2_logistic": custom_probability,
                "sklearn_logistic": sklearn_probability,
                "extra_trees": extra_probability,
                "hist_gradient_boosting": histgb_probability,
                "extra_trees_histgb_blend_050": (
                    0.50 * extra_probability + 0.50 * histgb_probability
                ),
            }
            for weight in HGB_BLEND_WEIGHTS:
                suffix = f"{int(round(weight * 100)):03d}"
                probabilities[f"custom_logistic_histgb_blend_{suffix}"] = (
                    (1.0 - weight) * custom_probability
                    + weight * histgb_probability
                )
                probabilities[f"sklearn_logistic_histgb_blend_{suffix}"] = (
                    (1.0 - weight) * sklearn_probability
                    + weight * histgb_probability
                )

            for variant, probability in probabilities.items():
                variant_seed_metrics[variant].append(
                    score(y_validation, probability)
                )
                for row, label, value in zip(
                    validation_rows, y_validation, probability
                ):
                    prediction_rows.append(
                        {
                            "split_name": split_name,
                            "variant": variant,
                            "seed": seed,
                            "sha256": row["sha256"],
                            "label": int(label),
                            "probability": float(value),
                            "threshold": THRESHOLD,
                            "predicted_label": int(value >= THRESHOLD),
                        }
                    )

        for variant, rows in sorted(variant_seed_metrics.items()):
            split_summaries.append(
                {
                    "split_name": split_name,
                    "variant": variant,
                    "train_count": len(train_rows),
                    "validation_count": len(validation_rows),
                    "feature_count": int(x_train.shape[1]),
                    "compact_feature_count": int(compact_count),
                    **mean_seed_metrics(rows),
                }
            )

    rows_by_variant: dict[str, list[dict]] = defaultdict(list)
    for row in split_summaries:
        rows_by_variant[row["variant"]].append(row)
    aggregate_summaries = []
    for variant, rows in sorted(rows_by_variant.items()):
        fold_ba = [row["balanced_accuracy_mean"] for row in rows]
        fold_accuracy = [row["accuracy_mean"] for row in rows]
        fold_auc = [row["auc_mean"] for row in rows]
        fold_brier = [row["brier_score_mean"] for row in rows]
        fold_log_loss = [row["log_loss_mean"] for row in rows]
        aggregate_summaries.append(
            {
                "variant": variant,
                "fold_count": len(rows),
                "mean_accuracy": float(np.mean(fold_accuracy)),
                "mean_balanced_accuracy": float(np.mean(fold_ba)),
                "minimum_fold_balanced_accuracy": float(np.min(fold_ba)),
                "std_fold_balanced_accuracy": float(np.std(fold_ba)),
                "stability_score_mean_minus_std_ba": float(
                    np.mean(fold_ba) - np.std(fold_ba)
                ),
                "mean_auc": float(np.mean(fold_auc)),
                "mean_brier_score": float(np.mean(fold_brier)),
                "mean_log_loss": float(np.mean(fold_log_loss)),
            }
        )
    aggregate_summaries.sort(
        key=lambda row: (
            -row["stability_score_mean_minus_std_ba"],
            -row["mean_balanced_accuracy"],
            row["variant"],
        )
    )

    prediction_path = output / "rolling_classifier_predictions.csv"
    prediction_fields = [
        "split_name",
        "variant",
        "seed",
        "sha256",
        "label",
        "probability",
        "threshold",
        "predicted_label",
    ]
    with prediction_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=prediction_fields)
        writer.writeheader()
        writer.writerows(prediction_rows)

    summary_path = output / "rolling_classifier_summary.csv"
    with summary_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(aggregate_summaries[0]),
        )
        writer.writeheader()
        writer.writerows(aggregate_summaries)

    audit = {
        "schema_version": "1.0.0",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "phase": "P5_2017_GROUPED_ROLLING_CLASSIFIER_ENSEMBLE_DEVELOPMENT",
        "formal_training_enabled": False,
        "strict_validation_isolation": True,
        "fixed_threshold": THRESHOLD,
        "graph_pool_mode": "role_stratified",
        "graph_weight": GRAPH_WEIGHT,
        "reliability_components": list(RELIABILITY_COMPONENTS),
        "seeds": list(SEEDS),
        "cache_sha256": sha256_file(paths["cache"]),
        "features_sha256": sha256_file(paths["features"]),
        "schema_sha256": sha256_file(paths["schema"]),
        "split_manifest_sha256": sha256_file(paths["splits"]),
        "split_summaries": split_summaries,
        "aggregate_summaries": aggregate_summaries,
        "selection_rule": (
            "maximize mean rolling balanced accuracy minus its across-fold "
            "standard deviation; tie-break by mean balanced accuracy"
        ),
        "outputs": {
            "predictions": str(prediction_path),
            "summary": str(summary_path),
        },
        "blocking_reasons": [
            "dex_date_is_not_audited_first_seen_utc",
            "classifier_and_ensemble_selection_is_development_not_confirmatory",
        ],
        "notes": [
            "Only rolling 2017 rows are read.",
            "The decision threshold is fixed at 0.5 for every variant.",
            "Reliability, filtering, scaling, and fitting are fold-training-only.",
            "No 2018--2021 result is used for variant selection.",
            "No raw graph, DEX, or APK is persisted.",
        ],
    }
    audit_path = output / "rolling_classifier_ensemble_audit.json"
    audit_path.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print("rank variant mean_BA min_fold_BA std_BA stability AUC")
    for rank, row in enumerate(aggregate_summaries, start=1):
        print(
            rank,
            row["variant"],
            f'{row["mean_balanced_accuracy"]:.4f}',
            f'{row["minimum_fold_balanced_accuracy"]:.4f}',
            f'{row["std_fold_balanced_accuracy"]:.4f}',
            f'{row["stability_score_mean_minus_std_ba"]:.4f}',
            f'{row["mean_auc"]:.4f}',
        )


if __name__ == "__main__":
    main()
