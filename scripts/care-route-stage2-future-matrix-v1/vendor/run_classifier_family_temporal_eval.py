#!/usr/bin/env python3
"""Compare fixed classifier families on the 2017 temporal development split.

The graph representation, graph weight, reliability definition, and decision
threshold are fixed.  Only the classifier family varies.  All preprocessing
and reliability evidence are fit on the early-2017 training partition; the
late-2017 partition is used for development scoring only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.svm import SVC

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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def mean_metrics(seed_rows: list[dict]) -> dict:
    names = (
        "accuracy",
        "balanced_accuracy",
        "precision",
        "recall",
        "f1",
        "auc",
        "brier_score",
        "log_loss",
    )
    return {
        f"{name}_mean": float(np.mean([row[name] for row in seed_rows]))
        for name in names
    } | {
        f"{name}_std": float(np.std([row[name] for row in seed_rows]))
        for name in names
    }


def probability_metrics(labels: np.ndarray, probability: np.ndarray) -> dict:
    result = metrics(labels, probability, threshold=THRESHOLD)
    clipped_probability = np.clip(probability, 1e-7, 1.0 - 1e-7)
    result["brier_score"] = float(brier_score_loss(labels, probability))
    result["log_loss"] = float(log_loss(labels, clipped_probability))
    result["predicted_positive_rate"] = float(np.mean(probability >= THRESHOLD))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", required=True, type=Path)
    parser.add_argument("--features", required=True, type=Path)
    parser.add_argument("--schema", required=True, type=Path)
    parser.add_argument("--splits", required=True, type=Path)
    parser.add_argument(
        "--split",
        default="train_2017_early__validate_2017_late",
    )
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
    if args.split not in split_map:
        raise SystemExit(f"unknown split: {args.split}")
    train_rows = split_map[args.split]["train"]
    validation_rows = split_map[args.split]["test"]
    if not train_rows or not validation_rows:
        raise SystemExit("empty temporal train or validation partition")

    for row in train_rows + validation_rows:
        sha256 = row["sha256"]
        if sha256 not in cache or sha256 not in feature_by_sha:
            raise SystemExit(f"missing cache or feature row: {sha256}")
        row["metadata_label"] = cache[sha256].get("metadata_label", "")
        row["metadata_year"] = cache[sha256].get("metadata_year", "")

    definition = PilotReliabilityDefinition(
        alpha_prior=1.0,
        kappa_support=20.0,
        kappa_context=20.0,
        r_min=0.1,
        r_max=0.95,
    )
    definition.validate()
    evidence = fit_training_fold_components(
        cache,
        {row["sha256"] for row in train_rows},
        definition,
        active_components=RELIABILITY_COMPONENTS,
    )
    model_args = SimpleNamespace(
        r_min=definition.r_min,
        r_max=definition.r_max,
        role_vocabulary_size=31,
        reliability_components=RELIABILITY_COMPONENTS,
        graph_edge_mode="all",
        graph_summary_mode="full",
        graph_pool_mode="role_stratified",
    )
    graph_train, y_train = graph_matrix(
        train_rows, cache, evidence, model_args, include_reliability=True
    )
    graph_validation, y_validation = graph_matrix(
        validation_rows, cache, evidence, model_args, include_reliability=True
    )
    compact_train, compact_validation, compact_count = compact_matrix(
        train_rows, validation_rows, feature_by_sha, schema, feature_mode="base"
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

    summaries = []

    custom_rows = []
    for seed in SEEDS:
        weights, bias = fit_logistic(
            x_train,
            y_train,
            seed,
            iterations=1500,
            learning_rate=0.05,
            l2=0.001,
        )
        probability = 1.0 / (
            1.0 + np.exp(-np.clip(x_validation @ weights + bias, -40.0, 40.0))
        )
        custom_rows.append(probability_metrics(y_validation, probability))
    summaries.append(
        {
            "classifier": "custom_l2_logistic",
            "configuration": {
                "iterations": 1500,
                "learning_rate": 0.05,
                "l2": 0.001,
            },
            **mean_metrics(custom_rows),
        }
    )

    classifier_specs = {
        "sklearn_logistic": {
            "configuration": {"C": 1.0, "solver": "liblinear", "max_iter": 5000},
            "factory": lambda seed: LogisticRegression(
                C=1.0,
                solver="liblinear",
                max_iter=5000,
                random_state=seed,
            ),
        },
        "extra_trees": {
            "configuration": {
                "n_estimators": 500,
                "max_features": "sqrt",
                "min_samples_leaf": 2,
                "class_weight": "balanced",
            },
            "factory": lambda seed: ExtraTreesClassifier(
                n_estimators=500,
                max_features="sqrt",
                min_samples_leaf=2,
                class_weight="balanced",
                n_jobs=-1,
                random_state=seed,
            ),
        },
        "hist_gradient_boosting": {
            "configuration": {
                "learning_rate": 0.05,
                "max_iter": 300,
                "max_leaf_nodes": 15,
                "min_samples_leaf": 10,
                "l2_regularization": 1.0,
            },
            "factory": lambda seed: HistGradientBoostingClassifier(
                learning_rate=0.05,
                max_iter=300,
                max_leaf_nodes=15,
                min_samples_leaf=10,
                l2_regularization=1.0,
                random_state=seed,
            ),
        },
        "rbf_svc": {
            "configuration": {
                "C": 1.0,
                "kernel": "rbf",
                "gamma": "scale",
                "probability": True,
            },
            "factory": lambda seed: SVC(
                C=1.0,
                kernel="rbf",
                gamma="scale",
                probability=True,
                random_state=seed,
            ),
        },
    }

    for classifier_name, spec in classifier_specs.items():
        seed_rows = []
        for seed in SEEDS:
            classifier = spec["factory"](seed)
            classifier.fit(x_train, y_train)
            probability = classifier.predict_proba(x_validation)[:, 1]
            seed_rows.append(probability_metrics(y_validation, probability))
        summaries.append(
            {
                "classifier": classifier_name,
                "configuration": spec["configuration"],
                **mean_metrics(seed_rows),
            }
        )

    calibration_specs = {
        "calibrated_extra_trees_sigmoid": {
            "configuration": {
                "method": "sigmoid",
                "cv": "stratified_group_5_fold",
                "ensemble": False,
                "base_n_estimators": 500,
                "base_max_features": "sqrt",
                "base_min_samples_leaf": 2,
                "base_class_weight": "balanced",
            },
            "factory": lambda seed: ExtraTreesClassifier(
                n_estimators=500,
                max_features="sqrt",
                min_samples_leaf=2,
                class_weight="balanced",
                n_jobs=-1,
                random_state=seed,
            ),
        },
        "calibrated_hist_gradient_boosting_sigmoid": {
            "configuration": {
                "method": "sigmoid",
                "cv": "stratified_group_5_fold",
                "ensemble": False,
                "base_learning_rate": 0.05,
                "base_max_iter": 300,
                "base_max_leaf_nodes": 15,
                "base_min_samples_leaf": 10,
                "base_l2_regularization": 1.0,
            },
            "factory": lambda seed: HistGradientBoostingClassifier(
                learning_rate=0.05,
                max_iter=300,
                max_leaf_nodes=15,
                min_samples_leaf=10,
                l2_regularization=1.0,
                early_stopping=False,
                random_state=seed,
            ),
        },
    }
    calibration_groups = np.asarray(
        [row["blocking_group"] for row in train_rows], dtype=object
    )
    for classifier_name, spec in calibration_specs.items():
        seed_rows = []
        for seed in SEEDS:
            splitter = StratifiedGroupKFold(
                n_splits=5,
                shuffle=True,
                random_state=seed,
            )
            calibration_cv = list(
                splitter.split(x_train, y_train, groups=calibration_groups)
            )
            classifier = CalibratedClassifierCV(
                estimator=spec["factory"](seed),
                method="sigmoid",
                cv=calibration_cv,
                n_jobs=1,
                ensemble=False,
            )
            classifier.fit(x_train, y_train)
            probability = classifier.predict_proba(x_validation)[:, 1]
            seed_rows.append(probability_metrics(y_validation, probability))
        summaries.append(
            {
                "classifier": classifier_name,
                "configuration": spec["configuration"],
                **mean_metrics(seed_rows),
            }
        )

    audit = {
        "schema_version": "1.0.0",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "phase": "P5_2017_TEMPORAL_CLASSIFIER_FAMILY_DEVELOPMENT",
        "formal_training_enabled": False,
        "strict_validation_isolation": True,
        "split_name": args.split,
        "train_count": len(train_rows),
        "validation_count": len(validation_rows),
        "feature_count": int(x_train.shape[1]),
        "compact_feature_count": int(compact_count),
        "graph_pool_mode": "role_stratified",
        "graph_weight": GRAPH_WEIGHT,
        "reliability_components": list(RELIABILITY_COMPONENTS),
        "fixed_threshold": THRESHOLD,
        "seeds": list(SEEDS),
        "cache_sha256": sha256_file(paths["cache"]),
        "features_sha256": sha256_file(paths["features"]),
        "schema_sha256": sha256_file(paths["schema"]),
        "split_manifest_sha256": sha256_file(paths["splits"]),
        "summaries": summaries,
        "blocking_reasons": [
            "dex_date_is_not_audited_first_seen_utc",
            "classifier_family_selection_is_development_not_confirmatory",
            "legacy_2018_2021_years_must_not_be_used_for_this_selection",
        ],
        "notes": [
            "Threshold is fixed at 0.5 for every classifier.",
            "Reliability, feature filtering, and scaling use early-2017 rows only.",
            "Calibration folds keep blocking groups intact and use training rows only.",
            "Late-2017 rows are used only for classifier-family development scoring.",
            "No raw graph or APK is persisted.",
        ],
    }
    audit_path = output / "classifier_family_temporal_audit.json"
    audit_path.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
