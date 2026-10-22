#!/usr/bin/env python3
"""Evaluate the locked graph-pool + compact-feature candidate.

This evaluator intentionally has no threshold search.  The threshold is read
from the locked protocol and defaults to 0.5.  Reliability, constant-feature
filtering, and standardization are fit on each outer training partition only;
the outer test partition is used for scoring only.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.model_selection import StratifiedGroupKFold

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
    validate_active_components,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--cache", required=True, type=Path)
    parser.add_argument("--features", required=True, type=Path)
    parser.add_argument("--schema", required=True, type=Path)
    parser.add_argument("--splits", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--split", action="append", dest="split_names")
    args = parser.parse_args()

    config_path = args.config.expanduser().resolve(strict=True)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("status") != "CANDIDATE_LOCKED_NOT_FORMAL":
        raise SystemExit("refusing to run an unexpected or unlocked protocol config")
    decision = config["decision"]
    threshold = float(decision["threshold"])
    if decision.get("threshold_selection") != "fixed_predeclared":
        raise SystemExit("locked evaluator requires fixed_predeclared threshold")
    if not (0.0 < threshold < 1.0):
        raise SystemExit("locked threshold must be between 0 and 1")

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
    if args.split_names:
        unknown = set(args.split_names) - set(split_rows)
        if unknown:
            raise SystemExit(f"unknown split names: {sorted(unknown)}")
        split_rows = {name: split_rows[name] for name in args.split_names}

    reliability_config = config["reliability"]
    reliability_components = validate_active_components(
        reliability_config.get("components")
    )
    definition = PilotReliabilityDefinition(
        alpha_prior=float(reliability_config["alpha_prior"]),
        kappa_support=float(reliability_config["kappa_support"]),
        kappa_context=float(reliability_config["kappa_context"]),
        r_min=float(reliability_config["r_min"]),
        r_max=float(reliability_config["r_max"]),
    )
    definition.validate()
    model_config = config["classifier"]
    classifier_type = str(model_config["type"])
    if classifier_type not in {
        "l2_logistic_regression",
        "sklearn_logistic_regression",
        "hist_gradient_boosting",
        "calibrated_hist_gradient_boosting_sigmoid",
        "sklearn_logistic_histgb_probability_blend",
    }:
        raise SystemExit(f"unsupported locked classifier: {classifier_type}")
    if classifier_type == "calibrated_hist_gradient_boosting_sigmoid":
        if model_config.get("calibration_method") != "sigmoid":
            raise SystemExit("calibrated HistGB requires sigmoid calibration")
        if model_config.get("calibration_grouping") != "blocking_group":
            raise SystemExit("calibration must preserve blocking groups")
        if model_config.get("calibration_scope") != "outer_training_partition_only":
            raise SystemExit("calibration must be fit on the outer training partition only")
        if model_config.get("calibration_ensemble") is not False:
            raise SystemExit("locked calibrated HistGB requires calibration_ensemble=false")
    if classifier_type == "sklearn_logistic_histgb_probability_blend":
        histgb_weight = float(model_config["histgb_probability_weight"])
        logistic_weight = float(model_config["logistic_probability_weight"])
        if not np.isclose(histgb_weight + logistic_weight, 1.0):
            raise SystemExit("locked blend probability weights must sum to 1")
        if histgb_weight <= 0.0 or logistic_weight <= 0.0:
            raise SystemExit("locked blend probability weights must be positive")
    seeds = [int(value) for value in model_config["seeds"]]
    if not seeds:
        raise SystemExit("locked protocol has no classifier seeds")

    graph_summary_config = config["graph_summary"]
    model_args = type("ModelArgs", (), {})()
    model_args.r_min = definition.r_min
    model_args.r_max = definition.r_max
    model_args.role_vocabulary_size = int(graph_summary_config["role_vocabulary_size"])
    model_args.reliability_components = reliability_components
    model_args.graph_edge_mode = graph_summary_config.get("edge_mode", "all")
    model_args.graph_summary_mode = graph_summary_config.get("summary_mode", "full")
    model_args.graph_pool_mode = graph_summary_config.get("pool_mode", "global")
    model_args.graph_prevalence_mode = graph_summary_config.get(
        "prevalence_weighting", "none"
    )
    model_args.graph_idf_floor = float(graph_summary_config.get("idf_floor", 0.05))
    model_args.graph_discriminative_mode = graph_summary_config.get(
        "discriminative_node_evidence", "none"
    )
    model_args.graph_contrast_alpha = float(
        graph_summary_config.get("contrast_alpha", 2.0)
    )
    model_args.graph_contrast_kappa = float(
        graph_summary_config.get("contrast_kappa", 10.0)
    )
    model_args.graph_contrast_clip = float(
        graph_summary_config.get("contrast_clip", 4.0)
    )
    if model_args.graph_edge_mode not in {"all", "direct_only"}:
        raise SystemExit(f"unsupported graph edge mode: {model_args.graph_edge_mode}")
    if model_args.graph_summary_mode not in {"full", "no_edge_summary"}:
        raise SystemExit(f"unsupported graph summary mode: {model_args.graph_summary_mode}")
    if model_args.graph_pool_mode not in {
        "global", "role_stratified", "role_edge_stratified",
        "role_edge_type_stratified", "role_edge_type_boundary_stratified",
        "role_edge_type_boundary_path_stratified",
    }:
        raise SystemExit(f"unsupported graph pool mode: {model_args.graph_pool_mode}")
    if model_args.graph_prevalence_mode not in {"none", "train_idf"}:
        raise SystemExit(
            f"unsupported graph prevalence mode: {model_args.graph_prevalence_mode}"
        )
    if not (0.0 < model_args.graph_idf_floor <= 1.0):
        raise SystemExit("graph IDF floor must be in (0, 1]")
    if model_args.graph_discriminative_mode not in {
        "none", "train_smoothed_log_odds"
    }:
        raise SystemExit(
            "unsupported graph discriminative node evidence: "
            f"{model_args.graph_discriminative_mode}"
        )
    if any(value <= 0.0 for value in (
        model_args.graph_contrast_alpha,
        model_args.graph_contrast_kappa,
        model_args.graph_contrast_clip,
    )):
        raise SystemExit("graph contrast parameters must be positive")
    graph_weight = float(graph_summary_config.get("graph_weight", 1.0))
    if graph_weight <= 0.0:
        raise SystemExit("graph_weight must be positive")
    compact_feature_mode = str(
        config.get("compact_features", {}).get("mode", "base")
    )
    if compact_feature_mode not in {
        "base", "coverage", "coverage_interactions",
        "size_density_additive", "size_density_only",
    }:
        raise SystemExit(
            f"unsupported compact feature mode: {compact_feature_mode}"
        )

    prediction_rows = []
    summaries = []
    for split_name in sorted(split_rows):
        train_rows = split_rows[split_name]["train"]
        test_rows = split_rows[split_name]["test"]
        if not train_rows or not test_rows:
            raise SystemExit(f"empty outer partition: {split_name}")
        for row in train_rows + test_rows:
            sha256 = row["sha256"]
            if sha256 not in cache or sha256 not in feature_by_sha:
                raise SystemExit(f"missing cache or feature row: {sha256}")
            row["metadata_label"] = cache[sha256].get("metadata_label", "")
            row["metadata_year"] = cache[sha256].get("metadata_year", "")

        evidence = fit_training_fold_components(
            cache,
            {row["sha256"] for row in train_rows},
            definition,
            active_components=reliability_components,
        )
        x_train, y_train = graph_matrix(
            train_rows, cache, evidence, model_args, include_reliability=True
        )
        x_test, y_test = graph_matrix(
            test_rows, cache, evidence, model_args, include_reliability=True
        )
        compact_train, compact_test, compact_count = compact_matrix(
            train_rows, test_rows, feature_by_sha, schema,
            feature_mode=compact_feature_mode,
        )
        x_train, x_test = standardize_pair(x_train, x_test)
        x_train = np.concatenate([graph_weight * x_train, compact_train], axis=1)
        x_test = np.concatenate([graph_weight * x_test, compact_test], axis=1)

        seed_metrics = []
        for seed in seeds:
            if classifier_type == "l2_logistic_regression":
                weights, bias = fit_logistic(
                    x_train,
                    y_train,
                    seed,
                    iterations=int(model_config["iterations"]),
                    learning_rate=float(model_config["learning_rate"]),
                    l2=float(model_config["l2"]),
                )
                probability = 1.0 / (
                    1.0
                    + np.exp(-np.clip(x_test @ weights + bias, -40.0, 40.0))
                )
            elif classifier_type == "sklearn_logistic_regression":
                classifier = LogisticRegression(
                    C=float(model_config["C"]),
                    solver=str(model_config["solver"]),
                    max_iter=int(model_config["max_iter"]),
                    random_state=seed,
                )
                classifier.fit(x_train, y_train)
                probability = classifier.predict_proba(x_test)[:, 1]
            else:
                base_classifier = HistGradientBoostingClassifier(
                    learning_rate=float(model_config["learning_rate"]),
                    max_iter=int(model_config["max_iter"]),
                    max_leaf_nodes=int(model_config["max_leaf_nodes"]),
                    min_samples_leaf=int(model_config["min_samples_leaf"]),
                    l2_regularization=float(model_config["l2_regularization"]),
                    early_stopping=False,
                    random_state=seed,
                )
                if classifier_type == "calibrated_hist_gradient_boosting_sigmoid":
                    calibration_folds = int(model_config["calibration_folds"])
                    if calibration_folds < 2:
                        raise SystemExit("calibration_folds must be at least 2")
                    calibration_groups = np.asarray(
                        [row["blocking_group"] for row in train_rows],
                        dtype=object,
                    )
                    splitter = StratifiedGroupKFold(
                        n_splits=calibration_folds,
                        shuffle=True,
                        random_state=seed,
                    )
                    calibration_cv = list(
                        splitter.split(
                            x_train,
                            y_train,
                            groups=calibration_groups,
                        )
                    )
                    classifier = CalibratedClassifierCV(
                        estimator=base_classifier,
                        method="sigmoid",
                        cv=calibration_cv,
                        n_jobs=1,
                        ensemble=False,
                    )
                elif classifier_type == "sklearn_logistic_histgb_probability_blend":
                    logistic_classifier = LogisticRegression(
                        C=float(model_config["logistic_C"]),
                        solver=str(model_config["logistic_solver"]),
                        max_iter=int(model_config["logistic_max_iter"]),
                        random_state=seed,
                    )
                    base_classifier.fit(x_train, y_train)
                    logistic_classifier.fit(x_train, y_train)
                    histgb_probability = base_classifier.predict_proba(x_test)[:, 1]
                    logistic_probability = logistic_classifier.predict_proba(x_test)[:, 1]
                    probability = (
                        histgb_weight * histgb_probability
                        + logistic_weight * logistic_probability
                    )
                    classifier = None
                else:
                    classifier = base_classifier
                if classifier is not None:
                    classifier.fit(x_train, y_train)
                    probability = classifier.predict_proba(x_test)[:, 1]
            result = metrics(y_test, probability, threshold=threshold)
            result["brier_score"] = float(
                brier_score_loss(y_test, probability)
            )
            result["log_loss"] = float(
                log_loss(y_test, np.clip(probability, 1e-7, 1.0 - 1e-7))
            )
            result["predicted_positive_rate"] = float(
                np.mean(probability >= threshold)
            )
            seed_metrics.append(result)
            for row, label, score in zip(test_rows, y_test, probability):
                prediction_rows.append(
                    {
                        "split_name": split_name,
                        "sha256": row["sha256"],
                        "seed": seed,
                        "label": int(label),
                        "probability": float(score),
                        "threshold": threshold,
                        "predicted_label": int(score >= threshold),
                    }
                )
        summaries.append(
            {
                "split_name": split_name,
                "train_count": len(train_rows),
                "test_count": len(test_rows),
                "feature_count": int(x_train.shape[1]),
                "compact_feature_count": int(compact_count),
                "threshold": threshold,
                "accuracy_mean": float(np.mean([row["accuracy"] for row in seed_metrics])),
                "balanced_accuracy_mean": float(np.mean([row["balanced_accuracy"] for row in seed_metrics])),
                "precision_mean": float(np.mean([row["precision"] for row in seed_metrics])),
                "recall_mean": float(np.mean([row["recall"] for row in seed_metrics])),
                "f1_mean": float(np.mean([row["f1"] for row in seed_metrics])),
                "auc_mean": float(np.mean([row["auc"] for row in seed_metrics])),
                "brier_score_mean": float(
                    np.mean([row["brier_score"] for row in seed_metrics])
                ),
                "log_loss_mean": float(
                    np.mean([row["log_loss"] for row in seed_metrics])
                ),
                "predicted_positive_rate_mean": float(
                    np.mean([row["predicted_positive_rate"] for row in seed_metrics])
                ),
                "test_rows_used_for_training_or_selection": False,
                "training_years": sorted({str(row.get("metadata_year", "")) for row in train_rows}),
            }
        )

    prediction_path = output / "graph_pool_locked_predictions.csv"
    fields = ["split_name", "sha256", "seed", "label", "probability", "threshold", "predicted_label"]
    with prediction_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(prediction_rows)
    audit = {
        "schema_version": "1.0.0",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "phase": "P3_GRAPH_POOL_FUSED_LOCKED_PILOT_EVALUATION",
        "formal_training_enabled": False,
        "protocol_status": "CANDIDATE_LOCKED_NOT_FORMAL",
        "strict_test_isolation": True,
        "protocol_config": str(config_path),
        "protocol_config_sha256": sha256_file(config_path),
        "cache": str(paths["cache"]),
        "cache_sha256": sha256_file(paths["cache"]),
        "features": str(paths["features"]),
        "features_sha256": sha256_file(paths["features"]),
        "schema": str(paths["schema"]),
        "schema_sha256": sha256_file(paths["schema"]),
        "split_manifest": str(paths["splits"]),
        "split_manifest_sha256": sha256_file(paths["splits"]),
        "fixed_threshold": threshold,
        "graph_weight": graph_weight,
        "graph_edge_mode": model_args.graph_edge_mode,
        "graph_summary_mode": model_args.graph_summary_mode,
        "graph_pool_mode": model_args.graph_pool_mode,
        "graph_prevalence_mode": model_args.graph_prevalence_mode,
        "graph_idf_floor": model_args.graph_idf_floor,
        "graph_discriminative_mode": model_args.graph_discriminative_mode,
        "graph_contrast_alpha": model_args.graph_contrast_alpha,
        "graph_contrast_kappa": model_args.graph_contrast_kappa,
        "graph_contrast_clip": model_args.graph_contrast_clip,
        "compact_feature_mode": compact_feature_mode,
        "classifier_type": classifier_type,
        "reliability_components": list(reliability_components),
        "seeds": seeds,
        "split_summaries": summaries,
        "evaluated_split_count": len(summaries),
        "mean_accuracy_over_evaluated_splits": float(
            np.mean([row["accuracy_mean"] for row in summaries])
        ),
        "mean_balanced_accuracy_over_evaluated_splits": float(
            np.mean([row["balanced_accuracy_mean"] for row in summaries])
        ),
        "outputs": {"predictions": str(prediction_path)},
        "blocking_reasons": config.get("formalization_blockers", []),
        "notes": [
            "Threshold is fixed at protocol value 0.5; no per-year threshold selection is performed.",
            "Reliability, compact filtering, and standardization use outer training rows only.",
            "When enabled, sigmoid calibration uses only grouped folds of the outer training partition.",
            "Raw DEX, raw graphs, and method signatures are not persisted.",
            "This locked candidate result is not a formal confirmatory result.",
        ],
    }
    audit_path = output / "graph_pool_locked_evaluation_audit.json"
    audit_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
