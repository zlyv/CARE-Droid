#!/usr/bin/env python3
"""Evaluate a training-fold-only CCA-consistent dual graph representation.

Entry and endpoint graph summaries of the same APK are aligned with regularized
canonical correlation analysis fitted on each chronological 2017 training
partition. The shared representation is a fixed 0.75/0.25 entry/endpoint
combination. Compact features are optionally retained as residual evidence.
The classification threshold is always 0.5. No APK or future-year row is read.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from run_dual_view_2017_rolling_eval import (  # noqa: E402
    ACTIVE_RELIABILITY_COMPONENTS,
    attach_audit_metadata,
    endpoint_matrix,
    entry_model_args,
)
from run_dual_view_error_attribution import compact_matrix_named  # noqa: E402
from run_graph_pool_diagnostic import (  # noqa: E402
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
ENTRY_WEIGHT = 0.75
ENDPOINT_WEIGHT = 0.25
DEFAULT_DIMENSIONS = (8, 16, 32)


@dataclass(frozen=True)
class CCAModel:
    x_mean: np.ndarray
    y_mean: np.ndarray
    x_projection: np.ndarray
    y_projection: np.ndarray
    canonical_correlations: np.ndarray


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _inverse_sqrt_psd(matrix: np.ndarray) -> np.ndarray:
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    floor = max(float(np.max(eigenvalues)) * 1e-12, 1e-12)
    inverse = 1.0 / np.sqrt(np.maximum(eigenvalues, floor))
    return (eigenvectors * inverse[None, :]) @ eigenvectors.T


def fit_regularized_cca(
    x_train: np.ndarray,
    y_train: np.ndarray,
    dimensions: int,
    regularization: float,
) -> CCAModel:
    x = np.asarray(x_train, dtype=np.float64)
    y = np.asarray(y_train, dtype=np.float64)
    if x.ndim != 2 or y.ndim != 2 or len(x) != len(y):
        raise ValueError("CCA inputs must be aligned two-dimensional matrices")
    if len(x) < 2 or x.shape[1] == 0 or y.shape[1] == 0:
        raise ValueError("CCA requires at least two rows and non-empty views")
    if regularization <= 0.0 or not np.isfinite(regularization):
        raise ValueError("CCA regularization must be positive and finite")
    maximum_dimensions = min(x.shape[1], y.shape[1], len(x) - 1)
    if dimensions <= 0 or dimensions > maximum_dimensions:
        raise ValueError(
            f"CCA dimensions must be in [1, {maximum_dimensions}]"
        )
    x_mean = x.mean(axis=0)
    y_mean = y.mean(axis=0)
    x_centered = x - x_mean
    y_centered = y - y_mean
    denominator = float(len(x) - 1)
    c_xx = x_centered.T @ x_centered / denominator
    c_yy = y_centered.T @ y_centered / denominator
    c_xy = x_centered.T @ y_centered / denominator
    c_xx += regularization * np.eye(c_xx.shape[0])
    c_yy += regularization * np.eye(c_yy.shape[0])
    whiten_x = _inverse_sqrt_psd(c_xx)
    whiten_y = _inverse_sqrt_psd(c_yy)
    u, singular_values, vt = np.linalg.svd(
        whiten_x @ c_xy @ whiten_y,
        full_matrices=False,
    )
    return CCAModel(
        x_mean=x_mean,
        y_mean=y_mean,
        x_projection=whiten_x @ u[:, :dimensions],
        y_projection=whiten_y @ vt.T[:, :dimensions],
        canonical_correlations=np.clip(
            singular_values[:dimensions], 0.0, 1.0
        ),
    )


def transform_cca(
    model: CCAModel,
    x: np.ndarray,
    y: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    x_array = np.asarray(x, dtype=np.float64)
    y_array = np.asarray(y, dtype=np.float64)
    if x_array.ndim != 2 or y_array.ndim != 2 or len(x_array) != len(y_array):
        raise ValueError("CCA transform inputs must be aligned matrices")
    if x_array.shape[1] != len(model.x_mean):
        raise ValueError("CCA entry feature dimension mismatch")
    if y_array.shape[1] != len(model.y_mean):
        raise ValueError("CCA endpoint feature dimension mismatch")
    return (
        (x_array - model.x_mean) @ model.x_projection,
        (y_array - model.y_mean) @ model.y_projection,
    )


def shared_representation(
    entry_latent: np.ndarray,
    endpoint_latent: np.ndarray,
) -> np.ndarray:
    if entry_latent.shape != endpoint_latent.shape:
        raise ValueError("aligned latent views must have equal shapes")
    return ENTRY_WEIGHT * entry_latent + ENDPOINT_WEIGHT * endpoint_latent


def disagreement_representation(
    entry_latent: np.ndarray,
    endpoint_latent: np.ndarray,
) -> np.ndarray:
    if entry_latent.shape != endpoint_latent.shape:
        raise ValueError("aligned latent views must have equal shapes")
    return np.abs(entry_latent - endpoint_latent)


def sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(values, -40.0, 40.0)))


def predict_logistic(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_eval: np.ndarray,
    seed: int,
    iterations: int,
    learning_rate: float,
    l2: float,
) -> np.ndarray:
    weights, bias = fit_logistic(
        x_train,
        y_train,
        seed,
        iterations=iterations,
        learning_rate=learning_rate,
        l2=l2,
    )
    return sigmoid(x_eval @ weights + bias)


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
    parser.add_argument(
        "--dimensions", nargs="+", type=int, default=list(DEFAULT_DIMENSIONS)
    )
    parser.add_argument("--cca-regularization", type=float, default=0.1)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--iterations", type=int, default=1500)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--l2", type=float, default=0.03)
    args = parser.parse_args()
    dimensions = sorted(set(args.dimensions))
    if not dimensions or any(value <= 0 for value in dimensions):
        raise SystemExit("CCA dimensions must be positive")
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
        "entry_cache": selected_shas - set(entry_cache),
        "endpoint_cache": selected_shas - set(endpoint_cache),
        "features": selected_shas - set(feature_by_sha),
    }
    if any(missing.values()):
        raise SystemExit(
            "missing selected rows: "
            + str({name: len(values) for name, values in missing.items()})
        )
    observed_years = {
        str(feature_by_sha[sha].get("metadata_year", ""))
        for sha in selected_shas
    }
    if observed_years != {"2017"}:
        raise SystemExit(
            "P35 accepts only 2017 development rows; "
            f"observed={sorted(observed_years)}"
        )

    definition = PilotReliabilityDefinition(
        alpha_prior=1.0,
        kappa_support=20.0,
        kappa_context=20.0,
        r_min=0.10,
        r_max=0.95,
    )
    entry_args = entry_model_args(definition)
    variants = ["control_probability_blend"]
    for dimension in dimensions:
        variants.extend([
            f"cca_shared_d{dimension}_compact",
            f"cca_shared_d{dimension}_only",
        ])
    middle_dimension = dimensions[len(dimensions) // 2]
    variants.append(
        f"cca_shared_disagreement_d{middle_dimension}_compact"
    )
    per_variant_fold = {name: [] for name in variants}
    prediction_rows = []
    canonical_audit = []

    for split_name in sorted(split_rows):
        train_rows = split_rows[split_name]["train"]
        validation_rows = split_rows[split_name]["test"]
        if not train_rows or not validation_rows:
            raise SystemExit(f"empty development partition: {split_name}")
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
            raise SystemExit(f"cache label mismatch: {split_name}")
        compact_train, compact_validation, compact_names = compact_matrix_named(
            train_rows,
            validation_rows,
            feature_by_sha,
            schema,
        )
        entry_train, entry_validation = standardize_pair(
            entry_train, entry_validation
        )
        endpoint_train, endpoint_validation = standardize_pair(
            endpoint_train, endpoint_validation
        )

        maximum_dimension = max(dimensions)
        cca = fit_regularized_cca(
            entry_train,
            endpoint_train,
            maximum_dimension,
            args.cca_regularization,
        )
        entry_latent_train, endpoint_latent_train = transform_cca(
            cca, entry_train, endpoint_train
        )
        entry_latent_validation, endpoint_latent_validation = transform_cca(
            cca, entry_validation, endpoint_validation
        )
        canonical_audit.append({
            "split_name": split_name,
            "train_count": len(train_rows),
            "validation_count": len(validation_rows),
            "entry_feature_count": int(entry_train.shape[1]),
            "endpoint_feature_count": int(endpoint_train.shape[1]),
            "compact_feature_count": len(compact_names),
            "canonical_correlations": [
                float(value) for value in cca.canonical_correlations
            ],
        })

        matrices = {}
        entry_expert_train = np.concatenate(
            [0.5 * entry_train, compact_train], axis=1
        )
        entry_expert_validation = np.concatenate(
            [0.5 * entry_validation, compact_validation], axis=1
        )
        endpoint_expert_train = np.concatenate(
            [endpoint_train, compact_train], axis=1
        )
        endpoint_expert_validation = np.concatenate(
            [endpoint_validation, compact_validation], axis=1
        )
        matrices["control_probability_blend"] = None

        for dimension in dimensions:
            train_shared = shared_representation(
                entry_latent_train[:, :dimension],
                endpoint_latent_train[:, :dimension],
            )
            validation_shared = shared_representation(
                entry_latent_validation[:, :dimension],
                endpoint_latent_validation[:, :dimension],
            )
            train_shared, validation_shared = standardize_pair(
                train_shared, validation_shared
            )
            matrices[f"cca_shared_d{dimension}_compact"] = (
                np.concatenate([train_shared, compact_train], axis=1),
                np.concatenate([validation_shared, compact_validation], axis=1),
            )
            matrices[f"cca_shared_d{dimension}_only"] = (
                train_shared,
                validation_shared,
            )
            if dimension == middle_dimension:
                train_disagreement = disagreement_representation(
                    entry_latent_train[:, :dimension],
                    endpoint_latent_train[:, :dimension],
                )
                validation_disagreement = disagreement_representation(
                    entry_latent_validation[:, :dimension],
                    endpoint_latent_validation[:, :dimension],
                )
                train_disagreement, validation_disagreement = standardize_pair(
                    train_disagreement, validation_disagreement
                )
                name = f"cca_shared_disagreement_d{dimension}_compact"
                matrices[name] = (
                    np.concatenate([
                        train_shared, train_disagreement, compact_train
                    ], axis=1),
                    np.concatenate([
                        validation_shared,
                        validation_disagreement,
                        compact_validation,
                    ], axis=1),
                )

        seed_metrics = {name: [] for name in variants}
        for seed in args.seeds:
            entry_probability = predict_logistic(
                entry_expert_train,
                y_train,
                entry_expert_validation,
                seed,
                args.iterations,
                args.learning_rate,
                args.l2,
            )
            endpoint_probability = predict_logistic(
                endpoint_expert_train,
                y_train,
                endpoint_expert_validation,
                seed,
                args.iterations,
                args.learning_rate,
                args.l2,
            )
            control_probability = (
                ENTRY_WEIGHT * entry_probability
                + ENDPOINT_WEIGHT * endpoint_probability
            )
            probabilities = {
                "control_probability_blend": control_probability
            }
            for variant in variants:
                if variant == "control_probability_blend":
                    continue
                x_train, x_validation = matrices[variant]
                probabilities[variant] = predict_logistic(
                    x_train,
                    y_train,
                    x_validation,
                    seed,
                    args.iterations,
                    args.learning_rate,
                    args.l2,
                )
            for variant, probability in probabilities.items():
                seed_metrics[variant].append(
                    metrics(y_validation, probability, threshold=THRESHOLD)
                )
                for row, label, score in zip(
                    validation_rows, y_validation, probability
                ):
                    prediction_rows.append({
                        "variant": variant,
                        "split_name": split_name,
                        "sha256": row["sha256"],
                        "seed": seed,
                        "label": int(label),
                        "probability": float(score),
                        "threshold": THRESHOLD,
                        "predicted_label": int(score >= THRESHOLD),
                    })

        for variant in variants:
            if variant == "control_probability_blend":
                feature_count = (
                    f"entry={entry_expert_train.shape[1]};"
                    f"endpoint={endpoint_expert_train.shape[1]}"
                )
            else:
                feature_count = int(matrices[variant][0].shape[1])
            fold = {
                "variant": variant,
                "split_name": split_name,
                "train_count": len(train_rows),
                "validation_count": len(validation_rows),
                "feature_count": feature_count,
                "threshold": THRESHOLD,
            }
            for metric_name in (
                "accuracy", "balanced_accuracy", "precision", "recall", "f1", "auc"
            ):
                fold[f"{metric_name}_mean"] = float(np.mean([
                    item[metric_name] for item in seed_metrics[variant]
                ]))
            per_variant_fold[variant].append(fold)
        print(f"completed {split_name}", flush=True)

    summaries = []
    for variant, folds in per_variant_fold.items():
        ba = np.asarray([row["balanced_accuracy_mean"] for row in folds])
        summaries.append({
            "variant": variant,
            "mean_accuracy": float(np.mean([row["accuracy_mean"] for row in folds])),
            "mean_balanced_accuracy": float(ba.mean()),
            "minimum_fold_balanced_accuracy": float(ba.min()),
            "std_balanced_accuracy": float(ba.std()),
            "stability_score": float(ba.mean() - ba.std()),
            "mean_auc": float(np.mean([row["auc_mean"] for row in folds])),
            "fold_summaries": folds,
        })
    summaries.sort(key=lambda row: (
        -row["stability_score"],
        -row["mean_balanced_accuracy"],
        row["variant"],
    ))
    for rank, row in enumerate(summaries, 1):
        row["rank"] = rank

    summary_path = output / "cca_consistent_2017_summary.csv"
    summary_fields = [
        "rank", "variant", "mean_accuracy", "mean_balanced_accuracy",
        "minimum_fold_balanced_accuracy", "std_balanced_accuracy",
        "stability_score", "mean_auc",
    ]
    write_csv(summary_path, summary_fields, summaries)
    fold_path = output / "cca_consistent_2017_per_fold.csv"
    fold_fields = [
        "variant", "split_name", "train_count", "validation_count",
        "feature_count", "threshold", "accuracy_mean",
        "balanced_accuracy_mean", "precision_mean", "recall_mean",
        "f1_mean", "auc_mean",
    ]
    write_csv(
        fold_path,
        fold_fields,
        [row for variant in variants for row in per_variant_fold[variant]],
    )
    prediction_path = output / "cca_consistent_2017_predictions.csv"
    write_csv(
        prediction_path,
        [
            "variant", "split_name", "sha256", "seed", "label",
            "probability", "threshold", "predicted_label",
        ],
        prediction_rows,
    )

    control = next(
        row for row in summaries
        if row["variant"] == "control_probability_blend"
    )
    control_target = 0.9213554444747093
    audit = {
        "schema_version": "1.0.0",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "phase": "P35_2017_ROLLING_CCA_CONSISTENT_DUAL_VIEW_EVALUATION",
        "formal_training_enabled": False,
        "protocol_status": "DEVELOPMENT_ONLY_NOT_OUTER_TESTED",
        "strict_future_year_isolation": True,
        "observed_years": sorted(observed_years),
        "fixed_threshold": THRESHOLD,
        "threshold_tuning_performed": False,
        "entry_view_weight": ENTRY_WEIGHT,
        "endpoint_view_weight": ENDPOINT_WEIGHT,
        "cca_dimensions": dimensions,
        "cca_regularization": args.cca_regularization,
        "canonical_audit": canonical_audit,
        "classifier": {
            "type": "custom_l2_logistic",
            "seeds": args.seeds,
            "iterations": args.iterations,
            "learning_rate": args.learning_rate,
            "l2": args.l2,
        },
        "selection_metric": "mean_balanced_accuracy_minus_fold_std",
        "ranked_variants": summaries,
        "selected_development_candidate": summaries[0]["variant"],
        "control_reproduction": {
            "observed_mean_balanced_accuracy": control["mean_balanced_accuracy"],
            "target_mean_balanced_accuracy": control_target,
            "absolute_difference": abs(
                control["mean_balanced_accuracy"] - control_target
            ),
        },
        "inputs": {
            name: {"path": str(path), "sha256": sha256_file(path)}
            for name, path in paths.items()
        },
        "outputs": {
            "summary": str(summary_path),
            "per_fold": str(fold_path),
            "predictions": str(prediction_path),
        },
        "blocking_reasons": [
            "cca_consistency_candidate_not_locked",
            "nonlinear_contrastive_encoder_not_implemented",
            "formal_dataset_not_evaluated",
        ],
        "notes": [
            "CCA projections are fitted without labels on each training partition only.",
            "Classification weights and all normalization are training-fold local.",
            "The control exactly reproduces the current two-expert probability blend.",
            "This linear CCA stage is an MVP test of cross-view consistency, not the final nonlinear encoder.",
            "No APK, raw graph, method signature, or future-year row is read.",
        ],
    }
    audit_path = output / "cca_consistent_2017_audit.json"
    audit_path.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print("rank variant mean_BA min_BA std_BA stability AUC", flush=True)
    for row in summaries:
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
