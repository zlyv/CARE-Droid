#!/usr/bin/env python3
"""Cross-fitted calibration of the endpoint-motif expert on 2017 only.

P29 showed improved ranking but worse fixed-threshold classification.  This
script tests whether that gap is probability calibration: within each outer
2017 development training partition, grouped out-of-fold motif predictions
fit a one-dimensional Platt calibrator.  The outer validation partition is
never used for model fitting, calibration, or selection.  The final malware
threshold remains hard-coded to 0.5.
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
    attach_audit_metadata,
    entry_model_args,
    sha256_file,
)
from run_endpoint_motif_2017_eval import endpoint_motif_matrix  # noqa: E402
from run_graph_pool_diagnostic import (  # noqa: E402
    compact_matrix,
    group_folds,
    label_of,
    read_csv_rows,
    read_split_rows,
    standardize_pair,
)
from run_pilot_compact_eval import fit_logistic, metrics  # noqa: E402
from run_uncertainty_gated_dual_view_2017_eval import (  # noqa: E402
    expert_matrices,
    predict_logistic,
)
from tasa_droid.p3.reliability import (  # noqa: E402
    PilotReliabilityDefinition,
    fit_training_fold_components,
    read_cache,
)


THRESHOLD = 0.5
MOTIF_L2 = 0.1
ENDPOINT_RESIDUAL_WEIGHT = 0.25
CALIBRATION_L2 = 0.01


def motif_expert_matrices(
    fit_rows: list[dict],
    eval_rows: list[dict],
    endpoint_cache: dict,
    feature_by_sha: dict,
    schema: dict,
    definition: PilotReliabilityDefinition,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    train_shas = {row["sha256"] for row in fit_rows}
    evidence = fit_training_fold_components(
        endpoint_cache,
        train_shas,
        definition,
        active_components=ACTIVE_RELIABILITY_COMPONENTS,
    )
    motif_fit = endpoint_motif_matrix(fit_rows, endpoint_cache, evidence)
    motif_eval = endpoint_motif_matrix(eval_rows, endpoint_cache, evidence)
    motif_fit, motif_eval = standardize_pair(motif_fit, motif_eval)
    compact_fit, compact_eval, _ = compact_matrix(
        fit_rows,
        eval_rows,
        feature_by_sha,
        schema,
        feature_mode="base",
    )
    x_fit = np.concatenate([motif_fit, compact_fit], axis=1)
    x_eval = np.concatenate([motif_eval, compact_eval], axis=1)
    y_fit = np.asarray([label_of(row) for row in fit_rows], dtype=np.int64)
    y_eval = np.asarray([label_of(row) for row in eval_rows], dtype=np.int64)
    return x_fit, x_eval, y_fit, y_eval


def _logit(probability: np.ndarray) -> np.ndarray:
    value = np.clip(np.asarray(probability, dtype=np.float64), 1e-6, 1.0 - 1e-6)
    return np.log(value / (1.0 - value))


def fit_platt_calibrator(
    probability: np.ndarray,
    labels: np.ndarray,
    seed: int,
    iterations: int,
    learning_rate: float,
) -> dict:
    logits = _logit(probability)
    mean = float(logits.mean())
    scale = float(logits.std())
    if scale < 1e-12:
        scale = 1.0
    x = ((logits - mean) / scale).reshape(-1, 1)
    weights, bias = fit_logistic(
        x,
        labels,
        seed,
        iterations=iterations,
        learning_rate=learning_rate,
        l2=CALIBRATION_L2,
    )
    return {
        "weight": float(weights[0]),
        "bias": float(bias),
        "logit_mean": mean,
        "logit_scale": scale,
    }


def apply_platt(calibrator: dict, probability: np.ndarray) -> np.ndarray:
    standardized = (
        (_logit(probability) - calibrator["logit_mean"])
        / calibrator["logit_scale"]
    )
    linear = calibrator["weight"] * standardized + calibrator["bias"]
    return 1.0 / (1.0 + np.exp(-np.clip(linear, -40.0, 40.0)))


def cross_fitted_motif_predictions(
    rows: list[dict],
    endpoint_cache: dict,
    feature_by_sha: dict,
    schema: dict,
    definition: PilotReliabilityDefinition,
    seed: int,
    folds: int,
    iterations: int,
    learning_rate: float,
) -> tuple[np.ndarray, np.ndarray]:
    fold_indices = group_folds(rows, folds)
    oof = np.full(len(rows), np.nan, dtype=np.float64)
    labels = np.full(len(rows), -1, dtype=np.int64)
    for fold_index, held_indices in enumerate(fold_indices):
        held_set = set(held_indices)
        fit_rows = [row for index, row in enumerate(rows) if index not in held_set]
        held_rows = [rows[index] for index in held_indices]
        x_fit, x_held, y_fit, y_held = motif_expert_matrices(
            fit_rows,
            held_rows,
            endpoint_cache,
            feature_by_sha,
            schema,
            definition,
        )
        oof[held_indices] = predict_logistic(
            x_fit,
            y_fit,
            x_held,
            seed + fold_index * 1009,
            iterations,
            learning_rate,
            MOTIF_L2,
        )
        labels[held_indices] = y_held
    if np.any(~np.isfinite(oof)) or np.any(labels < 0):
        raise ValueError("incomplete grouped OOF motif predictions")
    return oof, labels


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
    parser.add_argument("--calibration-folds", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=1500)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--baseline-l2", type=float, default=0.03)
    parser.add_argument("--calibration-iterations", type=int, default=1500)
    parser.add_argument("--calibration-learning-rate", type=float, default=0.05)
    args = parser.parse_args()
    if args.calibration_folds < 2:
        raise SystemExit("calibration folds must be at least 2")

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
        for partition in ("train", "test") for row in partitions[partition]
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
        str(feature_by_sha[sha].get("metadata_year", "")) for sha in selected_shas
    }
    if years != {"2017"}:
        raise SystemExit(f"P30 accepts only 2017 development rows; observed={sorted(years)}")
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
        "fixed_residual_blend_025_control",
        "motif_raw_expert_l2_0.1",
        "motif_platt_expert_l2_0.1",
        "motif_raw_residual_blend_025",
        "motif_platt_residual_blend_025",
    )
    split_results = {variant: [] for variant in variants}
    prediction_rows = []
    calibration_audits = []

    for split_name in sorted(split_rows):
        train_rows = split_rows[split_name]["train"]
        validation_rows = split_rows[split_name]["test"]
        matrices = expert_matrices(
            train_rows,
            validation_rows,
            entry_cache,
            endpoint_cache,
            feature_by_sha,
            schema,
            definition,
            entry_args,
        )
        (
            entry_train,
            entry_validation,
            endpoint_train,
            endpoint_validation,
            y_train,
            y_validation,
        ) = matrices
        motif_train, motif_validation, motif_y_train, motif_y_validation = (
            motif_expert_matrices(
                train_rows,
                validation_rows,
                endpoint_cache,
                feature_by_sha,
                schema,
                definition,
            )
        )
        if not (
            np.array_equal(y_train, motif_y_train)
            and np.array_equal(y_validation, motif_y_validation)
        ):
            raise SystemExit(f"label mismatch in {split_name}")

        seed_metrics = {variant: [] for variant in variants}
        for seed in args.seeds:
            oof_probability, oof_labels = cross_fitted_motif_predictions(
                train_rows,
                endpoint_cache,
                feature_by_sha,
                schema,
                definition,
                seed,
                args.calibration_folds,
                args.iterations,
                args.learning_rate,
            )
            calibrator = fit_platt_calibrator(
                oof_probability,
                oof_labels,
                seed + 1000003,
                args.calibration_iterations,
                args.calibration_learning_rate,
            )
            calibrated_oof = apply_platt(calibrator, oof_probability)
            calibration_audits.append({
                "split_name": split_name,
                "seed": seed,
                "training_oof_row_count": len(oof_labels),
                "platt_weight": calibrator["weight"],
                "platt_bias": calibrator["bias"],
                "raw_oof_brier": float(np.mean((oof_probability - oof_labels) ** 2)),
                "calibrated_oof_brier": float(np.mean((calibrated_oof - oof_labels) ** 2)),
                "outer_validation_labels_used": False,
            })

            entry_probability = predict_logistic(
                entry_train,
                y_train,
                entry_validation,
                seed,
                args.iterations,
                args.learning_rate,
                args.baseline_l2,
            )
            endpoint_probability = predict_logistic(
                endpoint_train,
                y_train,
                endpoint_validation,
                seed,
                args.iterations,
                args.learning_rate,
                args.baseline_l2,
            )
            motif_raw = predict_logistic(
                motif_train,
                y_train,
                motif_validation,
                seed,
                args.iterations,
                args.learning_rate,
                MOTIF_L2,
            )
            motif_calibrated = apply_platt(calibrator, motif_raw)
            scores = {
                "fixed_residual_blend_025_control": (
                    0.75 * entry_probability + 0.25 * endpoint_probability
                ),
                "motif_raw_expert_l2_0.1": motif_raw,
                "motif_platt_expert_l2_0.1": motif_calibrated,
                "motif_raw_residual_blend_025": (
                    0.75 * entry_probability + 0.25 * motif_raw
                ),
                "motif_platt_residual_blend_025": (
                    0.75 * entry_probability + 0.25 * motif_calibrated
                ),
            }
            for variant, probability in scores.items():
                seed_metrics[variant].append(
                    metrics(y_validation, probability, threshold=THRESHOLD)
                )
                for row, label, value in zip(
                    validation_rows, y_validation, probability
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
            "minimum_fold_balanced_accuracy": float(ba.min()),
            "std_balanced_accuracy": float(ba.std()),
            "stability_score": float(ba.mean() - ba.std()),
            "mean_auc": float(np.mean([row["auc_mean"] for row in rows])),
            "split_summaries": rows,
        })
    ranked.sort(key=lambda row: (
        -row["stability_score"], -row["mean_balanced_accuracy"], row["variant"]
    ))
    for rank, row in enumerate(ranked, 1):
        row["rank"] = rank

    summary_path = output / "endpoint_motif_calibrated_2017_summary.csv"
    summary_fields = [
        "rank", "variant", "mean_accuracy", "mean_balanced_accuracy",
        "minimum_fold_balanced_accuracy", "std_balanced_accuracy",
        "stability_score", "mean_auc",
    ]
    write_csv(summary_path, summary_fields, ranked)
    prediction_path = output / "endpoint_motif_calibrated_2017_predictions.csv"
    write_csv(
        prediction_path,
        [
            "variant", "split_name", "sha256", "seed", "label",
            "probability", "threshold", "predicted_label",
        ],
        prediction_rows,
    )
    audit = {
        "schema_version": "1.0.0",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "phase": "P30_GROUPED_OOF_PLATT_CALIBRATED_ENDPOINT_MOTIF_2017",
        "formal_training_enabled": False,
        "protocol_status": "DEVELOPMENT_ONLY_NOT_OUTER_TESTED",
        "strict_future_year_isolation": True,
        "observed_years": sorted(years),
        "fixed_threshold": THRESHOLD,
        "threshold_tuning_performed": False,
        "motif_l2_locked_from_p29": MOTIF_L2,
        "endpoint_residual_weight": ENDPOINT_RESIDUAL_WEIGHT,
        "calibration": {
            "type": "grouped_oof_platt",
            "folds": args.calibration_folds,
            "l2": CALIBRATION_L2,
            "outer_validation_labels_used": False,
        },
        "calibration_audits": calibration_audits,
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
            "predictions": str(prediction_path),
        },
        "blocking_reasons": [
            "calibrated_motif_candidate_not_locked",
            "formal_dataset_not_evaluated",
        ],
        "notes": [
            "Only chronological 2017 development rows are accepted.",
            "Calibration folds preserve blocking groups.",
            "Every OOF model refits reliability, filtering, and normalization on its inner training rows.",
            "The classification threshold remains fixed at 0.5.",
            "No 2018--2021 row is read or evaluated.",
        ],
    }
    audit_path = output / "endpoint_motif_calibrated_2017_audit.json"
    audit_path.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print("rank variant mean_BA min_BA std_BA stability AUC", flush=True)
    for row in ranked:
        print(
            row["rank"], row["variant"],
            f'{row["mean_balanced_accuracy"]:.4f}',
            f'{row["minimum_fold_balanced_accuracy"]:.4f}',
            f'{row["std_balanced_accuracy"]:.4f}',
            f'{row["stability_score"]:.4f}',
            f'{row["mean_auc"]:.4f}',
            flush=True,
        )


if __name__ == "__main__":
    main()
