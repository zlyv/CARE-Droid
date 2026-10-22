#!/usr/bin/env python3
"""Cross-fitted uncertainty-gated dual-view evaluation on 2017 only.

Two independent experts are trained: the established entry+compact expert and
an endpoint+compact expert.  A small gate is fitted exclusively from grouped
out-of-fold predictions of the current training partition.  It learns when
the endpoint expert is more reliable using view confidence, disagreement,
entry coverage, endpoint reachability, and code-provenance quality.  The
future validation partition is never used to fit either expert or the gate.

The final malware threshold is hard-coded to 0.5.
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


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from run_dual_view_2017_rolling_eval import (  # noqa: E402
    ACTIVE_RELIABILITY_COMPONENTS,
    ROLE_VOCABULARY_SIZE,
    attach_audit_metadata,
    endpoint_matrix,
    entry_model_args,
    sha256_file,
)
from run_graph_pool_diagnostic import (  # noqa: E402
    compact_matrix,
    graph_matrix,
    group_folds,
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


def predict_logistic(
    x_fit: np.ndarray,
    y_fit: np.ndarray,
    x_eval: np.ndarray,
    seed: int,
    iterations: int,
    learning_rate: float,
    l2: float,
) -> np.ndarray:
    weights, bias = fit_logistic(
        x_fit,
        y_fit,
        seed,
        iterations=iterations,
        learning_rate=learning_rate,
        l2=l2,
    )
    return 1.0 / (
        1.0 + np.exp(-np.clip(x_eval @ weights + bias, -40.0, 40.0))
    )


def expert_matrices(
    fit_rows: list[dict],
    eval_rows: list[dict],
    entry_cache: dict,
    endpoint_cache: dict,
    feature_by_sha: dict,
    schema: dict,
    definition: PilotReliabilityDefinition,
    entry_args,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    train_shas = {row["sha256"] for row in fit_rows}
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
    entry_fit, y_fit = graph_matrix(
        fit_rows, entry_cache, entry_evidence, entry_args, True
    )
    entry_eval, y_eval = graph_matrix(
        eval_rows, entry_cache, entry_evidence, entry_args, True
    )
    endpoint_fit, endpoint_y_fit = endpoint_matrix(
        fit_rows, endpoint_cache, endpoint_evidence
    )
    endpoint_eval, endpoint_y_eval = endpoint_matrix(
        eval_rows, endpoint_cache, endpoint_evidence
    )
    if not (
        np.array_equal(y_fit, endpoint_y_fit)
        and np.array_equal(y_eval, endpoint_y_eval)
    ):
        raise ValueError("entry/endpoint label mismatch")
    compact_fit, compact_eval, _ = compact_matrix(
        fit_rows,
        eval_rows,
        feature_by_sha,
        schema,
        feature_mode="base",
    )
    entry_fit, entry_eval = standardize_pair(entry_fit, entry_eval)
    endpoint_fit, endpoint_eval = standardize_pair(endpoint_fit, endpoint_eval)
    entry_expert_fit = np.concatenate([0.5 * entry_fit, compact_fit], axis=1)
    entry_expert_eval = np.concatenate([0.5 * entry_eval, compact_eval], axis=1)
    endpoint_expert_fit = np.concatenate([endpoint_fit, compact_fit], axis=1)
    endpoint_expert_eval = np.concatenate([endpoint_eval, compact_eval], axis=1)
    return (
        entry_expert_fit,
        entry_expert_eval,
        endpoint_expert_fit,
        endpoint_expert_eval,
        y_fit,
        y_eval,
    )


def _safe_ratio(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator > 0.0 else 0.0


def view_quality_features(
    rows: list[dict],
    entry_cache: dict,
    endpoint_cache: dict,
    entry_probability: np.ndarray,
    endpoint_probability: np.ndarray,
) -> np.ndarray:
    """Build compact label-free quality plus prediction-uncertainty features."""

    if len(rows) != len(entry_probability) or len(rows) != len(endpoint_probability):
        raise ValueError("quality rows and expert probabilities must align")
    values = []
    for row, entry_score, endpoint_score in zip(
        rows, entry_probability, endpoint_probability
    ):
        entry_payload = entry_cache[row["sha256"]]["payload"]
        endpoint_payload = endpoint_cache[row["sha256"]]["payload"]
        entry_role_ids = list(entry_payload.get("node_role_ids") or [])
        entry_sensitive_count = sum(bool(role_ids) for role_ids in entry_role_ids)
        endpoint_base = np.asarray(
            endpoint_payload.get("node_base_features") or [], dtype=np.float64
        )
        if len(endpoint_base) == 0:
            endpoint_base = np.zeros((0, 15), dtype=np.float64)
        if endpoint_base.ndim != 2 or endpoint_base.shape[1] != 15:
            raise ValueError("endpoint quality payload must have 15 base features")

        seed_mask = endpoint_base[:, 4] > 0.5
        endpoint_count = int(seed_mask.sum())
        reachable_ratio = (
            float(endpoint_base[seed_mask, 8].mean()) if endpoint_count else 0.0
        )
        unreachable_ratio = (
            float(endpoint_base[seed_mask, 9].mean()) if endpoint_count else 0.0
        )
        unknown_ratio = (
            float(endpoint_base[seed_mask, 10].mean()) if endpoint_count else 1.0
        )
        upstream_context = (endpoint_base[:, 12] > 0.5) & (~seed_mask)
        context_count = int(upstream_context.sum())
        package_ratio = (
            float(endpoint_base[upstream_context, 5].mean())
            if context_count else 0.0
        )
        bundled_ratio = (
            float(endpoint_base[upstream_context, 6].mean())
            if context_count else 0.0
        )
        external_ratio = (
            float(endpoint_base[upstream_context, 7].mean())
            if context_count else 0.0
        )
        entry_coverage = min(
            1.0, _safe_ratio(entry_sensitive_count, endpoint_count)
        )
        entry_confidence = abs(float(entry_score) - 0.5) * 2.0
        endpoint_confidence = abs(float(endpoint_score) - 0.5) * 2.0
        values.append([
            1.0 - entry_coverage,
            reachable_ratio,
            unreachable_ratio,
            unknown_ratio,
            package_ratio,
            bundled_ratio,
            external_ratio,
            float(np.log1p(endpoint_count)),
            float(np.log1p(context_count)),
            entry_confidence,
            endpoint_confidence,
            abs(float(endpoint_score) - float(entry_score)),
            float((entry_score >= THRESHOLD) != (endpoint_score >= THRESHOLD)),
            endpoint_confidence - entry_confidence,
        ])
    return np.asarray(values, dtype=np.float64)


def cross_fitted_expert_predictions(
    train_rows: list[dict],
    entry_cache: dict,
    endpoint_cache: dict,
    feature_by_sha: dict,
    schema: dict,
    definition: PilotReliabilityDefinition,
    entry_args,
    args,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    folds = group_folds(train_rows, args.gate_folds)
    entry_oof = np.full(len(train_rows), np.nan, dtype=np.float64)
    endpoint_oof = np.full(len(train_rows), np.nan, dtype=np.float64)
    labels = np.full(len(train_rows), -1, dtype=np.int64)
    for fold_index, validation_indices in enumerate(folds):
        validation_set = set(validation_indices)
        fit_rows = [
            row for index, row in enumerate(train_rows)
            if index not in validation_set
        ]
        gate_rows = [train_rows[index] for index in validation_indices]
        matrices = expert_matrices(
            fit_rows,
            gate_rows,
            entry_cache,
            endpoint_cache,
            feature_by_sha,
            schema,
            definition,
            entry_args,
        )
        entry_fit, entry_gate, endpoint_fit, endpoint_gate, y_fit, y_gate = matrices
        seed = args.gate_seed + fold_index
        entry_oof[validation_indices] = predict_logistic(
            entry_fit, y_fit, entry_gate, seed,
            args.iterations, args.learning_rate, args.l2,
        )
        endpoint_oof[validation_indices] = predict_logistic(
            endpoint_fit, y_fit, endpoint_gate, seed,
            args.iterations, args.learning_rate, args.l2,
        )
        labels[validation_indices] = y_gate
    if np.any(~np.isfinite(entry_oof)) or np.any(~np.isfinite(endpoint_oof)):
        raise ValueError("cross-fitted expert prediction is incomplete")
    if np.any(labels < 0):
        raise ValueError("cross-fitted labels are incomplete")
    return entry_oof, endpoint_oof, labels


def fit_gate(
    train_rows: list[dict],
    entry_cache: dict,
    endpoint_cache: dict,
    entry_oof: np.ndarray,
    endpoint_oof: np.ndarray,
    labels: np.ndarray,
    args,
) -> dict:
    gate_x = view_quality_features(
        train_rows, entry_cache, endpoint_cache, entry_oof, endpoint_oof
    )
    endpoint_error = np.abs(labels - endpoint_oof)
    entry_error = np.abs(labels - entry_oof)
    gate_y = (endpoint_error < entry_error).astype(np.int64)
    mean = gate_x.mean(axis=0)
    scale = gate_x.std(axis=0)
    scale[scale < 1e-12] = 1.0
    standardized = (gate_x - mean) / scale
    if np.unique(gate_y).size == 1:
        constant = float((gate_y.sum() + 1.0) / (len(gate_y) + 2.0))
        weights = np.zeros(gate_x.shape[1], dtype=np.float64)
        bias = float(np.log(constant / (1.0 - constant)))
    else:
        weights, bias = fit_logistic(
            standardized,
            gate_y,
            args.gate_seed,
            iterations=args.gate_iterations,
            learning_rate=args.gate_learning_rate,
            l2=args.gate_l2,
        )
    train_gate = 1.0 / (
        1.0 + np.exp(-np.clip(standardized @ weights + bias, -40.0, 40.0))
    )
    return {
        "weights": weights,
        "bias": float(bias),
        "mean": mean,
        "scale": scale,
        "endpoint_better_rate": float(gate_y.mean()),
        "mean_training_gate": float(train_gate.mean()),
        "gate_target_accuracy": float(np.mean((train_gate >= 0.5) == gate_y)),
    }


def apply_gate(
    gate: dict,
    rows: list[dict],
    entry_cache: dict,
    endpoint_cache: dict,
    entry_probability: np.ndarray,
    endpoint_probability: np.ndarray,
) -> np.ndarray:
    values = view_quality_features(
        rows, entry_cache, endpoint_cache, entry_probability, endpoint_probability
    )
    values = (values - gate["mean"]) / gate["scale"]
    return 1.0 / (
        1.0 + np.exp(
            -np.clip(values @ gate["weights"] + gate["bias"], -40.0, 40.0)
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--entry-cache", required=True, type=Path)
    parser.add_argument("--endpoint-cache", required=True, type=Path)
    parser.add_argument("--features", required=True, type=Path)
    parser.add_argument("--schema", required=True, type=Path)
    parser.add_argument("--splits", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--gate-strengths", nargs="+", type=float, default=[0.25, 0.5, 1.0])
    parser.add_argument("--gate-folds", type=int, default=5)
    parser.add_argument("--gate-seed", type=int, default=20260811)
    parser.add_argument("--iterations", type=int, default=1500)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--l2", type=float, default=0.03)
    parser.add_argument("--gate-iterations", type=int, default=1500)
    parser.add_argument("--gate-learning-rate", type=float, default=0.05)
    parser.add_argument("--gate-l2", type=float, default=0.10)
    args = parser.parse_args()
    if any(not (0.0 < value <= 1.0) for value in args.gate_strengths):
        raise SystemExit("gate strengths must be in (0, 1]")

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
    missing_counts = {
        "entry_cache": len(selected_shas - set(entry_cache)),
        "endpoint_cache": len(selected_shas - set(endpoint_cache)),
        "features": len(selected_shas - set(feature_by_sha)),
    }
    if any(missing_counts.values()):
        raise SystemExit(f"missing selected rows: {missing_counts}")
    years = {
        str(feature_by_sha[sha].get("metadata_year", "")) for sha in selected_shas
    }
    if years != {"2017"}:
        raise SystemExit(f"P27 accepts only 2017 development rows; observed={sorted(years)}")
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
    variant_names = ["entry_control", "endpoint_expert", "fixed_blend_025"] + [
        f"uncertainty_gate_strength_{value:g}" for value in args.gate_strengths
    ]
    split_results = {name: [] for name in variant_names}
    prediction_rows = []
    gate_audits = []

    for split_name in sorted(split_rows):
        train_rows = split_rows[split_name]["train"]
        validation_rows = split_rows[split_name]["test"]
        entry_oof, endpoint_oof, oof_y = cross_fitted_expert_predictions(
            train_rows,
            entry_cache,
            endpoint_cache,
            feature_by_sha,
            schema,
            definition,
            entry_args,
            args,
        )
        gate = fit_gate(
            train_rows,
            entry_cache,
            endpoint_cache,
            entry_oof,
            endpoint_oof,
            oof_y,
            args,
        )
        gate_audits.append({
            "split_name": split_name,
            "training_row_count": len(train_rows),
            "endpoint_better_oof_rate": gate["endpoint_better_rate"],
            "mean_training_gate": gate["mean_training_gate"],
            "gate_target_training_accuracy": gate["gate_target_accuracy"],
            "validation_labels_used_for_gate": False,
        })

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
        entry_train, entry_validation, endpoint_train, endpoint_validation, y_train, y_validation = matrices
        per_variant_seed_metrics = {name: [] for name in variant_names}
        per_variant_scores = {name: [] for name in variant_names}
        per_seed_gate_means = []
        for seed in args.seeds:
            entry_probability = predict_logistic(
                entry_train, y_train, entry_validation, seed,
                args.iterations, args.learning_rate, args.l2,
            )
            endpoint_probability = predict_logistic(
                endpoint_train, y_train, endpoint_validation, seed,
                args.iterations, args.learning_rate, args.l2,
            )
            gate_probability = apply_gate(
                gate,
                validation_rows,
                entry_cache,
                endpoint_cache,
                entry_probability,
                endpoint_probability,
            )
            per_seed_gate_means.append(float(gate_probability.mean()))
            scores_by_variant = {
                "entry_control": entry_probability,
                "endpoint_expert": endpoint_probability,
                "fixed_blend_025": 0.75 * entry_probability + 0.25 * endpoint_probability,
            }
            for strength in args.gate_strengths:
                effective_gate = strength * gate_probability
                scores_by_variant[f"uncertainty_gate_strength_{strength:g}"] = (
                    (1.0 - effective_gate) * entry_probability
                    + effective_gate * endpoint_probability
                )
            for variant, scores in scores_by_variant.items():
                per_variant_seed_metrics[variant].append(
                    metrics(y_validation, scores, threshold=THRESHOLD)
                )
                per_variant_scores[variant].append(scores)
                for row, label, score, gate_value in zip(
                    validation_rows, y_validation, scores, gate_probability
                ):
                    prediction_rows.append({
                        "variant": variant,
                        "split_name": split_name,
                        "sha256": row["sha256"],
                        "seed": seed,
                        "label": int(label),
                        "probability": float(score),
                        "gate_probability": float(gate_value),
                        "threshold": THRESHOLD,
                        "predicted_label": int(score >= THRESHOLD),
                    })
        gate_audits[-1]["mean_validation_gate"] = float(np.mean(per_seed_gate_means))
        for variant in variant_names:
            row = {
                "split_name": split_name,
                "train_count": len(train_rows),
                "validation_count": len(validation_rows),
                "threshold": THRESHOLD,
            }
            for metric_name in (
                "accuracy", "balanced_accuracy", "precision", "recall", "f1", "auc"
            ):
                row[f"{metric_name}_mean"] = float(np.mean([
                    result[metric_name]
                    for result in per_variant_seed_metrics[variant]
                ]))
            split_results[variant].append(row)
        print(f"completed {split_name}", flush=True)

    ranked = []
    for variant, rows in split_results.items():
        bas = np.asarray([row["balanced_accuracy_mean"] for row in rows])
        ranked.append({
            "variant": variant,
            "mean_accuracy": float(np.mean([row["accuracy_mean"] for row in rows])),
            "mean_balanced_accuracy": float(bas.mean()),
            "minimum_fold_balanced_accuracy": float(bas.min()),
            "std_balanced_accuracy": float(bas.std()),
            "stability_score": float(bas.mean() - bas.std()),
            "mean_auc": float(np.mean([row["auc_mean"] for row in rows])),
            "split_summaries": rows,
        })
    ranked.sort(key=lambda row: (
        -row["stability_score"], -row["mean_balanced_accuracy"], row["variant"]
    ))
    for rank, row in enumerate(ranked, 1):
        row["rank"] = rank

    summary_path = output / "uncertainty_gated_2017_summary.csv"
    summary_fields = [
        "rank", "variant", "mean_accuracy", "mean_balanced_accuracy",
        "minimum_fold_balanced_accuracy", "std_balanced_accuracy",
        "stability_score", "mean_auc",
    ]
    with summary_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=summary_fields)
        writer.writeheader()
        for row in ranked:
            writer.writerow({name: row.get(name, "") for name in summary_fields})
    prediction_path = output / "uncertainty_gated_2017_predictions.csv"
    prediction_fields = [
        "variant", "split_name", "sha256", "seed", "label", "probability",
        "gate_probability", "threshold", "predicted_label",
    ]
    with prediction_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=prediction_fields)
        writer.writeheader()
        writer.writerows(prediction_rows)

    audit = {
        "schema_version": "1.0.0",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "phase": "P27_CROSS_FITTED_UNCERTAINTY_GATED_DUAL_VIEW_2017_EVALUATION",
        "formal_training_enabled": False,
        "protocol_status": "DEVELOPMENT_ONLY_NOT_OUTER_TESTED",
        "strict_future_year_isolation": True,
        "observed_years": sorted(years),
        "fixed_threshold": THRESHOLD,
        "threshold_tuning_performed": False,
        "entry_cache": str(paths["entry_cache"]),
        "entry_cache_sha256": sha256_file(paths["entry_cache"]),
        "endpoint_cache": str(paths["endpoint_cache"]),
        "endpoint_cache_sha256": sha256_file(paths["endpoint_cache"]),
        "features": str(paths["features"]),
        "features_sha256": sha256_file(paths["features"]),
        "split_manifest": str(paths["splits"]),
        "split_manifest_sha256": sha256_file(paths["splits"]),
        "expert_classifier": {
            "type": "custom_l2_logistic",
            "seeds": args.seeds,
            "iterations": args.iterations,
            "learning_rate": args.learning_rate,
            "l2": args.l2,
        },
        "gate": {
            "training_mode": "grouped_cross_fitted_training_partition_only",
            "folds": args.gate_folds,
            "seed": args.gate_seed,
            "l2": args.gate_l2,
            "strengths_compared": args.gate_strengths,
            "quality_feature_count": 14,
            "validation_labels_used": False,
            "split_audits": gate_audits,
        },
        "candidate_selection_metric": "mean_balanced_accuracy_minus_fold_std",
        "ranked_variants": ranked,
        "selected_development_candidate": ranked[0]["variant"],
        "outputs": {
            "summary": str(summary_path),
            "predictions": str(prediction_path),
        },
        "blocking_reasons": [
            "gate_candidate_not_locked",
            "role_conditioned_multihop_motif_not_implemented",
            "formal_dataset_not_evaluated",
            "legacy_future_years_must_not_drive_candidate_selection",
        ],
        "notes": [
            "Only chronological 2017 development rows are accepted.",
            "Gate targets and gate weights use grouped OOF predictions from the current training partition only.",
            "Future validation labels are not used for expert fitting, gate fitting, or normalization.",
            "The final classification threshold is fixed at 0.5 for every variant.",
        ],
    }
    audit_path = output / "uncertainty_gated_2017_audit.json"
    audit_path.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
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
