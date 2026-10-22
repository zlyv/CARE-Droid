#!/usr/bin/env python3
"""Evaluate predeclared modules of the locked dual-view classifier.

The script compares compact-only, each graph view alone, both graph views,
each graph view fused with compact features, a graph-only 0.75/0.25 expert
probability blend, and the locked compact-anchored 0.75/0.25 expert blend.
Reliability, filtering, normalization, and model fitting are always local to
the corresponding training partition. The classification threshold is fixed
at 0.5 and no APK is opened.
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
ENTRY_GRAPH_WEIGHT = 0.5
ENTRY_PROBABILITY_WEIGHT = 0.75
ENDPOINT_PROBABILITY_WEIGHT = 0.25

FIT_VARIANTS = (
    "compact_only",
    "entry_graph_only",
    "endpoint_graph_only",
    "dual_graph_concat",
    "dual_graph_plus_compact_concat",
    "entry_plus_compact",
    "endpoint_plus_compact",
)
ALL_VARIANTS = FIT_VARIANTS + (
    "graph_only_probability_blend",
    "locked_probability_blend",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def build_variant_matrices(
    entry_train: np.ndarray,
    entry_eval: np.ndarray,
    endpoint_train: np.ndarray,
    endpoint_eval: np.ndarray,
    compact_train: np.ndarray,
    compact_eval: np.ndarray,
    entry_graph_weight: float = ENTRY_GRAPH_WEIGHT,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    if len({len(entry_train), len(endpoint_train), len(compact_train)}) != 1:
        raise ValueError("training module rows do not align")
    if len({len(entry_eval), len(endpoint_eval), len(compact_eval)}) != 1:
        raise ValueError("evaluation module rows do not align")
    if entry_graph_weight < 0.0:
        raise ValueError("entry graph weight must be nonnegative")
    return {
        "compact_only": (compact_train, compact_eval),
        "entry_graph_only": (
            entry_graph_weight * entry_train,
            entry_graph_weight * entry_eval,
        ),
        "endpoint_graph_only": (endpoint_train, endpoint_eval),
        "dual_graph_concat": (
            np.concatenate([
                entry_graph_weight * entry_train, endpoint_train
            ], axis=1),
            np.concatenate([
                entry_graph_weight * entry_eval, endpoint_eval
            ], axis=1),
        ),
        "dual_graph_plus_compact_concat": (
            np.concatenate([
                entry_graph_weight * entry_train,
                endpoint_train,
                compact_train,
            ], axis=1),
            np.concatenate([
                entry_graph_weight * entry_eval,
                endpoint_eval,
                compact_eval,
            ], axis=1),
        ),
        "entry_plus_compact": (
            np.concatenate([
                entry_graph_weight * entry_train, compact_train
            ], axis=1),
            np.concatenate([
                entry_graph_weight * entry_eval, compact_eval
            ], axis=1),
        ),
        "endpoint_plus_compact": (
            np.concatenate([endpoint_train, compact_train], axis=1),
            np.concatenate([endpoint_eval, compact_eval], axis=1),
        ),
    }


def locked_probability_blend(
    entry_probability: np.ndarray,
    endpoint_probability: np.ndarray,
) -> np.ndarray:
    if entry_probability.shape != endpoint_probability.shape:
        raise ValueError("expert probabilities do not align")
    return (
        ENTRY_PROBABILITY_WEIGHT * entry_probability
        + ENDPOINT_PROBABILITY_WEIGHT * endpoint_probability
    )


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
    parser.add_argument(
        "--entry-graph-weight",
        type=float,
        default=ENTRY_GRAPH_WEIGHT,
        help="entry-graph feature scale selected on 2017 rolling validation",
    )
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=ALL_VARIANTS,
        default=list(ALL_VARIANTS),
        help="report only these predeclared variants; blend dependencies are fitted automatically",
    )
    args = parser.parse_args()
    if not args.seeds:
        raise SystemExit("at least one seed is required")
    if args.entry_graph_weight < 0.0:
        raise SystemExit("entry graph weight must be nonnegative")

    selected_variants = list(dict.fromkeys(args.variants))
    fit_variants = {
        variant for variant in selected_variants if variant in FIT_VARIANTS
    }
    if "locked_probability_blend" in selected_variants:
        fit_variants.update({"entry_plus_compact", "endpoint_plus_compact"})
    if "graph_only_probability_blend" in selected_variants:
        fit_variants.update({"entry_graph_only", "endpoint_graph_only"})

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
    observed_years = sorted({
        str(feature_by_sha[sha].get("metadata_year", ""))
        for sha in selected_shas
    })
    development_only = observed_years == ["2017"]

    definition = PilotReliabilityDefinition(
        alpha_prior=1.0,
        kappa_support=20.0,
        kappa_context=20.0,
        r_min=0.10,
        r_max=0.95,
    )
    entry_args = entry_model_args(definition)
    per_variant_split = {name: [] for name in selected_variants}
    prediction_rows = []
    module_dimensions = []

    for split_name in sorted(split_rows):
        train_rows = split_rows[split_name]["train"]
        test_rows = split_rows[split_name]["test"]
        if not train_rows or not test_rows:
            raise SystemExit(f"empty partition: {split_name}")
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
        entry_test, y_test = graph_matrix(
            test_rows, entry_cache, entry_evidence, entry_args, True
        )
        endpoint_train, endpoint_y_train = endpoint_matrix(
            train_rows, endpoint_cache, endpoint_evidence
        )
        endpoint_test, endpoint_y_test = endpoint_matrix(
            test_rows, endpoint_cache, endpoint_evidence
        )
        if not (
            np.array_equal(y_train, endpoint_y_train)
            and np.array_equal(y_test, endpoint_y_test)
        ):
            raise SystemExit(f"cache label mismatch: {split_name}")
        compact_train, compact_test, compact_names = compact_matrix_named(
            train_rows,
            test_rows,
            feature_by_sha,
            schema,
        )
        entry_train, entry_test = standardize_pair(entry_train, entry_test)
        endpoint_train, endpoint_test = standardize_pair(
            endpoint_train, endpoint_test
        )
        matrices = build_variant_matrices(
            entry_train,
            entry_test,
            endpoint_train,
            endpoint_test,
            compact_train,
            compact_test,
            args.entry_graph_weight,
        )
        module_dimensions.append({
            "split_name": split_name,
            "entry_graph_feature_count": int(entry_train.shape[1]),
            "endpoint_graph_feature_count": int(endpoint_train.shape[1]),
            "compact_feature_count": len(compact_names),
            "variant_feature_counts": {
                name: int(values[0].shape[1])
                for name, values in matrices.items()
            },
        })

        seed_metrics = {name: [] for name in selected_variants}
        for seed in args.seeds:
            probabilities = {}
            for variant in FIT_VARIANTS:
                if variant not in fit_variants:
                    continue
                x_train, x_test = matrices[variant]
                probabilities[variant] = predict_logistic(
                    x_train,
                    y_train,
                    x_test,
                    seed,
                    args.iterations,
                    args.learning_rate,
                    args.l2,
                )
            if "locked_probability_blend" in selected_variants:
                probabilities["locked_probability_blend"] = (
                    locked_probability_blend(
                        probabilities["entry_plus_compact"],
                        probabilities["endpoint_plus_compact"],
                    )
                )
            if "graph_only_probability_blend" in selected_variants:
                probabilities["graph_only_probability_blend"] = (
                    locked_probability_blend(
                        probabilities["entry_graph_only"],
                        probabilities["endpoint_graph_only"],
                    )
                )
            for variant, probability in probabilities.items():
                if variant not in selected_variants:
                    continue
                seed_metrics[variant].append(
                    metrics(y_test, probability, threshold=THRESHOLD)
                )
                for row, label, score in zip(test_rows, y_test, probability):
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

        for variant in selected_variants:
            if variant in {
                "graph_only_probability_blend",
                "locked_probability_blend",
            }:
                feature_count = "two_expert_probability_blend"
            else:
                feature_count = int(matrices[variant][0].shape[1])
            summary = {
                "variant": variant,
                "split_name": split_name,
                "train_count": len(train_rows),
                "test_count": len(test_rows),
                "feature_count": feature_count,
                "threshold": THRESHOLD,
            }
            for metric_name in (
                "accuracy", "balanced_accuracy", "precision", "recall", "f1", "auc"
            ):
                summary[f"{metric_name}_mean"] = float(np.mean([
                    item[metric_name] for item in seed_metrics[variant]
                ]))
            per_variant_split[variant].append(summary)
        print(f"completed {split_name}", flush=True)

    aggregate_rows = []
    for variant in selected_variants:
        rows = per_variant_split[variant]
        ba = np.asarray([row["balanced_accuracy_mean"] for row in rows])
        aggregate_rows.append({
            "variant": variant,
            "mean_accuracy": float(np.mean([row["accuracy_mean"] for row in rows])),
            "mean_balanced_accuracy": float(ba.mean()),
            "minimum_split_balanced_accuracy": float(ba.min()),
            "std_balanced_accuracy": float(ba.std()),
            "stability_score": float(ba.mean() - ba.std()),
            "mean_auc": float(np.mean([row["auc_mean"] for row in rows])),
        })

    summary_path = output / "dual_view_module_ablation_summary.csv"
    summary_fields = [
        "variant", "mean_accuracy", "mean_balanced_accuracy",
        "minimum_split_balanced_accuracy", "std_balanced_accuracy",
        "stability_score", "mean_auc",
    ]
    write_csv(summary_path, summary_fields, aggregate_rows)
    split_path = output / "dual_view_module_ablation_per_split.csv"
    split_fields = [
        "variant", "split_name", "train_count", "test_count",
        "feature_count", "threshold", "accuracy_mean",
        "balanced_accuracy_mean", "precision_mean", "recall_mean",
        "f1_mean", "auc_mean",
    ]
    write_csv(
        split_path,
        split_fields,
        [row for variant in selected_variants for row in per_variant_split[variant]],
    )
    prediction_path = output / "dual_view_module_ablation_predictions.csv"
    write_csv(
        prediction_path,
        [
            "variant", "split_name", "sha256", "seed", "label",
            "probability", "threshold", "predicted_label",
        ],
        prediction_rows,
    )

    locked = next((
        row for row in aggregate_rows
        if row["variant"] == "locked_probability_blend"
    ), None)
    expected = (
        0.9213554444747093
        if development_only and np.isclose(args.entry_graph_weight, 0.5)
        else None
    )
    audit = {
        "schema_version": "1.0.0",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "phase": "P36_LOCKED_DUAL_VIEW_MODULE_ABLATION",
        "formal_training_enabled": False,
        "protocol_status": (
            "DEVELOPMENT_2017_ONLY" if development_only
            else "RETROSPECTIVE_LOCKED_ABLATION_NOT_FOR_SELECTION"
        ),
        "strict_test_isolation": True,
        "observed_years": observed_years,
        "fixed_threshold": THRESHOLD,
        "threshold_tuning_performed": False,
        "entry_graph_weight_within_expert": args.entry_graph_weight,
        "entry_probability_weight": ENTRY_PROBABILITY_WEIGHT,
        "endpoint_probability_weight": ENDPOINT_PROBABILITY_WEIGHT,
        "available_variants": list(ALL_VARIANTS),
        "reported_variants": selected_variants,
        "module_dimensions": module_dimensions,
        "classifier": {
            "type": "custom_l2_logistic",
            "seeds": args.seeds,
            "iterations": args.iterations,
            "learning_rate": args.learning_rate,
            "l2": args.l2,
        },
        "aggregate_results": aggregate_rows,
        "locked_control_reproduction": {
            "observed_mean_balanced_accuracy": (
                locked["mean_balanced_accuracy"] if locked else None
            ),
            "expected_2017_development_mean_balanced_accuracy": expected,
            "absolute_difference": (
                abs(locked["mean_balanced_accuracy"] - expected)
                if expected is not None and locked is not None else None
            ),
        },
        "inputs": {
            name: {"path": str(path), "sha256": sha256_file(path)}
            for name, path in paths.items()
        },
        "outputs": {
            "summary": str(summary_path),
            "per_split": str(split_path),
            "predictions": str(prediction_path),
        },
        "blocking_reasons": [
            "legacy_future_years_are_retrospective_if_present",
            "formal_dataset_not_evaluated",
        ],
        "notes": [
            "Ablation variants are predeclared and do not tune the locked protocol.",
            "All reliability, filtering, normalization, and fitting use training rows only.",
            "The graph-only blend fits independent entry-graph and endpoint-graph experts without compact features.",
            "The locked blend reuses independently fitted entry-plus-compact and endpoint-plus-compact experts.",
            "No APK or raw graph is opened or persisted.",
        ],
    }
    audit_path = output / "dual_view_module_ablation_audit.json"
    audit_path.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print("variant mean_BA min_BA std_BA stability AUC", flush=True)
    for row in aggregate_rows:
        print(
            row["variant"],
            f'{row["mean_balanced_accuracy"]:.4f}',
            f'{row["minimum_split_balanced_accuracy"]:.4f}',
            f'{row["std_balanced_accuracy"]:.4f}',
            f'{row["stability_score"]:.4f}',
            f'{row["mean_auc"]:.4f}',
            flush=True,
        )


if __name__ == "__main__":
    main()
