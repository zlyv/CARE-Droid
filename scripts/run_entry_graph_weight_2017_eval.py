#!/usr/bin/env python3
"""Select the entry-graph scale using only 2017 rolling validation folds.

For every candidate scale, this script evaluates both the entry-plus-compact
expert and the locked 0.75/0.25 entry/endpoint probability blend.  The
classification threshold remains fixed at 0.5.  Reliability, filtering,
normalization, and model fitting are recomputed from each rolling training
partition.  APKs and raw graphs are never opened.
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
from run_locked_dual_view_module_ablation import (  # noqa: E402
    predict_logistic,
)
from run_pilot_compact_eval import metrics  # noqa: E402
from tasa_droid.p3.reliability import (  # noqa: E402
    PilotReliabilityDefinition,
    fit_training_fold_components,
    read_cache,
)


THRESHOLD = 0.5
ENTRY_PROBABILITY_WEIGHT = 0.75
ENDPOINT_PROBABILITY_WEIGHT = 0.25
DEFAULT_ENTRY_GRAPH_WEIGHTS = [0.25, 0.5, 1.0, 2.0]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def weight_name(value: float) -> str:
    return f"{float(value):g}"


def entry_expert_matrices(
    entry_train: np.ndarray,
    entry_eval: np.ndarray,
    compact_train: np.ndarray,
    compact_eval: np.ndarray,
    entry_graph_weight: float,
) -> tuple[np.ndarray, np.ndarray]:
    weight = float(entry_graph_weight)
    if weight < 0.0:
        raise ValueError("entry graph weight must be nonnegative")
    if len(entry_train) != len(compact_train):
        raise ValueError("training entry and compact rows do not align")
    if len(entry_eval) != len(compact_eval):
        raise ValueError("evaluation entry and compact rows do not align")
    return (
        np.concatenate([weight * entry_train, compact_train], axis=1),
        np.concatenate([weight * entry_eval, compact_eval], axis=1),
    )


def blend_probability(
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


def average_metrics(results: list[dict]) -> dict:
    return {
        f"{name}_mean": float(np.mean([row[name] for row in results]))
        for name in (
            "accuracy", "balanced_accuracy", "precision", "recall", "f1", "auc"
        )
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--entry-cache", required=True, type=Path)
    parser.add_argument("--endpoint-cache", required=True, type=Path)
    parser.add_argument("--features", required=True, type=Path)
    parser.add_argument("--schema", required=True, type=Path)
    parser.add_argument("--splits", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--entry-graph-weights",
        nargs="+",
        type=float,
        default=DEFAULT_ENTRY_GRAPH_WEIGHTS,
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--iterations", type=int, default=1500)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--l2", type=float, default=0.03)
    args = parser.parse_args()

    weights = sorted(set(float(value) for value in args.entry_graph_weights))
    if not weights or any(value < 0.0 for value in weights):
        raise SystemExit("entry graph weights must be a nonempty nonnegative grid")
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
    observed_years = sorted({
        str(feature_by_sha[sha].get("metadata_year", ""))
        for sha in selected_shas
    })
    if observed_years != ["2017"]:
        raise SystemExit(
            "entry graph weight selection accepts only 2017 rows; "
            f"observed={observed_years}"
        )

    definition = PilotReliabilityDefinition(
        alpha_prior=1.0,
        kappa_support=20.0,
        kappa_context=20.0,
        r_min=0.10,
        r_max=0.95,
    )
    entry_args = entry_model_args(definition)
    per_variant_fold: dict[tuple[str, float], list[dict]] = defaultdict(list)
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
        endpoint_x_train = np.concatenate(
            [endpoint_train, compact_train], axis=1
        )
        endpoint_x_test = np.concatenate(
            [endpoint_test, compact_test], axis=1
        )
        entry_matrices = {
            weight: entry_expert_matrices(
                entry_train,
                entry_test,
                compact_train,
                compact_test,
                weight,
            )
            for weight in weights
        }
        module_dimensions.append({
            "split_name": split_name,
            "entry_graph_feature_count": int(entry_train.shape[1]),
            "endpoint_graph_feature_count": int(endpoint_train.shape[1]),
            "compact_feature_count": len(compact_names),
            "entry_expert_feature_count": int(
                entry_train.shape[1] + compact_train.shape[1]
            ),
            "endpoint_expert_feature_count": int(endpoint_x_train.shape[1]),
        })

        fold_metrics: dict[tuple[str, float], list[dict]] = defaultdict(list)
        for seed in args.seeds:
            endpoint_probability = predict_logistic(
                endpoint_x_train,
                y_train,
                endpoint_x_test,
                seed,
                args.iterations,
                args.learning_rate,
                args.l2,
            )
            for weight in weights:
                entry_x_train, entry_x_test = entry_matrices[weight]
                entry_probability = predict_logistic(
                    entry_x_train,
                    y_train,
                    entry_x_test,
                    seed,
                    args.iterations,
                    args.learning_rate,
                    args.l2,
                )
                probabilities = {
                    "entry_expert": entry_probability,
                    "locked_probability_blend": blend_probability(
                        entry_probability, endpoint_probability
                    ),
                }
                for model_family, probability in probabilities.items():
                    key = (model_family, weight)
                    fold_metrics[key].append(
                        metrics(y_test, probability, threshold=THRESHOLD)
                    )
                    for row, label, score in zip(
                        test_rows, y_test, probability
                    ):
                        prediction_rows.append({
                            "model_family": model_family,
                            "entry_graph_weight": weight,
                            "split_name": split_name,
                            "sha256": row["sha256"],
                            "seed": seed,
                            "label": int(label),
                            "probability": float(score),
                            "threshold": THRESHOLD,
                            "predicted_label": int(score >= THRESHOLD),
                        })

        for key, results in fold_metrics.items():
            model_family, weight = key
            row = {
                "model_family": model_family,
                "variant": f"{model_family}__entry_gw{weight_name(weight)}",
                "entry_graph_weight": weight,
                "split_name": split_name,
                "train_count": len(train_rows),
                "validation_count": len(test_rows),
                "seed_count": len(results),
                "threshold": THRESHOLD,
                **average_metrics(results),
            }
            per_variant_fold[key].append(row)
        print(f"completed {split_name}", flush=True)

    summary_rows = []
    for (model_family, weight), rows in per_variant_fold.items():
        ba = np.asarray([row["balanced_accuracy_mean"] for row in rows])
        summary_rows.append({
            "model_family": model_family,
            "variant": f"{model_family}__entry_gw{weight_name(weight)}",
            "entry_graph_weight": weight,
            "mean_accuracy": float(np.mean([
                row["accuracy_mean"] for row in rows
            ])),
            "mean_balanced_accuracy": float(ba.mean()),
            "minimum_fold_balanced_accuracy": float(ba.min()),
            "std_balanced_accuracy": float(ba.std()),
            "stability_score": float(ba.mean() - ba.std()),
            "mean_auc": float(np.mean([row["auc_mean"] for row in rows])),
        })
    summary_rows.sort(key=lambda row: (
        row["model_family"],
        -row["stability_score"],
        -row["mean_balanced_accuracy"],
        row["entry_graph_weight"],
    ))
    family_rank = defaultdict(int)
    for row in summary_rows:
        family_rank[row["model_family"]] += 1
        row["rank_within_family"] = family_rank[row["model_family"]]

    blend_rows = [
        row for row in summary_rows
        if row["model_family"] == "locked_probability_blend"
    ]
    selected_weight = blend_rows[0]["entry_graph_weight"]

    summary_path = output / "entry_graph_weight_2017_summary.csv"
    write_csv(
        summary_path,
        [
            "rank_within_family", "model_family", "variant",
            "entry_graph_weight", "mean_accuracy", "mean_balanced_accuracy",
            "minimum_fold_balanced_accuracy", "std_balanced_accuracy",
            "stability_score", "mean_auc",
        ],
        summary_rows,
    )
    fold_path = output / "entry_graph_weight_2017_per_fold.csv"
    fold_rows = [
        row
        for key in sorted(per_variant_fold)
        for row in per_variant_fold[key]
    ]
    write_csv(
        fold_path,
        [
            "model_family", "variant", "entry_graph_weight", "split_name",
            "train_count", "validation_count", "seed_count", "threshold",
            "accuracy_mean", "balanced_accuracy_mean", "precision_mean",
            "recall_mean", "f1_mean", "auc_mean",
        ],
        fold_rows,
    )
    prediction_path = output / "entry_graph_weight_2017_predictions.csv"
    write_csv(
        prediction_path,
        [
            "model_family", "entry_graph_weight", "split_name", "sha256",
            "seed", "label", "probability", "threshold", "predicted_label",
        ],
        prediction_rows,
    )

    control = next((
        row for row in blend_rows
        if row["entry_graph_weight"] == 0.5
    ), None)
    expected_control_ba = 0.9213554444747093
    audit = {
        "schema_version": "1.0.0",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "phase": "P37_ENTRY_GRAPH_WEIGHT_2017_ROLLING_SELECTION",
        "formal_training_enabled": False,
        "protocol_status": "DEVELOPMENT_2017_ONLY",
        "strict_test_isolation": True,
        "observed_years": observed_years,
        "entry_graph_weight_grid": weights,
        "selected_entry_graph_weight": selected_weight,
        "selection_target": "locked_probability_blend",
        "selection_rule": (
            "stability_score_desc_then_mean_balanced_accuracy_desc_"
            "then_entry_graph_weight_asc"
        ),
        "fixed_threshold": THRESHOLD,
        "threshold_tuning_performed": False,
        "fixed_entry_probability_weight": ENTRY_PROBABILITY_WEIGHT,
        "fixed_endpoint_probability_weight": ENDPOINT_PROBABILITY_WEIGHT,
        "classifier": {
            "type": "custom_l2_logistic",
            "seeds": args.seeds,
            "iterations": args.iterations,
            "learning_rate": args.learning_rate,
            "l2": args.l2,
        },
        "module_dimensions": module_dimensions,
        "summary": summary_rows,
        "weight_0_5_control_reproduction": {
            "observed_mean_balanced_accuracy": (
                control["mean_balanced_accuracy"] if control else None
            ),
            "expected_mean_balanced_accuracy": expected_control_ba,
            "absolute_difference": (
                abs(control["mean_balanced_accuracy"] - expected_control_ba)
                if control else None
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
            "dex_date_is_not_audited_first_seen_utc",
            "formal_dataset_not_evaluated",
        ],
        "notes": [
            "Only 2017 rolling training and validation rows are accepted.",
            "The classification threshold is fixed at 0.5 for every candidate.",
            "The endpoint probability blend weight is fixed at 0.25.",
            "All reliability, filtering, normalization, and fitting use the corresponding rolling training partition only.",
            "No APK, raw graph, DEX byte, or future-year row is opened.",
        ],
    }
    audit_path = output / "entry_graph_weight_2017_audit.json"
    audit_path.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(
        "family rank entry_gw mean_BA min_BA std_BA stability AUC",
        flush=True,
    )
    for row in summary_rows:
        print(
            row["model_family"],
            row["rank_within_family"],
            weight_name(row["entry_graph_weight"]),
            f'{row["mean_balanced_accuracy"]:.4f}',
            f'{row["minimum_fold_balanced_accuracy"]:.4f}',
            f'{row["std_balanced_accuracy"]:.4f}',
            f'{row["stability_score"]:.4f}',
            f'{row["mean_auc"]:.4f}',
            flush=True,
        )
    print(f"selected_entry_graph_weight={weight_name(selected_weight)}", flush=True)


if __name__ == "__main__":
    main()
