#!/usr/bin/env python3
"""Evaluate the 2017-locked entry/endpoint residual dual-view candidate.

The candidate was selected using chronological 2017 development folds only:
two independently trained entry+compact and endpoint+compact experts are
combined as 0.75 * entry_probability + 0.25 * endpoint_probability.  Both the
blend weight and the final classification threshold are hard-coded.  Test
rows are never used for fitting reliability, feature filtering,
standardization, model parameters, or selection.
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
    attach_audit_metadata,
    entry_model_args,
    sha256_file,
)
from run_graph_pool_diagnostic import read_csv_rows, read_split_rows  # noqa: E402
from run_pilot_compact_eval import metrics  # noqa: E402
from run_uncertainty_gated_dual_view_2017_eval import (  # noqa: E402
    expert_matrices,
    predict_logistic,
)
from tasa_droid.p3.reliability import (  # noqa: E402
    PilotReliabilityDefinition,
    read_cache,
)


THRESHOLD = 0.5
ENDPOINT_RESIDUAL_WEIGHT = 0.25
ENTRY_RESIDUAL_WEIGHT = 1.0 - ENDPOINT_RESIDUAL_WEIGHT
ALLOWED_TEST_YEARS = {"2018", "2019", "2020", "2021"}


def mean_metric(rows: list[dict], name: str) -> float:
    return float(np.mean([row[name] for row in rows]))


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

    all_rows = [
        row
        for partitions in split_rows.values()
        for partition in ("train", "test")
        for row in partitions[partition]
    ]
    selected_shas = {row["sha256"] for row in all_rows}
    missing_counts = {
        "entry_cache": len(selected_shas - set(entry_cache)),
        "endpoint_cache": len(selected_shas - set(endpoint_cache)),
        "features": len(selected_shas - set(feature_by_sha)),
    }
    if any(missing_counts.values()):
        raise SystemExit(f"missing selected rows: {missing_counts}")

    for row in all_rows:
        feature = feature_by_sha[row["sha256"]]
        row["metadata_label"] = feature.get("metadata_label", "")
        row["metadata_year"] = str(feature.get("metadata_year", ""))

    split_protocol = []
    for split_name, partitions in split_rows.items():
        train_years = {
            row["metadata_year"] for row in partitions["train"]
        }
        test_years = {
            row["metadata_year"] for row in partitions["test"]
        }
        if train_years != {"2017"}:
            raise SystemExit(
                f"{split_name}: locked training years must be exactly 2017; "
                f"observed={sorted(train_years)}"
            )
        if len(test_years) != 1 or not test_years <= ALLOWED_TEST_YEARS:
            raise SystemExit(
                f"{split_name}: expected one test year in 2018--2021; "
                f"observed={sorted(test_years)}"
            )
        split_protocol.append({
            "split_name": split_name,
            "training_years": sorted(train_years),
            "test_years": sorted(test_years),
        })

    definition = PilotReliabilityDefinition(
        alpha_prior=1.0,
        kappa_support=20.0,
        kappa_context=20.0,
        r_min=0.10,
        r_max=0.95,
    )
    entry_args = entry_model_args(definition)
    variants = ("entry_control", "endpoint_expert", "fixed_residual_blend_025")
    per_variant_split = {variant: [] for variant in variants}
    prediction_rows = []

    for split_name in sorted(split_rows):
        train_rows = split_rows[split_name]["train"]
        test_rows = split_rows[split_name]["test"]
        if not train_rows or not test_rows:
            raise SystemExit(f"empty partition: {split_name}")
        matrices = expert_matrices(
            train_rows,
            test_rows,
            entry_cache,
            endpoint_cache,
            feature_by_sha,
            schema,
            definition,
            entry_args,
        )
        (
            entry_train,
            entry_test,
            endpoint_train,
            endpoint_test,
            y_train,
            y_test,
        ) = matrices
        seed_metrics = {variant: [] for variant in variants}
        for seed in args.seeds:
            entry_probability = predict_logistic(
                entry_train,
                y_train,
                entry_test,
                seed,
                args.iterations,
                args.learning_rate,
                args.l2,
            )
            endpoint_probability = predict_logistic(
                endpoint_train,
                y_train,
                endpoint_test,
                seed,
                args.iterations,
                args.learning_rate,
                args.l2,
            )
            scores_by_variant = {
                "entry_control": entry_probability,
                "endpoint_expert": endpoint_probability,
                "fixed_residual_blend_025": (
                    ENTRY_RESIDUAL_WEIGHT * entry_probability
                    + ENDPOINT_RESIDUAL_WEIGHT * endpoint_probability
                ),
            }
            for variant, scores in scores_by_variant.items():
                seed_metrics[variant].append(
                    metrics(y_test, scores, threshold=THRESHOLD)
                )
                for row, label, score in zip(test_rows, y_test, scores):
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
            summary = {
                "variant": variant,
                "split_name": split_name,
                "train_count": len(train_rows),
                "test_count": len(test_rows),
                "threshold": THRESHOLD,
                "test_rows_used_for_training_or_selection": False,
            }
            for metric_name in (
                "accuracy", "balanced_accuracy", "precision", "recall", "f1", "auc"
            ):
                summary[f"{metric_name}_mean"] = mean_metric(
                    seed_metrics[variant], metric_name
                )
            per_variant_split[variant].append(summary)
        print(f"completed {split_name}", flush=True)

    aggregate_rows = []
    for variant in variants:
        split_summaries = per_variant_split[variant]
        ba = np.asarray([
            row["balanced_accuracy_mean"] for row in split_summaries
        ])
        aggregate_rows.append({
            "variant": variant,
            "mean_accuracy": mean_metric(split_summaries, "accuracy_mean"),
            "mean_balanced_accuracy": float(ba.mean()),
            "minimum_year_balanced_accuracy": float(ba.min()),
            "std_year_balanced_accuracy": float(ba.std()),
            "mean_auc": mean_metric(split_summaries, "auc_mean"),
        })

    summary_path = output / "locked_residual_dual_view_summary.csv"
    write_csv(
        summary_path,
        [
            "variant", "mean_accuracy", "mean_balanced_accuracy",
            "minimum_year_balanced_accuracy", "std_year_balanced_accuracy",
            "mean_auc",
        ],
        aggregate_rows,
    )
    split_path = output / "locked_residual_dual_view_per_year.csv"
    split_fields = [
        "variant", "split_name", "train_count", "test_count", "threshold",
        "accuracy_mean", "balanced_accuracy_mean", "precision_mean",
        "recall_mean", "f1_mean", "auc_mean",
        "test_rows_used_for_training_or_selection",
    ]
    write_csv(
        split_path,
        split_fields,
        [row for variant in variants for row in per_variant_split[variant]],
    )
    prediction_path = output / "locked_residual_dual_view_predictions.csv"
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
        "phase": "P28_2017_LOCKED_RESIDUAL_DUAL_VIEW_FUTURE_YEAR_EVALUATION",
        "formal_training_enabled": False,
        "protocol_status": "LOCKED_FROM_2017_DEVELOPMENT_EXPLORATORY_LEGACY_TEST",
        "strict_test_isolation": True,
        "fixed_threshold": THRESHOLD,
        "threshold_tuning_performed": False,
        "entry_probability_weight": ENTRY_RESIDUAL_WEIGHT,
        "endpoint_probability_weight": ENDPOINT_RESIDUAL_WEIGHT,
        "blend_weight_tuning_performed_in_this_evaluation": False,
        "selection_source": "P27 chronological 2017 development folds only",
        "entry_cache": str(paths["entry_cache"]),
        "entry_cache_sha256": sha256_file(paths["entry_cache"]),
        "endpoint_cache": str(paths["endpoint_cache"]),
        "endpoint_cache_sha256": sha256_file(paths["endpoint_cache"]),
        "features": str(paths["features"]),
        "features_sha256": sha256_file(paths["features"]),
        "schema": str(paths["schema"]),
        "schema_sha256": sha256_file(paths["schema"]),
        "split_manifest": str(paths["splits"]),
        "split_manifest_sha256": sha256_file(paths["splits"]),
        "split_protocol": split_protocol,
        "classifier": {
            "type": "two_independent_custom_l2_logistic_experts",
            "seeds": args.seeds,
            "iterations": args.iterations,
            "learning_rate": args.learning_rate,
            "l2": args.l2,
        },
        "aggregate_summaries": aggregate_rows,
        "split_summaries": per_variant_split,
        "outputs": {
            "summary": str(summary_path),
            "per_year": str(split_path),
            "predictions": str(prediction_path),
        },
        "blocking_reasons": [
            "legacy_2018_2021_test_years_were_used_during_prior_candidate_development",
            "audited_first_seen_utc_not_joined",
            "formal_dataset_not_evaluated",
        ],
        "notes": [
            "The blend and threshold were locked before this run from 2017 development only.",
            "Each expert fits reliability, filtering, normalization, and weights from 2017 training rows only.",
            "Test labels are used only after prediction to compute metrics.",
            "This legacy future-year result is exploratory and not a formal confirmatory result.",
        ],
    }
    audit_path = output / "locked_residual_dual_view_audit.json"
    audit_path.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print("variant mean_ACC mean_BA min_year_BA std_year_BA mean_AUC", flush=True)
    for row in aggregate_rows:
        print(
            row["variant"],
            f'{row["mean_accuracy"]:.4f}',
            f'{row["mean_balanced_accuracy"]:.4f}',
            f'{row["minimum_year_balanced_accuracy"]:.4f}',
            f'{row["std_year_balanced_accuracy"]:.4f}',
            f'{row["mean_auc"]:.4f}',
            flush=True,
        )


if __name__ == "__main__":
    main()
