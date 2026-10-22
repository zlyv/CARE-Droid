#!/usr/bin/env python3
"""Locked fixed-2017 evaluation for the manifest-exposure surface module.

The candidate was selected only on chronological 2017 development folds.  This
script keeps those choices fixed and evaluates 2018--2021 rows only for
scoring.  It compares the existing entry-graph + Compact control with the
selected entry-graph + Compact + exposure-surface candidate.
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
    entry_model_args,
)
from run_graph_pool_diagnostic import (  # noqa: E402
    graph_matrix,
    read_csv_rows,
    read_split_rows,
    standardize_pair,
)
from run_manifest_exposure_2017_eval import (  # noqa: E402
    AUDIT_COLUMNS,
    safe_float,
    select_and_standardize_pair,
    sha256_file,
    surface_features,
)
from run_pilot_compact_eval import fit_logistic, metrics  # noqa: E402
from tasa_droid.p3.reliability import (  # noqa: E402
    PilotReliabilityDefinition,
    fit_training_fold_components,
    read_cache,
)


THRESHOLD = 0.5
GRAPH_WEIGHT = 0.5
L2 = 0.01


def write_csv(path: Path, fields: list[str], rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--entry-cache", required=True, type=Path)
    parser.add_argument("--features", required=True, type=Path)
    parser.add_argument("--schema", required=True, type=Path)
    parser.add_argument("--splits", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--iterations", type=int, default=1500)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    args = parser.parse_args()
    if not args.seeds:
        raise SystemExit("at least one seed is required")

    paths = {
        "entry_cache": args.entry_cache.expanduser().resolve(strict=True),
        "features": args.features.expanduser().resolve(strict=True),
        "schema": args.schema.expanduser().resolve(strict=True),
        "splits": args.splits.expanduser().resolve(strict=True),
    }
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)

    cache = read_cache(paths["entry_cache"])
    feature_rows = read_csv_rows(paths["features"])
    feature_by_sha = {row["sha256"].casefold(): row for row in feature_rows}
    schema = json.loads(paths["schema"].read_text(encoding="utf-8"))
    split_rows = read_split_rows(paths["splits"])
    attach_audit_metadata(cache, feature_by_sha)

    if not feature_rows:
        raise SystemExit("feature CSV is empty")
    all_fields = list(feature_rows[0])
    schema_fields = list(schema.get("feature_fields", all_fields))
    base_names = sorted(
        name for name in schema_fields
        if name not in AUDIT_COLUMNS and not name.startswith("manifest_exposure__")
    )
    exposure_names = sorted(
        name for name in all_fields
        if name.startswith("manifest_exposure__")
        and name != "manifest_exposure__parse_ok"
    )
    if not exposure_names:
        raise SystemExit("manifest exposure fields are absent from the feature CSV")
    surface_names = sorted(
        surface_features(feature_rows[0], base_names, exposure_names)
    )

    definition = PilotReliabilityDefinition(
        alpha_prior=1.0, kappa_support=20.0, kappa_context=20.0,
        r_min=0.10, r_max=0.95,
    )
    entry_args = entry_model_args(definition)
    summary_rows = []
    prediction_rows = []

    for split_name in sorted(split_rows):
        train_rows = split_rows[split_name]["train"]
        test_rows = split_rows[split_name]["test"]
        if not train_rows or not test_rows:
            raise SystemExit(f"empty partition: {split_name}")
        train_years = {
            str(feature_by_sha[row["sha256"]].get("metadata_year", ""))
            for row in train_rows
        }
        test_years = {
            str(feature_by_sha[row["sha256"]].get("metadata_year", ""))
            for row in test_rows
        }
        if train_years != {"2017"}:
            raise SystemExit(
                f"fixed-2017 protocol violation in {split_name}: "
                f"training_years={sorted(train_years)}"
            )
        if not test_years.issubset({"2018", "2019", "2020", "2021"}):
            raise SystemExit(
                f"unexpected future years in {split_name}: {sorted(test_years)}"
            )

        selected = {
            row["sha256"] for row in train_rows + test_rows
        }
        missing_cache = selected - set(cache)
        missing_features = selected - set(feature_by_sha)
        if missing_cache or missing_features:
            raise SystemExit(
                f"missing rows in {split_name}: "
                f"cache={len(missing_cache)} features={len(missing_features)}"
            )

        evidence = fit_training_fold_components(
            cache,
            {row["sha256"] for row in train_rows},
            definition,
            active_components=ACTIVE_RELIABILITY_COMPONENTS,
        )
        entry_train, y_train = graph_matrix(
            train_rows, cache, evidence, entry_args, True
        )
        entry_test, y_test = graph_matrix(
            test_rows, cache, evidence, entry_args, True
        )
        entry_train, entry_test = standardize_pair(entry_train, entry_test)
        compact_train, compact_test, compact_count = select_and_standardize_pair(
            train_rows, test_rows, feature_by_sha, base_names
        )
        surface_train, surface_test, surface_count = select_and_standardize_pair(
            train_rows,
            test_rows,
            feature_by_sha,
            surface_names,
            derived=True,
            base_names=base_names,
            exposure_names=exposure_names,
        )
        matrices = {
            "entry_compact_control": (
                compact_train, compact_test, 0
            ),
            "entry_compact_exposure_surface": (
                np.concatenate([compact_train, surface_train], axis=1),
                np.concatenate([compact_test, surface_test], axis=1),
                surface_count,
            ),
        }

        for variant, (extra_train, extra_test, extra_count) in matrices.items():
            x_train = np.concatenate([GRAPH_WEIGHT * entry_train, extra_train], axis=1)
            x_test = np.concatenate([GRAPH_WEIGHT * entry_test, extra_test], axis=1)
            seed_metrics = []
            for seed in args.seeds:
                weights, bias = fit_logistic(
                    x_train,
                    y_train,
                    seed,
                    iterations=args.iterations,
                    learning_rate=args.learning_rate,
                    l2=L2,
                )
                probability = 1.0 / (
                    1.0 + np.exp(-np.clip(x_test @ weights + bias, -40.0, 40.0))
                )
                result = metrics(y_test, probability, THRESHOLD)
                seed_metrics.append(result)
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
            summary_rows.append({
                "variant": variant,
                "split_name": split_name,
                "train_count": len(train_rows),
                "test_count": len(test_rows),
                "training_years": ",".join(sorted(train_years)),
                "test_years": ",".join(sorted(test_years)),
                "feature_count": int(x_train.shape[1]),
                "compact_feature_count": int(compact_count),
                "surface_feature_count": int(extra_count),
                "threshold": THRESHOLD,
                "accuracy_mean": float(np.mean([item["accuracy"] for item in seed_metrics])),
                "balanced_accuracy_mean": float(np.mean([
                    item["balanced_accuracy"] for item in seed_metrics
                ])),
                "precision_mean": float(np.mean([item["precision"] for item in seed_metrics])),
                "recall_mean": float(np.mean([item["recall"] for item in seed_metrics])),
                "f1_mean": float(np.mean([item["f1"] for item in seed_metrics])),
                "auc_mean": float(np.mean([item["auc"] for item in seed_metrics])),
            })
        print(f"completed {split_name}", flush=True)

    summary_path = output / "manifest_exposure_fixed2017_per_split.csv"
    write_csv(
        summary_path,
        [
            "variant", "split_name", "train_count", "test_count",
            "training_years", "test_years", "feature_count",
            "compact_feature_count", "surface_feature_count", "threshold",
            "accuracy_mean", "balanced_accuracy_mean", "precision_mean",
            "recall_mean", "f1_mean", "auc_mean",
        ],
        summary_rows,
    )
    prediction_path = output / "manifest_exposure_fixed2017_predictions.csv"
    write_csv(
        prediction_path,
        [
            "variant", "split_name", "sha256", "seed", "label",
            "probability", "threshold", "predicted_label",
        ],
        prediction_rows,
    )

    aggregate_rows = []
    for variant in sorted({row["variant"] for row in summary_rows}):
        rows = [row for row in summary_rows if row["variant"] == variant]
        ba = np.asarray([row["balanced_accuracy_mean"] for row in rows])
        acc = np.asarray([row["accuracy_mean"] for row in rows])
        auc = np.asarray([row["auc_mean"] for row in rows])
        aggregate_rows.append({
            "variant": variant,
            "evaluated_split_count": len(rows),
            "mean_accuracy": float(acc.mean()),
            "mean_balanced_accuracy": float(ba.mean()),
            "minimum_year_balanced_accuracy": float(ba.min()),
            "std_year_balanced_accuracy": float(ba.std()),
            "mean_auc": float(auc.mean()),
        })
    aggregate_path = output / "manifest_exposure_fixed2017_summary.csv"
    write_csv(
        aggregate_path,
        [
            "variant", "evaluated_split_count", "mean_accuracy",
            "mean_balanced_accuracy", "minimum_year_balanced_accuracy",
            "std_year_balanced_accuracy", "mean_auc",
        ],
        aggregate_rows,
    )

    audit = {
        "schema_version": "1.0.0",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "phase": "P23_MANIFEST_EXPOSURE_FIXED2017_FUTURE_EVALUATION",
        "formal_training_enabled": False,
        "protocol_status": "LOCKED_CANDIDATE_RETROSPECTIVE_NOT_CONFIRMATORY",
        "strict_test_isolation": True,
        "training_years": ["2017"],
        "test_years": ["2018", "2019", "2020", "2021"],
        "fixed_threshold": THRESHOLD,
        "graph_weight": GRAPH_WEIGHT,
        "l2": L2,
        "seeds": args.seeds,
        "entry_cache": str(paths["entry_cache"]),
        "entry_cache_sha256": sha256_file(paths["entry_cache"]),
        "features": str(paths["features"]),
        "features_sha256": sha256_file(paths["features"]),
        "schema": str(paths["schema"]),
        "schema_sha256": sha256_file(paths["schema"]),
        "split_manifest": str(paths["splits"]),
        "split_manifest_sha256": sha256_file(paths["splits"]),
        "module": {
            "name": "Manifest Exposure Surface",
            "label_free": True,
            "call_graph_view_added": False,
            "surface_feature_count": len(surface_names),
        },
        "outputs": {
            "summary": str(aggregate_path),
            "per_split": str(summary_path),
            "predictions": str(prediction_path),
        },
        "blocking_reasons": [
            "2018_2021_rows_were_used_during_historical_development",
            "audited_first_seen_utc_not_joined",
            "tif_per_apk_predictions_not_available",
        ],
        "notes": [
            "Only the fixed 2017 training partition contributes to fitting.",
            "Future-year labels are used only for post-hoc scoring.",
            "The threshold and hyperparameters were selected before this evaluation.",
            "The endpoint-centred graph view is not used.",
        ],
    }
    audit_path = output / "manifest_exposure_fixed2017_evaluation_audit.json"
    audit_path.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
