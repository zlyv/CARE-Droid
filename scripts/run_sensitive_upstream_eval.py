#!/usr/bin/env python3
"""Evaluate frozen no-Entry sensitive-upstream context.

Variants:
  1. Compact only
  2. Sensitive-upstream only
  3. Compact + 0.5 * sensitive-upstream

Protocol is deliberately aligned with P50/P53:
  * fixed threshold 0.5;
  * custom L2 logistic regression;
  * seeds 0..4 by default;
  * L2 0.03 by default;
  * no test-set tuning.
"""

from __future__ import annotations

import argparse
import csv
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
    entry_model_args,
    sha256_file,
)
from run_dual_view_error_attribution import (  # noqa: E402
    compact_matrix_named,
)
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


def write_csv(path: Path, rows: list[dict]):
    if not rows:
        return
    with path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(rows[0]),
        )
        writer.writeheader()
        writer.writerows(rows)


def mean_metrics(metric_rows):
    result = {}
    for name in (
        "accuracy",
        "balanced_accuracy",
        "precision",
        "recall",
        "f1",
        "auc",
    ):
        values = [
            row[name]
            for row in metric_rows
            if row.get(name) is not None
        ]
        result[name] = (
            float(np.mean(values))
            if values
            else None
        )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--cache",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "--features",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "--schema",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "--splits",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
    )

    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[0, 1, 2, 3, 4],
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=1500,
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.05,
    )
    parser.add_argument(
        "--l2",
        type=float,
        default=0.03,
    )
    parser.add_argument(
        "--upstream-weight",
        type=float,
        default=0.5,
    )

    args = parser.parse_args()

    if args.upstream_weight <= 0:
        raise SystemExit(
            "upstream-weight must be positive"
        )

    cache_path = args.cache.expanduser().resolve(
        strict=True
    )
    feature_path = args.features.expanduser().resolve(
        strict=True
    )
    schema_path = args.schema.expanduser().resolve(
        strict=True
    )
    split_path = args.splits.expanduser().resolve(
        strict=True
    )

    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)

    cache = read_cache(cache_path)

    feature_rows = read_csv_rows(feature_path)
    feature_by_sha = {
        row["sha256"].casefold(): row
        for row in feature_rows
    }

    schema = json.loads(
        schema_path.read_text(encoding="utf-8")
    )

    split_rows = read_split_rows(split_path)

    attach_audit_metadata(
        cache,
        feature_by_sha,
    )

    selected_shas = {
        row["sha256"]
        for groups in split_rows.values()
        for partition in ("train", "test")
        for row in groups[partition]
    }

    missing = {
        "upstream_cache": len(
            selected_shas - set(cache)
        ),
        "compact_features": len(
            selected_shas - set(feature_by_sha)
        ),
    }

    if any(missing.values()):
        raise SystemExit(
            f"missing selected rows: {missing}"
        )

    definition = PilotReliabilityDefinition(
        alpha_prior=1.0,
        kappa_support=20.0,
        kappa_context=20.0,
        r_min=0.10,
        r_max=0.95,
    )

    graph_args = entry_model_args(definition)

    fused_name = (
        f"compact_plus_sensitive_upstream_w"
        f"{args.upstream_weight:g}"
    )

    variants = [
        "compact_only",
        "sensitive_upstream_only",
        fused_name,
    ]

    split_results = []
    prediction_rows = []

    for split_name in sorted(split_rows):
        train_rows = split_rows[split_name]["train"]
        test_rows = split_rows[split_name]["test"]

        if not train_rows or not test_rows:
            raise SystemExit(
                f"empty partition: {split_name}"
            )

        for row in train_rows + test_rows:
            sha = row["sha256"]
            row["metadata_label"] = (
                feature_by_sha[sha].get(
                    "metadata_label",
                    "",
                )
            )

        train_shas = {
            row["sha256"]
            for row in train_rows
        }

        evidence = fit_training_fold_components(
            cache,
            train_shas,
            definition,
            active_components=(
                ACTIVE_RELIABILITY_COMPONENTS
            ),
        )

        upstream_train, y_train = graph_matrix(
            train_rows,
            cache,
            evidence,
            graph_args,
            True,
        )
        upstream_test, y_test = graph_matrix(
            test_rows,
            cache,
            evidence,
            graph_args,
            True,
        )

        upstream_train, upstream_test = (
            standardize_pair(
                upstream_train,
                upstream_test,
            )
        )

        (
            compact_train,
            compact_test,
            compact_names,
        ) = compact_matrix_named(
            train_rows,
            test_rows,
            feature_by_sha,
            schema,
        )

        matrices = {
            "compact_only": (
                compact_train,
                compact_test,
            ),
            "sensitive_upstream_only": (
                upstream_train,
                upstream_test,
            ),
            fused_name: (
                np.concatenate(
                    [
                        compact_train,
                        args.upstream_weight
                        * upstream_train,
                    ],
                    axis=1,
                ),
                np.concatenate(
                    [
                        compact_test,
                        args.upstream_weight
                        * upstream_test,
                    ],
                    axis=1,
                ),
            ),
        }

        seed_metrics = {
            variant: []
            for variant in variants
        }

        for seed in args.seeds:
            for variant in variants:
                x_train, x_test = matrices[variant]

                probability = predict_logistic(
                    x_train,
                    y_train,
                    x_test,
                    seed,
                    args.iterations,
                    args.learning_rate,
                    args.l2,
                )

                metric = metrics(
                    y_test,
                    probability,
                    threshold=THRESHOLD,
                )
                seed_metrics[variant].append(metric)

                for (
                    row,
                    label,
                    value,
                ) in zip(
                    test_rows,
                    y_test,
                    probability,
                ):
                    prediction_rows.append({
                        "variant": variant,
                        "split_name": split_name,
                        "year": feature_by_sha[
                            row["sha256"]
                        ].get(
                            "metadata_year",
                            "",
                        ),
                        "sha256": row["sha256"],
                        "seed": seed,
                        "label": int(label),
                        "probability": float(value),
                        "predicted_label": int(
                            value >= THRESHOLD
                        ),
                    })

        for variant in variants:
            averaged = mean_metrics(
                seed_metrics[variant]
            )

            split_results.append({
                "variant": variant,
                "split_name": split_name,
                "train_count": len(train_rows),
                "test_count": len(test_rows),
                "feature_count": int(
                    matrices[variant][0].shape[1]
                ),
                "compact_feature_count": len(
                    compact_names
                ),
                "threshold": THRESHOLD,
                **{
                    f"{name}_mean": value
                    for name, value in averaged.items()
                },
            })

        print(
            f"completed {split_name}",
            flush=True,
        )

    # Seed-averaged rescue/harm relative to Compact.
    probabilities = defaultdict(list)
    metadata = {}

    for row in prediction_rows:
        key = (
            row["split_name"],
            row["sha256"],
            row["variant"],
        )
        probabilities[key].append(
            float(row["probability"])
        )
        metadata[
            (
                row["split_name"],
                row["sha256"],
            )
        ] = (
            row["year"],
            int(row["label"]),
        )

    pair_counts = defaultdict(
        lambda: {
            "rescue": 0,
            "harm": 0,
            "still_correct": 0,
            "still_wrong": 0,
        }
    )

    for (
        split_name,
        sha256,
    ), (
        year,
        label,
    ) in metadata.items():

        compact_p = float(np.mean(
            probabilities[
                (
                    split_name,
                    sha256,
                    "compact_only",
                )
            ]
        ))

        fused_p = float(np.mean(
            probabilities[
                (
                    split_name,
                    sha256,
                    fused_name,
                )
            ]
        ))

        compact_correct = int(
            compact_p >= THRESHOLD
        ) == label
        fused_correct = int(
            fused_p >= THRESHOLD
        ) == label

        if not compact_correct and fused_correct:
            transition = "rescue"
        elif compact_correct and not fused_correct:
            transition = "harm"
        elif compact_correct and fused_correct:
            transition = "still_correct"
        else:
            transition = "still_wrong"

        pair_counts[split_name][transition] += 1

    pair_rows = []

    for split_name in sorted(pair_counts):
        counts = pair_counts[split_name]

        pair_rows.append({
            "split_name": split_name,
            "rescue": counts["rescue"],
            "harm": counts["harm"],
            "net_rescue": (
                counts["rescue"]
                - counts["harm"]
            ),
            "still_correct": (
                counts["still_correct"]
            ),
            "still_wrong": (
                counts["still_wrong"]
            ),
        })

    summary_rows = []

    for variant in variants:
        rows = [
            row
            for row in split_results
            if row["variant"] == variant
        ]

        ba = np.asarray([
            row["balanced_accuracy_mean"]
            for row in rows
        ])
        auc = np.asarray([
            row["auc_mean"]
            for row in rows
        ])

        summary_rows.append({
            "variant": variant,
            "mean_balanced_accuracy": float(
                ba.mean()
            ),
            "std_balanced_accuracy": float(
                ba.std()
            ),
            "minimum_balanced_accuracy": float(
                ba.min()
            ),
            "mean_auc": float(auc.mean()),
            "feature_count": int(
                rows[0]["feature_count"]
            ),
        })

    summary_rows.sort(
        key=lambda row: (
            -row["mean_balanced_accuracy"],
            -row["mean_auc"],
        )
    )

    write_csv(
        output / "sensitive_upstream_eval_per_split.csv",
        split_results,
    )
    write_csv(
        output / "sensitive_upstream_eval_summary.csv",
        summary_rows,
    )
    write_csv(
        output / "sensitive_upstream_eval_predictions.csv",
        prediction_rows,
    )
    write_csv(
        output / "sensitive_upstream_eval_rescue_harm.csv",
        pair_rows,
    )

    audit = {
        "schema_version": "1.0.0",
        "generated_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "phase": "SENSITIVE_UPSTREAM_NO_ENTRY_EVAL",
        "fixed_threshold": THRESHOLD,
        "threshold_tuning_performed": False,
        "upstream_weight": args.upstream_weight,
        "weight_tuning_performed": False,
        "weight_source": (
            "inherited_from_existing_P50_P53_protocol"
        ),
        "classifier": {
            "type": "custom_l2_logistic",
            "seeds": args.seeds,
            "iterations": args.iterations,
            "learning_rate": args.learning_rate,
            "l2": args.l2,
        },
        "cache": str(cache_path),
        "cache_sha256": sha256_file(cache_path),
        "features": str(feature_path),
        "features_sha256": sha256_file(
            feature_path
        ),
        "schema": str(schema_path),
        "schema_sha256": sha256_file(
            schema_path
        ),
        "splits": str(split_path),
        "splits_sha256": sha256_file(
            split_path
        ),
        "representation": {
            "entry_information_used": False,
            "downstream_context_used": False,
            "upstream_path_only": True,
            "ordinary_external_context_removed": True,
        },
    }

    (
        output / "sensitive_upstream_eval_audit.json"
    ).write_text(
        json.dumps(
            audit,
            ensure_ascii=False,
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )

    print()
    print(
        "rank variant mean_BA std_BA min_BA mean_AUC dims"
    )

    for rank, row in enumerate(
        summary_rows,
        1,
    ):
        print(
            rank,
            row["variant"],
            f'{row["mean_balanced_accuracy"]:.6f}',
            f'{row["std_balanced_accuracy"]:.6f}',
            f'{row["minimum_balanced_accuracy"]:.6f}',
            f'{row["mean_auc"]:.6f}',
            row["feature_count"],
        )

    print()
    print("rescue/harm vs Compact")
    for row in pair_rows:
        print(
            row["split_name"],
            f"rescue={row['rescue']}",
            f"harm={row['harm']}",
            f"net={row['net_rescue']}",
        )


if __name__ == "__main__":
    main()
