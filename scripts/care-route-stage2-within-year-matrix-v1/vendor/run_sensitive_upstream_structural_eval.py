#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from run_graph_pool_diagnostic import (  # noqa: E402
    read_csv_rows,
    read_split_rows,
)
from run_dual_view_error_attribution import (  # noqa: E402
    compact_matrix_named,
)
from run_locked_dual_view_module_ablation import (  # noqa: E402
    predict_logistic,
)
from run_pilot_compact_eval import metrics  # noqa: E402


THRESHOLD = 0.5


def label_value(value):
    s = str(value).strip().lower()

    if s in {"0", "0.0", "benign"}:
        return 0

    if s in {"1", "1.0", "malware", "malicious"}:
        return 1

    raise ValueError(f"unknown label: {value!r}")


def read_gz_csv(path: Path):
    with gzip.open(
        path,
        "rt",
        encoding="utf-8",
        newline="",
    ) as handle:
        return list(csv.DictReader(handle))


def standardize_named(
    train_rows,
    test_rows,
    feature_by_sha,
    names,
):
    train = np.asarray(
        [
            [
                float(feature_by_sha[row["sha256"]][name])
                for name in names
            ]
            for row in train_rows
        ],
        dtype=np.float64,
    )

    test = np.asarray(
        [
            [
                float(feature_by_sha[row["sha256"]][name])
                for name in names
            ]
            for row in test_rows
        ],
        dtype=np.float64,
    )

    if not np.all(np.isfinite(train)):
        raise SystemExit(
            "non-finite upstream training values"
        )

    means = train.mean(axis=0)
    scales = train.std(axis=0)

    # Training-fold constant filtering.
    usable = scales > 1e-12

    if not np.any(usable):
        raise SystemExit(
            f"all features constant: {names[:5]}"
        )

    train = train[:, usable]
    test = test[:, usable]
    means = means[usable]
    scales = scales[usable]

    if not np.all(np.isfinite(test)):
        raise SystemExit(
            "non-finite upstream test values"
        )

    return (
        (train - means) / scales,
        (test - means) / scales,
        [name for name, keep in zip(names, usable) if keep],
    )


def size_matrix(
    train_rows,
    test_rows,
    feature_by_sha,
):
    def values(rows):
        result = []

        for row in rows:
            r = feature_by_sha[row["sha256"]]

            result.append([
                math.log1p(
                    float(r["retained_node_count"])
                ),
                math.log1p(
                    float(r["retained_edge_count"])
                ),
                math.log1p(
                    float(
                        r[
                            "retained_internal_caller_count"
                        ]
                    )
                ),
                float(r["max_upstream_distance"]),
            ])

        return np.asarray(
            result,
            dtype=np.float64,
        )

    train = values(train_rows)
    test = values(test_rows)

    means = train.mean(axis=0)
    scales = train.std(axis=0)

    usable = scales > 1e-12

    return (
        (train[:, usable] - means[usable])
        / scales[usable],
        (test[:, usable] - means[usable])
        / scales[usable],
        [
            name
            for name, keep in zip(
                [
                    "log_retained_node_count",
                    "log_retained_edge_count",
                    "log_internal_caller_count",
                    "max_upstream_distance",
                ],
                usable,
            )
            if keep
        ],
    )


def write_csv(path, rows):
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


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--compact-features",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--compact-schema",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--upstream-features",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--splits",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--output",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--weight",
        type=float,
        default=0.5,
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

    args = parser.parse_args()

    args.output.mkdir(
        parents=True,
        exist_ok=True,
    )

    compact_rows = read_csv_rows(
        args.compact_features
    )

    compact_by_sha = {
        str(row["sha256"]).strip().casefold(): row
        for row in compact_rows
    }

    compact_schema = json.loads(
        args.compact_schema.read_text(
            encoding="utf-8"
        )
    )

    upstream_rows = read_gz_csv(
        args.upstream_features
    )

    upstream_by_sha = {
        str(row["sha256"]).strip().casefold(): row
        for row in upstream_rows
    }

    if not upstream_rows:
        raise SystemExit(
            "empty upstream feature CSV"
        )

    header = list(upstream_rows[0])

    # IMPORTANT:
    # Only up_* are formal upstream features.
    # retained_* / truncation fields are audit only.
    full_names = [
        name
        for name in header
        if name.startswith("up_")
    ]

    if len(full_names) != 176:
        raise SystemExit(
            "expected exactly 176 up_* features; "
            f"observed={len(full_names)}"
        )

    inventory_names = [
        name
        for name in full_names
        if (
            name == "up_global_endpoint_log"
            or name == "up_global_role_coverage"
            or name.endswith("__endpoint_log")
        )
    ]

    global_structure_names = [
        name
        for name in full_names
        if (
            name.startswith("up_global_")
            and name not in inventory_names
        )
    ]

    role_structure_names = [
        name
        for name in full_names
        if (
            name.startswith("up_role__")
            and not name.endswith("__endpoint_log")
        )
    ]

    structural_names = (
        global_structure_names
        + role_structure_names
    )

    shared_names = [
        name
        for name in global_structure_names
        if "shared_" in name
    ]

    print(
        "upstream feature groups:",
        flush=True,
    )
    print(
        "  inventory       =",
        len(inventory_names),
        flush=True,
    )
    print(
        "  global_structure=",
        len(global_structure_names),
        flush=True,
    )
    print(
        "  role_structure  =",
        len(role_structure_names),
        flush=True,
    )
    print(
        "  structural      =",
        len(structural_names),
        flush=True,
    )
    print(
        "  shared          =",
        len(shared_names),
        flush=True,
    )
    print(
        "  full            =",
        len(full_names),
        flush=True,
    )

    splits = read_split_rows(args.splits)

    selected_shas = {
        row["sha256"]
        for split in splits.values()
        for partition in ("train", "test")
        for row in split[partition]
    }

    missing_compact = (
        selected_shas
        - set(compact_by_sha)
    )

    missing_upstream = (
        selected_shas
        - set(upstream_by_sha)
    )

    print(
        "selected split APKs:",
        len(selected_shas),
        flush=True,
    )

    print(
        "missing compact:",
        len(missing_compact),
        flush=True,
    )

    print(
        "missing upstream:",
        len(missing_upstream),
        flush=True,
    )

    # Do not silently change the evaluation population.
    if missing_compact or missing_upstream:
        if missing_upstream:
            print(
                "missing upstream SHAs:",
                sorted(missing_upstream)[:20],
                flush=True,
            )

        raise SystemExit(
            "selected train/test rows are missing; "
            "do not drop them silently"
        )

    split_results = []
    prediction_rows = []
    dimension_rows = []

    weight = float(args.weight)

    main_variant = (
        f"compact_plus_upstream_structural_w{weight:g}"
    )

    for split_name in sorted(splits):
        train_rows = splits[split_name]["train"]
        test_rows = splits[split_name]["test"]

        if not train_rows or not test_rows:
            raise SystemExit(
                f"empty split: {split_name}"
            )

        y_train = np.asarray(
            [
                label_value(
                    compact_by_sha[row["sha256"]].get(
                        "metadata_label", ""
                    )
                )
                for row in train_rows
            ],
            dtype=np.int64,
        )

        y_test = np.asarray(
            [
                label_value(
                    compact_by_sha[row["sha256"]].get(
                        "metadata_label", ""
                    )
                )
                for row in test_rows
            ],
            dtype=np.int64,
        )

        compact_train, compact_test, compact_names = (
            compact_matrix_named(
                train_rows,
                test_rows,
                compact_by_sha,
                compact_schema,
            )
        )

        matrices = {
            "compact_only": (
                compact_train,
                compact_test,
            ),
        }

        groups = {
            "inventory":
                inventory_names,

            "global_structure":
                global_structure_names,

            "role_structure":
                role_structure_names,

            "structural":
                structural_names,

            "shared":
                shared_names,

            "full":
                full_names,
        }

        group_dimensions = {}

        for group_name, names in groups.items():
            x_train, x_test, usable = (
                standardize_named(
                    train_rows,
                    test_rows,
                    upstream_by_sha,
                    names,
                )
            )

            group_dimensions[group_name] = len(
                usable
            )

            matrices[
                f"upstream_{group_name}_only"
            ] = (
                x_train,
                x_test,
            )

            matrices[
                f"compact_plus_upstream_{group_name}"
                f"_w{weight:g}"
            ] = (
                np.concatenate(
                    [
                        compact_train,
                        weight * x_train,
                    ],
                    axis=1,
                ),
                np.concatenate(
                    [
                        compact_test,
                        weight * x_test,
                    ],
                    axis=1,
                ),
            )

        size_train, size_test, size_names = (
            size_matrix(
                train_rows,
                test_rows,
                upstream_by_sha,
            )
        )

        matrices[
            f"compact_plus_size_only_w{weight:g}"
        ] = (
            np.concatenate(
                [
                    compact_train,
                    weight * size_train,
                ],
                axis=1,
            ),
            np.concatenate(
                [
                    compact_test,
                    weight * size_test,
                ],
                axis=1,
            ),
        )

        dimension_rows.append({
            "split_name": split_name,
            "compact_features":
                len(compact_names),
            "inventory_features":
                group_dimensions["inventory"],
            "global_structure_features":
                group_dimensions[
                    "global_structure"
                ],
            "role_structure_features":
                group_dimensions[
                    "role_structure"
                ],
            "structural_features":
                group_dimensions["structural"],
            "shared_features":
                group_dimensions["shared"],
            "full_upstream_features":
                group_dimensions["full"],
            "size_control_features":
                len(size_names),
        })

        for variant, (
            x_train,
            x_test,
        ) in matrices.items():

            seed_metrics = []

            for seed in args.seeds:
                probabilities = predict_logistic(
                    x_train,
                    y_train,
                    x_test,
                    seed,
                    args.iterations,
                    args.learning_rate,
                    args.l2,
                )

                m = metrics(
                    y_test,
                    probabilities,
                    threshold=THRESHOLD,
                )

                seed_metrics.append(m)

                for row, label, prob in zip(
                    test_rows,
                    y_test,
                    probabilities,
                ):
                    prediction_rows.append({
                        "variant": variant,
                        "split_name": split_name,
                        "sha256": row["sha256"],
                        "year": compact_by_sha[
                            row["sha256"]
                        ].get(
                            "metadata_year", ""
                        ),
                        "seed": seed,
                        "label": int(label),
                        "probability":
                            float(prob),
                        "predicted_label":
                            int(
                                prob >= THRESHOLD
                            ),
                    })

            result = {
                "variant": variant,
                "split_name": split_name,
                "train_count":
                    len(train_rows),
                "test_count":
                    len(test_rows),
                "feature_count":
                    x_train.shape[1],
            }

            for metric_name in (
                "accuracy",
                "balanced_accuracy",
                "precision",
                "recall",
                "f1",
                "auc",
                "fp",
                "fn",
            ):
                values = [
                    m[metric_name]
                    for m in seed_metrics
                ]

                result[
                    f"{metric_name}_mean"
                ] = float(np.mean(values))

            split_results.append(result)

        print(
            "completed",
            split_name,
            flush=True,
        )

    # ------------------------------------------------------
    # Aggregate across annual splits.
    # ------------------------------------------------------
    by_variant = defaultdict(list)

    for row in split_results:
        by_variant[
            row["variant"]
        ].append(row)

    summary_rows = []

    for variant, rows in by_variant.items():
        bas = np.asarray(
            [
                row[
                    "balanced_accuracy_mean"
                ]
                for row in rows
            ],
            dtype=float,
        )

        accs = np.asarray(
            [
                row["accuracy_mean"]
                for row in rows
            ],
            dtype=float,
        )

        aucs = np.asarray(
            [
                row["auc_mean"]
                for row in rows
            ],
            dtype=float,
        )

        f1s = np.asarray(
            [
                row["f1_mean"]
                for row in rows
            ],
            dtype=float,
        )

        summary_rows.append({
            "variant": variant,
            "split_count": len(rows),
            "mean_balanced_accuracy":
                float(bas.mean()),
            "minimum_balanced_accuracy":
                float(bas.min()),
            "std_balanced_accuracy":
                float(bas.std()),
            "mean_accuracy":
                float(accs.mean()),
            "mean_auc":
                float(aucs.mean()),
            "mean_f1":
                float(f1s.mean()),
            "mean_fp":
                float(np.mean(
                    [r["fp_mean"] for r in rows]
                )),
            "mean_fn":
                float(np.mean(
                    [r["fn_mean"] for r in rows]
                )),
        })

    summary_rows.sort(
        key=lambda row:
            row["mean_balanced_accuracy"],
        reverse=True,
    )

    # ------------------------------------------------------
    # Seed-averaged APK-level paired comparison.
    # Main variant is PREDECLARED, not selected here.
    # ------------------------------------------------------
    prob_groups = defaultdict(list)
    metadata = {}

    for row in prediction_rows:
        key = (
            row["split_name"],
            row["sha256"],
            row["variant"],
        )

        prob_groups[key].append(
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

    paired_rows = []

    for split_name, sha in sorted(metadata):
        year, label = metadata[
            (split_name, sha)
        ]

        compact_prob = float(np.mean(
            prob_groups[
                (
                    split_name,
                    sha,
                    "compact_only",
                )
            ]
        ))

        main_prob = float(np.mean(
            prob_groups[
                (
                    split_name,
                    sha,
                    main_variant,
                )
            ]
        ))

        compact_pred = int(
            compact_prob >= THRESHOLD
        )

        main_pred = int(
            main_prob >= THRESHOLD
        )

        compact_ok = (
            compact_pred == label
        )

        main_ok = (
            main_pred == label
        )

        if (
            not compact_ok
            and main_ok
        ):
            transition = "CORRECTED"

        elif (
            compact_ok
            and not main_ok
        ):
            transition = "REGRESSED"

        elif compact_ok:
            transition = "STILL_CORRECT"

        else:
            transition = "STILL_WRONG"

        paired_rows.append({
            "split_name": split_name,
            "year": year,
            "sha256": sha,
            "label": label,
            "compact_probability":
                compact_prob,
            "upstream_probability":
                main_prob,
            "probability_delta":
                main_prob - compact_prob,
            "compact_predicted_label":
                compact_pred,
            "upstream_predicted_label":
                main_pred,
            "transition": transition,
        })

    write_csv(
        args.output
        / "sensitive_upstream_per_split.csv",
        split_results,
    )

    write_csv(
        args.output
        / "sensitive_upstream_summary.csv",
        summary_rows,
    )

    write_csv(
        args.output
        / "sensitive_upstream_predictions.csv",
        prediction_rows,
    )

    write_csv(
        args.output
        / "sensitive_upstream_dimensions.csv",
        dimension_rows,
    )

    write_csv(
        args.output
        / "sensitive_upstream_paired_apk.csv",
        paired_rows,
    )

    print()
    print(
        "=== RETROSPECTIVE FIXED-2017 RESULTS ==="
    )

    print(
        "IMPORTANT: ranking below is diagnostic only;"
        " do not select a new protocol from future years."
    )

    print()

    for row in summary_rows:
        print(
            f"{row['variant']:48s} "
            f"BA={row['mean_balanced_accuracy']:.6f} "
            f"min={row['minimum_balanced_accuracy']:.6f} "
            f"AUC={row['mean_auc']:.6f} "
            f"Acc={row['mean_accuracy']:.6f} "
            f"F1={row['mean_f1']:.6f} "
            f"FP={row['mean_fp']:.2f} "
            f"FN={row['mean_fn']:.2f}"
        )

    transition_count = defaultdict(int)

    for row in paired_rows:
        transition_count[
            row["transition"]
        ] += 1

    print()
    print(
        "=== PREDECLARED STRUCTURAL MAIN ==="
    )
    print(
        main_variant
    )
    print(
        "corrected =",
        transition_count["CORRECTED"],
    )
    print(
        "regressed =",
        transition_count["REGRESSED"],
    )
    print(
        "net =",
        transition_count["CORRECTED"]
        - transition_count["REGRESSED"],
    )


if __name__ == "__main__":
    main()
