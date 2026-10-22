#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedKFold


SCRIPT_DIR = Path(__file__).resolve().parent

if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))


import run_frozen_anchor_residual_2017_2021 as base
import run_stable70_vs_allratio70_2017 as allratio

from run_pilot_compact_eval import fit_logistic


# ============================================================
# Fixed protocol
# ============================================================

N_SPLITS = 5
FOLD_RANDOM_STATE = 2026

LOGISTIC_ITERATIONS = 6000
LOGISTIC_LR = 0.05
LOGISTIC_L2 = 0.03

THRESHOLD = 0.5


def norm_sha(value):
    return str(value or "").strip().casefold()


def safe_float(value):
    try:
        value = float(value)

        if np.isfinite(value):
            return value
    except Exception:
        pass

    return 0.0


def read_csv(path: Path):
    with path.open(
        newline="",
        encoding="utf-8-sig",
    ) as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows, fieldnames):
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(rows)


def sigmoid(values):
    values = np.asarray(
        values,
        dtype=float,
    )

    values = np.clip(
        values,
        -50.0,
        50.0,
    )

    return 1.0 / (
        1.0 + np.exp(-values)
    )


def p71_matrix(
    rows,
    p71_by_sha,
    feature_names,
):
    if not feature_names:
        return np.zeros(
            (len(rows), 0),
            dtype=float,
        )

    matrix = []

    for row in rows:
        sha = norm_sha(
            row["sha256"]
        )

        source = p71_by_sha[sha]

        matrix.append([
            safe_float(
                source.get(name, 0.0)
            )
            for name in feature_names
        ])

    return np.asarray(
        matrix,
        dtype=float,
    )


def standardize_added(
    train_raw,
    val_raw,
    feature_names,
):
    if train_raw.shape[1] == 0:
        return (
            train_raw.copy(),
            val_raw.copy(),
            [],
        )

    train = np.zeros_like(
        train_raw,
        dtype=float,
    )

    val = np.zeros_like(
        val_raw,
        dtype=float,
    )

    audit = []

    for j, name in enumerate(
        feature_names
    ):
        tr = train_raw[:, j]
        va = val_raw[:, j]

        mean = float(
            np.mean(tr)
        )

        std = float(
            np.std(tr)
        )

        train_support = int(
            np.sum(tr != 0)
        )

        val_support = int(
            np.sum(va != 0)
        )

        constant_train = (
            not np.isfinite(std)
            or std < 1e-12
        )

        if constant_train:
            # Honest source-only behavior:
            # if current training fold never observes this feature,
            # the model cannot learn a coefficient for it.
            train[:, j] = 0.0
            val[:, j] = 0.0

        else:
            train[:, j] = (
                tr - mean
            ) / std

            val[:, j] = (
                va - mean
            ) / std

        audit.append({
            "feature": name,
            "train_mean": mean,
            "train_std": std,
            "train_support": train_support,
            "val_support": val_support,
            "constant_in_train": int(
                constant_train
            ),
        })

    return train, val, audit



def compact_matrix_allratio70_foldsafe(
    train_rows,
    test_rows,
    compact_by_sha,
    compact_schema,
    canonical_original_names,
):
    """
    Fold-safe version of AllRatio70.

    The legacy original_compact_matrix_named() can return fewer
    than 76 columns when a feature disappears in a particular
    training fold.

    We restore the already-frozen source Compact schema before
    applying the AllRatio70 transformation.

    Important:
      - canonical column NAMES are frozen once from source data;
      - ratio mean/std are still computed ONLY from train_rows;
      - missing fold columns are represented as zero;
      - no validation labels/statistics are used for fitting.
    """

    (
        fold_train,
        fold_test,
        fold_names,
    ) = allratio.original_compact_matrix_named(
        train_rows,
        test_rows,
        compact_by_sha,
        compact_schema,
    )

    fold_names = list(fold_names)
    canonical_original_names = list(
        canonical_original_names
    )

    if len(canonical_original_names) != 76:
        raise RuntimeError(
            "Expected frozen original Compact schema = 76, got "
            + str(len(canonical_original_names))
        )

    if len(set(canonical_original_names)) != 76:
        raise RuntimeError(
            "Duplicate feature names in frozen Compact schema"
        )

    if len(set(fold_names)) != len(fold_names):
        raise RuntimeError(
            "Duplicate feature names in current fold Compact"
        )

    canonical_set = set(
        canonical_original_names
    )

    unexpected = sorted(
        set(fold_names)
        - canonical_set
    )

    if unexpected:
        raise RuntimeError(
            "Current fold produced features outside frozen schema: "
            + repr(unexpected)
        )

    fold_index = {
        name: i
        for i, name in enumerate(
            fold_names
        )
    }

    missing = [
        name
        for name in canonical_original_names
        if name not in fold_index
    ]

    print(
        "fold raw compact dim =",
        len(fold_names),
    )

    if missing:
        print(
            "fold missing frozen compact features =",
            len(missing),
        )

        for name in missing:
            print(
                "  [ZERO-FILL]",
                name,
            )
    else:
        print(
            "fold missing frozen compact features = 0"
        )

    # --------------------------------------------------------
    # Restore fixed 76-column order.
    # --------------------------------------------------------

    restored_train = np.zeros(
        (
            len(train_rows),
            len(canonical_original_names),
        ),
        dtype=float,
    )

    restored_test = np.zeros(
        (
            len(test_rows),
            len(canonical_original_names),
        ),
        dtype=float,
    )

    for target_index, name in enumerate(
        canonical_original_names
    ):
        source_index = fold_index.get(
            name
        )

        if source_index is None:
            continue

        restored_train[
            :,
            target_index,
        ] = fold_train[
            :,
            source_index,
        ]

        restored_test[
            :,
            target_index,
        ] = fold_test[
            :,
            source_index,
        ]

    compact_train = restored_train
    compact_test = restored_test
    compact_names = list(
        canonical_original_names
    )

    # --------------------------------------------------------
    # AllRatio70 requires exactly 52 original semantic columns.
    # --------------------------------------------------------

    semantic_features = [
        name
        for name in compact_names
        if (
            name.startswith("role_node__")
            or name.startswith("role_edge__")
        )
    ]

    if len(semantic_features) != 52:
        raise RuntimeError(
            "Expected 52 original semantic features, got "
            + str(len(semantic_features))
        )

    ratio_features = [
        name
        for name in semantic_features
        if name not in allratio.DROP_FEATURES
    ]

    if len(ratio_features) != 46:
        raise RuntimeError(
            "Expected 46 retained ratio features, got "
            + str(len(ratio_features))
        )

    # --------------------------------------------------------
    # Recompute every semantic ratio with TRAIN-FOLD statistics.
    # --------------------------------------------------------

    for numerator_name in ratio_features:

        if numerator_name.startswith(
            "role_node__"
        ):
            denominator_name = (
                "sensitive_node_count"
            )

        elif numerator_name.startswith(
            "role_edge__"
        ):
            denominator_name = (
                "sensitive_edge_count"
            )

        else:
            raise RuntimeError(
                "Unknown semantic feature type: "
                + numerator_name
            )

        column_index = compact_names.index(
            numerator_name
        )

        train_ratio = (
            allratio.raw_ratio_values(
                train_rows,
                compact_by_sha,
                numerator_name,
                denominator_name,
            )
        )

        test_ratio = (
            allratio.raw_ratio_values(
                test_rows,
                compact_by_sha,
                numerator_name,
                denominator_name,
            )
        )

        mean = float(
            np.mean(train_ratio)
        )

        std = float(
            np.std(train_ratio)
        )

        if (
            not np.isfinite(std)
            or std <= 1e-12
        ):
            std = 1.0

        compact_train[
            :,
            column_index,
        ] = (
            train_ratio
            - mean
        ) / std

        compact_test[
            :,
            column_index,
        ] = (
            test_ratio
            - mean
        ) / std

    # --------------------------------------------------------
    # Drop the same frozen six role dimensions.
    # --------------------------------------------------------

    found = {
        name
        for name in compact_names
        if name in allratio.DROP_FEATURES
    }

    if found != allratio.DROP_FEATURES:
        raise RuntimeError(
            "Frozen schema does not contain all six DROP_FEATURES: "
            + repr(
                sorted(
                    allratio.DROP_FEATURES
                    - found
                )
            )
        )

    keep_idx = [
        i
        for i, name in enumerate(
            compact_names
        )
        if name not in allratio.DROP_FEATURES
    ]

    compact_train = compact_train[
        :,
        keep_idx,
    ]

    compact_test = compact_test[
        :,
        keep_idx,
    ]

    compact_names = [
        compact_names[i]
        for i in keep_idx
    ]

    if len(compact_names) != 70:
        raise RuntimeError(
            "Fold-safe AllRatio70 expected 70 dimensions, got "
            + str(len(compact_names))
        )

    return (
        compact_train,
        compact_test,
        compact_names,
    )



def fit_predict(
    x_train,
    y_train,
    x_val,
    seed,
):
    weight, bias = fit_logistic(
        x_train,
        y_train.astype(int),
        seed,
        iterations=LOGISTIC_ITERATIONS,
        learning_rate=LOGISTIC_LR,
        l2=LOGISTIC_L2,
    )

    weight = np.asarray(
        weight,
        dtype=float,
    ).reshape(-1)

    bias = float(
        np.asarray(
            bias,
            dtype=float,
        ).reshape(-1)[0]
    )

    logits = (
        x_val @ weight
        + bias
    )

    probability = sigmoid(
        logits
    )

    return (
        probability,
        weight,
        bias,
    )


def metric_row(
    y_true,
    probability,
):
    return base.metrics(
        np.asarray(
            y_true,
            dtype=int,
        ),
        np.asarray(
            probability,
            dtype=float,
        ),
        threshold=THRESHOLD,
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--compact",
        type=Path,
        default=Path(
            "artifacts/p1-features/features.csv"
        ),
    )

    parser.add_argument(
        "--compact-schema",
        type=Path,
        default=Path(
            "artifacts/p1-features/feature_schema.json"
        ),
    )

    parser.add_argument(
        "--p71",
        type=Path,
        default=Path(
            "artifacts/p71/"
            "verified_dependency_features_2017_v3.csv"
        ),
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "artifacts/"
            "p72-verified-dependency-source-ablation"
        ),
    )

    args = parser.parse_args()

    args.output.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ========================================================
    # Load source data
    # ========================================================

    compact_rows_all = read_csv(
        args.compact
    )

    p71_rows = read_csv(
        args.p71
    )

    with args.compact_schema.open(
        encoding="utf-8",
    ) as f:
        compact_schema = json.load(f)

    compact_by_sha = {}

    for row in compact_rows_all:
        sha = norm_sha(
            row.get("sha256")
        )

        if sha:
            compact_by_sha[sha] = row

    p71_by_sha = {}

    for row in p71_rows:
        sha = norm_sha(
            row.get("sha256")
        )

        if sha:
            p71_by_sha[sha] = row

    # ========================================================
    # Frozen complete-case source cohort
    # ========================================================

    cohort = []

    for row in compact_rows_all:
        if (
            str(
                row.get(
                    "metadata_year",
                    "",
                )
            ).strip()
            != "2017"
        ):
            continue

        if (
            str(
                row.get(
                    "status",
                    "",
                )
            ).strip()
            != "OK"
        ):
            continue

        sha = norm_sha(
            row.get("sha256")
        )

        if sha not in p71_by_sha:
            continue

        p71_row = p71_by_sha[
            sha
        ]

        if (
            safe_float(
                p71_row.get(
                    "dep_analysis_ok",
                    0,
                )
            )
            != 1.0
        ):
            continue

        row = dict(row)
        row["sha256"] = sha

        cohort.append(row)

    cohort = sorted(
        cohort,
        key=lambda x: x["sha256"],
    )

    labels = np.asarray(
        [
            base.label_value(
                row.get(
                    "metadata_label",
                    "",
                )
            )
            for row in cohort
        ],
        dtype=np.int64,
    )

    print(
        "========================================"
    )
    print(
        "P72 VERIFIED DEPENDENCY SOURCE ABLATION"
    )
    print(
        "========================================"
    )

    print("Cohort APKs :", len(cohort))

    print(
        "Benign      :",
        int(np.sum(labels == 0)),
    )

    print(
        "Malicious   :",
        int(np.sum(labels == 1)),
    )

    if len(cohort) != 799:
        raise RuntimeError(
            "Expected complete-case cohort = 799, got "
            + str(len(cohort))
        )

    # ========================================================
    # Feature groups
    # ========================================================

    p71_columns = set(
        p71_rows[0].keys()
    )

    pair_features = sorted(
        name
        for name in p71_columns
        if name.startswith(
            "dep_verified_pair__"
        )
    )

    same_features = sorted(
        name
        for name in p71_columns
        if name.startswith(
            "dep_verified_same__"
        )
    )

    V_CROSS = [
        "dep_verified_cross_role_count",
        *pair_features,
    ]

    V_SAME = [
        "dep_verified_same_role_count",
        *same_features,
    ]

    V_ALL = [
        *V_CROSS,
        *V_SAME,
    ]

    Q = [
        "dep_resolved_no_sensitive_count",
        "dep_unresolved_count",
        "dep_resolution_rate",
        "dep_unresolved_rate",
        "dep_carrier_unit_count",
        "dep_carrier_unresolved_count",
        "dep_non_payload_endpoint_count",
    ]

    S = [
        "dep_sink__dynamic_code_loading_count",
        "dep_sink__network_transmission_or_upload_count",
        "dep_sink__command_execution_count",
        "dep_sink__sms_send_receive_count",
        "dep_direct_payload_count",
    ]

    CONFIGS = {
        "C": [],
        "C+Vcross": V_CROSS,
        "C+Vsame": V_SAME,
        "C+V": V_ALL,
        "C+Q": Q,
        "C+S": S,
        "C+V+Q": V_ALL + Q,
    }

    required = {
        name
        for names in CONFIGS.values()
        for name in names
    }

    missing = sorted(
        required - p71_columns
    )

    if missing:
        raise RuntimeError(
            "Missing P71 features: "
            + repr(missing)
        )

    print()
    print("Strict pair features:")

    for name in pair_features:
        print(" ", name)

    print()
    print("Same-role features:")

    for name in same_features:
        print(" ", name)

    # ========================================================
    # Save feature-group definition
    # ========================================================

    group_rows = []

    for config_name, features in (
        CONFIGS.items()
    ):
        if not features:
            group_rows.append({
                "config": config_name,
                "feature": "(compact_only)",
            })

        for feature in features:
            group_rows.append({
                "config": config_name,
                "feature": feature,
            })

    write_csv(
        args.output
        / "p72_feature_groups.csv",
        group_rows,
        [
            "config",
            "feature",
        ],
    )


    # ========================================================
    # Freeze the already-established source Compact schema.
    #
    # Only feature NAMES are recovered here.
    # No labels or validation performance are used.
    #
    # Fold-specific transformations below remain train-only.
    # ========================================================

    (
        _canonical_train,
        _canonical_probe,
        canonical_original_names,
    ) = allratio.original_compact_matrix_named(
        cohort,
        cohort[:1],
        compact_by_sha,
        compact_schema,
    )

    canonical_original_names = list(
        canonical_original_names
    )

    print()
    print(
        "Frozen original Compact dimension:",
        len(canonical_original_names),
    )

    if len(canonical_original_names) != 76:
        raise RuntimeError(
            "Cannot recover frozen 76-dimensional Compact schema; "
            "got "
            + str(len(canonical_original_names))
        )

    canonical_semantic = [
        name
        for name in canonical_original_names
        if (
            name.startswith("role_node__")
            or name.startswith("role_edge__")
        )
    ]

    print(
        "Frozen semantic dimensions:",
        len(canonical_semantic),
    )

    if len(canonical_semantic) != 52:
        raise RuntimeError(
            "Frozen Compact schema expected 52 semantic dimensions; "
            "got "
            + str(len(canonical_semantic))
        )


    # ========================================================
    # Same fixed stratified folds for every configuration
    # ========================================================

    skf = StratifiedKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=FOLD_RANDOM_STATE,
    )

    n = len(cohort)

    fold_assignment = np.zeros(
        n,
        dtype=int,
    )

    oof_probability = {
        name: np.zeros(
            n,
            dtype=float,
        )
        for name in CONFIGS
    }

    per_fold_rows = []
    coefficient_rows = []
    support_rows = []

    # ========================================================
    # Train/evaluate
    # ========================================================

    for fold_id, (
        train_idx,
        val_idx,
    ) in enumerate(
        skf.split(
            np.zeros(n),
            labels,
        ),
        start=1,
    ):
        print()
        print(
            "----------------------------------------"
        )
        print(
            f"Fold {fold_id}/{N_SPLITS}"
        )
        print(
            "----------------------------------------"
        )

        fold_assignment[
            val_idx
        ] = fold_id

        train_rows = [
            cohort[i]
            for i in train_idx
        ]

        val_rows = [
            cohort[i]
            for i in val_idx
        ]

        y_train = labels[
            train_idx
        ]

        y_val = labels[
            val_idx
        ]

        # Exact AllRatio70 Compact construction.
        (
            compact_train,
            compact_val,
            compact_names,
        ) = compact_matrix_allratio70_foldsafe(
            train_rows,
            val_rows,
            compact_by_sha,
            compact_schema,
            canonical_original_names,
        )

        if len(compact_names) != 70:
            raise RuntimeError(
                "Expected Compact dimension 70, got "
                + str(
                    len(compact_names)
                )
            )

        print(
            "train/val =",
            len(train_rows),
            "/",
            len(val_rows),
        )

        print(
            "compact dim =",
            len(compact_names),
        )

        # Same optimization seed for all configurations
        # inside this fold.
        model_seed = fold_id - 1

        for (
            config_name,
            added_features,
        ) in CONFIGS.items():

            if added_features:
                train_raw = p71_matrix(
                    train_rows,
                    p71_by_sha,
                    added_features,
                )

                val_raw = p71_matrix(
                    val_rows,
                    p71_by_sha,
                    added_features,
                )

                (
                    added_train,
                    added_val,
                    added_audit,
                ) = standardize_added(
                    train_raw,
                    val_raw,
                    added_features,
                )

                x_train = np.concatenate(
                    [
                        compact_train,
                        added_train,
                    ],
                    axis=1,
                )

                x_val = np.concatenate(
                    [
                        compact_val,
                        added_val,
                    ],
                    axis=1,
                )

            else:
                x_train = (
                    compact_train.copy()
                )

                x_val = (
                    compact_val.copy()
                )

                added_audit = []

            (
                probability,
                weight,
                bias,
            ) = fit_predict(
                x_train,
                y_train,
                x_val,
                seed=model_seed,
            )

            oof_probability[
                config_name
            ][val_idx] = probability

            result = metric_row(
                y_val,
                probability,
            )

            per_fold_rows.append({
                "config": config_name,
                "fold": fold_id,
                "seed": model_seed,
                "train_n": len(train_idx),
                "val_n": len(val_idx),
                "compact_dim": 70,
                "added_dim": len(
                    added_features
                ),
                "total_dim": int(
                    x_train.shape[1]
                ),
                "BA": result[
                    "balanced_accuracy"
                ],
                "AUC": result[
                    "auc"
                ],
                "ACC": result[
                    "accuracy"
                ],
                "F1": result[
                    "f1"
                ],
                "precision": result[
                    "precision"
                ],
                "recall": result[
                    "recall"
                ],
                "TP": result["tp"],
                "TN": result["tn"],
                "FP": result["fp"],
                "FN": result["fn"],
            })

            print(
                f"{config_name:12s}",
                f"BA={result['balanced_accuracy']:.4f}",
                f"ACC={result['accuracy']:.4f}",
                f"AUC={result['auc']:.4f}",
            )

            # -----------------------------------------------
            # Added-feature coefficients only.
            # -----------------------------------------------

            if added_features:
                offset = 70

                audit_by_name = {
                    item["feature"]: item
                    for item in added_audit
                }

                for j, feature in enumerate(
                    added_features
                ):
                    audit = audit_by_name[
                        feature
                    ]

                    coefficient_rows.append({
                        "config": config_name,
                        "fold": fold_id,
                        "feature": feature,
                        "coefficient": float(
                            weight[
                                offset + j
                            ]
                        ),
                        "train_mean": audit[
                            "train_mean"
                        ],
                        "train_std": audit[
                            "train_std"
                        ],
                        "train_support": audit[
                            "train_support"
                        ],
                        "val_support": audit[
                            "val_support"
                        ],
                        "constant_in_train": audit[
                            "constant_in_train"
                        ],
                    })

                    support_rows.append({
                        "config": config_name,
                        "fold": fold_id,
                        **audit,
                    })

    # ========================================================
    # OOF summary
    # ========================================================

    summary_rows = []

    baseline_result = metric_row(
        labels,
        oof_probability["C"],
    )

    for config_name in CONFIGS:
        result = metric_row(
            labels,
            oof_probability[
                config_name
            ],
        )

        config_fold_rows = [
            row
            for row in per_fold_rows
            if row["config"]
            == config_name
        ]

        summary_rows.append({
            "config": config_name,
            "n": n,
            "added_dim": len(
                CONFIGS[
                    config_name
                ]
            ),
            "OOF_BA": result[
                "balanced_accuracy"
            ],
            "OOF_AUC": result[
                "auc"
            ],
            "OOF_ACC": result[
                "accuracy"
            ],
            "OOF_F1": result[
                "f1"
            ],
            "OOF_precision": result[
                "precision"
            ],
            "OOF_recall": result[
                "recall"
            ],
            "TP": result["tp"],
            "TN": result["tn"],
            "FP": result["fp"],
            "FN": result["fn"],
            "delta_BA_vs_C": (
                result[
                    "balanced_accuracy"
                ]
                - baseline_result[
                    "balanced_accuracy"
                ]
            ),
            "delta_ACC_vs_C": (
                result["accuracy"]
                - baseline_result[
                    "accuracy"
                ]
            ),
            "delta_AUC_vs_C": (
                result["auc"]
                - baseline_result[
                    "auc"
                ]
            ),
            "mean_fold_BA": float(
                np.mean([
                    row["BA"]
                    for row in config_fold_rows
                ])
            ),
            "min_fold_BA": float(
                np.min([
                    row["BA"]
                    for row in config_fold_rows
                ])
            ),
        })

    # ========================================================
    # Wide OOF prediction table
    # ========================================================

    oof_rows = []

    for i, row in enumerate(
        cohort
    ):
        sha = norm_sha(
            row["sha256"]
        )

        p71 = p71_by_sha[
            sha
        ]

        vcross_any = int(
            safe_float(
                p71.get(
                    "dep_verified_cross_role_count",
                    0,
                )
            )
            > 0
        )

        vsame_any = int(
            safe_float(
                p71.get(
                    "dep_verified_same_role_count",
                    0,
                )
            )
            > 0
        )

        output = {
            "sha256": sha,
            "label": int(
                labels[i]
            ),
            "metadata_label": row.get(
                "metadata_label",
                "",
            ),
            "fold": int(
                fold_assignment[i]
            ),
            "vcross_any": vcross_any,
            "vsame_any": vsame_any,
            "strict_verified_any": int(
                vcross_any
                or vsame_any
            ),
        }

        for config_name in CONFIGS:
            prob = float(
                oof_probability[
                    config_name
                ][i]
            )

            pred = int(
                prob >= THRESHOLD
            )

            safe_name = (
                config_name
                .replace("+", "_plus_")
            )

            output[
                safe_name + "_prob"
            ] = prob

            output[
                safe_name + "_pred"
            ] = pred

        oof_rows.append(
            output
        )

    # ========================================================
    # Changed-case audit relative to Compact
    # ========================================================

    changed_rows = []

    baseline_prob = (
        oof_probability["C"]
    )

    baseline_pred = (
        baseline_prob >= THRESHOLD
    ).astype(int)

    for config_name in CONFIGS:
        if config_name == "C":
            continue

        config_prob = (
            oof_probability[
                config_name
            ]
        )

        config_pred = (
            config_prob >= THRESHOLD
        ).astype(int)

        for i in range(n):
            if (
                config_pred[i]
                == baseline_pred[i]
            ):
                continue

            base_correct = int(
                baseline_pred[i]
                == labels[i]
            )

            config_correct = int(
                config_pred[i]
                == labels[i]
            )

            if (
                not base_correct
                and config_correct
            ):
                change_type = "CORRECTION"

            elif (
                base_correct
                and not config_correct
            ):
                change_type = "BREAK"

            else:
                change_type = "OTHER"

            sha = norm_sha(
                cohort[i]["sha256"]
            )

            p71 = p71_by_sha[
                sha
            ]

            changed_rows.append({
                "config": config_name,
                "sha256": sha,
                "fold": int(
                    fold_assignment[i]
                ),
                "label": int(
                    labels[i]
                ),
                "metadata_label": cohort[i].get(
                    "metadata_label",
                    "",
                ),
                "change_type": change_type,
                "compact_probability": float(
                    baseline_prob[i]
                ),
                "config_probability": float(
                    config_prob[i]
                ),
                "compact_prediction": int(
                    baseline_pred[i]
                ),
                "config_prediction": int(
                    config_pred[i]
                ),
                "dep_verified_cross_role_count":
                    safe_float(
                        p71.get(
                            "dep_verified_cross_role_count",
                            0,
                        )
                    ),
                "dep_verified_same_role_count":
                    safe_float(
                        p71.get(
                            "dep_verified_same_role_count",
                            0,
                        )
                    ),
            })

    # ========================================================
    # Persist
    # ========================================================

    write_csv(
        args.output
        / "p72_summary.csv",
        summary_rows,
        list(
            summary_rows[0].keys()
        ),
    )

    write_csv(
        args.output
        / "p72_per_fold.csv",
        per_fold_rows,
        list(
            per_fold_rows[0].keys()
        ),
    )

    write_csv(
        args.output
        / "p72_oof_predictions.csv",
        oof_rows,
        list(
            oof_rows[0].keys()
        ),
    )

    if changed_rows:
        write_csv(
            args.output
            / "p72_changed_cases.csv",
            changed_rows,
            list(
                changed_rows[0].keys()
            ),
        )
    else:
        (
            args.output
            / "p72_changed_cases.csv"
        ).write_text(
            "config,sha256\n",
            encoding="utf-8",
        )

    if coefficient_rows:
        write_csv(
            args.output
            / "p72_added_feature_coefficients.csv",
            coefficient_rows,
            list(
                coefficient_rows[0].keys()
            ),
        )

    if support_rows:
        write_csv(
            args.output
            / "p72_feature_support.csv",
            support_rows,
            list(
                support_rows[0].keys()
            ),
        )

    # ========================================================
    # Console summary
    # ========================================================

    print()
    print(
        "========================================"
    )
    print(
        "P72 OOF SUMMARY"
    )
    print(
        "========================================"
    )

    for row in summary_rows:
        print(
            f"{row['config']:12s}"
            f" BA={row['OOF_BA']:.6f}"
            f" ACC={row['OOF_ACC']:.6f}"
            f" AUC={row['OOF_AUC']:.6f}"
            f" dBA={row['delta_BA_vs_C']:+.6f}"
            f" dACC={row['delta_ACC_vs_C']:+.6f}"
        )

    print()
    print(
        "===== CHANGED CASES ====="
    )

    changed_counter = Counter(
        (
            row["config"],
            row["change_type"],
        )
        for row in changed_rows
    )

    for key, value in sorted(
        changed_counter.items()
    ):
        print(
            f"{key[0]:12s}"
            f" {key[1]:12s}"
            f" {value:4d}"
        )

    print()
    print(
        "Output:",
        args.output,
    )


if __name__ == "__main__":
    main()
