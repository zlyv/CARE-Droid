#!/usr/bin/env python3

from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold


ROOT = Path(__file__).resolve().parent.parent

sys.path.insert(0, str(ROOT / "scripts"))

import run_compact_quantity_shared_caller_2x3_2017_eval as exp


FEATURES = ROOT / "artifacts/p1-features/features.csv"
SCHEMA = ROOT / "artifacts/p1-features/feature_schema.json"

FUTURE = (
    ROOT
    / "artifacts/allratio70-compact-2017-2021"
    / "future_predictions.csv"
)

ENDPOINT_CACHE = (
    ROOT
    / "artifacts/p25-endpoint-cache-full-cap256"
    / "endpoint_centered_tensor_cache.jsonl.gz"
)

OUTPUT = (
    ROOT
    / "artifacts/p52-within-year-group5fold-rawcount-shared"
)

YEARS = (2017, 2018, 2019, 2020, 2021)

EXPECTED_COUNTS = {
    2017: 799,
    2018: 789,
    2019: 399,
    2020: 397,
    2021: 400,
}

VARIANTS = (
    "raw_count70_only",
    "raw_count70_plus_shared_caller",
    "allratio70_only",
    "allratio70_plus_shared_caller",
)

MODEL_SEEDS = (0, 1, 2, 3, 4)

N_SPLITS = 5
SPLIT_RANDOM_STATE = 20260819

ITERATIONS = 1500
LEARNING_RATE = 0.05
L2 = 0.03
THRESHOLD = 0.5


def norm_sha(value):
    return str(value).strip().casefold()


def parse_label(value):
    normalized = str(value).strip().casefold()

    mapping = {
        "benign": 0,
        "goodware": 0,
        "false": 0,
        "0": 0,
        "0.0": 0,
        "malicious": 1,
        "malware": 1,
        "true": 1,
        "1": 1,
        "1.0": 1,
    }

    if normalized not in mapping:
        raise RuntimeError(
            "unknown label: " + repr(value)
        )

    return mapping[normalized]


def write_csv(path, rows):
    if not rows:
        raise RuntimeError(
            "refusing to write empty CSV: " + str(path)
        )

    fields = []

    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)

    with path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fields,
        )

        writer.writeheader()

        for row in rows:
            writer.writerow(
                {
                    field: row.get(field, "")
                    for field in fields
                }
            )


def score(labels, probability):
    labels = np.asarray(labels, dtype=np.int64)
    probability = np.asarray(probability, dtype=np.float64)

    values = exp.metrics(
        labels,
        probability,
        THRESHOLD,
    )

    prediction = (
        probability >= THRESHOLD
    ).astype(np.int64)

    fp = int(
        np.sum(
            (labels == 0)
            & (prediction == 1)
        )
    )

    fn = int(
        np.sum(
            (labels == 1)
            & (prediction == 0)
        )
    )

    return {
        "accuracy": float(values["accuracy"]),
        "balanced_accuracy": float(
            values["balanced_accuracy"]
        ),
        "precision": float(values["precision"]),
        "recall": float(values["recall"]),
        "f1": float(values["f1"]),
        "auc": float(values["auc"]),
        "FP": fp,
        "FN": fn,
    }, prediction


def load_data():
    feature_rows = exp.read_csv_rows(FEATURES)

    feature_by_sha = {
        norm_sha(row["sha256"]): row
        for row in feature_rows
    }

    future_rows = exp.read_csv_rows(FUTURE)

    future_cohorts = defaultdict(set)

    for row in future_rows:
        year = int(float(row["year"]))

        if year in YEARS and year >= 2018:
            future_cohorts[year].add(
                norm_sha(row["sha256"])
            )

    rows_by_year = {
        year: {}
        for year in YEARS
    }

    for feature_row in feature_rows:
        try:
            year = int(
                float(
                    feature_row.get(
                        "metadata_year",
                        "",
                    )
                )
            )
        except (TypeError, ValueError):
            continue

        if year not in YEARS:
            continue

        sha = norm_sha(feature_row["sha256"])

        if (
            year >= 2018
            and sha not in future_cohorts[year]
        ):
            continue

        label = parse_label(
            feature_row["metadata_label"]
        )

        lineage = str(
            feature_row.get(
                "lineage_id",
                "",
            )
        ).strip().casefold()

        if not lineage:
            lineage = "sha256:" + sha

        row = {
            "sha256": sha,
            "label": label,
            "cv_group": lineage,
        }

        if sha in rows_by_year[year]:
            previous = rows_by_year[year][sha]

            if int(previous["label"]) != label:
                raise RuntimeError(
                    "inconsistent label: " + sha
                )

        rows_by_year[year][sha] = row

    normalized = {}

    for year in YEARS:
        rows = [
            rows_by_year[year][sha]
            for sha in sorted(rows_by_year[year])
        ]

        if len(rows) != EXPECTED_COUNTS[year]:
            raise RuntimeError(
                f"{year}: expected "
                f"{EXPECTED_COUNTS[year]}, "
                f"got {len(rows)}"
            )

        normalized[year] = rows

    with SCHEMA.open(
        "r",
        encoding="utf-8",
    ) as handle:
        schema = json.load(handle)

    return normalized, feature_by_sha, schema



# ============================================================
# Fold-safe fixed Compact feature dictionary
#
# 原来的 original_compact_matrix_named 会删掉当前训练集中的
# 常数列，因此某些 CV 折只有 74 维。
#
# 这里把原始 Compact 字段集合冻结为完整 2017 源域得到的
# 76 个字段。后续 76→70 的字段删除规则保持不变。
#
# 注意：
#   1. 不使用标签选择字段；
#   2. 不使用未来年份决定字段；
#   3. 每个 CV 折仍由 compact_quantity_matrices 使用该折
#      训练数据单独计算均值和标准差；
#   4. 训练折中的常数列保留，标准差按原规则置为 1。
# ============================================================

_EXP_COMPACT_QUANTITY_MATRICES = (
    exp.compact_quantity_matrices
)

_FIXED_ORIGINAL_COMPACT_CACHE = None
_FIXED_ORIGINAL_COMPACT_NAMES = None


def _normalise_feature_sha(value):
    return str(value).strip().casefold()


def _feature_metadata_year(row):
    value = row.get(
        "metadata_year",
        row.get("year", ""),
    )

    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _install_fixed_original_compact(
    feature_by_sha,
    schema,
):
    global _FIXED_ORIGINAL_COMPACT_CACHE
    global _FIXED_ORIGINAL_COMPACT_NAMES

    if _FIXED_ORIGINAL_COMPACT_CACHE is not None:
        return

    import run_stable70_vs_allratio70_2017 as ratio_exp

    original_builder = (
        ratio_exp.original_compact_matrix_named
    )

    reference_rows = [
        {"sha256": _normalise_feature_sha(sha)}
        for sha, row in sorted(feature_by_sha.items())
        if _feature_metadata_year(row) == 2017
    ]

    if len(reference_rows) != 799:
        raise RuntimeError(
            "expected 799 source-2017 feature rows, got "
            + str(len(reference_rows))
        )

    all_rows = [
        {"sha256": _normalise_feature_sha(sha)}
        for sha in sorted(feature_by_sha)
    ]

    (
        _reference_matrix,
        all_matrix,
        names,
    ) = original_builder(
        reference_rows,
        all_rows,
        feature_by_sha,
        schema,
    )

    all_matrix = np.asarray(
        all_matrix,
        dtype=float,
    )

    names = list(names)

    if len(names) != 76:
        raise RuntimeError(
            "source-frozen Compact dictionary must contain "
            "76 fields, got "
            + str(len(names))
        )

    if all_matrix.shape != (len(all_rows), 76):
        raise RuntimeError(
            "unexpected frozen Compact matrix shape: "
            + repr(all_matrix.shape)
        )

    cache = {
        _normalise_feature_sha(row["sha256"]):
            np.asarray(values, dtype=float)
        for row, values in zip(all_rows, all_matrix)
    }

    if len(cache) != len(all_rows):
        raise RuntimeError(
            "duplicate SHA while constructing fixed Compact cache"
        )

    def fixed_original_compact_matrix_named(
        train_rows,
        eval_rows,
        _feature_by_sha,
        _schema,
    ):
        def build(rows):
            values = []

            for row in rows:
                sha = _normalise_feature_sha(
                    row["sha256"]
                )

                if sha not in cache:
                    raise RuntimeError(
                        "fixed Compact cache missing SHA: "
                        + sha
                    )

                values.append(cache[sha])

            if not values:
                return np.zeros(
                    (0, len(names)),
                    dtype=float,
                )

            return np.vstack(values).astype(
                float,
                copy=True,
            )

        return (
            build(train_rows),
            build(eval_rows),
            list(names),
        )

    # compact_quantity_matrices 内部重新 import 的也是同一个模块，
    # 因此替换模块属性即可让所有折使用固定 76 字段。
    ratio_exp.original_compact_matrix_named = (
        fixed_original_compact_matrix_named
    )

    _FIXED_ORIGINAL_COMPACT_CACHE = cache
    _FIXED_ORIGINAL_COMPACT_NAMES = names

    print(
        "source-frozen original Compact fields =",
        len(names),
        "; final Compact fields = 70",
        flush=True,
    )


def compact_quantity_matrices_fold_safe(
    train_rows,
    test_rows,
    feature_by_sha,
    schema,
):
    """Use the fixed 76-field dictionary and preserve return type."""
    _install_fixed_original_compact(
        feature_by_sha,
        schema,
    )

    # compact_quantity_matrices 实际返回一个映射对象。
    # 不改变、不遍历、不检查其内部返回结构。
    return _EXP_COMPACT_QUANTITY_MATRICES(
        train_rows,
        test_rows,
        feature_by_sha,
        schema,
    )


def main():
    OUTPUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 92)
    print("WITHIN-YEAR STRATIFIED GROUP 5-FOLD EVALUATION")
    print("=" * 92)

    rows_by_year, feature_by_sha, schema = load_data()

    endpoint_cache = exp.read_cache(
        ENDPOINT_CACHE
    )

    per_fold = []
    per_seed = []
    predictions = []
    fold_assignments = []

    for year in YEARS:
        print()
        print("=" * 92)
        print("YEAR", year)
        print("=" * 92)

        rows = rows_by_year[year]

        labels = np.asarray(
            [int(row["label"]) for row in rows],
            dtype=np.int64,
        )

        groups = np.asarray(
            [row["cv_group"] for row in rows],
            dtype=object,
        )

        hyper_all, hyper_audit = (
            exp.hypergraph_matrix(
                rows,
                endpoint_cache,
            )
        )

        if len(hyper_audit) != len(rows):
            raise RuntimeError(
                f"{year}: hypergraph audit mismatch"
            )

        splitter = StratifiedGroupKFold(
            n_splits=N_SPLITS,
            shuffle=True,
            random_state=SPLIT_RANDOM_STATE,
        )

        folds = list(
            splitter.split(
                np.zeros(len(rows)),
                labels,
                groups=groups,
            )
        )

        oof_probability = {
            variant: {
                seed: np.full(
                    len(rows),
                    np.nan,
                    dtype=np.float64,
                )
                for seed in MODEL_SEEDS
            }
            for variant in VARIANTS
        }

        oof_fold = np.full(
            len(rows),
            -1,
            dtype=np.int64,
        )

        for fold_index, (
            train_indices,
            test_indices,
        ) in enumerate(folds, 1):

            train_rows = [
                rows[int(index)]
                for index in train_indices
            ]

            test_rows = [
                rows[int(index)]
                for index in test_indices
            ]

            train_labels = labels[train_indices]
            test_labels = labels[test_indices]

            train_groups = set(groups[train_indices])
            test_groups = set(groups[test_indices])

            if train_groups & test_groups:
                raise RuntimeError(
                    f"{year} fold {fold_index}: "
                    "group leakage"
                )

            oof_fold[test_indices] = fold_index

            for index in test_indices:
                row = rows[int(index)]

                fold_assignments.append(
                    {
                        "year": year,
                        "fold": fold_index,
                        "sha256": row["sha256"],
                        "label": int(row["label"]),
                        "cv_group": row["cv_group"],
                    }
                )

            (
                compact_modes,
                compact_names,
            ) = compact_quantity_matrices_fold_safe(
                train_rows,
                test_rows,
                feature_by_sha,
                schema,
            )

            if len(compact_names) != 70:
                raise RuntimeError(
                    f"{year} fold {fold_index}: "
                    "Compact dimension mismatch"
                )

            hyper_train = hyper_all[train_indices]
            hyper_test = hyper_all[test_indices]

            (
                hyper_train,
                hyper_test,
            ) = exp.standardize_pair(
                hyper_train,
                hyper_test,
            )

            shared_train = hyper_train[
                :,
                :exp.RELATION_DIMENSION,
            ]

            shared_test = hyper_test[
                :,
                :exp.RELATION_DIMENSION,
            ]

            raw_train, raw_test = (
                compact_modes["raw_count70"]
            )

            ratio_train, ratio_test = (
                compact_modes["allratio70"]
            )

            matrices = {
                "raw_count70_only": (
                    raw_train,
                    raw_test,
                ),

                "raw_count70_plus_shared_caller": (
                    np.concatenate(
                        [raw_train, shared_train],
                        axis=1,
                    ),
                    np.concatenate(
                        [raw_test, shared_test],
                        axis=1,
                    ),
                ),

                "allratio70_only": (
                    ratio_train,
                    ratio_test,
                ),

                "allratio70_plus_shared_caller": (
                    np.concatenate(
                        [ratio_train, shared_train],
                        axis=1,
                    ),
                    np.concatenate(
                        [ratio_test, shared_test],
                        axis=1,
                    ),
                ),
            }

            print(
                f"fold={fold_index} "
                f"train={len(train_indices)} "
                f"test={len(test_indices)} "
                f"benign={int(np.sum(test_labels == 0))} "
                f"malware={int(np.sum(test_labels == 1))}"
            )

            for variant in VARIANTS:
                x_train, x_test = matrices[variant]

                for seed in MODEL_SEEDS:
                    probability = exp.predict_logistic(
                        x_train,
                        train_labels,
                        x_test,
                        seed,
                        ITERATIONS,
                        LEARNING_RATE,
                        L2,
                    )

                    probability = np.asarray(
                        probability,
                        dtype=np.float64,
                    )

                    oof_probability[
                        variant
                    ][seed][test_indices] = probability

                    values, _ = score(
                        test_labels,
                        probability,
                    )

                    per_fold.append(
                        {
                            "year": year,
                            "fold": fold_index,
                            "variant": variant,
                            "seed": seed,
                            "train_count": len(
                                train_indices
                            ),
                            "test_count": len(
                                test_indices
                            ),
                            **values,
                            "test_rows_used_for_training":
                                False,
                        }
                    )

        if np.any(oof_fold < 0):
            raise RuntimeError(
                f"{year}: incomplete OOF coverage"
            )

        for variant in VARIANTS:
            for seed in MODEL_SEEDS:
                probability = oof_probability[
                    variant
                ][seed]

                if not np.all(
                    np.isfinite(probability)
                ):
                    raise RuntimeError(
                        f"{year} {variant} seed={seed}: "
                        "incomplete predictions"
                    )

                values, prediction = score(
                    labels,
                    probability,
                )

                per_seed.append(
                    {
                        "year": year,
                        "variant": variant,
                        "seed": seed,
                        "n": len(rows),
                        **values,
                    }
                )

                for index, row in enumerate(rows):
                    predictions.append(
                        {
                            "year": year,
                            "fold": int(
                                oof_fold[index]
                            ),
                            "variant": variant,
                            "seed": seed,
                            "sha256": row["sha256"],
                            "label": int(labels[index]),
                            "probability": float(
                                probability[index]
                            ),
                            "prediction": int(
                                prediction[index]
                            ),
                            "threshold": THRESHOLD,
                        }
                    )

    # ========================================================
    # 每年：先计算完整OOF指标，再对5个模型种子平均
    # ========================================================

    grouped = defaultdict(list)

    for row in per_seed:
        grouped[
            (
                int(row["year"]),
                row["variant"],
            )
        ].append(row)

    per_year = []

    metric_fields = (
        "accuracy",
        "balanced_accuracy",
        "precision",
        "recall",
        "f1",
        "auc",
        "FP",
        "FN",
    )

    for (year, variant), rows in sorted(
        grouped.items()
    ):
        output = {
            "year": year,
            "variant": variant,
            "seeds": len(rows),
            "n": rows[0]["n"],
        }

        for field in metric_fields:
            output[field] = float(
                np.mean(
                    [
                        float(row[field])
                        for row in rows
                    ]
                )
            )

        per_year.append(output)

    # ========================================================
    # 五个年份宏平均
    # ========================================================

    rows_by_variant = defaultdict(list)

    for row in per_year:
        rows_by_variant[
            row["variant"]
        ].append(row)

    summary = []

    for variant in VARIANTS:
        rows = sorted(
            rows_by_variant[variant],
            key=lambda row: int(row["year"]),
        )

        ba = np.asarray(
            [
                row["balanced_accuracy"]
                for row in rows
            ],
            dtype=np.float64,
        )

        auc = np.asarray(
            [
                row["auc"]
                for row in rows
            ],
            dtype=np.float64,
        )

        summary.append(
            {
                "rank": 0,
                "variant": variant,
                "evaluated_years": len(rows),
                "mean_accuracy": float(
                    np.mean(
                        [
                            row["accuracy"]
                            for row in rows
                        ]
                    )
                ),
                "mean_balanced_accuracy":
                    float(ba.mean()),
                "minimum_year_balanced_accuracy":
                    float(ba.min()),
                "std_year_balanced_accuracy":
                    float(ba.std()),
                "stability_score":
                    float(ba.mean() - ba.std()),
                "mean_auc": float(auc.mean()),
                "minimum_year_auc":
                    float(auc.min()),
                "mean_FP": float(
                    np.mean(
                        [
                            row["FP"]
                            for row in rows
                        ]
                    )
                ),
                "mean_FN": float(
                    np.mean(
                        [
                            row["FN"]
                            for row in rows
                        ]
                    )
                ),
            }
        )

    summary.sort(
        key=lambda row: (
            -row["mean_balanced_accuracy"],
            -row["mean_auc"],
            row["variant"],
        )
    )

    for rank, row in enumerate(summary, 1):
        row["rank"] = rank

    write_csv(
        OUTPUT / "per_fold.csv",
        per_fold,
    )

    write_csv(
        OUTPUT / "per_seed.csv",
        per_seed,
    )

    write_csv(
        OUTPUT / "per_year.csv",
        per_year,
    )

    write_csv(
        OUTPUT / "summary.csv",
        summary,
    )

    write_csv(
        OUTPUT / "predictions.csv",
        predictions,
    )

    write_csv(
        OUTPUT / "fold_assignments.csv",
        fold_assignments,
    )

    audit = {
        "phase":
            "P52_WITHIN_YEAR_GROUPED_5FOLD",

        "years":
            list(YEARS),

        "expected_counts":
            EXPECTED_COUNTS,

        "group_field":
            "lineage_id",

        "n_splits":
            N_SPLITS,

        "split_random_state":
            SPLIT_RANDOM_STATE,

        "model_seeds":
            list(MODEL_SEEDS),

        "iterations":
            ITERATIONS,

        "learning_rate":
            LEARNING_RATE,

        "l2":
            L2,

        "threshold":
            THRESHOLD,

        "compact_dimension":
            70,

        "shared_caller_dimension":
            int(exp.RELATION_DIMENSION),

        "variants":
            list(VARIANTS),

        "test_fold_used_for_scaling":
            False,

        "test_fold_used_for_feature_selection":
            False,

        "lineage_group_overlap":
            False,
    }

    with (
        OUTPUT / "audit.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            audit,
            handle,
            ensure_ascii=False,
            indent=2,
        )

    print()
    print("=" * 92)
    print("PER YEAR")
    print("=" * 92)

    for row in per_year:
        print(
            row["year"],
            row["variant"],
            "BA=",
            f"{row['balanced_accuracy']:.6f}",
            "AUC=",
            f"{row['auc']:.6f}",
            "FP=",
            f"{row['FP']:.2f}",
            "FN=",
            f"{row['FN']:.2f}",
        )

    print()
    print("=" * 92)
    print("SUMMARY")
    print("=" * 92)

    for row in summary:
        print(
            row["rank"],
            row["variant"],
            "mean_BA=",
            f"{row['mean_balanced_accuracy']:.6f}",
            "min_BA=",
            f"{row['minimum_year_balanced_accuracy']:.6f}",
            "mean_AUC=",
            f"{row['mean_auc']:.6f}",
        )

    print()
    print("saved =", OUTPUT)


if __name__ == "__main__":
    main()
