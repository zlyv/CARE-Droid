#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import gzip
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from run_graph_pool_diagnostic import (
    read_csv_rows,
    read_split_rows,
)

from run_dual_view_error_attribution import (
    compact_matrix_named,
)

from run_sensitive_upstream_structural_eval import (
    label_value,
    standardize_named,
)

from run_clean_semantic_relation_2017_eval import (
    evaluate_matrix,
)

from run_pilot_compact_eval import metrics


ALPHA = 0.25
THRESHOLD = 0.5

YEARS = (
    2018,
    2019,
    2020,
    2021,
)

EXPECTED_TEST_COUNTS = {
    2018: 789,
    2019: 399,
    2020: 397,
    2021: 400,
}

VARIANTS = (
    "compact_only",
    "compact_plus_support150",
)


def read_gz_csv(path: Path):
    with gzip.open(
        path,
        "rt",
        encoding="utf-8",
        newline="",
    ) as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows):
    if not rows:
        return

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fields = []
    seen = set()

    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fields.append(key)

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
        writer.writerows(rows)


def support150_names(header):
    support = sorted(
        name
        for name in header
        if name.startswith(
            "semrel_support__"
        )
    )

    if len(support) != 155:
        raise SystemExit(
            "support维数不是155："
            + str(len(support))
        )

    selected = []

    for name in support:
        parts = name.split("__")

        if len(parts) != 3:
            raise SystemExit(
                "异常语义特征名："
                + name
            )

        anchor = parts[1]
        context = parts[2]

        # 删除5个自身关系。
        if anchor != context:
            selected.append(name)

    if len(selected) != 150:
        raise SystemExit(
            "删除自身关系后不是150维："
            + str(len(selected))
        )

    return selected


def average_metric_rows(rows):
    if not rows:
        return {}

    fields = (
        "accuracy",
        "balanced_accuracy",
        "f1",
        "auc",
        "fp",
        "fn",
    )

    out = {}

    for field in fields:
        values = np.asarray(
            [
                float(row[field])
                for row in rows
            ],
            dtype=float,
        )

        out[field] = float(
            np.nanmean(values)
        )

    return out


def prediction_index(prediction_rows):
    """
    (split, variant, seed, sha) -> probability
    """
    probs = {}
    labels = {}

    for row in prediction_rows:
        key = (
            row["split_name"],
            row["variant"],
            int(row["seed"]),
            row["sha256"],
        )

        probs[key] = float(
            row["probability"]
        )

        labels[
            (
                row["split_name"],
                row["sha256"],
            )
        ] = int(row["label"])

    return probs, labels


def per_seed_metrics(
    prediction_rows,
    split_to_year,
):
    groups = defaultdict(list)

    for row in prediction_rows:
        key = (
            row["split_name"],
            row["variant"],
            int(row["seed"]),
        )

        groups[key].append(row)

    output = []

    for (
        split_name,
        variant,
        seed,
    ), rows in sorted(
        groups.items()
    ):
        y = np.asarray(
            [
                int(row["label"])
                for row in rows
            ],
            dtype=int,
        )

        p = np.asarray(
            [
                float(row["probability"])
                for row in rows
            ],
            dtype=float,
        )

        result = metrics(
            y,
            p,
            threshold=THRESHOLD,
        )

        output.append({
            "year":
                split_to_year[
                    split_name
                ],

            "split_name":
                split_name,

            "variant":
                variant,

            "seed":
                seed,

            "n":
                len(rows),

            **result,
        })

    return output


def summarize_by_year(per_seed_rows):
    grouped = defaultdict(list)

    for row in per_seed_rows:
        grouped[
            (
                int(row["year"]),
                row["variant"],
            )
        ].append(row)

    output = []

    for (
        year,
        variant,
    ), rows in sorted(
        grouped.items()
    ):
        avg = average_metric_rows(
            rows
        )

        ba_values = np.asarray(
            [
                float(
                    row[
                        "balanced_accuracy"
                    ]
                )
                for row in rows
            ],
            dtype=float,
        )

        output.append({
            "year":
                year,

            "variant":
                variant,

            "n":
                int(rows[0]["n"]),

            "seed_count":
                len(rows),

            "BA":
                avg[
                    "balanced_accuracy"
                ],

            "BA_std":
                float(
                    ba_values.std()
                ),

            "AUC":
                avg["auc"],

            "ACC":
                avg["accuracy"],

            "F1":
                avg["f1"],

            "FP":
                avg["fp"],

            "FN":
                avg["fn"],
        })

    return output


def year_deltas(summary_rows):
    by_year = defaultdict(dict)

    for row in summary_rows:
        by_year[
            int(row["year"])
        ][
            row["variant"]
        ] = row

    output = []

    for year in YEARS:
        base = by_year[
            year
        ]["compact_only"]

        new = by_year[
            year
        ][
            "compact_plus_support150"
        ]

        output.append({
            "year":
                year,

            "delta_BA":
                float(
                    new["BA"]
                    - base["BA"]
                ),

            "delta_AUC":
                float(
                    new["AUC"]
                    - base["AUC"]
                ),

            "delta_ACC":
                float(
                    new["ACC"]
                    - base["ACC"]
                ),

            "delta_F1":
                float(
                    new["F1"]
                    - base["F1"]
                ),

            "delta_FP":
                float(
                    new["FP"]
                    - base["FP"]
                ),

            "delta_FN":
                float(
                    new["FN"]
                    - base["FN"]
                ),
        })

    return output


def correction_analysis(
    prediction_rows,
    split_to_year,
):
    probs, labels = prediction_index(
        prediction_rows
    )

    # 每个样本先对5个seed的概率取均值，
    # 然后比较最终0.5分类翻转。
    sample_groups = defaultdict(
        lambda: defaultdict(list)
    )

    for row in prediction_rows:
        sample_key = (
            row["split_name"],
            row["sha256"],
        )

        sample_groups[
            sample_key
        ][
            row["variant"]
        ].append(
            float(row["probability"])
        )

    output = []

    for year in YEARS:
        corrected = 0
        regressed = 0
        still_wrong = 0
        still_correct = 0

        benign_corrected = 0
        malware_corrected = 0

        benign_regressed = 0
        malware_regressed = 0

        shifts = []

        n = 0

        for (
            split_name,
            sha,
        ), values in sample_groups.items():

            if (
                split_to_year[
                    split_name
                ]
                != year
            ):
                continue

            if (
                "compact_only"
                not in values
                or
                "compact_plus_support150"
                not in values
            ):
                continue

            label = labels[
                (
                    split_name,
                    sha,
                )
            ]

            p_base = float(
                np.mean(
                    values[
                        "compact_only"
                    ]
                )
            )

            p_new = float(
                np.mean(
                    values[
                        "compact_plus_support150"
                    ]
                )
            )

            old_ok = (
                int(
                    p_base
                    >= THRESHOLD
                )
                == label
            )

            new_ok = (
                int(
                    p_new
                    >= THRESHOLD
                )
                == label
            )

            if (
                not old_ok
                and new_ok
            ):
                corrected += 1

                if label == 0:
                    benign_corrected += 1
                else:
                    malware_corrected += 1

            elif (
                old_ok
                and not new_ok
            ):
                regressed += 1

                if label == 0:
                    benign_regressed += 1
                else:
                    malware_regressed += 1

            elif old_ok:
                still_correct += 1

            else:
                still_wrong += 1

            if label == 1:
                shift = (
                    p_new - p_base
                )
            else:
                shift = (
                    p_base - p_new
                )

            shifts.append(
                shift
            )

            n += 1

        shifts = np.asarray(
            shifts,
            dtype=float,
        )

        output.append({
            "year":
                year,

            "n":
                n,

            "corrected":
                corrected,

            "regressed":
                regressed,

            "net_corrected":
                corrected
                - regressed,

            "benign_corrected":
                benign_corrected,

            "malware_corrected":
                malware_corrected,

            "benign_regressed":
                benign_regressed,

            "malware_regressed":
                malware_regressed,

            "still_wrong":
                still_wrong,

            "still_correct":
                still_correct,

            "mean_truth_direction_shift":
                float(
                    shifts.mean()
                ),

            "truth_direction_rate":
                float(
                    np.mean(
                        shifts > 0
                    )
                ),
        })

    return output


def zero_nonzero_analysis(
    prediction_rows,
    split_to_year,
    zero_map,
):
    """
    按150维原始特征全零/非零拆分。
    这里保持与主结果一致：
    先逐seed计算指标，再对seed求均值。
    """

    groups = defaultdict(list)

    for row in prediction_rows:
        split_name = row[
            "split_name"
        ]

        sha = row["sha256"]

        year = split_to_year[
            split_name
        ]

        zero_state = (
            "zero"
            if zero_map[sha]
            else "nonzero"
        )

        key = (
            year,
            zero_state,
            row["variant"],
            int(row["seed"]),
        )

        groups[key].append(row)

    per_seed = []

    for (
        year,
        zero_state,
        variant,
        seed,
    ), rows in sorted(
        groups.items()
    ):
        y = np.asarray(
            [
                int(row["label"])
                for row in rows
            ],
            dtype=int,
        )

        p = np.asarray(
            [
                float(row["probability"])
                for row in rows
            ],
            dtype=float,
        )

        result = metrics(
            y,
            p,
            threshold=THRESHOLD,
        )

        per_seed.append({
            "year":
                year,

            "group":
                zero_state,

            "variant":
                variant,

            "seed":
                seed,

            "n":
                len(rows),

            "benign":
                int(
                    np.sum(
                        y == 0
                    )
                ),

            "malware":
                int(
                    np.sum(
                        y == 1
                    )
                ),

            **result,
        })

    grouped = defaultdict(list)

    for row in per_seed:
        grouped[
            (
                row["year"],
                row["group"],
                row["variant"],
            )
        ].append(row)

    summary = []

    for (
        year,
        group_name,
        variant,
    ), rows in sorted(
        grouped.items()
    ):
        avg = average_metric_rows(
            rows
        )

        summary.append({
            "year":
                year,

            "group":
                group_name,

            "variant":
                variant,

            "n":
                rows[0]["n"],

            "benign":
                rows[0]["benign"],

            "malware":
                rows[0]["malware"],

            "BA":
                avg[
                    "balanced_accuracy"
                ],

            "AUC":
                avg["auc"],

            "ACC":
                avg["accuracy"],

            "F1":
                avg["f1"],

            "FP":
                avg["fp"],

            "FN":
                avg["fn"],
        })

    return (
        per_seed,
        summary,
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--compact-features",
        type=Path,
        default=Path(
            "artifacts/"
            "p1-features/"
            "features.csv"
        ),
    )

    parser.add_argument(
        "--compact-schema",
        type=Path,
        default=Path(
            "artifacts/"
            "p1-features/"
            "feature_schema.json"
        ),
    )

    parser.add_argument(
        "--semantic-2017",
        type=Path,
        default=Path(
            "artifacts/"
            "clean-local-context-2017/"
            "semantic_relation_features.csv.gz"
        ),
    )

    parser.add_argument(
        "--semantic-future",
        type=Path,
        default=Path(
            "artifacts/"
            "clean-local-context-future/"
            "semantic_relation_features.csv.gz"
        ),
    )

    parser.add_argument(
        "--splits",
        type=Path,
        default=Path(
            "artifacts/"
            "p1-splits-fixed-2017-test-2018-2021/"
            "split_manifest.csv"
        ),
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "artifacts/"
            "frozen-semantic-2018-2021"
        ),
    )

    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[
            0,
            1,
            2,
            3,
            4,
        ],
    )

    parser.add_argument(
        "--iterations",
        type=int,
        default=6000,
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

    for path in (
        args.compact_features,
        args.compact_schema,
        args.semantic_2017,
        args.semantic_future,
        args.splits,
    ):
        if not path.exists():
            raise SystemExit(
                "文件不存在："
                + str(path)
            )

    args.output.mkdir(
        parents=True,
        exist_ok=True,
    )

    # =====================================
    # Compact
    # =====================================

    compact_rows = read_csv_rows(
        args.compact_features
    )

    compact_by_sha = {
        str(
            row["sha256"]
        ).strip().casefold():
            row
        for row in compact_rows
    }

    compact_schema = json.loads(
        args.compact_schema.read_text(
            encoding="utf-8"
        )
    )

    # =====================================
    # 2017 + future semantic
    # =====================================

    semantic_2017_rows = read_gz_csv(
        args.semantic_2017
    )

    semantic_future_rows = read_gz_csv(
        args.semantic_future
    )

    if (
        not semantic_2017_rows
        or not semantic_future_rows
    ):
        raise SystemExit(
            "语义特征文件为空"
        )

    names2017 = support150_names(
        semantic_2017_rows[
            0
        ].keys()
    )

    names_future = support150_names(
        semantic_future_rows[
            0
        ].keys()
    )

    if names2017 != names_future:
        raise SystemExit(
            "2017和未来150维定义不一致"
        )

    names150 = names2017

    semantic_by_sha = {}

    for row in (
        semantic_2017_rows
        + semantic_future_rows
    ):
        sha = str(
            row["sha256"]
        ).strip().casefold()

        if sha in semantic_by_sha:
            raise SystemExit(
                "语义文件SHA重复："
                + sha
            )

        semantic_by_sha[
            sha
        ] = row

    zero_map = {}

    for row in semantic_future_rows:
        sha = str(
            row["sha256"]
        ).strip().casefold()

        zero_map[sha] = all(
            abs(
                float(row[name])
            )
            <= 1e-15
            for name in names150
        )

    # =====================================
    # Fixed temporal splits
    # =====================================

    splits = read_split_rows(
        args.splits
    )

    selected_splits = {}
    split_to_year = {}

    all_train_shas = set()
    all_test_shas = set()

    for split_name, parts in (
        splits.items()
    ):
        train_rows = parts[
            "train"
        ]

        test_rows = parts[
            "test"
        ]

        if not test_rows:
            continue

        # read_split_rows() 返回的行不保留 year 字段。
        # 固定时间划分的年份直接由 split_name 唯一确定。
        year = None

        for candidate_year in YEARS:
            expected_name = (
                f"train_fixed_2017__test_{candidate_year}"
            )

            if split_name == expected_name:
                year = candidate_year
                break

        if year is None:
            continue

        if len(train_rows) != 799:
            raise SystemExit(
                f"{split_name}: "
                f"训练样本不是799，"
                f"实际={len(train_rows)}"
            )

        expected = (
            EXPECTED_TEST_COUNTS[
                year
            ]
        )

        if len(test_rows) != expected:
            raise SystemExit(
                f"{split_name}: "
                f"{year}测试数错误，"
                f"预期={expected}，"
                f"实际={len(test_rows)}"
            )

        selected_splits[
            split_name
        ] = parts

        split_to_year[
            split_name
        ] = year

        all_train_shas.update(
            row["sha256"]
            for row in train_rows
        )

        all_test_shas.update(
            row["sha256"]
            for row in test_rows
        )

    if (
        set(split_to_year.values())
        != set(YEARS)
    ):
        raise SystemExit(
            "没有找到完整的"
            "2018-2021四个测试划分"
        )

    if len(all_train_shas) != 799:
        raise SystemExit(
            "四个划分中的2017唯一SHA"
            f"不是799：{len(all_train_shas)}"
        )

    if len(all_test_shas) != 1985:
        raise SystemExit(
            "未来唯一测试SHA"
            f"不是1985：{len(all_test_shas)}"
        )

    # =====================================
    # Population integrity
    # =====================================

    needed = (
        all_train_shas
        | all_test_shas
    )

    missing_compact = (
        needed
        - set(compact_by_sha)
    )

    missing_semantic = (
        needed
        - set(semantic_by_sha)
    )

    if missing_compact:
        raise SystemExit(
            "Compact缺少样本："
            + str(
                len(
                    missing_compact
                )
            )
        )

    if missing_semantic:
        raise SystemExit(
            "语义特征缺少样本："
            + str(
                len(
                    missing_semantic
                )
            )
        )

    print(
        "======================================"
    )
    print(
        "冻结方法：2017 -> 2018-2021"
    )
    print(
        "======================================"
    )
    print(
        "2017唯一训练SHA =",
        len(all_train_shas),
    )
    print(
        "未来唯一测试SHA =",
        len(all_test_shas),
    )
    print(
        "冻结语义维数 =",
        len(names150),
    )
    print(
        "alpha =",
        ALPHA,
    )
    print(
        "threshold =",
        THRESHOLD,
    )
    print(
        "iterations =",
        args.iterations,
    )
    print(
        "learning_rate =",
        args.learning_rate,
    )
    print(
        "l2 =",
        args.l2,
    )
    print(
        "seeds =",
        args.seeds,
    )

    print()
    print(
        "未来150维全零 =",
        sum(
            zero_map[sha]
            for sha in all_test_shas
        ),
        "/",
        len(all_test_shas),
    )

    prediction_rows = []
    dimension_rows = []

    # =====================================
    # Formal evaluation
    # =====================================

    for split_name in sorted(
        selected_splits,
        key=lambda name:
            split_to_year[name],
    ):
        year = split_to_year[
            split_name
        ]

        train_rows = (
            selected_splits[
                split_name
            ]["train"]
        )

        test_rows = (
            selected_splits[
                split_name
            ]["test"]
        )

        y_train = np.asarray(
            [
                label_value(
                    compact_by_sha[
                        row["sha256"]
                    ].get(
                        "metadata_label",
                        "",
                    )
                )
                for row in train_rows
            ],
            dtype=np.int64,
        )

        y_test = np.asarray(
            [
                label_value(
                    compact_by_sha[
                        row["sha256"]
                    ].get(
                        "metadata_label",
                        "",
                    )
                )
                for row in test_rows
            ],
            dtype=np.int64,
        )

        # read_split_rows() 的行只用于划分和SHA索引。
        # 标签统一从 Compact 特征人口读取。

        (
            compact_train,
            compact_test,
            compact_names,
        ) = compact_matrix_named(
            train_rows,
            test_rows,
            compact_by_sha,
            compact_schema,
        )

        (
            semantic_train,
            semantic_test,
            semantic_usable,
        ) = standardize_named(
            train_rows,
            test_rows,
            semantic_by_sha,
            names150,
        )

        joint_train = np.concatenate(
            [
                compact_train,
                ALPHA
                * semantic_train,
            ],
            axis=1,
        )

        joint_test = np.concatenate(
            [
                compact_test,
                ALPHA
                * semantic_test,
            ],
            axis=1,
        )

        evaluate_matrix(
            split_name,
            train_rows,
            test_rows,
            y_train,
            y_test,
            compact_train,
            compact_test,
            "compact_only",
            args.seeds,
            args.iterations,
            args.learning_rate,
            args.l2,
            prediction_rows,
        )

        evaluate_matrix(
            split_name,
            train_rows,
            test_rows,
            y_train,
            y_test,
            joint_train,
            joint_test,
            "compact_plus_support150",
            args.seeds,
            args.iterations,
            args.learning_rate,
            args.l2,
            prediction_rows,
        )

        zero_count = sum(
            zero_map[
                row["sha256"]
            ]
            for row in test_rows
        )

        dimension_rows.append({
            "year":
                year,

            "split_name":
                split_name,

            "train_n":
                len(train_rows),

            "test_n":
                len(test_rows),

            "compact_dim":
                len(compact_names),

            "semantic_original_dim":
                150,

            "semantic_usable_dim":
                len(
                    semantic_usable
                ),

            "semantic_zero_n":
                zero_count,

            "semantic_zero_rate":
                zero_count
                / len(test_rows),
        })

        print(
            f"{year} 完成："
            f"train={len(train_rows)} "
            f"test={len(test_rows)} "
            f"Compact维数={len(compact_names)} "
            f"语义可用维数={len(semantic_usable)} "
            f"150维全零={zero_count}/"
            f"{len(test_rows)}",
            flush=True,
        )

    # =====================================
    # Metrics
    # =====================================

    seed_rows = per_seed_metrics(
        prediction_rows,
        split_to_year,
    )

    year_rows = summarize_by_year(
        seed_rows
    )

    delta_rows = year_deltas(
        year_rows
    )

    correction_rows = (
        correction_analysis(
            prediction_rows,
            split_to_year,
        )
    )

    (
        group_seed_rows,
        group_summary_rows,
    ) = zero_nonzero_analysis(
        prediction_rows,
        split_to_year,
        zero_map,
    )

    # =====================================
    # Mean over four future years
    # =====================================

    future_mean_rows = []

    for variant in VARIANTS:
        rows = [
            row
            for row in year_rows
            if row["variant"]
            == variant
        ]

        future_mean_rows.append({
            "variant":
                variant,

            "mean_BA":
                float(
                    np.mean(
                        [
                            row["BA"]
                            for row in rows
                        ]
                    )
                ),

            "min_BA":
                float(
                    np.min(
                        [
                            row["BA"]
                            for row in rows
                        ]
                    )
                ),

            "mean_AUC":
                float(
                    np.mean(
                        [
                            row["AUC"]
                            for row in rows
                        ]
                    )
                ),

            "mean_ACC":
                float(
                    np.mean(
                        [
                            row["ACC"]
                            for row in rows
                        ]
                    )
                ),

            "mean_F1":
                float(
                    np.mean(
                        [
                            row["F1"]
                            for row in rows
                        ]
                    )
                ),

            "mean_FP":
                float(
                    np.mean(
                        [
                            row["FP"]
                            for row in rows
                        ]
                    )
                ),

            "mean_FN":
                float(
                    np.mean(
                        [
                            row["FN"]
                            for row in rows
                        ]
                    )
                ),
        })

    base_future = next(
        row
        for row in future_mean_rows
        if row["variant"]
        == "compact_only"
    )

    new_future = next(
        row
        for row in future_mean_rows
        if row["variant"]
        == "compact_plus_support150"
    )

    future_delta = {
        "delta_mean_BA":
            new_future["mean_BA"]
            - base_future["mean_BA"],

        "delta_min_BA":
            new_future["min_BA"]
            - base_future["min_BA"],

        "delta_mean_AUC":
            new_future["mean_AUC"]
            - base_future["mean_AUC"],

        "delta_mean_ACC":
            new_future["mean_ACC"]
            - base_future["mean_ACC"],
    }

    # =====================================
    # Save
    # =====================================

    write_csv(
        args.output
        / "per_seed_metrics.csv",
        seed_rows,
    )

    write_csv(
        args.output
        / "per_year_summary.csv",
        year_rows,
    )

    write_csv(
        args.output
        / "delta_vs_compact.csv",
        delta_rows,
    )

    write_csv(
        args.output
        / "correction_summary.csv",
        correction_rows,
    )

    write_csv(
        args.output
        / "zero_nonzero_per_seed.csv",
        group_seed_rows,
    )

    write_csv(
        args.output
        / "zero_nonzero_summary.csv",
        group_summary_rows,
    )

    write_csv(
        args.output
        / "dimensions.csv",
        dimension_rows,
    )

    write_csv(
        args.output
        / "future_mean_summary.csv",
        future_mean_rows,
    )

    write_csv(
        args.output
        / "predictions.csv",
        prediction_rows,
    )

    # =====================================
    # Print
    # =====================================

    print()
    print(
        "======================================"
    )
    print(
        "逐年正式结果"
    )
    print(
        "======================================"
    )

    by_year = defaultdict(dict)

    for row in year_rows:
        by_year[
            row["year"]
        ][
            row["variant"]
        ] = row

    for year in YEARS:
        print()
        print(
            "----------",
            year,
            "----------",
        )

        for variant in VARIANTS:
            row = by_year[
                year
            ][
                variant
            ]

            print(
                f"{variant:28s} "
                f"BA={row['BA']:.6f} "
                f"AUC={row['AUC']:.6f} "
                f"ACC={row['ACC']:.6f} "
                f"F1={row['F1']:.6f} "
                f"FP={row['FP']:.2f} "
                f"FN={row['FN']:.2f}"
            )

        delta = next(
            row
            for row in delta_rows
            if row["year"]
            == year
        )

        print(
            "变化："
            f"dBA={delta['delta_BA']:+.6f} "
            f"dAUC={delta['delta_AUC']:+.6f} "
            f"dACC={delta['delta_ACC']:+.6f} "
            f"dFP={delta['delta_FP']:+.2f} "
            f"dFN={delta['delta_FN']:+.2f}"
        )

    print()
    print(
        "======================================"
    )
    print(
        "2018-2021四年平均"
    )
    print(
        "======================================"
    )

    for row in future_mean_rows:
        print(
            f"{row['variant']:28s} "
            f"BA={row['mean_BA']:.6f} "
            f"minBA={row['min_BA']:.6f} "
            f"AUC={row['mean_AUC']:.6f} "
            f"ACC={row['mean_ACC']:.6f} "
            f"FP={row['mean_FP']:.2f} "
            f"FN={row['mean_FN']:.2f}"
        )

    print(
        "四年平均变化："
        f"dBA="
        f"{future_delta['delta_mean_BA']:+.6f} "
        f"dMinBA="
        f"{future_delta['delta_min_BA']:+.6f} "
        f"dAUC="
        f"{future_delta['delta_mean_AUC']:+.6f} "
        f"dACC="
        f"{future_delta['delta_mean_ACC']:+.6f}"
    )

    print()
    print(
        "======================================"
    )
    print(
        "逐年修正/退化"
    )
    print(
        "======================================"
    )

    for row in correction_rows:
        print(
            f"{row['year']} "
            f"修正={row['corrected']} "
            f"退化={row['regressed']} "
            f"净修正="
            f"{row['net_corrected']:+d} "
            f"良性修正="
            f"{row['benign_corrected']} "
            f"恶意修正="
            f"{row['malware_corrected']} "
            f"向真实方向="
            f"{row['truth_direction_rate']:.2%}"
        )

    print()
    print(
        "======================================"
    )
    print(
        "全零 / 非零语义关系"
    )
    print(
        "======================================"
    )

    for year in YEARS:
        print()
        print(
            "----------",
            year,
            "----------",
        )

        for group_name in (
            "zero",
            "nonzero",
        ):
            rows = [
                row
                for row
                in group_summary_rows
                if (
                    row["year"]
                    == year
                    and row["group"]
                    == group_name
                )
            ]

            if not rows:
                continue

            print(
                group_name,
                "n=",
                rows[0]["n"],
                "良性=",
                rows[0]["benign"],
                "恶意=",
                rows[0]["malware"],
            )

            for row in rows:
                print(
                    f"  "
                    f"{row['variant']:28s} "
                    f"BA={row['BA']:.6f} "
                    f"AUC={row['AUC']:.6f} "
                    f"ACC={row['ACC']:.6f} "
                    f"FP={row['FP']:.2f} "
                    f"FN={row['FN']:.2f}"
                )

    print()
    print(
        "保存目录：",
        args.output,
    )


if __name__ == "__main__":
    main()
