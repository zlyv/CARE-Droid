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


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))


from run_graph_pool_diagnostic import (
    read_csv_rows,
    read_split_rows,
)

from run_dual_view_error_attribution import (
    compact_matrix_named,
)

from run_locked_dual_view_module_ablation import (
    predict_logistic,
)

from run_pilot_compact_eval import (
    metrics,
)

from run_sensitive_upstream_structural_eval import (
    label_value,
    standardize_named,
)


THRESHOLD = 0.5

ANCHOR_ROLES = {
    "microphone_access",
    "camera_access",
    "call_control_or_recording",
    "location_access",
    "background_service_control",
}

ALPHAS = (
    0.10,
    0.25,
    0.50,
    1.00,
)


def read_gz_csv(path: Path):
    with gzip.open(
        path,
        "rt",
        encoding="utf-8",
        newline="",
    ) as f:
        return list(csv.DictReader(f))


def write_csv(
    path: Path,
    rows,
):
    if not rows:
        return

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fields = list(rows[0])

    with path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fields,
        )
        writer.writeheader()
        writer.writerows(rows)


def mean_valid(values):
    xs = [
        float(x)
        for x in values
        if x is not None
    ]

    return (
        float(np.mean(xs))
        if xs
        else None
    )


def semantic_columns(header):
    support = sorted(
        name
        for name in header
        if name.startswith(
            "semrel_support__"
        )
    )

    weighted = sorted(
        name
        for name in header
        if name.startswith(
            "semrel_weighted__"
        )
    )

    if len(support) != 155:
        raise SystemExit(
            "支持比例列不是155维："
            f"{len(support)}"
        )

    if len(weighted) != 155:
        raise SystemExit(
            "距离加权列不是155维："
            f"{len(weighted)}"
        )

    def context_role(name):
        parts = name.split("__")

        if len(parts) != 3:
            raise ValueError(
                "异常语义关系列："
                + name
            )

        return parts[2]

    support_nonanchor = [
        name
        for name in support
        if context_role(name)
        not in ANCHOR_ROLES
    ]

    weighted_nonanchor = [
        name
        for name in weighted
        if context_role(name)
        not in ANCHOR_ROLES
    ]

    if len(support_nonanchor) != 130:
        raise SystemExit(
            "五类外支持比例不是130维："
            f"{len(support_nonanchor)}"
        )

    if len(weighted_nonanchor) != 130:
        raise SystemExit(
            "五类外距离加权不是130维："
            f"{len(weighted_nonanchor)}"
        )

    return {
        "support155":
            support,

        "weighted155":
            weighted,

        "all310":
            support + weighted,

        "nonanchor_support130":
            support_nonanchor,

        "nonanchor_weighted130":
            weighted_nonanchor,

        "nonanchor260":
            (
                support_nonanchor
                + weighted_nonanchor
            ),
    }


def assert_2017_only(
    splits,
    compact_by_sha,
):
    selected = set()

    for split_name, parts in splits.items():

        train = parts["train"]
        test = parts["test"]

        if not train:
            raise SystemExit(
                f"{split_name}: "
                "训练集为空"
            )

        if not test:
            raise SystemExit(
                f"{split_name}: "
                "测试集为空"
            )

        train_shas = {
            row["sha256"]
            for row in train
        }

        test_shas = {
            row["sha256"]
            for row in test
        }

        overlap = (
            train_shas
            & test_shas
        )

        if overlap:
            raise SystemExit(
                f"{split_name}: "
                "训练测试 SHA 重叠="
                f"{len(overlap)}"
            )

        for partition_name, rows in (
            ("train", train),
            ("test", test),
        ):
            for row in rows:
                sha = row["sha256"]

                if sha not in compact_by_sha:
                    raise SystemExit(
                        f"{split_name}/"
                        f"{partition_name}: "
                        f"Compact 缺少 {sha}"
                    )

                year = str(
                    compact_by_sha[
                        sha
                    ].get(
                        "metadata_year",
                        "",
                    )
                ).strip()

                if year != "2017":
                    raise SystemExit(
                        "发现非2017样本，停止："
                        f"{split_name}/"
                        f"{partition_name}/"
                        f"{sha}/year={year}"
                    )

                selected.add(sha)

    return selected


def evaluate_matrix(
    split_name,
    train_rows,
    test_rows,
    y_train,
    y_test,
    x_train,
    x_test,
    variant,
    seeds,
    iterations,
    learning_rate,
    l2,
    prediction_rows=None,
):
    seed_metrics = []

    for seed in seeds:
        probability = predict_logistic(
            x_train,
            y_train,
            x_test,
            seed,
            iterations,
            learning_rate,
            l2,
        )

        m = metrics(
            y_test,
            probability,
            threshold=THRESHOLD,
        )

        seed_metrics.append(m)

        if prediction_rows is not None:
            for (
                split_row,
                label,
                prob,
            ) in zip(
                test_rows,
                y_test,
                probability,
            ):
                prediction_rows.append({
                    "variant":
                        variant,

                    "split_name":
                        split_name,

                    "sha256":
                        split_row[
                            "sha256"
                        ],

                    "seed":
                        seed,

                    "label":
                        int(label),

                    "probability":
                        float(prob),

                    "predicted_label":
                        int(
                            prob
                            >= THRESHOLD
                        ),
                })

    result = {
        "variant":
            variant,

        "split_name":
            split_name,

        "train_count":
            len(train_rows),

        "test_count":
            len(test_rows),

        "feature_count":
            int(
                x_train.shape[1]
            ),
    }

    for name in (
        "accuracy",
        "balanced_accuracy",
        "precision",
        "recall",
        "f1",
        "auc",
        "fp",
        "fn",
    ):
        result[
            name + "_mean"
        ] = mean_valid(
            [
                m[name]
                for m in seed_metrics
            ]
        )

    return result


def aggregate(
    rows,
):
    grouped = defaultdict(list)

    for row in rows:
        grouped[
            row["variant"]
        ].append(row)

    result = []

    for variant, current in (
        grouped.items()
    ):
        ba = np.asarray(
            [
                float(
                    row[
                        "balanced_accuracy_mean"
                    ]
                )
                for row in current
            ],
            dtype=float,
        )

        result.append({
            "variant":
                variant,

            "split_count":
                len(current),

            "mean_BA":
                float(
                    ba.mean()
                ),

            "min_BA":
                float(
                    ba.min()
                ),

            "std_BA":
                float(
                    ba.std()
                ),

            "mean_AUC":
                mean_valid([
                    row["auc_mean"]
                    for row in current
                ]),

            "mean_ACC":
                mean_valid([
                    row["accuracy_mean"]
                    for row in current
                ]),

            "mean_F1":
                mean_valid([
                    row["f1_mean"]
                    for row in current
                ]),

            "mean_FP":
                mean_valid([
                    row["fp_mean"]
                    for row in current
                ]),

            "mean_FN":
                mean_valid([
                    row["fn_mean"]
                    for row in current
                ]),
        })

    result.sort(
        key=lambda row: (
            row["mean_BA"],
            row["min_BA"],
            (
                row["mean_AUC"]
                if row["mean_AUC"]
                is not None
                else -1.0
            ),
        ),
        reverse=True,
    )

    return result


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--compact-features",
        type=Path,
        default=Path(
            "artifacts/formal-p1-features/"
            "features.csv"
        ),
    )

    parser.add_argument(
        "--compact-schema",
        type=Path,
        default=Path(
            "artifacts/formal-p1-features/"
            "feature_schema.json"
        ),
    )

    parser.add_argument(
        "--semantic-features",
        type=Path,
        default=Path(
            "artifacts/"
            "clean-local-context-2017/"
            "semantic_relation_features.csv.gz"
        ),
    )

    parser.add_argument(
        "--splits",
        type=Path,
        default=Path(
            "artifacts/"
            "p9-2017-rolling-splits/"
            "split_manifest.csv"
        ),
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "artifacts/"
            "clean-semantic-relation-2017-eval"
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

    for path in (
        args.compact_features,
        args.compact_schema,
        args.semantic_features,
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

    semantic_rows = read_gz_csv(
        args.semantic_features
    )

    semantic_by_sha = {
        str(
            row["sha256"]
        ).strip().casefold():
            row

        for row in semantic_rows
    }

    if not semantic_rows:
        raise SystemExit(
            "语义特征文件为空"
        )

    columns = semantic_columns(
        set(
            semantic_rows[0]
        )
    )

    splits = read_split_rows(
        args.splits
    )

    if not splits:
        raise SystemExit(
            "划分文件为空"
        )

    selected_shas = assert_2017_only(
        splits,
        compact_by_sha,
    )

    missing_semantic = (
        selected_shas
        - set(
            semantic_by_sha
        )
    )

    if missing_semantic:
        raise SystemExit(
            "语义特征缺少划分样本："
            f"{len(missing_semantic)}"
        )

    print(
        "======================================"
    )
    print(
        "2017 新语义源域开发实验"
    )
    print(
        "======================================"
    )

    print(
        "划分数 =",
        len(splits),
    )

    print(
        "参与 SHA =",
        len(selected_shas),
    )

    print(
        "阈值 =",
        THRESHOLD,
    )

    print(
        "L2 =",
        args.l2,
    )

    print(
        "随机种子 =",
        args.seeds,
    )

    print()
    print(
        "支持比例 =",
        len(
            columns[
                "support155"
            ]
        ),
    )

    print(
        "距离加权 =",
        len(
            columns[
                "weighted155"
            ]
        ),
    )

    print(
        "全部语义 =",
        len(
            columns[
                "all310"
            ]
        ),
    )

    print(
        "五类外全部语义 =",
        len(
            columns[
                "nonanchor260"
            ]
        ),
    )

    # ==================================================
    # 第一阶段：
    # 只用主方法：
    #
    # Compact + 五类外260维
    #
    # 在2017上选择唯一 alpha。
    # ==================================================
    alpha_rows = []

    print()
    print(
        "======================================"
    )
    print(
        "第一阶段：只为主方法选择一个缩放系数"
    )
    print(
        "======================================"
    )

    for alpha in ALPHAS:
        current_rows = []

        for split_name in sorted(
            splits
        ):
            train_rows = (
                splits[
                    split_name
                ]["train"]
            )

            test_rows = (
                splits[
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
                usable_semantic,
            ) = standardize_named(
                train_rows,
                test_rows,
                semantic_by_sha,
                columns[
                    "nonanchor260"
                ],
            )

            x_train = np.concatenate(
                [
                    compact_train,
                    alpha
                    * semantic_train,
                ],
                axis=1,
            )

            x_test = np.concatenate(
                [
                    compact_test,
                    alpha
                    * semantic_test,
                ],
                axis=1,
            )

            current_rows.append(
                evaluate_matrix(
                    split_name,
                    train_rows,
                    test_rows,
                    y_train,
                    y_test,
                    x_train,
                    x_test,
                    (
                        "compact_plus_"
                        "nonanchor260_"
                        f"a{alpha:g}"
                    ),
                    args.seeds,
                    args.iterations,
                    args.learning_rate,
                    args.l2,
                )
            )

        summary = aggregate(
            current_rows
        )[0]

        alpha_rows.append({
            "alpha":
                alpha,

            "mean_BA":
                summary[
                    "mean_BA"
                ],

            "min_BA":
                summary[
                    "min_BA"
                ],

            "mean_AUC":
                summary[
                    "mean_AUC"
                ],

            "mean_ACC":
                summary[
                    "mean_ACC"
                ],

            "mean_F1":
                summary[
                    "mean_F1"
                ],

            "mean_FP":
                summary[
                    "mean_FP"
                ],

            "mean_FN":
                summary[
                    "mean_FN"
                ],
        })

        print(
            f"alpha={alpha:g} "
            f"meanBA="
            f"{summary['mean_BA']:.6f} "
            f"minBA="
            f"{summary['min_BA']:.6f} "
            f"AUC="
            f"{summary['mean_AUC']:.6f}",
            flush=True,
        )

    alpha_rows.sort(
        key=lambda row: (
            float(
                row["mean_BA"]
            ),
            float(
                row["min_BA"]
            ),
            float(
                row["mean_AUC"]
            ),
            -float(
                row["alpha"]
            ),
        ),
        reverse=True,
    )

    selected_alpha = float(
        alpha_rows[0][
            "alpha"
        ]
    )

    print()
    print(
        "2017选择的唯一缩放系数 =",
        selected_alpha,
    )

    write_csv(
        args.output
        / "alpha_sweep.csv",
        alpha_rows,
    )

    # ==================================================
    # 第二阶段：
    # alpha 冻结。
    # 所有表示消融统一使用它。
    # ==================================================
    per_split_rows = []
    prediction_rows = []
    dimension_rows = []

    semantic_sets = (
        "support155",
        "weighted155",
        "all310",
        "nonanchor_support130",
        "nonanchor_weighted130",
        "nonanchor260",
    )

    print()
    print(
        "======================================"
    )
    print(
        "第二阶段：固定缩放系数后的表示消融"
    )
    print(
        "======================================"
    )

    for split_name in sorted(
        splits
    ):
        train_rows = (
            splits[
                split_name
            ]["train"]
        )

        test_rows = (
            splits[
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

        semantic_matrices = {}
        semantic_dims = {}

        for set_name in semantic_sets:
            (
                sem_train,
                sem_test,
                usable_names,
            ) = standardize_named(
                train_rows,
                test_rows,
                semantic_by_sha,
                columns[
                    set_name
                ],
            )

            semantic_matrices[
                set_name
            ] = (
                sem_train,
                sem_test,
            )

            semantic_dims[
                set_name
            ] = len(
                usable_names
            )

        matrices = {
            "compact_only":
                (
                    compact_train,
                    compact_test,
                ),
        }

        for set_name in semantic_sets:
            (
                sem_train,
                sem_test,
            ) = semantic_matrices[
                set_name
            ]

            matrices[
                set_name + "_only"
            ] = (
                sem_train,
                sem_test,
            )

            matrices[
                "compact_plus_"
                + set_name
            ] = (
                np.concatenate(
                    [
                        compact_train,
                        selected_alpha
                        * sem_train,
                    ],
                    axis=1,
                ),
                np.concatenate(
                    [
                        compact_test,
                        selected_alpha
                        * sem_test,
                    ],
                    axis=1,
                ),
            )

        dimension_row = {
            "split_name":
                split_name,

            "compact_dim":
                len(
                    compact_names
                ),
        }

        for set_name in (
            semantic_sets
        ):
            dimension_row[
                set_name + "_dim"
            ] = semantic_dims[
                set_name
            ]

        dimension_rows.append(
            dimension_row
        )

        for (
            variant,
            (
                x_train,
                x_test,
            ),
        ) in matrices.items():

            result = evaluate_matrix(
                split_name,
                train_rows,
                test_rows,
                y_train,
                y_test,
                x_train,
                x_test,
                variant,
                args.seeds,
                args.iterations,
                args.learning_rate,
                args.l2,
                prediction_rows,
            )

            per_split_rows.append(
                result
            )

        print(
            "完成",
            split_name,
            "Compact维数=",
            len(compact_names),
            "五类外260可用维数=",
            semantic_dims[
                "nonanchor260"
            ],
            flush=True,
        )

    summary_rows = aggregate(
        per_split_rows
    )

    write_csv(
        args.output
        / "per_split.csv",
        per_split_rows,
    )

    write_csv(
        args.output
        / "summary.csv",
        summary_rows,
    )

    write_csv(
        args.output
        / "predictions.csv",
        prediction_rows,
    )

    write_csv(
        args.output
        / "dimensions.csv",
        dimension_rows,
    )

    # ==================================================
    # Compact vs 主方法的逐样本平均概率诊断。
    # ==================================================
    main_variant = (
        "compact_plus_nonanchor260"
    )

    groups = defaultdict(list)
    meta = {}

    for row in prediction_rows:
        key = (
            row["split_name"],
            row["sha256"],
            row["variant"],
        )

        groups[key].append(
            float(
                row["probability"]
            )
        )

        meta[
            (
                row["split_name"],
                row["sha256"],
            )
        ] = int(
            row["label"]
        )

    paired_rows = []

    for (
        split_name,
        sha,
    ), label in sorted(
        meta.items()
    ):
        compact_key = (
            split_name,
            sha,
            "compact_only",
        )

        main_key = (
            split_name,
            sha,
            main_variant,
        )

        if (
            compact_key not in groups
            or main_key not in groups
        ):
            continue

        compact_prob = float(
            np.mean(
                groups[
                    compact_key
                ]
            )
        )

        main_prob = float(
            np.mean(
                groups[
                    main_key
                ]
            )
        )

        compact_pred = int(
            compact_prob
            >= THRESHOLD
        )

        main_pred = int(
            main_prob
            >= THRESHOLD
        )

        compact_ok = (
            compact_pred
            == label
        )

        main_ok = (
            main_pred
            == label
        )

        if (
            not compact_ok
            and main_ok
        ):
            transition = (
                "CORRECTED"
            )

        elif (
            compact_ok
            and not main_ok
        ):
            transition = (
                "REGRESSED"
            )

        elif compact_ok:
            transition = (
                "STILL_CORRECT"
            )

        else:
            transition = (
                "STILL_WRONG"
            )

        paired_rows.append({
            "split_name":
                split_name,

            "sha256":
                sha,

            "label":
                label,

            "compact_probability":
                compact_prob,

            "semantic_probability":
                main_prob,

            "probability_delta":
                (
                    main_prob
                    - compact_prob
                ),

            "compact_predicted_label":
                compact_pred,

            "semantic_predicted_label":
                main_pred,

            "transition":
                transition,
        })

    write_csv(
        args.output
        / "paired_compact_vs_main.csv",
        paired_rows,
    )

    counts = defaultdict(int)

    for row in paired_rows:
        counts[
            row["transition"]
        ] += 1

    print()
    print(
        "======================================"
    )
    print(
        "2017 源域开发结果"
    )
    print(
        "======================================"
    )

    print(
        "选择的缩放系数 =",
        selected_alpha,
    )

    print()
    print(
        "按平均平衡准确率排序："
    )

    for row in summary_rows:
        print(
            f"{row['variant']:40s} "
            f"BA={row['mean_BA']:.6f} "
            f"min={row['min_BA']:.6f} "
            f"AUC={row['mean_AUC']:.6f} "
            f"ACC={row['mean_ACC']:.6f} "
            f"FP={row['mean_FP']:.2f} "
            f"FN={row['mean_FN']:.2f}"
        )

    print()
    print(
        "主方法相对 Compact："
    )

    print(
        "  修正 =",
        counts[
            "CORRECTED"
        ]
    )

    print(
        "  新增错误 =",
        counts[
            "REGRESSED"
        ]
    )

    print(
        "  净修正 =",
        (
            counts["CORRECTED"]
            - counts["REGRESSED"]
        ),
    )

    print(
        "  仍正确 =",
        counts[
            "STILL_CORRECT"
        ],
    )

    print(
        "  仍错误 =",
        counts[
            "STILL_WRONG"
        ],
    )

    print()
    print(
        "保存目录：",
        args.output,
    )


if __name__ == "__main__":
    main()
