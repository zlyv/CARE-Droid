#!/usr/bin/env python3
from __future__ import annotations

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

from run_pilot_compact_eval import (
    fit_logistic,
    sigmoid,
    metrics,
)

from run_compact_20_150_ablation_2017 import (
    support150_names,
)


ALPHA = 0.25

# 具体关系残差只保留0.5倍输入尺度。
# 在相同L2下，相当于原空间残差系数受到约4倍更强收缩。
RESIDUAL_SCALE = 0.50

THRESHOLD = 0.5
SEEDS = [0, 1, 2, 3, 4]

ITERATIONS = 6000
LEARNING_RATE = 0.05
L2 = 0.03


COMPACT_PATH = Path(
    "artifacts/p1-features/features.csv"
)

SCHEMA_PATH = Path(
    "artifacts/p1-features/feature_schema.json"
)

SEMANTIC_PATH = Path(
    "artifacts/clean-local-context-2017/"
    "semantic_relation_features.csv.gz"
)

SPLIT_PATH = Path(
    "artifacts/p9-2017-rolling-splits/"
    "split_manifest.csv"
)

OUTPUT_DIR = Path(
    "artifacts/structured-shared-semantic-2017"
)


def read_gz(path):
    with gzip.open(
        path,
        "rt",
        encoding="utf-8",
        newline="",
    ) as f:
        return list(csv.DictReader(f))


def write_csv(path, rows):
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
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fields,
        )
        writer.writeheader()
        writer.writerows(rows)


def parse_relation_name(name):
    parts = name.split("__")

    if len(parts) != 3:
        raise ValueError(
            "异常关系名：" + name
        )

    return (
        parts[1],
        parts[2],
    )


def build_shared_features(
    semantic_train,
    semantic_test,
    semantic_names,
    anchor_order,
    context_order,
):
    """
    semantic_train / test 是已经按训练集
    标准化后的具体关系特征。

    对每个锚点：
        A_i = sum_j S_ij

    对每个上下文：
        C_j = sum_i S_ij

    因此：
        sum_i a_i A_i + sum_j c_j C_j

    等价于：
        sum_ij S_ij (a_i + c_j)

    这正是“锚点作用 + 上下文作用”的
    结构化关系权重。
    """

    parsed = [
        parse_relation_name(name)
        for name in semantic_names
    ]

    train_columns = []
    test_columns = []
    feature_names = []

    # -------------------------
    # 锚点共享作用
    # -------------------------

    for anchor in anchor_order:
        indices = [
            idx
            for idx, (
                current_anchor,
                _,
            ) in enumerate(parsed)
            if current_anchor == anchor
        ]

        if not indices:
            continue

        train_col = (
            semantic_train[
                :,
                indices,
            ].sum(axis=1)
        )

        test_col = (
            semantic_test[
                :,
                indices,
            ].sum(axis=1)
        )

        # 理论上只要存在可用关系，
        # 这一列通常不会恒定。
        # 仍做安全检查。
        if np.std(train_col) <= 1e-12:
            continue

        train_columns.append(
            train_col
        )

        test_columns.append(
            test_col
        )

        feature_names.append(
            "shared_anchor__"
            + anchor
        )

    # -------------------------
    # 上下文共享作用
    # -------------------------

    for context in context_order:
        indices = [
            idx
            for idx, (
                _,
                current_context,
            ) in enumerate(parsed)
            if current_context == context
        ]

        if not indices:
            continue

        train_col = (
            semantic_train[
                :,
                indices,
            ].sum(axis=1)
        )

        test_col = (
            semantic_test[
                :,
                indices,
            ].sum(axis=1)
        )

        if np.std(train_col) <= 1e-12:
            continue

        train_columns.append(
            train_col
        )

        test_columns.append(
            test_col
        )

        feature_names.append(
            "shared_context__"
            + context
        )

    if not train_columns:
        raise RuntimeError(
            "没有生成共享结构特征"
        )

    x_train = np.column_stack(
        train_columns
    )

    x_test = np.column_stack(
        test_columns
    )

    return (
        x_train,
        x_test,
        feature_names,
    )


def evaluate(
    variant,
    split_name,
    seed,
    x_train,
    y_train,
    x_test,
    y_test,
):
    w, b = fit_logistic(
        x_train,
        y_train,
        seed,
        iterations=ITERATIONS,
        learning_rate=LEARNING_RATE,
        l2=L2,
    )

    probability = sigmoid(
        x_test @ w + b
    )

    result = metrics(
        y_test,
        probability,
        threshold=THRESHOLD,
    )

    return {
        "variant":
            variant,

        "split_name":
            split_name,

        "seed":
            seed,

        **result,
    }


def summarize(metric_rows):
    groups = defaultdict(list)

    for row in metric_rows:
        groups[
            (
                row["variant"],
                row["split_name"],
            )
        ].append(row)

    per_split = []

    for (
        variant,
        split_name,
    ), rows in sorted(groups.items()):

        out = {
            "variant":
                variant,

            "split_name":
                split_name,

            "seed_count":
                len(rows),
        }

        for field in (
            "accuracy",
            "balanced_accuracy",
            "f1",
            "auc",
            "fp",
            "fn",
        ):
            out[
                field + "_mean"
            ] = float(
                np.mean([
                    float(row[field])
                    for row in rows
                ])
            )

        per_split.append(out)

    by_variant = defaultdict(list)

    for row in per_split:
        by_variant[
            row["variant"]
        ].append(row)

    summary = []

    for variant, rows in (
        by_variant.items()
    ):
        ba = np.asarray(
            [
                row[
                    "balanced_accuracy_mean"
                ]
                for row in rows
            ],
            dtype=float,
        )

        summary.append({
            "variant":
                variant,

            "mean_BA":
                float(ba.mean()),

            "min_BA":
                float(ba.min()),

            "std_BA":
                float(ba.std()),

            "mean_AUC":
                float(np.mean([
                    row["auc_mean"]
                    for row in rows
                ])),

            "mean_ACC":
                float(np.mean([
                    row["accuracy_mean"]
                    for row in rows
                ])),

            "mean_F1":
                float(np.mean([
                    row["f1_mean"]
                    for row in rows
                ])),

            "mean_FP":
                float(np.mean([
                    row["fp_mean"]
                    for row in rows
                ])),

            "mean_FN":
                float(np.mean([
                    row["fn_mean"]
                    for row in rows
                ])),
        })

    summary.sort(
        key=lambda row: (
            row["mean_BA"],
            row["min_BA"],
            row["mean_AUC"],
        ),
        reverse=True,
    )

    return (
        per_split,
        summary,
    )


def main():
    for path in (
        COMPACT_PATH,
        SCHEMA_PATH,
        SEMANTIC_PATH,
        SPLIT_PATH,
    ):
        if not path.exists():
            raise SystemExit(
                "文件不存在："
                + str(path)
            )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    compact_rows = read_csv_rows(
        COMPACT_PATH
    )

    compact_by_sha = {
        str(
            row["sha256"]
        ).strip().casefold():
            row
        for row in compact_rows
    }

    schema = json.loads(
        SCHEMA_PATH.read_text(
            encoding="utf-8"
        )
    )

    semantic_rows = read_gz(
        SEMANTIC_PATH
    )

    semantic_by_sha = {
        str(
            row["sha256"]
        ).strip().casefold():
            row
        for row in semantic_rows
    }

    names150 = support150_names(
        semantic_rows[0].keys()
    )

    parsed150 = [
        parse_relation_name(name)
        for name in names150
    ]

    anchor_order = sorted({
        anchor
        for anchor, _ in parsed150
    })

    context_order = sorted({
        context
        for _, context in parsed150
    })

    print(
        "锚点数量 =",
        len(anchor_order),
    )

    print(
        "上下文角色全集数量 =",
        len(context_order),
    )

    print(
        "关系数量 =",
        len(names150),
    )

    print()
    print(
        "锚点："
    )

    for name in anchor_order:
        print(
            " ",
            name,
        )

    splits = read_split_rows(
        SPLIT_PATH
    )

    all_shas = set()

    for parts in splits.values():
        for partition in (
            "train",
            "test",
        ):
            for row in parts[
                partition
            ]:
                all_shas.add(
                    row["sha256"]
                )

    missing_compact = (
        all_shas
        - set(compact_by_sha)
    )

    missing_semantic = (
        all_shas
        - set(semantic_by_sha)
    )

    print()
    print(
        "======================================"
    )
    print(
        "人口检查"
    )
    print(
        "======================================"
    )

    print(
        "参与SHA =",
        len(all_shas),
    )

    print(
        "Compact缺失 =",
        len(missing_compact),
    )

    print(
        "150维缺失 =",
        len(missing_semantic),
    )

    if missing_compact:
        raise SystemExit(
            "Compact人口不完整"
        )

    if missing_semantic:
        raise SystemExit(
            "150维人口不完整"
        )

    print()
    print(
        "======================================"
    )
    print(
        "2017结构化语义权重共享"
    )
    print(
        "======================================"
    )

    print(
        "alpha =",
        ALPHA,
    )

    print(
        "关系残差缩放 =",
        RESIDUAL_SCALE,
    )

    print(
        "残差有效L2倍率 ≈",
        1.0
        / (
            RESIDUAL_SCALE
            ** 2
        ),
    )

    print(
        "threshold =",
        THRESHOLD,
    )

    print(
        "iterations =",
        ITERATIONS,
    )

    print(
        "L2 =",
        L2,
    )

    metric_rows = []
    dimension_rows = []

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
            schema,
        )

        (
            semantic_train,
            semantic_test,
            semantic_names,
        ) = standardize_named(
            train_rows,
            test_rows,
            semantic_by_sha,
            names150,
        )

        (
            shared_train,
            shared_test,
            shared_names,
        ) = build_shared_features(
            semantic_train,
            semantic_test,
            semantic_names,
            anchor_order,
            context_order,
        )

        # ----------------------------------
        # 普通150
        # ----------------------------------

        raw_train = np.concatenate(
            [
                compact_train,
                ALPHA
                * semantic_train,
            ],
            axis=1,
        )

        raw_test = np.concatenate(
            [
                compact_test,
                ALPHA
                * semantic_test,
            ],
            axis=1,
        )

        # ----------------------------------
        # 只有共享主效应
        #
        # 关系权重：
        # w_ij = a_i + c_j
        # ----------------------------------

        shared_only_train = (
            np.concatenate(
                [
                    compact_train,
                    ALPHA
                    * shared_train,
                ],
                axis=1,
            )
        )

        shared_only_test = (
            np.concatenate(
                [
                    compact_test,
                    ALPHA
                    * shared_test,
                ],
                axis=1,
            )
        )

        # ----------------------------------
        # 共享主效应 + 强收缩具体残差
        #
        # w_ij =
        # a_i + c_j + r_ij
        #
        # residual_scale=0.5
        # => r_ij约4倍更强L2
        # ----------------------------------

        hierarchical_train = (
            np.concatenate(
                [
                    compact_train,

                    ALPHA
                    * shared_train,

                    ALPHA
                    * RESIDUAL_SCALE
                    * semantic_train,
                ],
                axis=1,
            )
        )

        hierarchical_test = (
            np.concatenate(
                [
                    compact_test,

                    ALPHA
                    * shared_test,

                    ALPHA
                    * RESIDUAL_SCALE
                    * semantic_test,
                ],
                axis=1,
            )
        )

        dimension_rows.append({
            "split_name":
                split_name,

            "train_n":
                len(train_rows),

            "test_n":
                len(test_rows),

            "compact_dim":
                len(compact_names),

            "relation_usable_dim":
                len(semantic_names),

            "shared_dim":
                len(shared_names),

            "raw_total_dim":
                raw_train.shape[1],

            "shared_only_total_dim":
                shared_only_train.shape[1],

            "hierarchical_total_dim":
                hierarchical_train.shape[1],
        })

        print()
        print(
            split_name
        )

        print(
            " train=",
            len(train_rows),
            "test=",
            len(test_rows),
            "Compact=",
            len(compact_names),
            "关系可用=",
            len(semantic_names),
            "共享参数特征=",
            len(shared_names),
        )

        variants = (
            (
                "compact_only",
                compact_train,
                compact_test,
            ),

            (
                "compact_plus_raw150",
                raw_train,
                raw_test,
            ),

            (
                "compact_plus_shared_only",
                shared_only_train,
                shared_only_test,
            ),

            (
                "compact_plus_shared_residual",
                hierarchical_train,
                hierarchical_test,
            ),
        )

        for seed in SEEDS:
            for (
                variant,
                x_train,
                x_test,
            ) in variants:

                metric_rows.append(
                    evaluate(
                        variant,
                        split_name,
                        seed,
                        x_train,
                        y_train,
                        x_test,
                        y_test,
                    )
                )

        print(
            "完成",
            flush=True,
        )

    (
        per_split,
        summary,
    ) = summarize(
        metric_rows
    )

    write_csv(
        OUTPUT_DIR
        / "per_seed_metrics.csv",
        metric_rows,
    )

    write_csv(
        OUTPUT_DIR
        / "per_split.csv",
        per_split,
    )

    write_csv(
        OUTPUT_DIR
        / "summary.csv",
        summary,
    )

    write_csv(
        OUTPUT_DIR
        / "dimensions.csv",
        dimension_rows,
    )

    print()
    print(
        "======================================"
    )
    print(
        "2017结果"
    )
    print(
        "======================================"
    )

    for row in summary:
        print(
            f"{row['variant']:32s} "
            f"BA={row['mean_BA']:.6f} "
            f"min={row['min_BA']:.6f} "
            f"std={row['std_BA']:.6f} "
            f"AUC={row['mean_AUC']:.6f} "
            f"ACC={row['mean_ACC']:.6f} "
            f"FP={row['mean_FP']:.2f} "
            f"FN={row['mean_FN']:.2f}"
        )

    lookup = {
        row["variant"]:
            row
        for row in summary
    }

    raw = lookup[
        "compact_plus_raw150"
    ]

    print()
    print(
        "======================================"
    )
    print(
        "相对普通150"
    )
    print(
        "======================================"
    )

    for variant in (
        "compact_plus_shared_only",
        "compact_plus_shared_residual",
    ):
        row = lookup[
            variant
        ]

        print(
            f"{variant:32s} "
            f"dBA="
            f"{row['mean_BA'] - raw['mean_BA']:+.6f} "
            f"dMin="
            f"{row['min_BA'] - raw['min_BA']:+.6f} "
            f"dStd="
            f"{row['std_BA'] - raw['std_BA']:+.6f} "
            f"dAUC="
            f"{row['mean_AUC'] - raw['mean_AUC']:+.6f}"
        )

    print()
    print(
        "历史参考："
    )

    print(
        "普通150："
        "BA≈0.906712 "
        "min≈0.890250 "
        "AUC≈0.950344"
    )

    print(
        "20+150："
        "BA≈0.909085 "
        "min≈0.903948 "
        "AUC≈0.952079"
    )


if __name__ == "__main__":
    main()
