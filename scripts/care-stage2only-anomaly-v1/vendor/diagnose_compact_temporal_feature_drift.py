#!/usr/bin/env python3

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import run_frozen_anchor_residual_2017_2021 as base

from run_pilot_compact_eval import fit_logistic


COMPACT_FEATURES = Path(
    "artifacts/p1-features/features.csv"
)

COMPACT_SCHEMA = Path(
    "artifacts/p1-features/feature_schema.json"
)

FUTURE_SPLITS = Path(
    "artifacts/"
    "p1-splits-fixed-2017-test-2018-2021/"
    "split_manifest.csv"
)

OUT = Path(
    "artifacts/"
    "compact-temporal-feature-drift"
)

YEARS = (
    2017,
    2018,
    2019,
    2020,
    2021,
)

FUTURE_YEARS = (
    2018,
    2019,
    2020,
    2021,
)

THRESHOLD = 0.5


def feature_family(name):
    if name.startswith("role_node__"):
        return "role_node"

    if name.startswith("role_edge__"):
        return "role_edge"

    if name.startswith("context_role__"):
        return "context_role"

    if name.startswith("manifest_"):
        return "manifest"

    if name in (
        "min_sdk",
        "target_sdk",
    ):
        return "sdk"

    if name in (
        "method_count",
        "graph_node_count",
        "graph_edge_count",
        "internal_node_count",
        "internal_edge_count",
        "external_node_count",
        "external_edge_count",
        "mapped_api_node_count",
        "unmapped_external_node_count",
        "sensitive_node_count",
        "sensitive_edge_count",
        "manifest_context_match_count",
    ):
        return "program_or_graph_size"

    if name.endswith("_ratio"):
        return "ratio"

    return "other"


def is_count_like(name):
    if (
        name.endswith("_count")
        or name.startswith("role_node__")
        or name.startswith("role_edge__")
        or name.startswith("context_role__")
    ):
        return 1

    return 0


def safe_auc(y, x):
    y = np.asarray(
        y,
        dtype=int,
    )

    x = np.asarray(
        x,
        dtype=float,
    )

    if len(np.unique(y)) < 2:
        return np.nan

    if np.allclose(
        x,
        x[0],
    ):
        return 0.5

    return float(
        roc_auc_score(
            y,
            x,
        )
    )


def sigmoid(z):
    z = np.clip(
        np.asarray(z, dtype=float),
        -60.0,
        60.0,
    )

    return (
        1.0
        /
        (
            1.0
            + np.exp(-z)
        )
    )


def get_rows_for_year(
    future_parts,
    year,
):
    if year == 2017:
        name = (
            "train_fixed_2017__test_2018"
        )

        if name not in future_parts:
            raise SystemExit(
                "找不到固定2017训练人口"
            )

        return list(
            future_parts[
                name
            ][
                "train"
            ]
        )

    name = (
        f"train_fixed_2017__test_{year}"
    )

    if name not in future_parts:
        raise SystemExit(
            "找不到划分: "
            + name
        )

    return list(
        future_parts[
            name
        ][
            "test"
        ]
    )


def main():
    OUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    import json

    compact_by_sha = (
        base.normalized_rows(
            base.read_csv_rows(
                COMPACT_FEATURES
            )
        )
    )

    compact_schema = json.loads(
        COMPACT_SCHEMA.read_text(
            encoding="utf-8"
        )
    )

    future_parts = (
        base.read_split_rows(
            FUTURE_SPLITS
        )
    )

    rows_by_year = {
        year:
            get_rows_for_year(
                future_parts,
                year,
            )
        for year in YEARS
    }

    # ========================================================
    # 检查四个future split中的2017训练人口完全一致。
    # ========================================================

    source_sets = []

    for year in FUTURE_YEARS:
        name = (
            f"train_fixed_2017__test_{year}"
        )

        shas = {
            str(
                r[
                    "sha256"
                ]
            )
            .strip()
            .casefold()
            for r in future_parts[
                name
            ][
                "train"
            ]
        }

        source_sets.append(
            shas
        )

    if not all(
        x == source_sets[0]
        for x in source_sets[1:]
    ):
        raise SystemExit(
            "四个future split中的2017训练人口不一致"
        )

    train2017 = (
        rows_by_year[
            2017
        ]
    )

    y2017 = (
        base.labels_for(
            train2017,
            compact_by_sha,
        )
        .astype(int)
    )

    # ========================================================
    # 用2017训练人口本身确定真正进入Compact的维度。
    # ========================================================

    (
        x2017,
        x2017_test,
        compact_names,
    ) = base.compact_matrix_named(
        train2017,
        train2017,
        compact_by_sha,
        compact_schema,
    )

    if not np.allclose(
        x2017,
        x2017_test,
    ):
        raise SystemExit(
            "2017自身标准化结果不一致"
        )

    print(
        "Compact实际维度 =",
        len(
            compact_names
        ),
    )

    if len(
        compact_names
    ) != 76:
        raise SystemExit(
            "预期Compact为76维，实际="
            + str(
                len(
                    compact_names
                )
            )
        )

    print()
    print(
        "实际76维："
    )

    for i, name in enumerate(
        compact_names
    ):
        print(
            f"{i:2d}",
            name,
        )

    # ========================================================
    # 所有年份都严格使用2017训练均值/标准差。
    #
    # 这样：
    # 1. 与2017->未来正式模型的输入空间一致
    # 2. 能直接观察未来数据相对2017坐标系怎样移动
    # ========================================================

    x_by_year = {
        2017:
            x2017,
    }

    y_by_year = {
        2017:
            y2017,
    }

    for year in FUTURE_YEARS:
        (
            x_source_check,
            x_future,
            names_check,
        ) = base.compact_matrix_named(
            train2017,
            rows_by_year[
                year
            ],
            compact_by_sha,
            compact_schema,
        )

        if list(
            names_check
        ) != list(
            compact_names
        ):
            raise SystemExit(
                f"{year} Compact列发生变化"
            )

        if not np.allclose(
            x_source_check,
            x2017,
        ):
            raise SystemExit(
                f"{year} 的2017训练矩阵不一致"
            )

        x_by_year[
            year
        ] = x_future

        y_by_year[
            year
        ] = (
            base.labels_for(
                rows_by_year[
                    year
                ],
                compact_by_sha,
            )
            .astype(int)
        )

    # ========================================================
    # 训练正式2017 Compact。
    # ========================================================

    weight, bias = fit_logistic(
        x2017,
        y2017,
        0,
        iterations=6000,
        learning_rate=0.05,
        l2=0.03,
    )

    weight = np.asarray(
        weight,
        dtype=float,
    ).reshape(-1)

    bias = float(
        np.asarray(
            bias
        ).reshape(-1)[0]
    )

    if len(
        weight
    ) != len(
        compact_names
    ):
        raise SystemExit(
            "逻辑回归权重维数异常"
        )

    # ========================================================
    # 首先检查固定2017模型逐年实际表现。
    # ========================================================

    model_year_rows = []

    for year in YEARS:
        x = x_by_year[
            year
        ]

        y = y_by_year[
            year
        ]

        logit = (
            x @ weight
            + bias
        )

        probability = sigmoid(
            logit
        )

        m = base.metrics(
            y,
            probability,
            threshold=THRESHOLD,
        )

        benign = (
            y == 0
        )

        malware = (
            y == 1
        )

        model_year_rows.append({
            "year":
                year,

            "N":
                len(
                    y
                ),

            "benign_n":
                int(
                    benign.sum()
                ),

            "malware_n":
                int(
                    malware.sum()
                ),

            "BA":
                float(
                    m[
                        "balanced_accuracy"
                    ]
                ),

            "AUC":
                float(
                    m[
                        "auc"
                    ]
                ),

            "ACC":
                float(
                    m[
                        "accuracy"
                    ]
                ),

            "FP":
                int(
                    m[
                        "fp"
                    ]
                ),

            "FN":
                int(
                    m[
                        "fn"
                    ]
                ),

            "benign_mean_logit":
                float(
                    np.mean(
                        logit[
                            benign
                        ]
                    )
                ),

            "malware_mean_logit":
                float(
                    np.mean(
                        logit[
                            malware
                        ]
                    )
                ),

            "benign_mean_probability":
                float(
                    np.mean(
                        probability[
                            benign
                        ]
                    )
                ),

            "malware_mean_probability":
                float(
                    np.mean(
                        probability[
                            malware
                        ]
                    )
                ),
        })

    model_year_df = pd.DataFrame(
        model_year_rows
    )

    model_year_df.to_csv(
        OUT
        / "fixed_2017_compact_by_year.csv",
        index=False,
    )

    # ========================================================
    # 每个特征、每个年份统计。
    # ========================================================

    long_rows = []

    for j, name in enumerate(
        compact_names
    ):
        for year in YEARS:
            x = x_by_year[
                year
            ][
                :,
                j
            ]

            y = y_by_year[
                year
            ]

            xb = x[
                y == 0
            ]

            xm = x[
                y == 1
            ]

            auc = safe_auc(
                y,
                x,
            )

            long_rows.append({
                "feature":
                    name,

                "family":
                    feature_family(
                        name
                    ),

                "count_like":
                    is_count_like(
                        name
                    ),

                "year":
                    year,

                "weight_2017":
                    float(
                        weight[
                            j
                        ]
                    ),

                "abs_weight_2017":
                    float(
                        abs(
                            weight[
                                j
                            ]
                        )
                    ),

                "benign_mean":
                    float(
                        np.mean(
                            xb
                        )
                    ),

                "benign_median":
                    float(
                        np.median(
                            xb
                        )
                    ),

                "benign_std":
                    float(
                        np.std(
                            xb
                        )
                    ),

                "malware_mean":
                    float(
                        np.mean(
                            xm
                        )
                    ),

                "malware_median":
                    float(
                        np.median(
                            xm
                        )
                    ),

                "malware_std":
                    float(
                        np.std(
                            xm
                        )
                    ),

                "class_mean_gap":
                    float(
                        np.mean(
                            xm
                        )
                        -
                        np.mean(
                            xb
                        )
                    ),

                "auc_raw":
                    auc,

                "auc_discriminative":
                    float(
                        max(
                            auc,
                            1.0 - auc,
                        )
                    ),
            })

    long_df = pd.DataFrame(
        long_rows
    )

    # ========================================================
    # 以2017方向为基准，计算未来方向保持情况。
    # ========================================================

    source = (
        long_df[
            long_df[
                "year"
            ] == 2017
        ]
        .set_index(
            "feature"
        )
    )

    long_df[
        "source_direction"
    ] = 0

    long_df[
        "auc_oriented_to_2017"
    ] = np.nan

    long_df[
        "direction_flip_vs_2017"
    ] = 0

    long_df[
        "benign_shift_vs_2017"
    ] = np.nan

    long_df[
        "malware_shift_vs_2017"
    ] = np.nan

    long_df[
        "common_shift_vs_2017"
    ] = np.nan

    long_df[
        "class_gap_change_vs_2017"
    ] = np.nan

    long_df[
        "benign_logit_shift_contribution"
    ] = np.nan

    long_df[
        "malware_logit_shift_contribution"
    ] = np.nan

    for idx, row in long_df.iterrows():
        name = row[
            "feature"
        ]

        s = source.loc[
            name
        ]

        source_auc = float(
            s[
                "auc_raw"
            ]
        )

        source_direction = (
            1
            if source_auc >= 0.5
            else -1
        )

        year_auc = float(
            row[
                "auc_raw"
            ]
        )

        oriented_auc = (
            year_auc
            if source_direction == 1
            else 1.0 - year_auc
        )

        flip = int(
            oriented_auc < 0.5
        )

        benign_shift = (
            float(
                row[
                    "benign_mean"
                ]
            )
            -
            float(
                s[
                    "benign_mean"
                ]
            )
        )

        malware_shift = (
            float(
                row[
                    "malware_mean"
                ]
            )
            -
            float(
                s[
                    "malware_mean"
                ]
            )
        )

        common_shift = (
            0.5
            * (
                benign_shift
                + malware_shift
            )
        )

        gap_change = (
            float(
                row[
                    "class_mean_gap"
                ]
            )
            -
            float(
                s[
                    "class_mean_gap"
                ]
            )
        )

        w = float(
            row[
                "weight_2017"
            ]
        )

        long_df.at[
            idx,
            "source_direction"
        ] = source_direction

        long_df.at[
            idx,
            "auc_oriented_to_2017"
        ] = oriented_auc

        long_df.at[
            idx,
            "direction_flip_vs_2017"
        ] = flip

        long_df.at[
            idx,
            "benign_shift_vs_2017"
        ] = benign_shift

        long_df.at[
            idx,
            "malware_shift_vs_2017"
        ] = malware_shift

        long_df.at[
            idx,
            "common_shift_vs_2017"
        ] = common_shift

        long_df.at[
            idx,
            "class_gap_change_vs_2017"
        ] = gap_change

        long_df.at[
            idx,
            "benign_logit_shift_contribution"
        ] = (
            w
            * benign_shift
        )

        long_df.at[
            idx,
            "malware_logit_shift_contribution"
        ] = (
            w
            * malware_shift
        )

    long_df.to_csv(
        OUT
        / "feature_year_stats.csv",
        index=False,
    )

    # ========================================================
    # 汇总成一行一个特征。
    # ========================================================

    summary_rows = []

    for name in compact_names:
        g = (
            long_df[
                long_df[
                    "feature"
                ] == name
            ]
            .set_index(
                "year"
            )
        )

        s2017 = g.loc[
            2017
        ]

        future = g.loc[
            list(
                FUTURE_YEARS
            )
        ]

        source_auc_strength = (
            float(
                s2017[
                    "auc_discriminative"
                ]
            )
        )

        future_disc_mean = float(
            future[
                "auc_discriminative"
            ].mean()
        )

        future_oriented_mean = float(
            future[
                "auc_oriented_to_2017"
            ].mean()
        )

        flip_rate = float(
            future[
                "direction_flip_vs_2017"
            ].mean()
        )

        relation_loss = max(
            0.0,
            source_auc_strength
            - future_oriented_mean,
        )

        mean_abs_common_shift = float(
            np.mean(
                np.abs(
                    future[
                        "common_shift_vs_2017"
                    ].to_numpy(
                        dtype=float
                    )
                )
            )
        )

        mean_abs_gap_change = float(
            np.mean(
                np.abs(
                    future[
                        "class_gap_change_vs_2017"
                    ].to_numpy(
                        dtype=float
                    )
                )
            )
        )

        abs_weight = float(
            s2017[
                "abs_weight_2017"
            ]
        )

        # ----------------------------------------------------
        # A：
        # 2017权重大，而且标签关系后来明显失效/反转。
        # ----------------------------------------------------

        relationship_drift_score = (
            abs_weight
            * (
                relation_loss
                + 0.5
                * flip_rate
            )
        )

        # ----------------------------------------------------
        # B：
        # 后期仍有单特征区分能力，
        # 但两类整体共同移动。
        #
        # 这类特征最适合考虑比例化/归一化，
        # 因为“信息还在，坐标位置变了”。
        # ----------------------------------------------------

        future_strength_above_random = max(
            0.0,
            future_disc_mean
            - 0.5,
        )

        common_shift_score = (
            abs_weight
            * mean_abs_common_shift
            * future_strength_above_random
        )

        summary_rows.append({
            "feature":
                name,

            "family":
                s2017[
                    "family"
                ],

            "count_like":
                int(
                    s2017[
                        "count_like"
                    ]
                ),

            "weight_2017":
                float(
                    s2017[
                        "weight_2017"
                    ]
                ),

            "abs_weight_2017":
                abs_weight,

            "auc_2017":
                float(
                    s2017[
                        "auc_raw"
                    ]
                ),

            "disc_auc_2017":
                source_auc_strength,

            "disc_auc_2018":
                float(
                    g.loc[
                        2018,
                        "auc_discriminative"
                    ]
                ),

            "disc_auc_2019":
                float(
                    g.loc[
                        2019,
                        "auc_discriminative"
                    ]
                ),

            "disc_auc_2020":
                float(
                    g.loc[
                        2020,
                        "auc_discriminative"
                    ]
                ),

            "disc_auc_2021":
                float(
                    g.loc[
                        2021,
                        "auc_discriminative"
                    ]
                ),

            "oriented_auc_2018":
                float(
                    g.loc[
                        2018,
                        "auc_oriented_to_2017"
                    ]
                ),

            "oriented_auc_2019":
                float(
                    g.loc[
                        2019,
                        "auc_oriented_to_2017"
                    ]
                ),

            "oriented_auc_2020":
                float(
                    g.loc[
                        2020,
                        "auc_oriented_to_2017"
                    ]
                ),

            "oriented_auc_2021":
                float(
                    g.loc[
                        2021,
                        "auc_oriented_to_2017"
                    ]
                ),

            "future_disc_auc_mean":
                future_disc_mean,

            "future_oriented_auc_mean":
                future_oriented_mean,

            "direction_flip_rate":
                flip_rate,

            "mean_abs_common_shift":
                mean_abs_common_shift,

            "mean_abs_gap_change":
                mean_abs_gap_change,

            "relationship_drift_score":
                relationship_drift_score,

            "common_shift_score":
                common_shift_score,

            "benign_shift_2018":
                float(
                    g.loc[
                        2018,
                        "benign_shift_vs_2017"
                    ]
                ),

            "benign_shift_2019":
                float(
                    g.loc[
                        2019,
                        "benign_shift_vs_2017"
                    ]
                ),

            "benign_shift_2020":
                float(
                    g.loc[
                        2020,
                        "benign_shift_vs_2017"
                    ]
                ),

            "benign_shift_2021":
                float(
                    g.loc[
                        2021,
                        "benign_shift_vs_2017"
                    ]
                ),

            "malware_shift_2021":
                float(
                    g.loc[
                        2021,
                        "malware_shift_vs_2017"
                    ]
                ),

            "benign_logit_shift_2021":
                float(
                    g.loc[
                        2021,
                        "benign_logit_shift_contribution"
                    ]
                ),

            "malware_logit_shift_2021":
                float(
                    g.loc[
                        2021,
                        "malware_logit_shift_contribution"
                    ]
                ),
        })

    summary_df = pd.DataFrame(
        summary_rows
    )

    summary_df.to_csv(
        OUT
        / "feature_summary.csv",
        index=False,
    )

    # ========================================================
    # 排行A：
    # 高权重 + 标签关系失效。
    # ========================================================

    relationship_ranked = (
        summary_df
        .sort_values(
            [
                "relationship_drift_score",
                "abs_weight_2017",
            ],
            ascending=[
                False,
                False,
            ],
        )
        .reset_index(
            drop=True
        )
    )

    relationship_ranked.to_csv(
        OUT
        / "ranking_relationship_drift.csv",
        index=False,
    )

    # ========================================================
    # 排行B：
    # 后期仍有区分能力 + 整体位置漂移。
    # ========================================================

    common_ranked = (
        summary_df
        .sort_values(
            [
                "common_shift_score",
                "future_disc_auc_mean",
            ],
            ascending=[
                False,
                False,
            ],
        )
        .reset_index(
            drop=True
        )
    )

    common_ranked.to_csv(
        OUT
        / "ranking_common_shift.csv",
        index=False,
    )

    # ========================================================
    # 最适合尝试比例化的计数类候选。
    # ========================================================

    ratio_candidates = (
        summary_df[
            summary_df[
                "count_like"
            ] == 1
        ]
        .sort_values(
            [
                "common_shift_score",
                "future_disc_auc_mean",
            ],
            ascending=[
                False,
                False,
            ],
        )
        .reset_index(
            drop=True
        )
    )

    ratio_candidates.to_csv(
        OUT
        / "ranking_ratio_candidates.csv",
        index=False,
    )

    # ========================================================
    # 2021良性样本：
    # 哪些特征把2017模型的logit往“恶意”方向推得最多？
    #
    # 正值 = 相对2017良性均值，
    #        把2021良性平均分数往恶意方向推。
    # ========================================================

    benign2021 = (
        long_df[
            long_df[
                "year"
            ] == 2021
        ][
            [
                "feature",
                "family",
                "weight_2017",
                "benign_shift_vs_2017",
                "benign_logit_shift_contribution",
                "auc_discriminative",
                "auc_oriented_to_2017",
            ]
        ]
        .sort_values(
            "benign_logit_shift_contribution",
            ascending=False,
        )
        .reset_index(
            drop=True
        )
    )

    benign2021.to_csv(
        OUT
        / "ranking_2021_benign_score_push.csv",
        index=False,
    )

    # ========================================================
    # 按特征家族汇总2021良性logit漂移贡献。
    # ========================================================

    family_2021 = (
        benign2021
        .groupby(
            "family",
            as_index=False,
        )
        .agg(
            feature_n=(
                "feature",
                "size",
            ),

            benign_logit_shift_sum=(
                "benign_logit_shift_contribution",
                "sum",
            ),

            benign_logit_shift_abs_sum=(
                "benign_logit_shift_contribution",
                lambda x:
                    float(
                        np.abs(
                            x
                        ).sum()
                    ),
            ),
        )
        .sort_values(
            "benign_logit_shift_sum",
            ascending=False,
        )
    )

    family_2021.to_csv(
        OUT
        / "family_2021_benign_score_shift.csv",
        index=False,
    )

    # ========================================================
    # 打印结果
    # ========================================================

    print()
    print(
        "=" * 100
    )
    print(
        "固定2017 Compact逐年表现"
    )
    print(
        "=" * 100
    )

    print(
        model_year_df.to_string(
            index=False
        )
    )

    print()
    print(
        "=" * 100
    )
    print(
        "A. 高权重 + 标签关系漂移 前20"
    )
    print(
        "=" * 100
    )

    cols_a = [
        "feature",
        "weight_2017",
        "disc_auc_2017",
        "oriented_auc_2018",
        "oriented_auc_2019",
        "oriented_auc_2020",
        "oriented_auc_2021",
        "direction_flip_rate",
        "relationship_drift_score",
    ]

    print(
        relationship_ranked[
            cols_a
        ]
        .head(
            20
        )
        .to_string(
            index=False
        )
    )

    print()
    print(
        "=" * 100
    )
    print(
        "B. 后期仍能区分 + 整体位置漂移 前20"
    )
    print(
        "=" * 100
    )

    cols_b = [
        "feature",
        "family",
        "weight_2017",
        "future_disc_auc_mean",
        "mean_abs_common_shift",
        "common_shift_score",
        "benign_shift_2021",
        "malware_shift_2021",
    ]

    print(
        common_ranked[
            cols_b
        ]
        .head(
            20
        )
        .to_string(
            index=False
        )
    )

    print()
    print(
        "=" * 100
    )
    print(
        "C. 计数类比例化候选 前20"
    )
    print(
        "=" * 100
    )

    print(
        ratio_candidates[
            cols_b
        ]
        .head(
            20
        )
        .to_string(
            index=False
        )
    )

    print()
    print(
        "=" * 100
    )
    print(
        "D. 2021良性样本被推向恶意方向最多的特征 前20"
    )
    print(
        "=" * 100
    )

    print(
        benign2021
        .head(
            20
        )
        .to_string(
            index=False
        )
    )

    print()
    print(
        "=" * 100
    )
    print(
        "E. 2021良性分数漂移：按特征家族"
    )
    print(
        "=" * 100
    )

    print(
        family_2021.to_string(
            index=False
        )
    )

    print()
    print(
        "结果保存到:",
        OUT,
    )


if __name__ == "__main__":
    main()
