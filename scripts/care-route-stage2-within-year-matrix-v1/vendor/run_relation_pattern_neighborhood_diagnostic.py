#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(
    0,
    str(ROOT / "scripts"),
)

import run_frozen_anchor_residual_2017_2021 as base


SOURCE_SEM = Path(
    "artifacts/clean-local-context-2017/"
    "semantic_relation_features.csv.gz"
)

FUTURE_SEM = Path(
    "artifacts/clean-local-context-future/"
    "semantic_relation_features.csv.gz"
)

COMPACT = Path(
    "artifacts/p1-features/features.csv"
)

FIXED_SPLIT = Path(
    "artifacts/p1-splits-fixed-2017-test-2018-2021/"
    "split_manifest.csv"
)

MECHANISM = Path(
    "artifacts/residual-direction-suppression-diagnostic/"
    "apk_mean_probability_diagnostic.csv"
)

OUT = Path(
    "artifacts/"
    "relation-pattern-neighborhood-diagnostic"
)

YEARS = (
    2018,
    2019,
    2020,
    2021,
)

A_CATEGORY = (
    "A_wrong_residual_direction"
)

D_CATEGORY = (
    "D_successfully_corrected"
)

EPS = 1e-12


def norm_sha(series):
    return (
        series.astype(str)
        .str.strip()
        .str.lower()
    )


def numeric_matrix(
    df,
    columns,
):
    return (
        df[
            columns
        ]
        .apply(
            pd.to_numeric,
            errors="coerce",
        )
        .fillna(0.0)
        .to_numpy(
            dtype=np.float32
        )
    )


# ============================================================
# 二值关系集合的Jaccard相似度
#
# 对每个未来APK：
#
# 1. 找2017同类样本中最高相似度
# 2. 找2017异类样本中最高相似度
#
# preference =
#     same_class_max - opposite_class_max
#
# > 0:
#   更像2017同类关系组合
#
# < 0:
#   更像2017异类关系组合
#
# 注意：
# 同类/异类定义使用未来真实标签，
# 所以这里只能用于事后机制解释。
# ============================================================

def neighborhood_metrics(
    source_active,
    source_labels,
    future_active,
    future_labels,
    chunk_size=128,
):
    source_active_i = (
        source_active.astype(
            np.int16
        )
    )

    future_active_i = (
        future_active.astype(
            np.int16
        )
    )

    source_count = (
        source_active_i.sum(
            axis=1
        )
    )

    future_count = (
        future_active_i.sum(
            axis=1
        )
    )

    n_future = len(
        future_active
    )

    max_all = np.zeros(
        n_future,
        dtype=np.float64,
    )

    top5_mean = np.zeros(
        n_future,
        dtype=np.float64,
    )

    max_same = np.zeros(
        n_future,
        dtype=np.float64,
    )

    max_opposite = np.zeros(
        n_future,
        dtype=np.float64,
    )

    exact_any = np.zeros(
        n_future,
        dtype=np.int32,
    )

    exact_same = np.zeros(
        n_future,
        dtype=np.int32,
    )

    for start in range(
        0,
        n_future,
        chunk_size,
    ):
        end = min(
            start + chunk_size,
            n_future,
        )

        f = future_active_i[
            start:end
        ]

        intersections = (
            f
            @ source_active_i.T
        ).astype(
            np.float64
        )

        unions = (
            future_count[
                start:end,
                None,
            ]
            +
            source_count[
                None,
                :,
            ]
            -
            intersections
        )

        similarity = np.divide(
            intersections,
            unions,
            out=np.zeros_like(
                intersections,
                dtype=np.float64,
            ),
            where=(
                unions > 0
            ),
        )

        for local_i in range(
            end - start
        ):
            global_i = (
                start
                + local_i
            )

            row = similarity[
                local_i
            ]

            label = int(
                future_labels[
                    global_i
                ]
            )

            same_mask = (
                source_labels
                == label
            )

            opposite_mask = (
                source_labels
                != label
            )

            same_values = row[
                same_mask
            ]

            opposite_values = row[
                opposite_mask
            ]

            max_all[
                global_i
            ] = float(
                row.max()
            )

            max_same[
                global_i
            ] = float(
                same_values.max()
            )

            max_opposite[
                global_i
            ] = float(
                opposite_values.max()
            )

            k = min(
                5,
                len(row),
            )

            if k > 0:
                top = np.partition(
                    row,
                    len(row) - k,
                )[
                    -k:
                ]

                top5_mean[
                    global_i
                ] = float(
                    top.mean()
                )

            exact_any[
                global_i
            ] = int(
                np.any(
                    row
                    >= 1.0
                    - EPS
                )
            )

            exact_same[
                global_i
            ] = int(
                np.any(
                    same_values
                    >= 1.0
                    - EPS
                )
            )

    return {
        "max_source_jaccard":
            max_all,

        "top5_source_jaccard":
            top5_mean,

        "max_same_class_jaccard":
            max_same,

        "max_opposite_class_jaccard":
            max_opposite,

        "class_neighborhood_preference":
            (
                max_same
                - max_opposite
            ),

        "exact_source_pattern":
            exact_any,

        "exact_same_class_pattern":
            exact_same,
    }


def summarize_group(
    g,
    year,
    category,
):
    preference = (
        g[
            "class_neighborhood_preference"
        ]
        .to_numpy(
            dtype=float
        )
    )

    return {
        "year":
            year,

        "category":
            category,

        "N":
            len(g),

        "mean_active_relations":
            float(
                g[
                    "active_relation_count"
                ].mean()
            )
            if len(g)
            else np.nan,

        "mean_max_source_jaccard":
            float(
                g[
                    "max_source_jaccard"
                ].mean()
            )
            if len(g)
            else np.nan,

        "mean_top5_source_jaccard":
            float(
                g[
                    "top5_source_jaccard"
                ].mean()
            )
            if len(g)
            else np.nan,

        "mean_same_class_jaccard":
            float(
                g[
                    "max_same_class_jaccard"
                ].mean()
            )
            if len(g)
            else np.nan,

        "mean_opposite_class_jaccard":
            float(
                g[
                    "max_opposite_class_jaccard"
                ].mean()
            )
            if len(g)
            else np.nan,

        "mean_class_preference":
            float(
                np.mean(
                    preference
                )
            )
            if len(g)
            else np.nan,

        "median_class_preference":
            float(
                np.median(
                    preference
                )
            )
            if len(g)
            else np.nan,

        "same_class_closer_rate":
            float(
                np.mean(
                    preference > EPS
                )
            )
            if len(g)
            else np.nan,

        "opposite_class_closer_rate":
            float(
                np.mean(
                    preference < -EPS
                )
            )
            if len(g)
            else np.nan,

        "tie_rate":
            float(
                np.mean(
                    np.abs(
                        preference
                    )
                    <= EPS
                )
            )
            if len(g)
            else np.nan,

        "exact_source_pattern_rate":
            float(
                g[
                    "exact_source_pattern"
                ].mean()
            )
            if len(g)
            else np.nan,

        "exact_same_class_pattern_rate":
            float(
                g[
                    "exact_same_class_pattern"
                ].mean()
            )
            if len(g)
            else np.nan,

        "mean_margin":
            float(
                g[
                    "base_margin_probability"
                ].mean()
            )
            if len(g)
            else np.nan,

        "benign":
            int(
                (
                    g[
                        "label"
                    ]
                    == 0
                ).sum()
            ),

        "malware":
            int(
                (
                    g[
                        "label"
                    ]
                    == 1
                ).sum()
            ),
    }


def preference_bin(x):
    if x < -0.20:
        return "<=-0.20"

    if x < -0.05:
        return "-0.20--0.05"

    if x <= 0.05:
        return "-0.05-0.05"

    if x <= 0.20:
        return "0.05-0.20"

    return ">0.20"


def main():
    for path in (
        SOURCE_SEM,
        FUTURE_SEM,
        COMPACT,
        FIXED_SPLIT,
        MECHANISM,
    ):
        if not path.exists():
            raise SystemExit(
                "找不到: "
                + str(path)
            )

    OUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ========================================================
    # 1. 2017正式799
    # ========================================================

    source = pd.read_csv(
        SOURCE_SEM,
        compression="gzip",
    )

    source[
        "sha256"
    ] = norm_sha(
        source[
            "sha256"
        ]
    )

    source = (
        source
        .drop_duplicates(
            "sha256"
        )
        .copy()
    )

    names150 = (
        base.support150_names(
            source.columns
        )
    )

    if len(names150) != 150:
        raise SystemExit(
            "final150不是150维: "
            + str(
                len(names150)
            )
        )

    split = pd.read_csv(
        FIXED_SPLIT
    )

    split[
        "sha256"
    ] = norm_sha(
        split[
            "sha256"
        ]
    )

    formal_sha = (
        set(
            split[
                "sha256"
            ]
        )
        &
        set(
            source[
                "sha256"
            ]
        )
    )

    if len(
        formal_sha
    ) != 799:
        raise SystemExit(
            "正式2017人口不是799: "
            + str(
                len(
                    formal_sha
                )
            )
        )

    source = (
        source[
            source[
                "sha256"
            ].isin(
                formal_sha
            )
        ]
        .sort_values(
            "sha256"
        )
        .reset_index(
            drop=True
        )
    )

    if len(source) != 799:
        raise SystemExit(
            "2017过滤后不是799行"
        )

    compact_by_sha = (
        base.normalized_rows(
            base.read_csv_rows(
                COMPACT
            )
        )
    )

    source_rows = [
        {
            "sha256":
                sha
        }
        for sha in source[
            "sha256"
        ]
    ]

    source_labels = (
        base.labels_for(
            source_rows,
            compact_by_sha,
        )
        .astype(int)
        .ravel()
    )

    if set(
        np.unique(
            source_labels
        )
    ) != {
        0,
        1,
    }:
        raise SystemExit(
            "2017标签不是0/1"
        )

    print(
        "2017正式APK =",
        len(source),
    )

    print(
        "2017良性/恶意 =",
        int(
            (
                source_labels
                == 0
            ).sum()
        ),
        "/",
        int(
            (
                source_labels
                == 1
            ).sum()
        ),
    )

    # ========================================================
    # 2. 未来1985
    # ========================================================

    future_sem = pd.read_csv(
        FUTURE_SEM,
        compression="gzip",
    )

    future_sem[
        "sha256"
    ] = norm_sha(
        future_sem[
            "sha256"
        ]
    )

    future_sem = (
        future_sem
        .drop_duplicates(
            "sha256"
        )
    )

    future_names = (
        base.support150_names(
            future_sem.columns
        )
    )

    if (
        list(
            future_names
        )
        != list(
            names150
        )
    ):
        raise SystemExit(
            "2017与未来final150列不一致"
        )

    mechanism = pd.read_csv(
        MECHANISM
    )

    mechanism[
        "sha256"
    ] = norm_sha(
        mechanism[
            "sha256"
        ]
    )

    mechanism[
        "year"
    ] = pd.to_numeric(
        mechanism[
            "year"
        ],
        errors="raise",
    ).astype(int)

    mechanism[
        "label"
    ] = pd.to_numeric(
        mechanism[
            "label"
        ],
        errors="raise",
    ).astype(int)

    future = (
        future_sem.merge(
            mechanism[
                [
                    "sha256",
                    "year",
                    "label",
                    "presence150",
                    "base_correct",
                    "mechanism_category",
                    "base_margin_probability",
                ]
            ],
            on="sha256",
            how="inner",
            validate="one_to_one",
        )
        .sort_values(
            [
                "year",
                "sha256",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    if len(
        future
    ) != 1985:
        raise SystemExit(
            "未来人口不是1985: "
            + str(
                len(
                    future
                )
            )
        )

    print(
        "未来APK =",
        len(
            future
        ),
    )

    # ========================================================
    # 3. 原始150关系集合
    # ========================================================

    source_values = (
        numeric_matrix(
            source,
            names150,
        )
    )

    future_values = (
        numeric_matrix(
            future,
            names150,
        )
    )

    source_active = (
        np.abs(
            source_values
        )
        > EPS
    )

    future_active = (
        np.abs(
            future_values
        )
        > EPS
    )

    future[
        "active_relation_count"
    ] = (
        future_active.sum(
            axis=1
        )
    )

    # ========================================================
    # 4. 关系组合邻域
    # ========================================================

    result = (
        neighborhood_metrics(
            source_active,
            source_labels,
            future_active,
            future[
                "label"
            ].to_numpy(
                dtype=int
            ),
        )
    )

    for name, values in (
        result.items()
    ):
        future[
            name
        ] = values

    future[
        "preference_bin"
    ] = future[
        "class_neighborhood_preference"
    ].map(
        preference_bin
    )

    # ========================================================
    # 5. 重点看：
    # Compact错误 + presence150=1
    # ========================================================

    base_correct = (
        future[
            "base_correct"
        ]
        .astype(str)
        .str.strip()
        .str.lower()
        .map({
            "true": True,
            "false": False,
            "1": True,
            "0": False,
        })
    )

    if base_correct.isna().any():
        raise SystemExit(
            "base_correct无法解析"
        )

    future[
        "_base_correct_bool"
    ] = base_correct

    errors = future[
        (
            ~future[
                "_base_correct_bool"
            ]
        )
        &
        (
            pd.to_numeric(
                future[
                    "presence150"
                ],
                errors="raise",
            )
            > 0
        )
    ].copy()

    print(
        "Compact错误且presence150=1 =",
        len(
            errors
        ),
    )

    # ========================================================
    # 6. A vs D
    # ========================================================

    summary_rows = []

    for year_value in (
        list(
            YEARS
        )
        + [
            "ALL"
        ]
    ):
        if (
            year_value
            == "ALL"
        ):
            yg = errors
        else:
            yg = errors[
                errors[
                    "year"
                ]
                == year_value
            ]

        for category in (
            A_CATEGORY,
            D_CATEGORY,
        ):
            g = yg[
                yg[
                    "mechanism_category"
                ]
                == category
            ]

            summary_rows.append(
                summarize_group(
                    g,
                    year_value,
                    category,
                )
            )

    summary = pd.DataFrame(
        summary_rows
    )

    # ========================================================
    # 7. 邻域偏好分桶
    # ========================================================

    bin_rows = []

    bin_order = (
        "<=-0.20",
        "-0.20--0.05",
        "-0.05-0.05",
        "0.05-0.20",
        ">0.20",
    )

    for year_value in (
        list(
            YEARS
        )
        + [
            "ALL"
        ]
    ):
        if (
            year_value
            == "ALL"
        ):
            yg = errors
        else:
            yg = errors[
                errors[
                    "year"
                ]
                == year_value
            ]

        for bin_name in (
            bin_order
        ):
            g = yg[
                yg[
                    "preference_bin"
                ]
                == bin_name
            ]

            A_n = int(
                (
                    g[
                        "mechanism_category"
                    ]
                    == A_CATEGORY
                ).sum()
            )

            D_n = int(
                (
                    g[
                        "mechanism_category"
                    ]
                    == D_CATEGORY
                ).sum()
            )

            bin_rows.append({
                "year":
                    year_value,

                "preference_bin":
                    bin_name,

                "N":
                    len(g),

                "A_wrong_direction":
                    A_n,

                "D_successfully_corrected":
                    D_n,

                "A_rate":
                    (
                        A_n / len(g)
                        if len(g)
                        else np.nan
                    ),

                "D_rate":
                    (
                        D_n / len(g)
                        if len(g)
                        else np.nan
                    ),

                "mean_margin":
                    (
                        float(
                            g[
                                "base_margin_probability"
                            ].mean()
                        )
                        if len(g)
                        else np.nan
                    ),

                "mean_active_relations":
                    (
                        float(
                            g[
                                "active_relation_count"
                            ].mean()
                        )
                        if len(g)
                        else np.nan
                    ),
            })

    bin_summary = pd.DataFrame(
        bin_rows
    )

    # ========================================================
    # 保存
    # ========================================================

    future.drop(
        columns=[
            "_base_correct_bool"
        ]
    ).to_csv(
        OUT
        / "sample_pattern_neighborhood.csv",
        index=False,
    )

    summary.to_csv(
        OUT
        / "A_vs_D_pattern_summary.csv",
        index=False,
    )

    bin_summary.to_csv(
        OUT
        / "pattern_preference_bins.csv",
        index=False,
    )

    # ========================================================
    # 打印核心结果
    # ========================================================

    readable = {
        A_CATEGORY:
            "A方向错误",

        D_CATEGORY:
            "D成功纠正",
    }

    print()
    print(
        "======================================"
    )
    print(
        "A方向错误 vs D成功纠正：关系组合邻域"
    )
    print(
        "======================================"
    )

    for year_value in (
        list(
            YEARS
        )
        + [
            "ALL"
        ]
    ):
        print()
        print(
            year_value
        )

        yg = summary[
            summary[
                "year"
            ].astype(str)
            == str(
                year_value
            )
        ]

        for category in (
            A_CATEGORY,
            D_CATEGORY,
        ):
            row = yg[
                yg[
                    "category"
                ]
                == category
            ].iloc[
                0
            ]

            print(
                f"{readable[category]:10s} "
                f"N={int(row['N']):3d} "
                f"关系数={row['mean_active_relations']:.2f} "
                f"同类最近={row['mean_same_class_jaccard']:.4f} "
                f"异类最近={row['mean_opposite_class_jaccard']:.4f} "
                f"偏好={row['mean_class_preference']:+.4f} "
                f"同类更近={100*row['same_class_closer_rate']:.1f}% "
                f"异类更近={100*row['opposite_class_closer_rate']:.1f}% "
                f"margin={row['mean_margin']:.4f}"
            )

    print()
    print(
        "======================================"
    )
    print(
        "全部年份：按组合邻域偏好分桶"
    )
    print(
        "======================================"
    )

    all_bins = bin_summary[
        bin_summary[
            "year"
        ].astype(str)
        == "ALL"
    ]

    for bin_name in (
        bin_order
    ):
        row = all_bins[
            all_bins[
                "preference_bin"
            ]
            == bin_name
        ].iloc[
            0
        ]

        print(
            f"{bin_name:12s} "
            f"N={int(row['N']):3d} "
            f"A={int(row['A_wrong_direction']):3d} "
            f"D={int(row['D_successfully_corrected']):3d} "
            f"margin={row['mean_margin']:.4f} "
            f"关系数={row['mean_active_relations']:.2f}"
        )

    print()
    print(
        "结果保存到:",
        OUT,
    )

    print()
    print(
        "注意：同类/异类邻域使用未来真实标签，"
        "只允许用于事后机制解释，不能进入模型或未来调参。"
    )


if __name__ == "__main__":
    main()
