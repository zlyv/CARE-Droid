#!/usr/bin/env python3

from pathlib import Path

import numpy as np
import pandas as pd


INPUT = Path(
    "artifacts/semantic-relation-conflict-diagnostic/"
    "sample_semantic_conflict.csv"
)

OUT = Path(
    "artifacts/semantic-consistency-stratified-control"
)

A = "A_wrong_residual_direction"
D = "D_successfully_corrected"


def margin_bin(x):
    x = float(x)

    if x < 0.05:
        return "0-0.05"
    if x < 0.10:
        return "0.05-0.10"
    if x < 0.20:
        return "0.10-0.20"
    if x < 0.30:
        return "0.20-0.30"

    return ">=0.30"


def count_bin(x):
    x = int(x)

    if x <= 5:
        return "1-5"
    if x <= 10:
        return "6-10"
    if x <= 15:
        return "11-15"

    return "16+"


def summarize_strata(
    df,
    columns,
    name,
):
    rows = []

    for key, g in df.groupby(
        columns,
        dropna=False,
    ):
        if not isinstance(
            key,
            tuple,
        ):
            key = (
                key,
            )

        ga = g[
            g[
                "mechanism_category"
            ] == A
        ]

        gd = g[
            g[
                "mechanism_category"
            ] == D
        ]

        if (
            len(ga) == 0
            or
            len(gd) == 0
        ):
            continue

        row = {
            "scheme":
                name,

            "A_N":
                len(ga),

            "D_N":
                len(gd),

            "A_mean_consistency":
                float(
                    ga[
                        "relation_consistency"
                    ].mean()
                ),

            "D_mean_consistency":
                float(
                    gd[
                        "relation_consistency"
                    ].mean()
                ),

            "D_minus_A":
                float(
                    gd[
                        "relation_consistency"
                    ].mean()
                    -
                    ga[
                        "relation_consistency"
                    ].mean()
                ),

            # 使用两组较小样本数作为该层权重，
            # 防止极度不平衡的层支配总体结果。
            "weight":
                min(
                    len(ga),
                    len(gd),
                ),

            "A_mean_margin":
                float(
                    ga[
                        "base_margin_probability"
                    ].mean()
                ),

            "D_mean_margin":
                float(
                    gd[
                        "base_margin_probability"
                    ].mean()
                ),

            "A_mean_relations":
                float(
                    ga[
                        "active_relation_count"
                    ].mean()
                ),

            "D_mean_relations":
                float(
                    gd[
                        "active_relation_count"
                    ].mean()
                ),
        }

        for col, value in zip(
            columns,
            key,
        ):
            row[col] = value

        rows.append(
            row
        )

    return pd.DataFrame(
        rows
    )


def aggregate(x):
    if len(x) == 0:
        return {
            "strata": 0,
            "matched_weight": 0,
            "weighted_D_minus_A": np.nan,
            "median_D_minus_A": np.nan,
            "positive_strata": 0,
            "negative_strata": 0,
            "zero_strata": 0,
        }

    weights = (
        x[
            "weight"
        ].to_numpy(
            dtype=float
        )
    )

    diff = (
        x[
            "D_minus_A"
        ].to_numpy(
            dtype=float
        )
    )

    return {
        "strata":
            len(x),

        "matched_weight":
            int(
                weights.sum()
            ),

        "weighted_D_minus_A":
            float(
                np.average(
                    diff,
                    weights=weights,
                )
            ),

        "median_D_minus_A":
            float(
                np.median(
                    diff
                )
            ),

        "positive_strata":
            int(
                (
                    diff > 0
                ).sum()
            ),

        "negative_strata":
            int(
                (
                    diff < 0
                ).sum()
            ),

        "zero_strata":
            int(
                (
                    np.abs(diff)
                    < 1e-12
                ).sum()
            ),
    }


def main():
    if not INPUT.exists():
        raise SystemExit(
            "找不到: "
            + str(INPUT)
        )

    OUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    df = pd.read_csv(
        INPUT
    )

    required = {
        "year",
        "label",
        "mechanism_category",
        "base_margin_probability",
        "active_relation_count",
        "relation_consistency",
    }

    missing = (
        required
        - set(df.columns)
    )

    if missing:
        raise SystemExit(
            "缺少列: "
            + ", ".join(
                sorted(missing)
            )
        )

    # --------------------------------------------------------
    # 这里只回答：
    #
    # A：残差方向错误
    # vs
    # D：成功纠正
    #
    # B/C不混进来。
    # --------------------------------------------------------

    ad = df[
        df[
            "mechanism_category"
        ].isin(
            (
                A,
                D,
            )
        )
    ].copy()

    ad = ad[
        ad[
            "relation_consistency"
        ].notna()
    ].copy()

    ad[
        "margin_bin"
    ] = ad[
        "base_margin_probability"
    ].map(
        margin_bin
    )

    ad[
        "count_bin"
    ] = ad[
        "active_relation_count"
    ].map(
        count_bin
    )

    # ========================================================
    # 三种控制强度
    # ========================================================

    # 控制距离 + 关系数量。
    s1 = summarize_strata(
        ad,
        [
            "margin_bin",
            "count_bin",
        ],
        "margin+count",
    )

    # 再控制年份。
    s2 = summarize_strata(
        ad,
        [
            "year",
            "margin_bin",
            "count_bin",
        ],
        "year+margin+count",
    )

    # 最严格：
    # 年份 + 标签 + 距离 + 关系数量。
    s3 = summarize_strata(
        ad,
        [
            "year",
            "label",
            "margin_bin",
            "count_bin",
        ],
        "year+label+margin+count",
    )

    all_strata = pd.concat(
        [
            s1,
            s2,
            s3,
        ],
        ignore_index=True,
    )

    summary_rows = []

    for name, frame in (
        (
            "margin+count",
            s1,
        ),
        (
            "year+margin+count",
            s2,
        ),
        (
            "year+label+margin+count",
            s3,
        ),
    ):
        row = {
            "scheme":
                name,
        }

        row.update(
            aggregate(
                frame
            )
        )

        summary_rows.append(
            row
        )

    summary = pd.DataFrame(
        summary_rows
    )

    # ========================================================
    # 重点再看 margin < 0.20
    #
    # 因为之前实际纠正全部集中在这里。
    # ========================================================

    close = ad[
        ad[
            "base_margin_probability"
        ] < 0.20
    ].copy()

    close_s1 = summarize_strata(
        close,
        [
            "margin_bin",
            "count_bin",
        ],
        "close_margin+count",
    )

    close_s2 = summarize_strata(
        close,
        [
            "year",
            "margin_bin",
            "count_bin",
        ],
        "close_year+margin+count",
    )

    close_s3 = summarize_strata(
        close,
        [
            "year",
            "label",
            "margin_bin",
            "count_bin",
        ],
        "close_year+label+margin+count",
    )

    close_rows = []

    for name, frame in (
        (
            "close_margin+count",
            close_s1,
        ),
        (
            "close_year+margin+count",
            close_s2,
        ),
        (
            "close_year+label+margin+count",
            close_s3,
        ),
    ):
        row = {
            "scheme":
                name,
        }

        row.update(
            aggregate(
                frame
            )
        )

        close_rows.append(
            row
        )

    close_summary = pd.DataFrame(
        close_rows
    )

    # ========================================================
    # 保存
    # ========================================================

    ad.to_csv(
        OUT
        / "A_D_samples.csv",
        index=False,
    )

    all_strata.to_csv(
        OUT
        / "all_strata.csv",
        index=False,
    )

    summary.to_csv(
        OUT
        / "overall_control_summary.csv",
        index=False,
    )

    close_summary.to_csv(
        OUT
        / "close_margin_control_summary.csv",
        index=False,
    )

    pd.concat(
        [
            close_s1,
            close_s2,
            close_s3,
        ],
        ignore_index=True,
    ).to_csv(
        OUT
        / "close_margin_strata.csv",
        index=False,
    )

    # ========================================================
    # 打印
    # ========================================================

    print()
    print(
        "======================================"
    )
    print(
        "全部A/D样本：分层控制结果"
    )
    print(
        "======================================"
    )

    for _, row in (
        summary.iterrows()
    ):
        print()
        print(
            row[
                "scheme"
            ]
        )

        print(
            " 可比较分层数 =",
            int(
                row[
                    "strata"
                ]
            )
        )

        print(
            " 匹配权重 =",
            int(
                row[
                    "matched_weight"
                ]
            )
        )

        print(
            " 加权(D一致性-A一致性) =",
            f"{row['weighted_D_minus_A']:+.4f}",
        )

        print(
            " 中位差 =",
            f"{row['median_D_minus_A']:+.4f}",
        )

        print(
            " D>A / D<A =",
            int(
                row[
                    "positive_strata"
                ]
            ),
            "/",
            int(
                row[
                    "negative_strata"
                ]
            ),
        )

    print()
    print(
        "======================================"
    )
    print(
        "只看 Compact margin < 0.20"
    )
    print(
        "======================================"
    )

    for _, row in (
        close_summary.iterrows()
    ):
        print()
        print(
            row[
                "scheme"
            ]
        )

        print(
            " 可比较分层数 =",
            int(
                row[
                    "strata"
                ]
            )
        )

        print(
            " 匹配权重 =",
            int(
                row[
                    "matched_weight"
                ]
            )
        )

        print(
            " 加权(D一致性-A一致性) =",
            f"{row['weighted_D_minus_A']:+.4f}",
        )

        print(
            " 中位差 =",
            f"{row['median_D_minus_A']:+.4f}",
        )

        print(
            " D>A / D<A =",
            int(
                row[
                    "positive_strata"
                ]
            ),
            "/",
            int(
                row[
                    "negative_strata"
                ]
            ),
        )

    print()
    print(
        "======================================"
    )
    print(
        "margin<0.20的具体可比较分层"
    )
    print(
        "======================================"
    )

    # 打印最严格控制。
    if len(
        close_s3
    ) == 0:
        print(
            "严格分层下没有同时包含A和D的层。"
        )

    else:
        for _, row in (
            close_s3
            .sort_values(
                [
                    "year",
                    "label",
                    "margin_bin",
                    "count_bin",
                ]
            )
            .iterrows()
        ):
            print(
                f"year={int(row['year'])} "
                f"label={int(row['label'])} "
                f"margin={row['margin_bin']:10s} "
                f"count={row['count_bin']:5s} "
                f"A={int(row['A_N']):2d} "
                f"D={int(row['D_N']):2d} "
                f"A一致性={row['A_mean_consistency']:.3f} "
                f"D一致性={row['D_mean_consistency']:.3f} "
                f"差={row['D_minus_A']:+.3f}"
            )

    print()
    print(
        "结果保存到:",
        OUT,
    )


if __name__ == "__main__":
    main()
