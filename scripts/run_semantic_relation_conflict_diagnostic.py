#!/usr/bin/env python3

from pathlib import Path

import numpy as np
import pandas as pd


INPUT = Path(
    "artifacts/relation-label-direction-drift-diagnostic/"
    "sample_relation_direction_diagnostic.csv"
)

OUT = Path(
    "artifacts/semantic-relation-conflict-diagnostic"
)

YEARS = (2018, 2019, 2020, 2021)

CATEGORIES = (
    "A_wrong_residual_direction",
    "B_suppressed_by_uncertainty",
    "C_correct_direction_but_too_small",
    "D_successfully_corrected",
)

EPS = 1e-12


def to_bool(s):
    if s.dtype == bool:
        return s

    x = (
        s.astype(str)
        .str.strip()
        .str.lower()
    )

    mapping = {
        "true": True,
        "false": False,
        "1": True,
        "0": False,
    }

    bad = set(x.unique()) - set(mapping)

    if bad:
        raise SystemExit(
            "无法解析布尔值: "
            + str(sorted(bad))
        )

    return x.map(mapping).astype(bool)


def consistency_bin(x):
    if pd.isna(x):
        return "无方向关系"

    if x < 0.2:
        return "0-0.2"
    if x < 0.4:
        return "0.2-0.4"
    if x < 0.6:
        return "0.4-0.6"
    if x < 0.8:
        return "0.6-0.8"

    return "0.8-1.0"


def margin_bin(x):
    if x < 0.05:
        return "0-0.05"
    if x < 0.10:
        return "0.05-0.10"
    if x < 0.20:
        return "0.10-0.20"
    if x < 0.30:
        return "0.20-0.30"

    return ">=0.30"


def safe_rate(a, b):
    if b == 0:
        return np.nan

    return float(a) / float(b)


def pct(x):
    if pd.isna(x):
        return "nan"

    return f"{100*x:.2f}%"


def summarize(g, year_value, bin_value):
    row = {
        "year": year_value,
        "consistency_bin": bin_value,
        "N": len(g),
        "mean_consistency": (
            float(
                g["relation_consistency"].mean()
            )
            if len(g)
            else np.nan
        ),
        "mean_conflict": (
            float(
                g["relation_conflict"].mean()
            )
            if len(g)
            else np.nan
        ),
        "mean_margin": (
            float(
                g["base_margin_probability"].mean()
            )
            if len(g)
            else np.nan
        ),
        "mean_active_relations": (
            float(
                g["active_relation_count"].mean()
            )
            if len(g)
            else np.nan
        ),
        "mean_directional_fraction": (
            float(
                g["directional_fraction"].mean()
            )
            if len(g)
            else np.nan
        ),
        "mean_truth_alignment": (
            float(
                g["truth_alignment_normalized"].mean()
            )
            if len(g)
            else np.nan
        ),
        "benign": int(
            (g["label"] == 0).sum()
        ),
        "malware": int(
            (g["label"] == 1).sum()
        ),
    }

    for cat in CATEGORIES:
        n = int(
            (
                g["mechanism_category"]
                == cat
            ).sum()
        )

        row[cat] = n
        row[cat + "_rate"] = safe_rate(
            n,
            len(g),
        )

    return row


def main():
    if not INPUT.exists():
        raise SystemExit(
            "找不到输入文件: "
            + str(INPUT)
        )

    OUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    df = pd.read_csv(INPUT)

    required = {
        "year",
        "sha256",
        "label",
        "presence150",
        "base_correct",
        "mechanism_category",
        "base_margin_probability",
        "active_relation_count",
        "source_truth_support_fraction",
        "source_truth_opposition_fraction",
    }

    missing = required - set(df.columns)

    if missing:
        raise SystemExit(
            "缺少列: "
            + ", ".join(sorted(missing))
        )

    df["base_correct"] = to_bool(
        df["base_correct"]
    )

    df["presence150"] = (
        pd.to_numeric(
            df["presence150"],
            errors="raise",
        )
        .astype(int)
    )

    df["year"] = (
        pd.to_numeric(
            df["year"],
            errors="raise",
        )
        .astype(int)
    )

    # 只分析：
    # Compact原本错误 + 存在150关系。
    err = df[
        (
            ~df["base_correct"]
        )
        &
        (
            df["presence150"] == 1
        )
    ].copy()

    support = (
        err[
            "source_truth_support_fraction"
        ].astype(float)
    )

    oppose = (
        err[
            "source_truth_opposition_fraction"
        ].astype(float)
    )

    if (
        (support < -EPS).any()
        or
        (oppose < -EPS).any()
    ):
        raise SystemExit(
            "支持/反对比例出现负数"
        )

    directional = (
        support
        + oppose
    )

    err[
        "directional_fraction"
    ] = directional

    # --------------------------------------------------------
    # 关系方向一致性：
    #
    # 0 = 支持两侧几乎各半
    # 1 = 有方向的关系全部指向同一侧
    # --------------------------------------------------------

    err[
        "relation_consistency"
    ] = np.where(
        directional > EPS,
        np.abs(
            support - oppose
        ) / directional,
        np.nan,
    )

    err[
        "relation_conflict"
    ] = (
        1.0
        -
        err[
            "relation_consistency"
        ]
    )

    # 这个带正负号的量依赖真实标签，
    # 只允许用于事后解释。
    err[
        "truth_alignment_normalized"
    ] = np.where(
        directional > EPS,
        (
            support
            - oppose
        )
        / directional,
        np.nan,
    )

    err[
        "consistency_bin"
    ] = err[
        "relation_consistency"
    ].map(
        consistency_bin
    )

    err[
        "margin_bin"
    ] = err[
        "base_margin_probability"
    ].map(
        margin_bin
    )

    # ========================================================
    # 1. 每年 × 一致性分桶
    # ========================================================

    consistency_order = (
        "0-0.2",
        "0.2-0.4",
        "0.4-0.6",
        "0.6-0.8",
        "0.8-1.0",
        "无方向关系",
    )

    rows = []

    for year_value in list(YEARS) + ["ALL"]:
        if year_value == "ALL":
            yg = err
        else:
            yg = err[
                err["year"]
                == year_value
            ]

        for bin_value in consistency_order:
            g = yg[
                yg["consistency_bin"]
                == bin_value
            ]

            rows.append(
                summarize(
                    g,
                    year_value,
                    bin_value,
                )
            )

    consistency_summary = pd.DataFrame(
        rows
    )

    # ========================================================
    # 2. 一致性 × Compact margin
    #
    # 这是最关键的控制分析。
    # ========================================================

    margin_order = (
        "0-0.05",
        "0.05-0.10",
        "0.10-0.20",
        "0.20-0.30",
        ">=0.30",
    )

    grid_rows = []

    for year_value in list(YEARS) + ["ALL"]:
        if year_value == "ALL":
            yg = err
        else:
            yg = err[
                err["year"]
                == year_value
            ]

        for m in margin_order:
            mg = yg[
                yg["margin_bin"]
                == m
            ]

            for c in consistency_order:
                g = mg[
                    mg["consistency_bin"]
                    == c
                ]

                A = int(
                    (
                        g[
                            "mechanism_category"
                        ]
                        ==
                        "A_wrong_residual_direction"
                    ).sum()
                )

                D = int(
                    (
                        g[
                            "mechanism_category"
                        ]
                        ==
                        "D_successfully_corrected"
                    ).sum()
                )

                grid_rows.append({
                    "year":
                        year_value,

                    "margin_bin":
                        m,

                    "consistency_bin":
                        c,

                    "N":
                        len(g),

                    "A_wrong_direction":
                        A,

                    "D_corrected":
                        D,

                    "A_rate":
                        safe_rate(
                            A,
                            len(g),
                        ),

                    "D_rate":
                        safe_rate(
                            D,
                            len(g),
                        ),

                    "mean_consistency":
                        (
                            float(
                                g[
                                    "relation_consistency"
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

    grid = pd.DataFrame(
        grid_rows
    )

    # ========================================================
    # 3. A vs D直接比较
    # ========================================================

    ad_rows = []

    for year_value in list(YEARS) + ["ALL"]:
        if year_value == "ALL":
            yg = err
        else:
            yg = err[
                err["year"]
                == year_value
            ]

        for cat in (
            "A_wrong_residual_direction",
            "D_successfully_corrected",
        ):
            g = yg[
                yg["mechanism_category"]
                == cat
            ]

            ad_rows.append({
                "year":
                    year_value,

                "category":
                    cat,

                "N":
                    len(g),

                "mean_consistency":
                    (
                        float(
                            g[
                                "relation_consistency"
                            ].mean()
                        )
                        if len(g)
                        else np.nan
                    ),

                "median_consistency":
                    (
                        float(
                            g[
                                "relation_consistency"
                            ].median()
                        )
                        if len(g)
                        else np.nan
                    ),

                "mean_conflict":
                    (
                        float(
                            g[
                                "relation_conflict"
                            ].mean()
                        )
                        if len(g)
                        else np.nan
                    ),

                "mean_directional_fraction":
                    (
                        float(
                            g[
                                "directional_fraction"
                            ].mean()
                        )
                        if len(g)
                        else np.nan
                    ),

                "mean_truth_alignment":
                    (
                        float(
                            g[
                                "truth_alignment_normalized"
                            ].mean()
                        )
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

                "benign":
                    int(
                        (
                            g["label"]
                            == 0
                        ).sum()
                    ),

                "malware":
                    int(
                        (
                            g["label"]
                            == 1
                        ).sum()
                    ),
            })

    ad_summary = pd.DataFrame(
        ad_rows
    )

    # ========================================================
    # 保存
    # ========================================================

    err.to_csv(
        OUT
        / "sample_semantic_conflict.csv",
        index=False,
    )

    consistency_summary.to_csv(
        OUT
        / "year_consistency_bins.csv",
        index=False,
    )

    grid.to_csv(
        OUT
        / "year_consistency_margin_grid.csv",
        index=False,
    )

    ad_summary.to_csv(
        OUT
        / "A_vs_D_consistency.csv",
        index=False,
    )

    # ========================================================
    # 打印核心结果
    # ========================================================

    print()
    print(
        "======================================"
    )
    print(
        "A方向错误 vs D成功纠正：关系一致性"
    )
    print(
        "======================================"
    )

    for year_value in list(YEARS) + ["ALL"]:
        print()
        print(year_value)

        yg = ad_summary[
            ad_summary["year"].astype(str)
            == str(year_value)
        ]

        for _, row in yg.iterrows():
            name = (
                "A方向错误"
                if row["category"]
                ==
                "A_wrong_residual_direction"
                else
                "D成功纠正"
            )

            print(
                f"{name:10s} "
                f"N={int(row['N']):3d} "
                f"一致性mean={row['mean_consistency']:.4f} "
                f"median={row['median_consistency']:.4f} "
                f"冲突={row['mean_conflict']:.4f} "
                f"margin={row['mean_margin']:.4f} "
                f"关系数={row['mean_active_relations']:.2f}"
            )

    print()
    print(
        "======================================"
    )
    print(
        "按关系一致性分桶：全部年份"
    )
    print(
        "======================================"
    )

    all_bins = consistency_summary[
        consistency_summary[
            "year"
        ].astype(str) == "ALL"
    ]

    for c in consistency_order:
        row = all_bins[
            all_bins[
                "consistency_bin"
            ] == c
        ].iloc[0]

        print(
            f"{c:10s} "
            f"N={int(row['N']):3d} "
            f"A={int(row['A_wrong_residual_direction']):3d} "
            f"A率={pct(row['A_wrong_residual_direction_rate']):>7s} "
            f"D={int(row['D_successfully_corrected']):3d} "
            f"D率={pct(row['D_successfully_corrected_rate']):>7s} "
            f"margin={row['mean_margin']:.4f}"
        )

    print()
    print(
        "======================================"
    )
    print(
        "控制margin后的关键表"
    )
    print(
        "======================================"
    )

    # 优先打印最有纠正可能的margin<0.20区域。
    controlled = grid[
        (
            grid["year"].astype(str)
            == "ALL"
        )
        &
        (
            grid["margin_bin"].isin(
                (
                    "0-0.05",
                    "0.05-0.10",
                    "0.10-0.20",
                )
            )
        )
        &
        (
            grid["N"] > 0
        )
    ]

    for m in (
        "0-0.05",
        "0.05-0.10",
        "0.10-0.20",
    ):
        print()
        print(
            "margin",
            m,
        )

        mg = controlled[
            controlled[
                "margin_bin"
            ] == m
        ]

        for _, row in mg.iterrows():
            print(
                f"  一致性={row['consistency_bin']:10s} "
                f"N={int(row['N']):3d} "
                f"A={int(row['A_wrong_direction']):3d} "
                f"D={int(row['D_corrected']):3d} "
                f"D率={pct(row['D_rate']):>7s}"
            )

    print()
    print(
        "结果保存到:",
        OUT,
    )

    print()
    print(
        "注意：这一步只是机制诊断，"
        "不能据此用2018-2021重新选择阈值或调整模型。"
    )


if __name__ == "__main__":
    main()
