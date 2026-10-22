#!/usr/bin/env python3
from pathlib import Path

import numpy as np
import pandas as pd


INPUT = Path(
    "artifacts/"
    "frozen-anchor-residual-count10-150-2017-2021/"
    "future_predictions.csv"
)

OUT = Path(
    "artifacts/"
    "negative-residual-required-gain-diagnostic"
)

EPS = 1e-12
BEST_GNEG = 16.25


def logit(p):
    p = np.clip(
        np.asarray(p, dtype=float),
        1e-7,
        1.0 - 1e-7,
    )
    return np.log(
        p / (1.0 - p)
    )


def summarize_required_gain(
    df,
    name,
):
    if len(df) == 0:
        print(name, "N=0")
        return

    x = (
        df["required_gain"]
        .replace(
            [np.inf, -np.inf],
            np.nan,
        )
        .dropna()
        .to_numpy(dtype=float)
    )

    rho = (
        df["rho"]
        .replace(
            [np.inf, -np.inf],
            np.nan,
        )
        .dropna()
        .to_numpy(dtype=float)
    )

    print()
    print(name)
    print("-" * 72)
    print(
        f"N={len(df)} "
        f"median_required_gain={np.median(x):.4f} "
        f"mean={np.mean(x):.4f} "
        f"p25={np.quantile(x, 0.25):.4f} "
        f"p75={np.quantile(x, 0.75):.4f}"
    )

    print(
        f"median_rho={np.median(rho):.4f} "
        f"mean_rho={np.mean(rho):.4f}"
    )

    for g in (
        1.0,
        2.0,
        4.0,
        8.0,
        12.0,
        16.25,
        20.0,
    ):
        n = int(
            np.sum(
                x <= g
            )
        )

        print(
            f"required_gain <= {g:>5.2f}: "
            f"{n:4d}/{len(x):4d} "
            f"({n / len(x) * 100:6.2f}%)"
        )


def main():
    OUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    df = pd.read_csv(
        INPUT
    )

    print(
        "rows =",
        len(df),
        "APK =",
        df["sha256"].nunique(),
    )

    # ========================================================
    # 每个 seed 独立分析。
    # ========================================================

    df["base_logit"] = logit(
        df["base_probability"]
    )

    df["base_prediction"] = (
        df["base_probability"]
        >= 0.5
    ).astype(int)

    df["base_correct"] = (
        df["base_prediction"]
        ==
        df["label"]
    )

    # 当前真正加到Compact logit上的量。
    df["effective_residual"] = (
        df["gate"]
        *
        df["semantic_delta"]
    )

    # Compact误报：
    # label=0，但Compact预测1。
    df["base_FP"] = (
        (df["label"] == 0)
        &
        (df["base_prediction"] == 1)
    )

    # Compact漏报：
    # label=1，但Compact预测0。
    df["base_FN"] = (
        (df["label"] == 1)
        &
        (df["base_prediction"] == 0)
    )

    # 对误报，正确纠错方向必须是负。
    df["FP_direction_correct"] = (
        df["base_FP"]
        &
        (df["semantic_delta"] < 0)
        &
        (df["gate"] > EPS)
    )

    # 对漏报，正确纠错方向必须是正。
    df["FN_direction_correct"] = (
        df["base_FN"]
        &
        (df["semantic_delta"] > 0)
        &
        (df["gate"] > EPS)
    )

    # ========================================================
    # required_gain:
    #
    # z_final = z_base + g * gate * delta
    #
    # 要刚好跨过0：
    #
    # g_required
    #   = |z_base| / |gate * delta|
    #
    # rho是当前g=1的纠错量相对于所需量的比例：
    #
    # rho = |gate*delta| / |z_base|
    #     = 1 / required_gain
    #
    # rho >= 1 表示当前模型理论上已经足够跨过边界。
    # ========================================================

    direction_correct = (
        df["FP_direction_correct"]
        |
        df["FN_direction_correct"]
    )

    df["required_gain"] = np.nan
    df["rho"] = np.nan

    idx = direction_correct

    df.loc[
        idx,
        "required_gain",
    ] = (
        np.abs(
            df.loc[
                idx,
                "base_logit",
            ]
        )
        /
        np.maximum(
            np.abs(
                df.loc[
                    idx,
                    "effective_residual",
                ]
            ),
            EPS,
        )
    )

    df.loc[
        idx,
        "rho",
    ] = (
        np.abs(
            df.loc[
                idx,
                "effective_residual",
            ]
        )
        /
        np.maximum(
            np.abs(
                df.loc[
                    idx,
                    "base_logit",
                ]
            ),
            EPS,
        )
    )

    # ========================================================
    # 1. 首先看方向到底准不准。
    # ========================================================

    fp = df[
        df["base_FP"]
    ]

    fn = df[
        df["base_FN"]
    ]

    fp_dir = df[
        df["FP_direction_correct"]
    ]

    fn_dir = df[
        df["FN_direction_correct"]
    ]

    print()
    print("=" * 78)
    print("Compact错误上的残差方向")
    print("=" * 78)

    print(
        f"Compact FP records = {len(fp)}"
    )

    print(
        f"FP中负方向正确 = {len(fp_dir)} "
        f"({len(fp_dir) / max(len(fp), 1) * 100:.2f}%)"
    )

    print(
        f"Compact FN records = {len(fn)}"
    )

    print(
        f"FN中正方向正确 = {len(fn_dir)} "
        f"({len(fn_dir) / max(len(fn), 1) * 100:.2f}%)"
    )

    # ========================================================
    # 2. 看方向正确时，到底缺多少幅度。
    # ========================================================

    print()
    print("=" * 78)
    print("方向正确时所需放大倍数")
    print("=" * 78)

    summarize_required_gain(
        fp_dir,
        "Compact FP + 负方向正确",
    )

    summarize_required_gain(
        fn_dir,
        "Compact FN + 正方向正确",
    )

    # ========================================================
    # 3. 逐年。
    # ========================================================

    year_rows = []

    print()
    print("=" * 78)
    print("逐年：Compact FP + 负方向正确")
    print("=" * 78)

    for year in sorted(
        df["year"].unique()
    ):
        g = fp_dir[
            fp_dir["year"] == year
        ]

        if len(g) == 0:
            continue

        req = g[
            "required_gain"
        ].to_numpy(dtype=float)

        row = {
            "year":
                int(year),

            "N":
                len(g),

            "median_required_gain":
                float(
                    np.median(req)
                ),

            "mean_required_gain":
                float(
                    np.mean(req)
                ),

            "p25_required_gain":
                float(
                    np.quantile(
                        req,
                        0.25,
                    )
                ),

            "p75_required_gain":
                float(
                    np.quantile(
                        req,
                        0.75,
                    )
                ),

            "le_1":
                int(
                    np.sum(req <= 1.0)
                ),

            "le_4":
                int(
                    np.sum(req <= 4.0)
                ),

            "le_8":
                int(
                    np.sum(req <= 8.0)
                ),

            "le_16_25":
                int(
                    np.sum(
                        req <= BEST_GNEG
                    )
                ),
        }

        year_rows.append(
            row
        )

        print(
            f"{year}: "
            f"N={len(g):4d} "
            f"median={row['median_required_gain']:.3f} "
            f"p25={row['p25_required_gain']:.3f} "
            f"p75={row['p75_required_gain']:.3f} "
            f"<=1:{row['le_1']:3d} "
            f"<=4:{row['le_4']:3d} "
            f"<=8:{row['le_8']:3d} "
            f"<=16.25:{row['le_16_25']:3d}"
        )

    pd.DataFrame(
        year_rows
    ).to_csv(
        OUT / "fp_required_gain_by_year.csv",
        index=False,
    )

    # ========================================================
    # 4. 看负方向本身是否主要对应良性。
    #
    # 这里只看有语义证据的记录。
    # ========================================================

    active = df[
        df["gate"] > EPS
    ].copy()

    active["negative_delta"] = (
        active["semantic_delta"] < 0
    )

    active["positive_delta"] = (
        active["semantic_delta"] > 0
    )

    direction_table = (
        active.groupby(
            "label"
        )
        .agg(
            N=(
                "label",
                "size",
            ),

            negative_n=(
                "negative_delta",
                "sum",
            ),

            positive_n=(
                "positive_delta",
                "sum",
            ),

            mean_delta=(
                "semantic_delta",
                "mean",
            ),

            median_delta=(
                "semantic_delta",
                "median",
            ),
        )
        .reset_index()
    )

    direction_table[
        "negative_rate"
    ] = (
        direction_table[
            "negative_n"
        ]
        /
        direction_table[
            "N"
        ]
    )

    direction_table[
        "positive_rate"
    ] = (
        direction_table[
            "positive_n"
        ]
        /
        direction_table[
            "N"
        ]
    )

    print()
    print("=" * 78)
    print("所有有语义证据样本中的残差方向")
    print("=" * 78)

    print(
        direction_table.to_string(
            index=False
        )
    )

    direction_table.to_csv(
        OUT / "direction_by_true_label.csv",
        index=False,
    )

    # ========================================================
    # 5. 保存全部诊断记录。
    # ========================================================

    keep_cols = [
        "year",
        "sha256",
        "seed",
        "label",
        "base_probability",
        "base_logit",
        "gate",
        "semantic_delta",
        "effective_residual",
        "base_FP",
        "base_FN",
        "FP_direction_correct",
        "FN_direction_correct",
        "required_gain",
        "rho",
    ]

    df[
        keep_cols
    ].to_csv(
        OUT / "per_seed_diagnostic.csv",
        index=False,
    )

    print()
    print(
        "结果保存到:",
        OUT,
    )


if __name__ == "__main__":
    main()
