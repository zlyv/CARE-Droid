#!/usr/bin/env python3
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.metrics import (
    balanced_accuracy_score,
    roc_auc_score,
)

import run_count10_150_directional_gamma_sweep_future as exp


INPUT = exp.INPUT

OUT = Path(
    "artifacts/"
    "negative-gain-extended-threshold-control"
)

# ============================================================
# 扩大搜索：
#
# tau最大直接扫到0.50，
# 因为 |p-0.5| 理论最大就是0.5。
#
# g-扩大到20。
# ============================================================

TAUS = np.round(
    np.arange(
        0.25,
        0.5001,
        0.025,
    ),
    3,
)

NEG_GAMMAS = np.round(
    np.arange(
        4.00,
        20.0001,
        0.25,
    ),
    2,
)

# 正方向继续固定。
POS_GAMMA = 1.0

# 全局阈值对照。
THRESHOLDS = np.round(
    np.arange(
        0.40,
        0.7001,
        0.001,
    ),
    3,
)

YEARS = (
    2018,
    2019,
    2020,
    2021,
)


def threshold_evaluate(
    df,
    threshold,
):
    """
    不改任何模型概率，
    只改变全局分类阈值。

    正式协议：
        每年×每seed算BA
        再对seed平均。
    """

    seed_rows = []

    for (
        year,
        seed,
    ), g in df.groupby(
        [
            "year",
            "seed",
        ],
        sort=True,
    ):
        y = (
            g[
                "label"
            ]
            .to_numpy(
                dtype=int
            )
        )

        p = (
            g[
                "probability"
            ]
            .to_numpy(
                dtype=float
            )
        )

        pred = (
            p >= threshold
        ).astype(int)

        ba = float(
            balanced_accuracy_score(
                y,
                pred,
            )
        )

        auc = float(
            roc_auc_score(
                y,
                p,
            )
        )

        fp = int(
            (
                (y == 0)
                &
                (pred == 1)
            ).sum()
        )

        fn = int(
            (
                (y == 1)
                &
                (pred == 0)
            ).sum()
        )

        seed_rows.append({
            "threshold":
                float(
                    threshold
                ),

            "year":
                int(year),

            "seed":
                int(seed),

            "BA":
                ba,

            "AUC":
                auc,

            "FP":
                fp,

            "FN":
                fn,
        })

    seed_df = pd.DataFrame(
        seed_rows
    )

    year_rows = []

    for year, g in seed_df.groupby(
        "year",
        sort=True,
    ):
        year_rows.append({
            "threshold":
                float(
                    threshold
                ),

            "year":
                int(year),

            "BA":
                float(
                    g[
                        "BA"
                    ].mean()
                ),

            "AUC":
                float(
                    g[
                        "AUC"
                    ].mean()
                ),

            "FP":
                float(
                    g[
                        "FP"
                    ].mean()
                ),

            "FN":
                float(
                    g[
                        "FN"
                    ].mean()
                ),
        })

    year_df = pd.DataFrame(
        year_rows
    )

    row = {
        "threshold":
            float(
                threshold
            ),

        "mean_BA":
            float(
                year_df[
                    "BA"
                ].mean()
            ),

        "min_BA":
            float(
                year_df[
                    "BA"
                ].min()
            ),

        "mean_AUC":
            float(
                year_df[
                    "AUC"
                ].mean()
            ),
    }

    for year in YEARS:
        r = year_df[
            year_df[
                "year"
            ] == year
        ].iloc[0]

        row[
            f"BA_{year}"
        ] = float(
            r[
                "BA"
            ]
        )

        row[
            f"FP_{year}"
        ] = float(
            r[
                "FP"
            ]
        )

        row[
            f"FN_{year}"
        ] = float(
            r[
                "FN"
            ]
        )

    return row


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
        df[
            "sha256"
        ].nunique(),
    )

    # ========================================================
    # 当前0.5阈值正式基线。
    # ========================================================

    exp.TAU = 0.20

    baseline = exp.evaluate(
        df,
        1.0,
        1.0,
    )

    print()
    print(
        "当前基线:"
    )

    print(
        f"meanBA={baseline['mean_BA']:.6f} "
        f"minBA={baseline['min_BA']:.6f} "
        f"AUC={baseline['mean_AUC']:.6f}"
    )

    # ========================================================
    # 第一部分：
    # 扩大 tau × g- 搜索。
    # ========================================================

    negative_rows = []

    for tau in TAUS:

        exp.TAU = float(
            tau
        )

        for gamma_neg in NEG_GAMMAS:

            r = exp.evaluate(
                df,
                float(
                    gamma_neg
                ),
                POS_GAMMA,
            )

            r[
                "tau"
            ] = float(
                tau
            )

            negative_rows.append(
                r
            )

    negative_df = pd.DataFrame(
        negative_rows
    )

    negative_ranked = (
        negative_df
        .sort_values(
            [
                "mean_BA",
                "min_BA",
                "mean_AUC",
            ],
            ascending=[
                False,
                False,
                False,
            ],
        )
        .reset_index(
            drop=True
        )
    )

    negative_ranked.to_csv(
        OUT
        / "negative_gain_ranked.csv",
        index=False,
    )

    best_negative = (
        negative_ranked.iloc[
            0
        ]
    )

    # ========================================================
    # 每个tau自己的最佳g-
    # ========================================================

    best_per_tau = []

    for tau, g in negative_df.groupby(
        "tau",
        sort=True,
    ):

        r = (
            g.sort_values(
                [
                    "mean_BA",
                    "min_BA",
                    "mean_AUC",
                ],
                ascending=[
                    False,
                    False,
                    False,
                ],
            )
            .iloc[0]
        )

        best_per_tau.append(
            r.to_dict()
        )

    best_per_tau_df = pd.DataFrame(
        best_per_tau
    )

    best_per_tau_df.to_csv(
        OUT
        / "negative_gain_best_per_tau.csv",
        index=False,
    )

    # ========================================================
    # 第二部分：
    # 原始当前模型全局阈值扫描。
    #
    # 注意这里只改变threshold，
    # probability完全不修改。
    # ========================================================

    threshold_rows = []

    for threshold in THRESHOLDS:

        threshold_rows.append(
            threshold_evaluate(
                df,
                float(
                    threshold
                ),
            )
        )

    threshold_df = pd.DataFrame(
        threshold_rows
    )

    threshold_ranked = (
        threshold_df
        .sort_values(
            [
                "mean_BA",
                "min_BA",
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

    threshold_ranked.to_csv(
        OUT
        / "threshold_ranked.csv",
        index=False,
    )

    best_threshold = (
        threshold_ranked.iloc[
            0
        ]
    )

    # ========================================================
    # 保存一个总比较
    # ========================================================

    comparison = pd.DataFrame([
        {
            "method":
                "current_threshold_0.5",

            "mean_BA":
                baseline[
                    "mean_BA"
                ],

            "min_BA":
                baseline[
                    "min_BA"
                ],

            "mean_AUC":
                baseline[
                    "mean_AUC"
                ],

            "parameter":
                "threshold=0.5",
        },

        {
            "method":
                "negative_residual_gain",

            "mean_BA":
                best_negative[
                    "mean_BA"
                ],

            "min_BA":
                best_negative[
                    "min_BA"
                ],

            "mean_AUC":
                best_negative[
                    "mean_AUC"
                ],

            "parameter":
                (
                    f"tau={best_negative['tau']:.3f},"
                    f"g-={best_negative['gamma_neg']:.2f},"
                    f"g+=1"
                ),
        },

        {
            "method":
                "global_threshold_control",

            "mean_BA":
                best_threshold[
                    "mean_BA"
                ],

            "min_BA":
                best_threshold[
                    "min_BA"
                ],

            "mean_AUC":
                best_threshold[
                    "mean_AUC"
                ],

            "parameter":
                (
                    f"threshold="
                    f"{best_threshold['threshold']:.3f}"
                ),
        },
    ])

    comparison.to_csv(
        OUT
        / "main_comparison.csv",
        index=False,
    )

    # ========================================================
    # 输出
    # ========================================================

    print()
    print(
        "=" * 78
    )
    print(
        "扩大负方向搜索：前15"
    )
    print(
        "=" * 78
    )

    for _, r in (
        negative_ranked.head(
            15
        ).iterrows()
    ):

        print(
            f"tau={r['tau']:.3f} "
            f"g-={r['gamma_neg']:.2f} "
            f"meanBA={r['mean_BA']:.6f} "
            f"minBA={r['min_BA']:.6f} "
            f"AUC={r['mean_AUC']:.6f} "
            f"BA18={r['BA_2018']:.6f} "
            f"BA19={r['BA_2019']:.6f} "
            f"BA20={r['BA_2020']:.6f} "
            f"BA21={r['BA_2021']:.6f}"
        )

    print()
    print(
        "=" * 78
    )
    print(
        "各tau最佳"
    )
    print(
        "=" * 78
    )

    for _, r in (
        best_per_tau_df.iterrows()
    ):

        print(
            f"tau={r['tau']:.3f} "
            f"g-={r['gamma_neg']:.2f} "
            f"meanBA={r['mean_BA']:.6f} "
            f"minBA={r['min_BA']:.6f} "
            f"AUC={r['mean_AUC']:.6f}"
        )

    print()
    print(
        "=" * 78
    )
    print(
        "扩大搜索绝对最佳"
    )
    print(
        "=" * 78
    )

    print(
        f"tau="
        f"{best_negative['tau']:.3f}"
    )

    print(
        f"g-="
        f"{best_negative['gamma_neg']:.2f}"
    )

    print(
        f"meanBA="
        f"{best_negative['mean_BA']:.6f}"
    )

    print(
        f"minBA="
        f"{best_negative['min_BA']:.6f}"
    )

    print(
        f"AUC="
        f"{best_negative['mean_AUC']:.6f}"
    )

    print()
    print(
        "=" * 78
    )
    print(
        "全局阈值对照最佳"
    )
    print(
        "=" * 78
    )

    print(
        f"threshold="
        f"{best_threshold['threshold']:.3f}"
    )

    print(
        f"meanBA="
        f"{best_threshold['mean_BA']:.6f}"
    )

    print(
        f"minBA="
        f"{best_threshold['min_BA']:.6f}"
    )

    print(
        f"AUC="
        f"{best_threshold['mean_AUC']:.6f}"
    )

    for year in YEARS:

        print(
            f"{year}: "
            f"BA="
            f"{best_threshold[f'BA_{year}']:.6f} "
            f"FP="
            f"{best_threshold[f'FP_{year}']:.2f} "
            f"FN="
            f"{best_threshold[f'FN_{year}']:.2f}"
        )

    print()
    print(
        "=" * 78
    )
    print(
        "主比较"
    )
    print(
        "=" * 78
    )

    print(
        comparison.to_string(
            index=False
        )
    )


if __name__ == "__main__":
    main()
