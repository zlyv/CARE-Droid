#!/usr/bin/env python3

from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.metrics import (
    balanced_accuracy_score,
    roc_auc_score,
)

INPUT = Path(
    "artifacts/"
    "frozen-anchor-residual-count10-150-2017-2021/"
    "future_predictions.csv"
)

OUT = Path(
    "artifacts/"
    "minimal-negative-crossing"
)

EPS_P = 1e-7
EPS_Z = 1e-6

G_CAP = 16.25
YEARS = (2018, 2019, 2020, 2021)


def logit(p):
    p = np.clip(
        np.asarray(p, dtype=float),
        EPS_P,
        1.0 - EPS_P,
    )
    return np.log(
        p / (1.0 - p)
    )


def sigmoid(z):
    z = np.clip(
        np.asarray(z, dtype=float),
        -60.0,
        60.0,
    )
    return 1.0 / (
        1.0 + np.exp(-z)
    )


def build_probabilities(df):
    base_p = df[
        "base_probability"
    ].to_numpy(dtype=float)

    base_z = logit(
        base_p
    )

    r = (
        df["gate"].to_numpy(dtype=float)
        *
        df["semantic_delta"].to_numpy(dtype=float)
    )

    # 当前正式模型。
    current_z = (
        base_z + r
    )

    # 固定16.25倍负方向增益。
    fixed_z = np.where(
        r < 0,
        base_z + G_CAP * r,
        current_z,
    )

    # ========================================================
    # 最小跨界版本。
    #
    # 只处理：
    # 1. Compact原来在恶意侧
    # 2. 当前g=1还没有跨界
    # 3. 残差方向为负
    # 4. 所需倍数 <= 16.25
    #
    # 其余全部维持当前模型。
    # ========================================================

    required_gain = np.full(
        len(df),
        np.inf,
        dtype=float,
    )

    valid = (
        (base_z > 0)
        &
        (r < 0)
    )

    required_gain[
        valid
    ] = (
        base_z[valid]
        /
        (-r[valid])
    )

    eligible = (
        valid
        &
        (current_z >= 0)
        &
        (required_gain <= G_CAP)
    )

    minimal_z = (
        current_z.copy()
    )

    # 只刚刚越过0。
    minimal_z[
        eligible
    ] = -EPS_Z

    return {
        "current":
            sigmoid(current_z),

        "fixed_negative_gain":
            sigmoid(fixed_z),

        "minimal_crossing":
            sigmoid(minimal_z),

        "eligible":
            eligible,

        "required_gain":
            required_gain,
    }


def evaluate(
    df,
    probability,
):
    work = df[
        [
            "year",
            "seed",
            "label",
        ]
    ].copy()

    work[
        "probability"
    ] = probability

    rows = []

    for (
        year,
        seed,
    ), g in work.groupby(
        [
            "year",
            "seed",
        ],
        sort=True,
    ):
        y = g[
            "label"
        ].to_numpy(dtype=int)

        p = g[
            "probability"
        ].to_numpy(dtype=float)

        pred = (
            p >= 0.5
        ).astype(int)

        rows.append({
            "year":
                int(year),

            "seed":
                int(seed),

            "BA":
                float(
                    balanced_accuracy_score(
                        y,
                        pred,
                    )
                ),

            "AUC":
                float(
                    roc_auc_score(
                        y,
                        p,
                    )
                ),

            "FP":
                int(
                    (
                        (y == 0)
                        &
                        (pred == 1)
                    ).sum()
                ),

            "FN":
                int(
                    (
                        (y == 1)
                        &
                        (pred == 0)
                    ).sum()
                ),
        })

    per_seed = pd.DataFrame(
        rows
    )

    per_year = (
        per_seed
        .groupby(
            "year",
            as_index=False,
        )
        .agg(
            BA=("BA", "mean"),
            AUC=("AUC", "mean"),
            FP=("FP", "mean"),
            FN=("FN", "mean"),
        )
    )

    return (
        per_seed,
        per_year,
    )


def main():
    OUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    df = pd.read_csv(
        INPUT
    )

    values = build_probabilities(
        df
    )

    summary_rows = []
    all_year = []

    for name in (
        "current",
        "fixed_negative_gain",
        "minimal_crossing",
    ):
        (
            per_seed,
            per_year,
        ) = evaluate(
            df,
            values[name],
        )

        per_seed[
            "variant"
        ] = name

        per_year[
            "variant"
        ] = name

        per_seed.to_csv(
            OUT
            /
            f"{name}_per_seed.csv",
            index=False,
        )

        all_year.append(
            per_year
        )

        summary_rows.append({
            "variant":
                name,

            "mean_BA":
                float(
                    per_year["BA"].mean()
                ),

            "min_BA":
                float(
                    per_year["BA"].min()
                ),

            "mean_AUC":
                float(
                    per_year["AUC"].mean()
                ),

            "mean_FP":
                float(
                    per_year["FP"].mean()
                ),

            "mean_FN":
                float(
                    per_year["FN"].mean()
                ),
        })

    summary = pd.DataFrame(
        summary_rows
    )

    per_year_all = pd.concat(
        all_year,
        ignore_index=True,
    )

    summary.to_csv(
        OUT / "summary.csv",
        index=False,
    )

    per_year_all.to_csv(
        OUT / "per_year.csv",
        index=False,
    )

    # 检查固定增益和最小跨界的0.5分类是否完全相同。
    fixed_pred = (
        values[
            "fixed_negative_gain"
        ] >= 0.5
    )

    minimal_pred = (
        values[
            "minimal_crossing"
        ] >= 0.5
    )

    prediction_diff = int(
        np.sum(
            fixed_pred
            != minimal_pred
        )
    )

    eligible = values[
        "eligible"
    ]

    print(
        "eligible records =",
        int(
            eligible.sum()
        ),
    )

    print(
        "fixed vs minimal prediction diff =",
        prediction_diff,
    )

    print()
    print("=" * 76)
    print("总结果")
    print("=" * 76)

    print(
        summary.to_string(
            index=False
        )
    )

    print()
    print("=" * 76)
    print("逐年结果")
    print("=" * 76)

    for name in (
        "current",
        "fixed_negative_gain",
        "minimal_crossing",
    ):
        print()
        print(name)

        g = per_year_all[
            per_year_all[
                "variant"
            ] == name
        ]

        for _, r in g.iterrows():
            print(
                f"{int(r['year'])}: "
                f"BA={r['BA']:.6f} "
                f"AUC={r['AUC']:.6f} "
                f"FP={r['FP']:.2f} "
                f"FN={r['FN']:.2f}"
            )


if __name__ == "__main__":
    main()
