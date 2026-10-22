#!/usr/bin/env python3

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    balanced_accuracy_score,
    roc_auc_score,
)


PRED = Path(
    "artifacts/"
    "stable70-stable15-selector-bounded-residual-2017-2021/"
    "future_predictions.csv"
)

OUT = Path(
    "artifacts/"
    "selector-error-ranking-diagnostic"
)

EPS = 1e-12


def logit(p):
    p = np.clip(
        np.asarray(p, dtype=float),
        EPS,
        1.0 - EPS,
    )

    return np.log(
        p / (1.0 - p)
    )


def BA(y, pred):
    return float(
        balanced_accuracy_score(
            np.asarray(y, dtype=int),
            np.asarray(pred, dtype=int),
        )
    )


def safe_auc(y, score):
    y = np.asarray(
        y,
        dtype=int,
    )

    score = np.asarray(
        score,
        dtype=float,
    )

    if len(np.unique(y)) < 2:
        return np.nan

    return float(
        roc_auc_score(
            y,
            score,
        )
    )


def main():
    OUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    df = pd.read_csv(
        PRED
    )

    y = df[
        "label"
    ].to_numpy(
        dtype=int
    )

    base_p = df[
        "base_probability"
    ].to_numpy(
        dtype=float
    )

    base_z = logit(
        base_p
    )

    gate = df[
        "gate"
    ].to_numpy(
        dtype=float
    )

    delta = df[
        "semantic_delta"
    ].to_numpy(
        dtype=float
    )

    # 新选择器版本中：
    # |semantic_delta| = q
    # 但只有feasible样本非0。
    q = np.abs(
        delta
    )

    feasible = (
        q > EPS
    )

    base_pred = (
        base_p >= 0.5
    ).astype(int)

    base_wrong = (
        base_pred != y
    )

    # Stable70当前预测的反方向。
    direction = np.where(
        base_z >= 0.0,
        -1.0,
        1.0,
    )

    rows = []

    for (year, seed), idx in df.groupby(
        [
            "year",
            "seed",
        ]
    ).groups.items():

        idx = np.asarray(
            list(idx),
            dtype=int,
        )

        f = feasible[
            idx
        ]

        wrong = base_wrong[
            idx
        ]

        qq = q[
            idx
        ]

        candidate_wrong = wrong[
            f
        ].astype(int)

        candidate_q = qq[
            f
        ]

        auc = safe_auc(
            candidate_wrong,
            candidate_q,
        )

        rows.append({
            "year":
                int(year),

            "seed":
                int(seed),

            "N":
                len(idx),

            "feasible_n":
                int(f.sum()),

            "feasible_wrong_n":
                int(
                    (
                        f
                        & wrong
                    ).sum()
                ),

            "feasible_error_rate":
                float(
                    wrong[f].mean()
                )
                if f.any()
                else np.nan,

            "selector_error_auc":
                auc,

            "q_mean_wrong":
                float(
                    qq[
                        f & wrong
                    ].mean()
                )
                if (
                    f & wrong
                ).any()
                else np.nan,

            "q_mean_correct":
                float(
                    qq[
                        f & ~wrong
                    ].mean()
                )
                if (
                    f & ~wrong
                ).any()
                else np.nan,
        })

    per_seed = pd.DataFrame(
        rows
    )

    per_seed.to_csv(
        OUT
        / "error_ranking_per_seed_year.csv",
        index=False,
    )

    by_year = (
        per_seed
        .groupby(
            "year",
            as_index=False,
        )
        .mean(
            numeric_only=True
        )
    )

    by_year.to_csv(
        OUT
        / "error_ranking_by_year.csv",
        index=False,
    )

    print()
    print(
        "=" * 110
    )
    print(
        "选择器对 Stable70 错误样本的排序能力"
    )
    print(
        "=" * 110
    )

    show = [
        "year",
        "feasible_n",
        "feasible_wrong_n",
        "feasible_error_rate",
        "selector_error_auc",
        "q_mean_wrong",
        "q_mean_correct",
    ]

    print(
        by_year[
            show
        ].to_string(
            index=False,
            float_format=lambda x: f"{x:.6f}",
        )
    )

    print()
    print(
        "四年平均 error AUC =",
        f"{by_year['selector_error_auc'].mean():.6f}",
    )

    # ========================================================
    # 第二部分：
    #
    # 完全不再使用连续q作为修正幅度。
    #
    # 如果 q >= tau：
    #     直接使用当前 gate 允许的最大幅度，
    #     将可翻转样本翻到另一边。
    #
    # 这里只用于检查q有没有排序价值。
    #
    # tau使用未来标签搜索，因此绝不能当正式结果。
    # ========================================================

    threshold_rows = []

    for tau in np.arange(
        0.05,
        0.951,
        0.01,
    ):

        year_seed_rows = []

        for (year, seed), idx in df.groupby(
            [
                "year",
                "seed",
            ]
        ).groups.items():

            idx = np.asarray(
                list(idx),
                dtype=int,
            )

            yy = y[
                idx
            ]

            zz = base_z[
                idx
            ].copy()

            selected = (
                feasible[
                    idx
                ]
                & (
                    q[
                        idx
                    ] >= tau
                )
            )

            # 使用全部现有gate。
            zz[
                selected
            ] += (
                direction[
                    idx
                ][
                    selected
                ]
                * gate[
                    idx
                ][
                    selected
                ]
            )

            pred = (
                zz >= 0.0
            ).astype(int)

            base_pred_local = (
                base_z[
                    idx
                ] >= 0.0
            ).astype(int)

            year_seed_rows.append({
                "year":
                    int(year),

                "seed":
                    int(seed),

                "BA":
                    BA(
                        yy,
                        pred,
                    ),

                "base_BA":
                    BA(
                        yy,
                        base_pred_local,
                    ),

                "selected_n":
                    int(
                        selected.sum()
                    ),
            })

        ys = pd.DataFrame(
            year_seed_rows
        )

        by_y = (
            ys.groupby(
                "year"
            )[
                [
                    "BA",
                    "base_BA",
                    "selected_n",
                ]
            ]
            .mean()
        )

        mean_ba = float(
            by_y[
                "BA"
            ].mean()
        )

        base_mean = float(
            by_y[
                "base_BA"
            ].mean()
        )

        threshold_rows.append({
            "tau":
                float(tau),

            "mean_BA":
                mean_ba,

            "gain_pp":
                100.0
                * (
                    mean_ba
                    - base_mean
                ),

            "selected_mean":
                float(
                    by_y[
                        "selected_n"
                    ].mean()
                ),

            "BA_2018":
                float(
                    by_y.loc[
                        2018,
                        "BA"
                    ]
                ),

            "BA_2019":
                float(
                    by_y.loc[
                        2019,
                        "BA"
                    ]
                ),

            "BA_2020":
                float(
                    by_y.loc[
                        2020,
                        "BA"
                    ]
                ),

            "BA_2021":
                float(
                    by_y.loc[
                        2021,
                        "BA"
                    ]
                ),
        })

    sweep = pd.DataFrame(
        threshold_rows
    ).sort_values(
        [
            "mean_BA",
            "tau",
        ],
        ascending=[
            False,
            False,
        ],
    )

    sweep.to_csv(
        OUT
        / "future_oracle_threshold_sweep.csv",
        index=False,
    )

    print()
    print(
        "=" * 110
    )
    print(
        "未来标签阈值Oracle（仅诊断，绝不能当正式模型）前15"
    )
    print(
        "=" * 110
    )

    print(
        sweep
        .head(15)
        .to_string(
            index=False,
            float_format=lambda x: f"{x:.6f}",
        )
    )

    best = sweep.iloc[
        0
    ]

    print()
    print(
        "Oracle best tau =",
        f"{best['tau']:.2f}",
    )

    print(
        "Oracle mean BA =",
        f"{best['mean_BA']:.6f}",
    )

    print(
        "Oracle gain =",
        f"{best['gain_pp']:+.3f} pp",
    )

    print()
    print(
        "结果目录:",
        OUT,
    )


if __name__ == "__main__":
    main()
