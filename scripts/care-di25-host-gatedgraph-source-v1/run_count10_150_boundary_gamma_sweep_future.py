#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
)


INPUT = Path(
    "artifacts/"
    "frozen-anchor-residual-count10-150-2017-2021/"
    "future_predictions.csv"
)

OUT = Path(
    "artifacts/"
    "frozen-count10-150-boundary-gamma-sweep"
)

THRESHOLD = 0.5
EPS = 1e-7

YEARS = (
    2018,
    2019,
    2020,
    2021,
)

TAUS = (
    0.050,
    0.075,
    0.100,
    0.125,
    0.150,
    0.175,
    0.200,
    0.250,
)

GAMMAS = (
    1.25,
    1.50,
    1.75,
    2.00,
    2.25,
    2.50,
    2.75,
    3.00,
    3.50,
    4.00,
)


def logit(p):
    p = np.clip(
        np.asarray(p, dtype=float),
        EPS,
        1.0 - EPS,
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


def get_metrics(
    y,
    probability,
):
    y = np.asarray(
        y,
        dtype=int,
    )

    probability = np.asarray(
        probability,
        dtype=float,
    )

    prediction = (
        probability >= THRESHOLD
    ).astype(int)

    return {
        "BA":
            float(
                balanced_accuracy_score(
                    y,
                    prediction,
                )
            ),

        "AUC":
            float(
                roc_auc_score(
                    y,
                    probability,
                )
            ),

        "ACC":
            float(
                accuracy_score(
                    y,
                    prediction,
                )
            ),

        "F1":
            float(
                f1_score(
                    y,
                    prediction,
                )
            ),

        "FP":
            int(
                (
                    (y == 0)
                    &
                    (prediction == 1)
                ).sum()
            ),

        "FN":
            int(
                (
                    (y == 1)
                    &
                    (prediction == 0)
                ).sum()
            ),
    }


def probability_for(
    df,
    tau,
    gamma,
):
    base_p = (
        df[
            "base_probability"
        ].to_numpy(
            dtype=float
        )
    )

    base_z = logit(
        base_p
    )

    margin = np.abs(
        base_p - 0.5
    )

    multiplier = np.where(
        margin <= tau,
        float(gamma),
        1.0,
    )

    residual = (
        df[
            "gate"
        ].to_numpy(
            dtype=float
        )
        *
        df[
            "semantic_delta"
        ].to_numpy(
            dtype=float
        )
    )

    final_z = (
        base_z
        + multiplier
        * residual
    )

    return (
        sigmoid(
            final_z
        ),
        multiplier,
        margin,
    )


def evaluate(
    df,
    tau,
    gamma,
):
    work = df.copy()

    (
        probability,
        multiplier,
        margin,
    ) = probability_for(
        work,
        tau,
        gamma,
    )

    work[
        "candidate_probability"
    ] = probability

    work[
        "multiplier"
    ] = multiplier

    work[
        "base_margin"
    ] = margin

    seed_rows = []

    # ========================================================
    # 正式协议：
    # year × seed 分别计算，
    # 再对5个seed求平均。
    # ========================================================

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
        m = get_metrics(
            g[
                "label"
            ].to_numpy(
                dtype=int
            ),
            g[
                "candidate_probability"
            ].to_numpy(
                dtype=float
            ),
        )

        seed_rows.append({
            "tau":
                tau,

            "gamma":
                gamma,

            "year":
                int(year),

            "seed":
                int(seed),

            **m,
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
            "tau":
                tau,

            "gamma":
                gamma,

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

            "ACC":
                float(
                    g[
                        "ACC"
                    ].mean()
                ),

            "F1":
                float(
                    g[
                        "F1"
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
        "tau":
            tau,

        "gamma":
            gamma,

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

        "mean_ACC":
            float(
                year_df[
                    "ACC"
                ].mean()
            ),
    }

    for year in YEARS:
        yr = year_df[
            year_df[
                "year"
            ] == year
        ].iloc[0]

        row[
            f"BA_{year}"
        ] = float(
            yr["BA"]
        )

        row[
            f"AUC_{year}"
        ] = float(
            yr["AUC"]
        )

        row[
            f"FP_{year}"
        ] = float(
            yr["FP"]
        )

        row[
            f"FN_{year}"
        ] = float(
            yr["FN"]
        )

    return (
        row,
        year_df,
        seed_df,
    )


def averaged_sample_diagnostic(
    df,
    tau,
    gamma,
):
    work = df.copy()

    (
        cp,
        _,
        _,
    ) = probability_for(
        work,
        tau,
        gamma,
    )

    # 当前γ=1。
    (
        bp,
        _,
        _,
    ) = probability_for(
        work,
        tau,
        1.0,
    )

    work[
        "candidate_probability"
    ] = cp

    work[
        "baseline_probability"
    ] = bp

    avg = (
        work.groupby(
            [
                "year",
                "sha256",
                "label",
            ],
            as_index=False,
        )
        .agg(
            candidate_probability=(
                "candidate_probability",
                "mean",
            ),

            baseline_probability=(
                "baseline_probability",
                "mean",
            ),
        )
    )

    avg[
        "candidate_pred"
    ] = (
        avg[
            "candidate_probability"
        ] >= 0.5
    ).astype(int)

    avg[
        "baseline_pred"
    ] = (
        avg[
            "baseline_probability"
        ] >= 0.5
    ).astype(int)

    avg[
        "candidate_correct"
    ] = (
        avg[
            "candidate_pred"
        ]
        ==
        avg[
            "label"
        ]
    )

    avg[
        "baseline_correct"
    ] = (
        avg[
            "baseline_pred"
        ]
        ==
        avg[
            "label"
        ]
    )

    avg[
        "corrected"
    ] = (
        (~avg[
            "baseline_correct"
        ])
        &
        avg[
            "candidate_correct"
        ]
    )

    avg[
        "regressed"
    ] = (
        avg[
            "baseline_correct"
        ]
        &
        (~avg[
            "candidate_correct"
        ])
    )

    rows = []

    for year_value in (
        list(YEARS)
        + ["ALL"]
    ):
        if year_value == "ALL":
            g = avg
        else:
            g = avg[
                avg[
                    "year"
                ] == year_value
            ]

        corrected = int(
            g[
                "corrected"
            ].sum()
        )

        regressed = int(
            g[
                "regressed"
            ].sum()
        )

        rows.append({
            "year":
                year_value,

            "N":
                len(g),

            "corrected":
                corrected,

            "regressed":
                regressed,

            "net":
                corrected
                - regressed,

            "corr_benign":
                int(
                    (
                        g[
                            "corrected"
                        ]
                        &
                        (
                            g[
                                "label"
                            ] == 0
                        )
                    ).sum()
                ),

            "corr_malware":
                int(
                    (
                        g[
                            "corrected"
                        ]
                        &
                        (
                            g[
                                "label"
                            ] == 1
                        )
                    ).sum()
                ),

            "reg_benign":
                int(
                    (
                        g[
                            "regressed"
                        ]
                        &
                        (
                            g[
                                "label"
                            ] == 0
                        )
                    ).sum()
                ),

            "reg_malware":
                int(
                    (
                        g[
                            "regressed"
                        ]
                        &
                        (
                            g[
                                "label"
                            ] == 1
                        )
                    ).sum()
                ),
        })

    return (
        pd.DataFrame(
            rows
        ),
        avg,
    )


def main():
    if not INPUT.exists():
        raise SystemExit(
            "找不到输入文件"
        )

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
    )

    print(
        "APK =",
        df[
            "sha256"
        ].nunique(),
    )

    # ========================================================
    # 当前基线
    # ========================================================

    baseline_row, _, _ = (
        evaluate(
            df,
            0.20,
            1.0,
        )
    )

    print()
    print(
        "当前基线："
    )

    print(
        f"meanBA={baseline_row['mean_BA']:.6f} "
        f"minBA={baseline_row['min_BA']:.6f} "
        f"AUC={baseline_row['mean_AUC']:.6f}"
    )

    rows = []
    year_frames = []
    seed_frames = []

    for tau in TAUS:
        for gamma in GAMMAS:

            (
                row,
                year_df,
                seed_df,
            ) = evaluate(
                df,
                tau,
                gamma,
            )

            rows.append(
                row
            )

            year_frames.append(
                year_df
            )

            seed_frames.append(
                seed_df
            )

    summary = pd.DataFrame(
        rows
    )

    # ========================================================
    # 方案1：
    # 完全以平均BA最大化。
    # ========================================================

    ranked_ba = (
        summary.sort_values(
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

    best_ba = ranked_ba.iloc[
        0
    ]

    # ========================================================
    # 方案2：
    # 保守版本：
    #
    # 平均AUC最多损失0.005，
    # 最低BA最多损失0.001。
    #
    # 在满足约束的候选里最大化平均BA。
    # ========================================================

    auc_floor = (
        baseline_row[
            "mean_AUC"
        ]
        - 0.005
    )

    min_ba_floor = (
        baseline_row[
            "min_BA"
        ]
        - 0.001
    )

    conservative = summary[
        (
            summary[
                "mean_AUC"
            ]
            >= auc_floor
        )
        &
        (
            summary[
                "min_BA"
            ]
            >= min_ba_floor
        )
    ].copy()

    if len(
        conservative
    ):
        conservative = (
            conservative.sort_values(
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

        best_cons = (
            conservative.iloc[
                0
            ]
        )

    else:
        best_cons = None

    # ========================================================
    # 保存
    # ========================================================

    summary.to_csv(
        OUT
        / "summary.csv",
        index=False,
    )

    ranked_ba.to_csv(
        OUT
        / "ranked_by_BA.csv",
        index=False,
    )

    conservative.to_csv(
        OUT
        / "conservative_candidates.csv",
        index=False,
    )

    pd.concat(
        year_frames,
        ignore_index=True,
    ).to_csv(
        OUT
        / "per_year.csv",
        index=False,
    )

    pd.concat(
        seed_frames,
        ignore_index=True,
    ).to_csv(
        OUT
        / "per_seed.csv",
        index=False,
    )

    # ========================================================
    # 最佳BA样本诊断
    # ========================================================

    (
        diag_summary,
        diag_samples,
    ) = averaged_sample_diagnostic(
        df,
        float(
            best_ba[
                "tau"
            ]
        ),
        float(
            best_ba[
                "gamma"
            ]
        ),
    )

    diag_summary.to_csv(
        OUT
        / "best_BA_correction_summary.csv",
        index=False,
    )

    diag_samples.to_csv(
        OUT
        / "best_BA_sample_diagnostic.csv",
        index=False,
    )

    # ========================================================
    # 打印前10
    # ========================================================

    print()
    print(
        "======================================"
    )
    print(
        "平均BA最高的前10个"
    )
    print(
        "======================================"
    )

    for _, r in (
        ranked_ba.head(
            10
        ).iterrows()
    ):
        print(
            f"tau={r['tau']:.3f} "
            f"gamma={r['gamma']:.2f} "
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
        "======================================"
    )
    print(
        "平均BA绝对最佳"
    )
    print(
        "======================================"
    )

    print(
        f"tau={best_ba['tau']:.3f}"
    )

    print(
        f"gamma={best_ba['gamma']:.2f}"
    )

    print(
        f"meanBA "
        f"{baseline_row['mean_BA']:.6f}"
        f" -> "
        f"{best_ba['mean_BA']:.6f} "
        f"Δ="
        f"{best_ba['mean_BA']-baseline_row['mean_BA']:+.6f}"
    )

    print(
        f"minBA "
        f"{baseline_row['min_BA']:.6f}"
        f" -> "
        f"{best_ba['min_BA']:.6f}"
    )

    print(
        f"AUC "
        f"{baseline_row['mean_AUC']:.6f}"
        f" -> "
        f"{best_ba['mean_AUC']:.6f}"
    )

    print()
    print(
        "======================================"
    )
    print(
        "保守约束下最佳"
    )
    print(
        "======================================"
    )

    if best_cons is None:
        print(
            "没有候选满足约束"
        )

    else:
        print(
            f"tau={best_cons['tau']:.3f}"
        )

        print(
            f"gamma={best_cons['gamma']:.2f}"
        )

        print(
            f"meanBA="
            f"{best_cons['mean_BA']:.6f}"
        )

        print(
            f"minBA="
            f"{best_cons['min_BA']:.6f}"
        )

        print(
            f"AUC="
            f"{best_cons['mean_AUC']:.6f}"
        )

    print()
    print(
        "======================================"
    )
    print(
        "绝对最佳相对当前模型的样本变化"
    )
    print(
        "======================================"
    )

    for _, r in (
        diag_summary.iterrows()
    ):
        print(
            f"{str(r['year']):>4s} "
            f"corrected="
            f"{int(r['corrected']):3d} "
            f"regressed="
            f"{int(r['regressed']):3d} "
            f"net="
            f"{int(r['net']):+3d} "
            f"corrB="
            f"{int(r['corr_benign']):3d} "
            f"corrM="
            f"{int(r['corr_malware']):3d}"
        )


if __name__ == "__main__":
    main()
