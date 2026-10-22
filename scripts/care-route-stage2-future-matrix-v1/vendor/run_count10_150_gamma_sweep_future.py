#!/usr/bin/env python3
from __future__ import annotations

import json
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
    "frozen-count10-150-gamma-sweep"
)

THRESHOLD = 0.5
EPS = 1e-7

YEARS = (
    2018,
    2019,
    2020,
    2021,
)

# 当前正式结果，用于检查γ=1是否复现。
EXPECTED = {
    2018: {
        "BA": 0.922608,
        "AUC": 0.964549,
    },
    2019: {
        "BA": 0.902940,
        "AUC": 0.967050,
    },
    2020: {
        "BA": 0.878547,
        "AUC": 0.916146,
    },
    2021: {
        "BA": 0.866000,
        "AUC": 0.921580,
    },
}


def safe_logit(p):
    p = np.clip(
        np.asarray(p, dtype=float),
        EPS,
        1.0 - EPS,
    )

    return np.log(
        p / (1.0 - p)
    )


def sigmoid(z):
    z = np.asarray(
        z,
        dtype=float,
    )

    # 当前logit范围很正常，
    # 这里只做数值保险。
    z = np.clip(
        z,
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


def metrics(
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
        probability
        >= THRESHOLD
    ).astype(int)

    fp = int(
        np.sum(
            (y == 0)
            &
            (prediction == 1)
        )
    )

    fn = int(
        np.sum(
            (y == 1)
            &
            (prediction == 0)
        )
    )

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
            fp,

        "FN":
            fn,
    }


def gamma_probability(
    df,
    gamma,
):
    base_logit = safe_logit(
        df[
            "base_probability"
        ].to_numpy(dtype=float)
    )

    residual = (
        df[
            "gate"
        ].to_numpy(dtype=float)
        *
        df[
            "semantic_delta"
        ].to_numpy(dtype=float)
    )

    final_logit = (
        base_logit
        + float(gamma)
        * residual
    )

    return sigmoid(
        final_logit
    )


def evaluate_gamma(
    df,
    gamma,
):
    work = df.copy()

    work[
        "gamma_probability"
    ] = gamma_probability(
        work,
        gamma,
    )

    per_seed_rows = []

    # ========================================================
    # 正式协议：
    #
    # 每一个 year × seed 单独计算指标，
    # 然后再在同一年对5个seed求平均。
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
        m = metrics(
            g[
                "label"
            ].to_numpy(dtype=int),
            g[
                "gamma_probability"
            ].to_numpy(dtype=float),
        )

        per_seed_rows.append({
            "gamma":
                float(gamma),

            "year":
                int(year),

            "seed":
                int(seed),

            "N":
                len(g),

            **m,
        })

    per_seed = pd.DataFrame(
        per_seed_rows
    )

    per_year_rows = []

    for year, g in per_seed.groupby(
        "year",
        sort=True,
    ):
        per_year_rows.append({
            "gamma":
                float(gamma),

            "year":
                int(year),

            "seed_n":
                len(g),

            "BA":
                float(
                    g["BA"].mean()
                ),

            "AUC":
                float(
                    g["AUC"].mean()
                ),

            "ACC":
                float(
                    g["ACC"].mean()
                ),

            "F1":
                float(
                    g["F1"].mean()
                ),

            "FP":
                float(
                    g["FP"].mean()
                ),

            "FN":
                float(
                    g["FN"].mean()
                ),
        })

    per_year = pd.DataFrame(
        per_year_rows
    )

    summary = {
        "gamma":
            float(gamma),

        "mean_BA":
            float(
                per_year[
                    "BA"
                ].mean()
            ),

        "min_BA":
            float(
                per_year[
                    "BA"
                ].min()
            ),

        "mean_AUC":
            float(
                per_year[
                    "AUC"
                ].mean()
            ),

        "mean_ACC":
            float(
                per_year[
                    "ACC"
                ].mean()
            ),

        "mean_F1":
            float(
                per_year[
                    "F1"
                ].mean()
            ),

        "mean_FP":
            float(
                per_year[
                    "FP"
                ].mean()
            ),

        "mean_FN":
            float(
                per_year[
                    "FN"
                ].mean()
            ),
    }

    for year in YEARS:
        r = per_year[
            per_year[
                "year"
            ]
            == year
        ].iloc[0]

        summary[
            f"BA_{year}"
        ] = float(
            r[
                "BA"
            ]
        )

        summary[
            f"AUC_{year}"
        ] = float(
            r[
                "AUC"
            ]
        )

        summary[
            f"FP_{year}"
        ] = float(
            r[
                "FP"
            ]
        )

        summary[
            f"FN_{year}"
        ] = float(
            r[
                "FN"
            ]
        )

    return (
        summary,
        per_year,
        per_seed,
        work,
    )


def averaged_probability_diagnostic(
    df,
    gamma,
    reference_gamma=1.0,
):
    """
    只用于样本级 corrected / regressed。

    这里按照：
        5个seed概率平均
        -> threshold=.5

    不用于正式BA选择。
    """

    a = df.copy()

    a[
        "candidate_probability"
    ] = gamma_probability(
        a,
        gamma,
    )

    a[
        "reference_probability"
    ] = gamma_probability(
        a,
        reference_gamma,
    )

    grouped = (
        a.groupby(
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
            reference_probability=(
                "reference_probability",
                "mean",
            ),
        )
    )

    grouped[
        "candidate_prediction"
    ] = (
        grouped[
            "candidate_probability"
        ]
        >= THRESHOLD
    ).astype(int)

    grouped[
        "reference_prediction"
    ] = (
        grouped[
            "reference_probability"
        ]
        >= THRESHOLD
    ).astype(int)

    grouped[
        "candidate_correct"
    ] = (
        grouped[
            "candidate_prediction"
        ]
        ==
        grouped[
            "label"
        ]
    )

    grouped[
        "reference_correct"
    ] = (
        grouped[
            "reference_prediction"
        ]
        ==
        grouped[
            "label"
        ]
    )

    grouped[
        "corrected"
    ] = (
        (~grouped[
            "reference_correct"
        ])
        &
        grouped[
            "candidate_correct"
        ]
    )

    grouped[
        "regressed"
    ] = (
        grouped[
            "reference_correct"
        ]
        &
        (~grouped[
            "candidate_correct"
        ])
    )

    grouped[
        "corrected_FP"
    ] = (
        grouped[
            "corrected"
        ]
        &
        (
            grouped[
                "label"
            ]
            == 0
        )
    )

    grouped[
        "corrected_FN"
    ] = (
        grouped[
            "corrected"
        ]
        &
        (
            grouped[
                "label"
            ]
            == 1
        )
    )

    grouped[
        "regressed_FP"
    ] = (
        grouped[
            "regressed"
        ]
        &
        (
            grouped[
                "label"
            ]
            == 0
        )
    )

    grouped[
        "regressed_FN"
    ] = (
        grouped[
            "regressed"
        ]
        &
        (
            grouped[
                "label"
            ]
            == 1
        )
    )

    rows = []

    for year_value in (
        list(YEARS)
        + ["ALL"]
    ):
        if year_value == "ALL":
            g = grouped
        else:
            g = grouped[
                grouped[
                    "year"
                ]
                == year_value
            ]

        rows.append({
            "gamma":
                float(gamma),

            "year":
                year_value,

            "N":
                len(g),

            "corrected":
                int(
                    g[
                        "corrected"
                    ].sum()
                ),

            "regressed":
                int(
                    g[
                        "regressed"
                    ].sum()
                ),

            "net":
                int(
                    g[
                        "corrected"
                    ].sum()
                    -
                    g[
                        "regressed"
                    ].sum()
                ),

            "corrected_FP":
                int(
                    g[
                        "corrected_FP"
                    ].sum()
                ),

            "corrected_FN":
                int(
                    g[
                        "corrected_FN"
                    ].sum()
                ),

            "regressed_FP":
                int(
                    g[
                        "regressed_FP"
                    ].sum()
                ),

            "regressed_FN":
                int(
                    g[
                        "regressed_FN"
                    ].sum()
                ),
        })

    return (
        pd.DataFrame(
            rows
        ),
        grouped,
    )


def rank_summary(df):
    """
    开发集上的选择规则：

    1. 4年平均BA最高
    2. 最低年份BA最高
    3. 平均AUC最高
    """

    return (
        df.sort_values(
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


def main():
    if not INPUT.exists():
        raise SystemExit(
            "找不到输入: "
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
        "sha256",
        "seed",
        "label",
        "probability",
        "base_probability",
        "gate",
        "semantic_delta",
    }

    missing = (
        required
        - set(
            df.columns
        )
    )

    if missing:
        raise SystemExit(
            "缺少列: "
            + ", ".join(
                sorted(
                    missing
                )
            )
        )

    df[
        "year"
    ] = pd.to_numeric(
        df[
            "year"
        ],
        errors="raise",
    ).astype(int)

    df[
        "seed"
    ] = pd.to_numeric(
        df[
            "seed"
        ],
        errors="raise",
    ).astype(int)

    df[
        "label"
    ] = pd.to_numeric(
        df[
            "label"
        ],
        errors="raise",
    ).astype(int)

    print(
        "总行数 =",
        len(df),
    )

    print(
        "唯一APK =",
        df[
            "sha256"
        ].nunique(),
    )

    print(
        "seed =",
        sorted(
            df[
                "seed"
            ].unique()
        ),
    )

    if len(df) != 1985 * 5:
        raise SystemExit(
            "预期9925行，实际="
            + str(
                len(df)
            )
        )

    if (
        df[
            "sha256"
        ].nunique()
        != 1985
    ):
        raise SystemExit(
            "未来APK不是1985"
        )

    # ========================================================
    # 首先验证我们对semantic_delta的理解：
    #
    # probability应该严格等于：
    # sigmoid(
    #   logit(base_probability)
    #   + gate * semantic_delta
    # )
    # ========================================================

    reconstructed = gamma_probability(
        df,
        1.0,
    )

    stored = df[
        "probability"
    ].to_numpy(dtype=float)

    diff = np.abs(
        reconstructed
        - stored
    )

    print()
    print(
        "======================================"
    )
    print(
        "γ=1 概率重构检查"
    )
    print(
        "======================================"
    )

    print(
        "max diff =",
        f"{diff.max():.12g}",
    )

    print(
        "mean diff =",
        f"{diff.mean():.12g}",
    )

    if diff.max() > 1e-5:
        raise SystemExit(
            "γ=1无法重构原概率，停止扫描"
        )

    # ========================================================
    # γ=1正式结果复现
    # ========================================================

    (
        baseline_summary,
        baseline_year,
        baseline_seed,
        _,
    ) = evaluate_gamma(
        df,
        1.0,
    )

    print()
    print(
        "======================================"
    )
    print(
        "γ=1 正式结果检查"
    )
    print(
        "======================================"
    )

    for year in YEARS:
        r = baseline_year[
            baseline_year[
                "year"
            ]
            == year
        ].iloc[0]

        print(
            f"{year} "
            f"BA={r['BA']:.6f} "
            f"AUC={r['AUC']:.6f} "
            f"FP={r['FP']:.2f} "
            f"FN={r['FN']:.2f}"
        )

        if (
            abs(
                r[
                    "BA"
                ]
                -
                EXPECTED[
                    year
                ][
                    "BA"
                ]
            )
            > 5e-4
        ):
            raise SystemExit(
                f"{year} γ=1 BA未复现"
            )

    print(
        "4年平均BA =",
        f"{baseline_summary['mean_BA']:.6f}",
    )

    print(
        "最低BA =",
        f"{baseline_summary['min_BA']:.6f}",
    )

    print(
        "平均AUC =",
        f"{baseline_summary['mean_AUC']:.6f}",
    )

    # ========================================================
    # 第一轮粗扫
    # ========================================================

    coarse_gammas = [
        round(
            x,
            2,
        )
        for x in np.arange(
            0.0,
            3.0001,
            0.25,
        )
    ]

    all_summary_rows = []
    all_year_rows = []
    all_seed_rows = []

    print()
    print(
        "======================================"
    )
    print(
        "第一轮粗扫"
    )
    print(
        "======================================"
    )

    for gamma in coarse_gammas:
        (
            summary,
            per_year,
            per_seed,
            _,
        ) = evaluate_gamma(
            df,
            gamma,
        )

        summary[
            "stage"
        ] = "coarse"

        all_summary_rows.append(
            summary
        )

        per_year[
            "stage"
        ] = "coarse"

        per_seed[
            "stage"
        ] = "coarse"

        all_year_rows.append(
            per_year
        )

        all_seed_rows.append(
            per_seed
        )

        print(
            f"γ={gamma:>4.2f} "
            f"meanBA={summary['mean_BA']:.6f} "
            f"minBA={summary['min_BA']:.6f} "
            f"AUC={summary['mean_AUC']:.6f} "
            f"BA18={summary['BA_2018']:.6f} "
            f"BA19={summary['BA_2019']:.6f} "
            f"BA20={summary['BA_2020']:.6f} "
            f"BA21={summary['BA_2021']:.6f}"
        )

    coarse_df = pd.DataFrame(
        all_summary_rows
    )

    ranked_coarse = rank_summary(
        coarse_df
    )

    coarse_best = float(
        ranked_coarse.iloc[
            0
        ][
            "gamma"
        ]
    )

    print()
    print(
        "粗扫最佳γ =",
        coarse_best,
    )

    # ========================================================
    # 第二轮细扫：
    # 最佳粗值 ±0.30，步长0.05
    # ========================================================

    fine_start = max(
        0.0,
        coarse_best - 0.30,
    )

    fine_end = (
        coarse_best
        + 0.30
    )

    fine_gammas = sorted(
        {
            round(
                float(x),
                2,
            )
            for x in np.arange(
                fine_start,
                fine_end + 0.0001,
                0.05,
            )
        }
        |
        {
            coarse_best,
            1.0,
        }
    )

    fine_summary_rows = []
    fine_year_rows = []
    fine_seed_rows = []

    print()
    print(
        "======================================"
    )
    print(
        "第二轮细扫"
    )
    print(
        "======================================"
    )

    for gamma in fine_gammas:
        (
            summary,
            per_year,
            per_seed,
            _,
        ) = evaluate_gamma(
            df,
            gamma,
        )

        summary[
            "stage"
        ] = "fine"

        fine_summary_rows.append(
            summary
        )

        per_year[
            "stage"
        ] = "fine"

        per_seed[
            "stage"
        ] = "fine"

        fine_year_rows.append(
            per_year
        )

        fine_seed_rows.append(
            per_seed
        )

        print(
            f"γ={gamma:>4.2f} "
            f"meanBA={summary['mean_BA']:.6f} "
            f"minBA={summary['min_BA']:.6f} "
            f"AUC={summary['mean_AUC']:.6f}"
        )

    fine_df = pd.DataFrame(
        fine_summary_rows
    )

    ranked_fine = rank_summary(
        fine_df
    )

    best = ranked_fine.iloc[
        0
    ]

    best_gamma = float(
        best[
            "gamma"
        ]
    )

    # ========================================================
    # 最佳γ相对γ=1的样本级诊断
    # ========================================================

    (
        correction_summary,
        correction_samples,
    ) = averaged_probability_diagnostic(
        df,
        best_gamma,
        reference_gamma=1.0,
    )

    # ========================================================
    # 保存
    # ========================================================

    coarse_df.to_csv(
        OUT
        / "coarse_summary.csv",
        index=False,
    )

    ranked_coarse.to_csv(
        OUT
        / "coarse_ranked.csv",
        index=False,
    )

    fine_df.to_csv(
        OUT
        / "fine_summary.csv",
        index=False,
    )

    ranked_fine.to_csv(
        OUT
        / "fine_ranked.csv",
        index=False,
    )

    pd.concat(
        all_year_rows
        + fine_year_rows,
        ignore_index=True,
    ).to_csv(
        OUT
        / "per_year.csv",
        index=False,
    )

    pd.concat(
        all_seed_rows
        + fine_seed_rows,
        ignore_index=True,
    ).to_csv(
        OUT
        / "per_seed.csv",
        index=False,
    )

    correction_summary.to_csv(
        OUT
        / "best_gamma_correction_summary.csv",
        index=False,
    )

    correction_samples.to_csv(
        OUT
        / "best_gamma_sample_diagnostic.csv",
        index=False,
    )

    selected = {
        "best_gamma":
            best_gamma,

        "selection_rule":
            (
                "maximize four-year mean BA; "
                "tie-break by minimum yearly BA; "
                "then mean AUC"
            ),

        "baseline_gamma":
            1.0,

        "baseline_mean_BA":
            float(
                baseline_summary[
                    "mean_BA"
                ]
            ),

        "best_mean_BA":
            float(
                best[
                    "mean_BA"
                ]
            ),

        "delta_mean_BA":
            float(
                best[
                    "mean_BA"
                ]
                -
                baseline_summary[
                    "mean_BA"
                ]
            ),

        "baseline_min_BA":
            float(
                baseline_summary[
                    "min_BA"
                ]
            ),

        "best_min_BA":
            float(
                best[
                    "min_BA"
                ]
            ),

        "baseline_mean_AUC":
            float(
                baseline_summary[
                    "mean_AUC"
                ]
            ),

        "best_mean_AUC":
            float(
                best[
                    "mean_AUC"
                ]
            ),
    }

    (
        OUT
        / "selected_gamma.json"
    ).write_text(
        json.dumps(
            selected,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    # ========================================================
    # 最终打印
    # ========================================================

    print()
    print(
        "======================================"
    )
    print(
        "最佳γ"
    )
    print(
        "======================================"
    )

    print(
        f"γ = {best_gamma:.2f}"
    )

    print(
        f"meanBA "
        f"{baseline_summary['mean_BA']:.6f}"
        f" -> "
        f"{best['mean_BA']:.6f} "
        f"(Δ={best['mean_BA'] - baseline_summary['mean_BA']:+.6f})"
    )

    print(
        f"minBA  "
        f"{baseline_summary['min_BA']:.6f}"
        f" -> "
        f"{best['min_BA']:.6f}"
    )

    print(
        f"meanAUC "
        f"{baseline_summary['mean_AUC']:.6f}"
        f" -> "
        f"{best['mean_AUC']:.6f}"
    )

    print()
    print(
        "逐年："
    )

    for year in YEARS:
        print(
            f"{year}: "
            f"BA={best[f'BA_{year}']:.6f} "
            f"AUC={best[f'AUC_{year}']:.6f} "
            f"FP={best[f'FP_{year}']:.2f} "
            f"FN={best[f'FN_{year}']:.2f}"
        )

    print()
    print(
        "======================================"
    )
    print(
        "相对γ=1的五seed平均概率诊断"
    )
    print(
        "======================================"
    )

    for _, row in correction_summary.iterrows():
        print(
            f"{str(row['year']):>4s} "
            f"N={int(row['N']):4d} "
            f"corrected={int(row['corrected']):3d} "
            f"regressed={int(row['regressed']):3d} "
            f"net={int(row['net']):+3d} "
            f"corrFP={int(row['corrected_FP']):3d} "
            f"corrFN={int(row['corrected_FN']):3d} "
            f"regFP={int(row['regressed_FP']):3d} "
            f"regFN={int(row['regressed_FN']):3d}"
        )

    print()
    print(
        "结果保存到:",
        OUT,
    )


if __name__ == "__main__":
    main()
