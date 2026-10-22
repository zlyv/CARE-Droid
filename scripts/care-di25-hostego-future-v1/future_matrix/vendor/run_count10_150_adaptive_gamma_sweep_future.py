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
    "frozen-count10-150-adaptive-gamma-sweep"
)

THRESHOLD = 0.5
EPS = 1e-7

YEARS = (
    2018,
    2019,
    2020,
    2021,
)

# k=0就是当前正式模型。
K_VALUES = np.round(
    np.arange(
        0.0,
        5.0001,
        0.10,
    ),
    2,
)


def safe_logit(p):
    p = np.clip(
        np.asarray(
            p,
            dtype=float,
        ),
        EPS,
        1.0 - EPS,
    )

    return np.log(
        p
        /
        (
            1.0
            - p
        )
    )


def sigmoid(z):
    z = np.clip(
        np.asarray(
            z,
            dtype=float,
        ),
        -60.0,
        60.0,
    )

    return (
        1.0
        /
        (
            1.0
            + np.exp(
                -z
            )
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

    pred = (
        probability
        >= THRESHOLD
    ).astype(int)

    fp = int(
        np.sum(
            (y == 0)
            &
            (pred == 1)
        )
    )

    fn = int(
        np.sum(
            (y == 1)
            &
            (pred == 0)
        )
    )

    return {
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
                    probability,
                )
            ),

        "ACC":
            float(
                accuracy_score(
                    y,
                    pred,
                )
            ),

        "F1":
            float(
                f1_score(
                    y,
                    pred,
                )
            ),

        "FP":
            fp,

        "FN":
            fn,
    }


def adaptive_probability(
    df,
    k,
):
    p_base = (
        df[
            "base_probability"
        ].to_numpy(
            dtype=float
        )
    )

    z_base = safe_logit(
        p_base
    )

    # ========================================================
    # Compact自己的不确定度。
    #
    # 不直接使用gate，
    # 因为gate还含presence。
    # ========================================================

    uncertainty = (
        4.0
        * p_base
        * (
            1.0
            - p_base
        )
    )

    adaptive_gamma = (
        1.0
        + float(k)
        * uncertainty
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

    z = (
        z_base
        + adaptive_gamma
        * residual
    )

    return (
        sigmoid(
            z
        ),
        adaptive_gamma,
    )


def evaluate(
    df,
    k,
):
    work = df.copy()

    (
        probability,
        gamma,
    ) = adaptive_probability(
        work,
        k,
    )

    work[
        "adaptive_probability"
    ] = probability

    work[
        "adaptive_gamma"
    ] = gamma

    seed_rows = []

    # ========================================================
    # 正式协议：
    # 每个seed单独计算指标。
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
            ].to_numpy(
                dtype=int
            ),
            g[
                "adaptive_probability"
            ].to_numpy(
                dtype=float
            ),
        )

        seed_rows.append({
            "k":
                float(k),

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
            "k":
                float(k),

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

    result = {
        "k":
            float(k),

        "max_gamma":
            float(
                1.0
                + k
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

        "mean_ACC":
            float(
                year_df[
                    "ACC"
                ].mean()
            ),
    }

    for year in YEARS:
        row = year_df[
            year_df[
                "year"
            ]
            == year
        ].iloc[0]

        for metric in (
            "BA",
            "AUC",
            "FP",
            "FN",
        ):
            result[
                f"{metric}_{year}"
            ] = float(
                row[
                    metric
                ]
            )

    return (
        result,
        year_df,
        seed_df,
        work,
    )


def sample_diagnostic(
    df,
    best_k,
):
    candidate = df.copy()

    (
        cp,
        cg,
    ) = adaptive_probability(
        candidate,
        best_k,
    )

    candidate[
        "candidate_probability"
    ] = cp

    candidate[
        "adaptive_gamma"
    ] = cg

    # k=0就是正式当前模型。
    (
        bp,
        _,
    ) = adaptive_probability(
        candidate,
        0.0,
    )

    candidate[
        "baseline_probability"
    ] = bp

    avg = (
        candidate.groupby(
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

            mean_adaptive_gamma=(
                "adaptive_gamma",
                "mean",
            ),
        )
    )

    avg[
        "candidate_pred"
    ] = (
        avg[
            "candidate_probability"
        ]
        >= THRESHOLD
    ).astype(int)

    avg[
        "baseline_pred"
    ] = (
        avg[
            "baseline_probability"
        ]
        >= THRESHOLD
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
        + [
            "ALL"
        ]
    ):
        if year_value == "ALL":
            g = avg
        else:
            g = avg[
                avg[
                    "year"
                ]
                == year_value
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

            "corrected_benign":
                int(
                    (
                        g[
                            "corrected"
                        ]
                        &
                        (
                            g[
                                "label"
                            ]
                            == 0
                        )
                    ).sum()
                ),

            "corrected_malware":
                int(
                    (
                        g[
                            "corrected"
                        ]
                        &
                        (
                            g[
                                "label"
                            ]
                            == 1
                        )
                    ).sum()
                ),

            "regressed_benign":
                int(
                    (
                        g[
                            "regressed"
                        ]
                        &
                        (
                            g[
                                "label"
                            ]
                            == 0
                        )
                    ).sum()
                ),

            "regressed_malware":
                int(
                    (
                        g[
                            "regressed"
                        ]
                        &
                        (
                            g[
                                "label"
                            ]
                            == 1
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
            "找不到输入"
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

    summary_rows = []
    year_frames = []
    seed_frames = []

    print()
    print(
        "======================================"
    )
    print(
        "不确定度自适应残差放大"
    )
    print(
        "======================================"
    )

    for k in K_VALUES:
        (
            result,
            year_df,
            seed_df,
            _,
        ) = evaluate(
            df,
            k,
        )

        summary_rows.append(
            result
        )

        year_frames.append(
            year_df
        )

        seed_frames.append(
            seed_df
        )

        print(
            f"k={k:>4.2f} "
            f"maxγ={1+k:>4.2f} "
            f"meanBA={result['mean_BA']:.6f} "
            f"minBA={result['min_BA']:.6f} "
            f"AUC={result['mean_AUC']:.6f}"
        )

    summary = pd.DataFrame(
        summary_rows
    )

    # 仍然优先四年平均BA。
    ranked = (
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

    best = ranked.iloc[
        0
    ]

    baseline = summary[
        summary[
            "k"
        ]
        == 0.0
    ].iloc[0]

    best_k = float(
        best[
            "k"
        ]
    )

    (
        diag_summary,
        diag_samples,
    ) = sample_diagnostic(
        df,
        best_k,
    )

    summary.to_csv(
        OUT
        / "summary.csv",
        index=False,
    )

    ranked.to_csv(
        OUT
        / "ranked.csv",
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

    diag_summary.to_csv(
        OUT
        / "best_k_correction_summary.csv",
        index=False,
    )

    diag_samples.to_csv(
        OUT
        / "best_k_sample_diagnostic.csv",
        index=False,
    )

    print()
    print(
        "======================================"
    )
    print(
        "最佳结果"
    )
    print(
        "======================================"
    )

    print(
        f"k = {best_k:.2f}"
    )

    print(
        f"最大γ = {1+best_k:.2f}"
    )

    print(
        f"meanBA "
        f"{baseline['mean_BA']:.6f}"
        f" -> "
        f"{best['mean_BA']:.6f} "
        f"Δ="
        f"{best['mean_BA']-baseline['mean_BA']:+.6f}"
    )

    print(
        f"minBA "
        f"{baseline['min_BA']:.6f}"
        f" -> "
        f"{best['min_BA']:.6f}"
    )

    print(
        f"meanAUC "
        f"{baseline['mean_AUC']:.6f}"
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
        "相对当前模型的样本级变化"
    )
    print(
        "======================================"
    )

    for _, row in (
        diag_summary.iterrows()
    ):
        print(
            f"{str(row['year']):>4s} "
            f"corrected="
            f"{int(row['corrected']):3d} "
            f"regressed="
            f"{int(row['regressed']):3d} "
            f"net="
            f"{int(row['net']):+3d} "
            f"corrB="
            f"{int(row['corrected_benign']):3d} "
            f"corrM="
            f"{int(row['corrected_malware']):3d}"
        )


if __name__ == "__main__":
    main()
