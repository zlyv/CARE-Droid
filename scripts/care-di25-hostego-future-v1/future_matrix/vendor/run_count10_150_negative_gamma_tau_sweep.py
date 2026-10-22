#!/usr/bin/env python3
from pathlib import Path

import numpy as np
import pandas as pd

import run_count10_150_directional_gamma_sweep_future as exp


OUT = Path(
    "artifacts/"
    "frozen-count10-150-negative-gamma-tau-sweep"
)

TAUS = np.round(
    np.arange(
        0.10,
        0.3001,
        0.025,
    ),
    3,
)

NEG_GAMMAS = np.round(
    np.arange(
        2.00,
        10.0001,
        0.25,
    ),
    2,
)

POS_GAMMA = 1.0


def main():
    OUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    df = pd.read_csv(
        exp.INPUT
    )

    print(
        "rows =",
        len(df),
        "APK =",
        df["sha256"].nunique(),
    )

    # ========================================================
    # 当前正式基线
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

    rows = []

    # ========================================================
    # 联合扫描
    # ========================================================

    for tau in TAUS:

        exp.TAU = float(
            tau
        )

        for gamma_neg in NEG_GAMMAS:

            result = exp.evaluate(
                df,
                float(gamma_neg),
                POS_GAMMA,
            )

            result[
                "tau"
            ] = float(tau)

            rows.append(
                result
            )

    result = pd.DataFrame(
        rows
    )

    ranked = (
        result.sort_values(
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

    # ========================================================
    # 保守候选：
    #
    # AUC最多下降0.002
    # minBA不能低于当前
    # ========================================================

    conservative = (
        result[
            (
                result[
                    "mean_AUC"
                ]
                >=
                baseline[
                    "mean_AUC"
                ]
                - 0.002
            )
            &
            (
                result[
                    "min_BA"
                ]
                >=
                baseline[
                    "min_BA"
                ]
            )
        ]
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

    result.to_csv(
        OUT / "all_results.csv",
        index=False,
    )

    ranked.to_csv(
        OUT / "ranked.csv",
        index=False,
    )

    conservative.to_csv(
        OUT / "conservative.csv",
        index=False,
    )

    best = ranked.iloc[
        0
    ]

    # ========================================================
    # 最佳点样本诊断
    # ========================================================

    exp.TAU = float(
        best[
            "tau"
        ]
    )

    diag = exp.sample_diagnostic(
        df,
        float(
            best[
                "gamma_neg"
            ]
        ),
        1.0,
    )

    diag.to_csv(
        OUT / "best_sample_diagnostic.csv",
        index=False,
    )

    # ========================================================
    # 每个tau自己的最佳g-
    # ========================================================

    tau_best_rows = []

    for tau, g in result.groupby(
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

        tau_best_rows.append(
            r.to_dict()
        )

    tau_best = pd.DataFrame(
        tau_best_rows
    )

    tau_best.to_csv(
        OUT / "best_per_tau.csv",
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
        "前20名"
    )
    print(
        "=" * 78
    )

    for _, r in ranked.head(
        20
    ).iterrows():

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

    for _, r in tau_best.iterrows():

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
        "绝对最佳"
    )
    print(
        "=" * 78
    )

    print(
        f"tau={best['tau']:.3f}"
    )

    print(
        f"g-={best['gamma_neg']:.2f}"
    )

    print(
        "g+=1.00"
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
        f"AUC "
        f"{baseline['mean_AUC']:.6f}"
        f" -> "
        f"{best['mean_AUC']:.6f}"
    )

    print()

    for year in exp.YEARS:
        print(
            f"{year}: "
            f"BA={best[f'BA_{year}']:.6f} "
            f"AUC={best[f'AUC_{year}']:.6f} "
            f"FP={best[f'FP_{year}']:.2f} "
            f"FN={best[f'FN_{year}']:.2f}"
        )

    print()
    print(
        "=" * 78
    )
    print(
        "保守约束最佳"
    )
    print(
        "=" * 78
    )

    if len(
        conservative
    ) == 0:

        print(
            "没有满足条件的候选"
        )

    else:

        c = conservative.iloc[
            0
        ]

        print(
            f"tau={c['tau']:.3f} "
            f"g-={c['gamma_neg']:.2f} "
            f"meanBA={c['mean_BA']:.6f} "
            f"minBA={c['min_BA']:.6f} "
            f"AUC={c['mean_AUC']:.6f}"
        )

    print()
    print(
        "=" * 78
    )
    print(
        "绝对最佳样本变化"
    )
    print(
        "=" * 78
    )

    for _, r in diag.iterrows():

        print(
            f"{str(r['year']):>4s} "
            f"corrected={int(r['corrected']):3d} "
            f"regressed={int(r['regressed']):3d} "
            f"net={int(r['net']):+3d} "
            f"corrB={int(r['corr_benign']):3d} "
            f"corrM={int(r['corr_malware']):3d} "
            f"regB={int(r['reg_benign']):3d} "
            f"regM={int(r['reg_malware']):3d}"
        )


if __name__ == "__main__":
    main()
