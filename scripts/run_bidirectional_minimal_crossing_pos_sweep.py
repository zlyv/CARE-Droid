#!/usr/bin/env python3

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(
    0,
    str(
        Path(__file__).resolve().parent
    ),
)

import test_minimal_negative_crossing as base


INPUT = base.INPUT

OUT = Path(
    "artifacts/"
    "bidirectional-minimal-crossing-pos-sweep"
)

G_NEG = 16.25
EPS_Z = 1e-6
EPS_R = 1e-12

# 1表示不增加新的正方向跨界。
G_POS_VALUES = np.round(
    np.arange(
        1.0,
        20.0001,
        0.25,
    ),
    2,
)


def build_probability(
    df,
    g_pos,
):
    p_base = (
        df[
            "base_probability"
        ].to_numpy(
            dtype=float
        )
    )

    z_base = base.logit(
        p_base
    )

    r = (
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

    # 当前正式残差模型。
    z_current = (
        z_base
        + r
    )

    z = z_current.copy()

    # ========================================================
    # 负方向最小跨界
    #
    # Compact在恶意侧；
    # 当前残差还没有成功跨界；
    # 局部残差方向为负；
    # 所需放大倍数不超过G_NEG。
    # ========================================================

    neg_required = np.full(
        len(df),
        np.inf,
        dtype=float,
    )

    neg_valid = (
        (z_base > 0)
        &
        (r < -EPS_R)
    )

    neg_required[
        neg_valid
    ] = (
        z_base[
            neg_valid
        ]
        /
        (
            -r[
                neg_valid
            ]
        )
    )

    neg_eligible = (
        neg_valid
        &
        (z_current >= 0)
        &
        (
            neg_required
            <= G_NEG
        )
    )

    z[
        neg_eligible
    ] = -EPS_Z

    # ========================================================
    # 正方向最小跨界
    #
    # Compact在良性侧；
    # 当前残差还没有成功跨界；
    # 局部残差方向为正；
    # 所需放大倍数不超过G_POS。
    # ========================================================

    pos_required = np.full(
        len(df),
        np.inf,
        dtype=float,
    )

    pos_valid = (
        (z_base < 0)
        &
        (r > EPS_R)
    )

    pos_required[
        pos_valid
    ] = (
        (
            -z_base[
                pos_valid
            ]
        )
        /
        r[
            pos_valid
        ]
    )

    pos_eligible = (
        pos_valid
        &
        (z_current < 0)
        &
        (
            pos_required
            <= float(
                g_pos
            )
        )
    )

    z[
        pos_eligible
    ] = EPS_Z

    return (
        base.sigmoid(
            z
        ),
        neg_eligible,
        pos_eligible,
        neg_required,
        pos_required,
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
        df[
            "sha256"
        ].nunique(),
    )

    rows = []
    all_year = []

    for g_pos in G_POS_VALUES:

        (
            probability,
            neg_eligible,
            pos_eligible,
            neg_required,
            pos_required,
        ) = build_probability(
            df,
            g_pos,
        )

        (
            per_seed,
            per_year,
        ) = base.evaluate(
            df,
            probability,
        )

        row = {
            "g_neg":
                G_NEG,

            "g_pos":
                float(
                    g_pos
                ),

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

            "neg_eligible_records":
                int(
                    neg_eligible.sum()
                ),

            "pos_eligible_records":
                int(
                    pos_eligible.sum()
                ),
        }

        for year in (
            2018,
            2019,
            2020,
            2021,
        ):
            y = per_year[
                per_year[
                    "year"
                ] == year
            ].iloc[0]

            row[
                f"BA_{year}"
            ] = float(
                y[
                    "BA"
                ]
            )

            row[
                f"AUC_{year}"
            ] = float(
                y[
                    "AUC"
                ]
            )

            row[
                f"FP_{year}"
            ] = float(
                y[
                    "FP"
                ]
            )

            row[
                f"FN_{year}"
            ] = float(
                y[
                    "FN"
                ]
            )

        rows.append(
            row
        )

        per_year = (
            per_year.copy()
        )

        per_year[
            "g_pos"
        ] = float(
            g_pos
        )

        all_year.append(
            per_year
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

    result.to_csv(
        OUT
        / "all_results.csv",
        index=False,
    )

    ranked.to_csv(
        OUT
        / "ranked.csv",
        index=False,
    )

    pd.concat(
        all_year,
        ignore_index=True,
    ).to_csv(
        OUT
        / "per_year.csv",
        index=False,
    )

    baseline = result[
        np.isclose(
            result[
                "g_pos"
            ],
            1.0,
        )
    ].iloc[0]

    best = ranked.iloc[
        0
    ]

    print()
    print(
        "=" * 80
    )
    print(
        "当前单向最小跨界"
    )
    print(
        "=" * 80
    )

    print(
        f"G-={G_NEG:.2f} "
        f"G+=1.00 "
        f"meanBA={baseline['mean_BA']:.6f} "
        f"minBA={baseline['min_BA']:.6f} "
        f"AUC={baseline['mean_AUC']:.6f}"
    )

    print()
    print(
        "=" * 80
    )
    print(
        "正方向上限扫描前15"
    )
    print(
        "=" * 80
    )

    for _, r in (
        ranked.head(
            15
        ).iterrows()
    ):
        print(
            f"G+={r['g_pos']:.2f} "
            f"meanBA={r['mean_BA']:.6f} "
            f"minBA={r['min_BA']:.6f} "
            f"AUC={r['mean_AUC']:.6f} "
            f"posN={int(r['pos_eligible_records']):4d} "
            f"BA18={r['BA_2018']:.6f} "
            f"BA19={r['BA_2019']:.6f} "
            f"BA20={r['BA_2020']:.6f} "
            f"BA21={r['BA_2021']:.6f}"
        )

    print()
    print(
        "=" * 80
    )
    print(
        "双向最小跨界最佳"
    )
    print(
        "=" * 80
    )

    print(
        f"G-={G_NEG:.2f}"
    )

    print(
        f"G+={best['g_pos']:.2f}"
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

    print(
        f"正方向跨界记录数 = "
        f"{int(best['pos_eligible_records'])}"
    )

    print()

    for year in (
        2018,
        2019,
        2020,
        2021,
    ):
        print(
            f"{year}: "
            f"BA={best[f'BA_{year}']:.6f} "
            f"AUC={best[f'AUC_{year}']:.6f} "
            f"FP={best[f'FP_{year}']:.2f} "
            f"FN={best[f'FN_{year}']:.2f}"
        )


if __name__ == "__main__":
    main()
