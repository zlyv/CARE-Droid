#!/usr/bin/env python3

from pathlib import Path

import numpy as np
import pandas as pd


OUT = Path(
    "artifacts/"
    "frozen-count10-150-layerwise-representation-diagnostic"
)

INPUT = (
    OUT
    / "sample_layerwise_average.csv"
)

A = "A_wrong_residual_direction"
D = "D_successfully_corrected"


def summarize(
    g,
    year,
    category,
):
    return {
        "year": year,
        "category": category,
        "N": len(g),

        "semantic_nearest_alignment":
            g[
                "semantic_nearest_truth_alignment"
            ].mean(),

        "semantic_nearest_correct_rate":
            g[
                "semantic_nearest_correct_fraction"
            ].mean(),

        "semantic_prototype_alignment":
            g[
                "semantic_prototype_truth_alignment"
            ].mean(),

        "semantic_prototype_correct_rate":
            g[
                "semantic_prototype_correct_fraction"
            ].mean(),

        "delta_nearest_alignment":
            g[
                "delta_nearest_truth_alignment"
            ].mean(),

        "delta_nearest_correct_rate":
            g[
                "delta_nearest_correct_fraction"
            ].mean(),

        "delta_prototype_alignment":
            g[
                "delta_prototype_truth_alignment"
            ].mean(),

        "delta_prototype_correct_rate":
            g[
                "delta_prototype_correct_fraction"
            ].mean(),

        "raw_direction_correct_fraction":
            g[
                "raw_direction_correct_fraction"
            ].mean(),

        "mean_margin":
            g[
                "base_margin_probability"
            ].mean(),

        "benign":
            int(
                (
                    g["label"] == 0
                ).sum()
            ),

        "malware":
            int(
                (
                    g["label"] == 1
                ).sum()
            ),
    }


def quadrant(
    rep_correct,
    raw_correct,
):
    if (
        not rep_correct
        and not raw_correct
    ):
        return (
            "representation_wrong__output_wrong"
        )

    if (
        rep_correct
        and not raw_correct
    ):
        return (
            "representation_correct__output_wrong"
        )

    if (
        not rep_correct
        and raw_correct
    ):
        return (
            "representation_wrong__output_correct"
        )

    return (
        "representation_correct__output_correct"
    )


def transition(
    semantic_correct,
    delta_correct,
):
    if (
        semantic_correct
        and delta_correct
    ):
        return (
            "semantic_correct__delta_correct"
        )

    if (
        semantic_correct
        and not delta_correct
    ):
        return (
            "semantic_correct__delta_wrong"
        )

    if (
        not semantic_correct
        and delta_correct
    ):
        return (
            "semantic_wrong__delta_correct"
        )

    return (
        "semantic_wrong__delta_wrong"
    )


def main():
    if not INPUT.exists():
        raise SystemExit(
            "找不到 "
            + str(INPUT)
        )

    avg = pd.read_csv(
        INPUT
    )

    required = {
        "year",
        "sha256",
        "label",
        "mechanism_category",

        "semantic_nearest_truth_alignment",
        "semantic_nearest_correct_fraction",
        "semantic_prototype_truth_alignment",
        "semantic_prototype_correct_fraction",

        "delta_nearest_truth_alignment",
        "delta_nearest_correct_fraction",
        "delta_prototype_truth_alignment",
        "delta_prototype_correct_fraction",

        "raw_direction_correct_fraction",
        "base_margin_probability",
    }

    missing = (
        required
        - set(avg.columns)
    )

    if missing:
        raise SystemExit(
            "缺少列: "
            + ", ".join(
                sorted(missing)
            )
        )

    ad = avg[
        avg[
            "mechanism_category"
        ].isin(
            (
                A,
                D,
            )
        )
    ].copy()

    print(
        "A+D =",
        len(ad),
    )

    print(
        "A =",
        int(
            (
                ad[
                    "mechanism_category"
                ] == A
            ).sum()
        ),
    )

    print(
        "D =",
        int(
            (
                ad[
                    "mechanism_category"
                ] == D
            ).sum()
        ),
    )

    # ========================================================
    # A/D逐年及总体
    # ========================================================

    rows = []

    for year in (
        2018,
        2019,
        2020,
        2021,
        "ALL",
    ):
        if year == "ALL":
            yg = ad
        else:
            yg = ad[
                ad["year"] == year
            ]

        for category in (
            A,
            D,
        ):
            g = yg[
                yg[
                    "mechanism_category"
                ] == category
            ]

            rows.append(
                summarize(
                    g,
                    year,
                    category,
                )
            )

    summary = pd.DataFrame(
        rows
    )

    summary.to_csv(
        OUT
        / "A_vs_D_layerwise_summary.csv",
        index=False,
    )

    # ========================================================
    # 联合表示 vs 最后残差输出
    # ========================================================

    ad[
        "delta_representation_correct"
    ] = (
        ad[
            "delta_nearest_truth_alignment"
        ]
        > 0
    )

    ad[
        "semantic_representation_correct"
    ] = (
        ad[
            "semantic_nearest_truth_alignment"
        ]
        > 0
    )

    ad[
        "raw_majority_correct"
    ] = (
        ad[
            "raw_direction_correct_fraction"
        ]
        > 0.5
    )

    ad[
        "delta_output_quadrant"
    ] = [
        quadrant(
            r,
            o,
        )
        for r, o in zip(
            ad[
                "delta_representation_correct"
            ],
            ad[
                "raw_majority_correct"
            ],
        )
    ]

    ad[
        "semantic_to_delta_transition"
    ] = [
        transition(
            s,
            d,
        )
        for s, d in zip(
            ad[
                "semantic_representation_correct"
            ],
            ad[
                "delta_representation_correct"
            ],
        )
    ]

    quadrant_rows = []

    for category in (
        A,
        D,
    ):
        g = ad[
            ad[
                "mechanism_category"
            ] == category
        ]

        counts = (
            g[
                "delta_output_quadrant"
            ]
            .value_counts()
        )

        for name in (
            "representation_wrong__output_wrong",
            "representation_correct__output_wrong",
            "representation_wrong__output_correct",
            "representation_correct__output_correct",
        ):
            n = int(
                counts.get(
                    name,
                    0,
                )
            )

            quadrant_rows.append({
                "category":
                    category,

                "quadrant":
                    name,

                "N":
                    n,

                "share":
                    (
                        n / len(g)
                        if len(g)
                        else np.nan
                    ),
            })

    quadrant_df = pd.DataFrame(
        quadrant_rows
    )

    quadrant_df.to_csv(
        OUT
        / "delta_representation_output_quadrants.csv",
        index=False,
    )

    transition_rows = []

    for category in (
        A,
        D,
    ):
        g = ad[
            ad[
                "mechanism_category"
            ] == category
        ]

        counts = (
            g[
                "semantic_to_delta_transition"
            ]
            .value_counts()
        )

        for name in (
            "semantic_correct__delta_correct",
            "semantic_correct__delta_wrong",
            "semantic_wrong__delta_correct",
            "semantic_wrong__delta_wrong",
        ):
            n = int(
                counts.get(
                    name,
                    0,
                )
            )

            transition_rows.append({
                "category":
                    category,

                "transition":
                    name,

                "N":
                    n,

                "share":
                    (
                        n / len(g)
                        if len(g)
                        else np.nan
                    ),
            })

    transition_df = pd.DataFrame(
        transition_rows
    )

    transition_df.to_csv(
        OUT
        / "semantic_to_delta_transitions.csv",
        index=False,
    )

    ad.to_csv(
        OUT
        / "A_D_sample_layerwise.csv",
        index=False,
    )

    # ========================================================
    # 打印
    # ========================================================

    readable = {
        A: "A方向错误",
        D: "D成功纠正",
    }

    print()
    print(
        "======================================"
    )
    print(
        "A vs D：分层内部表示"
    )
    print(
        "======================================"
    )

    all_rows = summary[
        summary[
            "year"
        ].astype(str) == "ALL"
    ]

    for category in (
        A,
        D,
    ):
        row = all_rows[
            all_rows[
                "category"
            ] == category
        ].iloc[0]

        print()
        print(
            readable[
                category
            ]
        )

        print(
            " 关系编码16维："
            f"最近类正确="
            f"{100*row['semantic_nearest_correct_rate']:.2f}% "
            f"类别中心正确="
            f"{100*row['semantic_prototype_correct_rate']:.2f}% "
            f"对齐="
            f"{row['semantic_nearest_alignment']:+.4f}"
        )

        print(
            " 联合残差16维："
            f"最近类正确="
            f"{100*row['delta_nearest_correct_rate']:.2f}% "
            f"类别中心正确="
            f"{100*row['delta_prototype_correct_rate']:.2f}% "
            f"对齐="
            f"{row['delta_nearest_alignment']:+.4f}"
        )

        print(
            " 最终残差方向："
            f"{100*row['raw_direction_correct_fraction']:.2f}%"
        )

    print()
    print(
        "======================================"
    )
    print(
        "A方向错误：联合表示 vs 最终输出"
    )
    print(
        "======================================"
    )

    for _, row in quadrant_df[
        quadrant_df[
            "category"
        ] == A
    ].iterrows():
        print(
            f"{row['quadrant']:42s} "
            f"N={int(row['N']):3d} "
            f"share={100*row['share']:.2f}%"
        )

    print()
    print(
        "======================================"
    )
    print(
        "A方向错误：关系编码层 -> 联合残差层"
    )
    print(
        "======================================"
    )

    for _, row in transition_df[
        transition_df[
            "category"
        ] == A
    ].iterrows():
        print(
            f"{row['transition']:38s} "
            f"N={int(row['N']):3d} "
            f"share={100*row['share']:.2f}%"
        )

    print()
    print(
        "======================================"
    )
    print(
        "逐年A/D联合残差表示"
    )
    print(
        "======================================"
    )

    for year in (
        2018,
        2019,
        2020,
        2021,
    ):
        print()
        print(year)

        yg = summary[
            summary[
                "year"
            ].astype(str)
            == str(year)
        ]

        for category in (
            A,
            D,
        ):
            row = yg[
                yg[
                    "category"
                ] == category
            ].iloc[0]

            print(
                f" {readable[category]:10s} "
                f"N={int(row['N']):3d} "
                f"关系层="
                f"{100*row['semantic_nearest_correct_rate']:.1f}% "
                f"联合层="
                f"{100*row['delta_nearest_correct_rate']:.1f}% "
                f"输出方向="
                f"{100*row['raw_direction_correct_fraction']:.1f}%"
            )


if __name__ == "__main__":
    main()
