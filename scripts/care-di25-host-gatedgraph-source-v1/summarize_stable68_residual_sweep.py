#!/usr/bin/env python3

from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.metrics import (
    balanced_accuracy_score,
    roc_auc_score,
)


ROOT = Path(
    "artifacts"
)


EXPERIMENTS = [
    (
        "Stable70",
        ROOT
        / "frozen-anchor-stable70-stable15-2017-2021",
    ),

    (
        "Stable68-reflective",
        ROOT
        / "frozen-anchor-stable68-reflective_invocation-stable15-2017-2021",
    ),

    (
        "Stable68-wakelock",
        ROOT
        / "frozen-anchor-stable68-process_or_wakelock_control-stable15-2017-2021",
    ),

    (
        "Stable68-crypto",
        ROOT
        / "frozen-anchor-stable68-cryptography-stable15-2017-2021",
    ),

    (
        "Stable68-database",
        ROOT
        / "frozen-anchor-stable68-database_access-stable15-2017-2021",
    ),
]


def calc_metrics(
    y,
    p,
):
    y = np.asarray(
        y,
        dtype=int,
    )

    p = np.asarray(
        p,
        dtype=float,
    )

    pred = (
        p >= 0.5
    ).astype(int)

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
                    p,
                )
            ),
    }


def main():
    rows = []

    for name, directory in EXPERIMENTS:

        path = (
            directory
            / "future_predictions.csv"
        )

        if not path.exists():
            print(
                "SKIP:",
                name,
                path,
            )
            continue

        df = pd.read_csv(
            path
        )

        for (
            year,
            seed
        ), g in df.groupby(
            [
                "year",
                "seed",
            ]
        ):
            base_m = calc_metrics(
                g[
                    "label"
                ],
                g[
                    "base_probability"
                ],
            )

            final_m = calc_metrics(
                g[
                    "label"
                ],
                g[
                    "probability"
                ],
            )

            rows.append({
                "model":
                    name,

                "year":
                    int(
                        year
                    ),

                "seed":
                    int(
                        seed
                    ),

                "base_BA":
                    base_m[
                        "BA"
                    ],

                "final_BA":
                    final_m[
                        "BA"
                    ],

                "gain_pp":
                    100.0
                    * (
                        final_m[
                            "BA"
                        ]
                        - base_m[
                            "BA"
                        ]
                    ),

                "base_AUC":
                    base_m[
                        "AUC"
                    ],

                "final_AUC":
                    final_m[
                        "AUC"
                    ],
            })

    result = pd.DataFrame(
        rows
    )

    if len(result) == 0:
        raise RuntimeError(
            "没有找到结果"
        )

    by_year = (
        result
        .groupby(
            [
                "model",
                "year",
            ],
            as_index=False,
        )
        .mean(
            numeric_only=True
        )
    )

    summary_rows = []

    for model, g in by_year.groupby(
        "model"
    ):
        summary_rows.append({
            "model":
                model,

            "base_mean_BA":
                float(
                    g[
                        "base_BA"
                    ].mean()
                ),

            "base_min_BA":
                float(
                    g[
                        "base_BA"
                    ].min()
                ),

            "final_mean_BA":
                float(
                    g[
                        "final_BA"
                    ].mean()
                ),

            "final_min_BA":
                float(
                    g[
                        "final_BA"
                    ].min()
                ),

            "residual_gain_pp":
                100.0
                * (
                    g[
                        "final_BA"
                    ].mean()
                    - g[
                        "base_BA"
                    ].mean()
                ),

            "years_improved":
                int(
                    (
                        g[
                            "final_BA"
                        ]
                        > g[
                            "base_BA"
                        ]
                    ).sum()
                ),

            "base_mean_AUC":
                float(
                    g[
                        "base_AUC"
                    ].mean()
                ),

            "final_mean_AUC":
                float(
                    g[
                        "final_AUC"
                    ].mean()
                ),
        })

    summary = (
        pd.DataFrame(
            summary_rows
        )
        .sort_values(
            [
                "final_min_BA",
                "final_mean_BA",
                "residual_gain_pp",
            ],
            ascending=False,
        )
    )

    print()
    print(
        "=" * 125
    )
    print(
        "逐年结果"
    )
    print(
        "=" * 125
    )

    print(
        by_year[
            [
                "model",
                "year",
                "base_BA",
                "final_BA",
                "gain_pp",
                "base_AUC",
                "final_AUC",
            ]
        ].to_string(
            index=False,
            float_format=lambda x: f"{x:.6f}",
        )
    )

    print()
    print(
        "=" * 125
    )
    print(
        "四年汇总"
    )
    print(
        "=" * 125
    )

    print(
        summary.to_string(
            index=False,
            float_format=lambda x: f"{x:.6f}",
        )
    )


if __name__ == "__main__":
    main()
