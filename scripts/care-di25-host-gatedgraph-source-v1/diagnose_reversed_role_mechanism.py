#!/usr/bin/env python3

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


SCRIPT_DIR = Path(__file__).resolve().parent

if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import run_frozen_anchor_residual_2017_2021 as base


FEATURES = Path(
    "artifacts/p1-features/features.csv"
)

SPLITS = Path(
    "artifacts/"
    "p1-splits-fixed-2017-test-2018-2021/"
    "split_manifest.csv"
)

OUT = Path(
    "artifacts/"
    "reversed-role-mechanism"
)

YEARS = (
    2017,
    2018,
    2019,
    2020,
    2021,
)

ROLES = (
    "command_execution",
    "task_or_alarm_scheduling",
    "icon_or_component_hiding",
    "device_identifier_access",
)


def norm_sha(x):
    return str(x).strip().casefold()


def safe_float(x):
    try:
        v = float(x)

        if np.isfinite(v):
            return v

    except Exception:
        pass

    return 0.0


def safe_auc(y, x):
    y = np.asarray(
        y,
        dtype=int,
    )

    x = np.asarray(
        x,
        dtype=float,
    )

    if (
        len(y) == 0
        or len(np.unique(y)) < 2
    ):
        return np.nan

    if np.allclose(
        x,
        x[0],
    ):
        return 0.5

    return float(
        roc_auc_score(
            y,
            x,
        )
    )


def get_year_rows(
    future_parts,
    year,
):
    if year == 2017:
        name = (
            "train_fixed_2017__test_2018"
        )

        return list(
            future_parts[
                name
            ][
                "train"
            ]
        )

    name = (
        f"train_fixed_2017__test_{year}"
    )

    return list(
        future_parts[
            name
        ][
            "test"
        ]
    )


def values_for(
    rows,
    feature_by_sha,
    column,
):
    out = []

    for row in rows:
        sha = norm_sha(
            row[
                "sha256"
            ]
        )

        source = feature_by_sha[
            sha
        ]

        out.append(
            safe_float(
                source.get(
                    column,
                    0.0,
                )
            )
        )

    return np.asarray(
        out,
        dtype=float,
    )


def oriented_auc(
    auc,
    source_auc,
):
    if not np.isfinite(auc):
        return np.nan

    if source_auc >= 0.5:
        return float(auc)

    return float(
        1.0 - auc
    )


def main():
    OUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    feature_by_sha = (
        base.normalized_rows(
            base.read_csv_rows(
                FEATURES
            )
        )
    )

    future_parts = (
        base.read_split_rows(
            SPLITS
        )
    )

    rows_by_year = {
        year:
            get_year_rows(
                future_parts,
                year,
            )
        for year in YEARS
    }

    labels_by_year = {
        year:
            base.labels_for(
                rows_by_year[
                    year
                ],
                feature_by_sha,
            ).astype(int)
        for year in YEARS
    }

    detail_rows = []

    # ========================================================
    # 每个角色分别分析 node / edge
    # ========================================================

    for role in ROLES:
        for kind in (
            "node",
            "edge",
        ):
            column = (
                f"role_{kind}__{role}"
            )

            values_by_year = {
                year:
                    values_for(
                        rows_by_year[
                            year
                        ],
                        feature_by_sha,
                        column,
                    )
                for year in YEARS
            }

            # ------------------------------------------------
            # 2017三个基准方向：
            #
            # 1. 原始计数
            # 2. 是否出现
            # 3. 出现以后，数量大小
            # ------------------------------------------------

            source_values = (
                values_by_year[
                    2017
                ]
            )

            source_labels = (
                labels_by_year[
                    2017
                ]
            )

            source_raw_auc = (
                safe_auc(
                    source_labels,
                    source_values,
                )
            )

            source_presence = (
                source_values > 0
            ).astype(float)

            source_presence_auc = (
                safe_auc(
                    source_labels,
                    source_presence,
                )
            )

            source_positive_mask = (
                source_values > 0
            )

            source_positive_auc = (
                safe_auc(
                    source_labels[
                        source_positive_mask
                    ],
                    source_values[
                        source_positive_mask
                    ],
                )
            )

            for year in YEARS:
                values = (
                    values_by_year[
                        year
                    ]
                )

                labels = (
                    labels_by_year[
                        year
                    ]
                )

                benign = (
                    labels == 0
                )

                malware = (
                    labels == 1
                )

                presence = (
                    values > 0
                ).astype(float)

                raw_auc = safe_auc(
                    labels,
                    values,
                )

                presence_auc = (
                    safe_auc(
                        labels,
                        presence,
                    )
                )

                positive_mask = (
                    values > 0
                )

                positive_auc = (
                    safe_auc(
                        labels[
                            positive_mask
                        ],
                        values[
                            positive_mask
                        ],
                    )
                )

                benign_positive = (
                    benign
                    &
                    (
                        values > 0
                    )
                )

                malware_positive = (
                    malware
                    &
                    (
                        values > 0
                    )
                )

                benign_presence = float(
                    np.mean(
                        values[
                            benign
                        ]
                        > 0
                    )
                )

                malware_presence = float(
                    np.mean(
                        values[
                            malware
                        ]
                        > 0
                    )
                )

                benign_positive_mean = (
                    float(
                        np.mean(
                            values[
                                benign_positive
                            ]
                        )
                    )
                    if benign_positive.any()
                    else np.nan
                )

                malware_positive_mean = (
                    float(
                        np.mean(
                            values[
                                malware_positive
                            ]
                        )
                    )
                    if malware_positive.any()
                    else np.nan
                )

                benign_positive_median = (
                    float(
                        np.median(
                            values[
                                benign_positive
                            ]
                        )
                    )
                    if benign_positive.any()
                    else np.nan
                )

                malware_positive_median = (
                    float(
                        np.median(
                            values[
                                malware_positive
                            ]
                        )
                    )
                    if malware_positive.any()
                    else np.nan
                )

                detail_rows.append({
                    "role":
                        role,

                    "kind":
                        kind,

                    "year":
                        year,

                    "N":
                        len(
                            labels
                        ),

                    "benign_n":
                        int(
                            benign.sum()
                        ),

                    "malware_n":
                        int(
                            malware.sum()
                        ),

                    # -------------------------
                    # 原始数量
                    # -------------------------

                    "raw_auc":
                        raw_auc,

                    "raw_oriented_auc":
                        oriented_auc(
                            raw_auc,
                            source_raw_auc,
                        ),

                    "raw_flip":
                        int(
                            year != 2017
                            and
                            oriented_auc(
                                raw_auc,
                                source_raw_auc,
                            ) < 0.5
                        ),

                    # -------------------------
                    # 是否出现
                    # -------------------------

                    "benign_presence":
                        benign_presence,

                    "malware_presence":
                        malware_presence,

                    "presence_gap_malware_minus_benign":
                        (
                            malware_presence
                            - benign_presence
                        ),

                    "presence_auc":
                        presence_auc,

                    "presence_oriented_auc":
                        oriented_auc(
                            presence_auc,
                            source_presence_auc,
                        ),

                    "presence_flip":
                        int(
                            year != 2017
                            and
                            oriented_auc(
                                presence_auc,
                                source_presence_auc,
                            ) < 0.5
                        ),

                    # -------------------------
                    # 出现后的数量
                    # -------------------------

                    "positive_benign_n":
                        int(
                            benign_positive.sum()
                        ),

                    "positive_malware_n":
                        int(
                            malware_positive.sum()
                        ),

                    "benign_positive_mean":
                        benign_positive_mean,

                    "malware_positive_mean":
                        malware_positive_mean,

                    "benign_positive_median":
                        benign_positive_median,

                    "malware_positive_median":
                        malware_positive_median,

                    "positive_auc":
                        positive_auc,

                    "positive_oriented_auc":
                        (
                            oriented_auc(
                                positive_auc,
                                source_positive_auc,
                            )
                            if np.isfinite(
                                source_positive_auc
                            )
                            else np.nan
                        ),

                    "positive_flip":
                        (
                            int(
                                year != 2017
                                and
                                np.isfinite(
                                    source_positive_auc
                                )
                                and
                                np.isfinite(
                                    positive_auc
                                )
                                and
                                oriented_auc(
                                    positive_auc,
                                    source_positive_auc,
                                ) < 0.5
                            )
                        ),
                })

    detail_df = pd.DataFrame(
        detail_rows
    )

    detail_df.to_csv(
        OUT
        / "per_role_year.csv",
        index=False,
    )

    # ========================================================
    # 一行一个 role × kind
    # ========================================================

    summary_rows = []

    for (
        role,
        kind,
    ), g in detail_df.groupby(
        [
            "role",
            "kind",
        ],
        sort=True,
    ):
        g = (
            g
            .set_index(
                "year"
            )
            .sort_index()
        )

        future = g.loc[
            [
                2018,
                2019,
                2020,
                2021,
            ]
        ]

        summary_rows.append({
            "role":
                role,

            "kind":
                kind,

            "raw_auc_2017":
                float(
                    g.loc[
                        2017,
                        "raw_auc"
                    ]
                ),

            "raw_future_oriented_mean":
                float(
                    future[
                        "raw_oriented_auc"
                    ].mean()
                ),

            "raw_flip_count":
                int(
                    future[
                        "raw_flip"
                    ].sum()
                ),

            "presence_auc_2017":
                float(
                    g.loc[
                        2017,
                        "presence_auc"
                    ]
                ),

            "presence_future_oriented_mean":
                float(
                    future[
                        "presence_oriented_auc"
                    ].mean()
                ),

            "presence_flip_count":
                int(
                    future[
                        "presence_flip"
                    ].sum()
                ),

            "positive_auc_2017":
                float(
                    g.loc[
                        2017,
                        "positive_auc"
                    ]
                )
                if np.isfinite(
                    g.loc[
                        2017,
                        "positive_auc"
                    ]
                )
                else np.nan,

            "positive_future_oriented_mean":
                float(
                    future[
                        "positive_oriented_auc"
                    ].mean()
                ),

            "positive_flip_count":
                int(
                    future[
                        "positive_flip"
                    ].sum()
                ),

            "benign_presence_2017":
                float(
                    g.loc[
                        2017,
                        "benign_presence"
                    ]
                ),

            "malware_presence_2017":
                float(
                    g.loc[
                        2017,
                        "malware_presence"
                    ]
                ),

            "benign_presence_2021":
                float(
                    g.loc[
                        2021,
                        "benign_presence"
                    ]
                ),

            "malware_presence_2021":
                float(
                    g.loc[
                        2021,
                        "malware_presence"
                    ]
                ),

            "presence_gap_2017":
                float(
                    g.loc[
                        2017,
                        "presence_gap_malware_minus_benign"
                    ]
                ),

            "presence_gap_2021":
                float(
                    g.loc[
                        2021,
                        "presence_gap_malware_minus_benign"
                    ]
                ),
        })

    summary_df = pd.DataFrame(
        summary_rows
    )

    summary_df.to_csv(
        OUT
        / "mechanism_summary.csv",
        index=False,
    )

    show_cols = [
        "role",
        "kind",
        "raw_auc_2017",
        "raw_future_oriented_mean",
        "raw_flip_count",
        "presence_future_oriented_mean",
        "presence_flip_count",
        "positive_future_oriented_mean",
        "positive_flip_count",
        "benign_presence_2017",
        "malware_presence_2017",
        "benign_presence_2021",
        "malware_presence_2021",
        "presence_gap_2017",
        "presence_gap_2021",
    ]

    print()
    print(
        "=" * 130
    )
    print(
        "关系反转机制汇总"
    )
    print(
        "=" * 130
    )

    print(
        summary_df[
            show_cols
        ].to_string(
            index=False
        )
    )

    print()
    print(
        "=" * 130
    )
    print(
        "逐年份出现率"
    )
    print(
        "=" * 130
    )

    print(
        detail_df[
            [
                "role",
                "kind",
                "year",
                "benign_presence",
                "malware_presence",
                "presence_gap_malware_minus_benign",
                "raw_oriented_auc",
                "presence_oriented_auc",
                "positive_oriented_auc",
            ]
        ].to_string(
            index=False
        )
    )

    print()
    print(
        "结果目录:",
        OUT,
    )


if __name__ == "__main__":
    main()
