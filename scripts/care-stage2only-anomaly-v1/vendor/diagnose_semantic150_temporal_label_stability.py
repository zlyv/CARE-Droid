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


SEMANTIC_2017 = Path(
    "artifacts/"
    "clean-local-context-2017/"
    "semantic_relation_features.csv.gz"
)

SEMANTIC_FUTURE = Path(
    "artifacts/"
    "clean-local-context-future/"
    "semantic_relation_features.csv.gz"
)

FUTURE_SPLITS = Path(
    "artifacts/"
    "p1-splits-fixed-2017-test-2018-2021/"
    "split_manifest.csv"
)

COMPACT_FEATURES = Path(
    "artifacts/"
    "p1-features/"
    "features.csv"
)

OUT = Path(
    "artifacts/"
    "semantic150-temporal-label-stability"
)


YEARS = (
    2017,
    2018,
    2019,
    2020,
    2021,
)


def norm_sha(value):
    return (
        str(value)
        .strip()
        .casefold()
    )


def safe_float(value):
    try:
        value = float(value)

        if np.isfinite(value):
            return value

    except Exception:
        pass

    return 0.0


def safe_auc(
    labels,
    values,
):
    labels = np.asarray(
        labels,
        dtype=int,
    )

    values = np.asarray(
        values,
        dtype=float,
    )

    if len(labels) == 0:
        return np.nan

    if len(np.unique(labels)) < 2:
        return np.nan

    if np.allclose(
        values,
        values[0],
    ):
        return 0.5

    return float(
        roc_auc_score(
            labels,
            values,
        )
    )


def oriented_auc(
    auc,
    source_auc,
):
    if (
        not np.isfinite(auc)
        or not np.isfinite(source_auc)
    ):
        return np.nan

    if source_auc >= 0.5:
        return float(auc)

    return float(
        1.0 - auc
    )


def get_year_rows(
    future_parts,
    year,
):
    if year == 2017:
        split_name = (
            "train_fixed_2017__test_2018"
        )

        if split_name not in future_parts:
            raise SystemExit(
                "找不到固定2017训练人口"
            )

        return list(
            future_parts[
                split_name
            ][
                "train"
            ]
        )

    split_name = (
        f"train_fixed_2017__test_{year}"
    )

    if split_name not in future_parts:
        raise SystemExit(
            "找不到划分: "
            + split_name
        )

    return list(
        future_parts[
            split_name
        ][
            "test"
        ]
    )


def labels_for_rows(
    rows,
    compact_by_sha,
):
    return base.labels_for(
        rows,
        compact_by_sha,
    ).astype(int)


def values_for(
    rows,
    semantic_by_sha,
    name,
):
    values = []

    for row in rows:
        sha = norm_sha(
            row[
                "sha256"
            ]
        )

        if sha not in semantic_by_sha:
            raise RuntimeError(
                "语义关系缺失SHA: "
                + sha
            )

        values.append(
            safe_float(
                semantic_by_sha[
                    sha
                ].get(
                    name,
                    0.0,
                )
            )
        )

    return np.asarray(
        values,
        dtype=float,
    )


def parse_relation_name(
    name,
):
    """
    这里只用于输出更容易读的 anchor/context 名称。

    如果名称格式与预期不同，
    不影响正式统计。
    """

    text = str(name)

    # 尽量处理可能的命名形式。
    for prefix in (
        "support__",
        "relation__",
        "semantic_support__",
    ):
        if text.startswith(prefix):
            text = text[
                len(prefix):
            ]

    separators = (
        "__to__",
        "__ctx__",
        "__context__",
        "->",
    )

    for sep in separators:
        if sep in text:
            left, right = text.split(
                sep,
                1,
            )

            return (
                left,
                right,
            )

    return (
        "",
        text,
    )


def main():
    OUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ========================================================
    # 读取人口和标签
    # ========================================================

    compact_by_sha = (
        base.normalized_rows(
            base.read_csv_rows(
                COMPACT_FEATURES
            )
        )
    )

    semantic_2017 = (
        base.read_gz_csv(
            SEMANTIC_2017
        )
    )

    semantic_future = (
        base.read_gz_csv(
            SEMANTIC_FUTURE
        )
    )

    names150 = (
        base.support150_names(
            semantic_2017[
                0
            ].keys()
        )
    )

    future_names150 = (
        base.support150_names(
            semantic_future[
                0
            ].keys()
        )
    )

    if list(
        future_names150
    ) != list(
        names150
    ):
        raise SystemExit(
            "2017和未来150维定义不一致"
        )

    if len(
        names150
    ) != 150:
        raise SystemExit(
            "预期150维，实际="
            + str(
                len(
                    names150
                )
            )
        )

    semantic_by_sha = (
        base.normalized_rows(
            semantic_2017
            + semantic_future
        )
    )

    future_parts = (
        base.read_split_rows(
            FUTURE_SPLITS
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
            labels_for_rows(
                rows_by_year[
                    year
                ],
                compact_by_sha,
            )
        for year in YEARS
    }

    print(
        "150维关系数 =",
        len(
            names150
        ),
    )

    for year in YEARS:
        y = labels_by_year[
            year
        ]

        print(
            year,
            "N =",
            len(y),
            "benign =",
            int(
                (y == 0).sum()
            ),
            "malware =",
            int(
                (y == 1).sum()
            ),
        )

    # ========================================================
    # 每条关系 × 每个年份
    # ========================================================

    detail_rows = []

    for index, name in enumerate(
        names150
    ):
        anchor, context = (
            parse_relation_name(
                name
            )
        )

        values_by_year = {
            year:
                values_for(
                    rows_by_year[
                        year
                    ],
                    semantic_by_sha,
                    name,
                )
            for year in YEARS
        }

        # ----------------------------------------------------
        # 2017基准方向
        # ----------------------------------------------------

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

        source_raw_auc = safe_auc(
            source_labels,
            source_values,
        )

        source_presence = (
            source_values > 0.0
        ).astype(float)

        source_presence_auc = (
            safe_auc(
                source_labels,
                source_presence,
            )
        )

        source_positive_mask = (
            source_values > 0.0
        )

        if (
            source_positive_mask.sum() > 0
            and len(
                np.unique(
                    source_labels[
                        source_positive_mask
                    ]
                )
            ) == 2
        ):
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

        else:
            source_positive_auc = np.nan

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
                values > 0.0
            )

            raw_auc = safe_auc(
                labels,
                values,
            )

            presence_auc = safe_auc(
                labels,
                presence.astype(float),
            )

            positive_mask = presence

            if (
                positive_mask.sum() > 0
                and len(
                    np.unique(
                        labels[
                            positive_mask
                        ]
                    )
                ) == 2
            ):
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
            else:
                positive_auc = np.nan

            benign_values = (
                values[
                    benign
                ]
            )

            malware_values = (
                values[
                    malware
                ]
            )

            benign_positive = (
                benign
                &
                positive_mask
            )

            malware_positive = (
                malware
                &
                positive_mask
            )

            benign_presence = float(
                np.mean(
                    presence[
                        benign
                    ]
                )
            )

            malware_presence = float(
                np.mean(
                    presence[
                        malware
                    ]
                )
            )

            benign_mean = float(
                np.mean(
                    benign_values
                )
            )

            malware_mean = float(
                np.mean(
                    malware_values
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

            raw_oriented = oriented_auc(
                raw_auc,
                source_raw_auc,
            )

            presence_oriented = (
                oriented_auc(
                    presence_auc,
                    source_presence_auc,
                )
            )

            positive_oriented = (
                oriented_auc(
                    positive_auc,
                    source_positive_auc,
                )
                if np.isfinite(
                    source_positive_auc
                )
                else np.nan
            )

            detail_rows.append({
                "relation_index":
                    index,

                "feature":
                    name,

                "anchor":
                    anchor,

                "context":
                    context,

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
                # 原始支持值
                # -------------------------

                "raw_auc":
                    raw_auc,

                "raw_disc_auc":
                    max(
                        raw_auc,
                        1.0 - raw_auc,
                    ),

                "raw_oriented_auc":
                    raw_oriented,

                "raw_flip":
                    int(
                        year != 2017
                        and raw_oriented < 0.5
                    ),

                "benign_mean":
                    benign_mean,

                "malware_mean":
                    malware_mean,

                "mean_gap_malware_minus_benign":
                    (
                        malware_mean
                        - benign_mean
                    ),

                # -------------------------
                # 是否出现
                # -------------------------

                "active_n":
                    int(
                        positive_mask.sum()
                    ),

                "benign_active_n":
                    int(
                        benign_positive.sum()
                    ),

                "malware_active_n":
                    int(
                        malware_positive.sum()
                    ),

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
                    presence_oriented,

                "presence_flip":
                    int(
                        year != 2017
                        and presence_oriented < 0.5
                    ),

                # -------------------------
                # 出现后的支持强度
                # -------------------------

                "benign_positive_mean":
                    benign_positive_mean,

                "malware_positive_mean":
                    malware_positive_mean,

                "positive_auc":
                    positive_auc,

                "positive_oriented_auc":
                    positive_oriented,

                "positive_flip":
                    int(
                        year != 2017
                        and np.isfinite(
                            positive_oriented
                        )
                        and positive_oriented < 0.5
                    ),
            })

    detail_df = pd.DataFrame(
        detail_rows
    )

    detail_df.to_csv(
        OUT
        / "per_relation_year.csv",
        index=False,
    )

    # ========================================================
    # 汇总：一行一条关系
    # ========================================================

    summary_rows = []

    for name, g in detail_df.groupby(
        "feature",
        sort=False,
    ):
        g = (
            g
            .set_index(
                "year"
            )
            .sort_index()
        )

        source = g.loc[
            2017
        ]

        future = g.loc[
            [
                2018,
                2019,
                2020,
                2021,
            ]
        ]

        raw_future = (
            future[
                "raw_oriented_auc"
            ].to_numpy(
                dtype=float
            )
        )

        presence_future = (
            future[
                "presence_oriented_auc"
            ].to_numpy(
                dtype=float
            )
        )

        positive_future = (
            future[
                "positive_oriented_auc"
            ].to_numpy(
                dtype=float
            )
        )

        finite_positive = (
            positive_future[
                np.isfinite(
                    positive_future
                )
            ]
        )

        future_active = (
            future[
                "active_n"
            ].to_numpy(
                dtype=int
            )
        )

        source_disc = float(
            source[
                "raw_disc_auc"
            ]
        )

        future_oriented_mean = float(
            np.mean(
                raw_future
            )
        )

        future_oriented_min = float(
            np.min(
                raw_future
            )
        )

        raw_flip_count = int(
            future[
                "raw_flip"
            ].sum()
        )

        presence_flip_count = int(
            future[
                "presence_flip"
            ].sum()
        )

        positive_flip_count = int(
            future[
                "positive_flip"
            ].sum()
        )

        # ----------------------------------------------------
        # 保守稳定标志。
        #
        # 这里只作为开发诊断，
        # 不是最终论文中天然预定义的阈值。
        #
        # 要求：
        # 1. 2017有基本区分能力
        # 2. 四个未来年份没有方向反转
        # 3. 未来平均仍有一定方向性
        # 4. 关系不能极端稀疏
        # ----------------------------------------------------

        conservative_stable = int(
            source_disc >= 0.55
            and raw_flip_count == 0
            and future_oriented_mean >= 0.55
            and future_oriented_min >= 0.50
            and int(
                source[
                    "active_n"
                ]
            ) >= 20
            and int(
                np.min(
                    future_active
                )
            ) >= 5
        )

        # ----------------------------------------------------
        # 明显不稳定：
        # 2017有一定信号，但未来频繁反转。
        # ----------------------------------------------------

        clearly_unstable = int(
            source_disc >= 0.55
            and (
                raw_flip_count >= 2
                or future_oriented_mean < 0.48
            )
        )

        # ----------------------------------------------------
        # 出现率反转但“出现后的强度”相对稳定。
        # ----------------------------------------------------

        intensity_survives_presence_drift = int(
            presence_flip_count >= 2
            and len(
                finite_positive
            ) >= 3
            and positive_flip_count <= 1
            and float(
                np.mean(
                    finite_positive
                )
            ) >= 0.55
        )

        summary_rows.append({
            "feature":
                name,

            "relation_index":
                int(
                    source[
                        "relation_index"
                    ]
                ),

            "anchor":
                source[
                    "anchor"
                ],

            "context":
                source[
                    "context"
                ],

            "source_raw_auc":
                float(
                    source[
                        "raw_auc"
                    ]
                ),

            "source_disc_auc":
                source_disc,

            "source_presence_auc":
                float(
                    source[
                        "presence_auc"
                    ]
                ),

            "source_positive_auc":
                float(
                    source[
                        "positive_auc"
                    ]
                )
                if np.isfinite(
                    source[
                        "positive_auc"
                    ]
                )
                else np.nan,

            "source_active_n":
                int(
                    source[
                        "active_n"
                    ]
                ),

            "future_active_min":
                int(
                    np.min(
                        future_active
                    )
                ),

            "raw_oriented_2018":
                float(
                    g.loc[
                        2018,
                        "raw_oriented_auc"
                    ]
                ),

            "raw_oriented_2019":
                float(
                    g.loc[
                        2019,
                        "raw_oriented_auc"
                    ]
                ),

            "raw_oriented_2020":
                float(
                    g.loc[
                        2020,
                        "raw_oriented_auc"
                    ]
                ),

            "raw_oriented_2021":
                float(
                    g.loc[
                        2021,
                        "raw_oriented_auc"
                    ]
                ),

            "future_oriented_mean":
                future_oriented_mean,

            "future_oriented_min":
                future_oriented_min,

            "raw_flip_count":
                raw_flip_count,

            "presence_future_oriented_mean":
                float(
                    np.mean(
                        presence_future
                    )
                ),

            "presence_flip_count":
                presence_flip_count,

            "positive_future_oriented_mean":
                (
                    float(
                        np.mean(
                            finite_positive
                        )
                    )
                    if len(
                        finite_positive
                    )
                    else np.nan
                ),

            "positive_flip_count":
                positive_flip_count,

            "benign_presence_2017":
                float(
                    source[
                        "benign_presence"
                    ]
                ),

            "malware_presence_2017":
                float(
                    source[
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

            "conservative_stable":
                conservative_stable,

            "clearly_unstable":
                clearly_unstable,

            "intensity_survives_presence_drift":
                intensity_survives_presence_drift,
        })

    summary_df = pd.DataFrame(
        summary_rows
    )

    summary_df.to_csv(
        OUT
        / "relation_summary.csv",
        index=False,
    )

    # ========================================================
    # A. 最稳定关系
    # ========================================================

    stable_rank = (
        summary_df[
            summary_df[
                "conservative_stable"
            ] == 1
        ]
        .sort_values(
            [
                "future_oriented_min",
                "future_oriented_mean",
                "source_disc_auc",
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

    stable_rank.to_csv(
        OUT
        / "ranking_stable_relations.csv",
        index=False,
    )

    # ========================================================
    # B. 明显反转/不稳定关系
    # ========================================================

    unstable_rank = (
        summary_df[
            summary_df[
                "clearly_unstable"
            ] == 1
        ]
        .sort_values(
            [
                "raw_flip_count",
                "future_oriented_mean",
                "source_disc_auc",
            ],
            ascending=[
                False,
                True,
                False,
            ],
        )
        .reset_index(
            drop=True
        )
    )

    unstable_rank.to_csv(
        OUT
        / "ranking_unstable_relations.csv",
        index=False,
    )

    # ========================================================
    # C. 出现率不稳定，但出现后强度仍可能稳定
    # ========================================================

    conditional_rank = (
        summary_df[
            summary_df[
                "intensity_survives_presence_drift"
            ] == 1
        ]
        .sort_values(
            [
                "positive_future_oriented_mean",
                "presence_flip_count",
            ],
            ascending=[
                False,
                False,
            ],
        )
        .reset_index(
            drop=True
        )
    )

    conditional_rank.to_csv(
        OUT
        / "ranking_presence_drift_but_intensity_stable.csv",
        index=False,
    )

    # ========================================================
    # 总体统计
    # ========================================================

    counts = {
        "total":
            len(
                summary_df
            ),

        "stable":
            int(
                summary_df[
                    "conservative_stable"
                ].sum()
            ),

        "clearly_unstable":
            int(
                summary_df[
                    "clearly_unstable"
                ].sum()
            ),

        "presence_drift_intensity_stable":
            int(
                summary_df[
                    "intensity_survives_presence_drift"
                ].sum()
            ),

        "raw_flip_0":
            int(
                (
                    summary_df[
                        "raw_flip_count"
                    ] == 0
                ).sum()
            ),

        "raw_flip_1":
            int(
                (
                    summary_df[
                        "raw_flip_count"
                    ] == 1
                ).sum()
            ),

        "raw_flip_2plus":
            int(
                (
                    summary_df[
                        "raw_flip_count"
                    ] >= 2
                ).sum()
            ),
    }

    pd.DataFrame([
        counts
    ]).to_csv(
        OUT
        / "overall_counts.csv",
        index=False,
    )

    # ========================================================
    # 打印
    # ========================================================

    show_cols = [
        "feature",
        "source_disc_auc",
        "source_active_n",
        "future_active_min",
        "raw_oriented_2018",
        "raw_oriented_2019",
        "raw_oriented_2020",
        "raw_oriented_2021",
        "future_oriented_mean",
        "future_oriented_min",
        "raw_flip_count",
        "presence_flip_count",
        "positive_future_oriented_mean",
        "positive_flip_count",
    ]

    print()
    print(
        "=" * 130
    )
    print(
        "总体统计"
    )
    print(
        "=" * 130
    )

    for key, value in counts.items():
        print(
            f"{key:35s}",
            value,
        )

    print()
    print(
        "=" * 130
    )
    print(
        "A. 保守稳定关系 前30"
    )
    print(
        "=" * 130
    )

    if len(
        stable_rank
    ) == 0:
        print(
            "没有满足条件的关系"
        )
    else:
        print(
            stable_rank[
                show_cols
            ]
            .head(
                30
            )
            .to_string(
                index=False
            )
        )

    print()
    print(
        "=" * 130
    )
    print(
        "B. 明显不稳定/反转关系 前30"
    )
    print(
        "=" * 130
    )

    if len(
        unstable_rank
    ) == 0:
        print(
            "没有满足条件的关系"
        )
    else:
        print(
            unstable_rank[
                show_cols
            ]
            .head(
                30
            )
            .to_string(
                index=False
            )
        )

    print()
    print(
        "=" * 130
    )
    print(
        "C. 出现率漂移，但出现后强度相对稳定 前30"
    )
    print(
        "=" * 130
    )

    if len(
        conditional_rank
    ) == 0:
        print(
            "没有满足条件的关系"
        )
    else:
        print(
            conditional_rank[
                show_cols
            ]
            .head(
                30
            )
            .to_string(
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
