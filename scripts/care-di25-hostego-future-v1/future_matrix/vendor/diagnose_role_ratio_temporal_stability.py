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
    "role-ratio-temporal-stability"
)

YEARS = (
    2017,
    2018,
    2019,
    2020,
    2021,
)


def safe_float(x):
    try:
        if x is None:
            return 0.0

        text = str(x).strip()

        if not text:
            return 0.0

        value = float(text)

        if not np.isfinite(value):
            return 0.0

        return value

    except Exception:
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

    if len(np.unique(y)) < 2:
        return np.nan

    if len(x) == 0:
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


def norm_sha(x):
    return (
        str(x)
        .strip()
        .casefold()
    )


def get_rows_for_year(
    future_parts,
    year,
):
    if year == 2017:
        name = (
            "train_fixed_2017__test_2018"
        )

        if name not in future_parts:
            raise SystemExit(
                "找不到2017固定训练人口"
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

    if name not in future_parts:
        raise SystemExit(
            "找不到划分: "
            + name
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
    values = []

    for row in rows:
        sha = norm_sha(
            row[
                "sha256"
            ]
        )

        if sha not in feature_by_sha:
            raise RuntimeError(
                "features.csv缺失SHA: "
                + sha
            )

        values.append(
            safe_float(
                feature_by_sha[
                    sha
                ].get(
                    column,
                    0.0,
                )
            )
        )

    return np.asarray(
        values,
        dtype=float,
    )


def ratio_values(
    numerator,
    denominator,
):
    numerator = np.asarray(
        numerator,
        dtype=float,
    )

    denominator = np.asarray(
        denominator,
        dtype=float,
    )

    out = np.zeros_like(
        numerator,
        dtype=float,
    )

    mask = (
        denominator > 0.0
    )

    out[
        mask
    ] = (
        numerator[
            mask
        ]
        /
        denominator[
            mask
        ]
    )

    return out


def standardized(
    x,
    source_mean,
    source_std,
):
    if (
        not np.isfinite(
            source_std
        )
        or source_std < 1e-12
    ):
        return np.zeros_like(
            x,
            dtype=float,
        )

    return (
        np.asarray(
            x,
            dtype=float,
        )
        - source_mean
    ) / source_std


def analyze_representation(
    role,
    kind,
    representation,
    values_by_year,
    labels_by_year,
):
    """
    representation:
        raw
        ratio
    """

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

    source_auc = safe_auc(
        source_labels,
        source_values,
    )

    # 当前表示自己的2017方向。
    direction = (
        1
        if source_auc >= 0.5
        else -1
    )

    source_mean = float(
        np.mean(
            source_values
        )
    )

    source_std = float(
        np.std(
            source_values
        )
    )

    source_z = standardized(
        source_values,
        source_mean,
        source_std,
    )

    source_benign_mean = float(
        np.mean(
            source_z[
                source_labels == 0
            ]
        )
    )

    source_malware_mean = float(
        np.mean(
            source_z[
                source_labels == 1
            ]
        )
    )

    rows = []

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

        auc = safe_auc(
            labels,
            values,
        )

        oriented_auc = (
            auc
            if direction == 1
            else 1.0 - auc
        )

        z = standardized(
            values,
            source_mean,
            source_std,
        )

        benign_z = (
            z[
                labels == 0
            ]
        )

        malware_z = (
            z[
                labels == 1
            ]
        )

        benign_mean = float(
            np.mean(
                benign_z
            )
        )

        malware_mean = float(
            np.mean(
                malware_z
            )
        )

        benign_shift = (
            benign_mean
            - source_benign_mean
        )

        malware_shift = (
            malware_mean
            - source_malware_mean
        )

        # 两类如果一起向同一方向移动，
        # 这里就会比较大。
        common_shift = (
            0.5
            * (
                benign_shift
                + malware_shift
            )
        )

        rows.append({
            "role":
                role,

            "kind":
                kind,

            "representation":
                representation,

            "year":
                year,

            "source_direction":
                direction,

            "auc":
                auc,

            "disc_auc":
                max(
                    auc,
                    1.0 - auc,
                ),

            "oriented_auc":
                oriented_auc,

            "direction_flip":
                int(
                    year != 2017
                    and oriented_auc < 0.5
                ),

            "benign_z_mean":
                benign_mean,

            "malware_z_mean":
                malware_mean,

            "benign_shift":
                benign_shift,

            "malware_shift":
                malware_shift,

            "common_shift":
                common_shift,

            "abs_common_shift":
                abs(
                    common_shift
                ),
        })

    return rows


def main():
    OUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    raw_rows = (
        base.read_csv_rows(
            FEATURES
        )
    )

    feature_by_sha = (
        base.normalized_rows(
            raw_rows
        )
    )

    if not raw_rows:
        raise SystemExit(
            "features.csv为空"
        )

    columns = set(
        raw_rows[
            0
        ].keys()
    )

    node_roles = sorted([
        name[
            len(
                "role_node__"
            ):
        ]
        for name in columns
        if name.startswith(
            "role_node__"
        )
    ])

    edge_roles = sorted([
        name[
            len(
                "role_edge__"
            ):
        ]
        for name in columns
        if name.startswith(
            "role_edge__"
        )
    ])

    common_roles = sorted(
        set(
            node_roles
        )
        &
        set(
            edge_roles
        )
    )

    print(
        "node roles =",
        len(
            node_roles
        ),
    )

    print(
        "edge roles =",
        len(
            edge_roles
        ),
    )

    print(
        "common roles =",
        len(
            common_roles
        ),
    )

    future_parts = (
        base.read_split_rows(
            SPLITS
        )
    )

    rows_by_year = {
        year:
            get_rows_for_year(
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

    # ========================================================
    # 先缓存每年的总敏感节点/边。
    # ========================================================

    sensitive_node_by_year = {
        year:
            values_for(
                rows_by_year[
                    year
                ],
                feature_by_sha,
                "sensitive_node_count",
            )
        for year in YEARS
    }

    sensitive_edge_by_year = {
        year:
            values_for(
                rows_by_year[
                    year
                ],
                feature_by_sha,
                "sensitive_edge_count",
            )
        for year in YEARS
    }

    long_rows = []

    # ========================================================
    # 对每个角色：
    #
    # node:
    #   原始 role_node
    #   role_node / sensitive_node_count
    #
    # edge:
    #   原始 role_edge
    #   role_edge / sensitive_edge_count
    # ========================================================

    for role in common_roles:
        print(
            "role =",
            role,
        )

        node_column = (
            "role_node__"
            + role
        )

        edge_column = (
            "role_edge__"
            + role
        )

        raw_node = {
            year:
                values_for(
                    rows_by_year[
                        year
                    ],
                    feature_by_sha,
                    node_column,
                )
            for year in YEARS
        }

        ratio_node = {
            year:
                ratio_values(
                    raw_node[
                        year
                    ],
                    sensitive_node_by_year[
                        year
                    ],
                )
            for year in YEARS
        }

        raw_edge = {
            year:
                values_for(
                    rows_by_year[
                        year
                    ],
                    feature_by_sha,
                    edge_column,
                )
            for year in YEARS
        }

        ratio_edge = {
            year:
                ratio_values(
                    raw_edge[
                        year
                    ],
                    sensitive_edge_by_year[
                        year
                    ],
                )
            for year in YEARS
        }

        long_rows.extend(
            analyze_representation(
                role,
                "node",
                "raw",
                raw_node,
                labels_by_year,
            )
        )

        long_rows.extend(
            analyze_representation(
                role,
                "node",
                "ratio",
                ratio_node,
                labels_by_year,
            )
        )

        long_rows.extend(
            analyze_representation(
                role,
                "edge",
                "raw",
                raw_edge,
                labels_by_year,
            )
        )

        long_rows.extend(
            analyze_representation(
                role,
                "edge",
                "ratio",
                ratio_edge,
                labels_by_year,
            )
        )

    long_df = pd.DataFrame(
        long_rows
    )

    long_df.to_csv(
        OUT
        / "per_role_year.csv",
        index=False,
    )

    # ========================================================
    # 汇总：
    # 每个 role × node/edge × representation 一行。
    # ========================================================

    summary_rows = []

    for (
        role,
        kind,
        representation,
    ), g in long_df.groupby(
        [
            "role",
            "kind",
            "representation",
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

            "representation":
                representation,

            "source_auc":
                float(
                    g.loc[
                        2017,
                        "auc"
                    ]
                ),

            "source_disc_auc":
                float(
                    g.loc[
                        2017,
                        "disc_auc"
                    ]
                ),

            "oriented_auc_2018":
                float(
                    g.loc[
                        2018,
                        "oriented_auc"
                    ]
                ),

            "oriented_auc_2019":
                float(
                    g.loc[
                        2019,
                        "oriented_auc"
                    ]
                ),

            "oriented_auc_2020":
                float(
                    g.loc[
                        2020,
                        "oriented_auc"
                    ]
                ),

            "oriented_auc_2021":
                float(
                    g.loc[
                        2021,
                        "oriented_auc"
                    ]
                ),

            "future_oriented_auc_mean":
                float(
                    future[
                        "oriented_auc"
                    ].mean()
                ),

            "future_disc_auc_mean":
                float(
                    future[
                        "disc_auc"
                    ].mean()
                ),

            "flip_count":
                int(
                    future[
                        "direction_flip"
                    ].sum()
                ),

            "mean_abs_common_shift":
                float(
                    future[
                        "abs_common_shift"
                    ].mean()
                ),

            "shift_2021":
                float(
                    g.loc[
                        2021,
                        "common_shift"
                    ]
                ),
        })

    summary_df = pd.DataFrame(
        summary_rows
    )

    summary_df.to_csv(
        OUT
        / "representation_summary.csv",
        index=False,
    )

    # ========================================================
    # raw 与 ratio 一一配对。
    # ========================================================

    raw_summary = (
        summary_df[
            summary_df[
                "representation"
            ] == "raw"
        ]
        .drop(
            columns=[
                "representation",
            ]
        )
        .copy()
    )

    ratio_summary = (
        summary_df[
            summary_df[
                "representation"
            ] == "ratio"
        ]
        .drop(
            columns=[
                "representation",
            ]
        )
        .copy()
    )

    compare = raw_summary.merge(
        ratio_summary,
        on=[
            "role",
            "kind",
        ],
        suffixes=(
            "_raw",
            "_ratio",
        ),
        validate="one_to_one",
    )

    compare[
        "delta_future_oriented_auc"
    ] = (
        compare[
            "future_oriented_auc_mean_ratio"
        ]
        -
        compare[
            "future_oriented_auc_mean_raw"
        ]
    )

    compare[
        "delta_source_disc_auc"
    ] = (
        compare[
            "source_disc_auc_ratio"
        ]
        -
        compare[
            "source_disc_auc_raw"
        ]
    )

    compare[
        "flip_reduction"
    ] = (
        compare[
            "flip_count_raw"
        ]
        -
        compare[
            "flip_count_ratio"
        ]
    )

    compare[
        "common_shift_reduction"
    ] = (
        compare[
            "mean_abs_common_shift_raw"
        ]
        -
        compare[
            "mean_abs_common_shift_ratio"
        ]
    )

    # 一个保守的“值得进一步训练”标志：
    #
    # 未来旧方向平均区分能力提高，
    # 同时整体位置漂移没有变得更严重。
    compare[
        "promising"
    ] = (
        (
            compare[
                "delta_future_oriented_auc"
            ]
            > 0.01
        )
        &
        (
            compare[
                "common_shift_reduction"
            ]
            >= 0.0
        )
    ).astype(int)

    compare.to_csv(
        OUT
        / "raw_vs_ratio.csv",
        index=False,
    )

    # ========================================================
    # 排行1：
    # 比例化后，旧年份方向在未来保持得更好。
    # ========================================================

    rank_auc = (
        compare
        .sort_values(
            [
                "delta_future_oriented_auc",
                "flip_reduction",
                "common_shift_reduction",
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

    rank_auc.to_csv(
        OUT
        / "ranking_ratio_direction_improvement.csv",
        index=False,
    )

    # ========================================================
    # 排行2：
    # 比例化最明显减少整体位置漂移。
    # ========================================================

    rank_shift = (
        compare
        .sort_values(
            [
                "common_shift_reduction",
                "delta_future_oriented_auc",
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

    rank_shift.to_csv(
        OUT
        / "ranking_ratio_shift_reduction.csv",
        index=False,
    )

    # ========================================================
    # 特别关注之前已经看到的两组。
    # ========================================================

    focus_roles = [
        "command_execution",
        "task_or_alarm_scheduling",
        "icon_or_component_hiding",
        "device_identifier_access",
        "encoding_or_compression",
        "cryptography",
        "file_read",
        "dynamic_code_loading",
        "sensitive_file_or_content_write",
    ]

    focus = (
        compare[
            compare[
                "role"
            ].isin(
                focus_roles
            )
        ]
        .sort_values(
            [
                "role",
                "kind",
            ]
        )
    )

    focus.to_csv(
        OUT
        / "focus_roles.csv",
        index=False,
    )

    # ========================================================
    # 打印
    # ========================================================

    show_cols = [
        "role",
        "kind",
        "source_disc_auc_raw",
        "source_disc_auc_ratio",
        "future_oriented_auc_mean_raw",
        "future_oriented_auc_mean_ratio",
        "delta_future_oriented_auc",
        "flip_count_raw",
        "flip_count_ratio",
        "mean_abs_common_shift_raw",
        "mean_abs_common_shift_ratio",
        "common_shift_reduction",
    ]

    print()
    print(
        "=" * 120
    )
    print(
        "A. 比例化后，未来标签方向改善最多 前20"
    )
    print(
        "=" * 120
    )

    print(
        rank_auc[
            show_cols
        ]
        .head(
            20
        )
        .to_string(
            index=False
        )
    )

    print()
    print(
        "=" * 120
    )
    print(
        "B. 比例化后，整体位置漂移减少最多 前20"
    )
    print(
        "=" * 120
    )

    print(
        rank_shift[
            show_cols
        ]
        .head(
            20
        )
        .to_string(
            index=False
        )
    )

    print()
    print(
        "=" * 120
    )
    print(
        "C. 重点角色"
    )
    print(
        "=" * 120
    )

    print(
        focus[
            show_cols
        ]
        .to_string(
            index=False
        )
    )

    print()
    print(
        "=" * 120
    )
    print(
        "D. 同时满足：未来方向提升>0.01 且位置漂移减少"
    )
    print(
        "=" * 120
    )

    promising = (
        compare[
            compare[
                "promising"
            ] == 1
        ]
        .sort_values(
            "delta_future_oriented_auc",
            ascending=False,
        )
    )

    if len(
        promising
    ) == 0:
        print(
            "没有候选。"
        )
    else:
        print(
            promising[
                show_cols
            ]
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
