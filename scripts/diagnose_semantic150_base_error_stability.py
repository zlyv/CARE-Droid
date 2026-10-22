#!/usr/bin/env python3

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


PRED = Path(
    "artifacts/"
    "frozen-anchor-stable70-stable15-2017-2021/"
    "future_predictions.csv"
)

SEMANTIC = Path(
    "artifacts/"
    "clean-local-context-future/"
    "semantic_relation_features.csv.gz"
)

OUT = Path(
    "artifacts/"
    "semantic150-base-error-stability"
)

EPS = 1e-12


def logit(p):
    p = np.clip(
        np.asarray(p, dtype=float),
        EPS,
        1.0 - EPS,
    )
    return np.log(
        p / (1.0 - p)
    )


def safe_auc(y, score):
    y = np.asarray(
        y,
        dtype=int,
    )

    score = np.asarray(
        score,
        dtype=float,
    )

    if len(y) == 0:
        return np.nan

    if len(np.unique(y)) < 2:
        return np.nan

    if np.allclose(
        score,
        score[0],
    ):
        return 0.5

    return float(
        roc_auc_score(
            y,
            score,
        )
    )


def main():
    OUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    pred = pd.read_csv(
        PRED
    )

    sem = pd.read_csv(
        SEMANTIC
    )

    if "sha256" not in sem.columns:
        raise RuntimeError(
            "语义文件找不到 sha256"
        )

    sem["sha256"] = (
        sem["sha256"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    pred["sha256"] = (
        pred["sha256"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    all_relation_names = [
        c
        for c in sem.columns
        if c.startswith(
            "semrel_support__"
        )
    ]

    relation_names = []
    self_relations = []

    for name in all_relation_names:
        parts = name.split(
            "__",
            2,
        )

        if len(parts) != 3:
            raise RuntimeError(
                "无法解析语义关系名: "
                + name
            )

        anchor = parts[1]
        context = parts[2]

        if anchor == context:
            self_relations.append(
                name
            )
            continue

        relation_names.append(
            name
        )

    print(
        "原始support关系数 =",
        len(all_relation_names),
    )

    print(
        "排除self relation =",
        len(self_relations),
    )

    for name in self_relations:
        print(
            "  drop self:",
            name,
        )

    print(
        "正式关系数 =",
        len(relation_names),
    )

    if len(relation_names) != 150:
        raise RuntimeError(
            "排除self relation后预期150条关系，实际="
            + str(
                len(relation_names)
            )
        )

    sem = (
        sem[
            ["sha256"]
            + relation_names
        ]
        .drop_duplicates(
            "sha256"
        )
    )

    df = pred.merge(
        sem,
        on="sha256",
        how="left",
        validate="many_to_one",
    )

    if df[
        relation_names
    ].isna().any().any():
        missing = int(
            df[
                relation_names
            ].isna().all(
                axis=1
            ).sum()
        )

        raise RuntimeError(
            "存在语义关系缺失样本，N="
            + str(missing)
        )

    # ========================================================
    # Stable70基础错误
    # ========================================================

    base_p = df[
        "base_probability"
    ].to_numpy(
        dtype=float
    )

    base_z = logit(
        base_p
    )

    base_pred = (
        base_p >= 0.5
    ).astype(int)

    label = df[
        "label"
    ].to_numpy(
        dtype=int
    )

    df["base_wrong"] = (
        base_pred != label
    ).astype(int)

    # ========================================================
    # 不考虑任何具体关系presence时，
    # 当前4p(1-p)本身是否足以跨越边界。
    #
    # 这里只研究“理论可纠正区域”中的错误风险。
    # ========================================================

    uncertainty = (
        4.0
        * base_p
        * (
            1.0
            - base_p
        )
    )

    df["intrinsic_flippable"] = (
        uncertainty
        > (
            np.abs(
                base_z
            )
            + 1e-6
        )
    ).astype(int)

    print(
        "关系数 =",
        len(relation_names),
    )

    print()

    for year in sorted(
        df["year"].unique()
    ):
        x = df[
            df["year"] == year
        ]

        print(
            year,
            "N =",
            len(x) // x["seed"].nunique(),
            "可翻转候选/seed =",
            f"{x.groupby('seed')['intrinsic_flippable'].sum().mean():.1f}",
            "其中错误/seed =",
            f"{x[x['intrinsic_flippable'] == 1].groupby('seed')['base_wrong'].sum().mean():.1f}",
        )

    # ========================================================
    # 每条关系 × 年份 × seed
    # ========================================================

    rows = []

    for feature in relation_names:

        for (
            year,
            seed
        ), g in df.groupby(
            [
                "year",
                "seed",
            ],
            sort=True,
        ):
            # 只研究现有有界残差理论上能够翻转的区域。
            g = g[
                g[
                    "intrinsic_flippable"
                ] == 1
            ]

            y = g[
                "base_wrong"
            ].to_numpy(
                dtype=int
            )

            value = g[
                feature
            ].to_numpy(
                dtype=float
            )

            presence = (
                value > 0.0
            ).astype(float)

            raw_auc = safe_auc(
                y,
                value,
            )

            presence_auc = safe_auc(
                y,
                presence,
            )

            rows.append({
                "feature":
                    feature,

                "year":
                    int(year),

                "seed":
                    int(seed),

                "candidate_n":
                    len(g),

                "error_n":
                    int(
                        y.sum()
                    ),

                "active_n":
                    int(
                        presence.sum()
                    ),

                "raw_auc":
                    raw_auc,

                "presence_auc":
                    presence_auc,

                "error_rate_active":
                    (
                        float(
                            y[
                                presence > 0
                            ].mean()
                        )
                        if (
                            presence > 0
                        ).any()
                        else np.nan
                    ),

                "error_rate_inactive":
                    (
                        float(
                            y[
                                presence == 0
                            ].mean()
                        )
                        if (
                            presence == 0
                        ).any()
                        else np.nan
                    ),
            })

    detail = pd.DataFrame(
        rows
    )

    detail.to_csv(
        OUT
        / "per_relation_year_seed.csv",
        index=False,
    )

    # ========================================================
    # 先按年份平均5个seed
    # ========================================================

    by_year = (
        detail
        .groupby(
            [
                "feature",
                "year",
            ],
            as_index=False,
        )
        .agg({
            "candidate_n":
                "mean",

            "error_n":
                "mean",

            "active_n":
                "mean",

            "raw_auc":
                "mean",

            "presence_auc":
                "mean",

            "error_rate_active":
                "mean",

            "error_rate_inactive":
                "mean",
        })
    )

    by_year.to_csv(
        OUT
        / "per_relation_year.csv",
        index=False,
    )

    # ========================================================
    # 每条关系汇总
    #
    # 判断两种错误风险信号：
    #
    # 1. 原始支持强度
    # 2. 是否出现
    #
    # 不强制假设“越大越危险”，
    # 也允许越大反而表示Stable70更可靠。
    # ========================================================

    summaries = []

    for feature, g in by_year.groupby(
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

        if len(g) != 4:
            continue

        raw = g[
            "raw_auc"
        ].to_numpy(
            dtype=float
        )

        presence = g[
            "presence_auc"
        ].to_numpy(
            dtype=float
        )

        # 是否四年方向一致：
        # 全部 >= .5 或全部 <= .5
        raw_positive = bool(
            np.all(
                raw >= 0.5
            )
        )

        raw_negative = bool(
            np.all(
                raw <= 0.5
            )
        )

        presence_positive = bool(
            np.all(
                presence >= 0.5
            )
        )

        presence_negative = bool(
            np.all(
                presence <= 0.5
            )
        )

        raw_consistent = (
            raw_positive
            or raw_negative
        )

        presence_consistent = (
            presence_positive
            or presence_negative
        )

        if raw_positive:
            raw_oriented = raw

        elif raw_negative:
            raw_oriented = (
                1.0 - raw
            )

        else:
            # 仅用于排序，不代表稳定。
            direction = (
                1.0
                if np.nanmean(
                    raw
                ) >= 0.5
                else -1.0
            )

            raw_oriented = (
                raw
                if direction > 0
                else 1.0 - raw
            )

        if presence_positive:
            presence_oriented = (
                presence
            )

        elif presence_negative:
            presence_oriented = (
                1.0
                - presence
            )

        else:
            direction = (
                1.0
                if np.nanmean(
                    presence
                ) >= 0.5
                else -1.0
            )

            presence_oriented = (
                presence
                if direction > 0
                else 1.0 - presence
            )

        active = g[
            "active_n"
        ].to_numpy(
            dtype=float
        )

        summaries.append({
            "feature":
                feature,

            "raw_direction_consistent":
                int(
                    raw_consistent
                ),

            "raw_auc_2018":
                raw[0],

            "raw_auc_2019":
                raw[1],

            "raw_auc_2020":
                raw[2],

            "raw_auc_2021":
                raw[3],

            "raw_oriented_mean":
                float(
                    np.nanmean(
                        raw_oriented
                    )
                ),

            "raw_oriented_min":
                float(
                    np.nanmin(
                        raw_oriented
                    )
                ),

            "presence_direction_consistent":
                int(
                    presence_consistent
                ),

            "presence_auc_2018":
                presence[0],

            "presence_auc_2019":
                presence[1],

            "presence_auc_2020":
                presence[2],

            "presence_auc_2021":
                presence[3],

            "presence_oriented_mean":
                float(
                    np.nanmean(
                        presence_oriented
                    )
                ),

            "presence_oriented_min":
                float(
                    np.nanmin(
                        presence_oriented
                    )
                ),

            "future_active_min":
                float(
                    np.nanmin(
                        active
                    )
                ),

            "error_rate_active_mean":
                float(
                    np.nanmean(
                        g[
                            "error_rate_active"
                        ].to_numpy(
                            dtype=float
                        )
                    )
                ),

            "error_rate_inactive_mean":
                float(
                    np.nanmean(
                        g[
                            "error_rate_inactive"
                        ].to_numpy(
                            dtype=float
                        )
                    )
                ),
        })

    summary = pd.DataFrame(
        summaries
    )

    # ========================================================
    # 开发诊断候选：
    #
    # 不是最终论文阈值。
    #
    # 要求：
    # - 四年错误风险方向完全一致
    # - 平均区分能力 >= .58
    # - 最差年份 >= .52
    # - 每年平均至少5个激活样本
    # ========================================================

    summary[
        "raw_error_stable"
    ] = (
        (
            summary[
                "raw_direction_consistent"
            ] == 1
        )
        & (
            summary[
                "raw_oriented_mean"
            ] >= 0.58
        )
        & (
            summary[
                "raw_oriented_min"
            ] >= 0.52
        )
        & (
            summary[
                "future_active_min"
            ] >= 5
        )
    ).astype(int)

    summary[
        "presence_error_stable"
    ] = (
        (
            summary[
                "presence_direction_consistent"
            ] == 1
        )
        & (
            summary[
                "presence_oriented_mean"
            ] >= 0.58
        )
        & (
            summary[
                "presence_oriented_min"
            ] >= 0.52
        )
        & (
            summary[
                "future_active_min"
            ] >= 5
        )
    ).astype(int)

    summary.to_csv(
        OUT
        / "relation_error_stability_summary.csv",
        index=False,
    )

    raw_rank = (
        summary[
            summary[
                "raw_error_stable"
            ] == 1
        ]
        .sort_values(
            [
                "raw_oriented_min",
                "raw_oriented_mean",
            ],
            ascending=False,
        )
    )

    presence_rank = (
        summary[
            summary[
                "presence_error_stable"
            ] == 1
        ]
        .sort_values(
            [
                "presence_oriented_min",
                "presence_oriented_mean",
            ],
            ascending=False,
        )
    )

    raw_rank.to_csv(
        OUT
        / "ranking_raw_error_stable.csv",
        index=False,
    )

    presence_rank.to_csv(
        OUT
        / "ranking_presence_error_stable.csv",
        index=False,
    )

    print()
    print(
        "=" * 145
    )
    print(
        "总体"
    )
    print(
        "=" * 145
    )

    print(
        "raw稳定错误关系 =",
        int(
            summary[
                "raw_error_stable"
            ].sum()
        )
    )

    print(
        "presence稳定错误关系 =",
        int(
            summary[
                "presence_error_stable"
            ].sum()
        )
    )

    cols_raw = [
        "feature",
        "raw_auc_2018",
        "raw_auc_2019",
        "raw_auc_2020",
        "raw_auc_2021",
        "raw_oriented_mean",
        "raw_oriented_min",
        "future_active_min",
        "error_rate_active_mean",
        "error_rate_inactive_mean",
    ]

    print()
    print(
        "=" * 145
    )
    print(
        "原始支持值：稳定预测Stable70错误 前30"
    )
    print(
        "=" * 145
    )

    if len(raw_rank):
        print(
            raw_rank[
                cols_raw
            ]
            .head(30)
            .to_string(
                index=False
            )
        )
    else:
        print(
            "没有满足条件的关系"
        )

    cols_presence = [
        "feature",
        "presence_auc_2018",
        "presence_auc_2019",
        "presence_auc_2020",
        "presence_auc_2021",
        "presence_oriented_mean",
        "presence_oriented_min",
        "future_active_min",
        "error_rate_active_mean",
        "error_rate_inactive_mean",
    ]

    print()
    print(
        "=" * 145
    )
    print(
        "是否出现：稳定预测Stable70错误 前30"
    )
    print(
        "=" * 145
    )

    if len(
        presence_rank
    ):
        print(
            presence_rank[
                cols_presence
            ]
            .head(30)
            .to_string(
                index=False
            )
        )
    else:
        print(
            "没有满足条件的关系"
        )

    print()
    print(
        "结果目录:",
        OUT,
    )


if __name__ == "__main__":
    main()
