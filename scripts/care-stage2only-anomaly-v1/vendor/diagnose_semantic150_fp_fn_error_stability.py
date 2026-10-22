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
    "semantic150-fp-fn-error-stability"
)

EPS = 1e-12


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


def safe_mean(values):
    values = np.asarray(
        values,
        dtype=float,
    )

    values = values[
        np.isfinite(values)
    ]

    if len(values) == 0:
        return np.nan

    return float(
        np.mean(values)
    )


def safe_min(values):
    values = np.asarray(
        values,
        dtype=float,
    )

    values = values[
        np.isfinite(values)
    ]

    if len(values) == 0:
        return np.nan

    return float(
        np.min(values)
    )


def logit(p):
    p = np.clip(
        np.asarray(p, dtype=float),
        EPS,
        1.0 - EPS,
    )

    return np.log(
        p / (1.0 - p)
    )


def summarize_direction(values):
    values = np.asarray(
        values,
        dtype=float,
    )

    if len(values) != 4:
        return {
            "finite_all":
                0,
            "consistent":
                0,
            "direction":
                "unknown",
            "oriented_mean":
                np.nan,
            "oriented_min":
                np.nan,
        }

    if not np.all(
        np.isfinite(values)
    ):
        return {
            "finite_all":
                0,
            "consistent":
                0,
            "direction":
                "unknown",
            "oriented_mean":
                np.nan,
            "oriented_min":
                np.nan,
        }

    positive = bool(
        np.all(
            values >= 0.5
        )
    )

    negative = bool(
        np.all(
            values <= 0.5
        )
    )

    consistent = (
        positive
        or negative
    )

    if positive:
        oriented = values
        direction = "higher_means_more_error"

    elif negative:
        oriented = (
            1.0 - values
        )
        direction = "higher_means_less_error"

    else:
        # 只用于辅助排序。
        if np.mean(values) >= 0.5:
            oriented = values
            direction = "mixed_mean_positive"
        else:
            oriented = (
                1.0 - values
            )
            direction = "mixed_mean_negative"

    return {
        "finite_all":
            1,
        "consistent":
            int(
                consistent
            ),
        "direction":
            direction,
        "oriented_mean":
            float(
                np.mean(
                    oriented
                )
            ),
        "oriented_min":
            float(
                np.min(
                    oriented
                )
            ),
    }


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

    pred["sha256"] = (
        pred["sha256"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    sem["sha256"] = (
        sem["sha256"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    # ========================================================
    # 正式150维：
    # 155 support slots - 5 self relations。
    # ========================================================

    all_relations = [
        c
        for c in sem.columns
        if c.startswith(
            "semrel_support__"
        )
    ]

    relation_names = []

    for name in all_relations:
        parts = name.split(
            "__",
            2,
        )

        if len(parts) != 3:
            raise RuntimeError(
                "无法解析关系名: "
                + name
            )

        anchor = parts[1]
        context = parts[2]

        if anchor == context:
            continue

        relation_names.append(
            name
        )

    if len(relation_names) != 150:
        raise RuntimeError(
            "正式关系应为150，实际="
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
    ].isna().all(
        axis=1
    ).any():
        raise RuntimeError(
            "存在未来样本没有语义表示"
        )

    # ========================================================
    # Stable70基础预测。
    # ========================================================

    base_probability = df[
        "base_probability"
    ].to_numpy(
        dtype=float
    )

    base_logit = logit(
        base_probability
    )

    base_pred = (
        base_probability >= 0.5
    ).astype(int)

    label = df[
        "label"
    ].to_numpy(
        dtype=int
    )

    uncertainty = (
        4.0
        * base_probability
        * (
            1.0
            - base_probability
        )
    )

    df["base_pred"] = (
        base_pred
    )

    df["base_wrong"] = (
        base_pred != label
    ).astype(int)

    # 当前有界残差自身理论上可跨界。
    df["intrinsic_flippable"] = (
        uncertainty
        > (
            np.abs(
                base_logit
            )
            + 1e-6
        )
    ).astype(int)

    # ========================================================
    # 先打印两侧候选量。
    #
    # predicted_malware:
    #     Stable70当前预测恶意。
    #     error=1意味着FP。
    #
    # predicted_benign:
    #     Stable70当前预测良性。
    #     error=1意味着FN。
    # ========================================================

    print(
        "=" * 100
    )

    print(
        "当前有界结构中的两侧候选规模"
    )

    print(
        "=" * 100
    )

    for year in sorted(
        df["year"].unique()
    ):
        gy = df[
            (
                df["year"] == year
            )
            & (
                df[
                    "intrinsic_flippable"
                ] == 1
            )
        ]

        print()
        print(
            "year =",
            year,
        )

        for side_name, pred_value in (
            (
                "predicted_malware / FP风险",
                1,
            ),
            (
                "predicted_benign / FN风险",
                0,
            ),
        ):
            gs = gy[
                gy[
                    "base_pred"
                ] == pred_value
            ]

            if len(gs) == 0:
                continue

            by_seed = (
                gs.groupby(
                    "seed"
                )
                .agg(
                    candidate_n=(
                        "base_wrong",
                        "size",
                    ),
                    error_n=(
                        "base_wrong",
                        "sum",
                    ),
                )
            )

            print(
                f"  {side_name:30s}",
                "candidate/seed =",
                f"{by_seed['candidate_n'].mean():.1f}",
                "error/seed =",
                f"{by_seed['error_n'].mean():.1f}",
                "error rate =",
                f"{by_seed['error_n'].sum() / by_seed['candidate_n'].sum():.4f}",
            )

    # ========================================================
    # 每条关系 × 年份 × seed × 预测侧。
    # ========================================================

    detail_rows = []

    side_specs = (
        (
            "predicted_malware",
            1,
            "FP",
        ),
        (
            "predicted_benign",
            0,
            "FN",
        ),
    )

    grouped = list(
        df.groupby(
            [
                "year",
                "seed",
            ],
            sort=True,
        )
    )

    for feature in relation_names:

        for (
            side,
            pred_value,
            error_type,
        ) in side_specs:

            for (
                year,
                seed
            ), g0 in grouped:

                g = g0[
                    (
                        g0[
                            "intrinsic_flippable"
                        ] == 1
                    )
                    & (
                        g0[
                            "base_pred"
                        ] == pred_value
                    )
                ]

                target = g[
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
                    target,
                    value,
                )

                presence_auc = safe_auc(
                    target,
                    presence,
                )

                active = (
                    presence > 0.0
                )

                inactive = (
                    ~active
                )

                detail_rows.append({
                    "feature":
                        feature,

                    "side":
                        side,

                    "error_type":
                        error_type,

                    "year":
                        int(year),

                    "seed":
                        int(seed),

                    "candidate_n":
                        len(g),

                    "error_n":
                        int(
                            target.sum()
                        ),

                    "active_n":
                        int(
                            active.sum()
                        ),

                    "raw_auc":
                        raw_auc,

                    "presence_auc":
                        presence_auc,

                    "error_rate_active":
                        (
                            float(
                                target[
                                    active
                                ].mean()
                            )
                            if active.any()
                            else np.nan
                        ),

                    "error_rate_inactive":
                        (
                            float(
                                target[
                                    inactive
                                ].mean()
                            )
                            if inactive.any()
                            else np.nan
                        ),
                })

    detail = pd.DataFrame(
        detail_rows
    )

    detail.to_csv(
        OUT
        / "per_relation_side_year_seed.csv",
        index=False,
    )

    # ========================================================
    # 五个seed先按年份平均。
    # ========================================================

    by_year = (
        detail
        .groupby(
            [
                "feature",
                "side",
                "error_type",
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
        / "per_relation_side_year.csv",
        index=False,
    )

    # ========================================================
    # 汇总四年稳定性。
    # ========================================================

    summary_rows = []

    for (
        feature,
        side,
        error_type
    ), g in by_year.groupby(
        [
            "feature",
            "side",
            "error_type",
        ],
        sort=False,
    ):
        g = (
            g
            .set_index(
                "year"
            )
            .sort_index()
        )

        years = [
            2018,
            2019,
            2020,
            2021,
        ]

        if not all(
            year in g.index
            for year in years
        ):
            continue

        raw_values = np.asarray(
            [
                g.loc[
                    year,
                    "raw_auc"
                ]
                for year in years
            ],
            dtype=float,
        )

        presence_values = np.asarray(
            [
                g.loc[
                    year,
                    "presence_auc"
                ]
                for year in years
            ],
            dtype=float,
        )

        raw_info = (
            summarize_direction(
                raw_values
            )
        )

        presence_info = (
            summarize_direction(
                presence_values
            )
        )

        active_values = np.asarray(
            [
                g.loc[
                    year,
                    "active_n"
                ]
                for year in years
            ],
            dtype=float,
        )

        candidate_values = np.asarray(
            [
                g.loc[
                    year,
                    "candidate_n"
                ]
                for year in years
            ],
            dtype=float,
        )

        error_values = np.asarray(
            [
                g.loc[
                    year,
                    "error_n"
                ]
                for year in years
            ],
            dtype=float,
        )

        row = {
            "feature":
                feature,

            "side":
                side,

            "error_type":
                error_type,

            "candidate_min":
                safe_min(
                    candidate_values
                ),

            "error_min":
                safe_min(
                    error_values
                ),

            "active_min":
                safe_min(
                    active_values
                ),

            "raw_auc_2018":
                raw_values[0],

            "raw_auc_2019":
                raw_values[1],

            "raw_auc_2020":
                raw_values[2],

            "raw_auc_2021":
                raw_values[3],

            "raw_direction_consistent":
                raw_info[
                    "consistent"
                ],

            "raw_direction":
                raw_info[
                    "direction"
                ],

            "raw_oriented_mean":
                raw_info[
                    "oriented_mean"
                ],

            "raw_oriented_min":
                raw_info[
                    "oriented_min"
                ],

            "presence_auc_2018":
                presence_values[0],

            "presence_auc_2019":
                presence_values[1],

            "presence_auc_2020":
                presence_values[2],

            "presence_auc_2021":
                presence_values[3],

            "presence_direction_consistent":
                presence_info[
                    "consistent"
                ],

            "presence_direction":
                presence_info[
                    "direction"
                ],

            "presence_oriented_mean":
                presence_info[
                    "oriented_mean"
                ],

            "presence_oriented_min":
                presence_info[
                    "oriented_min"
                ],

            "error_rate_active_mean":
                safe_mean(
                    g[
                        "error_rate_active"
                    ].to_numpy(
                        dtype=float
                    )
                ),

            "error_rate_inactive_mean":
                safe_mean(
                    g[
                        "error_rate_inactive"
                    ].to_numpy(
                        dtype=float
                    )
                ),
        }

        # ----------------------------------------------------
        # 这里只作为开发诊断。
        #
        # 因为FP/FN拆开以后样本更少，
        # 阈值比上一轮稍微放松：
        #
        # 四年方向一致；
        # 平均定向AUC >= .57；
        # 最差年份 >= .51；
        # 每年至少3个active；
        # 每年该侧至少10个候选；
        # 每年至少2个错误。
        # ----------------------------------------------------

        row[
            "raw_side_stable"
        ] = int(
            row[
                "raw_direction_consistent"
            ] == 1
            and np.isfinite(
                row[
                    "raw_oriented_mean"
                ]
            )
            and row[
                "raw_oriented_mean"
            ] >= 0.57
            and row[
                "raw_oriented_min"
            ] >= 0.51
            and row[
                "active_min"
            ] >= 3
            and row[
                "candidate_min"
            ] >= 10
            and row[
                "error_min"
            ] >= 2
        )

        row[
            "presence_side_stable"
        ] = int(
            row[
                "presence_direction_consistent"
            ] == 1
            and np.isfinite(
                row[
                    "presence_oriented_mean"
                ]
            )
            and row[
                "presence_oriented_mean"
            ] >= 0.57
            and row[
                "presence_oriented_min"
            ] >= 0.51
            and row[
                "active_min"
            ] >= 3
            and row[
                "candidate_min"
            ] >= 10
            and row[
                "error_min"
            ] >= 2
        )

        summary_rows.append(
            row
        )

    summary = pd.DataFrame(
        summary_rows
    )

    summary.to_csv(
        OUT
        / "side_stability_summary.csv",
        index=False,
    )

    # ========================================================
    # 排名输出。
    # ========================================================

    print()
    print(
        "=" * 150
    )
    print(
        "分侧稳定关系总体"
    )
    print(
        "=" * 150
    )

    for side, error_type in (
        (
            "predicted_malware",
            "FP",
        ),
        (
            "predicted_benign",
            "FN",
        ),
    ):
        x = summary[
            summary[
                "side"
            ] == side
        ]

        print()
        print(
            error_type,
            "侧：raw稳定 =",
            int(
                x[
                    "raw_side_stable"
                ].sum()
            ),
            "presence稳定 =",
            int(
                x[
                    "presence_side_stable"
                ].sum()
            ),
        )

    show_raw = [
        "feature",
        "error_type",
        "raw_direction",
        "raw_auc_2018",
        "raw_auc_2019",
        "raw_auc_2020",
        "raw_auc_2021",
        "raw_oriented_mean",
        "raw_oriented_min",
        "active_min",
        "candidate_min",
        "error_min",
        "error_rate_active_mean",
        "error_rate_inactive_mean",
    ]

    show_presence = [
        "feature",
        "error_type",
        "presence_direction",
        "presence_auc_2018",
        "presence_auc_2019",
        "presence_auc_2020",
        "presence_auc_2021",
        "presence_oriented_mean",
        "presence_oriented_min",
        "active_min",
        "candidate_min",
        "error_min",
        "error_rate_active_mean",
        "error_rate_inactive_mean",
    ]

    for side, title in (
        (
            "predicted_malware",
            "FP风险",
        ),
        (
            "predicted_benign",
            "FN风险",
        ),
    ):
        x = summary[
            summary[
                "side"
            ] == side
        ]

        raw_rank = (
            x[
                x[
                    "raw_direction_consistent"
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
            x[
                x[
                    "presence_direction_consistent"
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
            / (
                f"ranking_{side}_raw.csv"
            ),
            index=False,
        )

        presence_rank.to_csv(
            OUT
            / (
                f"ranking_{side}_presence.csv"
            ),
            index=False,
        )

        print()
        print(
            "=" * 150
        )
        print(
            title,
            "：原始支持值方向一致 前20"
        )
        print(
            "=" * 150
        )

        if len(raw_rank):
            print(
                raw_rank[
                    show_raw
                ]
                .head(20)
                .to_string(
                    index=False
                )
            )
        else:
            print(
                "没有四年方向一致关系"
            )

        print()
        print(
            "=" * 150
        )
        print(
            title,
            "：是否出现方向一致 前20"
        )
        print(
            "=" * 150
        )

        if len(
            presence_rank
        ):
            print(
                presence_rank[
                    show_presence
                ]
                .head(20)
                .to_string(
                    index=False
                )
            )
        else:
            print(
                "没有四年方向一致关系"
            )

    print()
    print(
        "结果目录:",
        OUT,
    )


if __name__ == "__main__":
    main()
