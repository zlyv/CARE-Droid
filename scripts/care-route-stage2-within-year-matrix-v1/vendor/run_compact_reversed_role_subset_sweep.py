#!/usr/bin/env python3

from __future__ import annotations

import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent

if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))


import run_compact_ratio3_temporal_eval as r3


base = r3.base


OUT = Path(
    "artifacts/"
    "compact-reversed-role-subset-sweep"
)


ROLES = (
    "command_execution",
    "task_or_alarm_scheduling",
    "icon_or_component_hiding",
    "device_identifier_access",
)


BASE_VARIANTS = (
    "compact76",
    "ratio3",
)


def labels(
    rows,
    feature_by_sha,
):
    return base.labels_for(
        rows,
        feature_by_sha,
    ).astype(int)


def mask_to_roles(mask):
    return tuple(
        role
        for i, role in enumerate(ROLES)
        if mask & (1 << i)
    )


def drop_names_for_mask(mask):
    names = set()

    for role in mask_to_roles(mask):
        names.add(
            f"role_node__{role}"
        )

        names.add(
            f"role_edge__{role}"
        )

    return names


def config_name(
    base_name,
    mask,
):
    roles = mask_to_roles(
        mask
    )

    if not roles:
        suffix = "drop_none"
    else:
        short = {
            "command_execution":
                "command",

            "task_or_alarm_scheduling":
                "task",

            "icon_or_component_hiding":
                "hiding",

            "device_identifier_access":
                "deviceid",
        }

        suffix = (
            "drop_"
            + "_".join(
                short[
                    role
                ]
                for role in roles
            )
        )

    return (
        base_name
        + "__"
        + suffix
    )


def build_base_variants(
    train_rows,
    test_rows,
    feature_by_sha,
    schema,
):
    """
    只构造两个基础版本：

    1. 原Compact
    2. Ratio3替换Compact
    """

    configs = r3.build_configs(
        train_rows,
        test_rows,
        feature_by_sha,
        schema,
    )

    return {
        "compact76":
            configs[
                "compact76"
            ],

        "ratio3":
            configs[
                "compact_replace_ratio3"
            ],
    }


def apply_drop_mask(
    x_train,
    x_test,
    names,
    mask,
):
    names = list(
        names
    )

    drop_names = (
        drop_names_for_mask(
            mask
        )
    )

    keep_idx = [
        i
        for i, name in enumerate(
            names
        )
        if name not in drop_names
    ]

    kept_names = [
        names[
            i
        ]
        for i in keep_idx
    ]

    actually_dropped = [
        name
        for name in names
        if name in drop_names
    ]

    return (
        x_train[
            :,
            keep_idx,
        ],
        x_test[
            :,
            keep_idx,
        ],
        kept_names,
        actually_dropped,
    )


def run_one_population(
    train_rows,
    test_rows,
    feature_by_sha,
    schema,
):
    y_train = labels(
        train_rows,
        feature_by_sha,
    )

    y_test = labels(
        test_rows,
        feature_by_sha,
    )

    bases = build_base_variants(
        train_rows,
        test_rows,
        feature_by_sha,
        schema,
    )

    rows = []

    for base_name in BASE_VARIANTS:
        (
            base_train,
            base_test,
            base_names,
        ) = bases[
            base_name
        ]

        for mask in range(
            1 << len(ROLES)
        ):
            (
                x_train,
                x_test,
                names,
                actually_dropped,
            ) = apply_drop_mask(
                base_train,
                base_test,
                base_names,
                mask,
            )

            cfg = config_name(
                base_name,
                mask,
            )

            roles = mask_to_roles(
                mask
            )

            for seed in r3.SEEDS:
                m = r3.evaluate(
                    x_train,
                    y_train,
                    x_test,
                    y_test,
                    seed,
                )

                rows.append({
                    "config":
                        cfg,

                    "base":
                        base_name,

                    "mask":
                        mask,

                    "drop_k_roles":
                        len(
                            roles
                        ),

                    "drop_roles":
                        "|".join(
                            roles
                        ),

                    "actually_dropped_features":
                        len(
                            actually_dropped
                        ),

                    "dim":
                        x_train.shape[
                            1
                        ],

                    "seed":
                        seed,

                    "BA":
                        float(
                            m[
                                "balanced_accuracy"
                            ]
                        ),

                    "AUC":
                        float(
                            m[
                                "auc"
                            ]
                        ),

                    "ACC":
                        float(
                            m[
                                "accuracy"
                            ]
                        ),

                    "FP":
                        int(
                            m[
                                "fp"
                            ]
                        ),

                    "FN":
                        int(
                            m[
                                "fn"
                            ]
                        ),
                })

    return rows


def summarize_source(
    source_df,
):
    rows = []

    for config, g in source_df.groupby(
        "config",
        sort=False,
    ):
        split_ba = (
            g.groupby(
                "split_name"
            )[
                "BA"
            ]
            .mean()
        )

        first = g.iloc[
            0
        ]

        rows.append({
            "config":
                config,

            "base":
                first[
                    "base"
                ],

            "mask":
                int(
                    first[
                        "mask"
                    ]
                ),

            "drop_k_roles":
                int(
                    first[
                        "drop_k_roles"
                    ]
                ),

            "drop_roles":
                first[
                    "drop_roles"
                ],

            "dim_min":
                int(
                    g[
                        "dim"
                    ].min()
                ),

            "dim_max":
                int(
                    g[
                        "dim"
                    ].max()
                ),

            "source_mean_BA":
                float(
                    split_ba.mean()
                ),

            "source_min_BA":
                float(
                    split_ba.min()
                ),

            "source_mean_AUC":
                float(
                    g[
                        "AUC"
                    ].mean()
                ),

            "source_mean_ACC":
                float(
                    g[
                        "ACC"
                    ].mean()
                ),
        })

    return pd.DataFrame(
        rows
    )


def summarize_future(
    future_seed_df,
):
    year_rows = []

    for (
        year,
        config,
    ), g in future_seed_df.groupby(
        [
            "year",
            "config",
        ],
        sort=False,
    ):
        first = g.iloc[
            0
        ]

        year_rows.append({
            "year":
                int(
                    year
                ),

            "config":
                config,

            "base":
                first[
                    "base"
                ],

            "mask":
                int(
                    first[
                        "mask"
                    ]
                ),

            "drop_k_roles":
                int(
                    first[
                        "drop_k_roles"
                    ]
                ),

            "drop_roles":
                first[
                    "drop_roles"
                ],

            "dim":
                int(
                    first[
                        "dim"
                    ]
                ),

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

    summary_rows = []

    for config, g in year_df.groupby(
        "config",
        sort=False,
    ):
        first = g.iloc[
            0
        ]

        summary_rows.append({
            "config":
                config,

            "base":
                first[
                    "base"
                ],

            "mask":
                int(
                    first[
                        "mask"
                    ]
                ),

            "drop_k_roles":
                int(
                    first[
                        "drop_k_roles"
                    ]
                ),

            "drop_roles":
                first[
                    "drop_roles"
                ],

            "future_mean_BA":
                float(
                    g[
                        "BA"
                    ].mean()
                ),

            "future_min_BA":
                float(
                    g[
                        "BA"
                    ].min()
                ),

            "future_mean_AUC":
                float(
                    g[
                        "AUC"
                    ].mean()
                ),

            "future_mean_ACC":
                float(
                    g[
                        "ACC"
                    ].mean()
                ),

            "BA_2018":
                float(
                    g.loc[
                        g[
                            "year"
                        ] == 2018,
                        "BA",
                    ].iloc[0]
                ),

            "BA_2019":
                float(
                    g.loc[
                        g[
                            "year"
                        ] == 2019,
                        "BA",
                    ].iloc[0]
                ),

            "BA_2020":
                float(
                    g.loc[
                        g[
                            "year"
                        ] == 2020,
                        "BA",
                    ].iloc[0]
                ),

            "BA_2021":
                float(
                    g.loc[
                        g[
                            "year"
                        ] == 2021,
                        "BA",
                    ].iloc[0]
                ),
        })

    return (
        year_df,
        pd.DataFrame(
            summary_rows
        ),
    )


def compute_marginal_effects(
    combined,
):
    """
    对每个基础表示、每个角色：

    固定其他三个角色的删除状态，
    比较：

        不删当前角色
        vs
        删除当前角色

    每个角色有8个严格配对。

    这样能看删除该角色是否稳定有益，
    而不是只依赖某一个最佳组合。
    """

    metrics = (
        "source_mean_BA",
        "source_min_BA",
        "source_mean_AUC",
        "future_mean_BA",
        "future_min_BA",
        "future_mean_AUC",
        "BA_2018",
        "BA_2019",
        "BA_2020",
        "BA_2021",
    )

    rows = []

    indexed = {
        (
            row[
                "base"
            ],
            int(
                row[
                    "mask"
                ]
            ),
        ):
            row
        for _, row
        in combined.iterrows()
    }

    for base_name in BASE_VARIANTS:
        for role_index, role in enumerate(
            ROLES
        ):
            bit = (
                1
                << role_index
            )

            pair_deltas = {
                metric: []
                for metric in metrics
            }

            pair_rows = []

            for mask in range(
                1 << len(ROLES)
            ):
                # 只从“不含当前角色”的组合出发。
                if mask & bit:
                    continue

                without = indexed[
                    (
                        base_name,
                        mask,
                    )
                ]

                with_role = indexed[
                    (
                        base_name,
                        mask | bit,
                    )
                ]

                pair = {
                    "base":
                        base_name,

                    "role":
                        role,

                    "mask_without":
                        mask,

                    "mask_with":
                        mask | bit,
                }

                for metric in metrics:
                    delta = (
                        float(
                            with_role[
                                metric
                            ]
                        )
                        -
                        float(
                            without[
                                metric
                            ]
                        )
                    )

                    pair_deltas[
                        metric
                    ].append(
                        delta
                    )

                    pair[
                        "delta_"
                        + metric
                    ] = delta

                pair_rows.append(
                    pair
                )

            future_ba = np.asarray(
                pair_deltas[
                    "future_mean_BA"
                ],
                dtype=float,
            )

            rows.append({
                "base":
                    base_name,

                "role":
                    role,

                "pairs":
                    len(
                        future_ba
                    ),

                "future_BA_mean_delta":
                    float(
                        future_ba.mean()
                    ),

                "future_BA_median_delta":
                    float(
                        np.median(
                            future_ba
                        )
                    ),

                "future_BA_wins":
                    int(
                        (
                            future_ba
                            > 1e-12
                        ).sum()
                    ),

                "future_BA_ties":
                    int(
                        (
                            np.abs(
                                future_ba
                            )
                            <= 1e-12
                        ).sum()
                    ),

                "future_BA_losses":
                    int(
                        (
                            future_ba
                            < -1e-12
                        ).sum()
                    ),

                "source_BA_mean_delta":
                    float(
                        np.mean(
                            pair_deltas[
                                "source_mean_BA"
                            ]
                        )
                    ),

                "source_min_BA_mean_delta":
                    float(
                        np.mean(
                            pair_deltas[
                                "source_min_BA"
                            ]
                        )
                    ),

                "source_AUC_mean_delta":
                    float(
                        np.mean(
                            pair_deltas[
                                "source_mean_AUC"
                            ]
                        )
                    ),

                "future_min_BA_mean_delta":
                    float(
                        np.mean(
                            pair_deltas[
                                "future_min_BA"
                            ]
                        )
                    ),

                "future_AUC_mean_delta":
                    float(
                        np.mean(
                            pair_deltas[
                                "future_mean_AUC"
                            ]
                        )
                    ),

                "BA_2018_mean_delta":
                    float(
                        np.mean(
                            pair_deltas[
                                "BA_2018"
                            ]
                        )
                    ),

                "BA_2019_mean_delta":
                    float(
                        np.mean(
                            pair_deltas[
                                "BA_2019"
                            ]
                        )
                    ),

                "BA_2020_mean_delta":
                    float(
                        np.mean(
                            pair_deltas[
                                "BA_2020"
                            ]
                        )
                    ),

                "BA_2021_mean_delta":
                    float(
                        np.mean(
                            pair_deltas[
                                "BA_2021"
                            ]
                        )
                    ),
            })

    return pd.DataFrame(
        rows
    )


def main():
    import json

    OUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    feature_by_sha = (
        base.normalized_rows(
            base.read_csv_rows(
                r3.FEATURES
            )
        )
    )

    schema = json.loads(
        r3.SCHEMA.read_text(
            encoding="utf-8"
        )
    )

    # ========================================================
    # 2017滚动验证
    # ========================================================

    rolling_parts = (
        base.read_split_rows(
            r3.ROLLING_SPLITS
        )
    )

    rolling = {
        name: parts
        for name, parts
        in rolling_parts.items()
        if (
            "rolling" in name
            and parts[
                "train"
            ]
            and parts[
                "test"
            ]
        )
    }

    source_rows = []

    for split_name in sorted(
        rolling
    ):
        print(
            "source split =",
            split_name,
            flush=True,
        )

        rows = run_one_population(
            rolling[
                split_name
            ][
                "train"
            ],
            rolling[
                split_name
            ][
                "test"
            ],
            feature_by_sha,
            schema,
        )

        for row in rows:
            row[
                "split_name"
            ] = split_name

        source_rows.extend(
            rows
        )

    source_df = pd.DataFrame(
        source_rows
    )

    source_df.to_csv(
        OUT
        / "source_per_seed.csv",
        index=False,
    )

    source_summary = summarize_source(
        source_df
    )

    source_summary.to_csv(
        OUT
        / "source_summary.csv",
        index=False,
    )

    # ========================================================
    # 2017全部 -> 2018-2021
    # ========================================================

    future_parts = (
        base.read_split_rows(
            r3.FUTURE_SPLITS
        )
    )

    train_rows = (
        future_parts[
            "train_fixed_2017__test_2018"
        ][
            "train"
        ]
    )

    future_rows = []

    for year in r3.YEARS:
        print(
            "future year =",
            year,
            flush=True,
        )

        split_name = (
            f"train_fixed_2017__test_{year}"
        )

        rows = run_one_population(
            train_rows,
            future_parts[
                split_name
            ][
                "test"
            ],
            feature_by_sha,
            schema,
        )

        for row in rows:
            row[
                "year"
            ] = year

        future_rows.extend(
            rows
        )

    future_seed_df = pd.DataFrame(
        future_rows
    )

    future_seed_df.to_csv(
        OUT
        / "future_per_seed.csv",
        index=False,
    )

    (
        future_year_df,
        future_summary,
    ) = summarize_future(
        future_seed_df
    )

    future_year_df.to_csv(
        OUT
        / "future_by_year.csv",
        index=False,
    )

    future_summary.to_csv(
        OUT
        / "future_summary.csv",
        index=False,
    )

    # ========================================================
    # 合并源域和未来
    # ========================================================

    combined = source_summary.merge(
        future_summary,
        on=[
            "config",
            "base",
            "mask",
            "drop_k_roles",
            "drop_roles",
        ],
        validate="one_to_one",
    )

    combined = combined.sort_values(
        [
            "future_mean_BA",
            "future_min_BA",
            "future_mean_AUC",
            "drop_k_roles",
        ],
        ascending=[
            False,
            False,
            False,
            True,
        ],
    ).reset_index(
        drop=True
    )

    combined.to_csv(
        OUT
        / "combined_ranking.csv",
        index=False,
    )

    # ========================================================
    # 每个角色的严格配对边际作用
    # ========================================================

    marginal = compute_marginal_effects(
        combined
    )

    marginal = marginal.sort_values(
        [
            "base",
            "future_BA_mean_delta",
        ],
        ascending=[
            True,
            False,
        ],
    )

    marginal.to_csv(
        OUT
        / "marginal_role_effects.csv",
        index=False,
    )

    # ========================================================
    # 打印
    # ========================================================

    show_cols = [
        "config",
        "drop_k_roles",
        "drop_roles",
        "source_mean_BA",
        "source_min_BA",
        "source_mean_AUC",
        "future_mean_BA",
        "future_min_BA",
        "future_mean_AUC",
        "BA_2018",
        "BA_2019",
        "BA_2020",
        "BA_2021",
    ]

    print()
    print(
        "=" * 130
    )
    print(
        "全部32个配置：未来平均BA前15"
    )
    print(
        "=" * 130
    )

    print(
        combined[
            show_cols
        ]
        .head(
            15
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
        "原Compact：前8"
    )
    print(
        "=" * 130
    )

    print(
        combined[
            combined[
                "base"
            ] == "compact76"
        ][
            show_cols
        ]
        .head(
            8
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
        "Ratio3：前8"
    )
    print(
        "=" * 130
    )

    print(
        combined[
            combined[
                "base"
            ] == "ratio3"
        ][
            show_cols
        ]
        .head(
            8
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
        "每个角色的配对边际作用"
    )
    print(
        "正数 = 删除该角色后指标提高"
    )
    print(
        "=" * 130
    )

    marginal_cols = [
        "base",
        "role",
        "pairs",
        "future_BA_mean_delta",
        "future_BA_median_delta",
        "future_BA_wins",
        "future_BA_ties",
        "future_BA_losses",
        "future_min_BA_mean_delta",
        "future_AUC_mean_delta",
        "source_BA_mean_delta",
        "source_min_BA_mean_delta",
        "BA_2018_mean_delta",
        "BA_2019_mean_delta",
        "BA_2020_mean_delta",
        "BA_2021_mean_delta",
    ]

    print(
        marginal[
            marginal_cols
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
