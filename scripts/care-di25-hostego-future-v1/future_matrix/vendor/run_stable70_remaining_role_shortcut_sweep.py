#!/usr/bin/env python3

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent

if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(
        0,
        str(SCRIPT_DIR),
    )


# ============================================================
# 复用已经完整跑通的旧subset sweep。
#
# 保留：
#   数据读取
#   2017 rolling
#   2017 -> 2018-2021
#   5 seeds
#   逻辑回归参数
#   summarize_source
#   summarize_future
#   main
#
# 只替换：
#   1. base -> Stable70
#   2. 不做2^N组合
#   3. 每个剩余角色单独删除 node+edge
# ============================================================

import run_compact_reversed_role_subset_sweep as old


r3 = old.r3


OUT = Path(
    "artifacts/"
    "stable70-remaining-role-shortcut-sweep"
)


# ============================================================
# Stable70已经正式删除的6维。
# ============================================================

ALREADY_DROPPED = {
    "role_node__command_execution",
    "role_edge__command_execution",

    "role_node__task_or_alarm_scheduling",
    "role_edge__task_or_alarm_scheduling",

    "role_node__icon_or_component_hiding",
    "role_edge__icon_or_component_hiding",
}


# ============================================================
# 从Ratio3构造Stable70。
#
# Ratio3自身已经把：
#
# dynamic_code_loading node
# native_library_loading node
# native_library_loading edge
#
# 替换成比例表示。
#
# 所以后面自动寻找“node+edge成对原始角色”时，
# 这两个角色天然不会进入候选。
# ============================================================

def build_base_variants_stable70(
    train_rows,
    test_rows,
    feature_by_sha,
    schema,
):
    configs = r3.build_configs(
        train_rows,
        test_rows,
        feature_by_sha,
        schema,
    )

    (
        x_train,
        x_test,
        names,
    ) = configs[
        "compact_replace_ratio3"
    ]

    x_train = np.asarray(
        x_train,
        dtype=float,
    )

    x_test = np.asarray(
        x_test,
        dtype=float,
    )

    names = list(
        names
    )

    found = {
        name
        for name in names
        if name in ALREADY_DROPPED
    }

    if found != ALREADY_DROPPED:
        missing = (
            ALREADY_DROPPED
            - found
        )

        raise RuntimeError(
            "Ratio3中缺少Stable70正式删除特征: "
            + str(
                sorted(
                    missing
                )
            )
        )

    keep_idx = [
        i
        for i, name in enumerate(
            names
        )
        if name not in ALREADY_DROPPED
    ]

    x_train = (
        x_train[
            :,
            keep_idx,
        ].copy()
    )

    x_test = (
        x_test[
            :,
            keep_idx,
        ].copy()
    )

    names = [
        names[i]
        for i in keep_idx
    ]

    if len(names) != 70:
        raise RuntimeError(
            "Stable70预期70维，实际="
            + str(
                len(names)
            )
        )

    return {
        "stable70":
            (
                x_train,
                x_test,
                names,
            ),
    }


# ============================================================
# 自动发现Stable70中仍然同时存在：
#
# role_node__X
# role_edge__X
#
# 的角色。
#
# 只对这种角色做成对删除。
# ============================================================

def pairable_roles(
    names,
):
    names = set(
        names
    )

    node_roles = {
        name[
            len(
                "role_node__"
            ):
        ]
        for name in names
        if name.startswith(
            "role_node__"
        )
    }

    edge_roles = {
        name[
            len(
                "role_edge__"
            ):
        ]
        for name in names
        if name.startswith(
            "role_edge__"
        )
    }

    roles = sorted(
        node_roles
        & edge_roles
    )

    # 安全检查：
    # 已经删除的三个角色绝不能重新出现。
    forbidden = {
        "command_execution",
        "task_or_alarm_scheduling",
        "icon_or_component_hiding",
    }

    bad = (
        set(
            roles
        )
        & forbidden
    )

    if bad:
        raise RuntimeError(
            "Stable70中重新出现了已删除角色: "
            + str(
                sorted(
                    bad
                )
            )
        )

    return roles


def apply_single_role_drop(
    x_train,
    x_test,
    names,
    role,
):
    names = list(
        names
    )

    if role is None:
        return (
            x_train.copy(),
            x_test.copy(),
            names,
            [],
        )

    drop_names = {
        f"role_node__{role}",
        f"role_edge__{role}",
    }

    found = {
        name
        for name in names
        if name in drop_names
    }

    if found != drop_names:
        raise RuntimeError(
            "无法成对删除角色 "
            + role
            + "，实际找到="
            + str(
                sorted(
                    found
                )
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
        names[i]
        for i in keep_idx
    ]

    return (
        x_train[
            :,
            keep_idx,
        ].copy(),

        x_test[
            :,
            keep_idx,
        ].copy(),

        kept_names,

        sorted(
            drop_names
        ),
    )


# ============================================================
# 替换旧run_one_population。
#
# configs:
#
#   Stable70基准
#   Stable70 - role1 node/edge
#   Stable70 - role2 node/edge
#   ...
#
# 不做角色组合。
# ============================================================

def run_one_population_single_drop(
    train_rows,
    test_rows,
    feature_by_sha,
    schema,
):
    y_train = old.labels(
        train_rows,
        feature_by_sha,
    )

    y_test = old.labels(
        test_rows,
        feature_by_sha,
    )

    bases = (
        build_base_variants_stable70(
            train_rows,
            test_rows,
            feature_by_sha,
            schema,
        )
    )

    (
        base_train,
        base_test,
        base_names,
    ) = bases[
        "stable70"
    ]

    roles = pairable_roles(
        base_names
    )

    print()
    print(
        "======================================"
    )
    print(
        "Stable70 remaining-role audit"
    )
    print(
        "======================================"
    )
    print(
        "Stable70 dim =",
        len(
            base_names
        ),
    )
    print(
        "pairable roles =",
        len(
            roles
        ),
    )

    for i, role in enumerate(
        roles,
        1,
    ):
        print(
            f"{i:2d}",
            role,
        )

    rows = []

    # baseline + 每个角色单独删除
    candidates = [
        None
    ] + roles

    for candidate_index, role in enumerate(
        candidates
    ):
        (
            x_train,
            x_test,
            names,
            actually_dropped,
        ) = apply_single_role_drop(
            base_train,
            base_test,
            base_names,
            role,
        )

        if role is None:
            cfg = (
                "stable70__drop_none"
            )

            drop_roles = ""
            drop_k = 0
            mask = 0

        else:
            cfg = (
                "stable70__drop_"
                + role
            )

            drop_roles = role
            drop_k = 1

            # 这里只作为唯一编号，
            # 不再表示旧脚本里的bit mask。
            mask = candidate_index

        expected_dim = (
            70
            if role is None
            else 68
        )

        if x_train.shape[1] != expected_dim:
            raise RuntimeError(
                cfg
                + " 维数异常，预期="
                + str(
                    expected_dim
                )
                + " 实际="
                + str(
                    x_train.shape[1]
                )
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
                    "stable70",

                "mask":
                    mask,

                "drop_k_roles":
                    drop_k,

                "drop_roles":
                    drop_roles,

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


# ============================================================
# 旧compute_marginal_effects假设2^N完整组合。
#
# 这一轮只有：
#
#   baseline
#   baseline-role
#
# 所以重新定义为严格的单删除差值。
#
# 保持旧main可能读取的字段名。
# ============================================================

def compute_single_drop_effects(
    combined,
):
    base_rows = combined[
        combined[
            "drop_k_roles"
        ] == 0
    ]

    if len(base_rows) != 1:
        raise RuntimeError(
            "预期只有一个Stable70 baseline，实际="
            + str(
                len(
                    base_rows
                )
            )
        )

    baseline = (
        base_rows.iloc[
            0
        ]
    )

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

    candidates = combined[
        combined[
            "drop_k_roles"
        ] == 1
    ]

    for _, row in candidates.iterrows():

        result = {
            "base":
                "stable70",

            "role":
                row[
                    "drop_roles"
                ],

            "pairs":
                1,
        }

        deltas = {}

        for metric in metrics:
            delta = (
                float(
                    row[
                        metric
                    ]
                )
                - float(
                    baseline[
                        metric
                    ]
                )
            )

            deltas[
                metric
            ] = delta

        future_delta = deltas[
            "future_mean_BA"
        ]

        result.update({
            "future_BA_mean_delta":
                future_delta,

            "future_BA_median_delta":
                future_delta,

            "future_BA_wins":
                int(
                    future_delta
                    > 1e-12
                ),

            "future_BA_ties":
                int(
                    abs(
                        future_delta
                    )
                    <= 1e-12
                ),

            "future_BA_losses":
                int(
                    future_delta
                    < -1e-12
                ),

            "source_BA_mean_delta":
                deltas[
                    "source_mean_BA"
                ],

            "source_min_BA_mean_delta":
                deltas[
                    "source_min_BA"
                ],

            "source_AUC_mean_delta":
                deltas[
                    "source_mean_AUC"
                ],

            "future_min_BA_mean_delta":
                deltas[
                    "future_min_BA"
                ],

            "future_AUC_mean_delta":
                deltas[
                    "future_mean_AUC"
                ],

            "BA_2018_mean_delta":
                deltas[
                    "BA_2018"
                ],

            "BA_2019_mean_delta":
                deltas[
                    "BA_2019"
                ],

            "BA_2020_mean_delta":
                deltas[
                    "BA_2020"
                ],

            "BA_2021_mean_delta":
                deltas[
                    "BA_2021"
                ],
        })

        rows.append(
            result
        )

    return pd.DataFrame(
        rows
    )


# ============================================================
# old.main结束后，再生成一个我们真正需要的
# “时间捷径候选表”。
# ============================================================

def write_shortcut_candidate_table():
    source_path = (
        OUT
        / "source_summary.csv"
    )

    future_path = (
        OUT
        / "future_summary.csv"
    )

    if (
        not source_path.exists()
        or not future_path.exists()
    ):
        raise RuntimeError(
            "没有找到source/future summary"
        )

    source = pd.read_csv(
        source_path
    )

    future = pd.read_csv(
        future_path
    )

    combined = source.merge(
        future,
        on=[
            "config",
            "base",
            "mask",
            "drop_k_roles",
            "drop_roles",
        ],
        how="inner",
        validate="one_to_one",
    )

    baseline_rows = combined[
        combined[
            "drop_k_roles"
        ] == 0
    ]

    if len(baseline_rows) != 1:
        raise RuntimeError(
            "baseline数量异常"
        )

    baseline = (
        baseline_rows.iloc[
            0
        ]
    )

    rows = []

    for _, row in combined[
        combined[
            "drop_k_roles"
        ] == 1
    ].iterrows():

        delta_2018 = (
            float(
                row[
                    "BA_2018"
                ]
            )
            - float(
                baseline[
                    "BA_2018"
                ]
            )
        )

        delta_2019 = (
            float(
                row[
                    "BA_2019"
                ]
            )
            - float(
                baseline[
                    "BA_2019"
                ]
            )
        )

        delta_2020 = (
            float(
                row[
                    "BA_2020"
                ]
            )
            - float(
                baseline[
                    "BA_2020"
                ]
            )
        )

        delta_2021 = (
            float(
                row[
                    "BA_2021"
                ]
            )
            - float(
                baseline[
                    "BA_2021"
                ]
            )
        )

        year_deltas = np.asarray(
            [
                delta_2018,
                delta_2019,
                delta_2020,
                delta_2021,
            ],
            dtype=float,
        )

        source_delta = (
            float(
                row[
                    "source_mean_BA"
                ]
            )
            - float(
                baseline[
                    "source_mean_BA"
                ]
            )
        )

        source_min_delta = (
            float(
                row[
                    "source_min_BA"
                ]
            )
            - float(
                baseline[
                    "source_min_BA"
                ]
            )
        )

        future_delta = (
            float(
                row[
                    "future_mean_BA"
                ]
            )
            - float(
                baseline[
                    "future_mean_BA"
                ]
            )
        )

        future_min_delta = (
            float(
                row[
                    "future_min_BA"
                ]
            )
            - float(
                baseline[
                    "future_min_BA"
                ]
            )
        )

        future_auc_delta = (
            float(
                row[
                    "future_mean_AUC"
                ]
            )
            - float(
                baseline[
                    "future_mean_AUC"
                ]
            )
        )

        years_improved = int(
            (
                year_deltas
                > 1e-12
            ).sum()
        )

        years_worse = int(
            (
                year_deltas
                < -1e-12
            ).sum()
        )

        # ====================================================
        # 这里只定义“开发阶段强捷径候选”，
        # 不是看到分数上涨就直接宣布因果捷径。
        #
        # 条件提前固定：
        #
        # 2017 mean BA最多损失0.30pp
        # future mean BA至少提升0.50pp
        # 至少3/4未来年份提升
        # future最低年份不能下降
        # ====================================================

        strong_candidate = int(
            source_delta
            >= -0.0030
            and future_delta
            >= 0.0050
            and years_improved
            >= 3
            and future_min_delta
            >= -1e-12
        )

        rows.append({
            "role":
                row[
                    "drop_roles"
                ],

            "dim":
                int(
                    row[
                        "dim_min"
                    ]
                ),

            "source_mean_BA":
                float(
                    row[
                        "source_mean_BA"
                    ]
                ),

            "source_delta_pp":
                100.0
                * source_delta,

            "source_min_BA":
                float(
                    row[
                        "source_min_BA"
                    ]
                ),

            "source_min_delta_pp":
                100.0
                * source_min_delta,

            "future_mean_BA":
                float(
                    row[
                        "future_mean_BA"
                    ]
                ),

            "future_delta_pp":
                100.0
                * future_delta,

            "future_min_BA":
                float(
                    row[
                        "future_min_BA"
                    ]
                ),

            "future_min_delta_pp":
                100.0
                * future_min_delta,

            "future_mean_AUC":
                float(
                    row[
                        "future_mean_AUC"
                    ]
                ),

            "future_AUC_delta_pp":
                100.0
                * future_auc_delta,

            "delta_2018_pp":
                100.0
                * delta_2018,

            "delta_2019_pp":
                100.0
                * delta_2019,

            "delta_2020_pp":
                100.0
                * delta_2020,

            "delta_2021_pp":
                100.0
                * delta_2021,

            "years_improved":
                years_improved,

            "years_worse":
                years_worse,

            "strong_shortcut_candidate":
                strong_candidate,
        })

    result = (
        pd.DataFrame(
            rows
        )
        .sort_values(
            [
                "strong_shortcut_candidate",
                "future_delta_pp",
                "future_min_delta_pp",
                "future_AUC_delta_pp",
            ],
            ascending=False,
        )
    )

    result.to_csv(
        OUT
        / "shortcut_candidate_ranking.csv",
        index=False,
    )

    print()
    print(
        "=" * 170
    )
    print(
        "Stable70剩余角色：单删除时间捷径候选排名"
    )
    print(
        "=" * 170
    )

    print(
        result.to_string(
            index=False,
            float_format=lambda x: f"{x:.6f}",
        )
    )

    strong = result[
        result[
            "strong_shortcut_candidate"
        ] == 1
    ]

    print()
    print(
        "=" * 170
    )
    print(
        "强时间捷径候选"
    )
    print(
        "=" * 170
    )

    if len(strong) == 0:
        print(
            "没有角色满足预先固定的强候选条件"
        )
    else:
        print(
            strong.to_string(
                index=False,
                float_format=lambda x: f"{x:.6f}",
            )
        )

    print()
    print(
        "注意：这里是开发阶段候选，不等同于已经证明因果捷径。"
    )


# ============================================================
# Monkey patch旧脚本。
# ============================================================

old.OUT = OUT

old.BASE_VARIANTS = (
    "stable70",
)

old.build_base_variants = (
    build_base_variants_stable70
)

old.run_one_population = (
    run_one_population_single_drop
)

old.compute_marginal_effects = (
    compute_single_drop_effects
)


if __name__ == "__main__":
    print(
        "============================================================"
    )
    print(
        "Stable70 remaining role shortcut sweep"
    )
    print(
        "baseline + each pairable role node/edge deletion"
    )
    print(
        "no residual"
    )
    print(
        "no multi-role combinations"
    )
    print(
        "============================================================"
    )

    old.main()

    write_shortcut_candidate_table()
