#!/usr/bin/env python3

from __future__ import annotations

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
    "compact-ratio3-presence-neutral-eval"
)


# ============================================================
# 已确认“是否出现”发生时间反转的四个角色。
# ============================================================

UNSTABLE_RAW = {
    "role_node__command_execution",
    "role_edge__command_execution",

    "role_node__task_or_alarm_scheduling",
    "role_edge__task_or_alarm_scheduling",

    "role_node__icon_or_component_hiding",
    "role_edge__icon_or_component_hiding",

    "role_node__device_identifier_access",
    "role_edge__device_identifier_access",
}


# ============================================================
# 条件强度候选。
#
# 第一组：
# device_identifier 的节点/边。
#
# 第二组额外增加：
# task_alarm 节点
# icon_hiding 节点
# ============================================================

CONDITIONAL_ID2 = (
    "role_node__device_identifier_access",
    "role_edge__device_identifier_access",
)


CONDITIONAL_EXTRA2 = (
    "role_node__task_or_alarm_scheduling",
    "role_node__icon_or_component_hiding",
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


def raw_values(
    rows,
    feature_by_sha,
    name,
):
    values = []

    for row in rows:
        sha = norm_sha(
            row["sha256"]
        )

        values.append(
            safe_float(
                feature_by_sha[
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


def conditional_intensity(
    train_rows,
    test_rows,
    feature_by_sha,
    name,
):
    """
    只在“特征已出现”的训练样本上计算均值/标准差。

    x == 0:
        输出0

    x > 0:
        (log1p(x) - present_mean)
        / present_std

    因此：
        “完全没出现”
        和
        “典型出现强度”
    都接近0。

    目的是削弱出现率本身，
    保留出现后的强度差异。
    """

    train_raw = raw_values(
        train_rows,
        feature_by_sha,
        name,
    )

    test_raw = raw_values(
        test_rows,
        feature_by_sha,
        name,
    )

    present = (
        train_raw > 0.0
    )

    if present.sum() < 5:
        raise RuntimeError(
            f"{name}: "
            f"训练折出现样本过少="
            f"{present.sum()}"
        )

    train_log = np.log1p(
        train_raw
    )

    test_log = np.log1p(
        test_raw
    )

    mean = float(
        np.mean(
            train_log[
                present
            ]
        )
    )

    std = float(
        np.std(
            train_log[
                present
            ]
        )
    )

    if (
        not np.isfinite(std)
        or std < 1e-12
    ):
        raise RuntimeError(
            f"{name}: "
            "出现样本条件强度为常数"
        )

    train_out = np.zeros(
        len(train_raw),
        dtype=float,
    )

    test_out = np.zeros(
        len(test_raw),
        dtype=float,
    )

    train_out[
        train_raw > 0.0
    ] = (
        train_log[
            train_raw > 0.0
        ]
        - mean
    ) / std

    test_out[
        test_raw > 0.0
    ] = (
        test_log[
            test_raw > 0.0
        ]
        - mean
    ) / std

    return (
        train_out,
        test_out,
        (
            "conditional_intensity__"
            + name
        ),
    )


def build_variants(
    train_rows,
    test_rows,
    feature_by_sha,
    schema,
):
    # --------------------------------------------------------
    # 先得到已经验证过的 Ratio3 替换 Compact。
    # --------------------------------------------------------

    configs = r3.build_configs(
        train_rows,
        test_rows,
        feature_by_sha,
        schema,
    )

    (
        base_train,
        base_test,
        base_names,
    ) = configs[
        "compact_replace_ratio3"
    ]

    base_names = list(
        base_names
    )

    # --------------------------------------------------------
    # A. Ratio3基线
    # --------------------------------------------------------

    result = {
        "ratio3":
            (
                base_train,
                base_test,
                base_names,
            )
    }

    # --------------------------------------------------------
    # 删除四个反转角色的8个原始计数。
    # --------------------------------------------------------

    found = {
        name
        for name in base_names
        if name in UNSTABLE_RAW
    }

    if found != UNSTABLE_RAW:
        missing = (
            UNSTABLE_RAW
            - found
        )

        raise RuntimeError(
            "没有找到全部8个待删除特征: "
            + str(
                sorted(missing)
            )
        )

    keep_idx = [
        i
        for i, name in enumerate(
            base_names
        )
        if name not in UNSTABLE_RAW
    ]

    drop_train = (
        base_train[
            :,
            keep_idx,
        ]
    )

    drop_test = (
        base_test[
            :,
            keep_idx,
        ]
    )

    drop_names = [
        base_names[i]
        for i in keep_idx
    ]

    result[
        "ratio3_drop8"
    ] = (
        drop_train,
        drop_test,
        drop_names,
    )

    # --------------------------------------------------------
    # C. 加回 device_identifier
    #    节点/边条件强度。
    # --------------------------------------------------------

    id_train = []
    id_test = []
    id_names = []

    for name in CONDITIONAL_ID2:
        tr, te, new_name = (
            conditional_intensity(
                train_rows,
                test_rows,
                feature_by_sha,
                name,
            )
        )

        id_train.append(tr)
        id_test.append(te)
        id_names.append(new_name)

    id_train = np.column_stack(
        id_train
    )

    id_test = np.column_stack(
        id_test
    )

    result[
        "ratio3_drop8_conditional_id2"
    ] = (
        np.concatenate(
            [
                drop_train,
                id_train,
            ],
            axis=1,
        ),

        np.concatenate(
            [
                drop_test,
                id_test,
            ],
            axis=1,
        ),

        drop_names
        + id_names,
    )

    # --------------------------------------------------------
    # D. 再加：
    #    task_alarm节点条件强度
    #    icon_hiding节点条件强度
    # --------------------------------------------------------

    extra_train = []
    extra_test = []
    extra_names = []

    for name in CONDITIONAL_EXTRA2:
        tr, te, new_name = (
            conditional_intensity(
                train_rows,
                test_rows,
                feature_by_sha,
                name,
            )
        )

        extra_train.append(tr)
        extra_test.append(te)
        extra_names.append(new_name)

    extra_train = np.column_stack(
        extra_train
    )

    extra_test = np.column_stack(
        extra_test
    )

    result[
        "ratio3_drop8_conditional_id2_extra2"
    ] = (
        np.concatenate(
            [
                drop_train,
                id_train,
                extra_train,
            ],
            axis=1,
        ),

        np.concatenate(
            [
                drop_test,
                id_test,
                extra_test,
            ],
            axis=1,
        ),

        drop_names
        + id_names
        + extra_names,
    )

    return result


def labels(
    rows,
    feature_by_sha,
):
    return base.labels_for(
        rows,
        feature_by_sha,
    ).astype(int)


def run_eval(
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

    variants = build_variants(
        train_rows,
        test_rows,
        feature_by_sha,
        schema,
    )

    rows = []

    for config_name, (
        x_train,
        x_test,
        names,
    ) in variants.items():

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
                    config_name,

                "seed":
                    seed,

                "dim":
                    x_train.shape[1],

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
    # 2017滚动
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
            and parts["train"]
            and parts["test"]
        )
    }

    source_rows = []

    for split_name in sorted(
        rolling
    ):
        rows = run_eval(
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

    source_summary = []

    for config, g in source_df.groupby(
        "config"
    ):
        split_ba = (
            g.groupby(
                "split_name"
            )[
                "BA"
            ]
            .mean()
        )

        source_summary.append({
            "config":
                config,

            "dim":
                int(
                    g[
                        "dim"
                    ].iloc[0]
                ),

            "mean_BA":
                float(
                    split_ba.mean()
                ),

            "min_BA":
                float(
                    split_ba.min()
                ),

            "mean_AUC":
                float(
                    g[
                        "AUC"
                    ].mean()
                ),

            "mean_ACC":
                float(
                    g[
                        "ACC"
                    ].mean()
                ),
        })

    source_summary_df = (
        pd.DataFrame(
            source_summary
        )
        .sort_values(
            [
                "mean_BA",
                "min_BA",
                "mean_AUC",
            ],
            ascending=False,
        )
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
        split_name = (
            f"train_fixed_2017__test_{year}"
        )

        test_rows = (
            future_parts[
                split_name
            ][
                "test"
            ]
        )

        rows = run_eval(
            train_rows,
            test_rows,
            feature_by_sha,
            schema,
        )

        for row in rows:
            row["year"] = year

        future_rows.extend(
            rows
        )

    future_seed_df = pd.DataFrame(
        future_rows
    )

    # 每年每配置：
    # 先对5个seed取平均。
    future_year_rows = []

    for (
        year,
        config,
    ), g in future_seed_df.groupby(
        [
            "year",
            "config",
        ]
    ):
        future_year_rows.append({
            "year":
                int(year),

            "config":
                config,

            "dim":
                int(
                    g[
                        "dim"
                    ].iloc[0]
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

    future_year_df = pd.DataFrame(
        future_year_rows
    )

    future_summary = []

    for config, g in future_year_df.groupby(
        "config"
    ):
        future_summary.append({
            "config":
                config,

            "dim":
                int(
                    g[
                        "dim"
                    ].iloc[0]
                ),

            "mean_BA":
                float(
                    g[
                        "BA"
                    ].mean()
                ),

            "min_BA":
                float(
                    g[
                        "BA"
                    ].min()
                ),

            "mean_AUC":
                float(
                    g[
                        "AUC"
                    ].mean()
                ),

            "mean_ACC":
                float(
                    g[
                        "ACC"
                    ].mean()
                ),
        })

    future_summary_df = (
        pd.DataFrame(
            future_summary
        )
        .sort_values(
            [
                "mean_BA",
                "min_BA",
                "mean_AUC",
            ],
            ascending=False,
        )
    )

    source_summary_df.to_csv(
        OUT
        / "source_summary.csv",
        index=False,
    )

    future_seed_df.to_csv(
        OUT
        / "future_per_seed.csv",
        index=False,
    )

    future_year_df.to_csv(
        OUT
        / "future_by_year.csv",
        index=False,
    )

    future_summary_df.to_csv(
        OUT
        / "future_summary.csv",
        index=False,
    )

    print()
    print("=" * 100)
    print("2017滚动验证")
    print("=" * 100)

    print(
        source_summary_df.to_string(
            index=False
        )
    )

    print()
    print("=" * 100)
    print("2017 -> 2018-2021逐年")
    print("=" * 100)

    print(
        future_year_df
        .sort_values(
            [
                "year",
                "config",
            ]
        )
        .to_string(
            index=False
        )
    )

    print()
    print("=" * 100)
    print("未来四年汇总")
    print("=" * 100)

    print(
        future_summary_df.to_string(
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
