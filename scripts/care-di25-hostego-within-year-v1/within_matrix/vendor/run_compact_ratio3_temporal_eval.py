#!/usr/bin/env python3

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import run_frozen_anchor_residual_2017_2021 as base

from run_pilot_compact_eval import fit_logistic


FEATURES = Path(
    "artifacts/p1-features/features.csv"
)

SCHEMA = Path(
    "artifacts/p1-features/feature_schema.json"
)

ROLLING_SPLITS = Path(
    "artifacts/p9-2017-rolling-splits/split_manifest.csv"
)

FUTURE_SPLITS = Path(
    "artifacts/"
    "p1-splits-fixed-2017-test-2018-2021/"
    "split_manifest.csv"
)

OUT = Path(
    "artifacts/compact-ratio3-temporal-eval"
)

YEARS = (2018, 2019, 2020, 2021)

SEEDS = (0, 1, 2, 3, 4)

THRESHOLD = 0.5


RATIO_SPECS = (
    (
        "ratio3__dynamic_code_loading__node",
        "role_node__dynamic_code_loading",
        "sensitive_node_count",
    ),
    (
        "ratio3__native_library_loading__node",
        "role_node__native_library_loading",
        "sensitive_node_count",
    ),
    (
        "ratio3__native_library_loading__edge",
        "role_edge__native_library_loading",
        "sensitive_edge_count",
    ),
)

REPLACED_RAW_NAMES = {
    numerator
    for _, numerator, _ in RATIO_SPECS
}


def norm_sha(x):
    return str(x).strip().casefold()


def safe_float(x):
    try:
        x = float(x)
        if np.isfinite(x):
            return x
    except Exception:
        pass

    return 0.0


def labels(rows, feature_by_sha):
    return base.labels_for(
        rows,
        feature_by_sha,
    ).astype(int)


def ratio_matrix(
    train_rows,
    test_rows,
    feature_by_sha,
):
    train_cols = []
    test_cols = []
    used_names = []

    for (
        ratio_name,
        numerator_name,
        denominator_name,
    ) in RATIO_SPECS:
        train_values = []
        test_values = []

        for rows, target in (
            (train_rows, train_values),
            (test_rows, test_values),
        ):
            for row in rows:
                sha = norm_sha(
                    row["sha256"]
                )

                source = feature_by_sha[
                    sha
                ]

                num = safe_float(
                    source.get(
                        numerator_name,
                        0.0,
                    )
                )

                den = safe_float(
                    source.get(
                        denominator_name,
                        0.0,
                    )
                )

                target.append(
                    num / den
                    if den > 0.0
                    else 0.0
                )

        tr = np.asarray(
            train_values,
            dtype=float,
        )

        te = np.asarray(
            test_values,
            dtype=float,
        )

        mean = float(
            np.mean(tr)
        )

        std = float(
            np.std(tr)
        )

        # 与现有Compact一样：
        # 训练折常数列直接丢弃。
        if (
            not np.isfinite(std)
            or std < 1e-12
        ):
            continue

        train_cols.append(
            (
                tr - mean
            ) / std
        )

        test_cols.append(
            (
                te - mean
            ) / std
        )

        used_names.append(
            ratio_name
        )

    if not train_cols:
        raise RuntimeError(
            "当前折3个比例特征全部为常数"
        )

    return (
        np.column_stack(
            train_cols
        ),
        np.column_stack(
            test_cols
        ),
        used_names,
    )


def build_configs(
    train_rows,
    test_rows,
    feature_by_sha,
    schema,
):
    (
        compact_train,
        compact_test,
        compact_names,
    ) = base.compact_matrix_named(
        train_rows,
        test_rows,
        feature_by_sha,
        schema,
    )

    (
        ratio_train,
        ratio_test,
        ratio_names,
    ) = ratio_matrix(
        train_rows,
        test_rows,
        feature_by_sha,
    )

    result = {}

    # --------------------------------------------------------
    # 1. 原Compact76
    # --------------------------------------------------------

    result[
        "compact76"
    ] = (
        compact_train,
        compact_test,
        list(
            compact_names
        ),
    )

    # --------------------------------------------------------
    # 2. 原Compact + 3个比例
    # --------------------------------------------------------

    result[
        "compact76_plus_ratio3"
    ] = (
        np.concatenate(
            [
                compact_train,
                ratio_train,
            ],
            axis=1,
        ),
        np.concatenate(
            [
                compact_test,
                ratio_test,
            ],
            axis=1,
        ),
        list(
            compact_names
        )
        + list(
            ratio_names
        ),
    )

    # --------------------------------------------------------
    # 3. 删除对应3个绝对特征，再加入3个比例
    # --------------------------------------------------------

    keep_idx = [
        i
        for i, name in enumerate(
            compact_names
        )
        if name not in REPLACED_RAW_NAMES
    ]

    removed = [
        name
        for name in compact_names
        if name in REPLACED_RAW_NAMES
    ]

    if set(
        removed
    ) != REPLACED_RAW_NAMES:
        raise RuntimeError(
            "需要替换的3个原始Compact特征没有全部找到。\n"
            f"found={removed}"
        )

    result[
        "compact_replace_ratio3"
    ] = (
        np.concatenate(
            [
                compact_train[
                    :,
                    keep_idx,
                ],
                ratio_train,
            ],
            axis=1,
        ),
        np.concatenate(
            [
                compact_test[
                    :,
                    keep_idx,
                ],
                ratio_test,
            ],
            axis=1,
        ),
        [
            compact_names[i]
            for i in keep_idx
        ]
        + list(
            ratio_names
        ),
    )

    return result


def evaluate(
    x_train,
    y_train,
    x_test,
    y_test,
    seed,
):
    w, b = fit_logistic(
        x_train,
        y_train,
        seed,
        iterations=6000,
        learning_rate=0.05,
        l2=0.03,
    )

    z = (
        x_test @ w
        + b
    )

    z = np.clip(
        z,
        -60.0,
        60.0,
    )

    p = (
        1.0
        /
        (
            1.0
            + np.exp(-z)
        )
    )

    return base.metrics(
        y_test,
        p,
        threshold=THRESHOLD,
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

    schema = json.loads(
        SCHEMA.read_text(
            encoding="utf-8"
        )
    )

    # ========================================================
    # 2017滚动验证
    # ========================================================

    rolling_parts = base.read_split_rows(
        ROLLING_SPLITS
    )

    rolling = {
        name: parts
        for name, parts in rolling_parts.items()
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
        train_rows = (
            rolling[
                split_name
            ][
                "train"
            ]
        )

        test_rows = (
            rolling[
                split_name
            ][
                "test"
            ]
        )

        y_train = labels(
            train_rows,
            feature_by_sha,
        )

        y_test = labels(
            test_rows,
            feature_by_sha,
        )

        configs = build_configs(
            train_rows,
            test_rows,
            feature_by_sha,
            schema,
        )

        for config_name, (
            x_train,
            x_test,
            names,
        ) in configs.items():
            for seed in SEEDS:
                m = evaluate(
                    x_train,
                    y_train,
                    x_test,
                    y_test,
                    seed,
                )

                source_rows.append({
                    "split_name":
                        split_name,

                    "config":
                        config_name,

                    "seed":
                        seed,

                    "dim":
                        x_train.shape[
                            1
                        ],

                    "BA":
                        m[
                            "balanced_accuracy"
                        ],

                    "AUC":
                        m[
                            "auc"
                        ],

                    "ACC":
                        m[
                            "accuracy"
                        ],

                    "FP":
                        m[
                            "fp"
                        ],

                    "FN":
                        m[
                            "fn"
                        ],
                })

    source_df = pd.DataFrame(
        source_rows
    )

    source_df.to_csv(
        OUT
        / "source_rolling_per_seed.csv",
        index=False,
    )

    source_summary = []

    for config_name, g in source_df.groupby(
        "config"
    ):
        by_split = (
            g.groupby(
                "split_name"
            )[
                "BA"
            ]
            .mean()
        )

        source_summary.append({
            "config":
                config_name,

            "mean_BA":
                float(
                    by_split.mean()
                ),

            "min_BA":
                float(
                    by_split.min()
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

    source_summary_df.to_csv(
        OUT
        / "source_summary.csv",
        index=False,
    )

    # ========================================================
    # 固定2017全部训练 -> 2018-2021
    # ========================================================

    future_parts = base.read_split_rows(
        FUTURE_SPLITS
    )

    fixed_name = (
        "train_fixed_2017__test_2018"
    )

    fixed_train_rows = (
        future_parts[
            fixed_name
        ][
            "train"
        ]
    )

    future_rows = []

    for year in YEARS:
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

        y_train = labels(
            fixed_train_rows,
            feature_by_sha,
        )

        y_test = labels(
            test_rows,
            feature_by_sha,
        )

        configs = build_configs(
            fixed_train_rows,
            test_rows,
            feature_by_sha,
            schema,
        )

        for config_name, (
            x_train,
            x_test,
            names,
        ) in configs.items():
            seed_metrics = []

            for seed in SEEDS:
                seed_metrics.append(
                    evaluate(
                        x_train,
                        y_train,
                        x_test,
                        y_test,
                        seed,
                    )
                )

            future_rows.append({
                "year":
                    year,

                "config":
                    config_name,

                "dim":
                    x_train.shape[
                        1
                    ],

                "BA":
                    float(
                        np.mean([
                            m[
                                "balanced_accuracy"
                            ]
                            for m in seed_metrics
                        ])
                    ),

                "AUC":
                    float(
                        np.mean([
                            m[
                                "auc"
                            ]
                            for m in seed_metrics
                        ])
                    ),

                "ACC":
                    float(
                        np.mean([
                            m[
                                "accuracy"
                            ]
                            for m in seed_metrics
                        ])
                    ),

                "FP":
                    float(
                        np.mean([
                            m[
                                "fp"
                            ]
                            for m in seed_metrics
                        ])
                    ),

                "FN":
                    float(
                        np.mean([
                            m[
                                "fn"
                            ]
                            for m in seed_metrics
                        ])
                    ),
            })

    future_df = pd.DataFrame(
        future_rows
    )

    future_df.to_csv(
        OUT
        / "future_by_year.csv",
        index=False,
    )

    future_summary = []

    for config_name, g in future_df.groupby(
        "config"
    ):
        future_summary.append({
            "config":
                config_name,

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

    future_summary_df.to_csv(
        OUT
        / "future_summary.csv",
        index=False,
    )

    print()
    print(
        "=" * 90
    )
    print(
        "2017滚动验证"
    )
    print(
        "=" * 90
    )

    print(
        source_summary_df.to_string(
            index=False
        )
    )

    print()
    print(
        "=" * 90
    )
    print(
        "2017 -> 2018-2021"
    )
    print(
        "=" * 90
    )

    print(
        future_df.to_string(
            index=False
        )
    )

    print()
    print(
        "=" * 90
    )
    print(
        "未来四年汇总"
    )
    print(
        "=" * 90
    )

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
