#!/usr/bin/env python3

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))


# 复用刚刚已经跑通的 Stable70 扫描框架。
import run_stable70_remaining_role_shortcut_sweep as parent


old = parent.old
r3 = parent.r3


OUT = Path(
    "artifacts/"
    "stable70-crypto-component-sweep"
)


NODE = "role_node__cryptography"
EDGE = "role_edge__cryptography"


def norm_sha(value):
    return (
        str(value)
        .strip()
        .casefold()
    )


def get_raw_row(
    feature_by_sha,
    row,
):
    sha = norm_sha(
        row["sha256"]
    )

    if sha in feature_by_sha:
        return feature_by_sha[
            sha
        ]

    raw_sha = str(
        row["sha256"]
    )

    if raw_sha in feature_by_sha:
        return feature_by_sha[
            raw_sha
        ]

    raise KeyError(
        "找不到SHA: "
        + sha
    )


def safe_float(value):
    try:
        value = float(
            value
        )

        if np.isfinite(
            value
        ):
            return value

    except Exception:
        pass

    return 0.0


def raw_presence(
    rows,
    feature_by_sha,
    feature_name,
):
    values = []

    for row in rows:
        source = get_raw_row(
            feature_by_sha,
            row,
        )

        value = safe_float(
            source.get(
                feature_name,
                0.0,
            )
        )

        values.append(
            float(
                value > 0.0
            )
        )

    return np.asarray(
        values,
        dtype=float,
    )


def standardize_presence(
    train_rows,
    test_rows,
    feature_by_sha,
    feature_name,
):
    train = raw_presence(
        train_rows,
        feature_by_sha,
        feature_name,
    )

    test = raw_presence(
        test_rows,
        feature_by_sha,
        feature_name,
    )

    mean = float(
        train.mean()
    )

    std = float(
        train.std()
    )

    if (
        not np.isfinite(std)
        or std < 1e-12
    ):
        # 如果训练折为常数，
        # 该presence在这一折没有信息。
        return (
            np.zeros_like(
                train
            ),
            np.zeros_like(
                test
            ),
        )

    return (
        (
            train
            - mean
        )
        / std,

        (
            test
            - mean
        )
        / std,
    )


def remove_columns(
    x_train,
    x_test,
    names,
    drop_names,
):
    drop_names = set(
        drop_names
    )

    keep = [
        i
        for i, name in enumerate(
            names
        )
        if name not in drop_names
    ]

    return (
        x_train[
            :,
            keep,
        ].copy(),

        x_test[
            :,
            keep,
        ].copy(),

        [
            names[i]
            for i in keep
        ],
    )


def make_variants(
    train_rows,
    test_rows,
    feature_by_sha,
    schema,
):
    bases = (
        parent.build_base_variants_stable70(
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

    base_train = np.asarray(
        base_train,
        dtype=float,
    )

    base_test = np.asarray(
        base_test,
        dtype=float,
    )

    base_names = list(
        base_names
    )

    for name in (
        NODE,
        EDGE,
    ):
        if name not in base_names:
            raise RuntimeError(
                "Stable70找不到: "
                + name
            )

    node_idx = base_names.index(
        NODE
    )

    edge_idx = base_names.index(
        EDGE
    )

    node_train_p, node_test_p = (
        standardize_presence(
            train_rows,
            test_rows,
            feature_by_sha,
            NODE,
        )
    )

    edge_train_p, edge_test_p = (
        standardize_presence(
            train_rows,
            test_rows,
            feature_by_sha,
            EDGE,
        )
    )

    variants = {}

    # ========================================================
    # Stable70 baseline
    # ========================================================

    variants[
        "stable70"
    ] = (
        base_train.copy(),
        base_test.copy(),
        list(
            base_names
        ),
        0,
    )

    # ========================================================
    # node only drop
    # ========================================================

    a, b, n = remove_columns(
        base_train,
        base_test,
        base_names,
        {
            NODE,
        },
    )

    variants[
        "drop_crypto_node"
    ] = (
        a,
        b,
        n,
        1,
    )

    # ========================================================
    # edge only drop
    # ========================================================

    a, b, n = remove_columns(
        base_train,
        base_test,
        base_names,
        {
            EDGE,
        },
    )

    variants[
        "drop_crypto_edge"
    ] = (
        a,
        b,
        n,
        1,
    )

    # ========================================================
    # node + edge drop
    # ========================================================

    a, b, n = remove_columns(
        base_train,
        base_test,
        base_names,
        {
            NODE,
            EDGE,
        },
    )

    variants[
        "drop_crypto_both"
    ] = (
        a,
        b,
        n,
        2,
    )

    # ========================================================
    # node count -> node presence
    # ========================================================

    a = base_train.copy()
    b = base_test.copy()

    a[
        :,
        node_idx,
    ] = node_train_p

    b[
        :,
        node_idx,
    ] = node_test_p

    variants[
        "presence_crypto_node"
    ] = (
        a,
        b,
        list(
            base_names
        ),
        0,
    )

    # ========================================================
    # edge count -> edge presence
    # ========================================================

    a = base_train.copy()
    b = base_test.copy()

    a[
        :,
        edge_idx,
    ] = edge_train_p

    b[
        :,
        edge_idx,
    ] = edge_test_p

    variants[
        "presence_crypto_edge"
    ] = (
        a,
        b,
        list(
            base_names
        ),
        0,
    )

    # ========================================================
    # node + edge count -> presence
    # ========================================================

    a = base_train.copy()
    b = base_test.copy()

    a[
        :,
        node_idx,
    ] = node_train_p

    b[
        :,
        node_idx,
    ] = node_test_p

    a[
        :,
        edge_idx,
    ] = edge_train_p

    b[
        :,
        edge_idx,
    ] = edge_test_p

    variants[
        "presence_crypto_both"
    ] = (
        a,
        b,
        list(
            base_names
        ),
        0,
    )

    return variants


def run_one_population(
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

    variants = make_variants(
        train_rows,
        test_rows,
        feature_by_sha,
        schema,
    )

    rows = []

    print()
    print(
        "======================================"
    )

    print(
        "cryptography component audit"
    )

    print(
        "======================================"
    )

    for cfg, (
        x_train,
        x_test,
        names,
        dropped_n,
    ) in variants.items():

        print(
            cfg,
            "dim=",
            x_train.shape[1],
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

                # 保持旧汇总接口兼容。
                "mask":
                    list(
                        variants
                    ).index(
                        cfg
                    ),

                "drop_k_roles":
                    (
                        0
                        if cfg == "stable70"
                        else 1
                    ),

                "drop_roles":
                    (
                        ""
                        if cfg == "stable70"
                        else cfg
                    ),

                "actually_dropped_features":
                    dropped_n,

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


def simple_effects(
    combined,
):
    # old.main最后会调用这个函数；
    # 这里返回一个简单兼容表即可。

    base_row = combined[
        combined[
            "config"
        ] == "stable70"
    ].iloc[
        0
    ]

    rows = []

    for _, row in combined.iterrows():

        if row[
            "config"
        ] == "stable70":
            continue

        rows.append({
            "base":
                "stable70",

            "role":
                row[
                    "config"
                ],

            "pairs":
                1,

            "future_BA_mean_delta":
                float(
                    row[
                        "future_mean_BA"
                    ]
                    - base_row[
                        "future_mean_BA"
                    ]
                ),

            "future_BA_median_delta":
                float(
                    row[
                        "future_mean_BA"
                    ]
                    - base_row[
                        "future_mean_BA"
                    ]
                ),

            "future_BA_wins":
                int(
                    row[
                        "future_mean_BA"
                    ]
                    > base_row[
                        "future_mean_BA"
                    ]
                ),

            "future_BA_ties":
                0,

            "future_BA_losses":
                int(
                    row[
                        "future_mean_BA"
                    ]
                    < base_row[
                        "future_mean_BA"
                    ]
                ),

            "future_min_BA_mean_delta":
                float(
                    row[
                        "future_min_BA"
                    ]
                    - base_row[
                        "future_min_BA"
                    ]
                ),

            "future_AUC_mean_delta":
                float(
                    row[
                        "future_mean_AUC"
                    ]
                    - base_row[
                        "future_mean_AUC"
                    ]
                ),

            "source_BA_mean_delta":
                float(
                    row[
                        "source_mean_BA"
                    ]
                    - base_row[
                        "source_mean_BA"
                    ]
                ),

            "source_min_BA_mean_delta":
                float(
                    row[
                        "source_min_BA"
                    ]
                    - base_row[
                        "source_min_BA"
                    ]
                ),

            "BA_2018_mean_delta":
                float(
                    row[
                        "BA_2018"
                    ]
                    - base_row[
                        "BA_2018"
                    ]
                ),

            "BA_2019_mean_delta":
                float(
                    row[
                        "BA_2019"
                    ]
                    - base_row[
                        "BA_2019"
                    ]
                ),

            "BA_2020_mean_delta":
                float(
                    row[
                        "BA_2020"
                    ]
                    - base_row[
                        "BA_2020"
                    ]
                ),

            "BA_2021_mean_delta":
                float(
                    row[
                        "BA_2021"
                    ]
                    - base_row[
                        "BA_2021"
                    ]
                ),
        })

    return pd.DataFrame(
        rows
    )


def print_final_table():
    s = pd.read_csv(
        OUT
        / "source_summary.csv"
    )

    f = pd.read_csv(
        OUT
        / "future_summary.csv"
    )

    x = s.merge(
        f,
        on=[
            "config",
            "base",
            "mask",
            "drop_k_roles",
            "drop_roles",
        ],
        how="inner",
    )

    baseline = x[
        x[
            "config"
        ] == "stable70"
    ].iloc[
        0
    ]

    rows = []

    for _, row in x.iterrows():
        rows.append({
            "config":
                row[
                    "config"
                ],

            "dim":
                int(
                    row[
                        "dim_min"
                    ]
                ),

            "source_BA":
                float(
                    row[
                        "source_mean_BA"
                    ]
                ),

            "source_delta_pp":
                100.0
                * (
                    row[
                        "source_mean_BA"
                    ]
                    - baseline[
                        "source_mean_BA"
                    ]
                ),

            "source_min":
                float(
                    row[
                        "source_min_BA"
                    ]
                ),

            "future_BA":
                float(
                    row[
                        "future_mean_BA"
                    ]
                ),

            "future_delta_pp":
                100.0
                * (
                    row[
                        "future_mean_BA"
                    ]
                    - baseline[
                        "future_mean_BA"
                    ]
                ),

            "future_min":
                float(
                    row[
                        "future_min_BA"
                    ]
                ),

            "future_AUC":
                float(
                    row[
                        "future_mean_AUC"
                    ]
                ),

            "BA_2018":
                float(
                    row[
                        "BA_2018"
                    ]
                ),

            "BA_2019":
                float(
                    row[
                        "BA_2019"
                    ]
                ),

            "BA_2020":
                float(
                    row[
                        "BA_2020"
                    ]
                ),

            "BA_2021":
                float(
                    row[
                        "BA_2021"
                    ]
                ),
        })

    result = (
        pd.DataFrame(
            rows
        )
        .sort_values(
            [
                "future_BA",
                "source_BA",
            ],
            ascending=False,
        )
    )

    result.to_csv(
        OUT
        / "crypto_component_summary.csv",
        index=False,
    )

    print()
    print(
        "=" * 150
    )
    print(
        "cryptography捷径成分拆解"
    )
    print(
        "=" * 150
    )

    print(
        result.to_string(
            index=False,
            float_format=lambda x: f"{x:.6f}",
        )
    )


old.OUT = OUT

old.run_one_population = (
    run_one_population
)

old.compute_marginal_effects = (
    simple_effects
)


if __name__ == "__main__":
    old.main()
    print_final_table()
