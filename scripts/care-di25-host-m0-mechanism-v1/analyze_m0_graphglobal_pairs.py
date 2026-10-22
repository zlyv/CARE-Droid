#!/usr/bin/env python3

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(
    "artifacts/care-di25-host-m0-future-replay-v1/"
    "mechanism_diagnosis"
)

PAIR_FILE = (
    ROOT
    / "m0_graphglobal_pairs"
    / "per_seed_block_replacement.csv"
)

SINGLE_FILE = (
    ROOT
    / "m0_graphglobal_groups"
    / "per_seed_block_replacement.csv"
)

OVERLAP = (
    ROOT
    / "analysis"
    / "direct_vs_bounded"
    / "paired_overlap.csv"
)


PAIR_ARMS = {
    "order_scale":
        ("order5", "scale_diversity8"),

    "order_spectral":
        ("order5", "spectral8"),

    "order_topology":
        ("order5", "topology6"),

    "scale_spectral":
        ("scale_diversity8", "spectral8"),

    "scale_topology":
        ("scale_diversity8", "topology6"),

    "spectral_topology":
        ("spectral8", "topology6"),
}


CN = {
    "order_scale":
        "组合阶数 + 规模多样性",

    "order_spectral":
        "组合阶数 + 谱结构",

    "order_topology":
        "组合阶数 + 拓扑",

    "scale_spectral":
        "规模多样性 + 谱结构",

    "scale_topology":
        "规模多样性 + 拓扑",

    "spectral_topology":
        "谱结构 + 拓扑",
}


def norm(x):
    return (
        x.astype(str)
        .str.strip()
        .str.lower()
    )


def classify(row):

    if (
        row["bounded_state"]
        == "WRONG_DIRECTION"
        and
        row["direct_state"]
        == "WRONG_DIRECTION"
    ):
        return "PERSISTENT"

    if (
        row["bounded_state"]
        == "FIXED"
        and
        row["direct_state"]
        == "FIXED"
    ):
        return "BOTH_FIXED"

    return "OTHER"


def mean_effect(
    data,
    arm,
):
    return float(
        (
            data[
                f"cf_{arm}_bounded_delta"
            ]
            -
            data[
                "original_bounded_delta"
            ]
        ).mean()
    )


def summarize(
    data,
    target,
):

    g = data[
        data["group"] == target
    ].copy()

    rows = []

    single_effect = {}

    if target == "PERSISTENT":
        for arm in (
            "order5",
            "scale_diversity8",
            "spectral8",
            "topology6",
        ):
            single_effect[arm] = (
                mean_effect(
                    g_single,
                    arm,
                )
            )

    for arm, components in (
        PAIR_ARMS.items()
    ):

        dcol = (
            f"cf_{arm}_bounded_delta"
        )

        pcol = (
            f"cf_{arm}_probability"
        )

        effect = mean_effect(
            g,
            arm,
        )

        op = (
            g.groupby("sha256")[
                "original_probability"
            ].mean()
        )

        cp = (
            g.groupby("sha256")[
                pcol
            ].mean()
        )

        seed_positive = (
            g.assign(
                positive=(
                    g[dcol] > 0
                ).astype(int)
            )
            .groupby("sha256")[
                "positive"
            ]
            .sum()
        )

        per_sample_gain = (
            g.assign(
                gain=(
                    g[dcol]
                    -
                    g[
                        "original_bounded_delta"
                    ]
                )
            )
            .groupby("sha256")[
                "gain"
            ]
            .mean()
        )

        row = {
            "组合":
                CN[arm],

            "样本数":
                int(
                    g["sha256"]
                    .nunique()
                ),

            "平均修正改善":
                effect,

            "替换后平均修正":
                float(
                    g[dcol].mean()
                ),

            "改善样本数":
                int(
                    np.sum(
                        per_sample_gain > 0
                    )
                ),

            "至少2个seed正向样本数":
                int(
                    np.sum(
                        seed_positive >= 2
                    )
                ),

            "替换后最终修复数":
                int(
                    np.sum(
                        cp >= 0.5
                    )
                ),

            "替换后平均恶意概率":
                float(
                    cp.mean()
                ),

            "平均概率改善":
                float(
                    (
                        cp - op
                    ).mean()
                ),
        }

        if target == "PERSISTENT":

            additive = (
                single_effect[
                    components[0]
                ]
                +
                single_effect[
                    components[1]
                ]
            )

            row[
                "两个单组改善之和"
            ] = additive

            row[
                "非加性交互量"
            ] = (
                effect
                -
                additive
            )

        rows.append(
            row
        )

    return (
        pd.DataFrame(rows)
        .sort_values(
            "平均修正改善",
            ascending=False,
        )
    )


pair = pd.read_csv(
    PAIR_FILE
)

single = pd.read_csv(
    SINGLE_FILE
)

overlap = pd.read_csv(
    OVERLAP
)

for d in (
    pair,
    single,
    overlap,
):
    d["sha256"] = norm(
        d["sha256"]
    )


# Positive-control gate.
merged = pair.merge(
    single[
        [
            "sha256",
            "year",
            "seed",
            "cf_global_full27_probability",
            "cf_global_full27_bounded_delta",
        ]
    ],
    on=[
        "sha256",
        "year",
        "seed",
    ],
    suffixes=(
        "_pair",
        "_single",
    ),
)

p_diff = np.max(
    np.abs(
        merged[
            "cf_global_full27_probability_pair"
        ]
        -
        merged[
            "cf_global_full27_probability_single"
        ]
    )
)

d_diff = np.max(
    np.abs(
        merged[
            "cf_global_full27_bounded_delta_pair"
        ]
        -
        merged[
            "cf_global_full27_bounded_delta_single"
        ]
    )
)

print()
print(
    "FULL27复现门禁"
)

print(
    "max probability difference =",
    p_diff,
)

print(
    "max delta difference =",
    d_diff,
)

if (
    p_diff > 1e-7
    or
    d_diff > 1e-7
):
    raise RuntimeError(
        "FULL27 reproduction failed"
    )

print(
    "PAIR_FULL27_REPRODUCTION_PASS"
)


fn = overlap[
    (overlap["year"] == 2020)
    &
    (overlap["error_type"] == "FN")
].copy()

fn["group"] = fn.apply(
    classify,
    axis=1,
)


pair = pair.merge(
    fn[
        [
            "sha256",
            "group",
        ]
    ],
    on="sha256",
    how="inner",
)

g_single = single.merge(
    fn[
        [
            "sha256",
            "group",
        ]
    ],
    on="sha256",
    how="inner",
)

g_single = g_single[
    g_single["group"]
    == "PERSISTENT"
].copy()


persistent = summarize(
    pair,
    "PERSISTENT",
)

control = summarize(
    pair,
    "BOTH_FIXED",
)


print()
print(
    "=" * 130
)

print(
    "8个持续错误FN：GraphGlobal两两组合"
)

print(
    "=" * 130
)

print(
    persistent.to_string(
        index=False,
        float_format=lambda x:
            f"{x:.4f}",
    )
)


print()
print(
    "=" * 130
)

print(
    "3个原本正确修复FN：控制组"
)

print(
    "=" * 130
)

print(
    control.to_string(
        index=False,
        float_format=lambda x:
            f"{x:.4f}",
    )
)


out = (
    ROOT
    / "analysis"
    / "m0_graphglobal_pairs"
)

out.mkdir(
    parents=True,
    exist_ok=True,
)

persistent.to_csv(
    out
    / "persistent_pairs.csv",
    index=False,
)

control.to_csv(
    out
    / "control_pairs.csv",
    index=False,
)

print()
print(
    "saved:",
    out,
)
