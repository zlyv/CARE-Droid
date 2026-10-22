#!/usr/bin/env python3

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(
    "artifacts/care-di25-host-m0-future-replay-v1/"
    "mechanism_diagnosis"
)

TRIPLE_FILE = (
    ROOT
    / "m0_graphglobal_triples"
    / "per_seed_block_replacement.csv"
)

PAIR_FILE = (
    ROOT
    / "m0_graphglobal_pairs"
    / "per_seed_block_replacement.csv"
)

GROUP_FILE = (
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


triple = pd.read_csv(TRIPLE_FILE)
pair = pd.read_csv(PAIR_FILE)
group = pd.read_csv(GROUP_FILE)
overlap = pd.read_csv(OVERLAP)

for d in (
    triple,
    pair,
    group,
    overlap,
):
    d["sha256"] = norm(d["sha256"])


# ---------------------------------------------------------
# FULL27 reproduction gate
# ---------------------------------------------------------

m = triple.merge(
    group[
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
        "_triple",
        "_group",
    ),
    validate="one_to_one",
)

p_diff = float(
    np.max(
        np.abs(
            m[
                "cf_global_full27_probability_triple"
            ]
            -
            m[
                "cf_global_full27_probability_group"
            ]
        )
    )
)

d_diff = float(
    np.max(
        np.abs(
            m[
                "cf_global_full27_bounded_delta_triple"
            ]
            -
            m[
                "cf_global_full27_bounded_delta_group"
            ]
        )
    )
)

print()
print("=" * 100)
print("FULL27复现门禁")
print("=" * 100)

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
    "TRIPLE_FULL27_REPRODUCTION_PASS"
)


# ---------------------------------------------------------
# 2020 routed FN groups
# ---------------------------------------------------------

fn = overlap[
    (overlap["year"] == 2020)
    &
    (overlap["error_type"] == "FN")
].copy()

if len(fn) != 14:
    raise RuntimeError(
        "expected 14 routed 2020 FN"
    )

fn["group"] = fn.apply(
    classify,
    axis=1,
)


def attach_group(d):
    return d.merge(
        fn[
            [
                "sha256",
                "group",
            ]
        ],
        on="sha256",
        how="inner",
        validate="many_to_one",
    )


triple = attach_group(triple)
pair = attach_group(pair)
group = attach_group(group)


def effect(
    d,
    target_group,
    arm,
):
    g = d[
        d["group"]
        == target_group
    ]

    return float(
        (
            g[
                f"cf_{arm}_bounded_delta"
            ]
            -
            g[
                "original_bounded_delta"
            ]
        ).mean()
    )


def detailed(
    target_group,
):
    tg = triple[
        triple["group"]
        == target_group
    ].copy()

    pg = pair[
        pair["group"]
        == target_group
    ].copy()

    gg = group[
        group["group"]
        == target_group
    ].copy()

    pair_effect = effect(
        pair,
        target_group,
        "order_topology",
    )

    full_effect = effect(
        group,
        target_group,
        "global_full27",
    )

    rows = []

    configs = [
        (
            "ORDER+TOPOLOGY",
            pg,
            "order_topology",
        ),
        (
            "ORDER+TOPOLOGY+SCALE",
            tg,
            "order_topology_scale",
        ),
        (
            "ORDER+TOPOLOGY+SPECTRAL",
            tg,
            "order_topology_spectral",
        ),
        (
            "FULL27",
            tg,
            "global_full27",
        ),
    ]

    for name, data, arm in configs:

        dcol = (
            f"cf_{arm}_bounded_delta"
        )

        pcol = (
            f"cf_{arm}_probability"
        )

        delta_gain = (
            data[dcol]
            -
            data[
                "original_bounded_delta"
            ]
        )

        effect_value = float(
            delta_gain.mean()
        )

        sample_gain = (
            data.assign(
                gain=delta_gain
            )
            .groupby("sha256")[
                "gain"
            ]
            .mean()
        )

        seed_positive = (
            data.assign(
                positive=(
                    data[dcol] > 0
                ).astype(int)
            )
            .groupby("sha256")[
                "positive"
            ]
            .sum()
        )

        original_p = (
            data.groupby("sha256")[
                "original_probability"
            ]
            .mean()
        )

        cf_p = (
            data.groupby("sha256")[
                pcol
            ]
            .mean()
        )

        rows.append({
            "配置":
                name,

            "样本数":
                int(
                    data["sha256"]
                    .nunique()
                ),

            "平均修正改善":
                effect_value,

            "相对OrderTopology新增改善":
                effect_value
                -
                pair_effect,

            "距离Full27剩余差距":
                full_effect
                -
                effect_value,

            "Full27效果覆盖率":
                (
                    effect_value
                    /
                    full_effect
                    if abs(full_effect) > 1e-12
                    else np.nan
                ),

            "替换后平均修正":
                float(
                    data[dcol]
                    .mean()
                ),

            "改善样本数":
                int(
                    np.sum(
                        sample_gain > 0
                    )
                ),

            "至少2个seed正向样本数":
                int(
                    np.sum(
                        seed_positive >= 2
                    )
                ),

            "最终修复数":
                int(
                    np.sum(
                        cf_p >= 0.5
                    )
                ),

            "平均恶意概率":
                float(
                    cf_p.mean()
                ),

            "平均概率改善":
                float(
                    (
                        cf_p
                        -
                        original_p
                    ).mean()
                ),
        })

    return pd.DataFrame(rows)


persistent = detailed(
    "PERSISTENT"
)

control = detailed(
    "BOTH_FIXED"
)


print()
print("=" * 150)
print(
    "8个持续错误FN："
    "Order+Topology基础上的第三组贡献"
)
print("=" * 150)

print(
    persistent.to_string(
        index=False,
        float_format=lambda x:
            f"{x:.4f}",
    )
)


print()
print("=" * 150)
print(
    "3个原本正确修复FN：控制组"
)
print("=" * 150)

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
    / "m0_graphglobal_triples"
)

out.mkdir(
    parents=True,
    exist_ok=True,
)

persistent.to_csv(
    out
    / "persistent_triples.csv",
    index=False,
)

control.to_csv(
    out
    / "control_triples.csv",
    index=False,
)

print()
print(
    "saved:",
    out,
)
