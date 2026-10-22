#!/usr/bin/env python3

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(
    "artifacts/care-di25-host-m0-future-replay-v1/"
    "mechanism_diagnosis"
)

NEW = (
    ROOT
    / "m0_graphglobal_groups"
    / "per_seed_block_replacement.csv"
)

OLD = (
    ROOT
    / "m0_rolegraph_upstream"
    / "per_seed_block_replacement.csv"
)

PAIR = (
    ROOT
    / "analysis"
    / "direct_vs_bounded"
    / "paired_overlap.csv"
)


ARMS = [
    "order5",
    "scale_diversity8",
    "spectral8",
    "topology6",
    "global_full27",
]


CN = {
    "order5":
        "组合阶数分布5维",

    "scale_diversity8":
        "规模/多样性/来源8维",

    "spectral8":
        "谱结构8维",

    "topology6":
        "整体拓扑6维",

    "global_full27":
        "完整GraphGlobal27维",
}


def norm_sha(x):
    return (
        x.astype(str)
        .str.strip()
        .str.lower()
    )


def group_name(row):
    if (
        row["bounded_state"]
        == "WRONG_DIRECTION"
        and
        row["direct_state"]
        == "WRONG_DIRECTION"
    ):
        return "PERSISTENT_BOTH_WRONG"

    if (
        row["bounded_state"]
        == "FIXED"
        and
        row["direct_state"]
        == "FIXED"
    ):
        return "BOTH_FIXED"

    return "OTHER"


def reproduction_gate(
    new,
    old,
):
    a = new[
        [
            "sha256",
            "year",
            "seed",
            "cf_global_full27_probability",
            "cf_global_full27_bounded_delta",
        ]
    ].copy()

    b = old[
        [
            "sha256",
            "year",
            "seed",
            "cf_graph_global_probability",
            "cf_graph_global_bounded_delta",
        ]
    ].copy()

    m = a.merge(
        b,
        on=[
            "sha256",
            "year",
            "seed",
        ],
        how="inner",
        validate="one_to_one",
    )

    if len(m) != len(new):
        raise RuntimeError(
            "GLOBAL_FULL27 alignment failed"
        )

    p_diff = np.max(
        np.abs(
            m[
                "cf_global_full27_probability"
            ].to_numpy(float)
            -
            m[
                "cf_graph_global_probability"
            ].to_numpy(float)
        )
    )

    d_diff = np.max(
        np.abs(
            m[
                "cf_global_full27_bounded_delta"
            ].to_numpy(float)
            -
            m[
                "cf_graph_global_bounded_delta"
            ].to_numpy(float)
        )
    )

    print()
    print(
        "=" * 100
    )
    print(
        "完整27维复现门禁"
    )
    print(
        "=" * 100
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
            "GLOBAL_FULL27_REPRODUCTION_FAILED"
        )

    print(
        "GLOBAL_FULL27_REPRODUCTION_PASS"
    )


def summarize(
    data,
    target_group,
):
    g = data[
        data["group"]
        == target_group
    ].copy()

    rows = []

    for arm in ARMS:

        dcol = (
            f"cf_{arm}_bounded_delta"
        )

        pcol = (
            f"cf_{arm}_probability"
        )

        original_delta = (
            g[
                "original_bounded_delta"
            ]
            .to_numpy(float)
        )

        cf_delta = (
            g[dcol]
            .to_numpy(float)
        )

        delta_gain = (
            cf_delta
            -
            original_delta
        )

        original_p = (
            g.groupby("sha256")[
                "original_probability"
            ]
            .mean()
        )

        cf_p = (
            g.groupby("sha256")[
                pcol
            ]
            .mean()
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

        seed_positive = (
            g.assign(
                pos=(
                    g[dcol] > 0
                ).astype(int)
            )
            .groupby("sha256")[
                "pos"
            ]
            .sum()
        )

        rows.append({
            "实验组":
                CN[arm],

            "样本数":
                int(
                    g[
                        "sha256"
                    ].nunique()
                ),

            "原始平均修正":
                float(
                    np.mean(
                        original_delta
                    )
                ),

            "替换后平均修正":
                float(
                    np.mean(
                        cf_delta
                    )
                ),

            "平均修正改善":
                float(
                    np.mean(
                        delta_gain
                    )
                ),

            "替换后正向修正比例":
                float(
                    np.mean(
                        cf_delta > 0
                    )
                ),

            "改善样本数":
                int(
                    np.sum(
                        per_sample_gain > 0
                    )
                ),

            "至少2个seed正向的样本数":
                int(
                    np.sum(
                        seed_positive >= 2
                    )
                ),

            "原始最终修复数":
                int(
                    np.sum(
                        original_p >= 0.5
                    )
                ),

            "替换后最终修复数":
                int(
                    np.sum(
                        cf_p >= 0.5
                    )
                ),

            "原始平均恶意概率":
                float(
                    original_p.mean()
                ),

            "替换后平均恶意概率":
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

    return (
        pd.DataFrame(rows)
        .sort_values(
            "平均修正改善",
            ascending=False,
        )
    )


def main():
    new = pd.read_csv(
        NEW
    )

    old = pd.read_csv(
        OLD
    )

    pair = pd.read_csv(
        PAIR
    )

    for d in (
        new,
        old,
        pair,
    ):
        d["sha256"] = norm_sha(
            d["sha256"]
        )

    reproduction_gate(
        new,
        old,
    )

    fn = pair[
        (pair["year"] == 2020)
        &
        (pair["error_type"] == "FN")
    ].copy()

    if len(fn) != 14:
        raise RuntimeError(
            "expected 14 routed 2020 FN"
        )

    fn["group"] = fn.apply(
        group_name,
        axis=1,
    )

    print()
    print(
        "2020 FN 分组数量"
    )
    print(
        fn["group"]
        .value_counts()
        .to_string()
    )

    data = new.merge(
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

    persistent = summarize(
        data,
        "PERSISTENT_BOTH_WRONG",
    )

    control = summarize(
        data,
        "BOTH_FIXED",
    )

    print()
    print(
        "=" * 150
    )
    print(
        "8个持续错误2020 FN："
        "GraphGlobal分组归因"
    )
    print(
        "=" * 150
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
        "=" * 150
    )
    print(
        "3个原本可正确修复FN：控制组"
    )
    print(
        "=" * 150
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
        / "m0_graphglobal_groups"
    )

    out.mkdir(
        parents=True,
        exist_ok=True,
    )

    persistent.to_csv(
        out
        / "persistent_2020_fn.csv",
        index=False,
    )

    control.to_csv(
        out
        / "both_fixed_control.csv",
        index=False,
    )

    print()
    print(
        "saved:",
        out,
    )


if __name__ == "__main__":
    main()
