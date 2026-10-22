#!/usr/bin/env python3

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(
    "artifacts/care-di25-host-m0-future-replay-v1/"
    "mechanism_diagnosis"
)

UPSTREAM = (
    ROOT
    / "m0_rolegraph_upstream"
    / "per_seed_block_replacement.csv"
)

TOKEN = (
    ROOT
    / "m0_block_replacement"
    / "per_seed_block_replacement.csv"
)

PAIR = (
    ROOT
    / "analysis"
    / "direct_vs_bounded"
    / "paired_overlap.csv"
)

ARMS = [
    "node",
    "graph_global",
    "pair",
    "inter",
    "full_graph",
]


def norm(series):
    return (
        series.astype(str)
        .str.strip()
        .str.lower()
    )


def assign_group(row):

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


def summarize_group(
    data,
    group_name,
):

    rows = []

    g = data[
        data["group"]
        == group_name
    ].copy()

    for arm in ARMS:

        delta_col = (
            f"cf_{arm}_bounded_delta"
        )

        prob_col = (
            f"cf_{arm}_probability"
        )

        original_delta = (
            g[
                "original_bounded_delta"
            ]
            .to_numpy(float)
        )

        cf_delta = (
            g[
                delta_col
            ]
            .to_numpy(float)
        )

        original_p = (
            g.groupby("sha256")[
                "original_probability"
            ]
            .mean()
        )

        cf_p = (
            g.groupby("sha256")[
                prob_col
            ]
            .mean()
        )

        per_sample = (
            g.assign(
                delta_gain=(
                    g[delta_col]
                    -
                    g[
                        "original_bounded_delta"
                    ]
                )
            )
            .groupby("sha256")[
                "delta_gain"
            ]
            .mean()
        )

        seed_positive = (
            g.assign(
                positive=(
                    g[delta_col]
                    > 0
                ).astype(int)
            )
            .groupby("sha256")[
                "positive"
            ]
            .sum()
        )

        rows.append({
            "arm": arm,

            "n_samples":
                int(
                    g["sha256"]
                    .nunique()
                ),

            "mean_original_delta":
                float(
                    np.mean(
                        original_delta
                    )
                ),

            "mean_cf_delta":
                float(
                    np.mean(
                        cf_delta
                    )
                ),

            "mean_delta_change":
                float(
                    np.mean(
                        cf_delta
                        -
                        original_delta
                    )
                ),

            "positive_direction_rate_cf":
                float(
                    np.mean(
                        cf_delta > 0
                    )
                ),

            "samples_delta_gain_gt_0":
                int(
                    np.sum(
                        per_sample > 0
                    )
                ),

            "samples_ge2_positive_seeds":
                int(
                    np.sum(
                        seed_positive >= 2
                    )
                ),

            "ensemble_fixed_original":
                int(
                    np.sum(
                        original_p
                        >= 0.5
                    )
                ),

            "ensemble_fixed_cf":
                int(
                    np.sum(
                        cf_p >= 0.5
                    )
                ),

            "mean_ensemble_p_original":
                float(
                    original_p.mean()
                ),

            "mean_ensemble_p_cf":
                float(
                    cf_p.mean()
                ),

            "mean_probability_gain":
                float(
                    (
                        cf_p
                        -
                        original_p
                    ).mean()
                ),
        })

    return pd.DataFrame(
        rows
    )


def main():

    u = pd.read_csv(
        UPSTREAM
    )

    t = pd.read_csv(
        TOKEN
    )

    p = pd.read_csv(
        PAIR
    )

    for d in (
        u,
        t,
        p,
    ):
        d["sha256"] = norm(
            d["sha256"]
        )

    if len(u) != 1989 * 3:
        raise RuntimeError(
            "upstream row count changed"
        )

    fn = p[
        (p["year"] == 2020)
        &
        (p["error_type"] == "FN")
    ].copy()

    if len(fn) != 14:
        raise RuntimeError(
            "expected 14 2020 FN"
        )

    fn["group"] = fn.apply(
        assign_group,
        axis=1,
    )

    u = u.merge(
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

    print()
    print(
        "2020 FN GROUP COUNTS"
    )
    print(
        fn["group"]
        .value_counts()
        .to_string()
    )

    persistent = summarize_group(
        u,
        "PERSISTENT_BOTH_WRONG",
    )

    control = summarize_group(
        u,
        "BOTH_FIXED",
    )

    # ---------------------------------------------
    # Add previously validated 32-D token
    # replacement as a reference row.
    # ---------------------------------------------

    tp = t.merge(
        fn[
            [
                "sha256",
                "group",
            ]
        ],
        on="sha256",
        how="inner",
    )

    g = tp[
        tp["group"]
        ==
        "PERSISTENT_BOTH_WRONG"
    ]

    orig_delta = (
        g[
            "original_bounded_delta"
        ].to_numpy(float)
    )

    cf_delta = (
        g[
            "cf_role_graph_bounded_delta"
        ].to_numpy(float)
    )

    op = (
        g.groupby("sha256")[
            "original_probability"
        ].mean()
    )

    cp = (
        g.groupby("sha256")[
            "cf_role_graph_probability"
        ].mean()
    )

    token_row = {
        "arm":
            "token32_reference",

        "n_samples":
            8,

        "mean_original_delta":
            float(
                np.mean(
                    orig_delta
                )
            ),

        "mean_cf_delta":
            float(
                np.mean(
                    cf_delta
                )
            ),

        "mean_delta_change":
            float(
                np.mean(
                    cf_delta
                    -
                    orig_delta
                )
            ),

        "positive_direction_rate_cf":
            float(
                np.mean(
                    cf_delta > 0
                )
            ),

        "samples_delta_gain_gt_0":
            8,

        "samples_ge2_positive_seeds":
            int(
                np.sum(
                    g.assign(
                        pos=(
                            g[
                                "cf_role_graph_bounded_delta"
                            ]
                            > 0
                        ).astype(int)
                    )
                    .groupby("sha256")[
                        "pos"
                    ]
                    .sum()
                    >= 2
                )
            ),

        "ensemble_fixed_original":
            int(
                np.sum(
                    op >= 0.5
                )
            ),

        "ensemble_fixed_cf":
            int(
                np.sum(
                    cp >= 0.5
                )
            ),

        "mean_ensemble_p_original":
            float(
                op.mean()
            ),

        "mean_ensemble_p_cf":
            float(
                cp.mean()
            ),

        "mean_probability_gain":
            float(
                (
                    cp - op
                ).mean()
            ),
    }

    persistent = pd.concat(
        [
            persistent,
            pd.DataFrame(
                [token_row]
            ),
        ],
        ignore_index=True,
    )

    persistent = persistent.sort_values(
        "mean_delta_change",
        ascending=False,
    )

    print()
    print("=" * 150)
    print(
        "PERSISTENT 2020 FN:"
        " ROLE-GRAPH UPSTREAM ATTRIBUTION"
    )
    print("=" * 150)

    print(
        persistent.to_string(
            index=False,
            float_format=lambda z:
                f"{z:.4f}",
        )
    )

    print()
    print("=" * 150)
    print(
        "BOTH-FIXED CONTROL"
    )
    print("=" * 150)

    print(
        control.sort_values(
            "mean_delta_change",
            ascending=False,
        ).to_string(
            index=False,
            float_format=lambda z:
                f"{z:.4f}",
        )
    )

    out = (
        ROOT
        / "analysis"
        / "m0_rolegraph_upstream"
    )

    out.mkdir(
        parents=True,
        exist_ok=True,
    )

    persistent.to_csv(
        out
        / "persistent_summary.csv",
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
