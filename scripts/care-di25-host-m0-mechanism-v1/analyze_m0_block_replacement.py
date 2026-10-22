#!/usr/bin/env python3

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(
    "artifacts/care-di25-host-m0-future-replay-v1/"
    "mechanism_diagnosis"
)

ATTR = (
    ROOT
    / "m0_block_replacement"
    / "per_seed_block_replacement.csv"
)

PAIR = (
    ROOT
    / "analysis/direct_vs_bounded"
    / "paired_overlap.csv"
)

CLUSTER = (
    ROOT
    / "analysis/persistent_fn_cluster_audit"
    / "2020_routed_fn_cluster_audit.csv"
)

OUT = (
    ROOT
    / "analysis/m0_block_replacement"
)


BLOCKS = [
    "global24",
    "stage1_logit",
    "role_graph",
    "critical_host",
    "attended_host",
    "summary",
]


def norm(series):
    return (
        series.astype(str)
        .str.strip()
        .str.lower()
    )


def group_of(row):

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


def summarize(data, token):

    rows = []

    for block in BLOCKS:

        prob_col = (
            "cf_"
            + block
            + "_probability"
        )

        delta_col = (
            "cf_"
            + block
            + "_bounded_delta"
        )

        for group, g in data.groupby(
            "group"
        ):

            n_samples = (
                g["sha256"]
                .nunique()
            )

            # Seed-level direction effect.
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

            # Ensemble probability:
            # average probabilities, never latent vectors.
            ensemble = (
                g.groupby("sha256")[
                    prob_col
                ]
                .mean()
            )

            original_ensemble = (
                g.groupby("sha256")[
                    "original_probability"
                ]
                .mean()
            )

            rows.append({
                "analysis": token,
                "block": block,
                "group": group,
                "n_samples":
                    int(n_samples),
                "n_seed_sample":
                    int(len(g)),
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
                            - original_delta
                        )
                    ),
                "positive_direction_rate_original":
                    float(
                        np.mean(
                            original_delta > 0
                        )
                    ),
                "positive_direction_rate_cf":
                    float(
                        np.mean(
                            cf_delta > 0
                        )
                    ),
                "seed_level_neg_to_pos_rate":
                    float(
                        np.mean(
                            (
                                original_delta < 0
                            )
                            &
                            (
                                cf_delta > 0
                            )
                        )
                    ),
                "ensemble_fixed_original":
                    int(
                        np.sum(
                            original_ensemble
                            >= 0.5
                        )
                    ),
                "ensemble_fixed_cf":
                    int(
                        np.sum(
                            ensemble
                            >= 0.5
                        )
                    ),
                "ensemble_fix_gain":
                    int(
                        np.sum(
                            ensemble
                            >= 0.5
                        )
                        -
                        np.sum(
                            original_ensemble
                            >= 0.5
                        )
                    ),
                "mean_ensemble_p_original":
                    float(
                        original_ensemble
                        .mean()
                    ),
                "mean_ensemble_p_cf":
                    float(
                        ensemble.mean()
                    ),
            })

    return rows


def main():

    a = pd.read_csv(
        ATTR
    )

    p = pd.read_csv(
        PAIR
    )

    c = pd.read_csv(
        CLUSTER
    )

    for d in (
        a,
        p,
        c,
    ):
        d["sha256"] = norm(
            d["sha256"]
        )

    if len(a) != (
        1989 * 3
    ):
        raise RuntimeError(
            "attribution row count changed"
        )

    # ---------------------------------------------
    # 2020 routed Stage-I FN.
    # ---------------------------------------------

    fn = p[
        (p["year"] == 2020)
        &
        (p["error_type"] == "FN")
    ].copy()

    if len(fn) != 14:
        raise RuntimeError(
            f"expected 14 FN, got {len(fn)}"
        )

    fn["group"] = fn.apply(
        group_of,
        axis=1,
    )

    print(
        "2020 FN group counts:"
    )

    print(
        fn["group"]
        .value_counts()
        .to_string()
    )

    x = a.merge(
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

    if len(x) != (
        14 * 3
    ):
        raise RuntimeError(
            "2020 FN seed-row mismatch: "
            + str(len(x))
        )

    # ---------------------------------------------
    # Full 14-sample analysis.
    # ---------------------------------------------

    rows = summarize(
        x,
        "ALL_14_FN",
    )

    # ---------------------------------------------
    # Deduplicated sensitivity analysis.
    #
    # Only persistent cohort needs Host-bag
    # deduplication. Keep one SHA per exact bag.
    # ---------------------------------------------

    persistent_hash = (
        c[
            c["persistent"]
            .astype(str)
            .str.lower()
            .isin(
                [
                    "true",
                    "1",
                ]
            )
        ][
            [
                "sha256",
                "exact_host_hash",
            ]
        ]
        .copy()
    )

    if len(
        persistent_hash
    ) != 8:
        raise RuntimeError(
            "persistent cluster audit changed"
        )

    keep_persistent = (
        persistent_hash
        .sort_values(
            "sha256"
        )
        .drop_duplicates(
            "exact_host_hash",
            keep="first",
        )[
            "sha256"
        ]
        .tolist()
    )

    if len(
        keep_persistent
    ) != 7:
        raise RuntimeError(
            "expected 7 unique "
            "persistent Host bags"
        )

    keep = (
        (x["group"]
         != "PERSISTENT_BOTH_WRONG")
        |
        x["sha256"].isin(
            keep_persistent
        )
    )

    x_dedup = x[
        keep
    ].copy()

    rows.extend(
        summarize(
            x_dedup,
            "PERSISTENT_HOST_DEDUP",
        )
    )

    result = pd.DataFrame(
        rows
    )

    OUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    result.to_csv(
        OUT / "block_replacement_summary.csv",
        index=False,
    )

    # ---------------------------------------------
    # Main display:
    # persistent failures first.
    # ---------------------------------------------

    print()
    print("=" * 130)
    print(
        "PERSISTENT 2020 FN:"
        " SOURCE-FN BLOCK REPLACEMENT"
    )
    print("=" * 130)

    q = result[
        (
            result["analysis"]
            == "ALL_14_FN"
        )
        &
        (
            result["group"]
            == "PERSISTENT_BOTH_WRONG"
        )
    ].copy()

    q = q.sort_values(
        [
            "ensemble_fix_gain",
            "mean_delta_change",
        ],
        ascending=False,
    )

    print(
        q[
            [
                "block",
                "n_samples",
                "mean_original_delta",
                "mean_cf_delta",
                "mean_delta_change",
                "positive_direction_rate_cf",
                "ensemble_fixed_original",
                "ensemble_fixed_cf",
                "ensemble_fix_gain",
                "mean_ensemble_p_original",
                "mean_ensemble_p_cf",
            ]
        ].to_string(
            index=False,
            float_format=lambda z:
                f"{z:.4f}",
        )
    )

    print()
    print("=" * 130)
    print(
        "BOTH_FIXED 2020 FN:"
        " CONTROL GROUP"
    )
    print("=" * 130)

    q = result[
        (
            result["analysis"]
            == "ALL_14_FN"
        )
        &
        (
            result["group"]
            == "BOTH_FIXED"
        )
    ].copy()

    print(
        q[
            [
                "block",
                "n_samples",
                "mean_original_delta",
                "mean_cf_delta",
                "mean_delta_change",
                "positive_direction_rate_cf",
                "ensemble_fixed_original",
                "ensemble_fixed_cf",
                "ensemble_fix_gain",
            ]
        ].to_string(
            index=False,
            float_format=lambda z:
                f"{z:.4f}",
        )
    )

    print()
    print("=" * 130)
    print(
        "PERSISTENT DEDUP:"
        " 7 UNIQUE HOST BAGS"
    )
    print("=" * 130)

    q = result[
        (
            result["analysis"]
            == "PERSISTENT_HOST_DEDUP"
        )
        &
        (
            result["group"]
            == "PERSISTENT_BOTH_WRONG"
        )
    ].copy()

    q = q.sort_values(
        [
            "ensemble_fix_gain",
            "mean_delta_change",
        ],
        ascending=False,
    )

    print(
        q[
            [
                "block",
                "n_samples",
                "mean_original_delta",
                "mean_cf_delta",
                "mean_delta_change",
                "positive_direction_rate_cf",
                "ensemble_fixed_cf",
                "ensemble_fix_gain",
            ]
        ].to_string(
            index=False,
            float_format=lambda z:
                f"{z:.4f}",
        )
    )

    print()
    print(
        "saved:",
        OUT,
    )


if __name__ == "__main__":
    main()
