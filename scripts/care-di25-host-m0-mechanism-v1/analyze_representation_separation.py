#!/usr/bin/env python3

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


ROOT = Path(
    "artifacts/care-di25-host-m0-future-replay-v1/"
    "mechanism_diagnosis/analysis"
)

P = ROOT / "direction_consensus_rows.csv"

d = pd.read_csv(P)


# ----------------------------------------------------------
# Only PRE-HEAD / representation features here.
#
# Deliberately exclude:
#   effective_delta
#   candidate_margin
#   prob_shift
#   strength_ratio
#
# because they are consequences of the correction head.
# ----------------------------------------------------------

FEATURES = [
    "critical_attended_cosine",
    "critical_attended_l2",
    "critical_attended_mean_abs_diff",
    "critical_attended_max_abs_diff",
    "critical_norm",
    "attended_norm",
    "graph_token_norm",

    "critical_score",
    "mean_instance_score",
    "attention_entropy",
    "max_attention",

    "mean_external_distance",
    "mean_internal_distance",
    "mean_shared_commonness",
    "max_local_residual",
    "context_entropy",
    "graph_pool_entropy",

    "graph_seed_dispersion",
    "critical_seed_dispersion",
    "attended_seed_dispersion",
    "summary_seed_dispersion",
]


def standardized_difference(a, b):

    a = np.asarray(a, float)
    b = np.asarray(b, float)

    va = np.var(a, ddof=1)
    vb = np.var(b, ddof=1)

    pooled = np.sqrt(
        (
            (len(a) - 1) * va +
            (len(b) - 1) * vb
        )
        /
        max(
            len(a) + len(b) - 2,
            1
        )
    )

    if pooled <= 1e-12:
        return 0.0

    return float(
        (np.mean(b) - np.mean(a))
        / pooled
    )


def compare(
    left_name,
    right_name,
    year=None,
):

    q = d.copy()

    if year is not None:
        q = q[q["year"] == year]

    a = q[
        q["cohort"] == left_name
    ]

    if right_name == "WrongDirection":
        b = q[
            q["failure_type"]
            == "WRONG_DIRECTION"
        ]
    else:
        b = q[
            q["cohort"]
            == right_name
        ]

    rows = []

    for feature in FEATURES:

        x0 = a[feature].to_numpy(float)
        x1 = b[feature].to_numpy(float)

        if (
            len(x0) < 2
            or len(x1) < 2
        ):
            continue

        y = np.r_[
            np.zeros(len(x0)),
            np.ones(len(x1))
        ]

        x = np.r_[x0, x1]

        auc = roc_auc_score(
            y,
            x
        )

        if auc >= 0.5:
            sep_auc = auc
            orientation = "higher_in_right"
        else:
            sep_auc = 1.0 - auc
            orientation = "lower_in_right"

        rows.append({
            "scope":
                "pooled"
                if year is None
                else str(year),

            "left":
                left_name,

            "right":
                right_name,

            "feature":
                feature,

            "n_left":
                len(x0),

            "n_right":
                len(x1),

            "left_mean":
                float(np.mean(x0)),

            "right_mean":
                float(np.mean(x1)),

            "left_median":
                float(np.median(x0)),

            "right_median":
                float(np.median(x1)),

            "separation_auc":
                float(sep_auc),

            "orientation":
                orientation,

            "standardized_difference":
                standardized_difference(
                    x0,
                    x1
                ),
        })

    return rows


rows = []

# Primary representation question.
rows += compare(
    "Fixed",
    "WrongDirection"
)

# Safety question.
rows += compare(
    "KeptCorrect",
    "Broken"
)

# Most informative temporal contrast.
for year in (
    2018,
    2019,
    2020,
):
    rows += compare(
        "Fixed",
        "WrongDirection",
        year
    )

out = pd.DataFrame(rows)

out.to_csv(
    ROOT /
    "representation_separation.csv",
    index=False
)


for scope in (
    "pooled",
    "2018",
    "2020",
):

    print()
    print("=" * 110)
    print(
        "FIXED vs WRONG DIRECTION:",
        scope
    )
    print("=" * 110)

    q = out[
        (out["scope"] == scope)
        &
        (out["left"] == "Fixed")
        &
        (out["right"] == "WrongDirection")
    ].sort_values(
        "separation_auc",
        ascending=False
    )

    print(
        q[
            [
                "feature",
                "n_left",
                "n_right",
                "left_mean",
                "right_mean",
                "separation_auc",
                "orientation",
                "standardized_difference",
            ]
        ]
        .head(15)
        .to_string(
            index=False,
            float_format=lambda x:
                f"{x:.4f}"
        )
    )


print()
print("=" * 110)
print("KEPT CORRECT vs BROKEN: pooled")
print("=" * 110)

q = out[
    (out["scope"] == "pooled")
    &
    (out["left"] == "KeptCorrect")
    &
    (out["right"] == "Broken")
].sort_values(
    "separation_auc",
    ascending=False
)

print(
    q[
        [
            "feature",
            "n_left",
            "n_right",
            "left_mean",
            "right_mean",
            "separation_auc",
            "orientation",
            "standardized_difference",
        ]
    ]
    .head(15)
    .to_string(
        index=False,
        float_format=lambda x:
            f"{x:.4f}"
    )
)
