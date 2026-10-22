#!/usr/bin/env python3

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(
    "artifacts/care-di25-host-m0-future-replay-v1"
)

CAPTURE = (
    ROOT
    / "mechanism_diagnosis"
    / "instrumented_m0"
    / "routed_internal_capture.npz"
)

EQUIV = (
    ROOT
    / "mechanism_diagnosis"
    / "instrumented_m0"
    / "instrumentation_equivalence.json"
)

REPLAY = (
    ROOT
    / "replay_predictions_with_labels.csv"
)

OUT = (
    ROOT
    / "mechanism_diagnosis"
    / "analysis"
)

MAX_DELTA = 2.25
ZERO_EPS = 1e-6


def logit(x):
    x = np.clip(
        np.asarray(x, float),
        1e-12,
        1.0 - 1e-12,
    )

    return np.log(
        x / (1.0 - x)
    )


def cohort_name(
    stage1_wrong,
    final_wrong,
):
    if stage1_wrong and not final_wrong:
        return "Fixed"

    if stage1_wrong and final_wrong:
        return "Unfixed"

    if (
        not stage1_wrong
        and final_wrong
    ):
        return "Broken"

    return "KeptCorrect"


def failure_name(
    cohort,
    y,
    z1,
    delta,
):
    if cohort == "Fixed":
        return "FIXED"

    if cohort == "Broken":
        return "BROKEN_OVER_CORRECTION"

    if cohort == "KeptCorrect":
        return "KEPT_CORRECT"

    # Only Unfixed below.
    truth_sign = (
        1.0 if int(y) == 1
        else -1.0
    )

    aligned = (
        float(delta)
        * truth_sign
    )

    if abs(float(delta)) <= ZERO_EPS:
        return "NEAR_ZERO_EPS1E6"

    if aligned < 0.0:
        return "WRONG_DIRECTION"

    required = abs(
        float(z1)
    )

    # Correct direction, but even the
    # maximum permitted correction cannot
    # cross the Stage-I boundary.
    if required >= MAX_DELTA:
        return "BOUND_LIMIT"

    return "RIGHT_DIRECTION_TOO_WEAK"


def main():

    if not EQUIV.is_file():
        raise RuntimeError(
            "missing equivalence audit"
        )

    eq = json.loads(
        EQUIV.read_text(
            encoding="utf-8"
        )
    )

    if not eq.get("passed", False):
        raise RuntimeError(
            "instrumented replay did not pass"
        )

    if (
        float(
            eq[
                "max_abs_probability_diff_vs_locked"
            ]
        )
        > 1e-10
    ):
        raise RuntimeError(
            "locked probability mismatch"
        )

    d = pd.read_csv(
        REPLAY
    )

    d["sha256"] = (
        d["sha256"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    if "y" not in d.columns:
        raise RuntimeError(
            "replay file has no posthoc y"
        )

    with np.load(
        CAPTURE,
        allow_pickle=False,
    ) as z:

        ids = (
            z["sha256"]
            .astype(str)
        )

        years = np.asarray(
            z["year"],
            int,
        )

        fused = np.asarray(
            z["fused"],
            float,
        )

        raw = np.asarray(
            z["raw_delta"],
            float,
        )

        diagnostic = np.asarray(
            z["diagnostic"],
            float,
        )

        seed_probability = np.asarray(
            z["seed_probability"],
            float,
        )

        diagnostic_names = [
            str(x)
            for x in z[
                "diagnostic_names"
            ]
        ]

    if fused.shape != (
        3,
        497,
        113,
    ):
        raise RuntimeError(
            "capture fused shape changed: "
            + str(fused.shape)
        )

    if diagnostic.shape != (
        3,
        497,
        11,
    ):
        raise RuntimeError(
            "capture diagnostic shape changed"
        )

    if len(set(ids)) != 497:
        raise RuntimeError(
            "routed SHA uniqueness failed"
        )

    r = (
        d.set_index("sha256")
        .loc[ids]
        .reset_index()
    )

    if not np.all(
        r["route_DI"]
        .to_numpy(int)
        == 1
    ):
        raise RuntimeError(
            "capture contains unrouted APK"
        )

    if not np.array_equal(
        r["year"].to_numpy(int),
        years,
    ):
        raise RuntimeError(
            "capture/replay year mismatch"
        )

    y = r["y"].to_numpy(int)

    p1 = (
        r["p_stage1"]
        .to_numpy(float)
    )

    pe = (
        r["p_expert"]
        .to_numpy(float)
    )

    z1 = logit(p1)
    ze = logit(pe)

    effective_delta = (
        ze - z1
    )

    pred1 = (
        p1 >= 0.5
    ).astype(int)

    prede = (
        pe >= 0.5
    ).astype(int)

    stage1_wrong = (
        pred1 != y
    )

    final_wrong = (
        prede != y
    )

    cohorts = np.asarray([
        cohort_name(a, b)
        for a, b in zip(
            stage1_wrong,
            final_wrong,
        )
    ])

    failure = np.asarray([
        failure_name(
            cohorts[i],
            y[i],
            z1[i],
            effective_delta[i],
        )
        for i in range(len(ids))
    ])

    expected = {
        "Fixed": 91,
        "Unfixed": 57,
        "Broken": 24,
        "KeptCorrect": 325,
    }

    actual = {
        key: int(
            np.sum(
                cohorts == key
            )
        )
        for key in expected
    }

    if actual != expected:
        raise RuntimeError(
            "replay cohort changed: "
            f"{actual} != {expected}"
        )

    expected_year = {
        2018: {
            "Fixed": 53,
            "Unfixed": 28,
            "Broken": 12,
            "KeptCorrect": 105,
        },
        2019: {
            "Fixed": 14,
            "Unfixed": 9,
            "Broken": 6,
            "KeptCorrect": 71,
        },
        2020: {
            "Fixed": 9,
            "Unfixed": 17,
            "Broken": 5,
            "KeptCorrect": 68,
        },
        2021: {
            "Fixed": 15,
            "Unfixed": 3,
            "Broken": 1,
            "KeptCorrect": 81,
        },
    }

    for year, wanted in (
        expected_year.items()
    ):
        m = years == year

        got = {
            key: int(
                np.sum(
                    cohorts[m] == key
                )
            )
            for key in wanted
        }

        if got != wanted:
            raise RuntimeError(
                f"{year} cohort mismatch: "
                f"{got} != {wanted}"
            )

    # ---------------------------------------
    # Exact correction-input blocks.
    # ---------------------------------------

    graph = fused[:, :, 25:57]
    critical = fused[:, :, 57:81]
    attended = fused[:, :, 81:105]
    scalar = fused[:, :, 105:113]

    # Per-seed critical/attended consistency.
    dot = np.sum(
        critical * attended,
        axis=2,
    )

    critical_norm = np.linalg.norm(
        critical,
        axis=2,
    )

    attended_norm = np.linalg.norm(
        attended,
        axis=2,
    )

    cosine = (
        dot
        / np.maximum(
            critical_norm
            * attended_norm,
            1e-12,
        )
    )

    ca_l2 = np.linalg.norm(
        critical - attended,
        axis=2,
    )

    ca_mad = np.mean(
        np.abs(
            critical - attended
        ),
        axis=2,
    )

    ca_max = np.max(
        np.abs(
            critical - attended
        ),
        axis=2,
    )

    graph_norm = np.linalg.norm(
        graph,
        axis=2,
    )

    diag_index = {
        name: i
        for i, name in enumerate(
            diagnostic_names
        )
    }

    rows = []

    for i, sha in enumerate(ids):

        required = abs(
            float(z1[i])
        )

        strength_ratio = (
            abs(
                float(
                    effective_delta[i]
                )
            )
            / max(
                required,
                1e-12,
            )
        )

        rows.append({
            "sha256":
                sha,

            "year":
                int(years[i]),

            "label":
                int(y[i]),

            "cohort":
                cohorts[i],

            "failure_type":
                failure[i],

            "p_stage1":
                float(p1[i]),

            "p_expert":
                float(pe[i]),

            "stage1_logit":
                float(z1[i]),

            "expert_logit":
                float(ze[i]),

            "stage1_margin":
                float(
                    abs(
                        p1[i] - 0.5
                    )
                ),

            "candidate_margin":
                float(
                    abs(
                        pe[i] - 0.5
                    )
                ),

            "prob_shift":
                float(
                    pe[i] - p1[i]
                ),

            "abs_prob_shift":
                float(
                    abs(
                        pe[i] - p1[i]
                    )
                ),

            # Ensemble-effective delta.
            "effective_delta":
                float(
                    effective_delta[i]
                ),

            "required_delta_to_cross":
                required,

            "strength_ratio":
                strength_ratio,

            # Three-seed raw / bounded behavior.
            "seed_mean_raw_delta":
                float(
                    np.mean(
                        raw[:, i]
                    )
                ),

            "seed_sd_raw_delta":
                float(
                    np.std(
                        raw[:, i]
                    )
                ),

            "seed_mean_bounded_delta":
                float(
                    np.mean(
                        diagnostic[
                            :,
                            i,
                            diag_index[
                                "bounded_delta"
                            ],
                        ]
                    )
                ),

            "seed_sd_bounded_delta":
                float(
                    np.std(
                        diagnostic[
                            :,
                            i,
                            diag_index[
                                "bounded_delta"
                            ],
                        ]
                    )
                ),

            "seed_probability_sd":
                float(
                    np.std(
                        seed_probability[
                            :,
                            i,
                        ]
                    )
                ),

            # Host consistency.
            "critical_attended_cosine":
                float(
                    np.mean(
                        cosine[:, i]
                    )
                ),

            "critical_attended_l2":
                float(
                    np.mean(
                        ca_l2[:, i]
                    )
                ),

            "critical_attended_mean_abs_diff":
                float(
                    np.mean(
                        ca_mad[:, i]
                    )
                ),

            "critical_attended_max_abs_diff":
                float(
                    np.mean(
                        ca_max[:, i]
                    )
                ),

            "critical_norm":
                float(
                    np.mean(
                        critical_norm[:, i]
                    )
                ),

            "attended_norm":
                float(
                    np.mean(
                        attended_norm[:, i]
                    )
                ),

            "graph_token_norm":
                float(
                    np.mean(
                        graph_norm[:, i]
                    )
                ),

            # Exact scalar-summary blocks.
            "critical_score":
                float(
                    np.mean(
                        scalar[:, i, 0]
                    )
                ),

            "mean_instance_score":
                float(
                    np.mean(
                        scalar[:, i, 1]
                    )
                ),

            "attention_entropy":
                float(
                    np.mean(
                        scalar[:, i, 2]
                    )
                ),

            "max_attention":
                float(
                    np.mean(
                        scalar[:, i, 3]
                    )
                ),

            "mean_external_distance":
                float(
                    np.mean(
                        scalar[:, i, 4]
                    )
                ),

            "mean_internal_distance":
                float(
                    np.mean(
                        scalar[:, i, 5]
                    )
                ),

            "mean_shared_commonness":
                float(
                    np.mean(
                        scalar[:, i, 6]
                    )
                ),

            "max_local_residual":
                float(
                    np.mean(
                        scalar[:, i, 7]
                    )
                ),

            "context_entropy":
                float(
                    np.mean(
                        diagnostic[
                            :,
                            i,
                            diag_index[
                                "context_entropy"
                            ],
                        ]
                    )
                ),

            "graph_pool_entropy":
                float(
                    np.mean(
                        diagnostic[
                            :,
                            i,
                            diag_index[
                                "graph_pool_entropy"
                            ],
                        ]
                    )
                ),
        })

    out = pd.DataFrame(
        rows
    )

    OUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    out.to_csv(
        OUT
        / "routed_mechanism_rows.csv",
        index=False,
    )

    year_cohort = (
        out.groupby(
            [
                "year",
                "cohort",
            ]
        )
        .size()
        .rename("count")
        .reset_index()
    )

    year_cohort.to_csv(
        OUT
        / "year_cohort_counts.csv",
        index=False,
    )

    unfixed_failure = (
        out[
            out["cohort"]
            == "Unfixed"
        ]
        .groupby(
            [
                "year",
                "failure_type",
            ]
        )
        .size()
        .rename("count")
        .reset_index()
    )

    pooled_failure = (
        out[
            out["cohort"]
            == "Unfixed"
        ]
        ["failure_type"]
        .value_counts()
        .rename_axis(
            "failure_type"
        )
        .reset_index(
            name="count"
        )
    )

    unfixed_failure.to_csv(
        OUT
        / "unfixed_failure_by_year.csv",
        index=False,
    )

    pooled_failure.to_csv(
        OUT
        / "unfixed_failure_pooled.csv",
        index=False,
    )

    features = [
        "stage1_margin",
        "candidate_margin",
        "abs_prob_shift",
        "effective_delta",
        "required_delta_to_cross",
        "strength_ratio",
        "seed_sd_raw_delta",
        "seed_probability_sd",
        "critical_attended_cosine",
        "critical_attended_l2",
        "critical_attended_mean_abs_diff",
        "critical_score",
        "mean_instance_score",
        "attention_entropy",
        "max_attention",
        "mean_external_distance",
        "mean_internal_distance",
        "mean_shared_commonness",
        "max_local_residual",
        "graph_pool_entropy",
    ]

    summary_rows = []

    for cohort in (
        "Fixed",
        "Unfixed",
        "Broken",
        "KeptCorrect",
    ):

        g = out[
            out["cohort"]
            == cohort
        ]

        for feature in features:

            values = (
                g[feature]
                .to_numpy(float)
            )

            summary_rows.append({
                "cohort":
                    cohort,

                "feature":
                    feature,

                "n":
                    len(values),

                "mean":
                    float(
                        np.mean(values)
                    ),

                "median":
                    float(
                        np.median(values)
                    ),

                "q25":
                    float(
                        np.quantile(
                            values,
                            0.25,
                        )
                    ),

                "q75":
                    float(
                        np.quantile(
                            values,
                            0.75,
                        )
                    ),
            })

    summary = pd.DataFrame(
        summary_rows
    )

    summary.to_csv(
        OUT
        / "cohort_feature_summary.csv",
        index=False,
    )

    print()
    print("=" * 100)
    print("COHORT COUNTS")
    print("=" * 100)

    print(
        out["cohort"]
        .value_counts()
        .to_string()
    )

    print()
    print("=" * 100)
    print("YEAR x COHORT")
    print("=" * 100)

    print(
        out.groupby(
            [
                "year",
                "cohort",
            ]
        )
        .size()
        .unstack(
            fill_value=0
        )
        .to_string()
    )

    print()
    print("=" * 100)
    print(
        "UNFIXED FAILURE DECOMPOSITION"
    )
    print("=" * 100)

    print(
        pooled_failure
        .to_string(
            index=False
        )
    )

    print()
    print("=" * 100)
    print(
        "UNFIXED FAILURE BY YEAR"
    )
    print("=" * 100)

    print(
        unfixed_failure
        .pivot(
            index="year",
            columns="failure_type",
            values="count",
        )
        .fillna(0)
        .astype(int)
        .to_string()
    )

    print()
    print(
        "saved:",
        OUT
        / "routed_mechanism_rows.csv",
    )

    print(
        "saved:",
        OUT
        / "cohort_feature_summary.csv",
    )


if __name__ == "__main__":
    main()
