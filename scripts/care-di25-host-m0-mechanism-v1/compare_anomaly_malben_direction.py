#!/usr/bin/env python3

from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.metrics import (
    balanced_accuracy_score,
    roc_auc_score,
    f1_score,
)


ROOT = Path(
    "artifacts/care-di25-host-m0-future-replay-v1"
)

BASE = ROOT / "replay_predictions_with_labels.csv"

M0 = (
    ROOT
    / "anomaly_bounded"
    / "future_expert_predictions.csv"
)

MB = (
    ROOT
    / "mechanism_diagnosis"
    / "malben_bounded"
    / "future_expert_predictions.csv"
)

OUT = (
    ROOT
    / "mechanism_diagnosis"
    / "analysis"
    / "anomaly_vs_malben"
)

MAX_DELTA = 2.25
EPS = 1e-6


def norm(x):
    return (
        x.astype(str)
        .str.strip()
        .str.lower()
    )


def logit(p):
    p = np.clip(
        np.asarray(p, float),
        1e-12,
        1.0 - 1e-12,
    )
    return np.log(
        p / (1.0 - p)
    )


def failure_type(y, p1, pe):
    pred1 = int(p1 >= 0.5)
    prede = int(pe >= 0.5)

    if pred1 == y:
        return "STAGE1_CORRECT"

    if prede == y:
        return "FIXED"

    z1 = float(logit([p1])[0])
    ze = float(logit([pe])[0])

    delta = ze - z1

    truth_sign = (
        1.0 if int(y) == 1
        else -1.0
    )

    aligned = (
        delta * truth_sign
    )

    if abs(delta) <= EPS:
        return "NEAR_ZERO"

    if aligned < 0:
        return "WRONG_DIRECTION"

    if abs(z1) >= MAX_DELTA:
        return "BOUND_LIMIT"

    return "RIGHT_DIRECTION_TOO_WEAK"


def attach_expert(base, path, name):
    e = pd.read_csv(path)

    e["sha256"] = norm(
        e["sha256"]
    )

    if (
        len(e) != 1989
        or e["sha256"].nunique() != 1989
    ):
        raise RuntimeError(
            f"{name} cohort mismatch"
        )

    e = e[
        [
            "sha256",
            "p_expert",
        ]
    ].rename(
        columns={
            "p_expert":
                f"p_expert_{name}"
        }
    )

    return base.merge(
        e,
        on="sha256",
        how="inner",
        validate="one_to_one",
    )


def summarize(d, arm, year=None):

    q = d.copy()

    if year is not None:
        q = q[q["year"] == year]

    y = q["y"].to_numpy(int)

    p1 = (
        q["p_stage1"]
        .to_numpy(float)
    )

    pe = (
        q[f"p_expert_{arm}"]
        .to_numpy(float)
    )

    route = (
        q["route_DI"]
        .to_numpy(int)
        == 1
    )

    pf = np.where(
        route,
        pe,
        p1,
    )

    pred1 = (
        p1 >= 0.5
    ).astype(int)

    predf = (
        pf >= 0.5
    ).astype(int)

    stage1_wrong = (
        pred1 != y
    )

    final_wrong = (
        predf != y
    )

    fixed = int(
        np.sum(
            route
            & stage1_wrong
            & (~final_wrong)
        )
    )

    broken = int(
        np.sum(
            route
            & (~stage1_wrong)
            & final_wrong
        )
    )

    routed_errors = int(
        np.sum(
            route
            & stage1_wrong
        )
    )

    return {
        "arm": arm,
        "scope":
            "pooled"
            if year is None
            else str(year),

        "n":
            len(q),

        "routed":
            int(route.sum()),

        "routed_stage1_errors":
            routed_errors,

        "fixed":
            fixed,

        "broken":
            broken,

        "net":
            fixed - broken,

        "correction_success":
            (
                fixed / routed_errors
                if routed_errors
                else np.nan
            ),

        "BA":
            balanced_accuracy_score(
                y,
                predf,
            ),

        "AUC":
            roc_auc_score(
                y,
                pf,
            ),

        "F1":
            f1_score(
                y,
                predf,
            ),
    }


def main():

    d = pd.read_csv(BASE)

    d["sha256"] = norm(
        d["sha256"]
    )

    d = attach_expert(
        d,
        M0,
        "Anomaly",
    )

    d = attach_expert(
        d,
        MB,
        "MalBen",
    )

    if len(d) != 1989:
        raise RuntimeError(
            "merged cohort changed"
        )

    rows = []

    for arm in (
        "Anomaly",
        "MalBen",
    ):
        for year in (
            2018,
            2019,
            2020,
            2021,
        ):
            rows.append(
                summarize(
                    d,
                    arm,
                    year,
                )
            )

        rows.append(
            summarize(
                d,
                arm,
                None,
            )
        )

    metrics = pd.DataFrame(rows)

    OUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    metrics.to_csv(
        OUT / "metrics.csv",
        index=False,
    )

    # -----------------------------------------
    # Failure types only among ROUTED
    # Stage-I errors.
    # -----------------------------------------

    detail = []

    routed = d[
        d["route_DI"].astype(int)
        == 1
    ].copy()

    for _, r in routed.iterrows():

        y = int(r["y"])
        p1 = float(r["p_stage1"])

        pred1 = int(
            p1 >= 0.5
        )

        if pred1 == y:
            continue

        for arm in (
            "Anomaly",
            "MalBen",
        ):

            pe = float(
                r[f"p_expert_{arm}"]
            )

            z1 = float(
                logit([p1])[0]
            )

            ze = float(
                logit([pe])[0]
            )

            detail.append({
                "sha256":
                    r["sha256"],

                "year":
                    int(r["year"]),

                "label":
                    y,

                "arm":
                    arm,

                "p_stage1":
                    p1,

                "p_expert":
                    pe,

                "delta":
                    ze - z1,

                "failure_type":
                    failure_type(
                        y,
                        p1,
                        pe,
                    ),
            })

    detail = pd.DataFrame(
        detail
    )

    detail.to_csv(
        OUT /
        "routed_stage1_error_transitions.csv",
        index=False,
    )

    failure = (
        detail.groupby(
            [
                "arm",
                "year",
                "failure_type",
            ]
        )
        .size()
        .rename("count")
        .reset_index()
    )

    failure.to_csv(
        OUT /
        "failure_by_year.csv",
        index=False,
    )

    pooled = (
        detail.groupby(
            [
                "arm",
                "failure_type",
            ]
        )
        .size()
        .rename("count")
        .reset_index()
    )

    pooled.to_csv(
        OUT /
        "failure_pooled.csv",
        index=False,
    )

    print()
    print("=" * 120)
    print("ANOMALY vs MALBEN METRICS")
    print("=" * 120)

    print(
        metrics.to_string(
            index=False,
            float_format=lambda x:
                f"{x:.6f}",
        )
    )

    print()
    print("=" * 120)
    print("POOLED ROUTED-ERROR FAILURE TYPES")
    print("=" * 120)

    print(
        pooled.pivot(
            index="failure_type",
            columns="arm",
            values="count",
        )
        .fillna(0)
        .astype(int)
        .to_string()
    )

    print()
    print("=" * 120)
    print("FAILURE TYPES BY YEAR")
    print("=" * 120)

    for year in (
        2018,
        2019,
        2020,
        2021,
    ):
        print()
        print("YEAR", year)

        q = failure[
            failure["year"] == year
        ]

        print(
            q.pivot(
                index="failure_type",
                columns="arm",
                values="count",
            )
            .fillna(0)
            .astype(int)
            .to_string()
        )

    print()
    print("saved:", OUT)


if __name__ == "__main__":
    main()
