#!/usr/bin/env python3

from pathlib import Path
import numpy as np
import pandas as pd


SOURCE = Path(
    "artifacts/care-di25-host-input-replay-v1/"
    "arms/FULL/source_predictions.csv"
)

FUTURE = Path(
    "artifacts/care-di25-host-m0-future-replay-v1/"
    "mechanism_diagnosis/analysis/"
    "routed_mechanism_rows.csv"
)

OUT = Path(
    "artifacts/care-di25-host-m0-future-replay-v1/"
    "mechanism_diagnosis/analysis/"
    "source_future_direction_shift"
)

MAX_DELTA = 2.25
EPS = 1e-6


def logit(p):
    p = np.clip(
        np.asarray(p, float),
        1e-12,
        1.0 - 1e-12,
    )
    return np.log(p / (1.0 - p))


def find_one(columns, candidates, desc):
    for c in candidates:
        if c in columns:
            return c

    raise RuntimeError(
        f"cannot resolve {desc}; columns={list(columns)}"
    )


def find_final_prob(columns):

    # Prefer formal DI25 probability.
    candidates = [
        c for c in columns
        if (
            c.startswith("DI25__")
            and c.endswith("__prob")
            and "direct" not in c.lower()
        )
    ]

    if len(candidates) == 1:
        return candidates[0]

    # Fallback.
    candidates = [
        c for c in columns
        if (
            c.endswith("__prob")
            and c != "Global24__prob"
            and "direct" not in c.lower()
        )
    ]

    if len(candidates) == 1:
        return candidates[0]

    raise RuntimeError(
        "cannot uniquely resolve final probability: "
        + repr(candidates)
    )


def failure_type(y, p1, pf):

    pred1 = int(p1 >= 0.5)
    predf = int(pf >= 0.5)

    if pred1 == y:
        if predf != y:
            return "BROKEN"
        return "KEPT_CORRECT"

    if predf == y:
        return "FIXED"

    z1 = float(logit([p1])[0])
    zf = float(logit([pf])[0])

    delta = zf - z1

    truth_sign = (
        1.0 if int(y) == 1
        else -1.0
    )

    aligned = delta * truth_sign

    if abs(delta) <= EPS:
        return "NEAR_ZERO"

    if aligned < 0:
        return "WRONG_DIRECTION"

    if abs(z1) >= MAX_DELTA:
        return "BOUND_LIMIT"

    return "RIGHT_DIRECTION_TOO_WEAK"


def main():

    s = pd.read_csv(SOURCE)

    print("SOURCE COLUMNS:")
    print(list(s.columns))
    print()

    label_col = find_one(
        s.columns,
        ["label", "y", "metadata_label"],
        "label",
    )

    p1_col = find_one(
        s.columns,
        [
            "Global24__prob",
            "p_stage1",
            "stage1_prob",
        ],
        "stage1 probability",
    )

    route_col = find_one(
        s.columns,
        [
            "A25__routed",
            "DI25__routed",
            "route_DI",
        ],
        "route",
    )

    final_col = find_final_prob(
        s.columns
    )

    print("resolved label =", label_col)
    print("resolved p1    =", p1_col)
    print("resolved route =", route_col)
    print("resolved final =", final_col)
    print()

    y = s[label_col].astype(int).to_numpy()
    p1 = s[p1_col].astype(float).to_numpy()
    pf = s[final_col].astype(float).to_numpy()
    route = s[route_col].astype(int).to_numpy() == 1

    if len(s) != 799:
        raise RuntimeError(
            f"source n changed: {len(s)}"
        )

    if int(route.sum()) != 200:
        raise RuntimeError(
            f"source routed changed: {route.sum()}"
        )

    pred1 = (p1 >= 0.5).astype(int)
    predf = (pf >= 0.5).astype(int)

    stage1_wrong = pred1 != y
    final_wrong = predf != y

    source_rows = []

    for i in np.flatnonzero(route):

        source_rows.append({
            "domain": "2017",
            "label": int(y[i]),
            "p_stage1": float(p1[i]),
            "p_final": float(pf[i]),
            "stage1_wrong": int(stage1_wrong[i]),
            "final_wrong": int(final_wrong[i]),
            "failure_type": failure_type(
                y[i],
                p1[i],
                pf[i],
            ),
        })

    src = pd.DataFrame(source_rows)

    # Only routed Stage-I errors for mechanism comparison.
    src_err = src[
        src["stage1_wrong"] == 1
    ].copy()

    future = pd.read_csv(FUTURE)

    future_err = future[
        future["cohort"]
        .isin(["Fixed", "Unfixed"])
    ].copy()

    if len(future_err) != 148:
        raise RuntimeError(
            f"future routed Stage-I errors changed: "
            f"{len(future_err)}"
        )

    print("=" * 100)
    print("SOURCE 2017 ROUTING")
    print("=" * 100)

    source_stage1_errors = int(
        stage1_wrong.sum()
    )

    routed_source_errors = int(
        np.sum(
            route & stage1_wrong
        )
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

    print("Stage-I errors        =", source_stage1_errors)
    print("routed Stage-I errors =", routed_source_errors)
    print(
        "router error recall   =",
        routed_source_errors / source_stage1_errors,
    )
    print("fixed                 =", fixed)
    print("broken                =", broken)
    print("net                   =", fixed - broken)
    print(
        "correction success    =",
        fixed / routed_source_errors,
    )

    print()
    print("=" * 100)
    print("SOURCE 2017 ROUTED-ERROR FAILURE TYPES")
    print("=" * 100)

    src_count = (
        src_err["failure_type"]
        .value_counts()
    )

    print(src_count.to_string())

    print()
    print("=" * 100)
    print("FUTURE 2018-2021 ROUTED-ERROR FAILURE TYPES")
    print("=" * 100)

    fut_count = (
        future_err["failure_type"]
        .value_counts()
    )

    print(fut_count.to_string())

    categories = [
        "FIXED",
        "WRONG_DIRECTION",
        "RIGHT_DIRECTION_TOO_WEAK",
        "BOUND_LIMIT",
        "NEAR_ZERO",
    ]

    rows = []

    for domain, counts, total in [
        ("2017", src_count, len(src_err)),
        ("2018-2021", fut_count, len(future_err)),
    ]:

        for name in categories:

            count = int(
                counts.get(name, 0)
            )

            rows.append({
                "domain": domain,
                "failure_type": name,
                "count": count,
                "total_routed_stage1_errors": total,
                "fraction": (
                    count / total
                    if total
                    else np.nan
                ),
            })

    comp = pd.DataFrame(rows)

    OUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    comp.to_csv(
        OUT / "source_future_failure_fraction.csv",
        index=False,
    )

    src.to_csv(
        OUT / "source_routed_rows.csv",
        index=False,
    )

    print()
    print("=" * 100)
    print("SOURCE vs FUTURE FRACTIONS")
    print("=" * 100)

    pivot = comp.pivot(
        index="failure_type",
        columns="domain",
        values="fraction",
    )

    print(
        pivot.to_string(
            float_format=lambda x:
                f"{x:.4f}"
        )
    )

    if (
        "WRONG_DIRECTION"
        in pivot.index
    ):

        srate = float(
            pivot.loc[
                "WRONG_DIRECTION",
                "2017"
            ]
        )

        frate = float(
            pivot.loc[
                "WRONG_DIRECTION",
                "2018-2021"
            ]
        )

        print()
        print(
            "WrongDirection rate:"
        )
        print(
            f"2017      = {srate:.4f}"
        )
        print(
            f"2018-2021 = {frate:.4f}"
        )
        print(
            f"delta      = {frate-srate:+.4f}"
        )

    print()
    print("saved:", OUT)


if __name__ == "__main__":
    main()
