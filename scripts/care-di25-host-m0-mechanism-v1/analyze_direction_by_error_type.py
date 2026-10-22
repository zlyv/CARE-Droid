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
    "direction_by_error_type"
)

MAX_DELTA = 2.25


def logit(p):
    p = np.clip(
        np.asarray(p, float),
        1e-12,
        1.0 - 1e-12,
    )
    return np.log(p / (1.0 - p))


def error_type(y, p):
    pred = int(p >= 0.5)

    if pred == y:
        return "CORRECT"

    if y == 1:
        return "FN"

    return "FP"


def load_source():

    d = pd.read_csv(SOURCE)

    p1_col = "Global24__prob"
    route_col = "A25__routed"

    final_col = (
        "DI25__Host__Anomaly__FULL__"
        "Bounded__prob"
    )

    delta_col = (
        "DI25__Host__Anomaly__FULL__"
        "Bounded__bounded_delta"
    )

    rows = []

    for _, r in d.iterrows():

        if int(r[route_col]) != 1:
            continue

        y = int(r["label"])
        p1 = float(r[p1_col])

        et = error_type(y, p1)

        if et == "CORRECT":
            continue

        pf = float(r[final_col])

        z1 = float(logit([p1])[0])
        zf = float(logit([pf])[0])

        delta_effective = zf - z1

        if delta_col in d.columns:
            delta_logged = float(
                r[delta_col]
            )
        else:
            delta_logged = np.nan

        rows.append({
            "domain": "2017",
            "year": 2017,
            "sha256": r["sha256"],
            "label": y,
            "error_type": et,
            "p_stage1": p1,
            "p_final": pf,
            "stage1_logit": z1,
            "effective_delta": delta_effective,
            "logged_delta": delta_logged,
            "failure_type": (
                "FIXED"
                if int(pf >= 0.5) == y
                else (
                    "WRONG_DIRECTION"
                    if (
                        delta_effective > 0
                        and y == 0
                    )
                    or (
                        delta_effective < 0
                        and y == 1
                    )
                    else "OTHER_UNFIXED"
                )
            ),
        })

    return pd.DataFrame(rows)


def load_future():

    d = pd.read_csv(FUTURE)

    q = d[
        d["cohort"].isin(
            ["Fixed", "Unfixed"]
        )
    ].copy()

    q["domain"] = q["year"].astype(str)

    q["error_type"] = [
        error_type(
            int(y),
            float(p),
        )
        for y, p in zip(
            q["label"],
            q["p_stage1"],
        )
    ]

    q["effective_delta"] = (
        q["effective_delta"]
        .astype(float)
    )

    return q


def main():

    s = load_source()
    f = load_future()

    OUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    # -------------------------------------------------
    # Counts and wrong-direction rates.
    # -------------------------------------------------

    rows = []

    domains = [
        ("2017", s),
        ("2018", f[f["year"] == 2018]),
        ("2019", f[f["year"] == 2019]),
        ("2020", f[f["year"] == 2020]),
        ("2021", f[f["year"] == 2021]),
        ("2018-2021", f),
    ]

    for domain, d in domains:

        for et in ("FN", "FP"):

            g = d[
                d["error_type"] == et
            ]

            if len(g) == 0:
                continue

            wrong = int(
                np.sum(
                    g["failure_type"]
                    == "WRONG_DIRECTION"
                )
            )

            fixed = int(
                np.sum(
                    g["failure_type"]
                    == "FIXED"
                )
            )

            rows.append({
                "domain": domain,
                "error_type": et,
                "routed_stage1_errors":
                    len(g),
                "fixed": fixed,
                "wrong_direction":
                    wrong,
                "wrong_direction_rate":
                    wrong / len(g),
                "fix_rate":
                    fixed / len(g),
                "mean_abs_stage1_logit":
                    float(
                        np.mean(
                            np.abs(
                                logit(
                                    g["p_stage1"]
                                    .to_numpy(float)
                                )
                            )
                        )
                    ),
                "mean_effective_delta":
                    float(
                        g[
                            "effective_delta"
                        ].mean()
                    ),
                "mean_abs_effective_delta":
                    float(
                        np.mean(
                            np.abs(
                                g[
                                    "effective_delta"
                                ].to_numpy(float)
                            )
                        )
                    ),
            })

    out = pd.DataFrame(rows)

    out.to_csv(
        OUT / "direction_by_error_type.csv",
        index=False,
    )

    print()
    print("=" * 120)
    print("DIRECTION FAILURE BY STAGE-I ERROR TYPE")
    print("=" * 120)

    print(
        out.to_string(
            index=False,
            float_format=lambda x:
                f"{x:.4f}",
        )
    )

    print()
    print("=" * 120)
    print("WRONG-DIRECTION RATE PIVOT")
    print("=" * 120)

    print(
        out.pivot(
            index="domain",
            columns="error_type",
            values="wrong_direction_rate",
        )
        .to_string(
            float_format=lambda x:
                f"{x:.4f}"
        )
    )

    print()
    print("=" * 120)
    print("ROUTED ERROR COUNTS")
    print("=" * 120)

    print(
        out.pivot(
            index="domain",
            columns="error_type",
            values="routed_stage1_errors",
        )
        .to_string()
    )

    print()
    print("=" * 120)
    print("FIX RATE")
    print("=" * 120)

    print(
        out.pivot(
            index="domain",
            columns="error_type",
            values="fix_rate",
        )
        .to_string(
            float_format=lambda x:
                f"{x:.4f}"
        )
    )

    print()
    print("saved:", OUT)


if __name__ == "__main__":
    main()
