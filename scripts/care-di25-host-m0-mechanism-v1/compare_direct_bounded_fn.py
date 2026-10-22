#!/usr/bin/env python3

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, roc_auc_score, f1_score


ROOT = Path(
    "artifacts/care-di25-host-m0-future-replay-v1"
)

BASE = ROOT / "replay_predictions_with_labels.csv"

DIRECT_CANDIDATES = [
    ROOT / "mechanism_diagnosis/anomaly_direct/future_expert_predictions.csv",
    Path(
        "artifacts/care-route-stage2-future-matrix-v1/"
        "arms/anomaly_direct/future_expert_predictions.csv"
    ),
]

OUT = (
    ROOT
    / "mechanism_diagnosis/analysis"
    / "direct_vs_bounded"
)


def logit(p):
    p = np.clip(
        np.asarray(p, float),
        1e-12,
        1.0 - 1e-12,
    )
    return np.log(p / (1.0 - p))


def norm(s):
    return (
        s.astype(str)
        .str.strip()
        .str.lower()
    )


def classify_error(y, p1, pe):
    pred1 = int(p1 >= 0.5)
    pred2 = int(pe >= 0.5)

    if pred1 == y:
        return (
            "BROKEN"
            if pred2 != y
            else "KEPT_CORRECT"
        )

    if pred2 == y:
        return "FIXED"

    delta = (
        float(logit([pe])[0])
        - float(logit([p1])[0])
    )

    truth_sign = (
        1.0
        if int(y) == 1
        else -1.0
    )

    if delta * truth_sign < 0:
        return "WRONG_DIRECTION"

    return "RIGHT_DIRECTION_TOO_WEAK"


def summarize_arm(d, arm, prob_col):

    rows = []

    for scope in [
        2018,
        2019,
        2020,
        2021,
        "pooled",
    ]:

        q = (
            d
            if scope == "pooled"
            else d[d["year"] == scope]
        )

        y = q["y"].to_numpy(int)
        p1 = q["p_stage1"].to_numpy(float)
        pe = q[prob_col].to_numpy(float)

        route = (
            q["route_DI"].to_numpy(int)
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

        stage1_wrong = pred1 != y
        final_wrong = predf != y

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

        rows.append({
            "arm": arm,
            "scope": str(scope),
            "n": len(q),
            "fixed": fixed,
            "broken": broken,
            "net": fixed - broken,
            "BA": balanced_accuracy_score(
                y,
                predf,
            ),
            "AUC": roc_auc_score(
                y,
                pf,
            ),
            "F1": f1_score(
                y,
                predf,
            ),
        })

    return rows


def main():

    direct_path = next(
        (
            p for p in DIRECT_CANDIDATES
            if p.is_file()
        ),
        None,
    )

    if direct_path is None:
        raise RuntimeError(
            "anomaly direct prediction file not found"
        )

    print("DIRECT =", direct_path)

    d = pd.read_csv(BASE)
    d["sha256"] = norm(d["sha256"])

    direct = pd.read_csv(direct_path)
    direct["sha256"] = norm(
        direct["sha256"]
    )

    if (
        len(direct) != 1989
        or direct["sha256"].nunique() != 1989
    ):
        raise RuntimeError(
            "direct future cohort mismatch"
        )

    direct = direct[
        ["sha256", "p_expert"]
    ].rename(
        columns={
            "p_expert":
                "p_direct"
        }
    )

    d = d.merge(
        direct,
        on="sha256",
        how="inner",
        validate="one_to_one",
    )

    if len(d) != 1989:
        raise RuntimeError(
            "merged future cohort changed"
        )

    # Current exact bounded expert already present
    # in replay_predictions_with_labels.csv.
    if "p_expert" not in d.columns:
        raise RuntimeError(
            "bounded p_expert missing"
        )

    result_rows = []

    for _, r in d.iterrows():

        if int(r["route_DI"]) != 1:
            continue

        y = int(r["y"])
        p1 = float(r["p_stage1"])

        pred1 = int(p1 >= 0.5)

        error_type = (
            "FN"
            if pred1 == 0 and y == 1
            else
            "FP"
            if pred1 == 1 and y == 0
            else
            "STAGE1_CORRECT"
        )

        for arm, col in [
            ("Bounded", "p_expert"),
            ("Direct", "p_direct"),
        ]:

            pe = float(r[col])

            result_rows.append({
                "sha256": r["sha256"],
                "year": int(r["year"]),
                "label": y,
                "error_type": error_type,
                "arm": arm,
                "p_stage1": p1,
                "p_expert": pe,
                "effective_delta":
                    float(logit([pe])[0])
                    - float(logit([p1])[0]),
                "failure_type":
                    classify_error(
                        y,
                        p1,
                        pe,
                    ),
            })

    detail = pd.DataFrame(
        result_rows
    )

    OUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    detail.to_csv(
        OUT / "routed_transitions.csv",
        index=False,
    )

    # Overall final performance.
    metrics = pd.DataFrame(
        summarize_arm(
            d,
            "Bounded",
            "p_expert",
        )
        +
        summarize_arm(
            d,
            "Direct",
            "p_direct",
        )
    )

    metrics.to_csv(
        OUT / "metrics.csv",
        index=False,
    )

    # Only routed Stage-I errors.
    err = detail[
        detail["error_type"]
        .isin(["FN", "FP"])
    ]

    counts = (
        err.groupby(
            [
                "arm",
                "year",
                "error_type",
                "failure_type",
            ]
        )
        .size()
        .rename("count")
        .reset_index()
    )

    counts.to_csv(
        OUT / "failure_by_year_and_error_type.csv",
        index=False,
    )

    print()
    print("=" * 120)
    print("FINAL METRICS")
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
    print("FN FAILURE TYPES")
    print("=" * 120)

    fn = counts[
        counts["error_type"] == "FN"
    ]

    for year in (
        2018,
        2019,
        2020,
        2021,
    ):

        print()
        print("YEAR", year)

        q = fn[
            fn["year"] == year
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
    print("=" * 120)
    print("FN WRONG-DIRECTION RATE")
    print("=" * 120)

    fn_rows = []

    for arm in (
        "Bounded",
        "Direct",
    ):

        for year in (
            2018,
            2019,
            2020,
            2021,
        ):

            q = err[
                (err["arm"] == arm)
                &
                (err["year"] == year)
                &
                (err["error_type"] == "FN")
            ]

            if not len(q):
                continue

            wrong = int(
                np.sum(
                    q["failure_type"]
                    == "WRONG_DIRECTION"
                )
            )

            fixed = int(
                np.sum(
                    q["failure_type"]
                    == "FIXED"
                )
            )

            fn_rows.append({
                "arm": arm,
                "year": year,
                "n_fn": len(q),
                "wrong": wrong,
                "wrong_rate":
                    wrong / len(q),
                "fixed": fixed,
                "fix_rate":
                    fixed / len(q),
                "mean_delta":
                    float(
                        q[
                            "effective_delta"
                        ].mean()
                    ),
            })

    fn_summary = pd.DataFrame(
        fn_rows
    )

    fn_summary.to_csv(
        OUT / "fn_summary.csv",
        index=False,
    )

    print(
        fn_summary.to_string(
            index=False,
            float_format=lambda x:
                f"{x:.4f}",
        )
    )

    print()
    print("saved:", OUT)


if __name__ == "__main__":
    main()
