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
    "fn_collapse"
)


FEATURES = [
    "critical_score",
    "mean_instance_score",
    "attention_entropy",
    "max_attention",
    "context_entropy",
    "mean_external_distance",
    "mean_internal_distance",
    "mean_shared_commonness",
    "max_local_residual",
    "graph_pool_entropy",
]


PREFIX = (
    "DI25__Host__Anomaly__FULL__Bounded__"
)


def logit(p):
    p = np.clip(
        np.asarray(p, float),
        1e-12,
        1.0 - 1e-12,
    )
    return np.log(p / (1.0 - p))


def load_source():

    d = pd.read_csv(SOURCE)

    y = d["label"].to_numpy(int)
    p1 = d["Global24__prob"].to_numpy(float)

    pf = d[
        PREFIX + "prob"
    ].to_numpy(float)

    route = (
        d["A25__routed"]
        .to_numpy(int)
        == 1
    )

    pred1 = (p1 >= 0.5).astype(int)

    # Routed False Negatives only.
    mask = (
        route
        & (y == 1)
        & (pred1 == 0)
    )

    q = pd.DataFrame({
        "year": 2017,
        "sha256": d.loc[mask, "sha256"].astype(str),
        "p_stage1": p1[mask],
        "p_final": pf[mask],
    })

    q["stage1_logit"] = logit(
        q["p_stage1"]
    )

    q["abs_stage1_logit"] = np.abs(
        q["stage1_logit"]
    )

    q["final_logit"] = logit(
        q["p_final"]
    )

    q["effective_delta"] = (
        q["final_logit"]
        - q["stage1_logit"]
    )

    q["fixed"] = (
        q["p_final"] >= 0.5
    ).astype(int)

    q["wrong_direction"] = (
        q["effective_delta"] < 0
    ).astype(int)

    for feature in FEATURES:
        q[feature] = (
            d.loc[
                mask,
                PREFIX + feature
            ]
            .to_numpy(float)
        )

    return q.reset_index(drop=True)


def load_future():

    d = pd.read_csv(FUTURE)

    # Routed Stage-I FN only.
    q = d[
        (d["label"] == 1)
        &
        (d["p_stage1"] < 0.5)
        &
        (d["cohort"].isin(
            ["Fixed", "Unfixed"]
        ))
    ].copy()

    q["abs_stage1_logit"] = np.abs(
        q["stage1_logit"]
        .to_numpy(float)
    )

    q["fixed"] = (
        q["cohort"] == "Fixed"
    ).astype(int)

    q["wrong_direction"] = (
        q["failure_type"]
        == "WRONG_DIRECTION"
    ).astype(int)

    return q.reset_index(drop=True)


def assign_source_bins(
    values,
    edges,
):
    return pd.cut(
        values,
        bins=edges,
        labels=[
            "Q1_low",
            "Q2",
            "Q3",
            "Q4_high",
        ],
        include_lowest=True,
        right=True,
    )


def main():

    source = load_source()
    future = load_future()

    if len(source) != 85:
        raise RuntimeError(
            f"2017 routed FN changed: {len(source)}"
        )

    if len(future) != 73:
        raise RuntimeError(
            f"future routed FN changed: {len(future)}"
        )

    OUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    # -------------------------------------------------
    # 1. Source-derived Stage-I confidence bins
    # -------------------------------------------------

    q25, q50, q75 = np.quantile(
        source["abs_stage1_logit"],
        [0.25, 0.50, 0.75],
    )

    edges = [
        -np.inf,
        q25,
        q50,
        q75,
        np.inf,
    ]

    print()
    print("=" * 110)
    print("2017 FN SOURCE CONFIDENCE CUTS")
    print("=" * 110)

    print("Q25 =", q25)
    print("Q50 =", q50)
    print("Q75 =", q75)

    source["confidence_bin"] = (
        assign_source_bins(
            source["abs_stage1_logit"],
            edges,
        )
    )

    future["confidence_bin"] = (
        assign_source_bins(
            future["abs_stage1_logit"],
            edges,
        )
    )

    all_rows = pd.concat(
        [source, future],
        ignore_index=True,
    )

    confidence_rows = []

    for year in (
        2017,
        2018,
        2019,
        2020,
        2021,
    ):

        y = all_rows[
            all_rows["year"] == year
        ]

        for b in (
            "Q1_low",
            "Q2",
            "Q3",
            "Q4_high",
        ):

            g = y[
                y["confidence_bin"] == b
            ]

            if len(g) == 0:
                continue

            confidence_rows.append({
                "year": year,
                "confidence_bin": b,
                "n": len(g),

                "mean_abs_stage1_logit":
                    float(
                        g[
                            "abs_stage1_logit"
                        ].mean()
                    ),

                "wrong_direction":
                    int(
                        g[
                            "wrong_direction"
                        ].sum()
                    ),

                "wrong_direction_rate":
                    float(
                        g[
                            "wrong_direction"
                        ].mean()
                    ),

                "fixed":
                    int(
                        g["fixed"].sum()
                    ),

                "fix_rate":
                    float(
                        g["fixed"].mean()
                    ),

                "mean_delta":
                    float(
                        g[
                            "effective_delta"
                        ].mean()
                    ),
            })

    confidence = pd.DataFrame(
        confidence_rows
    )

    confidence.to_csv(
        OUT / "fn_by_source_confidence_bin.csv",
        index=False,
    )

    print()
    print("=" * 110)
    print("FN WRONG-DIRECTION RATE BY SOURCE CONFIDENCE BIN")
    print("=" * 110)

    print(
        confidence.pivot(
            index="year",
            columns="confidence_bin",
            values="wrong_direction_rate",
        )
        .to_string(
            float_format=lambda x:
                f"{x:.3f}"
        )
    )

    print()
    print("=" * 110)
    print("FN MEAN DELTA BY SOURCE CONFIDENCE BIN")
    print("=" * 110)

    print(
        confidence.pivot(
            index="year",
            columns="confidence_bin",
            values="mean_delta",
        )
        .to_string(
            float_format=lambda x:
                f"{x:.3f}"
        )
    )

    # -------------------------------------------------
    # 2. Feature drift relative to 2017 routed FN
    # -------------------------------------------------

    drift_rows = []

    for feature in FEATURES:

        src = (
            source[feature]
            .to_numpy(float)
        )

        src_mean = float(
            np.mean(src)
        )

        src_sd = float(
            np.std(
                src,
                ddof=1,
            )
        )

        for year in (
            2018,
            2019,
            2020,
            2021,
        ):

            g = future[
                future["year"] == year
            ][feature].to_numpy(float)

            if len(g) == 0:
                continue

            mean = float(
                np.mean(g)
            )

            shift_z = (
                (mean - src_mean)
                / max(src_sd, 1e-12)
            )

            drift_rows.append({
                "feature": feature,
                "year": year,
                "n": len(g),
                "source_mean": src_mean,
                "future_mean": mean,
                "source_sd": src_sd,
                "standardized_shift":
                    float(shift_z),
                "abs_standardized_shift":
                    float(abs(shift_z)),
            })

    drift = pd.DataFrame(
        drift_rows
    )

    drift.to_csv(
        OUT / "fn_feature_drift_vs_2017.csv",
        index=False,
    )

    print()
    print("=" * 110)
    print("TOP FN FEATURE SHIFTS VS 2017")
    print("=" * 110)

    for year in (
        2018,
        2019,
        2020,
        2021,
    ):

        print()
        print("YEAR", year)

        q = (
            drift[
                drift["year"] == year
            ]
            .sort_values(
                "abs_standardized_shift",
                ascending=False,
            )
            .head(10)
        )

        print(
            q[
                [
                    "feature",
                    "source_mean",
                    "future_mean",
                    "standardized_shift",
                ]
            ]
            .to_string(
                index=False,
                float_format=lambda x:
                    f"{x:.4f}",
            )
        )

    # -------------------------------------------------
    # 3. Relation between Stage-I logit and delta
    # -------------------------------------------------

    relation_rows = []

    for year in (
        2017,
        2018,
        2019,
        2020,
        2021,
    ):

        g = all_rows[
            all_rows["year"] == year
        ]

        x = (
            g["stage1_logit"]
            .to_numpy(float)
        )

        y = (
            g["effective_delta"]
            .to_numpy(float)
        )

        if (
            len(g) >= 3
            and np.std(x) > 0
            and np.std(y) > 0
        ):
            corr = float(
                np.corrcoef(
                    x,
                    y,
                )[0, 1]
            )
        else:
            corr = np.nan

        relation_rows.append({
            "year": year,
            "n": len(g),
            "mean_stage1_logit":
                float(np.mean(x)),
            "mean_delta":
                float(np.mean(y)),
            "corr_stage1_logit_delta":
                corr,
            "wrong_direction_rate":
                float(
                    g[
                        "wrong_direction"
                    ].mean()
                ),
        })

    relation = pd.DataFrame(
        relation_rows
    )

    relation.to_csv(
        OUT / "fn_logit_delta_relation.csv",
        index=False,
    )

    print()
    print("=" * 110)
    print("FN STAGE-I LOGIT vs CORRECTION DELTA")
    print("=" * 110)

    print(
        relation.to_string(
            index=False,
            float_format=lambda x:
                f"{x:.4f}",
        )
    )

    source.to_csv(
        OUT / "source_2017_routed_fn.csv",
        index=False,
    )

    future.to_csv(
        OUT / "future_routed_fn.csv",
        index=False,
    )

    print()
    print("saved:", OUT)


if __name__ == "__main__":
    main()
