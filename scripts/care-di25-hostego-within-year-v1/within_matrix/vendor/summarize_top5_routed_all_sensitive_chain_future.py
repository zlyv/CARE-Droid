#!/usr/bin/env python3

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    roc_auc_score,
)


OUTPUT = Path(
    "artifacts/p59-top5-routed-all-sensitive-chain-future"
)

PREDICTIONS = (
    OUTPUT
    / "predictions.csv"
)

FEATURES = Path(
    "artifacts/p1-features/features.csv"
)

TOP5 = (
    "background_service_control",
    "location_access",
    "camera_access",
    "call_control_or_recording",
    "microphone_access",
)

ROUTE_COLUMNS = tuple(
    f"{prefix}__{role}"
    for prefix in (
        "role_node",
        "role_edge",
    )
    for role in TOP5
)

RAW_VARIANT = (
    "raw_count70_only"
)

DIRECT_VARIANTS = {
    "all_apk_shared_caller":
        "raw_count70_plus_shared_caller",
}

SELECTIVE_VARIANTS = {
    "top5_selective_shared_caller":
        "raw_count70_plus_top5_shared_caller",

    "top5_selective_all_sensitive_chain":
        "raw_count70_plus_top5_all_sensitive_chain",

    "top5_selective_shared_and_all_sensitive_chain":
        "raw_count70_plus_top5_shared_and_all_sensitive_chain",
}


def norm_sha(value):
    return str(
        value
    ).strip().lower()


def route_table():
    features = pd.read_csv(
        FEATURES,
        dtype={
            "sha256": str,
        },
        low_memory=False,
    )

    features["sha256"] = (
        features["sha256"]
        .map(norm_sha)
    )

    missing = (
        set(ROUTE_COLUMNS)
        - set(features.columns)
    )

    if missing:
        raise RuntimeError(
            "missing route columns: "
            + repr(
                sorted(missing)
            )
        )

    values = (
        features[
            list(ROUTE_COLUMNS)
        ]
        .apply(
            pd.to_numeric,
            errors="coerce",
        )
        .fillna(0.0)
    )

    features["top5_routed"] = (
        values.abs().sum(
            axis=1
        ) > 0
    )

    result = (
        features[
            [
                "sha256",
                "top5_routed",
            ]
        ]
        .drop_duplicates(
            "sha256"
        )
    )

    if result["sha256"].duplicated().any():
        raise RuntimeError(
            "duplicate feature SHA"
        )

    return result


def score(
    labels,
    probability,
):
    labels = np.asarray(
        labels,
        dtype=int,
    )

    probability = np.asarray(
        probability,
        dtype=float,
    )

    prediction = (
        probability >= 0.5
    ).astype(int)

    tn, fp, fn, tp = (
        confusion_matrix(
            labels,
            prediction,
            labels=[
                0,
                1,
            ],
        )
        .ravel()
    )

    return {
        "ACC":
            accuracy_score(
                labels,
                prediction,
            ),

        "BA":
            balanced_accuracy_score(
                labels,
                prediction,
            ),

        "AUC":
            roc_auc_score(
                labels,
                probability,
            ),

        "TN":
            int(tn),

        "FP":
            int(fp),

        "FN":
            int(fn),

        "TP":
            int(tp),
    }


def main():
    predictions = pd.read_csv(
        PREDICTIONS,
        dtype={
            "sha256": str,
        },
    )

    predictions["sha256"] = (
        predictions["sha256"]
        .map(norm_sha)
    )

    routes = route_table()

    predictions = predictions.merge(
        routes,
        on="sha256",
        how="left",
        validate="many_to_one",
    )

    if predictions["top5_routed"].isna().any():
        raise RuntimeError(
            "prediction APK missing route features"
        )

    predictions["top5_routed"] = (
        predictions["top5_routed"]
        .astype(bool)
    )

    keys = [
        "year",
        "seed",
        "sha256",
    ]

    raw = predictions[
        predictions["variant"]
        == RAW_VARIANT
    ].copy()

    if raw.empty:
        raise RuntimeError(
            "raw variant is missing"
        )

    raw = raw.rename(
        columns={
            "probability":
                "raw_probability",
        }
    )

    raw_columns = (
        keys
        + [
            "label",
            "raw_probability",
            "top5_routed",
        ]
    )

    raw = raw[
        raw_columns
    ]

    output_rows = []

    def append_direct(
        scheme,
        variant,
    ):
        part = predictions[
            predictions["variant"]
            == variant
        ].copy()

        if part.empty:
            raise RuntimeError(
                "missing variant: "
                + variant
            )

        for row in part.itertuples(
            index=False
        ):
            output_rows.append({
                "scheme":
                    scheme,

                "year":
                    int(row.year),

                "seed":
                    int(row.seed),

                "sha256":
                    row.sha256,

                "label":
                    int(row.label),

                "top5_routed":
                    int(
                        row.top5_routed
                    ),

                "raw_probability":
                    np.nan,

                "candidate_probability":
                    float(
                        row.probability
                    ),

                "probability":
                    float(
                        row.probability
                    ),
            })

    append_direct(
        "raw_count70_only",
        RAW_VARIANT,
    )

    for scheme, variant in (
        DIRECT_VARIANTS.items()
    ):
        append_direct(
            scheme,
            variant,
        )

    for scheme, variant in (
        SELECTIVE_VARIANTS.items()
    ):
        candidate = predictions[
            predictions["variant"]
            == variant
        ][
            keys
            + [
                "label",
                "probability",
            ]
        ].copy()

        if candidate.empty:
            raise RuntimeError(
                "missing variant: "
                + variant
            )

        candidate = candidate.rename(
            columns={
                "probability":
                    "candidate_probability",
            }
        )

        merged = raw.merge(
            candidate,
            on=keys,
            how="inner",
            validate="one_to_one",
            suffixes=(
                "",
                "_candidate",
            ),
        )

        if len(merged) != len(raw):
            raise RuntimeError(
                "candidate/raw row mismatch: "
                + variant
            )

        if not np.array_equal(
            merged["label"].to_numpy(),
            merged[
                "label_candidate"
            ].to_numpy(),
        ):
            raise RuntimeError(
                "candidate/raw label mismatch"
            )

        merged["probability"] = np.where(
            merged["top5_routed"],
            merged[
                "candidate_probability"
            ],
            merged[
                "raw_probability"
            ],
        )

        for row in merged.itertuples(
            index=False
        ):
            output_rows.append({
                "scheme":
                    scheme,

                "year":
                    int(row.year),

                "seed":
                    int(row.seed),

                "sha256":
                    row.sha256,

                "label":
                    int(row.label),

                "top5_routed":
                    int(
                        row.top5_routed
                    ),

                "raw_probability":
                    float(
                        row.raw_probability
                    ),

                "candidate_probability":
                    float(
                        row.candidate_probability
                    ),

                "probability":
                    float(
                        row.probability
                    ),
            })

    output = pd.DataFrame(
        output_rows
    )

    output["prediction"] = (
        output["probability"]
        >= 0.5
    ).astype(int)

    output.to_csv(
        OUTPUT
        / "selective_predictions.csv",
        index=False,
    )

    seed_rows = []

    for (
        scheme,
        year,
        seed,
    ), part in output.groupby(
        [
            "scheme",
            "year",
            "seed",
        ],
        sort=False,
    ):
        seed_rows.append({
            "scheme":
                scheme,

            "year":
                int(year),

            "seed":
                int(seed),

            "apk_count":
                part[
                    "sha256"
                ].nunique(),

            "routed_apks":
                part.loc[
                    part[
                        "top5_routed"
                    ] == 1,
                    "sha256",
                ].nunique(),

            "route_coverage":
                part[
                    "top5_routed"
                ].mean(),

            **score(
                part["label"],
                part["probability"],
            ),
        })

    per_seed = pd.DataFrame(
        seed_rows
    )

    per_seed.to_csv(
        OUTPUT
        / "selective_per_seed.csv",
        index=False,
    )

    per_year = (
        per_seed.groupby(
            [
                "scheme",
                "year",
            ],
            as_index=False,
            sort=False,
        )
        .agg(
            apk_count=(
                "apk_count",
                "first",
            ),

            routed_apks=(
                "routed_apks",
                "first",
            ),

            route_coverage=(
                "route_coverage",
                "first",
            ),

            BA=(
                "BA",
                "mean",
            ),

            ACC=(
                "ACC",
                "mean",
            ),

            AUC=(
                "AUC",
                "mean",
            ),

            FP=(
                "FP",
                "mean",
            ),

            FN=(
                "FN",
                "mean",
            ),
        )
    )

    per_year.to_csv(
        OUTPUT
        / "selective_per_year.csv",
        index=False,
    )

    summary = (
        per_year.groupby(
            "scheme",
            as_index=False,
            sort=False,
        )
        .agg(
            future_years=(
                "year",
                "nunique",
            ),

            mean_BA=(
                "BA",
                "mean",
            ),

            min_BA=(
                "BA",
                "min",
            ),

            std_BA=(
                "BA",
                "std",
            ),

            mean_ACC=(
                "ACC",
                "mean",
            ),

            mean_AUC=(
                "AUC",
                "mean",
            ),

            min_AUC=(
                "AUC",
                "min",
            ),

            mean_FP=(
                "FP",
                "mean",
            ),

            mean_FN=(
                "FN",
                "mean",
            ),

            mean_route_coverage=(
                "route_coverage",
                "mean",
            ),
        )
    )

    raw_ba = float(
        summary.loc[
            summary["scheme"]
            == "raw_count70_only",
            "mean_BA",
        ].iloc[0]
    )

    summary[
        "delta_BA_vs_raw"
    ] = (
        summary["mean_BA"]
        - raw_ba
    )

    summary.to_csv(
        OUTPUT
        / "selective_summary.csv",
        index=False,
    )

    print(
        "=" * 120
    )

    print(
        "TOP-5 ROUTED ALL-SENSITIVE DIRECTED CHAIN SUMMARY"
    )

    print(
        "=" * 120
    )

    print(
        summary.to_string(
            index=False
        )
    )

    print(
        "\nPER YEAR"
    )

    print(
        per_year.to_string(
            index=False
        )
    )


if __name__ == "__main__":
    main()
