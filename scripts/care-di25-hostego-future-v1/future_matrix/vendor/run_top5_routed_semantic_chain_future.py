#!/usr/bin/env python3

from __future__ import annotations

import csv
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent.parent

sys.path.insert(
    0,
    str(ROOT / "scripts"),
)

import run_compact_quantity_shared_caller_2x3_2017_eval as exp
import run_endpoint_behavior_transition_2017_eval as chain_exp


FEATURES = (
    ROOT
    / "artifacts/p1-features/features.csv"
)

SCHEMA = (
    ROOT
    / "artifacts/p1-features/feature_schema.json"
)

SPLITS = (
    ROOT
    / "artifacts/p9-2017-rolling-splits/split_manifest.csv"
)

ENDPOINT_CACHE = (
    ROOT
    / "artifacts/p25-endpoint-cache-full-cap256"
    / "endpoint_centered_tensor_cache.jsonl.gz"
)

FUTURE_PREDICTIONS = (
    ROOT
    / "artifacts/allratio70-compact-2017-2021"
    / "future_predictions.csv"
)

OUTPUT = (
    ROOT
    / "artifacts/p57-top5-routed-semantic-chain-future"
)

YEARS = (
    2018,
    2019,
    2020,
    2021,
)

SEEDS = (
    0,
    1,
    2,
    3,
    4,
)

ITERATIONS = 1500
LEARNING_RATE = 0.05
L2 = 0.03
THRESHOLD = 0.5

EXPECTED_SOURCE_COUNT = 799

EXPECTED_FUTURE_COUNTS = {
    2018: 789,
    2019: 399,
    2020: 397,
    2021: 400,
}


def norm_sha(value):
    return str(value).strip().casefold()


def sha256_file(path):
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        while True:
            block = handle.read(
                1024 * 1024
            )

            if not block:
                break

            digest.update(
                block
            )

    return digest.hexdigest()


def write_csv(
    path,
    rows,
):
    if not rows:
        raise RuntimeError(
            "refusing to write empty CSV: "
            + str(path)
        )

    fieldnames = []

    for row in rows:
        for name in row:
            if name not in fieldnames:
                fieldnames.append(
                    name
                )

    with path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        for row in rows:
            writer.writerow(
                {
                    name:
                        row.get(
                            name,
                            "",
                        )
                    for name in fieldnames
                }
            )


def load_source_rows():
    # ========================================================
    # 正式未来实验使用完整2017源域。
    #
    # 三个滚动划分的并集只有794个APK，因为滚动协议会排除
    # 少量不进入任何有效训练/验证窗口的记录；但此前所有正式
    # 2017→未来实验的完整源域训练集合都是799个APK。
    #
    # 因此这里严格按：
    #     metadata_year == 2017
    # 从冻结的P1特征表构造完整源域。
    # ========================================================

    feature_rows = exp.read_csv_rows(
        FEATURES
    )

    source_by_sha = {}

    def parse_label(
        value,
        sha,
    ):
        try:
            label = int(
                float(
                    value
                )
            )

        except (
            TypeError,
            ValueError,
        ):
            normalized = (
                str(value)
                .strip()
                .casefold()
            )

            if normalized in {
                "benign",
                "goodware",
                "false",
            }:
                label = 0

            elif normalized in {
                "malware",
                "malicious",
                "true",
            }:
                label = 1

            else:
                raise RuntimeError(
                    "cannot parse metadata_label "
                    + repr(value)
                    + " for "
                    + sha
                )

        if label not in {
            0,
            1,
        }:
            raise RuntimeError(
                "invalid source label "
                + repr(label)
                + " for "
                + sha
            )

        return label

    for feature_row in feature_rows:
        try:
            year = int(
                float(
                    feature_row.get(
                        "metadata_year",
                        "",
                    )
                )
            )
        except (
            TypeError,
            ValueError,
        ):
            continue

        if year != 2017:
            continue

        sha = norm_sha(
            feature_row[
                "sha256"
            ]
        )

        label_value = feature_row.get(
            "metadata_label",
            "",
        )

        if str(
            label_value
        ).strip() == "":
            raise RuntimeError(
                "missing metadata_label for "
                + sha
            )

        label = parse_label(
            label_value,
            sha,
        )

        row = {
            "sha256":
                sha,

            "label":
                label,
        }

        if sha in source_by_sha:
            previous = int(
                source_by_sha[
                    sha
                ][
                    "label"
                ]
            )

            if previous != label:
                raise RuntimeError(
                    "inconsistent 2017 source label: "
                    + sha
                )

        source_by_sha[
            sha
        ] = row

    rows = [
        source_by_sha[
            sha
        ]
        for sha in sorted(
            source_by_sha
        )
    ]

    if len(rows) != EXPECTED_SOURCE_COUNT:
        raise RuntimeError(
            "expected "
            + str(
                EXPECTED_SOURCE_COUNT
            )
            + " complete 2017 source APKs, got "
            + str(
                len(rows)
            )
        )

    label_counts = {
        0:
            sum(
                int(
                    row[
                        "label"
                    ]
                ) == 0
                for row in rows
            ),

        1:
            sum(
                int(
                    row[
                        "label"
                    ]
                ) == 1
                for row in rows
            ),
    }

    print(
        "complete 2017 source label counts =",
        label_counts,
    )

    return rows


def load_future_rows():
    original_rows = exp.read_csv_rows(
        FUTURE_PREDICTIONS
    )

    labels = {}
    compact_probability = {}
    seed_coverage = defaultdict(
        set
    )

    for row in original_rows:
        year = int(
            row["year"]
        )

        if year not in YEARS:
            continue

        sha = norm_sha(
            row["sha256"]
        )

        seed = int(
            row["seed"]
        )

        label = int(
            row["label"]
        )

        probability = float(
            row["base_probability"]
        )

        key = (
            year,
            sha,
        )

        if (
            key in labels
            and labels[key] != label
        ):
            raise RuntimeError(
                "inconsistent future label: "
                + repr(key)
            )

        labels[
            key
        ] = label

        probability_key = (
            year,
            seed,
            sha,
        )

        if probability_key in compact_probability:
            raise RuntimeError(
                "duplicate future prediction: "
                + repr(
                    probability_key
                )
            )

        compact_probability[
            probability_key
        ] = probability

        seed_coverage[
            (
                year,
                sha,
            )
        ].add(
            seed
        )

    expected_seeds = set(
        SEEDS
    )

    for key, available in seed_coverage.items():
        if available != expected_seeds:
            raise RuntimeError(
                "incomplete seed coverage for "
                + repr(key)
                + ": "
                + repr(
                    sorted(
                        available
                    )
                )
            )

    future_rows = []

    for (
        year,
        sha,
    ) in sorted(
        labels
    ):
        # 未来标签不传给特征构造。
        future_rows.append(
            {
                "year":
                    year,

                "sha256":
                    sha,

                "label":
                    0,
            }
        )

    counts = defaultdict(
        int
    )

    for row in future_rows:
        counts[
            int(
                row["year"]
            )
        ] += 1

    if dict(counts) != EXPECTED_FUTURE_COUNTS:
        raise RuntimeError(
            "unexpected future cohort counts: "
            + repr(
                dict(counts)
            )
        )

    actual_labels = np.asarray(
        [
            labels[
                (
                    int(
                        row["year"]
                    ),
                    row["sha256"],
                )
            ]
            for row in future_rows
        ],
        dtype=np.int64,
    )

    return (
        future_rows,
        actual_labels,
        compact_probability,
    )


def score_predictions(
    labels,
    probability,
):
    labels = np.asarray(
        labels,
        dtype=np.int64,
    )

    probability = np.asarray(
        probability,
        dtype=np.float64,
    )

    values = exp.metrics(
        labels,
        probability,
        THRESHOLD,
    )

    prediction = (
        probability
        >= THRESHOLD
    ).astype(
        np.int64
    )

    fp = int(
        np.sum(
            (labels == 0)
            & (prediction == 1)
        )
    )

    fn = int(
        np.sum(
            (labels == 1)
            & (prediction == 0)
        )
    )

    result = {
        "accuracy":
            float(
                values[
                    "accuracy"
                ]
            ),

        "balanced_accuracy":
            float(
                values[
                    "balanced_accuracy"
                ]
            ),

        "precision":
            float(
                values[
                    "precision"
                ]
            ),

        "recall":
            float(
                values[
                    "recall"
                ]
            ),

        "f1":
            float(
                values[
                    "f1"
                ]
            ),

        "auc":
            float(
                values[
                    "auc"
                ]
            ),

        "FP":
            fp,

        "FN":
            fn,
    }

    return (
        result,
        prediction,
    )



# ============================================================
# TOP-5 routing
#
# Routing uses only direct node/edge evidence from the
# first-layer static feature table.  context_role is excluded
# because it is neighbourhood evidence rather than proof that
# the APK itself contains the TOP-5 endpoint.
# ============================================================

TOP5_ROLES = (
    "background_service_control",
    "location_access",
    "camera_access",
    "call_control_or_recording",
    "microphone_access",
)

TOP5_ROUTE_COLUMNS = tuple(
    f"{prefix}__{role}"
    for prefix in (
        "role_node",
        "role_edge",
    )
    for role in TOP5_ROLES
)


def safe_float(value):
    try:
        result = float(value)
        if np.isfinite(result):
            return result
    except (TypeError, ValueError):
        pass
    return 0.0


def top5_route_mask(
    rows,
    feature_by_sha,
):
    values = []

    for item in rows:
        sha = norm_sha(
            item["sha256"]
        )

        feature = feature_by_sha.get(
            sha
        )

        if feature is None:
            raise RuntimeError(
                "missing first-layer features for "
                + sha
            )

        routed = any(
            safe_float(
                feature.get(
                    column,
                    0.0,
                )
            ) > 0.0
            for column in TOP5_ROUTE_COLUMNS
        )

        values.append(
            routed
        )

    return np.asarray(
        values,
        dtype=bool,
    )


def standardize_routed_pair(
    source,
    future,
    source_route,
    future_route,
):
    source = np.asarray(
        source,
        dtype=np.float64,
    )

    future = np.asarray(
        future,
        dtype=np.float64,
    )

    if not np.any(source_route):
        raise RuntimeError(
            "no routed 2017 APKs"
        )

    # Scaling is fitted using routed 2017 APKs only.
    fit = source[
        source_route
    ]

    mean = fit.mean(
        axis=0
    )

    scale = fit.std(
        axis=0
    )

    scale[
        scale < 1e-12
    ] = 1.0

    source_result = (
        source - mean
    ) / scale

    future_result = (
        future - mean
    ) / scale

    # Non-routed APKs have no semantic representation.
    source_result[
        ~source_route
    ] = 0.0

    future_result[
        ~future_route
    ] = 0.0

    return (
        source_result,
        future_result,
    )


def print_route_summary(
    source_rows,
    future_rows,
    source_route,
    future_route,
):
    print(
        "TOP-5 route columns =",
        TOP5_ROUTE_COLUMNS,
        flush=True,
    )

    print(
        "2017 routed APKs =",
        int(source_route.sum()),
        "/",
        len(source_route),
        flush=True,
    )

    years = sorted({
        int(row["year"])
        for row in future_rows
    })

    for year in years:
        indices = np.asarray(
            [
                int(row["year"]) == year
                for row in future_rows
            ],
            dtype=bool,
        )

        count = int(
            indices.sum()
        )

        routed = int(
            future_route[
                indices
            ].sum()
        )

        print(
            year,
            "routed APKs =",
            routed,
            "/",
            count,
            "coverage =",
            routed / count
            if count
            else 0.0,
            flush=True,
        )

def main():
    OUTPUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        "=" * 88
    )

    print(
        "TOP-5 ROUTED SEMANTIC CALL-CHAIN FUTURE EVALUATION"
    )

    print(
        "=" * 88
    )

    print(
        "training years: 2017 only"
    )

    print(
        "future years:",
        YEARS,
    )

    print(
        "seeds:",
        SEEDS,
    )

    print(
        "iterations:",
        ITERATIONS,
    )

    # ========================================================
    # 1. Cohorts
    # ========================================================

    source_rows = load_source_rows()

    (
        future_rows,
        future_labels,
        frozen_compact_probability,
    ) = load_future_rows()

    print(
        "source APKs =",
        len(
            source_rows
        ),
    )

    for year in YEARS:
        count = sum(
            int(
                row["year"]
            ) == year
            for row in future_rows
        )

        print(
            year,
            "future APKs =",
            count,
        )

    # ========================================================
    # 2. Features and caches
    # ========================================================

    feature_rows = exp.read_csv_rows(
        FEATURES
    )

    feature_by_sha = {
        norm_sha(
            row["sha256"]
        ):
            row
        for row in feature_rows
    }

    required_shas = {
        row["sha256"]
        for row in source_rows
    } | {
        row["sha256"]
        for row in future_rows
    }

    missing_features = (
        required_shas
        - set(
            feature_by_sha
        )
    )

    if missing_features:
        raise RuntimeError(
            "missing feature rows: "
            + repr(
                sorted(
                    missing_features
                )[:20]
            )
        )

    with SCHEMA.open(
        "r",
        encoding="utf-8",
    ) as handle:
        schema = json.load(
            handle
        )

    endpoint_cache = exp.read_cache(
        ENDPOINT_CACHE
    )

    endpoint_shas = {
        norm_sha(
            sha
        )
        for sha in endpoint_cache
    }

    missing_endpoint = (
        required_shas
        - endpoint_shas
    )

    if missing_endpoint:
        raise RuntimeError(
            "missing endpoint cache rows: "
            + repr(
                sorted(
                    missing_endpoint
                )[:20]
            )
        )

    print(
        "feature coverage =",
        len(
            required_shas
        ),
        "/",
        len(
            required_shas
        ),
    )

    print(
        "endpoint coverage =",
        len(
            required_shas
        ),
        "/",
        len(
            required_shas
        ),
    )

    # ========================================================
    # 3. Matched Compact representations
    #
    # All means/scales are fitted using the 2017 source rows.
    # ========================================================

    (
        compact_modes,
        compact_names,
    ) = exp.compact_quantity_matrices(
        source_rows,
        future_rows,
        feature_by_sha,
        schema,
    )

    if len(
        compact_names
    ) != 70:
        raise RuntimeError(
            "unexpected Compact dimension"
        )

    # ========================================================
    # TOP-5 route is decided before semantic extraction.
    # No labels, predictions or future outcomes are used.
    # ========================================================

    source_route = top5_route_mask(
        source_rows,
        feature_by_sha,
    )

    future_route = top5_route_mask(
        future_rows,
        feature_by_sha,
    )

    print_route_summary(
        source_rows,
        future_rows,
        source_route,
        future_route,
    )

    # ========================================================
    # 4. Shared-caller representation
    #
    # Build all vectors without future labels.
    # Standardization is fitted using 2017 only.
    # ========================================================

    (
        hyper_source,
        source_hypergraph_audit,
    ) = exp.hypergraph_matrix(
        source_rows,
        endpoint_cache,
    )

    (
        hyper_future,
        future_hypergraph_audit,
    ) = exp.hypergraph_matrix(
        future_rows,
        endpoint_cache,
    )

    # hypergraph_matrix的第二个返回值是逐APK审计记录，
    # 不是标签。训练标签严格来自完整2017 source_rows。
    source_labels = np.asarray(
        [
            int(
                row["label"]
            )
            for row in source_rows
        ],
        dtype=np.int64,
    )

    if len(
        source_hypergraph_audit
    ) != len(
        source_rows
    ):
        raise RuntimeError(
            "source hypergraph audit length mismatch: "
            + str(
                len(
                    source_hypergraph_audit
                )
            )
            + " vs "
            + str(
                len(
                    source_rows
                )
            )
        )

    if len(
        future_hypergraph_audit
    ) != len(
        future_rows
    ):
        raise RuntimeError(
            "future hypergraph audit length mismatch: "
            + str(
                len(
                    future_hypergraph_audit
                )
            )
            + " vs "
            + str(
                len(
                    future_rows
                )
            )
        )

    if hyper_source.shape[0] != len(
        source_rows
    ):
        raise RuntimeError(
            "source hypergraph matrix row mismatch"
        )

    if hyper_future.shape[0] != len(
        future_rows
    ):
        raise RuntimeError(
            "future hypergraph matrix row mismatch"
        )

    # Preserve raw hypergraph matrices because we need
    # both the original all-APK control and TOP-5 routing.
    hyper_source_raw = np.asarray(
        hyper_source,
        dtype=np.float64,
    )

    hyper_future_raw = np.asarray(
        hyper_future,
        dtype=np.float64,
    )

    (
        hyper_source,
        hyper_future,
    ) = exp.standardize_pair(
        hyper_source_raw,
        hyper_future_raw,
    )

    shared_source = hyper_source[
        :,
        :exp.RELATION_DIMENSION,
    ]

    shared_future = hyper_future[
        :,
        :exp.RELATION_DIMENSION,
    ]

    (
        routed_hyper_source,
        routed_hyper_future,
    ) = standardize_routed_pair(
        hyper_source_raw,
        hyper_future_raw,
        source_route,
        future_route,
    )

    routed_shared_source = routed_hyper_source[
        :,
        :exp.RELATION_DIMENSION,
    ]

    routed_shared_future = routed_hyper_future[
        :,
        :exp.RELATION_DIMENSION,
    ]

    if shared_source.shape[1] != 120:
        raise RuntimeError(
            "unexpected shared-caller dimension: "
            + str(
                shared_source.shape
            )
        )

    # ========================================================
    # Ordered semantic call-chain representation
    #
    # P42 contains:
    #   caller/endpoint histograms,
    #   shared-caller pair matrix,
    #   ordered collapsed-path pair matrix,
    #   path-length histogram and audit values.
    #
    # This experiment keeps only:
    #   ordered semantic start-role -> end-role pairs
    #   and the 1/2/3/4-hop path-length histogram.
    #
    # Ordinary methods are therefore used to establish
    # reachability but are not retained as model features.
    # ========================================================

    (
        chain_source_raw_full,
        source_chain_audit,
    ) = chain_exp.behavior_matrix(
        source_rows,
        endpoint_cache,
    )

    (
        chain_future_raw_full,
        future_chain_audit,
    ) = chain_exp.behavior_matrix(
        future_rows,
        endpoint_cache,
    )

    role_count = int(
        chain_exp.ROLE_COUNT
    )

    prefix_dimension = (
        len(
            chain_exp.STAGES
        )
        * role_count
        + role_count
    )

    shared_pair_dimension = (
        role_count
        * role_count
    )

    collapsed_start = (
        prefix_dimension
        + shared_pair_dimension
    )

    collapsed_end = (
        collapsed_start
        + shared_pair_dimension
    )

    length_end = (
        collapsed_end
        + int(
            chain_exp.MAX_COLLAPSED_PATH_HOPS
        )
    )

    chain_source_raw = (
        chain_source_raw_full[
            :,
            collapsed_start:length_end,
        ]
    )

    chain_future_raw = (
        chain_future_raw_full[
            :,
            collapsed_start:length_end,
        ]
    )

    expected_chain_dimension = (
        role_count
        * role_count
        + int(
            chain_exp.MAX_COLLAPSED_PATH_HOPS
        )
    )

    if (
        chain_source_raw.shape[1]
        != expected_chain_dimension
    ):
        raise RuntimeError(
            "unexpected semantic-chain dimension: "
            + repr(
                chain_source_raw.shape
            )
        )

    # All-APK semantic-chain control.
    (
        chain_source,
        chain_future,
    ) = exp.standardize_pair(
        chain_source_raw,
        chain_future_raw,
    )

    # Proposed TOP-5-routed semantic chain.
    (
        routed_chain_source,
        routed_chain_future,
    ) = standardize_routed_pair(
        chain_source_raw,
        chain_future_raw,
        source_route,
        future_route,
    )

    print(
        "semantic-chain dimensions =",
        expected_chain_dimension,
        flush=True,
    )

    print(
        "semantic-chain source audit rows =",
        len(
            source_chain_audit
        ),
        flush=True,
    )

    print(
        "semantic-chain future audit rows =",
        len(
            future_chain_audit
        ),
        flush=True,
    )

    # ========================================================
    # 5. Freeze evaluated variants
    # ========================================================

    raw_source, raw_future = (
        compact_modes[
            "raw_count70"
        ]
    )

    ratio_source, ratio_future = (
        compact_modes[
            "allratio70"
        ]
    )

    matrices = {
        "raw_count70_only": (
            raw_source,
            raw_future,
        ),

        "raw_count70_plus_shared_caller": (
            np.concatenate(
                [
                    raw_source,
                    shared_source,
                ],
                axis=1,
            ),
            np.concatenate(
                [
                    raw_future,
                    shared_future,
                ],
                axis=1,
            ),
        ),

        # ------------------------------------------------
        # Controls: semantic processing for every APK.
        # ------------------------------------------------

        "raw_count70_plus_semantic_chain": (
            np.concatenate(
                [
                    raw_source,
                    chain_source,
                ],
                axis=1,
            ),
            np.concatenate(
                [
                    raw_future,
                    chain_future,
                ],
                axis=1,
            ),
        ),

        "raw_count70_plus_shared_and_semantic_chain": (
            np.concatenate(
                [
                    raw_source,
                    shared_source,
                    chain_source,
                ],
                axis=1,
            ),
            np.concatenate(
                [
                    raw_future,
                    shared_future,
                    chain_future,
                ],
                axis=1,
            ),
        ),

        # ------------------------------------------------
        # Proposed method:
        # semantic blocks are exactly zero for non-routed
        # APKs.  Their final prediction will additionally be
        # restored to raw_count70_only by the selective audit.
        # ------------------------------------------------

        "raw_count70_plus_top5_shared_caller": (
            np.concatenate(
                [
                    raw_source,
                    routed_shared_source,
                ],
                axis=1,
            ),
            np.concatenate(
                [
                    raw_future,
                    routed_shared_future,
                ],
                axis=1,
            ),
        ),

        "raw_count70_plus_top5_semantic_chain": (
            np.concatenate(
                [
                    raw_source,
                    routed_chain_source,
                ],
                axis=1,
            ),
            np.concatenate(
                [
                    raw_future,
                    routed_chain_future,
                ],
                axis=1,
            ),
        ),

        "raw_count70_plus_top5_shared_and_semantic_chain": (
            np.concatenate(
                [
                    raw_source,
                    routed_shared_source,
                    routed_chain_source,
                ],
                axis=1,
            ),
            np.concatenate(
                [
                    raw_future,
                    routed_shared_future,
                    routed_chain_future,
                ],
                axis=1,
            ),
        ),

        "allratio70_only": (
            ratio_source,
            ratio_future,
        ),

        "allratio70_plus_shared_caller": (
            np.concatenate(
                [
                    ratio_source,
                    shared_source,
                ],
                axis=1,
            ),
            np.concatenate(
                [
                    ratio_future,
                    shared_future,
                ],
                axis=1,
            ),
        ),
    }

    predictions = []
    per_seed = []

    year_indices = {
        year:
            np.asarray(
                [
                    index
                    for index, row in enumerate(
                        future_rows
                    )
                    if int(
                        row["year"]
                    ) == year
                ],
                dtype=np.int64,
            )
        for year in YEARS
    }

    # ========================================================
    # 6. Train on complete 2017, predict all future APKs
    # ========================================================

    for seed in SEEDS:
        print()
        print(
            "seed =",
            seed,
        )

        probability_by_variant = {}

        for variant, (
            x_source,
            x_future,
        ) in matrices.items():

            probability = exp.predict_logistic(
                x_source,
                source_labels,
                x_future,
                seed,
                ITERATIONS,
                LEARNING_RATE,
                L2,
            )

            probability_by_variant[
                variant
            ] = np.asarray(
                probability,
                dtype=np.float64,
            )

        frozen_probability = np.asarray(
            [
                frozen_compact_probability[
                    (
                        int(
                            row["year"]
                        ),
                        seed,
                        row["sha256"],
                    )
                ]
                for row in future_rows
            ],
            dtype=np.float64,
        )

        probability_by_variant[
            "frozen_allratio70_compact_base"
        ] = frozen_probability

        for variant, probability in (
            probability_by_variant.items()
        ):
            for year in YEARS:
                indices = year_indices[
                    year
                ]

                labels = future_labels[
                    indices
                ]

                values = probability[
                    indices
                ]

                (
                    result,
                    prediction,
                ) = score_predictions(
                    labels,
                    values,
                )

                per_seed.append(
                    {
                        "variant":
                            variant,

                        "year":
                            year,

                        "seed":
                            seed,

                        "n":
                            len(
                                indices
                            ),

                        "accuracy":
                            result[
                                "accuracy"
                            ],

                        "balanced_accuracy":
                            result[
                                "balanced_accuracy"
                            ],

                        "precision":
                            result[
                                "precision"
                            ],

                        "recall":
                            result[
                                "recall"
                            ],

                        "f1":
                            result[
                                "f1"
                            ],

                        "auc":
                            result[
                                "auc"
                            ],

                        "FP":
                            result[
                                "FP"
                            ],

                        "FN":
                            result[
                                "FN"
                            ],
                    }
                )

                print(
                    variant,
                    year,
                    "BA=",
                    f"{result['balanced_accuracy']:.6f}",
                    "AUC=",
                    f"{result['auc']:.6f}",
                    "FP=",
                    result["FP"],
                    "FN=",
                    result["FN"],
                )

                for local_index, global_index in enumerate(
                    indices
                ):
                    row = future_rows[
                        int(
                            global_index
                        )
                    ]

                    predictions.append(
                        {
                            "variant":
                                variant,

                            "year":
                                year,

                            "seed":
                                seed,

                            "sha256":
                                row[
                                    "sha256"
                                ],

                            "label":
                                int(
                                    labels[
                                        local_index
                                    ]
                                ),

                            "probability":
                                float(
                                    values[
                                        local_index
                                    ]
                                ),

                            "prediction":
                                int(
                                    prediction[
                                        local_index
                                    ]
                                ),

                            "threshold":
                                THRESHOLD,
                        }
                    )

    # ========================================================
    # 7. Aggregate seeds within each year
    # ========================================================

    grouped = defaultdict(
        list
    )

    for row in per_seed:
        grouped[
            (
                row["variant"],
                int(
                    row["year"]
                ),
            )
        ].append(
            row
        )

    per_year = []

    metric_fields = (
        "accuracy",
        "balanced_accuracy",
        "precision",
        "recall",
        "f1",
        "auc",
        "FP",
        "FN",
    )

    for (
        variant,
        year,
    ) in sorted(
        grouped
    ):
        rows = grouped[
            (
                variant,
                year,
            )
        ]

        output_row = {
            "variant":
                variant,

            "year":
                year,

            "seeds":
                len(
                    rows
                ),

            "n":
                rows[0][
                    "n"
                ],
        }

        for field in metric_fields:
            output_row[
                field
            ] = float(
                np.mean(
                    [
                        float(
                            row[
                                field
                            ]
                        )
                        for row in rows
                    ]
                )
            )

        per_year.append(
            output_row
        )

    # ========================================================
    # 8. Macro-year summary
    # ========================================================

    rows_by_variant = defaultdict(
        list
    )

    for row in per_year:
        rows_by_variant[
            row["variant"]
        ].append(
            row
        )

    raw_ba_by_year = {
        int(
            row["year"]
        ):
            float(
                row[
                    "balanced_accuracy"
                ]
            )
        for row in rows_by_variant[
            "raw_count70_only"
        ]
    }

    summary = []

    for variant in sorted(
        rows_by_variant
    ):
        rows = sorted(
            rows_by_variant[
                variant
            ],
            key=lambda row:
                int(
                    row["year"]
                ),
        )

        balanced_accuracy = np.asarray(
            [
                float(
                    row[
                        "balanced_accuracy"
                    ]
                )
                for row in rows
            ],
            dtype=np.float64,
        )

        auc = np.asarray(
            [
                float(
                    row[
                        "auc"
                    ]
                )
                for row in rows
            ],
            dtype=np.float64,
        )

        delta_vs_raw = np.asarray(
            [
                float(
                    row[
                        "balanced_accuracy"
                    ]
                )
                - raw_ba_by_year[
                    int(
                        row["year"]
                    )
                ]
                for row in rows
            ],
            dtype=np.float64,
        )

        summary.append(
            {
                "variant":
                    variant,

                "future_years":
                    len(
                        rows
                    ),

                "mean_accuracy":
                    float(
                        np.mean(
                            [
                                row[
                                    "accuracy"
                                ]
                                for row in rows
                            ]
                        )
                    ),

                "mean_balanced_accuracy":
                    float(
                        balanced_accuracy.mean()
                    ),

                "minimum_year_balanced_accuracy":
                    float(
                        balanced_accuracy.min()
                    ),

                "std_year_balanced_accuracy":
                    float(
                        balanced_accuracy.std()
                    ),

                "mean_auc":
                    float(
                        auc.mean()
                    ),

                "minimum_year_auc":
                    float(
                        auc.min()
                    ),

                "mean_FP":
                    float(
                        np.mean(
                            [
                                row[
                                    "FP"
                                ]
                                for row in rows
                            ]
                        )
                    ),

                "mean_FN":
                    float(
                        np.mean(
                            [
                                row[
                                    "FN"
                                ]
                                for row in rows
                            ]
                        )
                    ),

                "mean_delta_BA_vs_raw_count70_only":
                    float(
                        delta_vs_raw.mean()
                    ),

                "minimum_delta_BA_vs_raw_count70_only":
                    float(
                        delta_vs_raw.min()
                    ),
            }
        )

    summary.sort(
        key=lambda row: (
            -row[
                "mean_balanced_accuracy"
            ],
            -row[
                "mean_auc"
            ],
            row[
                "variant"
            ],
        )
    )

    for rank, row in enumerate(
        summary,
        1,
    ):
        row[
            "rank"
        ] = rank

    summary = sorted(
        summary,
        key=lambda row:
            row[
                "rank"
            ],
    )

    # ========================================================
    # 9. Save
    # ========================================================

    write_csv(
        OUTPUT
        / "per_seed.csv",
        per_seed,
    )

    write_csv(
        OUTPUT
        / "per_year.csv",
        per_year,
    )

    write_csv(
        OUTPUT
        / "summary.csv",
        summary,
    )

    write_csv(
        OUTPUT
        / "predictions.csv",
        predictions,
    )

    audit = {
        "phase":
            "P51_RAWCOUNT70_SHARED_CALLER_FUTURE_EXPLORATORY",

        "confirmatory":
            False,

        "reason":
            (
                "2018-2021 results had already been observed "
                "during earlier method development."
            ),

        "training_years":
            [2017],

        "evaluation_years":
            list(
                YEARS
            ),

        "source_apks":
            len(
                source_rows
            ),

        "future_apks":
            {
                str(year):
                    EXPECTED_FUTURE_COUNTS[
                        year
                    ]
                for year in YEARS
            },

        "variants":
            list(
                matrices
            )
            + [
                "frozen_allratio70_compact_base"
            ],

        "compact_dimension":
            70,

        "shared_caller_dimension":
            int(
                exp.RELATION_DIMENSION
            ),

        "seeds":
            list(
                SEEDS
            ),

        "iterations":
            ITERATIONS,

        "learning_rate":
            LEARNING_RATE,

        "l2":
            L2,

        "threshold":
            THRESHOLD,

        "future_labels_used_for_training":
            False,

        "future_features_used_for_standardization":
            False,

        "future_features_used_for_variance_selection":
            False,

        "files": {
            "features":
                str(
                    FEATURES
                ),

            "schema":
                str(
                    SCHEMA
                ),

            "splits":
                str(
                    SPLITS
                ),

            "endpoint_cache":
                str(
                    ENDPOINT_CACHE
                ),

            "future_predictions":
                str(
                    FUTURE_PREDICTIONS
                ),
        },

        "sha256": {
            "schema":
                sha256_file(
                    SCHEMA
                ),

            "splits":
                sha256_file(
                    SPLITS
                ),

            "future_predictions":
                sha256_file(
                    FUTURE_PREDICTIONS
                ),
        },
    }

    with (
        OUTPUT
        / "audit.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            audit,
            handle,
            ensure_ascii=False,
            indent=2,
        )

    print()
    print(
        "=" * 88
    )

    print(
        "PER YEAR"
    )

    print(
        "=" * 88
    )

    for row in per_year:
        print(
            row[
                "variant"
            ],
            row[
                "year"
            ],
            "BA=",
            f"{row['balanced_accuracy']:.6f}",
            "AUC=",
            f"{row['auc']:.6f}",
            "FP=",
            f"{row['FP']:.2f}",
            "FN=",
            f"{row['FN']:.2f}",
        )

    print()
    print(
        "=" * 88
    )

    print(
        "SUMMARY"
    )

    print(
        "=" * 88
    )

    for row in summary:
        print(
            row[
                "rank"
            ],
            row[
                "variant"
            ],
            "mean_BA=",
            f"{row['mean_balanced_accuracy']:.6f}",
            "min_BA=",
            f"{row['minimum_year_balanced_accuracy']:.6f}",
            "mean_AUC=",
            f"{row['mean_auc']:.6f}",
            "delta_vs_raw=",
            f"{row['mean_delta_BA_vs_raw_count70_only']:+.6f}",
        )

    print()
    print(
        "saved =",
        OUTPUT,
    )


if __name__ == "__main__":
    main()
