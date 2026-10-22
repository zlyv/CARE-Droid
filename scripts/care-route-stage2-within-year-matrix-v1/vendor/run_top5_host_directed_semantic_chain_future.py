#!/usr/bin/env python3

from __future__ import annotations

import csv
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path
from collections import defaultdict, deque
import re

import numpy as np


ROOT = Path(__file__).resolve().parent.parent

sys.path.insert(
    0,
    str(ROOT / "scripts"),
)

import run_compact_quantity_shared_caller_2x3_2017_eval as exp


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
    / "artifacts/p58-top5-host-directed-semantic-chain-future"
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


# ============================================================
# Correct host-directed semantic call-chain representation
# ============================================================

EXPECTED_ROLE_NAMES = (
    "command_execution",
    "dynamic_code_loading",
    "native_library_loading",
    "reflective_invocation",
    "package_installation",
    "package_uninstallation",
    "device_administration",
    "accessibility_control",
    "sms_send_receive",
    "call_control_or_recording",
    "account_or_credential_access",
    "contacts_or_calendar_access",
    "device_identifier_access",
    "location_access",
    "microphone_access",
    "camera_access",
    "clipboard_access",
    "network_transmission_or_upload",
    "sensitive_file_or_content_write",
    "boot_persistence",
    "task_or_alarm_scheduling",
    "background_service_control",
    "overlay_or_ui_deception",
    "icon_or_component_hiding",
    "process_or_wakelock_control",
    "cryptography",
    "encoding_or_compression",
    "url_or_http_setup",
    "file_read",
    "database_access",
    "other",
)

TOP5_ROLE_NAMES = (
    "background_service_control",
    "location_access",
    "camera_access",
    "call_control_or_recording",
    "microphone_access",
)

SEMANTIC_CHAIN_MAX_HOPS = 4
SEMANTIC_CHAIN_STATE_CAP = 100000
SEMANTIC_CHAIN_GLOBAL_WIDTH = 8


def load_and_validate_role_names():
    mapping = (
        Path(__file__)
        .resolve()
        .parents[1]
        / "configs/protocol/semantic_mapping_v1.yaml"
    )

    values = []

    pattern = re.compile(
        r"^\s*-\s+([a-z][a-z0-9_]*)\s*$"
    )

    for line in mapping.read_text(
        encoding="utf-8"
    ).splitlines():

        match = pattern.match(
            line
        )

        if match:
            values.append(
                match.group(1)
            )

        if len(values) == len(
            EXPECTED_ROLE_NAMES
        ):
            break

    result = tuple(
        values
    )

    if result != EXPECTED_ROLE_NAMES:
        raise RuntimeError(
            "semantic role order differs from the frozen "
            "31-role vocabulary:\n"
            + repr(result)
        )

    if len(result) != int(
        exp.ROLE_VOCABULARY_SIZE
    ):
        raise RuntimeError(
            "semantic role vocabulary size mismatch"
        )

    return result


SEMANTIC_ROLE_NAMES = (
    load_and_validate_role_names()
)

ROLE_NAME_TO_ID = {
    name: index
    for index, name in enumerate(
        SEMANTIC_ROLE_NAMES
    )
}

TOP5_ROLE_IDS = tuple(
    ROLE_NAME_TO_ID[
        name
    ]
    for name in TOP5_ROLE_NAMES
)

TOP5_ID_TO_POSITION = {
    role_id: position
    for position, role_id in enumerate(
        TOP5_ROLE_IDS
    )
}

SEMANTIC_ROLE_COUNT = len(
    SEMANTIC_ROLE_NAMES
)

SEMANTIC_CHAIN_FORWARD_WIDTH = (
    len(TOP5_ROLE_IDS)
    * SEMANTIC_ROLE_COUNT
    * SEMANTIC_CHAIN_MAX_HOPS
)

SEMANTIC_CHAIN_REVERSE_WIDTH = (
    SEMANTIC_ROLE_COUNT
    * len(TOP5_ROLE_IDS)
    * SEMANTIC_CHAIN_MAX_HOPS
)

SEMANTIC_CHAIN_DIMENSION = (
    SEMANTIC_CHAIN_FORWARD_WIDTH
    + SEMANTIC_CHAIN_REVERSE_WIDTH
    + SEMANTIC_CHAIN_GLOBAL_WIDTH
)


def valid_role_set(values):
    return frozenset(
        int(value)
        for value in (
            values
            if values is not None
            else ()
        )
        if (
            0
            <= int(value)
            < SEMANTIC_ROLE_COUNT
        )
    )


def directed_semantic_chain_vector(
    payload,
):
    (
        node_ids,
        base,
        role_rows,
        source,
        target,
    ) = exp._payload_arrays(
        payload
    )

    endpoint_mask = (
        base[:, 4] > 0.5
    )

    endpoint_nodes = set(
        map(
            int,
            np.flatnonzero(
                endpoint_mask
            ),
        )
    )

    node_roles = [
        valid_role_set(
            values
        )
        for values in role_rows
    ]

    outgoing = defaultdict(
        list
    )

    caller_targets = defaultdict(
        set
    )

    for source_value, target_value in zip(
        source,
        target,
    ):
        source_node = int(
            source_value
        )

        target_node = int(
            target_value
        )

        outgoing[
            source_node
        ].append(
            target_node
        )

        if target_node in endpoint_nodes:
            caller_targets[
                source_node
            ].add(
                target_node
            )

    # Each behaviour host is the method that directly calls
    # one or more sensitive API endpoints.
    host_roles = {}

    for host, endpoints in (
        caller_targets.items()
    ):
        roles = set()

        for endpoint in endpoints:
            roles.update(
                node_roles[
                    endpoint
                ]
            )

        if roles:
            host_roles[
                int(host)
            ] = frozenset(
                roles
            )

    anchor_hosts = {
        host
        for host, roles in (
            host_roles.items()
        )
        if any(
            role in TOP5_ID_TO_POSITION
            for role in roles
        )
    }

    forward = np.zeros(
        (
            len(TOP5_ROLE_IDS),
            SEMANTIC_ROLE_COUNT,
            SEMANTIC_CHAIN_MAX_HOPS,
        ),
        dtype=np.float64,
    )

    reverse = np.zeros(
        (
            SEMANTIC_ROLE_COUNT,
            len(TOP5_ROLE_IDS),
            SEMANTIC_CHAIN_MAX_HOPS,
        ),
        dtype=np.float64,
    )

    directed_host_pairs = 0
    forward_role_pairs = 0
    reverse_role_pairs = 0
    path_state_count = 0
    path_state_cap_hit = False
    observed_distances = []

    # Search from behaviour-host methods, not API endpoints.
    for start_host in sorted(
        host_roles
    ):
        distance_by_node = {
            start_host: 0,
        }

        queue = deque(
            [
                start_host,
            ]
        )

        while (
            queue
            and not path_state_cap_hit
        ):
            node = queue.popleft()

            current_distance = (
                distance_by_node[
                    node
                ]
            )

            if (
                current_distance
                >= SEMANTIC_CHAIN_MAX_HOPS
            ):
                continue

            for neighbour in outgoing.get(
                node,
                (),
            ):
                # Sensitive API vertices are terminal evidence;
                # traversal stays on application/framework methods.
                if neighbour in endpoint_nodes:
                    continue

                next_distance = (
                    current_distance + 1
                )

                old_distance = (
                    distance_by_node.get(
                        neighbour
                    )
                )

                if (
                    old_distance is not None
                    and old_distance
                    <= next_distance
                ):
                    continue

                distance_by_node[
                    neighbour
                ] = next_distance

                queue.append(
                    neighbour
                )

                path_state_count += 1

                if (
                    path_state_count
                    > SEMANTIC_CHAIN_STATE_CAP
                ):
                    path_state_cap_hit = True
                    break

        start_roles = host_roles[
            start_host
        ]

        for end_host, distance in (
            distance_by_node.items()
        ):
            if (
                end_host == start_host
                or end_host
                not in host_roles
            ):
                continue

            if not (
                1
                <= distance
                <= SEMANTIC_CHAIN_MAX_HOPS
            ):
                continue

            end_roles = host_roles[
                end_host
            ]

            start_top5 = [
                role
                for role in start_roles
                if role
                in TOP5_ID_TO_POSITION
            ]

            end_top5 = [
                role
                for role in end_roles
                if role
                in TOP5_ID_TO_POSITION
            ]

            if not (
                start_top5
                or end_top5
            ):
                continue

            directed_host_pairs += 1

            observed_distances.append(
                distance
            )

            bucket = (
                distance - 1
            )

            # TOP-5 is upstream/source:
            # TOP5 -> arbitrary semantic role.
            for anchor_role in start_top5:
                anchor_position = (
                    TOP5_ID_TO_POSITION[
                        anchor_role
                    ]
                )

                for context_role in end_roles:
                    forward[
                        anchor_position,
                        context_role,
                        bucket,
                    ] += 1.0

                    forward_role_pairs += 1

            # TOP-5 is downstream/target:
            # arbitrary semantic role -> TOP5.
            for anchor_role in end_top5:
                anchor_position = (
                    TOP5_ID_TO_POSITION[
                        anchor_role
                    ]
                )

                for context_role in start_roles:
                    reverse[
                        context_role,
                        anchor_position,
                        bucket,
                    ] += 1.0

                    reverse_role_pairs += 1

        if path_state_cap_hit:
            break

    if observed_distances:
        mean_distance = float(
            np.mean(
                observed_distances
            )
        )

        max_distance = float(
            np.max(
                observed_distances
            )
        )
    else:
        mean_distance = 0.0
        max_distance = 0.0

    global_values = np.asarray(
        [
            np.log1p(
                len(
                    endpoint_nodes
                )
            ),

            np.log1p(
                len(
                    host_roles
                )
            ),

            np.log1p(
                len(
                    anchor_hosts
                )
            ),

            np.log1p(
                directed_host_pairs
            ),

            np.log1p(
                forward_role_pairs
            ),

            np.log1p(
                reverse_role_pairs
            ),

            (
                mean_distance
                / SEMANTIC_CHAIN_MAX_HOPS
            ),

            (
                max_distance
                / SEMANTIC_CHAIN_MAX_HOPS
            ),
        ],
        dtype=np.float64,
    )

    vector = np.concatenate(
        [
            np.log1p(
                forward
            ).reshape(-1),

            np.log1p(
                reverse
            ).reshape(-1),

            global_values,
        ]
    )

    if vector.shape != (
        SEMANTIC_CHAIN_DIMENSION,
    ):
        raise RuntimeError(
            "unexpected directed semantic-chain shape: "
            + repr(
                vector.shape
            )
        )

    audit = {
        "selected_sensitive_endpoint_count":
            len(
                endpoint_nodes
            ),

        "behavior_host_count":
            len(
                host_roles
            ),

        "top5_anchor_host_count":
            len(
                anchor_hosts
            ),

        "directed_host_pair_count":
            directed_host_pairs,

        "forward_role_pair_count":
            forward_role_pairs,

        "reverse_role_pair_count":
            reverse_role_pairs,

        "path_state_count":
            path_state_count,

        "path_state_cap_hit":
            int(
                path_state_cap_hit
            ),

        "nonzero_feature_count":
            int(
                np.count_nonzero(
                    vector
                )
            ),
    }

    return (
        vector,
        audit,
    )


def host_semantic_chain_matrix(
    rows,
    cache,
    route_mask,
):
    vectors = []

    audits = []

    for row, routed in zip(
        rows,
        route_mask,
    ):
        sha = norm_sha(
            row["sha256"]
        )

        if not routed:
            vectors.append(
                np.zeros(
                    SEMANTIC_CHAIN_DIMENSION,
                    dtype=np.float64,
                )
            )

            audits.append({
                "sha256":
                    sha,

                "top5_routed":
                    0,

                "selected_sensitive_endpoint_count":
                    0,

                "behavior_host_count":
                    0,

                "top5_anchor_host_count":
                    0,

                "directed_host_pair_count":
                    0,

                "forward_role_pair_count":
                    0,

                "reverse_role_pair_count":
                    0,

                "path_state_count":
                    0,

                "path_state_cap_hit":
                    0,

                "nonzero_feature_count":
                    0,
            })

            continue

        payload = cache.get(
            sha
        )

        if payload is None:
            raise RuntimeError(
                "routed APK missing endpoint cache: "
                + sha
            )

        vector, audit = (
            directed_semantic_chain_vector(
                payload
            )
        )

        vectors.append(
            vector
        )

        audits.append({
            "sha256":
                sha,

            "top5_routed":
                1,

            **audit,
        })

    return (
        np.vstack(
            vectors
        ),

        audits,
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
        "TOP-5 HOST-DIRECTED SEMANTIC CALL-CHAIN FUTURE EVALUATION"
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
    # Correct host-directed TOP-5 semantic call chains
    #
    # Non-routed APKs are skipped completely.
    # ========================================================

    (
        chain_source_raw,
        source_chain_audit,
    ) = host_semantic_chain_matrix(
        source_rows,
        endpoint_cache,
        source_route,
    )

    (
        chain_future_raw,
        future_chain_audit,
    ) = host_semantic_chain_matrix(
        future_rows,
        endpoint_cache,
        future_route,
    )

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
        "directed semantic-chain dimensions =",
        SEMANTIC_CHAIN_DIMENSION,
        flush=True,
    )

    print(
        "TOP-5 role IDs =",
        {
            name:
                ROLE_NAME_TO_ID[
                    name
                ]
            for name in TOP5_ROLE_NAMES
        },
        flush=True,
    )

    print(
        "2017 routed APKs with nonzero directed chains =",
        sum(
            int(
                row[
                    "nonzero_feature_count"
                ]
                > 0
            )
            for row in source_chain_audit
            if row[
                "top5_routed"
            ]
        ),
        "/",
        int(
            source_route.sum()
        ),
        flush=True,
    )

    print(
        "future routed APKs with nonzero directed chains =",
        sum(
            int(
                row[
                    "nonzero_feature_count"
                ]
                > 0
            )
            for row in future_chain_audit
            if row[
                "top5_routed"
            ]
        ),
        "/",
        int(
            future_route.sum()
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
