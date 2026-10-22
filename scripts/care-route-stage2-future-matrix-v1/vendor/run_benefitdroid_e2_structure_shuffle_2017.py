#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent

if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))


import run_compact_quantity_shared_caller_2x3_2017_eval as exp
import run_frozen_anchor_residual_2017_2021 as base
import run_stable70_vs_allratio70_2017 as allratio
import run_p72_verified_dependency_source_ablation as p72
import run_benefitdroid_e1_exact_pair_2017 as e1


THRESHOLD = 0.5


def norm_sha(value):
    return str(value or "").strip().casefold()


def read_csv(path: Path):
    with path.open(
        newline="",
        encoding="utf-8-sig",
    ) as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows):
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not rows:
        return

    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(rows[0].keys()),
        )
        writer.writeheader()
        writer.writerows(rows)


def extract_host_roles(payload):
    """
    Recover exactly the same behavior-host -> semantic-role
    relation used by the existing Shared-Caller method.
    """

    (
        _node_ids,
        node_base,
        role_rows,
        source,
        target,
    ) = exp._payload_arrays(
        payload
    )

    endpoint_mask = (
        node_base[:, 4] > 0.5
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
        exp._roles(values)
        for values in role_rows
    ]

    caller_targets = defaultdict(
        set
    )

    for src_value, dst_value in zip(
        source,
        target,
    ):
        src = int(src_value)
        dst = int(dst_value)

        if dst in endpoint_nodes:
            caller_targets[
                src
            ].add(dst)

    host_roles = {}

    for caller, endpoints in (
        caller_targets.items()
    ):
        roles = frozenset().union(
            *(
                node_roles[index]
                for index in endpoints
            )
        )

        if roles:
            host_roles[
                int(caller)
            ] = set(
                int(x)
                for x in roles
            )

    return host_roles


def pair_raw_from_hosts(
    host_roles,
    pair_index,
    pair_dim,
):
    vector = np.zeros(
        pair_dim,
        dtype=np.float64,
    )

    for roles in host_roles.values():

        ordered = sorted(
            roles
        )

        for a, b in combinations(
            ordered,
            2,
        ):
            vector[
                pair_index[
                    (
                        a,
                        b,
                    )
                ]
            ] += 1.0

    return vector


def deterministic_rng(
    global_seed,
    sha256,
):
    digest = hashlib.sha256(
        (
            str(global_seed)
            + ":"
            + sha256
        ).encode(
            "utf-8"
        )
    ).digest()

    seed = int.from_bytes(
        digest[:8],
        byteorder="little",
        signed=False,
    )

    return np.random.default_rng(
        seed
    )


def degree_preserving_shuffle(
    host_roles,
    rng,
    swap_multiplier,
):
    """
    Bipartite double-edge swap.

    Before:
        H1 -- R1
        H2 -- R2

    After:
        H1 -- R2
        H2 -- R1

    This preserves:
        - degree of every host
        - degree of every semantic role
        - total edge count
        - host count

    Only the identity of co-hosted role combinations changes.
    """

    adjacency = {
        host: set(roles)
        for host, roles
        in host_roles.items()
    }

    edges = [
        (
            host,
            role,
        )
        for host, roles
        in adjacency.items()
        for role in roles
    ]

    edge_count = len(
        edges
    )

    if edge_count < 2:
        return (
            adjacency,
            0,
            0,
        )

    target_swaps = int(
        max(
            1,
            round(
                swap_multiplier
                * edge_count
            ),
        )
    )

    max_attempts = max(
        100,
        target_swaps * 50,
    )

    accepted = 0
    attempts = 0

    while (
        accepted < target_swaps
        and attempts < max_attempts
    ):
        attempts += 1

        i = int(
            rng.integers(
                0,
                edge_count,
            )
        )

        j = int(
            rng.integers(
                0,
                edge_count,
            )
        )

        if i == j:
            continue

        h1, r1 = edges[i]
        h2, r2 = edges[j]

        if h1 == h2:
            continue

        if r1 == r2:
            continue

        # New edges must not already exist.
        if r2 in adjacency[h1]:
            continue

        if r1 in adjacency[h2]:
            continue

        adjacency[
            h1
        ].remove(
            r1
        )

        adjacency[
            h2
        ].remove(
            r2
        )

        adjacency[
            h1
        ].add(
            r2
        )

        adjacency[
            h2
        ].add(
            r1
        )

        edges[i] = (
            h1,
            r2,
        )

        edges[j] = (
            h2,
            r1,
        )

        accepted += 1

    return (
        adjacency,
        attempts,
        accepted,
    )


def metrics_for(
    labels,
    probability,
):
    return base.metrics(
        labels.astype(int),
        probability,
        threshold=THRESHOLD,
    )


def comparison_counts(
    labels,
    reference_probability,
    new_probability,
):
    reference_pred = (
        reference_probability
        >= THRESHOLD
    ).astype(
        np.int64
    )

    new_pred = (
        new_probability
        >= THRESHOLD
    ).astype(
        np.int64
    )

    fixed = int(
        np.sum(
            (reference_pred != labels)
            & (new_pred == labels)
        )
    )

    broken = int(
        np.sum(
            (reference_pred == labels)
            & (new_pred != labels)
        )
    )

    return (
        fixed,
        broken,
        fixed - broken,
    )


def fit_structure_oof(
    structure_all,
    structure_names,
    fold_cache,
    labels,
):
    n = len(
        labels
    )

    oof = np.zeros(
        n,
        dtype=np.float64,
    )

    for fold in sorted(
        fold_cache
    ):
        data = fold_cache[
            fold
        ]

        train_idx = data[
            "train_idx"
        ]

        val_idx = data[
            "val_idx"
        ]

        train_raw = (
            structure_all[
                train_idx
            ]
        )

        val_raw = (
            structure_all[
                val_idx
            ]
        )

        (
            structure_train,
            structure_val,
            _,
        ) = p72.standardize_added(
            train_raw,
            val_raw,
            structure_names,
        )

        x_train = np.concatenate(
            [
                data[
                    "compact_train"
                ],
                structure_train,
            ],
            axis=1,
        )

        x_val = np.concatenate(
            [
                data[
                    "compact_val"
                ],
                structure_val,
            ],
            axis=1,
        )

        (
            probability,
            _,
            _,
        ) = p72.fit_predict(
            x_train,
            labels[
                train_idx
            ],
            x_val,
            seed=int(fold) - 1,
        )

        oof[
            val_idx
        ] = probability

    return oof


def empirical_p(
    real_value,
    shuffled_values,
):
    shuffled_values = np.asarray(
        shuffled_values,
        dtype=float,
    )

    count_ge = int(
        np.sum(
            shuffled_values
            >= real_value - 1e-12
        )
    )

    return (
        1.0 + count_ge
    ) / (
        1.0
        + len(
            shuffled_values
        )
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--cache",
        type=Path,
        default=Path(
            "artifacts/"
            "p25-endpoint-cache-full-cap256/"
            "endpoint_centered_tensor_cache.jsonl.gz"
        ),
    )

    parser.add_argument(
        "--compact",
        type=Path,
        default=Path(
            "artifacts/"
            "p1-features/"
            "features.csv"
        ),
    )

    parser.add_argument(
        "--compact-schema",
        type=Path,
        default=Path(
            "artifacts/"
            "p1-features/"
            "feature_schema.json"
        ),
    )

    parser.add_argument(
        "--p72-oof",
        type=Path,
        default=Path(
            "artifacts/"
            "p72-verified-dependency-source-ablation/"
            "p72_oof_predictions.csv"
        ),
    )

    parser.add_argument(
        "--e1b-oof",
        type=Path,
        default=Path(
            "artifacts/"
            "benefitdroid-e1b-exact-pair-2017/"
            "e1_oof_predictions.csv"
        ),
    )

    parser.add_argument(
        "--n-shuffles",
        type=int,
        default=20,
    )

    parser.add_argument(
        "--swap-multiplier",
        type=float,
        default=10.0,
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "artifacts/"
            "benefitdroid-e2-structure-shuffle-2017"
        ),
    )

    args = parser.parse_args()

    args.output.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ========================================================
    # Fixed source cohort and folds
    # ========================================================

    reference_rows = read_csv(
        args.p72_oof
    )

    if len(
        reference_rows
    ) != 799:
        raise RuntimeError(
            "Expected source cohort = 799"
        )

    manifest = []

    for row in reference_rows:
        manifest.append({
            "sha256":
                norm_sha(
                    row[
                        "sha256"
                    ]
                ),

            "label":
                int(
                    row[
                        "label"
                    ]
                ),

            "fold":
                int(
                    row[
                        "fold"
                    ]
                ),

            "C_probability":
                float(
                    row[
                        "C_prob"
                    ]
                ),
        })

    labels = np.asarray(
        [
            row[
                "label"
            ]
            for row in manifest
        ],
        dtype=np.int64,
    )

    fold_ids = np.asarray(
        [
            row[
                "fold"
            ]
            for row in manifest
        ],
        dtype=np.int64,
    )

    c_oof = np.asarray(
        [
            row[
                "C_probability"
            ]
            for row in manifest
        ],
        dtype=np.float64,
    )

    print(
        "========================================"
    )
    print(
        "BenefitDroid Mainline E2"
    )
    print(
        "Count-Preserving Structure Shuffle"
    )
    print(
        "========================================"
    )

    print(
        "APKs:",
        len(
            manifest
        ),
    )

    print(
        "Shuffles:",
        args.n_shuffles,
    )

    print(
        "Swap multiplier:",
        args.swap_multiplier,
    )

    # ========================================================
    # Compact data
    # ========================================================

    compact_rows = read_csv(
        args.compact
    )

    compact_by_sha = {}

    for row in compact_rows:
        sha = norm_sha(
            row.get(
                "sha256",
                "",
            )
        )

        if not sha:
            continue

        row = dict(
            row
        )

        row[
            "sha256"
        ] = sha

        compact_by_sha[
            sha
        ] = row

    cohort_rows = [
        compact_by_sha[
            row[
                "sha256"
            ]
        ]
        for row in manifest
    ]

    with args.compact_schema.open(
        encoding="utf-8",
    ) as f:
        compact_schema = json.load(
            f
        )

    (
        _,
        _,
        canonical_original_names,
    ) = allratio.original_compact_matrix_named(
        cohort_rows,
        cohort_rows[:1],
        compact_by_sha,
        compact_schema,
    )

    canonical_original_names = list(
        canonical_original_names
    )

    if len(
        canonical_original_names
    ) != 76:
        raise RuntimeError(
            "Expected frozen original Compact=76"
        )

    # ========================================================
    # Fixed fold Compact matrices
    # ========================================================

    fold_cache = {}

    for fold in sorted(
        np.unique(
            fold_ids
        )
    ):
        train_idx = np.flatnonzero(
            fold_ids != fold
        )

        val_idx = np.flatnonzero(
            fold_ids == fold
        )

        train_rows = [
            cohort_rows[i]
            for i in train_idx
        ]

        val_rows = [
            cohort_rows[i]
            for i in val_idx
        ]

        (
            compact_train,
            compact_val,
            compact_names,
        ) = (
            p72
            .compact_matrix_allratio70_foldsafe(
                train_rows,
                val_rows,
                compact_by_sha,
                compact_schema,
                canonical_original_names,
            )
        )

        if len(
            compact_names
        ) != 70:
            raise RuntimeError(
                "Expected Compact=70"
            )

        fold_cache[
            int(fold)
        ] = {
            "train_idx":
                train_idx,

            "val_idx":
                val_idx,

            "compact_train":
                compact_train,

            "compact_val":
                compact_val,
        }

    # ========================================================
    # Existing endpoint cache
    # ========================================================

    if not args.cache.is_file():
        raise RuntimeError(
            "Cache file not found: "
            + str(
                args.cache
            )
        )

    cache = exp.read_cache(
        args.cache
    )

    missing = [
        row[
            "sha256"
        ]
        for row in manifest
        if row[
            "sha256"
        ] not in cache
    ]

    if missing:
        raise RuntimeError(
            "Cache missing source APKs: "
            + repr(
                missing[:10]
            )
        )

    # ========================================================
    # Fixed pair schema
    # ========================================================

    n_roles = int(
        exp.ROLE_VOCABULARY_SIZE
    )

    pair_list = list(
        combinations(
            range(
                n_roles
            ),
            2,
        )
    )

    pair_index = {
        pair: i
        for i, pair
        in enumerate(
            pair_list
        )
    }

    pair_dim = len(
        pair_list
    )

    if pair_dim != 465:
        raise RuntimeError(
            "Expected 465 semantic pairs"
        )

    # ========================================================
    # Build REAL Stats + REAL Pair + host-role graphs
    # ========================================================

    stats_rows = []
    real_pair_rows = []
    host_graphs = []

    for i, row in enumerate(
        manifest,
        start=1,
    ):
        sha = row[
            "sha256"
        ]

        payload = cache[
            sha
        ][
            "payload"
        ]

        (
            stats,
            pair_raw,
            _pair_norm,
            _audit,
        ) = e1.reconstruct_host_structure(
            payload,
            pair_list,
            pair_index,
        )

        host_roles = (
            extract_host_roles(
                payload
            )
        )

        direct_pair = (
            pair_raw_from_hosts(
                host_roles,
                pair_index,
                pair_dim,
            )
        )

        if not np.array_equal(
            pair_raw,
            direct_pair,
        ):
            raise RuntimeError(
                "Pair reconstruction mismatch: "
                + sha
            )

        stats_rows.append(
            stats
        )

        real_pair_rows.append(
            pair_raw
        )

        host_graphs.append(
            host_roles
        )

        if (
            i % 100 == 0
            or i == len(
                manifest
            )
        ):
            print(
                "structure:",
                i,
                "/",
                len(
                    manifest
                ),
            )

    stats_all = np.asarray(
        stats_rows,
        dtype=np.float64,
    )

    real_pair_all = np.asarray(
        real_pair_rows,
        dtype=np.float64,
    )

    print()
    print(
        "Stats dimension:",
        stats_all.shape[1],
    )

    print(
        "Pair dimension:",
        real_pair_all.shape[1],
    )

    if stats_all.shape[1] != 120:
        raise RuntimeError(
            "Expected existing Stats=120"
        )

    stats_names = [
        f"shared_stats__{i:03d}"
        for i in range(
            stats_all.shape[1]
        )
    ]

    pair_names = [
        f"pair_raw__{i:03d}"
        for i in range(
            pair_dim
        )
    ]

    # ========================================================
    # Reproduce C+Stats and C+Stats+RealPair
    # ========================================================

    stats_oof = fit_structure_oof(
        stats_all,
        stats_names,
        fold_cache,
        labels,
    )

    real_structure = np.concatenate(
        [
            stats_all,
            real_pair_all,
        ],
        axis=1,
    )

    real_names = (
        stats_names
        + pair_names
    )

    real_oof = fit_structure_oof(
        real_structure,
        real_names,
        fold_cache,
        labels,
    )

    stats_result = metrics_for(
        labels,
        stats_oof,
    )

    real_result = metrics_for(
        labels,
        real_oof,
    )

    (
        real_fixed_vs_c,
        real_broken_vs_c,
        real_net_vs_c,
    ) = comparison_counts(
        labels,
        c_oof,
        real_oof,
    )

    (
        real_fixed_vs_stats,
        real_broken_vs_stats,
        real_net_vs_stats,
    ) = comparison_counts(
        labels,
        stats_oof,
        real_oof,
    )

    print()
    print(
        "===== REAL STRUCTURE ====="
    )

    print(
        "C+Stats:",
        f"BA={stats_result['balanced_accuracy']:.6f}",
        f"ACC={stats_result['accuracy']:.6f}",
        f"AUC={stats_result['auc']:.6f}",
    )

    print(
        "C+Stats+RealPair:",
        f"BA={real_result['balanced_accuracy']:.6f}",
        f"ACC={real_result['accuracy']:.6f}",
        f"AUC={real_result['auc']:.6f}",
    )

    print(
        "RealPair vs Stats:",
        "fixed=",
        real_fixed_vs_stats,
        "broken=",
        real_broken_vs_stats,
        "net=",
        real_net_vs_stats,
    )

    # ========================================================
    # Reproduce E1b exactly if previous output exists
    # ========================================================

    if args.e1b_oof.is_file():

        e1_rows = read_csv(
            args.e1b_oof
        )

        e1_by_sha = {
            norm_sha(
                row[
                    "sha256"
                ]
            ): row
            for row in e1_rows
        }

        e1_stats = np.asarray(
            [
                float(
                    e1_by_sha[
                        row[
                            "sha256"
                        ]
                    ][
                        "C_plus_Stats_prob"
                    ]
                )
                for row in manifest
            ],
            dtype=np.float64,
        )

        e1_real = np.asarray(
            [
                float(
                    e1_by_sha[
                        row[
                            "sha256"
                        ]
                    ][
                        "C_plus_Stats_plus_PairRaw_prob"
                    ]
                )
                for row in manifest
            ],
            dtype=np.float64,
        )

        stats_diff = float(
            np.max(
                np.abs(
                    stats_oof
                    - e1_stats
                )
            )
        )

        real_diff = float(
            np.max(
                np.abs(
                    real_oof
                    - e1_real
                )
            )
        )

        print()
        print(
            "E1b Stats reproduction max diff:",
            stats_diff,
        )

        print(
            "E1b RealPair reproduction max diff:",
            real_diff,
        )

        if (
            stats_diff > 1e-7
            or real_diff > 1e-7
        ):
            raise RuntimeError(
                "E2 does not reproduce E1b. "
                "Stop before shuffle experiment."
            )

    # ========================================================
    # Shuffle experiment
    # ========================================================

    shuffle_rows = []

    shuffled_acc = []
    shuffled_ba = []
    shuffled_auc = []
    shuffled_net_vs_stats = []

    for shuffle_seed in range(
        args.n_shuffles
    ):

        shuffled_pair_all = np.zeros_like(
            real_pair_all
        )

        total_attempts = 0
        total_accepted = 0
        changed_apks = 0
        unchanged_apks = 0
        total_l1_change = 0.0
        changed_pair_cells = 0

        for i, row in enumerate(
            manifest
        ):
            sha = row[
                "sha256"
            ]

            rng = deterministic_rng(
                shuffle_seed,
                sha,
            )

            (
                shuffled_hosts,
                attempts,
                accepted,
            ) = degree_preserving_shuffle(
                host_graphs[i],
                rng,
                args.swap_multiplier,
            )

            shuffled_pair = (
                pair_raw_from_hosts(
                    shuffled_hosts,
                    pair_index,
                    pair_dim,
                )
            )

            shuffled_pair_all[
                i
            ] = shuffled_pair

            total_attempts += (
                attempts
            )

            total_accepted += (
                accepted
            )

            difference = np.abs(
                shuffled_pair
                - real_pair_all[i]
            )

            l1 = float(
                np.sum(
                    difference
                )
            )

            total_l1_change += (
                l1
            )

            changed_pair_cells += int(
                np.sum(
                    difference > 0
                )
            )

            if l1 > 0:
                changed_apks += 1
            else:
                unchanged_apks += 1

        shuffled_structure = (
            np.concatenate(
                [
                    stats_all,
                    shuffled_pair_all,
                ],
                axis=1,
            )
        )

        shuffled_oof = (
            fit_structure_oof(
                shuffled_structure,
                real_names,
                fold_cache,
                labels,
            )
        )

        result = metrics_for(
            labels,
            shuffled_oof,
        )

        (
            fixed_vs_c,
            broken_vs_c,
            net_vs_c,
        ) = comparison_counts(
            labels,
            c_oof,
            shuffled_oof,
        )

        (
            fixed_vs_stats,
            broken_vs_stats,
            net_vs_stats,
        ) = comparison_counts(
            labels,
            stats_oof,
            shuffled_oof,
        )

        shuffled_acc.append(
            result[
                "accuracy"
            ]
        )

        shuffled_ba.append(
            result[
                "balanced_accuracy"
            ]
        )

        shuffled_auc.append(
            result[
                "auc"
            ]
        )

        shuffled_net_vs_stats.append(
            net_vs_stats
        )

        shuffle_rows.append({
            "shuffle_seed":
                shuffle_seed,

            "BA":
                result[
                    "balanced_accuracy"
                ],

            "ACC":
                result[
                    "accuracy"
                ],

            "AUC":
                result[
                    "auc"
                ],

            "fixed_vs_C":
                fixed_vs_c,

            "broken_vs_C":
                broken_vs_c,

            "net_vs_C":
                net_vs_c,

            "fixed_vs_Stats":
                fixed_vs_stats,

            "broken_vs_Stats":
                broken_vs_stats,

            "net_vs_Stats":
                net_vs_stats,

            "changed_APKs":
                changed_apks,

            "unchanged_APKs":
                unchanged_apks,

            "accepted_swaps":
                total_accepted,

            "attempted_swaps":
                total_attempts,

            "pair_L1_change":
                total_l1_change,

            "changed_pair_cells":
                changed_pair_cells,
        })

        print(
            f"shuffle {shuffle_seed:02d}",
            f"ACC={result['accuracy']:.6f}",
            f"BA={result['balanced_accuracy']:.6f}",
            f"AUC={result['auc']:.6f}",
            f"net_vs_stats={net_vs_stats:+d}",
            f"changed_apks={changed_apks}",
        )

    # ========================================================
    # Aggregate comparison
    # ========================================================

    shuffled_acc = np.asarray(
        shuffled_acc,
        dtype=float,
    )

    shuffled_ba = np.asarray(
        shuffled_ba,
        dtype=float,
    )

    shuffled_auc = np.asarray(
        shuffled_auc,
        dtype=float,
    )

    shuffled_net_vs_stats = np.asarray(
        shuffled_net_vs_stats,
        dtype=float,
    )

    comparison = [{
        "real_BA":
            real_result[
                "balanced_accuracy"
            ],

        "shuffle_mean_BA":
            float(
                np.mean(
                    shuffled_ba
                )
            ),

        "shuffle_std_BA":
            float(
                np.std(
                    shuffled_ba
                )
            ),

        "real_minus_shuffle_mean_BA":
            float(
                real_result[
                    "balanced_accuracy"
                ]
                - np.mean(
                    shuffled_ba
                )
            ),

        "empirical_p_BA":
            empirical_p(
                real_result[
                    "balanced_accuracy"
                ],
                shuffled_ba,
            ),

        "real_ACC":
            real_result[
                "accuracy"
            ],

        "shuffle_mean_ACC":
            float(
                np.mean(
                    shuffled_acc
                )
            ),

        "shuffle_std_ACC":
            float(
                np.std(
                    shuffled_acc
                )
            ),

        "real_minus_shuffle_mean_ACC":
            float(
                real_result[
                    "accuracy"
                ]
                - np.mean(
                    shuffled_acc
                )
            ),

        "empirical_p_ACC":
            empirical_p(
                real_result[
                    "accuracy"
                ],
                shuffled_acc,
            ),

        "real_AUC":
            real_result[
                "auc"
            ],

        "shuffle_mean_AUC":
            float(
                np.mean(
                    shuffled_auc
                )
            ),

        "real_minus_shuffle_mean_AUC":
            float(
                real_result[
                    "auc"
                ]
                - np.mean(
                    shuffled_auc
                )
            ),

        "empirical_p_AUC":
            empirical_p(
                real_result[
                    "auc"
                ],
                shuffled_auc,
            ),

        "real_fixed_vs_Stats":
            real_fixed_vs_stats,

        "real_broken_vs_Stats":
            real_broken_vs_stats,

        "real_net_vs_Stats":
            real_net_vs_stats,

        "shuffle_mean_net_vs_Stats":
            float(
                np.mean(
                    shuffled_net_vs_stats
                )
            ),

        "empirical_p_net_vs_Stats":
            empirical_p(
                real_net_vs_stats,
                shuffled_net_vs_stats,
            ),

        "n_shuffles":
            args.n_shuffles,
    }]

    # ========================================================
    # Save outputs
    # ========================================================

    write_csv(
        args.output
        / "e2_shuffle_runs.csv",
        shuffle_rows,
    )

    write_csv(
        args.output
        / "e2_comparison.csv",
        comparison,
    )

    real_prediction_rows = []

    for i, row in enumerate(
        manifest
    ):
        real_prediction_rows.append({
            "sha256":
                row[
                    "sha256"
                ],

            "label":
                row[
                    "label"
                ],

            "fold":
                row[
                    "fold"
                ],

            "C_probability":
                float(
                    c_oof[i]
                ),

            "Stats_probability":
                float(
                    stats_oof[i]
                ),

            "RealPair_probability":
                float(
                    real_oof[i]
                ),

            "C_prediction":
                int(
                    c_oof[i]
                    >= THRESHOLD
                ),

            "Stats_prediction":
                int(
                    stats_oof[i]
                    >= THRESHOLD
                ),

            "RealPair_prediction":
                int(
                    real_oof[i]
                    >= THRESHOLD
                ),
        })

    write_csv(
        args.output
        / "e2_real_oof_predictions.csv",
        real_prediction_rows,
    )

    audit = {
        "source_apks":
            len(
                manifest
            ),

        "folds":
            5,

        "semantic_roles":
            n_roles,

        "pair_dimensions":
            pair_dim,

        "existing_stats_dimensions":
            int(
                stats_all.shape[1]
            ),

        "shuffle_type":
            (
                "within-APK bipartite "
                "degree-preserving double-edge swap"
            ),

        "preserved":
            [
                "host degree",
                "role degree",
                "host count",
                "total host-role edge count",
                "real 120D Stats model input",
            ],

        "changed":
            (
                "identity of semantic roles "
                "co-hosted by the same method"
            ),

        "n_shuffles":
            args.n_shuffles,

        "swap_multiplier":
            args.swap_multiplier,
    }

    (
        args.output
        / "e2_audit.json"
    ).write_text(
        json.dumps(
            audit,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    # ========================================================
    # Console result
    # ========================================================

    row = comparison[0]

    print()
    print(
        "========================================"
    )
    print(
        "E2 FINAL COMPARISON"
    )
    print(
        "========================================"
    )

    print(
        "Real Pair:"
    )

    print(
        f"  ACC={row['real_ACC']:.6f}"
        f" BA={row['real_BA']:.6f}"
        f" AUC={row['real_AUC']:.6f}"
    )

    print()
    print(
        "Shuffled Pair:"
    )

    print(
        f"  mean ACC={row['shuffle_mean_ACC']:.6f}"
        f" std={row['shuffle_std_ACC']:.6f}"
    )

    print(
        f"  mean BA ={row['shuffle_mean_BA']:.6f}"
        f" std={row['shuffle_std_BA']:.6f}"
    )

    print(
        f"  mean AUC={row['shuffle_mean_AUC']:.6f}"
    )

    print()
    print(
        "Real - shuffled mean:"
    )

    print(
        f"  ACC={row['real_minus_shuffle_mean_ACC']:+.6f}"
    )

    print(
        f"  BA ={row['real_minus_shuffle_mean_BA']:+.6f}"
    )

    print(
        f"  AUC={row['real_minus_shuffle_mean_AUC']:+.6f}"
    )

    print()
    print(
        "Empirical one-sided p:"
    )

    print(
        f"  ACC={row['empirical_p_ACC']:.6f}"
    )

    print(
        f"  BA ={row['empirical_p_BA']:.6f}"
    )

    print(
        f"  AUC={row['empirical_p_AUC']:.6f}"
    )

    print()
    print(
        "Pair incremental correction vs Stats:"
    )

    print(
        "  Real:",
        f"fixed={row['real_fixed_vs_Stats']}",
        f"broken={row['real_broken_vs_Stats']}",
        f"net={row['real_net_vs_Stats']}",
    )

    print(
        "  Shuffled mean net:",
        f"{row['shuffle_mean_net_vs_Stats']:.3f}",
    )

    print()
    print(
        "Output:",
        args.output,
    )


if __name__ == "__main__":
    main()
