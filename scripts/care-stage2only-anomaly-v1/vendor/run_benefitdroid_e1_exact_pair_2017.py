#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import gzip
import json
import re
import sys
from collections import Counter, defaultdict
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


THRESHOLD = 0.5


def norm_sha(value):
    return str(value or "").strip().casefold()


def safe_name(value):
    value = re.sub(
        r"[^A-Za-z0-9_]+",
        "_",
        str(value),
    )
    return value.strip("_")


def read_csv(path: Path):
    with path.open(
        newline="",
        encoding="utf-8-sig",
    ) as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows, fieldnames=None):
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if fieldnames is None:
        if not rows:
            return
        fieldnames = list(
            rows[0].keys()
        )

    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
        )
        writer.writeheader()
        writer.writerows(rows)


def discover_role_names():
    n = int(
        exp.ROLE_VOCABULARY_SIZE
    )

    candidates = []

    for name in dir(exp):
        obj = getattr(
            exp,
            name,
        )

        if (
            isinstance(
                obj,
                (tuple, list),
            )
            and len(obj) == n
            and all(
                isinstance(x, str)
                for x in obj
            )
        ):
            score = 0

            low = name.casefold()

            if "role" in low:
                score += 10

            if "vocab" in low:
                score += 10

            if "semantic" in low:
                score += 5

            candidates.append(
                (
                    score,
                    name,
                    list(obj),
                )
            )

        if isinstance(obj, dict):
            # int -> string
            try:
                mapping = {
                    int(k): str(v)
                    for k, v in obj.items()
                    if 0 <= int(k) < n
                }

                if (
                    len(mapping) == n
                    and all(
                        isinstance(
                            mapping[i],
                            str,
                        )
                        for i in range(n)
                    )
                ):
                    candidates.append(
                        (
                            20,
                            name,
                            [
                                mapping[i]
                                for i in range(n)
                            ],
                        )
                    )
            except Exception:
                pass

            # string -> int
            try:
                reverse = {
                    int(v): str(k)
                    for k, v in obj.items()
                    if 0 <= int(v) < n
                }

                if len(reverse) == n:
                    candidates.append(
                        (
                            20,
                            name,
                            [
                                reverse[i]
                                for i in range(n)
                            ],
                        )
                    )
            except Exception:
                pass

    if candidates:
        candidates.sort(
            key=lambda x: (
                -x[0],
                x[1],
            )
        )

        score, source, names = (
            candidates[0]
        )

        print(
            "Role vocabulary source:",
            source,
        )

        return names

    print(
        "Role vocabulary source: "
        "fallback numeric IDs"
    )

    return [
        f"role_{i:02d}"
        for i in range(n)
    ]


def cache_overlap(
    path: Path,
    wanted: set[str],
):
    overlap = set()

    with gzip.open(
        path,
        "rt",
        encoding="utf-8",
    ) as handle:

        for line in handle:
            if not line.strip():
                continue

            try:
                row = json.loads(
                    line
                )
            except Exception:
                continue

            if (
                row.get("status")
                != "OK"
            ):
                continue

            sha = norm_sha(
                row.get(
                    "sha256",
                    "",
                )
            )

            if sha in wanted:
                overlap.add(sha)

    return len(overlap)


def resolve_cache(
    explicit: Path | None,
    wanted: set[str],
):
    if explicit is not None:
        if not explicit.exists():
            raise FileNotFoundError(
                explicit
            )

        n = cache_overlap(
            explicit,
            wanted,
        )

        print(
            "Explicit cache overlap:",
            n,
            "/",
            len(wanted),
        )

        if n != len(wanted):
            raise RuntimeError(
                "Explicit cache does not "
                "cover all source APKs."
            )

        return explicit

    candidates = sorted(
        Path("artifacts").rglob(
            "endpoint_centered_tensor_cache.jsonl.gz"
        )
    )

    if not candidates:
        candidates = sorted(
            Path("artifacts").rglob(
                "*tensor_cache*.jsonl.gz"
            )
        )

    if not candidates:
        raise RuntimeError(
            "Cannot find endpoint-centered "
            "tensor cache under artifacts/"
        )

    scored = []

    print()
    print(
        "===== CACHE CANDIDATES ====="
    )

    for path in candidates:
        n = cache_overlap(
            path,
            wanted,
        )

        scored.append(
            (
                n,
                path,
            )
        )

        print(
            f"{n:4d}/{len(wanted)}",
            path,
        )

    scored.sort(
        key=lambda x: (
            -x[0],
            str(x[1]),
        )
    )

    best_n = scored[0][0]

    best = [
        path
        for n, path in scored
        if n == best_n
    ]

    if best_n != len(wanted):
        raise RuntimeError(
            "No cache covers all "
            f"{len(wanted)} source APKs. "
            f"Best overlap={best_n}."
        )

    if len(best) != 1:
        raise RuntimeError(
            "More than one cache has full "
            "source overlap. Re-run with "
            "--cache PATH.\n"
            + "\n".join(
                str(x)
                for x in best
            )
        )

    print()
    print(
        "Selected cache:",
        best[0],
    )

    return best[0]


def reconstruct_host_structure(
    payload,
    pair_list,
    pair_index,
):
    (
        node_ids,
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

    if len(node_base):
        provenance = np.argmax(
            node_base[:, 5:8],
            axis=1,
        )
    else:
        provenance = np.zeros(
            0,
            dtype=np.int64,
        )

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
                caller
            ] = roles

    shared_hyperedges = []

    for caller, roles in (
        host_roles.items()
    ):
        if len(roles) >= 2:
            shared_hyperedges.append(
                (
                    roles,
                    int(
                        provenance[
                            caller
                        ]
                    ),
                )
            )

    # --------------------------------------------------------
    # Old Shared-Caller Stats representation.
    # This is the exact pre-existing summarization baseline.
    # --------------------------------------------------------

    stats = np.asarray(
        exp._relation_vector(
            shared_hyperedges
        ),
        dtype=np.float64,
    )

    # --------------------------------------------------------
    # Exact pair matrix.
    #
    # Each host contributes at most once to an unordered
    # semantic-role pair.
    # --------------------------------------------------------

    n_roles = int(
        exp.ROLE_VOCABULARY_SIZE
    )

    marginals = np.zeros(
        n_roles,
        dtype=np.float64,
    )

    pair_raw = np.zeros(
        len(pair_list),
        dtype=np.float64,
    )

    for roles in host_roles.values():

        ordered = sorted(
            int(x)
            for x in roles
        )

        for role in ordered:
            marginals[
                role
            ] += 1.0

        for a, b in combinations(
            ordered,
            2,
        ):
            pair_raw[
                pair_index[
                    (
                        a,
                        b,
                    )
                ]
            ] += 1.0

    pair_norm = np.zeros_like(
        pair_raw
    )

    for j, (
        a,
        b,
    ) in enumerate(
        pair_list
    ):
        denominator = np.sqrt(
            marginals[a]
            * marginals[b]
        )

        if denominator > 0.0:
            pair_norm[j] = (
                pair_raw[j]
                / denominator
            )

    audit = {
        "node_count":
            len(node_ids),

        "selected_sensitive_endpoint_count":
            len(endpoint_nodes),

        "behavior_host_method_count":
            len(host_roles),

        "shared_caller_hyperedge_count":
            len(
                shared_hyperedges
            ),

        "shared_caller_high_order_count":
            sum(
                len(roles) >= 3
                for roles
                in host_roles.values()
            ),
    }

    return (
        stats,
        pair_raw,
        pair_norm,
        audit,
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--cache",
        type=Path,
        default=None,
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
        "--output",
        type=Path,
        default=Path(
            "artifacts/"
            "benefitdroid-e1-exact-pair-2017"
        ),
    )

    args = parser.parse_args()

    args.output.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ========================================================
    # Fixed 799-APK / 5-fold source protocol
    # ========================================================

    reference_rows = read_csv(
        args.p72_oof
    )

    if len(reference_rows) != 799:
        raise RuntimeError(
            "Expected P72 source cohort=799, got "
            + str(
                len(reference_rows)
            )
        )

    manifest = []

    for row in reference_rows:
        manifest.append({
            "sha256":
                norm_sha(
                    row["sha256"]
                ),

            "label":
                int(
                    row["label"]
                ),

            "metadata_label":
                row[
                    "metadata_label"
                ],

            "fold":
                int(
                    row["fold"]
                ),

            "reference_C_prob":
                float(
                    row["C_prob"]
                ),

            "reference_C_pred":
                int(
                    row["C_pred"]
                ),
        })

    if len({
        row["sha256"]
        for row in manifest
    }) != 799:
        raise RuntimeError(
            "Duplicate SHA in source manifest"
        )

    wanted_sha = {
        row["sha256"]
        for row in manifest
    }

    print(
        "========================================"
    )
    print(
        "BenefitDroid Mainline E1"
    )
    print(
        "Exact Host-Role Pair Ablation"
    )
    print(
        "========================================"
    )

    print(
        "APKs:",
        len(manifest),
    )

    print(
        "labels:",
        Counter(
            row[
                "metadata_label"
            ]
            for row in manifest
        ),
    )

    print(
        "folds:",
        Counter(
            row["fold"]
            for row in manifest
        ),
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

        if sha:
            row = dict(row)
            row["sha256"] = sha
            compact_by_sha[
                sha
            ] = row

    missing_compact = sorted(
        wanted_sha
        - set(
            compact_by_sha
        )
    )

    if missing_compact:
        raise RuntimeError(
            "Missing compact APKs: "
            + repr(
                missing_compact[:10]
            )
        )

    with args.compact_schema.open(
        encoding="utf-8",
    ) as handle:
        compact_schema = json.load(
            handle
        )

    cohort_rows = [
        compact_by_sha[
            row["sha256"]
        ]
        for row in manifest
    ]

    (
        _,
        _,
        canonical_original_names,
    ) = (
        allratio
        .original_compact_matrix_named(
            cohort_rows,
            cohort_rows[:1],
            compact_by_sha,
            compact_schema,
        )
    )

    canonical_original_names = list(
        canonical_original_names
    )

    if len(
        canonical_original_names
    ) != 76:
        raise RuntimeError(
            "Expected frozen Compact schema=76, got "
            + str(
                len(
                    canonical_original_names
                )
            )
        )

    # ========================================================
    # Cache
    # ========================================================

    cache_path = resolve_cache(
        args.cache,
        wanted_sha,
    )

    cache = exp.read_cache(
        cache_path
    )

    missing_cache = sorted(
        wanted_sha
        - set(cache)
    )

    if missing_cache:
        raise RuntimeError(
            "Cache missing source APKs: "
            + repr(
                missing_cache[:10]
            )
        )

    # ========================================================
    # Fixed 31-role pair schema
    # ========================================================

    n_roles = int(
        exp.ROLE_VOCABULARY_SIZE
    )

    role_names = (
        discover_role_names()
    )

    if len(role_names) != n_roles:
        raise RuntimeError(
            "Role-name dimension mismatch"
        )

    pair_list = list(
        combinations(
            range(n_roles),
            2,
        )
    )

    pair_index = {
        pair: i
        for i, pair in enumerate(
            pair_list
        )
    }

    expected_pair_dim = (
        n_roles
        * (
            n_roles - 1
        )
        // 2
    )

    if len(pair_list) != (
        expected_pair_dim
    ):
        raise RuntimeError(
            "Pair dimension mismatch"
        )

    print()
    print(
        "Role vocabulary size:",
        n_roles,
    )

    print(
        "Exact pair dimensions:",
        len(pair_list),
    )

    # ========================================================
    # Build old Stats and new Pair representations
    # from exactly the same host-role information.
    # ========================================================

    stats_rows = []
    pair_raw_rows = []
    pair_norm_rows = []
    audit_rows = []

    for i, row in enumerate(
        manifest,
        start=1,
    ):
        sha = row["sha256"]

        (
            stats,
            pair_raw,
            pair_norm,
            audit,
        ) = reconstruct_host_structure(
            cache[
                sha
            ][
                "payload"
            ],
            pair_list,
            pair_index,
        )

        stats_rows.append(
            stats
        )

        pair_raw_rows.append(
            pair_raw
        )

        pair_norm_rows.append(
            pair_norm
        )

        audit_rows.append({
            "sha256":
                sha,

            "label":
                row["label"],

            **audit,
        })

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

    pair_raw_all = np.asarray(
        pair_raw_rows,
        dtype=np.float64,
    )

    pair_norm_all = np.asarray(
        pair_norm_rows,
        dtype=np.float64,
    )

    print()
    print(
        "Shared-Caller Stats dim:",
        stats_all.shape[1],
    )

    print(
        "PairRaw dim:",
        pair_raw_all.shape[1],
    )

    print(
        "PairNorm dim:",
        pair_norm_all.shape[1],
    )

    # ========================================================
    # Feature names
    # ========================================================

    stats_names = [
        f"shared_stats__{i:03d}"
        for i in range(
            stats_all.shape[1]
        )
    ]

    pair_raw_names = []
    pair_norm_names = []

    for a, b in pair_list:

        left = safe_name(
            role_names[a]
        )

        right = safe_name(
            role_names[b]
        )

        stem = (
            f"{a:02d}_{left}"
            "__"
            f"{b:02d}_{right}"
        )

        pair_raw_names.append(
            "pair_raw__"
            + stem
        )

        pair_norm_names.append(
            "pair_norm__"
            + stem
        )

    # ========================================================
    # Fixed experiment configurations
    # ========================================================

    CONFIGS = {
        "C": (
            None,
            [],
        ),

        "C+Stats": (
            stats_all,
            stats_names,
        ),

        "C+PairRaw": (
            pair_raw_all,
            pair_raw_names,
        ),

        "C+PairNorm": (
            pair_norm_all,
            pair_norm_names,
        ),

        "C+Stats+PairRaw": (
            np.concatenate(
                [
                    stats_all,
                    pair_raw_all,
                ],
                axis=1,
            ),
            stats_names
            + pair_raw_names,
        ),

        "C+Stats+PairNorm": (
            np.concatenate(
                [
                    stats_all,
                    pair_norm_all,
                ],
                axis=1,
            ),
            stats_names
            + pair_norm_names,
        ),
    }

    n = len(
        manifest
    )

    labels = np.asarray(
        [
            row["label"]
            for row in manifest
        ],
        dtype=np.int64,
    )

    fold_ids = np.asarray(
        [
            row["fold"]
            for row in manifest
        ],
        dtype=np.int64,
    )

    oof = {
        config:
            np.zeros(
                n,
                dtype=np.float64,
            )
        for config in CONFIGS
    }

    per_fold_rows = []

    # ========================================================
    # Same exact folds and logistic settings as P72.
    # ========================================================

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

        y_train = labels[
            train_idx
        ]

        y_val = labels[
            val_idx
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

        print()
        print(
            "----------------------------------------"
        )
        print(
            "Fold",
            int(fold),
        )
        print(
            "train/val:",
            len(train_idx),
            "/",
            len(val_idx),
        )
        print(
            "----------------------------------------"
        )

        model_seed = (
            int(fold) - 1
        )

        for config, (
            structural_all,
            structural_names,
        ) in CONFIGS.items():

            if structural_all is None:
                x_train = (
                    compact_train.copy()
                )

                x_val = (
                    compact_val.copy()
                )

            else:
                train_raw = (
                    structural_all[
                        train_idx
                    ]
                )

                val_raw = (
                    structural_all[
                        val_idx
                    ]
                )

                (
                    structural_train,
                    structural_val,
                    _,
                ) = (
                    p72
                    .standardize_added(
                        train_raw,
                        val_raw,
                        structural_names,
                    )
                )

                x_train = np.concatenate(
                    [
                        compact_train,
                        structural_train,
                    ],
                    axis=1,
                )

                x_val = np.concatenate(
                    [
                        compact_val,
                        structural_val,
                    ],
                    axis=1,
                )

            (
                probability,
                _,
                _,
            ) = p72.fit_predict(
                x_train,
                y_train,
                x_val,
                seed=model_seed,
            )

            oof[
                config
            ][
                val_idx
            ] = probability

            result = base.metrics(
                y_val,
                probability,
                threshold=THRESHOLD,
            )

            per_fold_rows.append({
                "config":
                    config,

                "fold":
                    int(fold),

                "seed":
                    model_seed,

                "train_n":
                    len(
                        train_idx
                    ),

                "val_n":
                    len(
                        val_idx
                    ),

                "compact_dim":
                    70,

                "structure_dim":
                    0
                    if structural_all is None
                    else structural_all.shape[1],

                "BA":
                    result[
                        "balanced_accuracy"
                    ],

                "AUC":
                    result[
                        "auc"
                    ],

                "ACC":
                    result[
                        "accuracy"
                    ],

                "F1":
                    result[
                        "f1"
                    ],

                "FP":
                    result[
                        "fp"
                    ],

                "FN":
                    result[
                        "fn"
                    ],
            })

            print(
                f"{config:20s}",
                f"BA={result['balanced_accuracy']:.4f}",
                f"ACC={result['accuracy']:.4f}",
                f"AUC={result['auc']:.4f}",
            )

    # ========================================================
    # Critical sanity check:
    # C must reproduce the previous P72 Compact OOF.
    # ========================================================

    reference_c = np.asarray(
        [
            row[
                "reference_C_prob"
            ]
            for row in manifest
        ],
        dtype=np.float64,
    )

    max_diff = float(
        np.max(
            np.abs(
                oof["C"]
                - reference_c
            )
        )
    )

    print()
    print(
        "C reproduction max abs diff:",
        max_diff,
    )

    if max_diff > 1e-7:
        raise RuntimeError(
            "Compact baseline does not reproduce "
            "P72. Stop before interpreting E1."
        )

    # ========================================================
    # OOF summary + fixed/broken
    # ========================================================

    base_probability = (
        oof["C"]
    )

    base_pred = (
        base_probability
        >= THRESHOLD
    ).astype(
        np.int64
    )

    base_error = (
        base_pred != labels
    )

    n_base_error = int(
        np.sum(
            base_error
        )
    )

    base_result = base.metrics(
        labels,
        base_probability,
        threshold=THRESHOLD,
    )

    summary_rows = []
    changed_rows = []

    for config in CONFIGS:

        probability = (
            oof[
                config
            ]
        )

        pred = (
            probability
            >= THRESHOLD
        ).astype(
            np.int64
        )

        result = base.metrics(
            labels,
            probability,
            threshold=THRESHOLD,
        )

        fixed_mask = (
            (base_pred != labels)
            & (pred == labels)
        )

        broken_mask = (
            (base_pred == labels)
            & (pred != labels)
        )

        fixed = int(
            np.sum(
                fixed_mask
            )
        )

        broken = int(
            np.sum(
                broken_mask
            )
        )

        net_fix = (
            fixed
            - broken
        )

        ncr = (
            net_fix
            / n_base_error
            if n_base_error
            else 0.0
        )

        summary_rows.append({
            "config":
                config,

            "n":
                n,

            "OOF_BA":
                result[
                    "balanced_accuracy"
                ],

            "OOF_AUC":
                result[
                    "auc"
                ],

            "OOF_ACC":
                result[
                    "accuracy"
                ],

            "OOF_F1":
                result[
                    "f1"
                ],

            "TP":
                result[
                    "tp"
                ],

            "TN":
                result[
                    "tn"
                ],

            "FP":
                result[
                    "fp"
                ],

            "FN":
                result[
                    "fn"
                ],

            "delta_BA_vs_C":
                result[
                    "balanced_accuracy"
                ]
                - base_result[
                    "balanced_accuracy"
                ],

            "delta_ACC_vs_C":
                result[
                    "accuracy"
                ]
                - base_result[
                    "accuracy"
                ],

            "delta_AUC_vs_C":
                result[
                    "auc"
                ]
                - base_result[
                    "auc"
                ],

            "fixed":
                fixed,

            "broken":
                broken,

            "net_fix":
                net_fix,

            "net_correction_rate":
                ncr,
        })

        if config == "C":
            continue

        changed = np.flatnonzero(
            pred != base_pred
        )

        for index in changed:
            if fixed_mask[
                index
            ]:
                change_type = (
                    "CORRECTION"
                )

            elif broken_mask[
                index
            ]:
                change_type = (
                    "BREAK"
                )

            else:
                change_type = (
                    "OTHER"
                )

            changed_rows.append({
                "config":
                    config,

                "sha256":
                    manifest[
                        index
                    ][
                        "sha256"
                    ],

                "fold":
                    int(
                        fold_ids[
                            index
                        ]
                    ),

                "label":
                    int(
                        labels[
                            index
                        ]
                    ),

                "change_type":
                    change_type,

                "C_probability":
                    float(
                        base_probability[
                            index
                        ]
                    ),

                "config_probability":
                    float(
                        probability[
                            index
                        ]
                    ),

                "C_prediction":
                    int(
                        base_pred[
                            index
                        ]
                    ),

                "config_prediction":
                    int(
                        pred[
                            index
                        ]
                    ),
            })

    # ========================================================
    # Pair support audit.
    # ========================================================

    pair_support_rows = []

    for j, (
        a,
        b,
    ) in enumerate(
        pair_list
    ):
        raw = (
            pair_raw_all[
                :,
                j
            ]
        )

        support = int(
            np.sum(
                raw > 0
            )
        )

        total_count = float(
            np.sum(
                raw
            )
        )

        pair_support_rows.append({
            "pair_index":
                j,

            "role_a":
                a,

            "role_a_name":
                role_names[a],

            "role_b":
                b,

            "role_b_name":
                role_names[b],

            "apk_support":
                support,

            "total_shared_host_count":
                total_count,
        })

    pair_support_rows.sort(
        key=lambda row: (
            -row[
                "apk_support"
            ],
            -row[
                "total_shared_host_count"
            ],
            row[
                "pair_index"
            ],
        )
    )

    # ========================================================
    # Wide OOF predictions
    # ========================================================

    prediction_rows = []

    for i, row in enumerate(
        manifest
    ):
        output = {
            "sha256":
                row[
                    "sha256"
                ],

            "label":
                row[
                    "label"
                ],

            "metadata_label":
                row[
                    "metadata_label"
                ],

            "fold":
                row[
                    "fold"
                ],
        }

        for config in CONFIGS:
            safe_config = (
                config
                .replace(
                    "+",
                    "_plus_",
                )
            )

            probability = float(
                oof[
                    config
                ][i]
            )

            output[
                safe_config
                + "_prob"
            ] = probability

            output[
                safe_config
                + "_pred"
            ] = int(
                probability
                >= THRESHOLD
            )

        prediction_rows.append(
            output
        )

    # ========================================================
    # Persist
    # ========================================================

    write_csv(
        args.output
        / "e1_summary.csv",
        summary_rows,
    )

    write_csv(
        args.output
        / "e1_per_fold.csv",
        per_fold_rows,
    )

    write_csv(
        args.output
        / "e1_oof_predictions.csv",
        prediction_rows,
    )

    if changed_rows:
        write_csv(
            args.output
            / "e1_changed_cases.csv",
            changed_rows,
        )
    else:
        write_csv(
            args.output
            / "e1_changed_cases.csv",
            [],
            [
                "config",
                "sha256",
            ],
        )

    write_csv(
        args.output
        / "e1_structure_audit.csv",
        audit_rows,
    )

    write_csv(
        args.output
        / "e1_pair_support.csv",
        pair_support_rows,
    )

    schema = {
        "source_apk_count":
            n,

        "role_vocabulary_size":
            n_roles,

        "role_names":
            role_names,

        "pair_dimension":
            len(
                pair_list
            ),

        "shared_caller_stats_dimension":
            int(
                stats_all.shape[1]
            ),

        "pair_definition":
            (
                "unordered semantic-role pair "
                "co-hosted by the same sensitive "
                "behavior host method"
            ),

        "normalization":
            (
                "C_ab / sqrt(c_a * c_b), "
                "where c_a is the number of "
                "behavior hosts containing role a"
            ),

        "cache":
            str(
                cache_path
            ),

        "compact_reproduction_max_abs_diff":
            max_diff,
    }

    (
        args.output
        / "e1_schema.json"
    ).write_text(
        json.dumps(
            schema,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    # ========================================================
    # Console summary
    # ========================================================

    print()
    print(
        "========================================"
    )
    print(
        "E1 OOF SUMMARY"
    )
    print(
        "========================================"
    )

    for row in summary_rows:
        print(
            f"{row['config']:20s}"
            f" BA={row['OOF_BA']:.6f}"
            f" ACC={row['OOF_ACC']:.6f}"
            f" AUC={row['OOF_AUC']:.6f}"
            f" dACC={row['delta_ACC_vs_C']:+.6f}"
            f" fixed={row['fixed']:3d}"
            f" broken={row['broken']:3d}"
            f" net={row['net_fix']:+3d}"
        )

    print()
    print(
        "Base errors:",
        n_base_error,
    )

    print(
        "Output:",
        args.output,
    )


if __name__ == "__main__":
    main()
