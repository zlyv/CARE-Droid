#!/usr/bin/env python3
"""
CARE-Droid VoLE25 nested Stage-II candidate generation.

Purpose
-------
Generate leakage-safe nested OOF Stage-I + Host-only Anomaly Bounded
candidate probabilities for learning a Value-of-Local-Evidence router.

Protocol
--------
2017 source only.

For each outer context k:
    outer fold k is NEVER used.

For each remaining original fold j:
    nested validation = fold j
    nested training   = the other three folds

Thus:
    5 outer contexts x 4 inner-validation folds = 20 nested runs.

The original CARE-Droid Stage-II architecture is NOT changed:
    - Host-only
    - Anomaly
    - RGCondAttn
    - FULL correction input
    - bounded correction
    - max_delta = 2.25
    - seeds = 0,1,2 by default

The router is NOT used when generating the candidate expert probability.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch


HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent.parent

# ------------------------------------------------------------
# Import exactly the formal CARE-Droid replay implementation.
# ------------------------------------------------------------

SEARCH_PATHS = [
    PROJECT / "scripts/care-di25-host-input-replay-v1",
    PROJECT / "scripts/care-rolegraph-conditioned-v4-script",
    PROJECT / "scripts/care-rgcondattn-dedup-bounded-source-v1",
    PROJECT / "scripts/care-malben-host-v2",
    PROJECT / "scripts",
]

for path in reversed(SEARCH_PATHS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


from masked_host_model import make_masked_host_class
from experiment_contract import MAX_DELTA, build_di_reference
from source_label_contract import install_source_label_contract

import run_care_rolegraph_conditioned_dualstream_v4 as v4
import run_care_rgcondattn_dedup_bounded_source_v1 as anomaly


ARM = "FULL"
VARIANT = "RGCondAttn"

EXPECTED_N = 799
EXPECTED_FOLDS = (1, 2, 3, 4, 5)
LOCKED_MAX_DELTA = 2.25


def write_csv(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        raise RuntimeError("refusing to write empty CSV: " + str(path))

    with path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(rows[0]),
        )
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path):
    with path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as handle:
        return list(csv.DictReader(handle))


def norm_sha(value):
    return str(value).strip().lower()


def parse_seeds(text):
    values = [
        int(x.strip())
        for x in str(text).split(",")
        if x.strip()
    ]

    if not values:
        raise RuntimeError("empty seed list")

    return values


def build_parser():
    parser = argparse.ArgumentParser(
        description=__doc__
    )

    # --------------------------------------------------------
    # Original v4 source inputs.
    # --------------------------------------------------------

    parser.add_argument(
        "--features",
        type=Path,
        default=Path(
            "artifacts/p1-features/features.csv"
        ),
    )

    parser.add_argument(
        "--schema",
        type=Path,
        default=Path(
            "artifacts/p1-features/feature_schema.json"
        ),
    )

    parser.add_argument(
        "--cache",
        type=Path,
        default=Path(
            "artifacts/care-source-local-assets-v2/"
            "endpoint_centered_tensor_cache_2017.jsonl.gz"
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
        "--source-multiscale-features",
        type=Path,
        default=Path(
            "artifacts/"
            "clipped-intra-inter-structure-ablation-2017-v1-full/"
            "multiscale_features.npz"
        ),
    )

    # This is overwritten after build_di_reference().
    parser.add_argument(
        "--previous-result",
        type=Path,
        default=None,
    )

    parser.add_argument(
        "--care-reference",
        type=Path,
        default=Path(
            "artifacts/care-bounded-fixkeep-v3-full/"
            "source_predictions.csv"
        ),
    )

    parser.add_argument(
        "--region-cache",
        type=Path,
        default=Path(
            "artifacts/care-source-local-assets-v2/"
            "regions_2017.npz"
        ),
    )

    parser.add_argument(
        "--rebuild-region-cache",
        action="store_true",
    )

    # --------------------------------------------------------
    # Locked model/training parameters copied from formal v4.
    # --------------------------------------------------------

    parser.add_argument(
        "--inner-folds",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--split-seed",
        type=int,
        default=20260901,
    )

    parser.add_argument(
        "--max-epochs",
        type=int,
        default=30,
    )

    parser.add_argument(
        "--early-stop-fraction",
        type=float,
        default=0.18,
    )

    parser.add_argument(
        "--early-stop-patience",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--early-stop-min-delta",
        type=float,
        default=1e-4,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
    )

    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=48,
    )

    parser.add_argument(
        "--embedding-dim",
        type=int,
        default=24,
    )

    parser.add_argument(
        "--graph-dim",
        type=int,
        default=32,
    )

    parser.add_argument(
        "--graph-heads",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--graph-layers",
        type=int,
        default=2,
    )

    parser.add_argument(
        "--dropout",
        type=float,
        default=0.20,
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=8e-4,
    )

    parser.add_argument(
        "--weight-decay",
        type=float,
        default=5e-4,
    )

    parser.add_argument(
        "--grad-clip",
        type=float,
        default=5.0,
    )

    # IMPORTANT:
    # v4 default is 1.5, but formal replay locks 2.25.
    parser.add_argument(
        "--max-delta",
        type=float,
        default=LOCKED_MAX_DELTA,
    )

    parser.add_argument(
        "--keep-weight",
        type=float,
        default=2.0,
    )

    parser.add_argument(
        "--fix-weight",
        type=float,
        default=1.0,
    )

    parser.add_argument(
        "--keep-consistency",
        type=float,
        default=0.5,
    )

    parser.add_argument(
        "--shared-prototypes",
        type=int,
        default=32,
    )

    parser.add_argument(
        "--benign-prototypes",
        type=int,
        default=24,
    )

    parser.add_argument(
        "--prototype-regions-per-apk",
        type=int,
        default=16,
    )

    parser.add_argument(
        "--internal-background-coverage",
        type=float,
        default=0.70,
    )

    parser.add_argument(
        "--device",
        default="auto",
    )

    parser.add_argument(
        "--progress-every",
        type=int,
        default=100,
    )

    parser.add_argument(
        "--seeds",
        default="0,1,2",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "artifacts/care-vole25/"
            "nested_stage2"
        ),
    )

    # Useful for one-context pilot.
    parser.add_argument(
        "--outer-fold",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--inner-val-fold",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--force",
        action="store_true",
    )

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    # --------------------------------------------------------
    # Formal contract.
    # --------------------------------------------------------

    if MAX_DELTA != LOCKED_MAX_DELTA:
        raise RuntimeError(
            "BLOCKED_BY_CONTRACT: "
            f"experiment_contract MAX_DELTA={MAX_DELTA}, "
            f"expected={LOCKED_MAX_DELTA}"
        )

    if abs(float(args.max_delta) - LOCKED_MAX_DELTA) > 1e-12:
        raise RuntimeError(
            "BLOCKED_BY_CONTRACT: "
            "max_delta must remain 2.25"
        )

    if args.inner_folds != 4:
        raise RuntimeError(
            "BLOCKED_BY_CONTRACT: "
            "internal Stage-II Global24 OOF must remain 4-fold"
        )

    seeds = parse_seeds(args.seeds)

    args.output.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Build the exact frozen source manifest.
    #
    # We use this for identity / labels / original fold only.
    # The old DI route does NOT enter nested expert training.
    # --------------------------------------------------------

    build_di_reference(
        PROJECT,
        args.output,
    )

    args.previous_result = (
        args.output /
        "preflight/di25_reference"
    )

    install_source_label_contract(v4)

    # --------------------------------------------------------
    # Use exact formal FULL Host-only model.
    # --------------------------------------------------------

    Model = make_masked_host_class(
        anomaly,
        ARM,
    )

    v4.RoleGraphConditionedEvidence = Model

    device = v4.resolve_device(
        args.device
    )

    print(
        "Device:",
        device,
        "seeds:",
        seeds,
        flush=True,
    )

    # --------------------------------------------------------
    # Load source-only assets exactly once.
    # --------------------------------------------------------

    (
        manifest,
        source_rows,
        _,
        compact_by_sha,
        compact_schema,
        original_names,
    ) = v4.audit.load_inputs(args)

    if len(manifest) != EXPECTED_N:
        raise RuntimeError(
            f"unexpected source N={len(manifest)}"
        )

    source_sha = np.asarray([
        norm_sha(row["sha256"])
        for row in manifest
    ])

    source_y = np.asarray(
        [row["label"] for row in manifest],
        dtype=np.int64,
    )

    source_fold = np.asarray(
        [row["fold"] for row in manifest],
        dtype=np.int64,
    )

    folds = tuple(
        sorted(
            int(x)
            for x in np.unique(source_fold)
        )
    )

    if folds != EXPECTED_FOLDS:
        raise RuntimeError(
            f"unexpected folds={folds}"
        )

    if len(set(source_sha.tolist())) != EXPECTED_N:
        raise RuntimeError(
            "source SHA is not unique"
        )

    endpoint_cache = (
        v4.audit.exp.read_cache(args.cache)
    )

    source_s7, _ = v4.v3.load_source_s7(
        source_rows,
        endpoint_cache,
        args,
    )

    if source_s7.shape[0] != EXPECTED_N:
        raise RuntimeError(
            "source S7 row mismatch"
        )

    (
        host_bags,
        bag_audit,
        region_cache_loaded,
    ) = v4.get_source_host_bags(
        source_rows,
        endpoint_cache,
        args,
    )

    if len(host_bags) != EXPECTED_N:
        raise RuntimeError(
            "Host bag row mismatch"
        )

    # --------------------------------------------------------
    # Select requested contexts.
    # --------------------------------------------------------

    outer_folds = list(folds)

    if args.outer_fold is not None:
        if args.outer_fold not in folds:
            raise RuntimeError(
                "invalid --outer-fold"
            )
        outer_folds = [
            int(args.outer_fold)
        ]

    context_dir = (
        args.output / "contexts"
    )

    context_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    fit_audit_rows = []

    # --------------------------------------------------------
    # 20 nested runs:
    # outer context k x inner-validation j != k.
    # --------------------------------------------------------

    for outer_fold in outer_folds:

        inner_folds = [
            fold
            for fold in folds
            if fold != outer_fold
        ]

        if args.inner_val_fold is not None:
            if args.inner_val_fold not in inner_folds:
                raise RuntimeError(
                    "--inner-val-fold must differ "
                    "from --outer-fold"
                )

            inner_folds = [
                int(args.inner_val_fold)
            ]

        for inner_val_fold in inner_folds:

            print()
            print(
                "=" * 72,
                flush=True,
            )

            print(
                "OUTER CONTEXT:",
                outer_fold,
                "INNER VAL:",
                inner_val_fold,
                flush=True,
            )

            print(
                "=" * 72,
                flush=True,
            )

            context_file = (
                context_dir /
                (
                    f"outer_{outer_fold}"
                    f"__inner_{inner_val_fold}.csv"
                )
            )

            if (
                context_file.is_file()
                and not args.force
            ):
                existing = read_csv(
                    context_file
                )

                expected_n = int(
                    np.sum(
                        source_fold
                        == inner_val_fold
                    )
                )

                if len(existing) == expected_n:
                    print(
                        "SKIP existing:",
                        context_file,
                        flush=True,
                    )
                    continue

                raise RuntimeError(
                    "existing context file has "
                    "wrong row count: "
                    + str(context_file)
                )

            # ------------------------------------------------
            # Leakage-safe indices.
            #
            # Train:
            # neither outer validation nor nested validation.
            #
            # Validation:
            # exactly the selected inner fold.
            # ------------------------------------------------

            train_mask = (
                (source_fold != outer_fold)
                &
                (source_fold != inner_val_fold)
            )

            val_mask = (
                source_fold == inner_val_fold
            )

            train_index = np.flatnonzero(
                train_mask
            )

            val_index = np.flatnonzero(
                val_mask
            )

            if np.any(
                source_fold[train_index]
                == outer_fold
            ):
                raise RuntimeError(
                    "OUTER LEAKAGE in train"
                )

            if np.any(
                source_fold[val_index]
                == outer_fold
            ):
                raise RuntimeError(
                    "outer fold entered nested validation"
                )

            expected_train_folds = {
                fold
                for fold in folds
                if fold not in (
                    outer_fold,
                    inner_val_fold,
                )
            }

            actual_train_folds = set(
                source_fold[
                    train_index
                ].tolist()
            )

            if (
                actual_train_folds
                != expected_train_folds
            ):
                raise RuntimeError(
                    "bad nested train folds: "
                    f"expected="
                    f"{expected_train_folds}, "
                    f"got={actual_train_folds}"
                )

            print(
                "train folds:",
                sorted(
                    actual_train_folds
                ),
                "train_n:",
                len(train_index),
                flush=True,
            )

            print(
                "nested val fold:",
                inner_val_fold,
                "val_n:",
                len(val_index),
                flush=True,
            )

            # ------------------------------------------------
            # Compact / Global24 matrices.
            # ------------------------------------------------

            train_rows = [
                source_rows[index]
                for index in train_index
            ]

            val_rows = [
                source_rows[index]
                for index in val_index
            ]

            (
                x_train,
                x_val,
                names,
            ) = (
                v4.audit.compact_matrix_variant(
                    v4.VARIANT,
                    train_rows,
                    val_rows,
                    compact_by_sha,
                    compact_schema,
                    original_names,
                )
            )

            (
                global_index,
                _,
            ) = (
                v4.evidence.split_view_indices(
                    names
                )
            )

            global_train = np.asarray(
                x_train[:, global_index],
                dtype=np.float32,
            )

            global_val = np.asarray(
                x_val[:, global_index],
                dtype=np.float32,
            )

            train_y = source_y[
                train_index
            ]

            # ------------------------------------------------
            # Stage-I training probability for correction-head
            # training:
            #
            # STRICT internal OOF, exactly as original v4.
            # ------------------------------------------------

            split_seed = (
                int(args.split_seed)
                + 100000 * int(outer_fold)
                + 1000 * int(inner_val_fold)
            )

            model_seed = (
                int(args.split_seed)
                + 1000000 * int(outer_fold)
                + 10000 * int(inner_val_fold)
            )

            print(
                "  internal OOF Global24 "
                "for nested training...",
                flush=True,
            )

            base_train = (
                v4.inner_oof_global24(
                    train_rows,
                    train_y,
                    compact_by_sha,
                    compact_schema,
                    original_names,
                    args.inner_folds,
                    split_seed,
                    model_seed,
                )
            )

            if np.any(
                ~np.isfinite(base_train)
            ):
                raise RuntimeError(
                    "non-finite base_train"
                )

            # ------------------------------------------------
            # Stage-I probability for nested validation.
            #
            # IMPORTANT:
            # DO NOT use ordinary source outer-OOF p_stage1,
            # because that model may have seen the current
            # outer context.
            #
            # Fit only on the three nested training folds.
            # ------------------------------------------------

            nested_stage1_seed = (
                int(args.split_seed)
                + 10000000
                + 100000 * int(outer_fold)
                + 1000 * int(inner_val_fold)
            )

            base_val = v4.g24.fit_view(
                x_train,
                x_val,
                train_y,
                global_index,
                nested_stage1_seed,
            )

            base_val = np.asarray(
                base_val,
                dtype=np.float64,
            )

            if np.any(
                ~np.isfinite(base_val)
            ):
                raise RuntimeError(
                    "non-finite base_val"
                )

            # ------------------------------------------------
            # Host / role graph data.
            # ------------------------------------------------

            train_bags = [
                host_bags[index]
                for index in train_index
            ]

            val_bags = [
                host_bags[index]
                for index in val_index
            ]

            # ------------------------------------------------
            # Formal candidate Stage-II.
            #
            # No router applied here.
            # ------------------------------------------------

            fold_seeds = [
                (
                    10000000
                    * int(outer_fold)
                    + 100000
                    * int(inner_val_fold)
                    + int(seed)
                )
                for seed in seeds
            ]

            print(
                "  fitting Host-only "
                "Anomaly FULL Bounded candidate...",
                flush=True,
            )

            (
                candidate_prob,
                diagnostic,
                audits,
            ) = v4.fit_predict_variant(
                global_train,
                global_val,
                source_s7[train_index],
                source_s7[val_index],
                train_bags,
                val_bags,
                base_train,
                base_val,
                train_y,
                VARIANT,
                fold_seeds,
                args,
                device,
            )

            candidate_prob = np.asarray(
                candidate_prob,
                dtype=np.float64,
            )

            diagnostic = np.asarray(
                diagnostic,
                dtype=np.float64,
            )

            if len(candidate_prob) != len(
                val_index
            ):
                raise RuntimeError(
                    "candidate length mismatch"
                )

            if np.any(
                ~np.isfinite(
                    candidate_prob
                )
            ):
                raise RuntimeError(
                    "non-finite candidate probability"
                )

            if diagnostic.shape != (
                len(val_index),
                len(v4.DIAGNOSTIC_NAMES),
            ):
                raise RuntimeError(
                    "diagnostic shape mismatch: "
                    + repr(diagnostic.shape)
                )

            # ------------------------------------------------
            # Save immediately so interrupted runs resume.
            # ------------------------------------------------

            rows = []

            for local_index, source_index in enumerate(
                val_index
            ):
                row = {
                    "context_outer_fold":
                        int(outer_fold),

                    "inner_val_fold":
                        int(inner_val_fold),

                    "sample_original_fold":
                        int(
                            source_fold[
                                source_index
                            ]
                        ),

                    "sha256":
                        source_sha[
                            source_index
                        ],

                    "label":
                        int(
                            source_y[
                                source_index
                            ]
                        ),

                    "stage1_prob":
                        float(
                            base_val[
                                local_index
                            ]
                        ),

                    "stage1_pred":
                        int(
                            base_val[
                                local_index
                            ]
                            >= 0.5
                        ),

                    "candidate_prob":
                        float(
                            candidate_prob[
                                local_index
                            ]
                        ),

                    "candidate_pred":
                        int(
                            candidate_prob[
                                local_index
                            ]
                            >= 0.5
                        ),
                }

                for column, name in enumerate(
                    v4.DIAGNOSTIC_NAMES
                ):
                    row[
                        "candidate_" + name
                    ] = float(
                        diagnostic[
                            local_index,
                            column,
                        ]
                    )

                rows.append(row)

            write_csv(
                context_file,
                rows,
            )

            for audit_row in audits:
                fit_audit_rows.append({
                    "context_outer_fold":
                        int(outer_fold),

                    "inner_val_fold":
                        int(inner_val_fold),

                    "train_folds":
                        "|".join(
                            str(x)
                            for x in sorted(
                                actual_train_folds
                            )
                        ),

                    "train_n":
                        int(
                            len(train_index)
                        ),

                    "validation_n":
                        int(
                            len(val_index)
                        ),

                    **audit_row,
                })

            if fit_audit_rows:
                write_csv(
                    args.output /
                    "nested_fit_audit_partial.csv",
                    fit_audit_rows,
                )

            print(
                "SAVED:",
                context_file,
                flush=True,
            )

    # --------------------------------------------------------
    # Aggregate all completed contexts.
    # --------------------------------------------------------

    context_files = sorted(
        context_dir.glob(
            "outer_*__inner_*.csv"
        )
    )

    combined = []

    for path in context_files:
        combined.extend(
            read_csv(path)
        )

    # If running all contexts, enforce complete contract.
    full_run = (
        args.outer_fold is None
        and args.inner_val_fold is None
    )

    if full_run:
        expected_rows = (
            EXPECTED_N
            * (
                len(EXPECTED_FOLDS)
                - 1
            )
        )

        if len(context_files) != 20:
            raise RuntimeError(
                f"expected 20 context files, "
                f"got={len(context_files)}"
            )

        if len(combined) != expected_rows:
            raise RuntimeError(
                f"expected {expected_rows} "
                f"nested rows, "
                f"got={len(combined)}"
            )

        # Every SHA occurs in exactly four outer contexts.
        counts = {}

        leakage = 0

        pairs = set()

        for row in combined:
            sha = norm_sha(
                row["sha256"]
            )

            counts[sha] = (
                counts.get(sha, 0)
                + 1
            )

            outer = int(
                row[
                    "context_outer_fold"
                ]
            )

            original = int(
                row[
                    "sample_original_fold"
                ]
            )

            if outer == original:
                leakage += 1

            key = (
                outer,
                sha,
            )

            if key in pairs:
                raise RuntimeError(
                    "duplicate "
                    "(outer context, SHA)"
                )

            pairs.add(key)

        if len(counts) != EXPECTED_N:
            raise RuntimeError(
                "nested unique SHA mismatch"
            )

        if set(
            counts.values()
        ) != {4}:
            raise RuntimeError(
                "each SHA must appear "
                "exactly four times"
            )

        if leakage != 0:
            raise RuntimeError(
                f"outer leakage={leakage}"
            )

    if combined:
        write_csv(
            args.output /
            "nested_stage2_predictions.csv",
            combined,
        )

    if fit_audit_rows:
        write_csv(
            args.output /
            "nested_fit_audit.csv",
            fit_audit_rows,
        )

    contract = {
        "experiment":
            "CARE-Droid VoLE25 "
            "nested Stage-II candidate",

        "source_only":
            True,

        "future_data_loaded":
            False,

        "expected_source_n":
            EXPECTED_N,

        "outer_folds":
            list(EXPECTED_FOLDS),

        "nested_contexts_expected":
            20,

        "nested_rows_expected":
            EXPECTED_N * 4,

        "outer_context_excluded_from_train":
            True,

        "outer_context_excluded_from_nested_validation":
            True,

        "stage2_representation":
            "Host-only",

        "stage2_semantics":
            "Anomaly",

        "stage2_variant":
            VARIANT,

        "input_arm":
            ARM,

        "max_delta_logit":
            LOCKED_MAX_DELTA,

        "seeds":
            seeds,

        "router_used_for_candidate_generation":
            False,

        "internal_stage1_train_probability":
            "strict inner OOF Global24",

        "nested_validation_stage1_probability":
            "Global24 fitted only on "
            "three nested training folds",
    }

    (
        args.output /
        "nested_contract.json"
    ).write_text(
        json.dumps(
            contract,
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    print()
    print("=" * 72)
    print("NESTED STAGE-II COMPLETE")
    print("=" * 72)

    print(
        "context files =",
        len(context_files),
    )

    print(
        "combined rows =",
        len(combined),
    )

    print(
        "output =",
        args.output,
    )


if __name__ == "__main__":
    main()
