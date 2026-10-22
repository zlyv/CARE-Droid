#!/usr/bin/env python3
"""
Leakage-safe nested CARE-Droid classifier committee for VoLE25.

For each outer context k:
    outer fold k is completely excluded.

For each remaining original fold j:
    nested validation = fold j
    nested training   = the other three folds.

The existing CARE committee implementation is reused directly:
LR / RF / XGBoost / SVM + training-side inner-OOF Platt calibration.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent.parent
SCRIPTS = PROJECT / "scripts"

for path in (
    SCRIPTS / "care-committee-uncertainty-router-v1",
    SCRIPTS / "care-committee-decision-instability-router-v1",
    SCRIPTS / "care-multiview-conformal-router-source-v1",
    SCRIPTS / "care-rolegraph-conditioned-v4-script",
    SCRIPTS / "care-rgcondattn-dedup-bounded-source-v1",
    SCRIPTS / "care-di25-host-input-replay-v1",
    SCRIPTS,
):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


import run_multiview_conformal_router_source_v1 as mv

from committee_uncertainty_core import (
    MEMBER_NAMES,
    committee_scores,
    fit_calibrated_committee,
)

from decision_instability_core import (
    decision_instability,
)

from source_label_contract import (
    install_source_label_contract,
)


EXPECTED_N = 799
EXPECTED_FOLDS = (1, 2, 3, 4, 5)


def norm_sha(x):
    return str(x).strip().lower()


def write_csv(path: Path, rows):
    if not rows:
        raise RuntimeError("refusing empty CSV")

    path.parent.mkdir(parents=True, exist_ok=True)

    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)

    with path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fields,
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


def parser():
    p = argparse.ArgumentParser(
        description=__doc__
    )

    p.add_argument(
        "--features",
        type=Path,
        default=Path(
            "artifacts/p1-features/features.csv"
        ),
    )

    p.add_argument(
        "--schema",
        type=Path,
        default=Path(
            "artifacts/p1-features/feature_schema.json"
        ),
    )

    p.add_argument(
        "--p72-oof",
        type=Path,
        default=Path(
            "artifacts/"
            "p72-verified-dependency-source-ablation/"
            "p72_oof_predictions.csv"
        ),
    )

    p.add_argument(
        "--inner-folds",
        type=int,
        default=4,
    )

    p.add_argument(
        "--jobs",
        type=int,
        default=4,
    )

    # Formal committee seed.
    p.add_argument(
        "--committee-seed",
        type=int,
        default=20260915,
    )

    # Must agree with nested Stage-II Stage-I construction.
    p.add_argument(
        "--stage1-seed",
        type=int,
        default=20260901,
    )

    p.add_argument(
        "--outer-fold",
        type=int,
        default=None,
    )

    p.add_argument(
        "--inner-val-fold",
        type=int,
        default=None,
    )

    p.add_argument(
        "--output",
        type=Path,
        default=Path(
            "artifacts/care-vole25/"
            "nested_committee"
        ),
    )

    p.add_argument(
        "--force",
        action="store_true",
    )

    return p


def main():
    args = parser().parse_args()

    if args.inner_folds != 4:
        raise RuntimeError(
            "BLOCKED_BY_CONTRACT: "
            "committee inner-fold calibration must remain 4"
        )

    args.output.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------
    # Formal recovered-source label contract.
    #
    # This is the same compatibility contract used by
    # the locked Host-only replay. It prevents legacy
    # loaders from rejecting textual source labels such
    # as "malware" / "benign".
    # --------------------------------------------------

    install_source_label_contract(
        mv.v4
    )

    (
        manifest,
        source_rows,
        _,
        compact_by_sha,
        compact_schema,
        original_names,
    ) = mv.v4.audit.load_inputs(args)

    labels = np.asarray(
        [int(row["label"]) for row in manifest],
        dtype=np.int64,
    )

    folds = np.asarray(
        [int(row["fold"]) for row in manifest],
        dtype=np.int64,
    )

    sha = np.asarray([
        norm_sha(row["sha256"])
        for row in manifest
    ])

    if len(labels) != EXPECTED_N:
        raise RuntimeError(
            f"unexpected source N={len(labels)}"
        )

    if tuple(
        sorted(
            int(x)
            for x in np.unique(folds)
        )
    ) != EXPECTED_FOLDS:
        raise RuntimeError(
            "source folds changed"
        )

    if len(set(sha.tolist())) != EXPECTED_N:
        raise RuntimeError(
            "SHA uniqueness failed"
        )

    years = {
        str(row.get(
            "metadata_year",
            ""
        )).strip()
        for row in source_rows
    }

    if years != {"2017"}:
        raise RuntimeError(
            "SOURCE-ONLY GUARD FAILED: "
            + repr(years)
        )

    outer_folds = list(
        EXPECTED_FOLDS
    )

    if args.outer_fold is not None:
        if args.outer_fold not in outer_folds:
            raise RuntimeError(
                "bad outer fold"
            )

        outer_folds = [
            args.outer_fold
        ]

    context_dir = (
        args.output / "contexts"
    )

    context_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    calibration_audit = []

    for outer_fold in outer_folds:

        inner_folds = [
            f for f in EXPECTED_FOLDS
            if f != outer_fold
        ]

        if args.inner_val_fold is not None:

            if (
                args.inner_val_fold
                not in inner_folds
            ):
                raise RuntimeError(
                    "inner val must differ "
                    "from outer context"
                )

            inner_folds = [
                args.inner_val_fold
            ]

        for inner_val_fold in inner_folds:

            print()
            print("=" * 72)
            print(
                "OUTER CONTEXT:",
                outer_fold,
                "INNER VAL:",
                inner_val_fold,
            )
            print("=" * 72)

            out_file = (
                context_dir /
                (
                    f"outer_{outer_fold}"
                    f"__inner_{inner_val_fold}.csv"
                )
            )

            if (
                out_file.is_file()
                and not args.force
            ):
                old = read_csv(out_file)

                expected = int(
                    np.sum(
                        folds
                        == inner_val_fold
                    )
                )

                if len(old) == expected:
                    print(
                        "SKIP existing:",
                        out_file,
                    )
                    continue

                raise RuntimeError(
                    "bad existing context file"
                )

            # -----------------------------------------
            # Strict nested split.
            # -----------------------------------------

            train_mask = (
                (folds != outer_fold)
                &
                (folds != inner_val_fold)
            )

            val_mask = (
                folds == inner_val_fold
            )

            train_index = np.flatnonzero(
                train_mask
            )

            val_index = np.flatnonzero(
                val_mask
            )

            train_folds = set(
                folds[
                    train_index
                ].tolist()
            )

            expected_train_folds = {
                f for f in EXPECTED_FOLDS
                if f not in (
                    outer_fold,
                    inner_val_fold,
                )
            }

            if (
                train_folds
                != expected_train_folds
            ):
                raise RuntimeError(
                    "nested train folds wrong"
                )

            if np.any(
                folds[train_index]
                == outer_fold
            ):
                raise RuntimeError(
                    "OUTER LEAKAGE"
                )

            print(
                "train folds:",
                sorted(train_folds),
                "train_n:",
                len(train_index),
            )

            print(
                "nested val fold:",
                inner_val_fold,
                "val_n:",
                len(val_index),
            )

            train_rows = [
                source_rows[i]
                for i in train_index
            ]

            val_rows = [
                source_rows[i]
                for i in val_index
            ]

            (
                x_train,
                x_val,
                names,
            ) = mv.global_matrices(
                train_rows,
                val_rows,
                compact_by_sha,
                compact_schema,
                original_names,
            )

            if len(names) != 24:
                raise RuntimeError(
                    "committee did not receive "
                    "exact Global24"
                )

            # -----------------------------------------
            # Nested Stage-I.
            #
            # Same seed rule as run_nested_stage2.py.
            # This later allows exact alignment audit.
            # -----------------------------------------

            global_index = np.arange(
                24,
                dtype=np.int64,
            )

            nested_stage1_seed = (
                int(args.stage1_seed)
                + 10000000
                + 100000 * int(outer_fold)
                + 1000 * int(inner_val_fold)
            )

            stage1_prob = (
                mv.v4.g24.fit_view(
                    x_train,
                    x_val,
                    labels[train_index],
                    global_index,
                    nested_stage1_seed,
                )
            )

            stage1_prob = np.asarray(
                stage1_prob,
                dtype=np.float64,
            )

            if (
                stage1_prob.shape
                != (len(val_index),)
            ):
                raise RuntimeError(
                    "bad nested Stage-I shape"
                )

            if not np.isfinite(
                stage1_prob
            ).all():
                raise RuntimeError(
                    "non-finite Stage-I"
                )

            # -----------------------------------------
            # EXACT established committee.
            # No LR/RF/XGB/SVM reimplementation here.
            # -----------------------------------------

            split_seed = (
                int(args.committee_seed)
                + 1000000
                * int(outer_fold)
                + 10000
                * int(inner_val_fold)
            )

            model_seed = (
                int(args.committee_seed)
                + 10000000
                * int(outer_fold)
                + 100000
                * int(inner_val_fold)
            )

            fitted = (
                fit_calibrated_committee(
                    x_train,
                    labels[train_index],
                    x_val,
                    inner_folds=args.inner_folds,
                    split_seed=split_seed,
                    model_seed=model_seed,
                    jobs=args.jobs,
                )
            )

            member = np.asarray(
                fitted.target_probability,
                dtype=np.float64,
            )

            if member.shape != (
                len(val_index),
                4,
            ):
                raise RuntimeError(
                    "bad committee shape: "
                    + repr(member.shape)
                )

            if not np.isfinite(
                member
            ).all():
                raise RuntimeError(
                    "non-finite committee"
                )

            # -----------------------------------------
            # Formal committee scores.
            # -----------------------------------------

            score_detail = (
                committee_scores(member)
            )

            di = decision_instability(
                stage1_prob,
                member,
            )

            support = np.asarray(
                di["support_by_member"],
                dtype=np.float64,
            )

            if support.shape != member.shape:
                raise RuntimeError(
                    "support matrix mismatch"
                )

            # -----------------------------------------
            # Optional strong consistency check with
            # already-generated nested Stage-II file.
            # -----------------------------------------

            stage2_file = (
                PROJECT /
                "artifacts/care-vole25/"
                "nested_stage2/contexts/"
                f"outer_{outer_fold}"
                f"__inner_{inner_val_fold}.csv"
            )

            stage2_prob_by_sha = None

            if stage2_file.is_file():

                stage2_rows = (
                    read_csv(stage2_file)
                )

                stage2_prob_by_sha = {
                    norm_sha(r["sha256"]):
                    float(r["stage1_prob"])
                    for r in stage2_rows
                }

                delta = []

                for local, src_i in enumerate(
                    val_index
                ):
                    key = sha[src_i]

                    if (
                        key
                        not in stage2_prob_by_sha
                    ):
                        raise RuntimeError(
                            "Stage-II SHA mismatch"
                        )

                    delta.append(
                        abs(
                            float(
                                stage1_prob[
                                    local
                                ]
                            )
                            -
                            stage2_prob_by_sha[
                                key
                            ]
                        )
                    )

                max_delta = max(
                    delta,
                    default=0.0,
                )

                print(
                    "max |committee nested Stage-I "
                    "- Stage-II nested Stage-I| =",
                    max_delta,
                )

                if max_delta > 1e-10:
                    raise RuntimeError(
                        "nested Stage-I mismatch "
                        "between two pipelines"
                    )

            # -----------------------------------------
            # Save deployment-time committee evidence.
            # -----------------------------------------

            rows = []

            for local, src_i in enumerate(
                val_index
            ):

                votes = (
                    member[local]
                    >= 0.5
                ).astype(np.float64)

                vote_frac = float(
                    np.mean(votes)
                )

                row = {
                    "context_outer_fold":
                        int(outer_fold),

                    "inner_val_fold":
                        int(inner_val_fold),

                    "sample_original_fold":
                        int(folds[src_i]),

                    "sha256":
                        sha[src_i],

                    "label":
                        int(labels[src_i]),

                    "stage1_prob":
                        float(
                            stage1_prob[local]
                        ),

                    "stage1_pred":
                        int(
                            stage1_prob[local]
                            >= 0.5
                        ),

                    "p_lr":
                        float(
                            member[local, 0]
                        ),

                    "p_rf":
                        float(
                            member[local, 1]
                        ),

                    "p_xgb":
                        float(
                            member[local, 2]
                        ),

                    "p_svm":
                        float(
                            member[local, 3]
                        ),

                    "p_bar":
                        float(
                            di["p_bar"][local]
                        ),

                    "committee_support_for_stage1":
                        float(
                            di[
                                "committee_support"
                            ][local]
                        ),

                    "U_DI":
                        float(
                            di[
                                "uncertainty"
                            ][local]
                        ),

                    "support_lr":
                        float(
                            support[local, 0]
                        ),

                    "support_rf":
                        float(
                            support[local, 1]
                        ),

                    "support_xgb":
                        float(
                            support[local, 2]
                        ),

                    "support_svm":
                        float(
                            support[local, 3]
                        ),

                    # Useful cheap derived signals.
                    "committee_std":
                        float(
                            np.std(
                                member[local]
                            )
                        ),

                    "committee_range":
                        float(
                            np.max(
                                member[local]
                            )
                            -
                            np.min(
                                member[local]
                            )
                        ),

                    "committee_vote_fraction_malicious":
                        vote_frac,

                    "committee_vote_disagreement":
                        float(
                            1.0
                            -
                            abs(
                                vote_frac
                                - 0.5
                            )
                            * 2.0
                        ),

                    # Saved for later ablation,
                    # not required in VoLE-v1.
                    "U_PE":
                        float(
                            score_detail[
                                "predictive_entropy"
                            ][local]
                        ),

                    "U_MI":
                        float(
                            score_detail[
                                "mutual_information"
                            ][local]
                        ),
                }

                rows.append(row)

            write_csv(
                out_file,
                rows,
            )

            for audit_row in fitted.audit:
                calibration_audit.append({
                    "context_outer_fold":
                        int(outer_fold),

                    "inner_val_fold":
                        int(inner_val_fold),

                    "train_folds":
                        "|".join(
                            str(x)
                            for x in sorted(
                                train_folds
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

                    "outer_context_labels_used":
                        False,

                    **audit_row,
                })

            if calibration_audit:
                write_csv(
                    args.output /
                    "nested_calibration_audit_partial.csv",
                    calibration_audit,
                )

            print(
                "SAVED:",
                out_file,
            )

    # ---------------------------------------------
    # Aggregate contexts.
    # ---------------------------------------------

    files = sorted(
        context_dir.glob(
            "outer_*__inner_*.csv"
        )
    )

    combined = []

    for path in files:
        combined.extend(
            read_csv(path)
        )

    full_run = (
        args.outer_fold is None
        and args.inner_val_fold is None
    )

    if full_run:

        if len(files) != 20:
            raise RuntimeError(
                f"expected 20 context files, "
                f"got {len(files)}"
            )

        if len(combined) != 3196:
            raise RuntimeError(
                f"expected 3196 rows, "
                f"got {len(combined)}"
            )

        count = {}
        seen = set()
        leakage = 0

        for row in combined:

            key_sha = norm_sha(
                row["sha256"]
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
                key_sha,
            )

            if key in seen:
                raise RuntimeError(
                    "duplicate "
                    "(outer context,SHA)"
                )

            seen.add(key)

            count[key_sha] = (
                count.get(
                    key_sha,
                    0,
                )
                + 1
            )

        if leakage:
            raise RuntimeError(
                f"outer leakage={leakage}"
            )

        if len(count) != 799:
            raise RuntimeError(
                "unique SHA mismatch"
            )

        if set(
            count.values()
        ) != {4}:
            raise RuntimeError(
                "each SHA must occur "
                "in exactly four contexts"
            )

    if combined:
        write_csv(
            args.output /
            "nested_committee_predictions.csv",
            combined,
        )

    if calibration_audit:
        write_csv(
            args.output /
            "nested_calibration_audit.csv",
            calibration_audit,
        )

    contract = {
        "experiment":
            "CARE-Droid VoLE25 "
            "nested committee",

        "source_only":
            True,

        "future_rows_loaded":
            0,

        "source_n":
            799,

        "outer_folds":
            list(EXPECTED_FOLDS),

        "nested_contexts_expected":
            20,

        "nested_rows_expected":
            3196,

        "committee_members":
            list(MEMBER_NAMES),

        "committee_input":
            "exact Global24 24 dimensions",

        "committee_calibration":
            "training-side inner-OOF "
            "Platt sigmoid",

        "committee_inner_folds":
            4,

        "committee_seed":
            int(args.committee_seed),

        "stage1_seed":
            int(args.stage1_seed),

        "outer_context_labels_used":
            False,
    }

    (
        args.output /
        "nested_committee_contract.json"
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
    print("NESTED COMMITTEE COMPLETE")
    print("=" * 72)
    print("context files =", len(files))
    print("combined rows =", len(combined))


if __name__ == "__main__":
    main()
