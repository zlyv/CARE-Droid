#!/usr/bin/env python3

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT = Path("/home/boyuanjie/projects/tasa-droid")

COMMITTEE_ROOT = (
    PROJECT
    / "artifacts"
    / "care-vole25"
    / "nested_committee"
    / "contexts"
)

STAGE2_ROOT = (
    PROJECT
    / "artifacts"
    / "care-vole25"
    / "nested_stage2"
    / "contexts"
)

OUT = (
    PROJECT
    / "artifacts"
    / "care-di25-host-m0-source-stability-v1"
    / "strict_reference"
)

OUT.mkdir(
    parents=True,
    exist_ok=True,
)


# ---------------------------------------------------------
# Reuse EXACT historical DI-budget selector.
# Do not reimplement sorting / tie handling here.
# ---------------------------------------------------------

ROUTER_SCRIPT = (
    PROJECT
    / "scripts"
    / "care-committee-uncertainty-router-v1"
    / "run_committee_source_v1.py"
)

spec = importlib.util.spec_from_file_location(
    "historical_di_router",
    ROUTER_SCRIPT,
)

router = importlib.util.module_from_spec(spec)
spec.loader.exec_module(router)

if not hasattr(
    router,
    "exact_top_fraction",
):
    raise RuntimeError(
        "historical exact_top_fraction not found"
    )

exact_top_fraction = router.exact_top_fraction


def norm_sha(series):
    return (
        series.astype(str)
        .str.strip()
        .str.lower()
    )


summary_rows = []
inner_rows = []


for outer_fold in range(1, 6):

    context_parts = []

    print()
    print(
        "=" * 90
    )
    print(
        f"OUTER CONTEXT {outer_fold}"
    )
    print(
        "=" * 90
    )

    for inner_fold in range(1, 6):

        if inner_fold == outer_fold:
            continue

        committee_file = (
            COMMITTEE_ROOT
            / (
                f"outer_{outer_fold}"
                f"__inner_{inner_fold}.csv"
            )
        )

        stage2_file = (
            STAGE2_ROOT
            / (
                f"outer_{outer_fold}"
                f"__inner_{inner_fold}.csv"
            )
        )

        if not committee_file.is_file():
            raise RuntimeError(
                f"missing committee context: {committee_file}"
            )

        if not stage2_file.is_file():
            raise RuntimeError(
                f"missing Stage-II context: {stage2_file}"
            )

        c = pd.read_csv(
            committee_file
        )

        s = pd.read_csv(
            stage2_file
        )

        required_committee = {
            "context_outer_fold",
            "inner_val_fold",
            "sample_original_fold",
            "sha256",
            "label",
            "stage1_prob",
            "stage1_pred",
            "U_DI",
        }

        if not required_committee.issubset(
            set(c.columns)
        ):
            raise RuntimeError(
                "committee columns changed"
            )

        required_stage2 = {
            "context_outer_fold",
            "inner_val_fold",
            "sha256",
            "stage1_prob",
        }

        if not required_stage2.issubset(
            set(s.columns)
        ):
            raise RuntimeError(
                "Stage-II columns changed"
            )

        c["sha256"] = norm_sha(
            c["sha256"]
        )

        s["sha256"] = norm_sha(
            s["sha256"]
        )

        if c["sha256"].duplicated().any():
            raise RuntimeError(
                "duplicate committee SHA"
            )

        if s["sha256"].duplicated().any():
            raise RuntimeError(
                "duplicate Stage-II SHA"
            )

        # ---------------------------------------------
        # Context contract.
        # ---------------------------------------------

        if set(
            c["context_outer_fold"]
            .astype(int)
            .unique()
        ) != {outer_fold}:
            raise RuntimeError(
                "bad outer context"
            )

        if set(
            c["inner_val_fold"]
            .astype(int)
            .unique()
        ) != {inner_fold}:
            raise RuntimeError(
                "bad inner context"
            )

        if set(
            c["sample_original_fold"]
            .astype(int)
            .unique()
        ) != {inner_fold}:
            raise RuntimeError(
                "nested validation fold mismatch"
            )

        if np.any(
            c[
                "sample_original_fold"
            ].astype(int)
            == outer_fold
        ):
            raise RuntimeError(
                "OUTER LEAKAGE detected"
            )

        # ---------------------------------------------
        # Stage-I reproduction gate:
        # nested committee and nested Stage-II must
        # contain identical Stage-I probabilities.
        # ---------------------------------------------

        m = c[
            [
                "sha256",
                "stage1_prob",
            ]
        ].merge(
            s[
                [
                    "sha256",
                    "stage1_prob",
                ]
            ],
            on="sha256",
            suffixes=(
                "_committee",
                "_stage2",
            ),
            how="inner",
            validate="one_to_one",
        )

        if len(m) != len(c):
            raise RuntimeError(
                "committee/Stage-II SHA mismatch"
            )

        max_stage1_diff = float(
            np.max(
                np.abs(
                    m[
                        "stage1_prob_committee"
                    ].to_numpy(float)
                    -
                    m[
                        "stage1_prob_stage2"
                    ].to_numpy(float)
                )
            )
        )

        if max_stage1_diff > 1e-10:
            raise RuntimeError(
                "nested Stage-I reproduction failed: "
                + str(max_stage1_diff)
            )

        # ---------------------------------------------
        # EXACT DI25 routing within this strict
        # inner-validation fold.
        # ---------------------------------------------

        u = c[
            "U_DI"
        ].to_numpy(
            dtype=np.float64
        )

        route = np.asarray(
            exact_top_fraction(
                u,
                0.25,
            ),
            dtype=bool,
        )

        if route.shape != (
            len(c),
        ):
            raise RuntimeError(
                "route shape mismatch"
            )

        c["strict_route_DI25"] = (
            route.astype(int)
        )

        c["strict_stage1_FN"] = (
            (
                c["label"].astype(int)
                == 1
            )
            &
            (
                c["stage1_prob"].astype(float)
                < 0.5
            )
        ).astype(int)

        c["strict_stage1_FP"] = (
            (
                c["label"].astype(int)
                == 0
            )
            &
            (
                c["stage1_prob"].astype(float)
                >= 0.5
            )
        ).astype(int)

        c["strict_routed_FN"] = (
            (
                c["strict_route_DI25"]
                == 1
            )
            &
            (
                c["strict_stage1_FN"]
                == 1
            )
        ).astype(int)

        c["strict_routed_FP"] = (
            (
                c["strict_route_DI25"]
                == 1
            )
            &
            (
                c["strict_stage1_FP"]
                == 1
            )
        ).astype(int)

        route_n = int(
            c[
                "strict_route_DI25"
            ].sum()
        )

        fn_n = int(
            c[
                "strict_routed_FN"
            ].sum()
        )

        fp_n = int(
            c[
                "strict_routed_FP"
            ].sum()
        )

        print(
            f"inner={inner_fold}"
            f" n={len(c)}"
            f" routed={route_n}"
            f" routed_FN={fn_n}"
            f" routed_FP={fp_n}"
            f" max_stage1_diff={max_stage1_diff:.3g}"
        )

        inner_rows.append({
            "outer_fold":
                outer_fold,

            "inner_fold":
                inner_fold,

            "n":
                len(c),

            "routed":
                route_n,

            "routed_FN":
                fn_n,

            "routed_FP":
                fp_n,

            "max_stage1_diff":
                max_stage1_diff,
        })

        context_parts.append(
            c
        )

    # -----------------------------------------------------
    # Combine four inner-validation folds:
    # this is the leakage-safe OOF view of OUTER TRAIN.
    # -----------------------------------------------------

    train_oof = pd.concat(
        context_parts,
        axis=0,
        ignore_index=True,
    )

    if train_oof[
        "sha256"
    ].duplicated().any():
        raise RuntimeError(
            f"duplicate SHA in outer={outer_fold}"
        )

    expected_n = (
        640
        if outer_fold == 5
        else 639
    )

    if len(train_oof) != expected_n:
        raise RuntimeError(
            f"outer={outer_fold} "
            f"strict train OOF n={len(train_oof)} "
            f"expected={expected_n}"
        )

    if np.any(
        train_oof[
            "sample_original_fold"
        ].astype(int)
        == outer_fold
    ):
        raise RuntimeError(
            "outer fold entered strict reference"
        )

    ref = train_oof[
        train_oof[
            "strict_routed_FN"
        ]
        == 1
    ].copy()

    routed_n = int(
        train_oof[
            "strict_route_DI25"
        ].sum()
    )

    fn_n = len(ref)

    fp_n = int(
        train_oof[
            "strict_routed_FP"
        ].sum()
    )

    print(
        "-- OUTER TRAIN SUMMARY --"
    )

    print(
        "outer_train_n =",
        len(train_oof),
    )

    print(
        "strict routed =",
        routed_n,
    )

    print(
        "strict routed FN reference =",
        fn_n,
    )

    print(
        "strict routed FP =",
        fp_n,
    )

    train_oof.to_csv(
        OUT
        / (
            f"outer_{outer_fold}"
            "_strict_train_oof.csv"
        ),
        index=False,
    )

    ref.to_csv(
        OUT
        / (
            f"outer_{outer_fold}"
            "_routed_fn_reference.csv"
        ),
        index=False,
    )

    summary_rows.append({
        "outer_fold":
            outer_fold,

        "outer_train_n":
            len(train_oof),

        "strict_routed_n":
            routed_n,

        "strict_routed_FN_reference_n":
            fn_n,

        "strict_routed_FP_n":
            fp_n,

        "reference_contains_outer_fold":
            int(
                np.any(
                    train_oof[
                        "sample_original_fold"
                    ].astype(int)
                    == outer_fold
                )
            ),
    })


summary = pd.DataFrame(
    summary_rows
)

inner = pd.DataFrame(
    inner_rows
)

summary.to_csv(
    OUT
    / "outer_reference_summary.csv",
    index=False,
)

inner.to_csv(
    OUT
    / "inner_route_summary.csv",
    index=False,
)


print()
print(
    "=" * 90
)
print(
    "FINAL STRICT REFERENCE SUMMARY"
)
print(
    "=" * 90
)

print(
    summary.to_string(
        index=False
    )
)

print()
print(
    "INNER ROUTE SUMMARY"
)

print(
    inner.to_string(
        index=False
    )
)

print()
print(
    "STRICT_NESTED_REFERENCE_PASS"
)
