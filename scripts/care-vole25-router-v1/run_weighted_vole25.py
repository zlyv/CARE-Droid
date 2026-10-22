#!/usr/bin/env python3
"""
CARE-Droid Weighted-VoLE25
==========================

Leakage-safe Value-of-Local-Evidence router.

Training:
    For each outer fold k, use ONLY nested OOF rows whose
    context_outer_fold == k.

Deployment/evaluation:
    Score the held-out outer fold using cheap Stage-I/committee evidence.
    Route EXACTLY the same number of APKs as frozen DI25 in that fold.

Utility:
    BENEFICIAL : Stage-I wrong, candidate correct  -> +1
    NEUTRAL    : same hard correctness             ->  0
    HARMFUL    : Stage-I correct, candidate wrong  -> -2

Ranking:
    contrast-balanced utility-gap weighted pairwise logistic ranking.

No future year is loaded.
No Host/Stage-II feature is visible to the router.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler


ROOT = Path("artifacts/care-vole25")

NESTED_META = ROOT / "nested_meta.csv"

ROUTER_INPUT = Path(
    "artifacts/care-source-di25-reconstructed-v1/"
    "source_router_inputs.csv"
)

FULL_PRED = Path(
    "artifacts/care-di25-host-input-replay-v1/"
    "arms/FULL/source_predictions.csv"
)

OUT = ROOT / "weighted_vole25"

EXPECTED_N = 799
EXPECTED_ROUTED = 200
EXPECTED_FOLDS = (1, 2, 3, 4, 5)

HARM_COST = 2.0
C_VALUE = 1.0

EXPERT_COL = (
    "DI25__Host__Anomaly__FULL__Bounded__expert_prob"
)

FINAL_COL = (
    "DI25__Host__Anomaly__FULL__Bounded__prob"
)

FEATURES = [
    "stage1_logit",
    "stage1_margin",

    "p_lr",
    "p_rf",
    "p_xgb",
    "p_svm",

    "p_bar",

    "committee_std",
    "committee_range",
    "committee_vote_disagreement",

    "U_DI",
]


def norm_sha(x):
    return str(x).strip().lower()


def utility_from_predictions(y, stage1, candidate):
    y = np.asarray(y, dtype=int)
    stage1 = np.asarray(stage1, dtype=int)
    candidate = np.asarray(candidate, dtype=int)

    beneficial = (
        (stage1 != y)
        &
        (candidate == y)
    )

    harmful = (
        (stage1 == y)
        &
        (candidate != y)
    )

    utility = np.zeros(
        len(y),
        dtype=np.float64,
    )

    utility[beneficial] = 1.0
    utility[harmful] = -HARM_COST

    return utility


def utility_name(u):
    u = float(u)

    if u > 0:
        return "BENEFICIAL"

    if u < 0:
        return "HARMFUL"

    return "NEUTRAL"


class WeightedPairwiseRanker:

    def __init__(self, C=1.0):
        self.C = float(C)

    def fit(self, X, utility):

        X = np.asarray(
            X,
            dtype=np.float64,
        )

        utility = np.asarray(
            utility,
            dtype=np.float64,
        )

        self.scaler = StandardScaler()

        Z = self.scaler.fit_transform(X)

        levels = sorted(
            np.unique(utility).tolist(),
            reverse=True,
        )

        if len(levels) < 2:
            raise RuntimeError(
                "Need >=2 utility levels."
            )

        diff_blocks = []
        label_blocks = []
        weight_blocks = []

        audit = []

        # --------------------------------------------------
        # For every ordered utility contrast:
        #
        # high utility should rank above low utility.
        #
        # Weight of each contrast is proportional to
        # utility gap, but normalized by number of pairs.
        #
        # Therefore:
        #
        # B > N contributes total weight 1
        # N > H contributes total weight 2
        # B > H contributes total weight 3
        #
        # rather than allowing the largest cohort to
        # dominate merely because it creates more pairs.
        # --------------------------------------------------

        for hi in levels:
            for lo in levels:

                if hi <= lo:
                    continue

                hi_idx = np.flatnonzero(
                    utility == hi
                )

                lo_idx = np.flatnonzero(
                    utility == lo
                )

                if (
                    len(hi_idx) == 0
                    or len(lo_idx) == 0
                ):
                    continue

                A = Z[hi_idx]
                B = Z[lo_idx]

                # All pair differences:
                #
                # score(A) > score(B)
                #
                D = (
                    A[:, None, :]
                    -
                    B[None, :, :]
                ).reshape(
                    -1,
                    Z.shape[1],
                )

                n_pairs = len(D)

                gap = float(
                    hi - lo
                )

                # Contrast-balanced weighting.
                per_pair_weight = (
                    gap / n_pairs
                )

                # Positive ordering.
                diff_blocks.append(D)

                label_blocks.append(
                    np.ones(
                        n_pairs,
                        dtype=np.int64,
                    )
                )

                weight_blocks.append(
                    np.full(
                        n_pairs,
                        per_pair_weight,
                        dtype=np.float64,
                    )
                )

                # Reverse pair provides symmetric
                # negative examples.
                diff_blocks.append(-D)

                label_blocks.append(
                    np.zeros(
                        n_pairs,
                        dtype=np.int64,
                    )
                )

                weight_blocks.append(
                    np.full(
                        n_pairs,
                        per_pair_weight,
                        dtype=np.float64,
                    )
                )

                audit.append({
                    "high_utility": hi,
                    "low_utility": lo,
                    "utility_gap": gap,
                    "high_n": len(hi_idx),
                    "low_n": len(lo_idx),
                    "pair_count": n_pairs,
                    "weight_per_pair":
                        per_pair_weight,
                })

        if not diff_blocks:
            raise RuntimeError(
                "No pairwise contrasts."
            )

        D = np.vstack(diff_blocks)

        y_pair = np.concatenate(
            label_blocks
        )

        sample_weight = np.concatenate(
            weight_blocks
        )

        # Normalize only numerical scale;
        # relative contrast weights remain unchanged.
        sample_weight = (
            sample_weight
            /
            np.mean(sample_weight)
        )

        self.model = LogisticRegression(
            C=self.C,
            penalty="l2",
            solver="liblinear",
            fit_intercept=False,
            max_iter=5000,
            random_state=20260922,
        )

        self.model.fit(
            D,
            y_pair,
            sample_weight=sample_weight,
        )

        self.audit = audit
        self.training_pairs = int(
            len(D) // 2
        )

        return self

    def score(self, X):

        X = np.asarray(
            X,
            dtype=np.float64,
        )

        Z = self.scaler.transform(X)

        return (
            Z
            @
            self.model.coef_.ravel()
        )


def top_k_route(
    score,
    sha,
    k,
):
    score = np.asarray(
        score,
        dtype=np.float64,
    )

    sha = np.asarray(
        sha,
        dtype=str,
    )

    if not (
        0 <= k <= len(score)
    ):
        raise RuntimeError(
            "invalid route k"
        )

    # Deterministic:
    # primary = descending VoLE score
    # tie break = ascending SHA
    order = np.lexsort(
        (
            sha,
            -score,
        )
    )

    routed = np.zeros(
        len(score),
        dtype=bool,
    )

    routed[
        order[:k]
    ] = True

    return routed


def metrics(
    y,
    p_stage1,
    p_candidate,
    routed,
):

    y = np.asarray(
        y,
        dtype=int,
    )

    p_stage1 = np.asarray(
        p_stage1,
        dtype=float,
    )

    p_candidate = np.asarray(
        p_candidate,
        dtype=float,
    )

    routed = np.asarray(
        routed,
        dtype=bool,
    )

    p_final = np.where(
        routed,
        p_candidate,
        p_stage1,
    )

    pred_stage1 = (
        p_stage1 >= 0.5
    ).astype(int)

    pred_candidate = (
        p_candidate >= 0.5
    ).astype(int)

    pred_final = (
        p_final >= 0.5
    ).astype(int)

    stage1_correct = (
        pred_stage1 == y
    )

    final_correct = (
        pred_final == y
    )

    fixed = int(
        np.sum(
            (~stage1_correct)
            &
            final_correct
        )
    )

    broken = int(
        np.sum(
            stage1_correct
            &
            (~final_correct)
        )
    )

    candidate_utility = (
        utility_from_predictions(
            y,
            pred_stage1,
            pred_candidate,
        )
    )

    beneficial = (
        candidate_utility > 0
    )

    harmful = (
        candidate_utility < 0
    )

    benefit_total = int(
        beneficial.sum()
    )

    harmful_total = int(
        harmful.sum()
    )

    benefit_routed = int(
        np.sum(
            beneficial
            &
            routed
        )
    )

    harmful_routed = int(
        np.sum(
            harmful
            &
            routed
        )
    )

    return {
        "n":
            int(len(y)),

        "routed":
            int(routed.sum()),

        "route_rate":
            float(routed.mean()),

        "BA":
            float(
                balanced_accuracy_score(
                    y,
                    pred_final,
                )
            ),

        "AUC":
            float(
                roc_auc_score(
                    y,
                    p_final,
                )
            ),

        "F1":
            float(
                f1_score(
                    y,
                    pred_final,
                    zero_division=0,
                )
            ),

        "fixed":
            fixed,

        "broken":
            broken,

        "net":
            fixed - broken,

        "beneficial_total":
            benefit_total,

        "beneficial_routed":
            benefit_routed,

        "benefit_capture":
            (
                benefit_routed
                /
                benefit_total
                if benefit_total
                else np.nan
            ),

        "harmful_total":
            harmful_total,

        "harmful_routed":
            harmful_routed,

        "harmful_capture":
            (
                harmful_routed
                /
                harmful_total
                if harmful_total
                else np.nan
            ),

        "utility_sum":
            float(
                np.sum(
                    candidate_utility[
                        routed
                    ]
                )
            ),
    }


def main():

    OUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ======================================================
    # Nested training data
    # ======================================================

    nested = pd.read_csv(
        NESTED_META
    )

    nested["sha256"] = (
        nested["sha256"]
        .map(norm_sha)
    )

    if len(nested) != 3196:
        raise RuntimeError(
            f"nested rows={len(nested)}"
        )

    if not np.isfinite(
        nested[FEATURES]
        .to_numpy(float)
    ).all():
        raise RuntimeError(
            "non-finite nested features"
        )

    # Recompute utility from primitive fields.
    nested_utility = (
        utility_from_predictions(
            nested["label"],
            nested["stage1_pred"],
            nested["candidate_pred"],
        )
    )

    # Check the stored utility.
    if not np.allclose(
        nested["utility"],
        nested_utility,
    ):
        raise RuntimeError(
            "nested utility mismatch"
        )

    nested["_utility"] = (
        nested_utility
    )

    # ======================================================
    # Outer deploy-time cheap evidence
    # ======================================================

    router = pd.read_csv(
        ROUTER_INPUT
    )

    full = pd.read_csv(
        FULL_PRED
    )

    for d in (
        router,
        full,
    ):
        d["sha256"] = (
            d["sha256"]
            .map(norm_sha)
        )

    if (
        len(router) != EXPECTED_N
        or
        len(full) != EXPECTED_N
    ):
        raise RuntimeError(
            "source N changed"
        )

    source = router.merge(
        full[
            [
                "sha256",
                "Global24__prob",
                EXPERT_COL,
                FINAL_COL,
                "A25__routed",
            ]
        ],
        on="sha256",
        validate="one_to_one",
    )

    if len(source) != EXPECTED_N:
        raise RuntimeError(
            "source merge failed"
        )

    if set(
        source["fold"].astype(int)
    ) != set(EXPECTED_FOLDS):
        raise RuntimeError(
            "outer folds changed"
        )

    # ------------------------------------------------------
    # Hard identity checks established earlier.
    # ------------------------------------------------------

    if not np.allclose(
        source["p_stage1"],
        source["Global24__prob"],
        atol=0,
        rtol=0,
    ):
        raise RuntimeError(
            "Stage-I identity failed"
        )

    if not np.array_equal(
        source["route_DI"].astype(int),
        source["A25__routed"].astype(int),
    ):
        raise RuntimeError(
            "legacy DI route identity failed"
        )

    if int(
        source["route_DI"].sum()
    ) != EXPECTED_ROUTED:
        raise RuntimeError(
            "DI25 route count changed"
        )

    # ------------------------------------------------------
    # Reconstruct the same 11 cheap features used in nested.
    # ------------------------------------------------------

    eps = 1e-7

    p1 = np.clip(
        source["p_stage1"]
        .to_numpy(float),
        eps,
        1.0 - eps,
    )

    source[
        "stage1_logit"
    ] = np.log(
        p1 / (1.0 - p1)
    )

    source[
        "stage1_margin"
    ] = np.abs(
        p1 - 0.5
    )

    member = source[
        [
            "p_lr",
            "p_rf",
            "p_xgb",
            "p_svm",
        ]
    ].to_numpy(float)

    source[
        "committee_std"
    ] = np.std(
        member,
        axis=1,
    )

    source[
        "committee_range"
    ] = (
        np.max(member, axis=1)
        -
        np.min(member, axis=1)
    )

    vote_frac = np.mean(
        member >= 0.5,
        axis=1,
    )

    source[
        "committee_vote_disagreement"
    ] = (
        1.0
        -
        np.abs(
            vote_frac - 0.5
        )
        * 2.0
    )

    if not np.isfinite(
        source[FEATURES]
        .to_numpy(float)
    ).all():
        raise RuntimeError(
            "non-finite outer features"
        )

    # ------------------------------------------------------
    # Reproduce frozen DI25 final probability exactly.
    # ------------------------------------------------------

    di_route = (
        source["route_DI"]
        .astype(int)
        .to_numpy()
        == 1
    )

    reconstructed_di_final = (
        np.where(
            di_route,
            source[EXPERT_COL]
            .to_numpy(float),
            source["p_stage1"]
            .to_numpy(float),
        )
    )

    final_delta = np.max(
        np.abs(
            reconstructed_di_final
            -
            source[FINAL_COL]
            .to_numpy(float)
        )
    )

    print(
        "max |reconstructed DI final "
        "- stored final| =",
        final_delta,
    )

    if final_delta > 1e-12:
        raise RuntimeError(
            "DI final identity failed"
        )

    # ======================================================
    # Outer-by-outer VoLE
    # ======================================================

    prediction_rows = []
    training_audit_rows = []
    coefficient_rows = []

    for outer_fold in EXPECTED_FOLDS:

        print()
        print("=" * 72)
        print(
            "OUTER FOLD",
            outer_fold,
        )
        print("=" * 72)

        train = nested[
            nested[
                "context_outer_fold"
            ].astype(int)
            ==
            outer_fold
        ].copy()

        test = source[
            source["fold"]
            .astype(int)
            ==
            outer_fold
        ].copy()

        # Nested context must not contain
        # the current outer fold.
        if np.any(
            train[
                "sample_original_fold"
            ].astype(int)
            ==
            outer_fold
        ):
            raise RuntimeError(
                "OUTER LEAKAGE"
            )

        counts = (
            train["_utility"]
            .value_counts()
            .to_dict()
        )

        n_b = int(
            counts.get(1.0, 0)
        )

        n_n = int(
            counts.get(0.0, 0)
        )

        n_h = int(
            counts.get(
                -HARM_COST,
                0,
            )
        )

        print(
            "nested utility:",
            "B=", n_b,
            "N=", n_n,
            "H=", n_h,
        )

        if min(
            n_b,
            n_n,
            n_h,
        ) <= 0:
            raise RuntimeError(
                "missing utility class"
            )

        ranker = (
            WeightedPairwiseRanker(
                C=C_VALUE
            )
            .fit(
                train[
                    FEATURES
                ].to_numpy(float),
                train[
                    "_utility"
                ].to_numpy(float),
            )
        )

        score = ranker.score(
            test[
                FEATURES
            ].to_numpy(float)
        )

        # EXACT same budget as DI in this fold.
        k = int(
            test[
                "route_DI"
            ].astype(int)
            .sum()
        )

        vole_route = top_k_route(
            score,
            test["sha256"],
            k,
        )

        print(
            "DI budget =",
            k,
            "VoLE routed =",
            int(
                vole_route.sum()
            ),
        )

        if int(
            vole_route.sum()
        ) != k:
            raise RuntimeError(
                "budget mismatch"
            )

        # Save training audits.
        for row in ranker.audit:
            training_audit_rows.append({
                "outer_fold":
                    outer_fold,

                "nested_n":
                    len(train),

                "beneficial_n":
                    n_b,

                "neutral_n":
                    n_n,

                "harmful_n":
                    n_h,

                **row,
            })

        coef = (
            ranker.model
            .coef_
            .ravel()
        )

        for feature, value in zip(
            FEATURES,
            coef,
        ):
            coefficient_rows.append({
                "outer_fold":
                    outer_fold,

                "feature":
                    feature,

                "coefficient":
                    float(value),
            })

        # Candidate utility on held-out outer fold
        # is evaluation-only and is NEVER used
        # to fit this fold's router.
        y = test[
            "label"
        ].to_numpy(int)

        s1_prob = test[
            "p_stage1"
        ].to_numpy(float)

        candidate_prob = test[
            EXPERT_COL
        ].to_numpy(float)

        s1_pred = (
            s1_prob >= 0.5
        ).astype(int)

        candidate_pred = (
            candidate_prob >= 0.5
        ).astype(int)

        outer_utility = (
            utility_from_predictions(
                y,
                s1_pred,
                candidate_pred,
            )
        )

        di_fold_route = (
            test["route_DI"]
            .astype(int)
            .to_numpy()
            == 1
        )

        for local, (
            idx,
            row,
        ) in enumerate(
            test.iterrows()
        ):

            prediction_rows.append({
                "sha256":
                    row["sha256"],

                "outer_fold":
                    outer_fold,

                "label":
                    int(row["label"]),

                "stage1_prob":
                    float(
                        row["p_stage1"]
                    ),

                "candidate_prob":
                    float(
                        row[EXPERT_COL]
                    ),

                "counterfactual_utility_eval_only":
                    float(
                        outer_utility[
                            local
                        ]
                    ),

                "counterfactual_class_eval_only":
                    utility_name(
                        outer_utility[
                            local
                        ]
                    ),

                "DI_score":
                    float(
                        row["U_DI"]
                    ),

                "DI25_routed":
                    int(
                        di_fold_route[
                            local
                        ]
                    ),

                "VoLE_score":
                    float(
                        score[
                            local
                        ]
                    ),

                "VoLE25_routed":
                    int(
                        vole_route[
                            local
                        ]
                    ),
            })

    # ======================================================
    # Pooled source evaluation
    # ======================================================

    pred = pd.DataFrame(
        prediction_rows
    )

    if (
        len(pred) != EXPECTED_N
        or
        pred["sha256"].nunique()
        != EXPECTED_N
    ):
        raise RuntimeError(
            "outer prediction coverage failed"
        )

    # Restore source order.
    pred = source[
        ["sha256"]
    ].merge(
        pred,
        on="sha256",
        validate="one_to_one",
    )

    source_by_sha = (
        source
        .set_index("sha256")
    )

    p_stage1 = np.asarray([
        float(
            source_by_sha.loc[
                sha,
                "p_stage1",
            ]
        )
        for sha in pred["sha256"]
    ])

    p_candidate = np.asarray([
        float(
            source_by_sha.loc[
                sha,
                EXPERT_COL,
            ]
        )
        for sha in pred["sha256"]
    ])

    y = pred[
        "label"
    ].to_numpy(int)

    route_di = (
        pred[
            "DI25_routed"
        ].to_numpy(int)
        == 1
    )

    route_vole = (
        pred[
            "VoLE25_routed"
        ].to_numpy(int)
        == 1
    )

    result_di = metrics(
        y,
        p_stage1,
        p_candidate,
        route_di,
    )

    result_vole = metrics(
        y,
        p_stage1,
        p_candidate,
        route_vole,
    )

    # ======================================================
    # Baseline reproduction gate
    # ======================================================

    print()
    print("=" * 72)
    print("DI25 REPRODUCTION")
    print("=" * 72)

    print(
        json.dumps(
            result_di,
            indent=2,
        )
    )

    if result_di["routed"] != 200:
        raise RuntimeError(
            "BLOCKED_BY_BASELINE_REPRODUCTION: "
            "routed"
        )

    if (
        result_di["fixed"] != 96
        or
        result_di["broken"] != 13
        or
        result_di["net"] != 83
    ):
        raise RuntimeError(
            "BLOCKED_BY_CURRENT_BASELINE: "
            "fixed/broken/net"
        )

    if abs(
        result_di["BA"]
        -
        0.923640350877193
    ) > 1e-8:
        raise RuntimeError(
            "BLOCKED_BY_CURRENT_BASELINE: BA"
        )

    # ======================================================
    # VoLE result
    # ======================================================

    print()
    print("=" * 72)
    print("WEIGHTED VoLE25")
    print("=" * 72)

    print(
        json.dumps(
            result_vole,
            indent=2,
        )
    )

    summary = {
        "experiment":
            "CARE-Droid Weighted-VoLE25 v1",

        "source_only":
            True,

        "future_data_loaded":
            False,

        "router_features":
            FEATURES,

        "router_feature_dim":
            len(FEATURES),

        "harm_cost":
            HARM_COST,

        "pairwise_weighting":
            (
                "contrast-balanced "
                "utility-gap weighting"
            ),

        "same_budget_as_DI25":
            True,

        "DI25":
            result_di,

        "Weighted_VoLE25":
            result_vole,

        "delta": {
            "BA":
                result_vole["BA"]
                -
                result_di["BA"],

            "AUC":
                result_vole["AUC"]
                -
                result_di["AUC"],

            "F1":
                result_vole["F1"]
                -
                result_di["F1"],

            "fixed":
                result_vole["fixed"]
                -
                result_di["fixed"],

            "broken":
                result_vole["broken"]
                -
                result_di["broken"],

            "net":
                result_vole["net"]
                -
                result_di["net"],

            "benefit_capture":
                result_vole[
                    "benefit_capture"
                ]
                -
                result_di[
                    "benefit_capture"
                ],

            "harmful_capture":
                result_vole[
                    "harmful_capture"
                ]
                -
                result_di[
                    "harmful_capture"
                ],
        },
    }

    pred.to_csv(
        OUT /
        "source_predictions.csv",
        index=False,
    )

    pd.DataFrame(
        training_audit_rows
    ).to_csv(
        OUT /
        "training_pair_audit.csv",
        index=False,
    )

    pd.DataFrame(
        coefficient_rows
    ).to_csv(
        OUT /
        "router_coefficients.csv",
        index=False,
    )

    (
        OUT /
        "summary.json"
    ).write_text(
        json.dumps(
            summary,
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    print()
    print("=" * 72)
    print("DELTA: VoLE25 - DI25")
    print("=" * 72)

    print(
        json.dumps(
            summary["delta"],
            indent=2,
        )
    )

    print()
    print(
        "SAVED:",
        OUT,
    )


if __name__ == "__main__":
    main()
