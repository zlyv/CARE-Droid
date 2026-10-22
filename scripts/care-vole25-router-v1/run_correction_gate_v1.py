#!/usr/bin/env python3

from pathlib import Path
import json

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    balanced_accuracy_score,
    roc_auc_score,
    f1_score,
)


ROOT = Path("artifacts/care-vole25")

S2_DIR = ROOT / "nested_stage2/contexts"
CM_DIR = ROOT / "nested_committee/contexts"

ROUTER_INPUT = Path(
    "artifacts/care-source-di25-reconstructed-v1/"
    "source_router_inputs.csv"
)

FULL_INPUT = Path(
    "artifacts/care-di25-host-input-replay-v1/"
    "arms/FULL/source_predictions.csv"
)

OUT = ROOT / "correction_gate_v1"

PREFIX = (
    "DI25__Host__Anomaly__FULL__Bounded__"
)

EXPERT_COL = PREFIX + "expert_prob"
FINAL_COL = PREFIX + "prob"

HARM_COST = 2.0

# Derived directly from utility:
#
# p(B)*1 + (1-p(B))*(-2) > 0
#
# -> p(B) > 2/3
ACCEPT_THRESHOLD = (
    HARM_COST /
    (1.0 + HARM_COST)
)

C_VALUE = 1.0

STRUCTURAL = [
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

FEATURES = [
    "stage1_margin",
    "candidate_margin",
    "prob_shift",
    "abs_prob_shift",
    "bounded_delta",
    "abs_bounded_delta",
] + STRUCTURAL


def norm_sha(x):
    return str(x).strip().lower()


def top_k(score, sha, k):

    score = np.asarray(
        score,
        dtype=float,
    )

    sha = np.asarray(
        sha,
        dtype=str,
    )

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


def add_nested_features(d):

    d = d.copy()

    p1 = (
        d["stage1_prob_stage2"]
        .to_numpy(float)
    )

    pc = (
        d["candidate_prob"]
        .to_numpy(float)
    )

    delta = (
        d["candidate_bounded_delta"]
        .to_numpy(float)
    )

    d["stage1_margin"] = np.abs(
        p1 - 0.5
    )

    d["candidate_margin"] = np.abs(
        pc - 0.5
    )

    d["prob_shift"] = (
        pc - p1
    )

    d["abs_prob_shift"] = np.abs(
        pc - p1
    )

    d["bounded_delta"] = delta

    d["abs_bounded_delta"] = np.abs(
        delta
    )

    for name in STRUCTURAL:

        source = (
            "candidate_" + name
        )

        if source not in d.columns:
            raise RuntimeError(
                f"missing nested feature: {source}"
            )

        d[name] = (
            d[source]
            .to_numpy(float)
        )

    return d


def add_outer_features(d):

    d = d.copy()

    p1 = (
        d["p_stage1"]
        .to_numpy(float)
    )

    pc = (
        d[EXPERT_COL]
        .to_numpy(float)
    )

    delta = (
        d[PREFIX + "bounded_delta"]
        .to_numpy(float)
    )

    d["stage1_margin"] = np.abs(
        p1 - 0.5
    )

    d["candidate_margin"] = np.abs(
        pc - 0.5
    )

    d["prob_shift"] = (
        pc - p1
    )

    d["abs_prob_shift"] = np.abs(
        pc - p1
    )

    d["bounded_delta"] = delta

    d["abs_bounded_delta"] = np.abs(
        delta
    )

    for name in STRUCTURAL:

        source = PREFIX + name

        if source not in d.columns:
            raise RuntimeError(
                f"missing outer feature: {source}"
            )

        d[name] = (
            d[source]
            .to_numpy(float)
        )

    return d


def metric_summary(
    y,
    p_stage1,
    p_final,
):

    y = np.asarray(
        y,
        dtype=int,
    )

    p_stage1 = np.asarray(
        p_stage1,
        dtype=float,
    )

    p_final = np.asarray(
        p_final,
        dtype=float,
    )

    pred0 = (
        p_stage1 >= 0.5
    ).astype(int)

    pred = (
        p_final >= 0.5
    ).astype(int)

    ok0 = (
        pred0 == y
    )

    ok = (
        pred == y
    )

    fixed = int(
        np.sum(
            (~ok0)
            &
            ok
        )
    )

    broken = int(
        np.sum(
            ok0
            &
            (~ok)
        )
    )

    return {
        "BA":
            float(
                balanced_accuracy_score(
                    y,
                    pred,
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
                    pred,
                    zero_division=0,
                )
            ),

        "fixed":
            fixed,

        "broken":
            broken,

        "net":
            fixed - broken,
    }


# ============================================================
# Load nested data
# ============================================================

s2_files = sorted(
    S2_DIR.glob(
        "outer_*__inner_*.csv"
    )
)

cm_files = sorted(
    CM_DIR.glob(
        "outer_*__inner_*.csv"
    )
)

assert len(s2_files) == 20
assert len(cm_files) == 20

s2 = pd.concat(
    [pd.read_csv(p) for p in s2_files],
    ignore_index=True,
)

cm = pd.concat(
    [pd.read_csv(p) for p in cm_files],
    ignore_index=True,
)

for d in (s2, cm):
    d["sha256"] = (
        d["sha256"]
        .map(norm_sha)
    )

keys = [
    "context_outer_fold",
    "inner_val_fold",
    "sha256",
]

nested = cm.merge(
    s2,
    on=keys,
    validate="one_to_one",
    suffixes=(
        "_committee",
        "_stage2",
    ),
)

assert len(nested) == 3196

stage1_delta = np.max(
    np.abs(
        nested[
            "stage1_prob_committee"
        ].to_numpy(float)
        -
        nested[
            "stage1_prob_stage2"
        ].to_numpy(float)
    )
)

assert stage1_delta < 1e-10

nested = add_nested_features(
    nested
)

assert np.isfinite(
    nested[FEATURES]
    .to_numpy(float)
).all()


# ============================================================
# Load current reproducible source baseline
# ============================================================

router = pd.read_csv(
    ROUTER_INPUT
)

full = pd.read_csv(
    FULL_INPUT
)

for d in (router, full):
    d["sha256"] = (
        d["sha256"]
        .map(norm_sha)
    )

full_keep = [
    "sha256",
    "Global24__prob",
    "A25__routed",
    EXPERT_COL,
    FINAL_COL,
    PREFIX + "bounded_delta",
]

full_keep += [
    PREFIX + name
    for name in STRUCTURAL
]

source = router.merge(
    full[full_keep],
    on="sha256",
    validate="one_to_one",
)

assert len(source) == 799
assert source["sha256"].nunique() == 799

# Stage-I identity
assert np.allclose(
    source["p_stage1"],
    source["Global24__prob"],
    atol=0,
    rtol=0,
)

# DI route identity
assert np.array_equal(
    source["route_DI"]
    .astype(int),
    source["A25__routed"]
    .astype(int),
)

assert int(
    source["route_DI"]
    .astype(int)
    .sum()
) == 200

source = add_outer_features(
    source
)

assert np.isfinite(
    source[FEATURES]
    .to_numpy(float)
).all()


# ============================================================
# Baseline identity
# ============================================================

route_di = (
    source["route_DI"]
    .astype(int)
    .to_numpy()
    == 1
)

p1_all = (
    source["p_stage1"]
    .to_numpy(float)
)

pc_all = (
    source[EXPERT_COL]
    .to_numpy(float)
)

baseline_reconstructed = np.where(
    route_di,
    pc_all,
    p1_all,
)

baseline_stored = (
    source[FINAL_COL]
    .to_numpy(float)
)

baseline_delta = np.max(
    np.abs(
        baseline_reconstructed
        -
        baseline_stored
    )
)

print(
    "max baseline reconstruction delta =",
    baseline_delta,
)

assert baseline_delta < 1e-12


# ============================================================
# DI budget per original fold
# ============================================================

budget_by_fold = {}

for fold in range(1, 6):

    q = source[
        source["fold"]
        .astype(int)
        == fold
    ]

    budget_by_fold[fold] = int(
        q["route_DI"]
        .astype(int)
        .sum()
    )

    assert (
        budget_by_fold[fold]
        == 40
    )


# ============================================================
# Outer OOF Gate
# ============================================================

prediction_rows = []
coef_rows = []
training_rows = []

for outer in range(1, 6):

    print()
    print("=" * 72)
    print(
        "OUTER FOLD",
        outer,
    )
    print("=" * 72)

    # --------------------------------------------------------
    # Build leakage-safe Gate training cohort.
    #
    # For this outer context:
    # use the four nested validation folds only.
    # --------------------------------------------------------

    context = nested[
        nested[
            "context_outer_fold"
        ].astype(int)
        == outer
    ].copy()

    if np.any(
        context[
            "sample_original_fold_stage2"
        ].astype(int)
        == outer
    ):
        raise RuntimeError(
            "OUTER LEAKAGE"
        )

    train_parts = []

    for inner in range(1, 6):

        if inner == outer:
            continue

        q = context[
            context[
                "inner_val_fold"
            ].astype(int)
            == inner
        ].copy()

        route = top_k(
            q["U_DI"],
            q["sha256"],
            budget_by_fold[inner],
        )

        s1 = (
            q["stage1_pred_stage2"]
            .astype(int)
            .to_numpy()
        )

        c2 = (
            q["candidate_pred"]
            .astype(int)
            .to_numpy()
        )

        y = (
            q["label_stage2"]
            .astype(int)
            .to_numpy()
        )

        flip = (
            s1 != c2
        )

        selected = (
            route
            &
            flip
        )

        t = q.loc[
            selected
        ].copy()

        t_s1 = (
            t["stage1_pred_stage2"]
            .astype(int)
            .to_numpy()
        )

        t_c2 = (
            t["candidate_pred"]
            .astype(int)
            .to_numpy()
        )

        t_y = (
            t["label_stage2"]
            .astype(int)
            .to_numpy()
        )

        # Among binary class flips:
        #
        # 1 = beneficial
        # 0 = harmful
        gate_y = (
            t_c2 == t_y
        ).astype(int)

        # Sanity:
        # if candidate is wrong,
        # Stage-I must have been correct.
        harmful = (
            gate_y == 0
        )

        if np.any(
            t_s1[harmful]
            != t_y[harmful]
        ):
            raise RuntimeError(
                "invalid binary flip semantics"
            )

        t["_gate_y"] = gate_y

        train_parts.append(
            t
        )

    train = pd.concat(
        train_parts,
        ignore_index=True,
    )

    if train[
        "sha256"
    ].duplicated().any():
        raise RuntimeError(
            "duplicate Gate training SHA"
        )

    B_train = int(
        (
            train["_gate_y"]
            == 1
        ).sum()
    )

    H_train = int(
        (
            train["_gate_y"]
            == 0
        ).sum()
    )

    print(
        "Gate train n =",
        len(train),
        "B =",
        B_train,
        "H =",
        H_train,
    )

    if (
        B_train == 0
        or H_train == 0
    ):
        raise RuntimeError(
            "Gate training class missing"
        )

    X_train = (
        train[FEATURES]
        .to_numpy(float)
    )

    y_train = (
        train["_gate_y"]
        .to_numpy(int)
    )

    scaler = StandardScaler()

    Z_train = scaler.fit_transform(
        X_train
    )

    gate = LogisticRegression(
        C=C_VALUE,
        penalty="l2",
        solver="liblinear",
        max_iter=5000,
        random_state=20260922,
    )

    # Deliberately no class_weight.
    #
    # Harm cost is expressed through the
    # fixed decision threshold 2/3.
    gate.fit(
        Z_train,
        y_train,
    )

    for feature, value in zip(
        FEATURES,
        gate.coef_.ravel(),
    ):
        coef_rows.append({
            "outer_fold":
                outer,

            "feature":
                feature,

            "coefficient":
                float(value),
        })

    training_rows.append({
        "outer_fold":
            outer,

        "train_flip_n":
            len(train),

        "train_beneficial":
            B_train,

        "train_harmful":
            H_train,

        "accept_threshold":
            ACCEPT_THRESHOLD,
    })

    # --------------------------------------------------------
    # Held-out outer fold
    # --------------------------------------------------------

    test = source[
        source["fold"]
        .astype(int)
        == outer
    ].copy()

    test_route = (
        test["route_DI"]
        .astype(int)
        .to_numpy()
        == 1
    )

    p1 = (
        test["p_stage1"]
        .to_numpy(float)
    )

    pc = (
        test[EXPERT_COL]
        .to_numpy(float)
    )

    s1 = (
        p1 >= 0.5
    ).astype(int)

    c2 = (
        pc >= 0.5
    ).astype(int)

    flip = (
        test_route
        &
        (s1 != c2)
    )

    gate_prob = np.full(
        len(test),
        np.nan,
        dtype=float,
    )

    accept = np.zeros(
        len(test),
        dtype=bool,
    )

    if flip.any():

        X_test = (
            test.loc[
                flip,
                FEATURES
            ]
            .to_numpy(float)
        )

        Z_test = scaler.transform(
            X_test
        )

        prob_B = (
            gate.predict_proba(
                Z_test
            )[:, 1]
        )

        gate_prob[
            np.flatnonzero(flip)
        ] = prob_B

        accept[
            np.flatnonzero(flip)
        ] = (
            prob_B
            >
            ACCEPT_THRESHOLD
        )

    # --------------------------------------------------------
    # Start from exact DI baseline.
    #
    # Only rejected class flips are reverted
    # to Stage-I probability.
    # --------------------------------------------------------

    p_di = np.where(
        test_route,
        pc,
        p1,
    )

    p_gate = p_di.copy()

    reject = (
        flip
        &
        (~accept)
    )

    p_gate[reject] = (
        p1[reject]
    )

    # Outer labels are evaluation-only.
    y = (
        test["label"]
        .astype(int)
        .to_numpy()
    )

    beneficial = (
        flip
        &
        (s1 != y)
        &
        (c2 == y)
    )

    harmful = (
        flip
        &
        (s1 == y)
        &
        (c2 != y)
    )

    print(
        "outer routed =",
        int(test_route.sum()),
        "flip =",
        int(flip.sum()),
    )

    print(
        "outer B =",
        int(beneficial.sum()),
        "H =",
        int(harmful.sum()),
    )

    print(
        "accepted B =",
        int(
            (
                beneficial
                &
                accept
            ).sum()
        ),
        "/",
        int(
            beneficial.sum()
        ),
    )

    print(
        "accepted H =",
        int(
            (
                harmful
                &
                accept
            ).sum()
        ),
        "/",
        int(
            harmful.sum()
        ),
    )

    for local, (_, row) in enumerate(
        test.iterrows()
    ):

        prediction_rows.append({
            "sha256":
                row["sha256"],

            "fold":
                outer,

            "label":
                int(row["label"]),

            "stage1_prob":
                float(p1[local]),

            "candidate_prob":
                float(pc[local]),

            "DI25_routed":
                int(
                    test_route[local]
                ),

            "candidate_flip":
                int(
                    flip[local]
                ),

            "gate_prob_beneficial":
                (
                    float(
                        gate_prob[local]
                    )
                    if np.isfinite(
                        gate_prob[local]
                    )
                    else np.nan
                ),

            "gate_accept":
                int(
                    accept[local]
                ),

            "DI25_final_prob":
                float(
                    p_di[local]
                ),

            "Gate_final_prob":
                float(
                    p_gate[local]
                ),

            "beneficial_eval_only":
                int(
                    beneficial[local]
                ),

            "harmful_eval_only":
                int(
                    harmful[local]
                ),
        })


# ============================================================
# Pooled evaluation
# ============================================================

pred = pd.DataFrame(
    prediction_rows
)

assert len(pred) == 799
assert pred["sha256"].nunique() == 799

y = (
    pred["label"]
    .to_numpy(int)
)

p1 = (
    pred["stage1_prob"]
    .to_numpy(float)
)

p_di = (
    pred["DI25_final_prob"]
    .to_numpy(float)
)

p_gate = (
    pred["Gate_final_prob"]
    .to_numpy(float)
)

baseline = metric_summary(
    y,
    p1,
    p_di,
)

gated = metric_summary(
    y,
    p1,
    p_gate,
)

B = (
    pred[
        "beneficial_eval_only"
    ]
    .astype(bool)
    .to_numpy()
)

H = (
    pred[
        "harmful_eval_only"
    ]
    .astype(bool)
    .to_numpy()
)

accept = (
    pred[
        "gate_accept"
    ]
    .astype(bool)
    .to_numpy()
)

flip = (
    pred[
        "candidate_flip"
    ]
    .astype(bool)
    .to_numpy()
)

gate_stats = {
    "flip_total":
        int(flip.sum()),

    "beneficial_total":
        int(B.sum()),

    "harmful_total":
        int(H.sum()),

    "accepted_total":
        int(
            (flip & accept).sum()
        ),

    "accepted_beneficial":
        int(
            (B & accept).sum()
        ),

    "rejected_beneficial":
        int(
            (B & ~accept).sum()
        ),

    "accepted_harmful":
        int(
            (H & accept).sum()
        ),

    "rejected_harmful":
        int(
            (H & ~accept).sum()
        ),
}

summary = {
    "experiment":
        "CARE-Droid Correction Acceptance Gate v1",

    "router":
        "frozen DI25",

    "router_budget":
        200,

    "gate_scope":
        "DI25-routed class-flip candidates only",

    "gate_model":
        "StandardScaler + L2 LogisticRegression",

    "gate_C":
        C_VALUE,

    "harm_cost":
        HARM_COST,

    "accept_threshold":
        ACCEPT_THRESHOLD,

    "feature_dim":
        len(FEATURES),

    "features":
        FEATURES,

    "DI25_baseline":
        baseline,

    "CorrectionGate":
        gated,

    "gate_stats":
        gate_stats,

    "delta": {
        key:
            gated[key]
            -
            baseline[key]

        for key in (
            "BA",
            "AUC",
            "F1",
            "fixed",
            "broken",
            "net",
        )
    },
}

OUT.mkdir(
    parents=True,
    exist_ok=True,
)

pred.to_csv(
    OUT / "source_predictions.csv",
    index=False,
)

pd.DataFrame(
    training_rows
).to_csv(
    OUT / "training_audit.csv",
    index=False,
)

pd.DataFrame(
    coef_rows
).to_csv(
    OUT / "coefficients.csv",
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
print("DI25 BASELINE")
print("=" * 72)
print(
    json.dumps(
        baseline,
        indent=2,
    )
)

print()
print("=" * 72)
print("CORRECTION ACCEPTANCE GATE")
print("=" * 72)
print(
    json.dumps(
        gated,
        indent=2,
    )
)

print()
print("=" * 72)
print("GATE STATISTICS")
print("=" * 72)
print(
    json.dumps(
        gate_stats,
        indent=2,
    )
)

print()
print("=" * 72)
print("DELTA: GATE - DI25")
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
