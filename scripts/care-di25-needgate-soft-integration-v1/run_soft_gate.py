#!/usr/bin/env python3

from pathlib import Path
import json
import numpy as np
import pandas as pd

from sklearn.metrics import (
    balanced_accuracy_score,
    roc_auc_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
)

ROOT = Path("artifacts/care-di25-needgate-soft-integration-v1")
ROOT.mkdir(parents=True, exist_ok=True)

ROUTER = Path(
    "artifacts/care-source-di25-reconstructed-v1/source_router_inputs.csv"
)

FULL = Path(
    "artifacts/care-di25-host-input-replay-v1/arms/FULL/source_predictions.csv"
)

GATE = Path(
    "artifacts/care-di25-noego-needgate-source-v1/oof_predictions.csv"
)

P_FINAL = "DI25__Host__Anomaly__FULL__Bounded__prob"
DELTA = "DI25__Host__Anomaly__FULL__Bounded__bounded_delta"


def blocked(msg):
    raise RuntimeError("BLOCKED_BY_CONTRACT: " + msg)


def logit(p):
    p = np.clip(np.asarray(p, dtype=float), 1e-8, 1 - 1e-8)
    return np.log(p / (1 - p))


def sigmoid(z):
    z = np.asarray(z, dtype=float)
    return 1.0 / (1.0 + np.exp(-z))


def metric(y, p):
    pred = (p >= 0.5).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y, pred, labels=[0, 1]
    ).ravel()

    return {
        "BA": balanced_accuracy_score(y, pred),
        "AUC": roc_auc_score(y, p),
        "F1": f1_score(y, pred, zero_division=0),
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred, zero_division=0),
        "specificity": tn / (tn + fp),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
    }


def correction_stats(y, p_stage1, p):
    base_pred = (p_stage1 >= 0.5).astype(int)
    pred = (p >= 0.5).astype(int)

    wrong = base_pred != y
    correct = ~wrong

    fixed = int(np.sum(wrong & (pred == y)))
    broken = int(np.sum(correct & (pred != y)))

    return {
        "fixed": fixed,
        "broken": broken,
        "net": fixed - broken,
        "class_flips_vs_stage1": int(np.sum(base_pred != pred)),
        "correction_precision":
            fixed / (fixed + broken)
            if fixed + broken else np.nan,
        "error_repair_recall":
            fixed / int(np.sum(wrong))
            if np.sum(wrong) else np.nan,
    }


router = pd.read_csv(ROUTER)
full = pd.read_csv(FULL)
gate = pd.read_csv(GATE)

# -------------------------
# Hard contracts
# -------------------------

if len(router) != 799:
    blocked("router != 799")

if int(router["route_DI"].sum()) != 200:
    blocked("DI25 != 200")

if len(full) != 799:
    blocked("FULL != 799")

if len(gate) != 200:
    blocked("Gate OOF != 200")

for df in (router, full, gate):
    df["sha"] = df["sha256"].astype(str).str.lower()

if set(router["sha"]) != set(full["sha"]):
    blocked("FULL cohort differs from router")

routed_sha = set(
    router.loc[router.route_DI == 1, "sha"]
)

if routed_sha != set(gate["sha"]):
    blocked("Gate cohort differs from routed DI25 cohort")

for c in ("score_S2", "score_S3"):
    if c not in gate.columns:
        blocked("missing " + c)

    if (
        (gate[c] < 0).any()
        or (gate[c] > 1).any()
        or (~np.isfinite(gate[c])).any()
    ):
        blocked(c + " invalid")

if P_FINAL not in full.columns:
    blocked("missing FULL final probability")

if DELTA not in full.columns:
    blocked("missing bounded delta")


# -------------------------
# Align rows
# -------------------------

full = (
    full.set_index("sha")
    .loc[router["sha"]]
    .reset_index()
)

gate = gate.set_index("sha")

y = router["label"].astype(int).to_numpy()
folds = router["fold"].astype(int).to_numpy()
route = router["route_DI"].astype(int).to_numpy()

p_stage1 = router["p_stage1"].astype(float).to_numpy()

# Verify frozen Stage-I
p_full_stage1 = full["Global24__prob"].astype(float).to_numpy()

if np.max(np.abs(p_stage1 - p_full_stage1)) > 1e-10:
    blocked("Stage-I mismatch")

p_b0 = full[P_FINAL].astype(float).to_numpy()
stored_delta = full[DELTA].astype(float).to_numpy()

# Unrouted samples must remain Stage-I
if np.max(
    np.abs(
        p_b0[route == 0]
        - p_stage1[route == 0]
    )
) > 1e-8:
    blocked("B0 modifies unrouted APK")

routed = route == 1

# IMPORTANT:
# B0 is an ensemble probability obtained after averaging multiple neural seeds.
# Therefore the stored seed-level/aggregated bounded_delta is not required to
# equal logit(mean_probability) - logit(stage1_probability).
#
# For this post-hoc ensemble-level soft-gating diagnostic, define the exact
# effective correction represented by the final frozen B0 probability.
effective_delta = np.zeros_like(p_stage1, dtype=float)

effective_delta[routed] = (
    logit(p_b0[routed])
    - logit(p_stage1[routed])
)

if not np.isfinite(effective_delta).all():
    blocked("effective ensemble logit delta contains non-finite values")

# Each constituent correction is bounded by 2.25. Because the ensemble
# probability lies between the minimum and maximum constituent probability,
# the effective ensemble logit correction must remain within the same bound.
max_effective_delta = float(
    np.max(np.abs(effective_delta[routed]))
)

if max_effective_delta > 2.250001:
    blocked(
        "effective ensemble correction exceeds frozen 2.25 bound: "
        f"{max_effective_delta}"
    )

# g=1 must exactly reconstruct frozen B0.
reconstructed_b0 = sigmoid(
    logit(p_stage1[routed])
    + effective_delta[routed]
)

reconstruction_max_abs_diff = float(
    np.max(
        np.abs(
            reconstructed_b0
            - p_b0[routed]
        )
    )
)

if reconstruction_max_abs_diff > 1e-10:
    blocked(
        "effective delta cannot reconstruct frozen B0: "
        f"maxdiff={reconstruction_max_abs_diff}"
    )

# Keep the mismatch only as an audit diagnostic.
stored_delta_mismatch_max_abs = float(
    np.max(
        np.abs(
            stored_delta[routed]
            - effective_delta[routed]
        )
    )
)

print(
    "effective_delta_max_abs =",
    max_effective_delta
)
print(
    "B0_reconstruction_max_abs_diff =",
    reconstruction_max_abs_diff
)
print(
    "stored_vs_effective_delta_max_abs_diff =",
    stored_delta_mismatch_max_abs
)


# -------------------------
# Gate scores aligned to 799
# -------------------------

g2 = np.zeros(799, dtype=float)
g3 = np.zeros(799, dtype=float)

for i, sha in enumerate(router["sha"]):
    if route[i] == 1:
        g2[i] = float(gate.loc[sha, "score_S2"])
        g3[i] = float(gate.loc[sha, "score_S3"])


# -------------------------
# Soft bounded correction
# -------------------------

z1 = logit(p_stage1)

p_soft_s2 = p_stage1.copy()
p_soft_s3 = p_stage1.copy()

p_soft_s2[routed] = sigmoid(
    z1[routed]
    + g2[routed] * effective_delta[routed]
)

p_soft_s3[routed] = sigmoid(
    z1[routed]
    + g3[routed] * effective_delta[routed]
)


# -------------------------
# Metrics
# -------------------------

arms = {
    "StageI_Global24": p_stage1,
    "B0_FULL": p_b0,
    "SoftGate_S2": p_soft_s2,
    "SoftGate_S3": p_soft_s3,
}

rows = []

for name, p in arms.items():
    rows.append({
        "arm": name,
        "n": 799,
        **metric(y, p),
        **correction_stats(
            y, p_stage1, p
        ),
    })

pd.DataFrame(rows).to_csv(
    ROOT / "end_to_end_metrics.csv",
    index=False,
)


# -------------------------
# Per fold
# -------------------------

fold_rows = []

for fold in range(1, 6):
    idx = folds == fold

    for name, p in arms.items():
        fold_rows.append({
            "fold": fold,
            "arm": name,
            "n": int(np.sum(idx)),
            **metric(y[idx], p[idx]),
        })

pd.DataFrame(fold_rows).to_csv(
    ROOT / "per_fold_metrics.csv",
    index=False,
)


# -------------------------
# Sample audit
# -------------------------

audit = router[
    [
        "sha256",
        "year",
        "label",
        "fold",
        "route_DI",
        "p_stage1",
    ]
].copy()

audit["stored_bounded_delta"] = stored_delta
audit["effective_ensemble_logit_delta"] = effective_delta
audit["stored_vs_effective_delta_absdiff"] = np.abs(
    stored_delta - effective_delta
)
audit["p_B0"] = p_b0

audit["gate_score_S2"] = g2
audit["p_soft_S2"] = p_soft_s2

audit["gate_score_S3"] = g3
audit["p_soft_S3"] = p_soft_s3

audit.to_csv(
    ROOT / "sample_audit.csv",
    index=False,
)


protocol = {
    "experiment":
        "CARE-Droid source-only soft Need-Gate integration v1",

    "year": 2017,
    "cohort": 799,
    "routed": 200,

    "formula":
        "final_logit = stage1_logit + gate_score * effective_ensemble_logit_delta",

    "effective_delta_definition":
        "logit(frozen_B0_final_probability) - logit(frozen_stage1_probability)",

    "stored_bounded_delta_used_for_soft_gate": False,

    "reason":
        "final B0 is a probability-space neural-seed ensemble; stored delta is not algebraically identical to the ensemble-level logit difference",

    "gate_scores": [
        "score_S2",
        "score_S3",
    ],

    "new_hyperparameters": False,
    "threshold_search": False,

    "bounded_model_retrained": False,
    "gate_model_retrained": False,

    "future_rows_loaded": 0,
    "ego_consumed": False,

    "effective_delta_max_abs":
        max_effective_delta,

    "B0_reconstruction_max_abs_diff":
        reconstruction_max_abs_diff,

    "stored_vs_effective_delta_max_abs_diff":
        stored_delta_mismatch_max_abs,
}

(
    ROOT / "protocol.json"
).write_text(
    json.dumps(
        protocol,
        indent=2,
        ensure_ascii=False,
    ) + "\n",
    encoding="utf-8",
)

print("SOFT_NEEDGATE_INTEGRATION_PASS")

print(
    pd.DataFrame(rows)
    .to_string(index=False)
)

print()
print("output =", ROOT)

