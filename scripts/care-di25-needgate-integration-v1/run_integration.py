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

ROOT = Path("artifacts/care-di25-needgate-integration-v1")
ROOT.mkdir(parents=True, exist_ok=True)

ROUTER = Path(
    "artifacts/care-source-di25-reconstructed-v1/source_router_inputs.csv"
)
GATE = Path(
    "artifacts/care-di25-noego-needgate-source-v1/oof_predictions.csv"
)

BASE_CANDIDATES = [
    Path("artifacts/care-di25-host-input-replay-v1/arms/FULL/source_predictions.csv"),
    Path("artifacts/care-di25-host-input-replay-v1/arms/FULL/predictions.csv"),
    Path("artifacts/care-di25-host-input-replay-v1/arms/FULL/oof_predictions.csv"),
]


def blocked(msg):
    raise RuntimeError("BLOCKED_BY_CONTRACT: " + msg)


def find_col(df, exact=(), contains=()):
    for c in exact:
        if c in df.columns:
            return c

    hits = []
    for c in df.columns:
        low = c.lower()
        if all(x.lower() in low for x in contains):
            hits.append(c)

    if len(hits) == 1:
        return hits[0]
    if len(hits) > 1:
        blocked(f"ambiguous columns for {contains}: {hits}")
    return None


def metrics(y, p):
    pred = (p >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "BA": balanced_accuracy_score(y, pred),
        "AUC": roc_auc_score(y, p),
        "F1": f1_score(y, pred, zero_division=0),
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred, zero_division=0),
        "specificity": tn / (tn + fp) if (tn + fp) else np.nan,
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
    }


router = pd.read_csv(ROUTER)
gate = pd.read_csv(GATE)

base_path = next((p for p in BASE_CANDIDATES if p.is_file()), None)
if base_path is None:
    blocked("cannot find FULL source prediction file")

base = pd.read_csv(base_path)

# --------------------------
# Router hard contract
# --------------------------
if len(router) != 799:
    blocked(f"router rows != 799: {len(router)}")

if router["sha256"].astype(str).str.lower().nunique() != 799:
    blocked("router SHA not unique")

if set(router["year"].astype(int)) != {2017}:
    blocked("router contains non-2017 rows")

if int(router["route_DI"].sum()) != 200:
    blocked("route_DI total != 200")

route_by_fold = (
    router.groupby("fold")["route_DI"].sum().astype(int).to_dict()
)
if route_by_fold != {1:40, 2:40, 3:40, 4:40, 5:40}:
    blocked(f"route_DI/fold changed: {route_by_fold}")

# --------------------------
# Gate hard contract
# --------------------------
if len(gate) != 200:
    blocked(f"Need-Gate OOF rows != 200: {len(gate)}")

if gate["sha256"].astype(str).str.lower().nunique() != 200:
    blocked("Need-Gate SHA not unique")

for c in ["pred_S2", "pred_S3"]:
    if c not in gate.columns:
        blocked(f"missing gate column {c}")
    vals = set(gate[c].astype(int).unique())
    if not vals.issubset({0, 1}):
        blocked(f"{c} is not binary")

router["sha"] = router["sha256"].astype(str).str.lower()
gate["sha"] = gate["sha256"].astype(str).str.lower()
base["sha"] = base["sha256"].astype(str).str.lower()

routed_sha = set(router.loc[router.route_DI == 1, "sha"])
gate_sha = set(gate["sha"])

if routed_sha != gate_sha:
    blocked(
        f"Need-Gate cohort != DI25 routed cohort; "
        f"missing={len(routed_sha-gate_sha)}, extra={len(gate_sha-routed_sha)}"
    )

if len(base) != 799 or base["sha"].nunique() != 799:
    blocked("FULL baseline must contain exactly 799 unique SHA")

if set(base["sha"]) != set(router["sha"]):
    blocked("FULL baseline cohort != router cohort")

# --------------------------
# Detect baseline columns
# --------------------------
stage1_col = find_col(
    base,
    exact=[
        "Global24__prob",
        "p_stage1",
        "Global24_prob",
    ],
    contains=["global24", "prob"],
)

if stage1_col is None:
    # canonical p_stage1 is always available from router
    stage1_source = "router.p_stage1"
else:
    stage1_source = f"baseline.{stage1_col}"

full_col = find_col(
    base,
    exact=[
        "DI25__Host__Anomaly__FULL__Bounded__prob",
        "DI25__Host__Anomaly__FULL__Bounded",
        "p_final",
        "final_probability",
    ],
    contains=["di25", "host", "anomaly", "full", "bounded"],
)

if full_col is None:
    # weaker fallback: one unique bounded probability column
    candidates = [
        c for c in base.columns
        if "bounded" in c.lower()
        and ("prob" in c.lower() or "di25" in c.lower())
    ]
    if len(candidates) != 1:
        blocked(
            "cannot uniquely identify FULL bounded probability column; "
            f"candidates={candidates}"
        )
    full_col = candidates[0]

# Align by router order
base = base.set_index("sha").loc[router["sha"]].reset_index()
gate_by = gate.set_index("sha")

y = router["label"].astype(int).to_numpy()
route = router["route_DI"].astype(int).to_numpy()

p_stage1 = router["p_stage1"].astype(float).to_numpy()

if stage1_col is not None:
    baseline_stage1 = base[stage1_col].astype(float).to_numpy()
    max_diff = float(np.max(np.abs(baseline_stage1 - p_stage1)))
    if max_diff > 1e-10:
        blocked(f"baseline Stage-I differs from router p_stage1: maxdiff={max_diff}")

p_b0 = base[full_col].astype(float).to_numpy()

if not np.isfinite(p_b0).all():
    blocked("FULL baseline contains non-finite probability")

# Unrouted FULL samples must preserve Stage-I probability.
unrouted = route == 0
max_unrouted_diff = float(
    np.max(np.abs(p_b0[unrouted] - p_stage1[unrouted]))
)
if max_unrouted_diff > 1e-8:
    blocked(
        "FULL baseline changes unrouted samples: "
        f"max abs diff={max_unrouted_diff}"
    )

pred_s2 = np.zeros(799, dtype=int)
pred_s3 = np.zeros(799, dtype=int)

for i, sha in enumerate(router["sha"]):
    if route[i] == 1:
        pred_s2[i] = int(gate_by.loc[sha, "pred_S2"])
        pred_s3[i] = int(gate_by.loc[sha, "pred_S3"])

# Gate semantics:
# 0 KEEP -> p_stage1
# 1 CORRECT -> original FULL bounded probability
p_b1 = p_stage1.copy()
p_b2 = p_stage1.copy()

allow_s2 = (route == 1) & (pred_s2 == 1)
allow_s3 = (route == 1) & (pred_s3 == 1)

p_b1[allow_s2] = p_b0[allow_s2]
p_b2[allow_s3] = p_b0[allow_s3]

# --------------------------
# End-to-end metrics
# --------------------------
stage1_pred = (p_stage1 >= 0.5).astype(int)


def correction_stats(p):
    pred = (p >= 0.5).astype(int)
    stage1_wrong = stage1_pred != y
    stage1_correct = ~stage1_wrong

    fixed = int(np.sum(stage1_wrong & (pred == y)))
    broken = int(np.sum(stage1_correct & (pred != y)))
    net = fixed - broken

    changed = int(np.sum(pred != stage1_pred))
    precision = fixed / (fixed + broken) if (fixed + broken) else np.nan
    recall = fixed / int(np.sum(stage1_wrong)) if np.sum(stage1_wrong) else np.nan

    return {
        "fixed": fixed,
        "broken": broken,
        "net": net,
        "class_flips_vs_stage1": changed,
        "correction_precision": precision,
        "error_repair_recall": recall,
    }


arms = {
    "StageI_Global24": p_stage1,
    "B0_FULL_no_gate": p_b0,
    "B1_S2_gate": p_b1,
    "B2_S3_gate": p_b2,
}

metric_rows = []
for name, p in arms.items():
    row = {
        "arm": name,
        "n": 799,
        **metrics(y, p),
        **correction_stats(p),
    }
    if name == "B1_S2_gate":
        row["gate_passed_routed"] = int(np.sum(allow_s2))
        row["gate_kept_routed"] = int(np.sum(route) - np.sum(allow_s2))
    elif name == "B2_S3_gate":
        row["gate_passed_routed"] = int(np.sum(allow_s3))
        row["gate_kept_routed"] = int(np.sum(route) - np.sum(allow_s3))
    else:
        row["gate_passed_routed"] = np.nan
        row["gate_kept_routed"] = np.nan
    metric_rows.append(row)

metrics_df = pd.DataFrame(metric_rows)
metrics_df.to_csv(ROOT / "end_to_end_metrics.csv", index=False)

# --------------------------
# Per-fold metrics
# --------------------------
fold_rows = []
for fold in range(1, 6):
    idx = router["fold"].astype(int).to_numpy() == fold
    for name, p in arms.items():
        fold_rows.append({
            "fold": fold,
            "arm": name,
            "n": int(np.sum(idx)),
            **metrics(y[idx], p[idx]),
        })
pd.DataFrame(fold_rows).to_csv(ROOT / "per_fold_metrics.csv", index=False)

# --------------------------
# Sample-level audit
# --------------------------
audit = router[
    ["sha256", "year", "label", "fold", "route_DI", "p_stage1"]
].copy()

audit["p_B0_FULL"] = p_b0
audit["gate_S2"] = pred_s2
audit["p_B1_S2_gate"] = p_b1
audit["gate_S3"] = pred_s3
audit["p_B2_S3_gate"] = p_b2

for name, p in [
    ("B0", p_b0),
    ("B1", p_b1),
    ("B2", p_b2),
]:
    pred = (p >= 0.5).astype(int)
    audit[f"pred_{name}"] = pred
    audit[f"fixed_{name}"] = (
        (stage1_pred != y) & (pred == y)
    ).astype(int)
    audit[f"broken_{name}"] = (
        (stage1_pred == y) & (pred != y)
    ).astype(int)

audit.to_csv(ROOT / "sample_audit.csv", index=False)

protocol = {
    "experiment": "CARE-Droid 2017 source-only Need-Gate integration v1",
    "cohort_n": 799,
    "routed_n": 200,
    "gate_threshold": 0.5,
    "gate_predictions": "strict outer-fold OOF from Need-Gate experiment",
    "B0": "original DI25 Host-only Anomaly FULL BoundedCorrection",
    "B1": "S2 Host-only gate; KEEP restores p_stage1; CORRECT uses original B0 probability",
    "B2": "S3 Host+Stage-I-logit gate; KEEP restores p_stage1; CORRECT uses original B0 probability",
    "bounded_model_retrained": False,
    "gate_retrained": False,
    "threshold_tuned": False,
    "future_rows_loaded": 0,
    "ego_consumed": False,
    "router": str(ROUTER),
    "gate_oof": str(GATE),
    "baseline_predictions": str(base_path),
    "baseline_probability_column": full_col,
    "stage1_probability_source": stage1_source,
    "unrouted_max_abs_diff_baseline_vs_stage1": max_unrouted_diff,
}

(ROOT / "protocol.json").write_text(
    json.dumps(protocol, indent=2, ensure_ascii=False) + "\n",
    encoding="utf-8",
)

print("NEEDGATE_INTEGRATION_PASS")
print(metrics_df.to_string(index=False))
print()
print("output =", ROOT)
