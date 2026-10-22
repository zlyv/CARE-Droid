#!/usr/bin/env python3
"""Shared contracts for the CARE-Droid no-Ego need-gate probe."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

YEARS = (2018, 2019, 2020, 2021)
EXPECTED_YEAR_COUNTS = {2018: 793, 2019: 399, 2020: 397, 2021: 400}
FOLDS = (1, 2, 3, 4, 5)
SEEDS = (900000, 900001, 900002)
ARMS = ("S0_STAGE1_LOGIT", "S1_GLOBAL24_LOGIT", "S2_NOEGO_HOST_ANOMALY", "S3_NOEGO_HOST_ANOMALY_LOGIT")
REPRESENTATION_WIDTH = 88
BOOTSTRAP_ITERATIONS = 10000
BOOTSTRAP_SEED = 20260920
PROBE_SEED = 20260920


def blocked(message: str) -> RuntimeError:
    return RuntimeError("BLOCKED_BY_CONTRACT: " + message)


def norm(value) -> str:
    return str(value or "").strip().lower()


def read_csv(path: Path) -> list[dict]:
    if not path.is_file():
        raise blocked(f"missing required file: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise blocked(f"refusing to write empty CSV: {path}")
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def stage1_logit(probability) -> np.ndarray:
    p = np.asarray(probability, dtype=np.float64)
    if not np.isfinite(p).all() or np.any((p < 0.0) | (p > 1.0)):
        raise blocked("p_stage1 must be finite probabilities in [0,1]")
    eps = 1e-6
    return np.log((p + eps) / (1.0 - p + eps))


def need_correct(label, probability) -> np.ndarray:
    y = np.asarray(label, dtype=np.int64)
    p = np.asarray(probability, dtype=np.float64)
    if not set(np.unique(y).tolist()).issubset({0, 1}):
        raise blocked("labels must be binary")
    return ((p >= 0.5) != y.astype(bool)).astype(np.int64)


def validate_representation(value, expected_rows: int) -> np.ndarray:
    x = np.asarray(value, dtype=np.float32)
    if x.shape != (expected_rows, REPRESENTATION_WIDTH):
        raise blocked(f"true no-Ego representation must be ({expected_rows},88), received {x.shape}")
    if not np.isfinite(x).all():
        raise blocked("true no-Ego representation contains non-finite values")
    return x


def validate_partition(train_ids, validation_ids) -> None:
    train = {norm(x) for x in train_ids}
    validation = {norm(x) for x in validation_ids}
    if not train or not validation:
        raise blocked("empty outer-train or outer-validation partition")
    overlap = train & validation
    if overlap:
        raise blocked(f"outer split overlap: {len(overlap)} SHA values")


def fit_probe(x_train, y_train, x_validation, scaler_factory=StandardScaler) -> np.ndarray:
    x_train = np.asarray(x_train, dtype=np.float64)
    x_validation = np.asarray(x_validation, dtype=np.float64)
    y_train = np.asarray(y_train, dtype=np.int64)
    if x_train.ndim != 2 or x_validation.ndim != 2 or x_train.shape[1] != x_validation.shape[1]:
        raise blocked("probe matrices have incompatible dimensions")
    if len(x_train) != len(y_train) or not np.isfinite(x_train).all() or not np.isfinite(x_validation).all():
        raise blocked("invalid probe matrices")
    if set(np.unique(y_train).tolist()) != {0, 1}:
        raise blocked("outer-train routed need_correct target has only one class")
    model = make_pipeline(
        scaler_factory(),
        LogisticRegression(C=1.0, solver="liblinear", class_weight="balanced", max_iter=5000, random_state=PROBE_SEED),
    )
    model.fit(x_train, y_train)
    return model.predict_proba(x_validation)[:, 1]


def classification_metrics(y_true, probability) -> dict:
    y = np.asarray(y_true, dtype=np.int64)
    p = np.asarray(probability, dtype=np.float64)
    if len(y) != len(p) or len(y) == 0 or not np.isfinite(p).all():
        raise blocked("invalid metric inputs")
    prediction = (p >= 0.5).astype(np.int64)
    two_classes = set(np.unique(y).tolist()) == {0, 1}
    tn, fp, fn, tp = confusion_matrix(y, prediction, labels=[0, 1]).ravel()
    return {
        "N": int(len(y)),
        "positive_n": int(np.sum(y == 1)),
        "ROC_AUC": float(roc_auc_score(y, p)) if two_classes else float("nan"),
        "PR_AUC": float(average_precision_score(y, p)) if two_classes else float("nan"),
        "BA": float(balanced_accuracy_score(y, prediction)) if two_classes else float("nan"),
        "F1": float(f1_score(y, prediction, zero_division=0)),
        "Precision": float(precision_score(y, prediction, zero_division=0)),
        "Recall": float(recall_score(y, prediction, zero_division=0)),
        "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp),
    }


def paired_bootstrap(y_true, baseline, candidate, metric: str, iterations: int = BOOTSTRAP_ITERATIONS,
                     seed: int = BOOTSTRAP_SEED) -> dict:
    y = np.asarray(y_true, dtype=np.int64)
    a = np.asarray(baseline, dtype=np.float64)
    b = np.asarray(candidate, dtype=np.float64)
    if not (len(y) == len(a) == len(b)) or set(np.unique(y).tolist()) != {0, 1}:
        raise blocked("paired bootstrap requires aligned two-class OOF predictions")
    if metric == "ROC_AUC":
        score = roc_auc_score
    elif metric == "PR_AUC":
        score = average_precision_score
    else:
        raise blocked(f"unsupported bootstrap metric: {metric}")
    rng = np.random.default_rng(seed)
    deltas = []
    for _ in range(int(iterations)):
        index = rng.integers(0, len(y), len(y))
        yy = y[index]
        if len(np.unique(yy)) < 2:
            continue
        deltas.append(float(score(yy, b[index]) - score(yy, a[index])))
    if not deltas:
        raise blocked("paired bootstrap produced no valid resample")
    values = np.asarray(deltas, dtype=np.float64)
    return {
        "metric": metric,
        "delta_point": float(score(y, b) - score(y, a)),
        "delta_bootstrap_mean": float(np.mean(values)),
        "ci_low": float(np.percentile(values, 2.5)),
        "ci_high": float(np.percentile(values, 97.5)),
        "iterations_requested": int(iterations),
        "iterations_valid": int(len(values)),
        "bootstrap_seed": int(seed),
    }


def noego_contract() -> dict:
    return {
        "contract": "TRUE_HOST_ONLY_NO_EGO_V1",
        "ego_consumed": False,
        "ego_zero_filled": False,
        "ego_encoder_instantiated": False,
        "ego_gate_instantiated": False,
        "host_asset_keys_consumed": ["host_values", "host_offsets", "real_host_count"],
        "ego_asset_keys_consumed": [],
        "representation": {
            "name": "Host-only Anomaly structural tail88",
            "dimension": 88,
            "components": [
                {"name": "role_graph_token", "dimension": 32},
                {"name": "critical_host", "dimension": 24},
                {"name": "attended_host", "dimension": 24},
                {"name": "scalar_summary", "dimension": 8},
            ],
            "excludes": ["Global24", "Stage-I logit", "Ego embedding", "Ego gate", "delta head output"],
        },
        "missing_host_policy": "reuse the historical deterministic Host-only placeholder bag; never drop an APK",
    }
