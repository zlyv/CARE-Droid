#!/usr/bin/env python3
from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np

from common import binary_auc, logit, read_gzip_json, sigmoid

ARMS = ("REAL", "ONE_HOP", "ROOT_SHUFFLE", "CALLSITE_SHUFFLE", "COUNT_ONLY", "PATH_ONLY")
COUNT_FIELDS = ("graph_method_count", "graph_edge_count", "root_count", "host_count")


def load_feature_matrix(rows: list[dict[str, Any]], feature_dir: Path, arm: str) -> tuple[np.ndarray, list[str]]:
    records = [read_gzip_json(feature_dir / f"{row['sha256']}.json.gz") for row in rows]
    if arm in {"REAL", "ONE_HOP", "ROOT_SHUFFLE", "CALLSITE_SHUFFLE"}:
        variant = arm
        fields = sorted(key for key in records[0]["variants"][variant] if key not in COUNT_FIELDS)
    elif arm == "COUNT_ONLY":
        variant = "REAL"; fields = list(COUNT_FIELDS)
    elif arm == "PATH_ONLY":
        variant = "REAL"
        fields = sorted(key for key in records[0]["variants"][variant]
                        if any(token in key for token in ("inverse_distance", "reachable", "user_share", "auto_share")))
    else:
        raise ValueError(arm)
    matrix = np.asarray([[float(record["variants"][variant].get(field, 0.0)) for field in fields] for record in records], float)
    if arm == "COUNT_ONLY":
        matrix = np.log1p(np.maximum(matrix, 0.0))
    if not np.isfinite(matrix).all():
        raise RuntimeError(f"nonfinite feature matrix arm={arm}")
    return matrix, fields


def fit_score(x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray, seed: int) -> np.ndarray:
    y_train = np.asarray(y_train, int)
    if len(np.unique(y_train)) < 2:
        return np.full(len(x_test), float(np.mean(y_train)) if len(y_train) else .5)
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    model = make_pipeline(StandardScaler(), LogisticRegression(C=.25, class_weight="balanced", solver="liblinear",
                                                                 max_iter=2000, random_state=seed))
    model.fit(x_train, y_train)
    return model.predict_proba(x_test)[:, 1]


def crossfit_scores(rows: list[dict[str, Any]], x: np.ndarray, folds: np.ndarray, seed: int) -> tuple[np.ndarray, list[dict[str, Any]]]:
    y = np.asarray([row["label"] for row in rows], int)
    pred = np.asarray([row["stage1_pred"] for row in rows], int)
    scores = np.full(len(rows), np.nan); fold_metrics = []
    for fold in sorted(set(folds.tolist())):
        train_all, test_all = folds != fold, folds == fold
        for side in (0, 1):
            train = train_all & (pred == side); test = test_all & (pred == side)
            target_train = (y[train] == (1 if side == 0 else 0)).astype(int)
            local = fit_score(x[train], target_train, x[test], seed + 100 * int(fold) + side)
            scores[test] = local
            target_test = (y[test] == (1 if side == 0 else 0)).astype(int)
            fold_metrics.append({"fold": int(fold), "side": "FN_vs_TN" if side == 0 else "FP_vs_TP",
                                 "n": int(test.sum()), "positive": int(target_test.sum()),
                                 "AUC": binary_auc(target_test, local)})
    if not np.isfinite(scores).all():
        raise RuntimeError("crossfit scores incomplete")
    return scores, fold_metrics


def correction_probability(rows: list[dict[str, Any]], score: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    base = np.asarray([row["stage1_prob"] for row in rows], float)
    route = np.asarray([row["routed"] for row in rows], bool)
    pred = base >= .5
    evidence = np.maximum(logit(score), 0.0)
    raw = np.where(pred, -evidence, evidence)
    delta = 2.25 * np.tanh(raw)
    delta[~route] = 0.0
    probability = sigmoid(logit(base) + delta)
    return probability, delta


def direction_summary(rows: list[dict[str, Any]], score: np.ndarray, fold_metrics: list[dict[str, Any]]) -> dict[str, Any]:
    y = np.asarray([row["label"] for row in rows], int)
    pred = np.asarray([row["stage1_pred"] for row in rows], int)
    route = np.asarray([row["routed"] for row in rows], bool)
    p0, p1 = pred == 0, pred == 1
    return {"FN_vs_TN_AUC": binary_auc((y[p0] == 1).astype(int), score[p0]),
            "FP_vs_TP_AUC": binary_auc((y[p1] == 0).astype(int), score[p1]),
            "routed_FN_vs_TN_AUC": binary_auc((y[p0 & route] == 1).astype(int), score[p0 & route]),
            "routed_FP_vs_TP_AUC": binary_auc((y[p1 & route] == 0).astype(int), score[p1 & route]),
            "FN_n": int(p0.sum()), "FP_n": int(p1.sum()), "folds": fold_metrics,
            "FN_consistent_folds": sum(math.isfinite(float(row["AUC"])) and float(row["AUC"]) > .5 for row in fold_metrics if row["side"] == "FN_vs_TN"),
            "FP_consistent_folds": sum(math.isfinite(float(row["AUC"])) and float(row["AUC"]) > .5 for row in fold_metrics if row["side"] == "FP_vs_TP")}
