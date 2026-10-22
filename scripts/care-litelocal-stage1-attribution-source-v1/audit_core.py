#!/usr/bin/env python3
from __future__ import annotations

import numpy as np
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score


MEMBERS = ("LR", "RF", "XGB", "SVM")


def stage1_probability(member_probabilities: np.ndarray, definition: str) -> np.ndarray:
    p = np.asarray(member_probabilities, dtype=float)
    if p.ndim != 2 or p.shape[1] != 4:
        raise ValueError("expected N x 4 member probabilities")
    if definition == "LR_ONLY":
        return p[:, 0]
    if definition == "MEAN_COMMITTEE":
        return p.mean(axis=1)
    if definition == "MEDIAN_COMMITTEE":
        return np.median(p, axis=1)
    raise KeyError(definition)


def mean_opposition(member_probabilities: np.ndarray, stage1_p: np.ndarray) -> np.ndarray:
    p = np.asarray(member_probabilities, dtype=float)
    decision = np.asarray(stage1_p) >= 0.5
    return np.where(decision[:, None], 1.0 - p, p).mean(axis=1)


def foldwise_top_budget(score, folds, ids, fraction=0.25):
    score = np.asarray(score, dtype=float)
    folds = np.asarray(folds, dtype=int)
    ids = np.asarray(ids).astype(str)
    route = np.zeros(len(score), dtype=bool)
    for fold in sorted(set(folds.tolist())):
        idx = np.flatnonzero(folds == fold)
        count = int(round(len(idx) * fraction))
        order = np.lexsort((ids[idx], -score[idx]))
        route[idx[order[:count]]] = True
    return route


def metric_row(y, p, sample_weight=None):
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)
    pred = p >= 0.5
    return {
        "BA": float(balanced_accuracy_score(y, pred, sample_weight=sample_weight)),
        "AUC": float(roc_auc_score(y, p, sample_weight=sample_weight)),
        "F1": float(f1_score(y, pred, sample_weight=sample_weight, zero_division=0)),
        "errors": int(np.sum(pred != y)),
    }


def effective_sample_size(weights):
    w = np.asarray(weights, dtype=float)
    return float(w.sum() ** 2 / np.square(w).sum())


def stratified_bootstrap_auc_delta(y, p_candidate, p_baseline, iterations=2000, seed=20260928):
    y = np.asarray(y, dtype=int)
    a = np.asarray(p_candidate, dtype=float)
    b = np.asarray(p_baseline, dtype=float)
    rng = np.random.default_rng(seed)
    class_idx = [np.flatnonzero(y == value) for value in (0, 1)]
    values = np.empty(iterations, dtype=float)
    for i in range(iterations):
        sample = np.concatenate([rng.choice(idx, len(idx), replace=True) for idx in class_idx])
        values[i] = roc_auc_score(y[sample], a[sample]) - roc_auc_score(y[sample], b[sample])
    return float(np.mean(values)), float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))
