#!/usr/bin/env python3
"""Pure statistical helpers for the GraphGlobal path paired audit."""

from __future__ import annotations

import math

import numpy as np
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score


def exact_mcnemar(n01: int, n10: int) -> float:
    """Two-sided exact binomial McNemar p-value."""
    n01, n10 = int(n01), int(n10)
    total = n01 + n10
    if total == 0:
        return 1.0
    lower = min(n01, n10)
    probability = sum(math.comb(total, k) for k in range(lower + 1)) / (2.0 ** total)
    return float(min(1.0, 2.0 * probability))


def metrics(y, probability, stage1_probability):
    y = np.asarray(y, dtype=np.int64)
    probability = np.asarray(probability, dtype=np.float64)
    stage1_probability = np.asarray(stage1_probability, dtype=np.float64)
    pred = probability >= 0.5
    base_pred = stage1_probability >= 0.5
    fixed = int(np.sum((base_pred != y) & (pred == y)))
    broken = int(np.sum((base_pred == y) & (pred != y)))
    return {
        "N": int(len(y)),
        "BA": float(balanced_accuracy_score(y, pred)),
        "AUC": float(roc_auc_score(y, probability)),
        "F1": float(f1_score(y, pred, zero_division=0)),
        "fixed": fixed,
        "broken": broken,
        "net": fixed - broken,
    }


def stratified_bootstrap_indices(y, iterations: int, seed: int):
    y = np.asarray(y, dtype=np.int64)
    groups = [np.flatnonzero(y == value) for value in (0, 1)]
    if any(len(group) == 0 for group in groups):
        raise ValueError("both labels are required")
    rng = np.random.RandomState(int(seed))
    for _ in range(int(iterations)):
        sampled = [rng.choice(group, size=len(group), replace=True) for group in groups]
        yield np.concatenate(sampled)


def percentile_interval(values):
    values = np.asarray(values, dtype=np.float64)
    return {
        "bootstrap_mean": float(np.mean(values)),
        "ci_low": float(np.percentile(values, 2.5)),
        "ci_high": float(np.percentile(values, 97.5)),
    }

