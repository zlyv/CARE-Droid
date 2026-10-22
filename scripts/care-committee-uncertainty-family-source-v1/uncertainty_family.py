#!/usr/bin/env python3
"""Canonical uncertainty scores for a calibrated binary classifier committee."""

from __future__ import annotations

import math
import numpy as np


MEMBERS = ("p_lr", "p_rf", "p_xgb", "p_svm")


def entropy(p: np.ndarray) -> np.ndarray:
    x = np.clip(np.asarray(p, dtype=np.float64), 1e-12, 1.0 - 1e-12)
    return -(x * np.log2(x) + (1.0 - x) * np.log2(1.0 - x))


def scores(stage1: np.ndarray, member: np.ndarray) -> dict[str, np.ndarray]:
    base = np.asarray(stage1, dtype=np.float64)
    p = np.asarray(member, dtype=np.float64)
    if base.ndim != 1 or p.shape != (len(base), 4):
        raise ValueError("expected stage1 [n] and committee [n,4]")
    if not np.isfinite(base).all() or not np.isfinite(p).all():
        raise ValueError("non-finite probability")
    if np.any((base < 0) | (base > 1)) or np.any((p < 0) | (p > 1)):
        raise ValueError("probabilities must be in [0,1]")

    mean = p.mean(axis=1)
    pe = entropy(mean)
    member_entropy = entropy(p)
    expected_entropy = member_entropy.mean(axis=1)
    mi = np.maximum(pe - expected_entropy, 0.0)
    decision = (base >= 0.5).astype(np.float64)
    opposition = np.where(decision[:, None] == 1.0, 1.0 - p, p)
    vote_fraction = (p >= 0.5).mean(axis=1)

    out = {
        "A25": 1.0 - 2.0 * np.abs(base - 0.5),
        "LR_Entropy25": member_entropy[:, 0],
        "RF_Entropy25": member_entropy[:, 1],
        "XGB_Entropy25": member_entropy[:, 2],
        "SVM_Entropy25": member_entropy[:, 3],
        "MeanMemberEntropy25": expected_entropy,
        "PredictiveEntropy25": pe,
        "MI_JSD25": mi,
        "ProbabilityStd25": p.std(axis=1),
        "ProbabilityRange25": p.max(axis=1) - p.min(axis=1),
        "VoteEntropy25": entropy(vote_fraction),
        "MeanOpposition_DI25": opposition.mean(axis=1),
        "MaxOpposition25": opposition.max(axis=1),
    }
    for name, value in out.items():
        if value.shape != base.shape or not np.isfinite(value).all():
            raise RuntimeError("invalid score: " + name)
    return out


def exact_fold_top(score: np.ndarray, folds: np.ndarray, sha: np.ndarray,
                   budget: float = 0.25) -> np.ndarray:
    """Exact foldwise budget with SHA as a deterministic tie breaker."""
    value = np.asarray(score, dtype=np.float64)
    group = np.asarray(folds, dtype=np.int64)
    ids = np.asarray(sha, dtype=str)
    route = np.zeros(len(value), dtype=bool)
    for fold in sorted(set(group.tolist())):
        index = np.flatnonzero(group == fold)
        count = int(round(float(budget) * len(index)))
        # lexsort: last key is primary; descending score, ascending SHA.
        order = np.lexsort((ids[index], -value[index]))
        route[index[order[:count]]] = True
    return route


def exact_binomial_two_sided(b: int, c: int) -> float:
    n = int(b + c)
    if n == 0:
        return 1.0
    k = min(int(b), int(c))
    tail = sum(math.comb(n, i) for i in range(k + 1)) / (2.0 ** n)
    return min(1.0, 2.0 * tail)

