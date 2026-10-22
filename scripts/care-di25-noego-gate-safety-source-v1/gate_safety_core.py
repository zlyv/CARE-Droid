#!/usr/bin/env python3
"""Fixed correction rule and train-fold-only safety-threshold selection."""
from __future__ import annotations

import numpy as np

MAX_DELTA = 2.25
CROSSING_MARGIN = 0.05
BROKEN_COST = 1.5
CANDIDATE_THRESHOLDS = np.round(np.arange(0.50, 1.001, 0.01), 2)


def logit(p):
    p = np.asarray(p, dtype=np.float64)
    return np.log(np.clip(p, 1e-6, 1 - 1e-6) / np.clip(1 - p, 1e-6, 1 - 1e-6))


def sigmoid(x):
    x = np.asarray(x, dtype=np.float64)
    return np.where(x >= 0, 1 / (1 + np.exp(-x)), np.exp(x) / (1 + np.exp(x)))


def apply_rule(p_stage1, gate_score, route, threshold):
    p = np.asarray(p_stage1, dtype=np.float64)
    q = np.asarray(gate_score, dtype=np.float64)
    route = np.asarray(route, dtype=bool)
    base = logit(p)
    active = route & (q >= float(threshold))
    direction = np.where(base >= 0, -1.0, 1.0)
    required = np.abs(base) + CROSSING_MARGIN
    magnitude = np.minimum(required, MAX_DELTA)
    delta = np.where(active, direction * magnitude, 0.0)
    return sigmoid(base + delta), active, active & (required > MAX_DELTA)


def counts(y, p_stage1, p_final):
    y = np.asarray(y, dtype=np.int64)
    before = (np.asarray(p_stage1) >= 0.5) == y
    after = (np.asarray(p_final) >= 0.5) == y
    fixed = int(np.sum((~before) & after))
    broken = int(np.sum(before & (~after)))
    return fixed, broken


def choose_threshold(y, p_stage1, gate_score, route):
    """Predeclared utility, with conservative deterministic tie breaks."""
    rows = []
    for threshold in CANDIDATE_THRESHOLDS:
        final, active, bound = apply_rule(p_stage1, gate_score, route, threshold)
        fixed, broken = counts(y, p_stage1, final)
        rows.append({"threshold": float(threshold), "fixed": fixed, "broken": broken,
                     "utility": float(fixed - BROKEN_COST * broken),
                     "active": int(active.sum()), "bound_limited": int(bound.sum())})
    # Max utility; then fewer broken; then more fixed; then higher threshold.
    best = max(rows, key=lambda r: (r["utility"], -r["broken"], r["fixed"], r["threshold"]))
    return best, rows

