#!/usr/bin/env python3
from __future__ import annotations

import numpy as np

MAX_DELTA = 2.25
MARGIN = 0.05
BROKEN_COST = 1.5
THRESHOLDS = np.round(np.arange(.50, 1.001, .01), 2)


def logit(p):
    p = np.asarray(p, dtype=np.float64)
    return np.log(np.clip(p, 1e-6, 1-1e-6) / np.clip(1-p, 1e-6, 1-1e-6))


def sigmoid(x):
    x = np.asarray(x, dtype=np.float64)
    return np.where(x >= 0, 1/(1+np.exp(-x)), np.exp(x)/(1+np.exp(x)))


def apply(p, score, route, threshold):
    p, score, route = np.asarray(p), np.asarray(score), np.asarray(route, dtype=bool)
    z = logit(p); active = route & (score >= float(threshold))
    required = np.abs(z) + MARGIN
    direction = np.where(z >= 0, -1., 1.)
    delta = np.where(active, direction * np.minimum(required, MAX_DELTA), 0.)
    return sigmoid(z + delta), active, active & (required > MAX_DELTA), delta


def fixed_broken(y, before, after):
    y = np.asarray(y, dtype=np.int64)
    a = (np.asarray(before) >= .5) == y; b = (np.asarray(after) >= .5) == y
    return int(np.sum((~a) & b)), int(np.sum(a & (~b)))


def choose(y, p, score, route):
    rows = []
    for t in THRESHOLDS:
        final, active, limited, _ = apply(p, score, route, t)
        fixed, broken = fixed_broken(y, p, final)
        rows.append({"threshold": float(t), "fixed": fixed, "broken": broken,
                     "utility": float(fixed-BROKEN_COST*broken),
                     "active": int(active.sum()), "bound_limited": int(limited.sum())})
    return max(rows, key=lambda r: (r["utility"], -r["broken"], r["fixed"], r["threshold"])), rows

