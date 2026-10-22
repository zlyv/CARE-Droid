#!/usr/bin/env python3
"""Pure functions for the source-only decoupled correction-authority audit."""
from __future__ import annotations

import numpy as np

MAX_DELTA = 2.25
GATE_THRESHOLD = 0.5
CROSSING_MARGIN = 0.05


def logit(probability):
    p = np.asarray(probability, dtype=np.float64)
    if not np.isfinite(p).all() or np.any((p < 0.0) | (p > 1.0)):
        raise ValueError("probability must be finite and in [0,1]")
    return np.log(np.clip(p, 1e-6, 1 - 1e-6) / np.clip(1 - p, 1e-6, 1 - 1e-6))


def sigmoid(value):
    x = np.asarray(value, dtype=np.float64)
    return np.where(x >= 0, 1.0 / (1.0 + np.exp(-x)), np.exp(x) / (1.0 + np.exp(x)))


def decoupled_correction(stage1_probability, gate_score, routed):
    """Apply fixed gate/direction/magnitude semantics without label access.

    Gate: fixed OOF score >= 0.5.
    Direction: opposite the current Stage-I hard decision.
    Magnitude: minimum amount needed to cross zero by 0.05 logit, capped at 2.25.
    """
    p = np.asarray(stage1_probability, dtype=np.float64)
    q = np.asarray(gate_score, dtype=np.float64)
    route = np.asarray(routed, dtype=bool)
    if p.shape != q.shape or p.shape != route.shape:
        raise ValueError("stage1_probability, gate_score and routed must have identical shapes")
    if not np.isfinite(q).all() or np.any((q < 0.0) | (q > 1.0)):
        raise ValueError("gate score must be finite and in [0,1]")
    base = logit(p)
    active = route & (q >= GATE_THRESHOLD)
    direction = np.where(base >= 0.0, -1.0, 1.0)
    required = np.abs(base) + CROSSING_MARGIN
    magnitude = np.minimum(required, MAX_DELTA)
    delta = np.where(active, direction * magnitude, 0.0)
    final = sigmoid(base + delta)
    bound_limited = active & (required > MAX_DELTA)
    return {
        "probability": final,
        "active": active,
        "direction": direction,
        "required_magnitude": required,
        "magnitude": np.where(active, magnitude, 0.0),
        "delta": delta,
        "bound_limited": bound_limited,
    }

