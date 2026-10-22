#!/usr/bin/env python3
"""Small, dependency-light helpers for frozen future routing audits."""

from __future__ import annotations

import numpy as np


def exact_top_fraction(score: np.ndarray, budget: float) -> np.ndarray:
    """Select an exact fraction with deterministic tie handling."""
    value = np.asarray(score, dtype=np.float64)
    if value.ndim != 1 or not np.isfinite(value).all():
        raise ValueError("score must be a finite one-dimensional array")
    if not 0.0 < float(budget) < 1.0:
        raise ValueError("budget must be in (0,1)")
    count = max(1, min(len(value), int(round(float(budget) * len(value)))))
    order = np.argsort(value, kind="mergesort")
    selected = np.zeros(len(value), dtype=bool)
    selected[order[-count:]] = True
    return selected


def groupwise_top_fraction(
    score: np.ndarray,
    group: np.ndarray,
    budget: float,
) -> np.ndarray:
    """Apply the same exact budget independently inside every group."""
    value = np.asarray(score, dtype=np.float64)
    group_value = np.asarray(group)
    if value.shape != group_value.shape:
        raise ValueError("score and group must have the same shape")
    selected = np.zeros(len(value), dtype=bool)
    for item in np.unique(group_value):
        index = np.flatnonzero(group_value == item)
        selected[index] = exact_top_fraction(value[index], budget)
    return selected


def source_quantile_threshold(score: np.ndarray, budget: float) -> float:
    """Return the fitted-source score threshold corresponding to a budget."""
    value = np.asarray(score, dtype=np.float64)
    route = exact_top_fraction(value, budget)
    return float(np.min(value[route]))


def routed_probability(
    base_probability: np.ndarray,
    expert_probability: np.ndarray,
    route: np.ndarray,
) -> np.ndarray:
    """Keep the frozen base outside the route and use the expert inside it."""
    base = np.asarray(base_probability, dtype=np.float64)
    expert = np.asarray(expert_probability, dtype=np.float64)
    selected = np.asarray(route, dtype=bool)
    if base.shape != expert.shape or base.shape != selected.shape:
        raise ValueError("base, expert, and route must have the same shape")
    result = base.copy()
    result[selected] = expert[selected]
    return result

