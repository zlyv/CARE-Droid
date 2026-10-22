#!/usr/bin/env python3
"""Artifact-independent scores for reliability-aware directional conflict routing."""

from __future__ import annotations

import numpy as np


AUX_VIEW_NAMES = ("manifest", "code_call", "sensitive", "graph")


def _probability_vector(value: np.ndarray, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.ndim != 1 or not np.isfinite(result).all():
        raise ValueError(name + " must be a finite one-dimensional array")
    return np.clip(result, 0.0, 1.0)


def _auxiliary_matrix(value: np.ndarray) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.ndim != 2 or result.shape[1] != len(AUX_VIEW_NAMES):
        raise ValueError("auxiliary probability must have four semantic views")
    if not np.isfinite(result).all():
        raise ValueError("auxiliary probability contains non-finite values")
    return np.clip(result, 0.0, 1.0)


def estimate_class_conditional_reliability(
    auxiliary_probability: np.ndarray,
    labels: np.ndarray,
    smoothing: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    """Estimate PPV and NPV for each view with symmetric Beta smoothing.

    Positive reliability answers P(y=1 | the view predicts 1), while negative
    reliability answers P(y=0 | the view predicts 0).  These are the two
    quantities needed when a view opposes the Stage-I decision.
    """
    probability = _auxiliary_matrix(auxiliary_probability)
    y = np.asarray(labels, dtype=np.int64)
    if y.shape != (len(probability),) or not np.isin(y, (0, 1)).all():
        raise ValueError("labels must be a binary vector aligned with probability")
    alpha = float(smoothing)
    if not alpha > 0.0:
        raise ValueError("smoothing must be positive")

    positive = np.zeros(probability.shape[1], dtype=np.float64)
    negative = np.zeros(probability.shape[1], dtype=np.float64)
    audit: list[dict] = []
    for column, view in enumerate(AUX_VIEW_NAMES):
        prediction = probability[:, column] >= 0.5
        tp = int(np.sum(prediction & (y == 1)))
        fp = int(np.sum(prediction & (y == 0)))
        tn = int(np.sum(~prediction & (y == 0)))
        fn = int(np.sum(~prediction & (y == 1)))
        positive[column] = (tp + alpha) / (tp + fp + 2.0 * alpha)
        negative[column] = (tn + alpha) / (tn + fn + 2.0 * alpha)
        audit.append({
            "view": view,
            "TP": tp,
            "FP": fp,
            "TN": tn,
            "FN": fn,
            "positive_reliability_ppv": float(positive[column]),
            "negative_reliability_npv": float(negative[column]),
            "smoothing": alpha,
        })
    return positive, negative, audit


def directional_conflict(
    base_probability: np.ndarray,
    auxiliary_probability: np.ndarray,
    positive_reliability: np.ndarray,
    negative_reliability: np.ndarray,
    aggregator: str = "max",
) -> tuple[np.ndarray, np.ndarray]:
    """Return one conflict score and the four view-wise opposing strengths."""
    base = _probability_vector(base_probability, "base_probability")
    auxiliary = _auxiliary_matrix(auxiliary_probability)
    if len(base) != len(auxiliary):
        raise ValueError("base and auxiliary probabilities are not aligned")
    r_positive = _probability_vector(positive_reliability, "positive_reliability")
    r_negative = _probability_vector(negative_reliability, "negative_reliability")
    if r_positive.shape != (auxiliary.shape[1],) or r_negative.shape != (
        auxiliary.shape[1],
    ):
        raise ValueError("one positive and negative reliability is required per view")

    stage1_malware = base >= 0.5
    opposing_malware = np.maximum(0.0, 2.0 * auxiliary - 1.0)
    opposing_benign = np.maximum(0.0, 1.0 - 2.0 * auxiliary)
    conflict = np.where(
        stage1_malware[:, None],
        opposing_benign * r_negative[None, :],
        opposing_malware * r_positive[None, :],
    )
    if aggregator == "max":
        score = np.max(conflict, axis=1)
    elif aggregator == "top2_mean":
        score = np.mean(np.sort(conflict, axis=1)[:, -2:], axis=1)
    else:
        raise ValueError("unknown conflict aggregator: " + aggregator)
    return np.clip(score, 0.0, 1.0), conflict


def boundary_uncertainty(base_probability: np.ndarray) -> np.ndarray:
    """Distance-to-boundary uncertainty, with one at p=.5 and zero at 0/1."""
    base = _probability_vector(base_probability, "base_probability")
    return np.clip(1.0 - 2.0 * np.abs(base - 0.5), 0.0, 1.0)


def soft_or(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    """Bounded monotone OR-style fusion; this is not a calibrated probability."""
    left = _probability_vector(first, "first")
    right = _probability_vector(second, "second")
    if left.shape != right.shape:
        raise ValueError("soft-OR inputs must be aligned")
    return np.clip(1.0 - (1.0 - left) * (1.0 - right), 0.0, 1.0)


def build_route_scores(
    base_probability: np.ndarray,
    auxiliary_probability: np.ndarray,
    positive_reliability: np.ndarray,
    negative_reliability: np.ndarray,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Build primary and ablation scores from the same frozen inputs."""
    boundary = boundary_uncertainty(base_probability)
    conflict_max, per_view = directional_conflict(
        base_probability,
        auxiliary_probability,
        positive_reliability,
        negative_reliability,
        "max",
    )
    conflict_top2, _ = directional_conflict(
        base_probability,
        auxiliary_probability,
        positive_reliability,
        negative_reliability,
        "top2_mean",
    )
    ones = np.ones(len(AUX_VIEW_NAMES), dtype=np.float64)
    conflict_unweighted, _ = directional_conflict(
        base_probability, auxiliary_probability, ones, ones, "max"
    )
    return {
        "Boundary": boundary,
        "ConflictMax": conflict_max,
        "SoftOR": soft_or(boundary, conflict_max),
        "SoftOR_NoReliability": soft_or(boundary, conflict_unweighted),
        "SoftOR_Top2Mean": soft_or(boundary, conflict_top2),
    }, per_view


def exact_top_fraction(score: np.ndarray, budget: float) -> np.ndarray:
    value = _probability_vector(score, "score")
    if not 0.0 < float(budget) < 1.0:
        raise ValueError("budget must be in (0,1)")
    count = max(1, min(len(value), int(round(float(budget) * len(value)))))
    order = np.argsort(value, kind="mergesort")
    route = np.zeros(len(value), dtype=bool)
    route[order[-count:]] = True
    return route


def groupwise_top_fraction(
    score: np.ndarray, group: np.ndarray, budget: float
) -> np.ndarray:
    value = _probability_vector(score, "score")
    group_value = np.asarray(group)
    if group_value.shape != value.shape:
        raise ValueError("group must be aligned with score")
    route = np.zeros(len(value), dtype=bool)
    for item in np.unique(group_value):
        index = np.flatnonzero(group_value == item)
        route[index] = exact_top_fraction(value[index], budget)
    return route


def source_quantile_threshold(score: np.ndarray, budget: float) -> float:
    value = _probability_vector(score, "score")
    route = exact_top_fraction(value, budget)
    return float(np.min(value[route]))


def routing_diagnostics(
    method: str,
    labels: np.ndarray,
    base_probability: np.ndarray,
    route: np.ndarray,
    score: np.ndarray,
    high_confidence_threshold: float = 0.8,
) -> dict:
    y = np.asarray(labels, dtype=np.int64)
    base = _probability_vector(base_probability, "base_probability")
    selected = np.asarray(route, dtype=bool)
    risk = np.asarray(score, dtype=np.float64)
    if y.shape != base.shape or selected.shape != base.shape or risk.shape != base.shape:
        raise ValueError("routing diagnostics inputs are not aligned")
    prediction = base >= 0.5
    error = prediction != y
    false_positive = prediction & (y == 0)
    false_negative = ~prediction & (y == 1)
    confidence = np.maximum(base, 1.0 - base)
    high_error = error & (confidence >= float(high_confidence_threshold))
    accepted = ~selected
    captured = int(np.sum(selected & error))
    high_captured = int(np.sum(selected & high_error))
    return {
        "method": method,
        "n": int(len(y)),
        "routed_n": int(np.sum(selected)),
        "routed_ratio": float(np.mean(selected)),
        "base_error_n": int(np.sum(error)),
        "routed_base_error_n": captured,
        "error_recall": float(captured / np.sum(error)) if np.any(error) else 0.0,
        "route_error_precision": float(captured / np.sum(selected)) if np.any(selected) else 0.0,
        "accepted_n": int(np.sum(accepted)),
        "accepted_error_n": int(np.sum(accepted & error)),
        "accepted_risk": float(np.mean(error[accepted])) if np.any(accepted) else 0.0,
        "high_confidence_threshold": float(high_confidence_threshold),
        "high_confidence_error_n": int(np.sum(high_error)),
        "routed_high_confidence_error_n": high_captured,
        "high_confidence_error_recall": (
            float(high_captured / np.sum(high_error)) if np.any(high_error) else 0.0
        ),
        "routed_FP_n": int(np.sum(selected & false_positive)),
        "routed_FN_n": int(np.sum(selected & false_negative)),
        "score_mean": float(np.mean(risk)),
        "score_max": float(np.max(risk)),
    }
