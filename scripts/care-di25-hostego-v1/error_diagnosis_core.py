#!/usr/bin/env python3
"""Shared, side-effect-free helpers for the frozen same-year error diagnosis."""

from __future__ import annotations

import math
from typing import Iterable

import numpy as np


DISCLAIMER = (
    "These are independent same-year OOF diagnostics and must not be described "
    "as temporal-generalization, future-year, zero-day, or unseen-future performance."
)
MAX_DELTA = 2.25
YEARS = (2018, 2019, 2020, 2021)


def clip_probability(value):
    return np.clip(np.asarray(value, dtype=float), 1e-9, 1.0 - 1e-9)


def logit(value):
    p = clip_probability(value)
    return np.log(p / (1.0 - p))


def cohort(stage1_correct: bool, routed: bool, final_correct: bool) -> str:
    if not routed:
        return "A" if stage1_correct else "B"
    if not stage1_correct:
        return "C" if final_correct else "D"
    return "E" if final_correct else "F"


def exact_mcnemar(n01: int, n10: int) -> float:
    n = int(n01) + int(n10)
    if n == 0:
        return 1.0
    k = min(int(n01), int(n10))
    return min(1.0, 2.0 * sum(math.comb(n, i) for i in range(k + 1)) / (2**n))


def safe_div(numerator: float, denominator: float) -> float:
    return float(numerator) / float(denominator) if denominator else float("nan")


def iqr(values: Iterable[float]) -> float:
    a = np.asarray(list(values), dtype=float)
    a = a[np.isfinite(a)]
    return (
        float(np.percentile(a, 75) - np.percentile(a, 25)) if len(a) else float("nan")
    )


def cohens_d(left: Iterable[float], right: Iterable[float]) -> float:
    a = np.asarray(list(left), dtype=float)
    b = np.asarray(list(right), dtype=float)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    pooled = math.sqrt(
        ((len(a) - 1) * np.var(a, ddof=1) + (len(b) - 1) * np.var(b, ddof=1))
        / (len(a) + len(b) - 2)
    )
    return float((np.mean(a) - np.mean(b)) / pooled) if pooled > 0 else 0.0


def residual_primary_failure(
    label: int, effective_delta: float, required_abs: float
) -> str:
    if required_abs > MAX_DELTA:
        return "D3_required_correction_exceeds_bound"
    aligned = effective_delta if int(label) == 1 else -effective_delta
    if aligned <= 0:
        return "D1_correction_direction_wrong"
    if abs(effective_delta) < 0.05:
        return "D4_almost_no_correction"
    return "D2_direction_correct_magnitude_insufficient"
