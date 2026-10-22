#!/usr/bin/env python3
from __future__ import annotations

import math

MAX_DELTA = 2.25
EPSILON = 1e-12
NEAR_ZERO_LOGIT = 0.05


def clip_probability(value: float) -> float:
    return min(max(float(value), 1e-8), 1.0 - 1e-8)


def logit(value: float) -> float:
    probability = clip_probability(value)
    return math.log(probability / (1.0 - probability))


def bounded_failure(label: int, base_probability: float,
                    bounded_expert_probability: float) -> dict:
    base_prediction = int(base_probability >= .5)
    if base_prediction == int(label):
        raise ValueError("bounded_failure requires a Stage-I error")
    base_logit = logit(base_probability)
    expert_logit = logit(bounded_expert_probability)
    delta = expert_logit - base_logit
    required_sign = 1 if int(label) == 1 else -1
    signed_delta = required_sign * delta
    required_magnitude = abs(base_logit) + 1e-12
    fixed = int((bounded_expert_probability >= .5) == bool(label))
    if fixed:
        primary = "FIXED_BOUNDED"
    elif required_magnitude > MAX_DELTA + 1e-9:
        primary = "BOUND_UNREACHABLE"
    elif signed_delta <= EPSILON:
        primary = "WRONG_OR_ZERO_DIRECTION"
    else:
        primary = "RIGHT_DIRECTION_TOO_WEAK"
    return {
        "base_logit": base_logit,
        "bounded_expert_logit": expert_logit,
        "effective_bounded_delta": delta,
        "required_direction": "POS" if required_sign > 0 else "NEG",
        "signed_effective_delta": signed_delta,
        "required_magnitude_to_cross": required_magnitude,
        "max_delta": MAX_DELTA,
        "theoretically_reachable_under_bound": int(required_magnitude <= MAX_DELTA + 1e-9),
        "near_zero_abs_delta_le_0p05": int(abs(delta) <= NEAR_ZERO_LOGIT),
        "bounded_fixed": fixed,
        "primary_failure_type": primary,
    }
