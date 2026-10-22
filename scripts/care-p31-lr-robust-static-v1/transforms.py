#!/usr/bin/env python3
from __future__ import annotations

import numpy as np


ARMS = (
    "L0_RAW",
    "L1_HELLINGER",
    "L2_LOG",
    "L3_PRESENCE",
    "L4_RAW_PRESENCE",
    "L5_CLR_PRESENCE",
    "L6_HELLINGER_PRESENCE",
)

EPSILON = 1e-4


def validate_p31(values):
    x = np.asarray(values, float)
    if x.ndim != 2 or x.shape[1] != 31:
        raise RuntimeError(f"BLOCKED_BY_CONTRACT: expected N x 31 P31, got {x.shape}")
    if not np.isfinite(x).all() or np.min(x) < -1e-10:
        raise RuntimeError("BLOCKED_BY_CONTRACT: invalid P31 values")
    x = np.clip(x, 0.0, None)
    mass = x.sum(axis=1)
    # A valid APK may contain no API event that maps to any of the 31 roles.
    # Preserve that state as the all-zero vector instead of inventing evidence.
    output = np.zeros_like(x)
    nonzero = mass > 0
    output[nonzero] = x[nonzero] / mass[nonzero, None]
    return output


def transform_p31(values, arm):
    x = validate_p31(values)
    present = (x > 0).astype(float)
    if arm == "L0_RAW":
        out = x
    elif arm == "L1_HELLINGER":
        out = np.sqrt(x)
    elif arm == "L2_LOG":
        out = np.log(x + EPSILON)
    elif arm == "L3_PRESENCE":
        out = present
    elif arm == "L4_RAW_PRESENCE":
        out = np.column_stack([x, present])
    elif arm == "L5_CLR_PRESENCE":
        logx = np.log(x + EPSILON)
        clr = logx - logx.mean(axis=1, keepdims=True)
        out = np.column_stack([clr, present])
    elif arm == "L6_HELLINGER_PRESENCE":
        out = np.column_stack([np.sqrt(x), present])
    else:
        raise KeyError(arm)
    if not np.isfinite(out).all():
        raise RuntimeError(f"BLOCKED_BY_CONTRACT: non-finite transformed P31 arm={arm}")
    return out


def feature_names(arm):
    raw = [f"role_{index:02d}_share" for index in range(31)]
    presence = [f"role_{index:02d}_present" for index in range(31)]
    if arm == "L0_RAW":
        return raw
    if arm == "L1_HELLINGER":
        return [name.replace("_share", "_hellinger") for name in raw]
    if arm == "L2_LOG":
        return [name.replace("_share", "_log_eps1e-4") for name in raw]
    if arm == "L3_PRESENCE":
        return presence
    if arm == "L4_RAW_PRESENCE":
        return raw + presence
    if arm == "L5_CLR_PRESENCE":
        return [name.replace("_share", "_clr_eps1e-4") for name in raw] + presence
    if arm == "L6_HELLINGER_PRESENCE":
        return [name.replace("_share", "_hellinger") for name in raw] + presence
    raise KeyError(arm)
