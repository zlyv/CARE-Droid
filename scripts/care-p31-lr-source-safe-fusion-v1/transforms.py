#!/usr/bin/env python3
from __future__ import annotations

import numpy as np


ARMS = (
    "C0_COUNT_SUMMARY",
    "B1_HELLINGER_PRESENCE",
    "B2_RAW_HELLINGER_PRESENCE",
    "B3_MULTI_BASIS",
    "B4_ADDITIVE_HINGE",
    "B5_PAIRWISE_HELLINGER",
    "B6_RAW_HP_SUMMARY",
)

SELECTABLE_ARMS = tuple(arm for arm in ARMS if arm != "C0_COUNT_SUMMARY")


def normalize_p31(values):
    x = np.asarray(values, float)
    if x.ndim != 2 or x.shape[1] != 31:
        raise RuntimeError(f"BLOCKED_BY_CONTRACT: expected N x 31 P31, got {x.shape}")
    if not np.isfinite(x).all() or np.min(x) < -1e-10:
        raise RuntimeError("BLOCKED_BY_CONTRACT: invalid P31")
    x = np.clip(x, 0.0, None)
    mass = x.sum(axis=1)
    out = np.zeros_like(x)
    valid = mass > 0
    out[valid] = x[valid] / mass[valid, None]
    return out


def summaries(x):
    present = x > 0
    active = present.sum(axis=1) / 31.0
    safe = np.where(present, x, 1.0)
    entropy = -np.sum(np.where(present, x * np.log(safe), 0.0), axis=1) / np.log(31.0)
    ordered = np.sort(x, axis=1)
    top1 = ordered[:, -1]
    top3 = ordered[:, -3:].sum(axis=1)
    hhi = np.square(x).sum(axis=1)
    zero = (x.sum(axis=1) == 0).astype(float)
    return np.column_stack([active, entropy, top1, top3, hhi, zero])


def transform(values, arm):
    x = normalize_p31(values)
    root = np.sqrt(x)
    present = (x > 0).astype(float)
    log1p = np.log1p(31.0 * x) / np.log(32.0)
    if arm == "C0_COUNT_SUMMARY":
        out = summaries(x)
    elif arm == "B1_HELLINGER_PRESENCE":
        out = np.column_stack([root, present])
    elif arm == "B2_RAW_HELLINGER_PRESENCE":
        out = np.column_stack([x, root, present])
    elif arm == "B3_MULTI_BASIS":
        out = np.column_stack([x, root, log1p, present])
    elif arm == "B4_ADDITIVE_HINGE":
        hinges = [np.maximum(x - threshold, 0.0) for threshold in (0.01, 0.03, 0.10)]
        out = np.column_stack([root, present, *hinges])
    elif arm == "B5_PAIRWISE_HELLINGER":
        pairs = [root[:, i] * root[:, j] for i in range(31) for j in range(i + 1, 31)]
        out = np.column_stack([root, present, *pairs])
    elif arm == "B6_RAW_HP_SUMMARY":
        out = np.column_stack([x, root, present, summaries(x)])
    else:
        raise KeyError(arm)
    if not np.isfinite(out).all():
        raise RuntimeError(f"BLOCKED_BY_CONTRACT: non-finite transform arm={arm}")
    return np.asarray(out, float)

