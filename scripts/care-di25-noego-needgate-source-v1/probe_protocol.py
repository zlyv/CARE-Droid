#!/usr/bin/env python3
"""Fixed target and arm definitions for the later authorized formal run."""
from __future__ import annotations

import numpy as np

from source_contract import blocked

ARMS = {
    "S0": {"components": ["stage1_logit"], "dimension": 1},
    "S1": {"components": ["Global24", "stage1_logit"], "dimension": 25},
    "S2": {
        "components": ["role_graph_token32", "critical_host24", "attended_host24", "scalar_summary8"],
        "dimension": 88,
    },
    "S3": {
        "components": ["role_graph_token32", "critical_host24", "attended_host24", "scalar_summary8", "stage1_logit"],
        "dimension": 89,
    },
}


def stage1_logit(probability):
    p = np.asarray(probability, dtype=np.float64)
    if not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise blocked("Stage-I probability is invalid")
    eps = 1e-6
    return np.log((p + eps) / (1.0 - p + eps))


def need_correct(label, probability):
    y = np.asarray(label, dtype=np.int64)
    p = np.asarray(probability, dtype=np.float64)
    if not set(np.unique(y).tolist()).issubset({0, 1}):
        raise blocked("need-correction labels must be binary")
    return ((p >= 0.5) != y.astype(bool)).astype(np.int64)


def validate_noego_88(value, rows: int):
    x = np.asarray(value, dtype=np.float32)
    if x.shape != (int(rows), 88) or not np.isfinite(x).all():
        raise blocked(f"true Host-only Anomaly representation must be finite ({rows},88), received {x.shape}")
    return x
